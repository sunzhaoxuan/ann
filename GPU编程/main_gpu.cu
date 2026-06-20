#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <set>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__       \
                      << " : " << cudaGetErrorString(err) << std::endl;        \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

template <typename T>
T* LoadData(const std::string& data_path, size_t& n, size_t& d) {
    std::ifstream fin(data_path, std::ios::in | std::ios::binary);
    if (!fin) {
        std::cerr << "cannot open file: " << data_path << std::endl;
        std::exit(EXIT_FAILURE);
    }

    fin.read(reinterpret_cast<char*>(&n), 4);
    fin.read(reinterpret_cast<char*>(&d), 4);

    T* data = new T[n * d];
    const size_t sz = sizeof(T);

    for (size_t i = 0; i < n; ++i) {
        fin.read(reinterpret_cast<char*>(data + i * d), d * sz);
    }

    fin.close();

    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d
              << "  number: " << n
              << "  size_per_element: " << sizeof(T) << "\n";

    return data;
}

// ============================================================
// GPU 矩阵乘法 baseline kernel
//
// base  : n × d，行主序
// query : m × d，行主序
// score : n × m，行主序
//
// score[i][j] = inner_product(base[i], query[j])
// ============================================================
__global__ void gpu_ip_matmul_kernel(
    const float* base,
    const float* query,
    float* score,
    int n,
    int d,
    int m
) {
    int qid = blockIdx.x * blockDim.x + threadIdx.x;   // query index in batch
    int bid = blockIdx.y * blockDim.y + threadIdx.y;   // base index

    if (bid >= n || qid >= m) {
        return;
    }

    float sum = 0.0f;

    for (int t = 0; t < d; ++t) {
        sum += base[bid * d + t] * query[qid * d + t];
    }

    score[bid * m + qid] = sum;
}

// 根据一列 score 取 Top-k。
// 注意：原 CPU flat_search 中使用 dis = 1 - inner_product。
// 因此这里也将 score 转成 dis，再维护距离最小的 k 个点。
std::priority_queue<std::pair<float, uint32_t>>
topk_from_score_column(
    const std::vector<float>& score,
    size_t base_number,
    size_t batch_size,
    size_t local_qid,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (size_t i = 0; i < base_number; ++i) {
        float inner_product = score[i * batch_size + local_qid];
        float dis = 1.0f - inner_product;

        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else if (dis < q.top().first) {
            q.push({dis, static_cast<uint32_t>(i)});
            q.pop();
        }
    }

    return q;
}

float compute_recall_at_k(
    std::priority_queue<std::pair<float, uint32_t>> res,
    const int* test_gt,
    size_t test_gt_d,
    size_t query_id,
    size_t k
) {
    std::set<uint32_t> gtset;

    for (size_t j = 0; j < k; ++j) {
        int t = test_gt[query_id * test_gt_d + j];
        gtset.insert(static_cast<uint32_t>(t));
    }

    size_t acc = 0;

    while (!res.empty()) {
        uint32_t x = res.top().second;

        if (gtset.find(x) != gtset.end()) {
            ++acc;
        }

        res.pop();
    }

    return static_cast<float>(acc) / static_cast<float>(k);
}

int main() {
    // 本机建议使用相对路径，把数据集放在 ./data/ 下
    std::string data_path = "./data/";

    size_t test_number = 0;
    size_t base_number = 0;
    size_t test_gt_d = 0;
    size_t vecdim = 0;

    float* test_query = LoadData<float>(
        data_path + "DEEP100K.query.fbin",
        test_number,
        vecdim
    );

    int* test_gt = LoadData<int>(
        data_path + "DEEP100K.gt.query.100k.top100.bin",
        test_number,
        test_gt_d
    );

    float* base = LoadData<float>(
        data_path + "DEEP100K.base.100k.fbin",
        base_number,
        vecdim
    );

    // 可根据机器性能调整测试 query 数量
    test_number = std::min<size_t>(test_number, 2000);

    const size_t k = 10;

    // GPU batch size
    const size_t batch_size = 8;

    std::cout << "base_number = " << base_number << "\n";
    std::cout << "test_number = " << test_number << "\n";
    std::cout << "vecdim      = " << vecdim << "\n";
    std::cout << "batch_size  = " << batch_size << "\n";

    // ============================================================
    // 1. GPU 显存分配
    // base 只拷贝一次，模拟实际 ANN index 常驻 GPU 显存
    // ============================================================
    float* d_base = nullptr;
    float* d_query = nullptr;
    float* d_score = nullptr;

    const size_t base_bytes = base_number * vecdim * sizeof(float);
    const size_t max_query_bytes = batch_size * vecdim * sizeof(float);
    const size_t max_score_bytes = base_number * batch_size * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_base, base_bytes));
    CUDA_CHECK(cudaMalloc(&d_query, max_query_bytes));
    CUDA_CHECK(cudaMalloc(&d_score, max_score_bytes));

    auto base_copy_begin = std::chrono::high_resolution_clock::now();
    CUDA_CHECK(cudaMemcpy(d_base, base, base_bytes, cudaMemcpyHostToDevice));
    auto base_copy_end = std::chrono::high_resolution_clock::now();

    double base_copy_ms =
        std::chrono::duration<double, std::milli>(
            base_copy_end - base_copy_begin
        ).count();

    // ============================================================
    // 2. batch 查询
    // ============================================================
    double total_recall = 0.0;

    double total_h2d_ms = 0.0;
    double total_kernel_ms = 0.0;
    double total_d2h_ms = 0.0;
    double total_topk_ms = 0.0;

    auto total_begin = std::chrono::high_resolution_clock::now();

    for (size_t batch_begin = 0;
         batch_begin < test_number;
         batch_begin += batch_size) {

        size_t cur_batch = std::min(batch_size, test_number - batch_begin);

        const float* query_ptr = test_query + batch_begin * vecdim;

        size_t query_bytes = cur_batch * vecdim * sizeof(float);
        size_t score_bytes = base_number * cur_batch * sizeof(float);

        // ----------------------------
        // Host to Device: query batch
        // ----------------------------
        auto h2d_begin = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMemcpy(
            d_query,
            query_ptr,
            query_bytes,
            cudaMemcpyHostToDevice
        ));

        auto h2d_end = std::chrono::high_resolution_clock::now();

        total_h2d_ms +=
            std::chrono::duration<double, std::milli>(
                h2d_end - h2d_begin
            ).count();

        // ----------------------------
        // GPU kernel: base × query^T
        // ----------------------------
        dim3 block(16, 16);
        dim3 grid(
            static_cast<unsigned int>((cur_batch + block.x - 1) / block.x),
            static_cast<unsigned int>((base_number + block.y - 1) / block.y)
        );

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        gpu_ip_matmul_kernel<<<grid, block>>>(
            d_base,
            d_query,
            d_score,
            static_cast<int>(base_number),
            static_cast<int>(vecdim),
            static_cast<int>(cur_batch)
        );

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        CUDA_CHECK(cudaGetLastError());

        float kernel_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

        total_kernel_ms += static_cast<double>(kernel_ms);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // ----------------------------
        // Device to Host: score matrix
        // baseline 版本直接把完整 score 拷回 CPU
        // 后续优化可以改成 GPU Top-k，只拷回候选
        // ----------------------------
        std::vector<float> score(base_number * cur_batch);

        auto d2h_begin = std::chrono::high_resolution_clock::now();

        CUDA_CHECK(cudaMemcpy(
            score.data(),
            d_score,
            score_bytes,
            cudaMemcpyDeviceToHost
        ));

        auto d2h_end = std::chrono::high_resolution_clock::now();

        total_d2h_ms +=
            std::chrono::duration<double, std::milli>(
                d2h_end - d2h_begin
            ).count();

        // ----------------------------
        // CPU Top-k + recall
        // ----------------------------
        auto topk_begin = std::chrono::high_resolution_clock::now();

        for (size_t local_qid = 0; local_qid < cur_batch; ++local_qid) {
            size_t global_qid = batch_begin + local_qid;

            auto res = topk_from_score_column(
                score,
                base_number,
                cur_batch,
                local_qid,
                k
            );

            float recall = compute_recall_at_k(
                res,
                test_gt,
                test_gt_d,
                global_qid,
                k
            );

            total_recall += static_cast<double>(recall);
        }

        auto topk_end = std::chrono::high_resolution_clock::now();

        total_topk_ms +=
            std::chrono::duration<double, std::milli>(
                topk_end - topk_begin
            ).count();
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto total_end = std::chrono::high_resolution_clock::now();

    double total_query_ms =
        std::chrono::duration<double, std::milli>(
            total_end - total_begin
        ).count();

    double avg_recall =
        total_recall / static_cast<double>(test_number);

    double avg_latency_us =
        total_query_ms * 1000.0 / static_cast<double>(test_number);

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\n========== GPU Flat-GEMM Baseline ==========\n";
    std::cout << "average recall: "
              << avg_recall
              << "\n";

    std::cout << "average latency (us): "
              << avg_latency_us
              << "\n";

    std::cout << "\n========== Time Breakdown ==========\n";
    std::cout << "base H2D copy once (ms): "
              << base_copy_ms
              << "\n";

    std::cout << "query H2D total (ms): "
              << total_h2d_ms
              << "\n";

    std::cout << "GEMM kernel total (ms): "
              << total_kernel_ms
              << "\n";

    std::cout << "score D2H total (ms): "
              << total_d2h_ms
              << "\n";

    std::cout << "CPU Top-k total (ms): "
              << total_topk_ms
              << "\n";

    std::cout << "total query time (ms): "
              << total_query_ms
              << "\n";

    std::cout << "\n========== Average Per Query ==========\n";
    std::cout << "query H2D avg (us): "
              << total_h2d_ms * 1000.0 / static_cast<double>(test_number)
              << "\n";

    std::cout << "GEMM kernel avg (us): "
              << total_kernel_ms * 1000.0 / static_cast<double>(test_number)
              << "\n";

    std::cout << "score D2H avg (us): "
              << total_d2h_ms * 1000.0 / static_cast<double>(test_number)
              << "\n";

    std::cout << "CPU Top-k avg (us): "
              << total_topk_ms * 1000.0 / static_cast<double>(test_number)
              << "\n";

    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_score));

    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}