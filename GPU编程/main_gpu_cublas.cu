#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
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

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = (call);                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__     \
                      << " : status = " << status << std::endl;                \
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

    uint32_t n32 = 0;
    uint32_t d32 = 0;

    fin.read(reinterpret_cast<char*>(&n32), 4);
    fin.read(reinterpret_cast<char*>(&d32), 4);

    n = static_cast<size_t>(n32);
    d = static_cast<size_t>(d32);

    T* data = new T[n * d];
    fin.read(reinterpret_cast<char*>(data), n * d * sizeof(T));
    fin.close();

    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d
              << "  number: " << n
              << "  size_per_element: " << sizeof(T) << "\n";

    return data;
}

// 根据 score 矩阵的一列取 Top-k。
// score 在内存中按 row-major 的 n × batch_size 存放：
// score[i * batch_size + qid] 表示 base[i] 与 query[qid] 的 inner product。
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

        // 与 flat_scan.h 保持一致：
        // DEEP100K 使用 IP 距离，距离定义为 dis = 1 - inner_product。
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

int main(int argc, char** argv) {
    std::string data_path = argc > 1 ? argv[1] : "./data/";
    if (!data_path.empty() && data_path.back() != '/' && data_path.back() != '\\') {
        data_path.push_back('/');
    }

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

    test_number = std::min<size_t>(test_number, 2000);

    const size_t k = 10;

    // 可以改成 8, 16, 32, 64, 128, 256 做 batch size 对比实验
    const size_t batch_size = 16;

    std::cout << "base_number = " << base_number << "\n";
    std::cout << "test_number = " << test_number << "\n";
    std::cout << "vecdim      = " << vecdim << "\n";
    std::cout << "batch_size  = " << batch_size << "\n";

    // ============================================================
    // cuBLAS 初始化
    // ============================================================
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // 默认 FP32 模式即可。若想尝试 TF32，可根据 CUDA 版本设置 math mode。
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    // ============================================================
    // GPU 显存分配
    // base 常驻 GPU，只拷贝一次
    // ============================================================
    float* d_base = nullptr;
    float* d_query = nullptr;
    float* d_score = nullptr;

    const size_t base_bytes =
        base_number * vecdim * sizeof(float);

    const size_t max_query_bytes =
        batch_size * vecdim * sizeof(float);

    const size_t max_score_bytes =
        base_number * batch_size * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_base, base_bytes));
    CUDA_CHECK(cudaMalloc(&d_query, max_query_bytes));
    CUDA_CHECK(cudaMalloc(&d_score, max_score_bytes));

    auto base_copy_begin = std::chrono::high_resolution_clock::now();

    CUDA_CHECK(cudaMemcpy(
        d_base,
        base,
        base_bytes,
        cudaMemcpyHostToDevice
    ));

    auto base_copy_end = std::chrono::high_resolution_clock::now();

    double base_copy_ms =
        std::chrono::duration<double, std::milli>(
            base_copy_end - base_copy_begin
        ).count();

    // ============================================================
    // batch 查询
    // ============================================================
    double total_recall = 0.0;

    double total_h2d_ms = 0.0;
    double total_gemm_ms = 0.0;
    double total_d2h_ms = 0.0;
    double total_topk_ms = 0.0;

    auto total_begin = std::chrono::high_resolution_clock::now();

    for (size_t batch_begin = 0;
         batch_begin < test_number;
         batch_begin += batch_size) {

        size_t cur_batch =
            std::min(batch_size, test_number - batch_begin);

        const float* query_ptr =
            test_query + batch_begin * vecdim;

        size_t query_bytes =
            cur_batch * vecdim * sizeof(float);

        size_t score_bytes =
            base_number * cur_batch * sizeof(float);

        // --------------------------------------------------------
        // 1. query batch 拷贝到 GPU
        // --------------------------------------------------------
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

        // --------------------------------------------------------
        // 2. cuBLAS 矩阵乘法
        //
        // 目标：
        //   base:  n × d，row-major
        //   query: m × d，row-major
        //   score: n × m，row-major
        //
        //   score = base × query^T
        //
        // cuBLAS 使用 column-major。
        // row-major 的 score(n × m) 在内存上等价于
        // column-major 的 score^T(m × n)。
        //
        // 因此用 cuBLAS 计算：
        //   score^T = query × base^T
        //
        // d_query 按 column-major 看是 d × m，即 query^T；
        // 对它使用 CUBLAS_OP_T 得到 query(m × d)。
        //
        // d_base 按 column-major 看是 d × n，即 base^T；
        // 对它使用 CUBLAS_OP_N 得到 base^T(d × n)。
        //
        // 最终 C 是 m × n 的 column-major，
        // 内存刚好等价于 n × m 的 row-major score。
        // --------------------------------------------------------
        const float alpha = 1.0f;
        const float beta = 0.0f;

        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        CUDA_CHECK(cudaEventRecord(start));

        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_T,
            CUBLAS_OP_N,
            static_cast<int>(cur_batch),      // m: C 的行数，即 query 数
            static_cast<int>(base_number),    // n: C 的列数，即 base 数
            static_cast<int>(vecdim),         // k: 维度 d
            &alpha,
            d_query,
            static_cast<int>(vecdim),         // A 原始为 d × cur_batch
            d_base,
            static_cast<int>(vecdim),         // B 原始为 d × base_number
            &beta,
            d_score,
            static_cast<int>(cur_batch)       // C leading dimension = cur_batch
        ));

        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float gemm_ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gemm_ms, start, stop));

        total_gemm_ms += static_cast<double>(gemm_ms);

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));

        // --------------------------------------------------------
        // 3. score 矩阵拷回 CPU
        // baseline 版本仍然拷回完整 score
        // 后续优化可以改成 GPU Top-k / Top-p
        // --------------------------------------------------------
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

        // --------------------------------------------------------
        // 4. CPU Top-k + recall
        // --------------------------------------------------------
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

    std::cout << "\n========== GPU Flat-cuBLAS Baseline ==========\n";
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

    std::cout << "cuBLAS GEMM total (ms): "
              << total_gemm_ms
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

    std::cout << "cuBLAS GEMM avg (us): "
              << total_gemm_ms * 1000.0 / static_cast<double>(test_number)
              << "\n";

    std::cout << "score D2H avg (us): "
              << total_d2h_ms * 1000.0 / static_cast<double>(test_number)
              << "\n";

    std::cout << "CPU Top-k avg (us): "
              << total_topk_ms * 1000.0 / static_cast<double>(test_number)
              << "\n";

    CUBLAS_CHECK(cublasDestroy(handle));

    CUDA_CHECK(cudaFree(d_base));
    CUDA_CHECK(cudaFree(d_query));
    CUDA_CHECK(cudaFree(d_score));

    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}
