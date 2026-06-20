#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <queue>
#include <set>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__        \
                      << " : " << cudaGetErrorString(err) << std::endl;         \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t status = (call);                                         \
        if (status != CUBLAS_STATUS_SUCCESS) {                                  \
            std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__      \
                      << " : status = " << status << std::endl;                 \
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

static inline float dot_product(
    const float* a,
    const float* b,
    size_t d
) {
    float sum = 0.0f;

    for (size_t i = 0; i < d; ++i) {
        sum += a[i] * b[i];
    }

    return sum;
}

void normalize_vector(
    float* x,
    size_t d
) {
    float norm = 0.0f;

    for (size_t i = 0; i < d; ++i) {
        norm += x[i] * x[i];
    }

    norm = std::sqrt(norm);

    if (norm < 1e-12f) {
        return;
    }

    for (size_t i = 0; i < d; ++i) {
        x[i] /= norm;
    }
}

uint32_t find_best_centroid_ip(
    const float* vec,
    const std::vector<float>& centroids,
    size_t nlist,
    size_t d
) {
    uint32_t best_id = 0;
    float best_score = -std::numeric_limits<float>::max();

    for (size_t c = 0; c < nlist; ++c) {
        const float* centroid = centroids.data() + c * d;
        float score = dot_product(vec, centroid, d);

        if (score > best_score) {
            best_score = score;
            best_id = static_cast<uint32_t>(c);
        }
    }

    return best_id;
}

std::vector<uint32_t> select_top_nprobe(
    const float* query,
    const std::vector<float>& centroids,
    size_t nlist,
    size_t d,
    size_t nprobe
) {
    using Pair = std::pair<float, uint32_t>;

    std::priority_queue<
        Pair,
        std::vector<Pair>,
        std::greater<Pair>
    > heap;

    for (size_t c = 0; c < nlist; ++c) {
        const float* centroid = centroids.data() + c * d;
        float score = dot_product(query, centroid, d);

        if (heap.size() < nprobe) {
            heap.push({score, static_cast<uint32_t>(c)});
        } else if (score > heap.top().first) {
            heap.pop();
            heap.push({score, static_cast<uint32_t>(c)});
        }
    }

    std::vector<uint32_t> result;

    while (!heap.empty()) {
        result.push_back(heap.top().second);
        heap.pop();
    }

    std::reverse(result.begin(), result.end());

    return result;
}

std::vector<float> train_kmeans_ip(
    const float* base,
    size_t base_number,
    size_t d,
    size_t nlist,
    size_t train_size,
    size_t iters
) {
    train_size = std::min(train_size, base_number);

    std::vector<float> centroids(nlist * d, 0.0f);

    // 初始化：从训练集中均匀选取 nlist 个向量作为初始中心
    for (size_t c = 0; c < nlist; ++c) {
        size_t idx = c * train_size / nlist;

        for (size_t j = 0; j < d; ++j) {
            centroids[c * d + j] = base[idx * d + j];
        }

        normalize_vector(centroids.data() + c * d, d);
    }

    std::vector<float> sums(nlist * d, 0.0f);
    std::vector<size_t> counts(nlist, 0);

    for (size_t iter = 0; iter < iters; ++iter) {
        std::fill(sums.begin(), sums.end(), 0.0f);
        std::fill(counts.begin(), counts.end(), 0);

        for (size_t i = 0; i < train_size; ++i) {
            const float* vec = base + i * d;

            uint32_t cid = find_best_centroid_ip(
                vec,
                centroids,
                nlist,
                d
            );

            counts[cid]++;

            for (size_t j = 0; j < d; ++j) {
                sums[cid * d + j] += vec[j];
            }
        }

        for (size_t c = 0; c < nlist; ++c) {
            if (counts[c] == 0) {
                continue;
            }

            for (size_t j = 0; j < d; ++j) {
                centroids[c * d + j] =
                    sums[c * d + j] / static_cast<float>(counts[c]);
            }

            normalize_vector(centroids.data() + c * d, d);
        }

        std::cout << "kmeans iter " << iter + 1
                  << " / " << iters << " finished\n";
    }

    return centroids;
}

struct IVFListCPU {
    std::vector<uint32_t> ids;
    std::vector<float> vecs;
};

struct IVFListGPU {
    float* d_vecs = nullptr;
    size_t list_size = 0;
    std::vector<uint32_t> ids;
};

struct IVFGPUIndex {
    size_t nlist = 0;
    size_t d = 0;
    size_t max_list_size = 0;
    std::vector<float> centroids;
    std::vector<IVFListGPU> lists;
};

std::vector<IVFListCPU> build_ivf_lists_cpu(
    const float* base,
    size_t base_number,
    size_t d,
    const std::vector<float>& centroids,
    size_t nlist
) {
    std::vector<IVFListCPU> lists(nlist);

    for (size_t i = 0; i < base_number; ++i) {
        const float* vec = base + i * d;

        uint32_t cid = find_best_centroid_ip(
            vec,
            centroids,
            nlist,
            d
        );

        lists[cid].ids.push_back(static_cast<uint32_t>(i));

        size_t old_size = lists[cid].vecs.size();
        lists[cid].vecs.resize(old_size + d);

        for (size_t j = 0; j < d; ++j) {
            lists[cid].vecs[old_size + j] = vec[j];
        }
    }

    return lists;
}

IVFGPUIndex build_ivf_gpu_index(
    const float* base,
    size_t base_number,
    size_t d,
    size_t nlist,
    size_t train_size,
    size_t kmeans_iters
) {
    IVFGPUIndex index;

    index.nlist = nlist;
    index.d = d;
    index.max_list_size = 0;

    std::cout << "training kmeans...\n";

    index.centroids = train_kmeans_ip(
        base,
        base_number,
        d,
        nlist,
        train_size,
        kmeans_iters
    );

    std::cout << "building ivf lists...\n";

    std::vector<IVFListCPU> cpu_lists = build_ivf_lists_cpu(
        base,
        base_number,
        d,
        index.centroids,
        nlist
    );

    index.lists.resize(nlist);

    std::cout << "copying ivf lists to GPU...\n";

    for (size_t c = 0; c < nlist; ++c) {
        size_t list_size = cpu_lists[c].ids.size();

        index.lists[c].list_size = list_size;
        index.lists[c].ids = std::move(cpu_lists[c].ids);

        index.max_list_size = std::max(index.max_list_size, list_size);

        if (list_size == 0) {
            index.lists[c].d_vecs = nullptr;
            continue;
        }

        size_t bytes = list_size * d * sizeof(float);

        CUDA_CHECK(cudaMalloc(&index.lists[c].d_vecs, bytes));

        CUDA_CHECK(cudaMemcpy(
            index.lists[c].d_vecs,
            cpu_lists[c].vecs.data(),
            bytes,
            cudaMemcpyHostToDevice
        ));
    }

    std::cout << "max list size = "
              << index.max_list_size << "\n";

    return index;
}

void free_ivf_gpu_index(
    IVFGPUIndex& index
) {
    for (size_t c = 0; c < index.nlist; ++c) {
        if (index.lists[c].d_vecs != nullptr) {
            CUDA_CHECK(cudaFree(index.lists[c].d_vecs));
            index.lists[c].d_vecs = nullptr;
        }
    }
}

void push_topk(
    std::priority_queue<std::pair<float, uint32_t>>& q,
    float dis,
    uint32_t id,
    size_t k
) {
    if (q.size() < k) {
        q.push({dis, id});
    } else if (dis < q.top().first) {
        q.push({dis, id});
        q.pop();
    }
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

    test_number = std::min<size_t>(test_number, 2000);

    const size_t k = 10;

    // IVF 参数
    const size_t nlist = 100;
    const size_t nprobe = 8;
    const size_t train_size = 10000;
    const size_t kmeans_iters = 6;

    // batch 参数
    const size_t batch_size = 64;

    std::cout << "base_number = " << base_number << "\n";
    std::cout << "test_number = " << test_number << "\n";
    std::cout << "vecdim      = " << vecdim << "\n";
    std::cout << "nlist       = " << nlist << "\n";
    std::cout << "nprobe      = " << nprobe << "\n";
    std::cout << "batch_size  = " << batch_size << "\n";

    // ============================================================
    // 构建 IVF-GPU 索引
    // ============================================================
    auto index_begin = std::chrono::high_resolution_clock::now();

    IVFGPUIndex index = build_ivf_gpu_index(
        base,
        base_number,
        vecdim,
        nlist,
        train_size,
        kmeans_iters
    );

    auto index_end = std::chrono::high_resolution_clock::now();

    double index_build_ms =
        std::chrono::duration<double, std::milli>(
            index_end - index_begin
        ).count();

    // ============================================================
    // cuBLAS 初始化
    // ============================================================
    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));
    CUBLAS_CHECK(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));

    // 临时 GPU buffer
    float* d_group_query = nullptr;
    float* d_score = nullptr;

    size_t max_query_bytes =
        batch_size * vecdim * sizeof(float);

    size_t max_score_bytes =
        index.max_list_size * batch_size * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_group_query, max_query_bytes));
    CUDA_CHECK(cudaMalloc(&d_score, max_score_bytes));

    cudaEvent_t gemm_start;
    cudaEvent_t gemm_stop;

    CUDA_CHECK(cudaEventCreate(&gemm_start));
    CUDA_CHECK(cudaEventCreate(&gemm_stop));

    // ============================================================
    // 查询
    // ============================================================
    double total_recall = 0.0;

    double total_centroid_ms = 0.0;
    double total_query_h2d_ms = 0.0;
    double total_gemm_ms = 0.0;
    double total_score_d2h_ms = 0.0;
    double total_merge_ms = 0.0;

    auto total_begin = std::chrono::high_resolution_clock::now();

    for (size_t batch_begin = 0;
         batch_begin < test_number;
         batch_begin += batch_size) {

        size_t cur_batch =
            std::min(batch_size, test_number - batch_begin);

        // 每个 query 维护一个 Top-k
        std::vector<
            std::priority_queue<std::pair<float, uint32_t>>
        > batch_topk(cur_batch);

        // --------------------------------------------------------
        // 1. 选择每个 query 需要访问的 Top-nprobe 个簇
        // --------------------------------------------------------
        auto centroid_begin = std::chrono::high_resolution_clock::now();

        std::vector<std::vector<uint32_t>> query_probes(cur_batch);

        for (size_t local_qid = 0; local_qid < cur_batch; ++local_qid) {
            size_t global_qid = batch_begin + local_qid;

            const float* query =
                test_query + global_qid * vecdim;

            query_probes[local_qid] = select_top_nprobe(
                query,
                index.centroids,
                nlist,
                vecdim,
                nprobe
            );
        }

        auto centroid_end = std::chrono::high_resolution_clock::now();

        total_centroid_ms +=
            std::chrono::duration<double, std::milli>(
                centroid_end - centroid_begin
            ).count();

        // --------------------------------------------------------
        // 2. 根据 cluster 对 batch 内 query 重新分组
        // cluster_to_qids[c] 记录当前 batch 中需要访问簇 c 的 query
        // --------------------------------------------------------
        std::vector<std::vector<uint32_t>> cluster_to_qids(nlist);

        for (size_t local_qid = 0; local_qid < cur_batch; ++local_qid) {
            for (uint32_t cid : query_probes[local_qid]) {
                cluster_to_qids[cid].push_back(
                    static_cast<uint32_t>(local_qid)
                );
            }
        }

        // --------------------------------------------------------
        // 3. 对每个非空 cluster 执行一次 cuBLAS GEMM
        // --------------------------------------------------------
        for (size_t cid = 0; cid < nlist; ++cid) {
            const std::vector<uint32_t>& qids =
                cluster_to_qids[cid];

            if (qids.empty()) {
                continue;
            }

            IVFListGPU& list = index.lists[cid];

            if (list.list_size == 0) {
                continue;
            }

            size_t group_size = qids.size();
            size_t list_size = list.list_size;

            // ----------------------------------------------------
            // 3.1 组织当前 cluster 对应的 query group
            // ----------------------------------------------------
            std::vector<float> h_group_query(group_size * vecdim);

            for (size_t g = 0; g < group_size; ++g) {
                size_t local_qid = qids[g];
                size_t global_qid = batch_begin + local_qid;

                const float* src =
                    test_query + global_qid * vecdim;

                float* dst =
                    h_group_query.data() + g * vecdim;

                for (size_t j = 0; j < vecdim; ++j) {
                    dst[j] = src[j];
                }
            }

            // ----------------------------------------------------
            // 3.2 query group H2D
            // ----------------------------------------------------
            size_t query_bytes =
                group_size * vecdim * sizeof(float);

            auto h2d_begin = std::chrono::high_resolution_clock::now();

            CUDA_CHECK(cudaMemcpy(
                d_group_query,
                h_group_query.data(),
                query_bytes,
                cudaMemcpyHostToDevice
            ));

            auto h2d_end = std::chrono::high_resolution_clock::now();

            total_query_h2d_ms +=
                std::chrono::duration<double, std::milli>(
                    h2d_end - h2d_begin
                ).count();

            // ----------------------------------------------------
            // 3.3 cuBLAS GEMM
            //
            // 目标：
            //   list_vecs: list_size × vecdim，row-major
            //   group_query: group_size × vecdim，row-major
            //   score: list_size × group_size，row-major
            //
            // 需要计算：
            //   score = list_vecs × group_query^T
            //
            // cuBLAS 是 column-major。
            // row-major 的 score(list_size × group_size)
            // 在内存上等价于 column-major 的
            // score^T(group_size × list_size)。
            //
            // 因此实际让 cuBLAS 计算：
            //   score^T = group_query × list_vecs^T
            // ----------------------------------------------------
            const float alpha = 1.0f;
            const float beta = 0.0f;

            CUDA_CHECK(cudaEventRecord(gemm_start));

            CUBLAS_CHECK(cublasSgemm(
                handle,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                static_cast<int>(group_size),   // C 的行数：query group size
                static_cast<int>(list_size),    // C 的列数：cluster 内向量数
                static_cast<int>(vecdim),       // 向量维度
                &alpha,
                d_group_query,
                static_cast<int>(vecdim),       // A 原始视为 vecdim × group_size
                list.d_vecs,
                static_cast<int>(vecdim),       // B 原始视为 vecdim × list_size
                &beta,
                d_score,
                static_cast<int>(group_size)    // C leading dimension
            ));

            CUDA_CHECK(cudaEventRecord(gemm_stop));
            CUDA_CHECK(cudaEventSynchronize(gemm_stop));

            float gemm_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(
                &gemm_ms,
                gemm_start,
                gemm_stop
            ));

            total_gemm_ms += static_cast<double>(gemm_ms);

            // ----------------------------------------------------
            // 3.4 score D2H
            // score 在 CPU 端按 row-major:
            // score[i * group_size + g]
            // ----------------------------------------------------
            size_t score_size = list_size * group_size;
            size_t score_bytes =
                score_size * sizeof(float);

            std::vector<float> h_score(score_size);

            auto d2h_begin = std::chrono::high_resolution_clock::now();

            CUDA_CHECK(cudaMemcpy(
                h_score.data(),
                d_score,
                score_bytes,
                cudaMemcpyDeviceToHost
            ));

            auto d2h_end = std::chrono::high_resolution_clock::now();

            total_score_d2h_ms +=
                std::chrono::duration<double, std::milli>(
                    d2h_end - d2h_begin
                ).count();

            // ----------------------------------------------------
            // 3.5 CPU merge Top-k
            // ----------------------------------------------------
            auto merge_begin = std::chrono::high_resolution_clock::now();

            for (size_t g = 0; g < group_size; ++g) {
                size_t local_qid = qids[g];

                for (size_t i = 0; i < list_size; ++i) {
                    float inner_product =
                        h_score[i * group_size + g];

                    // 与 flat baseline 保持一致
                    float dis = 1.0f - inner_product;

                    uint32_t global_base_id = list.ids[i];

                    push_topk(
                        batch_topk[local_qid],
                        dis,
                        global_base_id,
                        k
                    );
                }
            }

            auto merge_end = std::chrono::high_resolution_clock::now();

            total_merge_ms +=
                std::chrono::duration<double, std::milli>(
                    merge_end - merge_begin
                ).count();
        }

        // --------------------------------------------------------
        // 4. 当前 batch 的 recall
        // --------------------------------------------------------
        for (size_t local_qid = 0; local_qid < cur_batch; ++local_qid) {
            size_t global_qid = batch_begin + local_qid;

            float recall = compute_recall_at_k(
                batch_topk[local_qid],
                test_gt,
                test_gt_d,
                global_qid,
                k
            );

            total_recall += static_cast<double>(recall);
        }
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

    std::cout << "\n========== IVF-GPU cuBLAS Baseline ==========\n";
    std::cout << "average recall: "
              << avg_recall << "\n";

    std::cout << "average latency (us): "
              << avg_latency_us << "\n";

    std::cout << "\n========== Time Breakdown ==========\n";
    std::cout << "index build time (ms): "
              << index_build_ms << "\n";

    std::cout << "centroid select total (ms): "
              << total_centroid_ms << "\n";

    std::cout << "query group H2D total (ms): "
              << total_query_h2d_ms << "\n";

    std::cout << "cluster cuBLAS GEMM total (ms): "
              << total_gemm_ms << "\n";

    std::cout << "score D2H total (ms): "
              << total_score_d2h_ms << "\n";

    std::cout << "CPU merge Top-k total (ms): "
              << total_merge_ms << "\n";

    std::cout << "total query time (ms): "
              << total_query_ms << "\n";

    std::cout << "\n========== Average Per Query ==========\n";
    std::cout << "centroid select avg (us): "
              << total_centroid_ms * 1000.0
                    / static_cast<double>(test_number)
              << "\n";

    std::cout << "query group H2D avg (us): "
              << total_query_h2d_ms * 1000.0
                    / static_cast<double>(test_number)
              << "\n";

    std::cout << "cluster cuBLAS GEMM avg (us): "
              << total_gemm_ms * 1000.0
                    / static_cast<double>(test_number)
              << "\n";

    std::cout << "score D2H avg (us): "
              << total_score_d2h_ms * 1000.0
                    / static_cast<double>(test_number)
              << "\n";

    std::cout << "CPU merge Top-k avg (us): "
              << total_merge_ms * 1000.0
                    / static_cast<double>(test_number)
              << "\n";

    CUDA_CHECK(cudaEventDestroy(gemm_start));
    CUDA_CHECK(cudaEventDestroy(gemm_stop));

    CUDA_CHECK(cudaFree(d_group_query));
    CUDA_CHECK(cudaFree(d_score));

    CUBLAS_CHECK(cublasDestroy(handle));

    free_ivf_gpu_index(index);

    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}