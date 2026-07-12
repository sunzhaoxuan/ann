#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <random>
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

// ============================================================
// GPU kernel：计算一个 IVF 簇内向量与一个 query group 的内积矩阵
//
// list_vecs : list_size × d，行主序
// queries   : group_size × d，行主序
// score     : list_size × group_size，行主序
//
// score[i][j] = inner_product(list_vecs[i], queries[j])
// ============================================================
__global__ void gpu_ip_matmul_kernel(
    const float* list_vecs,
    const float* queries,
    float* score,
    int list_size,
    int d,
    int group_size
) {
    int qid = blockIdx.x * blockDim.x + threadIdx.x;
    int vid = blockIdx.y * blockDim.y + threadIdx.y;

    if (vid >= list_size || qid >= group_size) {
        return;
    }

    float sum = 0.0f;

    for (int t = 0; t < d; ++t) {
        sum += list_vecs[vid * d + t] * queries[qid * d + t];
    }

    score[vid * group_size + qid] = sum;
}

static inline float cpu_inner_product(
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

static void normalize_vector(float* x, size_t d) {
    double norm2 = 0.0;

    for (size_t i = 0; i < d; ++i) {
        norm2 += static_cast<double>(x[i]) * static_cast<double>(x[i]);
    }

    double norm = std::sqrt(norm2);

    if (norm < 1e-12) {
        return;
    }

    for (size_t i = 0; i < d; ++i) {
        x[i] = static_cast<float>(x[i] / norm);
    }
}

// 找到与向量 x 内积最大的 centroid。
// DEEP100K 使用 IP 距离，原始距离为 dis = 1 - ip。
// 因此分簇时选择 inner product 最大的 centroid。
static int find_best_centroid_ip(
    const float* x,
    const std::vector<float>& centroids,
    size_t nlist,
    size_t d
) {
    int best_id = 0;
    float best_score = cpu_inner_product(x, centroids.data(), d);

    for (size_t c = 1; c < nlist; ++c) {
        float score = cpu_inner_product(
            x,
            centroids.data() + c * d,
            d
        );

        if (score > best_score) {
            best_score = score;
            best_id = static_cast<int>(c);
        }
    }

    return best_id;
}

// CPU 简单 KMeans，用于训练 IVF coarse centroids。
// 这里为了代码独立，直接实现一个轻量版本。
// 实验中 nlist、train_size、iters 可以根据时间调整。
std::vector<float> train_kmeans_ip(
    const float* base,
    size_t base_number,
    size_t d,
    size_t nlist,
    size_t train_size,
    int iters
) {
    train_size = std::min(train_size, base_number);

    std::vector<uint32_t> train_ids(base_number);
    for (size_t i = 0; i < base_number; ++i) {
        train_ids[i] = static_cast<uint32_t>(i);
    }

    std::mt19937 rng(2026);
    std::shuffle(train_ids.begin(), train_ids.end(), rng);
    train_ids.resize(train_size);

    std::vector<float> centroids(nlist * d, 0.0f);

    // 随机选择训练样本初始化 centroid
    for (size_t c = 0; c < nlist; ++c) {
        uint32_t id = train_ids[c % train_size];
        const float* src = base + static_cast<size_t>(id) * d;

        std::copy(src, src + d, centroids.begin() + c * d);
        normalize_vector(centroids.data() + c * d, d);
    }

    std::vector<int> assign(train_size, 0);

    for (int iter = 0; iter < iters; ++iter) {
        std::vector<float> new_centroids(nlist * d, 0.0f);
        std::vector<int> counts(nlist, 0);

        // assignment
        for (size_t i = 0; i < train_size; ++i) {
            uint32_t id = train_ids[i];
            const float* x = base + static_cast<size_t>(id) * d;

            int best = find_best_centroid_ip(
                x,
                centroids,
                nlist,
                d
            );

            assign[i] = best;
            counts[best]++;

            float* dst = new_centroids.data() + static_cast<size_t>(best) * d;

            for (size_t j = 0; j < d; ++j) {
                dst[j] += x[j];
            }
        }

        // update
        for (size_t c = 0; c < nlist; ++c) {
            float* dst = new_centroids.data() + c * d;

            if (counts[c] > 0) {
                for (size_t j = 0; j < d; ++j) {
                    dst[j] /= static_cast<float>(counts[c]);
                }
                normalize_vector(dst, d);
            } else {
                // 空簇重新随机初始化
                uint32_t id = train_ids[c % train_size];
                const float* src = base + static_cast<size_t>(id) * d;
                std::copy(src, src + d, dst);
                normalize_vector(dst, d);
            }
        }

        centroids.swap(new_centroids);

        std::cerr << "[KMeans] iter "
                  << iter + 1
                  << " / "
                  << iters
                  << " finished\n";
    }

    return centroids;
}

struct IVFListCPU {
    std::vector<uint32_t> ids;
    std::vector<float> vecs;  // list_size × d
};

struct IVFListGPU {
    float* d_vecs = nullptr;
    size_t list_size = 0;
    std::vector<uint32_t> ids;
};

struct IVFGPUIndex {
    size_t nlist = 0;
    size_t d = 0;
    std::vector<float> centroids;
    std::vector<IVFListGPU> lists;
    size_t max_list_size = 1;
};

IVFGPUIndex build_ivf_gpu_index(
    const float* base,
    size_t base_number,
    size_t d,
    size_t nlist,
    size_t train_size,
    int kmeans_iters
) {
    IVFGPUIndex index;
    index.nlist = nlist;
    index.d = d;

    std::cerr << "[IVF] training centroids...\n";

    index.centroids = train_kmeans_ip(
        base,
        base_number,
        d,
        nlist,
        train_size,
        kmeans_iters
    );

    std::cerr << "[IVF] assigning base vectors to lists...\n";

    std::vector<IVFListCPU> cpu_lists(nlist);

    for (size_t i = 0; i < base_number; ++i) {
        const float* x = base + i * d;

        int cid = find_best_centroid_ip(
            x,
            index.centroids,
            nlist,
            d
        );

        cpu_lists[cid].ids.push_back(static_cast<uint32_t>(i));
        cpu_lists[cid].vecs.insert(
            cpu_lists[cid].vecs.end(),
            x,
            x + d
        );
    }

    index.lists.resize(nlist);

    std::cerr << "[IVF] copying each list to GPU...\n";

    for (size_t c = 0; c < nlist; ++c) {
        size_t list_size = cpu_lists[c].ids.size();

        index.lists[c].list_size = list_size;
        index.lists[c].ids.swap(cpu_lists[c].ids);

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

        std::cerr << "  list "
                  << c
                  << " size = "
                  << list_size
                  << "\n";
    }

    std::cerr << "[IVF] max_list_size = "
              << index.max_list_size
              << "\n";

    return index;
}

void free_ivf_gpu_index(IVFGPUIndex& index) {
    for (size_t c = 0; c < index.nlist; ++c) {
        if (index.lists[c].d_vecs != nullptr) {
            CUDA_CHECK(cudaFree(index.lists[c].d_vecs));
            index.lists[c].d_vecs = nullptr;
        }
    }
}

// 选择当前 query 最近的 Top-nprobe 个 IVF 簇。
// 这里最近等价于 inner product 最大。
std::vector<int> select_top_nprobe(
    const float* query,
    const std::vector<float>& centroids,
    size_t nlist,
    size_t d,
    size_t nprobe
) {
    std::vector<std::pair<float, int>> scores;
    scores.reserve(nlist);

    for (size_t c = 0; c < nlist; ++c) {
        float ip = cpu_inner_product(
            query,
            centroids.data() + c * d,
            d
        );

        scores.push_back({ip, static_cast<int>(c)});
    }

    if (nprobe < scores.size()) {
        std::partial_sort(
            scores.begin(),
            scores.begin() + static_cast<long long>(nprobe),
            scores.end(),
            [](const auto& a, const auto& b) {
                return a.first > b.first;
            }
        );
    } else {
        std::sort(
            scores.begin(),
            scores.end(),
            [](const auto& a, const auto& b) {
                return a.first > b.first;
            }
        );
    }

    std::vector<int> result;
    size_t take = std::min(nprobe, scores.size());

    for (size_t i = 0; i < take; ++i) {
        result.push_back(scores[i].second);
    }

    return result;
}

static inline void topk_push(
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

    // IVF 参数
    const size_t nlist = 100;
    const size_t nprobe = 8;
    const size_t train_size = 10000;
    const int kmeans_iters = 6;

    // GPU batch 参数
    const size_t batch_size = 64;

    std::cout << "base_number = " << base_number << "\n";
    std::cout << "test_number = " << test_number << "\n";
    std::cout << "vecdim      = " << vecdim << "\n";
    std::cout << "nlist       = " << nlist << "\n";
    std::cout << "nprobe      = " << nprobe << "\n";
    std::cout << "batch_size  = " << batch_size << "\n";

    auto build_begin = std::chrono::high_resolution_clock::now();

    IVFGPUIndex index = build_ivf_gpu_index(
        base,
        base_number,
        vecdim,
        nlist,
        train_size,
        kmeans_iters
    );

    auto build_end = std::chrono::high_resolution_clock::now();

    double build_ms =
        std::chrono::duration<double, std::milli>(
            build_end - build_begin
        ).count();

    // 用于每次 cluster group 查询的临时 GPU buffer
    float* d_query_group = nullptr;
    float* d_score = nullptr;

    size_t max_query_bytes = batch_size * vecdim * sizeof(float);
    size_t max_score_bytes =
        index.max_list_size * batch_size * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_query_group, max_query_bytes));
    CUDA_CHECK(cudaMalloc(&d_score, max_score_bytes));

    double total_recall = 0.0;

    double total_select_ms = 0.0;
    double total_h2d_ms = 0.0;
    double total_kernel_ms = 0.0;
    double total_d2h_ms = 0.0;
    double total_merge_ms = 0.0;

    auto total_begin = std::chrono::high_resolution_clock::now();

    for (size_t batch_begin = 0;
         batch_begin < test_number;
         batch_begin += batch_size) {

        size_t cur_batch = std::min(batch_size, test_number - batch_begin);

        // 每个 query 一个候选优先队列
        std::vector<std::priority_queue<std::pair<float, uint32_t>>> batch_res(
            cur_batch
        );

        // cluster_to_qids[c] 存放 batch 内哪些 query 需要搜索 cluster c
        std::vector<std::vector<int>> cluster_to_qids(nlist);

        // ------------------------------------------------------------
        // 1. CPU 端选择每个 query 的 Top-nprobe 簇
        // ------------------------------------------------------------
        auto select_begin = std::chrono::high_resolution_clock::now();

        for (size_t local_qid = 0; local_qid < cur_batch; ++local_qid) {
            size_t global_qid = batch_begin + local_qid;
            const float* qptr = test_query + global_qid * vecdim;

            std::vector<int> selected = select_top_nprobe(
                qptr,
                index.centroids,
                nlist,
                vecdim,
                nprobe
            );

            for (int cid : selected) {
                cluster_to_qids[cid].push_back(static_cast<int>(local_qid));
            }
        }

        auto select_end = std::chrono::high_resolution_clock::now();

        total_select_ms +=
            std::chrono::duration<double, std::milli>(
                select_end - select_begin
            ).count();

        // ------------------------------------------------------------
        // 2. 按 cluster 分组，在 GPU 上计算簇内矩阵乘法
        // ------------------------------------------------------------
        for (size_t cid = 0; cid < nlist; ++cid) {
            const auto& qids = cluster_to_qids[cid];

            if (qids.empty()) {
                continue;
            }

            size_t group_size = qids.size();
            size_t list_size = index.lists[cid].list_size;

            if (list_size == 0) {
                continue;
            }

            // 组织当前 cluster 对应的 query group
            std::vector<float> h_query_group(group_size * vecdim);

            for (size_t g = 0; g < group_size; ++g) {
                int local_qid = qids[g];
                size_t global_qid = batch_begin + static_cast<size_t>(local_qid);

                const float* src = test_query + global_qid * vecdim;
                float* dst = h_query_group.data() + g * vecdim;

                std::copy(src, src + vecdim, dst);
            }

            size_t query_bytes = group_size * vecdim * sizeof(float);
            size_t score_bytes = list_size * group_size * sizeof(float);

            auto h2d_begin = std::chrono::high_resolution_clock::now();

            CUDA_CHECK(cudaMemcpy(
                d_query_group,
                h_query_group.data(),
                query_bytes,
                cudaMemcpyHostToDevice
            ));

            auto h2d_end = std::chrono::high_resolution_clock::now();

            total_h2d_ms +=
                std::chrono::duration<double, std::milli>(
                    h2d_end - h2d_begin
                ).count();

            dim3 block(16, 16);
            dim3 grid(
                static_cast<unsigned int>((group_size + block.x - 1) / block.x),
                static_cast<unsigned int>((list_size + block.y - 1) / block.y)
            );

            cudaEvent_t start, stop;
            CUDA_CHECK(cudaEventCreate(&start));
            CUDA_CHECK(cudaEventCreate(&stop));

            CUDA_CHECK(cudaEventRecord(start));

            gpu_ip_matmul_kernel<<<grid, block>>>(
                index.lists[cid].d_vecs,
                d_query_group,
                d_score,
                static_cast<int>(list_size),
                static_cast<int>(vecdim),
                static_cast<int>(group_size)
            );

            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaGetLastError());

            float kernel_ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, start, stop));

            total_kernel_ms += static_cast<double>(kernel_ms);

            CUDA_CHECK(cudaEventDestroy(start));
            CUDA_CHECK(cudaEventDestroy(stop));

            std::vector<float> h_score(list_size * group_size);

            auto d2h_begin = std::chrono::high_resolution_clock::now();

            CUDA_CHECK(cudaMemcpy(
                h_score.data(),
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
            // 3. CPU 端把当前 cluster 的结果合并到每个 query 的 Top-k
            // --------------------------------------------------------
            auto merge_begin = std::chrono::high_resolution_clock::now();

            for (size_t g = 0; g < group_size; ++g) {
                int local_qid = qids[g];
                auto& pq = batch_res[local_qid];

                for (size_t row = 0; row < list_size; ++row) {
                    float ip = h_score[row * group_size + g];
                    float dis = 1.0f - ip;

                    uint32_t global_id = index.lists[cid].ids[row];

                    topk_push(pq, dis, global_id, k);
                }
            }

            auto merge_end = std::chrono::high_resolution_clock::now();

            total_merge_ms +=
                std::chrono::duration<double, std::milli>(
                    merge_end - merge_begin
                ).count();
        }

        // ------------------------------------------------------------
        // 4. 当前 batch 计算 recall
        // ------------------------------------------------------------
        for (size_t local_qid = 0; local_qid < cur_batch; ++local_qid) {
            size_t global_qid = batch_begin + local_qid;

            float recall = compute_recall_at_k(
                batch_res[local_qid],
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

    std::cout << "\n========== IVF-GPU Baseline ==========\n";
    std::cout << "average recall: "
              << avg_recall
              << "\n";

    std::cout << "average latency (us): "
              << avg_latency_us
              << "\n";

    std::cout << "\n========== Time Breakdown ==========\n";
    std::cout << "index build time (ms): "
              << build_ms
              << "\n";

    std::cout << "centroid select total (ms): "
              << total_select_ms
              << "\n";

    std::cout << "query group H2D total (ms): "
              << total_h2d_ms
              << "\n";

    std::cout << "cluster GEMM kernel total (ms): "
              << total_kernel_ms
              << "\n";

    std::cout << "score D2H total (ms): "
              << total_d2h_ms
              << "\n";

    std::cout << "CPU merge Top-k total (ms): "
              << total_merge_ms
              << "\n";

    std::cout << "total query time (ms): "
              << total_query_ms
              << "\n";

    std::cout << "\n========== Average Per Query ==========\n";
    std::cout << "centroid select avg (us): "
              << total_select_ms * 1000.0 / static_cast<double>(test_number)
              << "\n";

    std::cout << "query group H2D avg (us): "
              << total_h2d_ms * 1000.0 / static_cast<double>(test_number)
              << "\n";

    std::cout << "cluster GEMM kernel avg (us): "
              << total_kernel_ms * 1000.0 / static_cast<double>(test_number)
              << "\n";

    std::cout << "score D2H avg (us): "
              << total_d2h_ms * 1000.0 / static_cast<double>(test_number)
              << "\n";

    std::cout << "CPU merge Top-k avg (us): "
              << total_merge_ms * 1000.0 / static_cast<double>(test_number)
              << "\n";

    CUDA_CHECK(cudaFree(d_query_group));
    CUDA_CHECK(cudaFree(d_score));

    free_ivf_gpu_index(index);

    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}
