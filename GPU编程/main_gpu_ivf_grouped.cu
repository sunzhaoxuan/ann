#include <cuda_runtime.h>

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

    if (qid >= group_size || vid >= list_size) {
        return;
    }

    float sum = 0.0f;

    for (int t = 0; t < d; ++t) {
        sum += list_vecs[vid * d + t] * queries[qid * d + t];
    }

    score[vid * group_size + qid] = sum;
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

// ============================================================
// Top-2 centroid signature 分组
// ============================================================

struct QueryGroupKey {
    size_t qid = 0;
    uint32_t key0 = 0;
    uint32_t key1 = 0;
};

std::vector<size_t> build_top2_grouped_order(
    const float* test_query,
    size_t test_number,
    size_t vecdim,
    const std::vector<float>& centroids,
    size_t nlist,
    size_t nprobe,
    std::vector<std::vector<uint32_t>>& all_query_probes,
    double& centroid_select_ms,
    double& grouping_sort_ms
) {
    all_query_probes.resize(test_number);

    std::vector<QueryGroupKey> keys(test_number);

    auto centroid_begin = std::chrono::high_resolution_clock::now();

    for (size_t qid = 0; qid < test_number; ++qid) {
        const float* query = test_query + qid * vecdim;

        all_query_probes[qid] = select_top_nprobe(
            query,
            centroids,
            nlist,
            vecdim,
            nprobe
        );

        uint32_t key0 = all_query_probes[qid][0];
        uint32_t key1 = key0;

        if (all_query_probes[qid].size() >= 2) {
            key1 = all_query_probes[qid][1];
        }

        keys[qid].qid = qid;
        keys[qid].key0 = key0;
        keys[qid].key1 = key1;
    }

    auto centroid_end = std::chrono::high_resolution_clock::now();

    centroid_select_ms =
        std::chrono::duration<double, std::milli>(
            centroid_end - centroid_begin
        ).count();

    auto sort_begin = std::chrono::high_resolution_clock::now();

    std::sort(
        keys.begin(),
        keys.end(),
        [](const QueryGroupKey& a, const QueryGroupKey& b) {
            if (a.key0 != b.key0) {
                return a.key0 < b.key0;
            }

            if (a.key1 != b.key1) {
                return a.key1 < b.key1;
            }

            return a.qid < b.qid;
        }
    );

    std::vector<size_t> query_order(test_number);

    for (size_t i = 0; i < test_number; ++i) {
        query_order[i] = keys[i].qid;
    }

    auto sort_end = std::chrono::high_resolution_clock::now();

    grouping_sort_ms =
        std::chrono::duration<double, std::milli>(
            sort_end - sort_begin
        ).count();

    return query_order;
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
    // 查询阶段开始
    // 这里把 centroid select 和 query grouping 也计入 query time
    // ============================================================
    auto total_begin = std::chrono::high_resolution_clock::now();

    // ------------------------------------------------------------
    // 1. 预先计算每个 query 的 Top-nprobe 簇，并按 Top-2 簇签名排序
    // ------------------------------------------------------------
    std::vector<std::vector<uint32_t>> all_query_probes;

    double total_centroid_ms = 0.0;
    double grouping_sort_ms = 0.0;

    std::vector<size_t> query_order = build_top2_grouped_order(
        test_query,
        test_number,
        vecdim,
        index.centroids,
        nlist,
        nprobe,
        all_query_probes,
        total_centroid_ms,
        grouping_sort_ms
    );

    // ============================================================
    // GPU 临时 buffer
    // ============================================================
    float* d_group_query = nullptr;
    float* d_score = nullptr;

    size_t max_query_bytes =
        batch_size * vecdim * sizeof(float);

    size_t max_score_bytes =
        index.max_list_size * batch_size * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_group_query, max_query_bytes));
    CUDA_CHECK(cudaMalloc(&d_score, max_score_bytes));

    cudaEvent_t kernel_start;
    cudaEvent_t kernel_stop;

    CUDA_CHECK(cudaEventCreate(&kernel_start));
    CUDA_CHECK(cudaEventCreate(&kernel_stop));

    double total_recall = 0.0;

    double total_query_h2d_ms = 0.0;
    double total_kernel_ms = 0.0;
    double total_score_d2h_ms = 0.0;
    double total_merge_ms = 0.0;

    // 用于观察分组效果的统计指标
    size_t batch_count = 0;
    size_t total_unique_clusters_per_batch = 0;
    size_t total_nonempty_cluster_groups = 0;
    size_t total_query_cluster_entries = 0;
    size_t total_kernel_calls = 0;

    // ============================================================
    // 2. 按重排后的 query_order 进行 batch 查询
    // ============================================================
    for (size_t batch_begin = 0;
         batch_begin < test_number;
         batch_begin += batch_size) {

        size_t cur_batch =
            std::min(batch_size, test_number - batch_begin);

        batch_count++;

        std::vector<size_t> batch_global_qids(cur_batch);

        for (size_t local_qid = 0; local_qid < cur_batch; ++local_qid) {
            batch_global_qids[local_qid] =
                query_order[batch_begin + local_qid];
        }

        std::vector<
            std::priority_queue<std::pair<float, uint32_t>>
        > batch_topk(cur_batch);

        // --------------------------------------------------------
        // 2.1 根据 cluster 对当前 batch 的 query 分组
        // --------------------------------------------------------
        std::vector<std::vector<uint32_t>> cluster_to_qids(nlist);

        for (size_t local_qid = 0; local_qid < cur_batch; ++local_qid) {
            size_t global_qid = batch_global_qids[local_qid];

            const std::vector<uint32_t>& probes =
                all_query_probes[global_qid];

            for (uint32_t cid : probes) {
                cluster_to_qids[cid].push_back(
                    static_cast<uint32_t>(local_qid)
                );
            }
        }

        size_t unique_clusters_this_batch = 0;

        for (size_t cid = 0; cid < nlist; ++cid) {
            if (!cluster_to_qids[cid].empty()) {
                unique_clusters_this_batch++;
                total_query_cluster_entries += cluster_to_qids[cid].size();
            }
        }

        total_unique_clusters_per_batch += unique_clusters_this_batch;
        total_nonempty_cluster_groups += unique_clusters_this_batch;

        // --------------------------------------------------------
        // 2.2 对每个非空 cluster 执行簇内矩阵乘法
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

            total_kernel_calls++;

            // ----------------------------------------------------
            // 组织当前 cluster 对应的 query group
            // ----------------------------------------------------
            std::vector<float> h_group_query(group_size * vecdim);

            for (size_t g = 0; g < group_size; ++g) {
                size_t local_qid = qids[g];
                size_t global_qid = batch_global_qids[local_qid];

                const float* src =
                    test_query + global_qid * vecdim;

                float* dst =
                    h_group_query.data() + g * vecdim;

                for (size_t j = 0; j < vecdim; ++j) {
                    dst[j] = src[j];
                }
            }

            // ----------------------------------------------------
            // Query group H2D
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
            // GPU 簇内矩阵乘法
            // ----------------------------------------------------
            dim3 block(16, 16);
            dim3 grid(
                static_cast<unsigned int>((group_size + block.x - 1) / block.x),
                static_cast<unsigned int>((list_size + block.y - 1) / block.y)
            );

            CUDA_CHECK(cudaEventRecord(kernel_start));

            gpu_ip_matmul_kernel<<<grid, block>>>(
                list.d_vecs,
                d_group_query,
                d_score,
                static_cast<int>(list_size),
                static_cast<int>(vecdim),
                static_cast<int>(group_size)
            );

            CUDA_CHECK(cudaGetLastError());

            CUDA_CHECK(cudaEventRecord(kernel_stop));
            CUDA_CHECK(cudaEventSynchronize(kernel_stop));

            float kernel_ms = 0.0f;

            CUDA_CHECK(cudaEventElapsedTime(
                &kernel_ms,
                kernel_start,
                kernel_stop
            ));

            total_kernel_ms += static_cast<double>(kernel_ms);

            // ----------------------------------------------------
            // Score D2H
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
            // CPU merge Top-k
            // ----------------------------------------------------
            auto merge_begin = std::chrono::high_resolution_clock::now();

            for (size_t g = 0; g < group_size; ++g) {
                size_t local_qid = qids[g];

                for (size_t i = 0; i < list_size; ++i) {
                    float inner_product =
                        h_score[i * group_size + g];

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
        // 2.3 当前 batch 的 recall
        // --------------------------------------------------------
        for (size_t local_qid = 0; local_qid < cur_batch; ++local_qid) {
            size_t global_qid = batch_global_qids[local_qid];

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

    double avg_unique_clusters_per_batch =
        static_cast<double>(total_unique_clusters_per_batch)
        / static_cast<double>(batch_count);

    double avg_query_group_size =
        0.0;

    if (total_nonempty_cluster_groups > 0) {
        avg_query_group_size =
            static_cast<double>(total_query_cluster_entries)
            / static_cast<double>(total_nonempty_cluster_groups);
    }

    std::cout << std::fixed << std::setprecision(6);

    std::cout << "\n========== IVF-GPU Grouped Baseline ==========\n";
    std::cout << "grouping method: Top-2 centroid signature sorting\n";

    std::cout << "average recall: "
              << avg_recall << "\n";

    std::cout << "average latency (us): "
              << avg_latency_us << "\n";

    std::cout << "\n========== Time Breakdown ==========\n";
    std::cout << "index build time (ms): "
              << index_build_ms << "\n";

    std::cout << "centroid select total (ms): "
              << total_centroid_ms << "\n";

    std::cout << "query grouping sort total (ms): "
              << grouping_sort_ms << "\n";

    std::cout << "query group H2D total (ms): "
              << total_query_h2d_ms << "\n";

    std::cout << "cluster GEMM kernel total (ms): "
              << total_kernel_ms << "\n";

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

    std::cout << "query grouping sort avg (us): "
              << grouping_sort_ms * 1000.0
                    / static_cast<double>(test_number)
              << "\n";

    std::cout << "query group H2D avg (us): "
              << total_query_h2d_ms * 1000.0
                    / static_cast<double>(test_number)
              << "\n";

    std::cout << "cluster GEMM kernel avg (us): "
              << total_kernel_ms * 1000.0
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

    std::cout << "\n========== Grouping Statistics ==========\n";
    std::cout << "batch count: "
              << batch_count << "\n";

    std::cout << "avg unique clusters per batch: "
              << avg_unique_clusters_per_batch << "\n";

    std::cout << "avg query group size per visited cluster: "
              << avg_query_group_size << "\n";

    std::cout << "total cluster kernel calls: "
              << total_kernel_calls << "\n";

    CUDA_CHECK(cudaEventDestroy(kernel_start));
    CUDA_CHECK(cudaEventDestroy(kernel_stop));

    CUDA_CHECK(cudaFree(d_group_query));
    CUDA_CHECK(cudaFree(d_score));

    free_ivf_gpu_index(index);

    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}
