#pragma once

#include <mpi.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <utility>
#include <vector>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif

#include "hnswlib/hnswlib/hnswlib.h"

// ============================================================
// SIMD inner product
// 用于 rank 0 最终候选 rerank。
// HNSW 内部距离计算如果已经在 hnswlib 中替换为 NEON，
// 那么 local HNSW search 和 upper HNSW search 也会间接使用 SIMD。
// ============================================================
static inline float hoh_inner_product_neon_fma(
    const float* a,
    const float* b,
    size_t dim
) {
    float sum = 0.0f;

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
    float32x4_t acc = vdupq_n_f32(0.0f);

    size_t i = 0;

    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        acc = vfmaq_f32(acc, va, vb);
    }

    float tmp[4];
    vst1q_f32(tmp, acc);

    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < dim; ++i) {
        sum += a[i] * b[i];
    }
#else
    for (size_t i = 0; i < dim; ++i) {
        sum += a[i] * b[i];
    }
#endif

    return sum;
}

static inline float hoh_l2_distance_scalar(
    const float* a,
    const float* b,
    size_t dim
) {
    float sum = 0.0f;

    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }

    return sum;
}

// ============================================================
// Candidate / profile
// ============================================================
struct HNSWOnHNSWCandidate {
    float dist;
    uint32_t id;
};

struct HNSWOnHNSWLocalProfile {
    size_t selected_this_rank;
    size_t local_points;
    size_t returned_candidates;

    HNSWOnHNSWLocalProfile()
        : selected_this_rank(0),
          local_points(0),
          returned_candidates(0) {}
};

struct HNSWOnHNSWProfileOne {
    double route_us_min;
    double route_us_avg;
    double route_us_max;

    double local_search_us_min;
    double local_search_us_avg;
    double local_search_us_max;

    double pack_us_max;
    double gather_us_max;
    double merge_us;
    double total_us;

    unsigned long long selected_ranks_sum;
    unsigned long long selected_ranks_max;

    unsigned long long local_points_sum;
    unsigned long long local_points_max;

    unsigned long long returned_candidates_sum;
    unsigned long long returned_candidates_max;

    HNSWOnHNSWProfileOne()
        : route_us_min(0.0),
          route_us_avg(0.0),
          route_us_max(0.0),
          local_search_us_min(0.0),
          local_search_us_avg(0.0),
          local_search_us_max(0.0),
          pack_us_max(0.0),
          gather_us_max(0.0),
          merge_us(0.0),
          total_us(0.0),
          selected_ranks_sum(0),
          selected_ranks_max(0),
          local_points_sum(0),
          local_points_max(0),
          returned_candidates_sum(0),
          returned_candidates_max(0) {}
};

struct HNSWOnHNSWProfileTotal {
    int cnt;

    double route_us_min;
    double route_us_avg;
    double route_us_max;

    double local_search_us_min;
    double local_search_us_avg;
    double local_search_us_max;

    double pack_us_max;
    double gather_us_max;
    double merge_us;
    double total_us;

    double selected_ranks_sum;
    double selected_ranks_max;

    double local_points_sum;
    double local_points_max;

    double returned_candidates_sum;
    double returned_candidates_max;

    HNSWOnHNSWProfileTotal()
        : cnt(0),
          route_us_min(0.0),
          route_us_avg(0.0),
          route_us_max(0.0),
          local_search_us_min(0.0),
          local_search_us_avg(0.0),
          local_search_us_max(0.0),
          pack_us_max(0.0),
          gather_us_max(0.0),
          merge_us(0.0),
          total_us(0.0),
          selected_ranks_sum(0.0),
          selected_ranks_max(0.0),
          local_points_sum(0.0),
          local_points_max(0.0),
          returned_candidates_sum(0.0),
          returned_candidates_max(0.0) {}

    void add(const HNSWOnHNSWProfileOne& p) {
        ++cnt;

        route_us_min += p.route_us_min;
        route_us_avg += p.route_us_avg;
        route_us_max += p.route_us_max;

        local_search_us_min += p.local_search_us_min;
        local_search_us_avg += p.local_search_us_avg;
        local_search_us_max += p.local_search_us_max;

        pack_us_max += p.pack_us_max;
        gather_us_max += p.gather_us_max;
        merge_us += p.merge_us;
        total_us += p.total_us;

        selected_ranks_sum += static_cast<double>(p.selected_ranks_sum);
        selected_ranks_max += static_cast<double>(p.selected_ranks_max);

        local_points_sum += static_cast<double>(p.local_points_sum);
        local_points_max += static_cast<double>(p.local_points_max);

        returned_candidates_sum += static_cast<double>(p.returned_candidates_sum);
        returned_candidates_max += static_cast<double>(p.returned_candidates_max);
    }

    void print_average() const {
        if (cnt == 0) {
            return;
        }

        double inv = 1.0 / static_cast<double>(cnt);

        std::cout << "\nMPI HNSW-on-HNSW profile average per query (us):\n";
        std::cout << "  Route min       : " << route_us_min * inv << "\n";
        std::cout << "  Route avg       : " << route_us_avg * inv << "\n";
        std::cout << "  Route max       : " << route_us_max * inv << "\n";
        std::cout << "  Local search min: " << local_search_us_min * inv << "\n";
        std::cout << "  Local search avg: " << local_search_us_avg * inv << "\n";
        std::cout << "  Local search max: " << local_search_us_max * inv << "\n";
        std::cout << "  Pack max        : " << pack_us_max * inv << "\n";
        std::cout << "  Gather max      : " << gather_us_max * inv << "\n";
        std::cout << "  Merge           : " << merge_us * inv << "\n";
        std::cout << "  Total           : " << total_us * inv << "\n";

        std::cout << "\nMPI HNSW-on-HNSW workload average per query:\n";
        std::cout << "  selected ranks sum      : " << selected_ranks_sum * inv << "\n";
        std::cout << "  selected ranks max      : " << selected_ranks_max * inv << "\n";
        std::cout << "  local points sum        : " << local_points_sum * inv << "\n";
        std::cout << "  local points max        : " << local_points_max * inv << "\n";
        std::cout << "  returned candidates sum : " << returned_candidates_sum * inv << "\n";
        std::cout << "  returned candidates max : " << returned_candidates_max * inv << "\n";

        std::cout << "\nMPI HNSW-on-HNSW load balance indicators:\n";

        if (local_search_us_avg > 0.0) {
            std::cout << "  local search max/avg: "
                      << local_search_us_max / local_search_us_avg
                      << "\n";
        }

        if (local_points_sum > 0.0) {
            std::cout << "  local points max/sum: "
                      << local_points_max / local_points_sum
                      << "\n";
        }
    }
};

static inline void hoh_update_topk_dist(
    float dist,
    uint32_t id,
    std::priority_queue<std::pair<float, uint32_t> >& topk,
    size_t k
) {
    if (k == 0) {
        return;
    }

    if (topk.size() < k) {
        topk.push(std::make_pair(dist, id));
    } else if (dist < topk.top().first) {
        topk.pop();
        topk.push(std::make_pair(dist, id));
    }
}

static inline void hoh_pack_candidates(
    std::priority_queue<std::pair<float, uint32_t> > pq,
    size_t local_p,
    std::vector<HNSWOnHNSWCandidate>& local_cands
) {
    const float INF = std::numeric_limits<float>::infinity();

    local_cands.resize(local_p);

    for (size_t i = 0; i < local_p; ++i) {
        local_cands[i].dist = INF;
        local_cands[i].id = UINT32_MAX;
    }

    size_t pos = 0;

    while (!pq.empty() && pos < local_p) {
        local_cands[pos].dist = pq.top().first;
        local_cands[pos].id = pq.top().second;
        pq.pop();
        ++pos;
    }
}

// rank 0 最终 merge 前使用 SIMD 重新计算候选距离。
// 默认按 inner product 检索：dist = 1 - inner_product。
static inline std::priority_queue<std::pair<float, uint32_t> >
hoh_merge_global_topk_simd_rerank(
    const std::vector<HNSWOnHNSWCandidate>& all_cands,
    const float* query,
    const float* base,
    size_t vecdim,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t> > global_topk;

    for (size_t i = 0; i < all_cands.size(); ++i) {
        uint32_t id = all_cands[i].id;

        if (id == UINT32_MAX) {
            continue;
        }

        const float* base_vec =
            base + static_cast<size_t>(id) * vecdim;

        float score = hoh_inner_product_neon_fma(
            query,
            base_vec,
            vecdim
        );

        float dist = 1.0f - score;

        hoh_update_topk_dist(
            dist,
            id,
            global_topk,
            k
        );
    }

    return global_topk;
}

static inline float hoh_compute_recall_at_k(
    std::priority_queue<std::pair<float, uint32_t> > res,
    const int* test_gt,
    int test_gt_d,
    int qid,
    size_t k
) {
    std::set<uint32_t> gtset;

    for (size_t j = 0; j < k; ++j) {
        int t = test_gt[j + qid * test_gt_d];
        gtset.insert(static_cast<uint32_t>(t));
    }

    size_t acc = 0;

    while (!res.empty()) {
        uint32_t id = res.top().second;

        if (gtset.find(id) != gtset.end()) {
            ++acc;
        }

        res.pop();
    }

    return static_cast<float>(acc) / static_cast<float>(k);
}

class HNSWOnHNSWMPIIndex {
private:
    float* base_data;
    size_t base_number;
    size_t vecdim;

    int rank;
    int world_size;

    int lower_hnsw_M;
    int lower_ef_construction;
    int lower_ef_search;

    int upper_hnsw_M;
    int upper_ef_construction;
    int upper_ef_search;

    size_t train_size;
    int kmeans_iters;
    std::string local_hnsw_layout;
    size_t gorder_window;
    size_t reordered_moved_nodes;
    bool local_edge_profiling;

    std::vector<float> partition_reps;
    std::vector<uint32_t> local_ids;

    hnswlib::InnerProductSpace lower_space;
    hnswlib::InnerProductSpace upper_space;

    std::unique_ptr<hnswlib::HierarchicalNSW<float> > local_hnsw;
    std::unique_ptr<hnswlib::HierarchicalNSW<float> > upper_hnsw;

private:
    const float* base_vec(uint32_t id) const {
        return base_data + static_cast<size_t>(id) * vecdim;
    }

    void init_partition_reps() {
        size_t p = static_cast<size_t>(world_size);
        partition_reps.assign(p * vecdim, 0.0f);

        size_t real_train_size = std::min(train_size, base_number);

        if (real_train_size == 0) {
            return;
        }

        for (size_t c = 0; c < p; ++c) {
            size_t tid = c % real_train_size;
            size_t bid = tid * base_number / real_train_size;

            const float* src = base_data + bid * vecdim;
            float* dst = partition_reps.data() + c * vecdim;

            std::memcpy(dst, src, vecdim * sizeof(float));
        }
    }

    uint32_t nearest_rep_l2(const float* x) const {
        size_t p = static_cast<size_t>(world_size);

        uint32_t best = 0;
        float best_dist = std::numeric_limits<float>::infinity();

        for (size_t c = 0; c < p; ++c) {
            const float* rep = partition_reps.data() + c * vecdim;

            float dist = hoh_l2_distance_scalar(
                x,
                rep,
                vecdim
            );

            if (dist < best_dist) {
                best_dist = dist;
                best = static_cast<uint32_t>(c);
            }
        }

        return best;
    }

    void train_kmeans_partitions() {
        size_t p = static_cast<size_t>(world_size);

        init_partition_reps();

        size_t real_train_size = std::min(train_size, base_number);

        if (real_train_size == 0) {
            return;
        }

        std::vector<uint32_t> train_ids(real_train_size);

        for (size_t i = 0; i < real_train_size; ++i) {
            train_ids[i] =
                static_cast<uint32_t>(i * base_number / real_train_size);
        }

        std::vector<float> new_reps(p * vecdim);
        std::vector<size_t> counts(p);

        for (int iter = 0; iter < kmeans_iters; ++iter) {
            std::fill(new_reps.begin(), new_reps.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t i = 0; i < real_train_size; ++i) {
                uint32_t id = train_ids[i];
                const float* x = base_vec(id);

                uint32_t cid = nearest_rep_l2(x);

                float* acc =
                    new_reps.data()
                    + static_cast<size_t>(cid) * vecdim;

                for (size_t d = 0; d < vecdim; ++d) {
                    acc[d] += x[d];
                }

                counts[cid]++;
            }

            for (size_t c = 0; c < p; ++c) {
                if (counts[c] == 0) {
                    continue;
                }

                float* dst = partition_reps.data() + c * vecdim;
                float* src = new_reps.data() + c * vecdim;

                float inv = 1.0f / static_cast<float>(counts[c]);

                for (size_t d = 0; d < vecdim; ++d) {
                    dst[d] = src[d] * inv;
                }
            }
        }
    }

    void build_local_partition() {
        local_ids.clear();

        for (size_t i = 0; i < base_number; ++i) {
            const float* x = base_data + i * vecdim;
            uint32_t cid = nearest_rep_l2(x);

            if (static_cast<int>(cid) == rank) {
                local_ids.push_back(static_cast<uint32_t>(i));
            }
        }
    }

    void build_local_hnsw() {
        if (local_ids.empty()) {
            return;
        }

        local_hnsw.reset(
            new hnswlib::HierarchicalNSW<float>(
                &lower_space,
                local_ids.size(),
                lower_hnsw_M,
                lower_ef_construction
            )
        );

        local_hnsw->setEf(lower_ef_search);

        for (size_t i = 0; i < local_ids.size(); ++i) {
            uint32_t gid = local_ids[i];

            local_hnsw->addPoint(
                base_vec(gid),
                static_cast<hnswlib::labeltype>(gid)
            );
        }
    }

    void build_upper_hnsw() {
        size_t p = static_cast<size_t>(world_size);

        if (p == 0) {
            return;
        }

        upper_hnsw.reset(
            new hnswlib::HierarchicalNSW<float>(
                &upper_space,
                p,
                upper_hnsw_M,
                upper_ef_construction
            )
        );

        upper_hnsw->setEf(upper_ef_search);

        for (size_t pid = 0; pid < p; ++pid) {
            const float* rep =
                partition_reps.data() + pid * vecdim;

            upper_hnsw->addPoint(
                rep,
                static_cast<hnswlib::labeltype>(pid)
            );
        }
    }

public:
    HNSWOnHNSWMPIIndex(
        float* base,
        size_t base_num,
        size_t dim,
        int rank_,
        int world_size_,
        int lower_hnsw_M_,
        int lower_ef_construction_,
        int lower_ef_search_,
        int upper_hnsw_M_,
        int upper_ef_construction_,
        int upper_ef_search_,
        size_t train_size_ = 10000,
        int kmeans_iters_ = 6,
        const std::string& local_hnsw_layout_ = "original",
        size_t gorder_window_ = 5
    )
        : base_data(base),
          base_number(base_num),
          vecdim(dim),
          rank(rank_),
          world_size(world_size_),
          lower_hnsw_M(lower_hnsw_M_),
          lower_ef_construction(lower_ef_construction_),
          lower_ef_search(lower_ef_search_),
          upper_hnsw_M(upper_hnsw_M_),
          upper_ef_construction(upper_ef_construction_),
          upper_ef_search(upper_ef_search_),
          train_size(train_size_),
          kmeans_iters(kmeans_iters_),
          local_hnsw_layout(local_hnsw_layout_),
          gorder_window(gorder_window_),
          reordered_moved_nodes(0),
          local_edge_profiling(false),
          lower_space(static_cast<int>(dim)),
          upper_space(static_cast<int>(dim)) {
        train_kmeans_partitions();
        build_local_partition();
        build_upper_hnsw();
        build_local_hnsw();
        if (local_hnsw && local_hnsw_layout == "rcm") {
            reordered_moved_nodes = local_hnsw->reorderIndexRCM();
        } else if (local_hnsw && local_hnsw_layout == "gorder") {
            reordered_moved_nodes = local_hnsw->reorderIndexGorder(gorder_window);
        }
    }

    size_t local_size() const {
        return local_ids.size();
    }

    size_t reordered_moved_count() const {
        return reordered_moved_nodes;
    }

    void start_local_edge_profiling() {
        if (local_hnsw) {
            local_hnsw->startEdgeProfiling();
            local_edge_profiling = true;
        }
    }

    uint64_t local_profiled_edge_traversals() const {
        return local_hnsw
            ? local_hnsw->getProfiledEdgeTraversals()
            : 0;
    }

    size_t finish_local_porder(size_t window) {
        local_edge_profiling = false;
        reordered_moved_nodes = local_hnsw
            ? local_hnsw->reorderIndexPorder(window)
            : 0;
        return reordered_moved_nodes;
    }

    void set_lower_ef_search(int ef) {
        lower_ef_search = ef;

        if (local_hnsw) {
            local_hnsw->setEf(lower_ef_search);
        }
    }

    void set_upper_ef_search(int ef) {
        upper_ef_search = ef;

        if (upper_hnsw) {
            upper_hnsw->setEf(upper_ef_search);
        }
    }

    void select_partitions(
        const float* query,
        size_t route_p,
        std::vector<int>& selected_partitions
    ) const {
        selected_partitions.clear();

        if (!upper_hnsw || world_size <= 0) {
            return;
        }

        route_p = std::max(route_p, static_cast<size_t>(1));
        route_p = std::min(route_p, static_cast<size_t>(world_size));

        auto upper_res = upper_hnsw->searchKnn(
            query,
            route_p
        );

        while (!upper_res.empty()) {
            int pid = static_cast<int>(upper_res.top().second);
            upper_res.pop();

            if (pid >= 0 && pid < world_size) {
                selected_partitions.push_back(pid);
            }
        }

        std::sort(selected_partitions.begin(), selected_partitions.end());
        selected_partitions.erase(
            std::unique(selected_partitions.begin(), selected_partitions.end()),
            selected_partitions.end()
        );
    }

    std::priority_queue<std::pair<float, uint32_t> >
    search_mpi_local(
        const float* query,
        size_t k,
        size_t local_p,
        size_t route_p,
        HNSWOnHNSWLocalProfile* profile = nullptr,
        double* route_us_out = nullptr
    ) const {
        std::priority_queue<std::pair<float, uint32_t> > local_topk;

        if (profile != nullptr) {
            profile->selected_this_rank = 0;
            profile->local_points = local_ids.size();
            profile->returned_candidates = 0;
        }

        double route_begin = MPI_Wtime();

        std::vector<int> selected_partitions;
        select_partitions(
            query,
            route_p,
            selected_partitions
        );

        double route_end = MPI_Wtime();

        if (route_us_out != nullptr) {
            *route_us_out = (route_end - route_begin) * 1000000.0;
        }

        bool selected = false;

        for (size_t i = 0; i < selected_partitions.size(); ++i) {
            if (selected_partitions[i] == rank) {
                selected = true;
                break;
            }
        }

        if (!selected) {
            return local_topk;
        }

        if (profile != nullptr) {
            profile->selected_this_rank = 1;
        }

        if (!local_hnsw || local_ids.empty()) {
            return local_topk;
        }

        local_p = std::max(local_p, k);

        size_t search_k = std::min(local_p, local_ids.size());

        if (search_k == 0) {
            return local_topk;
        }

        auto sub_res = local_edge_profiling
            ? local_hnsw->searchKnnProfiled(query, search_k)
            : local_hnsw->searchKnn(query, search_k);

        while (!sub_res.empty()) {
            float dist = sub_res.top().first;
            uint32_t id = static_cast<uint32_t>(sub_res.top().second);
            sub_res.pop();

            hoh_update_topk_dist(
                dist,
                id,
                local_topk,
                local_p
            );
        }

        if (profile != nullptr) {
            profile->returned_candidates = local_topk.size();
        }

        return local_topk;
    }
};

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_hnsw_on_hnsw_search_one_no_bcast_profile(
    const HNSWOnHNSWMPIIndex& index,
    const float* base,
    float* query_on_this_rank,
    size_t vecdim,
    size_t k,
    size_t local_p,
    size_t route_p,
    int rank,
    int world_size,
    MPI_Comm comm,
    HNSWOnHNSWProfileOne* out_profile
) {
    std::priority_queue<std::pair<float, uint32_t> > final_result;

    double total_begin = MPI_Wtime();

    HNSWOnHNSWLocalProfile local_profile;

    double route_us = 0.0;

    double local_begin = MPI_Wtime();

    auto local_pq = index.search_mpi_local(
        query_on_this_rank,
        k,
        local_p,
        route_p,
        &local_profile,
        &route_us
    );

    double local_end = MPI_Wtime();

    double local_search_with_route_us =
        (local_end - local_begin) * 1000000.0;

    double local_search_us =
        local_search_with_route_us - route_us;

    if (local_search_us < 0.0) {
        local_search_us = 0.0;
    }

    double route_min_us = 0.0;
    double route_max_us = 0.0;
    double route_sum_us = 0.0;

    MPI_Reduce(
        &route_us,
        &route_min_us,
        1,
        MPI_DOUBLE,
        MPI_MIN,
        0,
        comm
    );

    MPI_Reduce(
        &route_us,
        &route_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &route_us,
        &route_sum_us,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        comm
    );

    double local_search_min_us = 0.0;
    double local_search_max_us = 0.0;
    double local_search_sum_us = 0.0;

    MPI_Reduce(
        &local_search_us,
        &local_search_min_us,
        1,
        MPI_DOUBLE,
        MPI_MIN,
        0,
        comm
    );

    MPI_Reduce(
        &local_search_us,
        &local_search_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &local_search_us,
        &local_search_sum_us,
        1,
        MPI_DOUBLE,
        MPI_SUM,
        0,
        comm
    );

    unsigned long long selected_local =
        static_cast<unsigned long long>(local_profile.selected_this_rank);

    unsigned long long local_points_local =
        static_cast<unsigned long long>(local_profile.local_points);

    unsigned long long returned_candidates_local =
        static_cast<unsigned long long>(local_profile.returned_candidates);

    unsigned long long selected_sum = 0;
    unsigned long long selected_max = 0;

    unsigned long long local_points_sum = 0;
    unsigned long long local_points_max = 0;

    unsigned long long returned_candidates_sum = 0;
    unsigned long long returned_candidates_max = 0;

    MPI_Reduce(
        &selected_local,
        &selected_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &selected_local,
        &selected_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &local_points_local,
        &local_points_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &local_points_local,
        &local_points_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &returned_candidates_local,
        &returned_candidates_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &returned_candidates_local,
        &returned_candidates_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    std::vector<HNSWOnHNSWCandidate> local_cands;

    double pack_begin = MPI_Wtime();

    hoh_pack_candidates(
        local_pq,
        local_p,
        local_cands
    );

    double pack_end = MPI_Wtime();

    double pack_local_us =
        (pack_end - pack_begin) * 1000000.0;

    double pack_max_us = 0.0;

    MPI_Reduce(
        &pack_local_us,
        &pack_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    std::vector<HNSWOnHNSWCandidate> all_cands;

    if (rank == 0) {
        all_cands.resize(
            static_cast<size_t>(world_size) * local_p
        );
    }

    double gather_begin = MPI_Wtime();

    MPI_Gather(
        local_cands.data(),
        static_cast<int>(local_p * sizeof(HNSWOnHNSWCandidate)),
        MPI_BYTE,
        rank == 0 ? all_cands.data() : NULL,
        static_cast<int>(local_p * sizeof(HNSWOnHNSWCandidate)),
        MPI_BYTE,
        0,
        comm
    );

    double gather_end = MPI_Wtime();

    double gather_local_us =
        (gather_end - gather_begin) * 1000000.0;

    double gather_max_us = 0.0;

    MPI_Reduce(
        &gather_local_us,
        &gather_max_us,
        1,
        MPI_DOUBLE,
        MPI_MAX,
        0,
        comm
    );

    double merge_us = 0.0;

    if (rank == 0) {
        double merge_begin = MPI_Wtime();

        final_result = hoh_merge_global_topk_simd_rerank(
            all_cands,
            query_on_this_rank,
            base,
            vecdim,
            k
        );

        double merge_end = MPI_Wtime();

        merge_us = (merge_end - merge_begin) * 1000000.0;
    }

    double total_end = MPI_Wtime();

    if (rank == 0 && out_profile != NULL) {
        out_profile->route_us_min = route_min_us;
        out_profile->route_us_avg =
            route_sum_us / static_cast<double>(world_size);
        out_profile->route_us_max = route_max_us;

        out_profile->local_search_us_min = local_search_min_us;
        out_profile->local_search_us_avg =
            local_search_sum_us / static_cast<double>(world_size);
        out_profile->local_search_us_max = local_search_max_us;

        out_profile->pack_us_max = pack_max_us;
        out_profile->gather_us_max = gather_max_us;
        out_profile->merge_us = merge_us;
        out_profile->total_us =
            (total_end - total_begin) * 1000000.0;

        out_profile->selected_ranks_sum = selected_sum;
        out_profile->selected_ranks_max = selected_max;

        out_profile->local_points_sum = local_points_sum;
        out_profile->local_points_max = local_points_max;

        out_profile->returned_candidates_sum = returned_candidates_sum;
        out_profile->returned_candidates_max = returned_candidates_max;
    }

    return final_result;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_hnsw_on_hnsw_search_one_no_bcast_light(
    const HNSWOnHNSWMPIIndex& index,
    const float* base,
    float* query_on_this_rank,
    size_t vecdim,
    size_t k,
    size_t local_p,
    size_t route_p,
    int rank,
    int world_size,
    MPI_Comm comm,
    double* latency_us_out
) {
    std::priority_queue<std::pair<float, uint32_t> > final_result;

    if (latency_us_out != NULL) {
        *latency_us_out = 0.0;
    }

    local_p = std::max(local_p, k);

    double total_begin = MPI_Wtime();

    // 1. 上层 HNSW 选择需要搜索的分区；
    //    如果当前 rank 被选中，则搜索本地 HNSW；
    //    如果未被选中，则返回空 local_pq。
    auto local_pq = index.search_mpi_local(
        query_on_this_rank,
        k,
        local_p,
        route_p,
        NULL,
        NULL
    );

    // 2. 打包 local Top-p 为 Candidate{dist, id}
    std::vector<HNSWOnHNSWCandidate> local_cands;

    hoh_pack_candidates(
        local_pq,
        local_p,
        local_cands
    );

    // 3. 所有 rank 参与一次 MPI_Gather
    //    未被 route 选中的 rank 也会返回空候选，占位为 UINT32_MAX。
    std::vector<HNSWOnHNSWCandidate> all_cands;

    if (rank == 0) {
        all_cands.resize(
            static_cast<size_t>(world_size) * local_p
        );
    }

    MPI_Gather(
        local_cands.data(),
        static_cast<int>(local_p * sizeof(HNSWOnHNSWCandidate)),
        MPI_BYTE,
        rank == 0 ? all_cands.data() : NULL,
        static_cast<int>(local_p * sizeof(HNSWOnHNSWCandidate)),
        MPI_BYTE,
        0,
        comm
    );

    // 4. rank 0 对所有候选做 SIMD rerank，并合并 global Top-k
    if (rank == 0) {
        final_result = hoh_merge_global_topk_simd_rerank(
            all_cands,
            query_on_this_rank,
            base,
            vecdim,
            k
        );
    }

    double total_end = MPI_Wtime();

    if (rank == 0 && latency_us_out != NULL) {
        *latency_us_out =
            (total_end - total_begin) * 1000000.0;
    }

    return final_result;
}
