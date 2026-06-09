#pragma once

#include <mpi.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <queue>
#include <random>
#include <set>
#include <utility>
#include <vector>

#include "hnswlib/hnswlib/hnswlib.h"

struct RandomHNSWCandidate {
    float dist;
    uint32_t id;
};

struct RandomHNSWLocalProfile {
    size_t local_points;
    size_t returned_candidates;

    RandomHNSWLocalProfile()
        : local_points(0),
          returned_candidates(0) {}
};

struct RandomHNSWProfileOne {
    double local_search_us_min;
    double local_search_us_avg;
    double local_search_us_max;
    double pack_us_max;
    double gather_us_max;
    double merge_us;
    double total_us;

    unsigned long long local_points_sum;
    unsigned long long local_points_max;
    unsigned long long returned_candidates_sum;
    unsigned long long returned_candidates_max;

    RandomHNSWProfileOne()
        : local_search_us_min(0.0),
          local_search_us_avg(0.0),
          local_search_us_max(0.0),
          pack_us_max(0.0),
          gather_us_max(0.0),
          merge_us(0.0),
          total_us(0.0),
          local_points_sum(0),
          local_points_max(0),
          returned_candidates_sum(0),
          returned_candidates_max(0) {}
};

struct RandomHNSWProfileTotal {
    int cnt;

    double local_search_us_min;
    double local_search_us_avg;
    double local_search_us_max;
    double pack_us_max;
    double gather_us_max;
    double merge_us;
    double total_us;

    double local_points_sum;
    double local_points_max;
    double returned_candidates_sum;
    double returned_candidates_max;

    RandomHNSWProfileTotal()
        : cnt(0),
          local_search_us_min(0.0),
          local_search_us_avg(0.0),
          local_search_us_max(0.0),
          pack_us_max(0.0),
          gather_us_max(0.0),
          merge_us(0.0),
          total_us(0.0),
          local_points_sum(0.0),
          local_points_max(0.0),
          returned_candidates_sum(0.0),
          returned_candidates_max(0.0) {}

    void add(const RandomHNSWProfileOne& p) {
        ++cnt;

        local_search_us_min += p.local_search_us_min;
        local_search_us_avg += p.local_search_us_avg;
        local_search_us_max += p.local_search_us_max;
        pack_us_max += p.pack_us_max;
        gather_us_max += p.gather_us_max;
        merge_us += p.merge_us;
        total_us += p.total_us;

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

        std::cout << "\nMPI Random-HNSW profile average per query (us):\n";
        std::cout << "  Local search min: " << local_search_us_min * inv << "\n";
        std::cout << "  Local search avg: " << local_search_us_avg * inv << "\n";
        std::cout << "  Local search max: " << local_search_us_max * inv << "\n";
        std::cout << "  Pack max        : " << pack_us_max * inv << "\n";
        std::cout << "  Gather max      : " << gather_us_max * inv << "\n";
        std::cout << "  Merge           : " << merge_us * inv << "\n";
        std::cout << "  Total           : " << total_us * inv << "\n";

        std::cout << "\nMPI Random-HNSW workload average per query:\n";
        std::cout << "  local points sum        : " << local_points_sum * inv << "\n";
        std::cout << "  local points max        : " << local_points_max * inv << "\n";
        std::cout << "  returned candidates sum : " << returned_candidates_sum * inv << "\n";
        std::cout << "  returned candidates max : " << returned_candidates_max * inv << "\n";

        std::cout << "\nMPI Random-HNSW load balance indicators:\n";
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

static inline std::vector<uint32_t> build_random_partition_ids(
    size_t base_number,
    int rank,
    int world_size,
    uint32_t seed
) {
    std::vector<uint32_t> ids(base_number);

    for (size_t i = 0; i < base_number; ++i) {
        ids[i] = static_cast<uint32_t>(i);
    }

    std::mt19937 rng(seed);
    std::shuffle(ids.begin(), ids.end(), rng);

    size_t base = base_number / static_cast<size_t>(world_size);
    size_t rem = base_number % static_cast<size_t>(world_size);

    size_t begin =
        static_cast<size_t>(rank) * base
        + std::min(static_cast<size_t>(rank), rem);

    size_t len =
        base + (static_cast<size_t>(rank) < rem ? 1 : 0);

    size_t end = begin + len;

    std::vector<uint32_t> local_ids;
    local_ids.reserve(len);

    for (size_t i = begin; i < end; ++i) {
        local_ids.push_back(ids[i]);
    }

    return local_ids;
}

static inline void update_random_hnsw_topk(
    float dist,
    uint32_t id,
    std::priority_queue<std::pair<float, uint32_t> >& topk,
    size_t k
) {
    if (k == 0) {
        return;
    }

    if (topk.size() < k) {
        topk.push({dist, id});
    } else if (dist < topk.top().first) {
        topk.pop();
        topk.push({dist, id});
    }
}

static inline void pack_random_hnsw_candidates(
    std::priority_queue<std::pair<float, uint32_t> > pq,
    size_t local_p,
    std::vector<RandomHNSWCandidate>& local_cands
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

static inline std::priority_queue<std::pair<float, uint32_t> >
merge_random_hnsw_global_topk(
    const std::vector<RandomHNSWCandidate>& all_cands,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t> > global_topk;

    for (size_t i = 0; i < all_cands.size(); ++i) {
        if (all_cands[i].id == UINT32_MAX) {
            continue;
        }

        update_random_hnsw_topk(
            all_cands[i].dist,
            all_cands[i].id,
            global_topk,
            k
        );
    }

    return global_topk;
}

static inline float compute_recall_at_k_random_hnsw(
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

class RandomHNSWMPIIndex {
private:
    float* base_data;
    size_t base_number;
    size_t vecdim;

    int rank;
    int world_size;

    int hnsw_M;
    int hnsw_ef_construction;
    int hnsw_ef_search;

    uint32_t split_seed;

    std::vector<uint32_t> local_ids;

    hnswlib::InnerProductSpace space;
    std::unique_ptr<hnswlib::HierarchicalNSW<float> > hnsw_index;

private:
    const float* base_vec(uint32_t id) const {
        return base_data + static_cast<size_t>(id) * vecdim;
    }

public:
    RandomHNSWMPIIndex(
        float* base,
        size_t base_num,
        size_t dim,
        int rank_,
        int world_size_,
        int hnsw_M_,
        int hnsw_ef_construction_,
        int hnsw_ef_search_,
        uint32_t split_seed_ = 2026
    )
        : base_data(base),
          base_number(base_num),
          vecdim(dim),
          rank(rank_),
          world_size(world_size_),
          hnsw_M(hnsw_M_),
          hnsw_ef_construction(hnsw_ef_construction_),
          hnsw_ef_search(hnsw_ef_search_),
          split_seed(split_seed_),
          space(static_cast<int>(dim)) {
        build();
    }

    void build() {
        local_ids = build_random_partition_ids(
            base_number,
            rank,
            world_size,
            split_seed
        );

        if (local_ids.empty()) {
            return;
        }

        hnsw_index.reset(
            new hnswlib::HierarchicalNSW<float>(
                &space,
                local_ids.size(),
                hnsw_M,
                hnsw_ef_construction
            )
        );

        hnsw_index->setEf(hnsw_ef_search);

        for (size_t i = 0; i < local_ids.size(); ++i) {
            uint32_t gid = local_ids[i];

            hnsw_index->addPoint(
                base_vec(gid),
                static_cast<hnswlib::labeltype>(gid)
            );
        }
    }

    size_t local_size() const {
        return local_ids.size();
    }

    void set_ef_search(int ef) {
        hnsw_ef_search = ef;

        if (hnsw_index) {
            hnsw_index->setEf(hnsw_ef_search);
        }
    }

    std::priority_queue<std::pair<float, uint32_t> >
    search_mpi_local(
        const float* query,
        size_t k,
        size_t local_p,
        RandomHNSWLocalProfile* profile = nullptr
    ) const {
        std::priority_queue<std::pair<float, uint32_t> > local_topk;

        if (profile != nullptr) {
            profile->local_points = local_ids.size();
            profile->returned_candidates = 0;
        }

        if (!hnsw_index || local_ids.empty()) {
            return local_topk;
        }

        local_p = std::max(local_p, k);
        size_t search_k = std::min(local_p, local_ids.size());

        if (search_k == 0) {
            return local_topk;
        }

        auto sub_res = hnsw_index->searchKnn(
            query,
            search_k
        );

        while (!sub_res.empty()) {
            float dist = sub_res.top().first;
            uint32_t id = static_cast<uint32_t>(sub_res.top().second);
            sub_res.pop();

            update_random_hnsw_topk(
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
mpi_random_hnsw_search_one_no_bcast_profile(
    const RandomHNSWMPIIndex& index,
    float* query_on_this_rank,
    size_t k,
    size_t local_p,
    int rank,
    int world_size,
    MPI_Comm comm,
    RandomHNSWProfileOne* out_profile
) {
    std::priority_queue<std::pair<float, uint32_t> > final_result;

    double total_begin = MPI_Wtime();

    RandomHNSWLocalProfile local_profile;

    double t0 = MPI_Wtime();

    auto local_pq = index.search_mpi_local(
        query_on_this_rank,
        k,
        local_p,
        &local_profile
    );

    double t1 = MPI_Wtime();

    double local_search_us = (t1 - t0) * 1000000.0;

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

    unsigned long long local_points_local =
        static_cast<unsigned long long>(local_profile.local_points);

    unsigned long long returned_candidates_local =
        static_cast<unsigned long long>(local_profile.returned_candidates);

    unsigned long long local_points_sum = 0;
    unsigned long long local_points_max = 0;
    unsigned long long returned_candidates_sum = 0;
    unsigned long long returned_candidates_max = 0;

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

    std::vector<RandomHNSWCandidate> local_cands;

    t0 = MPI_Wtime();

    pack_random_hnsw_candidates(
        local_pq,
        local_p,
        local_cands
    );

    t1 = MPI_Wtime();

    double pack_local_us = (t1 - t0) * 1000000.0;
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

    std::vector<RandomHNSWCandidate> all_cands;

    if (rank == 0) {
        all_cands.resize(
            static_cast<size_t>(world_size) * local_p
        );
    }

    t0 = MPI_Wtime();

    MPI_Gather(
        local_cands.data(),
        static_cast<int>(local_p * sizeof(RandomHNSWCandidate)),
        MPI_BYTE,
        rank == 0 ? all_cands.data() : NULL,
        static_cast<int>(local_p * sizeof(RandomHNSWCandidate)),
        MPI_BYTE,
        0,
        comm
    );

    t1 = MPI_Wtime();

    double gather_local_us = (t1 - t0) * 1000000.0;
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

        final_result = merge_random_hnsw_global_topk(
            all_cands,
            k
        );

        double merge_end = MPI_Wtime();
        merge_us = (merge_end - merge_begin) * 1000000.0;
    }

    double total_end = MPI_Wtime();

    if (rank == 0 && out_profile != NULL) {
        out_profile->local_search_us_min = local_search_min_us;
        out_profile->local_search_us_avg =
            local_search_sum_us / static_cast<double>(world_size);
        out_profile->local_search_us_max = local_search_max_us;

        out_profile->pack_us_max = pack_max_us;
        out_profile->gather_us_max = gather_max_us;
        out_profile->merge_us = merge_us;
        out_profile->total_us = (total_end - total_begin) * 1000000.0;

        out_profile->local_points_sum = local_points_sum;
        out_profile->local_points_max = local_points_max;

        out_profile->returned_candidates_sum = returned_candidates_sum;
        out_profile->returned_candidates_max = returned_candidates_max;
    }

    return final_result;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_random_hnsw_search_one_no_bcast_light(
    const RandomHNSWMPIIndex& index,
    float* query_on_this_rank,
    size_t k,
    size_t local_p,
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

    // 1. 每个 rank 在自己的随机划分子集上搜索本地 HNSW
    auto local_pq = index.search_mpi_local(
        query_on_this_rank,
        k,
        local_p,
        NULL
    );

    // 2. 打包 local Top-p 为 Candidate{dist, id}
    std::vector<RandomHNSWCandidate> local_cands;

    pack_random_hnsw_candidates(
        local_pq,
        local_p,
        local_cands
    );

    // 3. rank 0 收集所有 rank 的 local Top-p
    std::vector<RandomHNSWCandidate> all_cands;

    if (rank == 0) {
        all_cands.resize(
            static_cast<size_t>(world_size) * local_p
        );
    }

    MPI_Gather(
        local_cands.data(),
        static_cast<int>(local_p * sizeof(RandomHNSWCandidate)),
        MPI_BYTE,
        rank == 0 ? all_cands.data() : NULL,
        static_cast<int>(local_p * sizeof(RandomHNSWCandidate)),
        MPI_BYTE,
        0,
        comm
    );

    // 4. rank 0 合并所有候选，得到 global Top-k
    if (rank == 0) {
        final_result = merge_random_hnsw_global_topk(
            all_cands,
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