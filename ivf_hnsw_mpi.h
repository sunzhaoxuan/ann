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
#include <random>
#include <set>
#include <utility>
#include <vector>
#include <arm_neon.h>

#include "hnswlib/hnswlib/hnswlib.h"

enum class IVFHNSWMPISplitMode {
    Block,
    Cyclic
};

static inline bool ivf_hnsw_mpi_own_list(
    size_t list_id,
    size_t nlist,
    int rank,
    int world_size,
    IVFHNSWMPISplitMode mode
) {
    if (world_size <= 1) {
        return true;
    }

    if (mode == IVFHNSWMPISplitMode::Cyclic) {
        return static_cast<int>(list_id % static_cast<size_t>(world_size)) == rank;
    }

    size_t base = nlist / static_cast<size_t>(world_size);
    size_t rem = nlist % static_cast<size_t>(world_size);

    size_t begin =
        static_cast<size_t>(rank) * base
        + std::min(static_cast<size_t>(rank), rem);

    size_t len =
        base + (static_cast<size_t>(rank) < rem ? 1 : 0);

    size_t end = begin + len;

    return list_id >= begin && list_id < end;
}

struct IVFHNSWMPIProfile {
    size_t owned_probe_lists;
    size_t searched_points;
    size_t returned_candidates;

    IVFHNSWMPIProfile()
        : owned_probe_lists(0),
          searched_points(0),
          returned_candidates(0) {}
};

struct IVFHNSWMPIResult {
    float recall;
    double latency_us;

    IVFHNSWMPIResult()
        : recall(0.0f),
          latency_us(0.0) {}

    IVFHNSWMPIResult(float r, double l)
        : recall(r),
          latency_us(l) {}
};

struct IVFHNSWCandidate {
    float dist;
    uint32_t id;
};

static inline float l2_distance_scalar(
    const float* a,
    const float* b,
    size_t dim
) {
    float sum = 0.0f;

    for (size_t i = 0; i < dim; ++i) {
        float diff = a[i] - b[i];
        sum += diff * diff;
    }

    return sum;
}

static inline float l2_distance_neon(
    const float* a,
    const float* b,
    size_t dim
) {
    float sum = 0.0f;

#if defined(__ARM_NEON)
    float32x4_t acc = vdupq_n_f32(0.0f);

    size_t i = 0;
    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        acc = vfmaq_f32(acc, diff, diff);
    }

    float tmp[4];
    vst1q_f32(tmp, acc);
    sum = tmp[0] + tmp[1] + tmp[2] + tmp[3];

    for (; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
#else
    for (size_t i = 0; i < dim; ++i) {
        float d = a[i] - b[i];
        sum += d * d;
    }
#endif

    return sum;
}

static inline void update_topk_dist(
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

static inline void pack_pq_to_candidates(
    std::priority_queue<std::pair<float, uint32_t> > pq,
    size_t local_p,
    std::vector<IVFHNSWCandidate>& local_cands
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
merge_global_topk_from_candidates(
    const std::vector<IVFHNSWCandidate>& all_cands,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t> > global_topk;

    for (size_t i = 0; i < all_cands.size(); ++i) {
        if (all_cands[i].id == UINT32_MAX) {
            continue;
        }

        update_topk_dist(
            all_cands[i].dist,
            all_cands[i].id,
            global_topk,
            k
        );
    }

    return global_topk;
}

static inline float compute_recall_at_k_ivfhnsw(
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

class IVF_HNSW_MPI_Index {
private:
    float* base_data;
    size_t base_number;
    size_t vecdim;

    size_t nlist;
    size_t train_size;
    size_t kmeans_iters;

    int hnsw_M;
    int hnsw_ef_construction;
    int hnsw_ef_search;

    int rank;
    int world_size;
    IVFHNSWMPISplitMode split_mode;

    std::vector<float> centroids;
    std::vector<std::vector<uint32_t> > local_list_ids;

    hnswlib::InnerProductSpace space;

    std::vector<std::unique_ptr<hnswlib::HierarchicalNSW<float> > > hnsw_lists;

private:
    const float* base_vec(uint32_t id) const {
        return base_data + static_cast<size_t>(id) * vecdim;
    }

    void init_centroids() {
        centroids.assign(nlist * vecdim, 0.0f);

        size_t real_train_size = std::min(train_size, base_number);

        if (real_train_size == 0) {
            return;
        }

        for (size_t c = 0; c < nlist; ++c) {
            size_t tid = c % real_train_size;
            size_t bid = tid * base_number / real_train_size;

            const float* src = base_data + bid * vecdim;
            float* dst = centroids.data() + c * vecdim;

            std::memcpy(dst, src, vecdim * sizeof(float));
        }
    }

    uint32_t nearest_centroid(const float* x) const {
        uint32_t best = 0;
        float best_dist = std::numeric_limits<float>::infinity();

        for (size_t c = 0; c < nlist; ++c) {
            const float* centroid = centroids.data() + c * vecdim;

            float dist = l2_distance_scalar(x, centroid, vecdim);

            if (dist < best_dist) {
                best_dist = dist;
                best = static_cast<uint32_t>(c);
            }
        }

        return best;
    }

    void train_kmeans() {
        init_centroids();

        size_t real_train_size = std::min(train_size, base_number);

        if (real_train_size == 0) {
            return;
        }

        std::vector<uint32_t> train_ids(real_train_size);

        for (size_t i = 0; i < real_train_size; ++i) {
            train_ids[i] =
                static_cast<uint32_t>(i * base_number / real_train_size);
        }

        std::vector<float> new_centroids(nlist * vecdim);
        std::vector<size_t> counts(nlist);

        for (size_t iter = 0; iter < kmeans_iters; ++iter) {
            std::fill(new_centroids.begin(), new_centroids.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t i = 0; i < real_train_size; ++i) {
                uint32_t id = train_ids[i];
                const float* x = base_vec(id);

                uint32_t cid = nearest_centroid(x);

                float* acc = new_centroids.data()
                    + static_cast<size_t>(cid) * vecdim;

                for (size_t d = 0; d < vecdim; ++d) {
                    acc[d] += x[d];
                }

                counts[cid]++;
            }

            for (size_t c = 0; c < nlist; ++c) {
                float* dst = centroids.data() + c * vecdim;
                float* src = new_centroids.data() + c * vecdim;

                if (counts[c] == 0) {
                    continue;
                }

                float inv = 1.0f / static_cast<float>(counts[c]);

                for (size_t d = 0; d < vecdim; ++d) {
                    dst[d] = src[d] * inv;
                }
            }
        }
    }

    void build_local_lists() {
        local_list_ids.clear();
        local_list_ids.resize(nlist);

        for (size_t i = 0; i < base_number; ++i) {
            const float* x = base_data + i * vecdim;
            uint32_t cid = nearest_centroid(x);

            if (ivf_hnsw_mpi_own_list(
                    cid,
                    nlist,
                    rank,
                    world_size,
                    split_mode
                )) {
                local_list_ids[cid].push_back(static_cast<uint32_t>(i));
            }
        }
    }

    void build_local_hnsw_graphs() {
        hnsw_lists.clear();
        hnsw_lists.resize(nlist);

        for (size_t cid = 0; cid < nlist; ++cid) {
            if (!ivf_hnsw_mpi_own_list(
                    cid,
                    nlist,
                    rank,
                    world_size,
                    split_mode
                )) {
                continue;
            }

            size_t list_size = local_list_ids[cid].size();

            if (list_size == 0) {
                continue;
            }

            hnsw_lists[cid].reset(
                new hnswlib::HierarchicalNSW<float>(
                    &space,
                    list_size,
                    hnsw_M,
                    hnsw_ef_construction
                )
            );

            hnsw_lists[cid]->setEf(hnsw_ef_search);

            for (size_t j = 0; j < list_size; ++j) {
                uint32_t global_id = local_list_ids[cid][j];
                hnsw_lists[cid]->addPoint(
                    base_vec(global_id),
                    static_cast<hnswlib::labeltype>(global_id)
                );
            }
        }
    }

public:
    IVF_HNSW_MPI_Index(
        float* base,
        size_t base_num,
        size_t dim,
        size_t nlist_,
        size_t train_size_,
        size_t kmeans_iters_,
        int hnsw_M_,
        int hnsw_ef_construction_,
        int hnsw_ef_search_,
        int rank_,
        int world_size_,
        IVFHNSWMPISplitMode split_mode_
    )
        : base_data(base),
          base_number(base_num),
          vecdim(dim),
          nlist(nlist_),
          train_size(train_size_),
          kmeans_iters(kmeans_iters_),
          hnsw_M(hnsw_M_),
          hnsw_ef_construction(hnsw_ef_construction_),
          hnsw_ef_search(hnsw_ef_search_),
          rank(rank_),
          world_size(world_size_),
          split_mode(split_mode_),
          space(static_cast<int>(dim)) {
        train_kmeans();
        build_local_lists();
        build_local_hnsw_graphs();
    }

    void select_probe_ids(
        const float* query,
        size_t nprobe,
        std::vector<uint32_t>& probe_ids
    ) const {
        nprobe = std::min(nprobe, nlist);

        std::priority_queue<std::pair<float, uint32_t> > top_centroids;

        for (size_t c = 0; c < nlist; ++c) {
            const float* centroid = centroids.data() + c * vecdim;
            float dist = l2_distance_neon(query, centroid, vecdim);

            if (top_centroids.size() < nprobe) {
                top_centroids.push({dist, static_cast<uint32_t>(c)});
            } else if (dist < top_centroids.top().first) {
                top_centroids.pop();
                top_centroids.push({dist, static_cast<uint32_t>(c)});
            }
        }

        probe_ids.clear();
        probe_ids.reserve(nprobe);

        while (!top_centroids.empty()) {
            probe_ids.push_back(top_centroids.top().second);
            top_centroids.pop();
        }

        std::reverse(probe_ids.begin(), probe_ids.end());
    }

    std::priority_queue<std::pair<float, uint32_t> >
    search_mpi_local(
        const float* query,
        size_t k,
        size_t nprobe,
        size_t local_p,
        IVFHNSWMPIProfile* profile = nullptr
    ) const {
        std::priority_queue<std::pair<float, uint32_t> > rank_topk;

        if (profile != nullptr) {
            profile->owned_probe_lists = 0;
            profile->searched_points = 0;
            profile->returned_candidates = 0;
        }

        if (k == 0 || base_number == 0 || nlist == 0) {
            return rank_topk;
        }

        local_p = std::max(local_p, k);

        std::vector<uint32_t> probe_ids;
        select_probe_ids(query, nprobe, probe_ids);

        for (size_t pp = 0; pp < probe_ids.size(); ++pp) {
            uint32_t cid = probe_ids[pp];

            if (!ivf_hnsw_mpi_own_list(
                    cid,
                    nlist,
                    rank,
                    world_size,
                    split_mode
                )) {
                continue;
            }

            if (cid >= hnsw_lists.size() || !hnsw_lists[cid]) {
                continue;
            }

            size_t list_size = local_list_ids[cid].size();

            if (list_size == 0) {
                continue;
            }

            if (profile != nullptr) {
                profile->owned_probe_lists++;
                profile->searched_points += list_size;
            }

            size_t search_k = std::min(local_p, list_size);

            hnsw_lists[cid]->setEf(hnsw_ef_search);

            auto sub_res = hnsw_lists[cid]->searchKnn(
                query,
                search_k
            );

            while (!sub_res.empty()) {
                float dist = sub_res.top().first;
                uint32_t id = static_cast<uint32_t>(sub_res.top().second);
                sub_res.pop();

                update_topk_dist(
                    dist,
                    id,
                    rank_topk,
                    local_p
                );
            }
        }

        if (profile != nullptr) {
            profile->returned_candidates = rank_topk.size();
        }

        return rank_topk;
    }
};

struct IVFHNSWMPISearchProfileOne {
    double local_search_us_min;
    double local_search_us_avg;
    double local_search_us_max;
    double pack_us_max;
    double gather_us_max;
    double merge_us;
    double total_us;

    unsigned long long owned_lists_sum;
    unsigned long long owned_lists_max;
    unsigned long long searched_points_sum;
    unsigned long long searched_points_max;
    unsigned long long returned_candidates_sum;
    unsigned long long returned_candidates_max;

    IVFHNSWMPISearchProfileOne()
        : local_search_us_min(0.0),
          local_search_us_avg(0.0),
          local_search_us_max(0.0),
          pack_us_max(0.0),
          gather_us_max(0.0),
          merge_us(0.0),
          total_us(0.0),
          owned_lists_sum(0),
          owned_lists_max(0),
          searched_points_sum(0),
          searched_points_max(0),
          returned_candidates_sum(0),
          returned_candidates_max(0) {}
};

struct IVFHNSWMPISearchProfileTotal {
    int cnt;

    double local_search_us_min;
    double local_search_us_avg;
    double local_search_us_max;
    double pack_us_max;
    double gather_us_max;
    double merge_us;
    double total_us;

    double owned_lists_sum;
    double owned_lists_max;
    double searched_points_sum;
    double searched_points_max;
    double returned_candidates_sum;
    double returned_candidates_max;

    IVFHNSWMPISearchProfileTotal()
        : cnt(0),
          local_search_us_min(0.0),
          local_search_us_avg(0.0),
          local_search_us_max(0.0),
          pack_us_max(0.0),
          gather_us_max(0.0),
          merge_us(0.0),
          total_us(0.0),
          owned_lists_sum(0.0),
          owned_lists_max(0.0),
          searched_points_sum(0.0),
          searched_points_max(0.0),
          returned_candidates_sum(0.0),
          returned_candidates_max(0.0) {}

    void add(const IVFHNSWMPISearchProfileOne& p) {
        ++cnt;

        local_search_us_min += p.local_search_us_min;
        local_search_us_avg += p.local_search_us_avg;
        local_search_us_max += p.local_search_us_max;
        pack_us_max += p.pack_us_max;
        gather_us_max += p.gather_us_max;
        merge_us += p.merge_us;
        total_us += p.total_us;

        owned_lists_sum += static_cast<double>(p.owned_lists_sum);
        owned_lists_max += static_cast<double>(p.owned_lists_max);
        searched_points_sum += static_cast<double>(p.searched_points_sum);
        searched_points_max += static_cast<double>(p.searched_points_max);
        returned_candidates_sum += static_cast<double>(p.returned_candidates_sum);
        returned_candidates_max += static_cast<double>(p.returned_candidates_max);
    }

    void print_average() const {
        if (cnt == 0) {
            return;
        }

        double inv = 1.0 / static_cast<double>(cnt);

        std::cout << "\nMPI IVF-HNSW profile average per query (us):\n";
        std::cout << "  Local search min: " << local_search_us_min * inv << "\n";
        std::cout << "  Local search avg: " << local_search_us_avg * inv << "\n";
        std::cout << "  Local search max: " << local_search_us_max * inv << "\n";
        std::cout << "  Pack max        : " << pack_us_max * inv << "\n";
        std::cout << "  Gather max      : " << gather_us_max * inv << "\n";
        std::cout << "  Merge           : " << merge_us * inv << "\n";
        std::cout << "  Total           : " << total_us * inv << "\n";

        std::cout << "\nMPI IVF-HNSW workload average per query:\n";
        std::cout << "  owned lists sum        : " << owned_lists_sum * inv << "\n";
        std::cout << "  owned lists max        : " << owned_lists_max * inv << "\n";
        std::cout << "  searched points sum    : " << searched_points_sum * inv << "\n";
        std::cout << "  searched points max    : " << searched_points_max * inv << "\n";
        std::cout << "  returned candidates sum: " << returned_candidates_sum * inv << "\n";
        std::cout << "  returned candidates max: " << returned_candidates_max * inv << "\n";

        std::cout << "\nMPI IVF-HNSW load balance indicators:\n";

        if (local_search_us_avg > 0.0) {
            std::cout << "  local search max/avg: "
                      << local_search_us_max / local_search_us_avg
                      << "\n";
        }

        if (searched_points_sum > 0.0) {
            std::cout << "  searched points max/sum: "
                      << searched_points_max / searched_points_sum
                      << "\n";
        }
    }
};

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_ivf_hnsw_search_one_no_bcast_profile(
    const IVF_HNSW_MPI_Index& index,
    float* query_on_this_rank,
    size_t k,
    size_t nprobe,
    size_t local_p,
    int rank,
    int world_size,
    MPI_Comm comm,
    IVFHNSWMPISearchProfileOne* out_profile
) {
    std::priority_queue<std::pair<float, uint32_t> > final_result;

    double total_begin = MPI_Wtime();

    IVFHNSWMPIProfile local_work_profile;

    double t0 = MPI_Wtime();

    auto local_pq = index.search_mpi_local(
        query_on_this_rank,
        k,
        nprobe,
        local_p,
        &local_work_profile
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

    unsigned long long owned_lists_local =
        static_cast<unsigned long long>(local_work_profile.owned_probe_lists);

    unsigned long long searched_points_local =
        static_cast<unsigned long long>(local_work_profile.searched_points);

    unsigned long long returned_candidates_local =
        static_cast<unsigned long long>(local_work_profile.returned_candidates);

    unsigned long long owned_lists_sum = 0;
    unsigned long long owned_lists_max = 0;

    unsigned long long searched_points_sum = 0;
    unsigned long long searched_points_max = 0;

    unsigned long long returned_candidates_sum = 0;
    unsigned long long returned_candidates_max = 0;

    MPI_Reduce(
        &owned_lists_local,
        &owned_lists_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &owned_lists_local,
        &owned_lists_max,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_MAX,
        0,
        comm
    );

    MPI_Reduce(
        &searched_points_local,
        &searched_points_sum,
        1,
        MPI_UNSIGNED_LONG_LONG,
        MPI_SUM,
        0,
        comm
    );

    MPI_Reduce(
        &searched_points_local,
        &searched_points_max,
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

    std::vector<IVFHNSWCandidate> local_cands;

    t0 = MPI_Wtime();

    pack_pq_to_candidates(
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

    std::vector<IVFHNSWCandidate> all_cands;

    if (rank == 0) {
        all_cands.resize(static_cast<size_t>(world_size) * local_p);
    }

    t0 = MPI_Wtime();

    MPI_Gather(
        local_cands.data(),
        static_cast<int>(local_p * sizeof(IVFHNSWCandidate)),
        MPI_BYTE,
        rank == 0 ? all_cands.data() : NULL,
        static_cast<int>(local_p * sizeof(IVFHNSWCandidate)),
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

        final_result = merge_global_topk_from_candidates(
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

        out_profile->owned_lists_sum = owned_lists_sum;
        out_profile->owned_lists_max = owned_lists_max;

        out_profile->searched_points_sum = searched_points_sum;
        out_profile->searched_points_max = searched_points_max;

        out_profile->returned_candidates_sum = returned_candidates_sum;
        out_profile->returned_candidates_max = returned_candidates_max;
    }

    return final_result;
}

static inline std::priority_queue<std::pair<float, uint32_t> >
mpi_ivf_hnsw_search_one_no_bcast_light(
    const IVF_HNSW_MPI_Index& index,
    float* query_on_this_rank,
    size_t k,
    size_t nprobe,
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

    // 1. 本地 IVF+HNSW 搜索
    // 不再 profiling，因此 profile 参数传 NULL
    auto local_pq = index.search_mpi_local(
        query_on_this_rank,
        k,
        nprobe,
        local_p,
        NULL
    );

    // 2. 打包 local Top-p 为 Candidate{dist, id}
    std::vector<IVFHNSWCandidate> local_cands;

    pack_pq_to_candidates(
        local_pq,
        local_p,
        local_cands
    );

    // 3. 使用一次 MPI_Gather 收集所有进程的 local Top-p
    std::vector<IVFHNSWCandidate> all_cands;

    if (rank == 0) {
        all_cands.resize(
            static_cast<size_t>(world_size) * local_p
        );
    }

    MPI_Gather(
        local_cands.data(),
        static_cast<int>(local_p * sizeof(IVFHNSWCandidate)),
        MPI_BYTE,
        rank == 0 ? all_cands.data() : NULL,
        static_cast<int>(local_p * sizeof(IVFHNSWCandidate)),
        MPI_BYTE,
        0,
        comm
    );

    // 4. rank 0 合并得到 global Top-k
    if (rank == 0) {
        final_result = merge_global_topk_from_candidates(
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