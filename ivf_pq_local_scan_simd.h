#pragma once

#include "simd_compat.h"
#include "opq_transform.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <stdexcept>
#include <utility>
#include <vector>
#include <pthread.h>
#include <omp.h>
#include <atomic>

// ============================================================
// IVF-PQ-SIMD baseline, local PQ version
//
// 实现方式：先构建 IVF，再在每个倒排链表内分别训练 PQ
//
// 离线阶段：
//   1. 训练 IVF coarse centroids
//   2. 将 base id 分配到 inverted lists
//   3. 对每个 inverted list 内部单独训练 PQ codebook
//   4. 对每个 list 内部的向量编码成局部 PQ code
//
// 在线阶段：
//   1. Query 选择最近的 nprobe 个 coarse centroids
//   2. 对每个被访问的 list，使用该 list 自己的 PQ codebook 构建 LUT
//   3. 扫描该 list 内的 PQ code，维护 Top-rerank_p
//   4. 对粗排候选使用原始 float 向量重新精排
//   5. 返回 Top-k
// ============================================================

enum class IVFPQLocalInitMode {
    Uniform
};

enum class IVFPQMPISplitMode {
    Block,
    Cyclic
};

struct IVFPQLocalMPIProfile {
    size_t owned_probe_lists = 0;
    size_t scanned_codes = 0;
    size_t returned_candidates = 0;
};

static inline bool ivfpq_mpi_owns_list(
    size_t list_id,
    size_t nlist,
    int rank,
    int world_size,
    IVFPQMPISplitMode mode
) {
    if (world_size <= 1) {
        return true;
    }

    if (rank < 0 || rank >= world_size) {
        return false;
    }

    if (mode == IVFPQMPISplitMode::Cyclic) {
        return static_cast<int>(list_id % static_cast<size_t>(world_size)) == rank;
    }

    const size_t processes = static_cast<size_t>(world_size);
    const size_t base = nlist / processes;
    const size_t remainder = nlist % processes;
    const size_t rank_size = static_cast<size_t>(rank);
    const size_t begin = rank_size * base + std::min(rank_size, remainder);
    const size_t length = base + (rank_size < remainder ? 1 : 0);
    return list_id >= begin && list_id < begin + length;
}

// ------------------------------------------------------------
// SIMD L2 距离
// ------------------------------------------------------------
static inline float ivfpqlocal_l2_neon(
    const float* __restrict__ a,
    const float* __restrict__ b,
    size_t dim
) {
    size_t i = 0;
    float32x4_t sum = vdupq_n_f32(0.0f);

    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum = vfmaq_f32(sum, diff, diff);
    }

    float result = vaddvq_f32(sum);

    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}

// ------------------------------------------------------------
// SIMD 内积
// ------------------------------------------------------------
static inline float ivfpqlocal_inner_product_neon_fma(
    const float* __restrict__ a,
    const float* __restrict__ b,
    size_t dim
) {
    size_t i = 0;
    float32x4_t sum = vdupq_n_f32(0.0f);

    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        sum = vfmaq_f32(sum, va, vb);
    }

    float result = vaddvq_f32(sum);

    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

// ------------------------------------------------------------
// Top-k：score 越大越好
// ------------------------------------------------------------
template <typename ScoreT>
static inline void ivfpqlocal_recompute_worst_score(
    const ScoreT* score,
    size_t cnt,
    size_t& worst_pos,
    ScoreT& worst_score
) {
    worst_pos = 0;
    worst_score = score[0];

    for (size_t i = 1; i < cnt; ++i) {
        if (score[i] < worst_score) {
            worst_score = score[i];
            worst_pos = i;
        }
    }
}

template <typename ScoreT>
static inline void ivfpqlocal_update_top_score(
    ScoreT score,
    uint32_t id,
    ScoreT* best_score,
    uint32_t* best_id,
    size_t capacity,
    size_t& cnt,
    size_t& worst_pos,
    ScoreT& worst_score
) {
    if (capacity == 0) {
        return;
    }

    if (cnt < capacity) {
        best_score[cnt] = score;
        best_id[cnt] = id;
        ++cnt;

        if (cnt == capacity) {
            ivfpqlocal_recompute_worst_score(
                best_score,
                cnt,
                worst_pos,
                worst_score
            );
        }

        return;
    }

    if (score > worst_score) {
        best_score[worst_pos] = score;
        best_id[worst_pos] = id;

        ivfpqlocal_recompute_worst_score(
            best_score,
            cnt,
            worst_pos,
            worst_score
        );
    }
}

// ------------------------------------------------------------
// Top-nprobe：dist 越小越好
// ------------------------------------------------------------
static inline void ivfpqlocal_recompute_worst_dist(
    const float* dist,
    size_t cnt,
    size_t& worst_pos,
    float& worst_dist
) {
    worst_pos = 0;
    worst_dist = dist[0];

    for (size_t i = 1; i < cnt; ++i) {
        if (dist[i] > worst_dist) {
            worst_dist = dist[i];
            worst_pos = i;
        }
    }
}

static inline void ivfpqlocal_update_top_probe(
    float dist,
    uint32_t cid,
    float* best_dist,
    uint32_t* best_id,
    size_t nprobe,
    size_t& cnt,
    size_t& worst_pos,
    float& worst_dist
) {
    if (nprobe == 0) {
        return;
    }

    if (cnt < nprobe) {
        best_dist[cnt] = dist;
        best_id[cnt] = cid;
        ++cnt;

        if (cnt == nprobe) {
            ivfpqlocal_recompute_worst_dist(
                best_dist,
                cnt,
                worst_pos,
                worst_dist
            );
        }

        return;
    }

    if (dist < worst_dist) {
        best_dist[worst_pos] = dist;
        best_id[worst_pos] = cid;

        ivfpqlocal_recompute_worst_dist(
            best_dist,
            cnt,
            worst_pos,
            worst_dist
        );
    }
}

class IVFPQLocalIndexSIMD;

struct IVFPQLocalListThreadParam {
    int tid;

    const IVFPQLocalIndexSIMD* index;

    const std::vector<uint32_t>* probe_ids;
    std::atomic<size_t>* next_probe;

    float* query;

    size_t local_p;

    float* local_score;
    uint32_t* local_id;
    size_t* local_cnt;
};

static void* ivfpq_local_list_parallel_worker(void* arg);

class IVFPQLocalIndexSIMD {
private:
    struct LocalPQList {
        std::vector<uint32_t> ids;      // 该倒排链表中的 base id
        std::vector<float> codebooks;   // M * Ks * subdim
        std::vector<uint8_t> codes;     // ids.size() * M
    };

public:
    IVFPQLocalIndexSIMD(
        float* base,
        size_t base_number,
        size_t vecdim,
        size_t nlist = 100,
        size_t M = 16,
        size_t Ks = 256,
        size_t train_size = 10000,
        size_t kmeans_iters = 6,
        IVFPQLocalInitMode init_mode = IVFPQLocalInitMode::Uniform,
        size_t opq_iters = 0
    )
        : base_float(base),
          base_number(base_number),
          vecdim(vecdim),
          nlist(nlist),
          M(M),
          Ks(Ks),
          train_size(train_size),
          kmeans_iters(kmeans_iters),
          init_mode(init_mode),
          opq_iters(opq_iters)
    {
        normalize_params();

        coarse_centroids.resize(this->nlist * this->vecdim);
        lists.resize(this->nlist);

        build();
    }

    std::priority_queue<std::pair<float, uint32_t> > search(
        float* query,
        size_t k,
        size_t nprobe,
        size_t rerank_p
    ) const {
        std::priority_queue<std::pair<float, uint32_t> > result;

        if (k == 0 || base_number == 0 || nlist == 0) {
            return result;
        }

        k = std::min(k, base_number);

        nprobe = std::max(nprobe, static_cast<size_t>(1));
        nprobe = std::min(nprobe, nlist);

        rerank_p = std::max(rerank_p, k);
        rerank_p = std::min(rerank_p, base_number);

        // 1. 选择最近的 nprobe 个 IVF coarse centroids
        std::vector<uint32_t> probe_ids;
        select_probe_ids(
            query,
            nprobe,
            probe_ids
        );

        if (probe_ids.empty()) {
            return result;
        }

        std::vector<float> rotated_query;
        const float* pq_query = prepare_pq_query(query, rotated_query);

        // 2. 在选中 lists 内扫描局部 PQ code，维护 Top-rerank_p
        std::vector<float> cand_score(rerank_p);
        std::vector<uint32_t> cand_id(rerank_p);

        size_t cand_cnt = 0;
        size_t cand_worst_pos = 0;
        float cand_worst_score = 0.0f;

        std::vector<float> lut(M * Ks);

        for (size_t pp = 0; pp < probe_ids.size(); ++pp) {
            uint32_t cid = probe_ids[pp];
            const LocalPQList& list = lists[cid];

            if (list.ids.empty()) {
                continue;
            }

            // 注意：每个 list 有自己的 PQ codebook，
            // 因此每访问一个 list 都需要重新构建一次 LUT
            build_local_pq_lut(
                pq_query,
                list,
                lut.data()
            );

            for (size_t j = 0; j < list.ids.size(); ++j) {
                const uint8_t* code =
                    list.codes.data() + j * M;

                float approx_score = scan_local_pq_code(
                    code,
                    lut.data()
                );

                uint32_t id = list.ids[j];

                ivfpqlocal_update_top_score(
                    approx_score,
                    id,
                    cand_score.data(),
                    cand_id.data(),
                    rerank_p,
                    cand_cnt,
                    cand_worst_pos,
                    cand_worst_score
                );
            }
        }

        // 3. 对粗排候选进行原始 float 精排
        std::vector<float> best_score(k);
        std::vector<uint32_t> best_id(k);

        size_t best_cnt = 0;
        size_t best_worst_pos = 0;
        float best_worst_score = 0.0f;

        constexpr size_t PREFETCH_DIST = 4;

        for (size_t i = 0; i < cand_cnt; ++i) {
            uint32_t id = cand_id[i];

            if (i + PREFETCH_DIST < cand_cnt) {
                uint32_t next_id = cand_id[i + PREFETCH_DIST];

                __builtin_prefetch(
                    base_float + static_cast<size_t>(next_id) * vecdim,
                    0,
                    1
                );
            }

            const float* base_vec =
                base_float + static_cast<size_t>(id) * vecdim;

            float score = ivfpqlocal_inner_product_neon_fma(
                base_vec,
                query,
                vecdim
            );

            ivfpqlocal_update_top_score(
                score,
                id,
                best_score.data(),
                best_id.data(),
                k,
                best_cnt,
                best_worst_pos,
                best_worst_score
            );
        }

        // 4. 返回原评测接口格式：priority_queue<pair<dis, id>>
        for (size_t i = 0; i < best_cnt; ++i) {
            float dis = 1.0f - best_score[i];
            result.push({dis, best_id[i]});
        }

        return result;
    }

    // Release list-local PQ data that belongs to other MPI ranks. Coarse
    // centroids are intentionally retained because every rank performs the
    // same routing step for a query.
    void keep_only_mpi_local_lists(
        int rank,
        int world_size,
        IVFPQMPISplitMode split_mode
    ) {
        for (size_t cid = 0; cid < lists.size(); ++cid) {
            if (ivfpq_mpi_owns_list(cid, nlist, rank, world_size, split_mode)) {
                continue;
            }

            std::vector<uint32_t>().swap(lists[cid].ids);
            std::vector<float>().swap(lists[cid].codebooks);
            std::vector<uint8_t>().swap(lists[cid].codes);
        }
    }

    std::priority_queue<std::pair<float, uint32_t> > search_mpi_local(
        float* query,
        size_t k,
        size_t nprobe,
        size_t rerank_p,
        size_t local_p,
        int rank,
        int world_size,
        IVFPQMPISplitMode split_mode,
        IVFPQLocalMPIProfile* profile = nullptr
    ) const {
        std::priority_queue<std::pair<float, uint32_t> > result;

        if (profile != nullptr) {
            *profile = IVFPQLocalMPIProfile();
        }

        if (k == 0 || base_number == 0 || nlist == 0 || world_size <= 0) {
            return result;
        }

        k = std::min(k, base_number);
        nprobe = std::max<size_t>(1, std::min(nprobe, nlist));
        rerank_p = std::max(k, std::min(rerank_p, base_number));
        local_p = std::max(k, std::min(local_p, base_number));

        std::vector<uint32_t> probe_ids;
        select_probe_ids(query, nprobe, probe_ids);

        std::vector<float> rotated_query;
        const float* pq_query = prepare_pq_query(query, rotated_query);

        std::vector<float> candidate_scores(rerank_p);
        std::vector<uint32_t> candidate_ids(rerank_p);
        size_t candidate_count = 0;
        size_t candidate_worst_position = 0;
        float candidate_worst_score = 0.0f;
        std::vector<float> lut(M * Ks);

        for (uint32_t cid : probe_ids) {
            if (!ivfpq_mpi_owns_list(cid, nlist, rank, world_size, split_mode)) {
                continue;
            }

            const LocalPQList& list = lists[cid];
            if (list.ids.empty()) {
                continue;
            }

            if (profile != nullptr) {
                ++profile->owned_probe_lists;
                profile->scanned_codes += list.ids.size();
            }

            build_local_pq_lut(pq_query, list, lut.data());

            for (size_t j = 0; j < list.ids.size(); ++j) {
                const float score = scan_local_pq_code(
                    list.codes.data() + j * M,
                    lut.data()
                );

                ivfpqlocal_update_top_score(
                    score,
                    list.ids[j],
                    candidate_scores.data(),
                    candidate_ids.data(),
                    rerank_p,
                    candidate_count,
                    candidate_worst_position,
                    candidate_worst_score
                );
            }
        }

        std::vector<float> best_scores(local_p);
        std::vector<uint32_t> best_ids(local_p);
        size_t best_count = 0;
        size_t best_worst_position = 0;
        float best_worst_score = 0.0f;

        for (size_t i = 0; i < candidate_count; ++i) {
            const uint32_t id = candidate_ids[i];
            const float score = ivfpqlocal_inner_product_neon_fma(
                base_float + static_cast<size_t>(id) * vecdim,
                query,
                vecdim
            );

            ivfpqlocal_update_top_score(
                score,
                id,
                best_scores.data(),
                best_ids.data(),
                local_p,
                best_count,
                best_worst_position,
                best_worst_score
            );
        }

        for (size_t i = 0; i < best_count; ++i) {
            result.push({1.0f - best_scores[i], best_ids[i]});
        }

        if (profile != nullptr) {
            profile->returned_candidates = best_count;
        }

        return result;
    }

    std::priority_queue<std::pair<float, uint32_t> >
    search_mpi_local_omp_list_parallel(
        float* query,
        size_t k,
        size_t nprobe,
        size_t rerank_p,
        size_t local_p,
        int rank,
        int world_size,
        IVFPQMPISplitMode split_mode,
        int num_threads,
        IVFPQLocalMPIProfile* profile = nullptr
    ) const {
        if (num_threads <= 1) {
            return search_mpi_local(
                query,
                k,
                nprobe,
                rerank_p,
                local_p,
                rank,
                world_size,
                split_mode,
                profile
            );
        }

        std::priority_queue<std::pair<float, uint32_t> > result;

        if (profile != nullptr) {
            *profile = IVFPQLocalMPIProfile();
        }

        if (k == 0 || base_number == 0 || nlist == 0 || world_size <= 0) {
            return result;
        }

        k = std::min(k, base_number);
        nprobe = std::max<size_t>(1, std::min(nprobe, nlist));
        rerank_p = std::max(k, std::min(rerank_p, base_number));
        local_p = std::max(k, std::min(local_p, base_number));

        std::vector<uint32_t> routed_ids;
        select_probe_ids(query, nprobe, routed_ids);

        std::vector<float> rotated_query;
        const float* pq_query = prepare_pq_query(query, rotated_query);

        std::vector<uint32_t> owned_ids;
        size_t scanned_codes = 0;
        for (uint32_t cid : routed_ids) {
            if (ivfpq_mpi_owns_list(cid, nlist, rank, world_size, split_mode)
                && !lists[cid].ids.empty()) {
                owned_ids.push_back(cid);
                scanned_codes += lists[cid].ids.size();
            }
        }

        if (profile != nullptr) {
            profile->owned_probe_lists = owned_ids.size();
            profile->scanned_codes = scanned_codes;
        }

        if (owned_ids.empty()) {
            return result;
        }

        num_threads = std::min<int>(num_threads, static_cast<int>(owned_ids.size()));
        const size_t thread_capacity = local_p;
        std::vector<float> thread_scores(static_cast<size_t>(num_threads) * thread_capacity);
        std::vector<uint32_t> thread_ids(static_cast<size_t>(num_threads) * thread_capacity);
        std::vector<size_t> thread_counts(static_cast<size_t>(num_threads), 0);

        omp_set_dynamic(0);
        #pragma omp parallel num_threads(num_threads)
        {
            const int tid = omp_get_thread_num();
            float* scores = thread_scores.data() + static_cast<size_t>(tid) * thread_capacity;
            uint32_t* ids = thread_ids.data() + static_cast<size_t>(tid) * thread_capacity;
            size_t count = 0;
            size_t worst_position = 0;
            float worst_score = 0.0f;
            std::vector<float> lut(M * Ks);

            #pragma omp for schedule(dynamic, 1)
            for (long long pos = 0; pos < static_cast<long long>(owned_ids.size()); ++pos) {
                const LocalPQList& list = lists[owned_ids[static_cast<size_t>(pos)]];
                build_local_pq_lut(pq_query, list, lut.data());

                for (size_t j = 0; j < list.ids.size(); ++j) {
                    const float score = scan_local_pq_code(
                        list.codes.data() + j * M,
                        lut.data()
                    );
                    ivfpqlocal_update_top_score(
                        score,
                        list.ids[j],
                        scores,
                        ids,
                        thread_capacity,
                        count,
                        worst_position,
                        worst_score
                    );
                }
            }

            thread_counts[static_cast<size_t>(tid)] = count;
        }

        std::vector<float> candidate_scores(rerank_p);
        std::vector<uint32_t> candidate_ids(rerank_p);
        size_t candidate_count = 0;
        size_t candidate_worst_position = 0;
        float candidate_worst_score = 0.0f;

        for (int tid = 0; tid < num_threads; ++tid) {
            const float* scores = thread_scores.data() + static_cast<size_t>(tid) * thread_capacity;
            const uint32_t* ids = thread_ids.data() + static_cast<size_t>(tid) * thread_capacity;
            for (size_t j = 0; j < thread_counts[static_cast<size_t>(tid)]; ++j) {
                ivfpqlocal_update_top_score(
                    scores[j],
                    ids[j],
                    candidate_scores.data(),
                    candidate_ids.data(),
                    rerank_p,
                    candidate_count,
                    candidate_worst_position,
                    candidate_worst_score
                );
            }
        }

        std::vector<float> best_scores(local_p);
        std::vector<uint32_t> best_ids(local_p);
        size_t best_count = 0;
        size_t best_worst_position = 0;
        float best_worst_score = 0.0f;

        for (size_t i = 0; i < candidate_count; ++i) {
            const uint32_t id = candidate_ids[i];
            const float score = ivfpqlocal_inner_product_neon_fma(
                base_float + static_cast<size_t>(id) * vecdim,
                query,
                vecdim
            );
            ivfpqlocal_update_top_score(
                score,
                id,
                best_scores.data(),
                best_ids.data(),
                local_p,
                best_count,
                best_worst_position,
                best_worst_score
            );
        }

        for (size_t i = 0; i < best_count; ++i) {
            result.push({1.0f - best_scores[i], best_ids[i]});
        }

        if (profile != nullptr) {
            profile->returned_candidates = best_count;
        }

        return result;
    }

    std::priority_queue<std::pair<float, uint32_t> > search_omp_list_parallel(
        float* query,
        size_t k,
        size_t nprobe,
        size_t rerank_p,
        size_t local_p,
        int num_threads
    ) const {
        std::priority_queue<std::pair<float, uint32_t> > result;

        if (k == 0 || base_number == 0 || nlist == 0) {
            return result;
        }

        k = std::min(k, base_number);

        nprobe = std::max(nprobe, static_cast<size_t>(1));
        nprobe = std::min(nprobe, nlist);

        rerank_p = std::max(rerank_p, k);
        rerank_p = std::min(rerank_p, base_number);

        // local_p = 0 时默认等于 rerank_p，保证候选不容易被局部阶段丢掉
        if (local_p == 0) {
            local_p = rerank_p;
        }

        local_p = std::min(local_p, base_number);

        if (num_threads <= 1) {
            return search(query, k, nprobe, rerank_p);
        }

        if (static_cast<size_t>(num_threads) > nprobe) {
            num_threads = static_cast<int>(nprobe);
        }

        // 1. 单线程选择最近的 nprobe 个倒排链表
        std::vector<uint32_t> probe_ids;

        select_probe_ids(
            query,
            nprobe,
            probe_ids
        );

        if (probe_ids.empty()) {
            return result;
        }

        std::vector<float> rotated_query;
        const float* pq_query = prepare_pq_query(query, rotated_query);

        // 2. 为每个线程准备 local Top-p
        std::vector<float> local_scores(
            static_cast<size_t>(num_threads) * local_p
        );

        std::vector<uint32_t> local_ids(
            static_cast<size_t>(num_threads) * local_p
        );

        std::vector<size_t> local_cnt(
            static_cast<size_t>(num_threads),
            0
        );

        omp_set_dynamic(0);

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();

            float* score_t =
                local_scores.data() + static_cast<size_t>(tid) * local_p;

            uint32_t* id_t =
                local_ids.data() + static_cast<size_t>(tid) * local_p;

            size_t cnt = 0;
            size_t worst_pos = 0;
            float worst_score = 0.0f;

            // 每个线程维护自己的 LUT，避免反复分配
            std::vector<float> lut(M * Ks);

            // list 长度不均匀，因此优先使用 dynamic,1
            #pragma omp for schedule(dynamic, 1)
            for (long long pp = 0;
                pp < static_cast<long long>(probe_ids.size());
                ++pp) {

                uint32_t cid = probe_ids[static_cast<size_t>(pp)];
                const LocalPQList& list = lists[cid];

                if (list.ids.empty()) {
                    continue;
                }

                // 每个 list 有自己的局部 PQ codebook，
                // 因此处理每个 list 前都要构建该 list 的局部 LUT
                build_local_pq_lut(
                    pq_query,
                    list,
                    lut.data()
                );

                for (size_t j = 0; j < list.ids.size(); ++j) {
                    const uint8_t* code =
                        list.codes.data() + j * M;

                    float approx_score = scan_local_pq_code(
                        code,
                        lut.data()
                    );

                    uint32_t id = list.ids[j];

                    ivfpqlocal_update_top_score(
                        approx_score,
                        id,
                        score_t,
                        id_t,
                        local_p,
                        cnt,
                        worst_pos,
                        worst_score
                    );
                }
            }

            local_cnt[static_cast<size_t>(tid)] = cnt;
        }

        // 3. reduce：local Top-p -> global Top-rerank_p
        std::vector<float> cand_score(rerank_p);
        std::vector<uint32_t> cand_id(rerank_p);

        size_t cand_cnt = 0;
        size_t cand_worst_pos = 0;
        float cand_worst_score = 0.0f;

        for (int t = 0; t < num_threads; ++t) {
            float* score_t =
                local_scores.data() + static_cast<size_t>(t) * local_p;

            uint32_t* id_t =
                local_ids.data() + static_cast<size_t>(t) * local_p;

            for (size_t j = 0; j < local_cnt[static_cast<size_t>(t)]; ++j) {
                ivfpqlocal_update_top_score(
                    score_t[j],
                    id_t[j],
                    cand_score.data(),
                    cand_id.data(),
                    rerank_p,
                    cand_cnt,
                    cand_worst_pos,
                    cand_worst_score
                );
            }
        }

        // 4. 对粗排候选做原始 float rerank
        std::vector<float> best_score(k);
        std::vector<uint32_t> best_id(k);

        size_t best_cnt = 0;
        size_t best_worst_pos = 0;
        float best_worst_score = 0.0f;

        constexpr size_t PREFETCH_DIST = 4;

        for (size_t i = 0; i < cand_cnt; ++i) {
            uint32_t id = cand_id[i];

            if (i + PREFETCH_DIST < cand_cnt) {
                uint32_t next_id = cand_id[i + PREFETCH_DIST];

                __builtin_prefetch(
                    base_float + static_cast<size_t>(next_id) * vecdim,
                    0,
                    1
                );
            }

            const float* base_vec =
                base_float + static_cast<size_t>(id) * vecdim;

            float score = ivfpqlocal_inner_product_neon_fma(
                base_vec,
                query,
                vecdim
            );

            ivfpqlocal_update_top_score(
                score,
                id,
                best_score.data(),
                best_id.data(),
                k,
                best_cnt,
                best_worst_pos,
                best_worst_score
            );
        }

        // 5. 返回原评测接口格式
        for (size_t i = 0; i < best_cnt; ++i) {
            float dis = 1.0f - best_score[i];
            result.push({dis, best_id[i]});
        }

        return result;
    }

    void scan_probe_lists_dynamic(
        const std::vector<uint32_t>& probe_ids,
        std::atomic<size_t>& next_probe,
        float* query,
        size_t local_p,
        float* local_score,
        uint32_t* local_id,
        size_t* out_cnt
    ) const {
        size_t cnt = 0;
        size_t worst_pos = 0;
        float worst_score = 0.0f;

        // 每个线程只分配一次 LUT 缓冲区
        std::vector<float> lut(M * Ks);

        while (true) {
            size_t task = next_probe.fetch_add(1);

            if (task >= probe_ids.size()) {
                break;
            }

            uint32_t cid = probe_ids[task];
            const LocalPQList& list = lists[cid];

            if (list.ids.empty()) {
                continue;
            }

            // 每个 list 有自己的局部 PQ codebook
            build_local_pq_lut(
                query,
                list,
                lut.data()
            );

            for (size_t j = 0; j < list.ids.size(); ++j) {
                const uint8_t* code =
                    list.codes.data() + j * M;

                float approx_score = scan_local_pq_code(
                    code,
                    lut.data()
                );

                uint32_t id = list.ids[j];

                ivfpqlocal_update_top_score(
                    approx_score,
                    id,
                    local_score,
                    local_id,
                    local_p,
                    cnt,
                    worst_pos,
                    worst_score
                );
            }
        }

        *out_cnt = cnt;
    }

    size_t get_nlist() const {
        return nlist;
    }

    size_t get_M() const {
        return M;
    }

    size_t get_Ks() const {
        return Ks;
    }

    size_t get_subdim() const {
        return subdim;
    }

    size_t get_list_size(size_t cid) const {
        if (cid >= lists.size()) {
            return 0;
        }

        return lists[cid].ids.size();
    }

    bool uses_opq() const {
        return opq_iters > 0;
    }

    double get_opq_orthogonality_error() const {
        if (opq_rotation.empty()) {
            return 0.0;
        }
        return ann_opq::orthogonality_error(opq_rotation, vecdim);
    }

private:
    float* base_float;
    size_t base_number;
    size_t vecdim;

    size_t nlist;
    size_t M;
    size_t Ks;
    size_t subdim;

    size_t train_size;
    size_t kmeans_iters;

    IVFPQLocalInitMode init_mode;
    size_t opq_iters;

    std::vector<float> coarse_centroids;
    std::vector<LocalPQList> lists;
    std::vector<float> opq_rotation;
    std::vector<float> rotated_base;

    void normalize_params() {
        if (base_number == 0 || vecdim == 0) {
            nlist = 0;
            M = 0;
            Ks = 0;
            subdim = 0;
            train_size = 0;
            kmeans_iters = 0;
            return;
        }

        if (nlist == 0) {
            nlist = 100;
        }

        nlist = std::min(nlist, base_number);

        if (M == 0) {
            M = 1;
        }

        if (vecdim % M != 0) {
            throw std::runtime_error(
                "IVFPQLocalIndexSIMD: vecdim must be divisible by M."
            );
        }

        subdim = vecdim / M;

        if (Ks == 0) {
            Ks = 256;
        }

        if (Ks > 256) {
            Ks = 256;
        }

        if (train_size > base_number) {
            train_size = base_number;
        }

        if (train_size < nlist) {
            train_size = nlist;
        }

        if (kmeans_iters == 0) {
            kmeans_iters = 1;
        }
    }

    void build() {
        std::cerr << "IVF-PQ local build start. "
                  << "nlist = " << nlist
                  << ", M = " << M
                  << ", Ks = " << Ks
                  << ", subdim = " << subdim
                  << ", train_size = " << train_size
                  << ", iters = " << kmeans_iters
                  << "\n";

        train_coarse_centroids();
        build_invlists();

        if (opq_iters > 0) {
            train_opq_rotation();
            rotate_base_vectors();
        }

        train_local_pq_for_all_lists();
        encode_all_lists();

        print_list_stats();

        std::cerr << "IVF-PQ local build done.\n";
    }

    const float* get_base_vec(size_t id) const {
        return base_float + id * vecdim;
    }

    const float* get_pq_vec(size_t id) const {
        if (rotated_base.empty()) {
            return get_base_vec(id);
        }
        return rotated_base.data() + id * vecdim;
    }

    const float* prepare_pq_query(
        const float* query,
        std::vector<float>& storage
    ) const {
        if (opq_rotation.empty()) {
            return query;
        }
        storage.resize(vecdim);
        ann_opq::rotate_vector(
            query,
            opq_rotation,
            vecdim,
            storage.data()
        );
        return storage.data();
    }

    float* get_coarse_centroid(size_t c) {
        return coarse_centroids.data() + c * vecdim;
    }

    const float* get_coarse_centroid(size_t c) const {
        return coarse_centroids.data() + c * vecdim;
    }

    float* get_local_pq_centroid(
        LocalPQList& list,
        size_t m,
        size_t c
    ) {
        return list.codebooks.data() + (m * Ks + c) * subdim;
    }

    const float* get_local_pq_centroid(
        const LocalPQList& list,
        size_t m,
        size_t c
    ) const {
        return list.codebooks.data() + (m * Ks + c) * subdim;
    }

    size_t train_id(size_t t) const {
        if (train_size <= 1) {
            return 0;
        }

        size_t id = t * base_number / train_size;

        if (id >= base_number) {
            id = base_number - 1;
        }

        return id;
    }

    void train_opq_rotation() {
        std::cerr << "Training global OPQ rotation on IVF residuals...\n";
        std::vector<float> residuals(train_size * vecdim);
        for (size_t t = 0; t < train_size; ++t) {
            const size_t id = train_id(t);
            const float* vector = get_base_vec(id);
            const size_t list_id = nearest_coarse_centroid(vector);
            const float* centroid = get_coarse_centroid(list_id);
            float* residual = residuals.data() + t * vecdim;
            for (size_t d = 0; d < vecdim; ++d) {
                residual[d] = vector[d] - centroid[d];
            }
        }

        opq_rotation = ann_opq::train_rotation(
            residuals,
            train_size,
            vecdim,
            M,
            Ks,
            opq_iters,
            kmeans_iters
        );
        const double error = ann_opq::orthogonality_error(
            opq_rotation,
            vecdim
        );
        if (error > 1e-3) {
            throw std::runtime_error(
                "OPQ training produced a non-orthogonal rotation."
            );
        }
    }

    void rotate_base_vectors() {
        std::cerr << "Rotating base vectors for local PQ training...\n";
        rotated_base.resize(base_number * vecdim);
        for (size_t id = 0; id < base_number; ++id) {
            ann_opq::rotate_vector(
                get_base_vec(id),
                opq_rotation,
                vecdim,
                rotated_base.data() + id * vecdim
            );
        }
    }

    // ========================================================
    // IVF coarse centroid 训练
    // ========================================================
    void init_coarse_centroids_uniform() {
        for (size_t c = 0; c < nlist; ++c) {
            size_t t = c * train_size / nlist;

            if (t >= train_size) {
                t = train_size - 1;
            }

            size_t id = train_id(t);

            std::copy(
                get_base_vec(id),
                get_base_vec(id) + vecdim,
                get_coarse_centroid(c)
            );
        }
    }

    size_t nearest_coarse_centroid(
        const float* vec
    ) const {
        size_t best_c = 0;
        float best_dist = std::numeric_limits<float>::max();

        for (size_t c = 0; c < nlist; ++c) {
            float dist = ivfpqlocal_l2_neon(
                vec,
                get_coarse_centroid(c),
                vecdim
            );

            if (dist < best_dist) {
                best_dist = dist;
                best_c = c;
            }
        }

        return best_c;
    }

    void train_coarse_centroids() {
        std::cerr << "Training IVF coarse centroids...\n";

        init_coarse_centroids_uniform();

        std::vector<float> sums(nlist * vecdim);
        std::vector<size_t> counts(nlist);

        for (size_t iter = 0; iter < kmeans_iters; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t t = 0; t < train_size; ++t) {
                size_t id = train_id(t);
                const float* vec = get_base_vec(id);

                size_t cid = nearest_coarse_centroid(vec);

                float* sum_vec =
                    sums.data() + cid * vecdim;

                for (size_t d = 0; d < vecdim; ++d) {
                    sum_vec[d] += vec[d];
                }

                counts[cid]++;
            }

            for (size_t c = 0; c < nlist; ++c) {
                float* centroid = get_coarse_centroid(c);

                if (counts[c] == 0) {
                    size_t id = train_id((c + iter + 1) % train_size);

                    std::copy(
                        get_base_vec(id),
                        get_base_vec(id) + vecdim,
                        centroid
                    );
                } else {
                    float inv_count =
                        1.0f / static_cast<float>(counts[c]);

                    float* sum_vec =
                        sums.data() + c * vecdim;

                    for (size_t d = 0; d < vecdim; ++d) {
                        centroid[d] = sum_vec[d] * inv_count;
                    }
                }
            }
        }
    }

    void build_invlists() {
        std::cerr << "Building IVF inverted lists...\n";

        for (size_t c = 0; c < nlist; ++c) {
            lists[c].ids.clear();
            lists[c].codebooks.clear();
            lists[c].codes.clear();
        }

        std::vector<size_t> counts(nlist, 0);

        // 第一遍：统计每个倒排链表大小
        for (size_t i = 0; i < base_number; ++i) {
            size_t cid = nearest_coarse_centroid(
                get_base_vec(i)
            );

            counts[cid]++;
        }

        // 预留空间
        for (size_t c = 0; c < nlist; ++c) {
            lists[c].ids.reserve(counts[c]);
        }

        // 第二遍：真正写入 base id
        for (size_t i = 0; i < base_number; ++i) {
            size_t cid = nearest_coarse_centroid(
                get_base_vec(i)
            );

            lists[cid].ids.push_back(
                static_cast<uint32_t>(i)
            );
        }
    }

    // ========================================================
    // 每个 list 内部训练局部 PQ
    // ========================================================
    void train_local_pq_for_all_lists() {
        std::cerr << "Training local PQ codebooks for each list...\n";

        for (size_t c = 0; c < nlist; ++c) {
            LocalPQList& list = lists[c];

            if (list.ids.empty()) {
                continue;
            }

            list.codebooks.resize(M * Ks * subdim);

            for (size_t m = 0; m < M; ++m) {
                train_local_pq_subspace(list, m);
            }
        }
    }

    void train_local_pq_subspace(
        LocalPQList& list,
        size_t m
    ) {
        size_t list_size = list.ids.size();

        // 初始化局部 codebook
        for (size_t c = 0; c < Ks; ++c) {
            size_t pos = c * list_size / Ks;

            if (pos >= list_size) {
                pos = list_size - 1;
            }

            uint32_t id = list.ids[pos];

            const float* src =
                get_pq_vec(id) + m * subdim;

            float* dst =
                get_local_pq_centroid(list, m, c);

            std::copy(
                src,
                src + subdim,
                dst
            );
        }

        std::vector<float> sums(Ks * subdim);
        std::vector<size_t> counts(Ks);

        for (size_t iter = 0; iter < kmeans_iters; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            for (size_t j = 0; j < list_size; ++j) {
                uint32_t id = list.ids[j];

                const float* subvec =
                    get_pq_vec(id) + m * subdim;

                size_t cid = nearest_local_pq_centroid(
                    list,
                    subvec,
                    m
                );

                float* sum_vec =
                    sums.data() + cid * subdim;

                for (size_t d = 0; d < subdim; ++d) {
                    sum_vec[d] += subvec[d];
                }

                counts[cid]++;
            }

            for (size_t c = 0; c < Ks; ++c) {
                float* centroid =
                    get_local_pq_centroid(list, m, c);

                if (counts[c] == 0) {
                    uint32_t id =
                        list.ids[(c + iter + 1) % list_size];

                    const float* src =
                        get_pq_vec(id) + m * subdim;

                    std::copy(
                        src,
                        src + subdim,
                        centroid
                    );
                } else {
                    float inv_count =
                        1.0f / static_cast<float>(counts[c]);

                    const float* sum_vec =
                        sums.data() + c * subdim;

                    for (size_t d = 0; d < subdim; ++d) {
                        centroid[d] = sum_vec[d] * inv_count;
                    }
                }
            }
        }
    }

    size_t nearest_local_pq_centroid(
        const LocalPQList& list,
        const float* subvec,
        size_t m
    ) const {
        size_t best_c = 0;
        float best_dist = std::numeric_limits<float>::max();

        for (size_t c = 0; c < Ks; ++c) {
            float dist = ivfpqlocal_l2_neon(
                subvec,
                get_local_pq_centroid(list, m, c),
                subdim
            );

            if (dist < best_dist) {
                best_dist = dist;
                best_c = c;
            }
        }

        return best_c;
    }

    void encode_all_lists() {
        std::cerr << "Encoding vectors in each list by local PQ...\n";

        for (size_t c = 0; c < nlist; ++c) {
            LocalPQList& list = lists[c];

            if (list.ids.empty()) {
                continue;
            }

            list.codes.resize(list.ids.size() * M);

            for (size_t j = 0; j < list.ids.size(); ++j) {
                uint32_t id = list.ids[j];

                const float* vec = get_pq_vec(id);
                uint8_t* code = list.codes.data() + j * M;

                for (size_t m = 0; m < M; ++m) {
                    const float* subvec = vec + m * subdim;

                    size_t cid = nearest_local_pq_centroid(
                        list,
                        subvec,
                        m
                    );

                    code[m] = static_cast<uint8_t>(cid);
                }
            }
        }
    }

    // ========================================================
    // 查询相关
    // ========================================================
    void select_probe_ids(
        const float* query,
        size_t nprobe,
        std::vector<uint32_t>& probe_ids
    ) const {
        probe_ids.clear();

        nprobe = std::min(nprobe, nlist);

        std::vector<float> probe_dist(nprobe);
        std::vector<uint32_t> probe_id(nprobe);

        size_t probe_cnt = 0;
        size_t probe_worst_pos = 0;
        float probe_worst_dist = 0.0f;

        for (size_t c = 0; c < nlist; ++c) {
            float dist = ivfpqlocal_l2_neon(
                query,
                get_coarse_centroid(c),
                vecdim
            );

            ivfpqlocal_update_top_probe(
                dist,
                static_cast<uint32_t>(c),
                probe_dist.data(),
                probe_id.data(),
                nprobe,
                probe_cnt,
                probe_worst_pos,
                probe_worst_dist
            );
        }

        probe_ids.resize(probe_cnt);

        for (size_t i = 0; i < probe_cnt; ++i) {
            probe_ids[i] = probe_id[i];
        }
    }

    void build_local_pq_lut(
        const float* query,
        const LocalPQList& list,
        float* lut
    ) const {
        for (size_t m = 0; m < M; ++m) {
            const float* query_sub = query + m * subdim;

            for (size_t c = 0; c < Ks; ++c) {
                const float* centroid =
                    get_local_pq_centroid(list, m, c);

                float score = ivfpqlocal_inner_product_neon_fma(
                    query_sub,
                    centroid,
                    subdim
                );

                lut[m * Ks + c] = score;
            }
        }
    }

    float scan_local_pq_code(
        const uint8_t* code,
        const float* lut
    ) const {
        float score = 0.0f;

        for (size_t m = 0; m < M; ++m) {
            uint8_t cid = code[m];
            score += lut[m * Ks + cid];
        }

        return score;
    }

    void print_list_stats() const {
        if (nlist == 0) {
            return;
        }

        size_t min_size = std::numeric_limits<size_t>::max();
        size_t max_size = 0;
        size_t total = 0;
        size_t empty_cnt = 0;

        for (size_t c = 0; c < nlist; ++c) {
            size_t sz = lists[c].ids.size();

            min_size = std::min(min_size, sz);
            max_size = std::max(max_size, sz);
            total += sz;

            if (sz == 0) {
                empty_cnt++;
            }
        }

        double avg_size =
            static_cast<double>(total)
            / static_cast<double>(nlist);

        std::cerr << "IVF-PQ local list stats: "
                  << "min = " << min_size
                  << ", max = " << max_size
                  << ", avg = " << avg_size
                  << ", empty = " << empty_cnt
                  << "\n";
    }
};

static inline void ivfpq_local_search_omp_query_parallel(
    const IVFPQLocalIndexSIMD& index,
    float* queries,
    size_t query_number,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    size_t rerank_p,
    int num_threads,
    std::vector<std::priority_queue<std::pair<float, uint32_t> > >& results
) {
    results.resize(query_number);

    if (query_number == 0) {
        return;
    }

    if (num_threads <= 1) {
        for (size_t qid = 0; qid < query_number; ++qid) {
            float* query = queries + qid * vecdim;

            results[qid] = index.search(
                query,
                k,
                nprobe,
                rerank_p
            );
        }

        return;
    }

    if (static_cast<size_t>(num_threads) > query_number) {
        num_threads = static_cast<int>(query_number);
    }

    omp_set_dynamic(0);

    #pragma omp parallel for num_threads(num_threads) schedule(static)
    for (long long qid = 0;
         qid < static_cast<long long>(query_number);
         ++qid) {

        float* query =
            queries + static_cast<size_t>(qid) * vecdim;

        results[static_cast<size_t>(qid)] = index.search(
            query,
            k,
            nprobe,
            rerank_p
        );
    }
}

struct IVFPQLocalQueryThreadParam {
    int tid;
    int num_threads;

    const IVFPQLocalIndexSIMD* index;

    float* queries;

    size_t query_number;
    size_t vecdim;
    size_t k;
    size_t nprobe;
    size_t rerank_p;

    std::vector<std::priority_queue<std::pair<float, uint32_t> > >* results;
};

static void* ivfpq_local_query_parallel_worker(void* arg) {
    IVFPQLocalQueryThreadParam* param =
        static_cast<IVFPQLocalQueryThreadParam*>(arg);

    int tid = param->tid;
    int num_threads = param->num_threads;

    const IVFPQLocalIndexSIMD* index = param->index;
    float* queries = param->queries;

    size_t query_number = param->query_number;
    size_t vecdim = param->vecdim;
    size_t k = param->k;
    size_t nprobe = param->nprobe;
    size_t rerank_p = param->rerank_p;

    auto& results = *(param->results);

    // 循环划分 query：
    // thread 0: 0, T, 2T, ...
    // thread 1: 1, T+1, 2T+1, ...
    for (size_t qid = static_cast<size_t>(tid);
         qid < query_number;
         qid += static_cast<size_t>(num_threads)) {

        float* query = queries + qid * vecdim;

        results[qid] = index->search(
            query,
            k,
            nprobe,
            rerank_p
        );
    }

    return nullptr;
}

static inline void ivfpq_local_search_pthread_query_parallel(
    const IVFPQLocalIndexSIMD& index,
    float* queries,
    size_t query_number,
    size_t vecdim,
    size_t k,
    size_t nprobe,
    size_t rerank_p,
    int num_threads,
    std::vector<std::priority_queue<std::pair<float, uint32_t> > >& results
) {
    results.resize(query_number);

    if (query_number == 0) {
        return;
    }

    if (num_threads <= 1) {
        for (size_t qid = 0; qid < query_number; ++qid) {
            float* query = queries + qid * vecdim;

            results[qid] = index.search(
                query,
                k,
                nprobe,
                rerank_p
            );
        }

        return;
    }

    if (static_cast<size_t>(num_threads) > query_number) {
        num_threads = static_cast<int>(query_number);
    }

    std::vector<pthread_t> threads(num_threads);
    std::vector<IVFPQLocalQueryThreadParam> params(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        params[t].tid = t;
        params[t].num_threads = num_threads;
        params[t].index = &index;
        params[t].queries = queries;
        params[t].query_number = query_number;
        params[t].vecdim = vecdim;
        params[t].k = k;
        params[t].nprobe = nprobe;
        params[t].rerank_p = rerank_p;
        params[t].results = &results;

        pthread_create(
            &threads[t],
            nullptr,
            ivfpq_local_query_parallel_worker,
            &params[t]
        );
    }

    for (int t = 0; t < num_threads; ++t) {
        pthread_join(
            threads[t],
            nullptr
        );
    }
}
