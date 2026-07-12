#pragma once

#include <queue>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <random>
#include "simd_compat.h"
#include <pthread.h>
#include <omp.h>
#include <utility>
#include <atomic>

#include "flat_scan_simd.h"

// ============================================================
// IVF-SIMD baseline
//
// IVF = Inverted File Index
//
// 离线阶段：
//   1. KMeans 训练 nlist 个 coarse centroids
//   2. 每条 base 向量分配到最近 centroid
//   3. 建立 inverted lists
//
// 在线阶段：
//   1. Query 选择最近的 nprobe 个 coarse centroids
//   2. 只扫描这些 inverted lists
//   3. 扫描时使用 inner_product_neon16_fma
//   4. 维护 Top-k
// ============================================================

enum class IVFInitMode {
    Uniform,
    KMeansPP
};

// ------------------------------------------------------------
// SIMD L2 距离
// 用于 KMeans 分配和 coarse centroid 选择
// ------------------------------------------------------------
static inline float ivf_l2_neon(
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
// 固定数组 Top-k：最大内积分数
// score 越大越好
// ------------------------------------------------------------
static inline void ivf_recompute_worst_score(
    const float* score,
    size_t cnt,
    size_t& worst_pos,
    float& worst_score
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

static inline void ivf_update_topk_score(
    float score,
    uint32_t id,
    float* best_score,
    uint32_t* best_id,
    size_t k,
    size_t& cnt,
    size_t& worst_pos,
    float& worst_score
) {
    if (k == 0) {
        return;
    }

    if (cnt < k) {
        best_score[cnt] = score;
        best_id[cnt] = id;
        ++cnt;

        if (cnt == k) {
            ivf_recompute_worst_score(
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

        ivf_recompute_worst_score(
            best_score,
            cnt,
            worst_pos,
            worst_score
        );
    }
}

// ------------------------------------------------------------
// 固定数组 Top-nprobe：最小 L2 距离
// dist 越小越好
// ------------------------------------------------------------
static inline void ivf_recompute_worst_dist(
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

static inline void ivf_update_top_probe(
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
            ivf_recompute_worst_dist(
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

        ivf_recompute_worst_dist(
            best_dist,
            cnt,
            worst_pos,
            worst_dist
        );
    }
}

class IVFIndexSIMD;

struct IVFListPthreadParam {
    int tid;

    const IVFIndexSIMD* index;

    const std::vector<uint32_t>* probe_ids;
    std::atomic<size_t>* next_probe;

    float* query;

    size_t local_p;

    float* local_score;
    uint32_t* local_id;
    size_t* local_cnt;
};

static void* ivf_list_parallel_worker(void* arg);

class IVFIndexSIMD {
public:
    IVFIndexSIMD(
        float* base,
        size_t base_number,
        size_t vecdim,
        size_t nlist = 100,
        size_t train_size = 10000,
        size_t kmeans_iters = 6,
        IVFInitMode init_mode = IVFInitMode::Uniform
    )
        : base_float(base),
          base_number(base_number),
          vecdim(vecdim),
          nlist(nlist),
          train_size(train_size),
          kmeans_iters(kmeans_iters),
          init_mode(init_mode)
    {
        normalize_params();

        centroids.resize(this->nlist * this->vecdim);
        invlists.resize(this->nlist);

        build();
    }

    std::priority_queue<std::pair<float, uint32_t> > search(
        float* query,
        size_t k,
        size_t nprobe
    ) const {
        std::priority_queue<std::pair<float, uint32_t> > result;

        if (k == 0 || base_number == 0 || nlist == 0) {
            return result;
        }

        k = std::min(k, base_number);
        nprobe = std::max(nprobe, static_cast<size_t>(1));
        nprobe = std::min(nprobe, nlist);

        // ----------------------------------------------------
        // 1. 选择最近的 nprobe 个 coarse centroids
        // ----------------------------------------------------
        std::vector<float> probe_dist(nprobe);
        std::vector<uint32_t> probe_id(nprobe);

        size_t probe_cnt = 0;
        size_t probe_worst_pos = 0;
        float probe_worst_dist = 0.0f;

        for (size_t c = 0; c < nlist; ++c) {
            const float* centroid = get_centroid(c);

            float dist = ivf_l2_neon(
                query,
                centroid,
                vecdim
            );

            ivf_update_top_probe(
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

        // ----------------------------------------------------
        // 2. 扫描选中的 inverted lists
        // ----------------------------------------------------
        std::vector<float> best_score(k);
        std::vector<uint32_t> best_id(k);

        size_t best_cnt = 0;
        size_t best_worst_pos = 0;
        float best_worst_score = 0.0f;

        constexpr size_t PREFETCH_DIST = 8;

        for (size_t p = 0; p < probe_cnt; ++p) {
            uint32_t cid = probe_id[p];
            const std::vector<uint32_t>& list = invlists[cid];

            for (size_t j = 0; j < list.size(); ++j) {
                if (j + PREFETCH_DIST < list.size()) {
                    uint32_t next_id = list[j + PREFETCH_DIST];

                    __builtin_prefetch(
                        base_float + static_cast<size_t>(next_id) * vecdim,
                        0,
                        1
                    );
                }

                uint32_t id = list[j];

                const float* base_vec =
                    base_float + static_cast<size_t>(id) * vecdim;

                float score = inner_product_neon16_fma(
                    base_vec,
                    query,
                    vecdim
                );

                ivf_update_topk_score(
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
        }

        // ----------------------------------------------------
        // 3. 返回 main.cc 原有格式：priority_queue<pair<dis,id>>
        // ----------------------------------------------------
        for (size_t i = 0; i < best_cnt; ++i) {
            float dis = 1.0f - best_score[i];
            result.push({dis, best_id[i]});
        }

        return result;
    }

    // ------------------------------------------------------------
    // 选择最近的 nprobe 个 inverted lists
    // ------------------------------------------------------------
    void select_probe_ids(
        const float* query,
        size_t nprobe,
        std::vector<uint32_t>& probe_ids
    ) const {
        probe_ids.clear();

        if (base_number == 0 || nlist == 0 || nprobe == 0) {
            return;
        }

        nprobe = std::min(nprobe, nlist);

        std::vector<float> probe_dist(nprobe);
        std::vector<uint32_t> probe_id(nprobe);

        size_t probe_cnt = 0;
        size_t probe_worst_pos = 0;
        float probe_worst_dist = 0.0f;

        for (size_t c = 0; c < nlist; ++c) {
            const float* centroid = get_centroid(c);

            float dist = ivf_l2_neon(
                query,
                centroid,
                vecdim
            );

            ivf_update_top_probe(
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

    // ------------------------------------------------------------
    // Pthread worker 调用的动态 list 扫描函数
    // 每个线程不断从 next_probe 取一个 list，并维护自己的 local Top-p
    // ------------------------------------------------------------
    void scan_probe_lists_dynamic(
        const std::vector<uint32_t>& probe_ids,
        std::atomic<size_t>& next_probe,
        float* query,
        size_t local_p,
        float* local_score,
        uint32_t* local_id,
        size_t* out_cnt
    ) const {
        constexpr size_t PREFETCH_DIST = 8;

        size_t cnt = 0;
        size_t worst_pos = 0;
        float worst_score = 0.0f;

        while (true) {
            size_t task = next_probe.fetch_add(1);

            if (task >= probe_ids.size()) {
                break;
            }

            uint32_t cid = probe_ids[task];
            const std::vector<uint32_t>& list = invlists[cid];

            for (size_t j = 0; j < list.size(); ++j) {
                if (j + PREFETCH_DIST < list.size()) {
                    uint32_t next_id = list[j + PREFETCH_DIST];

                    __builtin_prefetch(
                        base_float + static_cast<size_t>(next_id) * vecdim,
                        0,
                        1
                    );
                }

                uint32_t id = list[j];

                const float* base_vec =
                    base_float + static_cast<size_t>(id) * vecdim;

                float score = inner_product_neon16_fma(
                    base_vec,
                    query,
                    vecdim
                );

                ivf_update_topk_score(
                    score,
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

    // ------------------------------------------------------------
    // OpenMP List 级并行
    // ------------------------------------------------------------
    std::priority_queue<std::pair<float, uint32_t> > search_omp_list_parallel(
        float* query,
        size_t k,
        size_t nprobe,
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

        // local_p = 0 时默认等于 k
        if (local_p == 0) {
            local_p = k;
        }

        local_p = std::min(local_p, base_number);

        if (num_threads <= 1) {
            return search(
                query,
                k,
                nprobe
            );
        }

        if (static_cast<size_t>(num_threads) > nprobe) {
            num_threads = static_cast<int>(nprobe);
        }

        // 1. 单线程选出最近的 nprobe 个倒排链表
        std::vector<uint32_t> probe_ids;
        select_probe_ids(
            query,
            nprobe,
            probe_ids
        );

        if (probe_ids.empty()) {
            return result;
        }

        // 2. 每个线程维护自己的 local Top-p
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

            constexpr size_t PREFETCH_DIST = 8;

            // list 长度可能不均匀，因此使用 dynamic,1
            #pragma omp for schedule(dynamic,1)
            for (long long pp = 0;
                pp < static_cast<long long>(probe_ids.size());
                ++pp) {

                uint32_t cid = probe_ids[static_cast<size_t>(pp)];
                const std::vector<uint32_t>& list = invlists[cid];

                for (size_t j = 0; j < list.size(); ++j) {
                    if (j + PREFETCH_DIST < list.size()) {
                        uint32_t next_id = list[j + PREFETCH_DIST];

                        __builtin_prefetch(
                            base_float + static_cast<size_t>(next_id) * vecdim,
                            0,
                            1
                        );
                    }

                    uint32_t id = list[j];

                    const float* base_vec =
                        base_float + static_cast<size_t>(id) * vecdim;

                    float score = inner_product_neon16_fma(
                        base_vec,
                        query,
                        vecdim
                    );

                    ivf_update_topk_score(
                        score,
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

        // 3. reduce：local Top-p -> global Top-k
        std::vector<float> best_score(k);
        std::vector<uint32_t> best_id(k);

        size_t best_cnt = 0;
        size_t best_worst_pos = 0;
        float best_worst_score = 0.0f;

        for (int t = 0; t < num_threads; ++t) {
            float* score_t =
                local_scores.data() + static_cast<size_t>(t) * local_p;

            uint32_t* id_t =
                local_ids.data() + static_cast<size_t>(t) * local_p;

            for (size_t j = 0; j < local_cnt[static_cast<size_t>(t)]; ++j) {
                ivf_update_topk_score(
                    score_t[j],
                    id_t[j],
                    best_score.data(),
                    best_id.data(),
                    k,
                    best_cnt,
                    best_worst_pos,
                    best_worst_score
                );
            }
        }

        // 4. 返回原有格式：priority_queue<pair<dis,id>>
        for (size_t i = 0; i < best_cnt; ++i) {
            result.push({
                1.0f - best_score[i],
                best_id[i]
            });
        }

        return result;
    }

    // ------------------------------------------------------------
    // Pthread List 级并行
    // ------------------------------------------------------------
    std::priority_queue<std::pair<float, uint32_t> > search_pthread_list_parallel(
        float* query,
        size_t k,
        size_t nprobe,
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

        // local_p = 0 时默认等于 k
        if (local_p == 0) {
            local_p = k;
        }

        local_p = std::min(local_p, base_number);

        if (num_threads <= 1) {
            return search(
                query,
                k,
                nprobe
            );
        }

        if (static_cast<size_t>(num_threads) > nprobe) {
            num_threads = static_cast<int>(nprobe);
        }

        // 1. 单线程选出最近的 nprobe 个倒排链表
        std::vector<uint32_t> probe_ids;

        select_probe_ids(
            query,
            nprobe,
            probe_ids
        );

        if (probe_ids.empty()) {
            return result;
        }

        // 2. 每个线程维护自己的 local Top-p
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

        std::vector<pthread_t> threads(num_threads);
        std::vector<IVFListPthreadParam> params(num_threads);

        std::atomic<size_t> next_probe(0);

        for (int t = 0; t < num_threads; ++t) {
            params[t].tid = t;
            params[t].index = this;
            params[t].probe_ids = &probe_ids;
            params[t].next_probe = &next_probe;
            params[t].query = query;
            params[t].local_p = local_p;

            params[t].local_score =
                local_scores.data() + static_cast<size_t>(t) * local_p;

            params[t].local_id =
                local_ids.data() + static_cast<size_t>(t) * local_p;

            params[t].local_cnt =
                &local_cnt[static_cast<size_t>(t)];

            pthread_create(
                &threads[t],
                nullptr,
                ivf_list_parallel_worker,
                &params[t]
            );
        }

        for (int t = 0; t < num_threads; ++t) {
            pthread_join(
                threads[t],
                nullptr
            );
        }

        // 3. reduce：local Top-p -> global Top-k
        std::vector<float> best_score(k);
        std::vector<uint32_t> best_id(k);

        size_t best_cnt = 0;
        size_t best_worst_pos = 0;
        float best_worst_score = 0.0f;

        for (int t = 0; t < num_threads; ++t) {
            float* score_t =
                local_scores.data() + static_cast<size_t>(t) * local_p;

            uint32_t* id_t =
                local_ids.data() + static_cast<size_t>(t) * local_p;

            for (size_t j = 0; j < local_cnt[static_cast<size_t>(t)]; ++j) {
                ivf_update_topk_score(
                    score_t[j],
                    id_t[j],
                    best_score.data(),
                    best_id.data(),
                    k,
                    best_cnt,
                    best_worst_pos,
                    best_worst_score
                );
            }
        }

        // 4. 返回原有格式
        for (size_t i = 0; i < best_cnt; ++i) {
            result.push({
                1.0f - best_score[i],
                best_id[i]
            });
        }

        return result;
    }

    void select_probe_ids_omp(
        const float* query,
        size_t nprobe,
        int num_threads,
        std::vector<uint32_t>& probe_ids
    ) const {
        probe_ids.clear();

        if (base_number == 0 || nlist == 0 || nprobe == 0) {
            return;
        }

        nprobe = std::min(nprobe, nlist);

        if (num_threads <= 1) {
            select_probe_ids(query, nprobe, probe_ids);
            return;
        }

        if (static_cast<size_t>(num_threads) > nlist) {
            num_threads = static_cast<int>(nlist);
        }

        std::vector<float> local_dist(
            static_cast<size_t>(num_threads) * nprobe
        );

        std::vector<uint32_t> local_id(
            static_cast<size_t>(num_threads) * nprobe
        );

        std::vector<size_t> local_cnt(
            static_cast<size_t>(num_threads),
            0
        );

        size_t block =
            (nlist + static_cast<size_t>(num_threads) - 1)
            / static_cast<size_t>(num_threads);

        omp_set_dynamic(0);

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();

            size_t begin = static_cast<size_t>(tid) * block;
            size_t end = std::min(begin + block, nlist);

            float* dist_t =
                local_dist.data() + static_cast<size_t>(tid) * nprobe;

            uint32_t* id_t =
                local_id.data() + static_cast<size_t>(tid) * nprobe;

            size_t cnt = 0;
            size_t worst_pos = 0;
            float worst_dist = 0.0f;

            for (size_t c = begin; c < end; ++c) {
                float dist = ivf_l2_neon(
                    query,
                    get_centroid(c),
                    vecdim
                );

                ivf_update_top_probe(
                    dist,
                    static_cast<uint32_t>(c),
                    dist_t,
                    id_t,
                    nprobe,
                    cnt,
                    worst_pos,
                    worst_dist
                );
            }

            local_cnt[static_cast<size_t>(tid)] = cnt;
        }

        // reduce：local Top-nprobe -> global Top-nprobe
        std::vector<float> global_dist(nprobe);
        std::vector<uint32_t> global_id(nprobe);

        size_t global_cnt = 0;
        size_t global_worst_pos = 0;
        float global_worst_dist = 0.0f;

        for (int t = 0; t < num_threads; ++t) {
            float* dist_t =
                local_dist.data() + static_cast<size_t>(t) * nprobe;

            uint32_t* id_t =
                local_id.data() + static_cast<size_t>(t) * nprobe;

            for (size_t j = 0; j < local_cnt[static_cast<size_t>(t)]; ++j) {
                ivf_update_top_probe(
                    dist_t[j],
                    id_t[j],
                    global_dist.data(),
                    global_id.data(),
                    nprobe,
                    global_cnt,
                    global_worst_pos,
                    global_worst_dist
                );
            }
        }

        probe_ids.resize(global_cnt);

        for (size_t i = 0; i < global_cnt; ++i) {
            probe_ids[i] = global_id[i];
        }
    }

    std::priority_queue<std::pair<float, uint32_t> > search_omp_centroid_parallel(
        float* query,
        size_t k,
        size_t nprobe,
        int num_threads
    ) const {
        std::vector<uint32_t> probe_ids;

        select_probe_ids_omp(
            query,
            nprobe,
            num_threads,
            probe_ids
        );

        return search_with_probe_ids(
            query,
            k,
            probe_ids
        );
    }

    std::priority_queue<std::pair<float, uint32_t> > search_with_probe_ids(
        float* query,
        size_t k,
        const std::vector<uint32_t>& probe_ids
    ) const {
        std::priority_queue<std::pair<float, uint32_t> > result;

        if (k == 0 || probe_ids.empty()) {
            return result;
        }

        k = std::min(k, base_number);

        std::vector<float> best_score(k);
        std::vector<uint32_t> best_id(k);

        size_t best_cnt = 0;
        size_t best_worst_pos = 0;
        float best_worst_score = 0.0f;

        constexpr size_t PREFETCH_DIST = 8;

        for (size_t p = 0; p < probe_ids.size(); ++p) {
            uint32_t cid = probe_ids[p];

            const std::vector<uint32_t>& list = invlists[cid];

            for (size_t j = 0; j < list.size(); ++j) {
                if (j + PREFETCH_DIST < list.size()) {
                    uint32_t next_id = list[j + PREFETCH_DIST];

                    __builtin_prefetch(
                        base_float + static_cast<size_t>(next_id) * vecdim,
                        0,
                        1
                    );
                }

                uint32_t id = list[j];

                const float* base_vec =
                    base_float + static_cast<size_t>(id) * vecdim;

                float score = inner_product_neon16_fma(
                    base_vec,
                    query,
                    vecdim
                );

                ivf_update_topk_score(
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
        }

        for (size_t i = 0; i < best_cnt; ++i) {
            result.push({
                1.0f - best_score[i],
                best_id[i]
            });
        }

        return result;
    }

    size_t get_nlist() const {
        return nlist;
    }

    size_t get_train_size() const {
        return train_size;
    }

    size_t get_kmeans_iters() const {
        return kmeans_iters;
    }

    size_t get_list_size(size_t cid) const {
        if (cid >= invlists.size()) {
            return 0;
        }
        return invlists[cid].size();
    }

    size_t get_total_list_size() const {
        size_t total = 0;
        for (const auto& list : invlists) {
            total += list.size();
        }
        return total;
    }

private:
    float* base_float;
    size_t base_number;
    size_t vecdim;

    size_t nlist;
    size_t train_size;
    size_t kmeans_iters;
    IVFInitMode init_mode;

    std::vector<float> centroids;
    std::vector<std::vector<uint32_t> > invlists;

    void normalize_params() {
        if (base_number == 0 || vecdim == 0) {
            nlist = 0;
            train_size = 0;
            kmeans_iters = 0;
            return;
        }

        if (nlist == 0) {
            nlist = 100;
        }

        if (nlist > base_number) {
            nlist = base_number;
        }

        if (nlist < 1) {
            nlist = 1;
        }

        if (train_size > base_number) {
            train_size = base_number;
        }

        if (train_size < nlist) {
            if (base_number >= nlist) {
                train_size = nlist;
            } else {
                train_size = base_number;
            }
        }

        if (kmeans_iters == 0) {
            kmeans_iters = 1;
        }
    }

    void build() {
        std::cerr << "IVF build start. nlist = " << nlist
                  << ", train_size = " << train_size
                  << ", iters = " << kmeans_iters
                  << ", init = "
                  << (init_mode == IVFInitMode::KMeansPP ? "KMeans++" : "Uniform")
                  << "\n";

        train_coarse_centroids();
        build_invlists();

        print_list_stats();

        std::cerr << "IVF build done.\n";
    }

    float* get_centroid(size_t c) {
        return &centroids[c * vecdim];
    }

    const float* get_centroid(size_t c) const {
        return &centroids[c * vecdim];
    }

    const float* get_base_vec(size_t id) const {
        return base_float + id * vecdim;
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

    size_t nearest_centroid_l2(
        const float* vec
    ) const {
        size_t best_c = 0;
        float best_dist = std::numeric_limits<float>::max();

        for (size_t c = 0; c < nlist; ++c) {
            float dist = ivf_l2_neon(
                vec,
                get_centroid(c),
                vecdim
            );

            if (dist < best_dist) {
                best_dist = dist;
                best_c = c;
            }
        }

        return best_c;
    }

    void init_centroids_uniform() {
        for (size_t c = 0; c < nlist; ++c) {
            size_t t = c * train_size / nlist;

            if (t >= train_size) {
                t = train_size - 1;
            }

            size_t id = train_id(t);

            const float* src = get_base_vec(id);
            float* dst = get_centroid(c);

            std::copy(
                src,
                src + vecdim,
                dst
            );
        }
    }

    void init_centroids_kmeanspp() {
        std::mt19937 rng(2026);

        std::vector<float> min_dist(
            train_size,
            std::numeric_limits<float>::max()
        );

        std::uniform_int_distribution<size_t> first_dist(
            0,
            train_size - 1
        );

        size_t first_t = first_dist(rng);
        size_t first_id = train_id(first_t);

        std::copy(
            get_base_vec(first_id),
            get_base_vec(first_id) + vecdim,
            get_centroid(0)
        );

        for (size_t c = 1; c < nlist; ++c) {
            const float* last_centroid = get_centroid(c - 1);

            float total_dist = 0.0f;

            for (size_t t = 0; t < train_size; ++t) {
                size_t id = train_id(t);
                const float* vec = get_base_vec(id);

                float dist = ivf_l2_neon(
                    vec,
                    last_centroid,
                    vecdim
                );

                if (dist < min_dist[t]) {
                    min_dist[t] = dist;
                }

                total_dist += min_dist[t];
            }

            size_t chosen_t = 0;

            if (total_dist <= 1e-12f) {
                chosen_t = c % train_size;
            } else {
                std::uniform_real_distribution<float> prob_dist(
                    0.0f,
                    total_dist
                );

                float r = prob_dist(rng);
                float acc = 0.0f;

                for (size_t t = 0; t < train_size; ++t) {
                    acc += min_dist[t];

                    if (acc >= r) {
                        chosen_t = t;
                        break;
                    }
                }
            }

            size_t chosen_id = train_id(chosen_t);

            std::copy(
                get_base_vec(chosen_id),
                get_base_vec(chosen_id) + vecdim,
                get_centroid(c)
            );
        }
    }

    void init_centroids() {
        if (init_mode == IVFInitMode::KMeansPP) {
            init_centroids_kmeanspp();
        } else {
            init_centroids_uniform();
        }
    }

    void train_coarse_centroids() {
        init_centroids();

        std::vector<float> sums(nlist * vecdim);
        std::vector<size_t> counts(nlist);

        for (size_t iter = 0; iter < kmeans_iters; ++iter) {
            std::fill(sums.begin(), sums.end(), 0.0f);
            std::fill(counts.begin(), counts.end(), 0);

            // 分配训练样本到最近 centroid
            for (size_t t = 0; t < train_size; ++t) {
                size_t id = train_id(t);
                const float* vec = get_base_vec(id);

                size_t cid = nearest_centroid_l2(vec);

                float* sum_vec = &sums[cid * vecdim];

                for (size_t d = 0; d < vecdim; ++d) {
                    sum_vec[d] += vec[d];
                }

                counts[cid]++;
            }

            // 更新 centroid
            for (size_t c = 0; c < nlist; ++c) {
                float* centroid = get_centroid(c);

                if (counts[c] == 0) {
                    // 空簇：重新使用某个训练样本初始化
                    size_t id = train_id((c + iter + 1) % train_size);
                    const float* src = get_base_vec(id);

                    std::copy(
                        src,
                        src + vecdim,
                        centroid
                    );
                } else {
                    float inv_count =
                        1.0f / static_cast<float>(counts[c]);

                    float* sum_vec = &sums[c * vecdim];

                    for (size_t d = 0; d < vecdim; ++d) {
                        centroid[d] = sum_vec[d] * inv_count;
                    }
                }
            }
        }
    }

    void build_invlists() {
        invlists.clear();
        invlists.resize(nlist);

        std::vector<size_t> counts(nlist, 0);

        // 第一遍：统计每个 list 大小，便于 reserve
        for (size_t i = 0; i < base_number; ++i) {
            const float* vec = get_base_vec(i);
            size_t cid = nearest_centroid_l2(vec);
            counts[cid]++;
        }

        for (size_t c = 0; c < nlist; ++c) {
            invlists[c].reserve(counts[c]);
        }

        // 第二遍：真正写入 id
        for (size_t i = 0; i < base_number; ++i) {
            const float* vec = get_base_vec(i);
            size_t cid = nearest_centroid_l2(vec);

            invlists[cid].push_back(
                static_cast<uint32_t>(i)
            );
        }
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
            size_t sz = invlists[c].size();

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

        std::cerr << "IVF list stats: "
                  << "min = " << min_size
                  << ", max = " << max_size
                  << ", avg = " << avg_size
                  << ", empty = " << empty_cnt
                  << "\n";
    }
};

//query级并行 OpenMP版本

static inline void ivf_search_simd_omp_query_parallel(
    const IVFIndexSIMD& index,
    float* queries,
    size_t query_number,
    size_t vecdim,
    size_t k,
    size_t nprobe,
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
                nprobe
            );
        }

        return;
    }

    if (static_cast<size_t>(num_threads) > query_number) {
        num_threads = static_cast<int>(query_number);
    }

    omp_set_dynamic(0);

    #pragma omp parallel for num_threads(num_threads) schedule(dynamic,2)
    for (long long qid = 0; qid < static_cast<long long>(query_number); ++qid) {
        float* query = queries + static_cast<size_t>(qid) * vecdim;

        results[static_cast<size_t>(qid)] = index.search(
            query,
            k,
            nprobe
        );
    }
}

// query级并行 pthread版本

struct IVFQueryThreadParam {
    int tid;
    int num_threads;

    const IVFIndexSIMD* index;

    float* queries;

    size_t query_number;
    size_t vecdim;
    size_t k;
    size_t nprobe;

    std::vector<std::priority_queue<std::pair<float, uint32_t> > >* results;
};

static void* ivf_query_parallel_worker(void* arg) {
    IVFQueryThreadParam* param =
        static_cast<IVFQueryThreadParam*>(arg);

    int tid = param->tid;
    int num_threads = param->num_threads;

    const IVFIndexSIMD* index = param->index;
    float* queries = param->queries;

    size_t query_number = param->query_number;
    size_t vecdim = param->vecdim;
    size_t k = param->k;
    size_t nprobe = param->nprobe;

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
            nprobe
        );
    }

    return nullptr;
}

static inline void ivf_search_simd_pthread_query_parallel(
    const IVFIndexSIMD& index,
    float* queries,
    size_t query_number,
    size_t vecdim,
    size_t k,
    size_t nprobe,
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
                nprobe
            );
        }

        return;
    }

    if (static_cast<size_t>(num_threads) > query_number) {
        num_threads = static_cast<int>(query_number);
    }

    std::vector<pthread_t> threads(num_threads);
    std::vector<IVFQueryThreadParam> params(num_threads);

    for (int t = 0; t < num_threads; ++t) {
        params[t].tid = t;
        params[t].num_threads = num_threads;
        params[t].index = &index;
        params[t].queries = queries;
        params[t].query_number = query_number;
        params[t].vecdim = vecdim;
        params[t].k = k;
        params[t].nprobe = nprobe;
        params[t].results = &results;

        pthread_create(
            &threads[t],
            nullptr,
            ivf_query_parallel_worker,
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

static void* ivf_list_parallel_worker(void* arg) {
    IVFListPthreadParam* param =
        static_cast<IVFListPthreadParam*>(arg);

    param->index->scan_probe_lists_dynamic(
        *(param->probe_ids),
        *(param->next_probe),
        param->query,
        param->local_p,
        param->local_score,
        param->local_id,
        param->local_cnt
    );

    return nullptr;
}
