#pragma once

#include "simd_compat.h"
#include "opq_transform.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

#if defined(__AVX2__) || defined(_M_AVX2)
#include <immintrin.h>
#endif

// ============================================================
// IVF-PQ-SIMD baseline
//
// 实现方式：全局 PQ + IVF inverted lists
//
// 离线阶段：
//   1. 训练全局 PQ codebook
//   2. 将所有 base 向量编码成 PQ code
//   3. 训练 IVF coarse centroids
//   4. 将 base id 分配到 inverted lists
//
// 在线阶段：
//   1. Query 选择最近的 nprobe 个 coarse centroids
//   2. 构建全局 PQ LUT
//   3. 只扫描选中 lists 中的 PQ code
//   4. 维护 Top-rerank_p 粗排候选
//   5. 使用原始 float 向量重新精排
//   6. 返回 Top-k
// ============================================================

enum class IVFPQInitMode {
    Uniform
};

// ------------------------------------------------------------
// SIMD L2 距离
// ------------------------------------------------------------
static inline float ivfpq_l2_neon(
    const float* __restrict__ a,
    const float* __restrict__ b,
    size_t dim
) {
#if defined(__AVX2__) || defined(_M_AVX2)
    size_t i = 0;
    __m256 sum = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        const __m256 diff = _mm256_sub_ps(va, vb);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(diff, diff));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3]
                 + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < dim; ++i) {
        const float diff = a[i] - b[i];
        result += diff * diff;
    }
    return result;
#else
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
#endif
}

// ------------------------------------------------------------
// SIMD 内积
// ------------------------------------------------------------
static inline float ivfpq_inner_product_neon_fma(
    const float* __restrict__ a,
    const float* __restrict__ b,
    size_t dim
) {
#if defined(__AVX2__) || defined(_M_AVX2)
    size_t i = 0;
    __m256 sum = _mm256_setzero_ps();
    for (; i + 8 <= dim; i += 8) {
        const __m256 va = _mm256_loadu_ps(a + i);
        const __m256 vb = _mm256_loadu_ps(b + i);
        sum = _mm256_add_ps(sum, _mm256_mul_ps(va, vb));
    }
    alignas(32) float lanes[8];
    _mm256_store_ps(lanes, sum);
    float result = lanes[0] + lanes[1] + lanes[2] + lanes[3]
                 + lanes[4] + lanes[5] + lanes[6] + lanes[7];
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    return result;
#else
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
#endif
}

// ------------------------------------------------------------
// Top-k：分数越大越好
// ------------------------------------------------------------
template <typename ScoreT>
static inline void ivfpq_recompute_worst_score(
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
static inline void ivfpq_update_top_score(
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
            ivfpq_recompute_worst_score(
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

        ivfpq_recompute_worst_score(
            best_score,
            cnt,
            worst_pos,
            worst_score
        );
    }
}

// ------------------------------------------------------------
// Top-nprobe：距离越小越好
// ------------------------------------------------------------
static inline void ivfpq_recompute_worst_dist(
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

static inline void ivfpq_update_top_probe(
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
            ivfpq_recompute_worst_dist(
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

        ivfpq_recompute_worst_dist(
            best_dist,
            cnt,
            worst_pos,
            worst_dist
        );
    }
}

class IVFPQIndexSIMD {
public:
    IVFPQIndexSIMD(
        float* base,
        size_t base_number,
        size_t vecdim,
        size_t nlist = 100,
        size_t M = 16,
        size_t Ks = 256,
        size_t train_size = 10000,
        size_t kmeans_iters = 6,
        IVFPQInitMode init_mode = IVFPQInitMode::Uniform,
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
        pq_codebooks.resize(this->M * this->Ks * this->subdim);
        codes.resize(this->base_number * this->M);
        invlists.resize(this->nlist);

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

        // ----------------------------------------------------
        // 1. 选择最近的 nprobe 个 IVF coarse centroids
        // ----------------------------------------------------
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

        // ----------------------------------------------------
        // 2. 构建全局 PQ LUT
        // LUT[m][c] = query_m · codebook[m][c]
        // ----------------------------------------------------
        std::vector<float> lut(M * Ks);

        build_pq_lut(
            pq_query,
            lut.data()
        );

        // ----------------------------------------------------
        // 3. 只扫描选中 inverted lists 中的 PQ code
        // 维护 Top-rerank_p 粗排候选
        // ----------------------------------------------------
        std::vector<float> cand_score(rerank_p);
        std::vector<uint32_t> cand_id(rerank_p);

        size_t cand_cnt = 0;
        size_t cand_worst_pos = 0;
        float cand_worst_score = 0.0f;

        for (size_t pp = 0; pp < probe_ids.size(); ++pp) {
            uint32_t cid = probe_ids[pp];
            const std::vector<uint32_t>& list = invlists[cid];

            for (size_t j = 0; j < list.size(); ++j) {
                uint32_t id = list[j];

                const uint8_t* code =
                    codes.data() + static_cast<size_t>(id) * M;

                float approx_score = scan_pq_code(
                    code,
                    lut.data()
                );

                ivfpq_update_top_score(
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

        // ----------------------------------------------------
        // 4. 对粗排候选做原始 float 精排
        // ----------------------------------------------------
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

            float score = ivfpq_inner_product_neon_fma(
                base_vec,
                query,
                vecdim
            );

            ivfpq_update_top_score(
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

        // ----------------------------------------------------
        // 5. 返回原评测接口使用的 priority_queue<pair<dis, id>>
        // ----------------------------------------------------
        for (size_t i = 0; i < best_cnt; ++i) {
            float dis = 1.0f - best_score[i];
            result.push({dis, best_id[i]});
        }

        return result;
    }

    // Return approximate OPQ-IVF-PQ candidates without exact reranking.
    // This is the hand-off point for CPU/GPU heterogeneous execution.
    std::vector<uint32_t> generate_candidates(
        float* query,
        size_t nprobe,
        size_t rerank_p
    ) const {
        if (base_number == 0 || nlist == 0 || rerank_p == 0) {
            return std::vector<uint32_t>();
        }
        nprobe = std::max(static_cast<size_t>(1), std::min(nprobe, nlist));
        rerank_p = std::min(rerank_p, base_number);
        std::vector<uint32_t> probe_ids;
        select_probe_ids(query, nprobe, probe_ids);
        std::vector<float> rotated_query;
        const float* pq_query = prepare_pq_query(query, rotated_query);
        std::vector<float> lut(M * Ks);
        build_pq_lut(pq_query, lut.data());
        std::vector<float> scores(rerank_p);
        std::vector<uint32_t> ids(rerank_p);
        size_t count = 0, worst_pos = 0;
        float worst_score = 0.0f;
        for (size_t pp = 0; pp < probe_ids.size(); ++pp) {
            const std::vector<uint32_t>& list = invlists[probe_ids[pp]];
            for (size_t j = 0; j < list.size(); ++j) {
                const uint32_t id = list[j];
                const float score = scan_pq_code(
                    codes.data() + static_cast<size_t>(id) * M,
                    lut.data()
                );
                ivfpq_update_top_score(
                    score, id, scores.data(), ids.data(), rerank_p,
                    count, worst_pos, worst_score
                );
            }
        }
        ids.resize(count);
        return ids;
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
        if (cid >= invlists.size()) {
            return 0;
        }

        return invlists[cid].size();
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

    IVFPQInitMode init_mode;
    size_t opq_iters;

    // IVF
    std::vector<float> coarse_centroids;
    std::vector<std::vector<uint32_t> > invlists;

    // 全局 PQ
    std::vector<float> pq_codebooks;
    std::vector<uint8_t> codes;
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
                "IVFPQIndexSIMD: vecdim must be divisible by M."
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

        size_t min_train = std::max(nlist, Ks);

        if (train_size < min_train) {
            train_size = std::min(base_number, min_train);
        }

        if (kmeans_iters == 0) {
            kmeans_iters = 1;
        }
    }

    void build() {
        std::cerr << "IVF-PQ build start. "
                  << "nlist = " << nlist
                  << ", M = " << M
                  << ", Ks = " << Ks
                  << ", subdim = " << subdim
                  << ", train_size = " << train_size
                  << ", iters = " << kmeans_iters
                  << "\n";

        if (opq_iters > 0) {
            train_opq_rotation();
            rotate_base_vectors();
        }

        train_global_pq();
        encode_all_base();

        train_coarse_centroids();
        build_invlists();

        print_list_stats();

        std::cerr << "IVF-PQ build done.\n";
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

    float* get_pq_centroid(size_t m, size_t c) {
        return pq_codebooks.data() + (m * Ks + c) * subdim;
    }

    const float* get_pq_centroid(size_t m, size_t c) const {
        return pq_codebooks.data() + (m * Ks + c) * subdim;
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
        std::cerr << "Training global OPQ rotation on original vectors...\n";
        std::vector<float> training_vectors(train_size * vecdim);
        for (size_t t = 0; t < train_size; ++t) {
            const size_t id = train_id(t);
            std::copy(
                get_base_vec(id),
                get_base_vec(id) + vecdim,
                training_vectors.data() + t * vecdim
            );
        }

        opq_rotation = ann_opq::train_rotation(
            training_vectors,
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
        std::cerr << "Rotating base vectors for global PQ training...\n";
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
    // 全局 PQ 训练
    // ========================================================
    void train_global_pq() {
        std::cerr << "Training global PQ codebooks...\n";

        for (size_t m = 0; m < M; ++m) {
            train_pq_subspace(m);
        }
    }

    void train_pq_subspace(size_t m) {
        // 初始化子空间 codebook
        for (size_t c = 0; c < Ks; ++c) {
            size_t t = c * train_size / Ks;

            if (t >= train_size) {
                t = train_size - 1;
            }

            size_t id = train_id(t);

            const float* src =
                get_pq_vec(id) + m * subdim;

            float* dst = get_pq_centroid(m, c);

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

            for (size_t t = 0; t < train_size; ++t) {
                size_t id = train_id(t);

                const float* subvec =
                    get_pq_vec(id) + m * subdim;

                size_t cid = nearest_pq_centroid(
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
                float* centroid = get_pq_centroid(m, c);

                if (counts[c] == 0) {
                    size_t id = train_id((c + iter + 1) % train_size);

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

    size_t nearest_pq_centroid(
        const float* subvec,
        size_t m
    ) const {
        size_t best_c = 0;
        float best_dist = std::numeric_limits<float>::max();

        for (size_t c = 0; c < Ks; ++c) {
            float dist = ivfpq_l2_neon(
                subvec,
                get_pq_centroid(m, c),
                subdim
            );

            if (dist < best_dist) {
                best_dist = dist;
                best_c = c;
            }
        }

        return best_c;
    }

    void encode_all_base() {
        std::cerr << "Encoding base vectors by global PQ...\n";

        for (size_t i = 0; i < base_number; ++i) {
            const float* vec = get_pq_vec(i);
            uint8_t* code = codes.data() + i * M;

            for (size_t m = 0; m < M; ++m) {
                const float* subvec = vec + m * subdim;

                size_t cid = nearest_pq_centroid(
                    subvec,
                    m
                );

                code[m] = static_cast<uint8_t>(cid);
            }
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
            float dist = ivfpq_l2_neon(
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

        invlists.clear();
        invlists.resize(nlist);

        std::vector<size_t> counts(nlist, 0);

        for (size_t i = 0; i < base_number; ++i) {
            size_t cid = nearest_coarse_centroid(
                get_base_vec(i)
            );

            counts[cid]++;
        }

        for (size_t c = 0; c < nlist; ++c) {
            invlists[c].reserve(counts[c]);
        }

        for (size_t i = 0; i < base_number; ++i) {
            size_t cid = nearest_coarse_centroid(
                get_base_vec(i)
            );

            invlists[cid].push_back(
                static_cast<uint32_t>(i)
            );
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
            float dist = ivfpq_l2_neon(
                query,
                get_coarse_centroid(c),
                vecdim
            );

            ivfpq_update_top_probe(
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

    void build_pq_lut(
        const float* query,
        float* lut
    ) const {
        for (size_t m = 0; m < M; ++m) {
            const float* query_sub = query + m * subdim;

            for (size_t c = 0; c < Ks; ++c) {
                const float* centroid =
                    get_pq_centroid(m, c);

                float score = ivfpq_inner_product_neon_fma(
                    query_sub,
                    centroid,
                    subdim
                );

                lut[m * Ks + c] = score;
            }
        }
    }

    float scan_pq_code(
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

        std::cerr << "IVF-PQ list stats: "
                  << "min = " << min_size
                  << ", max = " << max_size
                  << ", avg = " << avg_size
                  << ", empty = " << empty_cnt
                  << "\n";
    }
};
