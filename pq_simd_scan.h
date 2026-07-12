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
#include <chrono>
#include <omp.h>
#include <pthread.h>

#include "flat_scan_simd.h"

// ============================================================
// PQ-SIMD cleaned version
// ============================================================

enum class PQInitMode {
    Uniform,
    KMeansPP
};

static inline float pq_hsum_f32(float32x4_t v) {
    return vaddvq_f32(v);
}

static inline float pq_l2_8_neon(
    const float* __restrict__ a,
    const float* __restrict__ b
) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    float32x4_t a0 = vld1q_f32(a);
    float32x4_t b0 = vld1q_f32(b);
    float32x4_t d0 = vsubq_f32(a0, b0);
    sum = vfmaq_f32(sum, d0, d0);

    float32x4_t a1 = vld1q_f32(a + 4);
    float32x4_t b1 = vld1q_f32(b + 4);
    float32x4_t d1 = vsubq_f32(a1, b1);
    sum = vfmaq_f32(sum, d1, d1);

    return pq_hsum_f32(sum);
}

static inline float pq_l2_generic(
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

    float result = pq_hsum_f32(sum);

    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}

template <typename ScoreT>
static inline void pq_recompute_worst_score(
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
static inline void pq_update_top_p(
    ScoreT score,
    uint32_t id,
    ScoreT* cand_score,
    uint32_t* cand_id,
    size_t capacity,
    size_t& cand_cnt,
    size_t& cand_worst_pos,
    ScoreT& cand_worst_score
) {
    if (capacity == 0) {
        return;
    }

    if (cand_cnt < capacity) {
        cand_score[cand_cnt] = score;
        cand_id[cand_cnt] = id;
        ++cand_cnt;

        if (cand_cnt == capacity) {
            pq_recompute_worst_score(
                cand_score,
                cand_cnt,
                cand_worst_pos,
                cand_worst_score
            );
        }
        return;
    }

    if (score > cand_worst_score) {
        cand_score[cand_worst_pos] = score;
        cand_id[cand_worst_pos] = id;

        pq_recompute_worst_score(
            cand_score,
            cand_cnt,
            cand_worst_pos,
            cand_worst_score
        );
    }
}

// 将 float LUT 对称量化为 int16_t。
// 同一 query 使用统一 scale，保证各子空间相对权重不被破坏。
static inline float pq_quantize_lut_i16(
    const float* __restrict__ lut,
    int16_t* __restrict__ lut_q,
    size_t len
) {
    float max_abs = 0.0f;

    for (size_t i = 0; i < len; ++i) {
        max_abs = std::max(max_abs, std::fabs(lut[i]));
    }

    if (max_abs <= 1e-12f) {
        std::fill(lut_q, lut_q + len, static_cast<int16_t>(0));
        return 1.0f;
    }

    float scale = 32767.0f / max_abs;

    for (size_t i = 0; i < len; ++i) {
        int q = static_cast<int>(std::round(lut[i] * scale));
        q = std::max(-32767, std::min(32767, q));
        lut_q[i] = static_cast<int16_t>(q);
    }

    return scale;
}

static inline int32_t pq_scan_code_i16_generic(
    const uint8_t* __restrict__ code,
    const int16_t* __restrict__ lut_q,
    size_t M,
    size_t Ks
) {
    int32_t score = 0;

    for (size_t m = 0; m < M; ++m) {
        score += static_cast<int32_t>(
            lut_q[m * Ks + static_cast<size_t>(code[m])]
        );
    }

    return score;
}

template <size_t MFIX>
static inline int32_t pq_scan_code_i16_fixed(
    const uint8_t* __restrict__ code,
    const int16_t* __restrict__ lut_q,
    size_t Ks
) {
    int32_t score = 0;

    for (size_t m = 0; m < MFIX; ++m) {
        score += static_cast<int32_t>(
            lut_q[m * Ks + static_cast<size_t>(code[m])]
        );
    }

    return score;
}

static inline int32_t pq_scan_code_i16(
    const uint8_t* __restrict__ code,
    const int16_t* __restrict__ lut_q,
    size_t M,
    size_t Ks
) {
    switch (M) {
        case 4:  return pq_scan_code_i16_fixed<4>(code, lut_q, Ks);
        case 8:  return pq_scan_code_i16_fixed<8>(code, lut_q, Ks);
        case 12: return pq_scan_code_i16_fixed<12>(code, lut_q, Ks);
        case 16: return pq_scan_code_i16_fixed<16>(code, lut_q, Ks);
        default: return pq_scan_code_i16_generic(code, lut_q, M, Ks);
    }
}

static inline void pq_scan_code_i16_batch4_generic(
    const uint8_t* __restrict__ codes_ptr,
    const int16_t* __restrict__ lut_q,
    size_t M,
    size_t Ks,
    int32_t& s0,
    int32_t& s1,
    int32_t& s2,
    int32_t& s3
) {
    const uint8_t* code0 = codes_ptr;
    const uint8_t* code1 = codes_ptr + M;
    const uint8_t* code2 = codes_ptr + 2 * M;
    const uint8_t* code3 = codes_ptr + 3 * M;

    s0 = s1 = s2 = s3 = 0;

    for (size_t m = 0; m < M; ++m) {
        const int16_t* lut_m = lut_q + m * Ks;

        s0 += static_cast<int32_t>(lut_m[static_cast<size_t>(code0[m])]);
        s1 += static_cast<int32_t>(lut_m[static_cast<size_t>(code1[m])]);
        s2 += static_cast<int32_t>(lut_m[static_cast<size_t>(code2[m])]);
        s3 += static_cast<int32_t>(lut_m[static_cast<size_t>(code3[m])]);
    }
}

template <size_t MFIX>
static inline void pq_scan_code_i16_batch4_fixed(
    const uint8_t* __restrict__ codes_ptr,
    const int16_t* __restrict__ lut_q,
    size_t Ks,
    int32_t& s0,
    int32_t& s1,
    int32_t& s2,
    int32_t& s3
) {
    const uint8_t* code0 = codes_ptr;
    const uint8_t* code1 = codes_ptr + MFIX;
    const uint8_t* code2 = codes_ptr + 2 * MFIX;
    const uint8_t* code3 = codes_ptr + 3 * MFIX;

    s0 = s1 = s2 = s3 = 0;

    for (size_t m = 0; m < MFIX; ++m) {
        const int16_t* lut_m = lut_q + m * Ks;

        s0 += static_cast<int32_t>(lut_m[static_cast<size_t>(code0[m])]);
        s1 += static_cast<int32_t>(lut_m[static_cast<size_t>(code1[m])]);
        s2 += static_cast<int32_t>(lut_m[static_cast<size_t>(code2[m])]);
        s3 += static_cast<int32_t>(lut_m[static_cast<size_t>(code3[m])]);
    }
}

static inline void pq_scan_code_i16_batch4(
    const uint8_t* __restrict__ codes_ptr,
    const int16_t* __restrict__ lut_q,
    size_t M,
    size_t Ks,
    int32_t& s0,
    int32_t& s1,
    int32_t& s2,
    int32_t& s3
) {
    switch (M) {
        case 4:
            pq_scan_code_i16_batch4_fixed<4>(codes_ptr, lut_q, Ks, s0, s1, s2, s3);
            return;
        case 8:
            pq_scan_code_i16_batch4_fixed<8>(codes_ptr, lut_q, Ks, s0, s1, s2, s3);
            return;
        case 12:
            pq_scan_code_i16_batch4_fixed<12>(codes_ptr, lut_q, Ks, s0, s1, s2, s3);
            return;
        case 16:
            pq_scan_code_i16_batch4_fixed<16>(codes_ptr, lut_q, Ks, s0, s1, s2, s3);
            return;
        default:
            pq_scan_code_i16_batch4_generic(codes_ptr, lut_q, M, Ks, s0, s1, s2, s3);
            return;
    }
}

struct PQSearchProfile {
    double lut_build_us = 0.0;   // LUT 构建
    double lut_quant_us = 0.0;   // LUT int16 量化
    double pq_scan_us = 0.0;     // PQ code 查表粗排
    double rerank_us = 0.0;      // float 精排
    double output_us = 0.0;      // 构造返回结果
    double total_us = 0.0;       // 总时间

    void clear() {
        lut_build_us = 0.0;
        lut_quant_us = 0.0;
        pq_scan_us = 0.0;
        rerank_us = 0.0;
        output_us = 0.0;
        total_us = 0.0;
    }

    void add(const PQSearchProfile& other) {
        lut_build_us += other.lut_build_us;
        lut_quant_us += other.lut_quant_us;
        pq_scan_us += other.pq_scan_us;
        rerank_us += other.rerank_us;
        output_us += other.output_us;
        total_us += other.total_us;
    }
};

static inline double pq_now_us() {
    using Clock = std::chrono::high_resolution_clock;
    auto now = Clock::now();
    return std::chrono::duration<double, std::micro>(
        now.time_since_epoch()
    ).count();
}

struct PQScanPthreadParam {
    int tid;

    const uint8_t* codes;
    const int16_t* lut_q;

    size_t begin;
    size_t end;

    size_t M;
    size_t Ks;
    size_t local_p;

    int32_t* local_score;
    uint32_t* local_id;
    size_t* local_cnt;
};

static void* pq_scan_pthread_worker(void* arg) {
    PQScanPthreadParam* param =
        static_cast<PQScanPthreadParam*>(arg);

    const uint8_t* codes = param->codes;
    const int16_t* lut_q = param->lut_q;

    size_t begin = param->begin;
    size_t end = param->end;

    size_t M = param->M;
    size_t Ks = param->Ks;
    size_t local_p = param->local_p;

    int32_t* score_t = param->local_score;
    uint32_t* id_t = param->local_id;

    constexpr size_t PREFETCH_DIST = 16;

    size_t cnt = 0;
    size_t worst_pos = 0;
    int32_t worst_score = 0;

    size_t i = begin;

    // 每次处理 4 条连续 PQ code
    for (; i + 4 <= end; i += 4) {
        if (i + PREFETCH_DIST < end) {
            __builtin_prefetch(
                &codes[(i + PREFETCH_DIST) * M],
                0,
                1
            );
        }

        const uint8_t* codes_ptr = &codes[i * M];

        int32_t s0, s1, s2, s3;

        pq_scan_code_i16_batch4(
            codes_ptr,
            lut_q,
            M,
            Ks,
            s0,
            s1,
            s2,
            s3
        );

        pq_update_top_p(
            s0,
            static_cast<uint32_t>(i + 0),
            score_t,
            id_t,
            local_p,
            cnt,
            worst_pos,
            worst_score
        );

        pq_update_top_p(
            s1,
            static_cast<uint32_t>(i + 1),
            score_t,
            id_t,
            local_p,
            cnt,
            worst_pos,
            worst_score
        );

        pq_update_top_p(
            s2,
            static_cast<uint32_t>(i + 2),
            score_t,
            id_t,
            local_p,
            cnt,
            worst_pos,
            worst_score
        );

        pq_update_top_p(
            s3,
            static_cast<uint32_t>(i + 3),
            score_t,
            id_t,
            local_p,
            cnt,
            worst_pos,
            worst_score
        );
    }

    // 处理当前线程区间尾部不足 4 条的 PQ code
    for (; i < end; ++i) {
        if (i + PREFETCH_DIST < end) {
            __builtin_prefetch(
                &codes[(i + PREFETCH_DIST) * M],
                0,
                1
            );
        }

        int32_t score = pq_scan_code_i16(
            &codes[i * M],
            lut_q,
            M,
            Ks
        );

        pq_update_top_p(
            score,
            static_cast<uint32_t>(i),
            score_t,
            id_t,
            local_p,
            cnt,
            worst_pos,
            worst_score
        );
    }

    *(param->local_cnt) = cnt;

    return nullptr;
}

class PQIndexSIMD {
public:
    PQIndexSIMD(
        float* base,
        size_t base_number,
        size_t vecdim,
        size_t M = 12,
        size_t Ks = 64,
        size_t train_size = 10000,
        size_t kmeans_iters = 6,
        PQInitMode init_mode = PQInitMode::Uniform
    )
        : base_float(base),
          base_number(base_number),
          vecdim(vecdim),
          M(M),
          Ks(Ks),
          subdim(0),
          train_size(train_size),
          kmeans_iters(kmeans_iters),
          init_mode(init_mode),
          centroid_blocks(0)
    {
        normalize_params();
        centroids.resize(this->M * this->Ks * subdim);
        codes.resize(this->base_number * this->M);
        build();
    }

    std::priority_queue<std::pair<float, uint32_t> > search_omp_scan(
        float* query,
        
        size_t k,
        size_t rerank_p,
        size_t local_p,
        int num_threads
    ) {
        constexpr size_t MAX_P = 8192;
        constexpr size_t PREFETCH_DIST = 16;

        std::priority_queue<std::pair<float, uint32_t> > result;

        if (k == 0 || base_number == 0) {
            return result;
        }

        k = std::min(k, static_cast<size_t>(128));
        k = std::min(k, base_number);

        rerank_p = std::max(rerank_p, k);
        rerank_p = std::min(rerank_p, base_number);
        rerank_p = std::min(rerank_p, MAX_P);

        if (local_p == 0) {
            local_p = rerank_p;
        }

        local_p = std::min(local_p, base_number);

        if (num_threads <= 1) {
            return search(query, k, rerank_p);
        }

        if (static_cast<size_t>(num_threads) > base_number) {
            num_threads = static_cast<int>(base_number);
        }

        // ====================================================
        // 1. 单线程构建 Query LUT，并量化为 int16 LUT
        // ====================================================
        std::vector<float> lut(M * Ks);
        std::vector<int16_t> lut_q(M * Ks);

        build_lut_xc4(query, lut.data());

        pq_quantize_lut_i16(
            lut.data(),
            lut_q.data(),
            M * Ks
        );

        // ====================================================
        // 2. OpenMP 并行 PQ code 查表扫描
        // 每个线程维护自己的 local Top-p
        // ====================================================
        std::vector<int32_t> local_scores(
            static_cast<size_t>(num_threads) * local_p
        );

        std::vector<uint32_t> local_ids(
            static_cast<size_t>(num_threads) * local_p
        );

        std::vector<size_t> local_cnt(
            static_cast<size_t>(num_threads),
            0
        );

        size_t block =
            (base_number + static_cast<size_t>(num_threads) - 1)
            / static_cast<size_t>(num_threads);

        omp_set_dynamic(0);

        #pragma omp parallel num_threads(num_threads)
        {
            int tid = omp_get_thread_num();

            size_t begin = static_cast<size_t>(tid) * block;
            size_t end = std::min(begin + block, base_number);

            int32_t* score_t =
                local_scores.data() + static_cast<size_t>(tid) * local_p;

            uint32_t* id_t =
                local_ids.data() + static_cast<size_t>(tid) * local_p;

            size_t cnt = 0;
            size_t worst_pos = 0;
            int32_t worst_score = 0;

            size_t i = begin;

            // 每次处理 4 条连续 PQ code
            for (; i + 4 <= end; i += 4) {
                if (i + PREFETCH_DIST < end) {
                    __builtin_prefetch(
                        &codes[(i + PREFETCH_DIST) * M],
                        0,
                        1
                    );
                }

                const uint8_t* codes_ptr = &codes[i * M];

                int32_t s0, s1, s2, s3;

                pq_scan_code_i16_batch4(
                    codes_ptr,
                    lut_q.data(),
                    M,
                    Ks,
                    s0,
                    s1,
                    s2,
                    s3
                );

                pq_update_top_p(
                    s0,
                    static_cast<uint32_t>(i + 0),
                    score_t,
                    id_t,
                    local_p,
                    cnt,
                    worst_pos,
                    worst_score
                );

                pq_update_top_p(
                    s1,
                    static_cast<uint32_t>(i + 1),
                    score_t,
                    id_t,
                    local_p,
                    cnt,
                    worst_pos,
                    worst_score
                );

                pq_update_top_p(
                    s2,
                    static_cast<uint32_t>(i + 2),
                    score_t,
                    id_t,
                    local_p,
                    cnt,
                    worst_pos,
                    worst_score
                );

                pq_update_top_p(
                    s3,
                    static_cast<uint32_t>(i + 3),
                    score_t,
                    id_t,
                    local_p,
                    cnt,
                    worst_pos,
                    worst_score
                );
            }

            // 处理当前线程区间尾部不足 4 条的 PQ code
            for (; i < end; ++i) {
                if (i + PREFETCH_DIST < end) {
                    __builtin_prefetch(
                        &codes[(i + PREFETCH_DIST) * M],
                        0,
                        1
                    );
                }

                int32_t score = pq_scan_code_i16(
                    &codes[i * M],
                    lut_q.data(),
                    M,
                    Ks
                );

                pq_update_top_p(
                    score,
                    static_cast<uint32_t>(i),
                    score_t,
                    id_t,
                    local_p,
                    cnt,
                    worst_pos,
                    worst_score
                );
            }

            local_cnt[static_cast<size_t>(tid)] = cnt;
        }

        // ====================================================
        // 3. reduce：local Top-p -> global Top-rerank_p
        // ====================================================
        std::vector<int32_t> cand_score(rerank_p);
        std::vector<uint32_t> cand_id(rerank_p);

        size_t cand_cnt = 0;
        size_t cand_worst_pos = 0;
        int32_t cand_worst_score = 0;

        for (int t = 0; t < num_threads; ++t) {
            int32_t* score_t =
                local_scores.data() + static_cast<size_t>(t) * local_p;

            uint32_t* id_t =
                local_ids.data() + static_cast<size_t>(t) * local_p;

            for (size_t j = 0; j < local_cnt[static_cast<size_t>(t)]; ++j) {
                pq_update_top_p(
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

        // ====================================================
        // 4. 精排：对 global Top-rerank_p 候选使用原始 float + NEON-FMA
        // ====================================================
        std::vector<float> best_score(k);
        std::vector<uint32_t> best_id(k);

        size_t best_cnt = 0;
        size_t best_worst_pos = 0;
        float best_worst_score = 0.0f;

        for (size_t j = 0; j < cand_cnt; ++j) {
            uint32_t id = cand_id[j];

            if (j + 4 < cand_cnt) {
                uint32_t next_id = cand_id[j + 4];

                __builtin_prefetch(
                    base_float + static_cast<size_t>(next_id) * vecdim,
                    0,
                    1
                );
            }

            const float* base_vec =
                base_float + static_cast<size_t>(id) * vecdim;

            float score = inner_product_neon16_fma(
                base_vec,
                query,
                vecdim
            );

            pq_update_top_p(
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

        // ====================================================
        // 5. 返回 main.cc 原有格式：priority_queue<pair<dis,id>>
        // ====================================================
        for (size_t j = 0; j < best_cnt; ++j) {
            result.push({
                1.0f - best_score[j],
                best_id[j]
            });
        }

        return result;
    }

    std::priority_queue<std::pair<float, uint32_t> > search_pthread_scan(
        float* query,
        size_t k,
        size_t rerank_p,
        size_t local_p,
        int num_threads
    ) {
        constexpr size_t MAX_P = 8192;

        std::priority_queue<std::pair<float, uint32_t> > result;

        if (k == 0 || base_number == 0) {
            return result;
        }

        k = std::min(k, static_cast<size_t>(128));
        k = std::min(k, base_number);

        rerank_p = std::max(rerank_p, k);
        rerank_p = std::min(rerank_p, base_number);
        rerank_p = std::min(rerank_p, MAX_P);

        // local_p = 0 时默认等于 rerank_p
        if (local_p == 0) {
            local_p = rerank_p;
        }

        local_p = std::min(local_p, base_number);

        if (num_threads <= 1) {
            return search(query, k, rerank_p);
        }

        if (static_cast<size_t>(num_threads) > base_number) {
            num_threads = static_cast<int>(base_number);
        }

        // ====================================================
        // 1. 单线程构建 Query LUT，并量化为 int16 LUT
        // ====================================================
        std::vector<float> lut(M * Ks);
        std::vector<int16_t> lut_q(M * Ks);

        build_lut_xc4(
            query,
            lut.data()
        );

        pq_quantize_lut_i16(
            lut.data(),
            lut_q.data(),
            M * Ks
        );

        // ====================================================
        // 2. Pthread 并行 PQ code 查表扫描
        // 每个线程维护自己的 local Top-p
        // ====================================================
        std::vector<int32_t> local_scores(
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
        std::vector<PQScanPthreadParam> params(num_threads);

        size_t block =
            (base_number + static_cast<size_t>(num_threads) - 1)
            / static_cast<size_t>(num_threads);

        for (int t = 0; t < num_threads; ++t) {
            size_t begin = static_cast<size_t>(t) * block;
            size_t end = std::min(begin + block, base_number);

            params[t].tid = t;
            params[t].codes = codes.data();
            params[t].lut_q = lut_q.data();
            params[t].begin = begin;
            params[t].end = end;
            params[t].M = M;
            params[t].Ks = Ks;
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
                pq_scan_pthread_worker,
                &params[t]
            );
        }

        for (int t = 0; t < num_threads; ++t) {
            pthread_join(
                threads[t],
                nullptr
            );
        }

        // ====================================================
        // 3. reduce：local Top-p -> global Top-rerank_p
        // ====================================================
        std::vector<int32_t> cand_score(rerank_p);
        std::vector<uint32_t> cand_id(rerank_p);

        size_t cand_cnt = 0;
        size_t cand_worst_pos = 0;
        int32_t cand_worst_score = 0;

        for (int t = 0; t < num_threads; ++t) {
            int32_t* score_t =
                local_scores.data() + static_cast<size_t>(t) * local_p;

            uint32_t* id_t =
                local_ids.data() + static_cast<size_t>(t) * local_p;

            for (size_t j = 0; j < local_cnt[static_cast<size_t>(t)]; ++j) {
                pq_update_top_p(
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

        // ====================================================
        // 4. 精排：对 global Top-rerank_p 候选使用原始 float + NEON-FMA
        // ====================================================
        std::vector<float> best_score(k);
        std::vector<uint32_t> best_id(k);

        size_t best_cnt = 0;
        size_t best_worst_pos = 0;
        float best_worst_score = 0.0f;

        for (size_t j = 0; j < cand_cnt; ++j) {
            uint32_t id = cand_id[j];

            if (j + 4 < cand_cnt) {
                uint32_t next_id = cand_id[j + 4];

                __builtin_prefetch(
                    base_float + static_cast<size_t>(next_id) * vecdim,
                    0,
                    1
                );
            }

            const float* base_vec =
                base_float + static_cast<size_t>(id) * vecdim;

            float score = inner_product_neon16_fma(
                base_vec,
                query,
                vecdim
            );

            pq_update_top_p(
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

        // ====================================================
        // 5. 返回 main.cc 原有格式：priority_queue<pair<dis,id>>
        // ====================================================
        for (size_t j = 0; j < best_cnt; ++j) {
            result.push({
                1.0f - best_score[j],
                best_id[j]
            });
        }

        return result;
    }

    /* profiling 版本，返回结果不变，增加了每个阶段的耗时统计
    std::priority_queue<std::pair<float, uint32_t> > search(
        float* query,
        size_t k,
        size_t rerank_p,
        PQSearchProfile* profile = nullptr
    ) {
        constexpr size_t MAX_P = 8192;
        constexpr size_t PREFETCH_DIST = 16;

        PQSearchProfile local_profile;
        double total_t0 = pq_now_us();

        std::priority_queue<std::pair<float, uint32_t> > result;

        if (k == 0 || base_number == 0) {
            if (profile != nullptr) {
                *profile = local_profile;
            }
            return result;
        }

        // 参数修正
        k = std::min(k, static_cast<size_t>(128));
        k = std::min(k, base_number);

        rerank_p = std::max(rerank_p, k);
        rerank_p = std::min(rerank_p, base_number);
        rerank_p = std::min(rerank_p, MAX_P);

        // ====================================================
        // 1. 构建 Query LUT
        // ====================================================
        std::vector<float> lut(M * Ks);

        double t0 = pq_now_us();

        // build_lut_xc4(
        //     query,
        //     lut.data()
        // );

        build_lut_xc4_omp(
            query,
            lut.data(),
            4
        );

        double t1 = pq_now_us();
        local_profile.lut_build_us = t1 - t0;

        // ====================================================
        // 2. float LUT -> int16 LUT
        // ====================================================
        std::vector<int16_t> lut_q(M * Ks);

        t0 = pq_now_us();

        pq_quantize_lut_i16(
            lut.data(),
            lut_q.data(),
            M * Ks
        );

        t1 = pq_now_us();
        local_profile.lut_quant_us = t1 - t0;

        // ====================================================
        // 3. PQ 粗排：int16 LUT 查表累加，维护 Top-p
        // ====================================================
        std::vector<int32_t> cand_score(rerank_p);
        std::vector<uint32_t> cand_id(rerank_p);

        size_t cand_cnt = 0;
        size_t cand_worst_pos = 0;
        int32_t cand_worst_score = 0;

        size_t i = 0;

        t0 = pq_now_us();

        for (; i + 4 <= base_number; i += 4) {
            if (i + PREFETCH_DIST < base_number) {
                __builtin_prefetch(
                    &codes[(i + PREFETCH_DIST) * M],
                    0,
                    1
                );
            }

            const uint8_t* codes_ptr = &codes[i * M];

            int32_t s0, s1, s2, s3;

            pq_scan_code_i16_batch4(
                codes_ptr,
                lut_q.data(),
                M,
                Ks,
                s0,
                s1,
                s2,
                s3
            );

            pq_update_top_p(
                s0,
                static_cast<uint32_t>(i + 0),
                cand_score.data(),
                cand_id.data(),
                rerank_p,
                cand_cnt,
                cand_worst_pos,
                cand_worst_score
            );

            pq_update_top_p(
                s1,
                static_cast<uint32_t>(i + 1),
                cand_score.data(),
                cand_id.data(),
                rerank_p,
                cand_cnt,
                cand_worst_pos,
                cand_worst_score
            );

            pq_update_top_p(
                s2,
                static_cast<uint32_t>(i + 2),
                cand_score.data(),
                cand_id.data(),
                rerank_p,
                cand_cnt,
                cand_worst_pos,
                cand_worst_score
            );

            pq_update_top_p(
                s3,
                static_cast<uint32_t>(i + 3),
                cand_score.data(),
                cand_id.data(),
                rerank_p,
                cand_cnt,
                cand_worst_pos,
                cand_worst_score
            );
        }

        // 处理不足 4 条的尾部
        for (; i < base_number; ++i) {
            if (i + PREFETCH_DIST < base_number) {
                __builtin_prefetch(
                    &codes[(i + PREFETCH_DIST) * M],
                    0,
                    1
                );
            }

            int32_t score = pq_scan_code_i16(
                &codes[i * M],
                lut_q.data(),
                M,
                Ks
            );

            pq_update_top_p(
                score,
                static_cast<uint32_t>(i),
                cand_score.data(),
                cand_id.data(),
                rerank_p,
                cand_cnt,
                cand_worst_pos,
                cand_worst_score
            );
        }

        t1 = pq_now_us();
        local_profile.pq_scan_us = t1 - t0;

        // ====================================================
        // 4. 精排：对 Top-p 候选使用原始 float + NEON-FMA
        // ====================================================
        std::vector<float> best_score(k);
        std::vector<uint32_t> best_id(k);

        size_t best_cnt = 0;
        size_t best_worst_pos = 0;
        float best_worst_score = 0.0f;

        t0 = pq_now_us();

        for (size_t j = 0; j < cand_cnt; ++j) {
            uint32_t id = cand_id[j];

            if (j + 4 < cand_cnt) {
                uint32_t next_id = cand_id[j + 4];

                __builtin_prefetch(
                    base_float + static_cast<size_t>(next_id) * vecdim,
                    0,
                    1
                );
            }

            const float* base_vec =
                base_float + static_cast<size_t>(id) * vecdim;

            float score = inner_product_neon16_fma(
                base_vec,
                query,
                vecdim
            );

            pq_update_top_p(
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

        t1 = pq_now_us();
        local_profile.rerank_us = t1 - t0;

        // ====================================================
        // 5. 构造返回结果
        // ====================================================
        t0 = pq_now_us();

        for (size_t j = 0; j < best_cnt; ++j) {
            result.push({
                1.0f - best_score[j],
                best_id[j]
            });
        }

        t1 = pq_now_us();
        local_profile.output_us = t1 - t0;

        local_profile.total_us = pq_now_us() - total_t0;

        if (profile != nullptr) {
            *profile = local_profile;
        }

        return result;
    }
    */

    // 无profiling
    
    std::priority_queue<std::pair<float, uint32_t> > search(
        float* query,
        size_t k,
        size_t rerank_p
    ) {
        constexpr size_t MAX_P = 8192;
        constexpr size_t PREFETCH_DIST = 16;

        std::priority_queue<std::pair<float, uint32_t> > result;
        if (k == 0 || base_number == 0) {
            return result;
        }

        k = std::min(k, static_cast<size_t>(128));
        rerank_p = std::max(rerank_p, k);
        rerank_p = std::min(rerank_p, base_number);
        rerank_p = std::min(rerank_p, MAX_P);

        // 1. 构建 LUT，并量化为 int16 LUT
        std::vector<float> lut(M * Ks);
        std::vector<int16_t> lut_q(M * Ks);

        build_lut_xc4(query, lut.data());
        
        // build_lut_xc4_omp(
        //     query,
        //     lut.data(),
        //     4
        // );


        pq_quantize_lut_i16(lut.data(), lut_q.data(), M * Ks);

        // 2. PQ 粗排：int16 LUT 查表累加，维护 Top-p
        std::vector<int32_t> cand_score(rerank_p);
        std::vector<uint32_t> cand_id(rerank_p);

        size_t cand_cnt = 0;
        size_t cand_worst_pos = 0;
        int32_t cand_worst_score = 0;


        size_t i = 0;
        for (; i + 4 <= base_number; i += 4) {
            if (i + PREFETCH_DIST < base_number) {
                __builtin_prefetch(&codes[(i + PREFETCH_DIST) * M], 0, 1);
            }

            const uint8_t* codes_ptr = &codes[i * M];

            int32_t s0, s1, s2, s3;
            pq_scan_code_i16_batch4(
                codes_ptr,
                lut_q.data(),
                M,
                Ks,
                s0, s1, s2, s3
            );

            pq_update_top_p(s0, static_cast<uint32_t>(i + 0),
                            cand_score.data(), cand_id.data(), rerank_p,
                            cand_cnt, cand_worst_pos, cand_worst_score);
            pq_update_top_p(s1, static_cast<uint32_t>(i + 1),
                            cand_score.data(), cand_id.data(), rerank_p,
                            cand_cnt, cand_worst_pos, cand_worst_score);
            pq_update_top_p(s2, static_cast<uint32_t>(i + 2),
                            cand_score.data(), cand_id.data(), rerank_p,
                            cand_cnt, cand_worst_pos, cand_worst_score);
            pq_update_top_p(s3, static_cast<uint32_t>(i + 3),
                            cand_score.data(), cand_id.data(), rerank_p,
                            cand_cnt, cand_worst_pos, cand_worst_score);
        }

        for (; i < base_number; ++i) {
            if (i + PREFETCH_DIST < base_number) {
                __builtin_prefetch(&codes[(i + PREFETCH_DIST) * M], 0, 1);
            }

            int32_t score = pq_scan_code_i16(
                &codes[i * M],
                lut_q.data(),
                M,
                Ks
            );

            pq_update_top_p(score, static_cast<uint32_t>(i),
                            cand_score.data(), cand_id.data(), rerank_p,
                            cand_cnt, cand_worst_pos, cand_worst_score);
        }

        // 3. 精排：对 Top-p 候选使用原始 float + NEON-FMA
        std::vector<float> best_score(k);
        std::vector<uint32_t> best_id(k);

        size_t best_cnt = 0;
        size_t best_worst_pos = 0;
        float best_worst_score = 0.0f;

        for (size_t j = 0; j < cand_cnt; ++j) {
            uint32_t id = cand_id[j];

            if (j + 4 < cand_cnt) {
                uint32_t next_id = cand_id[j + 4];
                __builtin_prefetch(
                    base_float + static_cast<size_t>(next_id) * vecdim,
                    0,
                    1
                );
            }

            const float* base_vec =
                base_float + static_cast<size_t>(id) * vecdim;

            float score = inner_product_neon16_fma(
                base_vec,
                query,
                vecdim
            );

            pq_update_top_p(score, id,
                            best_score.data(), best_id.data(), k,
                            best_cnt, best_worst_pos, best_worst_score);
        }

        for (size_t j = 0; j < best_cnt; ++j) {
            result.push({1.0f - best_score[j], best_id[j]});
        }

        return result;
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

    PQInitMode get_init_mode() const {
        return init_mode;
    }

private:
    float* base_float;
    size_t base_number;
    size_t vecdim;

    size_t M;
    size_t Ks;
    size_t subdim;
    size_t train_size;
    size_t kmeans_iters;
    PQInitMode init_mode;

    std::vector<float> centroids;
    std::vector<uint8_t> codes;

    // 跨 centroid SIMD 使用的转置 centroid 布局:
    // centroids_xc4[m][centroid_block][dim][lane]
    std::vector<float> centroids_xc4;
    size_t centroid_blocks;

    void normalize_params() {
        if (M == 0) {
            M = 12;
        }

        if (vecdim % M != 0) {
            std::cerr << "PQ warning: vecdim % M != 0, reset M to 12\n";
            M = 12;
        }

        if (vecdim % M != 0) {
            std::cerr << "PQ error: vecdim cannot be divided by M\n";
            M = 1;
        }

        subdim = vecdim / M;

        if (Ks > 256) {
            std::cerr << "PQ warning: uint8_t code only supports Ks <= 256, clamp to 256\n";
            Ks = 256;
        }
        if (Ks < 2) {
            Ks = 2;
        }

        train_size = std::min(train_size, base_number);
        if (train_size < Ks) {
            std::cerr << "PQ warning: train_size < Ks, reset if possible\n";
            train_size = (base_number >= Ks) ? Ks : base_number;
        }
    }

    void build() {
        std::cerr << "PQ build start. M = " << M
                  << ", Ks = " << Ks
                  << ", subdim = " << subdim
                  << ", train_size = " << train_size
                  << ", iters = " << kmeans_iters
                  << ", init = "
                  << (init_mode == PQInitMode::KMeansPP ? "KMeans++" : "Uniform")
                  << "\n";

        train_codebooks();
        build_centroids_xc4();
        encode_base();

        std::cerr << "PQ build done.\n";
    }

    float* get_centroid(size_t m, size_t c) {
        return &centroids[(m * Ks + c) * subdim];
    }

    const float* get_centroid(size_t m, size_t c) const {
        return &centroids[(m * Ks + c) * subdim];
    }

    const float* get_base_subvec(size_t id, size_t m) const {
        return base_float + id * vecdim + m * subdim;
    }

    size_t train_id(size_t t) const {
        if (train_size <= 1) {
            return 0;
        }

        size_t id = t * base_number / train_size;
        return std::min(id, base_number - 1);
    }

    float sub_l2(
        const float* a,
        const float* b
    ) const {
        if (subdim == 8) {
            return pq_l2_8_neon(a, b);
        }
        return pq_l2_generic(a, b, subdim);
    }

    uint8_t nearest_centroid_l2(
        const float* subvec,
        size_t m
    ) const {
        size_t best_c = 0;
        float best_dis = std::numeric_limits<float>::max();

        for (size_t c = 0; c < Ks; ++c) {
            float dis = sub_l2(subvec, get_centroid(m, c));
            if (dis < best_dis) {
                best_dis = dis;
                best_c = c;
            }
        }

        return static_cast<uint8_t>(best_c);
    }

    void init_centroids_uniform(size_t m) {
        for (size_t c = 0; c < Ks; ++c) {
            size_t t = std::min(c * train_size / Ks, train_size - 1);
            size_t id = train_id(t);

            const float* src = get_base_subvec(id, m);
            float* dst = get_centroid(m, c);

            std::copy(src, src + subdim, dst);
        }
    }

    void init_centroids_kmeanspp(size_t m) {
        std::mt19937 rng(2026 + static_cast<unsigned int>(m));

        std::vector<float> min_dist(
            train_size,
            std::numeric_limits<float>::max()
        );

        std::uniform_int_distribution<size_t> first_dist(0, train_size - 1);
        size_t first_id = train_id(first_dist(rng));

        std::copy(
            get_base_subvec(first_id, m),
            get_base_subvec(first_id, m) + subdim,
            get_centroid(m, 0)
        );

        for (size_t c = 1; c < Ks; ++c) {
            const float* last_centroid = get_centroid(m, c - 1);
            float total_dist = 0.0f;

            for (size_t t = 0; t < train_size; ++t) {
                size_t id = train_id(t);
                const float* subvec = get_base_subvec(id, m);

                float dis = sub_l2(subvec, last_centroid);
                if (dis < min_dist[t]) {
                    min_dist[t] = dis;
                }
                total_dist += min_dist[t];
            }

            size_t chosen_t = 0;

            if (total_dist <= 1e-12f) {
                chosen_t = c % train_size;
            } else {
                std::uniform_real_distribution<float> prob_dist(0.0f, total_dist);
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
                get_base_subvec(chosen_id, m),
                get_base_subvec(chosen_id, m) + subdim,
                get_centroid(m, c)
            );
        }
    }

    void init_centroids(size_t m) {
        if (init_mode == PQInitMode::KMeansPP) {
            init_centroids_kmeanspp(m);
        } else {
            init_centroids_uniform(m);
        }
    }

    void train_codebooks() {
        std::vector<float> sums(Ks * subdim);
        std::vector<size_t> counts(Ks);

        for (size_t m = 0; m < M; ++m) {
            init_centroids(m);

            for (size_t iter = 0; iter < kmeans_iters; ++iter) {
                std::fill(sums.begin(), sums.end(), 0.0f);
                std::fill(counts.begin(), counts.end(), 0);

                for (size_t t = 0; t < train_size; ++t) {
                    size_t id = train_id(t);
                    const float* subvec = get_base_subvec(id, m);

                    size_t cid = static_cast<size_t>(
                        nearest_centroid_l2(subvec, m)
                    );

                    float* sum_vec = &sums[cid * subdim];

                    for (size_t d = 0; d < subdim; ++d) {
                        sum_vec[d] += subvec[d];
                    }

                    counts[cid]++;
                }

                for (size_t c = 0; c < Ks; ++c) {
                    float* centroid = get_centroid(m, c);

                    if (counts[c] == 0) {
                        size_t id = train_id((c + iter + 1) % train_size);
                        std::copy(
                            get_base_subvec(id, m),
                            get_base_subvec(id, m) + subdim,
                            centroid
                        );
                    } else {
                        float inv_count =
                            1.0f / static_cast<float>(counts[c]);

                        float* sum_vec = &sums[c * subdim];

                        for (size_t d = 0; d < subdim; ++d) {
                            centroid[d] = sum_vec[d] * inv_count;
                        }
                    }
                }
            }
        }
    }

    void encode_base() {
        constexpr size_t PREFETCH_DIST = 16;

        for (size_t i = 0; i < base_number; ++i) {
            if (i + PREFETCH_DIST < base_number) {
                __builtin_prefetch(
                    base_float + (i + PREFETCH_DIST) * vecdim,
                    0,
                    1
                );
            }

            uint8_t* code = &codes[i * M];

            for (size_t m = 0; m < M; ++m) {
                code[m] = nearest_centroid_l2(get_base_subvec(i, m), m);
            }
        }
    }

    void build_centroids_xc4() {
        centroid_blocks = (Ks + 3) / 4;

        centroids_xc4.assign(
            M * centroid_blocks * subdim * 4,
            0.0f
        );

        for (size_t m = 0; m < M; ++m) {
            for (size_t cb = 0; cb < centroid_blocks; ++cb) {
                for (size_t d = 0; d < subdim; ++d) {
                    for (size_t lane = 0; lane < 4; ++lane) {
                        size_t c = cb * 4 + lane;

                        if (c < Ks) {
                            centroids_xc4[
                                ((m * centroid_blocks + cb) * subdim + d) * 4 + lane
                            ] = centroids[(m * Ks + c) * subdim + d];
                        }
                    }
                }
            }
        }
    }

    void build_lut_xc4(
        const float* __restrict__ query,
        float* __restrict__ lut
    ) const {
        constexpr size_t MAX_SUBDIM = 64;
        float32x4_t qv_cache[MAX_SUBDIM];

        for (size_t m = 0; m < M; ++m) {
            const float* query_sub = query + m * subdim;
            float* lut_m = lut + m * Ks;

            bool use_cache = (subdim <= MAX_SUBDIM);
            if (use_cache) {
                for (size_t d = 0; d < subdim; ++d) {
                    qv_cache[d] = vdupq_n_f32(query_sub[d]);
                }
            }

            for (size_t cb = 0; cb < centroid_blocks; ++cb) {
                float32x4_t sum = vdupq_n_f32(0.0f);

                for (size_t d = 0; d < subdim; ++d) {
                    float32x4_t qv = use_cache
                        ? qv_cache[d]
                        : vdupq_n_f32(query_sub[d]);

                    const float* cent4 =
                        &centroids_xc4[
                            ((m * centroid_blocks + cb) * subdim + d) * 4
                        ];

                    float32x4_t cv = vld1q_f32(cent4);
                    sum = vfmaq_f32(sum, qv, cv);
                }

                size_t c0 = cb * 4;
                if (c0 + 3 < Ks) {
                    vst1q_f32(lut_m + c0, sum);
                } else {
                    float tmp[4];
                    vst1q_f32(tmp, sum);

                    for (size_t lane = 0; lane < 4; ++lane) {
                        size_t c = c0 + lane;
                        if (c < Ks) {
                            lut_m[c] = tmp[lane];
                        }
                    }
                }
            }
        }
    }

    void build_lut_xc4_omp(
        const float* __restrict__ query,
        float* __restrict__ lut,
        int num_threads
    ) const {
        constexpr size_t MAX_SUBDIM = 64;

        if (num_threads <= 1 || M < 2) {
            build_lut_xc4(query, lut);
            return;
        }

        if (static_cast<size_t>(num_threads) > M) {
            num_threads = static_cast<int>(M);
        }

        #pragma omp parallel for num_threads(num_threads) schedule(static)
        for (long long mm = 0; mm < static_cast<long long>(M); ++mm) {
            size_t m = static_cast<size_t>(mm);

            const float* query_sub = query + m * subdim;
            float* lut_m = lut + m * Ks;

            float32x4_t qv_cache[MAX_SUBDIM];

            if (subdim <= MAX_SUBDIM) {
                for (size_t d = 0; d < subdim; ++d) {
                    qv_cache[d] = vdupq_n_f32(query_sub[d]);
                }

                for (size_t cb = 0; cb < centroid_blocks; ++cb) {
                    float32x4_t sum = vdupq_n_f32(0.0f);

                    for (size_t d = 0; d < subdim; ++d) {
                        const float* cent4 =
                            &centroids_xc4[
                                ((m * centroid_blocks + cb) * subdim + d) * 4
                            ];

                        float32x4_t cv = vld1q_f32(cent4);
                        sum = vfmaq_f32(sum, qv_cache[d], cv);
                    }

                    size_t c0 = cb * 4;

                    if (c0 + 3 < Ks) {
                        vst1q_f32(lut_m + c0, sum);
                    } else {
                        float temp[4];
                        vst1q_f32(temp, sum);

                        for (size_t lane = 0; lane < 4; ++lane) {
                            size_t c = c0 + lane;
                            if (c < Ks) {
                                lut_m[c] = temp[lane];
                            }
                        }
                    }
                }
            } else {
                for (size_t cb = 0; cb < centroid_blocks; ++cb) {
                    float32x4_t sum = vdupq_n_f32(0.0f);

                    for (size_t d = 0; d < subdim; ++d) {
                        float32x4_t qv = vdupq_n_f32(query_sub[d]);

                        const float* cent4 =
                            &centroids_xc4[
                                ((m * centroid_blocks + cb) * subdim + d) * 4
                            ];

                        float32x4_t cv = vld1q_f32(cent4);
                        sum = vfmaq_f32(sum, qv, cv);
                    }

                    size_t c0 = cb * 4;

                    if (c0 + 3 < Ks) {
                        vst1q_f32(lut_m + c0, sum);
                    } else {
                        float temp[4];
                        vst1q_f32(temp, sum);

                        for (size_t lane = 0; lane < 4; ++lane) {
                            size_t c = c0 + lane;
                            if (c < Ks) {
                                lut_m[c] = temp[lane];
                            }
                        }
                    }
                }
            }
        }
    }
};
