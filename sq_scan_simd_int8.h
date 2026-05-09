#pragma once

#include <queue>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>
#include <arm_neon.h>

#include "flat_scan_simd.h"

// ============================================================
// int8 对称量化版本 SQ-SIMD
//
// 量化范围：[-max_abs, max_abs] -> [-127, 127]
// q = round(x * 127 / max_abs)
//
// 反量化：
// x ≈ q * step
// step = max_abs / 127
//
// 粗排时：
// dot(x, q) ≈ step^2 * Σ(base_q[d] * query_q[d])
//
// 因为 step^2 是正数，所以粗排阶段直接比较 int8 点积即可。
// ============================================================

static inline int32_t horizontal_sum_s32_direct(int32x4_t v) {
    return vaddvq_s32(v);
}

// int8 NEON 内积：Σ a[i] * b[i]
// 每次处理 16 个 int8_t
static inline int32_t inner_product_int8_neon(
    const int8_t* __restrict__ a,
    const int8_t* __restrict__ b,
    size_t dim
) {
    size_t i = 0;

    int32x4_t sum0 = vdupq_n_s32(0);
    int32x4_t sum1 = vdupq_n_s32(0);

    for (; i + 16 <= dim; i += 16) {
        int8x16_t va = vld1q_s8(a + i);
        int8x16_t vb = vld1q_s8(b + i);

        // 低 8 位：int8 * int8 -> int16
        int16x8_t prod_low = vmull_s8(
            vget_low_s8(va),
            vget_low_s8(vb)
        );

        // 高 8 位：int8 * int8 -> int16
        int16x8_t prod_high = vmull_s8(
            vget_high_s8(va),
            vget_high_s8(vb)
        );

        // int16 水平成对累加到 int32
        sum0 = vpadalq_s16(sum0, prod_low);
        sum1 = vpadalq_s16(sum1, prod_high);
    }

    int32x4_t sum = vaddq_s32(sum0, sum1);
    int32_t result = horizontal_sum_s32_direct(sum);

    // DEEP100K 维度为 96，96 % 16 == 0，正常不会进入这里
    for (; i < dim; ++i) {
        result += static_cast<int32_t>(a[i]) * static_cast<int32_t>(b[i]);
    }

    return result;
}

// 重新寻找 int32 分数数组中的当前最差项
static inline void recompute_worst_i32(
    const int32_t* score,
    size_t cnt,
    size_t& worst_pos,
    int32_t& worst_score
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

// 重新寻找 float 分数数组中的当前最差项
static inline void recompute_worst_f32(
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

class SQIndexSIMDInt8 {
public:
    SQIndexSIMDInt8(float* base, size_t base_number, size_t vecdim)
        : base_float(base),
          base_number(base_number),
          vecdim(vecdim),
          max_abs(1.0f),
          step(1.0f),
          inv_step(1.0f)
    {
        base_q = new int8_t[base_number * vecdim];
        //std::cout<<"base_number="<<base_number<<", vecdim="<<vecdim<<"\n";
        compute_quant_params();
        build_quantized_base();
    }

    ~SQIndexSIMDInt8() {
        delete[] base_q;
    }

    std::priority_queue<std::pair<float, uint32_t> > search(
        float* query,
        size_t k,
        size_t rerank_p
    ) {
        constexpr size_t MAX_P = 4096;
        constexpr size_t MAX_K = 128;
        constexpr size_t MAX_DIM = 128;
        constexpr size_t PREFETCH_DIST = 16;

        if (rerank_p < k) {
            rerank_p = k;
        }

        if (rerank_p > base_number) {
            rerank_p = base_number;
        }

        if (rerank_p > MAX_P) {
            rerank_p = MAX_P;
        }

        if (k > MAX_K) {
            k = MAX_K;
        }

        // DEEP100K 的 vecdim = 96，小于 MAX_DIM
        int8_t query_q[MAX_DIM];

        for (size_t d = 0; d < vecdim; ++d) {
            query_q[d] = quantize(query[d]);
        }

        // ====================================================
        // 1. int8 SQ 粗排：维护 Top-p
        // ====================================================
        int32_t cand_score[MAX_P];
        uint32_t cand_id[MAX_P];

        size_t cand_cnt = 0;
        size_t cand_worst_pos = 0;
        int32_t cand_worst_score = 0;

        for (size_t i = 0; i < base_number; ++i) {
            if (i + PREFETCH_DIST < base_number) {
                __builtin_prefetch(base_q + (i + PREFETCH_DIST) * vecdim, 0, 1);
            }

            const int8_t* base_vec_q = base_q + i * vecdim;

            int32_t approx_score = inner_product_int8_neon(
                base_vec_q,
                query_q,
                vecdim
            );

            // approx_score 越大，近似内积越大，距离越小
            if (cand_cnt < rerank_p) {
                cand_score[cand_cnt] = approx_score;
                cand_id[cand_cnt] = static_cast<uint32_t>(i);
                ++cand_cnt;

                if (cand_cnt == rerank_p) {
                    recompute_worst_i32(
                        cand_score,
                        cand_cnt,
                        cand_worst_pos,
                        cand_worst_score
                    );
                }
            } else {
                if (approx_score > cand_worst_score) {
                    cand_score[cand_worst_pos] = approx_score;
                    cand_id[cand_worst_pos] = static_cast<uint32_t>(i);

                    recompute_worst_i32(
                        cand_score,
                        cand_cnt,
                        cand_worst_pos,
                        cand_worst_score
                    );
                }
            }
        }

        // ====================================================
        // 2. 精排：对 Top-p 候选使用原始 float + NEON FMA
        // ====================================================
        float best_score[MAX_K];
        uint32_t best_id[MAX_K];

        size_t best_cnt = 0;
        size_t best_worst_pos = 0;
        float best_worst_score = 0.0f;

        for (size_t i = 0; i < cand_cnt; ++i) {
            uint32_t id = cand_id[i];

            if (i + 4 < cand_cnt) {
                uint32_t next_id = cand_id[i + 4];
                __builtin_prefetch(
                    base_float + static_cast<size_t>(next_id) * vecdim,
                    0,
                    1
                );
            }

            const float* base_vec =
                base_float + static_cast<size_t>(id) * vecdim;

            // 复用 Flat-SIMD 优化基线中的 NEON + FMA 内积函数
            float score = inner_product_neon16_fma(
                base_vec,
                query,
                vecdim
            );

            if (best_cnt < k) {
                best_score[best_cnt] = score;
                best_id[best_cnt] = id;
                ++best_cnt;

                if (best_cnt == k) {
                    recompute_worst_f32(
                        best_score,
                        best_cnt,
                        best_worst_pos,
                        best_worst_score
                    );
                }
            } else {
                if (score > best_worst_score) {
                    best_score[best_worst_pos] = score;
                    best_id[best_worst_pos] = id;

                    recompute_worst_f32(
                        best_score,
                        best_cnt,
                        best_worst_pos,
                        best_worst_score
                    );
                }
            }
        }

        // ====================================================
        // 3. 转换为原始框架需要的 priority_queue<pair<dis, id>>
        // ====================================================
        std::priority_queue<std::pair<float, uint32_t> > q;

        for (size_t i = 0; i < best_cnt; ++i) {
            float dis = 1.0f - best_score[i];
            q.push({dis, best_id[i]});
        }

        return q;
    }

    float get_max_abs() const {
        return max_abs;
    }

    float get_step() const {
        return step;
    }

private:
    float* base_float;
    size_t base_number;
    size_t vecdim;

    int8_t* base_q;

    float max_abs;
    float step;
    float inv_step;

    void compute_quant_params() {
        float data_min = std::numeric_limits<float>::max();
        float data_max = -std::numeric_limits<float>::max();

        const size_t total = base_number * vecdim;

        for (size_t i = 0; i < total; ++i) {
            float x = base_float[i];

            if (x < data_min) {
                data_min = x;
            }

            if (x > data_max) {
                data_max = x;
            }
        }

        float abs_min = std::fabs(data_min);
        float abs_max = std::fabs(data_max);

        max_abs = std::max(abs_min, abs_max);

        if (max_abs <= 1e-12f) {
            max_abs = 1.0f;
        }

        step = max_abs / 127.0f;
        inv_step = 127.0f / max_abs;
    }

    int8_t quantize(float x) const {
        int q = static_cast<int>(std::round(x * inv_step));

        if (q > 127) {
            q = 127;
        } else if (q < -127) {
            q = -127;
        }

        return static_cast<int8_t>(q);
    }

    void build_quantized_base() {
        for (size_t i = 0; i < base_number; ++i) {
            const float* base_vec = base_float + i * vecdim;
            int8_t* base_vec_q = base_q + i * vecdim;

            for (size_t d = 0; d < vecdim; ++d) {
                base_vec_q[d] = quantize(base_vec[d]);
            }
        }
    }
};