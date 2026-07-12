#pragma once

#include <queue>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>
#include "simd_compat.h"

#include "flat_scan_simd.h"

// ============================================================
// uint8 SQ 量化说明
//
// 根据 base 原始数据统计 data_min 和 data_max，
// 将 [data_min, data_max] 线性映射到 [0, 255]：
//
//     q = round((x - data_min) * 255 / (data_max - data_min))
//
// 等价于：
//
//     offset = -data_min
//     scale_factor = 255 / (data_max - data_min)
//     q = round((x + offset) * scale_factor)
//
// 反量化近似为：
//
//     x ≈ data_min + q * step
//     step = (data_max - data_min) / 255
// ============================================================


// uint8_t 点积：Σ a[i] * b[i]
// 每次处理 16 个 uint8_t
static inline uint32_t inner_product_uint8_neon(
    const uint8_t* __restrict__ a,
    const uint8_t* __restrict__ b,
    size_t dim
) {
    size_t i = 0;

    uint32x4_t sum0 = vdupq_n_u32(0);
    uint32x4_t sum1 = vdupq_n_u32(0);

    for (; i + 16 <= dim; i += 16) {
        uint8x16_t va = vld1q_u8(a + i);
        uint8x16_t vb = vld1q_u8(b + i);

        uint16x8_t prod_low  = vmull_u8(vget_low_u8(va),  vget_low_u8(vb));
        uint16x8_t prod_high = vmull_u8(vget_high_u8(va), vget_high_u8(vb));

        sum0 = vpadalq_u16(sum0, prod_low);
        sum1 = vpadalq_u16(sum1, prod_high);
    }

    uint32x4_t sum = vaddq_u32(sum0, sum1);
    uint32_t result = vaddvq_u32(sum);

    // DEEP100K 是 96 维，96 % 16 == 0，正常不会进入这里。
    /*
    for (; i < dim; ++i) {
        result += static_cast<uint32_t>(a[i]) * static_cast<uint32_t>(b[i]);
    }
    */

    return result;
}

// 统计 uint8 向量元素和：Σ q[i]
// 用于非对称 offset 量化后的近似内积修正
static inline uint32_t sum_uint8_scalar(
    const uint8_t* a,
    size_t dim
) {
    uint32_t s = 0;
    for (size_t i = 0; i < dim; ++i) {
        s += static_cast<uint32_t>(a[i]);
    }
    return s;
}

// 固定数组 Top-k / Top-p 中重新寻找当前最差项
static inline void recompute_worst_score(
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

class SQIndexSIMD {
public:
    SQIndexSIMD(float* base, size_t base_number, size_t vecdim)
        : base_float(base),
          base_number(base_number),
          vecdim(vecdim),
          data_min(0.0f),
          data_max(0.0f),
          step(1.0f),
          inv_step(1.0f)
    {
        base_q = new uint8_t[base_number * vecdim];
        base_q_sum = new uint32_t[base_number];

        compute_quant_params();
        build_quantized_base();
    }

    ~SQIndexSIMD() {
        delete[] base_q;
        delete[] base_q_sum;
    }

    std::priority_queue<std::pair<float, uint32_t> > search(
        float* query,
        size_t k,
        size_t rerank_p
    ) {
        constexpr size_t MAX_P = 4096;
        constexpr size_t MAX_K = 128;
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

        // ====================================================
        // 1. 在线量化 query
        // ====================================================
        uint8_t query_q[96];

        for (size_t d = 0; d < vecdim; ++d) {
            query_q[d] = quantize(query[d]);
        }

        uint32_t query_q_sum = sum_uint8_scalar(query_q, vecdim);

        // 反量化公式：
        //      x ≈ data_min + q * step
        //
        // 所以近似内积：
        //      Σ(data_min + qb * step)(data_min + qq * step)
        //
        // 展开：
        //      step^2 * Σ(qb * qq)
        //    + step * data_min * Σ(qb)
        //    + step * data_min * Σ(qq)
        //    + dim * data_min^2
        //
        // 对于同一个 query，Σ(qq) 和 dim * data_min^2 是常数；
        // 为了数值含义完整，这里仍然把完整近似 score 算出来。
        const float dot_coeff = step * step;
        const float sum_coeff = step * data_min;
        const float query_const =
            sum_coeff * static_cast<float>(query_q_sum)
            + static_cast<float>(vecdim) * data_min * data_min;

        // ====================================================
        // 2. SQ 粗排：用 uint8 近似内积维护 Top-p
        // ====================================================
        float cand_score[MAX_P];
        uint32_t cand_id[MAX_P];

        size_t cand_cnt = 0;
        size_t cand_worst_pos = 0;
        float cand_worst_score = 0.0f;

        for (size_t i = 0; i < base_number; ++i) {
            if (i + PREFETCH_DIST < base_number) {
                __builtin_prefetch(base_q + (i + PREFETCH_DIST) * vecdim, 0, 1);
            }

            const uint8_t* base_vec_q = base_q + i * vecdim;

            uint32_t dot_q = inner_product_uint8_neon(
                base_vec_q,
                query_q,
                vecdim
            );

            float approx_score =
                dot_coeff * static_cast<float>(dot_q)
                + sum_coeff * static_cast<float>(base_q_sum[i])
                + query_const;

            if (cand_cnt < rerank_p) {
                cand_score[cand_cnt] = approx_score;
                cand_id[cand_cnt] = static_cast<uint32_t>(i);
                ++cand_cnt;

                if (cand_cnt == rerank_p) {
                    recompute_worst_score(
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

                    recompute_worst_score(
                        cand_score,
                        cand_cnt,
                        cand_worst_pos,
                        cand_worst_score
                    );
                }
            }
        }

        // ====================================================
        // 3. 精排：对 Top-p 候选使用原始 float + NEON FMA
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

            // 复用 Flat-SIMD 优化基线的 FMA 内积函数
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
                    recompute_worst_score(
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

                    recompute_worst_score(
                        best_score,
                        best_cnt,
                        best_worst_pos,
                        best_worst_score
                    );
                }
            }
        }

        // ====================================================
        // 4. 返回 main.cc 原有格式：priority_queue<pair<dis, id>>
        // ====================================================
        std::priority_queue<std::pair<float, uint32_t> > q;

        for (size_t i = 0; i < best_cnt; ++i) {
            float dis = 1.0f - best_score[i];
            q.push({dis, best_id[i]});
        }

        return q;
    }

    float get_data_min() const {
        return data_min;
    }

    float get_data_max() const {
        return data_max;
    }

    float get_step() const {
        return step;
    }

private:
    float* base_float;
    size_t base_number;
    size_t vecdim;

    uint8_t* base_q;
    uint32_t* base_q_sum;

    float data_min;
    float data_max;
    float step;
    float inv_step;

    void compute_quant_params() {
        data_min = std::numeric_limits<float>::max();
        data_max = -std::numeric_limits<float>::max();

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

        float range = data_max - data_min;

        if (range <= 1e-12f) {
            step = 1.0f;
            inv_step = 1.0f;
        } else {
            step = range / 255.0f;
            inv_step = 255.0f / range;
        }
    }

    uint8_t quantize(float x) const {
        int q = static_cast<int>(std::round((x - data_min) * inv_step));

        if (q < 0) {
            q = 0;
        } else if (q > 255) {
            q = 255;
        }

        return static_cast<uint8_t>(q);
    }

    void build_quantized_base() {
        for (size_t i = 0; i < base_number; ++i) {
            const float* base_vec = base_float + i * vecdim;
            uint8_t* base_vec_q = base_q + i * vecdim;

            uint32_t s = 0;

            for (size_t d = 0; d < vecdim; ++d) {
                uint8_t q = quantize(base_vec[d]);
                base_vec_q[d] = q;
                s += static_cast<uint32_t>(q);
            }

            base_q_sum[i] = s;
        }
    }
};
