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
#include <arm_neon.h>

#include "flat_scan_simd.h"

// ============================================================
// PQ 初始化方式
// ============================================================

enum class PQInitMode {
    Uniform,   // 原始均匀抽样初始化
    KMeansPP   // KMeans++ 初始化
};

// ============================================================
// 基础 PQ-SIMD 版本
//
// 默认设置：
//   vecdim = 96
//   M      = 4, 8, 12, 16
//   subdim = 24, 12, 8, 6
//   Ks     = 256
//
// 离线阶段：
//   1. 每个子空间训练 codebook
//   2. 将每条 base 向量编码成 M 个 uint8_t code
//
// 在线阶段：
//   1. 对 query 构建 M × Ks 的 LUT
//   2. 遍历 base 的 PQ code，查表累加得到近似内积
//   3. 选出 Top-p 候选
//   4. 用原始 float + inner_product_neon16_fma 精排 Top-k
// ============================================================

static inline float pq_horizontal_sum_f32(float32x4_t v) {
    return vaddvq_f32(v);
}

// 计算 8 维子向量内积，适合 subdim = 8
static inline float pq_inner_product_8_neon(
    const float* __restrict__ a,
    const float* __restrict__ b
) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    float32x4_t a0 = vld1q_f32(a);
    float32x4_t b0 = vld1q_f32(b);
    sum = vfmaq_f32(sum, a0, b0);

    float32x4_t a1 = vld1q_f32(a + 4);
    float32x4_t b1 = vld1q_f32(b + 4);
    sum = vfmaq_f32(sum, a1, b1);

    return pq_horizontal_sum_f32(sum);
}

// 计算 8 维子向量 L2 距离，用于 KMeans 分配和 PQ 编码
static inline float pq_l2_8_neon(
    const float* __restrict__ a,
    const float* __restrict__ b
) {
    float32x4_t sum = vdupq_n_f32(0.0f);

    float32x4_t a0 = vld1q_f32(a);
    float32x4_t b0 = vld1q_f32(b);
    float32x4_t diff0 = vsubq_f32(a0, b0);
    sum = vfmaq_f32(sum, diff0, diff0);

    float32x4_t a1 = vld1q_f32(a + 4);
    float32x4_t b1 = vld1q_f32(b + 4);
    float32x4_t diff1 = vsubq_f32(a1, b1);
    sum = vfmaq_f32(sum, diff1, diff1);

    return pq_horizontal_sum_f32(sum);
}

// 通用 L2 距离，备用
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

    float result = pq_horizontal_sum_f32(sum);

    for (; i < dim; ++i) {
        float diff = a[i] - b[i];
        result += diff * diff;
    }

    return result;
}

// 通用子空间内积，备用
static inline float pq_inner_product_generic(
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

    float result = pq_horizontal_sum_f32(sum);

    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

// 固定数组中重新寻找当前最差项
static inline void pq_recompute_worst_score(
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

// 通用 PQ code 查表累加
static inline float pq_scan_code_generic(
    const uint8_t* __restrict__ code,
    const float* __restrict__ lut,
    size_t M,
    size_t Ks
) {
    float score = 0.0f;

    for (size_t m = 0; m < M; ++m) {
        score += lut[m * Ks + static_cast<size_t>(code[m])];
    }

    return score;
}

// M = 4 特化版本
static inline float pq_scan_code_m4(
    const uint8_t* __restrict__ code,
    const float* __restrict__ lut,
    size_t Ks
) {
    return
        lut[0 * Ks + static_cast<size_t>(code[0])] +
        lut[1 * Ks + static_cast<size_t>(code[1])] +
        lut[2 * Ks + static_cast<size_t>(code[2])] +
        lut[3 * Ks + static_cast<size_t>(code[3])];
}

// M = 8 特化版本
static inline float pq_scan_code_m8(
    const uint8_t* __restrict__ code,
    const float* __restrict__ lut,
    size_t Ks
) {
    return
        lut[0 * Ks + static_cast<size_t>(code[0])] +
        lut[1 * Ks + static_cast<size_t>(code[1])] +
        lut[2 * Ks + static_cast<size_t>(code[2])] +
        lut[3 * Ks + static_cast<size_t>(code[3])] +
        lut[4 * Ks + static_cast<size_t>(code[4])] +
        lut[5 * Ks + static_cast<size_t>(code[5])] +
        lut[6 * Ks + static_cast<size_t>(code[6])] +
        lut[7 * Ks + static_cast<size_t>(code[7])];
}

// M = 12 特化版本
static inline float pq_scan_code_m12(
    const uint8_t* __restrict__ code,
    const float* __restrict__ lut,
    size_t Ks
) {
    return
        lut[0  * Ks + static_cast<size_t>(code[0])]  +
        lut[1  * Ks + static_cast<size_t>(code[1])]  +
        lut[2  * Ks + static_cast<size_t>(code[2])]  +
        lut[3  * Ks + static_cast<size_t>(code[3])]  +
        lut[4  * Ks + static_cast<size_t>(code[4])]  +
        lut[5  * Ks + static_cast<size_t>(code[5])]  +
        lut[6  * Ks + static_cast<size_t>(code[6])]  +
        lut[7  * Ks + static_cast<size_t>(code[7])]  +
        lut[8  * Ks + static_cast<size_t>(code[8])]  +
        lut[9  * Ks + static_cast<size_t>(code[9])]  +
        lut[10 * Ks + static_cast<size_t>(code[10])] +
        lut[11 * Ks + static_cast<size_t>(code[11])];
}

// M = 16 特化版本
static inline float pq_scan_code_m16(
    const uint8_t* __restrict__ code,
    const float* __restrict__ lut,
    size_t Ks
) {
    return
        lut[0  * Ks + static_cast<size_t>(code[0])]  +
        lut[1  * Ks + static_cast<size_t>(code[1])]  +
        lut[2  * Ks + static_cast<size_t>(code[2])]  +
        lut[3  * Ks + static_cast<size_t>(code[3])]  +
        lut[4  * Ks + static_cast<size_t>(code[4])]  +
        lut[5  * Ks + static_cast<size_t>(code[5])]  +
        lut[6  * Ks + static_cast<size_t>(code[6])]  +
        lut[7  * Ks + static_cast<size_t>(code[7])]  +
        lut[8  * Ks + static_cast<size_t>(code[8])]  +
        lut[9  * Ks + static_cast<size_t>(code[9])]  +
        lut[10 * Ks + static_cast<size_t>(code[10])] +
        lut[11 * Ks + static_cast<size_t>(code[11])] +
        lut[12 * Ks + static_cast<size_t>(code[12])] +
        lut[13 * Ks + static_cast<size_t>(code[13])] +
        lut[14 * Ks + static_cast<size_t>(code[14])] +
        lut[15 * Ks + static_cast<size_t>(code[15])];
}

// 统一入口：特殊 M 走展开版本，其他 M 走通用版本
static inline float pq_scan_code(
    const uint8_t* __restrict__ code,
    const float* __restrict__ lut,
    size_t M,
    size_t Ks
) {
    switch (M) {
        case 4:
            return pq_scan_code_m4(code, lut, Ks);
        case 8:
            return pq_scan_code_m8(code, lut, Ks);
        case 12:
            return pq_scan_code_m12(code, lut, Ks);
        case 16:
            return pq_scan_code_m16(code, lut, Ks);
        default:
            return pq_scan_code_generic(code, lut, M, Ks);
    }
}

// 固定数组 Top-p 更新函数
static inline void pq_update_top_p(
    float score,
    uint32_t id,
    float* cand_score,
    uint32_t* cand_id,
    size_t rerank_p,
    size_t& cand_cnt,
    size_t& cand_worst_pos,
    float& cand_worst_score
) {
    if (cand_cnt < rerank_p) {
        cand_score[cand_cnt] = score;
        cand_id[cand_cnt] = id;
        ++cand_cnt;

        if (cand_cnt == rerank_p) {
            pq_recompute_worst_score(
                cand_score,
                cand_cnt,
                cand_worst_pos,
                cand_worst_score
            );
        }
    } else {
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
}

static inline void pq_scan_code_batch4(
    const uint8_t* __restrict__ codes_ptr,
    const float* __restrict__ lut,
    size_t M,
    size_t Ks,
    float& s0,
    float& s1,
    float& s2,
    float& s3
) {
    const uint8_t* code0 = codes_ptr;
    const uint8_t* code1 = codes_ptr + M;
    const uint8_t* code2 = codes_ptr + 2 * M;
    const uint8_t* code3 = codes_ptr + 3 * M;

    s0 = pq_scan_code(code0, lut, M, Ks);
    s1 = pq_scan_code(code1, lut, M, Ks);
    s2 = pq_scan_code(code2, lut, M, Ks);
    s3 = pq_scan_code(code3, lut, M, Ks);
}

static inline void pq_scan_code_batch4_interleaved(
    const uint8_t* __restrict__ codes_ptr,
    const float* __restrict__ lut,
    size_t M,
    size_t Ks,
    float& s0,
    float& s1,
    float& s2,
    float& s3
) {
    const uint8_t* code0 = codes_ptr;
    const uint8_t* code1 = codes_ptr + M;
    const uint8_t* code2 = codes_ptr + 2 * M;
    const uint8_t* code3 = codes_ptr + 3 * M;

    s0 = 0.0f;
    s1 = 0.0f;
    s2 = 0.0f;
    s3 = 0.0f;

    for (size_t m = 0; m < M; ++m) {
        const float* lut_m = lut + m * Ks;

        s0 += lut_m[static_cast<size_t>(code0[m])];
        s1 += lut_m[static_cast<size_t>(code1[m])];
        s2 += lut_m[static_cast<size_t>(code2[m])];
        s3 += lut_m[static_cast<size_t>(code3[m])];
    }
}

static inline void pq_scan_code_batch4_m12_interleaved(
    const uint8_t* __restrict__ codes_ptr,
    const float* __restrict__ lut,
    size_t Ks,
    float& s0,
    float& s1,
    float& s2,
    float& s3
) {
    const uint8_t* code0 = codes_ptr;
    const uint8_t* code1 = codes_ptr + 12;
    const uint8_t* code2 = codes_ptr + 24;
    const uint8_t* code3 = codes_ptr + 36;

    s0 = s1 = s2 = s3 = 0.0f;

    #define ADD_M(m) do { \
        const float* lut_m = lut + (m) * Ks; \
        s0 += lut_m[static_cast<size_t>(code0[m])]; \
        s1 += lut_m[static_cast<size_t>(code1[m])]; \
        s2 += lut_m[static_cast<size_t>(code2[m])]; \
        s3 += lut_m[static_cast<size_t>(code3[m])]; \
    } while (0)

    ADD_M(0);
    ADD_M(1);
    ADD_M(2);
    ADD_M(3);
    ADD_M(4);
    ADD_M(5);
    ADD_M(6);
    ADD_M(7);
    ADD_M(8);
    ADD_M(9);
    ADD_M(10);
    ADD_M(11);

    #undef ADD_M
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
          init_mode(init_mode)
    {
        if (this->M == 0) {
            this->M = 12;
        }

        if (vecdim % this->M != 0) {
            std::cerr << "PQ warning: vecdim % M != 0, reset M to 12\n";
            this->M = 12;
        }

        if (vecdim % this->M != 0) {
            std::cerr << "PQ error: vecdim cannot be divided by M\n";
        }

        subdim = vecdim / this->M;

        if (this->Ks > 256) {
            this->Ks = 256;
        }

        if (this->Ks < 2) {
            this->Ks = 2;
        }

        if (this->train_size > base_number) {
            this->train_size = base_number;
        }

        if (this->train_size < this->Ks) {
            std::cerr << "PQ warning: train_size < Ks, reset train_size to Ks if possible\n";
            if (base_number >= this->Ks) {
                this->train_size = this->Ks;
            } else {
                this->train_size = base_number;
            }
        }

        centroids.resize(this->M * this->Ks * subdim);
        codes.resize(base_number * this->M);

        build();
    }

    std::priority_queue<std::pair<float, uint32_t> > search(
        float* query,
        size_t k,
        size_t rerank_p
    ) {
        constexpr size_t MAX_P = 8192;
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

        // ----------------------------------------------------
        // 1. 构建 Query LUT
        //
        // LUT[m][c] = query 的第 m 个子向量 与
        //             第 m 个 codebook 中第 c 个中心的内积
        // ----------------------------------------------------
        std::vector<float> lut(M * Ks);

        for (size_t m = 0; m < M; ++m) {
            const float* query_sub = query + m * subdim;
            const float* centroid_base = &centroids[m * Ks * subdim];

            for (size_t c = 0; c < Ks; ++c) {
                const float* centroid = centroid_base + c * subdim;

                float score;
                if (subdim == 8) {
                    score = pq_inner_product_8_neon(query_sub, centroid);
                } else {
                    score = pq_inner_product_generic(query_sub, centroid, subdim);
                }

                lut[m * Ks + c] = score;
            }
        }

        // ----------------------------------------------------
        // 2. PQ 粗排：查表累加，维护 Top-p
        // ----------------------------------------------------
        float cand_score[MAX_P];
        uint32_t cand_id[MAX_P];

        size_t cand_cnt = 0;
        size_t cand_worst_pos = 0;
        float cand_worst_score = 0.0f;

        size_t i = 0;

        ///*
        // 一次处理 4 条 base code
        for (; i + 4 <= base_number; i += 4) {
            if (i + PREFETCH_DIST < base_number) {
                __builtin_prefetch(&codes[(i + PREFETCH_DIST) * M], 0, 1);
            }

            const uint8_t* codes_ptr = &codes[i * M];

            float s0, s1, s2, s3;

            if (M == 12) {
                pq_scan_code_batch4_m12_interleaved(
                    codes_ptr,
                    lut.data(),
                    Ks,
                    s0, s1, s2, s3
                );
            } else {
                pq_scan_code_batch4_interleaved(
                    codes_ptr,
                    lut.data(),
                    M,
                    Ks,
                    s0, s1, s2, s3
                );
            }

            pq_update_top_p(
                s0,
                static_cast<uint32_t>(i + 0),
                cand_score,
                cand_id,
                rerank_p,
                cand_cnt,
                cand_worst_pos,
                cand_worst_score
            );

            pq_update_top_p(
                s1,
                static_cast<uint32_t>(i + 1),
                cand_score,
                cand_id,
                rerank_p,
                cand_cnt,
                cand_worst_pos,
                cand_worst_score
            );

            pq_update_top_p(
                s2,
                static_cast<uint32_t>(i + 2),
                cand_score,
                cand_id,
                rerank_p,
                cand_cnt,
                cand_worst_pos,
                cand_worst_score
            );

            pq_update_top_p(
                s3,
                static_cast<uint32_t>(i + 3),
                cand_score,
                cand_id,
                rerank_p,
                cand_cnt,
                cand_worst_pos,
                cand_worst_score
            );
        }

        // 处理剩余不足 4 条的尾部
        for (; i < base_number; ++i) {
            if (i + PREFETCH_DIST < base_number) {
                __builtin_prefetch(&codes[(i + PREFETCH_DIST) * M], 0, 1);
            }

            const uint8_t* code = &codes[i * M];

            float approx_score = pq_scan_code(
                code,
                lut.data(),
                M,
                Ks
            );

            pq_update_top_p(
                approx_score,
                static_cast<uint32_t>(i),
                cand_score,
                cand_id,
                rerank_p,
                cand_cnt,
                cand_worst_pos,
                cand_worst_score
            );
        }
        //*/

        /*
        for (size_t i = 0; i < base_number; ++i) {
            if (i + PREFETCH_DIST < base_number) {
                __builtin_prefetch(&codes[(i + PREFETCH_DIST) * M], 0, 1);
            }

            const uint8_t* code = &codes[i * M];

            float approx_score = pq_scan_code(code, lut.data(), M, Ks);

            if (cand_cnt < rerank_p) {
                cand_score[cand_cnt] = approx_score;
                cand_id[cand_cnt] = static_cast<uint32_t>(i);
                ++cand_cnt;

                if (cand_cnt == rerank_p) {
                    pq_recompute_worst_score(
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

                    pq_recompute_worst_score(
                        cand_score,
                        cand_cnt,
                        cand_worst_pos,
                        cand_worst_score
                    );
                }
            }
        }
        */
        // ----------------------------------------------------
        // 3. 精排：复用 Flat-SIMD 基线的原始 float 内积
        // ----------------------------------------------------
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
                    pq_recompute_worst_score(
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

                    pq_recompute_worst_score(
                        best_score,
                        best_cnt,
                        best_worst_pos,
                        best_worst_score
                    );
                }
            }
        }

        // ----------------------------------------------------
        // 4. 返回 main.cc 原有格式：priority_queue<pair<dis,id>>
        // ----------------------------------------------------
        std::priority_queue<std::pair<float, uint32_t> > q;

        for (size_t i = 0; i < best_cnt; ++i) {
            float dis = 1.0f - best_score[i];
            q.push({dis, best_id[i]});
        }

        return q;
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

    // 确定训练样本 id：使用均匀抽样，避免训练集过大
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

    float sub_l2(
        const float* a,
        const float* b
    ) const {
        if (subdim == 8) {
            return pq_l2_8_neon(a, b);
        }

        return pq_l2_generic(a, b, subdim);
    }

    // 找到某个子向量最近的 centroid
    uint8_t nearest_centroid_l2(
        const float* subvec,
        size_t m
    ) const {
        size_t best_c = 0;
        float best_dis = std::numeric_limits<float>::max();

        for (size_t c = 0; c < Ks; ++c) {
            const float* centroid = get_centroid(m, c);
            float dis = sub_l2(subvec, centroid);

            if (dis < best_dis) {
                best_dis = dis;
                best_c = c;
            }
        }

        return static_cast<uint8_t>(best_c);
    }

    // ========================================================
    // 原始普通 KMeans 初始化：均匀抽样
    // ========================================================
    void init_centroids_uniform(size_t m) {
        for (size_t c = 0; c < Ks; ++c) {
            size_t t = c * train_size / Ks;

            if (t >= train_size) {
                t = train_size - 1;
            }

            size_t id = train_id(t);
            const float* src = get_base_subvec(id, m);
            float* dst = get_centroid(m, c);

            for (size_t d = 0; d < subdim; ++d) {
                dst[d] = src[d];
            }
        }
    }

    // ========================================================
    // KMeans++ 初始化
    //
    // 第一个中心随机选择；
    // 后续中心按样本到已有中心集合的最近距离进行加权采样。
    // sub_l2 返回的是平方 L2 距离，因此可以直接作为权重。
    // ========================================================
    void init_centroids_kmeanspp(size_t m) {
        std::mt19937 rng(2026 + static_cast<unsigned int>(m));

        std::vector<float> min_dist(
            train_size,
            std::numeric_limits<float>::max()
        );

        // 1. 随机选择第一个中心
        std::uniform_int_distribution<size_t> first_dist(0, train_size - 1);
        size_t first_t = first_dist(rng);
        size_t first_id = train_id(first_t);

        {
            const float* src = get_base_subvec(first_id, m);
            float* dst = get_centroid(m, 0);

            for (size_t d = 0; d < subdim; ++d) {
                dst[d] = src[d];
            }
        }

        // 2. 依次选择剩余 Ks-1 个中心
        for (size_t c = 1; c < Ks; ++c) {
            const float* last_centroid = get_centroid(m, c - 1);

            float total_dist = 0.0f;

            // 只需要用“新加入的中心”更新 min_dist
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
            const float* src = get_base_subvec(chosen_id, m);
            float* dst = get_centroid(m, c);

            for (size_t d = 0; d < subdim; ++d) {
                dst[d] = src[d];
            }
        }
    }

    // 统一初始化入口
    void init_centroids(size_t m) {
        if (init_mode == PQInitMode::KMeansPP) {
            init_centroids_kmeanspp(m);
        } else {
            init_centroids_uniform(m);
        }
    }

    // --------------------------------------------------------
    // 训练每个子空间的 codebook
    // --------------------------------------------------------
    void train_codebooks() {
        std::vector<float> sums(Ks * subdim);
        std::vector<size_t> counts(Ks);

        for (size_t m = 0; m < M; ++m) {
            // 初始化 centroids：Uniform 或 KMeans++
            init_centroids(m);

            for (size_t iter = 0; iter < kmeans_iters; ++iter) {
                std::fill(sums.begin(), sums.end(), 0.0f);
                std::fill(counts.begin(), counts.end(), 0);

                // 分配训练样本到最近中心
                for (size_t t = 0; t < train_size; ++t) {
                    size_t id = train_id(t);
                    const float* subvec = get_base_subvec(id, m);

                    uint8_t cid_u8 = nearest_centroid_l2(subvec, m);
                    size_t cid = static_cast<size_t>(cid_u8);

                    float* sum_vec = &sums[cid * subdim];

                    for (size_t d = 0; d < subdim; ++d) {
                        sum_vec[d] += subvec[d];
                    }

                    counts[cid]++;
                }

                // 更新 centroids
                for (size_t c = 0; c < Ks; ++c) {
                    float* centroid = get_centroid(m, c);

                    if (counts[c] == 0) {
                        // 空簇：重新用某个训练样本初始化
                        size_t id = train_id((c + iter + 1) % train_size);
                        const float* src = get_base_subvec(id, m);

                        for (size_t d = 0; d < subdim; ++d) {
                            centroid[d] = src[d];
                        }
                    } else {
                        float inv_count = 1.0f / static_cast<float>(counts[c]);
                        float* sum_vec = &sums[c * subdim];

                        for (size_t d = 0; d < subdim; ++d) {
                            centroid[d] = sum_vec[d] * inv_count;
                        }
                    }
                }
            }


            /*
            std::cerr << "PQ train subspace " << m + 1 << " / " << M
                      << " with "
                      << (init_mode == PQInitMode::KMeansPP ? "KMeans++" : "Uniform")
                      << " init done.\n";
            */
            
        }
    }

    // --------------------------------------------------------
    // 将所有 base 向量编码为 PQ code
    // 每条向量得到 M 个 uint8_t code
    // --------------------------------------------------------
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
                const float* subvec = get_base_subvec(i, m);
                code[m] = nearest_centroid_l2(subvec, m);
            }
        }
    }
};