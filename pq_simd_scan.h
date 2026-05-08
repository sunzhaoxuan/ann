#pragma once

#include <queue>
#include <vector>
#include <cstdint>
#include <cstddef>
#include <cmath>
#include <algorithm>
#include <limits>
#include <iostream>
#include <arm_neon.h>

#include "flat_scan_simd.h"

// ============================================================
// 基础 PQ-SIMD 版本
//
// 默认设置：
//   vecdim = 96
//   M      = 12
//   subdim = 8
//   Ks     = 64
//
// 离线阶段：
//   1. 每个子空间用训练样本做简化 KMeans
//   2. 得到 M 组 codebook
//   3. 将每条 base 向量编码成 M 个 uint8_t code
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
    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t va0 = vld1q_f32(a);
    float32x4_t vb0 = vld1q_f32(b);
    float32x4_t va1 = vld1q_f32(a + 4);
    float32x4_t vb1 = vld1q_f32(b + 4);

    sum0 = vfmaq_f32(sum0, va0, vb0);
    sum0 = vfmaq_f32(sum0, va1, vb1);

    return pq_horizontal_sum_f32(sum0);
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

// 重新寻找固定数组中当前最差项
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

class PQIndexSIMD {
public:
    PQIndexSIMD(
        float* base,
        size_t base_number,
        size_t vecdim,
        size_t M = 12,
        size_t Ks = 64,
        size_t train_size = 10000,
        size_t kmeans_iters = 6
    )
        : base_float(base),
          base_number(base_number),
          vecdim(vecdim),
          M(M),
          Ks(Ks),
          train_size(train_size),
          kmeans_iters(kmeans_iters)
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

            for (size_t c = 0; c < Ks; ++c) {
                const float* centroid = get_centroid(m, c);

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

        for (size_t i = 0; i < base_number; ++i) {
            if (i + PREFETCH_DIST < base_number) {
                __builtin_prefetch(&codes[(i + PREFETCH_DIST) * M], 0, 1);
            }

            const uint8_t* code = &codes[i * M];

            float approx_score = 0.0f;

            for (size_t m = 0; m < M; ++m) {
                uint8_t cid = code[m];
                approx_score += lut[m * Ks + static_cast<size_t>(cid)];
            }

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

private:
    float* base_float;
    size_t base_number;
    size_t vecdim;

    size_t M;
    size_t Ks;
    size_t subdim;
    size_t train_size;
    size_t kmeans_iters;

    std::vector<float> centroids;
    std::vector<uint8_t> codes;

    void build() {
        std::cerr << "PQ build start. M = " << M
                  << ", Ks = " << Ks
                  << ", subdim = " << subdim
                  << ", train_size = " << train_size
                  << ", iters = " << kmeans_iters << "\n";

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

    // 确定训练样本 id：使用均匀抽样，避免引入随机数
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

    // --------------------------------------------------------
    // 训练每个子空间的 codebook
    //
    // 这是一个基础 KMeans：
    //   1. 用均匀采样的训练点初始化 centroid
    //   2. 分配训练样本到最近中心
    //   3. 对每个中心求均值
    //   4. 重复若干轮
    // --------------------------------------------------------
    void train_codebooks() {
        std::vector<size_t> assign(train_size, 0);
        std::vector<float> sums(Ks * subdim);
        std::vector<size_t> counts(Ks);

        for (size_t m = 0; m < M; ++m) {
            // 初始化 centroids
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

            // KMeans 迭代
            for (size_t iter = 0; iter < kmeans_iters; ++iter) {
                std::fill(sums.begin(), sums.end(), 0.0f);
                std::fill(counts.begin(), counts.end(), 0);

                for (size_t t = 0; t < train_size; ++t) {
                    size_t id = train_id(t);
                    const float* subvec = get_base_subvec(id, m);

                    uint8_t cid = nearest_centroid_l2(subvec, m);
                    assign[t] = static_cast<size_t>(cid);

                    float* sum_vec = &sums[static_cast<size_t>(cid) * subdim];
                    for (size_t d = 0; d < subdim; ++d) {
                        sum_vec[d] += subvec[d];
                    }

                    counts[static_cast<size_t>(cid)]++;
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

            std::cerr << "PQ train subspace " << m + 1 << " / " << M << " done.\n";
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