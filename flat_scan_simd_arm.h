#pragma once

#include <queue>
#include <cstdint>
#include <cstddef>
#include <arm_neon.h>

// NEON 内积函数：计算 a · b
static inline float inner_product_neon(const float* a, const float* b, size_t dim) {
    size_t i = 0;

    // 一次处理 4 个 float
    float32x4_t sum_vec = vdupq_n_f32(0.0f);

    for (; i + 4 <= dim; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);

        // sum_vec += va * vb
        float32x4_t prod = vmulq_f32(va, vb);
        sum_vec = vaddq_f32(sum_vec, prod);
    }

    // 将 4 个通道的结果取出并水平求和
    float temp[4];
    vst1q_f32(temp, sum_vec);

    float result = temp[0] + temp[1] + temp[2] + temp[3];

    // 处理剩余维度，但这里其实没用，96%4=0
    /*
    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }
    */

    return result;
}

static inline float inner_product_neon8(const float* a, const float* b, size_t dim) {
    size_t i = 0;

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);

    for (; i + 8 <= dim; i += 8) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);

        sum0 = vaddq_f32(sum0, vmulq_f32(a0, b0));
        sum1 = vaddq_f32(sum1, vmulq_f32(a1, b1));
    }

    float32x4_t sum = vaddq_f32(sum0, sum1);

    float temp[4];
    vst1q_f32(temp, sum);

    float result = temp[0] + temp[1] + temp[2] + temp[3];

    return result;
}

static inline float inner_product_neon16(const float* a, const float* b, size_t dim) {
    size_t i = 0;

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (; i + 16 <= dim; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);

        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);

        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        sum0 = vaddq_f32(sum0, vmulq_f32(a0, b0));
        sum1 = vaddq_f32(sum1, vmulq_f32(a1, b1));
        sum2 = vaddq_f32(sum2, vmulq_f32(a2, b2));
        sum3 = vaddq_f32(sum3, vmulq_f32(a3, b3));
    }

    // 合并 4 条累加链
    float32x4_t sum01 = vaddq_f32(sum0, sum1);
    float32x4_t sum23 = vaddq_f32(sum2, sum3);
    float32x4_t sum = vaddq_f32(sum01, sum23);

    float temp[4];
    vst1q_f32(temp, sum);

    float result = temp[0] + temp[1] + temp[2] + temp[3];

    return result;
}

static inline float inner_product_neon32(const float* a, const float* b, size_t dim) {
    size_t i = 0;

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);
    float32x4_t sum4 = vdupq_n_f32(0.0f);
    float32x4_t sum5 = vdupq_n_f32(0.0f);
    float32x4_t sum6 = vdupq_n_f32(0.0f);
    float32x4_t sum7 = vdupq_n_f32(0.0f);

    for (; i + 32 <= dim; i += 32) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);

        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);

        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        float32x4_t a4 = vld1q_f32(a + i + 16);
        float32x4_t b4 = vld1q_f32(b + i + 16);

        float32x4_t a5 = vld1q_f32(a + i + 20);
        float32x4_t b5 = vld1q_f32(b + i + 20);

        float32x4_t a6 = vld1q_f32(a + i + 24);
        float32x4_t b6 = vld1q_f32(b + i + 24);

        float32x4_t a7 = vld1q_f32(a + i + 28);
        float32x4_t b7 = vld1q_f32(b + i + 28);

        sum0 = vaddq_f32(sum0, vmulq_f32(a0, b0));
        sum1 = vaddq_f32(sum1, vmulq_f32(a1, b1));
        sum2 = vaddq_f32(sum2, vmulq_f32(a2, b2));
        sum3 = vaddq_f32(sum3, vmulq_f32(a3, b3));
        sum4 = vaddq_f32(sum4, vmulq_f32(a4, b4));
        sum5 = vaddq_f32(sum5, vmulq_f32(a5, b5));
        sum6 = vaddq_f32(sum6, vmulq_f32(a6, b6));
        sum7 = vaddq_f32(sum7, vmulq_f32(a7, b7));
    }

    float32x4_t sum01 = vaddq_f32(sum0, sum1);
    float32x4_t sum23 = vaddq_f32(sum2, sum3);
    float32x4_t sum45 = vaddq_f32(sum4, sum5);
    float32x4_t sum67 = vaddq_f32(sum6, sum7);

    float32x4_t sum0123 = vaddq_f32(sum01, sum23);
    float32x4_t sum4567 = vaddq_f32(sum45, sum67);
    float32x4_t sum = vaddq_f32(sum0123, sum4567);

    float temp[4];
    vst1q_f32(temp, sum);

    float result = temp[0] + temp[1] + temp[2] + temp[3];

    return result;
}

// NEON + FMA + 4路累加器，一次循环处理 16 个 float
static inline float inner_product_neon16_fma(const float* __restrict__ a, const float* __restrict__ b, size_t dim) {
    size_t i = 0;

    float32x4_t sum0 = vdupq_n_f32(0.0f);
    float32x4_t sum1 = vdupq_n_f32(0.0f);
    float32x4_t sum2 = vdupq_n_f32(0.0f);
    float32x4_t sum3 = vdupq_n_f32(0.0f);

    for (; i + 16 <= dim; i += 16) {
        float32x4_t a0 = vld1q_f32(a + i);
        float32x4_t b0 = vld1q_f32(b + i);

        float32x4_t a1 = vld1q_f32(a + i + 4);
        float32x4_t b1 = vld1q_f32(b + i + 4);

        float32x4_t a2 = vld1q_f32(a + i + 8);
        float32x4_t b2 = vld1q_f32(b + i + 8);

        float32x4_t a3 = vld1q_f32(a + i + 12);
        float32x4_t b3 = vld1q_f32(b + i + 12);

        sum0 = vfmaq_f32(sum0, a0, b0);
        sum1 = vfmaq_f32(sum1, a1, b1);
        sum2 = vfmaq_f32(sum2, a2, b2);
        sum3 = vfmaq_f32(sum3, a3, b3);
    }

    float32x4_t sum01 = vaddq_f32(sum0, sum1);
    float32x4_t sum23 = vaddq_f32(sum2, sum3);
    float32x4_t sum = vaddq_f32(sum01, sum23);

    float result = vaddvq_f32(sum);

    return result;
}

// IP distance:
static inline float ip_distance_neon(const float* base_vec, const float* query, size_t vecdim) {
    return 1.0f - inner_product_neon16_fma(base_vec, query, vecdim);
}

std::priority_queue<std::pair<float, uint32_t> > flat_search_simd(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k
) {
    std::priority_queue<std::pair<float, uint32_t> > q;

    for (size_t i = 0; i < base_number; ++i) {
        const float* base_vec = base + i * vecdim;

        float dis = ip_distance_neon(base_vec, query, vecdim);

        if (q.size() < k) {
            q.push({dis, static_cast<uint32_t>(i)});
        } else {
            if (dis < q.top().first) {
                q.push({dis, static_cast<uint32_t>(i)});
                q.pop();
            }
        }
    }

    return q;
}


std::priority_queue<std::pair<float, uint32_t> > flat_search_simd_fasttopk(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k
) {
    const size_t MAX_K = 10;  // 本实验 k=10
    float best_score[MAX_K];
    uint32_t best_id[MAX_K];

    size_t cnt = 0;
    size_t worst_pos = 0;
    float worst_score = 0.0f;

    auto recompute_worst = [&]() {
        worst_pos = 0;
        worst_score = best_score[0];

        for (size_t j = 1; j < cnt; ++j) {
            if (best_score[j] < worst_score) {
                worst_score = best_score[j];
                worst_pos = j;
            }
        }
    };

    for (size_t i = 0; i < base_number; ++i) {
        ///*
        if (i + 16 < base_number) {
            __builtin_prefetch(base + (i + 16) * vecdim, 0, 1);
        }
        //*/
        const float* base_vec = base + i * vecdim;

        float score = inner_product_neon16_fma(base_vec, query, vecdim);

        if (cnt < k) {
            best_score[cnt] = score;
            best_id[cnt] = static_cast<uint32_t>(i);
            ++cnt;

            if (cnt == k) {
                recompute_worst();
            }
        } else {
            if (score > worst_score) {
                best_score[worst_pos] = score;
                best_id[worst_pos] = static_cast<uint32_t>(i);
                recompute_worst();
            }
        }
    }

    std::priority_queue<std::pair<float, uint32_t> > q;
    for (size_t j = 0; j < cnt; ++j) {
        float dis = 1.0f - best_score[j];
        q.push({dis, best_id[j]});
    }

    return q;
}