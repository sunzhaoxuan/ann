#pragma once

#include <queue>
#include <cstdint>
#include <cstddef>
#include <immintrin.h>

// SIMD 内积函数：计算 a · b
static inline float inner_product_simd(const float* a, const float* b, size_t dim) {
    float result = 0.0f;
    size_t i = 0;

#if defined(__AVX__)
    // AVX: 一次处理 8 个 float
    __m256 sum_vec = _mm256_setzero_ps();

    for (; i + 8 <= dim; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        __m256 prod = _mm256_mul_ps(va, vb);
        sum_vec = _mm256_add_ps(sum_vec, prod);
    }

    float temp[8];
    _mm256_storeu_ps(temp, sum_vec);
    result += temp[0] + temp[1] + temp[2] + temp[3]
            + temp[4] + temp[5] + temp[6] + temp[7];

#elif defined(__SSE__)
    // SSE: 一次处理 4 个 float
    __m128 sum_vec = _mm_setzero_ps();

    for (; i + 4 <= dim; i += 4) {
        __m128 va = _mm_loadu_ps(a + i);
        __m128 vb = _mm_loadu_ps(b + i);
        __m128 prod = _mm_mul_ps(va, vb);
        sum_vec = _mm_add_ps(sum_vec, prod);
    }

    float temp[4];
    _mm_storeu_ps(temp, sum_vec);
    result += temp[0] + temp[1] + temp[2] + temp[3];

#endif

    for (; i < dim; ++i) {
        result += a[i] * b[i];
    }

    return result;
}

// dis = 1 - inner_product
static inline float ip_distance_simd(const float* base_vec, const float* query, size_t vecdim) {
    return 1.0f - inner_product_simd(base_vec, query, vecdim);
}

// SIMD 版本 Flat Search
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

        float dis = ip_distance_simd(base_vec, query, vecdim);

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
