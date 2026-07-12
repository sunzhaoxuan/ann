#pragma once

// Architecture selector used by all ANN stages.

#if defined(__aarch64__) || defined(__ARM_NEON) || defined(__ARM_NEON__)

#include "flat_scan_simd_arm.h"

#else

#include "simd_compat.h"
#include "flat_scan_simd_x86.h"

static inline float inner_product_neon(const float* a, const float* b, size_t dim) {
    return inner_product_simd(a, b, dim);
}

static inline float inner_product_neon8(const float* a, const float* b, size_t dim) {
    return inner_product_simd(a, b, dim);
}

static inline float inner_product_neon16(const float* a, const float* b, size_t dim) {
    return inner_product_simd(a, b, dim);
}

static inline float inner_product_neon32(const float* a, const float* b, size_t dim) {
    return inner_product_simd(a, b, dim);
}

static inline float inner_product_neon16_fma(
    const float* a,
    const float* b,
    size_t dim
) {
    return inner_product_simd(a, b, dim);
}

static inline std::priority_queue<std::pair<float, uint32_t> >
flat_search_simd_fasttopk(
    float* base,
    float* query,
    size_t base_number,
    size_t vecdim,
    size_t k
) {
    return flat_search_simd(base, query, base_number, vecdim, k);
}

#endif
