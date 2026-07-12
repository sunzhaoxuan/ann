#pragma once

#include <cstddef>
#include <cstdint>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)

#include <arm_neon.h>

#else

template <typename T, std::size_t N>
struct AnnSimdVec {
    T lane[N];
};

using float32x4_t = AnnSimdVec<float, 4>;
using int8x8_t = AnnSimdVec<std::int8_t, 8>;
using int8x16_t = AnnSimdVec<std::int8_t, 16>;
using int16x8_t = AnnSimdVec<std::int16_t, 8>;
using int32x4_t = AnnSimdVec<std::int32_t, 4>;
using uint8x8_t = AnnSimdVec<std::uint8_t, 8>;
using uint8x16_t = AnnSimdVec<std::uint8_t, 16>;
using uint16x8_t = AnnSimdVec<std::uint16_t, 8>;
using uint32x4_t = AnnSimdVec<std::uint32_t, 4>;

static inline float32x4_t vdupq_n_f32(float value) {
    float32x4_t out{};
    for (std::size_t i = 0; i < 4; ++i) out.lane[i] = value;
    return out;
}

static inline int32x4_t vdupq_n_s32(std::int32_t value) {
    int32x4_t out{};
    for (std::size_t i = 0; i < 4; ++i) out.lane[i] = value;
    return out;
}

static inline uint32x4_t vdupq_n_u32(std::uint32_t value) {
    uint32x4_t out{};
    for (std::size_t i = 0; i < 4; ++i) out.lane[i] = value;
    return out;
}

static inline float32x4_t vld1q_f32(const float* input) {
    float32x4_t out{};
    for (std::size_t i = 0; i < 4; ++i) out.lane[i] = input[i];
    return out;
}

static inline int8x16_t vld1q_s8(const std::int8_t* input) {
    int8x16_t out{};
    for (std::size_t i = 0; i < 16; ++i) out.lane[i] = input[i];
    return out;
}

static inline uint8x16_t vld1q_u8(const std::uint8_t* input) {
    uint8x16_t out{};
    for (std::size_t i = 0; i < 16; ++i) out.lane[i] = input[i];
    return out;
}

static inline void vst1q_f32(float* output, float32x4_t value) {
    for (std::size_t i = 0; i < 4; ++i) output[i] = value.lane[i];
}

static inline float32x4_t vaddq_f32(float32x4_t a, float32x4_t b) {
    float32x4_t out{};
    for (std::size_t i = 0; i < 4; ++i) out.lane[i] = a.lane[i] + b.lane[i];
    return out;
}

static inline int32x4_t vaddq_s32(int32x4_t a, int32x4_t b) {
    int32x4_t out{};
    for (std::size_t i = 0; i < 4; ++i) out.lane[i] = a.lane[i] + b.lane[i];
    return out;
}

static inline uint32x4_t vaddq_u32(uint32x4_t a, uint32x4_t b) {
    uint32x4_t out{};
    for (std::size_t i = 0; i < 4; ++i) out.lane[i] = a.lane[i] + b.lane[i];
    return out;
}

static inline float32x4_t vsubq_f32(float32x4_t a, float32x4_t b) {
    float32x4_t out{};
    for (std::size_t i = 0; i < 4; ++i) out.lane[i] = a.lane[i] - b.lane[i];
    return out;
}

static inline float32x4_t vmulq_f32(float32x4_t a, float32x4_t b) {
    float32x4_t out{};
    for (std::size_t i = 0; i < 4; ++i) out.lane[i] = a.lane[i] * b.lane[i];
    return out;
}

static inline float32x4_t vfmaq_f32(float32x4_t acc, float32x4_t a, float32x4_t b) {
    for (std::size_t i = 0; i < 4; ++i) acc.lane[i] += a.lane[i] * b.lane[i];
    return acc;
}

static inline float vaddvq_f32(float32x4_t value) {
    return value.lane[0] + value.lane[1] + value.lane[2] + value.lane[3];
}

static inline std::int32_t vaddvq_s32(int32x4_t value) {
    return value.lane[0] + value.lane[1] + value.lane[2] + value.lane[3];
}

static inline std::uint32_t vaddvq_u32(uint32x4_t value) {
    return value.lane[0] + value.lane[1] + value.lane[2] + value.lane[3];
}

static inline int8x8_t vget_low_s8(int8x16_t value) {
    int8x8_t out{};
    for (std::size_t i = 0; i < 8; ++i) out.lane[i] = value.lane[i];
    return out;
}

static inline int8x8_t vget_high_s8(int8x16_t value) {
    int8x8_t out{};
    for (std::size_t i = 0; i < 8; ++i) out.lane[i] = value.lane[i + 8];
    return out;
}

static inline uint8x8_t vget_low_u8(uint8x16_t value) {
    uint8x8_t out{};
    for (std::size_t i = 0; i < 8; ++i) out.lane[i] = value.lane[i];
    return out;
}

static inline uint8x8_t vget_high_u8(uint8x16_t value) {
    uint8x8_t out{};
    for (std::size_t i = 0; i < 8; ++i) out.lane[i] = value.lane[i + 8];
    return out;
}

static inline int16x8_t vmull_s8(int8x8_t a, int8x8_t b) {
    int16x8_t out{};
    for (std::size_t i = 0; i < 8; ++i) {
        out.lane[i] = static_cast<std::int16_t>(a.lane[i]) * static_cast<std::int16_t>(b.lane[i]);
    }
    return out;
}

static inline uint16x8_t vmull_u8(uint8x8_t a, uint8x8_t b) {
    uint16x8_t out{};
    for (std::size_t i = 0; i < 8; ++i) {
        out.lane[i] = static_cast<std::uint16_t>(a.lane[i]) * static_cast<std::uint16_t>(b.lane[i]);
    }
    return out;
}

static inline int32x4_t vpadalq_s16(int32x4_t acc, int16x8_t value) {
    for (std::size_t i = 0; i < 4; ++i) {
        acc.lane[i] += static_cast<std::int32_t>(value.lane[i * 2])
                     + static_cast<std::int32_t>(value.lane[i * 2 + 1]);
    }
    return acc;
}

static inline uint32x4_t vpadalq_u16(uint32x4_t acc, uint16x8_t value) {
    for (std::size_t i = 0; i < 4; ++i) {
        acc.lane[i] += static_cast<std::uint32_t>(value.lane[i * 2])
                     + static_cast<std::uint32_t>(value.lane[i * 2 + 1]);
    }
    return acc;
}

#endif
