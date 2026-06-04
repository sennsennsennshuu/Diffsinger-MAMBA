/*
 * SIMD Utilities for SSM Optimizer
 * 
 * Provides AVX2/AVX512 implementations of common operations.
 */

#ifndef SSM_SIMD_UTILS_H
#define SSM_SIMD_UTILS_H

#include <immintrin.h>
#include <cmath>

namespace ssm {

// Detect SIMD support at compile time
#if defined(__AVX512F__)
    #define SSM_USE_AVX512
#elif defined(__AVX2__)
    #define SSM_USE_AVX2
#elif defined(__AVX__)
    #define SSM_USE_AVX
#endif

// Float vector size
#ifdef SSM_USE_AVX512
    constexpr int SIMD_FLOAT_WIDTH = 16;  // 512-bit / 32-bit = 16 floats
    using FloatVec = __m512;
    #define SSM_SIMD_PREFIX _mm512_
#elif defined(SSM_USE_AVX2)
    constexpr int SIMD_FLOAT_WIDTH = 8;   // 256-bit / 32-bit = 8 floats
    using FloatVec = __m256;
    #define SSM_SIMD_PREFIX _mm256_
#else
    constexpr int SIMD_FLOAT_WIDTH = 4;   // 128-bit / 32-bit = 4 floats (SSE)
    using FloatVec = __m128;
    #define SSM_SIMD_PREFIX _mm_
#endif

// SIMD Operations
inline FloatVec simd_load(const float* ptr) {
#ifdef SSM_USE_AVX512
    return _mm512_loadu_ps(ptr);
#elif defined(SSM_USE_AVX2)
    return _mm256_loadu_ps(ptr);
#else
    return _mm_loadu_ps(ptr);
#endif
}

inline void simd_store(float* ptr, FloatVec val) {
#ifdef SSM_USE_AVX512
    _mm512_storeu_ps(ptr, val);
#elif defined(SSM_USE_AVX2)
    _mm256_storeu_ps(ptr, val);
#else
    _mm_storeu_ps(ptr, val);
#endif
}

inline FloatVec simd_set1(float val) {
#ifdef SSM_USE_AVX512
    return _mm512_set1_ps(val);
#elif defined(SSM_USE_AVX2)
    return _mm256_set1_ps(val);
#else
    return _mm_set1_ps(val);
#endif
}

inline FloatVec simd_add(FloatVec a, FloatVec b) {
#ifdef SSM_USE_AVX512
    return _mm512_add_ps(a, b);
#elif defined(SSM_USE_AVX2)
    return _mm256_add_ps(a, b);
#else
    return _mm_add_ps(a, b);
#endif
}

inline FloatVec simd_mul(FloatVec a, FloatVec b) {
#ifdef SSM_USE_AVX512
    return _mm512_mul_ps(a, b);
#elif defined(SSM_USE_AVX2)
    return _mm256_mul_ps(a, b);
#else
    return _mm_mul_ps(a, b);
#endif
}

inline FloatVec simd_fma(FloatVec a, FloatVec b, FloatVec c) {
#ifdef SSM_USE_AVX512
    return _mm512_fmadd_ps(a, b, c);
#elif defined(SSM_USE_AVX2)
    return _mm256_fmadd_ps(a, b, c);
#else
    // FMA not available in SSE, emulate
    return _mm_add_ps(_mm_mul_ps(a, b), c);
#endif
}

inline FloatVec simd_exp(FloatVec x) {
    // exp is not available as a native SIMD instruction in AVX2/AVX512.
    // Fall back to scalar exp for all platforms.
    alignas(32) float temp[SIMD_FLOAT_WIDTH];
    simd_store(temp, x);
    for (int i = 0; i < SIMD_FLOAT_WIDTH; i++) {
        temp[i] = std::exp(temp[i]);
    }
    return simd_load(temp);
}

inline FloatVec simd_max(FloatVec a, FloatVec b) {
#ifdef SSM_USE_AVX512
    return _mm512_max_ps(a, b);
#elif defined(SSM_USE_AVX2)
    return _mm256_max_ps(a, b);
#else
    return _mm_max_ps(a, b);
#endif
}

// Horizontal sum
inline float simd_hsum(FloatVec x) {
#ifdef SSM_USE_AVX512
    return _mm512_reduce_add_ps(x);
#elif defined(SSM_USE_AVX2)
    __m256 sum_halves = _mm256_hadd_ps(x, x);
    sum_halves = _mm256_hadd_ps(sum_halves, sum_halves);
    __m128 low = _mm256_castps256_ps128(sum_halves);
    __m128 high = _mm256_extractf128_ps(sum_halves, 1);
    __m128 sum = _mm_add_ps(low, high);
    return _mm_cvtss_f32(sum);
#else
    __m128 shuf = _mm_movehdup_ps(x);
    __m128 sums = _mm_add_ps(x, shuf);
    shuf = _mm_movehl_ps(shuf, sums);
    sums = _mm_add_ss(sums, shuf);
    return _mm_cvtss_f32(sums);
#endif
}

// Prefetch hints
inline void simd_prefetch(const float* ptr) {
#ifdef SSM_USE_AVX512
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#elif defined(SSM_USE_AVX2)
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#else
    _mm_prefetch(reinterpret_cast<const char*>(ptr), _MM_HINT_T0);
#endif
}

} // namespace ssm

#endif // SSM_SIMD_UTILS_H
