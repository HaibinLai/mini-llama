#pragma once

// GGML CPU internal header

#include "ggml.h"
#include "ggml-impl.h"

#include <stdlib.h> // load `stdlib.h` before other headers to work around MinGW bug: https://sourceforge.net/p/mingw-w64/bugs/192/
//#include <stddef.h>
#include <stdbool.h>
#include <string.h> // memcpy
#include <math.h>   // fabsf

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_compute_params {
    // ith = thread index, nth = number of threads
    int ith, nth;

    // work buffer for all threads
    size_t wsize;
    void * wdata;

    struct ggml_threadpool * threadpool;
};


#if defined(_MSC_VER)

#define m512bh(p) p
#define m512i(p) p

#else

#define m512bh(p) (__m512bh)(p)
#define m512i(p) (__m512i)(p)

#endif

// __FMA__ and __F16C__ are not defined in MSVC, however they are implied with AVX2/AVX512
// #if defined(_MSC_VER) && (defined(__AVX2__) || defined(__AVX512F__))
// #ifndef __FMA__
// #define __FMA__
// #endif
// #ifndef __F16C__
// #define __F16C__
// #endif
// #endif

// __SSE3__ and __SSSE3__ are not defined in MSVC, but SSE3/SSSE3 are present when AVX/AVX2/AVX512 are available
// #if defined(_MSC_VER) && (defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__))
// #ifndef __SSE3__
// #define __SSE3__
// #endif
// #ifndef __SSSE3__
// #define __SSSE3__
// #endif
// #endif

#if defined(__s390x__) && defined(__VEC__)
#ifndef __VXE__
#define __VXE__
#endif
#ifndef __VXE2__
#define __VXE2__
#endif
#endif

#if defined(__ARM_FEATURE_SVE)
#include <sys/prctl.h>
#endif

#if defined(__ARM_NEON)

// ref: https://github.com/ggml-org/llama.cpp/pull/5404
#ifdef _MSC_VER
#define ggml_vld1q_u32(w,x,y,z) { ((w) + ((uint64_t)(x) << 32)), ((y) + ((uint64_t)(z) << 32)) }
#else
#define ggml_vld1q_u32(w,x,y,z) { (w), (x), (y), (z) }
#endif // _MSC_VER

#if !defined(__aarch64__)



//     return res;
// }

#else

#define ggml_int16x8x2_t  int16x8x2_t
#define ggml_uint8x16x2_t uint8x16x2_t
#define ggml_uint8x16x4_t uint8x16x4_t
#define ggml_int8x16x2_t  int8x16x2_t
#define ggml_int8x16x4_t  int8x16x4_t

#define ggml_vld1q_s16_x2 vld1q_s16_x2
#define ggml_vld1q_u8_x2  vld1q_u8_x2
#define ggml_vld1q_u8_x4  vld1q_u8_x4
#define ggml_vld1q_s8_x2  vld1q_s8_x2
#define ggml_vld1q_s8_x4  vld1q_s8_x4
#define ggml_vqtbl1q_s8   vqtbl1q_s8
#define ggml_vqtbl1q_u8   vqtbl1q_u8

#endif // !defined(__aarch64__)

#if !defined(__ARM_FEATURE_DOTPROD)

inline static int32x4_t ggml_vdotq_s32(int32x4_t acc, int8x16_t a, int8x16_t b) {
    const int16x8_t p0 = vmull_s8(vget_low_s8 (a), vget_low_s8 (b));
    const int16x8_t p1 = vmull_s8(vget_high_s8(a), vget_high_s8(b));

    return vaddq_s32(acc, vaddq_s32(vpaddlq_s16(p0), vpaddlq_s16(p1)));
}

#else

#define ggml_vdotq_s32(a, b, c) vdotq_s32(a, b, c)

#endif // !defined(__ARM_FEATURE_DOTPROD)

#endif // defined(__ARM_NEON)



#if defined(_MSC_VER) || defined(__MINGW32__)

#elif defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__) || defined(__SSSE3__) || defined(__SSE3__) || defined(__SSE__)
#include <immintrin.h>
#endif

// #ifdef __riscv_v_intrinsic
// #include <riscv_vector.h>
// #endif

// #if defined(__loongarch64)
// #if defined(__loongarch_asx)
// #include <lasxintrin.h>
// #endif
// #if defined(__loongarch_sx)
// #include <lsxintrin.h>
// #endif
// #endif

//  IBM
// #if defined(__VXE__) || defined(__VXE2__)
// #include <vecintrin.h>


// #endif



// TODO: move to ggml-threading
void ggml_barrier(struct ggml_threadpool * tp);

void ggml_threadpool_chunk_set(struct ggml_threadpool * tp, int value);
int  ggml_threadpool_chunk_add(struct ggml_threadpool * tp, int value);

#ifdef __cplusplus
}
#endif
