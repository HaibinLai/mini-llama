// Copyright 2024 Mozilla Foundation
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the
// "Software"), to deal in the Software without restriction, including
// without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to
// permit persons to whom the Software is furnished to do so, subject to
// the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
// BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN
// ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
// CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

//
//                   _   _          ___ _      _   ___
//                  | |_(_)_ _ _  _| _ ) |    /_\ / __|
//                  |  _| | ' \ || | _ \ |__ / _ \\__ \.
//                   \__|_|_||_\_, |___/____/_/ \_\___/
//                             |__/
//
//                    BASIC LINEAR ALGEBRA SUBPROGRAMS
//
//
// This file implements multithreaded CPU matrix multiplication for the
// common contiguous use case C = Aᵀ * B. These kernels are designed to
// have excellent performance[1] for matrices that fit in the CPU cache
// without imposing any overhead such as cache filling or malloc calls.
//
// This implementation does not guarantee any upper bound with rounding
// errors, which grow along with k. Our goal's to maximally exploit the
// hardware for performance, and then use whatever resources remain for
// improving numerical accuracy.
//
// [1] J. Tunney, ‘LLaMA Now Goes Faster on CPUs’, Mar. 2024. [Online].
//     Available: https://justine.lol/matmul/. [Accessed: 29-Mar-2024].

#if defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wpedantic"
#pragma GCC diagnostic ignored "-Wignored-attributes"
#endif

#include "sgemm.h"
#include "ggml-impl.h"
#include "ggml-cpu-impl.h"
#include "ggml-quants.h"

#include <array>
#include <type_traits>

#ifdef _MSC_VER
#define NOINLINE __declspec(noinline)
#else
#define NOINLINE __attribute__((__noinline__))
#endif

#if defined(__ARM_NEON) || defined(__AVX512F__) || defined(__VXE__) || defined(__VXE2__)
#define VECTOR_REGISTERS 32
#else
#define VECTOR_REGISTERS 16
#endif

#define MM256_SET_M128I(a, b) _mm256_insertf128_si256(_mm256_castsi128_si256(b), (a), 1)

namespace {

inline float unhalf(ggml_fp16_t d) {
    return GGML_FP16_TO_FP32(d);
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED ARITHMETIC OPERATIONS

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline __m128 add(__m128 x, __m128 y) { return _mm_add_ps(x, y); }
inline __m128 sub(__m128 x, __m128 y) { return _mm_sub_ps(x, y); }
inline __m128 mul(__m128 x, __m128 y) { return _mm_mul_ps(x, y); }
#endif  // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline __m256 add(__m256 x, __m256 y) { return _mm256_add_ps(x, y); }
inline __m256 sub(__m256 x, __m256 y) { return _mm256_sub_ps(x, y); }
inline __m256 mul(__m256 x, __m256 y) { return _mm256_mul_ps(x, y); }
#endif // __AVX__

#if defined(__AVX512F__)
inline __m512 add(__m512 x, __m512 y) { return _mm512_add_ps(x, y); }
inline __m512 sub(__m512 x, __m512 y) { return _mm512_sub_ps(x, y); }
inline __m512 mul(__m512 x, __m512 y) { return _mm512_mul_ps(x, y); }
#endif // __AVX512F__

#if defined(__ARM_NEON)
inline float32x4_t add(float32x4_t x, float32x4_t y) { return vaddq_f32(x, y); }
inline float32x4_t sub(float32x4_t x, float32x4_t y) { return vsubq_f32(x, y); }
inline float32x4_t mul(float32x4_t x, float32x4_t y) { return vmulq_f32(x, y); }
#endif // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
inline float16x8_t add(float16x8_t x, float16x8_t y) { return vaddq_f16(x, y); }
inline float16x8_t sub(float16x8_t x, float16x8_t y) { return vsubq_f16(x, y); }
inline float16x8_t mul(float16x8_t x, float16x8_t y) { return vmulq_f16(x, y); }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if defined(__VXE__) || defined(__VXE2__)
inline float32x4_t add(float32x4_t x, float32x4_t y) { return vec_add(x, y); }
inline float32x4_t sub(float32x4_t x, float32x4_t y) { return vec_sub(x, y); }
inline float32x4_t mul(float32x4_t x, float32x4_t y) { return vec_mul(x, y); }
#endif

#if defined(__MMA__)
typedef vector unsigned char vec_t;
typedef __vector_quad acc_t;
#endif
////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED FUSED MULTIPLY ADD

/**
 * Computes a * b + c.
 */
template <typename T, typename U>
inline U madd(T a, T b, U c) {
    return add(mul(a, b), c);
}

#if defined(__FMA__)
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <>
inline __m256 madd(__m256 a, __m256 b, __m256 c) {
    return _mm256_fmadd_ps(a, b, c);
}
#endif
#if defined(__AVX512F__)
template <>
inline __m512 madd(__m512 a, __m512 b, __m512 c) {
    return _mm512_fmadd_ps(a, b, c);
}
#endif
#if defined(__AVX512BF16__)
template <>
inline __m512 madd(__m512bh a, __m512bh b, __m512 c) {
    return _mm512_dpbf16_ps(c, a, b);
}
template <>
inline __m256 madd(__m256bh a, __m256bh b, __m256 c) {
    return _mm256_dpbf16_ps(c, a, b);
}
#endif
#endif

#if defined(__ARM_FEATURE_FMA)
// template <>
// inline float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
//     return vfmaq_f32(c, b, a);
// }
// #if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
// template <>
// inline float16x8_t madd(float16x8_t a, float16x8_t b, float16x8_t c) {
//     return vfmaq_f16(c, b, a);
// }
// #endif
#endif

#if defined(__VXE__) || defined(__VXE2__)
template <>
inline float32x4_t madd(float32x4_t a, float32x4_t b, float32x4_t c) {
    return vec_madd(a, b, c);
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED HORIZONTAL SUM

#if defined(__ARM_NEON)
// inline float hsum(float32x4_t x) {
//     return vaddvq_f32(x);
// }
#endif // __ARM_NEON

#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
// inline float hsum(float16x8_t x) {
//     return vaddvq_f32(vaddq_f32(vcvt_f32_f16(vget_low_f16(x)),
//                                 vcvt_f32_f16(vget_high_f16(x))));
// }
#endif // __ARM_FEATURE_FP16_VECTOR_ARITHMETIC

#if defined(__VXE__) || defined(__VXE2__)
inline float hsum(float32x4_t x) {
    float32x4_t tmp = x + vec_reve(x);
    return tmp[0] + tmp[1];
}
#endif

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m128 x) {
#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
    x = _mm_add_ps(x, _mm_movehl_ps(x, x));
    x = _mm_add_ss(x, _mm_movehdup_ps(x));
#else
    __m128 t;
    t = _mm_shuffle_ps(x, x, _MM_SHUFFLE(2, 3, 0, 1));
    x = _mm_add_ps(x, t);
    t = _mm_movehl_ps(t, x);
    x = _mm_add_ss(x, t);
#endif
    return _mm_cvtss_f32(x);
}
#endif

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
inline float hsum(__m256 x) {
    return hsum(_mm_add_ps(_mm256_extractf128_ps(x, 1),
                           _mm256_castps256_ps128(x)));
}
#endif // __AVX__

#if defined(__AVX512F__)
inline float hsum(__m512 x) {
    return _mm512_reduce_add_ps(x);
}
#endif // __AVX512F__

////////////////////////////////////////////////////////////////////////////////////////////////////
// VECTORIZED MEMORY LOADING

template <typename T, typename U> T load(const U *);

#if defined(__ARM_NEON)
// template <> inline float32x4_t load(const float *p) {
//     return vld1q_f32(p);
// }
// #if !defined(_MSC_VER)
// // FIXME: this should check for __ARM_FEATURE_FP16_VECTOR_ARITHMETIC
// template <> inline float16x8_t load(const ggml_fp16_t *p) {
//     return vld1q_f16((const float16_t *)p);
// }
// template <> inline float32x4_t load(const ggml_fp16_t *p) {
//     return vcvt_f32_f16(vld1_f16((const float16_t *)p));
// }
// #endif // _MSC_VER
#endif // __ARM_NEON

#if defined(__VXE__) || defined(__VXE2__)
template <> inline float32x4_t load(const ggml_fp16_t * p) {
    float tmp[4];

    for (int i = 0; i < 4; i++) {
        tmp[i] = GGML_FP16_TO_FP32(p[i]);
    }

    return vec_xl(0, (const float *)(tmp));
}
template <> inline float32x4_t load(const float * p) {
    return vec_xl(0, p);
}
#endif

#if defined(__SSE__) || defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m128 load(const float *p) {
    return _mm_loadu_ps(p);
}
#endif  // __SSE__

#if defined(__AVX__) || defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m256 load(const float *p) {
    return _mm256_loadu_ps(p);
}
#endif // __AVX__

#if defined(__AVX2__) || defined(__AVX512F__)
template <> inline __m256 load(const ggml_bf16_t *p) {
    return _mm256_castsi256_ps(
        _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i *)p)), 16));
}
#endif // __AVX2__

#if defined(__F16C__)
template <> inline __m256 load(const ggml_fp16_t *p) {
    return _mm256_cvtph_ps(_mm_loadu_si128((const __m128i *)p));
}
#endif // __F16C__

#if defined(__AVX512F__)
template <> inline __m512 load(const float *p) {
    return _mm512_loadu_ps(p);
}
template <> inline __m512 load(const ggml_fp16_t *p) {
    return _mm512_cvtph_ps(_mm256_loadu_si256((const __m256i *)p));
}
template <> inline __m512 load(const ggml_bf16_t *p) {
    return _mm512_castsi512_ps(
        _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i *)p)), 16));
}
#endif // __AVX512F__

#if defined(__AVX512BF16__)
template <> inline __m512bh load(const ggml_bf16_t *p) {
    return (__m512bh)_mm512_loadu_ps((const float *)p);
}
template <> inline __m256bh load(const ggml_bf16_t *p) {
    return (__m256bh)_mm256_loadu_ps((const float *)p);
}
template <> inline __m512bh load(const float *p) {
    return _mm512_cvtne2ps_pbh(_mm512_loadu_ps(p + 16), _mm512_loadu_ps(p));
}
template <> inline __m256bh load(const float *p) {
    return _mm512_cvtneps_pbh(_mm512_loadu_ps(p));
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
// FLOATING POINT MATRIX MULTIPLICATION

template <int M>
static inline int64_t BLOCK_SIZE(size_t m) {
    const int64_t NB_BLOC_M = (m + M - 1) / M;
    return (m % NB_BLOC_M == 0) ? m / NB_BLOC_M : (m / NB_BLOC_M) + 1;
}

static constexpr inline int64_t BLOC_POS(int64_t ib, int64_t ibN, int64_t bloc_size) {
    return ib < ibN ? ib * bloc_size : ibN * bloc_size + (ib - ibN) * (bloc_size - 1);
}

template <int KN, typename D, typename V, typename TA, typename TB, typename TC>
class tinyBLAS {
  public:
    tinyBLAS(const ggml_compute_params * params, int64_t k,
             const TA *A, int64_t lda,
             const TB *B, int64_t ldb,
             TC *C, int64_t ldc)
        : params(params), A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc) {
    }

    bool matmul(int64_t m, int64_t n) {
        if (k % KN != 0)
            return false;
        // compute RM for only need tile with size RM&RM-1
#if VECTOR_REGISTERS == 32
        if (m % 16 == 0 && (m/16 >= params->nth)) {
            const int64_t SIZE_N = BLOCK_SIZE<6>(n);
            mnpack<4, 6, 4>(m, n, SIZE_N, 12);
            return true;
        }
        if (m % 8 == 0 ) {
            const int64_t SIZE_N = BLOCK_SIZE<6>(n);
            mnpack<4, 6, 2>(m, n, SIZE_N, 12);
            return true;
        }
        if (m % 4 == 0) {
            const int64_t SIZE_N = BLOCK_SIZE<6>(n);
            mnpack<4, 6, 1>(m, n, SIZE_N, 12);
            return true;
        }
#else  // VECTOR_REGISTERS == 16
        if (m % 16 == 0 && (m/16 >= params->nth)) {
            const int64_t SIZE_N = BLOCK_SIZE<3>(n);
            mnpack<4, 3, 4>(m, n, SIZE_N, 24);
            return true;
        }
        if (m % 8 == 0 ) {
            const int64_t SIZE_N = BLOCK_SIZE<3>(n);
            mnpack<4, 3, 2>(m, n, SIZE_N, 24);
            return true;
        }
        if (m % 4 == 0) {
            const int64_t SIZE_N = BLOCK_SIZE<3>(n);
            mnpack<4, 3, 1>(m, n, SIZE_N, 24);
            return true;
        }
#endif
        return false;
    }

  private:
    template <int RM, int RN, int BM>
    inline void mnpack(int64_t m, int64_t n, int64_t SIZE_N, int64_t BN) {
        if (SIZE_N == RN) {
            return gemm<RM, RN, BM>(m, n, BN);
        }
        if constexpr (RN > 1) {
            return mnpack<RM, RN-1, BM>(m, n, SIZE_N, BN);
        } else {
            GGML_LOG_ERROR("mnpack<%d, %d> bloc size not supported\n", RM, (int)SIZE_N);
            GGML_ASSERT(false); // we have miss something.
        }
    }

    template <int RM, int RN>
    inline void gemm_bloc(int64_t ii, int64_t jj) {
        D Cv[RN][RM] = {};
        for (int64_t l = 0; l < k; l += KN) {
            // help compiler for op order.
            if constexpr (RM <= RN) {
                V Av[RM];
                for (int64_t i = 0; i < RM; ++i) {
                    Av[i] = load<V>(A + lda * (ii + i) + l);
                }
                for (int64_t j = 0; j < RN; ++j) {
                    V Bv = load<V>(B + ldb * (jj + j) + l);
                    for (int64_t i = 0; i < RM; ++i) {
                        Cv[j][i] = madd(Av[i], Bv, Cv[j][i]);
                    }
                }
            } else {
                V Bv[RN];
                for (int64_t j = 0; j < RN; ++j) {
                    Bv[j] = load<V>(B + ldb * (jj + j) + l);
                }
                for (int64_t i = 0; i < RM; ++i) {
                    V Av = load<V>(A + lda * (ii + i) + l);
                    for (int64_t j = 0; j < RN; ++j) {
                        Cv[j][i] = madd(Av, Bv[j], Cv[j][i]);
                    }
                }
            }
        }
        for (int64_t j = 0; j < RN; ++j)
            for (int64_t i = 0; i < RM; ++i)
                C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
    }

    template <int RM, int RN, int BM>
    NOINLINE void gemm(int64_t m, int64_t n, int64_t BN) {
        GGML_ASSERT(m % (RM * BM) == 0);
        const int64_t ytiles = m / (RM * BM);
        const int64_t xtiles = (n + RN -1) / RN;
        const int64_t jj_RN = (xtiles - (xtiles * RN - n));

        // "round" bloc_size to "nearest" BN
        const int64_t NB_BN = xtiles < BN ? 1 : (xtiles + BN / 2) / BN;
        const int64_t SIZE_BN = xtiles % NB_BN == 0 ? xtiles / NB_BN : xtiles / NB_BN + 1;
        const int64_t jj_BN = (NB_BN - (NB_BN * SIZE_BN - xtiles));
        const int64_t nb_job = ytiles * NB_BN;

        if (params->ith == 0) {
            GGML_ASSERT( jj_BN * SIZE_BN + (NB_BN - jj_BN) * (SIZE_BN - 1) == xtiles);
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_set(params->threadpool, params->nth);
        }

        ggml_barrier(params->threadpool);

        int64_t job = params->ith;
        while (job < nb_job) {
            const int64_t ii = (job % ytiles) * RM * BM;
            const int64_t jb =  job / ytiles;
            const int64_t jr0 = BLOC_POS(jb  , jj_BN, SIZE_BN);
            const int64_t jrN = BLOC_POS(jb+1, jj_BN, SIZE_BN);

            const int64_t jj0 = BLOC_POS(jr0, jj_RN, RN);
            const int64_t jj2 = BLOC_POS(jrN, jj_RN, RN);
            const int64_t jj1 = jj2 < jj_RN * RN ? jj2 : jj_RN * RN;

            for (int64_t bi = 0; bi < BM * RM; bi += RM) {
                int64_t jj = jj0;
                for (; jj < jj1; jj += RN) {
                    gemm_bloc<RM, RN>(ii + bi, jj);
                }
                if constexpr (RN > 1) {
                    for (; jj < jj2; jj += RN - 1) {
                        gemm_bloc<RM, RN-1>(ii + bi, jj);
                    }
                }
                GGML_ASSERT(jj == jj2);
            }

            job = ggml_threadpool_chunk_add(params->threadpool, 1);
        }

        ggml_barrier(params->threadpool);
        return;
    }

    const ggml_compute_params * params;
    const TA *const A;
    const TB *const B;
    TC *const C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
};

//////////////////////////////////////////////////////////////////////////////////////////
// QUANT ZERO MATRIX MULTIPLICATION

#if defined(__ARM_FEATURE_DOTPROD)
// This is a specialized implementation for ARMv8.2+ with dot product support.
#endif // __ARM_FEATURE_DOTPROD

#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
template <typename TA, typename TB, typename TC>
class tinyBLAS_Q0_AVX {
  public:
    tinyBLAS_Q0_AVX(int64_t k,
                    const TA *A, int64_t lda,
                    const TB *B, int64_t ldb,
                    TC *C, int64_t ldc,
                    int ith, int nth)
        : A(A), B(B), C(C), k(k), lda(lda), ldb(ldb), ldc(ldc), ith(ith), nth(nth) {
        const int8_t kvalues_iq4nl[16] = {
            -127, -104, -83, -65,
            -49,  -35,  -22, -10,
              1,   13,   25,  38,
             53,   69,   89, 113
        };

        iq4nlt = _mm_loadu_si128((const __m128i *)kvalues_iq4nl);
    }

    void matmul(int64_t m, int64_t n) {
        mnpack(0, m, 0, n);
    }

  private:
    void mnpack(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t mc, nc, mp, np;
        switch ((MIN(m - m0, 4) << 4) | MIN(n - n0, 4)) {
#if VECTOR_REGISTERS == 32
        case 0x44:
            mc = 4;
            nc = 4;
#if defined(__AVX2__) && defined(__F16C__)
            gemm4xN<4>(m0, m, n0, n);
#else
            gemm<4, 4>(m0, m, n0, n);
#endif
            break;
        case 0x43:
            mc = 4;
            nc = 3;
#if defined(__AVX2__) && defined(__F16C__)
            gemm4xN<3>(m0, m, n0, n);
#else
            gemm<4, 3>(m0, m, n0, n);
#endif
            break;
        case 0x34:
            mc = 3;
            nc = 4;
#if defined(__AVX2__) && defined(__F16C__)
            gemmMx4<3>(m0, m, n0, n);
#else
            gemm<3, 4>(m0, m, n0, n);
#endif
            break;
        case 0x33:
            mc = 3;
            nc = 3;
            gemm<3, 3>(m0, m, n0, n);
            break;
        case 0x42:
            mc = 4;
            nc = 2;
#if defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#else
            gemm<4, 2>(m0, m, n0, n);
#endif
            break;
        case 0x24:
            mc = 2;
            nc = 4;
#if defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#else
            gemm<2, 4>(m0, m, n0, n);
#endif
            break;
#else
        case 0x44:
        case 0x43:
        case 0x42:
            mc = 4;
            nc = 2;
#if defined(__AVX2__) && defined(__F16C__)
            gemm4xN<2>(m0, m, n0, n);
#else
            gemm<4, 2>(m0, m, n0, n);
#endif
            break;
        case 0x34:
        case 0x24:
            mc = 2;
            nc = 4;
#if defined(__AVX2__) && defined(__F16C__)
            gemmMx4<2>(m0, m, n0, n);
#else
            gemm<2, 4>(m0, m, n0, n);
#endif
            break;
        case 0x33:
#endif
        case 0x32:
            mc = 3;
            nc = 2;
            gemm<3, 2>(m0, m, n0, n);
            break;
        case 0x23:
            mc = 2;
            nc = 3;
            gemm<2, 3>(m0, m, n0, n);
            break;
        case 0x41:
            mc = 4;
            nc = 1;
#if defined(__AVX2__) && defined(__F16C__)
            gemm4xN<1>(m0, m, n0, n);
#else
            gemm<4, 1>(m0, m, n0, n);
#endif
            break;
        case 0x22:
            mc = 2;
            nc = 2;
            gemm<2, 2>(m0, m, n0, n);
            break;
        case 0x14:
            mc = 1;
            nc = 4;
#if defined(__AVX2__) && defined(__F16C__)
            gemmMx4<1>(m0, m, n0, n);
#else
            gemm<1, 4>(m0, m, n0, n);
#endif
            break;
        case 0x31:
            mc = 3;
            nc = 1;
            gemm<3, 1>(m0, m, n0, n);
            break;
        case 0x13:
            mc = 1;
            nc = 3;
            gemm<1, 3>(m0, m, n0, n);
            break;
        case 0x21:
            mc = 2;
            nc = 1;
            gemm<2, 1>(m0, m, n0, n);
            break;
        case 0x12:
            mc = 1;
            nc = 2;
            gemm<1, 2>(m0, m, n0, n);
            break;
        case 0x11:
            mc = 1;
            nc = 1;
            gemm<1, 1>(m0, m, n0, n);
            break;
        default:
            return;
        }
        mp = m0 + (m - m0) / mc * mc;
        np = n0 + (n - n0) / nc * nc;
        mnpack(mp, m, n0, np);
        mnpack(m0, m, np, n);
    }

#if defined(__AVX2__) && defined(__F16C__)
// Templated functions for gemm of dimensions 4xN
    template <int RN>
    NOINLINE void gemm4xN(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / 4;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * 4;
            int64_t jj = n0 + job % xtiles * RN;
            __m256 Cv[RN][4] = {};
            for (int64_t l = 0; l < k; ++l) {
                uint64_t a_delta = ((uint64_t)A[lda * (ii + 3) + l].d << 48) | ((uint64_t)A[lda * (ii + 2) + l].d << 32) | ((uint64_t)A[lda * (ii + 1) + l].d << 16) | (A[lda * (ii + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 da = _mm_cvtph_ps(_mm_set_epi64x(0, a_delta));
                __m256i avec0 = load(A + lda * (ii + 0) + l);
                __m256i avec1 = load(A + lda * (ii + 1) + l);
                __m256i avec2 = load(A + lda * (ii + 2) + l);
                __m256i avec3 = load(A + lda * (ii + 3) + l);
                for (int64_t j = 0; j < RN; ++j) {
                        __m128 db = _mm_set1_ps(unhalf(B[ldb * (jj + j) + l].d));
                        // Computation of product of delta values for four blocks and replicate it across 256 bit lane
                        __m256 dvec =  _mm256_castps128_ps256(_mm_mul_ps(da, db));
                        dvec = _mm256_permute2f128_ps(dvec ,dvec, 0);
                        // Computation of dot product and multiplication with appropriate delta value products
                        Cv[j][0] = madd(_mm256_shuffle_ps(dvec, dvec, 0),
                                    updot(_mm256_sign_epi8(avec0, avec0),
                                          _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec0)),
                                    Cv[j][0]);
                        Cv[j][1] = madd(_mm256_shuffle_ps(dvec, dvec, 85),
                                    updot(_mm256_sign_epi8(avec1, avec1),
                                            _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec1)),
                                    Cv[j][1]);
                        Cv[j][2] = madd(_mm256_shuffle_ps(dvec, dvec, 170),
                                    updot(_mm256_sign_epi8(avec2, avec2),
                                            _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec2)),
                                    Cv[j][2]);
                        Cv[j][3] = madd(_mm256_shuffle_ps(dvec, dvec, 255),
                                    updot(_mm256_sign_epi8(avec3, avec3),
                                            _mm256_sign_epi8(load(B + ldb * (jj + j) + l), avec3)),
                                    Cv[j][3]);
                }
            }

            for (int64_t j = 0; j < RN; ++j)
                for (int64_t i = 0; i < 4; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    // Templated functions for gemm of dimensions Mx4
    template <int RM>
    NOINLINE void gemmMx4(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / 4;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * 4;
            __m256 Cv[4][RM] = {};
            for (int64_t l = 0; l < k; ++l) {
                uint64_t b_delta = ((uint64_t)B[ldb * (jj + 3) + l].d << 48) | ((uint64_t)B[ldb * (jj + 2) + l].d << 32) | ((uint64_t)B[ldb * (jj + 1) + l].d << 16) | (B[ldb * (jj + 0) + l].d);
                // Convert delta values for four blocks to float values
                __m128 db = _mm_cvtph_ps(_mm_set_epi64x(0, b_delta));
                __m256i bvec0 = load(B + ldb * (jj + 0) + l);
                __m256i bvec1 = load(B + ldb * (jj + 1) + l);
                __m256i bvec2 = load(B + ldb * (jj + 2) + l);
                __m256i bvec3 = load(B + ldb * (jj + 3) + l);
                for (int64_t i = 0; i < RM; ++i) {
                    __m128 da = _mm_set1_ps(unhalf((A[lda * (ii + i) + l].d)));
                    // Computation of product of delta values for four blocks and replicate it across 256 bit lane
                    __m256 dvec =  _mm256_castps128_ps256(_mm_mul_ps(da, db));
                    dvec = _mm256_permute2f128_ps(dvec ,dvec, 0);
                    // Computation of dot product and multiplication with appropriate delta value products
                    Cv[0][i] = madd(_mm256_shuffle_ps(dvec, dvec, 0),
                                    updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                            load(A + lda * (ii + i) + l)),
                                            _mm256_sign_epi8(bvec0, load(A + lda * (ii + i) + l))),
                                    Cv[0][i]);
                    Cv[1][i] = madd(_mm256_shuffle_ps(dvec, dvec, 85),
                                    updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                            load(A + lda * (ii + i) + l)),
                                            _mm256_sign_epi8(bvec1, load(A + lda * (ii + i) + l))),
                                    Cv[1][i]);
                    Cv[2][i] = madd(_mm256_shuffle_ps(dvec, dvec, 170),
                                    updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                            load(A + lda * (ii + i) + l)),
                                            _mm256_sign_epi8(bvec2, load(A + lda * (ii + i) + l))),
                                    Cv[2][i]);
                    Cv[3][i] = madd(_mm256_shuffle_ps(dvec, dvec, 255),
                                    updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                            load(A + lda * (ii + i) + l)),
                                            _mm256_sign_epi8(bvec3, load(A + lda * (ii + i) + l))),
                                    Cv[3][i]);
                }
            }
            for (int64_t j = 0; j < 4; ++j)
                for (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }
#endif

    template <int RM, int RN>
    NOINLINE void gemm(int64_t m0, int64_t m, int64_t n0, int64_t n) {
        int64_t ytiles = (m - m0) / RM;
        int64_t xtiles = (n - n0) / RN;
        int64_t tiles = xtiles * ytiles;
        int64_t duty = (tiles + nth - 1) / nth;
        int64_t start = duty * ith;
        int64_t end = start + duty;
        if (end > tiles)
            end = tiles;
        for (int64_t job = start; job < end; ++job) {
            int64_t ii = m0 + job / xtiles * RM;
            int64_t jj = n0 + job % xtiles * RN;
            __m256 Cv[RN][RM] = {};
            for (int64_t l = 0; l < k; ++l)
                for (int64_t j = 0; j < RN; ++j)
                    for (int64_t i = 0; i < RM; ++i) {
#if defined(__AVX2__)
                        __m256 udTmp = updot(_mm256_sign_epi8(load(A + lda * (ii + i) + l),
                                                              load(A + lda * (ii + i) + l)),
                                             _mm256_sign_epi8(load(B + ldb * (jj + j) + l),
                                                              load(A + lda * (ii + i) + l)));
#else
                        __m128i ali0 = load0(A + lda * (ii + i) + l);
                        __m128i ali1 = load1(A + lda * (ii + i) + l);
                        __m128i blj0 = load0(B + ldb * (jj + j) + l);
                        __m128i blj1 = load1(B + ldb * (jj + j) + l);

                        __m128i sepAA0 = _mm_sign_epi8(ali0, ali0);
                        __m128i sepAA1 = _mm_sign_epi8(ali1, ali1);
                        __m128i sepBA0 = _mm_sign_epi8(blj0, ali0);
                        __m128i sepBA1 = _mm_sign_epi8(blj1, ali1);

                        // updot
                        const __m128i oneFill = _mm_set1_epi16(1);
                        __m128i mad0 = _mm_maddubs_epi16(sepAA0, sepBA0);
                        __m128i mad1 = _mm_maddubs_epi16(sepAA1, sepBA1);
                        __m256 udTmp = _mm256_cvtepi32_ps(MM256_SET_M128I(_mm_madd_epi16(oneFill, mad1), _mm_madd_epi16(oneFill, mad0)));
#endif
                        Cv[j][i] = madd(_mm256_set1_ps(unhalf(A[lda * (ii + i) + l].d) *
                                                       unhalf(B[ldb * (jj + j) + l].d)),
                                                       udTmp,
                                                       Cv[j][i]);
                    }
            for (int64_t j = 0; j < RN; ++j)
                for (int64_t i = 0; i < RM; ++i)
                    C[ldc * (jj + j) + (ii + i)] = hsum(Cv[j][i]);
        }
    }

    inline __m256i load(const block_q8_0 *b) {
        return _mm256_loadu_si256((const __m256i *)b->qs);
    }

    inline __m128i load0(const block_q8_0 *b) {
        return _mm_loadu_si128((const __m128i *)b->qs);
    }

    inline __m128i load1(const block_q8_0 *b) {
        return _mm_loadu_si128(((const __m128i *)b->qs) + 1);
    }

    inline __m256i load(const block_q4_0 *b) {
        return _mm256_sub_epi8(denibble(b->qs), _mm256_set1_epi8(8));
    }

    inline __m128i load0(const block_q4_0 *b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        return _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), x), _mm_set1_epi8(8));
    }

    inline __m128i load1(const block_q4_0 *b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        return _mm_sub_epi8(_mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)), _mm_set1_epi8(8));
    }

    inline __m256i load(const block_q5_0 *b) {
        return _mm256_or_si256(denibble(b->qs), bittobyte(b->qh));
    }

    inline __m128i load0(const block_q5_0* b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        uint32_t x32;
        memcpy(&x32, b->qh, sizeof(uint32_t));
        __m128i qxl = _mm_and_si128(_mm_set1_epi8(15), x);
        __m128i bytesl = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      _mm_set_epi64x(0x0101010101010101, 0x0000000000000000))));
        bytesl = _mm_andnot_si128(bytesl, _mm_set1_epi8((char)0xF0));
        return _mm_or_si128(qxl, bytesl);
    }

    inline __m128i load1(const block_q5_0* b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        uint32_t x32;
        memcpy(&x32, b->qh, sizeof(uint32_t));
        __m128i qxh = _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4));
        __m128i bytesh = _mm_cmpeq_epi8(_mm_set1_epi64x(-1),
                                        _mm_or_si128(_mm_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                     _mm_shuffle_epi8(_mm_set1_epi32(x32),
                                                                      _mm_set_epi64x(0x0303030303030303, 0x0202020202020202))));
        bytesh = _mm_andnot_si128(bytesh, _mm_set1_epi8((char)0xF0));
        return _mm_or_si128(qxh, bytesh);
    }

    inline __m256i load(const block_iq4_nl *b) {
        return MM256_SET_M128I(load1(b), load0(b));
    }

    inline __m128i load0(const block_iq4_nl *b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        return _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), x));
    }

    inline __m128i load1(const block_iq4_nl *b) {
        const __m128i x = _mm_loadu_si128((const __m128i *)(b->qs));
        return _mm_shuffle_epi8(iq4nlt, _mm_and_si128(_mm_set1_epi8(15), _mm_srli_epi16(x, 4)));
    }

    inline __m256 updot(__m256i u, __m256i s) {
        __m256i res;
#if defined(__AVX512VNNI__) && defined(__AVX512VL__)
        res = _mm256_dpbusd_epi32(_mm256_setzero_si256(), u, s);
#elif defined(__AVXVNNI__)
        res = _mm256_dpbusd_avx_epi32(_mm256_setzero_si256(), u, s);
#else
        res = _mm256_madd_epi16(_mm256_set1_epi16(1), _mm256_maddubs_epi16(u, s));
#endif
        return _mm256_cvtepi32_ps(res);
    }

    static inline __m256i denibble(const uint8_t *p) {
        __m128i x = _mm_loadu_si128((const __m128i *)p);
        return _mm256_and_si256(_mm256_set1_epi8(15),
                                _mm256_insertf128_si256(_mm256_castsi128_si256(x),
                                                        _mm_srli_epi16(x, 4), 1));
    }

    static inline __m256i bittobyte(const uint8_t *p) {
        uint32_t x32;
        memcpy(&x32, p, sizeof(uint32_t));
        __m256i bytes = _mm256_cmpeq_epi8(_mm256_set1_epi64x(-1),
                                          _mm256_or_si256(_mm256_set1_epi64x(0x7fbfdfeff7fbfdfe),
                                                          _mm256_shuffle_epi8(_mm256_set1_epi32(x32),
                                                                              _mm256_set_epi64x(0x0303030303030303, 0x0202020202020202,
                                                                                                0x0101010101010101, 0x0000000000000000))));
        return _mm256_andnot_si256(bytes, _mm256_set1_epi8((char)0xF0));
    }

    const TA *const A;
    const TB *const B;
    TC *const C;
    const int64_t k;
    const int64_t lda;
    const int64_t ldb;
    const int64_t ldc;
    const int ith;
    const int nth;
    __m128i iq4nlt;
};
#endif // __AVX__

//PPC Implementation
#if defined(__MMA__)

//  go fuck it

#endif
} // namespace

/**
 * Performs optimized matrix multiplication on CPU.
 *
 * This subroutine may compute C = Aᵀ * B with column major ordering.
 * Despite its name, this isn't a generalized implementation. Work is
 * only performed when a handwritten kernel is written and available.
 * Otherwise the caller should fall back to a general matmul routine.
 *
 * For example, for single-threaded single-precision GEMM you can say
 *
 *     llamafile_sgemm(m, n, k, A, lda, B, ldb, C, ldc,
 *                     0, 1,
 *                     GGML_TYPE_F32, GGML_TYPE_F32, GGML_TYPE_F32);
 *
 * @param m is rows in `A` and `C`
 * @param n is cols in `B` and `C`
 * @param k is cols in `A` and rows in `B`
 * @param A is first input matrix (always transposed)
 * @param lda is row stride of `A`
 * @param B is second input matrix (never transposed)
 * @param ldb is row stride of `B`
 * @param C is input/output array of output matrices
 * @param ldc is row stride of `C`
 * @param ith is thread id (must be less than `nth`)
 * @param nth is number of threads (must be greater than zero)
 * @param Atype is GGML data type of `A`
 * @param Btype is GGML data type of `B`
 * @param Ctype is GGML data type of `C`
 * @return true if this function was able to service the matmul request
 */
bool llamafile_sgemm(const struct ggml_compute_params * params, int64_t m, int64_t n, int64_t k,
                     const void *A, int64_t lda, const void *B, int64_t ldb, void *C,
                     int64_t ldc, int Atype, int Btype, int Ctype) {

    assert(m >= 0);
    assert(n >= 0);
    assert(k >= 0);
    assert(lda >= k);
    assert(ldb >= k);
    assert(ldc >= m);
    assert(params->nth > 0);
    assert(params->ith < params->nth);

    // only enable sgemm for prompt processing
#if !defined(__MMA__)
    if (n < 2)
        return false;
#endif

    if (Ctype != GGML_TYPE_F32)
        return false;

    switch (Atype) {

    case GGML_TYPE_F32: {
        if (Btype != GGML_TYPE_F32)
            return false;
#if defined(__AVX512F__)
        tinyBLAS<16, __m512, __m512, float, float, float> tb{ params,
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc};
        return tb.matmul(m, n);
#elif defined(__AVX__) || defined(__AVX2__)
        tinyBLAS<8, __m256, __m256, float, float, float> tb{ params,
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc};
        return tb.matmul(m, n);
// #elif defined(__ARM_NEON)
//         if (n < 4)
//             return false;
//         tinyBLAS<4, float32x4_t, float32x4_t, float, float, float> tb{ params,
//             k, (const float *)A, lda,
//             (const float *)B, ldb,
//             (float *)C, ldc};
//         return tb.matmul(m, n);
#elif defined(__VXE__) || defined(__VXE2__)
        if (n < 4)
            return false;
        tinyBLAS<4, float32x4_t, float32x4_t, float, float, float> tb{ params,
            k, (const float *)A, lda,
            (const float *)B, ldb,
            (float *)C, ldc};
        return tb.matmul(m, n);
// #elif defined(__MMA__)
//         if (k % 8)
//             return false;
//         tinyBLAS_PPC<float, float, float> tb{
//             k, (const float *)A, lda,
//             (const float *)B, ldb,
//             (float *)C, ldc,
//             params->ith, params->nth};
//         tb.matmul(m, n);
//         return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_BF16: {
#if defined(__AVX512BF16__)
        if (Btype == GGML_TYPE_BF16) {
            tinyBLAS<32, __m512, __m512bh, ggml_bf16_t, ggml_bf16_t, float> tb{ params, k,
                (const ggml_bf16_t *)A, lda,
                (const ggml_bf16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
#elif defined(__AVX512F__)
        if (Btype == GGML_TYPE_BF16) {
            tinyBLAS<16, __m512, __m512, ggml_bf16_t, ggml_bf16_t, float> tb{ params, k,
                (const ggml_bf16_t *)A, lda,
                (const ggml_bf16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
#elif defined(__AVX2__)
        if (Btype == GGML_TYPE_BF16) {
            tinyBLAS<8, __m256, __m256, ggml_bf16_t, ggml_bf16_t, float> tb{ params, k,
                (const ggml_bf16_t *)A, lda,
                (const ggml_bf16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
// #elif defined(__MMA__)
//         if ((k % 8))
//                 return false;
//         if(Btype == GGML_TYPE_BF16) {
//            tinyBLAS_BF16_PPC<ggml_bf16_t, ggml_bf16_t, float> tb{ k,
//             (const ggml_bf16_t *)A, lda,
//             (const ggml_bf16_t *)B, ldb,
//             (float *)C, ldc,
//             params->ith, params->nth};
//         tb.matmul(m, n);
//         return true;
//         }
#endif
        return false;
    }

    case GGML_TYPE_F16: {
#if defined(__AVX512F__)
        if (Btype == GGML_TYPE_F16) {
            tinyBLAS<16, __m512, __m512, ggml_fp16_t, ggml_fp16_t, float> tb{ params, k,
                (const ggml_fp16_t *)A, lda,
                (const ggml_fp16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
#elif (defined(__AVX__) || defined(__AVX2__)) && defined(__F16C__)
        if (Btype == GGML_TYPE_F16) {
            tinyBLAS<8, __m256, __m256, ggml_fp16_t, ggml_fp16_t, float> tb{ params, k,
                (const ggml_fp16_t *)A, lda,
                (const ggml_fp16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
// #elif defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC) && !defined(_MSC_VER)
//         if (n < 8)
//             return false;
//         if (Btype == GGML_TYPE_F16) {
//             tinyBLAS<8, float16x8_t, float16x8_t, ggml_fp16_t, ggml_fp16_t, float> tb{ params,
//                 k, (const ggml_fp16_t *)A, lda,
//                 (const ggml_fp16_t *)B, ldb,
//                 (float *)C, ldc};
//             return tb.matmul(m, n);
//         }
// #elif defined(__ARM_NEON) && !defined(_MSC_VER)
//         if (Btype == GGML_TYPE_F32) {
//             tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, float, float> tb{ params,
//                 k, (const ggml_fp16_t *)A, lda,
//                 (const float *)B, ldb,
//                 (float *)C, ldc};
//             return tb.matmul(m, n);
//         }
#elif defined(__VXE__) || defined(__VXE2__)
        if (n < 4)
            return false;
        if (Btype == GGML_TYPE_F16) {
            tinyBLAS<4, float32x4_t, float32x4_t, ggml_fp16_t, ggml_fp16_t, float> tb{ params,
                k, (const ggml_fp16_t *)A, lda,
                (const ggml_fp16_t *)B, ldb,
                (float *)C, ldc};
            return tb.matmul(m, n);
        }
#endif
        return false;
    }

    case GGML_TYPE_Q8_0: {
        if (Btype != GGML_TYPE_Q8_0)
           return false;
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_q8_0, block_q8_0, float> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
// #elif defined(__ARM_FEATURE_DOTPROD)
//         tinyBLAS_Q0_ARM<block_q8_0> tb{
//             k, (const block_q8_0 *)A, lda,
//             (const block_q8_0 *)B, ldb,
//             (float *)C, ldc,
//             params->ith, params->nth};
//         tb.matmul(m, n);
//         return true;
#elif defined(__MMA__)
    //TO-DO: Remove this condition once gemv forwarding is enabled.
        if (n < 8 && n != 4)
           return false;
        if (m < 8 && m != 4)
           return false;
        tinyBLAS_Q0_PPC<block_q8_0, block_q8_0, float> tb{
            k, (const block_q8_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_Q4_0: {
        if (Btype != GGML_TYPE_Q8_0)
            return false;
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_q4_0, block_q8_0, float> tb{
            k, (const block_q4_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
// #elif defined(__ARM_FEATURE_DOTPROD)
//         tinyBLAS_Q0_ARM<block_q4_0> tb{
//             k, (const block_q4_0 *)A, lda,
//             (const block_q8_0 *)B, ldb,
//             (float *)C, ldc,
//             params->ith, params->nth};
//         tb.matmul(m, n);
//         return true;
// #elif defined(__MMA__)
//     //TO-DO: Remove this condition once gemv forwarding is enabled.
//         if (n < 8 && n != 4)
//            return false;
//         if (m < 8 && m != 4)
//            return false;
//         tinyBLAS_Q0_PPC<block_q4_0, block_q8_0, float> tb{
//             k, (const block_q4_0 *)A, lda,
//             (const block_q8_0 *)B, ldb,
//             (float *)C, ldc,
//             params->ith, params->nth};
//         tb.matmul(m, n);
//         return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_Q5_0: {
        if (Btype != GGML_TYPE_Q8_0)
            return false;
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_q5_0, block_q8_0, float> tb{
            k, (const block_q5_0 *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
#else
        return false;
#endif
    }

    case GGML_TYPE_IQ4_NL: {
        if (Btype != GGML_TYPE_Q8_0)
            return false;
#if defined(__AVX2__) || defined(__AVX512F__) || defined(__AVX__)
        tinyBLAS_Q0_AVX<block_iq4_nl, block_q8_0, float> tb{
            k, (const block_iq4_nl *)A, lda,
            (const block_q8_0 *)B, ldb,
            (float *)C, ldc,
            params->ith, params->nth};
        tb.matmul(m, n);
        return true;
#else
        return false;
#endif
    }

    default:
        return false;
    }

    (void)params;
    (void)m;
    (void)n;
    (void)k;
    (void)A;
    (void)lda;
    (void)B;
    (void)ldb;
    (void)C;
    (void)ldc;
    (void)Atype;
    (void)Btype;
    (void)Ctype;
}
