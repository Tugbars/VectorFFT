/* oop_leaf_registry.h — hand-written registry for the OOP codelet families
 * (11-arg generic ABI), per docs section 16 item 4. Generator-emitted
 * version can replace this later.
 *
 * ABI: fn(src_re, src_im, dst_re, dst_im, W_re, W_im, L, G, OL, OG, count)
 *   - positions advance in vector groups of 8; group base = b*G; element j
 *     adds j*L; twiddle rows indexed tw[(l2-1)*(count/8) + b/8].
 *   - n1 leaves take W_re = W_im = NULL.
 *
 * Coverage (avx512): n1_oop leaves {2..17,19,20,25,32,64,128};
 * t1p_log3 twiddle {4,7,8,13,16,32,64}.
 */
#ifndef VFFT_OOP_LEAF_REGISTRY_H
#define VFFT_OOP_LEAF_REGISTRY_H

#include <stddef.h>

/* ISA selection, per-ISA-binary model (same as the proto executor).
 * avx512 builds bind the avx512 codelet symbols; otherwise avx2.
 * Override with -DVFFT_OOP_FORCE_AVX2 to force the avx2 set in an
 * avx512-capable build. GROUPW is the codelet position-group width and
 * governs the twiddle-table replication granularity (rows = count/GROUPW,
 * verified in the generated sources: avx512 b+=8 tw[r*(me/8)+b/8];
 * avx2 b+=4 tw[r*(me/4)+b/4]). */
#if defined(__AVX512F__) && !defined(VFFT_OOP_FORCE_AVX2)
#define VFFT_OOP_ISA avx512
#define VFFT_OOP_GROUPW 8u
#else
#define VFFT_OOP_ISA avx2
#define VFFT_OOP_GROUPW 4u
#endif
#define VFFT_OOP_CAT5_(a,b,c,d,e) a##b##c##d##e
#define VFFT_OOP_CAT5(a,b,c,d,e) VFFT_OOP_CAT5_(a,b,c,d,e)
#define VFFT_OOP_N1_NAME(R)       VFFT_OOP_CAT5(radix,R,_n1_oop_fwd_,VFFT_OOP_ISA,_UG_UG)
#define VFFT_OOP_T1P_NAME(R)      VFFT_OOP_CAT5(radix,R,_t1p_oop_fwd_,VFFT_OOP_ISA,_UG_UG_log3)
#define VFFT_OOP_T1P_FLAT_NAME(R) VFFT_OOP_CAT5(radix,R,_t1p_oop_fwd_,VFFT_OOP_ISA,_UG_UG)

typedef void (*vfft_oop11_fn)(const double *, const double *,
                              double *, double *,
                              const double *, const double *,
                              size_t, size_t, size_t, size_t, size_t);

#define VFFT_OOP_DECL_N1(R) \
  extern void VFFT_OOP_N1_NAME(R)( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#define VFFT_OOP_DECL_T1P(R) \
  extern void VFFT_OOP_T1P_NAME(R)( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#define VFFT_OOP_DECL_T1P_FLAT(R) \
  extern void VFFT_OOP_T1P_FLAT_NAME(R)( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);

VFFT_OOP_DECL_N1(2)  VFFT_OOP_DECL_N1(3)  VFFT_OOP_DECL_N1(4)
VFFT_OOP_DECL_N1(5)  VFFT_OOP_DECL_N1(6)  VFFT_OOP_DECL_N1(7)
VFFT_OOP_DECL_N1(8)  VFFT_OOP_DECL_N1(9)  VFFT_OOP_DECL_N1(10)
VFFT_OOP_DECL_N1(11) VFFT_OOP_DECL_N1(12) VFFT_OOP_DECL_N1(13)
VFFT_OOP_DECL_N1(14) VFFT_OOP_DECL_N1(15) VFFT_OOP_DECL_N1(16)
VFFT_OOP_DECL_N1(17) VFFT_OOP_DECL_N1(19) VFFT_OOP_DECL_N1(20)
VFFT_OOP_DECL_N1(25) VFFT_OOP_DECL_N1(32) VFFT_OOP_DECL_N1(64)
VFFT_OOP_DECL_N1(128)
VFFT_OOP_DECL_T1P(4)  VFFT_OOP_DECL_T1P(7)  VFFT_OOP_DECL_T1P(8)
VFFT_OOP_DECL_T1P(13) VFFT_OOP_DECL_T1P(16) VFFT_OOP_DECL_T1P(32)
VFFT_OOP_DECL_T1P(64)
VFFT_OOP_DECL_T1P_FLAT(4)  VFFT_OOP_DECL_T1P_FLAT(7)  VFFT_OOP_DECL_T1P_FLAT(8)
VFFT_OOP_DECL_T1P_FLAT(13) VFFT_OOP_DECL_T1P_FLAT(16) VFFT_OOP_DECL_T1P_FLAT(32)
VFFT_OOP_DECL_T1P_FLAT(64)

static inline vfft_oop11_fn vfft_oop_leaf_fn(int R)
{
    switch (R)
    {
#define C(R) case R: return VFFT_OOP_N1_NAME(R);
    C(2) C(3) C(4) C(5) C(6) C(7) C(8) C(9) C(10) C(11) C(12) C(13)
    C(14) C(15) C(16) C(17) C(19) C(20) C(25) C(32) C(64) C(128)
#undef C
    default: return 0;
    }
}

static inline vfft_oop11_fn vfft_oop_t1p_fn(int R)
{
    switch (R)
    {
#define C(R) case R: return VFFT_OOP_T1P_NAME(R);
    C(4) C(7) C(8) C(13) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
}

/* Flat (non-log3) t1p — the FMA-leaner twiddle stage. log3 is a port rebalance
 * that wins only when the stage is load-bound with FMA slack; on FMA-bound
 * stages flat is faster. The BAILEY2 tuner measures both and picks per cell. */
static inline vfft_oop11_fn vfft_oop_t1p_flat_fn(int R)
{
    switch (R)
    {
#define C(R) case R: return VFFT_OOP_T1P_FLAT_NAME(R);
    C(4) C(7) C(8) C(13) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
}

/* Unified getter: variant 0 = flat, 1 = log3. */
static inline vfft_oop11_fn vfft_oop_t1p_fn_v(int R, int variant)
{
    return variant ? vfft_oop_t1p_fn(R) : vfft_oop_t1p_flat_fn(R);
}

#endif /* VFFT_OOP_LEAF_REGISTRY_H */
