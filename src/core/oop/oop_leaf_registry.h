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
/* Per-GROUP (per-lane) twiddle s2 codelet: tw[(l2-1)*me + b] indexed by the
 * group var b (not a per-VW-block broadcast) -> arbitrary-K-correct + maskable. */
#define VFFT_OOP_T1_NAME(R)       VFFT_OOP_CAT5(radix,R,_t1_oop_fwd_,VFFT_OOP_ISA,_UG_UG)

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
#define VFFT_OOP_DECL_T1(R) \
  extern void VFFT_OOP_T1_NAME(R)( \
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
VFFT_OOP_DECL_T1(4)  VFFT_OOP_DECL_T1(7)  VFFT_OOP_DECL_T1(8)
VFFT_OOP_DECL_T1(13) VFFT_OOP_DECL_T1(16) VFFT_OOP_DECL_T1(32)
VFFT_OOP_DECL_T1(64)

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

/* Per-group (per-lane) twiddle s2 codelet — the arbitrary-K-correct BAILEY2
 * second stage (paired with a per-GROUP twiddle table: `count` entries/leg,
 * value W_N^{l2.k2} for group b where k2 = b/K). Used when K % GROUPW != 0;
 * for aligned K the per-block t1p above is the faster choice. */
static inline vfft_oop11_fn vfft_oop_t1_fn(int R)
{
    switch (R)
    {
#define C(R) case R: return VFFT_OOP_T1_NAME(R);
    C(4) C(7) C(8) C(13) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
}

/* ---- IL-boundary twins (P2, docs/roadmap/row_major_engine.md §11f) ----
 * Natively emitted (gen_radix --oop-il-in[-sw] / --oop-il-out[-sw]); same
 * 11-arg ABI with (z, unused) replacing the split pair on the folded side.
 * _sw = (im,re)-swapped lattice: the bwd swap identity folded into the
 * boundary (an interleaved z buffer cannot pointer-swap re/im).
 * avx2 only for now (avx512 masked IL lattice pending) — the resolvers
 * return 0 on other ISAs so callers degrade to the split path. */
#if VFFT_OOP_GROUPW == 4u
#define VFFT_OOP_DECL_IL(R) \
  extern void radix##R##_n1_oop_fwd_avx2_UG_UG_il_in( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
  extern void radix##R##_n1_oop_fwd_avx2_UG_UG_il_in_sw( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
  extern void radix##R##_t1_oop_fwd_avx2_UG_UG_il_out( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
  extern void radix##R##_t1_oop_fwd_avx2_UG_UG_il_out_sw( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
VFFT_OOP_DECL_IL(4) VFFT_OOP_DECL_IL(8) VFFT_OOP_DECL_IL(16)
VFFT_OOP_DECL_IL(32) VFFT_OOP_DECL_IL(64)
#endif

/* il_in leaf (reads interleaved z, writes split). sw=1 -> (im,re) read. */
static inline vfft_oop11_fn vfft_oop_leaf_il_fn(int R, int sw)
{
#if VFFT_OOP_GROUPW == 4u
    switch (R)
    {
#define C(R) case R: return sw ? radix##R##_n1_oop_fwd_avx2_UG_UG_il_in_sw \
                                : radix##R##_n1_oop_fwd_avx2_UG_UG_il_in;
    C(4) C(8) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
#else
    (void)R; (void)sw;
    return 0;
#endif
}

/* t1 with interleaved stores (reads split, writes z). sw=1 -> (im,re) write. */
static inline vfft_oop11_fn vfft_oop_t1_il_fn(int R, int sw)
{
#if VFFT_OOP_GROUPW == 4u
    switch (R)
    {
#define C(R) case R: return sw ? radix##R##_t1_oop_fwd_avx2_UG_UG_il_out_sw \
                                : radix##R##_t1_oop_fwd_avx2_UG_UG_il_out;
    C(4) C(8) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
#else
    (void)R; (void)sw;
    return 0;
#endif
}

/* ---- UnitLeg twins (two-pass restructure, row_major_engine.md §12.4) ----
 * t1 UL_UG: transpose fused into the t1's LOAD lattice (reads the column
 * pass's untransposed output at Ls=1/Gs=R1); n1 UG_UL: transpose fused into
 * the leaf's STORE lattice (writes the transposed intermediate at OLs=1/
 * OGs=R2). Either gives the MKL two-pass shape (no transpose sweep). avx2
 * only; me must be a multiple of 4 (UL configs carry no rem tail). */
#if VFFT_OOP_GROUPW == 4u
#define VFFT_OOP_DECL_UL(R) \
  extern void radix##R##_t1_oop_fwd_avx2_UL_UG( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
  extern void radix##R##_n1_oop_fwd_avx2_UG_UL( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
VFFT_OOP_DECL_UL(4) VFFT_OOP_DECL_UL(8) VFFT_OOP_DECL_UL(16)
VFFT_OOP_DECL_UL(32) VFFT_OOP_DECL_UL(64)
#endif

static inline vfft_oop11_fn vfft_oop_t1_ul_fn(int R)
{
#if VFFT_OOP_GROUPW == 4u
    switch (R)
    {
#define C(R) case R: return radix##R##_t1_oop_fwd_avx2_UL_UG;
    C(4) C(8) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
#else
    (void)R;
    return 0;
#endif
}

static inline vfft_oop11_fn vfft_oop_leaf_ugul_fn(int R)
{
#if VFFT_OOP_GROUPW == 4u
    switch (R)
    {
#define C(R) case R: return radix##R##_n1_oop_fwd_avx2_UG_UL;
    C(4) C(8) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
#else
    (void)R;
    return 0;
#endif
}

/* t1 UL twin with LINEAR twiddle layout (§12.4 4a): table packed in
 * consumption order (per group-quad, all legs' 4-vectors contiguous) —
 * one streaming cursor. Needs the linear-filled table (Qlr/Qli), NOT the
 * flat Qr/Qi. */
#if VFFT_OOP_GROUPW == 4u
#define VFFT_OOP_DECL_TWL(R) \
  extern void radix##R##_t1_oop_fwd_avx2_UL_UG_twl( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
VFFT_OOP_DECL_TWL(4) VFFT_OOP_DECL_TWL(8) VFFT_OOP_DECL_TWL(16)
VFFT_OOP_DECL_TWL(32) VFFT_OOP_DECL_TWL(64)
#endif

static inline vfft_oop11_fn vfft_oop_t1_ul_twl_fn(int R)
{
#if VFFT_OOP_GROUPW == 4u
    switch (R)
    {
#define C(R) case R: return radix##R##_t1_oop_fwd_avx2_UL_UG_twl;
    C(4) C(8) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
#else
    (void)R;
    return 0;
#endif
}

/* K1 MONO (§12.4 item 3): the whole K=1 four-step in one emitted function,
 * emit-time rodata twiddles, natural order, split, fwd. Uniform 11-arg ABI
 * (tw/strides/me ignored — call with 0s). M1 coverage: N=64. */
#if VFFT_OOP_GROUPW == 4u
extern void vfft_k1_mono64_fwd_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t, size_t, size_t, size_t, size_t);
extern void vfft_k1_mono128_16x8_fwd_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t, size_t, size_t, size_t, size_t);
extern void vfft_k1_mono128_8x16_fwd_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t, size_t, size_t, size_t, size_t);
extern void vfft_k1_mono256_16x16_fwd_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#endif

static inline vfft_oop11_fn vfft_k1_mono_fn(int N)
{
#if VFFT_OOP_GROUPW == 4u
    switch (N)
    {
    case 64:  return vfft_k1_mono64_fwd_avx2;
    case 128: return vfft_k1_mono128_16x8_fwd_avx2;
    case 256: return vfft_k1_mono256_16x16_fwd_avx2;
    default: return 0;
    }
#else
    (void)N;
    return 0;
#endif
}

/* pair-aware mono resolver: the wisdom's (R1) picks the emitted variant.
 * R1 == 0 -> the default for N. */
static inline vfft_oop11_fn vfft_k1_mono_pair_fn(int N, int R1);

/* alternate pair (the calibrator's other candidate; today only 128 has one) */
static inline vfft_oop11_fn vfft_k1_mono_alt_fn(int N)
{
#if VFFT_OOP_GROUPW == 4u
    switch (N)
    {
    case 128: return vfft_k1_mono128_8x16_fwd_avx2;
    default: return 0;
    }
#else
    (void)N;
    return 0;
#endif
}

/* t1 LOG3 twins (leg-axis twiddle derivation, FFTW-t3 class): load only the
 * base legs' twiddle VECTORS and derive the rest by vector cmuls — 252->24
 * tw loads at radix-64. SAME Qr/Qi table layout (sparse subset of the same
 * slots), so these are drop-in fn-pointer swaps on the existing plans. Values
 * differ from flat in last-ulp (derived vs loaded) — gate vs naive, not
 * bit-cross. */
#if VFFT_OOP_GROUPW == 4u
#define VFFT_OOP_DECL_T1L3(R) \
  extern void radix##R##_t1_oop_fwd_avx2_UG_UG_log3( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
  extern void radix##R##_t1_oop_fwd_avx2_UL_UG_log3( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
VFFT_OOP_DECL_T1L3(4) VFFT_OOP_DECL_T1L3(8) VFFT_OOP_DECL_T1L3(16)
VFFT_OOP_DECL_T1L3(32) VFFT_OOP_DECL_T1L3(64)
#endif

static inline vfft_oop11_fn vfft_oop_t1_l3_fn(int R)
{
#if VFFT_OOP_GROUPW == 4u
    switch (R)
    {
#define C(R) case R: return radix##R##_t1_oop_fwd_avx2_UG_UG_log3;
    C(4) C(8) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
#else
    (void)R;
    return 0;
#endif
}

static inline vfft_oop11_fn vfft_oop_t1_ul_l3_fn(int R)
{
#if VFFT_OOP_GROUPW == 4u
    switch (R)
    {
#define C(R) case R: return radix##R##_t1_oop_fwd_avx2_UL_UG_log3;
    C(4) C(8) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
#else
    (void)R;
    return 0;
#endif
}

#if VFFT_OOP_GROUPW == 4u
#define VFFT_OOP_DECL_IL_LOG3(R) \
  extern void radix##R##_t1_oop_fwd_avx2_UG_UG_log3_il_out( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
  extern void radix##R##_t1_oop_fwd_avx2_UG_UG_log3_il_out_sw( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
  extern void radix##R##_t1_oop_fwd_avx2_UL_UG_log3_il_out( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
  extern void radix##R##_t1_oop_fwd_avx2_UL_UG_log3_il_out_sw( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
VFFT_OOP_DECL_IL_LOG3(4)  VFFT_OOP_DECL_IL_LOG3(8)  VFFT_OOP_DECL_IL_LOG3(16)
VFFT_OOP_DECL_IL_LOG3(32) VFFT_OOP_DECL_IL_LOG3(64)
#endif

/* ---- IL log3 twins ----------------------------------------------------
 * The interleaved counterparts of vfft_oop_t1_l3_fn / vfft_oop_t1_ul_l3_fn.
 * log3 (TP_Log3) derives the external twiddles by binary composition instead
 * of streaming a flat table; on the SPLIT axis it is two of the eight shipped
 * K=1 routes (VFFT_K1_SP_2PA_L3 / _3P_L3), resolved as a create-time pointer
 * swap. IL never had them emitted, so the IL route set had no twiddle-strategy
 * variants at all — this closes that gap.
 *
 * NOTE the arity difference from the split resolvers, which take only (R):
 * split's backward is the pointer-swap identity, while IL backward needs the
 * _sw lattice twin. That is why IL costs 4 kernels per radix here where split
 * costs 2 — a structural property of the IL axis, not an oversight.
 *
 * Whether log3 actually WINS on IL kernels is unmeasured: it was calibrated on
 * split codelets, and IL has a different port profile (BYTW2 spends a cflip
 * per complex multiply). These exist so dp_planner_il.h can RACE them per
 * cell rather than anyone assuming the split verdict transfers. */
static inline vfft_oop11_fn vfft_oop_t1_il_l3_fn(int R, int sw)
{
#if VFFT_OOP_GROUPW == 4u
    switch (R)
    {
#define C(R) case R: return sw ? radix##R##_t1_oop_fwd_avx2_UG_UG_log3_il_out_sw \
                                : radix##R##_t1_oop_fwd_avx2_UG_UG_log3_il_out;
    C(4) C(8) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
#else
    (void)R; (void)sw;
    return 0;
#endif
}

static inline vfft_oop11_fn vfft_oop_t1_ul_il_l3_fn(int R, int sw)
{
#if VFFT_OOP_GROUPW == 4u
    switch (R)
    {
#define C(R) case R: return sw ? radix##R##_t1_oop_fwd_avx2_UL_UG_log3_il_out_sw \
                                : radix##R##_t1_oop_fwd_avx2_UL_UG_log3_il_out;
    C(4) C(8) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
#else
    (void)R; (void)sw;
    return 0;
#endif
}

/* mono-64 IL twins (M2/M4): z->z; bwd = fwd DAG with (im,re)-swapped
 * boundary lattices (swap identity), unnormalized inverse, output (re,im).
 * ABI: (in_z, unused, out_z, unused, ...). Split bwd needs NO codelet —
 * call the split fwd with re/im pointer pairs swapped. */
#if VFFT_OOP_GROUPW == 4u
extern void vfft_k1_mono64_8x8_il_fwd_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t, size_t, size_t, size_t, size_t);
extern void vfft_k1_mono64_8x8_il_bwd_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#endif

static inline vfft_oop11_fn vfft_k1_mono_pair_fn(int N, int R1)
{
#if VFFT_OOP_GROUPW == 4u
    if (N == 128 && R1 == 8) return vfft_k1_mono_alt_fn(128);
#endif
    (void)R1;
    return vfft_k1_mono_fn(N);
}

/* t1 UL-load + il_out store twins: the TRUE 2-pass IL exit (reads the
 * untransposed split column output at Ls=1/Gs=R1, transposes in the load
 * lattice, interleaves in the store lattice). _sw = bwd swap. */
#if VFFT_OOP_GROUPW == 4u
#define VFFT_OOP_DECL_UL_ILOUT(R) \
  extern void radix##R##_t1_oop_fwd_avx2_UL_UG_il_out( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
  extern void radix##R##_t1_oop_fwd_avx2_UL_UG_il_out_sw( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
VFFT_OOP_DECL_UL_ILOUT(4) VFFT_OOP_DECL_UL_ILOUT(8) VFFT_OOP_DECL_UL_ILOUT(16)
VFFT_OOP_DECL_UL_ILOUT(32) VFFT_OOP_DECL_UL_ILOUT(64)
#endif

static inline vfft_oop11_fn vfft_oop_t1_ul_il_fn(int R, int sw)
{
#if VFFT_OOP_GROUPW == 4u
    switch (R)
    {
#define C(R) case R: return sw ? radix##R##_t1_oop_fwd_avx2_UL_UG_il_out_sw \
                                : radix##R##_t1_oop_fwd_avx2_UL_UG_il_out;
    C(4) C(8) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
#else
    (void)R; (void)sw;
    return 0;
#endif
}

static inline vfft_oop11_fn vfft_k1_mono_il_fn(int N, int bwd)
{
#if VFFT_OOP_GROUPW == 4u
    switch (N)
    {
    case 64: return bwd ? vfft_k1_mono64_8x8_il_bwd_avx2
                        : vfft_k1_mono64_8x8_il_fwd_avx2;
    default: return 0;
    }
#else
    (void)N; (void)bwd;
    return 0;
#endif
}

#endif /* VFFT_OOP_LEAF_REGISTRY_H */
