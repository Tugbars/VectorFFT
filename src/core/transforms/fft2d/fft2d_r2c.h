/**
 * fft2d_r2c.h -- 2D Real-to-Complex / Complex-to-Real FFT
 *
 *   Forward (R2C 2D):  N1*N2 reals -> N1 * (N2/2+1) complex
 *     Phase 1: tiled R2C row pass — for each tile of B rows, transpose
 *              real input B*N2 -> N2*B, run 1D R2C (N=N2, K=B), transpose
 *              split-complex output (N2/2+1)*B -> B*(N2/2+1).
 *     Phase 2: 1D C2C col pass — N1-point complex FFT batched K=(N2/2+1).
 *              No transpose needed; layout already matches.
 *
 *   Backward (C2R 2D):  N1 * (N2/2+1) complex -> N1*N2 reals
 *     Phase 1: 1D C2C col IFFT — N1-point batched K=(N2/2+1).
 *     Phase 2: tiled C2R row pass, processed in REVERSE tile order
 *              (scatter writes longer rows than gather reads, so reverse
 *              avoids overwriting future tiles' input).
 *
 * Layout (split-batched):
 *   real input:   real[i * N2 + j]            for i=0..N1-1, j=0..N2-1
 *   complex out:  re[i*(N2/2+1) + f], im same  for i=0..N1-1, f=0..N2/2
 *
 * In-place semantics:
 *   forward: caller passes (re, im); re sized to hold max(N1*N2,
 *            N1*(N2/2+1)) = N1*N2 reals on input, becomes N1*(N2/2+1)
 *            Re bins on output. im sized N1*(N2/2+1).
 *   backward: same buffers, layout transitions back.
 *
 * Constraint: N2 must be even (inherits 1D R2C even-N constraint).
 *
 * Threading:
 *   Phase 1: tile-parallel — same model as 2D C2C tiled row pass.
 *   Phase 2: K-split via the C2C executor — K=(N2/2+1) usually large enough.
 *
 * Reuses transpose.h (8x4 line-filling kernel) and r2c.h.
 */
#ifndef STRIDE_FFT2D_R2C_H
#define STRIDE_FFT2D_R2C_H

#include <time.h> /* clock_gettime for the adoption A/B timing (win: mingw provides it) */
#include "executor.h"
#include "planner.h"
#include "threads.h"
#include "proto_stride_compat.h"
#include "transpose.h"
#include "r2c.h"
#include "rfft.h"                 /* §6a31: rfft-engine row inner (low-K winner) */
#include "c2r.h"                  /* §6a32: c2r-engine bwd row inner */
#ifdef VFFT_USE_JIT
#include "jit_runtime.h"          /* JIT/baked resolve for the inner column c2c FFT */
#endif

#ifdef VFFT_2D_PROFILE
static double _f2d_wrapin, _f2d_p1_tin, _f2d_p1_r2c, _f2d_p1_tout, _f2d_p2, _f2d_p3, _f2d_wrapout;
static double _f2d_now(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
#define _F2D_T0(v) double _t_##v = _f2d_now()
#define _F2D_T1(v) v += _f2d_now() - _t_##v
#else
#define _F2D_T0(v)
#define _F2D_T1(v)
#endif

#ifndef FFT2D_R2C_DEFAULT_TILE
#define FFT2D_R2C_DEFAULT_TILE 8
#endif

#ifndef FFT2D_R2C_MIN_TILE
#define FFT2D_R2C_MIN_TILE 4
#endif

#ifndef FFT2D_R2C_MAX_THREADS
#define FFT2D_R2C_MAX_THREADS STRIDE_POOL_MAX_DISPATCH /* the pool's bound, not a second one */
#endif


/* ═══════════════════════════════════════════════════════════════
 * 2D R2C PLAN DATA
 * ═══════════════════════════════════════════════════════════════ */

#include "strided_tw.h"
#include "../../planning/adopt_wisdom.h"  /* §6a49/Q3 */

/* §6a39: strided r2c/c2r row engines (the v2 family, --strided-r2c
 * emission, gated bit/roundtrip per size in §6a37/38). One sweep replaces
 * the ENTIRE tiled row pass: fwd reads the user real plane row-major and
 * writes re_pad/im_pad directly (out_stride = K_pad); bwd reads the pads
 * after the col IFFT and writes real rows. me = PAIRS; coverage N2 in
 * {8,12,16,20,32,64}, N1 % 8 == 0, ST only. Adoption is MEASURED at plan
 * create with the §6a34 >5% hysteresis. */
typedef void (*_f2d_sr2c_fwd_fn)(const double *, double *, double *,
                                 const double *, const double *,
                                 size_t, size_t, size_t);
typedef void (*_f2d_sr2c_bwd_fn)(const double *, const double *, double *,
                                 const double *, const double *,
                                 size_t, size_t, size_t);
#define _F2D_SR2C_DECL(N) \
    void radix##N##_n1_fwd_avx2_strided_r2c(const double *, double *, double *, \
        const double *, const double *, size_t, size_t, size_t); \
    void radix##N##_n1_bwd_avx2_strided_r2c(const double *, const double *, \
        double *, const double *, const double *, size_t, size_t, size_t);
_F2D_SR2C_DECL(8) _F2D_SR2C_DECL(12) _F2D_SR2C_DECL(16)
_F2D_SR2C_DECL(20) _F2D_SR2C_DECL(32) _F2D_SR2C_DECL(64)
_F2D_SR2C_DECL(128) _F2D_SR2C_DECL(256) _F2D_SR2C_DECL(512)
#undef _F2D_SR2C_DECL
#if defined(__AVX512F__) && defined(__AVX512DQ__)
/* §6a45: avx512 editions (build-target selected, the strided_rows.h
 * convention). N=12/20 have no width-8 edition (radix %% 8) and fall back
 * to avx2. AVX512-vs-AVX2 measured BIT-identical values; r256 fwd −9.0%
 * on this host. */
#define _F2D_SR2C_D512(N) \
    void radix##N##_n1_fwd_avx512_strided_r2c(const double *, double *, \
        double *, const double *, const double *, size_t, size_t, size_t); \
    void radix##N##_n1_bwd_avx512_strided_r2c(const double *, \
        const double *, double *, const double *, const double *, size_t, \
        size_t, size_t);
_F2D_SR2C_D512(8) _F2D_SR2C_D512(16) _F2D_SR2C_D512(32) _F2D_SR2C_D512(64)
_F2D_SR2C_D512(128) _F2D_SR2C_D512(256) _F2D_SR2C_D512(512)
#undef _F2D_SR2C_D512
#endif
/* §6a48/Q2: resolvers return the edition AND its block quantum (pairs per
 * codelet block). Tail staging (below) absorbs rows %% (2*blk) != 0 and odd
 * row counts, so the Q0 pairs constraint is retired — avx512 is preferred
 * whenever built. */
static inline _f2d_sr2c_fwd_fn _f2d_sr2c_fwd_resolve(int N2, int *blk) {
    *blk = 4;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    *blk = 8;
    switch (N2) {
    case 8:   return radix8_n1_fwd_avx512_strided_r2c;
    case 16:  return radix16_n1_fwd_avx512_strided_r2c;
    case 32:  return radix32_n1_fwd_avx512_strided_r2c;
    case 64:  return radix64_n1_fwd_avx512_strided_r2c;
    case 128: return radix128_n1_fwd_avx512_strided_r2c;
    case 256: return radix256_n1_fwd_avx512_strided_r2c;
    case 512: return radix512_n1_fwd_avx512_strided_r2c;
    default: break;
    }
    *blk = 4;
#endif
    switch (N2) {
    case 8:  return radix8_n1_fwd_avx2_strided_r2c;
    case 12: return radix12_n1_fwd_avx2_strided_r2c;
    case 16: return radix16_n1_fwd_avx2_strided_r2c;
    case 20: return radix20_n1_fwd_avx2_strided_r2c;
    case 32: return radix32_n1_fwd_avx2_strided_r2c;
    case 64: return radix64_n1_fwd_avx2_strided_r2c;
    case 128: return radix128_n1_fwd_avx2_strided_r2c;  /* §6a42 fused tw */
    case 256: return radix256_n1_fwd_avx2_strided_r2c;
    case 512: return radix512_n1_fwd_avx2_strided_r2c;
    default: return 0;
    }
}
/* §6a44: MT range-split for the strided mono tier. Chunks are masked to
 * 4-pair (8-row) multiples, so every thread executes exactly the blocks ST
 * would — MT output is BIT-IDENTICAL to ST by construction. The codelets
 * are scratch-free, so unlike the tiled path there are no per-thread slots
 * to ration. The stw tier stays ST (shared work buffer; dormant anyway). */
typedef struct {
    _f2d_sr2c_fwd_fn fn;
    const double *rio; double *ore, *oim;
    size_t rs_in, os, me;
} _f2d_sr2c_mtf_arg_t;
typedef struct {
    _f2d_sr2c_bwd_fn fn;
    const double *ire, *iim; double *out;
    size_t is, rs_in, me;
} _f2d_sr2c_mtb_arg_t;
static void _f2d_sr2c_mtf_tramp(void *vp) {
    _f2d_sr2c_mtf_arg_t *a = (_f2d_sr2c_mtf_arg_t *)vp;
    a->fn(a->rio, a->ore, a->oim, 0, 0, a->rs_in, a->os, a->me);
}
static void _f2d_sr2c_mtb_tramp(void *vp) {
    _f2d_sr2c_mtb_arg_t *a = (_f2d_sr2c_mtb_arg_t *)vp;
    a->fn(a->ire, a->iim, a->out, 0, 0, a->is, a->rs_in, a->me);
}
static void _f2d_sr2c_fwd_run(_f2d_sr2c_fwd_fn fn, const double *rio,
                              double *ore, double *oim,
                              size_t rs_in, size_t os, size_t me)
{
    int T = stride_pool_workers_for(0); /* the pool's one clamp; no plan handle here */
    if (T <= 1 || me < 16) { fn(rio, ore, oim, 0, 0, rs_in, os, me); return; }
    /* 8-aligned proportional ranges, empty ones skipped: the slots are PACKED,
     * and the caller (slot 0) runs whichever non-empty range comes first --
     * its own [0,p0_main_end) when that is non-empty, exactly as before. */
    _f2d_sr2c_mtf_arg_t args[STRIDE_POOL_MAX_DISPATCH];
    int m = 0;
    size_t p0_main_end = ((me * 1) / (size_t)T) & ~(size_t)7;
    if (p0_main_end > 0) {
        args[m].fn = fn; args[m].rio = rio; args[m].ore = ore; args[m].oim = oim;
        args[m].rs_in = rs_in; args[m].os = os; args[m].me = p0_main_end;
        m++;
    }
    for (int t = 1; t < T; t++) {
        size_t ps = ((me * (size_t)t) / (size_t)T) & ~(size_t)7;
        size_t pe = ((me * (size_t)(t + 1)) / (size_t)T) & ~(size_t)7;
        if (t == T - 1) pe = me;
        if (ps >= pe) continue;
        args[m].fn = fn;
        args[m].rio = rio + 2 * ps * rs_in;
        args[m].ore = ore + 2 * ps * os;
        args[m].oim = oim + 2 * ps * os;
        args[m].rs_in = rs_in; args[m].os = os; args[m].me = pe - ps;
        m++;
    }
    if (m > 0) stride_pool_run(m, _f2d_sr2c_mtf_tramp, args, sizeof args[0]);
}
static void _f2d_sr2c_bwd_run(_f2d_sr2c_bwd_fn fn, const double *ire,
                              const double *iim, double *out,
                              size_t is, size_t rs_in, size_t me)
{
    int T = stride_pool_workers_for(0); /* the pool's one clamp; no plan handle here */
    if (T <= 1 || me < 16) { fn(ire, iim, out, 0, 0, is, rs_in, me); return; }
    /* packed slots, caller = slot 0 (see the forward twin) */
    _f2d_sr2c_mtb_arg_t args[STRIDE_POOL_MAX_DISPATCH];
    int m = 0;
    size_t p0_main_end = ((me * 1) / (size_t)T) & ~(size_t)7;
    if (p0_main_end > 0) {
        args[m].fn = fn; args[m].ire = ire; args[m].iim = iim; args[m].out = out;
        args[m].is = is; args[m].rs_in = rs_in; args[m].me = p0_main_end;
        m++;
    }
    for (int t = 1; t < T; t++) {
        size_t ps = ((me * (size_t)t) / (size_t)T) & ~(size_t)7;
        size_t pe = ((me * (size_t)(t + 1)) / (size_t)T) & ~(size_t)7;
        if (t == T - 1) pe = me;
        if (ps >= pe) continue;
        args[m].fn = fn;
        args[m].ire = ire + 2 * ps * is;
        args[m].iim = iim + 2 * ps * is;
        args[m].out = out + 2 * ps * rs_in;
        args[m].is = is; args[m].rs_in = rs_in; args[m].me = pe - ps;
        m++;
    }
    if (m > 0) stride_pool_run(m, _f2d_sr2c_mtb_tramp, args, sizeof args[0]);
}

/* §6a48/Q2: rows-based, tail-capable entries. Full blocks go through the
 * MT _run path; the remainder (rows %% (2*blk), incl. an odd lone row) is
 * staged through a zeroed block — the lone row's zero partner makes X2 a
 * zero spectrum (discarded fwd / ignored bwd) by the two-for-one algebra.
 * tscr layout: [2*blk*N in] [2*blk*hp1 re] [2*blk*hp1 im]. NULL tscr =>
 * full-block-only (legacy callers). */
static inline void _f2d_sr2c_fwd_rows(_f2d_sr2c_fwd_fn fn, int blk, int N,
                                      const double *x, double *ore,
                                      double *oim, size_t rs_in, size_t os,
                                      size_t rows, double *tscr)
{
    size_t pairs = rows / 2;
    int odd = (int)(rows & 1);
    size_t main_p = pairs & ~(size_t)(blk - 1);
    if (main_p)
        _f2d_sr2c_fwd_run(fn, x, ore, oim, rs_in, os, main_p);
    size_t rem = pairs - main_p;
    if ((rem || odd) && tscr) {
        const size_t hp1 = (size_t)N / 2 + 1;
        double *si = tscr;
        double *sr = si + 2 * (size_t)blk * (size_t)N;
        double *sm = sr + 2 * (size_t)blk * hp1;
        memset(si, 0, 2 * (size_t)blk * (size_t)N * sizeof(double));
        size_t trows = 2 * rem + (size_t)odd;
        for (size_t r = 0; r < trows; r++)
            memcpy(si + r * (size_t)N, x + (2 * main_p + r) * rs_in,
                   (size_t)N * sizeof(double));
        fn(si, sr, sm, 0, 0, (size_t)N, hp1, (size_t)blk);
        for (size_t r = 0; r < trows; r++) {
            memcpy(ore + (2 * main_p + r) * os, sr + r * hp1,
                   hp1 * sizeof(double));
            memcpy(oim + (2 * main_p + r) * os, sm + r * hp1,
                   hp1 * sizeof(double));
        }
    }
}
static inline void _f2d_sr2c_bwd_rows(_f2d_sr2c_bwd_fn fn, int blk, int N,
                                      const double *ire, const double *iim,
                                      double *out, size_t is, size_t rs_out,
                                      size_t rows, double *tscr)
{
    size_t pairs = rows / 2;
    int odd = (int)(rows & 1);
    size_t main_p = pairs & ~(size_t)(blk - 1);
    if (main_p)
        _f2d_sr2c_bwd_run(fn, ire, iim, out, is, rs_out, main_p);
    size_t rem = pairs - main_p;
    if ((rem || odd) && tscr) {
        const size_t hp1 = (size_t)N / 2 + 1;
        double *si = tscr;                       /* real out staging */
        double *sr = si + 2 * (size_t)blk * (size_t)N;
        double *sm = sr + 2 * (size_t)blk * hp1;
        memset(sr, 0, 2 * 2 * (size_t)blk * hp1 * sizeof(double));
        size_t trows = 2 * rem + (size_t)odd;
        for (size_t r = 0; r < trows; r++) {
            memcpy(sr + r * hp1, ire + (2 * main_p + r) * is,
                   hp1 * sizeof(double));
            memcpy(sm + r * hp1, iim + (2 * main_p + r) * is,
                   hp1 * sizeof(double));
        }
        fn(sr, sm, si, 0, 0, hp1, (size_t)N, (size_t)blk);
        for (size_t r = 0; r < trows; r++)
            memcpy(out + (2 * main_p + r) * rs_out, si + r * (size_t)N,
                   (size_t)N * sizeof(double));
    }
}

static inline _f2d_sr2c_bwd_fn _f2d_sr2c_bwd_resolve(int N2, int *blk) {
    *blk = 4;
#if defined(__AVX512F__) && defined(__AVX512DQ__)
    *blk = 8;
    switch (N2) {
    case 8:   return radix8_n1_bwd_avx512_strided_r2c;
    case 16:  return radix16_n1_bwd_avx512_strided_r2c;
    case 32:  return radix32_n1_bwd_avx512_strided_r2c;
    case 64:  return radix64_n1_bwd_avx512_strided_r2c;
    case 128: return radix128_n1_bwd_avx512_strided_r2c;
    case 256: return radix256_n1_bwd_avx512_strided_r2c;
    case 512: return radix512_n1_bwd_avx512_strided_r2c;
    default: break;
    }
    *blk = 4;
#endif
    switch (N2) {
    case 8:  return radix8_n1_bwd_avx2_strided_r2c;
    case 12: return radix12_n1_bwd_avx2_strided_r2c;
    case 16: return radix16_n1_bwd_avx2_strided_r2c;
    case 20: return radix20_n1_bwd_avx2_strided_r2c;
    case 32: return radix32_n1_bwd_avx2_strided_r2c;
    case 64: return radix64_n1_bwd_avx2_strided_r2c;
    case 128: return radix128_n1_bwd_avx2_strided_r2c;  /* §6a42 fused tw */
    case 256: return radix256_n1_bwd_avx2_strided_r2c;
    case 512: return radix512_n1_bwd_avx2_strided_r2c;
    default: return 0;
    }
}

typedef struct {
    int N1;                       /* rows */
    int N2;                       /* cols (must be even) */
    size_t B;                     /* row tile height */
    size_t K_pad;                 /* col FFT batch dim, padded to multiple of 4
                                   * (codelet n1_fwd has no scalar tail at vl<4) */

    /* §6a31: optional rfft-engine row inner — measured 27% faster than the
     * stride inner at the tile shape ((256,8): 2.885 vs 3.969 µs/call).
     * Injected by the vfft layer (registry + lifetime owner); used only when
     * the row pass runs single-threaded (the rfft plan's planes are shared
     * state; the stride inner keeps per-tid scratch for MT). NULL = stride. */
    rfft_plan_t *rfft_row;
    /* §6a32: bwd twin — c2r natural-engine row inner (same rules: injected,
     * measured-adopted, ST only). NULL = stride inner. */
    c2r_plan_t *c2r_row;
    /* §6a39: strided r2c/c2r whole-row-pass engines (measured-adopted). */
    _f2d_sr2c_fwd_fn strided_fwd;
    _f2d_sr2c_bwd_fn strided_bwd;
    int str_blk;                  /* §6a48: edition block quantum (pairs) */
    double *tail_scr;             /* §6a48: staging for ragged row counts */
    /* §6a41: strided TWIDDLE-STAGE row engines (N2 in {128,256}; front +
     * r64 monos + mapped split, row-blocked — see strided_tw.h). */
    _stw_tables_t stw_tab;
    int stw_on_fwd, stw_on_bwd;
    double *stw_work;

    int num_scratch;              /* per-thread scratch slots */
    size_t tile_real_sz;          /* N2 * B */
    size_t tile_complex_sz;       /* K_pad * B (was (N2/2+1)*B; now padded) */
    double *scratch_re;           /* num_scratch * tile_real_sz doubles */
    double *scratch_im;           /* num_scratch * tile_complex_sz doubles */

    /* Padded col-FFT scratch sized N1 * K_pad doubles each. After tiled row
     * pass writes here, col FFT runs on this with K=K_pad. */
    double *re_pad;
    double *im_pad;

    /* Cached scratch for the OOP convenience wrappers (stride_execute_2d_r2c /
     * _c2r). The in-place override needs a re buffer sized real_sz (= N1*N2);
     * c2r additionally needs a temp im buffer (cplx_sz). Allocated ONCE at
     * plan-create (not per call) so the public OOP API does no malloc/free in
     * the hot path — MKL's descriptor likewise pre-allocates its scratch. Not
     * re-entrant per plan (one transform at a time, like an MKL descriptor); the
     * tile-parallel threads use the per-slot scratch_re/im, not these. */
    double *oop_re_tmp;   /* real_sz = N1*N2 doubles      (r2c forward scratch) */
    double *oop_im_tmp;   /* cplx_sz = N1*(N2/2+1) doubles (c2r backward temp im) */

    /* Mixed-radix digit-reversal permutation for col FFT (size N1). Multi-stage
     * DIT plans output at digit-reversed positions; pack/unpack uses perm to
     * remap user-natural i <-> col-FFT-output i. */
    int *perm;

    stride_plan_t *plan_r2c;      /* N=N2, K=B, R2C inner */
    stride_plan_t *plan_col;      /* N=N1, K=K_pad, C2C col */

    /* JIT/baked resolved column c2c executor (NULL -> generic). Filled by
     * _fft2d_r2c_jit_resolve under VFFT_USE_JIT; else NULL (zero behavior change).
     * The ROW r2c/c2r pass stays generic — it's a per-tile worker-shim entry over
     * the fused/sliced stride-r2c engine (tid-threaded scratch slots), NOT a
     * whole-plan call, so it's deferred (same blocker as strided-r2c JIT). */
    vfft_proto_exec_fn exec_col_fwd, exec_col_bwd;
} stride_fft2d_r2c_data_t;


/* ═══════════════════════════════════════════════════════════════
 * Helpers
 * ═══════════════════════════════════════════════════════════════ */

/* Resolve the column c2c pass (fwd+bwd) to its baked-or-JIT executor (NULL on
 * miss -> the passes fall back to the generic c2c executor). The col plan is a
 * plain whole-plan c2c stride_plan_t, identical in shape to what fft2d.h already
 * JITs; the JIT'd c2c is roundtrip/order-identical to the generic, so d->perm
 * (built from plan_col->factors) stays valid.
 *
 * ROW pass: the row r2c/c2r runs the stride r2c engine per tile (plan_r2c's inner
 * c2c). We JIT that inner's sliced stages (the row workers call it on per-tile,
 * per-tid scratch — reentrant, no shared mutable state). The fused pack/fold
 * stage 0 stays generic (bespoke codelet). r2c fwd uses the inner's fwd JIT, c2r
 * bwd the bwd JIT; both no-op if the inner isn't a stride r2c plan. */
static inline void _fft2d_r2c_jit_resolve(stride_fft2d_r2c_data_t *d) {
#ifdef VFFT_USE_JIT
    if (d->plan_col) {
        d->exec_col_fwd = vfft_proto_plan_jit_fwd(d->plan_col);
        d->exec_col_bwd = vfft_proto_plan_jit_bwd(d->plan_col);
    }
    if (d->plan_r2c) {
        stride_plan_t *rin = stride_r2c_inner_plan(d->plan_r2c);
        if (rin) {
            stride_r2c_set_inner_jit_fwd(d->plan_r2c, vfft_proto_plan_jit_fwd(rin));
            stride_r2c_set_inner_jit_bwd(d->plan_r2c, vfft_proto_plan_jit_bwd(rin));
        }
    }
#else
    (void)d;
#endif
}

static inline double *_fft2d_r2c_scratch_re(stride_fft2d_r2c_data_t *d, int t) {
    return d->scratch_re + (size_t)t * d->tile_real_sz;
}
static inline double *_fft2d_r2c_scratch_im(stride_fft2d_r2c_data_t *d, int t) {
    return d->scratch_im + (size_t)t * d->tile_complex_sz;
}

/* Run inner R2C single-threaded on the caller's thread, regardless of the
 * global num_threads setting. Tile-parallel outer wants no nested dispatch.
 * `tid` selects the inner plan's per-worker scratch slot — each concurrent
 * tile thread MUST pass a distinct tid (the inner uses d->scratch as its pack
 * buffer; a shared slot races → garbage). Slots exist up to d->n_threads. */
static inline void _fft2d_r2c_inner_fwd(stride_plan_t *plan, double *re, double *im, int tid) {
    stride_r2c_data_t *d = (stride_r2c_data_t *)plan->override_data;
    _r2c_worker_arg_t a = { d, re, im, 0, d->K, tid };
    _r2c_worker_fwd(&a);
}
static inline void _fft2d_r2c_inner_bwd(stride_plan_t *plan, double *re, double *im, int tid) {
    stride_r2c_data_t *d = (stride_r2c_data_t *)plan->override_data;
    _r2c_worker_arg_t a = { d, re, im, 0, d->K, tid };
    _r2c_worker_bwd(&a);
}


/* ═══════════════════════════════════════════════════════════════
 * TILED ROW PASS — forward (R2C)
 *
 * For tile of B rows starting at i_tile:
 *   gather real B x N2 -> scratch_re (N2 x B)
 *   inner R2C  (in-place on scratch_re/scratch_im, N=N2 K=B)
 *   scatter scratch (N2/2+1 x B split-complex) -> out (B x (N2/2+1))
 * ═══════════════════════════════════════════════════════════════ */

/* Forward row pass: re_in is the user's real input buffer (read-only here).
 * out_pad_re/out_pad_im are the padded col-FFT scratch (written here). */
static void _fft2d_r2c_tiled_fwd_range(stride_fft2d_r2c_data_t *d,
                                        const double *re_in,
                                        double *out_pad_re, double *out_pad_im,
                                        double *sr, double *si,
                                        size_t row_start, size_t row_end,
                                        int tid)
{
    const int N2 = d->N2;
    const int halfN_plus1 = N2 / 2 + 1;
    const size_t B = d->B;
    const size_t K_pad = d->K_pad;

    for (size_t i = row_start; i < row_end; i += B) {
        size_t this_B = B;
        if (i + B > row_end) this_B = row_end - i;

        /* Gather: real B x N2 -> scratch_re N2 x B (single-plane transpose). */
        _F2D_T0(_f2d_p1_tin);
        stride_transpose(re_in + i * (size_t)N2, (size_t)N2,
                         sr, B, this_B, (size_t)N2);
        _F2D_T1(_f2d_p1_tin);

        /* Inner R2C in-place on scratch. After: sr[f*B + k_local] holds Re bins,
         * si[f*B + k_local] holds Im bins for f=0..N2/2. */
        _F2D_T0(_f2d_p1_r2c);
        if (d->rfft_row && stride_get_num_threads() <= 1)
            /* §6a31: rfft engine, in-place-safe (leaf fully consumes x
             * before the terminator writes out); ST only. */
            rfft_execute_fwd_natural(d->rfft_row, sr, sr, si, NULL);
        else
            _fft2d_r2c_inner_fwd(d->plan_r2c, sr, si, tid);
        _F2D_T1(_f2d_p1_r2c);

        /* Scatter split-complex: (halfN_plus1) x B -> B x K_pad (padded).
         * Padding columns [halfN_plus1..K_pad) are zeroed for col-FFT. */
        _F2D_T0(_f2d_p1_tout);
        stride_transpose_pair(sr, si,
                              out_pad_re + i * K_pad,
                              out_pad_im + i * K_pad,
                              B, K_pad,
                              (size_t)halfN_plus1, this_B);
        /* Zero the padding columns of the rows we just wrote. */
        for (size_t r = 0; r < this_B; r++) {
            double *rr = out_pad_re + (i + r) * K_pad;
            double *ii = out_pad_im + (i + r) * K_pad;
            for (size_t f = (size_t)halfN_plus1; f < K_pad; f++) {
                rr[f] = 0.0;
                ii[f] = 0.0;
            }
        }
        _F2D_T1(_f2d_p1_tout);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * TILED ROW PASS — backward (C2R)
 *
 * Process tiles in REVERSE order: scatter writes wider rows (N2 reals)
 * than gather reads (N2/2+1 complex). Forward order would overwrite
 * future tiles' input. Reverse order keeps later tiles' read region
 * intact until they are processed.
 *
 * For tile of B rows starting at i_tile:
 *   gather complex B x (N2/2+1) -> scratch (N2/2+1 x B split)
 *   inner C2R (in-place on scratch_re/scratch_im, N=N2 K=B)
 *   scatter scratch_re (N2 x B real) -> re (B x N2 real)
 * ═══════════════════════════════════════════════════════════════ */

/* Backward row pass: gathers from padded col-FFT output (in_pad_re/im),
 * writes real samples to re_out. Reads only the lower halfN_plus1 columns
 * of the padded layout (the rest is zero/garbage from forward padding). */
static void _fft2d_r2c_tiled_bwd_range(stride_fft2d_r2c_data_t *d,
                                        const double *in_pad_re, const double *in_pad_im,
                                        double *re_out,
                                        double *sr, double *si,
                                        size_t row_start, size_t row_end,
                                        int tid)
{
    const int N2 = d->N2;
    const int halfN_plus1 = N2 / 2 + 1;
    const size_t B = d->B;
    const size_t K_pad = d->K_pad;

    if (row_end <= row_start) return;
    size_t span = row_end - row_start;
    size_t n_tiles = (span + B - 1) / B;

    for (size_t k = n_tiles; k > 0; k--) {
        size_t i = row_start + (k - 1) * B;
        size_t this_B = B;
        if (i + B > row_end) this_B = row_end - i;

        /* Gather split-complex: B x K_pad (read only halfN_plus1 cols) ->
         * (halfN_plus1) x B for the inner C2R. */
        stride_transpose_pair(in_pad_re + i * K_pad,
                              in_pad_im + i * K_pad,
                              sr, si,
                              K_pad, B,
                              this_B, (size_t)halfN_plus1);

        /* Inner C2R in-place on scratch. tid selects the inner's per-worker
         * pack-scratch slot — distinct per tile thread (was hardcoded 0, the
         * blocker that forced serial backward). */
        if (d->c2r_row && stride_get_num_threads() <= 1)
            /* §6a32: c2r natural engine — in-place-safe (the initiator
             * consumes all input rows in stage 0; out written last). */
            c2r_execute_natural(d->c2r_row, sr, si, sr, NULL);
        else
            _fft2d_r2c_inner_bwd(d->plan_r2c, sr, si, tid);

        /* Scatter real: N2 x B -> B x N2. */
        stride_transpose(sr, B, re_out + i * (size_t)N2, (size_t)N2,
                         (size_t)N2, this_B);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * TILE-PARALLEL THREADING (forward AND backward)
 *
 * Tiles are independent. Distribute across threads, each owns a scratch slot
 * + a distinct inner-pack tid. Backward threads too: its row pass reads from
 * the padded col-FFT scratch (re_pad/im_pad) and writes the real output to a
 * DISTINCT user buffer, so tiles never clobber each other (the reverse-order
 * note above describes an in-place-aliased layout that does not occur in the
 * c2r execute path — in_pad is always internal scratch, re_out the user
 * buffer). Each thread still walks its own tile range in reverse, harmless
 * when the buffers are disjoint.
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    stride_fft2d_r2c_data_t *d;
    const double *re_in;       /* read-only real input (forward) */
    double *out_re, *out_im;   /* padded col-FFT scratch destination */
    double *sr, *si;
    size_t row_start, row_end;
    int tid;
} _fft2d_r2c_tile_arg_t;

static void _fft2d_r2c_tile_fwd_trampoline(void *arg) {
    _fft2d_r2c_tile_arg_t *a = (_fft2d_r2c_tile_arg_t *)arg;
    _fft2d_r2c_tiled_fwd_range(a->d, a->re_in, a->out_re, a->out_im,
                                a->sr, a->si, a->row_start, a->row_end, a->tid);
}

static void _fft2d_r2c_tiled_fwd_mt(stride_fft2d_r2c_data_t *d,
                                     const double *re_in,
                                     double *out_re, double *out_im) {
    const size_t N1 = (size_t)d->N1;
    const size_t B = d->B;
    /* The plan's snapshot is the SMALLER of its own tile-scratch slots and the
     * inner r2c plan's pack-scratch slots (two tile threads on one inner slot
     * would collide: garbage output). The pool's one clamp takes it. */
    int slots = d->num_scratch;
    {
        stride_r2c_data_t *rd = (stride_r2c_data_t *)d->plan_r2c->override_data;
        if (slots > rd->n_threads) slots = rd->n_threads;
    }
    int T = stride_pool_workers_for(slots);

    size_t n_tiles = (N1 + B - 1) / B;
    if (T <= 1 || n_tiles <= 1) {
        _fft2d_r2c_tiled_fwd_range(d, re_in, out_re, out_im,
                                    d->scratch_re, d->scratch_im,
                                    0, N1, 0);
        return;
    }

    /* slot t owns scratch slot t and tid t; slot 0 is the caller */
    _fft2d_r2c_tile_arg_t args[STRIDE_POOL_MAX_DISPATCH];
    int n = 0;
    for (int t = 0; t < T; t++) {
        size_t tiles_start = (n_tiles * t) / T;
        size_t tiles_end   = (n_tiles * (t + 1)) / T;
        size_t row_start   = tiles_start * B;
        size_t row_end     = tiles_end * B;
        if (row_end > N1) row_end = N1;
        if (t > 0 && row_start >= N1) break;

        args[n].d = d;
        args[n].re_in = re_in;
        args[n].out_re = out_re;
        args[n].out_im = out_im;
        args[n].sr = _fft2d_r2c_scratch_re(d, t);
        args[n].si = _fft2d_r2c_scratch_im(d, t);
        args[n].row_start = row_start;
        args[n].row_end = row_end;
        args[n].tid = t;
        n++;
    }
    stride_pool_run(n, _fft2d_r2c_tile_fwd_trampoline, args, sizeof args[0]);
}

/* Backward (C2R) tile-parallel — same partition as the forward. Reads padded
 * col-FFT scratch (in_pad_re/im), writes reals to re_out (distinct buffer). */
typedef struct {
    stride_fft2d_r2c_data_t *d;
    const double *in_pad_re, *in_pad_im;
    double *re_out;
    double *sr, *si;
    size_t row_start, row_end;
    int tid;
} _fft2d_r2c_tile_bwd_arg_t;

static void _fft2d_r2c_tile_bwd_trampoline(void *arg) {
    _fft2d_r2c_tile_bwd_arg_t *a = (_fft2d_r2c_tile_bwd_arg_t *)arg;
    _fft2d_r2c_tiled_bwd_range(a->d, a->in_pad_re, a->in_pad_im, a->re_out,
                                a->sr, a->si, a->row_start, a->row_end, a->tid);
}

static void _fft2d_r2c_tiled_bwd_mt(stride_fft2d_r2c_data_t *d,
                                     const double *in_pad_re, const double *in_pad_im,
                                     double *re_out) {
    const size_t N1 = (size_t)d->N1;
    const size_t B = d->B;
    /* same snapshot rule as the forward: min(tile slots, inner pack slots) */
    int slots = d->num_scratch;
    {
        stride_r2c_data_t *rd = (stride_r2c_data_t *)d->plan_r2c->override_data;
        if (slots > rd->n_threads) slots = rd->n_threads;
    }
    int T = stride_pool_workers_for(slots);

    size_t n_tiles = (N1 + B - 1) / B;
    if (T <= 1 || n_tiles <= 1) {
        _fft2d_r2c_tiled_bwd_range(d, in_pad_re, in_pad_im, re_out,
                                    d->scratch_re, d->scratch_im,
                                    0, N1, 0);
        return;
    }

    _fft2d_r2c_tile_bwd_arg_t args[STRIDE_POOL_MAX_DISPATCH];
    int n = 0;
    for (int t = 0; t < T; t++) {
        size_t tiles_start = (n_tiles * t) / T;
        size_t tiles_end   = (n_tiles * (t + 1)) / T;
        size_t row_start   = tiles_start * B;
        size_t row_end     = tiles_end * B;
        if (row_end > N1) row_end = N1;
        if (t > 0 && row_start >= N1) break;

        args[n].d = d;
        args[n].in_pad_re = in_pad_re;
        args[n].in_pad_im = in_pad_im;
        args[n].re_out = re_out;
        args[n].sr = _fft2d_r2c_scratch_re(d, t);
        args[n].si = _fft2d_r2c_scratch_im(d, t);
        args[n].row_start = row_start;
        args[n].row_end = row_end;
        args[n].tid = t;
        n++;
    }
    stride_pool_run(n, _fft2d_r2c_tile_bwd_trampoline, args, sizeof args[0]);
}


/* ═══════════════════════════════════════════════════════════════
 * DISPATCH — forward and backward
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_r2c_execute_fwd(void *data, double *re, double *im) {
    stride_fft2d_r2c_data_t *d = (stride_fft2d_r2c_data_t *)data;

    /* Phase 1: tiled R2C row pass reads user's real input (re), writes to
     * padded col-FFT scratch (re_pad, im_pad) with row stride K_pad
     * (multiple of 4 — required by codelet vl). */
    _fft2d_r2c_tiled_fwd_mt(d, re, d->re_pad, d->im_pad);

    /* Phase 2: C2C col FFT at K=K_pad on padded scratch. */
    _F2D_T0(_f2d_p2);
    if (d->exec_col_fwd)
        d->exec_col_fwd(d->plan_col, d->re_pad, d->im_pad,
                        d->plan_col->K, d->plan_col->K, 0);   /* baked/JIT */
    else
        stride_execute_fwd(d->plan_col, d->re_pad, d->im_pad);
    _F2D_T1(_f2d_p2);

    /* Phase 3: pack padded N1*K_pad scratch -> user's N1*(N2/2+1) layout.
     * Col FFT output at row i is at digit-reversed position perm[i] in scratch.
     * Read at perm[i] to get natural-i output. */
    {
        _F2D_T0(_f2d_p3);
        const size_t hp1 = (size_t)(d->N2 / 2 + 1);
        for (int i = 0; i < d->N1; i++) {
            int p = d->perm[i];
            memcpy(re + (size_t)i * hp1,
                   d->re_pad + (size_t)p * d->K_pad,
                   hp1 * sizeof(double));
            memcpy(im + (size_t)i * hp1,
                   d->im_pad + (size_t)p * d->K_pad,
                   hp1 * sizeof(double));
        }
        _F2D_T1(_f2d_p3);
    }
}

static void _fft2d_r2c_execute_bwd(void *data, double *re, double *im) {
    stride_fft2d_r2c_data_t *d = (stride_fft2d_r2c_data_t *)data;
    const size_t hp1 = (size_t)(d->N2 / 2 + 1);
    const size_t K_pad = d->K_pad;

    /* Phase 1: unpack user's N1*(N2/2+1) packed input -> N1*K_pad padded
     * scratch with padding zeroed. Place row i at scratch row perm[i] —
     * col IFFT consumes its input in fwd-output (digit-reversed) layout
     * and produces natural-i output. */
    for (int i = 0; i < d->N1; i++) {
        int p = d->perm[i];
        memcpy(d->re_pad + (size_t)p * K_pad,
               re + (size_t)i * hp1,
               hp1 * sizeof(double));
        memcpy(d->im_pad + (size_t)p * K_pad,
               im + (size_t)i * hp1,
               hp1 * sizeof(double));
        for (size_t f = hp1; f < K_pad; f++) {
            d->re_pad[(size_t)p * K_pad + f] = 0.0;
            d->im_pad[(size_t)p * K_pad + f] = 0.0;
        }
    }

    /* Phase 2: C2C col IFFT at K=K_pad on padded scratch. */
    if (d->exec_col_bwd)
        d->exec_col_bwd(d->plan_col, d->re_pad, d->im_pad,
                        d->plan_col->K, d->plan_col->K, 0);   /* baked/JIT */
    else
        stride_execute_bwd(d->plan_col, d->re_pad, d->im_pad);

    /* Phase 3: tiled C2R row pass reads padded scratch (re_pad/im_pad), writes
     * reals to the user buffer `re`. Distinct buffers => tiles independent =>
     * tile-parallel (honors stride_get_num_threads(); serial when T<=1). */
    _fft2d_r2c_tiled_bwd_mt(d, d->re_pad, d->im_pad, re);
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_r2c_destroy(void *data) {
    stride_fft2d_r2c_data_t *d = (stride_fft2d_r2c_data_t *)data;
    if (!d) return;
    if (d->plan_r2c) stride_plan_destroy(d->plan_r2c);
    if (d->plan_col) stride_plan_destroy(d->plan_col);
    STRIDE_ALIGNED_FREE(d->scratch_re);
    STRIDE_ALIGNED_FREE(d->scratch_im);
    STRIDE_ALIGNED_FREE(d->re_pad);
    STRIDE_ALIGNED_FREE(d->im_pad);
    STRIDE_ALIGNED_FREE(d->oop_re_tmp);
    STRIDE_ALIGNED_FREE(d->oop_im_tmp);
    if (d->stw_work) { _stw_tables_free(&d->stw_tab); STRIDE_ALIGNED_FREE(d->stw_work); }
    STRIDE_ALIGNED_FREE(d->tail_scr);
    free(d->perm);
    free(d);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 *
 * Caller provides inner plans:
 *   plan_r2c: N=N2, K=B, R2C (will be owned by 2D plan)
 *   plan_col: N=N1, K=N2/2+1, C2C (will be owned by 2D plan)
 * ═══════════════════════════════════════════════════════════════ */

static void _fft2d_r2c_execute_fwd_oop(stride_fft2d_r2c_data_t *d,
                                       const double *real_in,
                                       double *out_re, double *out_im);
static void _fft2d_r2c_execute_bwd_oop(stride_fft2d_r2c_data_t *d,
                                       const double *in_re,
                                       const double *in_im,
                                       double *real_out);

static stride_plan_t *stride_plan_2d_r2c_from(int N1, int N2, size_t B,
                                               size_t K_pad,
                                               stride_plan_t *plan_r2c,
                                               stride_plan_t *plan_col)
{
    /* Caller must ensure:
     *   B == plan_r2c->K (they index the same row-pass scratch).
     *   K_pad == plan_col->K, K_pad multiple of 4, K_pad >= N2/2+1.
     * No clamping here — clamping would silently break layout invariants. */
    const size_t hp1 = (size_t)(N2 / 2 + 1);
    if (N1 < 2 || N2 < 2 || (N2 & 1) || !plan_r2c || !plan_col ||
        B < 2 || B > (size_t)N1 ||
        K_pad < hp1 || (K_pad & 3) != 0) {
        if (plan_r2c) stride_plan_destroy(plan_r2c);
        if (plan_col) stride_plan_destroy(plan_col);
        { fprintf(stderr, "[2dr2c NULL#%d]\n", 0); return NULL; }
    }

    stride_fft2d_r2c_data_t *d =
        (stride_fft2d_r2c_data_t *)calloc(1, sizeof(*d));
    if (!d) {
        stride_plan_destroy(plan_r2c);
        stride_plan_destroy(plan_col);
        { fprintf(stderr, "[2dr2c NULL#%d]\n", 1); return NULL; }
    }
    d->N1 = N1;
    d->N2 = N2;
    d->B = B;
    d->K_pad = K_pad;
    d->plan_r2c = plan_r2c;
    d->plan_col = plan_col;

    d->tile_real_sz = (size_t)N2 * B;
    /* Per-tile complex scratch: hp1 rows actually written by R2C, but we
     * size it generously at N2*B (= tile_real_sz) since that's the buffer
     * R2C reuses for input + Re bins. Im just needs hp1*B. */
    d->tile_complex_sz = hp1 * B;

    int T = stride_pool_workers_for(0); /* create time: the pool as it is now = this plan's slot count */
    d->num_scratch = T;

    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64,
        (size_t)T * d->tile_real_sz * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64,
        (size_t)T * d->tile_complex_sz * sizeof(double));
    /* Padded col-FFT scratch: N1 * K_pad doubles each. */
    d->re_pad = (double *)STRIDE_ALIGNED_ALLOC(64,
        (size_t)N1 * K_pad * sizeof(double));
    d->im_pad = (double *)STRIDE_ALIGNED_ALLOC(64,
        (size_t)N1 * K_pad * sizeof(double));
    /* OOP wrapper scratch, allocated once (see struct comment). */
    d->oop_re_tmp = (double *)STRIDE_ALIGNED_ALLOC(64,
        (size_t)N1 * (size_t)N2 * sizeof(double));
    d->oop_im_tmp = (double *)STRIDE_ALIGNED_ALLOC(64,
        (size_t)N1 * hp1 * sizeof(double));
    if (!d->scratch_re || !d->scratch_im || !d->re_pad || !d->im_pad ||
        !d->oop_re_tmp || !d->oop_im_tmp) {
        _fft2d_r2c_destroy(d);
        { fprintf(stderr, "[2dr2c NULL#%d]\n", 2); return NULL; }
    }

    /* Compute mixed-radix digit-reversal permutation for the col FFT. */
    d->perm = (int *)malloc((size_t)N1 * sizeof(int));
    if (!d->perm) { _fft2d_r2c_destroy(d); { fprintf(stderr, "[2dr2c NULL#%d]\n", 3); return NULL; } }
    {
        const int *factors = plan_col->factors;
        const int nf = plan_col->num_stages;
        for (int n = 0; n < N1; n++) {
            int idx = n, rev = 0, radix_product = 1;
            for (int s = 0; s < nf; s++) {
                int R = factors[s];
                int digit = idx % R;
                idx /= R;
                rev += digit * (N1 / (radix_product * R));
                radix_product *= R;
            }
            d->perm[n] = rev;
        }
    }

    /* §6a51: EMPIRICAL col-path verification (the §6a47b pattern ported).
     * The perm above is computed BLIND from plan_col->factors as standard
     * digit-reversal; for plans whose true output ordering differs (prime
     * N1 was the demonstrated case: cold-first creates half-succeeded with
     * rt~1.0 WRONG results) that assumption silently breaks. Impulse at
     * row 1, run the PRODUCTION col call, verify all N1 bins through the
     * perm against the closed form. Any mismatch => fail the build — never
     * a silently wrong spectrum. The row inner gets the same probe
     * (lane-batched impulse) as cheap insurance for its class. */
    {
        memset(d->re_pad, 0, (size_t)N1 * K_pad * sizeof(double));
        memset(d->im_pad, 0, (size_t)N1 * K_pad * sizeof(double));
        d->re_pad[1 * K_pad] = 1.0;
        if (d->exec_col_fwd)
            d->exec_col_fwd(d->plan_col, d->re_pad, d->im_pad,
                            d->plan_col->K, d->plan_col->K, 0);
        else
            stride_execute_fwd(d->plan_col, d->re_pad, d->im_pad);
        int cp_fail = 0;
        for (int k = 0; k < N1 && !cp_fail; k++) {
            int pp = d->perm[k];
            double vr = d->re_pad[(size_t)pp * K_pad];
            double vi = d->im_pad[(size_t)pp * K_pad];
            double a_ = -2.0 * 3.14159265358979323846 * (double)k / (double)N1;
            double er = vr - cos(a_), ei = vi - sin(a_);
            if (er * er + ei * ei > 1e-12) cp_fail = 1;
        }
        if (!cp_fail) {
            double *sr0 = _fft2d_r2c_scratch_re(d, 0);
            double *si0 = _fft2d_r2c_scratch_im(d, 0);
            memset(sr0, 0, d->tile_real_sz * sizeof(double));
            memset(si0, 0, d->tile_complex_sz * sizeof(double));
            sr0[1 * d->B] = 1.0;
            _fft2d_r2c_inner_fwd(d->plan_r2c, sr0, si0, 0);
            const int h_ = N2 / 2;
            for (int f = 0; f <= h_ && !cp_fail; f++) {
                double a_ = -2.0 * 3.14159265358979323846
                            * (double)f / (double)N2;
                double er = sr0[(size_t)f * d->B] - cos(a_);
                double ei = si0[(size_t)f * d->B] - sin(a_);
                if (er * er + ei * ei > 1e-12) cp_fail = 2;
            }
        }
        if (cp_fail) {
            fprintf(stderr, "[2dr2c probe FAIL#%d N1=%d N2=%d]\n",
                    cp_fail, N1, N2);
            _fft2d_r2c_destroy(d);
            return NULL;
        }
    }

    /* §6a39: measured adoption of the strided r2c/c2r whole-row-pass
     * engines (coverage N2 in {8,12,16,20,32,64}, N1 % 8 == 0; execute
     * re-guards T <= 1). fwd arms both read a preserved pattern input and
     * write the pads; bwd arms both read the pads read-only — no refills.
     * If create runs under MT the tiled arm may thread while strided is ST:
     * conservative under-adoption, consistent with the hysteresis
     * philosophy (§6a34: challenger must beat the incumbent by >5%). */
    {
        _f2d_sr2c_fwd_fn sf = _f2d_sr2c_fwd_resolve(N2, &d->str_blk);
        if ((sf) && ((size_t)N1 % (2 * (size_t)d->str_blk) != 0)) {
            const size_t hp1a = (size_t)(N2 / 2 + 1);
            d->tail_scr = (double *)STRIDE_ALIGNED_ALLOC(64,
                (2 * (size_t)d->str_blk * ((size_t)N2 + 2 * hp1a))
                    * sizeof(double));
            if (!d->tail_scr) sf = 0;   /* fail-safe: no staging, no engine */
        }
        _f2d_sr2c_bwd_fn sb = _f2d_sr2c_bwd_resolve(N2, &d->str_blk);
        if (sf && sb && N1 >= 8) {
            int aw_f = 0, aw_b = 0;
            if (vfft_adopt_lookup("2d", N1, N2, d->str_blk, &aw_f, &aw_b)) {
                /* §6a49: warm create — apply the persisted verdicts, skip
                 * both A/B blocks entirely. */
                d->strided_fwd = aw_f ? sf : 0;
                d->strided_bwd = aw_b ? sb : 0;
                goto aw2d_done;
            }
            double *xin = d->oop_re_tmp;
            for (size_t ii = 0; ii < (size_t)N1 * (size_t)N2; ii++)
                xin[ii] = 1.0 + 1e-3 * (double)(ii & 63);
            const size_t hp1s = (size_t)(N2 / 2 + 1);
            struct timespec t0_, t1_;
            double t_tile, t_str;
            _fft2d_r2c_tiled_fwd_mt(d, xin, d->re_pad, d->im_pad);
            clock_gettime(CLOCK_MONOTONIC, &t0_);
            for (int rr = 0; rr < 16; rr++)
                _fft2d_r2c_tiled_fwd_mt(d, xin, d->re_pad, d->im_pad);
            clock_gettime(CLOCK_MONOTONIC, &t1_);
            t_tile = (t1_.tv_sec - t0_.tv_sec) * 1e9
                   + (double)(t1_.tv_nsec - t0_.tv_nsec);
            _f2d_sr2c_fwd_rows(sf, d->str_blk, N2, xin, d->re_pad,
                               d->im_pad, (size_t)N2, K_pad, (size_t)N1,
                               d->tail_scr);
            clock_gettime(CLOCK_MONOTONIC, &t0_);
            for (int rr = 0; rr < 16; rr++) {
                _f2d_sr2c_fwd_rows(sf, d->str_blk, N2, xin, d->re_pad,
                                   d->im_pad, (size_t)N2, K_pad,
                                   (size_t)N1, d->tail_scr);
                for (int i2 = 0; i2 < N1; i2++)
                    for (size_t f2 = hp1s; f2 < K_pad; f2++) {
                        d->re_pad[(size_t)i2 * K_pad + f2] = 0.0;
                        d->im_pad[(size_t)i2 * K_pad + f2] = 0.0;
                    }
            }
            clock_gettime(CLOCK_MONOTONIC, &t1_);
            t_str = (t1_.tv_sec - t0_.tv_sec) * 1e9
                  + (double)(t1_.tv_nsec - t0_.tv_nsec);
            if (t_str * 20 < t_tile * 19)
                d->strided_fwd = sf;
            _fft2d_r2c_tiled_bwd_mt(d, d->re_pad, d->im_pad, xin);
            clock_gettime(CLOCK_MONOTONIC, &t0_);
            for (int rr = 0; rr < 16; rr++)
                _fft2d_r2c_tiled_bwd_mt(d, d->re_pad, d->im_pad, xin);
            clock_gettime(CLOCK_MONOTONIC, &t1_);
            t_tile = (t1_.tv_sec - t0_.tv_sec) * 1e9
                   + (double)(t1_.tv_nsec - t0_.tv_nsec);
            _f2d_sr2c_bwd_rows(sb, d->str_blk, N2, d->re_pad, d->im_pad,
                               xin, K_pad, (size_t)N2, (size_t)N1,
                               d->tail_scr);
            clock_gettime(CLOCK_MONOTONIC, &t0_);
            for (int rr = 0; rr < 16; rr++)
                _f2d_sr2c_bwd_rows(sb, d->str_blk, N2, d->re_pad,
                                   d->im_pad, xin, K_pad, (size_t)N2,
                                   (size_t)N1, d->tail_scr);
            clock_gettime(CLOCK_MONOTONIC, &t1_);
            t_str = (t1_.tv_sec - t0_.tv_sec) * 1e9
                  + (double)(t1_.tv_nsec - t0_.tv_nsec);
            if (t_str * 20 < t_tile * 19)
                d->strided_bwd = sb;
            vfft_adopt_record("2d", N1, N2, d->str_blk,
                              d->strided_fwd ? 1 : 0,
                              d->strided_bwd ? 1 : 0);
aw2d_done:;
        }
    }

    /* §6a41: measured adoption of the twiddle-stage engines (N2 128/256).
     * GATE-FIDELITY LESSON (measured the hard way at 256²): an isolated
     * hot-looped row-pass A/B misadopted stw (+15% execute-context
     * regression). The arms here therefore run the FULL fwd/bwd executors
     * with the flag toggled — same phase interleaving, same cache
     * behavior as production. Hysteresis unchanged (>5%). */
    /* §6a45: stw is a FALLBACK tier only — family 4 (emitted monos)
     * supersedes it at every covered N2, and its create A/B proved
     * unreliable under the avx512 build (misadopted +20%). Eligible only
     * where the mono resolver has no coverage. */
    if ((N2 == 128 || N2 == 256) && N1 >= 8
        && !_f2d_sr2c_fwd_resolve(N2, &(int){0})
        && _stw_tables_init(&d->stw_tab, N2)) {
        d->stw_work = (double *)STRIDE_ALIGNED_ALLOC(64,
            2 * 8 * (size_t)N2 * sizeof(double));
        if (!d->stw_work) { _stw_tables_free(&d->stw_tab); }
        else {
            double *xin = d->oop_re_tmp;
            double *xre = d->oop_im_tmp;                /* N1*hp1-sized */
            double *xim = (double *)STRIDE_ALIGNED_ALLOC(64,
                (size_t)N1 * hp1 * sizeof(double));
            if (!xim) goto stw_gate_done;
            for (size_t ii = 0; ii < (size_t)N1 * (size_t)N2; ii++)
                xin[ii] = 1.0 + 1e-3 * (double)(ii & 63);
            struct timespec t0_, t1_;
            double t_tile, t_str;
            d->stw_on_fwd = 0;
            _fft2d_r2c_execute_fwd_oop(d, xin, xre, xim);
            clock_gettime(CLOCK_MONOTONIC, &t0_);
            for (int rr = 0; rr < 8; rr++)
                _fft2d_r2c_execute_fwd_oop(d, xin, xre, xim);
            clock_gettime(CLOCK_MONOTONIC, &t1_);
            t_tile = (t1_.tv_sec - t0_.tv_sec) * 1e9
                   + (double)(t1_.tv_nsec - t0_.tv_nsec);
            d->stw_on_fwd = 1;
            _fft2d_r2c_execute_fwd_oop(d, xin, xre, xim);
            clock_gettime(CLOCK_MONOTONIC, &t0_);
            for (int rr = 0; rr < 8; rr++)
                _fft2d_r2c_execute_fwd_oop(d, xin, xre, xim);
            clock_gettime(CLOCK_MONOTONIC, &t1_);
            t_str = (t1_.tv_sec - t0_.tv_sec) * 1e9
                  + (double)(t1_.tv_nsec - t0_.tv_nsec);
            d->stw_on_fwd = (t_str * 20 < t_tile * 19) ? 1 : 0;
            d->stw_on_bwd = 0;
            _fft2d_r2c_execute_bwd_oop(d, xre, xim, xin);
            clock_gettime(CLOCK_MONOTONIC, &t0_);
            for (int rr = 0; rr < 8; rr++)
                _fft2d_r2c_execute_bwd_oop(d, xre, xim, xin);
            clock_gettime(CLOCK_MONOTONIC, &t1_);
            t_tile = (t1_.tv_sec - t0_.tv_sec) * 1e9
                   + (double)(t1_.tv_nsec - t0_.tv_nsec);
            d->stw_on_bwd = 1;
            _fft2d_r2c_execute_bwd_oop(d, xre, xim, xin);
            clock_gettime(CLOCK_MONOTONIC, &t0_);
            for (int rr = 0; rr < 8; rr++)
                _fft2d_r2c_execute_bwd_oop(d, xre, xim, xin);
            clock_gettime(CLOCK_MONOTONIC, &t1_);
            t_str = (t1_.tv_sec - t0_.tv_sec) * 1e9
                  + (double)(t1_.tv_nsec - t0_.tv_nsec);
            d->stw_on_bwd = (t_str * 20 < t_tile * 19) ? 1 : 0;
            STRIDE_ALIGNED_FREE(xim);
        }
    }
stw_gate_done: ;

    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _fft2d_r2c_destroy(d); { fprintf(stderr, "[2dr2c NULL#%d]\n", 4); return NULL; } }

    plan->N = N1 * N2;
    plan->K = 1;
    plan->num_stages = 0;
    plan->override_fwd     = _fft2d_r2c_execute_fwd;
    plan->override_bwd     = _fft2d_r2c_execute_bwd;
    plan->override_destroy = _fft2d_r2c_destroy;
    plan->override_data    = d;

    _fft2d_r2c_jit_resolve(d);   /* baked/JIT-resolve the column c2c (fwd+bwd) */
    return plan;
}


/* ═══════════════════════════════════════════════════════════════
 * CONVENIENCE API
 *
 * stride_execute_2d_r2c(plan, real_in, out_re, out_im):
 *   real_in: N1*N2 reals.
 *   out_re, out_im: each N1*(N2/2+1) doubles.
 *
 * stride_execute_2d_c2r(plan, in_re, in_im, real_out):
 *   in_re, in_im: each N1*(N2/2+1) doubles.
 *   real_out: N1*N2 reals.
 *
 * Both wrappers allocate temp scratch internally because the in-place
 * override requires the re buffer to be sized for the LARGER of input
 * (N1*N2 reals for forward) or output (N1*N2 reals for backward).
 * ═══════════════════════════════════════════════════════════════ */

/* §6a30: OOP-native executors — phase 1 reads the user input, phases 2-3 live
 * in the pad scratch, phase 3 writes the user output. The in-place ABI's
 * single re pointer forced the old wrappers to memcpy a full plane in and a
 * half plane out (measured 14.7% at 256²); these variants are copy-free and
 * bit-identical (same phases, same pad bytes). SPLIT callers only — the _z
 * veneer entries (fused interleave into the perm loops) were DELETED
 * 2026-08-26: interleaved 2D real callers are served by the native IL tier
 * (fft2d_real_il_design.md M3, no cross-layout serving by owner law). */
static void _fft2d_r2c_execute_fwd_oop(stride_fft2d_r2c_data_t *d,
                                       const double *real_in,
                                       double *out_re, double *out_im)
{
    if (d->strided_fwd) {
        /* §6a39/44/48: one strided sweep (MT full blocks + staged tail). */
        _f2d_sr2c_fwd_rows(d->strided_fwd, d->str_blk, d->N2, real_in,
                           d->re_pad, d->im_pad, (size_t)d->N2, d->K_pad,
                           (size_t)d->N1, d->tail_scr);
        const size_t hp1s = (size_t)(d->N2 / 2 + 1);
        for (int i = 0; i < d->N1; i++)
            for (size_t f = hp1s; f < d->K_pad; f++) {
                d->re_pad[(size_t)i * d->K_pad + f] = 0.0;
                d->im_pad[(size_t)i * d->K_pad + f] = 0.0;
            }
    } else if (d->stw_on_fwd && stride_get_num_threads() <= 1) {
        /* stw stays ST: shared d->stw_work buffer (and dormant tier). */
        _stw_r2c_fwd(&d->stw_tab, real_in, d->re_pad, d->im_pad,
                     (size_t)d->N2, d->K_pad, (size_t)d->N1, d->stw_work);
        const size_t hp1s = (size_t)(d->N2 / 2 + 1);
        for (int i = 0; i < d->N1; i++)
            for (size_t f = hp1s; f < d->K_pad; f++) {
                d->re_pad[(size_t)i * d->K_pad + f] = 0.0;
                d->im_pad[(size_t)i * d->K_pad + f] = 0.0;
            }
    } else
        _fft2d_r2c_tiled_fwd_mt(d, real_in, d->re_pad, d->im_pad);
    _F2D_T0(_f2d_p2);
    if (d->exec_col_fwd)
        d->exec_col_fwd(d->plan_col, d->re_pad, d->im_pad,
                        d->plan_col->K, d->plan_col->K, 0);
    else
        stride_execute_fwd(d->plan_col, d->re_pad, d->im_pad);
    _F2D_T1(_f2d_p2);
    {
        _F2D_T0(_f2d_p3);
        const size_t hp1 = (size_t)(d->N2 / 2 + 1);
        for (int i = 0; i < d->N1; i++) {
            int pp = d->perm[i];
            const double *sr = d->re_pad + (size_t)pp * d->K_pad;
            const double *si = d->im_pad + (size_t)pp * d->K_pad;
            memcpy(out_re + (size_t)i * hp1, sr, hp1 * sizeof(double));
            memcpy(out_im + (size_t)i * hp1, si, hp1 * sizeof(double));
        }
        _F2D_T1(_f2d_p3);
    }
}

static void _fft2d_r2c_execute_bwd_oop(stride_fft2d_r2c_data_t *d,
                                       const double *in_re, const double *in_im,
                                       double *real_out)
{
    const size_t hp1 = (size_t)(d->N2 / 2 + 1);
    const size_t K_pad = d->K_pad;
    for (int i = 0; i < d->N1; i++) {
        int pp = d->perm[i];
        double *dr = d->re_pad + (size_t)pp * K_pad;
        double *di = d->im_pad + (size_t)pp * K_pad;
        memcpy(dr, in_re + (size_t)i * hp1, hp1 * sizeof(double));
        memcpy(di, in_im + (size_t)i * hp1, hp1 * sizeof(double));
        for (size_t f = hp1; f < K_pad; f++) { dr[f] = 0.0; di[f] = 0.0; }
    }
    if (d->exec_col_bwd)
        d->exec_col_bwd(d->plan_col, d->re_pad, d->im_pad,
                        d->plan_col->K, d->plan_col->K, 0);
    else
        stride_execute_bwd(d->plan_col, d->re_pad, d->im_pad);
    if (d->strided_bwd)
        _f2d_sr2c_bwd_rows(d->strided_bwd, d->str_blk, d->N2, d->re_pad,
                           d->im_pad, real_out, d->K_pad, (size_t)d->N2,
                           (size_t)d->N1, d->tail_scr);
    else if (d->stw_on_bwd && stride_get_num_threads() <= 1)
        _stw_c2r_bwd(&d->stw_tab, d->re_pad, d->im_pad, real_out,
                     d->K_pad, (size_t)d->N2, (size_t)d->N1, d->stw_work);
    else
        _fft2d_r2c_tiled_bwd_mt(d, d->re_pad, d->im_pad, real_out);
}

static inline void stride_execute_2d_r2c(const stride_plan_t *plan,
                                          const double *real_in,
                                          double *out_re, double *out_im)
{
    stride_fft2d_r2c_data_t *d = (stride_fft2d_r2c_data_t *)plan->override_data;
    /* §6a30: copy-free OOP-native path (the old memcpy-around cost 14.7%;
     * wrapin/wrapout counters now legitimately read zero on this route). */
    _fft2d_r2c_execute_fwd_oop(d, real_in, out_re, out_im);
}

static inline void stride_execute_2d_c2r(const stride_plan_t *plan,
                                          const double *in_re, const double *in_im,
                                          double *real_out)
{
    stride_fft2d_r2c_data_t *d = (stride_fft2d_r2c_data_t *)plan->override_data;
    /* §6a30: copy-free OOP-native path. */
    _fft2d_r2c_execute_bwd_oop(d, in_re, in_im, real_out);
}


#endif /* STRIDE_FFT2D_R2C_H */
