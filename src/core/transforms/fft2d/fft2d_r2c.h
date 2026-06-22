/**
 * fft2d_r2c.h -- 2D Real-to-Complex / Complex-to-Real FFT
 *
 * Two phases (matching FFTW convention: reduce along inner axis):
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

#include "executor.h"
#include "planner.h"
#include "threads.h"
#include "proto_stride_compat.h"
#include "transpose.h"
#include "r2c.h"
#ifdef VFFT_USE_JIT
#include "jit_runtime.h"          /* JIT/baked resolve for the inner column c2c FFT */
#endif

#ifndef FFT2D_R2C_DEFAULT_TILE
#define FFT2D_R2C_DEFAULT_TILE 8
#endif

#ifndef FFT2D_R2C_MIN_TILE
#define FFT2D_R2C_MIN_TILE 4
#endif

#ifndef FFT2D_R2C_MAX_THREADS
#define FFT2D_R2C_MAX_THREADS 64
#endif


/* ═══════════════════════════════════════════════════════════════
 * 2D R2C PLAN DATA
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int N1;                       /* rows */
    int N2;                       /* cols (must be even) */
    size_t B;                     /* row tile height */
    size_t K_pad;                 /* col FFT batch dim, padded to multiple of 4
                                   * (codelet n1_fwd has no scalar tail at vl<4) */

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
        stride_transpose(re_in + i * (size_t)N2, (size_t)N2,
                         sr, B, this_B, (size_t)N2);

        /* Inner R2C in-place on scratch. After: sr[f*B + k_local] holds Re bins,
         * si[f*B + k_local] holds Im bins for f=0..N2/2. */
        _fft2d_r2c_inner_fwd(d->plan_r2c, sr, si, tid);

        /* Scatter split-complex: (halfN_plus1) x B -> B x K_pad (padded).
         * Padding columns [halfN_plus1..K_pad) are zeroed for col-FFT. */
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
    int T = stride_get_num_threads();
    if (T > d->num_scratch) T = d->num_scratch;
    /* The inner r2c plan uses its own per-slot pack scratch (n_threads slots).
     * Don't dispatch more tile threads than the inner has scratch slots, or
     * two threads would collide on an inner slot (garbage output). */
    {
        stride_r2c_data_t *rd = (stride_r2c_data_t *)d->plan_r2c->override_data;
        if (T > rd->n_threads) T = rd->n_threads;
    }

    size_t n_tiles = (N1 + B - 1) / B;
    if (T <= 1 || n_tiles <= 1) {
        _fft2d_r2c_tiled_fwd_range(d, re_in, out_re, out_im,
                                    d->scratch_re, d->scratch_im,
                                    0, N1, 0);
        return;
    }

    _fft2d_r2c_tile_arg_t args[FFT2D_R2C_MAX_THREADS];
    int n_dispatch = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++) {
        size_t tiles_start = (n_tiles * t) / T;
        size_t tiles_end   = (n_tiles * (t + 1)) / T;
        size_t row_start   = tiles_start * B;
        size_t row_end     = tiles_end * B;
        if (row_end > N1) row_end = N1;
        if (row_start >= N1) break;

        args[t].d = d;
        args[t].re_in = re_in;
        args[t].out_re = out_re;
        args[t].out_im = out_im;
        args[t].sr = _fft2d_r2c_scratch_re(d, t);
        args[t].si = _fft2d_r2c_scratch_im(d, t);
        args[t].row_start = row_start;
        args[t].row_end = row_end;
        args[t].tid = t;
        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _fft2d_r2c_tile_fwd_trampoline, &args[t]);
        n_dispatch++;
    }
    {
        size_t row_end = ((n_tiles * 1) / T) * B;
        if (row_end > N1) row_end = N1;
        _fft2d_r2c_tiled_fwd_range(d, re_in, out_re, out_im,
                                    d->scratch_re, d->scratch_im,
                                    0, row_end, 0);
    }
    if (n_dispatch > 0) _stride_pool_wait_all();
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
    int T = stride_get_num_threads();
    if (T > d->num_scratch) T = d->num_scratch;
    /* Same inner-slot cap as the forward: don't dispatch more tile threads than
     * the inner r2c plan has pack-scratch slots (else two tiles collide). */
    {
        stride_r2c_data_t *rd = (stride_r2c_data_t *)d->plan_r2c->override_data;
        if (T > rd->n_threads) T = rd->n_threads;
    }

    size_t n_tiles = (N1 + B - 1) / B;
    if (T <= 1 || n_tiles <= 1) {
        _fft2d_r2c_tiled_bwd_range(d, in_pad_re, in_pad_im, re_out,
                                    d->scratch_re, d->scratch_im,
                                    0, N1, 0);
        return;
    }

    _fft2d_r2c_tile_bwd_arg_t args[FFT2D_R2C_MAX_THREADS];
    int n_dispatch = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++) {
        size_t tiles_start = (n_tiles * t) / T;
        size_t tiles_end   = (n_tiles * (t + 1)) / T;
        size_t row_start   = tiles_start * B;
        size_t row_end     = tiles_end * B;
        if (row_end > N1) row_end = N1;
        if (row_start >= N1) break;

        args[t].d = d;
        args[t].in_pad_re = in_pad_re;
        args[t].in_pad_im = in_pad_im;
        args[t].re_out = re_out;
        args[t].sr = _fft2d_r2c_scratch_re(d, t);
        args[t].si = _fft2d_r2c_scratch_im(d, t);
        args[t].row_start = row_start;
        args[t].row_end = row_end;
        args[t].tid = t;
        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _fft2d_r2c_tile_bwd_trampoline, &args[t]);
        n_dispatch++;
    }
    {
        size_t row_end = ((n_tiles * 1) / T) * B;
        if (row_end > N1) row_end = N1;
        _fft2d_r2c_tiled_bwd_range(d, in_pad_re, in_pad_im, re_out,
                                    d->scratch_re, d->scratch_im,
                                    0, row_end, 0);
    }
    if (n_dispatch > 0) _stride_pool_wait_all();
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
    if (d->exec_col_fwd)
        d->exec_col_fwd(d->plan_col, d->re_pad, d->im_pad,
                        d->plan_col->K, d->plan_col->K, 0);   /* baked/JIT */
    else
        stride_execute_fwd(d->plan_col, d->re_pad, d->im_pad);

    /* Phase 3: pack padded N1*K_pad scratch -> user's N1*(N2/2+1) layout.
     * Col FFT output at row i is at digit-reversed position perm[i] in scratch.
     * Read at perm[i] to get natural-i output. */
    {
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
        return NULL;
    }

    stride_fft2d_r2c_data_t *d =
        (stride_fft2d_r2c_data_t *)calloc(1, sizeof(*d));
    if (!d) {
        stride_plan_destroy(plan_r2c);
        stride_plan_destroy(plan_col);
        return NULL;
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

    int T = stride_get_num_threads();
    if (T > FFT2D_R2C_MAX_THREADS) T = FFT2D_R2C_MAX_THREADS;
    if (T < 1) T = 1;
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
        return NULL;
    }

    /* Compute mixed-radix digit-reversal permutation for the col FFT. */
    d->perm = (int *)malloc((size_t)N1 * sizeof(int));
    if (!d->perm) { _fft2d_r2c_destroy(d); return NULL; }
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

    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _fft2d_r2c_destroy(d); return NULL; }

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

static inline void stride_execute_2d_r2c(const stride_plan_t *plan,
                                          const double *real_in,
                                          double *out_re, double *out_im)
{
    stride_fft2d_r2c_data_t *d = (stride_fft2d_r2c_data_t *)plan->override_data;
    size_t real_sz = (size_t)d->N1 * (size_t)d->N2;
    size_t cplx_sz = (size_t)d->N1 * (size_t)(d->N2 / 2 + 1);
    /* In-place override needs re sized real_sz (= max(real_sz, cplx_sz)). Use the
     * plan's cached scratch (alloc'd once), then copy the lower cplx_sz to out_re. */
    double *re_tmp = d->oop_re_tmp;
    memcpy(re_tmp, real_in, real_sz * sizeof(double));
    plan->override_fwd(plan->override_data, re_tmp, out_im);
    memcpy(out_re, re_tmp, cplx_sz * sizeof(double));
}

static inline void stride_execute_2d_c2r(const stride_plan_t *plan,
                                          const double *in_re, const double *in_im,
                                          double *real_out)
{
    stride_fft2d_r2c_data_t *d = (stride_fft2d_r2c_data_t *)plan->override_data;
    size_t real_sz = (size_t)d->N1 * (size_t)d->N2;
    size_t cplx_sz = (size_t)d->N1 * (size_t)(d->N2 / 2 + 1);
    /* real_out is sized real_sz; also serves as the re scratch (real_sz >= cplx_sz). */
    memcpy(real_out, in_re, cplx_sz * sizeof(double));
    /* Temp im buffer (override_bwd writes to im too) — plan's cached scratch. */
    double *im_tmp = d->oop_im_tmp;
    memcpy(im_tmp, in_im, cplx_sz * sizeof(double));
    plan->override_bwd(plan->override_data, real_out, im_tmp);
}


#endif /* STRIDE_FFT2D_R2C_H */
