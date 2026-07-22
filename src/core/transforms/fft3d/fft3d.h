/**
 * fft3d.h -- 3D c2c FFT: three tensor-factor passes over a row-major cube.
 *
 * DFT_{N1xN2xN3} = (DFT_N1 (x) I_{N2N3}) (I_N1 (x) DFT_N2 (x) I_N3) (I_{N1N2} (x) DFT_N3)
 *
 * The lane-batched engine IS the (A (x) I_K) vector form, so each factor maps
 * onto an existing primitive -- no new butterfly math:
 *
 *   Pass A (axis 0): N1-point FFT, K = N2*N3, run natively (the 2D column
 *     pass with a bigger K). Two execution modes:
 *       FLAT    -- whole-K lane ranges (the classic K-split shape).
 *       BLOCKED -- each thread's lane range is subdivided into a_block-lane
 *         chunks and ALL stages run per chunk before the next (FFTW's vector
 *         recursion / SPIRAL eq. 19 realized via the K-split slice primitive:
 *         a lane block is an independent sub-problem because axis-0
 *         butterflies never mix lanes). Collapses stages(N1) DRAM sweeps
 *         into ~1 and confines the TLB working set to 16*N1*a_block bytes.
 *     GATE: the lane-split path (both modes at any offset/T) requires a plain
 *     DIT chain -- use_dif_forward or an override plan (Rader/Bluestein for
 *     prime N1) falls back to one whole-K stride_execute_* call, mirroring
 *     the production executor's "DIF runs single-threaded for v1.1" policy
 *     and the 2D v1.0 K-split safety posture.
 *
 *   Pass B (axis 1): ONE plan at (N2, K = N3), executed once per i-plane at
 *     base re + i*N2*N3 through the ST proto executor -- the same call shape
 *     _fft2d_tiled_range uses inside workers, so plane-parallel MT is safe by
 *     existing precedent. Accidentally optimal memory shape: a plane
 *     (16*N2*N3 bytes) is cache-resident, so all stages(N2) run in L2/L3 and
 *     the pass costs ~1 DRAM sweep. Prime N2 -> the proto executor dispatches
 *     the override internally per plane.
 *
 *   Pass C (axis 2): the 2D tiled row pass verbatim with row count N1*N2 --
 *     per tile of B rows: SIMD-gather B x N3 -> N3 x B scratch, N3-point FFT
 *     at K=B (L1-resident), SIMD-scatter back. Tile-parallel, per-thread
 *     scratch, no barriers.
 *
 * Pass order: the three factors act on disjoint axes and commute, so any of
 * the 3! orders is mathematically valid. Convention here (matching 2D):
 * fwd = A -> B -> C, bwd = C -> B -> A. Order is a future calibration axis.
 *
 * Threading: three barrier-free modes (lane-range A, plane-range B, tile
 * range C); the pool is only joined at pass boundaries. The compat
 * stride_execute_* adapters in this include set are single-threaded, so
 * pass A carries its own lane-range dispatch rather than relying on them.
 *
 * Data layout (split-complex, row-major cube):
 *   re[(i*N2 + j)*N3 + k]   for i=0..N1-1, j=0..N2-1, k=0..N3-1
 *
 * Normalization: bwd(fwd(x)) = N1*N2*N3 * x. Output is digit-scrambled per
 * axis (the dag convention) -- roundtrip-definitive correctness; natural
 * order is a follow-up (nat_col_list carries the axis-2 hook, as in 2D).
 */
#ifndef STRIDE_FFT3D_H
#define STRIDE_FFT3D_H

#include "executor.h"
#include "planner.h"
#include "prime_dispatch.h"       /* auto_plan_dispatch: CT else Rader/Bluestein for prime dims */
#include "exhaustive_plan.h"
#include "threads.h"
#include "proto_stride_compat.h"
#include "transpose.h"
#include "natorder_exec.h"        /* axis-2 (within-row) reorder hook, mirrors 2D mechanism-2 */
#ifdef VFFT_USE_JIT
#include "jit_runtime.h"          /* JIT/baked resolve for the inner axis FFTs */
#endif

/* Minimum tile height for SIMD efficiency. */
#define FFT3D_MIN_TILE 4

/* Default tile height for pass C. B=8 keeps a tile in L1 for N3<=256. */
#include "strided_rows.h"

#ifndef FFT3D_DEFAULT_TILE
#define FFT3D_DEFAULT_TILE 8
#endif

/* Maximum threads for per-thread scratch / dispatch-arg arrays. */
#ifndef FFT3D_MAX_THREADS
#define FFT3D_MAX_THREADS 64
#endif

/* BLOCKED pass A: target split-complex working set per lane block,
 * 16 * N1 * a_block bytes. ~half an RPL P-core L2 by default; -D override
 * for other hosts (same spirit as TP_L2_BYTES in transpose.h). */
#ifndef FFT3D_A_BLOCK_BYTES
#define FFT3D_A_BLOCK_BYTES ((size_t)1 << 20)
#endif

/* Cubes at or under this split-complex footprint default to FLAT pass A
 * (cache-resident anyway; blocking is pure overhead there). Measured verdict
 * per (N1,N2,N3) cell belongs to the 3D calibrator; this is only the
 * builder-default heuristic. */
#ifndef FFT3D_A_FLAT_MAX_BYTES
#define FFT3D_A_FLAT_MAX_BYTES ((size_t)32 << 20)
#endif


/* ═══════════════════════════════════════════════════════════════
 * 3D PLAN DATA
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int N1;                    /* axis-0 FFT length (outermost)          */
    int N2;                    /* axis-1 FFT length                      */
    int N3;                    /* axis-2 FFT length (rows, innermost)    */

    stride_plan_t *plan_axis0; /* N1-point FFT, K = N2*N3 (native)       */
    stride_plan_t *plan_axis1; /* N2-point FFT, K = N3 (per-plane)       */
    stride_plan_t *plan_row;   /* N3-point FFT, K = B  (tiled row pass)  */
#ifdef VFFT_STRIDED_ROWS
    _vfft_strided_fn srow_fwd, srow_bwd;   /* NULL -> transpose+native */
#endif

    /* JIT/baked resolved inner executors (NULL -> generic). Filled by
     * _fft3d_jit_resolve under VFFT_USE_JIT; otherwise stay NULL and the
     * passes use the generic proto executor (zero behavior change). */
    vfft_proto_exec_fn exec_ax0_fwd, exec_ax0_bwd;
    vfft_proto_exec_fn exec_ax1_fwd, exec_ax1_bwd;
    vfft_proto_exec_fn exec_row_fwd, exec_row_bwd;

    size_t B;                  /* pass-C tile height */

    /* Pass-A mode: 0 = FLAT (whole lane range per thread), >0 = BLOCKED
     * lane-block size in lanes (multiple of 8). See header comment. */
    size_t a_block;

    /* Per-thread scratch for the tiled pass C (thread t uses
     * scratch_re + t * tile_sz). Allocated for num_scratch threads. */
    int num_scratch;
    size_t tile_sz;            /* N3 * B */
    double *scratch_re;
    double *scratch_im;

    /* Natural-order hook for the axis-2 (within-row) digit-reversal tape,
     * applied to the row-FFT SCRATCH at K=B, exactly as in fft2d.h
     * (mechanism-2). NULL = scrambled (default). BORROWED -- the vfft handle
     * owns the malloc; _fft3d_destroy must NOT free it. Axis-0/axis-1
     * whole-cube reorders are a vfft-level follow-up. */
    const int *nat_col_list;
} stride_fft3d_data_t;

/* Get scratch pointer for thread t */
static inline double *_fft3d_scratch(double *pool, size_t tile_sz, int t) {
    return pool + (size_t)t * tile_sz;
}

/* Resolve the three inner FFTs to their baked-or-JIT executors (NULL on miss
 * -> generic proto executor). The fns are called as
 * fn(plan, re, im, slice_K, plan->K, 0) -- the orchestrator's convention,
 * identical to fft2d.h's usage. */
static inline void _fft3d_jit_resolve(stride_fft3d_data_t *d) {
#ifdef VFFT_USE_JIT
    if (d->plan_axis0) { d->exec_ax0_fwd = vfft_proto_plan_jit_fwd(d->plan_axis0);
                         d->exec_ax0_bwd = vfft_proto_plan_jit_bwd(d->plan_axis0); }
    if (d->plan_axis1) { d->exec_ax1_fwd = vfft_proto_plan_jit_fwd(d->plan_axis1);
                         d->exec_ax1_bwd = vfft_proto_plan_jit_bwd(d->plan_axis1); }
    if (d->plan_row)   { d->exec_row_fwd = vfft_proto_plan_jit_fwd(d->plan_row);
                         d->exec_row_bwd = vfft_proto_plan_jit_bwd(d->plan_row); }
#else
    (void)d;
#endif
}


/* ═══════════════════════════════════════════════════════════════
 * PASS C — TILED ROW EXECUTOR (axis 2), single-threaded core
 *
 * _fft2d_tiled_range with the row set flattened to N1*N2 rows of
 * length N3. Rows are uniformly strided, so tiles never straddle
 * a plane boundary in any way that matters.
 * ═══════════════════════════════════════════════════════════════ */

static void _fft3d_tiled_range(stride_fft3d_data_t *d,
                                double *re, double *im,
                                double *sr, double *si,
                                size_t row_start, size_t row_end,
                                int is_bwd) {
    const int N3 = d->N3;
    const size_t B = d->B;
#ifdef VFFT_STRIDED_ROWS
    if (d->srow_fwd) {
        _vfft_strided_fn fn = is_bwd ? d->srow_bwd : d->srow_fwd;
        size_t span = row_end - row_start;
        size_t bulk = span - (span % (size_t)_VFFT_STRIDED_VW);
        if (bulk)
            fn(re + row_start * (size_t)N3, im + row_start * (size_t)N3,
               NULL, NULL, (size_t)N3, bulk);
        row_start += bulk;
        if (row_start < row_end)        /* rem < VW: padded strided tail --
                                         * SAME natural order as the bulk
                                         * (uniform per-row order for any R;
                                         * see strided_rows.h). Uses the
                                         * caller's tile scratch as staging. */
            _vfft_strided_tail_padded(fn, re, im, row_start,
                                      row_end - row_start, N3,
                                      sr, si);
        return;
    }
#endif
    /* axis-2 natural-order reorder scratch: cycle_pass at K=B needs 2*B
     * doubles; B is clamped to FFT3D_DEFAULT_TILE in _fft3d_choose_tile, so
     * this stack buffer always suffices. Optimized out when
     * nat_col_list==NULL (scrambled). Per-call => MT-safe. */
    double rtmp[2 * FFT3D_DEFAULT_TILE];

    for (size_t i = row_start; i < row_end; i += B) {
        size_t this_B = B;
        if (i + B > row_end) this_B = row_end - i;
        const size_t _f3d_runB = (B - this_B <= 1) ? B : this_B;  /* §6a60 */

        /* Gather: B x N3 -> N3 x B (ld_dst=B for plan's K=B layout) */
        stride_transpose_pair(
            re + i * N3, im + i * N3, sr, si,
            (size_t)N3, B, this_B, (size_t)N3);

        /* ORDER_NATURAL backward: re-scramble the N3 axis in scratch BEFORE
         * the inverse row FFT -- mirror of the forward unscramble. Junk lanes
         * [this_B,B) permute harmlessly; the scatter discards them. */
        if (is_bwd && d->nat_col_list)
            vfft_natorder_cycle_pass_inv(sr, si, B, d->nat_col_list, rtmp);

        /* FFT on scratch (sub-batch this_B of the B-wide tile). Full proto
         * executor -- dispatches DIT *or* DIF plus the specialized per-cell
         * executors (the 2D row-pass precedent; the old DIT-only slice helper
         * silently mis-ran DIF plans). */
        vfft_proto_exec_fn rf = is_bwd ? d->exec_row_bwd : d->exec_row_fwd;
        if (rf)
            rf(d->plan_row, sr, si, _f3d_runB, d->plan_row->K, 0);   /* baked/JIT */
        else if (is_bwd)
            vfft_proto_execute_bwd(d->plan_row, sr, si, _f3d_runB);
        else
            vfft_proto_execute_fwd(d->plan_row, sr, si, _f3d_runB);

        /* ORDER_NATURAL forward: unscramble the N3 axis in scratch (K=B,
         * full-SIMD, L1-hot) right after the row FFT, before the scatter. */
        if (!is_bwd && d->nat_col_list)
            vfft_natorder_cycle_pass(sr, si, B, d->nat_col_list, rtmp);

        /* Scatter: N3 x B -> B x N3 (ld_src=B) */
        stride_transpose_pair(
            sr, si, re + i * N3, im + i * N3,
            B, (size_t)N3, (size_t)N3, this_B);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * PASS C — TILE-PARALLEL THREADING (identical model to fft2d.h)
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    stride_fft3d_data_t *d;
    double *re, *im;
    double *sr, *si;           /* per-thread scratch */
    size_t row_start, row_end;
    int is_bwd;
} _fft3d_tile_arg_t;

static void _fft3d_tile_trampoline(void *arg) {
    _fft3d_tile_arg_t *a = (_fft3d_tile_arg_t *)arg;
    _fft3d_tiled_range(a->d, a->re, a->im, a->sr, a->si,
                        a->row_start, a->row_end, a->is_bwd);
}

static void _fft3d_tiled_mt(stride_fft3d_data_t *d,
                             double *re, double *im, int is_bwd) {
    const size_t NR = (size_t)d->N1 * (size_t)d->N2;   /* flattened rows */
    const size_t B = d->B;
    int T = stride_get_num_threads();

    if (T > d->num_scratch) T = d->num_scratch;

    size_t n_tiles = (NR + B - 1) / B;

    if (T <= 1 || n_tiles <= 1) {
        _fft3d_tiled_range(d, re, im,
                           d->scratch_re, d->scratch_im,
                           0, NR, is_bwd);
        return;
    }

    _fft3d_tile_arg_t args[FFT3D_MAX_THREADS];
    int n_dispatch = 0;

    for (int t = 1; t < T && t <= _stride_pool_size; t++) {
        size_t tiles_start = (n_tiles * (size_t)t) / (size_t)T;
        size_t tiles_end   = (n_tiles * (size_t)(t + 1)) / (size_t)T;
        size_t row_start   = tiles_start * B;
        size_t row_end     = tiles_end * B;
        if (row_end > NR) row_end = NR;
        if (row_start >= NR) break;

        args[t].d = d;
        args[t].re = re;
        args[t].im = im;
        args[t].sr = _fft3d_scratch(d->scratch_re, d->tile_sz, t);
        args[t].si = _fft3d_scratch(d->scratch_im, d->tile_sz, t);
        args[t].row_start = row_start;
        args[t].row_end = row_end;
        args[t].is_bwd = is_bwd;

        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _fft3d_tile_trampoline, &args[t]);
        n_dispatch++;
    }

    /* Thread 0 (caller) processes its own share */
    {
        size_t row_end = ((n_tiles * 1) / (size_t)T) * B;
        if (row_end > NR) row_end = NR;
        _fft3d_tiled_range(d, re, im,
                           d->scratch_re, d->scratch_im,
                           0, row_end, is_bwd);
    }

    if (n_dispatch > 0)
        _stride_pool_wait_all();
}


/* ═══════════════════════════════════════════════════════════════
 * PASS B — PER-PLANE AXIS-1 FFTs (plane-parallel)
 *
 * One (N2, K=N3) plan reused across all N1 planes with base offset
 * re + i*N2*N3, through the ST proto executor -- the exact call shape
 * the 2D tiled row pass already makes from inside workers (full-K,
 * lane-0 slice), so DIT/DIF and Rader/Bluestein overrides are all
 * dispatched correctly per plane.
 * ═══════════════════════════════════════════════════════════════ */

static void _fft3d_axis1_range(stride_fft3d_data_t *d,
                                double *re, double *im,
                                size_t p_start, size_t p_end,
                                int is_bwd) {
    const size_t plane = (size_t)d->N2 * (size_t)d->N3;
    const size_t K1 = d->plan_axis1->K;              /* == N3 */
    vfft_proto_exec_fn f = is_bwd ? d->exec_ax1_bwd : d->exec_ax1_fwd;

    for (size_t i = p_start; i < p_end; i++) {
        double *pr = re + i * plane;
        double *pi = im + i * plane;
        if (f)
            f(d->plan_axis1, pr, pi, K1, K1, 0);     /* baked/JIT */
        else if (is_bwd)
            vfft_proto_execute_bwd(d->plan_axis1, pr, pi, K1);
        else
            vfft_proto_execute_fwd(d->plan_axis1, pr, pi, K1);
    }
}

typedef struct {
    stride_fft3d_data_t *d;
    double *re, *im;
    size_t p_start, p_end;
    int is_bwd;
} _fft3d_plane_arg_t;

static void _fft3d_plane_trampoline(void *arg) {
    _fft3d_plane_arg_t *a = (_fft3d_plane_arg_t *)arg;
    _fft3d_axis1_range(a->d, a->re, a->im, a->p_start, a->p_end, a->is_bwd);
}

static void _fft3d_axis1_mt(stride_fft3d_data_t *d,
                             double *re, double *im, int is_bwd) {
    const size_t P = (size_t)d->N1;
    int T = stride_get_num_threads();
    if (T > FFT3D_MAX_THREADS) T = FFT3D_MAX_THREADS;

    if (T <= 1 || P <= 1) {
        _fft3d_axis1_range(d, re, im, 0, P, is_bwd);
        return;
    }

    _fft3d_plane_arg_t args[FFT3D_MAX_THREADS];
    int n_dispatch = 0;

    for (int t = 1; t < T && t <= _stride_pool_size; t++) {
        size_t p_start = (P * (size_t)t) / (size_t)T;
        size_t p_end   = (P * (size_t)(t + 1)) / (size_t)T;
        if (p_start >= p_end) continue;

        args[t].d = d;
        args[t].re = re;
        args[t].im = im;
        args[t].p_start = p_start;
        args[t].p_end = p_end;
        args[t].is_bwd = is_bwd;

        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _fft3d_plane_trampoline, &args[t]);
        n_dispatch++;
    }

    /* Thread 0 (caller) processes its own share */
    _fft3d_axis1_range(d, re, im, 0, P / (size_t)T, is_bwd);

    if (n_dispatch > 0)
        _stride_pool_wait_all();
}


/* ═══════════════════════════════════════════════════════════════
 * PASS A — AXIS-0 FFT at K = N2*N3 (lane-range parallel, optional
 * L2 blocking inside each range)
 *
 * A lane block [c, c+C) of the K-baked plan is an independent
 * sub-problem: axis-0 butterflies never mix lanes, group_base/stride
 * offsets are lane-uniform, and twiddles are K-replicated -- the same
 * invariants the production K-split MT path relies on. So a lane
 * block is executed by offsetting the base pointers and passing the
 * block width as slice_K. BLOCKED mode simply loops those slices with
 * an L2-sized C so all stages of a block complete before the next
 * block streams in.
 * ═══════════════════════════════════════════════════════════════ */

/* Lane-split legality: plain DIT chain only. DIF fwd is gated ST by the
 * production executor pending K-split validation (stride_executor.h v1.1
 * note); we mirror that for both directions out of the same caution, and
 * override plans (Rader/Bluestein for prime N1) run their own full-K
 * machinery. Gated plans take one whole-K stride_execute_* call. */
static inline int _fft3d_axis0_lane_split_ok(const stride_plan_t *p, int is_bwd) {
    if (p->use_dif_forward) return 0;
    if (is_bwd ? (p->override_bwd != NULL) : (p->override_fwd != NULL)) return 0;
    return 1;
}

static void _fft3d_axis0_lanes(stride_fft3d_data_t *d,
                                double *re, double *im,
                                size_t lane_start, size_t lane_end,
                                int is_bwd) {
    const stride_plan_t *p = d->plan_axis0;
    const size_t K = p->K;
    size_t C = d->a_block ? d->a_block : (lane_end - lane_start);
    vfft_proto_exec_fn f = is_bwd ? d->exec_ax0_bwd : d->exec_ax0_fwd;

    for (size_t c = lane_start; c < lane_end; c += C) {
        size_t this_C = C;
        if (c + C > lane_end) this_C = lane_end - c;
        if (f)
            f((stride_plan_t *)p, re + c, im + c, this_C, K, 0);  /* baked/JIT */
        else if (is_bwd)
            vfft_proto_execute_bwd(p, re + c, im + c, this_C);
        else
            vfft_proto_execute_fwd(p, re + c, im + c, this_C);
    }
}

typedef struct {
    stride_fft3d_data_t *d;
    double *re, *im;
    size_t lane_start, lane_end;
    int is_bwd;
} _fft3d_lane_arg_t;

static void _fft3d_lane_trampoline(void *arg) {
    _fft3d_lane_arg_t *a = (_fft3d_lane_arg_t *)arg;
    _fft3d_axis0_lanes(a->d, a->re, a->im, a->lane_start, a->lane_end, a->is_bwd);
}

static void _fft3d_axis0_mt(stride_fft3d_data_t *d,
                             double *re, double *im, int is_bwd) {
    const stride_plan_t *p = d->plan_axis0;

    if (!_fft3d_axis0_lane_split_ok(p, is_bwd)) {
        /* Whole-K fallback (compat adapter: override-dispatch or full-K
         * proto executor). Overrides carry their own internal machinery. */
        if (is_bwd) stride_execute_bwd((stride_plan_t *)p, re, im);
        else        stride_execute_fwd((stride_plan_t *)p, re, im);
        return;
    }

    const size_t K = p->K;
    int T = stride_get_num_threads();
    if (T > FFT3D_MAX_THREADS) T = FFT3D_MAX_THREADS;

    if (T <= 1 || K < 8) {
        _fft3d_axis0_lanes(d, re, im, 0, K, is_bwd);
        return;
    }

    /* Contiguous lane ranges, rounded to multiples of 8 (SIMD width for
     * doubles; matches the production K-split rounding). */
    const size_t S = ((K / (size_t)T) + 7) & ~(size_t)7;
    _fft3d_lane_arg_t args[FFT3D_MAX_THREADS];
    int n_dispatch = 0;

    for (int t = 1; t < T && t <= _stride_pool_size; t++) {
        size_t lane_start = (size_t)t * S;
        if (lane_start >= K) break;
        size_t lane_end = lane_start + S;
        if (lane_end > K) lane_end = K;

        args[t].d = d;
        args[t].re = re;
        args[t].im = im;
        args[t].lane_start = lane_start;
        args[t].lane_end = lane_end;
        args[t].is_bwd = is_bwd;

        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _fft3d_lane_trampoline, &args[t]);
        n_dispatch++;
    }

    /* Thread 0 (caller) processes lanes [0, min(S,K)) */
    {
        size_t s0 = S < K ? S : K;
        _fft3d_axis0_lanes(d, re, im, 0, s0, is_bwd);
    }

    if (n_dispatch > 0)
        _stride_pool_wait_all();
}


/* ═══════════════════════════════════════════════════════════════
 * DISPATCH
 *
 * fwd: A (axis 0) -> B (axis 1) -> C (axis 2)
 * bwd: C -> B -> A   (reverse by convention; factors commute)
 * ═══════════════════════════════════════════════════════════════ */

static void _fft3d_execute_fwd(void *data, double *re, double *im) {
    stride_fft3d_data_t *d = (stride_fft3d_data_t *)data;
    _fft3d_axis0_mt(d, re, im, 0);   /* pass A */
    _fft3d_axis1_mt(d, re, im, 0);   /* pass B */
    _fft3d_tiled_mt(d, re, im, 0);   /* pass C */
}

static void _fft3d_execute_bwd(void *data, double *re, double *im) {
    stride_fft3d_data_t *d = (stride_fft3d_data_t *)data;
    _fft3d_tiled_mt(d, re, im, 1);   /* pass C' */
    _fft3d_axis1_mt(d, re, im, 1);   /* pass B' */
    _fft3d_axis0_mt(d, re, im, 1);   /* pass A' */
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY
 * ═══════════════════════════════════════════════════════════════ */

static void _fft3d_destroy(void *data) {
    stride_fft3d_data_t *d = (stride_fft3d_data_t *)data;
    if (!d) return;
    if (d->plan_axis0) stride_plan_destroy(d->plan_axis0);
    if (d->plan_axis1) stride_plan_destroy(d->plan_axis1);
    if (d->plan_row)   stride_plan_destroy(d->plan_row);
    STRIDE_ALIGNED_FREE(d->scratch_re);
    STRIDE_ALIGNED_FREE(d->scratch_im);
    free(d);
}


/* ═══════════════════════════════════════════════════════════════
 * PLAN CREATION
 * ═══════════════════════════════════════════════════════════════ */

static stride_plan_t *_fft3d_wrap(stride_fft3d_data_t *d) {
    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _fft3d_destroy(d); return NULL; }
    d->nat_col_list = NULL;  /* scrambled by default; vfft sets it (borrowed) */
    _fft3d_jit_resolve(d);
#ifdef VFFT_STRIDED_ROWS
    /* B < VW would undersize the padded-tail staging (tile scratch is
     * N*B per plane); tiny-R cells stay native. */
    if (d->B >= (size_t)_VFFT_STRIDED_VW)
    _vfft_strided_lookup(d->N3, &d->srow_fwd, &d->srow_bwd);
    if (d->srow_fwd && !_vfft_strided_verify_natural(d->srow_fwd, d->N3))
        { d->srow_fwd = 0; d->srow_bwd = 0; }   /* fail-safe -> native(+tape) */
#endif
    plan->N = (int)((size_t)d->N1 * (size_t)d->N2 * (size_t)d->N3);
    plan->K = 1;
    plan->num_stages = 0;
    plan->override_fwd     = _fft3d_execute_fwd;
    plan->override_bwd     = _fft3d_execute_bwd;
    plan->override_destroy = _fft3d_destroy;
    plan->override_data    = d;
    return plan;
}

static size_t _fft3d_choose_tile(int N3, size_t n_rows) {
    size_t B = FFT3D_DEFAULT_TILE;
    (void)N3;
    if (B > n_rows) B = n_rows;
    if (B < FFT3D_MIN_TILE) B = FFT3D_MIN_TILE;
    return B;
}

/* Default pass-A mode. FLAT for cubes that are cache-resident anyway;
 * otherwise a lane block sized so 16*N1*a_block ~= FFT3D_A_BLOCK_BYTES,
 * rounded down to a multiple of 8 lanes. Returns 0 (FLAT) or the block. */
static size_t _fft3d_choose_ablock(int N1, size_t K) {
    size_t cube_bytes = (size_t)16 * (size_t)N1 * K;
    if (cube_bytes <= FFT3D_A_FLAT_MAX_BYTES) return 0;
    size_t C = FFT3D_A_BLOCK_BYTES / ((size_t)16 * (size_t)N1);
    C &= ~(size_t)7;
    if (C < 8) C = 8;
    if (C >= K) return 0;
    return C;
}

/* Allocate per-thread scratch buffers for pass C.
 * Returns number of scratch slots allocated. */
static int _fft3d_alloc_scratch(stride_fft3d_data_t *d, size_t tile_sz) {
    int T = stride_get_num_threads();
    if (T > FFT3D_MAX_THREADS) T = FFT3D_MAX_THREADS;
    if (T < 1) T = 1;

    d->tile_sz = tile_sz;
    d->num_scratch = T;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)T * tile_sz * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)T * tile_sz * sizeof(double));

    if (!d->scratch_re || !d->scratch_im) return 0;
    return T;
}

/** Default 3D plan -- exhaustive sub-plan search per axis (small N only:
 *  exhaustive-at-create MEASURES, same warning as stride_plan_2d; the vfft
 *  wisdom path should use stride_plan_3d_from with calibrated inners).
 *  Pass-A mode from the builder heuristic. */
static stride_plan_t *stride_plan_3d(
        int N1, int N2, int N3,
        const vfft_proto_registry_t *reg)
{
    if (N1 < 1 || N2 < 1 || N3 < 1) return NULL;
    size_t total = (size_t)N1 * (size_t)N2 * (size_t)N3;
    if (total > (size_t)0x7fffffff) return NULL;   /* plan->N is int */

    stride_fft3d_data_t *d =
        (stride_fft3d_data_t *)calloc(1, sizeof(*d));
    if (!d) return NULL;

    const size_t K0 = (size_t)N2 * (size_t)N3;
    const size_t NR = (size_t)N1 * (size_t)N2;

    d->N1 = N1;
    d->N2 = N2;
    d->N3 = N3;
    d->B  = _fft3d_choose_tile(N3, NR);
    d->a_block = _fft3d_choose_ablock(N1, K0);

    /* Axis-0: N1-point, K = N2*N3. auto_plan_dispatch = CT if factorable,
     * else Rader/Bluestein (override plan) for a PRIME dimension -- the
     * override runs whole-K through the gated fallback in _fft3d_axis0_mt. */
    d->plan_axis0 = vfft_proto_exhaustive_plan(N1, K0, reg, 0);
    if (!d->plan_axis0) d->plan_axis0 = vfft_proto_auto_plan_dispatch(N1, K0, reg, NULL);
    if (!d->plan_axis0) { free(d); return NULL; }

    /* Axis-1: N2-point, K = N3 (one plan, executed per plane). */
    d->plan_axis1 = vfft_proto_exhaustive_plan(N2, (size_t)N3, reg, 0);
    if (!d->plan_axis1) d->plan_axis1 = vfft_proto_auto_plan_dispatch(N2, (size_t)N3, reg, NULL);
    if (!d->plan_axis1) { stride_plan_destroy(d->plan_axis0); free(d); return NULL; }

    /* Rows: N3-point, K = B (prime N3 -> Rader/Bluestein at K=B; scratch is
     * packed at stride B, matching the plan's baked K=B). */
    d->plan_row = vfft_proto_exhaustive_plan(N3, d->B, reg, 0);
    if (!d->plan_row) d->plan_row = vfft_proto_auto_plan_dispatch(N3, d->B, reg, NULL);
    if (!d->plan_row) {
        stride_plan_destroy(d->plan_axis0);
        stride_plan_destroy(d->plan_axis1);
        free(d); return NULL;
    }

    if (!_fft3d_alloc_scratch(d, (size_t)N3 * d->B)) {
        _fft3d_destroy(d);
        return NULL;
    }

    return _fft3d_wrap(d);
}

/** 3D plan from caller-supplied inner plans (wisdom-driven). Avoids the slow
 *  exhaustive sub-plan search -- the caller builds the inners from wisdom, so
 *  large N is tractable. plan_axis0 = N1-point c2c with K=N2*N3;
 *  plan_axis1 = N2-point c2c with K=N3; plan_row = N3-point c2c with K=B.
 *  a_block: 0 = FLAT, >0 = BLOCKED lane-block size (rounded to 8),
 *  (size_t)-1 = builder heuristic. The 3D plan TAKES OWNERSHIP of all three
 *  inners (frees them on failure / destroy). */
static stride_plan_t *stride_plan_3d_from(
        int N1, int N2, int N3, size_t B, size_t a_block,
        stride_plan_t *plan_axis0, stride_plan_t *plan_axis1,
        stride_plan_t *plan_row)
{
    if (N1 < 1 || N2 < 1 || N3 < 1 || B < 1 ||
        (size_t)N1 * (size_t)N2 * (size_t)N3 > (size_t)0x7fffffff ||
        !plan_axis0 || !plan_axis1 || !plan_row) {
        if (plan_axis0) stride_plan_destroy(plan_axis0);
        if (plan_axis1) stride_plan_destroy(plan_axis1);
        if (plan_row)   stride_plan_destroy(plan_row);
        return NULL;
    }
    stride_fft3d_data_t *d = (stride_fft3d_data_t *)calloc(1, sizeof(*d));
    if (!d) {
        stride_plan_destroy(plan_axis0);
        stride_plan_destroy(plan_axis1);
        stride_plan_destroy(plan_row);
        return NULL;
    }
    const size_t K0 = (size_t)N2 * (size_t)N3;
    d->N1 = N1; d->N2 = N2; d->N3 = N3; d->B = B;
    d->plan_axis0 = plan_axis0;
    d->plan_axis1 = plan_axis1;
    d->plan_row   = plan_row;
    if (a_block == (size_t)-1)
        d->a_block = _fft3d_choose_ablock(N1, K0);
    else if (a_block > 0) {
        size_t C = a_block & ~(size_t)7;
        if (C < 8) C = 8;
        d->a_block = (C >= K0) ? 0 : C;
    } else
        d->a_block = 0;
    if (!_fft3d_alloc_scratch(d, (size_t)N3 * B)) { _fft3d_destroy(d); return NULL; }
    return _fft3d_wrap(d);
}

#endif /* STRIDE_FFT3D_H */
