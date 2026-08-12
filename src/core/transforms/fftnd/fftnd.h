/**
 * fftnd.h -- rank-general (2..4D) c2c FFT: unfused leading passes + a fused
 * trailing group
 *
 * DFT_{N1x...xNd} = PROD_m ( I_{O_m} (x) DFT_{N_m} (x) I_{K_m} ),
 *   O_m = prod_{i<m} N_i (outer count),  K_m = prod_{i>m} N_i (inner lanes).
 *
 * Every factor maps onto an existing engine primitive (fft3d.h precedent):
 *   - axis 0        : one native call at K_0 (lane-blocked or flat)
 *   - middle axis m : loop of O_m native calls at K_m (each optionally
 *                     lane-blocked -- in 4D the axis-1 sub-volume is a whole
 *                     cube and outgrows cache, so blocking is a per-axis
 *                     property here, not a pass-A special)
 *   - last axis     : tiled row pass over O_{d-1} rows (gather/FFT/scatter)
 *
 * FUSION IS BIT-EXACT vs unfused: passes on disjoint axes commute and each
 * element's op sequence is unchanged by interleaving blocks, so fwd output
 * must memcmp-match across s (a test gate).
 *
 * Threading (barrier-free, pool joined at pass/group boundaries only):
 *   - unfused axis: hierarchical (outer x lane-range) work items, so small
 *     outer counts (anisotropic shapes) still fill T
 *   - fused group : block-parallel over the O_s blocks (per-thread tile
 *     scratch); if O_s < T the extra threads idle -- the calibrator's job is
 *     to pick s with grain in mind
 *
 * Gates (inherited from fft3d pass A): lane-offset slicing requires a plain
 * DIT chain; DIF-forward or override plans (Rader/Bluestein for prime axes)
 * execute full-K from lane 0 per outer -- always legal through the proto
 * executor (fft2d/fft3d row- and plane-pass precedent) -- and are simply
 * never lane-split.
 *
 * Layout: split-complex row-major; normalization bwd(fwd(x)) = (prod N_i) x;
 * output digit-scrambled per axis (dag convention).
 */
#ifndef STRIDE_FFTND_H
#define STRIDE_FFTND_H

#include "executor.h"
#include "planner.h"
#include "prime_dispatch.h"
#include "exhaustive_plan.h"
#include "threads.h"
#include "proto_stride_compat.h"
#include "strided_rows.h"       /* opt-in strided row pass (VFFT_STRIDED_ROWS) */
#include "transpose.h"
#include "natorder_exec.h"
#ifdef VFFT_USE_JIT
#include "jit_runtime.h"
#endif

#ifndef FFTND_MAX_RANK
#define FFTND_MAX_RANK 4
#endif

#define FFTND_MIN_TILE 4
#ifndef FFTND_DEFAULT_TILE
#define FFTND_DEFAULT_TILE 8
#endif
#ifndef FFTND_MAX_THREADS
#define FFTND_MAX_THREADS 64
#endif

/* Per-lane-block split-complex working-set target (~L2/2), as fft3d. */
#ifndef FFTND_A_BLOCK_BYTES
#define FFTND_A_BLOCK_BYTES ((size_t)1 << 20)
#endif
/* Axes whose whole sub-volume (16*N_m*K_m bytes) is at or under this run
 * flat -- cache-resident anyway. */
#ifndef FFTND_A_FLAT_MAX_BYTES
#define FFTND_A_FLAT_MAX_BYTES ((size_t)32 << 20)
#endif
/* Heuristic split point: smallest s whose fused block (16*K_{s-1} bytes)
 * fits this budget. Calibrator overrides per cell. */
#ifndef FFTND_FUSE_MAX_BYTES
#define FFTND_FUSE_MAX_BYTES ((size_t)4 << 20)
#endif


/* ═══════════════════════════════════════════════════════════════
 * PLAN DATA
 * ═══════════════════════════════════════════════════════════════ */

typedef struct {
    int rank;                       /* 2..FFTND_MAX_RANK */
    int N[FFTND_MAX_RANK];
    int split;                      /* s in [1, rank-1] */

    stride_plan_t *plan[FFTND_MAX_RANK];
#ifdef VFFT_STRIDED_ROWS
    _vfft_strided_fn srow_fwd, srow_bwd;   /* last-axis strided rows; NULL ->
                                              transpose+native tiled path */
#endif  /* axis m: N[m]-pt at K[m] (last: K=B) */
    vfft_proto_exec_fn exf[FFTND_MAX_RANK], exb[FFTND_MAX_RANK];
    size_t lane_block[FFTND_MAX_RANK];    /* 0 = flat; last axis unused */

    size_t O[FFTND_MAX_RANK];       /* outer count per axis */
    size_t K[FFTND_MAX_RANK];       /* inner lanes per axis (last = 1) */
    size_t total;                   /* prod N_i */

    size_t B;                       /* last-axis tile height */
    int num_scratch;
    size_t tile_sz;                 /* N[rank-1] * B */
    double *scratch_re, *scratch_im;

    /* last-axis natural-order tape hook (scratch-side, K=B), as fft2d/fft3d.
     * BORROWED; NULL = scrambled. */
    const int *nat_col_list;
} stride_fftnd_data_t;

static inline double *_fftnd_scratch(double *pool, size_t tile_sz, int t) {
    return pool + (size_t)t * tile_sz;
}

static inline void _fftnd_jit_resolve(stride_fftnd_data_t *d) {
#ifdef VFFT_USE_JIT
    for (int m = 0; m < d->rank; m++) if (d->plan[m]) {
        d->exf[m] = vfft_proto_plan_jit_fwd(d->plan[m]);
        d->exb[m] = vfft_proto_plan_jit_bwd(d->plan[m]);
    }
#else
    (void)d;
#endif
}

/* Lane-offset slicing legality (fft3d gate): plain DIT chain, no override. */
static inline int _fftnd_lane_split_ok(const stride_plan_t *p, int is_bwd) {
    if (p->use_dif_forward) return 0;
    if (is_bwd ? (p->override_bwd != NULL) : (p->override_fwd != NULL)) return 0;
    return 1;
}


/* ═══════════════════════════════════════════════════════════════
 * AXIS BODY -- outer-range x lane-range execution of one axis pass.
 * Shared by unfused passes and the fused per-block loop. The lane
 * chunk loop realizes per-axis blocking (fft3d blocked pass A,
 * promoted to every non-last axis).
 * ═══════════════════════════════════════════════════════════════ */

static void _fftnd_axis_outer_range(stride_fftnd_data_t *d, int m,
                                    double *re, double *im,
                                    size_t o_lo, size_t o_hi,
                                    size_t lane_lo, size_t lane_hi,
                                    int is_bwd) {
    const stride_plan_t *p = d->plan[m];
    const size_t Km = d->K[m];
    const size_t sub = (size_t)d->N[m] * Km;   /* elements per outer = K[m-1] */
    vfft_proto_exec_fn f = is_bwd ? d->exb[m] : d->exf[m];

    size_t C = d->lane_block[m];
    if (!C || !_fftnd_lane_split_ok(p, is_bwd)) C = lane_hi - lane_lo;

    for (size_t o = o_lo; o < o_hi; o++) {
        double *br = re + o * sub;
        double *bi = im + o * sub;
        for (size_t c = lane_lo; c < lane_hi; c += C) {
            size_t this_C = C;
            if (c + C > lane_hi) this_C = lane_hi - c;
            if (f)
                f((stride_plan_t *)p, br + c, bi + c, this_C, Km, 0);
            else if (is_bwd)
                vfft_proto_execute_bwd(p, br + c, bi + c, this_C);
            else
                vfft_proto_execute_fwd(p, br + c, bi + c, this_C);
        }
    }
}

/* ── unfused-axis MT: hierarchical (outer x lane-range) items ── */

typedef struct {
    stride_fftnd_data_t *d;
    int m, is_bwd;
    double *re, *im;
    size_t it_lo, it_hi;      /* item = (o - o_base)*ls + li */
    size_t ls, Ls;            /* lane splits per outer, rounded lane span */
    size_t o_base;            /* window start (0 for global passes) */
} _fftnd_axis_arg_t;

static void _fftnd_axis_item_range(_fftnd_axis_arg_t *a) {
    stride_fftnd_data_t *d = a->d;
    const size_t Km = d->K[a->m];
    for (size_t it = a->it_lo; it < a->it_hi; it++) {
        size_t o  = a->o_base + it / a->ls;
        size_t li = it % a->ls;
        size_t lane_lo = li * a->Ls;
        if (lane_lo >= Km) continue;
        size_t lane_hi = lane_lo + a->Ls;
        if (lane_hi > Km) lane_hi = Km;
        _fftnd_axis_outer_range(d, a->m, a->re, a->im, o, o + 1,
                                lane_lo, lane_hi, a->is_bwd);
    }
}

static void _fftnd_axis_trampoline(void *arg) {
    _fftnd_axis_item_range((_fftnd_axis_arg_t *)arg);
}

/* Windowed variant: MT over outer indices [o_lo, o_hi) of axis m only.
 * The global pass is the full-window case; the starved fused path (§ fused
 * MT below) uses per-block windows. Item decomposition is hierarchical
 * (outer x lane-range) so small windows still fill T. */
static void _fftnd_axis_mt_win(stride_fftnd_data_t *d, int m,
                               double *re, double *im,
                               size_t o_lo, size_t o_hi, int is_bwd) {
    const size_t O = o_hi - o_lo, Km = d->K[m];
    int T = stride_get_num_threads();
    if (T > FFTND_MAX_THREADS) T = FFTND_MAX_THREADS;

    /* Lane splitting only when legal, useful, and needed to fill T. */
    size_t ls = 1;
    if (T > 1 && O < (size_t)T && Km >= 16 &&
        _fftnd_lane_split_ok(d->plan[m], is_bwd)) {
        ls = ((size_t)T + O - 1) / O;
        size_t max_ls = Km / 8;
        if (ls > max_ls) ls = max_ls ? max_ls : 1;
    }
    size_t Ls = ls > 1 ? (((Km / ls) + 7) & ~(size_t)7) : Km;
    if (Ls == 0) { ls = 1; Ls = Km; }
    size_t total = O * ls;

    if (T <= 1 || total <= 1) {
        _fftnd_axis_outer_range(d, m, re, im, o_lo, o_hi, 0, Km, is_bwd);
        return;
    }

    _fftnd_axis_arg_t args[FFTND_MAX_THREADS];
    int n_dispatch = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++) {
        size_t it_lo = (total * (size_t)t) / (size_t)T;
        size_t it_hi = (total * (size_t)(t + 1)) / (size_t)T;
        if (it_lo >= it_hi) continue;
        args[t] = (_fftnd_axis_arg_t){ d, m, is_bwd, re, im,
                                       it_lo, it_hi, ls, Ls, o_lo };
        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _fftnd_axis_trampoline, &args[t]);
        n_dispatch++;
    }
    _fftnd_axis_arg_t a0 = { d, m, is_bwd, re, im,
                             0, total / (size_t)T, ls, Ls, o_lo };
    _fftnd_axis_item_range(&a0);
    if (n_dispatch > 0)
        _stride_pool_wait_all();
}

static void _fftnd_axis_mt(stride_fftnd_data_t *d, int m,
                           double *re, double *im, int is_bwd) {
    _fftnd_axis_mt_win(d, m, re, im, 0, d->O[m], is_bwd);
}


/* ═══════════════════════════════════════════════════════════════
 * LAST AXIS -- tiled row pass (fft3d._fft3d_tiled_range verbatim,
 * parameterized on the row window so fused blocks can call it).
 * ═══════════════════════════════════════════════════════════════ */

static void _fftnd_tiled_range(stride_fftnd_data_t *d,
                               double *re, double *im,
                               double *sr, double *si,
                               size_t row_start, size_t row_end,
                               int is_bwd) {
    const int NL = d->N[d->rank - 1];
    const size_t B = d->B;
#ifdef VFFT_STRIDED_ROWS
    if (d->srow_fwd) {
        const int NL = d->N[d->rank - 1];
        _vfft_strided_fn fn = is_bwd ? d->srow_bwd : d->srow_fwd;
        size_t span = row_end - row_start;
        size_t bulk = span - (span % (size_t)_VFFT_STRIDED_VW);
        if (bulk)
            fn(re + row_start * (size_t)NL, im + row_start * (size_t)NL,
               NULL, NULL, (size_t)NL, bulk);
        row_start += bulk;
        if (row_start < row_end)        /* rem < VW: padded strided tail --
                                         * SAME natural order as the bulk
                                         * (uniform per-row order for any R;
                                         * see strided_rows.h). Uses the
                                         * caller's tile scratch as staging. */
            _vfft_strided_tail_padded(fn, re, im, row_start,
                                      row_end - row_start, NL,
                                      sr, si);
        return;
    }
#endif
    const stride_plan_t *pr = d->plan[d->rank - 1];
    vfft_proto_exec_fn rf = is_bwd ? d->exb[d->rank - 1] : d->exf[d->rank - 1];
    double rtmp[2 * FFTND_DEFAULT_TILE];

    for (size_t i = row_start; i < row_end; i += B) {
        size_t this_B = B;
        if (i + B > row_end) this_B = row_end - i;

        stride_transpose_pair(re + i * NL, im + i * NL, sr, si,
                              (size_t)NL, B, this_B, (size_t)NL);

        if (is_bwd && d->nat_col_list)
            vfft_natorder_cycle_pass_inv(sr, si, B, d->nat_col_list, rtmp);

        /* §6a60 measured guard: full-B ONLY at this_B == B-1 (the hybrid's
         * SSE2+scalar straggler on B-1 lanes costs more than one wasted
         * full-width lane: fullB -11..-32% there). Everywhere else the
         * hybrid at this_B wins outright (fullB +61..+819% at small
         * remainders) — fftnd's original choice was right; the guard just
         * captures the one measured edge. Slack lanes are stale scratch
         * (lane-independent, discarded at scatter). */
        size_t run_B = (B - this_B <= 1) ? B : this_B;
        if (rf)
            rf((stride_plan_t *)pr, sr, si, run_B, pr->K, 0);
        else if (is_bwd)
            vfft_proto_execute_bwd(pr, sr, si, run_B);
        else
            vfft_proto_execute_fwd(pr, sr, si, run_B);

        if (!is_bwd && d->nat_col_list)
            vfft_natorder_cycle_pass(sr, si, B, d->nat_col_list, rtmp);

        stride_transpose_pair(sr, si, re + i * NL, im + i * NL,
                              B, (size_t)NL, (size_t)NL, this_B);
    }
}

/* global tile-parallel wrapper (used when split == rank-1: no fused middles) */
typedef struct {
    stride_fftnd_data_t *d;
    double *re, *im, *sr, *si;
    size_t row_start, row_end;
    int is_bwd;
} _fftnd_tile_arg_t;

static void _fftnd_tile_trampoline(void *arg) {
    _fftnd_tile_arg_t *a = (_fftnd_tile_arg_t *)arg;
    _fftnd_tiled_range(a->d, a->re, a->im, a->sr, a->si,
                       a->row_start, a->row_end, a->is_bwd);
}

/* Windowed variant: tile-parallel over rows [row_lo, row_hi). The window
 * start is tile-aligned by construction at both call sites (global: 0;
 * fused block: b * Rb where Rb is a multiple of nothing in particular --
 * so tiles are formed WITHIN the window: partitions split the window's
 * tile count, and _fftnd_tiled_range's own B-stepping from row_lo keeps
 * per-element op order identical to the ST/global schedule). */
static void _fftnd_tiled_mt_win(stride_fftnd_data_t *d,
                                double *re, double *im,
                                size_t row_lo, size_t row_hi, int is_bwd) {
    const size_t NR = row_hi - row_lo;
    const size_t B = d->B;
    int T = stride_get_num_threads();
    if (T > d->num_scratch) T = d->num_scratch;
    size_t n_tiles = (NR + B - 1) / B;

    if (T <= 1 || n_tiles <= 1) {
        _fftnd_tiled_range(d, re, im, d->scratch_re, d->scratch_im,
                           row_lo, row_hi, is_bwd);
        return;
    }
    _fftnd_tile_arg_t args[FFTND_MAX_THREADS];
    int n_dispatch = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++) {
        size_t rs = row_lo + ((n_tiles * (size_t)t) / (size_t)T) * B;
        size_t re_ = row_lo + ((n_tiles * (size_t)(t + 1)) / (size_t)T) * B;
        if (re_ > row_hi) re_ = row_hi;
        if (rs >= row_hi) break;
        args[t] = (_fftnd_tile_arg_t){ d, re, im,
            _fftnd_scratch(d->scratch_re, d->tile_sz, t),
            _fftnd_scratch(d->scratch_im, d->tile_sz, t), rs, re_, is_bwd };
        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _fftnd_tile_trampoline, &args[t]);
        n_dispatch++;
    }
    {
        size_t re0 = row_lo + ((n_tiles * 1) / (size_t)T) * B;
        if (re0 > row_hi) re0 = row_hi;
        _fftnd_tiled_range(d, re, im, d->scratch_re, d->scratch_im,
                           row_lo, re0, is_bwd);
    }
    if (n_dispatch > 0)
        _stride_pool_wait_all();
}

static void _fftnd_tiled_mt(stride_fftnd_data_t *d,
                            double *re, double *im, int is_bwd) {
    _fftnd_tiled_mt_win(d, re, im, 0, d->O[d->rank - 1], is_bwd);
}


/* ═══════════════════════════════════════════════════════════════
 * FUSED TRAILING GROUP -- axes [split .. rank-1] per leading block.
 * Block b = elements [b*K[split-1], (b+1)*K[split-1]); everything
 * below runs while that block is cache-resident (the FFTW
 * rank-split's trailing child, executed per vector-loop index).
 * ═══════════════════════════════════════════════════════════════ */

static void _fftnd_fused_block_range(stride_fftnd_data_t *d,
                                     double *re, double *im,
                                     double *sr, double *si,
                                     size_t b_lo, size_t b_hi,
                                     int is_bwd) {
    const int s = d->split, last = d->rank - 1;
    const size_t Os = d->O[s];
    const size_t Rb = d->O[last] / Os;          /* rows per block */

    for (size_t b = b_lo; b < b_hi; b++) {
        if (!is_bwd) {
            for (int m = s; m < last; m++) {
                size_t per = d->O[m] / Os;      /* this axis's outers in block */
                _fftnd_axis_outer_range(d, m, re, im,
                                        b * per, (b + 1) * per,
                                        0, d->K[m], 0);
            }
            _fftnd_tiled_range(d, re, im, sr, si, b * Rb, (b + 1) * Rb, 0);
        } else {
            _fftnd_tiled_range(d, re, im, sr, si, b * Rb, (b + 1) * Rb, 1);
            for (int m = last - 1; m >= s; m--) {
                size_t per = d->O[m] / Os;
                _fftnd_axis_outer_range(d, m, re, im,
                                        b * per, (b + 1) * per,
                                        0, d->K[m], 1);
            }
        }
    }
}

typedef struct {
    stride_fftnd_data_t *d;
    double *re, *im, *sr, *si;
    size_t b_lo, b_hi;
    int is_bwd;
} _fftnd_block_arg_t;

static void _fftnd_block_trampoline(void *arg) {
    _fftnd_block_arg_t *a = (_fftnd_block_arg_t *)arg;
    _fftnd_fused_block_range(a->d, a->re, a->im, a->sr, a->si,
                             a->b_lo, a->b_hi, a->is_bwd);
}

/* Starved-grain path: fewer fused blocks than threads (O_s < T). Blocks
 * run SEQUENTIALLY -- preserving the cache-residency contract that is the
 * whole point of fusion -- while each block's internal passes run with the
 * full pool via the windowed per-pass MT (pool joined between the block's
 * passes; the joins amortize over the large per-block work that small O_s
 * implies). Per-element op order matches every other mode, so outputs stay
 * bit-identical across T and across modes. */
static void _fftnd_fused_seq_par(stride_fftnd_data_t *d,
                                 double *re, double *im, int is_bwd) {
    const int s = d->split, last = d->rank - 1;
    const size_t NB = d->O[s];
    const size_t Rb = d->O[last] / NB;
    for (size_t b = 0; b < NB; b++) {
        if (!is_bwd) {
            for (int m = s; m < last; m++) {
                size_t per = d->O[m] / NB;
                _fftnd_axis_mt_win(d, m, re, im, b * per, (b + 1) * per, 0);
            }
            _fftnd_tiled_mt_win(d, re, im, b * Rb, (b + 1) * Rb, 0);
        } else {
            _fftnd_tiled_mt_win(d, re, im, b * Rb, (b + 1) * Rb, 1);
            for (int m = last - 1; m >= s; m--) {
                size_t per = d->O[m] / NB;
                _fftnd_axis_mt_win(d, m, re, im, b * per, (b + 1) * per, 1);
            }
        }
    }
}

static void _fftnd_fused_mt(stride_fftnd_data_t *d,
                            double *re, double *im, int is_bwd) {
    const size_t NB = d->O[d->split];
    int T = stride_get_num_threads();
    if (T > d->num_scratch) T = d->num_scratch;

    if (T <= 1 || NB <= 1) {
        if (T > 1 && NB == 1) {         /* single block, many threads */
            _fftnd_fused_seq_par(d, re, im, is_bwd);
            return;
        }
        _fftnd_fused_block_range(d, re, im, d->scratch_re, d->scratch_im,
                                 0, NB, is_bwd);
        return;
    }
    if (NB < (size_t)T) {               /* starved: parallel-within-block */
        _fftnd_fused_seq_par(d, re, im, is_bwd);
        return;
    }
    _fftnd_block_arg_t args[FFTND_MAX_THREADS];
    int n_dispatch = 0;
    for (int t = 1; t < T && t <= _stride_pool_size; t++) {
        size_t lo = (NB * (size_t)t) / (size_t)T;
        size_t hi = (NB * (size_t)(t + 1)) / (size_t)T;
        if (lo >= hi) continue;
        args[t] = (_fftnd_block_arg_t){ d, re, im,
            _fftnd_scratch(d->scratch_re, d->tile_sz, t),
            _fftnd_scratch(d->scratch_im, d->tile_sz, t), lo, hi, is_bwd };
        _stride_pool_dispatch(&_stride_workers[t - 1],
                              _fftnd_block_trampoline, &args[t]);
        n_dispatch++;
    }
    _fftnd_fused_block_range(d, re, im, d->scratch_re, d->scratch_im,
                             0, NB / (size_t)T, is_bwd);
    if (n_dispatch > 0)
        _stride_pool_wait_all();
}


/* ═══════════════════════════════════════════════════════════════
 * DISPATCH
 *   fwd: unfused axes 0..s-1 (ascending), then fused group
 *   bwd: fused group (internally reversed), then axes s-1..0
 * ═══════════════════════════════════════════════════════════════ */

static void _fftnd_execute_fwd(void *data, double *re, double *im) {
    stride_fftnd_data_t *d = (stride_fftnd_data_t *)data;
    for (int m = 0; m < d->split; m++)
        _fftnd_axis_mt(d, m, re, im, 0);
    if (d->split == d->rank - 1)
        _fftnd_tiled_mt(d, re, im, 0);
    else
        _fftnd_fused_mt(d, re, im, 0);
}

static void _fftnd_execute_bwd(void *data, double *re, double *im) {
    stride_fftnd_data_t *d = (stride_fftnd_data_t *)data;
    if (d->split == d->rank - 1)
        _fftnd_tiled_mt(d, re, im, 1);
    else
        _fftnd_fused_mt(d, re, im, 1);
    for (int m = d->split - 1; m >= 0; m--)
        _fftnd_axis_mt(d, m, re, im, 1);
}


/* ═══════════════════════════════════════════════════════════════
 * DESTROY / WRAP / HEURISTICS / BUILDERS
 * ═══════════════════════════════════════════════════════════════ */

static void _fftnd_destroy(void *data) {
    stride_fftnd_data_t *d = (stride_fftnd_data_t *)data;
    if (!d) return;
    for (int m = 0; m < d->rank; m++)
        if (d->plan[m]) stride_plan_destroy(d->plan[m]);
    STRIDE_ALIGNED_FREE(d->scratch_re);
    STRIDE_ALIGNED_FREE(d->scratch_im);
    free(d);
}

static stride_plan_t *_fftnd_wrap(stride_fftnd_data_t *d) {
    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _fftnd_destroy(d); return NULL; }
    d->nat_col_list = NULL;
    _fftnd_jit_resolve(d);
#ifdef VFFT_STRIDED_ROWS
    /* B < VW would undersize the padded-tail staging (tile scratch is
     * N*B per plane); tiny-R cells stay native. */
    if (d->B >= (size_t)_VFFT_STRIDED_VW)
    _vfft_strided_lookup(d->N[d->rank - 1], &d->srow_fwd, &d->srow_bwd);
    if (d->srow_fwd &&
        !_vfft_strided_verify_natural(d->srow_fwd, d->N[d->rank - 1]))
        { d->srow_fwd = 0; d->srow_bwd = 0; }   /* fail-safe -> native */
#endif
    plan->N = (int)d->total;
    plan->K = 1;
    plan->num_stages = 0;
    plan->override_fwd     = _fftnd_execute_fwd;
    plan->override_bwd     = _fftnd_execute_bwd;
    plan->override_destroy = _fftnd_destroy;
    plan->override_data    = d;
    return plan;
}

static void _fftnd_fill_ok(stride_fftnd_data_t *d) {
    d->total = 1;
    for (int m = 0; m < d->rank; m++) d->total *= (size_t)d->N[m];
    for (int m = 0; m < d->rank; m++) {
        size_t O = 1, K = 1;
        for (int i = 0; i < m; i++) O *= (size_t)d->N[i];
        for (int i = m + 1; i < d->rank; i++) K *= (size_t)d->N[i];
        d->O[m] = O; d->K[m] = K;
    }
}

static size_t _fftnd_choose_tile(int NL, size_t n_rows) {
    size_t B = FFTND_DEFAULT_TILE;
    (void)NL;
    if (B > n_rows) B = n_rows;
    if (B < FFTND_MIN_TILE) B = FFTND_MIN_TILE;
    return B;
}

static size_t _fftnd_choose_block(int Nm, size_t Km) {
    size_t sub_bytes = (size_t)16 * (size_t)Nm * Km;
    if (sub_bytes <= FFTND_A_FLAT_MAX_BYTES) return 0;
    size_t C = FFTND_A_BLOCK_BYTES / ((size_t)16 * (size_t)Nm);
    C &= ~(size_t)7;
    if (C < 8) C = 8;
    if (C >= Km) return 0;
    return C;
}

/* Smallest s whose fused block fits the budget (max fusion), else rank-1. */
static int _fftnd_choose_split(const stride_fftnd_data_t *d) {
    for (int s = 1; s < d->rank; s++)
        if ((size_t)16 * d->K[s - 1] <= FFTND_FUSE_MAX_BYTES)
            return s;
    return d->rank - 1;
}

static int _fftnd_alloc_scratch(stride_fftnd_data_t *d, size_t tile_sz) {
    int T = stride_get_num_threads();
    if (T > FFTND_MAX_THREADS) T = FFTND_MAX_THREADS;
    if (T < 1) T = 1;
    d->tile_sz = tile_sz;
    d->num_scratch = T;
    d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)T * tile_sz * sizeof(double));
    d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)T * tile_sz * sizeof(double));
    return (d->scratch_re && d->scratch_im) ? T : 0;
}

/** nd plan from caller-supplied inner plans (wisdom/DP path). plans[m] =
 *  N[m]-point c2c baked at K[m] (last axis at K=B). split: 1..rank-1, or -1
 *  for the fuse heuristic. lane_block[m]: 0 flat, >0 lanes (rounded to 8),
 *  or (size_t)-1 heuristic; ignored for the last axis. TAKES OWNERSHIP. */
static stride_plan_t *stride_plan_nd_from(
        int rank, const int *N, size_t B, int split,
        const size_t *lane_block, stride_plan_t **plans)
{
    int ok = (rank >= 2 && rank <= FFTND_MAX_RANK && N && plans && B >= 1);
    if (ok) for (int m = 0; m < rank; m++)
        if (N[m] < 1 || !plans[m]) ok = 0;
    stride_fftnd_data_t *d = ok ?
        (stride_fftnd_data_t *)calloc(1, sizeof(*d)) : NULL;
    if (!d) {
        if (plans) for (int m = 0; m < (rank > 0 ? rank : 0) &&
                        rank <= FFTND_MAX_RANK; m++)
            if (plans[m]) stride_plan_destroy(plans[m]);
        return NULL;
    }
    d->rank = rank;
    for (int m = 0; m < rank; m++) { d->N[m] = N[m]; d->plan[m] = plans[m]; }
    _fftnd_fill_ok(d);
    if (d->total > (size_t)0x7fffffff) { _fftnd_destroy(d); return NULL; }
    d->B = B;
    d->split = (split >= 1 && split <= rank - 1) ? split : _fftnd_choose_split(d);
    for (int m = 0; m < rank - 1; m++) {
        size_t lb = lane_block ? lane_block[m] : (size_t)-1;
        if (lb == (size_t)-1)
            d->lane_block[m] = _fftnd_choose_block(N[m], d->K[m]);
        else if (lb > 0) {
            size_t C = lb & ~(size_t)7;
            if (C < 8) C = 8;
            d->lane_block[m] = (C >= d->K[m]) ? 0 : C;
        } else
            d->lane_block[m] = 0;
    }
    if (!_fftnd_alloc_scratch(d, (size_t)N[rank - 1] * B)) {
        _fftnd_destroy(d); return NULL;
    }
    return _fftnd_wrap(d);
}

/** Default nd plan -- exhaustive-then-auto inner search (small N only). */
static stride_plan_t *stride_plan_nd(
        int rank, const int *N, const vfft_proto_registry_t *reg)
{
    if (rank < 2 || rank > FFTND_MAX_RANK || !N) return NULL;
    stride_fftnd_data_t tmp; memset(&tmp, 0, sizeof tmp);
    tmp.rank = rank;
    for (int m = 0; m < rank; m++) {
        if (N[m] < 1) return NULL;
        tmp.N[m] = N[m];
    }
    _fftnd_fill_ok(&tmp);
    if (tmp.total > (size_t)0x7fffffff) return NULL;

    size_t B = _fftnd_choose_tile(N[rank - 1], tmp.O[rank - 1]);
    stride_plan_t *plans[FFTND_MAX_RANK] = { 0 };
    for (int m = 0; m < rank; m++) {
        size_t Kp = (m == rank - 1) ? B : tmp.K[m];
        plans[m] = vfft_proto_exhaustive_plan(N[m], Kp, reg, 0);
        if (!plans[m]) plans[m] = vfft_proto_auto_plan_dispatch(N[m], Kp, reg, NULL);
        if (!plans[m]) {
            for (int i = 0; i < m; i++) stride_plan_destroy(plans[i]);
            return NULL;
        }
    }
    return stride_plan_nd_from(rank, N, B, -1, NULL, plans);
}

#endif /* STRIDE_FFTND_H */
