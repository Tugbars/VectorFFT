/**
 * fftnd_r2c.h -- rank-general (2..4D) real-to-complex / complex-to-real,
 * reducing along the LAST axis (FFTW convention). Functional-completeness
 * port of the fft2d_r2c architecture into the fftnd taxonomy; it inherits
 * the split-layout real-FFT tax knowingly (the fused real codelets are a
 * separate, deprioritized workstream).
 *
 *   Forward:  N0 x .. x N_{d-1} reals -> N0 x .. x N_{d-2} x (N_{d-1}/2+1)
 *     Phase 1: tiled R2C row pass over R = prod(N_0..N_{d-2}) rows --
 *              gather real B x Nd, inner stride-r2c (N=Nd, K=B), scatter
 *              half-spectrum into an internal PADDED buffer at row stride
 *              K_pad = roundup8(Nd/2+1). Tile-parallel (per-thread scratch
 *              + distinct inner pack-slot tids).
 *     Phase 2: c2c along axes 0..d-2 on the padded cube -- axis m is a
 *              loop of O_m native calls at K'_m = (prod_{m<i<d-1} N_i) *
 *              K_pad, i.e. every c2c axis here is "middle-shaped" in the
 *              fftnd taxonomy (K' >= K_pad >= 4, always lane-batched;
 *              no tiled pass needed). Outer-parallel.
 *     Phase 3: unpack padded -> user packed (row stride Nd/2+1).
 *
 *   Backward: pack user -> padded; c2c inverse axes d-2..0; tiled C2R row
 *     pass padded -> user real. The 2D in-place reverse-tile-order hazard
 *     does not arise: the c2r gather reads internal padded scratch while
 *     the scatter writes the user buffer (disjoint), same argument as the
 *     2D MT path's note.
 *
 * ORDER CONTRACT (differs from fft2d_r2c, matches fftnd): the c2c axes are
 * emitted SCRAMBLED -- no per-axis unscramble, no perm bookkeeping.
 * Roundtrip is definitive (c2r(r2c(x)) = Ntotal * x), the half axis f is
 * natural, and per-bin addressing on the outer axes goes through the
 * chain-free phase-probe maps (fftnd_natorder.h's technique; see
 * fftnd_r2c_natorder_maps below, which probes THIS plan with real
 * impulses).
 *
 * Constraints: rank 2..FFTND_MAX_RANK, N[last] EVEN (the 1D stride-r2c odd
 * path exists but is not plumbed here), N[m] >= 2. User buffers: re holds
 * Ntotal reals in, R*(Nd/2+1) Re bins out; im holds R*(Nd/2+1).
 */
#ifndef STRIDE_FFTND_R2C_H
#define STRIDE_FFTND_R2C_H

#include <time.h> /* clock_gettime for the adoption A/B timing (win: mingw provides it) */
#include "fftnd.h"                /* taxonomy helpers + include set */
#include "r2c.h"                  /* stride_r2c_plan + worker shims */
#include "il_layout.h"
#include "../fft2d/fft2d_r2c.h"
#include "../natorder/natorder_perm.h" /* mk_cycles for axis naturalization */
#include "../../planning/adopt_wisdom.h"  /* §6a49/Q3 */   /* §6a47/Q1: strided r2c row engines,
                                     resolvers, MT run wrappers */            /* vfft_il2sp/sp2il (interleaved complex out) */
#ifdef VFFT_USE_JIT
#include "jit_runtime.h"          /* baked/JIT resolve: c2c axes + row inner */
#endif

#ifndef FFTND_R2C_MAX_THREADS
#define FFTND_R2C_MAX_THREADS STRIDE_POOL_MAX_DISPATCH /* the pool's bound, not a second one */
#endif

typedef struct {
    int rank;
    int N[FFTND_MAX_RANK];
    size_t R;                     /* rows = prod(N[0..rank-2])            */
    size_t hp1;                   /* N[last]/2 + 1                        */
    size_t K_pad;                 /* roundup4(hp1): padded row stride     */
    size_t B;                     /* row tile height                      */
    size_t total_real;            /* prod(N[])                            */

    size_t Oc[FFTND_MAX_RANK];    /* outer count per c2c axis m (< last)  */
    size_t Kc[FFTND_MAX_RANK];    /* padded inner lanes per c2c axis m    */

    stride_plan_t *plan_r2c;      /* N = N[last], K = B                   */
    stride_plan_t *cplan[FFTND_MAX_RANK];  /* c2c axis m at K = Kc[m]     */
    vfft_proto_exec_fn cjf[FFTND_MAX_RANK]; /* baked/JIT axis executors     */
    vfft_proto_exec_fn cjb[FFTND_MAX_RANK]; /* (NULL -> generic; speed cache,
                                               never a correctness dep)     */

    int num_scratch;
    size_t tile_real_sz;          /* N[last] * B                          */
    size_t tile_cplx_sz;          /* hp1 * B                              */
    double *scratch_re, *scratch_im;
    double *pad_re, *pad_im;      /* R * K_pad each                       */
    /* §6a47/Q1: strided mono row engines for the last-dim pass (family
     * 2/4 via the fft2d resolvers; MT via the _run wrappers, BIT-inv). */
    _f2d_sr2c_fwd_fn snd_fwd;
    _f2d_sr2c_bwd_fn snd_bwd;
    int snd_blk;
    double *snd_tail_scr;
    /* §6a47b: per-axis naturalization (scramble-prone odd/log3 axes). NULL
     * = axis already natural. Detected EMPIRICALLY at build (impulse probe,
     * angle-identified bins, bijection-verified — fail-safe: any anomaly
     * fails the build rather than ship a scrambled spectrum). */
    int *nat_ax[FFTND_MAX_RANK];
    double *nat_rtmp;              /* 2 * max(Kc) doubles */
    int il_out;                   /* 1: user complex side is INTERLEAVED
                                     pairs z[2f],z[2f+1] at packed row
                                     stride hp1 -- the pack/unpack sweeps
                                     already copy every row, so the layout
                                     costs nothing extra (v1.1 P1a). */
} stride_fftnd_r2c_data_t;

static inline double *_fndr_sre(stride_fftnd_r2c_data_t *d, int t) {
    return d->scratch_re + (size_t)t * d->tile_real_sz;
}
static inline double *_fndr_sim(stride_fftnd_r2c_data_t *d, int t) {
    return d->scratch_im + (size_t)t * d->tile_cplx_sz;
}

/* ST inner shims on a caller thread; tid selects the inner's pack slot
 * (fft2d_r2c precedent -- distinct per concurrent tile thread). */
static inline void _fndr_inner_fwd(stride_plan_t *p, double *re, double *im, int tid) {
    stride_r2c_data_t *rd = (stride_r2c_data_t *)p->override_data;
    _r2c_worker_arg_t a = { rd, re, im, 0, rd->K, tid };
    _r2c_worker_fwd(&a);
}
static inline void _fndr_inner_bwd(stride_plan_t *p, double *re, double *im, int tid) {
    stride_r2c_data_t *rd = (stride_r2c_data_t *)p->override_data;
    _r2c_worker_arg_t a = { rd, re, im, 0, rd->K, tid };
    _r2c_worker_bwd(&a);
}


/* ═══════════════════════════════════════════════════════════════
 * PHASE 1 / 3-of-bwd — tiled real row passes (fft2d_r2c workers with
 * the row count generalized to R)
 * ═══════════════════════════════════════════════════════════════ */

static void _fndr_rows_fwd_range(stride_fftnd_r2c_data_t *d,
                                 const double *re_in,
                                 double *sr, double *si,
                                 size_t row_start, size_t row_end, int tid) {
    const int NL = d->N[d->rank - 1];
    const size_t hp1 = d->hp1, B = d->B, K_pad = d->K_pad;
    for (size_t i = row_start; i < row_end; i += B) {
        size_t this_B = B;
        if (i + B > row_end) this_B = row_end - i;
        stride_transpose(re_in + i * (size_t)NL, (size_t)NL,
                         sr, B, this_B, (size_t)NL);
        _fndr_inner_fwd(d->plan_r2c, sr, si, tid);
        stride_transpose_pair(sr, si,
                              d->pad_re + i * K_pad, d->pad_im + i * K_pad,
                              B, K_pad, hp1, this_B);
        for (size_t r = 0; r < this_B; r++) {          /* zero pad cols */
            double *rr = d->pad_re + (i + r) * K_pad;
            double *ii = d->pad_im + (i + r) * K_pad;
            for (size_t f = hp1; f < K_pad; f++) { rr[f] = 0.0; ii[f] = 0.0; }
        }
    }
}

static void _fndr_rows_bwd_range(stride_fftnd_r2c_data_t *d,
                                 double *re_out,
                                 double *sr, double *si,
                                 size_t row_start, size_t row_end, int tid) {
    const int NL = d->N[d->rank - 1];
    const size_t hp1 = d->hp1, B = d->B, K_pad = d->K_pad;
    for (size_t i = row_start; i < row_end; i += B) {
        size_t this_B = B;
        if (i + B > row_end) this_B = row_end - i;
        stride_transpose_pair(d->pad_re + i * K_pad, d->pad_im + i * K_pad,
                              sr, si, K_pad, B, this_B, hp1);
        _fndr_inner_bwd(d->plan_r2c, sr, si, tid);
        stride_transpose(sr, B, re_out + i * (size_t)NL, (size_t)NL,
                         (size_t)NL, this_B);
    }
}

typedef struct {
    stride_fftnd_r2c_data_t *d;
    const double *re_in;
    double *re_out;
    double *sr, *si;
    size_t row_start, row_end;
    int tid, is_bwd;
} _fndr_tile_arg_t;

static void _fndr_tile_tramp(void *arg) {
    _fndr_tile_arg_t *a = (_fndr_tile_arg_t *)arg;
    if (a->is_bwd)
        _fndr_rows_bwd_range(a->d, a->re_out, a->sr, a->si,
                             a->row_start, a->row_end, a->tid);
    else
        _fndr_rows_fwd_range(a->d, a->re_in, a->sr, a->si,
                             a->row_start, a->row_end, a->tid);
}

static void _fndr_rows_mt(stride_fftnd_r2c_data_t *d,
                          const double *re_in, double *re_out, int is_bwd) {
    if (!is_bwd && d->snd_fwd) {
        _f2d_sr2c_fwd_rows(d->snd_fwd, d->snd_blk, d->N[d->rank - 1], re_in,
                           d->pad_re, d->pad_im, (size_t)d->N[d->rank - 1],
                           d->K_pad, d->R, d->snd_tail_scr);
        for (size_t i = 0; i < d->R; i++)
            for (size_t f = d->hp1; f < d->K_pad; f++) {
                d->pad_re[i * d->K_pad + f] = 0.0;
                d->pad_im[i * d->K_pad + f] = 0.0;
            }
        return;
    }
    if (is_bwd && d->snd_bwd) {
        _f2d_sr2c_bwd_rows(d->snd_bwd, d->snd_blk, d->N[d->rank - 1],
                           d->pad_re, d->pad_im, re_out, d->K_pad,
                           (size_t)d->N[d->rank - 1], d->R,
                           d->snd_tail_scr);
        return;
    }
    const size_t R = d->R, B = d->B;
    /* the plan's snapshot = min(its tile-scratch slots, the inner r2c plan's
     * pack-slot count); the pool's one clamp takes it */
    int slots = d->num_scratch;
    {
        stride_r2c_data_t *rd = (stride_r2c_data_t *)d->plan_r2c->override_data;
        if (slots > rd->n_threads) slots = rd->n_threads;
    }
    int T = stride_pool_workers_for(slots);
    size_t n_tiles = (R + B - 1) / B;
    if (T <= 1 || n_tiles <= 1) {
        if (is_bwd) _fndr_rows_bwd_range(d, re_out, _fndr_sre(d,0), _fndr_sim(d,0), 0, R, 0);
        else        _fndr_rows_fwd_range(d, re_in,  _fndr_sre(d,0), _fndr_sim(d,0), 0, R, 0);
        return;
    }
    /* slot t owns scratch slot t and tid t; slot 0 is the caller */
    _fndr_tile_arg_t args[STRIDE_POOL_MAX_DISPATCH];
    int n = 0;
    for (int t = 0; t < T; t++) {
        size_t rs = ((n_tiles * (size_t)t) / (size_t)T) * B;
        size_t re_ = ((n_tiles * (size_t)(t + 1)) / (size_t)T) * B;
        if (re_ > R) re_ = R;
        if (t > 0 && rs >= R) break;
        args[n++] = (_fndr_tile_arg_t){ d, re_in, re_out,
                                        _fndr_sre(d,t), _fndr_sim(d,t),
                                        rs, re_, t, is_bwd };
    }
    stride_pool_run(n, _fndr_tile_tramp, args, sizeof args[0]);
}


/* ═══════════════════════════════════════════════════════════════
 * PHASE 2 — c2c axes on the padded cube (outer-parallel loops of
 * native full-K' calls; K' >= K_pad >= 4 so every axis is
 * lane-batched natively, DIT/DIF/override all via the proto path)
 * ═══════════════════════════════════════════════════════════════ */

static void _fndr_axis_range(stride_fftnd_r2c_data_t *d, int m,
                             size_t o_lo, size_t o_hi, int is_bwd) {
    const size_t Kc = d->Kc[m];
    const size_t sub = (size_t)d->N[m] * Kc;
    for (size_t o = o_lo; o < o_hi; o++) {
        double *br = d->pad_re + o * sub;
        double *bi = d->pad_im + o * sub;
        vfft_proto_exec_fn jf = is_bwd ? d->cjb[m] : d->cjf[m];
        if (jf)          jf(d->cplan[m], br, bi, Kc, Kc, 0);
        else if (is_bwd) vfft_proto_execute_bwd(d->cplan[m], br, bi, Kc);
        else             vfft_proto_execute_fwd(d->cplan[m], br, bi, Kc);
    }
}

typedef struct {
    stride_fftnd_r2c_data_t *d;
    int m, is_bwd;
    size_t o_lo, o_hi;
} _fndr_axis_arg_t;

static void _fndr_axis_tramp(void *arg) {
    _fndr_axis_arg_t *a = (_fndr_axis_arg_t *)arg;
    _fndr_axis_range(a->d, a->m, a->o_lo, a->o_hi, a->is_bwd);
}

static void _fndr_axis_mt(stride_fftnd_r2c_data_t *d, int m, int is_bwd) {
    const size_t O = d->Oc[m];
    if (is_bwd && d->nat_ax[m]) {
        const size_t Kc = d->Kc[m], sub = (size_t)d->N[m] * Kc;
        for (size_t o = 0; o < O; o++)
            vfft_natorder_cycle_pass_inv(d->pad_re + o * sub,
                                         d->pad_im + o * sub,
                                         Kc, d->nat_ax[m], d->nat_rtmp);
    }
    int T = stride_pool_workers_for(0); /* the pool's one clamp; no per-slot scratch here */
    if (T <= 1 || O <= 1) {
        _fndr_axis_range(d, m, 0, O, is_bwd);
        if (!is_bwd && d->nat_ax[m]) {
            const size_t Kc = d->Kc[m], sub = (size_t)d->N[m] * Kc;
            for (size_t o = 0; o < O; o++)
                vfft_natorder_cycle_pass(d->pad_re + o * sub,
                                         d->pad_im + o * sub,
                                         Kc, d->nat_ax[m], d->nat_rtmp);
        }
        return;
    }
    /* slot 0 is the caller's [0, O/T); empty worker ranges skipped (packed) */
    _fndr_axis_arg_t args[STRIDE_POOL_MAX_DISPATCH];
    int n = 0;
    args[n++] = (_fndr_axis_arg_t){ d, m, is_bwd, 0, O / (size_t)T };
    for (int t = 1; t < T; t++) {
        size_t lo = (O * (size_t)t) / (size_t)T;
        size_t hi = (O * (size_t)(t + 1)) / (size_t)T;
        if (lo >= hi) continue;
        args[n++] = (_fndr_axis_arg_t){ d, m, is_bwd, lo, hi };
    }
    stride_pool_run(n, _fndr_axis_tramp, args, sizeof args[0]);
    if (!is_bwd && d->nat_ax[m]) {
        const size_t Kc = d->Kc[m], sub = (size_t)d->N[m] * Kc;
        for (size_t o = 0; o < O; o++)
            vfft_natorder_cycle_pass(d->pad_re + o * sub,
                                     d->pad_im + o * sub,
                                     Kc, d->nat_ax[m], d->nat_rtmp);
    }
}

/* pack (user packed -> pad, zeroing pad cols) / unpack (pad -> user) */
static void _fndr_pack(stride_fftnd_r2c_data_t *d,
                       const double *ure, const double *uim) {
    for (size_t i = 0; i < d->R; i++) {
        if (d->il_out)
            vfft_il2sp(ure + i * 2 * d->hp1,
                       d->pad_re + i * d->K_pad, d->pad_im + i * d->K_pad,
                       d->hp1);
        else {
            memcpy(d->pad_re + i * d->K_pad, ure + i * d->hp1, d->hp1 * 8);
            memcpy(d->pad_im + i * d->K_pad, uim + i * d->hp1, d->hp1 * 8);
        }
        for (size_t f = d->hp1; f < d->K_pad; f++) {
            d->pad_re[i * d->K_pad + f] = 0.0;
            d->pad_im[i * d->K_pad + f] = 0.0;
        }
    }
}
static void _fndr_unpack(stride_fftnd_r2c_data_t *d,
                         double *ure, double *uim) {
    if (d->il_out) {              /* ure = interleaved z; uim unused */
        for (size_t i = 0; i < d->R; i++)
            vfft_sp2il(d->pad_re + i * d->K_pad, d->pad_im + i * d->K_pad,
                       ure + i * 2 * d->hp1, d->hp1);
        return;
    }
    for (size_t i = 0; i < d->R; i++) {
        memcpy(ure + i * d->hp1, d->pad_re + i * d->K_pad, d->hp1 * 8);
        memcpy(uim + i * d->hp1, d->pad_im + i * d->K_pad, d->hp1 * 8);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * EXECUTE / DESTROY / BUILD
 * ═══════════════════════════════════════════════════════════════ */

/* ── THE ND real walk, owned here and nowhere else (2026-09-02: the
 * driver used to hand-inline this sequence and the two copies diverged in
 * backward axis order — the driver walked axes FORWARD, this file walked
 * them in reverse. Separable per-axis inverses commute mathematically but
 * NOT bitwise (axis order changes rounding), so the DRIVER's live order is
 * canonical: it is what has always shipped. ndreal_bits_probe verified the
 * unification byte-identical on 3D and 4D r2c+c2r cells.) */

/* fwd, OOP: `in` holds Ntotal reals -> (out_re, out_im) packed bins. */
static void _fndr_execute_fwd_oop(stride_fftnd_r2c_data_t *d,
                                  double *in,
                                  double *out_re, double *out_im) {
    _fndr_rows_mt(d, in, NULL, 0);                     /* real -> pad   */
    for (int m = 0; m < d->rank - 1; m++)
        _fndr_axis_mt(d, m, 0);                        /* c2c axes      */
    _fndr_unpack(d, out_re, out_im);                   /* pad -> packed */
}

/* bwd, OOP: (in_re, in_im) packed bins -> `out` holds Ntotal reals.
 * Axis order: FORWARD — the live order (see the banner above). */
static void _fndr_execute_bwd_oop(stride_fftnd_r2c_data_t *d,
                                  double *in_re, double *in_im,
                                  double *out) {
    _fndr_pack(d, in_re, in_im);                       /* packed -> pad */
    for (int m = 0; m < d->rank - 1; m++)
        _fndr_axis_mt(d, m, 1);                        /* c2c inverse   */
    _fndr_rows_mt(d, NULL, out, 1);                    /* pad -> real   */
}

/* the registered single-buffer ABI (override_fwd/bwd): thin wrappers. */
static void _fndr_execute_fwd(void *data, double *re, double *im) {
    stride_fftnd_r2c_data_t *d = (stride_fftnd_r2c_data_t *)data;
    _fndr_execute_fwd_oop(d, re, re, im);
}
static void _fndr_execute_bwd(void *data, double *re, double *im) {
    stride_fftnd_r2c_data_t *d = (stride_fftnd_r2c_data_t *)data;
    _fndr_execute_bwd_oop(d, re, im, re);
}

static void _fndr_destroy(void *data) {
    stride_fftnd_r2c_data_t *d = (stride_fftnd_r2c_data_t *)data;
    if (!d) return;
    if (d->plan_r2c) stride_plan_destroy(d->plan_r2c);
    for (int m = 0; m < d->rank - 1; m++)
        if (d->cplan[m]) stride_plan_destroy(d->cplan[m]);
    STRIDE_ALIGNED_FREE(d->scratch_re);
    STRIDE_ALIGNED_FREE(d->scratch_im);
    STRIDE_ALIGNED_FREE(d->pad_re);
    STRIDE_ALIGNED_FREE(d->pad_im);
    for (int m_ = 0; m_ < FFTND_MAX_RANK; m_++) free(d->nat_ax[m_]);
    STRIDE_ALIGNED_FREE(d->nat_rtmp);
    STRIDE_ALIGNED_FREE(d->snd_tail_scr);
    free(d);
}

/** As stride_plan_nd_r2c but the COMPLEX side is interleaved pairs: fwd
 *  leaves z (= the re buffer, 2*R*hp1 doubles) with (re,im) pairs at packed
 *  row stride hp1; bwd consumes the same; im param unused on the complex
 *  side. Same cost as split (the boundary copies carry the layout). */
static stride_plan_t *stride_plan_nd_r2c_il(int rank, const int *N,
                                            const vfft_proto_registry_t *reg);

/** Rank-general r2c/c2r plan, auto inners. N[rank-1] must be even. */
static stride_plan_t *stride_plan_nd_r2c(int rank, const int *N,
                                         const vfft_proto_registry_t *reg) {
    if (rank < 2 || rank > FFTND_MAX_RANK || !N) return NULL;
    for (int m = 0; m < rank; m++) if (N[m] < 2) return NULL;
    if (N[rank - 1] & 1) return NULL;

    stride_fftnd_r2c_data_t *d =
        (stride_fftnd_r2c_data_t *)calloc(1, sizeof(*d));
    if (!d) return NULL;
    d->rank = rank;
    d->total_real = 1;
    for (int m = 0; m < rank; m++) { d->N[m] = N[m]; d->total_real *= (size_t)N[m]; }
    if (d->total_real > (size_t)0x7fffffff) { free(d); return NULL; }
    d->R = d->total_real / (size_t)N[rank - 1];
    d->hp1 = (size_t)(N[rank - 1] / 2 + 1);
    d->K_pad = ((d->hp1 + 7) / 8) * 8;  /* §6a54: pad-to-8 — every axis Kc becomes a multiple of 8 (products include K_pad), all axis passes full-width */
    d->B = 8;
    if (d->B > d->R) d->B = d->R;
    if (d->B < 2) d->B = 2;

    for (int m = 0; m < rank - 1; m++) {
        size_t O = 1, Kc = d->K_pad;
        for (int i = 0; i < m; i++) O *= (size_t)N[i];
        for (int i = m + 1; i < rank - 1; i++) Kc *= (size_t)N[i];
        d->Oc[m] = O;
        d->Kc[m] = Kc;
    }

    stride_plan_t *inner = vfft_proto_auto_plan_dispatch(N[rank-1] / 2, d->B, reg, NULL);
    d->plan_r2c = inner ? stride_r2c_plan(N[rank-1], d->B, d->B, inner) : NULL;
    int ok = (d->plan_r2c != NULL);
    for (int m = 0; ok && m < rank - 1; m++) {
        d->cplan[m] = vfft_proto_auto_plan_dispatch(N[m], d->Kc[m], reg, NULL);
        if (!d->cplan[m]) ok = 0;
    }
    if (ok) {
        int T = stride_pool_workers_for(0); /* create time: the pool as it is now = this plan's slot count */
        d->num_scratch = T;
        d->tile_real_sz = (size_t)N[rank-1] * d->B;
        d->tile_cplx_sz = d->hp1 * d->B;
        d->scratch_re = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)T * d->tile_real_sz * 8);
        d->scratch_im = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)T * d->tile_cplx_sz * 8);
        d->pad_re = (double *)STRIDE_ALIGNED_ALLOC(64, d->R * d->K_pad * 8);
        d->pad_im = (double *)STRIDE_ALIGNED_ALLOC(64, d->R * d->K_pad * 8);
        ok = d->scratch_re && d->scratch_im && d->pad_re && d->pad_im;
    }
    if (!ok) { _fndr_destroy(d); return NULL; }

#ifdef VFFT_USE_JIT
    for (int m = 0; m < rank - 1; m++) {
        d->cjf[m] = vfft_proto_plan_jit_fwd(d->cplan[m]);
        d->cjb[m] = vfft_proto_plan_jit_bwd(d->cplan[m]);
    }
    {   /* row r2c: JIT the inner sliced c2c stages (fft2d_r2c precedent) */
        stride_plan_t *rin = stride_r2c_inner_plan(d->plan_r2c);
        if (rin) {
            stride_r2c_set_inner_jit_fwd(d->plan_r2c, vfft_proto_plan_jit_fwd(rin));
            stride_r2c_set_inner_jit_bwd(d->plan_r2c, vfft_proto_plan_jit_bwd(rin));
        }
    }
#endif
    /* §6a47b: empirical axis-order detection. Impulse at row 1, run the
     * axis fwd once, identify each output row's natural bin from its unit-
     * circle angle, verify bijection. Identity => natural (no list). */
    {
        size_t maxKc = 0;
        for (int m = 0; m < rank - 1; m++)
            if (d->Kc[m] > maxKc) maxKc = d->Kc[m];
        int det_fail = 0;
        for (int m = 0; m < rank - 1 && !det_fail; m++) {
            const int Nm = d->N[m];
            const size_t Kc = d->Kc[m];
            double *pr_ = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)Nm * Kc * 8);
            double *pi_ = (double *)STRIDE_ALIGNED_ALLOC(64, (size_t)Nm * Kc * 8);
            int *M_ = (int *)malloc((size_t)Nm * sizeof(int));
            char *seen_ = (char *)calloc((size_t)Nm, 1);
            if (!pr_ || !pi_ || !M_ || !seen_) { det_fail = 1; }
            else {
                memset(pr_, 0, (size_t)Nm * Kc * 8);
                memset(pi_, 0, (size_t)Nm * Kc * 8);
                pr_[1 * Kc] = 1.0;
                vfft_proto_exec_fn jf_ = d->cjf[m];
                if (jf_) jf_(d->cplan[m], pr_, pi_, Kc, Kc, 0);
                else     vfft_proto_execute_fwd(d->cplan[m], pr_, pi_, Kc);
                int ident = 1;
                for (int r = 0; r < Nm && !det_fail; r++) {
                    double vr = pr_[(size_t)r * Kc], vi = pi_[(size_t)r * Kc];
                    double mag = vr * vr + vi * vi;
                    if (mag < 0.999 || mag > 1.001) { det_fail = 1; break; }
                    double ang = atan2(vi, vr);
                    long kk = llround(-ang * (double)Nm /
                                      (2.0 * 3.14159265358979323846));
                    int k = (int)(((kk % Nm) + Nm) % Nm);
                    if (seen_[k]) { det_fail = 1; break; }
                    seen_[k] = 1;
                    M_[k] = r;
                    if (k != r) ident = 0;
                }
                if (!det_fail && !ident)
                    d->nat_ax[m] = vfft_natorder_mk_cycles(Nm, M_);
            }
            STRIDE_ALIGNED_FREE(pr_); STRIDE_ALIGNED_FREE(pi_); free(M_); free(seen_);
        }
        if (!det_fail && maxKc) {
            d->nat_rtmp = (double *)STRIDE_ALIGNED_ALLOC(64, 2 * maxKc * 8);
            if (!d->nat_rtmp) det_fail = 1;
        }
        if (det_fail) { _fndr_destroy(d); return NULL; }
    }

    /* §6a47/Q1: measured adoption of the strided row engines (last dim).
     * Arms toggle snd_fwd/snd_bwd and call the SAME _fndr_rows_mt entry —
     * strided arm MT-faithful via the wrappers, tiled arm the production
     * path. Hysteresis >5%. Eligibility: R %% 8 == 0 and last-dim coverage
     * (pairs-aware resolve — avx512 editions need pairs %% 8). */
    if (d->R >= 8) {
        const int NL_ = d->N[d->rank - 1];
        _f2d_sr2c_fwd_fn sf_ = _f2d_sr2c_fwd_resolve(NL_, &d->snd_blk);
        _f2d_sr2c_bwd_fn sb_ = _f2d_sr2c_bwd_resolve(NL_, &d->snd_blk);
        if ((sf_ || sb_) && (d->R % (2 * (size_t)d->snd_blk) != 0)) {
            d->snd_tail_scr = (double *)STRIDE_ALIGNED_ALLOC(64,
                (2 * (size_t)d->snd_blk
                     * ((size_t)NL_ + 2 * d->hp1)) * sizeof(double));
            if (!d->snd_tail_scr) { sf_ = 0; sb_ = 0; }
        }
        double *xin_ = (sf_ || sb_)
            ? (double *)STRIDE_ALIGNED_ALLOC(64, d->total_real * sizeof(double))
            : NULL;
        if (xin_) {
            int awf_ = 0, awb_ = 0;
            if (vfft_adopt_lookup("nd", (int)d->R, NL_, d->snd_blk,
                                  &awf_, &awb_)) {
                d->snd_fwd = (awf_ && sf_) ? sf_ : 0;
                d->snd_bwd = (awb_ && sb_) ? sb_ : 0;
                STRIDE_ALIGNED_FREE(xin_);
                goto awnd_done;
            }
            for (size_t ii = 0; ii < d->total_real; ii++)
                xin_[ii] = 1.0 + 1e-3 * (double)(ii & 63);
            struct timespec t0_, t1_;
            double t_a, t_b;
            if (sf_) {
                d->snd_fwd = 0;
                _fndr_rows_mt(d, xin_, NULL, 0);
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr = 0; rr < 8; rr++) _fndr_rows_mt(d, xin_, NULL, 0);
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_a = (t1_.tv_sec - t0_.tv_sec) * 1e9
                    + (double)(t1_.tv_nsec - t0_.tv_nsec);
                d->snd_fwd = sf_;
                _fndr_rows_mt(d, xin_, NULL, 0);
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr = 0; rr < 8; rr++) _fndr_rows_mt(d, xin_, NULL, 0);
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_b = (t1_.tv_sec - t0_.tv_sec) * 1e9
                    + (double)(t1_.tv_nsec - t0_.tv_nsec);
                d->snd_fwd = (t_b * 20 < t_a * 19) ? sf_ : 0;
            }
            if (sb_) {
                d->snd_bwd = 0;
                _fndr_rows_mt(d, NULL, xin_, 1);
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr = 0; rr < 8; rr++) _fndr_rows_mt(d, NULL, xin_, 1);
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_a = (t1_.tv_sec - t0_.tv_sec) * 1e9
                    + (double)(t1_.tv_nsec - t0_.tv_nsec);
                d->snd_bwd = sb_;
                _fndr_rows_mt(d, NULL, xin_, 1);
                clock_gettime(CLOCK_MONOTONIC, &t0_);
                for (int rr = 0; rr < 8; rr++) _fndr_rows_mt(d, NULL, xin_, 1);
                clock_gettime(CLOCK_MONOTONIC, &t1_);
                t_b = (t1_.tv_sec - t0_.tv_sec) * 1e9
                    + (double)(t1_.tv_nsec - t0_.tv_nsec);
                d->snd_bwd = (t_b * 20 < t_a * 19) ? sb_ : 0;
            }
            vfft_adopt_record("nd", (int)d->R, NL_, d->snd_blk,
                              d->snd_fwd ? 1 : 0, d->snd_bwd ? 1 : 0);
            STRIDE_ALIGNED_FREE(xin_);
awnd_done:;
        }
    }

    stride_plan_t *plan = (stride_plan_t *)calloc(1, sizeof(stride_plan_t));
    if (!plan) { _fndr_destroy(d); return NULL; }
    plan->N = (int)d->total_real;
    plan->K = 1;
    plan->override_fwd     = _fndr_execute_fwd;
    plan->override_bwd     = _fndr_execute_bwd;
    plan->override_destroy = _fndr_destroy;
    plan->override_data    = d;
    return plan;
}

static stride_plan_t *stride_plan_nd_r2c_il(int rank, const int *N,
                                            const vfft_proto_registry_t *reg) {
    stride_plan_t *p = stride_plan_nd_r2c(rank, N, reg);
    if (p) ((stride_fftnd_r2c_data_t *)p->override_data)->il_out = 1;
    return p;
}

#endif /* STRIDE_FFTND_R2C_H */
