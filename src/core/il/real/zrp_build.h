/* zrp_build.h - the real pair's, ZTT-r's and the real mono's handles and the
 * real door's engine race.
 *
 * THE DOOR (il/real/real_create_il.h) serves an even-N, K=1, interleaved
 * real request with one of four engines:
 *   zr2c  x read as z[N/2] -> a c2c(N/2) child -> the Hermitian fold pass
 *         (zr2c_build.h; it races its own child route and banks it)
 *   zrp   the real pair (zrp.h): form A = the stock n1t leaf over the packed
 *         view + t2h, form B = the real leaf r2z + t2m; no fold pass. The
 *         pair and the form are PLAN INPUT.
 *   zttr  ZTT-r (zttr.h): the ZTT at N/2 with the fold fused into the
 *         terminator (r2c) or the ingest (c2r); no fold pass, no scratch
 *         out of place. Its chain, tile and stack state are PLAN INPUT,
 *         found by the door's sweep: every chain with {4,8} ends and
 *         {4,8,3,5,7,9,15} mids (the ZTT's odd band: 2^a*odd cells run
 *         staged) at every tile width burst-timed, the four fastest (and
 *         the winner's other stack states) join the race.
 *   zfsr  the real four-step (zfsr.h): above ZTT-r's band (N >= 2^20) the
 *         c2c four-step at N/2 with the fold fused into its order sweep;
 *         the split is PLAN INPUT, swept at create.
 *   zrm   the real mono (zrm.h): the whole transform as one rn1 kernel at
 *         N <= 64; no plan input. Odd N races it in bridge/real_bridge.h
 *         against the odd-real routes (the door is even-N).
 *   zrf   the real flat DIT (zrf.h): odd N on the c2c flat DIT's stages
 *         behind a real leaf; the chain, the split-body switch and the
 *         tile budget are PLAN INPUT, swept by the odd race in
 *         bridge/real_bridge.h.
 * The cell's engine is read from the real shard (wisdom2_real_il.h); a miss
 * races zr2c against every legal pair in every form and the ZTT-r shortlist
 * through the finished handles, gates each arm's output against zr2c's
 * before timing, and banks the winner (3% hysteresis toward zr2c, the
 * incumbent). VFFT_ZRP=R1.R2[.f] pins a pair (f = 0 form A, 1 form B;
 * default A), VFFT_ZRP=0 pins zr2c, VFFT_ZTTR=chain/tile[/stk] (4.8.8.4/512/3)
 * [/mt] pins ZTT-r (mt = its threaded arm on a threaded plan), VFFT_ZRM=1 pins the real mono and VFFT_ZRM=0 keeps it out of
 * the race, VFFT_ZFSR=N1xN2 pins the real four-step and VFFT_ZFSR=0 keeps it
 * out; env beats wisdom and never banks.
 *
 * INCLUSION CONTRACT: after zr2c_build.h and _vw2_persist (vfft.c), the
 * kind-5 precedent.
 */
#ifndef VFFT_TRANSFORMS_REAL_ZRP_BUILD_H
#define VFFT_TRANSFORMS_REAL_ZRP_BUILD_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "vfft_internal.h"
#include "zrp.h"
#include "zttr.h"
#include "zrm.h"
#include "zfsr.h"
#include "zrf.h"
#include "wisdom2_real_il.h"
#include "common/support/race.h"

static struct vfft_plan_s *_zrp_build_pair(const vfft_config_t *cfg, int N,
                                           int R1, int R2, int form)
{
    vfft_zrp_plan_t *zp = vfft_zrp_create(N, R1, R2, form, cfg->placement == VFFT_INPLACE);
    struct vfft_plan_s *h;
    if (!zp)
        return NULL;
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_zrp_destroy(zp);
        return NULL;
    }
    h->transform = cfg->transform;
    h->placement = cfg->placement;
    h->layout = (int)VFFT_LAYOUT_INTERLEAVED;
    h->N = N;
    h->K = 1;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zrp = zp;
    return h;
}

static void _exec_zrp(struct vfft_plan_s *h, const double *sre, double *dre)
{
    if (h->transform == VFFT_R2C)
        vfft_zrp_execute_fwd(h->zrp, sre, dre);
    else
        vfft_zrp_execute_bwd(h->zrp, sre, dre);
}

/* ZTT-r's handle: the chain, tile and stack state are plan input; an
 * in-place placement owns a scratch plane */
static struct vfft_plan_s *_zttr_build_plan(const vfft_config_t *cfg, int N,
                                            const int *chain, int nf, size_t tile, int stk)
{
    vfft_zttr_plan_t *zp = vfft_zttr_create(N, chain, nf, tile);
    struct vfft_plan_s *h;
    if (!zp)
        return NULL;
    zp->stk = stk & 3;
    if (cfg->placement == VFFT_INPLACE && !vfft_zttr_set_inplace(zp))
    {
        vfft_zttr_destroy(zp);
        return NULL;
    }
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_zttr_destroy(zp);
        return NULL;
    }
    h->transform = cfg->transform;
    h->placement = cfg->placement;
    h->layout = (int)VFFT_LAYOUT_INTERLEAVED;
    h->N = N;
    h->K = 1;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zttr = zp;
    return h;
}

/* keyed on the POINTERS, as _exec_zr2c is: an aliased call takes the
 * scratch pipeline (in-place plans own one), a distinct pair the plane in
 * the destination */
static void _exec_zttr(struct vfft_plan_s *h, const double *sre, double *dre)
{
    const int aliased = (sre == dre) && h->zttr->scratch;
    if (h->zttr->mt > 0 && h->nthreads > 1)
    {   /* the threaded arm the race bound (zttr_mt.h); it declines on a clamped pool */
        _vfft_pool_arm(h->nthreads);
        if (vfft_zttr_execute_mt(h->zttr, sre, aliased ? h->zttr->scratch : dre, dre, h->transform != VFFT_R2C))
            return;
    }
    if (h->transform == VFFT_R2C)
    {
        if (aliased) vfft_zttr_execute_fwd_ip(h->zttr, dre);
        else vfft_zttr_execute_fwd(h->zttr, sre, dre);
    }
    else
    {
        if (aliased) vfft_zttr_execute_bwd_ip(h->zttr, dre);
        else vfft_zttr_execute_bwd(h->zttr, sre, dre);
    }
}

/* the real mono's handle: the kernel IS the plan (nothing owned) */
static struct vfft_plan_s *_zrm_build_plan(const vfft_config_t *cfg, int N)
{
    vfft_oop11_fn fn = vfft_zrm_fn(N, cfg->transform == VFFT_C2R);
    struct vfft_plan_s *h;
    if (!fn)
        return NULL;
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
        return NULL;
    h->transform = cfg->transform;
    h->placement = cfg->placement;
    h->layout = (int)VFFT_LAYOUT_INTERLEAVED;
    h->N = N;
    h->K = 1;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zrm = fn;
    return h;
}

/* one call, in place or out (the kind is alias-tolerant) */
static void _exec_zrm(struct vfft_plan_s *h, const double *sre, double *dre)
{
    vfft_zrm_execute(h->zrm, sre, dre);
}

/* VFFT_ZRM at create: 1 = the mono pinned, 0 = kept out of the race, -1 = unset */
static int _zrm_env(void)
{
    const char *e = getenv("VFFT_ZRM");
    if (!e || !e[0])
        return -1;
    return e[0] == '0' ? 0 : 1;
}

/* the real flat DIT's handle: the chain, the split-body switch and the tile budget are plan input */
static struct vfft_plan_s *_zrf_build_plan(const vfft_config_t *cfg, int N, const int *R, int K, int nomsz, int tile)
{
    vfft_zrf_plan_t *zp = vfft_zrf_create(N, R, K, nomsz, tile);
    struct vfft_plan_s *h;
    if (!zp)
        return NULL;
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_zrf_destroy(zp);
        return NULL;
    }
    h->transform = cfg->transform;
    h->placement = cfg->placement;
    h->layout = (int)VFFT_LAYOUT_INTERLEAVED;
    h->N = N;
    h->K = 1;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zrf = zp;
    return h;
}

/* both placements are the same pipeline (the planes are the plan's own) */
static void _exec_zrf(struct vfft_plan_s *h, const double *sre, double *dre)
{
    if (h->zrf->mt > 0 && h->nthreads > 1)
    {   /* the threaded form the race bound (zrf_mt.h); it declines on a clamped pool */
        _vfft_pool_arm(h->nthreads);
        if (vfft_zrf_execute_mt(h->zrf, sre, dre, h->transform != VFFT_R2C))
            return;
    }
    if (h->transform == VFFT_R2C)
        vfft_zrf_execute_fwd(h->zrf, sre, dre);
    else
        vfft_zrf_execute_bwd(h->zrf, sre, dre);
}

/* VFFT_ZRF at create: 1 = pinned at the chain "9.9.5" (then "/t" = the split
 * body off, "/w256" = the tile budget, "/m1" | "/m2" ("/m" = 1) = the
 * threaded arm at the plan's T), 0 = kept out of the race, -1 = unset */
static int _zrf_env(int *R, int *K, int *nomsz, int *tile, int *mt)
{
    const char *e = getenv("VFFT_ZRF");
    int n = 0;
    *K = 0; *nomsz = 0; *tile = 0; *mt = 0;
    if (!e || !e[0])
        return -1;
    while (*e && n < VFFT_ILFD_MAX_K)
    {
        char *end;
        const long v = strtol(e, &end, 10);
        if (end == e || v < 3)
            return 0;
        R[n++] = (int)v;
        e = end;
        if (*e == '.') e++;
        else break;
    }
    if (n < 2)
        return 0;
    if (!strncmp(e, "/t", 2)) { *nomsz = 1; e += 2; }
    if (!strncmp(e, "/w", 2)) { *tile = atoi(e + 2); e += 2; while (*e >= '0' && *e <= '9') e++; }
    if (!strncmp(e, "/m", 2)) { e += 2; *mt = (*e == '2') ? 2 : 1; if (*e == '1' || *e == '2') e++; }
    if (*e || *tile < 0)
        return 0;
    *K = n;
    return 1;
}

/* the real four-step's handle: the split is plan input */
static struct vfft_plan_s *_zfsr_build_plan(const vfft_config_t *cfg, int N, int n1, int n2,
                                            struct vfft_wisdom_s *W)
{
    vfft_zfsr_plan_t *zp = vfft_zfsr_create(N, n1, n2, W, cfg, _vfft_plan_threads(cfg));
    struct vfft_plan_s *h;
    if (!zp)
        return NULL;
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_zfsr_destroy(zp);
        return NULL;
    }
    h->transform = cfg->transform;
    h->placement = cfg->placement;
    h->layout = (int)VFFT_LAYOUT_INTERLEAVED;
    h->N = N;
    h->K = 1;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zfsr = zp;
    return h;
}

/* both placements are the same pipeline (the plane is the plan's own) */
static void _exec_zfsr(struct vfft_plan_s *h, const double *sre, double *dre)
{
    if (h->nthreads > 1)
        _vfft_pool_arm(h->nthreads); /* the four-step child's threaded verdict and the sweeps run on the snapshot pool */
    if (h->transform == VFFT_R2C)
        vfft_zfsr_execute_fwd(h->zfsr, sre, dre);
    else
        vfft_zfsr_execute_bwd(h->zfsr, sre, dre);
}

/* VFFT_ZFSR at create: 1 = pinned at *n1 x *n2, 0 = kept out of the race, -1 = unset */
static int _zfsr_env(int *n1, int *n2)
{
    const char *e = getenv("VFFT_ZFSR");
    int a = 0, b = 0;
    if (!e || !e[0])
        return -1;
    if (sscanf(e, "%dx%d", &a, &b) == 2 && a > 0 && b > 0)
    {
        *n1 = a; *n2 = b;
        return 1;
    }
    return 0;
}

/* the legal arms of N: (R1, R2, form), R1 ascending, form A before B */
#define VFFT_ZRP_MAX_ARMS 32
static int _zrp_arms(int N, int out[][3], int cap)
{
    int n = 0;
    for (int R1 = 4; R1 <= 64 && n < cap; R1 += 2)
    {
        if (N % R1)
            continue;
        int R2 = N / R1;
        if (R2 < 4 || R2 > 64 || (R2 & 1))
            continue;
        for (int form = 0; form < 2 && n < cap; form++)
        {
            if (!vfft_zrp_pair_ok(N, R1, R2, form))
                continue;
            out[n][0] = R1;
            out[n][1] = R2;
            out[n][2] = form;
            n++;
        }
    }
    return n;
}

/* the race arms: a finished handle run through its own executor */
typedef struct
{
    struct vfft_plan_s *h;
    const double *s0;
    double *b;
} _zrpr_arm_t;
static void _zrpr_arm_run(void *v)
{
    _zrpr_arm_t *c = (_zrpr_arm_t *)v;
    if (c->h->zfsr)
        _exec_zfsr(c->h, c->s0, c->b);
    else if (c->h->zrm)
        _exec_zrm(c->h, c->s0, c->b);
    else if (c->h->zttr)
        _exec_zttr(c->h, c->s0, c->b);
    else if (c->h->zrp)
        _exec_zrp(c->h, c->s0, c->b);
    else
        _exec_zr2c(c->h, c->s0, c->b);
}
static void _real_il_exec_any(struct vfft_plan_s *h, const double *s0, double *b)
{
    if (h->zfsr) _exec_zfsr(h, s0, b);
    else if (h->zrm) _exec_zrm(h, s0, b);
    else if (h->zttr) _exec_zttr(h, s0, b);
    else if (h->zrp) _exec_zrp(h, s0, b);
    else _exec_zr2c(h, s0, b);
}

static double _zrpr_relerr(const double *a, const double *b, size_t n);

/* ZTT-r's sweep: every {4,8} chain of N/2 at the tile widths {untiled, 512,
 * 1024, 2048, 3072}, each gated against zr2c's output, then burst-timed
 * (best of five); the four fastest and the winner's other three stack
 * states become race arms. The finished handles are returned in hz[]. */
#define VFFT_ZTTR_MAX_ARMS 8
static int _zttr_sweep(const vfft_config_t *cfg, int N, const double *a, const double *ref,
                       double *b, const double *s0, size_t xs, size_t nchk,
                       struct vfft_plan_s *hz[VFFT_ZTTR_MAX_ARMS])
{
    static const size_t tiles[5] = { 0, 512, 1024, 2048, 3072 };
    const int M = N / 2;
    int chains[VFFT_ZTTR_MAX_CHAINS][8], nfs[VFFT_ZTTR_MAX_CHAINS];
    const int nc = vfft_zttr_chains(M, chains, nfs, VFFT_ZTTR_MAX_CHAINS);
    typedef struct { int c, ti, mt; double ns; } cand_t;
    const int Tk = _vfft_plan_threads(cfg);   /* a threaded plan sweeps the threaded arms too */
    cand_t cand[VFFT_ZTTR_MAX_CHAINS * 5];
    int ncand = 0;
    for (int c = 0; c < nc; c++)
        for (int ti = 0; ti < 5; ti++)
        {
            const size_t tile = tiles[ti];
            const int R0 = chains[c][0], R1 = chains[c][1];
            if (tile && (tile >= (size_t)M || (size_t)M % tile || tile < (size_t)(R0 * R1))) continue;
            struct vfft_plan_s *h = _zttr_build_plan(cfg, N, chains[c], nfs[c], tile, 3);
            if (!h) continue;
            memcpy(b, a, xs * sizeof(double));
            _exec_zttr(h, s0, b);
            if (_zrpr_relerr(b, ref, nchk) >= 1e-10) { vfft_destroy((vfft_plan)h); continue; }
            double t0 = vfft_now_ns();
            memcpy(b, a, xs * sizeof(double));
            _exec_zttr(h, s0, b);
            double est = vfft_now_ns() - t0;
            int reps = (int)(1.5e5 / (est > 1.0 ? est : 1.0));
            if (reps < 2) reps = 2;
            if (reps > 64) reps = 64;
            double best = 1e30;
            int bmt = 0;
            for (int arm = 0; arm <= (Tk > 1 ? 2 : 0); arm++)
            {   /* serial, then BLOCKS and TILES at the plan's T */
                if (arm)
                {
                    if (!vfft_zttr_mt_bind(h->zttr, Tk, arm)) continue;
                    memcpy(b, a, xs * sizeof(double));
                    _exec_zttr(h, s0, b);     /* gated like the serial run, and warm */
                    if (_zrpr_relerr(b, ref, nchk) >= 1e-10) continue;
                    _exec_zttr(h, s0, b);
                }
                for (int r = 0; r < 5; r++)
                {
                    double t = vfft_now_ns();
                    for (int i = 0; i < reps; i++) _exec_zttr(h, s0, b);
                    t = (vfft_now_ns() - t) / reps;
                    if (t < best) { best = t; bmt = arm; }
                }
            }
            cand[ncand].c = c; cand[ncand].ti = ti; cand[ncand].ns = best; cand[ncand].mt = bmt;
            ncand++;
            vfft_destroy((vfft_plan)h);
        }
    if (ncand == 0)
        return 0;
    for (int i = 1; i < ncand; i++)
        for (int j = i; j > 0 && cand[j].ns < cand[j - 1].ns; j--) { cand_t t = cand[j]; cand[j] = cand[j - 1]; cand[j - 1] = t; }
    int n = 0;
    for (int i = 0; i < ncand && i < 4; i++)
    {
        struct vfft_plan_s *h = _zttr_build_plan(cfg, N, chains[cand[i].c], nfs[cand[i].c], tiles[cand[i].ti], 3);
        if (h && cand[i].mt) vfft_zttr_mt_bind(h->zttr, Tk, cand[i].mt);
        if (h) hz[n++] = h;
    }
    for (int st = 0; st < 3 && n < VFFT_ZTTR_MAX_ARMS; st++)
    {
        struct vfft_plan_s *h = _zttr_build_plan(cfg, N, chains[cand[0].c], nfs[cand[0].c], tiles[cand[0].ti], st);
        if (h && cand[0].mt) vfft_zttr_mt_bind(h->zttr, Tk, cand[0].mt);
        if (h) hz[n++] = h;
    }
    return n;
}

/* the real four-step's sweep: every split of N/2, each gated against zr2c's
 * output, then burst-timed (best of three); the fastest is returned as the
 * one race arm, the others destroyed. */
static struct vfft_plan_s *_zfsr_sweep(const vfft_config_t *cfg, int N, struct vfft_wisdom_s *W,
                                       const double *a, const double *ref, double *b,
                                       const double *s0, size_t xs, size_t nchk)
{
    int n1[8], n2[8];
    const int ns = vfft_k1fs_splits(N / 2, n1, n2, 8);
    struct vfft_plan_s *best = NULL;
    double bestns = 1e300;
    for (int i = 0; i < ns; i++)
    {
        struct vfft_plan_s *h = _zfsr_build_plan(cfg, N, n1[i], n2[i], W);
        double t = 1e300;
        if (!h) continue;
        memcpy(b, a, xs * sizeof(double));
        _exec_zfsr(h, s0, b);
        {
            const double e = _zrpr_relerr(b, ref, nchk);
            if (e >= 1e-10)
            {
                fprintf(stderr, "[zfsr] N=%d split %dx%d FAILS the gate (rel %.2e vs zr2c) -- dropped\n",
                        N, n1[i], n2[i], e);
                vfft_destroy((vfft_plan)h);
                continue;
            }
        }
        for (int r = 0; r < 3; r++)
        {
            double t0 = vfft_now_ns();
            _exec_zfsr(h, s0, b);
            t0 = vfft_now_ns() - t0;
            if (t0 < t) t = t0;
        }
        if (t < bestns)
        {
            if (best) vfft_destroy((vfft_plan)best);
            best = h;
            bestns = t;
        }
        else
            vfft_destroy((vfft_plan)h);
    }
    return best;
}

/* max |a - b| / max |b| over n doubles */
static double _zrpr_relerr(const double *a, const double *b, size_t n)
{
    double e = 0.0, m = 0.0;
    for (size_t i = 0; i < n; i++)
    {
        double d = fabs(a[i] - b[i]);
        if (d > e) e = d;
        if (fabs(b[i]) > m) m = fabs(b[i]);
    }
    return m > 0.0 ? e / m : e;
}

/* The engine race. Returns the serving handle; NULL only when nothing builds. */
static struct vfft_plan_s *_real_il_race(const vfft_config_t *cfg, int N,
                                         struct vfft_wisdom_s *W)
{
    const int c2r = cfg->transform == VFFT_C2R, ip = cfg->placement == VFFT_INPLACE;
    const int Tk = _vfft_plan_threads(cfg);   /* the verdict's thread key */
    int arms_in[VFFT_ZRP_MAX_ARMS][3];
    const int np = _zrp_arms(N, arms_in, VFFT_ZRP_MAX_ARMS);
    struct vfft_plan_s *hz = _zr2c_build(cfg, N, W);   /* races + banks its own route */
    /* the real mono: one kernel at N <= 64 (VFFT_ZRM=0 keeps it out) */
    struct vfft_plan_s *hm = (N <= VFFT_ZRM_MAX_N && _zrm_env() != 0) ? _zrm_build_plan(cfg, N) : NULL;
    if (np == 0 && N < 64 && !hm)
        return hz;
    struct vfft_plan_s *hp[VFFT_ZRP_MAX_ARMS];
    int spec[VFFT_ZRP_MAX_ARMS][3];
    int na = 0;
    for (int i = 0; i < np; i++)
    {
        struct vfft_plan_s *h = _zrp_build_pair(cfg, N, arms_in[i][0], arms_in[i][1], arms_in[i][2]);
        if (h)
        {
            hp[na] = h;
            memcpy(spec[na], arms_in[i], sizeof spec[na]);
            na++;
        }
    }
    if (!hz)
    {
        /* zr2c could not build: the mono, else the first pair, serves; no bank */
        if (hm)
        {
            for (int i = 0; i < na; i++) vfft_destroy((vfft_plan)hp[i]);
            return hm;
        }
        return na ? hp[0] : NULL;
    }
    const size_t xs = (size_t)N + 2;
    double *a = (double *)vfft_aligned_alloc(xs * sizeof(double));
    double *b = (double *)vfft_aligned_alloc(xs * sizeof(double));
    double *ref = (double *)vfft_aligned_alloc(xs * sizeof(double));
    if (!a || !b || !ref)
    {
        vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
        for (int i = 0; i < na; i++) vfft_destroy((vfft_plan)hp[i]);
        if (hm) vfft_destroy((vfft_plan)hm);
        return hz;
    }
    unsigned sd = 0x9e3779b9u ^ (unsigned)N ^ (unsigned)(c2r << 8) ^ (unsigned)(ip << 9);
    for (size_t i = 0; i < xs; i++)
    {
        sd = sd * 1664525u + 1013904223u;
        a[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
    }
    if (c2r)
        a[1] = a[2 * (N / 2) + 1] = 0.0; /* a CCE spectrum: real DC and Nyquist */
    const double *s0 = ip ? b : a;
    /* the gate: each arm's output against zr2c's, before any timing */
    memcpy(b, a, xs * sizeof(double));
    _exec_zr2c(hz, s0, b);
    memcpy(ref, b, xs * sizeof(double));
    const size_t nchk = c2r ? (size_t)N : xs;
    int keep = 0;
    for (int i = 0; i < na; i++)
    {
        memcpy(b, a, xs * sizeof(double));
        _exec_zrp(hp[i], s0, b);
        double e = _zrpr_relerr(b, ref, nchk);
        if (e < 1e-10)
        {
            hp[keep] = hp[i];
            memcpy(spec[keep], spec[i], sizeof spec[keep]);
            keep++;
        }
        else
        {
            fprintf(stderr, "[zrp] N=%d %s %s pair %d.%d form %c FAILS the gate (rel %.2e vs zr2c) -- dropped\n",
                    N, c2r ? "c2r" : "r2c", ip ? "ip" : "oop", spec[i][0], spec[i][1],
                    spec[i][2] ? 'B' : 'A', e);
            vfft_destroy((vfft_plan)hp[i]);
        }
    }
    na = keep;
    /* the real mono, gated the same way */
    if (hm)
    {
        memcpy(b, a, xs * sizeof(double));
        _exec_zrm(hm, s0, b);
        double e = _zrpr_relerr(b, ref, nchk);
        if (e >= 1e-10)
        {
            fprintf(stderr, "[zrm] N=%d %s %s the real mono FAILS the gate (rel %.2e vs zr2c) -- dropped\n",
                    N, c2r ? "c2r" : "r2c", ip ? "ip" : "oop", e);
            vfft_destroy((vfft_plan)hm);
            hm = NULL;
        }
    }
    /* ZTT-r: the sweep's shortlist, gated in the sweep */
    struct vfft_plan_s *ht[VFFT_ZTTR_MAX_ARMS];
    const int nt = N >= 64 ? _zttr_sweep(cfg, N, a, ref, b, s0, xs, nchk, ht) : 0;
    /* the real four-step above ZTT-r's band: the fastest split (VFFT_ZFSR=0 keeps it out) */
    struct vfft_plan_s *hf = NULL;
    {
        int e1, e2;
        if (vfft_zfsr_band(N) && _zfsr_env(&e1, &e2) != 0)
            hf = _zfsr_sweep(cfg, N, W, a, ref, b, s0, xs, nchk);
    }
    /* zr2c with its fold cut over the plan's threads: an arm of its own at T > 1
     * (the same child route, gated like every arm) */
    struct vfft_plan_s *hzm = NULL;
    if (Tk > 1 && N >= 64)
    {
        hzm = _zr2c_build_route(cfg, N, hz->zr2c_route);
        if (hzm)
        {
            hzm->zr2c_fold_mt = 1;
            memcpy(b, a, xs * sizeof(double));
            _exec_zr2c(hzm, s0, b);
            if (_zrpr_relerr(b, ref, nchk) >= 1e-10)
            {
                fprintf(stderr, "[zr2c] N=%d %s %s the threaded fold FAILS the gate -- dropped\n",
                        N, c2r ? "c2r" : "r2c", ip ? "ip" : "oop");
                vfft_destroy((vfft_plan)hzm);
                hzm = NULL;
            }
        }
    }
    if (na == 0 && nt == 0 && !hm && !hf && !hzm)
    {
        vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
        return hz;
    }
    _vfft_create_race_count++;
    double t0 = vfft_now_ns();
    memcpy(b, a, xs * sizeof(double));
    _exec_zr2c(hz, s0, b);
    double est = vfft_now_ns() - t0;
    int reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
    if (reps < 2) reps = 2;
    if (reps > 4096) reps = 4096; /* a sample stays ~0.3 ms: the tiny cells (tens of ns a shot) need the reps */
    enum { NARMS = 4 + VFFT_ZRP_MAX_ARMS + VFFT_ZTTR_MAX_ARMS };
    _zrpr_arm_t ctx[NARMS];
    vfft_race_arm_t arms[NARMS];
    char names[NARMS][40];
    struct vfft_plan_s *hall[NARMS];
    int nall = 0;
    ctx[0].h = hz; ctx[0].s0 = s0; ctx[0].b = b;
    arms[0].name = "zr2c"; arms[0].run = _zrpr_arm_run; arms[0].ctx = &ctx[0];
    hall[nall++] = hz;
    if (hzm)
    {
        ctx[nall].h = hzm; ctx[nall].s0 = s0; ctx[nall].b = b;
        snprintf(names[nall], sizeof names[nall], "zr2c+foldmt");
        arms[nall].name = names[nall]; arms[nall].run = _zrpr_arm_run; arms[nall].ctx = &ctx[nall];
        hall[nall] = hzm;
        nall++;
    }
    if (hm)
    {
        ctx[nall].h = hm; ctx[nall].s0 = s0; ctx[nall].b = b;
        snprintf(names[nall], sizeof names[nall], "zrm");
        arms[nall].name = names[nall]; arms[nall].run = _zrpr_arm_run; arms[nall].ctx = &ctx[nall];
        hall[nall] = hm;
        nall++;
    }
    for (int i = 0; i < na; i++)
    {
        ctx[nall].h = hp[i]; ctx[nall].s0 = s0; ctx[nall].b = b;
        snprintf(names[nall], sizeof names[nall], "zrp%d.%d%c", spec[i][0], spec[i][1], spec[i][2] ? 'B' : 'A');
        arms[nall].name = names[nall]; arms[nall].run = _zrpr_arm_run; arms[nall].ctx = &ctx[nall];
        hall[nall] = hp[i];
        nall++;
    }
    if (hf)
    {
        ctx[nall].h = hf; ctx[nall].s0 = s0; ctx[nall].b = b;
        snprintf(names[nall], sizeof names[nall], "zfsr%dx%d", hf->zfsr->N1, hf->zfsr->N2);
        arms[nall].name = names[nall]; arms[nall].run = _zrpr_arm_run; arms[nall].ctx = &ctx[nall];
        hall[nall] = hf;
        nall++;
    }
    for (int i = 0; i < nt; i++)
    {
        char cs[32];
        ctx[nall].h = ht[i]; ctx[nall].s0 = s0; ctx[nall].b = b;
        vfft_ztt_chain_str(ht[i]->zttr->zt, cs, sizeof cs);
        snprintf(names[nall], sizeof names[nall], "zttr%s/%zu/s%d/m%d", cs, ht[i]->zttr->zt->tile, ht[i]->zttr->stk, ht[i]->zttr->mt);
        arms[nall].name = names[nall]; arms[nall].run = _zrpr_arm_run; arms[nall].ctx = &ctx[nall];
        hall[nall] = ht[i];
        nall++;
    }
    na = nall - 1;
    double ns[NARMS];
    {
        /* 9 rounds alternated, median; the in-place arms walk b, which the
         * race never reseeds: the values drift but the work does not. A
         * threaded plan's arms are never paced (a pause parks the pool and the
         * next round pays the wake) and take two untimed passes first */
        const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, Tk > 1 ? 2 : 1, NULL, NULL, Tk > 1 ? 0 : 1 };
        vfft_race_run(&proto, arms, nall, ns);
    }
    int best = 0;
    for (int i = 1; i <= na; i++)
        if (ns[i] < ns[best]) best = i;
    /* 3% hysteresis toward zr2c, the incumbent */
    if (best != 0 && !vfft_race_beats(ns[best], ns[0], 0.97))
        best = 0;
    if (getenv("VFFT_ZRACE_VERBOSE"))
    {
        fprintf(stderr, "[zrp] N=%d %s %s engine race: reps=%d hyst=3%% | zr2c=%.0f", N,
                c2r ? "c2r" : "r2c", ip ? "ip" : "oop", reps, ns[0]);
        for (int i = 1; i <= na; i++) fprintf(stderr, " %s=%.0f", names[i], ns[i]);
        fprintf(stderr, " -> %s\n", best ? names[best] : "zr2c");
    }
    vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
    if (best == 0)
    {
        /* zr2c's own (thread-free) route record stands; a threaded plan's verdict is its own
         * row, so the engine is banked at the thread key too or the cell would re-race */
        if (Tk > 1 && W && !W->vw2_off_oop &&
            vw2_real_il_bank_zr2c_t(&W->vw2, N, c2r, ip, Tk, hz->zr2c_route, 0, ns[0]) == VW2_OK)
            _vw2_persist(W, cfg);
        for (int i = 1; i < nall; i++) vfft_destroy((vfft_plan)hall[i]);
        return hz;
    }
    if (W && !W->vw2_off_oop)
    {
        struct vfft_plan_s *hw = hall[best];
        int rc;
        if (hw->zr2c_child)
            rc = vw2_real_il_bank_zr2c_t(&W->vw2, N, c2r, ip, Tk, hw->zr2c_route, hw->zr2c_fold_mt, ns[best]);
        else if (hw->zfsr)
            rc = vw2_real_il_bank_zfsr(&W->vw2, N, c2r, ip, Tk, hw->zfsr->N1, hw->zfsr->N2, ns[best]);
        else if (hw->zrm)
            rc = vw2_real_il_bank_zrm(&W->vw2, N, c2r, ip, Tk, ns[best]);
        else if (hw->zttr)
            rc = vw2_real_il_bank_zttr(&W->vw2, N, c2r, ip, Tk, hw->zttr->zt->chain, hw->zttr->zt->nf,
                                       hw->zttr->zt->tile, hw->zttr->stk, hw->zttr->mt, ns[best]);
        else
            rc = vw2_real_il_bank_zrp(&W->vw2, N, c2r, ip, Tk, hw->zrp->R1, hw->zrp->R2,
                                      hw->zrp->form, ns[best]);
        if (rc == VW2_OK)
            _vw2_persist(W, cfg);
        else
            fprintf(stderr, "vfft: real engine verdict NOT banked at N=%d (rc=%d) -- the cell will re-race\n", N, rc);
    }
    for (int i = 0; i < nall; i++)
        if (i != best) vfft_destroy((vfft_plan)hall[i]);
    return hall[best];
}

/* The door's engine pick: env pin, the banked engine, or the race. */
static struct vfft_plan_s *_real_il_build(const vfft_config_t *cfg, int N,
                                          struct vfft_wisdom_s *W)
{
    const int c2r = cfg->transform == VFFT_C2R, ip = cfg->placement == VFFT_INPLACE;
    const int Tk = _vfft_plan_threads(cfg);   /* the verdict's thread key */
    {
        const char *e = getenv("VFFT_ZRP");
        if (e && e[0])
        {
            int R1 = 0, R2 = 0, form = 0;
            if (!strcmp(e, "0"))
                return _zr2c_build(cfg, N, W);
            if (sscanf(e, "%d.%d.%d", &R1, &R2, &form) >= 2)
            {
                struct vfft_plan_s *h = _zrp_build_pair(cfg, N, R1, R2, form ? 1 : 0);
                if (h)
                    return h;
                _vfft_warn("vfft_create: VFFT_ZRP=%s does not build at N=%d (falling through to the door)", e, N);
            }
        }
    }
    {
        const char *e = getenv("VFFT_ZTTR");
        if (e && e[0])
        {
            int chain[8], nf = 0, stk = 3, pmt = 0;
            unsigned long tile = 0;
            const char *q = e;
            while (*q && nf < 8)
            {
                char *end;
                long v = strtol(q, &end, 10);
                if (end == q) break;
                chain[nf++] = (int)v;
                q = end;
                if (*q == '.') q++; else break;
            }
            if (*q == '/')
            {   /* /tile[/stk[/mt]] */
                tile = strtoul(q + 1, (char **)&q, 10);
                if (*q == '/') { stk = (int)strtol(q + 1, (char **)&q, 10); if (*q == '/') pmt = atoi(q + 1); }
            }
            if (nf >= 2)
            {
                struct vfft_plan_s *h = _zttr_build_plan(cfg, N, chain, nf, (size_t)tile, stk);
                if (h && pmt > 0 && Tk > 1) vfft_zttr_mt_bind(h->zttr, Tk, pmt);
                if (h)
                    return h;
            }
            _vfft_warn("vfft_create: VFFT_ZTTR=%s does not build at N=%d (falling through to the door)", e, N);
        }
    }
    {
        int e1 = 0, e2 = 0;
        if (_zfsr_env(&e1, &e2) == 1)
        {
            struct vfft_plan_s *h = _zfsr_build_plan(cfg, N, e1, e2, W);
            if (h)
                return h;
            _vfft_warn("vfft_create: VFFT_ZFSR=%dx%d does not build at N=%d (falling through to the door)", e1, e2, N);
        }
    }
    if (_zrm_env() == 1)
    {
        struct vfft_plan_s *h = _zrm_build_plan(cfg, N);
        if (h)
            return h;
        _vfft_warn("vfft_create: VFFT_ZRM=1 has no real mono kernel at N=%d (falling through to the door)", N);
    }
    if (W && !W->vw2_off_oop && !cfg->recalibrate)
    {
        int R1, R2, form;
        const char *eng = vw2_real_il_lookup(&W->vw2, N, c2r, ip, Tk, &R1, &R2, &form);
        if (eng && !strcmp(eng, "zfsr"))
        {
            int s1, s2;
            if (vw2_real_il_lookup_zfsr(&W->vw2, N, c2r, ip, Tk, &s1, &s2))
            {
                struct vfft_plan_s *h = _zfsr_build_plan(cfg, N, s1, s2, W);
                if (h)
                    return h;
            }
            /* a banked split that no longer builds: fall through to the race */
        }
        else if (eng && !strcmp(eng, "zrm"))
        {
            struct vfft_plan_s *h = _zrm_env() == 0 ? NULL : _zrm_build_plan(cfg, N);
            if (h)
                return h;
            if (_zrm_env() == 0)
                return _zr2c_build(cfg, N, W); /* the mono kept out by env: the incumbent serves */
            /* a banked mono without a kernel at this ISA: fall through to the race */
        }
        else if (eng && !strcmp(eng, "zttr"))
        {
            int chain[8], nf, stk, bmt;
            size_t tile;
            if (vw2_real_il_lookup_zttr(&W->vw2, N, c2r, ip, Tk, chain, &nf, &tile, &stk, &bmt))
            {
                struct vfft_plan_s *h = _zttr_build_plan(cfg, N, chain, nf, tile, stk);
                if (h && bmt > 0 && Tk > 1) vfft_zttr_mt_bind(h->zttr, Tk, bmt);
                if (h)
                    return h;
            }
            /* a banked chain that no longer builds: fall through to the race */
        }
        else if (eng && !strcmp(eng, "zrp") && R1 > 0)
        {
            struct vfft_plan_s *h = _zrp_build_pair(cfg, N, R1, R2, form);
            if (h)
                return h;
            /* a banked pair that no longer builds: fall through to the race */
        }
        else if (eng && !strcmp(eng, "zr2c"))
        {
            int route, fmt;
            if (Tk > 1 && vw2_real_il_lookup_zr2c_t(&W->vw2, N, c2r, ip, Tk, &route, &fmt))
            {
                struct vfft_plan_s *h = _zr2c_build_route(cfg, N, route);
                if (h)
                {
                    h->zr2c_fold_mt = fmt;
                    return h;
                }
            }
            return _zr2c_build(cfg, N, W);
        }
    }
    if (!W || W->vw2_off_oop)
        return _zr2c_build(cfg, N, W);
    return _real_il_race(cfg, N, W);
}

#endif /* VFFT_TRANSFORMS_REAL_ZRP_BUILD_H */
