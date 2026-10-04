/* zrp_build.h - the real pair's, ZTT-r's and the real mono's handles and the
 * real door's engine race.
 *
 * THE DOOR (il/real/real_create_il.h) serves an even-N, K=1, interleaved
 * real request with one of four engines:
 *   zr2c  x read as z[N/2] -> a c2c(N/2) child -> the Hermitian fold pass
 *         (zr2c_build.h: the child raced in the real role per route, the
 *         two composites raced; the route and the child's recipe are
 *         banked here when zr2c wins) -- an ordinary candidate
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
 *         N <= 64; no plan input. Odd N races it in il/real/odd_build.h
 *         (the door is even-N).
 *   zrf   the real flat DIT (zrf.h): odd N on the c2c flat DIT's stages
 *         behind a real leaf; the chain, the split-body switch and the
 *         tile budget are PLAN INPUT, swept by the odd race in
 *         il/real/odd_build.h.
 * The cell's engine is read from the real shard (wisdom2_real_il.h); a miss
 * races every candidate the band admits (the c2c band map at N/2: zr2c at
 * every even N, its threaded fold at T > 1, the mono to 64, the pairs inside
 * their band, ZTT-r's shortlist, the four-step in its band) through the
 * finished handles, gates each against the independent reference before
 * timing, and banks the fastest -- no incumbent, no bias; one candidate is
 * still timed and banked, none is a refusal. VFFT_ZRP=R1.R2[.f] pins a pair
 * (f = 0 form A, 1 form B; default A), VFFT_ZTTR=chain/tile[/stk]
 * (4.8.8.4/512/3) [/mt] pins ZTT-r (mt = its threaded arm on a threaded
 * plan), VFFT_ZRM=1 pins the real mono, VFFT_ZFSR=N1xN2 pins the real
 * four-step; VFFT_ZRP=0, VFFT_ZRM=0 and VFFT_ZFSR=0 keep that engine out of
 * the race. Env beats wisdom and never banks: a pin serves unbanked, and a
 * race with an engine kept out banks nothing.
 *
 * INCLUSION CONTRACT: after zr2c_build.h and _vw2_persist (vfft.c).
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
#include "zrb.h"
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

/* the real Bluestein (zrb.h): both placements are the same pipeline (the
 * planes are the plan's own); the handle is built in the bridge, where the
 * inner descriptors live */
static void _exec_zrb(struct vfft_plan_s *h, const double *sre, double *dre)
{
    if (h->K > 1)
    {   /* the lane-major batch: the one-row pipeline per lane, edges at the lane stride */
        if (h->transform == VFFT_R2C)
            vfft_zrb_execute_fwd_lanes(h->zrb, sre, dre, (int)h->K);
        else
            vfft_zrb_execute_bwd_lanes(h->zrb, sre, dre, (int)h->K);
        return;
    }
    if (h->zrb->mt > 0 && h->nthreads > 1)
    {   /* the threaded form the race bound (zrb_mt.h); it declines on a clamped pool */
        _vfft_pool_arm(h->nthreads);
        if (vfft_zrb_execute_mt(h->zrb, sre, dre, h->transform != VFFT_R2C))
            return;
    }
    if (h->transform == VFFT_R2C)
        vfft_zrb_execute_fwd(h->zrb, sre, dre);
    else
        vfft_zrb_execute_bwd(h->zrb, sre, dre);
}

/* the lane Bluestein (zrb_lanes.h): K lanes, out of place, the plane the plan's own */
static void _exec_zrbl(struct vfft_plan_s *h, const double *sre, double *dre)
{
    if (h->transform == VFFT_R2C)
        vfft_zrbl_execute_fwd(h->zrbl, sre, dre);
    else
        vfft_zrbl_execute_bwd(h->zrbl, sre, dre);
}

/* the real Bluestein's cells: odd N past the mono's (64) with no flat DIT chain */
static int _zrb_ok(int N)
{
    return (N & 1) && N > VFFT_ZRM_MAX_N && !_zrf_has_chain(N);
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

/* the real four-step's handle: the split and the child are plan input. The
 * child races into the plan's private store (c2d, crow NULL) or replays the
 * rows it is given (the real row's fs_*, fs_row_*, and fs_row_bwd_* where
 * cbwd holds any); a replay whose child raced -- its store gained a row --
 * does not serve that row: NULL. */
static struct vfft_plan_s *_zfsr_build_plan(const vfft_config_t *cfg, int N, int n1, int n2,
                                            const vw2_rec_t *c2d, const vw2_rec_t *crow,
                                            const vw2_rec_t *cbwd)
{
    struct vfft_wisdom_s *S = _k1fs_store_new();
    const int T = _vfft_plan_threads(cfg);
    vfft_zfsr_plan_t *zp;
    struct vfft_plan_s *h;
    int seeded = 0;
    if (!S)
        return NULL;
    if (c2d && crow)
    {
        vw2_key_t k2, kr, kb;
        if (!_k1fs_child_keys(n1, n2, 0, T, &k2, &kr, &kb) || _k1fs_seed(S, &k2, c2d) || _k1fs_seed(S, &kr, crow) ||
            (cbwd && cbwd->ntok > 0 && _k1fs_seed(S, &kb, cbwd)))
        {
            vfft_wisdom_free((vfft_wisdom *)S);
            return NULL;
        }
        seeded = S->vw2.nrec;
    }
    zp = vfft_zfsr_create(N, n1, n2, S, cfg, T);   /* the plan owns S from here */
    if (!zp)
        return NULL;
    if (seeded && zp->S->vw2.nrec != seeded)
    {
        vfft_zfsr_destroy(zp);
        return NULL;
    }
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
 * 1024, 2048, 3072}, each gated against the reference, then burst-timed
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

/* the real four-step's sweep: every split of N/2, each gated against the
 * reference, then burst-timed (best of three); the fastest is returned as the
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
        struct vfft_plan_s *h = _zfsr_build_plan(cfg, N, n1[i], n2[i], NULL, NULL, NULL);
        double t = 1e300;
        if (!h) continue;
        memcpy(b, a, xs * sizeof(double));
        _exec_zfsr(h, s0, b);
        {
            const double e = _zrpr_relerr(b, ref, nchk);
            if (e >= 1e-10)
            {
                fprintf(stderr, "[zfsr] N=%d split %dx%d FAILS the gate (rel %.2e vs the reference) -- dropped\n",
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

/* THE GATE'S REFERENCE (both real doors: this one and il/real/odd_build.h, and
 * the lane Bluestein's create): the INDEPENDENT forward DFT the c2c planner
 * gates against (il/planning/dp_planner_il.h) -- the scalar radix-2 transform
 * at a pow2 N, the long-double mixed-radix DIT at any N whose primes the
 * chain radices carry, and at an N with a prime factor past them (where the
 * mixed-radix form turns quadratic: minutes at a prime near 2^18) a Bluestein
 * convolution over the radix-2 transform, its chirp in long double. It shares
 * nothing with the candidates (no codelet, no plan, no table) and touches no
 * wisdom; it checks itself against VFFT_IL_DP_REF_PROBES bins summed directly.
 * In place over N complex points; 0 = trusted, -1 = refuse the cell. */
static int _real_il_ref_bluestein(double *z, long N)
{
    long M = 1, k;
    double *a, *b;
    while (M < 2 * N - 1) M <<= 1;
    a = (double *)vfft_aligned_alloc((size_t)M * 4u * sizeof(double));
    if (!a)
        return -1;
    b = a + 2 * M;
    memset(a, 0, (size_t)M * 4u * sizeof(double));
    for (k = 0; k < N; k++)
    {   /* w_k = exp(-i pi k^2 / N), the angle reduced mod 2N exactly */
        const long double ang = VFFT_PI_L * (long double)(((long long)k * k) % (2LL * N)) / (long double)N;
        const double wr = (double)cosl(ang), wi = -(double)sinl(ang);
        a[2 * k] = z[2 * k] * wr - z[2 * k + 1] * wi;
        a[2 * k + 1] = z[2 * k] * wi + z[2 * k + 1] * wr;
        b[2 * k] = wr; b[2 * k + 1] = -wi;                    /* conj(w_k) */
        if (k)
        {
            b[2 * (M - k)] = wr; b[2 * (M - k) + 1] = -wi;
        }
    }
    _il_dp_ref_dft(a, M);
    _il_dp_ref_dft(b, M);
    for (k = 0; k < M; k++)
    {   /* the product, conjugated: the inverse is conj(forward(conj)) */
        const double xr = a[2 * k] * b[2 * k] - a[2 * k + 1] * b[2 * k + 1];
        const double xi = a[2 * k] * b[2 * k + 1] + a[2 * k + 1] * b[2 * k];
        a[2 * k] = xr; a[2 * k + 1] = -xi;
    }
    _il_dp_ref_dft(a, M);
    for (k = 0; k < N; k++)
    {   /* conv_k = conj(a_k) / M, then X_k = w_k conv_k */
        const long double ang = VFFT_PI_L * (long double)(((long long)k * k) % (2LL * N)) / (long double)N;
        const double wr = (double)cosl(ang), wi = -(double)sinl(ang);
        const double cr = a[2 * k] / (double)M, ci = -a[2 * k + 1] / (double)M;
        z[2 * k] = cr * wr - ci * wi;
        z[2 * k + 1] = cr * wi + ci * wr;
    }
    vfft_aligned_free(a);
    return 0;
}
static int _real_il_ref_dft(double *z, long N)
{
    double *z0 = (double *)vfft_aligned_alloc((size_t)N * 2u * sizeof(double));
    double scale = 0.0;
    if (!z0 || N < 2)
    {
        vfft_aligned_free(z0);
        return -1;
    }
    memcpy(z0, z, (size_t)N * 2u * sizeof(double));
    if ((N & (N - 1)) == 0)
        _il_dp_ref_dft(z, N);
    else if (vfft_policy_prime_cell((int)N))
    {
        if (_real_il_ref_bluestein(z, N) != 0)
        {
            vfft_aligned_free(z0);
            return -1;
        }
    }
    else
        _il_dp_ref_dft_mixed(z, N);
    for (long m = 0; m < N; m++)
    {
        const double g = fabs(z[2 * m]) + fabs(z[2 * m + 1]);
        if (g > scale) scale = g;
    }
    if (!(scale > 0.0))
    {
        vfft_aligned_free(z0);
        return -1;                            /* also catches a NaN reference */
    }
    for (int p = 0; p < VFFT_IL_DP_REF_PROBES; p++)
    {
        const long m = ((long)p * N) / VFFT_IL_DP_REF_PROBES + p;
        double sr = 0.0, si = 0.0, d;
        if (m >= N) break;
        for (long j = 0; j < N; j++)
        {
            const double ang = -2.0 * VFFT_PI * (double)(((long long)j * m) % N) / (double)N;
            const double cr = cos(ang), ci = sin(ang);
            sr += z0[2 * j] * cr - z0[2 * j + 1] * ci;
            si += z0[2 * j] * ci + z0[2 * j + 1] * cr;
        }
        d = fabs(z[2 * m] - sr) + fabs(z[2 * m + 1] - si);
        if (!(d / scale <= VFFT_IL_DP_REF_TOL))
        {
            vfft_aligned_free(z0);
            return -1;                        /* NaN-safe */
        }
    }
    vfft_aligned_free(z0);
    return 0;
}
/* the real cell's reference from the independent DFT: r2c = the bins
 * 0..N/2 of the N reals in a (2*(N/2+1) doubles); c2r = N x from the N/2+1
 * bins in a (an even N's Nyquist bin real): the unnormalised inverse is
 * conj(forward(conj)) of the Hermitian-extended spectrum. 0 = trusted. */
static int _real_il_ref(int c2r, int N, const double *a, double *ref)
{
    const size_t n = (size_t)N, hp1 = n / 2 + 1;
    double *z = (double *)vfft_aligned_alloc(2 * n * sizeof(double));
    int rc;
    if (!z)
        return -1;
    if (!c2r)
    {
        _il2d_row_promote(a, z, n);
        rc = _real_il_ref_dft(z, (long)N);
        if (rc == 0)
            memcpy(ref, z, 2 * hp1 * sizeof(double));
    }
    else
    {
        _il2d_row_extend(a, z, n, hp1);
        for (size_t k = 0; k < n; k++) z[2 * k + 1] = -z[2 * k + 1];
        rc = _real_il_ref_dft(z, (long)N);
        if (rc == 0)
            for (size_t k = 0; k < n; k++) ref[k] = z[2 * k];   /* Re(conj(.)) = Re(.) */
    }
    vfft_aligned_free(z);
    return rc;
}

/* The pairs' band: the c2c band map at N/2 (il/planning/policy_il.h) -- the
 * pairs lost every pow2 c2c cell at 2048 and above, so a real pair whose N/2
 * is such a cell is out of the race. */
static int _real_il_pair_band(int N)
{
    const int M = N / 2;
    return !((M & (M - 1)) == 0 && M >= 2048);
}

/* THE ENGINE RACE (no incumbent). The candidates, admitted per band (the c2c
 * band map at N/2; each family answers for itself whether it builds at N):
 *   zr2c   at every even N (the fold and a c2c(N/2) child raced in the real
 *          role, its route raced inside its build); its threaded fold at
 *          T > 1 is a candidate of its own;
 *   zrm    N <= 64 with an rn1 kernel;
 *   zrp    every legal pair in both forms, inside the pairs' band;
 *   zttr   ZTT-r's sweep shortlist (N >= 64, a ZTT chain at N/2);
 *   zfsr   the real four-step's fastest split, in its band.
 * Every candidate is gated against the independent reference before timing,
 * then all race on equal terms (9 rounds alternated, median; a threaded
 * plan's arms unpaced) and the fastest serves. A single candidate is still
 * timed and banked: it IS the cell's verdict. No candidate: NULL (the door
 * warns and refuses). bank = 0 (the store off, or an env switch keeping an
 * engine out of the field) races without banking. out_zrp / out_zrm keep the
 * pairs / the mono out (VFFT_ZRP=0 / VFFT_ZRM=0); the four-step's own switch
 * (VFFT_ZFSR=0) is read here. */
static struct vfft_plan_s *_real_il_race(const vfft_config_t *cfg, int N, struct vfft_wisdom_s *W,
                                         int bank, int out_zrp, int out_zrm)
{
    const int c2r = cfg->transform == VFFT_C2R, ip = cfg->placement == VFFT_INPLACE;
    const int Tk = _vfft_plan_threads(cfg);   /* the verdict's thread key */
    enum { NARMS = 4 + VFFT_ZRP_MAX_ARMS + VFFT_ZTTR_MAX_ARMS };
    const size_t xs = (size_t)N + 2;
    const size_t nchk = c2r ? (size_t)N : xs;
    struct vfft_plan_s *hall[NARMS];
    char names[NARMS][40];
    int nall = 0;
    double *a = (double *)vfft_aligned_alloc(xs * sizeof(double));
    double *b = (double *)vfft_aligned_alloc(xs * sizeof(double));
    double *ref = (double *)vfft_aligned_alloc(xs * sizeof(double));
    const double *s0 = ip ? b : a;
    if (!a || !b || !ref)
    {
        vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
        return NULL;
    }
    {
        unsigned sd = 0x9e3779b9u ^ (unsigned)N ^ (unsigned)(c2r << 8) ^ (unsigned)(ip << 9);
        for (size_t i = 0; i < xs; i++)
        {
            sd = sd * 1664525u + 1013904223u;
            a[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
        }
    }
    if (c2r)
        a[1] = a[2 * (N / 2) + 1] = 0.0; /* a CCE spectrum: real DC and Nyquist */
    memset(ref, 0, xs * sizeof(double));
    if (_real_il_ref(c2r, N, a, ref) != 0)
    {
        _vfft_warn("vfft_create: %s N=%d: the gate's reference failed its self-check; no engine is raced",
                   _vfft_tname(cfg->transform), N);
        vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
        return NULL;
    }
#define ZRPR_GATE(h, what)                                                                          \
    do {                                                                                            \
        memcpy(b, a, xs * sizeof(double));                                                          \
        _real_il_exec_any((h), s0, b);                                                              \
        const double e_ = _zrpr_relerr(b, ref, nchk);                                               \
        if (e_ >= 1e-10)                                                                            \
        {                                                                                           \
            fprintf(stderr, "[real] N=%d %s %s %s FAILS the gate (rel %.2e vs the reference) -- dropped\n", \
                    N, c2r ? "c2r" : "r2c", ip ? "ip" : "oop", (what), e_);                         \
            vfft_destroy((vfft_plan)(h));                                                           \
            (h) = NULL;                                                                             \
        }                                                                                           \
    } while (0)
    /* zr2c, and its threaded fold at T > 1 (the same child route) */
    {
        struct vfft_plan_s *hz = _zr2c_build(cfg, N, W);
        struct vfft_plan_s *hzm = NULL;
        if (hz && Tk > 1 && N >= 64)
        {
            hzm = _zr2c_build_route(cfg, N, hz->zr2c_route, &hz->zr2c_kid->c, &hz->zr2c_kid->pr,
                                    _zr2c_fs_dup(hz->zr2c_kid));
            if (hzm)
            {
                hzm->zr2c_fold_mt = 1;
                if (hzm->zr2c_kid->ilp)
                {   /* a prime child (N = 2p: no ZTT-r chain carries N/2): its
                     * threaded form, raced at T on its own convolution
                     * (il_prime_mt.h); serial when it does not win */
                    _vfft_pool_arm(Tk);
                    vfft_ilprime_mt_race(hzm->zr2c_kid->ilp, Tk, NULL);
                }
                ZRPR_GATE(hzm, "zr2c+foldmt");
            }
        }
        if (hz)
            ZRPR_GATE(hz, "zr2c");
        if (hz)
        {
            snprintf(names[nall], sizeof names[nall], "zr2c");
            hall[nall++] = hz;
        }
        if (hzm)
        {
            snprintf(names[nall], sizeof names[nall], "zr2c+foldmt");
            hall[nall++] = hzm;
        }
    }
    /* the real mono */
    if (N <= VFFT_ZRM_MAX_N && !out_zrm)
    {
        struct vfft_plan_s *hm = _zrm_build_plan(cfg, N);
        if (hm)
            ZRPR_GATE(hm, "zrm");
        if (hm)
        {
            snprintf(names[nall], sizeof names[nall], "zrm");
            hall[nall++] = hm;
        }
    }
    /* the pairs, inside their band */
    if (!out_zrp && _real_il_pair_band(N))
    {
        int arms_in[VFFT_ZRP_MAX_ARMS][3];
        const int np = _zrp_arms(N, arms_in, VFFT_ZRP_MAX_ARMS);
        for (int i = 0; i < np && nall < NARMS; i++)
        {
            char what[24];
            struct vfft_plan_s *h = _zrp_build_pair(cfg, N, arms_in[i][0], arms_in[i][1], arms_in[i][2]);
            snprintf(what, sizeof what, "zrp%d.%d%c", arms_in[i][0], arms_in[i][1], arms_in[i][2] ? 'B' : 'A');
            if (h)
                ZRPR_GATE(h, what);
            if (h)
            {
                snprintf(names[nall], sizeof names[nall], "%s", what);
                hall[nall++] = h;
            }
        }
    }
    /* ZTT-r: the sweep's shortlist, gated in the sweep */
    if (N >= 64)
    {
        struct vfft_plan_s *ht[VFFT_ZTTR_MAX_ARMS];
        const int nt = _zttr_sweep(cfg, N, a, ref, b, s0, xs, nchk, ht);
        for (int i = 0; i < nt; i++)
        {
            char cs[32];
            if (nall >= NARMS) { vfft_destroy((vfft_plan)ht[i]); continue; }
            vfft_ztt_chain_str(ht[i]->zttr->zt, cs, sizeof cs);
            snprintf(names[nall], sizeof names[nall], "zttr%s/%zu/s%d/m%d", cs, ht[i]->zttr->zt->tile, ht[i]->zttr->stk, ht[i]->zttr->mt);
            hall[nall++] = ht[i];
        }
    }
    /* the real four-step in its band: the fastest split (VFFT_ZFSR=0 keeps it out) */
    {
        int e1, e2;
        if (vfft_zfsr_band(N) && _zfsr_env(&e1, &e2) != 0 && nall < NARMS)
        {
            struct vfft_plan_s *hf = _zfsr_sweep(cfg, N, W, a, ref, b, s0, xs, nchk);
            if (hf)
            {
                snprintf(names[nall], sizeof names[nall], "zfsr%dx%d", hf->zfsr->N1, hf->zfsr->N2);
                hall[nall++] = hf;
            }
        }
    }
#undef ZRPR_GATE
    if (nall == 0)
    {
        vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
        return NULL;
    }
    _vfft_create_race_count++;
    double ns[NARMS];
    int best = 0;
    {
        _zrpr_arm_t ctx[NARMS];
        vfft_race_arm_t arms[NARMS];
        double t0, est;
        int reps;
        for (int i = 0; i < nall; i++)
        {
            ctx[i].h = hall[i]; ctx[i].s0 = s0; ctx[i].b = b;
            arms[i].name = names[i]; arms[i].run = _zrpr_arm_run; arms[i].ctx = &ctx[i];
        }
        memcpy(b, a, xs * sizeof(double));
        t0 = vfft_now_ns();
        _zrpr_arm_run(&ctx[0]);
        est = vfft_now_ns() - t0;
        reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
        if (reps < 2) reps = 2;
        if (reps > 4096) reps = 4096; /* a sample stays ~0.3 ms: the tiny cells (tens of ns a shot) need the reps */
        {
            /* 9 rounds alternated, median; the in-place arms walk b, which the
             * race never reseeds: the values drift but the work does not. A
             * threaded plan's arms are never paced (a pause parks the pool and
             * the next round pays the wake) and take two untimed passes first */
            const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, Tk > 1 ? 2 : 1, NULL, NULL, Tk > 1 ? 0 : 1 };
            vfft_race_run(&proto, arms, nall, ns);
        }
    }
    for (int i = 1; i < nall; i++)
        if (ns[i] < ns[best]) best = i;
    if (getenv("VFFT_ZRACE_VERBOSE"))
    {
        fprintf(stderr, "[real] N=%d %s %s engine race%s: |", N, c2r ? "c2r" : "r2c", ip ? "ip" : "oop",
                bank ? "" : " (not banked)");
        for (int i = 0; i < nall; i++) fprintf(stderr, " %s=%.0f%s", names[i], ns[i], i == best ? "*" : "");
        fprintf(stderr, "\n");
    }
    vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
    if (bank && W && !W->vw2_off_oop)
    {
        struct vfft_plan_s *hw = hall[best];
        int rc;
        if (hw->zr2c_kid)
            rc = _zr2c_bank(W, cfg, N, hw, ns[best]);
        else if (hw->zfsr)
        {   /* the split and the child's rows from the plan's private store */
            const vw2_rec_t *c2d, *crow, *cbwd;
            _k1fs_child_rows(hw->zfsr->S, hw->zfsr->N1, hw->zfsr->N2, 0, Tk, &c2d, &crow, &cbwd);
            rc = vw2_real_il_bank_zfsr(&W->vw2, N, c2r, ip, Tk, hw->zfsr->N1, hw->zfsr->N2, c2d, crow, cbwd,
                                       ns[best]);
        }
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

/* The door's engine pick: an env pin, the banked engine, or the race. A pin
 * serves its engine (never banks, never replays); an out-switch (VFFT_ZRP=0,
 * VFFT_ZRM=0, VFFT_ZFSR=0) keeps that engine out of the field, and a race run
 * with an engine out never banks (a verdict of a partial field is not the
 * cell's). A banked row naming an engine that no longer builds, or one kept
 * out, is a miss: the race. NULL = no engine builds at N. */
static struct vfft_plan_s *_real_il_build(const vfft_config_t *cfg, int N,
                                          struct vfft_wisdom_s *W)
{
    const int c2r = cfg->transform == VFFT_C2R, ip = cfg->placement == VFFT_INPLACE;
    const int Tk = _vfft_plan_threads(cfg);   /* the verdict's thread key */
    int out_zrp = 0;
    const int out_zrm = _zrm_env() == 0;
    int out_zfsr;
    {
        int e1 = 0, e2 = 0;
        out_zfsr = _zfsr_env(&e1, &e2) == 0;
    }
    {
        const char *e = getenv("VFFT_ZRP");
        if (e && e[0])
        {
            int R1 = 0, R2 = 0, form = 0;
            if (!strcmp(e, "0"))
                out_zrp = 1;
            else if (sscanf(e, "%d.%d.%d", &R1, &R2, &form) >= 2)
            {
                struct vfft_plan_s *h = _zrp_build_pair(cfg, N, R1, R2, form ? 1 : 0);
                if (h)
                    return h;
                _vfft_warn("vfft_create: VFFT_ZRP=%s does not build at N=%d (the door decides)", e, N);
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
            _vfft_warn("vfft_create: VFFT_ZTTR=%s does not build at N=%d (the door decides)", e, N);
        }
    }
    {
        int e1 = 0, e2 = 0;
        if (_zfsr_env(&e1, &e2) == 1)
        {
            struct vfft_plan_s *h = _zfsr_build_plan(cfg, N, e1, e2, NULL, NULL, NULL);
            if (h)
                return h;
            _vfft_warn("vfft_create: VFFT_ZFSR=%dx%d does not build at N=%d (the door decides)", e1, e2, N);
        }
    }
    if (_zrm_env() == 1)
    {
        struct vfft_plan_s *h = _zrm_build_plan(cfg, N);
        if (h)
            return h;
        _vfft_warn("vfft_create: VFFT_ZRM=1 has no real mono kernel at N=%d (the door decides)", N);
    }
    if (W && !W->vw2_off_oop && !cfg->recalibrate)
    {
        int R1, R2, form;
        const char *eng = vw2_real_il_lookup(&W->vw2, N, c2r, ip, Tk, &R1, &R2, &form);
        if (eng && !strcmp(eng, "zfsr") && !out_zfsr)
        {
            int s1, s2;
            vw2_rec_t c2d, crow, cbwd;
            if (vw2_real_il_lookup_zfsr(&W->vw2, N, c2r, ip, Tk, &s1, &s2, &c2d, &crow, &cbwd))
            {
                struct vfft_plan_s *h = _zfsr_build_plan(cfg, N, s1, s2, &c2d, &crow, &cbwd);
                vw2_rec_free(&c2d);
                vw2_rec_free(&crow);
                vw2_rec_free(&cbwd);
                if (h)
                    return h;
            }
        }
        else if (eng && !strcmp(eng, "zrm") && !out_zrm)
        {
            struct vfft_plan_s *h = _zrm_build_plan(cfg, N);
            if (h)
                return h;
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
        }
        else if (eng && !strcmp(eng, "zrp") && R1 > 0 && !out_zrp)
        {
            struct vfft_plan_s *h = _zrp_build_pair(cfg, N, R1, R2, form);
            if (h)
                return h;
        }
        else if (eng && !strcmp(eng, "zr2c"))
        {
            struct vfft_plan_s *h = _zr2c_replay(cfg, N, W);
            if (h)
                return h;
            /* a zr2c row without its child's recipe (it predates the in-role
             * child), or one that no longer builds: the race */
        }
        /* any other row (an engine kept out, a recipe that no longer builds):
         * the race, which overwrites it */
    }
    return _real_il_race(cfg, N, W, !out_zrp && !out_zrm && !out_zfsr, out_zrp, out_zrm);
}

#endif /* VFFT_TRANSFORMS_REAL_ZRP_BUILD_H */
