/* zr2c_build.h - the interleaved-CCE real route ("kind 5").
 *
 * THE IDEA
 * --------
 * A real transform of even N, K=1, on interleaved data does not need real
 * kernels at all. Read x[N] as z[N/2] - a reinterpretation, zero work - run a
 * complex child on it, and fold the result into the Hermitian half-spectrum.
 * c2r mirrors it exactly, with the fold leading instead of trailing.
 *
 * This is the like-for-like arm against a CCE-native engine: both engines
 * consume and produce the packed CCE plane, so neither is charged for a
 * conversion the other avoids.
 *
 * TWO CHILD ROUTES, RACED
 * -----------------------
 *   route 0  child_oop_il  - an out-of-place interleaved child, folding into a
 *                            separate plane.
 *   route 1  child_nat_ip  - an in-place child.
 *
 * The pick is per (transform, placement, thread count), raced and banked on
 * the real cell's row; where the two routes disagree the gap reaches 27-35%
 * (c2r out-of-place).
 *
 * THE CHILD IS RACED IN THE REAL ROLE (owner, 2026-10-02)
 * -------------------------------------------------------
 * The complex child does almost all the work, but it is not the c2c cell: the
 * real plan runs it between its own passes (the fold after it in r2c, before
 * it in c2r), at the route's placement, in one direction. So its plan is THIS
 * cell's verdict: per route, the IL planner's whole pool for N/2 races in that
 * role (dp_planner_il.h, the role: every candidate timed inside the
 * composite's pass), the two composites then race each other, and the
 * winner's route and child recipe are banked on the real cell's row in the
 * c2c K=1 record's vocabulary (wisdom2_real_il.h). Replay builds the child
 * from that recipe. A zr2c create reads and writes NO c2c row, and a
 * recalibrate re-races the child in role and nothing in the c2c library.
 *
 * The child is built straight from its recipe by the planner's own builder
 * (_il_dp_build): what serves is what raced, and the composite calls the
 * child's engines directly, never through the front door. One thread: a
 * threaded real plan threads the fold (zr2c_fold_mt), and the threaded real
 * engines (ZTT-r's, the four-step's) race beside it. A PRIME child's prime
 * cell is part of the recipe: its method and inner race on the cell's own
 * convolution with no store, ride on the real row (il_prime*) and replay from
 * there -- the prime shard is never read or written. The four-step's 2D child
 * keeps its own verdict (its rank-2 cell).
 *
 * INCLUSION CONTRACT
 * ------------------
 * Include after the IL planner (dp_planner_il.h) and the K=1 commit
 * (k1_commit.h: _ilprime_create_banked, _ilprime_build_with, _k1_il_fs_warm,
 * _k1_il_dp_busy), and
 * AFTER _vw2_persist (vfft.c).
 */
#ifndef VFFT_TRANSFORMS_REAL_ZR2C_BUILD_H
#define VFFT_TRANSFORMS_REAL_ZR2C_BUILD_H

#include <stdlib.h>
#include <string.h>

#include "vfft_internal.h"                 /* struct vfft_plan_s / vfft_wisdom_s */
#include "zr2c.h"                          /* the Hermitian fold kernels */
#include "wisdom2_real_il.h"               /* the real cell's row: route + the child's recipe */
#include "common/support/race.h"                  /* the shared race body */

/* Defined in vfft.c (tentative definition, external linkage). */
extern long _vfft_create_race_count;

/* ── THE CHILD ─────────────────────────────────────────────────────────────
 * The recipe and the engines built from it, for the route's placement. */

/* a PRIME child's prime cell: its method and inner (zero on every other route) */
typedef struct
{
    int m;                       /* 1 rader, 2 bluestein */
    _ilprime_inner_desc_t d;     /* the inner */
} _zr2c_prime_t;

struct vfft_zr2c_kid_s
{
    vfft_il_cand_t c;            /* the recipe: the c2c(N/2) plan raced in the real role */
    _zr2c_prime_t pr;            /* ... and, on route PRIME, its prime cell */
    _il_dp_built_t b;            /* its engines (the planner's builder) */
    vfft_ilprime_plan_t *ilp;    /* the prime cell, owned (b.ilp borrows it) */
    int n, inplace;
};

static void _zr2c_kid_destroy(struct vfft_zr2c_kid_s *k)
{
    if (!k)
        return;
    _il_dp_free(&k->b);   /* never frees b.ilp: borrowed */
    if (k->ilp)
        vfft_ilprime_destroy(k->ilp);
    free(k);
}

/* the child's run: z_in -> z_out, the direction the transform needs */
static inline int _zr2c_kid_exec(const struct vfft_zr2c_kid_s *k, const double *in, double *out, int bwd)
{
    return _il_dp_exec_io(&k->c, &k->b, in, out, bwd);
}

/* the child's own c2c request: what its components (the prime cell, the
 * four-step's 2D child) are created against */
static void _zr2c_kid_cfg(const vfft_config_t *cfg, int n, int inplace, vfft_config_t *c2)
{
    memset(c2, 0, sizeof *c2);
    c2->transform = VFFT_C2C;
    c2->placement = inplace ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    c2->rigor = cfg->rigor;
    c2->dims = 1;
    c2->n[0] = n;
    c2->howmany = 1;
    c2->order = VFFT_ORDER_NATURAL;
    c2->layout = VFFT_LAYOUT_INTERLEAVED;
    c2->nthreads = 1;
    c2->wisdom = cfg->wisdom;
    c2->wisdom_write = cfg->wisdom_write;
}

/* the recipe <-> the row's vocabulary */
static void _zr2c_child_of_cand(vw2_zr2c_child_t *o, const vfft_il_cand_t *c)
{
    memset(o, 0, sizeof *o);
    o->route = c->route;
    o->R1 = c->R1;
    o->R2 = c->R2;
    if (c->route == VFFT_K1_IL_CHAIN3)
    {   /* the chain is R2.A.B (the c2c record's il_chain) */
        o->c3[0] = c->R2;
        o->c3[1] = c->c3_A;
        o->c3[2] = c->c3_B;
    }
    if (c->route == VFFT_K1_IL_FLAT)
    {
        memcpy(o->fl, c->il_fl, sizeof(int) * (size_t)c->il_fl_n);
        o->fl_n = c->il_fl_n;
        memcpy(o->flf, c->il_flf, sizeof o->flf);
    }
    if (c->route == VFFT_K1_IL_ZTT || c->route == VFFT_K1_IL_FS)
    {
        memcpy(o->zt, c->il_zt, sizeof(int) * (size_t)c->il_zt_n);
        o->zt_n = c->il_zt_n;
    }
    o->tw = c->il_tw;
    o->kv = c->il_kv;
    o->bkv = c->il_bkv;
}
static void _zr2c_cand_of_child(vfft_il_cand_t *c, const vw2_zr2c_child_t *o)
{
    memset(c, 0, sizeof *c);
    c->route = o->route;
    c->R1 = o->R1;
    c->R2 = o->R2;
    if (o->route == VFFT_K1_IL_CHAIN3)
    {
        c->c3_A = o->c3[1];
        c->c3_B = o->c3[2];
    }
    if (o->route == VFFT_K1_IL_FLAT)
    {
        memcpy(c->il_fl, o->fl, sizeof(int) * (size_t)o->fl_n);
        c->il_fl_n = o->fl_n;
        memcpy(c->il_flf, o->flf, sizeof c->il_flf);
    }
    if (o->route == VFFT_K1_IL_ZTT || o->route == VFFT_K1_IL_FS)
    {
        memcpy(c->il_zt, o->zt, sizeof(int) * (size_t)o->zt_n);
        c->il_zt_n = o->zt_n;
    }
    c->il_tw = o->tw;
    c->il_kv = o->kv;
    c->il_bkv = o->bkv;
    c->il_bkv_raced = 1;
}
/* the whole recipe: the child's, plus its prime cell on route PRIME */
static void _zr2c_child_of_kid(vw2_zr2c_child_t *o, const struct vfft_zr2c_kid_s *k)
{
    _zr2c_child_of_cand(o, &k->c);
    if (k->c.route != VFFT_K1_IL_PRIME)
        return;
    o->pm = k->pr.m;
    _ilprime_desc_str(&k->pr.d, o->pin, sizeof o->pin, o->psh, sizeof o->psh);
    o->ptw = k->pr.d.tw;
}
/* the row's prime cell; 0 when a PRIME recipe's inner does not parse */
static int _zr2c_prime_of_child(_zr2c_prime_t *pr, const vw2_zr2c_child_t *o)
{
    memset(pr, 0, sizeof *pr);
    if (o->route != VFFT_K1_IL_PRIME)
        return 1;
    if ((o->pm != 1 && o->pm != 2) || !_ilprime_desc_parse(&pr->d, o->pin, o->psh, o->ptw))
        return 0;
    pr->m = o->pm;
    return 1;
}

/* build the child from its recipe (`pr`: its prime cell, route PRIME); NULL
 * when it no longer builds */
static struct vfft_zr2c_kid_s *_zr2c_kid_build(const vfft_config_t *cfg, struct vfft_wisdom_s *W,
                                               int n, int inplace, const vfft_il_cand_t *c,
                                               const _zr2c_prime_t *pr)
{
    vfft_config_t c2;
    struct vfft_zr2c_kid_s *k = (struct vfft_zr2c_kid_s *)calloc(1, sizeof *k);
    int rc;
    if (!k)
        return NULL;
    k->c = *c;
    k->n = n;
    k->inplace = inplace;
    _zr2c_kid_cfg(cfg, n, inplace, &c2);
    if (c->route == VFFT_K1_IL_PRIME)
    {   /* the prime cell, built from the recipe's method and inner */
        _ilprime_inner_desc_t d;
        if (pr && pr->m)
        {
            d = pr->d;
            k->ilp = _ilprime_build_with(n, pr->m == 1, &d);
        }
        if (!k->ilp)
        {
            free(k);
            return NULL;
        }
        k->pr = *pr;
        k->b.ilp = k->ilp;
        return k;
    }
    {   /* the four-step's 2D child is created against the child's request */
        struct vfft_wisdom_s *sw = _k1fs_ctx.W;
        const vfft_config_t *sc = _k1fs_ctx.cfg;
        _k1fs_ctx.W = W;
        _k1fs_ctx.cfg = &c2;
        rc = _il_dp_build(n, c, &k->b, inplace);
        _k1fs_ctx.W = sw;
        _k1fs_ctx.cfg = sc;
    }
    if (rc != 0 || (c->route == VFFT_K1_IL_MONO && !(cfg->transform == VFFT_C2R ? k->b.monob : k->b.mono)))
    {
        _il_dp_free(&k->b);
        free(k);
        return NULL;
    }
    return k;
}

/* ── THE CHILD'S RACE, IN THE REAL ROLE ────────────────────────────────────
 * The composite's pass with the candidate inside, at the route's placement:
 * r2c the child forward then the fold; c2r the fold then the child backward
 * (out of place through a scratch plane, as route 0 runs it). */
typedef struct
{
    int N, c2r;
    const double *aff;   /* [affS | affC | bwdS | bwdC], top + 1 each */
    double *scr;         /* N + 2 doubles */
} _zr2c_role_t;
static int _zr2c_role_run(void *v, const vfft_il_cand_t *c, const struct _il_dp_built_s *b,
                          double *zin, double *zout, int inplace)
{
    const _zr2c_role_t *r = (const _zr2c_role_t *)v;
    const int N = r->N, top = N / 4;
    const size_t xs = (size_t)N + 2;
    const double *aS = r->aff, *aC = r->aff + (top + 1);
    const double *bS = r->aff + 2 * (top + 1), *bC = r->aff + 3 * (top + 1);
    double *dst = inplace ? zin : zout;
    int rc;
    if (!r->c2r)
    {
        rc = _il_dp_exec_io(c, b, zin, dst, 0);
        if (rc == 0)
            _zr2c_fold_fwd(dst, dst, aS, aC, N, 1, xs, xs);
        return rc;
    }
    if (inplace)
    {
        _zr2c_fold_bwd(zin, zin, bS, bC, N, 1, xs, (size_t)N);
        return _il_dp_exec_io(c, b, zin, zin, 1);
    }
    _zr2c_fold_bwd(zin, r->scr, bS, bC, N, 1, xs, (size_t)N);
    return _il_dp_exec_io(c, b, r->scr, zout, 1);
}

static vfft_il_dp_context_t _zr2c_dp_ctx;   /* the child's planner: one create at a time */
static int _zr2c_dp_ready = 0;

/* The child's race for one route: the IL planner's pool for N/2 (the natural
 * class: the fold reads natural bins), each candidate gated (its forward
 * against the planner's own reference; a c2r child's backward by roundtrip)
 * and timed inside the composite's pass. Banks nothing; returns the winner's
 * in-role ns, 1e18 when nothing built. `pr` receives the winner's prime cell
 * (its method and inner) when the winner is PRIME, zero otherwise. */
static double _zr2c_kid_race(const vfft_config_t *cfg, struct vfft_wisdom_s *W, int N, int route,
                             vfft_il_cand_t *best, _zr2c_prime_t *pr)
{
    const int n = N / 2, top = N / 4, c2r = cfg->transform == VFFT_C2R;
    vfft_config_t c2;
    _zr2c_role_t role;
    double *aff, ns;
    memset(pr, 0, sizeof *pr);
    if (_k1_il_dp_busy || n < 2)
        return 1e18;   /* a race already holds the planner: never nested */
    aff = (double *)vfft_aligned_alloc(sizeof(double) * 4u * (size_t)(top + 1));
    role.scr = (double *)vfft_aligned_alloc(sizeof(double) * ((size_t)N + 2));
    if (!aff || !role.scr)
    {
        vfft_aligned_free(aff);
        vfft_aligned_free(role.scr);
        return 1e18;
    }
    _zr2c_init_aff(N, aff, aff + (top + 1), aff + 2 * (top + 1), aff + 3 * (top + 1));
    role.N = N;
    role.c2r = c2r;
    role.aff = aff;
    if (!_zr2c_dp_ready || n + 8 > _zr2c_dp_ctx.max_N)
    {   /* the planes hold the CCE plane (N/2 + 1 complex) the fold writes */
        if (_zr2c_dp_ready)
            vfft_il_dp_destroy(&_zr2c_dp_ctx);
        vfft_il_dp_init(&_zr2c_dp_ctx, n + 8 > 4096 ? n + 8 : 4096);
        _zr2c_dp_ready = 1;
    }
    if (cfg->rigor != VFFT_MEASURE)
        vfft_il_dp_set_patient(&_zr2c_dp_ctx);
    else
        vfft_il_dp_set_measure(&_zr2c_dp_ctx);
    _zr2c_dp_ctx.inplace = route;
    _zr2c_dp_ctx.role_run = _zr2c_role_run;
    _zr2c_dp_ctx.role_ctx = &role;
    _zr2c_dp_ctx.role_bwd = c2r;
    _zr2c_dp_ctx.role_key = 1 + c2r;
    _zr2c_kid_cfg(cfg, n, route, &c2);
    /* the components the pool's families build on: the prime cell where no
     * chain carries N/2 (vfft_policy_prime_cell) -- raced on its own
     * convolution with no store, its method and inner kept for the recipe --
     * and the four-step's row cells */
    _k1pr_release();
    if (vfft_policy_prime_cell(n))
    {
        _k1pr_ctx.plan = _ilprime_create_banked(NULL, &c2, n, &pr->d);
        _k1pr_ctx.N = _k1pr_ctx.plan ? n : 0;
        pr->m = _k1pr_ctx.plan ? (_k1pr_ctx.plan->method == 1 ? 1 : 2) : 0;
    }
    if (W)
        _k1_il_fs_warm(W, &c2, n);
    _k1fs_ctx.W = W;
    _k1fs_ctx.cfg = &c2;
    _k1_il_dp_busy = 1;
    ns = vfft_il_dp_plan(&_zr2c_dp_ctx, n, VFFT_IL_ORD_NATURAL, best, getenv("VFFT_ZRACE_VERBOSE") != NULL);
    _k1_il_dp_busy = 0;
    _k1fs_ctx.W = NULL;
    _k1fs_ctx.cfg = NULL;
    _k1pr_release();
    _zr2c_dp_ctx.role_run = NULL;
    _zr2c_dp_ctx.role_ctx = NULL;
    vfft_aligned_free(aff);
    vfft_aligned_free(role.scr);
    if (ns >= 1e17 || best->route != VFFT_K1_IL_PRIME)
        memset(pr, 0, sizeof *pr);   /* the recipe carries a prime cell only on route PRIME */
    if (getenv("VFFT_ZRACE_VERBOSE") && ns < 1e17)
        fprintf(stderr, "[zr2c] N=%d %s route %d child raced in role: route=%d %d.%d -> %.1f ns\n", N,
                c2r ? "c2r" : "r2c", route, best->route, best->R1, best->R2, ns);
    return ns;
}

/* the composite for one route, its child built from `child` and its prime
 * cell `pr` (child NULL: the child races in role first, banking nothing) */
static struct vfft_plan_s *_zr2c_build_route(const vfft_config_t *cfg, struct vfft_wisdom_s *W, int N,
                                             int route, const vfft_il_cand_t *child, const _zr2c_prime_t *pr)
{
    const int half = N / 2, top = N / 4;
    vfft_il_cand_t raced;
    _zr2c_prime_t rpr;
    struct vfft_zr2c_kid_s *kid;
    if (!child)
    {
        if (_zr2c_kid_race(cfg, W, N, route, &raced, &rpr) >= 1e17)
        {
            _vfft_warn("vfft_create: zr2c child c2c(%d) has no plan in the real role", half);
            return NULL;
        }
        child = &raced;
        pr = &rpr;
    }
    kid = _zr2c_kid_build(cfg, W, half, route, child, pr);
    if (!kid)
    {
        _vfft_warn("vfft_create: zr2c child c2c(%d) does not build from its recipe (route %d)", half, child->route);
        return NULL;
    }
    struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    /* 🔴 64-BYTE ALIGNED, not plain malloc. Both buffers are streamed by
     * AVX2 kernels: the fold reads aff and writes scr, then the child reads
     * scr end to end. malloc gives 16 bytes on this toolchain, so every
     * 32-byte access that straddles a 64-byte line costs an extra line touch.
     * The kernels use loadu/storeu, so this is pure throughput, not
     * correctness.
     *
     * Measured, N=2048, front-door arms: every route-0 arm that TOUCHES the
     * scratch ran slow (r2c IP 1469-1528 ns, c2r OOP 1374-1688, c2r IP
     * 1414-1674) while the one route-0 arm that does NOT touch it (r2c OOP,
     * which folds in place in dre) ran 1134-1221 -- and route 1, which
     * allocates no scratch at all, ran 1137-1261 everywhere. The correlation
     * is exact across all four arms. */
    double *aff = NULL, *scr = NULL;
    if (!(aff = vfft_aligned_alloc(sizeof(double) * 4u * (size_t)(top + 1))))
        aff = NULL;
    if (route == 0 &&
        !(scr = vfft_aligned_alloc(sizeof(double) * ((size_t)N + 2))))
        scr = NULL;
    if (!h || !aff || (route == 0 && !scr))
    {
        _zr2c_kid_destroy(kid);
        free(h);
        vfft_aligned_free(aff);
        vfft_aligned_free(scr);
        return NULL;
    }
    /* four tables: [affS | affC | bwdS | bwdC] in one allocation. The
     * backward pair is the RAW sin/cos -- see _zr2c_init_aff. */
    _zr2c_init_aff(N, aff, aff + (top + 1), aff + 2 * (top + 1),
                   aff + 3 * (top + 1));
    h->transform = cfg->transform;
    h->placement = cfg->placement;
    h->layout = (int)VFFT_LAYOUT_INTERLEAVED;
    h->N = N;
    h->K = 1;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zr2c_kid = kid;
    h->zr2c_route = route;
    h->zr2c_aff = aff;
    h->zr2c_scratch = scr;
    return h;
}

/* execute the composite. 2 transforms x 2 placements x 2 routes; the folds
 * are in-place-safe by construction (zr2c.h), scratch only where a
 * route-0 shape needs a second plane. */
static void _exec_zr2c(struct vfft_plan_s *h, const double *sre, double *dre)
{
    const int N = h->N, top = N / 4;
    const double *aS = h->zr2c_aff, *aC = h->zr2c_aff + (top + 1);
    const double *bS = h->zr2c_aff + 2 * (top + 1);
    const double *bC = h->zr2c_aff + 3 * (top + 1);
    const struct vfft_zr2c_kid_s *kid = h->zr2c_kid;
    size_t xs = (size_t)N + 2;
    /* the fold's serving: the plan's threads when the race said so (never on a
     * batch clone: the flag is set only on a K = 1 plan raced at T > 1) */
    const int fmt = h->zr2c_fold_mt && h->nthreads > 1;
    const int fT = fmt ? thread_pool_workers_for(h->nthreads) : 1;
#define ZR2C_FWD(zi, xo) do { if (fmt) _zr2c_fold_mt((zi), (xo), aS, aC, N, 0, fT); \
                              else _zr2c_fold_fwd((zi), (xo), aS, aC, N, 1, xs, xs); } while (0)
#define ZR2C_BWD(xi, zo) do { if (fmt) _zr2c_fold_mt((xi), (zo), bS, bC, N, 1, fT); \
                              else _zr2c_fold_bwd((xi), (zo), bS, bC, N, 1, xs, (size_t)N); } while (0)
    if (fmt)
        _vfft_pool_arm(h->nthreads); /* the snapshot pool, as the threaded fold asserts it */
    if (h->transform == VFFT_R2C)
    {
        if (h->zr2c_route == 0)
        {
            if (h->placement == VFFT_OUTOFPLACE)
            { /* child OOP sre->dre (its z view), fold in place in dre */
                (void)_zr2c_kid_exec(kid, sre, dre, 0);
                ZR2C_FWD(dre, dre);
            }
            else
            { /* in place: child OOP plane->scratch, fold scratch->plane */
                (void)_zr2c_kid_exec(kid, sre, h->zr2c_scratch, 0);
                ZR2C_FWD(h->zr2c_scratch, dre);
            }
        }
        else
        {
            /* 🔴 `dre != sre`, NOT `placement == OUTOFPLACE`. Route 1 runs
             * the child on dre, so gating the copy on PLACEMENT would make
             * an in-place plan called with a distinct dre transform whatever
             * was already in dre and never read sre at all (relerr 1.000,
             * silently). Route 0 reads sre and is correct under the
             * identical call. Keying on the POINTERS makes the two routes
             * behave the same way, so which one a cell banked cannot change
             * the answer. */
            if (dre != sre)
                memcpy(dre, sre, (size_t)N * sizeof(double));
            (void)_zr2c_kid_exec(kid, dre, dre, 0);
            ZR2C_FWD(dre, dre);
        }
    }
    else /* VFFT_C2R: CCE spectrum in sre -> N reals in dre */
    {
        if (h->zr2c_route == 0)
        { /* fold sre->scratch (zhat), child OOP scratch->dre */
            ZR2C_BWD(sre, h->zr2c_scratch);
            (void)_zr2c_kid_exec(kid, h->zr2c_scratch, dre, 1);
        }
        else
        { /* fold sre->dre (alias-safe when in place), child in place on dre */
            ZR2C_BWD(sre, dre);
            (void)_zr2c_kid_exec(kid, dre, dre, 1);
        }
    }
#undef ZR2C_FWD
#undef ZR2C_BWD
}

/* Bank a zr2c verdict on the real cell's row (its transform, placement and
 * thread count): the route, the fold's serving, the child's recipe. A prime
 * cell raced under the method pin (VFFT_ILPR_METHOD) is no race verdict and
 * is never banked, as on the prime row. */
static int _zr2c_bank(struct vfft_wisdom_s *W, const vfft_config_t *cfg, int N,
                      const struct vfft_plan_s *h, double ns)
{
    vw2_zr2c_child_t ch;
    if (!W || W->vw2_off_oop || !h->zr2c_kid)
        return -1;
    if (h->zr2c_kid->c.route == VFFT_K1_IL_PRIME && getenv("VFFT_ILPR_METHOD"))
        return -1;
    _zr2c_child_of_kid(&ch, h->zr2c_kid);
    return vw2_real_il_bank_zr2c(&W->vw2, N, cfg->transform == VFFT_C2R, cfg->placement == VFFT_INPLACE,
                                 _vfft_plan_threads(cfg), h->zr2c_route, h->zr2c_fold_mt, &ch, ns);
}

/* The banked verdict: the composite its row names, its child built from the
 * row's recipe. NULL on a miss -- and a zr2c row without a child recipe (it
 * predates the in-role child), a PRIME recipe without its prime cell, or one
 * whose recipe no longer builds is a miss. */
static struct vfft_plan_s *_zr2c_replay(const vfft_config_t *cfg, int N, struct vfft_wisdom_s *W)
{
    int route, fmt;
    vw2_zr2c_child_t ch;
    vfft_il_cand_t c;
    _zr2c_prime_t pr;
    struct vfft_plan_s *h;
    if (!W || W->vw2_off_oop)
        return NULL;
    if (!vw2_real_il_lookup_zr2c(&W->vw2, N, cfg->transform == VFFT_C2R, cfg->placement == VFFT_INPLACE,
                                 _vfft_plan_threads(cfg), &route, &fmt, &ch))
        return NULL;
    if (!_zr2c_prime_of_child(&pr, &ch))
        return NULL;
    _zr2c_cand_of_child(&c, &ch);
    h = _zr2c_build_route(cfg, W, N, route, &c, &pr);
    if (h)
        h->zr2c_fold_mt = fmt && _vfft_plan_threads(cfg) > 1;
    return h;
}

/* the two arms of the zr2c route race: two finished handles */
typedef struct { struct vfft_plan_s *h; const double *s0; double *b; } _zr2c_arm_t;
static void _zr2c_arm_run(void *v)
{
    _zr2c_arm_t *c = (_zr2c_arm_t *)v;
    _exec_zr2c(c->h, c->s0, c->b);
}
/* The zr2c composite: env pin, the banked verdict, or the race -- each route's
 * child raced in the real role, then the two FULL composites through
 * _exec_zr2c, 3% hysteresis toward the placement's structural route. Banks
 * nothing: the door's engine race banks the cell's winner (zrp_build.h), zr2c
 * included. */
static struct vfft_plan_s *_zr2c_build(const vfft_config_t *cfg, int N,
                                       struct vfft_wisdom_s *W)
{
    /* 1. env — the racing hook. Beats wisdom, never banks. */
    {
        const char *e = getenv("VFFT_ZR2C_ROUTE");
        if (e && e[0])
            return _zr2c_build_route(cfg, W, N, atoi(e) != 0, NULL, NULL);
    }
    const int def = (cfg->placement == VFFT_INPLACE) ? 1 : 0;
    const int slot = ((cfg->transform == VFFT_C2R) << 1) | (cfg->placement == VFFT_INPLACE);

    /* 2. the banked verdict */
    if (W && !cfg->recalibrate)
    {
        struct vfft_plan_s *h = _zr2c_replay(cfg, N, W);
        if (h)
            return h;
    }

    /* 3. the race: every rigor tier races (the library is measured-only --
     * there is no ESTIMATE tier) */
    _vfft_create_race_count++;   /* HARNESS: past the wisdom hit, the clock decides */
    vfft_il_cand_t c0, c1;
    _zr2c_prime_t p0, p1;
    struct vfft_plan_s *h0 = _zr2c_kid_race(cfg, W, N, 0, &c0, &p0) < 1e17 ? _zr2c_build_route(cfg, W, N, 0, &c0, &p0) : NULL;
    struct vfft_plan_s *h1 = _zr2c_kid_race(cfg, W, N, 1, &c1, &p1) < 1e17 ? _zr2c_build_route(cfg, W, N, 1, &c1, &p1) : NULL;
    if (!h0 || !h1) /* one route builds: it serves */
        return h0 ? h0 : h1;

    size_t xs = (size_t)N + 2;
    double *a = (double *)vfft_aligned_alloc((xs * 8 + 63) & ~(size_t)63);
    double *b = (double *)vfft_aligned_alloc((xs * 8 + 63) & ~(size_t)63);
    if (!a || !b)
    {
        vfft_aligned_free(a);
        vfft_aligned_free(b);
        vfft_destroy((vfft_plan)(def ? h0 : h1));
        return def ? h1 : h0;
    }
    unsigned sd = 0x243f6a88u ^ (unsigned)N ^ (unsigned)(slot << 8);
    for (size_t i = 0; i < xs; i++)
    {
        sd = sd * 1664525u + 1013904223u;
        a[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
        sd = sd * 1664525u + 1013904223u;
        b[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
    }
    const double *s0 = (cfg->placement == VFFT_OUTOFPLACE) ? a : b;
    /* est shots double as warmup; reps for ~300 us bursts */
    double t0 = vfft_now_ns();
    _exec_zr2c(h0, s0, b);
    double e0 = vfft_now_ns() - t0;
    t0 = vfft_now_ns();
    _exec_zr2c(h1, s0, b);
    double e1 = vfft_now_ns() - t0;
    double est = e0 > e1 ? e0 : e1;
    int reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
    if (reps < 2)
        reps = 2;
    if (reps > 64)
        reps = 64;
    double n0, n1;
    {
        _zr2c_arm_t ca = { h0, s0, b }, cb = { h1, s0, b };
        const vfft_race_arm_t arms[2] = { { "route0", _zr2c_arm_run, &ca },
                                          { "route1", _zr2c_arm_run, &cb } };
        /* 9 rounds alternated, median (the est shots above were the warm-up) */
        const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 0, NULL, NULL };
        double ns[2];
        vfft_race_run(&proto, arms, 2, ns);
        n0 = ns[0];
        n1 = ns[1];
    }
    vfft_aligned_free(a);
    vfft_aligned_free(b);
    int win = (def == 0) ? ((n1 < n0 * 0.97) ? 1 : 0)
                         : ((n0 < n1 * 0.97) ? 0 : 1);
    if (getenv("VFFT_ZRACE_VERBOSE"))
        fprintf(stderr, "[zr2c] N=%d %s %s route race: reps=%d hyst=3%% "
                        "alt-order median | oop-il=%.0f nat-ip=%.0f -> "
                        "route=%d\n",
                N, cfg->transform == VFFT_C2R ? "c2r" : "r2c",
                cfg->placement == VFFT_INPLACE ? "ip" : "oop",
                reps, n0, n1, win);
    if (win)
    {
        vfft_destroy((vfft_plan)h0);
        return h1;
    }
    vfft_destroy((vfft_plan)h1);
    return h0;
}

#endif /* VFFT_TRANSFORMS_REAL_ZR2C_BUILD_H */
