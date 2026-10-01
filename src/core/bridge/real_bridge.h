/* real_bridge.h — the 1D r2c / c2r CREATE: where the two layouts still meet.
 *
 * BRIDGE, AND TEMPORARY (owner decision D1, 2026-09-27)
 * -----------------------------------------------------
 * An IL real engine for K>1 will be built; until then the crossings below live
 * here, the one place allowed to include both split/ and il/:
 *   - an interleaved request that zr2c cannot serve (K>1, or an out-of-place
 *     zr2c child failure) runs on the SPLIT real engines (CCE contract);
 *   - the odd-real bridge serves either layout through an IL c2c child;
 *   - the smooth-odd r2c race sets the split rfft handle against that bridge.
 * No new crossing may be added. The front door (vfft.c) routes every 1D real
 * request here before its layout fork.
 *
 *
 * WHAT THIS IS
 * ------------
 * The front dispatcher of the real-transform create. The arms, in order (each
 * returns on every path):
 *
 *   1. the ODD-N BRIDGE — odd N, K==1, out-of-place. Builds the transform on
 *      a c2c child (_oddr_build) rather than on a real codelet, and refuses
 *      LOUDLY when that child cannot be built. This arm is what serves odd
 *      and prime real sizes.
 *      r2c takes the bridge only when N is NOT radix-smooth (a smooth odd N is
 *      better served by rfft), while c2r takes it unconditionally — the two
 *      directions do not have the same incumbent.
 *      Odd interleaved K==1 cells enter through the ODD REAL RACE below:
 *      the mono and the real flat DIT stand against these routes.
 *   2. the interleaved tier (il/real/real_create_il.h): the zr2c route;
 *   3. the split tier (split/real/real_create_split.h): r2c and c2r, then the
 *      smooth-odd r2c bridge race (_real_oddr_race) on the r2c handle.
 *
 * BOTH REAL DIRECTIONS ARE A 2-AXIS CHOICE
 * ----------------------------------------
 * NATURAL (the fast packed cascade run on split input via the stage-0 natural
 * initiator — no repack, the low/mid-K winner) vs STRIDE (decoupled, the
 * high-K and threaded winner). Both consume split re/im, so the pick is
 * invisible to the caller.
 *
 * 🔴 THE CROSSOVER IS ON K, NOT N. natural's win is non-monotonic in K, so a
 * fixed threshold cannot capture it: at high rigor the tier MEASURES both arms
 * over the contested low/mid-K zone; otherwise it reads wisdom first and only
 * then falls back to a threshold. No forced path, no hardcode.
 *
 * WHAT THIS TIER DOES NOT DECIDE
 * ------------------------------
 * The route race itself lives in split/real/real_route_race.h — those are
 * racers, not deciders. This tier is what calls them and what banks the
 * verdict.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. It calls file-scope statics that live in vfft.c
 * (_oddr_build among them), so it must be included after those are defined and
 * before _vfft_create_inner.
 */
#ifndef VFFT_BRIDGE_REAL_BRIDGE_H
#define VFFT_BRIDGE_REAL_BRIDGE_H

/* the two arms of the smooth-odd bridge race: two finished handles */
typedef struct { struct vfft_plan_s *h; double *xr, *zr; } _oddr_arm_t;
static void _oddr_arm_exec(void *v)
{
    _oddr_arm_t *c = (_oddr_arm_t *)v;
    vfft_execute((vfft_plan)c->h, VFFT_FORWARD, c->xr, NULL, c->zr, NULL);
}
/* ── the tier's ONE exit. No shared post-step exists here (no mt gate: the
 * real engines thread internally; no pool arm: create-entry owns it) — the
 * finish exists so a shared step would land in one place and so each early
 * serving's skips are spelled at its return, not implied. */
static vfft_plan _vfft_real_bind_exec(vfft_plan hp); /* il/il_execute.h: the bound 1D real dispatch */
static vfft_plan _real_finish(struct vfft_plan_s *h)
{
    return _vfft_real_bind_exec((vfft_plan)h);
}

#include "il/real/real_create_il.h"
#include "split/real/real_create_split.h"

/* The smooth-odd r2c bridge race (D1, moves to bridge/ in phase 7): the
 * split-built rfft handle h against the IL c2c bridge. Returns the serving
 * handle (the loser is destroyed). */
static struct vfft_plan_s *_real_oddr_race(struct vfft_plan_s *h,
                                           const vfft_config_t *cfg,
                                           struct vfft_wisdom_s *W,
                                           int N, size_t K)
{
    /* SMOOTH-ODD r2c: race this (rfft-served) handle against the
     * c2c bridge - both arms FINISHED handles, min-of-3 alternated,
     * loser destroyed. The winner flips per cell. K==1 OOP IL only;
     * the verdict is banked (vw2_oddr_route) and replayed. */
    if (K == 1 && (N & 1) && N >= 3 &&
        cfg->placement == VFFT_OUTOFPLACE &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
        !getenv("VFFT_ODDR_NORACE"))
    {
        /* REPLAY the banked route: 1 = the rfft handle serves as
         * built, 2 = the bridge serves; only a miss (or recalibrate)
         * races. */
        const int banked = (W && !W->vw2_off_oop && !cfg->recalibrate)
                               ? vw2_oddr_route_lookup(&W->vw2, N) : 0;
        struct vfft_plan_s *hb = NULL;
        if (banked == 1)
        {
            if (getenv("VFFT_ODDR_LOG"))
                fprintf(stderr, "[oddr] N=%d: replay rfft src=wisdom\n", N);
            return h;
        }
        hb = _oddr_build(cfg, N);
        if (hb && banked == 2)
        {
            if (getenv("VFFT_ODDR_LOG"))
                fprintf(stderr, "[oddr] N=%d: replay bridge src=wisdom\n", N);
            vfft_destroy((vfft_plan)h);
            return hb;
        }
        if (hb)
        {
            const size_t hp1r = (size_t)N / 2 + 1;
            double *xr = (double *)malloc((size_t)N
                                          * sizeof(double));
            double *zr2 = (double *)calloc(2 * (hp1r + 8),
                                           sizeof(double));
            double ta = 1e300, tb2 = 1e300;
            if (xr && zr2)
            {
                int r2, j2;
                for (j2 = 0; j2 < N; j2++)
                    xr[j2] = 1.0 + 1e-6 * (double)(j2 & 511);
                vfft_execute((vfft_plan)h, VFFT_FORWARD, xr, NULL,
                             zr2, NULL);
                vfft_execute((vfft_plan)hb, VFFT_FORWARD, xr, NULL,
                             zr2, NULL);
                {
                    _oddr_arm_t ca = { h, xr, zr2 }, cb = { hb, xr, zr2 };
                    const vfft_race_arm_t arms[2] = {
                        { "rfft", _oddr_arm_exec, &ca },
                        { "bridge", _oddr_arm_exec, &cb } };
                    const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL }; /* min-of-3, A then B */
                    double ns[2];
                    (void)r2;
                    vfft_race_run(&proto, arms, 2, ns);
                    ta = ns[0];
                    tb2 = ns[1];
                }
                if (getenv("VFFT_ODDR_LOG"))
                    fprintf(stderr, "[oddr] race N=%d: rfft=%.0f "
                                    "bridge=%.0f -> %s\n",
                            N, ta, tb2,
                            tb2 < ta ? "BRIDGE" : "rfft");
            }
            free(xr);
            free(zr2);
            if (xr && zr2 && W && !W->vw2_off_oop)
            {   /* bank the verdict (the race ran) */
                if (vw2_oddr_route_bank(&W->vw2, N, tb2 < ta ? 2 : 1) == VW2_OK)
                    _vw2_persist(W, cfg);
            }
            if (tb2 < ta)
            {
                vfft_destroy((vfft_plan)h);
                return hb; /* the bridge won: it replaces the fully-built h */
            }
            vfft_destroy((vfft_plan)hb);
        }
    }
    return h;
}

static vfft_plan _vfft_create_real_routes(const vfft_config_t *cfg,
                                          vfft_batch ob,
                                          struct vfft_wisdom_s *W,
                                          const vfft_proto_registry_t *reg,
                                          int N,
                                          size_t K)
{
    if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
        K == 1 && (N & 1) && N >= 3 &&
        cfg->placement == VFFT_OUTOFPLACE &&
        (cfg->transform == VFFT_C2R || !vfft_is_radix_smooth(N) ||
         getenv("VFFT_ODDR_FORCE") != NULL))
    {
        struct vfft_plan_s *hh = _oddr_build(cfg, N);
        if (hh)
            return _real_finish(hh); /* odd-real direct: skips every lookup,
                                      * calibrate and route race below BY
                                      * DESIGN (self-contained c2c bridge) */
        _vfft_warn("vfft_create: %s odd N=%d - the c2c bridge child "
                   "could not be built; unsupported",
                   _vfft_tname(cfg->transform), N);
        return NULL;
    }
    /* the interleaved tier: the zr2c route (even N, K==1, no batch). c2r odd
     * N never matches the even-N gate (the c2r odd refusal stays in the split
     * tier, as before). */
    if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED && K == 1 && (N % 2) == 0 && !ob)
    {
        int refused = 0;
        struct vfft_plan_s *hz = _vfft_create_real_il(cfg, W, N, &refused);
        if (hz)
            return _real_finish(hz); /* zr2c serving: banks its own
                                      * kind-5 cell; the split-path
                                      * calibrates are for rows it
                                      * never reads — skipped BY DESIGN */
        if (refused)
            return NULL;
    }
    struct vfft_plan_s *h = _vfft_create_real_split(cfg, ob, W, reg, N, K);
    if (h && cfg->transform == VFFT_R2C)
        h = _real_oddr_race(h, cfg, W, N, K);
    return _real_finish(h);
}

/* THE ODD REAL RACE. Three IL engines stand against the odd-real routes above
 * (the split rfft, the c2c bridge):
 *   zrm  the real mono (il/real/zrm.h): one rn1 kernel, N <= 64, out of place;
 *   zrf  the real flat DIT (il/real/zrf.h): a real leaf, the c2c flat DIT's
 *        stages on the digit blocks, a mono at the bottom; every N whose
 *        factors are the leaf's radices, both placements. Its chain, its
 *        split-body switch and its tile budget are plan input: every chain
 *        is gated and burst-timed untiled, the two fastest again at every
 *        budget (a threaded plan: each budget serial and in both threaded
 *        arms, il/real/zrf_mt.h), and the two fastest plans of all join the
 *        race;
 *   zrb  the real Bluestein (il/real/zrb.h): the chirp-z convolution at
 *        M >= (3N-1)/2 on a matched IL inner pair; every odd N past the mono
 *        with no chain, both placements. Its length and its inner are plan
 *        input: a handful of lengths at or above the bound, at each the
 *        prime route's own inner pool, every candidate gated and
 *        burst-timed, the two fastest plans of all join the race.
 * The cell's verdict is banked in the real shard beside the routes' own
 * record (wisdom2_real_il.h; a different key from wisdom2_oddr.h's, so
 * neither bank touches the other). Even N races its engines inside the door
 * (il/real/zrp_build.h). Every arm is gated against the out-of-place routes'
 * output. VFFT_ZRM=1 pins the mono, VFFT_ZRM=0 keeps it out; VFFT_ZRF=chain
 * (e.g. 9.9.5, then "/t" = the split body off, "/w256" = the tile budget)
 * pins the flat DIT, VFFT_ZRF=0 keeps it out; VFFT_ZRB=M/kind:shape[/tW]
 * (e.g. 1152/ztt:4.4.8.9) pins the Bluestein, VFFT_ZRB=0 keeps it out. */
typedef struct { struct vfft_plan_s *h; const double *in; double *out; } _odd_arm_t;
static void _odd_arm_run(void *v)
{
    _odd_arm_t *c = (_odd_arm_t *)v;
    if (c->h->zrm)
        _exec_zrm(c->h, c->in, c->out);
    else if (c->h->zrf)
        _exec_zrf(c->h, c->in, c->out);
    else if (c->h->zrb)
        _exec_zrb(c->h, c->in, c->out);
    else if (c->h->zrbl)
        _exec_zrbl(c->h, c->in, c->out);
    else
        vfft_execute((vfft_plan)c->h, c->h->transform == VFFT_C2R ? VFFT_BACKWARD : VFFT_FORWARD,
                     (double *)c->in, NULL, c->out, NULL);
}

/* the flat DIT's sweep. One candidate: built, gated against ref, burst-timed
 * (best of five); it enters the two-slot podium or is destroyed. Returns 1
 * when it built (whatever became of it). s0 = the arm's input (b itself in
 * place). */
static int _zrf_try(const vfft_config_t *cfg, int N, const int *R, int K, int nomsz, int tile, int mt,
                    const double *a, const double *ref, double *b, const double *s0, size_t xs, size_t nchk,
                    struct vfft_plan_s *out[2], double bns[2], int *any_msz, int need_tiled)
{
    struct vfft_plan_s *h = _zrf_build_plan(cfg, N, R, K, nomsz, tile);
    double t0, est, best = 1e300;
    int reps;
    if (!h)
        return 0;
    if (need_tiled && !vfft_zrf_tiled(h->zrf))
    {   /* no level takes this budget: the untiled plan, already timed */
        vfft_destroy((vfft_plan)h);
        return 1;
    }
    if (mt && !vfft_zrf_mt_bind(h->zrf, h->nthreads, mt))
    {   /* the threaded arm declines this plan */
        vfft_destroy((vfft_plan)h);
        return 1;
    }
    if (any_msz)
        for (int j = 0; j < h->zrf->J; j++)
            for (int st = 1; st <= h->zrf->lv[j].ns; st++)
                *any_msz |= h->zrf->lv[j].fd->msz[st];
    memcpy(b, a, xs * sizeof(double));
    _exec_zrf(h, s0, b);
    {
        const double e = _zrpr_relerr(b, ref, nchk);
        if (e >= 1e-10)
        {
            char cs[48];
            vfft_zrf_chain_str(R, K, cs, sizeof cs);
            fprintf(stderr, "[zrf] N=%d chain %s%s/w%d%s FAILS the gate (rel %.2e vs the odd route) -- dropped\n",
                    N, cs, nomsz ? "/t" : "", tile, mt ? "/m" : "", e);
            vfft_destroy((vfft_plan)h);
            return 1;
        }
    }
    _exec_zrf(h, s0, b); /* warm (a threaded form: its workers too) */
    t0 = vfft_now_ns();
    _exec_zrf(h, s0, b);
    est = vfft_now_ns() - t0;
    reps = (int)(1.0e5 / (est > 1.0 ? est : 1.0));
    if (reps < 2) reps = 2;
    if (reps > 1024) reps = 1024;
    for (int r = 0; r < 5; r++)
    {
        double t = vfft_now_ns();
        for (int i = 0; i < reps; i++) _exec_zrf(h, s0, b);
        t = (vfft_now_ns() - t) / reps;
        if (t < best) best = t;
    }
    if (best < bns[0])
    {
        if (out[1]) vfft_destroy((vfft_plan)out[1]);
        out[1] = out[0]; bns[1] = bns[0];
        out[0] = h; bns[0] = best;
    }
    else if (best < bns[1])
    {
        if (out[1]) vfft_destroy((vfft_plan)out[1]);
        out[1] = h; bns[1] = best;
    }
    else
        vfft_destroy((vfft_plan)h);
    return 1;
}

/* every chain x the split-body switch untiled; then the two fastest at each
 * tile budget. The two fastest plans of all are returned as finished handles. */
#define VFFT_ZRF_MAX_CAND 48
static int _zrf_chain_sweep(const vfft_config_t *cfg, int N, const double *a, const double *ref, double *b,
                            const double *s0, size_t xs, size_t nchk, struct vfft_plan_s *out[2])
{
    static const int budgets[] = { 64, 128, 256, 512, 1024, 2048 };
    int ch[VFFT_ZRF_MAX_CAND][VFFT_ILFD_MAX_K], len[VFFT_ZRF_MAX_CAND], dropped = 0;
    const int nc = vfft_zrf_chains(N, ch, len, VFFT_ZRF_MAX_CAND, &dropped);
    double bns[2] = { 1e300, 1e300 };
    int seedR[2][VFFT_ILFD_MAX_K], seedK[2] = { 0, 0 }, seedm[2] = { 0, 0 };
    out[0] = out[1] = NULL;
    if (dropped && getenv("VFFT_ZRACE_VERBOSE"))
        fprintf(stderr, "[zrf] N=%d: chain pool capped at %d (%d more compositions not swept)\n",
                N, VFFT_ZRF_MAX_CAND, dropped);
    for (int c = 0; c < nc; c++)
    {
        int any = 0;
        if (!_zrf_try(cfg, N, ch[c], len[c], 0, 0, 0, a, ref, b, s0, xs, nchk, out, bns, &any, 0))
            continue;
        if (any) /* a stage takes the split body: its twin without it is another plan */
            _zrf_try(cfg, N, ch[c], len[c], 1, 0, 0, a, ref, b, s0, xs, nchk, out, bns, NULL, 0);
    }
    for (int i = 0; i < 2; i++)
        if (out[i])
        {
            seedK[i] = out[i]->zrf->K; seedm[i] = out[i]->zrf->nomsz;
            memcpy(seedR[i], out[i]->zrf->R, sizeof(int) * (size_t)seedK[i]);
        }
    for (int i = 0; i < 2; i++)
        for (int t = 0; seedK[i] && t < (int)(sizeof budgets / sizeof budgets[0]); t++)
        {
            _zrf_try(cfg, N, seedR[i], seedK[i], seedm[i], budgets[t], 0, a, ref, b, s0, xs, nchk, out, bns, NULL, 1);
            if (_vfft_plan_threads(cfg) > 1)
            {   /* the two threaded arms: FIRST and LEVELS (il/real/zrf_mt.h) */
                _zrf_try(cfg, N, seedR[i], seedK[i], seedm[i], budgets[t], 1, a, ref, b, s0, xs, nchk, out, bns, NULL, 1);
                _zrf_try(cfg, N, seedR[i], seedK[i], seedm[i], budgets[t], 2, a, ref, b, s0, xs, nchk, out, bns, NULL, 1);
            }
        }
    return (out[0] != NULL) + (out[1] != NULL);
}

/* the real Bluestein's handle: the length and the inner descriptor are plan
 * input (the descriptor is the prime route's, il/rank1/k1_commit.h). A
 * descriptor that does not build is a miss, never the structural rule's plan. */
static struct vfft_plan_s *_zrb_build_plan(const vfft_config_t *cfg, int N, int M, _ilprime_inner_desc_t *d)
{
    char kind[8], shape[64];
    vfft_zrb_plan_t *zp;
    struct vfft_plan_s *h;
    d->failed = 0;
    zp = vfft_zrb_create(N, M, _ilprime_inner_from_desc, d);
    if (zp && d->failed) { vfft_zrb_destroy(zp); zp = NULL; }
    if (!zp)
        return NULL;
    _ilprime_desc_str(d, kind, sizeof kind, shape, sizeof shape);
    vfft_zrb_name_inner(zp, kind, shape, d->tw);
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_zrb_destroy(zp);
        return NULL;
    }
    h->transform = cfg->transform;
    h->placement = cfg->placement;
    h->layout = (int)VFFT_LAYOUT_INTERLEAVED;
    h->N = N;
    h->K = 1;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zrb = zp;
    return h;
}

/* VFFT_ZRB at create: 1 = pinned at "M/kind:shape[/tW]" (e.g. 1152/ztt:4.4.8.9,
 * 2048/2p:32.64), 0 = kept out of the race, -1 = unset */
static int _zrb_env(int *M, _ilprime_inner_desc_t *d)
{
    const char *e = getenv("VFFT_ZRB");
    char kind[8], shape[64], *end;
    const char *s;
    size_t n;
    int tw = 0;
    *M = 0;
    if (!e || !e[0])
        return -1;
    if (e[0] == '0' && !e[1])
        return 0;
    *M = (int)strtol(e, &end, 10);
    if (end == e || *end != '/' || *M < 4)
        return 0;
    s = end + 1;
    n = strcspn(s, ":");
    if (!s[n] || n == 0 || n >= sizeof kind)
        return 0;
    memcpy(kind, s, n); kind[n] = 0;
    s += n + 1;
    n = strcspn(s, "/");
    if (n == 0 || n >= sizeof shape)
        return 0;
    memcpy(shape, s, n); shape[n] = 0;
    s += n;
    if (s[0] == '/' && s[1] == 't') { tw = atoi(s + 2); s += 2; while (*s >= '0' && *s <= '9') s++; }
    if (*s || tw < 0)
        return 0;
    return _ilprime_desc_parse(d, kind, shape, tw) ? 1 : 0;
}

/* THE REAL BLUESTEIN'S SWEEP. The lengths: the smallest power of two at or
 * above the bound, the smooth lengths 2^a*{3,5,7,9,15} below it (the three
 * smallest) and, within the pair engines' reach, the two smallest lengths
 * any inner builds; at each length the prime route's own inner pool
 * (_ilprime_inner_cands). Every candidate is built, gated against ref and
 * burst-timed (best of three); the two fastest plans of all are returned as
 * finished handles. */
#define VFFT_ZRB_MAX_M 6
static int _zrb_m_cands(int N, int *out)
{
    static const int q[5] = { 3, 5, 7, 9, 15 };
    static _ilprime_inner_desc_t pool[_ILPR_MAX_CANDS];   /* off the stack */
    const int mn = vfft_zrb_min_m(N);
    int sm[64], nsm = 0, n = 0, p2 = 16, i, a, k;
    while (p2 < mn) p2 <<= 1;
    for (i = 0; i < 5; i++)
        for (a = 2; (1 << a) * q[i] < p2 && nsm < 64; a++)
            if ((1 << a) * q[i] >= mn) sm[nsm++] = (1 << a) * q[i];
    for (i = 1; i < nsm; i++)   /* ascending */
        for (k = i; k > 0 && sm[k] < sm[k - 1]; k--) { const int t = sm[k]; sm[k] = sm[k - 1]; sm[k - 1] = t; }
    for (i = 0, k = 0; i < nsm && k < 3; i++)
        if (_ilprime_inner_cands(sm[i], pool, _ILPR_MAX_CANDS) > 0) { out[n++] = sm[i]; k++; }
    if (mn <= 4096)
    {   /* the pair engines' lengths: the two smallest that build, not already listed */
        int M;
        for (M = mn, k = 0; M < p2 && M <= 4096 && k < 2; M++)
        {
            int dup = 0;
            for (i = 0; i < n; i++) if (out[i] == M) dup = 1;
            if (dup) continue;
            if (_ilprime_inner_cands(M, pool, _ILPR_MAX_CANDS) > 0) { out[n++] = M; k++; }
        }
    }
    out[n++] = p2;
    return n;
}
static int _zrb_try(const vfft_config_t *cfg, int N, int M, _ilprime_inner_desc_t *d,
                    const double *a, const double *ref, double *b, const double *s0, size_t xs, size_t nchk,
                    struct vfft_plan_s *out[2], double bns[2])
{
    struct vfft_plan_s *h = _zrb_build_plan(cfg, N, M, d);
    double t0, est, best = 1e300;
    int reps;
    if (!h)
        return 0;
    memcpy(b, a, xs * sizeof(double));
    _exec_zrb(h, s0, b);
    {
        const double e = _zrpr_relerr(b, ref, nchk);
        if (e >= 1e-10)
        {
            char cs[96];
            vfft_zrb_str(h->zrb, cs, sizeof cs);
            fprintf(stderr, "[zrb] N=%d %s FAILS the gate (rel %.2e vs the odd route) -- dropped\n", N, cs, e);
            vfft_destroy((vfft_plan)h);
            return 1;
        }
    }
    _exec_zrb(h, s0, b);
    t0 = vfft_now_ns();
    _exec_zrb(h, s0, b);
    est = vfft_now_ns() - t0;
    reps = (int)(5.0e4 / (est > 1.0 ? est : 1.0));
    if (reps < 2) reps = 2;
    if (reps > 512) reps = 512;
    for (int r = 0; r < 3; r++)
    {
        double t = vfft_now_ns();
        for (int i = 0; i < reps; i++) _exec_zrb(h, s0, b);
        t = (vfft_now_ns() - t) / reps;
        if (t < best) best = t;
    }
    if (best < bns[0])
    {
        if (out[1]) vfft_destroy((vfft_plan)out[1]);
        out[1] = out[0]; bns[1] = bns[0];
        out[0] = h; bns[0] = best;
    }
    else if (best < bns[1])
    {
        if (out[1]) vfft_destroy((vfft_plan)out[1]);
        out[1] = h; bns[1] = best;
    }
    else
        vfft_destroy((vfft_plan)h);
    return 1;
}
static int _zrb_sweep(const vfft_config_t *cfg, int N, const double *a, const double *ref, double *b,
                      const double *s0, size_t xs, size_t nchk, struct vfft_plan_s *out[2])
{
    static _ilprime_inner_desc_t pool[_ILPR_MAX_CANDS];   /* off the stack */
    int Ms[VFFT_ZRB_MAX_M + 2];
    const int nm = _zrb_m_cands(N, Ms);
    double bns[2] = { 1e300, 1e300 };
    int built = 0;
    out[0] = out[1] = NULL;
    for (int m = 0; m < nm; m++)
    {
        const int nc = _ilprime_inner_cands(Ms[m], pool, _ILPR_MAX_CANDS);
        int nb = 0;
        for (int c = 0; c < nc; c++)
            nb += _zrb_try(cfg, N, Ms[m], &pool[c], a, ref, b, s0, xs, nchk, out, bns);
        built += nb;
        if (getenv("VFFT_ZRACE_VERBOSE"))
            fprintf(stderr, "[zrb] N=%d M=%d: %d of %d inner(s) built; best so far %.0f ns\n", N, Ms[m], nb, nc, bns[0]);
    }
    (void)built;
    return (out[0] != NULL) + (out[1] != NULL);
}

/* the race's incumbent: the odd-real routes out of place; in place the odd
 * bridge (aliasing-safe by construction: it copies the plane out before
 * anything writes back), as the front door serves it */
static struct vfft_plan_s *_odd_incumbent(const vfft_config_t *cfg, vfft_batch ob,
                                          struct vfft_wisdom_s *W, const vfft_proto_registry_t *reg,
                                          int N, size_t K)
{
    if (cfg->placement == VFFT_INPLACE)
    {
        struct vfft_plan_s *hh = _oddr_build(cfg, N);
        if (hh)
            hh->placement = VFFT_INPLACE;
        return hh;
    }
    return (struct vfft_plan_s *)_vfft_create_real_routes(cfg, ob, W, reg, N, K);
}

static vfft_plan _vfft_create_real_odd_race(const vfft_config_t *cfg, vfft_batch ob,
                                            struct vfft_wisdom_s *W, const vfft_proto_registry_t *reg,
                                            int N, size_t K)
{
    const int c2r = cfg->transform == VFFT_C2R, ip = cfg->placement == VFFT_INPLACE;
    const int Tk = _vfft_plan_threads(cfg);   /* the verdict's thread key */
    const int menv = _zrm_env();
    const int mono_ok = !ip && N <= VFFT_ZRM_MAX_N && menv != 0 && vfft_zrm_fn(N, c2r) != 0;
    int fR[VFFT_ILFD_MAX_K], fK = 0, fnomsz = 0, ftile = 0, fmt = 0;
    const int fenv = _zrf_env(fR, &fK, &fnomsz, &ftile, &fmt);
    _ilprime_inner_desc_t bd;
    int bM = 0;
    const int benv = _zrb_env(&bM, &bd);
    const int blue_ok = _zrb_ok(N) && benv != 0;
    struct vfft_plan_s *hi, *href, *arm[6];
    int narm = 0, rt_gate = 0;
    if (menv == 1 && mono_ok)
    {
        struct vfft_plan_s *hm = _zrm_build_plan(cfg, N);
        if (hm)
            return _real_finish(hm);
    }
    if (fenv == 1)
    {
        struct vfft_plan_s *hf = _zrf_build_plan(cfg, N, fR, fK, fnomsz, ftile);
        if (hf)
        {
            if (fmt) vfft_zrf_mt_bind(hf->zrf, hf->nthreads, fmt);
            return _real_finish(hf);
        }
    }
    if (benv == 1)
    {
        struct vfft_plan_s *hb = _zrb_build_plan(cfg, N, bM, &bd);
        if (hb)
            return _real_finish(hb);
    }
    if (W && !W->vw2_off_oop && !cfg->recalibrate)
    {
        int R1, R2, form;
        const char *eng = vw2_real_il_lookup(&W->vw2, N, c2r, ip, Tk, &R1, &R2, &form);
        if (eng && !strcmp(eng, "zrm") && mono_ok)
        {
            struct vfft_plan_s *hm = _zrm_build_plan(cfg, N);
            if (hm)
                return _real_finish(hm);
        }
        else if (eng && !strcmp(eng, "zrf") && fenv != 0)
        {
            int ch[VFFT_ILFD_MAX_K], ck = 0, cn = 0, ct = 0, cm = 0;
            if (vw2_real_il_lookup_zrf(&W->vw2, N, c2r, ip, Tk, ch, VFFT_ILFD_MAX_K, &ck, &cn, &ct, &cm))
            {
                struct vfft_plan_s *hf = _zrf_build_plan(cfg, N, ch, ck, cn, ct);
                if (hf)
                {
                    if (cm) vfft_zrf_mt_bind(hf->zrf, hf->nthreads, cm); /* the banked threaded arm, at the row's T */
                    return _real_finish(hf);
                }
            }
        }
        else if (eng && !strcmp(eng, "zrb") && benv != 0)
        {
            char kind[8], shape[64];
            int M = 0, tw = 0;
            if (vw2_real_il_lookup_zrb(&W->vw2, N, c2r, ip, Tk, &M, kind, sizeof kind, shape, sizeof shape, &tw) &&
                _ilprime_desc_parse(&bd, kind, shape, tw))
            {
                struct vfft_plan_s *hb = _zrb_build_plan(cfg, N, M, &bd);
                if (hb)
                    return _real_finish(hb);
            }
        }
        else if (eng)
            return (vfft_plan)_odd_incumbent(cfg, ob, W, reg, N, K); /* decided: the routes replay their own record */
    }
    hi = _odd_incumbent(cfg, ob, W, reg, N, K); /* finished */
    if (!W || W->vw2_off_oop)
        return (vfft_plan)hi;
    /* the gate's reference: the out-of-place routes' output (in place, a second
     * handle). A c2r cell the odd routes cannot build (their c2c child stops
     * short of the largest N) is gated by the ROUNDTRIP through the r2c routes:
     * the candidate's input is that plan's spectrum of x, the reference N x. */
    href = hi;
    if (ip)
    {
        vfft_config_t co = *cfg;
        co.placement = VFFT_OUTOFPLACE;
        href = (struct vfft_plan_s *)_vfft_create_real_routes(&co, ob, W, reg, N, K);
    }
    if (!href && c2r)
    {
        vfft_config_t co = *cfg;
        co.placement = VFFT_OUTOFPLACE;
        co.transform = VFFT_R2C;
        href = (struct vfft_plan_s *)_vfft_create_real_routes(&co, ob, W, reg, N, K);
        rt_gate = href != NULL;
    }
    if (!href)
        return (vfft_plan)hi;
    {
        const size_t xs = (size_t)N + 3;
        const size_t nchk = c2r ? (size_t)N : (size_t)N + 1; /* odd N: (N+1)/2 bins, N reals */
        double *a = (double *)vfft_aligned_alloc(xs * sizeof(double));
        double *b = (double *)vfft_aligned_alloc(xs * sizeof(double));
        double *ref = (double *)vfft_aligned_alloc(xs * sizeof(double));
        const double *s0 = ip ? b : a;
        struct vfft_plan_s *hz[2] = { NULL, NULL };
        double ns[4], t0, est;
        int reps, win = 0, a0;
        if (!a || !b || !ref)
        {
            vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
            if (href != hi) vfft_destroy((vfft_plan)href);
            return (vfft_plan)hi;
        }
        {
            unsigned sd = 0x9e3779b9u ^ (unsigned)N ^ (unsigned)(c2r << 8);
            for (size_t i = 0; i < xs; i++)
            {
                sd = sd * 1664525u + 1013904223u;
                a[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
            }
        }
        if (c2r)
            a[1] = 0.0; /* a CCE spectrum: real DC (odd N has no Nyquist bin) */
        memset(ref, 0, xs * sizeof(double));
        memset(b, 0, xs * sizeof(double));
        if (rt_gate)
        {   /* ref = N x over the reals a holds; a becomes the r2c routes' spectrum of x */
            for (size_t i = 0; i < (size_t)N; i++) ref[i] = (double)N * a[i];
            vfft_execute((vfft_plan)href, VFFT_FORWARD, a, NULL, b, NULL);
            memcpy(a, b, ((size_t)N + 1) * sizeof(double));
            a[(size_t)N + 1] = a[(size_t)N + 2] = 0.0;
            memset(b, 0, xs * sizeof(double));
        }
        else
            vfft_execute((vfft_plan)href, c2r ? VFFT_BACKWARD : VFFT_FORWARD, a, NULL, ref, NULL);
        if (href != hi) vfft_destroy((vfft_plan)href);
        if (hi)
            arm[narm++] = hi;
        a0 = narm;
        if (mono_ok)
        {
            struct vfft_plan_s *hm = _zrm_build_plan(cfg, N);
            if (hm)
            {
                _exec_zrm(hm, a, b);
                const double e = _zrpr_relerr(b, ref, nchk);
                if (e >= 1e-10)
                {
                    fprintf(stderr, "[zrm] N=%d %s oop the real mono FAILS the gate (rel %.2e vs the odd route) -- dropped\n",
                            N, c2r ? "c2r" : "r2c", e);
                    vfft_destroy((vfft_plan)hm);
                }
                else
                    arm[narm++] = hm;
            }
        }
        if (fenv != 0 && _zrf_chain_sweep(cfg, N, a, ref, b, s0, xs, nchk, hz) > 0)
        {
            arm[narm++] = hz[0];
            if (hz[1]) arm[narm++] = hz[1];
        }
        if (blue_ok)
        {
            struct vfft_plan_s *hb[2] = { NULL, NULL };
            if (_zrb_sweep(cfg, N, a, ref, b, s0, xs, nchk, hb) > 0)
            {
                arm[narm++] = hb[0];
                if (hb[1]) arm[narm++] = hb[1];
            }
        }
        if (narm == a0)
        {   /* no challenger built: the routes stand, nothing to bank */
            vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
            return (vfft_plan)hi;
        }
        _vfft_create_race_count++;
        {
            _odd_arm_t ca[6];
            vfft_race_arm_t arms[6];
            static const char *nm[2] = { "zrf", "zrf2" }, *nb[2] = { "zrb", "zrb2" };
            int nz = 0, nbl = 0;
            for (int i = 0; i < narm; i++)
            {
                ca[i].h = arm[i]; ca[i].in = s0; ca[i].out = b;
                arms[i].name = (i < a0) ? "oddr" : arm[i]->zrm ? "zrm" : arm[i]->zrb ? nb[nbl++ & 1] : nm[nz++ & 1];
                arms[i].run = _odd_arm_run; arms[i].ctx = &ca[i];
            }
            memcpy(b, a, xs * sizeof(double));
            t0 = vfft_now_ns();
            _odd_arm_run(&ca[0]);
            est = vfft_now_ns() - t0;
            reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
            if (reps < 2) reps = 2;
            if (reps > 4096) reps = 4096;
            {
                /* a threaded plan's arms are never paused: two warm passes, no pacing */
                const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, Tk > 1 ? 2 : 1, NULL, NULL, Tk > 1 ? 0 : 1 };
                vfft_race_run(&proto, arms, narm, ns);
            }
        }
        /* the fastest challenger; it takes the cell from the routes past a 3% hysteresis */
        win = a0;
        for (int i = a0 + 1; i < narm; i++)
            if (ns[i] < ns[win]) win = i;
        if (a0 && !vfft_race_beats(ns[win], ns[0], 0.97))
            win = 0;
        if (getenv("VFFT_ZRACE_VERBOSE"))
        {
            fprintf(stderr, "[odd] N=%d %s %s race: reps=%d hyst=3%% |", N, c2r ? "c2r" : "r2c", ip ? "ip" : "oop", reps);
            for (int i = 0; i < narm; i++)
            {
                char cs[96] = "", ws[16] = "";
                if (arm[i]->zrf)
                {
                    vfft_zrf_chain_str(arm[i]->zrf->R, arm[i]->zrf->K, cs, sizeof cs);
                    snprintf(ws, sizeof ws, "/w%d%s", arm[i]->zrf->tile, arm[i]->zrf->mt == 2 ? "/m2" : arm[i]->zrf->mt ? "/m1" : "");
                }
                else if (arm[i]->zrb)
                    vfft_zrb_str(arm[i]->zrb, cs, sizeof cs);
                fprintf(stderr, " %s%s%s%s=%.0f%s", i < a0 ? "oddr" : arm[i]->zrm ? "zrm" : arm[i]->zrb ? "zrb:" : "zrf:", cs,
                        (arm[i]->zrf && arm[i]->zrf->nomsz) ? "/t" : "", ws, ns[i], i == win ? "*" : "");
            }
            fprintf(stderr, "\n");
        }
        vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
        {
            struct vfft_plan_s *hw = arm[win];
            const int rc = hw->zrf ? vw2_real_il_bank_zrf(&W->vw2, N, c2r, ip, Tk, hw->zrf->R, hw->zrf->K, hw->zrf->nomsz,
                                                          hw->zrf->tile, hw->zrf->mt, ns[win])
                         : hw->zrb ? vw2_real_il_bank_zrb(&W->vw2, N, c2r, ip, Tk, hw->zrb->M, hw->zrb->ikind, hw->zrb->ishape,
                                                          hw->zrb->itw, ns[win])
                         : hw->zrm ? vw2_real_il_bank_zrm(&W->vw2, N, c2r, ip, Tk, ns[win])
                                   : vw2_real_il_bank_eng(&W->vw2, N, c2r, ip, Tk, "oddr", ns[win]);
            if (rc == VW2_OK)
                _vw2_persist(W, cfg);
            else
                fprintf(stderr, "vfft: real engine verdict NOT banked at odd N=%d (rc=%d) -- the cell will re-race\n", N, rc);
            for (int i = 0; i < narm; i++)
                if (i != win) vfft_destroy((vfft_plan)arm[i]);
            return win < a0 ? (vfft_plan)hw : _real_finish(hw);
        }
    }
}

/* THE LANE BLUESTEIN'S CREATE (il/real/zrb_lanes.h): the interleaved real
 * batch in its lane-major geometry, K > 1, out of place, at an odd N without
 * a chain. The plan inputs -- the length M, the column chain at M with its
 * forms, the column window -- are swept at create: the column pool's lengths
 * at or above the bound (the smallest power of two, the three smallest
 * smooth lengths, the two smallest with a chain), every chain the enumerator
 * yields at each (the no-silent-caps law is the enumerator's), the window
 * ladder on the two fastest; every candidate gated against K one-lane
 * transforms through the odd routes and burst-timed; the two fastest race
 * (the library body, paced) and the winner is banked on the cell's q=K row.
 * VFFT_ZRBL=0 keeps it out; NULL = nothing built (the request goes where it
 * went). */
static struct vfft_plan_s *_zrbl_build_plan(const vfft_config_t *cfg, int N, int K, int M,
                                            const int *Rs, int nst, const char *forms, int wc)
{
    vfft_zrbl_plan_t *zp = vfft_zrbl_create(N, K, M, Rs, nst, forms, wc);
    struct vfft_plan_s *h;
    if (!zp)
        return NULL;
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_zrbl_destroy(zp);
        return NULL;
    }
    h->transform = cfg->transform;
    h->placement = VFFT_OUTOFPLACE;
    h->layout = (int)VFFT_LAYOUT_INTERLEAVED;
    h->N = N;
    h->K = K;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zrbl = zp;
    return h;
}
static int _zrbl_nchains(int M, int (*ch)[8], int *lens, int warn)
{
    int cur[8], n = 0, dropped = 0;
    memset(cur, 0, sizeof cur);
    if (warn) _il2d_enum_rec(M, 0, cur, ch, lens, &n, &dropped);
    else _il2d_enum_rec_body(M, 0, cur, ch, lens, &n, &dropped);
    return n;
}
static int _zrbl_m_cands(int N, int *out)
{
    static const int q[5] = { 3, 5, 7, 9, 15 };
    static int ch[VFFT_IL2D_MAXCAND][8], lens[VFFT_IL2D_MAXCAND];
    const int mn = vfft_zrb_min_m(N);
    int sm[64], nsm = 0, n = 0, p2 = 16, i, a, k, M;
    while (p2 < mn) p2 <<= 1;
    for (i = 0; i < 5; i++)
        for (a = 2; (1 << a) * q[i] < p2 && nsm < 64; a++)
            if ((1 << a) * q[i] >= mn) sm[nsm++] = (1 << a) * q[i];
    for (i = 1; i < nsm; i++)
        for (k = i; k > 0 && sm[k] < sm[k - 1]; k--) { const int t = sm[k]; sm[k] = sm[k - 1]; sm[k - 1] = t; }
    for (i = 0, k = 0; i < nsm && k < 3; i++)
        if (_zrbl_nchains(sm[i], ch, lens, 0) > 0) { out[n++] = sm[i]; k++; }
    for (M = mn, k = 0; M < p2 && M <= 65536 && k < 2; M++)
    {
        int dup = 0;
        for (i = 0; i < n; i++) if (out[i] == M) dup = 1;
        if (dup) continue;
        if (_zrbl_nchains(M, ch, lens, 0) > 0) { out[n++] = M; k++; }
    }
    out[n++] = p2;
    return n;
}
static int _zrbl_try(const vfft_config_t *cfg, int N, int K, int M, const int *Rs, int nst, const char *forms, int wc,
                     const double *a, const double *ref, double *b, size_t nout,
                     struct vfft_plan_s *out[2], double bns[2])
{
    struct vfft_plan_s *h = _zrbl_build_plan(cfg, N, K, M, Rs, nst, forms, wc);
    double t0, est, best = 1e300;
    int reps;
    if (!h)
        return 0;
    memset(b, 0, nout * sizeof(double));
    _exec_zrbl(h, a, b);
    {
        const double e = _zrpr_relerr(b, ref, nout);
        if (e >= 1e-10)
        {
            char cs[128];
            vfft_zrbl_str(h->zrbl, cs, sizeof cs);
            fprintf(stderr, "[zrbl] N=%d K=%d %s FAILS the gate (rel %.2e vs the odd routes) -- dropped\n", N, K, cs, e);
            vfft_destroy((vfft_plan)h);
            return 1;
        }
    }
    _exec_zrbl(h, a, b);
    t0 = vfft_now_ns();
    _exec_zrbl(h, a, b);
    est = vfft_now_ns() - t0;
    reps = (int)(1.0e5 / (est > 1.0 ? est : 1.0));
    if (reps < 2) reps = 2;
    if (reps > 256) reps = 256;
    for (int r = 0; r < 3; r++)
    {
        double t = vfft_now_ns();
        for (int i = 0; i < reps; i++) _exec_zrbl(h, a, b);
        t = (vfft_now_ns() - t) / reps;
        if (t < best) best = t;
    }
    if (best < bns[0])
    {
        if (out[1]) vfft_destroy((vfft_plan)out[1]);
        out[1] = out[0]; bns[1] = bns[0];
        out[0] = h; bns[0] = best;
    }
    else if (best < bns[1])
    {
        if (out[1]) vfft_destroy((vfft_plan)out[1]);
        out[1] = h; bns[1] = best;
    }
    else
        vfft_destroy((vfft_plan)h);
    return 1;
}
static int _zrbl_sweep(const vfft_config_t *cfg, int N, int K, const double *a, const double *ref, double *b,
                       size_t nout, struct vfft_plan_s *out[2])
{
    static const int wins[6] = { 2, 4, 8, 16, 32, 64 };
    static int ch[VFFT_IL2D_MAXCAND][8], lens[VFFT_IL2D_MAXCAND];
    int Ms[VFFT_ZRB_MAX_M + 2];
    const int nm = _zrbl_m_cands(N, Ms);
    double bns[2] = { 1e300, 1e300 };
    int sR[2][8], sn[2] = { 0, 0 }, sM[2] = { 0, 0 };
    char sf[2][64];
    out[0] = out[1] = NULL;
    for (int m = 0; m < nm; m++)
    {
        const int nc = _zrbl_nchains(Ms[m], ch, lens, 1);
        int nb = 0;
        for (int c = 0; c < nc; c++)
            nb += _zrbl_try(cfg, N, K, Ms[m], ch[c], lens[c], "", 0, a, ref, b, nout, out, bns);
        if (getenv("VFFT_ZRACE_VERBOSE"))
            fprintf(stderr, "[zrbl] N=%d K=%d M=%d: %d of %d chain(s) built; best so far %.0f ns\n", N, K, Ms[m], nb, nc, bns[0]);
    }
    for (int i = 0; i < 2; i++)
        if (out[i])
        {
            sn[i] = out[i]->zrbl->nst; sM[i] = out[i]->zrbl->M;
            memcpy(sR[i], out[i]->zrbl->Rs, sizeof(int) * (size_t)sn[i]);
            snprintf(sf[i], sizeof sf[i], "%s", out[i]->zrbl->forms);
        }
    for (int i = 0; i < 2; i++)
        for (int w = 0; sn[i] && w < 6 && wins[w] < K; w++)
            _zrbl_try(cfg, N, K, sM[i], sR[i], sn[i], sf[i], wins[w], a, ref, b, nout, out, bns);
    return (out[0] != NULL) + (out[1] != NULL);
}
static vfft_plan _vfft_create_real_odd_race(const vfft_config_t *cfg, vfft_batch ob, struct vfft_wisdom_s *W,
                                            const vfft_proto_registry_t *reg, int N, size_t K); /* below */
static vfft_plan _vfft_create_real_lanes(const vfft_config_t *cfg, struct vfft_wisdom_s *W,
                                         const vfft_proto_registry_t *reg, int N, int K)
{
    const int c2r = cfg->transform == VFFT_C2R;
    const int Tk = _vfft_plan_threads(cfg);
    const size_t hp1 = (size_t)N / 2 + 1, nin = c2r ? 2 * hp1 * (size_t)K : (size_t)N * (size_t)K;
    const size_t nout = c2r ? (size_t)N * (size_t)K : 2 * hp1 * (size_t)K;
    const char *e = getenv("VFFT_ZRBL");
    struct vfft_plan_s *href, *hz[2] = { NULL, NULL };
    double *a, *ref, *b, *ti, *to;
    vfft_config_t c1;
    if (e && e[0] == '0' && !e[1])
        return NULL;
    if (W && !W->vw2_off_oop && !cfg->recalibrate)
    {
        int M = 0, Rs[8], nst = 0, wc = 0, tw = 0;
        char forms[64], kind[8], shape[64];
        _ilprime_inner_desc_t d;
        if (vw2_real_il_lookup_zrbl(&W->vw2, N, K, c2r, 0, Tk, &M, Rs, 8, &nst, forms, sizeof forms, &wc))
        {
            struct vfft_plan_s *h = _zrbl_build_plan(cfg, N, K, M, Rs, nst, forms, wc);
            if (h)
                return _real_finish(h);
        }
        if (vw2_real_il_lookup_zrb_q(&W->vw2, N, K, c2r, 0, Tk, &M, kind, sizeof kind, shape, sizeof shape, &tw) &&
            _ilprime_desc_parse(&d, kind, shape, tw))
        {   /* the one-row engine over the lanes, its edges at the lane stride */
            struct vfft_plan_s *h = _zrb_build_plan(cfg, N, M, &d);
            if (h)
            {
                h->K = K;
                return _real_finish(h);
            }
        }
    }
    if (!W || W->vw2_off_oop)
        return NULL;
    /* the reference: K one-lane transforms through the odd routes */
    c1 = *cfg;
    c1.howmany = 1;
    c1.batch_geom = VFFT_BATCH_DEFAULT;
    c1.placement = VFFT_OUTOFPLACE;
    href = _oddr_build(&c1, N);
    if (!href)
        return NULL;
    a = (double *)vfft_aligned_alloc((nin + 8) * sizeof(double));
    ref = (double *)vfft_aligned_alloc((nout + 8) * sizeof(double));
    b = (double *)vfft_aligned_alloc((nout + 8) * sizeof(double));
    ti = (double *)vfft_aligned_alloc((2 * hp1 + 8) * sizeof(double));
    to = (double *)vfft_aligned_alloc((2 * hp1 + 8) * sizeof(double));
    if (!a || !ref || !b || !ti || !to)
    {
        vfft_aligned_free(a); vfft_aligned_free(ref); vfft_aligned_free(b); vfft_aligned_free(ti); vfft_aligned_free(to);
        vfft_destroy((vfft_plan)href);
        return NULL;
    }
    {
        unsigned sd = 0x9e3779b9u ^ (unsigned)N ^ (unsigned)(K << 12) ^ (unsigned)(c2r << 8);
        for (size_t i = 0; i < nin; i++)
        {
            sd = sd * 1664525u + 1013904223u;
            a[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
        }
        if (c2r)
            for (int t = 0; t < K; t++) a[2 * t + 1] = 0.0;   /* a CCE spectrum: real DC in every lane */
        memset(ref, 0, nout * sizeof(double));
        for (int t = 0; t < K; t++)
        {   /* lane t out, through the one-lane routes, back into its lane */
            if (c2r)
            {
                for (size_t f = 0; f < hp1; f++) { ti[2 * f] = a[2 * (f * (size_t)K + t)]; ti[2 * f + 1] = a[2 * (f * (size_t)K + t) + 1]; }
                vfft_execute((vfft_plan)href, VFFT_BACKWARD, ti, NULL, to, NULL);
                for (size_t n = 0; n < (size_t)N; n++) ref[n * (size_t)K + t] = to[n];
            }
            else
            {
                for (size_t n = 0; n < (size_t)N; n++) ti[n] = a[n * (size_t)K + t];
                vfft_execute((vfft_plan)href, VFFT_FORWARD, ti, NULL, to, NULL);
                for (size_t f = 0; f < hp1; f++) { ref[2 * (f * (size_t)K + t)] = to[2 * f]; ref[2 * (f * (size_t)K + t) + 1] = to[2 * f + 1]; }
            }
        }
    }
    vfft_destroy((vfft_plan)href);
    vfft_aligned_free(ti); vfft_aligned_free(to);
    {
        /* THE ARMS: the column form's podium (two), and the one-row engine
         * over the lanes with its edges at the lane stride -- built from the
         * cell's one-row verdict (raced now when the q=1 row is a miss) */
        struct vfft_plan_s *arm[3];
        double ns[3];
        int narm = 0, win = 0;
        if (_zrbl_sweep(cfg, N, K, a, ref, b, nout, hz) > 0)
        {
            arm[narm++] = hz[0];
            if (hz[1]) arm[narm++] = hz[1];
        }
        {
            struct vfft_plan_s *h1 = (struct vfft_plan_s *)_vfft_create_real_odd_race(&c1, NULL, W, reg, N, 1);
            if (h1 && h1->zrb)
            {
                _ilprime_inner_desc_t d;
                if (_ilprime_desc_parse(&d, h1->zrb->ikind, h1->zrb->ishape, h1->zrb->itw))
                {
                    struct vfft_plan_s *hk = _zrb_build_plan(cfg, N, h1->zrb->M, &d);
                    if (hk)
                    {
                        hk->K = K;
                        memset(b, 0, nout * sizeof(double));
                        _exec_zrb(hk, a, b);
                        if (_zrpr_relerr(b, ref, nout) < 1e-10)
                            arm[narm++] = hk;
                        else
                        {
                            fprintf(stderr, "[zrb] N=%d K=%d lanes FAIL the gate -- dropped\n", N, K);
                            vfft_destroy((vfft_plan)hk);
                        }
                    }
                }
            }
            if (h1) vfft_destroy((vfft_plan)h1);
        }
        if (narm == 0)
        {
            vfft_aligned_free(a); vfft_aligned_free(ref); vfft_aligned_free(b);
            _vfft_warn("vfft_create: %s N=%d K=%d lane-major: no IL batch engine built at this cell -- the request goes to the routes",
                       _vfft_tname(cfg->transform), N, K);
            return NULL;
        }
        _vfft_create_race_count++;
        if (narm > 1)
        {
            _odd_arm_t ca[3];
            vfft_race_arm_t arms[3];
            double t0, est;
            int reps;
            for (int i = 0; i < narm; i++)
            {
                ca[i].h = arm[i]; ca[i].in = a; ca[i].out = b;
                arms[i].name = arm[i]->zrbl ? (i ? "zrbl2" : "zrbl") : "zrbK";
                arms[i].run = _odd_arm_run; arms[i].ctx = &ca[i];
            }
            t0 = vfft_now_ns();
            _odd_arm_run(&ca[0]);
            est = vfft_now_ns() - t0;
            reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
            if (reps < 2) reps = 2;
            if (reps > 1024) reps = 1024;
            {
                const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
                vfft_race_run(&proto, arms, narm, ns);
            }
            for (int i = 1; i < narm; i++)
                if (ns[i] < ns[win]) win = i;
        }
        else
            ns[0] = 0.0;
        if (getenv("VFFT_ZRACE_VERBOSE"))
        {
            fprintf(stderr, "[lanes] N=%d K=%d %s race:", N, K, c2r ? "c2r" : "r2c");
            for (int i = 0; i < narm; i++)
            {
                char cs[128];
                if (arm[i]->zrbl) vfft_zrbl_str(arm[i]->zrbl, cs, sizeof cs);
                else vfft_zrb_str(arm[i]->zrb, cs, sizeof cs);
                fprintf(stderr, " %s:%s=%.0f%s", arm[i]->zrbl ? "zrbl" : "zrbK", cs, ns[i], i == win ? "*" : "");
            }
            fprintf(stderr, "\n");
        }
        vfft_aligned_free(a); vfft_aligned_free(ref); vfft_aligned_free(b);
        {
            struct vfft_plan_s *hw = arm[win];
            const int rc = hw->zrbl
                ? vw2_real_il_bank_zrbl(&W->vw2, N, K, c2r, 0, Tk, hw->zrbl->M, hw->zrbl->Rs, hw->zrbl->nst,
                                        hw->zrbl->forms, hw->zrbl->wc, ns[win])
                : vw2_real_il_bank_zrb_q(&W->vw2, N, K, c2r, 0, Tk, hw->zrb->M, hw->zrb->ikind, hw->zrb->ishape,
                                         hw->zrb->itw, ns[win]);
            if (rc == VW2_OK)
                _vw2_persist(W, cfg);
            else
                fprintf(stderr, "vfft: lane verdict NOT banked at N=%d K=%d (rc=%d) -- the cell will re-race\n", N, K, rc);
            for (int i = 0; i < narm; i++)
                if (i != win) vfft_destroy((vfft_plan)arm[i]);
            return _real_finish(hw);
        }
    }
}

static vfft_plan _vfft_create_real(const vfft_config_t *cfg,
                                   vfft_batch ob,
                                   struct vfft_wisdom_s *W,
                                   const vfft_proto_registry_t *reg,
                                   int N,
                                   size_t K)
{
    if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED && K > 1 && !ob && (N & 1) && N >= 3 &&
        cfg->placement == VFFT_OUTOFPLACE &&
        (cfg->batch_geom == VFFT_BATCH_DEFAULT || cfg->batch_geom == VFFT_BATCH_LANE_MAJOR) &&
        _zrb_ok(N))
    {
        /* the real batch in its lane-major geometry at an odd N without a
         * chain: the lane Bluestein (il/real/zrb_lanes.h), a native IL engine */
        vfft_plan h = _vfft_create_real_lanes(cfg, W, reg, N, (int)K);
        if (h)
            return h;
    }
    if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED && K == 1 && !ob && (N & 1) && N >= 3)
    {
        /* an odd cell with an IL real engine: the mono's (N <= 64, out of
         * place), the flat DIT's (a chain exists) or the Bluestein's (past
         * the mono, no chain) */
        if ((N <= VFFT_ZRM_MAX_N && cfg->placement == VFFT_OUTOFPLACE && vfft_zrm_fn(N, cfg->transform == VFFT_C2R)) ||
            _zrf_has_chain(N) || _zrb_ok(N))
            return _vfft_create_real_odd_race(cfg, ob, W, reg, N, K);
    }
    return _vfft_create_real_routes(cfg, ob, W, reg, N, K);
}

#endif /* VFFT_BRIDGE_REAL_BRIDGE_H */
