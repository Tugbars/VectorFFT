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

/* THE ODD REAL RACE. Two IL engines stand against the odd-real routes above
 * (the split rfft, the c2c bridge):
 *   zrm  the real mono (il/real/zrm.h): one rn1 kernel, N <= 64, out of place;
 *   zrf  the real flat DIT (il/real/zrf.h): a real leaf, the c2c flat DIT's
 *        stages on the digit blocks, a mono at the bottom; every N whose
 *        factors are the leaf's radices, both placements. Its chain and its
 *        split-body switch are plan input: every chain is gated and
 *        burst-timed, the two fastest join the race.
 * The cell's verdict is banked in the real shard beside the routes' own
 * record (wisdom2_real_il.h; a different key from wisdom2_oddr.h's, so
 * neither bank touches the other). Even N races its engines inside the door
 * (il/real/zrp_build.h). Every arm is gated against the out-of-place routes'
 * output. VFFT_ZRM=1 pins the mono, VFFT_ZRM=0 keeps it out; VFFT_ZRF=chain
 * (e.g. 9.9.5, "/t" = the split body off) pins the flat DIT, VFFT_ZRF=0
 * keeps it out. */
typedef struct { struct vfft_plan_s *h; const double *in; double *out; } _odd_arm_t;
static void _odd_arm_run(void *v)
{
    _odd_arm_t *c = (_odd_arm_t *)v;
    if (c->h->zrm)
        _exec_zrm(c->h, c->in, c->out);
    else if (c->h->zrf)
        _exec_zrf(c->h, c->in, c->out);
    else
        vfft_execute((vfft_plan)c->h, c->h->transform == VFFT_C2R ? VFFT_BACKWARD : VFFT_FORWARD,
                     (double *)c->in, NULL, c->out, NULL);
}

/* the flat DIT's sweep: every chain x the split-body switch, each gated
 * against ref, burst-timed (best of five); the two fastest are returned as
 * finished handles. s0 = the arm's input (b itself in place). */
#define VFFT_ZRF_MAX_CAND 24
static int _zrf_chain_sweep(const vfft_config_t *cfg, int N, const double *a, const double *ref, double *b,
                            const double *s0, size_t xs, size_t nchk, struct vfft_plan_s *out[2])
{
    int ch[VFFT_ZRF_MAX_CAND][VFFT_ILFD_MAX_K], len[VFFT_ZRF_MAX_CAND], dropped = 0;
    const int nc = vfft_zrf_chains(N, ch, len, VFFT_ZRF_MAX_CAND, &dropped);
    double bns[2] = { 1e300, 1e300 };
    int n = 0;
    out[0] = out[1] = NULL;
    if (dropped && getenv("VFFT_ZRACE_VERBOSE"))
        fprintf(stderr, "[zrf] N=%d: chain pool capped at %d (%d more compositions not swept)\n",
                N, VFFT_ZRF_MAX_CAND, dropped);
    for (int c = 0; c < nc; c++)
        for (int nomsz = 0, any = 0; nomsz < 2; nomsz++)
        {
            struct vfft_plan_s *h;
            double t0, est, best = 1e300;
            int reps;
            if (nomsz && !any)
                continue; /* no stage takes the split body: the twin is the same plan */
            h = _zrf_build_plan(cfg, N, ch[c], len[c], nomsz);
            if (!h)
                break;
            if (!nomsz)
                for (int j = 0; j < h->zrf->J; j++)
                    for (int st = 1; st <= h->zrf->lv[j].ns; st++)
                        any |= h->zrf->lv[j].fd->msz[st];
            memcpy(b, a, xs * sizeof(double));
            _exec_zrf(h, s0, b);
            {
                const double e = _zrpr_relerr(b, ref, nchk);
                if (e >= 1e-10)
                {
                    char cs[48];
                    vfft_zrf_chain_str(ch[c], len[c], cs, sizeof cs);
                    fprintf(stderr, "[zrf] N=%d chain %s%s FAILS the gate (rel %.2e vs the odd route) -- dropped\n",
                            N, cs, nomsz ? "/t" : "", e);
                    vfft_destroy((vfft_plan)h);
                    continue;
                }
            }
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
        }
    if (out[0]) n++;
    if (out[1]) n++;
    return n;
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
    int fR[VFFT_ILFD_MAX_K], fK = 0, fnomsz = 0;
    const int fenv = _zrf_env(fR, &fK, &fnomsz);
    struct vfft_plan_s *hi, *href, *arm[4];
    int narm = 0;
    if (menv == 1 && mono_ok)
    {
        struct vfft_plan_s *hm = _zrm_build_plan(cfg, N);
        if (hm)
            return _real_finish(hm);
    }
    if (fenv == 1)
    {
        struct vfft_plan_s *hf = _zrf_build_plan(cfg, N, fR, fK, fnomsz);
        if (hf)
            return _real_finish(hf);
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
            int ch[VFFT_ILFD_MAX_K], ck = 0, cn = 0;
            if (vw2_real_il_lookup_zrf(&W->vw2, N, c2r, ip, Tk, ch, VFFT_ILFD_MAX_K, &ck, &cn))
            {
                struct vfft_plan_s *hf = _zrf_build_plan(cfg, N, ch, ck, cn);
                if (hf)
                    return _real_finish(hf);
            }
        }
        else if (eng)
            return (vfft_plan)_odd_incumbent(cfg, ob, W, reg, N, K); /* decided: the routes replay their own record */
    }
    hi = _odd_incumbent(cfg, ob, W, reg, N, K); /* finished */
    if (!W || W->vw2_off_oop)
        return (vfft_plan)hi;
    /* the gate's reference: the out-of-place routes' output (in place, a second handle) */
    href = hi;
    if (ip)
    {
        vfft_config_t co = *cfg;
        co.placement = VFFT_OUTOFPLACE;
        href = (struct vfft_plan_s *)_vfft_create_real_routes(&co, ob, W, reg, N, K);
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
        if (narm == a0)
        {   /* no challenger built: the routes stand, nothing to bank */
            vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
            return (vfft_plan)hi;
        }
        _vfft_create_race_count++;
        {
            _odd_arm_t ca[4];
            vfft_race_arm_t arms[4];
            static const char *nm[2] = { "zrf", "zrf2" };
            int nz = 0;
            for (int i = 0; i < narm; i++)
            {
                ca[i].h = arm[i]; ca[i].in = s0; ca[i].out = b;
                arms[i].name = (i < a0) ? "oddr" : arm[i]->zrm ? "zrm" : nm[nz++ & 1];
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
                const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
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
                char cs[48] = "";
                if (arm[i]->zrf) vfft_zrf_chain_str(arm[i]->zrf->R, arm[i]->zrf->K, cs, sizeof cs);
                fprintf(stderr, " %s%s%s%s=%.0f%s", i < a0 ? "oddr" : arm[i]->zrm ? "zrm" : "zrf:", cs,
                        (arm[i]->zrf && arm[i]->zrf->nomsz) ? "/t" : "", "", ns[i], i == win ? "*" : "");
            }
            fprintf(stderr, "\n");
        }
        vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
        {
            struct vfft_plan_s *hw = arm[win];
            const int rc = hw->zrf ? vw2_real_il_bank_zrf(&W->vw2, N, c2r, ip, Tk, hw->zrf->R, hw->zrf->K, hw->zrf->nomsz, ns[win])
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

static vfft_plan _vfft_create_real(const vfft_config_t *cfg,
                                   vfft_batch ob,
                                   struct vfft_wisdom_s *W,
                                   const vfft_proto_registry_t *reg,
                                   int N,
                                   size_t K)
{
    if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED && K == 1 && !ob && (N & 1) && N >= 3)
    {
        /* an odd cell with an IL real engine: the mono's (N <= 64, out of
         * place) or the flat DIT's (a chain exists) */
        if ((N <= VFFT_ZRM_MAX_N && cfg->placement == VFFT_OUTOFPLACE && vfft_zrm_fn(N, cfg->transform == VFFT_C2R)) ||
            _zrf_has_chain(N))
            return _vfft_create_real_odd_race(cfg, ob, W, reg, N, K);
    }
    return _vfft_create_real_routes(cfg, ob, W, reg, N, K);
}

#endif /* VFFT_BRIDGE_REAL_BRIDGE_H */
