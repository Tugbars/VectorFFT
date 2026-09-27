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
static vfft_plan _real_finish(struct vfft_plan_s *h)
{
    return h;
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

static vfft_plan _vfft_create_real(const vfft_config_t *cfg,
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

#endif /* VFFT_BRIDGE_REAL_BRIDGE_H */
