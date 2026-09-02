/* zr2c_build.h - the interleaved-CCE real route ("kind 5").
 *
 * Extracted from vfft.c as migration step 18; see
 * docs/design/refactor_migration_plan.md.
 *
 * THE IDEA
 * --------
 * A real transform of even N, K=1, on interleaved data does not need real
 * kernels at all. Read x[N] as z[N/2] - a reinterpretation, zero work - run a
 * complex child on it, and fold the result into the Hermitian half-spectrum.
 * c2r mirrors it exactly, with the fold leading instead of trailing.
 *
 * This is the like-for-like arm against MKL's own home layout: both engines
 * consume and produce the packed CCE plane, so neither is charged for a
 * conversion the other avoids.
 *
 * TWO CHILD ROUTES, RACED
 * -----------------------
 *   route 0  child_oop_il  - an out-of-place interleaved child, folding into a
 *                            separate plane.
 *   route 1  child_nat_ip  - an in-place child.
 *
 * The pick is per (transform, placement) and banked in the real shard. It is
 * worth knowing that this axis was NOT always raced: the shipped kind-5 rows
 * were migrated with a structural rule (place=oop -> route 0) and no
 * measurement behind them, and where the race disagrees it is worth up to
 * 27-35% on c2r out-of-place. The mechanism here is sound; some of the banked
 * verdicts it reads are stale, which is a wisdom-campaign item and not a
 * property of this code.
 *
 * A CHILD PLAN CARRIES ITS OWN VERDICTS
 * -------------------------------------
 * The complex child is a full plan and runs its own c2c tournaments - chain,
 * kernel forms, order. So one interleaved real plan holds two independent
 * verdict sets: this route, and everything the child decided underneath it.
 * That is why a 1D IL r2c fingerprint shows a zr2c CHILD node.
 *
 * INCLUSION CONTRACT - AND ONE BACK-EDGE WORTH NAMING
 * ---------------------------------------------------
 * Include after the engine prelude, after vfft_internal.h, and specifically
 * AFTER _vw2_persist: the kind-5 banker calls it, and it is a general wisdom
 * helper that stays in vfft.c. That is a back-edge of the same shape
 * _vfft_warn had before step 6a moved it to support/, and it is the reason
 * this header is not yet freely placeable. Moving _vw2_persist into a support
 * header would remove the constraint; it was left alone here because it is
 * used from four call sites far above this point and belongs to a different
 * step's scope.
 */
#ifndef VFFT_TRANSFORMS_REAL_ZR2C_BUILD_H
#define VFFT_TRANSFORMS_REAL_ZR2C_BUILD_H

#include <stdlib.h>
#include <string.h>

#include "vfft_internal.h"                 /* struct vfft_plan_s / vfft_wisdom_s */
#include "zr2c.h"                          /* the Hermitian fold kernels */
#include "wisdom2/wisdom2_real_reader.h"   /* the kind-5 route codec */
#include "support/race.h"                  /* the shared race body */

/* Defined in vfft.c (tentative definition, external linkage). */
extern long _vfft_create_race_count;

/* zr2c (even N, K==1, INTERLEAVED): x[N] read as z[N/2] -> child c2c(N/2) NATURAL -> zr2c.h fold;
 * c2r mirrors it with the fold leading. route 0 = child_oop_il, route 1 = child_nat_ip.
 * See docs/design/vfft_front_door.md. */
static struct vfft_plan_s *_zr2c_build_route(const vfft_config_t *cfg, int N,
                                             int route)
{
    const int half = N / 2, top = N / 4;
    vfft_config_t c2;
    memset(&c2, 0, sizeof c2);
    c2.transform = VFFT_C2C;
    c2.placement = route ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    c2.rigor = cfg->rigor;
    c2.dims = 1;
    c2.n[0] = half;
    c2.howmany = 1;
    c2.order = VFFT_ORDER_NATURAL;
    c2.layout = VFFT_LAYOUT_INTERLEAVED;
    c2.nthreads = cfg->nthreads;
    c2.wisdom = cfg->wisdom;
    /* 🔴 PASS THE WISDOM-LIFECYCLE FIELDS THROUGH. The child does almost
     * all of the work in this composite -- the pair, il_kv, the dir=bwd
     * verdict and the @natoop mode all live in ITS cell, not in the route
     * bit. Dropping these narrowed two documented public contracts to the
     * route bit alone:
     *   recalibrate  ("1 = re-measure + overwrite", vfft.h:277) re-raced only
     *                the route, while every child verdict silently replayed.
     *   wisdom_write (the write guard, vfft.h:278) never reached the child,
     *                so a caller who asked for persistence got the route bit
     *                banked and nothing else.
     * Narrowing a user-visible capability is a contract violation, not a
     * tuning choice. Note the cost is real and intended: a recalibrate now
     * re-plans the child on BOTH arms of the route race. */
    c2.recalibrate = cfg->recalibrate;
    c2.wisdom_write = cfg->wisdom_write;
    struct vfft_plan_s *child = (struct vfft_plan_s *)vfft_create(&c2);
    if (!child)
    {
        _vfft_warn("vfft_create: zr2c child c2c(%d) create failed", half);
        return NULL;
    }
    struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    /* 🔴 64-BYTE ALIGNED, not plain malloc. Both buffers are streamed by
     * AVX2 kernels: the fold reads aff and writes scr, then the child reads
     * scr end to end. malloc gives 16 bytes on this toolchain, so every
     * 32-byte access that straddles a 64-byte line costs an extra line touch.
     * The kernels use loadu/storeu so this was never a CORRECTNESS issue,
     * which is why it survived -- it is pure throughput.
     *
     * Measured, N=2048, front-door arms: every route-0 arm that TOUCHES the
     * scratch ran slow (r2c IP 1469-1528 ns, c2r OOP 1374-1688, c2r IP
     * 1414-1674) while the one route-0 arm that does NOT touch it (r2c OOP,
     * which folds in place in dre) ran 1134-1221 -- and route 1, which
     * allocates no scratch at all, ran 1137-1261 everywhere. The correlation
     * is exact across all four arms. */
    double *aff = NULL, *scr = NULL;
    if (vfft_proto_posix_memalign((void **)&aff, 64,
                                  sizeof(double) * 4u * (size_t)(top + 1)) != 0)
        aff = NULL;
    if (route == 0 &&
        vfft_proto_posix_memalign((void **)&scr, 64,
                                  sizeof(double) * ((size_t)N + 2)) != 0)
        scr = NULL;
    if (!h || !aff || (route == 0 && !scr))
    {
        vfft_destroy((vfft_plan)child);
        free(h);
        vfft_proto_aligned_free(aff);
        vfft_proto_aligned_free(scr);
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
    h->zr2c_child = child;
    h->zr2c_route = route;
    h->zr2c_aff = aff;
    h->zr2c_scratch = scr;
    return h;
}

/* execute the composite. 2 transforms x 2 placements x 2 routes; the folds
 * are in-place-safe by construction (zr2c_gate.c), scratch only where a
 * route-0 shape needs a second plane. */
static void _exec_zr2c(struct vfft_plan_s *h, const double *sre, double *dre)
{
    const int N = h->N, top = N / 4;
    const double *aS = h->zr2c_aff, *aC = h->zr2c_aff + (top + 1);
    const double *bS = h->zr2c_aff + 2 * (top + 1);
    const double *bC = h->zr2c_aff + 3 * (top + 1);
    vfft_plan ch = (vfft_plan)h->zr2c_child;
    size_t xs = (size_t)N + 2;
    if (h->transform == VFFT_R2C)
    {
        if (h->zr2c_route == 0)
        {
            if (h->placement == VFFT_OUTOFPLACE)
            { /* child OOP sre->dre (its z view), fold in place in dre */
                /* 🔴 EXPLICIT cast, not an implicit discard. vfft_execute's
                 * public signature takes double* for sre because in-place
                 * plans legitimately write it; THIS child is out-of-place
                 * (route child_oop_il), so it only reads. Casting here says
                 * that deliberately instead of letting the compiler drop the
                 * qualifier silently -- the warning was real, the behaviour
                 * was not. */
                vfft_execute(ch, VFFT_FORWARD, (double *)sre, NULL, dre, NULL);
                _zr2c_fold_fwd(dre, dre, aS, aC, N, 1, xs, xs);
            }
            else
            { /* in place: child OOP plane->scratch, fold scratch->plane */
                vfft_execute(ch, VFFT_FORWARD, (double *)sre, NULL,
                             h->zr2c_scratch, NULL);   /* OOP child: reads only */
                _zr2c_fold_fwd(h->zr2c_scratch, dre, aS, aC, N, 1, xs, xs);
            }
        }
        else
        {
            /* 🔴 `dre != sre`, NOT `placement == OUTOFPLACE`. Route 1 runs
             * the child on dre, so gating the copy on PLACEMENT meant an
             * in-place plan called with a distinct dre transformed whatever
             * was already in dre and never read sre at all -- measured
             * relerr 1.000, silently. Route 0 reads sre and is correct under
             * the identical call. Keying on the POINTERS makes the two
             * routes behave the same way, so which one a cell banked can no
             * longer change the answer. */
            if (dre != sre)
                memcpy(dre, sre, (size_t)N * sizeof(double));
            vfft_execute(ch, VFFT_FORWARD, dre, NULL, dre, NULL);
            _zr2c_fold_fwd(dre, dre, aS, aC, N, 1, xs, xs);
        }
    }
    else /* VFFT_C2R: CCE spectrum in sre -> N reals in dre */
    {
        if (h->zr2c_route == 0)
        { /* fold sre->scratch (zhat), child OOP scratch->dre */
            _zr2c_fold_bwd(sre, h->zr2c_scratch, bS, bC, N, 1, xs, (size_t)N);
            vfft_execute(ch, VFFT_BACKWARD, h->zr2c_scratch, NULL, dre, NULL);
        }
        else
        { /* fold sre->dre (alias-safe when in place), child in place on dre */
            _zr2c_fold_bwd(sre, dre, bS, bC, N, 1, xs, (size_t)N);
            vfft_execute(ch, VFFT_BACKWARD, dre, NULL, dre, NULL);
        }
    }
}

/* Bank a kind-5 zr2c route verdict: one per-(transform,placement) record in
 * the wisdom2 real shard — no packed read-modify-write needed, the other
 * slots' records are untouched by construction. The in-memory bank alone
 * makes the verdict process-coherent; ns = the winner's per-shot median. */
static void _bank_zr2c(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                       int N, int slot, int route, double ns)
{
    /* 🔴 CHECK THE RETURN. The banker can decline (VW2_EOWNED: the cell
     * belongs to another engine) or fail the codec, and it says so. Firing
     * the persistence seam anyway wrote a file for a bank that never
     * happened, and hid the decline -- which is exactly how two engines end
     * up quietly fighting over one key. */
    int rc = vw2_oop_bank_zr2c_slot(&W->vw2, N, (slot >> 1) & 1, slot & 1,
                                    route, ns);
    if (rc != VW2_OK)
    {
        fprintf(stderr, "vfft: zr2c route verdict NOT banked at N=%d slot=%d "
                        "(rc=%d) -- the cell will re-race on the next create\n",
                N, slot, rc);
        return;
    }
    _vw2_persist(W, cfg);
}

/* forward decls: the race borrows the §6a59 timer/median helpers, defined
 * with the IL A/B machinery further down. */
static double _il_ab_now(void);

/* Race the FULL composite through _exec_zr2c; 3% hysteresis toward the
 * structural default. Both arms are gated correct (zr2c_fd_gate.c).
 * See docs/design/vfft_front_door.md. */
/* the two arms of the zr2c route race: two finished handles */
typedef struct { struct vfft_plan_s *h; const double *s0; double *b; } _zr2c_arm_t;
static void _zr2c_arm_run(void *v)
{
    _zr2c_arm_t *c = (_zr2c_arm_t *)v;
    _exec_zr2c(c->h, c->s0, c->b);
}
static struct vfft_plan_s *_zr2c_build(const vfft_config_t *cfg, int N,
                                       struct vfft_wisdom_s *W)
{
    /* 1. env — the racing hook. Beats wisdom, never banks. */
    {
        const char *e = getenv("VFFT_ZR2C_ROUTE");
        if (e && e[0])
            return _zr2c_build_route(cfg, N, atoi(e) != 0);
    }
    const int slot = vfft_zr2c_kv_slot(cfg->transform == VFFT_C2R,
                                       cfg->placement == VFFT_INPLACE);
    const int def = (cfg->placement == VFFT_INPLACE) ? 1 : 0;

    /* 2. banked kind-5 verdict for THIS (transform, placement) slot. */
    if (W && !cfg->recalibrate)
    {
        int f = 0;
        if (W->vw2_off_oop)
        {
            const vfft_oop_wisdom_entry_t *ke =
                vfft_oop_wisdom_lookup_zr2c(&W->oop, N);
            f = ke ? vfft_zr2c_kv_get(ke->zr_kv, slot) : 0;
        }
        else
        {
            int kv;
            if (vw2_oop_lookup_zr2c(&W->vw2, N, &kv))
                f = vfft_zr2c_kv_get(kv, slot);
        }
        if (f)
            return _zr2c_build_route(cfg, N, f - 1);
    }

    /* 3. no verdict and no wisdom to bank into -> structural default.
     * With wisdom, every rigor tier races (the library is measured-only —
     * there is no ESTIMATE tier); a missing cell races once and banks. */
    if (!W)
        return _zr2c_build_route(cfg, N, def);

    _vfft_create_race_count++;   /* HARNESS: past the wisdom hit, the clock decides */
    struct vfft_plan_s *h0 = _zr2c_build_route(cfg, N, 0);
    struct vfft_plan_s *h1 = _zr2c_build_route(cfg, N, 1);
    if (!h0 || !h1) /* one route can't build -> the other serves, no bank */
        return h0 ? h0 : h1;

    size_t xs = (size_t)N + 2;
    double *a = (double *)STRIDE_ALIGNED_ALLOC(64, (xs * 8 + 63) & ~(size_t)63);
    double *b = (double *)STRIDE_ALIGNED_ALLOC(64, (xs * 8 + 63) & ~(size_t)63);
    if (!a || !b)
    {
        STRIDE_ALIGNED_FREE(a);
        STRIDE_ALIGNED_FREE(b);
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
    double t0 = _il_ab_now();
    _exec_zr2c(h0, s0, b);
    double e0 = _il_ab_now() - t0;
    t0 = _il_ab_now();
    _exec_zr2c(h1, s0, b);
    double e1 = _il_ab_now() - t0;
    double est = e0 > e1 ? e0 : e1;
    int reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
    if (reps < 2)
        reps = 2;
    if (reps > 64)
        reps = 64;
    double n0, n1;
    {
        _zr2c_arm_t c0 = { h0, s0, b }, c1 = { h1, s0, b };
        const vfft_race_arm_t arms[2] = { { "route0", _zr2c_arm_run, &c0 },
                                          { "route1", _zr2c_arm_run, &c1 } };
        /* 9 rounds alternated, median (the est shots above were the warm-up) */
        const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 0, NULL, NULL };
        double ns[2];
        vfft_race_run(&proto, arms, 2, ns);
        n0 = ns[0];
        n1 = ns[1];
    }
    STRIDE_ALIGNED_FREE(a);
    STRIDE_ALIGNED_FREE(b);
    int win = (def == 0) ? ((n1 < n0 * 0.97) ? 1 : 0)
                         : ((n0 < n1 * 0.97) ? 0 : 1);
    if (getenv("VFFT_ZRACE_VERBOSE"))
        fprintf(stderr, "[zr2c] N=%d %s %s route race: reps=%d hyst=3%% "
                        "alt-order median | oop-il=%.0f nat-ip=%.0f -> "
                        "route=%d (bank slot %d)\n",
                N, cfg->transform == VFFT_C2R ? "c2r" : "r2c",
                cfg->placement == VFFT_INPLACE ? "ip" : "oop",
                reps, n0, n1, win, slot);
    _bank_zr2c(W, cfg, N, slot, win, win ? n1 : n0);
    if (win)
    {
        vfft_destroy((vfft_plan)h0);
        return h1;
    }
    vfft_destroy((vfft_plan)h1);
    return h0;
}

#endif /* VFFT_TRANSFORMS_REAL_ZR2C_BUILD_H */
