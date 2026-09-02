/* plane_queue.h - the 2D plane queue (howmany > 1).
 *
 * Extracted from vfft.c as migration step 20; see
 * docs/design/refactor_migration_plan.md.
 *
 * TWO MODES, RACED AT CREATE
 * --------------------------
 *   SERIAL  loop the PRIMARY plan over the K planes. Note this is NOT "the
 *           single-threaded option": each plane still intra-threads according
 *           to its own banked verdicts. It is the option where parallelism
 *           lives INSIDE one plane at a time.
 *   QUEUE   an atomic plane counter; worker t pulls whole planes onto its own
 *           SERIAL clone. Plane-per-worker, zero barriers, and no nested pool
 *           dispatch by construction.
 *
 * WHY THE CLONES MUST BE SERIAL
 * -----------------------------
 * A queue worker is already a pool thread. If its clone threaded again it
 * would dispatch to the pool from inside the pool, and a worker dispatching to
 * itself deadlocks the wait. Serial clones make that structurally impossible
 * rather than merely unlikely.
 *
 * WHICH MT CLASS THIS IS
 * ----------------------
 * ONE TRANSFORM PER CORE: each worker owns whole planes, so nothing about any
 * single plane's plan depends on the thread count. That is the T-FREE class -
 * a verdict here is valid at any T. Contrast the 2D column pass, where the
 * cores SHARE one transform, T decides how the work is cut, and the verdict
 * must carry the T it was raced at (cmt/cmtt).
 *
 * BANKED (2026-09-02, the 2D arm audit closed the gap)
 * ----------------------------------------------------
 * The loop-vs-queue verdict rides the PRIMARY plane's own wisdom row as
 * pq=<0|1> pqn=<P> pqt=<T> — the plane count and the worker count it was
 * raced at are its validity condition (a P or T mismatch re-races and
 * re-banks). The row is whichever one the inner howmany=1 create banked
 * (IL c2c, IL real, or the split-tier row), found by lookup and merged
 * into with vw2_update_field; a cold cell with no row banks nothing (loud
 * under VFFT_IL2D_LOG). Kill/force switch VFFT_PQ_NO_MT pins and never
 * replays or banks. Engagement counter: vfft_pq_mt_passes().
 *
 * INCLUSION CONTRACT
 * ------------------
 * Include after the engine prelude and after vfft_internal.h, as vfft.c does.
 */
#ifndef VFFT_TRANSFORMS_FFT2D_PLANE_QUEUE_H
#define VFFT_TRANSFORMS_FFT2D_PLANE_QUEUE_H

#include <stdlib.h>

#include "vfft_internal.h"     /* struct vfft_plan_s */
#include "support/threads.h"   /* the pool: dispatch, wait_all */

/* DEFINED in vfft.c with external linkage; see the note in zturn_mt.h for why
 * a counter incremented from a header must not live in one. */
extern long _vfft_pq_mt_count;

typedef struct
{
    struct vfft_plan_s *plan; /* this worker's serial clone */
    struct vfft_plan_s *h;    /* the queue handle (dists, count) */
    vfft_dir_t dir;
    const double *src;
    double *dst;
    volatile long *next;      /* the shared plane counter */
} _pq_arg;

static void _pq_tramp(void *v)
{
    _pq_arg *a = (_pq_arg *)v;
    const size_t P = a->h->pq_n;
    for (;;)
    {
#ifdef _WIN32
        const long p = InterlockedIncrement(a->next) - 1;
#else
        const long p = __sync_fetch_and_add(a->next, 1);
#endif
        if ((size_t)p >= P)
            return;
        vfft_execute((vfft_plan)a->plan, a->dir,
                     a->src + (size_t)p * a->h->pq_sdist, NULL,
                     a->dst + (size_t)p * a->h->pq_ddist, NULL);
    }
}

static void _pq_execute(struct vfft_plan_s *h, vfft_dir_t dir,
                        const double *sre, double *dre)
{
    if (!dre)
        dre = (double *)sre; /* in-place convenience (C2C) */
    if (h->pq_mt && h->pq_wn > 0)
    {
        _pq_arg a[STRIDE_POOL_MAX_DISPATCH];
        volatile long next = 0;
        /* pq_wn = the clones built at create (the plan's own snapshot); the
         * pool's one clamp bounds it by the LIVE pool too, so a pool that
         * shrank since create can no longer be over-dispatched. */
        int T = stride_pool_workers_for(h->pq_wn);
        int t;
        if ((size_t)T > h->pq_n)
            T = (int)h->pq_n;
        for (t = 0; t < T; t++)
        {
            a[t].plan = h->pq_w[t];
            a[t].h = h;
            a[t].dir = dir;
            a[t].src = sre;
            a[t].dst = dre;
            a[t].next = &next;
        }
        stride_pool_run(T, _pq_tramp, a, sizeof a[0]);
        _vfft_pq_mt_count++; /* engagement, see vfft.h */
        return;
    }
    {
        size_t p;
        for (p = 0; p < h->pq_n; p++)
            vfft_execute((vfft_plan)h->pq_inner, dir,
                         sre + p * h->pq_sdist, NULL,
                         dre + p * h->pq_ddist, NULL);
    }
}

/* ── the loop-vs-queue race (create-time, min-of-3 alternated on
 * scratch). The queue also self-gates: no pool, no clones, or a clone
 * failing the BITWISE probe against the primary => pq_mt stays 0 and
 * the loop serves (the primary keeps its own intra-MT verdicts). */
/* the two arms of the plane-queue race: one handle, pq_mt toggled */
typedef struct { struct vfft_plan_s *h; vfft_dir_t dir; double *src, *dst; } _pq_mt_arm_t;
static void _pq_mt_arm_loop(void *v)
{
    _pq_mt_arm_t *c = (_pq_mt_arm_t *)v;
    c->h->pq_mt = 0;
    _pq_execute(c->h, c->dir, c->src, c->dst);
}
static void _pq_mt_arm_queue(void *v)
{
    _pq_mt_arm_t *c = (_pq_mt_arm_t *)v;
    c->h->pq_mt = 1;
    _pq_execute(c->h, c->dir, c->src, c->dst);
}
/* the primary plane's own row: the first key that resolves, in the order
 * the tiers bank them. NULL-safe: a miss means "nothing to ride". */
static int _pq_row_key(const struct vfft_plan_s *h, const vfft_config_t *cfg,
                       const vw2_store_t *st, vw2_key_t *k)
{
    const int il = (cfg->layout == (int)VFFT_LAYOUT_INTERLEAVED);
    const int real = (h->transform != VFFT_C2C);
    const int nat = (cfg->order == VFFT_ORDER_NATURAL);
    int i;
    for (i = 0; i < 4; i++)
    {
        memset(k, 0, sizeof *k);
        switch (i)
        {
        case 0: if (!il) continue;            /* the IL rows (lay=il, ord=scr) */
            vw2__2d_key(k, real ? VW2_T_R2C : VW2_T_C2C, 2, h->N, h->N2, 0,
                        VW2_ORD_SCR, VW2_LAY_IL);
            break;
        case 1:                               /* split c2c, this order */
            if (real) continue;
            vw2__2d_key(k, VW2_T_C2C, 2, h->N, h->N2, 0,
                        nat ? VW2_ORD_NAT : VW2_ORD_SCR, VW2_LAY_ANY);
            break;
        case 2:                               /* split real (ord=nat rows) */
            if (!real) continue;
            vw2__2d_key(k, h->transform == VFFT_C2R ? VW2_T_C2R : VW2_T_R2C,
                        2, h->N, h->N2, 0, VW2_ORD_NAT, VW2_LAY_ANY);
            break;
        default:
            continue;
        }
        if (vw2_lookup(st, k)) return 1;
    }
    return 0;
}

static void _pq_mt_race(struct vfft_plan_s *h);

/* replay-or-race: the verdict is valid for the (P, T) it was raced at */
static void _pq_mt_replay_or_race(struct vfft_plan_s *h,
                                  struct vfft_wisdom_s *W,
                                  const vfft_config_t *cfg)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const int have = (W && !getenv("VFFT_PQ_NO_MT") &&
                      _pq_row_key(h, cfg, &W->vw2, &k));
    if (have && !cfg->recalibrate && (r = vw2_lookup(&W->vw2, &k)) != NULL)
    {
        const char *v = vw2_rec_get(r, "pq");
        const char *vn = vw2_rec_get(r, "pqn");
        const char *vt = vw2_rec_get(r, "pqt");
        if (v && vn && vt && (size_t)atol(vn) == h->pq_n &&
            atoi(vt) == h->pq_wn)
        {
            h->pq_mt = atoi(v) ? 1 : 0;
            if (getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[pq] %dx%d P=%zu T=%d: replay %s src=wisdom\n",
                        h->N, h->N2, h->pq_n, h->pq_wn,
                        h->pq_mt ? "QUEUE" : "loop");
            return;
        }
    }
    _pq_mt_race(h);
    if (have)
    {
        char b[24];
        int rc;
        snprintf(b, sizeof b, "%zu", h->pq_n);
        rc = vw2_update_field(&W->vw2, &k, "pqn", b);
        snprintf(b, sizeof b, "%d", h->pq_wn);
        rc |= vw2_update_field(&W->vw2, &k, "pqt", b);
        rc |= vw2_update_field(&W->vw2, &k, "pq", h->pq_mt ? "1" : "0");
        if (rc == VW2_OK)
            _vw2_persist(W, cfg);
        else if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[pq] %dx%d P=%zu T=%d: verdict NOT banked (the "
                            "primary row is a wildcard/migrated row: no "
                            "exact key to merge into)\n",
                    h->N, h->N2, h->pq_n, h->pq_wn);
    }
    else if (getenv("VFFT_IL2D_LOG") && !getenv("VFFT_PQ_NO_MT"))
        fprintf(stderr, "[pq] %dx%d P=%zu T=%d: no primary row to bank the "
                        "verdict on\n", h->N, h->N2, h->pq_n, h->pq_wn);
}

static void _pq_mt_race(struct vfft_plan_s *h)
{
    const size_t sb = h->pq_n * h->pq_sdist, db = h->pq_n * h->pq_ddist;
    double *src = (double *)malloc(sb * sizeof(double));
    double *dst = (double *)malloc(db * sizeof(double));
    double tl = 1e300, tq = 1e300;
    const vfft_dir_t dir =
        (h->transform == VFFT_C2R) ? VFFT_BACKWARD : VFFT_FORWARD;
    int r;
    size_t i;
    const char *ce = getenv("VFFT_PQ_NO_MT");
    if (ce)
    {
        h->pq_mt = (atoi(ce) == 0 && h->pq_wn > 0);
        free(src);
        free(dst);
        return;
    }
    if (!src || !dst || h->pq_wn <= 0)
    {
        free(src);
        free(dst);
        return; /* loop serves */
    }
    for (i = 0; i < sb; i++)
        src[i] = 1.0 + 1e-6 * (double)(i & 511);
    h->pq_mt = 0;
    _pq_execute(h, dir, src, dst); /* warm the loop arm */
    h->pq_mt = 1;
    _pq_execute(h, dir, src, dst); /* warm the queue arm */
    {
        _pq_mt_arm_t c = { h, dir, src, dst };
        const vfft_race_arm_t arms[2] = { { "loop", _pq_mt_arm_loop, &c },
                                          { "queue", _pq_mt_arm_queue, &c } };
        const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL }; /* min-of-3, A then B */
        double ns[2];
        (void)r;
        vfft_race_run(&proto, arms, 2, ns);
        tl = ns[0];
        tq = ns[1];
    }
    h->pq_mt = (tq < tl);
    if (getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[pq] race %dx%d P=%zu T=%d: loop=%.0f "
                        "queue=%.0f -> %s\n",
                h->N, h->N2, h->pq_n, h->pq_wn, tl, tq,
                h->pq_mt ? "QUEUE" : "loop");
    free(src);
    free(dst);
}

#endif /* VFFT_TRANSFORMS_FFT2D_PLANE_QUEUE_H */
