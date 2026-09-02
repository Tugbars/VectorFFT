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
 * NOT BANKED - A GAP, NOT A POLICY
 * --------------------------------
 * The loop-vs-queue verdict is plan-local and re-races on every create, in
 * every process, pending the wisdom2 1D cell convention. Kill/force switch:
 * VFFT_PQ_NO_MT. Engagement counter: vfft_pq_mt_passes().
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
