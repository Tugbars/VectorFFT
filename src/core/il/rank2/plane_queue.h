/* plane_queue.h - the 2D plane queue (howmany > 1), and one rank up the
 * VOLUME QUEUE (2026-10-18): the same wrapper over the single-volume rank-3
 * plan, K contiguous volumes at the per-volume distances. "Plane" below
 * reads "volume" at rank 3.
 *
 * THREE MODES, RACED AT CREATE
 * ----------------------------
 *   SERIAL  loop the PRIMARY plan over the K planes. Note this is NOT "the
 *           single-threaded option": each plane still intra-threads according
 *           to its own banked verdicts. It is the option where parallelism
 *           lives INSIDE one plane at a time.
 *   QUEUE   an atomic plane counter; worker t pulls whole planes onto its own
 *           SERIAL clone. Plane-per-worker, zero barriers, and no nested pool
 *           dispatch by construction.
 *   GROUPS  (2026-10-19, the volume queue) K groups of G = T / K threads, one
 *           volume each, all at once in ONE pool run: slot s is group s / G,
 *           member s % G, and every member runs its group's volume on the
 *           group's own G-thread clone through the tier's COOPERATIVE
 *           execute (fftnd_real_il.h, _ilndr_execute_group): the threaded
 *           walk's phase bodies over the same partition, each fork-join a
 *           wait on the group's own barrier -- no member ever asks the pool
 *           for anything. The arm exists at 2 <= K <= T / 2 (G >= 2: at
 *           G = 1 it IS the queue; policy_il.h, vfft_policy_pq_groups_ok),
 *           for the rank-3 real tier (the other tiers get it when they get a
 *           cooperative execute); the remainder T - K G idles, since a batch
 *           takes as long as its slowest group. Measured 64x64x64 x 4 at
 *           T = 8 R2C before the arm: loop 274 us, queue 353, MKL 190.
 *
 * IN PLACE (2026-10-18)
 * ---------------------
 * An in-place inner (the real tiers' padded plane or volume, the c2c tiers'
 * one buffer) is probed and raced in == out, as the queue runs it: an
 * in-place real plan refuses a distinct destination at the execute door, so
 * the probe runs every instance on its own copy of the seed, and the race's
 * one buffer is re-laid from its seed before every timed sample (an in-place
 * arm walks its own output).
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
 * is keyed by the plan's thread count like every threaded verdict (v1.3).
 *
 * BANKED
 * ------
 * The loop-vs-queue-vs-groups verdict rides the PRIMARY plane's own wisdom
 * row as pq=<0|1|2> pqn=<P> (2 = the groups arm; G = T / P is implied, the
 * row being keyed by T) — the row is keyed by the plan's thread count, and the
 * plane count it was raced at is its validity condition (a P mismatch
 * re-races and re-banks; a banked 2 also needs this plan's G clone set, else
 * the race decides again). A real cell's row is direction-shared (keyed
 * t=r2c), so the c2r queue's verdict has its own names there, pq_c2r=
 * pqn_c2r=, as the tiers' cmt_c2r= (2026-10-19): the two directions never
 * replay or overwrite each other's. The row is whichever one the inner howmany=1 create banked
 * (IL c2c, IL real, or the split-tier row; at rank 3 the single-volume row
 * of the 3D shard; an in-place real cell its own place=ip row, as the real
 * tiers key it), found by lookup and merged
 * into with vw2_update_field; a cold cell with no row banks nothing (loud
 * under VFFT_IL2D_LOG). Kill/force switch VFFT_PQ_NO_MT pins and never
 * replays or banks: 1 kills (the loop), 0 pins the queue, 2 pins the groups
 * (an absent set: the loop). Engagement counters: vfft_pq_mt_passes() for
 * the queue, vfft_pq_grp_passes() for the groups.
 *
 * INCLUSION CONTRACT
 * ------------------
 * Include after the engine prelude and after vfft_internal.h, as vfft.c does
 * (and after il/rank2/il2d_tier.h: the in-place race's reset hook is its).
 */
#ifndef VFFT_TRANSFORMS_FFT2D_PLANE_QUEUE_H
#define VFFT_TRANSFORMS_FFT2D_PLANE_QUEUE_H

#include <stdlib.h>

#include "vfft_internal.h"     /* struct vfft_plan_s */
#include "common/support/threads.h"   /* the pool: dispatch, wait_all */

/* DEFINED in vfft.c with external linkage: a counter incremented from a
 * header must not be defined in one (each including TU would count into its
 * own copy). */
extern long _vfft_pq_mt_count;
extern long _vfft_pq_grp_count;   /* the groups arm's executes (vfft_pq_grp_passes) */

/* the GROUPS arm's per-slot argument: slot s = group s / G, member s % G, the group's
 * G-thread clone, its volume, and its barrier */
typedef struct
{
    struct vfft_plan_s *plan; /* the group's clone at G threads */
    const double *src;        /* the group's volume */
    double *dst;
    int member, G;
    _vfft_grp_barrier_t *bar;
} _pq_grp_arg;

static void _pq_grp_tramp(void *v)
{
    _pq_grp_arg *a = (_pq_grp_arg *)v;
    _ilndr_execute_group(a->plan->ilndr, a->src, a->dst, a->member, a->G, a->bar);
}

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
        _vfft_exec6((vfft_plan)a->plan, a->dir,
                     a->src + (size_t)p * a->h->pq_sdist, NULL,
                     a->dst + (size_t)p * a->h->pq_ddist, NULL);
    }
}

static void _pq_execute(struct vfft_plan_s *h, vfft_dir_t dir,
                        const double *sre, double *dre)
{
    if (!dre)
        dre = (double *)sre; /* in-place convenience (C2C) */
    if (h->pq_mt == 2 && h->pq_gn > 0 && (size_t)h->pq_gn == h->pq_n && h->pq_G >= 2)
    {   /* GROUPS: one pool run of K G slots, every group's barrier re-armed first. Never a
         * partial group: a live pool below K G (it cannot shrink under a live plan, but the
         * clamp is the queue's idiom) hands this execute to the loop. */
        const int G = h->pq_G, K = h->pq_gn, Tg = K * G;
        if (thread_pool_workers_for(h->nthreads) >= Tg)
        {
            _pq_grp_arg a[THREAD_POOL_MAX_DISPATCH];
            int s;
            for (s = 0; s < K; s++)
                _thread_pool_barrier_init(&h->pq_bar[s].b, G);
            for (s = 0; s < Tg; s++)
            {
                const int g = s / G;
                a[s].plan = h->pq_g[g];
                a[s].src = sre + (size_t)g * h->pq_sdist;
                a[s].dst = dre + (size_t)g * h->pq_ddist;
                a[s].member = s % G;
                a[s].G = G;
                a[s].bar = &h->pq_bar[g];
            }
            thread_pool_run(Tg, _pq_grp_tramp, a, sizeof a[0]);
            _vfft_pq_grp_count++; /* engagement, vfft_pq_grp_passes() */
            return;
        }
    }
    else if (h->pq_mt == 1 && h->pq_wn > 0)
    {
        _pq_arg a[THREAD_POOL_MAX_DISPATCH];
        volatile long next = 0;
        /* pq_wn = the clones built at create (the plan's own snapshot); the
         * pool's one clamp bounds it by the LIVE pool too, so a pool that
         * shrank since create can no longer be over-dispatched. */
        int T = thread_pool_workers_for(h->pq_wn);
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
        thread_pool_run(T, _pq_tramp, a, sizeof a[0]);
        _vfft_pq_mt_count++; /* engagement, see vfft.h */
        return;
    }
    {
        size_t p;
        for (p = 0; p < h->pq_n; p++)
            _vfft_exec6((vfft_plan)h->pq_inner, dir,
                         sre + p * h->pq_sdist, NULL,
                         dre + p * h->pq_ddist, NULL);
    }
}

/* ── the loop-vs-queue(-vs-groups) race (create-time, min-of-3, A then B
 * then C, on scratch). The queue also self-gates: no pool, no clones, or a
 * clone failing the BITWISE probe against the primary => pq_mt stays 0 and
 * the loop serves (the primary keeps its own intra-MT verdicts); the groups
 * arm is in the race only when its clone set was built (pq_gn == K). */
/* the arms of the plane-queue race: one handle, pq_mt toggled */
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
static void _pq_mt_arm_groups(void *v)
{
    _pq_mt_arm_t *c = (_pq_mt_arm_t *)v;
    c->h->pq_mt = 2;
    _pq_execute(c->h, c->dir, c->src, c->dst);
}
/* the cell's shape for the log lines: N1xN2 at rank 2, N1xN2xN3 at rank 3 */
static const char *_pq_shape(const struct vfft_plan_s *h, char *b, size_t n)
{
    if (h->N3 > 0)
        snprintf(b, n, "%dx%dx%d", h->N, h->N2, h->N3);
    else
        snprintf(b, n, "%dx%d", h->N, h->N2);
    return b;
}
/* the primary plane's own row: the first key that resolves, in the order
 * the tiers bank them. NULL-safe: a miss means "nothing to ride". */
static int _pq_row_key(const struct vfft_plan_s *h, const vfft_cfg_t *cfg,
                       const vw2_store_t *st, vw2_key_t *k)
{
    const int il = (cfg->layout == (int)VFI_LAYOUT_IL);
    const int real = (h->transform != VFFT_C2C);
    const int nat = (vfft_policy_ord_rankn(cfg) == VW2_ORD_NAT);
    const int rank = (h->N3 > 0) ? 3 : 2;   /* the volume queue rides the single-volume row of the 3D shard (2026-10-18) */
    const int ip = real && (h->placement == VFFT_INPLACE);   /* the in-place REAL cell's own row (pl=ip), as the 2D and
                                                               * 3D real tiers key it; the c2c rows are placement-blind */
    int i;
    for (i = 0; i < 5; i++)
    {
        memset(k, 0, sizeof *k);
        switch (i)
        {
        /* the IL rows. The plane-queue verdict (queue vs loop) is SHARED by
         * both order classes, so it rides whichever IL row the cell has, in
         * this FIXED order -- scr first, then nat -- and both the replay and
         * the bank walk the same order, so exactly one verdict is ever
         * consulted or written (a store that has only seen NATURAL requests
         * still finds its row, rather than re-racing on every create). */
        case 0: if (!il) continue;
            vw2__2d_key(k, real ? VW2_T_R2C : VW2_T_C2C, rank, h->N, h->N2, h->N3,
                        VW2_ORD_SCR, VW2_LAY_IL, h->nthreads);
            if (ip) k->pl = VW2_PL_IP;
            break;
        case 1: if (!il) continue;
            vw2__2d_key(k, real ? VW2_T_R2C : VW2_T_C2C, rank, h->N, h->N2, h->N3,
                        VW2_ORD_NAT, VW2_LAY_IL, h->nthreads);
            if (ip) k->pl = VW2_PL_IP;
            break;
        case 2:                               /* split c2c, this order */
            if (real) continue;
            vw2__2d_key(k, VW2_T_C2C, rank, h->N, h->N2, h->N3,
                        nat ? VW2_ORD_NAT : VW2_ORD_SCR, VW2_LAY_ANY, h->nthreads);
            break;
        case 3:                               /* split real (ord=nat rows) */
            if (!real) continue;
            vw2__2d_key(k, h->transform == VFFT_C2R ? VW2_T_C2R : VW2_T_R2C,
                        rank, h->N, h->N2, h->N3, VW2_ORD_NAT, VW2_LAY_ANY, h->nthreads);
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
                                  const vfft_cfg_t *cfg)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    char sb[40];
    /* the pair by direction: a real cell's row is direction-shared (keyed t=r2c), so the
     * c2r queue reads and banks pq_c2r= pqn_c2r= there, as the tiers' cmt_c2r= (2026-10-19) */
    const int c2r = (h->transform == VFFT_C2R);
    const char *tk_pq = c2r ? "pq_c2r" : "pq", *tk_pqn = c2r ? "pqn_c2r" : "pqn";
    const int have = (W && !getenv("VFFT_PQ_NO_MT") &&
                      _pq_row_key(h, cfg, &W->vw2, &k));
    if (have && !cfg->recalibrate && (r = vw2_lookup(&W->vw2, &k)) != NULL)
    {
        const char *v = vw2_rec_get(r, tk_pq);
        const char *vn = vw2_rec_get(r, tk_pqn);
        if (v && vn && (size_t)atol(vn) == h->pq_n)   /* the row is the plan's own T (v1.3) */
        {
            const int pv = atoi(v);
            /* a banked 2 (the groups arm) needs this plan's G clone set; without it the
             * race decides again, as a P mismatch does */
            if (pv != 2 || (h->pq_gn > 0 && (size_t)h->pq_gn == h->pq_n))
            {
                h->pq_mt = pv == 2 ? 2 : pv ? 1 : 0;
                if (getenv("VFFT_IL2D_LOG"))
                    fprintf(stderr, "[pq] %s P=%zu T=%d: replay %s src=wisdom\n",
                            _pq_shape(h, sb, sizeof sb), h->pq_n, h->pq_wn,
                            h->pq_mt == 2 ? "GROUPS" : h->pq_mt ? "QUEUE" : "loop");
                return;
            }
        }
    }
    _pq_mt_race(h);
    if (have)
    {
        char b[24];
        int rc;
        snprintf(b, sizeof b, "%zu", h->pq_n);
        rc = vw2_update_field(&W->vw2, &k, tk_pqn, b);
        rc |= vw2_update_field(&W->vw2, &k, tk_pq, h->pq_mt == 2 ? "2" : h->pq_mt ? "1" : "0");
        if (rc == VW2_OK)
            _vw2_persist(W, cfg);
        else if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[pq] %s P=%zu T=%d: verdict NOT banked (the "
                            "primary row is a wildcard/migrated row: no "
                            "exact key to merge into)\n",
                    _pq_shape(h, sb, sizeof sb), h->pq_n, h->pq_wn);
    }
    else if (getenv("VFFT_IL2D_LOG") && !getenv("VFFT_PQ_NO_MT"))
        fprintf(stderr, "[pq] %s P=%zu T=%d: no primary row to bank the "
                        "verdict on\n", _pq_shape(h, sb, sizeof sb), h->pq_n, h->pq_wn);
}

static void _pq_mt_race(struct vfft_plan_s *h)
{
    const size_t sb = h->pq_n * h->pq_sdist, db = h->pq_n * h->pq_ddist;
    const int ip = (h->placement == VFFT_INPLACE);   /* in place (2026-10-18): one buffer, in == out, as the queue runs it */
    double *src = (double *)vfft_aligned_alloc(sb * sizeof(double));
    double *dst = ip ? src : (double *)vfft_aligned_alloc(db * sizeof(double));
    double *seed = ip ? (double *)vfft_aligned_alloc(sb * sizeof(double)) : NULL;
    double tl = 1e300, tq = 1e300, tg = 1e300;
    const vfft_dir_t dir =
        (h->transform == VFFT_C2R) ? VFFT_BACKWARD : VFFT_FORWARD;
    const int grp = (h->pq_gn > 0 && (size_t)h->pq_gn == h->pq_n);   /* the groups arm's clone set exists */
    int r;
    size_t i;
    char shp[40];
    const char *ce = getenv("VFFT_PQ_NO_MT");
    if (ce)
    {   /* 1 kills (the loop); 0 pins the queue, 2 the groups -- an absent set is the loop */
        const int cv = atoi(ce);
        h->pq_mt = (cv == 0 && h->pq_wn > 0) ? 1 : (cv == 2 && grp) ? 2 : 0;
        vfft_aligned_free(src);
        if (!ip) vfft_aligned_free(dst);
        vfft_aligned_free(seed);
        return;
    }
    if (!src || !dst || (ip && !seed) || h->pq_wn <= 0)
    {
        vfft_aligned_free(src);
        if (!ip) vfft_aligned_free(dst);
        vfft_aligned_free(seed);
        return; /* loop serves */
    }
    for (i = 0; i < sb; i++)
        src[i] = 1.0 + 1e-6 * (double)(i & 511);
    if (ip)
        memcpy(seed, src, sb * sizeof(double));
    h->pq_mt = 0;
    _pq_execute(h, dir, src, dst); /* warm the loop arm */
    h->pq_mt = 1;
    _pq_execute(h, dir, src, dst); /* warm the queue arm */
    if (grp)
    {
        if (ip) memcpy(src, seed, sb * sizeof(double));
        h->pq_mt = 2;
        _pq_execute(h, dir, src, dst); /* warm the groups arm */
    }
    {
        _pq_mt_arm_t c = { h, dir, src, dst };
        const vfft_race_arm_t arms[3] = { { "loop", _pq_mt_arm_loop, &c },
                                          { "queue", _pq_mt_arm_queue, &c },
                                          { "groups", _pq_mt_arm_groups, &c } };
        vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 0, 0, NULL, NULL, 0 }; /* min-of-3, A then B then C */ /* THREADED arms: never paused (VFFT_RACE_PACE_MS) */
        _il2d_ip_reset_t rs;
        double ns[3];
        (void)r;
        if (ip)
        {   /* the one buffer, re-laid from its seed before every timed sample: an in-place
             * arm walks its own output (a repeated c2r in place grows by N1 N2 N3 a pass) */
            rs.p = src; rs.seed = seed; rs.n = sb;
            proto.reset = _il2d_ip_reset;
            proto.reset_ctx = &rs;
        }
        vfft_race_run(&proto, arms, grp ? 3 : 2, ns);
        tl = ns[0];
        tq = ns[1];
        if (grp) tg = ns[2];
    }
    /* the fastest of the three; the loop keeps a tie, then the queue */
    h->pq_mt = (tg < tl && tg < tq) ? 2 : (tq < tl) ? 1 : 0;
    if (getenv("VFFT_IL2D_LOG"))
    {
        fprintf(stderr, "[pq] race %s P=%zu T=%d: loop=%.0f queue=%.0f",
                _pq_shape(h, shp, sizeof shp), h->pq_n, h->pq_wn, tl, tq);
        if (grp) fprintf(stderr, " groups(G=%d)=%.0f", h->pq_G, tg);
        fprintf(stderr, " -> %s\n", h->pq_mt == 2 ? "GROUPS" : h->pq_mt ? "QUEUE" : "loop");
    }
    vfft_aligned_free(src);
    if (!ip) vfft_aligned_free(dst);
    vfft_aligned_free(seed);
}

#endif /* VFFT_TRANSFORMS_FFT2D_PLANE_QUEUE_H */
