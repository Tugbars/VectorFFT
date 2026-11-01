/* c2c_estimate_lm_il.h — the ESTIMATE create of a LANE-MAJOR c2c batch (VFFT_FLAG_ESTIMATE with
 * VFFT_BATCH_LANE_MAJOR, howmany = K >= 2) at rank 1, 2 and 3: the batch's rows written from the formulas and
 * replayed by the lane-major doors with nothing measured (docs/roadmap/lane_major_estimate_plan.md).
 *
 * THE DOORS (il/rank1/lm_batch_il.h, il/rank2/lm_batch2d_il.h, il/rank3/lm_batch3d_il.h). Element i of transform
 * t sits at z[2 (i K + t)], so every axis is a column pass over lanes. Rank 1: one column pass of N rows over K
 * lanes. Rank 2: one column pass of N1 rows over N2 K lanes, then N1 blocks, each the 1D lane-major batch at
 * (N2, K) on one thread, in place (the block child, its rows under rp_). Rank 3: one column pass of N1 rows over
 * N2 N3 K lanes, then N1 blocks of the 2D lane-major batch at (N2, N3, K). Each create plans on the batch's own
 * row, t=c2c n=<shape> q=K ord place=oop lay=lm nthreads=T (both placements share it): the column builder's
 * verdicts (_il2d_col_build: the chain, its forms, a Rader stage, Bluestein, the turned prime pass and its plan
 * under tpc_), the serial lane tile sw= and at T > 1 the threaded verdict cmt= mtarm= msw= (_lm_race), the block
 * child under rp_.
 *
 * THE RECIPE (est_plan_lm), the 2D plane's formulas carried over, with the three rules below:
 *   - the column pass: est_col_plan at w = the lanes (K, N2 K, N2 N3 K), natural for a natural request (the
 *     first axis leaves the cell's class, vfft_policy_lm_axis_nat), unbanded (the doors run no band); at rank 1
 *     past the L1d the unbanded walk's chain; a Rader-stage column on few lanes as Bluestein with the turned pass;
 *   - the serial lane tile: sw = 0, the walk (the owner, 2026-11-01: no width rule beat it);
 *   - at T > 1, est_mt_2d's test: threaded once the plane, 16 N1 lanes bytes, passes
 *     L1d / EST_ND_MT_SERIAL_L1_DIV, except a column of 2 and a column through the turned prime pass; then the
 *     strips at est__msw_2d's width where every worker gets a vector of lanes (the rule below), else the
 *     threaded walk;
 *   - the block child: the lower-rank lane-major recipe on one thread, natural, in place.
 * THE BUILD (est_build_lm) takes the steps of c2c_estimate_nd_il.h: the pre-flight, the rows into a store held
 * in memory, the door's replay (est__replay), and the plan the door built checked against the recipe
 * (est__check_lm). An estimate create never hands back a plan that measured.
 *
 * POSITION IN vfft.c: included by c2c_estimate_il.h after c2c_estimate_nd_il.h, whose pieces it reuses. */
#ifndef VFFT_IL_C2C_ESTIMATE_LM_IL_H
#define VFFT_IL_C2C_ESTIMATE_LM_IL_H

/* THE THREE LANE-MAJOR RULES (the owner's, 2026-11-01; docs/roadmap/lane_major_estimate_plan.md section 8; each fitted
 * on the study's variant arms -- the estimate's own recipe changed in one place and built by est_build_lm -- against
 * the best arm of each cell, 212 cells at 1 and 8 threads):
 *   - A PRIME COLUMN ON FEW LANES: a column the chain plan gives a Rader stage runs as Bluestein with the turned
 *     prime pass while its lanes are at most est__lm_tpc_lanes: max(2, N / EST_LM_TPC_DIV) where p - 1 has a prime
 *     factor no chain radix carries, else 2. The switch grows with N where the stage's inner is costly: the turned
 *     pass won to 4 lanes (a tie at 8) at 1031 (1030 = 2 5 103), to 32 at 8 threads and every K at one thread at
 *     4099 (4098 = 2 3 683); where p - 1 is smooth the Rader stage wins from 3 lanes (131, 130 = 2 5 13; 2003 x 6
 *     at 8 threads, 2002 = 2 7 11 13: the turned pass 2.07 times the planner, an unseen cell, the p - 1 test added
 *     after it). Loss against the better of the two over the study's 50 prime cells: 1.009 (a fixed 4 lanes: 1.039).
 *   - THE STRIPS AT T > 1 only when every worker gets a vector of lanes (lanes >= the vector's complex count x T:
 *     2 x T on AVX2; 4 per worker scored 1.021),
 *     on a chain with no Rader stage (those walk faster threaded) whose strip of EST_LM_STRIP_MIN lanes fits the
 *     L2 (65536 rows walk); else the threaded walk. Loss against the best of serial, walk and strips over 103 cells
 *     at 8 threads: 1.019 (strips wherever no Bluestein: 1.188; at K = 4 the strips ran 3.5 - 7 times slower).
 *   - THE 1D CHAIN past the L1d: the unbanded walk's pick (est__col_unbanded) where the batch passes the L1d. Over the
 *     40 one-thread cells where the picks differ: 1.091 -> 1.018 against the better; at 8 threads 1.046 -> 1.042.
 *     Below the L1d est_col_plan's chain stays (the unbanded pick lost 1.215 there). */
#define EST_LM_TPC_DIV 128
#define EST_LM_STRIP_MIN 8

/* The lane-major batch's row, tokens in [brackets]. */
typedef struct est_planlm_s
{
    int rank;                    /* 1, 2 or 3 */
    int N1, N2, N3;              /* the shape; N2 = 0 at rank 1, N3 = 0 below rank 3 */
    size_t K;                    /* the batch: K transforms side by side */
    int T;                       /* the threads: 1, or T > 1 = the row keyed nthreads=T with the threaded verdict */
    int nat;                     /* 1 = a natural request, 0 = scrambled (rank 1 only) */
    est_col_t col;               /* the first axis: the column pass at w = the lanes, unbanded */
    int sw;                      /* [sw] the serial lane tile, 0 = the walk */
    int cmt;                     /* [cmt] T > 1: the column pass threaded (1) or serial (0) */
    int mtarm;                   /* [mtarm] T > 1, threaded: 1 the strips, 0 the threaded walk */
    int msw;                     /* [msw] the strips' width in lanes, 0 = each worker's whole range */
    est_recipe_t tpcp;           /* [tpc_] N1, in place, natural: col.tpc == 1 */
    struct est_planlm_s *child;  /* [rp_] rank 2: the 1D batch at (N2, K); rank 3: the 2D batch at (N2, N3, K);
                                  * one thread, natural, in place */
} est_planlm_t;

/* the lanes the first axis's column pass runs over */
static inline long long est__lm_lanes(const est_planlm_t *p)
{
    return p->rank == 1 ? (long long)p->K
           : p->rank == 2 ? (long long)p->N2 * (long long)p->K
                          : (long long)p->N2 * (long long)p->N3 * (long long)p->K;
}
static inline const char *est__lm_shape(const est_planlm_t *p, char *b, size_t n)
{
    if (p->rank == 1)
        snprintf(b, n, "%d x %zu", p->N1, p->K);
    else if (p->rank == 2)
        snprintf(b, n, "%dx%d x %zu", p->N1, p->N2, p->K);
    else
        snprintf(b, n, "%dx%dx%d x %zu", p->N1, p->N2, p->N3, p->K);
    return b;
}
/* the lanes up to which a column with a Rader stage at the prime p runs as Bluestein with the turned prime pass:
 * max(2, N / EST_LM_TPC_DIV) when p - 1 has a prime factor no chain radix carries (the stage's inner needs a prime
 * engine of its own, vfft_policy_prime_cell), else 2 */
static inline long long est__lm_tpc_lanes(int N, int p)
{
    return vfft_policy_prime_cell(p - 1) && N / EST_LM_TPC_DIV > 2 ? N / EST_LM_TPC_DIV : 2;
}
/* the recipe's block children, freed */
static inline void est_free_lm(est_planlm_t *p)
{
    if (!p || !p->child)
        return;
    est_free_lm(p->child);
    free(p->child);
    p->child = NULL;
}

/* est_plan_lm: the lane-major batch's recipe (the header's list). Returns 0 with *out filled (its block child
 * allocated: est_free_lm), or -1 with the reason. Pure. */
static inline int est_plan_lm(const est_machine_t *m, int rank, int N1, int N2, int N3, size_t K, int T, int nat,
                              est_planlm_t *out, char *why, size_t whylen)
{
    long long lanes;
    memset(out, 0, sizeof *out);
    if (est__machine_ok(m, why, whylen))
        return -1;
    if (rank < 1 || rank > 3 || K < 2 || N1 < 2 || (rank >= 2 && N2 < 2) || (rank == 3 && N3 < 2))
        return est_refuse(why, whylen, "a lane-major batch at rank %d of K = %zu: every extent and K at least 2", rank, K);
    if (rank >= 2 && !nat)
        return est_refuse(why, whylen, "a lane-major batch at rank %d: estimate plans natural order there", rank);
    out->rank = rank;
    out->N1 = N1;
    out->N2 = rank >= 2 ? N2 : 0;
    out->N3 = rank == 3 ? N3 : 0;
    out->K = K;
    out->T = T > 1 ? T : 1;
    out->nat = nat;
    lanes = est__lm_lanes(out);
    if (lanes > INT_MAX)
        return est_refuse(why, whylen, "a lane-major batch of %lld lanes: past the doors' lane count", lanes);
    if (est_col_plan(m, N1, lanes, nat, &out->col, why, whylen))
        return -1;
    if (rank == 1 && 16LL * N1 * lanes > m->l1d)
        (void)est__col_unbanded(m, &out->col);
    if (out->col.rd_n > 0 && lanes <= est__lm_tpc_lanes(N1, out->col.R[0]) && est__col_blu(m, &out->col, why, whylen))
        return -1;
    if (out->col.tpc == 1 && est__child_1d(m, N1, VFFT_PLACEMENT_IN, &out->tpcp, "the turned prime pass", why, whylen))
        return -1;
    if (out->T > 1 && 16LL * N1 * lanes > m->l1d / EST_ND_MT_SERIAL_L1_DIV && N1 != 2 && out->col.tpc != 1)
    {
        const int strips = out->col.blu <= 0 && !out->col.rd_n && lanes >= (long long)(m->lanes / 2) * out->T &&
                           16LL * N1 * EST_LM_STRIP_MIN <= m->l2;   /* m->lanes counts doubles: / 2 = complex lanes */
        out->cmt = 1;
        out->mtarm = strips;
        out->msw = strips ? est__msw_2d(m, N1, (int)lanes, out->T) : 0;
    }
    if (rank >= 2)
    {
        if (!(out->child = (est_planlm_t *)calloc(1, sizeof *out->child)))
            return est_refuse(why, whylen, "out of memory");
        if (est_plan_lm(m, rank - 1, N2, rank == 3 ? N3 : 0, 0, K, 1, 1, out->child, why, whylen))
        {
            free(out->child);
            out->child = NULL;
            return -1;
        }
    }
    return 0;
}

/* ── the pre-flight ─────────────────────────────────────────────────────── */

/* The lane-major batch as its door replays it: the column pass (est__preflight_col) over the batch's lanes,
 * unbanded; the serial tile the walk or a width of the door's ladder it admits (vfft_policy_lm_tile_ok); at
 * T > 1 a threaded verdict the door's race could have banked (_lm_race: the threaded walk, the strips over each
 * worker's lanes or at a ladder width, none through Bluestein or the turned prime pass); the turned pass's plan
 * and the block child. Returns 0, or -1 with the reason. */
static inline int est__preflight_lm(const est_planlm_t *p, char *why, size_t whylen)
{
    char w2[400], sh[64];
    const long long lanes = est__lm_lanes(p);
    int lad[6], n, i, in;
    est__lm_shape(p, sh, sizeof sh);
    if (est__preflight_col(&p->col, w2, sizeof w2))
        return est_refuse(why, whylen, "the lane-major batch %s: %s", sh, w2);
    if (p->col.N != p->N1 || p->col.w != lanes || p->col.nat != p->nat || p->col.wl)
        return est_refuse(why, whylen, "the lane-major batch %s: the column pass is not its pass over %lld lanes", sh, lanes);
    if (p->sw)
    {
        n = _il2d_sw_ladder(p->N1, (int)lanes, 1, lad, 6);
        for (i = 0, in = 0; i < n; i++)
            in |= lad[i] == p->sw;
        if (!in || !vfft_policy_lm_tile_ok(p->sw, (int)lanes, p->col.nat && p->col.nst > 1))
            return est_refuse(why, whylen, "the lane-major batch %s: the serial tile %d is not one its door races", sh, p->sw);
    }
    if (p->T > 1)
    {
        n = _il2d_sw_ladder(p->N1, (int)lanes, p->T, lad, 6);
        for (i = 0, in = (p->msw == 0); i < n; i++)
            in |= lad[i] == p->msw;
        if ((p->cmt != 0 && p->cmt != 1) || (!p->cmt && (p->mtarm || p->msw)) || (p->cmt && p->col.tpc == 1) ||
            (p->cmt && (p->mtarm == 1 ? p->col.blu > 0 || !in : p->mtarm != 0 || p->msw)))
            return est_refuse(why, whylen, "the lane-major batch %s T=%d: cmt=%d mtarm=%d msw=%d is no threaded verdict its "
                              "door serves", sh, p->T, p->cmt, p->mtarm, p->msw);
    }
    else if (p->cmt || p->mtarm || p->msw)
        return est_refuse(why, whylen, "the lane-major batch %s: a threaded verdict on one thread", sh);
    if (p->col.tpc == 1 && est__preflight_child(&p->tpcp, p->N1, VFFT_PLACEMENT_IN, "the turned prime pass", why, whylen))
        return -1;
    if (p->rank >= 2)
    {
        const est_planlm_t *c = p->child;
        if (!c || c->rank != p->rank - 1 || c->N1 != p->N2 || c->N2 != (p->rank == 3 ? p->N3 : 0) || c->K != p->K ||
            c->T != 1 || !c->nat)
            return est_refuse(why, whylen, "the lane-major batch %s: the block child is not the batch one rank down, one "
                              "thread, natural", sh);
        if (est__preflight_lm(c, w2, sizeof w2))
            return est_refuse(why, whylen, "the lane-major batch %s, its block child: %s", sh, w2);
    }
    else if (p->child)
        return est_refuse(why, whylen, "the lane-major batch %s: a block child at rank 1", sh);
    return 0;
}

/* ── the rows ───────────────────────────────────────────────────────────── */

/* the batch's row key, as its door keys it (lm_batch_il.h:336, lm_batch2d_il.h:130, lm_batch3d_il.h) */
static inline void est__lm_key(vw2_ilcol_key_t *ck, const est_planlm_t *p)
{
    memset(ck, 0, sizeof *ck);
    ck->rank = p->rank;
    ck->n0 = p->N1;
    if (p->rank == 1)
        ck->n1 = (int)p->K;
    else if (p->rank == 2)
    {
        ck->n1 = p->N2;
        ck->n2 = (int)p->K;
    }
    else
    {
        ck->n1 = p->N2;
        ck->n2 = p->N3;
        ck->q = (int)p->K;
    }
    ck->ord = p->nat ? VW2_ORD_NAT : VW2_ORD_SCR;
    ck->nthreads = p->T;
    ck->lm = 1;
}
/* est_put_rows_lm: the batch's row into st -- the column tokens (unbanded, no row route), sw, at T > 1 the
 * threaded verdict cmt mtarm msw, the turned pass's plan under tpc_, and the block child's rows under rp_ (written
 * into a store of their own and moved with the door's vfft_child_row_update, as est__put_child moves a 1D child).
 * Returns 0, or -1 with the reason. */
static inline int est_put_rows_lm(vw2_store_t *st, const est_planlm_t *p, char *why, size_t whylen)
{
    vw2_ilcol_key_t ck;
    vw2_key_t key;
    char sh[64];
    est__lm_key(&ck, p);
    est__lm_shape(p, sh, sizeof sh);
    if (est__put_col(st, &ck, &p->col, -1, 0) || vw2_ilcol_tok_seti(st, &ck, "sw", p->sw) != VW2_OK)
        return est_refuse(why, whylen, "the lane-major batch %s: its row was refused by the store's writers", sh);
    if (p->T > 1 &&
        (vw2_ilcol_tok_seti(st, &ck, "cmt", p->cmt) != VW2_OK || vw2_ilcol_tok_seti(st, &ck, "mtarm", p->mtarm) != VW2_OK ||
         vw2_ilcol_tok_seti(st, &ck, "msw", p->msw) != VW2_OK))
        return est_refuse(why, whylen, "the lane-major batch %s: its threaded verdict was refused by the store's writers", sh);
    vw2__ilcol_key(&ck, &key);
    if (p->col.tpc == 1 && est__put_child(st, &key, "tpc_", &p->tpcp, p->N1, VFFT_PLACEMENT_IN, why, whylen))
        return -1;
    if (p->child)
    {
        struct vfft_wisdom_s *C = vfft_child_store_new();
        int ok;
        if (!C)
            return est_refuse(why, whylen, "out of memory");
        if (est_put_rows_lm(&C->vw2, p->child, why, whylen))
        {
            vfft_child_store_free(C);
            return -1;
        }
        ok = vfft_child_row_update(st, &key, "rp_", C);
        vfft_child_store_free(C);
        if (!ok)
            return est_refuse(why, whylen, "the lane-major batch %s: its block child could not be put on its row", sh);
    }
    return 0;
}

/* ── the plan the door built ────────────────────────────────────────────── */

/* the lane-major batch against its recipe: the shape, the column pass, the serial tile, at T > 1 the threaded
 * verdict, the turned pass's plan, the block child */
static inline int est__check_lm(const struct vfft_plan_s *h, const est_planlm_t *p, char *why, size_t whylen)
{
    char sh[64];
    est__lm_shape(p, sh, sizeof sh);
    if (!h || !h->lm || h->N != p->N1 || h->N2 != p->N2 || h->N3 != p->N3 || h->K != p->K)
        return est_refuse(why, whylen, "the door built no lane-major batch %s", sh);
    if (est__check_col(&h->il2d_col, &p->col, why, whylen))
        return -1;
    if (h->il2d_col.wc != p->sw)
        return est_refuse(why, whylen, "the lane-major batch %s: the door serves the serial tile %d, the recipe %d", sh,
                          h->il2d_col.wc, p->sw);
    if (p->T > 1 && (h->nthreads != p->T || h->il2d_col.colmt != p->cmt || h->il2d_col.natarm != p->mtarm ||
                     h->il2d_col.msw != p->msw))
        return est_refuse(why, whylen, "the lane-major batch %s: the door serves T=%d cmt=%d mtarm=%d msw=%d, the recipe T=%d "
                          "cmt=%d mtarm=%d msw=%d", sh, h->nthreads, h->il2d_col.colmt, h->il2d_col.natarm, h->il2d_col.msw,
                          p->T, p->cmt, p->mtarm, p->msw);
    if (p->col.tpc == 1 &&
        est__check_child(h->il2d_col.tpcplan, &p->tpcp, p->N1, VFFT_PLACEMENT_IN, "turned prime pass's plan", why, whylen))
        return -1;
    if (p->child)
    {
        char w2[400];
        if (est__check_lm(h->lm_row, p->child, w2, sizeof w2))
            return est_refuse(why, whylen, "the lane-major batch %s, its block child: %s", sh, w2);
    }
    return 0;
}

/* ── the build ──────────────────────────────────────────────────────────── */

/* est_build_lm: the recipe p of the lane-major request cfg -> a plan built with nothing measured, or NULL with
 * the reason in why and the outcome in rep (may be NULL). cfg is the request as the front door translated it:
 * interleaved c2c, the lane-major geometry at K = p->K, the recipe's rank, shape, order and thread count; its
 * store and save flag are not used. Holds the create lock from the pre-flight to the end. The caller owns the
 * plan (vfft_destroy). */
static inline struct vfft_plan_s *est_build_lm(const est_planlm_t *p, const vfft_cfg_t *cfg, est_build_report_t *rep,
                                               char *why, size_t whylen)
{
    est_build_report_t lrep;
    struct vfft_wisdom_s *S = NULL;
    struct vfft_plan_s *h = NULL;
    est__meter_t m0;
    if (!rep)
        rep = &lrep;
    memset(rep, 0, sizeof *rep);
    rep->status = EST_REFUSED;
    if (why && whylen)
        why[0] = 0;
    if (!p || !cfg || cfg->transform != VFFT_C2C || cfg->layout != VFI_LAYOUT_IL ||
        cfg->batch_geom != VFFT_BATCH_LANE_MAJOR || !vfft_policy_batch_lane_major(cfg) || cfg->howmany != p->K ||
        cfg->dims != p->rank || cfg->n[0] != p->N1 || (p->rank >= 2 && cfg->n[1] != p->N2) ||
        (p->rank == 3 && cfg->n[2] != p->N3) || (cfg->nthreads > 1 ? cfg->nthreads : 1) != p->T ||
        (cfg->order == VFI_ORDER_SCRAMBLED) == (p->nat != 0))
    {
        est_refuse(why, whylen, "estimate builds a lane-major interleaved c2c batch at the recipe's shape, K, order and "
                   "thread count");
        return NULL;
    }
    vfft_env_init();          /* env.h:96: the pre-flight's tables in the create's FP mode */
    _vfft_create_lock();      /* recursive: the pre-flight touches the create's process statics */
    _vfft_create_depth++;     /* the whole build inside a create: a clock read enters a scope */
    est__meter_read(&m0);
    if (est__preflight_lm(p, why, whylen))
        goto out;
    if (!(S = vfft_child_store_new()))
    {
        est_refuse(why, whylen, "out of memory");
        goto out;
    }
    if (est_put_rows_lm(&S->vw2, p, why, whylen))
        goto out;
    h = est__replay(S, cfg, &m0, rep, why, whylen);
    if (h && est__check_lm(h, p, why, whylen))
    {
        rep->status = EST_NOPLAN;
        vfft_destroy((vfft_plan)h);
        h = NULL;
    }
    else if (h)
        rep->status = EST_BUILT;
out:
    vfft_child_store_free(S);
    if (--_vfft_create_depth == 0)
        _vfft_scope_create_done();   /* race_scope.h: leave a scope a clock read entered */
    _vfft_create_unlock();
    return h;
}

/* THE FRONT DOOR'S ESTIMATE CREATE of a lane-major batch (_vfft_create_estimate): the recipe and the build.
 * NULL with the reason on the error record (vfft_error). */
static vfft_plan _vfft_create_estimate_lm(const est_machine_t *m, const vfft_cfg_t *cfg)
{
    char why[640];
    struct vfft_plan_s *h = NULL;
    est_planlm_t p;
    if (!vfft_policy_batch_lane_major(cfg) || cfg->transform != VFFT_C2C)
        snprintf(why, sizeof why, "the lane-major door does not serve this request (it serves C2C at howmany >= 2 "
                 "with every extent at least 2)");
    else if (!est_plan_lm(m, cfg->dims, cfg->n[0], cfg->dims >= 2 ? cfg->n[1] : 0, cfg->dims == 3 ? cfg->n[2] : 0,
                          cfg->howmany, cfg->nthreads, cfg->order != VFI_ORDER_SCRAMBLED, &p, why, sizeof why))
    {
        h = est_build_lm(&p, cfg, NULL, why, sizeof why);
        est_free_lm(&p);
    }
    if (!h)
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE: %s", why);
    return (vfft_plan)h;
}

#endif /* VFFT_IL_C2C_ESTIMATE_LM_IL_H */
