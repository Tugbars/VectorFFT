/* real_estimate_lm_il.h — the ESTIMATE create of a LANE-MAJOR REAL batch (VFFT_FLAG_ESTIMATE on R2C or C2R at
 * howmany = K >= 2 in the lane-major geometry, the real batch's DEFAULT) at rank 1: the batch's row written from the
 * formulas and replayed by the door with nothing measured (docs/roadmap/real_estimate_plan.md, stage 2).
 *
 * THE DOOR (il/rank1/lm_batch_il.h, _vfft_create_lmr_il): an even N >= 4. Element e of transform t at x[e K + t] (the
 * reals) and bin f at z[2 (f K + t)]. The pack to M = N / 2 complex rows, one column pass of M rows over K lanes (the
 * column builder at M in its scrambled class: the fold reads row k where the pass leaves it), the fold; C2R the
 * mirror. Both directions and both placements share one row, t=r2c n=N q=K ord=nat place=oop lay=lm nthreads=T (the
 * column key with real set): the column builder's verdicts at M (the chain, its forms, a Rader stage, Bluestein, the
 * turned prime pass and its plan under tpc_), the serial lane tile sw= (0: the whole plane as one fused strip where
 * the scratch holds it, else the three-phase form) and at T > 1 the threaded verdict cmt= mtarm= msw= (_lm_race).
 *
 * THE RECIPE (est_plan_lmr): the C2C lane-major recipe of rank 1 at M over K lanes in the scrambled class (est_plan_lm:
 * est_col_plan at w = K, the unbanded walk's chain past the L1d, a Rader column on few lanes as Bluestein with the
 * turned prime pass, sw = 0, at T > 1 est_mt_2d's test with the strips rule), no rule of its own: the plan's stage 2
 * takes one only if the study shows the pack and the fold change it.
 * THE BUILD (est_build_lmr) takes est_build_lm's steps on the real row: the pre-flight, the row into a store held in
 * memory, the door's replay (est__replay), and the plan the door built checked against the recipe (est__check_lmr),
 * the lean door's entry in the transform's direction with it. An estimate create never hands back a plan that
 * measured.
 *
 * POSITION IN vfft.c: included by c2c_estimate_il.h after real_estimate_nd_il.h; it reuses the lane-major file's
 * pieces (est_plan_lm, est__preflight_lm, est__put_lm_row, est__check_lm_pass). */
#ifndef VFFT_IL_REAL_ESTIMATE_LM_IL_H
#define VFFT_IL_REAL_ESTIMATE_LM_IL_H

/* The lane-major real batch's recipe: the shape and the column pass, both directions' (they share the row). */
typedef struct
{
    int N;               /* the real length: even, at least 4 */
    size_t K;            /* the batch: K transforms side by side */
    int T;               /* the threads: 1, or T > 1 = the row keyed nthreads=T with the threaded verdict */
    est_planlm_t pass;   /* the column pass: the C2C lane-major recipe of rank 1 at M = N / 2 over K lanes, scrambled */
} est_planlmr_t;

/* est_plan_lmr: the lane-major real batch's recipe (the header's list). Returns 0 with *out filled, or -1 with the
 * reason. Pure. */
static inline int est_plan_lmr(const est_machine_t *m, int N, size_t K, int T, est_planlmr_t *out, char *why,
                               size_t whylen)
{
    memset(out, 0, sizeof *out);
    if (N < 4 || (N & 1) || K < 2)
        return est_refuse(why, whylen, "a lane-major real batch of N = %d, K = %zu: the door serves an even N of at least "
                          "4 and K of at least 2 (an odd N has no real column engine: name VFFT_BATCH_CONTIGUOUS)", N, K);
    out->N = N;
    out->K = K;
    out->T = T > 1 ? T : 1;
    return est_plan_lm(m, 1, N / 2, 0, 0, K, out->T, 0, &out->pass, why, whylen);
}

/* the batch's row key, as its door keys it (lm_batch_il.h, _vfft_create_lmr_il) */
static inline void est__lmr_key(vw2_ilcol_key_t *ck, const est_planlmr_t *p)
{
    memset(ck, 0, sizeof *ck);
    ck->rank = 1;
    ck->n0 = p->N;
    ck->n1 = (int)p->K;
    ck->ord = VW2_ORD_NAT;
    ck->real = 1;
    ck->nthreads = p->T;
    ck->lm = 1;
}

/* the plan the door built against the recipe: the real batch at N x K in the request's direction, its column pass
 * (est__check_lm_pass), the lean door's entry in that direction */
static inline int est__check_lmr(const struct vfft_plan_s *h, const est_planlmr_t *p, int c2r, char *why, size_t whylen)
{
    char sh[64];
    snprintf(sh, sizeof sh, "%s %d x %zu", c2r ? "C2R" : "R2C", p->N, p->K);
    if (!h || !h->lm || h->transform != (c2r ? VFFT_C2R : VFFT_R2C) || h->N != p->N || h->N2 || h->lm_M != p->N / 2 ||
        h->K != p->K)
        return est_refuse(why, whylen, "the door built no lane-major real batch %s", sh);
    if (est__check_lm_pass(h, &p->pass, sh, why, whylen))
        return -1;
    if (!h->k1_go[c2r ? VFFT_BACKWARD : VFFT_FORWARD])
        return est_refuse(why, whylen, "the lane-major real batch %s has no lean entry (k1_go)", sh);
    return 0;
}

/* est_build_lmr: the recipe p of the lane-major real request cfg -> a plan built with nothing measured, or NULL with
 * the reason in why and the outcome in rep (may be NULL). cfg is the request as the front door translated it: an
 * interleaved R2C or C2R batch at rank 1 in the lane-major geometry (DEFAULT or named), K = p->K, the recipe's N and
 * thread count; its store and save flag are not used. Holds the create lock from the pre-flight to the end. The caller
 * owns the plan (vfft_destroy). */
static inline struct vfft_plan_s *est_build_lmr(const est_planlmr_t *p, const vfft_cfg_t *cfg, est_build_report_t *rep,
                                                char *why, size_t whylen)
{
    est_build_report_t lrep;
    struct vfft_wisdom_s *S = NULL;
    struct vfft_plan_s *h = NULL;
    est__meter_t m0;
    vw2_ilcol_key_t ck;
    char sh[64], w2[400];
    if (!rep)
        rep = &lrep;
    memset(rep, 0, sizeof *rep);
    rep->status = EST_REFUSED;
    if (why && whylen)
        why[0] = 0;
    if (!p || !cfg || (cfg->transform != VFFT_R2C && cfg->transform != VFFT_C2R) || cfg->layout != VFI_LAYOUT_IL ||
        cfg->batch_geom == VFFT_BATCH_CONTIGUOUS || !vfft_policy_batch_lane_major(cfg) || cfg->howmany != p->K ||
        cfg->dims > 1 || cfg->n[0] != p->N || (cfg->nthreads > 1 ? cfg->nthreads : 1) != p->T)
    {
        est_refuse(why, whylen, "estimate builds a lane-major interleaved real batch at the recipe's N, K and thread count");
        return NULL;
    }
    snprintf(sh, sizeof sh, "%d x %zu", p->N, p->K);
    vfft_env_init();          /* env.h:96: the pre-flight's tables in the create's FP mode */
    _vfft_create_lock();      /* recursive: the pre-flight touches the create's process statics */
    _vfft_create_depth++;     /* the whole build inside a create: a clock read enters a scope */
    est__meter_read(&m0);
    if (p->pass.rank != 1 || p->pass.N1 != p->N / 2 || p->pass.K != p->K || p->pass.T != p->T || p->pass.nat)
    {
        est_refuse(why, whylen, "the lane-major real batch %s: the column pass is not the scrambled pass at M = %d over %zu "
                   "lanes", sh, p->N / 2, p->K);
        goto out;
    }
    if (est__preflight_lm(&p->pass, w2, sizeof w2))
    {
        est_refuse(why, whylen, "the lane-major real batch %s: %s", sh, w2);
        goto out;
    }
    if (!(S = vfft_child_store_new()))
    {
        est_refuse(why, whylen, "out of memory");
        goto out;
    }
    est__lmr_key(&ck, p);
    if (est__put_lm_row(&S->vw2, &ck, &p->pass, sh, why, whylen))
        goto out;
    h = est__replay(S, cfg, &m0, rep, why, whylen);
    if (h && est__check_lmr(h, p, cfg->transform == VFFT_C2R, why, whylen))
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

/* THE FRONT DOOR'S ESTIMATE CREATE of a lane-major real batch at rank 1 (_vfft_create_estimate): the recipe and the
 * build. NULL with the reason on the error record (vfft_error). */
static vfft_plan _vfft_create_estimate_lmr(const est_machine_t *m, const vfft_cfg_t *cfg)
{
    char why[640];
    struct vfft_plan_s *h = NULL;
    est_planlmr_t p;
    if (!est_plan_lmr(m, cfg->n[0], cfg->howmany, cfg->nthreads, &p, why, sizeof why))
    {
        h = est_build_lmr(&p, cfg, NULL, why, sizeof why);
        est_free_lm(&p.pass);
    }
    if (!h)
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE: %s", why);
    return (vfft_plan)h;
}

#endif /* VFFT_IL_REAL_ESTIMATE_LM_IL_H */
