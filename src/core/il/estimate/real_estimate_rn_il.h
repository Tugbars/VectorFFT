/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* real_estimate_rn_il.h — the ESTIMATE create of one R2C or C2R transform of rank 4 to 7 (VFFT_FLAG_ESTIMATE), or of
 * a contiguous batch of them: the door's rows written from the formulas and replayed by it with nothing measured.
 *
 * THE DOOR (il/rank3/fftnd_realn_il.h). r2c: the N1 blocks behind the first axis through the real plan one rank down
 * (the rank-3 real tier at rank 4, the door's own plan above; its rows under rp_), then one complex column pass of N1
 * rows over the spectrum block's width; c2r the mirror. The create plans on the row t=r2c n=<shape> q=1 lay=lm
 * nthreads=T: the column builder's verdicts, the serial lane tile sw= and at T > 1 the threaded verdict cmt= mtarm=
 * msw= (_lm_race), the block child under rp_.
 *
 * THE RECIPE (est_plan_realn) carries no rule of its own:
 *   - the first axis: the lane-major real batch's first-axis recipe at its lanes (est__lmr_first_axis,
 *     real_estimate_lm_il.h: the batch at one transform);
 *   - the block child: at rank 4 the 3D real recipe at one thread, out of place (est_plan_3d_real,
 *     real_estimate_nd_il.h); above it this recipe one rank down at one thread.
 * THE BUILD (est_build_realn) takes the steps of the other estimate builds: the pre-flight, the rows into a store held
 * in memory, the door's replay (est__replay), the plan the door built checked against the recipe.
 * THE CONTIGUOUS BATCH (howmany = K > 1): the queue over the plan of one transform, as est_build_real_ndb builds it
 * at rank 2 and 3 -- on T threads the copies' one-thread recipe and the way on the batch's own row, the real
 * batches' rule (est__real_nd_batch_whole) at one transform's spectrum bytes.
 *
 * POSITION IN vfft.c: included by c2c_estimate_il.h after real_estimate_lm_il.h, whose pieces it reuses. */
#ifndef VFFT_IL_REAL_ESTIMATE_RN_IL_H
#define VFFT_IL_REAL_ESTIMATE_RN_IL_H

/* The recipe of one real transform of rank 4 to 7. */
typedef struct est_planrr_s
{
    int rank;                  /* 4 to 7 */
    int n[7];                  /* the shape, the last extent the real one */
    int T, c2r;                /* the threads; the direction */
    est_planlm_t pass;         /* the first axis at N1 over the spectrum block's lanes, natural: a rank-1 lane-major
                                * recipe at those lanes */
    est_plan3dr_t *c3;         /* [rp_] rank 4: the block child, the 3D real recipe at (N2, N3, N4), one thread */
    struct est_planrr_s *cn;   /* [rp_] rank 5 to 7: the block child, this recipe one rank down, one thread */
} est_planrr_t;

static inline void est_free_realn(est_planrr_t *p)
{
    if (!p)
        return;
    est_free_lm(&p->pass);
    free(p->c3);
    p->c3 = NULL;
    if (p->cn)
    {
        est_free_realn(p->cn);
        free(p->cn);
        p->cn = NULL;
    }
}
/* the spectrum block's width in complex: N2 .. N(r-1) (Nr / 2 + 1), the lanes of the first axis's pass */
static inline long long est__realn_lanes(const est_planrr_t *p)
{
    long long l = (long long)(p->n[p->rank - 1] / 2 + 1);
    int i;
    for (i = 1; i < p->rank - 1; i++)
        l *= (long long)p->n[i];
    return l;
}
static inline const char *est__realn_shape(const est_planrr_t *p, char *b, size_t n)
{
    size_t off = 0;
    int i;
    if (n)
        b[0] = 0;
    for (i = 0; i < p->rank && i < 7 && off < n; i++)
    {
        const int w = snprintf(b + off, n - off, "%s%d", i ? "x" : "", p->n[i]);
        if (w < 0)
            break;
        off += (size_t)w;
    }
    return b;
}

/* est_plan_realn: the machine, the shape n[0 .. rank - 1], the direction and the threads -> the recipe; top = 0 for
 * a block child. Returns 0 with *out filled (its children allocated: est_free_realn), or -1 with the reason. Pure. */
static inline int est_plan_realn(const est_machine_t *m, int rank, const int *n, int c2r, int T, int top,
                                 est_planrr_t *out, char *why, size_t whylen)
{
    long long lanes;
    int i;
    memset(out, 0, sizeof *out);
    if (est__machine_ok(m, why, whylen))
        return -1;
    if (rank < 4 || rank > 7)
        return est_refuse(why, whylen, "a real transform of rank %d: this recipe plans ranks 4 to 7", rank);
    for (i = 0; i < rank; i++)
        if (n[i] < 2)
            return est_refuse(why, whylen, "a real transform of rank %d: every extent at least 2", rank);
    out->rank = rank;
    memcpy(out->n, n, sizeof(int) * (size_t)rank);
    out->T = T > 1 ? T : 1;
    out->c2r = c2r != 0;
    lanes = est__realn_lanes(out);
    if (est__lmr_first_axis(m, n[0], lanes, out->T, top, &out->pass, why, whylen))
        return -1;
    if (rank == 4)
    {
        if (!(out->c3 = (est_plan3dr_t *)calloc(1, sizeof *out->c3)))
            return est_refuse(why, whylen, "out of memory");
        if (est_plan_3d_real(m, n[1], n[2], n[3], out->c2r, 0, 1, out->c3, why, whylen))
        {
            free(out->c3);
            out->c3 = NULL;
            return -1;
        }
        return 0;
    }
    if (!(out->cn = (est_planrr_t *)calloc(1, sizeof *out->cn)))
        return est_refuse(why, whylen, "out of memory");
    if (est_plan_realn(m, rank - 1, n + 1, c2r, 1, 0, out->cn, why, whylen))
    {
        est_free_realn(out->cn);
        free(out->cn);
        out->cn = NULL;
        return -1;
    }
    return 0;
}

/* the door's row key (fftnd_realn_il.h): the real cell of that rank and shape at q = 1 on the lay=lm axis */
static inline void est__realn_key(vw2_ilcol_key_t *ck, const est_planrr_t *p)
{
    int i;
    memset(ck, 0, sizeof *ck);
    ck->rank = p->rank;
    ck->n0 = p->n[0];
    ck->n1 = (size_t)p->n[1];
    ck->n2 = (size_t)p->n[2];
    for (i = 3; i < p->rank; i++)
        ck->nx[i - 3] = p->n[i];
    ck->q = 1;
    ck->ord = VW2_ORD_NAT;
    ck->real = 1;
    ck->nthreads = p->T;
    ck->lm = 1;
}

/* the recipe's pre-flight: the first axis's pass as the door replays it (est__preflight_lm, on a copy spelled the C2C
 * way: its forms are the real pool's), at its lanes; the block child one rank down at one thread */
static inline int est__preflight_realn(const est_planrr_t *p, char *why, size_t whylen)
{
    char sh[96], w2[400], fr[64];
    const long long lanes = est__realn_lanes(p);
    est_planlm_t q = p->pass;
    est__realn_shape(p, sh, sizeof sh);
    if (p->pass.rank != 1 || p->pass.N1 != p->n[0] || (long long)p->pass.K != lanes || p->pass.T != p->T || !p->pass.nat)
        return est_refuse(why, whylen, "the real transform %s: the first axis is not its pass at %d over %lld lanes", sh,
                          p->n[0], lanes);
    est__col_forms_real(q.col.R, q.col.nst, 0, fr, sizeof fr);
    if (strcmp(fr, q.col.forms))
        return est_refuse(why, whylen, "the real transform %s: the first axis's forms \"%s\" are not the real pool's "
                          "\"%s\"", sh, q.col.forms, fr);
    est__col_forms(q.col.R, q.col.nst, q.col.forms, sizeof q.col.forms);
    if (est__preflight_lm(&q, w2, sizeof w2))
        return est_refuse(why, whylen, "the real transform %s: %s", sh, w2);
    if (p->rank == 4)
    {
        if (!p->c3 || p->cn || p->c3->N1 != p->n[1] || p->c3->N2 != p->n[2] || p->c3->N3 != p->n[3] || p->c3->T != 1 ||
            p->c3->ip || !p->c3->c2r != !p->c2r)
            return est_refuse(why, whylen, "the real transform %s: the block child is not the 3D real recipe at one "
                              "thread, out of place, in its direction", sh);
        return 0;
    }
    if (!p->cn || p->c3 || p->cn->rank != p->rank - 1 || memcmp(p->cn->n, p->n + 1, sizeof(int) * (size_t)(p->rank - 1)) ||
        p->cn->T != 1 || !p->cn->c2r != !p->c2r)
        return est_refuse(why, whylen, "the real transform %s: the block child is not the recipe one rank down at one "
                          "thread, in its direction", sh);
    if (est__preflight_realn(p->cn, w2, sizeof w2))
        return est_refuse(why, whylen, "the real transform %s, its block child: %s", sh, w2);
    return 0;
}

/* est_put_rows_realn: the door's row into st -- the first axis's verdicts (est__put_lm_row) and the block child's rows
 * under rp_ (written into a store of their own and moved with the door's vfft_child_row_update). 0, or -1 with the
 * reason. */
static inline int est_put_rows_realn(vw2_store_t *st, const est_planrr_t *p, char *why, size_t whylen)
{
    vw2_ilcol_key_t ck;
    vw2_key_t key;
    struct vfft_wisdom_s *C;
    char sh[96];
    int bad, ok;
    est__realn_key(&ck, p);
    est__realn_shape(p, sh, sizeof sh);
    if (est__put_lm_row(st, &ck, &p->pass, sh, why, whylen))
        return -1;
    if (!(C = vfft_child_store_new()))
        return est_refuse(why, whylen, "out of memory");
    bad = p->rank == 4 ? est_put_rows_3d_real(&C->vw2, p->c3, why, whylen) : est_put_rows_realn(&C->vw2, p->cn, why, whylen);
    if (bad)
    {
        vfft_child_store_free(C);
        return -1;
    }
    vw2__ilcol_key(&ck, &key);
    ok = vfft_child_row_update(st, &key, "rp_", C);
    vfft_child_store_free(C);
    if (!ok)
        return est_refuse(why, whylen, "the real transform %s: its block child could not be put on its row", sh);
    return 0;
}

/* the plan the door built against the recipe: the shape and direction, the first axis's pass (est__check_lm_pass),
 * the lean door's entry in that direction, the block child. 0, or -1 with the reason. */
static inline int est__check_realn(const struct vfft_plan_s *h, const est_planrr_t *p, char *why, size_t whylen)
{
    char sh[112], b[96], w2[400];
    snprintf(sh, sizeof sh, "%s %s", p->c2r ? "C2R" : "R2C", est__realn_shape(p, b, sizeof b));
    if (!h || !h->lm || h->transform != (p->c2r ? VFFT_C2R : VFFT_R2C) || h->K != 1 || h->N != p->n[0] ||
        h->N2 != p->n[1] || h->N3 != p->n[2] || h->N4 != p->n[3] || h->N5 != (p->rank > 4 ? p->n[4] : 0) ||
        h->N6 != (p->rank > 5 ? p->n[5] : 0) || h->N7 != (p->rank > 6 ? p->n[6] : 0))
        return est_refuse(why, whylen, "the door built no real transform %s", sh);
    if (est__check_lm_pass(h, &p->pass, sh, why, whylen))
        return -1;
    if (!h->k1_go[p->c2r ? VFFT_BACKWARD : VFFT_FORWARD])
        return est_refuse(why, whylen, "the real transform %s has no lean entry (k1_go)", sh);
    if (!h->lm_row)
        return est_refuse(why, whylen, "the real transform %s has no block child", sh);
    if (p->rank == 4 ? !est__check_3d_real(h->lm_row, p->c3, w2, sizeof w2) : est__check_realn(h->lm_row, p->cn, w2, sizeof w2))
        return est_refuse(why, whylen, "the real transform %s, its block child: %s", sh, w2);
    return 0;
}

/* est_build_realn: the recipe p of the request cfg -> a plan built with nothing measured, or NULL with the reason in
 * why and the outcome in rep (may be NULL). cfg is the request as the front door translated it: one interleaved R2C or
 * C2R transform at the recipe's rank, shape, direction and thread count, out of place, natural, or a contiguous batch
 * of them; its store and save flag are not used. A batch on T threads also takes the copies' one-thread recipe (q)
 * and the way (whole: 1 whole transforms per thread, 0 all threads on each transform in turn). Holds the create lock
 * from the pre-flight to the end. The caller owns the plan (vfft_destroy). */
static inline struct vfft_plan_s *est_build_realn(const est_planrr_t *p, const est_planrr_t *q, int whole,
                                                  const vfft_cfg_t *cfg, est_build_report_t *rep, char *why, size_t whylen)
{
    est_build_report_t lrep;
    struct vfft_wisdom_s *S = NULL;
    struct vfft_plan_s *h = NULL;
    const struct vfft_plan_s *h1;
    est__meter_t m0;
    const int T = cfg && cfg->nthreads > 1 ? cfg->nthreads : 1, batch = cfg && cfg->howmany > 1 && T > 1;
    if (!rep)
        rep = &lrep;
    memset(rep, 0, sizeof *rep);
    rep->status = EST_REFUSED;
    if (why && whylen)
        why[0] = 0;
    if (!p || !cfg || (cfg->transform != VFFT_R2C && cfg->transform != VFFT_C2R) ||
        (cfg->transform == VFFT_C2R) != (p->c2r != 0) || cfg->layout != VFI_LAYOUT_IL || cfg->dims != p->rank ||
        memcmp(cfg->n, p->n, sizeof(int) * (size_t)p->rank) || cfg->placement == VFFT_INPLACE ||
        (cfg->howmany > 1 && cfg->batch_geom == VFFT_BATCH_LANE_MAJOR) ||
        cfg->order == VFI_ORDER_SCRAMBLED || T != p->T ||
        (batch ? !q || q->T != 1 || q->rank != p->rank || memcmp(q->n, p->n, sizeof(int) * (size_t)p->rank) ||
                     !q->c2r != !p->c2r || (whole != 0 && whole != 1)
               : q || whole))
    {
        est_refuse(why, whylen, "estimate builds one interleaved R2C or C2R transform of rank 4 to 7, out of place, in "
                   "natural order, at the recipe's shape, direction and thread count, or a contiguous batch of them "
                   "(on more than one thread with its copies' one-thread recipe)");
        return NULL;
    }
    vfft_env_init();          /* the pre-flight's tables in the create's FP mode */
    _vfft_create_lock();      /* recursive: the pre-flight touches the create's process statics */
    _vfft_create_depth++;     /* the whole build inside a create: a clock read enters a scope */
    est__meter_read(&m0);
    if (est__preflight_realn(p, why, whylen) || (batch && est__preflight_realn(q, why, whylen)))
        goto out;
    if (!(S = vfft_child_store_new()))
    {
        est_refuse(why, whylen, "out of memory");
        goto out;
    }
    if (est_put_rows_realn(&S->vw2, p, why, whylen))
        goto out;
    if (batch)
    {   /* the copies' one-thread row; the way on the batch's own row, as _pq_mt_replay_or_race reads it */
        vw2_key_t key;
        int i;
        if (est_put_rows_realn(&S->vw2, q, why, whylen))
            goto out;
        vw2__pq_key(&key, p->c2r ? VW2_T_C2R : VW2_T_R2C, p->rank, cfg->n[0], cfg->n[1], cfg->n[2], cfg->howmany,
                    vfft_policy_ord_rankn(cfg), VW2_PL_OOP, _vw2_lay_of(cfg), T);
        for (i = 3; i < p->rank; i++)   /* the extents after the third (plane_queue.h, _pq_row_key) */
            key.n[i] = cfg->n[i];
        if (vw2_pq_bank(&S->vw2, &key, whole, 0.0) != VW2_OK)
        {
            est_refuse(why, whylen, "the batch's row (pq) was refused at %d threads", T);
            goto out;
        }
    }
    h = est__replay(S, cfg, &m0, rep, why, whylen);
    if (h && (est__batch_one(h, cfg, whole, &h1, why, whylen) || est__check_realn(h1, p, why, whylen) ||
              (whole && est__check_realn(h->pq_w[0], q, why, whylen))))
    {   /* with whole transforms per thread the first copy is the one-thread recipe's plan too */
        rep->status = EST_NOPLAN;
        vfft_destroy((vfft_plan)h);
        h = NULL;
    }
    else if (h)
        rep->status = EST_BUILT;
out:
    vfft_child_store_free(S);
    if (--_vfft_create_depth == 0)
        _vfft_scope_create_done();   /* leave a scope a clock read entered */
    _vfft_create_unlock();
    return h;
}

/* THE FRONT DOOR'S ESTIMATE CREATE of a real transform of rank 4 to 7 (_vfft_create_estimate): the recipe and the
 * build. NULL with the reason on the error record (vfft_error). */
static vfft_plan _vfft_create_estimate_realn(const est_machine_t *m, const vfft_cfg_t *cfg)
{
    char why[640];
    struct vfft_plan_s *h = NULL;
    const int T = cfg->nthreads > 1 ? cfg->nthreads : 1, batch = cfg->howmany > 1 && T > 1, c2r = cfg->transform == VFFT_C2R;
    est_planrr_t *p = (est_planrr_t *)calloc(1, sizeof *p);   /* off the stack */
    est_planrr_t *q = batch ? (est_planrr_t *)calloc(1, sizeof *q) : NULL;   /* the queue's copies: the plan on one thread */
    if (!p || (batch && !q))
        snprintf(why, sizeof why, "out of memory");
    else if (cfg->placement == VFFT_INPLACE || cfg->order == VFI_ORDER_SCRAMBLED ||
             (cfg->howmany > 1 && cfg->batch_geom == VFFT_BATCH_LANE_MAJOR))
        snprintf(why, sizeof why, "R2C and C2R of rank %d are served out of place, in natural order, one transform or a "
                 "contiguous batch", cfg->dims);
    else if (!est_plan_realn(m, cfg->dims, cfg->n, c2r, T, 1, p, why, sizeof why) &&
             (!q || !est_plan_realn(m, cfg->dims, cfg->n, c2r, 1, 1, q, why, sizeof why)))
    {
        int whole = 0, i;
        if (batch)
        {   /* the way: the real batches' rule at one transform's spectrum bytes */
            size_t bytes = 16 * ((size_t)cfg->n[cfg->dims - 1] / 2 + 1);
            for (i = 0; i < cfg->dims - 1; i++)
                bytes *= (size_t)cfg->n[i];
            whole = est__real_nd_batch_whole(m, bytes, cfg->howmany, T);
        }
        h = est_build_realn(p, q, whole, cfg, NULL, why, sizeof why);
    }
    est_free_realn(p);
    free(p);
    if (q)
    {
        est_free_realn(q);
        free(q);
    }
    if (!h)
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE: %s", why);
    return (vfft_plan)h;
}

#endif /* VFFT_IL_REAL_ESTIMATE_RN_IL_H */
