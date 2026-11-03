/* real_estimate_lm_il.h — the ESTIMATE create of a LANE-MAJOR REAL batch (VFFT_FLAG_ESTIMATE on R2C or C2R at
 * howmany = K >= 2 in the lane-major geometry: the real batch's DEFAULT at rank 1, named at rank 2 and 3) at rank 1, 2
 * and 3: the batch's row written from the formulas and replayed by the door with nothing measured
 * (docs/roadmap/real_estimate_plan.md, stages 2 and 4).
 *
 * THE DOORS. Rank 1 (il/rank1/lm_batch_il.h, _vfft_create_lmr_il): an even N >= 4. Element e of transform t at
 * x[e K + t] (the reals) and bin f at z[2 (f K + t)]. The pack to M = N / 2 complex rows, one column pass of M rows over
 * K lanes (the column builder at M in its scrambled class: the fold reads row k where the pass leaves it), the fold;
 * C2R the mirror. Rank 2 (il/rank2/lm_batch2dr_il.h): N1 blocks of the 1D real batch at (N2, K) on one thread (the
 * block child, its rows under rp_), then the first axis: a complex column pass of N1 rows over the spectrum's H K lanes,
 * H = N2 / 2 + 1, natural. Rank 3 (il/rank3/lm_batch3dr_il.h): N1 blocks of the 2D real batch at (N2, N3, K), then the
 * first axis over N2 H K lanes, H = N3 / 2 + 1. Each rank plans on the batch's own row, both directions on one row:
 * t=r2c n=<shape> q=K ord=nat lay=lm nthreads=T (the column key with real set; rank 3 keys an in-place cell's own row):
 * the column builder's verdicts (the chain, its forms, a Rader stage, Bluestein; no turned prime pass on a real key,
 * il2d_tier.h _il2d_tpc_admits), the serial lane tile sw= and at T > 1 the threaded verdict cmt= mtarm= msw= (_lm_race),
 * the block child under rp_.
 *
 * THE RECIPE (est_plan_lmrn). Rank 1: the C2C lane-major recipe of rank 1 at M over K lanes in the scrambled class
 * (est__plan_lm: est_col_plan at w = K, the unbanded walk's chain past the L1d), with the real batch's own threaded
 * verdict and serial tile (the two rules below: the pack and the fold change both). Rank 2 and 3 (stage 4, step 1): the
 * C2C lane-major recipe's first axis (est_col_plan at w = the spectrum's lanes, natural, the walk, at T > 1 est_mt_2d's
 * test with the strips rule), and the real batch one rank down as the block child (one thread, natural). A Rader column
 * keeps its stage at every lane count: the C2C rule's form on few lanes, Bluestein with the turned prime pass, is not
 * the real door's.
 * THE BUILD (est_build_lmr) takes est_build_lm's steps on the real row: the pre-flight, the rows into a store held in
 * memory, the door's replay (est__replay), and the plan the door built checked against the recipe (est__check_lmr),
 * the lean door's entry in the transform's direction with it. An estimate create never hands back a plan that
 * measured.
 *
 * POSITION IN vfft.c: included by c2c_estimate_il.h after real_estimate_nd_il.h; it reuses the lane-major file's
 * pieces (est__plan_lm, est__preflight_lm, est__put_lm_row, est__check_lm_pass, est__lm_threaded). */
#ifndef VFFT_IL_REAL_ESTIMATE_LM_IL_H
#define VFFT_IL_REAL_ESTIMATE_LM_IL_H

/* The lane-major real batch's recipe: the shape, the first axis's pass, the block child; both directions' (they share
 * the row). */
typedef struct est_planlmr_s
{
    int rank;                     /* 1, 2 or 3 */
    int N1, N2, N3;               /* the shape, the last extent the real one (rank 1: N1 the real length) */
    size_t K;                     /* the batch: K transforms side by side */
    int T;                        /* the threads: 1, or T > 1 = the row keyed nthreads=T with the threaded verdict */
    int ip;                       /* rank 3: in place (the cell's own row) */
    est_planlm_t pass;            /* rank 1: the scrambled pass at M = N1 / 2 over K lanes; rank 2 and 3: the first axis
                                   * at N1 over the spectrum's lanes, natural (a rank-1 recipe at those lanes) */
    struct est_planlmr_s *child;  /* [rp_] rank 2: the 1D real batch at (N2, K); rank 3: the 2D real batch at (N2, N3, K);
                                   * one thread */
} est_planlmr_t;

/* THE TWO LANE-MAJOR REAL RULES AT RANK 1 (the owner's, 2026-11-03; docs/roadmap/real_estimate_plan.md section 12, each
 * fitted on the study's arms -- the estimate's recipe changed in one place -- against the best arm of each cell, 86
 * cells at 1 and 8 threads, 13 lengths x K 4, 32, 256):
 *   - THE THREADED VERDICT AT T > 1: the door's fused strips give each worker EST_LMR_MT_LANES lanes at least (whole
 *     cache lines; lm_batch_il.h _lmr_fused), so the pass threads when every worker gets them (K >= 8 T), or past the
 *     L2 when one does (K >= 8), or with a Rader stage (whose per-lane work is the largest) when four do (K >= 4 x 8);
 *     and only past the C2C test's plane, L1d / EST_ND_MT_SERIAL_L1_DIV (10 x 114 in place, a 9 KiB plane: threaded
 *     5.2 times the serial pass); else serial, the threaded form's choice then est__lm_threaded's. The C2C test alone
 *     threaded K = 4 and 32 batches the door ran 1.4 - 4 times slower threaded: 1.203 -> 0.995 against the planner
 *     (1.223 -> 1.012 against the best arm). With the calibration set and the 12 cells drawn after the rules, 194 cells
 *     at 8 threads: 1.036 -> 1.026 against the best arm (a Rader stage from K = 16: 2062 x 16 serial 1.3 - 1.5 times
 *     faster; past the L2 from K = 16: 62250 x 15 serial 1.27 times slower).
 *   - THE SERIAL TILE: from K = EST_LMR_TILE_K, on a chain with no Rader stage (a tile there ran 1.7 times slower), the
 *     widest of the door's widths (_il2d_sw_ladder) up to EST_LMR_TILE_MAX lanes, below K, whose strip of 16 M w bytes
 *     fits the L2: the fused strip packs, chains and folds in the L2 and the plane is read and written once. One
 *     thread: 1.030 -> 1.005 (K = 256: 1.069 -> about 0.97); with the calibration set (96 cells over the four
 *     layers, K 4 - 1024), 182 cells, the walk 1.026, from K = 64 1.010, from K = 128 1.006 against the planner. Both
 *     placements share the row: in place a tile costs the door another pass, and at K = 32 and 64 (in place by 1.2)
 *     the walk won. */
#define EST_LMR_MT_LANES 8
#define EST_LMR_TILE_K 128
#define EST_LMR_TILE_MAX 64
/* THE RADER WINDOW OF A THREADED PASS (2026-11-03, the 2D real plane's rule 25 one geometry over): a Rader column whose
 * pass threads takes a window of EST_LMR_RD_W columns, not est__rader's 32. Timed on the 28 prime-half cells (262, 2062,
 * 8198 x K 4 - 1024, both directions) with the window the only change: at 8 threads 1.020 -> 0.767 against the planner
 * (K = 32: 0.32 - 0.44, three times the planner and the window of 32; 8198 x 64: 3.3 -> 1.04); on a serial pass, and at
 * one thread (22 cells: 1.008 -> 1.064), the window of 32 stays. */
#define EST_LMR_RD_W 8

/* THE STRIPS OF A THREADED RADER CHAIN (2026-11-03, after the unseen cells; docs/roadmap/real_estimate_plan.md section
 * 17): a pass that threads over a chain with a Rader stage among two stages or more runs in strips at est__msw_2d's
 * width, not the threaded walk. Found here (the first axis of a 2D or 3D batch: 25 cells, 53.2 - 131.2 over 4 - 16
 * batches, 4.535 -> 1.026 against the planner, the walk 3.3 - 5.7 times the planner, slower than the serial pass;
 * 122x288 x 55 in place an unseen cell at 5.16; the 1D pass, 26 cells, 0.948 -> 0.920), then timed on the C2C batch
 * too (section 18) and built into the C2C threaded verdict both share, est__lm_threaded. */

/* the recipe's block children and passes, freed */
static inline void est_free_lmr(est_planlmr_t *p)
{
    if (!p)
        return;
    est_free_lm(&p->pass);
    if (p->child)
    {
        est_free_lmr(p->child);
        free(p->child);
        p->child = NULL;
    }
}
static inline const char *est__lmr_shape(const est_planlmr_t *p, char *b, size_t n)
{
    if (p->rank == 1)
        snprintf(b, n, "%d x %zu", p->N1, p->K);
    else if (p->rank == 2)
        snprintf(b, n, "%dx%d x %zu", p->N1, p->N2, p->K);
    else
        snprintf(b, n, "%dx%dx%d x %zu", p->N1, p->N2, p->N3, p->K);
    return b;
}
/* the first axis's lanes: rank 1 K (the pass at M), rank 2 H K, rank 3 N2 H K (H the real axis's bins) */
static inline long long est__lmr_lanes(const est_planlmr_t *p)
{
    return p->rank == 1 ? (long long)p->K
           : p->rank == 2 ? (long long)(p->N2 / 2 + 1) * (long long)p->K
                          : (long long)p->N2 * (long long)(p->N3 / 2 + 1) * (long long)p->K;
}

/* est__plan_lmrn: the lane-major real batch's recipe at rank 1, 2 or 3 (the header's list; at rank 1 the two rules
 * above); top = 0 for a block child. Returns 0 with *out filled (its child allocated: est_free_lmr), or -1 with the
 * reason. Pure. */
static inline int est__plan_lmrn(const est_machine_t *m, int rank, int N1, int N2, int N3, size_t K, int T, int ip,
                                 int top, est_planlmr_t *out, char *why, size_t whylen)
{
    est_planlm_t *p = &out->pass;
    const int L = rank == 1 ? N1 : rank == 2 ? N2 : N3;
    long long M, lanes;
    memset(out, 0, sizeof *out);
    if (rank < 1 || rank > 3 || L < 4 || (L & 1) || K < 2 || (rank >= 2 && N1 < 2) || (rank == 3 && N2 < 2))
        return est_refuse(why, whylen, "a lane-major real batch at rank %d of K = %zu: the door serves an even last extent of "
                          "at least 4, every other extent at least 2 and K of at least 2 (an odd last extent has no real "
                          "column engine)", rank, K);
    out->rank = rank;
    out->N1 = N1;
    out->N2 = rank >= 2 ? N2 : 0;
    out->N3 = rank == 3 ? N3 : 0;
    out->K = K;
    out->T = T > 1 ? T : 1;
    out->ip = rank == 3 && ip;
    lanes = est__lmr_lanes(out);
    if (rank >= 2)
    {   /* the first axis over the spectrum's lanes, natural: the C2C rank-2 and rank-3 batch's (no unbanded pick), its
         * column at one thread and the C2C threaded test after the turned pass is cleared */
        if (lanes > INT_MAX)
            return est_refuse(why, whylen, "a lane-major real batch of %lld lanes: past the doors' lane count", lanes);
        if (est__plan_lm(m, 1, N1, 0, 0, (size_t)lanes, 1, 1, 0, p, why, whylen))
            return -1;
        p->col.tpc = -1;   /* a column Bluestein on a real key carries no turned prime pass (_il2d_tpc_admits) */
        p->T = out->T;
        if (out->T > 1 && 16LL * N1 * lanes > m->l1d / EST_ND_MT_SERIAL_L1_DIV && N1 != 2)
            est__lm_threaded(m, p, lanes);
        /* the column builder on a real key races the real pool's forms (R2C's adds the half-store twins at the leaf,
         * il2d_tier.h): the token spelled from it, the first form of every stage, which C2R's pool holds too */
        est__col_forms_real(p->col.R, p->col.nst, 0, p->col.forms, sizeof p->col.forms);
        /* THE FIRST AXIS'S SERIAL TILE (2026-11-03, stage 4; docs/roadmap/real_estimate_plan.md section 15): on a plain
         * chain of two stages or more whose plane, 16 N1 lanes bytes, passes the L1d, the widest of the door's widths
         * up to EST_LMR_TILE_MAX lanes, below the lanes, whose strip of 16 N1 w bytes fits the L2: the first axis runs
         * over the spectrum's whole width, every stage a pass over the plane, and a tile of lanes keeps a strip in the
         * cache from stage to stage. 66 cells at one thread (29 shapes over the four layers, K 4 - 128, both
         * directions), each tile timed with the tile the only change: the walk 1.100 against the planner, the rule
         * 0.986 (1.003 against the best tile). One stage is one pass with or without a tile: every tile there ran
         * slower (14 cells of 14, the best width 1.26 times the walk; 3x84 x 48 in place 1.32 among the unseen cells);
         * in the L1d the walk won too (8x8 x 16, 16x16 x 8: a tile 1.5 times slower). The batch's own first axis
         * only: on the 3D batch's block child (the 2D pass over one plane's lanes) a tile of 8 or 16 lanes ran slower
         * at one thread (8: 6 cells faster, 12 slower, 1.145 against the walk; 16: 1.054) */
        if (top && p->col.nst > 1 && !p->col.rd_n && p->col.blu <= 0 && 16LL * N1 * lanes > (long long)m->l1d)
        {
            int lad[6], n = _il2d_sw_ladder(N1, (int)lanes, 1, lad, 6), i;
            for (i = 0; i < n; i++)
                if (lad[i] > p->sw && lad[i] <= EST_LMR_TILE_MAX && (long long)lad[i] < lanes &&
                    16LL * N1 * lad[i] <= (long long)m->l2 &&
                    vfft_policy_lm_tile_ok(lad[i], (int)lanes, p->col.nat && p->col.nst > 1))
                    p->sw = lad[i];
        }
        if (!(out->child = (est_planlmr_t *)calloc(1, sizeof *out->child)))
            return est_refuse(why, whylen, "out of memory");
        if (est__plan_lmrn(m, rank - 1, N2, rank == 3 ? N3 : 0, 0, K, 1, 0, 0, out->child, why, whylen))
        {
            est_free_lmr(out->child);
            free(out->child);
            out->child = NULL;
            return -1;
        }
        return 0;
    }
    M = N1 / 2;
    if (est__plan_lm(m, 1, (int)M, 0, 0, K, 1, 0, EST_LM_UNB, p, why, whylen))
        return -1;
    p->col.tpc = -1;   /* a column Bluestein on a real key carries no turned prime pass (_il2d_tpc_admits) */
    p->T = out->T;
    if (out->T > 1)
    {
        const long long k = (long long)K, w = EST_LMR_MT_LANES;
        p->cmt = p->mtarm = p->msw = 0;
        if (M != 2 && 16 * M * k > (long long)m->l1d / EST_ND_MT_SERIAL_L1_DIV &&
            (k >= w * out->T || (k >= w && 16 * M * k > (long long)m->l2) || (k >= 4 * w && p->col.rd_n > 0)))
            est__lm_threaded(m, p, k);
        if (p->cmt && p->col.rd_n > 0)
        {   /* the threaded Rader window, clipped as est__rader clips one to the stage's columns */
            const size_t ncol = (size_t)(M / p->col.R[0]) * (size_t)K;
            size_t W = EST_LMR_RD_W;
            if (W > ncol)
                W = ncol & ~(size_t)1;
            if (W < VFFT_ILRD_W_MIN)
                W = VFFT_ILRD_W_MIN;
            p->col.rdw = (int)W;
        }
    }
    if (K >= EST_LMR_TILE_K && !p->col.rd_n && p->col.blu <= 0)
    {
        int lad[6], n = _il2d_sw_ladder((int)M, (int)K, 1, lad, 6), i;
        for (i = 0; i < n; i++)
            if (lad[i] > p->sw && lad[i] <= EST_LMR_TILE_MAX && (size_t)lad[i] < K && 16 * M * lad[i] <= (long long)m->l2 &&
                vfft_policy_lm_tile_ok(lad[i], (int)K, 0))
                p->sw = lad[i];
    }
    return 0;
}
/* est_plan_lmrn: the batch's recipe (est__plan_lmrn at the top) */
static inline int est_plan_lmrn(const est_machine_t *m, int rank, int N1, int N2, int N3, size_t K, int T, int ip,
                                est_planlmr_t *out, char *why, size_t whylen)
{
    return est__plan_lmrn(m, rank, N1, N2, N3, K, T, ip, 1, out, why, whylen);
}
/* est_plan_lmr: rank 1, the 1D real batch */
static inline int est_plan_lmr(const est_machine_t *m, int N, size_t K, int T, est_planlmr_t *out, char *why,
                               size_t whylen)
{
    return est_plan_lmrn(m, 1, N, 0, 0, K, T, 0, out, why, whylen);
}

/* the batch's row key, as its door keys it (lm_batch_il.h _vfft_create_lmr_il, lm_batch2dr_il.h, lm_batch3dr_il.h) */
static inline void est__lmr_key(vw2_ilcol_key_t *ck, const est_planlmr_t *p)
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
        ck->ip = p->ip;
    }
    ck->ord = VW2_ORD_NAT;
    ck->real = 1;
    ck->nthreads = p->T;
    ck->lm = 1;
}

/* the recipe's pre-flight: the first axis's pass as its door replays it (est__preflight_lm), at its lanes and class,
 * and the block child one rank down at one thread */
static inline int est__preflight_lmr(const est_planlmr_t *p, char *why, size_t whylen)
{
    char sh[64], w2[400];
    const long long lanes = est__lmr_lanes(p);
    const int N1p = p->rank == 1 ? p->N1 / 2 : p->N1;
    est__lmr_shape(p, sh, sizeof sh);
    if (p->pass.rank != 1 || p->pass.N1 != N1p || (long long)p->pass.K != lanes || p->pass.T != p->T ||
        p->pass.nat != (p->rank >= 2))
        return est_refuse(why, whylen, "the lane-major real batch %s: the first axis is not its pass at %d over %lld lanes",
                          sh, N1p, lanes);
    if (p->rank >= 2)
    {   /* the first axis: its forms the real pool's, the rest as the C2C pass's (checked on a copy spelled the C2C way) */
        est_planlm_t q = p->pass;
        char fr[64];
        est__col_forms_real(q.col.R, q.col.nst, 0, fr, sizeof fr);
        if (strcmp(fr, q.col.forms))
            return est_refuse(why, whylen, "the lane-major real batch %s: the first axis's forms \"%s\" are not the real "
                              "pool's \"%s\"", sh, q.col.forms, fr);
        est__col_forms(q.col.R, q.col.nst, q.col.forms, sizeof q.col.forms);
        if (est__preflight_lm(&q, w2, sizeof w2))
            return est_refuse(why, whylen, "the lane-major real batch %s: %s", sh, w2);
    }
    else if (est__preflight_lm(&p->pass, w2, sizeof w2))
        return est_refuse(why, whylen, "the lane-major real batch %s: %s", sh, w2);
    if (p->rank >= 2)
    {
        const est_planlmr_t *c = p->child;
        if (!c || c->rank != p->rank - 1 || c->N1 != p->N2 || c->N2 != (p->rank == 3 ? p->N3 : 0) || c->K != p->K ||
            c->T != 1)
            return est_refuse(why, whylen, "the lane-major real batch %s: the block child is not the batch one rank down "
                              "on one thread", sh);
        if (est__preflight_lmr(c, w2, sizeof w2))
            return est_refuse(why, whylen, "the lane-major real batch %s, its block child: %s", sh, w2);
    }
    else if (p->child)
        return est_refuse(why, whylen, "the lane-major real batch %s: a block child at rank 1", sh);
    return 0;
}

/* est_put_rows_lmr: the batch's row into st -- the first axis's verdicts (est__put_lm_row), and the block child's rows
 * under rp_ (written into a store of their own and moved with the door's vfft_child_row_update). Returns 0, or -1 with
 * the reason. */
static inline int est_put_rows_lmr(vw2_store_t *st, const est_planlmr_t *p, char *why, size_t whylen)
{
    vw2_ilcol_key_t ck;
    vw2_key_t key;
    char sh[64];
    est__lmr_key(&ck, p);
    est__lmr_shape(p, sh, sizeof sh);
    if (est__put_lm_row(st, &ck, &p->pass, sh, why, whylen))
        return -1;
    if (p->child)
    {
        struct vfft_wisdom_s *C = vfft_child_store_new();
        int ok;
        if (!C)
            return est_refuse(why, whylen, "out of memory");
        if (est_put_rows_lmr(&C->vw2, p->child, why, whylen))
        {
            vfft_child_store_free(C);
            return -1;
        }
        vw2__ilcol_key(&ck, &key);
        ok = vfft_child_row_update(st, &key, "rp_", C);
        vfft_child_store_free(C);
        if (!ok)
            return est_refuse(why, whylen, "the lane-major real batch %s: its block child could not be put on its row", sh);
    }
    return 0;
}

/* the plan the door built against the recipe: the real batch at its shape in the request's direction, its first axis's
 * pass (est__check_lm_pass), the lean door's entry in that direction, the block child */
static inline int est__check_lmr(const struct vfft_plan_s *h, const est_planlmr_t *p, int c2r, char *why, size_t whylen)
{
    char sh[64], b[64];
    snprintf(sh, sizeof sh, "%s %s", c2r ? "C2R" : "R2C", est__lmr_shape(p, b, sizeof b));
    if (!h || !h->lm || h->transform != (c2r ? VFFT_C2R : VFFT_R2C) || h->N != p->N1 || h->N2 != p->N2 ||
        (p->rank == 3 && h->N3 != p->N3) || h->K != p->K || (p->rank == 1 && h->lm_M != p->N1 / 2))
        return est_refuse(why, whylen, "the door built no lane-major real batch %s", sh);
    if (est__check_lm_pass(h, &p->pass, sh, why, whylen))
        return -1;
    if (!h->k1_go[c2r ? VFFT_BACKWARD : VFFT_FORWARD])
        return est_refuse(why, whylen, "the lane-major real batch %s has no lean entry (k1_go)", sh);
    if (p->child)
    {
        char w2[400];
        if (est__check_lmr(h->lm_row, p->child, c2r, w2, sizeof w2))
            return est_refuse(why, whylen, "the lane-major real batch %s, its block child: %s", sh, w2);
    }
    return 0;
}

/* est_build_lmr: the recipe p of the lane-major real request cfg -> a plan built with nothing measured, or NULL with
 * the reason in why and the outcome in rep (may be NULL). cfg is the request as the front door translated it: an
 * interleaved R2C or C2R batch at the recipe's rank and shape in the lane-major geometry (at rank 1 DEFAULT or named,
 * at rank 2 and 3 named), K = p->K, the recipe's thread count (and at rank 3 placement); its store and save flag are not
 * used. Holds the create lock from the pre-flight to the end. The caller owns the plan (vfft_destroy). */
static inline struct vfft_plan_s *est_build_lmr(const est_planlmr_t *p, const vfft_cfg_t *cfg, est_build_report_t *rep,
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
    if (!p || !cfg || (cfg->transform != VFFT_R2C && cfg->transform != VFFT_C2R) || cfg->layout != VFI_LAYOUT_IL ||
        (p->rank == 1 ? cfg->batch_geom == VFFT_BATCH_CONTIGUOUS : cfg->batch_geom != VFFT_BATCH_LANE_MAJOR) ||
        !vfft_policy_batch_lane_major(cfg) || cfg->howmany != p->K || (cfg->dims < 2 ? 1 : cfg->dims) != p->rank ||
        cfg->n[0] != p->N1 || (p->rank >= 2 && cfg->n[1] != p->N2) || (p->rank == 3 && cfg->n[2] != p->N3) ||
        (cfg->nthreads > 1 ? cfg->nthreads : 1) != p->T || (p->rank == 3 && (cfg->placement == VFFT_INPLACE) != p->ip))
    {
        est_refuse(why, whylen, "estimate builds a lane-major interleaved real batch at the recipe's rank, shape, K and "
                   "thread count");
        return NULL;
    }
    vfft_env_init();          /* env.h:96: the pre-flight's tables in the create's FP mode */
    _vfft_create_lock();      /* recursive: the pre-flight touches the create's process statics */
    _vfft_create_depth++;     /* the whole build inside a create: a clock read enters a scope */
    est__meter_read(&m0);
    if (est__preflight_lmr(p, why, whylen))
        goto out;
    if (!(S = vfft_child_store_new()))
    {
        est_refuse(why, whylen, "out of memory");
        goto out;
    }
    if (est_put_rows_lmr(&S->vw2, p, why, whylen))
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

/* THE FRONT DOOR'S ESTIMATE CREATE of a lane-major real batch (_vfft_create_estimate): the recipe and the build. NULL
 * with the reason on the error record (vfft_error). */
static vfft_plan _vfft_create_estimate_lmr(const est_machine_t *m, const vfft_cfg_t *cfg)
{
    char why[640];
    struct vfft_plan_s *h = NULL;
    est_planlmr_t p;
    const int rank = cfg->dims < 2 ? 1 : cfg->dims;
    if (!est_plan_lmrn(m, rank, cfg->n[0], rank >= 2 ? cfg->n[1] : 0, rank == 3 ? cfg->n[2] : 0, cfg->howmany,
                       cfg->nthreads, cfg->placement == VFFT_INPLACE, &p, why, sizeof why))
        h = est_build_lmr(&p, cfg, NULL, why, sizeof why);
    est_free_lmr(&p);
    if (!h)
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE: %s", why);
    return (vfft_plan)h;
}

#endif /* VFFT_IL_REAL_ESTIMATE_LM_IL_H */
