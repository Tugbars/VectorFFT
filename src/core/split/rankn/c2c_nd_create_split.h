/* c2c_nd_create_split.h — the split library's 2D and 3D C2C creates on
 * zturn-ts (docs/design/split_v1_0_contract.md, built 2026-11-05).
 *
 * A rank-2 or rank-3 split transform is its AXES, each a 1D cell of the
 * split library served by the same create as a 1D request
 * (split/rank1/c2c_create_split.h), raced and banked on that cell's own row:
 *
 *   rows   the last axis, contiguous in memory: the K = 1 cell at N_last run
 *          as a CONTIGUOUS batch of every row of the plane or volume (the
 *          _batch twin; in slabs over the pool at T > 1 under the batch's
 *          threading verdict), in the caller's placement — out of place it
 *          writes the destination, and every later pass runs there;
 *   mid    (rank 3) the middle axis: the lane-major cell at N2 with N3 lanes
 *          (one row of the plane = one element of every lane), run once per
 *          plane, in place;
 *   cols   the first axis: the lane-major cell at N1 with N2 (rank 2) or
 *          N2*N3 (rank 3) lanes, in place. Its BLOCK (blk= on the row) is the
 *          row-blocked ingest/terminator the column pass needs where the
 *          plane set exceeds L2 (the TLB law of the lane-major pass), raced
 *          per cell like its chain and tile.
 *
 * Natural order on every axis by construction (zturn-ts has no scrambled
 * form); both placements; one transform per plan (a batch of planes is not
 * served: run the plan per plane). Every extent is a registry cell (a power
 * of two from VFFT_ZTS_MIN_N to VFFT_ZTS_MAX_N); anything else is refused
 * naming the interleaved layout. Execution: split/split_execute.h.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: through split/split_create.h, after the
 * 1D create.
 */
#ifndef VFFT_SPLIT_C2C_ND_CREATE_SPLIT_H
#define VFFT_SPLIT_C2C_ND_CREATE_SPLIT_H

#include "split/rank1/c2c_create_split.h"

/* one axis as a 1D cell handle: lane_major = 0 runs K contiguous transforms
 * of N (the rows), 1 the lane-major pass of K lanes; the placement is the
 * pass's own (the first pass the caller's, every later one in place) */
static struct vfft_plan_s *_zts_axis(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W,
                                     int N, size_t K, int placement, int lane_major)
{
    vfft_cfg_t c1 = *cfg;
    c1.dims = 1;
    c1.n[0] = N;
    c1.n[1] = c1.n[2] = c1.n[3] = 0;
    c1.howmany = K;
    c1.placement = placement;
    c1.batch_geom = lane_major ? VFFT_BATCH_LANE_MAJOR : VFFT_BATCH_CONTIGUOUS;
    c1.order = VFI_ORDER_NATURAL;
    return (struct vfft_plan_s *)_vfft_create_c2c_split(&c1, W, N, K);
}

/* ── the 3D plan's one threading verdict of its own: the middle axis's
 * PLANES OVER SLOTS (one dispatch, each slot a range of planes run serially)
 * vs the per-plane pass (the axis cell's own verdict inside every plane),
 * raced at create on the plan's own volume at the plan's thread count and
 * banked eng=tcb tcmt= on the 3D row (common/wisdom/wisdom2_tcmt.h, the
 * rank-N key). No worker planes on the middle axis (one plane: nothing to
 * slice; VFFT_NO_TCMT) => per plane. VFFT_TCMT pins. */
typedef struct
{
    struct vfft_plan_s *h;
    double *re, *im, *re0, *im0;
    size_t nb;
    int pmt;
} _zts_pmt_arm_t;
static void _zts_pmt_arm_run(void *v)
{
    _zts_pmt_arm_t *a = (_zts_pmt_arm_t *)v;
    a->h->zts_pmt = a->pmt;
    _vfft_split_execute(a->h, VFFT_FORWARD, a->re, a->im, a->re, a->im);
}
static void _zts_pmt_reseed(void *v)
{
    _zts_pmt_arm_t *a = (_zts_pmt_arm_t *)v;
    memcpy(a->re, a->re0, a->nb);
    memcpy(a->im, a->im0, a->nb);
}
static void _zts_pmt_decide(struct vfft_plan_s *h, const vfft_cfg_t *cfg, struct vfft_wisdom_s *W)
{
    const int T = h->nthreads;
    const int ip = (h->placement == VFFT_INPLACE);
    const int lg = getenv("VFFT_TCMT_VERBOSE") || getenv("VFFT_TCMT_LOG");
    const char *pin = getenv("VFFT_TCMT");
    const int n[3] = { h->N, h->N2, h->N3 };
    h->zts_pmt = 0;
    if (T <= 1 || !h->zts_mid || !h->zts_mid->zts || getenv("VFFT_NO_TCMT"))
        return;   /* a Bluestein middle axis (no zts handle) runs per plane */
    /* the middle axis's worker plane sets: its own threading verdict built
     * them when it raced; build them here when it did not (one plane of N3
     * lanes may hold a single block) */
    if (h->zts_mid->zts->mt_n == 0)
    {
        int nw = T - 1;
        if (nw > h->N - 1) nw = h->N - 1;
        if (nw > VFFT_ZTS_MT_MAX) nw = VFFT_ZTS_MT_MAX;
        if (nw <= 0 || vfft_zts_mt_planes(h->zts_mid->zts, nw) == 0)
            return;
    }
    if (pin)
    {
        h->zts_pmt = atoi(pin) ? 1 : 0;
        if (lg) fprintf(stderr, "[tcmt] zts 3D %dx%dx%d T=%d: pinned pmt=%d (env; not banked)\n", n[0], n[1], n[2], T, h->zts_pmt);
        return;
    }
    if (W && !cfg->recalibrate)
    {
        int v = 0;
        if (vw2_tcmt_lookup_nd(&W->vw2, VW2_T_C2C, 3, n, 1, VW2_ORD_NAT, ip ? VW2_PL_IP : VW2_PL_OOP, VW2_LAY_SPLIT, T, &v))
        {
            h->zts_pmt = v;
            if (lg) fprintf(stderr, "[tcmt] zts 3D %dx%dx%d T=%d: replay pmt=%d src=wisdom\n", n[0], n[1], n[2], T, v);
            return;
        }
    }
    {   /* the race on the plan's own volume, in place (the passes after the
         * rows are in place in both placements; the rows' placement is the
         * same in both arms) */
        const size_t tot = (size_t)h->N * (size_t)h->N2 * (size_t)h->N3, nb = tot * sizeof(double);
        double *re = (double *)vfft_aligned_alloc(nb), *im = (double *)vfft_aligned_alloc(nb);
        double *re0 = (double *)vfft_aligned_alloc(nb), *im0 = (double *)vfft_aligned_alloc(nb);
        double st = 0, mt = 0;
        size_t i;
        const int saved_pl = h->placement;
        if (!re || !im || !re0 || !im0)
        {
            vfft_aligned_free(re); vfft_aligned_free(im); vfft_aligned_free(re0); vfft_aligned_free(im0);
            return;
        }
        for (i = 0; i < tot; i++)
        {
            re0[i] = 1.0 + 1e-6 * (double)(i & 511);
            im0[i] = 0.5 - 1e-6 * (double)(i & 255);
        }
        h->placement = VFFT_INPLACE; /* the arms run in place on the race's planes */
        {
            _zts_pmt_arm_t a = { h, re, im, re0, im0, nb, 0 };
            _zts_pmt_arm_t b = { h, re, im, re0, im0, nb, 1 };
            const vfft_race_arm_t arms[2] = { { "per-plane", _zts_pmt_arm_run, &a },
                                              { "planes", _zts_pmt_arm_run, &b } };
            vfft_race_proto_t proto;
            double ns[2];
            memset(&proto, 0, sizeof proto);
            proto.rounds = 7;
            proto.reps = 1;
            proto.agg = VFFT_RACE_MIN;
            proto.alternate = 1;
            proto.warm = 1;
            proto.reset = _zts_pmt_reseed;
            proto.reset_ctx = &a;
            vfft_race_run(&proto, arms, 2, ns); /* threaded arms: no pace */
            st = ns[0];
            mt = ns[1];
        }
        h->placement = saved_pl;
        h->zts_pmt = (mt < st);
        if (lg)
            fprintf(stderr, "[tcmt] zts 3D %dx%dx%d T=%d %s: race per-plane=%.0f planes=%.0f -> %s\n",
                    n[0], n[1], n[2], T, ip ? "ip" : "oop", st, mt, h->zts_pmt ? "PLANES" : "per-plane");
        vfft_aligned_free(re); vfft_aligned_free(im); vfft_aligned_free(re0); vfft_aligned_free(im0);
        if (W && vw2_tcmt_bank_nd(&W->vw2, VW2_T_C2C, 3, n, 1, VW2_ORD_NAT, ip ? VW2_PL_IP : VW2_PL_OOP, VW2_LAY_SPLIT,
                                  h->zts_pmt, T, h->zts_pmt ? mt : st) == VW2_OK)
            _vw2_persist(W, cfg);
    }
}

static vfft_plan _vfft_create_c2c_nd_split(const vfft_cfg_t *cfg,
                                           struct vfft_wisdom_s *W,
                                           size_t K)
{
    const int rank = cfg->dims;
    const int N1 = cfg->n[0], N2 = cfg->n[1], N3 = rank == 3 ? cfg->n[2] : 0;
    const int ip = (cfg->placement == VFFT_INPLACE);
    const char *pname = ip ? "in-place" : "out-of-place";
    struct vfft_plan_s *h, *hrow = NULL, *hmid = NULL, *hcol = NULL;
    char shape[48];
    if (rank == 3)
        snprintf(shape, sizeof shape, "%dx%dx%d", N1, N2, N3);
    else
        snprintf(shape, sizeof shape, "%dx%d", N1, N2);
    if (K > 1)
    {
        _vfft_warn("vfft_create: %s %dD C2C %s howmany=%zu on the SPLIT layout: one transform per "
                   "plan on the split side (a batch of planes is the interleaved plane queue) — "
                   "run the plan per plane, or use VFFT_LAYOUT_INTERLEAVED",
                   pname, rank, shape, K);
        return NULL;
    }
    if (cfg->order == VFI_ORDER_SCRAMBLED)
    {
        _vfft_warn("vfft_create: %s %dD C2C %s on the SPLIT layout at VFFT_ORDER_SCRAMBLED: the split "
                   "engine is natural on every axis by construction — ask VFFT_ORDER_NATURAL (or the default)",
                   pname, rank, shape);
        return NULL;
    }
    /* the last extent runs as rows (the S family: a power of two, or 2^a * odd
     * with a >= 4); every other extent runs lane-major (the V family: any
     * product of 2 to 7 radices over 3, 4, 5, 7, 8, 9, 15) */
    {   /* every axis a cell of its form: the bands, else Bluestein (any length
         * up to 131072); the lane-major passes take any lane count (the
         * product of the extents after the axis) up to the lane bound; a
         * count off the vector width runs its tail (2026-11-07) */
        const int Nl = rank == 3 ? N3 : N2;
        const size_t lanes1 = rank == 3 ? (size_t)N2 * (size_t)N3 : (size_t)N2;
        const int ok_rows = vfft_zts_serves(Nl) || vfft_zts_blu_m(Nl);
        const int ok_col = (vfft_zts_serves_v(N1) || vfft_zts_blu_m(N1)) && lanes1 <= (size_t)VFFT_ZTS_LANES_MAX;
        const int ok_mid = rank < 3 || ((vfft_zts_serves_v(N2) || vfft_zts_blu_m(N2)) && (size_t)N3 <= (size_t)VFFT_ZTS_LANES_MAX);
        if (!ok_rows || !ok_col || !ok_mid)
            _vfft_warn("vfft_create: %s %dD C2C %s on the SPLIT layout: every extent must be at most 131072 "
                       "and the lane count of a column pass at most %d — use VFFT_LAYOUT_INTERLEAVED for this shape",
                       pname, rank, shape, VFFT_ZTS_LANES_MAX);
            return NULL;
        }
    }
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
        return NULL;
    /* the axes, last to first: the rows in the caller's placement, the rest in place */
    if (rank == 3)
    {
        hrow = _zts_axis(cfg, W, N3, (size_t)N1 * (size_t)N2, cfg->placement, 0);
        hmid = hrow ? _zts_axis(cfg, W, N2, (size_t)N3, VFFT_INPLACE, 1) : NULL;
        hcol = hmid ? _zts_axis(cfg, W, N1, (size_t)N2 * (size_t)N3, VFFT_INPLACE, 1) : NULL;
    }
    else
    {
        hrow = _zts_axis(cfg, W, N2, (size_t)N1, cfg->placement, 0);
        hcol = hrow ? _zts_axis(cfg, W, N1, (size_t)N2, VFFT_INPLACE, 1) : NULL;
    }
    if (!hrow || !hcol || (rank == 3 && !hmid))
    {
        if (hrow) vfft_destroy((vfft_plan)hrow);
        if (hmid) vfft_destroy((vfft_plan)hmid);
        if (hcol) vfft_destroy((vfft_plan)hcol);
        free(h);
        return NULL; /* the axis create warned */
    }
    h->transform = VFFT_C2C;
    h->placement = cfg->placement;
    h->layout = (int)cfg->layout;
    h->N = N1;
    h->N2 = N2;
    h->N3 = N3;
    h->K = 1;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zts_row = hrow;
    h->zts_mid = hmid;
    h->zts_col = hcol;
    if (rank == 3)
        _zts_pmt_decide(h, cfg, W); /* the middle axis's planes over slots at T > 1 */
    return (vfft_plan)h;
}

#endif /* VFFT_SPLIT_C2C_ND_CREATE_SPLIT_H */
