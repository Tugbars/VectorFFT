/**
 * fftnd_il.h — the rank-N INTERLEAVED c2c tier (2026-09-06; design of
 * record: docs/roadmap/fftnd_il_design.md). Rank 3 today; rank 4 by the
 * same composition.
 *
 * NOT the split fftnd.h: different layout, different axis model, shares
 * nothing with it but the directory. Interleaved has no lane axis, so the
 * tier is built from the 2D IL tier's two pieces — the COLUMN-AXIS PASS
 * (il2d_col.h descriptor, il2d_tier.h build + execute: stage kernels
 * taking a leg stride and a contiguous count) and the K=1 IL ROW pass.
 *
 * Row-major with N3 contiguous, a rank-3 cube is:
 *   axis 0 : a column pass over the virtual plane of N1 rows x (N2*N3)
 *            complex — the 2D column pass with pitch N2*N3, unchanged
 *            kernels, chains and tables; wide over the cube.
 *   axis 1 : a column pass over each of the N1 planes of N2 rows x N3.
 *   axis 2 : the row pass over N1*N2 rows.
 *
 * THE STRUCTURE IS A RACED ARM (owner 2026-09-06, "only racing both to each
 * other can tell"), never an architectural default:
 *   arm 1, the RECURSION : axis 0 wide, then a plain 2D IL c2c CHILD plan
 *            (its own (N2,N3) wisdom cell, in place) executed per plane —
 *            axis 1 and the rows with every 2D verdict (chain, forms,
 *            band, row route) raced as a standalone 2D transform;
 *   arm 2, the FLAT tier : axis 0 wide, then per plane this tier's OWN
 *            axis-1 column pass (its chain raced in the 3D context, the
 *            same build function) and the row pass over the plane's rows.
 *
 * THE AXIS-0 BANDED WALK (the 2D tier's wl, E1.2, at rank 3): a "row" of
 * the virtual plane IS a plane of the cube, so a band of wl rows is a
 * band of wl planes. fwd = the wide prefix stages 0..cut-1 over the cube,
 * then per band the stage SUFFIX depth-first followed at once by the
 * per-plane structure on the band's planes while they are L2-hot (the 2D
 * tfuse: rows fused into the band). bwd mirrors the Hermitian chain: per
 * band the reversed suffix then the planes, then the reversed wide prefix.
 * Same kernels, same tables, same count as unbanded — only loop order and
 * base pointers differ (bitwise identical output; ilnd_probe checks it).
 *
 * Both structure arms x every legal width are ARMS OF ONE alternated race
 * on a scratch cube at create (min of 3); the winner banks s= and wl= tf=
 * on the cell's rank-3 lay=il row beside the axis-0 chain tokens (chain=
 * blu= forms=) and the flat arm's axis-1 tokens (chain1= ...); the child's
 * verdicts live on the child's cell. VFFT_ILND_ARM=1|2 and VFFT_ILND_WL=w
 * pin for a probe and never bank.
 *
 * MULTITHREADING (the 2D tier's INC-C ported, 2026-09-07): two partition
 * arms, both pure loop restrictions of the serving walk (no arithmetic
 * change => MT == ST bitwise, gated by the probe):
 *   BAND arm  (wl > 0): the wide prefix stages digit-split (INC-3b: whole
 *             planes per digit, one dispatch per stage), then workers take
 *             disjoint BANDS — each band = suffix stages + the fused
 *             per-plane structure, exchange-free (the planes a worker
 *             finishes are the planes it just produced);
 *   PLANE arm : workers take disjoint COLUMN STRIPS of the virtual plane
 *             and run the whole axis-0 chain (barrier-free: a column
 *             pass never mixes columns), then disjoint PLANE ranges for
 *             the structure. The only arm of an unbanded or Bluestein
 *             axis 0.
 * At the plan's T the partition arm and the STRUCTURE are raced together
 * (serial with the one-thread structure, then band and plane with each
 * buildable structure): the structure that wins at one thread is not the
 * one that wins threaded (measured 64^3: child+band 70 us, flat+plane
 * 103 us, while at one thread the two structures tie). Banked on the
 * rank-3 row as cmt= (0 serial | 1 band | 2 plane), cmtt= (the T raced
 * at) and cmts= (the structure the threaded verdict runs with); s= stays
 * the one-thread verdict. A verdict serves only at its own T.
 * Every threaded sample runs REPS executes after warm passes: a worker's
 * cache partition settles over the first milliseconds of executes
 * (measured: round-0 means 1.5-5x the steady state), and single-execute
 * samples alternating between arms time the transient, never the steady
 * state.
 * The per-plane structure mutates plan state (a 2D child's scratch, the
 * row plan's scratch, an axis-1 Bluestein scratch), so worker t > 0 runs
 * its CLONE: a 2D child clone route-equivalent to the primary
 * (_ilnd_child_equiv), or a row-plan clone route-equivalent
 * (_tc_clone_equiv) plus its own axis-1 scratch. Any clone failure tears
 * that structure's set down; with no clones MT declines — never a
 * half-cloned dispatch. The pool is the one owner (support/threads.h);
 * the plan's T is the snapshot, stride_pool_workers_for the one clamp.
 * VFFT_ILND_MT=0|1|2 pins the partition for a probe (never banks).
 * Engagement counter: vfft_ilnd_mt_passes() (vfft.c) — a threaded result
 * without it is vacuous. VFFT_ILND_PROF=1 prints per-phase ns.
 *
 * Every pass commutes with every other (each is a Kronecker factor), so
 * forward and backward run the same pass order. Output order: DEFAULT/
 * SCRAMBLED = each column axis digit-reversed by its chain, rows natural.
 *
 * Contracts (phase 2 + MT + in place): C2C, rank 3, howmany == 1, either
 * placement (one plan, one wisdom row: every pass is alias-tolerant), order
 * DEFAULT or SCRAMBLED. NATURAL, real and rank 4 follow in later phases —
 * refused loudly until then, never bridged.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: after il2d_tier.h (the column build
 * and execute, _tc_clone_equiv's declaration), k1_commit.h (support/race.h)
 * and the wisdom readers, before fftnd_create.h (which dispatches here for
 * rank-3 INTERLEAVED c2c).
 */
#ifndef VFFT_TRANSFORMS_FFTND_FFTND_IL_H
#define VFFT_TRANSFORMS_FFTND_FFTND_IL_H

extern long _vfft_ilnd_mt_count; /* vfft.c: the engagement counter */

/* a 2D IL c2c child clone is equivalent to its primary iff every verdict
 * that decides output bits matches: the column chain and its kernel
 * pointers (forms), the banded walk, the N-arm, the natural class, the row
 * route and the row plan (the TC army's _tc_clone_equiv, valid on any K=1
 * c2c plan). The same law as the 2D tier's row-clone check; the plan
 * fingerprint is harness-only (-DVFFT_FINGERPRINT), so it is not used. */
static int _ilnd_child_equiv(const struct vfft_plan_s *a, const struct vfft_plan_s *b)
{
    const vfft_ilcol_t *x = &a->il2d_col, *y = &b->il2d_col;
    int s;
    if (!a->il2d_row || !b->il2d_row)
        return 0;
    if (a->N != b->N || a->N2 != b->N2 || a->il2d_rowoop != b->il2d_rowoop)
        return 0;
    if (x->nst != y->nst || x->wl != y->wl || x->cut != y->cut || x->tfuse != y->tfuse ||
        x->staged != y->staged || x->nat != y->nat || x->natarm != y->natarm ||
        x->blu != y->blu || x->colmt != y->colmt)
        return 0;
    for (s = 0; s < x->nst; s++)
        if (x->R[s] != y->R[s] || x->L[s] != y->L[s] || x->f[s] != y->f[s] || x->b[s] != y->b[s])
            return 0;
    if (!_tc_clone_equiv(a->il2d_row, b->il2d_row))
        return 0;
    if (a->il2d_rowoop && (!a->il2d_rowo || !b->il2d_rowo ||
                           !_tc_clone_equiv(a->il2d_rowo, b->il2d_rowo)))
        return 0;
    return 1;
}

typedef struct vfft_ilnd_s {
    int rank;
    int N[4];
    size_t plane;                 /* complex per axis-0 row: N[1] * ... * N[rank-1] */
    int arm;                      /* the SERVING structure: 1 = the child per plane, 2 = flat */
    vfft_ilcol_t ax0;             /* axis 0: N[0] rows over `plane` complex (wl/cut = the banded walk) */
    struct vfft_plan_s *child;    /* arm 1: the rank-(n-1) IL c2c plan, in place, per plane */
    vfft_ilcol_t ax1;             /* arm 2: N[1] rows over N[2] complex, per plane */
    struct vfft_plan_s *row;      /* arm 2: the K=1 IL row plan, in place, natural */
    char forms0[64], forms1[64];
    /* MT: the raced verdict and the per-worker clones (worker t > 0 = slot t-1) */
    int mt;                       /* 0 serial | 1 band | 2 plane */
    int mt_t;                     /* the plan's thread snapshot (create's nthreads) */
    int wn1, wn2;                 /* clones built per structure (= T-1) or 0: that structure cannot thread */
    struct vfft_plan_s **childw;  /* arm 1 clones */
    struct vfft_plan_s **roww;    /* arm 2 row clones */
    vfft_ilcol_t *ax1w;           /* arm 2 axis-1 descriptors: shared tables, own scratch */
} vfft_ilnd_t;

/* ── the per-plane structure (tid 0 = the primary, t > 0 = its clone) ── */
static void _ilnd_plane_t(const vfft_ilnd_t *d, int tid, vfft_dir_t dir, double *pl)
{
    const int rev = (dir == VFFT_BACKWARD);
    if (d->arm == 1)
    {
        struct vfft_plan_s *c = tid > 0 ? d->childw[tid - 1] : d->child;
        vfft_execute((vfft_plan)c, dir, pl, NULL, pl, NULL);
    }
    else
    {
        const vfft_ilcol_t *ax1 = tid > 0 ? &d->ax1w[tid - 1] : &d->ax1;
        struct vfft_plan_s *row = tid > 0 ? d->roww[tid - 1] : d->row;
        const size_t rn = (size_t)d->N[2];
        size_t r;
        _il2d_col_exec(ax1, pl, pl, rev);
        for (r = 0; r < (size_t)d->N[1]; r++)
            vfft_execute((vfft_plan)row, dir, pl + 2 * r * rn, NULL,
                         pl + 2 * r * rn, NULL);
    }
}
static void _ilnd_plane(const vfft_ilnd_t *d, vfft_dir_t dir, double *pl)
{
    _ilnd_plane_t(d, 0, dir, pl);
}

/* ── the serial execute: axis 0 (src -> dst), the structure on dst ──── */
static void _ilnd_execute_st(const vfft_ilnd_t *d, vfft_dir_t dir,
                             const double *src, double *dst)
{
    const int rev = (dir == VFFT_BACKWARD);
    const vfft_ilcol_t *c = &d->ax0;
    const size_t N0 = (size_t)d->N[0], rn = d->plane;
    size_t p;
    if (c->wl > 0 && !c->blu && !c->nat)
    {   /* the banded walk: bands of wl planes, the structure fused */
        const int cut = c->cut, nst = c->nst;
        const size_t wl = (size_t)c->wl;
        vfft_il2p_fn const *fns = rev ? c->b : c->f;
        double *const *tabs = rev ? c->tb : c->tf;
        size_t b0;
        static int prof = -1;
        double t0 = 0, tp = 0, tsuf = 0, tpl = 0;
        if (prof < 0)
            prof = getenv("VFFT_ILND_PROF") != NULL;
        if (prof)
            t0 = _il_ab_now();
        if (!rev && cut > 0)
            _il2d_col_stages(src, dst, c->N, rn, 0, cut, c->R, c->L, fns, tabs, 0);
        if (prof)
            tp = _il_ab_now() - t0;
        for (b0 = 0; b0 < N0; b0 += wl)
        {
            const double *bs = (!rev && cut > 0) ? dst + 2 * b0 * rn : src + 2 * b0 * rn;
            double *bd = dst + 2 * b0 * rn;
            double ta = prof ? _il_ab_now() : 0, tb;
            _il2d_col_stages(bs, bd, (int)wl, rn, cut, nst, c->R, c->L, fns, tabs, rev);
            tb = prof ? _il_ab_now() : 0;
            for (p = 0; p < wl; p++)
                _ilnd_plane(d, dir, bd + 2 * p * rn);
            if (prof)
            {
                tsuf += tb - ta;
                tpl += _il_ab_now() - tb;
            }
        }
        if (rev && cut > 0)
            _il2d_col_stages(dst, dst, c->N, rn, 0, cut, c->R, c->L, fns, tabs, 1);
        if (prof)
            fprintf(stderr, "[ilnd-prof] serial-banded s=%d wl=%d cut=%d prefix=%.0f suffix=%.0f planes=%.0f total=%.0f\n",
                    d->arm, c->wl, cut, tp, tsuf, tpl, _il_ab_now() - t0);
        return;
    }
    {
        static int prof = -1;
        double t0 = 0, t1 = 0;
        if (prof < 0)
            prof = getenv("VFFT_ILND_PROF") != NULL;
        if (prof)
            t0 = _il_ab_now();
        _il2d_col_exec(c, src, dst, rev);
        if (prof)
            t1 = _il_ab_now();
        for (p = 0; p < N0; p++)
            _ilnd_plane(d, dir, dst + 2 * p * rn);
        if (prof)
            fprintf(stderr, "[ilnd-prof] serial-unbanded s=%d axis0=%.0f planes=%.0f total=%.0f\n",
                    d->arm, t1 - t0, _il_ab_now() - t1, _il_ab_now() - t0);
    }
}

/* ── the MT partitions: pure loop restrictions of the serial walk ───── */
typedef struct
{
    const vfft_ilnd_t *d;
    const double *src;
    double *dst;
    vfft_dir_t dir;
    int mode, tid;   /* 0 bands, 1 column strips, 2 planes */
    size_t lo, hi;
} _ilnd_mt_arg;

static void _ilnd_mt_tramp(void *v)
{
    _ilnd_mt_arg *a = (_ilnd_mt_arg *)v;
    const vfft_ilnd_t *d = a->d;
    const vfft_ilcol_t *c = &d->ax0;
    const int rev = (a->dir == VFFT_BACKWARD);
    const size_t rn = d->plane;
    vfft_il2p_fn const *fns = rev ? c->b : c->f;
    double *const *tabs = rev ? c->tb : c->tf;
    size_t i, b;
    switch (a->mode)
    {
    case 0: /* bands of wl planes: the suffix stages, then the planes —
             * the serving order in both directions */
        for (b = a->lo; b < a->hi; b++)
        {
            const size_t b0 = b * (size_t)c->wl;
            const double *bs = (!rev && c->cut > 0) ? a->dst + 2 * b0 * rn
                                                    : a->src + 2 * b0 * rn;
            double *bd = a->dst + 2 * b0 * rn;
            _il2d_col_stages(bs, bd, c->wl, rn, c->cut, c->nst, c->R, c->L,
                             fns, tabs, rev);
            for (i = 0; i < (size_t)c->wl; i++)
                _ilnd_plane_t(d, a->tid, a->dir, bd + 2 * i * rn);
        }
        break;
    case 1: /* a column strip of the virtual plane: the whole axis-0 chain
             * (Bluestein: the window pipeline, windows share scr disjointly) */
        if (c->blu)
            _il2d_blu_cols_range(a->src, a->dst, c->N, rn, a->lo, a->hi, c->blu,
                                 c->nst, c->R, c->L, c->f, c->b, c->tf, c->tb,
                                 rev ? c->bluchb : c->bluchf,
                                 rev ? c->blukb : c->blukf, c->bluscr);
        else
            _il2d_col_pass_range(a->src, a->dst, c->N, rn, a->lo, a->hi, c->nst,
                                 c->R, c->L, fns, tabs, rev);
        break;
    default: /* planes [lo, hi) on dst */
        for (i = a->lo; i < a->hi; i++)
            _ilnd_plane_t(d, a->tid, a->dir, a->dst + 2 * i * rn);
    }
}

/* one phase across T workers (the caller is tid 0) */
static void _ilnd_mt_phase(const vfft_ilnd_t *d, const double *src, double *dst,
                           vfft_dir_t dir, int mode, size_t units, int T)
{
    _ilnd_mt_arg a[STRIDE_POOL_MAX_DISPATCH];
    int t;
    for (t = 0; t < T; t++)
    {
        a[t].d = d;
        a[t].src = src;
        a[t].dst = dst;
        a[t].dir = dir;
        a[t].mode = mode;
        a[t].tid = t;
        a[t].lo = units * (size_t)t / (size_t)T;
        a[t].hi = units * (size_t)(t + 1) / (size_t)T;
    }
    stride_pool_run(T, _ilnd_mt_tramp, a, sizeof a[0]);
}

static int _ilnd_clones_of(const vfft_ilnd_t *d) { return d->arm == 1 ? d->wn1 : d->wn2; }

/* Returns 1 when it ran threaded, 0 when the caller must run serial.
 * VFFT_ILND_PROF=1 prints the per-phase ns of every threaded execute
 * (diagnostic only; the env is read once). */
static int _ilnd_execute_mt(const vfft_ilnd_t *d, vfft_dir_t dir,
                            const double *src, double *dst)
{
    const vfft_ilcol_t *c = &d->ax0;
    const int rev = (dir == VFFT_BACKWARD);
    const size_t N0 = (size_t)d->N[0], rn = d->plane;
    const int T = stride_pool_workers_for(d->mt_t);
    static int prof = -1;
    double t0 = 0, t1 = 0, t2 = 0;
    int nsplit = 0, nserial = 0;
    if (prof < 0)
        prof = getenv("VFFT_ILND_PROF") != NULL;
    if (T < 2 || _ilnd_clones_of(d) < T - 1 || c->nat)
        return 0; /* every arm runs the structure => clones are mandatory */
    if (prof)
        t0 = _il_ab_now();
    if (d->mt == 1)
    {
        const size_t nb = c->wl > 0 ? N0 / (size_t)c->wl : 0;
        const int Tb = nb < (size_t)T ? (int)nb : T;
        int s;
        if (c->wl <= 0 || c->blu || nb < 2)
            return 0;
        /* fwd: the wide prefix must complete before ANY band (stage 0's
         * legs span the cube): each prefix stage's digits over the
         * workers, stages ordered, one dispatch each; a stage that cannot
         * split runs serial. bwd: the reversed prefix runs after. */
        if (!rev && c->cut > 0)
            for (s = 0; s < c->cut; s++)
            {
                const double *ssrc = (s == 0) ? src : dst;
                if (!_il2d_stage_digits_mt(ssrc, dst, c->N, rn, rn, c->R[s], c->L[s],
                                           c->f[s], c->tf[s], T))
                {
                    _il2d_col_stages(ssrc, dst, c->N, rn, s, s + 1, c->R, c->L,
                                     c->f, c->tf, 0);
                    nserial++;
                }
                else
                    nsplit++;
            }
        if (prof)
            t1 = _il_ab_now();
        _ilnd_mt_phase(d, (!rev && c->cut > 0) ? dst : src, dst, dir, 0, nb, Tb);
        if (prof)
            t2 = _il_ab_now();
        if (rev && c->cut > 0)
            for (s = c->cut - 1; s >= 0; s--)
                if (!_il2d_stage_digits_mt(dst, dst, c->N, rn, rn, c->R[s], c->L[s],
                                           c->b[s], c->tb[s], T))
                    _il2d_col_stages(dst, dst, c->N, rn, s, s + 1, c->R, c->L,
                                     c->b, c->tb, 0);
        if (prof)
            fprintf(stderr, "[ilnd-prof] band s=%d T=%d Tb=%d nb=%zu prefix=%.0f (split %d, serial %d, D0=%d) bands=%.0f total=%.0f\n",
                    d->arm, T, Tb, nb, t1 - t0, nsplit, nserial, c->L[0] / c->R[0], t2 - t1, _il_ab_now() - t0);
    }
    else if (d->mt == 2)
    {
        const int Ts = rn < (size_t)T ? (int)rn : T;
        const int Tp = N0 < (size_t)T ? (int)N0 : T;
        if (Ts < 2 && Tp < 2)
            return 0;
        if (Ts >= 2)
            _ilnd_mt_phase(d, src, dst, dir, 1, rn, Ts);
        else
            _il2d_col_exec(c, src, dst, rev);
        if (prof)
            t1 = _il_ab_now();
        _ilnd_mt_phase(d, src, dst, dir, 2, N0, Tp);
        if (prof)
            fprintf(stderr, "[ilnd-prof] plane s=%d T=%d Ts=%d Tp=%d strips=%.0f planes=%.0f total=%.0f\n",
                    d->arm, T, Ts, Tp, t1 - t0, _il_ab_now() - t1, _il_ab_now() - t0);
    }
    else
        return 0;
    _vfft_ilnd_mt_count++; /* engagement, see vfft_ilnd_mt_passes() */
    return 1;
}

static void vfft_ilnd_execute(const vfft_ilnd_t *d, vfft_dir_t dir,
                              const double *src, double *dst)
{
    if (d->mt > 0 && d->mt_t > 1 && _ilnd_execute_mt(d, dir, src, dst))
        return;
    _ilnd_execute_st(d, dir, src, dst);
}

/* ── clones: worker t > 0 needs its own mutable structure state ─────── */
static void _ilnd_free_clones(vfft_ilnd_t *d, int arm)
{
    int t;
    if (arm == 1 && d->childw)
    {
        for (t = 0; t < d->wn1; t++)
            if (d->childw[t])
                vfft_destroy((vfft_plan)d->childw[t]);
        free(d->childw);
        d->childw = NULL;
        d->wn1 = 0;
    }
    if (arm == 2)
    {
        if (d->roww)
        {
            for (t = 0; t < d->wn2; t++)
                if (d->roww[t])
                    vfft_destroy((vfft_plan)d->roww[t]);
            free(d->roww);
            d->roww = NULL;
        }
        if (d->ax1w)
        {
            for (t = 0; t < d->wn2; t++)
                free(d->ax1w[t].bluscr); /* the only per-clone allocation */
            free(d->ax1w);
            d->ax1w = NULL;
        }
        d->wn2 = 0;
    }
}

static void _ilnd_free_arm(vfft_ilnd_t *d, int arm)
{
    _ilnd_free_clones(d, arm);
    if (arm == 1 && d->child)
    {
        vfft_destroy((vfft_plan)d->child);
        d->child = NULL;
    }
    if (arm == 2)
    {
        _il2d_col_free(&d->ax1);
        if (d->row)
            vfft_destroy((vfft_plan)d->row);
        d->row = NULL;
    }
}

static void vfft_ilnd_destroy(vfft_ilnd_t *d)
{
    if (!d)
        return;
    _ilnd_free_arm(d, 1);
    _ilnd_free_arm(d, 2);
    _il2d_col_free(&d->ax0);
    free(d);
}

/* build T-1 clones of one structure; clones read warm wisdom and never
 * bank. Returns the count built (0 = that structure cannot thread, loud). */
static int _ilnd_build_clones(vfft_ilnd_t *d, const vfft_config_t *cfg, int T, int arm)
{
    const int n = (T > STRIDE_POOL_MAX_DISPATCH ? STRIDE_POOL_MAX_DISPATCH : T) - 1;
    int t;
    if (n <= 0)
        return 0;
    if (arm == 1)
    {
        vfft_config_t cc;
        if (d->wn1 || !d->child)
            return d->wn1;
        memset(&cc, 0, sizeof cc);
        cc.transform = VFFT_C2C;
        cc.placement = VFFT_INPLACE;
        cc.rigor = cfg->rigor;
        cc.dims = 2;
        cc.n[0] = d->N[1];
        cc.n[1] = d->N[2];
        cc.howmany = 1;
        cc.order = cfg->order;
        cc.layout = VFFT_LAYOUT_INTERLEAVED;
        cc.nthreads = 1;
        cc.wisdom = cfg->wisdom;
        cc.wisdom_write = 0;
        d->childw = (struct vfft_plan_s **)calloc((size_t)n, sizeof *d->childw);
        if (!d->childw)
            return 0;
        for (t = 0; t < n; t++)
        {
            struct vfft_plan_s *c = (struct vfft_plan_s *)vfft_create(&cc);
            d->childw[t] = c;
            if (!c || !_ilnd_child_equiv(d->child, c) || c->nthreads > 1)
            {
                _vfft_warn("ilnd MT: 2D child clone %d %s at %dx%d — the child structure "
                           "cannot thread for this plan",
                           t, c ? "route-mismatched" : "failed to create",
                           d->N[1], d->N[2]);
                d->wn1 = t + 1;
                _ilnd_free_clones(d, 1);
                return 0;
            }
        }
        d->wn1 = n;
        return n;
    }
    else
    {
        vfft_config_t rc;
        if (d->wn2 || !d->row)
            return d->wn2;
        memset(&rc, 0, sizeof rc);
        rc.transform = VFFT_C2C;
        rc.placement = VFFT_INPLACE;
        rc.rigor = cfg->rigor;
        rc.dims = 1;
        rc.n[0] = d->N[2];
        rc.howmany = 1;
        rc.order = VFFT_ORDER_NATURAL;
        rc.layout = VFFT_LAYOUT_INTERLEAVED;
        rc.nthreads = 1;
        rc.wisdom = cfg->wisdom;
        rc.wisdom_write = 0;
        d->roww = (struct vfft_plan_s **)calloc((size_t)n, sizeof *d->roww);
        d->ax1w = (vfft_ilcol_t *)calloc((size_t)n, sizeof *d->ax1w);
        if (!d->roww || !d->ax1w)
        {
            free(d->roww); free(d->ax1w); d->roww = NULL; d->ax1w = NULL;
            return 0;
        }
        for (t = 0; t < n; t++)
        {
            struct vfft_plan_s *c = (struct vfft_plan_s *)vfft_create(&rc);
            d->roww[t] = c;
            d->ax1w[t] = d->ax1;            /* shared read-only tables */
            d->ax1w[t].bluscr = NULL;
            if (d->ax1.blu)
                d->ax1w[t].bluscr = (double *)malloc(
                    2 * (size_t)d->ax1.blu * (size_t)d->N[2] * sizeof(double));
            if (!c || !_tc_clone_equiv(d->row, c) || c->tcb || c->tcbw ||
                (d->ax1.blu && !d->ax1w[t].bluscr))
            {
                _vfft_warn("ilnd MT: row clone %d %s at N3=%d — the flat structure cannot "
                           "thread for this plan",
                           t, c ? "route-mismatched" : "failed to create", d->N[2]);
                d->wn2 = t + 1;
                _ilnd_free_clones(d, 2);
                return 0;
            }
        }
        d->wn2 = n;
        return n;
    }
}

/* ── the banded walk's width: legal iff wl | N and a suffix stage's span
 * divides wl (the tcut law: the width is the INPUT, the cut is DERIVED);
 * -1 = illegal (stay unbanded) ─────────────────────────────────────── */
static int _ilnd_wl_cut(const vfft_ilcol_t *c, int wl)
{
    int s;
    if (wl <= 0 || wl > c->N || c->N % wl)
        return -1;
    for (s = 0; s < c->nst; s++)
        if (wl % c->L[s] == 0)
            return s;
    return -1;
}
static void _ilnd_apply_wl(vfft_ilcol_t *c, int wl)
{
    const int cut = (c->blu || c->nat) ? -1 : _ilnd_wl_cut(c, wl);
    c->wl = cut >= 0 ? wl : 0;
    c->cut = cut >= 0 ? cut : 0;
    c->tfuse = (cut >= 0 && wl > 0);
}
/* the width pool (the 2D axis race's, E1.2): 0 + WPOOL filtered by
 * legality + the chain's own stage spans gated by live L2 residency of
 * a band (w * plane * 16 <= L2) — candidates, never defaults */
static int _ilnd_wl_pool(const vfft_ilcol_t *c, int *out, int max)
{
    static const int WPOOL[] = { 8, 16, 32, 64, 128, 256 };
    int n = 0, p, s;
    out[n++] = 0;
    if (c->blu || c->nat)
        return n;
    for (p = 0; p < 6 && n < max; p++)
        if (_ilnd_wl_cut(c, WPOOL[p]) >= 0)
            out[n++] = WPOOL[p];
    for (s = 1; s < c->nst && n < max; s++)
    {
        const int w = c->L[s];
        int dup = 0, q;
        if (w < 8 || _ilnd_wl_cut(c, w) < 0)
            continue;
        if ((long)w * (long)c->rn * 16 > vfft_cpu_l2_bytes())
            continue;
        for (q = 0; q < n; q++)
            if (out[q] == w)
                dup = 1;
        if (!dup)
            out[n++] = w;
    }
    return n;
}

/* ── the arms' builders ─────────────────────────────────────────────── */
static int _ilnd_build_child(vfft_ilnd_t *d, const vfft_config_t *cfg)
{
    vfft_config_t cc;
    if (d->child)
        return 1;
    memset(&cc, 0, sizeof cc);
    cc.transform = VFFT_C2C;
    cc.placement = VFFT_INPLACE;
    cc.rigor = cfg->rigor;
    cc.dims = 2;
    cc.n[0] = d->N[1];
    cc.n[1] = d->N[2];
    cc.howmany = 1;
    cc.order = cfg->order;
    cc.layout = VFFT_LAYOUT_INTERLEAVED;
    cc.nthreads = 1;
    cc.wisdom = cfg->wisdom;
    cc.wisdom_write = cfg->wisdom_write;
    cc.recalibrate = cfg->recalibrate;
    d->child = (struct vfft_plan_s *)vfft_create(&cc);
    return d->child != NULL;
}

static int _ilnd_build_flat(vfft_ilnd_t *d, struct vfft_wisdom_s *W,
                            const vfft_config_t *cfg, const vw2_ilcol_key_t *key0)
{
    vw2_ilcol_key_t key1 = *key0;
    vfft_config_t rc;
    int bwl, btf, bro, bcmt, bcmtt, bblu;
    if (d->row)
        return 1;
    key1.axis = 1;
    if (!_il2d_col_build(W, cfg, &key1, d->N[1], (size_t)d->N[2], 0, &d->ax1,
                         d->forms1, sizeof d->forms1, &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu))
        return 0;
    memset(&rc, 0, sizeof rc);
    rc.transform = VFFT_C2C;
    rc.placement = VFFT_INPLACE;
    rc.rigor = cfg->rigor;
    rc.dims = 1;
    rc.n[0] = d->N[2];
    rc.howmany = 1;
    rc.order = VFFT_ORDER_NATURAL;
    rc.layout = VFFT_LAYOUT_INTERLEAVED;
    rc.nthreads = 1;
    rc.wisdom = cfg->wisdom;
    rc.wisdom_write = cfg->wisdom_write;
    d->row = (struct vfft_plan_s *)vfft_create(&rc);
    if (!d->row)
    {
        _il2d_col_free(&d->ax1);
        return 0;
    }
    return 1;
}

/* the (structure, width) race: the whole forward, in place on scratch,
 * every configuration an arm of ONE alternated race */
typedef struct { vfft_ilnd_t *d; double *z; int arm; int wl; char name[24]; } _ilnd_arm_ctx_t;
static void _ilnd_arm_run(void *v)
{
    _ilnd_arm_ctx_t *c = (_ilnd_arm_ctx_t *)v;
    c->d->arm = c->arm;
    _ilnd_apply_wl(&c->d->ax0, c->wl);
    _ilnd_execute_st(c->d, VFFT_FORWARD, c->z, c->z);
}

/* the MT race at the plan's T: serial (the one-thread structure) vs each
 * (partition, structure) that can ENGAGE, the whole forward through the
 * very code execute serves with. Returns the winning (mt, structure). */
typedef struct { vfft_ilnd_t *d; double *z; int mt; int arm; int ok; char name[24]; } _ilnd_mt_ctx_t;
static void _ilnd_mt_arm_run(void *v)
{
    _ilnd_mt_ctx_t *c = (_ilnd_mt_ctx_t *)v;
    c->d->arm = c->arm;
    if (c->mt == 0)
    {
        _ilnd_execute_st(c->d, VFFT_FORWARD, c->z, c->z);
        return;
    }
    c->d->mt = c->mt;
    if (c->ok && !_ilnd_execute_mt(c->d, VFFT_FORWARD, c->z, c->z))
        c->ok = 0; /* the arm cannot engage on this cell */
}
static void _ilnd_mt_race(vfft_ilnd_t *d, const int s0, int *mt_out, int *arm_out)
{
    const size_t T = (size_t)d->N[0] * d->plane;
    double *z = (double *)malloc(2 * T * sizeof(double));
    _ilnd_mt_ctx_t cx[5];
    vfft_race_arm_t arms[5];
    double ns[5] = { 1e300, 1e300, 1e300, 1e300, 1e300 };
    int na = 0, a, best = 0, st, reps;
    size_t i;
    *mt_out = 0;
    *arm_out = s0;
    if (!z)
        return;
    for (i = 0; i < 2 * T; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 1023);
    /* reps from ONE serial timing: every sample runs >= ~20 ms of serial-
     * equivalent work so a threaded arm's steady state is what gets timed */
    {
        double t0;
        d->arm = s0;
        _ilnd_execute_st(d, VFFT_FORWARD, z, z);
        t0 = _il_ab_now();
        _ilnd_execute_st(d, VFFT_FORWARD, z, z);
        t0 = _il_ab_now() - t0;
        reps = (int)(20e6 / (t0 > 1.0 ? t0 : 1.0));
        if (reps < 2) reps = 2;
        if (reps > 256) reps = 256;
    }
#define ILND_ARM(MT, ARM, NAME) do { \
        cx[na].d = d; cx[na].z = z; cx[na].mt = (MT); cx[na].arm = (ARM); cx[na].ok = 1; \
        snprintf(cx[na].name, sizeof cx[na].name, "%s/%s", NAME, (ARM) == 1 ? "child" : "flat"); \
        arms[na].name = cx[na].name; arms[na].run = _ilnd_mt_arm_run; arms[na].ctx = &cx[na]; na++; \
    } while (0)
    ILND_ARM(0, s0, "serial");
    for (st = 1; st <= 2; st++)
    {
        const int have = (st == 1) ? (d->child != NULL && d->wn1 > 0) : (d->row != NULL && d->wn2 > 0);
        if (!have)
            continue;
        if (d->ax0.wl > 0 && !d->ax0.blu && (size_t)d->N[0] / (size_t)d->ax0.wl >= 2)
            ILND_ARM(1, st, "band");
        ILND_ARM(2, st, "plane");
    }
#undef ILND_ARM
    {
        const vfft_race_proto_t proto = { 3, reps, VFFT_RACE_MIN, 1, 2, NULL, NULL };
        vfft_race_run(&proto, arms, na, ns);
    }
    for (a = 1; a < na; a++)
        if (cx[a].ok && ns[a] < ns[best])
            best = a;
    *mt_out = cx[best].mt;
    *arm_out = cx[best].arm;
    free(z);
    if (getenv("VFFT_IL2D_LOG"))
    {
        fprintf(stderr, "[ilnd] %dx%dx%d: MT race T=%d reps=%d", d->N[0], d->N[1], d->N[2], d->mt_t, reps);
        for (a = 0; a < na; a++)
            fprintf(stderr, " %s=%.0f%s", cx[a].name, ns[a], cx[a].ok ? "" : "(no engage)");
        fprintf(stderr, " -> %s/%s\n", *mt_out == 0 ? "serial" : *mt_out == 1 ? "band" : "plane",
                *arm_out == 1 ? "child" : "flat");
    }
}

/* ── the create: rank-3 interleaved c2c, out of place, DEFAULT/SCRAMBLED ── */
static vfft_plan _vfft_create_fftnd_il(const vfft_config_t *cfg,
                                       struct vfft_wisdom_s *W,
                                       const vfft_proto_registry_t *reg,
                                       size_t K)
{
    const int N1 = cfg->n[0], N2 = cfg->n[1], N3 = cfg->n[2];
    vfft_ilnd_t *d;
    struct vfft_plan_s *h;
    vw2_ilcol_key_t key0;
    int bwl, btf, bro, bcmt, bcmtt, bblu;
    int sarm[2], nsarm = 0, wls[16], nwl = 0;
    int arm = 0, wl = 0, s_src = 0, wl_src = 0, mt_src = 0; /* src: 1 env, 2 wisdom, 3 race, 4 only-buildable */
    int mts = 0; /* the structure the threaded verdict runs with */
    const int usable_w = (W && !W->vw2_off_2d);
    const int nthr = _vfft_plan_threads(cfg);
    const char *pin = getenv("VFFT_ILND_ARM");
    const char *wpin = getenv("VFFT_ILND_WL");
    const char *mpin = getenv("VFFT_ILND_MT");
    (void)reg;
    /* IN PLACE (2026-09-07): the same plan and the same wisdom row serve
     * both placements — every pass is the 2D tier's alias-tolerant kind
     * (axis 0 src -> dst with src == dst, the bands and the structure in
     * place by construction, the strips per column), and the create race
     * already times in place on scratch. Output bitwise the out-of-place
     * output (the probe checks it). */
    if (cfg->transform != VFFT_C2C || cfg->dims != 3 || K != 1 ||
        (cfg->order != VFFT_ORDER_DEFAULT && cfg->order != VFFT_ORDER_SCRAMBLED))
    {
        _vfft_warn("vfft_create: 3D INTERLEAVED serves C2C, howmany==1, either placement, "
                   "order DEFAULT/SCRAMBLED today (got %s, howmany=%zu, order=%d); "
                   "natural order, real and rank 4 are the tier's next phases",
                   _vfft_tname(cfg->transform), K, cfg->order);
        return NULL;
    }
    if (N1 < 2 || N2 < 2 || N3 < 2)
    {
        _vfft_warn("vfft_create: 3D INTERLEAVED c2c needs every dim >= 2 (got %dx%dx%d)",
                   N1, N2, N3);
        return NULL;
    }
    d = (vfft_ilnd_t *)calloc(1, sizeof *d);
    if (!d)
        return NULL;
    d->rank = 3;
    d->N[0] = N1; d->N[1] = N2; d->N[2] = N3;
    d->plane = (size_t)N2 * (size_t)N3;
    d->mt_t = nthr;
    /* the column build's Bluestein inner-chain provider reads this create */
    _il2d_blu_ctx.W = W;
    _il2d_blu_ctx.cfg = cfg;
    /* axis 0: the rank-3 row's own tokens; the wisdom order cell is the
     * scrambled one (DEFAULT and SCRAMBLED spell the same serving) */
    key0.rank = 3; key0.n0 = N1; key0.n1 = N2; key0.n2 = N3;
    key0.ord = VW2_ORD_SCR; key0.axis = 0; key0.real = 0;
    if (!_il2d_col_build(W, cfg, &key0, N1, d->plane, 0, &d->ax0,
                         d->forms0, sizeof d->forms0, &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu))
    {
        free(d);
        return NULL;
    }
    /* the STRUCTURE candidates: env pin (never banks) > banked s= > both */
    if (pin && (atoi(pin) == 1 || atoi(pin) == 2))
    {
        sarm[nsarm++] = atoi(pin);
        s_src = 1;
    }
    else if (usable_w && !cfg->recalibrate && (arm = vw2_ilnd_arm_lookup(&W->vw2, &key0)) > 0)
    {
        sarm[nsarm++] = arm;
        s_src = 2;
    }
    else
    {
        sarm[nsarm++] = 1;
        sarm[nsarm++] = 2;
    }
    /* the WIDTH candidates: env pin > banked wl= > the pool */
    if (wpin)
    {
        const int w = atoi(wpin);
        if (w > 0 && _ilnd_wl_cut(&d->ax0, w) < 0)
            _vfft_warn("VFFT_ILND_WL=%d illegal at %dx%dx%d (needs wl | N1 and a stage "
                       "with L_s | wl) — unbanded", w, N1, N2, N3);
        wls[nwl++] = (w > 0 && _ilnd_wl_cut(&d->ax0, w) >= 0) ? w : 0;
        wl_src = 1;
    }
    else if (usable_w && !cfg->recalibrate && bwl >= 0)
    {
        if (bwl > 0 && _ilnd_wl_cut(&d->ax0, bwl) < 0)
            _vfft_warn("banked wl=%d does not fit the axis-0 chain at %dx%dx%d — unbanded",
                       bwl, N1, N2, N3);
        wls[nwl++] = (bwl > 0 && _ilnd_wl_cut(&d->ax0, bwl) >= 0) ? bwl : 0;
        wl_src = 2;
    }
    else
        nwl = _ilnd_wl_pool(&d->ax0, wls, 14);
    /* build every structure the candidates need */
    {
        int i, ok1 = 0, ok2 = 0, want1 = 0, want2 = 0;
        for (i = 0; i < nsarm; i++)
        {
            if (sarm[i] == 1) want1 = 1;
            if (sarm[i] == 2) want2 = 1;
        }
        if (want1) ok1 = _ilnd_build_child(d, cfg);
        if (want2) ok2 = _ilnd_build_flat(d, W, cfg, &key0);
        if (!ok1 && !ok2)
        {
            _vfft_warn("vfft_create: 3D INTERLEAVED c2c %dx%dx%d — no structure arm could "
                       "be built (%s)", N1, N2, N3,
                       s_src == 1 ? "env pin" : s_src == 2 ? "banked verdict"
                                  : "no 2D IL plan at the plane and no axis-1 chain");
            vfft_ilnd_destroy(d);
            return NULL;
        }
        nsarm = 0;
        if (ok1) sarm[nsarm++] = 1;
        if (ok2) sarm[nsarm++] = 2;
        if (want1 + want2 == 2 && nsarm == 1)
            s_src = 4;
    }
    if (nsarm * nwl == 1)
    {   /* nothing to race: serve the one configuration */
        arm = sarm[0];
        wl = wls[0];
    }
    else
    {
        const size_t T = (size_t)N1 * d->plane;
        double *z = (double *)malloc(2 * T * sizeof(double));
        _ilnd_arm_ctx_t ac[VFFT_RACE_MAX_ARMS];
        vfft_race_arm_t arms[VFFT_RACE_MAX_ARMS];
        double ns[VFFT_RACE_MAX_ARMS];
        int na = 0, a, si, wi, best = 0;
        int reps = (int)(1e6 / (double)(T + 1));
        if (reps < 1) reps = 1;
        if (reps > 64) reps = 64;
        if (!z)
        {
            vfft_ilnd_destroy(d);
            return NULL;
        }
        {
            size_t i;
            for (i = 0; i < 2 * T; i++)
                z[i] = 1.0 + 1e-6 * (double)(i & 1023);
        }
        for (si = 0; si < nsarm; si++)
            for (wi = 0; wi < nwl && na < VFFT_RACE_MAX_ARMS; wi++)
            {
                ac[na].d = d;
                ac[na].z = z;
                ac[na].arm = sarm[si];
                ac[na].wl = wls[wi];
                snprintf(ac[na].name, sizeof ac[na].name, "%s/wl%d",
                         sarm[si] == 1 ? "child" : "flat", wls[wi]);
                arms[na].name = ac[na].name;
                arms[na].run = _ilnd_arm_run;
                arms[na].ctx = &ac[na];
                na++;
            }
        {
            const vfft_race_proto_t proto = { 3, reps, VFFT_RACE_MIN, 1, 0, NULL, NULL };
            vfft_race_run(&proto, arms, na, ns);
        }
        for (a = 1; a < na; a++)
            if (ns[a] < ns[best])
                best = a;
        arm = ac[best].arm;
        wl = ac[best].wl;
        free(z);
        if (getenv("VFFT_IL2D_LOG"))
        {
            fprintf(stderr, "[ilnd] %dx%dx%d: race", N1, N2, N3);
            for (a = 0; a < na; a++)
                fprintf(stderr, " %s=%.0f", ac[a].name, ns[a]);
            fprintf(stderr, " -> %s wl=%d\n", arm == 1 ? "child" : "flat", wl);
        }
        if (nsarm > 1) s_src = 3;
        if (nwl > 1) wl_src = 3;
        /* bank what was RACED (pins never bank) */
        if (usable_w && cfg->wisdom_write)
        {
            int banked = 0;
            if (nsarm > 1 && vw2_ilnd_arm_bank(&W->vw2, &key0, arm))
                banked = 1;
            if (nwl > 1 && vw2_ilcol_chain_bank(&W->vw2, &key0, d->ax0.R, d->ax0.nst,
                                                wl, wl > 0, -1, -1, -1, -1, 0.0) == VW2_OK)
                banked = 1;
            if (banked)
                _vw2_persist(W, cfg);
        }
    }
    d->arm = arm;
    mts = arm;
    _ilnd_apply_wl(&d->ax0, wl);
    /* ── MT at the plan's T: env pin > the banked (cmt, cmts) at THIS T >
     * the race of (partition x structure) — the structure that wins at one
     * thread need not win threaded, so both structures stay alive with
     * their clones until the threaded verdict. No clones for a structure =
     * it cannot thread; no clones at all = cmt=0, banked like a yes. */
    d->mt = 0;
    if (nthr > 1)
    {
        int c1 = 0, c2 = 0;
        if (mpin)
        {
            d->mt = atoi(mpin);
            if (d->mt < 0 || d->mt > 2) d->mt = 0;
            if (d->mt > 0 && !_ilnd_build_clones(d, cfg, nthr, arm))
                d->mt = 0;
            mt_src = 1;
        }
        else if (usable_w && !cfg->recalibrate && bcmt >= 0 && bcmtt == nthr)
        {
            const int bs = vw2_ilnd_mts_lookup(&W->vw2, &key0);
            d->mt = (bcmt >= 0 && bcmt <= 2) ? bcmt : 0;
            mts = (bs == 1 || bs == 2) ? bs : arm;
            if (d->mt > 0)
            {   /* the threaded structure may differ from the one-thread one */
                const int okb = (mts == 1) ? _ilnd_build_child(d, cfg) : _ilnd_build_flat(d, W, cfg, &key0);
                if (!okb || !_ilnd_build_clones(d, cfg, nthr, mts))
                {
                    _vfft_warn("ilnd: the banked threaded structure (cmts=%d) cannot be served at "
                               "%dx%dx%d T=%d — serial", mts, N1, N2, N3, nthr);
                    d->mt = 0;
                    mts = arm;
                }
            }
            mt_src = 2;
        }
        else
        {
            /* both structures, each with its clones, race against serial */
            int mt_v = 0, arm_v = arm;
            if (_ilnd_build_child(d, cfg))
                c1 = _ilnd_build_clones(d, cfg, nthr, 1);
            if (_ilnd_build_flat(d, W, cfg, &key0))
                c2 = _ilnd_build_clones(d, cfg, nthr, 2);
            if (c1 || c2)
                _ilnd_mt_race(d, arm, &mt_v, &arm_v);
            d->mt = mt_v;
            mts = (mt_v > 0) ? arm_v : arm;
            mt_src = (c1 || c2) ? 3 : 4;
            if (usable_w && cfg->wisdom_write && !pin && !wpin)
            {
                int banked = 0;
                if (vw2_ilcol_chain_bank(&W->vw2, &key0, d->ax0.R, d->ax0.nst, -1, -1, -1,
                                         d->mt, nthr, -1, 0.0) == VW2_OK)
                    banked = 1;
                if (vw2_ilnd_mts_bank(&W->vw2, &key0, mts))
                    banked = 1;
                if (banked)
                    _vw2_persist(W, cfg);
            }
        }
        if (d->mt == 1 && (d->ax0.wl <= 0 || d->ax0.blu))
        {
            _vfft_warn("ilnd: the band MT arm needs a banded axis 0 at %dx%dx%d — serial",
                       N1, N2, N3);
            d->mt = 0;
            mts = arm;
        }
    }
    /* the serving structure: the threaded one when the plan threads (it is
     * correct serially too, and the pool clamp may leave it serial), else
     * the one-thread verdict; the other structure and its clones go */
    d->arm = (d->mt > 0) ? mts : arm;
    _ilnd_free_arm(d, d->arm == 1 ? 2 : 1);
    if (d->mt == 0)
        _ilnd_free_clones(d, d->arm);
    if (getenv("VFFT_IL2D_LOG"))
    {
        static const char *SRC[] = { "?", "env", "wisdom", "race", "only-buildable" };
        fprintf(stderr, "[ilnd] %dx%dx%d: structure %s src=%s | axis-0 wl=%d cut=%d src=%s"
                        " | T=%d mt=%s/%s src=%s clones=%d\n",
                N1, N2, N3, arm == 1 ? "child" : "flat", SRC[s_src],
                d->ax0.wl, d->ax0.cut, SRC[wl_src], nthr,
                d->mt == 0 ? "serial" : d->mt == 1 ? "band" : "plane",
                d->arm == 1 ? "child" : "flat",
                nthr > 1 ? SRC[mt_src] : "-", _ilnd_clones_of(d));
    }
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_ilnd_destroy(d);
        return NULL;
    }
    h->transform = VFFT_C2C;
    h->placement = cfg->placement;
    h->layout = (int)cfg->layout;
    h->N = N1;
    h->N2 = N2;
    h->N3 = N3;
    h->K = 1;
    h->nthreads = nthr;
    h->ilnd = d;
    return h;
}

#endif /* VFFT_TRANSFORMS_FFTND_FFTND_IL_H */
