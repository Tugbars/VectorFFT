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
 * Every pass commutes with every other (each is a Kronecker factor), so
 * forward and backward run the same pass order. Output order: DEFAULT/
 * SCRAMBLED = each column axis digit-reversed by its chain, rows natural.
 *
 * Contracts (phase 2): C2C, rank 3, howmany == 1, OUT OF PLACE, order
 * DEFAULT or SCRAMBLED, single thread. NATURAL, in place, MT, real and
 * rank 4 follow in later phases — refused loudly until then, never bridged.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: after il2d_tier.h (the column build
 * and execute), k1_commit.h (support/race.h) and the wisdom readers, before
 * fftnd_create.h (which dispatches here for rank-3 INTERLEAVED c2c).
 */
#ifndef VFFT_TRANSFORMS_FFTND_FFTND_IL_H
#define VFFT_TRANSFORMS_FFTND_FFTND_IL_H

typedef struct vfft_ilnd_s {
    int rank;
    int N[4];
    size_t plane;                 /* complex per axis-0 row: N[1] * ... * N[rank-1] */
    int arm;                      /* the RACED structure: 1 = the child per plane, 2 = flat */
    vfft_ilcol_t ax0;             /* axis 0: N[0] rows over `plane` complex (wl/cut = the banded walk) */
    struct vfft_plan_s *child;    /* arm 1: the rank-(n-1) IL c2c plan, in place, per plane */
    vfft_ilcol_t ax1;             /* arm 2: N[1] rows over N[2] complex, per plane */
    struct vfft_plan_s *row;      /* arm 2: the K=1 IL row plan, in place, natural */
    char forms0[64], forms1[64];
} vfft_ilnd_t;

/* ── the per-plane structure ─────────────────────────────────────────── */
static void _ilnd_plane(const vfft_ilnd_t *d, vfft_dir_t dir, double *pl)
{
    if (d->arm == 1)
        vfft_execute((vfft_plan)d->child, dir, pl, NULL, pl, NULL);
    else
    {
        const size_t rn = (size_t)d->N[2];
        size_t r;
        _il2d_col_exec(&d->ax1, pl, pl, dir == VFFT_BACKWARD);
        for (r = 0; r < (size_t)d->N[1]; r++)
            vfft_execute((vfft_plan)d->row, dir, pl + 2 * r * rn, NULL,
                         pl + 2 * r * rn, NULL);
    }
}

/* ── execute: axis 0 (src -> dst), the per-plane structure on dst ───── */
static void vfft_ilnd_execute(const vfft_ilnd_t *d, vfft_dir_t dir,
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
        if (!rev && cut > 0)
            _il2d_col_stages(src, dst, c->N, rn, 0, cut, c->R, c->L, fns, tabs, 0);
        for (b0 = 0; b0 < N0; b0 += wl)
        {
            const double *bs = (!rev && cut > 0) ? dst + 2 * b0 * rn : src + 2 * b0 * rn;
            double *bd = dst + 2 * b0 * rn;
            _il2d_col_stages(bs, bd, (int)wl, rn, cut, nst, c->R, c->L, fns, tabs, rev);
            for (p = 0; p < wl; p++)
                _ilnd_plane(d, dir, bd + 2 * p * rn);
        }
        if (rev && cut > 0)
            _il2d_col_stages(dst, dst, c->N, rn, 0, cut, c->R, c->L, fns, tabs, 1);
        return;
    }
    _il2d_col_exec(c, src, dst, rev);
    for (p = 0; p < N0; p++)
        _ilnd_plane(d, dir, dst + 2 * p * rn);
}

static void vfft_ilnd_destroy(vfft_ilnd_t *d)
{
    if (!d)
        return;
    _il2d_col_free(&d->ax0);
    _il2d_col_free(&d->ax1);
    if (d->child)
        vfft_destroy((vfft_plan)d->child);
    if (d->row)
        vfft_destroy((vfft_plan)d->row);
    free(d);
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

static void _ilnd_free_arm(vfft_ilnd_t *d, int arm)
{
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

/* the (structure, width) race: the whole forward, in place on scratch,
 * every configuration an arm of ONE alternated race */
typedef struct { vfft_ilnd_t *d; double *z; int arm; int wl; char name[24]; } _ilnd_arm_ctx_t;
static void _ilnd_arm_run(void *v)
{
    _ilnd_arm_ctx_t *c = (_ilnd_arm_ctx_t *)v;
    c->d->arm = c->arm;
    _ilnd_apply_wl(&c->d->ax0, c->wl);
    vfft_ilnd_execute(c->d, VFFT_FORWARD, c->z, c->z);
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
    int arm = 0, wl = 0, s_src = 0, wl_src = 0; /* src: 1 env, 2 wisdom, 3 race, 4 only-buildable */
    const int usable_w = (W && !W->vw2_off_2d);
    const char *pin = getenv("VFFT_ILND_ARM");
    const char *wpin = getenv("VFFT_ILND_WL");
    (void)reg;
    if (cfg->transform != VFFT_C2C || cfg->dims != 3 || K != 1 ||
        cfg->placement != VFFT_OUTOFPLACE ||
        (cfg->order != VFFT_ORDER_DEFAULT && cfg->order != VFFT_ORDER_SCRAMBLED))
    {
        _vfft_warn("vfft_create: 3D INTERLEAVED serves C2C, howmany==1, out of place, "
                   "order DEFAULT/SCRAMBLED today (got %s, howmany=%zu, %s, order=%d); "
                   "natural order, in place, real and rank 4 are the tier's next phases",
                   _vfft_tname(cfg->transform), K,
                   cfg->placement == VFFT_INPLACE ? "in place" : "out of place", cfg->order);
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
    _ilnd_apply_wl(&d->ax0, wl);
    _ilnd_free_arm(d, arm == 1 ? 2 : 1);
    if (getenv("VFFT_IL2D_LOG"))
    {
        static const char *SRC[] = { "?", "env", "wisdom", "race", "only-buildable" };
        fprintf(stderr, "[ilnd] %dx%dx%d: structure %s src=%s | axis-0 wl=%d cut=%d src=%s\n",
                N1, N2, N3, arm == 1 ? "child" : "flat", SRC[s_src],
                d->ax0.wl, d->ax0.cut, SRC[wl_src]);
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
    h->nthreads = _vfft_plan_threads(cfg);
    h->ilnd = d;
    return h;
}

#endif /* VFFT_TRANSFORMS_FFTND_FFTND_IL_H */
