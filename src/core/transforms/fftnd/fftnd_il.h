/**
 * fftnd_il.h — the rank-N INTERLEAVED c2c tier (2026-09-06; docs: memory
 * fftnd_il_campaign). Rank 3 today; rank 4 by the same composition.
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
 * Both are timed on the whole forward at create (alternated, min of 3),
 * the winner banks as s= on the cell's rank-3 lay=il row beside the axis-0
 * chain tokens (chain= wl= ... forms=) and the flat arm's axis-1 tokens
 * (chain1= ... forms1=); the child's verdicts live on the child's cell.
 *
 * Every pass commutes with every other (each is a Kronecker factor), so
 * forward and backward run the same pass order: axis 0 src -> dst (the
 * OOP move is stage 0's, the kinds are alias-tolerant), then per plane in
 * place on dst. Output order: DEFAULT/SCRAMBLED = each column axis
 * digit-reversed by its chain, rows natural (the 2D contract per axis).
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
    vfft_ilcol_t ax0;             /* axis 0: N[0] rows over `plane` complex */
    struct vfft_plan_s *child;    /* arm 1: the rank-(n-1) IL c2c plan, in place, per plane */
    vfft_ilcol_t ax1;             /* arm 2: N[1] rows over N[2] complex, per plane */
    struct vfft_plan_s *row;      /* arm 2: the K=1 IL row plan, in place, natural */
    char forms0[64], forms1[64];
} vfft_ilnd_t;

/* ── execute: axis 0 wide (src -> dst), then the per-plane arm on dst ── */
static void vfft_ilnd_execute(const vfft_ilnd_t *d, vfft_dir_t dir,
                              const double *src, double *dst)
{
    const int rev = (dir == VFFT_BACKWARD);
    const size_t N0 = (size_t)d->N[0];
    size_t p;
    _il2d_col_exec(&d->ax0, src, dst, rev);
    for (p = 0; p < N0; p++)
    {
        double *pl = dst + 2 * p * d->plane;
        if (d->arm == 1)
            vfft_execute((vfft_plan)d->child, dir, pl, NULL, pl, NULL);
        else
        {
            const size_t rn = (size_t)d->N[2];
            size_t r;
            _il2d_col_exec(&d->ax1, pl, pl, rev);
            for (r = 0; r < (size_t)d->N[1]; r++)
                vfft_execute((vfft_plan)d->row, dir, pl + 2 * r * rn, NULL,
                             pl + 2 * r * rn, NULL);
        }
    }
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

/* the arm race: the whole forward, in place on scratch, alternated */
typedef struct { vfft_ilnd_t *d; double *z; int arm; } _ilnd_arm_ctx_t;
static void _ilnd_arm_run(void *v)
{
    _ilnd_arm_ctx_t *c = (_ilnd_arm_ctx_t *)v;
    c->d->arm = c->arm;
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
    int arm = 0, banked_arm = 0;
    const char *pin = getenv("VFFT_ILND_ARM");
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
    /* the STRUCTURE: env pin (never banks) > banked verdict > the race */
    if (pin && (atoi(pin) == 1 || atoi(pin) == 2))
        arm = atoi(pin);
    else if (W && !W->vw2_off_2d && !cfg->recalibrate)
        arm = banked_arm = vw2_ilnd_arm_lookup(&W->vw2, &key0);
    if (arm == 1 || arm == 2)
    {
        const int ok = (arm == 1) ? _ilnd_build_child(d, cfg) : _ilnd_build_flat(d, W, cfg, &key0);
        if (!ok)
        {
            _vfft_warn("vfft_create: 3D INTERLEAVED c2c %dx%dx%d — the %s structure arm "
                       "could not be built (%s)", N1, N2, N3,
                       arm == 1 ? "child-per-plane" : "flat", pin ? "env pin" : "banked verdict");
            vfft_ilnd_destroy(d);
            return NULL;
        }
        d->arm = arm;
        if (getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[ilnd] %dx%dx%d: structure %s src=%s\n", N1, N2, N3,
                    arm == 1 ? "child" : "flat", pin ? "env" : "wisdom");
    }
    else
    {
        /* both arms built, the whole forward raced on scratch, the loser freed */
        const int ok1 = _ilnd_build_child(d, cfg);
        const int ok2 = _ilnd_build_flat(d, W, cfg, &key0);
        if (!ok1 && !ok2)
        {
            _vfft_warn("vfft_create: 3D INTERLEAVED c2c %dx%dx%d — neither structure arm "
                       "could be built (no 2D IL plan at %dx%d and no axis-1 chain)",
                       N1, N2, N3, N2, N3);
            vfft_ilnd_destroy(d);
            return NULL;
        }
        if (ok1 && ok2)
        {
            const size_t T = (size_t)N1 * d->plane;
            double *z = (double *)malloc(2 * T * sizeof(double));
            double ns[2] = { 1e300, 1e300 };
            if (z)
            {
                _ilnd_arm_ctx_t c1 = { d, z, 1 }, c2 = { d, z, 2 };
                const vfft_race_arm_t arms[2] = { { "child", _ilnd_arm_run, &c1 },
                                                  { "flat", _ilnd_arm_run, &c2 } };
                const vfft_race_proto_t proto = { 3, 1, VFFT_RACE_MIN, 1, 0, NULL, NULL };
                size_t i;
                for (i = 0; i < 2 * T; i++)
                    z[i] = 1.0 + 1e-6 * (double)(i & 1023);
                vfft_race_run(&proto, arms, 2, ns);
                free(z);
                arm = (ns[1] < ns[0]) ? 2 : 1;
            }
            else
                arm = 1;
            if (getenv("VFFT_IL2D_LOG"))
                fprintf(stderr, "[ilnd] %dx%dx%d: structure race child=%.0f flat=%.0f -> %s\n",
                        N1, N2, N3, ns[0], ns[1], arm == 1 ? "child" : "flat");
            _ilnd_free_arm(d, arm == 1 ? 2 : 1);
            if (W && !W->vw2_off_2d && cfg->wisdom_write)
            {
                if (vw2_ilnd_arm_bank(&W->vw2, &key0, arm))
                    _vw2_persist(W, cfg);
            }
        }
        else
            arm = ok1 ? 1 : 2;   /* one arm only: the verdict is the one that builds */
        d->arm = arm;
    }
    (void)banked_arm;
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
