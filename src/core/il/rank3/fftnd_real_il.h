/**
 * fftnd_real_il.h — the rank-3 INTERLEAVED REAL tier (design of record:
 * docs/roadmap/fft3d_real_il_design.md, decided with the owner 2026-10-07).
 * Phase 1: R2C, one plan per cell, out of place, natural order.
 *
 * A row-major real cube N1 x N2 x N3 (N3 contiguous) transforms to the CCE
 * volume N1 x N2 x hp3, hp3 = N3/2 + 1, interleaved. The real pass must come
 * FIRST, so the walk is fixed: PLANES FIRST, then axis 0 over the virtual
 * plane of N1 rows x (N2 hp3) complex. Made of two tiers' pieces: the 2D
 * real door (il/rank2, the plane) and the column-axis pass (il2d_col.h,
 * axis 0 and the pay-once form's axis 1).
 *
 * THE STRUCTURE IS A RACED ARM (s=), and so is axis 0's execution form (nf=):
 *   child    (s=1) the 2D real plan on (N2, N3) per plane, out of place into
 *            the caller's output (every rank-2 real verdict, raced on its own
 *            cell under the plane_ child store), then axis 0 in place;
 *   pay-once (s=2) the child's row engine per plane into a PRIVATE CCE
 *            volume, the PLAIN axis-1 chain in place there (no natural leaf),
 *            then axis 0 from the private volume with its one permuting write
 *            landing planes AND row blocks at their natural places: the order
 *            tax paid once, at the last pass's own write.
 *   nf=1     axis 0 IN PLACE: a one-stage chain's kernel (natural-native), or
 *            the natural pass (il2d_cols.h) through a cube-sized pre-leaf
 *            scratch at two stages or more; a Bluestein axis runs here only;
 *            the pay-once twin: the pre-leaf stages in place on the private
 *            volume and the LEAF out of place per row block;
 *   nf=2     axis 0 by DENSE COLUMN STRIPS (the c2c tier's strip form): N1 x
 *            nsw complex gathered into a strip scratch, the plain chain there,
 *            rows scattered to natural planes (the pay-once twin scatters row
 *            blocks to natural rows too; its strips stay inside a row block).
 * The four arms run the whole forward on scratch at create (min of 3 paced
 * rounds); the winner banks s= nf= nsw= on the rank-3 real row beside axis
 * 0's chain tokens (chain= blu= forms=) and the pay-once axis-1 chain
 * (chain1=); the loser's volume, scratch and descriptors are freed.
 * VFFT_ILNDR_ARM=1|2, VFFT_ILNDR_NF=1|2, VFFT_ILNDR_SW=w pin, never bank.
 *
 * WISDOM (owner 2026-10-07): the create MAY READ the 2D shard — when the
 * rank-3 row carries no plane_ recipe, the child store is seeded from the
 * 2D real row of (N2, N3), so a plane cell the 2D tier calibrated is not
 * re-raced here — and NEVER WRITES it: the whole plan, the child's recipe
 * included, lands on the rank-3 row of wisdom2_3d.txt.
 *
 * Contracts: R2C, rank 3, howmany 1, OUT OF PLACE, DEFAULT/NATURAL order,
 * every dim >= 2 (odd N3 through the row plan's odd door; a prime N1 serves
 * the child arm in place: a Bluestein axis has no plain chain). C2R and the
 * threaded forms are the next phases. Everything else refuses loudly.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: after fftnd_il.h (whose
 * _vfft_create_rank34_il dispatches here through a forward declaration) and
 * the 2D real plan headers, before vfft_execute.h.
 */
#ifndef VFFT_IL_RANK3_FFTND_REAL_IL_H
#define VFFT_IL_RANK3_FFTND_REAL_IL_H

typedef struct vfft_ilndr_s {
    int N[3], hp3;
    size_t plane;                 /* complex per plane = N2 * hp3: the virtual row */
    int arm;                      /* s=: 1 child | 2 pay-once */
    int nf;                       /* nf=: 1 in place | 2 strips */
    int nsw;                      /* nsw=: the strip width in columns */
    vfft_ilcol_t ax0;             /* axis 0: the plain chain (chain= blu= forms=) */
    vfft_ilcol_t ax0n;            /* axis 0 natural in place (ax0n_on): the natural pass at nst >= 2 or a Bluestein axis */
    int ax0n_on;
    vfft_ilcol_t ax1;             /* pay-once: axis 1 plain per plane (chain1=); ax1_on */
    int ax1_on;
    int *nat0, *nat1;             /* scr row -> natural row (the identity for a one-stage chain) */
    struct vfft_plan_s *cf;       /* the 2D real r2c child on (N2, N3), one thread */
    struct vfft_wisdom_s *childS; /* its private store: the recipe rides on the row as plane_* */
    double *V;                    /* the pay-once private CCE volume */
    double *sscr;                 /* the strip scratch, N1 x nsw complex */
    int mt_t;                     /* the plan's thread snapshot (phase 3 threads it) */
} vfft_ilndr_t;

/* a chain's natural permutation; a one-stage chain is natural-native: the identity */
static int *_ilndr_nat_perm(const int *R, int nst, int N)
{
    int *p, j;
    if (nst >= 2)
        return _il2d_nat_perm(R, nst, N);
    p = (int *)malloc((size_t)N * sizeof(int));
    if (p)
        for (j = 0; j < N; j++) p[j] = j;
    return p;
}

/* ═══ the passes ═══════════════════════════════════════════════════════ */
/* the child arm's planes: the 2D real plan, real plane -> CCE plane in dst */
static void _ilndr_planes_child(const vfft_ilndr_t *d, const double *x, double *dst)
{
    const size_t rp = (size_t)d->N[1] * (size_t)d->N[2];
    int q;
    for (q = 0; q < d->N[0]; q++)
        vfft_execute((vfft_plan)d->cf, VFFT_FORWARD, (double *)(x + (size_t)q * rp), NULL,
                     dst + 2 * (size_t)q * d->plane, NULL);
}
/* the pay-once planes: the child's row engine, then the plain axis-1 chain in place */
static void _ilndr_planes_payonce(const vfft_ilndr_t *d, const double *x, double *V)
{
    const size_t rp = (size_t)d->N[1] * (size_t)d->N[2];
    const vfft_ilcol_t *a = &d->ax1;
    int q;
    for (q = 0; q < d->N[0]; q++)
    {
        double *pl = V + 2 * (size_t)q * d->plane;
        _il2d_real_rows_fwd(d->cf, x + (size_t)q * rp, pl);
        _il2d_col_stages(pl, pl, d->N[1], (size_t)d->hp3, 0, a->nst, a->R, a->L, a->f, a->tf, 0);
    }
}
/* one strip: src columns [c0, c0+w) of every plane -> the plain chain on the strip scratch ->
 * row q to plane nat0[q] at the output columns [o0, o0+w) */
static void _ilndr_strip(const vfft_ilndr_t *d, const double *src, double *dst, size_t c0, size_t o0, size_t w)
{
    const vfft_ilcol_t *a = &d->ax0;
    const size_t P = d->plane;
    int q;
    for (q = 0; q < d->N[0]; q++)
        memcpy(d->sscr + 2 * (size_t)q * w, src + 2 * ((size_t)q * P + c0), 2 * w * sizeof(double));
    _il2d_col_stages(d->sscr, d->sscr, d->N[0], w, 0, a->nst, a->R, a->L, a->f, a->tf, 0);
    for (q = 0; q < d->N[0]; q++)
        memcpy(dst + 2 * ((size_t)d->nat0[q] * P + o0), d->sscr + 2 * (size_t)q * w, 2 * w * sizeof(double));
}
static void _ilndr_axis0_strips(const vfft_ilndr_t *d, const double *src, double *dst)
{
    const size_t W = (size_t)d->nsw;
    size_t c0;
    for (c0 = 0; c0 < d->plane; c0 += W)
    {
        const size_t w = (c0 + W <= d->plane) ? W : d->plane - c0;
        _ilndr_strip(d, src, dst, c0, c0, w);
    }
}
/* the pay-once strips stay inside a row block of hp3 columns: block b (a scrambled row) lands on row nat1[b] */
static void _ilndr_axis0_strips_blocks(const vfft_ilndr_t *d, const double *src, double *dst)
{
    const size_t W = (size_t)d->nsw, hp3 = (size_t)d->hp3;
    int b;
    for (b = 0; b < d->N[1]; b++)
    {
        const size_t ib = (size_t)b * hp3, ob = (size_t)d->nat1[b] * hp3;
        size_t c0;
        for (c0 = 0; c0 < hp3; c0 += W)
        {
            const size_t w = (c0 + W <= hp3) ? W : hp3 - c0;
            _ilndr_strip(d, src, dst, ib + c0, ob + c0, w);
        }
    }
}
/* the pay-once in-place form: the pre-leaf stages in place on V, the leaf V -> z per row block
 * with both permutations at its write (the leaf's legs are consecutive rows: group g = rows
 * g Rl .. g Rl + Rl - 1 -> natural rows nat0[g Rl] + r N1/Rl, block-affine) */
static void _ilndr_axis0_leafoop(const vfft_ilndr_t *d, double *V, double *z)
{
    const vfft_ilcol_t *a = &d->ax0;
    const size_t P = d->plane, hp3 = (size_t)d->hp3;
    const int nst = a->nst, Rl = a->R[nst - 1], G = d->N[0] / Rl;
    vfft_il2p_fn fn = a->f[nst - 1];
    int b, g;
    if (nst > 1)
        _il2d_col_stages(V, V, d->N[0], P, 0, nst - 1, a->R, a->L, a->f, a->tf, 0);
    for (b = 0; b < d->N[1]; b++)
    {
        const double *in = V + 2 * (size_t)b * hp3;
        double *out = z + 2 * (size_t)d->nat1[b] * hp3;
        for (g = 0; g < G; g++)
            fn(in + 2 * (size_t)g * Rl * P, NULL, out + 2 * (size_t)d->nat0[g * Rl] * P, NULL, NULL, NULL,
               P, 0, (size_t)G * P, 0, hp3);
    }
}
/* the child arm's axis 0 in place: the natural descriptor where one was built, else the plain
 * one-stage kernel (natural-native) */
static void _ilndr_axis0_inplace(const vfft_ilndr_t *d, double *z)
{
    if (d->ax0n_on)
        _il2d_col_exec(&d->ax0n, z, z, 0);
    else
        _il2d_col_exec(&d->ax0, z, z, 0);
}
/* THE SERIAL WALK, by the arm and the form: real cube x -> CCE volume z */
static void _ilndr_execute_st(const vfft_ilndr_t *d, const double *x, double *z)
{
    if (d->arm == 2)
    {
        _ilndr_planes_payonce(d, x, d->V);
        if (d->nf == 2)
            _ilndr_axis0_strips_blocks(d, d->V, z);
        else
            _ilndr_axis0_leafoop(d, d->V, z);
        return;
    }
    _ilndr_planes_child(d, x, z);
    if (d->nf == 2)
        _ilndr_axis0_strips(d, z, z);
    else
        _ilndr_axis0_inplace(d, z);
}
static void vfft_ilndr_execute(const vfft_ilndr_t *d, vfft_dir_t dir, const double *sre, double *dre)
{
    (void)dir;   /* an R2C plan runs forward */
    _ilndr_execute_st(d, sre, dre);
}

/* ═══ destroy ══════════════════════════════════════════════════════════ */
static void vfft_ilndr_destroy(vfft_ilndr_t *d)
{
    if (!d)
        return;
    if (d->cf)
        vfft_destroy((vfft_plan)d->cf);
    vfft_child_store_free(d->childS);
    _il2d_col_free(&d->ax0);
    if (d->ax0n_on)
        _il2d_col_free(&d->ax0n);
    if (d->ax1_on)
        _il2d_col_free(&d->ax1);
    free(d->nat0);
    free(d->nat1);
    vfft_aligned_free(d->V);
    vfft_aligned_free(d->sscr);
    free(d);
}

/* ═══ wisdom: the 2D-shard borrow (owner 2026-10-07) ═══════════════════
 * The child store is seeded from the parent row's plane_ tokens; when the row
 * carries none, from the 2D shard's real row of (N2, N3) at the plan's T — the
 * row's payload tokens copied whole (its own children ride on it), banked into
 * the private store as a seed (not raced). The 2D shard is never written. */
static void _ilndr_borrow_2d(struct vfft_wisdom_s *W, struct vfft_wisdom_s *S, int N2, int N3, int T)
{
    vw2_ilcol_key_t ck;
    vw2_key_t k;
    const vw2_rec_t *row;
    vw2_rec_t r;
    int i, ok = 1;
    if (!W || !S)
        return;
    memset(&ck, 0, sizeof ck);
    ck.rank = 2; ck.n0 = N2; ck.n1 = N3; ck.n2 = 0; ck.ord = VW2_ORD_NAT; ck.axis = 0; ck.real = 1; ck.nthreads = T;
    vw2__ilcol_key(&ck, &k);
    row = vw2_lookup(&W->vw2, &k);
    if (!row)
    {
        if (getenv("VFFT_IL2D_LOG"))
        {
            char kb[256];
            vw2__key_format(&k, kb, sizeof kb);
            fprintf(stderr, "[ilndr] no 2D row to borrow for the plane cell %dx%d (looked for %s)\n", N2, N3, kb);
        }
        return;
    }
    memset(&r, 0, sizeof r);
    r.key = row->key;
    for (i = 0; i < row->ntok && ok; i++)
        if (row->tok[i].sect == 1 && vw2_rec_set(&r, 1, row->tok[i].name, row->tok[i].val) != VW2_OK)
            ok = 0;
    if (!ok || vw2_bank(&S->vw2, &r) != VW2_OK)
    {
        vw2_rec_free(&r);
        return;
    }
    vw2_disown(&S->vw2);   /* seeded, not raced */
    if (getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[ilndr] the plane cell %dx%d seeded from the 2D shard\n", N2, N3);
}

/* ═══ the race ═════════════════════════════════════════════════════════ */
typedef struct { vfft_ilndr_t *d; const double *x; double *z; int arm, nf; char name[24]; } _ilndr_arm_ctx_t;
static void _ilndr_arm_run(void *v)
{
    _ilndr_arm_ctx_t *c = (_ilndr_arm_ctx_t *)v;
    c->d->arm = c->arm;
    c->d->nf = c->nf;
    _ilndr_execute_st(c->d, c->x, c->z);
}

/* ═══ create ═══════════════════════════════════════════════════════════ */
static vfft_plan _vfft_create_fftnd_real_il(const vfft_config_t *cfg, struct vfft_wisdom_s *W, size_t K)
{
    const int N1 = cfg->n[0], N2 = cfg->n[1], N3 = cfg->n[2];
    const int nthr = _vfft_plan_threads(cfg);
    const int usable_w = (W && !W->vw2_off_2d);
    const char *log = getenv("VFFT_IL2D_LOG");
    const char *apin = getenv("VFFT_ILNDR_ARM"), *fpin = getenv("VFFT_ILNDR_NF"), *wpin = getenv("VFFT_ILNDR_SW");
    vfft_ilndr_t *d;
    struct vfft_plan_s *h;
    vw2_ilcol_key_t key0, key1;
    vw2_key_t pk;
    char forms0[64], forms1[64];
    int bwl, btf, bro, bcmt, bcmtt, bblu;
    int payonce_ok, strips_ok, arm = 0, nf = 0, nsw = 0, raced = 0;
    if (!vfft_policy_ilndr_ok(cfg, K))
    {
        _vfft_warn("vfft_create: 3D INTERLEAVED real serves R2C, howmany==1, out of place, "
                   "DEFAULT/NATURAL order, every dim >= 2 (got %s, howmany=%zu, %dx%dx%d); C2R and "
                   "the threaded forms are the tier's next phases",
                   _vfft_tname(cfg->transform), K, N1, N2, N3);
        return NULL;
    }
    d = (vfft_ilndr_t *)calloc(1, sizeof *d);
    if (!d)
        return NULL;
    d->N[0] = N1; d->N[1] = N2; d->N[2] = N3;
    d->hp3 = N3 / 2 + 1;
    d->plane = (size_t)N2 * (size_t)d->hp3;
    d->mt_t = nthr;
    _il2d_blu_ctx.W = W;
    _il2d_blu_ctx.cfg = cfg;
    _il2d_blu_chain_hook = _il2d_blu_m_chain;
    memset(&key0, 0, sizeof key0);
    key0.rank = 3; key0.n0 = N1; key0.n1 = N2; key0.n2 = N3;
    key0.ord = VW2_ORD_NAT; key0.axis = 0; key0.real = 1; key0.nthreads = nthr;
    key1 = key0; key1.axis = 1;
    vw2__ilcol_key(&key0, &pk);
    /* ── the plane child: its store from the row's plane_ recipe, else from the 2D shard ── */
    d->childS = vfft_child_store_for(usable_w ? &W->vw2 : NULL, &pk, "plane_");
    if (!d->childS)
    {
        vfft_ilndr_destroy(d);
        return NULL;
    }
    if (usable_w && d->childS->vw2.nrec == 0)
        _ilndr_borrow_2d(W, d->childS, N2, N3, nthr);
    {
        vfft_config_t cc;
        memset(&cc, 0, sizeof cc);
        cc.transform = VFFT_R2C;
        cc.placement = VFFT_OUTOFPLACE;
        cc.rigor = cfg->rigor;
        cc.dims = 2;
        cc.n[0] = N2;
        cc.n[1] = N3;
        cc.howmany = 1;
        cc.order = VFFT_ORDER_NATURAL;
        cc.layout = VFFT_LAYOUT_INTERLEAVED;
        cc.nthreads = 1;
        cc.wisdom = (vfft_wisdom *)d->childS;
        cc.wisdom_write = 0;
        cc.recalibrate = cfg->recalibrate;
        d->cf = (struct vfft_plan_s *)vfft_create(&cc);
        if (!d->cf)
        {
            _vfft_warn("vfft_create: 3D INTERLEAVED r2c %dx%dx%d: no 2D real plane plan at %dx%d", N1, N2, N3, N2, N3);
            vfft_ilndr_destroy(d);
            return NULL;
        }
    }
    /* ── axis 0: the plain chain (raced over the pool on a miss; creates the row) ── */
    forms0[0] = 0;
    if (!_il2d_col_build(W, cfg, &key0, N1, d->plane, 0, &d->ax0, forms0, sizeof forms0,
                         &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu))
    {
        vfft_ilndr_destroy(d);
        return NULL;
    }
    d->nat0 = _ilndr_nat_perm(d->ax0.R, d->ax0.nst, N1);
    strips_ok = !d->ax0.blu && !d->ax0.tpc && d->nat0 != NULL;
    /* the in-place natural twin: the natural pass at two stages or more (a cube-sized pre-leaf
     * scratch), or a Bluestein axis; a one-stage chain is natural-native in place as it is */
    if (d->ax0.blu || d->ax0.tpc || d->ax0.nst >= 2)
    {
        char fb[64];
        fb[0] = 0;
        if (_il2d_col_build(W, cfg, &key0, N1, d->plane, 1, &d->ax0n, fb, sizeof fb,
                            &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu))
            d->ax0n_on = 1;
        else if (!strips_ok)
        {
            vfft_ilndr_destroy(d);
            return NULL;
        }
    }
    /* ── the pay-once pieces: axis 1's plain chain per plane (chain1=), the private volume ── */
    payonce_ok = strips_ok;
    if (payonce_ok)
    {
        forms1[0] = 0;
        if (_il2d_col_build(W, cfg, &key1, N2, (size_t)d->hp3, 0, &d->ax1, forms1, sizeof forms1,
                            &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu) && !d->ax1.blu && !d->ax1.tpc)
        {
            d->ax1_on = 1;
            d->nat1 = _ilndr_nat_perm(d->ax1.R, d->ax1.nst, N2);
            d->V = (double *)vfft_aligned_alloc((2 * (size_t)N1 * d->plane + 8) * sizeof(double));
            if (!d->nat1 || !d->V)
                payonce_ok = 0;
        }
        else
        {
            if (d->ax1.nst || d->ax1.blu)
                _il2d_col_free(&d->ax1);
            memset(&d->ax1, 0, sizeof d->ax1);
            payonce_ok = 0;
        }
    }
    /* ── the strip scratch ── */
    nsw = wpin ? atoi(wpin) : 0;
    if (nsw <= 0 && usable_w)
        nsw = vw2_ilnd_int_lookup(&W->vw2, &key0, "nsw");
    if (nsw <= 0)
        nsw = vfft_policy_ilndr_strip_w(N1, d->plane);
    if (strips_ok)
    {
        d->nsw = nsw;
        d->sscr = (double *)vfft_aligned_alloc((2 * (size_t)N1 * (size_t)nsw + 8) * sizeof(double));
        if (!d->sscr)
            strips_ok = 0;
    }
    /* ── the verdict: pin > banked > the race of (structure x form) ── */
    if (apin && apin[0])
        arm = atoi(apin);
    if (fpin && fpin[0])
        nf = atoi(fpin);
    if ((!arm || !nf) && usable_w && !cfg->recalibrate)
    {
        if (!arm) arm = vw2_ilnd_arm_lookup(&W->vw2, &key0);
        if (!nf) nf = vw2_ilnd_int_lookup(&W->vw2, &key0, "nf");
    }
    if (arm == 2 && !payonce_ok) arm = 0;
    if (nf == 2 && !strips_ok) nf = 0;
    if (nf == 1 && d->arm == 2 && !strips_ok) nf = 0;
    if (!arm || !nf)
    {
        const size_t RN = (size_t)N1 * (size_t)N2 * (size_t)N3, CN = 2 * (size_t)N1 * d->plane;
        double *x = (double *)vfft_aligned_alloc((RN + 8) * sizeof(double));
        double *z = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
        _ilndr_arm_ctx_t ac[4];
        vfft_race_arm_t arms[4];
        double ns[4];
        int na = 0, a, best = 0, reps;
        if (!x || !z)
        {
            vfft_aligned_free(x); vfft_aligned_free(z);
            vfft_ilndr_destroy(d);
            return NULL;
        }
        {
            unsigned sd = 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 10) ^ ((unsigned)N3 << 20);
            size_t j;
            for (j = 0; j < RN; j++)
            {
                sd = sd * 1664525u + 1013904223u;
                x[j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
            }
        }
        memset(z, 0, (CN + 8) * sizeof(double));
#define _ILNDR_ARM(A_, F_, NAME_)                                                   \
        do {                                                                        \
            if ((!arm || arm == (A_)) && (!nf || nf == (F_)))                       \
            {                                                                       \
                ac[na].d = d; ac[na].x = x; ac[na].z = z; ac[na].arm = (A_); ac[na].nf = (F_); \
                snprintf(ac[na].name, sizeof ac[na].name, "%s", NAME_);             \
                arms[na].name = ac[na].name; arms[na].run = _ilndr_arm_run; arms[na].ctx = &ac[na]; \
                na++;                                                               \
            }                                                                       \
        } while (0)
        _ILNDR_ARM(1, 1, "child/inplace");
        if (strips_ok) _ILNDR_ARM(1, 2, "child/strips");
        if (payonce_ok) _ILNDR_ARM(2, 1, "payonce/leafoop");
        if (payonce_ok && strips_ok) _ILNDR_ARM(2, 2, "payonce/strips");
#undef _ILNDR_ARM
        if (na == 0)
        {
            vfft_aligned_free(x); vfft_aligned_free(z);
            vfft_ilndr_destroy(d);
            return NULL;
        }
        reps = (int)(1e6 / (double)(N1 * d->plane + 1));
        if (reps < 1) reps = 1;
        if (reps > 64) reps = 64;
        if (na > 1)
        {
            const vfft_race_proto_t proto = { 3, reps, VFFT_RACE_MIN, 1, 0, NULL, NULL, 1 }; /* single-thread arms: paced */
            _vfft_create_race_count++;
            vfft_race_run(&proto, arms, na, ns);
            for (a = 1; a < na; a++)
                if (ns[a] < ns[best])
                    best = a;
            raced = 1;
            if (log)
            {
                fprintf(stderr, "[ilndr] %dx%dx%d r2c race:", N1, N2, N3);
                for (a = 0; a < na; a++)
                    fprintf(stderr, " %s=%.0f", ac[a].name, ns[a]);
                fprintf(stderr, " -> %s\n", ac[best].name);
            }
        }
        arm = ac[best].arm;
        nf = ac[best].nf;
        vfft_aligned_free(x);
        vfft_aligned_free(z);
    }
    d->arm = arm;
    d->nf = nf;
    /* the loser's resources go */
    if (arm == 1)
    {
        vfft_aligned_free(d->V); d->V = NULL;
        if (d->ax1_on) { _il2d_col_free(&d->ax1); d->ax1_on = 0; }
        free(d->nat1); d->nat1 = NULL;
    }
    if (nf == 1)
    {
        vfft_aligned_free(d->sscr); d->sscr = NULL;
    }
    if ((nf == 2 || arm == 2) && d->ax0n_on)
    {
        _il2d_col_free(&d->ax0n); d->ax0n_on = 0;
    }
    /* bank what was RACED (pins never bank) */
    if (usable_w && raced && !(apin && apin[0]) && !(fpin && fpin[0]))
    {
        int banked = 0;
        if (vw2_ilcol_row_ensure(&W->vw2, &key0, d->ax0.R, d->ax0.nst)) banked = 1;
        if (vw2_ilnd_arm_bank(&W->vw2, &key0, arm)) banked = 1;
        if (vw2_ilnd_int_bank(&W->vw2, &key0, "nf", nf)) banked = 1;
        if (nf == 2 && !wpin && vw2_ilnd_int_bank(&W->vw2, &key0, "nsw", d->nsw)) banked = 1;
        if (vw2_ilcol_forms_rebank(&W->vw2, &key0, forms0)) banked = 1;
        if (banked)
            _vw2_persist(W, cfg);
    }
    /* ── the handle ── */
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_ilndr_destroy(d);
        return NULL;
    }
    h->transform = cfg->transform;
    h->placement = cfg->placement;
    h->layout = (int)cfg->layout;
    h->N = N1;
    h->N2 = N2;
    h->N3 = N3;
    h->K = 1;
    h->nthreads = nthr;
    h->ilndr = d;
    if (usable_w)
    {   /* the child's recipe onto the cell's row (raced here, or the row did not carry it yet) */
        if (vfft_child_row_update(&W->vw2, &pk, "plane_", d->childS))
            _vw2_persist(W, cfg);
    }
    if (log)
        fprintf(stderr, "[ilndr] %dx%dx%d r2c: %s %s (axis 0 chain %d.. %s, plane rows %s)\n", N1, N2, N3,
                arm == 2 ? "pay-once" : "child", nf == 2 ? "strips" : "in place",
                d->ax0.nst ? d->ax0.R[0] : 0, d->ax0.blu ? "Bluestein" : "",
                d->cf->il2d_rx_on ? "engine" : "route");
    return (vfft_plan)h;
}

#endif /* VFFT_IL_RANK3_FFTND_REAL_IL_H */
