/**
 * fftnd_real_il.h — the rank-3 INTERLEAVED REAL tier (design of record:
 * docs/roadmap/fft3d_real_il_design.md, decided with the owner 2026-10-07).
 * Phase 1: R2C; phase 2: C2R (the same cell's twin). One plan per cell and
 * direction, out of place, natural order.
 *
 * A row-major real cube N1 x N2 x N3 (N3 contiguous) transforms to the CCE
 * volume N1 x N2 x hp3, hp3 = N3/2 + 1, interleaved. The real pass must come
 * FIRST in r2c and LAST in c2r, so each walk is fixed: r2c = PLANES FIRST,
 * then axis 0 over the virtual plane of N1 rows x (N2 hp3) complex; c2r =
 * AXIS 0 FIRST, then the planes. Made of two tiers' pieces: the 2D real door
 * (il/rank2, the plane) and the column-axis pass (il2d_col.h, axis 0 and the
 * pay-once form's axis 1).
 *
 * THE STRUCTURE IS A RACED ARM (s=), and so is axis 0's execution form (nf=):
 *   child    (s=1) the 2D real plan on (N2, N3) per plane (every rank-2 real
 *            verdict, raced on its own cell under the plane_ child store);
 *   pay-once (s=2) r2c: the child's row engine per plane into a PRIVATE CCE
 *            volume, the PLAIN axis-1 chain in place there (no natural leaf),
 *            then axis 0 from the private volume with its one permuting write
 *            landing planes AND row blocks at their natural places; c2r: the
 *            plain forward chain on axis 1 in place (the rows come out
 *            scrambled) and the backward rows written per LEAF GROUP to their
 *            natural rows (block-affine) — the order tax paid once, at the
 *            last pass's own write;
 *   band     (s=3, c2r only) axis 0's wide prefix stages out of place into the
 *            private volume, then per band of L[cut] planes the suffix stages
 *            in place and the band's planes' c2r (the child) while hot; every
 *            legal cut is an arm (wl_c2r= the band width in planes);
 *   nf=1     axis 0 IN PLACE at the cube pitch: r2c a one-stage chain's kernel
 *            (natural-native) or the natural pass through a cube-sized
 *            pre-leaf scratch (a Bluestein axis runs here only), the pay-once
 *            twin's pre-leaf stages in place and its LEAF out of place per row
 *            block; c2r the forward chain's stage 0 out of place and the rest
 *            in place on the private volume;
 *   nf=2     axis 0 by DENSE COLUMN STRIPS (the c2c tier's strip form): N1 x
 *            nsw complex gathered into a strip scratch, the plain chain there,
 *            the rows scattered back (r2c to natural planes, the pay-once twin
 *            to natural row blocks too; c2r in chain order).
 * The arms run the whole transform on scratch at create (min of 3..15 paced
 * rounds, at least 48 timed executes per arm); the winner banks s= nf= nsw=
 * (r2c) or s_c2r= nf_c2r= nsw_c2r= wl_c2r= (c2r) on the rank-3 real row
 * beside axis 0's chain tokens (chain= blu= forms=) and the pay-once axis-1
 * chain (chain1=), both direction-shared; the loser's volume, scratch and
 * descriptors are freed. VFFT_ILNDR_ARM=1|2|3, VFFT_ILNDR_NF=1|2,
 * VFFT_ILNDR_SW=w, VFFT_ILNDR_WL=w pin, never bank.
 *
 * C2R AND THE INPUT (owner 2026-10-07): the caller's CCE volume is preserved
 * — axis 0 runs out of place into a private volume the input's size; under
 * the destroy_input permission it runs in place in the caller's volume and
 * nothing is allocated. c2r pays no order tax at all: the axis-0 pass is the
 * forward chain (bwd[n] = fwd[(N - n) mod N]), so position q holds the real
 * plane (N1 - nat[q]) mod N1 and the planes' own out-of-place writes land
 * it; the same identity on axis 1 in the pay-once form.
 *
 * WISDOM (owner 2026-10-07): the create MAY READ the 2D shard — when the
 * rank-3 row carries no plane_ recipe, the child store is seeded from the
 * 2D real row of (N2, N3), so a plane cell the 2D tier calibrated is not
 * re-raced here — and NEVER WRITES it: the whole plan, the child's recipe
 * included, lands on the rank-3 row of wisdom2_3d.txt.
 *
 * Contracts: R2C / C2R, rank 3, howmany 1, OUT OF PLACE, DEFAULT/NATURAL
 * order, every dim >= 2 (odd N3 through the row plan's odd door; a prime N1
 * serves the child arm in place: a Bluestein axis has no plain chain). The
 * threaded forms are phase 3; a threaded request is served by the serial
 * walk until then. Everything else refuses loudly.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: after fftnd_il.h (whose
 * _vfft_create_rank34_il dispatches here through a forward declaration) and
 * the 2D real plan headers, before vfft_execute.h.
 */
#ifndef VFFT_IL_RANK3_FFTND_REAL_IL_H
#define VFFT_IL_RANK3_FFTND_REAL_IL_H

typedef struct vfft_ilndr_s {
    int N[3], hp3;                /* hp3 = the CCE row's pitch in complex: N3/2 + 1, or door 2's policy pitch */
    int hpn;                      /* the bins per row, N3/2 + 1 */
    size_t plane;                 /* complex per plane = N2 * hp3: the virtual row */
    int c2r;                      /* the plan's direction */
    int destroy;                  /* c2r: axis 0 in place in the caller's volume (destroy_input) */
    int ip;                       /* IN PLACE (real_inplace_design.md, 2026-10-08): one volume, every row padded
                                   * to 2 hp3 doubles, in == out; c2r destroys by nature. 1 = the caller's volume at
                                   * the bin pitch (door 1), 2 = the plan's own volume off it (door 2: the cell's
                                   * pitch twin, its own _p verdicts) */
    int arm;                      /* s=: 1 child | 2 pay-once | 3 band (c2r) */
    int nf;                       /* nf=: 1 in place | 2 strips */
    int nsw;                      /* nsw=: the strip width in columns */
    int cut, wl;                  /* the band arm: prefix stages [0, cut), band width L[cut] planes */
    vfft_ilcol_t ax0;             /* axis 0: the plain chain (chain= blu= forms=) */
    vfft_ilcol_t ax0n;            /* r2c axis 0 natural in place (ax0n_on): the natural pass at nst >= 2 or a Bluestein axis */
    int ax0n_on;
    vfft_ilcol_t ax1;             /* pay-once: axis 1 plain per plane (chain1=); ax1_on */
    int ax1_on;
    int *nat0, *nat1;             /* scr row -> natural row (the identity for a one-stage chain) */
    int *neg0, *neg1;             /* c2r: position -> the real index of the forward chain's bin, (N - nat) mod N */
    int *pos0;                    /* c2r in place: the inverse of neg0, the position holding real plane r */
    struct vfft_plan_s *cf;       /* r2c: the 2D real r2c child on (N2, N3), one thread */
    struct vfft_plan_s *cb;       /* c2r: the 2D real c2r child */
    struct vfft_wisdom_s *childS; /* their private store: the recipe rides on the row as plane_* */
    double *V;                    /* the private CCE volume (pay-once r2c; every c2r arm unless destroy) */
    double *sscr;                 /* the strip scratch, N1 x nsw complex */
    double *pbuf[3];              /* c2r in place: [0] the cycle's first plane, [2] the tight real plane the
                                   * out-of-place 2D child writes (the child arm), [1] door 2: a source plane
                                   * compacted to the bin pitch for that child */
    char *vis;                    /* the walk's visited flags, N1 */
    int *cycs, *cycl, ncyc;       /* c2r in place: the cycles of neg0 (starts, lengths, by length descending) */
    int *seq, *cyc0;              /* the walk sequence S (every cycle from its start, S[i+1] = pos0[S[i]]) and each
                                   * cycle's first index in it */
    int *pc_a, *pc_n, *pc_next, *pc_w, npc, npc_max;   /* the threaded walk's PIECES of S: start, length, the next
                                   * piece in the cycle (whose copy the last plane reads), the worker; dealt per execute */
    double **pbw;                 /* the workers' buffers for the planes (3 per worker: as pbuf; [0] unused) */
    int npbw;
    double **pcb;                 /* the pieces' first-plane copies (npc_max planes) */
    int mt_t;                     /* the plan's thread snapshot */
    int prof;                     /* VFFT_ILNDR_PROF, bound at create: per execute the planes' and axis 0's ns on stderr */
    /* THE THREADED FORMS (phase 3, 2026-10-07): the plane arm transposed. mt = the verdict at the
     * plan's T (0 serial | 2 plane); mts/mtf/mtwl/mtcut = the structure, form and band the
     * threaded verdict runs with (they may differ from s=/nf=); ptw = the plane team's width
     * (0 = the full team); cw = the 2D real child's clones (worker t > 0 = slot t-1), sscrw the
     * workers' strip scratches. */
    int mt, mts, mtf, mtwl, mtcut, ptw;
    struct vfft_plan_s **cw;
    int ncw;
    double **sscrw;
    int nsscrw;
} vfft_ilndr_t;

/* the profile: phase times accumulated by the walks, printed per execute (the c2c tier's
 * VFFT_ILND_PROF shape); the band arm's suffix stages count as axis 0 */
typedef struct { double planes, axis0; } _ilndr_prof_t;
static _ilndr_prof_t _ilndr_prof;
static void _ilndr_prof_print(const vfft_ilndr_t *d)
{
    fprintf(stderr, "[ilndr prof] %dx%dx%d %s %s/%s: planes %.0f ns, axis 0 %.0f ns (%.1f%% / %.1f%%)\n",
            d->N[0], d->N[1], d->N[2], d->c2r ? "c2r" : "r2c",
            d->arm == 3 ? "band" : d->arm == 2 ? "payonce" : "child", d->nf == 2 ? "strips" : "inplace",
            _ilndr_prof.planes, _ilndr_prof.axis0,
            100.0 * _ilndr_prof.planes / (_ilndr_prof.planes + _ilndr_prof.axis0 + 1e-9),
            100.0 * _ilndr_prof.axis0 / (_ilndr_prof.planes + _ilndr_prof.axis0 + 1e-9));
}

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
/* the inverse of a permutation */
static int *_ilndr_inv_of(const int *p, int N)
{
    int *q = (int *)malloc((size_t)N * sizeof(int));
    int j;
    if (q)
        for (j = 0; j < N; j++) q[p[j]] = j;
    return q;
}
static int *_ilndr_neg_of(const int *nat, int N)
{
    int *p = (int *)malloc((size_t)N * sizeof(int));
    int j;
    if (p)
        for (j = 0; j < N; j++) p[j] = (N - nat[j]) % N;
    return p;
}

/* ═══ the r2c passes ═══════════════════════════════════════════════════ */
/* the child arm's planes: the 2D real plan, real plane -> CCE plane in dst */
static void _ilndr_planes_child_range(const vfft_ilndr_t *d, struct vfft_plan_s *cf, const double *x, double *dst,
                                      int q_lo, int q_hi)
{
    const size_t rp = d->ip ? 2 * d->plane : (size_t)d->N[1] * (size_t)d->N[2];   /* in place: the padded plane is where its CCE plane lands */
    int q;
    for (q = q_lo; q < q_hi; q++)
        vfft_execute((vfft_plan)cf, VFFT_FORWARD, (double *)(x + (size_t)q * rp), NULL,
                     dst + 2 * (size_t)q * d->plane, NULL);
}
static void _ilndr_planes_child(const vfft_ilndr_t *d, const double *x, double *dst)
{
    _ilndr_planes_child_range(d, d->cf, x, dst, 0, d->N[0]);
}
/* the pay-once planes: the child's row engine, then the plain axis-1 chain in place */
static void _ilndr_planes_payonce_range(const vfft_ilndr_t *d, struct vfft_plan_s *cf, const double *x, double *V,
                                        int q_lo, int q_hi)
{
    const size_t rp = (size_t)d->N[1] * (size_t)d->N[2];
    const vfft_ilcol_t *a = &d->ax1;
    int q;
    for (q = q_lo; q < q_hi; q++)
    {
        double *pl = V + 2 * (size_t)q * d->plane;
        if (d->ip)
        {   /* in place: the caller's padded plane moved into the private volume, the in-place row engine there */
            memcpy(pl, x + 2 * (size_t)q * d->plane, 2 * d->plane * sizeof(double));
            _il2d_real_rows_fwd(cf, pl, pl);
        }
        else
            _il2d_real_rows_fwd(cf, x + (size_t)q * rp, pl);
        _il2d_col_stages(pl, pl, d->N[1], (size_t)d->hp3, 0, a->nst, a->R, a->L, a->f, a->tf, 0);
    }
}
static void _ilndr_planes_payonce(const vfft_ilndr_t *d, const double *x, double *V)
{
    _ilndr_planes_payonce_range(d, d->cf, x, V, 0, d->N[0]);
}
/* one strip: src columns [c0, c0+w) of every plane -> the plain chain on the strip scratch ->
 * row q to plane perm[q] (NULL = q) at the output columns [o0, o0+w) */
static void _ilndr_strip(const vfft_ilndr_t *d, double *sscr, const double *src, double *dst, size_t c0, size_t o0,
                         size_t w, const int *perm)
{
    const vfft_ilcol_t *a = &d->ax0;
    const size_t P = d->plane;
    int q;
    for (q = 0; q < d->N[0]; q++)
        memcpy(sscr + 2 * (size_t)q * w, src + 2 * ((size_t)q * P + c0), 2 * w * sizeof(double));
    _il2d_col_stages(sscr, sscr, d->N[0], w, 0, a->nst, a->R, a->L, a->f, a->tf, 0);
    for (q = 0; q < d->N[0]; q++)
        memcpy(dst + 2 * ((size_t)(perm ? perm[q] : q) * P + o0), sscr + 2 * (size_t)q * w, 2 * w * sizeof(double));
}
/* the strips over the columns [k_lo, k_hi) of the virtual row (a worker's range) */
static void _ilndr_axis0_strips_range(const vfft_ilndr_t *d, double *sscr, const double *src, double *dst,
                                      const int *perm, size_t k_lo, size_t k_hi)
{
    const size_t W = (size_t)d->nsw;
    size_t c0;
    for (c0 = k_lo; c0 < k_hi; c0 += W)
    {
        const size_t w = (c0 + W <= k_hi) ? W : k_hi - c0;
        _ilndr_strip(d, sscr, src, dst, c0, c0, w, perm);
    }
}
static void _ilndr_axis0_strips(const vfft_ilndr_t *d, const double *src, double *dst, const int *perm)
{
    _ilndr_axis0_strips_range(d, d->sscr, src, dst, perm, 0, d->plane);
}
/* the pay-once strips stay inside a row block of hp3 columns: block b (a scrambled row) lands on row
 * nat1[b]; over the blocks [b_lo, b_hi) */
static void _ilndr_axis0_strips_blocks_range(const vfft_ilndr_t *d, double *sscr, const double *src, double *dst,
                                             int b_lo, int b_hi)
{
    const size_t W = (size_t)d->nsw, hp3 = (size_t)d->hp3;
    int b;
    for (b = b_lo; b < b_hi; b++)
    {
        const size_t ib = (size_t)b * hp3, ob = (size_t)d->nat1[b] * hp3;
        size_t c0;
        for (c0 = 0; c0 < hp3; c0 += W)
        {
            const size_t w = (c0 + W <= hp3) ? W : hp3 - c0;
            _ilndr_strip(d, sscr, src, dst, ib + c0, ob + c0, w, d->nat0);
        }
    }
}
static void _ilndr_axis0_strips_blocks(const vfft_ilndr_t *d, const double *src, double *dst)
{
    _ilndr_axis0_strips_blocks_range(d, d->sscr, src, dst, 0, d->N[1]);
}
/* the pay-once in-place form: the pre-leaf stages in place on V, the leaf V -> z per row block
 * with both permutations at its write (the leaf's legs are consecutive rows: group g = rows
 * g Rl .. g Rl + Rl - 1 -> natural rows nat0[g Rl] + r N1/Rl, block-affine) */
static void _ilndr_axis0_leafoop_range(const vfft_ilndr_t *d, double *V, double *z, int b_lo, int b_hi)
{
    const vfft_ilcol_t *a = &d->ax0;
    const size_t P = d->plane, hp3 = (size_t)d->hp3;
    const int nst = a->nst, Rl = a->R[nst - 1], G = d->N[0] / Rl;
    vfft_il2p_fn fn = a->f[nst - 1];
    int b, g;
    if (nst > 1)   /* the pre-leaf stages on this range's columns, in place */
        _il2d_col_stages2(V + 2 * (size_t)b_lo * hp3, V + 2 * (size_t)b_lo * hp3, d->N[0], P,
                          (size_t)(b_hi - b_lo) * hp3, 0, nst - 1, a->R, a->L, a->f, a->tf, 0);
    for (b = b_lo; b < b_hi; b++)
    {
        const double *in = V + 2 * (size_t)b * hp3;
        double *out = z + 2 * (size_t)d->nat1[b] * hp3;
        for (g = 0; g < G; g++)
            fn(in + 2 * (size_t)g * Rl * P, NULL, out + 2 * (size_t)d->nat0[g * Rl] * P, NULL, NULL, NULL,
               P, 0, (size_t)G * P, 0, hp3);
    }
}
static void _ilndr_axis0_leafoop(const vfft_ilndr_t *d, double *V, double *z)
{
    _ilndr_axis0_leafoop_range(d, V, z, 0, d->N[1]);
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
/* THE r2c SERIAL WALK, by the arm and the form: real cube x -> CCE volume z */
static void _ilndr_execute_r2c(const vfft_ilndr_t *d, const double *x, double *z)
{
    double t0 = d->prof ? vfft_now_ns() : 0.0, t1;
    if (d->arm == 2)
        _ilndr_planes_payonce(d, x, d->V);
    else
        _ilndr_planes_child(d, x, z);
    if (d->prof) { t1 = vfft_now_ns(); _ilndr_prof.planes = t1 - t0; t0 = t1; }
    if (d->arm == 2)
    {
        if (d->nf == 2)
            _ilndr_axis0_strips_blocks(d, d->V, z);
        else
            _ilndr_axis0_leafoop(d, d->V, z);
    }
    else if (d->nf == 2)
        _ilndr_axis0_strips(d, z, z, d->nat0);
    else
        _ilndr_axis0_inplace(d, z);
    if (d->prof) { _ilndr_prof.axis0 = vfft_now_ns() - t0; _ilndr_prof_print(d); }
}

/* ═══ the c2r passes ═══════════════════════════════════════════════════ */
/* the pay-once c2r rows of one plane: position p of the scrambled axis-1 plane holds forward bin
 * nat1[p], i.e. the real row (N2 - nat1[p]) mod N2. The chain's leaf makes nat1 BLOCK-AFFINE:
 * positions g Rl + r hold bins b0 + r S (b0 = nat1[g Rl], S = N2/Rl), so each leaf group's rows
 * are an arithmetic progression (N2 - b0) - r S and the rows kernel (two rows at least, a stride
 * per side) writes a whole group per call with a descending output stride; the group that wraps
 * at row 0 takes two calls. An engine per row or the door route goes row by row. */
static void _ilndr_rows_bwd_payonce(const vfft_ilndr_t *d, struct vfft_plan_s *h, const double *pl, double *yp)
{
    const size_t hp3 = (size_t)d->hp3;
    const size_t rp3 = d->ip ? 2 * hp3 : (size_t)d->N[2];   /* the real rows' pitch: padded in place */
    const int N2 = d->N[1], nst = d->ax1.nst, Rl = nst ? d->ax1.R[nst - 1] : N2, S = N2 / Rl;
    const size_t negS = (size_t)0 - (size_t)S * rp3;   /* the descending row stride, as the kernel's size_t */
    int g, r;
    if (!(h->il2d_rx_on && h->il2d_rx_lm))
    {
        for (r = 0; r < N2; r++)
            _il2d_rows_bwd_set(h, pl + 2 * (size_t)r * hp3, hp3, (size_t)d->neg1[r], 1, 1, yp, rp3);
        return;
    }
    for (g = 0; g < N2 / Rl; g++)
    {
        const int b0 = d->nat1[g * Rl];
        const double *in = pl + 2 * (size_t)g * Rl * hp3;
        if (b0 == 0)
        {   /* rows 0, N2-S, N2-2S, ...: positions {0,1} -> rows {0, N2-S}; then positions 1.. descending */
            h->il2d_rx_lm(in, NULL, yp, NULL, NULL, NULL, hp3, 0, (size_t)(N2 - S) * rp3, 0, 2);
            if (Rl >= 3)
                h->il2d_rx_lm(in + 2 * hp3, NULL, yp + (size_t)(N2 - S) * rp3, NULL, NULL, NULL, hp3, 0, negS, 0, (size_t)(Rl - 1));
        }
        else
            h->il2d_rx_lm(in, NULL, yp + (size_t)(N2 - b0) * rp3, NULL, NULL, NULL, hp3, 0, negS, 0, (size_t)Rl);
    }
}
/* the planes' c2r from the axis-0 output V (position q = real plane neg0[q]) into y */
static void _ilndr_c2r_planes_on(const vfft_ilndr_t *d, struct vfft_plan_s *cb, int arm, double *V, double *y,
                                 int q_lo, int q_hi)
{
    const size_t rp = (size_t)d->N[1] * (size_t)d->N[2];
    int q;
    for (q = q_lo; q < q_hi; q++)
    {
        double *pl = V + 2 * (size_t)q * d->plane, *yp = y + (size_t)d->neg0[q] * rp;
        if (arm == 2)
        {
            const vfft_ilcol_t *a = &d->ax1;
            _il2d_col_stages(pl, pl, d->N[1], (size_t)d->hp3, 0, a->nst, a->R, a->L, a->f, a->tf, 0);
            _ilndr_rows_bwd_payonce(d, cb, pl, yp);
        }
        else
            vfft_execute((vfft_plan)cb, VFFT_BACKWARD, pl, NULL, yp, NULL);
    }
}
static void _ilndr_c2r_planes(const vfft_ilndr_t *d, double *V, double *y, int q_lo, int q_hi)
{
    _ilndr_c2r_planes_on(d, d->cb, d->arm, V, y, q_lo, q_hi);
}
/* THE PLANES IN PLACE (real_inplace_design.md, 2026-10-08). V is the caller's volume: axis 0 ran in
 * place in it, so position q holds the CCE plane of real plane neg0[q]. The planes walk the cycles
 * of that permutation BACKWARDS through one buffer: the cycle's first plane is copied out, then
 * every plane is produced from the position that holds it (pos0, the inverse) straight into its
 * own place -- the child arm through the out-of-place 2D c2r child and the tight plane (its rows
 * landed at the padded pitch), the pay-once arm's axis-1 stages in place on the source position
 * and its rows landing directly. One plane copy per cycle, nothing else moved. The band's planes
 * land across bands: not an in-place arm. */
static void _ilndr_c2r_plane_out_w(const vfft_ilndr_t *d, struct vfft_plan_s *cb, double *const *pb, int arm,
                                   double *src, double *dstplane)
{
    const size_t N3 = (size_t)d->N[2], rp3 = 2 * (size_t)d->hp3;
    size_t r;
    if (arm == 2)
    {   /* pay-once: the plain axis-1 chain in place on the source, the backward rows to their natural rows */
        const vfft_ilcol_t *a = &d->ax1;
        _il2d_col_stages(src, src, d->N[1], (size_t)d->hp3, 0, a->nst, a->R, a->L, a->f, a->tf, 0);
        _ilndr_rows_bwd_payonce(d, cb, src, dstplane);
        return;
    }
    if (d->ip == 2)
    {   /* door 2: the out-of-place child reads its rows at the bin pitch -- the source compacted first */
        const size_t hpn = (size_t)d->hpn;
        for (r = 0; r < (size_t)d->N[1]; r++)
            memcpy(pb[1] + 2 * r * hpn, src + 2 * r * (size_t)d->hp3, 2 * hpn * sizeof(double));
        src = pb[1];
    }
    vfft_execute((vfft_plan)cb, VFFT_BACKWARD, src, NULL, pb[2], NULL);
    for (r = 0; r < (size_t)d->N[1]; r++)
        memcpy(dstplane + r * rp3, pb[2] + r * N3, N3 * sizeof(double));
}
/* one cycle of the position permutation, walked backwards from its first position q0 through the
 * buffer pb[0]: every plane of the cycle produced from the position holding it into its own place */
static void _ilndr_c2r_cycle_w(const vfft_ilndr_t *d, struct vfft_plan_s *cb, double *const *pb, int arm,
                               double *V, int q0)
{
    const size_t pn = 2 * d->plane;
    double *tmp = pb[0];
    int cur = q0;   /* the plane being produced */
    memcpy(tmp, V + (size_t)q0 * pn, pn * sizeof(double));   /* the cycle's first position, consumed last */
    for (;;)
    {
        const int pred = d->pos0[cur];   /* the position holding real plane cur */
        if (pred == q0)
        {
            _ilndr_c2r_plane_out_w(d, cb, pb, arm, tmp, V + (size_t)cur * pn);
            break;
        }
        _ilndr_c2r_plane_out_w(d, cb, pb, arm, V + (size_t)pred * pn, V + (size_t)cur * pn);
        cur = pred;
    }
}
/* the serial walk: every cycle on the primary child and the plan's buffers */
static void _ilndr_c2r_planes_ip(const vfft_ilndr_t *d, double *V)
{
    int i;
    for (i = 0; i < d->ncyc; i++)
        _ilndr_c2r_cycle_w(d, d->cb, d->pbuf, d->arm, V, d->cycs[i]);
}
/* THE THREADED WALK CUTS THE CYCLES (2026-10-08). The cycle deal left most workers idle (the
 * cycles are few and long: N1 = 16 under chain 4.4 has one of 10 planes). The walk sequence S --
 * every cycle from its start, S[i+1] = pos0[S[i]]: plane S[i] is produced from position S[i+1]
 * into position S[i], the cycle's last plane from its first position -- is cut into PIECES of at
 * most ceil(N1/T) planes, dealt to the workers. Phase A: every piece's first plane is copied into
 * the piece's own buffer (position S[a] is overwritten first thing by its piece, while the
 * previous piece's last plane still needs it). Barrier. Phase B: every piece in order, its last
 * plane produced from the NEXT piece's copy (cyclic within the cycle). The same production per
 * plane as the serial walk: MT == ST bitwise. One copy per piece, at most T + cycles. */
static void _ilndr_c2r_copies_w(const vfft_ilndr_t *d, int tid, double *V)
{
    const size_t pn = 2 * d->plane;
    int p;
    for (p = 0; p < d->npc; p++)
        if (d->pc_w[p] == tid)
            memcpy(d->pcb[p], V + (size_t)d->seq[d->pc_a[p]] * pn, pn * sizeof(double));
}
static void _ilndr_c2r_pieces_w(const vfft_ilndr_t *d, struct vfft_plan_s *cb, int tid, int arm, double *V)
{
    double *const *pb = tid > 0 ? (double *const *)(d->pbw + 3 * (tid - 1)) : (double *const *)d->pbuf;
    const size_t pn = 2 * d->plane;
    int p, i;
    for (p = 0; p < d->npc; p++)
    {
        if (d->pc_w[p] != tid)
            continue;
        for (i = d->pc_a[p]; i < d->pc_a[p] + d->pc_n[p]; i++)
        {
            const int cur = d->seq[i];
            double *src = (i == d->pc_a[p] + d->pc_n[p] - 1) ? d->pcb[d->pc_next[p]] : V + (size_t)d->seq[i + 1] * pn;
            _ilndr_c2r_plane_out_w(d, cb, pb, arm, src, V + (size_t)cur * pn);
        }
    }
}
/* the pieces for T workers: at most ceil(N1/T) planes each, every cycle cut into equal pieces,
 * dealt to the least-loaded worker; 0 = more pieces than buffers (cannot engage) */
static int _ilndr_pieces_deal(const vfft_ilndr_t *d, int T)
{
    long load[THREAD_POOL_MAX_DISPATCH];
    const int N1 = d->N[0];
    const int cap = (N1 + T - 1) / T > 0 ? (N1 + T - 1) / T : 1;
    int i, j, t, npc = 0;
    for (t = 0; t < T; t++)
        load[t] = 0;
    for (i = 0; i < d->ncyc; i++)
    {
        const int L = d->cycl[i], k = (L + cap - 1) / cap, first = npc;
        if (npc + k > d->npc_max)
            return 0;
        for (j = 0; j < k; j++)
        {
            const int a = d->cyc0[i] + (int)((long)L * j / k), b = d->cyc0[i] + (int)((long)L * (j + 1) / k);
            int best = 0;
            d->pc_a[npc] = a;
            d->pc_n[npc] = b - a;
            d->pc_next[npc] = (j + 1 < k) ? npc + 1 : first;
            for (t = 1; t < T; t++)
                if (load[t] < load[best])
                    best = t;
            d->pc_w[npc] = best;
            load[best] += b - a;
            npc++;
        }
    }
    *(int *)&d->npc = npc;   /* the deal is the execute's; the struct is const to the walk */
    return npc > 0;
}
/* the cycles of neg0 at create: their starts and lengths, longest first; the walk sequence S */
static int _ilndr_cycles(vfft_ilndr_t *d)
{
    const int N1 = d->N[0];
    int q0, n = 0, i, j, k = 0;
    d->cycs = (int *)malloc((size_t)N1 * sizeof(int));
    d->cycl = (int *)malloc((size_t)N1 * sizeof(int));
    d->seq = (int *)malloc((size_t)N1 * sizeof(int));
    d->cyc0 = (int *)malloc((size_t)N1 * sizeof(int));
    if (!d->cycs || !d->cycl || !d->seq || !d->cyc0 || !d->vis)
        return 0;
    memset(d->vis, 0, (size_t)N1);
    for (q0 = 0; q0 < N1; q0++)
    {
        int cur = q0, len = 0;
        if (d->vis[q0])
            continue;
        while (!d->vis[cur])
        {
            d->vis[cur] = 1;
            len++;
            cur = d->pos0[cur];
        }
        d->cycs[n] = q0;
        d->cycl[n] = len;
        n++;
    }
    for (i = 1; i < n; i++)   /* longest first: the deal below balances the workers */
    {
        const int s = d->cycs[i], l = d->cycl[i];
        for (j = i; j > 0 && d->cycl[j - 1] < l; j--)
        {
            d->cycs[j] = d->cycs[j - 1];
            d->cycl[j] = d->cycl[j - 1];
        }
        d->cycs[j] = s;
        d->cycl[j] = l;
    }
    d->ncyc = n;
    for (i = 0; i < n; i++)
    {   /* S: cycle i from its start, position after position */
        int cur = d->cycs[i];
        d->cyc0[i] = k;
        for (j = 0; j < d->cycl[i]; j++)
        {
            d->seq[k++] = cur;
            cur = d->pos0[cur];
        }
    }
    return k == N1;
}
/* THE c2r SERIAL WALK: CCE volume z -> real cube y. Axis 0 is the forward chain on the natural
 * input (stage 0 out of place into V, or in place under destroy), then the planes */
static void _ilndr_execute_c2r(const vfft_ilndr_t *d, const double *z, double *y)
{
    const vfft_ilcol_t *a = &d->ax0;
    double *V = d->destroy ? (double *)z : d->V;
    double t0 = d->prof ? vfft_now_ns() : 0.0, t1;
    if (d->arm == 3)
    {   /* the band: the wide prefix z -> V, then per band the suffix and the band's planes */
        int q0;
        if (d->prof) { _ilndr_prof.planes = 0.0; _ilndr_prof.axis0 = 0.0; }
        if (d->cut > 0)
            _il2d_col_stages(z, V, d->N[0], d->plane, 0, d->cut, a->R, a->L, a->f, a->tf, 0);
        if (d->prof) { t1 = vfft_now_ns(); _ilndr_prof.axis0 += t1 - t0; t0 = t1; }
        for (q0 = 0; q0 < d->N[0]; q0 += d->wl)
        {
            double *band = V + 2 * (size_t)q0 * d->plane;
            const double *from = d->cut > 0 ? band : z + 2 * (size_t)q0 * d->plane;
            if (d->cut < a->nst)
                _il2d_col_stages(from, band, d->wl, d->plane, d->cut, a->nst, a->R, a->L, a->f, a->tf, 0);
            if (d->prof) { t1 = vfft_now_ns(); _ilndr_prof.axis0 += t1 - t0; t0 = t1; }
            _ilndr_c2r_planes(d, V, y, q0, q0 + d->wl);
            if (d->prof) { t1 = vfft_now_ns(); _ilndr_prof.planes += t1 - t0; t0 = t1; }
        }
        if (d->prof) _ilndr_prof_print(d);
        return;
    }
    if (d->nf == 2)
        _ilndr_axis0_strips(d, z, V, NULL);
    else
        _il2d_col_stages(z, V, d->N[0], d->plane, 0, a->nst, a->R, a->L, a->f, a->tf, 0);
    if (d->prof) { t1 = vfft_now_ns(); _ilndr_prof.axis0 = t1 - t0; t0 = t1; }
    if (d->ip)
        _ilndr_c2r_planes_ip(d, V);   /* V is the caller's volume: the planes to their positions */
    else
        _ilndr_c2r_planes(d, V, y, 0, d->N[0]);
    if (d->prof) { _ilndr_prof.planes = vfft_now_ns() - t0; _ilndr_prof_print(d); }
}
/* ═══ THE THREADED WALKS (phase 3): pure loop restrictions of the serial walks, so MT == ST bitwise.
 * r2c: workers take disjoint PLANE RANGES (the structure on their clone), then disjoint ROW-BLOCK
 * ranges of the virtual row for axis 0 (a block = hp3 columns, so the pay-once leaf's permuted writes
 * stay in rows nobody else writes); c2r: disjoint column ranges for axis 0's forward chain, then
 * plane ranges; the c2r band arm: the prefix by column ranges, then disjoint BANDS (the suffix and
 * the band's planes on the worker's clone). Worker t > 0 runs the child's clone cw[t-1] and its
 * own strip scratch. ══════════════════════════════════════════════════════════════════════════ */
typedef struct { const vfft_ilndr_t *d; const double *in; double *out; double *y; int mode, tid, nt, arm, nf; size_t lo, hi; } _ilndr_mt_arg;
static void _ilndr_mt_tramp(void *v)
{
    _ilndr_mt_arg *a = (_ilndr_mt_arg *)v;
    const vfft_ilndr_t *d = a->d;
    const vfft_ilcol_t *c = &d->ax0;
    struct vfft_plan_s *cf = a->tid > 0 ? d->cw[a->tid - 1] : (d->c2r ? d->cb : d->cf);
    double *sscr = a->tid > 0 ? d->sscrw[a->tid - 1] : d->sscr;
    const size_t hp3 = (size_t)d->hp3;
    switch (a->mode)
    {
    case 0: /* r2c planes [lo, hi) on this worker's clone */
        if (a->arm == 2)
            _ilndr_planes_payonce_range(d, cf, a->in, a->out, (int)a->lo, (int)a->hi);
        else
            _ilndr_planes_child_range(d, cf, a->in, a->out, (int)a->lo, (int)a->hi);
        break;
    case 1: /* r2c axis 0 over the row blocks [lo, hi): in = the planes' volume (z or V), out = z */
        if (a->arm == 2)
        {
            if (a->nf == 2)
                _ilndr_axis0_strips_blocks_range(d, sscr, a->in, a->out, (int)a->lo, (int)a->hi);
            else
                _ilndr_axis0_leafoop_range(d, (double *)a->in, a->out, (int)a->lo, (int)a->hi);
        }
        else if (a->nf == 2)
            _ilndr_axis0_strips_range(d, sscr, a->in, a->out, d->nat0, a->lo * hp3, a->hi * hp3);
        else   /* the one-stage kernel in place by column range (the natural pass never threads) */
            _il2d_col_pass_range(a->out, a->out, d->N[0], d->plane, a->lo * hp3, a->hi * hp3,
                                 c->nst, c->R, c->L, c->f, c->tf, 0);
        break;
    case 2: /* c2r axis 0: the forward chain over the columns [lo, hi), z -> V (or in place) */
        if (a->nf == 2)
            _ilndr_axis0_strips_range(d, sscr, a->in, a->out, NULL, a->lo, a->hi);
        else
            _il2d_col_pass_range(a->in, a->out, d->N[0], d->plane, a->lo, a->hi,
                                 c->nst, c->R, c->L, c->f, c->tf, 0);
        break;
    case 3: /* c2r planes [lo, hi) from V on this worker's clone */
        _ilndr_c2r_planes_on(d, cf, a->arm, (double *)a->in, a->out, (int)a->lo, (int)a->hi);
        break;
    case 6: /* c2r in place: this worker's pieces of the walk sequence on its clone and buffers */
        _ilndr_c2r_pieces_w(d, cf, a->tid, a->arm, a->out);
        break;
    case 7: /* c2r in place, before the pieces: this worker's pieces' first planes copied out */
        _ilndr_c2r_copies_w(d, a->tid, a->out);
        break;
    case 4: /* the c2r band's prefix over the columns [lo, hi), z -> V */
        _il2d_col_stages2(a->in + 2 * a->lo, a->out + 2 * a->lo, d->N[0], d->plane, a->hi - a->lo,
                          0, d->mtcut, c->R, c->L, c->f, c->tf, 0);
        break;
    case 5: /* the c2r bands [lo, hi): the suffix in place on V, then the band's planes on the clone */
    {
        const int wl = d->mtwl, cut = d->mtcut;
        size_t b;
        for (b = a->lo; b < a->hi; b++)
        {
            const int q0 = (int)b * wl;
            double *band = a->out + 2 * (size_t)q0 * d->plane;   /* out = V here; in = z */
            const double *from = cut > 0 ? band : a->in + 2 * (size_t)q0 * d->plane;
            if (cut < c->nst)
                _il2d_col_stages(from, band, wl, d->plane, cut, c->nst, c->R, c->L, c->f, c->tf, 0);
            _ilndr_c2r_planes_on(d, cf, 1, a->out, a->y, q0, q0 + wl);   /* out = V, y = the real cube */
        }
        break;
    }
    default:
        break;
    }
}
/* one phase across T workers (the caller is tid 0): units split evenly; y = the real output the
 * band mode writes (NULL elsewhere) */
static void _ilndr_mt_phase(const vfft_ilndr_t *d, const double *in, double *out, double *y, int mode, int arm, int nf,
                            size_t units, int T)
{
    _ilndr_mt_arg a[THREAD_POOL_MAX_DISPATCH];
    int t;
    for (t = 0; t < T; t++)
    {
        a[t].d = d; a[t].in = in; a[t].out = out; a[t].y = y; a[t].mode = mode; a[t].tid = t; a[t].nt = T;
        a[t].arm = arm; a[t].nf = nf;
        a[t].lo = units * (size_t)t / (size_t)T;
        a[t].hi = units * (size_t)(t + 1) / (size_t)T;
    }
    thread_pool_run(T, _ilndr_mt_tramp, a, sizeof a[0]);
}
/* the plane team at the dispatch's T: the raced width, never above min(N1, T); the half team the
 * race adds where some full-team worker would hold a single plane (the c2c tier's rule) */
static int _ilndr_plane_team(const vfft_ilndr_t *d, int T)
{
    const int Tp = d->N[0] < T ? d->N[0] : T;
    return (d->ptw > 0 && d->ptw < Tp) ? d->ptw : Tp;
}
static int _ilndr_half_team(const vfft_ilndr_t *d, int T)
{
    const int Tp = d->N[0] < T ? d->N[0] : T;
    return (Tp >= 4 && d->N[0] < 2 * Tp) ? Tp / 2 : 0;
}
/* the whole transform at the pool's T under the threaded verdict (mts, mtf, mtwl); 0 = cannot engage */
static int _ilndr_execute_mt(const vfft_ilndr_t *d, const double *in, double *out)
{
    const int T = thread_pool_workers_for(d->mt_t);
    const int arm = d->mts, nf = d->mtf;
    const size_t N2 = (size_t)d->N[1], P = d->plane;
    if (T < 2 || d->mt != 2 || d->ncw < T - 1)
        return 0;
    if (nf == 2 && d->nsscrw < T - 1)
        return 0;
    if (!d->c2r)
    {
        const int Tp = _ilndr_plane_team(d, T), Tb = N2 < (size_t)T ? (int)N2 : T;
        double *mid = arm == 2 ? d->V : out;
        if (arm == 1 && nf == 1 && d->ax0n_on)
            return 0;   /* the natural pass through the cube scratch does not thread */
        if (arm == 2 && !d->V)
            return 0;
        _ilndr_mt_phase(d, in, mid, NULL, 0, arm, nf, (size_t)d->N[0], Tp);
        _ilndr_mt_phase(d, mid, out, NULL, 1, arm, nf, N2, Tb);
    }
    else
    {
        double *V = d->destroy ? (double *)in : d->V;
        const int Ts = P < (size_t)T ? (int)P : T, Tp = _ilndr_plane_team(d, T);
        if (d->ip && (arm == 3 || d->npbw < Tp - 1 || d->ncyc < 1))
            return 0;   /* in place: the band is not an arm; every worker needs its buffers */
        if (arm == 3)
        {
            const int nb = d->mtwl > 0 ? d->N[0] / d->mtwl : 0;
            const int Tb = nb < T ? nb : T;
            if (nb < 2)
                return 0;
            if (d->mtcut > 0)
                _ilndr_mt_phase(d, in, V, NULL, 4, arm, nf, P, Ts);
            _ilndr_mt_phase(d, in, V, out, 5, arm, nf, (size_t)nb, Tb);
        }
        else
        {
            if (d->ip && !_ilndr_pieces_deal(d, Tp))
                return 0;   /* more pieces than buffers: serial -- decided before any phase touches the volume */
            _ilndr_mt_phase(d, in, V, NULL, 2, arm, nf, P, Ts);
            if (d->ip)
            {   /* the planes by PIECES of the walk sequence: the copies, the barrier, the walk */
                _ilndr_mt_phase(d, V, V, NULL, 7, arm, nf, (size_t)d->npc, Tp);
                _ilndr_mt_phase(d, V, V, NULL, 6, arm, nf, (size_t)d->npc, Tp);
            }
            else
                _ilndr_mt_phase(d, V, out, NULL, 3, arm, nf, (size_t)d->N[0], Tp);
        }
    }
    _vfft_ilnd_mt_count++;   /* engagement: the c2c tier's counter, vfft_ilnd_mt_passes() */
    return 1;
}
static void vfft_ilndr_execute(const vfft_ilndr_t *d, vfft_dir_t dir, const double *sre, double *dre)
{
    (void)dir;   /* the plan's direction is its transform's */
    if (d->mt == 2 && d->mt_t > 1 && _ilndr_execute_mt(d, sre, dre))
        return;
    if (d->c2r)
        _ilndr_execute_c2r(d, sre, dre);
    else
        _ilndr_execute_r2c(d, sre, dre);
}

/* ═══ destroy ══════════════════════════════════════════════════════════ */
static void vfft_ilndr_destroy(vfft_ilndr_t *d)
{
    if (!d)
        return;
    if (d->cf)
        vfft_destroy((vfft_plan)d->cf);
    if (d->cb)
        vfft_destroy((vfft_plan)d->cb);
    vfft_child_store_free(d->childS);
    _il2d_col_free(&d->ax0);
    if (d->ax0n_on)
        _il2d_col_free(&d->ax0n);
    if (d->ax1_on)
        _il2d_col_free(&d->ax1);
    free(d->nat0);
    free(d->nat1);
    free(d->neg0);
    free(d->neg1);
    free(d->pos0);
    vfft_aligned_free(d->V);
    vfft_aligned_free(d->sscr);
    vfft_aligned_free(d->pbuf[0]);
    vfft_aligned_free(d->pbuf[1]);
    vfft_aligned_free(d->pbuf[2]);
    free(d->vis);
    free(d->cycs);
    free(d->cycl);
    free(d->seq);
    free(d->cyc0);
    free(d->pc_a);
    free(d->pc_n);
    free(d->pc_next);
    free(d->pc_w);
    if (d->pbw)
    {
        int t;
        for (t = 0; t < 3 * d->npbw; t++)
            vfft_aligned_free(d->pbw[t]);
        free(d->pbw);
    }
    if (d->pcb)
    {
        int p;
        for (p = 0; p < d->npc_max; p++)
            vfft_aligned_free(d->pcb[p]);
        free(d->pcb);
    }
    if (d->cw)
    {
        int t;
        for (t = 0; t < d->ncw; t++)
            if (d->cw[t]) vfft_destroy((vfft_plan)d->cw[t]);
        free(d->cw);
    }
    if (d->sscrw)
    {
        int t;
        for (t = 0; t < d->nsscrw; t++)
            vfft_aligned_free(d->sscrw[t]);
        free(d->sscrw);
    }
    free(d);
}

/* ═══ clones (phase 3): a 2D real child clone is equivalent to its primary iff every verdict that
 * decides output bits matches -- the row engine (by its name: kernel or engine recipe), the column
 * plan (chain, kernel pointers, natural form, leaf), the whole-plan forms (fused walk, real axis,
 * skewed plane, destroying c2r), the column-inverse plane's pitch, the odd door, and the door
 * batch's row plan (_tc_clone_equiv). Stack states are not bits. ═══════════════════════════════ */
static int _ilndr_child_equiv(const struct vfft_plan_s *a, const struct vfft_plan_s *b)
{
    const vfft_ilcol_t *x = &a->il2d_col, *y = &b->il2d_col;
    const char *why = NULL;
    char na[64], nb[64];
    int s;
    if (a->N != b->N || a->N2 != b->N2 || a->transform != b->transform) why = "shape";
    else if (a->il2d_rx_on != b->il2d_rx_on || a->il2d_rx_lm != b->il2d_rx_lm || (a->il2d_rx_eng != NULL) != (b->il2d_rx_eng != NULL)) why = "row engine";
    else if (a->il2d_oddn2 != b->il2d_oddn2) why = "odd door";
    else if (a->il2d_cx_leaf != b->il2d_cx_leaf) why = "column leaf";
    else if (x->nst != y->nst || x->nat != y->nat || x->blu != y->blu || x->tpc != y->tpc || x->colmt != y->colmt) why = "column plan";
    else if (a->il2d_cx_st != b->il2d_cx_st) why = "staged leaf";
    else if (a->il2d_tf_on != b->il2d_tf_on || a->il2d_rax_on != b->il2d_rax_on || a->il2d_rcsk_on != b->il2d_rcsk_on || a->il2d_cxd_on != b->il2d_cxd_on) why = "whole-plan form";
    else if (a->il2d_rscr_P != b->il2d_rscr_P) why = "plane pitch";
    else if (a->il2d_ip != b->il2d_ip || a->il2d_ipP != b->il2d_ipP) why = "placement";
    else
    {
        for (s = 0; s < x->nst && !why; s++)
            if (x->R[s] != y->R[s] || x->L[s] != y->L[s] || x->f[s] != y->f[s] || x->b[s] != y->b[s]) why = "chain";
        if (!why && a->il2d_rx_on)
        {
            na[0] = nb[0] = 0;
            _il2d_rowx_name(a->il2d_rx_lm, a->il2d_rx_eng, na, sizeof na);
            _il2d_rowx_name(b->il2d_rx_lm, b->il2d_rx_eng, nb, sizeof nb);
            if (strcmp(na, nb) != 0) why = "row engine recipe";
        }
        if (!why && a->il2d_row && b->il2d_row && !_tc_clone_equiv(a->il2d_row, b->il2d_row)) why = "row plan";
        if (!why && (a->il2d_row != NULL) != (b->il2d_row != NULL)) why = "row plan";
    }
    if (why && getenv("VFFT_IL2D_LOG"))
        fprintf(stderr, "[ilndr] child clone %dx%d not equivalent: %s\n", a->N, a->N2, why);
    return why == NULL;
}
static void _ilndr_free_clones(vfft_ilndr_t *d)
{
    int t;
    if (d->cw)
    {
        for (t = 0; t < d->ncw; t++)
            if (d->cw[t]) vfft_destroy((vfft_plan)d->cw[t]);
        free(d->cw);
    }
    d->cw = NULL;
    d->ncw = 0;
}
/* T - 1 clones of the plane child on its own store (they replay its recipe, never bank), and the
 * workers' strip scratches; 0 = the structure cannot thread (every clone freed) */
static int _ilndr_build_clones(vfft_ilndr_t *d, const vfft_config_t *cfg, int T)
{
    const int n = (T > THREAD_POOL_MAX_DISPATCH ? THREAD_POOL_MAX_DISPATCH : T) - 1;
    struct vfft_plan_s *prim = d->c2r ? d->cb : d->cf;
    vfft_config_t cc;
    int t;
    if (n <= 0 || !prim || !d->childS)
        return 0;
    if (d->ncw >= n)
        return d->ncw;
    memset(&cc, 0, sizeof cc);
    cc.transform = cfg->transform;
    cc.placement = (d->ip && !d->c2r) ? VFFT_INPLACE : VFFT_OUTOFPLACE;   /* as the primary: in place for r2c's planes */
    cc.owned_buffers = (d->ip == 2 && !d->c2r);                            /* door 2: the pitch, no plane (a nested create) */
    cc.rigor = cfg->rigor;
    cc.dims = 2;
    cc.n[0] = d->N[1];
    cc.n[1] = d->N[2];
    cc.howmany = 1;
    cc.order = VFFT_ORDER_NATURAL;
    cc.layout = VFFT_LAYOUT_INTERLEAVED;
    cc.nthreads = 1;
    cc.wisdom = (vfft_wisdom *)d->childS;
    cc.wisdom_write = 0;
    d->cw = (struct vfft_plan_s **)calloc((size_t)n, sizeof *d->cw);
    if (!d->cw)
        return 0;
    for (t = 0; t < n; t++)
    {
        struct vfft_plan_s *c = (struct vfft_plan_s *)vfft_create(&cc);
        d->cw[t] = c;
        if (!c || c->nthreads > 1 || !_ilndr_child_equiv(prim, c))
        {
            _vfft_warn("ilndr MT: 2D real child clone %d %s at %dx%d -- the structure cannot thread for this plan",
                       t, c ? "route-mismatched" : "failed to create", d->N[1], d->N[2]);
            d->ncw = t + 1;
            _ilndr_free_clones(d);
            return 0;
        }
    }
    d->ncw = n;
    if (d->c2r && d->ip && d->npbw < n)
    {   /* the workers' buffers for the planes: door 2's compaction and the tight plane (the child arm) */
        d->pbw = (double **)calloc(3 * (size_t)n, sizeof *d->pbw);
        if (d->pbw)
            for (t = 0; t < n; t++)
            {
                d->pbw[3 * t] = NULL;
                d->pbw[3 * t + 1] = d->ip == 2 ? (double *)vfft_aligned_alloc((2 * (size_t)d->N[1] * (size_t)d->hpn + 8) * sizeof(double)) : NULL;
                d->pbw[3 * t + 2] = (double *)vfft_aligned_alloc(((size_t)d->N[1] * (size_t)d->N[2] + 8) * sizeof(double));
                if (!d->pbw[3 * t + 2] || (d->ip == 2 && !d->pbw[3 * t + 1])) break;
                d->npbw = t + 1;
            }
    }
    if (d->c2r && d->ip && !d->pcb)
    {   /* the pieces' first-plane copies: at most T + cycles pieces (every cycle cut into pieces of
         * ceil(N1/T) planes), and the piece tables */
        const int m = d->ncyc + n + 1;
        int p;
        d->pcb = (double **)calloc((size_t)m, sizeof *d->pcb);
        d->pc_a = (int *)malloc((size_t)m * sizeof(int));
        d->pc_n = (int *)malloc((size_t)m * sizeof(int));
        d->pc_next = (int *)malloc((size_t)m * sizeof(int));
        d->pc_w = (int *)malloc((size_t)m * sizeof(int));
        if (d->pcb && d->pc_a && d->pc_n && d->pc_next && d->pc_w)
            for (p = 0; p < m; p++)
            {
                d->pcb[p] = (double *)vfft_aligned_alloc((2 * d->plane + 8) * sizeof(double));
                if (!d->pcb[p]) break;
                d->npc_max = p + 1;
            }
    }
    if (d->sscr && d->nsscrw < n)
    {
        d->sscrw = (double **)calloc((size_t)n, sizeof *d->sscrw);
        if (d->sscrw)
            for (t = 0; t < n; t++)
            {
                d->sscrw[t] = (double *)vfft_aligned_alloc((2 * (size_t)d->N[0] * (size_t)d->nsw + 8) * sizeof(double));
                if (!d->sscrw[t]) break;
                d->nsscrw = t + 1;
            }
    }
    return n;
}

/* ═══ wisdom: the 2D-shard borrow (owner 2026-10-07) ═══════════════════
 * The child store is seeded from the parent row's plane_ tokens; when the row
 * carries none, from the 2D shard's real row of (N2, N3) at the plan's T — the
 * row's payload tokens copied whole (its own children ride on it), banked into
 * the private store as a seed (not raced). The 2D shard is never written. */
static void _ilndr_borrow_2d(struct vfft_wisdom_s *W, struct vfft_wisdom_s *S, int N2, int N3, int T, int ip)
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
    ck.ip = ip;   /* the borrow holds within a placement: an in-place plane child reads the 2D shard's pl=ip row */
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
typedef struct { vfft_ilndr_t *d; const double *in; double *out; int arm, nf, cut, wl; char name[24]; } _ilndr_arm_ctx_t;
static void _ilndr_arm_run(void *v)
{
    _ilndr_arm_ctx_t *c = (_ilndr_arm_ctx_t *)v;
    c->d->arm = c->arm;
    c->d->nf = c->nf;
    c->d->cut = c->cut;
    c->d->wl = c->wl;
    if (c->d->c2r)
        _ilndr_execute_c2r(c->d, c->in, c->out);
    else
        _ilndr_execute_r2c(c->d, c->in, c->out);
}

/* ═══ the threaded race (phase 3): serial on a small cube, else plane x {structures} x {forms that
 * thread} x {full team, half team}, every arm the whole transform at the plan's T ═══════════════ */
typedef struct { vfft_ilndr_t *d; const double *in; double *out; int mt, arm, nf, cut, wl, ptw, ok; char name[32]; } _ilndr_mt_ctx_t;
static void _ilndr_mt_arm_run(void *v)
{
    _ilndr_mt_ctx_t *c = (_ilndr_mt_ctx_t *)v;
    vfft_ilndr_t *d = c->d;
    if (c->mt == 0)
    {
        d->arm = c->arm; d->nf = c->nf; d->cut = c->cut; d->wl = c->wl;
        if (d->c2r) _ilndr_execute_c2r(d, c->in, c->out); else _ilndr_execute_r2c(d, c->in, c->out);
        return;
    }
    d->mt = 2; d->mts = c->arm; d->mtf = c->nf; d->mtcut = c->cut; d->mtwl = c->wl; d->ptw = c->ptw;
    if (c->ok && !_ilndr_execute_mt(d, c->in, c->out))
        c->ok = 0;   /* the arm cannot engage on this cell */
}

/* ═══ create ═══════════════════════════════════════════════════════════ */
static vfft_plan _vfft_create_fftnd_real_il(const vfft_config_t *cfg, struct vfft_wisdom_s *W, size_t K)
{
    const int N1 = cfg->n[0], N2 = cfg->n[1], N3 = cfg->n[2];
    const int c2r = (cfg->transform == VFFT_C2R);
    const int nthr = _vfft_plan_threads(cfg);
    const int ip = (cfg->placement == VFFT_INPLACE);   /* in place (real_inplace_design.md): its own row, its own race */
    const int usable_w = (W && !W->vw2_off_2d);
    const char *log = getenv("VFFT_IL2D_LOG");
    const char *apin = getenv("VFFT_ILNDR_ARM"), *fpin = getenv("VFFT_ILNDR_NF"), *wpin = getenv("VFFT_ILNDR_SW"), *bpin = getenv("VFFT_ILNDR_WL");
    char tkb_s[16], tkb_nf[16], tkb_nsw[16], tkb_wl[16];   /* the structure tokens; door 2 keeps its own _p set */
    const char *tk_s = tkb_s, *tk_nf = tkb_nf, *tk_nsw = tkb_nsw, *tk_wl = tkb_wl;
    vfft_ilndr_t *d;
    struct vfft_plan_s *h;
    vw2_ilcol_key_t key0, key1;
    vw2_key_t pk;
    char forms0[64], forms1[64];
    int bwl, btf, bro, bcmt, bcmtt, bblu;
    int payonce_ok, strips_ok, arm = 0, nf = 0, nsw = 0, wl = 0, cut = 0, raced = 0, pinned, ip2 = 0;
    if (!vfft_policy_ilndr_ok(cfg, K) && !vfft_policy_ilndr_ip_ok(cfg, K, nthr))
    {
        _vfft_warn("vfft_create: 3D INTERLEAVED real serves R2C and C2R, howmany==1, out of place, "
                   "or in place (one volume, every row padded to 2*(N3/2+1) doubles), "
                   "DEFAULT/NATURAL order, every dim >= 2 (got %s, howmany=%zu, %dx%dx%d, %s)",
                   _vfft_tname(cfg->transform), K, N1, N2, N3, ip ? "in place" : "out of place");
        return NULL;
    }
    d = (vfft_ilndr_t *)calloc(1, sizeof *d);
    if (!d)
        return NULL;
    d->N[0] = N1; d->N[1] = N2; d->N[2] = N3;
    d->hpn = N3 / 2 + 1;
    /* door 2 (owned_buffers = 1, real_inplace_design.md §1): the rows at the pitch the policy names;
     * the plan's hp3 IS that pitch, the plane and the virtual row follow (the pad columns are zeros) */
    d->hp3 = (ip && cfg->owned_buffers) ? (int)vfft_policy_il2d_ip_pitch((size_t)d->hpn) : d->hpn;
    ip2 = ip && d->hp3 != d->hpn;
    d->plane = (size_t)N2 * (size_t)d->hp3;
    d->c2r = c2r;
    d->ip = ip ? (ip2 ? 2 : 1) : 0;
    d->destroy = c2r && (cfg->destroy_input || ip);   /* in place: the input is the output */
    snprintf(tkb_s, sizeof tkb_s, "%s%s", c2r ? "s_c2r" : "s", ip2 ? "_p" : "");
    snprintf(tkb_nf, sizeof tkb_nf, "%s%s", c2r ? "nf_c2r" : "nf", ip2 ? "_p" : "");
    snprintf(tkb_nsw, sizeof tkb_nsw, "%s%s", c2r ? "nsw_c2r" : "nsw", ip2 ? "_p" : "");
    snprintf(tkb_wl, sizeof tkb_wl, "wl_c2r%s", ip2 ? "_p" : "");
    d->mt_t = nthr;
    d->prof = getenv("VFFT_ILNDR_PROF") != NULL;   /* bound once here: never read on the execute path */
    _il2d_blu_ctx.W = W;
    _il2d_blu_ctx.cfg = cfg;
    _il2d_blu_chain_hook = _il2d_blu_m_chain;
    memset(&key0, 0, sizeof key0);
    key0.rank = 3; key0.n0 = N1; key0.n1 = N2; key0.n2 = N3;
    key0.ord = VW2_ORD_NAT; key0.axis = 0; key0.real = 1; key0.nthreads = nthr;
    key0.ip = ip2 ? 2 : ip;   /* the in-place cell's own row of the 3D shard; door 2 its own _p verdicts there */
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
        _ilndr_borrow_2d(W, d->childS, N2, N3, nthr, (ip && !c2r) ? (ip2 ? 2 : 1) : 0);
    {
        vfft_config_t cc;
        struct vfft_plan_s *c;
        memset(&cc, 0, sizeof cc);
        cc.transform = cfg->transform;
        cc.placement = (ip && !c2r) ? VFFT_INPLACE : VFFT_OUTOFPLACE;   /* r2c in place: the 2D in-place plan per plane;
                                                                          * c2r: the out-of-place child feeds the cycle walk */
        cc.owned_buffers = (ip2 && !c2r);   /* door 2: the plane child at the plan's pitch (a nested create takes the pitch, no plane) */
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
        c = (struct vfft_plan_s *)vfft_create(&cc);
        if (!c)
        {
            _vfft_warn("vfft_create: 3D INTERLEAVED %s %dx%dx%d: no 2D real plane plan at %dx%d",
                       c2r ? "c2r" : "r2c", N1, N2, N3, N2, N3);
            vfft_ilndr_destroy(d);
            return NULL;
        }
        if (c2r) d->cb = c; else d->cf = c;
    }
    /* ── axis 0: the plain chain (raced over the pool on a miss; creates the row; direction-shared) ── */
    forms0[0] = 0;
    if (!_il2d_col_build(W, cfg, &key0, N1, d->plane, 0, &d->ax0, forms0, sizeof forms0,
                         &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu))
    {
        vfft_ilndr_destroy(d);
        return NULL;
    }
    d->nat0 = _ilndr_nat_perm(d->ax0.R, d->ax0.nst, N1);
    if (c2r && d->nat0)
        d->neg0 = _ilndr_neg_of(d->nat0, N1);
    if (c2r && ip && d->neg0 && !(d->pos0 = _ilndr_inv_of(d->neg0, N1)))
    {
        vfft_ilndr_destroy(d);
        return NULL;
    }
    if (c2r && ip && (!(d->vis = (char *)malloc((size_t)N1)) || !_ilndr_cycles(d)))
    {   /* the cycles of the planes' positions, for the serial walk and the workers' deal */
        vfft_ilndr_destroy(d);
        return NULL;
    }
    strips_ok = !d->ax0.blu && !d->ax0.tpc && d->nat0 != NULL;
    if (c2r && !strips_ok)
    {   /* c2r composes the forward chain's stages: a Bluestein or turned axis 0 has none */
        _vfft_warn("vfft_create: 3D INTERLEAVED c2r %dx%dx%d: axis 0 has no plain chain (a prime N1 is the tier's next phase)", N1, N2, N3);
        vfft_ilndr_destroy(d);
        return NULL;
    }
    /* r2c: the in-place natural twin: the natural pass at two stages or more (a cube-sized pre-leaf
     * scratch), or a Bluestein axis; a one-stage chain is natural-native in place as it is */
    if (!c2r && (d->ax0.blu || d->ax0.tpc || d->ax0.nst >= 2))
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
    /* ── the pay-once pieces: axis 1's plain chain per plane (chain1=, direction-shared) ── */
    payonce_ok = strips_ok;
    if (payonce_ok)
    {
        forms1[0] = 0;
        if (_il2d_col_build(W, cfg, &key1, N2, (size_t)d->hp3, 0, &d->ax1, forms1, sizeof forms1,
                            &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu) && !d->ax1.blu && !d->ax1.tpc)
        {
            d->ax1_on = 1;
            d->nat1 = _ilndr_nat_perm(d->ax1.R, d->ax1.nst, N2);
            if (c2r && d->nat1)
                d->neg1 = _ilndr_neg_of(d->nat1, N2);
            if (!d->nat1 || (c2r && !d->neg1))
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
    /* ── the private volume: the pay-once r2c, every c2r arm unless the input may be destroyed ── */
    if ((!c2r && payonce_ok) || (c2r && !d->destroy))
    {
        d->V = (double *)vfft_aligned_alloc((2 * (size_t)N1 * d->plane + 8) * sizeof(double));
        if (!d->V)
        {
            if (c2r) { vfft_ilndr_destroy(d); return NULL; }
            payonce_ok = 0;
        }
        else
            memset(d->V, 0, (2 * (size_t)N1 * d->plane + 8) * sizeof(double));   /* faulted in at create: the first execute's pages
                                                                                    * cost 2 ms at a 17 MB volume (profiled 2026-10-07) */
    }
    /* ── the strip scratch ── */
    nsw = wpin ? atoi(wpin) : 0;
    if (nsw <= 0 && usable_w)
        nsw = vw2_ilnd_int_lookup(&W->vw2, &key0, tk_nsw);
    if (nsw <= 0)
        nsw = vfft_policy_ilndr_strip_w(N1, d->plane);
    if (strips_ok)
    {
        d->nsw = nsw;
        d->sscr = (double *)vfft_aligned_alloc((2 * (size_t)N1 * (size_t)nsw + 8) * sizeof(double));
        if (!d->sscr)
            strips_ok = 0;
    }
    /* ── c2r in place: the planes' walk -- the cycle's first plane, and the child arm's tight plane ── */
    if (c2r && ip)
    {
        d->pbuf[0] = (double *)vfft_aligned_alloc((2 * d->plane + 8) * sizeof(double));
        d->pbuf[2] = (double *)vfft_aligned_alloc(((size_t)N2 * (size_t)N3 + 8) * sizeof(double));
        if (ip2)
            d->pbuf[1] = (double *)vfft_aligned_alloc((2 * (size_t)N2 * (size_t)d->hpn + 8) * sizeof(double));
        if (!d->pbuf[0] || !d->pbuf[2] || (ip2 && !d->pbuf[1]))
        {
            vfft_ilndr_destroy(d);
            return NULL;
        }
    }
    /* ── the verdict: pin > banked > the race of (structure x form) ── */
    pinned = (apin && apin[0]) || (fpin && fpin[0]) || (bpin && bpin[0]);
    if (apin && apin[0]) arm = atoi(apin);
    if (fpin && fpin[0]) nf = atoi(fpin);
    if (bpin && bpin[0]) wl = atoi(bpin);
    if (!pinned && usable_w && !cfg->recalibrate)
    {
        arm = (c2r || ip2) ? vw2_ilnd_int_lookup(&W->vw2, &key0, tk_s) : vw2_ilnd_arm_lookup(&W->vw2, &key0);
        nf = vw2_ilnd_int_lookup(&W->vw2, &key0, tk_nf);
        if (c2r && arm == 3) wl = vw2_ilnd_int_lookup(&W->vw2, &key0, tk_wl);
    }
    if (arm == 2 && !payonce_ok) arm = 0;
    if (arm == 3 && (!c2r || ip)) arm = 0;   /* the band lands its planes out of place: not an in-place arm yet */
    if (nf == 2 && !strips_ok) nf = 0;
    if (arm == 3)
    {   /* the band's cut from its width: the stage whose sub-length is wl; a band pin without a
         * width (or a banked band without one) leaves the cut to the race over the legal cuts */
        int s;
        cut = 0;
        if (wl > 0)
        {
            for (s = 1; s < d->ax0.nst; s++)
                if (d->ax0.L[s] == wl) { cut = s; break; }
            if (!cut) arm = 0;
        }
        if (arm == 3) nf = 1;
    }
    if (!arm || !nf || (arm == 3 && !cut))
    {
        const size_t RN = (size_t)N1 * (size_t)N2 * (size_t)N3, CN = 2 * (size_t)N1 * d->plane;
        const size_t nin = ip ? CN : (c2r ? CN : RN), nout = ip ? CN : (c2r ? RN : CN);   /* IN / ON are Windows macros; in place: the one volume */
        double *in = (double *)vfft_aligned_alloc((nin + 8) * sizeof(double));
        double *out = ip ? in : (double *)vfft_aligned_alloc((nout + 8) * sizeof(double));
        double *seed = ip ? (double *)vfft_aligned_alloc((nin + 8) * sizeof(double)) : NULL;   /* in place: the volume re-laid before every sample */
        _il2d_ip_reset_t rs;
        _ilndr_arm_ctx_t ac[12];
        vfft_race_arm_t arms[12];
        double ns[12];
        int na = 0, a, best = 0, reps;
        if (!in || !out || (ip && !seed))
        {
            vfft_aligned_free(in); if (!ip) vfft_aligned_free(out); vfft_aligned_free(seed);
            vfft_ilndr_destroy(d);
            return NULL;
        }
        {
            unsigned sd = 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 10) ^ ((unsigned)N3 << 20);
            size_t j;
            for (j = 0; j < nin; j++)
            {
                sd = sd * 1664525u + 1013904223u;
                in[j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
            }
        }
        if (ip)
        {
            memcpy(seed, in, (nin + 8) * sizeof(double));
            rs.p = in; rs.seed = seed; rs.n = nin + 8;
        }
        else
            memset(out, 0, (nout + 8) * sizeof(double));
#define _ILNDR_ARM(A_, F_, C_, W_, NAME_)                                           \
        do {                                                                        \
            if (na < 12 && (!arm || arm == (A_)) && (!nf || nf == (F_)) && (!wl || (A_) != 3 || wl == (W_))) \
            {                                                                       \
                ac[na].d = d; ac[na].in = in; ac[na].out = out;                     \
                ac[na].arm = (A_); ac[na].nf = (F_); ac[na].cut = (C_); ac[na].wl = (W_); \
                snprintf(ac[na].name, sizeof ac[na].name, "%s", NAME_);             \
                arms[na].name = ac[na].name; arms[na].run = _ilndr_arm_run; arms[na].ctx = &ac[na]; \
                na++;                                                               \
            }                                                                       \
        } while (0)
        _ILNDR_ARM(1, 1, 0, 0, "child/inplace");
        if (strips_ok) _ILNDR_ARM(1, 2, 0, 0, "child/strips");
        if (payonce_ok) _ILNDR_ARM(2, 1, 0, 0, c2r ? "payonce/inplace" : "payonce/leafoop");
        if (payonce_ok && strips_ok) _ILNDR_ARM(2, 2, 0, 0, "payonce/strips");
        if (c2r && !ip)
        {   /* the band at every legal cut: stages s >= cut act within L[cut] planes */
            int s;
            for (s = 1; s < d->ax0.nst; s++)
                if (d->ax0.L[s] >= 2 && d->ax0.L[s] < N1)
                {
                    char nm[24];
                    snprintf(nm, sizeof nm, "band/wl%d", d->ax0.L[s]);
                    _ILNDR_ARM(3, 1, s, d->ax0.L[s], nm);
                }
        }
#undef _ILNDR_ARM
        if (na == 0)
        {
            vfft_aligned_free(in); if (!ip) vfft_aligned_free(out); vfft_aligned_free(seed);
            vfft_ilndr_destroy(d);
            return NULL;
        }
        /* the budget (the c2c tier's longer race, 2026-09-26): reps from ~1 ms of work with a
         * floor of 4, rounds for at least 48 timed executes per arm (3..15) -- a 3 x 1 race at a
         * 2 ms cell picked a 7% slower arm (16x256x256, 2026-10-07) */
        reps = (int)(1e6 / (double)(N1 * d->plane + 1));
        if (reps < 4) reps = 4;
        if (reps > 64) reps = 64;
        if (ip && reps > 32) reps = 32;   /* in place the volume grows a factor N1 N2 N3 per pass: a sample stays finite */
        if (na > 1)
        {
            int rounds = (48 + reps - 1) / reps;
            vfft_race_proto_t proto = { 3, reps, VFFT_RACE_MIN, 1, 0, NULL, NULL, 1 }; /* single-thread arms: paced */
            if (ip)
            {   /* the one volume, re-laid before every sample */
                proto.reset = _il2d_ip_reset;
                proto.reset_ctx = &rs;
            }
            if (rounds < 3) rounds = 3;
            if (rounds > 15) rounds = 15;
            proto.rounds = rounds;
            _vfft_create_race_count++;
            vfft_race_run(&proto, arms, na, ns);
            for (a = 1; a < na; a++)
                if (ns[a] < ns[best])
                    best = a;
            raced = 1;
            if (log)
            {
                fprintf(stderr, "[ilndr] %dx%dx%d %s race:", N1, N2, N3, c2r ? "c2r" : "r2c");
                for (a = 0; a < na; a++)
                    fprintf(stderr, " %s=%.0f", ac[a].name, ns[a]);
                fprintf(stderr, " -> %s\n", ac[best].name);
            }
        }
        arm = ac[best].arm;
        nf = ac[best].nf;
        cut = ac[best].cut;
        wl = ac[best].wl;
        vfft_aligned_free(in);
        if (!ip) vfft_aligned_free(out);
        vfft_aligned_free(seed);
    }
    d->arm = arm;
    d->nf = nf;
    d->cut = cut;
    d->wl = wl;
    /* the loser's resources go (at T > 1 the threaded race still needs every piece: it frees after) */
    if (nthr <= 1)
    {
        if (arm != 2)
        {
            if (!c2r) { vfft_aligned_free(d->V); d->V = NULL; }
            if (d->ax1_on) { _il2d_col_free(&d->ax1); d->ax1_on = 0; }
            free(d->nat1); d->nat1 = NULL;
            free(d->neg1); d->neg1 = NULL;
        }
        if (nf == 1)
        {
            vfft_aligned_free(d->sscr); d->sscr = NULL;
        }
        if ((nf == 2 || arm == 2) && d->ax0n_on)
        {
            _il2d_col_free(&d->ax0n); d->ax0n_on = 0;
        }
    }
    /* bank what was RACED (pins never bank) */
    if (usable_w && raced && !pinned)
    {
        int banked = 0;
        if (vw2_ilcol_row_ensure(&W->vw2, &key0, d->ax0.R, d->ax0.nst)) banked = 1;
        if ((c2r || ip2) ? vw2_ilnd_int_bank(&W->vw2, &key0, tk_s, arm) : vw2_ilnd_arm_bank(&W->vw2, &key0, arm)) banked = 1;
        if (vw2_ilnd_int_bank(&W->vw2, &key0, tk_nf, nf)) banked = 1;
        if (nf == 2 && !wpin && vw2_ilnd_int_bank(&W->vw2, &key0, tk_nsw, d->nsw)) banked = 1;
        if (arm == 3 && vw2_ilnd_int_bank(&W->vw2, &key0, tk_wl, wl)) banked = 1;
        if (vw2_ilcol_forms_rebank(&W->vw2, &key0, forms0)) banked = 1;
        if (banked)
            _vw2_persist(W, cfg);
    }
    /* ═══ THE THREADED VERDICT at the plan's T (phase 3): pin > the banked (cmt, cmts, cmtf, cmtw,
     * cmtp) on THIS T's row > the race of serial (small cubes) + plane x structures x forms x teams.
     * The strips need every structure's clones and the workers' scratches; a structure whose clones
     * fail is out; no engaging arm = serial, banked like a yes. ═══════════════════════════════ */
    d->mt = 0; d->mts = arm; d->mtf = nf; d->mtcut = cut; d->mtwl = wl; d->ptw = 0;
    if (nthr > 1)
    {
        const char *mpin = getenv("VFFT_ILNDR_MT"), *ppin = getenv("VFFT_ILNDR_PT");
        char tkb_cmt[20], tkb_cmts[20], tkb_cmtf[20], tkb_cmtw[20], tkb_cmtp[20], tkb_cmtt[20];   /* door 2: the _p set */
        const char *tk_cmt = tkb_cmt, *tk_cmts = tkb_cmts, *tk_cmtf = tkb_cmtf, *tk_cmtw = tkb_cmtw, *tk_cmtp = tkb_cmtp;
        const char *tk_cmtt = tkb_cmtt;   /* the direction's own marker: the row is direction-shared */
        snprintf(tkb_cmt, sizeof tkb_cmt, "%s%s", c2r ? "cmt_c2r" : "cmt", ip2 ? "_p" : "");
        snprintf(tkb_cmts, sizeof tkb_cmts, "%s%s", c2r ? "cmts_c2r" : "cmts", ip2 ? "_p" : "");
        snprintf(tkb_cmtf, sizeof tkb_cmtf, "%s%s", c2r ? "cmtf_c2r" : "cmtf", ip2 ? "_p" : "");
        snprintf(tkb_cmtw, sizeof tkb_cmtw, "cmtw_c2r%s", ip2 ? "_p" : "");
        snprintf(tkb_cmtp, sizeof tkb_cmtp, "%s%s", c2r ? "cmtp_c2r" : "cmtp", ip2 ? "_p" : "");
        snprintf(tkb_cmtt, sizeof tkb_cmtt, "%s%s", c2r ? "cmtt_c2r" : "cmtt", ip2 ? "_p" : "");
        const int mpinned = (mpin && mpin[0]) || (ppin && ppin[0]);
        int mt_v = -1, mts_v = arm, mtf_v = nf, mtcut_v = cut, mtwl_v = wl, ptw_v = 0, mraced = 0;
        /* the strips can thread wherever the serial strips could; the in-place forms thread except the
         * natural pass; the pay-once needs its pieces; the band needs a legal cut */
        const int strips_mt = strips_ok && d->sscr != NULL;
        if (mpin && mpin[0])
        {
            mt_v = atoi(mpin) == 2 ? 2 : 0;
            if (ppin && ppin[0]) ptw_v = atoi(ppin);
        }
        else if (usable_w && !cfg->recalibrate)
        {
            const int bm = vw2_ilnd_int_lookup(&W->vw2, &key0, tk_cmt);
            if (bm == 0 || bm == 2)
            {
                const int bs = vw2_ilnd_int_lookup(&W->vw2, &key0, tk_cmts), bf = vw2_ilnd_int_lookup(&W->vw2, &key0, tk_cmtf);
                const int row_has = vw2_ilnd_int_lookup(&W->vw2, &key0, tk_cmtt) > 0;
                if (row_has)
                {
                    mt_v = bm;
                    if (bs == 1 || bs == 2 || (c2r && bs == 3)) mts_v = bs;
                    if (bf == 1 || bf == 2) mtf_v = bf;
                    if (mts_v == 3)
                    {
                        int s2;
                        mtwl_v = vw2_ilnd_int_lookup(&W->vw2, &key0, tk_cmtw);
                        mtcut_v = 0;
                        for (s2 = 1; s2 < d->ax0.nst; s2++)
                            if (d->ax0.L[s2] == mtwl_v) { mtcut_v = s2; break; }
                        if (!mtcut_v) mt_v = 0;
                    }
                    ptw_v = vw2_ilnd_int_lookup(&W->vw2, &key0, tk_cmtp);
                }
            }
        }
        if (mt_v == 2)
        {   /* the banked / pinned threaded verdict: its clones, else serial */
            const int can = (mts_v == 2 && !payonce_ok) || (mtf_v == 2 && !strips_mt) ||
                            (mts_v == 1 && mtf_v == 1 && !c2r && d->ax0n_on) ? 0 : 1;
            if (!can || !_ilndr_build_clones(d, cfg, nthr))
            {
                if (!mpinned)
                    _vfft_warn("ilndr: the banked threaded structure cannot be served at %dx%dx%d -- serial", N1, N2, N3);
                mt_v = 0;
            }
        }
        else if (mt_v < 0)
        {   /* the race */
            const size_t RN = (size_t)N1 * (size_t)N2 * (size_t)N3, CN = 2 * (size_t)N1 * d->plane;
            const size_t nin = ip ? CN : (c2r ? CN : RN), nout = ip ? CN : (c2r ? RN : CN);   /* in place: the one volume */
            double *in = (double *)vfft_aligned_alloc((nin + 8) * sizeof(double));
            double *out = ip ? in : (double *)vfft_aligned_alloc((nout + 8) * sizeof(double));
            double *mseed = ip ? (double *)vfft_aligned_alloc((nin + 8) * sizeof(double)) : NULL;   /* the volume re-laid before every sample */
            _il2d_ip_reset_t mrs;
            _ilndr_mt_ctx_t cx[24];
            vfft_race_arm_t arms[24];
            double ns[24];
            int na = 0, a, best = -1, reps, rounds;
            const int hw = _ilndr_half_team(d, nthr);
            const size_t cb = (c2r ? CN : RN) * sizeof(double);
            const int serial_ok = vfft_policy_ilnd_mt_serial_arm(cb > (size_t)0x7fffffff ? 0x7fffffffL : (long)cb);
            const int have_clones = _ilndr_build_clones(d, cfg, nthr) > 0;
            if (in && out && have_clones && (!ip || mseed))
            {
                size_t j;
                double t0;
                unsigned sd = 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 10) ^ ((unsigned)N3 << 20);
                for (j = 0; j < nin; j++) { sd = sd * 1664525u + 1013904223u; in[j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5; }
                if (ip)
                {
                    memcpy(mseed, in, (nin + 8) * sizeof(double));
                    mrs.p = in; mrs.seed = mseed; mrs.n = nin + 8;
                }
                else
                    memset(out, 0, (nout + 8) * sizeof(double));
                /* reps from one serial timing (~20 ms of serial-equivalent work per sample): a worker's
                 * cache partition settles over the first executes, and single-execute samples time the
                 * transient; at least 48 timed executes per arm over 3..15 rounds (the c2c law) */
                d->arm = arm; d->nf = nf; d->cut = cut; d->wl = wl;
                if (c2r) _ilndr_execute_c2r(d, in, out); else _ilndr_execute_r2c(d, in, out);
                t0 = vfft_now_ns();
                if (c2r) _ilndr_execute_c2r(d, in, out); else _ilndr_execute_r2c(d, in, out);
                t0 = vfft_now_ns() - t0;
                if (ip)
                    memcpy(in, mseed, (nin + 8) * sizeof(double));   /* the timing runs transformed the volume */
                reps = (int)(20e6 / (t0 > 1.0 ? t0 : 1.0));
                if (reps < 4) reps = 4;
                if (reps > 256) reps = 256;
                if (ip && reps > 32) reps = 32;   /* in place the volume grows a factor N1 N2 N3 per pass: a sample stays finite */
                rounds = (48 + reps - 1) / reps;
                if (rounds < 3) rounds = 3;
                if (rounds > 15) rounds = 15;
#define _ILNDR_MTARM(MT_, A_, F_, C_, W_, P_, NAME_)                                    \
                do { if (na < 24) {                                                     \
                    cx[na].d = d; cx[na].in = in; cx[na].out = out; cx[na].mt = (MT_); cx[na].arm = (A_); \
                    cx[na].nf = (F_); cx[na].cut = (C_); cx[na].wl = (W_); cx[na].ptw = (P_); cx[na].ok = 1; \
                    snprintf(cx[na].name, sizeof cx[na].name, "%s%s", NAME_, (P_) ? "/half" : "");     \
                    arms[na].name = cx[na].name; arms[na].run = _ilndr_mt_arm_run; arms[na].ctx = &cx[na]; na++; \
                } } while (0)
                if (serial_ok)
                    _ILNDR_MTARM(0, arm, nf, cut, wl, 0, "serial");
                {
                    int st, tm;
                    for (tm = 0; tm < 2; tm++)
                    {
                        const int pt = tm ? hw : 0;
                        if (tm && !hw) break;
                        for (st = 1; st <= 2; st++)
                        {
                            if (st == 2 && !payonce_ok) continue;
                            if (strips_mt) _ILNDR_MTARM(2, st, 2, 0, 0, pt, st == 1 ? "plane/child/strips" : "plane/payonce/strips");
                            if (!(st == 1 && !c2r && d->ax0n_on))
                                _ILNDR_MTARM(2, st, 1, 0, 0, pt, st == 1 ? "plane/child" : "plane/payonce");
                        }
                        if (c2r && !ip)
                        {
                            int s2;
                            for (s2 = 1; s2 < d->ax0.nst; s2++)
                                if (d->ax0.L[s2] >= 2 && d->ax0.L[s2] < N1 && N1 / d->ax0.L[s2] >= 2)
                                {
                                    char nm[32];
                                    snprintf(nm, sizeof nm, "plane/band%d", d->ax0.L[s2]);
                                    _ILNDR_MTARM(2, 3, 1, s2, d->ax0.L[s2], pt, nm);
                                }
                        }
                    }
                }
#undef _ILNDR_MTARM
                if (na > 0)
                {
                    vfft_race_proto_t proto = { rounds, reps, VFFT_RACE_MIN, 1, 2, NULL, NULL, 0 }; /* THREADED arms: never paused */
                    if (ip)
                    {   /* the one volume, re-laid before every sample */
                        proto.reset = _il2d_ip_reset;
                        proto.reset_ctx = &mrs;
                    }
                    _vfft_create_race_count++;
                    vfft_race_run(&proto, arms, na, ns);
                    for (a = 0; a < na; a++)
                        if (cx[a].ok && (best < 0 || ns[a] < ns[best]))
                            best = a;
                    mraced = 1;
                    if (log)
                    {
                        fprintf(stderr, "[ilndr] %dx%dx%d %s MT race T=%d reps=%d rounds=%d:", N1, N2, N3, c2r ? "c2r" : "r2c", nthr, reps, rounds);
                        for (a = 0; a < na; a++)
                            fprintf(stderr, " %s=%.0f%s", cx[a].name, ns[a], cx[a].ok ? "" : "(no engage)");
                        fprintf(stderr, " -> %s\n", best >= 0 ? cx[best].name : "serial");
                    }
                }
                if (best >= 0)
                {
                    mt_v = cx[best].mt; mts_v = cx[best].arm; mtf_v = cx[best].nf; mtcut_v = cx[best].cut; mtwl_v = cx[best].wl;
                    ptw_v = cx[best].mt == 2 ? cx[best].ptw : 0;
                }
                else
                    mt_v = 0;
            }
            else
                mt_v = 0;
            vfft_aligned_free(in);
            if (!ip) vfft_aligned_free(out);
            vfft_aligned_free(mseed);
            d->arm = arm; d->nf = nf; d->cut = cut; d->wl = wl;   /* the serial verdict restored */
            if (usable_w && mraced)
            {
                int banked = 0;
                if (vw2_ilcol_row_ensure(&W->vw2, &key0, d->ax0.R, d->ax0.nst)) banked = 1;
                if (vw2_ilnd_int_bank(&W->vw2, &key0, tk_cmt, mt_v)) banked = 1;
                if (vw2_ilnd_int_bank(&W->vw2, &key0, tk_cmtt, nthr)) banked = 1;
                if (vw2_ilnd_int_bank(&W->vw2, &key0, tk_cmts, mts_v)) banked = 1;
                if (vw2_ilnd_int_bank(&W->vw2, &key0, tk_cmtf, mtf_v)) banked = 1;
                if (mts_v == 3 && vw2_ilnd_int_bank(&W->vw2, &key0, tk_cmtw, mtwl_v)) banked = 1;
                if (hw > 0 && vw2_ilnd_int_bank(&W->vw2, &key0, tk_cmtp, ptw_v > 0 ? ptw_v : (N1 < nthr ? N1 : nthr))) banked = 1;
                if (banked)
                    _vw2_persist(W, cfg);
            }
        }
        d->mt = mt_v == 2 ? 2 : 0;
        d->mts = mts_v; d->mtf = mtf_v; d->mtcut = mtcut_v; d->mtwl = mtwl_v;
        d->ptw = (d->mt == 2 && ptw_v >= 2 && ptw_v < (N1 < nthr ? N1 : nthr)) ? ptw_v : 0;
        if (d->mt != 2)
            _ilndr_free_clones(d);
        else if (d->mts != 2 && arm != 2)
        {   /* the threaded verdict and the serial one both leave the pay-once pieces unused */
            if (!c2r) { vfft_aligned_free(d->V); d->V = NULL; }
            if (d->ax1_on) { _il2d_col_free(&d->ax1); d->ax1_on = 0; }
            free(d->nat1); d->nat1 = NULL;
            free(d->neg1); d->neg1 = NULL;
        }
        if (log)
            fprintf(stderr, "[ilndr] %dx%dx%d %s T=%d: %s%s\n", N1, N2, N3, c2r ? "c2r" : "r2c", nthr,
                    d->mt == 2 ? "plane arm" : "serial",
                    d->mt == 2 ? (d->mts == 3 ? "/band" : d->mts == 2 ? "/payonce" : "/child") : "");
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
        fprintf(stderr, "[ilndr] %dx%dx%d %s: %s %s%s (axis 0 chain %d.. %s, plane rows %s)\n", N1, N2, N3, c2r ? "c2r" : "r2c",
                arm == 3 ? "band" : arm == 2 ? "pay-once" : "child", nf == 2 ? "strips" : "in place",
                d->destroy ? ", the input destroyed" : "",
                d->ax0.nst ? d->ax0.R[0] : 0, d->ax0.blu ? "Bluestein" : "",
                (c2r ? d->cb : d->cf)->il2d_rx_on ? "engine" : "route");
    return (vfft_plan)h;
}

#endif /* VFFT_IL_RANK3_FFTND_REAL_IL_H */
