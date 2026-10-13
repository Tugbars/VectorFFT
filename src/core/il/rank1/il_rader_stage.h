/* il_rader_stage.h — the RADER STAGE of the flat chain: a prime radix p above the
 * kernel pool, served as a vectorized Rader convolution over the stage's columns
 * (docs/roadmap/rader_leaf_design.md, 2026-10-13).
 *
 * THE FORM. Stage s of the flat chain runs p-point DFTs on nb blocks of D columns:
 * element (b, l, c) sits at b*L + l*D + c (L = p*D the block span), with the
 * PRE-twiddle w_M^(l*Q_b) on leg l of block b (M = N/D, Q_b the block's natural
 * index; none at the leaf, s = 0). Read per block as a plane of p rows of D
 * complex, the stage is the 2D tier's column geometry, and Rader over the columns
 * is a vector operation with no turn and no per-column call:
 *
 *   X_0        = sum_l y_l                      y_l = x_l . w^(l Q)  (y_0 = x_0)
 *   a_q        = y_{g^q}                        q = 0 .. p-2
 *   b_q        = sum_m a_m . omega-conv         the cyclic convolution of length p-1
 *   X_{g^-q}   = x_0 + b_q
 *
 * g a primitive root mod p. The convolution is the inner column chain of length
 * p-1 forward (il2d_cols.h's column kinds, tables and loop, in place in a window
 * scratch), a row multiply by the kernel (the forward chain of w^(g^-q) scaled
 * by 1/(p-1), in the chain's own comb order), and the chain backward. The
 * permutations and the kernel are what il_prime.h builds for the whole-N Rader.
 *
 * THE EXECUTE. Column windows [c0, c1) of W columns: the gather copies the
 * window's rows g^q (twiddled where the stage has twiddles) into the scratch
 * and sums the DC row, the two chain passes and the multiply run in the scratch,
 * the scatter adds x_0 and writes rows g^-q (the natural redirection at the last
 * stage). The data is read once and written once; the scratch stays in cache.
 * A window is the unit: the MT column split hands each worker a column range
 * and its own scratch slot. In place is safe by construction (a window's rows
 * are read whole before any of them is written).
 *
 * THE INNER CHAIN. p-1 from the column pool (the mids t2c, the last n1c, the
 * n1c-only radices legal as the closing stage); when p-1 carries its own prime
 * factor above the pool, that prime takes stage 0 of the inner chain as a nested
 * Rader leaf over the scratch plane (ONE level: its own inner must chain from the
 * pool). The chain here is the pool's greedy pick; the planner's race over inner
 * chains, forms and widths comes with the planner admission (phase 2).
 *
 * The backward direction (the natural class's conjugate pipeline) is the same
 * stage with the conjugate kernel, the permutations swapped and the twiddles
 * conjugated, as _ilprime_exec_rader does. The scrambled class's transposed
 * pipeline is not provided: a chain with a Rader stage has scr_ok = 0.
 *
 * INCLUSION: after il/rank2/il2d_cols.h (the column machinery) and il_flatdit.h
 * (the plan); the flat DIT reaches the engine through the hooks installed at the
 * bottom, so a translation unit without this header builds no Rader stage. */
#ifndef VFFT_IL_RADER_STAGE_H
#define VFFT_IL_RADER_STAGE_H

#include <stdlib.h>
#include <string.h>
#include "numtheory.h"   /* vfft_is_prime, vfft_powmod, vfft_primitive_root */

#define VFFT_ILRD_POOL_MAX 47     /* the flat kernel pool's largest radix: above it, Rader */
#define VFFT_ILRD_MAX_NST 8
#define VFFT_ILRD_MAX_SLOTS 64    /* scratch slots: the caller + the pool's workers */
#define VFFT_ILRD_SCR_COMPLEX 4096 /* the window scratch budget: (p-1) * W complex <= this */
#define VFFT_ILRD_W_MAX 64
#define VFFT_ILRD_W_MIN 2

typedef struct vfft_ilrd_s {
    int p;                 /* the prime radix */
    size_t D, L, nb, ncol; /* legs stride, block span (p*D), blocks, columns = nb*D */
    int last;              /* 1 = the last stage: the natural redirection */
    const size_t *natbase; /* last: block -> natural base (the plan's table) */
    size_t nstride;        /* last: the output leg stride N/p */
    int *gpow, *ginvpow;   /* g^q and g^-q mod p, q in [0, p-1) */
    double *omf, *omb;     /* the comb-order kernels at p-1 (fwd / bwd), 1/(p-1) baked */
    double *twf, *twb;     /* the stage's twiddles (fwd / conj), p complex per set; NULL at a leaf */
    int twmode;            /* 0 none; 1 the flat chain's PRE-twiddle per block (set b = block b);
                            * 2 the DIF column chain's stage-0 twiddle per column group of twrun
                            *   columns (set d = c / twrun): POST on the forward, PRE-conj on the
                            *   reverse (the nested leaf inside an inner chain) */
    size_t twrun;
    /* the inner chain at p-1 */
    int nst, Rs[VFFT_ILRD_MAX_NST], Ls[VFFT_ILRD_MAX_NST];
    vfft_il2p_fn ff[VFFT_ILRD_MAX_NST], fb[VFFT_ILRD_MAX_NST];
    double *tf[VFFT_ILRD_MAX_NST], *tb[VFFT_ILRD_MAX_NST];
    struct vfft_ilrd_s *nest;   /* the inner's own Rader leaf at Rs[0], or NULL */
    size_t W;                   /* the window width in columns */
    double *scr[VFFT_ILRD_MAX_SLOTS];
    int nslot;
} vfft_ilrd_t;

/* ── row primitives (SIMD along the column axis) ─────────────────────── */
static inline void _ilrd_row_add(double *dst, const double *src, size_t n)
{
    size_t k = 0;
    const size_t n2 = 2 * n;
#if defined(__AVX2__)
    for (; k + 4 <= n2; k += 4)
        _mm256_storeu_pd(dst + k, _mm256_add_pd(_mm256_loadu_pd(dst + k), _mm256_loadu_pd(src + k)));
#endif
    for (; k < n2; k++) dst[k] += src[k];
}
static inline void _ilrd_row_sum(double *dst, const double *a, const double *b, size_t n)
{
    size_t k = 0;
    const size_t n2 = 2 * n;
#if defined(__AVX2__)
    for (; k + 4 <= n2; k += 4)
        _mm256_storeu_pd(dst + k, _mm256_add_pd(_mm256_loadu_pd(a + k), _mm256_loadu_pd(b + k)));
#endif
    for (; k < n2; k++) dst[k] = a[k] + b[k];
}

/* ── the inner chain: the column pool's greedy pick ──────────────────── */
static int _ilrd_closing_ok(int r)
{   /* n1c-only radices legal as the CLOSING stage alone (il2d_cols.h's pool) */
    static const int C[] = { 26, 22, 14, 12, 10, 6, 2 };
    size_t i;
    for (i = 0; i < sizeof C / sizeof C[0]; i++)
        if (C[i] == r) return vfft_il2p_n1c_fn(r, 0) && vfft_il2p_n1c_fn(r, 1);
    return 0;
}
static int _ilrd_largest_prime_factor(int n)
{
    int p = 2, m = 1;
    while ((long)p * p <= n) {
        while (n % p == 0) { m = p; n /= p; }
        p++;
    }
    return n > 1 ? n : m;
}
/* Rs[0..nst) for M = p-1; *nestp = the prime that takes stage 0 as a nested
 * leaf (0 = none). 0 = no chain. */
static int _ilrd_pick_chain(int M, int *Rs, int *nst, int *nestp)
{
    static const int POOL[] = { 64, 32, 16, 8, 4, 47, 43, 41, 37, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3 };
    const int npool = (int)(sizeof POOL / sizeof POOL[0]);
    int rem = M, n = 0, i, q;
    *nestp = 0;
    if (M < 2) return 0;
    q = _ilrd_largest_prime_factor(M);
    if (q > VFFT_ILRD_POOL_MAX) {   /* the nested leaf takes stage 0 */
        if (!vfft_is_prime(q)) return 0;
        Rs[n++] = q; rem /= q; *nestp = q;
    }
    while (rem > 1 && n < VFFT_ILRD_MAX_NST) {
        int hit = 0;
        if (_ilrd_closing_ok(rem)) { Rs[n++] = rem; rem = 1; break; }
        for (i = 0; i < npool; i++) {
            const int r = POOL[i];
            if (rem % r) continue;
            if (rem == r) {
                if (!vfft_il2p_n1c_fn(r, 0) || !vfft_il2p_n1c_fn(r, 1)) continue;
            } else if (!vfft_il2p_t2c_fn(r, 0) || !vfft_il2p_t2c_fn(r, 1)) continue;
            Rs[n++] = r; rem /= r; hit = 1; break;
        }
        if (!hit) break;
    }
    if (rem != 1 || n < 1) return 0;
    *nst = n;
    return 1;
}

static void _ilrd_free(vfft_ilrd_t *rd)
{
    int i;
    if (!rd) return;
    if (rd->nest) _ilrd_free(rd->nest);
    free(rd->gpow); free(rd->ginvpow);
    vfft_aligned_free(rd->omf); vfft_aligned_free(rd->omb);
    vfft_aligned_free(rd->twf); vfft_aligned_free(rd->twb);
    for (i = 0; i < VFFT_ILRD_MAX_NST; i++) { vfft_aligned_free(rd->tf[i]); vfft_aligned_free(rd->tb[i]); }
    for (i = 0; i < VFFT_ILRD_MAX_SLOTS; i++) vfft_aligned_free(rd->scr[i]);
    free(rd);
}

static size_t _ilrd_scr_doubles(const vfft_ilrd_t *rd)
{
    return 2 * ((size_t)(rd->p - 1) + 2) * rd->W;   /* the conv plane + the x0 row + the DC row */
}
/* the scratch of slot k (and the nested leaf's), allocated once; 1 = ok */
static int _ilrd_slot(vfft_ilrd_t *rd, int k)
{
    if (k < 0 || k >= VFFT_ILRD_MAX_SLOTS) return 0;
    if (!rd->scr[k]) {
        rd->scr[k] = (double *)vfft_aligned_alloc(_ilrd_scr_doubles(rd) * sizeof(double));
        if (!rd->scr[k]) return 0;
        memset(rd->scr[k], 0, _ilrd_scr_doubles(rd) * sizeof(double));
        if (k + 1 > rd->nslot) rd->nslot = k + 1;
    }
    return rd->nest ? _ilrd_slot(rd->nest, k) : 1;
}

static void _ilrd_window(const vfft_ilrd_t *rd, const double *src, double *dst,
                         size_t c0, size_t c1, int bwd, int slot);

/* the inner chain over the scratch plane sc (row stride rn, the first w lanes
 * live), forward or reverse; stage 0 is the nested leaf where one exists. */
static void _ilrd_inner_run(const vfft_ilrd_t *rd, double *sc, size_t rn, size_t w, int reverse, int slot)
{
    const int M = rd->p - 1;
    int si;
    for (si = 0; si < rd->nst; si++) {
        const int s = reverse ? rd->nst - 1 - si : si;
        const int R = rd->Rs[s], D = rd->Ls[s] / R;
        int b;
        if (s == 0 && rd->nest) {
            /* the nested leaf: the whole scratch row is its column space (the
             * lanes past w carry stale values and are never read back), in
             * its own windows */
            const vfft_ilrd_t *ne = rd->nest;
            size_t k;
            for (k = 0; k < ne->ncol; k += ne->W) {
                const size_t e = (k + ne->W < ne->ncol) ? k + ne->W : ne->ncol;
                _ilrd_window(ne, sc, sc, k, e, reverse, slot);
            }
            continue;
        }
        for (b = 0; b < M / rd->Ls[s]; b++) {
            const size_t off = 2 * (size_t)b * (size_t)rd->Ls[s] * rn;
            const vfft_il2p_fn fn = reverse ? rd->fb[s] : rd->ff[s];
            const double *tab = reverse ? rd->tb[s] : rd->tf[s];
            if (D == 1) fn(sc + off, NULL, sc + off, NULL, NULL, NULL, rn, 0, rn, 0, w);
            else fn(sc + off, NULL, sc + off, NULL, tab, NULL, (size_t)D * rn, rn, (size_t)D * rn, (size_t)D, w);
        }
    }
}

/* ONE WINDOW [c0, c1) of the stage's column space: the Rader convolution over
 * those columns, src -> dst (src may equal dst). */
static void _ilrd_window(const vfft_ilrd_t *rd, const double *src, double *dst,
                         size_t c0, size_t c1, int bwd, int slot)
{
    const int p = rd->p, M = p - 1;
    const int *gat = bwd ? rd->ginvpow : rd->gpow, *sct = bwd ? rd->gpow : rd->ginvpow;
    const double *om = bwd ? rd->omb : rd->omf, *tw = bwd ? rd->twb : rd->twf;
    const size_t D = rd->D, L = rd->L, W = rd->W, w = c1 - c0;
    /* a run = columns of one block, and of one twiddle set where the set is a
     * column group (mode 2); gi = the set's twiddles for this run, or NULL */
    const size_t grp = (rd->twmode == 2 && rd->twrun) ? rd->twrun : 0;
    const int pre = (rd->twmode == 1) || (rd->twmode == 2 && bwd);   /* twiddle on the way in */
    const int post = (rd->twmode == 2 && !bwd);                        /* twiddle on the way out */
    double *sc = rd->scr[slot], *x0 = sc + 2 * (size_t)M * W, *dc = x0 + 2 * W;
    size_t k;
    int q;
    if (c1 <= c0) return;
#define _ILRD_RUN(k, b, cb) \
    const size_t b = (k) / D; \
    size_t cb = ((b + 1) * D < c1) ? (b + 1) * D : c1; \
    if (grp) { const size_t ge = ((k) / grp + 1) * grp; if (ge < cb) cb = ge; }
    /* gather, per run of the window */
    for (k = c0; k < c1;) {
        _ILRD_RUN(k, b, cb)
        const size_t n = cb - k, c = k - b * D, o = k - c0;
        const double *blk = src + 2 * (b * L + c);
        const double *gi = !tw ? 0 : rd->twmode == 1 ? tw + 2 * (size_t)p * b : tw + 2 * (size_t)p * (k / grp);
        memcpy(x0 + 2 * o, blk, 2 * n * sizeof(double));
        memcpy(dc + 2 * o, blk, 2 * n * sizeof(double));
        for (q = 0; q < M; q++) {
            const int l = gat[q];
            const double *row = blk + 2 * (size_t)l * D;
            double *d = sc + 2 * ((size_t)q * W + o);
            if (pre && gi) _il2d_row_cmul(d, row, gi[2 * l], gi[2 * l + 1], n);
            else memcpy(d, row, 2 * n * sizeof(double));
            _ilrd_row_add(dc + 2 * o, d, n);
        }
        k = cb;
    }
    _ilrd_inner_run(rd, sc, W, w, 0, slot);
    for (q = 0; q < M; q++)
        _il2d_row_cmul(sc + 2 * (size_t)q * W, sc + 2 * (size_t)q * W, om[2 * q], om[2 * q + 1], w);
    _ilrd_inner_run(rd, sc, W, w, 1, slot);
    /* scatter */
    for (k = c0; k < c1;) {
        _ILRD_RUN(k, b, cb)
        const size_t n = cb - k, c = k - b * D, o = k - c0;
        double *ob = rd->last ? dst + 2 * (rd->natbase[b] + c) : dst + 2 * (b * L + c);
        const size_t os = rd->last ? rd->nstride : D;
        const double *gi = (post && tw) ? tw + 2 * (size_t)p * (k / grp) : 0;
        memcpy(ob, dc + 2 * o, 2 * n * sizeof(double));
        for (q = 0; q < M; q++) {
            double *orow = ob + 2 * (size_t)sct[q] * os;
            _ilrd_row_sum(orow, x0 + 2 * o, sc + 2 * ((size_t)q * W + o), n);
            if (gi) _il2d_row_cmul(orow, orow, gi[2 * sct[q]], gi[2 * sct[q] + 1], n);
        }
        k = cb;
    }
#undef _ILRD_RUN
}

/* the DIF column chain's stage-0 twiddles for a nested leaf: set d (the column
 * group d = c / twrun, d in [0, Dp)), leg l: w_span^(l d), fwd (c, -s) / conj */
static int _ilrd_dif_tw(vfft_ilrd_t *rd, int span, size_t Dp, size_t twrun)
{
    size_t d;
    int l;
    const size_t P = (size_t)rd->p;
    rd->twf = (double *)vfft_aligned_alloc(2 * Dp * P * sizeof(double));
    rd->twb = (double *)vfft_aligned_alloc(2 * Dp * P * sizeof(double));
    if (!rd->twf || !rd->twb) return 0;
    for (d = 0; d < Dp; d++)
        for (l = 0; l < rd->p; l++) {
            double c, sn;
            vfft_cs2pi_exact((long long)(((size_t)l * d) % (size_t)span), (long long)span, &c, &sn);
            rd->twf[2 * (d * P + (size_t)l)] = c;  rd->twf[2 * (d * P + (size_t)l) + 1] = -sn;
            rd->twb[2 * (d * P + (size_t)l)] = c;  rd->twb[2 * (d * P + (size_t)l) + 1] = sn;
        }
    rd->twmode = 2;
    rd->twrun = twrun;
    return 1;
}

/* ── create ──────────────────────────────────────────────────────────── */
static vfft_ilrd_t *_ilrd_create(int pr, size_t D, size_t L, size_t nb, int depth);

/* the inner chain's kernels and tables; the nested leaf where the pick names one */
static int _ilrd_build_inner(vfft_ilrd_t *rd, int depth)
{
    const int M = rd->p - 1;
    int nestp = 0, s;
    if (!_ilrd_pick_chain(M, rd->Rs, &rd->nst, &nestp)) return 0;
    if (nestp) {
        if (depth >= 1) return 0;   /* one level */
        /* stage 0 = the nested leaf over the scratch plane: legs at stride
         * (M/nestp) * W, that many columns, one block, no twiddles */
        rd->nest = _ilrd_create(nestp, (size_t)(M / nestp) * rd->W, (size_t)M * rd->W, 1, depth + 1);
        if (!rd->nest) return 0;
        /* stage 0 of a DIF column chain carries the span's twiddles on its outputs */
        if (!_ilrd_dif_tw(rd->nest, M, (size_t)(M / nestp), rd->W)) return 0;
        if (rd->nst > 1 && !_il2d_resolve(rd->Rs + 1, rd->nst - 1, rd->ff + 1, rd->fb + 1)) return 0;
        rd->ff[0] = rd->fb[0] = 0;
    } else if (!_il2d_resolve(rd->Rs, rd->nst, rd->ff, rd->fb)) return 0;
    if (_il2d_build_tables(M, rd->nst, rd->Rs, rd->Ls, rd->tf, rd->tb)) return 0;
    for (s = 0; s < rd->nst; s++) if (!rd->Ls[s]) return 0;
    return 1;
}

static vfft_ilrd_t *_ilrd_create(int pr, size_t D, size_t L, size_t nb, int depth)
{
    vfft_ilrd_t *rd;
    const int M = pr - 1;
    size_t W;
    int q;
    if (pr <= VFFT_ILRD_POOL_MAX || !vfft_is_prime(pr)) return 0;
    rd = (vfft_ilrd_t *)calloc(1, sizeof *rd);
    if (!rd) return 0;
    rd->p = pr; rd->D = D; rd->L = L; rd->nb = nb; rd->ncol = nb * D;
    W = VFFT_ILRD_SCR_COMPLEX / (size_t)M;
    if (W > VFFT_ILRD_W_MAX) W = VFFT_ILRD_W_MAX;
    if (W > rd->ncol) W = rd->ncol;
    W &= ~(size_t)1;
    if (W < VFFT_ILRD_W_MIN) W = VFFT_ILRD_W_MIN;
    rd->W = W;
    rd->gpow = (int *)malloc((size_t)M * sizeof(int));
    rd->ginvpow = (int *)malloc((size_t)M * sizeof(int));
    rd->omf = (double *)vfft_aligned_alloc(2 * (size_t)M * sizeof(double));
    rd->omb = (double *)vfft_aligned_alloc(2 * (size_t)M * sizeof(double));
    if (!rd->gpow || !rd->ginvpow || !rd->omf || !rd->omb) { _ilrd_free(rd); return 0; }
    {
        const int g = vfft_primitive_root(pr), ginv = (int)vfft_powmod(g, M - 1, pr);
        long long gp = 1, gip = 1;
        for (q = 0; q < M; q++) {
            rd->gpow[q] = (int)gp; rd->ginvpow[q] = (int)gip;
            gp = gp * g % pr; gip = gip * ginv % pr;
        }
    }
    if (!_ilrd_build_inner(rd, depth) || !_ilrd_slot(rd, 0)) { _ilrd_free(rd); return 0; }
    {   /* the kernels: the forward chain of w^(perm[m]) in comb order, scaled. The
         * sequence is one column (row stride 1): a nested leaf is run through a
         * twin built for that width (the same chain, the same values, its own
         * scratch), since the serving nest's geometry is the window's. */
        double *za = (double *)vfft_aligned_alloc(2 * (size_t)M * sizeof(double));
        vfft_ilrd_t *serving = rd->nest, *one = 0;
        int d, m;
        if (!za) { _ilrd_free(rd); return 0; }
        if (serving) {
            one = _ilrd_create(serving->p, (size_t)(M / serving->p), (size_t)M, 1, depth + 1);
            if (!one || !_ilrd_dif_tw(one, M, (size_t)(M / serving->p), 1)) {
                _ilrd_free(one); vfft_aligned_free(za); _ilrd_free(rd); return 0;
            }
            rd->nest = one;
        }
        for (d = 0; d < 2; d++) {
            const int *perm = d ? rd->gpow : rd->ginvpow;
            const double sign = d ? 1.0 : -1.0, inv = 1.0 / (double)M;
            double *om = d ? rd->omb : rd->omf;
            for (m = 0; m < M; m++) {
                double c, sn;   /* a = sign*2*pi*perm/p: sin(a) = sign*sn */
                vfft_cs2pi_exact((long long)perm[m], (long long)pr, &c, &sn);
                za[2 * m] = c; za[2 * m + 1] = sign * sn;
            }
            _ilrd_inner_run(rd, za, 1, 1, 0, 0);
            for (m = 0; m < 2 * M; m++) om[m] = za[m] * inv;
        }
        rd->nest = serving;
        if (one) _ilrd_free(one);
        vfft_aligned_free(za);
    }
    return rd;
}

/* ── the flat DIT's hooks ────────────────────────────────────────────── */
static int _ilrd_build_hook_fn(vfft_ilfd_plan_t *p, int s)
{
    const size_t D = p->D[s], L = (size_t)p->R[s] * D, nb = p->nblk[s];
    vfft_ilrd_t *rd = _ilrd_create(p->R[s], D, L, nb, 0);
    if (!rd) return 0;
    if (s >= 1) {   /* the pre-twiddles: per block, p complex, modulus N/D */
        const size_t M = (size_t)p->N / D;
        size_t b;
        int l;
        rd->twf = (double *)vfft_aligned_alloc(2 * nb * (size_t)p->R[s] * sizeof(double));
        rd->twb = (double *)vfft_aligned_alloc(2 * nb * (size_t)p->R[s] * sizeof(double));
        if (!rd->twf || !rd->twb) { _ilrd_free(rd); return 0; }
        rd->twmode = 1;
        for (b = 0; b < nb; b++) {
            const size_t Q = _ilfd_block_Q(p, s, b);
            for (l = 0; l < p->R[s]; l++) {
                double c, sn;   /* fwd w^(lQ) = (c, -sn); bwd its conjugate */
                vfft_cs2pi_exact((long long)(((size_t)l * (Q % M)) % M), (long long)M, &c, &sn);
                rd->twf[2 * (b * (size_t)p->R[s] + l)] = c;
                rd->twf[2 * (b * (size_t)p->R[s] + l) + 1] = -sn;
                rd->twb[2 * (b * (size_t)p->R[s] + l)] = c;
                rd->twb[2 * (b * (size_t)p->R[s] + l) + 1] = sn;
            }
        }
    }
    p->rd[s] = rd;
    p->has_rd = 1;
    p->scr_ok = 0;
    return 1;
}
/* the last stage's redirection is known only after the plan's natbase exists:
 * the create binds it (il_flatdit.h) through this */
static void _ilrd_last_hook_fn(vfft_ilfd_plan_t *p, int s)
{
    vfft_ilrd_t *rd = (vfft_ilrd_t *)p->rd[s];
    if (!rd) return;
    rd->last = (s == p->K - 1) && !p->scr;
    rd->natbase = p->natbase;
    rd->nstride = (size_t)p->N / (size_t)p->R[s];
}
static void _ilrd_free_hook_fn(void *rd) { _ilrd_free((vfft_ilrd_t *)rd); }
static void _ilrd_run_hook_fn(const vfft_ilfd_plan_t *p, const vfft_ilfd_call_t *c,
                              size_t c0, size_t c1, const double *zin, double *zout)
{
    const vfft_ilrd_t *rd = (const vfft_ilrd_t *)c->rd;
    const double *src = (c->in_sel == _ILFD_ZIN) ? zin : p->stg;
    double *dst = (c->out_sel == _ILFD_ZOUT) ? zout : p->stg;
    size_t k;
    for (k = c0; k < c1; k += rd->W) {
        const size_t e = (k + rd->W < c1) ? k + rd->W : c1;
        _ilrd_window(rd, src, dst, k, e, c->rdir, c->rdslot);
    }
}
static int _ilrd_mt_hook_fn(vfft_ilfd_plan_t *p, int T)
{
    int s, k;
    for (s = 0; s < p->K; s++)
        if (p->rd[s])
            for (k = 1; k < T; k++)
                if (!_ilrd_slot((vfft_ilrd_t *)p->rd[s], k)) return 0;
    return 1;
}

/* the installation: the tentative definitions in il_flatdit.h take these */
static int (*_ilfd_rd_build_hook)(vfft_ilfd_plan_t *, int) = _ilrd_build_hook_fn;
static void (*_ilfd_rd_last_hook)(vfft_ilfd_plan_t *, int) = _ilrd_last_hook_fn;
static void (*_ilfd_rd_free_hook)(void *) = _ilrd_free_hook_fn;
static void (*_ilfd_rd_run_hook)(const vfft_ilfd_plan_t *, const vfft_ilfd_call_t *, size_t, size_t,
                                 const double *, double *) = _ilrd_run_hook_fn;
static int (*_ilfd_rd_mt_hook)(vfft_ilfd_plan_t *, int) = _ilrd_mt_hook_fn;

/* ── THE TOOL DOOR (bench-visible, external linkage in the library's TU):
 * a flat chain by its radices, with Rader stages wherever a radix is a prime
 * above the pool; the probes race it before the planner admits it. ───── */
void *vfft_tool_ilfd_chain(int N, const int *R, int K)
{
    return vfft_ilfd_create_chain(N, R, K);
}
void vfft_tool_ilfd_exec(void *plan, const double *zin, double *zout, int bwd)
{
    if (bwd) vfft_ilfd_execute_bwd((vfft_ilfd_plan_t *)plan, zin, zout);
    else vfft_ilfd_execute_fwd((vfft_ilfd_plan_t *)plan, zin, zout);
}
void vfft_tool_ilfd_destroy(void *plan) { vfft_ilfd_destroy((vfft_ilfd_plan_t *)plan); }
/* the chain threaded over T workers (the BLOCKS form, every stage by units):
 * 1 = ran threaded, 0 = declined (the caller runs serial) */
int vfft_tool_ilfd_exec_mt(void *plan, const double *zin, double *zout, int bwd, int T)
{
    vfft_ilfd_plan_t *p = (vfft_ilfd_plan_t *)plan;
    if (T < 2) return 0;
    thread_pool_ensure(T);
    p->mt = 1; p->mt_t = T;
    if (!vfft_ilfd_mt_bind(p, T)) return 0;
    return vfft_ilfd_execute_mt(p, zin, zout, bwd);
}
/* the chain's Rader stages, described: "s0:p=191 W=20 inner=19.10" per stage */
int vfft_tool_ilfd_rd_desc(const void *plan, char *buf, size_t sz)
{
    const vfft_ilfd_plan_t *p = (const vfft_ilfd_plan_t *)plan;
    size_t n = 0;
    int s, i;
    buf[0] = 0;
    for (s = 0; s < p->K; s++) {
        const vfft_ilrd_t *rd = (const vfft_ilrd_t *)p->rd[s];
        if (!rd) continue;
        n += (size_t)snprintf(buf + n, sz > n ? sz - n : 0, "%ss%d:p=%d W=%zu inner=", n ? " " : "", s, rd->p, rd->W);
        for (i = 0; i < rd->nst; i++)
            n += (size_t)snprintf(buf + n, sz > n ? sz - n : 0, "%s%d%s", i ? "." : "", rd->Rs[i], (i == 0 && rd->nest) ? "n" : "");
        if (rd->nest) {
            n += (size_t)snprintf(buf + n, sz > n ? sz - n : 0, "(nest W=%zu inner=", rd->nest->W);
            for (i = 0; i < rd->nest->nst; i++)
                n += (size_t)snprintf(buf + n, sz > n ? sz - n : 0, "%s%d", i ? "." : "", rd->nest->Rs[i]);
            n += (size_t)snprintf(buf + n, sz > n ? sz - n : 0, ")");
        }
    }
    return (int)n;
}

#endif /* VFFT_IL_RADER_STAGE_H */
