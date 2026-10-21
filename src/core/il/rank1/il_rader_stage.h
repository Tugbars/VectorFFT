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
#include "dft_ld.h"      /* the Rader kernel in extended precision */

#define VFFT_ILRD_POOL_MAX 47     /* the flat kernel pool's largest radix: above it, Rader */
#define VFFT_ILRD_MAX_NST 8
#define VFFT_ILRD_MAX_SLOTS 64    /* scratch slots: the caller + the pool's workers */
#define VFFT_ILRD_SCR_COMPLEX 4096 /* the window scratch budget: (p-1) * W complex <= this */
#define VFFT_ILRD_W_MAX 64
#define VFFT_ILRD_W_MIN 2

/* a direction tag of the column stage: what the walkers find in a stage-0 table slot
 * (a block of a table set, freed with its tables; it names the engine, never owns it) */
typedef struct { struct vfft_ilrd_s *rd; int bwd; } _ilrd_ctab_t;

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
    /* THE COLUMN STAGE (the 2D tier's stage 0, section 13): Dp = the DIF stage's digit
     * count N1 / p (digit d's legs at rows l * Dp + d) */
    size_t Dp;
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

/* ONE WINDOW as a list of RUNS: a run is n contiguous columns sharing one
 * source base (row 0 of the first column; legs ss apart), one destination
 * base (legs ds apart) and one twiddle set gi (p complex, NULL = none). The
 * runs' columns fill the window's W lanes in order: gathered into the scratch
 * side by side, the inner chain and the multiply run ONCE over all of them,
 * the scatter returns each run to its own rows. The flat chain's windows and
 * the column stage's are both lists of such runs. */
typedef struct { const double *s0; double *d0; size_t ss, ds, n; const double *gi; } _ilrd_run_t;
static void _ilrd_window_runs(const vfft_ilrd_t *rd, const _ilrd_run_t *runs, int nr, int bwd, int slot)
{
    const int p = rd->p, M = p - 1;
    const int *gat = bwd ? rd->ginvpow : rd->gpow, *sct = bwd ? rd->gpow : rd->ginvpow;
    const double *om = bwd ? rd->omb : rd->omf;
    const int pre = (rd->twmode == 1) || (rd->twmode == 2 && bwd);   /* twiddle on the way in */
    const int post = (rd->twmode == 2 && !bwd);                        /* twiddle on the way out */
    const size_t W = rd->W;
    double *sc = rd->scr[slot], *x0 = sc + 2 * (size_t)M * W, *dc = x0 + 2 * W;
    size_t o = 0, w = 0;
    int r, q;
    (void)p;
    for (r = 0; r < nr; r++) w += runs[r].n;
    if (!w) return;
    /* gather, run by run */
    for (r = 0, o = 0; r < nr; o += runs[r].n, r++) {
        const _ilrd_run_t *u = &runs[r];
        memcpy(x0 + 2 * o, u->s0, 2 * u->n * sizeof(double));
        memcpy(dc + 2 * o, u->s0, 2 * u->n * sizeof(double));
        for (q = 0; q < M; q++) {
            const int l = gat[q];
            const double *row = u->s0 + 2 * (size_t)l * u->ss;
            double *d = sc + 2 * ((size_t)q * W + o);
            if (pre && u->gi) _il2d_row_cmul(d, row, u->gi[2 * l], u->gi[2 * l + 1], u->n);
            else memcpy(d, row, 2 * u->n * sizeof(double));
            _ilrd_row_add(dc + 2 * o, d, u->n);
        }
    }
    _ilrd_inner_run(rd, sc, W, w, 0, slot);
    for (q = 0; q < M; q++)
        _il2d_row_cmul(sc + 2 * (size_t)q * W, sc + 2 * (size_t)q * W, om[2 * q], om[2 * q + 1], w);
    _ilrd_inner_run(rd, sc, W, w, 1, slot);
    /* scatter, run by run */
    for (r = 0, o = 0; r < nr; o += runs[r].n, r++) {
        const _ilrd_run_t *u = &runs[r];
        memcpy(u->d0, dc + 2 * o, 2 * u->n * sizeof(double));
        for (q = 0; q < M; q++) {
            double *orow = u->d0 + 2 * (size_t)sct[q] * u->ds;
            _ilrd_row_sum(orow, x0 + 2 * o, sc + 2 * ((size_t)q * W + o), u->n);
            if (post && u->gi) _il2d_row_cmul(orow, orow, u->gi[2 * sct[q]], u->gi[2 * sct[q] + 1], u->n);
        }
    }
}

/* ONE WINDOW [c0, c1) of the flat stage's column space: the Rader convolution
 * over those columns, src -> dst (src may equal dst). A run = the columns of
 * one block, and of one twiddle set where the set is a column group (mode 2). */
static void _ilrd_window(const vfft_ilrd_t *rd, const double *src, double *dst,
                         size_t c0, size_t c1, int bwd, int slot)
{
    const int p = rd->p;
    const double *tw = bwd ? rd->twb : rd->twf;
    const size_t D = rd->D, L = rd->L;
    const size_t grp = (rd->twmode == 2 && rd->twrun) ? rd->twrun : 0;
    _ilrd_run_t runs[VFFT_ILRD_W_MAX + 2];
    int nr = 0;
    size_t k;
    if (c1 <= c0) return;
    for (k = c0; k < c1;) {
        const size_t b = k / D;
        size_t cb = ((b + 1) * D < c1) ? (b + 1) * D : c1;
        const size_t c = k - b * D;
        if (grp) { const size_t ge = (k / grp + 1) * grp; if (ge < cb) cb = ge; }
        runs[nr].s0 = src + 2 * (b * L + c);
        runs[nr].ss = D;
        runs[nr].d0 = rd->last ? dst + 2 * (rd->natbase[b] + c) : dst + 2 * (b * L + c);
        runs[nr].ds = rd->last ? rd->nstride : D;
        runs[nr].n = cb - k;
        runs[nr].gi = 0;
        if (tw && rd->twmode == 1) runs[nr].gi = tw + 2 * (size_t)p * b;
        else if (tw && grp) runs[nr].gi = tw + 2 * (size_t)p * (k / grp);
        nr++;
        k = cb;
        if (nr == VFFT_ILRD_W_MAX + 1) { _ilrd_window_runs(rd, runs, nr, bwd, slot); nr = 0; }
    }
    if (nr) _ilrd_window_runs(rd, runs, nr, bwd, slot);
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
static vfft_ilrd_t *_ilrd_create(int pr, size_t D, size_t L, size_t nb, int depth,
                                 const int *req, int nreq, size_t Wreq);

/* the inner chain's kernels and tables; the nested leaf where the pick (or the
 * request: a radix above the pool at stage 0) names one */
static int _ilrd_build_inner(vfft_ilrd_t *rd, int depth, const int *req, int nreq)
{
    const int M = rd->p - 1;
    int nestp = 0, s;
    if (req && nreq > 0) {
        long prod = 1;
        if (nreq > VFFT_ILRD_MAX_NST) return 0;
        for (s = 0; s < nreq; s++) prod *= req[s];
        if (prod != M) return 0;
        memcpy(rd->Rs, req, sizeof(int) * (size_t)nreq);
        rd->nst = nreq;
        if (req[0] > VFFT_ILRD_POOL_MAX) { if (!vfft_is_prime(req[0])) return 0; nestp = req[0]; }
    } else if (!_ilrd_pick_chain(M, rd->Rs, &rd->nst, &nestp)) return 0;
    if (nestp) {
        if (depth >= 1) return 0;   /* one level */
        if (getenv("VFFT_RD_ONE_LEVEL")) return 0;   /* the test of 2026-10-17: no second Rader level */
        /* stage 0 = the nested leaf over the scratch plane: legs at stride
         * (M/nestp) * W, that many columns, one block, no twiddles */
        rd->nest = _ilrd_create(nestp, (size_t)(M / nestp) * rd->W, (size_t)M * rd->W, 1, depth + 1, 0, 0, 0);
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

/* the default window: the scratch budget over the conv rows, even, capped */
static size_t _ilrd_default_W(int pr, size_t ncol)
{
    size_t W = VFFT_ILRD_SCR_COMPLEX / (size_t)(pr - 1);
    if (W > VFFT_ILRD_W_MAX) W = VFFT_ILRD_W_MAX;
    if (W > ncol) W = ncol;
    W &= ~(size_t)1;
    if (W < VFFT_ILRD_W_MIN) W = VFFT_ILRD_W_MIN;
    return W;
}
static vfft_ilrd_t *_ilrd_create(int pr, size_t D, size_t L, size_t nb, int depth,
                                 const int *req, int nreq, size_t Wreq)
{
    vfft_ilrd_t *rd;
    const int M = pr - 1;
    size_t W;
    int q;
    if (pr <= VFFT_ILRD_POOL_MAX || !vfft_is_prime(pr)) return 0;
    rd = (vfft_ilrd_t *)calloc(1, sizeof *rd);
    if (!rd) return 0;
    rd->p = pr; rd->D = D; rd->L = L; rd->nb = nb; rd->ncol = nb * D;
    W = _ilrd_default_W(pr, rd->ncol);
    if (Wreq >= VFFT_ILRD_W_MIN) {   /* a requested width: even, within the column space */
        W = Wreq & ~(size_t)1;
        if (W > rd->ncol) W = rd->ncol & ~(size_t)1;
        if (W < VFFT_ILRD_W_MIN) W = VFFT_ILRD_W_MIN;
    }
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
    if (!_ilrd_build_inner(rd, depth, req, nreq) || !_ilrd_slot(rd, 0)) { _ilrd_free(rd); return 0; }
    {   /* the kernels: the forward chain of w^(perm[m]) in comb order, scaled. The
         * sequence is one column (row stride 1): a nested leaf is run through a
         * twin built for that width (the same chain, the same values, its own
         * scratch), since the serving nest's geometry is the window's. */
        double *za = (double *)vfft_aligned_alloc(2 * (size_t)M * sizeof(double));
        vfft_ilrd_t *serving = rd->nest, *one = 0;
        int d, m;
        if (!za) { _ilrd_free(rd); return 0; }
        if (serving) {
            one = _ilrd_create(serving->p, (size_t)(M / serving->p), (size_t)M, 1, depth + 1, 0, 0, 0);
            if (!one || !_ilrd_dif_tw(one, M, (size_t)(M / serving->p), 1)) {
                _ilrd_free(one); vfft_aligned_free(za); _ilrd_free(rd); return 0;
            }
            rd->nest = one;
        }
        /* up to M = 2^22 (the long-double tables' memory) both kernels from one
         * DFT in extended precision (dft_ld.h), each entry rounded once, in the
         * chain's comb order (read back from the labelled sequence whose DFT is
         * the ramp); above it by the chain's double transform */
        int exact = 0;
        if (M <= (1 << 22)) {
            int *pos = (int *)malloc((size_t)M * sizeof(int));
            vfft_rader_label(M, za);
            _ilrd_inner_run(rd, za, 1, 1, 0, 0);
            if (pos && vfft_rader_positions(M, za, pos))
                exact = vfft_rader_kernels_ld(M, rd->gpow, pr, pos, rd->omf, rd->omb);
            else
                fprintf(stderr, "[rd-exact] leaf %d: the chain's order did not read back; the double kernel instead\n", pr);
            free(pos);
        }
        for (d = 0; !exact && d < 2; d++) {
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
/* stage s of the plan as a Rader stage: the pool's pick, or the requested
 * inner chain (req[0..nreq)) and width (0 = default); replaces any engine
 * the stage holds. The plan is NOT rebound here. */
static int _ilrd_build_stage(vfft_ilfd_plan_t *p, int s, const int *req, int nreq, size_t W)
{
    const size_t D = p->D[s], L = (size_t)p->R[s] * D, nb = p->nblk[s];
    vfft_ilrd_t *rd = _ilrd_create(p->R[s], D, L, nb, 0, req, nreq, W);
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
    if (p->rd[s]) _ilrd_free((vfft_ilrd_t *)p->rd[s]);
    p->rd[s] = rd;
    p->has_rd = 1;
    p->scr_ok = 0;
    if (p->natbase) {   /* a rebuild after the create: the redirection is known */
        rd->last = (s == p->K - 1) && !p->scr;
        rd->natbase = p->natbase;
        rd->nstride = (size_t)p->N / (size_t)p->R[s];
    }
    if (p->mtb) { free(p->mtb); p->mtb = 0; }   /* the MT records named the old engine: rebind on demand */
    return 1;
}
/* ── THE COLUMN STAGE (the 2D tier's stage 0; docs/roadmap/rader_leaf_design.md
 * section 13): the same engine over a plane of N1 = p . D0 rows, digit d's legs
 * at rows l * D0 + d, the windows gathered across digits and lanes; the DIF
 * stage's twiddles per digit (mode 2, set d; none at D0 = 1); the scratch
 * slot is the calling thread's (thread_pool_self_slot: 0 the caller, i + 1
 * pool worker i), so every walker runs it with no worker index of its own.
 * The walkers find the engine in the stage-0 table slots as two direction
 * tags (il2d_cols.h, _il2d_rd0_hook). ─────────────────────────────────── */
static void _ilrd_col0_run(const void *v, const double *src, size_t spitch, double *dst, size_t dpitch,
                           size_t sk, size_t dk, size_t w, size_t d0, size_t nd)
{
    const _ilrd_ctab_t *tag = (const _ilrd_ctab_t *)v;
    const vfft_ilrd_t *rd = tag->rd;
    const int bwd = tag->bwd;
    const size_t W = rd->W, D0 = rd->Dp;
    const double *tw = bwd ? rd->twb : rd->twf;
    /* the scratch slot: the thread's pool slot on an engine built for the plan's
     * team; an engine with ONE slot belongs to a plan created at one thread --
     * a per-worker clone of the 3D tier, or any one-thread plan -- and runs
     * that slot whichever thread executes it (one plan runs one execute at a
     * time: vfft.h) */
    int slot = thread_pool_self_slot();
    _ilrd_run_t runs[VFFT_ILRD_W_MAX + 2];
    if (slot >= rd->nslot) slot = 0;
    int nr = 0;
    size_t acc = 0, d, k;
    if (!w || !nd) return;
    for (d = d0; d < d0 + nd; d++)
        for (k = 0; k < w;) {
            size_t n = w - k;
            if (n > W - acc) n = W - acc;
            runs[nr].s0 = src + 2 * (d * spitch + sk + k);
            runs[nr].ss = D0 * spitch;
            runs[nr].d0 = dst + 2 * (d * dpitch + dk + k);
            runs[nr].ds = D0 * dpitch;
            runs[nr].n = n;
            runs[nr].gi = tw ? tw + 2 * (size_t)rd->p * d : 0;
            nr++; acc += n; k += n;
            if (acc == W) { _ilrd_window_runs(rd, runs, nr, bwd, slot); nr = 0; acc = 0; }
        }
    if (nr) _ilrd_window_runs(rd, runs, nr, bwd, slot);
}
/* a prime the column stage serves: above the pool, p - 1 a column chain (a
 * nested leaf allowed, one level) */
static int _ilrd_col0_ok(int p)
{
    int Rs[VFFT_ILRD_MAX_NST], nst = 0, nestp = 0;
    if (p <= VFFT_ILRD_POOL_MAX || !vfft_is_prime(p)) return 0;
    if (!_ilrd_pick_chain(p - 1, Rs, &nst, &nestp)) return 0;
    if (nestp) {   /* the nested leaf's own inner must chain from the pool */
        int R2[VFFT_ILRD_MAX_NST], n2 = 0, q2 = 0;
        if (!_ilrd_pick_chain(nestp - 1, R2, &n2, &q2) || q2) return 0;
    }
    return 1;
}
/* the engine for stage 0 of a column chain over a plane of N1 rows and rn lanes,
 * T scratch slots (the caller and T - 1 workers); the pool's inner pick, the
 * default width (the planner's race over both: phase A2). NULL = not built. */
static void *_ilrd_col0_build(int p, int N1, size_t rn, int T, const int *req, int nreq, size_t Wreq)
{
    const size_t D0 = (size_t)(N1 / p);
    vfft_ilrd_t *rd;
    int k;
    if (p <= VFFT_ILRD_POOL_MAX || N1 % p || rn == 0) return 0;
    rd = _ilrd_create(p, D0 * rn, (size_t)N1 * rn, 1, 0, req, nreq, Wreq);
    if (!rd) return 0;
    rd->Dp = D0;
    if (D0 > 1 && !_ilrd_dif_tw(rd, N1, D0, 1)) { _ilrd_free(rd); return 0; }
    for (k = 1; k < T && k < VFFT_ILRD_MAX_SLOTS; k++)
        if (!_ilrd_slot(rd, k)) { _ilrd_free(rd); return 0; }
    return rd;
}
/* THE COLUMN BUILD'S ENGINE CONTEXT (planning side, one build at a time, the
 * Bluestein provider's pattern): the engine the column build attached for its
 * (p, N1), which every table set built for that chain during the build -- the
 * descriptor's own, the forms race's, the N-arm race's, a chain race arm's --
 * names through a fresh tag block. Set by the attach, cleared when the build
 * ends or the descriptor is freed. */
static struct { vfft_ilrd_t *rd; int N1; } _ilrd_col0_ctx;
static void _ilrd_col0_ctx_set(void *rd, int N1) { _ilrd_col0_ctx.rd = (vfft_ilrd_t *)rd; _ilrd_col0_ctx.N1 = N1; }
static void _ilrd_col0_ctx_clear(const void *rd) { if (!rd || _ilrd_col0_ctx.rd == rd) { _ilrd_col0_ctx.rd = 0; _ilrd_col0_ctx.N1 = 0; } }
/* the column engine's verdict as the row's tokens (rd = the inner radices dotted, a radix
 * above the pool = the nested leaf; rdw = the window width) */
static int _ilrd_col0_desc(const void *v, char *rd, size_t rsz, int *W)
{
    const vfft_ilrd_t *e = (const vfft_ilrd_t *)v;
    int i, off = 0;
    if (!e) return 0;
    rd[0] = '\0';
    for (i = 0; i < e->nst && off < (int)rsz - 1; i++)
        off += snprintf(rd + off, rsz - (size_t)off, "%s%d", i ? "." : "", e->Rs[i]);
    if (W) *W = (int)e->W;
    return 1;
}
static void *_ilrd_col0_tag_fn(int N1, int p, int bwd)
{
    _ilrd_ctab_t *t;
    if (!_ilrd_col0_ctx.rd || _ilrd_col0_ctx.N1 != N1 || _ilrd_col0_ctx.rd->p != p) return 0;
    t = (_ilrd_ctab_t *)vfft_aligned_alloc(sizeof *t);
    if (!t) return 0;
    t->rd = _ilrd_col0_ctx.rd;
    t->bwd = bwd ? 1 : 0;
    return t;
}

static int _ilrd_build_hook_fn(vfft_ilfd_plan_t *p, int s) { return _ilrd_build_stage(p, s, 0, 0, 0); }
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

/* ── THE INNER RACE IN ROLE (the planner's bench) ──────────────────────
 * The candidates: every column chain of p-1 over the pool, a nested prime
 * fixed at stage 0, each radix set in two orders (large first, small first,
 * a closing radix kept last), at most VFFT_ILRD_MAX_INNERS; then the window
 * widths around the default. Every arm is a live engine (its own tables and
 * scratch) swapped into the stage between samples — the whole forward is
 * the sample, as the chain race's. The plan leaves with the best installed. */
#define VFFT_ILRD_MAX_INNERS 10
typedef struct { int Rs[VFFT_ILRD_MAX_NST]; int nst; } _ilrd_inner_t;

static void _ilrd_enum_rec(int rem, int i0, int *cur, int depth, int fixed, _ilrd_inner_t *out, int *n, int max)
{
    static const int POOL[] = { 64, 32, 16, 8, 4, 47, 43, 41, 37, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3 };
    const int npool = (int)(sizeof POOL / sizeof POOL[0]);
    int i;
    if (*n >= max || depth >= VFFT_ILRD_MAX_NST) return;
    if (rem == 1) {
        if (depth - fixed < 1) return;
        memcpy(out[*n].Rs, cur, sizeof(int) * (size_t)depth); out[*n].nst = depth; (*n)++;
        return;
    }
    if (depth >= 1 && _ilrd_closing_ok(rem)) {   /* the closing radix ends the set (right after the nested leaf too: p - 1 = q * 2) */
        cur[depth] = rem;
        memcpy(out[*n].Rs, cur, sizeof(int) * (size_t)(depth + 1)); out[*n].nst = depth + 1; (*n)++;
        if (*n >= max) return;
    }
    for (i = i0; i < npool; i++) {
        const int r = POOL[i];
        if (rem % r) continue;
        if (rem == r) { if (!vfft_il2p_n1c_fn(r, 0) || !vfft_il2p_n1c_fn(r, 1)) continue; }
        else if (!vfft_il2p_t2c_fn(r, 0) || !vfft_il2p_t2c_fn(r, 1)) continue;
        cur[depth] = r;
        _ilrd_enum_rec(rem / r, i, cur, depth + 1, fixed, out, n, max);
    }
}
static int _ilrd_enum_inners(int M, _ilrd_inner_t *out, int max)
{
    int cur[VFFT_ILRD_MAX_NST], n = 0, nsets, i, j, k;
    const int q = _ilrd_largest_prime_factor(M);
    int fixed = 0;
    if (q > VFFT_ILRD_POOL_MAX) {
        if (!vfft_is_prime(q)) return 0;
        cur[0] = q; fixed = 1;
        _ilrd_enum_rec(M / q, 0, cur, 1, 1, out, &n, max);
    } else
        _ilrd_enum_rec(M, 0, cur, 0, 0, out, &n, max);
    nsets = n;
    for (i = 0; i < nsets && n < max; i++) {   /* the second order: small first, a closing radix kept last */
        _ilrd_inner_t o = out[i];
        const int last = o.Rs[o.nst - 1];
        const int closing = (o.nst - fixed >= 2) && _ilrd_closing_ok(last) && !vfft_il2p_t2c_fn(last, 0);
        const int hi = closing ? o.nst - 1 : o.nst;
        for (j = fixed; j < hi; j++)
            for (k = j + 1; k < hi; k++)
                if (o.Rs[k] < o.Rs[j]) { const int t = o.Rs[j]; o.Rs[j] = o.Rs[k]; o.Rs[k] = t; }
        if (memcmp(o.Rs, out[i].Rs, sizeof(int) * (size_t)o.nst)) out[n++] = o;
    }
    return n;
}

typedef struct { vfft_ilfd_plan_t *p; int s; vfft_ilrd_t *rd; const double *zin; double *zout; char name[40]; } _ilrd_race_ctx_t;
static void _ilrd_race_arm_run(void *v)
{
    _ilrd_race_ctx_t *c = (_ilrd_race_ctx_t *)v;
    if (c->p->rd[c->s] != c->rd) { c->p->rd[c->s] = c->rd; vfft_ilfd_bind(c->p); }
    vfft_ilfd_execute_fwd(c->p, c->zin, c->zout);
}
/* one live engine for stage s with the given inner and width, the stage's
 * twiddles and redirection attached; NULL = not buildable */
static vfft_ilrd_t *_ilrd_arm_engine(vfft_ilfd_plan_t *p, int s, const int *Rs, int nst, size_t W)
{
    vfft_ilrd_t *keep = (vfft_ilrd_t *)p->rd[s], *rd;
    p->rd[s] = 0;
    if (!_ilrd_build_stage(p, s, Rs, nst, W)) { p->rd[s] = keep; return 0; }
    rd = (vfft_ilrd_t *)p->rd[s];
    p->rd[s] = keep;
    if (_ilfd_race_log()) {
        int i;
        fprintf(stderr, "[k1fd-race] N=%d rader s%d arm: p=%d D=%zu L=%zu nb=%zu ncol=%zu W=%zu nst=%d Ls=", p->N, s, rd->p, rd->D, rd->L, rd->nb, rd->ncol, rd->W, rd->nst);
        for (i = 0; i < rd->nst; i++) fprintf(stderr, "%s%d", i ? "." : "", rd->Ls[i]);
        fprintf(stderr, " twmode=%d twrun=%zu last=%d\n", rd->twmode, rd->twrun, rd->last);
    }
    return rd;
}
/* the race: inner chains, then widths; the plan leaves bound to the best.
 * Returns the best arm's ns (1e18 = nothing raced, the plan unchanged). */
static double vfft_ilrd_race(vfft_ilfd_plan_t *p, int s, const double *zin, double *zout, int pace)
{
    _ilrd_inner_t cand[VFFT_ILRD_MAX_INNERS];
    _ilrd_race_ctx_t cx[VFFT_ILRD_MAX_INNERS + 4];
    vfft_race_arm_t arms[VFFT_ILRD_MAX_INNERS + 4];
    double ns[VFFT_ILRD_MAX_INNERS + 4], best_ns = 1e18;
    vfft_ilrd_t *cur = (vfft_ilrd_t *)p->rd[s], *win;
    int n, na = 0, i, best, reps, nst_w;
    int Rs_w[VFFT_ILRD_MAX_NST];
    size_t W_w, wc[4];
    if (!cur) return 1e18;
    n = _ilrd_enum_inners(cur->p - 1, cand, VFFT_ILRD_MAX_INNERS);
    reps = _ilfd_race_reps(p, zin, zout);
    for (i = 0; i < n; i++) {   /* the chains at the default width */
        vfft_ilrd_t *rd = _ilrd_arm_engine(p, s, cand[i].Rs, cand[i].nst, 0);
        int j, off = 0;
        if (!rd) continue;
        cx[na].p = p; cx[na].s = s; cx[na].rd = rd; cx[na].zin = zin; cx[na].zout = zout;
        for (j = 0; j < cand[i].nst && off < (int)sizeof cx[na].name - 6; j++)
            off += snprintf(cx[na].name + off, sizeof cx[na].name - (size_t)off, "%s%d", j ? "." : "", cand[i].Rs[j]);
        arms[na].name = cx[na].name; arms[na].run = _ilrd_race_arm_run; arms[na].ctx = &cx[na];
        na++;
    }
    if (na == 0) { vfft_ilfd_bind(p); return 1e18; }
    {
        const vfft_race_proto_t proto = { VFFT_ILFD_RACE_ROUNDS, reps, VFFT_RACE_MIN, 1, 1, NULL, NULL, pace };
        best = vfft_race_run(&proto, arms, na, ns);
        if (best < 0) best = 0;
        if (_ilfd_race_log()) {
            fprintf(stderr, "[k1fd-race] N=%d rader s%d p=%d inner: reps=%d", p->N, s, cur->p, reps);
            for (i = 0; i < na; i++) fprintf(stderr, " %s=%.1f", cx[i].name, ns[i]);
            fprintf(stderr, " -> %s\n", cx[best].name);
        }
    }
    win = cx[best].rd; best_ns = ns[best];
    memcpy(Rs_w, win->Rs, sizeof Rs_w); nst_w = win->nst; W_w = win->W;
    for (i = 0; i < na; i++) if (cx[i].rd != win) _ilrd_free(cx[i].rd);
    _ilrd_free(cur);
    p->rd[s] = win;
    vfft_ilfd_bind(p);   /* the call records named the freed engine until now */
    /* the widths around the winner's: half, double, quadruple, the column space */
    {
        const size_t ncol = win->ncol;
        int nw = 0, k;
        wc[nw++] = W_w / 2; wc[nw++] = W_w * 2; wc[nw++] = W_w * 4; wc[nw++] = ncol & ~(size_t)1;
        if (_ilfd_race_log()) fprintf(stderr, "[k1fd-race] N=%d rader s%d width arms: W=%zu ncol=%zu\n", p->N, s, W_w, ncol);
        na = 0;
        cx[na].p = p; cx[na].s = s; cx[na].rd = win; cx[na].zin = zin; cx[na].zout = zout;
        snprintf(cx[na].name, sizeof cx[na].name, "W%zu", W_w);
        arms[na].name = cx[na].name; arms[na].run = _ilrd_race_arm_run; arms[na].ctx = &cx[na]; na++;
        for (k = 0; k < nw; k++) {
            vfft_ilrd_t *rd;
            int dup = 0, j;
            if (wc[k] < VFFT_ILRD_W_MIN || wc[k] > ncol || wc[k] == W_w) continue;
            for (j = 1; j < na; j++) if (cx[j].rd->W == wc[k]) dup = 1;
            if (dup) continue;
            rd = _ilrd_arm_engine(p, s, Rs_w, nst_w, wc[k]);
            if (!rd || rd->W == W_w) { _ilrd_free(rd); continue; }
            cx[na].p = p; cx[na].s = s; cx[na].rd = rd; cx[na].zin = zin; cx[na].zout = zout;
            snprintf(cx[na].name, sizeof cx[na].name, "W%zu", rd->W);
            arms[na].name = cx[na].name; arms[na].run = _ilrd_race_arm_run; arms[na].ctx = &cx[na]; na++;
        }
        if (na > 1) {
            const vfft_race_proto_t proto = { VFFT_ILFD_RACE_ROUNDS, reps, VFFT_RACE_MIN, 1, 1, NULL, NULL, pace };
            best = vfft_race_run(&proto, arms, na, ns);
            if (best < 0) best = 0;
            if (_ilfd_race_log()) {
                fprintf(stderr, "[k1fd-race] N=%d rader s%d width:", p->N, s);
                for (i = 0; i < na; i++) fprintf(stderr, " %s=%.1f", cx[i].name, ns[i]);
                fprintf(stderr, " -> %s\n", cx[best].name);
            }
            best_ns = ns[best];
            for (i = 0; i < na; i++) if (i != best) _ilrd_free(cx[i].rd);
            p->rd[s] = cx[best].rd;
        }
    }
    if (p->mtb) { free(p->mtb); p->mtb = 0; }
    vfft_ilfd_bind(p);
    return best_ns;
}

/* replay: stage s rebuilt with the banked inner and width; 1 = ok, the plan rebound */
static int vfft_ilrd_set_inner(vfft_ilfd_plan_t *p, int s, const int *Rs, int nst, size_t W)
{
    if (s < 0 || s >= p->K || !p->rd[s]) return 0;
    if (!_ilrd_build_stage(p, s, Rs, nst, W)) return 0;
    vfft_ilfd_bind(p);
    return 1;
}
/* the plan's (first) Rader stage and its verdict: s, the inner radices, the
 * width; 0 = the plan has none */
static int vfft_ilrd_get_inner(const vfft_ilfd_plan_t *p, int *s_out, int *Rs, int *nst, int *W)
{
    int s;
    for (s = 0; s < p->K; s++) {
        const vfft_ilrd_t *rd = (const vfft_ilrd_t *)p->rd[s];
        if (!rd) continue;
        if (s_out) *s_out = s;
        if (Rs) memcpy(Rs, rd->Rs, sizeof(int) * (size_t)rd->nst);
        if (nst) *nst = rd->nst;
        if (W) *W = (int)rd->W;
        return 1;
    }
    return 0;
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
static size_t _ilrd_width_hook_fn(const void *rd) { return rd ? ((const vfft_ilrd_t *)rd)->W : 1; }
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
static size_t (*_ilfd_rd_width_hook)(const void *) = _ilrd_width_hook_fn;
/* the column stage's hooks (il2d_cols.h's tentative definitions) */
static void (*_il2d_rd0_hook)(const void *, const double *, size_t, double *, size_t, size_t, size_t,
                              size_t, size_t, size_t) = _ilrd_col0_run;
static int (*_il2d_rd0_ok_hook)(int) = _ilrd_col0_ok;
static void (*_il2d_rd0_free_hook)(void *) = _ilrd_free_hook_fn;
static void *(*_il2d_rd0_tag_hook)(int, int, int) = _ilrd_col0_tag_fn;

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
/* the Rader stage's inner race in role on this chain (the planner's bench step);
 * returns the best arm's ns, 1e18 = no Rader stage */
double vfft_tool_ilfd_race_inner(void *plan, const double *zin, double *zout)
{
    vfft_ilfd_plan_t *p = (vfft_ilfd_plan_t *)plan;
    int s = 0;
    if (!vfft_ilrd_get_inner(p, &s, NULL, NULL, NULL)) return 1e18;
    return vfft_ilrd_race(p, s, zin, zout, 0);
}
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
