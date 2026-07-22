/* strided_tw.h — the strided TWIDDLE-STAGE row engine (§6a41).
 *
 * Extends the strided r2c/c2r row engines (§6a37-39, mono ceiling N2=64) to
 * N2 ∈ {128, 256} via the §6a40-measured composition law: ROW-BLOCKED
 * FUSION — per 8-row block, [DIF front stage → existing r64 strided c2c
 * monos on sub-bands → mapped conjugate split] — one DRAM pass over the
 * plane, everything between L1-resident.
 *
 * Taxonomy note (see docs/design/strided_codelet_families.md): the FRONTS
 * here are ENGINE KERNELS (transpose.h-class hand infrastructure — simple
 * strided butterflies vectorized ALONG the row, no lattice, no DAG), not
 * codelets. The sub-band FFTs are the emitted r64 c2c strided monos. The
 * split/merge are the §6a36/38 formulas applied through the DIF ordering
 * map. A single fused emitted codelet per (N, direction) is the recorded
 * future refinement (§6a40: kills per-block call overhead + the L1
 * front→mono reload; the composition here is its measured floor).
 *
 * Two-for-one at ROW level: even real rows enter as re lanes, odd as im
 * (pair addressing, §6a36's trick lifted to the composition) — so the c2c
 * machinery serves the r2c contract with a split at the door.
 *
 * DIF ordering map (front radix r, monos N/r = 64, one front stage):
 *   Z[r*k + j] lives at column j*64 + k,   j = 0..r-1, k = 0..63.
 * The split/merge address Z through this map; nothing is ever reordered in
 * memory.
 *
 * Direction conventions match the §6a37/38 family: fwd emits half-spectra
 * rows (out_stride-pitched); bwd consumes them and emits UNNORMALIZED real
 * rows (= N * x): mono-bwd contributes ×64, front-bwd ×r, total ×N. me is
 * PAIRS throughout; callers guarantee rows % 8 == 0 (4 pairs per block).
 */
#ifndef VFFT_STRIDED_TW_H
#define VFFT_STRIDED_TW_H
/* win-compat: mingw lacks C11 aligned_alloc; Windows must pair _aligned_malloc/_aligned_free
 * (same shim as proto_stride_compat.h, guarded so whichever comes first wins). */
#ifndef STRIDE_ALIGNED_ALLOC
#if defined(_WIN32) || defined(_MSC_VER)
#include <malloc.h>
#define STRIDE_ALIGNED_ALLOC(align, size) _aligned_malloc((size), (align))
#define STRIDE_ALIGNED_FREE(p) _aligned_free(p)
#else
#include <stdlib.h>
#define STRIDE_ALIGNED_ALLOC(align, size) \
    aligned_alloc((align), ((size) + (size_t)(align) - 1) & ~((size_t)(align) - 1))
#define STRIDE_ALIGNED_FREE(p) free(p)
#endif
#endif


#include <immintrin.h>
#include <stddef.h>
#include <math.h>

/* the emitted c2c strided monos (Design C quadrant) used as sub-band leaves */
void radix64_n1_fwd_avx2_strided(double *, double *, const double *,
                                 const double *, size_t, size_t);
void radix64_n1_bwd_avx2_strided(double *, double *, const double *,
                                 const double *, size_t, size_t);

/* ── twiddle tables ─────────────────────────────────────────────────── */

typedef struct {
    int     N;          /* row length: 128 or 256 */
    int     r;          /* front radix: N / 64    */
    double *twr, *twi;  /* r4: [W^m | W^2m | W^3m], m = 0..N/4-1; r2: W^m, m = 0..N/2-1 */
} _stw_tables_t;

static inline int _stw_tables_init(_stw_tables_t *t, int N)
{
    if (N != 128 && N != 256) return 0;
    t->N = N;
    t->r = N / 64;
    size_t per = (size_t)N / (size_t)t->r;
    size_t cnt = (t->r == 2) ? per : 3 * per;
    t->twr = (double *)STRIDE_ALIGNED_ALLOC(64, cnt * sizeof(double));
    t->twi = (double *)STRIDE_ALIGNED_ALLOC(64, cnt * sizeof(double));
    if (!t->twr || !t->twi) { STRIDE_ALIGNED_FREE(t->twr); STRIDE_ALIGNED_FREE(t->twi); return 0; }
    if (t->r == 2) {
        for (size_t m = 0; m < per; m++) {
            double a = -2.0 * M_PI * (double)m / (double)N;
            t->twr[m] = cos(a); t->twi[m] = sin(a);
        }
    } else {
        for (int j = 1; j <= 3; j++)
            for (size_t m = 0; m < per; m++) {
                double a = -2.0 * M_PI * (double)j * (double)m / (double)N;
                t->twr[(size_t)(j - 1) * per + m] = cos(a);
                t->twi[(size_t)(j - 1) * per + m] = sin(a);
            }
    }
    return 1;
}

static inline void _stw_tables_free(_stw_tables_t *t)
{
    STRIDE_ALIGNED_FREE(t->twr); STRIDE_ALIGNED_FREE(t->twi); t->twr = t->twi = 0;
}

/* ── DIF fronts (engine kernels; vectorized along the row) ──────────── */

/* radix-2 fwd: s = a+b at col c; p = (a-b)*W^c at col c+N/2. */
static inline void _stw_front2_fwd(const double *xr, const double *xi,
                                   double *yr, double *yi,
                                   size_t rs_src, size_t rs_dst,
                                   size_t pairs, int N,
                                   const double *twr, const double *twi)
{
    const size_t h = (size_t)N / 2;
    for (size_t p = 0; p < pairs; p++) {
        const double *ar = xr + p * rs_src, *ai = xi + p * rs_src;
        double *br = yr + p * rs_dst, *bi = yi + p * rs_dst;
        for (size_t c = 0; c < h; c += 4) {
            __m256d Ar = _mm256_loadu_pd(ar + c),     Ai = _mm256_loadu_pd(ai + c);
            __m256d Br = _mm256_loadu_pd(ar + c + h), Bi = _mm256_loadu_pd(ai + c + h);
            __m256d sr = _mm256_add_pd(Ar, Br), si = _mm256_add_pd(Ai, Bi);
            __m256d dr = _mm256_sub_pd(Ar, Br), di = _mm256_sub_pd(Ai, Bi);
            __m256d wr = _mm256_loadu_pd(twr + c), wi = _mm256_loadu_pd(twi + c);
            _mm256_storeu_pd(br + c, sr); _mm256_storeu_pd(bi + c, si);
            _mm256_storeu_pd(br + c + h,
                _mm256_fmsub_pd(dr, wr, _mm256_mul_pd(di, wi)));
            _mm256_storeu_pd(bi + c + h,
                _mm256_fmadd_pd(dr, wi, _mm256_mul_pd(di, wr)));
        }
    }
}

/* radix-2 bwd (unnormalized ×2): a = s + p*conj(W); b = s - p*conj(W). */
static inline void _stw_front2_bwd(double *xr, double *xi, size_t rs,
                                   size_t pairs, int N,
                                   const double *twr, const double *twi)
{
    const size_t h = (size_t)N / 2;
    for (size_t p = 0; p < pairs; p++) {
        double *ar = xr + p * rs, *ai = xi + p * rs;
        for (size_t c = 0; c < h; c += 4) {
            __m256d sr = _mm256_loadu_pd(ar + c),     si = _mm256_loadu_pd(ai + c);
            __m256d pr = _mm256_loadu_pd(ar + c + h), pi = _mm256_loadu_pd(ai + c + h);
            __m256d wr = _mm256_loadu_pd(twr + c), wi = _mm256_loadu_pd(twi + c);
            /* z = p * conj(W) = (pr*wr + pi*wi, pi*wr - pr*wi) */
            __m256d zr = _mm256_fmadd_pd(pr, wr, _mm256_mul_pd(pi, wi));
            __m256d zi = _mm256_fmsub_pd(pi, wr, _mm256_mul_pd(pr, wi));
            _mm256_storeu_pd(ar + c,     _mm256_add_pd(sr, zr));
            _mm256_storeu_pd(ai + c,     _mm256_add_pd(si, zi));
            _mm256_storeu_pd(ar + c + h, _mm256_sub_pd(sr, zr));
            _mm256_storeu_pd(ai + c + h, _mm256_sub_pd(si, zi));
        }
    }
}

/* radix-4 fwd DIF:
 *   t0=a+c t1=a-c t2=b+d t3=b-d  (a,b,c,d = x[m], x[m+Q], x[m+2Q], x[m+3Q], Q=N/4)
 *   y0=t0+t2; y1=(t1 - i*t3)*W^m; y2=(t0-t2)*W^2m; y3=(t1 + i*t3)*W^3m */
static inline void _stw_front4_fwd(const double *xr, const double *xi,
                                   double *yr, double *yi,
                                   size_t rs_src, size_t rs_dst,
                                   size_t pairs, int N,
                                   const double *twr, const double *twi)
{
    const size_t Q = (size_t)N / 4;
    const double *w1r = twr,         *w1i = twi;
    const double *w2r = twr + Q,     *w2i = twi + Q;
    const double *w3r = twr + 2 * Q, *w3i = twi + 2 * Q;
    for (size_t p = 0; p < pairs; p++) {
        const double *sr_ = xr + p * rs_src, *si_ = xi + p * rs_src;
        double *dr_ = yr + p * rs_dst, *di_ = yi + p * rs_dst;
        for (size_t m = 0; m < Q; m += 4) {
            __m256d ar = _mm256_loadu_pd(sr_ + m),         ai = _mm256_loadu_pd(si_ + m);
            __m256d br = _mm256_loadu_pd(sr_ + m + Q),     bi = _mm256_loadu_pd(si_ + m + Q);
            __m256d cr = _mm256_loadu_pd(sr_ + m + 2 * Q), ci = _mm256_loadu_pd(si_ + m + 2 * Q);
            __m256d er = _mm256_loadu_pd(sr_ + m + 3 * Q), ei = _mm256_loadu_pd(si_ + m + 3 * Q);
            __m256d t0r = _mm256_add_pd(ar, cr), t0i = _mm256_add_pd(ai, ci);
            __m256d t1r = _mm256_sub_pd(ar, cr), t1i = _mm256_sub_pd(ai, ci);
            __m256d t2r = _mm256_add_pd(br, er), t2i = _mm256_add_pd(bi, ei);
            __m256d t3r = _mm256_sub_pd(br, er), t3i = _mm256_sub_pd(bi, ei);
            /* y0 */
            _mm256_storeu_pd(dr_ + m, _mm256_add_pd(t0r, t2r));
            _mm256_storeu_pd(di_ + m, _mm256_add_pd(t0i, t2i));
            /* u1 = t1 - i*t3 = (t1r + t3i, t1i - t3r); y1 = u1*W1 */
            {
                __m256d ur = _mm256_add_pd(t1r, t3i), ui = _mm256_sub_pd(t1i, t3r);
                __m256d wr = _mm256_loadu_pd(w1r + m), wi = _mm256_loadu_pd(w1i + m);
                _mm256_storeu_pd(dr_ + m + Q,
                    _mm256_fmsub_pd(ur, wr, _mm256_mul_pd(ui, wi)));
                _mm256_storeu_pd(di_ + m + Q,
                    _mm256_fmadd_pd(ur, wi, _mm256_mul_pd(ui, wr)));
            }
            /* y2 = (t0 - t2)*W2 */
            {
                __m256d ur = _mm256_sub_pd(t0r, t2r), ui = _mm256_sub_pd(t0i, t2i);
                __m256d wr = _mm256_loadu_pd(w2r + m), wi = _mm256_loadu_pd(w2i + m);
                _mm256_storeu_pd(dr_ + m + 2 * Q,
                    _mm256_fmsub_pd(ur, wr, _mm256_mul_pd(ui, wi)));
                _mm256_storeu_pd(di_ + m + 2 * Q,
                    _mm256_fmadd_pd(ur, wi, _mm256_mul_pd(ui, wr)));
            }
            /* u3 = t1 + i*t3 = (t1r - t3i, t1i + t3r); y3 = u3*W3 */
            {
                __m256d ur = _mm256_sub_pd(t1r, t3i), ui = _mm256_add_pd(t1i, t3r);
                __m256d wr = _mm256_loadu_pd(w3r + m), wi = _mm256_loadu_pd(w3i + m);
                _mm256_storeu_pd(dr_ + m + 3 * Q,
                    _mm256_fmsub_pd(ur, wr, _mm256_mul_pd(ui, wi)));
                _mm256_storeu_pd(di_ + m + 3 * Q,
                    _mm256_fmadd_pd(ur, wi, _mm256_mul_pd(ui, wr)));
            }
        }
    }
}

/* radix-4 bwd (unnormalized ×4): un-twiddle with conj(Wj), then inverse DFT4
 * (the +i kernel): x[m]=y0+z1+z2+z3; x[m+Q]=y0+i*z1-z2-i*z3;
 * x[m+2Q]=y0-z1+z2-z3; x[m+3Q]=y0-i*z1-z2+i*z3. */
static inline void _stw_front4_bwd(double *xr, double *xi, size_t rs,
                                   size_t pairs, int N,
                                   const double *twr, const double *twi)
{
    const size_t Q = (size_t)N / 4;
    const double *w1r = twr,         *w1i = twi;
    const double *w2r = twr + Q,     *w2i = twi + Q;
    const double *w3r = twr + 2 * Q, *w3i = twi + 2 * Q;
    for (size_t p = 0; p < pairs; p++) {
        double *dr_ = xr + p * rs, *di_ = xi + p * rs;
        for (size_t m = 0; m < Q; m += 4) {
            __m256d y0r = _mm256_loadu_pd(dr_ + m),         y0i = _mm256_loadu_pd(di_ + m);
            __m256d a1r = _mm256_loadu_pd(dr_ + m + Q),     a1i = _mm256_loadu_pd(di_ + m + Q);
            __m256d a2r = _mm256_loadu_pd(dr_ + m + 2 * Q), a2i = _mm256_loadu_pd(di_ + m + 2 * Q);
            __m256d a3r = _mm256_loadu_pd(dr_ + m + 3 * Q), a3i = _mm256_loadu_pd(di_ + m + 3 * Q);
            __m256d wr, wi, z1r, z1i, z2r, z2i, z3r, z3i;
            wr = _mm256_loadu_pd(w1r + m); wi = _mm256_loadu_pd(w1i + m);
            z1r = _mm256_fmadd_pd(a1r, wr, _mm256_mul_pd(a1i, wi));
            z1i = _mm256_fmsub_pd(a1i, wr, _mm256_mul_pd(a1r, wi));
            wr = _mm256_loadu_pd(w2r + m); wi = _mm256_loadu_pd(w2i + m);
            z2r = _mm256_fmadd_pd(a2r, wr, _mm256_mul_pd(a2i, wi));
            z2i = _mm256_fmsub_pd(a2i, wr, _mm256_mul_pd(a2r, wi));
            wr = _mm256_loadu_pd(w3r + m); wi = _mm256_loadu_pd(w3i + m);
            z3r = _mm256_fmadd_pd(a3r, wr, _mm256_mul_pd(a3i, wi));
            z3i = _mm256_fmsub_pd(a3i, wr, _mm256_mul_pd(a3r, wi));
            __m256d s02r = _mm256_add_pd(y0r, z2r), s02i = _mm256_add_pd(y0i, z2i);
            __m256d d02r = _mm256_sub_pd(y0r, z2r), d02i = _mm256_sub_pd(y0i, z2i);
            __m256d s13r = _mm256_add_pd(z1r, z3r), s13i = _mm256_add_pd(z1i, z3i);
            __m256d d13r = _mm256_sub_pd(z1r, z3r), d13i = _mm256_sub_pd(z1i, z3i);
            _mm256_storeu_pd(dr_ + m,         _mm256_add_pd(s02r, s13r));
            _mm256_storeu_pd(di_ + m,         _mm256_add_pd(s02i, s13i));
            /* + i*d13 = (-d13i, d13r) */
            _mm256_storeu_pd(dr_ + m + Q,     _mm256_sub_pd(d02r, d13i));
            _mm256_storeu_pd(di_ + m + Q,     _mm256_add_pd(d02i, d13r));
            _mm256_storeu_pd(dr_ + m + 2 * Q, _mm256_sub_pd(s02r, s13r));
            _mm256_storeu_pd(di_ + m + 2 * Q, _mm256_sub_pd(s02i, s13i));
            _mm256_storeu_pd(dr_ + m + 3 * Q, _mm256_add_pd(d02r, d13i));
            _mm256_storeu_pd(di_ + m + 3 * Q, _mm256_sub_pd(d02i, d13r));
        }
    }
}

/* ── DIF ordering map:  Z[bin] is at column map(bin) ────────────────── */

static inline size_t _stw_map(int bin, int r)
{
    /* bin = r*k + j  ->  col = j*64 + k;   bin may equal N (wrap callers
     * pass bin & (N-1) themselves). */
    return (size_t)((bin & (r - 1)) * 64 + (bin / r));
}

/* ── mapped conjugate split (fwd door) and merge (bwd door) ─────────── */

/* Per pair-row: Z (packed spectrum in DIF map order, length N) →
 * X1 = even real row's half-spectrum, X2 = odd's; scalar per bin (row-
 * blocked, everything L1-hot; H = N/2+1 bins per output row). */
static inline void _stw_split_row(const double *zr, const double *zi,
                                  double *x1r, double *x1i,
                                  double *x2r, double *x2i, int N, int r)
{
    const int h = N / 2;
    for (int f = 0; f <= h; f++) {
        int g = (N - f) & (N - 1);
        size_t cf = _stw_map(f, r), cg = _stw_map(g, r);
        double Zr = zr[cf], Zi = zi[cf], Gr = zr[cg], Gi = zi[cg];
        x1r[f] = 0.5 * (Zr + Gr);
        x1i[f] = 0.5 * (Zi - Gi);
        x2r[f] = 0.5 * (Zi + Gi);
        x2i[f] = 0.5 * (Gr - Zr);
    }
}

static inline void _stw_merge_row(const double *x1r, const double *x1i,
                                  const double *x2r, const double *x2i,
                                  double *zr, double *zi, int N, int r)
{
    const int h = N / 2;
    for (int f = 0; f <= h; f++) {
        size_t cf = _stw_map(f, r);
        zr[cf] = x1r[f] - x2i[f];
        zi[cf] = x1i[f] + x2r[f];
        if (f >= 1 && f <= h - 1) {
            size_t cg = _stw_map(N - f, r);
            zr[cg] = x1r[f] + x2i[f];
            zi[cg] = x2r[f] - x1i[f];
        }
    }
}

/* ── row-blocked compositions (the §6a40 law: one DRAM pass) ────────── */

/* fwd: real rows (row-major, pitch rs_in) → half-spectra rows (pitch
 * out_stride ≥ N/2+1). rows must be a multiple of 8. work: caller scratch,
 * ≥ 2*8*N doubles (one 8-row block, re+im pair-packed). */
static inline void _stw_r2c_fwd(const _stw_tables_t *t,
                                const double *x, double *out_re,
                                double *out_im, size_t rs_in,
                                size_t out_stride, size_t rows, double *work)
{
    const int N = t->N, r = t->r;
    double *wre = work, *wim = work + 4 * (size_t)N; /* 4 pairs per block */
    for (size_t b = 0; b < rows; b += 8) {
        const double *blk = x + b * rs_in;
        /* pair packing: re = even rows, im = odd; source PAIR stride is
         * 2*rs_in; work rows are N-contiguous. */
        if (r == 2)
            _stw_front2_fwd(blk, blk + rs_in, wre, wim,
                            2 * rs_in, (size_t)N, 4, N, t->twr, t->twi);
        else
            _stw_front4_fwd(blk, blk + rs_in, wre, wim,
                            2 * rs_in, (size_t)N, 4, N, t->twr, t->twi);
        for (size_t j = 0; j < (size_t)r; j++)
            radix64_n1_fwd_avx2_strided(wre + j * 64, wim + j * 64,
                                        0, 0, (size_t)N, 4);
        for (size_t p = 0; p < 4; p++)
            _stw_split_row(wre + p * (size_t)N, wim + p * (size_t)N,
                out_re + (b + 2 * p) * out_stride,
                out_im + (b + 2 * p) * out_stride,
                out_re + (b + 2 * p + 1) * out_stride,
                out_im + (b + 2 * p + 1) * out_stride, N, r);
    }
}

/* bwd: half-spectra rows (pitch in_stride) → UNNORMALIZED real rows
 * (pitch rs_out); rows % 8 == 0; work as in fwd. */
static inline void _stw_c2r_bwd(const _stw_tables_t *t,
                                const double *in_re, const double *in_im,
                                double *x, size_t in_stride, size_t rs_out,
                                size_t rows, double *work)
{
    const int N = t->N, r = t->r;
    double *wre = work, *wim = work + 4 * (size_t)N;
    for (size_t b = 0; b < rows; b += 8) {
        for (size_t p = 0; p < 4; p++)
            _stw_merge_row(
                in_re + (b + 2 * p) * in_stride,
                in_im + (b + 2 * p) * in_stride,
                in_re + (b + 2 * p + 1) * in_stride,
                in_im + (b + 2 * p + 1) * in_stride,
                wre + p * (size_t)N, wim + p * (size_t)N, N, r);
        for (size_t j = 0; j < (size_t)r; j++)
            radix64_n1_bwd_avx2_strided(wre + j * 64, wim + j * 64,
                                        0, 0, (size_t)N, 4);
        if (r == 2)
            _stw_front2_bwd(wre, wim, (size_t)N, 4, N, t->twr, t->twi);
        else
            _stw_front4_bwd(wre, wim, (size_t)N, 4, N, t->twr, t->twi);
        double *blk = x + b * rs_out;
        for (size_t p = 0; p < 4; p++) {
            memcpy(blk + (2 * p) * rs_out,     wre + p * (size_t)N,
                   (size_t)N * sizeof(double));
            memcpy(blk + (2 * p + 1) * rs_out, wim + p * (size_t)N,
                   (size_t)N * sizeof(double));
        }
    }
}

#endif /* VFFT_STRIDED_TW_H */
