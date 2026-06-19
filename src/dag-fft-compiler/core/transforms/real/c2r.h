/* c2r.h — backward real transform (halfcomplex -> real), the inverse of
 * rfft.h's forward cascade. Unnormalized: r2c followed by c2r gives N*x.
 *
 * Execution structure (mirror of rfft, FFTW's hc2r/apply_dif duality):
 *   forward:  leaf (r2cf) FIRST -> stages d = nf-2 .. 0 (DIT)     -> packed out
 *   backward: stages d = 0 .. nf-2 (DIF backward) -> leaf (r2cb) LAST -> real out
 */
#ifndef VFFT_C2R_H
#define VFFT_C2R_H

#include <immintrin.h>
#include "rfft.h"

typedef struct {
    rfft_plan_t *base;
    rfft_r2cb_fn leaf;
    rfft_hc_fn   stage_hc[VFFT_RFFT_MAX_STAGES];
    rfft_hc_rng_fn stage_hcr[VFFT_RFFT_MAX_STAGES];
    rfft_r2cb_fn stage_dc[VFFT_RFFT_MAX_STAGES];
    size_t Kb;
    double      *mid_inv[VFFT_RFFT_MAX_STAGES];
} c2r_plan_t;

static inline void c2r_plan_destroy(c2r_plan_t *p)
{
    if (!p) return;
    for (int d = 0; d < VFFT_RFFT_MAX_STAGES; d++) free(p->mid_inv[d]);
    if (p->base) rfft_plan_destroy(p->base);
    free(p);
}

static inline double *c2r_build_mid_inv(const rfft_stage_t *st)
{
    const int r = st->radix, m = st->m, np = st->np;
    double *M = (double *)malloc((size_t)r * r * 8);
    double *A = (double *)malloc((size_t)r * r * 8);
    double *inv = (double *)malloc((size_t)r * r * 8);
    if (!M || !A || !inv) { free(M); free(A); free(inv); return NULL; }
    for (int t = 0; t < r; t++) {
        int pp = m / 2 + t * m;
        for (int j = 0; j < r; j++)
            M[t * r + j] = (pp <= np / 2) ? st->mid_c[t * r + j]
                                          : st->mid_s[(r - 1 - t) * r + j];
    }
    memcpy(A, M, (size_t)r * r * 8);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < r; j++) inv[i * r + j] = (i == j) ? 1.0 : 0.0;
    for (int col = 0; col < r; col++) {
        int piv = col;
        for (int i = col + 1; i < r; i++)
            if (fabs(A[i * r + col]) > fabs(A[piv * r + col])) piv = i;
        if (fabs(A[piv * r + col]) < 1e-12) { free(M); free(A); free(inv); return NULL; }
        if (piv != col)
            for (int j = 0; j < r; j++) {
                double t1 = A[col * r + j]; A[col * r + j] = A[piv * r + j]; A[piv * r + j] = t1;
                double t2 = inv[col * r + j]; inv[col * r + j] = inv[piv * r + j]; inv[piv * r + j] = t2;
            }
        double d = 1.0 / A[col * r + col];
        for (int j = 0; j < r; j++) { A[col * r + j] *= d; inv[col * r + j] *= d; }
        for (int i = 0; i < r; i++) {
            if (i == col) continue;
            double f = A[i * r + col];
            if (f == 0.0) continue;
            for (int j = 0; j < r; j++) {
                A[i * r + j] -= f * A[col * r + j];
                inv[i * r + j] -= f * inv[col * r + j];
            }
        }
    }
    for (int i = 0; i < r * r; i++) inv[i] *= (double)r;
    free(M); free(A);
    return inv;
}

/* c2r_plan_create_ex — variant-explicit plan build (mirror of rfft_plan_create_ex).
 * `variant[d]` selects the DIF-backward combine codelet for combine stage d
 * (d = 0..nf-2): 0=FLAT, 1=LOG3, 2=T1S(ranged). variant==NULL => default policy
 * (LOG3-if-present + ranged), the legacy behavior. The leaf (factors[nf-1]) takes
 * no variant. CRITICAL: stage_hcr MUST be cleared for FLAT/LOG3 or the executor's
 * `if (stage_hcr[d])` ranged branch fires regardless of stage_hc. */
static inline c2r_plan_t *c2r_plan_create_ex(int N, size_t K,
                                             const int *factors, int nf,
                                             const int *variant,
                                             const rfft_codelets_t *reg)
{
    if (nf < 1 || nf > VFFT_RFFT_MAX_STAGES) return NULL;
    if (K == 0 || (K % 8) != 0) return NULL;
    c2r_plan_t *p = (c2r_plan_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    int leaf_r = factors[nf - 1];
    if (leaf_r > VFFT_RFFT_MAX_RADIX || !reg->r2cb[leaf_r]) { free(p); return NULL; }
    rfft_plan_t *base = rfft_plan_create(N, K, factors, nf, reg);
    if (!base) { free(p); return NULL; }
    p->base = base;
    p->leaf = reg->r2cb[leaf_r];
    for (int d = nf - 2; d >= 0; d--) {
        int r = base->st[d].radix;
        int v = variant ? variant[d] : -1;   /* -1 = default policy */
        rfft_hc_fn hc; rfft_hc_rng_fn hcr;
        if (v == 0) {                          /* FLAT */
            hc = reg->hc2hc_dif_bwd[r]; hcr = NULL;
        } else if (v == 1) {                   /* LOG3 */
            hc = reg->hc2hc_dif_bwd_log3[r] ? reg->hc2hc_dif_bwd_log3[r] : reg->hc2hc_dif_bwd[r];
            hcr = NULL;
        } else if (v == 2) {                   /* T1S (ranged) */
            hc = reg->hc2hc_dif_bwd[r];
#ifdef VFFT_RFFT_RANGED
            hcr = reg->hc2hc_dif_rng_bwd[r];
#else
            hcr = NULL;
#endif
        } else {                               /* default policy (legacy) */
            hc = reg->hc2hc_dif_bwd_log3[r] ? reg->hc2hc_dif_bwd_log3[r] : reg->hc2hc_dif_bwd[r];
            hcr = reg->hc2hc_dif_rng_bwd[r];
        }
        if (!hc || !reg->r2cb[r]) { c2r_plan_destroy(p); return NULL; }
        p->stage_hc[d] = hc;
        p->stage_hcr[d] = hcr;
        p->stage_dc[d] = reg->r2cb[r];
        if (base->st[d].has_mid) {
            p->mid_inv[d] = c2r_build_mid_inv(&base->st[d]);
            if (!p->mid_inv[d]) { c2r_plan_destroy(p); return NULL; }
        }
    }
    p->Kb = K;
    {
        const char *e = getenv("VFFT_C2R_KB");
        if (e) {
            size_t kb = (size_t)atol(e);
            if (kb >= 8 && kb <= K && (kb % 8) == 0) p->Kb = kb;
        }
    }
    return p;
}

/* Default-policy wrapper (legacy behavior, variant=NULL). */
static inline c2r_plan_t *c2r_plan_create(int N, size_t K,
                                          const int *factors, int nf,
                                          const rfft_codelets_t *reg)
{
    return c2r_plan_create_ex(N, K, factors, nf, NULL, reg);
}

static inline void c2r_mid_inv_column(int r, int m, size_t Q, size_t K,
                                      size_t vl, const double *src,
                                      const double *Minv, double *dst_base)
{
    size_t v = 0;
#if defined(__AVX512F__)
    __m512d in[32];
    for (; v + 8 <= vl; v += 8) {
        for (int t = 0; t < r; t++)
            in[t] = _mm512_loadu_pd(src + (Q * (size_t)(m / 2 + t * m)) * K + v);
        for (int j = 0; j < r; j++) {
            __m512d acc = _mm512_setzero_pd();
            for (int t = 0; t < r; t++)
                acc = _mm512_fmadd_pd(_mm512_set1_pd(Minv[j * r + t]), in[t], acc);
            _mm512_storeu_pd(dst_base + (size_t)j * (Q * K) + v, acc);
        }
    }
#elif defined(__AVX2__)
    __m256d in[32];
    for (; v + 4 <= vl; v += 4) {
        for (int t = 0; t < r; t++)
            in[t] = _mm256_loadu_pd(src + (Q * (size_t)(m / 2 + t * m)) * K + v);
        for (int j = 0; j < r; j++) {
            __m256d acc = _mm256_setzero_pd();
            for (int t = 0; t < r; t++)
                acc = _mm256_fmadd_pd(_mm256_set1_pd(Minv[j * r + t]), in[t], acc);
            _mm256_storeu_pd(dst_base + (size_t)j * (Q * K) + v, acc);
        }
    }
#endif
    for (; v < vl; v++) {
        for (int j = 0; j < r; j++) {
            double acc = 0.0;
            for (int t = 0; t < r; t++)
                acc += Minv[j * r + t] * src[(Q * (size_t)(m / 2 + t * m)) * K + v];
            dst_base[(size_t)j * (Q * K) + v] = acc;
        }
    }
}

static inline void c2r_execute_packed(const c2r_plan_t *p,
                                      const double *in, double *out)
{
    const rfft_plan_t *b = p->base;
    const int N = b->N;
    const size_t K = b->K;
    const size_t NK = (size_t)N * K;
    if (b->nf == 1) {
        const ptrdiff_t SK = (ptrdiff_t)(b->S * K);
        p->leaf(in, in + NK, out, SK, -SK, SK, b->S * K);
        return;
    }
    for (size_t bb = 0; bb < K; bb += p->Kb) {
        const size_t bw = (bb + p->Kb <= K) ? p->Kb : (K - bb);
        const double *src = in;
        double *dst = b->planeA;
        for (int d = 0; d <= b->nf - 2; d++) {
            const rfft_stage_t *st = &b->st[d];
            const int r = st->radix, m = st->m;
            const size_t Q = st->Q;
            const ptrdiff_t QK = (ptrdiff_t)(Q * K);
            const ptrdiff_t QmK = (ptrdiff_t)(Q * (size_t)m * K);
            const int folded = (bw == K && bb == 0 && Q > 1);
            const size_t Qfold = folded ? 1 : Q;
            const size_t vlf = folded ? (Q * K) : bw;
            dst = (d % 2 == 0) ? b->planeA : b->planeB;
            for (size_t q = 0; q < Qfold; q++) {
                const double *srcq = src + q * K + bb;
                double *dstq = dst + q * K + bb;
                p->stage_dc[d](srcq, srcq + NK, dstq, QmK, -QmK, QK, vlf);
                if (p->stage_hcr[d] && st->kmax >= 1) {
                    p->stage_hcr[d](
                        srcq + (Q * (size_t)1) * K,
                        srcq + (Q * (size_t)(m - 1)) * K,
                        dstq + (Q * (size_t)(r * 1)) * K,
                        dstq + (Q * (size_t)(r * (m - 1))) * K,
                        st->tw_re, st->tw_im,
                        QmK, QK,
                        (ptrdiff_t)(Q * K),
                        (ptrdiff_t)(Q * (size_t)r * K),
                        st->kmax, vlf);
                } else
                for (int k = 1; k <= st->kmax; k++)
                    p->stage_hc[d](srcq + (Q * (size_t)k) * K,
                                   srcq + (Q * (size_t)(m - k)) * K,
                                   dstq + (Q * (size_t)(r * k)) * K,
                                   dstq + (Q * (size_t)(r * (m - k))) * K,
                                   st->tw_re + (size_t)(k - 1) * r,
                                   st->tw_im + (size_t)(k - 1) * r,
                                   QmK, QK, vlf);
                if (st->has_mid)
                    c2r_mid_inv_column(r, m, Q, K, vlf, srcq, p->mid_inv[d],
                                       dstq + (Q * (size_t)(r * (m / 2))) * K);
            }
            src = dst;
        }
        if (bw == K) {
            const ptrdiff_t SK = (ptrdiff_t)(b->S * K);
            p->leaf(src, src + NK, out, SK, -SK, SK, b->S * K);
        } else {
            const ptrdiff_t SK = (ptrdiff_t)(b->S * K);
            for (size_t g = 0; g < b->S; g++)
                p->leaf(src + g * K + bb, src + g * K + bb + NK,
                        out + g * K + bb, SK, -SK, SK, bw);
        }
    }
}

#endif
