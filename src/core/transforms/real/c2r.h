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
    /* stage-0 natural INITIATOR codelet (hc2c_nat_bwd): reads the SPLIT
     * half-spectrum interior columns directly. Set iff reg->hc2c_bwd[st0]
     * exists (nf>=2). NULL -> no natural path for this plan (c2r_execute_natural
     * not callable; the packed/stride paths are unaffected). */
    rfft_hc2c_nat_bwd_fn nat_init;
    /* Resolved JIT executor (void* to stay type-agnostic here; cast to c2r_jit_fn
     * by vfft_c2r_execute). NULL -> use the generic c2r_execute_packed. Set by the
     * dispatch under VFFT_USE_JIT = compiled+cached for the winning plan. */
    void        *jit_exec;
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
    if (K == 0) return NULL; /* arbitrary-K: rem-aware tail handles K % VW != 0 (was K%8-gated) */
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
    /* stage-0 natural initiator codelet (split-input). Optional: NULL leaves the
     * packed/stride paths untouched and just makes c2r_execute_natural unavailable. */
    if (nf >= 2)
        p->nat_init = reg->hc2c_bwd[base->st[0].radix];
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

/* Split-input mid (k=m/2) inverse, the natural analog of c2r_mid_inv_column.
 * The forward natural mid (rfft_mid_column, mode 1) stored, for each sI with
 * pp = m/2 + sI*m <= nh: acc_r[sI] at in_re[pp*K], acc_i[sI] at in_im[pp*K].
 * The packed slot t (= position m/2 + t*m) holds acc_r[t] if pp_t <= nh, else
 * the mirror's acc_i (sI = r-1-t, read from in_im). Reconstruct mid[t] from the
 * split planes, then apply the inverse matrix Minv -> dst (the mid column legs). */
static inline void c2r_mid_inv_column_natural(int r, int m, size_t K, size_t vl,
                                              size_t nh, const double *in_re,
                                              const double *in_im,
                                              const double *Minv, double *dst_base)
{
    for (size_t v = 0; v < vl; v++) {
        double mid[64];
        for (int t = 0; t < r; t++) {
            size_t pp = (size_t)(m / 2) + (size_t)t * (size_t)m;
            if (pp <= nh)
                mid[t] = in_re[pp * K + v];
            else {
                int sI = r - 1 - t;
                size_t ppm = (size_t)(m / 2) + (size_t)sI * (size_t)m;
                mid[t] = in_im[ppm * K + v];
            }
        }
        for (int j = 0; j < r; j++) {
            double acc = 0.0;
            for (int t = 0; t < r; t++) acc += Minv[j * r + t] * mid[t];
            dst_base[(size_t)j * K + v] = acc;
        }
    }
}

/* c2r_execute_natural — the SPLIT-input c2r (the inverse of rfft_execute_fwd_natural).
 * in_re/in_im are the (N/2+1)xK natural half-spectrum (row f = frequency f). The
 * stage-0 initiator inverts the forward terminator (interior via hc2c_nat_bwd, DC/
 * Nyquist via inverse-gather + r2cb, mid via the split mid inverse), writing the
 * packed cascade intermediate planeA; stages 1..nf-2 and the leaf then run exactly
 * as c2r_execute_packed. No repack — the mirror of how the forward terminator gives
 * split output at packed speed. Requires nf>=2 and p->nat_init != NULL. Full width
 * (no Kb blocking) — the natural path is the low/mid-K winner. out = N*x. */
static inline void c2r_execute_natural(const c2r_plan_t *p,
                                       const double *in_re, const double *in_im,
                                       double *out)
{
    const rfft_plan_t *b = p->base;
    const int N = b->N;
    const size_t K = b->K;
    const size_t NK = (size_t)N * K;
    const size_t nh = (size_t)(N / 2);

    /* ===== STAGE 0 INITIATOR: split half-spectrum -> planeA (packed cascade input) ===== */
    {
        const rfft_stage_t *st0 = &b->st[0];
        const int r = st0->radix, m = st0->m;
        const ptrdiff_t mK = (ptrdiff_t)((size_t)m * K);
        double *cur0 = b->planeA;

        /* (1) interior columns k=1..(m-1)/2: hc2c_nat_bwd reads split rows k,m-k */
        for (int k = 1; k <= (m - 1) / 2; k++)
            p->nat_init(in_re + (size_t)k * K, in_im + (size_t)k * K,
                        in_re + (size_t)(m - k) * K, in_im + (size_t)(m - k) * K,
                        cur0 + ((size_t)(r * k)) * K, cur0 + ((size_t)(r * (m - k))) * K,
                        st0->tw_re + (size_t)(k - 1) * r, st0->tw_im + (size_t)(k - 1) * r,
                        mK, mK, (ptrdiff_t)K, K);

        /* (2) DC / sub-DC / Nyquist: inverse-gather natural rows into nat_k0 (the
         * r2cf-output halfcomplex layout), then r2cb -> cur0 col-0 group. */
        double *nk = b->nat_k0;
        memcpy(nk, in_re, K * 8);
        for (int sI = 1; sI < (r + 1) / 2; sI++) {
            memcpy(nk + (size_t)sI * K, in_re + (size_t)sI * (size_t)m * K, K * 8);
            memcpy(nk + (size_t)(r - sI) * K, in_im + (size_t)sI * (size_t)m * K, K * 8);
        }
        if (r % 2 == 0)
            memcpy(nk + (size_t)(r / 2) * K, in_re + nh * K, K * 8);
        p->stage_dc[0](nk, nk + (size_t)r * K, cur0,
                       (ptrdiff_t)K, -(ptrdiff_t)K, (ptrdiff_t)K, K);

        /* (3) mid column (k=m/2): split mid inverse -> cur0 col r*(m/2). */
        if (st0->has_mid)
            c2r_mid_inv_column_natural(r, m, K, K, nh, in_re, in_im,
                                       p->mid_inv[0], cur0 + ((size_t)(r * (m / 2))) * K);
    }

    /* ===== STAGES d=1..nf-2: identical to c2r_execute_packed (full width) ===== */
    double *src = b->planeA;
    for (int d = 1; d <= b->nf - 2; d++) {
        const rfft_stage_t *st = &b->st[d];
        const int r = st->radix, m = st->m;
        const size_t Q = st->Q;
        const ptrdiff_t QK = (ptrdiff_t)(Q * K);
        const ptrdiff_t QmK = (ptrdiff_t)(Q * (size_t)m * K);
        const int folded = (Q > 1);
        const size_t Qfold = folded ? 1 : Q;
        const size_t vlf = folded ? (Q * K) : K;
        double *dst = (d % 2 == 0) ? b->planeA : b->planeB;
        for (size_t q = 0; q < Qfold; q++) {
            const double *srcq = src + q * K;
            double *dstq = dst + q * K;
            p->stage_dc[d](srcq, srcq + NK, dstq, QmK, -QmK, QK, vlf);
            if (p->stage_hcr[d] && st->kmax >= 1) {
                p->stage_hcr[d](
                    srcq + (Q * (size_t)1) * K, srcq + (Q * (size_t)(m - 1)) * K,
                    dstq + (Q * (size_t)(r * 1)) * K,
                    dstq + (Q * (size_t)(r * (m - 1))) * K,
                    st->tw_re, st->tw_im, QmK, QK,
                    (ptrdiff_t)(Q * K), (ptrdiff_t)(Q * (size_t)r * K), st->kmax, vlf);
            } else
                for (int k = 1; k <= st->kmax; k++)
                    p->stage_hc[d](srcq + (Q * (size_t)k) * K,
                                   srcq + (Q * (size_t)(m - k)) * K,
                                   dstq + (Q * (size_t)(r * k)) * K,
                                   dstq + (Q * (size_t)(r * (m - k))) * K,
                                   st->tw_re + (size_t)(k - 1) * r,
                                   st->tw_im + (size_t)(k - 1) * r, QmK, QK, vlf);
            if (st->has_mid)
                c2r_mid_inv_column(r, m, Q, K, vlf, srcq, p->mid_inv[d],
                                   dstq + (Q * (size_t)(r * (m / 2))) * K);
        }
        src = dst;
    }

    /* ===== LEAF -> out (real) ===== */
    {
        const ptrdiff_t SK = (ptrdiff_t)(b->S * K);
        p->leaf(src, src + NK, out, SK, -SK, SK, b->S * K);
    }
}

/* c2r_execute_natural, LANE-RANGE [k0,k0+kw): process only lanes [k0,kw) of the K
 * batch — the MT building block (mirror of rfft_execute_fwd_natural_range). Per-q
 * (NOT folded — the fold needs full width); all scratch (planeA/planeB/nat_k0) is
 * lane-indexed (+k0), so concurrent calls on disjoint [k0,kw) never collide. kw need
 * not be 8-aligned (codelet scalar tail covers the remainder). Requires nf>=2. */
static inline void c2r_execute_natural_range(const c2r_plan_t *p,
                                             const double *in_re, const double *in_im,
                                             double *out, size_t k0, size_t kw)
{
    const rfft_plan_t *b = p->base;
    const int N = b->N;
    const size_t K = b->K;
    const size_t NK = (size_t)N * K;
    const size_t nh = (size_t)(N / 2);

    /* ===== STAGE 0 INITIATOR (lane range) ===== */
    {
        const rfft_stage_t *st0 = &b->st[0];
        const int r = st0->radix, m = st0->m;
        const ptrdiff_t mK = (ptrdiff_t)((size_t)m * K);
        double *cur0 = b->planeA;
        for (int k = 1; k <= (m - 1) / 2; k++)
            p->nat_init(in_re + (size_t)k * K + k0, in_im + (size_t)k * K + k0,
                        in_re + (size_t)(m - k) * K + k0, in_im + (size_t)(m - k) * K + k0,
                        cur0 + ((size_t)(r * k)) * K + k0, cur0 + ((size_t)(r * (m - k))) * K + k0,
                        st0->tw_re + (size_t)(k - 1) * r, st0->tw_im + (size_t)(k - 1) * r,
                        mK, mK, (ptrdiff_t)K, kw);
        double *nk = b->nat_k0;
        memcpy(nk + k0, in_re + k0, kw * 8);
        for (int sI = 1; sI < (r + 1) / 2; sI++) {
            memcpy(nk + (size_t)sI * K + k0, in_re + (size_t)sI * (size_t)m * K + k0, kw * 8);
            memcpy(nk + (size_t)(r - sI) * K + k0, in_im + (size_t)sI * (size_t)m * K + k0, kw * 8);
        }
        if (r % 2 == 0)
            memcpy(nk + (size_t)(r / 2) * K + k0, in_re + nh * K + k0, kw * 8);
        p->stage_dc[0](nk + k0, nk + (size_t)r * K + k0, cur0 + k0,
                       (ptrdiff_t)K, -(ptrdiff_t)K, (ptrdiff_t)K, kw);
        if (st0->has_mid)
            c2r_mid_inv_column_natural(r, m, K, kw, nh, in_re + k0, in_im + k0,
                                       p->mid_inv[0], cur0 + ((size_t)(r * (m / 2))) * K + k0);
    }

    /* ===== STAGES d=1..nf-2 (per-q, lane range) ===== */
    double *src = b->planeA;
    for (int d = 1; d <= b->nf - 2; d++) {
        const rfft_stage_t *st = &b->st[d];
        const int r = st->radix, m = st->m;
        const size_t Q = st->Q;
        const ptrdiff_t QK = (ptrdiff_t)(Q * K);
        const ptrdiff_t QmK = (ptrdiff_t)(Q * (size_t)m * K);
        double *dst = (d % 2 == 0) ? b->planeA : b->planeB;
        for (size_t q = 0; q < Q; q++) {
            const double *srcq = src + q * K + k0;
            double *dstq = dst + q * K + k0;
            p->stage_dc[d](srcq, srcq + NK, dstq, QmK, -QmK, QK, kw);
            for (int k = 1; k <= st->kmax; k++)
                p->stage_hc[d](srcq + (Q * (size_t)k) * K,
                               srcq + (Q * (size_t)(m - k)) * K,
                               dstq + (Q * (size_t)(r * k)) * K,
                               dstq + (Q * (size_t)(r * (m - k))) * K,
                               st->tw_re + (size_t)(k - 1) * r,
                               st->tw_im + (size_t)(k - 1) * r, QmK, QK, kw);
            if (st->has_mid)
                c2r_mid_inv_column(r, m, Q, K, kw, srcq, p->mid_inv[d],
                                   dstq + (Q * (size_t)(r * (m / 2))) * K);
        }
        src = dst;
    }

    /* ===== LEAF -> out (per-g, lane range) ===== */
    {
        const ptrdiff_t SK = (ptrdiff_t)(b->S * K);
        for (size_t g = 0; g < b->S; g++)
            p->leaf(src + g * K + k0, src + g * K + k0 + NK, out + g * K + k0,
                    SK, -SK, SK, kw);
    }
}

#endif
