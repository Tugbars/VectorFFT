/* natorder_exec.h — execute-time reorder passes for VFFT_ORDER_NATURAL.
 *
 * Two mechanisms (the P1a set; SCRATCH terminator + LEAF-IP arrive in P1b):
 *   cycle pass — walk the plan-time cycle list, moving K-double rows with AVX2 + software
 *                prefetch (the T7 "cycle-UB" kernel: measured ceiling for the PURE mechanism,
 *                +16–28% at K≥32; interleaving/tiling/transpose-recursion all measured slower).
 *   pair pass  — involution swaps for PSWAP verdicts (injected palindromic chains).
 *
 * Rows are K doubles per split plane. Unlike the probes (K multiples of 4 only), these
 * carry a scalar tail so any K ≥ 1 is correct — including K=1..3 where rows are sub-vector.
 * ST only in P1a; MT splits by ROW RANGES (never K-split — thin sub-rows are the measured
 * catastrophic regime). natural_order_inplace_design.md §2e. */
#ifndef VFFT_NATORDER_EXEC_H
#define VFFT_NATORDER_EXEC_H

#include <string.h>
#include <immintrin.h>

static inline void vfft_natorder_row_mov(double *dr, double *di,
                                         const double *sr, const double *si, size_t K)
{
    size_t c = 0;
    for (; c + 4 <= K; c += 4) {
        _mm256_storeu_pd(dr + c, _mm256_loadu_pd(sr + c));
        _mm256_storeu_pd(di + c, _mm256_loadu_pd(si + c));
    }
    for (; c < K; c++) { dr[c] = sr[c]; di[c] = si[c]; }
}

static inline void vfft_natorder_row_swap(double *ar, double *ai,
                                          double *br, double *bi, size_t K)
{
    size_t c = 0;
    for (; c + 4 <= K; c += 4) {
        __m256d xr = _mm256_loadu_pd(ar + c), xi = _mm256_loadu_pd(ai + c);
        __m256d yr = _mm256_loadu_pd(br + c), yi = _mm256_loadu_pd(bi + c);
        _mm256_storeu_pd(ar + c, yr); _mm256_storeu_pd(ai + c, yi);
        _mm256_storeu_pd(br + c, xr); _mm256_storeu_pd(bi + c, xi);
    }
    for (; c < K; c++) {
        double tr = ar[c], ti = ai[c];
        ar[c] = br[c]; ai[c] = bi[c];
        br[c] = tr;    bi[c] = ti;
    }
}

/* Forward direction: rows currently SCRAMBLED, list built from M (natural[k]=scrambled[M[k]]).
 * Walks each cycle: save head, shift rows one step along the cycle, close from the temp.
 * tmp: caller-provided 2*K doubles (plan-owned; keeps this alloc-free on the hot path). */
static inline void vfft_natorder_cycle_pass(double *re, double *im, size_t K,
                                            const int *list, double *tmp)
{
    const int *L = list;
    double *tr = tmp, *ti = tmp + K;
    while (*L != -2) {
        const int *s = L;
        int len = 0;
        while (L[len] != -1) len++;
        memcpy(tr, re + (size_t)s[0] * K, K * 8);
        memcpy(ti, im + (size_t)s[0] * K, K * 8);
        for (int i = 0; i < len - 1; i++) {
            if (i + 4 < len) {
                _mm_prefetch((const char *)(re + (size_t)s[i + 4] * K), _MM_HINT_T0);
                _mm_prefetch((const char *)(im + (size_t)s[i + 4] * K), _MM_HINT_T0);
            }
            vfft_natorder_row_mov(re + (size_t)s[i] * K, im + (size_t)s[i] * K,
                                  re + (size_t)s[i + 1] * K, im + (size_t)s[i + 1] * K, K);
        }
        memcpy(re + (size_t)s[len - 1] * K, tr, K * 8);
        memcpy(im + (size_t)s[len - 1] * K, ti, K * 8);
        L += len + 1;
    }
}

/* Backward direction (natural spectrum in): the inverse reorder = walk each cycle in the
 * OPPOSITE shift direction. Same list, no second tape needed. */
static inline void vfft_natorder_cycle_pass_inv(double *re, double *im, size_t K,
                                                const int *list, double *tmp)
{
    const int *L = list;
    double *tr = tmp, *ti = tmp + K;
    while (*L != -2) {
        const int *s = L;
        int len = 0;
        while (L[len] != -1) len++;
        memcpy(tr, re + (size_t)s[len - 1] * K, K * 8);
        memcpy(ti, im + (size_t)s[len - 1] * K, K * 8);
        for (int i = len - 1; i > 0; i--) {
            if (i - 4 >= 0) {
                _mm_prefetch((const char *)(re + (size_t)s[i - 4] * K), _MM_HINT_T0);
                _mm_prefetch((const char *)(im + (size_t)s[i - 4] * K), _MM_HINT_T0);
            }
            vfft_natorder_row_mov(re + (size_t)s[i] * K, im + (size_t)s[i] * K,
                                  re + (size_t)s[i - 1] * K, im + (size_t)s[i - 1] * K, K);
        }
        memcpy(re + (size_t)s[0] * K, tr, K * 8);
        memcpy(im + (size_t)s[0] * K, ti, K * 8);
        L += len + 1;
    }
}

/* ── MT range kernels: process a disjoint CYCLE / PAIR range with full K-wide rows.
 * The reorder pass must split by ROW SETS, never by K (K-split makes 64B sub-rows — the
 * measured catastrophic regime, natural_order_inplace_design.md §2e integration note 2).
 * off[c] = start index of cycle c in the flattened list; off has ncyc+1 entries. tmp is
 * this worker's OWN 2*K scratch. inv toggles the backward (opposite shift) direction. */
static inline void vfft_natorder_cycle_range(double *re, double *im, size_t K,
                                             const int *list, const int *off,
                                             int c0, int c1, double *tmp, int inv)
{
    double *tr = tmp, *ti = tmp + K;
    for (int c = c0; c < c1; c++) {
        const int *s = list + off[c];
        int len = 0;
        while (s[len] != -1) len++;
        if (!inv) {
            memcpy(tr, re + (size_t)s[0] * K, K * 8);
            memcpy(ti, im + (size_t)s[0] * K, K * 8);
            for (int i = 0; i < len - 1; i++) {
                if (i + 4 < len) {
                    _mm_prefetch((const char *)(re + (size_t)s[i + 4] * K), _MM_HINT_T0);
                    _mm_prefetch((const char *)(im + (size_t)s[i + 4] * K), _MM_HINT_T0);
                }
                vfft_natorder_row_mov(re + (size_t)s[i] * K, im + (size_t)s[i] * K,
                                      re + (size_t)s[i + 1] * K, im + (size_t)s[i + 1] * K, K);
            }
            memcpy(re + (size_t)s[len - 1] * K, tr, K * 8);
            memcpy(im + (size_t)s[len - 1] * K, ti, K * 8);
        } else {
            memcpy(tr, re + (size_t)s[len - 1] * K, K * 8);
            memcpy(ti, im + (size_t)s[len - 1] * K, K * 8);
            for (int i = len - 1; i > 0; i--) {
                if (i - 4 >= 0) {
                    _mm_prefetch((const char *)(re + (size_t)s[i - 4] * K), _MM_HINT_T0);
                    _mm_prefetch((const char *)(im + (size_t)s[i - 4] * K), _MM_HINT_T0);
                }
                vfft_natorder_row_mov(re + (size_t)s[i] * K, im + (size_t)s[i] * K,
                                      re + (size_t)s[i - 1] * K, im + (size_t)s[i - 1] * K, K);
            }
            memcpy(re + (size_t)s[0] * K, tr, K * 8);
            memcpy(im + (size_t)s[0] * K, ti, K * 8);
        }
    }
}

/* Swap pairs [p0,p1) of a flat pair list [a0,b0,a1,b1,...]. Involution => dir-agnostic. */
static inline void vfft_natorder_pair_range(double *re, double *im, size_t K,
                                            const int *pairs, int p0, int p1)
{
    for (int i = p0; i < p1; i++) {
        if (i + 4 < p1) {
            _mm_prefetch((const char *)(re + (size_t)pairs[2 * (i + 4) + 1] * K), _MM_HINT_T0);
            _mm_prefetch((const char *)(im + (size_t)pairs[2 * (i + 4) + 1] * K), _MM_HINT_T0);
        }
        vfft_natorder_row_swap(re + (size_t)pairs[2 * i] * K,     im + (size_t)pairs[2 * i] * K,
                               re + (size_t)pairs[2 * i + 1] * K, im + (size_t)pairs[2 * i + 1] * K, K);
    }
}

/* Involution pair pass (PSWAP): self-inverse, so forward and backward are the same call. */
static inline void vfft_natorder_pair_pass(double *re, double *im, size_t K, const int *pairs)
{
    const int *L = pairs;
    int np = 0;
    while (L[2 * np] != -2) np++;
    for (int i = 0; i < np; i++) {
        if (i + 4 < np) {
            _mm_prefetch((const char *)(re + (size_t)L[2 * (i + 4) + 1] * K), _MM_HINT_T0);
            _mm_prefetch((const char *)(im + (size_t)L[2 * (i + 4) + 1] * K), _MM_HINT_T0);
        }
        vfft_natorder_row_swap(re + (size_t)L[2 * i] * K,     im + (size_t)L[2 * i] * K,
                               re + (size_t)L[2 * i + 1] * K, im + (size_t)L[2 * i + 1] * K, K);
    }
}

#endif /* VFFT_NATORDER_EXEC_H */
