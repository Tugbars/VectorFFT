/* natorder_2d.h — shared 2D natural-order axis reorder-tape builder.
 *
 * Probe one inner plan (row or column axis) on an impulse, orientation-detect the digit-reversal
 * permutation, and build its reorder tape: a PAIR list when the chain is palindromic (M is an
 * involution => independent swaps, no cycle-following dependency chain — the PSWAP fast path), else a
 * CYCLE list. num_stages<=1 => the axis is already natural (single radix = identity digit-rev, or a
 * prime override with natural output) => FREE (list left NULL). Returns 1 (ok) or 0 (detect failed =>
 * caller must refuse natural).
 *
 * Shared by the runtime 2D create (vfft.c) and the natural-aware 2D calibrator (fft2d_c2c_planner.h)
 * so both build tapes identically. See fft2d natural §.
 */
#ifndef VFFT_NATORDER_2D_H
#define VFFT_NATORDER_2D_H

#include "natorder_perm.h"   /* vfft_natorder_detect / mk_pairs / mk_cycles */

static inline int vfft_natorder_2d_build_axis(int N, const stride_plan_t *inner,
                                              int **out_list, int *out_is_pairs, int try_pairs)
{
    *out_list = NULL;
    *out_is_pairs = 0;
    if (inner->num_stages <= 1)
        return 1;                                   /* FREE: already natural on this axis */
    size_t K = inner->K, tot = (size_t)N * K;
    double *cre = (double *)calloc(tot, sizeof(double));
    double *cim = (double *)calloc(tot, sizeof(double));
    int *M = NULL;
    if (cre && cim)
    {
        cre[K] = 1.0;                               /* impulse at n0=1, lane 0 (row 1) */
        vfft_proto_execute_fwd((stride_plan_t *)inner, cre, cim, K);
        M = vfft_natorder_detect(N, inner->factors, inner->num_stages, K, cre, cim, 1);
    }
    free(cre);
    free(cim);
    if (!M)
        return 0;                                   /* orientation not detected => refuse natural */
    if (try_pairs)
    {
        int *pairs = vfft_natorder_mk_pairs(N, M);  /* NULL unless M is an involution (palindrome) */
        if (pairs)
        {
            *out_list = pairs;
            *out_is_pairs = 1;
            free(M);
            return 1;
        }
    }
    *out_list = vfft_natorder_mk_cycles(N, M);
    free(M);
    return *out_list != NULL;
}

/* Palindromic candidate chains for N: (a,a), (a,b,a), uniform a^m — radix must have an in-place codelet
 * (reg->n1_fwd). A palindromic chain's digit-reversal is an INVOLUTION => the natural-order reorder is
 * cheap independent pair-swaps, so these are the natorder-friendly factorizations the SCRAMBLED DP pool
 * prunes (they lose the FFT-only race). The natural-aware calibrator INJECTS them so the joint FFT+reorder
 * scoring can pick one. Distinct-name copy of natorder_calibrate.h's vfft_natorder_palindromes (that
 * header + this one are both included by vfft.c => can't share the symbol). Returns count (<= max). */
static inline int vfft_natorder_2d_palindromes(int N, const vfft_proto_registry_t *reg,
                                               int chains[][STRIDE_MAX_STAGES], int *nfs, int max)
{
    int cnt = 0;
    for (int a = 2; a <= 64 && cnt < max; a++) {
        if (!(a < VFFT_PROTO_REG_MAX_RADIX && reg->n1_fwd[a])) continue;
        long long aa = (long long)a * a;
        if (aa == N) { chains[cnt][0] = a; chains[cnt][1] = a; nfs[cnt++] = 2; continue; }
        if (aa < N && N % aa == 0) {
            int b = (int)(N / aa);
            if (b > 1 && b < VFFT_PROTO_REG_MAX_RADIX && reg->n1_fwd[b] && cnt < max) {
                chains[cnt][0] = a; chains[cnt][1] = b; chains[cnt][2] = a; nfs[cnt++] = 3;
            }
            if (cnt < max) {                            /* uniform a^m (m>=4): palindromic */
                long long p = aa; int m = 2;
                while (p < N && m < STRIDE_MAX_STAGES) { p *= a; m++; }
                if (p == N && m >= 4) {
                    for (int s = 0; s < m; s++) chains[cnt][s] = a;
                    nfs[cnt++] = m;
                }
            }
        }
    }
    return cnt;
}

#endif /* VFFT_NATORDER_2D_H */
