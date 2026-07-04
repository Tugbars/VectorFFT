/* natorder_perm.h — plan-time permutation machinery for VFFT_ORDER_NATURAL (in-place 1D c2c).
 *
 * The in-place forward emits bin n at row perm[n], where perm is the mixed-radix digit reversal
 * of the plan's factor chain. Probes proved the ORIENTATION (fwd- vs reversed-factor-order,
 * perm vs iperm) must be AUTO-DETECTED per plan, not assumed — the public path matched fwd-order
 * even on DIF-calibrated cells (natural_order_inplace_design.md §2b). Detection here is exact
 * and cheap: FFT one impulse δ[n0] once at plan time; the true spectrum is the closed form
 * X[k] = e^{-2πi·k·n0/N}, so each candidate map M is checked as scrambled[M[k]] ≈ X[k] at a
 * handful of k — no naive DFT, no tolerance games (errs are ~1e-12 vs O(1) for wrong maps).
 *
 * Also builds the execute-time tapes: flattened cycle list (PURE_CYCLE) and pair list (PSWAP,
 * involutions only). All builders are plan-time only — never on the execute path. */
#ifndef VFFT_NATORDER_PERM_H
#define VFFT_NATORDER_PERM_H

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef VFFT_NATORDER_PI
#define VFFT_NATORDER_PI 3.14159265358979323846
#endif

/* Mixed-radix digit reversal over chain f[0..nf-1]: little-endian digits of n become
 * big-endian slot digits in the SAME factor order. perm[n] = memory row of bin n. */
static inline void vfft_natorder_mk_perm(int N, const int *f, int nf, int *perm)
{
    for (int n = 0; n < N; n++) {
        int t = n, slot = 0, rem = N;
        for (int s = 0; s < nf; s++) {
            int d = t % f[s];
            t /= f[s];
            rem /= f[s];
            slot += d * rem;
        }
        perm[n] = slot;
    }
}

static inline void vfft_natorder_inv_perm(int N, const int *p, int *ip)
{
    for (int n = 0; n < N; n++) ip[p[n]] = n;
}

/* Orientation auto-detect. Caller ran the plan's forward ONCE on an impulse at row n0
 * (lane 0): re/im hold the scrambled spectrum, row stride K doubles. Returns the winning
 * map M (malloc'd, natural[k] = scrambled[M[k]]) or NULL if nothing matches (caller must
 * then refuse order=NATURAL for this plan — fail safe, never silently wrong). */
static inline int *vfft_natorder_detect(int N, const int *f, int nf, size_t K,
                                        const double *re, const double *im, int n0)
{
    int *pf = (int *)malloc((size_t)N * 4), *ipf = (int *)malloc((size_t)N * 4);
    int *pr = (int *)malloc((size_t)N * 4), *ipr = (int *)malloc((size_t)N * 4);
    int frev[32];
    for (int i = 0; i < nf && i < 32; i++) frev[i] = f[nf - 1 - i];
    vfft_natorder_mk_perm(N, f, nf, pf);    vfft_natorder_inv_perm(N, pf, ipf);
    vfft_natorder_mk_perm(N, frev, nf, pr); vfft_natorder_inv_perm(N, pr, ipr);
    int *cand[4] = { pf, ipf, pr, ipr };
    int best = -1;
    /* probe a spread of k values; closed form X[k] = e^{-2πi k n0 / N} */
    for (int c = 0; c < 4 && best < 0; c++) {
        int ok = 1;
        for (int t = 0; t < 12 && ok; t++) {
            int k = (int)(((long long)(t * 2654435761u)) % (unsigned)N);
            double a  = -2.0 * VFFT_NATORDER_PI * (double)k * (double)n0 / (double)N;
            double er = re[(size_t)cand[c][k] * K] - cos(a);
            double ei = im[(size_t)cand[c][k] * K] - sin(a);
            if (er * er + ei * ei > 1e-12) ok = 0;
        }
        if (ok) best = c;
    }
    int *out = NULL;
    if (best >= 0) { out = cand[best]; cand[best] = NULL; }
    for (int c = 0; c < 4; c++) free(cand[c]);
    return out;
}

/* Flattened cycle list for the cycle-UB executor: each nontrivial cycle's row indices in
 * move order, -1 terminated; -2 ends the list. Fixed points are omitted (nf==1 chains
 * produce an empty list — FREE detection can also key off that). malloc'd; size 2N+8 ints. */
static inline int *vfft_natorder_mk_cycles(int N, const int *M)
{
    int *lst = (int *)malloc((size_t)(2 * N + 8) * 4), *w = lst;
    char *vis = (char *)calloc((size_t)N, 1);
    for (int st = 0; st < N; st++) {
        if (vis[st] || M[st] == st) { vis[st] = 1; continue; }
        int cur = st;
        while (!vis[cur]) { vis[cur] = 1; *w++ = cur; cur = M[cur]; }
        *w++ = -1;
    }
    *w = -2;
    free(vis);
    return lst;
}

/* Scan a flattened cycle list -> malloc'd offset array (ncyc+1 ints; off[c]=list index of
 * cycle c's first row, off[ncyc]=index of the -2 sentinel) + count. For the MT split. */
static inline int *vfft_natorder_cycle_offsets(const int *list, int *ncyc_out)
{
    int cap = 16, ncyc = 0;
    int *off = (int *)malloc((size_t)cap * 4);
    int i = 0;
    while (list[i] != -2) {
        if (ncyc + 2 > cap) { cap *= 2; off = (int *)realloc(off, (size_t)cap * 4); }
        off[ncyc++] = i;
        while (list[i] != -1) i++;   /* to this cycle's terminator */
        i++;                          /* past the -1 */
    }
    off[ncyc] = i;                    /* sentinel index */
    *ncyc_out = ncyc;
    return off;
}

/* Number of pairs in a flat pair list [a0,b0,...,-2]. */
static inline int vfft_natorder_pair_count(const int *pairs)
{
    int n = 0;
    while (pairs[2 * n] != -2) n++;
    return n;
}

/* Pair list for involutions (PSWAP): flat [a0,b0,a1,b1,...], -2 sentinel. NULL if M is
 * not an involution (caller then rejects the PSWAP verdict — honorable fallback). */
static inline int *vfft_natorder_mk_pairs(int N, const int *M)
{
    for (int n = 0; n < N; n++)
        if (M[M[n]] != n) return NULL;
    int *lst = (int *)malloc((size_t)(N + 2) * 4), *w = lst;
    for (int n = 0; n < N; n++)
        if (M[n] > n) { *w++ = n; *w++ = M[n]; }
    *w = -2;
    return lst;
}

#endif /* VFFT_NATORDER_PERM_H */
