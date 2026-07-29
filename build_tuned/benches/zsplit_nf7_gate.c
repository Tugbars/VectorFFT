/* zsplit_nf7_gate.c — P2: the cascade chain cap nf 6 -> 7.
 *
 * WHY THIS CELL. The cascade chain space is {4,8}^nf with prod == N. At
 * N=16384 = 2^14, a chain of `a` fours and `b` eights needs 2a+3b = 14 with
 * a+b = nf, so:
 *     nf = 6  ->  a=4, b=2   (4.4.4.4.8.8 and its orderings)  — REACHABLE BEFORE
 *     nf = 7  ->  a=7, b=0   (4.4.4.4.4.4.4)                  — NEW, cap-blocked
 * i.e. at cap 6 the planner returned the most radix-4-heavy chain REACHABLE,
 * not the most radix-4-heavy chain. This gate proves the nf=7 chain now
 * builds AND is numerically right, and that an nf=6 control is unchanged.
 *
 * 🔴 ROUNDTRIP ALONE CANNOT GATE THIS. The cascade emits a SCRAMBLED comb and
 * its own bwd consumes that same comb, so bwd(fwd(x)) = N*x holds under ANY
 * self-consistent permutation — including a wrong one. (Same trap the prime
 * routes hit today: unimodular chirps gave exact roundtrips while both
 * directions computed the wrong transform.) So the forward is checked
 * PERMUTATION-AWARE against a scalar DFT at sampled bins, using the engines'
 * own documented output maps:
 *   legacy zsplit (zsplit.h:9-11):  out[l*NR + g]        = X[drev(g*Rt + l)]
 *   ZTURN-S       (dp_planner_il):  out[l*NR + S*k' + j] = legacy[l*NR + j*(NR/S) + k']
 * with NR = N/Rt, Rt = chain[nf-1], S = chain[0], drev = _vfft_zs_brev.
 *
 * Build: python build.py --src benches/zsplit_nf7_gate.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "executor.h"
#include "planner.h"
#include "oop_plan.h"
#include "zsplit.h"
#include "zturn.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* one naive DFT bin, X[k] = sum_n z[n] e^{-2pi i nk/N} */
static void naive_bin(const double *z, int N, long k, double *re, double *im)
{
    double sr = 0.0, si = 0.0;
    for (int n = 0; n < N; n++) {
        double a = -2.0 * M_PI * (double)((long)n * k % N) / (double)N;
        double c = cos(a), s = sin(a);
        sr += z[2 * n] * c - z[2 * n + 1] * s;
        si += z[2 * n] * s + z[2 * n + 1] * c;
    }
    *re = sr; *im = si;
}

/* spectral bin held by output slot i, per the engine's own map */
static long bin_of_slot(long i, int N, const int *chain, int nf, int zturn)
{
    long Rt = chain[nf - 1], NR = (long)N / Rt;
    long l = i / NR, r = i % NR;
    if (zturn) {                       /* undo the per-row Gamma transpose */
        long S = chain[0], kq = r / S, j = r % S;
        r = j * (NR / S) + kq;
    }
    return _vfft_zs_brev(r * Rt + l, nf, chain);
}

/* expect_refuse: the chain is KNOWN-illegal for this engine and create must
 * return NULL (a refusal is the correct answer, not a failure). */
static int cell_x(const char *name, int N, const int *chain, int nf, int zturn,
                  int expect_refuse)
{
    void *p = zturn ? (void *)vfft_zturn2_create_chain(N, chain, nf)
                    : (void *)vfft_zsplit_create(N, chain, nf);
    if (expect_refuse) {
        printf("  %-34s refused=%s  %s\n", name, p ? "no" : "yes",
               p ? "*** FAIL ***" : "ok (by design)");
        if (p) {
            if (zturn) vfft_zturn2_destroy((vfft_zturn2_plan_t *)p);
            else       vfft_zsplit_destroy((vfft_zsplit_plan_t *)p);
            return 1;
        }
        return 0;
    }
    if (!p) {
        printf("  %-34s create=NULL  *** FAIL ***\n", name);
        return 1;
    }
    size_t nd = (size_t)2 * N;
    double *z = malloc(nd * 8), *y = malloc(nd * 8), *r = malloc(nd * 8);
    srand(1234 + N + nf);
    for (size_t i = 0; i < nd; i++) z[i] = (double)rand() / RAND_MAX - 0.5;

    if (zturn) vfft_zturn2_execute_fwd((vfft_zturn2_plan_t *)p, z, y);
    else       vfft_zsplit_execute_fwd((vfft_zsplit_plan_t *)p, z, y);

    /* PERMUTATION-AWARE forward check at sampled slots */
    double wf = 0.0, scale = 0.0;
    for (int t = 0; t < 48; t++) {
        long i = (long)((double)rand() / ((double)RAND_MAX + 1.0) * N);
        long k = bin_of_slot(i, N, chain, nf, zturn);
        double br, bi;
        naive_bin(z, N, k, &br, &bi);
        double d = fabs(y[2 * i] - br) + fabs(y[2 * i + 1] - bi);
        double sc = fabs(br) + fabs(bi);
        if (d > wf) wf = d;
        if (sc > scale) scale = sc;
    }
    double relf = wf / (scale > 0.0 ? scale : 1.0);

    /* roundtrip: the route's own bwd consumes its own comb -> N*x */
    if (zturn) vfft_zturn2_execute_bwd((vfft_zturn2_plan_t *)p, y, r);
    else       vfft_zsplit_execute_bwd((vfft_zsplit_plan_t *)p, y, r);
    double wr = 0.0, sr2 = 0.0;
    for (size_t i = 0; i < nd; i++) {
        double want = (double)N * z[i];
        double d = fabs(r[i] - want);
        if (d > wr) wr = d;
        if (fabs(want) > sr2) sr2 = fabs(want);
    }
    double relr = wr / (sr2 > 0.0 ? sr2 : 1.0);

    int bad = !(relf < 1e-11) || !(relr < 1e-11);
    printf("  %-34s fwd(perm-aware)=%-9.2e rt=%-9.2e  %s\n",
           name, relf, relr, bad ? "*** FAIL ***" : "ok");
    if (zturn) vfft_zturn2_destroy((vfft_zturn2_plan_t *)p);
    else       vfft_zsplit_destroy((vfft_zsplit_plan_t *)p);
    free(z); free(y); free(r);
    return bad;
}
static int cell(const char *name, int N, const int *chain, int nf, int zturn)
{
    return cell_x(name, N, chain, nf, zturn, 0);
}

int main(void)
{
    int bad = 0;
    printf("# cascade chain-cap gate: VFFT_ZSPLIT_MAX_NF = %d\n", VFFT_ZSPLIT_MAX_NF);
    printf("# VFFT_K1_CC_MAX_NF = %d (codec digit cap)\n\n", VFFT_K1_CC_MAX_NF);

    if (VFFT_ZSPLIT_MAX_NF < 7) {
        printf("  cap is still %d — the nf=7 chain cannot be built  *** FAIL ***\n",
               VFFT_ZSPLIT_MAX_NF);
        return 1;
    }

    printf("-- NEW at nf=7 (cap-blocked before): 16384 = 4^7 --\n");
    { int c7[7] = { 4, 4, 4, 4, 4, 4, 4 };
      /* ZTURN-ONLY by construction: legacy zsplit refuses any chain whose
       * last factor is not 8 (its sterm terminator kernel is radix-8 only,
       * zsplit.h create). ZTURN carries the radix-4 terminator, and ZTURN is
       * the runtime route since the 2026-07-27 cutover, so this chain is
       * fully usable where it matters. */
      bad |= cell_x("16384 4.4.4.4.4.4.4 legacy", 16384, c7, 7, 0, /*refuse=*/1);
      bad |= cell("16384 4.4.4.4.4.4.4 zturn ", 16384, c7, 7, 1); }

    printf("\n-- CONTROLS at nf<=6 (reachable before; must be unchanged) --\n");
    { int c6[6] = { 4, 4, 4, 4, 8, 8 };
      bad |= cell("16384 4.4.4.4.8.8 legacy", 16384, c6, 6, 0);
      bad |= cell("16384 4.4.4.4.8.8 zturn ", 16384, c6, 6, 1); }
    { int c5[5] = { 4, 4, 8, 8, 8 };
      bad |= cell("8192  4.4.8.8.8 legacy", 8192, c5, 5, 0);
      bad |= cell("8192  4.4.8.8.8 zturn ", 8192, c5, 5, 1); }
    { int c4[4] = { 4, 8, 8, 8 };   /* 4*8^3 = 2048 */
      bad |= cell("2048  4.8.8.8 legacy", 2048, c4, 4, 0);
      bad |= cell("2048  4.8.8.8 zturn ", 2048, c4, 4, 1); }

    printf("\n-- cc_chain codec round-trip at the new width --\n");
    { int c7[7] = { 4, 4, 4, 4, 4, 4, 4 }, back[VFFT_K1_CC_MAX_NF];
      int code = vfft_k1_cc_chain_encode(c7, 7);
      int nb = vfft_k1_cc_chain_decode(code, back);
      int ok = (code == 2222222) && (nb == 7);
      for (int s = 0; s < 7 && ok; s++) ok = (back[s] == 4);
      printf("  encode(4^7)=%d decode_nf=%d  %s\n", code, nb,
             ok ? "ok" : "*** FAIL ***");
      if (!ok) bad = 1; }
    { /* an 8-digit code must still be REJECTED, not silently truncated */
      int back[VFFT_K1_CC_MAX_NF];
      int nb = vfft_k1_cc_chain_decode(22222222, back);
      printf("  decode(8 digits) rejected=%s  %s\n", nb ? "no" : "yes",
             nb ? "*** FAIL ***" : "ok");
      if (nb) bad = 1; }

    printf("\n%s\n", bad ? "*** ZSPLIT NF7 GATE FAILED ***" : "ZSPLIT NF7 GATE PASSED");
    return bad;
}
