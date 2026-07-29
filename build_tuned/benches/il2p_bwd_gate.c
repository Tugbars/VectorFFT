/* il2p_bwd_gate.c — hardware gate for the F-DIAG backward composition
 * (src/core/oop/il2p.h vfft_il2p_execute_bwd).
 *
 * The composition was derived by two BLIND derivations and validated in a
 * scalar simulator; this gate proves it on the REAL emitted kernels.
 *
 * TWO CHECKS per cell, because either alone can be fooled:
 *   [rt]  roundtrip  bwd(fwd(x)) == N*x    -- catches nothing if fwd and bwd
 *                                             share a compensating permutation
 *   [dir] direct     bwd(y) vs a naive UNNORMALIZED inverse DFT of y
 *                                          -- this is the one that bites
 *
 * 🔴 NON-SQUARE PAIRS ARE MANDATORY. The two mirror decompositions coincide
 * when R1 == R2, so 256 (16x16) / 1024 (32x32) / 4096 (64x64) CANNOT
 * adjudicate. 128 (8x16), 512 (16x32) and both orders of 8x16/16x8 are the
 * cells that actually discriminate; the square ones are controls.
 */
#define _CRT_SECURE_NO_WARNINGS 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include "il2p.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return ((double)(*s >> 8) / (double)(1u << 24)) - 0.5;
}

/* unnormalized inverse DFT: X[m] = sum_n x[n] e^{+2pi i n m / N} */
static void naive_idft(int N, const double *z, double *out)
{
    for (int m = 0; m < N; m++) {
        double sr = 0.0, si = 0.0;
        for (int n = 0; n < N; n++) {
            double th = 2.0 * M_PI * (double)n * (double)m / (double)N;
            double c = cos(th), s = sin(th);
            sr += z[2 * n] * c - z[2 * n + 1] * s;
            si += z[2 * n] * s + z[2 * n + 1] * c;
        }
        out[2 * m] = sr;
        out[2 * m + 1] = si;
    }
}

static int run(int N, int R1, int R2)
{
    const char *shape = (R1 == R2) ? "square (control)" : "NON-SQUARE";
    vfft_il2p_plan_t *p = vfft_il2p_create(N, R1, R2);
    if (!p) {
        printf("  N=%-6d %2dx%-3d  %-16s  create returned NULL (no kernels)\n",
               N, R1, R2, shape);
        return 0; /* not a failure: pair simply unavailable in this build */
    }

    size_t nd = (size_t)2 * N;
    double *x = malloc(nd * sizeof(double));
    double *y = malloc(nd * sizeof(double));
    double *r = malloc(nd * sizeof(double));
    double *ref = malloc(nd * sizeof(double));
    unsigned seed = 12345u + (unsigned)N + 7u * (unsigned)R1;
    for (size_t i = 0; i < nd; i++) x[i] = urand(&seed);

    /* [dir] bwd applied straight to x, against a naive unnormalized inverse */
    int rc_dir = vfft_il2p_execute_bwd(p, x, y);
    double dir = -1.0;
    if (rc_dir == 0) {
        naive_idft(N, x, ref);
        double worst = 0.0, scale = 0.0;
        for (int m = 0; m < N; m++) {
            double dr = fabs(y[2 * m] - ref[2 * m]);
            double di = fabs(y[2 * m + 1] - ref[2 * m + 1]);
            double sc = fabs(ref[2 * m]) + fabs(ref[2 * m + 1]);
            if (dr + di > worst) worst = dr + di;
            if (sc > scale) scale = sc;
        }
        dir = worst / (scale > 0.0 ? scale : 1.0);
    }

    /* [rt] roundtrip bwd(fwd(x)) == N*x */
    vfft_il2p_execute_fwd(p, x, y);
    int rc_rt = vfft_il2p_execute_bwd(p, y, r);
    double rt = -1.0;
    if (rc_rt == 0) {
        double worst = 0.0, scale = 0.0;
        for (int i = 0; i < 2 * N; i++) {
            double want = (double)N * x[i];
            double d = fabs(r[i] - want);
            if (d > worst) worst = d;
            if (fabs(want) > scale) scale = fabs(want);
        }
        rt = worst / (scale > 0.0 ? scale : 1.0);
    }

    /* (The [t2p] route-A arm and its bit-exactness check vs F-DIAG were
     * removed 2026-07-29 with the t2p kind — t2t is canonical everywhere.) */

    /* [fdiag] the unfused reference of the retired route-A math — the
     * availability fallback. Judge vs the naive inverse. */
    double fus = -1.0;
    int rc_fus = vfft_il2p_execute_bwd_fdiag(p, x, y);
    if (rc_fus == 0) {
        naive_idft(N, x, ref);
        double worst = 0.0, scale = 0.0;
        for (int m = 0; m < N; m++) {
            double dr = fabs(y[2 * m] - ref[2 * m]);
            double di = fabs(y[2 * m + 1] - ref[2 * m + 1]);
            double sc = fabs(ref[2 * m]) + fabs(ref[2 * m + 1]);
            if (dr + di > worst) worst = dr + di;
            if (sc > scale) scale = sc;
        }
        fus = worst / (scale > 0.0 ? scale : 1.0);
    }

    /* [t2t] the canonical decomposition, gated directly (it is also what
     * vfft_il2p_execute_bwd runs, so [dir] above already exercised it —
     * this arm pins the entry point itself). */
    double bt = -1.0;
    int rc_bt = vfft_il2p_execute_bwd_t2t(p, x, y);
    if (rc_bt == 0) {
        naive_idft(N, x, ref);
        double worst = 0.0, scale = 0.0;
        for (int m = 0; m < N; m++) {
            double dr = fabs(y[2 * m] - ref[2 * m]);
            double di = fabs(y[2 * m + 1] - ref[2 * m + 1]);
            double sc = fabs(ref[2 * m]) + fabs(ref[2 * m + 1]);
            if (dr + di > worst) worst = dr + di;
            if (sc > scale) scale = sc;
        }
        bt = worst / (scale > 0.0 ? scale : 1.0);
    }

    int bad = (rc_dir != 0) || (rc_rt != 0) || !(dir < 1e-11) || !(rt < 1e-11)
              || (rc_fus == 0 && !(fus < 1e-11))
              || (rc_bt == 0 && !(bt < 1e-11));
    printf("  N=%-6d %2dx%-3d  %-16s  dir=%-9.2e fdiag=%-9.2e t2t=%-9.2e  %s\n",
           N, R1, R2, shape, dir, fus, bt, bad ? "*** FAIL ***" : "ok");

    free(x); free(y); free(r); free(ref);
    vfft_il2p_destroy(p);
    return bad;
}

/* (The A-vs-B race harness lived here until 2026-07-29. Its verdict — t2t
 * canonical, winner tracked R1 with t2t ahead everywhere but R1=64 — is
 * recorded in il2p.h's backward-path comment; the losing t2p arm was then
 * deleted tree-wide, so there is nothing left to race.) */

/* ── COVERAGE INVARIANT ──────────────────────────────────────────────────
 * For EVERY (R1,R2) the K=1 IL pair search can select, vfft_il2p_create must
 * succeed and BOTH directions must run. Any hole means execute silently falls
 * back to the convert fallback (the il_in/il_out hybrid route this gate was
 * written to make unreachable was deleted 2026-07-29).
 *
 * This replaces the question "can R2=4 ever be reached?", which is the wrong
 * question: the answer depends on `per` (ISA), the codelet registries and the
 * search bounds, so it is platform-specific and goes stale. Assert coverage
 * over the whole domain instead — then the hybrid is unreachable BY
 * CONSTRUCTION, on any platform, and this gate says so the moment it isn't.
 *
 * The domain mirrors vfft.c:3110-3124 exactly. */
static int coverage(void)
{
    static const int IL[] = { 4, 8, 16, 32, 64 };
    int holes = 0, pairs = 0;
    for (int a = 0; a < 5; a++)
        for (int b = 0; b < 5; b++) {
            int R1 = IL[a], R2 = IL[b], N = R1 * R2;
            pairs++;
            vfft_il2p_plan_t *p = vfft_il2p_create(N, R1, R2);
            if (!p) {
                printf("  HOLE: N=%-6d %2dx%-3d  create returned NULL"
                       "  -> execute would fall back to the HYBRID\n", N, R1, R2);
                holes++;
                continue;
            }
            double *x = malloc((size_t)2 * N * sizeof(double));
            double *y = malloc((size_t)2 * N * sizeof(double));
            unsigned s = 7u + (unsigned)N;
            for (int i = 0; i < 2 * N; i++) x[i] = urand(&s);
            vfft_il2p_execute_fwd(p, x, y);
            if (vfft_il2p_execute_bwd(p, x, y) != 0) {
                printf("  HOLE: N=%-6d %2dx%-3d  bwd unavailable\n", N, R1, R2);
                holes++;
            }
            free(x); free(y);
            vfft_il2p_destroy(p);
        }
    printf("  %d/%d pairs covered%s\n", pairs - holes, pairs,
           holes ? "" : "  -> hybrid fallback is UNREACHABLE by construction");
    return holes;
}

int main(void)
{
    int bad = 0;
    printf("# il2p F-DIAG backward gate (real emitted kernels)\n");
    printf("# dir = bwd vs naive unnormalized IDFT; rt = bwd(fwd(x)) == N*x\n\n");

    printf("-- COVERAGE: every pair the IL search can select must build --\n");
    bad |= coverage();
    printf("\n");

    printf("-- NON-SQUARE: these are the cells that discriminate --\n");
    bad |= run(128, 8, 16);
    bad |= run(128, 16, 8);
    bad |= run(512, 16, 32);
    bad |= run(512, 32, 16);
    bad |= run(2048, 32, 64);
    bad |= run(2048, 64, 32);
    bad |= run(1024, 16, 64);
    bad |= run(1024, 64, 16);

    printf("\n-- SQUARE controls (A and B coincide here; cannot adjudicate) --\n");
    bad |= run(64, 8, 8);
    bad |= run(256, 16, 16);
    bad |= run(1024, 32, 32);
    bad |= run(4096, 64, 64);

    printf("\n%s\n", bad ? "*** IL2P BWD GATE FAILED ***" : "IL2P BWD GATE PASSED");
    return bad;
}
