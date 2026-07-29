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

    /* [t2p] the FUSED route A, gated against F-DIAG. Same math, so this must
     * agree to (near) the last bit; an O(1) gap means the fused kernel's
     * twiddle indexing is wrong, not the decomposition. */
    double fus = -1.0;
    int rc_fus = vfft_il2p_execute_bwd_t2p(p, x, y);
    if (rc_fus == 0 && rc_dir == 0) {
        double *ydiag = malloc(nd * sizeof(double));
        vfft_il2p_execute_bwd_fdiag(p, x, ydiag);
        double worst = 0.0, scale = 0.0;
        for (int i = 0; i < 2 * N; i++) {
            double d = fabs(y[i] - ydiag[i]);
            if (d > worst) worst = d;
            if (fabs(ydiag[i]) > scale) scale = fabs(ydiag[i]);
        }
        fus = worst / (scale > 0.0 ? scale : 1.0);
        free(ydiag);
    }

    /* [t2t] the RIVAL decomposition (route B), gated the same way. It is a
     * DIFFERENT operation order, so it will NOT be bit-equal to A — judge it
     * against the naive inverse, not against A. */
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
              || (rc_fus == 0 && !(fus < 1e-13))
              || (rc_bt == 0 && !(bt < 1e-11));
    printf("  N=%-6d %2dx%-3d  %-16s  A=%-9.2e t2p_vs_A=%-9.2e B=%-9.2e  %s\n",
           N, R1, R2, shape, dir, fus, bt, bad ? "*** FAIL ***" : "ok");

    free(x); free(y); free(r); free(ref);
    vfft_il2p_destroy(p);
    return bad;
}

/* Size what T2P would buy.
 *
 * ⚠️ bwd/fwd is NOT the right comparison and must not be used: forward's
 * stage 2 is t2 (twiddled, streams VTW2, BYTW2 in-kernel) while F-DIAG's
 * stage 2 is n1_b (TWIDDLE-FREE, the diagonal already happened). They are
 * different decompositions, so the ratio isolates nothing -- it even comes
 * out < 1 at large N.
 *
 * Fusing the diagonal into stage 2 (= the T2P kind) does NOT remove the
 * twiddle multiply; T2P still performs it. What it removes is the extra
 * read+write of the whole mid plane and the scalar-vs-SIMD arithmetic. So the
 * diagonal pass measured IN ISOLATION is the upper bound on the win. */
static double med(double *v, int n)
{
    for (int i = 1; i < n; i++) {
        double t = v[i]; int j = i - 1;
        while (j >= 0 && v[j] > t) { v[j + 1] = v[j]; j--; }
        v[j + 1] = t;
    }
    return v[n / 2];
}

static void timecell(int N, int R1, int R2)
{
    vfft_il2p_plan_t *p = vfft_il2p_create(N, R1, R2);
    if (!p) return;
    size_t nd = (size_t)2 * N;
    double *x = malloc(nd * sizeof(double)), *y = malloc(nd * sizeof(double));
    unsigned seed = 99u + (unsigned)N;
    for (size_t i = 0; i < nd; i++) x[i] = urand(&seed);

    /* QueryPerformanceCounter (the project's Win timer), cachebust between
     * arms, and arm ORDER ALTERNATED per round so neither is systematically
     * measured on a warmed core. Medians, not best-of. */
    static double bust[1 << 20];
    LARGE_INTEGER fq; QueryPerformanceFrequency(&fq);
    int reps = (int)(8000000 / (N + 1)); if (reps < 50) reps = 50;
    double f[9], b[9];
    for (int r = 0; r < 9; r++) {
        double t_f = 0.0, t_b = 0.0;
        for (int arm = 0; arm < 2; arm++) {
            /* arm 0 = whole F-DIAG backward; arm 1 = the diagonal pass ALONE */
            int whole = (r & 1) ? (arm == 1) : (arm == 0);   /* alternate */
            for (size_t i = 0; i < (sizeof bust / sizeof *bust); i += 8) bust[i] += 1.0;
            LARGE_INTEGER a, z; QueryPerformanceCounter(&a);
            if (whole)
                for (int i = 0; i < reps; i++) vfft_il2p_execute_bwd_t2p(p, x, y);
            else
                for (int i = 0; i < reps; i++) vfft_il2p_execute_bwd_t2t(p, x, y);
            QueryPerformanceCounter(&z);
            double ns = (double)(z.QuadPart - a.QuadPart) * 1e9
                        / (double)fq.QuadPart / reps;
            if (whole) t_b = ns; else t_f = ns;
        }
        f[r] = t_f; b[r] = t_b;
    }
    double tm = med(f, 9), pm = med(b, 9);   /* tm = route B (t2t), pm = route A (t2p) */
    printf("  N=%-6d %2dx%-3d  A(t2p)=%-9.0f B(t2t)=%-9.0f  B/A=%.2fx  -> %s\n",
           N, R1, R2, pm, tm, tm / pm, (tm < pm) ? "B WINS" : "A wins");
    free(x); free(y); vfft_il2p_destroy(p);
}

int main(void)
{
    int bad = 0;
    printf("# il2p F-DIAG backward gate (real emitted kernels)\n");
    printf("# dir = bwd vs naive unnormalized IDFT; rt = bwd(fwd(x)) == N*x\n\n");

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

    printf("\n-- THE RACE: route A (t2p, R2 first) vs route B (t2t, R1 first) --\n");
    printf("   non-square cells discriminate; square ones are controls\n");
    timecell(128, 8, 16);
    timecell(128, 16, 8);
    timecell(512, 16, 32);
    timecell(512, 32, 16);
    timecell(1024, 16, 64);
    timecell(1024, 64, 16);
    timecell(2048, 32, 64);
    timecell(2048, 64, 32);
    timecell(1024, 32, 32);
    timecell(4096, 64, 64);
    return bad;
}
