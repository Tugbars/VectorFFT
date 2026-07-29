/* il_odd_chain_gate.c — composition gate for the pure-IL 3-STAGE chain
 * (docs/roadmap/il_odd_chain.md): N = R2 · A · B, R1 = A·B, odd factors as
 * kernel RADICES only — every vectorized count stays even, no tail arm.
 *
 *   stage 1   n1t(R2), 1 call            (count = R1, even)
 *   stage 2a  t2(B),  A calls, c=0..A-1  (count = R2, even; tw modulus B·R2)
 *   stage 2b  t2(A),  B calls, b=0..B-1  (count = R2, even; tw modulus N)
 *
 * Oracle = naive scalar DFT. This gates the CALL SEQUENCE + tables + strides;
 * the kernels themselves are individually gated (odd_il_codelets_emitter).
 * Run BEFORE any il2p.h/vfft.c wiring — the il2p-backward discipline.
 *
 * Build: python build.py --src benches/il_odd_chain_gate.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef void (*zk_fn)(const double *, const double *, double *, double *,
                      const double *, const double *,
                      size_t, size_t, size_t, size_t, size_t);

#define DECL(sym) extern void sym(const double *, const double *, double *, \
    double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
DECL(radix4_z_n1t_fwd_avx2)  DECL(radix8_z_n1t_fwd_avx2)
DECL(radix16_z_n1t_fwd_avx2) DECL(radix32_z_n1t_fwd_avx2)
DECL(radix3_z_t2_fwd_avx2)   DECL(radix4_z_t2_fwd_avx2)
DECL(radix5_z_t2_fwd_avx2)   DECL(radix8_z_t2_fwd_avx2)
DECL(radix16_z_t2_fwd_avx2)  DECL(radix27_z_t2_fwd_avx2)
#undef DECL

static zk_fn n1t_fn(int R)
{
    switch (R) {
    case 4:  return radix4_z_n1t_fwd_avx2;
    case 8:  return radix8_z_n1t_fwd_avx2;
    case 16: return radix16_z_n1t_fwd_avx2;
    case 32: return radix32_z_n1t_fwd_avx2;
    default: return 0;
    }
}
static zk_fn t2_fn(int R)
{
    switch (R) {
    case 3:  return radix3_z_t2_fwd_avx2;
    case 4:  return radix4_z_t2_fwd_avx2;
    case 5:  return radix5_z_t2_fwd_avx2;
    case 8:  return radix8_z_t2_fwd_avx2;
    case 16: return radix16_z_t2_fwd_avx2;
    case 27: return radix27_z_t2_fwd_avx2;
    default: return 0;
    }
}
/* 🔴 BACKWARD: NOT YET BUILT — and deliberately NOT the conj-of-forward
 * (t2p) composition, which was drafted here and then RETIRED with the t2p
 * kind tree-wide (Tugbars 2026-07-29: t2t is the one canonical bwd
 * semantics). The chain bwd waits for the t2t-with-leg-stride store
 * variant (turned store with legs at stride A — wire the currently-unused
 * OGs slot in the emitter), then gets its own derivation + cells here. */

/* VTW2 table, il2p.h's builder generalized to (legs, cols, modulus):
 * record (pair pp, leg l) at (pp*(legs-1)+(l-1))*8 =
 * [c,c,c,c][-s,+s,-s,+s], angle -2*pi*l*k/modulus, k = 2pp+j.
 * conj=1 flips the sin lanes (table-side conjugation, il2p.h twb style). */
static double *build_vtw2(int legs, int cols, int modulus, int conj)
{
    size_t nrec = ((size_t)cols / 2u) * (size_t)(legs - 1);
    double *tw = (double *)malloc(nrec * 8u * sizeof(double));
    if (!tw) exit(2);
    for (int pp = 0; pp < cols / 2; pp++)
        for (int l = 1; l < legs; l++) {
            double *rf = tw + ((size_t)pp * (legs - 1) + (l - 1)) * 8u;
            for (int j = 0; j < 2; j++) {
                double k = (double)(2 * pp + j);
                double a = -2.0 * M_PI * (double)l * k / (double)modulus;
                double s = conj ? sin(a) : -sin(a);
                rf[2 * j] = cos(a); rf[2 * j + 1] = cos(a);
                rf[4 + 2 * j] = s; rf[4 + 2 * j + 1] = -s;
            }
        }
    return tw;
}

/* dir = -1 forward DFT, +1 unnormalized inverse */
static void naive_dft(const double *z, double *X, int N, int dir)
{
    for (int k = 0; k < N; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = (double)dir * 2.0 * M_PI * (double)n * (double)k / (double)N;
            double c = cos(a), s = sin(a);
            sr += z[2 * n] * c - z[2 * n + 1] * s;
            si += z[2 * n] * s + z[2 * n + 1] * c;
        }
        X[2 * k] = sr; X[2 * k + 1] = si;
    }
}

static int run_cell(int N, int R2, int A, int B)
{
    const int R1 = A * B;
    zk_fn leaf = n1t_fn(R2), kB = t2_fn(B), kA = t2_fn(A);
    if (R1 * R2 != N || (R1 & 1) || (R2 & 1) || !leaf || !kB || !kA) {
        printf("  N=%-5d %2dx(%2d.%2d)  SKIP (no kernel / odd count)\n", N, R2, A, B);
        return 0;
    }
    double *z    = (double *)malloc((size_t)2 * N * sizeof(double));
    double *mid1 = (double *)malloc((size_t)2 * N * sizeof(double));
    double *mid2 = (double *)malloc((size_t)2 * N * sizeof(double));
    double *out  = (double *)malloc((size_t)2 * N * sizeof(double));
    double *ref  = (double *)malloc((size_t)2 * N * sizeof(double));
    double *twB  = build_vtw2(B, R2, B * R2, 0); /* stage 2a: W_{N/A}^{jq}     */
    double *twA  = build_vtw2(A, B * R2, N, 0);  /* stage 2b: W_N^{c(q+b*R2)} —
                                                  * one table over ALL B*R2 cols;
                                                  * call b reads region b*R2.. */
    srand(1234 + N);
    for (int i = 0; i < 2 * N; i++) z[i] = (double)rand() / RAND_MAX - 0.5;
    naive_dft(z, ref, N, -1);

    leaf(z, 0, mid1, 0, 0, 0, (size_t)R1, 0, (size_t)R2, 0, (size_t)R1);
    for (int c = 0; c < A; c++)
        kB(mid1 + 2 * (size_t)c * R2, 0, mid2 + 2 * (size_t)c * R2, 0, twB, 0,
           (size_t)A * R2, 0, (size_t)A * R2, 0, (size_t)R2);
    for (int b = 0; b < B; b++)
        kA(mid2 + 2 * (size_t)b * A * R2, 0, out + 2 * (size_t)b * R2, 0,
           twA + ((size_t)b * R2 / 2u) * (size_t)(A - 1) * 8u, 0,
           (size_t)R2, 0, (size_t)B * R2, 0, (size_t)R2);

    double worst = 0, scale = 0;
    for (int i = 0; i < 2 * N; i++) {
        double d = fabs(out[i] - ref[i]);
        if (d > worst) worst = d;
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    double rel = worst / (scale > 0 ? scale : 1);
    int ok = rel < 1e-11;
    printf("  N=%-5d %2dx(%2d.%2d)  rel=%.2e  %s\n", N, R2, A, B, rel,
           ok ? "ok" : "FAIL");
    free(z); free(mid1); free(mid2); free(out); free(ref); free(twB); free(twA);
    return ok ? 0 : 1;
}


int main(void)
{
    int fails = 0;
    printf("-- IL 3-stage chain vs naive DFT (fwd) --\n");
    fails += run_cell(48,   4, 3, 4);   /* odd in 2b */
    fails += run_cell(48,   4, 4, 3);   /* odd in 2a */
    fails += run_cell(96,   8, 3, 4);
    fails += run_cell(96,   4, 3, 8);
    fails += run_cell(192, 16, 3, 4);
    fails += run_cell(192, 16, 4, 3);
    fails += run_cell(320, 16, 4, 5);
    fails += run_cell(320, 16, 5, 4);
    fails += run_cell(768, 16, 3, 16);
    fails += run_cell(1536, 32, 3, 16);
    fails += run_cell(1728, 8, 27, 8);  /* 27·64: big odd radix */
    fails += run_cell(256,  4, 8, 8);   /* pow2 control */
    printf(fails ? "IL ODD CHAIN GATE: %d FAIL\n" : "IL ODD CHAIN GATE PASSED\n",
           fails);
    return fails != 0;
}
