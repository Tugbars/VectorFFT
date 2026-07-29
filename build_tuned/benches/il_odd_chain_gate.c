/* il_odd_chain_gate.c — composition gate for the pure-IL 3-STAGE chain
 * (docs/roadmap/il_odd_chain.md): N = R2 · A · B, R1 = A·B, odd factors as
 * kernel RADICES only — every vectorized count stays even, no tail arm.
 *
 *   stage 1   n1t(R2), 1 call            (count = R1, even)
 *   stage 2a  t2(B),  A calls, c=0..A-1  (count = R2, even; tw modulus B·R2)
 *   stage 2b  t2(A),  B calls, b=0..B-1  (count = R2, even; tw modulus N)
 *
 * Oracle = naive scalar DFT. Three tiers, in dependency order:
 *   1. raw kernel composition (fwd + bwd)      — the math;
 *   2. the il3p PLAN API (il2p.h)              — create/execute/roundtrip;
 *   3. the PUBLIC front door (vfft.h)          — route selection + dispatch.
 *
 * Build: python build.py --src benches/il_odd_chain_gate.c --vfft --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "il2p.h"
#include "il_prime.h"
#include "vfft.h"

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
DECL(radix4_z_n1_bwd_avx2)   DECL(radix8_z_n1_bwd_avx2)
DECL(radix16_z_n1_bwd_avx2)  DECL(radix32_z_n1_bwd_avx2)
DECL(radix3_z_t2_bwd_avx2)   DECL(radix4_z_t2_bwd_avx2)
DECL(radix5_z_t2_bwd_avx2)   DECL(radix8_z_t2_bwd_avx2)
DECL(radix16_z_t2_bwd_avx2)  DECL(radix27_z_t2_bwd_avx2)
DECL(radix3_z_t2tg_bwd_avx2) DECL(radix4_z_t2tg_bwd_avx2)
DECL(radix5_z_t2tg_bwd_avx2) DECL(radix8_z_t2tg_bwd_avx2)
DECL(radix16_z_t2tg_bwd_avx2) DECL(radix27_z_t2tg_bwd_avx2)
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
static zk_fn n1_bwd_fn(int R)
{
    switch (R) {
    case 4:  return radix4_z_n1_bwd_avx2;
    case 8:  return radix8_z_n1_bwd_avx2;
    case 16: return radix16_z_n1_bwd_avx2;
    case 32: return radix32_z_n1_bwd_avx2;
    default: return 0;
    }
}
static zk_fn t2_bwd_fn(int R)
{
    switch (R) {
    case 3:  return radix3_z_t2_bwd_avx2;
    case 4:  return radix4_z_t2_bwd_avx2;
    case 5:  return radix5_z_t2_bwd_avx2;
    case 8:  return radix8_z_t2_bwd_avx2;
    case 16: return radix16_z_t2_bwd_avx2;
    case 27: return radix27_z_t2_bwd_avx2;
    default: return 0;
    }
}
static zk_fn t2tg_bwd_fn(int R)
{
    switch (R) {
    case 3:  return radix3_z_t2tg_bwd_avx2;
    case 4:  return radix4_z_t2tg_bwd_avx2;
    case 5:  return radix5_z_t2tg_bwd_avx2;
    case 8:  return radix8_z_t2tg_bwd_avx2;
    case 16: return radix16_z_t2tg_bwd_avx2;
    case 27: return radix27_z_t2tg_bwd_avx2;
    default: return 0;
    }
}

static double *build_vtw2(int legs, int cols, int modulus, int conj);
static void naive_dft(const double *z, double *X, int N, int dir);

/* BACKWARD — t2t semantics (t2p is retired), stages of the forward inverted
 * in REVERSE order (docs/roadmap/il_odd_chain.md):
 *   B1 = t2_bwd(A),   B calls: IDFT_A across the a legs, POST-twiddle
 *        conj W_N^{c(q+b*R2)} (conj big table, region b), straight store.
 *   B2 = t2tg_bwd(B), A calls: IDFT_B across the b legs, POST-twiddle
 *        conj W_{B*R2}^{jq}, TURNED store with LEG STRIDE OGs=A so leg
 *        groups from different c calls interleave: (leg j, col q) ->
 *        mid1[q*R1 + j*A + c].
 *   B3 = n1_bwd(R2), 1 call: IDFT_R2 across q (now the strided axis),
 *        natural output. Oracle = naive unnormalized IDFT. */
static int run_cell_bwd(int N, int R2, int A, int B)
{
    const int R1 = A * B;
    zk_fn kA = t2_bwd_fn(A), kB = t2tg_bwd_fn(B), leaf = n1_bwd_fn(R2);
    if (R1 * R2 != N || (R1 & 1) || (R2 & 1) || !leaf || !kB || !kA) {
        printf("  N=%-5d %2dx(%2d.%2d)  SKIP (no kernel / odd count)\n", N, R2, A, B);
        return 0;
    }
    double *z    = (double *)malloc((size_t)2 * N * sizeof(double));
    double *mid2 = (double *)malloc((size_t)2 * N * sizeof(double));
    double *mid1 = (double *)malloc((size_t)2 * N * sizeof(double));
    double *out  = (double *)malloc((size_t)2 * N * sizeof(double));
    double *ref  = (double *)malloc((size_t)2 * N * sizeof(double));
    double *twAc = build_vtw2(A, B * R2, N, 1);
    double *twBc = build_vtw2(B, R2, B * R2, 1);
    srand(4321 + N);
    for (int i = 0; i < 2 * N; i++) z[i] = (double)rand() / RAND_MAX - 0.5;
    naive_dft(z, ref, N, +1);

    for (int b = 0; b < B; b++)
        kA(z + 2 * (size_t)b * R2, 0, mid2 + 2 * (size_t)b * A * R2, 0,
           twAc + ((size_t)b * R2 / 2u) * (size_t)(A - 1) * 8u, 0,
           (size_t)B * R2, 0, (size_t)R2, 0, (size_t)R2);
    for (int c = 0; c < A; c++)
        kB(mid2 + 2 * (size_t)c * R2, 0, mid1 + 2 * (size_t)c, 0, twBc, 0,
           (size_t)A * R2, 0, (size_t)R1, (size_t)A, (size_t)R2);
    leaf(mid1, 0, out, 0, 0, 0, (size_t)R1, 0, (size_t)R1, 0, (size_t)R1);

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
    free(z); free(mid2); free(mid1); free(out); free(ref);
    free(twAc); free(twBc);
    return ok ? 0 : 1;
}

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


/* Tier 2: the il3p PLAN API with its own default chain — fwd and bwd vs
 * naive, plus roundtrip bwd(fwd(x)) == N*x. */
static int run_plan_cell(int N)
{
    int R2, A, B;
    if (!vfft_il3p_default_chain(N, &R2, &A, &B)) {
        printf("  N=%-5d no default chain  SKIP\n", N);
        return 0;
    }
    vfft_il3p_plan_t *p = vfft_il3p_create(N, R2, A, B);
    if (!p) {
        printf("  N=%-5d %2dx(%2d.%2d)  create=NULL  *** FAIL ***\n", N, R2, A, B);
        return 1;
    }
    double *z   = (double *)malloc((size_t)2 * N * sizeof(double));
    double *y   = (double *)malloc((size_t)2 * N * sizeof(double));
    double *r   = (double *)malloc((size_t)2 * N * sizeof(double));
    double *ref = (double *)malloc((size_t)2 * N * sizeof(double));
    srand(777 + N);
    for (int i = 0; i < 2 * N; i++) z[i] = (double)rand() / RAND_MAX - 0.5;

    double ef = 0, eb = 0, ert = 0, scale;
    vfft_il3p_execute_fwd(p, z, y);
    naive_dft(z, ref, N, -1);
    scale = 0;
    for (int i = 0; i < 2 * N; i++) {
        double d = fabs(y[i] - ref[i]);
        if (d > ef) ef = d;
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    ef /= (scale > 0 ? scale : 1);

    vfft_il3p_execute_bwd(p, z, y);
    naive_dft(z, ref, N, +1);
    scale = 0;
    for (int i = 0; i < 2 * N; i++) {
        double d = fabs(y[i] - ref[i]);
        if (d > eb) eb = d;
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    eb /= (scale > 0 ? scale : 1);

    vfft_il3p_execute_fwd(p, z, y);
    vfft_il3p_execute_bwd(p, y, r);
    scale = 0;
    for (int i = 0; i < 2 * N; i++) {
        double want = (double)N * z[i];
        double d = fabs(r[i] - want);
        if (d > ert) ert = d;
        if (fabs(want) > scale) scale = fabs(want);
    }
    ert /= (scale > 0 ? scale : 1);

    int bad = !(ef < 1e-11) || !(eb < 1e-11) || !(ert < 1e-11);
    printf("  N=%-5d %2dx(%2d.%2d)  fwd=%-9.2e bwd=%-9.2e rt=%-9.2e  %s\n",
           N, R2, A, B, ef, eb, ert, bad ? "*** FAIL ***" : "ok");
    vfft_il3p_destroy(p);
    free(z); free(y); free(r); free(ref);
    return bad;
}

/* Tier 2b: PRIME N via il_prime.h (Rader when the N-1 inner is
 * IL-expressible, else Bluestein) — fwd/bwd vs naive + roundtrip. */
static int run_prime_cell(int N)
{
    vfft_ilprime_plan_t *p = vfft_ilprime_create(N);
    if (!p) {
        printf("  N=%-5d prime create=NULL  *** FAIL ***\n", N);
        return 1;
    }
    const char *meth = p->method ? "rader" : "blue ";
    double *z   = (double *)malloc((size_t)2 * N * sizeof(double));
    double *y   = (double *)malloc((size_t)2 * N * sizeof(double));
    double *r   = (double *)malloc((size_t)2 * N * sizeof(double));
    double *ref = (double *)malloc((size_t)2 * N * sizeof(double));
    srand(555 + N);
    for (int i = 0; i < 2 * N; i++) z[i] = (double)rand() / RAND_MAX - 0.5;

    double ef = 0, eb = 0, ert = 0, scale;
    vfft_ilprime_execute_fwd(p, z, y);
    naive_dft(z, ref, N, -1);
    scale = 0;
    for (int i = 0; i < 2 * N; i++) {
        double d = fabs(y[i] - ref[i]);
        if (d > ef) ef = d;
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    ef /= (scale > 0 ? scale : 1);

    vfft_ilprime_execute_bwd(p, z, y);
    naive_dft(z, ref, N, +1);
    scale = 0;
    for (int i = 0; i < 2 * N; i++) {
        double d = fabs(y[i] - ref[i]);
        if (d > eb) eb = d;
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    eb /= (scale > 0 ? scale : 1);

    vfft_ilprime_execute_fwd(p, z, y);
    vfft_ilprime_execute_bwd(p, y, r);
    scale = 0;
    for (int i = 0; i < 2 * N; i++) {
        double want = (double)N * z[i];
        double d = fabs(r[i] - want);
        if (d > ert) ert = d;
        if (fabs(want) > scale) scale = fabs(want);
    }
    ert /= (scale > 0 ? scale : 1);

    int bad = !(ef < 1e-10) || !(eb < 1e-10) || !(ert < 1e-10);
    printf("  N=%-5d %s M=%-5d  fwd=%-9.2e bwd=%-9.2e rt=%-9.2e  %s\n",
           N, meth, p->M, ef, eb, ert, bad ? "*** FAIL ***" : "ok");
    vfft_ilprime_destroy(p);
    free(z); free(y); free(r); free(ref);
    return bad;
}

/* Tier 3: the public front door — an INTERLEAVED K=1 plan at odd·2^k N must
 * route to the chain (create succeeds where no split K=1 route exists) and
 * both directions must dispatch correctly. */
static int run_public_cell(int N)
{
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE;
    c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = N; c.howmany = 1;
    c.layout = VFFT_LAYOUT_INTERLEAVED;
    vfft_plan h = vfft_create(&c);
    if (!h) {
        printf("  N=%-5d create=NULL  *** FAIL ***\n", N);
        return 1;
    }
    double *z   = (double *)malloc((size_t)2 * N * sizeof(double));
    double *y   = (double *)malloc((size_t)2 * N * sizeof(double));
    double *r   = (double *)malloc((size_t)2 * N * sizeof(double));
    double *ref = (double *)malloc((size_t)2 * N * sizeof(double));
    srand(999 + N);
    for (int i = 0; i < 2 * N; i++) z[i] = (double)rand() / RAND_MAX - 0.5;

    double ef = 0, ert = 0, scale;
    vfft_execute(h, VFFT_FORWARD, z, NULL, y, NULL);
    naive_dft(z, ref, N, -1);
    scale = 0;
    for (int i = 0; i < 2 * N; i++) {
        double d = fabs(y[i] - ref[i]);
        if (d > ef) ef = d;
        if (fabs(ref[i]) > scale) scale = fabs(ref[i]);
    }
    ef /= (scale > 0 ? scale : 1);

    vfft_execute(h, VFFT_BACKWARD, y, NULL, r, NULL);
    scale = 0;
    for (int i = 0; i < 2 * N; i++) {
        double want = (double)N * z[i];
        double d = fabs(r[i] - want);
        if (d > ert) ert = d;
        if (fabs(want) > scale) scale = fabs(want);
    }
    ert /= (scale > 0 ? scale : 1);

    int bad = !(ef < 1e-11) || !(ert < 1e-11);
    printf("  N=%-5d public fwd=%-9.2e rt=%-9.2e  %s\n",
           N, ef, ert, bad ? "*** FAIL ***" : "ok");
    vfft_destroy(h);
    free(z); free(y); free(r); free(ref);
    return bad;
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
    printf("-- IL 3-stage chain vs naive IDFT (bwd: t2_bwd + t2tg_bwd + n1_bwd) --\n");
    fails += run_cell_bwd(48,   4, 3, 4);
    fails += run_cell_bwd(48,   4, 4, 3);
    fails += run_cell_bwd(96,   8, 3, 4);
    fails += run_cell_bwd(96,   4, 3, 8);
    fails += run_cell_bwd(192, 16, 3, 4);
    fails += run_cell_bwd(192, 16, 4, 3);
    fails += run_cell_bwd(320, 16, 4, 5);
    fails += run_cell_bwd(320, 16, 5, 4);
    fails += run_cell_bwd(768, 16, 3, 16);
    fails += run_cell_bwd(1536, 32, 3, 16);
    fails += run_cell_bwd(1728, 8, 27, 8);
    fails += run_cell_bwd(1728, 8, 8, 27);
    fails += run_cell_bwd(256,  4, 8, 8);   /* pow2 control */
    printf("-- il3p PLAN API: default chain, fwd/bwd vs naive + roundtrip --\n");
    {
        static const int PN[] = { 48, 96, 192, 320, 384, 768, 1280, 1536, 1728,
                                  200, 300, 400, 600, 1200 };
        for (int i = 0; i < (int)(sizeof PN / sizeof *PN); i++)
            fails += run_plan_cell(PN[i]);
    }
    printf("-- il_prime PLAN API: Rader/Bluestein on IL inners --\n");
    {
        static const int PR[] = { 7, 11, 13, 17, 31, 41, 97, 101, 127, 193,
                                  241, 257, 509, 769, 1021, 2039 };
        for (int i = 0; i < (int)(sizeof PR / sizeof *PR); i++)
            fails += run_prime_cell(PR[i]);
    }
    printf("-- PUBLIC API (vfft.h, INTERLEAVED K=1): route + dispatch --\n");
    {
        static const int UN[] = { 48, 96, 192, 320, 768, 1536,
                                  31, 97, 127, 257, 509, 1021,
                                  36, 100, 144, 200, 300, 101,
                                  /* odd-count-tail cells: all-odd pairs and
                                   * 2·odd pairs, plus tail-upgraded primes */
                                  45, 63, 225, 675, 18, 50, 150,
                                  19, 29, 43 };
        for (int i = 0; i < (int)(sizeof UN / sizeof *UN); i++)
            fails += run_public_cell(UN[i]);
    }
    printf(fails ? "IL ODD CHAIN GATE: %d FAIL\n" : "IL ODD CHAIN GATE PASSED\n",
           fails);
    return fails != 0;
}
