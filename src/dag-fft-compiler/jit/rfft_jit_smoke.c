/* rfft_jit_smoke.c — validate the rfft PACKED-forward JIT vs the generic executor.
 *
 *   plan = rfft_plan_create_ex(..., FLAT variants)   (generic reference)
 *   fn   = vfft_rfft_jit_resolve(...)                (emit+compile+load+cache)
 *   assert fn(plan) output == rfft_execute_fwd_packed(plan) output  (bit-exact)
 *   time both at low K (where the per-stage dispatch JIT removes actually shows).
 *
 * Build: cd build_tuned && python build.py --src ../src/dag-fft-compiler/jit/rfft_jit_smoke.c --jit --compile
 * Run  : PATH += mingw bin ; (python + gcc on PATH for the runtime compile)
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rfft_registry_avx2.h"     /* rfft.h + rfft_register_all_avx2 */
#include "rfft_jit_runtime.h"       /* after rfft.h: vfft_rfft_jit_resolve */

#if defined(_WIN32)
#include <intrin.h>
static inline unsigned long long rdtsc(void){ return __rdtsc(); }
#else
#include <x86intrin.h>
static inline unsigned long long rdtsc(void){ return __rdtsc(); }
#endif

/* variants: per-stage codelet (0=flat, 1=log3). The generic REFERENCE plan is
 * built with the SAME variants the JIT emits, so max|gen-jit| isolates the
 * executor specialization (not a codelet difference). Both flat and log3 are
 * exercised below — flat is not an emitter limitation, just the default gate. */
static int test_cell(rfft_codelets_t *reg, int N, size_t K, const int *f, int nf,
                     const int *variants, const char *vtag) {
    rfft_plan_t *p = rfft_plan_create_ex(N, K, f, nf, variants, reg);
    if (!p) { printf("  N=%-5d K=%-4zu  plan NULL (codelet gap?)\n", N, K); return 0; }

    rfft_jit_fn fn = vfft_rfft_jit_resolve(N, K, f, nf, variants, "avx2");
    if (!fn) { printf("  N=%-5d K=%-4zu  JIT resolve FAILED\n", N, K); rfft_plan_destroy(p); return 1; }

    size_t NK = (size_t)N * K;
    double *x  = (double *)malloc(NK * 8);
    double *gp = (double *)malloc(NK * 8);   /* generic packed out */
    double *jp = (double *)malloc(NK * 8);   /* jit packed out */
    srand(7 + N + (int)K);
    for (size_t i = 0; i < NK; i++) x[i] = (double)rand() / RAND_MAX * 2 - 1;

    rfft_execute_fwd_packed(p, x, gp);
    fn(p, x, jp);

    double md = 0;
    for (size_t i = 0; i < NK; i++) { double d = fabs(gp[i] - jp[i]); if (d > md) md = d; }

    /* timing — min of N reps */
    enum { REPS = 200 };
    unsigned long long bg = ~0ULL, bj = ~0ULL;
    for (int w = 0; w < 10; w++) { rfft_execute_fwd_packed(p, x, gp); fn(p, x, jp); }
    for (int r = 0; r < REPS; r++) {
        unsigned long long t0 = rdtsc(); rfft_execute_fwd_packed(p, x, gp); unsigned long long a = rdtsc() - t0;
        if (a < bg) bg = a;
        t0 = rdtsc(); fn(p, x, jp); unsigned long long b = rdtsc() - t0;
        if (b < bj) bj = b;
    }
    char fs[64]; size_t o = 0; fs[0] = '\0';
    for (int s = 0; s < nf; s++) o += (size_t)snprintf(fs + o, sizeof fs - o, "%s%d", s ? "x" : "", f[s]);
    printf("  N=%-5d K=%-4zu %-10s %-5s max|gen-jit|=%.1e %s | gen %8llu  jit %8llu cyc | jit/gen %.3f\n",
           N, K, fs, vtag, md, md < 1e-12 ? "BIT-EXACT" : "*** MISMATCH ***",
           bg, bj, bg > 0 ? (double)bj / bg : 0.0);

    free(x); free(gp); free(jp); rfft_plan_destroy(p);
    return md < 1e-12 ? 0 : 1;
}

/* NATURAL (split out_re/out_im) — the path the r2c dispatch executes. Reference is
 * rfft_execute_fwd_natural; the JIT terminator is log3-preferred to match p->hcn, so
 * we build the reference with whatever variants but compare the SPLIT output. */
static int test_cell_nat(rfft_codelets_t *reg, int N, size_t K, const int *f, int nf,
                        const int *variants) {
    rfft_plan_t *p = rfft_plan_create_ex(N, K, f, nf, variants, reg);
    if (!p) { printf("  N=%-5d K=%-4zu  [nat] plan NULL\n", N, K); return 0; }
    rfft_jit_nat_fn fn = vfft_rfft_jit_resolve_natural(N, K, f, nf, variants, "avx2");
    if (!fn) { printf("  N=%-5d K=%-4zu  [nat] JIT resolve FAILED\n", N, K); rfft_plan_destroy(p); return 1; }

    size_t NK = (size_t)N * K, RK = ((size_t)(N / 2) + 1) * K;
    double *x  = (double *)malloc(NK * 8);
    double *gr = (double *)calloc(RK, 8), *gi = (double *)calloc(RK, 8);
    double *jr = (double *)calloc(RK, 8), *ji = (double *)calloc(RK, 8);
    srand(7 + N + (int)K);
    for (size_t i = 0; i < NK; i++) x[i] = (double)rand() / RAND_MAX * 2 - 1;

    rfft_execute_fwd_natural(p, x, gr, gi);
    fn(p, x, jr, ji);
    double md = 0;
    for (size_t i = 0; i < RK; i++) {
        double a = fabs(gr[i] - jr[i]), b = fabs(gi[i] - ji[i]);
        if (a > md) md = a; if (b > md) md = b;
    }
    enum { REPS = 200 };
    unsigned long long bg = ~0ULL, bj = ~0ULL;
    for (int w = 0; w < 10; w++) { rfft_execute_fwd_natural(p, x, gr, gi); fn(p, x, jr, ji); }
    for (int r = 0; r < REPS; r++) {
        unsigned long long t0 = rdtsc(); rfft_execute_fwd_natural(p, x, gr, gi); unsigned long long a = rdtsc() - t0;
        if (a < bg) bg = a;
        t0 = rdtsc(); fn(p, x, jr, ji); unsigned long long b = rdtsc() - t0;
        if (b < bj) bj = b;
    }
    char fs[64]; size_t o = 0; fs[0] = '\0';
    for (int s = 0; s < nf; s++) o += (size_t)snprintf(fs + o, sizeof fs - o, "%s%d", s ? "x" : "", f[s]);
    printf("  N=%-5d K=%-4zu %-10s nat   max|gen-jit|=%.1e %s | gen %8llu  jit %8llu cyc | jit/gen %.3f\n",
           N, K, fs, md, md < 1e-12 ? "BIT-EXACT" : "*** MISMATCH ***",
           bg, bj, bg > 0 ? (double)bj / bg : 0.0);
    free(x); free(gr); free(gi); free(jr); free(ji); rfft_plan_destroy(p);
    return md < 1e-12 ? 0 : 1;
}

int main(void) {
    rfft_codelets_t reg; memset(&reg, 0, sizeof reg);
    rfft_register_all_avx2(&reg);

    printf("== rfft PACKED-forward JIT smoke (vs generic; FLAT + LOG3 variants; low K) ==\n");
    const int VF4[4] = {0,0,0,0}, VL4[4] = {1,1,1,1};
    const int VF3[3] = {0,0,0},   VL3[3] = {1,1,1};
    const int VF2[2] = {0,0},     VL2[2] = {1,1};
    int fails = 0;
    int f3[2] = {16, 16};       fails += test_cell(&reg, 256, 8,  f3, 2, VF2, "flat");
    int f4[4] = {4, 4, 4, 4};   fails += test_cell(&reg, 256, 8,  f4, 4, VF4, "flat");
    int f5[3] = {4, 8, 8};      fails += test_cell(&reg, 256, 8,  f5, 3, VF3, "flat");
    /* LOG3 path — proves the emitter is not flat-only (reference also built log3). */
    int g3[2] = {16, 16};       fails += test_cell(&reg, 256, 8,  g3, 2, VL2, "log3");
    int g4[4] = {4, 4, 4, 4};   fails += test_cell(&reg, 256, 8,  g4, 4, VL4, "log3");
    int g5[3] = {4, 8, 8};      fails += test_cell(&reg, 256, 8,  g5, 3, VL3, "log3");
    /* higher K too (the win shrinks as the codelet call amortizes) */
    int f6[4] = {4, 4, 4, 4};   fails += test_cell(&reg, 256, 64, f6, 4, VF4, "flat");
    /* NATURAL output (the r2c split path): JIT vs rfft_execute_fwd_natural. */
    printf("-- natural (split out_re/out_im) --\n");
    int n4[4] = {4, 4, 4, 4};   fails += test_cell_nat(&reg, 256, 8, n4, 4, VF4);
    int n5[3] = {4, 8, 8};      fails += test_cell_nat(&reg, 256, 8, n5, 3, VF3);
    int n6[4] = {4, 4, 4, 4};   fails += test_cell_nat(&reg, 256, 64, n6, 4, VF4);
    printf("# fails=%d\n", fails);
    return fails ? 1 : 0;
}
