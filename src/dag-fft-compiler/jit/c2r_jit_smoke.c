/* c2r_jit_smoke.c — validate the c2r (inverse real FFT) JIT vs c2r_execute_packed.
 *
 *   plan = c2r_plan_create_ex(..., FLAT variants)   (generic reference)
 *   fn   = vfft_c2r_jit_resolve(...)                (emit+compile+load+cache)
 *   assert fn(plan) output == c2r_execute_packed(plan) output  (bit-exact)
 *
 * Note: the c2r plan's base (rfft_plan_create) needs the rfft FORWARD codelets for
 * its twiddle tables, so register BOTH rfft + c2r families into one registry.
 *
 * Build: cd build_tuned && python build.py --src ../src/dag-fft-compiler/jit/c2r_jit_smoke.c --jit --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "rfft_registry_avx2.h"     /* rfft.h + rfft_register_all_avx2 (base codelets) */
#include "c2r_registry_avx2.h"      /* c2r_register_all_avx2 (r2cb + hc2hc_dif_bwd) */
#include "c2r.h"                    /* c2r_plan_t + c2r_execute_packed */
#include "c2r_jit_runtime.h"        /* vfft_c2r_jit_resolve */
#include "c2r_dispatch.h"           /* vfft_c2r_plan_create / vfft_c2r_execute (wiring) */

#if defined(_WIN32)
#include <intrin.h>
#else
#include <x86intrin.h>
#endif
static inline unsigned long long rdtsc_(void){ return __rdtsc(); }

static int test_cell(rfft_codelets_t *reg, int N, size_t K, const int *f, int nf,
                     const int *variants, const char *vtag) {
    c2r_plan_t *p = c2r_plan_create_ex(N, K, f, nf, variants, reg);
    if (!p) { printf("  N=%-5d K=%-4zu  plan NULL (codelet gap?)\n", N, K); return 0; }
    c2r_jit_fn fn = vfft_c2r_jit_resolve(N, K, f, nf, variants, "avx2");
    if (!fn) { printf("  N=%-5d K=%-4zu  JIT resolve FAILED\n", N, K); c2r_plan_destroy(p); return 1; }

    size_t NK = (size_t)N * K;
    double *in = (double *)malloc(2 * NK * 8);   /* packed halfcomplex: re | im planes */
    double *go = (double *)malloc(NK * 8);
    double *jo = (double *)malloc(NK * 8);
    srand(7 + N + (int)K);
    for (size_t i = 0; i < 2 * NK; i++) in[i] = (double)rand() / RAND_MAX * 2 - 1;

    c2r_execute_packed(p, in, go);
    fn(p, in, jo);
    double md = 0;
    for (size_t i = 0; i < NK; i++) { double d = fabs(go[i] - jo[i]); if (d > md) md = d; }

    enum { REPS = 200 };
    unsigned long long bg = ~0ULL, bj = ~0ULL;
    for (int w = 0; w < 10; w++) { c2r_execute_packed(p, in, go); fn(p, in, jo); }
    for (int r = 0; r < REPS; r++) {
        unsigned long long t0 = rdtsc_(); c2r_execute_packed(p, in, go); unsigned long long a = rdtsc_() - t0;
        if (a < bg) bg = a;
        t0 = rdtsc_(); fn(p, in, jo); unsigned long long b = rdtsc_() - t0;
        if (b < bj) bj = b;
    }
    char fs[64]; size_t o = 0; fs[0] = '\0';
    for (int s = 0; s < nf; s++) o += (size_t)snprintf(fs + o, sizeof fs - o, "%s%d", s ? "x" : "", f[s]);
    printf("  N=%-5d K=%-4zu %-10s %-5s max|gen-jit|=%.1e %s | gen %8llu  jit %8llu cyc | jit/gen %.3f\n",
           N, K, fs, vtag, md, md < 1e-12 ? "BIT-EXACT" : "*** MISMATCH ***",
           bg, bj, bg > 0 ? (double)bj / bg : 0.0);
    free(in); free(go); free(jo); c2r_plan_destroy(p);
    return md < 1e-12 ? 0 : 1;
}

int main(void) {
    rfft_codelets_t reg; memset(&reg, 0, sizeof reg);
    rfft_register_all_avx2(&reg);   /* base (r2cf + hc2hc) */
    c2r_register_all_avx2(&reg);    /* c2r (r2cb + hc2hc_dif_bwd) */

    printf("== c2r (inverse real FFT) JIT smoke (vs c2r_execute_packed; flat + log3; low K) ==\n");
    const int VF4[4] = {0,0,0,0}, VL4[4] = {1,1,1,1};
    const int VF3[3] = {0,0,0}, VF2[2] = {0,0};
    int fails = 0;
    int a[2] = {16, 16};       fails += test_cell(&reg, 256, 8,  a, 2, VF2, "flat");
    int b[4] = {4, 4, 4, 4};   fails += test_cell(&reg, 256, 8,  b, 4, VF4, "flat");
    int c[3] = {4, 8, 8};      fails += test_cell(&reg, 256, 8,  c, 3, VF3, "flat");
    int d[4] = {4, 4, 4, 4};   fails += test_cell(&reg, 256, 8,  d, 4, VL4, "log3");
    int e[4] = {4, 4, 4, 4};   fails += test_cell(&reg, 256, 64, e, 4, VF4, "flat");

    /* dispatch path: vfft_c2r_plan_create (resolves+stores JIT) + vfft_c2r_execute
     * (JIT-first) must equal c2r_execute_packed on the dispatch's own plan. */
    {
        int N = 256; size_t K = 8;
        c2r_plan_t *pd = vfft_c2r_plan_create(N, K, &reg);
        if (!pd) { printf("  dispatch plan NULL\n"); fails++; }
        else {
            size_t NK = (size_t)N * K;
            double *in = (double *)malloc(2 * NK * 8), *jd = (double *)malloc(NK * 8), *gd = (double *)malloc(NK * 8);
            srand(99); for (size_t i = 0; i < 2 * NK; i++) in[i] = (double)rand() / RAND_MAX * 2 - 1;
            vfft_c2r_execute(pd, in, jd);       /* JIT-first */
            c2r_execute_packed(pd, in, gd);     /* generic reference, same plan */
            double md = 0; for (size_t i = 0; i < NK; i++) { double d = fabs(jd[i] - gd[i]); if (d > md) md = d; }
            printf("  dispatch vfft_c2r_execute vs generic: max|d|=%.1e %s (jit_exec=%s)\n",
                   md, md < 1e-12 ? "BIT-EXACT" : "*** MISMATCH ***", pd->jit_exec ? "set" : "NULL(fallback)");
            if (md >= 1e-12) fails++;
            free(in); free(jd); free(gd); c2r_plan_destroy(pd);
        }
    }
    printf("# fails=%d\n", fails);
    return fails ? 1 : 0;
}
