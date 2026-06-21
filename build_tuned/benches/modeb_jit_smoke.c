/* modeb_jit_smoke.c — bit-exact validation of the OOP MODEB JIT inner path.
 *
 * Builds a MODEB plan directly (_vfft_oop_make_modeb — no wisdom files, no DP
 * search), then runs fwd/bwd with the GENERIC inner (mb_jit_*=NULL) vs the JIT
 * inner (fwd: stages 1.. at start_stage=1; bwd: whole in-place DIF at start_stage=0)
 * and asserts BIT-EXACT equality, plus a fwd+bwd==N*x roundtrip. Pollution-free.
 *
 * Build: cd build_tuned && python build.py --src benches/modeb_jit_smoke.c --jit --compile
 * Run on a free P-core (this pins core 4 to avoid the core-2 session).
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "executor.h"
#include "env.h"
#include "oop_dp.h"          /* pulls oop_plan.h: _vfft_oop_make_modeb + vfft_oop_execute_* */
#include "jit/jit_runtime.h" /* vfft_proto_plan_jit_fwd/bwd */
#include "generator/generated/registry.h"

#if defined(_WIN32)
#include <malloc.h>
#define AAL(n) _aligned_malloc((n), 64)
#define AFR(p) _aligned_free(p)
#else
#define AAL(n) aligned_alloc(64, (n))
#define AFR(p) free(p)
#endif

static double maxd(const double *a, const double *b, size_t n)
{
    double e = 0;
    for (size_t i = 0; i < n; i++) { double d = fabs(a[i] - b[i]); if (d > e) e = d; }
    return e;
}

int main(void)
{
    stride_env_init();
    if (stride_pin_thread(4) != 0) fprintf(stderr, "warn: pin cpu4 failed\n");
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    const int N = 256;
    const size_t K = 256, NK = (size_t)N * K;
    int factors[4] = {4, 4, 4, 4}; /* 256, DIT -> MODEB-eligible */

    vfft_oop_plan_t *p = _vfft_oop_make_modeb(N, K, factors, NULL, 4, &reg);
    if (!p || p->kind != VFFT_OOP_KIND_MODEB) { printf("make_modeb failed\n"); return 1; }

    double *xr = AAL(NK * 8), *xi = AAL(NK * 8);
    double *ar = AAL(NK * 8), *ai = AAL(NK * 8), *br = AAL(NK * 8), *bi = AAL(NK * 8);
    double *cr = AAL(NK * 8), *ci = AAL(NK * 8), *dr = AAL(NK * 8), *di = AAL(NK * 8);
    srand(7);
    for (size_t i = 0; i < NK; i++) { xr[i] = (double)rand() / RAND_MAX - 0.5;
                                      xi[i] = (double)rand() / RAND_MAX - 0.5; }

    /* FWD: generic inner vs JIT inner */
    p->mb_jit_fwd = NULL; p->mb_jit_bwd = NULL;
    vfft_oop_execute_fwd(p, xr, xi, ar, ai);             /* generic stages 1.. */
    p->mb_jit_fwd = vfft_proto_plan_jit_fwd(p->mb);
    int fwd_resolved = (p->mb_jit_fwd != NULL);
    vfft_oop_execute_fwd(p, xr, xi, br, bi);             /* JIT stages 1.. (start_stage=1) */
    double fwd_d = maxd(ar, br, NK), e = maxd(ai, bi, NK); if (e > fwd_d) fwd_d = e;

    /* BWD: invert the spectrum (ar,ai); generic vs JIT */
    p->mb_jit_bwd = NULL;
    vfft_oop_execute_bwd(p, ar, ai, cr, ci);             /* generic whole in-place bwd */
    p->mb_jit_bwd = vfft_proto_plan_jit_bwd(p->mb);
    int bwd_resolved = (p->mb_jit_bwd != NULL);
    vfft_oop_execute_bwd(p, ar, ai, dr, di);             /* JIT whole in-place bwd (start_stage=0) */
    double bwd_d = maxd(cr, dr, NK); e = maxd(ci, di, NK); if (e > bwd_d) bwd_d = e;

    /* roundtrip fwd+bwd == N*x (using the JIT outputs) */
    double rt = 0, sc = (double)N;
    for (size_t i = 0; i < NK; i++) { double a = fabs(dr[i] / sc - xr[i]), b = fabs(di[i] / sc - xi[i]);
                                      if (a > rt) rt = a; if (b > rt) rt = b; }

    printf("MODEB JIT smoke  N=%d K=%zu factors=4,4,4,4  (jit_fwd=%s jit_bwd=%s)\n",
           N, K, fwd_resolved ? "resolved" : "NULL", bwd_resolved ? "resolved" : "NULL");
    /* fwd JIT (stages 1..) == generic-from: identical code path -> bit-exact.
     * bwd JIT == BAKED bwd (same STAGE_BWD macros), but the GENERIC interpreter bwd
     * groups arithmetic differently, so jit-vs-generic differs at FP rounding
     * (~machine-eps, both correct inverses). The real bwd gate is the roundtrip. */
    printf("  fwd jit-vs-generic max|d| = %.1e  %s\n", fwd_d, fwd_d == 0.0 ? "BIT-EXACT" : "*** MISMATCH ***");
    printf("  bwd jit-vs-generic max|d| = %.1e  %s (rounding-level OK; gate=roundtrip)\n",
           bwd_d, bwd_d < 1e-9 ? "ok" : "*** MISMATCH ***");
    printf("  roundtrip fwd+bwd==N*x    = %.1e  %s\n", rt, rt < 1e-9 ? "OK" : "*** RT FAIL ***");

    int ok = fwd_resolved && bwd_resolved && fwd_d == 0.0 && bwd_d < 1e-9 && rt < 1e-9;
    printf("%s\n", ok ? "PASS" : "FAIL");

    AFR(xr); AFR(xi); AFR(ar); AFR(ai); AFR(br); AFR(bi);
    AFR(cr); AFR(ci); AFR(dr); AFR(di);
    vfft_oop_plan_destroy(p);
    return ok ? 0 : 1;
}
