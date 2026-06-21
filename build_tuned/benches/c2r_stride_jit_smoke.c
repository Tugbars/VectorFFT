/* c2r_stride_jit_smoke.c — bit-exact validation of the SPLIT stride c2r JIT inner.
 *
 * Builds a stride r2c plan directly (stride_r2c_plan over an inner c2c(N/2)), then
 * runs stride_execute_c2r with the GENERIC inner (slice_until) vs the JIT inner
 * (vfft_proto_plan_jit_bwd at start_stage=1) and compares. The JIT wire lives in the
 * FUSED backward branch (inner->stages[0].n1_scaled_bwd); the smoke reports whether
 * that branch fires for this cell. jit-vs-generic differs only at FP-rounding
 * (machine-eps) — both correct (JIT == baked, generic interpreter groups differently).
 *
 * Build: cd build_tuned && python build.py --src benches/c2r_stride_jit_smoke.c --jit --compile
 * Run on a free P-core (pins core 4).
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "executor.h"
#include "threads.h"             /* thread pool symbols r2c.h's MT workers reference */
#include "planner.h"
#include "proto_stride_compat.h" /* bridge: stride_* shims + STRIDE_ALIGNED_* (r2c.h is written against it) */
#include "env.h"
#include "r2c.h"                 /* stride_r2c_plan, stride_execute_c2r, stride_r2c_inner_plan/_set_inner_jit_bwd */
#include "jit/jit_runtime.h" /* vfft_proto_plan_jit_bwd */
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

    const int N = 512;
    const size_t K = 256, B = 256;   /* block_K=K -> single block (serial) */
    const int hp1 = N / 2 + 1;

    stride_plan_t *inner = vfft_proto_auto_plan(N / 2, B, &reg, NULL);
    if (!inner) { printf("inner NULL\n"); return 1; }
    stride_plan_t *sp = stride_r2c_plan(N, K, B, inner); /* owns inner */
    if (!sp) { printf("sp NULL\n"); return 1; }

    stride_plan_t *cin = stride_r2c_inner_plan(sp);
    int fused = (cin && cin->num_stages > 0 && cin->stages[0].n1_scaled_bwd != NULL);

    size_t CN = (size_t)hp1 * K, RN = (size_t)N * K;
    double *ore = AAL(CN * 8), *oim = AAL(CN * 8), *A = AAL(RN * 8), *Bb = AAL(RN * 8);
    srand(31);
    for (size_t i = 0; i < CN; i++) { ore[i] = (double)rand() / RAND_MAX - 0.5;
                                      oim[i] = (double)rand() / RAND_MAX - 0.5; }

    /* generic inner */
    stride_r2c_set_inner_jit_bwd(sp, NULL);
    stride_execute_c2r(sp, ore, oim, A);
    /* JIT inner */
    vfft_proto_exec_fn jbwd = cin ? vfft_proto_plan_jit_bwd(cin) : NULL;
    stride_r2c_set_inner_jit_bwd(sp, jbwd);
    stride_execute_c2r(sp, ore, oim, Bb);

    double d = maxd(A, Bb, RN);
    printf("stride c2r JIT smoke  N=%d K=%zu  fused-path=%s  jit_bwd=%s\n",
           N, K, fused ? "yes" : "NO (serial fallback)", jbwd ? "resolved" : "NULL");
    printf("  c2r jit-vs-generic max|d| = %.1e  %s\n", d, d < 1e-9 ? "ok (rounding-level)" : "*** MISMATCH ***");

    int fired = (d > 0.0); /* jit differs from generic at FP-rounding => the JIT path ran */
    int ok;
    const char *verdict;
    if (!jbwd) { ok = 0; verdict = "FAIL (jit_bwd did not resolve)"; }
    else if (d >= 1e-9) { ok = 0; verdict = "FAIL (jit != generic beyond rounding)"; }
    else if (!fired) { ok = 0; verdict = "WIRE-INACTIVE (jit==generic bit-exact: JIT path not invoked)"; }
    else { ok = 1; verdict = "PASS (JIT inner fired; rounding-level match)"; }
    printf("%s\n", verdict);

    AFR(ore); AFR(oim); AFR(A); AFR(Bb);
    stride_plan_destroy(sp);
    return ok ? 0 : 1;
}
