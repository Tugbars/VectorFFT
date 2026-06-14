/* jit_smoke.c — runtime JIT smoke + regression harness (native Windows).
 *
 *   plan = planner(...)                  // here: explicit factors via argv
 *   fn   = vfft_proto_plan_jit_fwd(plan) // PLANNER phase: emit+compile+load+cache
 *   ... hot loop: fn(plan, ...)          // zero JIT overhead, direct call
 *
 * Verifies: (1) fn is bit-exact vs the generic executor, (2) cache hit is fast,
 * (3) latency of the resolved fn vs generic. Pass factors as argv[1]
 * (default a COLD 8-stage cell); argv[2]=core (default P-core 14).
 */
#define VFFT_PROTO_DP_PACE_MS 0
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/core/env.h"
#include "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/core/executor.h"
#include "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/core/planner.h"
#include "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/prototype/generated/registry.h"
#include "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/jit/jit_runtime.h"

static double now_ns(void) {
    LARGE_INTEGER f, c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

int main(int argc, char **argv) {
    stride_env_init();
    int core = (argc > 2) ? atoi(argv[2]) : 14;
    if (stride_pin_thread(core) != 0) fprintf(stderr, "warn: pin failed\n");

    size_t K = 4;
    int factors[STRIDE_MAX_STAGES], nf = 0;
    const char *fs = (argc > 1) ? argv[1] : "4,4,4,4,4,4,4,8";
    { char buf[256]; strncpy(buf, fs, sizeof buf - 1); buf[sizeof buf - 1] = 0;
      char *t = strtok(buf, ","); while (t && nf < STRIDE_MAX_STAGES) { factors[nf++] = atoi(t); t = strtok(NULL, ","); } }
    int N = 1; for (int i = 0; i < nf; i++) N *= factors[i];

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_plan_t *plan = vfft_proto_plan_create_ex(N, K, factors, NULL, nf, 0, &reg);
    if (!plan) { printf("no plan\n"); return 1; }

    int is_baked = (vfft_proto_lookup_fwd_avx2(plan) != NULL);
    printf("=== JIT runtime: N=%d K=%zu factors=%s (%s) ===\n",
           N, K, fs, is_baked ? "BAKED cell" : "COLD cell -> JIT");

    /* PLANNER phase: resolve once (cold => emit+compile+load) */
    double r0 = now_ns();
    vfft_proto_exec_fn fn = vfft_proto_plan_jit_fwd(plan);
    double r1 = now_ns();
    /* second call: registry hit (in-process) */
    double s0 = now_ns(); vfft_proto_exec_fn fn2 = vfft_proto_plan_jit_fwd(plan); double s1 = now_ns();
    printf("  resolve 1st: %8.1f ms   resolve 2nd: %8.3f ms   fn=%p (matches=%d)\n",
           (r1 - r0) / 1e6, (s1 - s0) / 1e6, (void *)fn, fn == fn2);
    if (!fn) { printf("  resolve returned NULL (toolchain?) — generic fallback\n"); }

    size_t n = (size_t)N * K;
    double *src_re = _aligned_malloc(n*8,64), *src_im = _aligned_malloc(n*8,64);
    double *re     = _aligned_malloc(n*8,64), *im     = _aligned_malloc(n*8,64);
    double *ref_re = _aligned_malloc(n*8,64), *ref_im = _aligned_malloc(n*8,64);
    for (size_t i = 0; i < n; i++) { src_re[i] = sin(0.1*i); src_im[i] = cos(0.07*i); }

    /* reference: generic */
    memcpy(ref_re, src_re, n*8); memcpy(ref_im, src_im, n*8);
    vfft_proto_execute_fwd_generic(plan, ref_re, ref_im, K);

    /* resolved fn */
    memcpy(re, src_re, n*8); memcpy(im, src_im, n*8);
    if (fn) fn(plan, re, im, K, plan->K, 0);
    else    vfft_proto_execute_fwd_generic(plan, re, im, K);

    double md = 0;
    for (size_t i = 0; i < n; i++) { double d = fabs(re[i]-ref_re[i]) + fabs(im[i]-ref_im[i]); if (d > md) md = d; }
    printf("  accuracy: max_abs_diff vs generic = %.3e  %s\n", md, md == 0.0 ? "(BIT-EXACT)" : "");

    /* latency: best-of-N, untimed re-init */
    int iters = 200;
    double tg = 1e30, tj = 1e30;
    for (int w = 0; w < 5; w++) { memcpy(re,src_re,n*8); memcpy(im,src_im,n*8); vfft_proto_execute_fwd_generic(plan,re,im,K); }
    for (int it = 0; it < iters; it++) { memcpy(re,src_re,n*8); memcpy(im,src_im,n*8);
        double a=now_ns(); vfft_proto_execute_fwd_generic(plan,re,im,K); double b=now_ns(); if (b-a<tg) tg=b-a; }
    if (fn) {
        for (int w = 0; w < 5; w++) { memcpy(re,src_re,n*8); memcpy(im,src_im,n*8); fn(plan,re,im,K,plan->K,0); }
        for (int it = 0; it < iters; it++) { memcpy(re,src_re,n*8); memcpy(im,src_im,n*8);
            double a=now_ns(); fn(plan,re,im,K,plan->K,0); double b=now_ns(); if (b-a<tj) tj=b-a; }
    }
    printf("  latency (best of %d): generic %9.1f ns   resolved %9.1f ns   %.2fx\n",
           iters, tg, fn ? tj : tg, fn ? tg / tj : 1.0);
    return 0;
}
