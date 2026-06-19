/* bench_r2c_strided_vs_mkl.c — DECISIVE: does the stride-r2c path (half-size
 * complex FFT + Hermitian recombine, core/r2c.h) beat native rfft AND MKL r2c
 * at high K (the K=256 regime where native rfft loses ~2x to MKL)?
 *
 * N=256, K in {8,32,64,128,256}, single thread (MKL pinned), core-2 affinity.
 *
 * Three engines, all REAL forward (r2c), all transform-major real input:
 *   1) stride-r2c : inner c2c(128,K) via core/r2c.h. SPLIT output (out_re,out_im),
 *                   natural freq order, bins 0..N/2. Inner MUST be a guard-
 *                   whitelisted shape; for halfN=128 (no radix-128 codelet) the
 *                   only legal multi-stage shapes are (8,16)/(16,8). We use
 *                   (16,8) — the doc-58-validated shape. (Wisdom's N=128 plans
 *                   like (4,32) / (4,4,8) are REJECTED by the r2c guard, so the
 *                   inner cannot be wisdom-driven here; we note that.)
 *   2) native rfft: rfft_execute_fwd_packed, factors {8,32}, Kb=K, hc buf 2*N*K.
 *   3) MKL r2c    : DFTI_REAL, CONJUGATE_EVEN_STORAGE=COMPLEX_COMPLEX, NOT_INPLACE,
 *                   transform-major (INPUT_DISTANCE=N, OUTPUT_DISTANCE=N/2+1).
 *
 * CORRECTNESS GATE per cell: stride-r2c out_re[k*K]/out_im[k*K] (lane 0), k=0..N/2,
 * vs a direct reference DFT X[k]=sum_n x[n*K]*exp(-2pi i k n/N). Abort cell if
 * max abs err >= 1e-9.
 *
 * Build: cd build_tuned && python build.py --src benches/bench_r2c_strided_vs_mkl.c --mkl --compile
 * Run  : PATH += MKL bin + mingw bin, then run the .exe.
 */
#define _GNU_SOURCE 1
#define VFFT_RFFT_MAX_RADIX 32
#define VFFT_RFFT_RANGED 1      /* ranged terminator/stages for native rfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mkl_dfti.h>
#include <mkl_service.h>       /* mkl_set_num_threads — pin to 1 for a fair ST race */

#include "executor.h"
#include "env.h"          /* stride_env_init + stride_pin_thread */
#include "planner.h"
#include "dp_planner.h"   /* vfft_proto_now_ns */
#include "proto_stride_compat.h"  /* threads pool + STRIDE_ALIGNED_ALLOC, before r2c.h */
#include "r2c.h"          /* stride_r2c_plan, stride_execute_r2c, stride_r2c_data_t */
#include "generator/generated/registry.h"
/* native rfft (packed half-complex) — full ABI-typed registry (r2cf + hc2hc + RANGED) */
#include "rfft_registry_avx2.h"

#define PIN_CORE 2
#define BEST_OF 15

static double *alloc_d(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}
static int reps_for(size_t total) {
    int r = (int)(4e6 / (total + 1));
    if (r < 30) r = 30; if (r > 200000) r = 200000;
    return r;
}

/* min-of-BEST_OF ns wrappers */
static double bench_stride_r2c(const stride_plan_t *p, const double *in,
                               double *re, double *im, size_t total) {
    for (int w = 0; w < 10; w++) stride_execute_r2c(p, in, re, im);
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < BEST_OF; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) stride_execute_r2c(p, in, re, im);
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;
}
static double bench_rfft(const rfft_plan_t *pf, const double *x, double *hc, size_t total) {
    for (int w = 0; w < 10; w++) rfft_execute_fwd_packed(pf, x, hc);
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < BEST_OF; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) rfft_execute_fwd_packed(pf, x, hc);
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;
}
static double bench_mkl(DFTI_DESCRIPTOR_HANDLE h, const double *xin, double *cce, size_t total) {
    for (int w = 0; w < 10; w++) DftiComputeForward(h, (void *)xin, cce);
    int reps = reps_for(total); double best = 1e18;
    for (int t = 0; t < BEST_OF; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) DftiComputeForward(h, (void *)xin, cce);
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;
}

int main(int argc, char **argv) {
    (void)argc; (void)argv;
    stride_env_init();
    if (stride_pin_thread(PIN_CORE) != 0)
        fprintf(stderr, "warn: pin cpu%d failed\n", PIN_CORE);
    mkl_set_num_threads(1);   /* fair single-thread race (ours is ST) */

    const int N = 256, halfN = N / 2;
    const size_t Ks[] = {8, 32, 64, 128, 256};
    const int nK = (int)(sizeof Ks / sizeof Ks[0]);

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    /* native rfft registry (r2cf + hc2hc + RANGED variants) */
    rfft_codelets_t rreg; memset(&rreg, 0, sizeof rreg);
    rfft_register_all_avx2(&rreg);
    int rf[2] = {8, 32};

    /* PROBE: would a wisdom-driven inner (N=128) survive the r2c guard? */
    vfft_proto_wisdom_t wis; int have_wis =
        (vfft_proto_wisdom_load(&wis, "../src/dag-fft-compiler/generator/generated/spike_wisdom.txt") == 0);
    if (!have_wis)
        have_wis = (vfft_proto_wisdom_load(&wis,
                    "../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt") == 0);
    printf("# wisdom load: %s\n", have_wis ? "OK" : "FAILED (greedy inner only)");

    printf("=== stride-r2c vs native-rfft vs MKL  (N=256, ST, pinned cpu%d, best-of-%d) ===\n",
           PIN_CORE, BEST_OF);
    printf("# stride-r2c inner c2c(128,K) = (16,8) fixed [guard-whitelisted; no radix-128]\n");
    printf("%-5s %14s %14s %14s %12s %12s %10s\n",
           "K", "stride_r2c_ns", "rfft_ns", "mkl_ns",
           "stride/mkl", "rfft/mkl", "stride/rfft");
    printf("------+--------------+--------------+--------------+------------+------------+----------\n");

    for (int ki = 0; ki < nK; ki++) {
        size_t K = Ks[ki];
        size_t total = (size_t)N * K;

        /* ---- input: lane-batched real x[n*K + lane], same layout for all 3 ---- */
        double *x = alloc_d(total);
        srand(7 + (int)K);
        for (size_t i = 0; i < total; i++) x[i] = (double)rand() / RAND_MAX * 2 - 1;

        /* ===== build stride-r2c plan (inner c2c(128,K) = (16,8)) ===== */
        int f[2] = {16, 8}, v[2] = {2, 2};
        stride_plan_t *inner = vfft_proto_plan_create_ex(halfN, K, f, v, 2, /*use_dif=*/0, &reg);
        if (!inner) { printf("%-5zu  inner c2c(16,8) plan NULL\n", K); vfft_proto_aligned_free(x); continue; }
        stride_plan_t *p = stride_r2c_plan(N, K, /*block_K=*/K, inner);
        if (!p) { printf("%-5zu  stride_r2c_plan NULL (guard rejected inner)\n", K); vfft_proto_aligned_free(x); continue; }

        /* Confirm whether a wisdom-driven inner (N=128) is rejected by the r2c guard. */
        if (have_wis) {
            const vfft_proto_wisdom_entry_t *e = vfft_proto_wisdom_lookup(&wis, halfN, K);
            if (e && e->nf > 0) {
                stride_plan_t *win = vfft_proto_plan_create_ex(halfN, K, e->factors, e->variants,
                                                               e->nf, e->use_dif_forward, &reg);
                stride_plan_t *wp = win ? stride_r2c_plan(N, K, K, win) : NULL;
                char fs[64]; size_t pp = 0;
                for (int s = 0; s < e->nf; s++) pp += (size_t)snprintf(fs+pp, sizeof fs-pp, "%s%d", s?",":"", e->factors[s]);
                printf("# probe: wisdom inner N=128 K=%zu = (%s) -> r2c guard %s\n",
                       K, fs, wp ? "ACCEPTED" : "REJECTED (use fixed (16,8))");
                if (wp) stride_plan_destroy(wp);
            }
        }

        double *sr = alloc_d(total), *si = alloc_d(total);

        /* ===== correctness gate: lane 0 vs reference DFT, k=0..N/2 ===== */
        memset(sr, 0, total * sizeof(double)); memset(si, 0, total * sizeof(double));
        stride_execute_r2c(p, x, sr, si);
        double maxerr = 0.0;
        for (int k = 0; k <= halfN; k++) {
            double ref_re = 0.0, ref_im = 0.0;
            for (int n = 0; n < N; n++) {
                double xn = x[(size_t)n * K + 0];   /* lane 0 */
                double ang = -2.0 * M_PI * (double)k * (double)n / (double)N;
                ref_re += xn * cos(ang);
                ref_im += xn * sin(ang);
            }
            double gr = sr[(size_t)k * K + 0], gi = si[(size_t)k * K + 0];
            double er = fabs(gr - ref_re), ei = fabs(gi - ref_im);
            if (er > maxerr) maxerr = er;
            if (ei > maxerr) maxerr = ei;
        }
        if (maxerr >= 1e-9) {
            printf("%-5zu  *** CORRECTNESS FAIL: stride-r2c max abs err = %.3e (ABORT cell) ***\n",
                   K, maxerr);
            vfft_proto_aligned_free(x); vfft_proto_aligned_free(sr); vfft_proto_aligned_free(si);
            stride_plan_destroy(p);
            continue;
        }

        /* ===== native rfft plan ===== */
        rfft_plan_t *pf = rfft_plan_create(N, K, rf, 2, &rreg);
        double *hc = NULL;
        int rfft_ok = (pf != NULL);
        if (pf) { pf->Kb = K; hc = alloc_d(2 * total); }

        /* ===== MKL r2c (transform-major) ===== */
        DFTI_DESCRIPTOR_HANDLE h = 0; int mkl_ok = 0;
        double *xin = alloc_d(total), *cce = alloc_d((size_t)(halfN + 1) * K * 2);
        for (size_t t = 0; t < K; t++)
            for (int n = 0; n < N; n++) xin[t * N + n] = x[(size_t)n * K + t];
        DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N);
        DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
        DftiSetValue(h, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
        DftiSetValue(h, DFTI_OUTPUT_DISTANCE, (MKL_LONG)(halfN + 1));
        mkl_ok = (DftiCommitDescriptor(h) == DFTI_NO_ERROR);

        /* ===== time all three ===== */
        double s_ns = bench_stride_r2c(p, x, sr, si, total);
        double r_ns = rfft_ok ? bench_rfft(pf, x, hc, total) : 0;
        double m_ns = mkl_ok ? bench_mkl(h, xin, cce, total) : 0;

        double s_over_m = (m_ns > 0 && s_ns > 0) ? m_ns / s_ns : 0;
        double r_over_m = (m_ns > 0 && r_ns > 0) ? m_ns / r_ns : 0;
        double s_over_r = (r_ns > 0 && s_ns > 0) ? r_ns / s_ns : 0;
        printf("%-5zu %14.1f %14.1f %14.1f %11.3fx %11.3fx %9.3fx\n",
               K, s_ns, r_ns, m_ns, s_over_m, r_over_m, s_over_r);

        if (h) DftiFreeDescriptor(&h);
        vfft_proto_aligned_free(x); vfft_proto_aligned_free(sr); vfft_proto_aligned_free(si);
        vfft_proto_aligned_free(xin); vfft_proto_aligned_free(cce);
        if (hc) vfft_proto_aligned_free(hc);
        if (pf) rfft_plan_destroy(pf);
        stride_plan_destroy(p);   /* frees inner via override_destroy */
    }

    if (have_wis) vfft_proto_wisdom_free(&wis);
    printf("\n# ratio>1 = OURS faster. stride/mkl = MKL_ns/stride_ns; rfft/mkl = MKL_ns/rfft_ns.\n");
    printf("# stride/rfft = rfft_ns/stride_ns (>1 => stride-r2c faster than native-rfft).\n");
    return 0;
}
