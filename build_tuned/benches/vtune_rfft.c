/* vtune_rfft.c — VTune profiling target for the rfft high-K loss cell.
 *
 * Loops ONE engine (dag rfft "ours", or MKL r2c) at the located loss cell
 * N=256 K=256 for ~`seconds`, pinned to a P-core, so uarch-exploration samples
 * concentrate on that engine. Profile both, diff the Top-Down: is our rfft
 * SQ-full / store / L2-bound (prefetch/blocking-addressable) or just moving more
 * bytes than MKL (structural)? — see docs/vtune-profiles/ methodology.
 *
 * Build: build_tuned/build.py --src benches/vtune_rfft.c --mkl --compile
 * Run  : vtune_rfft [ours|mkl] [seconds=8] [core=2]
 * Profile (ADMIN PowerShell, EBS needs admin):
 *   vtune -collect uarch-exploration -knob sampling-interval=0.5 \
 *         -result-dir vt_ours -- vtune_rfft.exe ours 8 2
 */
#define VFFT_RFFT_MAX_RADIX 32
#define VFFT_RFFT_RANGED 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "core/env.h"            /* stride_env_init, stride_pin_thread */
#include "core/dp_planner.h"     /* vfft_proto_now_ns, posix_memalign, aligned_free */
#include "rfft_registry_avx2.h"  /* dag rfft (incl rfft.h) */

static double *ad(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void**)&p, 64, n*sizeof(double)) != 0) { fprintf(stderr,"oom\n"); exit(1); }
    return p;
}

int main(int argc, char **argv) {
    stride_env_init();
    const char *mode = (argc > 1) ? argv[1] : "ours";
    const double secs = (argc > 2) ? atof(argv[2]) : 8.0;
    const int core = (argc > 3) ? atoi(argv[3]) : 2;
    const int do_ours = (strcmp(mode, "mkl") != 0);
    if (stride_pin_thread(core) != 0) fprintf(stderr, "warn: pin cpu%d\n", core);
    mkl_set_num_threads(1);

    const int N = 256; const size_t K = 256, NK = (size_t)N*K, halfN = N/2;

    /* dag rfft (8,32) — the cell that loses ~2x to MKL r2c at K=256 */
    int f[2] = {8,32}, nf = 2;
    rfft_codelets_t reg; memset(&reg, 0, sizeof reg); rfft_register_all_avx2(&reg);
    rfft_plan_t *pf = rfft_plan_create(N, K, f, nf, &reg);
    if (!pf) { printf("plan NULL\n"); return 1; }
    double *x = ad(NK), *hc = ad(2*NK);
    srand(7); for (size_t i = 0; i < NK; i++) x[i] = (double)rand()/RAND_MAX*2-1;

    /* MKL r2c, transform-major (its native batch layout) */
    DFTI_DESCRIPTOR_HANDLE h = 0;
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N);
    DftiSetValue(h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
    DftiSetValue(h, DFTI_INPUT_DISTANCE, (MKL_LONG)N);
    DftiSetValue(h, DFTI_OUTPUT_DISTANCE, (MKL_LONG)(halfN+1));
    DftiCommitDescriptor(h);
    double *xin = ad(NK), *cce = ad((halfN+1)*K*2);
    for (size_t t = 0; t < K; t++) for (int n = 0; n < N; n++) xin[t*N+n] = x[n*K+t];

    for (int w = 0; w < 20; w++) {           /* warm both */
        rfft_execute_fwd_packed(pf, x, hc); DftiComputeForward(h, xin, cce);
    }
    long iters = 0; double t0 = vfft_proto_now_ns();
    while (vfft_proto_now_ns() - t0 < secs * 1e9) {
        if (do_ours) rfft_execute_fwd_packed(pf, x, hc);
        else         DftiComputeForward(h, xin, cce);
        iters++;
    }
    double el = (vfft_proto_now_ns() - t0) / 1e9;
    printf("%s  N=256 K=256: %ld iters in %.2fs  (%.1f us/iter)\n",
           do_ours ? "ours" : "mkl", iters, el, el*1e6/iters);

    DftiFreeDescriptor(&h);
    vfft_proto_aligned_free(x); vfft_proto_aligned_free(hc);
    vfft_proto_aligned_free(xin); vfft_proto_aligned_free(cce);
    rfft_plan_destroy(pf);
    return 0;
}
