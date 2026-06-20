/* test_rfft_calibrate.c — proves the rfft axis now honors rigor: vfft_rfft_calibrate
 * runs on a low-K rfft cell, populates W->rfft, and the plan it builds is forward-correct.
 * Single-TU: #includes vfft.c so it can see the opaque wisdom struct.
 * Build: cd build_tuned && python build.py --src benches/test_rfft_calibrate.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "vfft.c"
#if defined(_WIN32)
#include <malloc.h>
#define AAL(n) _aligned_malloc((n), 64)
#define AFR(p) _aligned_free(p)
#else
#define AAL(n) aligned_alloc(64, (n))
#define AFR(p) free(p)
#endif

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);   /* unbuffered: see progress live */
    stride_env_init();
    stride_pin_thread(0);
    printf("== rfft calibration: rigor -> the rfft's OWN axis (factor+variant) ==\n");

    /* (1) direct calibrator on a low-K rfft cell */
    vfft_proto_wisdom_entry_t e;
    int rc = vfft_rfft_calibrate(256, 8, _rfft_registry(), &e);
    printf("  vfft_rfft_calibrate(256,8) rc=%d  factors=[", rc);
    for (int s = 0; s < e.nf; s++) printf("%s%d", s ? "," : "", e.factors[s]);
    printf("]  variants=[");
    for (int s = 0; s < e.nf; s++) printf("%s%d", s ? "," : "", e.variants[s]);
    printf("]  %.0f ns\n", e.best_ns);

    /* (2) via the PUBLIC API with a fresh (empty) bundle: create must calibrate-on-miss,
     *     populate W->rfft, and the plan must be forward-correct vs a reference DFT. */
    vfft_wisdom *w = vfft_wisdom_load("/c/tmp/vfft_rfftcal");   /* empty bundle */
    struct vfft_wisdom_s *W = (struct vfft_wisdom_s *)w;
    int N = 256; size_t K = 8; int halfN = N / 2;
    int before = (vfft_proto_wisdom_lookup(&W->rfft, N, K) != NULL);
    vfft_config_t c = {.transform = VFFT_R2C, .placement = VFFT_OUTOFPLACE, .rigor = VFFT_MEASURE,
                       .dims = 1, .n = {N, 0}, .howmany = K, .nthreads = 1, .wisdom = w};
    vfft_plan p = vfft_create(&c);
    int after = (vfft_proto_wisdom_lookup(&W->rfft, N, K) != NULL);

    size_t insz = (size_t)N * K, outsz = (size_t)(halfN + 1) * K;
    double *x = AAL(insz * 8), *orr = AAL(outsz * 8), *oii = AAL(outsz * 8);
    srand(3); for (size_t i = 0; i < insz; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
    double err = -1;
    if (p) {
        vfft_execute(p, VFFT_FORWARD, x, NULL, orr, oii);
        err = 0;
        for (int k = 0; k <= halfN; k++) {
            double rr = 0, ri = 0;
            for (int n = 0; n < N; n++) { double a = -2.0 * M_PI * k * n / (double)N; rr += x[(size_t)n*K]*cos(a); ri += x[(size_t)n*K]*sin(a); }
            double er = fabs(orr[(size_t)k*K] - rr), ei = fabs(oii[(size_t)k*K] - ri);
            if (er > err) err = er; if (ei > err) err = ei;
        }
    }
    printf("  via API N=256 K=8: rfft-wisdom before=%d after=%d  fwd_err=%.0e  %s\n",
           before, after, err, (rc == 0 && !before && after && err >= 0 && err < 1e-9) ? "OK" : "CHECK");

    if (p) vfft_destroy(p);
    vfft_wisdom_free(w); AFR(x); AFR(orr); AFR(oii);
    return 0;
}
