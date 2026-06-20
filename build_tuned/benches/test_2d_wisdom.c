/* test_2d_wisdom.c — Stage 3: 2D c2c with wisdom-driven inners (vfft_create).
 * Proves large 2D (1024^2) is now tractable (no exhaustive-at-create stall) and
 * round-trips. Reports create time (the old exhaustive path was minutes at N=1024).
 * Build: cd build_tuned && python build.py --src benches/test_2d_wisdom.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
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
    setvbuf(stdout, NULL, _IONBF, 0);
    stride_env_init();
    stride_pin_thread(0);
    printf("== Stage 3: 2D c2c wisdom-driven inners (create time + roundtrip) ==\n");
    vfft_wisdom *w = vfft_wisdom_load("C:/tmp/vfft_2dcal");   /* native path so saves work */

    int Ns[] = {256, 512, 1024}; int nN = 3;
    for (int i = 0; i < nN; i++) {
        int N = Ns[i]; size_t plane = (size_t)N * N;
        double t0 = vfft_proto_now_ns();
        vfft_config_t c = {.transform = VFFT_C2C, .placement = VFFT_INPLACE, .rigor = VFFT_MEASURE,
                           .dims = 2, .n = {N, N}, .howmany = 1, .nthreads = 1, .wisdom = w};
        vfft_plan p = vfft_create(&c);
        double create_ms = (vfft_proto_now_ns() - t0) / 1e6;
        if (!p) { printf("  %dx%-4d  CREATE NULL\n", N, N); continue; }
        double *re = AAL(plane*8), *im = AAL(plane*8), *xr = AAL(plane*8), *xi = AAL(plane*8);
        srand(7 + N);
        for (size_t j = 0; j < plane; j++) { xr[j] = (double)rand()/RAND_MAX - 0.5; xi[j] = (double)rand()/RAND_MAX - 0.5; }
        memcpy(re, xr, plane*8); memcpy(im, xi, plane*8);
        vfft_execute(p, VFFT_FORWARD, re, im, re, im);
        vfft_execute(p, VFFT_BACKWARD, re, im, re, im);
        double rt = 0, sc = (double)N * N;
        for (size_t j = 0; j < plane; j++) { double a = fabs(re[j]/sc - xr[j]), b = fabs(im[j]/sc - xi[j]); if (a>rt)rt=a; if (b>rt)rt=b; }
        printf("  %dx%-4d  create=%7.0f ms  roundtrip=%.0e  %s\n", N, N, create_ms, rt, (rt < 1e-9) ? "OK" : "CHECK");
        vfft_destroy(p); AFR(re); AFR(im); AFR(xr); AFR(xi);
    }
    vfft_wisdom_free(w);
    return 0;
}
