/* test_2d_wisdom.c — Stage 3: 2D c2c with wisdom-driven inners + DIF support.
 * The 2D row pass now uses the full c2c executor (DIT *and* DIF), so a DIF inner
 * round-trips. This test builds 2D c2c plans with explicitly DIT and DIF inners
 * and checks fwd+bwd == N1*N2*x for both — the direct proof of DIF support.
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

/* Build a 2D c2c plan (N1xN2) with inners of a chosen orientation, roundtrip-test it.
 * N1=N2=64 so the factorization [8,8] is valid for both col (K=N2) and row (K=B). */
static void cell(int N1, int N2, int use_dif, const vfft_proto_registry_t *reg)
{
    size_t B = _fft2d_choose_tile(N2, N1);
    int f[] = {8, 8}; int nf = 2;   /* 64 = 8*8 */
    stride_plan_t *col = vfft_proto_plan_create_ex(N1, (size_t)N2, f, NULL, nf, use_dif, reg);
    stride_plan_t *row = vfft_proto_plan_create_ex(N2, B, f, NULL, nf, use_dif, reg);
    stride_plan_t *p = (col && row) ? stride_plan_2d_from(N1, N2, B, col, row) : NULL;
    if (!p) { if (col) stride_plan_destroy(col); if (row) stride_plan_destroy(row);
              printf("  %dx%-3d inner=%-3s  PLAN NULL\n", N1, N2, use_dif ? "DIF" : "DIT"); return; }
    size_t plane = (size_t)N1 * N2;
    double *re = AAL(plane*8), *im = AAL(plane*8), *xr = AAL(plane*8), *xi = AAL(plane*8);
    srand(7 + use_dif);
    for (size_t j = 0; j < plane; j++) { xr[j] = (double)rand()/RAND_MAX - 0.5; xi[j] = (double)rand()/RAND_MAX - 0.5; }
    memcpy(re, xr, plane*8); memcpy(im, xi, plane*8);
    stride_execute_fwd(p, re, im);
    stride_execute_bwd(p, re, im);
    double rt = 0, sc = (double)N1 * N2;
    for (size_t j = 0; j < plane; j++) { double a = fabs(re[j]/sc - xr[j]), b = fabs(im[j]/sc - xi[j]); if (a>rt)rt=a; if (b>rt)rt=b; }
    printf("  %dx%-3d inner=%-3s (B=%zu)  roundtrip=%.0e  %s\n", N1, N2, use_dif ? "DIF" : "DIT", B, rt, (rt < 1e-9) ? "OK" : "*** FAIL ***");
    stride_plan_destroy(p); AFR(re); AFR(im); AFR(xr); AFR(xi);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    stride_env_init(); stride_pin_thread(0);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("== Stage 3: 2D c2c row pass supports DIT *and* DIF inners ==\n");
    cell(64, 64, 0, &reg);   /* DIT inners */
    cell(64, 64, 1, &reg);   /* DIF inners (was broken: DIT-only slice helper) */
    return 0;
}
