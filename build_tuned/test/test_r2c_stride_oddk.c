/* test_r2c_stride_oddk.c — odd-K r2c/c2r via the DECOUPLED-STRIDE path, after removing the
 * K%8 gates (r2c_dispatch.h / c2r_dispatch.h) and teaching the stride r2c workers to route a
 * non-VW-aligned B through the explicit-pack fallback (unaligned scratch + rem-aware inner tail).
 * For K >= the decouple threshold (32) with SPLIT layout on even N, the dispatcher now picks
 * STRIDE even at odd K.  Checks: (1) routing = STRIDE, (2) roundtrip c2r(r2c(x)) == scale*x.
 * Split lane-batched layout: x[n*K+v] -> re[k*K+v], im[k*K+v].
 * Build: python build.py --src test/test_r2c_stride_oddk.c */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define VFFT_RFFT_MAX_RADIX 32
#include "r2c_dispatch.h"
#include "c2r_dispatch.h"
#include "rfft_registry_avx2.h"
#include "c2r_registry_avx2.h"
#include "registry.h"

static rfft_codelets_t  RFFT;
static vfft_proto_registry_t C2C;
#define AAL(nbytes) malloc((size_t)(nbytes))

static double rt_relerr(const double *y, const double *x, size_t n) {
    double num = 0, den = 0, xmax = 0;
    for (size_t i = 0; i < n; i++) { num += y[i]*x[i]; den += x[i]*x[i]; if (fabs(x[i])>xmax) xmax=fabs(x[i]); }
    double s = den > 0 ? num/den : 0, e = 0;
    for (size_t i = 0; i < n; i++) { double d = fabs(y[i] - s*x[i]); if (d > e) e = d; }
    return (s != 0 && xmax > 0) ? e / (fabs(s)*xmax) : e;
}

static int cell(int N, size_t K) {
    int H = N/2; size_t pad = 32;
    double *x  = AAL(((size_t)N*K + pad)*8);
    double *re = AAL(((size_t)(H+1)*K + pad)*8);
    double *im = AAL(((size_t)(H+1)*K + pad)*8);
    double *xb = AAL(((size_t)N*K + pad)*8);
    srand(200 + N + (int)K);
    for (size_t i = 0; i < (size_t)N*K; i++) x[i] = (double)rand()/RAND_MAX - 0.5;
    memset(re, 0, ((size_t)(H+1)*K + pad)*8);
    memset(im, 0, ((size_t)(H+1)*K + pad)*8);

    vfft_r2c_plan_t *rp = vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT, &RFFT, NULL, &C2C);
    if (!rp) { printf("  N=%-4d K=%-4zu  r2c plan NULL  FAIL\n", N, K); return 1; }
    int r2c_stride = (rp->path == VFFT_R2C_PATH_STRIDE);
    vfft_r2c_execute_fwd(rp, x, re, im);

    vfft_c2r_disp_t *cp = vfft_c2r_disp_create_auto(N, K, &RFFT, &C2C);
    if (!cp) { printf("  N=%-4d K=%-4zu  c2r disp NULL  FAIL\n", N, K); vfft_r2c_plan_destroy(rp); return 1; }
    int c2r_stride = (cp->layout == VFFT_C2R_SPLIT);
    vfft_c2r_disp_execute(cp, re, im, xb);
    double rt = rt_relerr(xb, x, (size_t)N*K);

    int want_stride = (K >= 32);           /* K>=decouple threshold -> STRIDE expected */
    int route_ok = !want_stride || (r2c_stride && c2r_stride);
    int fail = (rt > 1e-9) || !route_ok;
    printf("  N=%-4d K=%-4zu rem%zu  r2c=%-6s c2r=%-7s  roundtrip=%.2e %s%s\n",
           N, K, K % 8, r2c_stride?"STRIDE":"rfft", c2r_stride?"stride":"NATURAL",
           rt, fail?"FAIL":"ok", (want_stride && !route_ok)?"  <NOT STRIDE!>":"");
    vfft_r2c_plan_destroy(rp); vfft_c2r_disp_destroy(cp);
    free(x); free(re); free(im); free(xb);
    return fail;
}

int main(void) {
    memset(&RFFT, 0, sizeof RFFT);
    rfft_register_all_avx2(&RFFT);
    c2r_register_all_avx2(&RFFT);
    vfft_proto_registry_init(&C2C);
    printf("# odd-K r2c/c2r via the DECOUPLED-STRIDE path (K%%8 gates removed; odd B -> explicit-pack fallback)\n");
    printf("# K>=32 must route STRIDE; roundtrip c2r(r2c(x))==scale*x is the gate.\n");
    const int Ns[] = {256, 512, 1024};
    const size_t Ks[] = {32, 40, 33, 41, 99, 127};   /* 32/40 even (regression), 33/41/99/127 odd */
    int fails = 0;
    for (size_t ni=0; ni<sizeof(Ns)/sizeof(Ns[0]); ni++)
        for (size_t ki=0; ki<sizeof(Ks)/sizeof(Ks[0]); ki++)
            fails += cell(Ns[ni], Ks[ki]);
    printf("\n# %s (%d failing checks)\n", fails?"FAIL":"PASS", fails);
    return fails ? 1 : 0;
}
