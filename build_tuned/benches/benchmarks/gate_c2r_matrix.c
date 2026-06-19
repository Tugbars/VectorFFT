/* c2r gate MATRIX: round-trip random x -> rfft_execute_fwd_packed ->
 * c2r_execute_packed -> N*x, across factorizations exercising:
 *   - nf=1 leaf-only (regression of the previously gated path, new ABI)
 *   - nf=2 with mid (m even), without mid (m odd), r>2 stages
 *   - nf=3/4: plane ping-pong + Q>1 fold path
 *   - (4,4,16) the bench plan and (8,32) the MKL-beating forward plan
 * Oracle chain: the forward executor (gated vs MKL, doc 60). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../core/rfft.h"
#include "../core/c2r.h"

#define DECL_F(r) extern void radix##r##_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
#define DECL_B(r) extern void radix##r##_r2cb_avx512(const double*,const double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
#define DECL_HF(r) extern void radix##r##_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
#define DECL_HB(r) extern void radix##r##_hc2hc_dif_bwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
DECL_F(2) DECL_F(3) DECL_F(4) DECL_F(5) DECL_F(8) DECL_F(16) DECL_F(32)
DECL_B(2) DECL_B(3) DECL_B(4) DECL_B(5) DECL_B(8) DECL_B(16) DECL_B(32)
DECL_HF(2) DECL_HF(4) DECL_HF(5) DECL_HF(8)
DECL_HB(2) DECL_HB(4) DECL_HB(5) DECL_HB(8)

static int run_case(const rfft_codelets_t *reg, int N, size_t K,
                    const int *factors, int nf)
{
    rfft_plan_t *pf = rfft_plan_create(N, K, factors, nf, reg);
    c2r_plan_t  *pb = c2r_plan_create(N, K, factors, nf, reg);
    if (!pf || !pb) { printf("N=%-4d nf=%d plan FAIL\n", N, nf); return 1; }

    size_t NK = (size_t)N * K;
    double *x  = aligned_alloc(64, NK * 8);
    double *hc = aligned_alloc(64, NK * 8);
    double *y  = aligned_alloc(64, NK * 8);
    srand(42 + N);
    for (size_t i = 0; i < NK; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
    for (size_t i = 0; i < NK; i++) { hc[i] = NAN; y[i] = NAN; }

    rfft_execute_fwd_packed(pf, x, hc);
    c2r_execute_packed(pb, hc, y);

    double maxerr = 0.0; size_t bad = 0;
    for (size_t i = 0; i < NK; i++) {
        if (isnan(y[i])) { bad++; continue; }
        double e = fabs(y[i] - (double)N * x[i]);
        if (e > maxerr) maxerr = e;
    }
    char fs[64] = ""; int o = 0;
    for (int i = 0; i < nf; i++)
        o += snprintf(fs + o, sizeof fs - (size_t)o, "%s%d", i ? "," : "", factors[i]);
    int ok = (bad == 0 && maxerr < 1e-10);
    printf("N=%-4d (%s)%*s K=%-3zu  maxerr=%.3e  nan=%-3zu %s\n",
           N, fs, (int)(12 - strlen(fs)), "", K, maxerr, bad,
           ok ? "PASS" : "FAIL");
    free(x); free(hc); free(y);
    rfft_plan_destroy(pf); c2r_plan_destroy(pb);
    return !ok;
}

int main(void)
{
    rfft_codelets_t reg; memset(&reg, 0, sizeof reg);
    reg.r2cf[2]=radix2_r2cf_avx512;  reg.r2cb[2]=radix2_r2cb_avx512;
    reg.r2cf[3]=radix3_r2cf_avx512;  reg.r2cb[3]=radix3_r2cb_avx512;
    reg.r2cf[4]=radix4_r2cf_avx512;  reg.r2cb[4]=radix4_r2cb_avx512;
    reg.r2cf[5]=radix5_r2cf_avx512;  reg.r2cb[5]=radix5_r2cb_avx512;
    reg.r2cf[8]=radix8_r2cf_avx512;  reg.r2cb[8]=radix8_r2cb_avx512;
    reg.r2cf[16]=radix16_r2cf_avx512; reg.r2cb[16]=radix16_r2cb_avx512;
    reg.r2cf[32]=radix32_r2cf_avx512; reg.r2cb[32]=radix32_r2cb_avx512;
    reg.hc2hc[2]=radix2_hc2hc_dit_fwd_avx512; reg.hc2hc_dif_bwd[2]=radix2_hc2hc_dif_bwd_avx512;
    reg.hc2hc[4]=radix4_hc2hc_dit_fwd_avx512; reg.hc2hc_dif_bwd[4]=radix4_hc2hc_dif_bwd_avx512;
    reg.hc2hc[5]=radix5_hc2hc_dit_fwd_avx512; reg.hc2hc_dif_bwd[5]=radix5_hc2hc_dif_bwd_avx512;
    reg.hc2hc[8]=radix8_hc2hc_dit_fwd_avx512; reg.hc2hc_dif_bwd[8]=radix8_hc2hc_dif_bwd_avx512;

    int fail = 0;
    { int f[]={16};        fail += run_case(&reg, 16,  8, f, 1); }  /* nf=1 regression */
    { int f[]={2,8};       fail += run_case(&reg, 16,  8, f, 2); }  /* mid, the old blocker */
    { int f[]={2,16};      fail += run_case(&reg, 32,  8, f, 2); }  /* kmax=7 + mid */
    { int f[]={8,3};       fail += run_case(&reg, 24,  8, f, 2); }  /* m=3 odd: NO mid, r=8 stage */
    { int f[]={5,16};      fail += run_case(&reg, 80,  8, f, 2); }  /* odd radix stage + mid */
    { int f[]={2,2,16};    fail += run_case(&reg, 64,  8, f, 3); }  /* nf=3, Q=2 fold */
    { int f[]={2,2,2,8};   fail += run_case(&reg, 64,  8, f, 4); }  /* nf=4, ping-pong A,B,A */
    { int f[]={4,4,16};    fail += run_case(&reg, 256, 8, f, 3); }  /* the bench plan */
    { int f[]={8,32};      fail += run_case(&reg, 256, 8, f, 2); }  /* the MKL-beating fwd plan */
    { int f[]={8,32};      fail += run_case(&reg, 256, 64, f, 2); } /* wider K */
    printf(fail ? "== %d FAIL ==\n" : "== ALL PASS ==\n", fail);
    return fail;
}
