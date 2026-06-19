/* Round-trip gate: random x -> PROVEN forward (rfft_execute_fwd_packed)
 * -> c2r_execute_packed -> expect N*x elementwise. Strictly stronger than
 * gating against a constructed halfcomplex: it exercises the true packed
 * layout the forward actually emits. Oracle chain ends at the forward
 * executor, which is gated against MKL (doc 60). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../core/rfft.h"
#include "../core/c2r.h"

/* forward codelets (DIT) */
extern void radix8_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
extern void radix2_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
extern void radix2_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);
/* backward codelets */
extern void radix8_r2cb_avx512(const double*,const double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
extern void radix2_r2cb_avx512(const double*,const double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
extern void radix2_hc2hc_dif_bwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t);

int main(void)
{
    const int N = 16; const size_t K = 8;
    const int factors[2] = {2, 8}; const int nf = 2;

    rfft_codelets_t reg; memset(&reg, 0, sizeof reg);
    reg.r2cf[8] = radix8_r2cf_avx512;
    reg.r2cf[2] = radix2_r2cf_avx512;
    reg.hc2hc[2] = radix2_hc2hc_dit_fwd_avx512;
    reg.r2cb[8] = radix8_r2cb_avx512;
    reg.r2cb[2] = radix2_r2cb_avx512;
    reg.hc2hc_dif_bwd[2] = radix2_hc2hc_dif_bwd_avx512;

    rfft_plan_t *pf = rfft_plan_create(N, K, factors, nf, &reg);
    c2r_plan_t  *pb = c2r_plan_create(N, K, factors, nf, &reg);
    if (!pf || !pb) { printf("plan FAIL\n"); return 1; }

    size_t NK = (size_t)N * K;
    double *x   = aligned_alloc(64, NK * 8);
    double *hc  = aligned_alloc(64, NK * 8);
    double *y   = aligned_alloc(64, NK * 8);
    srand(7);
    for (size_t i = 0; i < NK; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
    for (size_t i = 0; i < NK; i++) { hc[i] = NAN; y[i] = NAN; } /* poison */

    rfft_execute_fwd_packed(pf, x, hc);
    c2r_execute_packed(pb, hc, y);

    double maxerr = 0.0; size_t bad = 0;
    for (size_t i = 0; i < NK; i++) {
        if (isnan(y[i])) { bad++; continue; }
        double e = fabs(y[i] - (double)N * x[i]);
        if (e > maxerr) maxerr = e;
    }
    printf("c2r roundtrip N=%d (%d,%d) K=%zu: maxerr=%.3e nan=%zu %s\n",
           N, factors[0], factors[1], K, maxerr, bad,
           (bad == 0 && maxerr < 1e-11) ? "PASS" : "FAIL");
    return !(bad == 0 && maxerr < 1e-11);
}
