/* test_r2c_oddk.c — arbitrary-K (odd batch) correctness for the r2c/c2r rfft cascade.
 *
 * After the rem-aware tail landed in the rfft/c2r codelets and the front-door guards
 * were selectively relaxed (rfft/natural opened to any K; the decoupled-stride path kept
 * K%8-gated so odd K is forced onto the rfft cascade), this validates END-TO-END at odd K:
 *
 *   1. ROUTING       — at odd K the r2c plan must pick path=RFFT and the c2r disp must pick
 *                      layout=NATURAL (never the transpose/stride path).
 *   2. ROUNDTRIP     — c2r(r2c(x)) == scale*x  (scale fit by least-squares; convention-free).
 *                      This is the decisive check: exercises BOTH r2c and c2r tails together,
 *                      including the hc2c_nat split terminator/initiator.
 *   3. SELF-CONSIST  — r2c at odd K vs r2c at the next 8-multiple Kp with the SAME first-K
 *                      input columns: every spectrum bin of column b<K must match (isolates
 *                      the r2c tail; masked lanes bit-ish-exact, scalar rem==1 ~1 ULP).
 *
 * Split, lane-batched layout: x[n*K + v]  ->  re[k*K + v], im[k*K + v]  (v in 0..K-1).
 */
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
#include "registry.h"   /* vfft_proto_registry_init (c2c inner; unused on the rfft path) */

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static rfft_codelets_t  RFFT;
static vfft_proto_registry_t C2C;

/* rfft/c2r codelets use unaligned loads (loadu/maskload), so plain malloc is fine;
 * mingw lacks C11 aligned_alloc anyway. */
#define AAL(nbytes) malloc((size_t)(nbytes))

/* least-squares scale s minimizing ||y - s*x||, then max rel err of y vs s*x. */
static double rt_relerr(const double *y, const double *x, size_t n) {
    double num = 0, den = 0, xmax = 0;
    for (size_t i = 0; i < n; i++) { num += y[i]*x[i]; den += x[i]*x[i]; if (fabs(x[i])>xmax) xmax=fabs(x[i]); }
    double s = den > 0 ? num/den : 0;
    double e = 0;
    for (size_t i = 0; i < n; i++) { double d = fabs(y[i] - s*x[i]); if (d > e) e = d; }
    return (s != 0 && xmax > 0) ? e / (fabs(s)*xmax) : e;
}

/* one cell: returns 0 on PASS, nonzero on FAIL; prints a row. */
static int cell(int N, size_t K) {
    int H = N/2;                          /* spectrum bins stored: 0..H-1 (gate convention) */
    size_t pad = 32;
    double *x   = AAL(((size_t)N*K + pad)*8);
    double *re  = AAL(((size_t)(H+1)*K + pad)*8);
    double *im  = AAL(((size_t)(H+1)*K + pad)*8);
    double *xb  = AAL(((size_t)N*K + pad)*8);     /* roundtrip output */
    srand(100 + N + (int)K);
    for (size_t i = 0; i < (size_t)N*K; i++) x[i] = (double)rand()/RAND_MAX - 0.5;
    memset(re, 0, ((size_t)(H+1)*K + pad)*8);
    memset(im, 0, ((size_t)(H+1)*K + pad)*8);

    /* --- r2c forward --- */
    vfft_r2c_plan_t *rp = vfft_r2c_plan_create(N, K, VFFT_R2C_SPLIT, &RFFT, NULL, &C2C);
    if (!rp) { printf("  N=%-4d K=%-3zu  r2c plan NULL  FAIL\n", N, K); return 1; }
    int r2c_rfft = (rp->path == VFFT_R2C_PATH_RFFT);
    vfft_r2c_execute_fwd(rp, x, re, im);

    /* --- c2r backward (roundtrip) --- */
    vfft_c2r_disp_t *cp = vfft_c2r_disp_create_auto(N, K, &RFFT, &C2C);
    if (!cp) { printf("  N=%-4d K=%-3zu  c2r disp NULL  FAIL\n", N, K); vfft_r2c_plan_destroy(rp); return 1; }
    int c2r_nat = (cp->layout == VFFT_C2R_NATURAL);
    vfft_c2r_disp_execute(cp, re, im, xb);
    double rt = rt_relerr(xb, x, (size_t)N*K);

    /* --- routing checks (odd K must avoid the stride/transpose path) --- */
    int odd = (K % 8) != 0;
    int route_ok = !odd || (r2c_rfft && c2r_nat);

    int fail = (rt > 1e-9) || !route_ok;
    printf("  N=%-4d K=%-3zu  r2c=%-6s c2r=%-7s  roundtrip=%.2e %s%s\n",
           N, K, r2c_rfft?"RFFT":"STRIDE", c2r_nat?"NATURAL":"stride/pk",
           rt, fail?"FAIL":"ok", (odd&&!route_ok)?"  <ROUTE!>":"");

    vfft_r2c_plan_destroy(rp);
    vfft_c2r_disp_destroy(cp);
    free(x); free(re); free(im); free(xb);
    return fail;
}

/* r2c self-consistency: odd K vs padded Kp=roundup(K,8), per spectrum bin per column. */
static int selfconsist(int N, size_t K) {
    int H = N/2; size_t Kp = ((K + 7)/8)*8, pad = 32;
    double *xk  = AAL(((size_t)N*K  + pad)*8), *xp = AAL(((size_t)N*Kp + pad)*8);
    double *rek = AAL(((size_t)(H+1)*K  + pad)*8), *imk = AAL(((size_t)(H+1)*K  + pad)*8);
    double *rep = AAL(((size_t)(H+1)*Kp + pad)*8), *imp = AAL(((size_t)(H+1)*Kp + pad)*8);
    srand(777 + N + (int)K);
    for (int n=0;n<N;n++) for (size_t v=0;v<Kp;v++) { double val=(double)rand()/RAND_MAX-0.5; xp[(size_t)n*Kp+v]=val; if(v<K) xk[(size_t)n*K+v]=val; }
    memset(rek,0,((size_t)(H+1)*K+pad)*8); memset(imk,0,((size_t)(H+1)*K+pad)*8);
    memset(rep,0,((size_t)(H+1)*Kp+pad)*8); memset(imp,0,((size_t)(H+1)*Kp+pad)*8);
    vfft_r2c_plan_t *pk = vfft_r2c_plan_create(N, K,  VFFT_R2C_SPLIT, &RFFT, NULL, &C2C);
    vfft_r2c_plan_t *pp = vfft_r2c_plan_create(N, Kp, VFFT_R2C_SPLIT, &RFFT, NULL, &C2C);
    if (!pk || !pp) { printf("  N=%-4d K=%-3zu  selfconsist plan NULL FAIL\n", N, K); return 1; }
    vfft_r2c_execute_fwd(pk, xk, rek, imk);
    vfft_r2c_execute_fwd(pp, xp, rep, imp);
    double e = 0;
    for (int k=0;k<H;k++) for (size_t v=0;v<K;v++) {
        double a = fabs(rek[(size_t)k*K+v]-rep[(size_t)k*Kp+v]) + fabs(imk[(size_t)k*K+v]-imp[(size_t)k*Kp+v]);
        if (a>e) e=a;
    }
    int fail = e > 1e-11;
    printf("  N=%-4d K=%-3zu  selfconsist(vs Kp=%zu) max err=%.2e %s\n", N, K, Kp, e, fail?"FAIL":"ok");
    vfft_r2c_plan_destroy(pk); vfft_r2c_plan_destroy(pp);
    free(xk);free(xp);free(rek);free(imk);free(rep);free(imp);
    return fail;
}

int main(void) {
    memset(&RFFT, 0, sizeof RFFT);
    rfft_register_all_avx2(&RFFT);   /* forward slots: r2cf, hc2hc, hc2c_nat */
    c2r_register_all_avx2(&RFFT);    /* backward slots: r2cb, hc2hc_dif_bwd, hc2c_nat_bwd */
    vfft_proto_registry_init(&C2C);

    const int  Ns[] = {64, 128, 256};
    const size_t Ks[] = {8, 16, 5, 7, 13, 15, 17, 31, 33};  /* 8/16 even baseline; rest odd */
    int fails = 0;

    printf("# r2c/c2r arbitrary-K end-to-end (roundtrip c2r(r2c(x))==scale*x; route: odd K must be RFFT/NATURAL)\n\n");
    printf("== roundtrip + routing ==\n");
    for (size_t ni=0; ni<sizeof(Ns)/sizeof(Ns[0]); ni++)
        for (size_t ki=0; ki<sizeof(Ks)/sizeof(Ks[0]); ki++)
            fails += cell(Ns[ni], Ks[ki]);

    printf("\n== r2c self-consistency (odd K vs padded even Kp) ==\n");
    for (size_t ni=0; ni<sizeof(Ns)/sizeof(Ns[0]); ni++)
        for (size_t ki=2; ki<sizeof(Ks)/sizeof(Ks[0]); ki++)   /* skip the two even baselines */
            fails += selfconsist(Ns[ni], Ks[ki]);

    printf("\n# %s (%d failing checks)\n", fails?"FAIL":"PASS", fails);
    return fails ? 1 : 0;
}
