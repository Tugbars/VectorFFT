/* test_oop_plan.c — gates for core/oop_plan.h.
 *
 * Per cell: checks the rule predicates picked the expected kind, then
 * order-appropriate correctness gates:
 *   LEAF / BAILEY2 (natural order): fwd vs FFTW FORWARD, bwd vs FFTW
 *     BACKWARD (unnormalized), relerr <= 1e-9, src preserved.
 *   MODEB (scrambled order): bit-exact vs vfft_proto_execute_fwd_generic
 *     in-place dataflow, src preserved.
 * Also checks K%8 rejection and the aliasing mask's K-dependence
 * (N=1024: K=128 masked -> MODEB; K=120 unmasked -> BAILEY2).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fftw3.h"
#include "../core/executor.h"
#include "../core/oop_plan.h"

static const char *kname(vfft_oop_kind_t k)
{
    return k == VFFT_OOP_KIND_LEAF ? "LEAF" :
           k == VFFT_OOP_KIND_BAILEY2 ? "BAILEY2" : "MODEB";
}

/* natural-order gate vs FFTW (fwd: sign=-1 plan FORWARD; bwd: BACKWARD) */
static double nat_gate(const vfft_oop_plan_t *p, int bwd,
                       const double *sr, const double *si,
                       double *dr, double *di,
                       const double *xr, const double *xi)
{
    const int N = p->N;
    const size_t K = p->K;
    int rc = bwd ? vfft_oop_execute_bwd(p, sr, si, dr, di)
                 : vfft_oop_execute_fwd(p, sr, si, dr, di);
    if (rc)
        return 1.0;
    fftw_complex *fi = fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *fo = fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan pl = fftw_plan_dft_1d(N, fi, fo,
                                    bwd ? FFTW_BACKWARD : FFTW_FORWARD,
                                    FFTW_ESTIMATE);
    double mr = 0;
    for (size_t t = 0; t < K; t += (K > 3 ? K / 3 : 1))
    {
        for (int e = 0; e < N; e++)
        {
            fi[e][0] = xr[t * (size_t)N + e];
            fi[e][1] = xi[t * (size_t)N + e];
        }
        fftw_execute(pl);
        double me = 0, mm = 0;
        for (int k = 0; k < N; k++)
        {
            double a = dr[(size_t)k * K + t] - fo[k][0];
            double b = di[(size_t)k * K + t] - fo[k][1];
            double e2 = sqrt(a * a + b * b), m = hypot(fo[k][0], fo[k][1]);
            if (e2 > me) me = e2;
            if (m > mm) mm = m;
        }
        if (mm > 0 && me / mm > mr) mr = me / mm;
    }
    fftw_destroy_plan(pl); fftw_free(fi); fftw_free(fo);
    return mr;
}

static int run_cell(int N, size_t K, const int *factors, int nf,
                    vfft_proto_registry_t *reg, vfft_oop_kind_t expect)
{
    vfft_oop_plan_t *p = vfft_oop_plan_create(N, K, factors, nf, reg);
    if (!p)
    {
        printf("  N=%-5d K=%-4zu plan NULL (expected %s)\n",
               N, K, kname(expect));
        return 1;
    }
    int kok = (p->kind == expect);
    size_t T = (size_t)N * K;
    double *sr = aligned_alloc(64, T * 8), *si = aligned_alloc(64, T * 8);
    double *s0r = malloc(T * 8), *s0i = malloc(T * 8);
    double *dr = aligned_alloc(64, T * 8), *di = aligned_alloc(64, T * 8);
    double *xr = malloc(T * 8), *xi = malloc(T * 8);
    srand(17 + N);
    for (size_t t = 0; t < K; t++)
        for (int e = 0; e < N; e++)
        {
            double vr = (double)rand() / RAND_MAX - 0.5;
            double vi = (double)rand() / RAND_MAX - 0.5;
            xr[t * (size_t)N + e] = vr; xi[t * (size_t)N + e] = vi;
            sr[(size_t)e * K + t] = vr; si[(size_t)e * K + t] = vi;
        }
    memcpy(s0r, sr, T * 8); memcpy(s0i, si, T * 8);

    int ok;
    if (p->kind == VFFT_OOP_KIND_MODEB)
    {
        double *rr = malloc(T * 8), *ri = malloc(T * 8);
        memcpy(rr, sr, T * 8); memcpy(ri, si, T * 8);
        vfft_proto_execute_fwd_generic(p->mb, rr, ri, K);
        memset(dr, 0xCD, T * 8); memset(di, 0xCD, T * 8);
        int rc = vfft_oop_execute_fwd(p, sr, si, dr, di);
        int g1 = rc == 0 && !memcmp(dr, rr, T * 8) && !memcmp(di, ri, T * 8);
        memcpy(rr, si, T * 8); memcpy(ri, sr, T * 8);
        vfft_proto_execute_fwd_generic(p->mb, rr, ri, K);
        int rcb = vfft_oop_execute_bwd(p, sr, si, dr, di);
        int g2 = rcb == 0 && !memcmp(di, rr, T * 8) && !memcmp(dr, ri, T * 8);
        int gp = !memcmp(sr, s0r, T * 8) && !memcmp(si, s0i, T * 8);
        printf("  N=%-5d K=%-4zu kind %-7s %s | fwd %s bwd %s preserve %s\n",
               N, K, kname(p->kind), kok ? "OK" : "WRONG",
               g1 ? "BITEXACT" : "FAIL", g2 ? "BITEXACT" : "FAIL",
               gp ? "OK" : "FAIL");
        ok = kok && g1 && g2 && gp;
        free(rr); free(ri);
    }
    else
    {
        double rf = nat_gate(p, 0, sr, si, dr, di, xr, xi);
        double rb = nat_gate(p, 1, sr, si, dr, di, xr, xi);
        int gp = !memcmp(sr, s0r, T * 8) && !memcmp(si, s0i, T * 8);
        printf("  N=%-5d K=%-4zu kind %-7s %s | fwd %.1e %s bwd %.1e %s preserve %s\n",
               N, K, kname(p->kind), kok ? "OK" : "WRONG",
               rf, rf < 1e-9 ? "PASS" : "FAIL",
               rb, rb < 1e-9 ? "PASS" : "FAIL", gp ? "OK" : "FAIL");
        ok = kok && rf < 1e-9 && rb < 1e-9 && gp;
    }
    free(sr); free(si); free(s0r); free(s0i);
    free(dr); free(di); free(xr); free(xi);
    vfft_oop_plan_destroy(p);
    return !ok;
}

int main(void)
{
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    int bad = 0;

    /* K%8 rejection */
    bad += (vfft_oop_plan_create(64, 12, NULL, 0, &reg) != NULL);
    printf("  K%%8 rejection %s\n", bad ? "FAIL" : "OK");

    bad += run_cell(64, 128, NULL, 0, &reg, VFFT_OOP_KIND_LEAF);
    bad += run_cell(13, 64, NULL, 0, &reg, VFFT_OOP_KIND_LEAF);
    bad += run_cell(169, 256, NULL, 0, &reg, VFFT_OOP_KIND_BAILEY2);
    bad += run_cell(169, 128, NULL, 0, &reg, VFFT_OOP_KIND_BAILEY2);
    { int f[] = {8, 8, 16};   /* (8,128) unmasked at the 32KB boundary */
      bad += run_cell(1024, 128, f, 3, &reg, VFFT_OOP_KIND_BAILEY2); }
    bad += run_cell(1024, 120, NULL, 0, &reg, VFFT_OOP_KIND_BAILEY2);
    /* odd mixed-radix Mode B: runs on BOTH ISAs since the full avx2
     * in-place set landed (codelets/inplace_avx2/) */
    { int f[] = {2, 3, 5, 7, 11};
      bad += run_cell(2310, 32, f, 5, &reg, VFFT_OOP_KIND_MODEB); }
    { int f[] = {8, 16, 16};
      bad += run_cell(2048, 256, f, 3, &reg, VFFT_OOP_KIND_MODEB); }
    { int f[] = {8, 8, 8};    /* (8,64) unmasked at the 32KB boundary */
      bad += run_cell(512, 256, f, 3, &reg, VFFT_OOP_KIND_BAILEY2); }

    printf(bad ? "SOME GATES FAILED\n" : "ALL GATES PASS\n");
    return bad;
}
