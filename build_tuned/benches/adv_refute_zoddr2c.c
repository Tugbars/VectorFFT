/* adv_refute_zoddr2c.c -- ADVERSARIAL PROBE (not a gate).
 * Q: does R2C, layout=INTERLEAVED (CCE), odd N, K=1, OOP, front door
 *    (vfft.h) compute correctly?  Claim under test: vfft_r2c_execute_fwd_z
 *    (r2c_dispatch.h:475) STRIDE branch calls the EVEN executor
 *    _r2c_execute_fwd_oop unconditionally, so an odd-N plan built by
 *    _r2c_plan_odd (r2c.h:1358) runs with tw_re/tw_im never initialised.
 * Control arms: even N (works), odd N covered by rfft factors {2,3,4,5,7,8,16}
 *    (should route RFFT, not STRIDE), and the SPLIT-layout sibling at the
 *    same odd N (goes through stride_execute_r2c's override_fwd guard).
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static void naive(const double *x, int N, double *Xr, double *Xi)
{
    for (int f = 0; f <= N / 2; f++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * (double)f * (double)n / (double)N;
            sr += x[n] * cos(a); si += x[n] * sin(a);
        }
        Xr[f] = sr; Xi[f] = si;
    }
}

static double run_cell(int N, int interleaved, int *built)
{
    int H = N / 2 + 1;
    double *x  = (double *)calloc((size_t)N + 8, sizeof(double));
    double *z  = (double *)calloc((size_t)2 * H + 8, sizeof(double));
    double *sr = (double *)calloc((size_t)H + 8, sizeof(double));
    double *si = (double *)calloc((size_t)H + 8, sizeof(double));
    double *Rr = (double *)calloc((size_t)H + 8, sizeof(double));
    double *Ri = (double *)calloc((size_t)H + 8, sizeof(double));
    for (int i = 0; i < N; i++) x[i] = sin(0.3 * i) + 0.25 * cos(1.7 * i);
    naive(x, N, Rr, Ri);

    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor     = VFFT_MEASURE;
    cfg.dims      = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout    = interleaved ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
    vfft_plan p = vfft_create(&cfg);
    *built = (p != NULL);
    if (!p) { free(x);free(z);free(sr);free(si);free(Rr);free(Ri); return -1.0; }

    if (interleaved) vfft_execute(p, VFFT_FORWARD, x, NULL, z, NULL);
    else             vfft_execute(p, VFFT_FORWARD, x, NULL, sr, si);

    double num = 0, den = 0;
    for (int f = 0; f < H; f++) {
        double gr = interleaved ? z[2 * f]     : sr[f];
        double gi = interleaved ? z[2 * f + 1] : si[f];
        double dr = gr - Rr[f], di = gi - Ri[f];
        num += dr * dr + di * di;
        den += Rr[f] * Rr[f] + Ri[f] * Ri[f];
    }
    double rel = sqrt(num / (den > 0 ? den : 1.0));
    vfft_destroy(p);
    free(x);free(z);free(sr);free(si);free(Rr);free(Ri);
    return rel;
}

int main(int argc, char **argv)
{
    static const int cells[] = { 8, 12, 16, 3, 5, 7, 9, 15, 21, 25, 27, 35, 45, 49,
                                 11, 13, 17, 19, 23, 29, 33, 39, 51, 55, 65, 77, 91, 97, 121, 143 };
    int only = (argc > 1) ? atoi(argv[1]) : 0;
    int arm  = (argc > 2) ? atoi(argv[2]) : -1; /* -1 both, 1 IL only, 0 SPLIT only */
    printf("%5s %4s   %-14s %-14s\n", "N", "par", "IL(CCE) rel", "SPLIT rel");
    for (unsigned i = 0; i < sizeof cells / sizeof cells[0]; i++) {
        int N = cells[i];
        if (only && N != only) continue;
        int b1 = 1, b2 = 1;
        double ril = -2.0, rsp = -2.0;
        if (arm != 0) { ril = run_cell(N, 1, &b1); }
        if (arm != 1) { rsp = run_cell(N, 0, &b2); }
        printf("%5d %4s   ", N, (N & 1) ? "odd" : "even");
        if (!b1) printf("%-14s ", "CREATE-NULL"); else printf("%-14.3e ", ril);
        if (!b2) printf("%-14s\n", "CREATE-NULL"); else printf("%-14.3e\n", rsp);
        fflush(stdout);
    }
    return 0;
}
