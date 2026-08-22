/* il_ragged_count_probe.c -- RESEARCH PROBE (not a gate, not a bench).
 * Q1: for which N does the FRONT-DOOR heuristic pair search (vfft.c:4952)
 *     hand an ODD count to a cil kernel today?
 * Q2: does the DP planner's enumerator (dp_planner_il.h:1053) ever produce
 *     such a pair?  (its RAD[] is pow2-only)
 * Q3: does the odd-count plan actually COMPUTE correctly through the
 *     public front door (vfft.h), i.e. is the tail reachable-today?
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "il2p.h"
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* EXACT copy of the front-door IL pair search, vfft.c:4952-4962 */
static int fd_pair(int N, int *oR1, int *oR2)
{
    int iR1 = 0, iR2 = 0;
    for (int R2c = (N < 64 ? N : 64); R2c >= 4; R2c--) {
        if (N % R2c) continue;
        int R1c = N / R2c;
        if (R1c < 3 || R1c > 64) continue;
        if (!vfft_il2p_leaf_fn(R2c, 0) || !vfft_il2p_mid_fn(R1c, 0)) continue;
        if (!iR1 || abs(R1c - R2c) < abs(iR1 - iR2)) { iR1 = R1c; iR2 = R2c; }
    }
    *oR1 = iR1; *oR2 = iR2;
    return iR1 != 0;
}

/* EXACT copy of _il_dp_enumerate's NATURAL pair loop, dp_planner_il.h:1053 */
static int dp_pairs(int N)
{
    static const int RAD[] = { 4, 8, 16, 32, 64 };
    int n = 0;
    for (int i = 0; i < 5; i++) {
        int R2 = RAD[i];
        if (N % R2) continue;
        int R1 = N / R2;
        if (R1 < 4 || R1 > 64 || (R1 & (R1 - 1))) continue;
        if (vfft_il2p_leaf_fn(R2, 0) && vfft_il2p_mid_fn(R1, 0)) n++;
    }
    return n;
}

static void naive(const double *z, double *o, int N, int sign)
{
    for (int k = 0; k < N; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = sign * 2.0 * M_PI * (double)k * (double)n / (double)N;
            double c = cos(a), s = sin(a);
            sr += z[2*n] * c - z[2*n+1] * s;
            si += z[2*n] * s + z[2*n+1] * c;
        }
        o[2*k] = sr; o[2*k+1] = si;
    }
}
static double relerr(const double *a, const double *b, int N)
{
    double num = 0, den = 0;
    for (int i = 0; i < 2*N; i++) { double d = a[i]-b[i]; num += d*d; den += b[i]*b[i]; }
    return den > 0 ? sqrt(num/den) : sqrt(num);
}

int main(void)
{
    static const int NS[] = { 15, 18, 45, 50, 75, 150, 192, 225, 96, 300, 675, 128, 512, 1024 };
    printf("%-6s %-10s %-8s %-8s %-6s %-10s %s\n",
           "N", "fd_pair", "leafcnt", "midcnt", "ODD?", "dp_cands", "frontdoor_relerr");
    for (unsigned i = 0; i < sizeof NS/sizeof NS[0]; i++) {
        int N = NS[i], R1 = 0, R2 = 0;
        char pair[24] = "-";
        int odd = 0, ok = fd_pair(N, &R1, &R2);
        if (ok) { snprintf(pair, sizeof pair, "%dx%d", R1, R2); odd = (R1 & 1) || (R2 & 1); }
        /* leaf runs at count=R1, mid at count=R2 (il2p.h:737-739) */
        int ndp = dp_pairs(N);

        /* public front door, OOP interleaved, K=1 */
        double *zi = (double*)calloc(2*(size_t)N, sizeof(double));
        double *zo = (double*)calloc(2*(size_t)N, sizeof(double));
        double *zr = (double*)calloc(2*(size_t)N, sizeof(double));
        for (int n = 0; n < N; n++) { zi[2*n] = sin(0.7*n)+0.3; zi[2*n+1] = cos(0.31*n)-0.2; }
        naive(zi, zr, N, -1);
        vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
        cfg.placement = VFFT_OUTOFPLACE; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.order = VFFT_ORDER_DEFAULT; cfg.rigor = VFFT_MEASURE;
        vfft_plan h = vfft_create(&cfg);
        double e = -1.0;
        if (h) { vfft_execute(h, VFFT_FORWARD, zi, NULL, zo, NULL); e = relerr(zo, zr, N); vfft_destroy(h); }
        printf("%-6d %-10s %-8d %-8d %-6s %-10d %.3e\n",
               N, pair, ok?R1:0, ok?R2:0, odd?"YES":"no", ndp, e);
        free(zi); free(zo); free(zr);
    }
    return 0;
}
