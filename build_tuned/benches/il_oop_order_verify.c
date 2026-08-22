/* il_oop_order_verify.c -- ADVERSARIAL VERIFICATION PROBE (not a gate).
 *
 * Q1: at a ragged-count N, is the SHIPPED OOP interleaved front-door plan
 *     bitwise equal to il2p(N, heuristic R1, heuristic R2) and DIFFERENT from
 *     the swapped il2p(N, R2, R1)?  (i.e. is the unraced ordering live?)
 * Q2: is the swapped ordering LEGAL (create returns non-NULL)?
 * Q3: does _il_dp_enumerate produce ANY 2P candidate for that N?  (i.e. can
 *     wisdom ever bank il_R1/il_R2 there?)
 * Q4: does the IN-PLACE front door produce the SAME bytes as OOP, or does its
 *     create-time ordering race pick the other one?
 *
 * Build: python build.py --src benches/il_oop_order_verify.c --compile
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "il2p.h"
#include "vfft.h"
#include "dp_planner_il.h"
#include <windows.h>
static double now_ns(void){LARGE_INTEGER f,c;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&c);return (double)c.QuadPart*1e9/(double)f.QuadPart;}

static int fd_pair(int N, int *oR1, int *oR2)   /* verbatim vfft.c:4944-4966 */
{
    int iR1 = 0, iR2 = 0;
    for (int R2c = (N < 64 ? N : 64); R2c >= 4; R2c--) {
        if (N % R2c) continue;
        int R1c = N / R2c;
        if (R1c < 3 || R1c > 64) continue;
        if (!vfft_il2p_leaf_fn(R2c, 0) || !vfft_il2p_mid_fn(R1c, 0)) continue;
        if (!iR1 || abs(R1c - R2c) < abs(iR1 - iR2)) { iR1 = R1c; iR2 = R2c; }
    }
    *oR1 = iR1; *oR2 = iR2; return iR1 != 0;
}

static double bestns(vfft_il2p_plan_t *p, const double *seed, double *w, int N)
{
    const size_t nb = 2 * (size_t)N * sizeof(double);
    const int reps = N <= 256 ? 64 : (N <= 1024 ? 24 : 8);
    double best = 1e30;
    for (int r = 0; r < 7; r++) {
        memcpy(w, seed, nb);
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) vfft_il2p_execute_fwd(p, w, w);
        double d = (now_ns() - t0) / reps;
        if (d < best) best = d;
    }
    return best;
}

int main(void)
{
    static const int NS[] = { 15, 21, 25, 27, 33, 45, 50, 75, 150, 225, 300, 675,
                              128, 512, 1024 };
    static vfft_il_cand_t CAND[VFFT_IL_DP_MAX_CAND];

    printf("%-6s %-8s %-8s %-7s %-9s %-9s %-9s %-8s %-8s\n",
           "N", "heur", "swap", "swapOK", "fd==heur", "fd==swap", "ip==heur",
           "dp2Pcnt", "ns h/s");
    for (unsigned i = 0; i < sizeof NS / sizeof NS[0]; i++) {
        int N = NS[i], R1 = 0, R2 = 0;
        if (!fd_pair(N, &R1, &R2)) { printf("%-6d no-pair\n", N); continue; }

        /* Q3: DP candidates naming route 2P_PURE for this N */
        vfft_il_cand_sink_t sk; memset(&sk, 0, sizeof sk);
        sk.out = CAND;
        _il_dp_enumerate(N, VFFT_IL_ORD_NATURAL, &sk);
        int n2p = 0;
        for (int c = 0; c < sk.n; c++)
            if (sk.out[c].route == VFFT_K1_IL_2P_PURE) n2p++;

        size_t nb = 2 * (size_t)N * sizeof(double);
        double *zi = (double *)malloc(nb), *zfd = (double *)malloc(nb);
        double *zh = (double *)malloc(nb), *zs = (double *)malloc(nb);
        double *zip = (double *)malloc(nb), *zw = (double *)malloc(nb);
        for (int n = 0; n < N; n++) { zi[2*n] = sin(0.7*n)+0.3; zi[2*n+1] = cos(0.31*n)-0.2; }

        /* shipped OOP front door */
        vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
        cfg.placement = VFFT_OUTOFPLACE; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.order = VFFT_ORDER_DEFAULT; cfg.rigor = VFFT_MEASURE;
        memset(zfd, 0, nb);
        vfft_plan h = vfft_create(&cfg);
        if (h) { vfft_execute(h, VFFT_FORWARD, zi, NULL, zfd, NULL); vfft_destroy(h); }

        /* shipped IN-PLACE front door */
        memset(zip, 0, nb); memcpy(zip, zi, nb);
        cfg.placement = VFFT_INPLACE;
        vfft_plan h2 = vfft_create(&cfg);
        if (h2) { vfft_execute(h2, VFFT_FORWARD, zip, NULL, zip, NULL); vfft_destroy(h2); }

        vfft_il2p_plan_t *ph = vfft_il2p_create(N, R1, R2);
        vfft_il2p_plan_t *ps = (R1 != R2) ? vfft_il2p_create(N, R2, R1) : NULL;
        memset(zh, 0, nb); memset(zs, 0, nb);
        if (ph) vfft_il2p_execute_fwd(ph, zi, zh);
        if (ps) vfft_il2p_execute_fwd(ps, zi, zs);

        double nh = -1, ns = -1;
        if (ph && ps) { nh = bestns(ph, zi, zw, N); ns = bestns(ps, zi, zw, N); }

        printf("%-6d %-8s %-8s %-7s %-9s %-9s %-9s %-8d ",
               N,
               ({ static char b[16]; snprintf(b,16,"%dx%d",R1,R2); b; }),
               ({ static char b[16]; snprintf(b,16,"%dx%d",R2,R1); b; }),
               ps ? "yes" : (R1==R2 ? "n/a" : "NO"),
               (h && ph && memcmp(zfd, zh, nb) == 0) ? "BITWISE" : "differs",
               (h && ps && memcmp(zfd, zs, nb) == 0) ? "BITWISE" : "differs",
               (h2 && ph && memcmp(zip, zh, nb) == 0) ? "BITWISE" : "differs",
               n2p);
        if (nh > 0) printf("%.1f/%.1f (%+.1f%%)\n", nh, ns, 100.0*(ns-nh)/nh);
        else printf("-\n");
        free(zi); free(zfd); free(zh); free(zs); free(zip); free(zw);
        if (ph) vfft_il2p_destroy(ph);
        if (ps) vfft_il2p_destroy(ps);
    }
    return 0;
}
