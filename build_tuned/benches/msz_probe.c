/* THE msz A/B (2026-09-05): MKL's Fact form on our contract (zsplit odd mid
 * body, interleaved z on both edges, unordered lanes) vs the t2cp / tail
 * forms, PER STAGE on ONE flat DIT plan (il_flatdit.h builds both record
 * sets; p->msz[s] picks the form). Cells: fully-odd N times 4, so the
 * non-last stages have runs D % 4 == 0 (the msz contract). Forward only:
 * correctness with msz ON (naive DFT at natural bins + DC), then per-stage
 * min-of-7 alternated, then the whole transform (all-t2cp vs all-msz vs
 * MKL) min-of-9 alternated. Build:
 *   python build_tuned/build.py --compile --src build_tuned/benches/msz_probe.c --mkl --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "vfft.h"
#include "../../src/core/oop/il_flatdit.h"
static double now_ns(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart*1e9/(double)f.QuadPart;}
static void chain_s(const int *R, int K, char *cs, size_t n) { int off = 0; for (int s = 0; s < K; s++) off += snprintf(cs + off, n - off, "%s%d", s ? "." : "", R[s]); }
static void set_all(vfft_ilfd_plan_t *p, int on) { for (int s = 1; s < p->K; s++) if (p->tz[s]) p->msz[s] = on; }
int main(int argc, char **argv) {
    static const int NS[] = { 405, 972, 1215, 1372, 2500, 3125, 4095, 6561, 9604, 12500, 15625, 16807,
                              19683, 26244, 59049, 62500, 67228, 78125, 78732, 98415, 117649, 137781 };
    const int n = (int)(sizeof NS / sizeof NS[0]);
    int bad = 0;
    (void)argc; (void)argv;
    mkl_set_num_threads(1);
    for (int i = 0; i < n; i++) {
        const int N = NS[i];
        vfft_ilfd_plan_t *p = vfft_ilfd_create(N);
        char cs[48] = "-";
        if (!p) { printf("%-7d NO CHAIN\n", N); bad++; continue; }
        chain_s(p->R, p->K, cs, sizeof cs);
        double *x = malloc(2*(size_t)N*8), *z = malloc(2*(size_t)N*8), *y = malloc(2*(size_t)N*8);
        double s0r = 0, s0i = 0;
        for (int j = 0; j < N; j++) { x[2*j] = (double)rand()/RAND_MAX-0.5; x[2*j+1] = (double)rand()/RAND_MAX-0.5; s0r += x[2*j]; s0i += x[2*j+1]; }
        /* correctness, msz ON everywhere eligible */
        set_all(p, 1);
        vfft_ilfd_execute_fwd(p, x, z);
        double dc = fabs(z[0]-s0r) + fabs(z[1]-s0i), wn = 0;
        for (int t = 0; t < 6; t++) {
            const int k = (t*997+3) % N; double er = 0, ei = 0;
            for (int a = 0; a < N; a++) {
                double an = -2.0*3.14159265358979323846*(double)((long)k*a % N)/N;
                er += x[2*a]*cos(an) - x[2*a+1]*sin(an);
                ei += x[2*a]*sin(an) + x[2*a+1]*cos(an);
            }
            double d = fabs(z[2*k]-er) + fabs(z[2*k+1]-ei);
            if (d > wn) wn = d;
        }
        const int ok = (wn < 1e-9 * sqrt((double)N) && dc < 1e-9 * N);
        if (!ok) bad++;
        printf("%-7d %-14s | dft@nat %.1e dc %.1e %s", N, cs, wn, dc, ok ? "OK " : "BAD");
        {   /* the inverse: bwd(fwd(x)) / N == x */
            double rt = 0;
            if (p->bwd_ok) {
                vfft_ilfd_execute_bwd(p, z, y);
                for (int j = 0; j < 2 * N; j++) { double d = fabs(y[j] / N - x[j]); if (d > rt) rt = d; }
            }
            const int okb = p->bwd_ok && (rt < 1e-9 * sqrt((double)N));
            if (!okb) bad++;
            printf(" | roundtrip %.1e %s\n", rt, !p->bwd_ok ? "NO BWD" : okb ? "OK " : "BAD");
        }
        /* the leaf (stage 0: n1c, no A/B) so the whole-transform split is complete */
        {
            double tl = 1e300;
            for (int r = 0; r < 7; r++) { double t0 = now_ns(); vfft_ilfd_stage(p, 0, x, z); t0 = now_ns()-t0; if (t0 < tl) tl = t0; }
            printf("      R%-2d cnt%-6zu leaf   n1c    %7.0f ns  (%.2f ns/pt)\n", p->R[0], p->D[0], tl, tl / N);
        }
        /* per-stage A/B on the eligible stages */
        for (int s = 1; s < p->K; s++) {
            double ta = 1e300, tb = 1e300;
            if (!p->tz[s] && p->fgl[s]) {
                /* the last stage: t2csg (one call per group) vs t2csgn in block
                 * order vs t2csgn in natural-base order — three arms, rotated */
                double tc = 1e300;
                for (int r = 0; r < 9; r++) {
                    for (int a2 = 0; a2 < 3; a2++) {
                        const int arm = (a2 + r) % 3;
                        double t0;
                        p->gl[s] = (arm != 0); p->gord = (arm == 2);
                        t0 = now_ns(); vfft_ilfd_stage(p, s, x, z); t0 = now_ns()-t0;
                        if (arm == 0) { if (t0 < ta) ta = t0; } else if (arm == 1) { if (t0 < tb) tb = t0; } else { if (t0 < tc) tc = t0; }
                    }
                }
                p->gl[s] = 1; p->gord = 1;
                printf("      R%-2d cnt%-6zu x%-6zu t2csg %7.0f | t2csgn %7.0f | t2csgn+order %7.0f ns | %.2fx %.2fx  (%.2f -> %.2f -> %.2f ns/pt)\n", p->R[s], p->D[s], p->nblk[s],
                       ta, tb, tc, ta / tb, ta / tc, ta / N, tb / N, tc / N);
                continue;
            }
            if (!p->tz[s]) {
                p->msz[s] = 0;
                for (int r = 0; r < 7; r++) { double t0 = now_ns(); vfft_ilfd_stage(p, s, x, z); t0 = now_ns()-t0; if (t0 < ta) ta = t0; }
                printf("      R%-2d cnt%-6zu x%-6zu %-6s %7.0f ns  (not msz-eligible; %.2f ns/pt, %.1f ns/call over %zu calls)\n", p->R[s], p->D[s], p->nblk[s],
                       p->tail[s] == 2 ? "t2csg" : p->tail[s] == 1 ? "t2cs" : "t2cp", ta, ta / N,
                       ta / ((double)p->nblk[s] / p->R[s - 1] * p->D[s]), (size_t)(p->nblk[s] / p->R[s - 1] * p->D[s]));
                continue;
            }
            for (int r = 0; r < 7; r++) {
                double t0;
                p->msz[s] = (r & 1);
                t0 = now_ns(); vfft_ilfd_stage(p, s, x, z); t0 = now_ns()-t0; if (p->msz[s]) { if (t0 < tb) tb = t0; } else { if (t0 < ta) ta = t0; }
                p->msz[s] = !(r & 1);
                t0 = now_ns(); vfft_ilfd_stage(p, s, x, z); t0 = now_ns()-t0; if (p->msz[s]) { if (t0 < tb) tb = t0; } else { if (t0 < ta) ta = t0; }
            }
            printf("      R%-2d cnt%-6zu x%-6zu %-6s %7.0f ns | msz %7.0f ns | %5.2fx  (%.2f -> %.2f ns/pt)\n", p->R[s], p->D[s], p->nblk[s],
                   p->tail[s] == 2 ? "t2csg" : p->tail[s] == 1 ? "t2cs" : "t2cp", ta, tb, ta / tb, ta / N, tb / N);
        }
        /* whole transform: all-t2cp (A) vs all-msz (B) vs MKL, alternated */
        {
            double ta = 1e300, tb = 1e300, tm = 1e300;
            DFTI_DESCRIPTOR_HANDLE hm = NULL;
            if (DftiCreateDescriptor(&hm, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) == 0 &&
                DftiSetValue(hm, DFTI_PLACEMENT, DFTI_NOT_INPLACE) == 0 && DftiCommitDescriptor(hm) == 0) {
                DftiComputeForward(hm, x, y);
                for (int r = 0; r < 9; r++) {
                    double t0;
                    set_all(p, 0); t0 = now_ns(); vfft_ilfd_execute_fwd(p, x, z); t0 = now_ns()-t0; if (t0 < ta) ta = t0;
                    set_all(p, 1); t0 = now_ns(); vfft_ilfd_execute_fwd(p, x, z); t0 = now_ns()-t0; if (t0 < tb) tb = t0;
                    t0 = now_ns(); DftiComputeForward(hm, x, y); t0 = now_ns()-t0; if (t0 < tm) tm = t0;
                }
                printf("      whole: t2cp %7.0f | msz %7.0f | mkl %7.0f | msz/t2cp %5.2fx | mkl/msz %5.2fx\n", ta, tb, tm, ta / tb, tm / tb);
                if (p->bwd_ok) {   /* the inverse vs MKL's, same protocol */
                    double tbw = 1e300, tmb = 1e300;
                    for (int r = 0; r < 9; r++) {
                        double t0;
                        t0 = now_ns(); vfft_ilfd_execute_bwd(p, z, y); t0 = now_ns()-t0; if (t0 < tbw) tbw = t0;
                        t0 = now_ns(); DftiComputeBackward(hm, z, y); t0 = now_ns()-t0; if (t0 < tmb) tmb = t0;
                    }
                    printf("      bwd:   flat %7.0f | mkl %7.0f | mkl/flat %5.2fx\n", tbw, tmb, tmb / tbw);
                }
                DftiFreeDescriptor(&hm);
            }
        }
        fflush(stdout);
        vfft_ilfd_destroy(p); free(x); free(z); free(y);
    }
    printf(bad ? "=== *** %d BAD *** ===\n" : "=== ALL OK (%d cells) ===\n", bad ? bad : n);
    return bad ? 1 : 0;
}
