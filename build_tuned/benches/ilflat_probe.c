/* THE ODD-N FLAT ENGINE, v1 (2026-09-04): correctness (naive DFT at
 * NATURAL indices, fwd->bwd roundtrip, DC) then the CHAIN RACE inside the
 * probe — every leaf radix x every interior chain the 2D enumerator
 * yields, min-of-3 alternated, best kept — and the winner timed vs MKL
 * DFTI, same run. No wisdom, no banking: the engine and its pool alone. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "vfft.h"
#include "../../src/core/oop/il_flat.h"
static double now_ns(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart*1e9/(double)f.QuadPart;}
static double time_fwd(vfft_ilflat_plan_t *p, const double *x, double *z, int reps) {
    double best = 1e300;
    for (int r = 0; r < reps; r++) { double t0 = now_ns(); vfft_ilflat_execute_fwd(p, x, z); t0 = now_ns() - t0; if (t0 < best) best = t0; }
    return best;
}
static void chain_str(const vfft_ilflat_plan_t *p, char *cs, size_t n) {
    int off = snprintf(cs, n, "%d|", p->R0);
    for (int s = 0; s < p->nst; s++) off += snprintf(cs + off, n - off, "%s%d", s ? "." : "", p->R[s]);
}
int main(int argc, char **argv) {
    static const int NS[] = { 405, 1215, 4095, 6561, 19683, 59049, 98415, 137781 };
    const int n = (int)(sizeof NS / sizeof NS[0]);
    const int race = (argc > 1 && !strcmp(argv[1], "--race"));
    int bad = 0;
    mkl_set_num_threads(1);
    printf("%-7s %-14s | dft@nat  rt       dc      | %-14s %9s | %9s %9s %6s\n", "N", "seed", "best chain", "best(ns)", "seed(ns)", "mkl(ns)", "ratio");
    for (int i = 0; i < n; i++) {
        const int N = NS[i];
        vfft_ilflat_plan_t *p = vfft_ilflat_create(N);
        char cs[48] = "-", bs[48] = "-";
        if (!p) { printf("%-7d NO CHAIN\n", N); bad++; continue; }
        chain_str(p, cs, sizeof cs);
        double *x = malloc(2*(size_t)N*8), *z = malloc(2*(size_t)N*8), *y = malloc(2*(size_t)N*8);
        double s0r = 0, s0i = 0;
        for (int j = 0; j < N; j++) { x[2*j] = (double)rand()/RAND_MAX-0.5; x[2*j+1] = (double)rand()/RAND_MAX-0.5; s0r += x[2*j]; s0i += x[2*j+1]; }
        vfft_ilflat_execute_fwd(p, x, z);
        double dc = fabs(z[0]-s0r) + fabs(z[1]-s0i), wn = 0;
        for (int t = 0; t < 5; t++) {
            const int k = (t*997+3) % N; double er = 0, ei = 0;
            for (int a = 0; a < N; a++) {
                double an = -2.0*3.14159265358979323846*(double)((long)k*a % N)/N;
                er += x[2*a]*cos(an) - x[2*a+1]*sin(an);
                ei += x[2*a]*sin(an) + x[2*a+1]*cos(an);
            }
            double d = fabs(z[2*k]-er) + fabs(z[2*k+1]-ei);
            if (d > wn) wn = d;
        }
        vfft_ilflat_execute_bwd(p, z, y);
        double rt = 0;
        for (int j = 0; j < 2*N; j++) { double d = fabs(y[j]/N - x[j]); if (d > rt) rt = d; }
        const int ok = (wn < 1e-9 * sqrt((double)N) && rt < 1e-12 * N && dc < 1e-9 * N);
        if (!ok) bad++;
        /* the chain race: leaf pool x interior enumerator */
        double tseed = time_fwd(p, x, z, 5), tbest = tseed;
        strcpy(bs, cs);
        if (race) {
            static const int POOL[] = { 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 27, 4, 8, 16 };
            int ncand_total = 0;
            for (int li = 0; li < (int)(sizeof POOL / sizeof POOL[0]); li++) {
                const int r0 = POOL[li];
                if (N % r0 || N / r0 < 3) continue;
                if (!vfft_il2p_leaf_fn(r0, 0) || !vfft_il2p_n1_bwd_fn(r0)) continue;
                int cand[VFFT_IL2D_MAXCAND][8], lens[VFFT_IL2D_MAXCAND], cur[8], nc = 0, dropped = 0;
                _il2d_enum_rec(N / r0, 0, cur, cand, lens, &nc, &dropped);
                for (int c = 0; c < nc; c++) {
                    int chain[9]; chain[0] = r0;
                    for (int s = 0; s < lens[c]; s++) chain[s + 1] = cand[c][s];
                    vfft_ilflat_plan_t *q = vfft_ilflat_create_chain(N, chain, lens[c] + 1);
                    if (!q) continue;
                    ncand_total++;
                    vfft_ilflat_execute_fwd(q, x, z);
                    double tq = time_fwd(q, x, z, 3);
                    if (tq < tbest) { tbest = tq; chain_str(q, bs, sizeof bs); }
                    vfft_ilflat_destroy(q);
                }
            }
            fprintf(stderr, "[race] N=%d: %d candidates -> %s %.0f ns\n", N, ncand_total, bs, tbest);
        }
        /* the winner vs MKL, same run, alternated min-of-9 */
        double tf = 1e300, tm = 1e300, t0;
        {
            int chain[9] = {0}; int K = 0;
            /* rebuild the best chain from its string */
            { char tmp[48]; strcpy(tmp, bs); char *bar = strchr(tmp, '|'); *bar = 0; chain[K++] = atoi(tmp);
              for (char *tok = strtok(bar + 1, "."); tok; tok = strtok(NULL, ".")) chain[K++] = atoi(tok); }
            vfft_ilflat_plan_t *w = vfft_ilflat_create_chain(N, chain, K);
            DFTI_DESCRIPTOR_HANDLE hm = NULL;
            if (w && DftiCreateDescriptor(&hm, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) == 0 &&
                DftiSetValue(hm, DFTI_PLACEMENT, DFTI_NOT_INPLACE) == 0 && DftiCommitDescriptor(hm) == 0) {
                DftiComputeForward(hm, x, y); vfft_ilflat_execute_fwd(w, x, z);
                for (int r = 0; r < 9; r++) {
                    t0 = now_ns(); vfft_ilflat_execute_fwd(w, x, z); t0 = now_ns()-t0; if (t0 < tf) tf = t0;
                    t0 = now_ns(); DftiComputeForward(hm, x, y);     t0 = now_ns()-t0; if (t0 < tm) tm = t0;
                }
                DftiFreeDescriptor(&hm);
            }
            if (w) vfft_ilflat_destroy(w);
        }
        printf("%-7d %-14s | %.1e %.1e %.1e %s | %-14s %9.0f | %9.0f %9.0f %5.2fx\n",
               N, cs, wn, rt, dc, ok ? "OK " : "BAD", bs, tf, tseed, tm, tm/tf);
        fflush(stdout);
        vfft_ilflat_destroy(p); free(x); free(z); free(y);
    }
    printf(bad ? "=== *** %d BAD *** ===\n" : "=== ALL OK (%d cells) ===\n", bad ? bad : n);
    return bad ? 1 : 0;
}
