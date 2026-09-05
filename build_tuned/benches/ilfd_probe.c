/* THE FLAT DIT STRUCTURE CHECK (2026-09-04): un-turned flat mixed-radix
 * DIT on shipped kinds (n1c leaf + t2 pre-twiddle mids with block-repeated
 * records + natural redirection of the last stage). Forward only: naive
 * DFT at NATURAL indices + DC, then the timing vs MKL and vs the four-step
 * bridge engine (il_flat.h) with the same seed chain family. With --race
 * both engines race their chains from the same enumerator. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "vfft.h"
#include "../../src/core/oop/il_flat.h"
#include "../../src/core/oop/il_flatdit.h"
static double now_ns(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart*1e9/(double)f.QuadPart;}
static void chain_s(const int *R, int K, char *cs, size_t n) { int off = 0; for (int s = 0; s < K; s++) off += snprintf(cs + off, n - off, "%s%d", s ? "." : "", R[s]); }
/* the plan's chain with an "m" on every stage that races onto msz */
static void plan_s(const vfft_ilfd_plan_t *p, char *cs, size_t n) { int off = 0; for (int s = 0; s < p->K; s++) off += snprintf(cs + off, n - off, "%s%d%s", s ? "." : "", p->R[s], (s && p->msz[s]) ? "m" : (s && p->gl[s]) ? ((s == p->K - 1 && p->gord) ? "o" : "n") : ""); }
/* --stages: per-stage timing of the flat DIT (seed chain) — where does a
 * sweep's time go: the leaf, the long-run mids, the short-run tail? */
static void stage_times(const vfft_ilfd_plan_t *p, const double *zin, double *zout) {
    double ts[VFFT_ILFD_MAX_K];
    for (int s = 0; s < p->K; s++) ts[s] = 1e300;
    for (int r = 0; r < 5; r++)
        for (int s = 0; s < p->K; s++) {
            double t0 = now_ns(); vfft_ilfd_stage(p, s, zin, zout); t0 = now_ns() - t0;
            if (t0 < ts[s]) ts[s] = t0;
        }
    printf("      stages: leaf(R%d,cnt%zu) %.0f", p->R[0], p->D[0], ts[0]);
    for (int s = 1; s < p->K; s++) printf(" | R%d cnt%zu x%zu%s: %.0f", p->R[s], p->D[s], p->nblk[s], p->tail[s] ? " t2cs" : "", ts[s]);
    printf("%s", "\n");
}
int main(int argc, char **argv) {
    static const int NS[] = { 405, 1215, 3125, 4095, 6561, 15625, 16807, 19683, 59049, 78125, 98415, 117649, 137781 };
    const int n = (int)(sizeof NS / sizeof NS[0]);
    const int race = (argc > 1 && !strcmp(argv[1], "--race"));
    int bad = 0;
    mkl_set_num_threads(1);
    printf("%-7s %-22s | dft@nat  dc      | %9s | %-14s %9s | %9s %6s %6s\n", "N", "flat chain (m=msz,o=ord)", "flat(ns)", "4step best", "4step(ns)", "mkl(ns)", "f/mkl", "f/4st");
    for (int i = 0; i < n; i++) {
        const int N = NS[i];
        vfft_ilfd_plan_t *p = vfft_ilfd_create(N);
        char cs[48] = "-", bs[48] = "-";
        if (!p) { printf("%-7d NO CHAIN\n", N); bad++; continue; }
        chain_s(p->R, p->K, cs, sizeof cs);
        double *x = malloc(2*(size_t)N*8), *z = malloc(2*(size_t)N*8), *y = malloc(2*(size_t)N*8);
        double s0r = 0, s0i = 0;
        for (int j = 0; j < N; j++) { x[2*j] = (double)rand()/RAND_MAX-0.5; x[2*j+1] = (double)rand()/RAND_MAX-0.5; s0r += x[2*j]; s0i += x[2*j+1]; }
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
        if (argc > 1 && !strcmp(argv[1], "--stages")) { printf("%-7d %-22s\n", N, cs); stage_times(p, x, z); }
        /* flat DIT: seed or raced chain */
        double tf = 1e300;
        { vfft_ilfd_plan_t *best = p; int owned = 0;
          if (race) {
              vfft_ilfd_race_forms(p, x, z, now_ns);   /* per-stage form race on the seed too */
              vfft_ilfd_execute_fwd(p, x, z);
              for (int r = 0; r < 3; r++) { double t0 = now_ns(); vfft_ilfd_execute_fwd(p, x, z); t0 = now_ns()-t0; if (t0 < tf) tf = t0; }  /* the seed races too */
              int cand[VFFT_IL2D_MAXCAND][8], lens[VFFT_IL2D_MAXCAND], cur[8], nc = 0, dropped = 0;
              _il2d_enum_rec(N, 0, cur, cand, lens, &nc, &dropped);
              for (int c = 0; c < nc; c++) {
                  vfft_ilfd_plan_t *q = vfft_ilfd_create_chain(N, cand[c], lens[c]);
                  if (!q) continue;
                  vfft_ilfd_race_forms(q, x, z, now_ns);
                  vfft_ilfd_execute_fwd(q, x, z);
                  double tq = 1e300;
                  for (int r = 0; r < 3; r++) { double t0 = now_ns(); vfft_ilfd_execute_fwd(q, x, z); t0 = now_ns()-t0; if (t0 < tq) tq = t0; }
                  if (tq < tf) { tf = tq; if (owned) vfft_ilfd_destroy(best); best = q; owned = 1; }
                  else vfft_ilfd_destroy(q);
              }
          }
          plan_s(best, cs, sizeof cs);
          /* the four-step bridge's best (same race) for the same cell */
          vfft_ilflat_plan_t *fs = vfft_ilflat_create(N);
          double t4 = 1e300;
          if (race) {
              static const int POOL[] = { 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 27 };
              for (int li = 0; li < (int)(sizeof POOL / sizeof POOL[0]); li++) {
                  const int r0 = POOL[li];
                  if (N % r0 || N / r0 < 3 || !vfft_il2p_leaf_fn(r0, 0) || !vfft_il2p_n1_bwd_fn(r0)) continue;
                  int cand[VFFT_IL2D_MAXCAND][8], lens[VFFT_IL2D_MAXCAND], cur[8], nc = 0, dropped = 0;
                  _il2d_enum_rec(N / r0, 0, cur, cand, lens, &nc, &dropped);
                  for (int c = 0; c < nc; c++) {
                      int chain[9]; chain[0] = r0; for (int s = 0; s < lens[c]; s++) chain[s + 1] = cand[c][s];
                      vfft_ilflat_plan_t *q = vfft_ilflat_create_chain(N, chain, lens[c] + 1);
                      if (!q) continue;
                      vfft_ilflat_execute_fwd(q, x, y);
                      double tq = 1e300;
                      for (int r = 0; r < 3; r++) { double t0 = now_ns(); vfft_ilflat_execute_fwd(q, x, y); t0 = now_ns()-t0; if (t0 < tq) tq = t0; }
                      if (tq < t4) { t4 = tq; if (fs) vfft_ilflat_destroy(fs); fs = q; } else vfft_ilflat_destroy(q);
                  }
              }
          }
          if (fs) { int off = snprintf(bs, sizeof bs, "%d|", fs->R0); for (int s = 0; s < fs->nst; s++) off += snprintf(bs + off, sizeof bs - off, "%s%d", s ? "." : "", fs->R[s]); }
          /* final same-run timing: flat DIT best vs 4-step best vs MKL, alternated */
          double tm = 1e300; tf = 1e300; t4 = 1e300;
          DFTI_DESCRIPTOR_HANDLE hm = NULL;
          if (DftiCreateDescriptor(&hm, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N) == 0 &&
              DftiSetValue(hm, DFTI_PLACEMENT, DFTI_NOT_INPLACE) == 0 && DftiCommitDescriptor(hm) == 0) {
              for (int r = 0; r < 9; r++) {
                  double t0 = now_ns(); vfft_ilfd_execute_fwd(best, x, z); t0 = now_ns()-t0; if (t0 < tf) tf = t0;
                  if (fs) { t0 = now_ns(); vfft_ilflat_execute_fwd(fs, x, y); t0 = now_ns()-t0; if (t0 < t4) t4 = t0; }
                  t0 = now_ns(); DftiComputeForward(hm, x, y); t0 = now_ns()-t0; if (t0 < tm) tm = t0;
              }
              DftiFreeDescriptor(&hm);
          }
          printf("%-7d %-22s | %.1e %.1e %s | %9.0f | %-14s %9.0f | %9.0f %5.2fx %5.2fx\n",
                 N, cs, wn, dc, ok ? "OK " : "BAD", tf, bs, t4, tm, tm/tf, t4/tf);
          fflush(stdout);
          if (owned) vfft_ilfd_destroy(best);
          if (fs) vfft_ilflat_destroy(fs);
        }
        vfft_ilfd_destroy(p); free(x); free(z); free(y);
    }
    printf(bad ? "=== *** %d BAD *** ===\n" : "=== ALL OK (%d cells) ===\n", bad ? bad : n);
    return bad ? 1 : 0;
}
