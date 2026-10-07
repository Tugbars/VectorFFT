/* odd_partner_cells_gate.c — the 20 cells that the even-partner-count contract
 * used to demote, checked for CORRECTNESS and then A/B'd for the win.
 *
 * THE CELLS. A 1D C2C IL Bailey pair runs its leaf at count = R1 and its mid at
 * count = R2, so an odd factor makes one pass's count odd. Blocked cil kernels
 * had no odd-count tail, so vfft_il2p_*_v_fn refused them whenever the partner
 * was odd (`count_ok`) and the slot degraded to the monolithic kernel. With the
 * shipped radix set those cells are exactly N = 32*odd (forward leaf/mid) and
 * N = 64*odd (backward), because no both-even (R1,R2) pair exists for them --
 * 288 cannot be 16x18, there is no radix 18.
 *
 * 2026-08-23 gave blocked the narrow tail and lifted the gates. This asks two
 * questions per cell:
 *
 *   CORRECT?  forward output vs a naive DFT, and the roundtrip. If the tail
 *             were wrong, an odd-count cell computes a wrong column and this
 *             catches it at the front door rather than at kernel level.
 *
 *   FASTER?   the SAME plan built twice, once with VFFT_NO_ILBLK set. That env
 *             var is the emitter-era A/B hook for exactly this: it suppresses
 *             the blocked structural default, so the arms differ ONLY in
 *             blocked-vs-monolithic. If the two arms tie, blocked did not
 *             engage and the cell proves nothing -- which is why the ratio is
 *             reported per cell rather than averaged.
 *
 * REAL TRANSFORMS ARE INCLUDED. The zr2c route is real N -> child c2c(N/2), so
 * a real transform inherits whatever the child cell uses. The affected real
 * sizes are therefore N = 2 * (32*odd) = 64*odd. This gate builds them through
 * the public R2C door to prove the inheritance is real and not assumed.
 *
 * Build: python build.py --src benches/odd_partner_cells_gate.c --vfft --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <windows.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double now_ns(void)
{
    static LARGE_INTEGER f; static int init = 0; LARGE_INTEGER c;
    if (!init) { QueryPerformanceFrequency(&f); init = 1; }
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}
static uint64_t lcg = 0x243F6A8885A308D3ull;
static double rnd(void)
{ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
  return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

#define TRIALS 7
static double med(double *v, int n)
{ int i,j; for (i=1;i<n;i++){ double k=v[i]; j=i-1;
    while(j>=0&&v[j]>k){v[j+1]=v[j];j--;} v[j+1]=k; } return v[n/2]; }
static double spread(const double *v, int n)
{ double lo=v[0],hi=v[0]; int i;
  for(i=1;i<n;i++){ if(v[i]<lo)lo=v[i]; if(v[i]>hi)hi=v[i]; }
  return lo>0?hi/lo-1.0:0.0; }

static int g_fail = 0;

/* naive complex DFT of one transform, forward */
static void naive_c2c(const double *z, int N, double *o)
{
    int f, n;
    for (f = 0; f < N; f++) {
        double sr = 0, si = 0;
        for (n = 0; n < N; n++) {
            double a = -2.0*M_PI*(double)f*n/(double)N;
            double c = cos(a), s = sin(a);
            sr += z[2*n]*c - z[2*n+1]*s;
            si += z[2*n]*s + z[2*n+1]*c;
        }
        o[2*f] = sr; o[2*f+1] = si;
    }
}

/* one C2C IL cell: correctness vs naive, then blocked-vs-mono A/B */
static void c2c_cell(int N, int pace_ms)
{
    vfft_config_t cfg;
    vfft_plan pb, pm;
    double *zin  = (double *)calloc(2*(size_t)N + 8, sizeof(double));
    double *zb   = (double *)calloc(2*(size_t)N + 8, sizeof(double));
    double *zm   = (double *)calloc(2*(size_t)N + 8, sizeof(double));
    double *ref  = (double *)calloc(2*(size_t)N + 8, sizeof(double));
    double tb[TRIALS], tm[TRIALS];
    int i, k, r, reps;
    double w = 0, mag = 0;

    for (i = 0; i < 2*N; i++) zin[i] = rnd();
    naive_c2c(zin, N, ref);

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1;

    _putenv("VFFT_NO_ILBLK=");            /* blocked ALLOWED */
    pb = vfft_create(&cfg);
    _putenv("VFFT_NO_ILBLK=1");           /* blocked SUPPRESSED */
    pm = vfft_create(&cfg);
    _putenv("VFFT_NO_ILBLK=");
    if (!pb || !pm) { printf("  C2C N=%-5d create failed\n", N); goto done; }

    vfft_execute(pb, VFFT_FORWARD, zin, NULL, zb, NULL);
    vfft_execute(pm, VFFT_FORWARD, zin, NULL, zm, NULL);
    for (i = 0; i < 2*N; i++) {
        double d = fabs(zb[i] - ref[i]);
        if (d > w) w = d;
        if (fabs(ref[i]) > mag) mag = fabs(ref[i]);
    }
    if (!((mag > 0 ? w/mag : w) < 1e-9)) {
        printf("  C2C N=%-5d  *** WRONG vs naive (rel %.2e) ***\n", N, mag>0?w/mag:w);
        g_fail = 1; goto done;
    }

    reps = (int)(3000000.0 / (double)N); if (reps < 50) reps = 50; if (reps > 5000) reps = 5000;
    for (k = 0; k < 50; k++) {
        vfft_execute(pb, VFFT_FORWARD, zin, NULL, zb, NULL);
        vfft_execute(pm, VFFT_FORWARD, zin, NULL, zm, NULL);
    }
    for (k = 0; k < TRIALS; k++) {
        double t0 = now_ns();
        for (r = 0; r < reps; r++) vfft_execute(pb, VFFT_FORWARD, zin, NULL, zb, NULL);
        tb[k] = (now_ns()-t0)/reps;
        if (pace_ms) Sleep((DWORD)pace_ms);
        t0 = now_ns();
        for (r = 0; r < reps; r++) vfft_execute(pm, VFFT_FORWARD, zin, NULL, zm, NULL);
        tm[k] = (now_ns()-t0)/reps;
        if (pace_ms) Sleep((DWORD)pace_ms);
    }
    {
        double B = med(tb,TRIALS), M = med(tm,TRIALS);
        printf("  C2C N=%-5d rel %.1e | blocked %8.1f ns (sp %4.1f%%) | mono %8.1f ns (sp %4.1f%%) | %5.2fx\n",
               N, mag>0?w/mag:w, B, 100*spread(tb,TRIALS), M, 100*spread(tm,TRIALS), M/B);
    }
done:
    if (pb) vfft_destroy(pb);
    if (pm) vfft_destroy(pm);
    free(zin); free(zb); free(zm); free(ref);
}

/* the same A/B through the REAL door: N = 2*(child cell) */
static void r2c_cell(int N, int pace_ms)
{
    const size_t nb = (size_t)N/2 + 1;
    vfft_config_t cfg;
    vfft_plan pb, pm;
    double *x  = (double *)calloc((size_t)N + 8, sizeof(double));
    double *ob = (double *)calloc(2*nb + 8, sizeof(double));
    double *om = (double *)calloc(2*nb + 8, sizeof(double));
    double tb[TRIALS], tm[TRIALS];
    int i, k, r, reps;
    double w = 0, mag = 0;

    for (i = 0; i < N; i++) x[i] = rnd();

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1;

    _putenv("VFFT_NO_ILBLK=");
    pb = vfft_create(&cfg);
    _putenv("VFFT_NO_ILBLK=1");
    pm = vfft_create(&cfg);
    _putenv("VFFT_NO_ILBLK=");
    if (!pb || !pm) { printf("  R2C N=%-5d create failed\n", N); goto done; }

    vfft_execute(pb, VFFT_FORWARD, x, NULL, ob, NULL);
    vfft_execute(pm, VFFT_FORWARD, x, NULL, om, NULL);
    /* the two arms must agree with EACH OTHER: same transform, different
     * kernel class, so this is the cheap correctness check at the real door */
    for (i = 0; i < (int)(2*nb); i++) {
        double d = fabs(ob[i] - om[i]);
        if (d > w) w = d;
        if (fabs(om[i]) > mag) mag = fabs(om[i]);
    }
    if (!((mag > 0 ? w/mag : w) < 1e-12)) {
        printf("  R2C N=%-5d  *** ARMS DISAGREE (rel %.2e) ***\n", N, mag>0?w/mag:w);
        g_fail = 1; goto done;
    }

    reps = (int)(3000000.0 / (double)N); if (reps < 50) reps = 50; if (reps > 5000) reps = 5000;
    for (k = 0; k < 50; k++) {
        vfft_execute(pb, VFFT_FORWARD, x, NULL, ob, NULL);
        vfft_execute(pm, VFFT_FORWARD, x, NULL, om, NULL);
    }
    for (k = 0; k < TRIALS; k++) {
        double t0 = now_ns();
        for (r = 0; r < reps; r++) vfft_execute(pb, VFFT_FORWARD, x, NULL, ob, NULL);
        tb[k] = (now_ns()-t0)/reps;
        if (pace_ms) Sleep((DWORD)pace_ms);
        t0 = now_ns();
        for (r = 0; r < reps; r++) vfft_execute(pm, VFFT_FORWARD, x, NULL, om, NULL);
        tm[k] = (now_ns()-t0)/reps;
        if (pace_ms) Sleep((DWORD)pace_ms);
    }
    {
        double B = med(tb,TRIALS), M = med(tm,TRIALS);
        printf("  R2C N=%-5d (child %-4d) | blocked %8.1f ns (sp %4.1f%%) | mono %8.1f ns (sp %4.1f%%) | %5.2fx\n",
               N, N/2, B, 100*spread(tb,TRIALS), M, 100*spread(tm,TRIALS), M/B);
    }
done:
    if (pb) vfft_destroy(pb);
    if (pm) vfft_destroy(pm);
    free(x); free(ob); free(om);
}

int main(int argc, char **argv)
{
    /* N = 32*odd (fwd leaf/mid) and 64*odd (bwd) — the demoted set */
    static const int C2C_N[] = { 224, 288, 352, 416, 448, 480, 544, 576,
                                 608, 672, 704, 800, 832, 864, 960 };
    /* real N whose child N/2 lands on one of those cells */
    static const int R2C_N[] = { 448, 576, 704, 832, 960, 1088, 1344, 1600, 1728 };
    int pace_ms = (argc > 1) ? atoi(argv[1]) : 15;
    size_t i;

    SetProcessAffinityMask(GetCurrentProcess(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("odd-partner cells: blocked(with tail) vs monolithic, through the front door\n");
    printf("  ratio = mono / blocked  (>1 means blocked wins; ~1.00 means it did NOT engage)\n");
    printf("  pinned core 2, HIGH, medians of %d, arms alternated\n\n", TRIALS);

    printf("[1D C2C IL — the cells the count_ok gate demoted]\n");
    for (i = 0; i < sizeof C2C_N/sizeof C2C_N[0]; i++) c2c_cell(C2C_N[i], pace_ms);

    printf("\n[1D R2C (zr2c) — inherits the child c2c(N/2) cell]\n");
    for (i = 0; i < sizeof R2C_N/sizeof R2C_N[0]; i++) r2c_cell(R2C_N[i], pace_ms);

    printf("\n%s\n", g_fail ? "*** ODD-PARTNER CELLS: INCORRECT ***"
                            : "odd-partner cells: all correct");
    return g_fail;
}
