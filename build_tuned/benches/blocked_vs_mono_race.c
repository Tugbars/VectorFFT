/* blocked_vs_mono_race.c — at the counts that actually occur, is the BLOCKED
 * kernel (now that it has an odd-count tail) faster than the MONOLITHIC one?
 *
 * WHY THIS IS NOT ALREADY KNOWN. Blocked was introduced because the
 * monolithic form spills hard at R>=32, and the datum quoted for it (+45%,
 * 858 -> 1251 ns) is N=1024 32x32 -- i.e. count = 32. Every cell that the
 * missing tail demoted has count in 7..27, where the narrow arm is 1-of-4 to
 * 1-of-14 iterations instead of 1-of-16 and the bulk has far fewer trips to
 * amortise the blocked form's S[] round-trip. Whether blocked still wins there
 * is an open question, and the +45% figure must NOT be carried over.
 *
 * The counts below are the real ones: the demoted cells are exactly
 * N = 32*odd (leaf/mid) and N = 64*odd (backward), so the partner count is the
 * odd factor -- 7, 9, 11, 13, 15, 17, 19, 21, 25, 27. Even counts are included
 * as controls: they are what ships today and must not have regressed when the
 * bulk loop's `k` was hoisted.
 *
 * PROTOCOL: one process, arms ALTERNATED, medians of 7, pinned to one P-core
 * at HIGH priority from inside the process (setting affinity on an already
 * running process straddles two machines and was measured at 113-200% spread).
 * Spread is printed beside every median; a ratio inside the spread is not a
 * verdict. Correctness is re-checked before timing so a fast wrong arm cannot
 * win.
 *
 * Build (from build_tuned/benches):
 *   gcc -O3 -mavx2 -mfma -march=native -o blocked_vs_mono_race.exe \
 *       blocked_vs_mono_race.c k_blk.c k_mono.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <windows.h>

#define R 32

void radix32_z_n1tb_fwd_avx2(const double *, const double *, double *, double *,
                             const double *, const double *,
                             size_t, size_t, size_t, size_t, size_t);
void radix32_z_n1t_fwd_avx2 (const double *, const double *, double *, double *,
                             const double *, const double *,
                             size_t, size_t, size_t, size_t, size_t);

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
{ int i, j; for (i = 1; i < n; i++) { double k = v[i]; j = i - 1;
    while (j >= 0 && v[j] > k) { v[j+1] = v[j]; j--; } v[j+1] = k; } return v[n/2]; }
static double spread(const double *v, int n)
{ double lo = v[0], hi = v[0]; int i;
  for (i = 1; i < n; i++) { if (v[i] < lo) lo = v[i]; if (v[i] > hi) hi = v[i]; }
  return lo > 0 ? hi/lo - 1.0 : 0.0; }

static void cell(size_t count, int pace_ms)
{
    const size_t Ls = count, OLs = R;
    const size_t nin = 2*R*Ls, nout = 2*count*OLs;
    double *zin = (double *)_aligned_malloc((nin + 16)*sizeof(double), 64);
    double *ob  = (double *)_aligned_malloc((nout + 16)*sizeof(double), 64);
    double *om  = (double *)_aligned_malloc((nout + 16)*sizeof(double), 64);
    double tb[TRIALS], tm[TRIALS];
    size_t i; int k, r, reps;
    double worst = 0, mag = 0;

    for (i = 0; i < nin; i++) zin[i] = rnd();
    memset(ob, 0, nout*sizeof(double)); memset(om, 0, nout*sizeof(double));

    /* correctness first */
    radix32_z_n1tb_fwd_avx2(zin, 0, ob, 0, 0, 0, Ls, 0, OLs, 0, count);
    radix32_z_n1t_fwd_avx2 (zin, 0, om, 0, 0, 0, Ls, 0, OLs, 0, count);
    for (i = 0; i < nout; i++) {
        double d = fabs(ob[i] - om[i]);
        if (d > worst) worst = d;
        if (fabs(om[i]) > mag) mag = fabs(om[i]);
    }
    if (!((mag > 0 ? worst/mag : worst) < 1e-12)) {
        printf("  count=%-3zu  *** ARMS DISAGREE -- NOT TIMED ***\n", count);
        _aligned_free(zin); _aligned_free(ob); _aligned_free(om); return;
    }

    reps = (int)(2000000.0 / (double)(R*count));
    if (reps < 200) reps = 200;
    if (reps > 20000) reps = 20000;

    for (k = 0; k < 200; k++) {          /* warm both */
        radix32_z_n1tb_fwd_avx2(zin, 0, ob, 0, 0, 0, Ls, 0, OLs, 0, count);
        radix32_z_n1t_fwd_avx2 (zin, 0, om, 0, 0, 0, Ls, 0, OLs, 0, count);
    }
    for (k = 0; k < TRIALS; k++) {       /* ALTERNATE, so drift hits both */
        double t0 = now_ns();
        for (r = 0; r < reps; r++)
            radix32_z_n1tb_fwd_avx2(zin, 0, ob, 0, 0, 0, Ls, 0, OLs, 0, count);
        tb[k] = (now_ns() - t0)/reps;
        if (pace_ms) Sleep((DWORD)pace_ms);
        t0 = now_ns();
        for (r = 0; r < reps; r++)
            radix32_z_n1t_fwd_avx2 (zin, 0, om, 0, 0, 0, Ls, 0, OLs, 0, count);
        tm[k] = (now_ns() - t0)/reps;
        if (pace_ms) Sleep((DWORD)pace_ms);
    }
    {
        double B = med(tb, TRIALS), M = med(tm, TRIALS);
        double sb = spread(tb, TRIALS), sm = spread(tm, TRIALS);
        printf("  count=%-3zu %-5s | blocked %8.2f ns (sp %4.1f%%) | mono %8.2f ns (sp %4.1f%%) | %5.2fx %s\n",
               count, (count & 1) ? "ODD" : "even", B, 100*sb, M, 100*sm, M/B,
               M/B > 1.0 ? "blocked" : "MONO");
    }
    _aligned_free(zin); _aligned_free(ob); _aligned_free(om);
}

int main(int argc, char **argv)
{
    /* the odd partner counts that actually occur at the demoted cells, plus
     * even controls */
    static const size_t CS[] = { 7, 8, 9, 11, 13, 15, 16, 17, 19, 21, 25, 27, 32, 64 };
    int pace_ms = (argc > 1) ? atoi(argv[1]) : 20;
    size_t i;
    SetProcessAffinityMask(GetCurrentProcess(), 0x4);   /* one P-core */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("radix32 n1t: BLOCKED(4.8)+tail vs MONOLITHIC, per count\n");
    printf("  pinned core 2, HIGH, medians of %d, arms alternated, pace %d ms\n", TRIALS, pace_ms);
    printf("  ratio = mono / blocked  (>1 means BLOCKED wins)\n\n");
    for (i = 0; i < sizeof CS/sizeof CS[0]; i++) cell(CS[i], pace_ms);
    return 0;
}
