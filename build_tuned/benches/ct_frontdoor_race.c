/* ct_frontdoor_race.c — what the odd-composite Cooley-Tukey mid is worth
 * through the PUBLIC API, measured in one process.
 *
 * WHY THIS EXISTS. The kernel alone says _ct wins big: the t2 (mid) form
 * measured 2.15x at radix 25 and 2.62x at radix 27, in-place exactly as il2p
 * calls it, spreads 1.7-6.4%. But a first front-door attempt compared two
 * SEPARATE PROCESSES and produced nonsense -- three alternating rounds at
 * N=864 gave base 1537/1678/1797 ns against ct 1618/1905/1926, arms fully
 * overlapping and the verdict flipping run to run. That is this box's
 * documented behaviour: a sub-5%-of-composite effect is not resolvable
 * cross-process, and the fix is to measure in ONE process with the arms
 * alternated (bench_has_three_pacing_knobs).
 *
 * METHOD. Both handles are built up front in the same process. VFFT_IL_KV is
 * read at CREATE time (_k1_il2p_apply_kv in vfft.c), so setting it around the
 * second create is enough to fix that plan's kernel choice for its lifetime --
 * the same create-time A/B hook VFFT_NO_TCMT uses. Then the two handles are
 * timed alternately, medians of 7, pinned to one P-core at HIGH priority.
 *
 * CORRECTNESS FIRST: both arms are compared elementwise against each other
 * before any timing. They compute the same transform by different kernels and
 * must agree to ~1e-12; a fast wrong arm is not a result.
 *
 * WHICH SLOT. At N = 32*odd the pair chooser gives (R1 = odd, R2 = 32), and
 * il2p indexes mid_f by R1 -- so the ODD radix is the MID, and _ct applies
 * there. il_kv = 0x05 means mid variant 5, leaf nibble 0 = leave create's
 * choice alone, so exactly one kernel changes between the arms.
 *
 * Build: python build.py --src benches/ct_frontdoor_race.c --vfft --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <windows.h>
#include "vfft.h"

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
{ int i,j; for(i=1;i<n;i++){ double k=v[i]; j=i-1;
    while(j>=0&&v[j]>k){v[j+1]=v[j];j--;} v[j+1]=k; } return v[n/2]; }
static double spread(const double *v, int n)
{ double lo=v[0],hi=v[0]; int i;
  for(i=1;i<n;i++){ if(v[i]<lo)lo=v[i]; if(v[i]>hi)hi=v[i]; }
  return lo>0?hi/lo-1.0:0.0; }

static vfft_plan build(int N, const char *kv)
{
    vfft_config_t cfg;
    vfft_plan p;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1;
    if (kv) _putenv_s("VFFT_IL_KV", kv); else _putenv_s("VFFT_IL_KV", "");
    p = vfft_create(&cfg);
    _putenv_s("VFFT_IL_KV", "");          /* never leak into the next create */
    return p;
}

static void cell(int N, int pace_ms)
{
    vfft_plan pb = build(N, NULL);        /* base: create's own choice   */
    vfft_plan pc = build(N, "0x05");      /* mid variant 5 = _ct         */
    double *zin, *zb, *zc, tb[TRIALS], tc[TRIALS];
    int i, k, r, reps;
    double w = 0, mag = 0;

    if (!pb || !pc) { printf("  N=%-6d create failed\n", N); return; }
    zin = (double *)calloc(2*(size_t)N + 8, sizeof(double));
    zb  = (double *)calloc(2*(size_t)N + 8, sizeof(double));
    zc  = (double *)calloc(2*(size_t)N + 8, sizeof(double));
    for (i = 0; i < 2*N; i++) zin[i] = rnd();

    vfft_execute(pb, VFFT_FORWARD, zin, NULL, zb, NULL);
    vfft_execute(pc, VFFT_FORWARD, zin, NULL, zc, NULL);
    for (i = 0; i < 2*N; i++) {
        double d = fabs(zb[i] - zc[i]);
        if (d > w) w = d;
        if (fabs(zb[i]) > mag) mag = fabs(zb[i]);
    }
    if (!((mag > 0 ? w/mag : w) < 1e-12)) {
        printf("  N=%-6d *** ARMS DISAGREE (rel %.2e) -- NOT TIMED ***\n",
               N, mag>0?w/mag:w);
        goto done;
    }

    reps = (int)(3000000.0 / (double)N); if (reps < 50) reps = 50; if (reps > 5000) reps = 5000;
    for (k = 0; k < 50; k++) {
        vfft_execute(pb, VFFT_FORWARD, zin, NULL, zb, NULL);
        vfft_execute(pc, VFFT_FORWARD, zin, NULL, zc, NULL);
    }
    for (k = 0; k < TRIALS; k++) {
        double t0 = now_ns();
        for (r = 0; r < reps; r++) vfft_execute(pb, VFFT_FORWARD, zin, NULL, zb, NULL);
        tb[k] = (now_ns()-t0)/reps;
        if (pace_ms) Sleep((DWORD)pace_ms);
        t0 = now_ns();
        for (r = 0; r < reps; r++) vfft_execute(pc, VFFT_FORWARD, zin, NULL, zc, NULL);
        tc[k] = (now_ns()-t0)/reps;
        if (pace_ms) Sleep((DWORD)pace_ms);
    }
    {
        double B = med(tb,TRIALS), C = med(tc,TRIALS);
        printf("  N=%-6d base %8.1f ns (sp %4.1f%%) | _ct mid %8.1f ns (sp %4.1f%%) | %5.2fx %s\n",
               N, B, 100*spread(tb,TRIALS), C, 100*spread(tc,TRIALS), B/C,
               B/C > 1.0 ? "_ct" : "BASE");
    }
done:
    free(zin); free(zb); free(zc);
    vfft_destroy(pb); vfft_destroy(pc);
}

int main(int argc, char **argv)
{
    /* N = 32*odd: the odd radix lands on the MID, which is where _ct applies */
    static const int NS[] = { 288, 480, 672, 800, 864 };
    int pace_ms = (argc > 1) ? atoi(argv[1]) : 20;
    size_t i;
    SetProcessAffinityMask(GetCurrentProcess(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("_ct MID through the public API — one process, arms alternated\n");
    printf("  ratio = base / _ct   (>1 means _ct wins)\n");
    printf("  pinned core 2, HIGH, medians of %d\n\n", TRIALS);
    for (i = 0; i < sizeof NS/sizeof NS[0]; i++) cell(NS[i], pace_ms);
    return 0;
}
