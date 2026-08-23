/* odd_ct_race.c — is the FACTORED odd-composite kernel actually faster?
 *
 * Static op counts say the Cooley-Tukey form is 1.07x (radix 9) to 1.76x
 * (radix 27) cheaper than the direct conjugate-pair form. This repo's rule is
 * that static cuts do not predict time, so this measures it: same process,
 * arms alternated, medians of 7, pinned to one P-core at HIGH priority.
 *
 * Correctness is not re-checked here (odd_ct_gate.c already proved both arms
 * against a long-double naive DFT at every radix x count); this file is
 * timing only, and refuses nothing.
 *
 * COUNT MATTERS and is swept. The IL leaf runs at count = the partner factor,
 * so at the affected cells (N = 32*odd) count is 32 -- but the odd radix also
 * appears as the MID at count = 32, and inside the 3-stage chain at other
 * counts. A form that wins only at one count is not a verdict.
 *
 * Build (from build_tuned/benches):
 *   gcc -O3 -mavx2 -mfma -march=native -o odd_ct_race.exe odd_ct_race.c \
 *       oc_off_9.c oc_on_9.c oc_off_15.c oc_on_15.c oc_off_21.c oc_on_21.c \
 *       oc_off_25.c oc_on_25.c oc_off_27.c oc_on_27.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <windows.h>

typedef void (*kfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);

#define DECL(R)                                                               \
  void radix##R##_z_n1t_fwd_avx2(const double *, const double *, double *,     \
      double *, const double *, const double *,                                \
      size_t, size_t, size_t, size_t, size_t);                                 \
  void radix##R##_z_n1t_oddct_avx2(const double *, const double *, double *,   \
      double *, const double *, const double *,                                \
      size_t, size_t, size_t, size_t, size_t);
DECL(9) DECL(15) DECL(21) DECL(25) DECL(27)

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

static void cell(int R, kfn off, kfn on, size_t count, int pace_ms)
{
    const size_t Ls = count, OLs = (size_t)R;
    const size_t nin = 2*(size_t)R*Ls, nout = 2*count*OLs;
    double *zin = (double *)_aligned_malloc(nin*sizeof(double), 64);
    double *za  = (double *)_aligned_malloc(nout*sizeof(double), 64);
    double *zb  = (double *)_aligned_malloc(nout*sizeof(double), 64);
    double ta[TRIALS], tb[TRIALS];
    size_t i; int k, r, reps;

    for (i = 0; i < nin; i++) zin[i] = rnd();
    reps = (int)(2000000.0 / ((double)R*count));
    if (reps < 300) reps = 300; if (reps > 20000) reps = 20000;

    for (k = 0; k < 200; k++) {
        off(zin,0,za,0,0,0,Ls,0,OLs,0,count);
        on (zin,0,zb,0,0,0,Ls,0,OLs,0,count);
    }
    for (k = 0; k < TRIALS; k++) {
        double t0 = now_ns();
        for (r = 0; r < reps; r++) off(zin,0,za,0,0,0,Ls,0,OLs,0,count);
        ta[k] = (now_ns()-t0)/reps;
        if (pace_ms) Sleep((DWORD)pace_ms);
        t0 = now_ns();
        for (r = 0; r < reps; r++) on (zin,0,zb,0,0,0,Ls,0,OLs,0,count);
        tb[k] = (now_ns()-t0)/reps;
        if (pace_ms) Sleep((DWORD)pace_ms);
    }
    {
        double A = med(ta,TRIALS), B = med(tb,TRIALS);
        printf("  radix %-3d count=%-3zu | direct %8.1f ns (sp %4.1f%%) | factored %8.1f ns (sp %4.1f%%) | %5.2fx %s\n",
               R, count, A, 100*spread(ta,TRIALS), B, 100*spread(tb,TRIALS), A/B,
               A/B > 1.0 ? "factored" : "DIRECT");
    }
    _aligned_free(zin); _aligned_free(za); _aligned_free(zb);
}

int main(int argc, char **argv)
{
    static const size_t CS[] = { 2, 8, 32 };
    int pace_ms = (argc > 1) ? atoi(argv[1]) : 15;
    size_t c;
    SetProcessAffinityMask(GetCurrentProcess(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("odd-composite CT race — direct (shipped) vs factored, radix n1t leaf\n");
    printf("  ratio = direct / factored  (>1 means FACTORED wins)\n");
    printf("  pinned core 2, HIGH, medians of %d, arms alternated\n\n", TRIALS);
    for (c = 0; c < sizeof CS/sizeof CS[0]; c++) {
        cell(9,  radix9_z_n1t_fwd_avx2,  radix9_z_n1t_oddct_avx2,  CS[c], pace_ms);
        cell(15, radix15_z_n1t_fwd_avx2, radix15_z_n1t_oddct_avx2, CS[c], pace_ms);
        cell(21, radix21_z_n1t_fwd_avx2, radix21_z_n1t_oddct_avx2, CS[c], pace_ms);
        cell(25, radix25_z_n1t_fwd_avx2, radix25_z_n1t_oddct_avx2, CS[c], pace_ms);
        cell(27, radix27_z_n1t_fwd_avx2, radix27_z_n1t_oddct_avx2, CS[c], pace_ms);
        printf("\n");
    }
    return 0;
}
