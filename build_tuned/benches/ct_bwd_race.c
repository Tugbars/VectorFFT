/* ct_bwd_race.c — do the BACKWARD _ct kernels earn their slots?
 *
 * WHAT _ct IS. Instead of one direct conjugate-pair DFT at an odd composite
 * radix, factor it (25 = 5x5, 27 = 3x9, 21 = 3x7, 15 = 3x5) and run the
 * four-step. Same trade as blocking: extra passes bought with recovered
 * register pressure, so it pays exactly where the direct form SPILLS.
 *
 * WHY THIS RACE EXISTS. The forward _ct variants were raced and wired
 * (il2p.h variant 5). The BACKWARD twins were emitted at the same time and
 * never raced -- unreachable not because they lost, but because nothing
 * offers them: the plan's leaf_b / mid_b slots resolve through
 * vfft_il2p_leaf_fn/mid_fn, which have no variant axis. That is a wiring
 * gap, not a verdict, and the pool policy is delete-AFTER-the-race.
 *
 * STATIC PREDICTION (compiled asm, bulk loop only) — the same law as forward:
 *     R=9   baseline 0.0% spill  ->  _ct 3.8%   and MORE instructions (105 v 99)
 *     R=15  baseline 22.1%       ->  _ct 12.8%  (219 v 267 insns)
 *     R=25  baseline 48.4%       ->  _ct 23.1%  (493 v 1009)
 *     R=27  baseline 49.9%       ->  _ct 24.7%  (530 v 1144)
 * So radix 9 should LOSE (nothing to recover) and 25/27 should win big.
 *
 * SHAPES, which decide the answer if you get them wrong:
 *   n1t bwd — twiddle-free, OUT-OF-PLACE, Ls = count, OLs = R. Reads R legs
 *             at zin[2*(l*Ls + k)], so the input plane must be sized 2*R*count
 *             — undersizing it reads uninitialised heap and the kernel appears
 *             non-deterministic (that exact bug cost a wrong verdict today).
 *   t2  bwd — twiddled, IN-PLACE, Ls = OLs = count.
 *
 * CORRECTNESS: _ct changes the index mapping, not the transform, so the two
 * forms must agree to rounding. Tolerance, not memcmp: PASS 2 goes through
 * the generic-ctw path.
 *
 * INSTRUMENT: MIN of N trials (timing noise is one-sided) plus a CONTROL arm
 * that times the baseline a second time from a second call site. An arm only
 * counts if it clears the control.
 *
 * Build: python build.py --src benches/ct_bwd_race.c --compile
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

#define DECL(SYM) void SYM(const double *, const double *, double *, double *, \
                           const double *, const double *,                     \
                           size_t, size_t, size_t, size_t, size_t);
#define PAIR(R)                              \
  DECL(radix##R##_z_n1t_bwd_avx2)            \
  DECL(radix##R##_z_n1t_ct_bwd_avx2)         \
  DECL(radix##R##_z_t2_bwd_avx2)             \
  DECL(radix##R##_z_t2_ct_bwd_avx2)
PAIR(9) PAIR(15) PAIR(21) PAIR(25) PAIR(27)
#undef PAIR
#undef DECL

static double now_ns(void)
{ static LARGE_INTEGER f; static int i2=0; LARGE_INTEGER c;
  if(!i2){QueryPerformanceFrequency(&f);i2=1;} QueryPerformanceCounter(&c);
  return (double)c.QuadPart*1e9/(double)f.QuadPart; }
static uint64_t lcg = 0x243F6A8885A308D3ull;
static double rnd(void)
{ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
  return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

#define TRIALS 21
static double best(const double *v,int n)
{ int i; double lo=v[0]; for(i=1;i<n;i++) if(v[i]<lo) lo=v[i]; return lo; }
static double relmax(const double*a,const double*b,size_t n)
{ size_t i; double w=0,mag=0;
  for(i=0;i<n;i++){ double d=fabs(a[i]-b[i]);
      if(d>w)w=d; if(fabs(a[i])>mag)mag=fabs(a[i]); }
  return mag>0?w/mag:w; }

static int g_fail = 0;

static void race(const char *what, int R, kfn base, kfn ct,
                 size_t count, int inplace, int pace_ms)
{
    const size_t Ls = count, OLs = inplace ? count : (size_t)R;
    const size_t groups = (count + 1) / 2;
    const size_t ntw = groups * (size_t)(R - 1) * 8u + 64u;
    const size_t n = 2 * (size_t)R * count + 64;   /* R legs — size for R */
    double *zin=(double*)_aligned_malloc(n*sizeof(double),64);
    double *za =(double*)_aligned_malloc(n*sizeof(double),64);
    double *zb =(double*)_aligned_malloc(n*sizeof(double),64);
    double *tw =(double*)_aligned_malloc(ntw*sizeof(double),64);
    double ta[TRIALS], tb[TRIALS], tc[TRIALS];
    size_t i; int k,r,reps; double rel;

    for(i=0;i<n;i++) zin[i]=rnd();
    for(i=0;i<ntw;i++){ double th=0.37*(double)i; tw[i]=((i>>2)&1)?sin(th):cos(th); }

#define CALL(F,DST) do{ if(inplace){ memcpy(DST,zin,n*sizeof(double));        \
                            F(DST,0,DST,0,tw,0,Ls,0,OLs,0,count); }           \
                        else { memset(DST,0,n*sizeof(double));                \
                            F(zin,0,DST,0,tw,0,Ls,0,OLs,0,count); } }while(0)
    CALL(base,za); CALL(ct,zb);
    rel = relmax(za,zb,2*(size_t)R*count);   /* the DEFINED region only */
    if(!(rel < 1e-12)){
        printf("  %-9s R=%-2d count=%-3zu *** DISAGREE rel %.2e ***\n",what,R,count,rel);
        g_fail=1; goto done;
    }

    reps=(int)(3000000.0/((double)R*count)); if(reps<400)reps=400; if(reps>30000)reps=30000;
    for(k=0;k<200;k++){ CALL(base,za); CALL(ct,zb); }
    for(k=0;k<TRIALS;k++){
        double t=now_ns();
        for(r=0;r<reps;r++) CALL(base,za);
        ta[k]=(now_ns()-t)/reps; if(pace_ms) Sleep((DWORD)pace_ms);
        t=now_ns();
        for(r=0;r<reps;r++) CALL(ct,zb);
        tb[k]=(now_ns()-t)/reps; if(pace_ms) Sleep((DWORD)pace_ms);
        t=now_ns();                       /* control: baseline again */
        for(r=0;r<reps;r++) CALL(base,za);
        tc[k]=(now_ns()-t)/reps; if(pace_ms) Sleep((DWORD)pace_ms);
    }
#undef CALL
    {
        double A=best(ta,TRIALS), B=best(tb,TRIALS), C=best(tc,TRIALS);
        double gap=100.0*(A/B-1.0), ctl=100.0*(A/C-1.0);
        const char *verdict = (fabs(gap) < 2.0*fabs(ctl) || fabs(gap) < 1.0)
                                ? "--" : (gap > 0 ? "_ct" : "BASELINE");
        printf("  %-9s R=%-2d count=%-3zu | base %8.2f | _ct %8.2f | %+7.1f%% | ctl %+5.1f%% | %s\n",
               what,R,count,A,B,gap,ctl,verdict);
    }
done:
    _aligned_free(zin);_aligned_free(za);_aligned_free(zb);_aligned_free(tw);
}

int main(int argc,char**argv)
{
    static const size_t CS[]={8,16,32};
    int pace=(argc>1)?atoi(argv[1]):10;
    size_t c;
    SetProcessAffinityMask(GetCurrentProcess(),0x4);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    setvbuf(stdout,NULL,_IONBF,0);
    printf("BACKWARD _ct vs direct — the twins that were never raced\n");
    printf("  +%% = _ct faster. control = baseline timed twice (the floor)\n");
    printf("  predicted by spill: R=9 LOSES (0%% spill), R=25/27 win big\n\n");
    for(c=0;c<3;c++){
#define ARM(R) \
        race("leaf n1t",R,radix##R##_z_n1t_bwd_avx2,radix##R##_z_n1t_ct_bwd_avx2,CS[c],0,pace); \
        race("mid  t2 ",R,radix##R##_z_t2_bwd_avx2, radix##R##_z_t2_ct_bwd_avx2, CS[c],1,pace);
        ARM(9) ARM(15) ARM(21) ARM(25) ARM(27)
#undef ARM
        printf("\n");
    }
    return g_fail;
}
