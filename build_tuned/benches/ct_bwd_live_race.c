/* ct_bwd_live_race.c — _ct on the kinds the BACKWARD route actually runs.
 *
 * BACKGROUND. The first backward _ct race (ct_bwd_race.c) showed _ct beating
 * its baseline by 20-155% at radices 15/21/25/27. But those kernels were
 * n1t_ct_bwd / t2_ct_bwd, i.e. variants of leaf_b / mid_b -- and BOTH slots
 * are dead:
 *     mid_b   assigned at create, never called.
 *     leaf_b  only reached by execute_bwd_fdiag, which is the FALLBACK arm of
 *             execute_bwd; execute_bwd_t2t returns -1 only when t2t_b or
 *             n1_b_r2 is NULL, and both cover all 20 radices, so it never
 *             fails and F-DIAG never runs.
 * The win was real and unreachable.
 *
 * THIS RACE is the same idea aimed at the LIVE pair:
 *     t2t_b    = t2t_bwd(R1)   turned, twiddled, out-of-place
 *     n1_b_r2  = n1_bwd(R2)    twiddle-free, in-place identity map
 * Those slots already have a variant axis (vfft_il2p_t2t_bwd_v_fn /
 * n1_bwd_v_fn) that _il_dp_race_bwd already walks, so a winner here is
 * raceable and bankable with no planner edit.
 *
 * STATIC PREDICTION (compiled asm, bulk loop only):
 *     R=25 t2t_bwd 1104 insns 49.1% spill -> _ct  558 insns 20.4%
 *     R=25 n1_bwd  1032 insns 53.4% spill -> _ct  486 insns 30.2%
 *     R=15 t2t_bwd  303 insns 18.2% spill -> _ct  253 insns  8.3%
 *
 * SHAPES, taken from the real call sites in vfft_il2p_execute_bwd_t2t:
 *     t2t_b  (zin,0,zout,0,twb,0, R2,0, R1,0, R2)  -> Ls = count, OLs = R, OOP
 *     n1_b_r2(zout,0,zout,0, 0,0, R1,0, R1,0, R1)  -> Ls = OLs = count, in-place
 * Both read R legs, so the plane must be sized 2*R*count.
 *
 * Radix 9 is absent: it spills 0.0% and _ct lost 5 of 6 cells there.
 *
 * Build: python build.py --src benches/ct_bwd_live_race.c --compile
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
#define PAIR(R)                            \
  DECL(radix##R##_z_t2t_bwd_avx2)          \
  DECL(radix##R##_z_t2t_ct_bwd_avx2)       \
  DECL(radix##R##_z_n1_bwd_avx2)           \
  DECL(radix##R##_z_n1_ct_bwd_avx2)
PAIR(15) PAIR(21) PAIR(25) PAIR(27)
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
    const size_t n = 2 * (size_t)R * count + 64;
    double *zin=(double*)_aligned_malloc(n*sizeof(double),64);
    double *za =(double*)_aligned_malloc(n*sizeof(double),64);
    double *zb =(double*)_aligned_malloc(n*sizeof(double),64);
    double *tw =(double*)_aligned_malloc(ntw*sizeof(double),64);
    double ta[TRIALS], tb[TRIALS], tc[TRIALS];
    size_t i; int k,r,reps; double rel;

    for(i=0;i<n;i++) zin[i]=rnd();
    for(i=0;i<ntw;i++){ double th=0.37*(double)i; tw[i]=((i>>2)&1)?sin(th):cos(th); }

#define CALL(F,DST) do{ if(inplace){ memcpy(DST,zin,n*sizeof(double));      \
                            F(DST,0,DST,0,tw,0,Ls,0,OLs,0,count); }         \
                        else { memset(DST,0,n*sizeof(double));              \
                            F(zin,0,DST,0,tw,0,Ls,0,OLs,0,count); } }while(0)
    CALL(base,za); CALL(ct,zb);
    rel = relmax(za,zb,2*(size_t)R*count);
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
        t=now_ns();
        for(r=0;r<reps;r++) CALL(base,za);
        tc[k]=(now_ns()-t)/reps; if(pace_ms) Sleep((DWORD)pace_ms);
    }
#undef CALL
    {
        double A=best(ta,TRIALS), B=best(tb,TRIALS), C=best(tc,TRIALS);
        double gap=100.0*(A/B-1.0), ctl=100.0*(A/C-1.0);
        const char *v = (fabs(gap) < 2.0*fabs(ctl) || fabs(gap) < 1.0)
                          ? "--" : (gap > 0 ? "_ct" : "BASELINE");
        printf("  %-9s R=%-2d count=%-3zu | base %8.2f | _ct %8.2f | %+7.1f%% | ctl %+5.1f%% | %s\n",
               what,R,count,A,B,gap,ctl,v);
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
    printf("_ct on the LIVE backward pair (t2t_b, n1_b_r2)\n");
    printf("  +%% = _ct faster. control = baseline timed twice (the floor)\n\n");
    for(c=0;c<3;c++){
#define ARM(R) \
        race("t2t (s1)",R,radix##R##_z_t2t_bwd_avx2,radix##R##_z_t2t_ct_bwd_avx2,CS[c],0,pace); \
        race("n1  (s2)",R,radix##R##_z_n1_bwd_avx2, radix##R##_z_n1_ct_bwd_avx2, CS[c],1,pace);
        ARM(15) ARM(21) ARM(25) ARM(27)
#undef ARM
        printf("\n");
    }
    return g_fail;
}
