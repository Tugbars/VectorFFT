/* r64_blocked_race.c — R64 forward: monolithic vs blocked (8.8 and 4.16).
 *
 * WHY. Radix 64 is the worst spiller in the tree. Measured on compiled
 * assembly, bulk loop only:
 *
 *     n1t  mono  1609 insns  723 spill  44.9%      t2  mono  1771  737  41.6%
 *     n1tb88     1197        243        20.3%      t2b88     1323  239  18.1%
 *     n1tb416    1215        289        23.8%      t2b416    1406  346  24.6%
 *
 * The shipped policy is "R>=32 blocked structurally" (R32 measured +17..+52%),
 * yet FORWARD R64 had no blocked variant at all -- only the backward side did
 * (n1b88/n1b416, t2bt88/t2bt416). These four are the missing forward twins.
 *
 * CORRECTNESS. Blocked PASS 2 goes through the generic ctw path, so the
 * result is NOT bit-identical to the monolithic form -- the R32 precedent
 * records relmax 3.1e-16 and gates at tol 1e-12, not memcmp. Same here.
 *
 * INSTRUMENT. Two lessons from this host, both paid for:
 *   - MIN of N trials, not median. Timing noise is one-sided (interrupts,
 *     migration, frequency dips only ADD time), so the fastest trial is the
 *     least contaminated estimate. Median gave sign-unstable results.
 *   - A CONTROL arm: the monolithic kernel timed a SECOND time, from a
 *     second call site. Whatever gap that shows is placement luck and drift,
 *     never a real effect. An arm only counts if it clears the control.
 *
 * SHAPES. Leaf n1t is OOP as il2p calls it (zin -> scratch, OLs = R); mid t2
 * is IN-PLACE, matching mid_f(zout,0,zout,0,...). Getting that wrong changes
 * the answer.
 *
 * Build: python build.py --src benches/r64_blocked_race.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <windows.h>

#define R 64
typedef void (*kfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);

void radix64_z_n1t_fwd_avx2    (const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix64_z_n1tb88_fwd_avx2 (const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix64_z_n1tb416_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix64_z_t2_fwd_avx2     (const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix64_z_t2b88_fwd_avx2  (const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix64_z_t2b416_fwd_avx2 (const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);

static double now_ns(void)
{ static LARGE_INTEGER f; static int i2=0; LARGE_INTEGER c;
  if(!i2){QueryPerformanceFrequency(&f);i2=1;} QueryPerformanceCounter(&c);
  return (double)c.QuadPart*1e9/(double)f.QuadPart; }
static uint64_t lcg = 0x243F6A8885A308D3ull;
static double rnd(void)
{ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
  return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

#define TRIALS 25
static double best(const double *v,int n)
{ int i; double lo=v[0]; for(i=1;i<n;i++) if(v[i]<lo) lo=v[i]; return lo; }
static double spread(const double*v,int n)
{ double lo=v[0],hi=v[0];int i;
  for(i=1;i<n;i++){if(v[i]<lo)lo=v[i];if(v[i]>hi)hi=v[i];}
  return lo>0?hi/lo-1.0:0.0; }

/* relative max difference, so the tolerance gate is scale-free */
static double relmax(const double*a,const double*b,size_t n)
{ size_t i; double w=0,mag=0;
  for(i=0;i<n;i++){ double d=fabs(a[i]-b[i]);
      if(d>w)w=d; if(fabs(a[i])>mag)mag=fabs(a[i]); }
  return mag>0?w/mag:w; }

static void race(const char *what, kfn mono, kfn b88, kfn b416,
                 size_t count, int inplace, int pace_ms)
{
    const size_t Ls = count, OLs = inplace ? count : (size_t)R;
    const size_t groups = (count + 1) / 2;
    const size_t ntw = groups * (size_t)(R - 1) * 8u + 64u;
    const size_t n = 2 * (size_t)R * count + 64;
    double *zin=(double*)_aligned_malloc(n*sizeof(double),64);
    double *z0 =(double*)_aligned_malloc(n*sizeof(double),64);
    double *z1 =(double*)_aligned_malloc(n*sizeof(double),64);
    double *z2 =(double*)_aligned_malloc(n*sizeof(double),64);
    double *tw =(double*)_aligned_malloc(ntw*sizeof(double),64);
    double t0a[TRIALS],t1a[TRIALS],t2a[TRIALS],tca[TRIALS];
    size_t i; int k,r,reps; double e1,e2;

    for(i=0;i<n;i++) zin[i]=rnd();
    for(i=0;i<ntw;i++){ double th=0.37*(double)i; tw[i]=((i>>2)&1)?sin(th):cos(th); }

#define CALL(F,DST) do{ if(inplace){ memcpy(DST,zin,n*sizeof(double)); \
                            F(DST,0,DST,0,tw,0,Ls,0,OLs,0,count); } \
                        else F(zin,0,DST,0,tw,0,Ls,0,OLs,0,count); }while(0)
    /* The kernels DEFINE exactly 2*R*count doubles; the +64 slack is
     * untouched and _aligned_malloc does not zero it. Compare the defined
     * region only, and zero first so a short write would still show up as a
     * mismatch rather than as whatever the allocator recycled. */
    memset(z0,0,n*sizeof(double));
    memset(z1,0,n*sizeof(double));
    memset(z2,0,n*sizeof(double));
    CALL(mono,z0); CALL(b88,z1); CALL(b416,z2);
    {
        const size_t nd = 2 * (size_t)R * count;   /* the defined region */
        e1=relmax(z0,z1,nd); e2=relmax(z0,z2,nd);
    }
    if(!(e1<1e-12 && e2<1e-12)){
        printf("  %-8s count=%-3zu *** CORRECTNESS FAIL: 8.8 rel %.2e, 4.16 rel %.2e ***\n",
               what,count,e1,e2);
        goto done;
    }

    reps=(int)(4000000.0/((double)R*count)); if(reps<500)reps=500; if(reps>40000)reps=40000;
    for(k=0;k<300;k++){ CALL(mono,z0); CALL(b88,z1); CALL(b416,z2); }
    for(k=0;k<TRIALS;k++){
        double t=now_ns();
        for(r=0;r<reps;r++) CALL(mono,z0);
        t0a[k]=(now_ns()-t)/reps; if(pace_ms) Sleep((DWORD)pace_ms);
        t=now_ns();
        for(r=0;r<reps;r++) CALL(b88,z1);
        t1a[k]=(now_ns()-t)/reps; if(pace_ms) Sleep((DWORD)pace_ms);
        t=now_ns();
        for(r=0;r<reps;r++) CALL(b416,z2);
        t2a[k]=(now_ns()-t)/reps; if(pace_ms) Sleep((DWORD)pace_ms);
        t=now_ns();   /* CONTROL: monolithic again, second call site */
        for(r=0;r<reps;r++) CALL(mono,z0);
        tca[k]=(now_ns()-t)/reps; if(pace_ms) Sleep((DWORD)pace_ms);
    }
#undef CALL
    {
        double M=best(t0a,TRIALS),A=best(t1a,TRIALS),B=best(t2a,TRIALS),C=best(tca,TRIALS);
        double ctl=100.0*(M/C-1.0);
        printf("  %-8s count=%-3zu | mono %8.1f | 8.8 %8.1f (%+6.1f%%) | 4.16 %8.1f (%+6.1f%%)"
               " | control %+5.1f%% | rel %.1e/%.1e | sp %.0f%%\n",
               what,count,M,A,100.0*(M/A-1.0),B,100.0*(M/B-1.0),ctl,e1,e2,
               100*spread(t0a,TRIALS));
    }
done:
    _aligned_free(zin);_aligned_free(z0);_aligned_free(z1);_aligned_free(z2);_aligned_free(tw);
}

int main(int argc,char**argv)
{
    static const size_t CS[]={8,16,32};
    int pace=(argc>1)?atoi(argv[1]):12;
    size_t c;
    SetProcessAffinityMask(GetCurrentProcess(),0x4);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    setvbuf(stdout,NULL,_IONBF,0);
    printf("radix 64 FORWARD: monolithic vs blocked. +%% = blocked faster\n");
    printf("  mono spills 44.9%% (leaf) / 41.6%% (mid); blocked 8.8 cuts it to ~20%%/18%%\n");
    printf("  pinned core 2, HIGH, BEST of %d, control = mono vs ITSELF\n\n",TRIALS);
    for(c=0;c<3;c++){
        race("leaf n1t",radix64_z_n1t_fwd_avx2,radix64_z_n1tb88_fwd_avx2,
             radix64_z_n1tb416_fwd_avx2,CS[c],0,pace);
        race("mid  t2 ",radix64_z_t2_fwd_avx2, radix64_z_t2b88_fwd_avx2,
             radix64_z_t2b416_fwd_avx2, CS[c],1,pace);
    }
    return 0;
}
