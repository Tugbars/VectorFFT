/* r16_threshold_race.c — is radix 16 above or below the blocking threshold?
 *
 * The register-pressure curve puts R=16 exactly at the crossover: it is the
 * first pow2 radix to saturate the 16 ymm registers and spill at all, but it
 * spills only 7 of 185 loop instructions (3.8%... 7.6% counting both halves of
 * each round trip), against 26.5% at R=32 and 34.8% at R=64. Blocking halves
 * it -- 7 spill ops to 3 -- but half of a small number is a small number.
 *
 * That is why the shipped policy makes MONOLITHIC the default at R=16 while
 * R>=32 defaults to blocked: "R=16 fits the file, so a non-monolithic form
 * must win per cell, not by structural rule". The banked evidence for that is
 * a 24-arm ranking at N=512 (4.4 = 362 ns < 2.8 = 367 < mono = 373 < 8.2 =
 * 376), i.e. blocked ahead by ~3% -- close enough that placement luck matters.
 *
 * This re-measures it directly, both kinds, three counts, so the threshold is
 * a current measurement rather than an inherited one. R=32 runs alongside as
 * the control: it is the radix the structural rule was built for, and it
 * should win decisively if the instrument is working.
 *
 * IN-PLACE for the mid, matching il2p's mid_f(zout,0,zout,0,...) call; the
 * leaf is out-of-place as il2p calls it (zin -> scratch). Getting that wrong
 * changes the answer -- an out-of-place mid measured differently earlier.
 *
 * Build (from build_tuned/benches):
 *   gcc -O3 -mavx2 -mfma -march=native -o r16_threshold_race.exe \
 *       r16_threshold_race.c -L. -l:libdagcodelets.a   (or list the .c files)
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

/* leaf: n1t (turned store, twiddle-free) — OOP, as il2p calls it */
void radix16_z_n1t_fwd_avx2   (const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix16_z_n1tb44_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix32_z_n1t_fwd_avx2   (const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix32_z_n1tb48_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
/* mid: t2 (streamed VTW2) — IN-PLACE */
void radix16_z_t2_fwd_avx2 (const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix16_z_t2b_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix32_z_t2_fwd_avx2 (const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix32_z_t2b48_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);

static double now_ns(void)
{ static LARGE_INTEGER f; static int i2=0; LARGE_INTEGER c;
  if(!i2){QueryPerformanceFrequency(&f);i2=1;} QueryPerformanceCounter(&c);
  return (double)c.QuadPart*1e9/(double)f.QuadPart; }
static uint64_t lcg = 0x243F6A8885A308D3ull;
static double rnd(void)
{ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
  return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

#define TRIALS 9
static double med(double *v,int n)
{ int i,j; for(i=1;i<n;i++){double k=v[i];j=i-1;
    while(j>=0&&v[j]>k){v[j+1]=v[j];j--;}v[j+1]=k;} return v[n/2]; }
static double spread(const double*v,int n)
{ double lo=v[0],hi=v[0];int i;
  for(i=1;i<n;i++){if(v[i]<lo)lo=v[i];if(v[i]>hi)hi=v[i];}
  return lo>0?hi/lo-1.0:0.0; }

static void race(const char *what, int R, kfn mono, kfn blk,
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
    double ta[TRIALS], tb[TRIALS];
    size_t i; int k,r,reps;

    for(i=0;i<n;i++) zin[i]=rnd();
    for(i=0;i<ntw;i++){ double th=0.37*(double)i; tw[i]=((i>>2)&1)?sin(th):cos(th); }
    reps=(int)(2000000.0/((double)R*count)); if(reps<400)reps=400; if(reps>20000)reps=20000;

#define CALL(F,DST) do{ if(inplace) F(DST,0,DST,0,tw,0,Ls,0,OLs,0,count); \
                        else        F(zin,0,DST,0,tw,0,Ls,0,OLs,0,count); }while(0)
    for(k=0;k<300;k++){ CALL(mono,za); CALL(blk,zb); }
    for(k=0;k<TRIALS;k++){
        double t0=now_ns();
        for(r=0;r<reps;r++) CALL(mono,za);
        ta[k]=(now_ns()-t0)/reps;
        if(pace_ms) Sleep((DWORD)pace_ms);
        t0=now_ns();
        for(r=0;r<reps;r++) CALL(blk,zb);
        tb[k]=(now_ns()-t0)/reps;
        if(pace_ms) Sleep((DWORD)pace_ms);
    }
#undef CALL
    {
        double A=med(ta,TRIALS), B=med(tb,TRIALS);
        double sa=100*spread(ta,TRIALS), sb=100*spread(tb,TRIALS);
        double gap=100.0*(A/B-1.0);
        printf("  %-10s r%-3d count=%-3zu | mono %8.2f ns (sp %4.1f%%) | blocked %8.2f ns (sp %4.1f%%) | %+6.1f%% %s\n",
               what,R,count,A,sa,B,sb,gap,
               fabs(gap) < (sa+sb)/2 ? "INSIDE NOISE" : (gap>0?"blocked":"MONO"));
    }
    _aligned_free(zin);_aligned_free(za);_aligned_free(zb);_aligned_free(tw);
}

int main(int argc,char**argv)
{
    static const size_t CS[]={8,16,32};
    int pace=(argc>1)?atoi(argv[1]):15;
    size_t c;
    SetProcessAffinityMask(GetCurrentProcess(),0x4);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    setvbuf(stdout,NULL,_IONBF,0);
    printf("radix 16 vs 32: is blocking worth it? (+%% = blocked faster)\n");
    printf("  R16 spills 7 ops of 185; R32 spills 80 of 604 — R32 is the control\n");
    printf("  pinned core 2, HIGH, medians of %d, alternated\n\n",TRIALS);
    for(c=0;c<3;c++){
        race("leaf n1t",16,radix16_z_n1t_fwd_avx2,radix16_z_n1tb44_fwd_avx2,CS[c],0,pace);
        race("leaf n1t",32,radix32_z_n1t_fwd_avx2,radix32_z_n1tb48_fwd_avx2,CS[c],0,pace);
        race("mid  t2 ",16,radix16_z_t2_fwd_avx2, radix16_z_t2b_fwd_avx2,   CS[c],1,pace);
        race("mid  t2 ",32,radix32_z_t2_fwd_avx2, radix32_z_t2b48_fwd_avx2, CS[c],1,pace);
        printf("\n");
    }
    return 0;
}
