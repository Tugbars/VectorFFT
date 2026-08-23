/* il_pair_verdict.c — settle a K=1 IL PAIR choice in ONE process.
 *
 * THE PROBLEM. Five independent calibration runs at N=1024 chose 16.64 three
 * times and 32.32 twice, with per-run times 798-862 ns. Those numbers are
 * from five different processes, and on this host cross-run times are not
 * comparable -- which is exactly the comparison that would be needed to say
 * which pair is faster. VFFT_IL_KV can force the FORM nibbles but there is no
 * env that forces the PAIR, so the usual same-process trick does not reach.
 *
 * THE ROUTE IN. config.wisdom is a public, caller-owned override: a plan
 * built with it uses THAT bundle exclusively (vfft.h:172). So two wisdom
 * directories -- each holding one of the two contested plans -- can be loaded
 * side by side and their plans timed alternately in one process, with the
 * machine in one thermal state.
 *
 * CORRECTNESS FIRST: both plans compute the same transform, so they must
 * agree elementwise. A mismatch means one is wrong, not that the timing is
 * interesting.
 *
 * ESTIMATOR: minimum of N trials, not median. Timing noise here is one-sided
 * (interrupts, migration, frequency dips only ADD time), so the fastest trial
 * is the least contaminated estimate. Median gave sign-unstable results on
 * this machine. The CONTROL arm times plan A a second time from a second call
 * site: whatever gap it shows is drift, and the real arm must clear it.
 *
 * Usage: il_pair_verdict.exe <dirA> <dirB> <N> [pace_ms]
 * Build: python build.py --src benches/il_pair_verdict.c --vfft --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <windows.h>
#include "vfft.h"

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

static vfft_plan build(int N, vfft_wisdom *w)
{
    vfft_config_t cfg; memset(&cfg,0,sizeof cfg);
    cfg.transform=VFFT_C2C; cfg.placement=VFFT_OUTOFPLACE;
    cfg.rigor=VFFT_MEASURE; cfg.dims=1; cfg.n[0]=N; cfg.howmany=1;
    cfg.layout=VFFT_LAYOUT_INTERLEAVED; cfg.order=VFFT_ORDER_NATURAL;
    cfg.nthreads=1;
    cfg.wisdom=w;          /* THAT bundle exclusively — vfft.h:172 */
    cfg.wisdom_write=0;    /* serving mode: never mutate the store */
    return vfft_create(&cfg);
}

int main(int argc,char**argv)
{
    const char *dA = (argc>1)?argv[1]:".";
    const char *dB = (argc>2)?argv[2]:".";
    const int   N  = (argc>3)?atoi(argv[3]):1024;
    const int pace = (argc>4)?atoi(argv[4]):15;
    vfft_wisdom *wA,*wB; vfft_plan pa,pb;
    double *zin,*a,*b,ta[TRIALS],tb[TRIALS],tc[TRIALS];
    int i,k,r,reps; double w=0,mag=0;

    SetProcessAffinityMask(GetCurrentProcess(),0x4);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    setvbuf(stdout,NULL,_IONBF,0);

    wA=vfft_wisdom_load(dA); wB=vfft_wisdom_load(dB);
    if(!wA||!wB){ printf("wisdom load failed (%p/%p)\n",(void*)wA,(void*)wB); return 1; }
    pa=build(N,wA); pb=build(N,wB);
    if(!pa||!pb){ printf("create failed (%p/%p)\n",(void*)pa,(void*)pb); return 1; }

    printf("N=%d  plan A from %s  vs  plan B from %s\n",N,dA,dB);
    printf("  +%% = B faster. control = plan A timed twice. BEST of %d, alternated\n\n",TRIALS);

    zin=(double*)calloc(2*(size_t)N+8,sizeof(double));
    a  =(double*)calloc(2*(size_t)N+8,sizeof(double));
    b  =(double*)calloc(2*(size_t)N+8,sizeof(double));
    for(i=0;i<2*N;i++) zin[i]=rnd();

    vfft_execute(pa,VFFT_FORWARD,zin,NULL,a,NULL);
    vfft_execute(pb,VFFT_FORWARD,zin,NULL,b,NULL);
    for(i=0;i<2*N;i++){ double d=fabs(a[i]-b[i]);
        if(d>w)w=d; if(fabs(a[i])>mag)mag=fabs(a[i]); }
    if(!((mag>0?w/mag:w) < 1e-12)){
        printf("  *** PLANS DISAGREE (rel %.2e) — not the same transform ***\n",
               mag>0?w/mag:w); return 1; }

    reps=(int)(4000000.0/(double)N); if(reps<500)reps=500;
    for(k=0;k<50;k++){ vfft_execute(pa,VFFT_FORWARD,zin,NULL,a,NULL);
                       vfft_execute(pb,VFFT_FORWARD,zin,NULL,b,NULL); }
    for(k=0;k<TRIALS;k++){
        double t=now_ns();
        for(r=0;r<reps;r++) vfft_execute(pa,VFFT_FORWARD,zin,NULL,a,NULL);
        ta[k]=(now_ns()-t)/reps; if(pace) Sleep((DWORD)pace);
        t=now_ns();
        for(r=0;r<reps;r++) vfft_execute(pb,VFFT_FORWARD,zin,NULL,b,NULL);
        tb[k]=(now_ns()-t)/reps; if(pace) Sleep((DWORD)pace);
        t=now_ns();
        for(r=0;r<reps;r++) vfft_execute(pa,VFFT_FORWARD,zin,NULL,a,NULL);
        tc[k]=(now_ns()-t)/reps; if(pace) Sleep((DWORD)pace);
    }
    {
        double A=best(ta,TRIALS),B=best(tb,TRIALS),C=best(tc,TRIALS);
        printf("  A %8.1f ns | B %8.1f ns | %+6.2f%% | control %+5.2f%% | rel %.1e\n",
               A,B,100.0*(A/B-1.0),100.0*(A/C-1.0), mag>0?w/mag:w);
    }
    free(zin); free(a); free(b);
    vfft_destroy(pa); vfft_destroy(pb);
    vfft_wisdom_free(wA); vfft_wisdom_free(wB);
    return 0;
}
