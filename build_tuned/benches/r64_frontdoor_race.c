/* r64_frontdoor_race.c — does R64 blocked reach the front door, and is it
 * still correct there?
 *
 * The kernel race said blocked wins 6/6 at radix 64 (+13% to +155%). That is
 * a kernel-level result; this asks the two questions that decide whether it
 * SHIPS:
 *
 *   1. CORRECTNESS through vfft_create/vfft_execute, against an independent
 *      O(N^2) long-double DFT. Blocked PASS 2 is the generic-ctw path, so
 *      this is a tolerance gate, not memcmp.
 *   2. Does it show up as TIME at the front door, where the kernel is one
 *      stage of a pair and plan overhead is included?
 *
 * A/B AXIS: VFFT_NO_ILBLK, read inside apply_blocked_default at CREATE. Both
 * handles are therefore built up front in ONE process and timed alternately
 * -- cross-process comparison has produced wrong verdicts on this host.
 *
 * N chosen so that 64 is one factor of the Bailey pair: 256 = 4x64,
 * 512 = 8x64, 1024 = 16x64. Below 128 and above 1024 the pair tier does not
 * apply and the arms would be identical by construction.
 *
 * Build: python build.py --src benches/r64_frontdoor_race.c --vfft --compile
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

#define TRIALS 21
static double best(const double *v,int n)
{ int i; double lo=v[0]; for(i=1;i<n;i++) if(v[i]<lo) lo=v[i]; return lo; }

/* independent reference: O(N^2) in long double, so it shares no code with
 * anything under test */
static double ref_relmax(const double *zin, const double *zout, int N)
{
    int k,j; double worst=0, mag=0;
    for (k = 0; k < N; k++) {
        long double sr=0, si=0;
        for (j = 0; j < N; j++) {
            long double th = -2.0L*3.14159265358979323846264338328L*(long double)k*(long double)j/(long double)N;
            long double c=cosl(th), s=sinl(th);
            sr += (long double)zin[2*j]*c - (long double)zin[2*j+1]*s;
            si += (long double)zin[2*j]*s + (long double)zin[2*j+1]*c;
        }
        { double dr=fabs((double)sr - zout[2*k]), di=fabs((double)si - zout[2*k+1]);
          double m = fabs((double)sr) > fabs((double)si) ? fabs((double)sr) : fabs((double)si);
          if (dr>worst) worst=dr; if (di>worst) worst=di; if (m>mag) mag=m; }
    }
    return mag>0 ? worst/mag : worst;
}

static vfft_plan build(int N, int noblk)
{
    vfft_config_t cfg; vfft_plan p;
    memset(&cfg,0,sizeof cfg);
    cfg.transform=VFFT_C2C; cfg.placement=VFFT_OUTOFPLACE;
    cfg.rigor=VFFT_MEASURE; cfg.dims=1; cfg.n[0]=N; cfg.howmany=1;
    cfg.layout=VFFT_LAYOUT_INTERLEAVED; cfg.order=VFFT_ORDER_NATURAL;
    cfg.nthreads=1;
    _putenv_s("VFFT_NO_ILBLK", noblk ? "1" : "");
    p = vfft_create(&cfg);
    _putenv_s("VFFT_NO_ILBLK", "");
    return p;
}

static void cell(int N, int pace_ms, int check)
{
    vfft_plan pm = build(N,1);   /* monolithic: blocked defaults disabled */
    vfft_plan pb = build(N,0);   /* blocked defaults active               */
    double *zin,*zm,*zb,tm[TRIALS],tb[TRIALS],tc[TRIALS];
    int i,k,r,reps; double em=0, eb=0;

    if(!pm||!pb){ printf("  N=%-5d create failed\n",N); return; }
    zin=(double*)calloc(2*(size_t)N+8,sizeof(double));
    zm =(double*)calloc(2*(size_t)N+8,sizeof(double));
    zb =(double*)calloc(2*(size_t)N+8,sizeof(double));
    for(i=0;i<2*N;i++) zin[i]=rnd();

    vfft_execute(pm,VFFT_FORWARD,zin,NULL,zm,NULL);
    vfft_execute(pb,VFFT_FORWARD,zin,NULL,zb,NULL);
    if(check){ em=ref_relmax(zin,zm,N); eb=ref_relmax(zin,zb,N);
        if(!(em<1e-13 && eb<1e-13)){
            printf("  N=%-5d *** WRONG vs O(N^2) reference: mono %.2e blocked %.2e ***\n",
                   N,em,eb); goto done; } }

    reps=(int)(4000000.0/(double)N); if(reps<200)reps=200; if(reps>20000)reps=20000;
    for(k=0;k<50;k++){ vfft_execute(pm,VFFT_FORWARD,zin,NULL,zm,NULL);
                       vfft_execute(pb,VFFT_FORWARD,zin,NULL,zb,NULL); }
    for(k=0;k<TRIALS;k++){
        double t=now_ns();
        for(r=0;r<reps;r++) vfft_execute(pm,VFFT_FORWARD,zin,NULL,zm,NULL);
        tm[k]=(now_ns()-t)/reps; if(pace_ms) Sleep((DWORD)pace_ms);
        t=now_ns();
        for(r=0;r<reps;r++) vfft_execute(pb,VFFT_FORWARD,zin,NULL,zb,NULL);
        tb[k]=(now_ns()-t)/reps; if(pace_ms) Sleep((DWORD)pace_ms);
        t=now_ns();   /* control: the monolithic plan again */
        for(r=0;r<reps;r++) vfft_execute(pm,VFFT_FORWARD,zin,NULL,zm,NULL);
        tc[k]=(now_ns()-t)/reps; if(pace_ms) Sleep((DWORD)pace_ms);
    }
    {
        double M=best(tm,TRIALS), B=best(tb,TRIALS), C=best(tc,TRIALS);
        printf("  N=%-5d mono %8.1f ns | blocked %8.1f ns | %+6.1f%% | control %+5.1f%%",
               N,M,B,100.0*(M/B-1.0),100.0*(M/C-1.0));
        if(check) printf(" | ref %.1e/%.1e",em,eb);
        printf("\n");
    }
done:
    free(zin); free(zm); free(zb);
    vfft_destroy(pm); vfft_destroy(pb);
}

int main(int argc,char**argv)
{
    static const int NS[]={256,512,1024};
    int pace=(argc>1)?atoi(argv[1]):15;
    size_t i;
    SetProcessAffinityMask(GetCurrentProcess(),0x4);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    setvbuf(stdout,NULL,_IONBF,0);
    printf("R64 blocked at the FRONT DOOR (VFFT_NO_ILBLK is the A/B axis)\n");
    printf("  +%% = blocked faster. control = the monolithic plan timed twice\n");
    printf("  correctness vs an independent O(N^2) long-double DFT\n\n");
    for(i=0;i<sizeof NS/sizeof NS[0];i++) cell(NS[i],pace,1);
    return 0;
}
