/* nonpow2_tier_probe.c — what does a non-pow2 N >= 2048 actually cost?
 *
 * The cascade (the N >= 2048 tier) only accepts chains of {4,8}: zsplit wants
 * chain[0] in {4,8}, mids in {4,8}, last == 8; zturn allows last in {4,8}.
 * So N must be a pure product of 4s and 8s. Every other N >= 2048 -- 6144,
 * 10240, 12288, anything with a 3 or 5 in it -- falls to a slower tier.
 *
 * That is the real content of the cascade's "count % 4 == 0" contract. The
 * contract itself is unreachable-by-construction (D[] is a product of {4,8},
 * and zturn.h:596 guards it), so it costs nothing directly; what costs is the
 * chain fence that makes it true.
 *
 * ns/point is the comparable quantity across different N. A pure-pow2 N and
 * its non-pow2 neighbour should cost roughly the same per point if both are
 * served well; a step means the non-pow2 one fell off the tier.
 *
 * Build: python build.py --src benches/nonpow2_tier_probe.c --vfft --compile
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

#define TRIALS 15
static double best(const double *v,int n)
{ int i; double lo=v[0]; for(i=1;i<n;i++) if(v[i]<lo) lo=v[i]; return lo; }

static int is_pow2(int n){ return n && !(n & (n-1)); }

/* factor N over {4,8} only -- the cascade's fence */
static int cascade_expressible(int n)
{
    while (n % 8 == 0) n /= 8;
    while (n % 4 == 0) n /= 4;
    return n == 1;
}

static void cell(int N, int pace)
{
    vfft_config_t cfg; vfft_plan p;
    double *zin,*zo,t[TRIALS]; int i,k,r,reps;
    memset(&cfg,0,sizeof cfg);
    cfg.transform=VFFT_C2C; cfg.placement=VFFT_OUTOFPLACE;
    cfg.rigor=VFFT_MEASURE; cfg.dims=1; cfg.n[0]=N; cfg.howmany=1;
    cfg.layout=VFFT_LAYOUT_INTERLEAVED; cfg.order=VFFT_ORDER_NATURAL;
    cfg.nthreads=1;
    p = vfft_create(&cfg);
    if(!p){ printf("  N=%-7d create FAILED\n",N); return; }

    zin=(double*)calloc(2*(size_t)N+8,sizeof(double));
    zo =(double*)calloc(2*(size_t)N+8,sizeof(double));
    for(i=0;i<2*N;i++) zin[i]=rnd();

    reps=(int)(8000000.0/(double)N); if(reps<50)reps=50; if(reps>4000)reps=4000;
    for(k=0;k<30;k++) vfft_execute(p,VFFT_FORWARD,zin,NULL,zo,NULL);
    for(k=0;k<TRIALS;k++){
        double t0=now_ns();
        for(r=0;r<reps;r++) vfft_execute(p,VFFT_FORWARD,zin,NULL,zo,NULL);
        t[k]=(now_ns()-t0)/reps;
        if(pace) Sleep((DWORD)pace);
    }
    {
        double B=best(t,TRIALS);
        printf("  N=%-7d %10.1f ns  %7.3f ns/pt   %-9s %s\n",
               N, B, B/(double)N,
               is_pow2(N) ? "pow2" : "non-pow2",
               cascade_expressible(N) ? "cascade-legal" : "NOT cascade-legal");
    }
    free(zin); free(zo); vfft_destroy(p);
}

int main(int argc,char**argv)
{
    /* pow2 cascade cells, and their nearest non-pow2 neighbours */
    static const int NS[]={2048,3072,4096,6144,8192,10240,12288,16384};
    int pace=(argc>1)?atoi(argv[1]):15;
    size_t i;
    SetProcessAffinityMask(GetCurrentProcess(),0x4);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    setvbuf(stdout,NULL,_IONBF,0);
    printf("non-pow2 at the cascade tier: what the {4,8} chain fence costs\n");
    printf("  ns/pt is the comparable quantity across sizes\n\n");
    for(i=0;i<sizeof NS/sizeof NS[0];i++) cell(NS[i],pace);
    return 0;
}
