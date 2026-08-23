/* zturn_vs_legacy_race.c — is the ZTURN cascade route actually faster than the
 * legacy one it replaced?
 *
 * WHAT DIFFERS. The two routes share the MID: msg is byte-identical on both
 * (zsplit.h and zturn.h call the same kernel; only the twiddle table repacks).
 * So the whole difference is the INGEST and the TERMINATOR:
 *
 *   legacy : s0s  -> msg xk -> sterm | sterm2      (terminator reads via
 *            E_blocks with TR4 register transposes)
 *   ZTURN  : s0t  -> msg xk -> stf | stf2 | stfn   (terminator reads 4 section
 *            taps, 128 B contiguous, NO load shuffles)
 *
 * The design claim is that s0t STORES in exactly the geometry stf wants to
 * READ, which deletes sterm's load-side TR4 entirely. This measures whether
 * that claim shows up as time at the front door.
 *
 * METHOD. VFFT_FORCE_ZROUTE is read at CREATE (vfft.c:2864), so both handles
 * are built up front in ONE process -- legacy forced on one, zturn on the
 * other -- and then timed alternately. Cross-process comparison has produced
 * two wrong verdicts on this host today; same-process alternating is the only
 * form that resolves anything here.
 *
 * CORRECTNESS FIRST: the two routes must agree elementwise. They are different
 * decompositions of the same transform, so a mismatch means one of them is
 * wrong, not that the timing is interesting. Note both are NATURAL order here
 * so an elementwise compare is meaningful (a roundtrip could not gate this --
 * it holds under any self-consistent permutation).
 *
 * Build: python build.py --src benches/zturn_vs_legacy_race.c --vfft --compile
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

#define TRIALS 9
static double med(double *v,int n)
{ int i,j; for(i=1;i<n;i++){double k=v[i];j=i-1;
    while(j>=0&&v[j]>k){v[j+1]=v[j];j--;}v[j+1]=k;} return v[n/2]; }
static double spread(const double*v,int n)
{ double lo=v[0],hi=v[0];int i;
  for(i=1;i<n;i++){if(v[i]<lo)lo=v[i];if(v[i]>hi)hi=v[i];}
  return lo>0?hi/lo-1.0:0.0; }

static vfft_plan build(int N, const char *route)
{
    vfft_config_t cfg; vfft_plan p;
    memset(&cfg,0,sizeof cfg);
    cfg.transform=VFFT_C2C; cfg.placement=VFFT_OUTOFPLACE;
    cfg.rigor=VFFT_MEASURE; cfg.dims=1; cfg.n[0]=N; cfg.howmany=1;
    cfg.layout=VFFT_LAYOUT_INTERLEAVED; cfg.order=VFFT_ORDER_NATURAL;
    cfg.nthreads=1;
    _putenv_s("VFFT_FORCE_ZROUTE", route);
    p = vfft_create(&cfg);
    _putenv_s("VFFT_FORCE_ZROUTE", "");   /* never leak to the next create */
    return p;
}

static void cell(int N, int pace_ms)
{
    vfft_plan pl = build(N, "legacy");
    vfft_plan pz = build(N, "zturn");
    double *zin,*zl,*zz,tl[TRIALS],tz[TRIALS];
    int i,k,r,reps; double w=0,mag=0;

    if(!pl||!pz){ printf("  N=%-6d create failed (legacy=%p zturn=%p)\n",
                          N,(void*)pl,(void*)pz); return; }
    zin=(double*)calloc(2*(size_t)N+8,sizeof(double));
    zl =(double*)calloc(2*(size_t)N+8,sizeof(double));
    zz =(double*)calloc(2*(size_t)N+8,sizeof(double));
    for(i=0;i<2*N;i++) zin[i]=rnd();

    vfft_execute(pl,VFFT_FORWARD,zin,NULL,zl,NULL);
    vfft_execute(pz,VFFT_FORWARD,zin,NULL,zz,NULL);
    for(i=0;i<2*N;i++){ double d=fabs(zl[i]-zz[i]);
        if(d>w)w=d; if(fabs(zl[i])>mag)mag=fabs(zl[i]); }
    if(!((mag>0?w/mag:w) < 1e-12)){
        printf("  N=%-6d *** ROUTES DISAGREE (rel %.2e) -- NOT TIMED ***\n",
               N, mag>0?w/mag:w); goto done; }

    reps=(int)(20000000.0/(double)N); if(reps<20)reps=20; if(reps>2000)reps=2000;
    for(k=0;k<20;k++){ vfft_execute(pl,VFFT_FORWARD,zin,NULL,zl,NULL);
                       vfft_execute(pz,VFFT_FORWARD,zin,NULL,zz,NULL); }
    for(k=0;k<TRIALS;k++){
        double t0=now_ns();
        for(r=0;r<reps;r++) vfft_execute(pl,VFFT_FORWARD,zin,NULL,zl,NULL);
        tl[k]=(now_ns()-t0)/reps;
        if(pace_ms) Sleep((DWORD)pace_ms);
        t0=now_ns();
        for(r=0;r<reps;r++) vfft_execute(pz,VFFT_FORWARD,zin,NULL,zz,NULL);
        tz[k]=(now_ns()-t0)/reps;
        if(pace_ms) Sleep((DWORD)pace_ms);
    }
    {
        double L=med(tl,TRIALS), Z=med(tz,TRIALS);
        double sl=100*spread(tl,TRIALS), sz=100*spread(tz,TRIALS);
        double gap=100.0*(L/Z-1.0);
        printf("  N=%-6d legacy %9.1f ns (sp %4.1f%%) | zturn %9.1f ns (sp %4.1f%%) | %+6.1f%% %s\n",
               N,L,sl,Z,sz,gap,
               fabs(gap) < (sl+sz)/2 ? "INSIDE NOISE" : (gap>0?"ZTURN":"LEGACY"));
    }
done:
    free(zin); free(zl); free(zz);
    vfft_destroy(pl); vfft_destroy(pz);
}

int main(int argc,char**argv)
{
    static const int NS[]={2048,4096,8192,16384};
    int pace=(argc>1)?atoi(argv[1]):20;
    size_t i;
    SetProcessAffinityMask(GetCurrentProcess(),0x4);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    setvbuf(stdout,NULL,_IONBF,0);
    printf("cascade route: LEGACY (s0s/sterm) vs ZTURN (s0t/stf) — same msg mid\n");
    printf("  +%% = zturn faster.  pinned core 2, HIGH, medians of %d, alternated\n\n",TRIALS);
    for(i=0;i<sizeof NS/sizeof NS[0];i++) cell(NS[i],pace);
    return 0;
}
