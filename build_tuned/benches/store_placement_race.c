/* store_placement_race.c — B3 probe: does STORE PLACEMENT move time?
 *
 * cascade_z.ml makes memory ops first-class scheduled nodes, but in B2 their
 * positions are PINNED to reproduce the committed bytes; the docstring says
 * "the cost-driven placement SEARCH is B3". Before building a search it is
 * worth knowing whether the axis has any range at all, and that costs
 * nothing: two placements of the SAME kernel are already committed.
 *
 *   radix8_z_stf_r4_bwd_avx2    ZS_legacy  — all 8 stores TRAILING (lines
 *                                            162-169 of a 179-line file)
 *   radix8_z_stf_r4sk_bwd_avx2  ZS_afterdef— each store at its sink's def
 *                                            (lines 108,124,138,140,...)
 *
 * Same math, same schedule, same edges — only where the stores sit. stfb is
 * the only kind that can take afterdef today (the gate requires an
 * E_sect_tap store edge; TR4/REINT edges "need a readiness-set interleave,
 * which is B3 emitter work").
 *
 * CORRECTNESS IS A memcmp. Placement cannot change the values, so the two
 * planes must be byte-identical. A difference means the variant is broken,
 * not that the timing is interesting. This is a stronger gate than a
 * tolerance compare and it costs nothing here.
 *
 * The sk variant is in the corpus but NOT wired into src/core — it has never
 * been raced. That is exactly the question B3 asks, in miniature: if
 * placement is worth searching, it should show up here.
 *
 * Build: python build.py --src benches/store_placement_race.c --compile
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

void radix8_z_stf_r4_bwd_avx2  (const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix8_z_stf_r4sk_bwd_avx2(const double*,const double*,double*,double*,
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
/* MINIMUM, not median: timing noise is one-sided (interrupts, migration
 * and frequency dips only ADD time), so the fastest trial is the least
 * contaminated estimate of the kernel's own cost. */
static double best(const double *v,int n)
{ int i; double lo=v[0]; for(i=1;i<n;i++) if(v[i]<lo) lo=v[i]; return lo; }
static double spread(const double*v,int n)
{ double lo=v[0],hi=v[0];int i;
  for(i=1;i<n;i++){if(v[i]<lo)lo=v[i];if(v[i]>hi)hi=v[i];}
  return lo>0?hi/lo-1.0:0.0; }

static void cell(int N, int pace_ms)
{
    /* ZTURN-S radix-8 terminator geometry: OLs = count = N/8. */
    const size_t count = (size_t)N / 8u, OLs = count;
    const size_t nin  = 2*(7*OLs + count) + 64;   /* radix 8 = EIGHT legs */
    const size_t nout = 4*(3*OLs + count) + 64;
    const size_t ntw  = 2*count + 64;
    double *zin =(double*)_aligned_malloc(nin *sizeof(double),64);
    double *za  =(double*)_aligned_malloc(nout*sizeof(double),64);
    double *zb  =(double*)_aligned_malloc(nout*sizeof(double),64);
    double *tw  =(double*)_aligned_malloc(ntw *sizeof(double),64);
    double ta[TRIALS], tb[TRIALS], tc[TRIALS];
    size_t i; int k,r,reps;

    for(i=0;i<nin;i++)  zin[i]=rnd();
    for(i=0;i<ntw;i++){ double th=0.37*(double)i; tw[i]=((i>>2)&1)?sin(th):cos(th); }
    memset(za,0,nout*sizeof(double)); memset(zb,0,nout*sizeof(double));

    radix8_z_stf_r4_bwd_avx2  (zin,0,za,0,tw,0,0,0,OLs,0,count);
    radix8_z_stf_r4sk_bwd_avx2(zin,0,zb,0,tw,0,0,0,OLs,0,count);
    if(memcmp(za,zb,nout*sizeof(double))!=0){
        printf("  N=%-6d *** PLANES DIFFER — sk is not a placement twin, NOT TIMED ***\n",N);
        goto done;
    }

    reps=(int)(40000000.0/(double)N); if(reps<2000)reps=2000; if(reps>200000)reps=200000;
    for(k=0;k<300;k++){ radix8_z_stf_r4_bwd_avx2  (zin,0,za,0,tw,0,0,0,OLs,0,count);
                        radix8_z_stf_r4sk_bwd_avx2(zin,0,zb,0,tw,0,0,0,OLs,0,count); }
    for(k=0;k<TRIALS;k++){
        double t0=now_ns();
        for(r=0;r<reps;r++) radix8_z_stf_r4_bwd_avx2(zin,0,za,0,tw,0,0,0,OLs,0,count);
        ta[k]=(now_ns()-t0)/reps;
        if(pace_ms) Sleep((DWORD)pace_ms);
        t0=now_ns();
        for(r=0;r<reps;r++) radix8_z_stf_r4sk_bwd_avx2(zin,0,zb,0,tw,0,0,0,OLs,0,count);
        tb[k]=(now_ns()-t0)/reps;
        if(pace_ms) Sleep((DWORD)pace_ms);
        t0=now_ns();   /* CONTROL: the incumbent again, a second call site.
                        * ONE kernel timed twice -- whatever gap this shows is
                        * placement luck and thermal drift, never a placement
                        * EFFECT. It is the floor the real arm must clear. */
        for(r=0;r<reps;r++) radix8_z_stf_r4_bwd_avx2(zin,0,za,0,tw,0,0,0,OLs,0,count);
        tc[k]=(now_ns()-t0)/reps;
        if(pace_ms) Sleep((DWORD)pace_ms);
    }
    {
        double A=best(ta,TRIALS), B=best(tb,TRIALS), C=best(tc,TRIALS);
        double sa=100*spread(ta,TRIALS), sb=100*spread(tb,TRIALS);
        double gap=100.0*(A/B-1.0);   /* the claimed effect      */
        double ctl=100.0*(A/C-1.0);   /* the same kernel, twice  */
        printf("  N=%-6d count=%-5zu | trailing %8.1f (sp %4.1f%%) | at-def %8.1f (sp %4.1f%%)"
               " | effect %+6.1f%% | control %+5.1f%% => %s\n",
               N,count,A,sa,B,sb,gap,ctl,
               (fabs(gap) > 2.0*fabs(ctl) && fabs(gap) > (sa+sb)/2)
                 ? (gap>0?"AT-DEF":"TRAILING") : "NOT A RESULT");
    }
done:
    _aligned_free(zin);_aligned_free(za);_aligned_free(zb);_aligned_free(tw);
}

int main(int argc,char**argv)
{
    static const int NS[]={2048,4096,8192,16384,32768};
    int pace=(argc>1)?atoi(argv[1]):15;
    size_t i;
    SetProcessAffinityMask(GetCurrentProcess(),0x4);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    setvbuf(stdout,NULL,_IONBF,0);
    printf("B3 probe — stfb store PLACEMENT: trailing (ZS_legacy) vs at-def (ZS_afterdef)\n");
    printf("  same math/schedule/edges; only WHERE the 8 stores sit. +%% = at-def faster\n");
    printf("  gate = memcmp of the whole plane. pinned core 2, HIGH, BEST of %d, alternated\n\n",TRIALS);
    for(i=0;i<sizeof NS/sizeof NS[0];i++) cell(NS[i],pace);
    return 0;
}
