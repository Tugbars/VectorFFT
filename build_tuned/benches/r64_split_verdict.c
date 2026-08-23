/* r64_split_verdict.c — at N=1024, was racing the R64 split worth it?
 *
 * The calibrator banked  n=1024  il_pair=16.64  il_kv=19  =  mid R16 tangent
 * (nibble 3) + leaf R64 variant 1 = radix64_z_n1tb416 (split 4.16).
 *
 * apply_blocked_default would NOT have chosen that. It resolves variant 2
 * first, i.e. radix64_z_n1tb88 (split 8.8), and only falls back to 1. So
 * before R64 entered the il_kv ladder, this cell ran 8.8 by rule. This
 * measures the difference that rule was costing.
 *
 *   kv 19 = PACK(mid 3, leaf 1) -> tangent mid + 4.16 leaf   (RACED pick)
 *   kv 35 = PACK(mid 3, leaf 2) -> tangent mid + 8.8  leaf   (rule's pick)
 *
 * WHY THE WISDOM DIR IS PINNED. The pair itself (16.64) must be held fixed,
 * or the two arms could race their way to different decompositions and the
 * comparison would not be about the split at all. VFFT_WISDOM_DIR supplies
 * the banked plan; VFFT_IL_KV then overrides only the form nibbles.
 *
 * N=1024 = 16 x 64, so the leaf runs at count = R1 = 16. The isolated kernel
 * race at exactly that count had 4.16 ahead of 8.8 in 3/3 runs by ~2-4%.
 * This checks whether that survives at the front door, where the leaf is one
 * stage of a pair.
 *
 * Build: python build.py --src benches/r64_split_verdict.c --vfft --compile
 * Run:   r64_split_verdict.exe <wisdir> [pace_ms]
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

static vfft_plan build(int N, const char *kv)
{
    vfft_config_t cfg; vfft_plan p;
    memset(&cfg,0,sizeof cfg);
    cfg.transform=VFFT_C2C; cfg.placement=VFFT_OUTOFPLACE;
    cfg.rigor=VFFT_MEASURE;   /* only consulted on a MISS; this hits and replays */
    cfg.dims=1; cfg.n[0]=N; cfg.howmany=1;
    cfg.layout=VFFT_LAYOUT_INTERLEAVED; cfg.order=VFFT_ORDER_NATURAL;
    cfg.nthreads=1;
    _putenv_s("VFFT_IL_KV", kv);
    p = vfft_create(&cfg);
    _putenv_s("VFFT_IL_KV", "");
    return p;
}

int main(int argc,char**argv)
{
    const char *wisdir = (argc>1) ? argv[1] : ".";
    const int  N   = (argc>2) ? atoi(argv[2]) : 1024;
    const char *kvA = (argc>3) ? argv[3] : "35";
    const char *kvB = (argc>4) ? argv[4] : "19";
    int pace = (argc>5) ? atoi(argv[5]) : 15;
    vfft_plan p88, p416;
    double *zin,*a,*b,t8[TRIALS],t4[TRIALS],tc[TRIALS];
    int i,k,r,reps; double w=0,mag=0;

    SetProcessAffinityMask(GetCurrentProcess(),0x4);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    setvbuf(stdout,NULL,_IONBF,0);
    _putenv_s("VFFT_WISDOM_DIR", wisdir);

    printf("N=1024 (16x64) leaf split: 8.8 (the RULE) vs 4.16 (the RACE)\n");
    printf("  pinned plan from %s; VFFT_IL_KV overrides only the form nibbles\n",wisdir);
    printf("  +%% = 4.16 faster. control = the 8.8 plan timed twice. BEST of %d\n\n",TRIALS);

    p88  = build(N,kvA);
    p416 = build(N,kvB);
    if(!p88||!p416){ printf("  create failed (%p / %p)\n",(void*)p88,(void*)p416); return 1; }

    zin=(double*)calloc(2*(size_t)N+8,sizeof(double));
    a  =(double*)calloc(2*(size_t)N+8,sizeof(double));
    b  =(double*)calloc(2*(size_t)N+8,sizeof(double));
    for(i=0;i<2*N;i++) zin[i]=rnd();

    vfft_execute(p88 ,VFFT_FORWARD,zin,NULL,a,NULL);
    vfft_execute(p416,VFFT_FORWARD,zin,NULL,b,NULL);
    for(i=0;i<2*N;i++){ double d=fabs(a[i]-b[i]);
        if(d>w)w=d; if(fabs(a[i])>mag)mag=fabs(a[i]); }
    if(!((mag>0?w/mag:w) < 1e-12)){
        printf("  *** ARMS DISAGREE (rel %.2e) — not the same transform ***\n",
               mag>0?w/mag:w); return 1; }

    reps=(int)(4000000.0/(double)N); if(reps<500)reps=500;
    for(k=0;k<50;k++){ vfft_execute(p88,VFFT_FORWARD,zin,NULL,a,NULL);
                       vfft_execute(p416,VFFT_FORWARD,zin,NULL,b,NULL); }
    for(k=0;k<TRIALS;k++){
        double t=now_ns();
        for(r=0;r<reps;r++) vfft_execute(p88,VFFT_FORWARD,zin,NULL,a,NULL);
        t8[k]=(now_ns()-t)/reps; if(pace) Sleep((DWORD)pace);
        t=now_ns();
        for(r=0;r<reps;r++) vfft_execute(p416,VFFT_FORWARD,zin,NULL,b,NULL);
        t4[k]=(now_ns()-t)/reps; if(pace) Sleep((DWORD)pace);
        t=now_ns();
        for(r=0;r<reps;r++) vfft_execute(p88,VFFT_FORWARD,zin,NULL,a,NULL);
        tc[k]=(now_ns()-t)/reps; if(pace) Sleep((DWORD)pace);
    }
    {
        double A=best(t8,TRIALS), B=best(t4,TRIALS), C=best(tc,TRIALS);
        printf("  8.8 (rule) %8.1f ns | 4.16 (raced) %8.1f ns | %+6.2f%% | control %+5.2f%% | rel %.1e\n",
               kvA,A,kvB,B,100.0*(A/B-1.0),100.0*(A/C-1.0), mag>0?w/mag:w);
    }
    free(zin); free(a); free(b);
    vfft_destroy(p88); vfft_destroy(p416);
    return 0;
}
