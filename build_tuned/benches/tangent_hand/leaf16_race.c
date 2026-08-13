/* leaf16_race.c — leaf-slot race at R16: production n1tb44 (blocked 4.4,
 * classic interior) vs w16tgL (tangent-Givens interior, twiddles stripped).
 * Both: untwiddled DFT-16 per column, corner-turn stores out[2*(k*OLs + p)].
 * Geometry = leaf of a 256 (16 columns, Ls=OLs=16). Paired protocol. */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>

void radix16_z_n1tb44_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix16_z_w16tgL_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);

typedef double _Complex cx;
static const double PI = 3.14159265358979323846;

static uint64_t lcg=0x9E3779B97F4A7C15ull;
static double rnd(void){lcg=lcg*6364136223846793005ull+1442695040888963407ull;
    return ((double)(int64_t)(lcg>>11))/4503599627370496.0;}
static LARGE_INTEGER qf;
static double now_ns(void){LARGE_INTEGER t;QueryPerformanceCounter(&t);
    return (double)t.QuadPart*(1e9/(double)qf.QuadPart);}
static int cmp_d(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y);}

#define BLOCK 4096
#define ROUNDS 35
#define PACE_MS 200

int main(void){
    QueryPerformanceFrequency(&qf);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(),0x4);

    /* ONE page-aligned arena, +64B skew between planes (banked protocol:
     * separate mallocs = 4KB-alias placement lottery across runs) */
    double *arena=_aligned_malloc(64*1024,4096);
    double *x  = arena;                    /* 512 doubles */
    double *oA = arena + 512 + 8;          /* +64B skew  */
    double *oB = oA    + 512 + 8;
    double *Tdum = oB  + 512 + 8;
    for(int i=0;i<512;i++) x[i]=rnd();
    memset(Tdum,0,8*120*sizeof(double));

    /* gate: per column b, DFT-16 over legs x[16a+b] -> out[2*(b*16+c)] */
    radix16_z_n1tb44_fwd_avx2(x,0,oA,0,Tdum,0,16,0,16,0,16);
    radix16_z_w16tgL_fwd_avx2(x,0,oB,0,Tdum,0,16,0,16,0,16);
    double ea=0,eb=0;
    for(int b=0;b<16;b++){
        for(int c=0;c<16;c++){
            cx ref=0;
            for(int a=0;a<16;a++){
                cx xv=x[2*(16*a+b)]+I*x[2*(16*a+b)+1];
                ref+=cexp(-2.0*PI*I*(double)c*a/16.0)*xv;
            }
            cx ga=oA[2*(b*16+c)]+I*oA[2*(b*16+c)+1];
            cx gb=oB[2*(b*16+c)]+I*oB[2*(b*16+c)+1];
            double e1=cabs(ga-ref),e2=cabs(gb-ref);
            if(e1>ea)ea=e1; if(e2>eb)eb=e2;
        }
    }
    printf("gate vs golden DFT-16 (corner-turned): n1tb44 %.3e   w16tgL %.3e   %s\n",
           ea,eb,(ea<1e-10&&eb<1e-10)?"BOTH CORRECT":"FAIL");
    if(ea>=1e-10||eb>=1e-10) return 1;

    /* paired race: 0=n1tb44(base) 1=w16tgL 2=n1tb44(ctrl) */
    double *sw=malloc(ROUNDS*sizeof(double)), *sb=malloc(ROUNDS*sizeof(double)),
           *sc=malloc(ROUNDS*sizeof(double));
    for(int r=0;r<ROUNDS;r++){
        for(int which=0;which<3;which++){
            int arm=(r&1)?(2-which):which;
            double *out=(arm==1)?oB:oA;
            void (*fn)(const double*,const double*,double*,double*,const double*,
                      const double*,size_t,size_t,size_t,size_t,size_t)
                = (arm==1)?radix16_z_w16tgL_fwd_avx2:radix16_z_n1tb44_fwd_avx2;
            for(int w=0;w<BLOCK;w++) fn(x,0,out,0,Tdum,0,16,0,16,0,16);
            double t0=now_ns();
            for(int w=0;w<BLOCK;w++) fn(x,0,out,0,Tdum,0,16,0,16,0,16);
            double ns=(now_ns()-t0)/BLOCK;
            if(arm==0) sb[r]=ns; else if(arm==1) sw[r]=ns; else sc[r]=ns;
            double until=now_ns()+PACE_MS*1e6; while(now_ns()<until) _mm_pause();
        }
    }
    double fb=1e30,fw=1e30,fc=1e30;
    for(int r=0;r<ROUNDS;r++){ if(sb[r]<fb)fb=sb[r]; if(sw[r]<fw)fw=sw[r]; if(sc[r]<fc)fc=sc[r]; }
    double *dw=malloc(ROUNDS*sizeof(double)), *dc=malloc(ROUNDS*sizeof(double));
    int winw=0,winc=0;
    for(int r=0;r<ROUNDS;r++){ dw[r]=sw[r]-sb[r]; dc[r]=sc[r]-sb[r];
        if(dw[r]<0)winw++; if(dc[r]<0)winc++; }
    qsort(dw,ROUNDS,sizeof(double),cmp_d); qsort(dc,ROUNDS,sizeof(double),cmp_d);
    double mb; { double*t=malloc(ROUNDS*sizeof(double)); memcpy(t,sb,ROUNDS*sizeof(double));
        qsort(t,ROUNDS,sizeof(double),cmp_d); mb=t[ROUNDS/2]; free(t); }
    printf("leaf R16, 16 cols, block=%d rounds=%d  core2 HIGH\n",BLOCK,ROUNDS);
    printf("floors(ns): n1tb44 %.1f   w16tgL %.1f   ctrl %.1f\n",fb,fw,fc);
    printf("paired median delta vs n1tb44:  w16tgL %+.2f ns (%+.2f%%)  wins %d/%d\n",
           dw[ROUNDS/2], dw[ROUNDS/2]/mb*100, winw, ROUNDS);
    printf("  control:                      %+.2f ns (%+.2f%%)  wins %d/%d  <- noise floor\n",
           dc[ROUNDS/2], dc[ROUNDS/2]/mb*100, winc, ROUNDS);
    return 0;
}
