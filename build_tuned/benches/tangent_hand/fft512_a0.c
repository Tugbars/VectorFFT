/* fft512_a0.c — A-0 of docs/roadmap/r32_tangent_parity_plan.md.
 *
 * THE DECISIVE CONTRAST: hand tangent kernels vs the SHIPPED EMITTED tangent
 * pool, same harness, same tables, same arena. Resolves whether the emitted
 * R32 construction is behind the hand one (opens A-1 wing32) or the
 * shared-layer thesis holds (weight shifts to Track B).
 *
 * Lanes (paired same-round, alternating order):
 *   A     (16,32) leaf16 n1tb44   + mid32 t2b        — pure classic anchor
 *   E     (16,32) leaf16 w16tgL   + mid32 w32tg      — HAND tangent, same shape as G
 *   F     (32,16) leaf32 w32tgL   + mid16 w16tg      — HAND tangent champion (~306 ns on 08-11)
 *   G     (16,32) leaf16 n1ttan   + mid32 t2btan216  — SHIPPED EMITTED tangent pool
 *   Actrl — control twin of A
 *
 * Decision reads:
 *   G vs E — construction-only contrast at the same (16,32) shape.
 *   G vs F — best constructible emitted route vs best hand route.
 *
 * Protocol: ONE arena +64B skews, BLOCK=1024 warm + 1024 timed, 35 rounds,
 * 200 ms pause-spin pace per arm, core 2 (mask 0x4), HIGH priority, paired
 * same-round deltas + floors + wins. (leaf16_race.c post-fix template.)
 *
 * Hand kernels: local copies (this dir), preserved 2026-08-13 from the
 * 2026-08-11 session scratchpad (w32tg_gen.py provenance).
 * Emitted kernels: the shipped tangent pool under
 *   src/dag-fft-compiler/codelets/zil/avx2/pure_il/tangent/.
 *
 * Build (cygwin gcc 12.4, from this directory):
 *   /c/tools/cygwin/bin/gcc.exe -O2 -mavx2 -mfma -o fft512_a0.exe fft512_a0.c \
 *     w16tgL_kernel.c w16tg_kernel.c w32tg_kernel.c w32tgL_kernel.c \
 *     ../../../src/dag-fft-compiler/codelets/zil/avx2/pure_il/radix16_z_n1tb44_avx2.c \
 *     ../../../src/dag-fft-compiler/codelets/zil/avx2/pure_il/radix32_z_t2b_avx2.c \
 *     ../../../src/dag-fft-compiler/codelets/zil/avx2/pure_il/tangent/radix16_z_n1ttan_avx2.c \
 *     ../../../src/dag-fft-compiler/codelets/zil/avx2/pure_il/tangent/radix32_z_t2btan216_avx2.c -lm
 *   (do NOT link w16tg_kernel_512.c — same symbol, AVX-512 target, no AVX-512 here)
 */
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>

void radix16_z_n1tb44_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix32_z_t2b_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix16_z_w16tgL_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix16_z_w16tg_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix32_z_w32tg_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix32_z_w32tgL_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix16_z_n1ttan_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix32_z_t2btan216_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);

typedef double _Complex cx;
static const double PI = 3.14159265358979323846;

/* VTW2 records for the mid stage at N=512, d = w512^{l*k}.
 * "ours" fold  = [c,c][-s,+s]  (emitted BYTW2 kernels: t2b, t2btan216)
 * "theirs" fold= [c,c][+s,-s]  (hand tangent-Givens kernels: w32tg, w16tg) */
static void gen32_ours(double *T,int kc){
    for(int l=1;l<32;l++){double*r=T+(l-1)*8;
        for(int col=0;col<2;col++){int kk=kc+col;
            cx d=cexp(-2.0*PI*I*(double)l*kk/512.0);
            r[col*2]=creal(d);r[col*2+1]=creal(d);
            r[4+col*2]=-cimag(d);r[4+col*2+1]=cimag(d);} }
}
static void gen32_theirs(double *T,int kc){
    for(int l=1;l<32;l++){double*r=T+(l-1)*8;
        for(int col=0;col<2;col++){int kk=kc+col;
            cx d=cexp(-2.0*PI*I*(double)l*kk/512.0);
            r[col*2]=creal(d);r[col*2+1]=creal(d);
            r[4+col*2]=cimag(d);r[4+col*2+1]=-cimag(d);} }
}
static void gen16_theirs(double *T,int kc){
    for(int l=1;l<16;l++){double*r=T+(l-1)*8;
        for(int col=0;col<2;col++){int kk=kc+col;
            cx d=cexp(-2.0*PI*I*(double)l*kk/512.0);
            r[col*2]=creal(d);r[col*2+1]=creal(d);
            r[4+col*2]=cimag(d);r[4+col*2+1]=-cimag(d);} }
}

static uint64_t lcg=0x9E3779B97F4A7C15ull;
static double rnd(void){lcg=lcg*6364136223846793005ull+1442695040888963407ull;
    return ((double)(int64_t)(lcg>>11))/4503599627370496.0;}
static LARGE_INTEGER qf;
static double now_ns(void){LARGE_INTEGER t;QueryPerformanceCounter(&t);
    return (double)t.QuadPart*(1e9/(double)qf.QuadPart);}
static int cmp_d(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y);}

#define BLOCK 1024
#define ROUNDS 35
#define PACE_MS 200
#define NLANE 7           /* A E F G H1 H2 Actrl */

static double *S;
static double *X,*Y,*Tm32,*Tf32,*Tf16,*Tdum;

/* (16,32): leaf16 32 cols -> S (corner-turn, OLs=16); mid32 over 16 cols */
static void route_A(void){  /* classic */
    radix16_z_n1tb44_fwd_avx2(X,0,S,0,Tdum,0,32,0,16,0,32);
    radix32_z_t2b_fwd_avx2   (S,0,Y,0,Tm32,0,16,0,16,0,16);
}
static void route_E(void){  /* hand tangent, (16,32) */
    radix16_z_w16tgL_fwd_avx2(X,0,S,0,Tdum,0,32,0,16,0,32);
    radix32_z_w32tg_fwd_avx2 (S,0,Y,0,Tf32,0,16,0,16,0,16);
}
/* (32,16): leaf32 16 cols -> S (corner-turn, OLs=32); mid16 over 32 cols */
static void route_F(void){  /* hand tangent champion, (32,16) */
    radix32_z_w32tgL_fwd_avx2(X,0,S,0,Tdum,0,16,0,32,0,16);
    radix16_z_w16tg_fwd_avx2 (S,0,Y,0,Tf16,0,32,0,32,0,32);
}
static void route_G(void){  /* SHIPPED EMITTED tangent pool, (16,32) */
    radix16_z_n1ttan_fwd_avx2    (X,0,S,0,Tdum,0,32,0,16,0,32);
    radix32_z_t2btan216_fwd_avx2 (S,0,Y,0,Tm32,0,16,0,16,0,16);
}
/* slot-attribution mixes, both (16,32):
 * H1 = emitted leaf + HAND mid  -> H1-E isolates the LEAF (emit vs hand)
 * H2 = HAND leaf + emitted mid  -> H2-E isolates the MID  (emit vs hand) */
static void route_H1(void){
    radix16_z_n1ttan_fwd_avx2    (X,0,S,0,Tdum,0,32,0,16,0,32);
    radix32_z_w32tg_fwd_avx2     (S,0,Y,0,Tf32,0,16,0,16,0,16);
}
static void route_H2(void){
    radix16_z_w16tgL_fwd_avx2    (X,0,S,0,Tdum,0,32,0,16,0,32);
    radix32_z_t2btan216_fwd_avx2 (S,0,Y,0,Tm32,0,16,0,16,0,16);
}
static void run_lane(int arm){
    switch(arm){
        case 0: case 6: route_A(); break;
        case 1: route_E(); break;
        case 2: route_F(); break;
        case 3: route_G(); break;
        case 4: route_H1(); break;
        default: route_H2(); break;
    }
}

static void *amalloc(size_t sz,size_t al){
    void *raw=malloc(sz+al+sizeof(void*));
    if(!raw) return NULL;
    uintptr_t p=((uintptr_t)raw+sizeof(void*)+al-1)&~(uintptr_t)(al-1);
    ((void**)p)[-1]=raw;
    return (void*)p;
}

int main(void){
    QueryPerformanceFrequency(&qf);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(),0x4);

    double *arena=amalloc(192*1024,4096);
    X    = arena;                    /* 1024 dbl */
    S    = X    + 1024 + 8;          /* +64B skews between planes */
    Y    = S    + 1024 + 8;
    Tm32 = Y    + 1024 + 8;          /* 8*248  = 1984 */
    Tf32 = Tm32 + 1984 + 8;
    /* spacer = the original harness's Td16 slot (16*120 = 1920 dbl), kept so
       Tf16/Tdum land at the SAME arena offsets as the 08-11 fft512_full binary
       — the 4KB-alias lottery must not differ between the two harnesses. */
    Tf16 = Tf32 + 1984 + 8 + 1920 + 8;
    Tdum = Tf16 + 1920 + 8;
    for(int i=0;i<1024;i++) X[i]=rnd();
    for(int g=0;g<8;g++){ gen32_ours(Tm32+g*248,2*g); gen32_theirs(Tf32+g*248,2*g); }
    for(int g=0;g<16;g++){ gen16_theirs(Tf16+g*120,2*g); }
    memset(Tdum,0,64);

    /* gate every raced arm vs direct DFT-512 */
    cx *ref=malloc(512*sizeof(cx)); cx *w=malloc(512*sizeof(cx));
    for(int i=0;i<512;i++) w[i]=cexp(-2.0*PI*I*(double)i/512.0);
    for(int j=0;j<512;j++){ cx s=0;
        for(int nn=0;nn<512;nn++){ cx xv=X[2*nn]+I*X[2*nn+1];
            s+=w[(size_t)j*nn%512]*xv; } ref[j]=s; }
    const char*nm[6]={"A  n1tb44+t2b     classic (16,32)",
                      "E  w16tgL+w32tg   HAND tg (16,32)",
                      "F  w32tgL+w16tg   HAND tg (32,16)",
                      "G  n1ttan+t2btan  EMITTED tg (16,32)",
                      "H1 n1ttan+w32tg   emL+handM (16,32)",
                      "H2 w16tgL+t2btan  handL+emM (16,32)"};
    for(int a=0;a<6;a++){
        run_lane(a);                 /* lanes 0..5 = A E F G H1 H2 */
        double e=0;
        for(int j=0;j<512;j++){ cx g=Y[2*j]+I*Y[2*j+1];
            double d=cabs(g-ref[j]); if(d>e)e=d; }
        printf("gate %-36s %.3e %s\n",nm[a],e,e<1e-9?"OK":"FAIL");
        if(e>=1e-9) return 1;
    }

    /* paired race */
    double *sm[NLANE]; for(int i=0;i<NLANE;i++) sm[i]=malloc(ROUNDS*sizeof(double));
    for(int r=0;r<ROUNDS;r++){
        for(int wch=0;wch<NLANE;wch++){
            int arm=(r&1)?(NLANE-1-wch):wch;
            for(int wu=0;wu<BLOCK;wu++) run_lane(arm);
            double t0=now_ns();
            for(int wu=0;wu<BLOCK;wu++) run_lane(arm);
            sm[arm][r]=(now_ns()-t0)/BLOCK;
            double until=now_ns()+PACE_MS*1e6; while(now_ns()<until) _mm_pause();
        }
    }
    double fl[NLANE]; for(int i=0;i<NLANE;i++){fl[i]=1e30;
        for(int r=0;r<ROUNDS;r++) if(sm[i][r]<fl[i]) fl[i]=sm[i][r];}
    printf("\nN=512 A-0 hand-vs-emitted, block=%d rounds=%d core2 HIGH\n",BLOCK,ROUNDS);
    printf("floors(ns): A %.1f  E %.1f  F %.1f  G %.1f  H1 %.1f  H2 %.1f  ctrl %.1f\n",
           fl[0],fl[1],fl[2],fl[3],fl[4],fl[5],fl[6]);
    double mA; { double*t=malloc(ROUNDS*sizeof(double)); memcpy(t,sm[0],ROUNDS*sizeof(double));
        qsort(t,ROUNDS,sizeof(double),cmp_d); mA=t[ROUNDS/2]; free(t); }
    const char*lbl[NLANE]={"A","E hand(16,32)","F hand(32,16)","G EMITTED",
                           "H1 emL+handM","H2 handL+emM","ctrl"};
    for(int i=1;i<NLANE;i++){
        double d[ROUNDS]; int wn=0;
        for(int r=0;r<ROUNDS;r++){ d[r]=sm[i][r]-sm[0][r]; if(d[r]<0)wn++; }
        qsort(d,ROUNDS,sizeof(double),cmp_d);
        printf("paired vs A:   %-14s %+8.2f ns (%+6.2f%%)  wins %d/%d\n",
               lbl[i],d[ROUNDS/2],d[ROUNDS/2]/mA*100,wn,ROUNDS);
    }
    /* decision contrasts, paired directly (first arm minus second) */
    struct { int a,b; const char*t; } dc[4]={
        {3,1,"G-E   emitted vs hand, SAME shape "},
        {3,2,"G-F   emitted vs hand champion    "},
        {4,1,"H1-E  LEAF slot: emitted vs hand  "},
        {5,1,"H2-E  MID  slot: emitted vs hand  "}};
    for(int k=0;k<4;k++){
        double d[ROUNDS]; int wn=0;
        for(int r=0;r<ROUNDS;r++){ d[r]=sm[dc[k].a][r]-sm[dc[k].b][r]; if(d[r]<0)wn++; }
        qsort(d,ROUNDS,sizeof(double),cmp_d);
        printf("DECISION %s %+8.2f ns (%+6.2f%%)  first-wins %d/%d\n",
               dc[k].t,d[ROUNDS/2],d[ROUNDS/2]/mA*100,wn,ROUNDS);
    }
    return 0;
}
