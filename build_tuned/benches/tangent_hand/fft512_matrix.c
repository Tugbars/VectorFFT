/* fft512_matrix.c — MIXED-PLAN MATRIX at N=512. Four routes from existing
 * kernels (classic = production cil pool, tg = tangent-Givens):
 *   A (16,32) leaf16 n1tb44 + mid32 t2b      — pure classic, banked winner
 *   B (32,16) leaf32 n1tb48 + mid16 w16tg    — MIXED: classic leaf + tg mid
 *   C (32,16) leaf32 n1tb48 + mid16 t2b44    — pure classic, flipped fact.
 *   D (16,32) leaf16 w16tgL + mid32 t2b      — MIXED: tg leaf + classic mid
 * All gated vs direct DFT-512. Paired same-round, 5 lanes (A,B,C,D,Actrl),
 * ONE arena +64B skews, BLOCK>=1024, 200ms pace, core 2 HIGH. */
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
void radix32_z_n1tb48_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix16_z_w16tgL_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix16_z_t2b44_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix16_z_w16tg_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void radix32_z_t2b_fwd_avx2(const double*,const double*,double*,double*,
    const double*,const double*,size_t,size_t,size_t,size_t,size_t);

typedef double _Complex cx;
static const double PI = 3.14159265358979323846;

/* mid-16 tables @512: 16 groups x 120, d_b = w512^{c*b} */
static void gen16_ours(double *T,int kc){
    for(int l=1;l<16;l++){double*r=T+(l-1)*8;
        for(int col=0;col<2;col++){int kk=kc+col;
            cx d=cexp(-2.0*PI*I*(double)l*kk/512.0);
            r[col*2]=creal(d);r[col*2+1]=creal(d);
            r[4+col*2]=-cimag(d);r[4+col*2+1]=cimag(d);} }
}
static void gen16_theirs(double *T,int kc){
    for(int l=1;l<16;l++){double*r=T+(l-1)*8;
        for(int col=0;col<2;col++){int kk=kc+col;
            cx d=cexp(-2.0*PI*I*(double)l*kk/512.0);
            r[col*2]=creal(d);r[col*2+1]=creal(d);
            r[4+col*2]=cimag(d);r[4+col*2+1]=-cimag(d);} }
}
/* mid-32 table @512: 8 groups x 248, d_b = w512^{c*b}, our fold */
static void gen32_ours(double *T,int kc){
    for(int l=1;l<32;l++){double*r=T+(l-1)*8;
        for(int col=0;col<2;col++){int kk=kc+col;
            cx d=cexp(-2.0*PI*I*(double)l*kk/512.0);
            r[col*2]=creal(d);r[col*2+1]=creal(d);
            r[4+col*2]=-cimag(d);r[4+col*2+1]=cimag(d);} }
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
#define NLANE 5           /* A B C D Actrl */

static double *S;

/* routes: (16,32): leaf16 32cols -> S[b*16+c]; mid32 over b, 16 cols */
static void route_1632(const double*x,double*y,const double*Tm32,int tg_leaf,const double*Tdum){
    if(tg_leaf) radix16_z_w16tgL_fwd_avx2(x,0,S,0,Tdum,0,32,0,16,0,32);
    else        radix16_z_n1tb44_fwd_avx2(x,0,S,0,Tdum,0,32,0,16,0,32);
    radix32_z_t2b_fwd_avx2(S,0,y,0,Tm32,0,16,0,16,0,16);
}
/* (32,16): leaf32 16cols -> S[b*32+c]; mid16 over b, 32 cols */
static void route_3216(const double*x,double*y,const double*Tm16,int tg_mid,const double*Tdum){
    radix32_z_n1tb48_fwd_avx2(x,0,S,0,Tdum,0,16,0,32,0,16);
    if(tg_mid) radix16_z_w16tg_fwd_avx2(S,0,y,0,Tm16,0,32,0,32,0,32);
    else       radix16_z_t2b44_fwd_avx2(S,0,y,0,Tm16,0,32,0,32,0,32);
}

int main(void){
    QueryPerformanceFrequency(&qf);
    SetPriorityClass(GetCurrentProcess(),HIGH_PRIORITY_CLASS);
    SetThreadAffinityMask(GetCurrentThread(),0x4);

    double *arena=_aligned_malloc(160*1024,4096);
    double *x   = arena;                    /* 1024 dbl */
    S           = x    + 1024 + 8;
    double *y   = S    + 1024 + 8;
    double *Tm32= y    + 1024 + 8;          /* 8*248  = 1984 */
    double *Td16= Tm32 + 1984 + 8;          /* 16*120 = 1920 */
    double *Tf16= Td16 + 1920 + 8;
    double *Tdum= Tf16 + 1920 + 8;
    for(int i=0;i<1024;i++) x[i]=rnd();
    for(int g=0;g<8;g++)  gen32_ours (Tm32+g*248,2*g);
    for(int g=0;g<16;g++){gen16_ours (Td16+g*120,2*g); gen16_theirs(Tf16+g*120,2*g);}
    memset(Tdum,0,64);

    /* gate all 4 routes vs direct DFT-512 */
    cx *ref=malloc(512*sizeof(cx)); cx *w=malloc(512*sizeof(cx));
    for(int i=0;i<512;i++) w[i]=cexp(-2.0*PI*I*(double)i/512.0);
    for(int j=0;j<512;j++){ cx s=0;
        for(int nn=0;nn<512;nn++){ cx xv=x[2*nn]+I*x[2*nn+1];
            s+=w[(size_t)j*nn%512]*xv; } ref[j]=s; }
    const char*nm[4]={"A leaf16+t2b48(classic)","B leaf32+w16tg(MIXED tg-mid)",
                      "C leaf32+t2b44(classic-flip)","D tgL+t2b48(MIXED tg-leaf)"};
    for(int r4=0;r4<4;r4++){
        if(r4==0) route_1632(x,y,Tm32,0,Tdum);
        else if(r4==1) route_3216(x,y,Tf16,1,Tdum);
        else if(r4==2) route_3216(x,y,Td16,0,Tdum);
        else route_1632(x,y,Tm32,1,Tdum);
        double e=0;
        for(int j=0;j<512;j++){ cx g=y[2*j]+I*y[2*j+1];
            double d=cabs(g-ref[j]); if(d>e)e=d; }
        printf("gate %-30s %.3e %s\n",nm[r4],e,e<1e-9?"OK":"FAIL");
        if(e>=1e-9) return 1;
    }

    /* paired race, 5 lanes */
    double *sm[NLANE]; for(int i=0;i<NLANE;i++) sm[i]=malloc(ROUNDS*sizeof(double));
    for(int r=0;r<ROUNDS;r++){
        for(int wch=0;wch<NLANE;wch++){
            int arm=(r&1)?(NLANE-1-wch):wch;
            void run_arm(void){
                if(arm==0||arm==4) route_1632(x,y,Tm32,0,Tdum);
                else if(arm==1) route_3216(x,y,Tf16,1,Tdum);
                else if(arm==2) route_3216(x,y,Td16,0,Tdum);
                else route_1632(x,y,Tm32,1,Tdum);
            }
            for(int wu=0;wu<BLOCK;wu++) run_arm();
            double t0=now_ns();
            for(int wu=0;wu<BLOCK;wu++) run_arm();
            sm[arm][r]=(now_ns()-t0)/BLOCK;
            double until=now_ns()+PACE_MS*1e6; while(now_ns()<until) _mm_pause();
        }
    }
    double fl[NLANE]; for(int i=0;i<NLANE;i++){fl[i]=1e30;
        for(int r=0;r<ROUNDS;r++) if(sm[i][r]<fl[i]) fl[i]=sm[i][r];}
    printf("\nN=512 matrix, block=%d rounds=%d  core2 HIGH\n",BLOCK,ROUNDS);
    printf("floors(ns): A %.1f   B %.1f   C %.1f   D %.1f   Actrl %.1f\n",
           fl[0],fl[1],fl[2],fl[3],fl[4]);
    double mb; { double*t=malloc(ROUNDS*sizeof(double)); memcpy(t,sm[0],ROUNDS*sizeof(double));
        qsort(t,ROUNDS,sizeof(double),cmp_d); mb=t[ROUNDS/2]; free(t); }
    const char*lbl[NLANE]={"A","B mixed tg-mid","C classic-flip","D mixed tg-leaf","ctrl"};
    for(int i=1;i<NLANE;i++){
        double *d=malloc(ROUNDS*sizeof(double)); int wn=0;
        for(int r=0;r<ROUNDS;r++){ d[r]=sm[i][r]-sm[0][r]; if(d[r]<0)wn++; }
        qsort(d,ROUNDS,sizeof(double),cmp_d);
        printf("paired median vs A:  %-16s %+8.2f ns (%+6.2f%%)  wins %d/%d\n",
               lbl[i],d[ROUNDS/2],d[ROUNDS/2]/mb*100,wn,ROUNDS);
        free(d);
    }
    return 0;
}
