#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef USE512
#define I avx512
#else
#define I avx2
#endif
#define C2(a,b) a##b
#define C(a,b) C2(a,b)
#define D(R) \
 void C(radix##R##_t1s_dit_fwd_,I)(double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_t1_dif_fwd_,I)(double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_t1_dif_bwd_,I)(double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_t1s_dit_fwd_,C(I,_il_out))(const double*,const double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_t1_dif_bwd_,C(I,_il_out))(const double*,const double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_t1_dif_fwd_,C(I,_il_in))(const double*,double*,double*,const double*,const double*,size_t,size_t);
D(10) D(20)
typedef void(*sfn)(double*,double*,const double*,const double*,size_t,size_t);
typedef void(*ifn)(const double*,double*,double*,const double*,const double*,size_t,size_t);
typedef void(*ofn)(const double*,const double*,double*,const double*,const double*,size_t,size_t);
static int ulp_ok(double a,double b){
    if(a==b) return 1;
    double m=fabs(a)>fabs(b)?fabs(a):fabs(b); if(m<1.0)m=1.0;
    if(fabs(a-b)<=4.0*2.220446049250313e-16*m) return 1;
    long long x,y; memcpy(&x,&a,8); memcpy(&y,&b,8);
    if((x<0)!=(y<0)) return 0;
    long long d=x-y; if(d<0)d=-d; return d<=4; }
static int go(int R,const char*nm,sfn orig,ofn il,size_t me,int seed){
    size_t ios=me+8,big=(size_t)R*ios,vf=me&~(size_t)7;
    double *r1=aligned_alloc(64,big*8),*i1=aligned_alloc(64,big*8);
    double *r2=aligned_alloc(64,big*8),*i2=aligned_alloc(64,big*8);
    double *z1=aligned_alloc(64,2*big*8),*z2=aligned_alloc(64,2*big*8);
    double *tr=aligned_alloc(64,(size_t)R*me*8),*ti=aligned_alloc(64,(size_t)R*me*8);
    srand(seed);
    for(size_t i=0;i<big;i++){r1[i]=2.0*rand()/RAND_MAX-1;i1[i]=2.0*rand()/RAND_MAX-1;}
    for(size_t i=0;i<(size_t)R*me;i++){tr[i]=2.0*rand()/RAND_MAX-1;ti[i]=2.0*rand()/RAND_MAX-1;}
    memcpy(r2,r1,big*8); memcpy(i2,i1,big*8);
    orig(r2,i2,tr,ti,ios,me);
    memset(z1,0xA5,2*big*8);
    for(int j=0;j<R;j++) for(size_t k=0;k<me;k++){
        z1[2*(j*ios+k)]=r2[j*ios+k]; z1[2*(j*ios+k)+1]=i2[j*ios+k]; }
    memset(z2,0xA5,2*big*8);
    il(r1,i1,z2,tr,ti,ios,me);
    int ok=1;
    for(int j=0;j<R&&ok;j++){
        ok&=!memcmp(z1+2*(size_t)j*ios,z2+2*(size_t)j*ios,2*vf*8);
        for(size_t k=vf;k<me&&ok;k++){
            ok&=ulp_ok(z1[2*(j*ios+k)],z2[2*(j*ios+k)]);
            ok&=ulp_ok(z1[2*(j*ios+k)+1],z2[2*(j*ios+k)+1]);}}
    printf("  %-12s r%-2d me=%-3zu %s\n",nm,R,me,ok?"BIT":"**FAIL**");
    free(r1);free(i1);free(r2);free(i2);free(z1);free(z2);free(tr);free(ti);
    return ok; }
static int gi(int R,const char*nm,sfn orig,ifn il,size_t me,int seed){
    size_t ios=me+8,big=(size_t)R*ios,vf=me&~(size_t)7;
    double *z=aligned_alloc(64,2*big*8);
    double *r1=aligned_alloc(64,big*8),*i1=aligned_alloc(64,big*8);
    double *r2=aligned_alloc(64,big*8),*i2=aligned_alloc(64,big*8);
    double *tr=aligned_alloc(64,(size_t)R*me*8),*ti=aligned_alloc(64,(size_t)R*me*8);
    srand(seed);
    for(size_t i=0;i<2*big;i++) z[i]=2.0*rand()/RAND_MAX-1;
    for(size_t i=0;i<(size_t)R*me;i++){tr[i]=2.0*rand()/RAND_MAX-1;ti[i]=2.0*rand()/RAND_MAX-1;}
    memset(r1,0x5A,big*8); memset(i1,0x5A,big*8);
    for(int j=0;j<R;j++) for(size_t k=0;k<me;k++){
        r1[j*ios+k]=z[2*(j*ios+k)]; i1[j*ios+k]=z[2*(j*ios+k)+1]; }
    orig(r1,i1,tr,ti,ios,me);
    memset(r2,0x5A,big*8); memset(i2,0x5A,big*8);
    il(z,r2,i2,tr,ti,ios,me);
    int ok=1;
    for(int j=0;j<R&&ok;j++){
        ok&=!memcmp(r1+j*ios,r2+j*ios,vf*8);
        ok&=!memcmp(i1+j*ios,i2+j*ios,vf*8);
        for(size_t k=vf;k<me&&ok;k++){
            ok&=ulp_ok(r1[j*ios+k],r2[j*ios+k]);
            ok&=ulp_ok(i1[j*ios+k],i2[j*ios+k]);}}
    printf("  %-12s r%-2d me=%-3zu %s\n",nm,R,me,ok?"BIT":"**FAIL**");
    free(z);free(r1);free(i1);free(r2);free(i2);free(tr);free(ti);
    return ok; }
#define RUN(R) for(int m=0;m<3;m++){ size_t me=(size_t[]){64,65,67}[m]; \
  ok&=go(R,"t1s   ilo",C(radix##R##_t1s_dit_fwd_,I),C(radix##R##_t1s_dit_fwd_,C(I,_il_out)),me,31+m); \
  ok&=go(R,"difb  ilo",C(radix##R##_t1_dif_bwd_,I),C(radix##R##_t1_dif_bwd_,C(I,_il_out)),me,41+m); \
  ok&=gi(R,"dif   ili",C(radix##R##_t1_dif_fwd_,I),C(radix##R##_t1_dif_fwd_,C(I,_il_in)),me,51+m); }
int main(void){ int ok=1; RUN(10) RUN(20)
    puts(ok?"ALL PASS":"FAILURES"); return ok?0:1; }
