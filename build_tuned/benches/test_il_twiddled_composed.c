/* Gate: t1s_dit {fwd il_out, bwd il_in} + t1_dif {fwd il_in, bwd il_out}
 * vs composed references with real twiddles. */
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
 void C(radix##R##_t1s_dit_bwd_,I)(double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_t1_dif_fwd_,I)(double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_t1_dif_bwd_,I)(double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_t1s_dit_bwd_,C(I,_il_in))(const double*,double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_t1s_dit_fwd_,C(I,_il_out))(const double*,const double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_t1_dif_fwd_,C(I,_il_in))(const double*,double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_t1_dif_bwd_,C(I,_il_out))(const double*,const double*,double*,const double*,const double*,size_t,size_t);
D(4) D(5) D(8) D(16) D(25) D(32)
typedef void(*sfn)(double*,double*,const double*,const double*,size_t,size_t);
typedef void(*ifn)(const double*,double*,double*,const double*,const double*,size_t,size_t);
typedef void(*ofn)(const double*,const double*,double*,const double*,const double*,size_t,size_t);
static int ulp_ok(double a, double b){
    if(a==b) return 1;
    /* epsilon-scaled absolute floor: catches catastrophic-cancellation sites
       where a ~1e-17 result flips sign under different FMA contraction */
    double m = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    if(m < 1.0) m = 1.0;
    if(fabs(a-b) <= 4.0*2.220446049250313e-16*m) return 1;
    long long x,y; memcpy(&x,&a,8); memcpy(&y,&b,8);
    if((x<0)!=(y<0)) return 0;
    long long d=x-y; if(d<0)d=-d; return d<=4;
}
static int gi(int R, const char*nm, sfn orig, ifn il, size_t me, int seed){
    size_t ios=me+8, big=(size_t)R*ios, vfull=me & ~(size_t)7;
    double *z=aligned_alloc(64,2*big*8);
    double *r1=aligned_alloc(64,big*8),*i1=aligned_alloc(64,big*8);
    double *r2=aligned_alloc(64,big*8),*i2=aligned_alloc(64,big*8);
    double *twr=aligned_alloc(64,(size_t)R*me*8),*twi=aligned_alloc(64,(size_t)R*me*8);
    srand(seed);
    for(size_t i=0;i<2*big;i++) z[i]=2.0*rand()/RAND_MAX-1;
    for(size_t i=0;i<(size_t)R*me;i++){twr[i]=2.0*rand()/RAND_MAX-1;twi[i]=2.0*rand()/RAND_MAX-1;}
    memset(r1,0x5A,big*8); memset(i1,0x5A,big*8);
    for(int j=0;j<R;j++) for(size_t k=0;k<me;k++){
        r1[j*ios+k]=z[2*(j*ios+k)]; i1[j*ios+k]=z[2*(j*ios+k)+1]; }
    orig(r1,i1,twr,twi,ios,me);
    memset(r2,0x5A,big*8); memset(i2,0x5A,big*8);
    il(z,r2,i2,twr,twi,ios,me);
    int ok=1;
    for(int j=0;j<R && ok;j++){
        ok &= !memcmp(r1+j*ios,r2+j*ios,vfull*8);
        ok &= !memcmp(i1+j*ios,i2+j*ios,vfull*8);
        for(size_t k=vfull;k<me && ok;k++){
            ok &= ulp_ok(r1[j*ios+k],r2[j*ios+k]);
            ok &= ulp_ok(i1[j*ios+k],i2[j*ios+k]); } }
    printf("  %-14s r%-2d me=%-3zu %s\n",nm,R,me,ok?"BIT":"**FAIL**");
    free(z);free(r1);free(i1);free(r2);free(i2);free(twr);free(twi);
    return ok;
}
static int go(int R, const char*nm, sfn orig, ofn il, size_t me, int seed){
    size_t ios=me+8, big=(size_t)R*ios, vfull=me & ~(size_t)7;
    double *r1=aligned_alloc(64,big*8),*i1=aligned_alloc(64,big*8);
    double *r2=aligned_alloc(64,big*8),*i2=aligned_alloc(64,big*8);
    double *z1=aligned_alloc(64,2*big*8),*z2=aligned_alloc(64,2*big*8);
    double *twr=aligned_alloc(64,(size_t)R*me*8),*twi=aligned_alloc(64,(size_t)R*me*8);
    srand(seed);
    for(size_t i=0;i<big;i++){r1[i]=2.0*rand()/RAND_MAX-1;i1[i]=2.0*rand()/RAND_MAX-1;}
    for(size_t i=0;i<(size_t)R*me;i++){twr[i]=2.0*rand()/RAND_MAX-1;twi[i]=2.0*rand()/RAND_MAX-1;}
    memcpy(r2,r1,big*8); memcpy(i2,i1,big*8);
    orig(r2,i2,twr,twi,ios,me);
    memset(z1,0xA5,2*big*8);
    for(int j=0;j<R;j++) for(size_t k=0;k<me;k++){
        z1[2*(j*ios+k)]=r2[j*ios+k]; z1[2*(j*ios+k)+1]=i2[j*ios+k]; }
    memset(z2,0xA5,2*big*8);
    il(r1,i1,z2,twr,twi,ios,me);
    int ok=1;
    for(int j=0;j<R && ok;j++){
        ok &= !memcmp(z1+2*(size_t)j*ios,z2+2*(size_t)j*ios,2*vfull*8);
        for(size_t k=vfull;k<me && ok;k++){
            ok &= ulp_ok(z1[2*(j*ios+k)],z2[2*(j*ios+k)]);
            ok &= ulp_ok(z1[2*(j*ios+k)+1],z2[2*(j*ios+k)+1]); } }
    printf("  %-14s r%-2d me=%-3zu %s\n",nm,R,me,ok?"BIT":"**FAIL**");
    free(r1);free(i1);free(r2);free(i2);free(z1);free(z2);free(twr);free(twi);
    return ok;
}
#define RUN(R) for(int m=0;m<3;m++){ size_t me=(size_t[]){64,65,67}[m]; \
  ok&=gi(R,"t1s bwd il_in",C(radix##R##_t1s_dit_bwd_,I),C(radix##R##_t1s_dit_bwd_,C(I,_il_in)),me,31+m); \
  ok&=go(R,"t1s fwd il_out",C(radix##R##_t1s_dit_fwd_,I),C(radix##R##_t1s_dit_fwd_,C(I,_il_out)),me,41+m); \
  ok&=gi(R,"dif fwd il_in",C(radix##R##_t1_dif_fwd_,I),C(radix##R##_t1_dif_fwd_,C(I,_il_in)),me,51+m); \
  ok&=go(R,"dif bwd il_out",C(radix##R##_t1_dif_bwd_,I),C(radix##R##_t1_dif_bwd_,C(I,_il_out)),me,61+m); }
int main(void){ int ok=1; RUN(4) RUN(5) RUN(8) RUN(16) RUN(25) RUN(32)
    puts(ok?"ALL PASS":"FAILURES"); return ok?0:1; }
