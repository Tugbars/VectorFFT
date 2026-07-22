/* Gate: n1 fwd_il_out + n1 bwd_il_in (the DIF-plan n1 boundaries) vs
 * composed references. BIT on vector region, <=4ulp on partial tail. */
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
 void C(radix##R##_n1_fwd_,I)(double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_n1_bwd_,I)(double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_n1_bwd_,C(I,_il_in))(const double*,double*,double*,const double*,const double*,size_t,size_t); \
 void C(radix##R##_n1_fwd_,C(I,_il_out))(const double*,const double*,double*,const double*,const double*,size_t,size_t);
D(2) D(3) D(4) D(5) D(6) D(7) D(8) D(10) D(11) D(12) D(13) D(16) D(17) D(19) D(20) D(25) D(32) D(64)
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
static int one(int R, sfn of, sfn ob, ifn ni, ofn no, size_t me, int seed){
    size_t ios = me + 8, big = (size_t)R*ios, vfull = me & ~(size_t)7;
    double *z=aligned_alloc(64,2*big*8);
    double *r1=aligned_alloc(64,big*8),*i1=aligned_alloc(64,big*8);
    double *r2=aligned_alloc(64,big*8),*i2=aligned_alloc(64,big*8);
    double *z1=aligned_alloc(64,2*big*8),*z2=aligned_alloc(64,2*big*8);
    srand(seed);
    for(size_t i=0;i<2*big;i++) z[i]=2.0*rand()/RAND_MAX-1;
    /* bwd il_in: manual deint -> original bwd  vs  il_in */
    memset(r1,0x5A,big*8); memset(i1,0x5A,big*8);
    for(int j=0;j<R;j++) for(size_t k=0;k<me;k++){
        r1[j*ios+k]=z[2*(j*ios+k)]; i1[j*ios+k]=z[2*(j*ios+k)+1]; }
    ob(r1,i1,NULL,NULL,ios,me);
    memset(r2,0x5A,big*8); memset(i2,0x5A,big*8);
    ni(z,r2,i2,NULL,NULL,ios,me);
    int ok=1;
    for(int j=0;j<R && ok;j++){
        ok &= !memcmp(r1+j*ios, r2+j*ios, vfull*8);
        ok &= !memcmp(i1+j*ios, i2+j*ios, vfull*8);
        for(size_t k=vfull;k<me && ok;k++){
            ok &= ulp_ok(r1[j*ios+k], r2[j*ios+k]);
            ok &= ulp_ok(i1[j*ios+k], i2[j*ios+k]); } }
    /* fwd il_out: original fwd on copies -> manual interleave  vs  il_out */
    for(size_t i=0;i<big;i++){r1[i]=2.0*rand()/RAND_MAX-1;i1[i]=2.0*rand()/RAND_MAX-1;}
    memcpy(r2,r1,big*8); memcpy(i2,i1,big*8);
    of(r2,i2,NULL,NULL,ios,me);
    memset(z1,0xA5,2*big*8);
    for(int j=0;j<R;j++) for(size_t k=0;k<me;k++){
        z1[2*(j*ios+k)]=r2[j*ios+k]; z1[2*(j*ios+k)+1]=i2[j*ios+k]; }
    memset(z2,0xA5,2*big*8);
    no(r1,i1,z2,NULL,NULL,ios,me);
    for(int j=0;j<R && ok;j++){
        ok &= !memcmp(z1+2*(size_t)j*ios, z2+2*(size_t)j*ios, 2*vfull*8);
        for(size_t k=vfull;k<me && ok;k++){
            ok &= ulp_ok(z1[2*(j*ios+k)], z2[2*(j*ios+k)]);
            ok &= ulp_ok(z1[2*(j*ios+k)+1], z2[2*(j*ios+k)+1]); } }
    printf("  r%-2d me=%-3zu %s\n",R,me,ok?"BIT":"**FAIL**");
    free(z);free(r1);free(i1);free(r2);free(i2);free(z1);free(z2);
    return ok;
}
#define RUN(R) for(int m=0;m<4;m++) ok&=one(R, \
  C(radix##R##_n1_fwd_,I), C(radix##R##_n1_bwd_,I), \
  C(radix##R##_n1_bwd_,C(I,_il_in)), C(radix##R##_n1_fwd_,C(I,_il_out)), \
  (size_t[]){64,65,66,67}[m], 21+m);
int main(void){ int ok=1; RUN(2) RUN(3) RUN(4) RUN(5) RUN(6) RUN(7) RUN(8) RUN(10) RUN(11) RUN(12) RUN(13) RUN(16) RUN(17) RUN(19) RUN(20) RUN(25) RUN(32) RUN(64)
    puts(ok?"ALL PASS":"FAILURES"); return ok?0:1; }
