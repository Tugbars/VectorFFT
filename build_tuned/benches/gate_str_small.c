#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef USE512
#define ISA avx512
#else
#define ISA avx2
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
#define DECL(R) \
 void CAT(radix##R##_n1_fwd_,CAT(ISA,_strided))(double*,double*,const double*,const double*,size_t,size_t); \
 void CAT(radix##R##_n1_bwd_,CAT(ISA,_strided))(double*,double*,const double*,const double*,size_t,size_t); \
 void CAT(radix##R##_n1_fwd_,CAT(ISA,_strided_il_out))(const double*,const double*,double*,const double*,const double*,size_t,size_t); \
 void CAT(radix##R##_n1_fwd_,CAT(ISA,_strided_il_out_nt))(const double*,const double*,double*,const double*,const double*,size_t,size_t); \
 void CAT(radix##R##_n1_bwd_,CAT(ISA,_strided_il_in))(const double*,double*,double*,const double*,const double*,size_t,size_t);
DECL(8) DECL(16) DECL(32)
typedef void(*sp_fn)(double*,double*,const double*,const double*,size_t,size_t);
typedef void(*ilo_fn)(const double*,const double*,double*,const double*,const double*,size_t,size_t);
typedef void(*ili_fn)(const double*,double*,double*,const double*,const double*,size_t,size_t);
static int one(int R, sp_fn sf, sp_fn sb, ilo_fn io, ilo_fn nt, ili_fn ii,
               size_t rs, size_t me, int seed){
    size_t big = rs*me + rs;
    double *re=aligned_alloc(64,big*8),*im=aligned_alloc(64,big*8);
    double *r2=aligned_alloc(64,big*8),*i2=aligned_alloc(64,big*8);
    double *z=aligned_alloc(64,2*big*8),*zr=aligned_alloc(64,2*big*8);
    srand(seed); for(size_t i=0;i<big;i++){re[i]=2.0*rand()/RAND_MAX-1;im[i]=2.0*rand()/RAND_MAX-1;}
    /* fwd reference: plain strided on copies + manual interleave */
    memcpy(r2,re,big*8); memcpy(i2,im,big*8);
    sf(r2,i2,NULL,NULL,rs,me);
    for(size_t r=0;r<me;r++) for(int c=0;c<R;c++){
        zr[2*(r*rs+c)]=r2[r*rs+c]; zr[2*(r*rs+c)+1]=i2[r*rs+c]; }
    memset(z,0xAB,2*big*8);
    io(re,im,z,NULL,NULL,rs,me);
    int ok=1;
    for(size_t r=0;r<me && ok;r++)
        ok &= !memcmp(z+2*r*rs, zr+2*r*rs, 2*(size_t)R*8);
    memset(z,0xAB,2*big*8);
    nt(re,im,z,NULL,NULL,rs,me);
    for(size_t r=0;r<me && ok;r++)
        ok &= !memcmp(z+2*r*rs, zr+2*r*rs, 2*(size_t)R*8);
    /* bwd: manual deinterleave -> plain bwd reference; il_in direct */
    srand(seed+1); for(size_t i=0;i<2*big;i++) z[i]=2.0*rand()/RAND_MAX-1;
    for(size_t r=0;r<me;r++) for(int c=0;c<R;c++){
        r2[r*rs+c]=z[2*(r*rs+c)]; i2[r*rs+c]=z[2*(r*rs+c)+1]; }
    sb(r2,i2,NULL,NULL,rs,me);
    memset(re,0xCD,big*8); memset(im,0xCD,big*8);
    ii(z,re,im,NULL,NULL,rs,me);
    for(size_t r=0;r<me && ok;r++){
        ok &= !memcmp(re+r*rs, r2+r*rs, (size_t)R*8);
        ok &= !memcmp(im+r*rs, i2+r*rs, (size_t)R*8); }
    printf("  r%-2d rs=%zu me=%zu : %s\n",R,rs,me,ok?"BIT-EXACT":"**FAIL**");
    free(re);free(im);free(r2);free(i2);free(z);free(zr);
    return ok;
}
#define RUN(R) \
 ok&=one(R,CAT(radix##R##_n1_fwd_,CAT(ISA,_strided)),CAT(radix##R##_n1_bwd_,CAT(ISA,_strided)), \
   CAT(radix##R##_n1_fwd_,CAT(ISA,_strided_il_out)),CAT(radix##R##_n1_fwd_,CAT(ISA,_strided_il_out_nt)), \
   CAT(radix##R##_n1_bwd_,CAT(ISA,_strided_il_in)),(size_t)R,64,7); \
 ok&=one(R,CAT(radix##R##_n1_fwd_,CAT(ISA,_strided)),CAT(radix##R##_n1_bwd_,CAT(ISA,_strided)), \
   CAT(radix##R##_n1_fwd_,CAT(ISA,_strided_il_out)),CAT(radix##R##_n1_fwd_,CAT(ISA,_strided_il_out_nt)), \
   CAT(radix##R##_n1_bwd_,CAT(ISA,_strided_il_in)),(size_t)R+16,8,11);
int main(void){ int ok=1; RUN(8) RUN(16) RUN(32) return ok?0:1; }
