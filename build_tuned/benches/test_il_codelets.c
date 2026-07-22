/* Bit gates for the derived IL codelets vs (convert + split original). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "prime_dispatch.h"
#include "il_layout.h"
#ifdef USE512
#define ISA avx512
#define VW 8
#else
#define ISA avx2
#define VW 4
#endif
#define CAT2(a,b) a##b
#define CAT(a,b) CAT2(a,b)
void CAT(radix64_n1_oop_fwd_,CAT(ISA,_UG_UG))(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void CAT(radix64_n1_oop_fwd_,CAT(ISA,_UG_UG_il_in))(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void CAT(radix64_n1_oop_fwd_,CAT(ISA,_UG_UG_il_out_sw))(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
void CAT(radix64_n1_fwd_,CAT(ISA,_strided))(double*,double*,const double*,const double*,size_t,size_t);
void CAT(radix64_n1_bwd_,CAT(ISA,_strided))(double*,double*,const double*,const double*,size_t,size_t);
void CAT(radix64_n1_fwd_,CAT(ISA,_strided_il_out))(const double*,const double*,double*,const double*,const double*,size_t,size_t);
void CAT(radix64_n1_bwd_,CAT(ISA,_strided_il_in))(const double*,double*,double*,const double*,const double*,size_t,size_t);
static int g=0;
static void chk(const char*t,int ok){ if(!ok)g++; printf("  %-40s %s\n",t,ok?"BIT":"**FAIL**"); }
int main(void){
    const int N=64;
    size_t mes[2]={64,61};
    for(int c=0;c<2;c++){
        size_t me=mes[c], n=(size_t)N*me;
        double *z=aligned_alloc(64,(n+8)*16),*re=aligned_alloc(64,(n+8)*8),*im=aligned_alloc(64,(n+8)*8);
        double *oR=aligned_alloc(64,(n+8)*8),*oI=aligned_alloc(64,(n+8)*8);
        double *dR=aligned_alloc(64,(n+8)*8),*dI=aligned_alloc(64,(n+8)*8);
        double *dz=aligned_alloc(64,(n+8)*16),*ez=aligned_alloc(64,(n+8)*16);
        srand(5+(int)me);
        for(size_t i=0;i<n;i++){re[i]=2.0*rand()/RAND_MAX-1; im[i]=2.0*rand()/RAND_MAX-1;
                                z[2*i]=re[i]; z[2*i+1]=im[i];}
        /* 1) oop il_in fwd: derived(z) == orig(split) */
        CAT(radix64_n1_oop_fwd_,CAT(ISA,_UG_UG))(re,im,oR,oI,NULL,NULL,me,1,me,1,me);
        CAT(radix64_n1_oop_fwd_,CAT(ISA,_UG_UG_il_in))(z,NULL,dR,dI,NULL,NULL,me,1,me,1,me);
        char b[64]; snprintf(b,64,"oop il_in me=%zu",me);
        chk(b, !memcmp(oR,dR,n*8)&&!memcmp(oI,dI,n*8));
        /* 2) oop il_out_sw: derived(Xim,Xre)->z == flip-interleave(orig(Xim,Xre)) */
        CAT(radix64_n1_oop_fwd_,CAT(ISA,_UG_UG))(im,re,oR,oI,NULL,NULL,me,1,me,1,me);
        for(size_t f=0;f<n;f++){ez[2*f]=oI[f];ez[2*f+1]=oR[f];}
        CAT(radix64_n1_oop_fwd_,CAT(ISA,_UG_UG_il_out_sw))(im,re,dz,NULL,NULL,NULL,me,1,me,1,me);
        snprintf(b,64,"oop il_out_sw me=%zu",me);
        chk(b, !memcmp(dz,ez,n*16));
        free(z);free(re);free(im);free(oR);free(oI);free(dR);free(dI);free(dz);free(ez);
    }
    /* strided: me multiple of VW only */
    {
        size_t me=64, n=(size_t)N*me;
        double *re=aligned_alloc(64,n*8),*im=aligned_alloc(64,n*8);
        double *cr=aligned_alloc(64,n*8),*ci=aligned_alloc(64,n*8);
        double *z=aligned_alloc(64,n*16),*ez=aligned_alloc(64,n*16);
        double *dR=aligned_alloc(64,n*8),*dI=aligned_alloc(64,n*8);
        srand(9);
        for(size_t i=0;i<n;i++){re[i]=2.0*rand()/RAND_MAX-1;im[i]=2.0*rand()/RAND_MAX-1;}
        /* 3) strided il_out fwd */
        memcpy(cr,re,n*8); memcpy(ci,im,n*8);
        CAT(radix64_n1_fwd_,CAT(ISA,_strided))(cr,ci,NULL,NULL,(size_t)N,me);
        vfft_sp2il(cr,ci,ez,n);
        CAT(radix64_n1_fwd_,CAT(ISA,_strided_il_out))(re,im,z,NULL,NULL,(size_t)N,me);
        chk("strided il_out fwd me=64", !memcmp(z,ez,n*16));
        /* 4) strided il_in bwd */
        srand(10); for(size_t i=0;i<2*n;i++) z[i]=2.0*rand()/RAND_MAX-1;
        vfft_il2sp(z,cr,ci,n);
        CAT(radix64_n1_bwd_,CAT(ISA,_strided))(cr,ci,NULL,NULL,(size_t)N,me);
        CAT(radix64_n1_bwd_,CAT(ISA,_strided_il_in))(z,dR,dI,NULL,NULL,(size_t)N,me);
        chk("strided il_in bwd me=64", !memcmp(dR,cr,n*8)&&!memcmp(dI,ci,n*8));
        free(re);free(im);free(cr);free(ci);free(z);free(ez);free(dR);free(dI);
    }
    printf(g?"%d FAIL\n":"ALL BIT-EXACT\n",g);
    return g;
}
