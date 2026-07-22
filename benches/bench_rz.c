/* §6a24 public-boundary, drift-proof: interleaved split/z trials + MKL CCE. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <mkl_dfti.h>
#include "vfft.h"
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double med(double *v){ qsort(v,11,8,dcmp); return v[5]; }
int main(void){
    int N=512; size_t K=256, H=(size_t)N/2+1; int L=40;
    vfft_wisdom *w=vfft_wisdom_load("/tmp/wbr2c3");
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N; cf.howmany=K; cf.wisdom=w;
    vfft_plan p=vfft_create(&cf);
    double *x=aligned_alloc(64,(size_t)N*K*8);
    double *rr=aligned_alloc(64,H*K*8),*ri=aligned_alloc(64,H*K*8);
    double *z=aligned_alloc(64,2*H*K*8);
    srand(3); for(size_t i=0;i<(size_t)N*K;i++)x[i]=2.0*rand()/RAND_MAX-1;
    DFTI_DESCRIPTOR_HANDLE mh;
    DftiCreateDescriptor(&mh,DFTI_DOUBLE,DFTI_REAL,1,(MKL_LONG)N);
    DftiSetValue(mh,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(mh,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mh,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
    DftiSetValue(mh,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
    DftiSetValue(mh,DFTI_OUTPUT_DISTANCE,(MKL_LONG)H);
    DftiCommitDescriptor(mh);
    MKL_Complex16 *mo=aligned_alloc(64,H*K*sizeof(MKL_Complex16));
    double ts[11],tz[11],tm[11];
    for(int wu=0;wu<3;wu++){ vfft_execute(p,VFFT_FORWARD,x,NULL,rr,ri);
        vfft_execute(p,VFFT_FORWARD,x,NULL,z,NULL); DftiComputeForward(mh,x,mo); }
    for(int t=0;t<11;t++){
        double t0=bnow(); for(int i=0;i<L;i++) vfft_execute(p,VFFT_FORWARD,x,NULL,rr,ri);
        ts[t]=(bnow()-t0)/L;
        t0=bnow(); for(int i=0;i<L;i++) vfft_execute(p,VFFT_FORWARD,x,NULL,z,NULL);
        tz[t]=(bnow()-t0)/L;
        t0=bnow(); for(int i=0;i<L;i++) DftiComputeForward(mh,x,mo);
        tm[t]=(bnow()-t0)/L;
    }
    double S=med(ts),Z=med(tz),M=med(tm);
    printf("(512,256) public: split=%.2f  z=%.2f (%+.1f%%)  MKL-CCE=%.2f  z/MKL=%.3fx  split/MKL=%.3fx\n",
        S,Z,100*(Z-S)/S,M,M/Z,M/S);
    vfft_destroy(p); if(w)vfft_wisdom_free(w);
    DftiFreeDescriptor(&mh); free(x);free(rr);free(ri);free(z);free(mo);
    return 0; }
