/* bwd z-in tax: c2r split-in vs z-in (current deinterleave-around) vs MKL CCE bwd. */
#include "src/core/vfft.c"
#include "mkl_dfti.h"
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double med11(double *v){ qsort(v,11,8,dcmp); return v[5]; }
int main(int argc,char**argv){
    int N=argc>1?atoi(argv[1]):2000; size_t K=argc>2?(size_t)atoi(argv[2]):4;
    size_t H=(size_t)N/2+1;
    int L=(int)(2e7/((double)N*K)); if(L<20)L=20; if(L>2000)L=2000;
    vfft_wisdom *w=vfft_wisdom_load("/tmp/wbr2c3");
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N; cf.howmany=K; cf.wisdom=w;
    vfft_plan pf=vfft_create(&cf);                 /* r2c SPLIT */
    cf.layout=VFFT_LAYOUT_INTERLEAVED;
    vfft_plan pfz=vfft_create(&cf);                /* r2c CCE */
    cf.transform=VFFT_C2R;
    vfft_plan pbz=vfft_create(&cf);                /* c2r CCE-in */
    cf.layout=VFFT_LAYOUT_SPLIT;
    vfft_plan pb=vfft_create(&cf);                 /* c2r SPLIT-in */
    double *x=aligned_alloc(64,(size_t)N*K*8);
    double *rr=aligned_alloc(64,H*K*8),*ri=aligned_alloc(64,H*K*8);
    double *z=aligned_alloc(64,2*H*K*8);
    double *y=aligned_alloc(64,(size_t)N*K*8),*ym=aligned_alloc(64,(size_t)N*K*8);
    srand(11); for(size_t i=0;i<(size_t)N*K;i++)x[i]=2.0*rand()/RAND_MAX-1;
    vfft_execute(pf,VFFT_FORWARD,x,NULL,rr,ri);
    vfft_execute(pfz,VFFT_FORWARD,x,NULL,z,NULL);
    DFTI_DESCRIPTOR_HANDLE mh;
    DftiCreateDescriptor(&mh,DFTI_DOUBLE,DFTI_REAL,1,(MKL_LONG)N);
    DftiSetValue(mh,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
    DftiSetValue(mh,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mh,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
    DftiSetValue(mh,DFTI_INPUT_DISTANCE,(MKL_LONG)H);
    DftiSetValue(mh,DFTI_OUTPUT_DISTANCE,(MKL_LONG)N);
    DftiCommitDescriptor(mh);
    MKL_Complex16 *mo=aligned_alloc(64,H*K*sizeof(MKL_Complex16));
    { DFTI_DESCRIPTOR_HANDLE mf;
      DftiCreateDescriptor(&mf,DFTI_DOUBLE,DFTI_REAL,1,(MKL_LONG)N);
      DftiSetValue(mf,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
      DftiSetValue(mf,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
      DftiSetValue(mf,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
      DftiSetValue(mf,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
      DftiSetValue(mf,DFTI_OUTPUT_DISTANCE,(MKL_LONG)H);
      DftiCommitDescriptor(mf); DftiComputeForward(mf,x,mo); DftiFreeDescriptor(&mf); }
    double ts[11],tz[11],tm[11];
    for(int wu=0;wu<3;wu++){ vfft_execute(pb,VFFT_BACKWARD,rr,ri,y,NULL);
        vfft_execute(pbz,VFFT_BACKWARD,z,NULL,y,NULL); DftiComputeBackward(mh,mo,ym); }
    for(int t=0;t<11;t++){
        double t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pb,VFFT_BACKWARD,rr,ri,y,NULL);
        ts[t]=(bnow()-t0)/L;
        t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pbz,VFFT_BACKWARD,z,NULL,y,NULL);
        tz[t]=(bnow()-t0)/L;
        t0=bnow(); for(int i=0;i<L;i++) DftiComputeBackward(mh,mo,ym);
        tm[t]=(bnow()-t0)/L;
    }
    double S=med11(ts),Z=med11(tz),M=med11(tm);
    printf("(%d,%zu) bwd: split-in=%.2f  z-in=%.2f (%+.1f%%)  MKL-CCE-bwd=%.2f  z/MKL=%.3fx  split/MKL=%.3fx\n",
        N,K,S,Z,100*(Z-S)/S,M,M/Z,M/S);
    return 0; }
