/* 2D z contract tax: split vs z(convert) vs MKL-2D-CCE, fwd + bwd. */
#include "src/core/vfft.c"
#include "mkl_dfti.h"
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double med(double *v,int n){ qsort(v,n,8,dcmp); return v[n/2]; }
int main(int argc,char**argv){
    int N1=argc>1?atoi(argv[1]):256, N2=argc>2?atoi(argv[2]):256;
    size_t H2=(size_t)N2/2+1, M=(size_t)N1*H2;
    int L=(int)(3e7/((double)N1*N2)); if(L<5)L=5; if(L>500)L=500;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=2; cf.n[0]=N1; cf.n[1]=N2; cf.howmany=1;
    vfft_plan pf=vfft_create(&cf);
    cf.transform=VFFT_C2R;
    vfft_plan pb=vfft_create(&cf);
    double *x=aligned_alloc(64,(size_t)N1*N2*8);
    double *rr=aligned_alloc(64,M*8),*ri=aligned_alloc(64,M*8);
    double *z=aligned_alloc(64,2*M*8);
    double *y=aligned_alloc(64,(size_t)N1*N2*8);
    srand(31); for(size_t i=0;i<(size_t)N1*N2;i++)x[i]=2.0*rand()/RAND_MAX-1;
    DFTI_DESCRIPTOR_HANDLE mf,mb;
    MKL_LONG dims[2]={N1,N2};
    MKL_LONG is[3]={0,(MKL_LONG)N2,1}, os[3]={0,(MKL_LONG)H2,1};
    DftiCreateDescriptor(&mf,DFTI_DOUBLE,DFTI_REAL,2,dims);
    DftiSetValue(mf,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mf,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
    DftiSetValue(mf,DFTI_INPUT_STRIDES,is); DftiSetValue(mf,DFTI_OUTPUT_STRIDES,os);
    DftiCommitDescriptor(mf);
    DftiCreateDescriptor(&mb,DFTI_DOUBLE,DFTI_REAL,2,dims);
    DftiSetValue(mb,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mb,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
    DftiSetValue(mb,DFTI_INPUT_STRIDES,os); DftiSetValue(mb,DFTI_OUTPUT_STRIDES,is);
    DftiCommitDescriptor(mb);
    MKL_Complex16 *mo=aligned_alloc(64,M*sizeof(MKL_Complex16));
    double *ym=aligned_alloc(64,(size_t)N1*N2*8);
    vfft_execute(pf,VFFT_FORWARD,x,NULL,rr,ri);
    vfft_execute(pf,VFFT_FORWARD,x,NULL,z,NULL);
    DftiComputeForward(mf,x,mo);
    double fs[9],fz[9],fm[9],bs[9],bz[9],bm[9];
    for(int t=0;t<9;t++){
        double t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pf,VFFT_FORWARD,x,NULL,rr,ri);
        fs[t]=(bnow()-t0)/L;
        t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pf,VFFT_FORWARD,x,NULL,z,NULL);
        fz[t]=(bnow()-t0)/L;
        t0=bnow(); for(int i=0;i<L;i++) DftiComputeForward(mf,x,mo);
        fm[t]=(bnow()-t0)/L;
        t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pb,VFFT_BACKWARD,rr,ri,y,NULL);
        bs[t]=(bnow()-t0)/L;
        t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pb,VFFT_BACKWARD,z,NULL,y,NULL);
        bz[t]=(bnow()-t0)/L;
        t0=bnow(); for(int i=0;i<L;i++) DftiComputeBackward(mb,mo,ym);
        bm[t]=(bnow()-t0)/L;
    }
    double FS=med(fs,9),FZ=med(fz,9),FM=med(fm,9),BS=med(bs,9),BZ=med(bz,9),BM=med(bm,9);
    printf("(%dx%d) fwd: split=%.1f z=%.1f (%+.1f%%) MKL=%.1f  z/MKL=%.3fx split/MKL=%.3fx\n",
        N1,N2,FS,FZ,100*(FZ-FS)/FS,FM,FM/FZ,FM/FS);
    printf("(%dx%d) bwd: split=%.1f z=%.1f (%+.1f%%) MKL=%.1f  z/MKL=%.3fx split/MKL=%.3fx\n",
        N1,N2,BS,BZ,100*(BZ-BS)/BS,BM,BM/BZ,BM/BS);
    return 0; }
