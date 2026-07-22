#include "src/core/vfft.c"
#include "mkl_dfti.h"
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double med9(double *v){ qsort(v,9,8,dcmp); return v[4]; }
int main(int argc,char**argv){
    int N1=argc>1?atoi(argv[1]):4096, N2=argc>2?atoi(argv[2]):64;
    size_t H2=(size_t)N2/2+1, M=(size_t)N1*H2;
    int L=(int)(3e7/((double)N1*N2)); if(L<5)L=5; if(L>400)L=400;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=2; cf.n[0]=N1; cf.n[1]=N2; cf.howmany=1;
    vfft_plan pf=vfft_create(&cf);
    cf.transform=VFFT_C2R; vfft_plan pb=vfft_create(&cf);
    struct vfft_plan_s *hf=(struct vfft_plan_s*)pf, *hb=(struct vfft_plan_s*)pb;
    stride_fft2d_r2c_data_t *df=(stride_fft2d_r2c_data_t*)hf->tplan->override_data;
    stride_fft2d_r2c_data_t *db=(stride_fft2d_r2c_data_t*)hb->tplan->override_data;
    printf("(%dx%d) adoption: fwd=%s bwd=%s\n",N1,N2,
        df->strided_fwd?"MONO":(df->stw_on_fwd?"STW":"tiled"), db->strided_bwd?"MONO":(db->stw_on_bwd?"STW":"tiled"));
    double *x=aligned_alloc(64,(size_t)N1*N2*8);
    double *rr=aligned_alloc(64,M*8),*ri=aligned_alloc(64,M*8);
    double *y=aligned_alloc(64,(size_t)N1*N2*8);
    srand(107); for(size_t i=0;i<(size_t)N1*N2;i++)x[i]=2.0*rand()/RAND_MAX-1;
    MKL_LONG dims[2]={N1,N2}, is[3]={0,(MKL_LONG)N2,1}, os[3]={0,(MKL_LONG)H2,1};
    DFTI_DESCRIPTOR_HANDLE mr;
    DftiCreateDescriptor(&mr,DFTI_DOUBLE,DFTI_REAL,2,dims);
    DftiSetValue(mr,DFTI_CONJUGATE_EVEN_STORAGE,DFTI_COMPLEX_COMPLEX);
    DftiSetValue(mr,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
    DftiSetValue(mr,DFTI_INPUT_STRIDES,is); DftiSetValue(mr,DFTI_OUTPUT_STRIDES,os);
    DftiCommitDescriptor(mr);
    MKL_Complex16 *mo=aligned_alloc(64,M*sizeof(MKL_Complex16));
    vfft_execute(pf,VFFT_FORWARD,x,NULL,rr,ri);
    vfft_execute(pb,VFFT_BACKWARD,rr,ri,y,NULL);
    DftiComputeForward(mr,x,mo);
    _f2d_sr2c_fwd_fn fsave=df->strided_fwd; _f2d_sr2c_bwd_fn bsave=db->strided_bwd; int sf1=df->stw_on_fwd, sb1=db->stw_on_bwd;
    double fa[9],fb[9],ba[9],bb[9],fm[9];
    for(int t=0;t<9;t++){
        df->strided_fwd=fsave; df->stw_on_fwd=sf1;
        double t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pf,VFFT_FORWARD,x,NULL,rr,ri);
        fa[t]=(bnow()-t0)/L;
        df->strided_fwd=NULL; df->stw_on_fwd=0;
        t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pf,VFFT_FORWARD,x,NULL,rr,ri);
        fb[t]=(bnow()-t0)/L;
        db->strided_bwd=bsave; db->stw_on_bwd=sb1;
        t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pb,VFFT_BACKWARD,rr,ri,y,NULL);
        ba[t]=(bnow()-t0)/L;
        db->strided_bwd=NULL; db->stw_on_bwd=0;
        t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pb,VFFT_BACKWARD,rr,ri,y,NULL);
        bb[t]=(bnow()-t0)/L;
        t0=bnow(); for(int i=0;i<L;i++) DftiComputeForward(mr,x,mo);
        fm[t]=(bnow()-t0)/L;
    }
    df->strided_fwd=fsave; df->stw_on_fwd=sf1; db->strided_bwd=bsave; db->stw_on_bwd=sb1; db->stw_on_bwd=sb1;
    double FA=med9(fa),FB=med9(fb),BA=med9(ba),BB=med9(bb),FM=med9(fm);
    printf("  fwd: strided=%.1f tiled=%.1f  -> %+.1f%%   MKL=%.1f  split/MKL %.3fx -> %.3fx\n",
        FA,FB,100*(FA-FB)/FB,FM,FM/FB,FM/FA);
    printf("  bwd: strided=%.1f tiled=%.1f  -> %+.1f%%\n",BA,BB,100*(BA-BB)/BB);
    return 0; }
