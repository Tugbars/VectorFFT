/* §6a31/32 verdict: same-process adopted-vs-stride, fwd + bwd. */
#include "src/core/vfft.c"
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double med9(double *v){ qsort(v,9,8,dcmp); return v[4]; }
int main(int argc,char**argv){
    int N1=argc>1?atoi(argv[1]):256, N2=argc>2?atoi(argv[2]):256;
    size_t H2=(size_t)N2/2+1, M=(size_t)N1*H2;
    int L=(int)(2e7/((double)N1*N2)); if(L<5)L=5; if(L>200)L=200;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=2; cf.n[0]=N1; cf.n[1]=N2; cf.howmany=1;
    vfft_plan pf=vfft_create(&cf);
    cf.transform=VFFT_C2R; vfft_plan pb=vfft_create(&cf);
    struct vfft_plan_s *hf=(struct vfft_plan_s*)pf, *hb=(struct vfft_plan_s*)pb;
    stride_fft2d_r2c_data_t *df=(stride_fft2d_r2c_data_t*)hf->tplan->override_data;
    stride_fft2d_r2c_data_t *db=(stride_fft2d_r2c_data_t*)hb->tplan->override_data;
    printf("(%dx%d) fwd adopted=%s  bwd adopted=%s\n",N1,N2,
        df->rfft_row?"RFFT":"stride", db->c2r_row?"C2R-NAT":"stride");
    double *x=aligned_alloc(64,(size_t)N1*N2*8);
    double *rr=aligned_alloc(64,M*8),*ri=aligned_alloc(64,M*8);
    double *y=aligned_alloc(64,(size_t)N1*N2*8);
    srand(61); for(size_t i=0;i<(size_t)N1*N2;i++)x[i]=2.0*rand()/RAND_MAX-1;
    vfft_execute(pf,VFFT_FORWARD,x,NULL,rr,ri);
    vfft_execute(pb,VFFT_BACKWARD,rr,ri,y,NULL);
    double fa[9],fb[9],ba[9],bb[9];
    rfft_plan_t *rsave=df->rfft_row; c2r_plan_t *csave=db->c2r_row;
    for(int t=0;t<9;t++){
        df->rfft_row=rsave;
        double t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pf,VFFT_FORWARD,x,NULL,rr,ri);
        fa[t]=(bnow()-t0)/L;
        df->rfft_row=NULL;
        t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pf,VFFT_FORWARD,x,NULL,rr,ri);
        fb[t]=(bnow()-t0)/L;
        db->c2r_row=csave;
        t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pb,VFFT_BACKWARD,rr,ri,y,NULL);
        ba[t]=(bnow()-t0)/L;
        db->c2r_row=NULL;
        t0=bnow(); for(int i=0;i<L;i++) vfft_execute(pb,VFFT_BACKWARD,rr,ri,y,NULL);
        bb[t]=(bnow()-t0)/L;
    }
    df->rfft_row=rsave; db->c2r_row=csave;
    double FA=med9(fa),FB=med9(fb),BA=med9(ba),BB=med9(bb);
    printf("  fwd: adopted=%.1f stride=%.1f  -> engine delta %+.1f%%\n",FA,FB,100*(FA-FB)/FB);
    printf("  bwd: adopted=%.1f stride=%.1f  -> engine delta %+.1f%%\n",BA,BB,100*(BA-BB)/BB);
    return 0; }
