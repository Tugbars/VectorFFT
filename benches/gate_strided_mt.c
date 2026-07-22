#include "src/core/vfft.c"
int main(void){
    enum { N1=64, N2=64, H=33 };
    double x[N1*N2], r1[N1*H], i1[N1*H], rT[N1*H], iT[N1*H], y1[N1*N2], yT[N1*N2];
    srand(131); for(int i=0;i<N1*N2;i++) x[i]=2.0*rand()/RAND_MAX-1;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=2; cf.n[0]=N1; cf.n[1]=N2; cf.howmany=1;
    vfft_plan pf=vfft_create(&cf); cf.transform=VFFT_C2R; vfft_plan pb=vfft_create(&cf);
    struct vfft_plan_s *hf=(struct vfft_plan_s*)pf;
    stride_fft2d_r2c_data_t *df=(stride_fft2d_r2c_data_t*)hf->tplan->override_data;
    printf("adoption fwd=%s  pool_size=%d\n", df->strided_fwd?"MONO":"other", _stride_pool_size);
    stride_set_num_threads(1);
    vfft_execute(pf,VFFT_FORWARD,x,NULL,r1,i1);
    vfft_execute(pb,VFFT_BACKWARD,r1,i1,y1,NULL);
    for(int T=2;T<=4;T+=2){
        stride_set_num_threads(T);
        printf("T=%d pool=%d: ",T,_stride_pool_size);
        vfft_execute(pf,VFFT_FORWARD,x,NULL,rT,iT);
        vfft_execute(pb,VFFT_BACKWARD,rT,iT,yT,NULL);
        size_t d1=0,d2=0;
        for(int i=0;i<N1*H;i++) if(rT[i]!=r1[i]||iT[i]!=i1[i]) d1++;
        for(int i=0;i<N1*N2;i++) if(yT[i]!=y1[i]) d2++;
        printf("fwd %s (%zu)  bwd %s (%zu)\n", d1?"**DIFF**":"BIT",d1, d2?"**DIFF**":"BIT",d2);
        if(d1||d2) return 1;
    }
    stride_set_num_threads(1);
    printf("MT T-invariance: ALL BIT\n");
    return 0; }
