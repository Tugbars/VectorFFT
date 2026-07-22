/* §6a24 drift-proof: round-robin the three arms within each trial so all
 * medians see the same container weather. */
#include "src/core/vfft.c"
static double bnow2(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp2(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double med11(double *v){ qsort(v,11,8,dcmp2); return v[5]; }
int main(void){
    int N=512; size_t K=256, H=(size_t)N/2+1; int L=40;
    vfft_wisdom *w=vfft_wisdom_load("/tmp/wbr2c3");
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N; cf.howmany=K; cf.wisdom=w;
    vfft_plan ph=vfft_create(&cf);
    struct vfft_plan_s *h=(struct vfft_plan_s*)ph;
    stride_r2c_data_t *d=(stride_r2c_data_t*)h->rplan->stride->override_data;
    stride_plan_t *sp=h->rplan->stride;
    double *x=aligned_alloc(64,(size_t)N*K*8);
    double *rr=aligned_alloc(64,(size_t)N*K*8),*ri=aligned_alloc(64,(size_t)N*K*8);
    double *z=aligned_alloc(64,2*H*K*8);
    srand(3); for(size_t i=0;i<(size_t)N*K;i++)x[i]=2.0*rand()/RAND_MAX-1;
    double ta[11],tb[11],tc[11];
    for(int wu=0;wu<3;wu++){ stride_execute_r2c(sp,x,rr,ri);
        _r2c_execute_fwd_oop(d,x,rr,ri);
        d->zo=z; _r2c_execute_fwd_oop(d,x,NULL,NULL); d->zo=NULL; }
    for(int t=0;t<11;t++){
        double t0=bnow2(); for(int i=0;i<L;i++) stride_execute_r2c(sp,x,rr,ri);
        ta[t]=(bnow2()-t0)/L;
        t0=bnow2(); for(int i=0;i<L;i++) _r2c_execute_fwd_oop(d,x,rr,ri);
        tb[t]=(bnow2()-t0)/L;
        t0=bnow2(); d->zo=z; for(int i=0;i<L;i++) _r2c_execute_fwd_oop(d,x,NULL,NULL);
        d->zo=NULL; tc[t]=(bnow2()-t0)/L;
    }
    double A=med11(ta),B=med11(tb),C=med11(tc);
    printf("A inplace+copy split = %7.2f us\nB oop split          = %7.2f us  (vs A: %+.1f%%)\nC oop z              = %7.2f us  (vs B: %+.1f%%, vs A: %+.1f%%)\n",
        A,B,100*(B-A)/A,C,100*(C-B)/B,100*(C-A)/A);
    vfft_destroy(ph); if(w)vfft_wisdom_free(w);
    free(x);free(rr);free(ri);free(z); return 0; }
