/* #8 generalization: A vs B(avx512) across geometries, r8-last-stage only. */
#include "src/core/vfft.c"
extern void radix256_r2c_term_ls_r8_fwd_avx512(
    const double*, const double*, const double*, const double*,
    double*, double*, double*, double*, const double*, const double*,
    ptrdiff_t, ptrdiff_t, ptrdiff_t, size_t);
static double bnow2(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp2(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static double med11(double *v){ qsort(v,11,8,dcmp2); return v[5]; }
static void cell(vfft_wisdom *w, int N, size_t K){
    size_t H=(size_t)N/2+1;
    int L = (int)(4e7/((double)N*K)); if(L<8)L=8; if(L>200)L=200;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N; cf.howmany=K; cf.wisdom=w;
    vfft_plan ph=vfft_create(&cf);
    struct vfft_plan_s *h=(struct vfft_plan_s*)ph;
    if(!h->rplan||h->rplan->path!=VFFT_R2C_PATH_STRIDE){
        printf("(%5d,%4zu) SKIP: rfft dispatch\n",N,K); vfft_destroy(ph); return;}
    stride_r2c_data_t *d=(stride_r2c_data_t*)h->rplan->stride->override_data;
    stride_plan_t *ip=d->inner;
    const stride_stage_t *last=&ip->stages[ip->num_stages-1];
    char shape[64]={0}; for(int s=0;s<ip->num_stages;s++)
        snprintf(shape+strlen(shape),8," r%d",ip->stages[s].radix);
    if(last->radix!=8){
        printf("(%5d,%4zu) SKIP: last radix %d [%s]\n",N,K,last->radix,shape+1);
        vfft_destroy(ph); return;}
    if(d->inner_jit_fwd){
        printf("(%5d,%4zu) SKIP: jit inner bound (slice_until has no jit) [%s]\n",N,K,shape+1);
        vfft_destroy(ph); return;}
    double *x=aligned_alloc(64,(size_t)N*K*8);
    double *rr=aligned_alloc(64,H*K*8),*ri=aligned_alloc(64,H*K*8);
    double *br=aligned_alloc(64,H*K*8),*bi=aligned_alloc(64,H*K*8);
    srand(3+N); for(size_t i=0;i<(size_t)N*K;i++)x[i]=2.0*rand()/RAND_MAX-1;
    d->ls_fwd=NULL; vfft_execute(ph,VFFT_FORWARD,x,NULL,rr,ri);
    d->ls_fwd=radix256_r2c_term_ls_r8_fwd_avx512;
    vfft_execute(ph,VFFT_FORWARD,x,NULL,br,bi);
    size_t bad=0;
    for(size_t i=0;i<H*K;i++)
        if(fabs(br[i]-rr[i])>1e-9||fabs(bi[i]-ri[i])>1e-9)bad++;
    if(bad){printf("(%5d,%4zu) **CORRECTNESS FAIL** BAD=%zu [%s]\n",N,K,bad,shape+1);
        d->ls_fwd=NULL; vfft_destroy(ph);
        free(x);free(rr);free(ri);free(br);free(bi); return;}
    double ta[11],tb[11];
    for(int t=0;t<11;t++){
        d->ls_fwd=NULL;
        double t0=bnow2(); for(int i=0;i<L;i++) vfft_execute(ph,VFFT_FORWARD,x,NULL,rr,ri);
        ta[t]=(bnow2()-t0)/L;
        d->ls_fwd=radix256_r2c_term_ls_r8_fwd_avx512;
        t0=bnow2(); for(int i=0;i<L;i++) vfft_execute(ph,VFFT_FORWARD,x,NULL,br,bi);
        tb[t]=(bnow2()-t0)/L;
    }
    double A=med11(ta),B=med11(tb);
    printf("(%5d,%4zu) A=%9.2f  B512=%9.2f  %+6.1f%%  [%s]\n",N,K,A,B,100*(B-A)/A,shape+1);
    d->ls_fwd=NULL;
    vfft_destroy(ph);
    free(x);free(rr);free(ri);free(br);free(bi);
}
int main(int argc,char**argv){
    vfft_wisdom *w=vfft_wisdom_load("/tmp/wbr2c3");
    for(int i=1;i+1<argc;i+=2) cell(w, atoi(argv[i]), (size_t)atoi(argv[i+1]));
    if(w)vfft_wisdom_free(w);
    return 0; }
