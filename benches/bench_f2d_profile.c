#include "src/core/vfft.c"
int main(int argc,char**argv){
    int N1=argc>1?atoi(argv[1]):256, N2=argc>2?atoi(argv[2]):256;
    size_t H2=(size_t)N2/2+1, M=(size_t)N1*H2;
    int L=(int)(3e7/((double)N1*N2)); if(L<5)L=5; if(L>200)L=200;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=2; cf.n[0]=N1; cf.n[1]=N2; cf.howmany=1;
    vfft_plan pf=vfft_create(&cf);
    double *x=aligned_alloc(64,(size_t)N1*N2*8);
    double *rr=aligned_alloc(64,M*8),*ri=aligned_alloc(64,M*8);
    srand(31); for(size_t i=0;i<(size_t)N1*N2;i++)x[i]=2.0*rand()/RAND_MAX-1;
    for(int w=0;w<3;w++) vfft_execute(pf,VFFT_FORWARD,x,NULL,rr,ri);
    _f2d_wrapin=_f2d_p1_tin=_f2d_p1_r2c=_f2d_p1_tout=_f2d_p2=_f2d_p3=_f2d_wrapout=0;
    struct timespec t0,t1; clock_gettime(CLOCK_MONOTONIC,&t0);
    for(int i=0;i<L;i++) vfft_execute(pf,VFFT_FORWARD,x,NULL,rr,ri);
    clock_gettime(CLOCK_MONOTONIC,&t1);
    double tot=((t1.tv_sec-t0.tv_sec)*1e6+(t1.tv_nsec-t0.tv_nsec)*1e-3)/L;
    double wi=_f2d_wrapin/L, ti=_f2d_p1_tin/L, r2=_f2d_p1_r2c/L,
           to=_f2d_p1_tout/L, p2=_f2d_p2/L, p3=_f2d_p3/L, wo=_f2d_wrapout/L;
    double acc=wi+ti+r2+to+p2+p3+wo;
    printf("(%dx%d) total=%.1f us\n",N1,N2,tot);
    printf("  wrapper-in    %7.1f  (%4.1f%%)\n",wi,100*wi/tot);
    printf("  p1 transp-in  %7.1f  (%4.1f%%)\n",ti,100*ti/tot);
    printf("  p1 inner-r2c  %7.1f  (%4.1f%%)\n",r2,100*r2/tot);
    printf("  p1 transp-out %7.1f  (%4.1f%%)\n",to,100*to/tot);
    printf("  p2 col-c2c    %7.1f  (%4.1f%%)\n",p2,100*p2/tot);
    printf("  p3 pack       %7.1f  (%4.1f%%)\n",p3,100*p3/tot);
    printf("  wrapper-out   %7.1f  (%4.1f%%)\n",wo,100*wo/tot);
    printf("  [accounted %.1f = %.1f%%; copies (wi+to-pad+p3+wo lower bound) = %.1f%%]\n",
        acc,100*acc/tot,100*(wi+p3+wo)/tot);
    return 0; }
