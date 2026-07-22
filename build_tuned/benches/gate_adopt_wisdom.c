#include "src/core/vfft.c"
#include <time.h>
static double cms(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e3+t.tv_nsec*1e-6;}
static void one(int N1,int N2,int*f,int*b,double*ms){
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=2; cf.n[0]=N1; cf.n[1]=N2; cf.howmany=1;
    double t0=cms(); vfft_plan pf=vfft_create(&cf); *ms=cms()-t0;
    stride_fft2d_r2c_data_t *d=(stride_fft2d_r2c_data_t*)((struct vfft_plan_s*)pf)->tplan->override_data;
    *f=d->strided_fwd?1:0; *b=d->strided_bwd?1:0;
    vfft_destroy(pf);
}
int main(int argc,char**argv){
    int f1,b1,f2,b2; double m1,m2;
    one(64,64,&f1,&b1,&m1); one(256,32,&f2,&b2,&m2);
    printf("run: (64,64) f=%d b=%d %.1fms | (256,32) f=%d b=%d %.1fms\n",f1,b1,m1,f2,b2,m2);
    if(argc>1){ /* verify mode: expect args f1 b1 f2 b2 to match */
        int e[4]={atoi(argv[1]),atoi(argv[2]),atoi(argv[3]),atoi(argv[4])};
        int ok = (f1==e[0]&&b1==e[1]&&f2==e[2]&&b2==e[3]);
        printf("decision match: %s\n", ok?"PASS":"**FAIL**");
        return ok?0:1;
    }
    printf("%d %d %d %d\n",f1,b1,f2,b2);
    return 0; }
