#include "src/core/vfft.c"
#include <math.h>
static double bn(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec*1e-3;}
static int dc(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
int main(void){
    vfft_proto_registry_init(&_reg);
    int N=atoi(getenv("BN")); size_t K=atoi(getenv("BK")),B=K;
    int factors[2]={atoi(getenv("BF0")),atoi(getenv("BF1"))}, vdif[2]={0,0};
    stride_plan_t *inner=vfft_proto_plan_create_ex(N/2,K,factors,vdif,2,1,&_reg);
    stride_plan_t *sp=stride_r2c_plan(N,K,B,inner);
    if(!sp){printf("plan FAIL\n");return 1;}
    double *x=aligned_alloc(64,(size_t)N*K*8),*orr=aligned_alloc(64,(size_t)N*K*8),*oi=aligned_alloc(64,(size_t)N*K*8);
    srand(77); for(size_t i=0;i<(size_t)N*K;i++)x[i]=2.0*rand()/RAND_MAX-1;
    setenv("VFFT_DIF_FUSED","1",1);
    stride_execute_r2c(sp,x,orr,oi);
    double *fr=malloc((size_t)N*K*8),*fi=malloc((size_t)N*K*8);
    memcpy(fr,orr,(size_t)N*K*8); memcpy(fi,oi,(size_t)N*K*8);
    unsetenv("VFFT_DIF_FUSED");
    stride_execute_r2c(sp,x,orr,oi);
    setenv("VFFT_DIF_FUSED","1",1);
    double mx=0; size_t H=(size_t)N/2+1;
    for(size_t i=0;i<H*K;i++){double m1=fabs(orr[i])>1?fabs(orr[i]):1;
        double d=fabs(fr[i]-orr[i])/m1; if(d>mx)mx=d;
        m1=fabs(oi[i])>1?fabs(oi[i]):1; d=fabs(fi[i]-oi[i])/m1; if(d>mx)mx=d;}
    printf("fused-vs-explicit rel: %.2e %s\n",mx,mx<1e-11?"OK":"**BAD**");
    int L=200; double tf[9],te[9],t0;
    for(int t=0;t<9;t++){
        setenv("VFFT_DIF_FUSED","1",1);
        t0=bn(); for(int i=0;i<L;i++) stride_execute_r2c(sp,x,orr,oi); tf[t]=(bn()-t0)/L;
        unsetenv("VFFT_DIF_FUSED");
        t0=bn(); for(int i=0;i<L;i++) stride_execute_r2c(sp,x,orr,oi); te[t]=(bn()-t0)/L;
    }
    qsort(tf,9,8,dc); qsort(te,9,8,dc);
    printf("(%d,{%d,%d},B=K=%zu) r2c fwd: explicit=%.2fus fused=%.2fus (fused %+.1f%%)\n",
        N,factors[0],factors[1],K,te[4],tf[4],100*(tf[4]-te[4])/te[4]);
    return 0; }
