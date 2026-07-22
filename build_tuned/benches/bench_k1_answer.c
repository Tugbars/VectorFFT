#include "src/core/vfft.c"
#include <math.h>
static double bn(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec*1e-3;}
static int dc(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static void run(int N){
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_C2C; cf.placement=VFFT_INPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N;
    double *z=aligned_alloc(64,2*(size_t)N*8*8), *w=aligned_alloc(64,2*(size_t)N*8*8);
    srand(3); for(size_t i=0;i<2*(size_t)N*8;i++) z[i]=2.0*rand()/RAND_MAX-1;
    /* a: split K=1 (the scalar tier) */
    cf.howmany=1; vfft_plan pa=vfft_create(&cf);
    double *ar=aligned_alloc(64,(size_t)N*8), *ai=aligned_alloc(64,(size_t)N*8);
    for(int i=0;i<N;i++){ar[i]=z[2*i];ai[i]=z[2*i+1];}
    /* b: z K=1 padded arm (Kp=8 full-width) ; c: z K=1 fused (scalar folds) */
    setenv("VFFT_IL_PAD","1",1); vfft_plan pb=vfft_create(&cf);
    memcpy(w,z,2*(size_t)N*8); vfft_execute(pb,VFFT_FORWARD,w,NULL,w,NULL);
    setenv("VFFT_IL_PAD","0",1); vfft_plan pc=vfft_create(&cf);
    memcpy(w,z,2*(size_t)N*8); vfft_execute(pc,VFFT_FORWARD,w,NULL,w,NULL);
    unsetenv("VFFT_IL_PAD");
    /* d: K=8 batch ceiling */
    cf.howmany=8; vfft_plan pd=vfft_create(&cf);
    int L=(int)(200000.0/(N*0.02))+16;
    double ta[9],tb[9],tc[9],td[9],t0;
    for(int r=0;r<9;r++){
        t0=bn(); for(int i=0;i<L;i++) vfft_execute(pa,VFFT_FORWARD,ar,ai,ar,ai); ta[r]=(bn()-t0)/L;
        t0=bn(); for(int i=0;i<L;i++){ memcpy(w,z,2*(size_t)N*8); vfft_execute(pb,VFFT_FORWARD,w,NULL,w,NULL);} tb[r]=(bn()-t0)/L;
        t0=bn(); for(int i=0;i<L;i++){ memcpy(w,z,2*(size_t)N*8); vfft_execute(pc,VFFT_FORWARD,w,NULL,w,NULL);} tc[r]=(bn()-t0)/L;
        t0=bn(); for(int i=0;i<L;i++) vfft_execute(pd,VFFT_FORWARD,z,NULL,z,NULL); td[r]=(bn()-t0)/L;
    }
    qsort(ta,9,8,dc);qsort(tb,9,8,dc);qsort(tc,9,8,dc);qsort(td,9,8,dc);
    printf("N=%-5d split-K1(scalar)=%.2fus  z-pad8=%.2fus (%+.0f%%)  z-fusedK1=%.2fus  K8/8=%.2fus (ceiling)\n",
        N,ta[4],tb[4],100*(tb[4]-ta[4])/ta[4],tc[4],td[4]/8);
    vfft_destroy(pa);vfft_destroy(pb);vfft_destroy(pc);vfft_destroy(pd);
    free(z);free(w);free(ar);free(ai);
}
int main(void){ run(256); run(1024); run(4096); return 0; }
