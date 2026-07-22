#include "src/core/vfft.c"
#include <math.h>
static double bn(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e6+t.tv_nsec*1e-3;}
static int dc(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
static void race(int N, size_t B){
    stride_plan_t *p = vfft_proto_auto_plan_dispatch(N, B, _registry(), NULL);
    if(!p){ printf("N=%d B=%zu plan NULL\n",N,B); return; }
    vfft_proto_exec_fn jf = vfft_proto_plan_jit_fwd(p);
    size_t sz=(size_t)N*B;
    double *r=aligned_alloc(64,sz*8), *m=aligned_alloc(64,sz*8);
    srand(3); for(size_t i=0;i<sz;i++){r[i]=2.0*rand()/RAND_MAX-1;m[i]=2.0*rand()/RAND_MAX-1;}
    size_t rems[4]={1, B/4?B/4:1, B/2, B-1};
    printf("N=%d B=%zu jit=%s:\n",N,B,jf?"Y":"n");
    for(int t=0;t<4;t++){
        size_t tb=rems[t]; if(tb==0||tb>=B) continue;
        int L=(int)(300000.0/( (double)N*B*0.02 ))+8;
        double ta[9],tf[9],t0;
        for(int q=0;q<9;q++){
            t0=bn(); for(int i=0;i<L;i++){ if(jf) jf(p,r,m,tb,p->K,0); else vfft_proto_execute_fwd(p,r,m,tb);} ta[q]=(bn()-t0)/L;
            t0=bn(); for(int i=0;i<L;i++){ if(jf) jf(p,r,m,B,p->K,0); else vfft_proto_execute_fwd(p,r,m,B);} tf[q]=(bn()-t0)/L;
        }
        qsort(ta,9,8,dc); qsort(tf,9,8,dc);
        printf("  this_B=%-3zu hybrid=%.2fus fullB=%.2fus (fullB %+.1f%%)\n",
            tb,ta[4],tf[4],100*(tf[4]-ta[4])/ta[4]);
    }
    free(r);free(m);
}
int main(void){ race(64,8); race(64,32); race(100,8); race(128,16); return 0; }
