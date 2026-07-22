#include "src/core/vfft.c"
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
int main(void){
    enum { N=1000 }; size_t K=4, sz=(size_t)N*K;
    int factors[3]={10,10,10};
    stride_plan_t *p = vfft_proto_plan_create_ex(N, K, factors, NULL, 3, 1, _registry());
    if(!p){ printf("plan NULL\n"); return 1; }
    vfft_proto_exec_range_fn rfn = vfft_proto_plan_jit_bwd_range(p);
    printf("jit bwd RANGE: %s\n", rfn?"RESOLVED":"NULL");
    double *z=aligned_alloc(64,2*sz*8), *br=aligned_alloc(64,sz*8), *bi=aligned_alloc(64,sz*8);
    double *cr=aligned_alloc(64,sz*8), *ci=aligned_alloc(64,sz*8);
    double *jr=aligned_alloc(64,sz*8), *ji=aligned_alloc(64,sz*8);
    srand(11); for(size_t i=0;i<2*sz;i++) z[i]=2.0*rand()/RAND_MAX-1;
    int rc1=vfft_proto_execute_bwd_ilin_core(p, z, cr, ci, K);
    int rc2=rfn?vfft_proto_execute_bwd_ilin_jit2(p, z, jr, ji, K, rfn):-1;
    printf("pipelines: core rc=%d jit rc=%d\n", rc1, rc2);
    if(!rc1 && !rc2){
        size_t d=0; for(size_t i=0;i<sz;i++) if(jr[i]!=cr[i]||ji[i]!=ci[i]) d++;
        printf("BIT ilin jit-vs-core: %zu diffs\n", d);
        int L=400; double tc[9],tj[9],t0;
        for(int t=0;t<9;t++){
            t0=bnow(); for(int i=0;i<L;i++) vfft_proto_execute_bwd_ilin_core(p,z,cr,ci,K);
            tc[t]=(bnow()-t0)/L;
            t0=bnow(); for(int i=0;i<L;i++) vfft_proto_execute_bwd_ilin_jit2(p,z,jr,ji,K,rfn);
            tj[t]=(bnow()-t0)/L;
        }
        qsort(tc,9,8,dcmp); qsort(tj,9,8,dcmp);
        printf("(1000,4) ilin bwd DIF: core=%.2fus jit=%.2fus (jit %+.1f%%)\n",
            tc[4],tj[4],100*(tj[4]-tc[4])/tc[4]);
    }
    return 0; }
