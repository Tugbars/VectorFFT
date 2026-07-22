#include "src/core/vfft.c"
static double bnow(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int dcmp(const void*a,const void*b){double x=*(const double*)a,y=*(const double*)b;return (x>y)-(x<y);}
int main(void){
    enum { N=1000 }; size_t K=4;
    int factors[3]={10,10,10};
    const vfft_proto_registry_t *reg = _registry();
    stride_plan_t *p = vfft_proto_plan_create_ex(N, K, factors, NULL, 3, 1, reg);
    if(!p){ printf("plan NULL\n"); return 1; }
    printf("plan: nf=%d stage-radices %d %d %d\n", p->num_stages,
        p->stages[0].radix, p->stages[1].radix, p->stages[2].radix);
    vfft_proto_exec_fn jb = vfft_proto_plan_jit_bwd(p);
    printf("jit bwd fn: %s\n", jb?"RESOLVED":"NULL (rule or miss)");
    size_t sz=(size_t)N*K;
    double *br=aligned_alloc(64,sz*8), *bi=aligned_alloc(64,sz*8);
    double *cr=aligned_alloc(64,sz*8), *ci=aligned_alloc(64,sz*8);
    double *jr=aligned_alloc(64,sz*8), *ji=aligned_alloc(64,sz*8);
    srand(9); for(size_t i=0;i<sz;i++){br[i]=2.0*rand()/RAND_MAX-1;bi[i]=2.0*rand()/RAND_MAX-1;}
    memcpy(cr,br,sz*8); memcpy(ci,bi,sz*8);
    vfft_proto_execute_bwd(p,cr,ci,K);
    if(jb){ memcpy(jr,br,sz*8); memcpy(ji,bi,sz*8);
        jb(p,jr,ji,K,K,0);
        size_t d=0; for(size_t i=0;i<sz;i++) if(jr[i]!=cr[i]||ji[i]!=ci[i]) d++;
        printf("BIT jit-vs-core: %zu diffs\n", d);
    }
    int L=400; double tc[9],tj[9],t0;
    for(int t=0;t<9;t++){
        t0=bnow(); for(int i=0;i<L;i++){ memcpy(cr,br,sz*8); memcpy(ci,bi,sz*8);
            vfft_proto_execute_bwd(p,cr,ci,K); }
        tc[t]=(bnow()-t0)/L;
        if(jb){ t0=bnow(); for(int i=0;i<L;i++){ memcpy(cr,br,sz*8); memcpy(ci,bi,sz*8);
            jb(p,cr,ci,K,K,0); }
            tj[t]=(bnow()-t0)/L; }
    }
    qsort(tc,9,8,dcmp); if(jb) qsort(tj,9,8,dcmp);
    if(jb) printf("(1000,4) bwd DIF: core=%.2fus jit=%.2fus (jit %+.1f%%)\n",
        tc[4],tj[4],100*(tj[4]-tc[4])/tc[4]);
    else printf("(1000,4) bwd DIF core=%.2fus (jit unresolved)\n",tc[4]);
    return 0; }
