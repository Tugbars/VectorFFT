#include "src/core/vfft.c"
#include <math.h>
static int cellmt(int N, size_t K, int T){
    size_t sz=(size_t)N*K;
    double *z=aligned_alloc(64,((2*sz*8)+63)&~63UL), *a=aligned_alloc(64,((2*sz*8)+63)&~63UL);
    double *b=aligned_alloc(64,((2*sz*8)+63)&~63UL);
    srand(31+N); for(size_t i=0;i<2*sz;i++) z[i]=2.0*rand()/RAND_MAX-1;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_C2C; cf.placement=VFFT_INPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N; cf.howmany=K;
    stride_set_num_threads(1);
    vfft_plan p=vfft_create(&cf);
    memcpy(a,z,2*sz*8); vfft_execute(p,VFFT_FORWARD,a,NULL,a,NULL);
    stride_set_num_threads(T);
    memcpy(b,z,2*sz*8); vfft_execute(p,VFFT_FORWARD,b,NULL,b,NULL);
    size_t df=0; for(size_t i=0;i<2*sz;i++) if(a[i]!=b[i]) df++;
    vfft_execute(p,VFFT_BACKWARD,b,NULL,b,NULL);
    stride_set_num_threads(1);
    vfft_execute(p,VFFT_BACKWARD,a,NULL,a,NULL);
    size_t db=0; for(size_t i=0;i<2*sz;i++) if(a[i]!=b[i]) db++;
    double rt=0; for(size_t i=0;i<2*sz;i++){double d=fabs(a[i]/N-z[i]); if(d>rt)rt=d;}
    printf("  [il2il MT] (%d,%-3zu T=%d) fwd BIT=%zu bwd BIT=%zu rt=%.1e %s\n",
        N,K,T,df,db,rt,(df==0&&db==0&&rt<1e-12)?"PASS":"**FAIL**");
    vfft_destroy(p); free(z);free(a);free(b);
    return df==0&&db==0&&rt<1e-12;
}
static int cellnatmt(int N, size_t K, int T){
    size_t sz=(size_t)N*K;
    double *z=aligned_alloc(64,((2*sz*8)+63)&~63UL), *a=aligned_alloc(64,((2*sz*8)+63)&~63UL);
    double *b=aligned_alloc(64,((2*sz*8)+63)&~63UL);
    srand(41); for(size_t i=0;i<2*sz;i++) z[i]=2.0*rand()/RAND_MAX-1;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_C2C; cf.placement=VFFT_INPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=1; cf.n[0]=N; cf.howmany=K; cf.order=VFFT_ORDER_NATURAL;
    vfft_plan p=vfft_create(&cf);
    if(!p){ printf("  [nat MT] create FAIL\n"); return 0; }
    stride_set_num_threads(1);
    memcpy(a,z,2*sz*8); vfft_execute(p,VFFT_FORWARD,a,NULL,a,NULL);
    stride_set_num_threads(T);
    memcpy(b,z,2*sz*8); vfft_execute(p,VFFT_FORWARD,b,NULL,b,NULL);
    size_t df=0; for(size_t i=0;i<2*sz;i++) if(a[i]!=b[i]) df++;
    stride_set_num_threads(1);
    printf("  [nat MT ] (%d,%-3zu T=%d) fallback fwd BIT=%zu %s\n",
        N,K,T,df,df?"**FAIL**":"PASS");
    vfft_destroy(p); free(z);free(a);free(b);
    return df==0;
}
int main(void){
    int ok=1;
    stride_set_num_threads(4);
    printf("pool size after set(4): %d\n", _stride_pool_size);
    ok &= cellmt(200, 12, 4);
    ok &= cellmt(256, 64, 8);
    ok &= cellmt(100, 67, 4);
    ok &= cellmt(504, 40, 8);
    ok &= cellmt(100, 7, 4);    /* K<8 single-slab boundary */
    ok &= cellnatmt(100, 96, 4);
    stride_set_num_threads(1);
    printf(ok?"IL MT GATE: ALL PASS\n":"IL MT GATE: **FAILURES**\n");
    return ok?0:1;
}
