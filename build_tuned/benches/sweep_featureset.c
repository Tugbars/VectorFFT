#include "src/core/vfft.c"
#include <math.h>
static int NS[5][4] = {{0},{60,0,0,0},{24,20,0,0},{12,10,8,0},{8,6,6,4}};
static size_t prod(int d){ size_t p=1; for(int i=0;i<d;i++) p*=NS[d][i]; return p; }
static const char *on(int o){ return o==VFFT_ORDER_NATURAL?"nat":"def"; }

static int cell_c2c(int d, size_t K, int order, int zlay, int T){
    size_t P=prod(d), sz=P*K;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_C2C; cf.rigor=VFFT_MEASURE; cf.dims=d;
    for(int i=0;i<d;i++) cf.n[i]=NS[d][i];
    cf.howmany=K; cf.order=order; cf.placement=VFFT_INPLACE;
    cf.layout = zlay ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
    stride_set_num_threads(1);
    vfft_plan p=vfft_create(&cf);
    if(!p){ printf("d=%d c2c K=%zu %s %-5s : REJECT\n",d,K,on(order),zlay?"z":"split"); return 2; }
    double *a=aligned_alloc(64,(2*sz*8+64)&~63UL), *b=aligned_alloc(64,(2*sz*8+64)&~63UL);
    double *c=aligned_alloc(64,(2*sz*8+64)&~63UL);
    srand(17+d); for(size_t i=0;i<2*sz;i++) a[i]=2.0*rand()/RAND_MAX-1;
    stride_set_num_threads(T);
    double rt=0;
    if(zlay){
        memcpy(b,a,2*sz*8);
        vfft_execute(p,VFFT_FORWARD,b,NULL,b,NULL);
        vfft_execute(p,VFFT_BACKWARD,b,NULL,b,NULL);
        for(size_t i=0;i<2*sz;i++){double dd=fabs(b[i]/P-a[i]); if(dd>rt)rt=dd;}
    } else {
        for(size_t i=0;i<sz;i++){ b[i]=a[2*i]; c[i]=a[2*i+1]; }
        vfft_execute(p,VFFT_FORWARD,b,c,b,c);
        vfft_execute(p,VFFT_BACKWARD,b,c,b,c);
        for(size_t i=0;i<sz;i++){
            double d1=fabs(b[i]/P-a[2*i]), d2=fabs(c[i]/P-a[2*i+1]);
            if(d1>rt)rt=d1; if(d2>rt)rt=d2; }
    }
    stride_set_num_threads(1);
    int ok=rt<=1e-11;
    printf("d=%d c2c K=%zu %s %-5s : %s rt=%.1e%s\n",d,K,on(order),zlay?"z":"split",
        ok?"OK    ":"**FAIL**",rt,T>1?" [T4]":"");
    vfft_destroy(p); free(a);free(b);free(c);
    return ok?0:1;
}
static int cell_real(int d, size_t K, int order, int zlay, int T){
    size_t P=prod(d), sz=P*K;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.rigor=VFFT_MEASURE; cf.dims=d;
    for(int i=0;i<d;i++) cf.n[i]=NS[d][i];
    cf.howmany=K; cf.order=order; cf.placement=VFFT_OUTOFPLACE;
    cf.layout = zlay ? VFFT_LAYOUT_INTERLEAVED : VFFT_LAYOUT_SPLIT;
    stride_set_num_threads(1);
    cf.transform=VFFT_R2C;  vfft_plan pf=vfft_create(&cf);
    cf.transform=VFFT_C2R;  vfft_plan pb=vfft_create(&cf);
    if(!pf||!pb){
        printf("d=%d r2c/c2r K=%zu %s %-5s : REJECT (r2c=%s c2r=%s)\n",d,K,on(order),
            zlay?"z":"split", pf?"ok":"NULL", pb?"ok":"NULL");
        if(pf)vfft_destroy(pf); if(pb)vfft_destroy(pb);
        return 2;
    }
    size_t spec=2*sz+2*(P/NS[d][d-1])*K+64;
    double *a=aligned_alloc(64,(sz*8+64)&~63UL), *b=aligned_alloc(64,(spec*8+64)&~63UL);
    double *e=aligned_alloc(64,(spec*8+64)&~63UL), *c=aligned_alloc(64,(sz*8+64)&~63UL);
    srand(29+d); for(size_t i=0;i<sz;i++) a[i]=2.0*rand()/RAND_MAX-1;
    stride_set_num_threads(T);
    if(zlay){
        vfft_execute(pf,VFFT_FORWARD,a,NULL,b,NULL);
        vfft_execute(pb,VFFT_FORWARD,b,NULL,c,NULL);
    } else {
        vfft_execute(pf,VFFT_FORWARD,a,NULL,b,e);
        vfft_execute(pb,VFFT_FORWARD,b,e,c,NULL);
    }
    stride_set_num_threads(1);
    double rt=0; for(size_t i=0;i<sz;i++){double dd=fabs(c[i]/P-a[i]); if(dd>rt)rt=dd;}
    int ok=rt<=1e-11;
    printf("d=%d r2c/c2r K=%zu %s %-5s : %s rt=%.1e%s\n",d,K,on(order),zlay?"z":"split",
        ok?"OK    ":"**FAIL**",rt,T>1?" [T4]":"");
    vfft_destroy(pf); vfft_destroy(pb); free(a);free(b);free(e);free(c);
    return ok?0:1;
}
int main(void){
    int fails=0;
    printf("== primary matrix (T=1) ==\n");
    for(int d=1; d<=4; d++)
      for(int kind=0; kind<2; kind++)
        for(int ki=0; ki<2; ki++)
          for(int o=0; o<2; o++)
            for(int z=0; z<2; z++){
                size_t K=ki?5:1;
                int ord=o?VFFT_ORDER_NATURAL:VFFT_ORDER_DEFAULT;
                int r = kind? cell_real(d,K,ord,z,1) : cell_c2c(d,K,ord,z,1);
                if(r==1) fails++;
            }
    printf("== MT spot pass (T=4, def, K=1) ==\n");
    for(int d=1; d<=4; d++)
      for(int kind=0; kind<2; kind++)
        for(int z=0; z<2; z++){
            int r = kind? cell_real(d,1,VFFT_ORDER_DEFAULT,z,4)
                        : cell_c2c(d,1,VFFT_ORDER_DEFAULT,z,4);
            if(r==1) fails++;
        }
    printf(fails?"SWEEP: %d **FAIL** cells\n":"SWEEP: no wrong-number cells\n",fails);
    return fails?1:0;
}
