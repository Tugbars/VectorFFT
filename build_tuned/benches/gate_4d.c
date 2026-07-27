#include "src/core/vfft.c"
#include <math.h>
int main(void){
    int ok=1;
    /* 4D r2c vs naive real-DFT (fndr output order is defined) */
    int n[4]={6,4,4,4}; size_t P=6*4*4*4;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=4; for(int i=0;i<4;i++) cf.n[i]=n[i]; cf.howmany=1;
    vfft_plan pf=vfft_create(&cf);
    cf.transform=VFFT_C2R; vfft_plan pb=vfft_create(&cf);
    if(!pf||!pb){ printf("4D real create FAIL (r2c=%s c2r=%s)\n",pf?"ok":"NULL",pb?"ok":"NULL"); return 1; }
    double *a=aligned_alloc(64,P*8), *br=aligned_alloc(64,2*P*8), *bi=aligned_alloc(64,2*P*8), *c=aligned_alloc(64,P*8);
    srand(7); for(size_t i=0;i<P;i++) a[i]=2.0*rand()/RAND_MAX-1;
    vfft_execute(pf,VFFT_FORWARD,a,NULL,br,bi);
    /* naive at a few random natural bins (fndr spectrum order: [k0][k1][k2][f], f<hp1) */
    size_t hp1=n[3]/2+1; double worst=0;
    for(int t=0;t<20;t++){
        int k0=rand()%n[0],k1=rand()%n[1],k2=rand()%n[2],f=rand()%(int)hp1;
        double sr=0,si=0;
        for(int x0=0;x0<n[0];x0++)for(int x1=0;x1<n[1];x1++)for(int x2=0;x2<n[2];x2++)for(int x3=0;x3<n[3];x3++){
            double ph=-2.0*M_PI*((double)k0*x0/n[0]+(double)k1*x1/n[1]+(double)k2*x2/n[2]+(double)f*x3/n[3]);
            double v=a[(((size_t)x0*n[1]+x1)*n[2]+x2)*n[3]+x3];
            sr+=v*cos(ph); si+=v*sin(ph);
        }
        size_t idx=(((size_t)k0*n[1]+k1)*n[2]+k2)*hp1+f;
        double d1=fabs(br[idx]-sr), d2=fabs(bi[idx]-si);
        if(d1>worst)worst=d1; if(d2>worst)worst=d2;
    }
    printf("4D r2c vs naive (20 bins): %.2e %s\n",worst,worst<1e-9?"PASS":"**FAIL**");
    ok &= worst<1e-9;
    vfft_execute(pb,VFFT_FORWARD,br,bi,c,NULL);
    double rt=0; for(size_t i=0;i<P;i++){double d=fabs(c[i]/P-a[i]); if(d>rt)rt=d;}
    printf("4D r2c->c2r roundtrip: %.2e %s\n",rt,rt<1e-11?"PASS":"**FAIL**");
    ok &= rt<1e-11;
    /* 4D c2c: roundtrip + Parseval */
    cf.transform=VFFT_C2C; cf.placement=VFFT_INPLACE;
    cf.layout=VFFT_LAYOUT_INTERLEAVED;   /* 4D c2c cell runs the z contract */
    vfft_plan pc=vfft_create(&cf);
    if(!pc){ printf("4D c2c create FAIL\n"); return 1; }
    double *z=aligned_alloc(64,2*P*8), *z0=aligned_alloc(64,2*P*8);
    for(size_t i=0;i<2*P;i++){ z[i]=2.0*rand()/RAND_MAX-1; z0[i]=z[i]; }
    double e0=0; for(size_t i=0;i<2*P;i++) e0+=z[i]*z[i];
    vfft_execute(pc,VFFT_FORWARD,z,NULL,z,NULL);
    double e1=0; for(size_t i=0;i<2*P;i++) e1+=z[i]*z[i];
    double pv=fabs(e1/P-e0)/e0;
    vfft_execute(pc,VFFT_BACKWARD,z,NULL,z,NULL);
    double rc=0; for(size_t i=0;i<2*P;i++){double d=fabs(z[i]/P-z0[i]); if(d>rc)rc=d;}
    printf("4D c2c Parseval rel=%.2e roundtrip=%.2e %s\n",pv,rc,(pv<1e-12&&rc<1e-11)?"PASS":"**FAIL**");
    ok &= (pv<1e-12&&rc<1e-11);
    printf(ok?"4D GATE: ALL PASS\n":"4D GATE: **FAILURES**\n");
    return ok?0:1;
}
