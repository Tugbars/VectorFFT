#include "src/core/vfft.c"
int main(void){
    /* FIRST create in a fresh process = the cold path that used to
     * half-succeed with wrong values at prime N1. Must now be NULL. */
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=2; cf.n[0]=41; cf.n[1]=32; cf.howmany=1;
    vfft_plan p=vfft_create(&cf);
    printf("cold-first (41,32): %s\n", p?"**FAIL** (created)":"PASS (NULL)");
    if(p) return 1;
    /* and a healthy odd-composite must still create + roundtrip */
    cf.n[0]=27; vfft_plan pf=vfft_create(&cf);
    cf.transform=VFFT_C2R; vfft_plan pb=vfft_create(&cf);
    if(!pf||!pb){ printf("(27,32) create **FAIL**\n"); return 1; }
    enum{N1=27,N2=32,H=17};
    static double x[N1*N2],r[N1*H],m[N1*H],y[N1*N2];
    srand(5); for(int i=0;i<N1*N2;i++)x[i]=2.0*rand()/RAND_MAX-1;
    vfft_execute(pf,VFFT_FORWARD,x,NULL,r,m);
    vfft_execute(pb,VFFT_BACKWARD,r,m,y,NULL);
    double mx=0; for(int i=0;i<N1*N2;i++){double d=fabs(y[i]/((double)N1*N2)-x[i]); if(d>mx)mx=d;}
    printf("(27,32) rt=%.2e %s\n",mx,mx<1e-12?"PASS":"**FAIL**");
    return mx<1e-12?0:1; }
