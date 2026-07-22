#include "src/core/vfft.c"
#include <math.h>
static int cellnd(int rank, const int *N, int il){
    size_t tot=1; for(int i=0;i<rank;i++) tot*=(size_t)N[i];
    int NL=N[rank-1]; size_t hp1=(size_t)NL/2+1, R=tot/(size_t)NL;
    double *x=aligned_alloc(64,tot*8), *y=aligned_alloc(64,tot*8);
    double *re=aligned_alloc(64,R*hp1*8), *im=aligned_alloc(64,R*hp1*8);
    double *z = il ? aligned_alloc(64,R*hp1*16) : NULL;
    srand(41+rank); for(size_t i=0;i<tot;i++) x[i]=2.0*rand()/RAND_MAX-1;
    vfft_config_t cf; memset(&cf,0,sizeof cf);
    cf.transform=VFFT_R2C; cf.placement=VFFT_OUTOFPLACE; cf.rigor=VFFT_MEASURE;
    cf.dims=rank; for(int i=0;i<rank;i++) cf.n[i]=N[i]; cf.howmany=1;
    vfft_plan pf=vfft_create(&cf); cf.transform=VFFT_C2R; vfft_plan pb=vfft_create(&cf);
    if(!pf||!pb){ printf("  [ND"); for(int i=0;i<rank;i++)printf(" %d",N[i]); printf("] plan FAIL\n"); return 0; }
    struct vfft_plan_s *hf=(struct vfft_plan_s*)pf;
    stride_fftnd_r2c_data_t *df=(stride_fftnd_r2c_data_t*)hf->tplan->override_data;
    memcpy(y,x,tot*8);
    vfft_execute(pf,VFFT_FORWARD,y,NULL,re,im);
    if(il){
        vfft_execute(pf,VFFT_FORWARD,y,NULL,z,NULL);
        size_t bad=0;
        for(size_t i=0;i<R*hp1;i++) if(z[2*i]!=re[i]||z[2*i+1]!=im[i]) bad++;
        printf("  [ND z] fwd z-vs-split BIT: %s (%zu)\n", bad?"**FAIL**":"PASS", bad);
        if(bad) { free(z); return 0; }
    }
    int ok=1;
    if(tot<=4096 && !il){  /* naive ND real DFT at a few bins */
        for(int chk=0;chk<40;chk++){
            size_t bin=(size_t)rand()%(R*hp1);
            size_t row=bin/hp1, f=bin%hp1;
            size_t idx[FFTND_MAX_RANK]; size_t rr=row;
            for(int m=rank-2;m>=0;m--){ idx[m]=rr%(size_t)N[m]; rr/=(size_t)N[m]; }
            double sr=0,si=0;
            size_t c[FFTND_MAX_RANK]; memset(c,0,sizeof c);
            for(size_t t=0;t<tot;t++){
                size_t tt=t; double ph=0;
                for(int m=rank-1;m>=0;m--){ c[m]=tt%(size_t)N[m]; tt/=(size_t)N[m]; }
                for(int m=0;m<rank-1;m++) ph += (double)idx[m]*(double)c[m]/(double)N[m];
                ph += (double)f*(double)c[rank-1]/(double)N[rank-1];
                double a=-2.0*M_PI*ph;
                sr+=x[t]*cos(a); si+=x[t]*sin(a);
            }
            double d1=fabs(re[row*hp1+f]-sr), d2=fabs(im[row*hp1+f]-si);
            if(d1>1e-9||d2>1e-9){ ok=0; printf("naive mismatch %.2e %.2e\n",d1,d2); break; }
        }
        printf("  [ND"); for(int i=0;i<rank;i++)printf(" %d",N[i]);
        printf("] fwd vs naive: %s (snd_fwd=%s)\n", ok?"PASS":"**FAIL**", df->snd_fwd?"MONO":"tiled");
    }
    if(il) vfft_execute(pb,VFFT_BACKWARD,z,NULL,y,NULL);
    else   vfft_execute(pb,VFFT_BACKWARD,re,im,y,NULL);
    double mx=0; for(size_t i=0;i<tot;i++){ double dd=fabs(y[i]/(double)tot-x[i]); if(dd>mx)mx=dd; }
    printf("  [ND"); for(int i=0;i<rank;i++)printf(" %d",N[i]);
    printf("]%s roundtrip=%.3e %s (fwd=%s bwd=%s)\n", il?" il":"", mx,
        mx<1e-11?"PASS":"**FAIL**",
        df->snd_fwd?"MONO":"tiled",
        ((stride_fftnd_r2c_data_t*)((struct vfft_plan_s*)pb)->tplan->override_data)->snd_bwd?"MONO":"tiled");
    if(mx>=1e-11) ok=0;
    vfft_destroy(pf); vfft_destroy(pb);
    free(x);free(y);free(re);free(im); if(z)free(z);
    return ok;
}
int main(void){
    int ok=1;
    { int N[]={4,6,8};      ok&=cellnd(3,N,0); }  /* naive-checkable */
    { int N[]={8,24,32};    ok&=cellnd(3,N,0); }  /* R=192, covered */
    { int N[]={6,10,64};    ok&=cellnd(3,N,0); }  /* R=60: pairs%8!=0 -> avx512 fallback */
    { int N[]={8,9,10};     ok&=cellnd(3,N,0); }  /* NL=10 uncovered -> tiled */
    { int N[]={3,9,32};     ok&=cellnd(3,N,0); }  /* Q2: R=27 ragged+odd */
    { int N[]={8,24,32};    ok&=cellnd(3,N,1); }  /* il_out contract */
    { int N[]={16,16,256};  ok&=cellnd(3,N,0); }  /* meaty: R=256 x N=256 */
    printf(ok?"FNDR Q1 GATE: ALL PASS\n":"FNDR Q1 GATE: **FAILURES**\n");
    return ok?0:1;
}
