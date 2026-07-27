/* Public-API gate: INTERLEAVED z contract on vfft_execute (layout axis).
 * Post layout-axis migration: the z contract is COMMITTED at create
 * (config.layout = VFFT_LAYOUT_INTERLEAVED), so this gate runs TWO plans per
 * cell — a SPLIT reference plan and an INTERLEAVED plan — instead of the old
 * one-plan NULL-pointer inference (removed). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
static int ulp_ok(double a,double b){ if(a==b)return 1;
    double m=fabs(a)>fabs(b)?fabs(a):fabs(b); if(m<1)m=1;
    return fabs(a-b)<=8.0*2.220446049250313e-16*m; }
static int cmp(const double*a,const double*b,size_t n,const char*t){
    size_t bit=0,ok=0,bad=0;
    for(size_t i=0;i<n;i++){ if(a[i]==b[i])bit++; else if(ulp_ok(a[i],b[i]))ok++; else bad++; }
    printf("    %-22s bit=%zu ulp=%zu BAD=%zu %s\n",t,bit,ok,bad,bad?"**FAIL**":(ok?"ULP":"BIT"));
    return bad==0; }
static int run_cell(int N,size_t K,int order,int nth,const char*tag){
    size_t NK=(size_t)N*K; int all=1;
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=1; c.n[0]=N; c.howmany=K; c.order=order; c.nthreads=nth;
    vfft_plan p=vfft_create(&c);              /* SPLIT reference plan */
    c.layout=VFFT_LAYOUT_INTERLEAVED;
    vfft_plan pz=vfft_create(&c);             /* committed z plan */
    if(!p||!pz){ printf("  [%s] create FAIL\n",tag); return 0; }
    double *z=aligned_alloc(64,2*NK*8),*z2=aligned_alloc(64,2*NK*8);
    double *sr=aligned_alloc(64,NK*8),*si=aligned_alloc(64,NK*8);
    double *zin=aligned_alloc(64,2*NK*8);
    srand(7+N);
    for(size_t i=0;i<2*NK;i++) zin[i]=2.0*rand()/RAND_MAX-1;
    printf("  [%s] N=%d K=%zu order=%d nth=%d\n",tag,N,K,order,nth);
    /* fwd: interleaved vs split reference */
    for(size_t i=0;i<NK;i++){sr[i]=zin[2*i];si[i]=zin[2*i+1];}
    vfft_execute(p,VFFT_FORWARD,sr,si,sr,si);
    memcpy(z,zin,2*NK*8);
    vfft_execute(p,VFFT_FORWARD,z,NULL,z,NULL);
    for(size_t i=0;i<NK;i++){z2[2*i]=sr[i];z2[2*i+1]=si[i];}
    all&=cmp(z,z2,2*NK,"fwd il vs split");
    /* bwd: interleaved vs split (spectrum in both layouts) */
    vfft_execute(p,VFFT_BACKWARD,sr,si,sr,si);
    vfft_execute(p,VFFT_BACKWARD,z,NULL,z,NULL);
    for(size_t i=0;i<NK;i++){z2[2*i]=sr[i];z2[2*i+1]=si[i];}
    all&=cmp(z,z2,2*NK,"bwd il vs split");
    /* roundtrip scale check vs input */
    { size_t bad=0; for(size_t i=0;i<2*NK;i++) if(!ulp_ok(z[i]/N,zin[i])) bad++;
      printf("    %-22s BAD=%zu %s\n","roundtrip/N vs in",bad,bad?"**FAIL**":"OK"); all&=!bad; }
    /* conv pattern: z fwd -> pointwise -> z bwd  vs split same */
    { double *kr=aligned_alloc(64,NK*8),*ki=aligned_alloc(64,NK*8);
      for(size_t i=0;i<NK;i++){kr[i]=0.5+0.001*(double)(i%37);ki[i]=0.25-0.002*(double)(i%23);}
      memcpy(z,zin,2*NK*8);
      vfft_execute(p,VFFT_FORWARD,z,NULL,z,NULL);
      for(size_t i=0;i<NK;i++){ double a=z[2*i],b=z[2*i+1];
          z[2*i]=a*kr[i]-b*ki[i]; z[2*i+1]=a*ki[i]+b*kr[i]; }
      vfft_execute(p,VFFT_BACKWARD,z,NULL,z,NULL);
      for(size_t i=0;i<NK;i++){sr[i]=zin[2*i];si[i]=zin[2*i+1];}
      vfft_execute(p,VFFT_FORWARD,sr,si,sr,si);
      for(size_t i=0;i<NK;i++){ double a=sr[i],b=si[i];
          sr[i]=a*kr[i]-b*ki[i]; si[i]=a*ki[i]+b*kr[i]; }
      vfft_execute(p,VFFT_BACKWARD,sr,si,sr,si);
      for(size_t i=0;i<NK;i++){z2[2*i]=sr[i];z2[2*i+1]=si[i];}
      all&=cmp(z,z2,2*NK,"conv pattern il/split");
      free(kr);free(ki); }
    vfft_destroy(p);
    free(z);free(z2);free(sr);free(si);free(zin);
    return all; }
int main(void){
    int all=1;
    all&=run_cell(64,8,VFFT_ORDER_DEFAULT,0,"DIT 64x8");
    all&=run_cell(100,8,VFFT_ORDER_DEFAULT,0,"DIF 100x8");
    all&=run_cell(101,8,VFFT_ORDER_DEFAULT,0,"prime 101 (override->fallback)");
    all&=run_cell(64,8,VFFT_ORDER_NATURAL,0,"NATURAL 64x8 (fallback)");
    all&=run_cell(64,16,VFFT_ORDER_DEFAULT,2,"MT nth=2 (fallback)");
    puts(all?"VFFT IL GATE: ALL PASS":"VFFT IL GATE: FAILURES");
    return all?0:1; }
