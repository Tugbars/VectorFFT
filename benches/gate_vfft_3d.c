/* Public-API gate: dims==3 C2C + dedicated fft3d wisdom (bank -> hit). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "vfft.h"
static double now_us(void){ struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return t.tv_sec*1e6+t.tv_nsec*1e-3; }
static int ulp_ok(double a,double b){ if(a==b)return 1;
    double m=fabs(a)>fabs(b)?fabs(a):fabs(b); if(m<1)m=1;
    return fabs(a-b)<=16.0*2.220446049250313e-16*m; }
static int check_cell(vfft_plan p,int NT,const char*tag){
    double *re=aligned_alloc(64,(size_t)NT*8),*im=aligned_alloc(64,(size_t)NT*8);
    double *r0=aligned_alloc(64,(size_t)NT*8),*i0=aligned_alloc(64,(size_t)NT*8);
    int all=1; size_t bad;
    /* delta -> |X|==1 everywhere (order-agnostic) */
    memset(re,0,(size_t)NT*8); memset(im,0,(size_t)NT*8); re[0]=1.0;
    vfft_execute(p,VFFT_FORWARD,re,im,re,im);
    bad=0; for(int i=0;i<NT;i++){ double m2=re[i]*re[i]+im[i]*im[i];
        if(!ulp_ok(m2,1.0)) bad++; }
    printf("    [%s] delta |X|^2==1   BAD=%zu %s\n",tag,bad,bad?"**FAIL**":"OK"); all&=!bad;
    /* roundtrip/NT vs random input */
    srand(11); for(int i=0;i<NT;i++){ r0[i]=re[i]=2.0*rand()/RAND_MAX-1;
                                      i0[i]=im[i]=2.0*rand()/RAND_MAX-1; }
    vfft_execute(p,VFFT_FORWARD,re,im,re,im);
    vfft_execute(p,VFFT_BACKWARD,re,im,re,im);
    bad=0; for(int i=0;i<NT;i++){ if(!ulp_ok(re[i]/NT,r0[i]))bad++;
                                  if(!ulp_ok(im[i]/NT,i0[i]))bad++; }
    printf("    [%s] roundtrip/N      BAD=%zu %s\n",tag,bad,bad?"**FAIL**":"OK"); all&=!bad;
    free(re);free(im);free(r0);free(i0);
    return all; }
int main(void){
    int all=1, N1=16,N2=20,N3=8, NT=N1*N2*N3;
    vfft_wisdom *w=vfft_wisdom_load("/tmp/wb3");
    printf("bundle load(/tmp/wb3): %s\n", w?"ok":"NULL");
    vfft_config_t c; memset(&c,0,sizeof c);
    c.transform=VFFT_C2C; c.placement=VFFT_INPLACE; c.rigor=VFFT_MEASURE;
    c.dims=3; c.n[0]=N1; c.n[1]=N2; c.n[2]=N3; c.howmany=1; c.wisdom=w;
    double t0=now_us();
    vfft_plan p1=vfft_create(&c);
    double t_miss=now_us()-t0;
    if(!p1){ puts("create#1 FAIL"); return 1; }
    printf("  create#1 (greedy+bank) %.0f us\n",t_miss);
    all&=check_cell(p1,NT,"greedy");
    vfft_destroy(p1);
    /* wisdom file written? */
    { FILE*f=fopen("/tmp/wb3/fft3d_c2c_wisdom.txt","r"); int lines=0; char b[2048];
      if(f){ while(fgets(b,sizeof b,f)) if(b[0]!='@'&&b[0]!='#'&&b[0]!='\n') lines++; fclose(f);}
      printf("  wisdom file entries: %d %s\n",lines,lines>=1?"OK":"**MISSING**"); all&=lines>=1; }
    /* fresh bundle -> HIT path */
    vfft_wisdom *w2=vfft_wisdom_load("/tmp/wb3");
    c.wisdom=w2;
    t0=now_us();
    vfft_plan p2=vfft_create(&c);
    double t_hit=now_us()-t0;
    if(!p2){ puts("create#2 FAIL"); return 1; }
    printf("  create#2 (wisdom hit)  %.0f us  (miss/hit = %.1fx)\n",t_hit,t_miss/t_hit);
    all&=check_cell(p2,NT,"wisdom-hit");
    vfft_destroy(p2);
    /* rejects */
    { vfft_config_t r=c; r.howmany=2;  all&=(vfft_create(&r)==NULL); }
    { vfft_config_t r=c; r.order=VFFT_ORDER_NATURAL; all&=(vfft_create(&r)==NULL); }
    { vfft_config_t r=c; r.transform=VFFT_DCT2; all&=(vfft_create(&r)==NULL); }
    printf("  reject contracts (K>1 / NATURAL / non-C2C): %s\n",all?"OK":"**FAIL**");
    if(w) vfft_wisdom_free(w); if(w2) vfft_wisdom_free(w2);
    puts(all?"VFFT 3D GATE: ALL PASS":"VFFT 3D GATE: FAILURES");
    return all?0:1; }
