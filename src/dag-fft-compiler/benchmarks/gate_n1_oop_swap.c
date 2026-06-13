/* gate_n1_oop_swap.c — correctness gate per radix, no timing:
 *   1. fwd direct leaf (N=RADIX) vs FFTW FORWARD
 *   2. bwd via pointer swap: IDFT(re,im) = swap(DFT(im,re)), same codelet,
 *      same twiddles, vs FFTW BACKWARD (both unnormalized)
 *   3. (-DWITH_T1P) two-stage N=RADIX^2 engine, fwd and swapped-bwd
 * PASS threshold 1e-9 relative. Lane-blocked split layout, V=8.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "fftw3.h"

#ifndef RADIX
#define RADIX 8
#endif
#define V 8
#ifndef ISA
#define ISA avx512
#endif
#define CAT5(a,b,c,d,e) a##b##c##d##e
#define P5(a,b,c,d,e) CAT5(a,b,c,d,e)
#define NM(R) P5(radix,R,_n1_oop_fwd_,ISA,_UG_UG)
extern void NM(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
#ifdef WITH_T1P
#ifdef T1P_LOG3
#define TM(R) P5(radix,R,_t1p_oop_fwd_,ISA,_UG_UG_log3)
#else
#define TM(R) P5(radix,R,_t1p_oop_fwd_,ISA,_UG_UG)
#endif
extern void TM(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
#endif
#ifdef SPEC
/* stride-specialized engine variants: strides baked (in_leg_stride = R*V),
 * 7-arg (in,in,out,out,tw,tw,me); valid ONLY at the engine geometry */
#define NMS(R) P5(radix,R,_n1_oop_fwd_,ISA,_UG_UG_spec)
#ifdef SPEC_PLAIN
#define TMS(R) P5(radix,R,_t1p_oop_fwd_,ISA,_UG_UG_spec)
#else
#define TMS(R) P5(radix,R,_t1p_oop_fwd_,ISA,_UG_UG_log3_spec)
#endif
extern void NMS(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t);
extern void TMS(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t);
#endif

/* max relerr of lane-blocked split output vs fftw_complex reference */
static double cmp(const double*o,const double*oi,fftw_complex*fo,int n,size_t bk,size_t l){
  double me=0,mm=0;
  for(int k=0;k<n;k++){
    double dr=o [bk*(size_t)n*V+(size_t)k*V+l]-fo[k][0];
    double di=oi[bk*(size_t)n*V+(size_t)k*V+l]-fo[k][1];
    double e=sqrt(dr*dr+di*di),m=hypot(fo[k][0],fo[k][1]);
    if(e>me)me=e; if(m>mm)mm=m;
  }
  return mm>0?me/mm:0;
}

int main(void){
  const int N1=RADIX; const size_t K=64, T1=(size_t)N1*K;
  double *ir=aligned_alloc(64,T1*8),*ii=aligned_alloc(64,T1*8);
  double *o=aligned_alloc(64,T1*8),*oi=aligned_alloc(64,T1*8);
  double *xr=malloc(T1*8),*xi=malloc(T1*8);
  srand(7+RADIX);
  for(size_t t=0;t<K;t++)for(int e=0;e<N1;e++){
    double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;
    size_t off=(t/V)*(size_t)N1*V+(size_t)e*V+(t%V);
    xr[t*(size_t)N1+e]=vr;xi[t*(size_t)N1+e]=vi;ir[off]=vr;ii[off]=vi;
  }
  fftw_complex *fi=fftw_malloc(sizeof(fftw_complex)*N1),*fo=fftw_malloc(sizeof(fftw_complex)*N1);
  fftw_plan pf=fftw_plan_dft_1d(N1,fi,fo,FFTW_FORWARD,FFTW_ESTIMATE);
  fftw_plan pb=fftw_plan_dft_1d(N1,fi,fo,FFTW_BACKWARD,FFTW_ESTIMATE);

  /* 1. fwd leaf */
  for(size_t b=0;b<K/V;b++){size_t f=b*(size_t)N1*V;
    NM(RADIX)(ir+f,ii+f,o+f,oi+f,0,0,(size_t)V,1,(size_t)V,1,(size_t)V);}
  double mf=0;
  for(size_t t=0;t<K;t+=K/4){
    for(int e=0;e<N1;e++){fi[e][0]=xr[t*(size_t)N1+e];fi[e][1]=xi[t*(size_t)N1+e];}
    fftw_execute(pf);
    double r=cmp(o,oi,fo,N1,t/V,t%V); if(r>mf)mf=r;
  }

  /* 2. bwd via swap: inputs (im,re), outputs (im,re) */
  for(size_t b=0;b<K/V;b++){size_t f=b*(size_t)N1*V;
    NM(RADIX)(ii+f,ir+f,oi+f,o+f,0,0,(size_t)V,1,(size_t)V,1,(size_t)V);}
  double mb=0;
  for(size_t t=0;t<K;t+=K/4){
    for(int e=0;e<N1;e++){fi[e][0]=xr[t*(size_t)N1+e];fi[e][1]=xi[t*(size_t)N1+e];}
    fftw_execute(pb);
    double r=cmp(o,oi,fo,N1,t/V,t%V); if(r>mb)mb=r;
  }
  printf("R=%-3d leaf fwd %.1e %s | leaf bwd-swap %.1e %s",
    RADIX,mf,mf<1e-9?"PASS":"FAIL",mb,mb<1e-9?"PASS":"FAIL");

#ifdef WITH_T1P
  {
    const int N2=RADIX*RADIX; const size_t T2=(size_t)N2*K, RV=(size_t)RADIX*V;
    /* twiddle rows = count/GROUPW; avx512 groups are 8-wide (one row per
     * k2), avx2 4-wide (each k2 duplicated V/GROUPW times) */
#ifndef GROUPW
#define GROUPW 8
#endif
    enum { SUB = V / GROUPW };
    static double Qr[(RADIX-1)*RADIX*SUB],Qi[(RADIX-1)*RADIX*SUB];
    for(int l2=1;l2<RADIX;l2++)for(int k2=0;k2<RADIX;k2++){
      double a=-2.0*M_PI*(double)(l2*k2)/(double)N2;
      for(int u=0;u<SUB;u++){
        Qr[((l2-1)*RADIX+k2)*SUB+u]=cos(a);
        Qi[((l2-1)*RADIX+k2)*SUB+u]=sin(a);}}
    double *jr=aligned_alloc(64,T2*8),*ji=aligned_alloc(64,T2*8);
    double *p=aligned_alloc(64,T2*8),*pi=aligned_alloc(64,T2*8);
    double *yr=malloc(T2*8),*yi=malloc(T2*8);
    for(size_t t=0;t<K;t++)for(int e=0;e<N2;e++){
      double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;
      size_t off=(t/V)*(size_t)N2*V+(size_t)e*V+(t%V);
      yr[t*(size_t)N2+e]=vr;yi[t*(size_t)N2+e]=vi;jr[off]=vr;ji[off]=vi;
    }
    fftw_complex *gi2=fftw_malloc(sizeof(fftw_complex)*N2),*go2=fftw_malloc(sizeof(fftw_complex)*N2);
    fftw_plan pf2=fftw_plan_dft_1d(N2,gi2,go2,FFTW_FORWARD,FFTW_ESTIMATE);
    fftw_plan pb2=fftw_plan_dft_1d(N2,gi2,go2,FFTW_BACKWARD,FFTW_ESTIMATE);
    /* engine fwd */
    for(size_t b=0;b<K/V;b++){size_t f=b*(size_t)N2*V;
#ifdef SPEC
      NMS(RADIX)(jr+f,ji+f,p+f,pi+f,0,0,RV);
      TMS(RADIX)(p+f,pi+f,p+f,pi+f,Qr,Qi,RV);
#else
      NM(RADIX)(jr+f,ji+f,p+f,pi+f,0,0,RV,1,(size_t)V,RADIX,RV);
      TM(RADIX)(p+f,pi+f,p+f,pi+f,Qr,Qi,RV,1,RV,1,RV);
#endif
    }
    double ef=0;
    for(size_t t=0;t<K;t+=K/4){
      for(int e=0;e<N2;e++){gi2[e][0]=yr[t*(size_t)N2+e];gi2[e][1]=yi[t*(size_t)N2+e];}
      fftw_execute(pf2);
      double r=cmp(p,pi,go2,N2,t/V,t%V); if(r>ef)ef=r;
    }
    /* engine bwd via swap: same codelets, same Q table, swapped pointer pairs */
    for(size_t b=0;b<K/V;b++){size_t f=b*(size_t)N2*V;
#ifdef SPEC
      NMS(RADIX)(ji+f,jr+f,pi+f,p+f,0,0,RV);
      TMS(RADIX)(pi+f,p+f,pi+f,p+f,Qr,Qi,RV);
#else
      NM(RADIX)(ji+f,jr+f,pi+f,p+f,0,0,RV,1,(size_t)V,RADIX,RV);
      TM(RADIX)(pi+f,p+f,pi+f,p+f,Qr,Qi,RV,1,RV,1,RV);
#endif
    }
    double eb=0;
    for(size_t t=0;t<K;t+=K/4){
      for(int e=0;e<N2;e++){gi2[e][0]=yr[t*(size_t)N2+e];gi2[e][1]=yi[t*(size_t)N2+e];}
      fftw_execute(pb2);
      double r=cmp(p,pi,go2,N2,t/V,t%V); if(r>eb)eb=r;
    }
    printf(" | engine fwd %.1e %s | engine bwd-swap %.1e %s",
      ef,ef<1e-9?"PASS":"FAIL",eb,eb<1e-9?"PASS":"FAIL");
  }
#endif
  printf("\n");
  return 0;
}
