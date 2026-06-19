/* bench_strategy_iso_rxr.c — strategy isolation generalized to N = R1 x R2.
 * Stage 1 = n1_oop_R2 (DFT over n2), stage 2 = t1p_log3_R1 (twiddled DFT over
 * n1). Twiddle table row stride = R2 (codelet indexes tw[(l2-1)*(me/8)+b/8]).
 * Four dataflows, identical codelets, round-robin, verify gate per variant:
 *   A fused Bailey | F ping-pong Bailey | B Stockham-2stage | E six-step.
 * Compile: -DR1=... -DR2=...  Run: ./a K
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"

#ifndef R1
#define R1 32
#endif
#ifndef R2
#define R2 32
#endif
#define N (R1*R2)
#define V 8
#define CNT1 ((size_t)R1*V)
#define CNT2 ((size_t)R2*V)

#define CAT5(a,b,c,d,e) a##b##c##d##e
#define P5(a,b,c,d,e) CAT5(a,b,c,d,e)
#define NM(R) P5(radix,R,_n1_oop_fwd_,avx512,_UG_UG)
#define TL(R) P5(radix,R,_t1p_oop_fwd_,avx512,_UG_UG_log3)
extern void NM(R2)(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void TL(R1)(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);

static double Qr[(R1-1)*R2],Qi[(R1-1)*R2];
static void initw(void){
  for(int l2=1;l2<R1;l2++)for(int k2=0;k2<R2;k2++){
    double a=-2.0*M_PI*(double)((size_t)l2*k2)/(double)N;
    Qr[(l2-1)*R2+k2]=cos(a);Qi[(l2-1)*R2+k2]=sin(a);}
}
static double *wr,*wi;

/* natural (V*n1 + R1*V*k2) -> A-layout (R2*V*n1 + V*k2), V doubles per move */
static inline void xpose(const double*s,double*d){
  for(int n1=0;n1<R1;n1++)for(int k2=0;k2<R2;k2++)
    _mm512_storeu_pd(d+(size_t)R2*V*n1+(size_t)V*k2,
                     _mm512_loadu_pd(s+(size_t)V*n1+(size_t)R1*V*k2));
}

#define LOOP(BODY) for(size_t b=0;b<K/V;b++){size_t f=b*(size_t)N*V; BODY }
static void eA(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( NM(R2)(ir+f,ii+f,o+f,oi+f,0,0,CNT1,1,(size_t)V,R2,CNT1);
        TL(R1)(o+f,oi+f,o+f,oi+f,Qr,Qi,CNT2,1,CNT2,1,CNT2); )
}
static void eF(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( NM(R2)(ir+f,ii+f,wr+f,wi+f,0,0,CNT1,1,(size_t)V,R2,CNT1);
        TL(R1)(wr+f,wi+f,o+f,oi+f,Qr,Qi,CNT2,1,CNT2,1,CNT2); )
}
static void eB(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( NM(R2)(ir+f,ii+f,wr+f,wi+f,0,0,CNT1,1,CNT1,1,CNT1);
        TL(R1)(wr+f,wi+f,o+f,oi+f,Qr,Qi,(size_t)V,R1,CNT2,1,CNT2); )
}
static void eE(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( NM(R2)(ir+f,ii+f,wr+f,wi+f,0,0,CNT1,1,CNT1,1,CNT1);
        xpose(wr+f,o+f); xpose(wi+f,oi+f);
        TL(R1)(o+f,oi+f,o+f,oi+f,Qr,Qi,CNT2,1,CNT2,1,CNT2); )
}
typedef void engfn(const double*,const double*,double*,double*,size_t);
static unsigned long long mn2(unsigned long long a,unsigned long long b){return a<b?a:b;}

static double verify(engfn*eng,const double*ir,const double*ii,double*o,double*oi,
                     const double*xr,const double*xi,size_t K){
  eng(ir,ii,o,oi,K);
  fftw_complex *fi=fftw_malloc(sizeof(fftw_complex)*N),*fo=fftw_malloc(sizeof(fftw_complex)*N);
  fftw_plan p=fftw_plan_dft_1d(N,fi,fo,FFTW_FORWARD,FFTW_ESTIMATE);
  double mr=0;
  for(size_t t=0;t<K;t+=(K>3?K/3:1)){
    size_t bk=t/V,l=t%V;
    for(int e=0;e<N;e++){fi[e][0]=xr[t*(size_t)N+e];fi[e][1]=xi[t*(size_t)N+e];}
    fftw_execute(p);
    double me=0,mm=0;
    for(int k=0;k<N;k++){
      double dr=o [bk*(size_t)N*V+(size_t)k*V+l]-fo[k][0];
      double di=oi[bk*(size_t)N*V+(size_t)k*V+l]-fo[k][1];
      double e2=sqrt(dr*dr+di*di),m=hypot(fo[k][0],fo[k][1]);
      if(e2>me)me=e2; if(m>mm)mm=m;
    }
    if(mm>0&&me/mm>mr)mr=me/mm;
  }
  fftw_destroy_plan(p);fftw_free(fi);fftw_free(fo);return mr;
}

int main(int argc,char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  initw();
  size_t K=argc>1?(size_t)atol(argv[1]):512; if(K%V)return 1;
  size_t TOT=(size_t)N*K;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*o=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  wr=aligned_alloc(64,TOT*8); wi=aligned_alloc(64,TOT*8);
  double *xr=malloc(TOT*8),*xi=malloc(TOT*8);
  srand(13);
  for(size_t t=0;t<K;t++)for(int e=0;e<N;e++){
    double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;
    size_t off=(t/V)*(size_t)N*V+(size_t)e*V+(t%V);
    xr[t*(size_t)N+e]=vr;xi[t*(size_t)N+e]=vi;ir[off]=vr;ii[off]=vi;
  }
  double rA=verify(eA,ir,ii,o,oi,xr,xi,K),rF=verify(eF,ir,ii,o,oi,xr,xi,K);
  double rB=verify(eB,ir,ii,o,oi,xr,xi,K),rE=verify(eE,ir,ii,o,oi,xr,xi,K);
  if(rA>=1e-9||rF>=1e-9||rB>=1e-9||rE>=1e-9){
    printf("%2dx%-3d N=%-5d K=%-5zu GATE FAIL A=%.1e F=%.1e B=%.1e E=%.1e\n",R1,R2,N,K,rA,rF,rB,rE);
    return 2;
  }
  enum{ROUNDS=30};
  unsigned long long mA=~0ULL,mF=~0ULL,mB=~0ULL,mE=~0ULL,c;
  for(int w=0;w<3;w++){eA(ir,ii,o,oi,K);eF(ir,ii,o,oi,K);eB(ir,ii,o,oi,K);eE(ir,ii,o,oi,K);}
  for(int r=0;r<ROUNDS;r++){
    c=__rdtsc();eA(ir,ii,o,oi,K);mA=mn2(mA,__rdtsc()-c);
    c=__rdtsc();eF(ir,ii,o,oi,K);mF=mn2(mF,__rdtsc()-c);
    c=__rdtsc();eB(ir,ii,o,oi,K);mB=mn2(mB,__rdtsc()-c);
    c=__rdtsc();eE(ir,ii,o,oi,K);mE=mn2(mE,__rdtsc()-c);
  }
  printf("%2dx%-3d N=%-5d K=%-5zu | A %9llu | F/A %.3f  B/A %.3f  E/A %.3f  verif %.0e\n",
    R1,R2,N,K,mA,(double)mF/mA,(double)mB/mA,(double)mE/mA,rA);
  return 0;
}
