/* bench_strategy_iso_1024.c — STRATEGY ISOLATION: identical radix-32 codelets
 * (generic n1_oop + t1p log3), N=1024=32x32, four dataflow wrappings:
 *   A  fused Bailey   : s1 scatter-stores src->dst,   t1p in-place on dst (current)
 *   F  ping-pong Bailey: s1 scatter-stores src->work, t1p work->dst natural
 *   B  Stockham-2stage: s1 natural src->work,         t1p absorbs transpose in loads
 *   E  six-step-like  : s1 natural src->work, explicit transpose copy work->dst,
 *                       t1p in-place on dst (3 passes)
 * Same Q table, same arithmetic, same total permutation everywhere; only the
 * absorption point and buffer topology differ. All ops remain full-width
 * contiguous vector loads/stores (lane-blocked layout: lanes ride inside the
 * zmm), so this is a pure locality + pass-count contest.
 * Round-robin one binary, FFTW guru reference, verify gate per variant.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"
#define N 1024
#define R 32
#define V 8
#define CNT ((size_t)R*V)   /* 256 positions per block */

extern void radix32_n1_oop_fwd_avx512_UG_UG (const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix32_t1p_oop_fwd_avx512_UG_UG_log3(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
#define N1   radix32_n1_oop_fwd_avx512_UG_UG
#define T1L3 radix32_t1p_oop_fwd_avx512_UG_UG_log3

static double Qr[31*32],Qi[31*32];
static void initw(void){
  for(int l2=1;l2<32;l2++)for(int k2=0;k2<32;k2++){
    double a=-2.0*M_PI*(double)(l2*k2)/1024.0;
    Qr[(l2-1)*32+k2]=cos(a);Qi[(l2-1)*32+k2]=sin(a);}
}
static double *wr,*wi;  /* work buffers (ping-pong variants) */

/* explicit transpose for E: Ynat (8*n1+256*k2+l) -> A-layout (256*n1+8*k2+l) */
static inline void xpose(const double*s,double*d){
  for(int n1=0;n1<32;n1++)for(int k2=0;k2<32;k2++)
    _mm512_storeu_pd(d+256*n1+8*k2,_mm512_loadu_pd(s+8*n1+256*k2));
}

#define LOOP(BODY) for(size_t b=0;b<K/V;b++){size_t f=b*(size_t)N*V; BODY }
static void eA(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( N1(ir+f,ii+f,o+f,oi+f,0,0,CNT,1,(size_t)V,R,CNT);
        T1L3(o+f,oi+f,o+f,oi+f,Qr,Qi,CNT,1,CNT,1,CNT); )
}
static void eF(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( N1(ir+f,ii+f,wr+f,wi+f,0,0,CNT,1,(size_t)V,R,CNT);
        T1L3(wr+f,wi+f,o+f,oi+f,Qr,Qi,CNT,1,CNT,1,CNT); )
}
static void eB(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( N1(ir+f,ii+f,wr+f,wi+f,0,0,CNT,1,CNT,1,CNT);
        T1L3(wr+f,wi+f,o+f,oi+f,Qr,Qi,(size_t)V,R,CNT,1,CNT); )
}
static void eE(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( N1(ir+f,ii+f,wr+f,wi+f,0,0,CNT,1,CNT,1,CNT);
        xpose(wr+f,o+f); xpose(wi+f,oi+f);
        T1L3(o+f,oi+f,o+f,oi+f,Qr,Qi,CNT,1,CNT,1,CNT); )
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
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);
  fftw_iodim64 d1={N,1,1},h1={(ptrdiff_t)K,N,N};
  fftw_plan pg=fftw_plan_guru64_dft(1,&d1,1,&h1,gi,go,FFTW_FORWARD,FFTW_PATIENT);
  printf("== strategy isolation N=1024 K=%zu (same codelets, four wrappings) ==\nplan: ",K);
  fftw_print_plan(pg);printf("\n");

  srand(13);
  for(size_t t=0;t<K;t++)for(int e=0;e<N;e++){
    double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;
    size_t off=(t/V)*(size_t)N*V+(size_t)e*V+(t%V);
    xr[t*(size_t)N+e]=vr;xi[t*(size_t)N+e]=vi;ir[off]=vr;ii[off]=vi;
    gi[t*(size_t)N+e][0]=vr;gi[t*(size_t)N+e][1]=vi;
  }
  double rA=verify(eA,ir,ii,o,oi,xr,xi,K),rF=verify(eF,ir,ii,o,oi,xr,xi,K);
  double rB=verify(eB,ir,ii,o,oi,xr,xi,K),rE=verify(eE,ir,ii,o,oi,xr,xi,K);
  printf("verify: A %.1e %s | F %.1e %s | B %.1e %s | E %.1e %s\n",
    rA,rA<1e-9?"OK":"BAD",rF,rF<1e-9?"OK":"BAD",rB,rB<1e-9?"OK":"BAD",rE,rE<1e-9?"OK":"BAD");
  if(rA>=1e-9||rF>=1e-9||rB>=1e-9||rE>=1e-9){printf("ABORT\n");return 2;}

  enum{ROUNDS=30};
  unsigned long long mA=~0ULL,mF=~0ULL,mB=~0ULL,mE=~0ULL,mG=~0ULL,c;
  for(int w=0;w<3;w++){eA(ir,ii,o,oi,K);eF(ir,ii,o,oi,K);eB(ir,ii,o,oi,K);eE(ir,ii,o,oi,K);fftw_execute(pg);}
  for(int r=0;r<ROUNDS;r++){
    c=__rdtsc();eA(ir,ii,o,oi,K);mA=mn2(mA,__rdtsc()-c);
    c=__rdtsc();eF(ir,ii,o,oi,K);mF=mn2(mF,__rdtsc()-c);
    c=__rdtsc();eB(ir,ii,o,oi,K);mB=mn2(mB,__rdtsc()-c);
    c=__rdtsc();eE(ir,ii,o,oi,K);mE=mn2(mE,__rdtsc()-c);
    c=__rdtsc();fftw_execute(pg);mG=mn2(mG,__rdtsc()-c);
  }
  printf("min cycles: A_fusedBailey %llu | F_pingpong %llu | B_stockham %llu | E_sixstep %llu | fftw %llu\n",
    mA,mF,mB,mE,mG);
  printf("vs A: F %.3f  B %.3f  E %.3f | fftw/A %.3fx\n",
    (double)mF/mA,(double)mB/mA,(double)mE/mA,(double)mG/mA);
  return 0;
}
