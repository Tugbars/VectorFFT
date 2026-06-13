/* bench_strategy_bigN.c — ONE-CELL big multi-stage OOP test, N = 64*64*64 =
 * 262144, K=8 (one lane block), working set ~100MB >> L3.
 *
 *   A3: fully fused recursive Bailey, three one-call codelet passes, zero
 *       copies. ALL passes carry 64-stream loads at 32768-double (256KB)
 *       stride. Output is digit-swapped (k1<->k2 within k_m): affine-stride
 *       parity makes natural order unreachable in three fused passes.
 *   E6: six-step shape: blocked transpose T1, 64 contiguous inner 64x64
 *       engines, blocked transpose T3, contiguous outer t1p. Five passes,
 *       every codelet load contiguous. Natural order output.
 *
 * Index algebra (element units; lane-blocked addr = 8*e + l):
 *   x: e = n1 + 64*u1 + 4096*u2          (m = u1 + 64*u2, inner M=4096)
 *   A3 P1 n1_oop_64 over u2: in(L=32768,G=1) out(OL=8,OG=64)
 *        -> Z{k2:8, n1:512, u1:32768}
 *   A3 P2 t1p_64 over u1 (W_M, rows g2=k2+64*n1): in(L=32768,G=1)
 *        out(OL=8,OG=64) -> {q1:8, k2:512, n1:32768}     (q1 = inner t1p freq)
 *   A3 P3 t1p_64 over n1 (W_N, rows g3=q1+64*k2, k_m=k2+64*q1):
 *        in(L=32768,G=1) out in-place(OL=32768,OG=1)
 *        -> X at e' = q1 + 64*k2 + 4096*k1   (digit-swapped vs natural)
 *   E6: S{n1:32768, m:8}; inner engine per slab (validated square strides);
 *       U{n1:8, k_m:512}; outer t1p in(L=8,G=64) out(OL=32768,OG=1) natural.
 * Verify: each variant against its own permuted FFTW reference. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"
#define R 64
#define M 4096
#define NBIG 262144
#define V 8
#define CNT3 ((size_t)4096*V)   /* 32768 positions, 4096 groups */
#define CNTS ((size_t)R*V)      /* 512: inner square engine */

extern void radix64_n1_oop_fwd_avx512_UG_UG (const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix64_t1p_oop_fwd_avx512_UG_UG_log3(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
#define NL radix64_n1_oop_fwd_avx512_UG_UG
#define TL radix64_t1p_oop_fwd_avx512_UG_UG_log3

static double *Q2r,*Q2i,*Q3Ar,*Q3Ai,*Q3Er,*Q3Ei,*QMr,*QMi;
static void initw(void){
  Q2r=malloc(63*4096*8);Q2i=malloc(63*4096*8);
  Q3Ar=malloc(63*4096*8);Q3Ai=malloc(63*4096*8);
  Q3Er=malloc(63*4096*8);Q3Ei=malloc(63*4096*8);
  QMr=malloc(63*64*8);QMi=malloc(63*64*8);
  for(int l2=1;l2<64;l2++){
    for(int g=0;g<4096;g++){
      int k2=g&63;                       /* Q2 rows g2=k2+64*n1: W_M^(l2*k2) */
      double a=-2.0*M_PI*(double)((long)l2*k2)/(double)M;
      Q2r[(l2-1)*4096+g]=cos(a);Q2i[(l2-1)*4096+g]=sin(a);
      int q1=g&63,kk2=g>>6,km=kk2+64*q1; /* Q3A rows g3=q1+64*k2: W_N^(l2*km) */
      double b=-2.0*M_PI*(double)((long)l2*km)/(double)NBIG;
      Q3Ar[(l2-1)*4096+g]=cos(b);Q3Ai[(l2-1)*4096+g]=sin(b);
      double c=-2.0*M_PI*(double)((long)l2*g)/(double)NBIG; /* Q3E rows km=g */
      Q3Er[(l2-1)*4096+g]=cos(c);Q3Ei[(l2-1)*4096+g]=sin(c);
    }
    for(int k2=0;k2<64;k2++){
      double a=-2.0*M_PI*(double)((long)l2*k2)/(double)M;
      QMr[(l2-1)*64+k2]=cos(a);QMi[(l2-1)*64+k2]=sin(a);
    }
  }
}
static double *wr,*wi,*sr2,*si2;

/* A3: three fused one-call passes, src->dst, dst finishes in-place */
static void eA3(const double*ir,const double*ii,double*o,double*oi){
  NL(ir,ii,wr,wi,0,0,(size_t)32768,1,(size_t)8,(size_t)64,CNT3);
  TL(wr,wi,o,oi,Q2r,Q2i,(size_t)32768,1,(size_t)8,(size_t)64,CNT3);
  TL(o,oi,o,oi,Q3Ar,Q3Ai,(size_t)32768,1,(size_t)32768,1,CNT3);
}
/* E6: T1 gather to slabs, inner engines, T3 regroup, outer t1p */
static inline void t1cp(const double*s,double*d){ /* x{n1:8,u1:512,u2:32768} -> S{n1:32768,m=u1+64u2:8} */
  for(int u2=0;u2<64;u2++)for(int u1=0;u1<64;u1++)for(int n1=0;n1<64;n1++)
    _mm512_storeu_pd(d+(size_t)32768*n1+8*(u1+64*u2),
      _mm512_loadu_pd(s+(size_t)8*n1+512*u1+32768*u2));
}
static inline void t3cp(const double*s,double*d){ /* S{n1:32768,km:8} -> U{n1:8,km:512} */
  for(int km=0;km<4096;km++)for(int n1=0;n1<64;n1++)
    _mm512_storeu_pd(d+(size_t)8*n1+512*km,
      _mm512_loadu_pd(s+(size_t)32768*n1+8*km));
}
static void eE6(const double*ir,const double*ii,double*o,double*oi){
  t1cp(ir,wr); t1cp(ii,wi);
  for(int n1=0;n1<64;n1++){size_t f=(size_t)32768*n1;
    NL(wr+f,wi+f,sr2+f,si2+f,0,0,(size_t)512,1,(size_t)8,(size_t)64,CNTS);
    TL(sr2+f,si2+f,sr2+f,si2+f,QMr,QMi,(size_t)512,1,(size_t)512,1,CNTS);
  }
  t3cp(sr2,wr); t3cp(si2,wi);
  TL(wr,wi,o,oi,Q3Er,Q3Ei,(size_t)8,(size_t)64,(size_t)32768,1,CNT3);
}

static unsigned long long mn2(unsigned long long a,unsigned long long b){return a<b?a:b;}

int main(void){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  initw();
  size_t TOT=(size_t)NBIG*V;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8);
  double *o=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  wr=aligned_alloc(64,TOT*8);wi=aligned_alloc(64,TOT*8);
  sr2=aligned_alloc(64,TOT*8);si2=aligned_alloc(64,TOT*8);
  srand(13);
  for(size_t e=0;e<(size_t)NBIG;e++)for(int l=0;l<V;l++){
    ir[e*V+l]=(double)rand()/RAND_MAX-0.5; ii[e*V+l]=(double)rand()/RAND_MAX-0.5;
  }
  /* reference: lane 3 */
  fftw_complex *fi=fftw_malloc(sizeof(fftw_complex)*NBIG),*fo=fftw_malloc(sizeof(fftw_complex)*NBIG);
  for(size_t e=0;e<(size_t)NBIG;e++){fi[e][0]=ir[e*V+3];fi[e][1]=ii[e*V+3];}
  fftw_plan pr=fftw_plan_dft_1d(NBIG,fi,fo,FFTW_FORWARD,FFTW_ESTIMATE);
  fftw_execute(pr);
  const int LREF=3;

  eA3(ir,ii,o,oi);
  double mA=0,mmA=0;
  for(size_t k=0;k<(size_t)NBIG;k++){
    size_t k2=k&63,q1=(k>>6)&63,k1=k>>12;     /* ref k = k2 + 64*q1 + 4096*k1 */
    size_t ep=q1+64*k2+4096*k1;               /* ours at digit-swapped e' */
    double dr=o[ep*V+LREF]-fo[k][0],di=oi[ep*V+LREF]-fo[k][1];
    double e2=sqrt(dr*dr+di*di),m=hypot(fo[k][0],fo[k][1]);
    if(e2>mA)mA=e2; if(m>mmA)mmA=m;
  }
  double rA=mA/mmA;
  eE6(ir,ii,o,oi);
  double mE=0,mmE=0;
  for(size_t k=0;k<(size_t)NBIG;k++){
    double dr=o[k*V+LREF]-fo[k][0],di=oi[k*V+LREF]-fo[k][1];
    double e2=sqrt(dr*dr+di*di),m=hypot(fo[k][0],fo[k][1]);
    if(e2>mE)mE=e2; if(m>mmE)mmE=m;
  }
  double rE=mE/mmE;
  printf("N=262144 K=8 gates: A3 %.1e %s (digit-swapped order) | E6 %.1e %s (natural)\n",
    rA,rA<1e-9?"OK":"BAD",rE,rE<1e-9?"OK":"BAD");
  if(rA>=1e-9||rE>=1e-9){printf("ABORT\n");return 2;}

  enum{ROUNDS=12};
  unsigned long long a3=~0ULL,e6=~0ULL,c;
  for(int w=0;w<2;w++){eA3(ir,ii,o,oi);eE6(ir,ii,o,oi);}
  for(int r=0;r<ROUNDS;r++){
    c=__rdtsc();eA3(ir,ii,o,oi);a3=mn2(a3,__rdtsc()-c);
    c=__rdtsc();eE6(ir,ii,o,oi);e6=mn2(e6,__rdtsc()-c);
  }
  printf("min cycles: A3_fused3pass %llu | E6_sixstep %llu\n",a3,e6);
  printf("relative speed (faster=bigger, A3=1.000): A3 1.000 | E6 %.3f\n",(double)a3/e6);
  return 0;
}
