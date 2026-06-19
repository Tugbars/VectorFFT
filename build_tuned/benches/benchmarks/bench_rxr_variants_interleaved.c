/* bench_rxr_variants_interleaved.c — N=R*R two-stage OOP engine, variant race
 * in ONE binary, round-interleaved so VM noise hits everyone equally:
 *   eB  = base flat      (first-cut codelets, 11-arg)
 *   eL  = base log3
 *   eO  = +fuse+store-fused flat (symbols renamed rR_n1_o / rR_t1p_o)
 *   eS  = spec flat      (_spec symbols, 7-arg baked-stride ABI)
 *   eSL = spec log3      (_log3_spec, the stacked-wins candidate)
 *   FFTW guru interleaved PATIENT (plan printed)
 * Compile with -DRADIX=7|13. Correctness gate on every variant.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"

#ifndef RADIX
#define RADIX 7
#endif
#define N (RADIX*RADIX)
#define V 8
#define RV ((size_t)RADIX*V)

#define P3(a,b,c) a##b##c
#define NM(R) P3(radix,R,_n1_oop_fwd_avx512_UG_UG)
#define TM(R) P3(radix,R,_t1p_oop_fwd_avx512_UG_UG)
#define LM(R) P3(radix,R,_t1p_oop_fwd_avx512_UG_UG_log3)
#define NS(R) P3(radix,R,_n1_oop_fwd_avx512_UG_UG_spec)
#define TS(R) P3(radix,R,_t1p_oop_fwd_avx512_UG_UG_spec)
#define LS(R) P3(radix,R,_t1p_oop_fwd_avx512_UG_UG_log3_spec)
#define NO(R) P3(r,R,_n1_o)
#define TO(R) P3(r,R,_t1p_o)

extern void NM(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void TM(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void LM(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void NO(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void TO(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void NS(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t);
extern void TS(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t);
extern void LS(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t);

static double Qr[(RADIX-1)*RADIX],Qi[(RADIX-1)*RADIX];
static void initw(void){
  for(int l2=1;l2<RADIX;l2++)for(int k2=0;k2<RADIX;k2++){
    double a=-2.0*M_PI*(double)(l2*k2)/(double)N;
    Qr[(l2-1)*RADIX+k2]=cos(a);Qi[(l2-1)*RADIX+k2]=sin(a);}
}
#define LOOP(BODY) for(size_t b=0;b<K/V;b++){size_t f=b*(size_t)N*V; BODY }
static void eB(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( NM(RADIX)(ir+f,ii+f,o+f,oi+f,0,0,RV,1,(size_t)V,RADIX,RV);
        TM(RADIX)(o+f,oi+f,o+f,oi+f,Qr,Qi,RV,1,RV,1,RV); )
}
static void eL(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( NM(RADIX)(ir+f,ii+f,o+f,oi+f,0,0,RV,1,(size_t)V,RADIX,RV);
        LM(RADIX)(o+f,oi+f,o+f,oi+f,Qr,Qi,RV,1,RV,1,RV); )
}
static void eO(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( NO(RADIX)(ir+f,ii+f,o+f,oi+f,0,0,RV,1,(size_t)V,RADIX,RV);
        TO(RADIX)(o+f,oi+f,o+f,oi+f,Qr,Qi,RV,1,RV,1,RV); )
}
static void eS(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( NS(RADIX)(ir+f,ii+f,o+f,oi+f,0,0,RV);
        TS(RADIX)(o+f,oi+f,o+f,oi+f,Qr,Qi,RV); )
}
static void eSL(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( NS(RADIX)(ir+f,ii+f,o+f,oi+f,0,0,RV);
        LS(RADIX)(o+f,oi+f,o+f,oi+f,Qr,Qi,RV); )
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
      double dr=o[bk*(size_t)N*V+(size_t)k*V+l]-fo[k][0];
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
  size_t K=argc>1?(size_t)atol(argv[1]):1024; if(K%V)return 1;
  size_t TOT=(size_t)N*K;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*o=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  double *xr=malloc(TOT*8),*xi=malloc(TOT*8);
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);
  fftw_iodim64 d1={N,1,1},h1={(ptrdiff_t)K,N,N};
  fftw_plan pg=fftw_plan_guru64_dft(1,&d1,1,&h1,gi,go,FFTW_FORWARD,FFTW_PATIENT);
  printf("== RADIX=%d N=%d K=%zu ==\nplan: ",RADIX,N,K);fftw_print_plan(pg);printf("\n");

  srand(13);
  for(size_t t=0;t<K;t++)for(int e=0;e<N;e++){
    double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;
    size_t off=(t/V)*(size_t)N*V+(size_t)e*V+(t%V);
    xr[t*(size_t)N+e]=vr;xi[t*(size_t)N+e]=vi;ir[off]=vr;ii[off]=vi;
    gi[t*(size_t)N+e][0]=vr;gi[t*(size_t)N+e][1]=vi;
  }
  double rB=verify(eB,ir,ii,o,oi,xr,xi,K),rL=verify(eL,ir,ii,o,oi,xr,xi,K);
  double rO=verify(eO,ir,ii,o,oi,xr,xi,K),rS=verify(eS,ir,ii,o,oi,xr,xi,K);
  double rSL=verify(eSL,ir,ii,o,oi,xr,xi,K);
  printf("verify: base %.1e %s | log3 %.1e %s | opt %.1e %s | spec %.1e %s | specL3 %.1e %s\n",
    rB,rB<1e-9?"OK":"BAD",rL,rL<1e-9?"OK":"BAD",rO,rO<1e-9?"OK":"BAD",
    rS,rS<1e-9?"OK":"BAD",rSL,rSL<1e-9?"OK":"BAD");
  if(rB>=1e-9||rL>=1e-9||rO>=1e-9||rS>=1e-9||rSL>=1e-9){printf("ABORT\n");return 2;}

  enum{ROUNDS=40};
  unsigned long long mB=~0ULL,mL=~0ULL,mO=~0ULL,mS=~0ULL,mSL=~0ULL,mG=~0ULL,c;
  for(int w=0;w<4;w++){eB(ir,ii,o,oi,K);eL(ir,ii,o,oi,K);eO(ir,ii,o,oi,K);eS(ir,ii,o,oi,K);eSL(ir,ii,o,oi,K);fftw_execute(pg);}
  for(int r=0;r<ROUNDS;r++){
    c=__rdtsc();eB(ir,ii,o,oi,K);mB=mn2(mB,__rdtsc()-c);
    c=__rdtsc();eL(ir,ii,o,oi,K);mL=mn2(mL,__rdtsc()-c);
    c=__rdtsc();eO(ir,ii,o,oi,K);mO=mn2(mO,__rdtsc()-c);
    c=__rdtsc();eS(ir,ii,o,oi,K);mS=mn2(mS,__rdtsc()-c);
    c=__rdtsc();eSL(ir,ii,o,oi,K);mSL=mn2(mSL,__rdtsc()-c);
    c=__rdtsc();fftw_execute(pg);mG=mn2(mG,__rdtsc()-c);
  }
  printf("min cycles: base %llu | log3 %llu | opt %llu | spec %llu | specL3 %llu | fftw %llu\n",mB,mL,mO,mS,mSL,mG);
  printf("vs fftw: base %.3fx log3 %.3fx opt %.3fx spec %.3fx specL3 %.3fx\n",
    (double)mG/mB,(double)mG/mL,(double)mG/mO,(double)mG/mS,(double)mG/mSL);
  printf("deltas: spec/base %.3f  specL3/base %.3f  specL3/spec %.3f  log3/base %.3f\n",
    (double)mS/mB,(double)mSL/mB,(double)mSL/mS,(double)mL/mB);
  return 0;
}
