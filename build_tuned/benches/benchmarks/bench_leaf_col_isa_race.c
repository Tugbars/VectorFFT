/* bench_leaf_col_isa_race.c — leaf ISA race in COLUMN layout (production
 * wisdom-bench layout: element e of transform t at [e*K + t]). ONE codelet
 * call per pass with count=K, matching FFTW's one-call internal batch loop,
 * instead of the lane-blocked 8-lane-crumb calls of bench_leaf_isa_race.c.
 * avx512 vs avx2 vs FFTW guru, same binary, round-robin. -DRADIX=...
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"

#ifndef RADIX
#define RADIX 8
#endif
#define CAT5(a,b,c,d,e) a##b##c##d##e
#define P5(a,b,c,d,e) CAT5(a,b,c,d,e)
#define NZ(R) P5(radix,R,_n1_oop_fwd_,avx512,_UG_UG)
#define NY(R) P5(radix,R,_n1_oop_fwd_,avx2,_UG_UG)
extern void NZ(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void NY(RADIX)(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);

/* one call: L=K (element stride), G=1 (position stride), count=K */
static void eZ(const double*ir,const double*ii,double*o,double*oi,size_t K){
  NZ(RADIX)(ir,ii,o,oi,0,0,K,1,K,1,K);
}
static void eY(const double*ir,const double*ii,double*o,double*oi,size_t K){
  NY(RADIX)(ir,ii,o,oi,0,0,K,1,K,1,K);
}
typedef void engfn(const double*,const double*,double*,double*,size_t);
static unsigned long long mn2(unsigned long long a,unsigned long long b){return a<b?a:b;}

static double verify(engfn*eng,const double*ir,const double*ii,double*o,double*oi,
                     const double*xr,const double*xi,size_t K){
  const int N1=RADIX;
  eng(ir,ii,o,oi,K);
  fftw_complex *fi=fftw_malloc(sizeof(fftw_complex)*N1),*fo=fftw_malloc(sizeof(fftw_complex)*N1);
  fftw_plan p=fftw_plan_dft_1d(N1,fi,fo,FFTW_FORWARD,FFTW_ESTIMATE);
  double mr=0;
  for(size_t t=0;t<K;t+=K/4){
    for(int e=0;e<N1;e++){fi[e][0]=xr[t*(size_t)N1+e];fi[e][1]=xi[t*(size_t)N1+e];}
    fftw_execute(p);
    double me=0,mm=0;
    for(int k=0;k<N1;k++){
      double dr=o[(size_t)k*K+t]-fo[k][0];
      double di=oi[(size_t)k*K+t]-fo[k][1];
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
  const int N1=RADIX;
  size_t K=argc>1?(size_t)atol(argv[1]):8192; if(K%8)return 1;
  size_t TOT=(size_t)N1*K;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*o=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  double *xr=malloc(TOT*8),*xi=malloc(TOT*8);
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);
  fftw_iodim64 d1={N1,1,1},h1={(ptrdiff_t)K,N1,N1};
  fftw_plan pg=fftw_plan_guru64_dft(1,&d1,1,&h1,gi,go,FFTW_FORWARD,FFTW_PATIENT);
  printf("== leaf ISA race (column, one-call) R=%d K=%zu ==\nplan: ",RADIX,K);
  fftw_print_plan(pg);printf("\n");

  srand(13);
  for(size_t t=0;t<K;t++)for(int e=0;e<N1;e++){
    double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;
    xr[t*(size_t)N1+e]=vr;xi[t*(size_t)N1+e]=vi;
    ir[(size_t)e*K+t]=vr;ii[(size_t)e*K+t]=vi;
    gi[t*(size_t)N1+e][0]=vr;gi[t*(size_t)N1+e][1]=vi;
  }
  double rZ=verify(eZ,ir,ii,o,oi,xr,xi,K),rY=verify(eY,ir,ii,o,oi,xr,xi,K);
  printf("verify: avx512 %.1e %s | avx2 %.1e %s\n",rZ,rZ<1e-9?"OK":"BAD",rY,rY<1e-9?"OK":"BAD");
  if(rZ>=1e-9||rY>=1e-9)return 2;

  enum{ROUNDS=40};
  unsigned long long mZ=~0ULL,mY=~0ULL,mG=~0ULL,c;
  for(int w=0;w<4;w++){eZ(ir,ii,o,oi,K);eY(ir,ii,o,oi,K);fftw_execute(pg);}
  for(int r=0;r<ROUNDS;r++){
    c=__rdtsc();eZ(ir,ii,o,oi,K);mZ=mn2(mZ,__rdtsc()-c);
    c=__rdtsc();eY(ir,ii,o,oi,K);mY=mn2(mY,__rdtsc()-c);
    c=__rdtsc();fftw_execute(pg);mG=mn2(mG,__rdtsc()-c);
  }
  printf("min cycles: avx512 %llu | avx2 %llu | fftw %llu\n",mZ,mY,mG);
  printf("ratios: avx2/avx512 %.3f | fftw/avx512 %.3fx | fftw/avx2 %.3fx\n",
    (double)mY/mZ,(double)mG/mZ,(double)mG/mY);
  return 0;
}
