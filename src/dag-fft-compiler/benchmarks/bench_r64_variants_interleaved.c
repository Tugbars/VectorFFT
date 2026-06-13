/* bench_r64_variants_interleaved.c — v1 generic vs v2 fuse+storefused vs
 * v3 spec-strides vs FFTW guru, ALL IN ONE BINARY, round-interleaved timing
 * so VM noise hits all contenders equally. min across rounds per contender.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"
#define N 64
#define V 8
extern void radix64_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void r64_v2(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void r64_v3(const double*,const double*,double*,double*,const double*,const double*,size_t); /* spec: strides baked 8,1,8,1 */

static void e1(const double*ir,const double*ii,double*o,double*oi,size_t K){
  for(size_t b=0;b<K/V;b++){size_t off=b*(size_t)N*V;
    radix64_n1_oop_fwd_avx512_UG_UG(ir+off,ii+off,o+off,oi+off,0,0,(size_t)V,1,(size_t)V,1,(size_t)V);}
}
static void e2(const double*ir,const double*ii,double*o,double*oi,size_t K){
  for(size_t b=0;b<K/V;b++){size_t off=b*(size_t)N*V;
    r64_v2(ir+off,ii+off,o+off,oi+off,0,0,(size_t)V,1,(size_t)V,1,(size_t)V);}
}
static void e3(const double*ir,const double*ii,double*o,double*oi,size_t K){
  for(size_t b=0;b<K/V;b++){size_t off=b*(size_t)N*V;
    r64_v3(ir+off,ii+off,o+off,oi+off,0,0,(size_t)V);}
}
typedef void engfn(const double*,const double*,double*,double*,size_t);

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
      double e2v=sqrt(dr*dr+di*di),m=hypot(fo[k][0],fo[k][1]);
      if(e2v>me)me=e2v; if(m>mm)mm=m;
    }
    if(mm>0&&me/mm>mr)mr=me/mm;
  }
  fftw_destroy_plan(p);fftw_free(fi);fftw_free(fo);return mr;
}

int main(int argc,char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  size_t K=argc>1?(size_t)atol(argv[1]):2048; if(K%V)return 1;
  size_t TOT=(size_t)N*K;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*o=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  double *xr=malloc(TOT*8),*xi=malloc(TOT*8);
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);
  fftw_iodim64 d1={N,1,1},h1={(ptrdiff_t)K,N,N};
  fftw_plan pg=fftw_plan_guru64_dft(1,&d1,1,&h1,gi,go,FFTW_FORWARD,FFTW_PATIENT);
  printf("plan: ");fftw_print_plan(pg);printf("\n");

  srand(13);
  for(size_t t=0;t<K;t++)for(int e=0;e<N;e++){
    double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;
    size_t off=(t/V)*(size_t)N*V+(size_t)e*V+(t%V);
    xr[t*(size_t)N+e]=vr;xi[t*(size_t)N+e]=vi;ir[off]=vr;ii[off]=vi;
    gi[t*(size_t)N+e][0]=vr;gi[t*(size_t)N+e][1]=vi;
  }
  double r1=verify(e1,ir,ii,o,oi,xr,xi,K),r2=verify(e2,ir,ii,o,oi,xr,xi,K),r3=verify(e3,ir,ii,o,oi,xr,xi,K);
  printf("verify: v1 %.1e %s | v2 %.1e %s | v3 %.1e %s\n",
         r1,r1<1e-9?"OK":"BAD",r2,r2<1e-9?"OK":"BAD",r3,r3<1e-9?"OK":"BAD");
  if(r1>=1e-9||r2>=1e-9||r3>=1e-9)return 2;

  enum{ROUNDS=40};
  unsigned long long m1=~0ULL,m2=~0ULL,m3=~0ULL,mg=~0ULL,c;
  for(int w=0;w<4;w++){e1(ir,ii,o,oi,K);e2(ir,ii,o,oi,K);e3(ir,ii,o,oi,K);fftw_execute(pg);}
  for(int r=0;r<ROUNDS;r++){
    c=__rdtsc(); e1(ir,ii,o,oi,K); c=__rdtsc()-c; if(c<m1)m1=c;
    c=__rdtsc(); e2(ir,ii,o,oi,K); c=__rdtsc()-c; if(c<m2)m2=c;
    c=__rdtsc(); e3(ir,ii,o,oi,K); c=__rdtsc()-c; if(c<m3)m3=c;
    c=__rdtsc(); fftw_execute(pg); c=__rdtsc()-c; if(c<mg)mg=c;
  }
  printf("min cycles: v1_generic %llu | v2_opt %llu | v3_spec %llu | fftw %llu\n",m1,m2,m3,mg);
  printf("ratios vs fftw: v1 %.3fx  v2 %.3fx  v3 %.3fx   | v2/v1 %.3f  v3/v1 %.3f\n",
         (double)mg/m1,(double)mg/m2,(double)mg/m3,(double)m2/m1,(double)m3/m1);
  return 0;
}
