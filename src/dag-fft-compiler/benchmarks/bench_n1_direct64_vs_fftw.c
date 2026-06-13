/* bench_n1_direct64_vs_fftw.c — the structurally fair N=64 race:
 * our single-call radix64 n1_oop leaf (no twiddle stage) vs FFTW's direct
 * n2fv_64 leaf, both OOP, both natural order, guru interface, PATIENT.
 * Same lane-blocked split layout and method as bench_t1p_rxr_vs_fftw.c.
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

static void eng(const double*ir,const double*ii,double*o,double*oi,size_t K){
  for (size_t b=0;b<K/V;b++){
    const double*sr=ir+b*(size_t)N*V, *si=ii+b*(size_t)N*V;
    double *dr=o+b*(size_t)N*V, *di=oi+b*(size_t)N*V;
    /* one direct 64-point DFT per lane: element e at e*V+l, natural in/out */
    radix64_n1_oop_fwd_avx512_UG_UG(sr,si,dr,di,0,0,(size_t)V,1,(size_t)V,1,(size_t)V);
  }
}
static unsigned long long mn(unsigned long long*a,int c){unsigned long long b=~0ULL;for(int i=0;i<c;i++)if(a[i]<b)b=a[i];return b;}

int main(int argc,char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  size_t K=argc>1?(size_t)atol(argv[1]):2048; if(K%V){fprintf(stderr,"K%%8\n");return 1;}
  size_t TOT=(size_t)N*K;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*o=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  double *xr=malloc(TOT*8),*xi=malloc(TOT*8);
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);

  fftw_iodim64 d1={N,1,1}, h1={(ptrdiff_t)K,N,N};
  fftw_plan pg=fftw_plan_guru64_dft(1,&d1,1,&h1,gi,go,FFTW_FORWARD,FFTW_PATIENT);
  printf("== direct-leaf race N=64 K=%zu ==\n-- FFTW guru plan --\n",K);
  fftw_print_plan(pg); printf("\n");

  srand(13);
  for(size_t t=0;t<K;t++)for(int e=0;e<N;e++){
    double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;
    size_t off=(t/V)*(size_t)N*V+(size_t)e*V+(t%V);
    xr[t*(size_t)N+e]=vr; xi[t*(size_t)N+e]=vi; ir[off]=vr; ii[off]=vi;
    gi[t*(size_t)N+e][0]=vr; gi[t*(size_t)N+e][1]=vi;
  }

  /* correctness gate vs scalar reference */
  eng(ir,ii,o,oi,K);
  fftw_complex *fi=fftw_malloc(sizeof(fftw_complex)*N),*fo=fftw_malloc(sizeof(fftw_complex)*N);
  fftw_plan pr=fftw_plan_dft_1d(N,fi,fo,FFTW_FORWARD,FFTW_ESTIMATE);
  double mr=0;
  for(size_t t=0;t<K;t+=(K>3?K/3:1)){
    size_t bk=t/V,l=t%V;
    for(int e=0;e<N;e++){fi[e][0]=xr[t*(size_t)N+e];fi[e][1]=xi[t*(size_t)N+e];}
    fftw_execute(pr);
    double me=0,mm=0;
    for(int k=0;k<N;k++){
      double dr=o[bk*(size_t)N*V+(size_t)k*V+l]-fo[k][0];
      double di=oi[bk*(size_t)N*V+(size_t)k*V+l]-fo[k][1];
      double e2=sqrt(dr*dr+di*di),m=hypot(fo[k][0],fo[k][1]);
      if(e2>me)me=e2; if(m>mm)mm=m;
    }
    if(mm>0&&me/mm>mr)mr=me/mm;
  }
  printf("verify: relerr %.2e %s\n",mr,mr<1e-9?"OK":"BAD");
  if(mr>=1e-9){printf("ABORT\n");return 2;}

  unsigned long long ae[64],ag[64];
  for(int w=0;w<6;w++){eng(ir,ii,o,oi,K);fftw_execute(pg);}
  for(int r=0;r<60;r++){unsigned long long c=__rdtsc();eng(ir,ii,o,oi,K);ae[r%64]=__rdtsc()-c;}
  for(int r=0;r<60;r++){unsigned long long c=__rdtsc();fftw_execute(pg);ag[r%64]=__rdtsc()-c;}
  unsigned long long E=mn(ae,60),G=mn(ag,60);
  printf("cycles: n1_oop_64 %llu | fftw_guru %llu | ratio guru/n1 %.3fx\n",E,G,(double)G/E);
  return 0;
}
