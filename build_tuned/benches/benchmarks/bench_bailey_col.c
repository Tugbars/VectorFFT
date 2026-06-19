/* bench_bailey_col.c — gate + race for the COLUMN-LAYOUT Bailey two-stage,
 * the layout-unified variant for the proto-core OOP plan kind.
 *
 * Layout: element e of transform t at [e*K + t] (proto/stride-executor
 * convention, MKL split-style). N = R1 x R2, x[n1 + R1*n2].
 *
 *   s1: per n1 (R1 long-count calls): n1_oop(R2)
 *       in : base src + n1*K,  L = R1*K, G = 1, count = K
 *       out: base dst + n1*R2*K, OL = K, OG = 1
 *       -> Y[n1,k2] at element (k2 + R2*n1)
 *   s2: one call, t1p(R1) in-place on dst:
 *       L = R2*K, G = 1, count = R2*K
 *       twiddle rows g = k2*(K/8) + k/8: Q[(l2-1)*(R2*K/8) + g] = W_N^(l2*k2)
 *       (K-replicated table, same model as production grp_tw)
 *       -> X[k2 + R2*k1] NATURAL order.
 *
 * Gate vs FFTW per sampled transform; race vs FFTW guru PATIENT.
 * -DR1=<t1p radix> -DR2=<leaf radix>.
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
#define CAT5(a,b,c,d,e) a##b##c##d##e
#define P5(a,b,c,d,e) CAT5(a,b,c,d,e)
#define NLEAF P5(radix,R2,_n1_oop_fwd_,avx512,_UG_UG)
#define TTWID P5(radix,R1,_t1p_oop_fwd_,avx512,_UG_UG_log3)
extern void NLEAF(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void TTWID(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);

static double *Qr,*Qi;
static void initw(size_t K){
  size_t rows=(size_t)R2*(K/8);
  Qr=malloc((size_t)(R1-1)*rows*8);Qi=malloc((size_t)(R1-1)*rows*8);
  for(int l2=1;l2<R1;l2++)for(int k2=0;k2<R2;k2++){
    double a=-2.0*M_PI*(double)((long)l2*k2)/(double)N;
    double cr=cos(a),ci=sin(a);
    for(size_t kb=0;kb<K/8;kb++){
      Qr[(size_t)(l2-1)*rows+(size_t)k2*(K/8)+kb]=cr;
      Qi[(size_t)(l2-1)*rows+(size_t)k2*(K/8)+kb]=ci;
    }
  }
}

static void eCOL(const double*ir,const double*ii,double*o,double*oi,size_t K){
  for(int n1=0;n1<R1;n1++)
    NLEAF(ir+(size_t)n1*K,ii+(size_t)n1*K,
          o+(size_t)n1*R2*K,oi+(size_t)n1*R2*K,
          0,0,(size_t)R1*K,1,K,1,K);
  TTWID(o,oi,o,oi,Qr,Qi,(size_t)R2*K,1,(size_t)R2*K,1,(size_t)R2*K);
}

static unsigned long long mn2(unsigned long long a,unsigned long long b){return a<b?a:b;}

int main(int argc,char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  size_t K=argc>1?(size_t)atol(argv[1]):128; if(K%8)return 1;
  initw(K);
  size_t TOT=(size_t)N*K;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*o=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  double *xr=malloc(TOT*8),*xi=malloc(TOT*8);
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);
  fftw_iodim64 d1={N,1,1},h1={(ptrdiff_t)K,N,N};
  fftw_plan pg=fftw_plan_guru64_dft(1,&d1,1,&h1,gi,go,FFTW_FORWARD,FFTW_PATIENT);

  srand(13);
  for(size_t t=0;t<K;t++)for(int e=0;e<N;e++){
    double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;
    xr[t*(size_t)N+e]=vr;xi[t*(size_t)N+e]=vi;
    ir[(size_t)e*K+t]=vr;ii[(size_t)e*K+t]=vi;
    gi[t*(size_t)N+e][0]=vr;gi[t*(size_t)N+e][1]=vi;
  }
  eCOL(ir,ii,o,oi,K);
  fftw_complex *fi=fftw_malloc(sizeof(fftw_complex)*N),*fo=fftw_malloc(sizeof(fftw_complex)*N);
  fftw_plan pr=fftw_plan_dft_1d(N,fi,fo,FFTW_FORWARD,FFTW_ESTIMATE);
  double mr=0;
  for(size_t t=0;t<K;t+=(K>3?K/3:1)){
    for(int e=0;e<N;e++){fi[e][0]=xr[t*(size_t)N+e];fi[e][1]=xi[t*(size_t)N+e];}
    fftw_execute(pr);
    double me=0,mm=0;
    for(int k=0;k<N;k++){
      double dr=o[(size_t)k*K+t]-fo[k][0],di=oi[(size_t)k*K+t]-fo[k][1];
      double e2=sqrt(dr*dr+di*di),m=hypot(fo[k][0],fo[k][1]);
      if(e2>me)me=e2; if(m>mm)mm=m;
    }
    if(mm>0&&me/mm>mr)mr=me/mm;
  }
  if(mr>=1e-9){printf("%2dx%-3d N=%-5d K=%-5zu GATE FAIL %.1e\n",R1,R2,N,K,mr);return 2;}

  enum{ROUNDS=30};
  unsigned long long mc=~0ULL,mg=~0ULL,c;
  for(int w=0;w<3;w++){eCOL(ir,ii,o,oi,K);fftw_execute(pg);}
  for(int r=0;r<ROUNDS;r++){
    c=__rdtsc();eCOL(ir,ii,o,oi,K);mc=mn2(mc,__rdtsc()-c);
    c=__rdtsc();fftw_execute(pg);mg=mn2(mg,__rdtsc()-c);
  }
  printf("%2dx%-3d N=%-5d K=%-5zu gate %.0e OK | col-bailey %llu cyc | speed vs fftw %.3f\n",
    R1,R2,N,K,mr,mc,(double)mg/mc);
  return 0;
}
