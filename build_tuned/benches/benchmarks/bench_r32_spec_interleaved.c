/* bench_r32_spec_interleaved.c — the interleaved confirm of spec-vs-generic
 * for the 32x32 N=1024 one-call engine. Four engine variants + FFTW guru in
 * ONE binary, round-interleaved:
 *   gF = generic n1 + t1p flat        gL = generic n1 + t1p log3
 *   sF = spec n1 + spec t1p flat      sL = spec n1 + spec t1p log3
 * Spec ABI: strides baked (256,1,8,32 / 256,1,256,1), 7 args, count=256.
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
#define CNT ((size_t)R*V)

extern void radix32_n1_oop_fwd_avx512_UG_UG (const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix32_t1p_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix32_t1p_oop_fwd_avx512_UG_UG_log3(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix32_n1_oop_fwd_avx512_UG_UG_spec (const double*,const double*,double*,double*,const double*,const double*,size_t);
extern void radix32_t1p_oop_fwd_avx512_UG_UG_spec(const double*,const double*,double*,double*,const double*,const double*,size_t);
extern void radix32_t1p_oop_fwd_avx512_UG_UG_log3_spec(const double*,const double*,double*,double*,const double*,const double*,size_t);

static double Qr[31*32],Qi[31*32];
static void initw(void){
  for(int l2=1;l2<32;l2++)for(int k2=0;k2<32;k2++){
    double a=-2.0*M_PI*(double)(l2*k2)/1024.0;
    Qr[(l2-1)*32+k2]=cos(a);Qi[(l2-1)*32+k2]=sin(a);}
}
#define LOOP(BODY) for(size_t b=0;b<K/V;b++){size_t f=b*(size_t)N*V; BODY }
static void gF(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( radix32_n1_oop_fwd_avx512_UG_UG(ir+f,ii+f,o+f,oi+f,0,0,CNT,1,(size_t)V,R,CNT);
        radix32_t1p_oop_fwd_avx512_UG_UG(o+f,oi+f,o+f,oi+f,Qr,Qi,CNT,1,CNT,1,CNT); )
}
static void gL(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( radix32_n1_oop_fwd_avx512_UG_UG(ir+f,ii+f,o+f,oi+f,0,0,CNT,1,(size_t)V,R,CNT);
        radix32_t1p_oop_fwd_avx512_UG_UG_log3(o+f,oi+f,o+f,oi+f,Qr,Qi,CNT,1,CNT,1,CNT); )
}
static void sF(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( radix32_n1_oop_fwd_avx512_UG_UG_spec(ir+f,ii+f,o+f,oi+f,0,0,CNT);
        radix32_t1p_oop_fwd_avx512_UG_UG_spec(o+f,oi+f,o+f,oi+f,Qr,Qi,CNT); )
}
static void sL(const double*ir,const double*ii,double*o,double*oi,size_t K){
  LOOP( radix32_n1_oop_fwd_avx512_UG_UG_spec(ir+f,ii+f,o+f,oi+f,0,0,CNT);
        radix32_t1p_oop_fwd_avx512_UG_UG_log3_spec(o+f,oi+f,o+f,oi+f,Qr,Qi,CNT); )
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
  size_t K=argc>1?(size_t)atol(argv[1]):512; if(K%V)return 1;
  size_t TOT=(size_t)N*K;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*o=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  double *xr=malloc(TOT*8),*xi=malloc(TOT*8);
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);
  fftw_iodim64 d1={N,1,1},h1={(ptrdiff_t)K,N,N};
  fftw_plan pg=fftw_plan_guru64_dft(1,&d1,1,&h1,gi,go,FFTW_FORWARD,FFTW_PATIENT);
  printf("== N=1024 32x32 spec confirm, K=%zu ==\nplan: ",K);fftw_print_plan(pg);printf("\n");

  srand(13);
  for(size_t t=0;t<K;t++)for(int e=0;e<N;e++){
    double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;
    size_t off=(t/V)*(size_t)N*V+(size_t)e*V+(t%V);
    xr[t*(size_t)N+e]=vr;xi[t*(size_t)N+e]=vi;ir[off]=vr;ii[off]=vi;
    gi[t*(size_t)N+e][0]=vr;gi[t*(size_t)N+e][1]=vi;
  }
  double r1=verify(gF,ir,ii,o,oi,xr,xi,K),r2=verify(gL,ir,ii,o,oi,xr,xi,K);
  double r3=verify(sF,ir,ii,o,oi,xr,xi,K),r4=verify(sL,ir,ii,o,oi,xr,xi,K);
  printf("verify: gF %.1e %s | gL %.1e %s | sF %.1e %s | sL %.1e %s\n",
    r1,r1<1e-9?"OK":"BAD",r2,r2<1e-9?"OK":"BAD",r3,r3<1e-9?"OK":"BAD",r4,r4<1e-9?"OK":"BAD");
  if(r1>=1e-9||r2>=1e-9||r3>=1e-9||r4>=1e-9){printf("ABORT\n");return 2;}

  enum{ROUNDS=30};
  unsigned long long mgF=~0ULL,mgL=~0ULL,msF=~0ULL,msL=~0ULL,mG=~0ULL,c;
  for(int w=0;w<3;w++){gF(ir,ii,o,oi,K);gL(ir,ii,o,oi,K);sF(ir,ii,o,oi,K);sL(ir,ii,o,oi,K);fftw_execute(pg);}
  for(int r=0;r<ROUNDS;r++){
    c=__rdtsc();gF(ir,ii,o,oi,K);mgF=mn2(mgF,__rdtsc()-c);
    c=__rdtsc();gL(ir,ii,o,oi,K);mgL=mn2(mgL,__rdtsc()-c);
    c=__rdtsc();sF(ir,ii,o,oi,K);msF=mn2(msF,__rdtsc()-c);
    c=__rdtsc();sL(ir,ii,o,oi,K);msL=mn2(msL,__rdtsc()-c);
    c=__rdtsc();fftw_execute(pg);mG=mn2(mG,__rdtsc()-c);
  }
  printf("min cycles: genF %llu | genL3 %llu | specF %llu | specL3 %llu | fftw %llu\n",mgF,mgL,msF,msL,mG);
  printf("vs fftw: genF %.3fx genL3 %.3fx specF %.3fx specL3 %.3fx | specF/genF %.3f specL3/genL3 %.3f\n",
    (double)mG/mgF,(double)mG/mgL,(double)mG/msF,(double)mG/msL,
    (double)msF/mgF,(double)msL/mgL);
  return 0;
}
