#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>
#include <pmmintrin.h>
extern void radix64_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix16_t1s_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
#define N1 64
#define N2 16
#define N 1024
#define V 8
static double twr[N1*(N2-1)],twi[N1*(N2-1)];
static double WR[N*V],WI[N*V];
#define PAD 520
static double PR[PAD*N2+64],PI[PAD*N2+64];
static void initw(void){for(int k1=0;k1<N1;k1++)for(int n2=1;n2<N2;n2++){double a=-2.0*M_PI*(double)(n2*k1)/(double)N; twr[k1*(N2-1)+(n2-1)]=cos(a); twi[k1*(N2-1)+(n2-1)]=sin(a);}}
static inline void inner_only(const double*ir,const double*ii,double*orr,double*oi,size_t K){
  for(size_t b=0;b<K/V;b++)
    radix64_n1_oop_fwd_avx512_UG_UG(ir+b*N*V,ii+b*N*V,WR,WI,0,0,(size_t)N2*V,1,(size_t)N2*V,1,(size_t)N2*V);
}
static inline void outer_only(double*orr,double*oi,size_t K){
  for(size_t b=0;b<K/V;b++) for(int k1=0;k1<N1;k1++)
    radix16_t1s_oop_fwd_avx512_UG_UG(WR+(size_t)N2*k1*V,WI+(size_t)N2*k1*V, orr+b*N*V+(size_t)k1*V, oi+b*N*V+(size_t)k1*V, twr+(size_t)k1*(N2-1),twi+(size_t)k1*(N2-1), (size_t)V,1,(size_t)N1*V,1,(size_t)V);
}
static inline void outer_padded(size_t K){  /* write transpose at non-power-of-2 stride 520 to break L1 set aliasing */
  for(size_t b=0;b<K/V;b++) for(int k1=0;k1<N1;k1++)
    radix16_t1s_oop_fwd_avx512_UG_UG(WR+(size_t)N2*k1*V,WI+(size_t)N2*k1*V, PR+(size_t)k1*V, PI+(size_t)k1*V, twr+(size_t)k1*(N2-1),twi+(size_t)k1*(N2-1), (size_t)V,1,(size_t)PAD,1,(size_t)V);
}
static inline void outer_permuted(double*orr,double*oi,size_t K){
  for(size_t b=0;b<K/V;b++) for(int k1=0;k1<N1;k1++)
    radix16_t1s_oop_fwd_avx512_UG_UG(WR+(size_t)N2*k1*V,WI+(size_t)N2*k1*V, orr+b*N*V+(size_t)N2*k1*V, oi+b*N*V+(size_t)N2*k1*V, twr+(size_t)k1*(N2-1),twi+(size_t)k1*(N2-1), (size_t)V,1,(size_t)V,1,(size_t)V);
}
static inline void full(const double*ir,const double*ii,double*orr,double*oi,size_t K){
  for(size_t b=0;b<K/V;b++){
    radix64_n1_oop_fwd_avx512_UG_UG(ir+b*N*V,ii+b*N*V,WR,WI,0,0,(size_t)N2*V,1,(size_t)N2*V,1,(size_t)N2*V);
    for(int k1=0;k1<N1;k1++)
      radix16_t1s_oop_fwd_avx512_UG_UG(WR+(size_t)N2*k1*V,WI+(size_t)N2*k1*V, orr+b*N*V+(size_t)k1*V, oi+b*N*V+(size_t)k1*V, twr+(size_t)k1*(N2-1),twi+(size_t)k1*(N2-1), (size_t)V,1,(size_t)N1*V,1,(size_t)V);
  }
}
static unsigned long long mn(unsigned long long*a,int c){unsigned long long b=~0ULL;for(int i=0;i<c;i++)if(a[i]<b)b=a[i];return b;}
int main(int argc,char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  initw(); size_t K=argc>1?atol(argv[1]):512; size_t TOT=(size_t)N*K;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*orr=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  for(size_t i=0;i<TOT;i++){ir[i]=(i%101)-50.0;ii[i]=(i%97)-48.0;}
  unsigned long long ia[64],oa[64],fa[64]; int it=200;
  for(int w=0;w<8;w++) inner_only(ir,ii,orr,oi,K);
  for(int r=0;r<it;r++){unsigned long long c0=__rdtsc(); inner_only(ir,ii,orr,oi,K); ia[r%64]=__rdtsc()-c0;}
  for(int w=0;w<8;w++) outer_only(orr,oi,K);
  for(int r=0;r<it;r++){unsigned long long c0=__rdtsc(); outer_only(orr,oi,K); oa[r%64]=__rdtsc()-c0;}
  unsigned long long pa[64],da[64];
  for(int w=0;w<8;w++) outer_permuted(orr,oi,K);
  for(int r=0;r<it;r++){unsigned long long c0=__rdtsc(); outer_permuted(orr,oi,K); pa[r%64]=__rdtsc()-c0;}
  for(int w=0;w<8;w++) outer_padded(K);
  for(int r=0;r<it;r++){unsigned long long c0=__rdtsc(); outer_padded(K); da[r%64]=__rdtsc()-c0;}
  for(int w=0;w<8;w++) full(ir,ii,orr,oi,K);
  for(int r=0;r<it;r++){unsigned long long c0=__rdtsc(); full(ir,ii,orr,oi,K); fa[r%64]=__rdtsc()-c0;}
  unsigned long long I=mn(ia,it<64?it:64),O=mn(oa,it<64?it:64),F=mn(fa,it<64?it:64);
  printf("K=%zu  inner(radix64) %llu  |  outer(64x t1s16 + transpose) %llu  |  full %llu\n",K,I,O,F);
  unsigned long long P=mn(pa,it<64?it:64);
  printf("       outer is %.1f%% of full;  inner %.1f%%\n",100.0*O/F,100.0*I/F);
  unsigned long long D=mn(da,it<64?it:64);
  printf("       outer natural(stride512) %llu | permuted(contig) %llu | PADDED(stride520) %llu\n",O,P,D);
  printf("       padding recovers %.1f%% of the %.1f%% scatter penalty\n",100.0*(O-D)/(O-P),100.0*(O-P)/P);
  return 0;
}
