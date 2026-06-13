#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <x86intrin.h>
#include <pmmintrin.h>
extern void radix16_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix16_t1s_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix64_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix64_t1s_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
static unsigned long long mn(unsigned long long*a,int c){unsigned long long b=~0ULL;for(int i=0;i<c;i++)if(a[i]<b)b=a[i];return b;}
int main(void){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  size_t K=4096;
  double *a=aligned_alloc(64,64*K*8),*b=aligned_alloc(64,64*K*8),*c=aligned_alloc(64,64*K*8),*d=aligned_alloc(64,64*K*8);
  double *tw=aligned_alloc(64,64*K*8); for(size_t i=0;i<64*K;i++){a[i]=(i%101)-50.0;b[i]=(i%97)-48.0;tw[i]=cos(0.01*i);}
  unsigned long long n16[64],t16[64],n64[64],t64[64]; int it=2000;
  for(int w=0;w<8;w++) radix16_n1_oop_fwd_avx512_UG_UG(a,b,c,d,0,0,K,1,K,1,K);
  for(int r=0;r<it;r++){unsigned long long s=__rdtsc(); radix16_n1_oop_fwd_avx512_UG_UG(a,b,c,d,0,0,K,1,K,1,K); n16[r%64]=__rdtsc()-s;}
  for(int w=0;w<8;w++) radix16_t1s_oop_fwd_avx512_UG_UG(a,b,c,d,tw,tw,K,1,K,1,K);
  for(int r=0;r<it;r++){unsigned long long s=__rdtsc(); radix16_t1s_oop_fwd_avx512_UG_UG(a,b,c,d,tw,tw,K,1,K,1,K); t16[r%64]=__rdtsc()-s;}
  for(int w=0;w<8;w++) radix64_n1_oop_fwd_avx512_UG_UG(a,b,c,d,0,0,K,1,K,1,K);
  for(int r=0;r<it;r++){unsigned long long s=__rdtsc(); radix64_n1_oop_fwd_avx512_UG_UG(a,b,c,d,0,0,K,1,K,1,K); n64[r%64]=__rdtsc()-s;}
  for(int w=0;w<8;w++) radix64_t1s_oop_fwd_avx512_UG_UG(a,b,c,d,tw,tw,K,1,K,1,K);
  for(int r=0;r<it;r++){unsigned long long s=__rdtsc(); radix64_t1s_oop_fwd_avx512_UG_UG(a,b,c,d,tw,tw,K,1,K,1,K); t64[r%64]=__rdtsc()-s;}
  unsigned long long N16=mn(n16,64),T16=mn(t16,64),N64=mn(n64,64),T64=mn(t64,64);
  printf("size-16: n1 %llu  t1s %llu  -> twiddle adds %.0f%%\n",N16,T16,100.0*(T16-N16)/N16);
  printf("size-64: n1 %llu  t1s %llu  -> twiddle adds %.0f%%\n",N64,T64,100.0*(T64-N64)/N64);
  printf("twiddle on size-16 stage (mine, x4 the count): %.0f extra cyc/call-set vs size-64 stage (FFTW): %.0f\n",(double)(T16-N16),(double)(T64-N64));
  return 0;
}
