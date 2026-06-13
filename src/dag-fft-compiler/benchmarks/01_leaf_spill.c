#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <pmmintrin.h>
extern void radix64_n1_fwd_avx512(double*,double*,const double*,const double*,size_t,size_t);            /* in-place */
extern void radix64_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t); /* M2 OOP */
static unsigned long long mn(unsigned long long*a,int c){unsigned long long b=~0ULL;for(int i=0;i<c;i++)if(a[i]<b)b=a[i];return b;}
int main(int argc,char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  size_t K=argc>1?atol(argv[1]):64; size_t N=64*K;
  double *a=aligned_alloc(64,N*8),*b=aligned_alloc(64,N*8),*c=aligned_alloc(64,N*8),*d=aligned_alloc(64,N*8);
  for(size_t i=0;i<N;i++){a[i]=(i%101)-50.0;b[i]=(i%97)-48.0;}
  int it=4000; unsigned long long ip[64],op[64];
  for(int w=0;w<8;w++) radix64_n1_fwd_avx512(a,b,0,0,K,K);
  for(int r=0;r<it;r++){unsigned long long c0=__rdtsc(); radix64_n1_fwd_avx512(a,b,0,0,K,K); ip[r%64]=__rdtsc()-c0;}
  for(int w=0;w<8;w++) radix64_n1_oop_fwd_avx512_UG_UG(a,b,c,d,0,0,K,1,K,1,K);
  for(int r=0;r<it;r++){unsigned long long c0=__rdtsc(); radix64_n1_oop_fwd_avx512_UG_UG(a,b,c,d,0,0,K,1,K,1,K); op[r%64]=__rdtsc()-c0;}
  unsigned long long I=mn(ip,it<64?it:64),O=mn(op,it<64?it:64);
  printf("radix64 K=%zu | in-place(floor) %llu | M2 OOP %llu | OOP is %.2fx the in-place cost\n",K,I,O,(double)O/I);
  return 0;
}
