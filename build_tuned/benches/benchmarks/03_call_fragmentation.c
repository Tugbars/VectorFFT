#include <stdio.h>
#include <stdlib.h>
#include <x86intrin.h>
#include <pmmintrin.h>
extern void radix16_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
static unsigned long long mn(unsigned long long*a,int c){unsigned long long b=~0ULL;for(int i=0;i<c;i++)if(a[i]<b)b=a[i];return b;}
int main(void){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  size_t G=512;                      /* 512 groups total = 64 blocks * 8 lanes, the engine's per-block outer work scaled */
  size_t sz=16*G;
  double *a=aligned_alloc(64,sz*8),*b=aligned_alloc(64,sz*8),*c=aligned_alloc(64,sz*8),*d=aligned_alloc(64,sz*8);
  for(size_t i=0;i<sz;i++){a[i]=(i%101)-50.0;b[i]=(i%97)-48.0;}
  unsigned long long fr[64],on[64]; int it=3000;
  /* fragmented: 64 calls of me=8 (mimics per-group), leg stride 16*8 within each chunk */
  for(int w=0;w<8;w++) for(int g=0;g<64;g++) radix16_n1_oop_fwd_avx512_UG_UG(a+g*16*8,b+g*16*8,c+g*16*8,d+g*16*8,0,0,8,1,8,1,8);
  for(int r=0;r<it;r++){unsigned long long t0=__rdtsc(); for(int g=0;g<64;g++) radix16_n1_oop_fwd_avx512_UG_UG(a+g*16*8,b+g*16*8,c+g*16*8,d+g*16*8,0,0,8,1,8,1,8); fr[r%64]=__rdtsc()-t0;}
  /* one call: me=512, same 512 butterflies */
  for(int w=0;w<8;w++) radix16_n1_oop_fwd_avx512_UG_UG(a,b,c,d,0,0,G,1,G,1,G);
  for(int r=0;r<it;r++){unsigned long long t0=__rdtsc(); radix16_n1_oop_fwd_avx512_UG_UG(a,b,c,d,0,0,G,1,G,1,G); on[r%64]=__rdtsc()-t0;}
  unsigned long long Fr=mn(fr,it<64?it:64),On=mn(on,it<64?it:64);
  printf("radix16 same 512 butterflies:  64 calls(me=8) %llu  |  1 call(me=512) %llu  |  fragmentation overhead %.2fx\n",Fr,On,(double)Fr/On);
  return 0;
}
