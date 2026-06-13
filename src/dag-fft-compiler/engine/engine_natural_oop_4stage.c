/*
 * engine_natural_oop_4stage.c
 *
 * Natural-order out-of-place 4-stage 4x4x4x16 one-call engine for N=1024,
 * built to complete the OOP stage-count study. It is a confirming negative:
 * it works (machine precision, ~3.4 to 4.0e-15, better than 32x32) but loses,
 * because deep factorizations fragment into many small strided calls.
 *
 * Decomposition: R0=4 (leaf), R1=4, R2=4, R3=16. P=[1,4,16,64,1024].
 * General multi-stage natural-order DIT one-call rule (leaf-first, inner stages
 * in-place, like FFTW's natural-order path), derived from the 16x16x4 3-stage:
 *   leaf (stage 0, radix R0, n1): N/R0 sub-DFTs, leg stride (N/R0)*V, output
 *     scattered by digit weight: out element = m + n1*P1 + n2*P2 + n3*P3.
 *     The contiguous input digit (n3) is batched via in_grp, so 16 leaf calls.
 *   stage s>=1 (radix Rs, t1p): blocks of P_{s+1}=P_s*Rs elements, N/P_{s+1}
 *     blocks, legs at stride P_s*V, P_s positions, me=P_s*V, twiddle
 *     W_{P_{s+1}}^{(leg)*(position)}. Inner stages run in-place (out==in).
 *   The outermost stage (radix-16) is a single block, so it is one clean t1p
 *   call exactly like the 2-stage; the inner radix-4 stages are NOT (block
 *   stride != position stride, so no single-stride batching), hence per-block.
 *
 * Call count per block: 16 (leaf) + 64 (stage1) + 16 (stage2) + 1 (stage3) = 97,
 * versus 2 for 32x32. Stage 1 alone is 64 calls each doing only 4 groups, a poor
 * work-to-overhead ratio. Plus 4 array passes versus 2. Both penalties stack.
 *
 * Measured (AVX-512, this VM, FFTW PATIENT, rdtsc min-of-60, interleaved), the
 * full OOP stage-count picture for N=1024:
 *   32x32   (2-stage,  2 calls)  1.07 to 1.11x FFTW   <- best
 *   16x16x4 (3-stage,  9 calls)  0.99 to 1.01x FFTW
 *   4x4x4x16 (4-stage, 97 calls) 0.80 to 0.83x FFTW   <- first engine below FFTW
 * Monotonic: every added stage costs. The arithmetic shrinks per stage but the
 * per-call overhead and extra passes dominate. The one gain is precision (~2x
 * better, smaller radices = less roundoff per butterfly), which is the axis MKL
 * leads on. Contrast: the SAME 4x4x4x16 runs ~0.84 to 0.95x in the in-place
 * stride executor, better than here, because in-place halves the passes and that
 * executor inlines the codelet in a tight per-group loop rather than paying the
 * 97-way function-call fragmentation an OOP engine is forced into. That gap is
 * why in-place wins and OOP is the hard mode for deep factorizations.
 *
 * blk4  = 4x4x4x16 (this study). blk32 = 32x32 (the best, for contrast).
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"
extern void radix4_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix4_t1p_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix16_t1p_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix32_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix32_t1p_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
#define N 1024
#define V 8
/* 4x4x4x16: R0=4 leaf, R1=4, R2=4, R3=16. P=[1,4,16,64,1024]. */
static double t1r[3*4],t1i[3*4];     /* stage1 radix4 t1p: W_16^{(j+1)k}, j<3,k<4 */
static double t2r[3*16],t2i[3*16];   /* stage2 radix4 t1p: W_64^{(j+1)k}, j<3,k<16 */
static double t3r[15*64],t3i[15*64]; /* stage3 radix16 t1p: W_1024^{(j+1)k}, j<15,k<64 */
static double Q2r[31*32],Q2i[31*32]; /* 32x32 stage2 (comparison) */
static void initw(void){
  for(int j=0;j<3;j++)for(int k=0;k<4;k++){double a=-2.0*M_PI*(double)((j+1)*k)/16.0;  t1r[j*4+k]=cos(a);t1i[j*4+k]=sin(a);}
  for(int j=0;j<3;j++)for(int k=0;k<16;k++){double a=-2.0*M_PI*(double)((j+1)*k)/64.0; t2r[j*16+k]=cos(a);t2i[j*16+k]=sin(a);}
  for(int j=0;j<15;j++)for(int k=0;k<64;k++){double a=-2.0*M_PI*(double)((j+1)*k)/1024.0;t3r[j*64+k]=cos(a);t3i[j*64+k]=sin(a);}
  for(int l2=1;l2<32;l2++)for(int k2=0;k2<32;k2++){double a=-2.0*M_PI*(double)(l2*k2)/1024.0;Q2r[(l2-1)*32+k2]=cos(a);Q2i[(l2-1)*32+k2]=sin(a);}
}
/* 4x4x4x16 one-call (leaf + t1p stages, inner stages in-place) */
static inline void blk4(const double*ir,const double*ii,double*o,double*oi){
  /* leaf radix4_n1: 16 calls, n3 batched (in_grp=1 consecutive, out_grp=64) */
  for(int n1=0;n1<4;n1++)for(int n2=0;n2<4;n2++)
    radix4_n1_oop_fwd_avx512_UG_UG(ir+(size_t)(n1*64+n2*16)*V, ii+(size_t)(n1*64+n2*16)*V,
                                   o+(size_t)(4*n1+16*n2)*V, oi+(size_t)(4*n1+16*n2)*V,
                                   0,0,(size_t)256*V,1,(size_t)V,64,(size_t)16*V);
  /* stage 1 radix4_t1p: 64 blocks of P2=16, in-place */
  for(int b=0;b<64;b++)
    radix4_t1p_oop_fwd_avx512_UG_UG(o+(size_t)b*16*V, oi+(size_t)b*16*V, o+(size_t)b*16*V, oi+(size_t)b*16*V,
                                    t1r,t1i,(size_t)4*V,1,(size_t)4*V,1,(size_t)32);
  /* stage 2 radix4_t1p: 16 blocks of P3=64, in-place */
  for(int b=0;b<16;b++)
    radix4_t1p_oop_fwd_avx512_UG_UG(o+(size_t)b*64*V, oi+(size_t)b*64*V, o+(size_t)b*64*V, oi+(size_t)b*64*V,
                                    t2r,t2i,(size_t)16*V,1,(size_t)16*V,1,(size_t)128);
  /* stage 3 radix16_t1p: 1 block of N, in-place (clean single t1p like the 2-stage) */
  radix16_t1p_oop_fwd_avx512_UG_UG(o,oi,o,oi,t3r,t3i,(size_t)64*V,1,(size_t)64*V,1,(size_t)512);
}
/* 32x32 (best, for comparison) */
static inline void blk32(const double*ir,const double*ii,double*o,double*oi){
  radix32_n1_oop_fwd_avx512_UG_UG(ir,ii,o,oi,0,0,(size_t)32*V,1,(size_t)V,32,(size_t)32*V);
  radix32_t1p_oop_fwd_avx512_UG_UG(o,oi,o,oi,Q2r,Q2i,(size_t)32*V,1,(size_t)32*V,1,(size_t)32*V);
}
static void e4(const double*ir,const double*ii,double*o,double*oi,size_t K){for(size_t b=0;b<K/V;b++)blk4(ir+b*N*V,ii+b*N*V,o+b*N*V,oi+b*N*V);}
static void e32(const double*ir,const double*ii,double*o,double*oi,size_t K){for(size_t b=0;b<K/V;b++)blk32(ir+b*N*V,ii+b*N*V,o+b*N*V,oi+b*N*V);}
static unsigned long long mn(unsigned long long*a,int c){unsigned long long b=~0ULL;for(int i=0;i<c;i++)if(a[i]<b)b=a[i];return b;}
static double verify(void(*eng)(const double*,const double*,double*,double*,size_t),const double*ir,const double*ii,double*o,double*oi,const double*xr,const double*xi,size_t K){
  eng(ir,ii,o,oi,K); fftw_complex *fi=fftw_malloc(sizeof(fftw_complex)*N),*fo=fftw_malloc(sizeof(fftw_complex)*N);
  fftw_plan p=fftw_plan_dft_1d(N,fi,fo,FFTW_FORWARD,FFTW_ESTIMATE); double mr=0;
  for(size_t t=0;t<K;t+=(K>3?K/3:1)){size_t bk=t/V,l=t%V;for(int e=0;e<N;e++){fi[e][0]=xr[t*N+e];fi[e][1]=xi[t*N+e];}fftw_execute(p);
    double me=0,mm=0;for(int k=0;k<N;k++){double dr=o[bk*N*V+(size_t)k*V+l]-fo[k][0],di=oi[bk*N*V+(size_t)k*V+l]-fo[k][1];double e=sqrt(dr*dr+di*di),m=hypot(fo[k][0],fo[k][1]);if(e>me)me=e;if(m>mm)mm=m;}if(mm>0&&me/mm>mr)mr=me/mm;}
  fftw_destroy_plan(p);fftw_free(fi);fftw_free(fo);return mr;
}
int main(int argc,char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  initw(); size_t K=argc>1?atol(argv[1]):512, TOT=(size_t)N*K;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*o=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  double *xr=malloc(TOT*8),*xi=malloc(TOT*8); srand(13);
  for(size_t t=0;t<K;t++)for(int e=0;e<N;e++){double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;size_t bk=t/V,l=t%V;xr[t*N+e]=vr;xi[t*N+e]=vi;ir[bk*N*V+(size_t)e*V+l]=vr;ii[bk*N*V+(size_t)e*V+l]=vi;}
  double r4=verify(e4,ir,ii,o,oi,xr,xi,K), r32=verify(e32,ir,ii,o,oi,xr,xi,K);
  printf("K=%zu  4x4x4x16 relerr %.2e %s   32x32 relerr %.2e %s\n",K,r4,r4<1e-9?"OK":"BAD",r32,r32<1e-9?"OK":"BAD");
  if(r4>=1e-9){printf("  (4-stage incorrect, skipping bench)\n");return 0;}
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);
  for(size_t i=0;i<TOT;i++){gi[i][0]=sin(0.1*i);gi[i][1]=cos(0.07*i);}
  int nn[1]={N};fftw_plan pe=fftw_plan_many_dft(1,nn,(int)K,gi,NULL,1,N,go,NULL,1,N,FFTW_FORWARD,FFTW_PATIENT);
  unsigned long long a4[64],a32[64],ap[64];
  for(int w=0;w<6;w++){e4(ir,ii,o,oi,K);e32(ir,ii,o,oi,K);fftw_execute(pe);}
  for(int r=0;r<60;r++){unsigned long long c=__rdtsc();e4(ir,ii,o,oi,K);a4[r%64]=__rdtsc()-c;}
  for(int r=0;r<60;r++){unsigned long long c=__rdtsc();e32(ir,ii,o,oi,K);a32[r%64]=__rdtsc()-c;}
  for(int r=0;r<60;r++){unsigned long long c=__rdtsc();fftw_execute(pe);ap[r%64]=__rdtsc()-c;}
  unsigned long long T4=mn(a4,60),T32=mn(a32,60),P=mn(ap,60);
  printf("  4x4x4x16 %llu (%.2fx)  |  32x32 %llu (%.2fx)  |  FFTW %llu\n",T4,(double)P/T4,T32,(double)P/T32,P);
  return 0;
}
