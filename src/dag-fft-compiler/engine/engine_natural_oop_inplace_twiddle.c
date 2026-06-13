/*
 * engine_natural_oop_inplace_twiddle.c
 *
 * The in-place-twiddle (FFTW-method) natural-order out-of-place engine for
 * N=1024, and a head-to-head against both the older work-buffer engine and
 * FFTW PATIENT. See docs/OOP_DESIGN.md section 3 for the derivation.
 *
 * Structure: one out-of-place move (size-16 notw leaf, input -> output), then
 * an in-place twiddle stage (size-64, in == out) on the output. No work buffer,
 * no second shuffle. Natural order, input preserved, machine precision.
 *
 * block_fftw = this engine (16x64 in-place twiddle).
 * block_mine = the older work-buffer engine (64x16, two OOP moves) for contrast.
 *
 * Codelets (in ../codelets): radix16_n1_oop, radix64_t1s_oop, radix64_n1_oop,
 * radix16_t1s_oop. Build: see build.sh.
 *
 * Measured (AVX-512, this box, FFTW PATIENT, rdtsc min-of-many): the in-place
 * twiddle engine is ~0.87-0.93x of FFTW and ~1.17-1.20x over the work-buffer
 * engine in the L3-resident regime. Numbers are directional on a noisy VM.
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <x86intrin.h>
#include <pmmintrin.h>
#include "fftw3.h"
extern void radix16_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix64_t1s_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix64_n1_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
extern void radix16_t1s_oop_fwd_avx512_UG_UG(const double*,const double*,double*,double*,const double*,const double*,size_t,size_t,size_t,size_t,size_t);
#define N 1024
#define V 8
#define R 64   /* radix / twiddle step */
#define M 16   /* cofactor / notw leaf */
/* FFTW-method twiddles: W_N^{n1*k2}, k2 in [0,M), leg n1 in [1,R) */
static double FwR[M*(R-1)], FwI[M*(R-1)];
/* my-current-engine twiddles: W_N^{n2*k1}, k1 in [0,64) groups, leg n2 in [1,16) */
static double CwR[64*15], CwI[64*15];
static double WORK_R[N*V], WORK_I[N*V];
static void initw(void){
  for(int k2=0;k2<M;k2++)for(int n1=1;n1<R;n1++){double a=-2.0*M_PI*(double)(n1*k2)/(double)N; FwR[k2*(R-1)+(n1-1)]=cos(a); FwI[k2*(R-1)+(n1-1)]=sin(a);}
  for(int k1=0;k1<64;k1++)for(int n2=1;n2<16;n2++){double a=-2.0*M_PI*(double)(n2*k1)/(double)N; CwR[k1*15+(n2-1)]=cos(a); CwI[k1*15+(n2-1)]=sin(a);}
}
/* FFTW method: notw leaf input->output (OOP), then in-place twiddle on output. NO work buffer. */
static inline void block_fftw(const double*ir,const double*ii,double*orr,double*oi){
  for(int n1=0;n1<R;n1++)   /* stage1: 64 size-16 notw sub-DFTs, gather in (stride R*V), scatter out (stride V) */
    radix16_n1_oop_fwd_avx512_UG_UG(ir+(size_t)n1*V, ii+(size_t)n1*V, orr+(size_t)M*n1*V, oi+(size_t)M*n1*V, 0,0,
        (size_t)R*V, 1, (size_t)V, 1, (size_t)V);
  for(int k2=0;k2<M;k2++)   /* stage2: 16 size-64 twiddle radix, IN-PLACE on output (in==out), stride M*V */
    radix64_t1s_oop_fwd_avx512_UG_UG(orr+(size_t)k2*V, oi+(size_t)k2*V, orr+(size_t)k2*V, oi+(size_t)k2*V,
        FwR+(size_t)k2*(R-1), FwI+(size_t)k2*(R-1), (size_t)M*V, 1, (size_t)M*V, 1, (size_t)V);
}
/* my current engine: notw inner -> WORK (OOP), OOP twiddle WORK -> output. Work buffer + 2 OOP moves. */
static inline void block_mine(const double*ir,const double*ii,double*orr,double*oi){
  radix64_n1_oop_fwd_avx512_UG_UG(ir,ii,WORK_R,WORK_I,0,0,(size_t)16*V,1,(size_t)16*V,1,(size_t)16*V);
  for(int k1=0;k1<64;k1++)
    radix16_t1s_oop_fwd_avx512_UG_UG(WORK_R+(size_t)16*k1*V,WORK_I+(size_t)16*k1*V, orr+(size_t)k1*V,oi+(size_t)k1*V,
        CwR+(size_t)k1*15,CwI+(size_t)k1*15, (size_t)V,1,(size_t)64*V,1,(size_t)V);
}
static void eng_fftw(const double*ir,const double*ii,double*orr,double*oi,size_t K){for(size_t b=0;b<K/V;b++)block_fftw(ir+b*N*V,ii+b*N*V,orr+b*N*V,oi+b*N*V);}
static void eng_mine(const double*ir,const double*ii,double*orr,double*oi,size_t K){for(size_t b=0;b<K/V;b++)block_mine(ir+b*N*V,ii+b*N*V,orr+b*N*V,oi+b*N*V);}
static unsigned long long mn(unsigned long long*a,int c){unsigned long long b=~0ULL;for(int i=0;i<c;i++)if(a[i]<b)b=a[i];return b;}
int main(int argc,char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  initw(); size_t K=argc>1?atol(argv[1]):512, TOT=(size_t)N*K;
  double *ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*orr=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8);
  double *xr=malloc(TOT*8),*xi=malloc(TOT*8); srand(13);
  for(size_t t=0;t<K;t++)for(int e=0;e<N;e++){double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5; size_t blk=t/V,lane=t%V; xr[t*N+e]=vr;xi[t*N+e]=vi; ir[blk*N*V+(size_t)e*V+lane]=vr; ii[blk*N*V+(size_t)e*V+lane]=vi;}
  eng_fftw(ir,ii,orr,oi,K);
  /* verify FFTW-method (natural order) */
  fftw_complex *fin=fftw_malloc(sizeof(fftw_complex)*N),*fout=fftw_malloc(sizeof(fftw_complex)*N);
  fftw_plan p=fftw_plan_dft_1d(N,fin,fout,FFTW_FORWARD,FFTW_ESTIMATE);
  double mr=0; for(size_t t=0;t<K;t+=(K>3?K/3:1)){size_t blk=t/V,lane=t%V; for(int e=0;e<N;e++){fin[e][0]=xr[t*N+e];fin[e][1]=xi[t*N+e];} fftw_execute(p);
    double me=0,mm=0; for(int k=0;k<N;k++){double dr=orr[blk*N*V+(size_t)k*V+lane]-fout[k][0],di=oi[blk*N*V+(size_t)k*V+lane]-fout[k][1]; double e=sqrt(dr*dr+di*di),m=hypot(fout[k][0],fout[k][1]); if(e>me)me=e; if(m>mm)mm=m;} if(me/mm>mr)mr=me/mm;}
  printf("K=%zu  FFTW-method rel err %.2e %s\n",K,mr,mr<1e-9?"OK":"BAD");
  /* benchmark: FFTW-method vs my current vs FFTW PATIENT */
  fftw_complex *gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);
  for(size_t i=0;i<TOT;i++){gi[i][0]=sin(0.1*i);gi[i][1]=cos(0.07*i);}
  int nn[1]={N}; fftw_plan pe=fftw_plan_many_dft(1,nn,(int)K,gi,NULL,1,N,go,NULL,1,N,FFTW_FORWARD,FFTW_PATIENT);
  unsigned long long fa[64],ma[64],pa[64];
  for(int w=0;w<6;w++){eng_fftw(ir,ii,orr,oi,K);eng_mine(ir,ii,orr,oi,K);fftw_execute(pe);}
  for(int r=0;r<60;r++){unsigned long long c0=__rdtsc();eng_fftw(ir,ii,orr,oi,K);fa[r%64]=__rdtsc()-c0;}
  for(int r=0;r<60;r++){unsigned long long c0=__rdtsc();eng_mine(ir,ii,orr,oi,K);ma[r%64]=__rdtsc()-c0;}
  for(int r=0;r<60;r++){unsigned long long c0=__rdtsc();fftw_execute(pe);pa[r%64]=__rdtsc()-c0;}
  unsigned long long F=mn(fa,60),Mi=mn(ma,60),P=mn(pa,60);
  printf("  FFTW-method %llu (%.2fx FFTW)  | my current 64x16 %llu (%.2fx)  | FFTW-PATIENT %llu\n",F,(double)P/F,Mi,(double)P/Mi,P);
  return 0;
}
