/* ============================================================================
 * 09_compare_vs_mkl.c  (OPTIONAL benchmark, requires Intel oneMKL)
 *
 * Three-way comparison of the optimized 32x32 OOP engine (log3, stride-
 * specialized + M-project + store-on-compute codelets) against FFTW and
 * Intel MKL DFTI. All single-thread, out-of-place, batched K x N=1024 forward
 * complex FFTs, interleaved rdtsc min-of-120.
 *
 * MKL is NOT a dependency of the rest of the package. To build this one file
 * you must install MKL and point the compiler/linker at it, e.g.:
 *   pip install --break-system-packages mkl mkl-include mkl-devel
 *   gcc -O3 -march=native -mavx512f -mavx512dq -mfma \
 *       -I<fftw>/api -I/usr/local/include \
 *       09_compare_vs_mkl.c \
 *       ../codelets/radix32_n1_oop_avx512_spec.c \
 *       ../codelets/radix32_t1p_log3_oop_avx512_spec.c \
 *       <fftw>/.libs/libfftw3.a -L/usr/local/lib -lmkl_rt -lm -o cmp_mkl
 *   MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 LD_LIBRARY_PATH=/usr/local/lib ./cmp_mkl 128
 *
 * IMPORTANT: set thread count via env vars only. Calling mkl_set_num_threads()
 * in code segfaults in this build.
 *
 * READING THE RESULT: the cache-resident regime (small-ish K where the three
 * buffer sets fit in cache, roughly K<=256 -> ~12MB) is the codelet-bound
 * comparison. Larger K is memory-bandwidth bound and the ratios reflect layout,
 * not codelet quality. See docs/mkl_comparison.md for measured numbers and
 * caveats. MKL/VF > 1 means VectorFFT is faster.
 * ========================================================================== */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#include <fftw3.h>
#include <mkl_dfti.h>
#define N 1024
#define V 8
extern void radix32_n1_oop_fwd_avx512_UG_UG_spec(const double*,const double*,double*,double*,const double*,const double*,size_t);
extern void radix32_t1p_oop_fwd_avx512_UG_UG_log3_spec(const double*,const double*,double*,double*,const double*,const double*,size_t);
static double Q2r[31*32],Q2i[31*32];
static void initw(void){for(int l=1;l<32;l++)for(int k=0;k<32;k++){double a=-2.0*M_PI*(double)(l*k)/1024.0;Q2r[(l-1)*32+k]=cos(a);Q2i[(l-1)*32+k]=sin(a);}}
static inline void Vl(const double*ir,const double*ii,double*o,double*oi){
  radix32_n1_oop_fwd_avx512_UG_UG_spec(ir,ii,o,oi,0,0,256);
  radix32_t1p_oop_fwd_avx512_UG_UG_log3_spec(o,oi,o,oi,Q2r,Q2i,256);
}
static void eVl(const double*ir,const double*ii,double*o,double*oi,size_t K){for(size_t b=0;b<K/V;b++)Vl(ir+b*N*V,ii+b*N*V,o+b*N*V,oi+b*N*V);}
static unsigned long long mn(unsigned long long*a,int c){unsigned long long b=~0ULL;for(int i=0;i<c;i++)if(a[i]<b)b=a[i];return b;}
static double vfyV(const double*ir,const double*ii,double*o,double*oi,const double*xr,const double*xi,size_t K){
  eVl(ir,ii,o,oi,K);fftw_complex*fi=fftw_malloc(sizeof(fftw_complex)*N),*fo=fftw_malloc(sizeof(fftw_complex)*N);
  fftw_plan p=fftw_plan_dft_1d(N,fi,fo,FFTW_FORWARD,FFTW_ESTIMATE);double mr=0;
  for(size_t t=0;t<K;t+=(K>3?K/3:1)){size_t bk=t/V,l=t%V;for(int e=0;e<N;e++){fi[e][0]=xr[t*N+e];fi[e][1]=xi[t*N+e];}fftw_execute(p);
    double me=0,mm=0;for(int k=0;k<N;k++){double dr=o[bk*N*V+(size_t)k*V+l]-fo[k][0],di=oi[bk*N*V+(size_t)k*V+l]-fo[k][1],er=sqrt(dr*dr+di*di),m=hypot(fo[k][0],fo[k][1]);if(er>me)me=er;if(m>mm)mm=m;}if(me/mm>mr)mr=me/mm;}
  fftw_destroy_plan(p);fftw_free(fi);fftw_free(fo);return mr;}
int main(int argc,char**argv){
  _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);_MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  initw();size_t K=argc>1?atol(argv[1]):128,TOT=(size_t)N*K;
  double*ir=aligned_alloc(64,TOT*8),*ii=aligned_alloc(64,TOT*8),*o=aligned_alloc(64,TOT*8),*oi=aligned_alloc(64,TOT*8),*xr=malloc(TOT*8),*xi=malloc(TOT*8);srand(13);
  for(size_t t=0;t<K;t++)for(int e=0;e<N;e++){double vr=(double)rand()/RAND_MAX-0.5,vi=(double)rand()/RAND_MAX-0.5;size_t bk=t/V,l=t%V;xr[t*N+e]=vr;xi[t*N+e]=vi;ir[bk*N*V+(size_t)e*V+l]=vr;ii[bk*N*V+(size_t)e*V+l]=vi;}
  double rVl=vfyV(ir,ii,o,oi,xr,xi,K);
  fftw_complex*gi=fftw_malloc(sizeof(fftw_complex)*TOT),*go=fftw_malloc(sizeof(fftw_complex)*TOT);
  for(size_t i=0;i<TOT;i++){gi[i][0]=sin(0.1*i);gi[i][1]=cos(0.07*i);}
  int nn[1]={N};fftw_plan pe=fftw_plan_many_dft(1,nn,(int)K,gi,NULL,1,N,go,NULL,1,N,FFTW_FORWARD,FFTW_PATIENT);
  double*mi=aligned_alloc(64,2*TOT*8),*mo=aligned_alloc(64,2*TOT*8);
  for(size_t i=0;i<TOT;i++){mi[2*i]=sin(0.1*i);mi[2*i+1]=cos(0.07*i);}
  DFTI_DESCRIPTOR_HANDLE h;
  DftiCreateDescriptor(&h,DFTI_DOUBLE,DFTI_COMPLEX,1,(MKL_LONG)N);
  DftiSetValue(h,DFTI_NUMBER_OF_TRANSFORMS,(MKL_LONG)K);
  DftiSetValue(h,DFTI_INPUT_DISTANCE,(MKL_LONG)N);
  DftiSetValue(h,DFTI_OUTPUT_DISTANCE,(MKL_LONG)N);
  DftiSetValue(h,DFTI_PLACEMENT,DFTI_NOT_INPLACE);
  DftiCommitDescriptor(h);
  DftiComputeForward(h,mi,mo);
  /* MKL correctness: fresh non-destructive FFTW ESTIMATE on block 0 (PATIENT
     planning above would have trashed gi, so do not compare against go here) */
  {fftw_complex*ci=fftw_malloc(sizeof(fftw_complex)*N),*co=fftw_malloc(sizeof(fftw_complex)*N);
   fftw_plan cp=fftw_plan_dft_1d(N,ci,co,FFTW_FORWARD,FFTW_ESTIMATE);
   for(int e=0;e<N;e++){ci[e][0]=mi[2*e];ci[e][1]=mi[2*e+1];}fftw_execute(cp);
   double me=0,mm=0;for(int k=0;k<N;k++){double dr=mo[2*k]-co[k][0],di=mo[2*k+1]-co[k][1],er=sqrt(dr*dr+di*di),m=hypot(co[k][0],co[k][1]);if(er>me)me=er;if(m>mm)mm=m;}
   printf("K=%zu  relerr VF_log3 %.1e  MKL %.1e\n",K,rVl,me/mm);
   fftw_destroy_plan(cp);fftw_free(ci);fftw_free(co);}
  unsigned long long aVl[128],ap[128],am[128];
  for(int w=0;w<8;w++){eVl(ir,ii,o,oi,K);fftw_execute(pe);DftiComputeForward(h,mi,mo);}
  for(int r=0;r<120;r++){unsigned long long c;
    c=__rdtsc();eVl(ir,ii,o,oi,K);aVl[r%128]=__rdtsc()-c;
    c=__rdtsc();fftw_execute(pe);ap[r%128]=__rdtsc()-c;
    c=__rdtsc();DftiComputeForward(h,mi,mo);am[r%128]=__rdtsc()-c;}
  unsigned long long Vl_=mn(aVl,120),P=mn(ap,120),M=mn(am,120);
  printf("  VF_log3 %llu   FFTW %llu   MKL %llu\n",Vl_,P,M);
  printf("  MKL/VF_log3=%.3f (>1 = we beat MKL)   FFTW/MKL=%.3f\n",(double)M/Vl_,(double)P/M);
  DftiFreeDescriptor(&h);return 0;}
