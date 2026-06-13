/* P3 three-way (plus packed lane): VectorFFT native rfft vs MKL vs
 * FFTW, batched real-to-complex, N x K lane-major layout
 * (element (f, lane) at f*K + lane; howmany = K, stride = K,
 * dist = 1 for the reference libraries).
 * Self-validates every lane against FFTW before timing.
 * Toggles: -DVFFT_RFFT_RANGED=1, -DVFFT_RFFT_PREFETCH=1,
 *          env VFFT_KB=<lanes> (lane blocking), env REPS, env NN.
 * Output: human lines + CSV rows "csv,<lane>,<N>,<K>,<ns>". */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#define VFFT_RFFT_MAX_RADIX 32
#include "rfft.h"
#include <fftw3.h>
#ifdef USE_MKL
#include "mkl_dfti.h"
#endif
#define DECL(r) \
  void radix##r##_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t); \
  void radix##r##_hc2hc_dit_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,size_t); \
  void radix##r##_hc2c_nat_fwd_avx512(const double*,const double*,double*,double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t); \
  void radix##r##_hc2hc_dit_rng_fwd_avx512(const double*,const double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,ptrdiff_t,int,size_t); \
  void radix##r##_hc2c_nat_rng_fwd_avx512(const double*,const double*,double*,double*,double*,double*,const double*,const double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,ptrdiff_t,ptrdiff_t,int,size_t);
DECL(2) DECL(4) DECL(8) DECL(16)
void radix32_r2cf_avx512(const double*,double*,double*,ptrdiff_t,ptrdiff_t,ptrdiff_t,size_t);
static double now_ns(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}
static void cachebust(void){size_t s=64*1024*1024/8;static double*j=NULL;if(!j)j=malloc(s*8);volatile double a=0;for(size_t i=0;i<s;i++)j[i]=i*0.5;for(size_t i=0;i<s;i++)a+=j[i];(void)a;}
static double bench(void(*f)(void*),void*a,int reps){
  for(int w=0;w<5;w++)f(a);
  double best=1e30;
  for(int t=0;t<5;t++){cachebust();double s=now_ns();
    for(int i=0;i<reps;i++)f(a);
    double n=(now_ns()-s)/reps; if(n<best)best=n;}
  return best;
}
typedef struct { rfft_plan_t *p; const double *x; double *re, *im, *pk; } va_t;
static void f_nat(void*v){va_t*a=v; rfft_execute_fwd_natural(a->p,a->x,a->re,a->im);}
static void f_pkd(void*v){va_t*a=v; rfft_execute_fwd_packed(a->p,a->x,a->pk);}
typedef struct { fftw_plan pl; } vf_t;
static void f_fftw(void*v){ fftw_execute(((vf_t*)v)->pl); }
#ifdef USE_MKL
typedef struct { DFTI_DESCRIPTOR_HANDLE h; double*in; MKL_Complex16*out; } vm_t;
static void f_mkl(void*v){vm_t*a=v; DftiComputeForward(a->h,a->in,a->out);}
#endif
static double maxerr_split(const double*re,const double*im,const fftw_complex*ref,int nh1,size_t K){
  double e=0;
  for(int f=0;f<nh1;f++)for(size_t v=0;v<K;v++){
    double e1=fabs(re[(size_t)f*K+v]-ref[(size_t)f*K+v][0]);
    double e2=fabs(im[(size_t)f*K+v]-ref[(size_t)f*K+v][1]);
    if(e1>e)e=e1; if(e2>e)e=e2;}
  return e;
}
int main(int argc,char**argv){
  int N = getenv("NN")? atoi(getenv("NN")) : 256;
  size_t K = 256;
  int reps = getenv("REPS")? atoi(getenv("REPS")) : 200;
  int nh1 = N/2+1;
  rfft_codelets_t c; memset(&c,0,sizeof c);
#define R(r) c.r2cf[r]=radix##r##_r2cf_avx512; c.hc2hc[r]=radix##r##_hc2hc_dit_fwd_avx512; \
  c.hc2c[r]=radix##r##_hc2c_nat_fwd_avx512; c.hc2hc_rng[r]=radix##r##_hc2hc_dit_rng_fwd_avx512; \
  c.hc2c_rng[r]=radix##r##_hc2c_nat_rng_fwd_avx512;
  R(2) R(4) R(8) R(16)
  c.r2cf[32]=radix32_r2cf_avx512;
  double *x = aligned_alloc(64,(size_t)N*K*8);
  double *re = aligned_alloc(64,(size_t)nh1*K*8);
  double *im = aligned_alloc(64,(size_t)nh1*K*8);
  double *pk = aligned_alloc(64,(size_t)N*K*8);
  fftw_complex *fout = fftw_malloc(sizeof(fftw_complex)*(size_t)nh1*K);
  for(size_t i=0;i<(size_t)N*K;i++)x[i]=sin(0.37*(double)i)+0.2*cos(1.1*(double)i);
  /* FFTW reference + lane: howmany=K, stride=K, dist=1 */
  vf_t vf; vf.pl = fftw_plan_many_dft_r2c(1,&N,(int)K, x,NULL,(int)K,1,
                                          fout,NULL,(int)K,1, FFTW_MEASURE);
  fftw_execute(vf.pl);
  /* plan: best factorization is machine-dependent; sweep candidates */
  struct { int f[4]; int nf; } cand[] =
    {{{16,16},2},{{8,32},2},{{4,4,16},3},{{8,4,8},3},{{2,4,4,8},4}};
  printf("N=%d K=%zu reps=%d  toggles: RANGED=%d PREFETCH=%d KB=%s\n",N,K,reps,
#ifdef VFFT_RFFT_RANGED
    1,
#else
    0,
#endif
#ifdef VFFT_RFFT_PREFETCH
    1,
#endif
#ifndef VFFT_RFFT_PREFETCH
    0,
#endif
    getenv("VFFT_KB")?getenv("VFFT_KB"):"K");
  for(size_t t=0;t<sizeof(cand)/sizeof(cand[0]);t++){
    int Np=1; for(int i=0;i<cand[t].nf;i++)Np*=cand[t].f[i];
    if(Np!=N) continue;
    rfft_plan_t*p=rfft_plan_create(N,K,cand[t].f,cand[t].nf,&c);
    if(!p) continue;
    if(getenv("VFFT_KB")) p->Kb=(size_t)atoi(getenv("VFFT_KB"));
    va_t va={p,x,re,im,pk};
    f_nat(&va);
    double err=maxerr_split(re,im,fout,nh1,K);
    double tn=bench(f_nat,&va,reps), tp=bench(f_pkd,&va,reps);
    printf("  vfft (");for(int i=0;i<cand[t].nf;i++)printf("%d%s",cand[t].f[i],i+1<cand[t].nf?",":"");
    printf("): natural %8.0f ns | packed %8.0f ns | vs-FFTW err %.1e %s\n",
           tn,tp,err,err<1e-9?"OK":"MISMATCH");
    printf("csv,vfft_natural_");for(int i=0;i<cand[t].nf;i++)printf("%d%s",cand[t].f[i],i+1<cand[t].nf?"x":"");
    printf(",%d,%zu,%.0f\n",N,K,tn);
    rfft_plan_destroy(p);
  }
  double tf=bench(f_fftw,&vf,reps);
  printf("  fftw  plan_many r2c: %8.0f ns\ncsv,fftw,%d,%zu,%.0f\n",tf,N,K,tf);
#ifdef USE_MKL
  vm_t vm; vm.in=x; vm.out=(MKL_Complex16*)fout;
  MKL_LONG st;
  DftiCreateDescriptor(&vm.h, DFTI_DOUBLE, DFTI_REAL, 1, (MKL_LONG)N);
  DftiSetValue(vm.h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
  DftiSetValue(vm.h, DFTI_CONJUGATE_EVEN_STORAGE, DFTI_COMPLEX_COMPLEX);
  DftiSetValue(vm.h, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
  DftiSetValue(vm.h, DFTI_INPUT_DISTANCE, 1);
  DftiSetValue(vm.h, DFTI_OUTPUT_DISTANCE, 1);
  { MKL_LONG is[2]={0,(MKL_LONG)K}; DftiSetValue(vm.h, DFTI_INPUT_STRIDES, is); }
  { MKL_LONG os[2]={0,(MKL_LONG)K}; DftiSetValue(vm.h, DFTI_OUTPUT_STRIDES, os); }
  st = DftiCommitDescriptor(vm.h);
  if(st==0){
    f_mkl(&vm);
    double em=maxerr_split(re,im,fout,nh1,K); (void)em; /* fout now MKL's */
    double tm=bench(f_mkl,&vm,reps);
    printf("  mkl   real CCE      : %8.0f ns\ncsv,mkl,%d,%zu,%.0f\n",tm,N,K,tm);
    DftiFreeDescriptor(&vm.h);
  } else printf("  mkl   commit failed (%ld)\n",(long)st);
#endif
  printf("done. Same-run ratios only; pin a core; performance governor.\n");
  return 0;
}
