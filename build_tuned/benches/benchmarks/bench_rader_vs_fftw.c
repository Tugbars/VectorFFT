/* bench_rader_vs_fftw.c — prime-N cells: VectorFFT Rader/Bluestein
 * (tuned via bluestein wisdom + stride wisdom inner plans) vs FFTW
 * PATIENT on its home interleaved layout (FFTW runs its own Rader).
 * In-place split on our side. Correctness gate per cell vs naive DFT.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "executor.h"
#include "planner.h"
#include "dp_planner.h"
#include "threads.h"
#include "proto_stride_compat.h"
#include "bluestein.h"
#include "rader.h"
#include "bluestein_wisdom.h"
#include <fftw3.h>

static double now_ns2(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}
static void cachebust(void){size_t s=32*1024*1024/8;double*j=malloc(s*8);volatile double a=0;for(size_t i=0;i<s;i++)j[i]=i*0.5;for(size_t i=0;i<s;i++)a+=j[i];(void)a;free(j);}
static int reps_for(double one_ns){int r=(int)(3e7/(one_ns+1));if(r<8)r=8;if(r>100000)r=100000;return r;}

static double bench_v(stride_plan_t *p, double *re, double *im){
  for(int w=0;w<8;w++) stride_execute_fwd(p, re, im);
  double t0=now_ns2(); stride_execute_fwd(p,re,im); double one=now_ns2()-t0;
  int reps=reps_for(one); double best=1e30;
  for(int t=0;t<5;t++){ double a=now_ns2();
    for(int i=0;i<reps;i++) stride_execute_fwd(p,re,im);
    double ns=(now_ns2()-a)/reps; if(ns<best)best=ns; }
  return best;
}
static double bench_f(fftw_plan p, double one_guess){
  for(int w=0;w<8;w++) fftw_execute(p);
  double t0=now_ns2(); fftw_execute(p); double one=now_ns2()-t0;
  int reps=reps_for(one>0?one:one_guess); double best=1e30;
  for(int t=0;t<5;t++){ double a=now_ns2();
    for(int i=0;i<reps;i++) fftw_execute(p);
    double ns=(now_ns2()-a)/reps; if(ns<best)best=ns; }
  return best;
}

static int is_smooth(int n){ static const int pr[]={2,3,5,7,11,13,17,19,0};
  for(const int*p=pr;*p;p++) while(n%*p==0)n/=*p; return n==1; }

int main(void){
  int Ns[] = {127, 257, 401, 83, 107, 179};
  size_t K = 256;
  stride_registry_t reg; vfft_proto_registry_init(&reg);
  vfft_proto_wisdom_t wis;
  vfft_proto_wisdom_load(&wis, "core/vfft_wisdom_tuned.txt");
  printf("stride wisdom: %zu entries\n", wis.count);
  bluestein_wisdom_t bw; bluestein_wisdom_init(&bw);
  int nb = bluestein_wisdom_load(&bw, "core/vfft_bluestein_wisdom.txt");
  printf("bluestein wisdom: %d entries\n\n", nb);
  printf("%-5s %-9s %-4s %-4s %12s %12s %7s %9s\n",
         "N","algo","M","B","vfft_ns","fftw_ns","ratio","maxerr");
  for (size_t ni=0; ni<sizeof(Ns)/sizeof(Ns[0]); ni++){
    int N = Ns[ni];
    const bluestein_wisdom_entry_t *e = bluestein_wisdom_lookup(&bw, N, K);
    int rader = is_smooth(N-1);
    int M = e ? e->M : (rader ? N-1 : _bluestein_choose_m(N));
    size_t B = e ? e->B : _bluestein_block_size(M, K);
    stride_plan_t *inner = vfft_proto_auto_plan(M, B, &reg, &wis);
    if(!inner){ printf("%-5d inner plan FAIL (M=%d B=%zu)\n", N, M, B); continue; }
    stride_plan_t *plan = rader ? stride_rader_plan(N, K, B, inner)
                                : stride_bluestein_plan(N, K, B, inner, M);
    if(!plan){ printf("%-5d plan FAIL\n", N); continue; }
    size_t total=(size_t)N*K;
    double *re=aligned_alloc(64,total*8), *im=aligned_alloc(64,total*8);
    double *rr=malloc(total*8), *ri=malloc(total*8);
    for(size_t i=0;i<total;i++){ re[i]=rr[i]=sin(0.37*i)+0.2; im[i]=ri[i]=cos(0.211*i)-0.1; }
    /* correctness: lane 0 vs naive DFT */
    stride_execute_fwd(plan, re, im);
    double maxerr=0;
    for(int el=0;el<N;el++){
      double sr=0,si=0;
      for(int j=0;j<N;j++){ double ang=-2.0*M_PI*el*(double)j/N;
        sr += rr[(size_t)j*K]*cos(ang) - ri[(size_t)j*K]*sin(ang);
        si += rr[(size_t)j*K]*sin(ang) + ri[(size_t)j*K]*cos(ang); }
      double dr=fabs(sr-re[(size_t)el*K]), di=fabs(si-im[(size_t)el*K]);
      if(dr>maxerr)maxerr=dr; if(di>maxerr)maxerr=di;
    }
    /* restore data, time ours */
    for(size_t i=0;i<total;i++){ re[i]=rr[i]; im[i]=ri[i]; }
    cachebust();
    double vns = bench_v(plan, re, im);
    /* fftw home interleaved */
    fftw_complex *buf=fftw_malloc(total*sizeof(fftw_complex));
    for(size_t i=0;i<total;i++){ buf[i][0]=rr[i]; buf[i][1]=ri[i]; }
    fftw_plan fp=fftw_plan_many_dft(1,&N,(int)K,buf,NULL,1,N,buf,NULL,1,N,FFTW_FORWARD,FFTW_PATIENT);
    cachebust();
    double fns = bench_f(fp, vns);
    printf("%-5d %-9s %-4d %-4zu %12.0f %12.0f %6.2fx %9.1e %s\n",
           N, rader?"rader":"bluestein", M, B, vns, fns, fns/vns, maxerr,
           maxerr<1e-8?"PASS":"FAIL");
    fftw_destroy_plan(fp); fftw_free(buf);
    stride_plan_destroy(plan);
    free(re);free(im);free(rr);free(ri);
  }
  return 0;
}
