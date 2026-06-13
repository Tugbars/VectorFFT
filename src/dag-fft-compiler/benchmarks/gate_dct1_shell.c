/* gate_dct1_shell.c — section 58 gate: DCT-I / DST-I production
 * shells (core/dct1.h) at arbitrary N via pad-embedding.
 *
 * dct1 N in {6,12,24,33,100} -> inner M/2 in {5,11,23,32,99}
 * dst1 N in {4,10,22,31,99}  -> inner M/2 in {5,11,23,32,100}
 * (23 is the non-smooth prime: exercises a Rader inner.)
 * Checks: brute formula (lanes 0, K-1) and involution scale,
 * threshold 1e-11.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "executor.h"
#include "planner.h"
#include "dp_planner.h"
#include "threads.h"
#include "proto_stride_compat.h"
#include "bluestein.h"
#include "rader.h"
#include "bluestein_wisdom.h"
#include "r2c.h"
#include "dct1.h"

static stride_registry_t reg;
static vfft_proto_wisdom_t wis;
static bluestein_wisdom_t bw;

static int is_smooth(int n){ static const int pr[]={2,3,5,7,11,13,17,19,0};
  for(const int*p=pr;*p;p++) while(n%*p==0)n/=*p; return n==1; }

/* full-coverage c2c builder: smooth -> auto, else Rader/Bluestein */
static stride_plan_t *build_c2c(int N, size_t K){
  if (is_smooth(N)) return vfft_proto_auto_plan(N, K, &reg, &wis);
  if (is_smooth(N-1)){
    stride_plan_t *rin = vfft_proto_auto_plan(N-1, K, &reg, &wis);
    return rin ? stride_rader_plan(N, K, K, rin) : NULL;
  }
  const bluestein_wisdom_entry_t *e = bluestein_wisdom_lookup(&bw, N, K);
  int M = e ? e->M : _bluestein_choose_m(N);
  size_t B = e ? e->B : _bluestein_block_size(M, K);
  stride_plan_t *bin = vfft_proto_auto_plan(M, B, &reg, &wis);
  return bin ? stride_bluestein_plan(N, K, B, bin, M) : NULL;
}

static stride_plan_t *build_r2c_evenM(int M, size_t K){
  stride_plan_t *inner = build_c2c(M/2, K);
  return inner ? stride_r2c_plan(M, K, K, inner) : NULL;
}

int main(void){
  size_t K = 64;
  vfft_proto_registry_init(&reg);
  vfft_proto_wisdom_load(&wis, "core/vfft_wisdom_tuned.txt");
  bluestein_wisdom_init(&bw);
  bluestein_wisdom_load(&bw, "core/vfft_bluestein_wisdom.txt");

  int all_pass = 1;
  printf("%-6s %-5s %5s %12s %12s %8s\n",
         "kind","N","inner","fwd_err","inv_err","verdict");

  int dct1_N[] = {6, 12, 24, 33, 100};
  for (size_t i = 0; i < 5; i++){
    int N = dct1_N[i], M = 2*(N-1);
    stride_plan_t *r2c = build_r2c_evenM(M, K);
    stride_plan_t *p = r2c ? stride_dct1_plan(N, K, r2c) : NULL;
    if (!p){ printf("dct1   %-5d plan FAIL\n", N); all_pass=0; continue; }
    size_t NK = (size_t)N*K;
    double *x=aligned_alloc(64,NK*8), *y=aligned_alloc(64,NK*8), *z=aligned_alloc(64,NK*8);
    for(size_t q=0;q<NK;q++) x[q]=sin(0.41*(double)q)+0.3*cos(1.7*(double)q);
    stride_execute_dct1(p, x, y);
    double fe=0;
    for (int lane=0; lane<(int)K; lane+=(int)K-1){
      for (int k=0;k<N;k++){
        double s = x[lane] + ((k&1)?-1.0:1.0)*x[(size_t)(N-1)*K+lane];
        for (int n=1;n<=N-2;n++)
          s += 2.0*x[(size_t)n*K+lane]*cos(M_PI*(double)n*(double)k/(double)(N-1));
        double d=fabs(s - y[(size_t)k*K+lane]); if(d>fe)fe=d;
      }
    }
    stride_execute_dct1(p, y, z);
    double sc = 2.0*(N-1), ie=0;
    for (size_t q=0;q<NK;q++){ double d=fabs(z[q]/sc - x[q]); if(d>ie)ie=d; }
    int ok = fe<1e-11 && ie<1e-11; if(!ok) all_pass=0;
    printf("dct1   %-5d %5d %12.3e %12.3e %8s\n", N, M/2, fe, ie, ok?"PASS":"FAIL");
    stride_plan_destroy(p); free(x);free(y);free(z);
  }

  int dst1_N[] = {4, 10, 22, 31, 99};
  for (size_t i = 0; i < 5; i++){
    int N = dst1_N[i], M = 2*(N+1);
    stride_plan_t *r2c = build_r2c_evenM(M, K);
    stride_plan_t *p = r2c ? stride_dst1_plan(N, K, r2c) : NULL;
    if (!p){ printf("dst1   %-5d plan FAIL\n", N); all_pass=0; continue; }
    size_t NK = (size_t)N*K;
    double *x=aligned_alloc(64,NK*8), *y=aligned_alloc(64,NK*8), *z=aligned_alloc(64,NK*8);
    for(size_t q=0;q<NK;q++) x[q]=sin(0.41*(double)q)+0.3*cos(1.7*(double)q);
    stride_execute_dst1(p, x, y);
    double fe=0;
    for (int lane=0; lane<(int)K; lane+=(int)K-1){
      for (int k=0;k<N;k++){
        double s=0;
        for (int n=0;n<N;n++)
          s += 2.0*x[(size_t)n*K+lane]
               *sin(M_PI*(double)(n+1)*(double)(k+1)/(double)(N+1));
        double d=fabs(s - y[(size_t)k*K+lane]); if(d>fe)fe=d;
      }
    }
    stride_execute_dst1(p, y, z);
    double sc = 2.0*(N+1), ie=0;
    for (size_t q=0;q<NK;q++){ double d=fabs(z[q]/sc - x[q]); if(d>ie)ie=d; }
    int ok = fe<1e-11 && ie<1e-11; if(!ok) all_pass=0;
    printf("dst1   %-5d %5d %12.3e %12.3e %8s\n", N, M/2, fe, ie, ok?"PASS":"FAIL");
    stride_plan_destroy(p); free(x);free(y);free(z);
  }

  printf("\n%s\n", all_pass ? "ALL PASS" : "FAILURES PRESENT");
  return all_pass ? 0 : 1;
}
