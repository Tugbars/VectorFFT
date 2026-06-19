/* gate_r2c_odd.c — section 57 gate: odd-N r2c/c2r Phase 1.
 *
 * fwd: stride_execute_r2c vs brute real-input DFT, rows 0..N/2.
 * bwd: c2r(r2c(x)) == N*x (convention witnessed by the even-N control).
 * Inner plans: smooth odd via vfft_proto_auto_plan; primes via
 * Rader (N-1 smooth) or Bluestein, exactly as bench_rader does.
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

static int is_smooth(int n){ static const int pr[]={2,3,5,7,11,13,17,19,0};
  for(const int*p=pr;*p;p++) while(n%*p==0)n/=*p; return n==1; }

int main(void){
  int Ns[] = {8, 9, 15, 21, 105, 17, 31, 47};
  size_t K = 64;
  stride_registry_t reg; vfft_proto_registry_init(&reg);
  vfft_proto_wisdom_t wis;
  vfft_proto_wisdom_load(&wis, "core/vfft_wisdom_tuned.txt");
  bluestein_wisdom_t bw; bluestein_wisdom_init(&bw);
  bluestein_wisdom_load(&bw, "core/vfft_bluestein_wisdom.txt");

  int all_pass = 1;
  printf("%-5s %-10s %12s %12s %12s\n", "N", "inner", "fwd_err", "rt_err", "verdict");
  for (size_t ni = 0; ni < sizeof(Ns)/sizeof(Ns[0]); ni++){
    int N = Ns[ni];
    int H = N/2 + 1;
    const char *kind;
    stride_plan_t *inner;
    if (!(N & 1) || is_smooth(N)) {
      /* even path takes a HALF-N inner; odd Phase 1 takes full-N */
      int innerN = (N & 1) ? N : N / 2;
      inner = vfft_proto_auto_plan(innerN, K, &reg, &wis);
      kind = "auto";
    } else if (is_smooth(N-1)) {
      stride_plan_t *rin = vfft_proto_auto_plan(N-1, K, &reg, &wis);
      inner = rin ? stride_rader_plan(N, K, K, rin) : NULL;
      kind = "rader";
    } else {
      const bluestein_wisdom_entry_t *e = bluestein_wisdom_lookup(&bw, N, K);
      int M = e ? e->M : _bluestein_choose_m(N);
      size_t B = e ? e->B : _bluestein_block_size(M, K);
      stride_plan_t *bin = vfft_proto_auto_plan(M, B, &reg, &wis);
      inner = bin ? stride_bluestein_plan(N, K, B, bin, M) : NULL;
      kind = "bluestein";
    }
    if (!inner){ printf("%-5d %-10s inner plan FAIL\n", N, kind); all_pass = 0; continue; }
    stride_plan_t *plan = stride_r2c_plan(N, K, K, inner);
    if (!plan){ printf("%-5d %-10s r2c plan FAIL\n", N, kind); all_pass = 0; continue; }

    size_t NK = (size_t)N * K;
    double *x  = aligned_alloc(64, NK*8);
    double *yr = aligned_alloc(64, NK*8);
    double *yi = aligned_alloc(64, NK*8);
    double *z  = aligned_alloc(64, NK*8);
    for (size_t i = 0; i < NK; i++)
      x[i] = sin(0.41*(double)i) + 0.3*cos(1.7*(double)i);

    stride_execute_r2c(plan, x, yr, yi);

    /* brute, lanes 0 and K-1 */
    double fwd_err = 0;
    for (int lane = 0; lane < (int)K; lane += (int)K - 1){
      for (int k = 0; k < H; k++){
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++){
          double a = -2.0*M_PI*(double)n*(double)k/(double)N;
          sr += x[(size_t)n*K+lane]*cos(a);
          si += x[(size_t)n*K+lane]*sin(a);
        }
        double dr = fabs(sr - yr[(size_t)k*K+lane]);
        double di = fabs(si - yi[(size_t)k*K+lane]);
        if (dr > fwd_err) fwd_err = dr;
        if (di > fwd_err) fwd_err = di;
      }
      if (K == 1) break;
    }

    stride_execute_c2r(plan, yr, yi, z);
    double rt_err = 0;
    for (size_t i = 0; i < NK; i++){
      double d = fabs(z[i]/(double)N - x[i]);
      if (d > rt_err) rt_err = d;
    }

    int ok = (fwd_err < 1e-11) && (rt_err < 1e-11);
    if (!ok) all_pass = 0;
    printf("%-5d %-10s %12.3e %12.3e %12s\n", N, kind, fwd_err, rt_err,
           ok ? "PASS" : "FAIL");
    stride_plan_destroy(plan);
    free(x); free(yr); free(yi); free(z);
  }
  printf("\n%s\n", all_pass ? "ALL PASS" : "FAILURES PRESENT");
  return all_pass ? 0 : 1;
}
