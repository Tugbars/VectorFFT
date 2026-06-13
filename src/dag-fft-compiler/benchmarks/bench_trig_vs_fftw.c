/* bench_trig_vs_fftw.c — OCaml-generated trig codelets (lean r2r ABI,
 * codelets/trig/avx512) vs FFTW r2r PATIENT at the same kinds/sizes.
 *
 * Kinds: dct2/REDFT10, dct3/REDFT01, dct4/REDFT11, dht/FFTW_DHT,
 * dst2/RODFT10, dst3/RODFT01, dst4/RODFT11 (pow2 sizes), and the
 * boundary kinds dct1/REDFT00 (N=5/9/17/33), dst1/RODFT00
 * (N=3/7/15/31) at their natural codelet sizes (section 58).
 * Sizes: N = 8, 16, 32, 64. Batch K = 256, split-batched layout
 * in[n*K + k] on our side.
 *
 * Two FFTW columns per the house methodology:
 *   home  — fftw_plan_many_r2r, istride=1, idist=N (their layout)
 *   split — istride=K, idist=1 (our layout; the adapter-tax column)
 *
 * Correctness: ours vs FFTW elementwise per cell (conventions match
 * FFTW's unnormalized definitions exactly, per core/dct.h docs).
 * Timing: warmup, adaptive reps, min-of-5, cachebust per cell.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <fftw3.h>

#define MAXN 64
#define KB 256

typedef void (*r2r_fn)(const double *, double *, size_t);

void radix8_dct2_avx512(const double*,double*,size_t);
void radix16_dct2_avx512(const double*,double*,size_t);
void radix32_dct2_avx512(const double*,double*,size_t);
void radix64_dct2_avx512(const double*,double*,size_t);
void radix8_dct3_avx512(const double*,double*,size_t);
void radix16_dct3_avx512(const double*,double*,size_t);
void radix32_dct3_avx512(const double*,double*,size_t);
void radix64_dct3_avx512(const double*,double*,size_t);
void radix8_dct4_avx512(const double*,double*,size_t);
void radix16_dct4_avx512(const double*,double*,size_t);
void radix32_dct4_avx512(const double*,double*,size_t);
void radix64_dct4_avx512(const double*,double*,size_t);
void radix8_dht_avx512(const double*,double*,size_t);
void radix16_dht_avx512(const double*,double*,size_t);
void radix32_dht_avx512(const double*,double*,size_t);
void radix64_dht_avx512(const double*,double*,size_t);
void radix8_dst2_avx512(const double*,double*,size_t);
void radix16_dst2_avx512(const double*,double*,size_t);
void radix32_dst2_avx512(const double*,double*,size_t);
void radix64_dst2_avx512(const double*,double*,size_t);
void radix8_dst3_avx512(const double*,double*,size_t);
void radix16_dst3_avx512(const double*,double*,size_t);
void radix32_dst3_avx512(const double*,double*,size_t);
void radix64_dst3_avx512(const double*,double*,size_t);
void radix8_dst4_avx512(const double*,double*,size_t);
void radix16_dst4_avx512(const double*,double*,size_t);
void radix32_dst4_avx512(const double*,double*,size_t);
void radix64_dst4_avx512(const double*,double*,size_t);
void radix5_dct1_avx512(const double*,double*,size_t);
void radix9_dct1_avx512(const double*,double*,size_t);
void radix17_dct1_avx512(const double*,double*,size_t);
void radix33_dct1_avx512(const double*,double*,size_t);
void radix3_dst1_avx512(const double*,double*,size_t);
void radix7_dst1_avx512(const double*,double*,size_t);
void radix15_dst1_avx512(const double*,double*,size_t);
void radix31_dst1_avx512(const double*,double*,size_t);

static double now_ns2(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return ts.tv_sec*1e9+ts.tv_nsec;}
static void cachebust(void){size_t s=32*1024*1024/8;double*j=malloc(s*8);volatile double a=0;for(size_t i=0;i<s;i++)j[i]=i*0.5;for(size_t i=0;i<s;i++)a+=j[i];(void)a;free(j);}

static double bench_ours(r2r_fn f, const double *in, double *out){
  for (int w = 0; w < 8; w++) f(in, out, KB);
  double t0 = now_ns2(); f(in, out, KB); double one = now_ns2() - t0;
  int reps = (int)(3e7 / (one + 1)); if (reps < 20) reps = 20; if (reps > 200000) reps = 200000;
  double best = 1e30;
  for (int t = 0; t < 5; t++) {
    double a = now_ns2();
    for (int i = 0; i < reps; i++) f(in, out, KB);
    double ns = (now_ns2() - a) / reps; if (ns < best) best = ns;
  }
  return best;
}

static double bench_fftw(fftw_plan p){
  for (int w = 0; w < 8; w++) fftw_execute(p);
  double t0 = now_ns2(); fftw_execute(p); double one = now_ns2() - t0;
  int reps = (int)(3e7 / (one + 1)); if (reps < 20) reps = 20; if (reps > 200000) reps = 200000;
  double best = 1e30;
  for (int t = 0; t < 5; t++) {
    double a = now_ns2();
    for (int i = 0; i < reps; i++) fftw_execute(p);
    double ns = (now_ns2() - a) / reps; if (ns < best) best = ns;
  }
  return best;
}

int main(void){
  typedef struct { const char *name; fftw_r2r_kind fk; int N; r2r_fn fn; } cell_t;
  static const cell_t cells[] = {
    {"dct2", FFTW_REDFT10,  8, radix8_dct2_avx512},
    {"dct2", FFTW_REDFT10, 16, radix16_dct2_avx512},
    {"dct2", FFTW_REDFT10, 32, radix32_dct2_avx512},
    {"dct2", FFTW_REDFT10, 64, radix64_dct2_avx512},
    {"dct3", FFTW_REDFT01,  8, radix8_dct3_avx512},
    {"dct3", FFTW_REDFT01, 16, radix16_dct3_avx512},
    {"dct3", FFTW_REDFT01, 32, radix32_dct3_avx512},
    {"dct3", FFTW_REDFT01, 64, radix64_dct3_avx512},
    {"dct4", FFTW_REDFT11,  8, radix8_dct4_avx512},
    {"dct4", FFTW_REDFT11, 16, radix16_dct4_avx512},
    {"dct4", FFTW_REDFT11, 32, radix32_dct4_avx512},
    {"dct4", FFTW_REDFT11, 64, radix64_dct4_avx512},
    {"dht",  FFTW_DHT,      8, radix8_dht_avx512},
    {"dht",  FFTW_DHT,     16, radix16_dht_avx512},
    {"dht",  FFTW_DHT,     32, radix32_dht_avx512},
    {"dht",  FFTW_DHT,     64, radix64_dht_avx512},
    {"dst2", FFTW_RODFT10,  8, radix8_dst2_avx512},
    {"dst2", FFTW_RODFT10, 16, radix16_dst2_avx512},
    {"dst2", FFTW_RODFT10, 32, radix32_dst2_avx512},
    {"dst2", FFTW_RODFT10, 64, radix64_dst2_avx512},
    {"dst3", FFTW_RODFT01,  8, radix8_dst3_avx512},
    {"dst3", FFTW_RODFT01, 16, radix16_dst3_avx512},
    {"dst3", FFTW_RODFT01, 32, radix32_dst3_avx512},
    {"dst3", FFTW_RODFT01, 64, radix64_dst3_avx512},
    {"dst4", FFTW_RODFT11,  8, radix8_dst4_avx512},
    {"dst4", FFTW_RODFT11, 16, radix16_dst4_avx512},
    {"dst4", FFTW_RODFT11, 32, radix32_dst4_avx512},
    {"dst4", FFTW_RODFT11, 64, radix64_dst4_avx512},
    {"dct1", FFTW_REDFT00,  5, radix5_dct1_avx512},
    {"dct1", FFTW_REDFT00,  9, radix9_dct1_avx512},
    {"dct1", FFTW_REDFT00, 17, radix17_dct1_avx512},
    {"dct1", FFTW_REDFT00, 33, radix33_dct1_avx512},
    {"dst1", FFTW_RODFT00,  3, radix3_dst1_avx512},
    {"dst1", FFTW_RODFT00,  7, radix7_dst1_avx512},
    {"dst1", FFTW_RODFT00, 15, radix15_dst1_avx512},
    {"dst1", FFTW_RODFT00, 31, radix31_dst1_avx512},
  };
  const int n_cells = (int)(sizeof(cells)/sizeof(cells[0]));

  static double xin[MAXN*KB], yout[MAXN*KB];
  static double fin[MAXN*KB], fout[MAXN*KB];

  printf("%-5s %-3s %10s %12s %12s %8s %8s %9s\n",
         "kind","N","vfft_ns","fftw_home","fftw_split","home/v","split/v","xcheck");
  for (int ci = 0; ci < n_cells; ci++) {
    {
      int N = cells[ci].N;
      r2r_fn FN = cells[ci].fn;
      size_t total = (size_t)N * KB;
      for (size_t i = 0; i < total; i++) xin[i] = sin(0.37*i) + 0.2*cos(2.1*i);

      /* FFTW home: contiguous transforms. Feed transposed data so the
       * mathematical inputs match ours (lane k's sequence x[0..N-1]). */
      for (int n = 0; n < N; n++)
        for (int k = 0; k < KB; k++)
          fin[(size_t)k*N + n] = xin[(size_t)n*KB + k];
      int n_arr = N;
      fftw_r2r_kind kk = cells[ci].fk;
      fftw_plan ph = fftw_plan_many_r2r(1, &n_arr, KB,
                       fin, NULL, 1, N, fout, NULL, 1, N, &kk, FFTW_PATIENT);
      /* FFTW split: our layout directly. */
      fftw_plan ps = fftw_plan_many_r2r(1, &n_arr, KB,
                       (double*)xin, NULL, KB, 1, fout, NULL, KB, 1, &kk, FFTW_PATIENT);

      /* correctness: ours vs FFTW-split on identical input/layout */
      FN(xin, yout, KB);
      fftw_execute(ps);
      double xc = 0;
      for (size_t i = 0; i < total; i++) {
        double d = fabs(yout[i] - fout[i]); if (d > xc) xc = d;
      }

      cachebust();
      double vns = bench_ours(FN, xin, yout);
      cachebust();
      double hns = bench_fftw(ph);
      cachebust();
      double sns = bench_fftw(ps);

      printf("%-5s %-3d %10.0f %12.0f %12.0f %7.2fx %7.2fx %9.1e %s\n",
             cells[ci].name, N, vns, hns, sns, hns/vns, sns/vns, xc,
             xc < 1e-9 ? "PASS" : "FAIL");
      fftw_destroy_plan(ph); fftw_destroy_plan(ps);
    }
  }
  return 0;
}
