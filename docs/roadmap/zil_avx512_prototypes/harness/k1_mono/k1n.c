#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
typedef void (*kfn)(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#define D(S) void S(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
D(vfft_k1_mono128_16x8_il_fwd_avx512) D(vfft_k1_mono128_16x8_il_bwd_avx512) D(vfft_k1_mono128_8x16_il_fwd_avx512) D(vfft_k1_mono128_8x16_il_bwd_avx512) D(vfft_k1_mono256_16x16_il_fwd_avx512) D(vfft_k1_mono256_16x16_il_bwd_avx512)
static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static void dft(const double *x, long double *y, int n, int sign) { for (int k = 0; k < n; k++) { long double re = 0, im = 0; for (int j = 0; j < n; j++) { long double a = sign * 2.0L * 3.14159265358979323846264338327950288L * (long double)((long)j * k % n) / n; re += x[2*j] * cosl(a) - x[2*j+1] * sinl(a); im += x[2*j] * sinl(a) + x[2*j+1] * cosl(a); } y[2*k] = re; y[2*k+1] = im; } }
int main(void) {
  struct { const char *n; int N, dir; kfn f; } t[] = { {"128 16x8 fwd",128,0,vfft_k1_mono128_16x8_il_fwd_avx512}, {"128 16x8 bwd",128,1,vfft_k1_mono128_16x8_il_bwd_avx512}, {"128 8x16 fwd",128,0,vfft_k1_mono128_8x16_il_fwd_avx512}, {"128 8x16 bwd",128,1,vfft_k1_mono128_8x16_il_bwd_avx512}, {"256 16x16 fwd",256,0,vfft_k1_mono256_16x16_il_fwd_avx512}, {"256 16x16 bwd",256,1,vfft_k1_mono256_16x16_il_bwd_avx512} };
  double *x = aligned_alloc(64, 4096), *o = aligned_alloc(64, 4096); long double r[512]; srand(3); for (int i = 0; i < 512; i++) x[i] = rand() / (double)RAND_MAX - .5;
  for (int i = 0; i < 6; i++) { int N = t[i].N; dft(x, r, N, t[i].dir ? 1 : -1); t[i].f(x,0,o,0,0,0,0,0,0,0,0); long double e = 0, m = 0; for (int j = 0; j < 2*N; j++) { if (fabsl(o[j]-r[j]) > e) e = fabsl(o[j]-r[j]); if (fabsl(r[j]) > m) m = fabsl(r[j]); }
    double best = 1e30; for (int q = 0; q < 9; q++) { double t0 = now_ns(); for (int k = 0; k < 50000; k++) { t[i].f(x,0,o,0,0,0,0,0,0,0,0); __asm__ volatile("":::"memory"); } double dt = (now_ns()-t0)/50000; if (dt < best) best = dt; }
    printf("mono %-14s rel err %.2e   %.1f ns/call\n", t[i].n, (double)(e/m), best); }
  return 0; }
