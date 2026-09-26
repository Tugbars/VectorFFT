/* k1test.c — the avx512 prototype of vfft_k1_mono64_8x8_il_{fwd,bwd}: correctness vs a
 * long-double DFT, bitwise vs the avx2 original, and ns/call vs the avx2 mono and the
 * form-0 solo n1 kernel (radix64_z_n1 at count = 1, the whole transform in the tail).
 * INDICATIVE timing (VM, 4 vCPU). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

typedef void (*kfn)(const double *, const double *, double *, double *,
                    const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#define D(S) void S(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
D(vfft_k1_mono64_8x8_il_fwd_avx2) D(vfft_k1_mono64_8x8_il_bwd_avx2)
D(vfft_k1_mono64_8x8_il_fwd_avx512) D(vfft_k1_mono64_8x8_il_bwd_avx512)
D(radix64_z_n1_fwd_avx2) D(L_radix64_z_n1_fwd_avx512) D(M_radix64_z_n1_fwd_avx512)

static double now_ns(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }

static void dft(const double *x, long double *y, int n, int sign)
{
    for (int k = 0; k < n; k++) {
        long double re = 0, im = 0;
        for (int j = 0; j < n; j++) {
            long double a = sign * 2.0L * 3.14159265358979323846264338327950288L * (long double)((long)j * k % n) / n;
            re += x[2*j] * cosl(a) - x[2*j+1] * sinl(a);
            im += x[2*j] * sinl(a) + x[2*j+1] * cosl(a);
        }
        y[2*k] = re; y[2*k+1] = im;
    }
}
static double err(const double *o, const long double *r, int n)
{ long double e = 0, m = 0; for (int i = 0; i < 2*n; i++) { long double d = fabsl(o[i] - r[i]); if (d > e) e = d; if (fabsl(r[i]) > m) m = fabsl(r[i]); } return (double)(e / m); }

static double bench(kfn f, const double *in, double *out, size_t Ls)
{
    double best = 1e30;
    for (int t = 0; t < 15; t++) {
        double t0 = now_ns();
        for (int i = 0; i < 100000; i++) { f(in, 0, out, 0, 0, 0, Ls, 0, Ls, 0, 1); __asm__ volatile("" ::: "memory"); }
        double dt = (now_ns() - t0) / 100000; if (dt < best) best = dt;
    }
    return best;
}

int main(void)
{
    const int N = 64;
    double *x = aligned_alloc(64, 2*N*8), *o2 = aligned_alloc(64, 2*N*8), *o5 = aligned_alloc(64, 2*N*8);
    long double ref[2*64];
    srand(7); for (int i = 0; i < 2*N; i++) x[i] = rand() / (double)RAND_MAX - 0.5;
    int bad = 0;
    for (int dir = 0; dir < 2; dir++) {
        dft(x, ref, N, dir ? +1 : -1);
        kfn f2 = dir ? vfft_k1_mono64_8x8_il_bwd_avx2 : vfft_k1_mono64_8x8_il_fwd_avx2;
        kfn f5 = dir ? vfft_k1_mono64_8x8_il_bwd_avx512 : vfft_k1_mono64_8x8_il_fwd_avx512;
        f2(x, 0, o2, 0, 0, 0, 0, 0, 0, 0, 0); f5(x, 0, o5, 0, 0, 0, 0, 0, 0, 0, 0);
        double e2 = err(o2, ref, N), e5 = err(o5, ref, N);
        size_t nd = 0; for (int i = 0; i < 2*N; i++) if (memcmp(&o2[i], &o5[i], 8)) nd++;
        printf("mono64 8x8 %s: rel err vs long-double DFT  avx2 %.2e  avx512 %.2e ; avx512 vs avx2: %zu/%d doubles differ\n",
               dir ? "bwd" : "fwd", e2, e5, nd, 2*N);
        if (e5 > 1e-14) bad = 1;
    }
    printf("\nns/call, N=64 K=1 interleaved forward (whole transform, L1):\n");
    printf("  mono64 8x8 avx2 (form 1)                 %6.1f\n", bench(vfft_k1_mono64_8x8_il_fwd_avx2, x, o2, 0));
    printf("  mono64 8x8 avx512 (form 1, prototype)    %6.1f\n", bench(vfft_k1_mono64_8x8_il_fwd_avx512, x, o5, 0));
    printf("  radix64_z_n1 avx2  count=1 (form 0)      %6.1f\n", bench(radix64_z_n1_fwd_avx2, x, o2, 1));
    printf("  radix64_z_n1 avx512 ladder-tail count=1  %6.1f\n", bench(L_radix64_z_n1_fwd_avx512, x, o5, 1));
    printf("  radix64_z_n1 avx512 masked-tail count=1  %6.1f\n", bench(M_radix64_z_n1_fwd_avx512, x, o5, 1));
    return bad;
}
