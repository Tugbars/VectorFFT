/* r1c_gate.c -- the real flat leaf (radixR_z_r1c_{fwd,bwd}_avx2, the real
 * flat DIT's one kind) against a naive digit transform, at every odd radix
 * of the kind and at counts that reach the wide loop, the two-column step
 * and the lone last column (1, 2, 3, 4, 5, 7, 9, 15).
 *   fwd: R real legs (leg l of column k at zin[l*D + k]) -> digit 0 real at
 *        zout[k], digit p in 1..(R-1)/2 complex at zout[2*p*D + 2*k]:
 *        digit p = sum_l x[l*D + k] * exp(-2 pi i l p / R).
 *   bwd: the digit runs -> R real legs, R times the input.
 * The second half of block 0 (doubles D..2D) is never written: checked.
 * The kernels link as their own objects (each file declares its own static
 * constants).
 * Build: gcc -O2 -mavx2 -mfma gauntlet/r1c_gate.c src/dag-fft-compiler/codelets/zil/avx2/real/flat/radix*_z_r1c*_avx2.c -o r1c_gate */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>

typedef void (*fn11)(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#define RADICES(X) X(3) X(5) X(7) X(9) X(11) X(13) X(15) X(17) X(19) X(21) X(23) X(25) X(27) X(29) X(31) X(37) X(41) X(43) X(47)
#define DECL(n) \
    void radix##n##_z_r1c_fwd_avx2(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
    void radix##n##_z_r1c_bwd_avx2(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
RADICES(DECL)
#define ROW(n) { n, radix##n##_z_r1c_fwd_avx2, radix##n##_z_r1c_bwd_avx2 },
static const struct { int R; fn11 fwd, bwd; } K[] = { RADICES(ROW) };

int main(void)
{
    static const int counts[] = { 1, 2, 3, 4, 5, 7, 9, 15 };
    const double PI = 3.14159265358979323846;
    const double GUARD = 12345.0;
    int fails = 0;
    unsigned seed = 0x5eedu;
    for (int ki = 0; ki < (int)(sizeof K / sizeof K[0]); ki++)
    {
        const int R = K[ki].R, h = R / 2;
        double worst_f = 0, worst_b = 0;
        for (int ci = 0; ci < (int)(sizeof counts / sizeof counts[0]); ci++)
        {
            const size_t D = (size_t)counts[ci], N = (size_t)R * D, P = ((size_t)R + 1) * D;
            double *x = (double *)_aligned_malloc((N + 8) * 8, 64), *y = (double *)_aligned_malloc((N + 8) * 8, 64);
            double *pl = (double *)_aligned_malloc((P + 8) * 8, 64);
            for (size_t i = 0; i < N; i++) { seed = seed * 1664525u + 1013904223u; x[i] = (double)(seed >> 8) / (double)(1u << 24) - 0.5; }
            for (size_t i = 0; i < P + 8; i++) pl[i] = GUARD;
            K[ki].fwd(x, NULL, pl, NULL, NULL, NULL, D, 0, D, 0, D);
            for (size_t k = 0; k < D; k++)
                for (int p = 0; p <= h; p++)
                {
                    long double re = 0, im = 0;
                    for (int l = 0; l < R; l++)
                    {
                        const long double a = -2.0L * (long double)PI * (long double)((l * p) % R) / (long double)R;
                        re += (long double)x[(size_t)l * D + k] * cosl(a);
                        im += (long double)x[(size_t)l * D + k] * sinl(a);
                    }
                    const double gr = p ? pl[2 * (size_t)p * D + 2 * k] : pl[k];
                    const double gi = p ? pl[2 * (size_t)p * D + 2 * k + 1] : 0.0;
                    const double e = fabs((double)(re - gr)) + fabs((double)(im - gi));
                    if (e > worst_f) worst_f = e;
                }
            for (size_t i = D; i < 2 * D; i++) if (pl[i] != GUARD) { printf("R=%d count=%zu: fwd wrote the unused half of block 0\n", R, D); fails++; break; }
            for (size_t i = P; i < P + 8; i++) if (pl[i] != GUARD) { printf("R=%d count=%zu: fwd wrote past the plane\n", R, D); fails++; break; }
            for (size_t i = 0; i < N + 8; i++) y[i] = GUARD;
            K[ki].bwd(pl, NULL, y, NULL, NULL, NULL, D, 0, D, 0, D);
            for (size_t i = 0; i < N; i++) { const double e = fabs(y[i] - (double)R * x[i]) / (double)R; if (e > worst_b) worst_b = e; }
            for (size_t i = N; i < N + 8; i++) if (y[i] != GUARD) { printf("R=%d count=%zu: bwd wrote past the legs\n", R, D); fails++; break; }
            _aligned_free(x); _aligned_free(y); _aligned_free(pl);
        }
        const int ok = worst_f < 1e-13 && worst_b < 1e-13;
        if (!ok) fails++;
        printf("R=%-2d fwd max err %.1e  roundtrip max err %.1e  %s\n", R, worst_f, worst_b, ok ? "ok" : "*** FAIL ***");
    }
    printf("%s\n", fails ? "FAILURES" : "ALL PASS");
    return fails ? 1 : 0;
}
