/* test_zil_n1.c — checklist item 2 gate: emitted z-native n1 leaves
 * (radix 4/8/16/32/64) vs naive R-point DFT, point-major z batch, K=8.
 * (radix-8 additionally holds bit-identity vs the hand oracle in
 * il_r8_m1_race.c; this gate covers the recursive-builder radices.)
 *
 * Build: python build.py --src test/test_zil_n1.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef void (*zn1_fn)(const double *, const double *, double *, double *,
                       const double *, const double *,
                       unsigned long long, unsigned long long,
                       unsigned long long, unsigned long long,
                       unsigned long long);
extern void radix4_z_n1_fwd_avx2(const double *, const double *, double *, double *,
    const double *, const double *, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
extern void radix8_z_n1_fwd_avx2(const double *, const double *, double *, double *,
    const double *, const double *, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
extern void radix16_z_n1_fwd_avx2(const double *, const double *, double *, double *,
    const double *, const double *, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
extern void radix32_z_n1_fwd_avx2(const double *, const double *, double *, double *,
    const double *, const double *, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
extern void radix64_z_n1_fwd_avx2(const double *, const double *, double *, double *,
    const double *, const double *, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);

int main(void)
{
    struct { int R; zn1_fn f; } cs[] = {
        { 4, radix4_z_n1_fwd_avx2 }, { 8, radix8_z_n1_fwd_avx2 },
        { 16, radix16_z_n1_fwd_avx2 }, { 32, radix32_z_n1_fwd_avx2 },
        { 64, radix64_z_n1_fwd_avx2 },
    };
    int fails = 0;
    const int K = 8;
    for (int ci = 0; ci < 5; ci++) {
        int R = cs[ci].R;
        size_t S = (size_t)2 * K;
        double *zin = (double *)malloc(R * S * 8), *zout = (double *)malloc(R * S * 8);
        srand(42 + R);
        for (size_t i = 0; i < (size_t)R * S; i++)
            zin[i] = (double)rand() / RAND_MAX - 0.5;
        cs[ci].f(zin, 0, zout, 0, 0, 0, K, 0, K, 0, K);
        double err = 0;
        for (int k = 0; k < K; k++)
            for (int m = 0; m < R; m++) {
                double sr = 0, si = 0;
                for (int n = 0; n < R; n++) {
                    double a = -2.0 * M_PI * (double)((n * m) % R) / (double)R;
                    double c = cos(a), s = sin(a);
                    double re = zin[(size_t)n * S + 2 * k], im = zin[(size_t)n * S + 2 * k + 1];
                    sr += re * c - im * s;
                    si += re * s + im * c;
                }
                double d = fabs(zout[(size_t)m * S + 2 * k] - sr)
                         + fabs(zout[(size_t)m * S + 2 * k + 1] - si);
                if (d > err) err = d;
            }
        double tol = 1e-12 * R;
        const char *bad = (err > tol || err != err) ? "  <FAIL>" : "";
        if (bad[0]) fails++;
        printf("  z-n1 R=%-3d vs naive = %.2e%s\n", R, err, bad);
        free(zin); free(zout);
    }

    /* ---- t2 kernels: streamed VTW2 twiddles (cos-first, sign-folded,
     * consumption order) applied to legs>=1, then the R-DFT ---- */
    {
        extern void radix8_z_t2_fwd_avx2(const double *, const double *, double *, double *,
            const double *, const double *, unsigned long long, unsigned long long,
            unsigned long long, unsigned long long, unsigned long long);
        extern void radix16_z_t2_fwd_avx2(const double *, const double *, double *, double *,
            const double *, const double *, unsigned long long, unsigned long long,
            unsigned long long, unsigned long long, unsigned long long);
        extern void radix32_z_t2_fwd_avx2(const double *, const double *, double *, double *,
            const double *, const double *, unsigned long long, unsigned long long,
            unsigned long long, unsigned long long, unsigned long long);
        struct { int R; zn1_fn f; } ts[] = {
            { 8, radix8_z_t2_fwd_avx2 },
            { 16, radix16_z_t2_fwd_avx2 },
            { 32, radix32_z_t2_fwd_avx2 },
        };
        for (int ci = 0; ci < 3; ci++) {
            int R = ts[ci].R;
            size_t S = (size_t)2 * K;
            int N = R * K;   /* four-step-style twiddles W_N^(l*k) */
            double *zin = (double *)malloc(R * S * 8), *zout = (double *)malloc(R * S * 8);
            double *tw = (double *)malloc((size_t)(K / 2) * (R - 1) * 8 * 8);
            srand(142 + R);
            for (size_t i = 0; i < (size_t)R * S; i++)
                zin[i] = (double)rand() / RAND_MAX - 0.5;
            /* VTW2 fill: per column-pair p, per leg l, cos-dup then sign-folded sin */
            for (int p = 0; p < K / 2; p++)
                for (int l = 1; l < R; l++) {
                    double *rec = tw + ((size_t)p * (R - 1) + (l - 1)) * 8;
                    for (int j = 0; j < 2; j++) {
                        int k = 2 * p + j;
                        double a = -2.0 * M_PI * (double)(l * k) / (double)N;
                        rec[2 * j] = cos(a); rec[2 * j + 1] = cos(a);
                        rec[4 + 2 * j] = -sin(a); rec[4 + 2 * j + 1] = sin(a);
                    }
                }
            ts[ci].f(zin, 0, zout, 0, tw, 0, K, 0, K, 0, K);
            double err = 0;
            for (int k = 0; k < K; k++)
                for (int m = 0; m < R; m++) {
                    double sr = 0, si = 0;
                    for (int n = 0; n < R; n++) {
                        double re = zin[(size_t)n * S + 2 * k], im = zin[(size_t)n * S + 2 * k + 1];
                        if (n > 0) {   /* twiddle leg n by W_N^(n*k) */
                            double a = -2.0 * M_PI * (double)(n * k) / (double)N;
                            double c = cos(a), s = sin(a);
                            double tr = re * c - im * s, ti = re * s + im * c;
                            re = tr; im = ti;
                        }
                        double a2 = -2.0 * M_PI * (double)((n * m) % R) / (double)R;
                        double c2 = cos(a2), s2 = sin(a2);
                        sr += re * c2 - im * s2;
                        si += re * s2 + im * c2;
                    }
                    double d = fabs(zout[(size_t)m * S + 2 * k] - sr)
                             + fabs(zout[(size_t)m * S + 2 * k + 1] - si);
                    if (d > err) err = d;
                }
            double tol = 1e-12 * R;
            const char *bad = (err > tol || err != err) ? "  <FAIL>" : "";
            if (bad[0]) fails++;
            printf("  z-t2 R=%-3d (VTW2 stream) vs naive = %.2e%s\n", R, err, bad);
            free(zin); free(zout); free(tw);
        }
    }
    printf("%s (%d fail)\n", fails ? "FAILURES" : "Z FAMILY (n1+t2): ALL GATES GREEN", fails);
    return fails ? 1 : 0;
}
