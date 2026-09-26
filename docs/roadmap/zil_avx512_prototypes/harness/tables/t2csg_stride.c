/* Column-stride kinds (t2csg): lane j of a vector must come from block k+j at
 * stride Gs (AVX2 does it with loadu2_m128d pairs). Identity twiddles, so the
 * expected result is a plain radix-R DFT down each block's legs. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
extern void radix5_z_t2csg_fwd_avx512(const double*, const double*, double*, double*, const double*, const double*, size_t, size_t, size_t, size_t, size_t);
int main(void) {
    const int R = 5;
    for (int count = 1; count <= 9; count++) {
        const size_t D = 1, Ls = D, Gs = R * D, OLs = D, OGs = R * D;
        size_t n = 2 * (Gs * count + 8);
        double *x = calloc(n, 8), *y = calloc(n, 8);
        double *t1 = calloc(16 * 4, 8), *t2 = calloc(16, 8);
        for (int i = 0; i < 4; i++) for (int j = 0; j < 8; j++) t1[16 * i + j] = 1.0;
        for (int j = 0; j < 8; j++) t2[j] = 1.0;
        srand(3);
        for (size_t i = 0; i < n; i++) x[i] = rand() / (double)RAND_MAX - 0.5;
        radix5_z_t2csg_fwd_avx512(x, 0, y, 0, t1, t2, Ls, Gs, OLs, OGs, (size_t)count);
        double err = 0;
        for (int k = 0; k < count; k++)
            for (int lp = 0; lp < R; lp++) {
                double complex acc = 0;
                for (int l = 0; l < R; l++) {
                    size_t o = 2 * (l * Ls + k * Gs);
                    acc += (x[o] + I * x[o + 1]) * cexp(-2 * M_PI * I * (double)(l * lp % R) / R);
                }
                size_t o = 2 * (lp * OLs + k * OGs);
                double e = cabs(y[o] + I * y[o + 1] - acc);
                if (e > err) err = e;
            }
        printf("count=%d  max|err|=%.2e\n", count, err);
        free(x); free(y); free(t1); free(t2);
    }
    return 0;
}
