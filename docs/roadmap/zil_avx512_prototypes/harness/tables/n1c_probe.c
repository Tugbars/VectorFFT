#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
typedef void (*fn_t)(const double*, const double*, double*, double*, const double*, const double*, size_t, size_t, size_t, size_t, size_t);
extern void radix8_z_n1c_fwd_avx512(const double*, const double*, double*, double*, const double*, const double*, size_t, size_t, size_t, size_t, size_t);
extern void radix5_z_n1c_fwd_avx512(const double*, const double*, double*, double*, const double*, const double*, size_t, size_t, size_t, size_t, size_t);
extern void radix16_z_n1_fwd_avx512(const double*, const double*, double*, double*, const double*, const double*, size_t, size_t, size_t, size_t, size_t);
static double run(fn_t f, int R, int count, int inplace) {
    double *x = malloc(16 * R * (count + 4)), *y = malloc(16 * R * (count + 4)), *x0 = malloc(16 * R * (count + 4));
    srand(9);
    for (int i = 0; i < 2 * R * count; i++) x0[i] = x[i] = rand() / (double)RAND_MAX - 0.5;
    double *o = inplace ? x : y;
    f(x, 0, o, 0, 0, 0, count, 0, count, 0, count);
    double err = 0;
    for (int k = 0; k < count; k++)
        for (int lp = 0; lp < R; lp++) {
            double complex acc = 0;
            for (int l = 0; l < R; l++)
                acc += (x0[2 * (l * count + k)] + I * x0[2 * (l * count + k) + 1]) * cexp(-2 * M_PI * I * (double)(l * lp % R) / R);
            double e = cabs(o[2 * (lp * count + k)] + I * o[2 * (lp * count + k) + 1] - acc);
            if (e > err) err = e;
        }
    free(x); free(y); free(x0);
    return err;
}
int main(void) {
    struct { const char *n; fn_t f; int R, ip; } K[] = {
        {"radix8_z_n1c_fwd_avx512 (in place)", radix8_z_n1c_fwd_avx512, 8, 1},
        {"radix5_z_n1c_fwd_avx512 (in place)", radix5_z_n1c_fwd_avx512, 5, 1},
        {"radix16_z_n1_fwd_avx512 (oop)", radix16_z_n1_fwd_avx512, 16, 0}};
    for (int i = 0; i < 3; i++) {
        printf("%-36s", K[i].n);
        for (int c = 1; c <= 9; c++) printf(" %8.1e", run(K[i].f, K[i].R, c, K[i].ip));
        printf("\n");
    }
}
