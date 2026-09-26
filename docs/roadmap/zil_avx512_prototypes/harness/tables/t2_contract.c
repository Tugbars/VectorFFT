/* VTW2 contract probe: il2p's t2(R) mid, count columns, Ls=OLs=count.
 * Reference: X[lp][k] = sum_l x[l][k] * w_N^{l*k} * w_R^{l*lp}, N = R*count.
 * Table layouts:
 *   cpv=2 : today's il2p.h builder (8 doubles per (pair, leg))
 *   cpv=4 : the same record generalized to 4 complex per vector (16 doubles) */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
typedef void (*fn_t)(const double*, const double*, double*, double*, const double*, const double*,
                     size_t, size_t, size_t, size_t, size_t);
extern void radix4_z_t2_fwd_avx512(const double*, const double*, double*, double*, const double*, const double*, size_t, size_t, size_t, size_t, size_t);
extern void radix8_z_t2_fwd_avx512(const double*, const double*, double*, double*, const double*, const double*, size_t, size_t, size_t, size_t, size_t);
extern void radix5_z_t2_fwd_avx512(const double*, const double*, double*, double*, const double*, const double*, size_t, size_t, size_t, size_t, size_t);

static double *build_vtw2(int R, int cols, int N, int cpv) {
    size_t ngrp = ((size_t)cols + cpv - 1) / cpv, rec = 4u * cpv;
    double *tw = aligned_alloc(64, ((ngrp * (R - 1) * rec * 8 + 63) / 64) * 64);
    for (size_t g = 0; g < ngrp; g++)
        for (int l = 1; l < R; l++) {
            double *r = tw + (g * (R - 1) + (l - 1)) * rec;
            for (int j = 0; j < cpv; j++) {
                long k = (long)g * cpv + j;
                double a = 2 * M_PI * (double)((long)l * k % N) / N, c = cos(a), s = sin(a);
                /* il2p.h: s_used = -sin(2pi lk/N); rf[4+2j] = -s_used; rf[4+2j+1] = s_used */
                double su = -s;
                r[2 * j] = c; r[2 * j + 1] = c;
                r[2 * cpv + 2 * j] = -su; r[2 * cpv + 2 * j + 1] = su;
            }
        }
    return tw;
}
static double run(fn_t f, int R, int count, int cpv) {
    int N = R * count;
    double *x = malloc(sizeof(double) * 2 * N), *y = malloc(sizeof(double) * 2 * N);
    srand(7);
    for (int i = 0; i < 2 * N; i++) x[i] = rand() / (double)RAND_MAX - 0.5;
    double *tw = build_vtw2(R, count, N, cpv);
    f(x, 0, y, 0, tw, 0, count, 0, count, 0, count);
    double err = 0, mag = 0;
    for (int lp = 0; lp < R; lp++)
        for (int k = 0; k < count; k++) {
            double complex acc = 0;
            for (int l = 0; l < R; l++) {
                double complex xv = x[2 * (l * count + k)] + I * x[2 * (l * count + k) + 1];
                acc += xv * cexp(-2 * M_PI * I * (double)((long)l * k % N) / N) * cexp(-2 * M_PI * I * (double)(l * lp % R) / R);
            }
            double complex yv = y[2 * (lp * count + k)] + I * y[2 * (lp * count + k) + 1];
            double e = cabs(yv - acc);
            if (e > err) err = e;
            if (cabs(acc) > mag) mag = cabs(acc);
        }
    free(x); free(y); free(tw);
    return err / mag;
}
int main(void) {
    struct { const char *n; fn_t f; int R; } K[] = {
        {"radix4_z_t2_fwd_avx512", radix4_z_t2_fwd_avx512, 4},
        {"radix5_z_t2_fwd_avx512", radix5_z_t2_fwd_avx512, 5},
        {"radix8_z_t2_fwd_avx512", radix8_z_t2_fwd_avx512, 8}};
    for (int i = 0; i < 3; i++) {
        printf("%s\n  count :", K[i].n);
        for (int c = 1; c <= 12; c++) printf(" %8d", c);
        printf("\n  cpv=2 :");
        for (int c = 1; c <= 12; c++) printf(" %8.1e", run(K[i].f, K[i].R, c, 2));
        printf("\n  cpv=4 :");
        for (int c = 1; c <= 12; c++) printf(" %8.1e", run(K[i].f, K[i].R, c, 4));
        printf("\n");
    }
    return 0;
}
