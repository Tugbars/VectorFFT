/* t2cp broadcast records: per (block, leg) one record [c x 2cpv][(-s,+s) x cpv].
 * il_flatdit.h / il2d_cols.h build it at cpv=2 (8 doubles). */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
extern void radix5_z_t2cp_fwd_avx512(const double*, const double*, double*, double*, const double*, const double*, size_t, size_t, size_t, size_t, size_t);
static double run(int count, int cpv) {
    const int R = 5, Lmod = 7 * R; const long Q = 3;
    size_t rec = 4u * cpv;
    double *x = malloc(16 * R * (count + 4)), *y = malloc(16 * R * (count + 4));
    double *tw = calloc((R - 1) * rec, 8);
    for (int l = 1; l < R; l++) {
        double a = 2 * M_PI * (double)(l * Q % Lmod) / Lmod, c = cos(a), sn = -sin(a);
        double *r = tw + (l - 1) * rec;
        for (int j = 0; j < 2 * cpv; j++) { r[j] = c; r[2 * cpv + j] = (j & 1) ? sn : -sn; }
    }
    srand(5);
    for (int i = 0; i < 2 * R * count; i++) x[i] = rand() / (double)RAND_MAX - 0.5;
    radix5_z_t2cp_fwd_avx512(x, 0, y, 0, tw, 0, count, count * R, count, 1, count);
    double err = 0;
    for (int k = 0; k < count; k++)
        for (int lp = 0; lp < R; lp++) {
            double complex acc = 0;
            for (int l = 0; l < R; l++)
                acc += (x[2 * (l * count + k)] + I * x[2 * (l * count + k) + 1]) *
                       cexp(-2 * M_PI * I * (double)(l * Q % Lmod) / Lmod) * cexp(-2 * M_PI * I * (double)(l * lp % R) / R);
            double e = cabs(y[2 * (lp * count + k)] + I * y[2 * (lp * count + k) + 1] - acc);
            if (e > err) err = e;
        }
    free(x); free(y); free(tw);
    return err;
}
int main(void) {
    printf("count :"); for (int c = 1; c <= 9; c++) printf(" %8d", c);
    printf("\ncpv=2 :"); for (int c = 1; c <= 9; c++) printf(" %8.1e", run(c, 2));
    printf("\ncpv=4 :"); for (int c = 1; c <= 9; c++) printf(" %8.1e", run(c, 4));
    printf("\n");
}
