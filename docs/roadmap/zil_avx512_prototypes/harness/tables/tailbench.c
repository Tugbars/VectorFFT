#include <stdio.h>
#include <stdlib.h>
#include <time.h>
typedef void (*fn_t)(const double*, const double*, double*, double*, const double*, const double*, size_t, size_t, size_t, size_t, size_t);
void r8_vex_tail(const double*, const double*, double*, double*, const double*, const double*, size_t, size_t, size_t, size_t, size_t);
void r8_masked_tail(const double*, const double*, double*, double*, const double*, const double*, size_t, size_t, size_t, size_t, size_t);
static double ns(fn_t f, size_t count, double *x, double *y, double *tw) {
    struct timespec a, b; double best = 1e30;
    for (int rep = 0; rep < 7; rep++) {
        clock_gettime(CLOCK_MONOTONIC, &a);
        for (int i = 0; i < 200000; i++) f(x, 0, y, 0, tw, 0, count, 0, count, 0, count);
        clock_gettime(CLOCK_MONOTONIC, &b);
        double t = ((b.tv_sec - a.tv_sec) * 1e9 + (b.tv_nsec - a.tv_nsec)) / 200000.0;
        if (t < best) best = t;
    }
    return best;
}
int main(void) {
    double *x = aligned_alloc(64, 8 * 2 * 8 * 64), *y = aligned_alloc(64, 8 * 2 * 8 * 64), *tw = aligned_alloc(64, 8 * 7 * 16 * 16);
    for (int i = 0; i < 2 * 8 * 64; i++) x[i] = 0.001 * i;
    for (int i = 0; i < 7 * 16 * 16; i++) tw[i] = 0.5;
    printf("count  vex128-tail(ns)  masked-tail(ns)\n");
    for (size_t c = 4; c <= 12; c++) printf("%5zu  %14.1f  %14.1f\n", c, ns(r8_vex_tail, c, x, y, tw), ns(r8_masked_tail, c, x, y, tw));
}
