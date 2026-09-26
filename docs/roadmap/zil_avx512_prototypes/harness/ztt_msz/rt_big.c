/* large cells: roundtrip bwd(fwd(x)) == N*x and a spot DFT check of 64 bins (long double) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "il_reg_kinds.h"
#include "ztt_vw.h"
int main(void)
{
    int i, n = 0, bad = 0; double worst = 0, worsts = 0;
    for (i = 0; i < VFFT_ZTT_NCELLS; i++) {
        const vfft_ztt_cell_t *c = &vfft_ztt_cells[i]; const int N = c->n; int scr;
        if (N < 16384) continue;
        for (scr = 0; scr < 2; scr++) {
            vfft_ztt_plan_t *p = _ztt_create(N, c->chain, c->nf, scr, 0); long k, j; int b;
            double *x, *y, *z, e = 0, m = 0, es = 0;
            if (!p) continue;
            x = aligned_alloc(64, 16 * N); y = aligned_alloc(64, 16 * N); z = aligned_alloc(64, 16 * N);
            for (j = 0; j < 2 * N; j++) x[j] = sin(0.7 * j + N) + 0.1 * cos(3.1 * j);
            vfft_ztt_bind(p, 1);
            vfft_ztt_execute_fwd(p, x, y);
            for (b = 0; b < 64; b++) {   /* spot bins */
                const long kk = (b * 7919L + 13) % N; long double re = 0, im = 0; const size_t q = vfft_ztt_perm(p, kk);
                for (j = 0; j < N; j++) { long double a = -2.0L * 3.14159265358979323846264338327950288L * (long double)((j * kk) % N) / N;
                    re += x[2*j] * cosl(a) - x[2*j+1] * sinl(a); im += x[2*j] * sinl(a) + x[2*j+1] * cosl(a); }
                if (fabsl(re - y[2*q]) / sqrt(N) > es) es = fabsl(re - y[2*q]) / sqrt(N);
                if (fabsl(im - y[2*q+1]) / sqrt(N) > es) es = fabsl(im - y[2*q+1]) / sqrt(N);
            }
            vfft_ztt_execute_bwd(p, y, z);
            for (j = 0; j < 2 * N; j++) { if (fabs(z[j] / N - x[j]) > e) e = fabs(z[j] / N - x[j]); if (fabs(x[j]) > m) m = fabs(x[j]); }
            e /= m; n++; if (e > worst) worst = e; if (es > worsts) worsts = es;
            if (e > 1e-13 || es > 1e-13) { bad++; printf("BAD N=%d scr=%d rt=%g spot=%g\n", N, scr, e, es); }
            vfft_ztt_destroy(p); free(x); free(y); free(z);
        }
    }
    printf("large fused plans (N >= 16384, in place): %d, bad %d, worst roundtrip rel err %.3g, worst spot-bin err/sqrtN %.3g\n", n, bad, worst, worsts);
    return bad;
}
