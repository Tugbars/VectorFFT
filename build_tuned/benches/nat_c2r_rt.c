/* nat_c2r_rt.c — roundtrip GATE for the split-input natural c2r.
 *
 *   x (real) --rfft_execute_fwd_natural--> (out_re,out_im) split half-spectrum
 *            --c2r_execute_natural--------> y
 *   expect y == N*x  (unnormalized real-FFT roundtrip).
 *
 * Validates the backward-natural codelet family + c2r_execute_natural's stage-0
 * initiator (interior hc2c_nat_bwd, DC/Nyquist gather+r2cb, split mid inverse)
 * across factorizations with/without a mid column. No MKL.
 *
 * Build (Windows): cd build_tuned && python build.py --src benches/nat_c2r_rt.c --compile
 */
#define VFFT_RFFT_MAX_RADIX 32
#define VFFT_RFFT_RANGED 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "executor.h"
#include "planner.h"
#include "proto_stride_compat.h"
#include "env.h"
#include "rfft.h"
#include "c2r.h"
#include "rfft_registry_avx2.h"
#include "c2r_registry_avx2.h"

int main(void)
{
    stride_env_init();
    rfft_codelets_t reg;
    memset(&reg, 0, sizeof reg);
    rfft_register_all_avx2(&reg); /* fwd: r2cf + hc2hc_dit + hc2c_nat (fwd) */
    c2r_register_all_avx2(&reg);  /* bwd: r2cb + hc2hc_dif_bwd + hc2c_bwd (nat init) */

    struct { int N; int f[4]; int nf; } cells[] = {
        { 64,  { 8, 8 },        2 },   /* m even at st0 -> exercises mid */
        { 24,  { 8, 3 },        2 },   /* m odd -> no mid */
        { 256, { 4, 8, 8 },     3 },
        { 256, { 16, 16 },      2 },
        { 512, { 8, 8, 8 },     3 },
        { 1024,{ 4, 16, 16 },   3 },
    };
    size_t Ks[] = { 8, 16, 64, 256 };
    int ncell = (int)(sizeof cells / sizeof cells[0]);
    int nK = (int)(sizeof Ks / sizeof Ks[0]);
    int fails = 0;

    for (int ci = 0; ci < ncell; ci++) {
        int N = cells[ci].N, nf = cells[ci].nf;
        for (int kj = 0; kj < nK; kj++) {
            size_t K = Ks[kj];
            rfft_plan_t *rp = rfft_plan_create(N, K, cells[ci].f, nf, &reg);
            c2r_plan_t  *cp = c2r_plan_create(N, K, cells[ci].f, nf, &reg);
            if (!rp || !cp) { printf("N=%-5d K=%-4zu plan NULL\n", N, K); fails++; continue; }
            if (!cp->nat_init) {
                printf("N=%-5d K=%-4zu NO nat_init (st0 radix=%d)\n",
                       N, K, cp->base->st[0].radix);
                fails++; rfft_plan_destroy(rp); c2r_plan_destroy(cp); continue;
            }
            size_t NK = (size_t)N * K;
            double *x  = (double*)malloc(NK * 8);
            double *ore= (double*)malloc(NK * 8);   /* (N/2+1)*K used; NK is safe */
            double *oim= (double*)malloc(NK * 8);
            double *y  = (double*)malloc(NK * 8);
            srand(1234 + ci * 17 + kj);
            for (size_t i = 0; i < NK; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
            memset(ore, 0, NK * 8); memset(oim, 0, NK * 8); memset(y, 0, NK * 8);

            rfft_execute_fwd_natural(rp, x, ore, oim);
            c2r_execute_natural(cp, ore, oim, y);

            /* RANGE-DECOMPOSITION check (the MT building block): cover K with disjoint
             * lane slabs via c2r_execute_natural_range; must be bit-identical to the
             * full folded c2r_execute_natural. If so, the pool K-split MT is correct by
             * construction (disjoint lanes, lane-indexed scratch). */
            double *yr = (double *)malloc(NK * 8);
            memset(yr, 0, NK * 8);
            size_t S = ((K / 3) + 7) & ~(size_t)7; /* uneven slab to stress edges */
            if (S == 0) S = 8;
            for (size_t k0 = 0; k0 < K; k0 += S) {
                size_t kw = (k0 + S <= K) ? S : (K - k0);
                c2r_execute_natural_range(cp, ore, oim, yr, k0, kw);
            }
            double rng_max = 0.0;
            for (size_t i = 0; i < NK; i++) { double e = fabs(yr[i] - y[i]); if (e > rng_max) rng_max = e; }
            free(yr);

            /* standard FFT roundtrip metric: max abs error normalized by the
             * signal scale (max|ref|), not per-element (which divides by ~0
             * where x[i]~0 and spuriously inflates). */
            double maxe = 0.0, maxref = 0.0;
            for (size_t i = 0; i < NK; i++) {
                double ref = (double)N * x[i];
                double e = fabs(y[i] - ref);
                if (e > maxe) maxe = e;
                if (fabs(ref) > maxref) maxref = fabs(ref);
            }
            double maxrel = maxe / (maxref + 1e-30);
            const char *fac =
                nf == 2 ? "2-stage" : (nf == 3 ? "3-stage" : "n");
            int ok = (maxrel < 1e-10) && (rng_max == 0.0);
            printf("N=%-5d K=%-4zu nf=%d st0r=%-2d  rel=%.2e  range_d=%.0e  %s\n",
                   N, K, nf, cp->base->st[0].radix, maxrel, rng_max,
                   ok ? "OK" : "FAIL");
            if (!ok) fails++;
            free(x); free(ore); free(oim); free(y);
            rfft_plan_destroy(rp); c2r_plan_destroy(cp);
            (void)fac;
        }
    }
    printf(fails ? "\n%d FAILURES\n" : "\nALL ROUNDTRIPS OK\n", fails);
    return fails ? 1 : 0;
}
