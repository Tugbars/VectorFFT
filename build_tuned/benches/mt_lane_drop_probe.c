/* mt_lane_drop_probe.c — do the threaded real batch paths WRITE EVERY LANE?
 *
 * THE SUSPICION. Several MT slab sizers in this tree compute
 *     size_t S = ((K / (size_t)T) + 7) & ~(size_t)7;      <-- FLOOR
 * and then cover [0, min(S,K)) on the caller plus [t*S, min(t*S+S, K)) on each
 * worker. Total coverage is therefore [0, min(T*S, K)). When T*S < K the top
 * lanes are handed to NOBODY: never transformed, never written, no warning and
 * no return code -- the caller's output plane simply keeps whatever was in it.
 *
 * The transform-contiguous wrapper and _il_mt both use CEIL instead
 *     size_t S = (((K + T - 1) / T) + 7) & ~(size_t)7;    <-- CEIL, covers K
 * so this is an inconsistency inside the tree, not a design choice.
 *
 * Sites with the FLOOR form (grep '+ 7) & ~'):
 *   src/core/transforms/real/r2c_dispatch.h   rfft_natural_mt
 *   src/core/transforms/real/c2r_dispatch.h   c2r_natural_mt
 *   src/core/transforms/fft3d/fft3d.h         _fft3d_axis0_mt
 *
 * Arithmetic says the smallest reproducer is K=17, T=2: floor(17/2)=8, S=8,
 * T*S=16 < 17, so lane 16 is dropped. 121 such (K,T) exist for K<200, T<=8.
 *
 * THIS PROBE DOES NOT ARGUE. It fills the destination with a CANARY, runs the
 * transform through the PUBLIC API, and reports any lane that still holds the
 * canary afterwards -- i.e. a lane nothing ever wrote. A canary survivor is not
 * a tolerance question, it is proof that no store reached that address.
 *
 * Controls, so a positive result cannot be a probe artefact:
 *   - K values where the arithmetic predicts FULL coverage must show 0 drops;
 *   - nthreads=1 must show 0 drops at every K (no slabbing at all);
 *   - the canary is checked on a buffer the probe itself filled, so "never
 *     written" is unambiguous.
 *
 * Build: python build.py --src benches/mt_lane_drop_probe.c --vfft --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "vfft.h"

#define CANARY (-1.2345678901234e300)

static uint64_t lcg = 0x243F6A8885A308D3ull;
static double rnd(void)
{ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
  return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

static int g_fail = 0, g_cells = 0;

/* predicted coverage of the FLOOR sizer */
static size_t s_floor(size_t K, int T)
{ size_t s = ((K / (size_t)T) + 7) & ~(size_t)7; return s ? s : 8; }

static void probe(int is_c2r, int N, size_t K, int T)
{
    const size_t nb = (size_t)N/2 + 1;
    vfft_config_t cfg;
    vfft_plan p;
    double *src, *dst;
    size_t ndst, i, t, dropped = 0, first = (size_t)-1;
    size_t predict = s_floor(K, T) * (size_t)T;

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = is_c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = K;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;   /* DEFAULT batch_geom = lane-major */
    cfg.nthreads = T;

    p = vfft_create(&cfg);
    if (!p) { printf("  %-3s N=%-5d K=%-4zu T=%d  create refused\n",
                     is_c2r?"c2r":"r2c", N, K, T); return; }

    ndst = is_c2r ? (size_t)N*K : 2*nb*K;
    src  = (double *)malloc(((is_c2r ? 2*nb*K : (size_t)N*K) + 16)*sizeof(double));
    dst  = (double *)malloc((ndst + 16)*sizeof(double));
    if (!src || !dst) { free(src); free(dst); vfft_destroy(p); return; }

    for (i = 0; i < (is_c2r ? 2*nb*K : (size_t)N*K); i++) src[i] = rnd();
    for (i = 0; i < ndst; i++) dst[i] = CANARY;

    vfft_execute(p, is_c2r ? VFFT_BACKWARD : VFFT_FORWARD, src, NULL, dst, NULL);

    /* lane-major: element e of transform t lives at [e*K + t] (real out) or
     * [2*(f*K + t)] (CCE out). A lane is DROPPED if every one of its slots
     * still holds the canary -- one surviving slot could be a coincidence of
     * the data, all of them cannot. */
    for (t = 0; t < K; t++) {
        size_t n = is_c2r ? (size_t)N : nb, survived = 0;
        for (i = 0; i < n; i++) {
            double v = is_c2r ? dst[i*K + t] : dst[2*(i*K + t)];
            if (v == CANARY) survived++;
        }
        if (survived == n) {
            dropped++;
            if (first == (size_t)-1) first = t;
        }
    }

    g_cells++;
    if (dropped) {
        printf("  %-3s N=%-5d K=%-4zu T=%d  *** %zu LANE(S) NEVER WRITTEN *** "
               "first=%zu  (FLOOR sizer covers %zu of %zu)\n",
               is_c2r?"c2r":"r2c", N, K, T, dropped, first, predict, K);
        g_fail = 1;
    } else {
        printf("  %-3s N=%-5d K=%-4zu T=%d  all %zu lanes written%s\n",
               is_c2r?"c2r":"r2c", N, K, T, K,
               predict < K ? "   (predicted a drop -- this path is NOT the FLOOR sizer)" : "");
    }

    free(src); free(dst); vfft_destroy(p);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("MT lane-drop probe -- canary survivors in threaded real batches\n");
    printf("  a LANE NEVER WRITTEN is proof, not a tolerance question\n\n");

    printf("[predicted DROP by the FLOOR sizer]\n");
    probe(0, 256, 17, 2);
    probe(0, 256, 33, 2);
    probe(0, 256, 25, 3);
    probe(0, 512, 17, 2);
    probe(1, 256, 17, 2);
    probe(1, 256, 33, 2);
    probe(1, 512, 25, 3);

    printf("\n[CONTROL: predicted FULL coverage -- must all pass]\n");
    probe(0, 256, 16, 2);
    probe(0, 256, 32, 2);
    probe(0, 256, 24, 3);
    probe(1, 256, 16, 2);
    probe(1, 256, 32, 2);

    printf("\n[CONTROL: single-threaded -- no slabbing at all, must all pass]\n");
    probe(0, 256, 17, 1);
    probe(0, 256, 25, 1);
    probe(1, 256, 17, 1);

    printf("\n%d cell(s). %s\n", g_cells,
           g_fail ? "*** LANES ARE BEING DROPPED ***"
                  : "no lane drops observed");
    return g_fail;
}
