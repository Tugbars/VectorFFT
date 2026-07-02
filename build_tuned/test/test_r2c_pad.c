/* test_r2c_pad.c — r2c/c2r PADDING (opt-in Kp fast path) through the public vfft.h.
 *   (A) MATCH gate:  padded r2c output lanes 0..K-1 == tight r2c output (each lane is an
 *                    independent transform; the Kp-plan just runs full-SIMD over more lanes).
 *                    Not bit-exact (tight-K and padded-Kp are separately calibrated cells, so
 *                    factorizations can differ) — numerically equal ~1e-13 proves correctness.
 *   (B) ROUNDTRIP:   padded r2c -> padded c2r recovers N*x on lanes 0..K-1.
 * The padded handle is transform-aware: R2C = real INPUT (N*Kp) + split spectrum OUT
 * ((N/2+1)*Kp); C2R the reverse. All planes at stride Kp = roundup(K,4); pad lanes zeroed.
 * Build: python build.py --src test/test_r2c_pad.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int fails = 0;

static void cell(int N, int K)
{
    int    H  = N / 2 + 1;                       /* half-spectrum rows */
    size_t Kp = ((size_t)K + 3u) & ~(size_t)3u;  /* roundup(K, VW=4) */

    /* ---------- tight reference (stride K) ---------- */
    double *xt  = (double *)calloc((size_t)N * K, sizeof(double));
    double *ret = (double *)calloc((size_t)H * K, sizeof(double));
    double *imt = (double *)calloc((size_t)H * K, sizeof(double));
    srand(11 + N + K);
    for (int i = 0; i < N * K; i++) xt[i] = (double)rand() / RAND_MAX - 0.5;

    vfft_config_t rc; memset(&rc, 0, sizeof rc);
    rc.transform = VFFT_R2C; rc.placement = VFFT_OUTOFPLACE; rc.rigor = VFFT_MEASURE;
    rc.dims = 1; rc.n[0] = N; rc.howmany = (size_t)K;
    vfft_plan pt = vfft_create(&rc);
    if (pt) vfft_execute(pt, VFFT_FORWARD, xt, NULL, ret, imt);
    else { printf("  N=%-5d K=%-3d  tight r2c create FAILED\n", N, K); fails++; }

    /* ---------- padded r2c (stride Kp) ---------- */
    double match = -1, rt = -1;
    vfft_batch bf = vfft_alloc_batch_ex(VFFT_R2C, N, (size_t)K);
    vfft_plan  pf = NULL;
    if (!bf) { printf("  N=%-5d K=%-3d  vfft_alloc_batch_ex(R2C) FAILED\n", N, K); fails++; }
    else if (pt)
    {
        double *xr = vfft_batch_real(bf);
        for (int n = 0; n < N; n++)                      /* same K signals at stride Kp */
            for (int k = 0; k < K; k++)
                xr[(size_t)n * Kp + k] = xt[(size_t)n * K + k];

        vfft_config_t rcp = rc; rcp.batch = bf;
        pf = vfft_create(&rcp);
        if (!pf) { printf("  N=%-5d K=%-3d Kp=%zu  padded r2c create FAILED (Kp -> gated stride?)\n", N, K, Kp); fails++; }
        else
        {
            double *rep = vfft_batch_re(bf), *imp = vfft_batch_im(bf);
            vfft_execute(pf, VFFT_FORWARD, xr, NULL, rep, imp);
            /* (A) match padded lanes 0..K-1 vs tight */
            match = 0;
            for (int h = 0; h < H; h++)
                for (int k = 0; k < K; k++)
                {
                    double dr = fabs(rep[(size_t)h * Kp + k] - ret[(size_t)h * K + k]);
                    double di = fabs(imp[(size_t)h * Kp + k] - imt[(size_t)h * K + k]);
                    if (dr > match) match = dr;
                    if (di > match) match = di;
                }

            /* (B) padded roundtrip: c2r on the padded spectrum -> real; recover N*x */
            vfft_batch bb = vfft_alloc_batch_ex(VFFT_C2R, N, (size_t)K);
            if (!bb) { printf("  N=%-5d K=%-3d  vfft_alloc_batch_ex(C2R) FAILED\n", N, K); fails++; }
            else
            {
                double *cre = vfft_batch_re(bb), *cim = vfft_batch_im(bb);
                for (int h = 0; h < H; h++)
                    for (int k = 0; k < K; k++)
                    {
                        cre[(size_t)h * Kp + k] = rep[(size_t)h * Kp + k];
                        cim[(size_t)h * Kp + k] = imp[(size_t)h * Kp + k];
                    }
                vfft_config_t cc; memset(&cc, 0, sizeof cc);
                cc.transform = VFFT_C2R; cc.placement = VFFT_OUTOFPLACE; cc.rigor = VFFT_MEASURE;
                cc.dims = 1; cc.n[0] = N; cc.howmany = (size_t)K; cc.batch = bb;
                vfft_plan pb = vfft_create(&cc);
                if (!pb) { printf("  N=%-5d K=%-3d  padded c2r create FAILED\n", N, K); fails++; }
                else
                {
                    double *yr = vfft_batch_real(bb);
                    vfft_execute(pb, VFFT_BACKWARD, cre, cim, yr, NULL);
                    rt = 0; double inv = 1.0 / (double)N;
                    for (int n = 0; n < N; n++)
                        for (int k = 0; k < K; k++)
                        {
                            double d = fabs(yr[(size_t)n * Kp + k] * inv - xt[(size_t)n * K + k]);
                            if (d > rt) rt = d;
                        }
                    vfft_destroy(pb);
                }
                vfft_free_batch(bb);
            }
        }
    }

    int bad = (match < 0) || (match > 1e-12) || (rt < 0) || (rt > 1e-10);
    if (bad) fails++;
    printf("  N=%-5d K=%-3d Kp=%zu rem%d  match(pad0..K vs tight)=%9.1e  roundtrip=%9.1e  %s\n",
           N, K, Kp, K % 4, match, rt, bad ? "<FAIL>" : "ok");

    if (pf) vfft_destroy(pf);
    if (bf) vfft_free_batch(bf);
    if (pt) vfft_destroy(pt);
    free(xt); free(ret); free(imt);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    putenv("VFFT_WISDOM_DIR=r2c_pad_test");     /* isolate calibrate-on-miss writes */
    system("mkdir r2c_pad_test 2>nul");
    printf("# r2c/c2r PADDING through public vfft.h (transform-aware handle; pad-only Kp build)\n");
    printf("# cascade regime, small misaligned K (Kp<32): the padding payoff zone\n");
    cell(256, 7);  cell(256, 11); cell(256, 15); cell(256, 19); cell(256, 23);
    cell(512, 7);  cell(512, 11); cell(512, 23);
    cell(1024, 7); cell(1024, 15);
    printf("# even/aligned K (Kp==K): padded handle == tight, must still match + roundtrip\n");
    cell(256, 8);  cell(256, 16);
    printf(fails ? "\nRESULT: %d FAILURE(S)\n" : "\nRESULT: all r2c/c2r padded cells match tight + roundtrip\n", fails);
    return fails ? 1 : 0;
}
