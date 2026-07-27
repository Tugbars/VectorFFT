/* test_r2c_pad_mt.c — MT padding for r2c/c2r: a padded handle with nthreads>1 must be correct.
 * The cascade-regime MT (rfft_natural_mt) K-splits the batch into DISJOINT lane slabs; for a
 * padded (Kp) plan the pad lanes t in [K,Kp) ride the last slab as harmless zeros, so no new
 * code is needed — this test just proves it. NOTE: MT engages only at Kp>=16 (else single-thread
 * fallback, still correct), so small K threads little but stays correct.
 *   (A) MT r2c fwd == ST r2c fwd on lanes 0..K-1 (the lane-slab split is race-free + correct).
 *   (B) MT r2c -> MT c2r roundtrip recovers N*x on lanes 0..K-1.
 * Build: python build.py --src test/test_r2c_pad_mt.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int fails = 0;

static void mk_cfg(vfft_config_t *c, vfft_transform_t xf, int N, int K, vfft_batch b, int nthreads)
{
    memset(c, 0, sizeof *c);
    c->transform = xf; c->placement = VFFT_OUTOFPLACE; c->rigor = VFFT_MEASURE;
    c->dims = 1; c->n[0] = N; c->howmany = (size_t)K; c->batch = b; c->nthreads = nthreads;
}
static vfft_plan mk(vfft_transform_t xf, int N, int K, vfft_batch b, int nthreads)
{
    vfft_config_t c;
    mk_cfg(&c, xf, N, K, b, nthreads);
    return vfft_create(&c);
}
static vfft_batch mk_batch(vfft_transform_t xf, int N, int K)
{
    vfft_config_t c;
    mk_cfg(&c, xf, N, K, NULL, 0);
    return vfft_alloc_batch_for(&c);
}

static void cell(int N, int K, int T)
{
    int H = N / 2 + 1;
    size_t Kp = ((size_t)K + 3u) & ~(size_t)3u;
    int threads = (Kp >= 16);                         /* rfft_natural_mt floor; else ST fallback */

    vfft_batch bmt = mk_batch(VFFT_R2C, N, K);
    vfft_batch bst = mk_batch(VFFT_R2C, N, K);
    double *x = (double *)malloc((size_t)N * K * sizeof(double));
    double *mreal = NULL, *mre = NULL, *mim = NULL, *sreal = NULL, *sre_ = NULL, *sim_ = NULL, *dum;
    if (bmt) vfft_batch_planes(bmt, &mreal, &dum, &mre, &mim);
    if (bst) vfft_batch_planes(bst, &sreal, &dum, &sre_, &sim_);
    if (!bmt || !bst || !x) { printf("  N=%-5d K=%-3d  alloc FAILED\n", N, K); fails++; goto done; }
    srand(9 + N + K);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
        {
            double v = (double)rand() / RAND_MAX - 0.5;
            x[n * K + k] = v;
            mreal[(size_t)n * Kp + k] = v;
            sreal[(size_t)n * Kp + k] = v;
        }
    vfft_plan pmt = mk(VFFT_R2C, N, K, bmt, T);
    vfft_plan pst = mk(VFFT_R2C, N, K, bst, 1);
    if (!pmt || !pst) { printf("  N=%-5d K=%-3d  r2c create FAILED\n", N, K); fails++; goto done; }
    vfft_execute(pmt, VFFT_FORWARD, mreal, NULL, mre, mim);
    vfft_execute(pst, VFFT_FORWARD, sreal, NULL, sre_, sim_);

    /* (A) MT vs ST forward, lanes 0..K-1 */
    double mtst = 0;
    double *rem = mre, *imm = mim;
    double *res = sre_, *ims = sim_;
    for (int h = 0; h < H; h++)
        for (int k = 0; k < K; k++)
        {
            double dr = fabs(rem[(size_t)h * Kp + k] - res[(size_t)h * Kp + k]);
            double di = fabs(imm[(size_t)h * Kp + k] - ims[(size_t)h * Kp + k]);
            if (dr > mtst) mtst = dr;
            if (di > mtst) mtst = di;
        }

    /* (B) MT roundtrip: c2r (MT) on the MT spectrum -> real; recover N*x */
    double rt = -1;
    vfft_batch bc = mk_batch(VFFT_C2R, N, K);
    if (!bc) { printf("  N=%-5d K=%-3d  c2r alloc FAILED\n", N, K); fails++; }
    else
    {
        double *cre, *cim, *creal;
        vfft_batch_planes(bc, &cre, &cim, &creal, &dum); /* spectrum in -> real out */
        for (int h = 0; h < H; h++)
            for (int k = 0; k < K; k++)
            {
                cre[(size_t)h * Kp + k] = rem[(size_t)h * Kp + k];
                cim[(size_t)h * Kp + k] = imm[(size_t)h * Kp + k];
            }
        vfft_plan pc = mk(VFFT_C2R, N, K, bc, T);
        if (!pc) { printf("  N=%-5d K=%-3d  c2r create FAILED\n", N, K); fails++; }
        else
        {
            vfft_execute(pc, VFFT_BACKWARD, cre, cim, creal, NULL);
            double *y = creal; rt = 0; double inv = 1.0 / (double)N;
            for (int n = 0; n < N; n++)
                for (int k = 0; k < K; k++)
                { double d = fabs(y[(size_t)n * Kp + k] * inv - x[n * K + k]); if (d > rt) rt = d; }
            vfft_destroy(pc);
        }
        vfft_free_batch(bc);
    }
    vfft_destroy(pmt); vfft_destroy(pst);

    int bad = (mtst > 1e-12) || (rt < 0) || (rt > 1e-10);
    if (bad) fails++;
    printf("  N=%-5d K=%-3d Kp=%-3zu T=%d %-9s  MT-vs-ST=%9.1e  roundtrip=%9.1e  %s\n",
           N, K, Kp, T, threads ? "threaded" : "ST-fallbk", mtst, rt, bad ? "<FAIL>" : "ok");
done:
    if (bmt) vfft_free_batch(bmt);
    if (bst) vfft_free_batch(bst);
    free(x);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    putenv("VFFT_WISDOM_DIR=r2c_pad_mt_test");
    system("mkdir r2c_pad_mt_test 2>nul");
    printf("# MT r2c/c2r PADDING through public vfft.h (nthreads=4; lane-slab split over Kp)\n");
    printf("# threaded = Kp>=16 (rfft_natural_mt floor); smaller = single-thread fallback (still correct)\n");
    cell(256, 7, 4);   cell(256, 15, 4);  cell(256, 19, 4);  cell(256, 23, 4);
    cell(512, 15, 4);  cell(512, 23, 4);  cell(512, 31, 4);
    cell(1024, 23, 4); cell(1024, 31, 4);
    printf(fails ? "\nRESULT: %d FAILURE(S)\n" : "\nRESULT: all MT-padded r2c cells match ST + roundtrip\n", fails);
    return fails ? 1 : 0;
}
