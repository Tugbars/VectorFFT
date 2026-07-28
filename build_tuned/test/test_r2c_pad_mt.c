/* test_r2c_pad_mt.c — MT padding for r2c/c2r: an owned-buffer plan with nthreads>1 must be correct.
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

static void mk_cfg(vfft_config_t *c, vfft_transform_t xf, int N, int K, int nthreads)
{
    memset(c, 0, sizeof *c);
    c->transform = xf; c->placement = VFFT_OUTOFPLACE; c->rigor = VFFT_MEASURE;
    c->dims = 1; c->n[0] = N; c->howmany = (size_t)K; c->owned_buffers = 1; c->nthreads = nthreads;
}
static vfft_plan mk(vfft_transform_t xf, int N, int K, int nthreads)
{
    vfft_config_t c;
    mk_cfg(&c, xf, N, K, nthreads);
    return vfft_create(&c);
}

static void cell(int N, int K, int T)
{
    int H = N / 2 + 1;
    /* create FIRST: the plan owns the padded planes and chooses the stride */
    vfft_plan pmt = mk(VFFT_R2C, N, K, T);
    vfft_plan pst = mk(VFFT_R2C, N, K, 1);
    double *x = (double *)malloc((size_t)N * K * sizeof(double));
    double *mreal = NULL, *mre = NULL, *mim = NULL, *sreal = NULL, *sre_ = NULL, *sim_ = NULL, *dum;
    if (pmt) vfft_plan_planes(pmt, &mreal, &dum, &mre, &mim);
    if (pst) vfft_plan_planes(pst, &sreal, &dum, &sre_, &sim_);
    if (!x) { printf("  N=%-5d K=%-3d  alloc FAILED\n", N, K); fails++; goto done; }
    if (!pmt || !pst) { printf("  N=%-5d K=%-3d  r2c create FAILED\n", N, K); fails++; goto done; }
    /* stride comes from the plan, never a local roundup (see vfft.h) */
    size_t Kp = vfft_plan_stride(pmt);
    int threads = (Kp >= 16);                         /* rfft_natural_mt floor; else ST fallback */
    srand(9 + N + K);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
        {
            double v = (double)rand() / RAND_MAX - 0.5;
            x[n * K + k] = v;
            mreal[(size_t)n * Kp + k] = v;
            sreal[(size_t)n * Kp + k] = v;
        }
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
    vfft_plan pc = mk(VFFT_C2R, N, K, T);
    if (!pc) { printf("  N=%-5d K=%-3d  c2r create FAILED\n", N, K); fails++; }
    else
    {
        double *cre = NULL, *cim = NULL, *creal = NULL;
        vfft_plan_planes(pc, &cre, &cim, &creal, &dum); /* spectrum in -> real out */
        size_t Kc = vfft_plan_stride(pc);              /* this plan's own stride */
        for (int h = 0; h < H; h++)
            for (int k = 0; k < K; k++)
            {
                cre[(size_t)h * Kc + k] = rem[(size_t)h * Kp + k];
                cim[(size_t)h * Kc + k] = imm[(size_t)h * Kp + k];
            }
        vfft_execute(pc, VFFT_BACKWARD, cre, cim, creal, NULL);
        double *y = creal; rt = 0; double inv = 1.0 / (double)N;
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
            { double d = fabs(y[(size_t)n * Kc + k] * inv - x[n * K + k]); if (d > rt) rt = d; }
        vfft_destroy(pc);
    }

    int bad = (mtst > 1e-12) || (rt < 0) || (rt > 1e-10);
    if (bad) fails++;
    printf("  N=%-5d K=%-3d Kp=%-3zu T=%d %-9s  MT-vs-ST=%9.1e  roundtrip=%9.1e  %s\n",
           N, K, Kp, T, threads ? "threaded" : "ST-fallbk", mtst, rt, bad ? "<FAIL>" : "ok");
done:
    if (pmt) vfft_destroy(pmt);   /* frees the planes it handed out */
    if (pst) vfft_destroy(pst);
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
