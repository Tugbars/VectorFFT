/* test_trig_pad.c — trig (DCT/DST/DHT) PADDING through the public vfft.h.
 * Trig padding is PAD-ONLY (the trig stride_r2c_plan bakes K) and sidesteps the UNBUILT odd-K
 * trig tail: building at aligned Kp is the ONLY full-SIMD path for misaligned K. Handle is
 * real->real: vfft_batch_real = INPUT (N*Kp), vfft_batch_re = OUTPUT (N*Kp), stride Kp.
 *   (A) ABSOLUTE: padded DCT-II output lanes 0..K-1 == naive FFTW REDFT10 (incl. odd K).
 *   (B) ROUNDTRIP: fwd->inv recovers scale*x on lanes 0..K-1 for DCT2 (via DCT3), DHT, DCT4.
 * Build: python build.py --src test/test_trig_pad.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int fails = 0;
static const double PI = 3.14159265358979323846;

/* naive FFTW REDFT10 (DCT-II): X[k] = 2 sum_n x[n] cos(pi (2n+1) k / (2N)) */
static void naive_dct2(const double *x, double *X, int N)
{
    for (int k = 0; k < N; k++)
    {
        double s = 0;
        for (int n = 0; n < N; n++)
            s += x[n] * cos(PI * (2 * n + 1) * k / (2.0 * N));
        X[k] = 2.0 * s;
    }
}

/* roundtrip a padded trig handle of kind xf: fwd(real->re) then inv(re->real);
 * return max rel error of the recovered lanes 0..K-1 vs scale*x0 (scale = least-squares). */
static double trig_roundtrip(vfft_transform_t xf, int N, int K, unsigned seed)
{
    vfft_batch b = vfft_alloc_batch_ex(xf, N, (size_t)K);
    if (!b) return -1;
    size_t Kp = vfft_batch_stride(b);
    double *x0 = (double *)malloc((size_t)N * K * sizeof(double));
    double *in = vfft_batch_real(b);
    srand(seed);
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
        {
            double v = (double)rand() / RAND_MAX - 0.5;
            x0[n * K + k] = v;
            in[(size_t)n * Kp + k] = v;
        }
    vfft_config_t c; memset(&c, 0, sizeof c);
    c.transform = xf; c.placement = VFFT_OUTOFPLACE; c.rigor = VFFT_ESTIMATE;
    c.dims = 1; c.n[0] = N; c.howmany = (size_t)K; c.batch = b;
    vfft_plan p = vfft_create(&c);
    double err = -1;
    if (p)
    {
        vfft_execute(p, VFFT_FORWARD,  vfft_batch_real(b), NULL, vfft_batch_re(b),   NULL);
        vfft_execute(p, VFFT_BACKWARD, vfft_batch_re(b),   NULL, vfft_batch_real(b), NULL);
        double *y = vfft_batch_real(b);
        double sxy = 0, sxx = 0;                                  /* least-squares scale */
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
            { double a = x0[n * K + k], yy = y[(size_t)n * Kp + k]; sxy += a * yy; sxx += a * a; }
        double sc = sxx > 0 ? sxy / sxx : 0, denom = 0;
        err = 0;
        for (int n = 0; n < N; n++)
            for (int k = 0; k < K; k++)
            {
                double a = x0[n * K + k] * sc, yy = y[(size_t)n * Kp + k], d = fabs(yy - a);
                if (d > err) err = d;
                if (fabs(a) > denom) denom = fabs(a);
            }
        if (denom > 0) err /= denom;
        vfft_destroy(p);
    }
    free(x0);
    vfft_free_batch(b);
    return err;
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    putenv("VFFT_WISDOM_DIR=trig_pad_test");
    system("mkdir trig_pad_test 2>nul");
    printf("# trig PADDING through public vfft.h (pad-only, aligned Kp build; sidesteps the odd-K trig tail)\n");

    /* (A) DCT-II padded output vs naive REDFT10 — absolute correctness, incl. odd K */
    printf("# (A) DCT-II padded output vs naive FFTW REDFT10 (absolute; odd K = otherwise-unbuilt path):\n");
    int As[][2] = {{8, 3}, {16, 5}, {32, 7}, {64, 11}, {256, 7}, {256, 15}};
    for (int i = 0; i < 6; i++)
    {
        int N = As[i][0], K = As[i][1];
        vfft_batch b = vfft_alloc_batch_ex(VFFT_DCT2, N, (size_t)K);
        size_t Kp = b ? vfft_batch_stride(b) : 0;
        double maxe = -1;
        if (b)
        {
            double *in = vfft_batch_real(b);
            double *x = (double *)malloc((size_t)N * K * sizeof(double));
            srand(3 + N + K);
            for (int n = 0; n < N; n++)
                for (int k = 0; k < K; k++)
                { double v = (double)rand() / RAND_MAX - 0.5; x[n * K + k] = v; in[(size_t)n * Kp + k] = v; }
            vfft_config_t c; memset(&c, 0, sizeof c);
            c.transform = VFFT_DCT2; c.placement = VFFT_OUTOFPLACE; c.rigor = VFFT_ESTIMATE;
            c.dims = 1; c.n[0] = N; c.howmany = (size_t)K; c.batch = b;
            vfft_plan p = vfft_create(&c);
            if (p)
            {
                vfft_execute(p, VFFT_FORWARD, vfft_batch_real(b), NULL, vfft_batch_re(b), NULL);
                double *out = vfft_batch_re(b);
                double *Xr = (double *)malloc(N * sizeof(double)), *xl = (double *)malloc(N * sizeof(double));
                maxe = 0;
                for (int k = 0; k < K; k++)
                {
                    for (int n = 0; n < N; n++) xl[n] = x[n * K + k];
                    naive_dct2(xl, Xr, N);
                    for (int kk = 0; kk < N; kk++)
                    { double d = fabs(out[(size_t)kk * Kp + k] - Xr[kk]); if (d > maxe) maxe = d; }
                }
                free(Xr); free(xl); vfft_destroy(p);
            }
            free(x); vfft_free_batch(b);
        }
        int bad = (maxe < 0) || (maxe > 1e-10);
        if (bad) fails++;
        printf("  DCT2  N=%-5d K=%-3d Kp=%-3zu rem%d  |padded-naive|=%9.1e  %s\n", N, K, Kp, K % 4, maxe, bad ? "<FAIL>" : "ok");
    }

    /* (B) roundtrip fwd->inv for DCT2 (via DCT3) / DHT (involutory) / DCT4 (involutory) */
    printf("# (B) roundtrip fwd->inv recovers scale*x (odd + even K):\n");
    struct { vfft_transform_t xf; const char *nm; } K3[] = { {VFFT_DCT2, "DCT2"}, {VFFT_DHT, "DHT "}, {VFFT_DCT4, "DCT4"} };
    int Bs[][2] = {{256, 7}, {256, 8}, {256, 15}, {512, 11}, {512, 23}};
    for (int t = 0; t < 3; t++)
        for (int i = 0; i < 5; i++)
        {
            int N = Bs[i][0], K = Bs[i][1];
            double e = trig_roundtrip(K3[t].xf, N, K, 5 + N + K + t);
            int bad = (e < 0) || (e > 1e-9);
            if (bad) fails++;
            printf("  %s  N=%-5d K=%-3d rem%d  roundtrip=%9.1e  %s\n", K3[t].nm, N, K, K % 4, e, bad ? "<FAIL>" : "ok");
        }

    printf(fails ? "\nRESULT: %d FAILURE(S)\n" : "\nRESULT: all trig padded cells correct (naive DCT-II + roundtrip)\n", fails);
    return fails ? 1 : 0;
}
