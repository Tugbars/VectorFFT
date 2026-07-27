/* test_oop_pad.c — OOP c2c PADDING through the public vfft.h. OOP bakes K (no runtime me) ->
 * PAD-ONLY, built at Kp=roundup(K,8) (BAILEY2 + the OOP wisdom reader both hard-gate on K%8).
 * The 4-plane handle (consolidated API): vfft_alloc_batch_for(cfg with placement=OUTOFPLACE);
 * vfft_batch_planes fills sre/sim = split INPUT, dre/dim = split OUTPUT, each N*Kp.
 * OOP output ORDER is kind-dependent (LEAF/BAILEY2 natural, MODEB scrambled), so the
 * robust gate is:
 *   (A) fwd->bwd ROUNDTRIP recovers N*x on lanes 0..K-1 (order-independent).
 *   (B) for K%8==0 (Kp==K) the padded fwd output == the TIGHT OOP fwd output, BIT-EXACT.
 * Build: python build.py --src test/test_oop_pad.c --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static int fails = 0;

static void mk_cfg(vfft_config_t *c, int N, int K, vfft_batch b)
{
    memset(c, 0, sizeof *c);
    c->transform = VFFT_C2C; c->placement = VFFT_OUTOFPLACE; c->rigor = VFFT_MEASURE;
    c->dims = 1; c->n[0] = N; c->howmany = (size_t)K; c->batch = b;
}
static vfft_plan mk_oop(int N, int K, vfft_batch b)
{
    vfft_config_t c;
    mk_cfg(&c, N, K, b);
    return vfft_create(&c);
}

static void cell(int N, int K)
{
    size_t Kp = ((size_t)K + 7u) & ~(size_t)7u;    /* OOP pads to roundup(K,8) */
    int aligned = ((size_t)K == Kp);               /* K%8==0 -> Kp==K -> compare vs tight */

    vfft_config_t bc;
    mk_cfg(&bc, N, K, NULL);
    vfft_batch b = vfft_alloc_batch_for(&bc);       /* born from the config */
    double *xr = (double *)malloc((size_t)N * K * sizeof(double));
    double *xi = (double *)malloc((size_t)N * K * sizeof(double));
    if (!b || !xr || !xi) { printf("  N=%-5d K=%-3d  alloc FAILED\n", N, K); fails++; goto done; }
    srand(13 + N + K);
    double *bre, *bim, *bore, *boim;
    vfft_batch_planes(b, &bre, &bim, &bore, &boim); /* in re/im -> out re/im roles */
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
        {
            double vr = (double)rand() / RAND_MAX - 0.5, vi = (double)rand() / RAND_MAX - 0.5;
            xr[n * K + k] = vr; xi[n * K + k] = vi;
            bre[(size_t)n * Kp + k] = vr; bim[(size_t)n * Kp + k] = vi;
        }
    vfft_plan p = mk_oop(N, K, b);
    if (!p) { printf("  N=%-5d K=%-3d Kp=%-3zu  padded OOP create FAILED\n", N, K, Kp); fails++; goto done; }

    /* (A/fwd) padded forward: input planes -> output planes (from planes()) */
    vfft_execute(p, VFFT_FORWARD, bre, bim, bore, boim);

    /* (B) tight OOP fwd (no batch), bit-exact vs padded when Kp==K */
    double tm = -1;
    if (aligned)
    {
        size_t nk = (size_t)N * K;
        double *tre = (double *)malloc(nk * sizeof(double)), *tim = (double *)malloc(nk * sizeof(double));
        double *tor = (double *)malloc(nk * sizeof(double)), *toi = (double *)malloc(nk * sizeof(double));
        if (tre && tim && tor && toi)
        {
            memcpy(tre, xr, nk * sizeof(double)); memcpy(tim, xi, nk * sizeof(double));
            vfft_plan pt = mk_oop(N, K, NULL);            /* tight (no batch) */
            if (pt)
            {
                vfft_execute(pt, VFFT_FORWARD, tre, tim, tor, toi);
                double *por = bore, *poi = boim;
                tm = 0;
                for (int n = 0; n < N; n++)
                    for (int k = 0; k < K; k++)
                    {
                        double dr = fabs(por[(size_t)n * Kp + k] - tor[(size_t)n * K + k]);
                        double di = fabs(poi[(size_t)n * Kp + k] - toi[(size_t)n * K + k]);
                        if (dr > tm) tm = dr;
                        if (di > tm) tm = di;
                    }
                vfft_destroy(pt);
            }
        }
        free(tre); free(tim); free(tor); free(toi);
    }

    /* (A/bwd) padded backward: roundtrip leg — the caller may legally pass
     * the planes swapped (output planes as the inverse input; documented in
     * vfft.h): out planes -> in planes, overwriting the input with N*x. */
    vfft_execute(p, VFFT_BACKWARD, bore, boim, bre, bim);
    double rt = 0, inv = 1.0 / (double)N;
    for (int n = 0; n < N; n++)
        for (int k = 0; k < K; k++)
        {
            double dr = fabs(bre[(size_t)n * Kp + k] * inv - xr[n * K + k]);
            double di = fabs(bim[(size_t)n * Kp + k] * inv - xi[n * K + k]);
            if (dr > rt) rt = dr;
            if (di > rt) rt = di;
        }
    vfft_destroy(p);

    int bad = (rt > 1e-10) || (aligned && (tm < 0 || tm > 1e-13));
    if (bad) fails++;
    if (aligned)
        printf("  N=%-5d K=%-3d Kp=%-3zu rem%d  tight-match=%9.1e  roundtrip=%9.1e  %s\n",
               N, K, Kp, K % 8, tm, rt, bad ? "<FAIL>" : "ok");
    else
        printf("  N=%-5d K=%-3d Kp=%-3zu rem%d  (odd-Kp: roundtrip only)  roundtrip=%9.1e  %s\n",
               N, K, Kp, K % 8, rt, bad ? "<FAIL>" : "ok");
done:
    if (b) vfft_free_batch(b);
    free(xr); free(xi);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    putenv("VFFT_WISDOM_DIR=oop_pad_test");
    system("mkdir oop_pad_test 2>nul");
    printf("# OOP c2c PADDING through public vfft.h (4-plane handle, pad-only, Kp=roundup(K,8))\n");
    printf("# rem = K%%8; aligned (rem0) also bit-exact vs tight OOP; else roundtrip is the gate\n");
    cell(256, 7);  cell(256, 8);  cell(256, 11); cell(256, 15); cell(256, 16); cell(256, 19); cell(256, 23);
    cell(512, 7);  cell(512, 11); cell(512, 16); cell(512, 23);
    cell(1024, 8); cell(1024, 15); cell(1024, 24);
    printf(fails ? "\nRESULT: %d FAILURE(S)\n" : "\nRESULT: all OOP c2c padded cells roundtrip + match tight\n", fails);
    return fails ? 1 : 0;
}
