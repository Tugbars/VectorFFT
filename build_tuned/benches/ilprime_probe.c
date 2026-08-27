/* ilprime_probe.c — coverage probe: which odd/prime N does the 1D C2C
 * INTERLEAVED tier serve natively, and how (roundtrip + DC identity)?
 * Classes: small/large primes in band, prime > 2048 (the il2p inner
 * ceiling), odd smooth composites, odd composites with a big prime
 * factor, a prime square.
 * Build: python build.py --src benches/ilprime_probe.c --vfft --mkl --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

int main(int argc, char **argv)
{
    static const struct { int n; const char *what; } C[] = {
        { 61,   "prime" },        { 127,  "prime" },
        { 509,  "prime" },        { 1021, "prime" },
        { 2039, "prime (band edge)" },
        { 4099, "prime > 2048" },
        { 45,   "odd 9*5" },      { 105,  "odd 3*5*7" },
        { 675,  "odd 27*25" },
        { 121,  "11^2" },         { 115,  "5*23" },
        { 202,  "2*101" },
    };
    const int NC = (int)(sizeof C / sizeof C[0]);
    const char *wisdir = argc > 1 ? argv[1] : ".";
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    int ci;
    setvbuf(stdout, NULL, _IONBF, 0);
    for (ci = 0; ci < NC; ci++) {
        const int N = C[ci].n;
        double *x = malloc(2 * (size_t)N * 8), *z = malloc(2 * (size_t)N * 8);
        double *y = malloc(2 * (size_t)N * 8);
        vfft_config_t cfg;
        vfft_plan p;
        double rt = 0, dc = 0, s0 = 0, s1 = 0;
        int i;
        if (!x || !z || !y) return 2;
        for (i = 0; i < 2 * N; i++)
            x[i] = (double)rand() / RAND_MAX - 0.5;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C;
        cfg.placement = VFFT_OUTOFPLACE;
        cfg.rigor = VFFT_MEASURE;
        cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.wisdom = W; cfg.wisdom_write = 0;
        p = vfft_create(&cfg);
        if (!p) {
            printf("N=%-5d %-18s REFUSED at create\n", N, C[ci].what);
            free(x); free(z); free(y);
            continue;
        }
        vfft_execute(p, VFFT_FORWARD, x, NULL, z, NULL);
        /* DC bin must equal the plain sum (order-independent) */
        for (i = 0; i < N; i++) { s0 += x[2 * i]; s1 += x[2 * i + 1]; }
        dc = fabs(z[0] - s0) + fabs(z[1] - s1);
        vfft_execute(p, VFFT_BACKWARD, z, NULL, y, NULL);
        for (i = 0; i < 2 * N; i++) {
            double d = fabs(y[i] / N - x[i]);
            if (d > rt) rt = d;
        }
        printf("N=%-5d %-18s rt %.1e  dc %.1e  %s\n", N, C[ci].what, rt,
               dc, (rt < 1e-9 && dc < 1e-9) ? "OK" : "*** WRONG ***");
        vfft_destroy(p);
        free(x); free(z); free(y);
    }
    if (W) vfft_wisdom_free(W);
    return 0;
}
