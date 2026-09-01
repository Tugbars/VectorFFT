/* oddmid_mt_probe.c — the odd-mid cascade MT pricing trial (2026-09-02).
 *
 * Two OOP scrambled K=1 creates at nthreads=T for an odd-mid N on fresh
 * bundles: plan A with VFFT_ZT_NO_MT=1 (zt_mt forced 0 = today's serving),
 * plan B with the race enabled (the trial's zt_mt=1 finish flag). Then:
 *   1. BITWISE: fwd outputs of A and B on identical input must memcmp==0
 *      (the MT==ST law; the cascade MT partitioning on an ODD chain has
 *      never been gated before);
 *   2. SPEED: min-of-R fwd executes of each, alternated;
 *   3. the [zt-mt] stderr line says whether the race engaged and its pick.
 * usage: probe N [T=8] [reps=9] [wisdir-base=omt]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "vfft.h"

static double now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

static vfft_plan mk(const char *dir, int N, int T)
{
    vfft_wisdom *W = vfft_wisdom_load(dir);
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.order = VFFT_ORDER_SCRAMBLED;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = 1;
    cfg.nthreads = T;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom = W;
    cfg.wisdom_write = 0;
    vfft_plan p = vfft_create(&cfg);
    /* W deliberately leaked for the probe's lifetime (the plan may read it) */
    return p;
}

int main(int argc, char **argv)
{
    int N = argc > 1 ? atoi(argv[1]) : 6144;
    int T = argc > 2 ? atoi(argv[2]) : 8;
    int R = argc > 3 ? atoi(argv[3]) : 9;
    const char *base = argc > 4 ? argv[4] : "omt";
    char d1[512], d2[512];
    snprintf(d1, sizeof d1, "%s_ser", base);
    snprintf(d2, sizeof d2, "%s_mt", base);
    static char lg[] = "VFFT_ZT_LOG=1";
    putenv(lg);
    setvbuf(stderr, NULL, _IONBF, 0);

    static char off[] = "VFFT_ZT_NO_MT=1";
    putenv(off); /* plan A: forced serial (today's odd-mid serving) */
    vfft_plan pa = mk(d1, N, T);
    static char on[] = "VFFT_ZT_NO_MT="; /* unset: the race decides */
    putenv(on);
    vfft_plan pb = mk(d2, N, T);
    if (!pa || !pb)
    {
        printf("N=%d create FAILED (a=%p b=%p)\n", N, (void *)pa, (void *)pb);
        return 2;
    }
    size_t nb = 2 * (size_t)N * sizeof(double);
    double *zi = (double *)malloc(nb), *za = (double *)malloc(nb),
           *zb = (double *)malloc(nb);
    if (!zi || !za || !zb) return 2;
    for (long i = 0; i < 2L * N; i++)
        zi[i] = (double)((i * 2654435761u) & 1023) / 1024.0 - 0.5;

    vfft_execute(pa, VFFT_FORWARD, zi, NULL, za, NULL); /* warm both */
    vfft_execute(pb, VFFT_FORWARD, zi, NULL, zb, NULL);
    int bitwise = memcmp(za, zb, nb) == 0;

    double ta = 1e300, tb = 1e300;
    for (int r = 0; r < R; r++)
    {
        double t0 = now_ns();
        vfft_execute(pa, VFFT_FORWARD, zi, NULL, za, NULL);
        double d = now_ns() - t0;
        if (d < ta) ta = d;
        t0 = now_ns();
        vfft_execute(pb, VFFT_FORWARD, zi, NULL, zb, NULL);
        d = now_ns() - t0;
        if (d < tb) tb = d;
    }
    printf("N=%-7d T=%d bitwise=%s serial=%.0fns raced=%.0fns -> %s (%.2fx)\n",
           N, T, bitwise ? "IDENTICAL" : "*** MISMATCH ***", ta, tb,
           tb < ta ? "MT-serving faster" : "serial faster",
           ta / (tb > 0 ? tb : 1));
    vfft_destroy(pa);
    vfft_destroy(pb);
    free(zi); free(za); free(zb);
    return bitwise ? 0 : 1;
}
