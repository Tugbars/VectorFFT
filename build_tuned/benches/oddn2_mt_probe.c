/* oddn2_mt_probe.c - does column-MT engage on an ODD-N2 real 2D IL plan, and
 * if so does it still produce the single-threaded answer?
 *
 * src/core/vfft.c:6914-6919 guards the REAL tier's row-route race with
 * `!il2d_oddn2`; the column-MT guard at 6923-6924 carries no oddn2 term. So an
 * odd-N2 plan is excluded from the row race but NOT from column threading.
 * Whether that is deliberate or an oversight is a design question - this only
 * establishes the FACT: does cmt engage, and is MT bit-identical to ST?
 *
 * MT==ST bitwise is this library's own gate standard, so a mismatch here is a
 * defect regardless of which way the design question is answered.
 *
 * Build: VFFT_FINGERPRINT=1 python build.py --src benches/oddn2_mt_probe.c --vfft --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

static int N1 = 128, N2 = 127;

static void fill(double *v, size_t n, unsigned s)
{
    size_t i;
    for (i = 0; i < n; i++) { s = s * 1103515245u + 12345u;
        v[i] = (double)((s >> 16) & 0x7fff) / 32768.0 - 0.5; }
}

/* returns 0 on success; writes the complex output into out */
static int run(int nthr, double *out, size_t ocount, int *cmt, int *oddn2, long *races)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS];
    vfft_config_t cfg; vfft_plan p;
    double *in = (double *)calloc((size_t)N1 * N2, sizeof(double));
    char *q;
    if (!in) return 1;
    fill(in, (size_t)N1 * N2, 4242u);

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = VFFT_ORDER_DEFAULT;
    cfg.dims      = 2;
    cfg.n[0] = N1; cfg.n[1] = N2;
    cfg.howmany = 1;
    cfg.nthreads = nthr;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = 0;

    p = vfft_create(&cfg);
    if (!p) { free(in); printf("  nthr=%d REFUSED\n", nthr); return 1; }
    vfft__fp_counters(c); *races = c[5];
    vfft__fingerprint(p, buf, sizeof buf);
    q = strstr(buf, "cmt=");    *cmt   = q ? atoi(q + 4) : -1;
    q = strstr(buf, "oddn2=");  *oddn2 = q ? atoi(q + 6) : -1;

    memset(out, 0, ocount * sizeof(double));
    vfft_execute(p, VFFT_FORWARD, in, NULL, out, NULL);
    vfft_destroy(p);
    free(in);
    return 0;
}

int main(void)
{
    size_t ocount = (size_t)N1 * (N2 / 2 + 1) * 2 + 64;
    double *a = (double *)calloc(ocount, sizeof(double));
    double *b = (double *)calloc(ocount, sizeof(double));
    int cmt1 = -1, cmt8 = -1, o1 = -1, o8 = -1;
    long r1 = 0, r8 = 0;
    size_t i, diff = 0;
    double worst = 0.0;

    if (!a || !b) return 2;
    printf("2D IL OOP r2c %dx%d  (N2=%d is ODD)\n", N1, N2, N2);
    if (run(1, a, ocount, &cmt1, &o1, &r1)) return 2;
    printf("  nthr=1  oddn2=%d cmt=%d races=%ld\n", o1, cmt1, r1);
    if (run(8, b, ocount, &cmt8, &o8, &r8)) return 2;
    printf("  nthr=8  oddn2=%d cmt=%d races=%ld\n", o8, cmt8, r8);

    for (i = 0; i < ocount; i++) {
        double d = a[i] - b[i];
        if (d != 0.0) { diff++; if (d < 0) d = -d; if (d > worst) worst = d; }
    }
    printf("  MT vs ST: %zu of %zu doubles differ", diff, ocount);
    if (diff) printf(", worst |delta| = %.3e  *** NOT BITWISE ***\n", worst);
    else      printf("  -> BIT-IDENTICAL\n");
    printf("  VERDICT: column-MT %s on an odd-N2 plan\n",
           cmt8 > 0 ? "ENGAGED" : "did NOT engage");
    free(a); free(b);
    return 0;
}
