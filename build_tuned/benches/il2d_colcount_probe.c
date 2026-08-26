/* il2d_colcount_probe.c — count-axis attribution for the IL-2D column
 * kinds at TINY counts (the real tier's hp1=9-class cells). Times the
 * SERVING kernels (blocked t2cb88/n1cb88 per the static resolvers) over
 * a two-stage 64.64 chain at N1=4096, sweeping count = columns:
 *   {9, 12, 16, 33, 36, 64, 129, 132, 513}
 * (each odd count paired with its next multiple-of-4 — the count-PADDING
 * prize is the odd-vs-padded delta at equal semantics; the VTW-twin
 * ceiling is the flat large-count rate). Same-run, min-of-5, ns/point.
 * Buffers sized for the largest count; pitch == count (bare plane).
 * Build: python build.py --src benches/il2d_colcount_probe.c --vfft --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef void (*zfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);
extern void radix64_z_t2cb88_fwd_avx2(const double *, const double *,
    double *, double *, const double *, const double *, size_t, size_t,
    size_t, size_t, size_t);
extern void radix64_z_n1cb88_fwd_avx2(const double *, const double *,
    double *, double *, const double *, const double *, size_t, size_t,
    size_t, size_t, size_t);

static double now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e9 + t.tv_nsec;
}

static double *mk_table(int R, int D_, int L)
{
    const double pi = 3.14159265358979323846;
    double *f = malloc((size_t)D_ * (R - 1) * 8 * sizeof(double));
    int d, r, ln;
    for (d = 0; d < D_; d++)
        for (r = 1; r < R; r++) {
            double a = -2.0 * pi * (double)(d * r % L) / (double)L;
            double c = cos(a), si = sin(a);
            double *rec = f + ((size_t)d * (R - 1) + (r - 1)) * 8;
            for (ln = 0; ln < 4; ln++) {
                rec[ln] = c;
                rec[4 + ln] = (ln & 1) ? si : -si;
            }
        }
    return f;
}

int main(void)
{
    static const int COUNTS[] = { 9, 12, 16, 33, 36, 64, 129, 132, 513 };
    const int NC = (int)(sizeof COUNTS / sizeof COUNTS[0]);
    const int N1 = 4096, R = 64, D1 = 64;      /* chain 64.64 */
    const size_t maxc = 513;
    double *z = malloc(2 * (size_t)N1 * maxc * sizeof(double));
    double *tab = mk_table(R, D1, N1);
    int ci, p;
    size_t i;
    if (!z || !tab) return 2;
    for (i = 0; i < 2 * (size_t)N1 * maxc; i++)
        z[i] = 1.0 + 1e-6 * (double)(i & 1023);
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("=== il2d column-kind count sweep (N1=4096 chain 64.64, "
           "t2cb88+n1cb88, in place, min-of-5) ===\n");
    printf(" count   pass ns    ns/pt   (pts = 4096*count*2)\n");
    for (ci = 0; ci < NC; ci++) {
        const size_t c = (size_t)COUNTS[ci];
        double best = 1e300;
        for (p = 0; p < 5; p++) {
            double t0 = now_ns(), dt;
            /* stage 0: t2c R=64 L=4096 D=64 — one block, count=c */
            radix64_z_t2cb88_fwd_avx2(z, NULL, z, NULL, tab, NULL,
                                      (size_t)D1 * c, c, (size_t)D1 * c,
                                      (size_t)D1, c);
            /* stage 1: n1c R=64 L=64 D=1 — 64 blocks, count=c */
            for (int b = 0; b < N1 / R; b++) {
                const size_t off = 2 * ((size_t)b * R * c);
                radix64_z_n1cb88_fwd_avx2(z + off, NULL, z + off, NULL,
                                          NULL, NULL, c, 0, c, 0, c);
            }
            dt = now_ns() - t0;
            if (dt < best) best = dt;
        }
        printf("  %4zu  %9.0f   %6.3f\n", c, best,
               best / ((double)N1 * c * 2));
    }
    free(z);
    free(tab);
    return 0;
}
