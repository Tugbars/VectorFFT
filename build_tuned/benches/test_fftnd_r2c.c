/* test_fftnd_r2c.c — rank-general r2c/c2r validation.
 *
 * Gates per cell:
 *   1. ROUNDTRIP  c2r(r2c(x)) == Ntotal * x            (definitive)
 *   2. PER-BIN vs long-double reference: phase-probe the r2c plan itself
 *      for the outer-axis scramble maps (real impulse at position 1 on
 *      axis m -> spectrum e^(-2*pi*i k_m/N_m) constant over all other
 *      axes; read the packed-output pencil at f=0), verify the half axis
 *      is natural the same way, gather, and compare against a long-double
 *      full DFT restricted to f <= Nd/2. External elementwise validation
 *      of the real path.
 *   3. PARSEVAL (real form): sum x^2 * Ntot == |X0|^2+|X_{N/2}|^2-ish...
 *      covered implicitly by gate 2; skipped as a separate gate.
 *   4. MT: fwd output bit-identical T in {1,2,4}.
 *   Prime axes appear on OUTER positions (the last axis must be even).
 *
 * Build: python build.py --src benches/test_fftnd_r2c.c --compile
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex.h>
#include "fftnd_r2c.h"
#include "generator/generated/registry.h"

#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n),64)
#define AFREE(p)  _aligned_free(p)
#else
#define AALLOC(n) aligned_alloc(64,(n))
#define AFREE(p)  free(p)
#endif
#define EPS53 1.1102230246251565e-16

typedef long double complex lcplx;
static const long double LPI = 3.14159265358979323846264338327950288L;
static int g_fail = 0;

/* long-double DFT along axis m of a complex cube (small sizes, O(N^2)) */
static void ref_axis(lcplx *x, int rank, const int *N, int m) {
    size_t K = 1, O = 1;
    for (int i = m + 1; i < rank; i++) K *= (size_t)N[i];
    for (int i = 0; i < m; i++) O *= (size_t)N[i];
    int n = N[m];
    lcplx *tmp = (lcplx *)malloc((size_t)n * sizeof(lcplx));
    for (size_t o = 0; o < O; o++)
        for (size_t l = 0; l < K; l++) {
            lcplx *base = x + o * (size_t)n * K + l;
            for (int k = 0; k < n; k++) {
                lcplx acc = 0;
                for (int j = 0; j < n; j++)
                    acc += base[(size_t)j * K] *
                           cexpl(-2.0L * LPI * I * ((long double)j * k) / n);
                tmp[k] = acc;
            }
            for (int k = 0; k < n; k++) base[(size_t)k * K] = tmp[k];
        }
    free(tmp);
}

/* phase-probe one OUTER axis of the r2c plan (real impulse; pencil at f=0) */
static int *probe_axis(stride_plan_t *p, stride_fftnd_r2c_data_t *d, int m,
                       double *re, double *im) {
    const int N = d->N[m];
    size_t stride = d->hp1;                    /* packed output row stride */
    for (int i = m + 1; i < d->rank - 1; i++) stride *= (size_t)d->N[i];
    memset(re, 0, d->total_real * 8);
    size_t impulse = 1;                        /* axis-m position 1, real  */
    for (int i = m + 1; i < d->rank; i++) impulse *= (size_t)d->N[i];
    re[impulse] = 1.0;
    stride_execute_fwd(p, re, im);
    int *q = malloc((size_t)N * 4), *map = malloc((size_t)N * 4);
    char *seen = calloc((size_t)N, 1);
    int ok = 1;
    for (int j = 0; j < N && ok; j++) {
        double sr = re[(size_t)j * stride], si = im[(size_t)j * stride];
        if (fabs(hypot(sr, si) - 1.0) > 1e-6) { ok = 0; break; }
        double qf = -atan2(si, sr) * (double)N / (2.0 * M_PI);
        long qi = lround(qf);
        if (fabs(qf - (double)qi) > 0.01) { ok = 0; break; }
        int qm = (int)(((qi % N) + N) % N);
        if (seen[qm]) { ok = 0; break; }
        seen[qm] = 1;
        q[j] = qm;
    }
    if (ok) for (int j = 0; j < N; j++) map[q[j]] = j;
    free(q); free(seen);
    if (!ok) { free(map); return NULL; }
    return map;
}

static void cell(int rank, const int *N, const vfft_proto_registry_t *reg) {
    size_t nre = 1; for (int m = 0; m < rank; m++) nre *= (size_t)N[m];
    stride_set_num_threads(1);
    stride_plan_t *p = stride_plan_nd_r2c(rank, N, reg);
    if (!p) { printf("  plan FAIL\n"); g_fail++; return; }
    stride_fftnd_r2c_data_t *d = (stride_fftnd_r2c_data_t *)p->override_data;
    size_t ncx = d->R * d->hp1;

    double *x = AALLOC(nre * 8);
    double *re = AALLOC(nre * 8), *im = AALLOC(ncx * 8);
    srand(101 + N[0] + rank);
    for (size_t i = 0; i < nre; i++) x[i] = 2.0 * rand() / RAND_MAX - 1.0;

    /* gate 1: roundtrip */
    memcpy(re, x, nre * 8);
    stride_execute_fwd(p, re, im);
    double *fr = AALLOC(ncx * 8), *fi = AALLOC(ncx * 8);
    memcpy(fr, re, ncx * 8); memcpy(fi, im, ncx * 8);
    stride_execute_bwd(p, re, im);
    /* Real-data roundtrip metric: relative to the ARRAY scale, not
     * per-element (samples near zero would inflate a per-element ratio by
     * 1/|x_min| ~ Ntotal for uniform inputs; the per-bin gate below is the
     * stronger, externally-referenced check). */
    double sc = (double)nre, rt = 0, xmax = 0;
    for (size_t i = 0; i < nre; i++)
        if (fabs(x[i]) > xmax) xmax = fabs(x[i]);
    for (size_t i = 0; i < nre; i++) {
        double e = fabs(re[i] - sc * x[i]);
        if (e > rt) rt = e;
    }
    rt /= sc * (xmax > 0 ? xmax : 1.0);

    /* gate 2: per-bin vs long-double (probe outer maps + natural-f check) */
    int *maps[FFTND_MAX_RANK] = { 0 };
    int probes_ok = 1;
    for (int m = 0; m < rank - 1 && probes_ok; m++) {
        maps[m] = probe_axis(p, d, m, re, im);
        if (!maps[m]) probes_ok = 0;
    }
    int f_natural = 1;
    if (probes_ok) {              /* impulse at last-axis position 1 */
        memset(re, 0, nre * 8);
        re[1] = 1.0;
        stride_execute_fwd(p, re, im);
        for (size_t f = 0; f < d->hp1; f++) {
            double er = cos(-2.0 * M_PI * (double)f / N[rank-1]);
            double ei = sin(-2.0 * M_PI * (double)f / N[rank-1]);
            if (fabs(re[f] - er) > 1e-6 || fabs(im[f] - ei) > 1e-6) {
                f_natural = 0; break;
            }
        }
    }
    double l2 = 1e30;
    if (probes_ok && f_natural) {
        lcplx *ref = malloc(nre * sizeof(lcplx));
        for (size_t i = 0; i < nre; i++) ref[i] = (long double)x[i];
        for (int m = 0; m < rank; m++) ref_axis(ref, rank, N, m);
        long double e2 = 0, r2 = 0;
        int idx[FFTND_MAX_RANK] = { 0 };
        for (size_t row = 0; row < d->R; row++) {
            /* decompose row -> outer multi-index (natural), map to scrambled */
            size_t t = row, srow = 0, mul = 1;
            for (int m = rank - 2; m >= 0; m--) {
                idx[m] = (int)(t % (size_t)N[m]); t /= (size_t)N[m];
            }
            for (int m = 0; m < rank - 1; m++) {
                srow = srow * (size_t)N[m] + (size_t)maps[m][idx[m]];
                (void)mul;
            }
            for (size_t f = 0; f < d->hp1; f++) {
                size_t rf = 0;   /* natural flat index into full ref cube */
                for (int m = 0; m < rank - 1; m++)
                    rf = rf * (size_t)N[m] + (size_t)idx[m];
                rf = rf * (size_t)N[rank-1] + f;
                long double dr = (long double)fr[srow * d->hp1 + f] - creall(ref[rf]);
                long double di = (long double)fi[srow * d->hp1 + f] - cimagl(ref[rf]);
                e2 += dr * dr + di * di;
                r2 += creall(ref[rf]) * creall(ref[rf]) +
                      cimagl(ref[rf]) * cimagl(ref[rf]);
            }
        }
        l2 = (double)sqrtl(e2 / (r2 > 0 ? r2 : 1));
        free(ref);
    }

    /* gate 4: MT bit-consistency */
    double *r2_ = AALLOC(ncx * 8), *i2_ = AALLOC(ncx * 8);
    int mt_ok = 1;
    int Ts[2] = { 2, 4 };
    for (int ti = 0; ti < 2; ti++) {
        stride_set_num_threads(Ts[ti]);
        memcpy(re, x, nre * 8);
        stride_execute_fwd(p, re, im);
        memcpy(r2_, re, ncx * 8); memcpy(i2_, im, ncx * 8);
        if (memcmp(r2_, fr, ncx * 8) || memcmp(i2_, fi, ncx * 8)) mt_ok = 0;
    }
    stride_set_num_threads(1);

    int ok = rt < 1e-11 && probes_ok && f_natural && l2 < 10 * EPS53 && mt_ok;
    if (!ok) g_fail++;
    printf("  r%d ", rank);
    for (int m = 0; m < rank; m++) printf("%d%s", N[m], m+1<rank?"x":"");
    printf("  rt=%.1e  perbin=%.2e (%4.2f eps)  f-nat=%s  MT-bit=%s  %s\n",
           rt, l2, l2 / EPS53, f_natural ? "Y" : "N",
           mt_ok ? "EXACT" : "NO", ok ? "OK" : "**FAIL**");

    for (int m = 0; m < rank - 1; m++) free(maps[m]);
    AFREE(x); AFREE(re); AFREE(im); AFREE(fr); AFREE(fi);
    AFREE(r2_); AFREE(i2_);
    stride_plan_destroy(p);
}

int main(void) {
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    printf("fftnd r2c/c2r matrix\n");

    int a[2] = { 32, 48 };        cell(2, a, &reg);
    int b[2] = { 61, 32 };        cell(2, b, &reg);   /* prime outer  */
    int c[3] = { 16, 12, 20 };    cell(3, c, &reg);
    int e[3] = { 13, 7, 16 };     cell(3, e, &reg);   /* prime outers */
    int f[3] = { 32, 32, 64 };    cell(3, f, &reg);
    int g[4] = { 8, 12, 10, 16 }; cell(4, g, &reg);
    int h[4] = { 7, 8, 5, 24 };   cell(4, h, &reg);   /* primes mid   */

    printf(g_fail ? "\n%d FAILURE(S)\n" : "\nALL PASS\n", g_fail);
    return g_fail ? 1 : 0;
}
