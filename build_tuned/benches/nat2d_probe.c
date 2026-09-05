/* scratch: odd-N 2D IL c2c through the front door vs MKL DFTI 2D, plus the
 * ROW pass alone (N1 rows of the K=1 plan at N2) to expose the column
 * axis's share. Same run, arms alternated, min of 11 rounds, median/min. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "vfft.h"

static double now_ns(void)
{
    static LARGE_INTEGER f; LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}
static vfft_plan mk2(vfft_wisdom *W, int N1, int N2, int nat)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE; cfg.rigor = VFFT_MEASURE;
    cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2; cfg.howmany = 1;
    cfg.order = nat ? VFFT_ORDER_NATURAL : VFFT_ORDER_DEFAULT;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1;
    cfg.wisdom = W; cfg.wisdom_write = 1;
    return vfft_create(&cfg);
}
static vfft_plan mk1(vfft_wisdom *W, int N)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_INPLACE; cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1; cfg.order = VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1;
    cfg.wisdom = W; cfg.wisdom_write = 1;
    return vfft_create(&cfg);
}
typedef struct { double mn, med; } stat_t;
static stat_t stat_of(double *t, int n)
{
    stat_t s; int p, q;
    for (p = 0; p < n; p++) for (q = p + 1; q < n; q++) if (t[q] < t[p]) { double x = t[p]; t[p] = t[q]; t[q] = x; }
    s.mn = t[0]; s.med = t[n / 2]; return s;
}
int main(int argc, char **argv)
{
    static const int C[][2] = { {81, 81}, {243, 243}, {405, 405}, {729, 729}, {243, 1215}, {1215, 243} };
    const int nc = (int)(sizeof C / sizeof C[0]), R = 11;
    vfft_wisdom *W = vfft_wisdom_load(argc > 1 ? argv[1] : ".");
    if (!W) { printf("no wisdom\n"); return 2; }
    mkl_set_num_threads(1);
    printf("%-10s | %9s %9s %9s %9s | vs MKL nat  vs MKL def | rows/2D(nat) | err(nat vs mkl)\n",
           "N1xN2", "mkl ns", "vfft nat", "vfft def", "rows ns");
    for (int i = 0; i < nc; i++) {
        const int N1 = C[i][0], N2 = C[i][1];
        const size_t n = (size_t)N1 * N2, nb = 2 * n * sizeof(double);
        double *x = _aligned_malloc(nb, 64), *y = _aligned_malloc(nb, 64), *m = _aligned_malloc(nb, 64), *a = _aligned_malloc(nb, 64);
        double tm[16], tn[16], td[16], tr[16];
        DFTI_DESCRIPTOR_HANDLE hm = NULL;
        MKL_LONG dims[2] = { N1, N2 };
        for (size_t j = 0; j < 2 * n; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
        if (DftiCreateDescriptor(&hm, DFTI_DOUBLE, DFTI_COMPLEX, 2, dims) != 0 ||
            DftiSetValue(hm, DFTI_PLACEMENT, DFTI_NOT_INPLACE) != 0 || DftiCommitDescriptor(hm) != 0)
        { printf("%dx%d MKL descriptor failed\n", N1, N2); continue; }
        vfft_plan hn = mk2(W, N1, N2, 1), hd = mk2(W, N1, N2, 0), h1 = mk1(W, N2);
        if (!hn || !hd || !h1) { printf("%dx%d create failed (nat %p def %p row %p)\n", N1, N2, (void *)hn, (void *)hd, (void *)h1); continue; }
        /* correctness: natural vs MKL elementwise */
        DftiComputeForward(hm, x, m);
        vfft_execute(hn, VFFT_FORWARD, x, NULL, y, NULL);
        double mx = 0, sc = 0;
        for (size_t j = 0; j < 2 * n; j++) { double d = fabs(y[j] - m[j]); if (d > mx) mx = d; if (fabs(m[j]) > sc) sc = fabs(m[j]); }
        for (int r = 0; r < R; r++) {
            for (int k = 0; k < 4; k++) {
                const int arm = (k + r) % 4;
                double t0;
                if (arm == 0) { t0 = now_ns(); DftiComputeForward(hm, x, m); tm[r] = now_ns() - t0; }
                else if (arm == 1) { t0 = now_ns(); vfft_execute(hn, VFFT_FORWARD, x, NULL, y, NULL); tn[r] = now_ns() - t0; }
                else if (arm == 2) { t0 = now_ns(); vfft_execute(hd, VFFT_FORWARD, x, NULL, y, NULL); td[r] = now_ns() - t0; }
                else {
                    memcpy(a, x, nb);
                    t0 = now_ns();
                    for (int rr = 0; rr < N1; rr++) vfft_execute(h1, VFFT_FORWARD, a + 2 * (size_t)rr * N2, NULL, a + 2 * (size_t)rr * N2, NULL);
                    tr[r] = now_ns() - t0;
                }
            }
        }
        {
            stat_t sm = stat_of(tm, R), sn = stat_of(tn, R), sd = stat_of(td, R), sr = stat_of(tr, R);
            printf("%-10s | %9.0f %9.0f %9.0f %9.0f | %6.2fx      %6.2fx     | %5.2f       | %.1e  (spread mkl %.2f nat %.2f def %.2f rows %.2f)\n",
                   (char[24]){0} + 0 == 0 ? "" : "", sm.mn, sn.mn, sd.mn, sr.mn, sm.mn / sn.mn, sm.mn / sd.mn, sr.mn / sn.mn, mx / sc,
                   sm.med / sm.mn, sn.med / sn.mn, sd.med / sd.mn, sr.med / sr.mn);
            printf("  ^ %dx%d\n", N1, N2);
        }
        DftiFreeDescriptor(&hm);
        vfft_destroy(hn); vfft_destroy(hd); vfft_destroy(h1);
        _aligned_free(x); _aligned_free(y); _aligned_free(m); _aligned_free(a);
    }
    vfft_wisdom_free(W);
    return 0;
}
