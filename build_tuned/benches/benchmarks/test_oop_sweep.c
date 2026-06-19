/* test_oop_sweep.c — phase 5 validation sweep: auto-create across the
 * production wisdom table. For every sampled (N, K) entry: create with no
 * caller factors, gate by kind (MODEB: bit-exact vs the in-place generic
 * dataflow; LEAF/BAILEY2: vs FFTW at <=1e-9 on sampled transforms).
 * Cells with N*K beyond the element cap are skipped (container memory).
 * Usage: test_oop_sweep [stride] [cap_elems]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fftw3.h"
#include "../core/executor.h"
#include "../core/oop_auto.h"

static double nat_gate(const vfft_oop_plan_t *p,
                       const double *sr, const double *si,
                       double *dr, double *di,
                       const double *xr, const double *xi)
{
    const int N = p->N;
    const size_t K = p->K;
    if (vfft_oop_execute_fwd(p, sr, si, dr, di))
        return 1.0;
    fftw_complex *fi = fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *fo = fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan pl = fftw_plan_dft_1d(N, fi, fo, FFTW_FORWARD, FFTW_ESTIMATE);
    double mr = 0;
    size_t step = K > 2 ? K / 2 : 1;
    for (size_t t = 0; t < K; t += step)
    {
        for (int e = 0; e < N; e++)
        {
            fi[e][0] = xr[t * (size_t)N + e];
            fi[e][1] = xi[t * (size_t)N + e];
        }
        fftw_execute(pl);
        double me = 0, mm = 0;
        for (int k = 0; k < N; k++)
        {
            double a = dr[(size_t)k * K + t] - fo[k][0];
            double b = di[(size_t)k * K + t] - fo[k][1];
            double e2 = sqrt(a * a + b * b), m = hypot(fo[k][0], fo[k][1]);
            if (e2 > me) me = e2;
            if (m > mm) mm = m;
        }
        if (mm > 0 && me / mm > mr) mr = me / mm;
    }
    fftw_destroy_plan(pl); fftw_free(fi); fftw_free(fo);
    return mr;
}

int main(int argc, char **argv)
{
    int stride = argc > 1 ? atoi(argv[1]) : 4;
    size_t cap = argc > 2 ? (size_t)atol(argv[2]) : 4u << 20;
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    vfft_proto_wisdom_t wis;
    if (vfft_proto_wisdom_load(&wis, "/tmp/wis/wisdom_v198.txt") != 0)
    {
        printf("wisdom load FAIL\n");
        return 1;
    }
    int tested = 0, skipped = 0, sk_k = 0, fails = 0;
    int n_leaf = 0, n_bailey = 0, n_modeb = 0, n_null = 0, n_dif = 0;

    for (size_t i = 0; i < wis.count; i += (size_t)stride)
    {
        const vfft_proto_wisdom_entry_t *e = &wis.entries[i];
        const int N = e->N;
        const size_t K = e->K;
        if ((size_t)N * K > cap)
        {
            skipped++;
            continue;
        }
        if ((K % 8u) != 0)
        {
            sk_k++; /* below the lane contract; v1 unsupported */
            continue;
        }
        vfft_oop_plan_t *p =
            vfft_oop_plan_create_auto(N, K, &wis, NULL, 0, &reg);
        if (!p)
        {
            printf("  N=%-7d K=%-5zu NULL plan\n", N, K);
            n_null++;
            fails++;
            continue;
        }
        if (e->use_dif_forward)
            n_dif++;

        size_t T = (size_t)N * K;
        double *sr = aligned_alloc(64, T * 8), *si = aligned_alloc(64, T * 8);
        double *dr = aligned_alloc(64, T * 8), *di = aligned_alloc(64, T * 8);
        double *xr = malloc(T * 8), *xi = malloc(T * 8);
        srand(101 + N + (int)K);
        for (size_t t = 0; t < K; t++)
            for (int n = 0; n < N; n++)
            {
                double vr = (double)rand() / RAND_MAX - 0.5;
                double vi = (double)rand() / RAND_MAX - 0.5;
                xr[t * (size_t)N + n] = vr; xi[t * (size_t)N + n] = vi;
                sr[(size_t)n * K + t] = vr; si[(size_t)n * K + t] = vi;
            }
        int ok = 1;
        if (p->kind == VFFT_OOP_KIND_MODEB)
        {
            n_modeb++;
            double *rr = malloc(T * 8), *ri = malloc(T * 8);
            memcpy(rr, sr, T * 8); memcpy(ri, si, T * 8);
            vfft_proto_execute_fwd_generic(p->mb, rr, ri, K);
            ok = vfft_oop_execute_fwd(p, sr, si, dr, di) == 0 &&
                 !memcmp(dr, rr, T * 8) && !memcmp(di, ri, T * 8);
            if (!ok)
                printf("  N=%-7d K=%-5zu MODEB FAIL\n", N, K);
            free(rr); free(ri);
        }
        else
        {
            if (p->kind == VFFT_OOP_KIND_LEAF) n_leaf++; else n_bailey++;
            double r = nat_gate(p, sr, si, dr, di, xr, xi);
            ok = r < 1e-9;
            if (!ok)
                printf("  N=%-7d K=%-5zu %s FAIL %.1e\n", N, K,
                       p->kind == VFFT_OOP_KIND_LEAF ? "LEAF" : "BAILEY2", r);
        }
        fails += !ok;
        tested++;
        free(sr); free(si); free(dr); free(di); free(xr); free(xi);
        vfft_oop_plan_destroy(p);
    }
    printf("sweep: %d tested (%d leaf, %d bailey2, %d modeb; %d dif-pref "
           "cells ran DIT), %d skipped cap, %d skipped K-contract, "
           "%d null, %d FAIL\n",
           tested, n_leaf, n_bailey, n_modeb, n_dif, skipped, sk_k, n_null,
           fails);
    printf(fails || n_null ? "SWEEP FAILED\n" : "SWEEP PASS\n");
    vfft_proto_wisdom_free(&wis);
    return fails || n_null;
}
