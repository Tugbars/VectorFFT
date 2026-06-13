/* test_oop_auto.c — phase 4 gates: wisdom-backed auto-create + pair tuner.
 *
 * Auto-create cells span all three kinds with NO factors supplied by the
 * caller (the product-shaped call): LEAF, BAILEY2, and MODEB resolved from
 * the wisdom file. Gates as in test_oop_plan.c (FFTW for natural-order
 * kinds, bit-exact vs generic for MODEB). Then the tuner demo on cells
 * with multiple unmasked pairs, and a hint round-trip (tuner winner fed
 * back through create_auto).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "fftw3.h"
#include "../core/executor.h"
#include "../core/oop_auto.h"

static const char *kname(vfft_oop_kind_t k)
{
    return k == VFFT_OOP_KIND_LEAF ? "LEAF" :
           k == VFFT_OOP_KIND_BAILEY2 ? "BAILEY2" : "MODEB";
}

static double nat_gate(const vfft_oop_plan_t *p, int bwd,
                       const double *sr, const double *si,
                       double *dr, double *di,
                       const double *xr, const double *xi)
{
    const int N = p->N;
    const size_t K = p->K;
    int rc = bwd ? vfft_oop_execute_bwd(p, sr, si, dr, di)
                 : vfft_oop_execute_fwd(p, sr, si, dr, di);
    if (rc)
        return 1.0;
    fftw_complex *fi = fftw_malloc(sizeof(fftw_complex) * N);
    fftw_complex *fo = fftw_malloc(sizeof(fftw_complex) * N);
    fftw_plan pl = fftw_plan_dft_1d(N, fi, fo,
                                    bwd ? FFTW_BACKWARD : FFTW_FORWARD,
                                    FFTW_ESTIMATE);
    double mr = 0;
    for (size_t t = 0; t < K; t += (K > 3 ? K / 3 : 1))
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

static int run_auto(int N, size_t K,
                    const vfft_proto_wisdom_t *wis,
                    const vfft_oop_pair_hint_t *hints, int nh,
                    vfft_proto_registry_t *reg, vfft_oop_kind_t expect)
{
    vfft_oop_plan_t *p =
        vfft_oop_plan_create_auto(N, K, wis, hints, nh, reg);
    if (!p)
    {
        printf("  N=%-5d K=%-4zu auto NULL (expected %s)\n",
               N, K, kname(expect));
        return 1;
    }
    int kok = (p->kind == expect);
    size_t T = (size_t)N * K;
    double *sr = aligned_alloc(64, T * 8), *si = aligned_alloc(64, T * 8);
    double *dr = aligned_alloc(64, T * 8), *di = aligned_alloc(64, T * 8);
    double *xr = malloc(T * 8), *xi = malloc(T * 8);
    srand(29 + N);
    for (size_t t = 0; t < K; t++)
        for (int e = 0; e < N; e++)
        {
            double vr = (double)rand() / RAND_MAX - 0.5;
            double vi = (double)rand() / RAND_MAX - 0.5;
            xr[t * (size_t)N + e] = vr; xi[t * (size_t)N + e] = vi;
            sr[(size_t)e * K + t] = vr; si[(size_t)e * K + t] = vi;
        }

    int ok;
    if (p->kind == VFFT_OOP_KIND_MODEB)
    {
        double *rr = malloc(T * 8), *ri = malloc(T * 8);
        memcpy(rr, sr, T * 8); memcpy(ri, si, T * 8);
        vfft_proto_execute_fwd_generic(p->mb, rr, ri, K);
        int rc = vfft_oop_execute_fwd(p, sr, si, dr, di);
        int g1 = rc == 0 && !memcmp(dr, rr, T * 8) && !memcmp(di, ri, T * 8);
        printf("  N=%-5d K=%-4zu kind %-7s %s | fwd %s (nf=%d)\n",
               N, K, kname(p->kind), kok ? "OK" : "WRONG",
               g1 ? "BITEXACT" : "FAIL", p->mb->num_stages);
        ok = kok && g1;
        free(rr); free(ri);
    }
    else
    {
        double rf = nat_gate(p, 0, sr, si, dr, di, xr, xi);
        double rb = nat_gate(p, 1, sr, si, dr, di, xr, xi);
        printf("  N=%-5d K=%-4zu kind %-7s %s%s | fwd %.1e %s bwd %.1e %s\n",
               N, K, kname(p->kind), kok ? "OK" : "WRONG",
               p->kind == VFFT_OOP_KIND_BAILEY2 ?
                   (p->R1 == p->R2 ? "" : " (uneven)") : "",
               rf, rf < 1e-9 ? "PASS" : "FAIL",
               rb, rb < 1e-9 ? "PASS" : "FAIL");
        ok = kok && rf < 1e-9 && rb < 1e-9;
    }
    free(sr); free(si); free(dr); free(di); free(xr); free(xi);
    vfft_oop_plan_destroy(p);
    return !ok;
}

int main(void)
{
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    vfft_proto_wisdom_t wis;
    if (vfft_proto_wisdom_load(&wis, "/tmp/wis/wisdom_v198.txt") != 0)
    {
        printf("wisdom load FAIL\n");
        return 1;
    }
    printf("wisdom: %zu entries\n", wis.count);
    int bad = 0;

    printf("== auto-create (no factors supplied) ==\n");
    bad += run_auto(64, 128, &wis, NULL, 0, &reg, VFFT_OOP_KIND_LEAF);
    bad += run_auto(169, 256, &wis, NULL, 0, &reg, VFFT_OOP_KIND_BAILEY2);
    bad += run_auto(1024, 120, &wis, NULL, 0, &reg, VFFT_OOP_KIND_BAILEY2);
    bad += run_auto(1024, 256, &wis, NULL, 0, &reg, VFFT_OOP_KIND_BAILEY2);
    bad += run_auto(4096, 256, &wis, NULL, 0, &reg, VFFT_OOP_KIND_MODEB);

    printf("== pair tuner ==\n");
    int r1, r2;
    int nc = vfft_oop_tune_pairs(512, 120, &r1, &r2, 1);
    int nc2 = vfft_oop_tune_pairs(1024, 120, &r1, &r2, 1);
    if (nc < 2 || nc2 < 2)
    {
        printf("  expected multi-candidate cells, got %d and %d\n", nc, nc2);
        bad++;
    }

    printf("== hint round-trip (tuner winner via create_auto) ==\n");
    {
        vfft_oop_pair_hint_t h = {1024, 120, r1, r2};
        vfft_oop_plan_t *p =
            vfft_oop_plan_create_auto(1024, 120, &wis, &h, 1, &reg);
        int ok = p && p->kind == VFFT_OOP_KIND_BAILEY2 &&
                 p->R1 == r1 && p->R2 == r2;
        printf("  hint %dx%d -> %s\n", r1, r2, ok ? "APPLIED" : "FAIL");
        bad += !ok;
        vfft_oop_plan_destroy(p);
    }

    printf(bad ? "SOME GATES FAILED\n" : "ALL GATES PASS\n");
    vfft_proto_wisdom_free(&wis);
    return bad;
}
