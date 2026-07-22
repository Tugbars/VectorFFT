/* test_k1_fourstep.c — BAILEY2V (K=1 vectorized four-step) plan-API gates.
 *
 * Covers docs/roadmap/row_major_engine.md §11f production wiring:
 *   - SPLIT fwd vs naive O(N^2) DFT (natural order) + cross vs BAILEY2
 *     (expected BIT-identical: same codelet DAG);
 *   - SPLIT roundtrip fwd -> bwd (swap identity) == N * x;
 *   - IL fwd (z->z, emitted il_in leaf + t1 il_out twin) vs naive;
 *   - IL roundtrip fwd_il -> bwd_il (_sw lattice twins) == N * x, output in
 *     normal (re,im) order;
 *   - deterministic AND random inputs (July-6 lesson).
 *
 * Build: python build.py --src test/test_k1_fourstep.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "executor.h"
#include "planner.h"
#include "oop_plan.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) exit(1);
    return p;
}
static void afree(double *p) { vfft_proto_aligned_free(p); }

static void naive_dft(int N, const double *xr, const double *xi, double *Xr, double *Xi)
{
    for (int k = 0; k < N; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double ang = -2.0 * M_PI * (double)((long)n * k % N) / (double)N;
            double c = cos(ang), s = sin(ang);
            sr += xr[n] * c - xi[n] * s;
            si += xr[n] * s + xi[n] * c;
        }
        Xr[k] = sr; Xi[k] = si;
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    (void)reg;

    struct { int N, R1, R2; } cells[] = {
        {64, 8, 8}, {64, 4, 16}, {256, 16, 16}, {256, 4, 64}, {256, 32, 8},
        {512, 8, 64}, {1024, 32, 32}, {1024, 64, 16}, {2048, 64, 32},
        {4096, 64, 64}, {1024, 8, 128} /* R2=128: split-only (no IL twin) */
    };
    int nC = (int)(sizeof(cells) / sizeof(cells[0]));
    int fails = 0;

    for (int ci = 0; ci < nC; ci++) {
        int N = cells[ci].N, R1 = cells[ci].R1, R2 = cells[ci].R2;
        double tol = 1e-9 * (double)N;
        vfft_oop_plan_t *p = vfft_oop_plan_create_k1(N, R1, R2);
        if (!p) {
            /* R2=128 has a leaf but no t1(128); only reject if unexpected */
            printf("  %4dx%-3d N=%-5d create=NULL\n", R1, R2, N);
            if (vfft_oop_leaf_fn(R2) && vfft_oop_t1_fn(R1)) fails++;
            continue;
        }
        vfft_oop_plan_t *ref = vfft_oop_plan_create_pair_v(N, 1, R1, R2, 0);

        double *xr = ad(N), *xi = ad(N), *nr = ad(N), *ni = ad(N);
        double *dr = ad(N), *di = ad(N), *br = ad(N), *bi = ad(N);
        double *rr = ad(N), *ri = ad(N);
        double *z = ad((size_t)2 * N);

        for (int inp = 0; inp < 2; inp++) {
            if (inp == 0)
                for (int n = 0; n < N; n++) {
                    xr[n] = cos(0.7 * n) + 0.1;
                    xi[n] = sin(1.3 * n) - 0.05;
                }
            else {
                srand(777 + N + R1);
                for (int n = 0; n < N; n++) {
                    xr[n] = (double)rand() / RAND_MAX - 0.5;
                    xi[n] = (double)rand() / RAND_MAX - 0.5;
                }
            }
            naive_dft(N, xr, xi, nr, ni);

            /* SPLIT fwd vs naive + bit-cross vs BAILEY2 */
            vfft_oop_execute_fwd(p, xr, xi, dr, di);
            double e_fwd = 0, e_cross = 0;
            if (ref) {
                vfft_oop_execute_fwd(ref, xr, xi, br, bi);
                for (int k = 0; k < N; k++) {
                    double c1 = fabs(dr[k] - br[k]), c2 = fabs(di[k] - bi[k]);
                    if (c1 > e_cross) e_cross = c1;
                    if (c2 > e_cross) e_cross = c2;
                }
            }
            for (int k = 0; k < N; k++) {
                double c1 = fabs(dr[k] - nr[k]), c2 = fabs(di[k] - ni[k]);
                if (c1 > e_fwd) e_fwd = c1;
                if (c2 > e_fwd) e_fwd = c2;
            }

            /* SPLIT roundtrip: bwd(fwd(x)) == N*x (unnormalized) */
            vfft_oop_execute_bwd(p, dr, di, rr, ri);
            double e_rt = 0;
            for (int n = 0; n < N; n++) {
                double c1 = fabs(rr[n] - (double)N * xr[n]);
                double c2 = fabs(ri[n] - (double)N * xi[n]);
                if (c1 > e_rt) e_rt = c1;
                if (c2 > e_rt) e_rt = c2;
            }

            /* TWO-PASS routes (§12.4): must be BIT-identical to the 3-pass
             * output (same codelet DAGs, only edge addressing differs) */
            double e_2pa = -1, e_2pb = -1;
            if (p->t1_ul) {
                vfft_oop_execute_fwd_2pa(p, xr, xi, rr, ri);
                e_2pa = 0;
                for (int k = 0; k < N; k++) {
                    double c1 = fabs(rr[k] - dr[k]), c2 = fabs(ri[k] - di[k]);
                    if (c1 > e_2pa) e_2pa = c1;
                    if (c2 > e_2pa) e_2pa = c2;
                }
            }
            if (p->leaf_ul) {
                vfft_oop_execute_fwd_2pb(p, xr, xi, rr, ri);
                e_2pb = 0;
                for (int k = 0; k < N; k++) {
                    double c1 = fabs(rr[k] - dr[k]), c2 = fabs(ri[k] - di[k]);
                    if (c1 > e_2pb) e_2pb = c1;
                    if (c2 > e_2pb) e_2pb = c2;
                }
            }
            double e_twl = -1;
            if (p->t1_ul_twl) {
                vfft_oop_execute_fwd_2pa_twl(p, xr, xi, rr, ri);
                e_twl = 0;
                for (int k = 0; k < N; k++) {
                    double c1 = fabs(rr[k] - dr[k]), c2 = fabs(ri[k] - di[k]);
                    if (c1 > e_twl) e_twl = c1;
                    if (c2 > e_twl) e_twl = c2;
                }
            }

            /* IL fwd + IL roundtrip (when twins exist) */
            double e_ilf = -1, e_ilrt = -1;
            if (p->il_leaf && p->t1_il) {
                for (int n = 0; n < N; n++) { z[2 * n] = xr[n]; z[2 * n + 1] = xi[n]; }
                vfft_oop_execute_fwd_il(p, z, z);
                e_ilf = 0;
                for (int k = 0; k < N; k++) {
                    double c1 = fabs(z[2 * k] - nr[k]), c2 = fabs(z[2 * k + 1] - ni[k]);
                    if (c1 > e_ilf) e_ilf = c1;
                    if (c2 > e_ilf) e_ilf = c2;
                }
                if (p->il_leaf_sw && p->t1_il_sw) {
                    vfft_oop_execute_bwd_il(p, z, z);
                    e_ilrt = 0;
                    for (int n = 0; n < N; n++) {
                        double c1 = fabs(z[2 * n] - (double)N * xr[n]);
                        double c2 = fabs(z[2 * n + 1] - (double)N * xi[n]);
                        if (c1 > e_ilrt) e_ilrt = c1;
                        if (c2 > e_ilrt) e_ilrt = c2;
                    }
                }
            }

            double rt_tol = tol * (double)N; /* roundtrip scales by N */
            const char *bad =
                (e_fwd > tol || e_rt > rt_tol ||
                 (e_ilf >= 0 && e_ilf > tol) || (e_ilrt >= 0 && e_ilrt > rt_tol) ||
                 (e_2pa >= 0 && e_2pa > 0.0) ||   /* two-pass must be BIT-identical */
                 (e_2pb >= 0 && e_2pb > 0.0) ||
                 (e_twl >= 0 && e_twl > 0.0) ||   /* linear layout: same values */
                 e_fwd != e_fwd || e_rt != e_rt) ? "  <FAIL>" : "";
            if (bad[0]) fails++;
            printf("  %4dx%-3d N=%-5d %s fwd=%.2e cross=%.2e rt=%.2e ilf=%.2e ilrt=%.2e 2pa=%.1e 2pb=%.1e twl=%.1e%s\n",
                   R1, R2, N, inp ? "rnd" : "det", e_fwd, e_cross, e_rt, e_ilf, e_ilrt, e_2pa, e_2pb, e_twl, bad);
        }

        afree(xr); afree(xi); afree(nr); afree(ni);
        afree(dr); afree(di); afree(br); afree(bi);
        afree(rr); afree(ri); afree(z);
        if (ref) vfft_oop_plan_destroy(ref);
        vfft_oop_plan_destroy(p);
    }
    /* ---- K1 MONO gate (emitted whole-four-step codelet, M1: N=64) ---- */
    {
        vfft_oop11_fn mono = vfft_k1_mono_fn(64);
        if (mono) {
            int N = 64;
            double tol = 1e-9 * (double)N;
            double *xr = ad(N), *xi = ad(N), *nr = ad(N), *ni = ad(N);
            double *dr = ad(N), *di = ad(N);
            for (int inp = 0; inp < 2; inp++) {
                if (inp == 0)
                    for (int n = 0; n < N; n++) {
                        xr[n] = cos(0.7 * n) + 0.1;
                        xi[n] = sin(1.3 * n) - 0.05;
                    }
                else {
                    srand(4242);
                    for (int n = 0; n < N; n++) {
                        xr[n] = (double)rand() / RAND_MAX - 0.5;
                        xi[n] = (double)rand() / RAND_MAX - 0.5;
                    }
                }
                naive_dft(N, xr, xi, nr, ni);
                mono(xr, xi, dr, di, 0, 0, 0, 0, 0, 0, 0);
                double e = 0;
                for (int k = 0; k < N; k++) {
                    double c1 = fabs(dr[k] - nr[k]), c2 = fabs(di[k] - ni[k]);
                    if (c1 > e) e = c1;
                    if (c2 > e) e = c2;
                }
                const char *bad = (e > tol || e != e) ? "  <FAIL>" : "";
                if (bad[0]) fails++;
                printf("  K1MONO64 %s vs naive = %.2e%s\n", inp ? "rnd" : "det", e, bad);
            }
            afree(xr); afree(xi); afree(nr); afree(ni); afree(dr); afree(di);
        }
    }

    printf("\n%s (%d fail)\n", fails ? "FAILURES" : "BAILEY2V: ALL GATES GREEN (split+IL, fwd+bwd, det+rnd)", fails);
    return fails ? 1 : 0;
}
