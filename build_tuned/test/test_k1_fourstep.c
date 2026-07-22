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
            /* TRUE 2-pass IL: fwd vs naive on z + roundtrip via _sw twins */
            double e_2il = -1, e_2ilrt = -1;
            if (p->il_leaf && p->t1_ul_il) {
                for (int n = 0; n < N; n++) { z[2*n] = xr[n]; z[2*n+1] = xi[n]; }
                vfft_oop_execute_fwd_2p_il(p, z, z);
                e_2il = 0;
                for (int k = 0; k < N; k++) {
                    double c1 = fabs(z[2*k] - nr[k]), c2 = fabs(z[2*k+1] - ni[k]);
                    if (c1 > e_2il) e_2il = c1;
                    if (c2 > e_2il) e_2il = c2;
                }
                if (p->il_leaf_sw && p->t1_ul_il_sw) {
                    vfft_oop_execute_bwd_2p_il(p, z, z);
                    e_2ilrt = 0;
                    for (int n = 0; n < N; n++) {
                        double c1 = fabs(z[2*n] - (double)N * xr[n]);
                        double c2 = fabs(z[2*n+1] - (double)N * xi[n]);
                        if (c1 > e_2ilrt) e_2ilrt = c1;
                        if (c2 > e_2ilrt) e_2ilrt = c2;
                    }
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
            /* LOG3 twins: same Qr/Qi, derived twiddles -> gate vs NAIVE at
             * tol (last-ulp difference vs flat is expected) */
            double e_l3 = -1, e_ull3 = -1;
            if (p->t1_l3) {
                vfft_oop11_fn sv = p->t1p;
                p->t1p = p->t1_l3;
                vfft_oop_execute_fwd(p, xr, xi, rr, ri);
                p->t1p = sv;
                e_l3 = 0;
                for (int k = 0; k < N; k++) {
                    double c1 = fabs(rr[k] - nr[k]), c2 = fabs(ri[k] - ni[k]);
                    if (c1 > e_l3) e_l3 = c1;
                    if (c2 > e_l3) e_l3 = c2;
                }
            }
            if (p->t1_ul_l3 && p->t1_ul) {
                vfft_oop11_fn sv = p->t1_ul;
                p->t1_ul = p->t1_ul_l3;
                vfft_oop_execute_fwd_2pa(p, xr, xi, rr, ri);
                p->t1_ul = sv;
                e_ull3 = 0;
                for (int k = 0; k < N; k++) {
                    double c1 = fabs(rr[k] - nr[k]), c2 = fabs(ri[k] - ni[k]);
                    if (c1 > e_ull3) e_ull3 = c1;
                    if (c2 > e_ull3) e_ull3 = c2;
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
                 (e_2il >= 0 && e_2il > tol) ||
                 (e_2ilrt >= 0 && e_2ilrt > rt_tol) ||
                 (e_l3 >= 0 && e_l3 > tol) ||
                 (e_ull3 >= 0 && e_ull3 > tol) ||
                 e_fwd != e_fwd || e_rt != e_rt) ? "  <FAIL>" : "";
            if (bad[0]) fails++;
            printf("  %4dx%-3d N=%-5d %s fwd=%.2e cross=%.2e rt=%.2e ilf=%.2e ilrt=%.2e 2pa=%.1e 2pb=%.1e twl=%.1e 2il=%.1e 2ilrt=%.1e l3=%.1e ul-l3=%.1e%s\n",
                   R1, R2, N, inp ? "rnd" : "det", e_fwd, e_cross, e_rt, e_ilf, e_ilrt, e_2pa, e_2pb, e_twl, e_2il, e_2ilrt, e_l3, e_ull3, bad);
        }

        afree(xr); afree(xi); afree(nr); afree(ni);
        afree(dr); afree(di); afree(br); afree(bi);
        afree(rr); afree(ri); afree(z);
        if (ref) vfft_oop_plan_destroy(ref);
        vfft_oop_plan_destroy(p);
    }
    /* ---- stride-spec twins (§13.3 A): must be BIT-identical to their
     * unbaked routes (same emission, strides folded to constants) ---- */
    {
        extern void radix32_n1_oop_fwd_avx2_UG_UL_spec32_1_1_32(
            const double *, const double *, double *, double *,
            const double *, const double *, size_t);
        extern void radix32_t1_oop_fwd_avx2_UG_UG_spec32_1_32_1(
            const double *, const double *, double *, double *,
            const double *, const double *, size_t);
        extern void radix64_n1_oop_fwd_avx2_UG_UG_spec64_1_64_1(
            const double *, const double *, double *, double *,
            const double *, const double *, size_t);
        extern void radix64_t1_oop_fwd_avx2_UL_UG_log3_spec1_64_64_1(
            const double *, const double *, double *, double *,
            const double *, const double *, size_t);
        struct { int N, R1, R2, which; } SC[] = { {1024,32,32,0}, {4096,64,64,1} };
        for (int si = 0; si < 2; si++) {
            int N = SC[si].N, R1 = SC[si].R1, R2 = SC[si].R2;
            vfft_oop_plan_t *p = vfft_oop_plan_create_k1(N, R1, R2);
            if (!p) { fails++; continue; }
            double *xr = ad(N), *xi = ad(N), *ar = ad(N), *ai = ad(N);
            double *br = ad(N), *bi = ad(N);
            srand(99 + N);
            for (int n = 0; n < N; n++) {
                xr[n] = (double)rand() / RAND_MAX - 0.5;
                xi[n] = (double)rand() / RAND_MAX - 0.5;
            }
            if (SC[si].which == 0) {
                vfft_oop_execute_fwd_2pb(p, xr, xi, ar, ai);
                radix32_n1_oop_fwd_avx2_UG_UL_spec32_1_1_32(xr, xi, p->col_re, p->col_im, 0, 0, (size_t)R1);
                radix32_t1_oop_fwd_avx2_UG_UG_spec32_1_32_1(p->col_re, p->col_im, br, bi, p->Qr, p->Qi, (size_t)R2);
            } else {
                vfft_oop11_fn sv = p->t1_ul; p->t1_ul = p->t1_ul_l3;
                vfft_oop_execute_fwd_2pa(p, xr, xi, ar, ai);
                p->t1_ul = sv;
                radix64_n1_oop_fwd_avx2_UG_UG_spec64_1_64_1(xr, xi, p->col_re, p->col_im, 0, 0, (size_t)R1);
                radix64_t1_oop_fwd_avx2_UL_UG_log3_spec1_64_64_1(p->col_re, p->col_im, br, bi, p->Qr, p->Qi, (size_t)R2);
            }
            double e = 0;
            for (int k = 0; k < N; k++) {
                double c1 = fabs(br[k] - ar[k]), c2 = fabs(bi[k] - ai[k]);
                if (c1 > e) e = c1;
                if (c2 > e) e = c2;
            }
            const char *bad = (e > 0.0) ? "  <FAIL>" : "";
            if (bad[0]) fails++;
            printf("  SPEC %dx%d N=%d vs unbaked route = %.1e%s\n", R1, R2, N, e, bad);
            afree(xr); afree(xi); afree(ar); afree(ai); afree(br); afree(bi);
            vfft_oop_plan_destroy(p);
        }
    }

    /* ---- K1 MONO gates (emitted whole-four-step codelets, M1+M3) ---- */
    int monoNs[] = { 64, 128, 256, -128 };  /* -N = alt pair */
    for (int mi = 0; mi < 4; mi++) {
        int N = monoNs[mi] > 0 ? monoNs[mi] : -monoNs[mi];
        vfft_oop11_fn mono = monoNs[mi] > 0 ? vfft_k1_mono_fn(N)
                                            : vfft_k1_mono_alt_fn(N);
        if (mono) {
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
                printf("  K1MONO%d%s %s vs naive = %.2e%s\n", N,
                       monoNs[mi] < 0 ? "-alt" : "", inp ? "rnd" : "det", e, bad);
            }
            afree(xr); afree(xi); afree(nr); afree(ni); afree(dr); afree(di);
        }
    }

    /* ---- mono-64 IL twins (M2 fwd + M4 bwd) + split-bwd swap identity ---- */
    {
        vfft_oop11_fn mf = vfft_k1_mono_il_fn(64, 0);
        vfft_oop11_fn mb = vfft_k1_mono_il_fn(64, 1);
        vfft_oop11_fn ms = vfft_k1_mono_fn(64);
        if (mf && mb && ms) {
            int N = 64;
            double tol = 1e-9 * (double)N, rt_tol = tol * (double)N;
            double *xr = ad(N), *xi = ad(N), *nr = ad(N), *ni = ad(N);
            double *dr = ad(N), *di = ad(N), *z = ad((size_t)2 * N);
            for (int inp = 0; inp < 2; inp++) {
                if (inp == 0)
                    for (int n = 0; n < N; n++) {
                        xr[n] = cos(0.7 * n) + 0.1;
                        xi[n] = sin(1.3 * n) - 0.05;
                    }
                else {
                    srand(64642);
                    for (int n = 0; n < N; n++) {
                        xr[n] = (double)rand() / RAND_MAX - 0.5;
                        xi[n] = (double)rand() / RAND_MAX - 0.5;
                    }
                }
                naive_dft(N, xr, xi, nr, ni);
                /* IL fwd vs naive */
                for (int n = 0; n < N; n++) { z[2*n] = xr[n]; z[2*n+1] = xi[n]; }
                mf(z, 0, z, 0, 0, 0, 0, 0, 0, 0, 0);
                double ef = 0;
                for (int k = 0; k < N; k++) {
                    double c1 = fabs(z[2*k] - nr[k]), c2 = fabs(z[2*k+1] - ni[k]);
                    if (c1 > ef) ef = c1;
                    if (c2 > ef) ef = c2;
                }
                /* IL roundtrip: bwd(fwd(x)) == N*x */
                mb(z, 0, z, 0, 0, 0, 0, 0, 0, 0, 0);
                double ert = 0;
                for (int n = 0; n < N; n++) {
                    double c1 = fabs(z[2*n] - (double)N * xr[n]);
                    double c2 = fabs(z[2*n+1] - (double)N * xi[n]);
                    if (c1 > ert) ert = c1;
                    if (c2 > ert) ert = c2;
                }
                /* split bwd via the pointer-swap identity on the fwd mono:
                 * IDFT(X) = swap(DFT(swap(X))) — feed the naive spectrum
                 * swapped, read the result swapped: should give N*x. */
                double eb = 0;
                ms(ni, nr, di, dr, 0, 0, 0, 0, 0, 0, 0);
                for (int n = 0; n < N; n++) {
                    double c1 = fabs(dr[n] - (double)N * xr[n]);
                    double c2 = fabs(di[n] - (double)N * xi[n]);
                    if (c1 > eb) eb = c1;
                    if (c2 > eb) eb = c2;
                }
                const char *bad = (ef > tol || ert > rt_tol || eb > rt_tol ||
                                   ef != ef || ert != ert || eb != eb) ? "  <FAIL>" : "";
                if (bad[0]) fails++;
                printf("  K1MONO64-IL %s fwd=%.2e ilrt=%.2e split-bwd=%.2e%s\n",
                       inp ? "rnd" : "det", ef, ert, eb, bad);
            }
            afree(xr); afree(xi); afree(nr); afree(ni);
            afree(dr); afree(di); afree(z);
        }
    }

    /* ---- CCOL route (§12.4 item 5: composed column pass) ----
     * OOP fwd vs naive; IN-PLACE fwd bit-identical to OOP (colp fully drains
     * src before the transpose writes it); roundtrip via the pointer-swap
     * identity; chain encode/decode round-trip. Includes 16384 — the first
     * cell past the leaf ceiling — and a single-stage chain (identity perm). */
    {
        struct { int N, R1, nf, chain[4]; } cc[] = {
            {1024,  64, 1, {16}},
            {2048,  64, 2, {8, 4}},
            {4096,  64, 2, {8, 8}},
            {8192,  64, 2, {8, 16}},
            {16384, 64, 3, {8, 8, 4}},
        };
        vfft_proto_registry_t creg;
        vfft_proto_registry_init(&creg);
        for (int ci = 0; ci < (int)(sizeof cc / sizeof cc[0]); ci++) {
            int N = cc[ci].N;
            double tol = 1e-9 * (double)N, rt_tol = tol * (double)N;
            int code = vfft_k1_cc_chain_encode(cc[ci].chain, cc[ci].nf);
            int dchain[6], dnf = vfft_k1_cc_chain_decode(code, dchain);
            int enc_ok = (dnf == cc[ci].nf);
            for (int s = 0; enc_ok && s < dnf; s++)
                if (dchain[s] != cc[ci].chain[s]) enc_ok = 0;
            if (!enc_ok) {
                printf("  K1CCOL %d chain encode/decode MISMATCH  <FAIL>\n", N);
                fails++; continue;
            }
            vfft_oop_plan_t *p = vfft_oop_plan_create_k1_cc(
                N, cc[ci].R1, cc[ci].chain, cc[ci].nf, &creg);
            if (!p) {
                printf("  K1CCOL %d create=NULL  <FAIL>\n", N);
                fails++; continue;
            }
            double *xr = ad(N), *xi = ad(N), *nr = ad(N), *ni = ad(N);
            double *dr = ad(N), *di = ad(N), *wr = ad(N), *wi = ad(N);
            double *rr = ad(N), *ri = ad(N);
            for (int inp = 0; inp < 2; inp++) {
                if (inp == 0)
                    for (int n = 0; n < N; n++) {
                        xr[n] = cos(0.7 * n) + 0.1;
                        xi[n] = sin(1.3 * n) - 0.05;
                    }
                else {
                    srand(999 + N);
                    for (int n = 0; n < N; n++) {
                        xr[n] = (double)rand() / RAND_MAX - 0.5;
                        xi[n] = (double)rand() / RAND_MAX - 0.5;
                    }
                }
                naive_dft(N, xr, xi, nr, ni);
                /* OOP fwd vs naive */
                vfft_oop_execute_fwd_ccol(p, xr, xi, dr, di);
                double ef = 0;
                for (int k = 0; k < N; k++) {
                    double c1 = fabs(dr[k] - nr[k]), c2 = fabs(di[k] - ni[k]);
                    if (c1 > ef) ef = c1;
                    if (c2 > ef) ef = c2;
                }
                /* IN-PLACE fwd == OOP fwd, bit (0.0) */
                memcpy(wr, xr, (size_t)N * 8);
                memcpy(wi, xi, (size_t)N * 8);
                vfft_oop_execute_fwd_ccol(p, wr, wi, wr, wi);
                double eip = 0;
                for (int k = 0; k < N; k++) {
                    double c1 = fabs(wr[k] - dr[k]), c2 = fabs(wi[k] - di[k]);
                    if (c1 > eip) eip = c1;
                    if (c2 > eip) eip = c2;
                }
                /* roundtrip via the split pointer-swap identity */
                vfft_oop_execute_fwd_ccol(p, di, dr, ri, rr);
                double ert = 0;
                for (int n = 0; n < N; n++) {
                    double c1 = fabs(rr[n] - (double)N * xr[n]);
                    double c2 = fabs(ri[n] - (double)N * xi[n]);
                    if (c1 > ert) ert = c1;
                    if (c2 > ert) ert = c2;
                }
                const char *bad = (ef > tol || eip != 0.0 || ert > rt_tol ||
                                   ef != ef || ert != ert) ? "  <FAIL>" : "";
                if (bad[0]) fails++;
                printf("  K1CCOL %-5d [%d", N, cc[ci].chain[0]);
                for (int s = 1; s < cc[ci].nf; s++) printf(",%d", cc[ci].chain[s]);
                printf("] %s fwd=%.2e ip-bit=%.1f rt=%.2e%s\n",
                       inp ? "rnd" : "det", ef, eip, ert, bad);
            }
            afree(xr); afree(xi); afree(nr); afree(ni);
            afree(dr); afree(di); afree(wr); afree(wi); afree(rr); afree(ri);
            vfft_oop_plan_destroy(p);
        }
    }

    printf("\n%s (%d fail)\n", fails ? "FAILURES" : "BAILEY2V: ALL GATES GREEN (split+IL, fwd+bwd, det+rnd)", fails);
    return fails ? 1 : 0;
}
