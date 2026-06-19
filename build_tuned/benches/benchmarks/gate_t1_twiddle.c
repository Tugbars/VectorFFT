/* gate_t1_twiddle.c — OOP/in-place t1 twiddle-addressing correctness gate.
 *
 * Verifies that the t1 twiddle ADDRESSING (the per-group tw[j*me+b], the
 * sparse log3 slots, and the scalar t1s tw[j]) is numerically correct
 * against a naive DFT, for forward (PRE-twiddle) and backward
 * (POST-twiddle + conj) directions.
 *
 * This exists because docs/codelet_oop.ml:505 carried a stale
 * "twiddle addressing may need follow-up fixup" comment with NO in-tree
 * numeric check (coverage.ml is a generation manifest, not a correctness
 * harness). It is also the REGRESSION GATE for the emit_c-helper
 * extraction (the de-dup of the six "Mirror of emit_c.ml line NNNN"
 * blocks in codelet_oop.ml): run before and after the refactor; output
 * must be unchanged.
 *
 * Conventions covered (see docs/fence_pin_decomposition.md companion and
 * dft.ml twiddle_expr):
 *   flat  (t1)        tw_re[j*me + b]            slot s holds W_N^{s+1}
 *   log3  (t1_log3)   sparse flat slots 2^k-1    SAME fill as flat
 *   t1s               tw_re[j] scalar            SAME values, scalar load
 * The t1p (per-position OOP, tw[j*(me/vw)+b/vw], 9-arg signature) case is
 * covered by gate_t1p_twiddle.c (different signature + fill).
 *
 * Reference structure (from dft.ml dft_expand_twiddled):
 *   DIT fwd = PRE-twiddle : leg k>=1 multiplied by W_N^k, then forward DFT
 *   DIT bwd = POST-twiddle: inverse DFT, then output n>=1 * conj(W_N^n)
 * Comparing against a PLAIN DFT is WRONG and yields phantom failures.
 *
 * Build (one codelet per binary):
 *   gcc-13 -O3 -march=native -ffp-contract=fast \
 *     -DRN=<R> -DFN=<symbol> -DBWD=<0|1> -DT1S=<0|1> \
 *     gate_t1_twiddle.c <codelet.c> -lm -o gate
 * PASS iff max_abs < 1e-9.
 */
#define _POSIX_C_SOURCE 200809L
#define _GNU_SOURCE 1
#define MPI 3.14159265358979323846
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifndef RN
#define RN 16
#endif
#ifndef FN
#define FN radix16_t1_dit_fwd_avx512
#endif
#ifndef BWD
#define BWD 0
#endif
#ifndef T1S
#define T1S 0
#endif
/* 6-arg in-place rio signature shared by t1/t1s/log3. */
extern void FN(double*, double*, const double*, const double*, size_t, size_t);

int main(void) {
    int N = RN;
    size_t me = 8, ios = me;            /* me multiple of AVX-512 width */
    size_t buf = (size_t)N * ios;
    double *re, *im, *tw_re, *tw_im, *re0, *im0;
    posix_memalign((void **)&re, 64, buf * sizeof(double));
    posix_memalign((void **)&im, 64, buf * sizeof(double));
    posix_memalign((void **)&tw_re, 64, ((size_t)N * me + 128) * sizeof(double));
    posix_memalign((void **)&tw_im, 64, ((size_t)N * me + 128) * sizeof(double));
    re0 = malloc(buf * sizeof(double));
    im0 = malloc(buf * sizeof(double));

    /* Twiddle table: slot s in [0,N-1) holds W_N^{s+1} (forward angle).
     * Flat layout serves TP_Flat AND TP_Log3 (identical slot layout;
     * log3 just reads a sparse subset). t1s reads tw_re[s] scalar. */
    for (int s = 0; s < N; s++) {
        double ang = -2.0 * MPI * (double)(s + 1) / (double)N;
        double cr = cos(ang), ci = sin(ang);
        if (T1S) { tw_re[s] = cr; tw_im[s] = ci; }
        else {
            for (size_t b = 0; b < me; b++) {
                tw_re[(size_t)s * me + b] = cr;
                tw_im[(size_t)s * me + b] = ci;
            }
        }
    }

    srand(99);
    for (size_t i = 0; i < buf; i++) {
        re0[i] = (rand() % 2000) / 100.0 - 10.0;
        im0[i] = (rand() % 2000) / 100.0 - 10.0;
    }
    memcpy(re, re0, buf * sizeof(double));
    memcpy(im, im0, buf * sizeof(double));
    FN(re, im, tw_re, tw_im, ios, me);

    double maxabs = 0;
    for (size_t col = 0; col < me; col++) {
        double xr[64], xi[64], outr[64], outi[64];
        for (int j = 0; j < N; j++) {
            xr[j] = re0[(size_t)j * ios + col];
            xi[j] = im0[(size_t)j * ios + col];
        }
#if BWD == 0
        /* DIT fwd = PRE-twiddle: leg k>=1 *= W_N^k, then forward DFT. */
        double tr[64], ti[64];
        tr[0] = xr[0]; ti[0] = xi[0];
        for (int k = 1; k < N; k++) {
            double a = -2.0 * MPI * k / N, c = cos(a), s = sin(a);
            tr[k] = xr[k] * c - xi[k] * s;
            ti[k] = xr[k] * s + xi[k] * c;
        }
        for (int n = 0; n < N; n++) {
            double sr = 0, si = 0;
            for (int k = 0; k < N; k++) {
                double a = -2.0 * MPI * n * k / N, c = cos(a), s = sin(a);
                sr += tr[k] * c - ti[k] * s;
                si += tr[k] * s + ti[k] * c;
            }
            outr[n] = sr; outi[n] = si;
        }
#else
        /* DIT bwd = POST-twiddle + conj: inverse DFT, then out n>=1 * conj(W_N^n). */
        double rr[64], ri[64];
        for (int n = 0; n < N; n++) {
            double sr = 0, si = 0;
            for (int k = 0; k < N; k++) {
                double a = +2.0 * MPI * n * k / N, c = cos(a), s = sin(a);
                sr += xr[k] * c - xi[k] * s;
                si += xr[k] * s + xi[k] * c;
            }
            rr[n] = sr; ri[n] = si;
        }
        outr[0] = rr[0]; outi[0] = ri[0];
        for (int n = 1; n < N; n++) {
            double a = +2.0 * MPI * n / N, c = cos(a), s = sin(a);  /* conj(W^n) */
            outr[n] = rr[n] * c - ri[n] * s;
            outi[n] = rr[n] * s + ri[n] * c;
        }
#endif
        for (int n = 0; n < N; n++) {
            double gr = re[(size_t)n * ios + col], gi = im[(size_t)n * ios + col];
            double e = fabs(gr - outr[n]) + fabs(gi - outi[n]);
            if (e > maxabs) maxabs = e;
        }
    }
    printf("max_abs=%.2e\n", maxabs);
    return (maxabs < 1e-9) ? 0 : 1;
}
