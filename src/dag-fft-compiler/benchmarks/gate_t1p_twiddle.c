/* gate_t1p_twiddle.c — per-position (t1p) OOP twiddle-addressing gate.
 *
 * Covers the PerPositionTwiddles convention that gate_t1_twiddle.c does
 * NOT: the OOP path renders tw_re[j*(me/vw) + b/vw] (one twiddle value
 * per vw-lane block per leg) and uses the 9-arg in-place-OOP signature.
 * The me % vw != 0 integer-division boundary is the risk this gate
 * guards. vw = 8 for AVX-512.
 *
 * Reference structure identical to gate_t1_twiddle.c:
 *   fwd = PRE-twiddle (leg k>=1 *= W_N^k, then forward DFT)
 *   bwd = POST-twiddle + conj
 *
 * Build:
 *   gcc-13 -O3 -march=native -ffp-contract=fast \
 *     -DRN=<R> -DFN=<symbol> -DBWD=<0|1> \
 *     gate_t1p_twiddle.c <oop_t1p_codelet.c> -lm -o gate
 * Codelet generated with:
 *   gen_radix.exe R --oop --twiddled-pos [--bwd] --oop-load UG --oop-store UG --isa avx512 --emit-c
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
#define FN radix16_t1p_inplace_fwd_avx512_UG_UG
#endif
#ifndef BWD
#define BWD 0
#endif
/* 9-arg in-place-OOP signature: (rio_re, rio_im, tw_re, tw_im,
 * in_leg_stride, in_group_stride, out_leg_stride, out_group_stride, me). */
extern void FN(double *, double *, const double *, const double *,
               size_t, size_t, size_t, size_t, size_t);

int main(void) {
    int N = RN;
    size_t me = 16;            /* multiple of vw; >vw so b/vw spans >1 block */
    int vw = 8;
    size_t ils = me, igs = 1, ols = me, ogs = 1;  /* UG, in-place */
    size_t buf = (size_t)N * me;
    double *re, *im, *tw_re, *tw_im, *re0, *im0;
    posix_memalign((void **)&re, 64, buf * sizeof(double));
    posix_memalign((void **)&im, 64, buf * sizeof(double));
    size_t twn = (size_t)N * (me / vw) + 64;
    posix_memalign((void **)&tw_re, 64, twn * sizeof(double));
    posix_memalign((void **)&tw_im, 64, twn * sizeof(double));
    re0 = malloc(buf * sizeof(double));
    im0 = malloc(buf * sizeof(double));

    /* per-position fill: leg j (0-based; actual leg = j+1, value W_N^{j+1}),
     * block bb in [0, me/vw): slot j*(me/vw) + bb. All vw lanes in a block
     * share one twiddle value (constant across the vw-batch). */
    for (int j = 0; j < N; j++) {
        double ang = -2.0 * MPI * (double)(j + 1) / (double)N;
        double cr = cos(ang), ci = sin(ang);
        for (size_t bb = 0; bb < me / vw; bb++) {
            tw_re[(size_t)j * (me / vw) + bb] = cr;
            tw_im[(size_t)j * (me / vw) + bb] = ci;
        }
    }

    srand(99);
    for (size_t i = 0; i < buf; i++) {
        re0[i] = (rand() % 2000) / 100.0 - 10.0;
        im0[i] = (rand() % 2000) / 100.0 - 10.0;
    }
    memcpy(re, re0, buf * sizeof(double));
    memcpy(im, im0, buf * sizeof(double));
    FN(re, im, tw_re, tw_im, ils, igs, ols, ogs, me);

    double maxabs = 0;
    for (size_t b = 0; b < me; b++) {
        double xr[64], xi[64], outr[64], outi[64];
        for (int j = 0; j < N; j++) {
            xr[j] = re0[(size_t)j * ils + b];
            xi[j] = im0[(size_t)j * ils + b];
        }
#if BWD == 0
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
            double a = +2.0 * MPI * n / N, c = cos(a), s = sin(a);
            outr[n] = rr[n] * c - ri[n] * s;
            outi[n] = rr[n] * s + ri[n] * c;
        }
#endif
        for (int n = 0; n < N; n++) {
            double gr = re[(size_t)n * ols + b], gi = im[(size_t)n * ols + b];
            double e = fabs(gr - outr[n]) + fabs(gi - outi[n]);
            if (e > maxabs) maxabs = e;
        }
    }
    printf("max_abs=%.2e\n", maxabs);
    return (maxabs < 1e-9) ? 0 : 1;
}
