/* gate_inplace_full.c — full-coverage naive-DFT correctness gate for the
 * in-place C2C codelet families, FFTW-free. Generalizes gate_t1_twiddle.c
 * to also cover n1 (no-twiddle leaf) and DIF orientation.
 *
 * Reference conventions (must match the generated codelets exactly, or you
 * get phantom failures — see docs/codelet_findings/dif_convention_finding.md):
 *
 *   MODE=0 n1  : plain radix-N DFT, no twiddle.
 *   MODE=1 DIT : fwd = PRE-tw (leg k>=1 *= W^k) then DFT;
 *                bwd = IDFT then POST-tw-conj (out n>=1 *= conj(W^n)).
 *   MODE=2 DIF : fwd = DFT then POST-tw (out n>=1 *= W^n);
 *                bwd = PRE-tw-conj (leg k>=1 *= conj(W^k)) then IDFT.
 *
 *   W^m = exp(-2*pi*i*m/N) (forward angle). Backward DFT uses +angle.
 *   Twiddle table slot s in [0,N-1) holds W^{s+1}; bwd codelets read the
 *   same fwd-filled table and conjugate internally. t1s reads tw[s] scalar.
 *
 * Config macros: RN (radix), FN (symbol), MODE (0/1/2), BWD (0/1), T1S (0/1).
 * PASS iff max_abs < 1e-9.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define MPI 3.14159265358979323846

#ifndef RN
#define RN 16
#endif
#ifndef FN
#define FN radix16_n1_fwd_avx2
#endif
#ifndef MODE
#define MODE 0          /* 0=n1, 1=DIT, 2=DIF */
#endif
#ifndef BWD
#define BWD 0
#endif
#ifndef T1S
#define T1S 0
#endif

/* 6-arg in-place rio signature shared by n1/t1/t1s/log3. */
extern void FN(double*, double*, const double*, const double*, size_t, size_t);

int main(void) {
    int N = RN;
    size_t me = 8, ios = me;
    size_t buf = (size_t)N * ios;
    double *re, *im, *tw_re, *tw_im, *re0, *im0;
    posix_memalign((void **)&re, 64, buf * sizeof(double));
    posix_memalign((void **)&im, 64, buf * sizeof(double));
    posix_memalign((void **)&tw_re, 64, ((size_t)N * me + 128) * sizeof(double));
    posix_memalign((void **)&tw_im, 64, ((size_t)N * me + 128) * sizeof(double));
    re0 = malloc(buf * sizeof(double));
    im0 = malloc(buf * sizeof(double));

    /* Twiddle table: slot s holds W_N^{s+1} (forward angle). Flat layout
     * serves t1/log3; t1s reads tw_re[s] scalar. (Unused for n1.) */
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

    /* DFT sign: forward uses -angle; backward (inverse, unnormalized) +angle.
     * n1/DIF-fwd/DIT-fwd use forward DFT; DIT-bwd/DIF-bwd use backward DFT. */
#if (MODE == 0)
    int dft_bwd = BWD;                 /* n1: fwd=forward DFT, bwd=inverse DFT */
#elif (MODE == 1)
    int dft_bwd = BWD;                 /* DIT: same */
#else
    int dft_bwd = BWD;                 /* DIF: same */
#endif
    double dsign = dft_bwd ? +1.0 : -1.0;

    double maxabs = 0;
    for (size_t col = 0; col < me; col++) {
        double xr[64], xi[64], outr[64], outi[64], tr[64], ti[64];
        for (int j = 0; j < N; j++) {
            xr[j] = re0[(size_t)j * ios + col];
            xi[j] = im0[(size_t)j * ios + col];
        }

        /* Stage A: optional PRE-twiddle (DIT fwd: W^k ; DIF bwd: conj(W^k)). */
        for (int k = 0; k < N; k++) { tr[k] = xr[k]; ti[k] = xi[k]; }
#if (MODE == 1 && BWD == 0)            /* DIT fwd: pre-tw W^k */
        for (int k = 1; k < N; k++) {
            double a = -2.0 * MPI * k / N, c = cos(a), s = sin(a);
            tr[k] = xr[k] * c - xi[k] * s;
            ti[k] = xr[k] * s + xi[k] * c;
        }
#elif (MODE == 2 && BWD == 1)          /* DIF bwd: pre-tw conj(W^k) = +angle */
        for (int k = 1; k < N; k++) {
            double a = +2.0 * MPI * k / N, c = cos(a), s = sin(a);
            tr[k] = xr[k] * c - xi[k] * s;
            ti[k] = xr[k] * s + xi[k] * c;
        }
#endif

        /* Stage B: the DFT (sign per direction). */
        double rr[64], ri[64];
        for (int n = 0; n < N; n++) {
            double sr = 0, si = 0;
            for (int k = 0; k < N; k++) {
                double a = dsign * 2.0 * MPI * n * k / N, c = cos(a), s = sin(a);
                sr += tr[k] * c - ti[k] * s;
                si += tr[k] * s + ti[k] * c;
            }
            rr[n] = sr; ri[n] = si;
        }

        /* Stage C: optional POST-twiddle (DIT bwd: conj(W^n) ; DIF fwd: W^n). */
        for (int n = 0; n < N; n++) { outr[n] = rr[n]; outi[n] = ri[n]; }
#if (MODE == 1 && BWD == 1)            /* DIT bwd: post-tw conj(W^n) = +angle */
        for (int n = 1; n < N; n++) {
            double a = +2.0 * MPI * n / N, c = cos(a), s = sin(a);
            outr[n] = rr[n] * c - ri[n] * s;
            outi[n] = rr[n] * s + ri[n] * c;
        }
#elif (MODE == 2 && BWD == 0)          /* DIF fwd: post-tw W^n = -angle */
        for (int n = 1; n < N; n++) {
            double a = -2.0 * MPI * n / N, c = cos(a), s = sin(a);
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
