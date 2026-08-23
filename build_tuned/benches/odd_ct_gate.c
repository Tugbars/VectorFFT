/* odd_ct_gate.c — does factoring an odd composite compute the SAME DFT?
 *
 * dft_small sent every odd n to dft_cx_odd, the direct conjugate-pair form
 * (O(n^2/2)). VFFT_CX_ODDCT=1 makes it Cooley-Tukey factor odd COMPOSITES
 * instead (9->3x3, 15->3x5, 21->3x7, 25->5x5, 27->3x9), recursing through
 * dft_small so the leaves may themselves be odd. Odd primes are untouched --
 * they have nothing to factor.
 *
 * Static op counts say the factored form is 1.07x-1.76x cheaper. This asks
 * the only question that decides whether that matters: is it the same
 * transform?
 *
 * METHOD. Both variants are emitted from the same generator into the same
 * binary (the ON arm sed-renamed to ..._oddct), run on identical input, and
 * compared against a NAIVE DFT computed in long double. Comparing the two
 * kernels only against EACH OTHER would not catch a shared error in the
 * index algebra -- and the index mapping is exactly what changed, so the
 * reference has to be independent.
 *
 * n1t is a corner turn: reads zin[2*(l*Ls + k)] for l<R, k<count and writes
 * zout[2*(k*OLs + l)]. Every count in 1..5 is exercised, because the odd-count
 * tail arm and the new factored body are independent changes that must
 * compose.
 *
 * Build (from build_tuned/benches):
 *   gcc -O3 -mavx2 -mfma -march=native -o odd_ct_gate.exe odd_ct_gate.c \
 *       oc_off_9.c oc_on_9.c oc_off_15.c oc_on_15.c oc_off_21.c oc_on_21.c \
 *       oc_off_25.c oc_on_25.c oc_off_27.c oc_on_27.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef void (*kfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);

#define DECL(R)                                                               \
  void radix##R##_z_n1t_fwd_avx2(const double *, const double *, double *,     \
      double *, const double *, const double *,                                \
      size_t, size_t, size_t, size_t, size_t);                                 \
  void radix##R##_z_n1t_oddct_avx2(const double *, const double *, double *,   \
      double *, const double *, const double *,                                \
      size_t, size_t, size_t, size_t, size_t);
DECL(9) DECL(15) DECL(21) DECL(25) DECL(27)

static uint64_t lcg = 0x243F6A8885A308D3ull;
static double rnd(void)
{ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
  return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

static int g_fail = 0;

/* naive R-point complex DFT in long double — the independent reference */
static void naive(const double *in, int R, long double *or_, long double *oi)
{
    int f, n;
    for (f = 0; f < R; f++) {
        long double sr = 0.0L, si = 0.0L;
        for (n = 0; n < R; n++) {
            long double a = -2.0L*(long double)M_PI*(long double)f*n/(long double)R;
            long double c = cosl(a), s = sinl(a);
            sr += (long double)in[2*n]*c - (long double)in[2*n+1]*s;
            si += (long double)in[2*n]*s + (long double)in[2*n+1]*c;
        }
        or_[f] = sr; oi[f] = si;
    }
}

static double run_one(kfn f, int R, size_t count, const double *zin, double *zout)
{
    const size_t Ls = count, OLs = (size_t)R;
    long double *rr = (long double *)malloc((size_t)R*sizeof(long double));
    long double *ri = (long double *)malloc((size_t)R*sizeof(long double));
    double *col = (double *)malloc(2*(size_t)R*sizeof(double));
    double worst = 0.0, mag = 0.0;
    size_t k; int l;

    memset(zout, 0, 2*count*OLs*sizeof(double));
    f(zin, 0, zout, 0, 0, 0, Ls, 0, OLs, 0, count);

    for (k = 0; k < count; k++) {
        for (l = 0; l < R; l++) {           /* gather column k of the input */
            col[2*l]   = zin[2*((size_t)l*Ls + k)];
            col[2*l+1] = zin[2*((size_t)l*Ls + k) + 1];
        }
        naive(col, R, rr, ri);
        for (l = 0; l < R; l++) {
            double gr = zout[2*(k*OLs + (size_t)l)];
            double gi = zout[2*(k*OLs + (size_t)l) + 1];
            double dr = fabs(gr - (double)rr[l]), di = fabs(gi - (double)ri[l]);
            double m  = fabsl(rr[l]) + fabsl(ri[l]);
            if (dr > worst) worst = dr;
            if (di > worst) worst = di;
            if (m > mag) mag = m;
        }
    }
    free(rr); free(ri); free(col);
    return mag > 0 ? worst/mag : worst;
}

static void arm(int R, kfn off, kfn on)
{
    size_t count;
    for (count = 1; count <= 5; count++) {
        const size_t nin = 2*(size_t)R*count, nout = 2*count*(size_t)R;
        double *zin = (double *)malloc(nin*sizeof(double));
        double *za  = (double *)malloc(nout*sizeof(double));
        double *zb  = (double *)malloc(nout*sizeof(double));
        double ea, eb;
        size_t i;
        for (i = 0; i < nin; i++) zin[i] = rnd();
        ea = run_one(off, R, count, zin, za);
        eb = run_one(on,  R, count, zin, zb);
        {
            int ok = (ea < 1e-13) && (eb < 1e-13);
            printf("  radix %-3d count=%zu  direct rel %.2e   factored rel %.2e  %s\n",
                   R, count, ea, eb, ok ? "OK" : "*** FAIL ***");
            if (!ok) g_fail = 1;
        }
        free(zin); free(za); free(zb);
    }
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("odd-composite Cooley-Tukey gate — factored vs direct, both vs a naive DFT\n");
    printf("  the index mapping is what changed, so the reference is INDEPENDENT\n\n");
    arm(9,  radix9_z_n1t_fwd_avx2,  radix9_z_n1t_oddct_avx2);
    arm(15, radix15_z_n1t_fwd_avx2, radix15_z_n1t_oddct_avx2);
    arm(21, radix21_z_n1t_fwd_avx2, radix21_z_n1t_oddct_avx2);
    arm(25, radix25_z_n1t_fwd_avx2, radix25_z_n1t_oddct_avx2);
    arm(27, radix27_z_n1t_fwd_avx2, radix27_z_n1t_oddct_avx2);
    printf("\n%s\n", g_fail ? "*** ODD-CT: INCORRECT ***"
                            : "odd-CT: factored form is correct at every radix and count");
    return g_fail;
}
