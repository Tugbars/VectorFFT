/* zlaw.c — every ZTURN-T chain the grammar admits (ends 4/8, mids 4/8/3/5/7/9/15),
 * N in a pow2 + odd-band set, both order classes: create or refuse; every created plan
 * checked single-threaded at every legal tile: fwd vs a reference DFT (scrambled decoded
 * through vfft_ztt_perm), both
 * classes by roundtrip (bwd(fwd(x)) == N x), OOP and in place (natural). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "ztt.h"
static const int ENDS[] = { 4, 8 }, MIDS[] = { 4, 8, 3, 5, 7, 9, 15 };
static int ncre, nref, nbad, nchk;
static double *REF_X; static long REF_N;
static void dft(const double *x, double *y, long N)
{
    double *c = malloc(16 * N);
    for (long e = 0; e < N; e++) { long double a = -2.0L * 3.14159265358979323846264338327950288L * e / N; c[2*e] = cosl(a); c[2*e+1] = sinl(a); }
    for (long k = 0; k < N; k++) { long double sr = 0, si = 0; for (long n = 0; n < N; n++) { long e = (k * n) % N; sr += (long double)x[2*n]*c[2*e] - (long double)x[2*n+1]*c[2*e+1]; si += (long double)x[2*n]*c[2*e+1] + (long double)x[2*n+1]*c[2*e]; } y[2*k] = sr; y[2*k+1] = si; }
    free(c);
}
static double rel(const double *a, const double *b, long n) { double num=0,den=0; for (long i=0;i<n;i++){num+=(a[i]-b[i])*(a[i]-b[i]);den+=b[i]*b[i];} return sqrt(num/den); }
static void check(int N, const int *ch, int nf, int scr, const double *x, const double *ref)
{
    vfft_ztt_plan_t *p = vfft_ztt_create_chain_ord(N, ch, nf, scr);
    char cs[64]; int o = 0; for (int s = 0; s < nf; s++) o += sprintf(cs + o, "%s%d", s ? "." : "", ch[s]);
    if (!p) { nref++; return; }
    ncre++;
    static const size_t TILES[] = { 0, 64, 128, 256, 512, 768, 1024, 1536, 2048, 3072 };
    double *y = malloc(16 * N), *z = malloc(16 * N), *w = malloc(16 * N);
    for (size_t t = 0; t < sizeof TILES / sizeof *TILES; t++) {
        if (!vfft_ztt_set_tile(p, TILES[t])) continue;
        for (int ip = 0; ip < (scr ? 1 : 2); ip++) {
            vfft_ztt_bind(p, ip);
            double ef = 0;
            if (ip) { memcpy(y, x, 16 * N); vfft_ztt_execute_fwd(p, y, y); } else vfft_ztt_execute_fwd(p, x, y);
            if (!scr) ef = rel(y, ref, 2 * N);
            else {   /* scrambled: decode through the plan's permutation (bin k at y[perm(k)]) */
                double *d = malloc(16 * N);
                for (long k = 0; k < N; k++) { const size_t q = vfft_ztt_perm(p, k); d[2*k] = y[2*q]; d[2*k+1] = y[2*q+1]; }
                ef = rel(d, ref, 2 * N); free(d);
            }
            if (ip) { memcpy(z, y, 16 * N); vfft_ztt_execute_bwd(p, z, z); } else vfft_ztt_execute_bwd(p, y, z);
            for (long i = 0; i < 2 * N; i++) w[i] = z[i] / N;
            double eb = rel(w, x, 2 * N);
            nchk++;
            if (ef > 1e-12 || eb > 1e-12) { nbad++; printf("  WRONG N=%d %s %s tile=%zu %s fwd %.1e roundtrip %.1e%s\n", N, cs, scr ? "scr" : "nat", TILES[t], ip ? "inplace" : "oop", ef, eb, p->staged ? " (staged)" : ""); }
        }
    }
    free(y); free(z); free(w);
    vfft_ztt_destroy(p);
}
static void rec(int N, int rem, int *ch, int nf, const double *x, const double *ref)
{
    if (nf >= 2) {   /* close with an end radix */
        for (int e = 0; e < 2; e++) if (rem == ENDS[e]) { ch[nf] = ENDS[e]; for (int scr = 0; scr < 2; scr++) check(N, ch, nf + 1, scr, x, ref); }
    }
    if (nf == 1) for (int e = 0; e < 2; e++) if (rem == ENDS[e]) { ch[1] = ENDS[e]; for (int scr = 0; scr < 2; scr++) check(N, ch, 2, scr, x, ref); }
    if (nf + 2 > VFFT_ZTT_MAX_NF) return;
    for (int m = 0; m < 7; m++) if (rem % MIDS[m] == 0 && rem / MIDS[m] >= 4) { ch[nf] = MIDS[m]; rec(N, rem / MIDS[m], ch, nf + 1, x, ref); }
}
int main(void)
{
    static const int NS[] = { 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 48, 80, 96, 144, 192, 240, 320, 384, 448, 576, 720, 960, 1152, 1344, 1920, 2880 };
    printf("VFFT_IL_VW=%d\n", VFFT_IL_VW);
    for (size_t i = 0; i < sizeof NS / sizeof *NS; i++) {
        const int N = NS[i]; int c0 = ncre, r0 = nref, b0 = nbad;
        double *x = malloc(16 * N), *ref = malloc(16 * N);
        for (long j = 0; j < 2 * N; j++) x[j] = sin(0.37 * j) + 0.25 * cos(1.3 * j + 0.1 * (j % 7));
        dft(x, ref, N);
        int ch[VFFT_ZTT_MAX_NF];
        for (int e = 0; e < 2; e++) if (N % ENDS[e] == 0) { ch[0] = ENDS[e]; rec(N, N / ENDS[e], ch, 1, x, ref); }
        printf("N=%-5d created %3d refused %3d wrong %d\n", N, ncre - c0, nref - r0, nbad - b0);
        free(x); free(ref);
    }
    printf("TOTAL created %d refused %d checks %d wrong %d\n", ncre, nref, nchk, nbad);
    return nbad != 0;
}
