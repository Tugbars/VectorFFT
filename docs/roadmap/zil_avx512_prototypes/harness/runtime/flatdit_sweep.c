/* flat DIT engine sweep: chains x stage forms x tiles x (serial, MT blocks, MT tiles at T=4),
 * natural fwd vs a reference DFT + roundtrip, scrambled class roundtrip. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.c"   /* textually: the engine headers and the pool */
static const int POOL[] = { 3, 4, 5, 7, 8, 9, 11, 13, 15, 16, 25, 27 };
static long nplan, nchk, nbad, nform_m, nform_n, nform_o, ntile, nmt;
static void dft(const double *x, double *y, int N)
{
    double *c = malloc(16 * (size_t)N);
    for (int e = 0; e < N; e++) { long double a = -2.0L * 3.14159265358979323846264338327950288L * e / N; c[2*e] = cosl(a); c[2*e+1] = sinl(a); }
    for (int k = 0; k < N; k++) { long double sr = 0, si = 0; for (int n = 0; n < N; n++) { int e = (int)(((long long)k * n) % N); sr += (long double)x[2*n]*c[2*e] - (long double)x[2*n+1]*c[2*e+1]; si += (long double)x[2*n]*c[2*e+1] + (long double)x[2*n+1]*c[2*e]; } y[2*k] = sr; y[2*k+1] = si; }
    free(c);
}
static double rel(const double *a, const double *b, size_t n) { double num=0,den=0; for (size_t i=0;i<n;i++){num+=(a[i]-b[i])*(a[i]-b[i]);den+=b[i]*b[i];} return sqrt(num/den); }
static void run1(vfft_ilfd_plan_t *p, const double *x, const double *ref, const char *tag, int mt)
{
    const int N = p->N; double *y = malloc(16 * (size_t)N), *z = malloc(16 * (size_t)N);
    double ef = 0, eb;
    if (mt) { p->mt = mt; if (!vfft_ilfd_execute_mt(p, x, y, 0)) { free(y); free(z); return; } }
    else vfft_ilfd_execute_fwd(p, x, y);
    if (!p->scr) ef = rel(y, ref, 2 * (size_t)N);
    if (mt) { if (!vfft_ilfd_execute_mt(p, y, z, 1)) vfft_ilfd_execute_bwd(p, y, z); } else vfft_ilfd_execute_bwd(p, y, z);
    for (int i = 0; i < 2 * N; i++) z[i] /= N;
    eb = (p->scr ? p->scr_ok : p->bwd_ok) ? rel(z, x, 2 * (size_t)N) : 0;
    nchk++; if (mt) nmt++;
    if (ef > 1e-12 || eb > 1e-12) { nbad++; printf("  WRONG %s fwd %.1e roundtrip %.1e\n", tag, ef, eb); }
    p->mt = 0;
    free(y); free(z);
}
static void check_chain(int N, const int *R, int K, const double *x, const double *ref)
{
    char forms[16]; int nf = 1; for (int s = 1; s < K; s++) nf *= 4;
    static const char L[4] = { 't', 'm', 'n', 'o' };
    for (int scr = 0; scr < 2; scr++)
    for (int f = 0; f < nf; f++) {
        int q = f, o = 0;
        for (int s = 1; s < K; s++) { forms[o++] = L[q % 4]; q /= 4; if (s < K - 1) forms[o++] = '.'; }
        forms[o] = 0;
        vfft_ilfd_plan_t *p = scr ? vfft_ilfd_create_scr_of(N, R, K, forms, 0) : vfft_ilfd_create_chain(N, R, K);
        if (!p) continue;
        if (!scr && !vfft_ilfd_apply_forms(p, forms)) { vfft_ilfd_destroy(p); continue; }
        nplan++; if (strchr(forms, 'm')) nform_m++; if (strchr(forms, 'n')) nform_n++; if (strchr(forms, 'o')) nform_o++;
        int tws[16]; const int nt = vfft_ilfd_tw_candidates(p, 0, tws, 16);
        for (int t = 0; t < nt; t++) {
            if (!vfft_ilfd_apply_tw(p, tws[t])) continue;
            if (tws[t]) ntile++;
            char tag[128]; int c = snprintf(tag, sizeof tag, "N=%d ", N);
            for (int s = 0; s < K; s++) c += snprintf(tag + c, sizeof tag - c, "%s%d", s ? "." : "", R[s]);
            snprintf(tag + c, sizeof tag - c, " %s forms=%s tw=%d", scr ? "scr" : "nat", forms, tws[t]);
            run1(p, x, ref, tag, 0);
            if (vfft_ilfd_mt_bind(p, 4)) {
                p->mt_t = 4;
                char t2[160]; snprintf(t2, sizeof t2, "%s MT-blocks", tag); run1(p, x, ref, t2, 1);
                if (tws[t]) { snprintf(t2, sizeof t2, "%s MT-tiles", tag); run1(p, x, ref, t2, 2); }
            }
        }
        vfft_ilfd_destroy(p);
    }
}
static int nch;
static void rec(int N, int rem, int *R, int k, const double *x, const double *ref)
{
    if (rem == 1) { if (k >= 2 && nch < 60) { nch++; check_chain(N, R, k, x, ref); } return; }
    if (k == VFFT_ILFD_MAX_K || k == 4) return;
    for (size_t i = 0; i < sizeof POOL / sizeof *POOL; i++) if (rem % POOL[i] == 0) { R[k] = POOL[i]; rec(N, rem / POOL[i], R, k + 1, x, ref); }
}
int main(void)
{
    static const int NS[] = { 45, 63, 75, 105, 125, 135, 189, 225, 243, 315, 375, 441, 525, 675, 729, 945, 1125, 1215, 2025, 3375, 180, 360, 600, 1100 };
    vfft_set_num_threads(4);
    printf("VFFT_IL_VW=%d\n", VFFT_IL_VW);
    for (size_t i = 0; i < sizeof NS / sizeof *NS; i++) {
        const int N = NS[i]; long b0 = nbad, c0 = nchk, p0 = nplan;
        double *x = malloc(16 * (size_t)N), *ref = malloc(16 * (size_t)N);
        for (int j = 0; j < 2 * N; j++) x[j] = sin(0.37 * j) + 0.25 * cos(1.3 * j + 0.1 * (j % 7));
        dft(x, ref, N);
        int R[8]; nch = 0; rec(N, N, R, 0, x, ref);
        printf("N=%-5d chains %2d plans %4ld checks %5ld wrong %ld\n", N, nch, nplan - p0, nchk - c0, nbad - b0); fflush(stdout);
        free(x); free(ref);
    }
    printf("TOTAL plans %ld (with msz %ld, t2csgn %ld, natural-order group loop %ld; tiled runs %ld) checks %ld (MT %ld) wrong %ld\n",
           nplan, nform_m, nform_n, nform_o, ntile, nchk, nmt, nbad);
    return nbad != 0;
}
