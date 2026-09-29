/* zttr_debug.c -- where the Hermitian terminator goes wrong: at one (N,
 * chain) run the ingest + mids into a plane P, then (a) the plain tlf + the
 * zr2c fold = X_ref2, checked against the long double DFT; (b) the Hermitian
 * terminator out of place on a copy of P, checked against X_ref2, the first
 * wrong bins printed. Build: python gauntlet/build.py --compile --vfft --src
 * gauntlet/zttr_debug.c;  run: zttr_debug N r0.r1.r2 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "zttr.h"
#include "zr2c.h"
#include "race_timing.h"
#ifdef _WIN32
#include <windows.h>
#endif

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (double)(*s >> 8) / (double)(1u << 24) - 0.5;
}
static void ref_r2c(int N, const double *x, double *X)
{
    const long double tp = 6.283185307179586476925286766559L;
    for (int k = 0; k <= N / 2; k++) {
        long double re = 0, im = 0;
        for (int n = 0; n < N; n++) {
            long double a = tp * (long double)((long long)k * n % N) / (long double)N;
            re += (long double)x[n] * cosl(a);
            im -= (long double)x[n] * sinl(a);
        }
        X[2 * k] = (double)re; X[2 * k + 1] = (double)im;
    }
}
static double relerr(const double *a, const double *b, size_t n)
{
    double e = 0, m = 0;
    for (size_t i = 0; i < n; i++) { if (fabs(a[i] - b[i]) > e) e = fabs(a[i] - b[i]); if (fabs(b[i]) > m) m = fabs(b[i]); }
    return m > 0 ? e / m : e;
}

int main(int argc, char **argv)
{
    if (argc < 3) { fprintf(stderr, "usage: zttr_debug N r0.r1[.r2..]\n"); return 2; }
    const int N = atoi(argv[1]), M = N / 2;
    int chain[8], nf = 0;
    { char *s = argv[2], *t; for (t = strtok(s, "."); t && nf < 8; t = strtok(NULL, ".")) chain[nf++] = atoi(t); }
    vfft_zttr_plan_t *p = vfft_zttr_create(N, chain, nf, 0);
    if (!p) { printf("create refused\n"); return 1; }
    vfft_ztt_plan_t *zt = p->zt;
    unsigned seed = 0xBEEFu;
    double *x = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
    double *Rf = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
    double *P = (double *)vfft_aligned_alloc((size_t)(2 * M) * sizeof(double) + 4096);
    double *P2 = (double *)vfft_aligned_alloc((size_t)(2 * M) * sizeof(double) + 4096);
    double *Z = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
    double *X2 = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
    double *Xh = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
    for (int i = 0; i < N; i++) x[i] = urand(&seed);
    x[N] = x[N + 1] = 0;
    ref_r2c(N, x, Rf);
    /* the plane: ingest + mids (untiled) */
    memset(P, 0, (size_t)(2 * M) * sizeof(double) + 4096);
    zt->st_fwd[0](x, 0, P, 0, 0, (const double *)zt->rb, (size_t)zt->ncol, 0, 0, 0, (size_t)zt->ncol);
    for (int s = 1; s < nf - 1; s++)
        zt->st_fwd[s](P, 0, P, 0, zt->tw + zt->twoff[s], 0, (size_t)zt->L[s], (size_t)zt->Gs[s], 0, 0, (size_t)zt->L[s]);
    memcpy(P2, P, (size_t)(2 * M) * sizeof(double) + 4096);
    /* (a) tlf -> Z, then the fold -> X2 */
    const size_t L = (size_t)zt->L[nf - 1];
    zt->st_fwd[nf - 1](P, 0, Z, 0, zt->tw + zt->twoff[nf - 1], 0, L, 1, L, 0, L);
    {
        int top = N / 4;
        double *aff = (double *)vfft_aligned_alloc(sizeof(double) * 4u * (size_t)(top + 1));
        _zr2c_init_aff(N, aff, aff + (top + 1), aff + 2 * (top + 1), aff + 3 * (top + 1));
        _zr2c_fold_fwd(Z, X2, aff, aff + (top + 1), N, 1, (size_t)N + 2, (size_t)N + 2);
        vfft_aligned_free(aff);
    }
    printf("N=%d chain %s: R=%d L=%ld; tlf+fold vs DFT: %.2e\n", N, argv[2], p->R, p->L, relerr(X2, Rf, (size_t)N + 2));
    /* (b) the Hermitian terminator, out of place from the plane copy */
    memset(Xh, 0, (size_t)(N + 2) * sizeof(double));
#if defined(__AVX2__)
    _zttr_tlfh(p, P2, Xh);
#endif
    printf("tlfh (oop) vs tlf+fold: %.2e\n", relerr(Xh, X2, (size_t)N + 2));
    int shown = 0;
    for (int f = 0; f <= M && shown < 24; f++)
    {
        double dr = Xh[2 * f] - X2[2 * f], di = Xh[2 * f + 1] - X2[2 * f + 1];
        if (fabs(dr) > 1e-9 || fabs(di) > 1e-9)
        {
            printf("  f=%4d (leg %d col %d): got (%+.4f %+.4f) want (%+.4f %+.4f)\n", f, (int)(f / L), (int)(f % L),
                   Xh[2 * f], Xh[2 * f + 1], X2[2 * f], X2[2 * f + 1]);
            shown++;
        }
    }
    /* (c) in place: the terminator on the plane itself */
    memcpy(P2, P, (size_t)(2 * M) * sizeof(double) + 4096);
#if defined(__AVX2__)
    _zttr_tlfh(p, P2, P2);
#endif
    printf("tlfh (in place) vs tlf+fold: %.2e\n", relerr(P2, X2, (size_t)N + 2));
    /* the terminators' own cost on the intact plane: tlf alone, the fold
     * alone, tlfh out of place, tlfh in place (the plane restored each rep) */
    {
        int top = N / 4;
        double *aff = (double *)vfft_aligned_alloc(sizeof(double) * 4u * (size_t)(top + 1));
        _zr2c_init_aff(N, aff, aff + (top + 1), aff + 2 * (top + 1), aff + 3 * (top + 1));
        double t_tlf = 0, t_fold = 0, t_h = 0, t_hi = 0, t_cp = 0;
        int reps = (int)(4.0e5 / (double)(M > 256 ? M : 256));
        if (reps < 3) reps = 3;
        for (int rr = 0; rr < 5; rr++)
        {
            double s0 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                zt->st_fwd[nf - 1](P, 0, Z, 0, zt->tw + zt->twoff[nf - 1], 0, L, 1, L, 0, L);
            double s1 = vfft_now_ns();
            for (int i = 0; i < reps; i++)
                _zr2c_fold_fwd(Z, X2, aff, aff + (top + 1), N, 1, (size_t)N + 2, (size_t)N + 2);
            double s2 = vfft_now_ns();
#if defined(__AVX2__)
            for (int i = 0; i < reps; i++) _zttr_tlfh(p, P, Xh);
#endif
            double s3 = vfft_now_ns();
            for (int i = 0; i < reps; i++) memcpy(P2, P, (size_t)(2 * M) * sizeof(double));
            double s4 = vfft_now_ns();
#if defined(__AVX2__)
            for (int i = 0; i < reps; i++) { memcpy(P2, P, (size_t)(2 * M) * sizeof(double)); _zttr_tlfh(p, P2, P2); }
#endif
            double s5 = vfft_now_ns();
            double a = (s1 - s0) / reps, b = (s2 - s1) / reps, c = (s3 - s2) / reps, d = (s4 - s3) / reps, e = (s5 - s4) / reps - d;
            if (rr == 0 || a < t_tlf) t_tlf = a;
            if (rr == 0 || b < t_fold) t_fold = b;
            if (rr == 0 || c < t_h) t_h = c;
            if (rr == 0 || d < t_cp) t_cp = d;
            if (rr == 0 || e < t_hi) t_hi = e;
        }
        printf("terminators (ns, best of 5): tlf %.0f + fold %.0f = %.0f | tlfh oop %.0f | tlfh in place %.0f (copy %.0f subtracted)\n",
               t_tlf, t_fold, t_tlf + t_fold, t_h, t_hi, t_cp);
        vfft_aligned_free(aff);
    }
    vfft_zttr_destroy(p);
    return 0;
}
