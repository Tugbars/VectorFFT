/* zr2c_gate.c — permanent correctness gate for the INTERLEAVED real-transform
 * folds (zr2c.h — the D2 route's two z->z passes), per the campaign's law:
 * ELEMENTWISE vs a naive DFT, PER DIRECTION, never a roundtrip (a roundtrip
 * survives self-consistent permutations and consistent conjugation errors).
 *
 * Covers: fwd fold OOP + IN-PLACE, bwd fold OOP + IN-PLACE, K in {1, 4}
 * (transform-contiguous), even N incl. an N == 2 (mod 4) cell (half odd — no
 * center bin), plus the structural-zero contract (X[0].im / X[N/2].im are
 * literal +0.0 on the fwd side).
 *
 * The folds are gated in ISOLATION against naive inputs — the interior c2c is
 * naive here on purpose, so a fold bug cannot hide behind an interior bug.
 * The WIRED route (IL c2c interior + these folds through the front door) gets
 * its own gate when the route lands, on the il2p_tangent_gate model.
 *
 * Build (from build_tuned/): python build.py --src benches/zr2c_gate.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "zr2c.h"

static uint64_t lcg = 0x9E3779B97F4A7C15ull;
static double rnd(void){ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
    return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

static void naive_real_dft(const double *x, int N, double *Xr, double *Xi)
{
    for (int f = 0; f <= N/2; f++){ double sr = 0, si = 0;
        for (int n = 0; n < N; n++){
            double a = -2.0*VFFT_ZR2C_PI*(double)f*n/(double)N;
            sr += x[n]*cos(a); si += x[n]*sin(a); }
        Xr[f] = sr; Xi[f] = si; }
}
static void naive_c2c_half(const double *x, int N, double *z)
{
    int h = N/2;
    for (int p = 0; p < h; p++){ double sr = 0, si = 0;
        for (int j = 0; j < h; j++){
            double a = -2.0*VFFT_ZR2C_PI*(double)p*j/(double)h;
            double xr = x[2*j], xi = x[2*j+1], c = cos(a), s = sin(a);
            sr += xr*c - xi*s; si += xr*s + xi*c; }
        z[2*p] = sr; z[2*p+1] = si; }
}

static int g_fail = 0;
static void judge(const char *what, int N, size_t K, double err, double tol)
{
    int ok = err < tol;
    printf("  %-26s N=%-6d K=%zu  %.2e  %s\n", what, N, K, err, ok ? "OK" : "*** FAIL ***");
    if (!ok) g_fail = 1;
}

static void run_cell(int N, size_t K)
{
    const int half = N/2, top = N/4;
    size_t xs = (size_t)N + 2, zs = (size_t)N;   /* per-transform strides */
    double *x   = malloc(8*(size_t)N*K);
    double *z   = malloc(8*zs*K);
    double *Xr  = malloc(8*(size_t)(half+1)*K);
    double *Xi  = malloc(8*(size_t)(half+1)*K);
    double *out = malloc(8*xs*K);
    double *io  = malloc(8*xs*K);
    double *affS = malloc(8*(size_t)(top+1));
    double *affC = malloc(8*(size_t)(top+1));
    double *bwdS = malloc(8*(size_t)(top+1));   /* raw sin/cos: the backward */
    double *bwdC = malloc(8*(size_t)(top+1));   /* fold's own coefficients   */
    _zr2c_init_aff(N, affS, affC, bwdS, bwdC);
    for (size_t t = 0; t < K; t++)
        for (int i = 0; i < N; i++) x[t*(size_t)N + i] = rnd();
    for (size_t t = 0; t < K; t++){
        naive_c2c_half(x + t*(size_t)N, N, z + t*zs);
        naive_real_dft(x + t*(size_t)N, N, Xr + t*(size_t)(half+1), Xi + t*(size_t)(half+1));
    }
    double xm = 0;
    for (size_t t = 0; t < K; t++)
        for (int f = 0; f <= half; f++){
            double a = fabs(Xr[t*(size_t)(half+1)+f]) + fabs(Xi[t*(size_t)(half+1)+f]);
            if (a > xm) xm = a; }

    /* ── fwd OOP ── */
    memset(out, 0, 8*xs*K);
    _zr2c_fold_fwd(z, out, affS, affC, N, K, zs, xs);
    double w = 0; int zok = 1;
    for (size_t t = 0; t < K; t++)
        for (int f = 0; f <= half; f++){
            double dr = fabs(out[t*xs+2*f]   - Xr[t*(size_t)(half+1)+f]);
            double di = fabs(out[t*xs+2*f+1] - Xi[t*(size_t)(half+1)+f]);
            if (dr > w) w = dr; if (di > w) w = di; }
    for (size_t t = 0; t < K; t++)
        if (out[t*xs+1] != 0.0 || out[t*xs+2*half+1] != 0.0) zok = 0;
    judge("fwd fold OOP", N, K, w/xm, 1e-12);
    judge("fwd structural zeros", N, K, zok ? 0.0 : 1.0, 0.5);

    /* ── fwd IN-PLACE (padded plane; X aliases Z) ── */
    memset(io, 0, 8*xs*K);
    for (size_t t = 0; t < K; t++) memcpy(io + t*xs, z + t*zs, 8*zs);
    _zr2c_fold_fwd(io, io, affS, affC, N, K, xs, xs);
    w = 0;
    for (size_t t = 0; t < K; t++)
        for (int f = 0; f <= half; f++){
            double dr = fabs(io[t*xs+2*f]   - Xr[t*(size_t)(half+1)+f]);
            double di = fabs(io[t*xs+2*f+1] - Xi[t*(size_t)(half+1)+f]);
            if (dr > w) w = dr; if (di > w) w = di; }
    judge("fwd fold IN-PLACE", N, K, w/xm, 1e-12);

    /* ── bwd OOP: fold(X_naive) must equal 2*z_naive ── */
    for (size_t t = 0; t < K; t++)
        for (int f = 0; f <= half; f++){
            io[t*xs+2*f]   = Xr[t*(size_t)(half+1)+f];
            io[t*xs+2*f+1] = Xi[t*(size_t)(half+1)+f]; }
    memset(out, 0, 8*xs*K);
    _zr2c_fold_bwd(io, out, bwdS, bwdC, N, K, xs, xs);
    double zm = 0; w = 0;
    for (size_t t = 0; t < K; t++)
        for (int p = 0; p < half; p++){
            double a = fabs(z[t*zs+2*p]) + fabs(z[t*zs+2*p+1]);
            if (a > zm) zm = a;
            double dr = fabs(out[t*xs+2*p]   - 2.0*z[t*zs+2*p]);
            double di = fabs(out[t*xs+2*p+1] - 2.0*z[t*zs+2*p+1]);
            if (dr > w) w = dr; if (di > w) w = di; }
    judge("bwd fold OOP", N, K, w/(2.0*zm), 1e-12);

    /* ── bwd IN-PLACE ── */
    _zr2c_fold_bwd(io, io, bwdS, bwdC, N, K, xs, xs);
    w = 0;
    for (size_t t = 0; t < K; t++)
        for (int p = 0; p < half; p++){
            double dr = fabs(io[t*xs+2*p]   - 2.0*z[t*zs+2*p]);
            double di = fabs(io[t*xs+2*p+1] - 2.0*z[t*zs+2*p+1]);
            if (dr > w) w = dr; if (di > w) w = di; }
    judge("bwd fold IN-PLACE", N, K, w/(2.0*zm), 1e-12);

    /* ── PERM-AWARE variants, gated with a RANDOM permutation (any mutually
     * inverse iperm/perm must work — stronger than any specific chain) ── */
    {
        int *iperm = malloc(sizeof(int)*(size_t)half);
        int *perm  = malloc(sizeof(int)*(size_t)half);
        for (int i = 0; i < half; i++) iperm[i] = i;
        for (int i = half - 1; i > 0; i--){        /* Fisher-Yates via lcg */
            int j = (int)(lcg % (uint64_t)(i + 1)); rnd();
            int tswp = iperm[i]; iperm[i] = iperm[j]; iperm[j] = tswp; }
        for (int i = 0; i < half; i++) perm[iperm[i]] = i;
        double *zscr = malloc(8*zs*K);
        for (size_t t = 0; t < K; t++)
            for (int p = 0; p < half; p++){
                zscr[t*zs+2*p]   = z[t*zs+2*iperm[p]];
                zscr[t*zs+2*p+1] = z[t*zs+2*iperm[p]+1]; }
        memset(out, 0, 8*xs*K);
        _zr2c_fold_fwd_perm(zscr, out, affS, affC, iperm, perm, N, K, zs, xs);
        w = 0;
        for (size_t t = 0; t < K; t++)
            for (int f = 0; f <= half; f++){
                double dr = fabs(out[t*xs+2*f]   - Xr[t*(size_t)(half+1)+f]);
                double di = fabs(out[t*xs+2*f+1] - Xi[t*(size_t)(half+1)+f]);
                if (dr > w) w = dr; if (di > w) w = di; }
        judge("fwd fold PERM (random)", N, K, w/xm, 1e-12);
        /* bwd: fold natural X -> scrambled slots; slot p must hold 2*z[iperm[p]] */
        for (size_t t = 0; t < K; t++)
            for (int f = 0; f <= half; f++){
                io[t*xs+2*f]   = Xr[t*(size_t)(half+1)+f];
                io[t*xs+2*f+1] = Xi[t*(size_t)(half+1)+f]; }
        memset(zscr, 0, 8*zs*K);
        _zr2c_fold_bwd_perm(io, zscr, bwdS, bwdC, iperm, perm, N, K, xs, zs);
        w = 0;
        for (size_t t = 0; t < K; t++)
            for (int p = 0; p < half; p++){
                double dr = fabs(zscr[t*zs+2*p]   - 2.0*z[t*zs+2*iperm[p]]);
                double di = fabs(zscr[t*zs+2*p+1] - 2.0*z[t*zs+2*iperm[p]+1]);
                if (dr > w) w = dr; if (di > w) w = di; }
        judge("bwd fold PERM (random)", N, K, w/(2.0*zm), 1e-12);
        free(iperm); free(perm); free(zscr);
    }

    free(x); free(z); free(Xr); free(Xi); free(out); free(io);
    free(affS); free(affC); free(bwdS); free(bwdC);
}

int main(void)
{
    printf("zr2c fold gate — elementwise vs naive DFT, per direction, never a roundtrip\n\n");
    const int Ns[] = { 16, 510, 512, 1024, 4096 };   /* 510: half odd, no center bin */
    for (size_t i = 0; i < sizeof Ns / sizeof Ns[0]; i++){
        run_cell(Ns[i], 1);
        run_cell(Ns[i], 4);
    }
    printf("\n%s\n", g_fail ? "ZR2C FOLD GATE: FAILURE" : "ZR2C FOLD GATE: ALL CORRECT");
    return g_fail;
}
