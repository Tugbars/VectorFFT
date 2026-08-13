/* zr2c_fd_gate.c — FRONT-DOOR gate for the §D2 zr2c route: everything through
 * the PUBLIC API (vfft_create / vfft_execute), nothing plan-level.
 *
 * Covers, all elementwise vs a naive DFT (never a roundtrip):
 *   - R2C fwd + C2R bwd, INTERLEAVED CCE, OOP and IN-PLACE, both routes
 *     (VFFT_ZR2C_ROUTE=0 OOP-IL child / =1 NAT-IP cascade child), K=1,
 *     N in {512, 2048, 4096}.
 *   - REGRESSION smokes for paths the new branch must NOT disturb:
 *     K=4 interleaved r2c (split-interior CCE path) and odd-N (510)
 *     interleaved r2c OOP.
 *   - The in-place plane contract: 2*(N/2+1) doubles, MKL convention.
 *
 * Wisdom: argv[1] = wisdom DIR (default ../../src/dag-fft-compiler/generator/
 * generated). Loaded caller-owned (vfft_wisdom_load) — the override table is
 * never auto-persisted, so pointing at the shipped folder is read-safe.
 *
 * Build (from build_tuned/): python build.py --src benches/zr2c_fd_gate.c --vfft
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static uint64_t lcg = 0x9E3779B97F4A7C15ull;
static double rnd(void){ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
    return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

static void naive_real_dft(const double *x, int N, double *Xr, double *Xi)
{
    for (int f = 0; f <= N/2; f++){ double sr = 0, si = 0;
        for (int n = 0; n < N; n++){
            double a = -2.0*M_PI*(double)f*n/(double)N;
            sr += x[n]*cos(a); si += x[n]*sin(a); }
        Xr[f] = sr; Xi[f] = si; }
}

static int g_fail = 0;
static void judge(const char *what, int N, double err, double tol)
{
    int ok = err >= 0 && err < tol;
    printf("  %-38s N=%-6d %.2e %s\n", what, N, err, ok ? "OK" : "*** FAIL ***");
    if (!ok) g_fail = 1;
}

static vfft_wisdom *g_W;

static void run_cell(int N, int route)
{
    char lbl[64];
    static char env0[] = "VFFT_ZR2C_ROUTE=0", env1[] = "VFFT_ZR2C_ROUTE=1";
    putenv(route ? env1 : env0);

    const int half = N/2;
    size_t xs = (size_t)N + 2;
    double *x   = malloc(8*xs);
    double *X   = malloc(8*xs);
    double *y   = malloc(8*xs);
    double *pl  = malloc(8*xs);
    double *Xr  = malloc(8*(size_t)(half+1));
    double *Xi  = malloc(8*(size_t)(half+1));
    for (int i = 0; i < N; i++) x[i] = rnd();
    x[N] = x[N+1] = 0.0;
    naive_real_dft(x, N, Xr, Xi);
    double xm = 0, gm = 0;
    for (int f = 0; f <= half; f++){
        double a = fabs(Xr[f]) + fabs(Xi[f]); if (a > xm) xm = a; }
    for (int i = 0; i < N; i++){ double a = fabs(x[i]); if (a > gm) gm = a; }

    vfft_config_t cfg;

    /* ── R2C OOP ── */
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1; cfg.wisdom = g_W;
    vfft_plan p = vfft_create(&cfg);
    snprintf(lbl, sizeof lbl, "r2c OOP  route=%d", route);
    if (!p) { judge(lbl, N, -1, 1); }
    else {
        memset(X, 0, 8*xs);
        vfft_execute(p, VFFT_FORWARD, x, NULL, X, NULL);
        double w = 0;
        for (int f = 0; f <= half; f++){
            double dr = fabs(X[2*f]   - Xr[f]);
            double di = fabs(X[2*f+1] - Xi[f]);
            if (dr > w) w = dr; if (di > w) w = di; }
        judge(lbl, N, w/xm, 1e-9);
        vfft_destroy(p);
    }

    /* ── R2C IN-PLACE (one padded plane) ── */
    cfg.placement = VFFT_INPLACE;
    p = vfft_create(&cfg);
    snprintf(lbl, sizeof lbl, "r2c IN-PLACE  route=%d", route);
    if (!p) { judge(lbl, N, -1, 1); }
    else {
        memcpy(pl, x, 8*xs);
        vfft_execute(p, VFFT_FORWARD, pl, NULL, pl, NULL);
        double w = 0;
        for (int f = 0; f <= half; f++){
            double dr = fabs(pl[2*f]   - Xr[f]);
            double di = fabs(pl[2*f+1] - Xi[f]);
            if (dr > w) w = dr; if (di > w) w = di; }
        judge(lbl, N, w/xm, 1e-9);
        vfft_destroy(p);
    }

    /* ── C2R OOP: naive spectrum in -> must return N*x ── */
    cfg.transform = VFFT_C2R; cfg.placement = VFFT_OUTOFPLACE;
    p = vfft_create(&cfg);
    snprintf(lbl, sizeof lbl, "c2r OOP  route=%d", route);
    if (!p) { judge(lbl, N, -1, 1); }
    else {
        for (int f = 0; f <= half; f++){ X[2*f] = Xr[f]; X[2*f+1] = Xi[f]; }
        memset(y, 0, 8*xs);
        vfft_execute(p, VFFT_BACKWARD, X, NULL, y, NULL);
        double w = 0;
        for (int i = 0; i < N; i++){
            double d = fabs(y[i] - (double)N * x[i]); if (d > w) w = d; }
        judge(lbl, N, w/((double)N*gm), 1e-9);
        vfft_destroy(p);
    }

    /* ── C2R IN-PLACE ── */
    cfg.placement = VFFT_INPLACE;
    p = vfft_create(&cfg);
    snprintf(lbl, sizeof lbl, "c2r IN-PLACE  route=%d", route);
    if (!p) { judge(lbl, N, -1, 1); }
    else {
        for (int f = 0; f <= half; f++){ pl[2*f] = Xr[f]; pl[2*f+1] = Xi[f]; }
        vfft_execute(p, VFFT_BACKWARD, pl, NULL, pl, NULL);
        double w = 0;
        for (int i = 0; i < N; i++){
            double d = fabs(pl[i] - (double)N * x[i]); if (d > w) w = d; }
        judge(lbl, N, w/((double)N*gm), 1e-9);
        vfft_destroy(p);
    }

    free(x); free(X); free(y); free(pl); free(Xr); free(Xi);
}

/* regression smokes: the paths the zr2c branch must NOT have disturbed */
static void run_regressions(void)
{
    /* K=4 interleaved r2c OOP — the split-interior CCE path (K>1 keeps it) */
    {
        const int N = 512, half = N/2; const size_t K = 4;
        double *x  = malloc(8*(size_t)N*K);
        double *X  = malloc(8*((size_t)N+2)*K);
        double *Xr = malloc(8*(size_t)(half+1)), *Xi = malloc(8*(size_t)(half+1));
        for (size_t i = 0; i < (size_t)N*K; i++) x[i] = rnd();
        vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE;
        cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = K;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1; cfg.wisdom = g_W;
        vfft_plan p = vfft_create(&cfg);
        if (!p) judge("REGRESSION r2c K=4 (old path)", N, -1, 1);
        else {
            vfft_execute(p, VFFT_FORWARD, x, NULL, X, NULL);
            /* lane-batched split-engine geometry: element e of lane t at
             * [e*K + t]; the CCE plane mirrors it. Gate lane 1. */
            double xt[512];
            for (int e = 0; e < N; e++) xt[e] = x[(size_t)e*K + 1];
            naive_real_dft(xt, N, Xr, Xi);
            double w = 0, xm = 0;
            for (int f = 0; f <= half; f++){
                double a = fabs(Xr[f]) + fabs(Xi[f]); if (a > xm) xm = a;
                double dr = fabs(X[2*((size_t)f*K + 1)]     - Xr[f]);
                double di = fabs(X[2*((size_t)f*K + 1) + 1] - Xi[f]);
                if (dr > w) w = dr; if (di > w) w = di; }
            judge("REGRESSION r2c K=4 (old path)", N, w/xm, 1e-9);
            vfft_destroy(p);
        }
        free(x); free(X); free(Xr); free(Xi);
    }
    /* odd N interleaved r2c OOP — must still serve via the old path */
    {
        const int N = 510, half = N/2;
        double *x  = malloc(8*((size_t)N+2));
        double *X  = malloc(8*((size_t)N+2));
        double *Xr = malloc(8*(size_t)(half+1)), *Xi = malloc(8*(size_t)(half+1));
        for (int i = 0; i < N; i++) x[i] = rnd();
        naive_real_dft(x, N, Xr, Xi);
        vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE;
        cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1; cfg.wisdom = g_W;
        vfft_plan p = vfft_create(&cfg);
        if (!p) judge("REGRESSION r2c odd N (old path)", N, -1, 1);
        else {
            memset(X, 0, 8*((size_t)N+2));
            vfft_execute(p, VFFT_FORWARD, x, NULL, X, NULL);
            double w = 0, xm = 0;
            for (int f = 0; f <= half; f++){
                double a = fabs(Xr[f]) + fabs(Xi[f]); if (a > xm) xm = a;
                double dr = fabs(X[2*f]   - Xr[f]);
                double di = fabs(X[2*f+1] - Xi[f]);
                if (dr > w) w = dr; if (di > w) w = di; }
            judge("REGRESSION r2c odd N (old path)", N, w/xm, 1e-9);
            vfft_destroy(p);
        }
        free(x); free(X); free(Xr); free(Xi);
    }
}

int main(int argc, char **argv)
{
    const char *wdir = (argc >= 2) ? argv[1]
                                   : "../../src/dag-fft-compiler/generator/generated";
    g_W = vfft_wisdom_load(wdir);
    printf("zr2c FRONT-DOOR gate — public API, both transforms x placements x routes\n");
    printf("wisdom dir: %s (%s)\n\n", wdir, g_W ? "loaded" : "MISS — create may calibrate");
    const int Ns[] = { 512, 2048, 4096 };
    for (size_t i = 0; i < sizeof Ns / sizeof Ns[0]; i++)
        for (int route = 0; route <= 1; route++)
            run_cell(Ns[i], route);
    run_regressions();
    printf("\n%s\n", g_fail ? "ZR2C FRONT-DOOR GATE: FAILURE"
                            : "ZR2C FRONT-DOOR GATE: ALL CORRECT");
    if (g_W) vfft_wisdom_free(g_W);
    return g_fail;
}
