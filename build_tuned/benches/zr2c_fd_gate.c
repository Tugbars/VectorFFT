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

/* ── stderr tap (the k1z_inplace_gate / natural-front-gate pattern) ────────
 * The route race announces itself on stderr under VFFT_ZRACE_VERBOSE, so the
 * only way to assert "a race actually happened" from the public API is to
 * read it back. Log lives in the CWD, never inside a wisdom dir. */
static char g_errpath[512];
static long g_errpos = 0;
static int err_tap_open(void)
{
    snprintf(g_errpath, sizeof g_errpath, "_zr2c_fd_gate.log");
    if (!freopen(g_errpath, "w", stderr)) return 0;
    setvbuf(stderr, NULL, _IONBF, 0);
    g_errpos = 0;
    return 1;
}
static const char *err_tap_read(void)
{
    static char buf[16384];
    buf[0] = 0;
    fflush(stderr);
    FILE *f = fopen(g_errpath, "rb");
    if (!f) return buf;
    if (fseek(f, g_errpos, SEEK_SET) == 0) {
        size_t n = fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        g_errpos += (long)n;
    }
    fclose(f);
    return buf;
}

/* route 0/1 force the child route via env (the racing hook); route -1
 * REMOVES the env (msvcrt: putenv("VAR=") deletes) so create serves the
 * WISDOM path — banked kind-5 verdict, or the in-context race on a miss
 * (which banks into the loaded wisdom dir). */
static void run_cell(int N, int route)
{
    char lbl[64];
    static char env0[] = "VFFT_ZR2C_ROUTE=0", env1[] = "VFFT_ZR2C_ROUTE=1";
    static char envW[] = "VFFT_ZR2C_ROUTE=";
    putenv(route < 0 ? envW : (route ? env1 : env0));
    const char *rt = route < 0 ? "W" : (route ? "1" : "0");

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
    cfg.wisdom_write = 1;  /* measurement gate: banks must persist */
    vfft_plan p = vfft_create(&cfg);
    snprintf(lbl, sizeof lbl, "r2c OOP  route=%s", rt);
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
    snprintf(lbl, sizeof lbl, "r2c IN-PLACE  route=%s", rt);
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
    snprintf(lbl, sizeof lbl, "c2r OOP  route=%s", rt);
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
    snprintf(lbl, sizeof lbl, "c2r IN-PLACE  route=%s", rt);
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
        cfg.wisdom_write = 1;  /* measurement gate: banks must persist */
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
    /* N=510: EVEN non-pow2 -> the zr2c branch legitimately serves it
     * (child c2c at the ODD half 255) — covers the odd-half child. The
     * genuine old-path regression smoke is the K=4 cell above. */
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
        cfg.wisdom_write = 1;  /* measurement gate: banks must persist */
        vfft_plan p = vfft_create(&cfg);
        if (!p) judge("r2c non-pow2 510 (zr2c, half 255)", N, -1, 1);
        else {
            memset(X, 0, 8*((size_t)N+2));
            vfft_execute(p, VFFT_FORWARD, x, NULL, X, NULL);
            double w = 0, xm = 0;
            for (int f = 0; f <= half; f++){
                double a = fabs(Xr[f]) + fabs(Xi[f]); if (a > xm) xm = a;
                double dr = fabs(X[2*f]   - Xr[f]);
                double di = fabs(X[2*f+1] - Xi[f]);
                if (dr > w) w = dr; if (di > w) w = di; }
            judge("r2c non-pow2 510 (zr2c, half 255)", N, w/xm, 1e-9);
            vfft_destroy(p);
        }
        free(x); free(X); free(Xr); free(Xi);
    }
}

/* == TRANSFORM-CONTIGUOUS REAL BATCH (2026-08-22) =========================
 * K>1 interleaved r2c/c2r asked for by name. This is the ONLY geometry the
 * zr2c route can serve at K>1 -- the route reinterprets a transform's N
 * contiguous reals as N/2 complex points, and under lane-major the two halves
 * of one complex sample sit K apart, so the reinterpret is not expressible
 * there at all. Served as K independent K=1 transforms end to end, which is
 * the same wrapper the interleaved C2C batch has used since 2026-08-04.
 *
 * What each check is FOR:
 *   fwd      every transform independently correct vs a naive DFT, and
 *            correct AT ITS OWN OFFSET -- the failure this catches is a
 *            stride bug that computes K right answers in K wrong places, so
 *            each transform is seeded with DIFFERENT data (a shared seed
 *            would pass with every block aliased onto one another).
 *   bwd      the c2r mirror. The strides differ per direction (r2c reads N
 *            and writes N+2; c2r is the reverse), so this is NOT redundant
 *            with fwd -- it exercises the other stride pair.
 *   in-place ONE padded plane per transform, dre == sre. New capability:
 *            before this the in-place real contract was K==1 only.
 *   MT==ST   BITWISE. The wrapper's MT runs each worker on its own plan
 *            CLONE, so a clone that resolved to a different kernel would give
 *            a numerically fine but bit-different batch. Bitwise is the only
 *            assertion that catches that, and it is the same rule the C2C
 *            wrapper is held to.
 */
static void tc_batch_fwd(int N, size_t K, int nthreads, double *out)
{
    const size_t nb = (size_t)N/2 + 1;
    double *x = (double *)malloc(8*(size_t)N*K);
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = K;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
    cfg.nthreads = nthreads; cfg.wisdom = g_W; cfg.wisdom_write = 1;
    /* per-transform data: block t is seeded from a t-dependent stream, so a
     * stride bug cannot pass by aliasing the blocks together.
     *
     * RESEED FIRST. rnd() is a FILE-GLOBAL LCG, so two calls to this function
     * draw two different input sets -- and the MT==ST check below compares the
     * outputs of two calls. Without this line that check compares transforms
     * of different data and reports DIFFERS every time, which is a defect in
     * the gate that looks exactly like a defect in the library. */
    lcg = 0x243F6A8885A308D3ull;
    for (size_t t = 0; t < K; t++)
        for (int e = 0; e < N; e++) x[t*(size_t)N + e] = rnd() + 0.25*(double)t;
    vfft_plan p = vfft_create(&cfg);
    if (!p) { free(x); out[0] = -1.0; return; }
    memset(out+1, 0, 8*2*nb*K);
    vfft_execute(p, VFFT_FORWARD, x, NULL, out+1, NULL);
    {
        double *Xr = (double *)malloc(8*nb), *Xi = (double *)malloc(8*nb);
        double w = 0, xm = 0;
        size_t t, f;
        for (t = 0; t < K; t++) {
            naive_real_dft(x + t*(size_t)N, N, Xr, Xi);
            for (f = 0; f < nb; f++) {
                double a = fabs(Xr[f]) + fabs(Xi[f]); if (a > xm) xm = a;
                {
                    double dr = fabs(out[1 + t*2*nb + 2*f]     - Xr[f]);
                    double di = fabs(out[1 + t*2*nb + 2*f + 1] - Xi[f]);
                    if (dr > w) w = dr; if (di > w) w = di;
                }
            }
        }
        out[0] = w/xm;
        free(Xr); free(Xi);
    }
    free(x); vfft_destroy(p);
}

static void run_tc_batch(void)
{
    const int Ns[2] = { 512, 2048 };
    const size_t K = 4;
    size_t i;
    for (i = 0; i < 2; i++) {
        const int N = Ns[i];
        const size_t nb = (size_t)N/2 + 1;

        /* forward, single-threaded */
        double *a = (double *)malloc(8*(1 + 2*nb*K));
        double *b = (double *)malloc(8*(1 + 2*nb*K));
        tc_batch_fwd(N, K, 1, a);
        judge("TC batch r2c K=4 fwd (vs naive)", N, a[0], 1e-9);

        /* MT == ST, BITWISE */
        tc_batch_fwd(N, K, 4, b);
        if (a[0] < 0 || b[0] < 0) judge("TC batch r2c K=4 MT==ST", N, -1, 1);
        else {
            int same = (memcmp(a+1, b+1, 8*2*nb*K) == 0);
            printf("  %-38s N=%-6d %-9s %s\n", "TC batch r2c K=4 MT==ST (bitwise)",
                   N, same ? "identical" : "DIFFERS", same ? "OK" : "*** FAIL ***");
            if (!same) g_fail = 1;
        }
        free(a); free(b);

        /* backward (c2r): the OTHER stride pair */
        {
            double *X  = (double *)malloc(8*2*nb*K);
            double *y  = (double *)malloc(8*(size_t)N*K);
            double *x0 = (double *)malloc(8*(size_t)N*K);
            vfft_config_t cf, cb; vfft_plan pf, pb;
            size_t j;
            memset(&cf, 0, sizeof cf);
            cf.transform = VFFT_R2C; cf.placement = VFFT_OUTOFPLACE;
            cf.rigor = VFFT_MEASURE; cf.dims = 1; cf.n[0] = N; cf.howmany = K;
            cf.layout = VFFT_LAYOUT_INTERLEAVED;
            cf.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
            cf.nthreads = 1; cf.wisdom = g_W; cf.wisdom_write = 1;
            cb = cf; cb.transform = VFFT_C2R;
            for (j = 0; j < (size_t)N*K; j++) x0[j] = rnd();
            pf = vfft_create(&cf); pb = vfft_create(&cb);
            if (!pf || !pb) judge("TC batch c2r K=4 roundtrip", N, -1, 1);
            else {
                double w = 0, xm = 0;
                vfft_execute(pf, VFFT_FORWARD,  x0, NULL, X, NULL);
                vfft_execute(pb, VFFT_BACKWARD, X,  NULL, y, NULL);
                for (j = 0; j < (size_t)N*K; j++) {
                    double d = fabs(y[j]/(double)N - x0[j]);
                    if (d > w) w = d;
                    if (fabs(x0[j]) > xm) xm = fabs(x0[j]);
                }
                judge("TC batch c2r K=4 roundtrip", N, w/xm, 1e-12);
            }
            if (pf) vfft_destroy(pf);
            if (pb) vfft_destroy(pb);
            free(X); free(y); free(x0);
        }

        /* IN-PLACE, K>1: one padded plane per transform */
        {
            double *z   = (double *)malloc(8*2*nb*K);
            double *ref = (double *)malloc(8*(size_t)N*K);
            double *Xr  = (double *)malloc(8*nb);
            double *Xi  = (double *)malloc(8*nb);
            vfft_config_t cfg; vfft_plan p;
            size_t t; int e;
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_R2C; cfg.placement = VFFT_INPLACE;
            cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = K;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED;
            cfg.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
            cfg.nthreads = 1; cfg.wisdom = g_W; cfg.wisdom_write = 1;
            memset(z, 0, 8*2*nb*K);
            for (t = 0; t < K; t++)
                for (e = 0; e < N; e++) {
                    double v = rnd();
                    ref[t*(size_t)N + (size_t)e] = v;
                    z[t*2*nb + (size_t)e] = v;   /* reals at the front of the plane */
                }
            p = vfft_create(&cfg);
            if (!p) judge("TC batch r2c K=4 IN-PLACE", N, -1, 1);
            else {
                double w = 0, xm = 0;
                size_t f;
                vfft_execute(p, VFFT_FORWARD, z, NULL, z, NULL);
                for (t = 0; t < K; t++) {
                    naive_real_dft(ref + t*(size_t)N, N, Xr, Xi);
                    for (f = 0; f < nb; f++) {
                        double m = fabs(Xr[f]) + fabs(Xi[f]); if (m > xm) xm = m;
                        {
                            double dr = fabs(z[t*2*nb + 2*f]     - Xr[f]);
                            double di = fabs(z[t*2*nb + 2*f + 1] - Xi[f]);
                            if (dr > w) w = dr; if (di > w) w = di;
                        }
                    }
                }
                judge("TC batch r2c K=4 IN-PLACE (vs naive)", N, w/xm, 1e-9);
                vfft_destroy(p);
            }
            free(z); free(ref); free(Xr); free(Xi);
        }
    }
}

/* ── COLD -> REPLAY: the route axis' own lifecycle ────────────────────────
 * 🔴 The route bit is the ONLY verdict kind-5 wisdom owns, and nothing in
 * the tree asserted that it is ever actually RACED, or that a banked verdict
 * is then REPLAYED rather than re-raced. Both halves matter: a cell that
 * never races has no verdict, and a cell that re-races on every create has a
 * verdict nobody reads.
 *
 * A wisdom handle loaded from a path that does not exist is a valid EMPTY
 * bundle (documented at the resolution site below), which is exactly a cold
 * store -- and it can never touch the shipped one.
 *
 * ZERO timing assertions: this checks THAT a race ran and that replay is
 * byte-identical, never how fast anything was. Safe on a noisy machine. */
static void run_cold_replay(int N)
{
    static char envW[] = "VFFT_ZR2C_ROUTE=";
    static char envV[] = "VFFT_ZRACE_VERBOSE=1";
    putenv(envW);                       /* wisdom decides, not the env hook */
    putenv(envV);

    vfft_wisdom *W = vfft_wisdom_load("_zr2c_cold_gate_no_such_dir");
    size_t xs = (size_t)N + 2;
    double *x = (double *)malloc(8 * xs);
    double *A = (double *)malloc(8 * xs);
    double *B = (double *)malloc(8 * xs);
    if (!x || !A || !B) { free(x); free(A); free(B); return; }
    for (int i = 0; i < N; i++) x[i] = rnd();
    x[N] = x[N + 1] = 0.0;

    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1; cfg.wisdom = W;

    (void)err_tap_read();               /* drop anything already buffered */

    vfft_plan p1 = vfft_create(&cfg);
    if (p1) { vfft_execute(p1, VFFT_FORWARD, x, NULL, A, NULL); vfft_destroy(p1); }
    int raced1 = strstr(err_tap_read(), "route race") != NULL;

    vfft_plan p2 = vfft_create(&cfg);   /* SAME handle: the verdict is banked */
    if (p2) { vfft_execute(p2, VFFT_FORWARD, x, NULL, B, NULL); vfft_destroy(p2); }
    int raced2 = strstr(err_tap_read(), "route race") != NULL;

    int same = (p1 && p2) &&
               memcmp(A, B, (size_t)(2 * (N / 2 + 1)) * sizeof(double)) == 0;

    printf("  cold->replay N=%-6d  built=%d/%d  raced=%d/%d  replay %s   %s\n",
           N, p1 ? 1 : 0, p2 ? 1 : 0, raced1, raced2,
           same ? "byte-identical" : "DIFFERS",
           (p1 && p2 && raced1 && !raced2 && same) ? "OK" : "*** FAIL ***");
    if (!(p1 && p2 && raced1 && !raced2 && same)) g_fail = 1;

    if (W) vfft_wisdom_free(W);
    free(x); free(A); free(B);
}

int main(int argc, char **argv)
{
    /* CWD-proof wisdom resolution: build.py runs binaries from build_tuned/
     * while a manual run starts in benches/ — probe both relative roots and
     * take the one whose oop_wisdom.txt actually opens. vfft_wisdom_load
     * returns a (valid, EMPTY) bundle on a total miss, so a pointer check
     * alone would hide the miss — hence the fopen probe + row-count print. */
    const char *cand[3] = {
        (argc >= 2) ? argv[1] : NULL,
        "../src/dag-fft-compiler/generator/generated",    /* from build_tuned */
        "../../src/dag-fft-compiler/generator/generated", /* from benches */
    };
    const char *wdir = NULL;
    for (int i = 0; i < 3 && !wdir; i++) {
        if (!cand[i]) continue;
        char pp[512];
        snprintf(pp, sizeof pp, "%s/wisdom2_oop.txt", cand[i]); /* dir marker = the LIVE store (legacy file is frozen) */
        FILE *pf = fopen(pp, "r");
        if (pf) { fclose(pf); wdir = cand[i]; }
    }
    printf("zr2c FRONT-DOOR gate — public API, both transforms x placements x routes\n");
    if (!wdir) {
        printf("wisdom dir: NOT FOUND (probed argv[1] + both relative roots) "
               "— refusing to gate against empty wisdom\n");
        return 1;
    }
    g_W = vfft_wisdom_load(wdir);
    printf("wisdom dir: %s\n\n", wdir);
    const int Ns[] = { 512, 2048, 4096 };
    for (size_t i = 0; i < sizeof Ns / sizeof Ns[0]; i++)
        for (int route = 0; route <= 1; route++)
            run_cell(Ns[i], route);
    for (size_t i = 0; i < sizeof Ns / sizeof Ns[0]; i++)
        run_cell(Ns[i], -1); /* wisdom pass: race-and-bank on miss */
    run_regressions();
    run_tc_batch();
    /* the tap must be open BEFORE the first cold create, and only now --
     * redirecting earlier would swallow the diagnostics the passes above
     * rely on a human seeing. */
    if (!err_tap_open())
        printf("  cold->replay SKIPPED: stderr tap failed\n");
    else {
        run_cold_replay(1024);
        run_cold_replay(2048);
    }
    printf("\n%s\n", g_fail ? "ZR2C FRONT-DOOR GATE: FAILURE"
                            : "ZR2C FRONT-DOOR GATE: ALL CORRECT");
    if (g_W) vfft_wisdom_free(g_W);
    return g_fail;
}
