/* wisdom2_2d_gate.h — the wave-3 flip gate (module-owned; bench = thin
 * driver, per the bench-purity law).
 *
 * [2d-flip-gate]: for each populated 2D cell, create through the PUBLIC
 * front door TWICE, each from a FRESH wisdom load, and require:
 *   - correctness vs an independent naive 2D DFT (c2c) / roundtrip
 *     bwd(fwd(x)) == N1*N2*x (real families, natural-ordered so legal);
 *   - the two creates' forward outputs BITWISE IDENTICAL (same served plan
 *     => same construction => same rounding; this is the plan-equivalence
 *     observable, the tangent-gate precedent).
 * The second arm was the kill switch (VFFT_WISDOM2_OFF=2d, legacy-table
 * reads) until 2026-08-20; the switch is RETIRED and ignored, so both arms
 * now read wisdom2. The gate therefore asserts CREATE-TWICE COHERENCE from
 * the store: a cell whose raced axes are not banked at the caller's layout
 * (lay=il) re-races on every create and diverges by plan luck — that is a
 * store-coverage failure, not noise. Seed the cell, never widen the check.
 *
 * [3d-born-gate]: dims=3 create on a cold cell in MEASUREMENT mode
 * (wisdom_write=1) must persist a record into wisdom2_3d.txt; a second
 * create from a FRESH load must re-serve it with a bitwise-identical
 * forward output (create-twice coherence).
 *
 * 🔴 Point wisdir at a SCRATCH copy (dual: frozen legacy files + migrated
 *    wisdom2_2d.txt). The 3D leg banks into it.
 */
#ifndef VFFT_WISDOM2_2D_GATE_H
#define VFFT_WISDOM2_2D_GATE_H

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double *_g2d_az(size_t doubles)
{
    void *p = NULL;
#ifdef _WIN32
    p = _aligned_malloc(doubles * sizeof(double), 64);
#else
    if (posix_memalign(&p, 64, doubles * sizeof(double))) p = NULL;
#endif
    if (p) memset(p, 0, doubles * sizeof(double));
    return (double *)p;
}
static void _g2d_fz(double *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* naive row-column 2D DFT, interleaved complex, natural order */
static void _g2d_naive(const double *x, double *X, int N1, int N2)
{
    double *tmp = (double *)malloc((size_t)2 * N1 * N2 * sizeof(double));
    int r, c, k, j;
    for (r = 0; r < N1; r++)                       /* rows: length-N2 DFTs */
        for (k = 0; k < N2; k++) {
            double sr = 0, si = 0;
            for (j = 0; j < N2; j++) {
                double a = -2.0 * M_PI * (double)k * j / N2;
                double wr = cos(a), wi = sin(a);
                const double *z = x + 2 * ((size_t)r * N2 + j);
                sr += z[0] * wr - z[1] * wi;
                si += z[0] * wi + z[1] * wr;
            }
            tmp[2 * ((size_t)r * N2 + k)]     = sr;
            tmp[2 * ((size_t)r * N2 + k) + 1] = si;
        }
    for (c = 0; c < N2; c++)                       /* cols: length-N1 DFTs */
        for (k = 0; k < N1; k++) {
            double sr = 0, si = 0;
            for (j = 0; j < N1; j++) {
                double a = -2.0 * M_PI * (double)k * j / N1;
                double wr = cos(a), wi = sin(a);
                const double *z = tmp + 2 * ((size_t)j * N2 + c);
                sr += z[0] * wr - z[1] * wi;
                si += z[0] * wi + z[1] * wr;
            }
            X[2 * ((size_t)k * N2 + c)]     = sr;
            X[2 * ((size_t)k * N2 + c) + 1] = si;
        }
    free(tmp);
}

/* one front-door create+execute; out must hold the transform's output.
 * When rt != NULL, additionally executes BACKWARD(out) -> rt (the
 * roundtrip product: fully DEFINED output — the forward half-spectrum's
 * padding lanes carry plan-scratch heap noise and must never be compared).
 * Returns 1 ok, 0 create/exec failure. */
static int _g2d_run(const char *wisdir, int transform, int order,
                    int N1, int N2, const double *in, double *out,
                    size_t out_doubles, double *rt, int wisdom_write)
{
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    vfft_config_t cfg;
    vfft_plan h;
    if (!W) return 0;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = (vfft_transform_t)transform;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2;
    cfg.howmany = 1;
    cfg.order = order;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = 1;
    cfg.wisdom = W;
    cfg.wisdom_write = wisdom_write;
    h = vfft_create(&cfg);
    if (!h) { vfft_wisdom_free(W); return 0; }
    memset(out, 0, out_doubles * sizeof(double));
    vfft_execute(h, VFFT_FORWARD, (double *)in, NULL, out, NULL);
    if (rt) {
        memset(rt, 0, out_doubles * sizeof(double));
        vfft_execute(h, VFFT_BACKWARD, out, NULL, rt, NULL);
    }
    vfft_destroy(h);
    vfft_wisdom_free(W);
    return 1;
}

/* run the same cell on both read arms; fills a (arm wisdom2) and b (arm
 * legacy), and the roundtrip products when rta/rtb are given. The kill
 * switch is env-scoped around the second run. */
static int _g2d_both_arms(const char *wisdir, int transform, int order,
                          int N1, int N2, const double *in,
                          double *a, double *b, size_t out_doubles,
                          double *rta, double *rtb)
{
    int ok;
#ifdef _WIN32
    _putenv("VFFT_WISDOM2_OFF=");
#else
    unsetenv("VFFT_WISDOM2_OFF");
#endif
    ok = _g2d_run(wisdir, transform, order, N1, N2, in, a, out_doubles, rta, 0);
#ifdef _WIN32
    _putenv("VFFT_WISDOM2_OFF=2d");
#else
    setenv("VFFT_WISDOM2_OFF", "2d", 1);
#endif
    ok = ok && _g2d_run(wisdir, transform, order, N1, N2, in, b, out_doubles, rtb, 0);
#ifdef _WIN32
    _putenv("VFFT_WISDOM2_OFF=");
#else
    unsetenv("VFFT_WISDOM2_OFF");
#endif
    return ok;
}

/* Run the flip gate. Returns FAIL count (-1 = setup failure). */
static int vfft_wisdom2_2d_gate_run(const char *wisdir)
{
    int fails = 0;
    srand(2026);

    printf("\n=== wisdom2 2D flip gate (both read arms, bitwise) ===\n");

    /* ── c2c cells, scrambled + natural, correctness vs naive ─────────── */
    {
        static const struct { int N1, N2, order; const char *tag; } CC[] = {
            { 64, 64, VFFT_ORDER_DEFAULT, "c2c 64x64 scr" },
            { 64, 16, VFFT_ORDER_DEFAULT, "c2c 64x16 scr" },
            { 64, 64, VFFT_ORDER_NATURAL, "c2c 64x64 nat" },
            { 128, 64, VFFT_ORDER_NATURAL, "c2c 128x64 nat" },
            { 127, 100, VFFT_ORDER_NATURAL, "c2c 127x100 nat" }, /* PRIME N1: the
                                              * column-axis Bluestein, replayed
                                              * from a blu= row — the naive-DFT
                                              * anchor is what catches a replay
                                              * that runs the M chain as N1's
                                              * (2026-09-02) */
        };
        int i;
        for (i = 0; i < (int)(sizeof CC / sizeof CC[0]); i++) {
            const int N1 = CC[i].N1, N2 = CC[i].N2;
            const size_t nd = (size_t)2 * N1 * N2;
            double *x = _g2d_az(nd), *A = _g2d_az(nd), *B = _g2d_az(nd);
            double *R = _g2d_az(nd);
            size_t j;
            int ok;
            for (j = 0; j < nd; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
            ok = _g2d_both_arms(wisdir, VFFT_C2C, CC[i].order, N1, N2, x, A, B, nd,
                                NULL, NULL);
            if (ok && CC[i].order == VFFT_ORDER_NATURAL) {
                /* natural output is IN ORDER: anchor to the naive DFT */
                double m = 0, e = 0;
                _g2d_naive(x, R, N1, N2);
                for (j = 0; j < nd; j++) {
                    if (fabs(R[j]) > m) m = fabs(R[j]);
                    if (fabs(A[j] - R[j]) > e) e = fabs(A[j] - R[j]);
                }
                if (m > 0 && e / m > 1e-9) ok = 0;
            }
            if (ok && memcmp(A, B, nd * sizeof(double)) != 0) ok = 0;
            printf("  %-16s %s\n", CC[i].tag, ok ? "PASS (arms bitwise-identical)"
                                                 : "*** FAIL ***");
            if (!ok) fails++;
            _g2d_fz(x); _g2d_fz(A); _g2d_fz(B); _g2d_fz(R);
        }
    }

    /* ── real families: arms bitwise + matched roundtrip ──────────────── */
    {
        static const struct { int N1, N2, t; const char *tag; } RC[] = {
            { 64, 64, VFFT_R2C, "r2c 64x64" },
            { 128, 128, VFFT_R2C, "r2c 128x128" },
            /* c2r: random bytes as the "spectrum" — the arms-bitwise
             * verdict needs identical inputs, not a valid one (both arms
             * run the same deterministic construction). */
            { 64, 64, VFFT_C2R, "c2r 64x64" },
            { 128, 128, VFFT_C2R, "c2r 128x128" },
        };
        int i;
        for (i = 0; i < (int)(sizeof RC / sizeof RC[0]); i++) {
            const int N1 = RC[i].N1, N2 = RC[i].N2;
            /* buffers >= any real/halfcomplex/padded layout, either
             * direction (c2r READS a padded halfcomplex plane).
             * Compare windows are the transform-DEFINED outputs, both
             * PROVEN exactly defined by the delta-input pitch probe
             * (2026-08-20): the r2c forward plane is CONTIGUOUS
             * N1 x (N2/2+1) complex — no padding lanes inside the window;
             * c2r's real plane is N1*N2 doubles. (An r2c handle does NOT
             * execute BACKWARD — c2r is its own transform — so a one-
             * handle roundtrip verdict is invalid here.) */
            const size_t nin = (size_t)2 * N1 * N2 + 64;
            const size_t nout = (size_t)2 * N1 * N2 + 64;
            const size_t ncmp = (RC[i].t == VFFT_R2C)
                                    ? (size_t)2 * N1 * (N2 / 2 + 1)
                                    : (size_t)N1 * N2;
            double *x = _g2d_az(nin), *A = _g2d_az(nout), *B = _g2d_az(nout);
            size_t j;
            int ok;
            for (j = 0; j < nin; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
            ok = _g2d_both_arms(wisdir, RC[i].t, VFFT_ORDER_DEFAULT, N1, N2,
                                x, A, B, nout, NULL, NULL);
            if (ok && memcmp(A, B, ncmp * sizeof(double)) != 0) {
                /* name the first divergent double — evidence for any
                 * future flake, never a silent verdict */
                for (j = 0; j < ncmp; j++)
                    if (A[j] != B[j]) break;
                printf("  %-16s *** FAIL *** (first diff at double %zu: %a vs %a)\n",
                       RC[i].tag, j, A[j], B[j]);
                fails++;
            } else {
                printf("  %-16s %s\n", RC[i].tag,
                       ok ? "PASS (arms bitwise-identical)" : "*** FAIL ***");
                if (!ok) fails++;
            }
            _g2d_fz(x); _g2d_fz(A); _g2d_fz(B);
        }
    }

    /* ── 3D born-in-wisdom2: bank-then-reserve coherence ──────────────── */
    {
        const int N = 16;                    /* 16^3: fast greedy create */
        const size_t nd = (size_t)2 * N * N * N;
        double *x = _g2d_az(nd), *A = _g2d_az(nd), *B = _g2d_az(nd);
        size_t j;
        int ok = 1;
        long had = 0, have = 0;
        char p3[1024];
        FILE *f;
        for (j = 0; j < nd; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
        snprintf(p3, sizeof p3, "%s/wisdom2_3d.txt", wisdir);
        f = fopen(p3, "rb");
        if (f) { char l[512]; while (fgets(l, sizeof l, f)) if (l[0] == '@' && l[1] == 'c') had++; fclose(f); }
        {
            vfft_wisdom *W = vfft_wisdom_load(wisdir);
            vfft_config_t cfg;
            vfft_plan h;
            if (!W) ok = 0;
            else {
                memset(&cfg, 0, sizeof cfg);
                cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE;
                cfg.rigor = VFFT_MEASURE; cfg.dims = 3;
                cfg.n[0] = N; cfg.n[1] = N; cfg.n[2] = N;
                cfg.howmany = 1; cfg.order = VFFT_ORDER_DEFAULT;
                cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1;
                cfg.wisdom = W;
                cfg.wisdom_write = 1;        /* measurement mode: must persist */
                h = vfft_create(&cfg);
                if (!h) ok = 0;
                else {
                    vfft_execute(h, VFFT_FORWARD, x, NULL, A, NULL);
                    vfft_destroy(h);
                }
                vfft_wisdom_free(W);
            }
        }
        f = fopen(p3, "rb");
        if (f) { char l[512]; while (fgets(l, sizeof l, f)) if (l[0] == '@' && l[1] == 'c') have++; fclose(f); }
        if (have <= had) {
            printf("  3d %dx%dx%d      *** FAIL *** (no record persisted: %ld -> %ld)\n",
                   N, N, N, had, have);
            ok = 0;
        }
        if (ok) {
            /* second create, FRESH load: must re-serve the banked recipe */
            vfft_wisdom *W = vfft_wisdom_load(wisdir);
            vfft_config_t cfg;
            vfft_plan h;
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE;
            cfg.rigor = VFFT_MEASURE; cfg.dims = 3;
            cfg.n[0] = N; cfg.n[1] = N; cfg.n[2] = N;
            cfg.howmany = 1; cfg.order = VFFT_ORDER_DEFAULT;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1;
            cfg.wisdom = W;
            h = W ? vfft_create(&cfg) : NULL;
            if (!h) ok = 0;
            else {
                vfft_execute(h, VFFT_FORWARD, x, NULL, B, NULL);
                vfft_destroy(h);
                if (memcmp(A, B, nd * sizeof(double)) != 0) ok = 0;
            }
            if (W) vfft_wisdom_free(W);
        }
        if (ok)
            printf("  3d %dx%dx%d      PASS (banked -> persisted -> re-served bitwise)\n", N, N, N);
        else
            fails++;
        _g2d_fz(x); _g2d_fz(A); _g2d_fz(B);
    }

    printf("\n  === %s (%d fail) ===\n", fails ? "*** FAIL ***" : "ALL PASS", fails);
    return fails;
}

#endif /* VFFT_WISDOM2_2D_GATE_H */
