/* il2d_real_gate.c — M1 gate for the native IL 2D REAL tier
 * (docs/roadmap/fft2d_real_il_design.md; driver in vfft.c, env-gated
 * VFFT_IL2D_REAL=1 for M1).
 *
 * FWD (r2c): elementwise vs a naive separable real DFT through the
 * chain's row map — N1 digit-reversed at nst>1, so the mapped compare is
 * SELF-PROVING on engagement there (the §6a30 veneer serves natural-N1
 * CCE and can never pass it). A cell where the NATURAL compare passes
 * instead is the designed fall-through (veneer served) and reports SKIP,
 * not FAIL. nst==1 cells cannot distinguish by order — their engagement
 * proof is the [il2d-real] stderr log (runner counts lines).
 *
 * BWD (c2r): the PAIR contract — the c2r plan consumes the r2c pair's
 * own fwd output (the comb) and must return N1*N2*x elementwise; fwd is
 * independently proven above, so this pins bwd (the roundtrip-cannot-
 * gate law is satisfied by the fwd arm). PLUS the §2.6 contract: the
 * caller's z is INPUT-PRESERVED (bitwise) across the c2r execute.
 *
 * Cells cover BOTH hp1 parities (§2.1: hp1 = N2/2+1 is odd iff 4|N2,
 * even at N2 == 2 mod 4) — the first-ever odd-count n1c/t2c coverage.
 *
 * SECOND PASS: the ROWSPLIT row route (VFFT_IL2D_ROWSPLIT=W — "il at
 * the boundary, split inside", the cascade pattern) over the W-legal
 * subset (W%8, W|N1, N2%4): identical checks, identical tolerances.
 *
 * Build: python build.py --src benches/il2d_real_gate.c --vfft --compile
 * Run  : il2d_real_gate.exe <SCRATCH wisdir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

static void naive_rows_r2c(const double *x, double *t1, int N1, int N2)
{
    const double pi = 3.14159265358979323846;
    const int hp1 = N2 / 2 + 1;
    int i, k, j;
    for (i = 0; i < N1; i++)
        for (k = 0; k < hp1; k++) {
            double sr = 0.0, si = 0.0;
            for (j = 0; j < N2; j++) {
                const double a = -2.0 * pi * (double)(j * k) / (double)N2;
                sr += x[(size_t)i * N2 + j] * cos(a);
                si += x[(size_t)i * N2 + j] * sin(a);
            }
            t1[2 * ((size_t)i * hp1 + k)] = sr;
            t1[2 * ((size_t)i * hp1 + k) + 1] = si;
        }
}

static void naive_cols(const double *in, double *out, int N1, int hp1)
{
    const double pi = 3.14159265358979323846;
    int k, ki, i;
    for (ki = 0; ki < N1; ki++)
        for (k = 0; k < hp1; k++) {
            double sr = 0.0, si = 0.0;
            for (i = 0; i < N1; i++) {
                const double a =
                    -2.0 * pi * (double)((size_t)ki * i % (size_t)N1)
                    / (double)N1;
                const double c = cos(a), s = sin(a);
                const size_t x = 2 * ((size_t)i * hp1 + k);
                sr += in[x] * c - in[x + 1] * s;
                si += in[x] * s + in[x + 1] * c;
            }
            out[2 * ((size_t)ki * hp1 + k)] = sr;
            out[2 * ((size_t)ki * hp1 + k) + 1] = si;
        }
}

/* the driver's greedy chain (MIRRORED from _il2d_build_chain; the gate
 * pins it via env so the map below is the binding) + the DIF output map:
 * the FIRST stage's digit is the MOST significant position digit. */
static int chain_of(int N1, int *Rs)
{
    static const int POOL[] = { 64, 32, 16, 8, 4 };
    int L = N1, m = 0;
    while (L > 1) {
        int p, R = 0;
        if (m >= 8) return 0;
        for (p = 0; p < 5; p++)
            if (L % POOL[p] == 0 && (L / POOL[p] == 1 || L / POOL[p] >= 4)) {
                R = POOL[p];
                break;
            }
        if (!R) return 0;
        Rs[m++] = R;
        L /= R;
    }
    return m;
}

static int row_pos(int k, int N1, const int *Rs, int nf)
{
    int pos = 0, w = N1, s;
    for (s = 0; s < nf; s++) {
        w /= Rs[s];
        pos += (k % Rs[s]) * w;
        k /= Rs[s];
    }
    return pos;
}

/* one full cell: fwd mapped-elementwise + c2r pair + §2.6 preservation.
 * Returns the fail count; *skips bumped on the designed fall-through. */
static int run_cell(vfft_wisdom *W, int N1, int N2, const char *tag,
                    int *skips)
{
    const int hp1 = N2 / 2 + 1;
    const size_t RN = (size_t)N1 * N2, CN = (size_t)N1 * hp1;
    double *x = malloc(RN * 8), *xr = malloc(RN * 8);
    double *z = malloc(2 * CN * 8), *zc = malloc(2 * CN * 8);
    double *t1 = malloc(2 * CN * 8), *ref = malloc(2 * CN * 8);
    int Rs[8], nf, engaged = 0, fails = 0;
    size_t i;
    if (!x || !xr || !z || !zc || !t1 || !ref) {
        printf("OOM\n");
        exit(2);
    }
    nf = chain_of(N1, Rs);
    if (nf == 0) { Rs[0] = N1; nf = 1; }
    {
        char cb[80];
        int s_, o_ = snprintf(cb, sizeof cb, "VFFT_IL2D_CHAIN=");
        for (s_ = 0; s_ < nf; s_++)
            o_ += snprintf(cb + o_, sizeof cb - o_, "%s%d",
                           s_ ? "." : "", Rs[s_]);
#ifdef _WIN32
        _putenv(cb);
#else
        putenv(strdup(cb));
#endif
    }
    for (i = 0; i < RN; i++)
        x[i] = (double)rand() / RAND_MAX - 0.5;
    fprintf(stderr, "[gate] %s cell %dx%d (hp1=%d %s, nst=%d)...\n",
            tag, N1, N2, hp1, (hp1 & 1) ? "odd" : "even", nf);
    naive_rows_r2c(x, t1, N1, N2);
    naive_cols(t1, ref, N1, hp1);
    /* ── FWD ── */
    {
        vfft_config_t cfg;
        vfft_plan h;
        double em = 0, en = 0, mag = 0, relm, reln;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_R2C;
        cfg.placement = VFFT_OUTOFPLACE;
        cfg.rigor = VFFT_MEASURE;
        cfg.dims = 2;
        cfg.n[0] = N1;
        cfg.n[1] = N2;
        cfg.howmany = 1;
        cfg.order = VFFT_ORDER_DEFAULT;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.nthreads = 1;
        cfg.wisdom = W;
        cfg.wisdom_write = 0;
        h = vfft_create(&cfg);
        if (!h) {
            printf("  %s %4dx%-3d r2c create FAIL\n", tag, N1, N2);
            fails++;
            goto done;
        }
        memset(z, 0, 2 * CN * 8);
        vfft_execute(h, VFFT_FORWARD, x, NULL, z, NULL);
        vfft_destroy(h);
        {
            int ki, kj;
            for (ki = 0; ki < N1; ki++) {
                const int p_ = row_pos(ki, N1, Rs, nf);
                for (kj = 0; kj < 2 * hp1; kj++) {
                    const double r = ref[2 * (size_t)ki * hp1 + kj];
                    double d;
                    if (fabs(r) > mag) mag = fabs(r);
                    d = fabs(z[2 * (size_t)p_ * hp1 + kj] - r);
                    if (d > em) em = d;
                    d = fabs(z[2 * (size_t)ki * hp1 + kj] - r);
                    if (d > en) en = d;
                }
            }
        }
        relm = em / (mag > 0 ? mag : 1.0);
        reln = en / (mag > 0 ? mag : 1.0);
        if (relm < 1e-10) {
            engaged = 1;
            printf("  %s %4dx%-3d r2c fwd rel %.2e  PASS%s\n", tag, N1,
                   N2, relm, nf > 1 ? " (engagement self-proven)" : "");
        } else if (nf > 1 && reln < 1e-10) {
            printf("  %s %4dx%-3d r2c SKIP (tier fell through — veneer "
                   "served natural)\n", tag, N1, N2);
            (*skips)++;
        } else if (nf == 1 && reln < 1e-10) {
            engaged = 1;
            printf("  %s %4dx%-3d r2c fwd rel %.2e  PASS (order-blind "
                   "cell — see [il2d-real] log)\n", tag, N1, N2, reln);
        } else {
            printf("  %s %4dx%-3d r2c fwd rel %.2e (natural %.2e) "
                   "*** FAIL ***\n", tag, N1, N2, relm, reln);
            fails++;
        }
    }
    /* ── BWD ── */
    if (engaged) {
        vfft_config_t cfg;
        vfft_plan h;
        double rt = 0;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2R;
        cfg.placement = VFFT_OUTOFPLACE;
        cfg.rigor = VFFT_MEASURE;
        cfg.dims = 2;
        cfg.n[0] = N1;
        cfg.n[1] = N2;
        cfg.howmany = 1;
        cfg.order = VFFT_ORDER_DEFAULT;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.nthreads = 1;
        cfg.wisdom = W;
        cfg.wisdom_write = 0;
        h = vfft_create(&cfg);
        if (!h) {
            printf("  %s %4dx%-3d c2r create FAIL\n", tag, N1, N2);
            fails++;
            goto done;
        }
        memcpy(zc, z, 2 * CN * 8);
        memset(xr, 0, RN * 8);
        vfft_execute(h, VFFT_BACKWARD, z, NULL, xr, NULL);
        vfft_destroy(h);
        for (i = 0; i < RN; i++) {
            const double d = fabs(xr[i] / ((double)N1 * N2) - x[i]);
            if (d > rt) rt = d;
        }
        if (memcmp(z, zc, 2 * CN * 8) != 0) {
            printf("  %s %4dx%-3d c2r INPUT DESTROYED (§2.6 contract) "
                   "*** FAIL ***\n", tag, N1, N2);
            fails++;
        }
        printf("  %s %4dx%-3d c2r pair rt %.2e  %s%s\n", tag, N1, N2, rt,
               rt < 1e-10 ? "PASS" : "*** FAIL ***",
               nf > 1 ? " (engagement self-proven)" : "");
        if (rt >= 1e-10) fails++;
    }
done:
    free(x); free(xr); free(z); free(zc); free(t1); free(ref);
    return fails;
}

int main(int argc, char **argv)
{
    /* both hp1 parities at single-stage AND multi-stage N1 */
    static const int CELLS[][2] = {
        { 16, 8 },   { 16, 10 },  { 32, 16 },  { 64, 64 },  { 64, 10 },
        { 128, 8 },  { 128, 64 }, { 256, 16 }, { 256, 10 }, { 512, 32 },
        { 1024, 16 }, { 1024, 18 }, { 4096, 8 },
    };
    /* ROWSPLIT subset: W=64 needs N1%64==0 && N2%4==0 (plus one W=32
     * arm at a mid cell for the width axis). */
    static const int RS64[][2] = {
        { 128, 8 }, { 128, 64 }, { 256, 16 }, { 512, 32 },
        { 1024, 16 }, { 4096, 8 },
    };
    static const int RS32[][2] = { { 64, 64 }, { 256, 16 } };
    const char *wisdir = argc > 1 ? argv[1] : ".";
    int fails = 0, skips = 0, ci;
    vfft_wisdom *W;
#ifdef _WIN32
    _putenv("VFFT_IL2D_REAL=1");
    _putenv("VFFT_IL2D_LOG=1");
#else
    putenv("VFFT_IL2D_REAL=1");
    putenv("VFFT_IL2D_LOG=1");
#endif
    setvbuf(stdout, NULL, _IONBF, 0);
    W = vfft_wisdom_load(wisdir);
    printf("=== il2d REAL M1 gate (native tier + rowsplit route, "
           "elementwise vs naive + pair contract, wisdom=%s %s) ===\n",
           wisdir, W ? "loaded" : "MISSING");
    srand(20260826);
    for (ci = 0; ci < (int)(sizeof CELLS / sizeof CELLS[0]); ci++)
        fails += run_cell(W, CELLS[ci][0], CELLS[ci][1], "dflt", &skips);
#ifdef _WIN32
    _putenv("VFFT_IL2D_ROWSPLIT=64");
#else
    putenv("VFFT_IL2D_ROWSPLIT=64");
#endif
    for (ci = 0; ci < (int)(sizeof RS64 / sizeof RS64[0]); ci++)
        fails += run_cell(W, RS64[ci][0], RS64[ci][1], "rs64", &skips);
#ifdef _WIN32
    _putenv("VFFT_IL2D_ROWSPLIT=32");
#else
    putenv("VFFT_IL2D_ROWSPLIT=32");
#endif
    for (ci = 0; ci < (int)(sizeof RS32 / sizeof RS32[0]); ci++)
        fails += run_cell(W, RS32[ci][0], RS32[ci][1], "rs32", &skips);
    /* ── row-route race/replay (the c2c M3 gate pattern): env silent,
     * FRESH cells (not in the battery, so the first create genuinely
     * RACES the route and banks chain+rw in memory); the second create
     * must SERVE the verdict — proven by bitwise-equal fwd outputs —
     * and the pair roundtrip closes bwd on the raced route. */
#ifdef _WIN32
    _putenv("VFFT_IL2D_ROWSPLIT=");
    _putenv("VFFT_IL2D_CHAIN=");
#else
    unsetenv("VFFT_IL2D_ROWSPLIT");
    unsetenv("VFFT_IL2D_CHAIN");
#endif
    {
        static const int RC[][2] = { { 4096, 16 }, { 512, 16 } };
        int rc_;
        for (rc_ = 0; rc_ < 2; rc_++) {
            const int N1 = RC[rc_][0], N2 = RC[rc_][1];
            const int hp1 = N2 / 2 + 1;
            const size_t RN = (size_t)N1 * N2, CN = (size_t)N1 * hp1;
            double *x = malloc(RN * 8), *za = malloc(2 * CN * 8);
            double *zb = malloc(2 * CN * 8), *xr = malloc(RN * 8);
            vfft_config_t cfg;
            vfft_plan ha, hb, hc;
            size_t i;
            double rt = 0;
            if (!x || !za || !zb || !xr) { printf("OOM\n"); return 2; }
            for (i = 0; i < RN; i++)
                x[i] = (double)rand() / RAND_MAX - 0.5;
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_R2C;
            cfg.placement = VFFT_OUTOFPLACE;
            cfg.rigor = VFFT_MEASURE;
            cfg.dims = 2;
            cfg.n[0] = N1;
            cfg.n[1] = N2;
            cfg.howmany = 1;
            cfg.order = VFFT_ORDER_DEFAULT;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED;
            cfg.nthreads = 1;
            cfg.wisdom = W;
            cfg.wisdom_write = 0;
            ha = vfft_create(&cfg); /* races + banks (memory) */
            hb = vfft_create(&cfg); /* must SERVE the banked verdict */
            cfg.transform = VFFT_C2R;
            hc = vfft_create(&cfg); /* reads the direction-shared cell */
            if (!ha || !hb || !hc) {
                printf("  race/replay %dx%d create FAIL\n", N1, N2);
                fails++;
            } else {
                vfft_execute(ha, VFFT_FORWARD, x, NULL, za, NULL);
                vfft_execute(hb, VFFT_FORWARD, x, NULL, zb, NULL);
                if (memcmp(za, zb, 2 * CN * 8) != 0) {
                    printf("  race/replay %dx%d: serve != race winner "
                           "*** FAIL ***\n", N1, N2);
                    fails++;
                } else {
                    vfft_execute(hc, VFFT_BACKWARD, za, NULL, xr, NULL);
                    for (i = 0; i < RN; i++) {
                        const double d =
                            fabs(xr[i] / ((double)N1 * N2) - x[i]);
                        if (d > rt) rt = d;
                    }
                    printf("  race/replay %dx%d: serve==race bitwise, "
                           "pair rt %.1e  %s\n", N1, N2, rt,
                           rt < 1e-10 ? "PASS" : "*** FAIL ***");
                    if (rt >= 1e-10) fails++;
                }
            }
            if (ha) vfft_destroy(ha);
            if (hb) vfft_destroy(hb);
            if (hc) vfft_destroy(hc);
            free(x); free(za); free(zb); free(xr);
        }
    }
    /* ── the banded column walk's F0 law: banding is pure column loop
     * interchange (rows OUTSIDE per §2.5) — banded output must be
     * BITWISE identical to unbanded, both directions. Chains greedy
     * (deterministic) in both arms; rowsplit pinned off to isolate wl. */
#ifdef _WIN32
    _putenv("VFFT_IL2D_ROWSPLIT=0");
#else
    putenv("VFFT_IL2D_ROWSPLIT=0");
#endif
    {
        static const int FC[][3] = { { 1024, 16, 16 }, { 512, 32, 64 } };
        int fc_;
        for (fc_ = 0; fc_ < 2; fc_++) {
            const int N1 = FC[fc_][0], N2 = FC[fc_][1], WL = FC[fc_][2];
            const int hp1 = N2 / 2 + 1;
            const size_t RN = (size_t)N1 * N2, CN = (size_t)N1 * hp1;
            double *x = malloc(RN * 8), *za = malloc(2 * CN * 8);
            double *zb = malloc(2 * CN * 8);
            double *xa = malloc(RN * 8), *xb = malloc(RN * 8);
            vfft_config_t cfg;
            vfft_plan hu, hb2, cu, cb;
            char wb[32];
            size_t i;
            if (!x || !za || !zb || !xa || !xb) { printf("OOM\n"); return 2; }
            for (i = 0; i < RN; i++)
                x[i] = (double)rand() / RAND_MAX - 0.5;
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_R2C;
            cfg.placement = VFFT_OUTOFPLACE;
            cfg.rigor = VFFT_MEASURE;
            cfg.dims = 2;
            cfg.n[0] = N1;
            cfg.n[1] = N2;
            cfg.howmany = 1;
            cfg.order = VFFT_ORDER_DEFAULT;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED;
            cfg.nthreads = 1;
            cfg.wisdom = W;
            cfg.wisdom_write = 0;
#ifdef _WIN32
            _putenv("VFFT_IL2D_WL=0");
#else
            putenv("VFFT_IL2D_WL=0");
#endif
            hu = vfft_create(&cfg);
            cfg.transform = VFFT_C2R;
            cu = vfft_create(&cfg);
            snprintf(wb, sizeof wb, "VFFT_IL2D_WL=%d", WL);
#ifdef _WIN32
            _putenv(wb);
#else
            putenv(strdup(wb));
#endif
            cfg.transform = VFFT_R2C;
            hb2 = vfft_create(&cfg);
            cfg.transform = VFFT_C2R;
            cb = vfft_create(&cfg);
#ifdef _WIN32
            _putenv("VFFT_IL2D_WL=");
#else
            unsetenv("VFFT_IL2D_WL");
#endif
            if (!hu || !hb2 || !cu || !cb) {
                printf("  F0 wl %dx%d create FAIL\n", N1, N2);
                fails++;
            } else {
                int ok = 1;
                vfft_execute(hu, VFFT_FORWARD, x, NULL, za, NULL);
                vfft_execute(hb2, VFFT_FORWARD, x, NULL, zb, NULL);
                if (memcmp(za, zb, 2 * CN * 8) != 0) {
                    printf("  F0 wl=%d %dx%d r2c: banded != unbanded "
                           "*** FAIL ***\n", WL, N1, N2);
                    fails++;
                    ok = 0;
                }
                vfft_execute(cu, VFFT_BACKWARD, za, NULL, xa, NULL);
                vfft_execute(cb, VFFT_BACKWARD, za, NULL, xb, NULL);
                if (memcmp(xa, xb, RN * 8) != 0) {
                    printf("  F0 wl=%d %dx%d c2r: banded != unbanded "
                           "*** FAIL ***\n", WL, N1, N2);
                    fails++;
                    ok = 0;
                }
                if (ok)
                    printf("  F0 wl=%d %4dx%-3d banded==unbanded BITWISE "
                           "both dirs  PASS\n", WL, N1, N2);
            }
            if (hu) vfft_destroy(hu);
            if (hb2) vfft_destroy(hb2);
            if (cu) vfft_destroy(cu);
            if (cb) vfft_destroy(cb);
            free(x); free(za); free(zb); free(xa); free(xb);
        }
    }
    /* ── MT INC-1: the ROW DOOR threads. Two independent engagement
     * gates (clones BUILT and work DISPATCHED — both have failed
     * silently in this library before, producing a perfect all-1.00x
     * table that never threaded), then MT == ST BITWISE. The ST arm is
     * a plan created with nthreads=1: same route, same chain, no clone
     * set — _tc_clone_equiv already guarantees clones are output-
     * equivalent, so any difference here is a threading bug. */
    /* Pin the PER-ROW door: INC-1 threads THAT route. (A cell whose race
     * picks ROWSPLIT runs the band loop instead and would show workers>0
     * with dispatches==0 — real, and exactly what this gate is for, but
     * not what INC-1 changes; rowsplit band MT is refused by design.)
     * N2 >= 32 also keeps the row child's zr2c inner on a native IL c2c
     * — at N2=16 the inner is the convert-served c2c(8), which
     * _tc_inner_mt_safe rightly refuses to clone. */
#ifdef _WIN32
    _putenv("VFFT_IL2D_ROWSPLIT=0");
    _putenv("VFFT_IL2D_CHAIN=");
#else
    putenv("VFFT_IL2D_ROWSPLIT=0");
    unsetenv("VFFT_IL2D_CHAIN");
#endif
    {
        static const int MC[][2] = { { 512, 32 }, { 128, 64 } };
        int mi;
        vfft_set_num_threads(8);
        for (mi = 0; mi < 2; mi++) {
            const int N1 = MC[mi][0], N2 = MC[mi][1];
            const int hp1 = N2 / 2 + 1;
            const size_t RN = (size_t)N1 * N2, CN = (size_t)N1 * hp1;
            double *x = malloc(RN * 8), *zs = malloc(2 * CN * 8);
            double *zm = malloc(2 * CN * 8);
            double *xs = malloc(RN * 8), *xm = malloc(RN * 8);
            vfft_config_t cfg;
            vfft_plan fs, fm, cs, cm;
            int workers;
            long d0, d1, cmt0, cmt1;
            size_t i;
            if (!x || !zs || !zm || !xs || !xm) { printf("OOM\n"); return 2; }
            for (i = 0; i < RN; i++)
                x[i] = (double)rand() / RAND_MAX - 0.5;
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_R2C;
            cfg.placement = VFFT_OUTOFPLACE;
            cfg.rigor = VFFT_MEASURE;
            cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2;
            cfg.howmany = 1;
            cfg.order = VFFT_ORDER_DEFAULT;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED;
            cfg.wisdom = W; cfg.wisdom_write = 0;
            cfg.nthreads = 1;
            fs = vfft_create(&cfg);
            cfg.transform = VFFT_C2R;
            cs = vfft_create(&cfg);
            cfg.nthreads = 8;
            cfg.transform = VFFT_R2C;
            fm = vfft_create(&cfg);
            cfg.transform = VFFT_C2R;
            cm = vfft_create(&cfg);
            if (!fs || !fm || !cs || !cm) {
                printf("  MT %4dx%-3d create FAIL\n", N1, N2);
                fails++;
                continue;
            }
            /* engagement gate 1: clones BUILT on the threaded plan */
            workers = vfft_plan_tc_workers(fm);
            if (workers <= 0) {
                printf("  MT %4dx%-3d NOT ENGAGED (row-door workers=%d) "
                       "*** FAIL ***\n", N1, N2, workers);
                fails++;
            }
            /* engagement gate 2: work actually DISPATCHED (rows), plus
             * the INC-3 column-pass engagement counter */
            d0 = vfft_tc_mt_dispatches();
            cmt0 = vfft_il2d_col_mt_passes();
            vfft_execute(fm, VFFT_FORWARD, x, NULL, zm, NULL);
            d1 = vfft_tc_mt_dispatches();
            cmt1 = vfft_il2d_col_mt_passes();
            vfft_execute(fs, VFFT_FORWARD, x, NULL, zs, NULL);
            if (d1 == d0) {
                printf("  MT %4dx%-3d r2c DISPATCHED NOTHING (under the "
                       "engage floor) *** FAIL ***\n", N1, N2);
                fails++;
            }
            if (memcmp(zs, zm, 2 * CN * 8) != 0) {
                printf("  MT %4dx%-3d r2c MT != ST *** FAIL ***\n",
                       N1, N2);
                fails++;
            } else {
                printf("  MT %4dx%-3d r2c workers=%d dispatches=%ld "
                       "colmt=%ld  MT==ST BITWISE  PASS\n",
                       N1, N2, workers, d1 - d0, cmt1 - cmt0);
            }
            d0 = vfft_tc_mt_dispatches();
            vfft_execute(cm, VFFT_BACKWARD, zm, NULL, xm, NULL);
            d1 = vfft_tc_mt_dispatches();
            vfft_execute(cs, VFFT_BACKWARD, zm, NULL, xs, NULL);
            if (d1 == d0) {
                printf("  MT %4dx%-3d c2r DISPATCHED NOTHING "
                       "*** FAIL ***\n", N1, N2);
                fails++;
            }
            if (memcmp(xs, xm, RN * 8) != 0) {
                printf("  MT %4dx%-3d c2r MT != ST *** FAIL ***\n",
                       N1, N2);
                fails++;
            } else {
                printf("  MT %4dx%-3d c2r workers=%d dispatches=%ld  "
                       "MT==ST BITWISE  PASS\n",
                       N1, N2, vfft_plan_tc_workers(cm), d1 - d0);
            }
            vfft_destroy(fs); vfft_destroy(fm);
            vfft_destroy(cs); vfft_destroy(cm);
            free(x); free(zs); free(zm); free(xs); free(xm);
        }
        vfft_set_num_threads(1);
    }
    /* ── INC-3b: the BANDED path with the columns FORCED threaded, so
     * the digit-split prefix AND the band arm are deterministically
     * exercised. Two pins are needed, and the reason matters:
     *   - VFFT_IL2D_NO_COLMT=0 forces the column MT on (the cells above
     *     honestly bank colmt=0, which would leave this untested);
     *   - VFFT_IL2D_WL forces a BAND, so cut > 0 and a prefix stage
     *     exists at all.
     * The cell must also have D = N1/R0 >= T at stage 0 or the digit
     * split declines and runs serial — 256x256 (chain 64.4 => D=4)
     * SILENTLY did exactly that, which is why 1024x16 (chain 64.16 =>
     * D=16, and L1=16 divides wl=64 so cut=1) is the cell here. */
    {
        static const int BC[][2] = { { 1024, 16 } };
        int bi;
        vfft_set_num_threads(8);
        for (bi = 0; bi < 1; bi++) {
            const int N1 = BC[bi][0], N2 = BC[bi][1];
            const int hp1 = N2 / 2 + 1;
            const size_t RN = (size_t)N1 * N2, CN = (size_t)N1 * hp1;
            double *x = malloc(RN * 8), *zs = malloc(2 * CN * 8);
            double *zm = malloc(2 * CN * 8), *xs = malloc(RN * 8);
            double *xm = malloc(RN * 8);
            vfft_config_t cfg;
            vfft_plan fs, fm, cs, cm;
            long c0, c1;
            size_t i;
            if (!x || !zs || !zm || !xs || !xm) { printf("OOM\n"); return 2; }
            for (i = 0; i < RN; i++)
                x[i] = (double)rand() / RAND_MAX - 0.5;
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_R2C;
            cfg.placement = VFFT_OUTOFPLACE;
            cfg.rigor = VFFT_MEASURE;
            cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2;
            cfg.howmany = 1;
            cfg.order = VFFT_ORDER_DEFAULT;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED;
            cfg.wisdom = W; cfg.wisdom_write = 0;
#ifdef _WIN32
            _putenv("VFFT_IL2D_WL=64");        /* force a band => cut>0 */
            _putenv("VFFT_IL2D_NO_COLMT=1");   /* columns SERIAL */
#else
            putenv("VFFT_IL2D_WL=64");
            putenv("VFFT_IL2D_NO_COLMT=1");
#endif
            cfg.nthreads = 8;
            fs = vfft_create(&cfg);
            cfg.transform = VFFT_C2R; cs = vfft_create(&cfg);
#ifdef _WIN32
            _putenv("VFFT_IL2D_NO_COLMT=0");   /* columns THREADED */
#else
            putenv("VFFT_IL2D_NO_COLMT=0");
#endif
            cfg.transform = VFFT_R2C; fm = vfft_create(&cfg);
            cfg.transform = VFFT_C2R; cm = vfft_create(&cfg);
#ifdef _WIN32
            _putenv("VFFT_IL2D_NO_COLMT=");
            _putenv("VFFT_IL2D_WL=");
#else
            unsetenv("VFFT_IL2D_NO_COLMT");
            unsetenv("VFFT_IL2D_WL");
#endif
            if (!fs || !fm || !cs || !cm) {
                printf("  COLMT %4dx%-4d create FAIL\n", N1, N2);
                fails++;
                continue;
            }
            c0 = vfft_il2d_col_mt_passes();
            vfft_execute(fm, VFFT_FORWARD, x, NULL, zm, NULL);
            c1 = vfft_il2d_col_mt_passes();
            vfft_execute(fs, VFFT_FORWARD, x, NULL, zs, NULL);
            if (c1 == c0) {
                printf("  COLMT %4dx%-4d columns NEVER THREADED "
                       "*** FAIL ***\n", N1, N2);
                fails++;
            }
            if (memcmp(zs, zm, 2 * CN * 8) != 0) {
                printf("  COLMT %4dx%-4d r2c threaded != serial "
                       "*** FAIL ***\n", N1, N2);
                fails++;
            } else {
                printf("  COLMT %4dx%-4d r2c colmt-passes=%ld  "
                       "threaded==serial BITWISE  PASS\n",
                       N1, N2, c1 - c0);
            }
            vfft_execute(cm, VFFT_BACKWARD, zm, NULL, xm, NULL);
            vfft_execute(cs, VFFT_BACKWARD, zm, NULL, xs, NULL);
            if (memcmp(xs, xm, RN * 8) != 0) {
                printf("  COLMT %4dx%-4d c2r threaded != serial "
                       "*** FAIL ***\n", N1, N2);
                fails++;
            } else {
                printf("  COLMT %4dx%-4d c2r threaded==serial BITWISE"
                       "  PASS\n", N1, N2);
            }
            vfft_destroy(fs); vfft_destroy(fm);
            vfft_destroy(cs); vfft_destroy(cm);
            free(x); free(zs); free(zm); free(xs); free(xm);
        }
        vfft_set_num_threads(1);
    }
    if (W) vfft_wisdom_free(W);
    printf("\n%s (%d fail, %d skip)\n",
           fails ? "*** FAIL ***" : "=== ALL PASS ===", fails, skips);
    return fails ? 1 : 0;
}
