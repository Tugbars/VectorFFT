/* il2d_m1_gate.c — M1 gate for the native IL 2D c2c tier
 * (docs/roadmap/fft2d_il_c2c_design.md; driver in vfft.c, kernels = n1c).
 *
 * FORWARD elementwise vs a naive separable DFT, PER DIRECTION (roundtrip
 * never gates a permuted transform — and here it also could not detect a
 * wrong-but-consistent permutation). The gate is SELF-PROVING on
 * engagement: the native tier serves natural x natural while the convert
 * wrapper serves scrambled, so an elementwise pass IS proof the native
 * tier ran (mt_results_need_engagement_proof, adapted). Cells whose row
 * child (1D K=1 IL in-place NATURAL at N2) cannot be created are SKIPPED
 * loudly — that is the designed fallback, not a failure.
 *
 * Build: python build.py --src benches/il2d_m1_gate.c --vfft --compile
 * Run  : il2d_m1_gate.exe <SCRATCH wisdir>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
#include "vfft_diagnostics.h"

static void naive_axis_i(const double *in, double *out, int N1, int N2,
                         double sgn)
{
    const double pi = 3.14159265358979323846;
    int k, i, j;
    for (k = 0; k < N1; k++)
        for (j = 0; j < N2; j++) {
            double sr = 0.0, si = 0.0;
            for (i = 0; i < N1; i++) {
                const double a = sgn * 2.0 * pi * (double)(k * i) / (double)N1;
                const double c = cos(a), s = sin(a);
                const size_t x = 2 * ((size_t)i * N2 + j);
                sr += in[x] * c - in[x + 1] * s;
                si += in[x] * s + in[x + 1] * c;
            }
            out[2 * ((size_t)k * N2 + j)] = sr;
            out[2 * ((size_t)k * N2 + j) + 1] = si;
        }
}

static void naive_axis_j(const double *in, double *out, int N1, int N2,
                         double sgn)
{
    const double pi = 3.14159265358979323846;
    int i, k, j;
    for (i = 0; i < N1; i++)
        for (k = 0; k < N2; k++) {
            double sr = 0.0, si = 0.0;
            for (j = 0; j < N2; j++) {
                const double a = sgn * 2.0 * pi * (double)(k * j) / (double)N2;
                const double c = cos(a), s = sin(a);
                const size_t x = 2 * ((size_t)i * N2 + j);
                sr += in[x] * c - in[x + 1] * s;
                si += in[x] * s + in[x + 1] * c;
            }
            out[2 * ((size_t)i * N2 + k)] = sr;
            out[2 * ((size_t)i * N2 + k) + 1] = si;
        }
}

/* the driver's greedy chain (MIRRORED — the binding is the v1 structural
 * default in _il2d_build_chain; M3 makes the chain a raced wisdom axis and
 * this gate will read it from the plan instead) + the DIF output map: the
 * FIRST stage's digit is the MOST significant position digit
 * (il2d_proto.h derivation, simulator-proven). nf==1 -> identity (M1). */
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

/* can the row child exist? probe the exact plan the driver builds */
static int row_child_ok(vfft_wisdom *W, int N2)
{
    vfft_config_t rc;
    vfft_plan p;
    memset(&rc, 0, sizeof rc);
    rc.transform = VFFT_C2C;
    rc.placement = VFFT_INPLACE;
    rc.rigor = VFFT_MEASURE;
    rc.dims = 1;
    rc.n[0] = N2;
    rc.howmany = 1;
    rc.order = VFFT_ORDER_NATURAL;
    rc.layout = VFFT_LAYOUT_INTERLEAVED;
    rc.nthreads = 1;
    rc.wisdom = W;
    p = vfft_create(&rc);
    if (!p) return 0;
    vfft_destroy(p);
    return 1;
}

int main(int argc, char **argv)
{
    static const int CELLS[][2] = {
        { 4, 16 },  { 8, 64 },   { 16, 16 },  { 16, 100 }, { 32, 64 },
        { 64, 64 }, { 64, 256 }, { 32, 5 },   { 64, 100 },
        /* M2 multi-stage chains (t2c mids + n1c leaf); output along i is
         * digit-reversed by the driver's chain — row_pos below mirrors it */
        { 128, 64 }, { 256, 64 }, { 512, 32 }, { 1024, 16 }, { 4096, 8 },
        { 128, 100 },
    };
    const char *wisdir = argc > 1 ? argv[1] : ".";
    int fails = 0, skips = 0, ci, dir, oop;
    vfft_wisdom *W;
#ifdef _WIN32
    _putenv("VFFT_IL2D_NATIVE=1");
#else
    putenv("VFFT_IL2D_NATIVE=1");
#endif
    setvbuf(stdout, NULL, _IONBF, 0);
    W = vfft_wisdom_load(wisdir);
    printf("=== il2d M1 gate (native tier, elementwise vs naive, wisdom=%s %s) ===\n",
           wisdir, W ? "loaded" : "MISSING");
    srand(20260825);
    for (ci = 0; ci < (int)(sizeof CELLS / sizeof CELLS[0]); ci++) {
        const int N1 = CELLS[ci][0], N2 = CELLS[ci][1];
        const size_t T = (size_t)N1 * N2;
        double *x = malloc(2 * T * 8), *z = malloc(2 * T * 8);
        double *oz = malloc(2 * T * 8);
        double *t1 = malloc(2 * T * 8), *ref = malloc(2 * T * 8);
        size_t i;
        if (!x || !z || !oz || !t1 || !ref) { printf("OOM\n"); return 2; }
        fprintf(stderr, "[m1] cell %dx%d: probing row child...\n", N1, N2);
        if (!row_child_ok(W, N2)) {
            printf("  %3dx%-4d SKIP (row child at N2=%d unavailable — tier falls back)\n",
                   N1, N2, N2);
            skips++;
            free(x); free(z); free(oz); free(t1); free(ref);
            continue;
        }
        for (i = 0; i < 2 * T; i++)
            x[i] = (double)rand() / RAND_MAX - 0.5;
        /* M3: the driver RACES the chain at create when env/wisdom give
         * none — the winner's digit-reversal need not match the mirrored
         * greedy map below. PIN the chain via env (env > race) so the
         * elementwise map is known; the race path itself is gated by the
         * race/replay section after the cell loop. */
        {
            int Rs[8], nf = chain_of(N1, Rs);
            char cb[80];
            int s_, o_;
            if (nf == 0) { Rs[0] = N1; nf = 1; }
            o_ = snprintf(cb, sizeof cb, "VFFT_IL2D_CHAIN=");
            for (s_ = 0; s_ < nf; s_++)
                o_ += snprintf(cb + o_, sizeof cb - o_, "%s%d",
                               s_ ? "." : "", Rs[s_]);
#ifdef _WIN32
            _putenv(cb);
#else
            putenv(strdup(cb));
#endif
        }
        /* dir 0: FORWARD elementwise vs naive through the chain map.
         * dir 1: the PAIR contract — bwd CONSUMES the plan's own fwd
         * output (the comb) and must return N*x; fwd is independently
         * proven by dir 0, so this isolates bwd (the matched-roundtrip
         * law: the tier's bwd is the Hermitian-transpose chain). */
        for (dir = 0; dir < 2; dir++) {
            if (dir == 0) {
                naive_axis_i(x, t1, N1, N2, -1.0);
                naive_axis_j(t1, ref, N1, N2, -1.0);
            }
            for (oop = 0; oop < 2; oop++) {
                vfft_config_t cfg;
                vfft_plan h;
                double maxe = 0.0, maxr = 0.0, rel;
                memset(&cfg, 0, sizeof cfg);
                cfg.transform = VFFT_C2C;
                cfg.placement = oop ? VFFT_OUTOFPLACE : VFFT_INPLACE;
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
                    printf("  %3dx%-4d %s %s create FAIL\n", N1, N2,
                           dir ? "bwd" : "fwd", oop ? "oop" : "ip ");
                    fails++;
                    continue;
                }
                memcpy(z, x, 2 * T * 8);
                if (dir == 0) {
                    if (oop) {
                        memset(oz, 0, 2 * T * 8);
                        vfft_execute(h, VFFT_FORWARD, z, NULL, oz, NULL);
                    } else {
                        vfft_execute(h, VFFT_FORWARD, z, NULL, z, NULL);
                    }
                    {
                        const double *got = oop ? oz : z;
                        int Rs[8], nf = chain_of(N1, Rs), ki, kj;
                        for (ki = 0; ki < N1; ki++) {
                            const int p_ = row_pos(ki, N1, Rs, nf);
                            for (kj = 0; kj < 2 * N2; kj++) {
                                double d =
                                    fabs(got[2 * (size_t)p_ * N2 + kj]
                                         - ref[2 * (size_t)ki * N2 + kj]);
                                double m =
                                    fabs(ref[2 * (size_t)ki * N2 + kj]);
                                if (d > maxe) maxe = d;
                                if (m > maxr) maxr = m;
                            }
                        }
                    }
                } else {
                    if (oop) {
                        memset(oz, 0, 2 * T * 8);
                        vfft_execute(h, VFFT_FORWARD, z, NULL, oz, NULL);
                        vfft_execute(h, VFFT_BACKWARD, oz, NULL, z, NULL);
                    } else {
                        vfft_execute(h, VFFT_FORWARD, z, NULL, z, NULL);
                        vfft_execute(h, VFFT_BACKWARD, z, NULL, z, NULL);
                    }
                    for (i = 0; i < 2 * T; i++) {
                        double d = fabs(z[i] / (double)T - x[i]);
                        double m = fabs(x[i]);
                        if (d > maxe) maxe = d;
                        if (m > maxr) maxr = m;
                    }
                }
                rel = maxe / (maxr > 0 ? maxr : 1.0);
                printf("  %3dx%-4d %s %s rel %.2e  %s\n", N1, N2,
                       dir ? "bwd" : "fwd", oop ? "oop" : "ip ", rel,
                       rel < 1e-10 ? "PASS" : "*** FAIL ***");
                if (rel >= 1e-10) fails++;
                vfft_destroy(h);
            }
        }
        free(x); free(z); free(oz); free(t1); free(ref);
    }
    /* ── M3 race/replay probe: with NO env chain, the first create RACES
     * the composition pool and banks the winner (memory; wisdom_write=0);
     * the second create must SERVE that verdict — proven by bitwise-equal
     * forward outputs (serve == race winner, deterministic replay), plus
     * a roundtrip sanity on the raced plan. */
#ifdef _WIN32
    _putenv("VFFT_IL2D_CHAIN=");
#else
    unsetenv("VFFT_IL2D_CHAIN");
#endif
    {
        static const int RC[][2] = { { 256, 64 }, { 1024, 16 } };
        int rc_;
        for (rc_ = 0; rc_ < 2; rc_++) {
            const int N1 = RC[rc_][0], N2 = RC[rc_][1];
            const size_t T = (size_t)N1 * N2;
            double *x = malloc(2 * T * 8), *za = malloc(2 * T * 8);
            double *zb = malloc(2 * T * 8);
            vfft_config_t cfg;
            vfft_plan ha, hb;
            size_t i;
            double rt = 0;
            for (i = 0; i < 2 * T; i++)
                x[i] = (double)rand() / RAND_MAX - 0.5;
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_C2C;
            cfg.placement = VFFT_INPLACE;
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
            if (!ha || !hb) {
                printf("  race/replay %dx%d create FAIL\n", N1, N2);
                fails++;
            } else {
                memcpy(za, x, 2 * T * 8);
                vfft_execute(ha, VFFT_FORWARD, za, NULL, za, NULL);
                memcpy(zb, x, 2 * T * 8);
                vfft_execute(hb, VFFT_FORWARD, zb, NULL, zb, NULL);
                if (memcmp(za, zb, 2 * T * 8) != 0) {
                    printf("  race/replay %dx%d: serve != race winner "
                           "*** FAIL ***\n", N1, N2);
                    fails++;
                } else {
                    vfft_execute(ha, VFFT_BACKWARD, za, NULL, za, NULL);
                    for (i = 0; i < 2 * T; i++) {
                        double d = fabs(za[i] / (double)T - x[i]);
                        if (d > rt) rt = d;
                    }
                    printf("  race/replay %dx%d: serve==race bitwise, rt "
                           "%.1e  %s\n", N1, N2, rt,
                           rt < 1e-10 ? "PASS" : "*** FAIL ***");
                    if (rt >= 1e-10) fails++;
                }
            }
            if (ha) vfft_destroy(ha);
            if (hb) vfft_destroy(hb);
            free(x); free(za); free(zb);
        }
    }
    /* ── INC-C: c2c MT. Same two-gate law as the real tier: (1) the
     * engagement counter must MOVE across the threaded execute (clones
     * built alone is not engagement), (2) MT == ST BITWISE, both dirs.
     * The ST arm is a plan created with nthreads=1 (same route, no
     * clones). Cells: one square big enough for a banded chain, one
     * short-N1 cell (single-stage => the strip+slab shape). Forced on
     * via VFFT_IL2D_NO_COLMT=0 so the gate is deterministic even where
     * the race would honestly bank "no". */
    {
        static const int MC[][2] = { { 512, 512 }, { 64, 1024 } };
        int mi;
        vfft_set_num_threads(8);
#ifdef _WIN32
        _putenv("VFFT_IL2D_NO_COLMT=0");
#else
        putenv("VFFT_IL2D_NO_COLMT=0");
#endif
        for (mi = 0; mi < 2; mi++) {
            const int N1 = MC[mi][0], N2 = MC[mi][1];
            const size_t PN = (size_t)N1 * N2;
            double *x = malloc(2 * PN * 8), *zs = malloc(2 * PN * 8);
            double *zm = malloc(2 * PN * 8);
            /* PIN the banded axis per cell: the axis race's wl winner is
             * run-dependent, and a raced wl == N1 (one whole-plane band)
             * has NO band axis, so the engagement assertion would flake
             * (hit 2026-08-27 at 64x1024). 512x512 pins its known
             * multi-band width (nb=4); 64x1024 pins UNBANDED so the
             * strip+slab shape is what this cell deterministically
             * gates. Env wl never banks (the tcut law). */
#ifdef _WIN32
            _putenv(mi == 0 ? "VFFT_IL2D_WL=128" : "VFFT_IL2D_WL=0");
#else
            putenv(mi == 0 ? "VFFT_IL2D_WL=128" : "VFFT_IL2D_WL=0");
#endif
            vfft_config_t cfg;
            vfft_plan ps, pm;
            long c0, c1;
            size_t i;
            int d2;
            if (!x || !zs || !zm) { printf("OOM\n"); return 2; }
            for (i = 0; i < 2 * PN; i++)
                x[i] = (double)rand() / RAND_MAX - 0.5;
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_C2C;
            cfg.placement = VFFT_INPLACE;
            cfg.rigor = VFFT_MEASURE;
            cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2;
            cfg.howmany = 1;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED;
            cfg.wisdom = W; cfg.wisdom_write = 0;
            cfg.nthreads = 1;
            ps = vfft_create(&cfg);
            cfg.nthreads = 8;
            pm = vfft_create(&cfg);
            if (!ps || !pm) {
                printf("  MT %4dx%-4d create FAIL\n", N1, N2);
                fails++;
                continue;
            }
            for (d2 = 0; d2 < 2; d2++) {
                const vfft_dir_t dd = d2 ? VFFT_BACKWARD : VFFT_FORWARD;
                memcpy(zs, x, 2 * PN * 8);
                memcpy(zm, x, 2 * PN * 8);
                c0 = vfft_il2d_col_mt_passes();
                vfft_execute(pm, dd, zm, NULL, zm, NULL);
                c1 = vfft_il2d_col_mt_passes();
                vfft_execute(ps, dd, zs, NULL, zs, NULL);
                if (c1 == c0) {
                    printf("  MT %4dx%-4d %s NEVER THREADED (colmt "
                           "passes unmoved) *** FAIL ***\n",
                           N1, N2, d2 ? "bwd" : "fwd");
                    fails++;
                }
                if (memcmp(zs, zm, 2 * PN * 8) != 0) {
                    printf("  MT %4dx%-4d %s MT != ST *** FAIL ***\n",
                           N1, N2, d2 ? "bwd" : "fwd");
                    fails++;
                } else if (c1 != c0) {
                    printf("  MT %4dx%-4d %s colmt-passes=%ld  MT==ST "
                           "BITWISE  PASS\n",
                           N1, N2, d2 ? "bwd" : "fwd", c1 - c0);
                }
            }
            vfft_destroy(ps); vfft_destroy(pm);
            free(x); free(zs); free(zm);
        }
#ifdef _WIN32
        _putenv("VFFT_IL2D_NO_COLMT=");
        _putenv("VFFT_IL2D_WL=");
#else
        unsetenv("VFFT_IL2D_NO_COLMT");
        unsetenv("VFFT_IL2D_WL");
#endif
        vfft_set_num_threads(1);
    }
    if (W) vfft_wisdom_free(W);
    printf("\n%s (%d fail, %d skip)\n",
           fails ? "*** FAIL ***" : "=== ALL PASS ===", fails, skips);
    return fails ? 1 : 0;
}
