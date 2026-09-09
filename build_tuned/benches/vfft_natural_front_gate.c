/* vfft_natural_front_gate.c — B5: order=NATURAL through the PUBLIC front
 * door, K=1 interleaved, in place and out of place, the pow2 cells 2048..32768.
 *
 * RESTAMPED 2026-09-09 to the owner's law (design_contracts.md section 4):
 * at a power of two up to ZTURN-T's ceiling NO door builds or races a
 * cascade candidate — "no cascade race arm please, eliminate". The cells
 * below are ZTURN-T's alone, so the flow per cell is now:
 *   1. a SCRAMBLED OOP create banks the kind-4 cascade line on the scratch
 *      (the cascade is still the scrambled writer there); the natural
 *      creates below must never read it as a candidate;
 *   2. MEASURE (recalibrate) in place: NO race fires ([ipil] never prints a
 *      "zcasc=" line); the K=1 engine serves ("attach ILP" or "replay ILP"
 *      in the tap) and banks mode=ilp;
 *   3. correctness — roundtrip cannot gate this; each direction gates
 *      against an independent reference:
 *        fwd:  execute(z,NULL,z,NULL) == naive DFT, elementwise IN ORDER;
 *        bwd:  execute(naive spectrum) == N*x, elementwise;
 *   4. CONSUME in place: no race, the banked ILP verdict replays (or the
 *      engine attaches when the cell had only a stale cascade row — the
 *      door treats such a row as unset in the band);
 *   5. small-N regression: N=256 natural in-place (tape tier) still correct.
 *
 * Out of place (the former Phase D arm, same law):
 *   6. MEASURE: NO [natorder] OOP race fires; the K=1 engine serves; src->dst
 *      distinct buffers, src must come back UNTOUCHED; fwd/bwd gate against
 *      the same references;
 *   7. CONSUME: no race; measure-vs-consume fwd outputs must be BITWISE
 *      identical;
 *   8. round-trip: free+reload the wisdom — both verdicts still consume
 *      silently;
 *   9. small-N OOP regression: N=256 natural OOP still correct.
 *
 * Before this restamp the gate asserted the OLD law: a MEASURE create had to
 * race the natord cascade, and from 8192 up the cascade had to win ("3-5x",
 * a 2026-08 measurement that ZTURN-T overturned on 2026-09-09).
 *
 * Run:   vfft_natural_front_gate.exe --wisdir <scratch dir>
 * Build: python build.py --src benches/vfft_natural_front_gate.c --vfft --compile
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── stderr tap (k1z_inplace_gate pattern) ─────────────────────────────── */
static char g_errpath[1024];
static long g_errpos = 0;
static int err_tap_open(const char *dir)
{
    snprintf(g_errpath, sizeof g_errpath, "build_tuned/benches/_nat_front_gate.log" /* cwd, 0.12: never inside a wisdom dir */ );
    if (!freopen(g_errpath, "w", stderr)) return 0;
    setvbuf(stderr, NULL, _IONBF, 0);
    g_errpos = 0;
    return 1;
}
static const char *err_tap_read(void)
{
    static char buf[8192];
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
static void env_set(const char *k, const char *v)
{
    static char slots[8][128];
    static int n = 0;
    char *s = slots[n++ & 7];
    snprintf(s, 128, "%s=%s", k, v ? v : "");
    putenv(s);
}

static double *az(size_t n)
{
#ifdef _WIN32
    return (double *)_aligned_malloc(2 * n * sizeof(double), 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, 2 * n * sizeof(double))) p = NULL;
    return (double *)p;
#endif
}
static void fz(double *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

static vfft_plan mk(vfft_wisdom *W, int N, int inplace, int natural)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = inplace ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = 1;
    cfg.order = natural ? VFFT_ORDER_NATURAL : VFFT_ORDER_SCRAMBLED;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = 1;
    cfg.wisdom = W;
    cfg.wisdom_write = 1;  /* measurement gate: banks must persist */
    return vfft_create(&cfg);
}

static void naive_dft(const double *x, double *X, long N)
{
    double *wr = (double *)malloc(sizeof(double) * (size_t)N);
    double *wi = (double *)malloc(sizeof(double) * (size_t)N);
    for (long j = 0; j < N; j++)
    {
        const double a = -2.0 * M_PI * (double)j / (double)N;
        wr[j] = cos(a);
        wi[j] = sin(a);
    }
    for (long k = 0; k < N; k++)
    {
        double sr = 0.0, si = 0.0;
        long idx = 0;
        for (long j = 0; j < N; j++)
        {
            const double xr = x[2 * j], xi = x[2 * j + 1];
            sr += xr * wr[idx] - xi * wi[idx];
            si += xr * wi[idx] + xi * wr[idx];
            idx += k;
            if (idx >= N) idx -= N;
        }
        X[2 * k] = sr;
        X[2 * k + 1] = si;
    }
    free(wr);
    free(wi);
}

/* max |a-b| / max|b| over 2N doubles */
static double relerr(const double *a, const double *b, long n2)
{
    double m = 0.0, e = 0.0;
    for (long i = 0; i < n2; i++)
    {
        if (fabs(b[i]) > m) m = fabs(b[i]);
        if (fabs(a[i] - b[i]) > e) e = fabs(a[i] - b[i]);
    }
    return e / m;
}

/* one cell: create natural in-place on W, gate both dirs vs naive refs.
 * expect_zcasc: 1 = the [natorder] tap must show a ZCASC pick (MEASURE) or
 * silence (CONSUME replay); 2 = CONSUME (silence REQUIRED); 0 = tape tier. */
static int run_cell(vfft_wisdom *W, int N, const double *x, const double *X,
                    int expect, const char *tag)
{
    vfft_plan hn = mk(W, N, /*inplace*/ 1, /*natural*/ 1);
    const char *log = err_tap_read();
    if (!hn)
    {
        printf("%-7d %-8s create FAILED\n", N, tag);
        return 0;
    }
    const int raced = strstr(log, "zcasc=") != NULL;      /* a race ran — never, in the band */
    const int served = strstr(log, "attach ILP") != NULL || /* the K=1 engine serves */
                       strstr(log, "replay ILP") != NULL;
    const int replayed = strstr(log, "replay ILP") != NULL; /* CONSUME hit the banked row */

    double *a = az((size_t)N);
    memcpy(a, x, 2 * (size_t)N * sizeof(double));
    vfft_execute(hn, VFFT_FORWARD, a, NULL, a, NULL);
    const double ef = relerr(a, X, 2L * N);

    memcpy(a, X, 2 * (size_t)N * sizeof(double));
    vfft_execute(hn, VFFT_BACKWARD, a, NULL, a, NULL);
    double *nx = az((size_t)N);
    for (long i = 0; i < 2L * N; i++) nx[i] = (double)N * x[i];
    const double eb = relerr(a, nx, 2L * N);
    fz(nx);
    fz(a);
    vfft_destroy(hn);

    int ok = ef < 1e-9 && eb < 1e-9;
    const char *eng = "tape";
    if (expect == 1)
    {
        /* MEASURE: no cascade arm exists in the band, so NO race may fire;
         * the K=1 engine must serve (owner's law, 2026-09-09) */
        ok = ok && !raced && served;
        eng = raced ? "RACED(!)" : (served ? "ILP(no race)" : "tape(!)");
    }
    else if (expect == 2)
    {
        /* CONSUME: no race; the banked ILP verdict replays, or the engine
         * attaches where the cell carried only a stale cascade row */
        ok = ok && !raced && served;
        eng = raced ? "RACED(!)" : (replayed ? "replay" : (served ? "attach" : "tape(!)"));
    }
    printf("%-7d %-8s fwd=%.1e bwd=%.1e  %-13s%s\n",
           N, tag, ef, eb, eng, ok ? "" : "   *** FAIL ***");
    return ok;
}

/* Phase D cell: NATURAL OOP through the front door. expect:
 *   0 = no-cascade tier (sub-2048), correctness only;
 *   1 = MEASURE, the [natorder] OOP race must RUN, either winner is legal
 *       (competitive cells: the per-cell winner is the race's business);
 *   3 = MEASURE, race must run AND ZCASC must win (≥4096: D1 measured the
 *       incumbent at 0.39x..0.17x of the cascade's class — an engine win
 *       there is a wiring bug, not a race outcome);
 *   2 = CONSUME, no race + "replay ZCASC-OOP" required;
 *   4 = CONSUME, no race required, banked FREE serving the engine is legal
 *       (the measure/consume bitwise memcmp in main covers identity).
 * fwd_out (optional): receives the fwd spectrum bytes for that check. */
static int run_cell_oop(vfft_wisdom *W, int N, const double *x,
                        const double *X, int expect, const char *tag,
                        double *fwd_out)
{
    vfft_plan hn = mk(W, N, /*inplace*/ 0, /*natural*/ 1);
    const char *log = err_tap_read();
    if (!hn)
    {
        printf("%-7d %-8s OOP create FAILED\n", N, tag);
        return 0;
    }
    const int raced = strstr(log, "OOP zcasc=") != NULL;
    const int zwon = strstr(log, "-> ZCASC-OOP") != NULL;
    const int replayed = strstr(log, "replay ZCASC-OOP") != NULL;

    double *s = az((size_t)N), *d = az((size_t)N);
    memcpy(s, x, 2 * (size_t)N * sizeof(double));
    vfft_execute(hn, VFFT_FORWARD, s, NULL, d, NULL);
    const double ef = relerr(d, X, 2L * N);
    const int src_ok =
        memcmp(s, x, 2 * (size_t)N * sizeof(double)) == 0;
    if (fwd_out)
        memcpy(fwd_out, d, 2 * (size_t)N * sizeof(double));

    memcpy(s, X, 2 * (size_t)N * sizeof(double));
    vfft_execute(hn, VFFT_BACKWARD, s, NULL, d, NULL);
    double *nx = az((size_t)N);
    for (long i = 0; i < 2L * N; i++) nx[i] = (double)N * x[i];
    const double eb = relerr(d, nx, 2L * N);
    fz(nx);
    fz(d);
    fz(s);
    vfft_destroy(hn);

    int ok = ef < 1e-9 && eb < 1e-9 && src_ok;
    const char *eng = "engine";
    /* every expect value, MEASURE or CONSUME: in ZTURN-T's band no cascade
     * arm exists, so no [natorder] OOP race may fire and the K=1 engine
     * serves (owner's law, 2026-09-09). A ZCASC replay would mean a stale
     * cascade verdict got read as one — a failure too. */
    (void)expect;
    ok = ok && !raced && !replayed;
    eng = raced ? "RACED(!)" : (replayed ? "ZCASC(!)" : (zwon ? "ZCASC(!)" : "engine(no race)"));
    printf("%-7d %-8s fwd=%.1e bwd=%.1e  %-13s%s%s\n",
           N, tag, ef, eb, eng, src_ok ? "" : " SRC-CLOBBERED",
           ok ? "" : "   *** FAIL ***");
    return ok;
}

int main(int argc, char **argv)
{
    const char *wisdir = NULL;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--wisdir") && i + 1 < argc) wisdir = argv[++i];
    if (!wisdir) { printf("usage: %s --wisdir <SCRATCH dir>\n", argv[0]); return 2; }
    if (!err_tap_open(wisdir)) { printf("stderr tap failed\n"); return 2; }

    env_set("VFFT_NAT_LOG", "1");
    env_set("VFFT_NO_NAT_ZCASC", "");
    env_set("VFFT_FORCE_ZROUTE", "");

    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { printf("vfft_wisdom_load FAILED\n"); return 2; }

    printf("\n=== B5: order=NATURAL front door, K=1 IL in-place ===\n");
    printf("%-7s %-8s %-24s %-13s\n", "N", "pass", "correctness", "engine");
    int fails = 0;

    static const int NS[] = { 2048, 4096, 8192, 16384, 32768 };
    for (size_t i = 0; i < sizeof NS / sizeof NS[0]; i++)
    {
        const int N = NS[i];
        /* 1. bank the kind-4 cascade line (scrambled OOP race, this scratch) */
        vfft_plan hs = mk(W, N, /*inplace*/ 0, /*natural*/ 0);
        if (!hs) { printf("%-7d scrambled create FAILED\n", N); fails++; continue; }
        vfft_destroy(hs);
        (void)err_tap_read(); /* drop the scrambled create's log lines */

        srand(515 + N);
        double *x = az((size_t)N), *X = az((size_t)N);
        for (long j = 0; j < 2L * N; j++)
            x[j] = (double)rand() / RAND_MAX - 0.5;
        naive_dft(x, X, N);

        if (!run_cell(W, N, x, X, /*MEASURE*/ 1, "measure")) fails++;
        if (!run_cell(W, N, x, X, /*CONSUME*/ 2, "consume")) fails++;

        /* Phase D: same references, NATURAL OOP. Measure races + banks
         * @natoop; consume replays; the two fwd spectra must be BITWISE
         * identical (the banked verdict is the coherence memo). */
        double *f1 = az((size_t)N), *f2 = az((size_t)N);
        /* 2026-09-09: every cell here is in ZTURN-T's band — no race, the
         * engine serves, measure and consume alike (em/ec kept for the
         * call shape only). */
        const int em = 1, ec = 4;
        if (!run_cell_oop(W, N, x, X, em, "oop-meas", f1)) fails++;
        if (!run_cell_oop(W, N, x, X, ec, "oop-cons", f2)) fails++;
        if (memcmp(f1, f2, 2 * (size_t)N * sizeof(double)) != 0)
        {
            printf("%-7d oop measure/consume fwd DIFF (coherence)   *** FAIL ***\n", N);
            fails++;
        }
        fz(f1);
        fz(f2);

        fz(x);
        fz(X);
    }

    /* small-N tape-tier regression: the ZCASC arm must not perturb it */
    {
        const int N = 256;
        srand(515 + N);
        double *x = az((size_t)N), *X = az((size_t)N);
        for (long j = 0; j < 2L * N; j++)
            x[j] = (double)rand() / RAND_MAX - 0.5;
        naive_dft(x, X, N);
        if (!run_cell(W, N, x, X, /*tape tier*/ 0, "smallN")) fails++;
        /* small-N OOP: native il2p tier, NO cascade race may fire */
        if (!run_cell_oop(W, N, x, X, /*no cascade*/ 0, "smallN-o", NULL))
            fails++;
        fz(x);
        fz(X);
    }

    /* round-trip: free + reload the wisdom — the @nat AND @natoop verdicts
     * must survive the save/load cycle (the saver now emits @natoop lines;
     * a parse regression would surface here as a RACED(!) consume). */
    {
        vfft_wisdom_free(W);
        W = vfft_wisdom_load(wisdir);
        if (!W) { printf("wisdom RELOAD FAILED\n"); return 2; }
        const int N = 4096;
        srand(515 + N);
        double *x = az((size_t)N), *X = az((size_t)N);
        for (long j = 0; j < 2L * N; j++)
            x[j] = (double)rand() / RAND_MAX - 0.5;
        naive_dft(x, X, N);
        printf("--- reload round-trip ---\n");
        if (!run_cell(W, N, x, X, /*CONSUME*/ 2, "rt-ip")) fails++;
        /* 4096 oop is a COMPETITIVE cell (see em/ec above): the banked
         * verdict may be mode=free, which reloads as engine(free) — that IS
         * the verdict surviving the save/load cycle. expect 4 still asserts
         * NO race fires on reload, which is the parse-regression coverage
         * this section exists for. */
        if (!run_cell_oop(W, N, x, X, /*CONSUME, free ok*/ 4, "rt-oop", NULL)) fails++;
        fz(x);
        fz(X);
    }

    vfft_wisdom_free(W);
    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
