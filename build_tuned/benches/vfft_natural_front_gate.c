/* vfft_natural_front_gate.c — B5: order=NATURAL through the PUBLIC front
 * door, K=1 interleaved IN-PLACE, cascade tier.
 *
 * The cold-start flow, per cell (SCRATCH wisdom dir — never generated/):
 *   1. a SCRAMBLED OOP create races + banks the kind-4 cascade line;
 *   2. the NATURAL in-place create must build the ZCASC candidate from that
 *      line (stfn terminator, no reorder pass), race it END-TO-END against
 *      the tape incumbent, win, bank mode=6, attach ([natorder] log line);
 *   3. correctness — 🔴 roundtrip cannot gate this; each direction gates
 *      against an independent reference:
 *        fwd:  execute(z,NULL,z,NULL) == naive DFT, elementwise IN ORDER;
 *        bwd:  execute(naive spectrum) == N*x, elementwise;
 *   4. CONSUME: a second create on the same wisdom must replay mode=ZCASC
 *      with NO race (no [natorder] race line), same correctness.
 *   5. small-N regression: N=256 natural in-place (tape tier, no cascade)
 *      still correct — the ZCASC arm must not perturb the classic path.
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
    snprintf(g_errpath, sizeof g_errpath, "%s/_nat_front_gate.log", dir);
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
    const int raced = strstr(log, "zcasc=") != NULL;      /* MEASURE race ran */
    const int zwon = strstr(log, "-> ZCASC") != NULL;     /* ...and ZCASC won */
    const int replayed = strstr(log, "replay ZCASC") != NULL; /* CONSUME hit  */

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
        ok = ok && raced && zwon;   /* MEASURE must race and ZCASC must win */
        eng = zwon ? "ZCASC(raced)" : (raced ? "tape(raced)" : "NO RACE");
    }
    else if (expect == 2)
    {
        /* CONSUME must NOT race AND must have actually attached ZCASC —
         * the first gate run mislabeled a silent tape replay as ZCASC. */
        ok = ok && !raced && replayed;
        eng = raced ? "RACED(!)" : (replayed ? "ZCASC(replay)" : "tape(!)");
    }
    printf("%-7d %-8s fwd=%.1e bwd=%.1e  %-13s%s\n",
           N, tag, ef, eb, eng, ok ? "" : "   *** FAIL ***");
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
        fz(x);
        fz(X);
    }

    vfft_wisdom_free(W);
    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
