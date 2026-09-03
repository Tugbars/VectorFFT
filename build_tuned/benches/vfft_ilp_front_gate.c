/* vfft_ilp_front_gate.c — Phase B gate: sub-2048 K=1 interleaved IN-PLACE
 * through the front door (il_coverage_plan.md B4).
 *
 * Per cell {128, 256, 512, 1024}, fresh scratch wisdom:
 *   1. NATURAL measure: create must RACE (an "[natorder] ... ilp=" line);
 *      the VERDICT (ILP vs tape) belongs to the race, not this gate — both
 *      outcomes pass; correctness always gates: fwd == naive DFT IN ORDER,
 *      bwd(naive spectrum) == N·x, aliased (z,NULL,z,NULL). 🔴 roundtrip
 *      cannot gate ordering.
 *   2. NATURAL consume: NO race; if measure banked ILP, the replay line
 *      must appear (mode coherence).
 *   3. SCRAMBLED in-place (B3, hit-only): if the banked verdict is ILP,
 *      the scrambled handle's fwd output must be memcmp-EXACT == the
 *      natural handle's (identity permutation, same engine) and its
 *      matched roundtrip must hold. If the verdict is tape, the scrambled
 *      handle serves the classic convert path — matched roundtrip only.
 *   4. Boundary: 2048 NATURAL must still go ZCASC (no ILP shadowing).
 *
 * Run:   vfft_ilp_front_gate.exe --wisdir <scratch dir>
 * Build: python build.py --src benches/vfft_ilp_front_gate.c --vfft --compile
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static char g_errpath[1024];
static long g_errpos = 0;
static int err_tap_open(const char *dir)
{
    snprintf(g_errpath, sizeof g_errpath, "build_tuned/benches/_ilp_gate.log" /* cwd, 0.12: never inside a wisdom dir */ );
    if (!freopen(g_errpath, "w", stderr)) return 0;
    setvbuf(stderr, NULL, _IONBF, 0);
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

static vfft_plan mk(vfft_wisdom *W, int N, int order /*0=nat 1=scr*/)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_INPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = 1;
    cfg.order = order ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = 1;
    cfg.wisdom = W;
    cfg.wisdom_write = 1;  /* measurement gate: banks must persist */
    return vfft_create(&cfg);
}

static void naive_dft(const double *x, double *X, long N)
{
    for (long k = 0; k < N; k++)
    {
        double sr = 0, si = 0;
        for (long j = 0; j < N; j++)
        {
            const double a = -2.0 * M_PI * (double)((j * k) % N) / (double)N;
            const double c = cos(a), s = sin(a);
            sr += x[2 * j] * c - x[2 * j + 1] * s;
            si += x[2 * j] * s + x[2 * j + 1] * c;
        }
        X[2 * k] = sr;
        X[2 * k + 1] = si;
    }
}
static double relerr(const double *a, const double *b, long n2)
{
    double m = 0, e = 0;
    for (long i = 0; i < n2; i++)
    {
        if (fabs(b[i]) > m) m = fabs(b[i]);
        if (fabs(a[i] - b[i]) > e) e = fabs(a[i] - b[i]);
    }
    return e / m;
}

int main(int argc, char **argv)
{
    const char *wisdir = NULL;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--wisdir") && i + 1 < argc) wisdir = argv[++i];
    if (!wisdir) { printf("usage: %s --wisdir <SCRATCH dir>\n", argv[0]); return 2; }
    if (!err_tap_open(wisdir)) { printf("stderr tap failed\n"); return 2; }
    env_set("VFFT_NAT_LOG", "1");
    env_set("VFFT_NO_NAT_ILP", "");
    env_set("VFFT_NO_NAT_ZCASC", "");
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { printf("vfft_wisdom_load FAILED\n"); return 2; }

    printf("\n=== Phase B: sub-2048 in-place IL tier, front door ===\n");
    printf("%-7s %-8s | %-10s %-10s | %-12s | %s\n",
           "N", "pass", "fwd", "bwd", "engine", "scrambled arm");
    int fails = 0;

    static const int NS[] = { 128, 256, 512, 1024 };
    for (size_t i = 0; i < sizeof NS / sizeof NS[0]; i++)
    {
        const int N = NS[i];
        srand(909 + N);
        double *x = az((size_t)N), *X = az((size_t)N), *a = az((size_t)N);
        double *yn = az((size_t)N);
        for (long j = 0; j < 2L * N; j++)
            x[j] = (double)rand() / RAND_MAX - 0.5;
        naive_dft(x, X, N);

        /* 1. NATURAL measure */
        vfft_plan hn = mk(W, N, 0);
        const char *log = err_tap_read();
        const int raced = strstr(log, "ilp=") != NULL;
        const int ilp_won = strstr(log, "-> ILP") != NULL;
        if (!hn) { printf("%-7d create FAILED\n", N); fails++; continue; }
        memcpy(a, x, 2 * (size_t)N * sizeof(double));
        vfft_execute(hn, VFFT_FORWARD, a, NULL, a, NULL);
        memcpy(yn, a, 2 * (size_t)N * sizeof(double));
        const double ef = relerr(a, X, 2L * N);
        memcpy(a, X, 2 * (size_t)N * sizeof(double));
        vfft_execute(hn, VFFT_BACKWARD, a, NULL, a, NULL);
        double *nx = az((size_t)N);
        for (long j = 0; j < 2L * N; j++) nx[j] = (double)N * x[j];
        const double eb = relerr(a, nx, 2L * N);
        fz(nx);
        vfft_destroy(hn);
        int ok = raced && ef < 1e-9 && eb < 1e-9;

        /* 2. NATURAL consume — no race; replay line iff ILP banked */
        vfft_plan hc = mk(W, N, 0);
        const char *log2 = err_tap_read();
        const int reraced = strstr(log2, "ilp=") != NULL;
        const int replayed = strstr(log2, "replay ILP") != NULL;
        if (hc)
        {
            memcpy(a, x, 2 * (size_t)N * sizeof(double));
            vfft_execute(hc, VFFT_FORWARD, a, NULL, a, NULL);
            if (memcmp(a, yn, 2 * (size_t)N * sizeof(double)) != 0)
                ok = 0; /* consume must reproduce measure bitwise */
            vfft_destroy(hc);
        }
        else
            ok = 0;
        if (reraced || (ilp_won && !replayed))
            ok = 0;

        /* 3. SCRAMBLED in-place, hit-only */
        const char *scrarm = "rt-only(tape)";
        {
            vfft_plan hs = mk(W, N, 1);
            (void)err_tap_read();
            if (!hs)
                ok = 0;
            else
            {
                memcpy(a, x, 2 * (size_t)N * sizeof(double));
                vfft_execute(hs, VFFT_FORWARD, a, NULL, a, NULL);
                if (ilp_won)
                {
                    scrarm = memcmp(a, yn,
                                    2 * (size_t)N * sizeof(double)) == 0
                                 ? "IDENT(ilp)"
                                 : "DIFF(!)";
                    if (strcmp(scrarm, "IDENT(ilp)") != 0)
                        ok = 0;
                }
                vfft_execute(hs, VFFT_BACKWARD, a, NULL, a, NULL);
                double *nx2 = az((size_t)N);
                for (long j = 0; j < 2L * N; j++)
                    nx2[j] = (double)N * x[j];
                if (relerr(a, nx2, 2L * N) > 1e-9)
                    ok = 0;
                fz(nx2);
                vfft_destroy(hs);
            }
        }

        if (!ok) fails++;
        printf("%-7d %-8s | %.2e   %.2e | %-12s | %s%s\n",
               N, "m+c+s", ef, eb,
               ilp_won ? "ILP(raced)" : (raced ? "tape(raced)" : "NO RACE"),
               scrarm, ok ? "" : "   *** FAIL ***");
        fz(x); fz(X); fz(a); fz(yn);
    }

    /* 4. boundary: 2048 NATURAL must still go ZCASC */
    {
        const int N = 2048;
        vfft_plan hb0 = mk(W, N, 1); /* bank kind-4 first (scrambled OOP?) —
                                      * in-place scrambled replays kind-4;
                                      * a miss here is fine, the natural
                                      * create races ZCASC only on a kind-4
                                      * hit. Use an OOP create to bank. */
        if (hb0) vfft_destroy(hb0);
        vfft_config_t cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C;
        cfg.placement = VFFT_OUTOFPLACE;
        cfg.rigor = VFFT_MEASURE;
        cfg.dims = 1;
        cfg.n[0] = N;
        cfg.howmany = 1;
        cfg.order = VFFT_ORDER_SCRAMBLED;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.nthreads = 1;
        cfg.wisdom = W;
        cfg.wisdom_write = 1;  /* measurement gate: banks must persist */
        vfft_plan ho = vfft_create(&cfg);
        if (ho) vfft_destroy(ho);
        (void)err_tap_read();
        vfft_plan hn = mk(W, N, 0);
        const char *log = err_tap_read();
        const int zc = strstr(log, "zcasc=") != NULL ||
                       strstr(log, "replay ZCASC") != NULL;
        const int ilp = strstr(log, "ilp=") != NULL ||
                        strstr(log, "replay ILP") != NULL;
        const int ok = hn && zc;   /* 2026-09-03: an ilp ARM is present at 2048 (the
                                    * IL-vs-IL race); ZCASC must still appear */
        if (!ok) fails++;
        printf("%-7d %-8s | boundary: %s%s\n", N, "zcasc",
               zc ? (ilp ? "ZCASC+ILP(!)" : "ZCASC") : "NO ZCASC(!)",
               ok ? "" : "   *** FAIL ***");
        if (hn) vfft_destroy(hn);
    }

    vfft_wisdom_free(W);
    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
