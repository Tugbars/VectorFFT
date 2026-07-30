/* zturn_tcut_ab.c — INTERNAL A/B for the tiled mid stages (`tcut`) of the K=1
 * ZTURN-S z-cascade.  docs/research/tcut_spec.md §F1/§8.
 *
 * 🔴 THIS IS NOT A vs-MKL MEASUREMENT.  Rule 2 stands: any number quoted
 * against MKL comes from build_tuned/benches/bench_1d_vs_mkl.c (a new --mode),
 * never from a new harness.  This file races OUR OWN ARMS against each other,
 * which rule 3 explicitly separates from a vs-MKL number.
 *
 * PACED PROTOCOL, as the project requires (the tree has been burned repeatedly):
 *   · MEDIANS over rounds, never best-of.
 *   · Arm ORDER ROTATES every round, so no arm is permanently second.
 *   · ONE shared input buffer and ONE shared output buffer for EVERY arm — not
 *     merely a 64 B skew.  The arms are bit-identical by construction (the
 *     correctness gate proves it), so they can share the exact same addresses;
 *     that removes the 4 KB-aliasing/bimodality confound completely instead of
 *     merely mitigating it.
 *   · cachebust + pace between arms; one cell per process; pinned to one core.
 *   · An A/A control arm ("off2" — a SECOND, independently created untiled
 *     plan) is measured alongside the real arms.  It is the noise floor: each
 *     plan owns its own `plane` allocation, so plan-address luck is a real
 *     per-arm confound and off2-vs-off is the only honest measure of it.  If
 *     off2 does not read ~0%, nothing smaller than |off2-off| is readable.
 *   · Per-pair SIGN CONSISTENCY is reported next to the median ratio.  A ratio
 *     is invariant to nothing; the sign of a within-round difference is
 *     invariant to any monotone thermal drift, so a real 3% effect shows ~100%
 *     sign consistency while a 0% effect shows ~50%.
 *
 * ARM DECOMPOSITION (why "off vs tcut=0" alone is not a tiling number):
 *   A0 "off"  untiled, full-size twiddle tables streamed (65 280 B at 4096).
 *   A1 "a1"   stage-major, section-split, RESET twiddles.  Reproduces A0's
 *             plane address order EXACTLY (section q is precisely groups
 *             [q*G/4,(q+1)*G/4) of the stage) and differs from it only in the
 *             twiddle footprint (65 280 -> 16 320 B) and the call count.
 *   A2 "0"    tiled, tcut = 0: same call count and same twiddle window as A1,
 *             different plane order.
 *   ⇒ A1-A0 is the TABLE effect; A2-A1 is the PURE TILING effect.
 *     A2-A0 is their sum and must never be quoted as "tiling".
 *
 * Modes: default = warm/L2-resident.  --cold cachebusts before EVERY timed
 * transform (spec §8 R2: compulsory DRAM traffic is unchanged by tiling, so
 * this arm must NOT move; if it does, something other than tiling changed).
 *
 * Build: cd build_tuned && python build.py --src benches/zturn_tcut_ab.c --vfft --compile
 * Run  : zturn_tcut_ab.exe --wisdir <scratch wisdir> --cell 4096 [--rounds 25] [--cold]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "vfft.h"
#ifdef _WIN32
#include <windows.h>
#endif

/* ───────────────────────────── timing ─────────────────────────────────── */
static double now_ns(void)
{
#ifdef _WIN32
    static double f = 0.0;
    LARGE_INTEGER t;
    if (f == 0.0) { LARGE_INTEGER q; QueryPerformanceFrequency(&q);
                    f = 1e9 / (double)q.QuadPart; }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * f;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
#endif
}
static void pace(int ms)
{
#ifdef _WIN32
    if (ms > 0) Sleep((DWORD)ms);
#else
    if (ms > 0) { struct timespec ts = {ms/1000, (long)(ms%1000)*1000000L};
                  nanosleep(&ts, NULL); }
#endif
}
static void cachebust(void)
{
    size_t s = 32 * 1024 * 1024 / sizeof(double);
    double *j = (double *)malloc(s * sizeof(double));
    volatile double a = 0;
    if (!j) return;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a;
    free(j);
}
static double *alloc_z(size_t n)
{
#ifdef _WIN32
    return (double *)_aligned_malloc(2 * n * sizeof(double), 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, 2 * n * sizeof(double))) p = NULL;
    return (double *)p;
#endif
}
static int dcmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}
static double med(double *v, int n)
{
    qsort(v, (size_t)n, sizeof(double), dcmp);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* ─────────────────────────── arm construction ─────────────────────────── */
static void env_set(const char *k, const char *v)
{
    static char slots[8][128];
    static int n = 0;
    char *s = slots[n++ & 7];
    snprintf(s, 128, "%s=%s", k, v ? v : "");
    putenv(s);
}

typedef struct { const char *label, *tcut, *tw; int bundle; vfft_plan h;
                 int engaged; } arm_t;

int main(int argc, char **argv)
{
    const char *wisdir = NULL, *wisdir2 = NULL;
    int N = 4096, rounds = 25, cold = 0, core = 2, cool_ms = 3;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--wisdir") && i + 1 < argc) wisdir = argv[++i];
        else if (!strcmp(argv[i], "--wisdir2") && i + 1 < argc) wisdir2 = argv[++i];
        else if (!strcmp(argv[i], "--cell") && i + 1 < argc) N = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--core") && i + 1 < argc) core = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cool") && i + 1 < argc) cool_ms = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--cold")) cold = 1;
    }
    if (!wisdir) { printf("usage: %s --wisdir <dir> --cell N [--rounds R] "
                          "[--cold] [--core C]\n", argv[0]); return 2; }
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << core);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { printf("vfft_wisdom_load(%s) FAILED\n", wisdir); return 2; }
    vfft_wisdom *W2 = wisdir2 ? vfft_wisdom_load(wisdir2) : NULL;
    if (wisdir2 && !W2)
        printf("note: --wisdir2 %s failed to load — chain-2 arms skipped\n",
               wisdir2);

    /* 🔴 THREE identical untiled arms, not one. Each plan owns its own `plane`
     * allocation, so plan-address luck is a real per-arm confound; the SPREAD
     * across A0/A0'/A0'' is the honest noise floor and it is measured in the
     * same run as the effect. One A/A arm can only ever give a point estimate
     * of that floor. */
    arm_t arms[] = {
        {"A0 off      ", "off", "", 0},
        {"A0' off (A/A)", "off", "", 0},
        {"A0'' off(A/A)", "off", "", 0},
        {"A1 sect-split", "a1",  "", 0},     /* table effect, A0 plane order   */
        {"A2 tcut=0   ", "0",   "", 0},
        {"A2 tcut=1   ", "1",   "", 0},
        {"A2 tcut=2   ", "2",   "", 0},
        {"A2 tcut=0:f ", "0:1", "", 0},
        {"A1 honest tw ", "a1",  "honest", 0}, /* F1's twiddle discriminator   */
        /* CREATION-ORDER controls, created LAST. Every plan owns its own
         * `plane`, so "allocated later" is a candidate explanation for any
         * delta. If it were the driver, this late untiled arm would look as
         * fast as the tiled ones. It must read ~A0. */
        {"A0''' off late", "off", "", 0},
        {"A2 tcut=0 late", "0",   "", 0},
        /* SECOND CHAIN (--wisdir2), same run, same buffers. §8 R4: tcut changes
         * the chain objective function, so the chain that wins untiled is not
         * necessarily the chain that wins tiled — and comparing chains ACROSS
         * processes is exactly the cross-session comparison this machine
         * forbids. These arms make the joint (chain x tcut) comparison a
         * within-run one. Skipped when --wisdir2 is absent. */
        {"B0 chain2 off", "off", "", 1},
        {"B2 chain2 t=0", "0",   "", 1},
        {"B2 chain2 t=1", "1",   "", 1},
        {"B2 chain2 t=2", "2",   "", 1},
    };
    const int na = (int)(sizeof arms / sizeof arms[0]);

    /* create every arm up front: VFFT_TCUT is read once, inside create */
    for (int a = 0; a < na; a++) {
        if (arms[a].bundle && !W2) { arms[a].engaged = 0; arms[a].h = NULL; continue; }
        env_set("VFFT_TCUT", arms[a].tcut);
        env_set("VFFT_TCUT_TW", arms[a].tw);
        env_set("VFFT_TCUT_VERBOSE", "");
        env_set("VFFT_FORCE_ZROUTE", "zturn");
        vfft_config_t cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE;
        cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
        cfg.order = VFFT_ORDER_SCRAMBLED; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.nthreads = 1; cfg.wisdom = arms[a].bundle ? W2 : W;
        arms[a].h = vfft_create(&cfg);
        arms[a].engaged = arms[a].h != NULL;
    }

    /* ONE shared arena: identical input and output addresses for every arm. */
    double *x = alloc_z((size_t)N), *y = alloc_z((size_t)N);
    if (!x || !y) { printf("alloc failed\n"); return 2; }
    srand(7 + N);
    for (int i = 0; i < 2 * N; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
    memset(y, 0, 2 * (size_t)N * sizeof(double));

    /* rep count: aim ~400 us per timed block (warm) */
    int reps = 1;
    if (!cold) {
        for (int w = 0; w < 20; w++) vfft_execute(arms[0].h, VFFT_FORWARD, x, NULL, y, NULL);
        double t0 = now_ns();
        for (int i = 0; i < 8; i++) vfft_execute(arms[0].h, VFFT_FORWARD, x, NULL, y, NULL);
        double per = (now_ns() - t0) / 8.0;
        reps = (int)(400000.0 / (per > 1 ? per : 1));
        if (reps < 8) reps = 8;
        if (reps > 20000) reps = 20000;
    }

    double *smp = (double *)calloc((size_t)na * (size_t)rounds, sizeof(double));
    printf("\n=== tcut INTERNAL A/B  N=%d  %s  rounds=%d reps/round=%d "
           "core=%d ===\n", N, cold ? "COLD (cachebust per transform)" : "warm",
           rounds, reps, core);

    for (int r = 0; r < rounds; r++) {
        for (int k = 0; k < na; k++) {
            const int a = (k + r) % na;            /* ROTATE the arm order */
            if (!arms[a].engaged) continue;
            cachebust();
            pace(cool_ms);
            double ns;
            if (cold) {
                double t0 = now_ns();
                vfft_execute(arms[a].h, VFFT_FORWARD, x, NULL, y, NULL);
                ns = now_ns() - t0;
            } else {
                for (int w = 0; w < 8; w++)
                    vfft_execute(arms[a].h, VFFT_FORWARD, x, NULL, y, NULL);
                double t0 = now_ns();
                for (int i = 0; i < reps; i++)
                    vfft_execute(arms[a].h, VFFT_FORWARD, x, NULL, y, NULL);
                ns = (now_ns() - t0) / reps;
            }
            smp[(size_t)a * rounds + r] = ns;
        }
    }

    /* medians + within-round sign consistency against A0 and against A1 */
    double *tmp = (double *)malloc(sizeof(double) * (size_t)rounds);
    double m[32];
    for (int a = 0; a < na; a++) {
        if (!arms[a].engaged) { m[a] = 0; continue; }
        memcpy(tmp, smp + (size_t)a * rounds, sizeof(double) * (size_t)rounds);
        m[a] = med(tmp, rounds);
    }
    const int IA1 = 3;                          /* index of the A1 arm */
    printf("\n  %-14s %10s %7s %9s %8s %9s %8s\n", "arm", "median ns",
           "p10-p90", "vs A0", "sign/A0", "vs A1", "sign/A1");
    for (int a = 0; a < na; a++) {
        if (!arms[a].engaged) { printf("  %-14s   (create failed)\n", arms[a].label); continue; }
        int f0 = 0, n0 = 0, f1 = 0, n1 = 0;
        for (int r = 0; r < rounds; r++) {
            double v = smp[(size_t)a * rounds + r];
            double b0 = smp[(size_t)0 * rounds + r];
            double b1 = smp[(size_t)IA1 * rounds + r];
            if (v > 0 && b0 > 0) { n0++; if (v < b0) f0++; }
            if (v > 0 && b1 > 0) { n1++; if (v < b1) f1++; }
        }
        memcpy(tmp, smp + (size_t)a * rounds, sizeof(double) * (size_t)rounds);
        qsort(tmp, (size_t)rounds, sizeof(double), dcmp);
        double spread = 100.0 * (tmp[(int)(0.9 * (rounds - 1))]
                                 - tmp[(int)(0.1 * (rounds - 1))]) / m[a];
        printf("  %-14s %10.1f %6.1f%% %+8.2f%% %6d/%-3d %+8.2f%% %6d/%-3d\n",
               arms[a].label, m[a], spread,
               100.0 * (m[a] / m[0] - 1.0), f0, n0,
               100.0 * (m[a] / m[IA1] - 1.0), f1, n1);
    }
    printf("\n  reading: 'vs A0' negative = FASTER than untiled. 'sign/X' = how "
           "many of the R\n  within-round comparisons favoured this arm; ~50%% "
           "means the effect is not\n  resolvable, ~100%% means it is real "
           "regardless of thermal drift.\n  🔴 A0' (A/A) is the noise floor: no "
           "delta smaller than |A0'| is readable.\n  🔴 A1-A0 is the TWIDDLE-TABLE "
           "effect; A2-A1 (the 'vs A1' column) is the\n  PURE TILING effect. "
           "A2-A0 is their sum and is NOT a tiling number.\n");

    for (int a = 0; a < na; a++) if (arms[a].h) vfft_destroy(arms[a].h);
    vfft_wisdom_free(W);
    if (W2) vfft_wisdom_free(W2);
    return 0;
}
