/* k1z_inplace_gate.c — A2: front-door gate for IN-PLACE K=1 SCRAMBLED
 * interleaved c2c served by the cascade.
 *
 * 🔴 EVERYTHING through the PUBLIC front door (vfft_create / vfft_execute) —
 * the plan-level probe (P0a) proved the kernels; this gate proves the
 * PLUMBING: the in-place handle actually carries the cascade, the banked tile
 * width actually engages, and the aliased execute matches the OOP handle's
 * output byte for byte.
 *
 * ENGAGEMENT is checked two independent ways, because a silently-fallen-back
 * arm is the recorded failure class of this whole campaign:
 *   1. the create-time [tcut] line (stderr tap) must show src=wisdom on the
 *      cells that banked a width — for the IN-PLACE create;
 *   2. memcmp vs the OOP cascade output IS itself an engagement check: the
 *      classic in-place path serves a DIFFERENT plan whose scrambled output
 *      cannot be byte-identical to the cascade's.
 *
 * Matrix: banked cells 2048..32768 × {fwd, bwd} + a VFFT_TCUT=off arm at one
 * cell (env override must beat wisdom on the in-place path too).
 *
 * Run: k1z_inplace_gate.exe --wisdir <scratch wisdir>
 * Build: python build.py --src benches/k1z_inplace_gate.c --vfft --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

/* ── stderr tap ────────────────────────────────────────────────────────── */
static char g_errpath[1024];
static long g_errpos = 0;
static int err_tap_open(const char *dir)
{
    snprintf(g_errpath, sizeof g_errpath, "_k1z_ip_gate.log" /* cwd, 0.12: never inside a wisdom dir */ );
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

static vfft_plan mk(vfft_wisdom *W, int N, int inplace)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = inplace ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = 1;
    cfg.order = VFFT_ORDER_SCRAMBLED;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = 1;
    cfg.wisdom = W;
    cfg.wisdom_write = 1;  /* measurement gate: banks must persist */
    return vfft_create(&cfg);
}

int main(int argc, char **argv)
{
    const char *wisdir = NULL;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--wisdir") && i + 1 < argc) wisdir = argv[++i];
    if (!wisdir) { printf("usage: %s --wisdir <SCRATCH dir>\n", argv[0]); return 2; }
    if (!err_tap_open(wisdir)) { printf("stderr tap failed\n"); return 2; }

    env_set("VFFT_TCUT", "");
    env_set("VFFT_TCUT_W", "");
    env_set("VFFT_TCUT_TW", "");
    env_set("VFFT_TCUT_VERBOSE", "1");
    env_set("VFFT_FORCE_ZROUTE", "");

    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { printf("vfft_wisdom_load FAILED\n"); return 2; }

    static const int NS[] = { 2048, 4096, 8192, 16384, 32768 };
    /* which cells banked a width (this scratch bank) — [tcut] src=wisdom must
     * appear for these on the IN-PLACE create too */
    static const int WANT_TILE[] = { 0, 1, 1, 1, 1 };

    int fails = 0;
    printf("\n=== A2: front-door IN-PLACE cascade gate (public API) ===\n");
    printf("%-7s | %-22s | %-12s %-12s | %s\n",
           "N", "in-place create engaged", "fwd vs OOP", "bwd vs OOP", "verdict");

    for (size_t i = 0; i < sizeof NS / sizeof NS[0]; i++)
    {
        const int N = NS[i];
        vfft_plan ho = mk(W, N, 0);
        (void)err_tap_read();
        vfft_plan hi = mk(W, N, 1);
        const char *log = err_tap_read();
        const int saw_wisdom = strstr(log, "src=wisdom") != NULL;

        if (!ho || !hi)
        {
            printf("%-7d | create FAILED (oop=%p ip=%p)\n", N, (void *)ho, (void *)hi);
            fails++;
            continue;
        }

        double *x = az((size_t)N), *r = az((size_t)N), *a = az((size_t)N);
        double *rb = az((size_t)N);
        srand(23 + N);
        for (int k = 0; k < 2 * N; k++) x[k] = (double)rand() / RAND_MAX - 0.5;

        /* OOP references */
        vfft_execute(ho, VFFT_FORWARD, x, NULL, r, NULL);
        vfft_execute(ho, VFFT_BACKWARD, r, NULL, rb, NULL);

        /* in-place INTERLEAVED call form: dre==sre aliased (the interleaved
         * contract requires dre non-NULL — the NULL-dest form is the SPLIT
         * contract; first gate run used it and executed NOTHING, 10/10). */
        memcpy(a, x, 2 * (size_t)N * sizeof(double));
        vfft_execute(hi, VFFT_FORWARD, a, NULL, a, NULL);
        const int fok = memcmp(a, r, 2 * (size_t)N * sizeof(double)) == 0;

        memcpy(a, r, 2 * (size_t)N * sizeof(double));
        vfft_execute(hi, VFFT_BACKWARD, a, NULL, a, NULL);
        const int bok = memcmp(a, rb, 2 * (size_t)N * sizeof(double)) == 0;

        const int eng_ok = WANT_TILE[i] ? saw_wisdom : 1;
        const int ok = fok && bok && eng_ok;
        if (!ok) fails++;
        printf("%-7d | %-22s | %-12s %-12s | %s\n", N,
               WANT_TILE[i] ? (saw_wisdom ? "tiled [src=wisdom]" : "*** NO WIDTH ***")
                            : "untiled (as banked)",
               fok ? "EXACT" : "*** DIFFER ***",
               bok ? "EXACT" : "*** DIFFER ***",
               ok ? "PASS" : "*** FAIL ***");

        fz(x); fz(r); fz(a); fz(rb);
        vfft_destroy(ho);
        vfft_destroy(hi);
    }

    /* env override must beat wisdom on the in-place path too */
    {
        const int N = 16384;
        env_set("VFFT_TCUT", "off");
        (void)err_tap_read();
        vfft_plan hi = mk(W, N, 1);
        const char *log = err_tap_read();
        const int suppressed = strstr(log, "SUPPRESSED") != NULL;
        env_set("VFFT_TCUT", "");
        printf("%-7d | VFFT_TCUT=off: %-24s | %s\n", N,
               suppressed ? "banked width suppressed" : "*** NOT SUPPRESSED ***",
               suppressed ? "PASS" : "*** FAIL ***");
        if (!suppressed) fails++;
        if (hi) vfft_destroy(hi);
    }

    vfft_wisdom_free(W);
    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    printf("memcmp-EXACT is itself an engagement proof: the classic in-place\n"
           "path serves a different plan whose output cannot match the\n"
           "cascade's byte-for-byte.\n");
    return fails ? 1 : 0;
}
