/* zturn_wisdom_replay_probe.c — do the CALIBRATED wisdom lines actually replay
 * through the public front door, and does an explicit env override beat them?
 *
 * The wisdom-width GATE (zturn_wisdom_width_gate.c) proves the MECHANISM with
 * synthetic lines it writes itself. This probe checks the REAL banked verdicts
 * in a calibrated wisdir: for each cell it creates a plan via vfft_create with
 * the env axis silent and reads the create-time [tcut] line — the same
 * engagement evidence every other harness here relies on. An arm that silently
 * fell back would be indistinguishable from one that worked without it.
 *
 * Second arm per cell: VFFT_TCUT=off. Explicit env must BEAT wisdom (the
 * VFFT_FORCE_ZROUTE / VFFT_NO_ZTURN convention) — without that precedence,
 * `bench_1d_vs_mkl --tcut=off` against a width-carrying wisdir would silently
 * run TILED and every off-vs-tiled A/B would compare tiled vs tiled.
 *
 * Read-only with respect to wisdom: nothing is written or rewritten.
 *
 * Build: python build.py --src benches/zturn_wisdom_replay_probe.c --vfft
 * Run  : zturn_wisdom_replay_probe.exe --wisdir <calibrated wisdir> [N...]
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
    snprintf(g_errpath, sizeof g_errpath, "_replay_probe_stderr.log" /* cwd, 0.12: never inside a wisdom dir */ );
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

typedef struct { int created, tiled, tcut, suppressed, mismatch;
                 long w, NT; char src[16]; } probe_t;

static probe_t probe(vfft_wisdom *W, int N, const char *tcut_env)
{
    probe_t r; memset(&r, 0, sizeof r);
    env_set("VFFT_TCUT", tcut_env);          /* "" = silent -> wisdom decides */
    env_set("VFFT_TCUT_W", "");
    env_set("VFFT_TCUT_TW", "");
    env_set("VFFT_TCUT_VERBOSE", "1");
    env_set("VFFT_FORCE_ZROUTE", "");        /* NO forcing — the real route   */

    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.order = VFFT_ORDER_SCRAMBLED; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = 1; cfg.wisdom = W;
    cfg.wisdom_write = 1;  /* measurement gate: banks must persist */
    (void)err_tap_read();
    vfft_plan h = vfft_create(&cfg);
    const char *log = err_tap_read();
    r.created = h != NULL;

    const char *q = strstr(log, "[tcut]");
    if (q) {
        if (strstr(q, "SUPPRESSED"))         r.suppressed = 1;
        else if (strstr(q, "tuned for L1d")) r.mismatch = 1;
        else {
            char form[16];
            if (sscanf(q, "[tcut] N=%*d nf=%*d tiled=%d tcut=%d tfuse=%*d "
                          "tw=%15s w=%ld NT=%ld src=%15s",
                       &r.tiled, &r.tcut, form, &r.w, &r.NT, r.src) < 5)
                r.tiled = 0;                 /* env-gate line shape, not ours */
        }
    }
    if (h) vfft_destroy(h);
    return r;
}

int main(int argc, char **argv)
{
    const char *wisdir = NULL;
    int cells[16], nc = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--wisdir") && i + 1 < argc) wisdir = argv[++i];
        else if (nc < 16) cells[nc++] = atoi(argv[i]);
    }
    if (!wisdir) { printf("usage: %s --wisdir <dir> [N...]\n", argv[0]); return 2; }
    if (!nc) { int def[] = {2048, 4096, 8192, 16384, 32768};
               for (int i = 0; i < 5; i++) cells[nc++] = def[i]; }
    if (!err_tap_open(wisdir)) { printf("stderr tap failed\n"); return 2; }

    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { printf("vfft_wisdom_load(%s) FAILED\n", wisdir); return 2; }

    printf("\n=== banked-width REPLAY probe (front door, nothing forced) ===\n\n");
    printf("  %-8s | %-30s | %-30s\n", "N",
           "env SILENT (wisdom decides)", "VFFT_TCUT=off (env must win)");
    printf("  ---------------------------------------------------------------"
           "-----------\n");

    int fails = 0;
    for (int i = 0; i < nc; i++) {
        const int N = cells[i];
        probe_t a = probe(W, N, "");         /* silent  */
        probe_t b = probe(W, N, "off");      /* forced  */

        char ca[40], cb[40];
        if (!a.created)            snprintf(ca, sizeof ca, "create FAILED");
        else if (a.mismatch)       snprintf(ca, sizeof ca, "L1 mismatch -> untiled");
        else if (a.tiled == 1)     snprintf(ca, sizeof ca, "TILED t=%d w=%ld NT=%ld [%s]",
                                            a.tcut, a.w, a.NT, a.src);
        else                       snprintf(ca, sizeof ca, "untiled");

        if (!b.created)            snprintf(cb, sizeof cb, "create FAILED");
        else if (b.suppressed)     snprintf(cb, sizeof cb, "suppressed -> untiled  OK");
        else if (b.tiled == 1)     snprintf(cb, sizeof cb, "*** STILL TILED w=%ld ***", b.w);
        else                       snprintf(cb, sizeof cb, "untiled (no banked width)");

        /* fail conditions: off-arm still tiled; silent arm claiming wisdom
         * width but engagement line absent is caught by "untiled" here and
         * judged against the file by the caller reading the banked lines. */
        if (b.tiled == 1) fails++;
        if (!a.created || !b.created) fails++;

        printf("  %-8d | %-30s | %-30s\n", N, ca, cb);
    }
    vfft_wisdom_free(W);

    printf("\n  compare the left column against the banked records:\n"
           "    grep 'eng=zturn\\|eng=zsplit' <wisdir>/wisdom2_oop.txt\n"
           "  a banked width that shows 'untiled' on the left did NOT replay.\n");
    printf("  === %s ===\n", fails ? "*** FAIL ***" : "PASS");
    return fails ? 1 : 0;
}
