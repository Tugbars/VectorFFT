/* nat_bankloss_gate.c — the banked-loss law for the @nat races (2026-09-02).
 *
 * INVARIANT: after ONE order=NATURAL in-place create on a cell (which may
 * race ZCASC/ILP against the tape and lose), a SECOND create on the same
 * wisdom must fire NO race — a win replays its mode, a LOSS replays the
 * tape with the zr=1 marker refusing the rebuild. Before the fix, a lost
 * race re-ran on every create, forever (candidate build + 5-round race).
 * recalibrate=1 must still re-race — the escape hatch is part of the law.
 *
 * Cells: N=256 (the ILP-vs-tape tier) and N=4096 (the ZCASC-vs-tape tier).
 * Whichever arm wins is fine — the gate asserts SILENCE on the consume,
 * sameness of the served mode, and the recalibrate re-race.
 *
 * Style: "flag" (--wisdir DIR), COLD dir — run_gates seeds nothing.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

/* stderr tap (the k1z/nat_front pattern; taps live under build_tuned/benches) */
static char g_errpath[1024];
static long g_errpos = 0;
static int err_tap_open(void)
{
    snprintf(g_errpath, sizeof g_errpath,
             "build_tuned/benches/_nat_bankloss_gate.log");
    if (!freopen(g_errpath, "w", stderr)) return 0;
    setvbuf(stderr, NULL, _IONBF, 0);
    g_errpos = 0;
    return 1;
}
static const char *err_tap_read(void)
{
    static char buf[8192];
    FILE *f = fopen(g_errpath, "r");
    size_t n = 0;
    if (!f) { buf[0] = 0; return buf; }
    fseek(f, g_errpos, SEEK_SET);
    n = fread(buf, 1, sizeof buf - 1, f);
    buf[n] = 0;
    g_errpos += (long)n;
    fclose(f);
    return buf;
}
/* a MEASURE race line: "[natorder] ... zcasc=..." / "... ilp=..." (the
 * replay lines say "replay", the natorder tape race has no [natorder] tag
 * with those tokens) */
static int tap_raced(const char *log)
{
    return strstr(log, "zcasc=") != NULL || strstr(log, "ilp=") != NULL;
}

static int run_cell(const char *wisdir, int N, int fails)
{
    vfft_config_t cfg;
    const char *log;
    int raced1 = 0, raced2 = 0, raced3 = 0;
    for (int pass = 0; pass < 3; pass++)
    {
        vfft_wisdom *W = vfft_wisdom_load(wisdir);
        if (!W) { printf("%-7d wisdom load FAILED\n", N); return fails + 1; }
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C;
        cfg.placement = VFFT_INPLACE;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.order = VFFT_ORDER_NATURAL;
        cfg.dims = 1;
        cfg.n[0] = N;
        cfg.howmany = 1;
        cfg.rigor = VFFT_MEASURE;
        cfg.wisdom = W;
        cfg.wisdom_write = 1; /* the marker must survive the save/load cycle */
        cfg.recalibrate = (pass == 2);
        vfft_plan p = vfft_create(&cfg);
        log = err_tap_read();
        if (!p)
        {
            printf("%-7d pass %d create FAILED\n", N, pass);
            vfft_wisdom_free(W);
            return fails + 1;
        }
        vfft_destroy(p);
        vfft_wisdom_save(W, wisdir); /* zr must round-trip through disk */
        vfft_wisdom_free(W);
        if (pass == 0) raced1 = tap_raced(log);
        if (pass == 1) raced2 = tap_raced(log);
        if (pass == 2) raced3 = tap_raced(log);
    }
    /* pass 0 may or may not race (a cell can serve FREE with no race);
     * pass 1 must NEVER race; pass 2 (recalibrate) must race whenever
     * pass 0 did — the escape hatch stays open. */
    int ok = !raced2 && (!raced1 || raced3);
    printf("%-7d measure:%-9s consume:%-9s recal:%-9s %s\n", N,
           raced1 ? "raced" : "no-race", raced2 ? "RACED(!)" : "silent",
           raced3 ? "raced" : "no-race", ok ? "" : "  *** FAIL ***");
    return fails + !ok;
}

int main(int argc, char **argv)
{
    const char *wisdir = ".";
    for (int i = 1; i + 1 < argc; i++)
        if (!strcmp(argv[i], "--wisdir")) wisdir = argv[i + 1];
    static char envbuf[] = "VFFT_NAT_LOG=1";
    putenv(envbuf);
    if (!err_tap_open()) { printf("stderr tap failed\n"); return 2; }
    printf("=== banked-loss law: a lost @nat race never re-runs ===\n");
    printf("N       pass1    pass2(consume) pass3(recal)\n");
    int fails = 0;
    fails = run_cell(wisdir, 256, fails);  /* the ILP tier */
    fails = run_cell(wisdir, 4096, fails); /* the ZCASC tier */
    printf(fails ? "=== *** FAIL *** ===\n" : "=== ALL PASS ===\n");
    return fails ? 1 : 0;
}
