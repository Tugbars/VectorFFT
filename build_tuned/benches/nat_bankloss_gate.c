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

/* ── the seeded-loss arm. No cell on this box loses its race naturally
 * (challengers win 3-8x everywhere probed, 2026-09-02), so the LOSS side
 * is exercised by munging the gate's OWN scratch store: the banked 4096
 * @nat line becomes a tape verdict (mode=pcyc) — once WITHOUT zr (the
 * CONTROL: the create must rebuild the candidate and re-race, which is
 * exactly the pre-fix disease) and once WITH zr=1 (the TREATMENT: the
 * banked loss must consume silently, no candidate, no race). */
static int munge_4096(const char *wisdir, int with_zr)
{
    /* 2026-09-03 (no split baseline for IL): the seeded LOSS is the OTHER IL
     * verdict on the 4096 @nat row — mode=zcasc becomes mode=ilp (the pair
     * won, the cascade is the banked loss) and mode=ilp becomes mode=zcasc
     * with the comp-recipe signpost (the cascade won, the pair is the loss).
     * Either way the row is a valid IL verdict that must CONSUME SILENTLY.
     * The old munge seeded a split TAPE (mode=pcyc), which is no IL verdict
     * any more and would re-race by law. with_zr: 0 = flip to ilp, 1 = flip
     * to zcasc. */
    char pb[1024];
    snprintf(pb, sizeof pb, "%s/wisdom2_oop.txt", wisdir);
    FILE *f = fopen(pb, "r");
    if (f)
    {
        char probe[65536];
        size_t pn = fread(probe, 1, sizeof probe - 1, f);
        probe[pn] = 0;
        if (!strstr(probe, "n=4096 q=1 ord=nat place=ip")) { fclose(f); f = NULL; }
        else { fclose(f); f = fopen(pb, "r"); }
    }
    if (!f)
    {
        snprintf(pb, sizeof pb, "%s/wisdom2_scr.txt", wisdir);
        f = fopen(pb, "r");
    }
    if (!f) return 0;
    static char text[65536], out[65536];
    size_t n = fread(text, 1, sizeof text - 1, f);
    text[n] = 0;
    fclose(f);
    char *line = strstr(text, "n=4096 q=1 ord=nat place=ip");
    if (!line) return 0;
    char *eol = strchr(line, '\n');
    if (!eol) eol = text + n;
    char *mz = strstr(line, "mode=zcasc"), *mi = strstr(line, "mode=ilp");
    if (mz && mz > eol) mz = NULL;
    if (mi && mi > eol) mi = NULL;
    if (!with_zr && !mz) return 0;   /* flip-to-ilp needs a zcasc row */
    if (with_zr && !mi) return 0;    /* flip-to-zcasc needs an ilp row */
    {
        size_t head = (size_t)(line - text), o = 0;
        char *tok = with_zr ? mi : mz;
        const char *tail_from = tok + (with_zr ? 8 : 10);   /* past mode=ilp / mode=zcasc */
        memcpy(out, text, head); o = head;
        memcpy(out + o, line, (size_t)(tok - line)); o += (size_t)(tok - line);
        if (with_zr)
        {
            const char *rep = "mode=zcasc ref=cell(t=c2c,n=4096,q=1,ord=scr,place=oop,role=comp)";
            memcpy(out + o, rep, strlen(rep)); o += strlen(rep);
        }
        else
        {
            const char *rep = "mode=ilp";
            const char *r = strstr(tail_from, " ref=cell(");
            memcpy(out + o, rep, strlen(rep)); o += strlen(rep);
            if (r && r < eol) tail_from = strchr(r + 1, ')') + 1;   /* drop the old signpost */
        }
        memcpy(out + o, tail_from, (size_t)(text + n - tail_from)); o += (size_t)(text + n - tail_from);
        out[o] = 0;
        n = o;
    }
    f = fopen(pb, "w");
    if (!f) return 0;
    fwrite(out, 1, n, f);
    fclose(f);
    return 1;
}

static int run_seeded(const char *wisdir, int with_zr, int expect_race, int fails)
{
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    vfft_config_t cfg;
    if (!W) { printf("seeded: wisdom load FAILED\n"); return fails + 1; }
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_INPLACE;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.order = VFFT_ORDER_NATURAL;
    cfg.dims = 1;
    cfg.n[0] = 4096;
    cfg.howmany = 1;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom = W;
    cfg.wisdom_write = 1;
    vfft_plan p = vfft_create(&cfg);
    const char *log = err_tap_read();
    int raced = tap_raced(log);
    int ok = p && (raced == expect_race);
    if (p) vfft_destroy(p);
    vfft_wisdom_free(W);
    printf("4096    seeded-%-7s -> %-9s (expect %s) %s\n",
           with_zr ? "zcasc" : "ilp", raced ? "raced" : "silent",
           expect_race ? "raced" : "silent", ok ? "" : "  *** FAIL ***");
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
    /* the loss side, seeded (see munge_4096): control then treatment.
     * The control re-banks a fresh ZCASC win, so munge again for the
     * treatment. */
    /* the seeded LOSS (2026-09-03 law): whichever IL verdict the 4096
     * @nat row holds, flip it to the other one; it must consume silently
     * (a banked IL verdict never re-races; the loser is implied). */
    if (munge_4096(wisdir, /*with_zr=*/0))
        fails = run_seeded(wisdir, 0, /*expect_race=*/0, fails);
    else if (munge_4096(wisdir, /*with_zr=*/1))
        fails = run_seeded(wisdir, 1, /*expect_race=*/0, fails);
    else
        { printf("seeded-loss munge FAILED (no IL verdict on the 4096 @nat row)\n"); fails++; }
    printf(fails ? "=== *** FAIL *** ===\n" : "=== ALL PASS ===\n");
    return fails ? 1 : 0;
}
