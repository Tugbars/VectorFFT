/* zturn_wisdom_width_gate.c — does a BANKED tcut width replay, and does a
 * width banked for a DIFFERENT cache refuse to replay?
 *
 * The width axis is only shippable if a wisdom record can carry it safely. Two
 * properties have to hold, and neither is checkable by staring at the code:
 *
 *   B0 BACK-COMPAT. A line banked before this axis existed has no width field,
 *      and MUST replay as UNTILED — byte-for-byte today's driver. There is no
 *      sentinel to forget: absent parses as 0 and 0 means untiled.
 *   B1 CACHE FENCE. A width is a property of one machine's L1, not of the
 *      transform. Replaying a 48 KB-tuned width on a 32 KB cache overshoots by
 *      50%, and overshoot loses the whole benefit at once rather than
 *      degrading. So a stamped-L1 mismatch must fall back to UNTILED and SAY
 *      SO — never "use it anyway".
 *
 * Also checked: an illegal width for the banked chain is refused (not
 * force-fit), and a legal one actually engages with the geometry that was
 * banked.
 *
 * Everything goes through the PUBLIC front door (vfft_create), and the verdict
 * is read from the create-time [tcut] line — an arm that silently fell back
 * would otherwise be indistinguishable from one that worked.
 *
 * 🔴 Point --wisdir at a SCRATCH copy. This gate REWRITES oop_wisdom.txt.
 *
 * Build: python build.py --src benches/zturn_wisdom_width_gate.c --vfft
 * Run  : zturn_wisdom_width_gate.exe --wisdir <scratch wisdir> [--cell 16384]
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
    snprintf(g_errpath, sizeof g_errpath, "%s/_wisdom_width_gate.log", dir);
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

/* ── wisdom surgery: rewrite the kind-4 line for one cell ──────────────── */
#define MAXL 4096
static char  g_lines[MAXL][512];
static int   g_nlines = 0;
static char  g_path[1024];
static int   g_target = -1;          /* index of the "N 1 4 ..." line */

static int wis_load(const char *dir, int N)
{
    snprintf(g_path, sizeof g_path, "%s/oop_wisdom.txt", dir);
    FILE *f = fopen(g_path, "r");
    if (!f) return 0;
    g_nlines = 0; g_target = -1;
    char l[512];
    while (g_nlines < MAXL && fgets(l, sizeof l, f)) {
        strncpy(g_lines[g_nlines], l, sizeof g_lines[0] - 1);
        g_lines[g_nlines][sizeof g_lines[0] - 1] = 0;
        int n = 0, k = 0, kind = 0;
        if (sscanf(l, "%d %d %d", &n, &k, &kind) == 3
            && n == N && k == 1 && kind == 4)
            g_target = g_nlines;
        g_nlines++;
    }
    fclose(f);
    return g_target >= 0;
}

/* Replace the target line. `tail` is everything AFTER ns. */
static void wis_write(int N, double ns, int cc_chain, const char *tail)
{
    snprintf(g_lines[g_target], sizeof g_lines[0],
             "%d 1 4 0 %d %.1f%s%s\n", N, cc_chain, ns,
             tail && tail[0] ? " " : "", tail ? tail : "");
    FILE *f = fopen(g_path, "w");
    if (!f) return;
    for (int i = 0; i < g_nlines; i++) fputs(g_lines[i], f);
    fclose(f);
}

/* ── create one plan and report what tiling it engaged ─────────────────── */
typedef struct { int ok, tiled, tcut, refused_or_absent; long w, NT; char note[160]; } res_t;

static res_t make(const char *wisdir, int N)
{
    res_t r; memset(&r, 0, sizeof r);
    /* the env axis must be OFF: we are testing the WISDOM path */
    putenv((char *)"VFFT_TCUT=");
    putenv((char *)"VFFT_TCUT_W=");
    putenv((char *)"VFFT_TCUT_VERBOSE=1");
    putenv((char *)"VFFT_FORCE_ZROUTE=zturn");

    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { snprintf(r.note, sizeof r.note, "wisdom load failed"); return r; }
    (void)err_tap_read();

    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.order = VFFT_ORDER_SCRAMBLED; cfg.layout = VFFT_LAYOUT_INTERLEAVED;
    cfg.nthreads = 1; cfg.wisdom = W;
    vfft_plan h = vfft_create(&cfg);
    const char *log = err_tap_read();
    r.ok = h != NULL;

    const char *q = strstr(log, "[tcut]");
    if (q) {
        char form[16];
        if (strstr(q, "tuned for L1d") || strstr(q, "ILLEGAL")) {
            r.refused_or_absent = 1;
            const char *nl = strchr(q, '\n');
            size_t len = nl ? (size_t)(nl - q) : strlen(q);
            if (len > sizeof r.note - 1) len = sizeof r.note - 1;
            memcpy(r.note, q, len); r.note[len] = 0;
        } else {
            sscanf(q, "[tcut] N=%*d nf=%*d tiled=%d tcut=%d tfuse=%*d tw=%15s "
                      "w=%ld NT=%ld", &r.tiled, &r.tcut, form, &r.w, &r.NT);
        }
    } else {
        r.refused_or_absent = 1;
        snprintf(r.note, sizeof r.note, "(no [tcut] line -> untiled)");
    }
    if (h) vfft_destroy(h);
    vfft_wisdom_free(W);
    return r;
}

int main(int argc, char **argv)
{
    const char *wisdir = NULL;
    int N = 16384;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--wisdir") && i + 1 < argc) wisdir = argv[++i];
        else if (!strcmp(argv[i], "--cell") && i + 1 < argc) N = atoi(argv[++i]);
    }
    if (!wisdir) { printf("usage: %s --wisdir <SCRATCH dir> [--cell N]\n", argv[0]); return 2; }
    if (!err_tap_open(wisdir)) { printf("stderr tap failed\n"); return 2; }
    if (!wis_load(wisdir, N)) {
        printf("no kind-4 (N=%d K=1) line in %s/oop_wisdom.txt — nothing to test\n",
               N, wisdir);
        return 2;
    }
    char original[512];
    strncpy(original, g_lines[g_target], sizeof original - 1);
    original[sizeof original - 1] = 0;

    const int CC = 232223;              /* 4.8.4.4.4.8, the banked 16384 chain */
    const long L1 = 48 * 1024;          /* what this build sizes against       */
    int fails = 0;

    printf("\n=== banked tcut WIDTH replay gate  N=%d ===\n", N);
    printf("  (env axis disabled; every verdict comes from oop_wisdom.txt)\n\n");
    printf("  %-34s %-26s %s\n", "banked tail (after ns)", "engaged", "verdict");
    printf("  --------------------------------------------------------------"
           "------------------\n");

    struct { const char *tail; const char *want; int want_tiled; long want_w; } CASES[] = {
        /* B0: legacy line, no route pair at all -> untiled                  */
        { "",              "UNTILED (legacy line)",            0, 0 },
        /* legacy route pair, still no width -> untiled                      */
        { "1 0",           "UNTILED (route, no width)",        0, 0 },
        /* B1a: width + MATCHING L1 -> engages                               */
        { "1 0 1024 49152","TILED w=1024",                     1, 1024 },
        { "1 0 2048 49152","TILED w=2048",                     1, 2048 },
        /* B1b: width tuned for a DIFFERENT cache -> untiled + loud          */
        { "1 0 1024 32768","UNTILED (L1 mismatch)",            0, 0 },
        /* illegal width for this chain (1536 does not divide SEC=4096)      */
        { "1 0 1536 49152","UNTILED (illegal width)",          0, 0 },
    };
    const int nc = (int)(sizeof CASES / sizeof CASES[0]);

    for (int c = 0; c < nc; c++) {
        wis_write(N, 49306.2, CC, CASES[c].tail);
        res_t r = make(wisdir, N);
        const int got_tiled = (r.tiled == 1 && !r.refused_or_absent);
        const int ok = r.ok
                    && got_tiled == (CASES[c].want_tiled != 0)
                    && (!CASES[c].want_tiled || r.w == CASES[c].want_w);
        if (!ok) fails++;
        char eng[40];
        if (got_tiled) snprintf(eng, sizeof eng, "tiled t=%d w=%ld NT=%ld",
                                r.tcut, r.w, r.NT);
        else           snprintf(eng, sizeof eng, "untiled");
        printf("  %-34s %-26s %s\n",
               CASES[c].tail[0] ? CASES[c].tail : "(nothing after ns)",
               eng, ok ? "PASS" : "*** FAIL ***");
        if (!ok || r.note[0])
            printf("      expected %s%s%s\n", CASES[c].want,
                   r.note[0] ? "  |  " : "", r.note[0] ? r.note : "");
    }

    /* restore, so a scratch dir stays reusable */
    strncpy(g_lines[g_target], original, sizeof g_lines[0] - 1);
    { FILE *f = fopen(g_path, "w");
      if (f) { for (int i = 0; i < g_nlines; i++) fputs(g_lines[i], f); fclose(f); } }

    printf("\n  === %d PASS, %d FAIL ===\n", nc - fails, fails);
    printf("  B0 back-compat: a pre-width line replays UNTILED, no sentinel needed.\n"
           "  B1 cache fence: a width tuned for another L1 is REFUSED, not reused.\n");
    printf("  (original wisdom line restored)\n");
    return fails ? 1 : 0;
}
