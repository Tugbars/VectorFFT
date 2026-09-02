/* fp_sweep.c — thin driver for the create-time plan fingerprint.
 *
 * Emits one @fp tree per corpus cell, plus the engagement/purity counters.
 * The artifact is diffed byte-for-byte across every migration step; see
 * docs/design/refactor_safety_harness.md section 2.8.
 *
 * ONE PROCESS PER CELL is the only mode whose output may be diffed. Several
 * things in the library are process-lifetime, not per-plan: the K=1 pair-order
 * memo (keyed by N, consulted before the race), the thread-pool size, and the
 * QPC frequency cache. Run several cells in one process and the second one
 * inherits the first one's memo, so reordering the corpus produces a diff that
 * is not a bug. `--all` exists for triage only and writes nowhere a verdict
 * reads it.
 *
 * REPLAY PURITY is an assertion, not an observation: under replay the
 * create-race counter must end at ZERO. A cell that races while replaying has
 * the clock inside its own baseline and will false-diff on the first thermal
 * wobble. This driver reports the counter so the sweep can fail on it.
 *
 * Build: VFFT_FINGERPRINT=1 python build.py --src benches/fp_sweep.c --vfft --compile
 *        (build.py reads the flag from the ENVIRONMENT; it has no --define)
 * Run  : fp_sweep.exe --cell <i>        (one cell, the diffable mode)
 *        fp_sweep.exe --all             (triage only)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct {
    const char *name;
    vfft_transform_t xf;
    vfft_placement_t place;
    int layout, order, dims, n0, n1;
    size_t K;
} cell_t;

static const cell_t CELLS[] = {
    {"c2c.split.ip.N256",    VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 256, 0, 1},
    {"c2c.split.ip.N1024",   VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 1024,0, 1},
    {"c2c.split.ip.K32",     VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 256, 0, 32},
    {"c2c.split.ip.K4",      VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 256, 0, 4},
    {"c2c.split.oop.N256",   VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 256, 0, 1},
    {"c2c.il.ip.N256",       VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT,   1, 256, 0, 1},
    {"c2c.split.ip.scr",     VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_SCRAMBLED, 1, 256, 0, 1},
    {"c2c.split.ip.nat",     VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_NATURAL,   1, 256, 0, 1},
    {"r2c.split.oop.N1024",  VFFT_R2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 1024,0, 1},
    {"c2r.split.oop.N1024",  VFFT_C2R, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 1024,0, 1},
    {"dct2.split.oop.N256",  VFFT_DCT2,VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT,   1, 256, 0, 1},
    {"c2c.2d.ip.64x64",      VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT,   2, 64,  64,1},
};
#define NCELLS ((int)(sizeof CELLS / sizeof CELLS[0]))

static void one(int i)
{
    static char buf[16384];
    vfft_config_t cfg;
    vfft_plan p;
    long c[VFFT__FP_NCOUNTERS];

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = CELLS[i].xf;
    cfg.placement = CELLS[i].place;
    cfg.layout    = (vfft_layout_t)CELLS[i].layout;
    cfg.order     = CELLS[i].order;
    cfg.dims      = CELLS[i].dims;
    cfg.n[0]      = CELLS[i].n0;
    cfg.n[1]      = CELLS[i].n1;
    cfg.howmany   = CELLS[i].K;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = 0;              /* serving mode: never write the store */

    printf("@cell %s\n", CELLS[i].name);
    p = vfft_create(&cfg);
    if (!p) { printf("@fp REFUSED\n"); return; }
    /* Absolute counters are correct HERE, with no entry baseline to subtract,
     * because this driver does exactly one create per process. harness_golden
     * needs a baseline only because its refusal check creates a plan first and
     * absorbs that plan's race. */
    vfft__fp_counters(c);
    if (c[5] != 0) {
        /* RACED, so the plan is a coin flip and neither the fingerprint nor the
         * engagement counters may be written: they would re-diff on the next
         * thermal wobble and train us to ignore this file. Record the FACT of
         * the race instead. That is not a hole - it is the sharper check. The
         * race count is a property of the wisdom store, not of the weather, so
         * a step that makes a banked cell start racing (a broken lookup key, a
         * dropped shard) flips this line and the diff catches it, while a step
         * that leaves it alone is silent. Same convention as harness_golden's
         * NOT_BANKED_RACED. */
        printf("@fp NOT_BANKED_RACED races=%ld\n", c[5]);
        vfft_destroy(p);
        return;
    }
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    printf("@counters tc=%ld il2dcol=%ld zt=%ld pq=%ld trig=%ld races=%ld\n",
           c[0], c[1], c[2], c[3], c[4], c[5]);
    vfft_destroy(p);
}

#ifdef _WIN32
#include <io.h>
#include <fcntl.h>
#endif

int main(int argc, char **argv)
{
    int i;
#ifdef _WIN32
    /* stdout to BINARY. In text mode msvcrt turns every \n into \r\n, but
     * .gitattributes pins the baseline directory to eol=lf, so a committed
     * artifact is LF while a fresh capture comes back CRLF - and the diff then
     * reports every single row as changed on a byte-identical result. An
     * artifact compared byte-for-byte has to be written byte-for-byte. */
    _setmode(_fileno(stdout), _O_BINARY);
#endif
    if (argc > 2 && !strcmp(argv[1], "--cell")) {
        i = atoi(argv[2]);
        if (i < 0 || i >= NCELLS) { printf("cell out of range 0..%d\n", NCELLS - 1); return 2; }
        one(i);
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--list")) {
        for (i = 0; i < NCELLS; i++) printf("%2d %s\n", i, CELLS[i].name);
        return 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--all")) {
        printf("# TRIAGE MODE - one process, so process-lifetime state leaks\n"
               "# between cells. Do NOT diff this output.\n");
        for (i = 0; i < NCELLS; i++) one(i);
        return 0;
    }
    printf("usage: %s --cell <0..%d> | --list | --all\n", argv[0], NCELLS - 1);
    return 2;
}
