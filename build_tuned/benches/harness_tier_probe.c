/* harness_tier_probe.c - do the three IL 1D tiers replay, and which fingerprint
 * fields does each light up?
 *
 * docs/design/il_codelet_design.md S1: interleaved 1D is THREE engines by N band
 *   N <= 64      mono, whole-N IL kernel
 *   128..1024    pure IL Bailey pair (il2p/il3p), interleaved throughout
 *   N >= 2048    boundary-split cascade (zsplit/zturn), split interior
 * and S3: dp_planner_il.h races BOTH cascade routes on every miss, so an
 * unbanked cascade cell races and cannot carry harness signal.
 *
 * Build: VFFT_FINGERPRINT=1 python build.py --src benches/harness_tier_probe.c --vfft --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *name; int lay, place, ord, n; size_t K; } cell_t;

static const cell_t CELLS[] = {
    {"il.mono.N16",    VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE, VFFT_ORDER_DEFAULT,   16,   1},
    {"il.mono.N64",    VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE, VFFT_ORDER_DEFAULT,   64,   1},
    {"il.bailey.N128", VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE, VFFT_ORDER_DEFAULT,  128,   1},
    {"il.bailey.N256", VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE, VFFT_ORDER_DEFAULT,  256,   1},
    {"il.bailey.N1024",VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE, VFFT_ORDER_DEFAULT, 1024,   1},
    {"il.casc.N2048",  VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE, VFFT_ORDER_DEFAULT, 2048,   1},
    {"il.casc.N4096",  VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE, VFFT_ORDER_DEFAULT, 4096,   1},
    {"il.casc.N8192",  VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE, VFFT_ORDER_DEFAULT, 8192,   1},
    {"il.casc.N4096.oop", VFFT_LAYOUT_INTERLEAVED, VFFT_OUTOFPLACE, VFFT_ORDER_DEFAULT, 4096, 1},
    {"il.casc.N4096.nat", VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE, VFFT_ORDER_NATURAL, 4096, 1},
    {"sp.K1.N256",     VFFT_LAYOUT_SPLIT, VFFT_INPLACE, VFFT_ORDER_DEFAULT, 256,  1},
    {"sp.K8.N256",     VFFT_LAYOUT_SPLIT, VFFT_INPLACE, VFFT_ORDER_DEFAULT, 256,  8},
    {"sp.K32.N256",    VFFT_LAYOUT_SPLIT, VFFT_INPLACE, VFFT_ORDER_DEFAULT, 256, 32},
    {"sp.casc.N4096",  VFFT_LAYOUT_SPLIT, VFFT_INPLACE, VFFT_ORDER_DEFAULT, 4096, 1},
};
#define NCELLS ((int)(sizeof CELLS / sizeof CELLS[0]))

int main(int argc, char **argv)
{
    static char buf[16384];
    long c[VFFT__FP_NCOUNTERS];
    vfft_config_t cfg; vfft_plan p; int i;

    if (argc > 1 && !strcmp(argv[1], "--list")) {
        for (i = 0; i < NCELLS; i++) printf("%2d %s\n", i, CELLS[i].name);
        return 0;
    }
    if (argc < 3 || strcmp(argv[1], "--cell")) { printf("usage: --cell i | --list\n"); return 2; }
    i = atoi(argv[2]);
    if (i < 0 || i >= NCELLS) return 2;

    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = (vfft_placement_t)CELLS[i].place;
    cfg.layout    = (vfft_layout_t)CELLS[i].lay;
    cfg.order     = CELLS[i].ord;
    cfg.dims      = 1;
    cfg.n[0]      = CELLS[i].n;
    cfg.howmany   = CELLS[i].K;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = 0;

    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    if (!p) { printf("%-20s REFUSE\n", CELLS[i].name); return 0; }
    vfft__fingerprint(p, buf, sizeof buf);
    {   /* pull just the route selectors out of the fp tree */
        char *k1 = strstr(buf, "k1="), *zr = strstr(buf, "zroute=");
        printf("%-20s races=%ld  %.*s\n", CELLS[i].name, c[5],
               zr ? (int)(strstr(zr, " ilme=") ? strstr(zr, " ilme=") - k1 : 60) : 40,
               k1 ? k1 : "(no k1)");
    }
    vfft_destroy(p);
    return 0;
}
