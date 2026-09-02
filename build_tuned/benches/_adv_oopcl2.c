/* ADVERSARIAL reach probe #2: K not a multiple of 8, and K=16, split+IL. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
typedef struct { const char *tag; int lay, ord, n; size_t K; int geom; } cell_t;
#define IL VFFT_LAYOUT_INTERLEAVED
#define SP VFFT_LAYOUT_SPLIT
#define DEF VFFT_ORDER_DEFAULT
#define SCR VFFT_ORDER_SCRAMBLED
#define NAT VFFT_ORDER_NATURAL
static const cell_t CELLS[] = {
  {"sp.256.K16.def", SP,DEF,256,16,0},
  {"sp.256.K16.nat", SP,NAT,256,16,0},
  {"sp.256.K16.scr", SP,SCR,256,16,0},
  {"sp.256.K12.def", SP,DEF,256,12,0},
  {"sp.256.K12.nat", SP,NAT,256,12,0},
  {"il.256.K16.def", IL,DEF,256,16,0},
  {"il.256.K16.lm",  IL,DEF,256,16,VFFT_BATCH_LANE_MAJOR},
  {"il.256.K12.lm",  IL,DEF,256,12,VFFT_BATCH_LANE_MAJOR},
  {"il.256.K16.tc",  IL,DEF,256,16,VFFT_BATCH_TRANSFORM_CONTIGUOUS},
};
#define NCELLS ((int)(sizeof CELLS / sizeof CELLS[0]))
int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p; int i;
    if (argc > 1 && !strcmp(argv[1], "--list")) {
        for (i = 0; i < NCELLS; i++) printf("%2d %s\n", i, CELLS[i].tag);
        return 0;
    }
    if (argc < 3 || strcmp(argv[1], "--cell")) return 2;
    i = atoi(argv[2]); if (i < 0 || i >= NCELLS) return 2;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout = (vfft_layout_t)CELLS[i].lay; cfg.order = CELLS[i].ord;
    cfg.dims = 1; cfg.n[0] = CELLS[i].n; cfg.howmany = CELLS[i].K;
    cfg.batch_geom = CELLS[i].geom; cfg.rigor = VFFT_MEASURE; cfg.wisdom_write = 1;
    p = vfft_create(&cfg);
    printf("@@cell %s\n", CELLS[i].tag);
    vfft__fp_counters(c);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf); fputs(buf, stdout);
    vfft_destroy(p); return 0;
}
