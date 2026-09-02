/* ADVERSARIAL reach probe: does the classic OOP champions race (oop_dp.h:136/:146)
 * actually get entered for the claimed grid cells? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int lay, ord, n; size_t K; int geom, own; } cell_t;
#define IL VFFT_LAYOUT_INTERLEAVED
#define SP VFFT_LAYOUT_SPLIT
#define DEF VFFT_ORDER_DEFAULT
#define SCR VFFT_ORDER_SCRAMBLED
#define NAT VFFT_ORDER_NATURAL

static const cell_t CELLS[] = {
  {"oop.sp.c2c.256.K32.def", SP,DEF,256,32,0,0},
  {"oop.il.c2c.256.K32.def", IL,DEF,256,32,0,0},
  {"oop.sp.c2c.256.K32.scr", SP,SCR,256,32,0,0},
  {"oop.sp.c2c.256.K32.nat", SP,NAT,256,32,0,0},
  {"oop.sp.c2c.256.K4.def",  SP,DEF,256, 4,0,0},
  {"oop.il.c2c.256.K4.def",  IL,DEF,256, 4,0,0},
  {"oop.sp.c2c.4096.K1.scr", SP,SCR,4096,1,0,0},
  {"oop.il.c2c.4096.K1.scr", IL,SCR,4096,1,0,0},
  {"oop.sp.c2c.256.K1.scr",  SP,SCR,256, 1,0,0},
  {"oop.il.c2c.256.K32.lm",  IL,DEF,256,32,VFFT_BATCH_LANE_MAJOR,0},
  {"oop.il.c2c.256.K8.lm",   IL,DEF,256, 8,VFFT_BATCH_LANE_MAJOR,0},
  {"oop.sp.c2c.256.K32.lm",  SP,DEF,256,32,VFFT_BATCH_LANE_MAJOR,0},
  {"oop.sp.c2c.256.K1.own",  SP,DEF,256, 1,0,1},
  {"oop.il.c2c.256.K1.own",  IL,DEF,256, 1,0,1},
  {"oop.sp.c2c.256.K8.own",  SP,DEF,256, 8,0,1},
  {"oop.il.c2c.256.K8.own",  IL,DEF,256, 8,0,1},
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
    if (argc < 3 || strcmp(argv[1], "--cell")) { printf("usage: --cell <0..%d>|--list\n", NCELLS-1); return 2; }
    i = atoi(argv[2]);
    if (i < 0 || i >= NCELLS) return 2;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout    = (vfft_layout_t)CELLS[i].lay;
    cfg.order     = CELLS[i].ord;
    cfg.dims      = 1;
    cfg.n[0]      = CELLS[i].n;
    cfg.howmany   = CELLS[i].K;
    cfg.batch_geom = CELLS[i].geom;
    cfg.owned_buffers = CELLS[i].own;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = 1;
    p = vfft_create(&cfg);
    printf("@@cell %s\n", CELLS[i].tag);
    vfft__fp_counters(c);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
