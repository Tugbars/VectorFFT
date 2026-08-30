/* _reach_blu_probe.c - reach check for the il2d Bluestein column route. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int t, lay, place, ord, dims, n0, n1; size_t K; } cell_t;
#define C2C VFFT_C2C
#define R2C VFFT_R2C
#define IL  VFFT_LAYOUT_INTERLEAVED
#define IP  VFFT_INPLACE
#define OP  VFFT_OUTOFPLACE
#define DEF VFFT_ORDER_DEFAULT
#define SCR VFFT_ORDER_SCRAMBLED
#define NAT VFFT_ORDER_NATURAL

static const cell_t CELLS[] = {
  {"2d.il.oop.c2c.127x100.def", C2C,IL,OP,DEF,2,127,100,1},
  {"2d.il.ip.c2c.127x100.def",  C2C,IL,IP,DEF,2,127,100,1},
  {"2d.il.oop.c2c.127x100.nat", C2C,IL,OP,NAT,2,127,100,1},
  {"2d.il.oop.c2c.127x100.scr", C2C,IL,OP,SCR,2,127,100,1},
  {"2d.il.oop.c2c.100x127.def", C2C,IL,OP,DEF,2,100,127,1},
  {"2d.il.oop.c2c.127x128.def", C2C,IL,OP,DEF,2,127,128,1},
  {"2d.il.oop.c2c.256x256.def", C2C,IL,OP,DEF,2,256,256,1},
  {"2d.il.oop.c2c.15x64.def",   C2C,IL,OP,DEF,2, 15, 64,1},
  {"2d.il.oop.r2c.127x100.def", R2C,IL,OP,DEF,2,127,100,1},
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
    if (i < 0 || i >= NCELLS) { printf("out of range\n"); return 2; }
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = (vfft_transform_t)CELLS[i].t;
    cfg.placement = (vfft_placement_t)CELLS[i].place;
    cfg.layout    = (vfft_layout_t)CELLS[i].lay;
    cfg.order     = CELLS[i].ord;
    cfg.dims      = CELLS[i].dims;
    cfg.n[0] = CELLS[i].n0; cfg.n[1] = CELLS[i].n1;
    cfg.howmany = CELLS[i].K;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = 0;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell %s\n", CELLS[i].tag);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
