/* _reach_oddn2_probe.c - reach check for the "2D IL REAL odd N2 -> c2c row
 * child" axis. Same shape as harness_grid_probe.c: one cell per process. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int t, lay, place, ord, dims, n0, n1; size_t K; } cell_t;
#define C2C VFFT_C2C
#define R2C VFFT_R2C
#define C2R VFFT_C2R
#define IL  VFFT_LAYOUT_INTERLEAVED
#define SP  VFFT_LAYOUT_SPLIT
#define IP  VFFT_INPLACE
#define OP  VFFT_OUTOFPLACE
#define DEF VFFT_ORDER_DEFAULT
#define NAT VFFT_ORDER_NATURAL

static const cell_t CELLS[] = {
  {"2d.il.oop.r2c.64x63",  R2C,IL,OP,DEF,2, 64, 63,1},
  {"2d.il.oop.c2r.64x63",  C2R,IL,OP,DEF,2, 64, 63,1},
  {"2d.il.ip.r2c.64x63",   R2C,IL,IP,DEF,2, 64, 63,1},
  {"2d.sp.oop.r2c.64x63",  R2C,SP,OP,DEF,2, 64, 63,1},
  {"2d.il.oop.r2c.64x64",  R2C,IL,OP,DEF,2, 64, 64,1},
  {"2d.il.oop.r2c.63x64",  R2C,IL,OP,DEF,2, 63, 64,1},
  {"2d.il.oop.r2c.100x127",R2C,IL,OP,DEF,2,100,127,1},
  {"2d.il.oop.c2c.64x63",  C2C,IL,OP,DEF,2, 64, 63,1},
  {"2d.il.oop.r2c.64x63.nat",R2C,IL,OP,NAT,2,64,63,1},
  {"2d.il.oop.c2r.100x127",C2R,IL,OP,DEF,2,100,127,1},
  {"2d.sp.oop.c2r.64x63",  C2R,SP,OP,DEF,2, 64, 63,1},
  {"2d.il.oop.r2c.128x63", R2C,IL,OP,DEF,2,128, 63,1},
  {"2d.il.oop.r2c.64x115", R2C,IL,OP,DEF,2, 64,115,1},
  {"2d.il.oop.r2c.64x3",   R2C,IL,OP,DEF,2, 64,  3,1},
  {"2d.il.oop.r2c.127x63", R2C,IL,OP,DEF,2,127, 63,1},
  {"2d.il.oop.r2c.64x1021",R2C,IL,OP,DEF,2, 64,1021,1},
  {"2d.il.oop.r2c.64x9",   R2C,IL,OP,DEF,2, 64,  9,1},
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
    if (argc < 3 || strcmp(argv[1], "--cell")) {
        printf("usage: %s --cell <0..%d> | --list\n", argv[0], NCELLS - 1);
        return 2;
    }
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
    cfg.wisdom_write = (argc > 3) ? atoi(argv[3]) : 0;
    cfg.nthreads = (argc > 4) ? atoi(argv[4]) : 0;

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
