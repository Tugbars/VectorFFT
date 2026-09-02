/* _adv_oddr_reach.c - reach check for the smooth-odd r2c rfft-vs-bridge race. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int t, lay, place, dims, n0; size_t K; } cell_t;
#define R2C VFFT_R2C
#define C2R VFFT_C2R
#define IL  VFFT_LAYOUT_INTERLEAVED
#define SP  VFFT_LAYOUT_SPLIT
#define IP  VFFT_INPLACE
#define OP  VFFT_OUTOFPLACE

static const cell_t CELLS[] = {
  {"1d.il.oop.r2c.255",   R2C,IL,OP,1,255,1},   /* 3*5*17 smooth odd */
  {"1d.il.oop.r2c.4095",  R2C,IL,OP,1,4095,1},  /* 3^2*5*7*13 smooth odd */
  {"1d.il.oop.r2c.2187",  R2C,IL,OP,1,2187,1},  /* 3^7 smooth odd */
  {"1d.il.oop.r2c.9",     R2C,IL,OP,1,9,1},
  {"1d.il.oop.r2c.253",   R2C,IL,OP,1,253,1},   /* 11*23 NON-smooth odd */
  {"1d.il.oop.r2c.127",   R2C,IL,OP,1,127,1},   /* prime, NON-smooth */
  {"1d.il.ip.r2c.255",    R2C,IL,IP,1,255,1},
  {"1d.sp.oop.r2c.255",   R2C,SP,OP,1,255,1},
  {"1d.sp.ip.r2c.255",    R2C,SP,IP,1,255,1},
  {"1d.il.oop.r2c.255.K4",R2C,IL,OP,1,255,4},
  {"1d.il.oop.c2r.255",   C2R,IL,OP,1,255,1},
  {"1d.il.oop.r2c.1024",  R2C,IL,OP,1,1024,1},
};
#define NCELLS ((int)(sizeof CELLS / sizeof CELLS[0]))

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p; int i;
    if (argc < 3 || strcmp(argv[1], "--cell")) {
        for (i = 0; i < NCELLS; i++) printf("%2d %s\n", i, CELLS[i].tag);
        return 0;
    }
    i = atoi(argv[2]);
    if (i < 0 || i >= NCELLS) return 2;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = CELLS[i].t;
    cfg.layout = CELLS[i].lay;
    cfg.placement = CELLS[i].place;
    cfg.order = VFFT_ORDER_DEFAULT;
    cfg.dims = CELLS[i].dims;
    cfg.n[0] = CELLS[i].n0;
    cfg.howmany = CELLS[i].K;
    cfg.rigor = VFFT_PATIENT;
    cfg.nthreads = 1;
    fprintf(stderr, "[cell] %s\n", CELLS[i].tag);
    p = vfft_create(&cfg);
    if (!p) { printf("%s REFUSED\n", CELLS[i].tag); return 0; }
    vfft__fp_counters(c);
    vfft__fingerprint(p, buf, sizeof buf);
    printf("%s races=%ld\n%s\n", CELLS[i].tag, c[5], buf);
    vfft_destroy(p);
    return 0;
}
