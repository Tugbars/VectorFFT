/* oddprime_probe.c - odd / prime / awkward-composite cells of the served grid.
 * Mirrors harness_grid_probe.c exactly (one process per cell, fingerprint out). */
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
#define SCR VFFT_ORDER_SCRAMBLED
#define NAT VFFT_ORDER_NATURAL

static const cell_t CELLS[] = {
  /* --- 1D c2c prime (ilprime: Rader vs Bluestein method race) --- */
  {"1d.il.oop.c2c.127",   C2C,IL,OP,DEF,1, 127,0,1},
  {"1d.il.ip.c2c.127",    C2C,IL,IP,DEF,1, 127,0,1},
  {"1d.il.oop.c2c.47",    C2C,IL,OP,DEF,1,  47,0,1},
  {"1d.il.oop.c2c.1009",  C2C,IL,OP,DEF,1,1009,0,1},
  /* --- 1D c2c awkward composite (ilprime Bluestein only) --- */
  {"1d.il.ip.c2c.129",    C2C,IL,IP,DEF,1, 129,0,1},
  {"1d.il.oop.c2c.129",   C2C,IL,OP,DEF,1, 129,0,1},
  {"1d.il.ip.c2c.115",    C2C,IL,IP,DEF,1, 115,0,1},
  {"1d.il.ip.c2c.202",    C2C,IL,IP,DEF,1, 202,0,1},
  /* --- 1D c2c odd composite (il3p chain) --- */
  {"1d.il.oop.c2c.45",    C2C,IL,OP,DEF,1,  45,0,1},
  {"1d.il.ip.c2c.45",     C2C,IL,IP,DEF,1,  45,0,1},
  {"1d.il.ip.c2c.255",    C2C,IL,IP,DEF,1, 255,0,1},
  /* --- 1D c2c SPLIT prime: the bluestein (M,B) calibrator (needs K>=4) --- */
  {"1d.sp.ip.c2c.47.K1",  C2C,SP,IP,DEF,1,  47,0,1},
  {"1d.sp.ip.c2c.47.K4",  C2C,SP,IP,DEF,1,  47,0,4},
  {"1d.sp.ip.c2c.47.K16", C2C,SP,IP,DEF,1,  47,0,16},
  {"1d.sp.ip.c2c.127.K16",C2C,SP,IP,DEF,1, 127,0,16},
  {"1d.sp.ip.c2c.129.K16",C2C,SP,IP,DEF,1, 129,0,16},
  {"1d.sp.oop.c2c.47",    C2C,SP,OP,DEF,1,  47,0,1},
  /* --- 1D odd CASCADE (N = 2^a * odd, N>=2048, N%16==0) --- */
  {"1d.il.oop.c2c.3072.scr",C2C,IL,OP,SCR,1,3072,0,1},
  {"1d.il.ip.c2c.3072",   C2C,IL,IP,DEF,1,3072,0,1},
  {"1d.il.oop.c2c.3072",  C2C,IL,OP,DEF,1,3072,0,1},
  {"1d.il.oop.c2c.5120.scr",C2C,IL,OP,SCR,1,5120,0,1},
  {"1d.sp.oop.c2c.3072.scr",C2C,SP,OP,SCR,1,3072,0,1},
  /* --- 1D real odd (bridge vs rfft) --- */
  {"1d.il.oop.r2c.255",   R2C,IL,OP,DEF,1, 255,0,1},
  {"1d.il.oop.r2c.127",   R2C,IL,OP,DEF,1, 127,0,1},
  {"1d.il.ip.r2c.255",    R2C,IL,IP,DEF,1, 255,0,1},
  {"1d.il.ip.r2c.127",    R2C,IL,IP,DEF,1, 127,0,1},
  {"1d.il.oop.c2r.255",   C2R,IL,OP,DEF,1, 255,0,1},
  {"1d.il.oop.c2r.127",   C2R,IL,OP,DEF,1, 127,0,1},
  {"1d.il.ip.c2r.255",    C2R,IL,IP,DEF,1, 255,0,1},
  {"1d.sp.oop.r2c.255",   R2C,SP,OP,DEF,1, 255,0,1},
  {"1d.sp.oop.c2r.255",   C2R,SP,OP,DEF,1, 255,0,1},
  /* --- 2D odd/prime N1 --- */
  {"2d.il.oop.c2c.45x64", C2C,IL,OP,DEF,2,  45, 64,1},
  {"2d.il.oop.c2c.127x100",C2C,IL,OP,DEF,2,127,100,1},
  {"2d.il.ip.c2c.127x100",C2C,IL,IP,DEF,2, 127,100,1},
  {"2d.il.oop.c2c.101x129",C2C,IL,OP,DEF,2,101,129,1},
  {"2d.il.oop.c2c.64x63", C2C,IL,OP,DEF,2,  64, 63,1},
  {"2d.il.oop.c2c.64x129",C2C,IL,OP,DEF,2,  64,129,1},
  /* --- 2D real odd/prime --- */
  {"2d.il.oop.r2c.127x100",R2C,IL,OP,DEF,2,127,100,1},
  {"2d.il.oop.r2c.64x63", R2C,IL,OP,DEF,2,  64, 63,1},
  {"2d.il.oop.c2r.64x63", C2R,IL,OP,DEF,2,  64, 63,1},
  {"2d.il.oop.r2c.45x64", R2C,IL,OP,DEF,2,  45, 64,1},
  {"2d.sp.oop.c2c.127x100",C2C,SP,OP,DEF,2,127,100,1},
  {"2d.sp.oop.r2c.64x63", R2C,SP,OP,DEF,2,  64, 63,1},
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
    cfg.wisdom_write = (getenv("PROBE_WRITE") != NULL);

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
