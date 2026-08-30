/* _reach_zroute_probe.c - reach check for the cascade ENGINE (zroute) axis.
 * Door (a): OOP order=SCRAMBLED, EITHER layout, N>=2048.  Does a SPLIT-layout
 * OOP scrambled caller attach the cascade (have-zturn) the way an IL one does? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int t, lay, place, ord, dims, n0, n1; size_t K; } cell_t;
#define C2C VFFT_C2C
#define IL  VFFT_LAYOUT_INTERLEAVED
#define SP  VFFT_LAYOUT_SPLIT
#define OP  VFFT_OUTOFPLACE
#define IP  VFFT_INPLACE
#define SCR VFFT_ORDER_SCRAMBLED
#define DEF VFFT_ORDER_DEFAULT

static const cell_t CELLS[] = {
  {"1d.il.oop.c2c.4096.scr", C2C,IL,OP,SCR,1,4096,0,1},
  {"1d.sp.oop.c2c.4096.scr", C2C,SP,OP,SCR,1,4096,0,1},
  {"1d.il.oop.c2c.1024.scr", C2C,IL,OP,SCR,1,1024,0,1},
  {"1d.sp.oop.c2c.1024.scr", C2C,SP,OP,SCR,1,1024,0,1},
  {"1d.sp.ip.c2c.4096.scr",  C2C,SP,IP,SCR,1,4096,0,1},
  {"1d.il.ip.c2c.4096.scr",  C2C,IL,IP,SCR,1,4096,0,1},
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
    if (argc < 3 || strcmp(argv[1], "--cell")) { printf("usage: --cell <0..%d>\n", NCELLS-1); return 2; }
    i = atoi(argv[2]);
    if (i < 0 || i >= NCELLS) return 2;
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
