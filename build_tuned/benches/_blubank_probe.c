/* _blureach_probe.c - does 2D IL c2c with prime N1 reach the blu fallback? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int lay, place, ord, n0, n1; } cell_t;
#define IL VFFT_LAYOUT_INTERLEAVED
#define SP VFFT_LAYOUT_SPLIT
#define IP VFFT_INPLACE
#define OP VFFT_OUTOFPLACE
#define DEF VFFT_ORDER_DEFAULT
#define SCR VFFT_ORDER_SCRAMBLED
#define NAT VFFT_ORDER_NATURAL

static const cell_t CELLS[] = {
  {"2d.il.oop.c2c.127x100.def", IL,OP,DEF,127,100},
  {"2d.il.ip.c2c.127x100.def",  IL,IP,DEF,127,100},
  {"2d.il.oop.c2c.127x100.nat", IL,OP,NAT,127,100},
  {"2d.il.oop.c2c.127x100.scr", IL,OP,SCR,127,100},
  {"2d.il.oop.c2c.100x127.def", IL,OP,DEF,100,127},
  {"2d.il.oop.c2c.101x129.def", IL,OP,DEF,101,129},
  {"2d.il.oop.c2c.256x256.def", IL,OP,DEF,256,256},
  {"2d.sp.oop.c2c.127x100.def", SP,OP,DEF,127,100},
};
#define NCELLS ((int)(sizeof CELLS / sizeof CELLS[0]))

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p; int i;
    if (argc < 3 || strcmp(argv[1], "--cell")) { printf("usage --cell n (0..%d)\n", NCELLS-1); return 2; }
    i = atoi(argv[2]);
    if (i < 0 || i >= NCELLS) return 2;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = (vfft_placement_t)CELLS[i].place;
    cfg.layout    = (vfft_layout_t)CELLS[i].lay;
    cfg.order     = CELLS[i].ord;
    cfg.dims      = 2;
    cfg.n[0] = CELLS[i].n0; cfg.n[1] = CELLS[i].n1;
    cfg.howmany = 1;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = 1;
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
