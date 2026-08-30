/* _adv_zt2q_reach.c - reach probe for the zturn t2q (stf/stf2) create race.
 * One process per cell. Prints races= and the fingerprint. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int t, lay, place, ord, dims, n0, n1; size_t K; } cell_t;
#define C2C VFFT_C2C
#define IL  VFFT_LAYOUT_INTERLEAVED
#define SP  VFFT_LAYOUT_SPLIT
#define IP  VFFT_INPLACE
#define OP  VFFT_OUTOFPLACE
#define DEF VFFT_ORDER_DEFAULT
#define SCR VFFT_ORDER_SCRAMBLED
#define NAT VFFT_ORDER_NATURAL

static const cell_t CELLS[] = {
  {"sp.oop.scr.4096", C2C,SP,OP,SCR,1,4096,0,1},
  {"il.oop.scr.4096", C2C,IL,OP,SCR,1,4096,0,1},
  {"sp.oop.scr.1024", C2C,SP,OP,SCR,1,1024,0,1},
  {"sp.ip.scr.4096",  C2C,SP,IP,SCR,1,4096,0,1},
  {"il.ip.scr.4096",  C2C,IL,IP,SCR,1,4096,0,1},
  {"il.oop.nat.4096", C2C,IL,OP,NAT,1,4096,0,1},
  {"sp.oop.scr.4096.K4", C2C,SP,OP,SCR,1,4096,0,4},
  {"sp.oop.def.4096", C2C,SP,OP,DEF,1,4096,0,1},
  {"il.ip.nat.4096",  C2C,IL,IP,NAT,1,4096,0,1},
  {"il.ip.def.4096",  C2C,IL,IP,DEF,1,4096,0,1},
  {"sp.ip.nat.4096",  C2C,SP,IP,NAT,1,4096,0,1},
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
    if (argc < 3 || strcmp(argv[1], "--cell")) { printf("usage: --cell <i>|--list\n"); return 2; }
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
    cfg.wisdom_write = getenv("PROBE_WRITE") ? 1 : 0;
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
