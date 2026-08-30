/* _reach_oopk_probe.c - ADVERSARIAL reach probe for the classic-OOP champions race
 * (vfft.c:8895 -> oop_dp.h:136/:146, banked at vfft.c:8909/:8915).
 * One process per cell. Reports accept/refuse + races + fingerprint. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int t, lay, place, ord, dims, n0; size_t K; int geom; } cell_t;
#define C2C VFFT_C2C
#define IL  VFFT_LAYOUT_INTERLEAVED
#define SP  VFFT_LAYOUT_SPLIT
#define OP  VFFT_OUTOFPLACE
#define DEF VFFT_ORDER_DEFAULT
#define SCR VFFT_ORDER_SCRAMBLED
#define NAT VFFT_ORDER_NATURAL
#define GD  VFFT_BATCH_DEFAULT
#define GL  VFFT_BATCH_LANE_MAJOR
#define GT  VFFT_BATCH_TRANSFORM_CONTIGUOUS

static const cell_t CELLS[] = {
  {"sp.oop.c2c.256.K8.def",   C2C,SP,OP,DEF,1,256,8, GD},
  {"sp.oop.c2c.256.K8.nat",   C2C,SP,OP,NAT,1,256,8, GD},
  {"sp.oop.c2c.256.K8.scr",   C2C,SP,OP,SCR,1,256,8, GD},
  {"sp.oop.c2c.256.K4.def",   C2C,SP,OP,DEF,1,256,4, GD},
  {"il.oop.c2c.256.K8.def",   C2C,IL,OP,DEF,1,256,8, GD},
  {"il.oop.c2c.256.K8.lane",  C2C,IL,OP,DEF,1,256,8, GL},
  {"il.oop.c2c.256.K8.tc",    C2C,IL,OP,DEF,1,256,8, GT},
  {"il.oop.c2c.4096.K1.scr",  C2C,IL,OP,SCR,1,4096,1,GD},
  {"sp.oop.c2c.4096.K1.scr",  C2C,SP,OP,SCR,1,4096,1,GD},
  {"sp.oop.c2c.2048.K8.def",  C2C,SP,OP,DEF,1,2048,8,GD},
  {"sp.oop.c2c.64.K8.scr",    C2C,SP,OP,SCR,1,64,8,  GD},
  {"sp.oop.c2c.512.K8.scr",   C2C,SP,OP,SCR,1,512,8, GD},
  {"sp.oop.c2c.1024.K8.scr",  C2C,SP,OP,SCR,1,1024,8,GD},
  {"sp.oop.c2c.1024.K32.scr", C2C,SP,OP,SCR,1,1024,32,GD},
  {"sp.oop.c2c.1024.K32.nat", C2C,SP,OP,NAT,1,1024,32,GD},
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
    cfg.n[0]      = CELLS[i].n0;
    cfg.howmany   = CELLS[i].K;
    cfg.batch_geom= CELLS[i].geom;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = (argc > 3 && !strcmp(argv[3], "--write")) ? 1 : 0;
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
