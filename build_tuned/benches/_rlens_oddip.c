/* reach lens: does the >=2048 in-place IL scrmode axis fire for odd (2^a*odd) N? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int lay, place, ord, n0; } cell_t;
#define IL  VFFT_LAYOUT_INTERLEAVED
#define SP  VFFT_LAYOUT_SPLIT
#define IP  VFFT_INPLACE
#define OP  VFFT_OUTOFPLACE
#define DEF VFFT_ORDER_DEFAULT
#define SCR VFFT_ORDER_SCRAMBLED
#define NAT VFFT_ORDER_NATURAL

static const cell_t CELLS[] = {
  {"il.ip.def.2048",  IL,IP,DEF,2048},
  {"il.ip.def.4096",  IL,IP,DEF,4096},
  {"il.ip.def.3072",  IL,IP,DEF,3072},
  {"il.ip.scr.3072",  IL,IP,SCR,3072},
  {"il.ip.nat.3072",  IL,IP,NAT,3072},
  {"il.ip.def.5120",  IL,IP,DEF,5120},
  {"il.ip.def.3840",  IL,IP,DEF,3840},
  {"il.ip.def.6144",  IL,IP,DEF,6144},
  {"sp.ip.def.3072",  SP,IP,DEF,3072},
  {"il.oop.def.3072", IL,OP,DEF,3072},
  {"il.ip.def.1536",  IL,IP,DEF,1536},
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
    cfg.transform = VFFT_C2C;
    cfg.placement = (vfft_placement_t)CELLS[i].place;
    cfg.layout    = (vfft_layout_t)CELLS[i].lay;
    cfg.order     = CELLS[i].ord;
    cfg.dims      = 1;
    cfg.n[0] = CELLS[i].n0;
    cfg.howmany = 1;
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
