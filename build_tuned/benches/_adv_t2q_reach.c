/* _adv_t2q_reach.c - D: IL in-place NATURAL cold-store; E: odd-mid cascade create */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
typedef struct { const char *tag; int t, lay, place, ord, dims, n0; size_t K; } cell_t;
#define C2C VFFT_C2C
#define IL  VFFT_LAYOUT_INTERLEAVED
#define SP  VFFT_LAYOUT_SPLIT
#define IP  VFFT_INPLACE
#define OP  VFFT_OUTOFPLACE
#define DEF VFFT_ORDER_DEFAULT
#define SCR VFFT_ORDER_SCRAMBLED
#define NAT VFFT_ORDER_NATURAL
static const cell_t CELLS[] = {
  {"D.il.ip.nat.4096",  C2C,IL,IP,NAT,1,4096,1},   /* claim D */
  {"E.il.ip.def.6144",  C2C,IL,IP,DEF,1,6144,1},   /* odd-mid 2^11*3 */
  {"E.il.oop.scr.6144", C2C,IL,OP,SCR,1,6144,1},
  {"E.il.ip.def.2560",  C2C,IL,IP,DEF,1,2560,1},   /* odd-mid 2^9*5 */
  {"E.il.oop.scr.2560", C2C,IL,OP,SCR,1,2560,1},
  {"E.il.oop.scr.3072", C2C,IL,OP,SCR,1,3072,1},   /* 2^10*3 */
  {"E.il.oop.scr.5120", C2C,IL,OP,SCR,1,5120,1},
  {"D.il.oop.nat.4096", C2C,IL,OP,NAT,1,4096,1},
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
    cfg.dims      = 1;
    cfg.n[0] = CELLS[i].n0;
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
