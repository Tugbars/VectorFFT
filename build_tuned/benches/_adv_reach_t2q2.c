/* _adv_reach_t2q2.c — REACH probe for the zturn t2q create race (_calibrate_zturn_t2q).
 * Question: which {layout,placement,order} cells at N>=2048 actually reach it,
 * and which of them BANK the kind-4 (t=c2c ord=scr place=oop) row. */
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
  {"A.sp.oop.scr.4096", SP,OP,SCR,4096},   /* the claim's split-banks cell */
  {"B.il.oop.scr.4096", IL,OP,SCR,4096},   /* control: IL oop scrambled     */
  {"C.il.oop.nat.4096", IL,OP,NAT,4096},   /* claim cites 8635 as a racer   */
  {"D.il.ip.nat.4096",  IL,IP,NAT,4096},   /* 7453 in-place natural         */
  {"E.il.ip.def.4096",  IL,IP,DEF,4096},   /* 7957/7996/8005 in-place       */
  {"F.sp.ip.scr.4096",  SP,IP,SCR,4096},   /* layout-gated out at 7957      */
  {"G.sp.oop.scr.1024", SP,OP,SCR,1024},   /* below the 2048 tier boundary  */
  {"H.sp.oop.def.4096", SP,OP,DEF,4096},   /* order=DEFAULT oop             */
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
    cfg.n[0]      = CELLS[i].n0;
    cfg.howmany   = 1;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = (argc > 3) ? atoi(argv[3]) : 0;
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
