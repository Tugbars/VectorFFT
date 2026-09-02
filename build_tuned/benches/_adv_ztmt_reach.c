/* adversarial reach probe for the zt_mt (cascade MT engage) axis */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int t, lay, place, ord, n; int T; size_t K; } cell_t;

static const cell_t CELLS[] = {
 {"A oop il scr 4096 T8",  VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_SCRAMBLED,4096,8,1},
 {"B oop il def 4096 T8",  VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_DEFAULT,  4096,8,1},
 {"C oop il nat 4096 T8",  VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_NATURAL,  4096,8,1},
 {"D oop sp  scr 4096 T8", VFFT_C2C,VFFT_LAYOUT_SPLIT,       VFFT_OUTOFPLACE,VFFT_ORDER_SCRAMBLED,4096,8,1},
 {"E ip  il  scr 4096 T8", VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_INPLACE,   VFFT_ORDER_SCRAMBLED,4096,8,1},
 {"F oop il scr 1024 T8",  VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_SCRAMBLED,1024,8,1},
 {"G oop il scr 4096 T1",  VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_SCRAMBLED,4096,1,1},
 {"H oop il scr 65536 T8", VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_SCRAMBLED,65536,8,1},
 {"I oop il scr 4096 T8 K2",VFFT_C2C,VFFT_LAYOUT_INTERLEAVED,VFFT_OUTOFPLACE,VFFT_ORDER_SCRAMBLED,4096,8,2},
};
#define NCELLS ((int)(sizeof CELLS / sizeof CELLS[0]))

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p; int i;
    if (argc < 3 || strcmp(argv[1], "--cell")) { printf("usage --cell n (0..%d)\n", NCELLS-1); return 2; }
    i = atoi(argv[2]);
    if (i < 0 || i >= NCELLS) return 2;
    vfft_set_num_threads(CELLS[i].T);
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = (vfft_transform_t)CELLS[i].t;
    cfg.placement = (vfft_placement_t)CELLS[i].place;
    cfg.layout    = (vfft_layout_t)CELLS[i].lay;
    cfg.order     = CELLS[i].ord;
    cfg.dims      = 1;
    cfg.n[0]      = CELLS[i].n;
    cfg.howmany   = CELLS[i].K;
    cfg.rigor     = VFFT_MEASURE;
    cfg.wisdom_write = 0;
    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell %s pool=%d\n", CELLS[i].tag, vfft_get_num_threads());
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld\n", c[5]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
