/* _adv_r2croute_reach.c - does the r2c ROUTE race (rfft vs stride) reach the grid?
 * Build: VFFT_FINGERPRINT=1 python build.py --src benches/_adv_r2croute_reach.c --vfft --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int t, lay, place, rig, dims, n0; size_t K; } cell_t;
#define R2C VFFT_R2C
#define C2R VFFT_C2R
#define IL  VFFT_LAYOUT_INTERLEAVED
#define SP  VFFT_LAYOUT_SPLIT
#define IP  VFFT_INPLACE
#define OP  VFFT_OUTOFPLACE

static const cell_t CELLS[] = {
  {"sp.oop.r2c.1024.K8.pat",  R2C,SP,OP,VFFT_PATIENT,1,1024,8},
  {"sp.oop.r2c.1024.K8.meas", R2C,SP,OP,VFFT_MEASURE, 1,1024,8},
  {"il.oop.r2c.1024.K8.pat",  R2C,IL,OP,VFFT_PATIENT,1,1024,8},
  {"sp.ip.r2c.1024.K8.pat",   R2C,IP==0?SP:SP,IP,VFFT_PATIENT,1,1024,8},
  {"il.ip.r2c.1024.K8.pat",   R2C,IL,IP,VFFT_PATIENT,1,1024,8},
  {"sp.oop.r2c.1024.K1.pat",  R2C,SP,OP,VFFT_PATIENT,1,1024,1},
  {"sp.oop.r2c.1024.K128.pat",R2C,SP,OP,VFFT_PATIENT,1,1024,128},
  {"sp.oop.r2c.1023.K8.pat",  R2C,SP,OP,VFFT_PATIENT,1,1023,8},
  {"sp.oop.c2r.1024.K8.pat",  C2R,SP,OP,VFFT_PATIENT,1,1024,8},
  {"il.oop.c2r.1024.K8.pat",  C2R,IL,OP,VFFT_PATIENT,1,1024,8},
  {"sp.oop.r2c.256.K8.pat",   R2C,SP,OP,VFFT_PATIENT,1, 256,8},
  {"il.oop.r2c.256.K8.pat",   R2C,IL,OP,VFFT_PATIENT,1, 256,8},
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
    cfg.dims      = CELLS[i].dims;
    cfg.n[0] = CELLS[i].n0;
    cfg.howmany = CELLS[i].K;
    cfg.rigor = (vfft_rigor_t)CELLS[i].rig;
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
