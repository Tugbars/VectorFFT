/* _ilmono_probe.c — IL mono/Bailey tier: which axes race at create.
 * scratch probe (temporary; not a tracked harness). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"
#include <time.h>
#include <windows.h>
static double nowms(void){LARGE_INTEGER f,c;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&c);return 1e3*(double)c.QuadPart/(double)f.QuadPart;}

typedef struct { const char *tag; int t, lay, place, ord, n0; size_t K; } cell_t;
static const cell_t CELLS[] = {
  {"il.ip.c2c.64.def",   VFFT_C2C, VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE,    VFFT_ORDER_DEFAULT,  64,1},
  {"il.ip.c2c.128.def",  VFFT_C2C, VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE,    VFFT_ORDER_DEFAULT, 128,1},
  {"il.ip.c2c.256.def",  VFFT_C2C, VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE,    VFFT_ORDER_DEFAULT, 256,1},
  {"il.ip.c2c.256.nat",  VFFT_C2C, VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE,    VFFT_ORDER_NATURAL, 256,1},
  {"il.ip.c2c.64.nat",   VFFT_C2C, VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE,    VFFT_ORDER_NATURAL,  64,1},
  {"il.ip.c2c.512.nat",  VFFT_C2C, VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE,    VFFT_ORDER_NATURAL, 512,1},
  {"il.oop.c2c.128.def", VFFT_C2C, VFFT_LAYOUT_INTERLEAVED, VFFT_OUTOFPLACE, VFFT_ORDER_DEFAULT, 128,1},
  {"il.oop.c2c.512.def", VFFT_C2C, VFFT_LAYOUT_INTERLEAVED, VFFT_OUTOFPLACE, VFFT_ORDER_DEFAULT, 512,1},
  {"il.ip.c2c.32.def",   VFFT_C2C, VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE,    VFFT_ORDER_DEFAULT,  32,1},
  {"il.ip.c2c.48.def",   VFFT_C2C, VFFT_LAYOUT_INTERLEAVED, VFFT_INPLACE,    VFFT_ORDER_DEFAULT,  48,1},
};
#define NCELLS ((int)(sizeof CELLS / sizeof CELLS[0]))

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS]; vfft_config_t cfg; vfft_plan p; int i, rep, nrep = 2;
    if (argc < 3 || strcmp(argv[1], "--cell")) {
        for (i = 0; i < NCELLS; i++) printf("%2d %s\n", i, CELLS[i].tag);
        return 2;
    }
    i = atoi(argv[2]);
    if (i < 0 || i >= NCELLS) return 2;
    if (argc > 3) nrep = atoi(argv[3]);
    for (rep = 0; rep < nrep; rep++) {
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
        double _t0 = nowms();
        p = vfft_create(&cfg);
        double _dt = nowms() - _t0;
        vfft__fp_counters(c);
        printf("@@cell %s rep=%d create_ms=%.2f\n", CELLS[i].tag, rep, _dt);
        if (!p) { printf("@@status refuse races=%ld\n", c[5]); continue; }
        printf("@@status accept races=%ld\n", c[5]);
        vfft__fingerprint(p, buf, sizeof buf);
        fputs(buf, stdout);
        vfft_destroy(p);
    }
    return 0;
}
