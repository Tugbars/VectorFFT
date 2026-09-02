/* harness_dark2_probe.c - the last six fields.
 *
 * After the base grid and the dark probe, 26 of 32 decision fields have been
 * observed non-zero. These six had not:
 *   ztmt        cascade MT engage - memory says the race banks SERIAL at N<=8192
 *               and the wins are at 16384..262144, zturn with ord=scr
 *   il2d.oddn2  "odd N2" - needs REAL (r2c/c2r) with an ODD second dimension
 *   il2d.roop   row out-of-place
 *   il2d.wc     a 2D width axis distinct from wl
 *   ilrace      set when the IL race actually runs rather than replays
 *   mtunsafe    the plan is marked unsafe to thread
 *
 * A field that stays dark after a targeted attempt is a real finding: either
 * unreachable through the public API, or reachable only via a path we have not
 * identified. Both are worth stating precisely.
 *
 * Build: VFFT_FINGERPRINT=1 python build.py --src benches/harness_dark2_probe.c --vfft --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct {
    const char *tag;
    int t, lay, place, ord, dims, n0, n1, nthr, bgeom;
    size_t K;
} cell_t;

#define C2C VFFT_C2C
#define R2C VFFT_R2C
#define C2R VFFT_C2R
#define IL  VFFT_LAYOUT_INTERLEAVED
#define SP  VFFT_LAYOUT_SPLIT
#define IP  VFFT_INPLACE
#define OP  VFFT_OUTOFPLACE
#define DEF VFFT_ORDER_DEFAULT
#define SCR VFFT_ORDER_SCRAMBLED
#define NAT VFFT_ORDER_NATURAL
#define TC  VFFT_BATCH_TRANSFORM_CONTIGUOUS

static const cell_t CELLS[] = {
  /* ---- ztmt: cascade MT. big N, both orders, both placements ---- */
  {"zt.mt8.1d.il.ip.c2c.65536.scr",  C2C,IL,IP,SCR,1, 65536,0,8,0,1},
  {"zt.mt8.1d.il.ip.c2c.65536.def",  C2C,IL,IP,DEF,1, 65536,0,8,0,1},
  {"zt.mt8.1d.il.oop.c2c.65536.scr", C2C,IL,OP,SCR,1, 65536,0,8,0,1},
  {"zt.mt8.1d.il.ip.c2c.262144.scr", C2C,IL,IP,SCR,1,262144,0,8,0,1},
  {"zt.mt8.1d.il.ip.c2c.16384.scr",  C2C,IL,IP,SCR,1, 16384,0,8,0,1},
  {"zt.mt8.1d.sp.ip.c2c.65536.scr",  C2C,SP,IP,SCR,1, 65536,0,8,0,1},
  /* ---- il2d.oddn2: REAL 2D with an ODD SECOND dim ---- */
  {"o2.2d.il.oop.r2c.128x127", R2C,IL,OP,DEF,2,128,127,0,0,1},
  {"o2.2d.il.oop.r2c.100x127", R2C,IL,OP,DEF,2,100,127,0,0,1},
  {"o2.2d.il.oop.r2c.64x101",  R2C,IL,OP,DEF,2, 64,101,0,0,1},
  {"o2.2d.il.oop.c2r.128x127", C2R,IL,OP,DEF,2,128,127,0,0,1},
  {"o2.2d.il.oop.r2c.256x255", R2C,IL,OP,DEF,2,256,255,0,0,1},
  {"o2.2d.il.oop.r2c.128x15",  R2C,IL,OP,DEF,2,128, 15,0,0,1},
  /* ---- il2d.roop / il2d.wc: more 2D shapes, both placements ---- */
  {"w.2d.il.ip.c2c.1024x1024", C2C,IL,IP,DEF,2,1024,1024,0,0,1},
  {"w.2d.il.oop.c2c.32x1024",  C2C,IL,OP,DEF,2,  32,1024,0,0,1},
  {"w.2d.il.oop.c2c.1024x32",  C2C,IL,OP,DEF,2,1024,  32,0,0,1},
  {"w.2d.il.oop.c2c.16384x64", C2C,IL,OP,DEF,2,16384, 64,0,0,1},
  {"w.2d.il.oop.r2c.1024x1024",R2C,IL,OP,DEF,2,1024,1024,0,0,1},
  {"w.2d.il.oop.r2c.64x256",   R2C,IL,OP,DEF,2,  64, 256,0,0,1},
  {"w.2d.il.oop.r2c.256x64",   R2C,IL,OP,DEF,2, 256,  64,0,0,1},
  {"w.mt8.2d.il.oop.r2c.1024x1024",R2C,IL,OP,DEF,2,1024,1024,8,0,1},
  /* ---- mtunsafe / ilrace: MT on shapes that may be refused threading ---- */
  {"u.mt8.1d.sp.ip.c2c.256.K32",C2C,SP,IP,DEF,1,256,0,8,0,32},
  {"u.mt8.1d.il.ip.c2c.256",    C2C,IL,IP,DEF,1,256,0,8,0, 1},
  {"u.mt8.1d.il.ip.c2c.64",     C2C,IL,IP,DEF,1, 64,0,8,0, 1},
  {"u.mt8.2d.il.oop.c2c.64",    C2C,IL,OP,DEF,2, 64,64,8,0,1},
  {"u.mt8.1d.il.ip.c2c.255",    C2C,IL,IP,DEF,1,255,0,8,0, 1},
  {"u.mt8.1d.il.oop.c2c.4096.nat",C2C,IL,OP,NAT,1,4096,0,8,0,1},
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
    if (argc < 3 || strcmp(argv[1], "--cell")) return 2;
    i = atoi(argv[2]);
    if (i < 0 || i >= NCELLS) return 2;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = (vfft_transform_t)CELLS[i].t;
    cfg.placement = (vfft_placement_t)CELLS[i].place;
    cfg.layout    = (vfft_layout_t)CELLS[i].lay;
    cfg.order     = CELLS[i].ord;
    cfg.dims      = CELLS[i].dims;
    cfg.n[0] = CELLS[i].n0; cfg.n[1] = CELLS[i].n1;
    cfg.howmany   = CELLS[i].K;
    cfg.nthreads  = CELLS[i].nthr;
    cfg.batch_geom = CELLS[i].bgeom;
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
