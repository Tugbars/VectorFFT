/* harness_dark_probe.c - light up the fingerprint fields the single-threaded,
 * K=1, power-of-two, default-order grid never reaches.
 *
 * Twenty of the thirty-two decision fields read zero across that grid:
 *   ztmt ilrace nat2d natpairs nat2dcyc mtunsafe tcbw tcbsn tcbdn pqw pqmt pqn
 *   il2d.{wc roop rw cmt oddn2 nat blu norowz}
 * A field nobody can make fire is either unreachable or reachable only under a
 * configuration we never test - and those are very different facts. This probe
 * walks the axes the base grid holds fixed (threads, howmany, batch geometry,
 * 2D order, odd/prime N, asymmetric 2D shapes) to find which.
 *
 * Build: VFFT_FINGERPRINT=1 python build.py --src benches/harness_dark_probe.c --vfft --compile
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
#define NAT VFFT_ORDER_NATURAL
#define TC  VFFT_BATCH_TRANSFORM_CONTIGUOUS
#define LM  VFFT_BATCH_LANE_MAJOR

static const cell_t CELLS[] = {
  /* ---- MT: ztmt (cascade), il2d.cmt (2D columns), mtunsafe ---- */
  {"mt8.1d.il.ip.c2c.4096",  C2C,IL,IP,DEF,1,4096,   0,8,0, 1},
  {"mt8.1d.il.oop.c2c.4096", C2C,IL,OP,DEF,1,4096,   0,8,0, 1},
  {"mt8.1d.il.ip.c2c.16384", C2C,IL,IP,DEF,1,16384,  0,8,0, 1},
  {"mt8.2d.il.oop.c2c.1024", C2C,IL,OP,DEF,2,1024,1024,8,0,1},
  {"mt8.2d.il.oop.c2c.512",  C2C,IL,OP,DEF,2, 512, 512,8,0, 1},
  {"mt8.2d.il.oop.r2c.512",  R2C,IL,OP,DEF,2, 512, 512,8,0, 1},
  {"mt8.2d.il.ip.c2c.512",   C2C,IL,IP,DEF,2, 512, 512,8,0, 1},
  /* ---- tcb: transform-contiguous batch (tcbw tcbsn tcbdn) ---- */
  {"tc.1d.il.oop.r2c.1024.K8",R2C,IL,OP,DEF,1,1024,0,0,TC, 8},
  {"tc.1d.il.oop.c2c.1024.K8",C2C,IL,OP,DEF,1,1024,0,0,TC, 8},
  {"tc.mt8.1d.il.oop.r2c.1024.K8",R2C,IL,OP,DEF,1,1024,0,8,TC,8},
  {"lm.1d.sp.ip.c2c.1024.K8", C2C,SP,IP,DEF,1,1024,0,0,LM,  8},
  /* ---- plane queue: 2D howmany>1 (pqw pqmt pqn) ---- */
  {"pq.2d.il.oop.c2c.64.K16", C2C,IL,OP,DEF,2,  64,  64,0,TC,16},
  {"pq.2d.il.oop.c2c.256.K8", C2C,IL,OP,DEF,2, 256, 256,0,TC, 8},
  {"pq.mt8.2d.il.oop.c2c.64.K16",C2C,IL,OP,DEF,2,64, 64,8,TC,16},
  /* ---- 2D natural order (nat2d nat2dcyc il2d.nat) ---- */
  {"nat.2d.il.oop.c2c.256",  C2C,IL,OP,NAT,2, 256, 256,0,0, 1},
  {"nat.2d.il.ip.c2c.256",   C2C,IL,IP,NAT,2, 256, 256,0,0, 1},
  {"nat.2d.sp.oop.c2c.256",  C2C,SP,OP,NAT,2, 256, 256,0,0, 1},
  {"nat.2d.il.oop.c2c.64",   C2C,IL,OP,NAT,2,  64,  64,0,0, 1},
  /* ---- odd / prime N (il2d.oddn2, il2d.blu, ilrace) ---- */
  {"odd.1d.il.ip.c2c.255",   C2C,IL,IP,DEF,1, 255,   0,0,0, 1},
  {"odd.1d.il.ip.c2c.127",   C2C,IL,IP,DEF,1, 127,   0,0,0, 1},
  {"odd.2d.il.oop.c2c.127x100",C2C,IL,OP,DEF,2,127,100,0,0, 1},
  {"odd.2d.il.oop.r2c.127x100",R2C,IL,OP,DEF,2,127,100,0,0, 1},
  {"odd.2d.il.oop.c2c.100x127",C2C,IL,OP,DEF,2,100,127,0,0, 1},
  {"odd.1d.il.ip.c2c.3072",  C2C,IL,IP,DEF,1,3072,   0,0,0, 1},
  /* ---- asymmetric 2D: il2d.wc / roop / rw / norowz ---- */
  {"asym.2d.il.oop.c2c.16x4096",C2C,IL,OP,DEF,2,  16,4096,0,0,1},
  {"asym.2d.il.oop.c2c.4096x16",C2C,IL,OP,DEF,2,4096,  16,0,0,1},
  {"asym.2d.il.oop.c2c.8192x64",C2C,IL,OP,DEF,2,8192,  64,0,0,1},
  {"asym.2d.il.oop.c2c.64x256", C2C,IL,OP,DEF,2,  64, 256,0,0,1},
  {"asym.2d.il.oop.r2c.4096x16",R2C,IL,OP,DEF,2,4096,  16,0,0,1},
  {"asym.2d.il.oop.r2c.16x4096",R2C,IL,OP,DEF,2,  16,4096,0,0,1},
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
