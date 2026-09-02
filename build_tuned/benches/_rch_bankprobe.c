/* harness_grid_probe.c - the served grid, and the RACED AXES that decide each cell.
 *
 * Walks {split,IL} x {in-place,oop} x {1D,2D} x {c2c,r2c,c2r} (plus the N bands
 * that switch engine) and, per cell, reports:
 *   - ACCEPT / refuse            (the front-door contract)
 *   - races=N                    (did THIS create measure, or replay a bank)
 *   - the full plan fingerprint  (which decision fields the winner actually set)
 *
 * Why the fingerprint and not just a timing: the question is which measurement
 * arms decide a winner that then gets SAVED to wisdom. A field that is non-zero
 * names a decision that was made for this cell; a field that is zero across the
 * whole grid names a decision no cell in the grid reaches.
 *
 * ONE PROCESS PER CELL is mandatory - the K=1 pair-order memo and the pool are
 * process-lifetime, so a shared process makes later cells inherit earlier ones.
 *
 * Build: VFFT_FINGERPRINT=1 python build.py --src benches/harness_grid_probe.c --vfft --compile
 * Run  : VFFT_WISDOM_DIR=<seeded scratch> harness_grid_probe.exe --cell <i>
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

typedef struct { const char *tag; int t, lay, place, ord, dims, n0, n1; size_t K; } cell_t;

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

static const cell_t CELLS[] = {
  /* ---- 1D c2c, the three IL N-bands x placement (il_codelet_design S1) ---- */
  {"1d.il.ip.c2c.64",    C2C,IL,IP,DEF,1,  64,0,1}, {"1d.il.oop.c2c.64",   C2C,IL,OP,DEF,1,  64,0,1},
  {"1d.il.ip.c2c.256",   C2C,IL,IP,DEF,1, 256,0,1}, {"1d.il.oop.c2c.256",  C2C,IL,OP,DEF,1, 256,0,1},
  {"1d.il.ip.c2c.1024",  C2C,IL,IP,DEF,1,1024,0,1}, {"1d.il.oop.c2c.1024", C2C,IL,OP,DEF,1,1024,0,1},
  {"1d.il.ip.c2c.4096",  C2C,IL,IP,DEF,1,4096,0,1}, {"1d.il.oop.c2c.4096", C2C,IL,OP,DEF,1,4096,0,1},
  /* ---- 1D c2c split x placement x K (split names 3 engines: see S4) ---- */
  {"1d.sp.ip.c2c.256",   C2C,SP,IP,DEF,1, 256,0,1}, {"1d.sp.oop.c2c.256",  C2C,SP,OP,DEF,1, 256,0,1},
  {"1d.sp.ip.c2c.4096",  C2C,SP,IP,DEF,1,4096,0,1}, {"1d.sp.oop.c2c.4096", C2C,SP,OP,DEF,1,4096,0,1},
  {"1d.sp.ip.c2c.256.K32",C2C,SP,IP,DEF,1,256,0,32},{"1d.sp.ip.c2c.256.K4",C2C,SP,IP,DEF,1,256,0,4},
  /* ---- order axis (natural/scrambled are separate banked verdicts) ---- */
  {"1d.sp.ip.c2c.256.nat",C2C,SP,IP,NAT,1,256,0,1}, {"1d.sp.ip.c2c.256.scr",C2C,SP,IP,SCR,1,256,0,1},
  {"1d.il.ip.c2c.4096.nat",C2C,IL,IP,NAT,1,4096,0,1},{"1d.il.oop.c2c.4096.nat",C2C,IL,OP,NAT,1,4096,0,1},
  /* ---- 1D real x layout x placement ---- */
  {"1d.il.ip.r2c.1024",  R2C,IL,IP,DEF,1,1024,0,1}, {"1d.il.oop.r2c.1024", R2C,IL,OP,DEF,1,1024,0,1},
  {"1d.sp.ip.r2c.1024",  R2C,SP,IP,DEF,1,1024,0,1}, {"1d.sp.oop.r2c.1024", R2C,SP,OP,DEF,1,1024,0,1},
  {"1d.il.ip.c2r.1024",  C2R,IL,IP,DEF,1,1024,0,1}, {"1d.il.oop.c2r.1024", C2R,IL,OP,DEF,1,1024,0,1},
  {"1d.sp.ip.c2r.1024",  C2R,SP,IP,DEF,1,1024,0,1}, {"1d.sp.oop.c2r.1024", C2R,SP,OP,DEF,1,1024,0,1},
  /* ---- 2D x layout x placement x transform ---- */
  {"2d.il.ip.c2c.256",   C2C,IL,IP,DEF,2,256,256,1},{"2d.il.oop.c2c.256",  C2C,IL,OP,DEF,2,256,256,1},
  {"2d.sp.ip.c2c.256",   C2C,SP,IP,DEF,2,256,256,1},{"2d.sp.oop.c2c.256",  C2C,SP,OP,DEF,2,256,256,1},
  {"2d.il.ip.r2c.256",   R2C,IL,IP,DEF,2,256,256,1},{"2d.il.oop.r2c.256",  R2C,IL,OP,DEF,2,256,256,1},
  {"2d.sp.ip.r2c.256",   R2C,SP,IP,DEF,2,256,256,1},{"2d.sp.oop.r2c.256",  R2C,SP,OP,DEF,2,256,256,1},
  {"2d.il.ip.c2r.256",   C2R,IL,IP,DEF,2,256,256,1},{"2d.il.oop.c2r.256",  C2R,IL,OP,DEF,2,256,256,1},
  {"2d.sp.ip.c2r.256",   C2R,SP,IP,DEF,2,256,256,1},{"2d.sp.oop.c2r.256",  C2R,SP,OP,DEF,2,256,256,1},
  {"2d.il.oop.c2c.64",   C2C,IL,OP,DEF,2, 64, 64,1},{"2d.il.oop.r2c.64",   R2C,IL,OP,DEF,2, 64, 64,1},
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
    if (argc < 3 || strcmp(argv[1], "--cell")) {
        printf("usage: %s --cell <0..%d> | --list\n", argv[0], NCELLS - 1);
        return 2;
    }
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
