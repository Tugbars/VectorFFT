/* THE API MATRIX GATE — the machine proof of the SUPPORT MATRIX in
 * include/vfft.h (2026-09-04). Every row is a cell the header makes a
 * serve/refuse claim about; the gate creates it and asserts the claim.
 * A cell that starts serving (or refusing) flips a row here BEFORE the
 * header can drift: change the header and this table together.
 * Refusals must be LOUD: the library prints to stderr; the gate only
 * checks the NULL. Usage: api_matrix_gate.exe --wisdir <scratch dir>. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

typedef struct {
    const char *name;
    int expect_serve;
    int transform, placement, layout, order, dims, n0, n1, n2, n3, K, geom;
} cell_t;

#define IL VFFT_LAYOUT_INTERLEAVED
#define SP VFFT_LAYOUT_SPLIT
#define OOP VFFT_OUTOFPLACE
#define IP VFFT_INPLACE
#define DEF VFFT_ORDER_DEFAULT
#define NAT VFFT_ORDER_NATURAL
#define SCR VFFT_ORDER_SCRAMBLED
#define TC VFFT_BATCH_TRANSFORM_CONTIGUOUS
#define LM VFFT_BATCH_LANE_MAJOR

int main(int argc, char **argv) {
    const char *wd = NULL;
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--wisdir")) wd = argv[i + 1];
    vfft_wisdom *W = wd ? vfft_wisdom_load(wd) : NULL;
    static const cell_t C[] = {
        /* ── 1D C2C: INTERLEAVED serves any N in both placements ── */
        { "1D c2c OOP IL prime 127",              1, VFFT_C2C, OOP, IL, DEF, 1, 127, 0, 0, 0, 1, 0 },
        { "1D c2c OOP IL prime 1021",             1, VFFT_C2C, OOP, IL, DEF, 1, 1021, 0, 0, 0, 1, 0 },
        { "1D c2c OOP IL awkward 129",            1, VFFT_C2C, OOP, IL, DEF, 1, 129, 0, 0, 0, 1, 0 },
        { "1D c2c OOP IL prime 127 NATURAL",      1, VFFT_C2C, OOP, IL, NAT, 1, 127, 0, 0, 0, 1, 0 },
        { "1D c2c IP IL prime 127",               1, VFFT_C2C, IP,  IL, DEF, 1, 127, 0, 0, 0, 1, 0 },
        { "1D c2c IP IL 4096 NATURAL",            1, VFFT_C2C, IP,  IL, NAT, 1, 4096, 0, 0, 0, 1, 0 },
        { "1D c2c IP IL K=4 (default geometry)",  1, VFFT_C2C, IP,  IL, DEF, 1, 256, 0, 0, 0, 4, 0 },
        { "1D c2c OOP IL K=4 TRANSFORM_CONTIG",   1, VFFT_C2C, OOP, IL, DEF, 1, 256, 0, 0, 0, 4, TC },
        { "1D c2c OOP IL K=4 LANE_MAJOR",         0, VFFT_C2C, OOP, IL, DEF, 1, 256, 0, 0, 0, 4, LM },
        /* ── 1D C2C: SPLIT — prime in place only, no TC geometry ── */
        { "1D c2c IP SPLIT prime 127",            1, VFFT_C2C, IP,  SP, DEF, 1, 127, 0, 0, 0, 1, 0 },
        { "1D c2c OOP SPLIT prime 127",           0, VFFT_C2C, OOP, SP, DEF, 1, 127, 0, 0, 0, 1, 0 },
        { "1D c2c OOP SPLIT K=4 TRANSFORM_CONTIG",0, VFFT_C2C, OOP, SP, DEF, 1, 256, 0, 0, 0, 4, TC },
        /* ── 1D real: odd/prime in both layouts OOP; in place IL any N ── */
        { "1D r2c OOP IL prime 101",              1, VFFT_R2C, OOP, IL, DEF, 1, 101, 0, 0, 0, 1, 0 },
        { "1D r2c OOP SPLIT prime 101",           1, VFFT_R2C, OOP, SP, DEF, 1, 101, 0, 0, 0, 1, 0 },
        { "1D c2r OOP IL odd 63",                 1, VFFT_C2R, OOP, IL, DEF, 1, 63, 0, 0, 0, 1, 0 },
        { "1D c2r OOP SPLIT odd 63",              1, VFFT_C2R, OOP, SP, DEF, 1, 63, 0, 0, 0, 1, 0 },
        { "1D c2r OOP IL odd 63 K=4 TC",          1, VFFT_C2R, OOP, IL, DEF, 1, 63, 0, 0, 0, 4, TC },
        { "1D r2c IP IL odd 63",                  1, VFFT_R2C, IP,  IL, DEF, 1, 63, 0, 0, 0, 1, 0 },
        { "1D c2r IP IL odd 63",                  1, VFFT_C2R, IP,  IL, DEF, 1, 63, 0, 0, 0, 1, 0 },
        { "1D r2c IP IL even 64 K=4 TC",          1, VFFT_R2C, IP,  IL, DEF, 1, 64, 0, 0, 0, 4, TC },
        { "1D r2c IP SPLIT 64",                   0, VFFT_R2C, IP,  SP, DEF, 1, 64, 0, 0, 0, 1, 0 },
        { "1D r2c OOP IL 64 order=NATURAL",       0, VFFT_R2C, OOP, IL, NAT, 1, 64, 0, 0, 0, 1, 0 },
        /* ── 2D INTERLEAVED: native tier, both transforms ── */
        { "2D c2c OOP IL 256x64",                 1, VFFT_C2C, OOP, IL, DEF, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2c OOP IL 256x64 NATURAL",         1, VFFT_C2C, OOP, IL, NAT, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2c OOP IL 256x64 SCRAMBLED",       1, VFFT_C2C, OOP, IL, SCR, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2c IP IL 256x64",                  1, VFFT_C2C, IP,  IL, DEF, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2c OOP IL prime 127x64",           1, VFFT_C2C, OOP, IL, DEF, 2, 127, 64, 0, 0, 1, 0 },
        { "2D c2c OOP IL prime 127x64 NATURAL",   1, VFFT_C2C, OOP, IL, NAT, 2, 127, 64, 0, 0, 1, 0 },
        { "2D c2c OOP IL odd 63x63",              1, VFFT_C2C, OOP, IL, DEF, 2, 63, 63, 0, 0, 1, 0 },
        { "2D c2c OOP IL 101x129",                1, VFFT_C2C, OOP, IL, DEF, 2, 101, 129, 0, 0, 1, 0 },
        { "2D c2c OOP IL 64x64 howmany=4",        1, VFFT_C2C, OOP, IL, DEF, 2, 64, 64, 0, 0, 4, 0 },
        { "2D r2c OOP IL 256x64",                 1, VFFT_R2C, OOP, IL, DEF, 2, 256, 64, 0, 0, 1, 0 },
        { "2D r2c OOP IL 256x64 NATURAL",         1, VFFT_R2C, OOP, IL, NAT, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2r OOP IL 256x64 NATURAL",         1, VFFT_C2R, OOP, IL, NAT, 2, 256, 64, 0, 0, 1, 0 },
        { "2D r2c OOP IL prime 127x100",          1, VFFT_R2C, OOP, IL, DEF, 2, 127, 100, 0, 0, 1, 0 },
        { "2D r2c OOP IL odd N2 64x63",           1, VFFT_R2C, OOP, IL, DEF, 2, 64, 63, 0, 0, 1, 0 },
        { "2D r2c OOP IL 64x64 howmany=4",        1, VFFT_R2C, OOP, IL, DEF, 2, 64, 64, 0, 0, 4, 0 },
        { "2D r2c IP IL 256x64",                  0, VFFT_R2C, IP,  IL, DEF, 2, 256, 64, 0, 0, 1, 0 },
        /* ── 2D SPLIT ── */
        { "2D c2c OOP SPLIT 256x64",              1, VFFT_C2C, OOP, SP, DEF, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2c OOP SPLIT 256x64 NATURAL",      1, VFFT_C2C, OOP, SP, NAT, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2c OOP SPLIT prime 127x64",        1, VFFT_C2C, OOP, SP, DEF, 2, 127, 64, 0, 0, 1, 0 },
        { "2D c2c OOP SPLIT 64x64 howmany=4",     0, VFFT_C2C, OOP, SP, DEF, 2, 64, 64, 0, 0, 4, 0 },
        { "2D r2c OOP SPLIT 256x64",              1, VFFT_R2C, OOP, SP, DEF, 2, 256, 64, 0, 0, 1, 0 },
        { "2D r2c OOP SPLIT prime 127x100",       0, VFFT_R2C, OOP, SP, DEF, 2, 127, 100, 0, 0, 1, 0 },
        /* ── 3D / 4D ── */
        { "3D c2c OOP IL 16^3",                   0, VFFT_C2C, OOP, IL, DEF, 3, 16, 16, 16, 0, 1, 0 },
        { "3D c2c IP IL 16^3",                    0, VFFT_C2C, IP,  IL, DEF, 3, 16, 16, 16, 0, 1, 0 },
        { "3D r2c OOP IL 16^3",                   0, VFFT_R2C, OOP, IL, DEF, 3, 16, 16, 16, 0, 1, 0 },
        { "3D c2c OOP SPLIT 16^3",                1, VFFT_C2C, OOP, SP, DEF, 3, 16, 16, 16, 0, 1, 0 },
        { "3D c2c OOP SPLIT 16^3 NATURAL",        0, VFFT_C2C, OOP, SP, NAT, 3, 16, 16, 16, 0, 1, 0 },
        { "3D c2c OOP SPLIT 16^3 howmany=2",      0, VFFT_C2C, OOP, SP, DEF, 3, 16, 16, 16, 0, 2, 0 },
        { "4D c2c OOP IL 8^4",                    0, VFFT_C2C, OOP, IL, DEF, 4, 8, 8, 8, 8, 1, 0 },
        { "4D c2c OOP SPLIT 8^4",                 1, VFFT_C2C, OOP, SP, DEF, 4, 8, 8, 8, 8, 1, 0 },
        /* ── trig ── */
        { "DCT-II OOP SPLIT 64",                  1, VFFT_DCT2, OOP, SP, DEF, 1, 64, 0, 0, 0, 1, 0 },
        { "DCT-II OOP IL 64",                     0, VFFT_DCT2, OOP, IL, DEF, 1, 64, 0, 0, 0, 1, 0 },
        { "DCT-II OOP SPLIT 64 order=NATURAL",    0, VFFT_DCT2, OOP, SP, NAT, 1, 64, 0, 0, 0, 1, 0 },
    };
    const int n = (int)(sizeof C / sizeof C[0]);
    int fail = 0;
    printf("=== API matrix gate: %d cells (the vfft.h SUPPORT MATRIX, asserted) ===\n", n);
    for (int i = 0; i < n; i++) {
        const cell_t *c = &C[i];
        vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
        cfg.transform = c->transform; cfg.placement = c->placement;
        cfg.layout = c->layout; cfg.order = c->order;
        cfg.rigor = VFFT_MEASURE; cfg.dims = c->dims;
        cfg.n[0] = c->n0; cfg.n[1] = c->n1; cfg.n[2] = c->n2; cfg.n[3] = c->n3;
        cfg.howmany = c->K; cfg.batch_geom = c->geom; cfg.nthreads = 1;
        cfg.wisdom = W; cfg.wisdom_write = 0;
        fflush(stdout);
        vfft_plan p = vfft_create(&cfg);
        const int served = (p != NULL);
        if (served != c->expect_serve) {
            fail++;
            printf("  *** FAIL *** %-42s expected %s, got %s\n", c->name,
                   c->expect_serve ? "SERVED" : "REFUSED", served ? "SERVED" : "REFUSED");
        } else
            printf("  ok  %-42s %s\n", c->name, served ? "served" : "refused (loud)");
        fflush(stdout);
        if (p) vfft_destroy(p);
    }
    printf(fail ? "  === *** FAIL *** (%d fail) ===\n" : "  === ALL PASS === (%d cells)\n", fail ? fail : n);
    return fail ? 1 : 0;
}
