/* THE API MATRIX, empirically: every cell the vfft.h support matrix makes
 * a claim about, created for real. Prints SERVED / REFUSED per cell (the
 * refusal text goes to stderr from the library itself). The header is
 * rewritten from THIS table, and api_matrix_gate.c asserts it. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

typedef struct {
    const char *name;
    int transform, placement, layout, order, dims, n0, n1, n2, n3, K, geom, nthreads;
} cell_t;

int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const cell_t C[] = {
        /* name                              tr        pl              lay                      ord                 d  n0   n1  n2 n3  K  geom nt */
        { "1D c2c OOP IL prime 127",         VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 127, 0, 0, 0, 1, 0, 1 },
        { "1D c2c OOP IL prime 1021",        VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 1021, 0, 0, 0, 1, 0, 1 },
        { "1D c2c OOP IL awkward 129",       VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 129, 0, 0, 0, 1, 0, 1 },
        { "1D c2c OOP SPLIT prime 127",      VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 1, 127, 0, 0, 0, 1, 0, 1 },
        { "1D c2c IP IL prime 127",          VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 127, 0, 0, 0, 1, 0, 1 },
        { "1D c2c IP SPLIT prime 127",       VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 1, 127, 0, 0, 0, 1, 0, 1 },
        { "1D c2c OOP IL prime 127 NATURAL", VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_NATURAL, 1, 127, 0, 0, 0, 1, 0, 1 },
        { "1D c2c IP IL 4096 NATURAL",       VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_NATURAL, 1, 4096, 0, 0, 0, 1, 0, 1 },
        { "1D c2c OOP IL K=4 TC",            VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 256, 0, 0, 0, 4, VFFT_BATCH_TRANSFORM_CONTIGUOUS, 1 },
        { "1D c2c OOP IL K=4 lane-major",    VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 256, 0, 0, 0, 4, VFFT_BATCH_LANE_MAJOR, 1 },
        { "1D c2c IP IL K=4 default geom",   VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 256, 0, 0, 0, 4, 0, 1 },
        { "1D c2c OOP SPLIT K=4 TC",         VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 1, 256, 0, 0, 0, 4, VFFT_BATCH_TRANSFORM_CONTIGUOUS, 1 },
        { "1D r2c IP IL odd 63",             VFFT_R2C, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 63, 0, 0, 0, 1, 0, 1 },
        { "1D c2r IP IL odd 63",             VFFT_C2R, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 63, 0, 0, 0, 1, 0, 1 },
        { "1D r2c IP IL even 64 K=4 TC",     VFFT_R2C, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 64, 0, 0, 0, 4, VFFT_BATCH_TRANSFORM_CONTIGUOUS, 1 },
        { "1D r2c IP SPLIT 64",              VFFT_R2C, VFFT_INPLACE,    VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 1, 64, 0, 0, 0, 1, 0, 1 },
        { "1D r2c OOP IL prime 101",         VFFT_R2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 101, 0, 0, 0, 1, 0, 1 },
        { "1D r2c OOP SPLIT prime 101",      VFFT_R2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 1, 101, 0, 0, 0, 1, 0, 1 },
        { "1D c2r OOP IL odd 63",            VFFT_C2R, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 63, 0, 0, 0, 1, 0, 1 },
        { "1D c2r OOP SPLIT odd 63",         VFFT_C2R, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 1, 63, 0, 0, 0, 1, 0, 1 },
        { "1D c2r OOP IL odd 63 K=4 TC",     VFFT_C2R, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 63, 0, 0, 0, 4, VFFT_BATCH_TRANSFORM_CONTIGUOUS, 1 },
        { "1D r2c OOP IL 64 NATURAL (order on real)", VFFT_R2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_NATURAL, 1, 64, 0, 0, 0, 1, 0, 1 },
        { "2D c2c OOP IL 256x64",            VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 2, 256, 64, 0, 0, 1, 0, 1 },
        { "2D c2c OOP IL 256x64 NATURAL",    VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_NATURAL, 2, 256, 64, 0, 0, 1, 0, 1 },
        { "2D c2c OOP IL 256x64 SCRAMBLED",  VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_SCRAMBLED, 2, 256, 64, 0, 0, 1, 0, 1 },
        { "2D c2c IP IL 256x64",             VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 2, 256, 64, 0, 0, 1, 0, 1 },
        { "2D c2c OOP IL prime 127x64",      VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 2, 127, 64, 0, 0, 1, 0, 1 },
        { "2D c2c OOP IL prime 127x64 NATURAL", VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_NATURAL, 2, 127, 64, 0, 0, 1, 0, 1 },
        { "2D c2c OOP IL odd 63x63",         VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 2, 63, 63, 0, 0, 1, 0, 1 },
        { "2D c2c OOP IL 101x129",           VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 2, 101, 129, 0, 0, 1, 0, 1 },
        { "2D c2c OOP SPLIT 256x64",         VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 2, 256, 64, 0, 0, 1, 0, 1 },
        { "2D c2c OOP SPLIT 256x64 NATURAL", VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_NATURAL, 2, 256, 64, 0, 0, 1, 0, 1 },
        { "2D c2c OOP SPLIT prime 127x64",   VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 2, 127, 64, 0, 0, 1, 0, 1 },
        { "2D c2c OOP IL 64x64 howmany=4",   VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 2, 64, 64, 0, 0, 4, 0, 1 },
        { "2D c2c OOP SPLIT 64x64 howmany=4",VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 2, 64, 64, 0, 0, 4, 0, 1 },
        { "2D r2c OOP IL 256x64",            VFFT_R2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 2, 256, 64, 0, 0, 1, 0, 1 },
        { "2D r2c OOP IL 256x64 NATURAL",    VFFT_R2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_NATURAL, 2, 256, 64, 0, 0, 1, 0, 1 },
        { "2D c2r OOP IL 256x64 NATURAL",    VFFT_C2R, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_NATURAL, 2, 256, 64, 0, 0, 1, 0, 1 },
        { "2D r2c OOP IL prime 127x100",     VFFT_R2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 2, 127, 100, 0, 0, 1, 0, 1 },
        { "2D r2c OOP IL odd N2 64x63",      VFFT_R2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 2, 64, 63, 0, 0, 1, 0, 1 },
        { "2D r2c OOP SPLIT 256x64",         VFFT_R2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 2, 256, 64, 0, 0, 1, 0, 1 },
        { "2D r2c OOP SPLIT prime 127x100",  VFFT_R2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 2, 127, 100, 0, 0, 1, 0, 1 },
        { "2D r2c IP IL 256x64",             VFFT_R2C, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 2, 256, 64, 0, 0, 1, 0, 1 },
        { "2D r2c OOP IL 64x64 howmany=4",   VFFT_R2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 2, 64, 64, 0, 0, 4, 0, 1 },
        { "3D c2c OOP IL 16^3",              VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 3, 16, 16, 16, 0, 1, 0, 1 },
        { "3D c2c IP IL 16^3",               VFFT_C2C, VFFT_INPLACE,    VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 3, 16, 16, 16, 0, 1, 0, 1 },
        { "3D c2c OOP SPLIT 16^3 NATURAL",   VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_NATURAL, 3, 16, 16, 16, 0, 1, 0, 1 },
        { "3D c2c OOP IL prime 16x16x17",    VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 3, 16, 16, 17, 0, 1, 0, 1 },
        { "3D r2c OOP IL 16^3",              VFFT_R2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 3, 16, 16, 16, 0, 1, 0, 1 },
        { "3D c2c OOP SPLIT 16^3 howmany=2", VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 3, 16, 16, 16, 0, 2, 0, 1 },
        { "4D c2c OOP IL 8^4",               VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 4, 8, 8, 8, 8, 1, 0, 1 },
        { "4D c2c OOP SPLIT 8^4",            VFFT_C2C, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 4, 8, 8, 8, 8, 1, 0, 1 },
        { "DCT-II OOP IL 64",                VFFT_DCT2, VFFT_OUTOFPLACE, VFFT_LAYOUT_INTERLEAVED, VFFT_ORDER_DEFAULT, 1, 64, 0, 0, 0, 1, 0, 1 },
        { "DCT-II OOP SPLIT 64",             VFFT_DCT2, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_DEFAULT, 1, 64, 0, 0, 0, 1, 0, 1 },
        { "DCT-II OOP SPLIT 64 NATURAL",     VFFT_DCT2, VFFT_OUTOFPLACE, VFFT_LAYOUT_SPLIT,       VFFT_ORDER_NATURAL, 1, 64, 0, 0, 0, 1, 0, 1 },
    };
    const int n = (int)(sizeof C / sizeof C[0]);
    for (int i = 0; i < n; i++) {
        const cell_t *c = &C[i];
        vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
        cfg.transform = c->transform; cfg.placement = c->placement;
        cfg.layout = c->layout; cfg.order = c->order;
        cfg.rigor = VFFT_MEASURE; cfg.dims = c->dims;
        cfg.n[0] = c->n0; cfg.n[1] = c->n1; cfg.n[2] = c->n2; cfg.n[3] = c->n3;
        cfg.howmany = c->K; cfg.batch_geom = c->geom; cfg.nthreads = c->nthreads;
        cfg.wisdom = W; cfg.wisdom_write = 0;
        fflush(stdout);
        fprintf(stderr, "----- %s\n", c->name);
        vfft_plan p = vfft_create(&cfg);
        printf("%-44s %s\n", c->name, p ? "SERVED" : "REFUSED");
        fflush(stdout);
        if (p) vfft_destroy(p);
    }
    return 0;
}
