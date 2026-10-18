/* THE API MATRIX GATE — the machine proof of the support table in
 * docs/design/vfft_front_door.md ("What is served"). Every row is a cell the
 * table makes a serve/refuse claim about; the gate creates it and asserts the
 * claim. A cell that starts serving (or refusing) flips a row here and in the
 * table together.
 *
 * THE ERROR CONTRACT (include/vfft.h): a refused create returns NULL and leaves
 * its reason in vfft_error(); a served create leaves vfft_error() empty. Both are
 * asserted per row. The zeroed configuration is INTERLEAVED, OUT OF PLACE,
 * NATURAL: every row names only what differs from it.
 *
 * Usage: api_matrix_gate.exe --wisdir <scratch dir>. Registered with a seeded
 * scratch copy (run_gates.py): its creates race and bank, and must never bank
 * into the library's own store. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

typedef struct {
    const char *name;
    int expect_serve;
    int transform, placement, layout, order, rank, n0, n1, n2, n3, K, batch;
} cell_t;

#define IL  VFFT_LAYOUT_INTERLEAVED
#define SP  VFFT_LAYOUT_SPLIT
#define OOP VFFT_PLACEMENT_OUT
#define IP  VFFT_PLACEMENT_IN
#define NAT VFFT_ORDER_NATURAL
#define SCR VFFT_ORDER_SCRAMBLED
#define TC  VFFT_BATCH_CONTIGUOUS
#define LM  VFFT_BATCH_LANE_MAJOR

int main(int argc, char **argv) {
    const char *W = NULL;
    for (int i = 1; i + 1 < argc; i++) if (!strcmp(argv[i], "--wisdir")) W = argv[i + 1];
    static const cell_t C[] = {
        /* ── 1D C2C: INTERLEAVED serves any N in both placements ── */
        { "1D c2c 127, the zeroed config",        1, VFFT_C2C, OOP, IL, NAT, 1, 127, 0, 0, 0, 1, 0 },
        { "1D c2c OOP IL prime 1021",             1, VFFT_C2C, OOP, IL, NAT, 1, 1021, 0, 0, 0, 1, 0 },
        { "1D c2c OOP IL awkward 129",            1, VFFT_C2C, OOP, IL, NAT, 1, 129, 0, 0, 0, 1, 0 },
        { "1D c2c OOP IL 4096 SCRAMBLED",         1, VFFT_C2C, OOP, IL, SCR, 1, 4096, 0, 0, 0, 1, 0 },
        { "1D c2c IP IL prime 127",               1, VFFT_C2C, IP,  IL, NAT, 1, 127, 0, 0, 0, 1, 0 },
        { "1D c2c IP IL 4096",                    1, VFFT_C2C, IP,  IL, NAT, 1, 4096, 0, 0, 0, 1, 0 },
        { "1D c2c IP IL K=4 (default geometry)",  1, VFFT_C2C, IP,  IL, NAT, 1, 256, 0, 0, 0, 4, 0 },
        { "1D c2c OOP IL K=4 CONTIGUOUS",         1, VFFT_C2C, OOP, IL, NAT, 1, 256, 0, 0, 0, 4, TC },
        { "1D c2c OOP IL K=4 LANE_MAJOR",         1, VFFT_C2C, OOP, IL, NAT, 1, 256, 0, 0, 0, 4, LM },
        { "1D c2c IP IL K=4 LANE_MAJOR",          1, VFFT_C2C, IP,  IL, NAT, 1, 256, 0, 0, 0, 4, LM },
        /* ── 1D C2C: SPLIT — prime in place only, no contiguous geometry ── */
        { "1D c2c IP SPLIT prime 127",            1, VFFT_C2C, IP,  SP, NAT, 1, 127, 0, 0, 0, 1, 0 },
        { "1D c2c OOP SPLIT prime 127",           0, VFFT_C2C, OOP, SP, NAT, 1, 127, 0, 0, 0, 1, 0 },
        { "1D c2c OOP SPLIT K=4 CONTIGUOUS",      0, VFFT_C2C, OOP, SP, NAT, 1, 256, 0, 0, 0, 4, TC },
        /* ── 1D real: INTERLEAVED any N in both placements; SPLIT even N; natural only ── */
        { "1D r2c OOP IL prime 101",              1, VFFT_R2C, OOP, IL, NAT, 1, 101, 0, 0, 0, 1, 0 },
        { "1D r2c OOP SPLIT prime 101",           0, VFFT_R2C, OOP, SP, NAT, 1, 101, 0, 0, 0, 1, 0 },
        { "1D c2r OOP IL odd 63",                 1, VFFT_C2R, OOP, IL, NAT, 1, 63, 0, 0, 0, 1, 0 },
        { "1D c2r OOP SPLIT odd 63",              0, VFFT_C2R, OOP, SP, NAT, 1, 63, 0, 0, 0, 1, 0 },
        { "1D c2r OOP IL odd 63 K=4 CONTIGUOUS",  1, VFFT_C2R, OOP, IL, NAT, 1, 63, 0, 0, 0, 4, TC },
        { "1D r2c IP IL odd 63",                  1, VFFT_R2C, IP,  IL, NAT, 1, 63, 0, 0, 0, 1, 0 },
        { "1D c2r IP IL odd 63",                  1, VFFT_C2R, IP,  IL, NAT, 1, 63, 0, 0, 0, 1, 0 },
        { "1D r2c IP IL even 64 K=4 CONTIGUOUS",  1, VFFT_R2C, IP,  IL, NAT, 1, 64, 0, 0, 0, 4, TC },
        { "1D r2c OOP IL 64 K=4 (default = lm)",  1, VFFT_R2C, OOP, IL, NAT, 1, 64, 0, 0, 0, 4, 0 },
        { "1D r2c OOP IL 64 K=4 LANE_MAJOR",      1, VFFT_R2C, OOP, IL, NAT, 1, 64, 0, 0, 0, 4, LM },
        { "1D c2r OOP IL 64 K=4 LANE_MAJOR",      1, VFFT_C2R, OOP, IL, NAT, 1, 64, 0, 0, 0, 4, LM },
        { "1D r2c OOP IL 63 K=4 LANE_MAJOR",      0, VFFT_R2C, OOP, IL, NAT, 1, 63, 0, 0, 0, 4, LM },   /* odd N: no real column engine */
        { "1D r2c IP IL 64 K=4 LANE_MAJOR",       1, VFFT_R2C, IP,  IL, NAT, 1, 64, 0, 0, 0, 4, LM },   /* one (N + 2) K plane */
        { "1D c2r IP IL 64 K=4 LANE_MAJOR",       1, VFFT_C2R, IP,  IL, NAT, 1, 64, 0, 0, 0, 4, LM },
        { "1D r2c OOP SPLIT even 64",             1, VFFT_R2C, OOP, SP, NAT, 1, 64, 0, 0, 0, 1, 0 },
        { "1D c2r OOP SPLIT even 64",             1, VFFT_C2R, OOP, SP, NAT, 1, 64, 0, 0, 0, 1, 0 },
        { "1D r2c IP SPLIT 64",                   0, VFFT_R2C, IP,  SP, NAT, 1, 64, 0, 0, 0, 1, 0 },
        { "1D r2c OOP IL 64 SCRAMBLED",           0, VFFT_R2C, OOP, IL, SCR, 1, 64, 0, 0, 0, 1, 0 },    /* a real spectrum is natural */
        /* ── 2D INTERLEAVED: the native tier, both transforms, both placements ── */
        { "2D c2c OOP IL 256x64",                 1, VFFT_C2C, OOP, IL, NAT, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2c OOP IL 256x64 SCRAMBLED",       1, VFFT_C2C, OOP, IL, SCR, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2c IP IL 256x64",                  1, VFFT_C2C, IP,  IL, NAT, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2c OOP IL prime 127x64",           1, VFFT_C2C, OOP, IL, NAT, 2, 127, 64, 0, 0, 1, 0 },
        { "2D c2c OOP IL odd 63x63",              1, VFFT_C2C, OOP, IL, NAT, 2, 63, 63, 0, 0, 1, 0 },
        { "2D c2c OOP IL 101x129",                1, VFFT_C2C, OOP, IL, NAT, 2, 101, 129, 0, 0, 1, 0 },
        { "2D c2c OOP IL 64x64 howmany=4",        1, VFFT_C2C, OOP, IL, NAT, 2, 64, 64, 0, 0, 4, 0 },
        { "2D r2c OOP IL 256x64",                 1, VFFT_R2C, OOP, IL, NAT, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2r OOP IL 256x64",                 1, VFFT_C2R, OOP, IL, NAT, 2, 256, 64, 0, 0, 1, 0 },
        { "2D r2c OOP IL prime 127x100",          1, VFFT_R2C, OOP, IL, NAT, 2, 127, 100, 0, 0, 1, 0 },
        { "2D r2c OOP IL odd N2 64x63",           1, VFFT_R2C, OOP, IL, NAT, 2, 64, 63, 0, 0, 1, 0 },
        { "2D r2c OOP IL 64x64 howmany=4",        1, VFFT_R2C, OOP, IL, NAT, 2, 64, 64, 0, 0, 4, 0 },
        { "2D r2c IP IL 256x64",                  1, VFFT_R2C, IP,  IL, NAT, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2c OOP IL 64x64 K=4 LANE_MAJOR",   1, VFFT_C2C, OOP, IL, NAT, 2, 64, 64, 0, 0, 4, LM },  /* the 2D lane-major batch (step 6, 2026-10-15) */
        { "2D c2c IP IL 64x64 K=4 LANE_MAJOR",    1, VFFT_C2C, IP,  IL, NAT, 2, 64, 64, 0, 0, 4, LM },
        { "2D r2c OOP IL 64x64 K=4 LANE_MAJOR",   1, VFFT_R2C, OOP, IL, NAT, 2, 64, 64, 0, 0, 4, LM },  /* the real 2D batch (step 6) */
        { "2D c2r IP IL 64x64 K=4 LANE_MAJOR",    1, VFFT_C2R, IP,  IL, NAT, 2, 64, 64, 0, 0, 4, LM },
        { "2D r2c OOP IL 64x63 K=4 LANE_MAJOR",   0, VFFT_R2C, OOP, IL, NAT, 2, 64, 63, 0, 0, 4, LM },  /* an odd N2: no real column engine */
        { "3D c2c OOP IL 16^3 K=4 LANE_MAJOR",    1, VFFT_C2C, OOP, IL, NAT, 3, 16, 16, 16, 0, 4, LM },  /* the 3D lane-major batch (step 6) */
        { "3D c2c IP IL 16^3 K=4 LANE_MAJOR",     1, VFFT_C2C, IP,  IL, NAT, 3, 16, 16, 16, 0, 4, LM },
        { "3D r2c OOP IL 16^3 K=4 LANE_MAJOR",    0, VFFT_R2C, OOP, IL, NAT, 3, 16, 16, 16, 0, 4, LM },  /* no lane-major real 3D door; the refusal names the contiguous volumes */
        /* ── the contiguous batches through the plane and volume queues (2026-10-18) ── */
        { "2D r2c IP IL 64x64 howmany=4",         1, VFFT_R2C, IP,  IL, NAT, 2, 64, 64, 0, 0, 4, 0 },   /* the plane queue in place: K padded planes */
        { "2D c2r IP IL 64x64 howmany=4 CONTIG",  1, VFFT_C2R, IP,  IL, NAT, 2, 64, 64, 0, 0, 4, TC },
        { "3D r2c OOP IL 16^3 howmany=2",         1, VFFT_R2C, OOP, IL, NAT, 3, 16, 16, 16, 0, 2, 0 },  /* the volume queue (the header's default geometry) */
        { "3D r2c IP IL 16^3 howmany=2",          1, VFFT_R2C, IP,  IL, NAT, 3, 16, 16, 16, 0, 2, 0 },  /* in place: K padded volumes */
        { "3D c2r OOP IL 16^3 howmany=2 CONTIG",  1, VFFT_C2R, OOP, IL, NAT, 3, 16, 16, 16, 0, 2, TC },
        { "3D c2r IP IL 16^3 howmany=2 CONTIG",   1, VFFT_C2R, IP,  IL, NAT, 3, 16, 16, 16, 0, 2, TC },
        { "3D r2c OOP IL odd 9x15x27 howmany=2",  1, VFFT_R2C, OOP, IL, NAT, 3, 9, 15, 27, 0, 2, 0 },   /* an odd N3 through the row plan's odd door */
        { "3D r2c OOP IL 1x16x16 howmany=2",      0, VFFT_R2C, OOP, IL, NAT, 3, 1, 16, 16, 0, 2, 0 },   /* every dim >= 2 */
        { "3D c2c IP IL 16^3 howmany=2 CONTIG",   1, VFFT_C2C, IP,  IL, NAT, 3, 16, 16, 16, 0, 2, TC },  /* the c2c volume queue, in place */
        /* ── 2D SPLIT: natural and scrambled ── */
        { "2D c2c OOP SPLIT 256x64",              1, VFFT_C2C, OOP, SP, NAT, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2c OOP SPLIT 256x64 SCRAMBLED",    1, VFFT_C2C, OOP, SP, SCR, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2c OOP SPLIT prime 127x64 SCR",    1, VFFT_C2C, OOP, SP, SCR, 2, 127, 64, 0, 0, 1, 0 },
        { "2D c2c IP SPLIT 256x64 SCRAMBLED",     1, VFFT_C2C, IP,  SP, SCR, 2, 256, 64, 0, 0, 1, 0 },
        { "2D c2c OOP SPLIT 64x64 howmany=4 SCR", 0, VFFT_C2C, OOP, SP, SCR, 2, 64, 64, 0, 0, 4, 0 },
        { "2D r2c OOP SPLIT 256x64",              1, VFFT_R2C, OOP, SP, NAT, 2, 256, 64, 0, 0, 1, 0 },
        { "2D r2c OOP SPLIT prime 127x100",       0, VFFT_R2C, OOP, SP, NAT, 2, 127, 100, 0, 0, 1, 0 },
        /* ── 3D / 4D: SPLIT serves the scrambled order only ── */
        { "3D c2c OOP IL 16^3",                   1, VFFT_C2C, OOP, IL, NAT, 3, 16, 16, 16, 0, 1, 0 },
        { "3D c2c OOP IL 16^3 SCRAMBLED",         1, VFFT_C2C, OOP, IL, SCR, 3, 16, 16, 16, 0, 1, 0 },
        { "3D c2c OOP IL odd 9x15x27",            1, VFFT_C2C, OOP, IL, NAT, 3, 9, 15, 27, 0, 1, 0 },
        { "3D c2c IP IL odd 9x15x27",             1, VFFT_C2C, IP,  IL, NAT, 3, 9, 15, 27, 0, 1, 0 },
        { "3D c2c IP IL odd 9x15x27 SCRAMBLED",   1, VFFT_C2C, IP,  IL, SCR, 3, 9, 15, 27, 0, 1, 0 },
        { "3D c2c IP IL 16^3",                    1, VFFT_C2C, IP,  IL, NAT, 3, 16, 16, 16, 0, 1, 0 },
        { "3D c2c OOP IL 16^3 howmany=2",         1, VFFT_C2C, OOP, IL, NAT, 3, 16, 16, 16, 0, 2, 0 },  /* contiguous volumes through the volume queue (2026-10-18) */
        { "3D r2c OOP IL 16^3",                   1, VFFT_R2C, OOP, IL, NAT, 3, 16, 16, 16, 0, 1, 0 },
        { "3D r2c IP IL 16^3",                    1, VFFT_R2C, IP,  IL, NAT, 3, 16, 16, 16, 0, 1, 0 },
        { "3D c2c OOP SPLIT 16^3",                0, VFFT_C2C, OOP, SP, NAT, 3, 16, 16, 16, 0, 1, 0 },
        { "3D c2c OOP SPLIT 16^3 SCRAMBLED",      1, VFFT_C2C, OOP, SP, SCR, 3, 16, 16, 16, 0, 1, 0 },
        { "3D c2c OOP SPLIT 16^3 howmany=2 SCR",  0, VFFT_C2C, OOP, SP, SCR, 3, 16, 16, 16, 0, 2, 0 },
        { "4D c2c OOP IL 8^4",                    0, VFFT_C2C, OOP, IL, NAT, 4, 8, 8, 8, 8, 1, 0 },
        { "4D c2c OOP SPLIT 8^4",                 0, VFFT_C2C, OOP, SP, NAT, 4, 8, 8, 8, 8, 1, 0 },
        { "4D c2c OOP SPLIT 8^4 SCRAMBLED",       1, VFFT_C2C, OOP, SP, SCR, 4, 8, 8, 8, 8, 1, 0 },
        /* ── real-to-real: 1D; layout, order and batch geometry are ignored ── */
        { "DCT-II OOP 64",                        1, VFFT_DCT2, OOP, IL, NAT, 1, 64, 0, 0, 0, 1, 0 },
        { "DCT-II OOP 64, SPLIT and SCRAMBLED",   1, VFFT_DCT2, OOP, SP, SCR, 1, 64, 0, 0, 0, 1, 0 },
        { "DCT-II 2D 64x64",                      0, VFFT_DCT2, OOP, IL, NAT, 2, 64, 64, 0, 0, 1, 0 },
    };
    const int n = (int)(sizeof C / sizeof C[0]);
    int fail = 0;
    printf("=== API matrix gate: %d cells (the front door's support table, asserted; store %s) ===\n",
           n, W ? W : "the library's own -- pass --wisdir <scratch>");
    for (int i = 0; i < n; i++) {
        const cell_t *c = &C[i];
        vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
        cfg.transform = c->transform; cfg.placement = c->placement;
        cfg.layout = c->layout; cfg.order = c->order;
        cfg.rank = c->rank;
        cfg.n[0] = c->n0; cfg.n[1] = c->n1; cfg.n[2] = c->n2; cfg.n[3] = c->n3;
        cfg.howmany = c->K; cfg.batch = c->batch; cfg.nthreads = 1;
        cfg.wisdom_dir = W;
        fflush(stdout);
        vfft_plan p = vfft_create(&cfg);
        const int served = (p != NULL);
        const char *why = vfft_error();
        if (served != c->expect_serve) {
            fail++;
            printf("  *** FAIL *** %-42s expected %s, got %s%s%s\n", c->name,
                   c->expect_serve ? "SERVED" : "REFUSED", served ? "SERVED" : "REFUSED",
                   served ? "" : ": ", served ? "" : why);
        } else if (!served && !why[0]) {
            fail++;
            printf("  *** FAIL *** %-42s refused with an EMPTY vfft_error()\n", c->name);
        } else if (served && why[0]) {
            fail++;
            printf("  *** FAIL *** %-42s served, but vfft_error() still says: %s\n", c->name, why);
        } else if (served)
            printf("  ok  %-42s served\n", c->name);
        else
            printf("  ok  %-42s refused: %.110s\n", c->name, why);
        fflush(stdout);
        if (p) vfft_destroy(p);
    }
    printf(fail ? "  === *** FAIL *** (%d fail) ===\n" : "  === ALL PASS === (%d cells)\n", fail ? fail : n);
    return fail ? 1 : 0;
}
