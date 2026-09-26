/* best fused cell per N (natural fwd, out of place, untiled + 1024/2048 tiles),
 * min-of-reps ns; a buffer offset sweep (0/16/32/48 B) on the winner */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "il_reg_kinds.h"
#include "ztt_vw.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static double timeit(vfft_ztt_plan_t *p, const double *x, double *y, int N)
{
    int reps = (int)(2e7 / (N * 10.0)) + 3, r, k; double best = 1e30;
    for (k = 0; k < 9; k++) { double t = now(); for (r = 0; r < reps; r++) vfft_ztt_execute_fwd(p, x, y); t = (now() - t) / reps; if (t < best) best = t; }
    return best;
}
int main(int argc, char **argv)
{
    static const int Ns[] = { 256, 1024, 4096, 16384, 65536 };
    static const size_t tiles[] = { 0, 1024, 2048 };
    int ni, i, scr;
    for (scr = 0; scr < 2; scr++)
    for (ni = 0; ni < 5; ni++) {
        const int N = Ns[ni];
        double *xb = aligned_alloc(64, 16 * N + 128), *yb = aligned_alloc(64, 16 * N + 128);
        double bt = 1e30; char bs[64] = ""; int bi = -1; size_t btile = 0;
        for (i = 0; i < 2 * N + 16; i++) xb[i] = sin(0.1 * i);
        for (i = 0; i < VFFT_ZTT_NCELLS; i++) {
            const vfft_ztt_cell_t *c = &vfft_ztt_cells[i]; size_t ti;
            vfft_ztt_plan_t *p;
            if (c->n != N) continue;
            p = _ztt_create(N, c->chain, c->nf, scr, 0);
            if (!p || p->staged) { if (p) vfft_ztt_destroy(p); continue; }
            vfft_ztt_bind(p, 0);
            for (ti = 0; ti < 3; ti++) {
                double t;
                if (!vfft_ztt_set_tile(p, tiles[ti])) continue;
                t = timeit(p, xb, yb, N);
                if (t < bt) { bt = t; bi = i; btile = tiles[ti]; vfft_ztt_chain_str(p, bs, sizeof bs); }
            }
            vfft_ztt_destroy(p);
        }
        if (bi >= 0) {
            const vfft_ztt_cell_t *c = &vfft_ztt_cells[bi];
            vfft_ztt_plan_t *p = _ztt_create(N, c->chain, c->nf, scr, 0); int off; double to[4];
            vfft_ztt_bind(p, 0); vfft_ztt_set_tile(p, btile);
            for (off = 0; off < 4; off++) to[off] = timeit(p, (double *)((char *)xb + 16 * off), (double *)((char *)yb + 16 * off), N);
            printf("VW=%d %s N=%-6d best %-10s tile=%-5zu %9.0f ns  (%.2f ns/pt)   buffer offset 0/16/32/48 B: %.0f %.0f %.0f %.0f\n",
                   VFFT_IL_VW, scr ? "plain  " : "natural", N, bs, btile, bt, bt / N, to[0], to[1], to[2], to[3]);
            vfft_ztt_destroy(p);
        }
        free(xb); free(yb);
    }
    return 0;
}
