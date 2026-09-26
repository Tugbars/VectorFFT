/* dest mode runs the WHOLE interior in zout: an unaligned zout splits every
 * interior zmm access. Compare dest vs plane binding at zout offsets 0/16/32/48. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "il_reg_kinds.h"
#include "ztt_vw.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static double timeit(vfft_ztt_plan_t *p, const double *x, double *y, int N)
{ int reps = (int)(3e7 / (N * 10.0)) + 3, r, k; double best = 1e30;
  for (k = 0; k < 15; k++) { double t = now(); for (r = 0; r < reps; r++) vfft_ztt_execute_fwd(p, x, y); t = (now() - t) / reps; if (t < best) best = t; } return best; }
int main(void)
{
    struct { int N, nf, ch[6]; } C[] = { { 4096, 4, { 8, 8, 8, 8 } }, { 16384, 5, { 8, 4, 8, 8, 8 } }, { 65536, 6, { 8, 4, 4, 8, 8, 8 } } };
    int ci, off, mode;
    for (ci = 0; ci < 3; ci++) {
        const int N = C[ci].N; double *xb = aligned_alloc(64, 16 * N + 128), *yb = aligned_alloc(64, 16 * N + 128); int i;
        vfft_ztt_plan_t *p = _ztt_create(N, C[ci].ch, C[ci].nf, 0, 0);
        for (i = 0; i < 2 * N + 16; i++) xb[i] = sin(0.1 * i);
        printf("VW=%d N=%-6d", VFFT_IL_VW, N);
        for (mode = 0; mode < 2; mode++) {
            vfft_ztt_bind(p, mode);
            printf("  %s:", mode ? "plane" : "dest ");
            for (off = 0; off < 4; off++) printf(" %7.0f", timeit(p, xb, (double *)((char *)yb + 16 * off), N));
        }
        printf("   (ns; zout offset 0/16/32/48 B, zin aligned)\n");
        vfft_ztt_destroy(p); free(xb); free(yb);
    }
    return 0;
}
