/* r32exp_reorder_gate — BIT-IDENTITY gate for the schedule-reordered r32 t2.
 *
 * Arms:
 *   ref   = shipped radix32_z_t2_fwd_avx2 (linked from the codelet lib)
 *   probe = radix32_z_t2_reord_fwd_avx2 (VFFT_SCHED_ORDER-injected order,
 *           #included below from docs/research/twmem_campaign/probes/)
 *
 * Same assignments in a different topological order compute identical
 * values => memcmp EXACT required. Table = the il2p VTW2 fwd fill verbatim
 * (src/core/oop/il2p.h create: record (pp,l) at (pp*(R1-1)+(l-1))*8,
 * [c,c,c,c][-s,+s,-s,+s], angle -2*pi*l*k/N), call shape = the il2p mid
 * call: mid_f(mid,0,zout,0,tw,0,R2,0,R2,0,R2). Odd R2 exercises the
 * VEX-128 odd-count tail in both arms.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

extern void radix32_z_t2_fwd_avx2(
    const double *zin, const double *zin_u, double *zout, double *zout_u,
    const double *tw_re, const double *tw_im,
    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count);

#include "../../docs/research/twmem_campaign/probes/r32exp__radix32_z_t2_reord_avx2.c"

static uint64_t lcg = 0x243F6A8885A308D3ull;
static double rnd(void)
{
    lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
    return ((double)(int64_t)(lcg >> 11)) / 9007199254740992.0; /* [-0.5,0.5) */
}

int main(void)
{
    const int R1 = 32;
    const int counts[] = { 1, 2, 3, 5, 8, 16, 31, 64, 129 };
    const double PI = 3.14159265358979323846;
    int fail = 0;

    for (size_t ci = 0; ci < sizeof(counts) / sizeof(counts[0]); ci++) {
        const int R2 = counts[ci];
        const size_t N = (size_t)R1 * (size_t)R2;
        const size_t nz = 2u * (size_t)R1 * (size_t)R2; /* doubles */
        const size_t npair = ((size_t)R2 + 1u) / 2u;
        const size_t ntw = npair * (size_t)(R1 - 1) * 8u;

        double *zin = (double *)malloc(nz * sizeof(double));
        double *za  = (double *)malloc(nz * sizeof(double));
        double *zb  = (double *)malloc(nz * sizeof(double));
        double *tw  = (double *)malloc(ntw * sizeof(double));
        if (!zin || !za || !zb || !tw) { fprintf(stderr, "alloc fail\n"); return 2; }

        for (size_t i = 0; i < nz; i++) zin[i] = rnd();
        memset(za, 0xAA, nz * sizeof(double));
        memset(zb, 0x55, nz * sizeof(double));

        /* il2p.h fwd table fill, verbatim */
        for (size_t pp = 0; pp < npair; pp++)
            for (int l = 1; l < R1; l++) {
                size_t off = (pp * (size_t)(R1 - 1) + (size_t)(l - 1)) * 8u;
                double *rf = tw + off;
                for (int j = 0; j < 2; j++) {
                    double k = (double)(2u * pp + (size_t)j);
                    double a = -2.0 * PI * (double)l * k / (double)N;
                    double c = cos(a), s = sin(a);
                    rf[2 * j] = c;      rf[2 * j + 1] = c;
                    rf[4 + 2 * j] = -s; rf[4 + 2 * j + 1] = s;
                }
            }

        radix32_z_t2_fwd_avx2(zin, 0, za, 0, tw, 0,
                              (size_t)R2, 0, (size_t)R2, 0, (size_t)R2);
        radix32_z_t2_reord_fwd_avx2(zin, 0, zb, 0, tw, 0,
                              (size_t)R2, 0, (size_t)R2, 0, (size_t)R2);

        int eq = memcmp(za, zb, nz * sizeof(double)) == 0;
        /* locate first mismatch for diagnostics */
        size_t bad = nz;
        if (!eq)
            for (size_t i = 0; i < nz; i++)
                if (memcmp(&za[i], &zb[i], sizeof(double)) != 0) { bad = i; break; }
        printf("count=%3d  memcmp=%s", R2, eq ? "EXACT" : "FAIL");
        if (!eq) {
            printf("  first@%zu ref=%.17g probe=%.17g", bad, za[bad], zb[bad]);
            fail = 1;
        }
        printf("\n");
        free(zin); free(za); free(zb); free(tw);
    }
    printf(fail ? "GATE: FAIL\n" : "GATE: ALL EXACT\n");
    return fail;
}
