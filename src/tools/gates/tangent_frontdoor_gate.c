/* tangent_frontdoor_gate.c — correctness of the PROMOTED wisdom, through the
 * public API.
 *
 * The other two gates check the kernels (tangent_gate.c) and the il2p wiring
 * (il2p_tangent_gate.c). This one checks the layer that the re-race actually
 * changed: vfft_create resolving a plan from the banked kind-3 row, including
 * its il_kv nibbles. If a banked variant were wrong, or the nibble decoded to
 * the wrong kernel, the transform is simply incorrect — and that shows up
 * here and nowhere else.
 *
 * Build (from build_tuned/):
 *   python build.py --src benches/tangent_frontdoor_gate.c --vfft
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <stdint.h>
#include <string.h>
#include "vfft.h"

typedef double _Complex cx;
static const double PI = 3.14159265358979323846;
static uint64_t lcg = 0x9E3779B97F4A7C15ull;
static double rnd(void){ lcg = lcg*6364136223846793005ull + 1442695040888963407ull;
    return ((double)(int64_t)(lcg>>11))/4503599627370496.0; }

static int one(int N)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = 1;
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;

    vfft_plan p = vfft_create(&cfg);
    if (!p) { printf("  N=%-5d  vfft_create returned NULL\n", N); return 0; }

    double *sre = (double *)malloc((size_t)N*sizeof(double));
    double *sim = (double *)malloc((size_t)N*sizeof(double));
    double *dre = (double *)malloc((size_t)N*sizeof(double));
    double *dim = (double *)malloc((size_t)N*sizeof(double));
    for (int i = 0; i < N; i++) { sre[i] = rnd(); sim[i] = rnd(); }

    vfft_execute(p, VFFT_FORWARD, sre, sim, dre, dim);

    double worst = 0;
    for (int k = 0; k < N; k++) {
        cx ref = 0;
        for (int n = 0; n < N; n++) {
            cx x = sre[n] + I*sim[n];
            ref += cexp(-2.0*PI*I*(double)((long long)k*n % N)/(double)N) * x;
        }
        cx got = dre[k] + I*dim[k];
        double e = cabs(got - ref);
        if (e > worst) worst = e;
    }
    int ok = worst < 1e-9;
    printf("  N=%-5d  worst |plan - DFT| = %.3e   %s\n", N, worst,
           ok ? "CORRECT" : "*** WRONG ***");

    free(sre); free(sim); free(dre); free(dim);
    vfft_destroy(p);
    return ok;
}

int main(void)
{
    printf("front-door gate: plans built from the re-raced kind-3 wisdom\n"
           "(TURNED-axis era, 2026-08-16:\n"
           " 128 -> pair 4x32,  il_kv 64 = mono mid + T256 wing32 leaf;\n"
           " 256 -> pair 16x16, il_kv 51 = tangent mid + tangent leaf;\n"
           " 512 -> pair 16x32, il_kv 67 = tangent mid + T256 wing32 leaf)\n\n");
    int ok = 1;
    ok &= one(128);
    ok &= one(256);
    ok &= one(512);
    /* neighbour that was NOT re-raced — it must be unaffected */
    ok &= one(1024);
    printf("\n%s\n", ok ? "FRONT-DOOR GATE: ALL CORRECT"
                        : "FRONT-DOOR GATE: FAILURE");
    return ok ? 0 : 1;
}
