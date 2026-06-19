/* orchestrator_smoke.c — exercise the plan_orchestrator sketch end-to-end.
 *
 * For each cell: vfft_proto_plan() -> execute fwd+bwd -> roundtrip must recover
 * N*x. Covers CT (wisdom hit), Rader, Bluestein (wisdom hits), and a cold CT
 * cell with an EMPTY wisdom to drive the MEASURE sweep-on-miss path.
 *
 * Build: build_tuned/build.py --src calibrator/orchestrator_smoke.c --jit --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../core/env.h"
#include "../core/plan_orchestrator.h"
#include "../generator/generated/registry.h"

static double roundtrip(vfft_proto_handle_t *h, int N, size_t K) {
    size_t total = (size_t)N * K;
    double *re, *im, *ore, *oim;
    vfft_proto_posix_memalign((void**)&re,  64, total*sizeof(double));
    vfft_proto_posix_memalign((void**)&im,  64, total*sizeof(double));
    vfft_proto_posix_memalign((void**)&ore, 64, total*sizeof(double));
    vfft_proto_posix_memalign((void**)&oim, 64, total*sizeof(double));
    srand(11);
    for (size_t i = 0; i < total; i++) { ore[i]=(double)rand()/RAND_MAX-0.5; oim[i]=(double)rand()/RAND_MAX-0.5; }
    memcpy(re, ore, total*sizeof(double)); memcpy(im, oim, total*sizeof(double));
    vfft_proto_plan_execute_fwd(h, re, im);
    vfft_proto_plan_execute_bwd(h, re, im);
    double m = 0;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(re[i]/(double)N - ore[i]), ei = fabs(im[i]/(double)N - oim[i]);
        if (er > m) m = er; if (ei > m) m = ei;
    }
    vfft_proto_aligned_free(re); vfft_proto_aligned_free(im);
    vfft_proto_aligned_free(ore); vfft_proto_aligned_free(oim);
    return m;
}

static int try_cell(const char *label, int N, size_t K, unsigned flags,
                    const vfft_proto_registry_t *reg,
                    vfft_proto_wisdom_t *wis, bluestein_wisdom_t *bw) {
    vfft_proto_handle_t h;
    if (vfft_proto_plan(&h, N, K, flags, reg, wis, bw) != 0) {
        printf("  N=%-5d %-10s  plan FAILED\n", N, label); return 1;
    }
    const char *kind = h.is_override ? "override" : (h.exec_fwd ? "CT/JIT" : "CT/generic");
    double rt = roundtrip(&h, N, K);
    int ok = rt < 1e-9;
    printf("  N=%-5d %-10s  %-11s rt=%.2e  %s\n", N, label, kind, rt, ok ? "PASS" : "FAIL");
    vfft_proto_handle_destroy(&h);
    return ok ? 0 : 1;
}

int main(void) {
    stride_env_init();
    stride_pin_thread(2);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    size_t K = 4;

    const char *ct  = "../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt";
    const char *blu = "../../src/dag-fft-compiler/generator/generated/vfft_bluestein_wisdom.txt";
    const char *e;
    if ((e = getenv("VFFT_PROTO_WIS")))      ct  = e;
    if ((e = getenv("VFFT_PROTO_BLUE_WIS"))) blu = e;

    vfft_proto_wisdom_t wis; memset(&wis, 0, sizeof wis);
    int have_ct = (vfft_proto_wisdom_load(&wis, ct) == 0);
    bluestein_wisdom_t bw; bluestein_wisdom_init(&bw);
    bluestein_wisdom_load(&bw, blu);
    printf("=== plan_orchestrator smoke (K=%zu, CT wisdom %s) ===\n", K, have_ct ? "loaded" : "MISSING");

    int fails = 0;
    fails += try_cell("CT",        256, K, VFFT_PROTO_MEASURE, &reg, &wis, &bw);  /* wisdom hit */
    fails += try_cell("Rader",     127, K, VFFT_PROTO_MEASURE, &reg, &wis, &bw);  /* prime, hit */
    fails += try_cell("Bluestein",  47, K, VFFT_PROTO_MEASURE, &reg, &wis, &bw);  /* prime, hit */
    fails += try_cell("ESTIMATE",  512, K, VFFT_PROTO_ESTIMATE, &reg, &wis, &bw); /* factorizer default */

    /* Cold CT cell + EMPTY wisdom -> drives the MEASURE sweep-on-miss (slow). */
    vfft_proto_wisdom_t empty; memset(&empty, 0, sizeof empty);
    printf("  -- sweep-on-miss (cold cell, empty wisdom; runs MEASURE) --\n");
    fails += try_cell("sweep",     240, K, VFFT_PROTO_MEASURE, &reg, &empty, &bw);

    printf("=== %s ===\n", fails == 0 ? "ALL PASS" : "FAILURES");
    return fails ? 1 : 0;
}
