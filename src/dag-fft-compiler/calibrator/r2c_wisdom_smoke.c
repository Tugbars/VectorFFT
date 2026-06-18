/* r2c_wisdom_smoke.c — validate the wisdom-first r2c dispatch.
 *
 * Plan N=256 K=8 (1) WITHOUT wisdom (heuristic chooser, the trusted reference)
 * and (2) WITH the calibrated rfft_wisdom. Confirm: the wisdom plan adopts the
 * calibrated factorization/variants, and both produce the same packed DFT.
 *
 * Build: build_tuned/build.py --src calibrator/r2c_wisdom_smoke.c --compile
 */
#define VFFT_RFFT_MAX_RADIX 32
#define VFFT_RFFT_RANGED 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../core/env.h"
#include "../core/r2c_dispatch.h"     /* vfft_r2c_plan_create + dispatch_set_wisdom */
#include "../generator/generated/rfft_registry_avx2.h"

int main(int argc, char **argv) {
    stride_env_init();
    int N = 256; size_t K = 8;
    rfft_codelets_t reg; memset(&reg, 0, sizeof reg); rfft_register_all_avx2(&reg);

    const char *wp = getenv("VFFT_PROTO_RFFT_WIS");
    if (!wp) wp = "../../src/dag-fft-compiler/generator/generated/rfft_wisdom.txt";
    if (argc > 1) wp = argv[1];
    vfft_proto_wisdom_t wis; memset(&wis, 0, sizeof wis);
    int have_w = (vfft_proto_wisdom_load(&wis, wp) == 0);

    size_t total = (size_t)N * K;
    double *x, *ref, *out;
    vfft_proto_posix_memalign((void**)&x,   64, total*sizeof(double));
    vfft_proto_posix_memalign((void**)&ref, 64, total*sizeof(double));
    vfft_proto_posix_memalign((void**)&out, 64, total*sizeof(double));
    srand(7); for (size_t i = 0; i < total; i++) x[i] = (double)rand()/RAND_MAX*2-1;

    /* (1) heuristic reference */
    vfft_r2c_dispatch_set_wisdom(NULL);
    vfft_r2c_plan_t *ph = vfft_r2c_plan_create(N, K, VFFT_R2C_PACKED, &reg, NULL, NULL);
    if (!ph) { printf("heuristic plan NULL\n"); return 1; }
    memset(ref, 0, total*sizeof(double));
    vfft_r2c_execute_fwd(ph, x, ref, NULL);
    printf("heuristic: factors=["); for (int i=0;i<ph->rfft->nf;i++) printf("%s%d", i?"x":"", ph->rfft->factors[i]); printf("]\n");

    /* (2) wisdom-first */
    vfft_r2c_dispatch_set_wisdom(have_w ? &wis : NULL);
    vfft_r2c_plan_t *pw = vfft_r2c_plan_create(N, K, VFFT_R2C_PACKED, &reg, NULL, NULL);
    if (!pw) { printf("wisdom plan NULL\n"); return 1; }
    memset(out, 0, total*sizeof(double));
    vfft_r2c_execute_fwd(pw, x, out, NULL);
    printf("wisdom   : factors=["); for (int i=0;i<pw->rfft->nf;i++) printf("%s%d", i?"x":"", pw->rfft->factors[i]); printf("]  (wisdom %s)\n", have_w?"loaded":"MISSING");

    double md = 0; for (size_t i = 0; i < total; i++) { double e = fabs(out[i]-ref[i]); if (e>md) md=e; }
    printf("wisdom-vs-heuristic max abs diff: %.2e  (%s)\n", md, md<1e-9?"SAME DFT — PASS":"DIFFER — FAIL");

    vfft_r2c_plan_destroy(ph); vfft_r2c_plan_destroy(pw);
    return md < 1e-9 ? 0 : 1;
}
