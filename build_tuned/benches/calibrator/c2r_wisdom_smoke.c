/* c2r_wisdom_smoke.c — validate the wisdom-first c2r dispatch.
 *
 * Build a valid packed spectrum (rfft forward of random real), then plan c2r
 * (1) WITHOUT wisdom (heuristic chooser, trusted reference) and (2) WITH the
 * calibrated c2r_wisdom. Confirm: the wisdom plan adopts the calibrated
 * factorization/variants, both produce the same inverse DFT, and the roundtrip
 * c2r(rfft(x)) == N*x holds.
 *
 * Build: build_tuned/build.py --src calibrator/c2r_wisdom_smoke.c --compile
 */
#define VFFT_RFFT_MAX_RADIX 32
#define VFFT_RFFT_RANGED 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../core/env.h"
#include "../core/c2r_dispatch.h"   /* vfft_c2r_plan_create + dispatch_set_wisdom + chooser */
#include "../generator/generated/rfft_registry_avx2.h"
#include "../generator/generated/c2r_registry_avx2.h"

int main(int argc, char **argv) {
    stride_env_init();
    int N = 256; size_t K = 8;
    rfft_codelets_t reg; memset(&reg, 0, sizeof reg);
    rfft_register_all_avx2(&reg);
    c2r_register_all_avx2(&reg);

    const char *wp = getenv("VFFT_PROTO_C2R_WIS");
    if (!wp) wp = "../../src/dag-fft-compiler/generator/generated/c2r_wisdom.txt";
    if (argc > 1) wp = argv[1];
    vfft_proto_wisdom_t wis; memset(&wis, 0, sizeof wis);
    int have_w = (vfft_proto_wisdom_load(&wis, wp) == 0);

    size_t total = (size_t)N * K;
    double *x, *spec, *ref, *out;
    vfft_proto_posix_memalign((void**)&x,    64, total*sizeof(double));
    vfft_proto_posix_memalign((void**)&spec, 64, 2*total*sizeof(double));
    vfft_proto_posix_memalign((void**)&ref,  64, total*sizeof(double));
    vfft_proto_posix_memalign((void**)&out,  64, total*sizeof(double));
    srand(7); for (size_t i = 0; i < total; i++) x[i] = (double)rand()/RAND_MAX*2-1;

    /* valid spectrum: rfft forward of x (any coverable factorization) */
    int ff[VFFT_RFFT_MAX_STAGES]; int fnf = vfft_c2r_choose_factors(N, ff, VFFT_RFFT_MAX_STAGES);
    rfft_plan_t *fp = rfft_plan_create(N, K, ff, fnf, &reg);
    if (!fp) { printf("forward plan NULL\n"); return 1; }
    rfft_execute_fwd_packed(fp, x, spec);
    rfft_plan_destroy(fp);

    /* (1) heuristic reference */
    vfft_c2r_dispatch_set_wisdom(NULL);
    c2r_plan_t *ph = vfft_c2r_plan_create(N, K, &reg);
    if (!ph) { printf("heuristic c2r plan NULL\n"); return 1; }
    memset(ref, 0, total*sizeof(double));
    c2r_execute_packed(ph, spec, ref);
    printf("heuristic: factors=["); for (int i=0;i<ph->base->nf;i++) printf("%s%d", i?"x":"", ph->base->factors[i]); printf("]\n");

    /* (2) wisdom-first */
    vfft_c2r_dispatch_set_wisdom(have_w ? &wis : NULL);
    c2r_plan_t *pw = vfft_c2r_plan_create(N, K, &reg);
    if (!pw) { printf("wisdom c2r plan NULL\n"); return 1; }
    memset(out, 0, total*sizeof(double));
    c2r_execute_packed(pw, spec, out);
    printf("wisdom   : factors=["); for (int i=0;i<pw->base->nf;i++) printf("%s%d", i?"x":"", pw->base->factors[i]); printf("]  (wisdom %s)\n", have_w?"loaded":"MISSING");

    double md = 0, rt = 0;
    for (size_t i = 0; i < total; i++) {
        double e = fabs(out[i]-ref[i]); if (e>md) md=e;
        double r = fabs(out[i]-(double)N*x[i]); if (r>rt) rt=r;
    }
    printf("wisdom-vs-heuristic max abs diff: %.2e  (%s)\n", md, md<1e-9?"SAME DFT — PASS":"DIFFER — FAIL");
    printf("roundtrip c2r(rfft(x)) vs N*x   : %.2e  (%s)\n", rt, rt<1e-7?"PASS":"FAIL");

    c2r_plan_destroy(ph); c2r_plan_destroy(pw);
    return (md < 1e-9 && rt < 1e-7) ? 0 : 1;
}
