/* flat_vs_t1s.c — isolate the t1s-vs-flat codelet benefit on ONE fixed plan.
 *
 * Builds the SAME factorization (DIT) twice and times both thermal-fair:
 *   all-FLAT : variants all 0  -> radix{R}_t1_dit  on every twiddled stage
 *   T1S      : stage0=0, rest=2 -> radix{R}_t1s_dit (the wisdom config)
 * Cooldown before each bench + measure both orders, keep the min per plan, so the
 * second-is-hotter bias cancels. Ratio flat/t1s > 1 => t1s codelets help.
 *
 * usage: flat_vs_t1s <N> <K> <f,f,...> [core=2]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../core/env.h"
#include "../core/executor.h"
#include "../core/planner.h"
#include "../core/dp_planner.h"   /* vfft_proto_now_ns + _vfft_proto_dp_sleep_ms */
#include "../generator/generated/registry.h"
#include <windows.h>

#define COOLDOWN_MS    10000
#define WARMUPS        5
#define TRIALS         9
#define INTER_TRIAL_MS 100

static double bench(stride_plan_t *plan, size_t total, size_t K,
                    double *re, double *im, const double *ore, const double *oim) {
    for (int w = 0; w < WARMUPS; w++) {
        memcpy(re, ore, total * sizeof(double));
        memcpy(im, oim, total * sizeof(double));
        vfft_proto_execute_fwd(plan, re, im, K);
    }
    int reps = (int)(2e5 / (total + 1));
    if (reps < 20) reps = 20;
    if (reps > 50000) reps = 50000;
    double best = 1e18;
    for (int t = 0; t < TRIALS; t++) {
        if (t > 0) _vfft_proto_dp_sleep_ms(INTER_TRIAL_MS);
        memcpy(re, ore, total * sizeof(double));
        memcpy(im, oim, total * sizeof(double));
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) vfft_proto_execute_fwd(plan, re, im, K);
        double ns = (vfft_proto_now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    return best;
}

int main(int argc, char **argv) {
    stride_env_init();
    if (argc < 4) { fprintf(stderr, "usage: flat_vs_t1s <N> <K> <f,f,...> [core=2]\n"); return 2; }
    int N = atoi(argv[1]); size_t K = (size_t)atoi(argv[2]);
    int core = (argc > 5) ? atoi(argv[5]) : 2;
    if (stride_pin_thread(core) != 0) fprintf(stderr, "warn: pin cpu%d\n", core);

    int f[STRIDE_MAX_STAGES], nf = 0;
    { char b[256]; strncpy(b, argv[3], sizeof b - 1); b[sizeof b - 1] = 0;
      char *t = strtok(b, ","); while (t && nf < STRIDE_MAX_STAGES) { f[nf++] = atoi(t); t = strtok(NULL, ","); } }

    /* "tuned" variants from argv[4] (comma codes 0=flat 1=log3 2=t1s) if given,
     * else default stage0=FLAT + rest=T1S. Flat column is always all-0. */
    int v_flat[STRIDE_MAX_STAGES], v_tuned[STRIDE_MAX_STAGES];
    for (int s = 0; s < nf; s++) { v_flat[s] = 0; v_tuned[s] = (s == 0) ? 0 : 2; }
    if (argc > 4 && argv[4][0]) {
        int vc[STRIDE_MAX_STAGES], nv = 0;
        char vb[128]; strncpy(vb, argv[4], sizeof vb - 1); vb[sizeof vb - 1] = 0;
        char *t = strtok(vb, ","); while (t && nv < STRIDE_MAX_STAGES) { vc[nv++] = atoi(t); t = strtok(NULL, ","); }
        if (nv == nf) { for (int s = 0; s < nf; s++) v_tuned[s] = vc[s]; }
        else fprintf(stderr, "warn: variants count %d != nf %d; using default t1s\n", nv, nf);
    }

    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    stride_plan_t *pf = vfft_proto_plan_create_ex(N, K, f, v_flat,  nf, 0, &reg);
    stride_plan_t *pt = vfft_proto_plan_create_ex(N, K, f, v_tuned, nf, 0, &reg);
    if (!pf || !pt) { fprintf(stderr, "plan build failed (flat=%p t1s=%p)\n", (void*)pf, (void*)pt); return 1; }

    size_t total = (size_t)N * K;
    double *re, *im, *ore, *oim;
    vfft_proto_posix_memalign((void**)&re, 64, total * sizeof(double));
    vfft_proto_posix_memalign((void**)&im, 64, total * sizeof(double));
    vfft_proto_posix_memalign((void**)&ore, 64, total * sizeof(double));
    vfft_proto_posix_memalign((void**)&oim, 64, total * sizeof(double));
    srand(42);
    for (size_t j = 0; j < total; j++) { ore[j] = (double)rand() / RAND_MAX - 0.5; oim[j] = (double)rand() / RAND_MAX - 0.5; }

    /* thermal-fair: cool before each; bench both orders (flat,t1s,t1s,flat), min per plan */
    _vfft_proto_dp_sleep_ms(COOLDOWN_MS); double f1 = bench(pf, total, K, re, im, ore, oim);
    _vfft_proto_dp_sleep_ms(COOLDOWN_MS); double t1 = bench(pt, total, K, re, im, ore, oim);
    _vfft_proto_dp_sleep_ms(COOLDOWN_MS); double t2 = bench(pt, total, K, re, im, ore, oim);
    _vfft_proto_dp_sleep_ms(COOLDOWN_MS); double f2 = bench(pf, total, K, re, im, ore, oim);
    double flat_ns = (f1 < f2) ? f1 : f2;
    double t1s_ns  = (t1 < t2) ? t1 : t2;

    char fs[64]; size_t p = 0; fs[0] = 0;
    for (int s = 0; s < nf; s++) p += (size_t)snprintf(fs + p, sizeof fs - p, "%s%d", s ? "x" : "", f[s]);

    char vs[64]; size_t q = 0; vs[0] = 0;
    for (int s = 0; s < nf; s++) q += (size_t)snprintf(vs + q, sizeof vs - q, "%s%d", s ? "," : "", v_tuned[s]);
    printf("=== flat vs tuned on %s DIT (N=%d K=%zu cpu%d, best-of-both-orders) ===\n", fs, N, K, core);
    printf("  variant codes: 0=flat(t1)  1=log3  2=t1s\n");
    printf("  all-FLAT (all 0)    : %.1f ns   (f1=%.1f f2=%.1f)\n", flat_ns, f1, f2);
    printf("  tuned  [%s]   : %.1f ns   (t1=%.1f t2=%.1f)\n", vs, t1s_ns, t1, t2);
    double r = flat_ns / t1s_ns;
    printf("  flat/tuned = %.3fx\n", r);
    if (r > 1.0) printf("  -> tuned variants are %.1f%% FASTER than all-flat\n", (r - 1.0) * 100.0);
    else         printf("  -> tuned variants are %.1f%% SLOWER than all-flat\n", (1.0 / r - 1.0) * 100.0);

    vfft_proto_plan_destroy(pf); vfft_proto_plan_destroy(pt);
    vfft_proto_aligned_free(re); vfft_proto_aligned_free(im);
    vfft_proto_aligned_free(ore); vfft_proto_aligned_free(oim);
    return 0;
}
