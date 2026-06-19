/* rader_bench.c — time dag's Rader at the old-wisdom K=4 Rader primes.
 *
 * The 4 radix-smooth-(N-1) primes from production's bluestein wisdom
 * (127/251/257/401). dag's Rader rides CT wisdom for the (N-1) inner FFT, so
 * with dag's improved CT codelets it may beat production's recorded ns. Cross-
 * session caveat applies (old_ns is from a different box/build) — directional.
 *
 * Uses stride_plan_destroy (bridge, override_destroy-aware) — no leak.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../core/env.h"
#include "../core/executor.h"
#include "../core/prime_dispatch.h"   /* auto_plan_dispatch + stride_plan_destroy via the bridge */
#include "../generator/generated/registry.h"
#include <windows.h>

static double now_ns(void){ LARGE_INTEGER f,c; QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return (double)c.QuadPart*1e9/(double)f.QuadPart; }

static double roundtrip(stride_plan_t *p, size_t total, size_t K, int N,
                        double *re, double *im, const double *ore, const double *oim) {
    memcpy(re, ore, total*sizeof(double)); memcpy(im, oim, total*sizeof(double));
    vfft_proto_execute_fwd(p, re, im, K);
    vfft_proto_execute_bwd(p, re, im, K);
    double m = 0;
    for (size_t i = 0; i < total; i++) {
        double er = fabs(re[i]/(double)N - ore[i]), ei = fabs(im[i]/(double)N - oim[i]);
        if (er > m) m = er; if (ei > m) m = ei;
    }
    return m;
}

static double bench(stride_plan_t *p, size_t total, size_t K,
                    double *re, double *im, const double *ore, const double *oim) {
    double tw = now_ns();
    while (now_ns() - tw < 0.3e9) { memcpy(re,ore,total*sizeof(double)); memcpy(im,oim,total*sizeof(double));
        vfft_proto_execute_fwd(p, re, im, K); }
    int reps = (int)(2e6/(total+1)); if (reps < 50) reps = 50; if (reps > 200000) reps = 200000;
    double best = 1e18;
    for (int t = 0; t < 9; t++) {
        memcpy(re,ore,total*sizeof(double)); memcpy(im,oim,total*sizeof(double));
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) vfft_proto_execute_fwd(p, re, im, K);
        double ns = (now_ns()-t0)/reps; if (ns < best) best = ns;
    }
    return best;
}

int main(int argc, char **argv) {
    stride_env_init();
    int core = (argc > 1) ? atoi(argv[1]) : 2;
    if (stride_pin_thread(core) != 0) fprintf(stderr, "warn: pin cpu%d\n", core);
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);
    size_t K = 4;

    struct { int N; double old_ns; } cells[] = {
        {127, 2635.77}, {251, 3966.30}, {257, 3740.74}, {401, 7002.33},
    };
    int nc = (int)(sizeof(cells)/sizeof(cells[0]));

    printf("=== dag Rader vs production bluestein-wisdom, K=4 (cpu%d) ===\n", core);
    printf("  %-6s %-14s %11s %11s %8s  %s\n", "N", "plan", "dag_ns", "old_ns", "old/dag", "rt_err");
    double logsum = 0; int n = 0;
    for (int i = 0; i < nc; i++) {
        int N = cells[i].N;
        stride_plan_t *p = vfft_proto_auto_plan_dispatch(N, K, &reg, NULL);
        if (!p) { printf("  N=%-4d  auto_plan_dispatch NULL\n", N); continue; }
        size_t total = (size_t)N * K;
        double *re, *im, *ore, *oim;
        vfft_proto_posix_memalign((void**)&re,  64, total*sizeof(double));
        vfft_proto_posix_memalign((void**)&im,  64, total*sizeof(double));
        vfft_proto_posix_memalign((void**)&ore, 64, total*sizeof(double));
        vfft_proto_posix_memalign((void**)&oim, 64, total*sizeof(double));
        srand(7);
        for (size_t j = 0; j < total; j++) { ore[j]=(double)rand()/RAND_MAX-0.5; oim[j]=(double)rand()/RAND_MAX-0.5; }
        double rt = roundtrip(p, total, K, N, re, im, ore, oim);
        double dag = bench(p, total, K, re, im, ore, oim);
        double r = cells[i].old_ns / dag;
        printf("  %-6d Rader(N-1=%-3d) %11.1f %11.1f %7.3fx  %.1e\n", N, N-1, dag, cells[i].old_ns, r, rt);
        if (r > 0) { logsum += log(r); n++; }
        vfft_proto_aligned_free(re); vfft_proto_aligned_free(im);
        vfft_proto_aligned_free(ore); vfft_proto_aligned_free(oim);
        stride_plan_destroy(p);  /* bridge: honors override_destroy */
    }
    if (n) printf("\n  geomean old/dag over %d cells: %.3fx  (>1 => dag faster, cross-session)\n", n, exp(logsum/n));
    return 0;
}
