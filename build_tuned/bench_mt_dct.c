/* bench_mt_dct.c — MT scaling for DCT/DST/DHT.
 *
 * For each transform × (N, K) cell × T ∈ {1, 2, 4, 8}, measure wall time
 * via min-of-21 reps with warmup. Report ns/call and speedup vs T=1.
 *
 * Build: python build.py --src bench_mt_dct.c --vfft
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#include "vfft.h"

static double now_ns(void) {
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}

static double bench_min(vfft_plan p,
                        void (*fn)(vfft_plan, const double *, double *),
                        const double *in, double *out, size_t NK,
                        int n_warmup, int n_reps) {
    /* Warmup */
    for (int i = 0; i < n_warmup; i++) fn(p, in, out);
    double best = 1e30;
    for (int i = 0; i < n_reps; i++) {
        double t0 = now_ns();
        fn(p, in, out);
        double dt = now_ns() - t0;
        if (dt < best) best = dt;
    }
    (void)NK;
    return best;
}

typedef struct {
    const char *name;
    vfft_plan (*plan_fn)(int, size_t, unsigned);
    void (*exec_fn)(vfft_plan, const double *, double *);
} transform_t;

static void run_one(const transform_t *t, int N, size_t K) {
    size_t NK = (size_t)N * K;
    double *in  = (double *)vfft_alloc(NK * sizeof(double));
    double *out = (double *)vfft_alloc(NK * sizeof(double));
    srand(42 + N + (int)K);
    for (size_t i = 0; i < NK; i++) in[i] = (double)rand()/RAND_MAX - 0.5;

    int Ts[] = {1, 2, 4, 8};
    double ns[4];
    for (int ti = 0; ti < 4; ti++) {
        int T = Ts[ti];
        vfft_set_num_threads(T);
        vfft_plan p = t->plan_fn(N, K, VFFT_ESTIMATE);
        ns[ti] = bench_min(p, t->exec_fn, in, out, NK, 5, 21);
        vfft_destroy(p);
    }

    printf("  %-6s N=%-5d K=%-5zu  T=1: %8.0f ns  T=2: %8.0f (%.2fx)  T=4: %8.0f (%.2fx)  T=8: %8.0f (%.2fx)\n",
           t->name, N, K,
           ns[0],
           ns[1], ns[0] / ns[1],
           ns[2], ns[0] / ns[2],
           ns[3], ns[0] / ns[3]);

    vfft_free(in); vfft_free(out);
}

int main(void) {
    vfft_init();
    vfft_pin_thread(0);

    /* Pick cells where MT is plausible. K is the parallelism axis.
     * N*K = 1M..16M is the sweet spot — small enough to run many reps,
     * big enough that MT win exceeds dispatch overhead. */
    struct { int N; size_t K; } cells[] = {
        {  256, 1024 },   /* 256K elems  — small */
        { 1024, 1024 },   /* 1M elems    — medium */
        { 4096, 1024 },   /* 4M elems    — large */
        { 1024, 4096 },   /* 4M elems    — large K */
        { 4096, 4096 },   /* 16M elems   — very large */
    };
    int n_cells = (int)(sizeof(cells)/sizeof(cells[0]));

    transform_t transforms[] = {
        { "DCT-II", vfft_plan_dct2, vfft_execute_dct2 },
        { "DCT-IV", vfft_plan_dct4, vfft_execute_dct4 },
        { "DST-II", vfft_plan_dst2, vfft_execute_dst2 },
        { "DHT",    vfft_plan_dht,  vfft_execute_dht  },
    };
    int n_transforms = (int)(sizeof(transforms)/sizeof(transforms[0]));

    printf("=== bench_mt_dct -- MT scaling for DCT/DST/DHT ===\n");
    printf("Each cell: ns/call (min over 21 reps after 5 warmup), speedup vs T=1\n\n");

    for (int ci = 0; ci < n_cells; ci++)
    for (int ti = 0; ti < n_transforms; ti++) {
        run_one(&transforms[ti], cells[ci].N, cells[ci].K);
    }

    vfft_set_num_threads(1);
    return 0;
}
