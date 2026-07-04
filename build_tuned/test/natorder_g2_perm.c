/* natorder_g2_perm.c — GATE G2: cost of a digit-reversal ROW-PERMUTATION pass
 * relative to the FFT, on this host (i9-14900KF, AVX2, core-0 pinned).
 *
 * Pure memory-kernel measurement — NO FFT code. Rows are K doubles, data is
 * split-complex: two planes (re, im) of N x K doubles each, 64B-aligned.
 * A "perm pass" permutes the N rows of BOTH planes by a mixed-radix
 * digit-reversal permutation (the permutation an in-place scrambled-order
 * c2c would need to undo to give natural order).
 *
 * Kernels per cell:
 *   (a) in-place cycle-following row moves, memcpy per row, K-double stack temp
 *   (b) gather rows into scratch + memcpy whole plane back (2-pass)
 *   (c) same-order memcpy of the whole plane (1 pass) = bandwidth baseline
 *
 * Timing: QueryPerformanceCounter, reps-calibrated blocks, best-of-5,
 * SetThreadAffinityMask core 0.
 *
 * Output: one CSV line per cell:
 *   N,K,factors,MB_total,moved_frac,a_ns,a_GBs,b_ns,b_GBs,c_ns,c_GBs,a_over_c,b_over_c
 */
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define KMAX 256
#define BEST_OF 5
#define TARGET_S 0.008 /* ~8 ms per timed block */

static double g_qpf;
static double now_s(void) {
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart / g_qpf;
}

/* ---- mixed-radix digit reversal --------------------------------------- */
/* n = d0 + f0*(d1 + f1*(d2 + ...)); rev accumulates digits in reverse:
 * r = ((d0*f1 + d1)*f2 + d2)... i.e. r = r*f[i] + (n % f[i]); n /= f[i]. */
static void build_perm(int N, const int *f, int nf, int *perm) {
    for (int n = 0; n < N; n++) {
        int r = 0, x = n;
        for (int i = 0; i < nf; i++) { r = r * f[i] + (x % f[i]); x /= f[i]; }
        perm[n] = r;
    }
    /* verify bijection */
    unsigned char *seen = (unsigned char *)calloc((size_t)N, 1);
    if (!seen) { fprintf(stderr, "alloc fail\n"); exit(1); }
    for (int n = 0; n < N; n++) {
        if (perm[n] < 0 || perm[n] >= N || seen[perm[n]]) {
            fprintf(stderr, "perm is not a bijection at n=%d\n", n); exit(1);
        }
        seen[perm[n]] = 1;
    }
    free(seen);
}

/* ---- kernel (a): in-place cycle-following ----------------------------- */
static void perm_inplace(double *re, double *im, const int *perm, int N, int K,
                         unsigned char *visited) {
    const size_t rb = (size_t)K * sizeof(double);
    double tr[KMAX], ti[KMAX];
    memset(visited, 0, (size_t)N);
    for (int s = 0; s < N; s++) {
        if (visited[s]) continue;
        visited[s] = 1;
        if (perm[s] == s) continue;
        /* gather semantics: row[cur] <- row[perm[cur]] along the cycle */
        memcpy(tr, re + (size_t)s * K, rb);
        memcpy(ti, im + (size_t)s * K, rb);
        int cur = s;
        for (;;) {
            int nxt = perm[cur];
            if (nxt == s) {
                memcpy(re + (size_t)cur * K, tr, rb);
                memcpy(im + (size_t)cur * K, ti, rb);
                break;
            }
            memcpy(re + (size_t)cur * K, re + (size_t)nxt * K, rb);
            memcpy(im + (size_t)cur * K, im + (size_t)nxt * K, rb);
            visited[nxt] = 1;
            cur = nxt;
        }
    }
}

/* ---- kernel (b): gather into scratch + copy back (2-pass) -------------- */
static void perm_gather(double *re, double *im, double *sre, double *sim,
                        const int *perm, int N, int K) {
    const size_t rb = (size_t)K * sizeof(double);
    const size_t pb = (size_t)N * K * sizeof(double);
    for (int i = 0; i < N; i++) {
        memcpy(sre + (size_t)i * K, re + (size_t)perm[i] * K, rb);
        memcpy(sim + (size_t)i * K, im + (size_t)perm[i] * K, rb);
    }
    memcpy(re, sre, pb);
    memcpy(im, sim, pb);
}

/* ---- kernel (c): same-order whole-plane memcpy (1 pass) ---------------- */
static void copy_baseline(double *re, double *im, double *sre, double *sim,
                          int N, int K) {
    const size_t pb = (size_t)N * K * sizeof(double);
    memcpy(sre, re, pb);
    memcpy(sim, im, pb);
}

/* ---- timing helper: calibrate reps, then best-of-5 blocks -------------- */
typedef void (*kern_fn)(void *ctx);

typedef struct {
    double *re, *im, *sre, *sim;
    const int *perm;
    int N, K;
    unsigned char *visited;
} cell_ctx;

static void run_a(void *p) { cell_ctx *c = (cell_ctx *)p; perm_inplace(c->re, c->im, c->perm, c->N, c->K, c->visited); }
static void run_b(void *p) { cell_ctx *c = (cell_ctx *)p; perm_gather(c->re, c->im, c->sre, c->sim, c->perm, c->N, c->K); }
static void run_c(void *p) { cell_ctx *c = (cell_ctx *)p; copy_baseline(c->re, c->im, c->sre, c->sim, c->N, c->K); }

static double time_kernel(kern_fn fn, void *ctx) {
    /* warmup + calibrate rep count for ~TARGET_S per block */
    fn(ctx);
    double t0 = now_s(); fn(ctx); double t1 = now_s();
    double one = t1 - t0;
    long reps = (one > 0) ? (long)(TARGET_S / one) : 1000000L;
    if (reps < 1) reps = 1;
    if (reps > 2000000L) reps = 2000000L;

    double best = 1e300;
    for (int b = 0; b < BEST_OF; b++) {
        double s = now_s();
        for (long r = 0; r < reps; r++) fn(ctx);
        double e = now_s();
        double per = (e - s) / (double)reps;
        if (per < best) best = per;
    }
    return best; /* seconds per application */
}

static volatile double g_sink;

int main(void) {
    LARGE_INTEGER f; QueryPerformanceFrequency(&f);
    g_qpf = (double)f.QuadPart;

    SetThreadAffinityMask(GetCurrentThread(), 1); /* pin core 0 */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);

    static const struct { int N; int f[6]; int nf; const char *name; } shapes[] = {
        {   256, {16, 16},          2, "16x16"     },
        {  1024, {32, 32},          2, "32x32"     },
        {  4096, { 8,  8,  8,  8},  4, "8x8x8x8"   },
        { 16384, { 8,  8, 16, 16},  4, "8x8x16x16" },
    };
    static const int Ks[] = { 8, 64, 256 };

    printf("N,K,factors,MB_total,moved_frac,a_ns,a_GBs,b_ns,b_GBs,c_ns,c_GBs,a_over_c,b_over_c\n");
    fflush(stdout);

    for (size_t si = 0; si < sizeof(shapes) / sizeof(shapes[0]); si++) {
        const int N = shapes[si].N;
        int *perm = (int *)malloc((size_t)N * sizeof(int));
        unsigned char *visited = (unsigned char *)malloc((size_t)N);
        if (!perm || !visited) { fprintf(stderr, "alloc fail\n"); return 1; }
        build_perm(N, shapes[si].f, shapes[si].nf, perm);

        int moved = 0;
        for (int i = 0; i < N; i++) if (perm[i] != i) moved++;

        for (size_t ki = 0; ki < sizeof(Ks) / sizeof(Ks[0]); ki++) {
            const int K = Ks[ki];
            const size_t plane = (size_t)N * K * sizeof(double);
            const size_t total = 4 * plane; /* re+im + scratch re+im */
            if (total > (size_t)256 * 1024 * 1024) {
                printf("%d,%d,%s,SKIP(>256MB),,,,,,,,,\n", N, K, shapes[si].name);
                continue;
            }
            double *re  = (double *)_aligned_malloc(plane, 64);
            double *im  = (double *)_aligned_malloc(plane, 64);
            double *sre = (double *)_aligned_malloc(plane, 64);
            double *sim = (double *)_aligned_malloc(plane, 64);
            if (!re || !im || !sre || !sim) { fprintf(stderr, "alloc fail\n"); return 1; }
            for (size_t i = 0; i < (size_t)N * K; i++) {
                re[i] = (double)(i & 1023) * 0.5;
                im[i] = (double)(i & 511) * 0.25;
                sre[i] = 0.0; sim[i] = 0.0;
            }

            cell_ctx ctx = { re, im, sre, sim, perm, N, K, visited };

            double tc_ = time_kernel(run_c, &ctx);   /* baseline first */
            double tb  = time_kernel(run_b, &ctx);
            double ta  = time_kernel(run_a, &ctx);

            /* traffic models (bytes touched, read+write, both planes) */
            const double pb = (double)plane;
            const double traf_a = 2.0 * 2.0 * ((double)moved / N) * pb; /* r+w x 2 planes, moved rows only */
            const double traf_b = 8.0 * pb;  /* gather r+w + copyback r+w, 2 planes */
            const double traf_c = 4.0 * pb;  /* r+w, 2 planes */

            printf("%d,%d,%s,%.1f,%.3f,%.0f,%.2f,%.0f,%.2f,%.0f,%.2f,%.3f,%.3f\n",
                   N, K, shapes[si].name,
                   (double)(2 * plane) / (1024.0 * 1024.0), /* data set = 2 planes MB */
                   (double)moved / N,
                   ta * 1e9, traf_a / ta * 1e-9,
                   tb * 1e9, traf_b / tb * 1e-9,
                   tc_ * 1e9, traf_c / tc_ * 1e-9,
                   ta / tc_, tb / tc_);
            fflush(stdout);

            g_sink = re[(size_t)N * K - 1] + im[0] + sre[1] + sim[2];

            _aligned_free(re); _aligned_free(im);
            _aligned_free(sre); _aligned_free(sim);
        }
        free(perm); free(visited);
    }
    return 0;
}
