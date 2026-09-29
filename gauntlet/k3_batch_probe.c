/* k3_batch_probe.c -- a small batch at one thread, ours vs MKL (2026-09-29).
 *
 * K transforms of length N, transform-contiguous (transform t at z[2tN..]),
 * interleaved, natural order, out of place, ONE thread on both sides:
 *   ours      the front door, howmany = K, VFFT_BATCH_TRANSFORM_CONTIGUOUS
 *   mkl       DFTI NUMBER_OF_TRANSFORMS = K, DISTANCE = N (the same memory)
 *   mkl x K   K calls of an N-point DFTI descriptor (MKL's own K=1 path)
 * No threaded arm in the process (a threaded MKL arm leaves spinning OpenMP
 * workers that the one-thread arms would then share the core with).
 * Protocol: core 2 HIGH + sibling guard, 15 rounds with the arms in
 * alternating order, each the minimum of 5 batches of ~50 us, median; 200 ms
 * between cells. Correctness: ours vs MKL batched, elementwise.
 * Usage: k3_batch_probe <wisdom dir> <K> <cell> [cell ...]; a cell is N (1D) or
 * N1xN2 (2D, N1 = the column length, N2 contiguous; MKL DFTI 2D with the same
 * lengths, DISTANCE = N1*N2)
 * Build: python gauntlet/build.py --compile --mkl --vfft --src gauntlet/k3_batch_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "sibling_guard.h"

enum { ROUNDS = 15, BATCHES = 5, NARM = 3 };
static const char *ARMN[NARM] = { "ours", "mkl", "mkl x K" };

static double now_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
static int cmpd(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

typedef struct
{
    vfft_plan h;
    DFTI_DESCRIPTOR_HANDLE db, d1;
    int N, K;
    double *zi, *zo;
} cell_t;

static void run_arm(const cell_t *c, int arm)
{
    const size_t tn = 2 * (size_t)c->N;
    if (arm == 0)
        vfft_execute(c->h, VFFT_FORWARD, c->zi, NULL, c->zo, NULL);
    else if (arm == 1)
        DftiComputeForward(c->db, c->zi, c->zo);
    else
        for (int t = 0; t < c->K; t++) DftiComputeForward(c->d1, c->zi + t * tn, c->zo + t * tn);
}
static double time_arm(const cell_t *c, int arm, int reps)
{
    double best = 1e30;
    for (int b = 0; b < BATCHES; b++)
    {
        const double t0 = now_ns();
        for (int i = 0; i < reps; i++) run_arm(c, arm);
        const double t = (now_ns() - t0) / reps;
        if (t < best) best = t;
    }
    return best;
}

int main(int argc, char **argv)
{
    if (argc < 4) { fprintf(stderr, "usage: k3_batch_probe <wisdom dir> <K> <N> [N ...]\n"); return 2; }
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    const int K = atoi(argv[2]);
    bench_pin_caller(2);
    bench_guard_sibling(2);
    mkl_set_num_threads(1);
    printf("K=%d, one thread, transform-contiguous; ns per batch (median of %d rounds); x = MKL / ours\n", K, ROUNDS);
    printf("%10s %10s %10s %10s %8s %8s %10s\n", "cell", "ours", "mkl", "mkl x K", "x mkl", "x mklxK", "err");
    for (int a = 3; a < argc; a++)
    {
        cell_t c;
        memset(&c, 0, sizeof c);
        int n1 = atoi(argv[a]), n2 = 0;
        const char *xs = strchr(argv[a], 'x');
        if (xs) n2 = atoi(xs + 1);
        c.N = n2 ? n1 * n2 : n1;   /* points per transform */
        c.K = K;
        const size_t total = (size_t)c.N * K;
        vfft_config_t cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C;
        cfg.placement = VFFT_OUTOFPLACE;
        cfg.dims = n2 ? 2 : 1;
        cfg.n[0] = n1;
        cfg.n[1] = n2;
        cfg.howmany = (size_t)K;
        cfg.order = VFFT_ORDER_NATURAL;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED;
        cfg.batch_geom = VFFT_BATCH_TRANSFORM_CONTIGUOUS;
        cfg.nthreads = 1;
        cfg.wisdom = W;
        c.h = vfft_create(&cfg);
        if (!c.h) { printf("%10s   ours: vfft_create refused\n", argv[a]); continue; }
        MKL_LONG lens[2] = { n1, n2 };
        if (n2) DftiCreateDescriptor(&c.db, DFTI_DOUBLE, DFTI_COMPLEX, 2, lens);
        else DftiCreateDescriptor(&c.db, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)c.N);
        DftiSetValue(c.db, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiSetValue(c.db, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
        DftiSetValue(c.db, DFTI_INPUT_DISTANCE, (MKL_LONG)c.N);
        DftiSetValue(c.db, DFTI_OUTPUT_DISTANCE, (MKL_LONG)c.N);
        DftiCommitDescriptor(c.db);
        if (n2) DftiCreateDescriptor(&c.d1, DFTI_DOUBLE, DFTI_COMPLEX, 2, lens);
        else DftiCreateDescriptor(&c.d1, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)c.N);
        DftiSetValue(c.d1, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiCommitDescriptor(c.d1);
        c.zi = _aligned_malloc(16 * total + 64, 64);
        c.zo = _aligned_malloc(16 * total + 64, 64);
        double *ref = malloc(16 * total);
        for (size_t j = 0; j < 2 * total; j++) c.zi[j] = sin(0.37 * j) + 0.25 * cos(1.3 * j);
        run_arm(&c, 1);
        memcpy(ref, c.zo, 16 * total);
        run_arm(&c, 0);
        double err = 0, mag = 0;
        for (size_t j = 0; j < 2 * total; j++)
        {
            err = fmax(err, fabs(c.zo[j] - ref[j]));
            mag = fmax(mag, fabs(ref[j]));
        }
        for (int arm = 0; arm < NARM; arm++) for (int i = 0; i < 20; i++) run_arm(&c, arm);
        const double t0 = now_ns();
        for (int i = 0; i < 20; i++) run_arm(&c, 0);
        int reps = (int)(50e3 / ((now_ns() - t0) / 20));
        if (reps < 4) reps = 4;
        double t[NARM][ROUNDS], med[NARM];
        for (int r = 0; r < ROUNDS; r++)
            for (int k = 0; k < NARM; k++)
            {
                const int arm = (r & 1) ? NARM - 1 - k : k;
                t[arm][r] = time_arm(&c, arm, reps);
            }
        for (int arm = 0; arm < NARM; arm++)
        {
            qsort(t[arm], ROUNDS, sizeof(double), cmpd);
            med[arm] = t[arm][ROUNDS / 2];
        }
        printf("%10s %10.0f %10.0f %10.0f %7.2fx %7.2fx %10.1e\n", argv[a], med[0], med[1], med[2], med[1] / med[0],
               med[2] / med[0], err / mag);
        fflush(stdout);
        vfft_destroy(c.h);
        DftiFreeDescriptor(&c.db);
        DftiFreeDescriptor(&c.d1);
        _aligned_free(c.zi);
        _aligned_free(c.zo);
        free(ref);
        Sleep(200);
    }
    (void)ARMN;
    return 0;
}
