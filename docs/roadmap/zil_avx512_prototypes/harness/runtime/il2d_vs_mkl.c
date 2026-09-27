/* b2d.c — 2D interleaved C2C, natural order, OOP, 1 thread: vfft calibrated cold
 * (PATIENT, scratch wisdom, nothing persisted) vs MKL DFTI 2D, alternating rounds,
 * min of 5 batches per sample, median of 21, plus a control arm (a second vfft plan). */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#include "mkl.h"
#include "vfft.h"
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }
typedef struct { vfft_plan p; DFTI_DESCRIPTOR_HANDLE h; } arm_t;
static void run(const arm_t *a, double *x, double *y)
{ if (a->p) vfft_execute(a->p, VFFT_FORWARD, x, NULL, y, NULL); else DftiComputeForward(a->h, x, y); }
static double sample(const arm_t *a, double *x, double *y, int reps)
{ double best = 1e30; for (int b = 0; b < 5; b++) { double t = now(); for (int r = 0; r < reps; r++) run(a, x, y); t = (now() - t) / reps; if (t < best) best = t; } return best; }
int main(int argc, char **argv)
{
    static const int C[][2] = { {64,64}, {256,256}, {1024,1024} };
    const int ROUNDS = 21;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(2, &cs); sched_setaffinity(0, sizeof cs, &cs);
    mkl_set_num_threads(1);
    printf("# vfft_isa=%s  MKL %s, 1 thread, pinned core 2, natural order, OOP\n", vfft_isa(),
           getenv("MKL_ENABLE_INSTRUCTIONS") ? getenv("MKL_ENABLE_INSTRUCTIONS") : "default dispatch");
    for (int c = 0; c < 3; c++) {
        const int N1 = C[c][0], N2 = C[c][1]; const size_t n = 2 * (size_t)N1 * N2;
        double *x = aligned_alloc(64, 8 * n), *y = aligned_alloc(64, 8 * n), *ym = aligned_alloc(64, 8 * n);
        for (size_t j = 0; j < n; j++) x[j] = sin(0.37 * j) + 0.25 * cos(1.3 * j + 0.1 * (j % 7));
        vfft_config_t cf; memset(&cf, 0, sizeof cf);
        cf.dims = 2; cf.n[0] = N1; cf.n[1] = N2; cf.howmany = 1; cf.layout = VFFT_LAYOUT_INTERLEAVED;
        cf.nthreads = 1; cf.order = VFFT_ORDER_NATURAL; cf.rigor = VFFT_PATIENT;
        double tc = now();
        vfft_plan p1 = vfft_create(&cf);            /* the calibration: a cold race */
        tc = (now() - tc) / 1e6;
        vfft_plan p2 = vfft_create(&cf);            /* the control: served from the in-memory verdict */
        DFTI_DESCRIPTOR_HANDLE h; MKL_LONG len[2] = { N1, N2 };
        DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 2, len);
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE); DftiCommitDescriptor(h);
        arm_t arms[3] = { { p1, 0 }, { 0, h }, { p2, 0 } };
        run(&arms[0], x, y); run(&arms[1], x, ym);
        double num = 0, den = 0; for (size_t j = 0; j < n; j++) { num += (y[j]-ym[j])*(y[j]-ym[j]); den += ym[j]*ym[j]; }
        const int reps = (int)(2e7 / (double)n) + 2;
        double t[3][64];
        for (int r = 0; r < ROUNDS; r++)
            for (int k = 0; k < 3; k++) { int a = (r & 1) ? 2 - k : k; t[a][r] = sample(&arms[a], x, y, reps); }
        double med[3], lo[3], hi[3];
        for (int a = 0; a < 3; a++) { qsort(t[a], ROUNDS, sizeof(double), cmpd); med[a] = t[a][ROUNDS/2]; lo[a] = t[a][ROUNDS/4]; hi[a] = t[a][3*ROUNDS/4]; }
        const double N = (double)N1 * N2, fl = 5.0 * N * log2(N);
        printf("%4dx%-4d route %-10s calib %6.0f ms | vfft %10.0f ns (%5.1f GF) [%0.f-%0.f] | MKL %10.0f ns (%5.1f GF) [%0.f-%0.f] | control %10.0f | MKL/vfft %.2fx | rel err vs MKL %.1e\n",
               N1, N2, vfft_plan_route(p1), tc, med[0], fl / med[0], lo[0], hi[0], med[1], fl / med[1], lo[1], hi[1], med[2], med[1] / med[0], sqrt(num / den));
        fflush(stdout);
        vfft_destroy(p1); vfft_destroy(p2); DftiFreeDescriptor(&h); free(x); free(y); free(ym);
    }
    return 0;
}
