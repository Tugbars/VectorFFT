/* zttcal.c — calibrate ZTURN-T per N (every fused cell x every legal tile,
 * natural order, out of place, K=1, the gauntlet contract), then time the
 * winner against MKL in alternating rounds with a control arm.
 * Build twice: VFFT_IL_VW=4 (avx2 kernels, clamped) and VFFT_IL_VW=8. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
#include "il_reg_kinds.h"
#include "ztt_vw.h"
#include "mkl.h"

static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }

typedef struct { vfft_ztt_plan_t *p; DFTI_DESCRIPTOR_HANDLE h; } arm_t;
static void run(const arm_t *a, double *x, double *y)
{
    if (a->p) vfft_ztt_execute_fwd(a->p, x, y);
    else DftiComputeForward(a->h, x, y);
}
/* one timing sample: min over 5 batches of `reps` calls, ns per call */
static double sample(const arm_t *a, double *x, double *y, int reps)
{
    double best = 1e30; int b, r;
    for (b = 0; b < 5; b++) {
        double t = now();
        for (r = 0; r < reps; r++) run(a, x, y);
        t = (now() - t) / reps; if (t < best) best = t;
    }
    return best;
}
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

int main(int argc, char **argv)
{
    static const int Ns[] = { 1024, 2048, 4096 };
    static const size_t tiles[] = { 0, 256, 512, 768, 1024, 1536, 2048, 3072 };
    const int ROUNDS = 21;
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(2, &cs); sched_setaffinity(0, sizeof cs, &cs);
    mkl_set_num_threads(1);
    printf("# VW=%d (%s)  MKL: %s\n", VFFT_IL_VW, VFFT_IL_VW == 8 ? "avx512" : "avx2",
           getenv("MKL_ENABLE_INSTRUCTIONS") ? getenv("MKL_ENABLE_INSTRUCTIONS") : "default dispatch");
    for (int ni = 0; ni < 3; ni++) {
        const int N = Ns[ni];
        const int reps = (int)(4e6 / (N * 10.0)) + 3;
        double *x = aligned_alloc(64, 16 * N), *y = aligned_alloc(64, 16 * N), *ym = aligned_alloc(64, 16 * N);
        for (int i = 0; i < 2 * N; i++) x[i] = sin(0.37 * i) + 0.25 * cos(1.3 * i);
        /* ---- calibration: every fused cell x every legal tile ---- */
        int ncand = 0; double bt = 1e30; int bi = -1; size_t btile = 0; char bs[64] = "";
        for (int i = 0; i < VFFT_ZTT_NCELLS; i++) {
            const vfft_ztt_cell_t *c = &vfft_ztt_cells[i];
            if (c->n != N) continue;
            vfft_ztt_plan_t *p = _ztt_create(N, c->chain, c->nf, 0, 0);
            if (!p) continue;
            if (p->staged) { vfft_ztt_destroy(p); continue; }
            vfft_ztt_bind(p, 0);
            for (size_t ti = 0; ti < sizeof tiles / sizeof *tiles; ti++) {
                if (!vfft_ztt_set_tile(p, tiles[ti])) continue;
                arm_t a = { p, 0 };
                run(&a, x, y);
                double t = sample(&a, x, y, reps);
                ncand++;
                if (t < bt) { bt = t; bi = i; btile = tiles[ti]; vfft_ztt_chain_str(p, bs, sizeof bs); }
            }
            vfft_ztt_destroy(p);
        }
        if (bi < 0) { printf("N=%d: no fused cell\n", N); continue; }
        /* ---- verdict: winner vs MKL vs control (the winner again), alternating ---- */
        const vfft_ztt_cell_t *c = &vfft_ztt_cells[bi];
        vfft_ztt_plan_t *pw = _ztt_create(N, c->chain, c->nf, 0, 0), *pc = _ztt_create(N, c->chain, c->nf, 0, 0);
        vfft_ztt_bind(pw, 0); vfft_ztt_set_tile(pw, btile); vfft_ztt_bind(pc, 0); vfft_ztt_set_tile(pc, btile);
        DFTI_DESCRIPTOR_HANDLE h;
        DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE); DftiCommitDescriptor(h);
        arm_t arms[3] = { { pw, 0 }, { 0, h }, { pc, 0 } };
        /* correctness: ours vs MKL */
        run(&arms[0], x, y); run(&arms[1], x, ym);
        double num = 0, den = 0;
        for (int i = 0; i < 2 * N; i++) { num += (y[i] - ym[i]) * (y[i] - ym[i]); den += ym[i] * ym[i]; }
        double t[3][64];
        for (int r = 0; r < ROUNDS; r++)
            for (int k = 0; k < 3; k++) { int a = (r & 1) ? 2 - k : k; t[a][r] = sample(&arms[a], x, y, reps); }
        double med[3], lo[3], hi[3];
        for (int a = 0; a < 3; a++) { qsort(t[a], ROUNDS, sizeof(double), cmpd); med[a] = t[a][ROUNDS / 2]; lo[a] = t[a][ROUNDS / 4]; hi[a] = t[a][3 * ROUNDS / 4]; }
        const double fl = 5.0 * N * log2((double)N);
        printf("N=%-5d cands=%-3d winner %-12s tile=%-4zu | ours %8.0f ns (%5.1f GF) [%0.f-%0.f] | MKL %8.0f ns (%5.1f GF) [%0.f-%0.f] | control %8.0f [%0.f-%0.f] | MKL/ours %.2fx | rel err vs MKL %.1e\n",
               N, ncand, bs, btile, med[0], fl / med[0], lo[0], hi[0], med[1], fl / med[1], lo[1], hi[1], med[2], lo[2], hi[2],
               med[1] / med[0], sqrt(num / den));
        vfft_ztt_destroy(pw); vfft_ztt_destroy(pc); DftiFreeDescriptor(&h);
        free(x); free(y); free(ym);
    }
    return 0;
}
