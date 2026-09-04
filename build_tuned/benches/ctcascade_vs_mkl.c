/* ctcascade_vs_mkl.c — our small-radix CT cascade vs MKL BELOW 2048.
 *
 * WHY THIS FILE EXISTS
 * MKL's sub-2048 method is a small-radix (4/8) Cooley-Tukey cascade driven by
 * a 2 KB BAKED QUARTER-WAVE sine table (mkl_dft_fft_fix_twiddle_table_64f =
 * 257 doubles, sin(k*pi/512), verified by disassembly — docs/research/mkl_dfti).
 * We have the same shape (zturn: s0t ingest -> msg mids -> stf terminator) but
 * the engine REFUSES it below 2048: k1_commit.h:719 guards `if (N < 2048)
 * return 0;` because a 2026-08-06 race found the cascade 2.2x SLOWER than the
 * IL route at 128..1024.
 *
 * That guard blocks the ENGINE, not the cascade. vfft_zturn2_create/
 * execute_fwd are directly callable, so this harness drives the production
 * cascade at any N and races it against MKL, bypassing route selection.
 *
 * WHAT IT SEPARATES
 *   execute  — does our small-radix cascade beat MKL's below 2048?
 *   create   — MKL's table is BAKED (zero build); ours is computed per plan.
 *              Bakedness is a CREATE-time property and cannot show up in an
 *              execute measurement, so it is timed on its own axis.
 * Compactness (quarter-wave + symmetry) is NOT tested here: it needs a codelet
 * that reads a compact table, which our VTW2 contract does not do.
 *
 * Both engines produce NATURAL order (vfft_zturn2_set_natord(p,1)), so the
 * correctness column is a real elementwise cross-engine compare.
 *
 * Build:  python build.py --compile --src benches/ctcascade_vs_mkl.c --mkl --vfft
 * Run  :  ctcascade_vs_mkl.exe [rounds]
 */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <windows.h>

#include "vfft.h"
#include "oop/zturn.h"
#include "mkl_dfti.h"

static double qpc(void)
{
    LARGE_INTEGER f, t;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&t);
    return 1000.0 * (double)t.QuadPart / (double)f.QuadPart;
}
static int cmpd(const void *a, const void *b)
{ double x = *(const double *)a, y = *(const double *)b; return (x > y) - (x < y); }
static double med(double *v, int n) { qsort(v, n, sizeof(double), cmpd); return v[n / 2]; }
static double spr(double *v, int n) { return v[n - 1 - n / 10] - v[n / 10]; }

int main(int argc, char **argv)
{
    SetProcessAffinityMask(GetCurrentProcess(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
    int R = argc > 1 ? atoi(argv[1]) : 15;
    if (R > 64) R = 64;
    const double PACE = 200.0;

    /* EXPLICIT chains: vfft_zsplit_default_chain is a hardcoded switch that
     * only seeds 1024/2048/4096/8192/16384/32768, so 128..512 have no default
     * -- not a structural limit.  create_chain takes any legal chain:
     *   chain[0] must be 4 (radix4_z_s0t is the only ingest),
     *   terminator chain[nf-1] in {4,8}, and D[nf-2] %% 4 == 0. */
    struct { int N, nf, ch[6]; } cases[] = {
        { 128,  3, {4,4,8} },
        { 128,  3, {4,8,4} },
        { 256,  4, {4,4,4,4} },
        { 256,  3, {4,8,8} },
        { 512,  4, {4,4,4,8} },
        { 512,  4, {4,4,8,4} },
        { 1024, 5, {4,4,4,4,4} },
        { 1024, 4, {4,4,8,8} },
        { 2048, 4, {4,8,8,8} },
    };
    const int NS = (int)(sizeof cases / sizeof *cases);

    printf("our CT cascade (zturn, natord) vs MKL DFTI — BELOW 2048 the engine\n");
    printf("refuses this route (k1_commit.h:719); here it is driven directly.\n");
    printf("core 2, HIGH, %d rounds, %.0f ms pace, alternating order\n\n", R, PACE);
    printf("%-6s %-16s %10s %10s %8s %6s | %10s %10s | %s\n",
           "N", "chain", "casc ns", "MKL ns", "MKL/casc", "wins", "casc create", "MKL commit", "max relerr");

    for (int si = 0; si < NS; si++) {
        int N = cases[si].N;
        double *zin  = (double *)_aligned_malloc((size_t)2 * N * sizeof(double), 64);
        double *zc   = (double *)_aligned_malloc((size_t)2 * N * sizeof(double), 64);
        double *zm   = (double *)_aligned_malloc((size_t)2 * N * sizeof(double), 64);
        if (!zin || !zc || !zm) { printf("alloc failed\n"); return 1; }
        for (int i = 0; i < 2 * N; i++) zin[i] = sin(0.001 * (double)i) + 0.3 * cos(0.007 * (double)i);

        vfft_zturn2_plan_t *p =
            vfft_zturn2_create_chain(N, cases[si].ch, cases[si].nf);
        if (!p) {
            char cb[64] = "";
            for (int s = 0; s < cases[si].nf; s++) {
                char t[8]; snprintf(t, sizeof t, s ? ".%d" : "%d", cases[si].ch[s]);
                strncat(cb, t, sizeof cb - strlen(cb) - 1);
            }
            printf("%-6d %-16s (chain rejected by create_chain)\n", N, cb);
            continue;
        }
        if (!vfft_zturn2_set_natord(p, 1)) {
            printf("%-6d  (natord unavailable — cannot compare to MKL)\n", N);
            vfft_zturn2_destroy(p); continue;
        }
        char chain[64] = "";
        for (int s = 0; s < p->nf; s++) {
            char t[8]; snprintf(t, sizeof t, s ? ".%d" : "%d", p->chain[s]);
            strncat(chain, t, sizeof chain - strlen(chain) - 1);
        }

        DFTI_DESCRIPTOR_HANDLE h = NULL;
        DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
        DftiSetValue(h, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
        DftiCommitDescriptor(h);

        /* correctness: both natural order, elementwise */
        vfft_zturn2_execute_fwd(p, zin, zc);
        DftiComputeForward(h, zin, zm);
        double err = 0, den = 0;
        for (int i = 0; i < 2 * N; i++) den += fabs(zm[i]);
        den = den / (2 * N) + 1e-300;
        for (int i = 0; i < 2 * N; i++) { double d = fabs(zc[i] - zm[i]) / den; if (d > err) err = d; }

        int reps = 200000 / N; if (reps < 20) reps = 20;
        double A[64], B[64];
        for (int r = 0; r < R; r++) {
            if (r & 1) {
                Sleep((DWORD)PACE);
                double t0 = qpc(); for (int q = 0; q < reps; q++) DftiComputeForward(h, zin, zm);
                B[r] = (qpc() - t0) * 1e6 / reps;
                Sleep((DWORD)PACE);
                t0 = qpc(); for (int q = 0; q < reps; q++) vfft_zturn2_execute_fwd(p, zin, zc);
                A[r] = (qpc() - t0) * 1e6 / reps;
            } else {
                Sleep((DWORD)PACE);
                double t0 = qpc(); for (int q = 0; q < reps; q++) vfft_zturn2_execute_fwd(p, zin, zc);
                A[r] = (qpc() - t0) * 1e6 / reps;
                Sleep((DWORD)PACE);
                t0 = qpc(); for (int q = 0; q < reps; q++) DftiComputeForward(h, zin, zm);
                B[r] = (qpc() - t0) * 1e6 / reps;
            }
        }
        int wins = 0; for (int r = 0; r < R; r++) if (B[r] > A[r]) wins++;

        /* NATORD COST: MKL's output is natively natural, so forcing natord on
         * our side to get an elementwise compare charges us a permutation MKL
         * never pays.  stfn is 88 insns vs stf's 74 (same arith/mem, +14 of
         * pure permute).  Time a SCRAMBLED plan to price it. */
        /* SAME-RUN A/B of the dispatch change: cached msg_f[] vs the old
         * per-stage radix switch, alternated, scrambled plan on both so the
         * only difference is the dispatch. */
        double S[64], Q[64];
        vfft_zturn2_plan_t *ps =
            vfft_zturn2_create_chain(N, cases[si].ch, cases[si].nf);
        double sm = 0.0, qm = 0.0;
        if (ps) {
            for (int r = 0; r < R; r++) {
                if (r & 1) {
                    Sleep((DWORD)PACE);
                    double t0 = qpc();
                    for (int q = 0; q < reps; q++) vfft_zturn2_execute_fwd_nocache(ps, zin, zc);
                    Q[r] = (qpc() - t0) * 1e6 / reps;
                    Sleep((DWORD)PACE);
                    t0 = qpc();
                    for (int q = 0; q < reps; q++) vfft_zturn2_execute_fwd(ps, zin, zc);
                    S[r] = (qpc() - t0) * 1e6 / reps;
                } else {
                    Sleep((DWORD)PACE);
                    double t0 = qpc();
                    for (int q = 0; q < reps; q++) vfft_zturn2_execute_fwd(ps, zin, zc);
                    S[r] = (qpc() - t0) * 1e6 / reps;
                    Sleep((DWORD)PACE);
                    t0 = qpc();
                    for (int q = 0; q < reps; q++) vfft_zturn2_execute_fwd_nocache(ps, zin, zc);
                    Q[r] = (qpc() - t0) * 1e6 / reps;
                }
            }
            sm = med(S, R); qm = med(Q, R);
            vfft_zturn2_destroy(ps);
        }

        /* CREATE axis: ours computes the twiddle tables; MKL's are baked */
        double CA[16], CB[16];
        for (int r = 0; r < 9; r++) {
            vfft_zturn2_plan_t *tmp;
            double t0 = qpc();
            tmp = vfft_zturn2_create_chain(N, cases[si].ch, cases[si].nf);
            CA[r] = (qpc() - t0) * 1e6;
            if (tmp) { (void)vfft_zturn2_set_natord(tmp, 1); vfft_zturn2_destroy(tmp); }
            DFTI_DESCRIPTOR_HANDLE hh = NULL;
            t0 = qpc();
            DftiCreateDescriptor(&hh, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
            DftiCommitDescriptor(hh);
            CB[r] = (qpc() - t0) * 1e6;
            DftiFreeDescriptor(&hh);
        }

        double am = med(A, R), bm = med(B, R);
        printf("%-6d %-12s cached %8.0f  switch %8.0f  cache %+6.2f%%"
               "  | scr/MKL %6.3fx | %.1e\n",
               N, chain, sm, qm,
               (qm > 0 && sm > 0) ? 100.0 * (qm - sm) / qm : 0.0,
               sm > 0 ? bm / sm : 0.0, err);
        fflush(stdout);
        (void)spr;
        DftiFreeDescriptor(&h);
        vfft_zturn2_destroy(p);
        _aligned_free(zin); _aligned_free(zc); _aligned_free(zm);
    }
    return 0;
}
