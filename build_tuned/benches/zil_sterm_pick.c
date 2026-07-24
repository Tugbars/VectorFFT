/* zil_sterm_pick.c — per-cell terminator-schedule pick (z_cascade_plan §4.9993).
 *
 * sterm (single-quad) vs sterm2 (2-quad unroll-and-jam) are BIT-IDENTICAL
 * schedules whose delta is the same order as code-placement luck, so the
 * choice is MEASURED per cell through the REAL production path
 * (vfft_zsplit_execute_fwd with p->t2q toggled) in the real lib layout.
 *
 * Usage: zil_sterm_pick.exe [N]   (no arg = all 4 cells in-process;
 *        per-cell isolated invocation preferred for finals)
 * Build: python build.py --src benches/zil_sterm_pick.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>

#include "zsplit.h"

static double qpc_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}
static char *g_bust;
#define BUST_SZ (32u * 1024u * 1024u)
static void cachebust(void)
{
    for (size_t i = 0; i < BUST_SZ; i += 64) g_bust[i]++;
}

static int run_cell(int N)
{
    int chain[VFFT_ZSPLIT_MAX_NF];
    int nf = vfft_zsplit_default_chain(N, chain);
    vfft_zsplit_plan_t *p = vfft_zsplit_create(N, chain, nf);
    if (!p) { printf("N=%d create FAIL\n", N); return 1; }

    double *in = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
    double *o0 = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
    double *o1 = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
    srand(20260725 + N);
    for (int i = 0; i < 2 * N; i++)
        in[i] = (double)rand() / RAND_MAX - 0.5;

    p->t2q = 0; vfft_zsplit_execute_fwd(p, in, o0);
    p->t2q = 1; vfft_zsplit_execute_fwd(p, in, o1);
    if (memcmp(o0, o1, (size_t)2 * N * 8) != 0) {
        printf("N=%-6d GATE sterm/sterm2 BIT-MISMATCH FAIL\n", N);
        return 1;
    }
    printf("N=%-6d GATE sterm==sterm2 bit-identical PASS\n", N);

    const int RF = (N <= 2048) ? 300 : (N <= 4096) ? 150 : (N <= 8192) ? 80 : 40;
    const int ROUNDS = 9;
    double best[2] = { 1e30, 1e30 };
    for (int r = 0; r < ROUNDS; r++) {
        for (int j = 0; j < 2; j++) {
            int a = (j + r) & 1;
            p->t2q = a;
            cachebust();
            vfft_zsplit_execute_fwd(p, in, o0);
            double t0 = qpc_ms();
            for (int i = 0; i < RF; i++)
                vfft_zsplit_execute_fwd(p, in, o0);
            double ns = (qpc_ms() - t0) * 1e6 / RF;
            if (ns < best[a]) best[a] = ns;
            Sleep(80);
        }
        Sleep(150);
    }
    int pick = (best[1] < best[0]) ? 1 : 0;
    printf("N=%-6d sterm %9.1f ns   sterm2 %9.1f ns   (d=%+.2f%%)  PICK=%s\n",
           N, best[0], best[1], 100.0 * (best[1] / best[0] - 1.0),
           pick ? "sterm2" : "sterm");
    _aligned_free(in); _aligned_free(o0); _aligned_free(o1);
    vfft_zsplit_destroy(p);
    return 0;
}

int main(int argc, char **argv)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    g_bust = (char *)malloc(BUST_SZ);
    memset(g_bust, 1, BUST_SZ);
    int rc = 0;
    if (argc > 1) rc = run_cell(atoi(argv[1]));
    else {
        const int cells[] = { 2048, 4096, 8192, 16384 };
        for (int i = 0; i < 4; i++) rc |= run_cell(cells[i]);
    }
    return rc;
}
