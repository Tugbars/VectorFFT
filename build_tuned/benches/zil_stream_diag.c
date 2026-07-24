/* zil_stream_diag.c — DIAGNOSTIC: is the terminator's cost the 8 concurrent
 * power-of-2-strided output streams (all aliasing to ONE L1 set)?
 *
 * Arithmetic that motivates this: the terminator writes leg l at
 * zout + 2*(l*OLs + k), i.e. 8 streams exactly 16*OLs bytes apart. OLs = N/8
 * is a power of two >= 256, so the stride is always a multiple of 4096 =
 * (64 sets x 64 B). Set index = (stride>>6) & 63 == 0 for EVERY cell:
 * all 8 output streams land in the SAME L1D set (48 KB, 12-way, 64 sets),
 * and each iteration issues 8 concurrent RFO line-fills. VTune says exactly
 * this shape of pain: FB Full 12.9% of clockticks, Store Latency 35.1%.
 *
 * The probe: call the SAME kernel with a PADDED output stride (OLs + pad
 * complex). Addressing stays legal, the layout is no longer the contract's
 * (so the OUTPUT IS DELIBERATELY IN A DIFFERENT PLACE — this is a timing
 * probe only, not a correctness arm). If padding is materially faster, the
 * aliasing/FB thesis is confirmed and a staging-tile terminator (write 8
 * legs into an L1-resident tile, flush each leg as one burst) is the lever.
 *
 * Also probes the LEAF, whose 4 read + 4 write streams are 16*Ls bytes apart
 * (same aliasing property), by running it with a padded stride.
 *
 * Build: python build.py --src benches/zil_stream_diag.c
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

/* pads in COMPLEX units added to the per-leg stride */
static const int PADS[] = { 0, 4, 8, 32, 68 };
#define NPAD ((int)(sizeof(PADS) / sizeof(PADS[0])))

int main(void)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    g_bust = (char *)malloc(BUST_SZ);
    memset(g_bust, 1, BUST_SZ);

    const int cells[] = { 2048, 4096, 8192, 16384 };
    const int ROUNDS = 7;

    printf("=== TERMINATOR: 8 output streams, stride = 16*(OLs+pad) bytes ===\n");
    for (int ci = 0; ci < 4; ci++) {
        const int N = cells[ci];
        int chain[VFFT_ZSPLIT_MAX_NF];
        int nf = vfft_zsplit_default_chain(N, chain);
        vfft_zsplit_plan_t *p = vfft_zsplit_create(N, chain, nf);
        if (!p) { printf("N=%d create FAIL\n", N); return 1; }
        double *in = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
        srand(N); for (int i = 0; i < 2 * N; i++) in[i] = (double)rand() / RAND_MAX - 0.5;
        vfft_zsplit_execute_fwd(p, in, in);  /* leave p->sp hot/valid */

        const int cols = N / 8;
        const int reps = (N <= 4096) ? 400 : (N <= 8192) ? 200 : 100;
        double best[NPAD];
        double *out[NPAD];
        for (int q = 0; q < NPAD; q++) {
            size_t oLs = (size_t)cols + PADS[q];
            out[q] = (double *)_aligned_malloc(2 * 8 * oLs * 8 + 4096, 64);
            memset(out[q], 0, 2 * 8 * oLs * 8);
            best[q] = 1e30;
        }
        for (int r = 0; r < ROUNDS; r++) {
            for (int j = 0; j < NPAD; j++) {
                int q = (j + r) % NPAD;
                unsigned long long oLs = (unsigned long long)cols + PADS[q];
                cachebust();
                radix8_z_sterm_fwd_avx2(p->sp, 0, out[q], 0, p->twq, 0, 0, 0,
                                        oLs, 0, (unsigned long long)cols);
                double t0 = qpc_ms();
                for (int i = 0; i < reps; i++)
                    radix8_z_sterm_fwd_avx2(p->sp, 0, out[q], 0, p->twq, 0, 0, 0,
                                            oLs, 0, (unsigned long long)cols);
                double ns = (qpc_ms() - t0) * 1e6 / reps;
                if (ns < best[q]) best[q] = ns;
                Sleep(40);
            }
            Sleep(120);
        }
        printf("N=%-6d ", N);
        for (int q = 0; q < NPAD; q++)
            printf("pad%-3d %7.1f (%+6.2f%%)  ", PADS[q], best[q],
                   100.0 * (best[q] / best[0] - 1.0));
        printf("\n");
        for (int q = 0; q < NPAD; q++) _aligned_free(out[q]);
        _aligned_free(in);
        vfft_zsplit_destroy(p);
    }

    printf("\n=== LEAF: 4 read + 4 write streams, stride = 16*(Ls+pad) bytes ===\n");
    for (int ci = 0; ci < 4; ci++) {
        const int N = cells[ci];
        int chain[VFFT_ZSPLIT_MAX_NF];
        int nf = vfft_zsplit_default_chain(N, chain);
        if (chain[0] != 4) { printf("N=%-6d (leaf radix %d, skipped)\n", N, chain[0]); continue; }
        const int Ls = N / 4;
        const int reps = (N <= 4096) ? 400 : (N <= 8192) ? 200 : 100;
        double best[NPAD];
        double *buf[NPAD], *obuf[NPAD];
        for (int q = 0; q < NPAD; q++) {
            size_t ls = (size_t)Ls + PADS[q];
            buf[q]  = (double *)_aligned_malloc(2 * 4 * ls * 8 + 4096, 64);
            obuf[q] = (double *)_aligned_malloc(2 * 4 * ls * 8 + 4096, 64);
            for (size_t i = 0; i < 2 * 4 * ls; i++) buf[q][i] = 0.5;
            memset(obuf[q], 0, 2 * 4 * ls * 8);
            best[q] = 1e30;
        }
        for (int r = 0; r < ROUNDS; r++) {
            for (int j = 0; j < NPAD; j++) {
                int q = (j + r) % NPAD;
                unsigned long long ls = (unsigned long long)Ls + PADS[q];
                cachebust();
                radix4_z_s0s_fwd_avx2(buf[q], 0, obuf[q], 0, 0, 0,
                                      ls, 0, ls, 0, (unsigned long long)Ls);
                double t0 = qpc_ms();
                for (int i = 0; i < reps; i++)
                    radix4_z_s0s_fwd_avx2(buf[q], 0, obuf[q], 0, 0, 0,
                                          ls, 0, ls, 0, (unsigned long long)Ls);
                double ns = (qpc_ms() - t0) * 1e6 / reps;
                if (ns < best[q]) best[q] = ns;
                Sleep(40);
            }
            Sleep(120);
        }
        printf("N=%-6d ", N);
        for (int q = 0; q < NPAD; q++)
            printf("pad%-3d %7.1f (%+6.2f%%)  ", PADS[q], best[q],
                   100.0 * (best[q] / best[0] - 1.0));
        printf("\n");
        for (int q = 0; q < NPAD; q++) { _aligned_free(buf[q]); _aligned_free(obuf[q]); }
    }
    return 0;
}
