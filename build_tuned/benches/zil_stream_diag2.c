/* zil_stream_diag2.c — DIAGNOSTIC v2: separate the two confounded causes in
 * diag v1 (leaf -25% at 16384 with a padded stride).
 *
 * The leaf reads user z at leg stride 16*Ls bytes and writes the split plane
 * at the SAME stride. Ls = N/R0 is a power of two, so 16*Ls is a multiple of
 * 4096 = (64 L1 sets x 64 B): every leg stream — 4 reads AND 4 writes — has
 * the same L1D set index. Two independently fixable things follow:
 *   (A) BASE decorrelation: in-buffer and out-buffer lines collide with each
 *       other. Fixable by offsetting the scratch plane's base — ONE LINE.
 *   (B) STRIDE decorrelation: the 4 streams within each buffer collide with
 *       each other. Fixable only by a padded plane pitch — INVASIVE.
 * v1 changed both at once. Here they are varied independently, on ONE
 * allocation per arm-family so allocation luck cannot masquerade as a win.
 *
 * Arms (leaf, radix-4, count = Ls):
 *   base<off>  : contract stride, out-plane base shifted by <off> bytes
 *   pad<p>     : out AND in stride = Ls + p complex, base unshifted
 *   both       : stride pad + base shift
 * Timing-only probe: padded arms do not produce the contract layout.
 *
 * Build: python build.py --src benches/zil_stream_diag2.c
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
static void cachebust(void) { for (size_t i = 0; i < BUST_SZ; i += 64) g_bust[i]++; }

typedef struct { const char *name; int off_b; int pad_c; } arm_t;

int main(void)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    g_bust = (char *)malloc(BUST_SZ);
    memset(g_bust, 1, BUST_SZ);

    arm_t arms[] = {
        { "contract",   0,    0 },
        { "base+64B",   64,   0 },
        { "base+512B",  512,  0 },
        { "base+2KB",   2048, 0 },
        { "base+2KB+64",2112, 0 },
        { "pad4",       0,    4 },
        { "pad32",      0,    32 },
        { "both2K/4",   2048, 4 },
    };
    const int NA = (int)(sizeof(arms) / sizeof(arms[0]));
    const int cells[] = { 2048, 4096, 8192, 16384 };
    const int ROUNDS = 9;

    printf("=== LEAF (radix-4 s0s): 4 read + 4 write streams @ 16*Ls bytes ===\n");
    printf("%-8s", "N");
    for (int a = 0; a < NA; a++) printf("%14s", arms[a].name);
    printf("\n");

    for (int ci = 0; ci < 4; ci++) {
        const int N = cells[ci];
        const int Ls = N / 4;                    /* every cell's chain[0] == 4 */
        const int reps = (N <= 4096) ? 400 : (N <= 8192) ? 200 : 100;
        /* ONE allocation family: max stride + max offset, all arms inside it */
        size_t maxls = (size_t)Ls + 32;
        size_t span  = 2 * 4 * maxls * 8 + 8192;
        double *zIN = (double *)_aligned_malloc(span, 4096);
        double *zOUT = (double *)_aligned_malloc(span, 4096);
        for (size_t i = 0; i < span / 8; i++) zIN[i] = 0.25 + (double)(i & 15) * 0.01;
        memset(zOUT, 0, span);

        double best[16];
        for (int a = 0; a < NA; a++) best[a] = 1e30;
        for (int r = 0; r < ROUNDS; r++) {
            for (int j = 0; j < NA; j++) {
                int a = (j + r) % NA;
                unsigned long long ls = (unsigned long long)Ls + arms[a].pad_c;
                double *o = zOUT + arms[a].off_b / 8;
                cachebust();
                radix4_z_s0s_fwd_avx2(zIN, 0, o, 0, 0, 0, ls, 0, ls, 0,
                                      (unsigned long long)Ls);
                double t0 = qpc_ms();
                for (int i = 0; i < reps; i++)
                    radix4_z_s0s_fwd_avx2(zIN, 0, o, 0, 0, 0, ls, 0, ls, 0,
                                          (unsigned long long)Ls);
                double ns = (qpc_ms() - t0) * 1e6 / reps;
                if (ns < best[a]) best[a] = ns;
                Sleep(40);
            }
            Sleep(120);
        }
        printf("%-8d", N);
        for (int a = 0; a < NA; a++)
            printf("%8.1f%+5.1f%%", best[a], 100.0 * (best[a] / best[0] - 1.0));
        printf("\n");
        _aligned_free(zIN); _aligned_free(zOUT);
    }
    return 0;
}
