/* CIL race: pipeline-generated interleaved kernels (codelet_cil.ml, shared
 * SR scheduler) vs the legacy hand-scheduled ones (codelet_zil.ml).
 * Bit-identical OUTPUT does not imply identical INSTRUCTION ORDER — the
 * scheduler reorders — so speed has to be measured, not assumed.
 *
 * Canonical discipline: pin logical core 2 (mask 4), HIGH_PRIORITY_CLASS,
 * 32 MB cachebust, arms rotated each round inside ONE allocation, Sleep
 * pacing, best-of-N. Acceptance: within +/-3%.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>

#define DECL(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long,         \
    unsigned long long, unsigned long long, unsigned long long);
DECL(radix8_z_n1ref_fwd_avx2)   DECL(radix8_z_n1_fwd_avx2)
DECL(radix16_z_n1ref_fwd_avx2)  DECL(radix16_z_n1_fwd_avx2)
DECL(radix8_z_n1tref_fwd_avx2)  DECL(radix8_z_n1t_fwd_avx2)
DECL(radix8_z_t2ref_fwd_avx2)   DECL(radix8_z_t2_fwd_avx2)

typedef void (*kfn)(const double *, const double *, double *, double *,
                    const double *, const double *, unsigned long long,
                    unsigned long long, unsigned long long, unsigned long long,
                    unsigned long long);

static double qpc_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}
static char *g_bust;
#define BUST_SZ (32u * 1024u * 1024u)
static void cachebust(void) { for (size_t i = 0; i < BUST_SZ; i += 64) g_bust[i]++; }
static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return ((double)(*s >> 8) / (double)(1u << 24)) - 0.5;
}

typedef struct {
    const char *name;
    kfn legacy, pipe;
    int R, use_tw, corner;
} spec_t;

int main(void)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    g_bust = (char *)malloc(BUST_SZ);
    memset(g_bust, 1, BUST_SZ);

    const size_t count = 256;
    const size_t maxR = 16;
    /* ONE allocation, with a 64 B (one cache line, NOT a 4 KB multiple) skew
       between the two planes. Two independently page-aligned buffers put every
       zin/zout stream on the SAME L1 set, which made timings bimodal (~585 vs
       ~975 ns for the same binary, flipping run to run). */
    size_t plane = 2 * maxR * count * 8;
    char *arena = (char *)_aligned_malloc(2 * plane + 8192, 4096);
    double *zin = (double *)arena;
    double *zout = (double *)(arena + plane + 64);
    double *tw = (double *)_aligned_malloc((count / 2) * 15 * 8 * 8, 64);
    unsigned seed = 5150;
    for (size_t i = 0; i < 2 * maxR * count; i++) zin[i] = urand(&seed);
    for (size_t i = 0; i < (count / 2) * 15 * 8; i++) tw[i] = urand(&seed);

    spec_t specs[] = {
        { "n1  r8 ", radix8_z_n1ref_fwd_avx2,  radix8_z_n1_fwd_avx2,  8,  0, 0 },
        { "n1  r16", radix16_z_n1ref_fwd_avx2, radix16_z_n1_fwd_avx2, 16, 0, 0 },
        { "n1t r8 ", radix8_z_n1tref_fwd_avx2, radix8_z_n1t_fwd_avx2, 8,  0, 1 },
        { "t2  r8 ", radix8_z_t2ref_fwd_avx2,  radix8_z_t2_fwd_avx2,  8,  1, 0 },
    };
    const int NS = (int)(sizeof(specs) / sizeof(specs[0]));
    const int ROUNDS = 11, REPS = 3000;

    printf("=== interleaved kernels: pipeline vs legacy hand-scheduled ===\n");
    printf("(one allocation, arms rotated, best-of-%d, pinned core 2)\n\n", ROUNDS);
    printf("%-8s %12s %12s %9s   %s\n", "kernel", "legacy ns", "pipeline ns", "pipe/leg", "verdict");

    for (int si = 0; si < NS; si++) {
        spec_t *s = &specs[si];
        const double *twp = s->use_tw ? tw : NULL;
        unsigned long long OLs = s->corner ? (unsigned long long)s->R : count;
        double best[2] = { 1e30, 1e30 };
        for (int r = 0; r < ROUNDS; r++) {
            for (int a = 0; a < 2; a++) {
                int arm = (a + r) & 1;
                kfn f = arm ? s->pipe : s->legacy;
                cachebust();
                f(zin, 0, zout, 0, twp, 0, count, 0, OLs, 0, count);
                double t0 = qpc_ms();
                for (int i = 0; i < REPS; i++)
                    f(zin, 0, zout, 0, twp, 0, count, 0, OLs, 0, count);
                double ns = (qpc_ms() - t0) * 1e6 / REPS;
                if (ns < best[arm]) best[arm] = ns;
                Sleep(30);
            }
            Sleep(100);
        }
        double ratio = best[1] / best[0];
        const char *v = (ratio <= 1.03 && ratio >= 0.97) ? "PARITY"
                        : (ratio < 0.97) ? "pipeline FASTER" : "pipeline SLOWER";
        printf("%-8s %12.1f %12.1f %9.3f   %s\n", s->name, best[0], best[1], ratio, v);
    }
    _aligned_free(arena); _aligned_free(tw); free(g_bust);
    return 0;
}
