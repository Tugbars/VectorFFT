/* zturn_dit_stageprobe.c — WHERE does DIT-natural lose its 5–18%?
 *
 * The race verdict (dit_race_run{1,2}.txt) attributed the loss to the
 * ingest's rho-scattered user-memory loads — plausibly, not provenly. This
 * probe times each stage of both natural pipelines SEPARATELY:
 *
 *   N (DIF-natural):  s0t ingest | msg fwd mids (1..nf-2) | stfn terminator
 *   D (DIT-natural):  dtsn ingest | msd mids (nf-2..1)    | dtt finisher
 *
 * If the gap lives in the mids -> emission/scheduling issue (fixable
 * wiring; msd was never speed-gated vs msg — this probe closes that hole).
 * If it lives in the ingest -> the scatter is structural for THIS ingest
 * design, and the candidate fix is a store-side-permuted ingest (scatter
 * into the HOT PLANE, keep user reads contiguous — the stfn dual).
 *
 * Cells include the r8-ingest chain (4.8.4.4.4.8) — the race cells all
 * ended in 4, so the r8 DIT ingest never raced. Paced, 9 rounds, medians.
 *
 * Build: python build_tuned/build.py --src build_tuned/benches/zturn_dit_stageprobe.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "zturn.h"

static double now_ns(void)
{
#ifdef _WIN32
    static double f = 0.0;
    LARGE_INTEGER t;
    if (f == 0.0) { LARGE_INTEGER q; QueryPerformanceFrequency(&q);
                    f = 1e9 / (double)q.QuadPart; }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * f;
#else
    return 0.0;
#endif
}
static void pace(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#endif
}
static double *az(size_t doubles)
{
#ifdef _WIN32
    return (double *)_aligned_malloc(doubles * sizeof(double), 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, doubles * sizeof(double))) p = NULL;
    return (double *)p;
#endif
}
static int dcmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}
static long rho0(long v, const int *r, int m)
{
    long d[16];
    for (int i = m - 1; i >= 0; i--) { d[i] = v % r[i]; v /= r[i]; }
    long out = 0;
    for (int i = m - 1; i >= 0; i--) out = out * r[i] + d[i];
    return out;
}

typedef struct { int N, nf, chain[8]; int reps; } cell_t;
static const cell_t CELLS[] = {
    { 4096,  6, {4,4,4,4,4,4},     400 },
    { 16384, 6, {4,8,4,4,4,8},     100 },  /* r8 INGEST form for D          */
    { 32768, 7, {4,8,4,4,4,4,4},    50 },
};

enum { P_ING_N, P_ING_D, P_MID_N, P_MID_D, P_FIN_N, P_FIN_D, P_COUNT };
static const char *PN[P_COUNT] =
    { "ingest s0t ", "ingest dtsn", "mids msg   ", "mids msd   ",
      "term  stfn ", "fin   dtt  " };

int main(void)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    printf("\n=== stage decomposition: DIF-natural vs DIT-natural ===\n"
           "paced medians, 9 rounds, alternated. Attribution of the D/N gap.\n");

    for (size_t ci = 0; ci < sizeof CELLS / sizeof CELLS[0]; ci++)
    {
        const cell_t *c = &CELLS[ci];
        const int N = c->N, Rt = c->chain[c->nf - 1], reps = c->reps;
        char cs[32] = {0};
        for (int i = 0, o = 0; i < c->nf; i++)
            o += snprintf(cs + o, sizeof cs - (size_t)o, i ? ".%d" : "%d",
                          c->chain[i]);

        vfft_zturn2_plan_t *p =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        if (!p || !vfft_zturn2_set_natord(p, 1))
        { printf("%d %s REFUSED\n", N, cs); return 1; }
        const size_t OLt = (size_t)N / (size_t)Rt;

        srand(3301 + N);
        double *zin = az(2 * (size_t)N), *zout = az(2 * (size_t)N);
        for (long i = 0; i < 2L * N; i++)
        {
            zin[i] = (double)rand() / RAND_MAX - 0.5;
            p->plane[i] = (double)rand() / RAND_MAX - 0.5;
        }

        double smp[P_COUNT][9];
        for (int r = 0; r < 9; r++)
        {
            for (int ai = 0; ai < P_COUNT; ai++)
            {
                const int a = (r & 1) ? (P_COUNT - 1 - ai) : ai;
                const double t0 = now_ns();
                for (int i = 0; i < reps; i++)
                    switch (a)
                    {
                    case P_ING_N:
                        radix4_z_s0t_r4_fwd_avx2(zin, 0, p->plane, 0, 0, 0,
                            (size_t)N / 4, 0, 0, 0, (size_t)N / 4);
                        break;
                    case P_ING_D:
                        ((Rt == 4) ? radix4_z_dtsn_r4_fwd_avx2
                                   : radix8_z_dtsn_r4_fwd_avx2)(
                            zin, 0, p->plane, 0, p->tzq,
                            (const double *)p->ntb, 0, 0, OLt, 0, OLt);
                        break;
                    case P_MID_N:
                        for (int s = 1; s <= p->nf - 2; s++)
                            ((p->chain[s] == 8) ? radix8_z_msg_fwd_avx2
                                                : radix4_z_msg_fwd_avx2)(
                                p->plane, 0, p->plane, 0, p->twz[s], 0,
                                (unsigned long long)p->D[s],
                                (unsigned long long)p->G[s],
                                0, 0, (unsigned long long)p->D[s]);
                        break;
                    case P_MID_D:
                        for (int s = p->nf - 2; s >= 1; s--)
                            ((p->chain[s] == 8) ? radix8_z_msd_fwd_avx2
                                                : radix4_z_msd_fwd_avx2)(
                                p->plane, 0, p->plane, 0, p->twz[s], 0,
                                (unsigned long long)p->D[s],
                                (unsigned long long)p->G[s],
                                0, 0, (unsigned long long)p->D[s]);
                        break;
                    case P_FIN_N:
                        ((Rt == 4) ? radix4_z_stfn_r4_fwd_avx2
                                   : radix8_z_stfn_r4_fwd_avx2)(
                            p->plane, 0, zout, 0, p->tzq,
                            (const double *)p->ntf, 0, 0, OLt, 0, OLt);
                        break;
                    case P_FIN_D:
                        radix4_z_dtt_r4_fwd_avx2(p->plane, 0, zout, 0, 0, 0,
                            (size_t)N / 4, 0, 0, 0, (size_t)N / 4);
                        break;
                    }
                smp[a][r] = (now_ns() - t0) / reps / 1000.0;
                pace(150);
            }
        }

        printf("\nN=%d  chain=%s  (Rt=%d)\n", N, cs, Rt);
        double med[P_COUNT];
        for (int a = 0; a < P_COUNT; a++)
        {
            qsort(smp[a], 9, sizeof(double), dcmp);
            med[a] = smp[a][4];
            printf("  %s med=%7.2f us   p10..p90 %6.2f..%-6.2f\n",
                   PN[a], med[a], smp[a][1], smp[a][7]);
        }
        printf("  deltas (D-N): ingest %+.2f  mids %+.2f  finisher %+.2f"
               "  | sum N=%.2f D=%.2f  D/N=%.3f\n",
               med[P_ING_D] - med[P_ING_N], med[P_MID_D] - med[P_MID_N],
               med[P_FIN_D] - med[P_FIN_N],
               med[P_ING_N] + med[P_MID_N] + med[P_FIN_N],
               med[P_ING_D] + med[P_MID_D] + med[P_FIN_D],
               (med[P_ING_D] + med[P_MID_D] + med[P_FIN_D])
                   / (med[P_ING_N] + med[P_MID_N] + med[P_FIN_N]));

#ifdef _WIN32
        _aligned_free(zin); _aligned_free(zout);
#endif
        vfft_zturn2_destroy(p);
        pace(1000);
    }
    return 0;
}
