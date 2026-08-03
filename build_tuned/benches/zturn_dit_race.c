/* zturn_dit_race.c — Phase C gate 3: DIT-natural vs DIF-natural, same run.
 *
 * The question the whole DIT campaign exists to answer (adoption rule:
 * canonical VIA THE RACE per cell, never by decree): does the DIT-forward
 * cascade — natural output with ZERO store-side indirection, rho absorbed
 * in the ingest loads — beat the DIF cascade's load-permuted terminator
 * (stfn, the B4 winner at +2.5–5.7 % over scrambled)?
 *
 * ARMS (fwd only — DIT bwd twins are not emitted yet):
 *   S    scrambled DIF, untiled           (baseline)
 *   S'   identical twin                   (CONTROL — noise floor)
 *   N    DIF-natural (stfn), untiled      (incumbent natural)
 *   D    DIT-natural (dtsn->msd->dtt)     (challenger; untiled by design —
 *                                          tiled-DIT is a later campaign)
 *   St   scrambled DIF, tiled+tfuse       (production context ≥8192)
 *   Nt   DIF-natural, tiled               (what D must ULTIMATELY beat
 *                                          where tiling wins)
 *
 * PROTOCOL (thermal-noise directive): pinned 0x4 HIGH, 17 rounds, one
 * sample per arm per round, alternated order, 200 ms arm pace, 1 s cell
 * pace; paced AVERAGE + median + p10..p90. Correctness pre-flight: D == N
 * elementwise (tolerance — different summation order), both natural.
 *
 * Build: python build_tuned/build.py --src build_tuned/benches/zturn_dit_race.c
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "zturn.h"

#define ROUNDS 17

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
static void fz(double *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}
static int dcmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

typedef struct { int N, nf, chain[8]; int reps; } cell_t;
static const cell_t CELLS[] = {
    { 2048,  5, {4,8,4,4,4},       500 },
    { 4096,  6, {4,4,4,4,4,4},     300 },
    { 8192,  6, {4,8,4,4,4,4},     150 },
    { 16384, 7, {4,4,4,4,4,4,4},    75 },
    { 32768, 7, {4,8,4,4,4,4,4},    40 },
    /* r8-TAIL chains — the r8 DIT ingest never raced in v1 (the DIF-banked
     * chains all end in 4), and the stage probe showed dtt BEATS the r8
     * stfn terminator by ~19%. Unbanked chains: legal plan-level probe. */
    { 16384, 6, {4,8,4,4,4,8},      75 },
    { 32768, 6, {4,8,4,4,8,8},      40 },
};

enum { A_S, A_S2, A_N, A_D, A_D2, A_St, A_Nt, A_COUNT };
static const char *ANAME[A_COUNT] = { "S", "S'", "N", "D", "D2", "St", "Nt" };

int main(void)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    printf("\n=== Phase C race: DIT-natural (D) vs DIF-natural (N), fwd ===\n"
           "protocol: %d rounds, alternated order, 200ms arm pace, 1s cell\n"
           "pace, pinned 0x4 HIGH. Verdict column = paced AVG. Control S'/S\n"
           "= noise floor. D is UNTILED BY DESIGN (tiled-DIT = later).\n",
           ROUNDS);

    for (size_t ci = 0; ci < sizeof CELLS / sizeof CELLS[0]; ci++)
    {
        const cell_t *c = &CELLS[ci];
        const int N = c->N, reps = c->reps;
        char cs[32] = {0};
        for (int i = 0, o = 0; i < c->nf; i++)
            o += snprintf(cs + o, sizeof cs - (size_t)o, i ? ".%d" : "%d",
                          c->chain[i]);

        vfft_zturn2_plan_t *ps =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        vfft_zturn2_plan_t *pn =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        vfft_zturn2_plan_t *pd =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        vfft_zturn2_plan_t *pst =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        vfft_zturn2_plan_t *pnt =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        if (!ps || !pn || !pd || !pst || !pnt
            || !vfft_zturn2_set_natord(pn, 1) || !vfft_zturn2_set_natord(pd, 1)
            || !vfft_zturn2_set_natord(pnt, 1))
        { printf("%d %s: create REFUSED\n", N, cs); return 1; }

        long tiled_w = 0;
        {
            long tws[3] = { 2048, 1024, 512 };
            for (int i = 0; i < 3 && !tiled_w; i++)
                if (vfft_zturn2_set_tile_w(pst, 1, tws[i], 1, 0)
                    && vfft_zturn2_set_tile_w(pnt, 1, tws[i], 1, 0))
                    tiled_w = tws[i];
        }

        srand(6367 + N);
        double *zin = az(2 * (size_t)N), *zout = az(2 * (size_t)N);
        double *ref = az(2 * (size_t)N);
        for (long i = 0; i < 2L * N; i++)
            zin[i] = (double)rand() / RAND_MAX - 0.5;

        /* pre-flight: D == N (tolerance; both natural order), and
         * D2 == D memcmp-EXACT (dtso builds the identical plane bit-for-bit
         * — this IS the dtso correctness gate, enforced every race) */
        vfft_zturn2_execute_fwd(pn, zin, ref);
        vfft_zturn2_execute_dit_fwd(pd, zin, zout);
        double xm = 0.0, e = 0.0;
        for (long i = 0; i < 2L * N; i++)
        {
            if (fabs(ref[i]) > xm) xm = fabs(ref[i]);
            if (fabs(zout[i] - ref[i]) > e) e = fabs(zout[i] - ref[i]);
        }
        if (e / xm > 1e-9)
        { printf("%d %s: PRE-FLIGHT FAILED (%.1e) — run void\n",
                 N, cs, e / xm); return 1; }
        {
            double *z2 = az(2 * (size_t)N);
            vfft_zturn2_execute_dit2_fwd(pd, zin, z2);
            const int eq =
                memcmp(z2, zout, 2 * (size_t)N * sizeof(double)) == 0;
            fz(z2);
            if (!eq)
            { printf("%d %s: D2 != D (dtso plane identity broken) — run "
                     "void\n", N, cs); return 1; }
        }

        double smp[A_COUNT][ROUNDS];
        for (int r = 0; r < ROUNDS; r++)
        {
            for (int ai = 0; ai < A_COUNT; ai++)
            {
                const int a = (r & 1) ? (A_COUNT - 1 - ai) : ai;
                if ((a == A_St || a == A_Nt) && !tiled_w)
                { smp[a][r] = 0.0; continue; }
                const double t0 = now_ns();
                for (int i = 0; i < reps; i++)
                    switch (a)
                    {
                    case A_S: case A_S2:
                        vfft_zturn2_execute_fwd(ps, zin, zout); break;
                    case A_N:
                        vfft_zturn2_execute_fwd(pn, zin, zout); break;
                    case A_D:
                        vfft_zturn2_execute_dit_fwd(pd, zin, zout); break;
                    case A_D2:
                        vfft_zturn2_execute_dit2_fwd(pd, zin, zout); break;
                    case A_St:
                        vfft_zturn2_execute_fwd(pst, zin, zout); break;
                    case A_Nt:
                        vfft_zturn2_execute_fwd(pnt, zin, zout); break;
                    }
                smp[a][r] = (now_ns() - t0) / reps / 1000.0;
                pace(200);
            }
        }

        printf("\nN=%d  chain=%s  tiled_w=%s%ld  reps=%d\n",
               N, cs, tiled_w ? "" : "(none) ", tiled_w, reps);
        printf("  %-4s %9s %9s %9s..%-9s\n",
               "arm", "AVG us", "med", "p10", "p90");
        double avg[A_COUNT];
        for (int a = 0; a < A_COUNT; a++)
        {
            if ((a == A_St || a == A_Nt) && !tiled_w) { avg[a] = 0; continue; }
            double srt[ROUNDS];
            memcpy(srt, smp[a], sizeof srt);
            qsort(srt, ROUNDS, sizeof(double), dcmp);
            double s = 0;
            for (int r = 0; r < ROUNDS; r++) s += smp[a][r];
            avg[a] = s / ROUNDS;
            printf("  %-4s %9.2f %9.2f %9.2f..%-9.2f\n",
                   ANAME[a], avg[a], srt[ROUNDS / 2],
                   srt[1], srt[ROUNDS - 2]);
        }
        printf("  natural: N/S=%.3f  D/S=%.3f  D2/S=%.3f  D2/N=%.3f  "
               "D2/D=%.3f  control S'/S=%.3f\n",
               avg[A_N] / avg[A_S], avg[A_D] / avg[A_S],
               avg[A_D2] / avg[A_S], avg[A_D2] / avg[A_N],
               avg[A_D2] / avg[A_D], avg[A_S2] / avg[A_S]);
        if (tiled_w)
            printf("  vs tiled: D2/Nt=%.3f  (untiled DIT-v2 vs tiled "
                   "DIF-natural)\n", avg[A_D2] / avg[A_Nt]);

        fz(zin); fz(zout); fz(ref);
        vfft_zturn2_destroy(ps); vfft_zturn2_destroy(pn);
        vfft_zturn2_destroy(pd); vfft_zturn2_destroy(pst);
        vfft_zturn2_destroy(pnt);
        pace(1000);
    }
    printf("\nadoption rule: D becomes canonical for natural order per cell\n"
           "ONLY where D/N < 1 beyond the control floor — via the race.\n");
    return 0;
}
