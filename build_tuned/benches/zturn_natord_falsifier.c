/* zturn_natord_falsifier.c — B4: the natural-order claim, falsified or made.
 *
 * CLAIM under test (docs/roadmap/cascade_natural_inplace_plan.md, resting on
 * P0c + the MKL RE): a load-permuted terminator delivers natural order for
 * ~0–3% over the scrambled cascade, where the incumbent — scrambled cascade
 * + a separate PURE-cycle reorder pass — pays +16–36%. If the natural arm
 * does not clearly beat the incumbent here, the reframe is wrong and the
 * docs say so.
 *
 * ARMS (all in the SAME run, alternated order per round):
 *   fwd:  S   scrambled cascade, untiled            (baseline)
 *         S'  identical twin of S                   (CONTROL: S'/S = noise
 *                                                    floor; deltas under it
 *                                                    are NOT results)
 *         P   S + in-place 64B-block cycle reorder  (the incumbent, rebuilt
 *             on zout (natural[u] <- scr[tf[u]])     for the interleaved
 *             via precomputed cycles                 layout — the CHEAPEST
 *                                                    possible pass, so a
 *                                                    fair-to-generous foe)
 *         N   natord cascade (stfn terminator)      (the challenger)
 *         St  scrambled TILED mids + tfuse          (what natural gives up:
 *         Nt  natord TILED mids (tfuse forced 0)     the fused-terminator
 *                                                    arm; natterm_spec §4)
 *   bwd:  Sb  scrambled bwd (scrambled spectrum in)
 *         Pb  gather-reorder natural->scrambled into scratch + Sb
 *         Nb  natord bwd (natural spectrum in, stfbn)
 *
 * PROTOCOL (thermal-noise directive, Tugbars 2026-08-03): pinned core 2
 * (mask 0x4), HIGH priority, >=17 rounds, ONE sample per arm per round,
 * arm order alternated per round, 200 ms pace between arms, 1 s between
 * cells; report paced AVERAGE + median + p10/p90 per arm. Correctness
 * pre-flight per cell: P == N and Pb == Nb == Sb, memcmp EXACT.
 *
 * Plan-level on purpose (create_chain): this is an internal A/B probe, not a
 * calibration and not an MKL bench — the front-door rule binds those.
 *
 * Build: python build_tuned/build.py --src build_tuned/benches/zturn_natord_falsifier.c
 */
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

/* conv0 (MSF) digit reversal — local copy, independent of the driver. */
static long rho0(long v, const int *r, int m)
{
    long d[16];
    for (int i = m - 1; i >= 0; i--) { d[i] = v % r[i]; v /= r[i]; }
    long out = 0;
    for (int i = m - 1; i >= 0; i--) out = out * r[i] + d[i];
    return out;
}

/* ── the incumbent's fwd pass: in-place cycle reorder on 64 B blocks.
 * natural[u] = scr[tf[u]] per lane; cycles of u -> tf[u] precomputed once
 * (planning side), stored flat as [len, i0, i1, ...]* terminated by 0.     */
static long *build_cycles(const size_t *tf, long M)
{
    char *vis = (char *)calloc((size_t)M, 1);
    long *cy = (long *)malloc(sizeof(long) * (size_t)(2 * M + 2));
    long o = 0;
    for (long u = 0; u < M; u++)
    {
        if (vis[u] || (long)tf[u] == u) { vis[u] = 1; continue; }
        long len_at = o++;
        long n = 0, v = u;
        while (!vis[v]) { vis[v] = 1; cy[o++] = v; n++; v = (long)tf[v]; }
        cy[len_at] = n;
    }
    cy[o] = 0;
    free(vis);
    return cy;
}
static void cycle_pass(double *z, const long *cy, const size_t *tf,
                       int Rt, size_t OLs)
{
    for (int l = 0; l < Rt; l++)
    {
        double *base = z + 2 * (size_t)l * OLs;
        const long *c = cy;
        while (*c)
        {
            const long n = *c++;
            double tmp[8];
            memcpy(tmp, base + 8 * c[0], sizeof tmp);
            for (long i = 0; i < n - 1; i++)
                memcpy(base + 8 * c[i], base + 8 * (long)tf[c[i]],
                       sizeof tmp);
            memcpy(base + 8 * c[n - 1], tmp, sizeof tmp);
            c += n;
        }
    }
}
/* the incumbent's bwd pass: OOP gather natural -> scrambled scratch.
 * scr[block t] = nat[block rho(t)] = nat[tb[t]] per lane.                  */
static void gather_pass(const double *zn, double *scr, const size_t *tb,
                        int Rt, size_t OLs)
{
    const long M = (long)(OLs / 4);
    for (int l = 0; l < Rt; l++)
    {
        const double *nb = zn + 2 * (size_t)l * OLs;
        double *sb = scr + 2 * (size_t)l * OLs;
        for (long t = 0; t < M; t++)
            memcpy(sb + 8 * t, nb + 8 * (long)tb[t], 8 * sizeof(double));
    }
}

typedef struct { int N, nf, chain[8]; int reps; } cell_t;
static const cell_t CELLS[] = {
    { 4096,  6, {4,4,4,4,4,4},     300 },
    { 8192,  6, {4,8,4,4,4,4},     150 },
    { 16384, 7, {4,4,4,4,4,4,4},    75 },
    { 32768, 7, {4,8,4,4,4,4,4},    40 },
};

enum { A_S, A_S2, A_P, A_N, A_St, A_Nt, A_Sb, A_Pb, A_Nb, A_COUNT };
static const char *ANAME[A_COUNT] =
    { "S", "S'", "P", "N", "St", "Nt", "Sb", "Pb", "Nb" };

int main(void)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    printf("\n=== B4 falsifier: natural terminator vs scrambled+PURE-cycle ===\n"
           "protocol: %d rounds, 1 sample/arm/round, alternated order,\n"
           "200ms arm pace, 1s cell pace, pinned 0x4 HIGH. AVG is the verdict\n"
           "column; spread = p10..p90 of rounds. Control S'/S = noise floor.\n",
           ROUNDS);

    for (size_t ci = 0; ci < sizeof CELLS / sizeof CELLS[0]; ci++)
    {
        const cell_t *c = &CELLS[ci];
        const int N = c->N, Rt = c->chain[c->nf - 1], reps = c->reps;
        char cs[32] = {0};
        for (int i = 0, o = 0; i < c->nf; i++)
            o += snprintf(cs + o, sizeof cs - (size_t)o, i ? ".%d" : "%d",
                          c->chain[i]);

        vfft_zturn2_plan_t *ps =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        vfft_zturn2_plan_t *pn =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        vfft_zturn2_plan_t *pst =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        vfft_zturn2_plan_t *pnt =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        if (!ps || !pn || !pst || !pnt || !vfft_zturn2_set_natord(pn, 1)
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

        const size_t OLs = (size_t)N / (size_t)Rt;
        const long M = (long)(OLs / 4);
        size_t *tf = (size_t *)malloc(sizeof(size_t) * (size_t)M);
        size_t *tb = (size_t *)malloc(sizeof(size_t) * (size_t)M);
        for (long t = 0; t < M; t++)
        {
            const long r = rho0(t, c->chain + 1, c->nf - 2);
            tf[r] = (size_t)t;
            tb[t] = (size_t)r;
        }
        long *cyc = build_cycles(tf, M);

        srand(4241 + N);
        double *zin = az(2 * (size_t)N), *zout = az(2 * (size_t)N);
        double *zn  = az(2 * (size_t)N), *zs = az(2 * (size_t)N);
        double *scr = az(2 * (size_t)N), *ref = az(2 * (size_t)N);
        for (long i = 0; i < 2L * N; i++)
        {
            zin[i] = (double)rand() / RAND_MAX - 0.5;
            zs[i]  = (double)rand() / RAND_MAX - 0.5;
        }
        for (int l = 0; l < Rt; l++)          /* zn = natural arr. of zs    */
            for (long t = 0; t < M; t++)
                memcpy(zn + 2 * (l * (long)OLs) + 8 * (long)tb[t],
                       zs + 2 * (l * (long)OLs) + 8 * t,
                       8 * sizeof(double));

        /* ── correctness pre-flight: the arms must agree EXACTLY ───────── */
        vfft_zturn2_execute_fwd(pn, zin, ref);            /* N              */
        vfft_zturn2_execute_fwd(ps, zin, zout);           /* P              */
        cycle_pass(zout, cyc, tf, Rt, OLs);
        int pre_ok = memcmp(zout, ref, 2 * (size_t)N * sizeof(double)) == 0;
        vfft_zturn2_execute_bwd(ps, zs, ref);             /* Sb             */
        vfft_zturn2_execute_bwd(pn, zn, zout);            /* Nb             */
        pre_ok &= memcmp(zout, ref, 2 * (size_t)N * sizeof(double)) == 0;
        gather_pass(zn, scr, tb, Rt, OLs);                /* Pb             */
        vfft_zturn2_execute_bwd(ps, scr, zout);
        pre_ok &= memcmp(zout, ref, 2 * (size_t)N * sizeof(double)) == 0;
        if (!pre_ok)
        { printf("%d %s: PRE-FLIGHT FAILED — arms disagree, run void\n",
                 N, cs); return 1; }

        /* ── timed rounds ──────────────────────────────────────────────── */
        double smp[A_COUNT][ROUNDS];
        const int narm = tiled_w ? A_COUNT : A_COUNT;  /* tiled arms always
                                                        * present; skipped
                                                        * below if !tiled_w */
        for (int r = 0; r < ROUNDS; r++)
        {
            for (int ai = 0; ai < narm; ai++)
            {
                const int a = (r & 1) ? (narm - 1 - ai) : ai;
                if ((a == A_St || a == A_Nt) && !tiled_w)
                { smp[a][r] = 0.0; continue; }
                const double t0 = now_ns();
                for (int i = 0; i < reps; i++)
                    switch (a)
                    {
                    case A_S: case A_S2:
                        vfft_zturn2_execute_fwd(ps, zin, zout); break;
                    case A_P:
                        vfft_zturn2_execute_fwd(ps, zin, zout);
                        cycle_pass(zout, cyc, tf, Rt, OLs); break;
                    case A_N:
                        vfft_zturn2_execute_fwd(pn, zin, zout); break;
                    case A_St:
                        vfft_zturn2_execute_fwd(pst, zin, zout); break;
                    case A_Nt:
                        vfft_zturn2_execute_fwd(pnt, zin, zout); break;
                    case A_Sb:
                        vfft_zturn2_execute_bwd(ps, zs, zout); break;
                    case A_Pb:
                        gather_pass(zn, scr, tb, Rt, OLs);
                        vfft_zturn2_execute_bwd(ps, scr, zout); break;
                    case A_Nb:
                        vfft_zturn2_execute_bwd(pn, zn, zout); break;
                    }
                smp[a][r] = (now_ns() - t0) / reps / 1000.0;   /* us */
                pace(200);
            }
        }

        /* ── report: AVG (the verdict), median, p10..p90 ───────────────── */
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
        printf("  fwd: P/S=%.3f  N/S=%.3f  control S'/S=%.3f%s\n",
               avg[A_P] / avg[A_S], avg[A_N] / avg[A_S],
               avg[A_S2] / avg[A_S],
               tiled_w ? "" : "  (no legal tile width)");
        if (tiled_w)
            printf("  tiled: Nt/St=%.3f  (natural's cost WITH tiling, tfuse "
                   "lost included)\n", avg[A_Nt] / avg[A_St]);
        printf("  bwd: Pb/Sb=%.3f  Nb/Sb=%.3f\n",
               avg[A_Pb] / avg[A_Sb], avg[A_Nb] / avg[A_Sb]);

        free(tf); free(tb); free(cyc);
        fz(zin); fz(zout); fz(zn); fz(zs); fz(scr); fz(ref);
        vfft_zturn2_destroy(ps); vfft_zturn2_destroy(pn);
        vfft_zturn2_destroy(pst); vfft_zturn2_destroy(pnt);
        pace(1000);
    }
    printf("\nverdict key: the claim holds if N/S sits near the control and\n"
           "P/S sits clearly above both; deltas under the control are noise.\n");
    return 0;
}
