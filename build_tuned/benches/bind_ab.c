/* bind_ab.c — the flat DIT executor's identity check and same-run A/B
 * (2026-09-05). Per cell: the output hashes of the four served paths
 * (natural fwd, conjugate bwd, scrambled fwd, transposed bwd) and both
 * roundtrips; the bound call list against the per-stage reference and the
 * per-block t2cp loop it replaced — bitwise identical, then timed in one
 * run. Exit 1 on any mismatch. Build: build.py --compile --src <this> --vfft */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <windows.h>
#include "../../src/core/oop/il_flatdit.h"

static double now_ns(void)
{
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}
static uint64_t h64(const double *z, size_t n)
{
    const unsigned char *b = (const unsigned char *)z;
    uint64_t h = 1469598103934665603ull;
    size_t i;
    for (i = 0; i < n * sizeof(double); i++) { h ^= b[i]; h *= 1099511628211ull; }
    return h;
}
static void fill(double *x, int N, unsigned seed)
{
    srand(seed);
    for (int j = 0; j < 2 * N; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
}
static double rt_err(const double *x, const double *y, int N)
{
    double m = 0;
    for (int j = 0; j < 2 * N; j++) { double d = fabs(y[j] / N - x[j]); if (d > m) m = d; }
    return m;
}

/* the per-block t2cp loop the bound list replaced: every ONE record whose
 * kernel is the stage's t2cp runs one block per call (OGs = 1) */
static void run_perblock(const vfft_ilfd_plan_t *p, const double *zin, double *zout)
{
    for (int s = 0; s < p->K; s++) {
        const vfft_ilfd_call_t *c = &p->cf[s];
        if (s > 0 && c->op == _ILFD_ONE && c->fn == p->f[s]) {
            const double *in = c->in_sel == _ILFD_STG ? p->stg : zin;
            double *out = c->out_sel == _ILFD_STG ? p->stg : zout;
            const size_t L = c->Gs, nb = c->OGs, tws = (size_t)(p->R[s] - 1) * 8;
            for (size_t b = 0; b < nb; b++)
                c->fn(in + 2 * b * L, 0, out + 2 * b * L, 0, c->tw + b * tws, 0,
                      c->Ls, 0, c->OLs, 1, c->count);
        } else
            _ilfd_run(p, c, 1, zin, zout);
    }
}
static void run_stages(const vfft_ilfd_plan_t *p, const double *zin, double *zout)
{
    for (int s = 0; s < p->K; s++) vfft_ilfd_stage(p, s, zin, zout);
}
static void run_stages_bwd(const vfft_ilfd_plan_t *p, const double *zin, double *zout)
{
    for (int s = 0; s < p->K; s++) vfft_ilfd_stage_bwd(p, s, zin, zout);
}
typedef void (*runner)(const vfft_ilfd_plan_t *, const double *, double *);
static void race3(const vfft_ilfd_plan_t *p, const double *x, double *z, runner a, runner b, runner c,
                  double *ta, double *tb, double *tc)
{
    *ta = *tb = *tc = 1e300;
    for (int r = 0; r < 15; r++)
        for (int k = 0; k < 3; k++) {
            const int arm = (k + r) % 3;
            double t0 = now_ns();
            (arm == 0 ? a : arm == 1 ? b : c)(p, x, z);
            t0 = now_ns() - t0;
            if (arm == 0) { if (t0 < *ta) *ta = t0; } else if (arm == 1) { if (t0 < *tb) *tb = t0; } else { if (t0 < *tc) *tc = t0; }
        }
}

int main(void)
{
    static const struct { int N; int R[8]; int K; const char *forms; } C[] = {
        { 1215,  { 9, 3, 9, 5 }, 4, "t.m.o" },
        { 4095,  { 7, 5, 13, 9 }, 4, "t.t.o" },
        { 6561,  { 9, 9, 9, 9 }, 4, "t.m.o" },
        { 6561,  { 9, 9, 9, 9 }, 4, "t.t.t" },      /* the column-loop tail form at the last stage */
        { 19683, { 9, 9, 9, 9, 3 }, 5, "t.t.n.o" },
        { 59049, { 9, 9, 3, 3, 9, 9 }, 6, "t.t.t.m.o" },
        { 98415, { 9, 9, 9, 3, 9, 5 }, 6, "t.t.t.t.o" },
        { 177147, { 9, 9, 9, 9, 9, 3 }, 6, "" },        /* above the L2 edge: the create defaults' forms */
        { 194481, { 9, 9, 7, 7, 7, 7 }, 6, "" },
        { 245025, { 9, 9, 5, 5, 11, 11 }, 6, "" },
    };
    const int nc = (int)(sizeof C / sizeof C[0]);
    int bad = 0;
    printf("%-6s %-10s | %-16s %-16s %-16s %-16s | rt nat  rt scr\n", "N", "forms", "nat fwd", "conj bwd", "scr fwd", "T bwd");
    for (int i = 0; i < nc; i++) {
        const int N = C[i].N;
        vfft_ilfd_plan_t *p = vfft_ilfd_create_chain(N, C[i].R, C[i].K);
        vfft_ilfd_plan_t *q = vfft_ilfd_create_scr_of(N, C[i].R, C[i].K, C[i].forms, 0);
        double *x = malloc(2 * (size_t)N * 8), *z = malloc(2 * (size_t)N * 8), *y = malloc(2 * (size_t)N * 8);
        uint64_t hf, hb, hs, ht;
        double en, es;
        if (!p || !vfft_ilfd_apply_forms(p, C[i].forms) || !q) { printf("%-6d %-10s | PLAN REFUSED\n", N, C[i].forms); bad++; continue; }
        fill(x, N, 12345u + (unsigned)N);
        vfft_ilfd_execute_fwd(p, x, z); hf = h64(z, 2 * (size_t)N);
        vfft_ilfd_execute_bwd(p, z, y); hb = h64(y, 2 * (size_t)N); en = rt_err(x, y, N);
        vfft_ilfd_execute_fwd(q, x, z); hs = h64(z, 2 * (size_t)N);
        vfft_ilfd_execute_bwd(q, z, y); ht = h64(y, 2 * (size_t)N); es = rt_err(x, y, N);
        printf("%-6d %-10s | %016llx %016llx %016llx %016llx | %.1e %.1e%s\n", N, C[i].forms,
               (unsigned long long)hf, (unsigned long long)hb, (unsigned long long)hs, (unsigned long long)ht, en, es,
               (en < 1e-9 && es < 1e-9) ? "" : "  BAD");
        if (!(en < 1e-9 && es < 1e-9)) bad++;
        {   /* bitwise: the served list == the per-stage reference == the per-block loop */
            double *w = malloc(2 * (size_t)N * 8);
            int same = 1;
            run_stages(p, x, w); same &= (memcmp(w, z, 0) == 0) && (h64(w, 2 * (size_t)N) == hf);
            run_perblock(p, x, w); same &= (h64(w, 2 * (size_t)N) == hf);
            run_stages(q, x, w); same &= (h64(w, 2 * (size_t)N) == hs);
            run_perblock(q, x, w); same &= (h64(w, 2 * (size_t)N) == hs);
            vfft_ilfd_execute_fwd(p, x, z); run_stages_bwd(p, z, w); same &= (h64(w, 2 * (size_t)N) == hb);
            printf("       bound == per-stage == per-block: %s\n", same ? "bitwise" : "DIFFER");
            if (!same) bad++;
            free(w);
        }
        {   /* same-run timing */
            double ta, tb, tc, sa, sb, sc;
            race3(p, x, z, vfft_ilfd_execute_fwd, run_stages, run_perblock, &ta, &tb, &tc);
            race3(q, x, z, vfft_ilfd_execute_fwd, run_stages, run_perblock, &sa, &sb, &sc);
            printf("       nat fwd: bound %8.0f | per-stage %8.0f (%.3fx) | per-block %8.0f (%.3fx)   scr fwd: bound %8.0f | per-block %8.0f (%.3fx)\n",
                   ta, tb, tb / ta, tc, tc / ta, sa, sc, sc / sa);
            vfft_ilfd_execute_fwd(p, x, z);
            race3(p, z, y, vfft_ilfd_execute_bwd, run_stages_bwd, vfft_ilfd_execute_bwd, &ta, &tb, &tc);
            printf("       nat bwd: bound %8.0f | per-stage %8.0f (%.3fx)\n", ta, tb, tb / ta);
        }
        {   /* THE TILE AXIS: every candidate width bitwise-identical to untiled on all four
             * paths, then the candidates timed same-run (alternated, min of 15), forward */
            int cand[12], nw = vfft_ilfd_tw_candidates(p, 2L << 20, cand, 12), a, r, same = 1;
            double *w = malloc(2 * (size_t)N * 8), tn[12], ts[12];
            for (a = 0; a < nw; a++) {
                if (!vfft_ilfd_apply_tw(p, cand[a]) || !vfft_ilfd_apply_tw(q, cand[a])) { same = 0; continue; }
                vfft_ilfd_execute_fwd(p, x, z); same &= (h64(z, 2 * (size_t)N) == hf);
                vfft_ilfd_execute_bwd(p, z, w); same &= (h64(w, 2 * (size_t)N) == hb);
                vfft_ilfd_execute_fwd(q, x, z); same &= (h64(z, 2 * (size_t)N) == hs);
                vfft_ilfd_execute_bwd(q, z, w); same &= (h64(w, 2 * (size_t)N) == ht);
                tn[a] = ts[a] = 1e300;
            }
            for (r = 0; r < 15; r++)
                for (a = 0; a < nw; a++) {
                    const int k = (r & 1) ? nw - 1 - a : a;
                    double t0;
                    vfft_ilfd_apply_tw(p, cand[k]);
                    t0 = now_ns(); vfft_ilfd_execute_fwd(p, x, z); t0 = now_ns() - t0; if (t0 < tn[k]) tn[k] = t0;
                    vfft_ilfd_apply_tw(q, cand[k]);
                    t0 = now_ns(); vfft_ilfd_execute_fwd(q, x, z); t0 = now_ns() - t0; if (t0 < ts[k]) ts[k] = t0;
                }
            printf("       tiles %s:", same ? "bitwise" : "DIFFER");
            for (a = 0; a < nw; a++) printf("  w%d nat %.0f (%.3fx) scr %.0f (%.3fx)", cand[a], tn[a], tn[a] / tn[0], ts[a], ts[a] / ts[0]);
            printf("\n");
            if (!same) bad++;
            free(w);
        }
        vfft_ilfd_destroy(p); vfft_ilfd_destroy(q);
        free(x); free(z); free(y);
    }
    printf(bad ? "=== %d BAD ===\n" : "=== ALL OK (%d) ===\n", bad);
    return bad != 0;
}
