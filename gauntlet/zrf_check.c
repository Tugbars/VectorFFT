/* zrf_check.c -- the real flat DIT (il/real/zrf.h) as an engine, before the
 * door: per odd N, every chain x the split-body switch x the tile budgets
 * (0, 64, 256, 1024; a budget no level takes is the untiled plan, skipped):
 *   gate   r2c against a long double real DFT, c2r(r2c(x)) = N x, both out
 *          of place and in place;
 *   race   the best three chains per direction (a quick best-of-5 pass picks
 *          them) against the door's serving plan for the cell, the library
 *          race body, paced, alternated.
 * Build: python gauntlet/build.py --compile --vfft --src gauntlet/zrf_check.c
 * Run:   zrf_check <scratch wisdom dir> [N ...] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include "vfft.h"
#include "zrf.h"
#include "common/support/race.h"
#include "common/support/race_timing.h"
#ifdef _WIN32
#include <windows.h>
#endif

#define MAXC 24
typedef struct { vfft_zrf_plan_t *p; int bwd; const double *in; double *out; } zarm_t;
static void zarm_run(void *v) { zarm_t *a = (zarm_t *)v; if (a->bwd) vfft_zrf_execute_bwd(a->p, a->in, a->out); else vfft_zrf_execute_fwd(a->p, a->in, a->out); }
typedef struct { vfft_plan h; vfft_dir_t dir; double *in, *out; } darm_t;
static void darm_run(void *v) { darm_t *a = (darm_t *)v; vfft_execute(a->h, a->dir, a->in, NULL, a->out, NULL); }

static vfft_plan door(vfft_wisdom *W, int N, int c2r)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = c2r ? VFFT_C2R : VFFT_R2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_DEFAULT;
    cfg.rigor = VFFT_PATIENT; cfg.wisdom = W; cfg.nthreads = 1;
    return vfft_create(&cfg);
}
static double maxrel(const double *a, const double *b, size_t n)
{
    double sc = 0, e = 0;
    for (size_t i = 0; i < n; i++) { if (fabs(b[i]) > sc) sc = fabs(b[i]); const double d = fabs(a[i] - b[i]); if (d > e) e = d; }
    return sc > 0 ? e / sc : e;
}
static double quick(void (*run)(void *), void *ctx, int reps)
{
    double best = 1e300;
    for (int r = 0; r < 5; r++)
    {
        const double t0 = vfft_now_ns();
        for (int i = 0; i < reps; i++) run(ctx);
        const double t = (vfft_now_ns() - t0) / reps;
        if (t < best) best = t;
    }
    return best;
}

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (argc < 2) { fprintf(stderr, "usage: zrf_check <scratch wisdom dir> [N ...]\n"); return 2; }
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int dflt[] = { 45, 63, 75, 105, 135, 225, 243, 315, 405, 525, 625, 675, 729, 945, 1125, 1215, 1365, 1375, 1575, 2025 };
    int Ns[64], nn = 0, fails = 0;
    for (int i = 2; i < argc && nn < 64; i++) Ns[nn++] = atoi(argv[i]);
    if (!nn) for (nn = 0; nn < (int)(sizeof dflt / sizeof dflt[0]); nn++) Ns[nn] = dflt[nn];
    const long double PI = 3.14159265358979323846264338327950288L;
    unsigned seed = 0x2f6e2b1u;
    for (int ni = 0; ni < nn; ni++)
    {
        const int N = Ns[ni];
        const size_t NX = (size_t)N + 9, nb = (size_t)N / 2 + 1;
        int ch[MAXC][VFFT_ILFD_MAX_K], len[MAXC], dropped = 0;
        const int nc = vfft_zrf_chains(N, ch, len, MAXC, &dropped);
        double *x = (double *)_aligned_malloc(NX * 8, 64), *ref = (double *)_aligned_malloc(NX * 8, 64);
        double *X = (double *)_aligned_malloc(NX * 8, 64), *y = (double *)_aligned_malloc(NX * 8, 64), *b = (double *)_aligned_malloc(NX * 8, 64);
        vfft_zrf_plan_t *pl[8 * MAXC]; char nm[8 * MAXC][48]; double qf[8 * MAXC], qb[8 * MAXC];
        static const int tiles[4] = { 0, 64, 256, 1024 };
        int np = 0;
        if (!nc) { printf("N=%d: no chain\n", N); continue; }
        for (size_t i = 0; i < NX; i++) { seed = seed * 1664525u + 1013904223u; x[i] = (double)(seed >> 8) / (double)(1u << 24) - 0.5; }
        for (size_t k = 0; k < nb; k++)
        {
            long double re = 0, im = 0;
            for (int n = 0; n < N; n++)
            {
                const long double a = -2.0L * PI * (long double)(((long long)k * n) % N) / (long double)N;
                re += (long double)x[n] * cosl(a); im += (long double)x[n] * sinl(a);
            }
            ref[2 * k] = (double)re; ref[2 * k + 1] = (double)im;
        }
        for (int ci = 0; ci < nc; ci++)
            for (int nomsz = 0, any = 0; nomsz < 2; nomsz++)
            for (int ti = 0; ti < 4; ti++)
            {
                vfft_zrf_plan_t *p = vfft_zrf_create(N, ch[ci], len[ci], nomsz, tiles[ti]);
                char cs[40];
                vfft_zrf_chain_str(ch[ci], len[ci], cs, sizeof cs);
                if (!p) { if (!nomsz && !ti) printf("N=%d %s: not built\n", N, cs); continue; }
                if (ti && !vfft_zrf_tiled(p)) { vfft_zrf_destroy(p); continue; }
                if (!nomsz && !ti)
                    for (int j = 0; j < p->J; j++) for (int s = 1; s <= p->lv[j].ns; s++) any |= p->lv[j].fd->msz[s];
                else if (nomsz && !any)
                {   /* no stage takes the split body: the twin is the same plan */
                    vfft_zrf_destroy(p);
                    continue;
                }
                if (ti) snprintf(nm[np], sizeof nm[np], "%s%s/w%d", cs, nomsz ? "/t" : "", tiles[ti]);
                else snprintf(nm[np], sizeof nm[np], "%s%s", cs, nomsz ? "/t" : "");
                double e1, e2, e3, e4;
                memset(X, 0, NX * 8); memset(y, 0, NX * 8);
                X[2 * nb] = 777.0; y[N] = 777.0;
                vfft_zrf_execute_fwd(p, x, X);
                e1 = maxrel(X, ref, 2 * nb);
                vfft_zrf_execute_bwd(p, X, y);
                for (int i = 0; i < N; i++) y[i] /= (double)N;
                e2 = maxrel(y, x, (size_t)N);
                memcpy(b, x, NX * 8); vfft_zrf_execute_fwd(p, b, b);
                e3 = maxrel(b, ref, 2 * nb);
                vfft_zrf_execute_bwd(p, b, b);
                for (int i = 0; i < N; i++) b[i] /= (double)N;
                e4 = maxrel(b, x, (size_t)N);
                const int ok = e1 < 1e-12 && e2 < 1e-12 && e3 < 1e-12 && e4 < 1e-12 && X[2 * nb] == 777.0 && y[N] == 777.0;
                if (!ok)
                {
                    fails++;
                    printf("N=%d %-14s *** FAIL *** r2c %.1e roundtrip %.1e | in place %.1e %.1e | guards %g %g\n",
                           N, nm[np], e1, e2, e3, e4, X[2 * nb], y[N]);
                    vfft_zrf_destroy(p);
                    continue;
                }
                pl[np++] = p;
            }
        if (!np) { printf("N=%d: no plan passed\n", N); continue; }
        {
            vfft_plan hf = door(W, N, 0), hb = door(W, N, 1);
            memcpy(X, ref, 2 * nb * 8);
            for (int d = 0; d < 2; d++)
            {
                vfft_plan h = d ? hb : hf;
                double *q = d ? qb : qf;
                zarm_t za[8 * MAXC];
                int best[3] = { -1, -1, -1 }, reps;
                for (int i = 0; i < np; i++) { za[i].p = pl[i]; za[i].bwd = d; za[i].in = d ? X : x; za[i].out = d ? y : b; }
                { const double t0 = vfft_now_ns(); zarm_run(&za[0]); const double est = vfft_now_ns() - t0;
                  reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0)); if (reps < 2) reps = 2; if (reps > 4096) reps = 4096; }
                for (int i = 0; i < np; i++) q[i] = quick(zarm_run, &za[i], reps);
                for (int r = 0; r < 3; r++)
                {
                    int bi = -1;
                    for (int i = 0; i < np; i++)
                        if (i != best[0] && i != best[1] && (bi < 0 || q[i] < q[bi])) bi = i;
                    best[r] = bi;
                    if (np <= r + 1) break;
                }
                vfft_race_arm_t arms[4]; double ns[4]; int na = 0;
                darm_t da = { h, d ? VFFT_BACKWARD : VFFT_FORWARD, d ? X : x, d ? y : b };
                if (h) { arms[na].name = "door"; arms[na].run = darm_run; arms[na].ctx = &da; na++; }
                const int a0 = na;
                for (int r = 0; r < 3; r++) if (best[r] >= 0 && (r == 0 || best[r] != best[r - 1]))
                { arms[na].name = nm[best[r]]; arms[na].run = zarm_run; arms[na].ctx = &za[best[r]]; na++; }
                const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
                vfft_race_run(&proto, arms, na, ns);
                printf("N=%-5d %s  %d plans ok%s | door %7.0f ns |", N, d ? "c2r" : "r2c", np, dropped ? " (pool capped)" : "", h ? ns[0] : 0.0);
                for (int a = a0; a < na; a++) printf(" %s %.0f (%.2fx)", arms[a].name, ns[a], h ? ns[0] / ns[a] : 0.0);
                printf("\n");
                fflush(stdout);
            }
            if (hf) vfft_destroy(hf);
            if (hb) vfft_destroy(hb);
        }
        if (getenv("ZRF_PARTS"))
        {   /* the forward's parts on its quickest plan, each alone in a hot loop */
            int bi = -1;
            for (int i = 0; i < np; i++) if (!vfft_zrf_tiled(pl[i]) && (bi < 0 || qf[i] < qf[bi])) bi = i;
            const vfft_zrf_plan_t *p = pl[bi];
            const int reps = 2000;
            double whole, sum = 0;
            { double best = 1e300; for (int r = 0; r < 7; r++) { const double t0 = vfft_now_ns(); for (int i = 0; i < reps; i++) vfft_zrf_execute_fwd(p, x, b); const double t = (vfft_now_ns() - t0) / reps; if (t < best) best = t; } whole = best; }
            printf("   parts %s (r2c %.0f ns):", nm[bi], whole);
            const double *src = x;
            for (int j = 0; j < p->J; j++)
            {
                const _zrf_level_t *lv = &p->lv[j];
                double best = 1e300;
                for (int r = 0; r < 7; r++) { const double t0 = vfft_now_ns(); for (int i = 0; i < reps; i++) lv->lf(src, NULL, lv->plane, NULL, NULL, NULL, lv->D, 0, lv->D, 0, lv->D); const double t = (vfft_now_ns() - t0) / reps; if (t < best) best = t; }
                printf(" | L%d leaf r%d %.0f", j, lv->R, best); sum += best;
                for (int s = 0; s < lv->ns; s++)
                {
                    best = 1e300;
                    for (int r = 0; r < 7; r++) { const double t0 = vfft_now_ns(); for (int i = 0; i < reps; i++) _ilfd_call(lv->fd, &lv->cf[s], 1, lv->plane, lv->plane); const double t = (vfft_now_ns() - t0) / reps; if (t < best) best = t; }
                    printf(" s%d r%d %.0f", s + 1, lv->fd->R[s + 1], best); sum += best;
                }
                best = 1e300;
                for (int r = 0; r < 7; r++) { const double t0 = vfft_now_ns(); for (int i = 0; i < reps; i++) _zrf_sweep(lv, b, 0, lv->nlb); const double t = (vfft_now_ns() - t0) / reps; if (t < best) best = t; }
                printf(" sweep %.0f", best); sum += best;
                src = lv->plane;
            }
            {
                double best = 1e300;
                for (int r = 0; r < 7; r++) { const double t0 = vfft_now_ns(); for (int i = 0; i < reps; i++) p->mf(src, NULL, b, NULL, NULL, NULL, 1, 0, p->MJ, 0, 1); const double t = (vfft_now_ns() - t0) / reps; if (t < best) best = t; }
                printf(" | mono %d %.0f", (int)p->NJ, best); sum += best;
            }
            printf(" | sum %.0f\n", sum);
        }
        for (int i = 0; i < np; i++) vfft_zrf_destroy(pl[i]);
        _aligned_free(x); _aligned_free(ref); _aligned_free(X); _aligned_free(y); _aligned_free(b);
    }
    printf("%s\n", fails ? "FAILURES" : "ALL PASS");
    return fails ? 1 : 0;
}
