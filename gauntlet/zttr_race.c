/* zttr_race.c -- ZTURN-T REAL calibrated per real cell: every legal ZTT
 * chain of M = N/2 at the tile widths {untiled, 512, 1024, 2048, 3072}
 * swept once (best-of-5 bursts, the shortlist of four), then the front
 * door's r2c (zr2c, its banked route) raced against the shortlist through
 * the library's race body -- 9 rounds, median, alternated, 200 ms pace --
 * the same-run protocol this machine's thermal noise demands. One line per
 * N. Build: python gauntlet/build.py --compile --vfft --src gauntlet/zttr_race.c
 * Run: zttr_race <wisdom dir> [N] [--c2r]   (--c2r: the c2r twin against the
 * door's c2r, the input a CCE spectrum, the gate the door's c2r output) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
#include "zttr.h"
#include "common/support/race.h"
#include "race_timing.h"
#ifdef _WIN32
#include <windows.h>
#endif

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (double)(*s >> 8) / (double)(1u << 24) - 0.5;
}
static double relerr(const double *a, const double *b, size_t n)
{
    double e = 0, m = 0;
    for (size_t i = 0; i < n; i++) { if (fabs(a[i] - b[i]) > e) e = fabs(a[i] - b[i]); if (fabs(b[i]) > m) m = fabs(b[i]); }
    return m > 0 ? e / m : e;
}

static int g_c2r = 0;
typedef struct { vfft_plan h; vfft_zttr_plan_t *p; double *x, *X; } ctx_t;
static void zttr_exec(const vfft_zttr_plan_t *p, const double *in, double *out)
{
    if (g_c2r) vfft_zttr_execute_bwd(p, in, out); else vfft_zttr_execute_fwd(p, in, out);
}
static void arm_door(void *v) { ctx_t *c = (ctx_t *)v; vfft_execute(c->h, g_c2r ? VFFT_BACKWARD : VFFT_FORWARD, c->x, NULL, c->X, NULL); }
static void arm_zttr(void *v) { ctx_t *c = (ctx_t *)v; zttr_exec(c->p, c->x, c->X); }


typedef struct { int chain[8], nf; size_t tile; double ns; } arm_t;

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (argc < 2) { fprintf(stderr, "usage: zttr_race <wisdom dir> [N]\n"); return 2; }
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const int Ns_pow2[] = { 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536 };
    static const size_t tiles[] = { 0, 512, 1024, 2048, 3072 };
    int only = 0;
    for (int a = 2; a < argc; a++) { if (!strcmp(argv[a], "--c2r")) g_c2r = 1; else only = atoi(argv[a]); }
    const int *Ns = only ? &only : Ns_pow2;
    const int nNs = only ? 1 : (int)(sizeof Ns_pow2 / sizeof Ns_pow2[0]);
    unsigned seed = 0x13579u;
    if (g_c2r) printf("c2r: the fused backward ingest against the door's c2r\n");
    printf("%-6s %-16s %5s  %8s %8s  %s\n", "N", "best chain/tile", "err", "door", "zttr", "door/zttr (shortlist)");
    for (int ni = 0; ni < nNs; ni++)
    {
        const int N = Ns[ni], M = N / 2;
        double *x = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
        double *X = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
        double *Xd = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
        for (int i = 0; i < N; i++) x[i] = urand(&seed);
        x[N] = x[N + 1] = 0;
        vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE; cfg.dims = 1; cfg.n[0] = N;
        cfg.howmany = 1; cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_DEFAULT;
        cfg.rigor = VFFT_PATIENT; cfg.wisdom = W; cfg.nthreads = 1;
        if (g_c2r)
        {   /* the input: a valid CCE spectrum, the door's r2c of the random reals */
            vfft_plan hf = vfft_create(&cfg);
            if (!hf) { printf("%-6d no r2c plan\n", N); continue; }
            vfft_execute(hf, VFFT_FORWARD, x, NULL, Xd, NULL);
            memcpy(x, Xd, (size_t)(N + 2) * sizeof(double));
            vfft_destroy(hf);
            cfg.transform = VFFT_C2R;
        }
        vfft_plan h = vfft_create(&cfg);
        if (!h) { printf("%-6d no door plan\n", N); continue; }
        vfft_execute(h, g_c2r ? VFFT_BACKWARD : VFFT_FORWARD, x, NULL, Xd, NULL);
        const size_t gn = g_c2r ? (size_t)N : (size_t)N + 2;
        /* phase 1: the sweep, best-of-5 bursts per arm */
        int chains[VFFT_ZTTR_MAX_CHAINS][8], nfs[VFFT_ZTTR_MAX_CHAINS];
        const int nc = vfft_zttr_chains(M, chains, nfs, VFFT_ZTTR_MAX_CHAINS);
        arm_t arms[VFFT_ZTTR_MAX_CHAINS * 5];
        int na = 0, bad = 0;
        for (int c = 0; c < nc; c++)
            for (int ti = 0; ti < 5; ti++)
            {
                if (tiles[ti] && (tiles[ti] >= (size_t)M || (size_t)M % tiles[ti] || tiles[ti] < (size_t)(chains[c][0] * chains[c][1]))) continue;
                vfft_zttr_plan_t *p = vfft_zttr_create(N, chains[c], nfs[c], tiles[ti]);
                if (!p) continue;
                zttr_exec(p, x, X);
                if (relerr(X, Xd, gn) > 1e-12) { bad++; vfft_zttr_destroy(p); continue; }
                double t0 = vfft_now_ns(); zttr_exec(p, x, X); double est = vfft_now_ns() - t0;
                int reps = (int)(1.5e5 / (est > 1.0 ? est : 1.0)); if (reps < 2) reps = 2; if (reps > 64) reps = 64;
                double best = 1e30;
                for (int r = 0; r < 5; r++)
                {
                    double s = vfft_now_ns();
                    for (int i = 0; i < reps; i++) zttr_exec(p, x, X);
                    double t = (vfft_now_ns() - s) / reps;
                    if (t < best) best = t;
                }
                memcpy(arms[na].chain, chains[c], sizeof chains[c]); arms[na].nf = nfs[c]; arms[na].tile = tiles[ti]; arms[na].ns = best;
                na++;
                vfft_zttr_destroy(p);
            }
        /* the shortlist of four */
        for (int i = 1; i < na; i++)
            for (int j = i; j > 0 && arms[j].ns < arms[j - 1].ns; j--) { arm_t t = arms[j]; arms[j] = arms[j - 1]; arms[j - 1] = t; }
        const int ns_n = na < 4 ? na : 4;
        if (ns_n == 0) { printf("%-6d no zttr arm builds (%d gate failures)\n", N, bad); vfft_destroy(h); continue; }
        /* phase 2: the paced, alternated race of the door against the shortlist */
        vfft_zttr_plan_t *ps[4];
        ctx_t cx[5];
        vfft_race_arm_t ra[5];
        char names[5][32];
        cx[0].h = h; cx[0].p = NULL; cx[0].x = x; cx[0].X = X;
        ra[0].name = "door"; ra[0].run = arm_door; ra[0].ctx = &cx[0];
        for (int i = 0; i < ns_n; i++)
        {
            ps[i] = vfft_zttr_create(N, arms[i].chain, arms[i].nf, arms[i].tile);
            cx[i + 1].h = h; cx[i + 1].p = ps[i]; cx[i + 1].x = x; cx[i + 1].X = X;
            int off = 0;
            for (int s = 0; s < arms[i].nf; s++) off += snprintf(names[i + 1] + off, sizeof names[i + 1] - (size_t)off, "%s%d", s ? "." : "", arms[i].chain[s]);
            snprintf(names[i + 1] + off, sizeof names[i + 1] - (size_t)off, "/%zu", arms[i].tile);
            ra[i + 1].name = names[i + 1]; ra[i + 1].run = arm_zttr; ra[i + 1].ctx = &cx[i + 1];
        }
        double t0 = vfft_now_ns(); arm_door(&cx[0]); double est = vfft_now_ns() - t0;
        int reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0)); if (reps < 2) reps = 2; if (reps > 64) reps = 64;
        const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
        double ns[5];
        vfft_race_run(&proto, ra, ns_n + 1, ns);
        int best = 1;
        for (int i = 2; i <= ns_n; i++) if (ns[i] < ns[best]) best = i;
        double err;
        zttr_exec(ps[best - 1], x, X);
        err = relerr(X, Xd, gn);
        printf("%-6d %-16s %5.0e  %8.0f %8.0f  %.2fx  (", N, names[best], err, ns[0], ns[best], ns[0] / ns[best]);
        for (int i = 1; i <= ns_n; i++) printf("%s%s %.0f", i > 1 ? ", " : "", names[i], ns[i]);
        printf(")%s%s\n", bad ? "  [gate failures in the sweep]" : "", err > 1e-12 ? "  GATE FAIL" : "");
        for (int i = 0; i < ns_n; i++) vfft_zttr_destroy(ps[i]);
        vfft_destroy(h);
        vfft_aligned_free(x); vfft_aligned_free(X); vfft_aligned_free(Xd);
    }
    return 0;
}
