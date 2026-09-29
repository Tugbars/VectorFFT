/* zttr_parts.c -- where ZTURN-T REAL loses: one paced, alternated race per
 * N over five arms at the c2c cell's banked chain/tile of M = N/2:
 *   c2c    the public c2c(M) plan (the door's child as the library builds it)
 *   fused  vfft_ztt_execute_fwd on our own ZTT plan at (chain, tile)
 *   staged the ingest + mids as zttr walks them, then the plain tlf
 *   zttr   the same walk, then the Hermitian terminator (= ZTT-real)
 *   door   the front door's r2c (zr2c: the child + the fold pass)
 * Build: python gauntlet/build.py --compile --vfft --src gauntlet/zttr_parts.c
 * Run: zttr_parts <wisdom dir> [N] */
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

typedef struct { vfft_plan hc, hr; vfft_zttr_plan_t *p; double *x, *X; } ctx_t;
static void arm_c2c(void *v)   { ctx_t *c = (ctx_t *)v; vfft_execute(c->hc, VFFT_FORWARD, c->x, NULL, c->X, NULL); }
static void arm_fused(void *v) { ctx_t *c = (ctx_t *)v; vfft_ztt_execute_fwd(c->p->zt, c->x, c->X); }
static void staged_mids(const vfft_zttr_plan_t *p, const double *x, double *W)
{
    const vfft_ztt_plan_t *zt = p->zt;
    const int nf = zt->nf;
    const size_t M = (size_t)p->M, tile = zt->tile;
    const double *tw = zt->tw;
    zt->st_fwd[0](x, 0, W, 0, 0, (const double *)zt->rb, (size_t)zt->ncol, 0, 0, 0, (size_t)zt->ncol);
    if (tile)
        for (size_t t = 0; t < M / tile; t++)
        {
            double *B = W + t * tile * 2;
            for (int s = 1; s < nf - 1; s++)
            {
                const size_t RL = (size_t)zt->L[s] * (size_t)zt->chain[s];
                if (tile % RL == 0)
                    zt->st_fwd[s](B, 0, B, 0, tw + zt->twoff[s], 0, (size_t)zt->L[s], tile / RL, 0, 0, (size_t)zt->L[s]);
            }
        }
    for (int s = 1; s < nf - 1; s++)
    {
        const size_t RL = (size_t)zt->L[s] * (size_t)zt->chain[s];
        if (!tile || tile % RL)
            zt->st_fwd[s](W, 0, W, 0, tw + zt->twoff[s], 0, (size_t)zt->L[s], (size_t)zt->Gs[s], 0, 0, (size_t)zt->L[s]);
    }
}
static void arm_staged(void *v)
{
    ctx_t *c = (ctx_t *)v;
    const vfft_ztt_plan_t *zt = c->p->zt;
    const size_t L = (size_t)zt->L[zt->nf - 1];
    staged_mids(c->p, c->x, c->X);
    zt->st_fwd[zt->nf - 1](c->X, 0, c->X, 0, zt->tw + zt->twoff[zt->nf - 1], 0, L, 1, L, 0, L);
}
static void arm_zttr(void *v) { ctx_t *c = (ctx_t *)v; vfft_zttr_execute_fwd(c->p, c->x, c->X); }
static void arm_door(void *v) { ctx_t *c = (ctx_t *)v; vfft_execute(c->hr, VFFT_FORWARD, c->x, NULL, c->X, NULL); }

typedef struct { int N; int nf; int chain[7]; size_t tile; } cell_t;
static const cell_t cells[] = {
    { 2048,  4, { 4, 8, 8, 4 }, 0 },
    { 4096,  4, { 8, 8, 8, 4 }, 1024 },
    { 8192,  4, { 8, 8, 8, 8 }, 2048 },
    { 16384, 5, { 8, 4, 8, 4, 8 }, 1024 },
};

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (argc < 2) { fprintf(stderr, "usage: zttr_parts <wisdom dir> [N]\n"); return 2; }
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    const int only = argc > 2 ? atoi(argv[2]) : 0;
    unsigned seed = 0x2468u;
    printf("%-6s %-14s %8s %8s %8s %8s %8s\n", "N", "chain/tile", "c2c(M)", "fused", "staged", "zttr", "door");
    for (int ci = 0; ci < (int)(sizeof cells / sizeof cells[0]); ci++)
    {
        const cell_t *c = &cells[ci];
        const int N = c->N, M = N / 2;
        if (only && N != only) continue;
        double *x = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
        double *X = (double *)vfft_aligned_alloc((size_t)(N + 2) * sizeof(double));
        for (int i = 0; i < N + 2; i++) x[i] = urand(&seed);
        vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE; cfg.dims = 1; cfg.n[0] = M;
        cfg.howmany = 1; cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_NATURAL;
        cfg.rigor = VFFT_PATIENT; cfg.wisdom = W; cfg.nthreads = 1;
        vfft_plan hc = vfft_create(&cfg);
        cfg.transform = VFFT_R2C; cfg.n[0] = N; cfg.order = VFFT_ORDER_DEFAULT;
        vfft_plan hr = vfft_create(&cfg);
        vfft_zttr_plan_t *p = vfft_zttr_create(N, c->chain, c->nf, c->tile);
        if (!hc || !hr || !p) { printf("%-6d plans: c2c %p r2c %p zttr %p\n", N, (void *)hc, (void *)hr, (void *)p); continue; }
        ctx_t cx = { hc, hr, p, x, X };
        const vfft_race_arm_t arms[5] = {
            { "c2c", arm_c2c, &cx }, { "fused", arm_fused, &cx }, { "staged", arm_staged, &cx },
            { "zttr", arm_zttr, &cx }, { "door", arm_door, &cx } };
        double t0 = vfft_now_ns(); arm_door(&cx); double est = vfft_now_ns() - t0;
        int reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0)); if (reps < 2) reps = 2; if (reps > 64) reps = 64;
        const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
        double ns[5];
        vfft_race_run(&proto, arms, 5, ns);
        char cs[40]; int off = 0;
        for (int s = 0; s < c->nf; s++) off += snprintf(cs + off, sizeof cs - (size_t)off, "%s%d", s ? "." : "", c->chain[s]);
        snprintf(cs + off, sizeof cs - (size_t)off, "/%zu", c->tile);
        printf("%-6d %-14s %8.0f %8.0f %8.0f %8.0f %8.0f\n", N, cs, ns[0], ns[1], ns[2], ns[3], ns[4]);
        vfft_destroy(hc); vfft_destroy(hr); vfft_zttr_destroy(p);
        vfft_aligned_free(x); vfft_aligned_free(X);
    }
    return 0;
}
