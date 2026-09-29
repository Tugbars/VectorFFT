/* zttr_c2r.c -- the c2r twin (the backward fold fused into the ZTT's
 * backward ingest) in ONE race per cell against the door, gated:
 *   door   the front door's c2r (zr2c: the fold pass into scratch, the c2c(M) child backward)
 *   zttr   ZTT-r c2r: the fused ingest t0h, the backward mids and last stage, in the output
 *   child  the public c2c(M) plan backward alone, Zhat -> x (the door's child)
 *   fold   the backward fold pass alone, X -> Zhat, out of place
 *   t0tp   the plain backward ingest alone, Zhat -> the plane
 *   t0h    the fused ingest alone, X -> the plane
 * at the chain/tile the r2c calibration chose (gauntlet/zttr_race.c). One
 * arena at a 1088-B skew; the fused ingest runs through the stack-aligning
 * entry at the plan's state.
 * Build: python gauntlet/build.py --compile --vfft --src gauntlet/zttr_c2r.c
 * Run:   VFFT_ZRP=0 zttr_c2r <wisdom dir> [N] */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
#include "zttr.h"
#include "zr2c.h"
#include "common/support/race.h"
#include "common/support/race_timing.h"
#ifdef _WIN32
#include <windows.h>
#endif

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (double)(*s >> 8) / (double)(1u << 24) - 0.5;
}

enum { P_X, P_XD, P_XZ, P_ZH, P_XC, P_ZF, P_W0, P_W1, N_PLANES };
typedef struct {
    int N;
    vfft_plan hr, hc;          /* the door's c2r; the public c2c(M) (backward for the child arm) */
    vfft_zttr_plan_t *p;
    double *pl[N_PLANES];      /* X (CCE), the door's x, zttr's x, the fold's Zhat (prepared), the child's x,
                                * the fold arm's output, t0tp's plane, t0h's plane */
    double *bS, *bC;           /* the door's backward tables */
} ctx_t;
static void arm_door(void *v)  { ctx_t *c = (ctx_t *)v; vfft_execute(c->hr, VFFT_BACKWARD, c->pl[P_X], NULL, c->pl[P_XD], NULL); }
static void arm_zttr(void *v)  { ctx_t *c = (ctx_t *)v; vfft_zttr_execute_bwd(c->p, c->pl[P_X], c->pl[P_XZ]); }
static void arm_child(void *v) { ctx_t *c = (ctx_t *)v; vfft_execute(c->hc, VFFT_BACKWARD, c->pl[P_ZH], NULL, c->pl[P_XC], NULL); }
static void arm_fold(void *v)  { ctx_t *c = (ctx_t *)v; _zr2c_fold_bwd(c->pl[P_X], c->pl[P_ZF], c->bS, c->bC, c->N, 1, (size_t)c->N + 2, (size_t)c->N); }
static void arm_t0tp(void *v)
{
    ctx_t *c = (ctx_t *)v;
    const vfft_ztt_plan_t *zt = c->p->zt;
    zt->st_bwd[0](c->pl[P_ZH], 0, c->pl[P_W0], 0, 0, (const double *)zt->rb, (size_t)zt->ncol, 0, 0, 0, (size_t)zt->ncol);
}
static void arm_t0h(void *v)
{
    ctx_t *c = (ctx_t *)v;
    _zttr_call_aligned(c->p->zt->chain[0] == 4 ? (_zttr_term_fn)_zttr_t0h4 : (_zttr_term_fn)_zttr_t0h8, c->p, c->pl[P_X], c->pl[P_W1]);
}

static double maxrel(const double *a, const double *b, int n)
{
    double sc = 0, e = 0;
    for (int i = 0; i < n; i++) { if (fabs(a[i]) > sc) sc = fabs(a[i]); double d = fabs(a[i] - b[i]); if (d > e) e = d; }
    return sc > 0 ? e / sc : e;
}

typedef struct { int N; int nf; int chain[7]; size_t tile; } cell_t;
static const cell_t cells[] = {
    { 2048,  4, { 4, 8, 4, 8 },    0 },
    { 4096,  4, { 8, 8, 4, 8 },    1024 },
    { 8192,  4, { 8, 8, 8, 8 },    2048 },
    { 32768, 5, { 8, 8, 8, 4, 8 }, 2048 },
};

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    if (argc < 2) { fprintf(stderr, "usage: zttr_c2r <wisdom dir> [N]\n"); return 2; }
    vfft_wisdom *Wis = vfft_wisdom_load(argv[1]);
    const int only = argc > 2 ? atoi(argv[2]) : 0;
    unsigned seed = 0x2468u;
    enum { NA = 6 };
    static const char *names[NA] = { "door", "zttr", "child", "fold", "t0tp", "t0h" };
    static void (*const fns[NA])(void *) = { arm_door, arm_zttr, arm_child, arm_fold, arm_t0tp, arm_t0h };
    printf("%-6s %-12s", "N", "chain/tile");
    for (int a = 0; a < NA; a++) printf(" %7s", names[a]);
    printf("  door/zttr  (t0h-t0tp)/fold  zttr-child\n");
    for (int ci = 0; ci < (int)(sizeof cells / sizeof cells[0]); ci++)
    {
        const cell_t *c = &cells[ci];
        const int N = c->N, M = N / 2, NX = N + 2;
        if (only && N != only) continue;
        ctx_t cx; memset(&cx, 0, sizeof cx);
        cx.N = N;
        const size_t stride = (((size_t)NX * sizeof(double) + 4095u) & ~(size_t)4095u) + 1088u;
        char *arena = (char *)vfft_aligned_alloc((size_t)N_PLANES * stride + 4096u);
        for (int k = 0; k < N_PLANES; k++) { cx.pl[k] = (double *)(arena + (size_t)k * stride); memset(cx.pl[k], 0, (size_t)NX * sizeof(double)); }
        /* a valid CCE spectrum: the door's r2c of random reals */
        vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_R2C; cfg.placement = VFFT_OUTOFPLACE; cfg.dims = 1; cfg.n[0] = N;
        cfg.howmany = 1; cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_DEFAULT;
        cfg.rigor = VFFT_PATIENT; cfg.wisdom = Wis; cfg.nthreads = 1;
        {
            vfft_plan hf = vfft_create(&cfg);
            double *xr = (double *)vfft_aligned_alloc((size_t)NX * sizeof(double));
            for (int i = 0; i < NX; i++) xr[i] = urand(&seed);
            if (!hf) { printf("%-6d no r2c plan\n", N); continue; }
            vfft_execute(hf, VFFT_FORWARD, xr, NULL, cx.pl[P_X], NULL);
            vfft_destroy(hf); vfft_aligned_free(xr);
        }
        cfg.transform = VFFT_C2R;
        cx.hr = vfft_create(&cfg);
        cfg.transform = VFFT_C2C; cfg.n[0] = M; cfg.order = VFFT_ORDER_NATURAL;
        cx.hc = vfft_create(&cfg);
        cx.p = vfft_zttr_create(N, c->chain, c->nf, c->tile);
        if (!cx.hc || !cx.hr || !cx.p) { printf("%-6d plans: c2r %p c2c %p zttr %p\n", N, (void *)cx.hr, (void *)cx.hc, (void *)cx.p); continue; }
        cx.bS = (double *)vfft_aligned_alloc(4u * (size_t)(N / 4 + 1) * sizeof(double));
        cx.bC = cx.bS + (N / 4 + 1);
        _zr2c_init_aff(N, cx.bS + 2 * (N / 4 + 1), cx.bS + 3 * (N / 4 + 1), cx.bS, cx.bC);
        arm_fold(&cx);                                            /* Zhat, as the door folds it */
        memcpy(cx.pl[P_ZH], cx.pl[P_ZF], (size_t)N * sizeof(double));
        for (int a = 0; a < NA; a++) fns[a](&cx);
        const double gz = maxrel(cx.pl[P_XD], cx.pl[P_XZ], N), gc = maxrel(cx.pl[P_XD], cx.pl[P_XC], N);
        const double gi = maxrel(cx.pl[P_W0], cx.pl[P_W1], N);
        const int ok = gz < 1e-12 && gc < 1e-12 && gi < 1e-12;
        printf("%-6d gates: zttr vs door %.1e  child vs door %.1e  t0h vs t0tp %.1e  %s\n", N, gz, gc, gi, ok ? "PASS" : "FAIL");
        if (!ok) continue;
        vfft_race_arm_t arms[NA];
        for (int a = 0; a < NA; a++) { arms[a].name = names[a]; arms[a].run = fns[a]; arms[a].ctx = &cx; }
        double t0 = vfft_now_ns(); arm_door(&cx); double est = vfft_now_ns() - t0;
        int reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0)); if (reps < 2) reps = 2; if (reps > 64) reps = 64;
        const vfft_race_proto_t proto = { 15, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
        double ns[NA];
        vfft_race_run(&proto, arms, NA, ns);
        char cs[40]; int off = 0;
        for (int s = 0; s < c->nf; s++) off += snprintf(cs + off, sizeof cs - (size_t)off, "%s%d", s ? "." : "", c->chain[s]);
        snprintf(cs + off, sizeof cs - (size_t)off, "/%zu", c->tile);
        printf("%-6d %-12s", N, cs);
        for (int a = 0; a < NA; a++) printf(" %7.0f", ns[a]);
        printf("  %9.3f  %14.3f  %10.0f\n", ns[0] / ns[1], (ns[5] - ns[4]) / ns[3], ns[1] - ns[2]);
        fflush(stdout);
        vfft_destroy(cx.hc); vfft_destroy(cx.hr); vfft_zttr_destroy(cx.p);
        vfft_aligned_free(arena); vfft_aligned_free(cx.bS);
    }
    return 0;
}
