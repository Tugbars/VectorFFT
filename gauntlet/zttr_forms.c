/* zttr_forms.c -- the ZTT-r terminator forms in ONE race per cell, gated:
 *   door   the front door's r2c (zr2c: the c2c(M) child + the fold pass)
 *   zttr1  ZTT-r with the blocked terminator (tlfhb), in place in X
 *   zttr2  ZTT-r with the natural-order blocked terminator (tlfhc)
 *   child  the public c2c(M) plan alone, x -> Z (the door's child)
 *   walk   the ingest + mids as ZTT-r walks them, x -> W (no last stage)
 *   wtlf   the walk, then the plain last stage in place (= the c2c as ZTT-r runs it)
 *   fold   the fold pass alone, Z -> X, out of place
 *   t1s0..3 tlfhb alone, W -> X, out of place, at stack state 0..3 (p->stk)
 *   t2s0..3 tlfhc alone, the same
 *   tlf    the plain last stage alone, W -> Z
 * at the chain/tile the per-cell calibration chose (gauntlet/zttr_race.c).
 * Every plane sits in ONE arena at a 1088-B skew mod 4096, so no two planes
 * alias in the 4-KB store-load check (an isolated out-of-place pass reads
 * one plane at the offsets it writes the other).
 * Build: python gauntlet/build.py --compile --vfft --src gauntlet/zttr_forms.c
 * Run:   VFFT_ZRP=0 zttr_forms <wisdom dir> [N] */
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
static void last_stage(const vfft_ztt_plan_t *zt, const double *in, double *out)
{
    const size_t L = (size_t)zt->L[zt->nf - 1];
    zt->st_fwd[zt->nf - 1](in, 0, out, 0, zt->tw + zt->twoff[zt->nf - 1], 0, L, 1, L, 0, L);
}

enum { P_X, P_XD, P_X1, P_X2, P_Z, P_W, P_W2, P_WT, P_XF, P_XT1, P_XT2, P_ZT, N_PLANES };
typedef struct {
    int N;
    vfft_plan hr, hc;
    vfft_zttr_plan_t *p1, *p2;        /* the pipelines' plans (state 0) */
    vfft_zttr_plan_t *q1[4], *q2[4];  /* the isolated terminators at states 0..3 */
    double *pl[N_PLANES];
    double *aS, *aC;
} ctx_t;
static void arm_door(void *v)  { ctx_t *c = (ctx_t *)v; vfft_execute(c->hr, VFFT_FORWARD, c->pl[P_X], NULL, c->pl[P_XD], NULL); }
static void arm_zttr1(void *v) { ctx_t *c = (ctx_t *)v; vfft_zttr_execute_fwd(c->p1, c->pl[P_X], c->pl[P_X1]); }
static void arm_zttr2(void *v) { ctx_t *c = (ctx_t *)v; vfft_zttr_execute_fwd(c->p2, c->pl[P_X], c->pl[P_X2]); }
static void arm_child(void *v) { ctx_t *c = (ctx_t *)v; vfft_execute(c->hc, VFFT_FORWARD, c->pl[P_X], NULL, c->pl[P_Z], NULL); }
static void arm_walk(void *v)  { ctx_t *c = (ctx_t *)v; staged_mids(c->p1, c->pl[P_X], c->pl[P_W2]); }
static void arm_wtlf(void *v)  { ctx_t *c = (ctx_t *)v; staged_mids(c->p1, c->pl[P_X], c->pl[P_WT]); last_stage(c->p1->zt, c->pl[P_WT], c->pl[P_WT]); }
static void arm_fold(void *v)  { ctx_t *c = (ctx_t *)v; _zr2c_fold_fwd(c->pl[P_Z], c->pl[P_XF], c->aS, c->aC, c->N, 1, (size_t)c->N + 2, (size_t)c->N + 2); }
#define TERM_ARM(F, S) static void arm_t##F##s##S(void *v) { ctx_t *c = (ctx_t *)v; _zttr_tlfh(c->q##F[S], c->pl[P_W], c->pl[P_XT##F]); }
TERM_ARM(1, 0) TERM_ARM(1, 1) TERM_ARM(1, 2) TERM_ARM(1, 3)
TERM_ARM(2, 0) TERM_ARM(2, 1) TERM_ARM(2, 2) TERM_ARM(2, 3)
#undef TERM_ARM
static void arm_tlf(void *v)   { ctx_t *c = (ctx_t *)v; last_stage(c->p1->zt, c->pl[P_W], c->pl[P_ZT]); }

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
    if (argc < 2) { fprintf(stderr, "usage: zttr_forms <wisdom dir> [N]\n"); return 2; }
    vfft_wisdom *Wis = vfft_wisdom_load(argv[1]);
    const int only = argc > 2 ? atoi(argv[2]) : 0;
    unsigned seed = 0x2468u;
    enum { NA = 16 };
    static const char *names[NA] = { "door", "zttr1", "zttr2", "child", "walk", "wtlf", "fold", "tlf",
                                     "t1s0", "t1s1", "t1s2", "t1s3", "t2s0", "t2s1", "t2s2", "t2s3" };
    static void (*const fns[NA])(void *) = { arm_door, arm_zttr1, arm_zttr2, arm_child, arm_walk, arm_wtlf, arm_fold, arm_tlf,
                                             arm_t1s0, arm_t1s1, arm_t1s2, arm_t1s3, arm_t2s0, arm_t2s1, arm_t2s2, arm_t2s3 };
    printf("%-6s %-12s", "N", "chain/tile");
    for (int a = 0; a < NA; a++) printf(" %6s", names[a]);
    printf("  door/z1 door/z2 wtlf/child z2-walk\n");
    for (int ci = 0; ci < (int)(sizeof cells / sizeof cells[0]); ci++)
    {
        const cell_t *c = &cells[ci];
        const int N = c->N, M = N / 2, NX = N + 2;
        if (only && N != only) continue;
        ctx_t cx; memset(&cx, 0, sizeof cx);
        cx.N = N;
        /* one arena, planes at a 1088-B skew mod 4096 */
        const size_t stride = (((size_t)NX * sizeof(double) + 4095u) & ~(size_t)4095u) + 1088u;
        char *arena = (char *)vfft_aligned_alloc((size_t)N_PLANES * stride + 4096u);
        for (int k = 0; k < N_PLANES; k++) { cx.pl[k] = (double *)(arena + (size_t)k * stride); memset(cx.pl[k], 0, (size_t)NX * sizeof(double)); }
        for (int i = 0; i < NX; i++) cx.pl[P_X][i] = urand(&seed);
        vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE; cfg.dims = 1; cfg.n[0] = M;
        cfg.howmany = 1; cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.order = VFFT_ORDER_NATURAL;
        cfg.rigor = VFFT_PATIENT; cfg.wisdom = Wis; cfg.nthreads = 1;
        cx.hc = vfft_create(&cfg);
        cfg.transform = VFFT_R2C; cfg.n[0] = N; cfg.order = VFFT_ORDER_DEFAULT;
        cx.hr = vfft_create(&cfg);
        cx.p1 = vfft_zttr_create(N, c->chain, c->nf, c->tile);
        cx.p2 = vfft_zttr_create(N, c->chain, c->nf, c->tile);
        if (!cx.hc || !cx.hr || !cx.p1 || !cx.p2) { printf("%-6d plans: c2c %p r2c %p zttr %p %p\n", N, (void *)cx.hc, (void *)cx.hr, (void *)cx.p1, (void *)cx.p2); continue; }
        cx.p1->blocked = 1; cx.p2->blocked = 2;
        for (int st = 0; st < 4; st++)
        {
            cx.q1[st] = vfft_zttr_create(N, c->chain, c->nf, c->tile); cx.q1[st]->blocked = 1; cx.q1[st]->stk = st;
            cx.q2[st] = vfft_zttr_create(N, c->chain, c->nf, c->tile); cx.q2[st]->blocked = 2; cx.q2[st]->stk = st;
        }
        cx.aS = (double *)vfft_aligned_alloc(2u * (size_t)(N / 4 + 1) * sizeof(double));
        cx.aC = cx.aS + (N / 4 + 1);
        {
            double *bS = (double *)malloc(2u * (size_t)(N / 4 + 1) * sizeof(double));
            _zr2c_init_aff(N, cx.aS, cx.aC, bS, bS + (N / 4 + 1));
            free(bS);
        }
        arm_child(&cx);                                   /* Z: the child's plane */
        staged_mids(cx.p1, cx.pl[P_X], cx.pl[P_W]);       /* W: the mids' plane */
        for (int a = 0; a < NA; a++) fns[a](&cx);
        const double g1 = maxrel(cx.pl[P_XD], cx.pl[P_X1], NX), g2 = maxrel(cx.pl[P_XD], cx.pl[P_X2], NX);
        const double gf = maxrel(cx.pl[P_XD], cx.pl[P_XF], NX), gt1 = maxrel(cx.pl[P_XD], cx.pl[P_XT1], NX);
        for (int st = 0; st < 4; st++)   /* every state must reproduce the door */
        {
            memset(cx.pl[P_XT1], 0, (size_t)NX * sizeof(double)); fns[8 + st](&cx);
            memset(cx.pl[P_XT2], 0, (size_t)NX * sizeof(double)); fns[12 + st](&cx);
            if (maxrel(cx.pl[P_XD], cx.pl[P_XT1], NX) > 1e-12 || maxrel(cx.pl[P_XD], cx.pl[P_XT2], NX) > 1e-12)
                printf("%-6d state %d FAILS the gate\n", N, st);
        }
        const double gt2 = maxrel(cx.pl[P_XD], cx.pl[P_XT2], NX), gl = maxrel(cx.pl[P_Z], cx.pl[P_ZT], N);
        const double gw = maxrel(cx.pl[P_W], cx.pl[P_W2], N), gwt = maxrel(cx.pl[P_Z], cx.pl[P_WT], N);
        const int ok = g1 < 1e-12 && g2 < 1e-12 && gf < 1e-12 && gt1 < 1e-12 && gt2 < 1e-12 && gl < 1e-12 && gw < 1e-12 && gwt < 1e-12;
        printf("%-6d gates vs door: zttr1 %.1e zttr2 %.1e fold %.1e term1 %.1e term2 %.1e | vs child: tlf %.1e wtlf %.1e | walk %.1e  %s\n",
               N, g1, g2, gf, gt1, gt2, gl, gwt, gw, ok ? "PASS" : "FAIL");
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
        for (int a = 0; a < NA; a++) printf(" %6.0f", ns[a]);
        printf("  %7.3f %7.3f %10.3f %7.0f\n", ns[0] / ns[1], ns[0] / ns[2], ns[5] / ns[3], ns[2] - ns[4]);
        fflush(stdout);
        vfft_destroy(cx.hc); vfft_destroy(cx.hr); vfft_zttr_destroy(cx.p1); vfft_zttr_destroy(cx.p2);
        for (int st = 0; st < 4; st++) { vfft_zttr_destroy(cx.q1[st]); vfft_zttr_destroy(cx.q2[st]); }
        vfft_aligned_free(arena); vfft_aligned_free(cx.aS);
    }
    return 0;
}
