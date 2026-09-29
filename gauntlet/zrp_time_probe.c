/* zrp_time_probe.c -- where the real pair's time goes: per (R1, R2) and form
 * (A = n1t + t2h, B = r2z + t2m) the leaf alone, the top alone, and both,
 * forward and backward; beside them the il2p c2c pair at N/2 (its default
 * kernels: blocked at R >= 32) stage by stage, and the zr2c fold pass. One
 * core, best-of-7 medians of ~200 us bursts.
 * Build: python gauntlet/build.py --compile --vfft --src gauntlet/zrp_time_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "zrp.h"
#include "zr2c.h"
#include "race_timing.h"
#ifdef _WIN32
#include <windows.h>
#endif

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (double)(*s >> 8) / (double)(1u << 24) - 0.5;
}

typedef void (*body_fn)(void *);
static double time_body(body_fn f, void *ctx)
{
    double t0 = vfft_now_ns();
    f(ctx);
    double e = vfft_now_ns() - t0;
    int reps = (int)(2.0e5 / (e > 1.0 ? e : 1.0));
    if (reps < 4) reps = 4;
    if (reps > 4000) reps = 4000;
    double best[7];
    for (int r = 0; r < 7; r++)
    {
        double s = vfft_now_ns();
        for (int i = 0; i < reps; i++) f(ctx);
        best[r] = (vfft_now_ns() - s) / reps;
    }
    for (int i = 1; i < 7; i++)
        for (int j = i; j > 0 && best[j] < best[j - 1]; j--) { double t = best[j]; best[j] = best[j - 1]; best[j - 1] = t; }
    return best[3];
}

typedef struct { const vfft_zrp_plan_t *p; const double *x; double *X, *mid; } zctx_t;
static void zrp_leaf(void *v)
{
    zctx_t *c = (zctx_t *)v;
    if (c->p->form == VFFT_ZRP_FORM_A)
        c->p->leaf_f(c->x, 0, c->mid, 0, 0, 0, (size_t)c->p->R1 / 2, 0, (size_t)c->p->R2, 0, (size_t)c->p->R1 / 2);
    else
        c->p->leaf_f(c->x, 0, c->mid, 0, 0, 0, (size_t)c->p->R1, 0, (size_t)c->p->R2, 0, (size_t)c->p->R1);
}
static void zrp_top(void *v) { zctx_t *c = (zctx_t *)v; c->p->top_f(c->mid, 0, c->X, 0, c->p->tw, 0, (size_t)c->p->R2, 0, (size_t)c->p->R2, 0, (size_t)c->p->R2 / 2); }
static void zrp_both(void *v) { zctx_t *c = (zctx_t *)v; vfft_zrp_execute_fwd(c->p, c->x, c->X); }
static void zrp_topb(void *v)
{
    zctx_t *c = (zctx_t *)v;
    const size_t ols = c->p->form == VFFT_ZRP_FORM_A ? (size_t)c->p->R1 / 2 : (size_t)c->p->R1;
    c->p->top_b(c->X, 0, c->mid, 0, c->p->twb, 0, (size_t)c->p->R2, 0, ols, 0, (size_t)c->p->R2 / 2);
}
static void zrp_leafb(void *v)
{
    zctx_t *c = (zctx_t *)v;
    if (c->p->form == VFFT_ZRP_FORM_A)
        c->p->leaf_b(c->mid, 0, c->mid, 0, 0, 0, (size_t)c->p->R1 / 2, 0, (size_t)c->p->R1 / 2, 0, (size_t)c->p->R1 / 2);
    else
        c->p->leaf_b(c->mid, 0, (double *)c->x, 0, 0, 0, (size_t)c->p->R1, 0, (size_t)c->p->R1, 0, (size_t)c->p->R1);
}
static void zrp_bothb(void *v) { zctx_t *c = (zctx_t *)v; vfft_zrp_execute_bwd(c->p, c->X, (double *)c->x); }

typedef struct { const vfft_il2p_plan_t *p; const double *z; double *out, *mid; } ictx_t;
static void il_leaf(void *v) { ictx_t *c = (ictx_t *)v; c->p->leaf_f(c->z, 0, c->mid, 0, 0, 0, (size_t)c->p->R1, 0, (size_t)c->p->R2, 0, (size_t)c->p->R1); }
static void il_mid(void *v) { ictx_t *c = (ictx_t *)v; c->p->mid_f(c->mid, 0, c->out, 0, c->p->tw, 0, (size_t)c->p->R2, 0, (size_t)c->p->R2, 0, (size_t)c->p->R2); }
static void il_both(void *v) { ictx_t *c = (ictx_t *)v; vfft_il2p_execute_fwd(c->p, c->z, c->out); }

typedef struct { int N; const double *z; double *X, *aff; } fctx_t;
static void fold(void *v) { fctx_t *c = (fctx_t *)v; _zr2c_fold_fwd(c->z, c->X, c->aff, c->aff + (c->N / 4 + 1), c->N, 1, (size_t)c->N + 2, (size_t)c->N + 2); }

int main(int argc, char **argv)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), 0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    static const int Ns[] = { 256, 512, 1024, 2048, 4096 };
    int nN = (int)(sizeof Ns / sizeof Ns[0]);
    unsigned seed = 0x1234567u;
    for (int ni = 0; ni < nN; ni++)
    {
        const int N = Ns[ni];
        if (argc > 1 && atoi(argv[1]) != N) continue;
        double *x = (double *)vfft_aligned_alloc((size_t)(2 * N + 2) * sizeof(double));
        double *X = (double *)vfft_aligned_alloc((size_t)(2 * N + 2) * sizeof(double));
        double *mid = (double *)vfft_aligned_alloc((size_t)(2 * N + 2) * sizeof(double));
        for (int i = 0; i < 2 * N + 2; i++) { x[i] = urand(&seed); X[i] = 0; mid[i] = 0; }
        printf("N=%d\n", N);
        for (int form = 0; form < 2; form++)
        for (int R1 = 4; R1 <= 64; R1 += 2)
        {
            if (N % R1) continue;
            int R2 = N / R1;
            if (R2 < 4 || R2 > 64 || (R2 & 1) || !vfft_zrp_pair_ok(N, R1, R2, form)) continue;
            vfft_zrp_plan_t *p = vfft_zrp_create(N, R1, R2, form, 0);
            zctx_t c = { p, x, X, mid };
            double tl = time_body(zrp_leaf, &c), tt = time_body(zrp_top, &c), tb = time_body(zrp_both, &c);
            double ttb = time_body(zrp_topb, &c), tlb = time_body(zrp_leafb, &c), tbb = time_body(zrp_bothb, &c);
            printf("  zrp%c %2d.%-2d  leaf %6.0f  top %6.0f  fwd %6.0f   | topb %6.0f  leafb %6.0f  bwd %6.0f ns\n",
                   form ? 'B' : 'A', R1, R2, tl, tt, tb, ttb, tlb, tbb);
            vfft_zrp_destroy(p);
        }
        for (int R1 = 4; R1 <= 64; R1 *= 2)
        {
            int M = N / 2;
            if (M % R1) continue;
            int R2 = M / R1;
            if (R2 < 4 || R2 > 64) continue;
            vfft_il2p_plan_t *p = vfft_il2p_create(M, R1, R2);
            if (!p) continue;
            ictx_t c = { p, x, X, mid };
            double tl = time_body(il_leaf, &c), tm = time_body(il_mid, &c), tb = time_body(il_both, &c);
            printf("  il2p(N/2) %2d.%-2d  leaf %6.0f  mid %6.0f  fwd %6.0f ns\n", R1, R2, tl, tm, tb);
            vfft_il2p_destroy(p);
        }
        {
            int top = N / 4;
            double *aff = (double *)vfft_aligned_alloc(sizeof(double) * 4u * (size_t)(top + 1));
            _zr2c_init_aff(N, aff, aff + (top + 1), aff + 2 * (top + 1), aff + 3 * (top + 1));
            fctx_t c = { N, x, X, aff };
            printf("  zr2c fold              %6.0f ns\n", time_body(fold, &c));
            vfft_aligned_free(aff);
        }
        vfft_aligned_free(x); vfft_aligned_free(X); vfft_aligned_free(mid);
    }
    return 0;
}
