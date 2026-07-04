/* natorder_g1_leaf_alias.c — GATE G1: are the n1_oop LEAF codelets alias-safe
 * when dst==src (free in-place NATURAL c2c at N<=128)?
 *
 * Method: for every N in 2..128 with a leaf fn and K in {4,8,12,23,64}:
 *   1. fill src deterministically, run leaf src->dst (separate) = reference
 *   2. re-fill identical src, run leaf src->src (aliased)
 *   3. compare BIT-EXACT (memcmp)
 * Also: harness sanity check (one cell vs naive DFT), and a cheap-fix bench
 * at N=64 K=64 (copy src to a stack buffer, leaf buffer->src) with QPC
 * best-of-5, vs separate-dst and vs direct-aliased.
 *
 * Build: python build.py --src test/natorder_g1_leaf_alias.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "executor.h"
#include "planner.h"
#include "oop_plan.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static vfft_proto_registry_t REG;

static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0)
        exit(1);
    return p;
}
static void afree(double *p) { vfft_proto_aligned_free(p); }

static void fill(double *re, double *im, size_t tot, int N, size_t K)
{
    srand(1000003u + 131u * (unsigned)N + (unsigned)K);
    for (size_t t = 0; t < tot; t++)
    {
        re[t] = (double)rand() / RAND_MAX - 0.5;
        im[t] = (double)rand() / RAND_MAX - 0.5;
    }
}

/* one aliased-vs-separate cell; returns 0 = bit-exact, 1 = mismatch */
static int alias_cell(int N, size_t K, vfft_oop11_fn f)
{
    size_t tot = (size_t)N * K;
    double *sre = ad(tot), *sim = ad(tot);
    double *dre = ad(tot), *dim = ad(tot);
    double *are = ad(tot), *aim = ad(tot);

    fill(sre, sim, tot, N, K);
    f(sre, sim, dre, dim, 0, 0, K, 1, K, 1, K); /* reference: separate dst */

    fill(are, aim, tot, N, K); /* identical refill */
    f(are, aim, are, aim, 0, 0, K, 1, K, 1, K); /* aliased: dst==src */

    int bad = (memcmp(dre, are, tot * sizeof(double)) != 0) ||
              (memcmp(dim, aim, tot * sizeof(double)) != 0);
    if (bad)
    {
        double md = 0.0;
        for (size_t t = 0; t < tot; t++)
        {
            double er = fabs(dre[t] - are[t]), ei = fabs(dim[t] - aim[t]);
            if (er > md) md = er;
            if (ei > md) md = ei;
        }
        printf("    [N=%d K=%zu aliased max-abs-diff=%.3e]\n", N, K, md);
    }

    afree(sre); afree(sim); afree(dre); afree(dim); afree(are); afree(aim);
    return bad;
}

/* harness sanity: reference separate-dst output vs naive DFT at one cell */
static double naive_check(int N, size_t K, vfft_oop11_fn f)
{
    size_t tot = (size_t)N * K;
    double *sre = ad(tot), *sim = ad(tot), *dre = ad(tot), *dim = ad(tot);
    fill(sre, sim, tot, N, K);
    f(sre, sim, dre, dim, 0, 0, K, 1, K, 1, K);
    double md = 0.0;
    for (size_t b = 0; b < K; b++)
        for (int k = 0; k < N; k++)
        {
            double xr = 0, xi = 0;
            for (int n = 0; n < N; n++)
            {
                double ang = -2.0 * M_PI * (double)((long)n * k % N) / (double)N;
                double c = cos(ang), s = sin(ang);
                double ar = sre[(size_t)n * K + b], ai = sim[(size_t)n * K + b];
                xr += ar * c - ai * s;
                xi += ar * s + ai * c;
            }
            double er = fabs(dre[(size_t)k * K + b] - xr);
            double ei = fabs(dim[(size_t)k * K + b] - xi);
            double e = er > ei ? er : ei;
            if (e > md) md = e;
        }
    afree(sre); afree(sim); afree(dre); afree(dim);
    return md;
}

/* ---- timing: QPC best-of-5 samples of `reps` calls ---- */
static double qpc_freq(void)
{
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    return (double)f.QuadPart;
}

typedef void (*bench_body)(void *);
static double best_of_5(bench_body body, void *ctx, int reps)
{
    double fq = qpc_freq(), best = 1e300;
    for (int s = 0; s < 5; s++)
    {
        LARGE_INTEGER a, b;
        QueryPerformanceCounter(&a);
        for (int r = 0; r < reps; r++)
            body(ctx);
        QueryPerformanceCounter(&b);
        double t = (double)(b.QuadPart - a.QuadPart) / fq / (double)reps;
        if (t < best) best = t;
    }
    return best;
}

typedef struct
{
    vfft_oop11_fn f;
    double *sre, *sim, *dre, *dim;
    size_t K, tot;
} bctx_t;

static void body_sep(void *v)
{
    bctx_t *c = (bctx_t *)v;
    c->f(c->sre, c->sim, c->dre, c->dim, 0, 0, c->K, 1, c->K, 1, c->K);
}
static void body_alias(void *v)
{
    bctx_t *c = (bctx_t *)v;
    c->f(c->sre, c->sim, c->sre, c->sim, 0, 0, c->K, 1, c->K, 1, c->K);
}
static void body_fix(void *v) /* cheap fix: copy to stack buffer, leaf buf->src */
{
    bctx_t *c = (bctx_t *)v;
    double buf_re[64 * 64], buf_im[64 * 64]; /* N=64 K=64 stack block, 64KB */
    memcpy(buf_re, c->sre, c->tot * sizeof(double));
    memcpy(buf_im, c->sim, c->tot * sizeof(double));
    c->f(buf_re, buf_im, c->sre, c->sim, 0, 0, c->K, 1, c->K, 1, c->K);
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), 1ull);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    vfft_proto_registry_init(&REG);

    /* harness sanity: the calling convention must produce a correct DFT */
    {
        vfft_oop11_fn f8 = vfft_oop_leaf_fn(8);
        if (!f8) { printf("SANITY: no leaf fn for N=8\n"); return 2; }
        double e = naive_check(8, 23, f8);
        printf("# sanity naive-DFT N=8 K=23 err=%.2e %s\n", e,
               e < 1e-12 ? "ok" : "<HARNESS-BROKEN>");
        if (e >= 1e-12) return 2;
    }

    size_t Ks[] = {4, 8, 12, 23, 64};
    const int NK = (int)(sizeof Ks / sizeof Ks[0]);

    int n_leaf = 0, n_pass = 0, n_fail = 0;
    int pass_list[130], fail_list[130];
    int np = 0, nf = 0;

    printf("# GATE G1: n1_oop LEAF alias-safety (dst==src) — bit-exact vs separate-dst\n");
    printf("# K sweep: {4,8,12,23,64}\n");
    for (int N = 2; N <= 128; N++)
    {
        vfft_oop11_fn f = vfft_oop_leaf_fn(N);
        if (!f) continue;
        n_leaf++;
        int bad = 0;
        char detail[128] = "";
        for (int i = 0; i < NK; i++)
        {
            int m = alias_cell(N, Ks[i], f);
            if (m)
            {
                bad++;
                char t[24];
                snprintf(t, sizeof t, " K=%zu", Ks[i]);
                strncat(detail, t, sizeof(detail) - strlen(detail) - 1);
            }
        }
        if (bad) { fail_list[nf++] = N; n_fail++; printf("N=%-3d FAIL%s\n", N, detail); }
        else     { pass_list[np++] = N; n_pass++; }
    }

    printf("\nLEAF fns found: %d\n", n_leaf);
    printf("PASS-all-K (%d):", n_pass);
    for (int i = 0; i < np; i++) printf(" %d", pass_list[i]);
    printf("\nFAIL-any-K (%d):", n_fail);
    for (int i = 0; i < nf; i++) printf(" %d", fail_list[i]);
    printf("\n");

    /* ---- cheap-fix bench at N=64 K=64 (always run; alias body only if safe) ---- */
    {
        const int N = 64;
        const size_t K = 64;
        vfft_oop11_fn f = vfft_oop_leaf_fn(N);
        size_t tot = (size_t)N * K;
        bctx_t c;
        c.f = f; c.K = K; c.tot = tot;
        c.sre = ad(tot); c.sim = ad(tot); c.dre = ad(tot); c.dim = ad(tot);
        fill(c.sre, c.sim, tot, N, K);

        /* correctness of the cheap fix at this cell (bit-exact vs separate) */
        {
            double *rre = ad(tot), *rim = ad(tot);
            fill(c.sre, c.sim, tot, N, K);
            f(c.sre, c.sim, rre, rim, 0, 0, K, 1, K, 1, K);
            fill(c.sre, c.sim, tot, N, K);
            body_fix(&c); /* result lands in c.sre/c.sim */
            int m = (memcmp(rre, c.sre, tot * 8) != 0) || (memcmp(rim, c.sim, tot * 8) != 0);
            printf("\n# cheap-fix (stack-buffer copy) correctness N=64 K=64: %s\n",
                   m ? "MISMATCH" : "bit-exact");
            afree(rre); afree(rim);
        }

        /* order-neutralized round-robin: 15 rounds x (sep,fix,alias) with
         * rotating start, best sample per body (canonical-bench style). */
        int reps = 2000;
        bench_body bodies[3] = {body_sep, body_fix, body_alias};
        double best[3] = {1e300, 1e300, 1e300};
        for (int r = 0; r < 100; r++) { body_sep(&c); body_fix(&c); body_alias(&c); } /* warm */
        for (int round = 0; round < 15; round++)
            for (int j = 0; j < 3; j++)
            {
                int i = (round + j) % 3;
                double fq = qpc_freq();
                LARGE_INTEGER a, b;
                QueryPerformanceCounter(&a);
                for (int r = 0; r < reps; r++) bodies[i](&c);
                QueryPerformanceCounter(&b);
                double t = (double)(b.QuadPart - a.QuadPart) / fq / (double)reps;
                if (t < best[i]) best[i] = t;
            }
        double t_sep = best[0], t_fix = best[1], t_alias = best[2];

        printf("# N=64 K=64 QPC best-of-15 rotating rounds, %d reps/sample\n", reps);
        printf("t_separate_dst   = %9.1f ns\n", t_sep * 1e9);
        printf("t_aliased_direct = %9.1f ns  (%.3fx of separate)\n",
               t_alias * 1e9, t_alias / t_sep);
        printf("t_copyfix        = %9.1f ns  (%.3fx of separate, overhead %+.1f%%)\n",
               t_fix * 1e9, t_fix / t_sep, (t_fix / t_sep - 1.0) * 100.0);

        afree(c.sre); afree(c.sim); afree(c.dre); afree(c.dim);
    }

    printf("\nG1 %s: %d/%d leaf codelets bit-exact when dst==src\n",
           n_fail == 0 ? "PASS" : "PARTIAL/FAIL", n_pass, n_leaf);
    return n_fail ? 1 : 0;
}
