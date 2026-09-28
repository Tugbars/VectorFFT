/* avx2_tail_mech_bench.c -- WHY a masked pass wins or loses at AVX2 (2026-09-28).
 *
 * At AVX-512 one k-masked zmm pass won when 3 of 4 columns were left: it does
 * nearly a full vector's work in one pass, where the ladder ran two. The
 * argument rests on two costs, measured here at AVX2 on the leftover column:
 *   WIDTH  a full ymm pass vs the xmm pass (same DAG): overrun / narrow and
 *          blk_overrun / blk_narrow -- at AVX-512 a zmm pass cost 1.5-1.8
 *          narrow passes;
 *   MASK   the masked ymm pass vs the same pass unmasked: masked / overrun and
 *          blk_masked / blk_overrun.
 * Arms (gen_avx2_arms.sh with the odd frame fix on, VFFT_CX_ODDROLL=1
 * VFFT_CX_ODDP1=1, so the bulk loop is the same stable code in every arm):
 * narrow, masked, overrun (the monolithic DAG), blk_narrow, blk_masked,
 * blk_overrun (the odd blocked passes, radix >= 9, n1 / t2). The overrun arms
 * read and write one column past the end: cost floors, never kernels.
 *
 * Per (kind, radix, count c in 1, 5): every arm at c and at c-1 (no remainder
 * runs at an even count), each with the caller's stack at shift 0 and 16 (the
 * two states a Win64 caller can hand the kernel: ymm spills are 16-B aligned);
 * narrow at c+1 for the column cost. tail = (t(c) - t(c-1)) / column, per
 * state; reported: the worse state. L1-hot (one buffer set): the costs are
 * compute. Core 2 HIGH + sibling guard, 21 rounds alternating, min of 5
 * batches of ~80 us, median; 200 ms between cells. Accuracy: every non-overrun
 * arm against a direct DFT. Build: build_avx2_arms.sh-style (see the log header). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <malloc.h>
#include <math.h>
#include <windows.h>
#include "sibling_guard.h"

#define KARGS const double *, const double *, double *, double *, const double *, const double *, \
              size_t, size_t, size_t, size_t, size_t
typedef void (*kfn)(KARGS);
enum { A_NAR, A_MSK, A_OVR, A_BNAR, A_BMSK, A_BOVR, NARM };
static const char *ARMN[NARM] = { "narrow", "masked", "overrun", "blk_nar", "blk_msk", "blk_ovr" };
enum { K_N1, K_N1T, K_T2, NKIND };
static const char *KINDN[NKIND] = { "n1", "n1t", "t2" };

#define SMALL(X) X(3) X(5) X(7)
#define BIG(X) X(9) X(11) X(13) X(15) X(17) X(19) X(21) X(23) X(25) X(27) X(29) X(31) X(37) X(41) X(43) X(47)
#define KD(R, K, A) void radix##R##_z_##K##_fwd_avx2_##A(KARGS);
#define D3(R, K) KD(R, K, narrow) KD(R, K, masked) KD(R, K, overrun)
#define D6(R, K) D3(R, K) KD(R, K, blk_narrow) KD(R, K, blk_masked) KD(R, K, blk_overrun)
#define DS(R) D3(R, n1) D3(R, n1t) D3(R, t2)
#define DB(R) D6(R, n1) D3(R, n1t) D6(R, t2)
SMALL(DS) BIG(DB)
#define F(R, K, A) radix##R##_z_##K##_fwd_avx2_##A
#define E3(R, K) { F(R, K, narrow), F(R, K, masked), F(R, K, overrun), 0, 0, 0 }
#define E6(R, K) { F(R, K, narrow), F(R, K, masked), F(R, K, overrun), F(R, K, blk_narrow), F(R, K, blk_masked), F(R, K, blk_overrun) }
#define ES(R) { R, { E3(R, n1), E3(R, n1t), E3(R, t2) } },
#define EB(R) { R, { E6(R, n1), E3(R, n1t), E6(R, t2) } },
static const struct { int R; kfn f[NKIND][NARM]; } KT[] = { SMALL(ES) BIG(EB) };
enum { NRAD = sizeof KT / sizeof *KT, ROUNDS = 21 };

static double now_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
static int cmpd(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}
static double *vtw2(int R, int cols, int N)
{
    const size_t np = ((size_t)cols + 1) / 2;
    double *t = _aligned_malloc(sizeof(double) * (np * (R - 1) * 8 + 8), 64);
    for (size_t pp = 0; pp < np; pp++)
        for (int l = 1; l < R; l++)
        {
            double *r = t + (pp * (R - 1) + (l - 1)) * 8;
            for (int j = 0; j < 2; j++)
            {
                const double a = -2.0 * 3.14159265358979323846 * (double)l * (double)(2 * pp + j) / (double)N;
                r[2 * j] = r[2 * j + 1] = cos(a);
                r[4 + 2 * j] = -sin(a);
                r[4 + 2 * j + 1] = sin(a);
            }
        }
    return t;
}
typedef struct { kfn f; int kind, R, c; const double *tw; const double *x; double *y; } call_t;
static inline void call1(const call_t *u)
{
    const size_t c = (size_t)u->c, ols = u->kind == K_N1T ? (size_t)u->R : c;
    u->f(u->x, 0, u->y, 0, u->tw, 0, c, 0, ols, 0, c);
}
static __attribute__((noinline)) double timed(const call_t *u, int reps)
{
    double best = 1e30;
    for (int b = 0; b < 5; b++)
    {
        const double t0 = now_ns();
        for (int i = 0; i < reps; i++) call1(u);
        const double t = (now_ns() - t0) / reps;
        if (t < best) best = t;
    }
    return best;
}
static __attribute__((noinline)) double shifted(int shift, const call_t *u, int reps)
{
    volatile char *pad = alloca(64 + shift);
    pad[0] = 0;
    return timed(u, reps);
}
/* max relative error of one kernel call against a direct DFT per column */
static double accuracy(kfn f, int kind, int R, int c, const double *tw)
{
    const size_t n = (size_t)R * c, ols = kind == K_N1T ? (size_t)R : (size_t)c;
    double *x = _aligned_malloc(16 * n + 128, 64), *y = _aligned_malloc(16 * n + 128, 64);
    for (size_t j = 0; j < 2 * n; j++) x[j] = sin(0.61 * j + 0.3) - 0.4 * cos(2.1 * j);
    memset(y, 0, 16 * n + 128);
    f(x, 0, y, 0, tw, 0, (size_t)c, 0, ols, 0, (size_t)c);
    double err = 0, mag = 0;
    for (int k = 0; k < c; k++)
        for (int m = 0; m < R; m++)
        {
            double re = 0, im = 0;
            for (int l = 0; l < R; l++)
            {
                double xr = x[2 * ((size_t)l * c + k)], xi = x[2 * ((size_t)l * c + k) + 1];
                if (kind == K_T2 && l > 0)
                {
                    const double *r = tw + ((size_t)(k / 2) * (R - 1) + (l - 1)) * 8;
                    const double cc = r[2 * (k & 1)], ss = r[4 + 2 * (k & 1) + 1], tr = xr * cc - xi * ss;
                    xi = xr * ss + xi * cc;
                    xr = tr;
                }
                const double a = -2.0 * 3.14159265358979323846 * (double)((long)l * m % R) / R;
                re += xr * cos(a) - xi * sin(a);
                im += xr * sin(a) + xi * cos(a);
            }
            const size_t at = kind == K_N1T ? (size_t)k * ols + m : (size_t)m * ols + k;
            const double dr = y[2 * at] - re, di = y[2 * at + 1] - im;
            err = fmax(err, sqrt(dr * dr + di * di));
            mag = fmax(mag, sqrt(re * re + im * im));
        }
    _aligned_free(x);
    _aligned_free(y);
    return err / mag;
}

static double TAIL[NKIND][2][NARM][NRAD];   /* worse-state tail columns, per (kind, count idx, arm, radix) */
static int HAS[NKIND][2][NARM][NRAD];

int main(int argc, char **argv)
{
    FILE *lg = fopen(argc > 1 ? argv[1] : "avx2_tail_mech_results.log", "w");
    if (!lg) return 1;
    bench_pin_caller(2);
    bench_guard_sibling(2);
    static const int CNT[2] = { 1, 5 };
    fprintf(lg, "AVX2 remainder mechanism: the leftover column's cost in ordinary columns, worse of the two stack states\n");
    fprintf(lg, "(the better state in brackets); odd frame fix on in every arm; L1-hot\n\n");
    const double tstart = now_ns();
    for (int kind = 0; kind < NKIND; kind++)
        for (int ci = 0; ci < 2; ci++)
            for (int ri = 0; ri < NRAD; ri++)
            {
                const int R = KT[ri].R, c = CNT[ci];
                const size_t n = (size_t)R * (c + 1);
                double *x = _aligned_malloc(16 * n + 256, 64), *y = _aligned_malloc(16 * n + 256, 64);
                for (size_t j = 0; j < 2 * n + 32; j++) x[j] = sin(0.37 * j) + 0.25 * cos(1.3 * j);
                memset(y, 0, 16 * n + 256);
                double *tw[3] = { 0, 0, 0 };
                if (kind == K_T2)
                    for (int j = 0; j < 3; j++) if (c - 1 + j > 0) tw[j] = vtw2(R, c - 1 + j, R * (c - 1 + j));
                int arms[NARM], na = 0, bad = 0;
                double acc[NARM] = { 0 };
                for (int a = 0; a < NARM; a++) if (KT[ri].f[kind][a]) arms[na++] = a;
                for (int i = 0; i < na; i++)
                    if (arms[i] != A_OVR && arms[i] != A_BOVR)
                    {
                        acc[arms[i]] = accuracy(KT[ri].f[kind][arms[i]], kind, R, c, tw[1]);
                        if (acc[arms[i]] > 1e-12) bad = 1;
                    }
                /* units: arm i at c (state 0, 16), arm i at c-1 (state 0, 16), then narrow at c+1 */
                call_t U[4 * NARM + 1];
                int st[4 * NARM + 1], nu = 0;
                for (int i = 0; i < na; i++)
                    for (int q = 0; q < 4; q++)
                    {
                        const int cc = (q < 2) ? c : c - 1;
                        U[nu] = (call_t){ KT[ri].f[kind][arms[i]], kind, R, cc, tw[cc - (c - 1)], x, y };
                        st[nu++] = 16 * (q & 1);
                    }
                U[nu] = (call_t){ KT[ri].f[kind][A_NAR], kind, R, c + 1, tw[2], x, y };
                st[nu++] = 0;
                for (int u = 0; u < nu; u++) for (int i = 0; i < 32; i++) call1(&U[u]);
                const double t0 = now_ns();
                for (int i = 0; i < 64; i++) call1(&U[0]);
                int reps = (int)(80e3 / ((now_ns() - t0) / 64));
                if (reps < 8) reps = 8;
                static double t[4 * NARM + 1][ROUNDS];
                double med[4 * NARM + 1];
                for (int r = 0; r < ROUNDS; r++)
                    for (int k = 0; k < nu; k++)
                    {
                        const int u = (r & 1) ? nu - 1 - k : k;
                        t[u][r] = shifted(st[u], &U[u], reps);
                    }
                for (int u = 0; u < nu; u++)
                {
                    qsort(t[u], ROUNDS, sizeof(double), cmpd);
                    med[u] = t[u][ROUNDS / 2];
                }
                /* narrow is arm 0: its c-1 units are 2, 3; the column from narrow c-1 (state 0) and c+1 */
                const double col = (med[nu - 1] - med[2]) / 2;
                fprintf(lg, "%-4s R=%-3d c=%d  col %5.1f ns", KINDN[kind], R, c, col);
                for (int i = 0; i < na; i++)
                {
                    const int a = arms[i];
                    const double t0s = (med[4 * i] - med[4 * i + 2]) / col, t16 = (med[4 * i + 1] - med[4 * i + 3]) / col;
                    const double w = fmax(t0s, t16), b = fmin(t0s, t16);
                    fprintf(lg, " | %s %5.2f [%5.2f]", ARMN[a], w, b);
                    TAIL[kind][ci][a][ri] = w;
                    HAS[kind][ci][a][ri] = 1;
                }
                fprintf(lg, "%s\n", bad ? "  ACCURACY FAIL" : "");
                if (bad)
                {
                    fprintf(lg, "  accuracy:");
                    for (int i = 0; i < na; i++) fprintf(lg, " %s %.1e", ARMN[arms[i]], acc[arms[i]]);
                    fprintf(lg, "\n");
                }
                fflush(lg);
                for (int j = 0; j < 3; j++) if (tw[j]) _aligned_free(tw[j]);
                _aligned_free(x);
                _aligned_free(y);
                Sleep(200);
            }
    /* the mechanism, per kind and count: medians over radices of the tail columns and of the ratios */
    fprintf(lg, "\nSUMMARY -- medians over radices (worse state). WIDTH = unmasked ymm pass / xmm pass; MASK = masked / unmasked ymm pass\n");
    for (int kind = 0; kind < NKIND; kind++)
        for (int ci = 0; ci < 2; ci++)
        {
            double v[NARM][NRAD], wm[NRAD], mm[NRAD], wb[NRAD], mb[NRAD];
            int nv[NARM] = { 0 }, nwm = 0, nmm = 0, nwb = 0, nmb = 0;
            for (int ri = 0; ri < NRAD; ri++)
            {
                for (int a = 0; a < NARM; a++) if (HAS[kind][ci][a][ri]) v[a][nv[a]++] = TAIL[kind][ci][a][ri];
                if (HAS[kind][ci][A_OVR][ri] && TAIL[kind][ci][A_NAR][ri] > 0.05)
                    wm[nwm++] = TAIL[kind][ci][A_OVR][ri] / TAIL[kind][ci][A_NAR][ri];
                if (HAS[kind][ci][A_MSK][ri] && TAIL[kind][ci][A_OVR][ri] > 0.05)
                    mm[nmm++] = TAIL[kind][ci][A_MSK][ri] / TAIL[kind][ci][A_OVR][ri];
                if (HAS[kind][ci][A_BOVR][ri] && TAIL[kind][ci][A_BNAR][ri] > 0.05)
                    wb[nwb++] = TAIL[kind][ci][A_BOVR][ri] / TAIL[kind][ci][A_BNAR][ri];
                if (HAS[kind][ci][A_BMSK][ri] && TAIL[kind][ci][A_BOVR][ri] > 0.05)
                    mb[nmb++] = TAIL[kind][ci][A_BMSK][ri] / TAIL[kind][ci][A_BOVR][ri];
            }
            fprintf(lg, "%-4s c=%d  tail:", KINDN[kind], CNT[ci]);
            for (int a = 0; a < NARM; a++)
                if (nv[a])
                {
                    qsort(v[a], nv[a], sizeof(double), cmpd);
                    fprintf(lg, " %s %.2f", ARMN[a], v[a][nv[a] / 2]);
                }
            qsort(wm, nwm, sizeof(double), cmpd);
            qsort(mm, nmm, sizeof(double), cmpd);
            fprintf(lg, " | WIDTH mono %.2f", nwm ? wm[nwm / 2] : 0.0);
            fprintf(lg, " MASK mono %.2f", nmm ? mm[nmm / 2] : 0.0);
            if (nwb)
            {
                qsort(wb, nwb, sizeof(double), cmpd);
                qsort(mb, nmb, sizeof(double), cmpd);
                fprintf(lg, " | WIDTH blk %.2f MASK blk %.2f", wb[nwb / 2], nmb ? mb[nmb / 2] : 0.0);
            }
            fprintf(lg, "\n");
        }
    fprintf(lg, "\nwall %.0f s\n", (now_ns() - tstart) / 1e9);
    fclose(lg);
    printf("done\n");
    return 0;
}
