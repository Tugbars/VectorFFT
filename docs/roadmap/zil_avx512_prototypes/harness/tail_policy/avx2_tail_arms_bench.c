/* avx2_tail_arms_bench.c -- the AVX2 remainder arms (VFFT_TAIL256) at four
 * cache depths, i9-14900KF (2026-09-28).
 *
 * At AVX2 a ymm holds 2 complex, so an odd column count leaves ONE column.
 * The arms (gen_avx2_arms.sh emits each kernel under each, symbol _<arm>):
 *   narrow       the shipped remainder: the monolithic DAG at VEX-128
 *   masked       the monolithic DAG in one ymm pass, lane 1 masked off
 *   blk_narrow   the odd blocked passes at VEX-128          (radix >= 9, n1 / t2)
 *   blk_masked   the odd blocked passes in ymm, lane 1 masked
 *   blk_overrun  the odd blocked passes unmasked: reads and writes one column
 *                past the end -- a COST FLOOR, never a kernel (no accuracy row)
 *
 * PART A, kernels: n1, n1t (the 2p route's corner-turned leaf) and t2 at every
 * odd radix 3..47, counts 1 / 5 / 25. Per cell and arm: the call time at count
 * c; the bulk alone, t_arm(c-1) / t_narrow(c-1) (no remainder runs at an even
 * count, but the arm can move the code gcc emits for the bulk loop); and the
 * leftover column's cost in ordinary columns,
 *   tail = (t_arm(c) - t_arm(c-1)) / ((t_narrow(c+1) - t_narrow(c-1)) / 2),
 * 1.0 = the leftover costs one column, 2.0 = a full two-column pass.
 * PART B, transforms: every odd N the store serves with the 2p route (the
 * banked il_pair R1.R2 cells), run as the route runs them out of place:
 *   n1t(R2) zin -> zout, count R1, Ls R1, OLs R2;
 *   t2(R1) zout -> zout in place, count R2, Ls = OLs = R2, the VTW2 table.
 * Both counts are odd, so both stages run a remainder.
 *
 * DEPTHS: each call takes the next set of a ring of buffer sets (a batch of
 * independent transforms walked in memory order; one cursor per cell, so no
 * unit reuses the sets the unit before it touched); the ring's footprint puts
 * the data in L1 (24 KB; "L1*" = one set already exceeds it), L2 (1 MB),
 * L3 (16 MB) or DRAM (256 MB). Twiddle tables are the plan's: one per cell,
 * shared by the whole ring.
 *
 * PROTOCOL: core 2 at HIGH priority, its SMT sibling held by the gauntlet's
 * guard; per timing unit the minimum of 5 batches of ~80 us; 21 rounds, the
 * units in alternating order; median. A second narrow unit is the same-run
 * CONTROL (its deviation is the noise floor). 200 ms pause between cells,
 * never inside one. Accuracy: every arm but blk_overrun against a direct DFT.
 *
 * Build: build_avx2_arms.sh (Bash tool, mingw gcc). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "sibling_guard.h"

#define KARGS const double *, const double *, double *, double *, const double *, const double *, \
              size_t, size_t, size_t, size_t, size_t
typedef void (*kfn)(KARGS);
enum { A_NAR, A_MSK, A_BNAR, A_BMSK, A_BOVR, NARM };
static const char *ARMN[NARM] = { "narrow", "masked", "blk_nar", "blk_msk", "blk_ovr" };
enum { K_N1, K_N1T, K_T2, NKIND };
static const char *KINDN[NKIND] = { "n1", "n1t", "t2" };

#define SMALL(X) X(3) X(5) X(7)
#define BIG(X) X(9) X(11) X(13) X(15) X(17) X(19) X(21) X(23) X(25) X(27) X(29) X(31) X(37) X(41) X(43) X(47)
#define KD(R, K, A) void radix##R##_z_##K##_fwd_avx2_##A(KARGS);
#define DS(R) KD(R, n1, narrow) KD(R, n1, masked) KD(R, n1t, narrow) KD(R, n1t, masked) KD(R, t2, narrow) KD(R, t2, masked)
#define DB(R) DS(R) KD(R, n1, blk_narrow) KD(R, n1, blk_masked) KD(R, n1, blk_overrun) \
              KD(R, t2, blk_narrow) KD(R, t2, blk_masked) KD(R, t2, blk_overrun)
SMALL(DS) BIG(DB)
#define F(R, K, A) radix##R##_z_##K##_fwd_avx2_##A
#define ES(R) { R, { { F(R, n1, narrow), F(R, n1, masked), 0, 0, 0 }, \
                     { F(R, n1t, narrow), F(R, n1t, masked), 0, 0, 0 }, \
                     { F(R, t2, narrow), F(R, t2, masked), 0, 0, 0 } } },
#define EB(R) { R, { { F(R, n1, narrow), F(R, n1, masked), F(R, n1, blk_narrow), F(R, n1, blk_masked), F(R, n1, blk_overrun) }, \
                     { F(R, n1t, narrow), F(R, n1t, masked), 0, 0, 0 }, \
                     { F(R, t2, narrow), F(R, t2, masked), F(R, t2, blk_narrow), F(R, t2, blk_masked), F(R, t2, blk_overrun) } } },
static const struct { int R; kfn f[NKIND][NARM]; } KT[] = { SMALL(ES) BIG(EB) };
enum { NRAD = sizeof KT / sizeof *KT };
static kfn kern(int R, int kind, int arm)
{
    for (int i = 0; i < NRAD; i++) if (KT[i].R == R) return KT[i].f[kind][arm];
    return 0;
}

enum { ROUNDS = 21, BATCHES = 5, NDEPTH = 4 };
static const char *DEPTHN[NDEPTH] = { "L1", "L2", "L3", "DRAM" };
static const size_t DEPTHB[NDEPTH] = { 24u << 10, 1u << 20, 16u << 20, 256u << 20 };
static const double BATCH_NS = 80e3;

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
static size_t up64(size_t b) { return (b + 63) & ~(size_t)63; }

/* the arena: one allocation for every ring (DRAM depth + slack), pages touched once */
static char *ARENA;
static const size_t ARENA_B = (256u << 20) + (8u << 20);

/* VTW2 for `cols` columns of radix R at modulus N: pair pp, leg l at
 * (pp*(R-1) + l-1)*8, [c,c,c',c'][-s,+s,-s',+s'], angle -2*pi*l*k/N; the lone
 * last column's pair carries the k = cols angle in lane 1 (never read) */
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
/* the twiddle a VTW2 table holds for (leg l, column k): c + i*s */
static void tw_at(const double *t, int R, int l, int k, double *c, double *s)
{
    const double *r = t + ((size_t)(k / 2) * (R - 1) + (l - 1)) * 8;
    *c = r[2 * (k & 1)];
    *s = r[4 + 2 * (k & 1) + 1];
}
/* direct radix-R DFT of one column: legs in[l*ls + k], pre-twiddled by tw (or not) */
static void dft_col(int R, const double *in, size_t ls, int k, const double *tw, double *o)
{
    for (int m = 0; m < R; m++)
    {
        double re = 0, im = 0;
        for (int l = 0; l < R; l++)
        {
            double xr = in[2 * ((size_t)l * ls + k)], xi = in[2 * ((size_t)l * ls + k) + 1];
            if (tw && l > 0)
            {
                double c, s;
                tw_at(tw, R, l, k, &c, &s);
                const double tr = xr * c - xi * s;
                xi = xr * s + xi * c;
                xr = tr;
            }
            const double a = -2.0 * 3.14159265358979323846 * (double)((long)l * m % R) / R;
            re += xr * cos(a) - xi * sin(a);
            im += xr * sin(a) + xi * cos(a);
        }
        o[2 * m] = re;
        o[2 * m + 1] = im;
    }
}
static void fill(double *x, size_t nd, unsigned seed)
{
    for (size_t j = 0; j < nd; j++)
    {
        seed = seed * 1664525u + 1013904223u;
        x[j] = (double)(seed >> 8) / 16777216.0 - 0.5;
    }
}

/* ── the timed unit: one call of `f` (or a 2p transform) on ring set `s` ── */
typedef struct
{
    int part;            /* 0 = kernel, 1 = 2p transform */
    kfn f, g;            /* kernel / (n1t, t2) */
    int kind, R, count;  /* kernel */
    int R1, R2;          /* transform */
    const double *tw;
    size_t stride, nsets, yoff;
    size_t *cur;         /* the cell's ONE ring cursor, shared by its units: each
                            unit takes the sets after the last unit's, never the
                            ones the unit before it just pulled into cache */
} unit_t;
/* n calls of the unit. Everything the loop needs lives in locals (registers
 * across the opaque calls): no per-call reload of the unit or the cursor from
 * memory, where the kernel's stores could alias them 4 KB apart, and a wrap
 * instead of a 64-bit modulo -- at a 3 ns call the harness must cost the same
 * for every unit. */
static void run_n(unit_t *u, int n)
{
    const kfn f = u->f, g = u->g;
    const double *tw = u->tw;
    const size_t nsets = u->nsets, stride = u->stride, yoff = u->yoff;
    size_t idx = *u->cur % nsets;
    if (u->part == 1)
    {
        const size_t r1 = (size_t)u->R1, r2 = (size_t)u->R2;
        for (int i = 0; i < n; i++)
        {
            char *base = ARENA + idx * stride;
            double *y = (double *)(base + yoff);
            f((const double *)base, 0, y, 0, 0, 0, r1, 0, r2, 0, r1);
            g(y, 0, y, 0, tw, 0, r2, 0, r2, 0, r2);
            if (++idx == nsets) idx = 0;
        }
    }
    else
    {
        const size_t c = (size_t)u->count, ols = u->kind == K_N1T ? (size_t)u->R : c;
        for (int i = 0; i < n; i++)
        {
            char *base = ARENA + idx * stride;
            f((const double *)base, 0, (double *)(base + yoff), 0, tw, 0, c, 0, ols, 0, c);
            if (++idx == nsets) idx = 0;
        }
    }
    *u->cur = idx;
}
static double time_unit(unit_t *u, int reps)
{
    double best = 1e30;
    for (int b = 0; b < BATCHES; b++)
    {
        const double t0 = now_ns();
        run_n(u, reps);
        const double t = (now_ns() - t0) / reps;
        if (t < best) best = t;
    }
    return best;
}
/* ring geometry for sets holding `in_b` + `out_b` bytes (each + 64 B slack, the
 * output skewed 64 B off the input's 4 KB phase) */
static void ring(unit_t *u, size_t in_b, size_t out_b, int depth)
{
    u->yoff = up64(in_b + 64) + 64;
    u->stride = up64(u->yoff + out_b + 64);
    u->nsets = DEPTHB[depth] / u->stride;
    if (u->nsets < 1) u->nsets = 1;
}

/* run the units of one cell: warm, size reps from unit 0, 21 alternating rounds;
 * med[u] = median ns, iqr[u] = interquartile range / median */
static void race(unit_t *U, int nu, double *med, double *iqr)
{
    static double t[16][ROUNDS];
    for (int u = 0; u < nu; u++) run_n(&U[u], 64);
    const double t0 = now_ns();
    run_n(&U[0], 64);
    const double per = (now_ns() - t0) / 64;
    int reps = (int)(BATCH_NS / (per > 1 ? per : 1));
    if (reps < 4) reps = 4;
    for (int r = 0; r < ROUNDS; r++)
        for (int k = 0; k < nu; k++)
        {
            const int u = (r & 1) ? nu - 1 - k : k;
            t[u][r] = time_unit(&U[u], reps);
        }
    for (int u = 0; u < nu; u++)
    {
        qsort(t[u], ROUNDS, sizeof(double), cmpd);
        med[u] = t[u][ROUNDS / 2];
        iqr[u] = (t[u][(3 * ROUNDS) / 4] - t[u][ROUNDS / 4]) / med[u];
    }
}

/* ── PART A ─────────────────────────────────────────────────────────────── */
static double sumA[NKIND][NDEPTH][NARM][64];   /* per (kind, depth, arm): tail columns over cells */
static int nA[NKIND][NDEPTH][NARM];
static double ratA[NKIND][NDEPTH][NARM][64];   /* t_arm(c) / t_narrow(c) */
static double bulkA[NKIND][NDEPTH][NARM][64];  /* t_arm(c-1) / t_narrow(c-1): the bulk alone */

static double acc_kernel(kfn f, int kind, int R, int c, const double *tw)
{
    const size_t ols = kind == K_N1T ? (size_t)R : (size_t)c, n = (size_t)R * c;
    double *x = _aligned_malloc(16 * n + 128, 64), *y = _aligned_malloc(16 * n + 128, 64), o[2 * 64];
    fill(x, 2 * n, 12345u + R * 7 + c);
    memset(y, 0, 16 * n + 128);
    f(x, 0, y, 0, tw, 0, (size_t)c, 0, ols, 0, (size_t)c);
    double err = 0, mag = 0;
    for (int k = 0; k < c; k++)
    {
        dft_col(R, x, (size_t)c, k, kind == K_T2 ? tw : 0, o);
        for (int m = 0; m < R; m++)
        {
            const size_t at = kind == K_N1T ? (size_t)k * ols + m : (size_t)m * ols + k;
            const double dr = y[2 * at] - o[2 * m], di = y[2 * at + 1] - o[2 * m + 1];
            const double e = sqrt(dr * dr + di * di), a = sqrt(o[2 * m] * o[2 * m] + o[2 * m + 1] * o[2 * m + 1]);
            if (e > err) err = e;
            if (a > mag) mag = a;
        }
    }
    _aligned_free(x);
    _aligned_free(y);
    return err / (mag > 0 ? mag : 1);
}

static void part_a(FILE *lg)
{
    static const int CNT[] = { 1, 5, 25 };
    fprintf(lg, "\nPART A -- kernels: ns per call at count c per arm; tail = the leftover column in ordinary columns\n");
    for (int kind = 0; kind < NKIND; kind++)
        for (int ri = 0; ri < NRAD; ri++)
            for (int ci = 0; ci < 3; ci++)
            {
                const int R = KT[ri].R, c = CNT[ci];
                const double *tw[3] = { 0, 0, 0 };
                double *twb[3] = { 0, 0, 0 };
                if (kind == K_T2)
                    for (int j = 0; j < 3; j++)
                        if (c - 1 + j > 0) tw[j] = twb[j] = vtw2(R, c - 1 + j, R * (c - 1 + j));
                int arms[NARM], na = 0;
                for (int a = 0; a < NARM; a++) if (kern(R, kind, a)) arms[na++] = a;
                /* accuracy, every arm but the overrun floor */
                double acc[NARM] = { 0 };
                int bad = 0;
                for (int i = 0; i < na; i++)
                    if (arms[i] != A_BOVR)
                    {
                        acc[arms[i]] = acc_kernel(kern(R, kind, arms[i]), kind, R, c, tw[1]);
                        if (acc[arms[i]] > 1e-12) bad = 1;
                    }
                for (int d = 0; d < NDEPTH; d++)
                {
                    unit_t U[16];
                    int nu = 0;
                    size_t cur = 0;
                    const size_t setb = (size_t)R * (c + 1) * 16;
                    /* units: the arms at c; the control (narrow again); the arms at
                     * c-1, where no remainder runs (the arm's own bulk: a remainder
                     * arm can move the code gcc emits for the bulk loop); narrow at c+1 */
                    for (int i = 0; i < 2 * na + 2; i++)
                    {
                        unit_t *u = &U[nu++];
                        memset(u, 0, sizeof *u);
                        u->part = 0; u->kind = kind; u->R = R;
                        const int a = i < na ? arms[i] : (i > na && i <= 2 * na) ? arms[i - na - 1] : A_NAR;
                        u->f = kern(R, kind, a);
                        u->count = i <= na ? c : (i <= 2 * na ? c - 1 : c + 1);
                        u->tw = tw[u->count - (c - 1)];
                        u->cur = &cur;
                        ring(u, setb, setb, d);
                    }
                    double med[16], iqr[16];
                    race(U, nu, med, iqr);
                    const double tm1 = med[na + 1], tp1 = med[2 * na + 1], col = (tp1 - tm1) / 2;
                    fprintf(lg, "%-4s R=%-3d c=%-3d %-4s%s", KINDN[kind], R, c, DEPTHN[d],
                            (d == 0 && 2 * setb > DEPTHB[0]) ? "*" : " ");
                    for (int i = 0; i < na; i++)
                    {
                        const int a = arms[i];
                        const double tail = col > 0 ? (med[i] - med[na + 1 + i]) / col : 0;
                        const double bulk = med[na + 1 + i] / tm1;
                        fprintf(lg, " | %s %8.1f ns bulk %.2f tail %5.2f", ARMN[a], med[i], bulk, tail);
                        if (nA[kind][d][a] < 64)
                        {
                            sumA[kind][d][a][nA[kind][d][a]] = tail;
                            ratA[kind][d][a][nA[kind][d][a]] = med[i] / med[0];
                            bulkA[kind][d][a][nA[kind][d][a]] = bulk;
                            nA[kind][d][a]++;
                        }
                    }
                    fprintf(lg, " | control %+.1f%% | spread %.1f%% | col %.1f ns%s\n",
                            100 * (med[na] / med[0] - 1), 100 * iqr[0], col, bad ? "  ACCURACY FAIL" : "");
                    fflush(lg);
                    Sleep(200);
                }
                if (bad)
                {
                    fprintf(lg, "  accuracy:");
                    for (int i = 0; i < na; i++) fprintf(lg, " %s %.1e", ARMN[arms[i]], acc[arms[i]]);
                    fprintf(lg, "\n");
                }
                for (int j = 0; j < 3; j++) if (twb[j]) _aligned_free(twb[j]);
            }
}

/* ── PART B ─────────────────────────────────────────────────────────────── */
/* the banked odd 2p cells, il_pair R1.R2 (src/wisdom/wisdom2_oop.txt, ord=nat oop fwd) */
static const short PAIRS[][2] = {
    {3,5},{3,7},{5,5},{3,11},{5,7},{3,13},{5,9},{7,7},{3,17},{5,11},{3,19},{7,9},{5,13},{3,23},{5,15},{7,11},
    {9,9},{5,17},{7,13},{5,19},{11,9},{15,7},{5,23},{9,13},{17,7},{11,11},{5,25},{7,19},{9,15},{11,13},{29,5},
    {17,9},{5,31},{23,7},{15,11},{13,13},{9,19},{5,37},{29,7},{41,5},{9,23},{19,11},{7,31},{17,13},{9,25},{5,47},
    {9,27},{19,13},{23,11},{17,15},{37,7},{29,9},{13,21},{9,31},{15,19},{41,7},{17,17},{23,13},{29,11},{19,17},
    {13,25},{9,37},{15,23},{19,19},{9,41},{13,29},{43,9},{23,17},{13,31},{37,11},{17,25},{29,15},{23,19},{11,41},
    {15,31},{43,11},{19,25},{37,13},{21,23},{29,17},{19,27},{25,21},{31,17},{23,23},{41,13},{29,19},{37,15},{43,13},
    {21,27},{25,23},{31,19},{21,29},{41,15},{23,27},{25,25},{17,37},{43,15},{21,31},{23,29},{27,25},{41,17},{37,19},
    {47,15},{23,31},{27,27},{43,17},{31,25},{37,21},{41,19},{47,17},{37,23},{41,21},{31,29},{25,37},{41,23},{47,21},
    {27,37},{41,25},{37,29},{43,25},{23,47},{41,27},{43,27},{41,29},{43,31},{29,47},{37,37},{31,47},{37,41},{37,43},
    {41,41},{41,43},
};
enum { NPAIR = sizeof PAIRS / sizeof *PAIRS };
/* transform arms: (n1t arm, t2 arm) */
enum { NTARM = 6 };
static const int TA[NTARM][2] = { { A_NAR, A_NAR }, { A_MSK, A_NAR }, { A_NAR, A_MSK }, { A_NAR, A_BNAR }, { A_NAR, A_BMSK }, { A_NAR, A_BOVR } };
static const char *TAN[NTARM] = { "shipped", "n1t:msk", "t2:msk", "t2:blkn", "t2:blkm", "t2:ovr" };
static double ratB[NDEPTH][NTARM][NPAIR];
static int nB[NDEPTH][NTARM];

static double acc_2p(kfn f, kfn g, int R1, int R2, const double *tw)
{
    const int N = R1 * R2;
    double *x = _aligned_malloc(16 * (size_t)N + 128, 64), *y = _aligned_malloc(16 * (size_t)N + 128, 64);
    fill(x, 2 * (size_t)N, 777u + N);
    f(x, 0, y, 0, 0, 0, (size_t)R1, 0, (size_t)R2, 0, (size_t)R1);
    g(y, 0, y, 0, tw, 0, (size_t)R2, 0, (size_t)R2, 0, (size_t)R2);
    double err = 0, mag = 0, *ct = malloc(sizeof(double) * N), *st = malloc(sizeof(double) * N);
    for (int n = 0; n < N; n++)
    {
        ct[n] = cos(-2.0 * 3.14159265358979323846 * n / N);
        st[n] = sin(-2.0 * 3.14159265358979323846 * n / N);
    }
    for (int j = 0; j < N; j++)
    {
        double re = 0, im = 0;
        for (int n = 0, e = 0; n < N; n++, e = (e + j) % N)
        {
            re += x[2 * n] * ct[e] - x[2 * n + 1] * st[e];
            im += x[2 * n] * st[e] + x[2 * n + 1] * ct[e];
        }
        const double dr = y[2 * j] - re, di = y[2 * j + 1] - im;
        const double e = sqrt(dr * dr + di * di), m = sqrt(re * re + im * im);
        if (e > err) err = e;
        if (m > mag) mag = m;
    }
    free(ct);
    free(st);
    _aligned_free(x);
    _aligned_free(y);
    return err / mag;
}

static void part_b(FILE *lg)
{
    fprintf(lg, "\nPART B -- 2p transforms, n1t(R2) then t2(R1) in place: ns per transform per arm (n1t arm, t2 arm)\n");
    for (int pi = 0; pi < NPAIR; pi++)
    {
        const int R1 = PAIRS[pi][0], R2 = PAIRS[pi][1], N = R1 * R2;
        double *tw = vtw2(R1, R2, N);
        int arms[NTARM], na = 0, bad = 0;
        double acc[NTARM] = { 0 };
        for (int a = 0; a < NTARM; a++)
            if (kern(R2, K_N1T, TA[a][0]) && kern(R1, K_T2, TA[a][1])) arms[na++] = a;
        for (int i = 0; i < na; i++)
            if (TA[arms[i]][1] != A_BOVR)
            {
                acc[arms[i]] = acc_2p(kern(R2, K_N1T, TA[arms[i]][0]), kern(R1, K_T2, TA[arms[i]][1]), R1, R2, tw);
                if (acc[arms[i]] > 1e-12) bad = 1;
            }
        for (int d = 0; d < NDEPTH; d++)
        {
            unit_t U[16];
            int nu = 0;
            size_t cur = 0;
            for (int i = 0; i <= na; i++)   /* the arms, then the control */
            {
                unit_t *u = &U[nu++];
                memset(u, 0, sizeof *u);
                const int a = i < na ? arms[i] : 0;
                u->part = 1; u->R1 = R1; u->R2 = R2; u->tw = tw;
                u->f = kern(R2, K_N1T, TA[a][0]);
                u->g = kern(R1, K_T2, TA[a][1]);
                u->cur = &cur;
                ring(u, 16 * (size_t)N, 16 * (size_t)N, d);
            }
            double med[16], iqr[16];
            race(U, nu, med, iqr);
            fprintf(lg, "N=%-5d %2d.%-2d %-4s%s", N, R1, R2, DEPTHN[d], (d == 0 && 32 * (size_t)N > DEPTHB[0]) ? "*" : " ");
            for (int i = 0; i < na; i++)
            {
                fprintf(lg, " | %s %8.1f", TAN[arms[i]], med[i]);
                ratB[d][arms[i]][nB[d][arms[i]]++] = med[i] / med[0];
            }
            fprintf(lg, " | control %+.1f%% | spread %.1f%%%s\n", 100 * (med[na] / med[0] - 1), 100 * iqr[0],
                    bad ? "  ACCURACY FAIL" : "");
            fflush(lg);
            Sleep(200);
        }
        fprintf(lg, "  accuracy:");
        for (int i = 0; i < na; i++) if (TA[arms[i]][1] != A_BOVR) fprintf(lg, " %s %.1e", TAN[arms[i]], acc[arms[i]]);
        fprintf(lg, "\n");
        _aligned_free(tw);
    }
}

static double median_of(double *v, int n)
{
    if (n <= 0) return 0;
    qsort(v, n, sizeof(double), cmpd);
    return v[n / 2];
}
static void summary(FILE *lg)
{
    fprintf(lg, "\nSUMMARY A -- median over radices and counts: tail columns (call ratio to narrow; bulk-alone ratio)\n");
    for (int kind = 0; kind < NKIND; kind++)
        for (int d = 0; d < NDEPTH; d++)
        {
            fprintf(lg, "%-4s %-4s", KINDN[kind], DEPTHN[d]);
            for (int a = 0; a < NARM; a++)
                if (nA[kind][d][a])
                    fprintf(lg, " | %s %5.2f (%.3f; %.3f, n=%d)", ARMN[a], median_of(sumA[kind][d][a], nA[kind][d][a]),
                            median_of(ratA[kind][d][a], nA[kind][d][a]), median_of(bulkA[kind][d][a], nA[kind][d][a]),
                            nA[kind][d][a]);
            fprintf(lg, "\n");
        }
    fprintf(lg, "\nSUMMARY B -- 2p transforms, per depth: median ratio to shipped [min, max] (cells)\n");
    for (int d = 0; d < NDEPTH; d++)
    {
        fprintf(lg, "%-4s", DEPTHN[d]);
        for (int a = 1; a < NTARM; a++)
        {
            const int n = nB[d][a];
            if (!n) continue;
            const double m = median_of(ratB[d][a], n);
            fprintf(lg, " | %s %.3f [%.3f, %.3f] (%d)", TAN[a], m, ratB[d][a][0], ratB[d][a][n - 1], n);
        }
        fprintf(lg, "\n");
    }
}

int main(int argc, char **argv)
{
    const char *path = argc > 1 ? argv[1] : "avx2_tail_arms_results.log";
    const int only = argc > 2 ? atoi(argv[2]) : 0;   /* 0 = both parts, 1 = A, 2 = B */
    FILE *lg = fopen(path, "w");
    if (!lg) { perror(path); return 1; }
    bench_pin_caller(2);
    bench_guard_sibling(2);
    ARENA = _aligned_malloc(ARENA_B, 4096);
    if (!ARENA) { fprintf(stderr, "arena alloc failed\n"); return 1; }
    memset(ARENA, 0, ARENA_B);
    fill((double *)ARENA, ARENA_B / 8, 99u);
    fprintf(lg, "AVX2 remainder arms, i9-14900KF core 2 (sibling guarded), %d rounds x min of %d batches (~%.0f us)\n",
            ROUNDS, BATCHES, BATCH_NS / 1e3);
    fprintf(lg, "depths: ring footprint L1 24 KB (L1* = one set exceeds it), L2 1 MB, L3 16 MB, DRAM 256 MB\n");
    const double t0 = now_ns();
    if (only != 2) part_a(lg);
    if (only != 1) part_b(lg);
    summary(lg);
    fprintf(lg, "\nwall %.0f s\n", (now_ns() - t0) / 1e9);
    fclose(lg);
    printf("done -> %s\n", path);
    return 0;
}
