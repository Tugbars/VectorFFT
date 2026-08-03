/* zturn_dit_gate.c — Phase C gate 1: the DIT-forward boundary kinds are
 * EXACT conjugates of their bwd donors.
 *
 * The emission algebra (dit_cascade_spec.md): F = conj∘B∘conj, and IEEE
 * conjugation is a sign-bit flip — exact. So each DIT kernel must satisfy,
 * memcmp-EXACT (not tolerance):
 *
 *   dts (x, w)        == conj( stfb (conj x, conj w) )        r8 + r4
 *   dtsn(x, w, tb)    == conj( stfbn(conj x, conj w, tb) )    r8 + r4
 *   dtt (plane)       == conj( s0tb (conj plane) )            r4
 *
 * where conj on a user-z buffer negates z[2i+1], and conj on a plane or
 * packed-w^1 stream negates the im/sin quad (doubles 4..7 of each 64 B
 * record — both use the [re×4][im×4] record shape). The conjugated twiddle
 * stream is built HERE from the plan's fwd tzq (self-contained identity —
 * not routed through the driver's tzqb, so a driver table bug cannot mask
 * an emitter bug).
 *
 * 🔴 PINNED BY THIS GATE: dtsn's tw_im block table is rho (tb — the SAME
 * table stfbn takes), NOT rho⁻¹. Conjugation does not touch addressing.
 * Mixed-radix chains mandatory (rho is an involution on uniform chains).
 *
 * GATE 2 (speed, gate_new_kernels_on_speed_too): dts/stfb, dtsn/stfbn,
 * dtt/s0tb paced-median ratios — identical instruction mix modulo sign
 * constants, so ≈1.0 expected; <1.15 green (thermal).
 *
 * Build: python build_tuned/build.py --src build_tuned/benches/zturn_dit_gate.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#include <windows.h>
#endif

#include "zturn.h"

/* DIT boundary kernels (not yet declared in zturn.h — driver wiring is the
 * next step; the gate declares them itself) */
#define VFFT_DIT_DECL(fn) extern void fn(const double *, const double *, \
    double *, double *, const double *, const double *,                  \
    size_t, size_t, size_t, size_t, size_t);
VFFT_DIT_DECL(radix8_z_dts_r4_fwd_avx2)
VFFT_DIT_DECL(radix4_z_dts_r4_fwd_avx2)
VFFT_DIT_DECL(radix8_z_dtsn_r4_fwd_avx2)
VFFT_DIT_DECL(radix4_z_dtsn_r4_fwd_avx2)
VFFT_DIT_DECL(radix4_z_dtt_r4_fwd_avx2)
#undef VFFT_DIT_DECL

static double now_ns(void)
{
#ifdef _WIN32
    static double f = 0.0;
    LARGE_INTEGER t;
    if (f == 0.0) { LARGE_INTEGER q; QueryPerformanceFrequency(&q);
                    f = 1e9 / (double)q.QuadPart; }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * f;
#else
    return 0.0;
#endif
}
static void pace(int ms) {
#ifdef _WIN32
    Sleep((DWORD)ms);
#endif
}
static double *az(size_t doubles)
{
#ifdef _WIN32
    return (double *)_aligned_malloc(doubles * sizeof(double), 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, doubles * sizeof(double))) p = NULL;
    return (double *)p;
#endif
}
static void fz(double *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}
static int dcmp(const void *a, const void *b)
{
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : (x > y ? 1 : 0);
}

static long rho0(long v, const int *r, int m)
{
    long d[16];
    for (int i = m - 1; i >= 0; i--) { d[i] = v % r[i]; v /= r[i]; }
    long out = 0;
    for (int i = m - 1; i >= 0; i--) out = out * r[i] + d[i];
    return out;
}

/* conj on interleaved user z: negate z[2i+1] */
static void conj_il(double *z, long n)
{
    for (long i = 0; i < n; i++) z[2 * i + 1] = -z[2 * i + 1];
}
/* conj on [re×4][im×4] records (plane / packed-w^1 stream): negate 4..7 */
static void conj_blk(double *p, size_t doubles)
{
    for (size_t b = 0; b + 8 <= doubles; b += 8)
        for (int j = 4; j < 8; j++) p[b + j] = -p[b + j];
}

typedef void (*termfn)(const double *, const double *, double *, double *,
                       const double *, const double *,
                       size_t, size_t, size_t, size_t, size_t);

typedef struct { int N, nf, chain[8]; } cell_t;
static const cell_t CELLS[] = {
    { 2048,  5, {4,8,4,4,4} },
    { 4096,  6, {4,4,4,4,4,4} },
    { 8192,  6, {4,8,4,4,4,4} },
    { 16384, 7, {4,4,4,4,4,4,4} },
    { 32768, 7, {4,8,4,4,4,4,4} },
    { 16384, 6, {4,8,4,4,4,8} },   /* r8 ingest form */
};

int main(void)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)0x4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
#endif
    int fails = 0;
    printf("\n=== Phase C gate 1: DIT kinds == conj(bwd donors) ===\n");
    printf("%-7s %-16s %-3s | %-6s %-6s %-6s | %-9s %-9s %-9s\n",
           "N", "chain", "Rt", "dts", "dtsn", "dtt",
           "dts/stfb", "dtsn/bn", "dtt/s0tb");

    for (size_t ci = 0; ci < sizeof CELLS / sizeof CELLS[0]; ci++)
    {
        const cell_t *c = &CELLS[ci];
        const int N = c->N, Rt = c->chain[c->nf - 1];
        char cs[32] = {0};
        for (int i = 0, o = 0; i < c->nf; i++)
            o += snprintf(cs + o, sizeof cs - (size_t)o, i ? ".%d" : "%d",
                          c->chain[i]);

        vfft_zturn2_plan_t *p =
            vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
        if (!p) { printf("%-7d %-16s REFUSED\n", N, cs); fails++; continue; }

        const size_t OLs = (size_t)N / (size_t)Rt;
        const long M = (long)(OLs / 4);
        size_t *tb = (size_t *)malloc(sizeof(size_t) * (size_t)M); /* rho */
        for (long t = 0; t < M; t++)
            tb[t] = (size_t)rho0(t, c->chain + 1, c->nf - 2);

        const termfn DTS  = (Rt == 8) ? radix8_z_dts_r4_fwd_avx2
                                      : radix4_z_dts_r4_fwd_avx2;
        const termfn DTSN = (Rt == 8) ? radix8_z_dtsn_r4_fwd_avx2
                                      : radix4_z_dtsn_r4_fwd_avx2;
        const termfn STFB = (Rt == 8) ? radix8_z_stf_r4_bwd_avx2
                                      : radix4_z_stf_r4_bwd_avx2;
        const termfn STFBN = (Rt == 8) ? radix8_z_stfn_r4_bwd_avx2
                                       : radix4_z_stfn_r4_bwd_avx2;

        srand(7717 + N + Rt);
        double *x   = az(2 * (size_t)N), *xc = az(2 * (size_t)N);
        double *pa  = az(2 * (size_t)N), *pb = az(2 * (size_t)N);
        double *wc  = az(2 * OLs);                 /* conj w^1 stream       */
        for (long i = 0; i < 2L * N; i++)
        {
            x[i] = (double)rand() / RAND_MAX - 0.5;
            p->plane[i] = (double)rand() / RAND_MAX - 0.5;
        }
        memcpy(xc, x, 2 * (size_t)N * sizeof(double));
        conj_il(xc, N);
        memcpy(wc, p->tzq, 2 * OLs * sizeof(double));
        conj_blk(wc, 2 * OLs);

        /* ── dts == conj(stfb(conj x, conj w)) ── */
        memset(pa, 0, 2 * (size_t)N * sizeof(double));
        memset(pb, 0, 2 * (size_t)N * sizeof(double));
        DTS (x,  0, pa, 0, p->tzq, 0, 0, 0, OLs, 0, OLs);
        STFB(xc, 0, pb, 0, wc,     0, 0, 0, OLs, 0, OLs);
        conj_blk(pb, 2 * (size_t)N);
        const int ok1 = memcmp(pa, pb, 2 * (size_t)N * sizeof(double)) == 0;

        /* ── dtsn == conj(stfbn(conj x, conj w, SAME rho table)) ── */
        memset(pa, 0, 2 * (size_t)N * sizeof(double));
        memset(pb, 0, 2 * (size_t)N * sizeof(double));
        DTSN (x,  0, pa, 0, p->tzq, (const double *)tb, 0, 0, OLs, 0, OLs);
        STFBN(xc, 0, pb, 0, wc,     (const double *)tb, 0, 0, OLs, 0, OLs);
        conj_blk(pb, 2 * (size_t)N);
        const int ok2 = memcmp(pa, pb, 2 * (size_t)N * sizeof(double)) == 0;

        /* ── dtt == conj(s0tb(conj plane)) — radix-4, chain-independent ── */
        double *plc = az(2 * (size_t)N);
        memcpy(plc, p->plane, 2 * (size_t)N * sizeof(double));
        conj_blk(plc, 2 * (size_t)N);
        radix4_z_dtt_r4_fwd_avx2(p->plane, 0, pa, 0, 0, 0,
                                 (size_t)N / 4, 0, 0, 0, (size_t)N / 4);
        radix4_z_s0t_r4_bwd_avx2(plc, 0, pb, 0, 0, 0,
                                 (size_t)N / 4, 0, 0, 0, (size_t)N / 4);
        conj_il(pb, N);
        const int ok3 = memcmp(pa, pb, 2 * (size_t)N * sizeof(double)) == 0;

        /* ── speed: paced medians, 9 rounds ── */
        double s[6][9];
        const int reps = N <= 4096 ? 400 : (N <= 16384 ? 100 : 50);
        for (int r = 0; r < 9; r++)
        {
            double t0;
            t0 = now_ns(); for (int i = 0; i < reps; i++)
                DTS(x, 0, pa, 0, p->tzq, 0, 0, 0, OLs, 0, OLs);
            s[0][r] = (now_ns() - t0) / reps;
            t0 = now_ns(); for (int i = 0; i < reps; i++)
                STFB(xc, 0, pb, 0, wc, 0, 0, 0, OLs, 0, OLs);
            s[1][r] = (now_ns() - t0) / reps;
            t0 = now_ns(); for (int i = 0; i < reps; i++)
                DTSN(x, 0, pa, 0, p->tzq, (const double *)tb, 0, 0, OLs, 0, OLs);
            s[2][r] = (now_ns() - t0) / reps;
            t0 = now_ns(); for (int i = 0; i < reps; i++)
                STFBN(xc, 0, pb, 0, wc, (const double *)tb, 0, 0, OLs, 0, OLs);
            s[3][r] = (now_ns() - t0) / reps;
            t0 = now_ns(); for (int i = 0; i < reps; i++)
                radix4_z_dtt_r4_fwd_avx2(p->plane, 0, pa, 0, 0, 0,
                                         (size_t)N / 4, 0, 0, 0, (size_t)N / 4);
            s[4][r] = (now_ns() - t0) / reps;
            t0 = now_ns(); for (int i = 0; i < reps; i++)
                radix4_z_s0t_r4_bwd_avx2(plc, 0, pb, 0, 0, 0,
                                         (size_t)N / 4, 0, 0, 0, (size_t)N / 4);
            s[5][r] = (now_ns() - t0) / reps;
            pace(100);
        }
        for (int a = 0; a < 6; a++) qsort(s[a], 9, sizeof(double), dcmp);
        const double r1 = s[0][4] / s[1][4], r2 = s[2][4] / s[3][4],
                     r3 = s[4][4] / s[5][4];

        const int ok = ok1 && ok2 && ok3;
        if (!ok) fails++;
        printf("%-7d %-16s %-3d | %-6s %-6s %-6s | %7.3fx %7.3fx %7.3fx%s\n",
               N, cs, Rt,
               ok1 ? "EXACT" : "DIFF!", ok2 ? "EXACT" : "DIFF!",
               ok3 ? "EXACT" : "DIFF!",
               r1, r2, r3, ok ? "" : "   *** FAIL ***");

        free(tb);
        fz(x); fz(xc); fz(pa); fz(pb); fz(wc); fz(plc);
        vfft_zturn2_destroy(p);
    }

    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
