/* zil_cascade.c — checklist item 16: the z STAGED CASCADE spike (docs/roadmap/
 * z_cascade_plan.md). Replaces the flat four-step two-pass (which collapses at
 * N>=2048, 0.46x MKL) with a multi-stage DIT cascade through ping-pong scratch —
 * the stride-engine shape in z clothing, confirmed by the MKL >=2048 census
 * (docs/research/mkl_highN_cascade_anatomy.md).
 *
 * STRUCTURE (this file grows in two commits):
 *   [now]  harness + FLAT baseline arm (radix64 n1b2 + t2s, the 8849ns champion)
 *          + live MKL-IL arm. Reconfirms the baselines fresh in-session.
 *   [next] CASCADE arm: fill run_cascade()/fill_cascade_tw() from the 3-way
 *          gated derivation (per-stage Ls/OLs/count/VTW2), flip CASCADE_READY.
 *
 * Build: python build.py --src benches/zil_cascade.c --mkl
 * Pin:   logical CPU 2 (P-core), affinity mask 4, HIGH_PRIORITY_CLASS.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <windows.h>
#include <mkl_dfti.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* Cascade wired from the 3-way gated derivation (workflow wz3pyemlg, arm A:
 * self-sorting radix-8 four-step, NATURAL output, only n1+t2 + one corner-turn.
 * Scalar proto cascade_proto_A.c gated 3.06e-15 @4096). */
#define CASCADE_READY 1

#define D(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long, \
    unsigned long long, unsigned long long, unsigned long long);
/* flat baseline (64x64 two-pass) */
D(radix64_z_n1b2_fwd_avx2) D(radix64_z_t2s_fwd_avx2)
/* cascade leaves (radix-8 stages) */
D(radix8_z_n1_fwd_avx2) D(radix8_z_t2_fwd_avx2) D(radix8_z_t2s_fwd_avx2)

enum { N = 4096 };

static double now_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}
static void cachebust(void)
{
    size_t s = 32u * 1024u * 1024u / 8u;
    double *j = (double *)malloc(s * 8);
    volatile double a = 0;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a; free(j);
}

/* ── FLAT baseline: 64x64 four-step two-pass (the current champion) ──
 * pass1 radix64 n1b2 (Ls=64,count=64), pass2 radix64 t2s (W_4096^{l*k}), 126KB VTW2. */
static double *flat_tw;   /* 32 col-pairs * 63 legs * 8 doubles */
static void fill_flat_tw(void)
{
    flat_tw = (double *)_mm_malloc((size_t)32 * 63 * 8 * 8, 64);
    for (int p = 0; p < 32; p++)
        for (int l = 1; l < 64; l++) {
            double *rec = flat_tw + ((size_t)p * 63 + (l - 1)) * 8;
            for (int jj = 0; jj < 2; jj++) {
                int k = 2 * p + jj;
                double a = -2.0 * M_PI * (double)((long)l * k) / (double)N;
                rec[2 * jj] = cos(a); rec[2 * jj + 1] = cos(a);
                rec[4 + 2 * jj] = -sin(a); rec[4 + 2 * jj + 1] = sin(a);
            }
        }
}
static void run_flat(const double *z0, double *zs, double *out)
{
    radix64_z_n1b2_fwd_avx2(z0, 0, zs, 0, 0, 0, 64, 0, 64, 0, 64);
    radix64_z_t2s_fwd_avx2(zs, 0, out, 0, flat_tw, 0, 1, 64, 64, 1, 64);
}

/* ── CASCADE arm (radix-8 four-step, arm A) ──────────────────────────────
 * N=4096=8^4. Flat unroll of the gated recursive proto, twiddles FOLDED into
 * each t2 pre-multiply (no separate twiddle passes), pure ping-pong + 1 CT:
 *   z0->A ; S0 n1 A->A ; S1 t2 A->B ; S2 t2 B->A ; TR A->B ; S3 t2 B->out
 * Stage twiddles (pre-mult leg j=1..7, col k): S1 W_64^{j*m1} (col-const, per
 * group m1=0..7), S2 W_512^{j*m1} (col-const, m1=0..63), S3 W_4096^{j*k}
 * (per-column). VTW2 record = 56 doubles/col-pair, leg (l-1)*8, cos-first. */
#if CASCADE_READY
static double *tw_s1;   /* 8 groups  * 32 colpairs * 56 */
static double *tw_s2;   /* 64 groups *  4 colpairs * 56 */
static double *tw_s3;   /* 1         * 256 colpairs* 56 */

/* fill one VTW2 record (8 doubles) for a col-pair, angles a0,a1 = -2pi*e/M */
static void vtw2_rec(double *rec, double a0, double a1)
{
    rec[0] = cos(a0); rec[1] = cos(a0); rec[2] = cos(a1); rec[3] = cos(a1);
    rec[4] = -sin(a0); rec[5] = sin(a0); rec[6] = -sin(a1); rec[7] = sin(a1);
}
static void fill_cascade_tw(void)
{
    tw_s1 = (double *)_mm_malloc((size_t)8 * 32 * 56 * 8, 64);
    tw_s2 = (double *)_mm_malloc((size_t)64 * 4 * 56 * 8, 64);
    tw_s3 = (double *)_mm_malloc((size_t)256 * 56 * 8, 64);
    const double TAU = 2.0 * M_PI;

    /* S1: per group m1=0..7, col-const W_64^{l*m1}, count=64 -> 32 colpairs */
    for (int m1 = 0; m1 < 8; m1++) {
        double *g = tw_s1 + (size_t)m1 * 32 * 56;
        for (int p = 0; p < 32; p++)
            for (int l = 1; l < 8; l++) {
                long e = ((long)l * m1) % 64;
                double a = -TAU * (double)e / 64.0;
                vtw2_rec(g + (size_t)p * 56 + (l - 1) * 8, a, a);
            }
    }
    /* S2: per group m1=0..63, col-const W_512^{l*m1}, count=8 -> 4 colpairs */
    for (int m1 = 0; m1 < 64; m1++) {
        double *g = tw_s2 + (size_t)m1 * 4 * 56;
        for (int p = 0; p < 4; p++)
            for (int l = 1; l < 8; l++) {
                long e = ((long)l * m1) % 512;
                double a = -TAU * (double)e / 512.0;
                vtw2_rec(g + (size_t)p * 56 + (l - 1) * 8, a, a);
            }
    }
    /* S3: single stream, per-column W_4096^{l*k}, count=512 -> 256 colpairs */
    for (int p = 0; p < 256; p++)
        for (int l = 1; l < 8; l++) {
            int k0 = 2 * p, k1 = 2 * p + 1;
            double a0 = -TAU * (double)(((long)l * k0) % 4096) / 4096.0;
            double a1 = -TAU * (double)(((long)l * k1) % 4096) / 4096.0;
            vtw2_rec(tw_s3 + (size_t)p * 56 + (l - 1) * 8, a0, a1);
        }
}

/* corner-turn A[m1*8+p] -> B[p*512+m1] (complex, [512][8]->[8][512]) */
static void transpose_512x8(const double *A, double *B)
{
    for (int m1 = 0; m1 < 512; m1++)
        for (int p = 0; p < 8; p++) {
            size_t s = (size_t)m1 * 8 + p, d = (size_t)p * 512 + m1;
            B[2 * d] = A[2 * s]; B[2 * d + 1] = A[2 * s + 1];
        }
}

static void run_cascade(const double *z0, double *A, double *B, double *out)
{
    /* S0: n1, z0->A out-of-place (no copy) */
    radix8_z_n1_fwd_avx2(z0, 0, A, 0, 0, 0, 512, 0, 512, 0, 512);
    /* S1: t2 W_64, 8 groups, A->B */
    for (int m1 = 0; m1 < 8; m1++)
        radix8_z_t2_fwd_avx2(A + 2 * ((size_t)m1 * 512), 0, B + 2 * ((size_t)m1 * 64), 0,
                             tw_s1 + (size_t)m1 * 32 * 56, 0, 64, 0, 512, 0, 64);
    /* S2: t2 W_512, 64 groups, B->A */
    for (int m1 = 0; m1 < 64; m1++)
        radix8_z_t2_fwd_avx2(B + 2 * ((size_t)m1 * 64), 0, A + 2 * ((size_t)m1 * 8), 0,
                             tw_s2 + (size_t)m1 * 4 * 56, 0, 8, 0, 512, 0, 8);
    /* S3: t2s W_4096 per-col — strided READ A[k*8+j] (Ls=1,Gs=8) folds the
     * corner-turn into the load; natural out. Replaces transpose + t2. */
    radix8_z_t2s_fwd_avx2(A, 0, out, 0, tw_s3, 0, 1, 8, 512, 1, 512);
}

/* ── arm-B: grid-preserving IN-PLACE cascade + t2s gather terminator ──────
 * Middle stages in-place, small stride (Ls==OLs=64 then 8) => L1-hot blocks,
 * MKL-like contiguous streaming. Digit-reversed output. Twiddles: S1
 * W_64^{l*g}, S2 W_512^{l*digitrev2(g)}, S3 W_4096^{l*digitrev3(k)}. */
static double *tw_s1b, *tw_s2b, *tw_s3b;
static long digitrev(long x, int nf, int R)
{ long r = 0; for (int i = 0; i < nf; i++) { r = r * R + (x % R); x /= R; } return r; }
static void fill_cascadeB_tw(void)
{
    tw_s1b = (double *)_mm_malloc((size_t)8 * 32 * 56 * 8, 64);
    tw_s2b = (double *)_mm_malloc((size_t)64 * 4 * 56 * 8, 64);
    tw_s3b = (double *)_mm_malloc((size_t)256 * 56 * 8, 64);
    const double TAU = 2.0 * M_PI;
    for (int g = 0; g < 8; g++) {   /* S1 col-const W_64^{l*g} */
        double *gp = tw_s1b + (size_t)g * 32 * 56;
        for (int p = 0; p < 32; p++)
            for (int l = 1; l < 8; l++) {
                double a = -TAU * (double)(((long)l * g) % 64) / 64.0;
                vtw2_rec(gp + (size_t)p * 56 + (l - 1) * 8, a, a);
            }
    }
    for (int g = 0; g < 64; g++) {  /* S2 col-const W_512^{l*digitrev2(g)} */
        long gr = digitrev(g, 2, 8);
        double *gp = tw_s2b + (size_t)g * 4 * 56;
        for (int p = 0; p < 4; p++)
            for (int l = 1; l < 8; l++) {
                double a = -TAU * (double)(((long)l * gr) % 512) / 512.0;
                vtw2_rec(gp + (size_t)p * 56 + (l - 1) * 8, a, a);
            }
    }
    for (int p = 0; p < 256; p++)   /* S3 per-col W_4096^{l*digitrev3(k)} */
        for (int l = 1; l < 8; l++) {
            long r0 = digitrev(2 * p, 3, 8), r1 = digitrev(2 * p + 1, 3, 8);
            double a0 = -TAU * (double)(((long)l * r0) % 4096) / 4096.0;
            double a1 = -TAU * (double)(((long)l * r1) % 4096) / 4096.0;
            vtw2_rec(tw_s3b + (size_t)p * 56 + (l - 1) * 8, a0, a1);
        }
}
static void run_cascadeB(const double *z0, double *A, double *B, double *out)
{
    /* S0: n1 z0->A */
    radix8_z_n1_fwd_avx2(z0, 0, A, 0, 0, 0, 512, 0, 512, 0, 512);
    /* S1: t2 in-place idx, 8 groups base g*512, Ls=OLs=64, A->B */
    for (int g = 0; g < 8; g++)
        radix8_z_t2_fwd_avx2(A + 2 * ((size_t)g * 512), 0, B + 2 * ((size_t)g * 512), 0,
                             tw_s1b + (size_t)g * 32 * 56, 0, 64, 0, 64, 0, 64);
    /* S2: t2, 64 groups base (g%8)*64+(g/8)*512, Ls=OLs=8, B->A */
    for (int g = 0; g < 64; g++) {
        size_t base = (size_t)(g % 8) * 64 + (size_t)(g / 8) * 512;
        radix8_z_t2_fwd_avx2(B + 2 * base, 0, A + 2 * base, 0,
                             tw_s2b + (size_t)g * 4 * 56, 0, 8, 0, 8, 0, 8);
    }
    /* S3: t2s gather terminator, Ls=1,Gs=8,OLs=512, A->out (digit-reversed) */
    radix8_z_t2s_fwd_avx2(A, 0, out, 0, tw_s3b, 0, 1, 8, 512, 1, 512);
}
#endif

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    double *z0 = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *z  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *A  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *B  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *Rr = (double *)_mm_malloc((size_t)N * 8, 64);
    double *Ri = (double *)_mm_malloc((size_t)N * 8, 64);
    srand(4096);
    for (int i = 0; i < 2 * N; i++) z0[i] = (double)rand() / RAND_MAX - 0.5;

    fill_flat_tw();
#if CASCADE_READY
    fill_cascade_tw();
    fill_cascadeB_tw();
#endif

    /* naive DFT reference (natural order) */
    for (int m = 0; m < N; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * (double)((long)n * m % N) / (double)N;
            double c = cos(a), s = sin(a);
            sr += z0[2 * n] * c - z0[2 * n + 1] * s;
            si += z0[2 * n] * s + z0[2 * n + 1] * c;
        }
        Rr[m] = sr; Ri[m] = si;
    }

    /* ── GATES ── */
    {
        run_flat(z0, A, z);
        double err = 0, mag = 0;
        for (int m = 0; m < N; m++) {
            double d = fabs(z[2 * m] - Rr[m]) + fabs(z[2 * m + 1] - Ri[m]);
            if (d > err) err = d;
            double g = fabs(Rr[m]) + fabs(Ri[m]);
            if (g > mag) mag = g;
        }
        printf("GATE flat(64x64)    relerr=%.3e %s\n", err / mag,
               (err / mag < 1e-12) ? "PASS" : "FAIL");
    }
#if CASCADE_READY
    {
        run_cascade(z0, A, B, z);
        double err = 0, mag = 0;
        for (int m = 0; m < N; m++) {
            double d = fabs(z[2 * m] - Rr[m]) + fabs(z[2 * m + 1] - Ri[m]);
            if (d > err) err = d;
            double g = fabs(Rr[m]) + fabs(Ri[m]);
            if (g > mag) mag = g;
        }
        printf("GATE cascadeA       relerr=%.3e %s\n", err / mag,
               (err / mag < 1e-12) ? "PASS" : "FAIL");
        if (err / mag >= 1e-12) return 1;
    }
    {   /* arm-B: digit-reversed output — compare at permuted index */
        run_cascadeB(z0, A, B, z);
        double err = 0, mag = 0;
        for (int idx = 0; idx < N; idx++) {
            long l = idx / 512, g = idx % 512, aA = g * 8 + l;
            int m = (int)digitrev(aA, 4, 8);
            double d = fabs(z[2 * idx] - Rr[m]) + fabs(z[2 * idx + 1] - Ri[m]);
            if (d > err) err = d;
            double gg = fabs(Rr[m]) + fabs(Ri[m]);
            if (gg > mag) mag = gg;
        }
        printf("GATE cascadeB       relerr=%.3e %s\n", err / mag,
               (err / mag < 1e-12) ? "PASS" : "FAIL");
        if (err / mag >= 1e-12) return 1;
    }
#else
    printf("GATE cascade        PENDING (verified stage-math in flight)\n");
#endif

    /* ── RACE: best-of-7, cachebust between trials, arm order flipped ── */
    DFTI_DESCRIPTOR_HANDLE h = NULL;
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiCommitDescriptor(h);

    enum { ARM_FLAT = 0, ARM_CASCA = 1, ARM_CASCB = 2, ARM_MKL = 3, NARM = 4 };
    int reps = 2500;
    double best[NARM] = { 1e18, 1e18, 1e18, 1e18 };
    for (int t = 0; t < 9; t++) {
        if (t) cachebust();
        for (int q = 0; q < NARM; q++) {
            int a = (t & 1) ? (NARM - 1 - q) : q;
            double t0, ns;
            if (a == ARM_FLAT) {
                for (int w = 0; w < 8; w++) run_flat(z0, A, z);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) run_flat(z0, A, z);
                ns = (now_ms() - t0) * 1e6 / reps;
            } else if (a == ARM_CASCA) {
                for (int w = 0; w < 8; w++) run_cascade(z0, A, B, z);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) run_cascade(z0, A, B, z);
                ns = (now_ms() - t0) * 1e6 / reps;
            } else if (a == ARM_CASCB) {
                for (int w = 0; w < 8; w++) run_cascadeB(z0, A, B, z);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) run_cascadeB(z0, A, B, z);
                ns = (now_ms() - t0) * 1e6 / reps;
            } else {
                for (int w = 0; w < 8; w++) DftiComputeForward(h, z);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) DftiComputeForward(h, z);
                ns = (now_ms() - t0) * 1e6 / reps;
            }
            if (ns < best[a]) best[a] = ns;
        }
    }

    printf("\n# N=4096 z cascade (A=natural four-step, B=in-place grid) vs flat vs MKL-IL\n");
    printf("flat 64x64 two-pass   %8.1f ns   vsMKL %.2f\n", best[ARM_FLAT], best[ARM_MKL] / best[ARM_FLAT]);
    printf("z cascade A (natural) %8.1f ns   vsMKL %.2f   vsFlat %.2f\n",
           best[ARM_CASCA], best[ARM_MKL] / best[ARM_CASCA], best[ARM_FLAT] / best[ARM_CASCA]);
    printf("z cascade B (inplace) %8.1f ns   vsMKL %.2f   vsFlat %.2f\n",
           best[ARM_CASCB], best[ARM_MKL] / best[ARM_CASCB], best[ARM_FLAT] / best[ARM_CASCB]);
    printf("MKL-IL                %8.1f ns\n", best[ARM_MKL]);
    printf("DONE\n");
    return 0;
}
