/* zil_cascade_baked.c — LEVER 4 (AOT bake) measurement: fused single-TU
 * executors for the z cascade winners vs the call-per-group driver vs MKL.
 *
 * The baked arm #includes the needed z codelets (macro-renamed) so gcc -O3
 * inlines the bodies into constant-trip stage loops with immediate strides —
 * MKL's fused-function shape (census-verified AOT, not JIT). Three arms
 * isolate the lever:
 *   drv   = driver-as-is (runtime base_of() in the hot path, extern calls)
 *   drvT  = driver with precomputed base tables (extern calls remain)
 *   baked = fused TU (inlined bodies, constant trip counts, tabled bases)
 * Chains (r8-family, file-scope-const-clean kernels only):
 *   N= 4096: 8.8.8.8   t2c mids + t2sp  term  (within 3% of the 4096 winner)
 *   N=16384: 4.8.8.8.8 t2c mids + t2spt term  (THE 16384 winner)
 * Production path: the OCaml emit_executor_h.ml pattern (wisdom -> emitted
 * executors) gets a z-chain backend; this bench only prices the lever.
 *
 * Build: python build.py --src benches/zil_cascade_baked.c --mkl
 * Run:   zil_cascade_baked.exe [N]   (4096 | 16384)
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

/* ── baked kernel copies: macro-rename + include (single TU => inlinable).
 * These bodies have NO file-scope consts except _M_IM, which we rename. ── */
#define _M_IM _MIM_bk1
#define radix4_z_n1_fwd_avx2 bk_r4n1
#include "../../src/dag-fft-compiler/codelets/zil/avx2/pure_il/radix4_z_n1_avx2.c"
#undef _M_IM
#undef radix4_z_n1_fwd_avx2

#define _M_IM _MIM_bk2
#define radix8_z_n1_fwd_avx2 bk_r8n1
#include "../../src/dag-fft-compiler/codelets/zil/avx2/pure_il/radix8_z_n1_avx2.c"
#undef _M_IM
#undef radix8_z_n1_fwd_avx2

#define _M_IM _MIM_bk3
#define radix8_z_t2c_fwd_avx2 bk_r8t2c
#include "../../src/dag-fft-compiler/codelets/zil/avx2/pure_il/radix8_z_t2c_avx2.c"
#undef _M_IM
#undef radix8_z_t2c_fwd_avx2

#define _M_IM _MIM_bk4
#define radix8_z_t2sp_fwd_avx2 bk_r8t2sp
#include "../../src/dag-fft-compiler/codelets/zil/avx2/pure_il/radix8_z_t2sp_avx2.c"
#undef _M_IM
#undef radix8_z_t2sp_fwd_avx2

#define _M_IM _MIM_bk5
#define radix8_z_t2spt_fwd_avx2 bk_r8t2spt
#include "../../src/dag-fft-compiler/codelets/zil/avx2/pure_il/radix8_z_t2spt_avx2.c"
#undef _M_IM
#undef radix8_z_t2spt_fwd_avx2

/* ── extern (library) kernels for the driver arms ── */
typedef void (*zfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    unsigned long long, unsigned long long,
                    unsigned long long, unsigned long long, unsigned long long);
#define D(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long, \
    unsigned long long, unsigned long long, unsigned long long);
D(radix4_z_n1_fwd_avx2) D(radix8_z_n1_fwd_avx2) D(radix8_z_t2c_fwd_avx2)
D(radix8_z_t2sp_fwd_avx2) D(radix8_z_t2spt_fwd_avx2)

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

/* ── chain plumbing (mirrors zil_chain_dp.c) ── */
#define MAXNF 8
static int N, NF;
static int R[MAXNF];
static long DD[MAXNF], GG[MAXNF];

static long drev_full(long x, const int *r, int nf)
{
    long v = 0;
    for (int i = nf - 1; i >= 0; i--) { v = v * r[i] + (x % r[i]); x /= r[i]; }
    return v;
}
static long brev_prefix(long g, int s, const int *r)
{
    long f[MAXNF];
    for (int i = s - 1; i >= 0; i--) { f[i] = g % r[i]; g /= r[i]; }
    long P = 1, v = 0;
    for (int i = 0; i < s; i++) { v += f[i] * P; P *= r[i]; }
    return v;
}
static long base_of(long g, int s, const int *r, const long *d)
{
    long b = 0;
    for (int i = s - 1; i >= 0; i--) { long f = g % r[i]; g /= r[i]; b += f * d[i]; }
    return b;
}
static void vtw2_rec(double *rec, double a0, double a1)
{
    rec[0] = cos(a0); rec[1] = cos(a0); rec[2] = cos(a1); rec[3] = cos(a1);
    rec[4] = -sin(a0); rec[5] = sin(a0); rec[6] = -sin(a1); rec[7] = sin(a1);
}

static double *twc[MAXNF];      /* per-stage group-constant record sets */
static double *twp1;            /* terminator w^1 stream */
static long   *gbase[MAXNF];    /* precomputed group bases (drvT + baked) */

static void build_tables(void)
{
    const double TAU = 2.0 * M_PI;
    DD[NF - 1] = 1;
    for (int i = NF - 2; i >= 0; i--) DD[i] = DD[i + 1] * R[i + 1];
    GG[0] = 1;
    for (int i = 1; i < NF; i++) GG[i] = GG[i - 1] * R[i - 1];
    for (int s = 1; s <= NF - 2; s++) {
        size_t rl = (size_t)(R[s] - 1) * 8;
        long M = (long)N / DD[s];
        twc[s] = (double *)_mm_malloc((size_t)GG[s] * rl * 8, 64);
        gbase[s] = (long *)malloc((size_t)GG[s] * sizeof(long));
        for (long g = 0; g < GG[s]; g++) {
            gbase[s][g] = base_of(g, s, R, DD);
            long brev = brev_prefix(g, s, R);
            for (int l = 1; l < R[s]; l++) {
                double a = -TAU * (double)(((long)l * brev) % M) / (double)M;
                vtw2_rec(twc[s] + (size_t)g * rl + (size_t)(l - 1) * 8, a, a);
            }
        }
    }
    {
        int Rt = R[NF - 1];
        long pairs = ((long)N / Rt) / 2;
        twp1 = (double *)_mm_malloc((size_t)pairs * 64, 64);
        for (long p = 0; p < pairs; p++) {
            long b0 = brev_prefix(2 * p, NF - 1, R);
            long b1 = brev_prefix(2 * p + 1, NF - 1, R);
            double a0 = -TAU * (double)(b0 % N) / (double)N;
            double a1 = -TAU * (double)(b1 % N) / (double)N;
            vtw2_rec(twp1 + (size_t)p * 8, a0, a1);
        }
    }
}

/* ── arm 1: driver-as-is (base_of in hot path, extern calls) ── */
static void run_drv(const double *z0, double *A, double *out)
{
    zfn s0 = (R[0] == 4) ? radix4_z_n1_fwd_avx2 : radix8_z_n1_fwd_avx2;
    s0(z0, 0, A, 0, 0, 0, (unsigned long long)DD[0], 0,
       (unsigned long long)DD[0], 0, (unsigned long long)DD[0]);
    for (int s = 1; s <= NF - 2; s++) {
        size_t rl = (size_t)(R[s] - 1) * 8;
        for (long g = 0; g < GG[s]; g++) {
            long b = base_of(g, s, R, DD);
            radix8_z_t2c_fwd_avx2(A + 2 * b, 0, A + 2 * b, 0,
                                  twc[s] + (size_t)g * rl, 0,
                                  (unsigned long long)DD[s], 0,
                                  (unsigned long long)DD[s], 0,
                                  (unsigned long long)DD[s]);
        }
    }
    zfn tf = (N == 16384) ? radix8_z_t2spt_fwd_avx2 : radix8_z_t2sp_fwd_avx2;
    tf(A, 0, out, 0, twp1, 0, 1, 8, (unsigned long long)(N / 8), 1,
       (unsigned long long)(N / 8));
}

/* ── arm 2: driver with precomputed base tables (extern calls remain) ── */
static void run_drvT(const double *z0, double *A, double *out)
{
    zfn s0 = (R[0] == 4) ? radix4_z_n1_fwd_avx2 : radix8_z_n1_fwd_avx2;
    s0(z0, 0, A, 0, 0, 0, (unsigned long long)DD[0], 0,
       (unsigned long long)DD[0], 0, (unsigned long long)DD[0]);
    for (int s = 1; s <= NF - 2; s++) {
        size_t rl = (size_t)(R[s] - 1) * 8;
        const long *gb = gbase[s];
        for (long g = 0; g < GG[s]; g++) {
            long b = gb[g];
            radix8_z_t2c_fwd_avx2(A + 2 * b, 0, A + 2 * b, 0,
                                  twc[s] + (size_t)g * rl, 0,
                                  (unsigned long long)DD[s], 0,
                                  (unsigned long long)DD[s], 0,
                                  (unsigned long long)DD[s]);
        }
    }
    zfn tf = (N == 16384) ? radix8_z_t2spt_fwd_avx2 : radix8_z_t2sp_fwd_avx2;
    tf(A, 0, out, 0, twp1, 0, 1, 8, (unsigned long long)(N / 8), 1,
       (unsigned long long)(N / 8));
}

/* ── arm 3: BAKED fused executors (inlined bodies, immediate strides) ── */
static void baked_4096(const double *z0, double *A, double *out)
{
    /* 8.8.8.8, t2c mids + t2sp term; D = [512,64,8,1], G = [1,8,64,512] */
    bk_r8n1(z0, 0, A, 0, 0, 0, 512, 0, 512, 0, 512);
    for (long g = 0; g < 8; g++)
        bk_r8t2c(A + 2 * gbase[1][g], 0, A + 2 * gbase[1][g], 0,
                 twc[1] + (size_t)g * 56, 0, 64, 0, 64, 0, 64);
    for (long g = 0; g < 64; g++)
        bk_r8t2c(A + 2 * gbase[2][g], 0, A + 2 * gbase[2][g], 0,
                 twc[2] + (size_t)g * 56, 0, 8, 0, 8, 0, 8);
    bk_r8t2sp(A, 0, out, 0, twp1, 0, 1, 8, 512, 1, 512);
}
static void baked_16384(const double *z0, double *A, double *out)
{
    /* 4.8.8.8.8, t2c mids + t2spt term; D = [4096,512,64,8,1], G = [1,4,32,256,2048] */
    bk_r4n1(z0, 0, A, 0, 0, 0, 4096, 0, 4096, 0, 4096);
    for (long g = 0; g < 4; g++)
        bk_r8t2c(A + 2 * gbase[1][g], 0, A + 2 * gbase[1][g], 0,
                 twc[1] + (size_t)g * 56, 0, 512, 0, 512, 0, 512);
    for (long g = 0; g < 32; g++)
        bk_r8t2c(A + 2 * gbase[2][g], 0, A + 2 * gbase[2][g], 0,
                 twc[2] + (size_t)g * 56, 0, 64, 0, 64, 0, 64);
    for (long g = 0; g < 256; g++)
        bk_r8t2c(A + 2 * gbase[3][g], 0, A + 2 * gbase[3][g], 0,
                 twc[3] + (size_t)g * 56, 0, 8, 0, 8, 0, 8);
    bk_r8t2spt(A, 0, out, 0, twp1, 0, 1, 8, 2048, 1, 2048);
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    N = argc > 1 ? atoi(argv[1]) : 4096;
    if (N == 4096)      { NF = 4; R[0] = 8; R[1] = 8; R[2] = 8; R[3] = 8; }
    else if (N == 16384){ NF = 5; R[0] = 4; R[1] = 8; R[2] = 8; R[3] = 8; R[4] = 8; }
    else { printf("N must be 4096 or 16384\n"); return 1; }
    build_tables();

    double *z0 = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *A  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *z  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *Rr = (double *)_mm_malloc((size_t)N * 8, 64);
    double *Ri = (double *)_mm_malloc((size_t)N * 8, 64);
    srand(N);
    for (int i = 0; i < 2 * N; i++) z0[i] = (double)rand() / RAND_MAX - 0.5;
    for (int m = 0; m < N; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * (double)(((long)n * m) % N) / (double)N;
            double cc = cos(a), ss = sin(a);
            sr += z0[2 * n] * cc - z0[2 * n + 1] * ss;
            si += z0[2 * n] * ss + z0[2 * n + 1] * cc;
        }
        Rr[m] = sr; Ri[m] = si;
    }
    double mag = 0;
    for (int m = 0; m < N; m++) {
        double g = fabs(Rr[m]) + fabs(Ri[m]);
        if (g > mag) mag = g;
    }

    /* gates: all three arms vs naive (digit-reversed) */
    const char *AN[3] = { "drv  ", "drvT ", "baked" };
    for (int a = 0; a < 3; a++) {
        if (a == 0) run_drv(z0, A, z);
        else if (a == 1) run_drvT(z0, A, z);
        else if (N == 4096) baked_4096(z0, A, z);
        else baked_16384(z0, A, z);
        long NR = (long)N / 8;
        double err = 0;
        for (long idx = 0; idx < N; idx++) {
            long l = idx / NR, g = idx % NR;
            long m = drev_full(g * 8 + l, R, NF);
            double d = fabs(z[2 * idx] - Rr[m]) + fabs(z[2 * idx + 1] - Ri[m]);
            if (d > err) err = d;
        }
        printf("GATE %s relerr=%.3e %s\n", AN[a], err / mag,
               (err / mag < 1e-11) ? "PASS" : "FAIL");
        if (err / mag >= 1e-11) return 1;
    }

    DFTI_DESCRIPTOR_HANDLE h = NULL;
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiCommitDescriptor(h);
    int reps = (int)(4.0e6 / N); if (reps < 200) reps = 200;
    double best[4] = { 1e18, 1e18, 1e18, 1e18 };
    for (int t = 0; t < 9; t++) {
        if (t) cachebust();
        for (int q = 0; q < 4; q++) {
            int a = (t & 1) ? 3 - q : q;
            double t0, ns;
            if (a == 0) {
                for (int w = 0; w < 6; w++) run_drv(z0, A, z);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) run_drv(z0, A, z);
            } else if (a == 1) {
                for (int w = 0; w < 6; w++) run_drvT(z0, A, z);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) run_drvT(z0, A, z);
            } else if (a == 2) {
                if (N == 4096) {
                    for (int w = 0; w < 6; w++) baked_4096(z0, A, z);
                    t0 = now_ms();
                    for (int i = 0; i < reps; i++) baked_4096(z0, A, z);
                } else {
                    for (int w = 0; w < 6; w++) baked_16384(z0, A, z);
                    t0 = now_ms();
                    for (int i = 0; i < reps; i++) baked_16384(z0, A, z);
                }
            } else {
                for (int w = 0; w < 6; w++) DftiComputeForward(h, z);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) DftiComputeForward(h, z);
            }
            ns = (now_ms() - t0) * 1e6 / reps;
            if (ns < best[a]) best[a] = ns;
        }
    }
    printf("\n# N=%d LEVER-4 (fusion) attribution, chain %s\n", N,
           N == 4096 ? "8.8.8.8 t2c/t2sp" : "4.8.8.8.8 t2c/t2spt");
    printf("drv   (base_of hot)   %9.1f ns   vsMKL %.2f\n", best[0], best[3] / best[0]);
    printf("drvT  (tabled bases)  %9.1f ns   vsMKL %.2f\n", best[1], best[3] / best[1]);
    printf("baked (fused, inline) %9.1f ns   vsMKL %.2f   vs drv %.2f\n",
           best[2], best[3] / best[2], best[0] / best[2]);
    printf("MKL-IL                %9.1f ns\n", best[3]);
    printf("DONE\n");
    return 0;
}
