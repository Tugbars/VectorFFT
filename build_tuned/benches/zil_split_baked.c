/* zil_split_baked.c — PACED FINALS: the four per-cell split-interior winners
 * (z_cascade_plan §4.995) with BAKED fused executors (lever 4 stacked on
 * lever 5). Baked arm = hand kernels + #include-renamed z terminator, all
 * in-TU (gcc-inlined), constant trip counts, precomputed base tables.
 * Driver arm = the sweep's shape (runtime base_of, extern terminator).
 * Trials are PACED (Sleep 150ms/trial) per the dp_planner.h thermal lesson —
 * these are the wisdom-grade numbers.
 *
 * Winners baked:  2048 = 4.8.8.8   SPL/t2spt
 *                 4096 = 4.4.4.8.8 SPL/t2spt
 *                 8192 = 4.4.8.8.8 SPL/t2sp
 *                16384 = 4.8.8.8.8 SPL/t2spt
 *
 * Build: python build.py --src benches/zil_split_baked.c --mkl
 * Run:   zil_split_baked.exe [N]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <malloc.h>
#include <immintrin.h>
#include <windows.h>
#include <mkl_dfti.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ── baked terminator copies (const-clean r8 files; rename via macro) ── */
#define _M_IM _MIM_bk1
#define radix8_z_t2sp_fwd_avx2 bk_t2sp
#include "../../src/dag-fft-compiler/codelets/zil/avx2/pure_il/radix8_z_t2sp_avx2.c"
#undef _M_IM
#undef radix8_z_t2sp_fwd_avx2
#define _M_IM _MIM_bk2
#define radix8_z_t2spt_fwd_avx2 bk_t2spt
#include "../../src/dag-fft-compiler/codelets/zil/avx2/pure_il/radix8_z_t2spt_avx2.c"
#undef _M_IM
#undef radix8_z_t2spt_fwd_avx2

/* ── extern terminators for the driver arm ── */
typedef void (*zfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    unsigned long long, unsigned long long,
                    unsigned long long, unsigned long long, unsigned long long);
extern void radix8_z_t2sp_fwd_avx2(const double *, const double *, double *, double *,
    const double *, const double *, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
extern void radix8_z_t2spt_fwd_avx2(const double *, const double *, double *, double *,
    const double *, const double *, unsigned long long, unsigned long long,
    unsigned long long, unsigned long long, unsigned long long);
/* EMITTED split-family kernels (codelet_zil.ml emit_z_split — the promotion;
 * drv/drvT arms run THESE; the baked arm keeps the local hand copies, so
 * matching gates = the promotion bit-gate) */
#define DE(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long, \
    unsigned long long, unsigned long long, unsigned long long);
DE(radix4_z_s0s_fwd_avx2) DE(radix8_z_s0s_fwd_avx2)
DE(radix4_z_ms_fwd_avx2)  DE(radix8_z_ms_fwd_avx2)
DE(radix4_z_msz_fwd_avx2) DE(radix8_z_msz_fwd_avx2)

/* ── split kernel block (same as zil_chain_dp.c §4.995; static => inlinable) ── */
#define DEINT(zlo, zhi, re, im) do {                                  \
    __m256d _u = _mm256_unpacklo_pd(zlo, zhi);                        \
    __m256d _v = _mm256_unpackhi_pd(zlo, zhi);                        \
    re = _mm256_permute4x64_pd(_u, 0xD8);                             \
    im = _mm256_permute4x64_pd(_v, 0xD8);                             \
} while (0)
#define REINT(re, im, zlo, zhi) do {                                  \
    __m256d _p = _mm256_permute4x64_pd(re, 0xD8);                     \
    __m256d _q = _mm256_permute4x64_pd(im, 0xD8);                     \
    zlo = _mm256_unpacklo_pd(_p, _q);                                 \
    zhi = _mm256_unpackhi_pd(_p, _q);                                 \
} while (0)
#define SPLIT_BFLY4(i0r,i0i,i1r,i1i,i2r,i2i,i3r,i3i, o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i) do { \
    __m256d t0r=_mm256_add_pd(i0r,i2r), t0i=_mm256_add_pd(i0i,i2i);   \
    __m256d t1r=_mm256_sub_pd(i0r,i2r), t1i=_mm256_sub_pd(i0i,i2i);   \
    __m256d t2r=_mm256_add_pd(i1r,i3r), t2i=_mm256_add_pd(i1i,i3i);   \
    __m256d t3r=_mm256_sub_pd(i1r,i3r), t3i=_mm256_sub_pd(i1i,i3i);   \
    o0r=_mm256_add_pd(t0r,t2r); o0i=_mm256_add_pd(t0i,t2i);           \
    o2r=_mm256_sub_pd(t0r,t2r); o2i=_mm256_sub_pd(t0i,t2i);           \
    o1r=_mm256_add_pd(t1r,t3i); o1i=_mm256_sub_pd(t1i,t3r);           \
    o3r=_mm256_sub_pd(t1r,t3i); o3i=_mm256_add_pd(t1i,t3r);           \
} while (0)
#define SPLIT_CMUL(ar,ai, ct,st, or_,oi_) do {                        \
    or_ = _mm256_fnmadd_pd(st, ai, _mm256_mul_pd(ct, ar));            \
    oi_ = _mm256_fmadd_pd(st, ar, _mm256_mul_pd(ct, ai));             \
} while (0)
#define SPLIT_BFLY8(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i, \
                    o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i) do { \
    const __m256d _C = _mm256_set1_pd(0.70710678118654752440);        \
    __m256d t0r=_mm256_add_pd(x0r,x4r), t0i=_mm256_add_pd(x0i,x4i);   \
    __m256d t1r=_mm256_sub_pd(x0r,x4r), t1i=_mm256_sub_pd(x0i,x4i);   \
    __m256d t2r=_mm256_add_pd(x2r,x6r), t2i=_mm256_add_pd(x2i,x6i);   \
    __m256d t3r=_mm256_sub_pd(x2r,x6r), t3i=_mm256_sub_pd(x2i,x6i);   \
    __m256d E0r=_mm256_add_pd(t0r,t2r), E0i=_mm256_add_pd(t0i,t2i);   \
    __m256d E2r=_mm256_sub_pd(t0r,t2r), E2i=_mm256_sub_pd(t0i,t2i);   \
    __m256d E1r=_mm256_add_pd(t1r,t3i), E1i=_mm256_sub_pd(t1i,t3r);   \
    __m256d E3r=_mm256_sub_pd(t1r,t3i), E3i=_mm256_add_pd(t1i,t3r);   \
    __m256d s0r=_mm256_add_pd(x1r,x5r), s0i=_mm256_add_pd(x1i,x5i);   \
    __m256d s1r=_mm256_sub_pd(x1r,x5r), s1i=_mm256_sub_pd(x1i,x5i);   \
    __m256d s2r=_mm256_add_pd(x3r,x7r), s2i=_mm256_add_pd(x3i,x7i);   \
    __m256d s3r=_mm256_sub_pd(x3r,x7r), s3i=_mm256_sub_pd(x3i,x7i);   \
    __m256d O0r=_mm256_add_pd(s0r,s2r), O0i=_mm256_add_pd(s0i,s2i);   \
    __m256d O2r=_mm256_sub_pd(s0r,s2r), O2i=_mm256_sub_pd(s0i,s2i);   \
    __m256d O1r=_mm256_add_pd(s1r,s3i), O1i=_mm256_sub_pd(s1i,s3r);   \
    __m256d O3r=_mm256_sub_pd(s1r,s3i), O3i=_mm256_add_pd(s1i,s3r);   \
    __m256d X1r=_mm256_add_pd(O1r,O1i), X1i=_mm256_sub_pd(O1i,O1r);   \
    __m256d X3r=_mm256_sub_pd(O3i,O3r), X3n=_mm256_add_pd(O3r,O3i);   \
    o0r=_mm256_add_pd(E0r,O0r); o0i=_mm256_add_pd(E0i,O0i);           \
    o4r=_mm256_sub_pd(E0r,O0r); o4i=_mm256_sub_pd(E0i,O0i);           \
    o1r=_mm256_fmadd_pd(_C,X1r,E1r); o1i=_mm256_fmadd_pd(_C,X1i,E1i); \
    o5r=_mm256_fnmadd_pd(_C,X1r,E1r); o5i=_mm256_fnmadd_pd(_C,X1i,E1i); \
    o2r=_mm256_add_pd(E2r,O2i); o2i=_mm256_sub_pd(E2i,O2r);           \
    o6r=_mm256_sub_pd(E2r,O2i); o6i=_mm256_add_pd(E2i,O2r);           \
    o3r=_mm256_fmadd_pd(_C,X3r,E3r); o3i=_mm256_fnmadd_pd(_C,X3n,E3i); \
    o7r=_mm256_fnmadd_pd(_C,X3r,E3r); o7i=_mm256_fmadd_pd(_C,X3n,E3i); \
} while (0)

__attribute__((target("avx2,fma")))
static void s0z2s_r4(const double *z, double *sp, long Ls, long count)
{
    for (long k = 0; k + 4 <= count; k += 4) {
        __m256d xr[4], xi[4], or_[4], oi_[4];
        for (int l = 0; l < 4; l++) {
            __m256d zlo = _mm256_loadu_pd(z + 2 * ((size_t)l * Ls + k));
            __m256d zhi = _mm256_loadu_pd(z + 2 * ((size_t)l * Ls + k) + 4);
            DEINT(zlo, zhi, xr[l], xi[l]);
        }
        SPLIT_BFLY4(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3]);
        for (int l = 0; l < 4; l++) {
            _mm256_storeu_pd(sp + 2 * ((size_t)l * Ls + k), or_[l]);
            _mm256_storeu_pd(sp + 2 * ((size_t)l * Ls + k) + 4, oi_[l]);
        }
    }
}
__attribute__((target("avx2,fma")))
static void mid_s2s_r4(double *sp, const double *tw, long Dv, long count)
{
    for (long k = 0; k + 4 <= count; k += 4) {
        __m256d xr[4], xi[4], or_[4], oi_[4];
        xr[0] = _mm256_loadu_pd(sp + 2 * (size_t)k);
        xi[0] = _mm256_loadu_pd(sp + 2 * (size_t)k + 4);
        for (int l = 1; l < 4; l++) {
            __m256d ar = _mm256_loadu_pd(sp + 2 * ((size_t)l * Dv + k));
            __m256d ai = _mm256_loadu_pd(sp + 2 * ((size_t)l * Dv + k) + 4);
            __m256d ct = _mm256_loadu_pd(tw + (size_t)(l - 1) * 8);
            __m256d st = _mm256_loadu_pd(tw + (size_t)(l - 1) * 8 + 4);
            SPLIT_CMUL(ar, ai, ct, st, xr[l], xi[l]);
        }
        SPLIT_BFLY4(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3]);
        for (int l = 0; l < 4; l++) {
            _mm256_storeu_pd(sp + 2 * ((size_t)l * Dv + k), or_[l]);
            _mm256_storeu_pd(sp + 2 * ((size_t)l * Dv + k) + 4, oi_[l]);
        }
    }
}
__attribute__((target("avx2,fma")))
static void mid_s2s_r8(double *sp, const double *tw, long Dv, long count)
{
    for (long k = 0; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8], or_[8], oi_[8];
        xr[0] = _mm256_loadu_pd(sp + 2 * (size_t)k);
        xi[0] = _mm256_loadu_pd(sp + 2 * (size_t)k + 4);
        for (int l = 1; l < 8; l++) {
            __m256d ar = _mm256_loadu_pd(sp + 2 * ((size_t)l * Dv + k));
            __m256d ai = _mm256_loadu_pd(sp + 2 * ((size_t)l * Dv + k) + 4);
            __m256d ct = _mm256_loadu_pd(tw + (size_t)(l - 1) * 8);
            __m256d st = _mm256_loadu_pd(tw + (size_t)(l - 1) * 8 + 4);
            SPLIT_CMUL(ar, ai, ct, st, xr[l], xi[l]);
        }
        SPLIT_BFLY8(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                    xr[4],xi[4],xr[5],xi[5],xr[6],xi[6],xr[7],xi[7],
                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3],
                    or_[4],oi_[4],or_[5],oi_[5],or_[6],oi_[6],or_[7],oi_[7]);
        for (int l = 0; l < 8; l++) {
            _mm256_storeu_pd(sp + 2 * ((size_t)l * Dv + k), or_[l]);
            _mm256_storeu_pd(sp + 2 * ((size_t)l * Dv + k) + 4, oi_[l]);
        }
    }
}
__attribute__((target("avx2,fma")))
static void mid_s2z_r8(const double *sp, double *z, const double *tw, long Dv, long count)
{
    for (long k = 0; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8], or_[8], oi_[8];
        xr[0] = _mm256_loadu_pd(sp + 2 * (size_t)k);
        xi[0] = _mm256_loadu_pd(sp + 2 * (size_t)k + 4);
        for (int l = 1; l < 8; l++) {
            __m256d ar = _mm256_loadu_pd(sp + 2 * ((size_t)l * Dv + k));
            __m256d ai = _mm256_loadu_pd(sp + 2 * ((size_t)l * Dv + k) + 4);
            __m256d ct = _mm256_loadu_pd(tw + (size_t)(l - 1) * 8);
            __m256d st = _mm256_loadu_pd(tw + (size_t)(l - 1) * 8 + 4);
            SPLIT_CMUL(ar, ai, ct, st, xr[l], xi[l]);
        }
        SPLIT_BFLY8(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                    xr[4],xi[4],xr[5],xi[5],xr[6],xi[6],xr[7],xi[7],
                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3],
                    or_[4],oi_[4],or_[5],oi_[5],or_[6],oi_[6],or_[7],oi_[7]);
        for (int l = 0; l < 8; l++) {
            __m256d zlo, zhi;
            REINT(or_[l], oi_[l], zlo, zhi);
            _mm256_storeu_pd(z + 2 * ((size_t)l * Dv + k), zlo);
            _mm256_storeu_pd(z + 2 * ((size_t)l * Dv + k) + 4, zhi);
        }
    }
}

/* ── chain plumbing ── */
static int N, NF;
static int R[8];
static long DD[8], GG[8];
static double *twsp[8], *twp1;
static long *gb[8];
static double *SP;

static long drev_full(long x, const int *r, int nf)
{ long v = 0; for (int i = nf - 1; i >= 0; i--) { v = v * r[i] + (x % r[i]); x /= r[i]; } return v; }
static long brev_prefix(long g, int s, const int *r)
{
    long f[8];
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
static void build_tables(void)
{
    const double TAU = 2.0 * M_PI;
    DD[NF - 1] = 1;
    for (int i = NF - 2; i >= 0; i--) DD[i] = DD[i + 1] * R[i + 1];
    GG[0] = 1;
    for (int i = 1; i < NF; i++) GG[i] = GG[i - 1] * R[i - 1];
    for (int s = 1; s <= NF - 2; s++) {
        int Rm1 = R[s] - 1;
        long M = (long)N / DD[s];
        twsp[s] = (double *)_mm_malloc((size_t)GG[s] * Rm1 * 8 * 8, 64);
        gb[s] = (long *)malloc((size_t)GG[s] * sizeof(long));
        for (long g = 0; g < GG[s]; g++) {
            gb[s][g] = base_of(g, s, R, DD);
            long brev = brev_prefix(g, s, R);
            for (int l = 1; l < R[s]; l++) {
                double a = -TAU * (double)(((long)l * brev) % M) / (double)M;
                double *sp2 = twsp[s] + ((size_t)g * Rm1 + (l - 1)) * 8;
                for (int j = 0; j < 4; j++) { sp2[j] = cos(a); sp2[4 + j] = sin(a); }
            }
        }
    }
    {
        long pairs = ((long)N / 8) / 2;
        twp1 = (double *)_mm_malloc((size_t)pairs * 64, 64);
        for (long p = 0; p < pairs; p++) {
            long b0 = brev_prefix(2 * p, NF - 1, R);
            long b1 = brev_prefix(2 * p + 1, NF - 1, R);
            double a0 = -TAU * (double)(b0 % N) / (double)N;
            double a1 = -TAU * (double)(b1 % N) / (double)N;
            double *rec = twp1 + (size_t)p * 8;
            rec[0] = cos(a0); rec[1] = cos(a0); rec[2] = cos(a1); rec[3] = cos(a1);
            rec[4] = -sin(a0); rec[5] = sin(a0); rec[6] = -sin(a1); rec[7] = sin(a1);
        }
    }
}

/* ── driver arm (sweep shape: runtime base_of, extern terminator) ── */
static void run_drv(const double *z0, double *A, double *out, int termt)
{
    radix4_z_s0s_fwd_avx2(z0, 0, SP, 0, 0, 0, (unsigned long long)DD[0], 0,
                          (unsigned long long)DD[0], 0, (unsigned long long)DD[0]);
    for (int s = 1; s <= NF - 3; s++) {
        int Rm1 = R[s] - 1;
        zfn f = (R[s] == 8) ? radix8_z_ms_fwd_avx2 : radix4_z_ms_fwd_avx2;
        for (long g = 0; g < GG[s]; g++) {
            long b = base_of(g, s, R, DD);
            f(SP + 2 * b, 0, SP + 2 * b, 0, twsp[s] + (size_t)g * Rm1 * 8, 0,
              (unsigned long long)DD[s], 0, (unsigned long long)DD[s], 0,
              (unsigned long long)DD[s]);
        }
    }
    {
        int s = NF - 2, Rm1 = R[s] - 1;
        for (long g = 0; g < GG[s]; g++) {
            long b = base_of(g, s, R, DD);
            radix8_z_msz_fwd_avx2(SP + 2 * b, 0, A + 2 * b, 0,
                                  twsp[s] + (size_t)g * Rm1 * 8, 0,
                                  (unsigned long long)DD[s], 0,
                                  (unsigned long long)DD[s], 0,
                                  (unsigned long long)DD[s]);
        }
    }
    zfn tf = termt ? radix8_z_t2spt_fwd_avx2 : radix8_z_t2sp_fwd_avx2;
    tf(A, 0, out, 0, twp1, 0, 1, 8, (unsigned long long)(N / 8), 1,
       (unsigned long long)(N / 8));
}

/* ── drvT arm: TABLED bases + compact called kernels (bake the tables, not
 * the code — the production shape if full fusion loses to I-cache bloat) ── */
static void run_drvT(const double *z0, double *A, double *out, int termt)
{
    radix4_z_s0s_fwd_avx2(z0, 0, SP, 0, 0, 0, (unsigned long long)DD[0], 0,
                          (unsigned long long)DD[0], 0, (unsigned long long)DD[0]);
    for (int s = 1; s <= NF - 3; s++) {
        int Rm1 = R[s] - 1;
        zfn f = (R[s] == 8) ? radix8_z_ms_fwd_avx2 : radix4_z_ms_fwd_avx2;
        const long *gbs = gb[s];
        for (long g = 0; g < GG[s]; g++) {
            long b = gbs[g];
            f(SP + 2 * b, 0, SP + 2 * b, 0, twsp[s] + (size_t)g * Rm1 * 8, 0,
              (unsigned long long)DD[s], 0, (unsigned long long)DD[s], 0,
              (unsigned long long)DD[s]);
        }
    }
    {
        int s = NF - 2, Rm1 = R[s] - 1;
        const long *gbs = gb[s];
        for (long g = 0; g < GG[s]; g++) {
            long b = gbs[g];
            radix8_z_msz_fwd_avx2(SP + 2 * b, 0, A + 2 * b, 0,
                                  twsp[s] + (size_t)g * Rm1 * 8, 0,
                                  (unsigned long long)DD[s], 0,
                                  (unsigned long long)DD[s], 0,
                                  (unsigned long long)DD[s]);
        }
    }
    zfn tf = termt ? radix8_z_t2spt_fwd_avx2 : radix8_z_t2sp_fwd_avx2;
    tf(A, 0, out, 0, twp1, 0, 1, 8, (unsigned long long)(N / 8), 1,
       (unsigned long long)(N / 8));
}

/* ── BAKED fused executors: constant trips, tabled bases, inlined term ── */
__attribute__((target("avx2,fma")))
static void baked_2048(const double *z0, double *A, double *out)
{   /* 4.8.8.8 t2spt: D=[512,64,8,1], G=[1,4,32,256] */
    s0z2s_r4(z0, SP, 512, 512);
    for (long g = 0; g < 4; g++)  mid_s2s_r8(SP + 2 * gb[1][g], twsp[1] + g * 56, 64, 64);
    for (long g = 0; g < 32; g++) mid_s2z_r8(SP + 2 * gb[2][g], A + 2 * gb[2][g],
                                             twsp[2] + g * 56, 8, 8);
    bk_t2spt(A, 0, out, 0, twp1, 0, 1, 8, 256, 1, 256);
}
__attribute__((target("avx2,fma")))
static void baked_4096(const double *z0, double *A, double *out)
{   /* 4.4.4.8.8 t2spt: D=[1024,256,64,8,1], G=[1,4,16,64,512] */
    s0z2s_r4(z0, SP, 1024, 1024);
    for (long g = 0; g < 4; g++)  mid_s2s_r4(SP + 2 * gb[1][g], twsp[1] + g * 24, 256, 256);
    for (long g = 0; g < 16; g++) mid_s2s_r4(SP + 2 * gb[2][g], twsp[2] + g * 24, 64, 64);
    for (long g = 0; g < 64; g++) mid_s2z_r8(SP + 2 * gb[3][g], A + 2 * gb[3][g],
                                             twsp[3] + g * 56, 8, 8);
    bk_t2spt(A, 0, out, 0, twp1, 0, 1, 8, 512, 1, 512);
}
__attribute__((target("avx2,fma")))
static void baked_8192(const double *z0, double *A, double *out)
{   /* 4.4.8.8.8 t2sp: D=[2048,512,64,8,1], G=[1,4,16,128,1024] */
    s0z2s_r4(z0, SP, 2048, 2048);
    for (long g = 0; g < 4; g++)   mid_s2s_r4(SP + 2 * gb[1][g], twsp[1] + g * 24, 512, 512);
    for (long g = 0; g < 16; g++)  mid_s2s_r8(SP + 2 * gb[2][g], twsp[2] + g * 56, 64, 64);
    for (long g = 0; g < 128; g++) mid_s2z_r8(SP + 2 * gb[3][g], A + 2 * gb[3][g],
                                              twsp[3] + g * 56, 8, 8);
    bk_t2sp(A, 0, out, 0, twp1, 0, 1, 8, 1024, 1, 1024);
}
__attribute__((target("avx2,fma")))
static void baked_16384(const double *z0, double *A, double *out)
{   /* 4.8.8.8.8 t2spt: D=[4096,512,64,8,1], G=[1,4,32,256,2048] */
    s0z2s_r4(z0, SP, 4096, 4096);
    for (long g = 0; g < 4; g++)   mid_s2s_r8(SP + 2 * gb[1][g], twsp[1] + g * 56, 512, 512);
    for (long g = 0; g < 32; g++)  mid_s2s_r8(SP + 2 * gb[2][g], twsp[2] + g * 56, 64, 64);
    for (long g = 0; g < 256; g++) mid_s2z_r8(SP + 2 * gb[3][g], A + 2 * gb[3][g],
                                              twsp[3] + g * 56, 8, 8);
    bk_t2spt(A, 0, out, 0, twp1, 0, 1, 8, 2048, 1, 2048);
}

static double now_ms(void)
{
    LARGE_INTEGER f, cc;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&cc);
    return 1000.0 * (double)cc.QuadPart / (double)f.QuadPart;
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

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    N = argc > 1 ? atoi(argv[1]) : 4096;
    int termt = 1;
    void (*baked)(const double *, double *, double *) = 0;
    if (N == 2048)      { NF = 4; R[0]=4; R[1]=8; R[2]=8; R[3]=8; baked = baked_2048; }
    else if (N == 4096) { NF = 5; R[0]=4; R[1]=4; R[2]=4; R[3]=8; R[4]=8; baked = baked_4096; }
    else if (N == 8192) { NF = 5; R[0]=4; R[1]=4; R[2]=8; R[3]=8; R[4]=8; baked = baked_8192; termt = 0; }
    else if (N == 16384){ NF = 5; R[0]=4; R[1]=8; R[2]=8; R[3]=8; R[4]=8; baked = baked_16384; }
    else { printf("N must be 2048|4096|8192|16384\n"); return 1; }
    build_tables();

    double *z0 = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *A  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *z  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    SP = (double *)_mm_malloc((size_t)2 * N * 8, 64);
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

    static const char *AN[3] = { "drv  ", "drvT ", "baked" };
    for (int a = 0; a < 3; a++) {
        if (a == 0) run_drv(z0, A, z, termt);
        else if (a == 1) run_drvT(z0, A, z, termt);
        else baked(z0, A, z);
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
        if (t) { cachebust(); Sleep(150); }   /* PACED finals (dp_planner lesson) */
        for (int q = 0; q < 4; q++) {
            int a = (t & 1) ? 3 - q : q;
            double t0, ns;
            if (a == 0) {
                for (int w = 0; w < 6; w++) run_drv(z0, A, z, termt);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) run_drv(z0, A, z, termt);
            } else if (a == 1) {
                for (int w = 0; w < 6; w++) run_drvT(z0, A, z, termt);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) run_drvT(z0, A, z, termt);
            } else if (a == 2) {
                for (int w = 0; w < 6; w++) baked(z0, A, z);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) baked(z0, A, z);
            } else {
                for (int w = 0; w < 6; w++) DftiComputeForward(h, z);
                t0 = now_ms();
                for (int i = 0; i < reps; i++) DftiComputeForward(h, z);
            }
            ns = (now_ms() - t0) * 1e6 / reps;
            if (ns < best[a]) best[a] = ns;
        }
    }
    const char *cn = N == 2048 ? "4.8.8.8" : N == 4096 ? "4.4.4.8.8"
                   : N == 8192 ? "4.4.8.8.8" : "4.8.8.8.8";
    printf("\n# N=%d PACED FINALS: split winner %s (%s term)\n", N, cn,
           termt ? "t2spt" : "t2sp");
    printf("drv   (runtime bases) %9.1f ns   vsMKL %.2f\n", best[0], best[3] / best[0]);
    printf("drvT  (tabled bases)  %9.1f ns   vsMKL %.2f   vs drv %.2f\n",
           best[1], best[3] / best[1], best[0] / best[1]);
    printf("baked (fused code)    %9.1f ns   vsMKL %.2f   vs drv %.2f\n",
           best[2], best[3] / best[2], best[0] / best[2]);
    printf("MKL-IL                %9.1f ns\n", best[3]);
    printf("DONE\n");
    return 0;
}
