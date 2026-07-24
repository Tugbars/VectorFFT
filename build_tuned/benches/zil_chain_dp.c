/* zil_chain_dp.c — the chain PLANNER for the z staged cascade: exhaustively
 * enumerate ALL factor chains of N over the available z kernels, gate each vs
 * naive, race all + MKL-IL, rank. "We can't just arbitrarily choose the stages"
 * — at these Ns the chain space is tiny (18 @4096, 34 @16384), so measured
 * exhaustive search IS the planner (cost models hit the OOO/cache ceiling;
 * plan-level measurement doesn't). Winner per cell -> wisdom (like cc_chain).
 *
 * Executor = generalized arm-B (grid-preserving DIT, gated in zil_cascade.c):
 *   S0   n1 leaf   z0 -> A          (Ls=OLs=D0, count=D0)
 *   mids t2 TRUE-IN-PLACE on A      (Ls=OLs=Ds, count=Ds, group base digit-decomposed,
 *                                    col-const VTW2 W_{N/Ds}^{l*brev(g)})
 *   term t2s gather A -> out        (Ls=1, Gs=Rt, OLs=N/Rt, per-col VTW2 W_N^{l*brev(k)})
 * Output digit-reversed: out[l*(N/Rt)+g] = X[drev(g*Rt+l)] (mixed-radix drev).
 * nf=2 chains ARE the flat two-pass (64.64 = the old champion) — subsumed.
 *
 * Kernel coverage: S0 n1 {4,8,16b,32b,64b2}; mid t2 {8,16,32,64}; term t2s
 * {8,16,32,64}. radix-4 t2/t2s = pending emitter run (MKL's census favors
 * radix-4 deep stages — follow-up once emitted).
 *
 * Build: python build.py --src benches/zil_chain_dp.c --mkl
 * Run:   zil_chain_dp.exe [N]      (N in {2048,4096,8192,16384}, default 4096)
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

typedef void (*zfn)(const double *, const double *, double *, double *,
                    const double *, const double *,
                    unsigned long long, unsigned long long,
                    unsigned long long, unsigned long long, unsigned long long);
#define D(fn) extern void fn(const double *, const double *, double *, double *, \
    const double *, const double *, unsigned long long, unsigned long long, \
    unsigned long long, unsigned long long, unsigned long long);
D(radix4_z_n1_fwd_avx2)  D(radix8_z_n1_fwd_avx2)  D(radix16_z_n1b_fwd_avx2)
D(radix32_z_n1b_fwd_avx2) D(radix64_z_n1b2_fwd_avx2)
D(radix8_z_t2_fwd_avx2)  D(radix16_z_t2_fwd_avx2) D(radix32_z_t2_fwd_avx2)
D(radix64_z_t2_fwd_avx2)
D(radix8_z_t2s_fwd_avx2) D(radix16_z_t2s_fwd_avx2) D(radix32_z_t2s_fwd_avx2)
D(radix64_z_t2s_fwd_avx2)
D(radix8_z_t2c_fwd_avx2) D(radix16_z_t2c_fwd_avx2) D(radix32_z_t2c_fwd_avx2)
D(radix64_z_t2c_fwd_avx2)
D(radix8_z_t2sp_fwd_avx2) D(radix16_z_t2sp_fwd_avx2)
D(radix4_z_t2_fwd_avx2) D(radix4_z_t2c_fwd_avx2) D(radix4_z_t2s_fwd_avx2)
D(radix4_z_t2sp_fwd_avx2)
D(radix4_z_t2st_fwd_avx2) D(radix8_z_t2st_fwd_avx2) D(radix16_z_t2st_fwd_avx2)
D(radix4_z_t2spt_fwd_avx2) D(radix8_z_t2spt_fwd_avx2) D(radix16_z_t2spt_fwd_avx2)

static zfn n1_of(int R)  { switch (R) { case 4: return radix4_z_n1_fwd_avx2;
    case 8: return radix8_z_n1_fwd_avx2;   case 16: return radix16_z_n1b_fwd_avx2;
    case 32: return radix32_z_n1b_fwd_avx2; case 64: return radix64_z_n1b2_fwd_avx2; }
    return 0; }
static zfn t2_of(int R)  { switch (R) { case 4: return radix4_z_t2_fwd_avx2;
    case 8: return radix8_z_t2_fwd_avx2;
    case 16: return radix16_z_t2_fwd_avx2; case 32: return radix32_z_t2_fwd_avx2;
    case 64: return radix64_z_t2_fwd_avx2; } return 0; }
static zfn t2s_of(int R) { switch (R) { case 4: return radix4_z_t2s_fwd_avx2;
    case 8: return radix8_z_t2s_fwd_avx2;
    case 16: return radix16_z_t2s_fwd_avx2; case 32: return radix32_z_t2s_fwd_avx2;
    case 64: return radix64_z_t2s_fwd_avx2; } return 0; }
static zfn t2c_of(int R) { switch (R) { case 4: return radix4_z_t2c_fwd_avx2;
    case 8: return radix8_z_t2c_fwd_avx2;
    case 16: return radix16_z_t2c_fwd_avx2; case 32: return radix32_z_t2c_fwd_avx2;
    case 64: return radix64_z_t2c_fwd_avx2; } return 0; }
static zfn t2sp_of(int R) { switch (R) { case 4: return radix4_z_t2sp_fwd_avx2;
    case 8: return radix8_z_t2sp_fwd_avx2;
    case 16: return radix16_z_t2sp_fwd_avx2; } return 0; }
static zfn t2st_of(int R) { switch (R) { case 4: return radix4_z_t2st_fwd_avx2;
    case 8: return radix8_z_t2st_fwd_avx2;
    case 16: return radix16_z_t2st_fwd_avx2; } return 0; }
static zfn t2spt_of(int R) { switch (R) { case 4: return radix4_z_t2spt_fwd_avx2;
    case 8: return radix8_z_t2spt_fwd_avx2;
    case 16: return radix16_z_t2spt_fwd_avx2; } return 0; }
/* terminator fn for variant v: bit1 = powers tw, bit2 = tiled loads */
static zfn term_of(int R, int v)
{
    int p = (v >> 1) & 1, t = (v >> 2) & 1;
    if (t) return p ? t2spt_of(R) : t2st_of(R);
    return p ? t2sp_of(R) : t2s_of(R);
}

/* ── LEVER-5 BLOCK-SPLIT INTERIOR kernels (from zil_split_interior.c, §4.99):
 * scratch = 64B [re x4][im x4] blocks (z addressing, +4 for im). S0 converts
 * z->split in its stores; mids are shuffle-free; last mid re-interleaves. ── */
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
static void s0z2s_r8(const double *z, double *sp, long Ls, long count)
{
    for (long k = 0; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8], or_[8], oi_[8];
        for (int l = 0; l < 8; l++) {
            __m256d zlo = _mm256_loadu_pd(z + 2 * ((size_t)l * Ls + k));
            __m256d zhi = _mm256_loadu_pd(z + 2 * ((size_t)l * Ls + k) + 4);
            DEINT(zlo, zhi, xr[l], xi[l]);
        }
        SPLIT_BFLY8(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                    xr[4],xi[4],xr[5],xi[5],xr[6],xi[6],xr[7],xi[7],
                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3],
                    or_[4],oi_[4],or_[5],oi_[5],or_[6],oi_[6],or_[7],oi_[7]);
        for (int l = 0; l < 8; l++) {
            _mm256_storeu_pd(sp + 2 * ((size_t)l * Ls + k), or_[l]);
            _mm256_storeu_pd(sp + 2 * ((size_t)l * Ls + k) + 4, oi_[l]);
        }
    }
}
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
__attribute__((target("avx2,fma")))
static void mid_s2z_r4(const double *sp, double *z, const double *tw, long Dv, long count)
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
            __m256d zlo, zhi;
            REINT(or_[l], oi_[l], zlo, zhi);
            _mm256_storeu_pd(z + 2 * ((size_t)l * Dv + k), zlo);
            _mm256_storeu_pd(z + 2 * ((size_t)l * Dv + k) + 4, zhi);
        }
    }
}

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

/* ── chain machinery ──────────────────────────────────────────────────── */
#define MAXNF 8
#define MAXC  256
static int MINPART = 3;   /* argv2: 2 enables radix-4 mids/terminators */
static int LEAN = 0;      /* argv3: 1 = only t2c/t2sp variant (bounds tw memory
                             on the big radix-4 field); FLAT tables not built */
/* executor variants: v0..7 = z interior: bit0 = t2c mids (group-constant tw),
 * bit1 = t2sp term (w^1 powers), bit2 = tiled terminator loads.
 * v8/v9 = LEVER-5 BLOCK-SPLIT interior (S0 z->split, shuffle-free split mids,
 * last mid re-interleaves; v8 term = t2sp, v9 term = t2spt). Split arms
 * require nf>=3, R0 in {4,8}, all mids in {4,8}. */
#define NV 10
typedef struct {
    int nf, R[MAXNF];
    long D[MAXNF], G[MAXNF];       /* D=stride (prod after), G=groups (prod before) */
    double *twm[MAXNF];            /* middle-stage VTW2, replicated (t2 arm) */
    double *twc[MAXNF];            /* middle-stage VTW2, one set/group (t2c arm) */
    double *twsp[MAXNF];           /* middle-stage SPLAT pairs [c x4][s x4] (split arms) */
    double *twt;                   /* terminator VTW2, FLAT per-column (t2s arm) */
    double *twp1;                  /* terminator w^1-only stream (t2sp arm) */
    size_t twb[NV];                /* streamed table bytes per variant */
    char name[40];
    double best[NV];               /* <0 = gate fail / unavailable */
} chain_t;
static chain_t C[MAXC];
static int NC = 0;
static int N;

/* mixed-radix digit reversal of grid index x over the full chain */
static long drev_full(long x, const int *R, int nf)
{
    long r = 0;
    for (int i = nf - 1; i >= 0; i--) { r = r * R[i] + (x % R[i]); x /= R[i]; }
    return r;
}
/* reversed value of the s prefix digits of group/block index g:
 * brev = sum f_i * (prod_{k<i} R_k), f_0 = MSB digit of g */
static long brev_prefix(long g, int s, const int *R)
{
    long f[MAXNF];
    for (int i = s - 1; i >= 0; i--) { f[i] = g % R[i]; g /= R[i]; }
    long P = 1, r = 0;
    for (int i = 0; i < s; i++) { r += f[i] * P; P *= R[i]; }
    return r;
}
/* complex base offset of stage-s group g: sum f_i * D_i */
static long base_of(long g, int s, const int *R, const long *D)
{
    long b = 0;
    for (int i = s - 1; i >= 0; i--) { long f = g % R[i]; g /= R[i]; b += f * D[i]; }
    return b;
}

/* one VTW2 record (8 doubles): cos-first, sign-folded, cols k0/k1 */
static void vtw2_rec(double *rec, double a0, double a1)
{
    rec[0] = cos(a0); rec[1] = cos(a0); rec[2] = cos(a1); rec[3] = cos(a1);
    rec[4] = -sin(a0); rec[5] = sin(a0); rec[6] = -sin(a1); rec[7] = sin(a1);
}

static void build_chain(chain_t *c)
{
    const int nf = c->nf;
    c->D[nf - 1] = 1;
    for (int i = nf - 2; i >= 0; i--) c->D[i] = c->D[i + 1] * c->R[i + 1];
    c->G[0] = 1;
    for (int i = 1; i < nf; i++) c->G[i] = c->G[i - 1] * c->R[i - 1];
    const double TAU = 2.0 * M_PI;
    size_t mid_full = 0, mid_cst = 0;

    /* middles: group-constant twiddles. t2 arm streams a replicated record per
     * column-pair; t2c arm reads ONE record set per group (L1-hot). */
    for (int s = 1; s <= nf - 2; s++) {
        long pairs = c->D[s] / 2, M = (long)N / c->D[s];
        size_t rl = (size_t)(c->R[s] - 1) * 8;                 /* rec doubles/pair */
        size_t sz = (size_t)c->G[s] * pairs * rl * 8;
        c->twm[s] = LEAN ? 0 : (double *)_mm_malloc(sz, 64);
        c->twc[s] = (double *)_mm_malloc((size_t)c->G[s] * rl * 8, 64);
        c->twsp[s] = (double *)_mm_malloc((size_t)c->G[s] * rl * 8, 64);
        mid_full += sz;
        mid_cst += (size_t)c->G[s] * rl * 8;
        for (long g = 0; g < c->G[s]; g++) {
            long brev = brev_prefix(g, s, c->R);
            double *gc = c->twc[s] + (size_t)g * rl;
            double *gs = c->twsp[s] + (size_t)g * rl;
            for (int l = 1; l < c->R[s]; l++) {
                double a = -TAU * (double)(((long)l * brev) % M) / (double)M;
                vtw2_rec(gc + (size_t)(l - 1) * 8, a, a);
                for (int j = 0; j < 4; j++) {
                    gs[(size_t)(l - 1) * 8 + j] = cos(a);
                    gs[(size_t)(l - 1) * 8 + 4 + j] = sin(a);
                }
            }
            if (!LEAN) {
                double *gp = c->twm[s] + (size_t)g * pairs * rl;
                memcpy(gp, gc, rl * 8);
                for (long p = 1; p < pairs; p++)               /* replicate */
                    memcpy(gp + (size_t)p * rl, gc, rl * 8);
            }
        }
    }
    /* terminator: FLAT per-column records W_N^{l*brev(k)} (t2s arm) + w^1-only
     * stream (t2sp arm: 64B/pair, legs 2.. built in-register) */
    {
        int Rt = c->R[nf - 1];
        long cols = (long)N / Rt, pairs = cols / 2;
        size_t rl = (size_t)(Rt - 1) * 8;
        c->twt = LEAN ? 0 : (double *)_mm_malloc((size_t)pairs * rl * 8, 64);
        c->twp1 = (double *)_mm_malloc((size_t)pairs * 8 * 8, 64);
        for (long p = 0; p < pairs; p++) {
            long b0 = brev_prefix(2 * p, nf - 1, c->R);
            long b1 = brev_prefix(2 * p + 1, nf - 1, c->R);
            for (int l = 1; l < Rt; l++) {
                double a0 = -TAU * (double)(((long)l * b0) % N) / (double)N;
                double a1 = -TAU * (double)(((long)l * b1) % N) / (double)N;
                if (!LEAN)
                    vtw2_rec(c->twt + (size_t)p * rl + (size_t)(l - 1) * 8, a0, a1);
                if (l == 1) vtw2_rec(c->twp1 + (size_t)p * 8, a0, a1);
            }
        }
        size_t term_full = (size_t)pairs * rl * 8, term_p1 = (size_t)pairs * 64;
        c->twb[0] = mid_full + term_full;   /* t2  mids + t2s  term */
        c->twb[1] = mid_cst + term_full;    /* t2c mids + t2s  term */
        c->twb[2] = mid_full + term_p1;     /* t2  mids + t2sp term */
        c->twb[3] = mid_cst + term_p1;      /* t2c mids + t2sp term */
        for (int v = 4; v < 8; v++) c->twb[v] = c->twb[v - 4];   /* tiled: same tables */
        c->twb[8] = mid_cst + term_p1;      /* split arms: splat sets + w^1 */
        c->twb[9] = mid_cst + term_p1;
    }
    char *w = c->name;
    for (int i = 0; i < nf; i++) w += sprintf(w, "%s%d", i ? "." : "", c->R[i]);
}

/* split-arm eligibility: hand kernels cover R0 in {4,8}, mids in {4,8}, nf>=3 */
static int split_ok(const chain_t *c)
{
    if (c->nf < 3) return 0;
    if (c->R[0] != 4 && c->R[0] != 8) return 0;
    for (int s = 1; s <= c->nf - 2; s++)
        if (c->R[s] != 4 && c->R[s] != 8) return 0;
    return 1;
}
static double *SP;   /* block-split scratch (allocated in main) */

/* v: 0..7 z interior (bit0 t2c, bit1 powers, bit2 tiled); 8/9 split interior */
static void run_chain(const chain_t *c, int v, const double *z0, double *A,
                      double *out)
{
    const int nf = c->nf;
    if (v >= 8) {
        if (c->R[0] == 8) s0z2s_r8(z0, SP, c->D[0], c->D[0]);
        else              s0z2s_r4(z0, SP, c->D[0], c->D[0]);
        for (int s = 1; s <= nf - 3; s++) {
            size_t rl = (size_t)(c->R[s] - 1) * 8;
            for (long g = 0; g < c->G[s]; g++) {
                long b = base_of(g, s, c->R, c->D);
                const double *tw = c->twsp[s] + (size_t)g * rl;
                if (c->R[s] == 8) mid_s2s_r8(SP + 2 * b, tw, c->D[s], c->D[s]);
                else              mid_s2s_r4(SP + 2 * b, tw, c->D[s], c->D[s]);
            }
        }
        {
            int s = nf - 2;
            size_t rl = (size_t)(c->R[s] - 1) * 8;
            for (long g = 0; g < c->G[s]; g++) {
                long b = base_of(g, s, c->R, c->D);
                const double *tw = c->twsp[s] + (size_t)g * rl;
                if (c->R[s] == 8) mid_s2z_r8(SP + 2 * b, A + 2 * b, tw, c->D[s], c->D[s]);
                else              mid_s2z_r4(SP + 2 * b, A + 2 * b, tw, c->D[s], c->D[s]);
            }
        }
        {
            int Rt = c->R[nf - 1];
            zfn f = (v == 9) ? t2spt_of(Rt) : t2sp_of(Rt);
            f(A, 0, out, 0, c->twp1, 0, 1, (unsigned long long)Rt,
              (unsigned long long)(N / Rt), 1, (unsigned long long)(N / Rt));
        }
        return;
    }
    const int midc = v & 1, termp = (v >> 1) & 1;
    n1_of(c->R[0])(z0, 0, A, 0, 0, 0,
                   (unsigned long long)c->D[0], 0, (unsigned long long)c->D[0], 0,
                   (unsigned long long)c->D[0]);
    for (int s = 1; s <= nf - 2; s++) {
        zfn f = midc ? t2c_of(c->R[s]) : t2_of(c->R[s]);
        long pairs = c->D[s] / 2;
        size_t rl = (size_t)(c->R[s] - 1) * 8;
        for (long g = 0; g < c->G[s]; g++) {
            long b = base_of(g, s, c->R, c->D);
            const double *tw = midc ? c->twc[s] + (size_t)g * rl
                                    : c->twm[s] + (size_t)g * pairs * rl;
            f(A + 2 * b, 0, A + 2 * b, 0, tw, 0,
              (unsigned long long)c->D[s], 0, (unsigned long long)c->D[s], 0,
              (unsigned long long)c->D[s]);
        }
    }
    {
        int Rt = c->R[nf - 1];
        zfn f = term_of(Rt, v);
        f(A, 0, out, 0, termp ? c->twp1 : c->twt, 0,
          1, (unsigned long long)Rt, (unsigned long long)(N / Rt), 1,
          (unsigned long long)(N / Rt));
    }
}

/* enumerate chains: first log2-part in {2..6} (n1 4..64), rest in {3..6} */
static void enumerate(int rem, int pos, int *parts)
{
    if (rem == 0) {
        if (pos < 2 || NC >= MAXC) return;
        chain_t *c = &C[NC];
        memset(c, 0, sizeof(*c));
        c->nf = pos;
        for (int i = 0; i < pos; i++) c->R[i] = 1 << parts[i];
        NC++;
        return;
    }
    int lo = pos ? MINPART : 2, hi = 6;
    for (int p = lo; p <= hi && p <= rem; p++) {
        parts[pos] = p;
        enumerate(rem - p, pos + 1, parts);
    }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    N = argc > 1 ? atoi(argv[1]) : 4096;
    MINPART = argc > 2 ? atoi(argv[2]) : 3;
    LEAN = argc > 3 ? atoi(argv[3]) : 0;
    int L = 0; while ((1 << L) < N) L++;
    if ((1 << L) != N) { printf("N must be 2^k\n"); return 1; }

    int parts[MAXNF];
    enumerate(L, 0, parts);
    printf("# N=%d: %d candidate chains (minpart=%d%s)\n", N, NC, MINPART,
           LEAN ? ", LEAN: t2c/t2sp only" : "");

    double *z0 = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *A  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *z  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *Rr = (double *)_mm_malloc((size_t)N * 8, 64);
    double *Ri = (double *)_mm_malloc((size_t)N * 8, 64);
    SP = (double *)_mm_malloc((size_t)2 * N * 8, 64);   /* block-split scratch */
    srand(N);
    for (int i = 0; i < 2 * N; i++) z0[i] = (double)rand() / RAND_MAX - 0.5;

    /* naive reference */
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

    /* build + gate every chain x variant (digit-reversed compare) */
    int ok = 0, tot = 0;
    for (int ci = 0; ci < NC; ci++) {
        chain_t *c = &C[ci];
        build_chain(c);
        int Rt = c->R[c->nf - 1];
        long NR = (long)N / Rt;
        for (int v = 0; v < NV; v++) {
            if (LEAN && v != 3 && v != 7 && v != 8 && v != 9) { c->best[v] = -1; continue; }
            if (v >= 8) {
                if (!split_ok(c) || !t2sp_of(Rt)) { c->best[v] = -1; continue; }
                if (v == 9 && !t2spt_of(Rt)) { c->best[v] = -1; continue; }
            } else {
                if (((v >> 1) & 1) && !t2sp_of(Rt)) { c->best[v] = -1; continue; }
                if (((v >> 2) & 1) && !t2st_of(Rt)) { c->best[v] = -1; continue; }
            }
            tot++;
            run_chain(c, v, z0, A, z);
            double err = 0;
            for (long idx = 0; idx < N; idx++) {
                long l = idx / NR, g = idx % NR;
                long m = drev_full(g * Rt + l, c->R, c->nf);
                double d = fabs(z[2 * idx] - Rr[m]) + fabs(z[2 * idx + 1] - Ri[m]);
                if (d > err) err = d;
            }
            double rel = err / mag;
            c->best[v] = 1e18;
            if (rel < 1e-11) ok++;
            else {
                printf("GATE FAIL %-14s v%d relerr=%.3e\n", c->name, v, rel);
                c->best[v] = -1;
            }
        }
    }
    printf("# gates: %d/%d PASS (chain x variant)\n", ok, tot);

    /* race: all gated chains + MKL, best-of-7, cachebust, rotated order */
    DFTI_DESCRIPTOR_HANDLE h = NULL;
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiCommitDescriptor(h);
    int reps = (int)(4.0e6 / N); if (reps < 200) reps = 200;
    double mkl_best = 1e18;
    int narm = NC * NV + 1;
    for (int t = 0; t < 7; t++) {
        if (t) cachebust();
        for (int q = 0; q < narm; q++) {
            int a = (t & 1) ? (narm - 1 - q) : q;
            if (a < NC * NV) {
                chain_t *c = &C[a / NV];
                int v = a % NV;
                if (c->best[v] < 0) continue;
                for (int w = 0; w < 6; w++) run_chain(c, v, z0, A, z);
                double t0 = now_ms();
                for (int i = 0; i < reps; i++) run_chain(c, v, z0, A, z);
                double ns = (now_ms() - t0) * 1e6 / reps;
                if (ns < c->best[v]) c->best[v] = ns;
            } else {
                for (int w = 0; w < 6; w++) DftiComputeForward(h, z);
                double t0 = now_ms();
                for (int i = 0; i < reps; i++) DftiComputeForward(h, z);
                double ns = (now_ms() - t0) * 1e6 / reps;
                if (ns < mkl_best) mkl_best = ns;
            }
        }
    }

    /* ranked report over (chain, variant) tuples */
    static const char *VN[NV] = { "t2 /t2s  ", "t2c/t2s  ", "t2 /t2sp ", "t2c/t2sp ",
                                  "t2 /t2st ", "t2c/t2st ", "t2 /t2spt", "t2c/t2spt",
                                  "SPL/t2sp ", "SPL/t2spt" };
    typedef struct { int ci, v; double ns; } row_t;
    row_t rows[MAXC * NV];
    int nr = 0;
    for (int ci = 0; ci < NC; ci++)
        for (int v = 0; v < NV; v++)
            if (C[ci].best[v] >= 0 && C[ci].best[v] < 1e17) {
                rows[nr].ci = ci; rows[nr].v = v; rows[nr].ns = C[ci].best[v]; nr++;
            }
    for (int i = 0; i < nr; i++)
        for (int j = i + 1; j < nr; j++)
            if (rows[j].ns < rows[i].ns) { row_t tt = rows[i]; rows[i] = rows[j]; rows[j] = tt; }
    printf("\n# N=%d chain x variant race vs MKL-IL %.1f ns  (mids/term: t2c=group-const tw, t2sp=w^1 powers)\n",
           N, mkl_best);
    printf("%-4s %-14s %-9s %-7s %9s %7s %s\n", "rank", "chain", "variant", "twKB", "ns", "vsMKL", "note");
    int shown = 0;
    for (int i = 0; i < nr && shown < 16; i++, shown++) {
        chain_t *c = &C[rows[i].ci];
        printf("%-4d %-14s %-9s %-7.0f %9.1f %7.2f %s\n", i + 1, c->name,
               VN[rows[i].v], (double)c->twb[rows[i].v] / 1024.0, rows[i].ns,
               mkl_best / rows[i].ns, c->nf == 2 ? "flat two-pass" : "");
    }
    /* per-lever attribution on the fastest chain */
    if (nr) {
        chain_t *c = &C[rows[0].ci];
        printf("\n# attribution on %s:", c->name);
        for (int v = 0; v < NV; v++)
            if (c->best[v] >= 0 && c->best[v] < 1e17)
                printf("  %s=%.1f", VN[v], c->best[v]);
        printf("\n");
    }
    printf("DONE\n");
    return 0;
}
