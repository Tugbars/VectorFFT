/* zil_split_interior.c — LEVER 5 spike: BLOCK-SPLIT SCRATCH INTERIOR for the z
 * cascade (docs/roadmap/z_cascade_plan.md §4.98; discovery in
 * docs/research/mkl_highN_cascade_anatomy.md §4.5).
 *
 * MKL's high-N cascade is z only at the API boundary — its interior runs on
 * split re/im planes, making every mid pass SHUFFLE-FREE (elementwise split
 * cmul, rotations = free operand renames). This spike reproduces that shape
 * with the z boundary kept:
 *   S0    z loads -> deinterleave (shuffles paid ONCE) -> split butterfly
 *         -> split-plane stores                                  [z -> split]
 *   mids  in-place split butterflies, splat-pair group-const twiddles,
 *         ZERO shuffles                                          [split]
 *   last  same mid but re-interleaving stores                    [split -> z]
 *   term  EXISTING gated z terminator (t2sp/t2spt), unchanged contract
 * Chains raced @4096: A = 8.8.8.8 split-interior vs the SAME chain z-interior
 * (t2c/t2sp — the direct lever measurement), B = 4.4.4.4.4.4 all-radix-4
 * split (MKL's own radix choice; r8 split mids hold 16 planes live -> spill
 * risk, r4 holds 8 — measurement decides), + champion 4.8.16.8 + MKL.
 * Scratch layout = BLOCK-split: 64B [re x4][im x4] blocks (same bytes as 4
 * z-complex; addressing = z's with +4 for the im half; ONE stream per leg
 * row — MKL's granularity). The first cut used FULL split planes and LOST at
 * 16384 (+7.7%: two streams per leg row); block-split swung it +29% — banked.
 *
 * Build: python build.py --src benches/zil_split_interior.c --mkl
 * Run:   zil_split_interior.exe [N]     (4096 | 16384)
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
D(radix8_z_n1_fwd_avx2) D(radix8_z_t2c_fwd_avx2) D(radix8_z_t2sp_fwd_avx2)
D(radix4_z_n1_fwd_avx2) D(radix4_z_t2sp_fwd_avx2)
D(radix16_z_t2c_fwd_avx2) D(radix8_z_t2_fwd_avx2)

/* ── helpers: z<->split 4-column repack (AVX2 vpermpd + unpck) ──────────── */
#define DEINT(zlo, zhi, re, im) do {                                  \
    __m256d _u = _mm256_unpacklo_pd(zlo, zhi);  /* r0 r2 | r1 r3 */   \
    __m256d _v = _mm256_unpackhi_pd(zlo, zhi);  /* i0 i2 | i1 i3 */   \
    re = _mm256_permute4x64_pd(_u, 0xD8);       /* r0 r1 r2 r3 */     \
    im = _mm256_permute4x64_pd(_v, 0xD8);                             \
} while (0)
#define REINT(re, im, zlo, zhi) do {                                  \
    __m256d _p = _mm256_permute4x64_pd(re, 0xD8); /* r0 r2 r1 r3 */   \
    __m256d _q = _mm256_permute4x64_pd(im, 0xD8);                     \
    zlo = _mm256_unpacklo_pd(_p, _q);           /* r0 i0 | r1 i1 */   \
    zhi = _mm256_unpackhi_pd(_p, _q);           /* r2 i2 | r3 i3 */   \
} while (0)

/* NOTE on DEINT lane math: zlo=[r0 i0 | r1 i1], zhi=[r2 i2 | r3 i3].
 * unpacklo(zlo,zhi) = [zlo0 zhi0 | zlo2 zhi2] = [r0 r2 | r1 r3]; permute
 * 0xD8 (0,2,1,3) -> [r0 r1 r2 r3]. Verified by the gates below. */

/* ── split radix-4 butterfly (per plane pair, 4 columns, ZERO shuffles) ──
 * outputs may alias inputs (all loaded first by the callers) */
#define SPLIT_BFLY4(i0r,i0i,i1r,i1i,i2r,i2i,i3r,i3i, o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i) do { \
    __m256d t0r=_mm256_add_pd(i0r,i2r), t0i=_mm256_add_pd(i0i,i2i);   \
    __m256d t1r=_mm256_sub_pd(i0r,i2r), t1i=_mm256_sub_pd(i0i,i2i);   \
    __m256d t2r=_mm256_add_pd(i1r,i3r), t2i=_mm256_add_pd(i1i,i3i);   \
    __m256d t3r=_mm256_sub_pd(i1r,i3r), t3i=_mm256_sub_pd(i1i,i3i);   \
    o0r=_mm256_add_pd(t0r,t2r); o0i=_mm256_add_pd(t0i,t2i);           \
    o2r=_mm256_sub_pd(t0r,t2r); o2i=_mm256_sub_pd(t0i,t2i);           \
    o1r=_mm256_add_pd(t1r,t3i); o1i=_mm256_sub_pd(t1i,t3r);  /* +(-i)t3 */ \
    o3r=_mm256_sub_pd(t1r,t3i); o3i=_mm256_add_pd(t1i,t3r);  /* -(-i)t3 */ \
} while (0)

/* split cmul by (ct, st): re' = a*ct - b*st ; im' = a*st + b*ct */
#define SPLIT_CMUL(ar,ai, ct,st, or_,oi_) do {                        \
    or_ = _mm256_fnmadd_pd(st, ai, _mm256_mul_pd(ct, ar));            \
    oi_ = _mm256_fmadd_pd(st, ar, _mm256_mul_pd(ct, ai));             \
} while (0)

/* ── split radix-8 butterfly core (planes; C=0.7071; zero shuffles).
 * Derived from the gated z r8 body with rotations as renames. ── */
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

/* ── S0: z -> split leaf, radix 8 (twiddle-free), OOP z0 -> (re,im) ── */
__attribute__((target("avx2,fma")))
static void s0z2s_r8(const double *z, double *sp, long Ls, long count)
{
    for (long k = 0; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8];
        for (int l = 0; l < 8; l++) {
            __m256d zlo = _mm256_loadu_pd(z + 2 * ((size_t)l * Ls + k));
            __m256d zhi = _mm256_loadu_pd(z + 2 * ((size_t)l * Ls + k) + 4);
            DEINT(zlo, zhi, xr[l], xi[l]);
        }
        __m256d or_[8], oi_[8];
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

/* ── split mids (in-place; tw = per-leg splat pairs [c x4][s x4], legs 1..R-1).
 * z2 variant re-interleaves into the z buffer instead (split -> z). ── */
__attribute__((target("avx2,fma")))
static void mid_s2s_r8(double *sp, const double *tw, long Dv, long count)
{
    for (long k = 0; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8];
        xr[0] = _mm256_loadu_pd(sp + 2 * (size_t)k);
        xi[0] = _mm256_loadu_pd(sp + 2 * (size_t)k + 4);
        for (int l = 1; l < 8; l++) {
            __m256d ar = _mm256_loadu_pd(sp + 2 * ((size_t)l * Dv + k));
            __m256d ai = _mm256_loadu_pd(sp + 2 * ((size_t)l * Dv + k) + 4);
            __m256d ct = _mm256_loadu_pd(tw + (size_t)(l - 1) * 8);
            __m256d st = _mm256_loadu_pd(tw + (size_t)(l - 1) * 8 + 4);
            SPLIT_CMUL(ar, ai, ct, st, xr[l], xi[l]);
        }
        __m256d or_[8], oi_[8];
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
static void mid_s2z_r8(const double *sp, double *z,
                       const double *tw, long Dv, long count)
{
    for (long k = 0; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8];
        xr[0] = _mm256_loadu_pd(sp + 2 * (size_t)k);
        xi[0] = _mm256_loadu_pd(sp + 2 * (size_t)k + 4);
        for (int l = 1; l < 8; l++) {
            __m256d ar = _mm256_loadu_pd(sp + 2 * ((size_t)l * Dv + k));
            __m256d ai = _mm256_loadu_pd(sp + 2 * ((size_t)l * Dv + k) + 4);
            __m256d ct = _mm256_loadu_pd(tw + (size_t)(l - 1) * 8);
            __m256d st = _mm256_loadu_pd(tw + (size_t)(l - 1) * 8 + 4);
            SPLIT_CMUL(ar, ai, ct, st, xr[l], xi[l]);
        }
        __m256d or_[8], oi_[8];
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
static void mid_s2z_r4(const double *sp, double *z,
                       const double *tw, long Dv, long count)
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

/* ── chain plumbing (mirrors zil_chain_dp.c) ── */
static int N;
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
static void vtw2_rec(double *rec, double a0, double a1)
{
    rec[0] = cos(a0); rec[1] = cos(a0); rec[2] = cos(a1); rec[3] = cos(a1);
    rec[4] = -sin(a0); rec[5] = sin(a0); rec[6] = -sin(a1); rec[7] = sin(a1);
}

typedef struct {
    int nf, R[8];
    long Dv[8], G[8];
    double *tws[8];     /* split splat-pair tables (mids) */
    double *twz[8];     /* z t2c tables (z-interior control arm) */
    long *gb[8];        /* base tables */
    double *twp1;       /* terminator w^1 stream (both arms share) */
    char name[40];
} sc_t;

static void build_sc(sc_t *c)
{
    const double TAU = 2.0 * M_PI;
    const int nf = c->nf;
    c->Dv[nf - 1] = 1;
    for (int i = nf - 2; i >= 0; i--) c->Dv[i] = c->Dv[i + 1] * c->R[i + 1];
    c->G[0] = 1;
    for (int i = 1; i < nf; i++) c->G[i] = c->G[i - 1] * c->R[i - 1];
    for (int s = 1; s <= nf - 2; s++) {
        int Rm1 = c->R[s] - 1;
        long M = (long)N / c->Dv[s];
        c->tws[s] = (double *)_mm_malloc((size_t)c->G[s] * Rm1 * 8 * 8, 64);
        c->twz[s] = (double *)_mm_malloc((size_t)c->G[s] * Rm1 * 8 * 8, 64);
        c->gb[s] = (long *)malloc((size_t)c->G[s] * sizeof(long));
        for (long g = 0; g < c->G[s]; g++) {
            c->gb[s][g] = base_of(g, s, c->R, c->Dv);
            long brev = brev_prefix(g, s, c->R);
            for (int l = 1; l < c->R[s]; l++) {
                double a = -TAU * (double)(((long)l * brev) % M) / (double)M;
                double *sp = c->tws[s] + ((size_t)g * Rm1 + (l - 1)) * 8;
                for (int j = 0; j < 4; j++) { sp[j] = cos(a); sp[4 + j] = sin(a); }
                vtw2_rec(c->twz[s] + ((size_t)g * Rm1 + (l - 1)) * 8, a, a);
            }
        }
    }
    {
        int Rt = c->R[nf - 1];
        long pairs = ((long)N / Rt) / 2;
        c->twp1 = (double *)_mm_malloc((size_t)pairs * 64, 64);
        for (long p = 0; p < pairs; p++) {
            long b0 = brev_prefix(2 * p, nf - 1, c->R);
            long b1 = brev_prefix(2 * p + 1, nf - 1, c->R);
            vtw2_rec(c->twp1 + (size_t)p * 8,
                     -TAU * (double)(b0 % N) / (double)N,
                     -TAU * (double)(b1 % N) / (double)N);
        }
    }
    char *w = c->name;
    for (int i = 0; i < nf; i++) w += sprintf(w, "%s%d", i ? "." : "", c->R[i]);
}

/* split-interior execute: S0 z->split, mids split (last converts to z), z term */
static void run_split(const sc_t *c, const double *z0, double *sp,
                      double *A, double *out)
{
    const int nf = c->nf;
    if (c->R[0] == 8) s0z2s_r8(z0, sp, c->Dv[0], c->Dv[0]);
    else              s0z2s_r4(z0, sp, c->Dv[0], c->Dv[0]);
    for (int s = 1; s <= nf - 3; s++) {          /* pure split mids */
        int Rm1 = c->R[s] - 1;
        for (long g = 0; g < c->G[s]; g++) {
            long b = c->gb[s][g];
            const double *tw = c->tws[s] + (size_t)g * Rm1 * 8;
            if (c->R[s] == 8) mid_s2s_r8(sp + 2 * b, tw, c->Dv[s], c->Dv[s]);
            else              mid_s2s_r4(sp + 2 * b, tw, c->Dv[s], c->Dv[s]);
        }
    }
    {                                            /* last mid: split -> z */
        int s = nf - 2, Rm1 = c->R[s] - 1;
        for (long g = 0; g < c->G[s]; g++) {
            long b = c->gb[s][g];
            const double *tw = c->tws[s] + (size_t)g * Rm1 * 8;
            if (c->R[s] == 8) mid_s2z_r8(sp + 2 * b, A + 2 * b, tw, c->Dv[s], c->Dv[s]);
            else              mid_s2z_r4(sp + 2 * b, A + 2 * b, tw, c->Dv[s], c->Dv[s]);
        }
    }
    {
        int Rt = c->R[nf - 1];
        zfn tf = (Rt == 8) ? radix8_z_t2sp_fwd_avx2 : radix4_z_t2sp_fwd_avx2;
        tf(A, 0, out, 0, c->twp1, 0, 1, (unsigned long long)Rt,
           (unsigned long long)(N / Rt), 1, (unsigned long long)(N / Rt));
    }
}

/* z-interior control (t2c mids + t2sp term), same chain */
static void run_zctl(const sc_t *c, const double *z0, double *A, double *out)
{
    const int nf = c->nf;
    zfn s0 = (c->R[0] == 8) ? radix8_z_n1_fwd_avx2 : radix4_z_n1_fwd_avx2;
    s0(z0, 0, A, 0, 0, 0, (unsigned long long)c->Dv[0], 0,
       (unsigned long long)c->Dv[0], 0, (unsigned long long)c->Dv[0]);
    for (int s = 1; s <= nf - 2; s++) {
        int Rm1 = c->R[s] - 1;
        for (long g = 0; g < c->G[s]; g++) {
            long b = c->gb[s][g];
            radix8_z_t2c_fwd_avx2(A + 2 * b, 0, A + 2 * b, 0,
                                  c->twz[s] + (size_t)g * Rm1 * 8, 0,
                                  (unsigned long long)c->Dv[s], 0,
                                  (unsigned long long)c->Dv[s], 0,
                                  (unsigned long long)c->Dv[s]);
        }
    }
    {
        int Rt = c->R[nf - 1];
        zfn tf = (Rt == 8) ? radix8_z_t2sp_fwd_avx2 : radix4_z_t2sp_fwd_avx2;
        tf(A, 0, out, 0, c->twp1, 0, 1, (unsigned long long)Rt,
           (unsigned long long)(N / Rt), 1, (unsigned long long)(N / Rt));
    }
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
    if (N != 4096 && N != 16384) { printf("spike supports N=4096|16384\n"); return 1; }

    sc_t A8 = { 4, { 8, 8, 8, 8 } };        /* 4096 chain A: all-r8 */
    sc_t B4 = { 6, { 4, 4, 4, 4, 4, 4 } };  /* 4096 chain B: all-r4 (MKL's pick) */
    sc_t C5 = { 5, { 4, 8, 8, 8, 8 } };     /* 16384 winner chain */
    if (N == 4096) { build_sc(&A8); build_sc(&B4); }
    else build_sc(&C5);

    double *z0 = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *zA = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    double *z  = (double *)_mm_malloc((size_t)2 * N * 8, 64);
    /* BLOCK-split scratch: [re x4][im x4] 64B blocks = one stream per leg row
     * (MKL's layout; the full-split two-plane variant doubled streams and lost
     * at 16384 � banked in-run) */
    double *sp = (double *)_mm_malloc((size_t)2 * N * 8, 64);
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

    /* gates */
    struct { const char *nm; int arm; sc_t *c; } arms[3];
    int na;
    if (N == 4096) {
        arms[0].nm = "A 8.8.8.8   SPLIT-interior"; arms[0].arm = 0; arms[0].c = &A8;
        arms[1].nm = "A 8.8.8.8   z-interior ctl"; arms[1].arm = 1; arms[1].c = &A8;
        arms[2].nm = "B 4^6       SPLIT-interior"; arms[2].arm = 0; arms[2].c = &B4;
        na = 3;
    } else {
        arms[0].nm = "C 4.8.8.8.8 SPLIT-interior"; arms[0].arm = 0; arms[0].c = &C5;
        arms[1].nm = "C 4.8.8.8.8 z-interior ctl"; arms[1].arm = 1; arms[1].c = &C5;
        na = 2;
    }
    for (int a = 0; a < na; a++) {
        sc_t *c = arms[a].c;
        if (arms[a].arm == 0) run_split(c, z0, sp, zA, z);
        else run_zctl(c, z0, zA, z);
        int Rt = c->R[c->nf - 1];
        long NR = (long)N / Rt;
        double err = 0;
        for (long idx = 0; idx < N; idx++) {
            long l = idx / NR, g = idx % NR;
            long m = drev_full(g * Rt + l, c->R, c->nf);
            double d = fabs(z[2 * idx] - Rr[m]) + fabs(z[2 * idx + 1] - Ri[m]);
            if (d > err) err = d;
        }
        printf("GATE %-28s relerr=%.3e %s\n", arms[a].nm, err / mag,
               (err / mag < 1e-11) ? "PASS" : "FAIL");
        if (err / mag >= 1e-11) return 1;
    }

    DFTI_DESCRIPTOR_HANDLE h = NULL;
    DftiCreateDescriptor(&h, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N);
    DftiCommitDescriptor(h);
    int reps = (int)(4.0e6 / N); if (reps < 200) reps = 200;
    int narm = na + 1;
    double best[4] = { 1e18, 1e18, 1e18, 1e18 };
    for (int t = 0; t < 9; t++) {
        if (t) cachebust();
        for (int q = 0; q < narm; q++) {
            int a = (t & 1) ? (narm - 1 - q) : q;
            double t0, ns;
            if (a < na) {
                sc_t *c = arms[a].c;
                if (arms[a].arm == 0) {
                    for (int w = 0; w < 6; w++) run_split(c, z0, sp, zA, z);
                    t0 = now_ms();
                    for (int i = 0; i < reps; i++) run_split(c, z0, sp, zA, z);
                } else {
                    for (int w = 0; w < 6; w++) run_zctl(c, z0, zA, z);
                    t0 = now_ms();
                    for (int i = 0; i < reps; i++) run_zctl(c, z0, zA, z);
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
    printf("\n# N=%d LEVER-5 block-split interior vs z interior (same chains)\n", N);
    for (int a = 0; a < na; a++)
        printf("%-30s %9.1f ns   vsMKL %.2f\n", arms[a].nm, best[a], best[na] / best[a]);
    printf("MKL-IL                         %9.1f ns\n", best[na]);
    printf("DONE\n");
    return 0;
}
