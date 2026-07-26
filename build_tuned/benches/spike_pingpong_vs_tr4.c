/* spike_pingpong_vs_tr4.c — EARLY-KILL microbench for the z-cascade restructure.
 *
 * QUESTION
 * --------
 * The proposed restructure buys "no load-side corner-turn in the terminator"
 * by paying "a SECOND N-complex interior plane" (ping-pong instead of in-place
 * mids).  Is the extra plane's memory traffic cheaper than the ~1 shuffle per
 * complex it removes?
 *
 *   ARM A  ("in-place interior"):  read plane P, TR4 corner-turn on the LOAD
 *                                  side, radix-8 arithmetic, REINT stores back
 *                                  to plane P (IN PLACE).  Footprint = 1 plane.
 *   ARM B  ("ping-pong interior"): read plane X, NO corner-turn (the ingest
 *                                  already turned it), IDENTICAL radix-8
 *                                  arithmetic, IDENTICAL REINT stores, but to a
 *                                  DISTINCT plane Y.  Footprint = 2 planes.
 *
 * Everything except (TR4 present/absent) and (1 plane / 2 planes) is bit-for-bit
 * the same instruction stream: same load count (16 x _mm256_loadu_pd), same
 * store count (16 x _mm256_storeu_pd), same twiddle squaring tree, same
 * SPLIT_BFLY8, same REINT store edge.  So  A - B  =  (cost of 1 shuffle/complex)
 * - (cost of the second plane).   A > B  ==>  the extra plane pays for itself.
 *
 * FIDELITY
 * --------
 * TR4, REINT, SPLIT_CMUL, SPLIT_BFLY8 and the packed-w^1 squaring tree are
 * copied verbatim out of the generated production kernels
 *   src/dag-fft-compiler/codelets/zil/avx2/radix8_z_sterm_avx2.c   (TR4, REINT,
 *                                                                  squaring tree)
 *   src/dag-fft-compiler/codelets/zil/avx2/radix4_z_ms_avx2.c      (SPLIT_CMUL,
 *                                                                  SPLIT_BFLY8)
 * Layout is the production 64-B [re x4][im x4] block-split plane; a "pass"
 * covers exactly N complex (count = N/8 columns, 4 columns / iteration,
 * 32 complex / iteration, N/32 iterations).
 *
 * Port-5 accounting per iteration (32 complex):
 *   ARM A: 4 x TR4 (8 shuf each) + 8 x REINT (4 shuf each) = 32 + 32 = 64
 *   ARM B:                         8 x REINT (4 shuf each) =  0 + 32 = 32
 *   delta = 32 shuffles / 32 complex = 1.00 shuffle per complex.   <-- the brief
 *
 * HARNESS DISCIPLINE (this project has been burned repeatedly)
 * -----------------------------------------------------------
 *   - pinned to logical core 2 (affinity mask 4), HIGH_PRIORITY_CLASS
 *   - ONE arena, planes carved out with 64-BYTE SKEWS so no two streams are
 *     4KB-congruent (the 4KB-aliasing trap has been hit twice in this project)
 *   - 32 MB cachebust + plane re-init before EVERY timed region (identical
 *     treatment for every arm)
 *   - Sleep() pacing between arms
 *   - arm order ROTATED every rep (never fixed order)
 *   - best-of-5 over reps
 *   - CONTROL ARMS: A1/A2 are byte-identical code on byte-identical data; so are
 *     B1/B2.  They land in different rotation positions, so min(A2)/min(A1) and
 *     min(B2)/min(B1) MUST read ~1.00.  Two extra placement controls (Ap = ARM A
 *     in-place on a different plane, Bp = ARM B on a different plane pair) catch
 *     arena-placement bias, which the position control cannot see.
 *   - FTZ/DAZ on (equal treatment; keeps ARM A's in-place iteration out of
 *     denormal-stall territory).
 *
 * NOTE ON "one shared output buffer": the arms differ *precisely* in whether the
 * output aliases the input, so they structurally cannot share one buffer -- that
 * is the variable under test.  The placement controls Ap/Bp are the substitute:
 * if buffer identity were driving the result, Ap != A and/or Bp != B.
 */

#include <immintrin.h>
#include <windows.h>
#include <malloc.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ------------------------------------------------------------------ timing */
static double g_qpc_hz;
static void   timer_init(void){ LARGE_INTEGER f; QueryPerformanceFrequency(&f); g_qpc_hz=(double)f.QuadPart; }
static double now_s(void){ LARGE_INTEGER c; QueryPerformanceCounter(&c); return (double)c.QuadPart/g_qpc_hz; }

/* --------------------------------------------------- production primitives */
/* TR4: load-side 4x4 corner-turn (verbatim from radix8_z_sterm_avx2.c).
 * 4 unpack + 4 vperm2f128 = 8 port-5 ops. */
#define TR4(b0,b1,b2,b3, l0,l1,l2,l3) do {                                     \
    __m256d _u0 = _mm256_unpacklo_pd(b0, b1);                                  \
    __m256d _u1 = _mm256_unpackhi_pd(b0, b1);                                  \
    __m256d _u2 = _mm256_unpacklo_pd(b2, b3);                                  \
    __m256d _u3 = _mm256_unpackhi_pd(b2, b3);                                  \
    l0 = _mm256_permute2f128_pd(_u0, _u2, 0x20);                               \
    l1 = _mm256_permute2f128_pd(_u1, _u3, 0x20);                               \
    l2 = _mm256_permute2f128_pd(_u0, _u2, 0x31);                               \
    l3 = _mm256_permute2f128_pd(_u1, _u3, 0x31);                               \
} while (0)

/* REINT: store-side re-interleave (verbatim). 2 vpermpd + 2 unpack = 4 port-5. */
#define REINT(re, im, zlo, zhi) do {                                           \
    __m256d _p = _mm256_permute4x64_pd(re, 0xD8);                              \
    __m256d _q = _mm256_permute4x64_pd(im, 0xD8);                              \
    zlo = _mm256_unpacklo_pd(_p, _q);                                          \
    zhi = _mm256_unpackhi_pd(_p, _q);                                          \
} while (0)

#define SPLIT_CMUL(ar,ai, ct,st, or_,oi_) do {                                 \
    or_ = _mm256_fnmadd_pd(st, ai, _mm256_mul_pd(ct, ar));                     \
    oi_ = _mm256_fmadd_pd (st, ar, _mm256_mul_pd(ct, ai));                     \
} while (0)

#define SPLIT_BFLY8(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i, \
                    o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i) do { \
    const __m256d _C = _mm256_set1_pd(0.70710678118654752440);                 \
    __m256d t0r=_mm256_add_pd(x0r,x4r), t0i=_mm256_add_pd(x0i,x4i);            \
    __m256d t1r=_mm256_sub_pd(x0r,x4r), t1i=_mm256_sub_pd(x0i,x4i);            \
    __m256d t2r=_mm256_add_pd(x2r,x6r), t2i=_mm256_add_pd(x2i,x6i);            \
    __m256d t3r=_mm256_sub_pd(x2r,x6r), t3i=_mm256_sub_pd(x2i,x6i);            \
    __m256d E0r=_mm256_add_pd(t0r,t2r), E0i=_mm256_add_pd(t0i,t2i);            \
    __m256d E2r=_mm256_sub_pd(t0r,t2r), E2i=_mm256_sub_pd(t0i,t2i);            \
    __m256d E1r=_mm256_add_pd(t1r,t3i), E1i=_mm256_sub_pd(t1i,t3r);            \
    __m256d E3r=_mm256_sub_pd(t1r,t3i), E3i=_mm256_add_pd(t1i,t3r);            \
    __m256d s0r=_mm256_add_pd(x1r,x5r), s0i=_mm256_add_pd(x1i,x5i);            \
    __m256d s1r=_mm256_sub_pd(x1r,x5r), s1i=_mm256_sub_pd(x1i,x5i);            \
    __m256d s2r=_mm256_add_pd(x3r,x7r), s2i=_mm256_add_pd(x3i,x7i);            \
    __m256d s3r=_mm256_sub_pd(x3r,x7r), s3i=_mm256_sub_pd(x3i,x7i);            \
    __m256d O0r=_mm256_add_pd(s0r,s2r), O0i=_mm256_add_pd(s0i,s2i);            \
    __m256d O2r=_mm256_sub_pd(s0r,s2r), O2i=_mm256_sub_pd(s0i,s2i);            \
    __m256d O1r=_mm256_add_pd(s1r,s3i), O1i=_mm256_sub_pd(s1i,s3r);            \
    __m256d O3r=_mm256_sub_pd(s1r,s3i), O3i=_mm256_add_pd(s1i,s3r);            \
    __m256d X1r=_mm256_add_pd(O1r,O1i), X1i=_mm256_sub_pd(O1i,O1r);            \
    __m256d X3r=_mm256_sub_pd(O3i,O3r), X3n=_mm256_add_pd(O3r,O3i);            \
    o0r=_mm256_add_pd(E0r,O0r); o0i=_mm256_add_pd(E0i,O0i);                    \
    o4r=_mm256_sub_pd(E0r,O0r); o4i=_mm256_sub_pd(E0i,O0i);                    \
    o1r=_mm256_fmadd_pd (_C,X1r,E1r); o1i=_mm256_fmadd_pd (_C,X1i,E1i);        \
    o5r=_mm256_fnmadd_pd(_C,X1r,E1r); o5i=_mm256_fnmadd_pd(_C,X1i,E1i);        \
    o2r=_mm256_add_pd(E2r,O2i); o2i=_mm256_sub_pd(E2i,O2r);                    \
    o6r=_mm256_sub_pd(E2r,O2i); o6i=_mm256_add_pd(E2i,O2r);                    \
    o3r=_mm256_fmadd_pd (_C,X3r,E3r); o3i=_mm256_fnmadd_pd(_C,X3n,E3i);        \
    o7r=_mm256_fnmadd_pd(_C,X3r,E3r); o7i=_mm256_fmadd_pd (_C,X3n,E3i);        \
} while (0)

/* packed per-column w^1 -> w^2..w^7 by the production squaring tree (verbatim
 * shape from radix8_z_sterm_avx2.c: t1,t2 loaded, t3..t10/t28/t29/t52/t53 built) */
#define TWTREE()                                                               \
    const __m256d c1 = _mm256_loadu_pd(&tw[2*k + 0]);                          \
    const __m256d s1 = _mm256_loadu_pd(&tw[2*k + 4]);                          \
    const __m256d c2 = _mm256_fnmadd_pd(s1,s1,_mm256_mul_pd(c1,c1));           \
    const __m256d s2 = _mm256_fmadd_pd (c1,s1,_mm256_mul_pd(s1,c1));           \
    const __m256d c3 = _mm256_fnmadd_pd(s2,s1,_mm256_mul_pd(c2,c1));           \
    const __m256d s3 = _mm256_fmadd_pd (c2,s1,_mm256_mul_pd(s2,c1));           \
    const __m256d c4 = _mm256_fnmadd_pd(s2,s2,_mm256_mul_pd(c2,c2));           \
    const __m256d s4 = _mm256_fmadd_pd (c2,s2,_mm256_mul_pd(s2,c2));           \
    const __m256d c5 = _mm256_fnmadd_pd(s4,s1,_mm256_mul_pd(c4,c1));           \
    const __m256d s5 = _mm256_fmadd_pd (c4,s1,_mm256_mul_pd(s4,c1));           \
    const __m256d c6 = _mm256_fnmadd_pd(s4,s2,_mm256_mul_pd(c4,c2));           \
    const __m256d s6 = _mm256_fmadd_pd (c4,s2,_mm256_mul_pd(s4,c2));           \
    const __m256d c7 = _mm256_fnmadd_pd(s4,s3,_mm256_mul_pd(c4,c3));           \
    const __m256d s7 = _mm256_fmadd_pd (c4,s3,_mm256_mul_pd(s4,c3));

/* twiddle + butterfly + unit-gain rescale + REINT stores.  Shared verbatim by
 * both arms so the ONLY difference between them is the load edge. */
#define BODY_AND_STORE()                                                       \
    TWTREE();                                                                  \
    __m256d x0r = lane_re_0, x0i = lane_im_0;                                  \
    __m256d x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i;           \
    SPLIT_CMUL(lane_re_1,lane_im_1, c1,s1, x1r,x1i);                           \
    SPLIT_CMUL(lane_re_2,lane_im_2, c2,s2, x2r,x2i);                           \
    SPLIT_CMUL(lane_re_3,lane_im_3, c3,s3, x3r,x3i);                           \
    SPLIT_CMUL(lane_re_4,lane_im_4, c4,s4, x4r,x4i);                           \
    SPLIT_CMUL(lane_re_5,lane_im_5, c5,s5, x5r,x5i);                           \
    SPLIT_CMUL(lane_re_6,lane_im_6, c6,s6, x6r,x6i);                           \
    SPLIT_CMUL(lane_re_7,lane_im_7, c7,s7, x7r,x7i);                           \
    __m256d o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i;   \
    SPLIT_BFLY8(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i,\
                o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i);\
    /* unit-L2-gain rescale so the in-place arm can iterate forever w/o drift */\
    { const __m256d _RS = _mm256_set1_pd(0.35355339059327376220);              \
      o0r=_mm256_mul_pd(o0r,_RS); o0i=_mm256_mul_pd(o0i,_RS);                  \
      o1r=_mm256_mul_pd(o1r,_RS); o1i=_mm256_mul_pd(o1i,_RS);                  \
      o2r=_mm256_mul_pd(o2r,_RS); o2i=_mm256_mul_pd(o2i,_RS);                  \
      o3r=_mm256_mul_pd(o3r,_RS); o3i=_mm256_mul_pd(o3i,_RS);                  \
      o4r=_mm256_mul_pd(o4r,_RS); o4i=_mm256_mul_pd(o4i,_RS);                  \
      o5r=_mm256_mul_pd(o5r,_RS); o5i=_mm256_mul_pd(o5i,_RS);                  \
      o6r=_mm256_mul_pd(o6r,_RS); o6i=_mm256_mul_pd(o6i,_RS);                  \
      o7r=_mm256_mul_pd(o7r,_RS); o7i=_mm256_mul_pd(o7i,_RS); }                \
    { __m256d zlo, zhi;                                                        \
      REINT(o0r,o0i,zlo,zhi); _mm256_storeu_pd(&out[16*k +  0],zlo); _mm256_storeu_pd(&out[16*k +  4],zhi); \
      REINT(o1r,o1i,zlo,zhi); _mm256_storeu_pd(&out[16*k +  8],zlo); _mm256_storeu_pd(&out[16*k + 12],zhi); \
      REINT(o2r,o2i,zlo,zhi); _mm256_storeu_pd(&out[16*k + 16],zlo); _mm256_storeu_pd(&out[16*k + 20],zhi); \
      REINT(o3r,o3i,zlo,zhi); _mm256_storeu_pd(&out[16*k + 24],zlo); _mm256_storeu_pd(&out[16*k + 28],zhi); \
      REINT(o4r,o4i,zlo,zhi); _mm256_storeu_pd(&out[16*k + 32],zlo); _mm256_storeu_pd(&out[16*k + 36],zhi); \
      REINT(o5r,o5i,zlo,zhi); _mm256_storeu_pd(&out[16*k + 40],zlo); _mm256_storeu_pd(&out[16*k + 44],zhi); \
      REINT(o6r,o6i,zlo,zhi); _mm256_storeu_pd(&out[16*k + 48],zlo); _mm256_storeu_pd(&out[16*k + 52],zhi); \
      REINT(o7r,o7i,zlo,zhi); _mm256_storeu_pd(&out[16*k + 56],zlo); _mm256_storeu_pd(&out[16*k + 60],zhi); }

/* ------------------------------------------------- ARM A: in-place + TR4 */
__attribute__((noinline, target("avx2,fma")))
static void pass_turn(double * __restrict__ in, double * __restrict__ out,
                      const double * __restrict__ tw, size_t count)
{
    for (size_t k = 0; k + 4 <= count; k += 4) {
        /* Block load edge (TR4) — verbatim addressing from the sterm kernel */
        const __m256d br0_0 = _mm256_loadu_pd(&in[16*k +  0]);
        const __m256d bi0_0 = _mm256_loadu_pd(&in[16*k +  4]);
        const __m256d br1_0 = _mm256_loadu_pd(&in[16*k +  8]);
        const __m256d bi1_0 = _mm256_loadu_pd(&in[16*k + 12]);
        const __m256d br0_1 = _mm256_loadu_pd(&in[16*(k+1) +  0]);
        const __m256d bi0_1 = _mm256_loadu_pd(&in[16*(k+1) +  4]);
        const __m256d br1_1 = _mm256_loadu_pd(&in[16*(k+1) +  8]);
        const __m256d bi1_1 = _mm256_loadu_pd(&in[16*(k+1) + 12]);
        const __m256d br0_2 = _mm256_loadu_pd(&in[16*(k+2) +  0]);
        const __m256d bi0_2 = _mm256_loadu_pd(&in[16*(k+2) +  4]);
        const __m256d br1_2 = _mm256_loadu_pd(&in[16*(k+2) +  8]);
        const __m256d bi1_2 = _mm256_loadu_pd(&in[16*(k+2) + 12]);
        const __m256d br0_3 = _mm256_loadu_pd(&in[16*(k+3) +  0]);
        const __m256d bi0_3 = _mm256_loadu_pd(&in[16*(k+3) +  4]);
        const __m256d br1_3 = _mm256_loadu_pd(&in[16*(k+3) +  8]);
        const __m256d bi1_3 = _mm256_loadu_pd(&in[16*(k+3) + 12]);
        __m256d lane_re_0,lane_re_1,lane_re_2,lane_re_3;
        __m256d lane_im_0,lane_im_1,lane_im_2,lane_im_3;
        __m256d lane_re_4,lane_re_5,lane_re_6,lane_re_7;
        __m256d lane_im_4,lane_im_5,lane_im_6,lane_im_7;
        TR4(br0_0,br0_1,br0_2,br0_3, lane_re_0,lane_re_1,lane_re_2,lane_re_3);
        TR4(bi0_0,bi0_1,bi0_2,bi0_3, lane_im_0,lane_im_1,lane_im_2,lane_im_3);
        TR4(br1_0,br1_1,br1_2,br1_3, lane_re_4,lane_re_5,lane_re_6,lane_re_7);
        TR4(bi1_0,bi1_1,bi1_2,bi1_3, lane_im_4,lane_im_5,lane_im_6,lane_im_7);
        BODY_AND_STORE();
    }
}

/* ------------------------------------------- ARM B: ping-pong, no corner-turn
 * The ingest already wrote the plane turned, so leg j's (re,im) quad is simply
 * a contiguous 64-B block.  Exactly the same 64 doubles are read per iteration
 * as ARM A reads — same bytes, same stream, same 16 vector loads. */
__attribute__((noinline, target("avx2,fma")))
static void pass_noturn(const double * __restrict__ in, double * __restrict__ out,
                        const double * __restrict__ tw, size_t count)
{
    for (size_t k = 0; k + 4 <= count; k += 4) {
        const __m256d lane_re_0 = _mm256_loadu_pd(&in[16*k +  0]);
        const __m256d lane_im_0 = _mm256_loadu_pd(&in[16*k +  4]);
        const __m256d lane_re_1 = _mm256_loadu_pd(&in[16*k +  8]);
        const __m256d lane_im_1 = _mm256_loadu_pd(&in[16*k + 12]);
        const __m256d lane_re_2 = _mm256_loadu_pd(&in[16*k + 16]);
        const __m256d lane_im_2 = _mm256_loadu_pd(&in[16*k + 20]);
        const __m256d lane_re_3 = _mm256_loadu_pd(&in[16*k + 24]);
        const __m256d lane_im_3 = _mm256_loadu_pd(&in[16*k + 28]);
        const __m256d lane_re_4 = _mm256_loadu_pd(&in[16*k + 32]);
        const __m256d lane_im_4 = _mm256_loadu_pd(&in[16*k + 36]);
        const __m256d lane_re_5 = _mm256_loadu_pd(&in[16*k + 40]);
        const __m256d lane_im_5 = _mm256_loadu_pd(&in[16*k + 44]);
        const __m256d lane_re_6 = _mm256_loadu_pd(&in[16*k + 48]);
        const __m256d lane_im_6 = _mm256_loadu_pd(&in[16*k + 52]);
        const __m256d lane_re_7 = _mm256_loadu_pd(&in[16*k + 56]);
        const __m256d lane_im_7 = _mm256_loadu_pd(&in[16*k + 60]);
        BODY_AND_STORE();
    }
}

/* --------------------------------------------------------------- harness */
/* 64 MB > the i9-14900KF's 36 MB L3, so a bust really evicts the planes.
 * (32 MB left ~4 MB of L3 unswept and the small planes could survive it.) */
#define CACHEBUST_BYTES (64u << 20)
static volatile double g_sink;
static unsigned char *g_bust;

static void cachebust(void)
{
    volatile unsigned char *p = g_bust;
    unsigned char acc = 0;
    for (size_t i = 0; i < CACHEBUST_BYTES; i += 64) { p[i] += (unsigned char)(i + 1); acc += p[i]; }
    g_sink += (double)acc;
}

/* C is the DECOMPOSITION baseline: 1 plane, in place, and NO corner-turn.
 *   Q2 = cost of 1 shuffle/complex   = A  - C   (add TR4 to a 1-plane stream)
 *   Q1 = cost of the 2nd plane       = Ba - C   (add plane 2 to a no-TR4 stream)
 * The design pays off iff Q2 > Q1. */
typedef enum { ARM_A1, ARM_A2, ARM_AP, ARM_B1, ARM_B2, ARM_BP, ARM_BALT, ARM_C, N_ARMS } arm_t;
static const char *arm_name[N_ARMS] = {
    "A1  in-place +TR4  (P)",
    "A2  in-place +TR4  (P)    [position control for A]",
    "Ap  in-place +TR4  (X)    [placement control for A]",
    "B1  pingpong -TR4  (X->Y)",
    "B2  pingpong -TR4  (X->Y) [position control for B]",
    "Bp  pingpong -TR4  (Y->P) [placement control for B]",
    "Ba  pingpong -TR4  (X<->Y ALTERNATING: both planes dirty — the honest one)",
    "C   in-place -TR4  (P)    [decomposition baseline: 1 plane, no shuffles]",
};

static int cmp_dbl(const void *a, const void *b)
{ double x = *(const double*)a, y = *(const double*)b; return (x < y) ? -1 : (x > y); }

static double agg_min(const double *v, int n)
{ double m = v[0]; for (int i = 1; i < n; i++) if (v[i] < m) m = v[i]; return m; }

static double agg_med(const double *v, int n)
{
    double *t = (double*)malloc((size_t)n * sizeof(double));
    memcpy(t, v, (size_t)n * sizeof(double));
    qsort(t, (size_t)n, sizeof(double), cmp_dbl);
    double m = (n & 1) ? t[n/2] : 0.5*(t[n/2 - 1] + t[n/2]);
    free(t); return m;
}

static void fill_plane(double *p, size_t n2, unsigned seed)
{
    unsigned s = seed | 1u;
    for (size_t i = 0; i < n2; i++) { s = s*1664525u + 1013904223u; p[i] = (double)((int)(s>>8) % 2001 - 1000) * 1e-3; }
}

int main(int argc, char **argv)
{
    const size_t Ns[] = { 2048, 4096, 8192, 16384 };
    const int    nN   = (int)(sizeof(Ns)/sizeof(Ns[0]));
    int reps = (argc > 1) ? atoi(argv[1]) : 12;
    if (reps < (int)N_ARMS) reps = (int)N_ARMS;
    /* CRITICAL: reps MUST be a multiple of N_ARMS.  Otherwise the arms cover
     * DIFFERENT slot sets, and because best-of tends to select the same
     * (coolest) rep for every arm, the intra-rep thermal drift survives the
     * rotation and shows up as control-arm skew.  This is exactly the
     * "fixed-order A/B inflates the second arm" trap already on record. */
    reps = ((reps + (int)N_ARMS - 1) / (int)N_ARMS) * (int)N_ARMS;

    timer_init();
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_TIME_CRITICAL);
    DWORD_PTR prev = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4); /* logical core 2 */
    if (!prev) { fprintf(stderr, "FATAL: SetThreadAffinityMask(4) failed\n"); return 2; }
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    g_bust = (unsigned char*)_aligned_malloc(CACHEBUST_BYTES, 4096);
    if (!g_bust) { fprintf(stderr, "FATAL: cachebust alloc\n"); return 2; }
    memset(g_bust, 1, CACHEBUST_BYTES);

    printf("spike_pingpong_vs_tr4 — does a SECOND N-complex interior plane pay for\n");
    printf("itself by deleting the terminator's load-side corner-turn?\n");
    printf("core=2 (mask 4)  prio=HIGH/TIME_CRITICAL  reps=%d (= %d full arm rotations)  FTZ/DAZ=on\n",
           reps, reps / (int)N_ARMS);
    printf("headline = MIN over all reps (every arm visits every slot equally, so min is unbiased);\n");
    printf("           median printed to stderr as a cross-check\n");
    printf("port-5 per 32 complex:  A = 4xTR4(32) + 8xREINT(32) = 64   |   B = 8xREINT(32) = 32\n");
    printf("delta = 1.00 shuffle per complex; footprint A = 1 plane, B = 2 planes\n\n");

    printf("%6s %8s %9s | %10s %10s %10s | %9s %9s | %7s %7s | %7s %7s\n",
           "N", "plane", "passes", "A ns/pass", "Ba ns/pas", "C ns/pass",
           "Q2 shuf", "Q1 plane", "ctlA", "ctlB", "A/Ba", "verdict");
    printf("        A = 1 plane in-place + TR4   Ba = 2 planes alternating, no TR4   C = 1 plane in-place, no TR4\n");
    printf("        Q2 = A-C = what the corner-turn costs    Q1 = Ba-C = what the second plane costs\n");
    printf("------------------------------------------------------------------------------------------------------------------------\n");

    double sumA_over_B = 0; int nrow = 0;
    double worst_ctl = 1.0;

    for (int ni = 0; ni < nN; ni++) {
        const size_t N     = Ns[ni];
        const size_t count = N / 8;              /* radix-8 columns */
        const size_t n2    = 2 * N;              /* doubles per plane */
        const size_t pbytes= n2 * sizeof(double);
        /* passes chosen so each timed region streams ~16M complex (~ms scale) */
        size_t passes = (size_t)(1u << 25) / N;
        if (passes < 16) passes = 16;

        /* ONE arena; 64-byte skews so no two planes are 4KB-congruent */
        const size_t stride = pbytes + 4096;
        unsigned char *arena = (unsigned char*)_aligned_malloc(3*stride + 8192, 4096);
        if (!arena) { fprintf(stderr, "FATAL: arena alloc N=%zu\n", N); return 2; }
        double *P = (double*)(arena + 0*stride +  64);
        double *X = (double*)(arena + 1*stride + 128);
        double *Y = (double*)(arena + 2*stride + 192);
        /* pairwise byte offsets mod 4096: X-P=64, Y-X=64, Y-P=128  -> no 4KB alias */

        /* packed per-column w^1: at tw + 2k -> [c(k..k+3)][s(k..k+3)] */
        double *tw = (double*)_aligned_malloc((2*count + 16) * sizeof(double), 64);
        if (!tw) { fprintf(stderr, "FATAL: tw alloc\n"); return 2; }
        memset(tw, 0, (2*count + 16) * sizeof(double));
        for (size_t k = 0; k + 4 <= count; k += 4) {
            for (int j = 0; j < 4; j++) {
                double th = 6.283185307179586 * (double)(k + (size_t)j) / (double)N;
                tw[2*k + j]     = cos(th);
                tw[2*k + 4 + j] = sin(th);
            }
        }

        double *samp[N_ARMS];
        for (int a = 0; a < N_ARMS; a++) samp[a] = (double*)malloc((size_t)reps * sizeof(double));

        /* warm-up (untimed) */
        fill_plane(P, n2, 1); fill_plane(X, n2, 2); fill_plane(Y, n2, 3);
        pass_turn(P, P, tw, count);
        pass_noturn(X, Y, tw, count);

        for (int r = 0; r < reps; r++) {
            for (int slot = 0; slot < N_ARMS; slot++) {
                int a = (slot + r) % N_ARMS;     /* ROTATE arm order every rep */

                /* identical treatment for every arm: re-init, bust, pace */
                fill_plane(P, n2, 1); fill_plane(X, n2, 2); fill_plane(Y, n2, 3);
                cachebust();
                Sleep(3);

                double t0 = now_s();
                switch (a) {
                case ARM_A1: case ARM_A2:
                    for (size_t it = 0; it < passes; it++) pass_turn(P, P, tw, count); break;
                case ARM_AP:
                    for (size_t it = 0; it < passes; it++) pass_turn(X, X, tw, count); break;
                case ARM_B1: case ARM_B2:
                    for (size_t it = 0; it < passes; it++) pass_noturn(X, Y, tw, count); break;
                case ARM_BP:
                    for (size_t it = 0; it < passes; it++) pass_noturn(Y, P, tw, count); break;
                case ARM_BALT:
                    /* the REAL proposed dataflow: successive passes swap the
                     * planes, so BOTH carry dirty lines (writeback pressure the
                     * X->Y-only arm never sees). */
                    for (size_t it = 0; it < passes; it++) {
                        if (it & 1) pass_noturn(Y, X, tw, count);
                        else        pass_noturn(X, Y, tw, count);
                    }
                    break;
                case ARM_C:
                    for (size_t it = 0; it < passes; it++) pass_noturn(P, P, tw, count); break;
                }
                double t1 = now_s();

                /* consume results: blocks DCE and catches NaN/blowup */
                double chk = 0;
                for (size_t i = 0; i < n2; i += 97) chk += P[i] + X[i] + Y[i];
                g_sink += chk;
                if (!(chk == chk)) { fprintf(stderr, "FATAL: NaN in arm %d N=%zu\n", a, N); return 3; }

                samp[a][r] = (t1 - t0) * 1e9 / (double)passes;
            }
        }

        double mn[N_ARMS], md[N_ARMS];
        for (int a = 0; a < N_ARMS; a++) { mn[a] = agg_min(samp[a], reps); md[a] = agg_med(samp[a], reps); }

        /* Headline uses the MIN over a whole number of rotations.  Because reps
         * is a multiple of N_ARMS, every arm visits every slot the same number
         * of times, so the min is NOT slot-biased (that was the bug in the
         * first cut).  Min is also far cleaner than the median here: the
         * medians carry a heavy tail from intermittent OS/interrupt noise that
         * hits the larger-footprint arms harder, while the mins of the four
         * independent B variants agree to <0.5%.  Median is kept as a
         * cross-check and printed to stderr. */
        double A  = 0.5*(mn[ARM_A1] + mn[ARM_A2]);
        double B  = 0.5*(mn[ARM_B1] + mn[ARM_B2]);
        double Ba = mn[ARM_BALT];                   /* honest alternating ping-pong */
        double ctlA = mn[ARM_A2] / mn[ARM_A1];      /* identical code+data */
        double ctlB = mn[ARM_B2] / mn[ARM_B1];      /* identical code+data */
        double plA  = mn[ARM_AP] / A;               /* placement control */
        double plB  = mn[ARM_BP] / B;               /* placement control */
        double ratio = A / Ba;                      /* THE verdict ratio */
        double ratio_med = (0.5*(md[ARM_A1]+md[ARM_A2])) / md[ARM_BALT];

        double dev;
        dev = fabs(ctlA - 1.0); if (dev > worst_ctl - 1.0) worst_ctl = 1.0 + dev;
        dev = fabs(ctlB - 1.0); if (dev > worst_ctl - 1.0) worst_ctl = 1.0 + dev;

        double C  = mn[ARM_C];
        double Q2 = A  - C;   /* port-5 cost of 1 shuffle per complex */
        double Q1 = Ba - C;   /* memory cost of the second N-complex plane */

        printf("%6zu %7zuK %9zu | %10.1f %10.1f %10.1f | %9.1f %9.1f | %7.4f %7.4f | %7.4f %7s\n",
               N, pbytes >> 10, passes, A, Ba, C, Q2, Q1,
               ctlA, ctlB, ratio, (ratio > 1.0) ? "B wins" : "A wins");
        fprintf(stderr, "    [N=%5zu] DECOMPOSITION  C(1 plane,no turn)=%.1f  |  Q2 = A-C = %.1f ns (the shuffles)"
                        "   Q1 = Ba-C = %.1f ns (the 2nd plane)   Q2/Q1 = %.2fx\n",
                N, C, Q2, Q1, (Q1 > 1.0) ? Q2/Q1 : 0.0);
        fprintf(stderr, "    [N=%5zu] min    ns/pass  A1 %8.1f  A2 %8.1f  Ap %8.1f | B1 %8.1f  B2 %8.1f  Bp %8.1f  Ba %8.1f  C %8.1f   <- headline\n",
                N, mn[ARM_A1], mn[ARM_A2], mn[ARM_AP], mn[ARM_B1], mn[ARM_B2], mn[ARM_BP], mn[ARM_BALT], mn[ARM_C]);
        fprintf(stderr, "    [N=%5zu] median ns/pass  A1 %8.1f  A2 %8.1f  Ap %8.1f | B1 %8.1f  B2 %8.1f  Bp %8.1f  Ba %8.1f  C %8.1f   (A/Ba by median = %.4f)\n",
                N, md[ARM_A1], md[ARM_A2], md[ARM_AP], md[ARM_B1], md[ARM_B2], md[ARM_BP], md[ARM_BALT], md[ARM_C], ratio_med);
        fprintf(stderr, "    [N=%5zu] placement controls (must be ~1.00): Ap/A = %.4f   Bp/B = %.4f   |  Ba/B (dirty-plane tax) = %.4f\n",
                N, plA, plB, Ba / B);

        sumA_over_B += ratio; nrow++;
        for (int a = 0; a < N_ARMS; a++) free(samp[a]);
        _aligned_free(tw);
        _aligned_free(arena);
    }

    printf("\nworst control deviation from 1.00: %.4f  (%s)\n", worst_ctl,
           (fabs(worst_ctl - 1.0) <= 0.03) ? "OK — harness trustworthy"
                                           : "BROKEN — numbers are VOID");
    printf("mean A/B over N: %.4f   (>1 => the second plane pays for itself)\n",
           sumA_over_B / (double)nrow);
    printf("sink %.3e\n", g_sink);
    _aligned_free(g_bust);
    return 0;
}
