/* k1_fourstep_spike.c — K=1 single-transform bakeoff, Arm A (four-step over the
 * batch engine) vs BAILEY2 (fused four-step with a scalar column pass).
 *
 * The controlled A/B (docs/roadmap/row_major_engine.md §8):
 *   BAILEY2 stage 1 = R1 calls of n1_oop(R2) at count=1 (SCALAR columns, the
 *                     transpose fused into contiguous per-column stores);
 *   Arm A   stage 1 = ONE call of the SAME n1_oop(R2) at count=R1 (VECTORIZED
 *                     across columns — the batch identity: column c IS lane c),
 *                     then an explicit SIMD 4x4 transpose pass;
 *   stage 2 = IDENTICAL for both arms: the pair plan's own t1 codelet + Qr/Qi
 *             (per-lane path at K=1; Qr[(l2-1)*R2 + k2] = W_N^{l2.k2} is
 *             exactly the four-step diagonal).
 * The measured delta is therefore purely [vectorized leaf + lump-sum transpose]
 * vs [scalar leaf with fused transposing stores]. Natural order, OOP, split.
 *
 * Baselines: LEAF (production K=1 answer at N<=128, count=1 tail path) and the
 * scalar tier (in-place lane engine at K=1, estimate chain).
 *
 * Gates: fwd vs naive O(N^2) DFT in NATURAL order, deterministic AND random
 * inputs, tol 1e-9*N; Arm A vs BAILEY2 cross-diff reported informationally.
 *
 * Build: python build.py --src benches/k1_fourstep_spike.c --compile
 * Run:   k1_fourstep_spike [N ...]      (default: 64 256 1024 4096)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>

#include "executor.h"
#include "planner.h"
#include "oop_plan.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

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
    (void)a;
    free(j);
}

static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) exit(1);
    return p;
}
static void afree(double *p) { vfft_proto_aligned_free(p); }

/* d[t*R2 + m] = s[m*R1 + t]  (s: R2 rows x R1 cols, row stride R1). R1,R2 % 4 == 0. */
static void transpose_split(const double *s, double *d, int R2, int R1)
{
    for (int m = 0; m < R2; m += 4)
        for (int t = 0; t < R1; t += 4) {
            __m256d a = _mm256_loadu_pd(s + (size_t)(m + 0) * R1 + t);
            __m256d b = _mm256_loadu_pd(s + (size_t)(m + 1) * R1 + t);
            __m256d c = _mm256_loadu_pd(s + (size_t)(m + 2) * R1 + t);
            __m256d e = _mm256_loadu_pd(s + (size_t)(m + 3) * R1 + t);
            __m256d u0 = _mm256_unpacklo_pd(a, b), u1 = _mm256_unpackhi_pd(a, b);
            __m256d u2 = _mm256_unpacklo_pd(c, e), u3 = _mm256_unpackhi_pd(c, e);
            _mm256_storeu_pd(d + (size_t)(t + 0) * R2 + m, _mm256_permute2f128_pd(u0, u2, 0x20));
            _mm256_storeu_pd(d + (size_t)(t + 1) * R2 + m, _mm256_permute2f128_pd(u1, u3, 0x20));
            _mm256_storeu_pd(d + (size_t)(t + 2) * R2 + m, _mm256_permute2f128_pd(u0, u2, 0x31));
            _mm256_storeu_pd(d + (size_t)(t + 3) * R2 + m, _mm256_permute2f128_pd(u1, u3, 0x31));
        }
}

/* ---------------- IL-native arms (the layout-tax killer) ----------------
 * Column pass reads the user's INTERLEAVED z directly via the unwired
 * radixN_n1_oop_fwd_avx2_UG_UG_il_in codelets (il_derive.py twins of the split
 * leaves: same free-stride 11-arg ABI, in_z[2*(b*Gs + l*Ls)], deinterleave
 * in-register). Derived-tail caveat is moot here: me=R1 is always %4==0, only
 * the full-vector path runs. Output side: t1 has no il_out twin yet (that is
 * the scoped emitter job), so v1 pays ONE explicit interleave sweep d->z. */
#define K1_IL_IN_DECL(R) extern void radix##R##_n1_oop_fwd_avx2_UG_UG_il_in( \
    const double *, const double *, double *, double *,                       \
    const double *, const double *, size_t, size_t, size_t, size_t, size_t);
K1_IL_IN_DECL(4) K1_IL_IN_DECL(8) K1_IL_IN_DECL(16) K1_IL_IN_DECL(32) K1_IL_IN_DECL(64)

/* NATIVE-emitted t1_oop il_out twins (P2: the exit sweep, fused) */
#define K1_T1_ILOUT_DECL(R) extern void radix##R##_t1_oop_fwd_avx2_UG_UG_il_out( \
    const double *, const double *, double *, double *,                           \
    const double *, const double *, size_t, size_t, size_t, size_t, size_t);
K1_T1_ILOUT_DECL(4) K1_T1_ILOUT_DECL(8) K1_T1_ILOUT_DECL(16) K1_T1_ILOUT_DECL(32) K1_T1_ILOUT_DECL(64)

static vfft_oop11_fn k1_il_in_leaf(int R)
{
    switch (R) {
    case 4:  return radix4_n1_oop_fwd_avx2_UG_UG_il_in;
    case 8:  return radix8_n1_oop_fwd_avx2_UG_UG_il_in;
    case 16: return radix16_n1_oop_fwd_avx2_UG_UG_il_in;
    case 32: return radix32_n1_oop_fwd_avx2_UG_UG_il_in;
    case 64: return radix64_n1_oop_fwd_avx2_UG_UG_il_in;
    default: return 0;
    }
}

static vfft_oop11_fn k1_t1_ilout(int R)
{
    switch (R) {
    case 4:  return radix4_t1_oop_fwd_avx2_UG_UG_il_out;
    case 8:  return radix8_t1_oop_fwd_avx2_UG_UG_il_out;
    case 16: return radix16_t1_oop_fwd_avx2_UG_UG_il_out;
    case 32: return radix32_t1_oop_fwd_avx2_UG_UG_il_out;
    case 64: return radix64_t1_oop_fwd_avx2_UG_UG_il_out;
    default: return 0;
    }
}

/* z[2k]=re[k], z[2k+1]=im[k] — the v1 exit sweep (removed by the t1_oop_ilout twin) */
static void interleave_store(const double *re, const double *im, double *z, int N)
{
    for (int k = 0; k < N; k += 4) {
        __m256d r = _mm256_loadu_pd(re + k), q = _mm256_loadu_pd(im + k);
        __m256d lo = _mm256_unpacklo_pd(r, q), hi = _mm256_unpackhi_pd(r, q);
        _mm256_storeu_pd(z + 2 * k,     _mm256_permute2f128_pd(lo, hi, 0x20));
        _mm256_storeu_pd(z + 2 * k + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
    }
}

/* ---------------- Arm B: fused mono, N=64 = 8x8 four-step ENTIRELY in registers
 * (the MKL small-N method, split layout so zero re/im shuffle tax). Per t-half:
 * 8 vector loads -> DFT8 across j -> four-step twiddle cmul; then 4x4 register
 * transposes (the "in-register stage reshaping" MKL smears per stage); then DFT8
 * across t per m-half with contiguous natural-order stores. ---------------- */

/* 8-pt DFT across the ARRAY index of re[8]/im[8], natural in/out, in place. */
static inline void dft8v(__m256d *re, __m256d *im)
{
    const __m256d C = _mm256_set1_pd(0.70710678118654752440);
    const __m256d CN = _mm256_set1_pd(-0.70710678118654752440);
    __m256d t0r = _mm256_add_pd(re[0], re[4]), t0i = _mm256_add_pd(im[0], im[4]);
    __m256d t1r = _mm256_sub_pd(re[0], re[4]), t1i = _mm256_sub_pd(im[0], im[4]);
    __m256d t2r = _mm256_add_pd(re[2], re[6]), t2i = _mm256_add_pd(im[2], im[6]);
    __m256d t3r = _mm256_sub_pd(re[2], re[6]), t3i = _mm256_sub_pd(im[2], im[6]);
    __m256d E0r = _mm256_add_pd(t0r, t2r), E0i = _mm256_add_pd(t0i, t2i);
    __m256d E2r = _mm256_sub_pd(t0r, t2r), E2i = _mm256_sub_pd(t0i, t2i);
    __m256d E1r = _mm256_add_pd(t1r, t3i), E1i = _mm256_sub_pd(t1i, t3r); /* t1 - i t3 */
    __m256d E3r = _mm256_sub_pd(t1r, t3i), E3i = _mm256_add_pd(t1i, t3r); /* t1 + i t3 */
    __m256d s0r = _mm256_add_pd(re[1], re[5]), s0i = _mm256_add_pd(im[1], im[5]);
    __m256d s1r = _mm256_sub_pd(re[1], re[5]), s1i = _mm256_sub_pd(im[1], im[5]);
    __m256d s2r = _mm256_add_pd(re[3], re[7]), s2i = _mm256_add_pd(im[3], im[7]);
    __m256d s3r = _mm256_sub_pd(re[3], re[7]), s3i = _mm256_sub_pd(im[3], im[7]);
    __m256d O0r = _mm256_add_pd(s0r, s2r), O0i = _mm256_add_pd(s0i, s2i);
    __m256d O2r = _mm256_sub_pd(s0r, s2r), O2i = _mm256_sub_pd(s0i, s2i);
    __m256d O1r = _mm256_add_pd(s1r, s3i), O1i = _mm256_sub_pd(s1i, s3r);
    __m256d O3r = _mm256_sub_pd(s1r, s3i), O3i = _mm256_add_pd(s1i, s3r);
    /* W8^k O[k]: k=1: c(Or+Oi) + i c(Oi-Or); k=2 (=-i): (Oi, -Or); k=3: c(Oi-Or) - i c(Or+Oi) */
    __m256d W1r = _mm256_mul_pd(C, _mm256_add_pd(O1r, O1i));
    __m256d W1i = _mm256_mul_pd(C, _mm256_sub_pd(O1i, O1r));
    __m256d W2r = O2i, W2i = _mm256_sub_pd(_mm256_setzero_pd(), O2r);
    __m256d W3r = _mm256_mul_pd(C, _mm256_sub_pd(O3i, O3r));
    __m256d W3i = _mm256_mul_pd(CN, _mm256_add_pd(O3r, O3i));
    re[0] = _mm256_add_pd(E0r, O0r); im[0] = _mm256_add_pd(E0i, O0i);
    re[4] = _mm256_sub_pd(E0r, O0r); im[4] = _mm256_sub_pd(E0i, O0i);
    re[1] = _mm256_add_pd(E1r, W1r); im[1] = _mm256_add_pd(E1i, W1i);
    re[5] = _mm256_sub_pd(E1r, W1r); im[5] = _mm256_sub_pd(E1i, W1i);
    re[2] = _mm256_add_pd(E2r, W2r); im[2] = _mm256_add_pd(E2i, W2i);
    re[6] = _mm256_sub_pd(E2r, W2r); im[6] = _mm256_sub_pd(E2i, W2i);
    re[3] = _mm256_add_pd(E3r, W3r); im[3] = _mm256_add_pd(E3i, W3i);
    re[7] = _mm256_sub_pd(E3r, W3r); im[7] = _mm256_sub_pd(E3i, W3i);
}

#define T4(r0, r1, r2, r3, o0, o1, o2, o3) do {                          \
    __m256d _u0 = _mm256_unpacklo_pd(r0, r1), _u1 = _mm256_unpackhi_pd(r0, r1); \
    __m256d _u2 = _mm256_unpacklo_pd(r2, r3), _u3 = _mm256_unpackhi_pd(r2, r3); \
    o0 = _mm256_permute2f128_pd(_u0, _u2, 0x20);                          \
    o1 = _mm256_permute2f128_pd(_u1, _u3, 0x20);                          \
    o2 = _mm256_permute2f128_pd(_u0, _u2, 0x31);                          \
    o3 = _mm256_permute2f128_pd(_u1, _u3, 0x31);                          \
} while (0)

static double K1M64_TWR[2][8][4], K1M64_TWI[2][8][4]; /* [t-half][m][lane j]: W_64^{m(4h+j)} */
static void k1m64_init(void)
{
    for (int h = 0; h < 2; h++)
        for (int m = 0; m < 8; m++)
            for (int j = 0; j < 4; j++) {
                double a = -2.0 * M_PI * (double)(m * (4 * h + j)) / 64.0;
                K1M64_TWR[h][m][j] = cos(a);
                K1M64_TWI[h][m][j] = sin(a);
            }
}

static void mono64_fwd(const double *xr, const double *xi, double *dr, double *di)
{
    __m256d Ur[2][8], Ui[2][8], Vr[2][8], Vi[2][8];
    for (int h = 0; h < 2; h++) {
        __m256d r[8], q[8];
        for (int j = 0; j < 8; j++) {
            r[j] = _mm256_loadu_pd(xr + (size_t)j * 8 + 4 * h);
            q[j] = _mm256_loadu_pd(xi + (size_t)j * 8 + 4 * h);
        }
        dft8v(r, q);                                  /* Y_t[m], lanes t = 4h..4h+3 */
        for (int m = 1; m < 8; m++) {                 /* four-step diagonal W_64^{mt} */
            __m256d cr = _mm256_loadu_pd(K1M64_TWR[h][m]);
            __m256d ci = _mm256_loadu_pd(K1M64_TWI[h][m]);
            __m256d zr = _mm256_sub_pd(_mm256_mul_pd(r[m], cr), _mm256_mul_pd(q[m], ci));
            __m256d zi = _mm256_add_pd(_mm256_mul_pd(r[m], ci), _mm256_mul_pd(q[m], cr));
            r[m] = zr; q[m] = zi;
        }
        for (int m = 0; m < 8; m++) { Ur[h][m] = r[m]; Ui[h][m] = q[m]; }
    }
    for (int h = 0; h < 2; h++)
        for (int mh = 0; mh < 2; mh++) {
            T4(Ur[h][4 * mh + 0], Ur[h][4 * mh + 1], Ur[h][4 * mh + 2], Ur[h][4 * mh + 3],
               Vr[mh][4 * h + 0], Vr[mh][4 * h + 1], Vr[mh][4 * h + 2], Vr[mh][4 * h + 3]);
            T4(Ui[h][4 * mh + 0], Ui[h][4 * mh + 1], Ui[h][4 * mh + 2], Ui[h][4 * mh + 3],
               Vi[mh][4 * h + 0], Vi[mh][4 * h + 1], Vi[mh][4 * h + 2], Vi[mh][4 * h + 3]);
        }
    for (int mh = 0; mh < 2; mh++) {                  /* X[m+8j'] = sum_t Z_t[m] W_8^{tj'} */
        dft8v(Vr[mh], Vi[mh]);
        for (int jp = 0; jp < 8; jp++) {
            _mm256_storeu_pd(dr + (size_t)jp * 8 + 4 * mh, Vr[mh][jp]);
            _mm256_storeu_pd(di + (size_t)jp * 8 + 4 * mh, Vi[mh][jp]);
        }
    }
}

/* mono-64 on an interleaved z buffer: same compute, deinterleave folded into the
 * loads (unpack + permute4x64 0xD8, the il codelet lattice) and interleave folded
 * into the stores. Boundary shuffles only — the cmul stays split/shuffle-free,
 * unlike MKL's IL mono which pays the shuffle inside every complex multiply. */
static void mono64_il_fwd(const double *z_in, double *z_out)
{
    __m256d Ur[2][8], Ui[2][8], Vr[2][8], Vi[2][8];
    for (int h = 0; h < 2; h++) {
        __m256d r[8], q[8];
        for (int j = 0; j < 8; j++) {
            __m256d za = _mm256_loadu_pd(z_in + 2 * ((size_t)j * 8 + 4 * h));
            __m256d zb = _mm256_loadu_pd(z_in + 2 * ((size_t)j * 8 + 4 * h) + 4);
            r[j] = _mm256_permute4x64_pd(_mm256_unpacklo_pd(za, zb), 0xD8);
            q[j] = _mm256_permute4x64_pd(_mm256_unpackhi_pd(za, zb), 0xD8);
        }
        dft8v(r, q);
        for (int m = 1; m < 8; m++) {
            __m256d cr = _mm256_loadu_pd(K1M64_TWR[h][m]);
            __m256d ci = _mm256_loadu_pd(K1M64_TWI[h][m]);
            __m256d zr = _mm256_sub_pd(_mm256_mul_pd(r[m], cr), _mm256_mul_pd(q[m], ci));
            __m256d zi = _mm256_add_pd(_mm256_mul_pd(r[m], ci), _mm256_mul_pd(q[m], cr));
            r[m] = zr; q[m] = zi;
        }
        for (int m = 0; m < 8; m++) { Ur[h][m] = r[m]; Ui[h][m] = q[m]; }
    }
    for (int h = 0; h < 2; h++)
        for (int mh = 0; mh < 2; mh++) {
            T4(Ur[h][4 * mh + 0], Ur[h][4 * mh + 1], Ur[h][4 * mh + 2], Ur[h][4 * mh + 3],
               Vr[mh][4 * h + 0], Vr[mh][4 * h + 1], Vr[mh][4 * h + 2], Vr[mh][4 * h + 3]);
            T4(Ui[h][4 * mh + 0], Ui[h][4 * mh + 1], Ui[h][4 * mh + 2], Ui[h][4 * mh + 3],
               Vi[mh][4 * h + 0], Vi[mh][4 * h + 1], Vi[mh][4 * h + 2], Vi[mh][4 * h + 3]);
        }
    for (int mh = 0; mh < 2; mh++) {
        dft8v(Vr[mh], Vi[mh]);
        for (int jp = 0; jp < 8; jp++) {
            __m256d lo = _mm256_unpacklo_pd(Vr[mh][jp], Vi[mh][jp]);
            __m256d hi = _mm256_unpackhi_pd(Vr[mh][jp], Vi[mh][jp]);
            _mm256_storeu_pd(z_out + 2 * ((size_t)jp * 8 + 4 * mh),
                             _mm256_permute2f128_pd(lo, hi, 0x20));
            _mm256_storeu_pd(z_out + 2 * ((size_t)jp * 8 + 4 * mh) + 4,
                             _mm256_permute2f128_pd(lo, hi, 0x31));
        }
    }
}

enum { W_BAILEY = 0, W_ARMA, W_ARMA_IP, W_ARMA_IL, W_ARMA_IL2, W_LEAFPASS, W_TRANSPOSE,
       W_T1, W_SCALAR, W_LEAFN, W_MONO64, W_MONO64_IL };

typedef struct {
    vfft_oop_plan_t *p;      /* forced BAILEY2 pair (leaf, t1p, Qr/Qi, R1, R2) */
    vfft_oop11_fn leafN;     /* whole-N leaf, N<=128, else NULL */
    vfft_oop11_fn il_leaf;   /* il_in column-pass leaf for the current pair (R2 pow2<=64) */
    vfft_oop11_fn t1_il;     /* emitted t1_oop il_out twin for the current pair (R1 pow2<=64) */
    stride_plan_t *sp;       /* scalar tier (in-place lane engine, K=1) */
    int N;
    const double *xr, *xi;   /* input (const during OOP arms) */
    double *tr_, *ti_;       /* Arm A column-pass scratch */
    double *dr, *di;         /* destination */
    double *wr, *wi;         /* scalar-tier in-place work */
    double *z;               /* interleaved buffer (2N), in-place for IL arms */
} ctx_t;

static void run(ctx_t *c, int what)
{
    switch (what) {
    case W_BAILEY:
        vfft_oop_execute_fwd(c->p, c->xr, c->xi, c->dr, c->di);
        break;
    case W_ARMA: {
        const size_t R1 = (size_t)c->p->R1, R2 = (size_t)c->p->R2;
        c->p->leaf(c->xr, c->xi, c->tr_, c->ti_, 0, 0, R1, 1, R1, 1, R1);
        transpose_split(c->tr_, c->dr, (int)R2, (int)R1);
        transpose_split(c->ti_, c->di, (int)R2, (int)R1);
        c->p->t1p(c->dr, c->di, c->dr, c->di, c->p->Qr, c->p->Qi, R2, 1, R2, 1, R2);
        break; }
    case W_ARMA_IP: {
        /* in-place 2-buffer variant: user buffer wr/wi + scratch tr_/ti_ only
         * (matches MKL's DFTI_INPLACE config; halves the working set vs OOP) */
        const size_t R1 = (size_t)c->p->R1, R2 = (size_t)c->p->R2;
        c->p->leaf(c->wr, c->wi, c->tr_, c->ti_, 0, 0, R1, 1, R1, 1, R1);
        transpose_split(c->tr_, c->wr, (int)R2, (int)R1);
        transpose_split(c->ti_, c->wi, (int)R2, (int)R1);
        c->p->t1p(c->wr, c->wi, c->wr, c->wi, c->p->Qr, c->p->Qi, R2, 1, R2, 1, R2);
        break; }
    case W_ARMA_IL: {
        /* z (interleaved, in-place) -> il_in leaf -> split scratch -> transpose ->
         * t1 split -> interleave sweep -> z. Buffers: z + 2 split scratch pairs. */
        const size_t R1 = (size_t)c->p->R1, R2 = (size_t)c->p->R2;
        c->il_leaf(c->z, 0, c->tr_, c->ti_, 0, 0, R1, 1, R1, 1, R1);
        transpose_split(c->tr_, c->dr, (int)R2, (int)R1);
        transpose_split(c->ti_, c->di, (int)R2, (int)R1);
        c->p->t1p(c->dr, c->di, c->dr, c->di, c->p->Qr, c->p->Qi, R2, 1, R2, 1, R2);
        interleave_store(c->dr, c->di, c->z, c->N);
        break; }
    case W_ARMA_IL2: {
        /* v2: 3 passes, NO exit sweep — the emitted t1_oop il_out twin reads
         * split d and stores interleaved z directly. z -> z in place. */
        const size_t R1 = (size_t)c->p->R1, R2 = (size_t)c->p->R2;
        c->il_leaf(c->z, 0, c->tr_, c->ti_, 0, 0, R1, 1, R1, 1, R1);
        transpose_split(c->tr_, c->dr, (int)R2, (int)R1);
        transpose_split(c->ti_, c->di, (int)R2, (int)R1);
        c->t1_il(c->dr, c->di, c->z, 0, c->p->Qr, c->p->Qi, R2, 1, R2, 1, R2);
        break; }
    case W_MONO64_IL:
        mono64_il_fwd(c->z, c->z);
        break;
    case W_LEAFPASS: {
        const size_t R1 = (size_t)c->p->R1;
        c->p->leaf(c->xr, c->xi, c->tr_, c->ti_, 0, 0, R1, 1, R1, 1, R1);
        break; }
    case W_TRANSPOSE:
        transpose_split(c->tr_, c->dr, c->p->R2, c->p->R1);
        transpose_split(c->ti_, c->di, c->p->R2, c->p->R1);
        break;
    case W_T1: {
        const size_t R2 = (size_t)c->p->R2;
        c->p->t1p(c->dr, c->di, c->dr, c->di, c->p->Qr, c->p->Qi, R2, 1, R2, 1, R2);
        break; }
    case W_SCALAR:
        vfft_proto_execute_fwd(c->sp, c->wr, c->wi, 1);
        break;
    case W_LEAFN:
        c->leafN(c->xr, c->xi, c->dr, c->di, 0, 0, 1, 1, 1, 1, 1);
        break;
    case W_MONO64:
        mono64_fwd(c->xr, c->xi, c->dr, c->di);
        break;
    }
}

static double bench(ctx_t *c, int what)
{
    int reps = (int)(2e6 / (double)c->N);
    if (reps < 100) reps = 100;
    if (reps > 400000) reps = 400000;
    for (int w = 0; w < 10; w++) run(c, what);
    double best = 1e18;
    for (int t = 0; t < 5; t++) {
        if (t) cachebust();
        double t0 = now_ms();
        for (int i = 0; i < reps; i++) run(c, what);
        double ns = (now_ms() - t0) * 1e6 / reps;
        if (ns < best) best = ns;
    }
    return best;
}

/* naive O(N^2) DFT, natural order */
static void naive_dft(int N, const double *xr, const double *xi, double *Xr, double *Xi)
{
    for (int k = 0; k < N; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double ang = -2.0 * M_PI * (double)((long)n * k % N) / (double)N;
            double cc = cos(ang), ss = sin(ang);
            sr += xr[n] * cc - xi[n] * ss;
            si += xr[n] * ss + xi[n] * cc;
        }
        Xr[k] = sr; Xi[k] = si;
    }
}

static double max_err(int N, const double *ar, const double *ai,
                      const double *br, const double *bi)
{
    double m = 0;
    for (int k = 0; k < N; k++) {
        double er = fabs(ar[k] - br[k]), ei = fabs(ai[k] - bi[k]);
        if (er > m) m = er;
        if (ei > m) m = ei;
    }
    return m;
}

static double max_err_il(int N, const double *z, const double *br, const double *bi)
{
    double m = 0;
    for (int k = 0; k < N; k++) {
        double er = fabs(z[2 * k] - br[k]), ei = fabs(z[2 * k + 1] - bi[k]);
        if (er > m) m = er;
        if (ei > m) m = ei;
    }
    return m;
}

static void fill_il(double *z, const double *re, const double *im, int N)
{
    for (int k = 0; k < N; k++) { z[2 * k] = re[k]; z[2 * k + 1] = im[k]; }
}

int main(int argc, char **argv)
{
    setvbuf(stdout, NULL, _IONBF, 0);
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4); /* P-core 2, ST convention */
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);

    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    k1m64_init();

    int Ns_default[] = { 64, 256, 1024, 4096 };
    int nN = argc > 1 ? argc - 1 : 4;

    printf("# K=1 four-step bakeoff, Arm A vs BAILEY2 (same leaf/t1/Qr-Qi; delta = stage-1 vectorization + explicit transpose)\n");
    printf("# pinned core 2, HIGH prio, best-of-5, 32MB cachebust between trials\n");

    for (int ni = 0; ni < nN; ni++) {
        int N = argc > 1 ? atoi(argv[ni + 1]) : Ns_default[ni];
        printf("\n=== N=%d ===\n", N);

        double *xr = ad(N), *xi = ad(N);
        double *nr = ad(N), *nsi = ad(N);        /* naive reference */
        double *tr_ = ad(N), *ti_ = ad(N);
        double *dr = ad(N), *di = ad(N);
        double *wr = ad(N), *wi = ad(N);
        double *z = ad((size_t)2 * N);

        /* enumerate valid pairs */
        int pR1[16], pR2[16], nP = 0;
        for (int R2 = (N < 128 ? N : 128); R2 >= 4; R2--) {
            if (N % R2) continue;
            int R1 = N / R2;
            if (R1 < 4 || R1 > 128) continue;
            if (R1 % 4 || R2 % 4) continue;                 /* transpose blocks */
            if (!vfft_oop_leaf_fn(R2) || !vfft_oop_t1_fn(R1)) continue;
            if (nP < 16) { pR1[nP] = R1; pR2[nP] = R2; nP++; }
        }

        vfft_oop_plan_t *plans[16] = { 0 };
        for (int i = 0; i < nP; i++)
            plans[i] = vfft_oop_plan_create_pair_v(N, 1, pR1[i], pR2[i], /*flat*/0);

        ctx_t c; memset(&c, 0, sizeof c);
        c.N = N; c.xr = xr; c.xi = xi;
        c.tr_ = tr_; c.ti_ = ti_; c.dr = dr; c.di = di; c.wr = wr; c.wi = wi; c.z = z;
        c.leafN = (N <= 128) ? vfft_oop_leaf_fn(N) : NULL;

        /* ---- gates: det then rand ---- */
        double tol = 1e-9 * (double)N;
        int fails = 0;
        for (int inp = 0; inp < 2; inp++) {
            if (inp == 0) {
                for (int n = 0; n < N; n++) {
                    xr[n] = cos(0.7 * n) + 0.1;
                    xi[n] = sin(1.3 * n) - 0.05;
                }
            } else {
                srand(12345 + N);
                for (int n = 0; n < N; n++) {
                    xr[n] = (double)rand() / RAND_MAX - 0.5;
                    xi[n] = (double)rand() / RAND_MAX - 0.5;
                }
            }
            naive_dft(N, xr, xi, nr, nsi);
            for (int i = 0; i < nP; i++) {
                if (!plans[i]) continue;
                c.p = plans[i];
                run(&c, W_BAILEY);
                double eb = max_err(N, dr, di, nr, nsi);
                double *b2r = ad(N), *b2i = ad(N);
                memcpy(b2r, dr, (size_t)N * 8); memcpy(b2i, di, (size_t)N * 8);
                run(&c, W_ARMA);
                double ea = max_err(N, dr, di, nr, nsi);
                double xd = max_err(N, dr, di, b2r, b2i);
                afree(b2r); afree(b2i);
                memcpy(wr, xr, (size_t)N * 8); memcpy(wi, xi, (size_t)N * 8);
                run(&c, W_ARMA_IP);
                double ei = max_err(N, wr, wi, nr, nsi);
                double eil = -1.0, eil2 = -1.0;
                c.il_leaf = k1_il_in_leaf(pR2[i]);
                c.t1_il = k1_t1_ilout(pR1[i]);
                if (c.il_leaf) {
                    fill_il(z, xr, xi, N);
                    run(&c, W_ARMA_IL);
                    eil = max_err_il(N, z, nr, nsi);
                    if (c.t1_il) {
                        fill_il(z, xr, xi, N);
                        run(&c, W_ARMA_IL2);
                        eil2 = max_err_il(N, z, nr, nsi);
                    }
                }
                const char *bad = (eb > tol || ea > tol || ei > tol ||
                                   (c.il_leaf && eil > tol) ||
                                   (c.il_leaf && c.t1_il && eil2 > tol) ||
                                   eb != eb || ea != ea || ei != ei ||
                                   eil != eil || eil2 != eil2) ? "  <FAIL>" : "";
                if (bad[0]) fails++;
                printf("  gate %-3s %3dx%-3d  B2=%.2e  A=%.2e  Aip=%.2e  Ail=%.2e  Ail2=%.2e  cross=%.2e%s\n",
                       inp ? "rnd" : "det", pR1[i], pR2[i], eb, ea, ei, eil, eil2, xd, bad);
            }
            if (c.leafN) {
                run(&c, W_LEAFN);
                double el = max_err(N, dr, di, nr, nsi);
                const char *bad = (el > tol || el != el) ? "  <FAIL>" : "";
                if (bad[0]) fails++;
                printf("  gate %-3s LEAF(%d)  %.2e%s\n", inp ? "rnd" : "det", N, el, bad);
            }
            if (N == 64) {
                run(&c, W_MONO64);
                double em = max_err(N, dr, di, nr, nsi);
                fill_il(z, xr, xi, N);
                run(&c, W_MONO64_IL);
                double emi = max_err_il(N, z, nr, nsi);
                const char *bad = (em > tol || em != em || emi > tol || emi != emi) ? "  <FAIL>" : "";
                if (bad[0]) fails++;
                printf("  gate %-3s MONO64    %.2e  IL=%.2e%s\n", inp ? "rnd" : "det", em, emi, bad);
            }
        }
        if (fails) { printf("  GATE FAILURES (%d) — timings skipped\n", fails); continue; }

        /* ---- timings (rand input persists from gate loop) ---- */
        printf("  %-9s %10s %10s %10s %10s %10s %7s   %s\n", "pair", "B2(ns)", "A(ns)", "Aip(ns)", "Ail(ns)", "Ail2(ns)", "A/B2", "A parts: leaf/tr/t1 (ns)");
        double bestB = 1e18, bestA = 1e18, bestI = 1e18, bestL = 1e18, bestL2 = 1e18;
        int bestBp = -1, bestAp = -1, bestIp = -1, bestLp = -1, bestL2p = -1;
        for (int i = 0; i < nP; i++) {
            if (!plans[i]) continue;
            c.p = plans[i];
            double tb = bench(&c, W_BAILEY);
            double ta = bench(&c, W_ARMA);
            memcpy(wr, xr, (size_t)N * 8); memcpy(wi, xi, (size_t)N * 8);
            double ti = bench(&c, W_ARMA_IP);
            double til = -1, til2 = -1;
            c.il_leaf = k1_il_in_leaf(pR2[i]);
            c.t1_il = k1_t1_ilout(pR1[i]);
            if (c.il_leaf) { fill_il(z, xr, xi, N); til = bench(&c, W_ARMA_IL); }
            if (c.il_leaf && c.t1_il) { fill_il(z, xr, xi, N); til2 = bench(&c, W_ARMA_IL2); }
            double tl = bench(&c, W_LEAFPASS);
            double tt = bench(&c, W_TRANSPOSE);
            double t1 = bench(&c, W_T1);
            if (tb < bestB) { bestB = tb; bestBp = i; }
            if (ta < bestA) { bestA = ta; bestAp = i; }
            if (ti < bestI) { bestI = ti; bestIp = i; }
            if (til > 0 && til < bestL) { bestL = til; bestLp = i; }
            if (til2 > 0 && til2 < bestL2) { bestL2 = til2; bestL2p = i; }
            printf("  %3dx%-5d %10.0f %10.0f %10.0f %10.0f %10.0f %7.2f   %0.f / %0.f / %0.f\n",
                   pR1[i], pR2[i], tb, ta, ti, til, til2, ta / tb, tl, tt, t1);
        }
        if (bestIp >= 0)
            printf("  BEST-IP: %dx%d = %.0f ns (in-place 2-buffer)\n", pR1[bestIp], pR2[bestIp], bestI);
        if (bestLp >= 0)
            printf("  BEST-IL: %dx%d = %.0f ns (z->z native interleaved, v1 with exit sweep)\n",
                   pR1[bestLp], pR2[bestLp], bestL);
        if (bestL2p >= 0)
            printf("  BEST-IL2: %dx%d = %.0f ns (z->z, EMITTED t1_oop il_out, 3 passes, no sweep)\n",
                   pR1[bestL2p], pR2[bestL2p], bestL2);
        if (c.leafN)
            printf("  LEAF(%d) : %10.0f ns   (production K=1 route at N<=128)\n", N, bench(&c, W_LEAFN));
        if (N == 64) {
            printf("  MONO64   : %10.0f ns   (Arm B: fused in-register 8x8, MKL method)\n", bench(&c, W_MONO64));
            fill_il(z, xr, xi, N);
            printf("  MONO64-IL: %10.0f ns   (same, z->z interleaved, boundary shuffles only)\n", bench(&c, W_MONO64_IL));
        }
        c.sp = vfft_proto_estimate_plan(N, 1, &reg);
        if (c.sp) {
            memcpy(wr, xr, (size_t)N * 8); memcpy(wi, xi, (size_t)N * 8);
            printf("  scalar-tier(est,K=1,in-place): %10.0f ns\n", bench(&c, W_SCALAR));
            vfft_proto_plan_destroy(c.sp); c.sp = NULL;
        }
        if (bestBp >= 0 && bestAp >= 0)
            printf("  BEST: B2 %dx%d=%.0fns   A %dx%d=%.0fns   A/B2=%.2f\n",
                   pR1[bestBp], pR2[bestBp], bestB, pR1[bestAp], pR2[bestAp], bestA, bestA / bestB);

        for (int i = 0; i < nP; i++) vfft_oop_plan_destroy(plans[i]);
        afree(xr); afree(xi); afree(nr); afree(nsi);
        afree(tr_); afree(ti_); afree(dr); afree(di); afree(wr); afree(wi); afree(z);
    }
    printf("\nDONE\n");
    return 0;
}
