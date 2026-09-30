/* zfsr.h - the real FOUR-STEP: the 1D real transform above ZTT-r's band on
 * the 1D c2c four-step's recipe (il/rank1/k1_fourstep.h), the Hermitian fold
 * acting on the mirror pairs through the Bailey plane.
 *
 * x[N] is read as z[M], M = N/2, and the c2c four-step runs on it as it runs
 * for a 1D c2c request: the signal as N1 segments of N2 (M = N1 x N2), the
 * column chain across the segments, the inter-pass twiddle, the K=1 door's
 * plan on every segment (ZTURN-T at 2048 / 4096). Its SCRAMBLED class leaves
 * frequency k1(p) + N1*k2 at position k2 of segment p. The fold pairs bin f
 * with M - f, and there the pair is a SEGMENT PAIR at mirrored positions:
 *     (k1, k2)  <->  (N1 - k1, N2 - 1 - k2)      k1 > 0
 *     (0,  k2)  <->  (0,       N2 - k2)          the self-paired segment
 * so the fold is fused into the four-step's own order pass. r2c ends in ONE
 * sweep that reads a 16 x 16 block and its mirror block, folds, and writes
 * X[f] and X[M - f] as whole aligned runs in natural order; c2r starts with
 * the mirror sweep (runs of X in, the backward fold, both blocks into the
 * plane) and the four-step's last backward column stage writes the caller's
 * real plane. No fold pass, no natural transpose, no copy back: zr2c above
 * the ZTT band pays all three around the same child.
 *
 * THE SWEEP'S TWO LAWS (measured at 2^20, gauntlet/fs_split_time.c: the c2c
 * order pass 361 us, zr2c's fold ~450; a first sweep with unaligned mirror
 * runs and the full pair tables cost ~1000):
 *   every output run leaves as whole aligned lines, streamed. X[f] for
 *     k1 = k1b..k1b+15 is an aligned run; its mirrors are the 15 upper
 *     entries of the aligned run [M-f0-16, M-f0-1], whose first entry is the
 *     mirror of k1b+16 -- so each block folds a SEVENTEENTH lane for that one
 *     entry (1/16 of the fold redone) and no store is ever split or read
 *     for ownership;
 *   the pair twiddle is factored, w^f = w^k1 * w^(N1*k2), from two small
 *     tables (N1 + 1 and N2/2 + 1 entries, once-rounded): the full pair
 *     tables are as large as the data and were two more memory streams.
 *
 * The plane is the plan's own (M complex), so both placements are the same
 * pipeline: the caller's buffer is fully read before it is written.
 *
 * The plan input is the split (N1, N2): the real door sweeps the splits of
 * M at create, races the best against the other engines and banks
 * eng=zfsr split=N1xN2 (il/real/zrp_build.h, wisdom2_real_il.h).
 */
#ifndef VFFT_ZFSR_H
#define VFFT_ZFSR_H

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>

#include "k1_fourstep.h" /* the c2c four-step: plan, create, the scrambled class's execute */
#include "tw_exact.h"    /* once-rounded cos/sin(2*pi*e/n) */

#define VFFT_ZFSR_TB 16  /* the sweep's block: 16 x 16 complexes and its mirror block */

/* Defined in vfft.c with external linkage (the engagement counters' rule). */
extern long _vfft_zfsr_mt_count;

typedef struct vfft_zfsr_s
{
    int N, M, N1, N2;
    vfft_k1fs_plan_t *fs; /* the c2c four-step at M, SCRAMBLED class, out of place (owned) */
    double *plane;        /* M complex: the four-step's plane (64-B aligned, owned) */
    double *V;            /* (cos, sin)(2*pi*k1/N), k1 = 0..N1 (owned) */
    double *U;            /* (cos, sin)(2*pi*N1*k2/N), k2 = 0..N2/2 (owned) */
} vfft_zfsr_plan_t;

static void vfft_zfsr_destroy(vfft_zfsr_plan_t *p)
{
    if (!p) return;
    vfft_k1fs_destroy(p->fs);
    vfft_aligned_free(p->plane);
    vfft_aligned_free(p->V);
    vfft_aligned_free(p->U);
    free(p);
}

/* does N have a real four-step at all: N/2 in the c2c four-step's band */
static inline int vfft_zfsr_band(int N)
{
    return N >= 4 && (N & 1) == 0 && vfft_k1fs_band(N / 2);
}

static vfft_zfsr_plan_t *vfft_zfsr_create(int N, int N1, int N2, struct vfft_wisdom_s *W,
                                          const vfft_config_t *cfg, int nthreads)
{
    vfft_zfsr_plan_t *p;
    const int M = N / 2;
    int k;
    if (!vfft_zfsr_band(N) || (long)N1 * (long)N2 != (long)M) return NULL;
    if (N1 % VFFT_ZFSR_TB || N2 % (2 * VFFT_ZFSR_TB)) return NULL;
    p = (vfft_zfsr_plan_t *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->N = N; p->M = M; p->N1 = N1; p->N2 = N2;
    p->fs = vfft_k1fs_create(M, N1, N2, /*scr=*/1, W, cfg, /*inplace=*/0, nthreads, 0, NULL, 0);
    p->plane = (double *)vfft_aligned_alloc((2 * (size_t)M + 8) * sizeof(double));
    p->V = (double *)vfft_aligned_alloc(2 * ((size_t)N1 + 2) * sizeof(double));
    p->U = (double *)vfft_aligned_alloc(2 * ((size_t)N2 / 2 + 2) * sizeof(double));
    if (!p->fs || !p->plane || !p->V || !p->U) { vfft_zfsr_destroy(p); return NULL; }
    for (k = 0; k <= N1; k++)
        vfft_cs2pi_exact((long long)k, (long long)N, &p->V[2 * k], &p->V[2 * k + 1]);
    for (k = 0; k <= N2 / 2; k++)
        vfft_cs2pi_exact((long long)k, 2LL * (long long)N2, &p->U[2 * k], &p->U[2 * k + 1]);
    p->U[2 * (N2 / 2 + 1)] = p->U[2 * (N2 / 2 + 1) + 1] = 0.0;
    return p;
}

/* the block gathers and scatters. A block is k1 = k1b..k1b+15 (the segments
 * p_of_k1), k2 = k2b..k2b+15; its mirror block is the segments of N1 - k1 at
 * the positions N2-1-k2. Both land in a local buffer k2-major ([i][j] = the
 * bin k1b+j + N1*(k2b+i), and its partner at the same [i][j]): two complexes
 * turn as one 128-bit lane permute. The k1 = 0 lane of the mirror block
 * reads its own segment (a valid address); the self-paired segment is fixed
 * up after. */
static inline void _zfsr_gather(const vfft_zfsr_plan_t *p, const double *plane, int k1b, int k2b,
                                double *bufA, double *bufB)
{
    const int N1 = p->N1, N2 = p->N2, cmax = N2 - 1 - k2b;
    const int *P = p->fs->p_of_k1;
    int i, j;
    for (j = 0; j < VFFT_ZFSR_TB; j += 2)
    {
        const double *r0 = plane + 2 * ((size_t)P[k1b + j] * (size_t)N2 + (size_t)k2b);
        const double *r1 = plane + 2 * ((size_t)P[k1b + j + 1] * (size_t)N2 + (size_t)k2b);
        const double *q0 = plane + 2 * ((size_t)P[(N1 - k1b - j) % N1] * (size_t)N2);
        const double *q1 = plane + 2 * ((size_t)P[N1 - k1b - j - 1] * (size_t)N2);
        for (i = 0; i < VFFT_ZFSR_TB; i += 2)
        {
            const __m256d a = _mm256_loadu_pd(r0 + 2 * i), c = _mm256_loadu_pd(r1 + 2 * i);
            const __m256d d = _mm256_loadu_pd(q0 + 2 * (cmax - i - 1)), e = _mm256_loadu_pd(q1 + 2 * (cmax - i - 1));
            _mm256_store_pd(bufA + 2 * (i * VFFT_ZFSR_TB + j), _mm256_permute2f128_pd(a, c, 0x20));
            _mm256_store_pd(bufA + 2 * ((i + 1) * VFFT_ZFSR_TB + j), _mm256_permute2f128_pd(a, c, 0x31));
            _mm256_store_pd(bufB + 2 * (i * VFFT_ZFSR_TB + j), _mm256_permute2f128_pd(d, e, 0x31));
            _mm256_store_pd(bufB + 2 * ((i + 1) * VFFT_ZFSR_TB + j), _mm256_permute2f128_pd(d, e, 0x20));
        }
    }
}
/* the scatter: both blocks into the plane, whole 32-B vectors at aligned
 * addresses (the plane is 64-B aligned, N2 and the block edges multiples of
 * 16 complexes), streamed */
static inline void _zfsr_scatter(const vfft_zfsr_plan_t *p, double *plane, int k1b, int k2b,
                                 const double *bufA, const double *bufB)
{
    const int N1 = p->N1, N2 = p->N2, cmax = N2 - 1 - k2b;
    const int *P = p->fs->p_of_k1;
    int i, j;
    for (j = 0; j < VFFT_ZFSR_TB; j += 2)
    {
        double *r0 = plane + 2 * ((size_t)P[k1b + j] * (size_t)N2 + (size_t)k2b);
        double *r1 = plane + 2 * ((size_t)P[k1b + j + 1] * (size_t)N2 + (size_t)k2b);
        double *q0 = plane + 2 * ((size_t)P[(N1 - k1b - j) % N1] * (size_t)N2);
        double *q1 = plane + 2 * ((size_t)P[N1 - k1b - j - 1] * (size_t)N2);
        for (i = 0; i < VFFT_ZFSR_TB; i += 2)
        {
            const __m256d u = _mm256_load_pd(bufA + 2 * (i * VFFT_ZFSR_TB + j));
            const __m256d v = _mm256_load_pd(bufA + 2 * ((i + 1) * VFFT_ZFSR_TB + j));
            const __m256d s = _mm256_load_pd(bufB + 2 * (i * VFFT_ZFSR_TB + j));
            const __m256d t = _mm256_load_pd(bufB + 2 * ((i + 1) * VFFT_ZFSR_TB + j));
            _mm256_stream_pd(r0 + 2 * i, _mm256_permute2f128_pd(u, v, 0x20));
            _mm256_stream_pd(r1 + 2 * i, _mm256_permute2f128_pd(u, v, 0x31));
            _mm256_stream_pd(q0 + 2 * (cmax - i - 1), _mm256_permute2f128_pd(t, s, 0x20));
            _mm256_stream_pd(q1 + 2 * (cmax - i - 1), _mm256_permute2f128_pd(t, s, 0x31));
        }
    }
}

/* the block's lane twiddles w^(k1b + j), two lanes a vector, each value
 * doubled over its complex: (c_j, c_j, c_j+1, c_j+1) */
static inline void _zfsr_lane_tw(const vfft_zfsr_plan_t *p, int k1b, __m256d *vc, __m256d *vs)
{
    int j;
    for (j = 0; j < VFFT_ZFSR_TB; j += 2)
    {
        const __m256d v = _mm256_loadu_pd(p->V + 2 * (k1b + j));   /* (c_j, s_j, c_j+1, s_j+1) */
        vc[j / 2] = _mm256_movedup_pd(v);
        vs[j / 2] = _mm256_permute_pd(v, 0xF);
    }
}

/* r2c's last sweep: the four-step's plane -> X (CCE, M + 1 bins, natural).
 * Per pair (f, m = M - f), A = Z[f], B = Z[m] (zr2c.h's fold), (c, s) =
 * cos/sin(2*pi*f/N):
 *     t = A - conj(B);  x = t * (1/2 - s/2, -c/2);  X[f] = conj(B) + x;  X[m] = conj(A - x) */
static void _zfsr_sweep_fwd_range(const vfft_zfsr_plan_t *p, const double *plane, double *X, int k1lo, int k1hi)
{
    const int N1 = p->N1, N2 = p->N2, M = p->M;
    const int *P = p->fs->p_of_k1;
    const __m256d CONJ = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
    const __m256d HALF = _mm256_set1_pd(0.5), NHALF = _mm256_set1_pd(-0.5);
    const int stream = (((uintptr_t)X & 31) == 0);
    __attribute__((aligned(64))) double bufA[VFFT_ZFSR_TB * VFFT_ZFSR_TB * 2];
    __attribute__((aligned(64))) double bufB[VFFT_ZFSR_TB * VFFT_ZFSR_TB * 2];
    __attribute__((aligned(64))) double m16[VFFT_ZFSR_TB * 2];        /* the seventeenth lane's mirrors, per i */
    __attribute__((aligned(64))) double mb[(VFFT_ZFSR_TB + 2) * 2];   /* one mirror run + the unused lane-0 slot */
    __m256d vc[VFFT_ZFSR_TB / 2], vs[VFFT_ZFSR_TB / 2];
    int k1b, k2b, i, j;
    for (k1b = k1lo; k1b < k1hi; k1b += VFFT_ZFSR_TB)
    {
        const int kx = (k1b + VFFT_ZFSR_TB) % N1;
        const double *a16row = plane + 2 * (size_t)P[kx] * (size_t)N2;
        const double *b16row = plane + 2 * (size_t)P[(N1 - kx) % N1] * (size_t)N2;
        const __m256d c16 = _mm256_set1_pd(p->V[2 * (k1b + VFFT_ZFSR_TB)]), s16 = _mm256_set1_pd(p->V[2 * (k1b + VFFT_ZFSR_TB) + 1]);
        _zfsr_lane_tw(p, k1b, vc, vs);
        for (k2b = 0; k2b < N2 / 2; k2b += VFFT_ZFSR_TB)
        {
            const int cmax = N2 - 1 - k2b;
            _zfsr_gather(p, plane, k1b, k2b, bufA, bufB);
            /* the seventeenth lane, k1 = k1b + 16: its mirrors only, two positions a vector */
            for (i = 0; i < VFFT_ZFSR_TB; i += 2)
            {
                const __m256d a = _mm256_loadu_pd(a16row + 2 * (k2b + i));
                const __m256d br = _mm256_loadu_pd(b16row + 2 * (cmax - i - 1));
                const __m256d cb = _mm256_xor_pd(_mm256_permute2f128_pd(br, br, 0x01), CONJ);
                const __m256d u = _mm256_loadu_pd(p->U + 2 * (k2b + i));
                const __m256d c2 = _mm256_movedup_pd(u), s2 = _mm256_permute_pd(u, 0xF);
                const __m256d c = _mm256_fnmadd_pd(s16, s2, _mm256_mul_pd(c16, c2));
                const __m256d s = _mm256_fmadd_pd(s16, c2, _mm256_mul_pd(c16, s2));
                const __m256d wr = _mm256_fnmadd_pd(HALF, s, HALF), wi = _mm256_mul_pd(NHALF, c);
                const __m256d t = _mm256_sub_pd(a, cb);
                const __m256d x = _mm256_fmaddsub_pd(wr, t, _mm256_mul_pd(wi, _mm256_permute_pd(t, 0x5)));
                _mm256_store_pd(m16 + 2 * i, _mm256_xor_pd(_mm256_sub_pd(a, x), CONJ));
            }
            for (i = 0; i < VFFT_ZFSR_TB; i++)
            {
                const size_t f0 = (size_t)k1b + (size_t)N1 * (size_t)(k2b + i);
                const double *ra = bufA + 2 * i * VFFT_ZFSR_TB, *rb = bufB + 2 * i * VFFT_ZFSR_TB;
                double *of = X + 2 * f0, *om = X + 2 * ((size_t)M - f0 - VFFT_ZFSR_TB);
                const __m256d c2 = _mm256_set1_pd(p->U[2 * (k2b + i)]), s2 = _mm256_set1_pd(p->U[2 * (k2b + i) + 1]);
                for (j = 0; j < VFFT_ZFSR_TB; j += 2)
                {
                    const __m256d a = _mm256_load_pd(ra + 2 * j);
                    const __m256d cb = _mm256_xor_pd(_mm256_load_pd(rb + 2 * j), CONJ);
                    const __m256d c = _mm256_fnmadd_pd(vs[j / 2], s2, _mm256_mul_pd(vc[j / 2], c2));
                    const __m256d s = _mm256_fmadd_pd(vs[j / 2], c2, _mm256_mul_pd(vc[j / 2], s2));
                    const __m256d wr = _mm256_fnmadd_pd(HALF, s, HALF), wi = _mm256_mul_pd(NHALF, c);
                    const __m256d t = _mm256_sub_pd(a, cb);
                    const __m256d x = _mm256_fmaddsub_pd(wr, t, _mm256_mul_pd(wi, _mm256_permute_pd(t, 0x5)));
                    const __m256d xf = _mm256_add_pd(cb, x);
                    const __m256d xm = _mm256_xor_pd(_mm256_sub_pd(a, x), CONJ);
                    if (stream) _mm256_stream_pd(of + 2 * j, xf);
                    else _mm256_storeu_pd(of + 2 * j, xf);
                    /* lanes (j, j+1) mirror to run slots (16-j, 15-j): slot 16 is lane 0's, not this run's */
                    _mm256_storeu_pd(mb + 2 * (VFFT_ZFSR_TB - 1 - j), _mm256_permute2f128_pd(xm, xm, 0x01));
                }
                _mm_store_pd(mb, _mm_load_pd(m16 + 2 * i));   /* slot 0: the mirror of k1b + 16 */
                if (stream) for (j = 0; j < 2 * VFFT_ZFSR_TB; j += 4) _mm256_stream_pd(om + j, _mm256_load_pd(mb + j));
                else        for (j = 0; j < 2 * VFFT_ZFSR_TB; j += 4) _mm256_storeu_pd(om + j, _mm256_load_pd(mb + j));
            }
        }
    }
    if (stream) _mm_sfence();
}
/* the self-paired segment k1 = 0: (0, k2) <-> (0, N2 - k2); its centre; DC and Nyquist */
static void _zfsr_fix_fwd(const vfft_zfsr_plan_t *p, const double *plane, double *X)
{
    const int N1 = p->N1, N2 = p->N2, M = p->M;
    {
        const double *r = plane + 2 * (size_t)p->fs->p_of_k1[0] * (size_t)N2;
        const double z0r = r[0], z0i = r[1];
        int k2;
        for (k2 = 1; k2 < N2 / 2; k2++)
        {
            const size_t f = (size_t)N1 * (size_t)k2, m = (size_t)M - f;
            const double Ar = r[2 * k2], Ai = r[2 * k2 + 1];
            const double Br = r[2 * (N2 - k2)], Bi = r[2 * (N2 - k2) + 1];
            const double S = 0.5 - 0.5 * p->U[2 * k2 + 1], C = 0.5 * p->U[2 * k2];
            const double t1 = Ar - Br, t2 = Ai + Bi;
            const double xr = S * t1 + C * t2, xi = S * t2 - C * t1;
            X[2 * f] = Br + xr; X[2 * f + 1] = xi - Bi;
            X[2 * m] = Ar - xr; X[2 * m + 1] = xi - Ai;
        }
        X[M] = r[2 * (N2 / 2)]; X[M + 1] = -r[2 * (N2 / 2) + 1];   /* bin M/2: conj(Z) */
        X[0] = z0r + z0i; X[1] = 0.0;
        X[2 * (size_t)M] = z0r - z0i; X[2 * (size_t)M + 1] = 0.0;
    }
}

/* c2r's first sweep: X (CCE) -> the four-step's plane, scaled 2x as zr2c's
 * backward fold is. Per pair, F = X[f], Mv = X[m]:
 *     t = F - conj(Mv);  cy = t * (s, -c);  E = F + conj(Mv)
 *     Zhat[f] = E - cy;  Zhat[m] = conj(E + cy) */
static void _zfsr_sweep_bwd_range(const vfft_zfsr_plan_t *p, const double *X, double *plane, int k1lo, int k1hi)
{
    const int N1 = p->N1, N2 = p->N2, M = p->M;
    const __m256d CONJ = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
    const __m256d NEG = _mm256_set1_pd(-0.0);
    __attribute__((aligned(64))) double bufA[VFFT_ZFSR_TB * VFFT_ZFSR_TB * 2];
    __attribute__((aligned(64))) double bufB[VFFT_ZFSR_TB * VFFT_ZFSR_TB * 2];
    __m256d vc[VFFT_ZFSR_TB / 2], vs[VFFT_ZFSR_TB / 2];
    int k1b, k2b, i, j;
    for (k1b = k1lo; k1b < k1hi; k1b += VFFT_ZFSR_TB)
    {
        _zfsr_lane_tw(p, k1b, vc, vs);
        for (k2b = 0; k2b < N2 / 2; k2b += VFFT_ZFSR_TB)
        {
            for (i = 0; i < VFFT_ZFSR_TB; i++)
            {
                const size_t f0 = (size_t)k1b + (size_t)N1 * (size_t)(k2b + i);
                double *ra = bufA + 2 * i * VFFT_ZFSR_TB, *rb = bufB + 2 * i * VFFT_ZFSR_TB;
                const double *xf = X + 2 * f0, *xm = X + 2 * ((size_t)M - f0);
                const __m256d c2 = _mm256_set1_pd(p->U[2 * (k2b + i)]), s2 = _mm256_set1_pd(p->U[2 * (k2b + i) + 1]);
                for (j = 0; j < VFFT_ZFSR_TB; j += 2)
                {
                    const __m256d F = _mm256_loadu_pd(xf + 2 * j);
                    const __m256d mv = _mm256_loadu_pd(xm - 2 * (j + 1));
                    const __m256d cm = _mm256_xor_pd(_mm256_permute2f128_pd(mv, mv, 0x01), CONJ);
                    const __m256d c = _mm256_fnmadd_pd(vs[j / 2], s2, _mm256_mul_pd(vc[j / 2], c2));
                    const __m256d s = _mm256_fmadd_pd(vs[j / 2], c2, _mm256_mul_pd(vc[j / 2], s2));
                    const __m256d wi = _mm256_xor_pd(c, NEG);
                    const __m256d t = _mm256_sub_pd(F, cm);
                    const __m256d cy = _mm256_fmaddsub_pd(s, t, _mm256_mul_pd(wi, _mm256_permute_pd(t, 0x5)));
                    const __m256d E = _mm256_add_pd(F, cm);
                    _mm256_store_pd(ra + 2 * j, _mm256_sub_pd(E, cy));
                    _mm256_store_pd(rb + 2 * j, _mm256_xor_pd(_mm256_add_pd(E, cy), CONJ));
                }
            }
            _zfsr_scatter(p, plane, k1b, k2b, bufA, bufB);
        }
    }
    _mm_sfence();
}
/* the self-paired segment k1 = 0, the backward's */
static void _zfsr_fix_bwd(const vfft_zfsr_plan_t *p, const double *X, double *plane)
{
    const int N1 = p->N1, N2 = p->N2, M = p->M;
    {
        double *r = plane + 2 * (size_t)p->fs->p_of_k1[0] * (size_t)N2;
        const double X0 = X[0], XN = X[2 * (size_t)M];
        int k2;
        for (k2 = 1; k2 < N2 / 2; k2++)
        {
            const size_t f = (size_t)N1 * (size_t)k2, m = (size_t)M - f;
            const double Fr = X[2 * f], Fi = X[2 * f + 1], Mr = X[2 * m], Mi = X[2 * m + 1];
            const double s = p->U[2 * k2 + 1], c = p->U[2 * k2];
            const double t1 = Fr - Mr, t2 = Fi + Mi;
            const double yr = c * t2 + s * t1, yi = c * t1 - s * t2;
            const double Epr = Fr + Mr, Epi = Fi - Mi;
            r[2 * k2] = Epr - yr; r[2 * k2 + 1] = Epi + yi;
            r[2 * (N2 - k2)] = Epr + yr; r[2 * (N2 - k2) + 1] = yi - Epi;
        }
        r[2 * (N2 / 2)] = 2.0 * X[M]; r[2 * (N2 / 2) + 1] = -2.0 * X[M + 1];   /* bin M/2 */
        r[0] = X0 + XN; r[1] = X0 - XN;
    }
}

/* the sweep over the plan's threads: the k1 blocks cut evenly across the
 * workers (each block writes its own runs of X, or its own halves of the
 * plane's segments: disjoint, the output bitwise the serial walk), then the
 * self-paired segment. Runs on the caller thread only, before or after the
 * four-step -- never from a worker (the pool's nesting law). */
typedef struct { const vfft_zfsr_plan_t *p; const double *src; double *dst; int bwd, k1lo, k1hi; } _zfsr_sw_arg_t;
static void _zfsr_sw_tramp(void *v)
{
    const _zfsr_sw_arg_t *a = (const _zfsr_sw_arg_t *)v;
    if (a->k1lo >= a->k1hi) return;
    if (a->bwd) _zfsr_sweep_bwd_range(a->p, a->src, a->dst, a->k1lo, a->k1hi);
    else _zfsr_sweep_fwd_range(a->p, a->src, a->dst, a->k1lo, a->k1hi);
}
static void _zfsr_order_sweep(const vfft_zfsr_plan_t *p, const double *src, double *dst, int bwd)
{
    const int T = thread_pool_workers_for(p->fs->nthreads);
    if (T <= 1)
    {
        if (bwd) _zfsr_sweep_bwd_range(p, src, dst, 0, p->N1);
        else _zfsr_sweep_fwd_range(p, src, dst, 0, p->N1);
    }
    else
    {
        _zfsr_sw_arg_t a[THREAD_POOL_MAX_DISPATCH];
        const int nb = p->N1 / VFFT_ZFSR_TB;
        int t;
        for (t = 0; t < T; t++)
        {
            a[t].p = p; a[t].src = src; a[t].dst = dst; a[t].bwd = bwd;
            a[t].k1lo = (int)((long)nb * t / T) * VFFT_ZFSR_TB;
            a[t].k1hi = (int)((long)nb * (t + 1) / T) * VFFT_ZFSR_TB;
        }
        thread_pool_run(T, _zfsr_sw_tramp, a, sizeof a[0]);
        _vfft_zfsr_mt_count++;
    }
    if (bwd) _zfsr_fix_bwd(p, src, dst);
    else _zfsr_fix_fwd(p, src, dst);
}

/* r2c: x[N] (read as z[M]) -> X (N + 2 doubles). x == X is legal (the padded plane). */
static inline void vfft_zfsr_execute_fwd(const vfft_zfsr_plan_t *p, const double *x, double *X)
{
    vfft_k1fs_execute_fwd(p->fs, x, p->plane);
    _zfsr_order_sweep(p, p->plane, X, 0);
}
/* c2r: X (N + 2 doubles) -> x[N], unnormalised (N * x). X == x is legal. */
static inline void vfft_zfsr_execute_bwd(const vfft_zfsr_plan_t *p, const double *X, double *x)
{
    _zfsr_order_sweep(p, X, p->plane, 1);
    vfft_k1fs_execute_bwd(p->fs, p->plane, x);
}

#endif /* VFFT_ZFSR_H */
