/* zsplit.h — K=1 SCRAMBLED-order interleaved c2c: the BLOCK-SPLIT-interior
 * staged cascade (z_cascade_plan.md §4.99–§4.998; wisdom
 * build_tuned/k1_wis/zsplit_wisdom.txt).
 *
 * Pipeline (all kernels generator-owned, codelet_zil.ml emit_z_split):
 *   fwd: s0s (z -> block-split, deinterleave paid once) -> ms xk (in-place,
 *        SHUFFLE-FREE split mids, group-constant splat-pair twiddles)
 *        -> sterm (split -> z, 4 cols/iter, packed per-col w^1, re-interleave
 *        in stores). Output = digit-reversed comb (SCRAMBLED contract):
 *        out[l*(N/8) + g] = X[drev(g*8 + l)] over the chain's mixed radices.
 *   bwd: stermb -> msb (REVERSE stage order, conj tables) -> s0sb.
 *        bwd(fwd(x)) = N*x with the MATCHED permutation (roundtrip gated
 *        1.1–1.4e-15 at 2048/4096/8192/16384).
 * Execution shape = the measured production winner (§4.996): plan-time base
 * tables + compact CALLED kernels (code fusion measured NEGATIVE).
 * In-place safe (zin == zout): the terminator is the only writer of zout and
 * reads only the scratch plane.
 *
 * Coverage: N = prod(chain), chain[0] in {4,8}, mids in {4,8}, last == 8,
 * nf in [3,6]. Default chains = the calibrated per-cell winners.
 */
#ifndef VFFT_ZSPLIT_H
#define VFFT_ZSPLIT_H

#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <malloc.h>
#define VFFT_ZS_ALLOC(sz) _aligned_malloc((sz), 64)
#define VFFT_ZS_FREE(p) _aligned_free(p)
#else
#define VFFT_ZS_ALLOC(sz) aligned_alloc(64, (sz))
#define VFFT_ZS_FREE(p) free(p)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* generator-owned kernels (codelets/zil/avx2, 11-arg z ABI) */
#define VFFT_ZS_DECL(fn) extern void fn(const double *, const double *, \
    double *, double *, const double *, const double *,                 \
    unsigned long long, unsigned long long, unsigned long long,         \
    unsigned long long, unsigned long long);
VFFT_ZS_DECL(radix4_z_s0s_fwd_avx2)  VFFT_ZS_DECL(radix8_z_s0s_fwd_avx2)
VFFT_ZS_DECL(radix4_z_msg_fwd_avx2)  VFFT_ZS_DECL(radix8_z_msg_fwd_avx2)
VFFT_ZS_DECL(radix8_z_sterm_fwd_avx2)
VFFT_ZS_DECL(radix8_z_sterm2_fwd_avx2)   /* 2-quad unroll-and-jam twin (§4.9993) */
VFFT_ZS_DECL(radix4_z_s0s_bwd_avx2)  VFFT_ZS_DECL(radix8_z_s0s_bwd_avx2)
VFFT_ZS_DECL(radix4_z_msg_bwd_avx2)  VFFT_ZS_DECL(radix8_z_msg_bwd_avx2)
VFFT_ZS_DECL(radix8_z_sterm_bwd_avx2)
#undef VFFT_ZS_DECL

#define VFFT_ZSPLIT_MAX_NF 6

typedef struct {
    int N, nf;
    int chain[VFFT_ZSPLIT_MAX_NF];
    long D[VFFT_ZSPLIT_MAX_NF], G[VFFT_ZSPLIT_MAX_NF];
    long *gb[VFFT_ZSPLIT_MAX_NF];       /* plan-time group base tables */
    double *twsp[VFFT_ZSPLIT_MAX_NF];   /* fwd splat-pair sets (per group) */
    double *twspb[VFFT_ZSPLIT_MAX_NF];  /* bwd (conjugated) sets */
    double *twq, *twqb;                 /* terminator packed per-col w^1, fwd/bwd */
    double *sp;                         /* block-split scratch, N complex */
    int t2q;                            /* fwd terminator schedule: 0 = sterm
                                         * (single-quad), 1 = sterm2 (2-quad
                                         * unroll-and-jam). Bit-identical pair;
                                         * per-cell pick — their delta is the
                                         * same order as code-placement luck
                                         * (§4.9993), so it is MEASURED, not
                                         * reasoned. bwd keeps single-quad
                                         * (2-quad refuted: +29..36% kernel). */
} vfft_zsplit_plan_t;

/* calibrated per-cell winners (zsplit_wisdom.txt, paced finals 2026-07-24) */
static inline int vfft_zsplit_default_chain(int N, int *chain)
{
    switch (N) {
    case 2048:  chain[0]=4; chain[1]=8; chain[2]=8; chain[3]=8; return 4;
    case 4096:  chain[0]=4; chain[1]=4; chain[2]=4; chain[3]=8; chain[4]=8; return 5;
    case 8192:  chain[0]=4; chain[1]=4; chain[2]=8; chain[3]=8; chain[4]=8; return 5;
    case 16384: chain[0]=4; chain[1]=8; chain[2]=8; chain[3]=8; chain[4]=8; return 5;
    default: return 0;
    }
}

static inline long _vfft_zs_brev(long g, int s, const int *r)
{
    long f[VFFT_ZSPLIT_MAX_NF];
    for (int i = s - 1; i >= 0; i--) { f[i] = g % r[i]; g /= r[i]; }
    long P = 1, v = 0;
    for (int i = 0; i < s; i++) { v += f[i] * P; P *= r[i]; }
    return v;
}
static inline long _vfft_zs_base(long g, int s, const int *r, const long *d)
{
    long b = 0;
    for (int i = s - 1; i >= 0; i--) { long f = g % r[i]; g /= r[i]; b += f * d[i]; }
    return b;
}

static inline void vfft_zsplit_destroy(vfft_zsplit_plan_t *p)
{
    if (!p) return;
    for (int s = 0; s < VFFT_ZSPLIT_MAX_NF; s++) {
        free(p->gb[s]);
        VFFT_ZS_FREE(p->twsp[s]);
        VFFT_ZS_FREE(p->twspb[s]);
    }
    VFFT_ZS_FREE(p->twq);
    VFFT_ZS_FREE(p->twqb);
    VFFT_ZS_FREE(p->sp);
    free(p);
}

static inline vfft_zsplit_plan_t *vfft_zsplit_create(int N, const int *chain,
                                                     int nf)
{
    if (nf < 3 || nf > VFFT_ZSPLIT_MAX_NF) return NULL;
    if (chain[0] != 4 && chain[0] != 8) return NULL;
    if (chain[nf - 1] != 8) return NULL;               /* sterm is radix-8 */
    long prod = 1;
    for (int s = 0; s < nf; s++) {
        if (s >= 1 && s <= nf - 2 && chain[s] != 4 && chain[s] != 8) return NULL;
        prod *= chain[s];
    }
    if (prod != N || (N / 8) % 4) return NULL;         /* sterm: count %% 4 */

    vfft_zsplit_plan_t *p = (vfft_zsplit_plan_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->N = N; p->nf = nf;
    /* Terminator-schedule DEFAULT = the incumbent for the create-time pick
     * race (§4.9994): vfft.c's hook races sterm vs sterm2 on wisdom miss
     * (3% hysteresis toward this value) and banks the verdict as a kind-4
     * oop_wisdom line; a wisdom hit overwrites t2q with the banked pick.
     * Standalone users of zsplit.h (benches) get this static default. */
    p->t2q = (N == 4096) ? 1 : 0;
    for (int s = 0; s < nf; s++) p->chain[s] = chain[s];
    p->D[nf - 1] = 1;
    for (int i = nf - 2; i >= 0; i--) p->D[i] = p->D[i + 1] * chain[i + 1];
    p->G[0] = 1;
    for (int i = 1; i < nf; i++) p->G[i] = p->G[i - 1] * chain[i - 1];
    if (p->D[nf - 2] % 4) goto fail;                   /* ms: count %% 4 */

    const double TAU = 2.0 * M_PI;
    for (int s = 1; s <= nf - 2; s++) {
        int Rm1 = chain[s] - 1;
        long M = (long)N / p->D[s];
        p->twsp[s] = (double *)VFFT_ZS_ALLOC((size_t)p->G[s] * Rm1 * 8 * 8);
        p->twspb[s] = (double *)VFFT_ZS_ALLOC((size_t)p->G[s] * Rm1 * 8 * 8);
        p->gb[s] = (long *)malloc((size_t)p->G[s] * sizeof(long));
        if (!p->twsp[s] || !p->twspb[s] || !p->gb[s]) goto fail;
        for (long g = 0; g < p->G[s]; g++) {
            p->gb[s][g] = _vfft_zs_base(g, s, chain, p->D);
            long brev = _vfft_zs_brev(g, s, chain);
            for (int l = 1; l < chain[s]; l++) {
                double a = -TAU * (double)(((long)l * brev) % M) / (double)M;
                double *f = p->twsp[s] + ((size_t)g * Rm1 + (l - 1)) * 8;
                double *b = p->twspb[s] + ((size_t)g * Rm1 + (l - 1)) * 8;
                for (int j = 0; j < 4; j++) {
                    f[j] = cos(a); f[4 + j] = sin(a);
                    b[j] = cos(a); b[4 + j] = -sin(a);
                }
            }
        }
    }
    {
        long cols = (long)N / 8;
        p->twq = (double *)VFFT_ZS_ALLOC((size_t)cols * 16);
        p->twqb = (double *)VFFT_ZS_ALLOC((size_t)cols * 16);
        p->sp = (double *)VFFT_ZS_ALLOC((size_t)2 * N * 8);
        if (!p->twq || !p->twqb || !p->sp) goto fail;
        for (long k = 0; k < cols; k++) {
            double a = -TAU * (double)(_vfft_zs_brev(k, nf - 1, chain) % N)
                       / (double)N;
            p->twq[2 * (k & ~3L) + (k & 3L)] = cos(a);
            p->twq[2 * (k & ~3L) + 4 + (k & 3L)] = sin(a);
            p->twqb[2 * (k & ~3L) + (k & 3L)] = cos(a);
            p->twqb[2 * (k & ~3L) + 4 + (k & 3L)] = -sin(a);
        }
    }
    return p;
fail:
    vfft_zsplit_destroy(p);
    return NULL;
}

/* natural z in -> digit-reversed comb z out (SCRAMBLED). zin==zout OK. */
static inline void vfft_zsplit_execute_fwd(const vfft_zsplit_plan_t *p,
                                           const double *zin, double *zout)
{
    const int nf = p->nf;
    if (p->chain[0] == 8)
        radix8_z_s0s_fwd_avx2(zin, 0, p->sp, 0, 0, 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0]);
    else
        radix4_z_s0s_fwd_avx2(zin, 0, p->sp, 0, 0, 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0]);
    /* mids: ONE group-looped call per stage (msg — §4.9991 addendum: kills
     * the per-group call overhead + trip-count-2 loop-exit mispredicts) */
    for (int s = 1; s <= nf - 2; s++) {
        void (*f)(const double *, const double *, double *, double *,
                  const double *, const double *, unsigned long long,
                  unsigned long long, unsigned long long, unsigned long long,
                  unsigned long long) =
            (p->chain[s] == 8) ? radix8_z_msg_fwd_avx2 : radix4_z_msg_fwd_avx2;
        f(p->sp, 0, p->sp, 0, p->twsp[s], 0,
          (unsigned long long)p->D[s], (unsigned long long)p->G[s],
          0, 0, (unsigned long long)p->D[s]);
    }
    (p->t2q ? radix8_z_sterm2_fwd_avx2 : radix8_z_sterm_fwd_avx2)(
        p->sp, 0, zout, 0, p->twq, 0, 0, 0,
        (unsigned long long)(p->N / 8), 0,
        (unsigned long long)(p->N / 8));
}

/* digit-reversed comb z in -> N * natural z out. zin==zout OK. */
static inline void vfft_zsplit_execute_bwd(const vfft_zsplit_plan_t *p,
                                           const double *zin, double *zout)
{
    const int nf = p->nf;
    radix8_z_sterm_bwd_avx2(zin, 0, p->sp, 0, p->twqb, 0, 0, 0,
                            (unsigned long long)(p->N / 8), 0,
                            (unsigned long long)(p->N / 8));
    for (int s = nf - 2; s >= 1; s--) {
        void (*f)(const double *, const double *, double *, double *,
                  const double *, const double *, unsigned long long,
                  unsigned long long, unsigned long long, unsigned long long,
                  unsigned long long) =
            (p->chain[s] == 8) ? radix8_z_msg_bwd_avx2 : radix4_z_msg_bwd_avx2;
        f(p->sp, 0, p->sp, 0, p->twspb[s], 0,
          (unsigned long long)p->D[s], (unsigned long long)p->G[s],
          0, 0, (unsigned long long)p->D[s]);
    }
    if (p->chain[0] == 8)
        radix8_z_s0s_bwd_avx2(p->sp, 0, zout, 0, 0, 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0]);
    else
        radix4_z_s0s_bwd_avx2(p->sp, 0, zout, 0, 0, 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0]);
}

#endif /* VFFT_ZSPLIT_H */
