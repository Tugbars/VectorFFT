/* il_prime.h — PRIME-N K=1 route on the PURE-IL machinery (OOP, NATURAL,
 * interleaved z->z, both directions).
 *
 * The IL counterpart of the split engine's src/core/primes/ (Rader when
 * N-1 is radix-smooth, else Bluestein), built natively on the IL machinery,
 * not as a wrapper around the split one:
 *
 *   - the inner M-point / (N-1)-point FFTs are PURE-IL plans (see
 *     _ilprime_inner_t) — packed complex end to end;
 *   - chirp / pointwise multiplies are complex multiplies on packed z,
 *     the layout's native operation;
 *   - both directions ride the same tables (conjugated twins), because
 *     every inner serves both directions.
 *
 * METHOD: Rader (convolution length N-1) and Bluestein (M = next pow2 >=
 * 2N-1), each with every inner of its pool, are RACED at create and the
 * verdict banks (k1_commit.h: _ilprime_create_banked). The band is whatever
 * the inner can build.
 *
 * MATH (from the gated split implementations — rader.h / bluestein.h —
 * with layout translated, not re-derived):
 *
 * Bluestein: W^{nk} = c[n]·c[k]·b[k-n], c[n] = e^{-i·pi·n^2/N},
 *   b[j] = conj(c[j]) => X[k] = c[k] · (x·c ⊛ b)[k]. Convolution at
 *   M >= 2N-1 via FFT: kern = FFT_M(b_wrapped)/M (the 1/M of the
 *   unnormalized inverse is BAKED into the kernel). Backward = the same
 *   pipeline on conjugated tables (unnormalized inverse, N·x semantics,
 *   matching every other IL bwd in the tree).
 *
 * Rader (prime N, generator g): X[0] = Σ x[n];
 *   X[g^{-q}] = x[0] + (a ⊛ b)[q], a[p] = x[g^p],
 *   kern_fwd = FFT_{N-1}(e^{-2πi·g^{-m}/N})/(N-1).
 *   Backward: gather by g^{-p}, kernel sign +1 on g^m, scatter by g^p.
 *   Chirp-squared angles use n^2 mod 2N in INTEGER arithmetic before the
 *   sin/cos (large-angle accuracy).
 */
#ifndef VFFT_IL_PRIME_H
#define VFFT_IL_PRIME_H
#include "common/math/numtheory.h"  /* vfft_is_prime, vfft_powmod, vfft_primitive_root */

#include "tw_exact.h"   /* once-rounded cos/sin(2*pi*p/n) for the create-time tables */
#include "il2p.h"
#include "common/support/race_timing.h" /* vfft_now_ns: the one monotonic clock */

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif


/* ── inner IL plan: an il2p pair, an il3p chain, or ZTURN-T. Any matched
 * fwd/bwd pair serves a convolution: the kernel is FFT'd once at create
 * through the SAME forward, so the pointwise multiply happens in that
 * forward's output order and the bwd consumes it back (the
 * matched-roundtrip law). ZTURN-T is guarded on VFFT_ZTT_H: a TU without
 * ztt.h keeps the il2p/il3p rule and its 4096 ceiling. ─────────────────── */
typedef struct {
    vfft_il2p_plan_t *p2;
    vfft_il3p_plan_t *p3;
#ifdef VFFT_ZTT_H
    vfft_ztt_plan_t *pt;    /* ZTURN-T: the banked ord=scr K=1 verdict at M when it
                             * names ZTURN-T — natural fwd/bwd is a matched roundtrip
                             * too, so the convolution's pointwise multiply runs in
                             * natural order */
#endif
} _ilprime_inner_t;

/* INNER PROVIDER: the banked wrapper installs a function that
 * fills the inner from wisdom (the K=1 IL pair verdict at length M, kernel
 * forms applied, raced and banked on a miss) for the duration of ONE
 * create; NULL = the structural rule below. Planning side, single-threaded
 * by the same contract as every create-time race. */
typedef int (*_ilprime_inner_provider_fn)(int M, _ilprime_inner_t *in, void *ctx);
static _ilprime_inner_provider_fn _ilprime_inner_provider = 0;
static void *_ilprime_inner_provider_ctx = 0;

static inline int _ilprime_inner_make(int M, _ilprime_inner_t *in)
{
    in->p2 = 0; in->p3 = 0;
#ifdef VFFT_ZTURN_H
    in->pz = 0;
#endif
#ifdef VFFT_ZTT_H
    in->pt = 0;
#endif
    if (_ilprime_inner_provider)
    {
        _ilprime_inner_t w;
        memset(&w, 0, sizeof w);
        if (_ilprime_inner_provider(M, &w, _ilprime_inner_provider_ctx))
        {
            *in = w;              /* the provider filled exactly one plan */
            return 1;
        }
    }
    if (M > 4096)
        return 0; /* il2p/il3p ceiling */
    /* balanced il2p pair first (any pair the registries cover — no parity
     * constraint since the odd-count tail; matches the front door), chain
     * second. Rader inners at N-1 = 2·odd (30 = 5x6, 28 = 7x4) resolve
     * here now, upgrading those primes from Bluestein. */
    {
        int bR1 = 0, bR2 = 0;
        for (int R2 = (M < 64 ? M : 64); R2 >= 3; R2--) {
            if (M % R2) continue;
            int R1 = M / R2;
            if (R1 < 3 || R1 > 64) continue;
            if (!vfft_il2p_leaf_fn(R2, 0) || !vfft_il2p_mid_fn(R1, 0)) continue;
            if (!bR1 || abs(R1 - R2) < abs(bR1 - bR2)) { bR1 = R1; bR2 = R2; }
        }
        if (bR1) in->p2 = vfft_il2p_create(M, bR1, bR2);
        if (in->p2) return 1;
    }
    {
        int cR2, cA, cB;
        if (vfft_il3p_default_chain(M, &cR2, &cA, &cB))
            in->p3 = vfft_il3p_create(M, cR2, cA, cB);
        return in->p3 != 0;
    }
}
static inline void _ilprime_inner_free(_ilprime_inner_t *in)
{
    vfft_il2p_destroy(in->p2);
    vfft_il3p_destroy(in->p3);
    in->p2 = 0; in->p3 = 0;
#ifdef VFFT_ZTT_H
    if (in->pt) vfft_ztt_destroy(in->pt);
    in->pt = 0;
#endif
}
static inline void _ilprime_inner_fwd(const _ilprime_inner_t *in,
                                      const double *zi, double *zo)
{
#ifdef VFFT_ZTT_H
    if (in->pt) { vfft_ztt_execute_fwd(in->pt, zi, zo); return; }   /* za -> zb, distinct */
#endif
    if (in->p2) vfft_il2p_execute_fwd(in->p2, zi, zo);
    else        vfft_il3p_execute_fwd(in->p3, zi, zo);
}
static inline void _ilprime_inner_bwd(const _ilprime_inner_t *in,
                                      const double *zi, double *zo)
{
#ifdef VFFT_ZTT_H
    if (in->pt) { vfft_ztt_execute_bwd(in->pt, zi, zo); return; }
#endif
    if (in->p2) (void)vfft_il2p_execute_bwd(in->p2, zi, zo);
    else        vfft_il3p_execute_bwd(in->p3, zi, zo);
}

/* packed complex a[i] *= b[i], i in [0, cnt) — cnt EVEN (M pow2 / N-1 even) */
static inline void _ilprime_cmul_vec(double *a, const double *b, size_t cnt)
{
#if defined(__AVX2__)
    /* (a·b: real = ar·br − ai·bi, imag = ai·br + ar·bi. The mask negates
     * the EVEN lanes of [ai, ar] so t = [−ai·bi, +ar·bi]; negating the odd
     * lanes instead computes a·conj(b), which leaves both prime methods
     * O(1) wrong with EXACT roundtrips (chirp autocorrelation forgives a
     * consistent conjugation). Roundtrip alone cannot gate Bluestein/Rader:
     * gate the forward.) */
    static const __m256d RMSK = { -0.0, 0.0, -0.0, 0.0 };
    for (size_t i = 0; i + 2 <= cnt; i += 2) {
        __m256d x = _mm256_loadu_pd(a + 2 * i);
        __m256d w = _mm256_loadu_pd(b + 2 * i);
        __m256d wr = _mm256_movedup_pd(w);              /* [br br] lanes  */
        __m256d wi = _mm256_permute_pd(w, 0xF);         /* [bi bi]        */
        __m256d xs = _mm256_permute_pd(x, 0x5);         /* [ai ar]        */
        __m256d t  = _mm256_mul_pd(_mm256_xor_pd(xs, RMSK), wi);
        _mm256_storeu_pd(a + 2 * i, _mm256_fmadd_pd(x, wr, t));
    }
#else
    for (size_t i = 0; i < cnt; i++) {
        double ar = a[2 * i], ai = a[2 * i + 1];
        double br = b[2 * i], bi = b[2 * i + 1];
        a[2 * i]     = ar * br - ai * bi;
        a[2 * i + 1] = ar * bi + ai * br;
    }
#endif
}

typedef struct {
    int N;
    int method;        /* 0 = Bluestein, 1 = Rader */
    int M;             /* Bluestein conv size; Rader: N-1 */
    _ilprime_inner_t inner;
    /* Bluestein: chirp c (2N doubles, packed) per direction — used for BOTH
     * modulate and demodulate (X[k] = c[k]·conv[k]); kernels (2M) carry the
     * unnormalized-inverse 1/M. */
    double *chf, *chb;         /* e^{∓i·pi·n^2/N} */
    double *kf, *kb;           /* FFT_M(b)/M, fwd/bwd */
    /* Rader */
    int *gpow, *ginvpow;       /* N-1 each */
    double *omf, *omb;         /* 2(N-1): FFT kernels, 1/(N-1) baked */
    /* scratch: two packed planes of the inner size */
    double *za, *zb;
} vfft_ilprime_plan_t;

static inline void vfft_ilprime_destroy(vfft_ilprime_plan_t *p)
{
    if (!p) return;
    _ilprime_inner_free(&p->inner);
    vfft_aligned_free(p->chf); vfft_aligned_free(p->chb);
    vfft_aligned_free(p->kf);  vfft_aligned_free(p->kb);
    vfft_aligned_free(p->omf); vfft_aligned_free(p->omb);
    free(p->gpow); free(p->ginvpow);
    vfft_aligned_free(p->za);  vfft_aligned_free(p->zb);
    free(p);
}

static inline double *_ilprime_alloc(size_t doubles)
{
    return (double *)vfft_aligned_alloc(doubles * sizeof(double));
}

/* Bluestein plan: M = next pow2 >= max(16, 2N-1) (16 = il2p's floor pair
 * 4x4). VALID FOR ANY N (the chirp uses n^2 mod 2N in integer arithmetic
 * — nothing here assumes primality); the band is whatever the inner can
 * construct (no cap of our own). M is a FREE parameter: next pow2 here,
 * though the split engine's bluestein_wisdom shows a smooth non-pow2 M can
 * win. */
static inline vfft_ilprime_plan_t *_ilprime_create_bluestein(int N)
{
    int M = 16;
    while (M < 2 * N - 1) M <<= 1;

    vfft_ilprime_plan_t *p = (vfft_ilprime_plan_t *)calloc(1, sizeof(*p));
    if (!p) return 0;
    p->N = N; p->method = 0; p->M = M;
    if (!_ilprime_inner_make(M, &p->inner)) { vfft_ilprime_destroy(p); return 0; }
    p->chf = _ilprime_alloc((size_t)2 * N);
    p->chb = _ilprime_alloc((size_t)2 * N);
    p->kf  = _ilprime_alloc((size_t)2 * M);
    p->kb  = _ilprime_alloc((size_t)2 * M);
    p->za  = _ilprime_alloc((size_t)2 * M);
    p->zb  = _ilprime_alloc((size_t)2 * M);
    if (!p->chf || !p->chb || !p->kf || !p->kb || !p->za || !p->zb) {
        vfft_ilprime_destroy(p);
        return 0;
    }
    for (int n = 0; n < N; n++) {
        long long m2 = ((long long)n * n) % (2LL * N); /* accuracy: mod first */
        double c, s;   /* a = -pi*m2/N = -2*pi*m2/(2N): sin(a) = -s */
        vfft_cs2pi_exact(m2, 2LL * N, &c, &s);
        p->chf[2 * n] = c;  p->chf[2 * n + 1] = -s;
        p->chb[2 * n] = c;  p->chb[2 * n + 1] = s;
    }
    /* kernels: b_f = conj(c_f) wrapped symmetric; kern = FFT_M(b)/M */
    for (int d = 0; d < 2; d++) {
        const double *c = d ? p->chb : p->chf;
        double *kern = d ? p->kb : p->kf;
        memset(p->za, 0, (size_t)2 * M * sizeof(double));
        p->za[0] = c[0]; p->za[1] = -c[1];
        for (int n = 1; n < N; n++) {
            double br = c[2 * n], bi = -c[2 * n + 1];
            p->za[2 * n] = br;            p->za[2 * n + 1] = bi;
            p->za[2 * (M - n)] = br;      p->za[2 * (M - n) + 1] = bi;
        }
        _ilprime_inner_fwd(&p->inner, p->za, p->zb);
        double inv = 1.0 / (double)M;
        for (int i = 0; i < 2 * M; i++) kern[i] = p->zb[i] * inv;
    }
    return p;
}

/* Rader plan: inner size N-1. NULL when the inner cannot be built — the
 * caller falls to Bluestein. */
static inline vfft_ilprime_plan_t *_ilprime_create_rader(int N)
{
    const int nm1 = N - 1;
    /* RADER IS PRIME-ONLY, and this is the guard that makes it so. The
     * reduction to a cyclic convolution needs a primitive root mod N, which
     * exists only for a prime modulus here. vfft_primitive_root tests
     * candidates with `powmod(g, (N-1)/f, N) != 1` for each prime factor f of
     * N - 1 -- a test that characterises a primitive root ONLY modulo a
     * prime. Handed a composite it can return a value that passes every
     * check and generates nothing, and the plan would then build cleanly and
     * compute the WRONG transform: a silent wrong answer. The guard lives
     * here so every caller is covered by construction. */
    if (!vfft_is_prime(N)) return 0;
    /* No size ceiling: the inner either builds (from the raced pool: at a
     * prime whose N - 1 is a power of two, the whole ZTURN-T registry up to
     * 262144) or the arm drops out of the race. */

    vfft_ilprime_plan_t *p = (vfft_ilprime_plan_t *)calloc(1, sizeof(*p));
    if (!p) return 0;
    p->N = N; p->method = 1; p->M = nm1;
    if (!_ilprime_inner_make(nm1, &p->inner)) { vfft_ilprime_destroy(p); return 0; }
    p->gpow    = (int *)malloc((size_t)nm1 * sizeof(int));
    p->ginvpow = (int *)malloc((size_t)nm1 * sizeof(int));
    p->omf = _ilprime_alloc((size_t)2 * nm1);
    p->omb = _ilprime_alloc((size_t)2 * nm1);
    p->za  = _ilprime_alloc((size_t)2 * nm1);
    p->zb  = _ilprime_alloc((size_t)2 * nm1);
    if (!p->gpow || !p->ginvpow || !p->omf || !p->omb || !p->za || !p->zb) {
        vfft_ilprime_destroy(p);
        return 0;
    }
    {
        int g = vfft_primitive_root(N);
        int ginv = (int)vfft_powmod(g, nm1 - 1, N);
        long long gp = 1, gip = 1;
        for (int i = 0; i < nm1; i++) {
            p->gpow[i] = (int)gp;
            p->ginvpow[i] = (int)gip;
            gp = gp * g % N;
            gip = gip * ginv % N;
        }
    }
    /* kernels (rader.h convention): fwd perm = ginvpow, sign -1;
     * bwd perm = gpow, sign +1; FFT_{N-1}, 1/(N-1) baked. */
    for (int d = 0; d < 2; d++) {
        const int *perm = d ? p->gpow : p->ginvpow;
        double sign = d ? 1.0 : -1.0;
        double *om = d ? p->omb : p->omf;
        for (int m = 0; m < nm1; m++) {
            double c, s;   /* a = sign*2*pi*perm/N: sin(a) = sign*s */
            vfft_cs2pi_exact((long long)perm[m], (long long)N, &c, &s);
            p->za[2 * m] = c;
            p->za[2 * m + 1] = sign * s;
        }
        _ilprime_inner_fwd(&p->inner, p->za, p->zb);
        double inv = 1.0 / (double)nm1;
        for (int i = 0; i < 2 * nm1; i++) om[i] = p->zb[i] * inv;
    }
    return p;
}

static inline void _ilprime_exec_bluestein(const vfft_ilprime_plan_t *p,
                                           const double *zin, double *zout,
                                           int bwd)
{
    const int N = p->N, M = p->M;
    const double *c = bwd ? p->chb : p->chf;
    const double *kern = bwd ? p->kb : p->kf;
    memset(p->za, 0, (size_t)2 * M * sizeof(double));
    for (int n = 0; n < N; n++) {
        double xr = zin[2 * n], xi = zin[2 * n + 1];
        double cr = c[2 * n], ci = c[2 * n + 1];
        p->za[2 * n]     = xr * cr - xi * ci;
        p->za[2 * n + 1] = xr * ci + xi * cr;
    }
    _ilprime_inner_fwd(&p->inner, p->za, p->zb);
    _ilprime_cmul_vec(p->zb, kern, (size_t)M);
    _ilprime_inner_bwd(&p->inner, p->zb, p->za);
    for (int k = 0; k < N; k++) {
        double vr = p->za[2 * k], vi = p->za[2 * k + 1];
        double cr = c[2 * k], ci = c[2 * k + 1];
        zout[2 * k]     = vr * cr - vi * ci;
        zout[2 * k + 1] = vr * ci + vi * cr;
    }
}

static inline void _ilprime_exec_rader(const vfft_ilprime_plan_t *p,
                                       const double *zin, double *zout,
                                       int bwd)
{
    const int N = p->N, nm1 = p->M;
    const int *gat = bwd ? p->ginvpow : p->gpow;      /* gather perm  */
    const int *sct = bwd ? p->gpow : p->ginvpow;      /* scatter perm */
    const double *om = bwd ? p->omb : p->omf;
    double dcr = 0.0, dci = 0.0;
    const double x0r = zin[0], x0i = zin[1];
    for (int n = 0; n < N; n++) {
        dcr += zin[2 * n];
        dci += zin[2 * n + 1];
    }
    for (int q = 0; q < nm1; q++) {
        int s = gat[q];
        p->za[2 * q] = zin[2 * s];
        p->za[2 * q + 1] = zin[2 * s + 1];
    }
    _ilprime_inner_fwd(&p->inner, p->za, p->zb);
    _ilprime_cmul_vec(p->zb, om, (size_t)nm1);
    _ilprime_inner_bwd(&p->inner, p->zb, p->za);
    for (int q = 0; q < nm1; q++) {
        int s = sct[q];
        zout[2 * s]     = x0r + p->za[2 * q];
        zout[2 * s + 1] = x0i + p->za[2 * q + 1];
    }
    zout[0] = dcr;
    zout[1] = dci;
}

/* zin == zout IS safe in both methods: every read of zin (Bluestein's
 * modulate; Rader's x0 capture + DC sum + gather) completes before the
 * first zout write. */
static inline void vfft_ilprime_execute_fwd(const vfft_ilprime_plan_t *p,
                                            const double *zin, double *zout)
{
    if (p->method) _ilprime_exec_rader(p, zin, zout, 0);
    else           _ilprime_exec_bluestein(p, zin, zout, 0);
}
/* unnormalized inverse (N·x), matching every IL bwd in the tree */
static inline void vfft_ilprime_execute_bwd(const vfft_ilprime_plan_t *p,
                                            const double *zin, double *zout)
{
    if (p->method) _ilprime_exec_rader(p, zin, zout, 1);
    else           _ilprime_exec_bluestein(p, zin, zout, 1);
}

#endif /* VFFT_IL_PRIME_H */
