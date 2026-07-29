/* il_prime.h — PRIME-N K=1 route on the PURE-IL machinery (OOP, NATURAL,
 * interleaved z->z, both directions).
 *
 * The in-place SPLIT engine has served prime N for a while
 * (src/core/primes/: prime_dispatch -> Rader when N-1 is radix-smooth,
 * else Bluestein; both recurse their inner FFT through the engine's own
 * planner). This header is the IL translation of that design, NOT a
 * memcpy wrapper around it (the wrapper shortcut was built and deleted on
 * Tugbars' call — serve it natively):
 *
 *   - the inner M-point / (N-1)-point FFTs are PURE-IL plans (il2p pair
 *     for pow2, il3p chain for odd·2^k) — packed complex end to end;
 *   - chirp / pointwise multiplies are complex multiplies on packed z,
 *     the layout's native operation;
 *   - both directions ride the same tables (conjugated twins), because
 *     il2p/il3p serve both directions.
 *
 * METHOD PREFERENCE mirrors prime_dispatch: RADER when the (N-1) inner is
 * IL-expressible (convolution length N-1 vs Bluestein's ~2N; measured ~2x
 * on the split engine), else BLUESTEIN (M = next pow2 >= 2N-1, always
 * expressible in-band). ⚠ This is an availability preference, not a
 * measured per-cell pick — that belongs to the wisdom campaign.
 *
 * BAND: inner size <= 4096 (il2p's ceiling) => prime N <= 2048, exactly
 * the Bailey band; the cascade owns larger N.
 *
 * MATH (cribbed from the gated split implementations — rader.h /
 * bluestein.h — with layout translated, not re-derived):
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

#include "il2p.h"

#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

/* ── tiny integer helpers (self-contained: vfft.c's TU does not include
 * primes/prime_dispatch.h, and coupling to proto_stride_compat.h is what
 * this header exists to avoid) ─────────────────────────────────────── */
static inline int _ilprime_is_prime(int n)
{
    if (n < 2) return 0;
    if ((n & 1) == 0) return n == 2;
    for (int p = 3; (long long)p * p <= n; p += 2)
        if (n % p == 0) return 0;
    return 1;
}
static inline long long _ilprime_powmod(long long b, long long e, long long m)
{
    long long r = 1;
    b %= m;
    while (e > 0) {
        if (e & 1) r = r * b % m;
        b = b * b % m;
        e >>= 1;
    }
    return r;
}
static inline int _ilprime_find_generator(int N)
{
    int f[16], nf = 0, r = N - 1;
    for (int p = 2; (long long)p * p <= r; p += (p == 2 ? 1 : 2))
        if (r % p == 0) {
            f[nf++] = p;
            while (r % p == 0) r /= p;
        }
    if (r > 1) f[nf++] = r;
    for (int g = 2; g < N; g++) {
        int ok = 1;
        for (int i = 0; i < nf && ok; i++)
            if (_ilprime_powmod(g, (N - 1) / f[i], N) == 1) ok = 0;
        if (ok) return g;
    }
    return 0; /* unreachable for prime N */
}

/* ── inner IL plan: il2p pair (pow2) or il3p chain (odd·2^k) ─────────── */
typedef struct {
    vfft_il2p_plan_t *p2;
    vfft_il3p_plan_t *p3;
} _ilprime_inner_t;

static inline int _ilprime_inner_make(int M, _ilprime_inner_t *in)
{
    in->p2 = 0; in->p3 = 0;
    /* balanced il2p pair first (any even pair the registries cover — %2,
     * matching the front door: 100 = 10x10 is a PAIR now), chain second. */
    {
        int bR1 = 0, bR2 = 0;
        for (int R2 = (M < 64 ? M : 64); R2 >= 4; R2--) {
            if (M % R2) continue;
            int R1 = M / R2;
            if (R1 < 4 || R1 > 64 || (R1 % 2) || (R2 % 2)) continue;
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
}
static inline void _ilprime_inner_fwd(const _ilprime_inner_t *in,
                                      const double *zi, double *zo)
{
    if (in->p2) vfft_il2p_execute_fwd(in->p2, zi, zo);
    else        vfft_il3p_execute_fwd(in->p3, zi, zo);
}
static inline void _ilprime_inner_bwd(const _ilprime_inner_t *in,
                                      const double *zi, double *zo)
{
    if (in->p2) (void)vfft_il2p_execute_bwd(in->p2, zi, zo);
    else        vfft_il3p_execute_bwd(in->p3, zi, zo);
}

/* packed complex a[i] *= b[i], i in [0, cnt) — cnt EVEN (M pow2 / N-1 even) */
static inline void _ilprime_cmul_vec(double *a, const double *b, size_t cnt)
{
#if defined(__AVX2__)
    /* (a·b: real = ar·br − ai·bi, imag = ai·br + ar·bi. The mask negates
     * the EVEN lanes of [ai, ar] so t = [−ai·bi, +ar·bi]; negating the odd
     * lanes instead computes a·conj(b) — the exact bug the transform-
     * identity probe caught: both prime methods O(1) wrong with EXACT
     * roundtrips, because chirp autocorrelation forgives a consistent
     * conjugation. Roundtrip alone cannot gate Bluestein/Rader.) */
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
    VFFT_IL2P_FREE(p->chf); VFFT_IL2P_FREE(p->chb);
    VFFT_IL2P_FREE(p->kf);  VFFT_IL2P_FREE(p->kb);
    VFFT_IL2P_FREE(p->omf); VFFT_IL2P_FREE(p->omb);
    free(p->gpow); free(p->ginvpow);
    VFFT_IL2P_FREE(p->za);  VFFT_IL2P_FREE(p->zb);
    free(p);
}

static inline double *_ilprime_alloc(size_t doubles)
{
    return (double *)VFFT_IL2P_ALLOC(doubles * sizeof(double));
}

/* Bluestein plan: M = next pow2 >= max(16, 2N-1) (16 = il2p's floor pair
 * 4x4). NULL when M would exceed the band (> 4096) or any alloc fails. */
static inline vfft_ilprime_plan_t *_ilprime_create_bluestein(int N)
{
    int M = 16;
    while (M < 2 * N - 1) M <<= 1;
    if (M > 4096) return 0;

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
        double a = -VFFT_IL2P_PI * (double)m2 / (double)N;
        p->chf[2 * n] = cos(a);  p->chf[2 * n + 1] = sin(a);
        p->chb[2 * n] = cos(a);  p->chb[2 * n + 1] = -sin(a);
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

/* Rader plan: inner size N-1 (pow2 -> il2p pair; odd·2^k -> il3p chain).
 * NULL when the inner is not IL-expressible — the caller falls to
 * Bluestein, which always is (in-band). */
static inline vfft_ilprime_plan_t *_ilprime_create_rader(int N)
{
    const int nm1 = N - 1;
    if (nm1 > 4096) return 0;

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
        int g = _ilprime_find_generator(N);
        int ginv = (int)_ilprime_powmod(g, nm1 - 1, N);
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
            double a = sign * 2.0 * VFFT_IL2P_PI * (double)perm[m] / (double)N;
            p->za[2 * m] = cos(a);
            p->za[2 * m + 1] = sin(a);
        }
        _ilprime_inner_fwd(&p->inner, p->za, p->zb);
        double inv = 1.0 / (double)nm1;
        for (int i = 0; i < 2 * nm1; i++) om[i] = p->zb[i] * inv;
    }
    return p;
}

/* Front door: Rader preferred (shorter convolution; the split engine
 * measured it ~2x over Bluestein), Bluestein otherwise. ⚠ availability
 * preference — the measured per-cell pick belongs in wisdom. */
static inline vfft_ilprime_plan_t *vfft_ilprime_create(int N)
{
    if (!_ilprime_is_prime(N) || N < 5 || N > 2048) return 0;
    vfft_ilprime_plan_t *p = _ilprime_create_rader(N);
    if (p) return p;
    return _ilprime_create_bluestein(N);
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
