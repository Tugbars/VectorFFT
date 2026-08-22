/* zr2c.h — INTERLEAVED (z) real-transform folds: the D2 "MKL-mirror IL" route's
 * two hand-written passes (docs/research/mkl_r2c_campaign/DESIGN_interleaved_r2c.md,
 * Phase 0 verdict box 2026-08-13: arm e's shape, validated at MKL-recombine
 * parity ±15%, −30…−59% vs the shipped K-axis fold).
 *
 * THE ROUTE (even N only; rfft keeps odd N — coverage is additive):
 *   r2c fwd: x[N] ==reinterpret(0 work)==> z[N/2] interleaved
 *            --(IL c2c(N/2) fwd, NATURAL order, in place or OOP)--> Z
 *            --(_zr2c_fold_fwd)--> X[0..N/2] CCE interleaved (N+2 doubles)
 *   c2r bwd: X ==(_zr2c_fold_bwd)==> Zhat[N/2] (scaled 2x)
 *            --(IL c2c(N/2) bwd, unnormalized N/2)--> N * x pairs, reinterpret.
 *
 * LAW CHECK: both passes are z->z (no in_re/in_im anywhere) — clean under the
 * hybrid ban; the interior is the existing IL c2c pool, zero new codelets.
 *
 * FREQUENCY-AXIS SIMD (AVX2): 4 ascending bins + the 4 REVERSED mirror bins per
 * iteration; table is the affine pre-biased HALVED pair-table, FULLY SPLIT
 * (plain loads, zero table shuffles).  INTERLEAVED-NATIVE since 2026-08-21:
 * the loop no longer de-interleaves to Ar/Ai/Br/Bi -- the math is one complex
 * multiply, so it runs on packed pairs and the un/re-interleave shuffles are
 * gone.  MEASURED: the old form was 20.1 cyc/iter at EVERY N (flat => pure
 * issue-port cost); ALL 20 of its shuffles are port-5 on Raptor Cove, in-lane
 * vunpck*pd ymm included.  The new form runs 10 shuffles/iter -> 10.6 (fwd) /
 * 12.4 (bwd) cyc/iter below the L1 cliff: -48%/-38% at N<=2048, tapering to
 * ~3-9% at N>=4096 where the z+X working set exceeds L1 and memory binds.
 * TRADE ACCEPTED: signs are now explicit vxorpd (the old "zero xor" boast is
 * retired) -- they issue on p0/p1/p5 instead of piling onto the saturated p5,
 * which is exactly why it is faster.  fwd stays BITWISE vs the old kernel;
 * bwd differs at rel ~1.3e-16 (the s=1-2S~ / c=2C~ derivation reassociates).
 *   S~[f] = 1/2 - 1/2 sin(2*pi*f/N),   C~[f] = 1/2 cos(2*pi*f/N),  f = 0..N/4
 * Per pair (f, m=N/2-f), A=Z[f], B=Z[m]:
 *   t1 = Ar-Br   t2 = Ai+Bi
 *   xr = S~*t1 + C~*t2          xi = S~*t2 - C~*t1
 *   X[f] = (Br+xr, xi-Bi)       X[m] = (Ar-xr, xi-Ai)
 * DC/Nyquist: X[0] = Zr0+Zi0, X[N/2] = Zr0-Zi0, imag parts LITERAL +0.0
 * (structural zeros written; the backward never reads them).
 * Center (half even, f=half/2): X[f] = conj(Z[f]).
 *
 * BACKWARD (derived from the exact inverse, includes the x2 so that
 * c2c_bwd(N/2) lands on the library-wide bwd(fwd(x)) = N*x convention;
 * c = 2*C~, s = 1-2*S~ are derived IN-LOOP — one shared table, no bwd table):
 *   t1 = Fr-Mr   t2 = Fi+Mi     (F=X[f], M=X[m])
 *   yr = c*t2 + s*t1            yi = c*t1 - s*t2
 *   Zhat[f] = (Fr+Mr-yr, Fi-Mi+yi)   Zhat[m] = (Fr+Mr+yr, yi-Fi+Mi)
 *   DC/Nyq: Zhat[0] = (X0+XN, X0-XN). Center: Zhat[f] = (2*Xr, -2*Xi).
 *
 * IN-PLACE SAFE by construction: each pair is fully read before either bin is
 * written; DC/Nyquist writes only touch bin 0 and the N/2 pad slot. The same
 * function serves OOP and in-place (X == z requires the caller's plane to
 * carry the 2*(N/2+1)-double padding — the CONCLUSIONS §2.3 contract).
 *
 * BATCH: transform-contiguous. Per transform t the input plane is
 * z + t*zs (zs >= N doubles) and the output plane X + t*xs (xs >= N+2).
 * The frequency-axis SIMD is per-transform, so K=1 runs FULL vector width —
 * the G1 (K=1 scalar) disease of the K-axis codelets does not exist here.
 */
#ifndef VFFT_ZR2C_H
#define VFFT_ZR2C_H

#include <stddef.h>
#include <math.h>
#if defined(__AVX2__) || defined(__AVX512F__)
#include <immintrin.h>
#endif

#ifndef VFFT_ZR2C_PI
#define VFFT_ZR2C_PI 3.14159265358979323846
#endif

/* Affine pair-table, N/4+1 entries each (f = 0..N/4 inclusive covers every
 * pair index for any even N; entry 0 is never read — kept for direct
 * indexing). ~4N bytes total, half of a plain (cos,sin)-over-N/2 table. */
/* FOUR tables, not two.
 *
 * 🔴 The affine pair (0.5 - 0.5 sin, 0.5 cos) is a FORWARD convenience:
 * the forward fold consumes it almost directly (one negate for -C). The
 * BACKWARD's coefficients are the RAW twiddles -- work the algebra through
 * and 1 - 2*(0.5 - 0.5 sin) = sin, 2*(0.5 cos) = cos -- so the backward was
 * UNDOING the encoding on every iteration, four lanes at a time, forever:
 * an fnmadd and a mul plus two live constant registers in the vector body,
 * and the same two ops again in the scalar tail. Bank them instead. Cost:
 * N/4 doubles per plan, paid once at create. */
static void _zr2c_init_aff(int N, double *affS, double *affC,
                           double *bwdS, double *bwdC)
{
    int top = N / 4;
    for (int f = 0; f <= top; f++)
    {
        double th = 2.0 * VFFT_ZR2C_PI * (double)f / (double)N;
        double sn = sin(th), cs = cos(th);
        affS[f] = 0.5 - 0.5 * sn;   /* forward: the affine encoding */
        affC[f] = 0.5 * cs;
        /* 🔴 bwdC holds -cos, the CONJUGATED coefficient. The backward
         * needs cy = conj(conj(t)*w), and conj(conj(t)*w) == t*conj(w), so
         * multiplying t by conj(w) = (sin, -cos) produces cy DIRECTLY -- no
         * conj(t) before the multiply and no conj(y) after it. Banking the
         * sign makes that free; computing it would cost an op per iteration,
         * which is the encoding mistake this table exists to stop repeating. */
        bwdS[f] = sn;               /* backward: raw sin          */
        bwdC[f] = -cs;              /* backward: -cos == conj(w)  */
    }
}

/* forward fold: Z (N/2 complex, natural, interleaved) -> X (CCE, N/2+1
 * complex, interleaved). X may alias Z (in-place; needs the padded plane). */
/* 🔴 NO __restrict__ ON THE DATA PLANES. Both of these folds are called
 * with the SAME pointer for input and output -- vfft.c:2291 and :2312 pass
 * (dre, dre), and :2324 passes (sre, dre) which alias whenever the plan is
 * in-place. __restrict__ is a promise to the compiler that the two do not
 * alias; making that promise and then breaking it is undefined behaviour
 * even though this loop happens to be safe (each iteration loads bin f and
 * its mirror m before storing either, and the indices never revisit). The
 * promise is removed rather than the aliasing, because in-place IS the
 * shipped contract. affS/affC keep theirs: the twiddle tables are plan-owned
 * and never alias a data plane. */
static void _zr2c_fold_fwd(const double *z_in,
                           double *X_out,
                           const double *affS, const double *affC,
                           int N, size_t K, size_t zs, size_t xs)
{
    const int half = N / 2;
    for (size_t t = 0; t < K; t++)
    {
        const double *z = z_in + t * zs;
        double *o = X_out + t * xs;
        double z0r = z[0], z0i = z[1];
        int f = 1;
#if defined(__AVX2__)
        {
            const __m256d CONJ = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
            for (; 2 * (f + 3) < half; f += 4)
            {
                int m3 = half - f - 3;
                /* A: bins f..f+3, already interleaved -- ZERO shuffles. */
                __m256d a0 = _mm256_loadu_pd(z + 2 * f);
                __m256d a1 = _mm256_loadu_pd(z + 2 * f + 4);
                /* B: mirror bins, needed REVERSED so each pairs with its
                 * partner.  4-complex reverse over two ymm = a register swap
                 * (free, renaming) plus one lane swap each. */
                __m256d b0 = _mm256_loadu_pd(z + 2 * m3);
                __m256d b1 = _mm256_loadu_pd(z + 2 * m3 + 4);
                __m256d cb0 = _mm256_xor_pd(_mm256_permute2f128_pd(b1, b1, 0x01), CONJ);
                __m256d cb1 = _mm256_xor_pd(_mm256_permute2f128_pd(b0, b0, 0x01), CONJ);
                __m256d t0 = _mm256_sub_pd(a0, cb0);      /* t = A - conj(B) */
                __m256d t1 = _mm256_sub_pd(a1, cb1);
                __m256d S4 = _mm256_loadu_pd(affS + f);
                __m256d C4 = _mm256_loadu_pd(affC + f);
                __m256d nC4 = _mm256_sub_pd(_mm256_setzero_pd(), C4);
                __m256d wr0 = _mm256_permute4x64_pd(S4, 0x50);
                __m256d wr1 = _mm256_permute4x64_pd(S4, 0xFA);
                __m256d wi0 = _mm256_permute4x64_pd(nC4, 0x50);
                __m256d wi1 = _mm256_permute4x64_pd(nC4, 0xFA);
                __m256d ts0 = _mm256_permute_pd(t0, 0x5);
                __m256d ts1 = _mm256_permute_pd(t1, 0x5);
                __m256d x0 = _mm256_fmaddsub_pd(wr0, t0, _mm256_mul_pd(wi0, ts0));
                __m256d x1 = _mm256_fmaddsub_pd(wr1, t1, _mm256_mul_pd(wi1, ts1));
                /* X[f] = conj(B) + x -- stores straight out, no shuffle. */
                _mm256_storeu_pd(o + 2 * f,     _mm256_add_pd(cb0, x0));
                _mm256_storeu_pd(o + 2 * f + 4, _mm256_add_pd(cb1, x1));
                /* X[m] = conj(A - x), reversed back to memory order. */
                __m256d m0 = _mm256_xor_pd(_mm256_sub_pd(a0, x0), CONJ);
                __m256d m1 = _mm256_xor_pd(_mm256_sub_pd(a1, x1), CONJ);
                _mm256_storeu_pd(o + 2 * m3,     _mm256_permute2f128_pd(m1, m1, 0x01));
                _mm256_storeu_pd(o + 2 * m3 + 4, _mm256_permute2f128_pd(m0, m0, 0x01));
            }
        }
#endif
        for (; 2 * f < half; f++)
        {
            int m = half - f;
            double Ar = z[2 * f], Ai = z[2 * f + 1];
            double Br = z[2 * m], Bi = z[2 * m + 1];
            double S = affS[f], C = affC[f];
            double t1 = Ar - Br, t2 = Ai + Bi;
            double xr = S * t1 + C * t2, xi = S * t2 - C * t1;
            o[2 * f] = Br + xr; o[2 * f + 1] = xi - Bi;
            o[2 * m] = Ar - xr; o[2 * m + 1] = xi - Ai;
        }
        if ((half & 1) == 0)
        { /* center bin: X = conj(Z) */
            int cb = half / 2;
            double cr = z[2 * cb], cim = z[2 * cb + 1];
            o[2 * cb] = cr; o[2 * cb + 1] = -cim;
        }
        /* DC / Nyquist last (in-place: z[0..1] were saved up front) */
        o[0] = z0r + z0i; o[1] = 0.0;
        o[2 * half] = z0r - z0i; o[2 * half + 1] = 0.0;
    }
}

/* backward fold: X (CCE, N/2+1 complex) -> Zhat (N/2 complex, natural),
 * scaled 2x so IL c2c_bwd(N/2) (unnormalized, x N/2) lands on N*x.
 * Zhat may alias X (in-place). X[0].im / X[N/2].im are never read. */
/* Same aliasing contract as _zr2c_fold_fwd above: called (sre, dre) with
 * dre == sre on every in-place c2r, so no __restrict__ on the data planes. */
static void _zr2c_fold_bwd(const double *X_in,
                           double *z_out,
                           const double *bwdS, const double *bwdC,
                           int N, size_t K, size_t xs, size_t zs)
{
    const int half = N / 2;
    for (size_t t = 0; t < K; t++)
    {
        const double *x = X_in + t * xs;
        double *o = z_out + t * zs;
        double X0 = x[0], XN = x[2 * half];
        int f = 1;
#if defined(__AVX2__)
        {
            const __m256d CONJ = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
            for (; 2 * (f + 3) < half; f += 4)
            {
                int m3 = half - f - 3;
                __m256d a0 = _mm256_loadu_pd(x + 2 * f);        /* F */
                __m256d a1 = _mm256_loadu_pd(x + 2 * f + 4);
                __m256d b0 = _mm256_loadu_pd(x + 2 * m3);       /* M, reversed */
                __m256d b1 = _mm256_loadu_pd(x + 2 * m3 + 4);
                __m256d cm0 = _mm256_xor_pd(_mm256_permute2f128_pd(b1, b1, 0x01), CONJ);
                __m256d cm1 = _mm256_xor_pd(_mm256_permute2f128_pd(b0, b0, 0x01), CONJ);
                __m256d t0 = _mm256_sub_pd(a0, cm0);     /* t  = F - conj(M) */
                __m256d t1 = _mm256_sub_pd(a1, cm1);
                __m256d e0 = _mm256_add_pd(a0, cm0);     /* Ep = F + conj(M) */
                __m256d e1 = _mm256_add_pd(a1, cm1);
                /* conj(w) = (sin, -cos), banked at create. cy = t*conj(w)
                 * is exactly conj(conj(t)*w), so BOTH conjugations fall out
                 * of the loop -- same products, same signs, bitwise equal. */
                __m256d s4 = _mm256_loadu_pd(bwdS + f);
                __m256d c4 = _mm256_loadu_pd(bwdC + f);
                __m256d wr0 = _mm256_permute4x64_pd(s4, 0x50);
                __m256d wr1 = _mm256_permute4x64_pd(s4, 0xFA);
                __m256d wi0 = _mm256_permute4x64_pd(c4, 0x50);
                __m256d wi1 = _mm256_permute4x64_pd(c4, 0xFA);
                __m256d ts0 = _mm256_permute_pd(t0, 0x5);
                __m256d ts1 = _mm256_permute_pd(t1, 0x5);
                __m256d cy0 = _mm256_fmaddsub_pd(wr0, t0, _mm256_mul_pd(wi0, ts0));
                __m256d cy1 = _mm256_fmaddsub_pd(wr1, t1, _mm256_mul_pd(wi1, ts1));
                _mm256_storeu_pd(o + 2 * f,     _mm256_sub_pd(e0, cy0));
                _mm256_storeu_pd(o + 2 * f + 4, _mm256_sub_pd(e1, cy1));
                __m256d zm0 = _mm256_xor_pd(_mm256_add_pd(e0, cy0), CONJ);
                __m256d zm1 = _mm256_xor_pd(_mm256_add_pd(e1, cy1), CONJ);
                _mm256_storeu_pd(o + 2 * m3,     _mm256_permute2f128_pd(zm1, zm1, 0x01));
                _mm256_storeu_pd(o + 2 * m3 + 4, _mm256_permute2f128_pd(zm0, zm0, 0x01));
            }
        }
#endif
        for (; 2 * f < half; f++)
        {
            int m = half - f;
            double Fr = x[2 * f], Fi = x[2 * f + 1];
            double Mr = x[2 * m], Mi = x[2 * m + 1];
            double s = bwdS[f], c = -bwdC[f];   /* bwdC banks -cos */
            double t1 = Fr - Mr, t2 = Fi + Mi;
            double yr = c * t2 + s * t1, yi = c * t1 - s * t2;
            double Epr = Fr + Mr, Epi = Fi - Mi;
            o[2 * f] = Epr - yr; o[2 * f + 1] = Epi + yi;
            o[2 * m] = Epr + yr; o[2 * m + 1] = yi - Epi;
        }
        if ((half & 1) == 0)
        { /* center: Zhat = 2*conj(X) */
            int cb = half / 2;
            double cr = x[2 * cb], cim = x[2 * cb + 1];
            o[2 * cb] = 2.0 * cr; o[2 * cb + 1] = -2.0 * cim;
        }
        o[0] = X0 + XN; o[1] = X0 - XN;   /* DC/Nyq (pads saved up front) */
    }
}

/* Mixed-radix digit-reversal perm builders (self-contained copies of the
 * r2c.h pair, so the zr2c route never touches split machinery). DIT: factor
 * order as listed. DIF: factor order REVERSED (a DIF-forward inner emits the
 * reversed-factor digit reversal — verified in r2c.h's dif_order_probe note).
 * Contract produced: iperm[slot] = freq, perm[freq] = slot. Which convention
 * a served cascade plan uses is decided BY THE GATE, never assumed. */
static void _zr2c_perm_dit(const int *factors, int nf, int N, int *perm, int *iperm)
{
    for (int n = 0; n < N; n++)
    {
        int idx = n, rev = 0, rp = 1;
        for (int s = 0; s < nf; s++)
        {
            int R = factors[s];
            rev += (idx % R) * (N / (rp * R));
            idx /= R; rp *= R;
        }
        perm[n] = rev;
    }
    for (int n = 0; n < N; n++) iperm[perm[n]] = n;
}
static void _zr2c_perm_dif(const int *factors, int nf, int N, int *perm, int *iperm)
{
    for (int s = 0; s < N; s++)
    {
        int idx = s, rev = 0, rp = 1;
        for (int k = nf - 1; k >= 0; k--)
        {
            int R = factors[k];
            rev += (idx % R) * (N / (rp * R));
            idx /= R; rp *= R;
        }
        iperm[s] = rev;
    }
    for (int s = 0; s < N; s++) perm[iperm[s]] = s;
}

/* ── PERM-AWARE variants (owner directive #2, 2026-08-13) ────────────────
 * For halves >= 2048 the interior belongs to the CASCADE; its strongest form
 * emits SCRAMBLED order. These folds consume/produce the scrambled slot
 * layout directly — sequential PRIMARY stream over scratch slots, scattered
 * mirror (exactly `_r2c_postprocess`'s access pattern) — killing BOTH the
 * deinterleave and the ordering conversion. Contract: iperm[slot] = freq,
 * perm[freq] = slot, mutually inverse; ANY such pair works (gated with a
 * random permutation). v1 is SCALAR on purpose: scrambled orders break the
 * natural folds' contiguous 4-bin blocks, and whether a gathered SIMD form
 * pays is a RACE question, not an assumption — the natural-stfn + plain-fold
 * arm is the vectorized competitor in the same pool.
 *
 * fwd: Zs (scrambled slots, interleaved) -> X (natural CCE).  NOT in-place
 * (reads scattered mirrors after writes would collide across slot order).
 * bwd: X (natural CCE) -> Zs_hat (scrambled slots, x2) for the cascade bwd,
 * which consumes the same scrambled order its fwd emits.  NOT in-place. */
static void _zr2c_fold_fwd_perm(const double *__restrict__ zs_in,
                                double *__restrict__ X_out,
                                const double *affS, const double *affC,
                                const int *iperm, const int *perm,
                                int N, size_t K, size_t zs, size_t xs)
{
    const int half = N / 2;
    for (size_t t = 0; t < K; t++)
    {
        const double *z = zs_in + t * zs;
        double *o = X_out + t * xs;
        {   /* DC / Nyquist: frequency 0 lives at slot perm[0] */
            int p0 = perm[0];
            double z0r = z[2 * p0], z0i = z[2 * p0 + 1];
            o[0] = z0r + z0i; o[1] = 0.0;
            o[2 * half] = z0r - z0i; o[2 * half + 1] = 0.0;
        }
        for (int p = 0; p < half; p++)
        {
            int f = iperm[p];
            if (f == 0) continue;
            int m = half - f;
            if (f > m) continue;                    /* partner already done */
            double Ar = z[2 * p], Ai = z[2 * p + 1];
            if (f == m) { o[2 * f] = Ar; o[2 * f + 1] = -Ai; continue; }
            int q = perm[m];
            double Br = z[2 * q], Bi = z[2 * q + 1];
            double S = affS[f], C = affC[f];
            double t1 = Ar - Br, t2 = Ai + Bi;
            double xr = S * t1 + C * t2, xi = S * t2 - C * t1;
            o[2 * f] = Br + xr; o[2 * f + 1] = xi - Bi;
            o[2 * m] = Ar - xr; o[2 * m + 1] = xi - Ai;
        }
    }
}

static void _zr2c_fold_bwd_perm(const double *__restrict__ X_in,
                                double *__restrict__ zs_out,
                                const double *bwdS, const double *bwdC,
                                const int *iperm, const int *perm,
                                int N, size_t K, size_t xs, size_t zs)
{
    const int half = N / 2;
    for (size_t t = 0; t < K; t++)
    {
        const double *x = X_in + t * xs;
        double *o = zs_out + t * zs;
        {   /* DC/Nyquist -> slot perm[0] */
            int p0 = perm[0];
            double X0 = x[0], XN = x[2 * half];
            o[2 * p0] = X0 + XN; o[2 * p0 + 1] = X0 - XN;
        }
        for (int p = 0; p < half; p++)
        {
            int f = iperm[p];
            if (f == 0) continue;
            int m = half - f;
            if (f > m) continue;
            if (f == m)
            {
                o[2 * p] = 2.0 * x[2 * f]; o[2 * p + 1] = -2.0 * x[2 * f + 1];
                continue;
            }
            int q = perm[m];
            double Fr = x[2 * f], Fi = x[2 * f + 1];
            double Mr = x[2 * m], Mi = x[2 * m + 1];
            double s = bwdS[f], c = -bwdC[f];   /* bwdC banks -cos */
            double t1 = Fr - Mr, t2 = Fi + Mi;
            double yr = c * t2 + s * t1, yi = c * t1 - s * t2;
            double Epr = Fr + Mr, Epi = Fi - Mi;
            o[2 * p] = Epr - yr; o[2 * p + 1] = Epi + yi;
            o[2 * q] = Epr + yr; o[2 * q + 1] = yi - Epi;
        }
    }
}


/* ═══════════════════════════════════════════════════════════════════════
 * SUPERSEDED 2026-08-21 — the DE-INTERLEAVING folds, kept as a reminder.
 *
 * These are the kernels the interleaved-native versions above replaced.
 * They split each packed pair into Ar/Ai/Br/Bi, computed on separated
 * planes, then re-interleaved to store.
 *
 * WHY THEY LOST (measured, not reasoned):
 *   - 20 shuffles per 4-bin iteration, and ALL of them issue on port 5 of
 *     Raptor Cove — in-lane vunpcklpd/vunpckhpd ymm included.  That was the
 *     surprise: an earlier model assumed only the 12 lane-crossing ones were
 *     p5-bound and predicted 12 cyc/iter.
 *   - Measured 20.1 cyc/iter at N=256 through N=16384 — DEAD FLAT, which is
 *     the signature of an issue-port limit rather than a cache effect.
 *   - The interleaved form needs 10 (it pays 6 shuffles at the complex
 *     multiply but saves 16 on the un/re-interleave), giving -48% fwd /
 *     -38% bwd below the L1 cliff, ~3-9% above it where memory binds.
 *
 * KEEP THIS BLOCK.  It is the reference for the two things that are easy to
 * get wrong if the fold is ever rewritten again: (1) the exact bin/mirror
 * pairing and the 0x1B reversal convention, and (2) that "fewer lane-crossing
 * shuffles" is the WRONG objective on this uarch — total shuffle count is.
 * ═══════════════════════════════════════════════════════════════════════ */
#if 0
/* ---- forward fold, de-interleaving form (superseded) ---- */
#if defined(__AVX2__)
        for (; 2 * (f + 3) < half; f += 4)
        {
            int m3 = half - f - 3;
            __m256d a0 = _mm256_loadu_pd(z + 2 * f);
            __m256d a1 = _mm256_loadu_pd(z + 2 * f + 4);
            __m256d t0 = _mm256_permute2f128_pd(a0, a1, 0x20);
            __m256d t1 = _mm256_permute2f128_pd(a0, a1, 0x31);
            __m256d Ar = _mm256_unpacklo_pd(t0, t1), Ai = _mm256_unpackhi_pd(t0, t1);
            __m256d b0 = _mm256_loadu_pd(z + 2 * m3);
            __m256d b1 = _mm256_loadu_pd(z + 2 * m3 + 4);
            __m256d u0 = _mm256_permute2f128_pd(b0, b1, 0x20);
            __m256d u1 = _mm256_permute2f128_pd(b0, b1, 0x31);
            __m256d Br = _mm256_permute4x64_pd(_mm256_unpacklo_pd(u0, u1), 0x1B);
            __m256d Bi = _mm256_permute4x64_pd(_mm256_unpackhi_pd(u0, u1), 0x1B);
            __m256d S = _mm256_loadu_pd(affS + f);
            __m256d C = _mm256_loadu_pd(affC + f);
            __m256d t1v = _mm256_sub_pd(Ar, Br), t2v = _mm256_add_pd(Ai, Bi);
            __m256d xr = _mm256_fmadd_pd(S, t1v, _mm256_mul_pd(C, t2v));
            __m256d xi = _mm256_fmsub_pd(S, t2v, _mm256_mul_pd(C, t1v));
            __m256d Xfr = _mm256_add_pd(Br, xr), Xfi = _mm256_sub_pd(xi, Bi);
            __m256d Xmr = _mm256_sub_pd(Ar, xr), Xmi = _mm256_sub_pd(xi, Ai);
            __m256d lo = _mm256_unpacklo_pd(Xfr, Xfi), hi = _mm256_unpackhi_pd(Xfr, Xfi);
            _mm256_storeu_pd(o + 2 * f,     _mm256_permute2f128_pd(lo, hi, 0x20));
            _mm256_storeu_pd(o + 2 * f + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
            __m256d Rr = _mm256_permute4x64_pd(Xmr, 0x1B);
            __m256d Ri = _mm256_permute4x64_pd(Xmi, 0x1B);
            __m256d ml = _mm256_unpacklo_pd(Rr, Ri), mh = _mm256_unpackhi_pd(Rr, Ri);
            _mm256_storeu_pd(o + 2 * m3,     _mm256_permute2f128_pd(ml, mh, 0x20));
            _mm256_storeu_pd(o + 2 * m3 + 4, _mm256_permute2f128_pd(ml, mh, 0x31));
        }
#endif
/* ---- backward fold, de-interleaving form (superseded) ---- */
#if defined(__AVX2__)
        {
            const __m256d one = _mm256_set1_pd(1.0), two = _mm256_set1_pd(2.0);
            for (; 2 * (f + 3) < half; f += 4)
            {
                int m3 = half - f - 3;
                __m256d a0 = _mm256_loadu_pd(x + 2 * f);
                __m256d a1 = _mm256_loadu_pd(x + 2 * f + 4);
                __m256d t0 = _mm256_permute2f128_pd(a0, a1, 0x20);
                __m256d t1 = _mm256_permute2f128_pd(a0, a1, 0x31);
                __m256d Fr = _mm256_unpacklo_pd(t0, t1), Fi = _mm256_unpackhi_pd(t0, t1);
                __m256d b0 = _mm256_loadu_pd(x + 2 * m3);
                __m256d b1 = _mm256_loadu_pd(x + 2 * m3 + 4);
                __m256d u0 = _mm256_permute2f128_pd(b0, b1, 0x20);
                __m256d u1 = _mm256_permute2f128_pd(b0, b1, 0x31);
                __m256d Mr = _mm256_permute4x64_pd(_mm256_unpacklo_pd(u0, u1), 0x1B);
                __m256d Mi = _mm256_permute4x64_pd(_mm256_unpackhi_pd(u0, u1), 0x1B);
                __m256d S = _mm256_loadu_pd(affS + f);
                __m256d C = _mm256_loadu_pd(affC + f);
                __m256d c = _mm256_mul_pd(two, C);                    /* 2*C~      */
                __m256d s = _mm256_fnmadd_pd(two, S, one);            /* 1 - 2*S~  */
                __m256d t1v = _mm256_sub_pd(Fr, Mr), t2v = _mm256_add_pd(Fi, Mi);
                __m256d yr = _mm256_fmadd_pd(c, t2v, _mm256_mul_pd(s, t1v));
                __m256d yi = _mm256_fmsub_pd(c, t1v, _mm256_mul_pd(s, t2v));
                __m256d Epr = _mm256_add_pd(Fr, Mr), Epi = _mm256_sub_pd(Fi, Mi);
                __m256d Zfr = _mm256_sub_pd(Epr, yr), Zfi = _mm256_add_pd(Epi, yi);
                __m256d Zmr = _mm256_add_pd(Epr, yr), Zmi = _mm256_sub_pd(yi, Epi);
                __m256d lo = _mm256_unpacklo_pd(Zfr, Zfi), hi = _mm256_unpackhi_pd(Zfr, Zfi);
                _mm256_storeu_pd(o + 2 * f,     _mm256_permute2f128_pd(lo, hi, 0x20));
                _mm256_storeu_pd(o + 2 * f + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
                __m256d Rr = _mm256_permute4x64_pd(Zmr, 0x1B);
                __m256d Ri = _mm256_permute4x64_pd(Zmi, 0x1B);
                __m256d ml = _mm256_unpacklo_pd(Rr, Ri), mh = _mm256_unpackhi_pd(Rr, Ri);
                _mm256_storeu_pd(o + 2 * m3,     _mm256_permute2f128_pd(ml, mh, 0x20));
                _mm256_storeu_pd(o + 2 * m3 + 4, _mm256_permute2f128_pd(ml, mh, 0x31));
            }
        }
#endif
#endif /* 0 — superseded de-interleaving folds */

#endif /* VFFT_ZR2C_H */
