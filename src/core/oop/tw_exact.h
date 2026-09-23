/* tw_exact.h -- cos/sin(2*pi*p/n) rounded ONCE to double, from the integers.
 *
 * THE ONE TRIG SOURCE OF EVERY CREATE-TIME TABLE. A twiddle, a chirp or a
 * stage record built as cos(a)/sin(a) of an angle a = 2*pi*p/n that was
 * already formed in double carries the angle's rounding (about 2|a|u) into
 * the value, hundreds of ulp where the value is small, and mirror angles
 * (k and n-k, k and n/2-k) come out as different doubles. Measured on the
 * 1D interleaved front door, N = 2..4096: that rounding is most of the
 * accuracy gap to the reference vendor library (median relative L2 error
 * 5.80e-16 against 4.05e-16; with the tables from here, 3.90e-16).
 *
 * Contract: vfft_cs2pi_exact(p, n, &c, &s) gives c = cos(2*pi*p/n),
 * s = sin(2*pi*p/n) for integers p, n (n > 0, any p):
 *   1. p is reduced mod n in integers;
 *   2. the fraction of a turn t/(8n), t = 8p, is folded into [0, 1/8] with
 *      EXACT integer comparisons, each fold recording a sign flip or a
 *      cos/sin swap;
 *   3. only then is the angle pi*t/(4n) (at most pi/4) formed and evaluated;
 *   4. the result is rounded once to double and the recorded swap and flips
 *      are applied.
 * Mirror angles fold to the same t and come out bit-identical. A caller
 * whose angle carries a sign (-2*pi*p/n) negates s itself; one whose angle is
 * pi*m/n passes (m, 2n).
 *
 * Evaluation: 80-bit long double (cosl/sinl), which is what gcc gives on
 * every host this library builds for (mingw on Windows, gcc on Linux), so
 * near-ties aside the result is the correctly rounded double. Under an MSVC
 * ABI front end (cl, clang-cl, ICX on Windows) long double is 64 bits and
 * the octant fold alone still holds; the double-double evaluation with
 * precise FP semantics pinned on this function is the portable form for that
 * toolchain. No build guard: the fold is exact on every toolchain, and the
 * evaluation degrades gracefully.
 *
 * Create-time only. Every caller builds a table inside create; nothing here
 * runs on an execute path. Callers (all under src/core): il2p.h (pair and
 * chain3 twiddles), il_flatdit.h (the flat DIT's five tables), ztt.h (the
 * 2^a*odd moduli and the fine table above the octave), il_prime.h (Bluestein
 * chirp, Rader table), k1_fourstep.h (the coarse and fine four-step records),
 * transforms/fft2d/il2d_cols.h (the 2D column chirp and stage twiddles).
 * NOT the pow2 ZTURN-T: its streams expand from the baked quarter-wave
 * ztt_qw16384.h, which stays as shipped. A re-bake of that table from this
 * fold was measured 2026-09-24 and rejected: the table's entries get closer
 * to the exact sine (1,016 of 4,097 by 1 ulp) and a textbook radix-2 or
 * radix-4 FFT on it reads 4-8% more accurate, but the pow2 ZTURN-T engine
 * reads 2-3% LESS accurate at 2048 and 4096 (8 of 8 draws) and the Bluestein
 * route, whose convolution runs that engine, 10-20% less (2,232 of 2,718
 * prime-route lengths worse). The mechanism is not identified. */
#ifndef VFFT_TW_EXACT_H
#define VFFT_TW_EXACT_H

#include <math.h>

static inline void vfft_cs2pi_exact(long long p, long long n, double *c, double *s)
{
    long long t;
    int flip_s = 0, flip_c = 0, swap = 0;
    long double th;
    double cv, sv;

    p %= n;
    if (p < 0) p += n;
    t = 8 * p;                                        /* t / (8n) of a turn, t in [0, 8n)   */
    if (t > 4 * n) { t = 8 * n - t; flip_s = 1; }     /* (1/2, 1):  cos even, sin odd        */
    if (t > 2 * n) { t = 4 * n - t; flip_c = 1; }     /* (1/4, 1/2]: cos(pi - x) = -cos x    */
    if (t > n)     { t = 2 * n - t; swap = 1; }       /* (1/8, 1/4]: cos(pi/2 - x) = sin x   */

    th = 3.141592653589793238462643383279502884L * (long double)t / (4.0L * (long double)n);
    cv = (double)cosl(th);
    sv = (double)sinl(th);
    if (swap) { const double tmp = cv; cv = sv; sv = tmp; }
    if (flip_c) cv = -cv;
    if (flip_s) sv = -sv;
    *c = cv;
    *s = sv;
}

#endif /* VFFT_TW_EXACT_H */
