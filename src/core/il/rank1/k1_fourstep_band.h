/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* k1_fourstep_band.h — the K=1 interleaved four-step's BAND (docs/design/
 * k1_fourstep_design.md): the sizes the engine serves and the
 * side ladders its splits draw from. Self-contained so the benches admit the
 * band without the engine (k1_fourstep.h needs the plan internals). */
#ifndef VFFT_K1_FOURSTEP_BAND_H
#define VFFT_K1_FOURSTEP_BAND_H

#ifdef VFFT_FLOAT
#define VFFT_K1FS_MIN_N 1048576  /* FP32: raced beside ZTURN-T at ITS ceiling (ztt.h) */
#else
#define VFFT_K1FS_MIN_N 262144   /* raced beside ZTURN-T at its ceiling */
#endif
#define VFFT_K1FS_MAX_N 16777216 /* 2^24 = 4096 x 4096: the pow2 side ladder's reach (both
                                  * sides <= 4096); the non-pow2 ladder stops at the same
                                  * N (its sides reach 16384, see below) */

/* the sides a POWER-OF-TWO split may take: every row length is a K=1 cell
 * the door serves natively (the pairs/solos to 1024, ZTURN-T at 2048 and
 * 4096) and every column length a chain the 2D tier builds */
static const int VFFT_K1FS_SIDES[] = { 256, 512, 1024, 2048, 4096 };
#define VFFT_K1FS_NSIDES 5

/* the NON-POWER-OF-TWO ladder (2026-10-12, field trial item 1): a split
 * (N1, N2) of N with both sides in [VFFT_K1FS_SIDE_MIN, VFFT_K1FS_SIDE_MAX],
 * the column side N1 a chain the 2D tier builds and the row side N2 a K=1
 * cell the door serves (vfft_k1fs_splits, k1_fourstep.h, asks both); the
 * VFFT_K1FS_MAX_SPLITS most balanced are the race's arms. The band asks
 * only the divisor question, so the benches admit it without the engine. */
#define VFFT_K1FS_SIDE_MIN 64
#define VFFT_K1FS_SIDE_MAX 16384
#define VFFT_K1FS_MAX_SPLITS 8

/* does a non-pow2 N have a divisor pair in the side range at all */
static inline int vfft_k1fs_has_side_pair(int N)
{
    int d;
    for (d = VFFT_K1FS_SIDE_MIN; d <= VFFT_K1FS_SIDE_MAX && (long)d * d <= (long)N; d++)
        if (N % d == 0 && N / d >= VFFT_K1FS_SIDE_MIN && N / d <= VFFT_K1FS_SIDE_MAX)
            return 1;
    return 0;
}

/* THE BAND: every power of two in [262144, 2^24] (FP32: [1048576, 2^24]), and every other N in
 * (262144, 2^24] with a divisor pair in the side range (the engine's splits
 * may still all refuse -- a side with no chain or no row cell -- and then
 * the cell has no four-step arm; policy admits the family, the family
 * answers for itself) */
static inline int vfft_k1fs_band(int N)
{
    if (N < VFFT_K1FS_MIN_N || N > VFFT_K1FS_MAX_N) return 0;
    if ((N & (N - 1)) == 0) return 1;
    return vfft_k1fs_has_side_pair(N);
}

/* THE PRIME SIDES (owner, 2026-11-20; docs/design/k1_fourstep_prime_sides.md). A composite N with a prime
 * factor above the chain radices splits with one side holding that prime: the 2D child serves it as a
 * column chain led by a Rader stage or as a row the K=1 door serves (its prime cell, its flat chain's Rader
 * stage), and the split is a four-step like any other. Measured 2026-11-20 (14900KF): 2.0-2.3x the
 * whole-length Bluestein at one thread, 6-7x at eight, at 2097151, 4194303 and 4194305. Such an N takes the
 * wider side range: a column side down to VFFT_K1FS_PSIDE_MIN (vfft_k1fs_create's own floor; up to
 * VFFT_K1FS_SIDE_MAX as any column) and a row side up to VFFT_K1FS_PSIDE_ROW_MAX (a row past it would be a
 * four-step itself), so 1048577 = 17 x 61681 and 8388607 = 47 x 178481 have their one split. Above VFFT_K1FS_MIN_N at any thread count; from VFFT_K1FS_PRIME_MIN_N to
 * it for a threaded plan only (there the whole-length plan and the flat chain lose to MKL's split at T > 1
 * and win at one thread: the 2026-11-20 sweep). */
#define VFFT_K1FS_PSIDE_MIN 4
#define VFFT_K1FS_PSIDE_ROW_MAX 262144
#define VFFT_K1FS_PRIME_MIN_N 4096
#define VFFT_K1FS_CHAIN_MAX_PRIME 47   /* the chain kernels' largest prime radix (policy_il.h
                                        * VFFT_POLICY_CHAIN_MAX_PRIME), repeated: this header stands alone */

/* N is composite and has a prime factor above VFFT_K1FS_CHAIN_MAX_PRIME */
static inline int vfft_k1fs_prime_composite(int N)
{
    int m = N, p, small = 0;
    if (N < 4)
        return 0;
    for (p = 2; p <= VFFT_K1FS_CHAIN_MAX_PRIME; p++)
        while (m % p == 0)
        {
            m /= p;
            small = 1;
        }
    if (m == 1)
        return 0;      /* every factor a chain radix */
    if (small)
        return 1;      /* a chain radix beside the factors past them */
    for (p = VFFT_K1FS_CHAIN_MAX_PRIME + 2; (long)p * p <= (long)m; p += 2)
        if (m % p == 0)
            return 1;  /* two factors past the chain radices */
    return 0;          /* N is prime */
}
/* the column sides of the prime side range at N: [lo, hi] -- at least VFFT_K1FS_PSIDE_MIN and so that the row
 * side N / d is at most VFFT_K1FS_PSIDE_ROW_MAX; at most VFFT_K1FS_SIDE_MAX (the 2D column chain's reach) and so
 * that the row side is at least VFFT_K1FS_PSIDE_MIN. The ladder (k1_fourstep.h) scans exactly these. */
static inline void vfft_k1fs_prime_col_range(int N, int *lo, int *hi)
{
    const long l = ((long)N + VFFT_K1FS_PSIDE_ROW_MAX - 1) / VFFT_K1FS_PSIDE_ROW_MAX;
    const long h = (long)N / VFFT_K1FS_PSIDE_MIN;
    *lo = (int)(l > VFFT_K1FS_PSIDE_MIN ? l : VFFT_K1FS_PSIDE_MIN);
    *hi = (int)(h < VFFT_K1FS_SIDE_MAX ? h : VFFT_K1FS_SIDE_MAX);
}
/* a split in the prime side range: a column side d in vfft_k1fs_prime_col_range that divides N */
static inline int vfft_k1fs_has_prime_pair(int N)
{
    int d, lo, hi;
    vfft_k1fs_prime_col_range(N, &lo, &hi);
    for (d = lo; d <= hi; d++)
        if (N % d == 0)
            return 1;
    return 0;
}
/* THE BAND AT A THREAD COUNT: vfft_k1fs_band, and the prime sides -- a composite N with a prime factor past the
 * chain radices and a split in the prime side range, in (VFFT_K1FS_MIN_N, VFFT_K1FS_MAX_N] at any T and in
 * [VFFT_K1FS_PRIME_MIN_N, VFFT_K1FS_MIN_N] at T > 1 */
static inline int vfft_k1fs_band_t(int N, int T)
{
    if (vfft_k1fs_band(N))
        return 1;
    if (N > VFFT_K1FS_MAX_N || N < VFFT_K1FS_PRIME_MIN_N || (N <= VFFT_K1FS_MIN_N && T < 2))
        return 0;
    return vfft_k1fs_prime_composite(N) && vfft_k1fs_has_prime_pair(N);
}

#endif /* VFFT_K1_FOURSTEP_BAND_H */
