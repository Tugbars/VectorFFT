/* k1_fourstep_band.h — the K=1 interleaved four-step's BAND (docs/design/
 * k1_fourstep_design.md): the sizes the engine serves and the
 * side ladders its splits draw from. Self-contained so the benches admit the
 * band without the engine (k1_fourstep.h needs the plan internals). */
#ifndef VFFT_K1_FOURSTEP_BAND_H
#define VFFT_K1_FOURSTEP_BAND_H

#define VFFT_K1FS_MIN_N 262144   /* raced beside ZTURN-T at its ceiling */
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

/* THE BAND: every power of two in [262144, 2^24], and every other N in
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

#endif /* VFFT_K1_FOURSTEP_BAND_H */
