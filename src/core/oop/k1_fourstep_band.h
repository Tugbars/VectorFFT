/* k1_fourstep_band.h — the K=1 interleaved four-step's BAND (docs/design/
 * k1_fourstep_design.md): the sizes the engine serves and the
 * side ladder its splits draw from. Self-contained so the benches admit the
 * band without the engine (k1_fourstep.h needs the plan internals). */
#ifndef VFFT_K1_FOURSTEP_BAND_H
#define VFFT_K1_FOURSTEP_BAND_H

#define VFFT_K1FS_MIN_N 262144   /* raced beside ZTURN-T at its ceiling */
#define VFFT_K1FS_MAX_N 16777216 /* 2^24 = 4096 x 4096: the side ladder's reach (both
                                  * sides <= 4096); a larger N needs a longer side */

static inline int vfft_k1fs_band(int N)
{
    return N >= VFFT_K1FS_MIN_N && N <= VFFT_K1FS_MAX_N && (N & (N - 1)) == 0;
}

/* the sides a split may take: every row length is a K=1 cell the door
 * serves natively (the pairs/solos to 1024, ZTURN-T at 2048 and 4096) and
 * every column length a chain the 2D tier builds */
static const int VFFT_K1FS_SIDES[] = { 256, 512, 1024, 2048, 4096 };
#define VFFT_K1FS_NSIDES 5

#endif /* VFFT_K1_FOURSTEP_BAND_H */
