/* pi.h — THE value of pi, once (the utilities merge, 2026-09-27).
 *
 *   VFFT_PI    the double, 0x1.921fb54442d18p+1: every twiddle, chirp and
 *              table built in double.
 *   VFFT_PI_L  the long double (x87 80-bit: 0xc.90fdaa22168c235p-2): the
 *              long-double evaluations (tw_exact.h, the IL planner's reference).
 *
 * These are the digits every copy in the tree carried: eight M_PI fallbacks,
 * VFFT_IL2P_PI, VFFT_ZR2C_PI, VFFT_NATORDER_PI, and the literals written
 * inline. The 21-, 35- and 36-digit long-double spellings were checked to
 * round to the same value before they were merged.
 *
 * M_PI is not ISO C; <math.h> may leave it undefined (MSVC-ABI toolchains
 * without _USE_MATH_DEFINES). Core spells pi VFFT_PI. The fallback below keeps
 * code outside src/core that writes M_PI (the benches) building there, as the
 * per-file fallbacks did. It MUST stay the bare literal: glibc's <math.h>
 * defines M_PI unguarded with exactly these tokens, so a later <math.h> is an
 * identical redefinition, not a warning.
 */
#ifndef VFFT_COMMON_PI_H
#define VFFT_COMMON_PI_H

#define VFFT_PI   3.14159265358979323846
#define VFFT_PI_L 3.141592653589793238462643383279502884L

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#endif /* VFFT_COMMON_PI_H */
