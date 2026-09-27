/* race_timing.h — THE monotonic clock: every racer and planner times with it.
 *
 * vfft_now_ns() is nanoseconds as a double, for interval measurement only.
 * Windows reads QueryPerformanceCounter directly: MSVC-ABI toolchains (ICX,
 * clang-cl) have no clock_gettime, and on mingw-w64 CLOCK_MONOTONIC is itself
 * QPC-backed, so this is the same counter either way. Elsewhere
 * clock_gettime(CLOCK_MONOTONIC).
 *
 * The utilities merge (2026-09-27) folded the split planner's
 * vfft_now_ns (planning/dp_planner.h) into this one; on Linux the two
 * were already the same computation. The median lives beside the race body in
 * support/race.h (vfft_race_median).
 */
#ifndef VFFT_SUPPORT_RACE_TIMING_H
#define VFFT_SUPPORT_RACE_TIMING_H

#ifdef _WIN32
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
static inline double vfft_now_ns(void)
{
    static LARGE_INTEGER freq = {0};
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart / (double)freq.QuadPart * 1e9;
}
#else
#  include <time.h>
static inline double vfft_now_ns(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}
#endif

#endif /* VFFT_SUPPORT_RACE_TIMING_H */
