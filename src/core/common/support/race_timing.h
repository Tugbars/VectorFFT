/* race_timing.h — the monotonic clock the in-process racers time with.
 *
 * Depends on nothing but <time.h>, so any header can include it. On mingw-w64
 * CLOCK_MONOTONIC is backed by QueryPerformanceCounter: the same 100 ns
 * resolution as vfft_proto_now_ns (planning/dp_planner.h), so the two are
 * interchangeable for interval measurement. The median lives beside the race
 * body in support/race.h (vfft_race_median).
 */
#ifndef VFFT_SUPPORT_RACE_TIMING_H
#define VFFT_SUPPORT_RACE_TIMING_H

#include <time.h>

static double _il_ab_now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

#endif /* VFFT_SUPPORT_RACE_TIMING_H */
