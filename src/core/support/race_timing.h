/* race_timing.h — the measurement floor shared by the in-process racers.
 *
 * A monotonic clock and a fixed-size median. That is deliberately all: this is
 * step 5 of docs/design/refactor_migration_plan.md, the PILOT move, sized so
 * that every rung of the safety ladder can be exercised end to end on something
 * that reverts with one `git checkout`.
 *
 * WHY THESE TWO
 * -------------
 * vfft.c contains twelve racers with a drifted protocol between them - three
 * round counts (six use 5, four use 9, one uses 3), four reps formulas, three
 * median implementations and three clock spellings. Collapsing that protocol is
 * OUT OF SCOPE and stays that way: no check in the harness can tell whether a
 * unified protocol still picks the same winner, and re-racing to find out is
 * forbidden during development. What CAN move safely is the floor underneath
 * it - the primitives themselves, byte for byte, changing nothing about how any
 * racer uses them.
 *
 * FLOOR-LEGAL BY CONSTRUCTION
 * ---------------------------
 * Depends on nothing outside <time.h>. Touches no plan type, no wisdom type, no
 * engine header. In particular it does NOT pull engine/stride_executor.h, which
 * redefines executor symbols and is excluded from the build by design.
 *
 * NO MUTABLE FILE-SCOPE STATE, EVER
 * ---------------------------------
 * A `static` in a header is one copy PER INCLUDER. `_il_ab_runs` - the race
 * budget counter these primitives are used alongside - therefore stays in
 * vfft.c and is not moved here. The same rule is why the harness counters are
 * tentative definitions in vfft.c rather than statics in a header: split state
 * lets an accessor read a different object than the increment writes, and the
 * result is a counter that silently reads zero while everything looks correct.
 *
 * ON THE CLOCK
 * ------------
 * clock_gettime(CLOCK_MONOTONIC) and vfft_proto_now_ns (QueryPerformanceCounter
 * on Windows) are both in use across the racers. Measured on this toolchain they
 * have the SAME 100 ns resolution - mingw-w64 maps CLOCK_MONOTONIC onto QPC - so
 * the two spellings are equivalent for interval measurement. That was checked
 * rather than assumed, because if they had differed, unifying them later would
 * silently change what every racer measures.
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

/* Median of exactly 9, in place. Insertion-sorts the whole array rather than
 * selecting, which is irrelevant at n=9 and keeps the body obviously correct.
 * Returns v[4]; _pad_med(v, 9) returns the same element by the same rule, so
 * the two are interchangeable at this size - verified before they were ever
 * described as duplicates. */
static double _il_ab_med9(double *v)
{
    for (int i = 0; i < 9; i++)
        for (int j = i + 1; j < 9; j++)
            if (v[j] < v[i])
            {
                double t = v[i];
                v[i] = v[j];
                v[j] = t;
            }
    return v[4];
}

#endif /* VFFT_SUPPORT_RACE_TIMING_H */
