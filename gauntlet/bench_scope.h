/* bench_scope.h -- the gauntlet's measurement scope: THE LIBRARY'S.
 *
 * The pin to a P-core, the SMT-sibling guard, the priority and the
 * machine-wide measurement lock are vfft.h's ("the measurement scope"): the
 * conditions vfft_create's own races run under. A gauntlet program enters
 * that scope once and times every engine inside it, so a benched number and a
 * raced verdict are measured the same way. This header holds no pin or guard
 * code, only the gauntlet's environment switches mapped onto the library's
 * settings:
 *
 *   VFFT_BENCH_GUARD=0      no sibling guard; =pause holds the sibling with a
 *                           PAUSE spinner (a ~12% level cost to every engine
 *                           alike, ratios unaffected) on a host with neither
 *                           TPAUSE nor MWAITX
 *   VFFT_PCORE_MASK=<mask>  the threaded protocol's process mask; 0 = do not
 *                           confine (the control for a mask-distorted threaded
 *                           engine). Unset: one logical CPU per P-core, read
 *                           from the machine by the library.
 */
#ifndef VFFT_GAUNTLET_BENCH_SCOPE_H
#define VFFT_GAUNTLET_BENCH_SCOPE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

static int bench_guard_mode(void)
{
    const char *g = getenv("VFFT_BENCH_GUARD");
    return (g && !strcmp(g, "0")) ? VFFT_MEASURE_GUARD_OFF
         : (g && !strcmp(g, "pause")) ? VFFT_MEASURE_GUARD_PAUSE : VFFT_MEASURE_GUARD_DEFAULT;
}

/* Enter the measurement scope for the rest of the process: pinned to `core`
 * (-1 = the library's own measuring core), the process at high priority when
 * asked, the sibling held as `guard` says (VFFT_MEASURE_GUARD_*). A second
 * call replaces the first scope. Returns the scope's flags (0 = clean). */
static int bench_scope(int core, int process_priority, int guard)
{
    static int open = 0;
    vfft_measure_config_t mc;
    char d[320];
    int f;
    if (open)
        vfft_measure_end();
    memset(&mc, 0, sizeof mc);
    mc.pin = core >= 0 ? VFFT_MEASURE_PIN_CORE : VFFT_MEASURE_PIN_DEFAULT;
    mc.pin_core = core;
    mc.guard = guard;
    mc.priority = process_priority ? VFFT_MEASURE_PRIORITY_PROCESS : VFFT_MEASURE_PRIORITY_LEAVE;
    vfft_measure_configure(&mc);
    f = vfft_measure_begin();
    open = 1;
    printf("# measurement scope: %s\n", vfft_measure_describe(d, sizeof d));
    if (f & VFFT_MEASURE_UNPINNED)
        fprintf(stderr, "warn: the caller is not pinned to a P-core (asked for cpu %d)\n", core);
    if (f & VFFT_MEASURE_CONTENDED)
        fprintf(stderr, "warn: another VectorFFT measurement is running on this machine\n");
    return f;
}

/* The control for "did the pin itself move a number?": nothing is pinned,
 * guarded or raised, by this program or by the library's races. */
static void bench_scope_lifted(void)
{
    vfft_measure_config_t mc;
    memset(&mc, 0, sizeof mc);
    mc.pin = VFFT_MEASURE_PIN_OFF;
    mc.guard = VFFT_MEASURE_GUARD_OFF;
    mc.priority = VFFT_MEASURE_PRIORITY_LEAVE;
    vfft_measure_configure(&mc);
}

/* THE THREADED PROTOCOL'S CONFINEMENT: the process on one logical CPU per
 * P-core, for every engine's threads, before any of them exist (Intel OpenMP
 * reads the mask at init; on Linux threads inherit their creator's). The
 * caller then pins itself to logical 0, the core the library's pool reserves
 * for it. */
static void bench_pin_pcores(void)
{
    const char *e = getenv("VFFT_PCORE_MASK");
    const unsigned long long want = e ? strtoull(e, NULL, 0) : 0ull;
    unsigned long long got;
    if (e && want == 0)
    {
        printf("# process affinity UNSET (VFFT_PCORE_MASK=0) -- threads float over every logical CPU incl. E-cores\n");
        return;
    }
    got = vfft_measure_confine(want);
    if (!got)
        fprintf(stderr, "pin: the process could not be confined (mask 0x%llx) -- threads may land on E-cores\n", want);
    else
        printf("# process affinity = 0x%llx (%s)\n", got, e ? "VFFT_PCORE_MASK" : "one logical CPU per P-core");
}

#endif /* VFFT_GAUNTLET_BENCH_SCOPE_H */
