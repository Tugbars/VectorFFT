/* sibling_guard.h -- THE SIBLING GUARD (2026-09-21), shared since 2026-09-23 by
 * the gauntlet's bench and its calibrate probe (recal_1d_probe: the front
 * door's races run in that process, and unguarded they run in the same
 * two-speed lottery the bench had). The gauntlet benched the same banked
 * chain3 plan 1.1-1.5x slower than the planner had raced it, sticky for tens
 * of ms after any idle (the 300 ms cool before an arm), with the core clock,
 * the memory and the buffer placement all verified unchanged: after the timed
 * thread idles, the OS parks another process's thread on the SMT SIBLING of
 * its core and leaves it there for a while, and a high-IPC kernel sharing the
 * core runs at 60% (chain3_skew_probe.c: a busy loop pinned to the sibling
 * reproduces 1.47x; a thread of our own holding the sibling removes the mode).
 * A PAUSE spinner costs the timed thread 12%; TPAUSE into C0.2 (WAITPKG)
 * costs nothing measurable and still counts as busy to the scheduler, so no
 * foreign thread lands there. The guard lives for the process; every timed
 * window runs with the sibling reserved. Hosts without WAITPKG (Zen 4) run
 * unguarded; VFFT_BENCH_GUARD=0 lifts it.
 *
 * Use: bench_guard_sibling(cpu) once, after pinning the calling thread to cpu. */
#ifndef VFFT_GAUNTLET_SIBLING_GUARD_H
#define VFFT_GAUNTLET_SIBLING_GUARD_H
#ifdef _WIN32
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <windows.h>
#include <immintrin.h>
#include <x86intrin.h>
#include "cpu_cache.h"   /* _vfft_cpuid, VFFT_CPU_HAVE_CPUID (the tree's own spelling) */

static int bench_sibling_of(int cpu)
{
    DWORD len = 0;
    char *buf;
    int sib = -1;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    buf = (char *)malloc(len);
    if (!buf || !GetLogicalProcessorInformationEx(RelationProcessorCore, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len)) { free(buf); return -1; }
    for (DWORD off = 0; off < len;)
    {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *x = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf + off);
        if (x->Relationship == RelationProcessorCore && x->Processor.GroupCount >= 1)
        {
            KAFFINITY m = x->Processor.GroupMask[0].Mask;
            if (m & ((KAFFINITY)1 << cpu))
                for (int c = 0; c < 64; c++) if ((m & ((KAFFINITY)1 << c)) && c != cpu) sib = c;
        }
        off += x->Size;
    }
    free(buf);
    return sib;
}
static int bench_has_waitpkg(void)
{   /* CPUID.(7,0):ECX[5] through the tree's own spelling (cpu_cache.h: the
     * MinGW/MSVC CPUID collision of 2026-08-31 lives in the raw names) */
    unsigned r[4] = { 0, 0, 0, 0 };
#if VFFT_CPU_HAVE_CPUID
    _vfft_cpuid(7, 0, r);
#endif
    return (r[2] >> 5) & 1u;
}
__attribute__((target("waitpkg")))
static DWORD WINAPI bench_sibling_guard(LPVOID arg)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << (int)(intptr_t)arg);
    for (;;)
        _tpause(0, __rdtsc() + 200000ull);   /* C0.2, ~35 us slices (the OS caps them); the loop is the guard */
    return 0;
}
/* hold the SMT sibling of the pinned cpu for the process's life (once); every
 * single-thread mode's pin passes through here (2026-09-22), not only the
 * K=1 / 3D one-thread protocol */
static void bench_guard_sibling(int cpu)
{
    static int done = 0;
    const char *g = getenv("VFFT_BENCH_GUARD");
    const int lifted = (g && !strcmp(g, "0"));
    const int sib = bench_sibling_of(cpu);
    if (done) return;
    done = 1;
    if (!lifted && sib >= 0 && bench_has_waitpkg() &&
        CreateThread(NULL, 0, bench_sibling_guard, (LPVOID)(intptr_t)sib, 0, NULL))
        printf("# sibling guard: cpu %d's SMT sibling cpu %d held by a TPAUSE-C0.2 thread for this process (VFFT_BENCH_GUARD=0 lifts)\n", cpu, sib);
    else
        printf("# sibling guard: cpu %d's sibling UNGUARDED (%s)\n", cpu,
               lifted ? "VFFT_BENCH_GUARD=0" : sib < 0 ? "no SMT sibling" : !bench_has_waitpkg() ? "no WAITPKG on this host" : "thread create failed");
}
#endif /* _WIN32 */
#endif /* VFFT_GAUNTLET_SIBLING_GUARD_H */
