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
 * window runs with the sibling reserved. Hosts without WAITPKG use AMD's
 * user-mode MONITORX/MWAITX the same way (Zen 4, 2026-09-28); with neither,
 * VFFT_BENCH_GUARD=pause holds the sibling with a PAUSE spinner (a ~12% level
 * cost to both engines, ratios unaffected), else the sibling runs unguarded.
 * VFFT_BENCH_GUARD=0 lifts the guard.
 *
 * Use: bench_pin_caller(cpu) to pin the calling thread at high priority, then
 * bench_guard_sibling(cpu) once.
 *
 * LINUX (2026-09-26): the same three entry points. The pin is
 * pthread_setaffinity_np + setpriority(-10) (the priority needs CAP_SYS_NICE;
 * without it the pin still holds and the probe says so); the sibling comes from
 * /sys/devices/system/cpu/cpuN/topology/thread_siblings_list; the guard thread
 * is a pthread. VFFT_PCORE_MASK is the only way to confine the process to a
 * core set here: the Windows default mask (0x5555, the i9's eight P-cores) is
 * host-specific, so an unset mask leaves the affinity alone. */
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
#else
#ifndef _GNU_SOURCE
#define _GNU_SOURCE      /* pthread_setaffinity_np, CPU_SET */
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <pthread.h>
#include <sched.h>
#include <sys/resource.h>
#include <immintrin.h>
#include <x86intrin.h>
#include "cpu_cache.h"
#endif

#ifdef _WIN32
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
#else
static int bench_sibling_of(int cpu)
{   /* thread_siblings_list is "2,10" or "2-3": the first listed cpu != cpu */
    char path[96], line[256];
    FILE *f;
    int sib = -1;
    snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
    f = fopen(path, "r");
    if (!f) return -1;
    if (fgets(line, sizeof line, f))
    {
        char *s = line;
        while (*s && sib < 0)
        {
            char *e;
            long a = strtol(s, &e, 10), b;
            if (e == s) break;
            b = a;
            if (*e == '-') { s = e + 1; b = strtol(s, &e, 10); }
            for (long c = a; c <= b; c++) if (c != cpu) { sib = (int)c; break; }
            s = (*e == ',') ? e + 1 : e;
        }
    }
    fclose(f);
    return sib;
}
#endif
static int bench_has_waitpkg(void)
{   /* CPUID.(7,0):ECX[5] through the tree's own spelling (cpu_cache.h: the
     * MinGW/MSVC CPUID collision of 2026-08-31 lives in the raw names) */
    unsigned r[4] = { 0, 0, 0, 0 };
#if VFFT_CPU_HAVE_CPUID
    _vfft_cpuid(7, 0, r);
#endif
    return (r[2] >> 5) & 1u;
}
/* AMD's analogue, CPUID.0x80000001:ECX[29]. MONITORX/MWAITX is the user-mode
 * pair (the X suffix is exactly what makes it ring-3; Intel's MONITOR/MWAIT is
 * ring-0 and unusable here), and MWAITX takes a TSC timeout, so it holds the
 * core for a slice and wakes -- the same shape as TPAUSE, and like TPAUSE it
 * releases the core's shared resources to the SMT sibling instead of spending
 * issue slots the way PAUSE does. MEASURED present on Zen 4 (Ryzen 5 PRO
 * 8640HS, 2026-09-28) where WAITPKG reads 0. */
static int bench_has_monitorx(void)
{
    unsigned r[4] = { 0, 0, 0, 0 };
#if VFFT_CPU_HAVE_CPUID
    _vfft_cpuid(0x80000000u, 0, r);
    if (r[0] < 0x80000001u) return 0;
    _vfft_cpuid(0x80000001u, 0, r);
#endif
    return (r[2] >> 29) & 1u;
}
/* THE AMD GUARD (2026-09-28): MONITORX/MWAITX, the user-mode timed wait, used
 * exactly as the TPAUSE guard and for the same reason. Raw encodings rather
 * than the <mwaitxintrin.h> intrinsics because those take their three
 * operands in an order that has differed between compiler versions, while the
 * instruction's register contract has not: MONITORX = 0f 01 fa (EAX = the
 * address, ECX/EDX = extensions/hints), MWAITX = 0f 01 fb (EAX = the C-state
 * hint, EBX = the TSC timeout, ECX = extensions, bit 1 enabling that timeout).
 * This is the encoding and the contract Linux's own delay loop uses.
 *
 * The monitored line is this thread's own stack slot, never written, so the
 * only wake is the timer -- the loop re-arms and the sibling stays held for
 * the process's life. ~200k TSC ticks is ~40 us at this clock, matching the
 * TPAUSE guard's slice. */
static void bench_mwaitx_loop(void)
{
    volatile int watch = 0;
    for (;;)
    {
        /* MONITORX: arm on &watch */
        __asm__ __volatile__(".byte 0x0f, 0x01, 0xfa"
                             :: "a"((void *)&watch), "c"(0), "d"(0));
        /* MWAITX: C-state hint 0, timeout in EBX, ECX bit 1 = use the timer */
        __asm__ __volatile__(".byte 0x0f, 0x01, 0xfb"
                             :: "a"(0u), "b"(200000u), "c"(2u));
    }
}
/* THE PAUSE FALLBACK (2026-09-28), for hosts with SMT but neither WAITPKG nor
 * MONITORX. Without a guard those hosts run in the two-speed lottery this
 * header exists to remove: measured on the Zen 4 laptop, engine order alone
 * moved a cell by 28-60% and one 3D cell's ratio spanned 1.7-5.9x across
 * repeats.
 *
 * The trade is why this is OPT-IN, never the default: a PAUSE spinner costs
 * the timed thread ~12% where TPAUSE costs nothing measurable. That 12% is a
 * LEVEL shift, not a variance -- it applies to both engines in the same
 * process and the same window, so a RATIO is unaffected while the bimodality
 * is removed. Absolute ns from a guarded run are therefore not comparable
 * with an unguarded run's; the csv's ratio column is. VFFT_BENCH_GUARD=pause
 * turns it on, =0 lifts the guard entirely. */
#ifdef _WIN32
__attribute__((target("waitpkg")))
static DWORD WINAPI bench_sibling_guard(LPVOID arg)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << (int)(intptr_t)arg);
    for (;;)
        _tpause(0, __rdtsc() + 200000ull);   /* C0.2, ~35 us slices (the OS caps them); the loop is the guard */
    return 0;
}
static int bench_spawn_guard(int sib)
{
    return CreateThread(NULL, 0, bench_sibling_guard, (LPVOID)(intptr_t)sib, 0, NULL) != NULL;
}
static DWORD WINAPI bench_sibling_guard_mwaitx(LPVOID arg)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << (int)(intptr_t)arg);
    bench_mwaitx_loop();
    return 0;
}
static int bench_spawn_guard_mwaitx(int sib)
{
    return CreateThread(NULL, 0, bench_sibling_guard_mwaitx, (LPVOID)(intptr_t)sib, 0, NULL) != NULL;
}
static DWORD WINAPI bench_sibling_guard_pause(LPVOID arg)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << (int)(intptr_t)arg);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
    for (;;)
        _mm_pause();
    return 0;
}
static int bench_spawn_guard_pause(int sib)
{
    return CreateThread(NULL, 0, bench_sibling_guard_pause, (LPVOID)(intptr_t)sib, 0, NULL) != NULL;
}
#else
__attribute__((target("waitpkg")))
static void *bench_sibling_guard(void *arg)
{
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET((int)(intptr_t)arg, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    for (;;)
        _tpause(0, __rdtsc() + 200000ull);   /* C0.2, as on Windows */
    return NULL;
}
static int bench_spawn_guard(int sib)
{
    pthread_t t;
    if (pthread_create(&t, NULL, bench_sibling_guard, (void *)(intptr_t)sib) != 0) return 0;
    pthread_detach(t);
    return 1;
}
/* the Linux twins of the Windows MWAITX and PAUSE guards (the same loops,
 * the affinity through pthread_setaffinity_np; the spinner keeps its default
 * priority, which needs no capability) */
static void *bench_sibling_guard_mwaitx(void *arg)
{
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET((int)(intptr_t)arg, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    bench_mwaitx_loop();
    return NULL;
}
static void *bench_sibling_guard_pause(void *arg)
{
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET((int)(intptr_t)arg, &s);
    pthread_setaffinity_np(pthread_self(), sizeof s, &s);
    for (;;)
        _mm_pause();
    return NULL;
}
static int bench_spawn_thread(void *(*fn)(void *), int sib)
{
    pthread_t t;
    if (pthread_create(&t, NULL, fn, (void *)(intptr_t)sib) != 0) return 0;
    pthread_detach(t);
    return 1;
}
static int bench_spawn_guard_mwaitx(int sib) { return bench_spawn_thread(bench_sibling_guard_mwaitx, sib); }
static int bench_spawn_guard_pause(int sib) { return bench_spawn_thread(bench_sibling_guard_pause, sib); }
#endif

/* pin the calling thread to one logical cpu at the bench's HIGH priority;
 * returns 0 when the pin held */
static int bench_pin_caller(int cpu)
{
#ifdef _WIN32
    const int ok = SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu) != 0;
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    return ok ? 0 : -1;
#else
    cpu_set_t s;
    int ok;
    CPU_ZERO(&s);
    CPU_SET(cpu, &s);
    ok = pthread_setaffinity_np(pthread_self(), sizeof s, &s) == 0;
    if (setpriority(PRIO_PROCESS, 0, -10) != 0)
        printf("# pin: priority unchanged (setpriority needs CAP_SYS_NICE); the pin to cpu %d %s\n",
               cpu, ok ? "holds" : "FAILED");
    return ok ? 0 : -1;
#endif
}

/* hold the SMT sibling of the pinned cpu for the process's life (once); every
 * single-thread mode's pin passes through here (2026-09-22), not only the
 * K=1 / 3D one-thread protocol */
/* THE THREADED PROTOCOL'S CONFINEMENT (2026-09-25, from the bench): the process
 * on the 8 distinct P-cores (logical 0,2,..,14), VFFT_PCORE_MASK overriding
 * (0 = do not mask: the control for a mask-distorted threaded engine). The
 * caller then pins itself to logical 0, the core the library's pool reserves
 * for it; workers 1..7 take logical 2..14. */
static void bench_pin_pcores(void)
{
#ifdef _WIN32
    const char *e = getenv("VFFT_PCORE_MASK");
    DWORD_PTR mask = e ? (DWORD_PTR)strtoull(e, NULL, 0) : (DWORD_PTR)0x5555;
    if (mask == 0)
    {
        printf("# process affinity UNSET (VFFT_PCORE_MASK=0) -- threads float over every logical CPU incl. E-cores\n");
        return;
    }
    if (!SetProcessAffinityMask(GetCurrentProcess(), mask))
        fprintf(stderr, "pin: SetProcessAffinityMask(0x%llx) FAILED -- threads may land on E-cores\n",
                (unsigned long long)mask);
    else
        printf("# process affinity = 0x%llx (8 distinct P-cores: logical 0,2,..,14)\n",
               (unsigned long long)mask);
#else
    /* sched_setaffinity(0) sets the CALLING thread's mask; every thread created
     * after it (the library's pool, MKL/OpenMP) inherits it, which is why this
     * runs before any of them exist, as on Windows */
    const char *e = getenv("VFFT_PCORE_MASK");
    unsigned long long mask = e ? strtoull(e, NULL, 0) : 0ull;
    cpu_set_t s;
    if (!e || mask == 0)
    {
        printf("# process affinity UNSET (%s) -- threads float over every logical CPU\n",
               e ? "VFFT_PCORE_MASK=0" : "Linux: set VFFT_PCORE_MASK to confine");
        return;
    }
    CPU_ZERO(&s);
    for (int c = 0; c < 64; c++) if (mask & (1ull << c)) CPU_SET(c, &s);
    if (sched_setaffinity(0, sizeof s, &s) != 0)
        fprintf(stderr, "pin: sched_setaffinity(0x%llx) FAILED\n", mask);
    else
        printf("# process affinity = 0x%llx (VFFT_PCORE_MASK)\n", mask);
#endif
}

static void bench_guard_sibling(int cpu)
{
    static int done = 0;
    const char *g = getenv("VFFT_BENCH_GUARD");
    const int lifted = (g && !strcmp(g, "0"));
    const int want_pause = (g && !strcmp(g, "pause"));
    const int sib = bench_sibling_of(cpu);
    if (done) return;
    done = 1;
    /* Preference: the host's own free instruction first (TPAUSE on Intel,
     * MWAITX on AMD -- neither costs the timed sibling anything measurable),
     * then the PAUSE spinner only when asked for, since that one does cost
     * ~12%. VFFT_BENCH_GUARD=pause forces the spinner, =0 lifts the guard. */
    if (!lifted && want_pause && sib >= 0 && bench_spawn_guard_pause(sib))
    {
        printf("# sibling guard: cpu %d's SMT sibling cpu %d held by a PAUSE spinner "
               "(VFFT_BENCH_GUARD=pause; ~12%% level cost to BOTH engines, ratios unaffected)\n",
               cpu, sib);
        return;
    }
    if (!lifted && sib >= 0 && bench_has_waitpkg() && bench_spawn_guard(sib))
        printf("# sibling guard: cpu %d's SMT sibling cpu %d held by a TPAUSE-C0.2 thread for this process (VFFT_BENCH_GUARD=0 lifts)\n", cpu, sib);
    else if (!lifted && sib >= 0 && bench_has_monitorx() && bench_spawn_guard_mwaitx(sib))
        printf("# sibling guard: cpu %d's SMT sibling cpu %d held by a MONITORX/MWAITX thread for this process (AMD; VFFT_BENCH_GUARD=0 lifts, =pause forces the spinner)\n", cpu, sib);
    else
        printf("# sibling guard: cpu %d's sibling UNGUARDED (%s)\n", cpu,
               lifted ? "VFFT_BENCH_GUARD=0" : sib < 0 ? "no SMT sibling"
               : (!bench_has_waitpkg() && !bench_has_monitorx()) ? "neither WAITPKG nor MONITORX on this host"
               : "thread create failed");
}
#endif /* VFFT_GAUNTLET_SIBLING_GUARD_H */
