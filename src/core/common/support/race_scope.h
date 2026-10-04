/* race_scope.h — THE RACE SCOPE: the conditions every measurement runs under.
 *
 * A race decides which plan a cell is served from, and its winner is saved
 * (owner, 2026-10-04), so the conditions it runs under are the library's own
 * business, not a test tool's: "every feature that makes the races better
 * should also apply to planning phase of the library". One scope surrounds
 * every race:
 *
 *   THE MEASUREMENT LOCK  one lock for the machine, so two processes -- or two
 *       threads -- never race at once (2026-10-03: a probe overlapped a T=8
 *       calibration and 15 rows were banked from contended races). The wait
 *       is bounded (60 s by default): past it the race runs anyway and the
 *       scope is CONTENDED.
 *   THE PIN  the racing thread on the caller's own core in the library's
 *       layout: the second P-core (logical CPU 2 on a hyperthreaded part) in a
 *       process without workers, logical CPU 0 once the pool exists (the pool
 *       reserves it for the caller and its first worker spins on the second
 *       P-core). Only an allowed P-core is taken; the others are walked in
 *       order. A thread that can pin nowhere is UNPINNED: an E-core runs a
 *       transform up to 3x slower and picks other winners.
 *   THE SIBLING GUARD  a thread that holds the pinned core's hyperthread
 *       sibling for the scope, so the OS cannot park another process's thread
 *       there (a high-IPC kernel sharing its core runs at 60%; measured as a
 *       1.1-1.5x two-speed lottery, 2026-09-21). It waits with the host's free
 *       instruction -- TPAUSE into C0.2 (WAITPKG, Intel), MONITORX/MWAITX
 *       (AMD, measured on Zen 4) -- which costs the timed thread nothing
 *       measurable. A PAUSE spinner costs it ~12% and is used only when asked
 *       for. A guard whose own pin is refused (its CPU is outside the process
 *       mask) ends at once.
 *   THE PRIORITY  the racing thread raised for the scope. Best effort: where
 *       it cannot be raised (Linux without CAP_SYS_NICE) the scope is still
 *       clean.
 *
 * A CONTENDED or UNPINNED scope is one the library knows measured badly: its
 * winner is served for the process and NOT saved (_vfft_scope_nosave, read by
 * the save in vfft.c).
 *
 * WHO ENTERS IT. Inside vfft_create the FIRST CLOCK READ enters the scope
 * (vfft_now_ns is redefined below to touch it; the split out-of-place
 * planner's rdtsc sites touch it by name), and the outermost create leaves it
 * when it returns (vfft.c). So no race site can run outside the scope and no
 * exit path can leak a pin, a priority or the lock; a create that hits wisdom
 * reads no clock and enters nothing. A caller that times code of its own -- the
 * gauntlet's bench -- enters it through vfft_measure_begin / vfft_measure_end
 * (vfft.h; src/core/vfft_measure.h), and a create inside that scope enters
 * nothing more.
 *
 * WHAT IS RESTORED. The thread's affinity and priority return to what they
 * were. One exception keeps today's threaded layout: a create that brings the
 * pool up pins its caller to logical CPU 0 (vfft.c, _vfft_pool_arm), and the
 * scope leaves that pin in place and moves its guard to that core's sibling.
 *
 * NOT COVERED: the application's own threads on the race core, and other
 * software's load. That is the user's to keep quiet.
 *
 * The scope's state is per thread. Depends on race_timing.h, cpu_topology.h,
 * threads.h (the pool's size) and the OS.
 */
#ifndef VFFT_SUPPORT_RACE_SCOPE_H
#define VFFT_SUPPORT_RACE_SCOPE_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <immintrin.h>
#include <x86intrin.h>
#if defined(_WIN32)
#  ifndef WIN32_LEAN_AND_MEAN
#    define WIN32_LEAN_AND_MEAN
#  endif
#  include <windows.h>
#elif defined(__linux__)
#  include <pthread.h>
#  include <sched.h>
#  include <unistd.h>
#  include <fcntl.h>
#  include <time.h>
#  include <errno.h>
#  include <sys/file.h>
#  include <sys/stat.h>
#  include <sys/resource.h>
#  include <sys/syscall.h>
#endif
#include "common/support/race_timing.h"   /* vfft_now_ns: defined before the redefinition below */
#include "common/support/cpu_cache.h"     /* _vfft_cpuid */
#include "common/support/cpu_topology.h"  /* the P-cores and the siblings */

/* what a scope's entry reports (vfft.h repeats the two public values) */
#define VFFT_SCOPE_CONTENDED 1   /* the measurement lock was not obtained within the wait */
#define VFFT_SCOPE_UNPINNED  2   /* the thread could not be pinned to an allowed P-core */

/* the process's settings (vfft_measure_configure): zero = the defaults */
typedef struct
{
    int guard;        /* 0 = the host's free wait instruction, 1 = none, 2 = a PAUSE spinner */
    int priority;     /* 0 = raise the measuring thread, 1 = leave, 2 = raise the whole process */
    int pin;          /* 0 = the library's core, 1 = no pin, 2 = pin_core */
    int pin_core;
    int lock_wait_ms; /* 0 = 60000 */
} _vfs_cfg_t;
static _vfs_cfg_t _vfs_cfg;
#define VFFT_SCOPE_LOCK_WAIT_MS 60000

/* how deep this thread is inside vfft_create: 0 outside, 1 in the caller's
 * create, more in a create the library makes for a child plan or a clone */
static _Thread_local int _vfft_create_depth;

typedef struct
{
    volatile int stop;
    int cpu;       /* the sibling to hold */
    int kind;      /* 1 TPAUSE, 2 MWAITX, 3 PAUSE */
    volatile int placed;   /* 1 = the guard pinned itself; -1 = refused, it ended */
#if defined(_WIN32)
    HANDLE th;
#elif defined(__linux__)
    pthread_t th;
#endif
    int live;
} _vfs_guard_t;

/* this thread's scope */
static _Thread_local struct
{
    int depth;        /* begin/enter nesting */
    int auto_on;      /* entered by a create's first clock read; left by the outermost create */
    int flags;        /* VFFT_SCOPE_CONTENDED | VFFT_SCOPE_UNPINNED */
    int core;         /* the pinned logical CPU, -1 = none */
    int pool_pinned;  /* the pool came up during the scope: its caller pin stays */
    int prio_raised;
    _vfs_guard_t guard;
#if defined(_WIN32)
    HANDLE lock;
    DWORD_PTR old_aff;
    int old_prio;
    DWORD old_class;
#elif defined(__linux__)
    int lock;
    cpu_set_t old_aff;
    int have_old_aff;
    int old_nice;
#endif
} _vfs = { 0 };

static long _vfs_entries;   /* scopes entered by this process (diagnostic; the scope check reads it) */

/* ── the host's wait instructions ───────────────────────────────────────── */

static int _vfs_has_waitpkg(void)
{   /* CPUID.(7,0):ECX[5] */
    unsigned r[4] = { 0, 0, 0, 0 };
#if VFFT_CPU_HAVE_CPUID
    _vfft_cpuid(7, 0, r);
#endif
    return (int)((r[2] >> 5) & 1u);
}
/* AMD's user-mode timed wait, CPUID.0x80000001:ECX[29] (measured present on
 * Zen 4, where WAITPKG reads 0) */
static int _vfs_has_monitorx(void)
{
    unsigned r[4] = { 0, 0, 0, 0 };
#if VFFT_CPU_HAVE_CPUID
    _vfft_cpuid(0x80000000u, 0, r);
    if (r[0] < 0x80000001u) return 0;
    _vfft_cpuid(0x80000001u, 0, r);
#endif
    return (int)((r[2] >> 29) & 1u);
}

#if defined(__GNUC__) || defined(__clang__)
/* C0.2 for ~35 us (the OS caps the slice): busy to the scheduler, free for
 * the sibling. The loop around it is the guard. */
__attribute__((target("waitpkg")))
static void _vfs_tpause_slice(void)
{
    _tpause(0, __rdtsc() + 200000ull);
}
/* MONITORX = 0f 01 fa (EAX = the address), MWAITX = 0f 01 fb (EBX = the TSC
 * timeout, ECX bit 1 enables it): the raw encodings, because the intrinsics'
 * operand order has differed between compilers and the register contract has
 * not. The monitored word is never written, so the only wake is the timer. */
static void _vfs_mwaitx_slice(volatile int *watch)
{
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xfa" :: "a"((void *)watch), "c"(0), "d"(0));
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xfb" :: "a"(0u), "b"(200000u), "c"(2u));
}
#else
static void _vfs_tpause_slice(void) { _mm_pause(); }
static void _vfs_mwaitx_slice(volatile int *watch) { (void)watch; _mm_pause(); }
#endif

/* ── the sibling guard ──────────────────────────────────────────────────── */

static void _vfs_guard_body(_vfs_guard_t *g)
{
    volatile int watch = 0;
    if (g->kind == 1)
        while (!g->stop) _vfs_tpause_slice();
    else if (g->kind == 2)
        while (!g->stop) _vfs_mwaitx_slice(&watch);
    else
        while (!g->stop) _mm_pause();
}
#if defined(_WIN32)
static DWORD WINAPI _vfs_guard_thread(LPVOID arg)
{
    _vfs_guard_t *g = (_vfs_guard_t *)arg;
    if (!SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << g->cpu))
    {   /* the sibling is outside the process's mask: a guard that floats would
         * land on a core the measurement uses */
        g->placed = -1;
        return 0;
    }
    SetThreadPriority(GetCurrentThread(), g->kind == 3 ? THREAD_PRIORITY_LOWEST : THREAD_PRIORITY_HIGHEST);
    g->placed = 1;
    _vfs_guard_body(g);
    return 0;
}
#elif defined(__linux__)
static void *_vfs_guard_thread(void *arg)
{
    _vfs_guard_t *g = (_vfs_guard_t *)arg;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(g->cpu, &s);
    if (pthread_setaffinity_np(pthread_self(), sizeof s, &s) != 0)
    {
        g->placed = -1;
        return NULL;
    }
    g->placed = 1;
    _vfs_guard_body(g);
    return NULL;
}
#endif

/* hold the sibling of `core` (no sibling, no wait instruction, or a guard
 * turned off: nothing is started) */
static void _vfs_guard_start(int core)
{
    _vfs_guard_t *g = &_vfs.guard;
    const char *e = getenv("VFFT_MEASURE_GUARD");
    int mode = _vfs_cfg.guard;
    const int sib = vfft_topo_sibling(core);
    if (mode == 0 && e && !strcmp(e, "0")) mode = 1;
    if (mode == 0 && e && !strcmp(e, "pause")) mode = 2;
    g->live = 0;
    if (mode == 1 || sib < 0) return;
    g->kind = (mode == 2) ? 3 : _vfs_has_waitpkg() ? 1 : _vfs_has_monitorx() ? 2 : 0;
    if (!g->kind) return;
    g->stop = 0;
    g->placed = 0;
    g->cpu = sib;
#if defined(_WIN32)
    g->th = CreateThread(NULL, 0, _vfs_guard_thread, g, 0, NULL);
    g->live = (g->th != NULL);
#elif defined(__linux__)
    g->live = (pthread_create(&g->th, NULL, _vfs_guard_thread, g) == 0);
#endif
}
static void _vfs_guard_stop(void)
{
    _vfs_guard_t *g = &_vfs.guard;
    if (!g->live) return;
    g->stop = 1;
#if defined(_WIN32)
    WaitForSingleObject(g->th, INFINITE);
    CloseHandle(g->th);
#elif defined(__linux__)
    pthread_join(g->th, NULL);
#endif
    g->live = 0;
}

/* ── the measurement lock ───────────────────────────────────────────────── */

/* 1 = held; 0 = not obtained within wait_ms (or it cannot be opened) */
static int _vfs_lock_take(int wait_ms)
{
    static int said_wait;
    int waited = 0;
#if defined(_WIN32)
    /* a named mutex in the Global namespace: one per machine, released by the
     * OS when its holder dies (the next taker sees it abandoned and owns it).
     * The NULL DACL lets every account on the machine open it. */
    SECURITY_DESCRIPTOR sd;
    SECURITY_ATTRIBUTES sa;
    HANDLE h;
    ULONGLONG t0;
    InitializeSecurityDescriptor(&sd, SECURITY_DESCRIPTOR_REVISION);
    SetSecurityDescriptorDacl(&sd, TRUE, NULL, FALSE);
    sa.nLength = sizeof sa;
    sa.lpSecurityDescriptor = &sd;
    sa.bInheritHandle = FALSE;
    h = CreateMutexA(&sa, FALSE, "Global\\VectorFFT-measure");
    if (!h) h = OpenMutexA(SYNCHRONIZE, FALSE, "Global\\VectorFFT-measure");
    _vfs.lock = NULL;
    if (!h) return 0;
    t0 = GetTickCount64();
    for (;;)
    {
        const DWORD w = WaitForSingleObject(h, 100);
        if (w == WAIT_OBJECT_0 || w == WAIT_ABANDONED) { _vfs.lock = h; return 1; }
        if (w != WAIT_TIMEOUT) break;
        waited = (int)(GetTickCount64() - t0);
        if (waited >= wait_ms) break;
        if (waited >= 2000 && !said_wait)
        {
            said_wait = 1;
            fprintf(stderr, "[measure] waiting for another VectorFFT measurement on this machine "
                            "(up to %d s)\n", wait_ms / 1000);
        }
    }
    CloseHandle(h);
    return 0;
#elif defined(__linux__)
    /* flock on one file every account can open: released when its holder dies */
    struct timespec a, n;
    int fd = open("/tmp/vectorfft-measure.lock", O_RDWR | O_CREAT, 0666);
    if (fd >= 0) (void)fchmod(fd, 0666);
    if (fd < 0) fd = open("/tmp/vectorfft-measure.lock", O_RDONLY);
    _vfs.lock = -1;
    if (fd < 0) return 0;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (;;)
    {
        struct timespec ts = { 0, 20 * 1000000L };
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) { _vfs.lock = fd; return 1; }
        clock_gettime(CLOCK_MONOTONIC, &n);
        waited = (int)((n.tv_sec - a.tv_sec) * 1000L + (n.tv_nsec - a.tv_nsec) / 1000000L);
        if (waited >= wait_ms) break;
        if (waited >= 2000 && !said_wait)
        {
            said_wait = 1;
            fprintf(stderr, "[measure] waiting for another VectorFFT measurement on this machine "
                            "(up to %d s)\n", wait_ms / 1000);
        }
        nanosleep(&ts, NULL);
    }
    close(fd);
    return 0;
#else
    (void)wait_ms; (void)waited; (void)said_wait;
    return 1;
#endif
}
static void _vfs_lock_release(void)
{
#if defined(_WIN32)
    if (_vfs.lock) { ReleaseMutex(_vfs.lock); CloseHandle(_vfs.lock); _vfs.lock = NULL; }
#elif defined(__linux__)
    if (_vfs.lock >= 0) { flock(_vfs.lock, LOCK_UN); close(_vfs.lock); _vfs.lock = -1; }
#endif
}

/* ── the pin ────────────────────────────────────────────────────────────── */

/* is logical CPU c in the set this thread may run on (read before the pin) */
static int _vfs_allowed(int c)
{
#if defined(_WIN32)
    DWORD_PTR pm = 0, sm = 0;
    if (c < 0 || c >= 64) return 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm)) return 1;
    return (pm >> c) & 1 ? 1 : 0;
#elif defined(__linux__)
    if (c < 0 || c >= CPU_SETSIZE) return 0;
    return _vfs.have_old_aff ? (CPU_ISSET(c, &_vfs.old_aff) ? 1 : 0) : 1;
#else
    (void)c;
    return 0;
#endif
}
static int _vfs_pin_to(int c)
{
#if defined(_WIN32)
    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << c) != 0;
#elif defined(__linux__)
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET(c, &s);
    return pthread_setaffinity_np(pthread_self(), sizeof s, &s) == 0;
#else
    (void)c;
    return 0;
#endif
}
/* the calling thread's affinity, before the scope touches it */
static void _vfs_save_affinity(void)
{
#if defined(_WIN32)
    /* there is no getter: set the process mask, which returns the old one, and put it back */
    DWORD_PTR pm = 0, sm = 0;
    _vfs.old_aff = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm) && pm)
    {
        _vfs.old_aff = SetThreadAffinityMask(GetCurrentThread(), pm);
        if (_vfs.old_aff) SetThreadAffinityMask(GetCurrentThread(), _vfs.old_aff);
    }
#elif defined(__linux__)
    _vfs.have_old_aff = (pthread_getaffinity_np(pthread_self(), sizeof _vfs.old_aff, &_vfs.old_aff) == 0);
#endif
}
static void _vfs_restore_affinity(void)
{
#if defined(_WIN32)
    if (_vfs.old_aff) SetThreadAffinityMask(GetCurrentThread(), _vfs.old_aff);
#elif defined(__linux__)
    if (_vfs.have_old_aff) pthread_setaffinity_np(pthread_self(), sizeof _vfs.old_aff, &_vfs.old_aff);
#endif
}

static int thread_pool_size(void);   /* threads.h */

/* pin the calling thread; returns the logical CPU, -1 = nowhere */
static int _vfs_pin_pick(void)
{
    const vfft_topology_t *t = vfft_topology();
    int k, c;
    if (_vfs_cfg.pin == 1) return -1;                        /* the pin is turned off */
    if (_vfs_cfg.pin == 2)
        return (_vfs_allowed(_vfs_cfg.pin_core) && _vfs_pin_to(_vfs_cfg.pin_core)) ? _vfs_cfg.pin_core : -1;
    if (thread_pool_size() > 1)
    {   /* the pool's layout: the caller's core is the first P-core */
        c = vfft_topo_pcore_cpu(0);
        if (c >= 0 && _vfs_allowed(c) && _vfs_pin_to(c)) return c;
    }
    /* the second P-core first, then the rest in order, the first one last (it
     * collects the OS's housekeeping) */
    for (k = 1; (c = vfft_topo_pcore_cpu(k)) >= 0; k++)
        if (_vfs_allowed(c) && _vfs_pin_to(c)) return c;
    c = vfft_topo_pcore_cpu(0);
    if (c >= 0 && _vfs_allowed(c) && _vfs_pin_to(c)) return c;
    /* the P-cores' other logical CPUs (a mask that allows only siblings) */
    for (c = 0; c < t->ncpu; c++)
        if (t->pcore[c] && _vfs_allowed(c) && _vfs_pin_to(c)) return c;
    return -1;
}

/* ── the priority ───────────────────────────────────────────────────────── */

static void _vfs_priority_raise(void)
{
    _vfs.prio_raised = 0;
    if (_vfs_cfg.priority == 1) return;
#if defined(_WIN32)
    if (_vfs_cfg.priority == 2)
    {
        _vfs.old_class = GetPriorityClass(GetCurrentProcess());
        if (_vfs.old_class && SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS)) _vfs.prio_raised = 2;
        return;
    }
    _vfs.old_prio = GetThreadPriority(GetCurrentThread());
    if (_vfs.old_prio != THREAD_PRIORITY_ERROR_RETURN && _vfs.old_prio < THREAD_PRIORITY_HIGHEST &&
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST))
        _vfs.prio_raised = 1;
#elif defined(__linux__)
    {
        const id_t who = (_vfs_cfg.priority == 2) ? 0 : (id_t)syscall(SYS_gettid);
        errno = 0;
        _vfs.old_nice = getpriority(PRIO_PROCESS, who);
        if (errno == 0 && _vfs.old_nice > -10 && setpriority(PRIO_PROCESS, who, -10) == 0)
            _vfs.prio_raised = (_vfs_cfg.priority == 2) ? 2 : 1;
    }
#endif
}
static void _vfs_priority_restore(void)
{
    if (!_vfs.prio_raised) return;
#if defined(_WIN32)
    if (_vfs.prio_raised == 2) SetPriorityClass(GetCurrentProcess(), _vfs.old_class);
    else SetThreadPriority(GetCurrentThread(), _vfs.old_prio);
#elif defined(__linux__)
    setpriority(PRIO_PROCESS, _vfs.prio_raised == 2 ? 0 : (id_t)syscall(SYS_gettid), _vfs.old_nice);
#endif
    _vfs.prio_raised = 0;
}

/* ── enter, leave ───────────────────────────────────────────────────────── */

static const char *_vfs_guard_name(void)
{
    const _vfs_guard_t *g = &_vfs.guard;
    return !g->live ? "none" : g->placed < 0 ? "refused" : g->kind == 1 ? "tpause" : g->kind == 2 ? "mwaitx" : "pause";
}

/* enter the scope on this thread (nested entries only count); returns its flags */
static int _vfs_enter(void)
{
    static int said_contended, said_unpinned;
    int wait_ms = _vfs_cfg.lock_wait_ms > 0 ? _vfs_cfg.lock_wait_ms : VFFT_SCOPE_LOCK_WAIT_MS;
    if (_vfs.depth++ > 0)
        return _vfs.flags;
    {
        const char *e = getenv("VFFT_RACE_WAIT_MS");
        if (e && e[0]) wait_ms = atoi(e);
    }
    _vfs.flags = 0;
    _vfs.pool_pinned = 0;
    _vfs_entries++;
    if (!_vfs_lock_take(wait_ms))
    {
        _vfs.flags |= VFFT_SCOPE_CONTENDED;
        if (!said_contended)
        {
            said_contended = 1;
            fprintf(stderr, "[measure] the measurement lock was not obtained within %d ms: this race runs "
                            "beside another one, and its winner is kept for this process only\n", wait_ms);
        }
    }
    _vfs_save_affinity();
    _vfs.core = _vfs_pin_pick();
    if (_vfs.core < 0)
    {
        _vfs.flags |= VFFT_SCOPE_UNPINNED;
        if (!said_unpinned)
        {
            said_unpinned = 1;
            fprintf(stderr, "[measure] this thread could not be pinned to a P-core (%s): the race runs "
                            "where the scheduler puts it, and its winner is kept for this process only\n",
                    _vfs_cfg.pin == 1 ? "the pin is turned off" : "none is allowed to this process");
        }
    }
    _vfs_priority_raise();
    if (_vfs.core >= 0)
        _vfs_guard_start(_vfs.core);
    if (getenv("VFFT_MEASURE_LOG"))
        fprintf(stderr, "[measure] enter t=%.3f ms core=%d guard=%s priority=%s lock=%s\n", vfft_now_ns() * 1e-6,
                _vfs.core, _vfs_guard_name(), _vfs.prio_raised ? "raised" : "unchanged",
                (_vfs.flags & VFFT_SCOPE_CONTENDED) ? "NOT HELD" : "held");
    return _vfs.flags;
}

static void _vfs_leave(void)
{
    if (_vfs.depth <= 0)
        return;
    if (--_vfs.depth > 0)
        return;
    _vfs_guard_stop();
    _vfs_priority_restore();
    if (!_vfs.pool_pinned)
        _vfs_restore_affinity();          /* else: the pool's caller pin stays, as without a scope */
    _vfs_lock_release();
    if (getenv("VFFT_MEASURE_LOG"))
        fprintf(stderr, "[measure] leave t=%.3f ms\n", vfft_now_ns() * 1e-6);
    _vfs.core = -1;
    _vfs.flags = 0;
}

/* the pool came up (or was resized) inside a scope and pinned the caller to
 * `core`: the guard moves to that core's sibling and the pin stays at leave */
static void _vfs_rehome(int core)
{
    if (_vfs.depth <= 0)
        return;
    _vfs.pool_pinned = 1;
    if (_vfs.core == core)
        return;
    _vfs_guard_stop();
    _vfs.core = core;
    _vfs.flags &= ~VFFT_SCOPE_UNPINNED;
    _vfs_guard_start(core);
}

/* a clock is about to be read: inside a create that has not entered the
 * scope yet, enter it (the outermost vfft_create leaves it) */
static inline void _vfft_scope_touch(void)
{
    if (_vfft_create_depth > 0 && _vfs.depth == 0)
    {
        _vfs_enter();
        _vfs.auto_on = 1;
    }
}
/* the outermost create returns */
static inline void _vfft_scope_create_done(void)
{
    if (_vfs.auto_on)
    {
        _vfs.auto_on = 0;
        _vfs_leave();
    }
}
/* 1 = the scope this thread is in measured under known-bad conditions: its
 * winners are served, not saved */
static inline int _vfft_scope_nosave(void)
{
    return _vfs.depth > 0 && _vfs.flags != 0;
}

/* EVERY CLOCK READ OF THE LIBRARY PASSES THE SCOPE. A probe that counts clock
 * reads defines VFFT_CLOCK_PROBE as the expression to evaluate at each one. */
#ifdef VFFT_CLOCK_PROBE
#  define vfft_now_ns() ((VFFT_CLOCK_PROBE), _vfft_scope_touch(), vfft_now_ns())
#else
#  define vfft_now_ns() (_vfft_scope_touch(), vfft_now_ns())
#endif

#endif /* VFFT_SUPPORT_RACE_SCOPE_H */
