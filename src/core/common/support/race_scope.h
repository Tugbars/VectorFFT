/* race_scope.h — THE RACE SCOPE: the conditions every measurement runs under.
 *
 * A race decides which plan a cell is served from, and its winner is saved,
 * so the conditions it runs under are the library's own business, not a test
 * tool's (owner, 2026-10-04: "every feature that makes the races better
 * should also apply to planning phase of the library"). One scope surrounds
 * every race:
 *
 *   THE MEASUREMENT LOCK  one lock for the machine, so two processes -- or two
 *       threads -- never race at once (2026-10-03: a probe overlapped a T=8
 *       calibration and 15 rows were banked from contended races). The wait
 *       is bounded (60 s by default): past it the race runs anyway and the
 *       scope is CONTENDED. A system that cannot provide the lock at all
 *       races without one.
 *   THE PIN  the racing thread on the caller's own core in the library's
 *       layout: the second P-core (logical CPU 2 on a hyperthreaded part) in a
 *       process without workers; logical CPU 0 once the pool exists, the core
 *       the pool reserves for its caller (threads.h; worker 1 spins on the
 *       second P-core). Only an allowed P-core is taken; the others are walked
 *       in order. A thread that ends on no P-core is UNPINNED: an E-core runs
 *       a transform up to 3x slower and picks other winners.
 *   THE SIBLING GUARD  a thread that holds the pinned core's hyperthread
 *       sibling for the scope, so the OS cannot park another thread there (a
 *       high-IPC kernel sharing its core runs at 60%; measured as a 1.1-1.5x
 *       two-speed lottery, 2026-09-21). It waits with the host's free
 *       instruction -- TPAUSE into C0.2 (WAITPKG, Intel), MONITORX/MWAITX
 *       (AMD, measured on Zen 4) -- which costs the timed thread nothing
 *       measurable. A PAUSE spinner costs it ~12% (a level shift on every
 *       arm, so ratios hold) and is used only when asked for. No guard is
 *       started on a CPU outside the set the thread was confined to, and one
 *       whose own pin is refused ends at once: floating, it would land on a
 *       core the measurement uses.
 *   THE PRIORITY  the racing thread raised for the scope. Best effort: where
 *       it cannot be raised (Linux without CAP_SYS_NICE) the scope is still
 *       clean.
 *
 * A CONTENDED or UNPINNED scope is one the library knows measured badly: its
 * winner is served for the process and NOT saved (_vfft_scope_nosave, read by
 * the save in vfft.c).
 *
 * WHO ENTERS IT. Inside vfft_create the FIRST CLOCK READ enters the scope
 * (vfft_now_ns is redefined at the end of this file to touch it; the split
 * out-of-place planner's rdtsc reads touch it by name), and the outermost
 * create leaves it when it returns (vfft.c). So no race site can run outside
 * the scope and no exit path can leak a pin, a priority, a guard or the lock;
 * a create that hits wisdom reads no clock and enters nothing. A caller that
 * times code of its own -- the gauntlet's bench -- enters it through
 * vfft_measure_begin / vfft_measure_end (vfft.h; bodies in vfft_measure.h),
 * and a create inside that scope enters nothing more.
 *
 * WHAT IS RESTORED. The thread's affinity and priority return to what they
 * were. One exception keeps the threaded layout: a create that brings the
 * pool up pins its caller to logical CPU 0 (vfft.c, _vfft_pool_arm); the
 * scope moves its guard to that core's sibling and leaves that pin in place.
 *
 * NOT COVERED: the application's own threads on the race core, and other
 * software's load. Those are the user's to keep quiet.
 *
 * The scope's state is per thread; the settings are the process's. This file
 * is included by vfft.c only, before every header that reads the clock.
 */
#ifndef VFFT_SUPPORT_RACE_SCOPE_H
#define VFFT_SUPPORT_RACE_SCOPE_H

#include <stdio.h>
#include <immintrin.h>
#include <x86intrin.h>
#include "common/support/threads.h"      /* the pool's size; windows.h / pthread.h */
#if defined(_WIN32) && defined(_MSC_VER)
#  pragma comment(lib, "advapi32")        /* the measurement lock's security descriptor: not among the
                                           * MSVC-ABI toolchains' default libraries (ICX, clang-cl) */
#endif
#if defined(__linux__)
#  include <sched.h>
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

/* what a scope reports (vfft_diagnostics.h repeats them as VFFT_MEASURE_CONTENDED / _UNPINNED, 2026-10-07) */
#define VFFT_SCOPE_CONTENDED 1   /* the measurement lock was not obtained within the wait */
#define VFFT_SCOPE_UNPINNED  2   /* the thread is not pinned to a P-core */

/* the process's settings (vfft_measure_configure): zero = the defaults */
typedef struct
{
    int pin;          /* 0 = the library's core, 1 = no pin, 2 = pin_core */
    int pin_core;
    int guard;        /* 0 = the host's free wait instruction, 1 = none, 2 = a PAUSE spinner */
    int priority;     /* 0 = raise the measuring thread, 1 = leave, 2 = raise the whole process */
    int lock_wait_ms; /* 0 = VFFT_SCOPE_LOCK_WAIT_MS */
} _vfs_cfg_t;
static _vfs_cfg_t _vfs_cfg;
#define VFFT_SCOPE_LOCK_WAIT_MS 60000

/* how deep this thread is inside vfft_create: 0 outside, 1 in the caller's
 * create, more in a create the library makes for a child plan or a clone */
static _Thread_local int _vfft_create_depth;

typedef struct
{
    volatile int stop;
    volatile int placed;   /* 1 = the guard pinned itself; -1 = refused, it ended */
    int cpu;               /* the sibling it holds */
    int kind;              /* 1 TPAUSE, 2 MWAITX, 3 PAUSE */
    int live;
#if defined(_WIN32)
    HANDLE th;
#elif defined(__linux__)
    pthread_t th;
#endif
} _vfs_guard_t;

/* this thread's scope */
static _Thread_local struct
{
    int depth;        /* begin/enter nesting */
    int auto_on;      /* entered by a create's first clock read; left by the outermost create */
    int flags;        /* VFFT_SCOPE_CONTENDED | VFFT_SCOPE_UNPINNED */
    int core;         /* the pinned logical CPU, -1 = none */
    int pool_pinned;  /* the pool pinned the caller during the scope: that pin stays */
    int prio_raised;  /* 1 = the thread, 2 = the process */
    int lock_state;   /* 1 held, 0 not obtained, -1 the system has none */
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
} _vfs;

/* the most recent scope of this thread, as vfft_measure_describe reports it */
static _Thread_local struct
{
    int valid, core, pcore, guard_kind, guard_cpu, prio, flags, lock_state;
} _vfs_last;

long _vfft_scope_count = 0;   /* scopes entered by this process (vfft_measure_scopes) */

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

/* C0.2 for ~35 us (the OS caps the slice): busy to the scheduler, free for
 * the sibling. The loop around it is the guard. */
__attribute__((target("waitpkg")))
static void _vfs_tpause_slice(void)
{
    _tpause(0, __rdtsc() + 200000ull);
}
/* MONITORX = 0f 01 fa (EAX = the address), MWAITX = 0f 01 fb (EAX = the
 * C-state hint, EBX = the TSC timeout, ECX bit 1 enables it): the raw
 * encodings, because the intrinsics' operand order has differed between
 * compilers and the register contract has not (it is the one Linux's own
 * delay loop uses). The monitored word is the guard's own stack slot, never
 * written, so the only wake is the timer; ~200k TSC ticks, the TPAUSE slice. */
static void _vfs_mwaitx_slice(volatile int *watch)
{
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xfa" :: "a"((void *)watch), "c"(0), "d"(0));
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xfb" :: "a"(0u), "b"(200000u), "c"(2u));
}

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
    {
        g->placed = -1;
        return 0;
    }
    if (g->kind == 3)
        SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_LOWEST);
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

static int _vfs_allowed(int c);   /* below, with the pin: the set this process may run on */

/* hold the sibling of `core`; returns once the guard sits there. Nothing is
 * started without a sibling, without a wait instruction, or with the guard
 * turned off. */
static void _vfs_guard_start(int core)
{
    _vfs_guard_t *g = &_vfs.guard;
    const int sib = vfft_topo_sibling(core);
    g->live = 0;
    g->kind = 0;
    g->cpu = -1;
    if (_vfs_cfg.guard == 1 || sib < 0 || sib >= 64) return;
    if (!_vfs_allowed(sib)) return;      /* outside the set this thread was confined to: no thread of ours goes there */
    g->kind = (_vfs_cfg.guard == 2) ? 3 : _vfs_has_waitpkg() ? 1 : _vfs_has_monitorx() ? 2 : 0;
    if (!g->kind) return;
    g->stop = 0;
    g->placed = 0;
    g->cpu = sib;
    /* wait until it has pinned itself (or was refused); a thread that takes
     * longer than a second to start is left to arrive */
#if defined(_WIN32)
    g->th = CreateThread(NULL, 0, _vfs_guard_thread, g, 0, NULL);
    g->live = (g->th != NULL);
    {
        const ULONGLONG t0 = GetTickCount64();
        while (g->live && !g->placed && GetTickCount64() - t0 < 1000) Sleep(0);
    }
#elif defined(__linux__)
    g->live = (pthread_create(&g->th, NULL, _vfs_guard_thread, g) == 0);
    {
        struct timespec a, n;
        clock_gettime(CLOCK_MONOTONIC, &a);
        n = a;
        while (g->live && !g->placed && (n.tv_sec - a.tv_sec) * 1000L + (n.tv_nsec - a.tv_nsec) / 1000000L < 1000)
        {
            sched_yield();
            clock_gettime(CLOCK_MONOTONIC, &n);
        }
    }
#endif
    if (g->live && g->placed < 0)
        _vfs_guard_stop();   /* refused: it has ended */
}

/* ── the measurement lock ───────────────────────────────────────────────── */

static void _vfs_lock_wait_note(int wait_ms)
{
    static int said;
    if (said) return;
    said = 1;
    fprintf(stderr, "[measure] waiting for another VectorFFT measurement on this machine "
                    "(up to %d s)\n", wait_ms / 1000);
}

/* 1 = held; 0 = not obtained within wait_ms; -1 = this system gives no lock */
static int _vfs_lock_take(int wait_ms)
{
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
    if (!h) return -1;
    t0 = GetTickCount64();
    for (;;)
    {
        const DWORD w = WaitForSingleObject(h, 100);
        int waited;
        if (w == WAIT_OBJECT_0 || w == WAIT_ABANDONED) { _vfs.lock = h; return 1; }
        if (w != WAIT_TIMEOUT) { CloseHandle(h); return -1; }
        waited = (int)(GetTickCount64() - t0);
        if (waited >= wait_ms) break;
        if (waited >= 2000) _vfs_lock_wait_note(wait_ms);
    }
    CloseHandle(h);
    return 0;
#elif defined(__linux__)
    /* flock on one file every account can open: released when its holder
     * dies. An existing file is opened as it is (a sticky /tmp refuses
     * O_CREAT on another account's file). */
    static const char *path = "/tmp/vectorfft-measure.lock";
    struct timespec a, n;
    int fd = open(path, O_RDWR);
    if (fd < 0) fd = open(path, O_RDONLY);
    if (fd < 0)
    {
        fd = open(path, O_RDWR | O_CREAT, 0666);
        if (fd >= 0) (void)fchmod(fd, 0666);
    }
    _vfs.lock = -1;
    if (fd < 0) return -1;
    clock_gettime(CLOCK_MONOTONIC, &a);
    for (;;)
    {
        struct timespec ts = { 0, 20 * 1000000L };
        int waited;
        if (flock(fd, LOCK_EX | LOCK_NB) == 0) { _vfs.lock = fd; return 1; }
        if (errno != EWOULDBLOCK && errno != EINTR) { close(fd); return -1; }
        clock_gettime(CLOCK_MONOTONIC, &n);
        waited = (int)((n.tv_sec - a.tv_sec) * 1000L + (n.tv_nsec - a.tv_nsec) / 1000000L);
        if (waited >= wait_ms) break;
        if (waited >= 2000) _vfs_lock_wait_note(wait_ms);
        nanosleep(&ts, NULL);
    }
    close(fd);
    return 0;
#else
    (void)wait_ms;
    return -1;
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

/* the calling thread's affinity, before the scope touches it */
static void _vfs_save_affinity(void)
{
#if defined(_WIN32)
    /* there is no getter: set the process's mask, which returns the old one, and put it back */
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
/* THE SET THIS PROCESS MAY RUN ON. Windows keeps one (the process affinity
 * mask). Linux does not: a thread's own mask is all there is, a cgroup or
 * taskset confines through it, and the library itself narrows the caller's
 * (the pool pins it to logical CPU 0). So on Linux the set is read ONCE, from
 * the first thread that reaches the library's pinning code, before anything is
 * pinned (here and in vfft.c's two pool sites); vfft_measure_confine replaces
 * it. */
#if defined(__linux__)
static cpu_set_t _vfs_proc_set;
static int _vfs_proc_set_state;   /* 0 unread, 1 read, -1 unreadable */
#endif
static void _vfs_proc_set_read(void)
{
#if defined(__linux__)
    if (_vfs_proc_set_state == 0)
        _vfs_proc_set_state = (sched_getaffinity(0, sizeof _vfs_proc_set, &_vfs_proc_set) == 0) ? 1 : -1;
#endif
}
/* may a thread of this process run on logical CPU c */
static int _vfs_allowed(int c)
{
#if defined(_WIN32)
    DWORD_PTR pm = 0, sm = 0;
    if (c < 0 || c >= 64) return 0;
    if (!GetProcessAffinityMask(GetCurrentProcess(), &pm, &sm)) return 1;
    return (int)((pm >> c) & 1);
#elif defined(__linux__)
    if (c < 0 || c >= CPU_SETSIZE) return 0;
    _vfs_proc_set_read();
    return _vfs_proc_set_state == 1 ? (CPU_ISSET(c, &_vfs_proc_set) ? 1 : 0) : 1;
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
static int _vfs_is_pcore(int c)
{
    const vfft_topology_t *t = vfft_topology();
    return c >= 0 && c < t->ncpu && t->pcore[c];
}

/* pin the calling thread; returns the logical CPU, -1 = it stays where it is */
static int _vfs_pin_pick(void)
{
    int k, c, pass;
    if (_vfs_cfg.pin == 1) return -1;                        /* the pin is turned off */
    if (_vfs_cfg.pin == 2)
        return (_vfs_allowed(_vfs_cfg.pin_core) && _vfs_pin_to(_vfs_cfg.pin_core)) ? _vfs_cfg.pin_core : -1;
    if (thread_pool_size() > 1 && _vfs_is_pcore(0) && _vfs_allowed(0) && _vfs_pin_to(0))
        return 0;                                            /* the pool's layout: its caller's core */
    /* the second P-core, then the rest in order, the first one last; each
     * core's first logical CPU, then (a mask that allows only siblings) its
     * other one */
    for (pass = 0; pass < 2; pass++)
    {
        for (k = 1; (c = vfft_topo_pcore_cpu(k)) >= 0; k++)
        {
            if (pass) c = vfft_topo_sibling(c);
            if (c >= 0 && _vfs_allowed(c) && _vfs_pin_to(c)) return c;
        }
        c = vfft_topo_pcore_cpu(0);
        if (pass && c >= 0) c = vfft_topo_sibling(c);
        if (c >= 0 && _vfs_allowed(c) && _vfs_pin_to(c)) return c;
    }
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
        if (_vfs.old_class && _vfs.old_class != HIGH_PRIORITY_CLASS && _vfs.old_class != REALTIME_PRIORITY_CLASS &&
            SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS))
            _vfs.prio_raised = 2;
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

static void _vfs_note(void)
{
    _vfs_last.valid = 1;
    _vfs_last.core = _vfs.core;
    _vfs_last.pcore = _vfs_is_pcore(_vfs.core);
    _vfs_last.guard_kind = _vfs.guard.live ? _vfs.guard.kind : 0;
    _vfs_last.guard_cpu = _vfs.guard.live ? _vfs.guard.cpu : -1;
    _vfs_last.prio = _vfs.prio_raised;
    _vfs_last.flags = _vfs.flags;
    _vfs_last.lock_state = _vfs.lock_state;
}

/* enter the scope on this thread (a nested entry only counts); returns its flags */
static int _vfs_enter(void)
{
    static int said_contended, said_unpinned;
    const int wait_ms = _vfs_cfg.lock_wait_ms > 0 ? _vfs_cfg.lock_wait_ms : VFFT_SCOPE_LOCK_WAIT_MS;
    if (_vfs.depth++ > 0)
        return _vfs.flags;
    _vfs.flags = 0;
    _vfs.pool_pinned = 0;
    _vfft_scope_count++;
    _vfs_proc_set_read();
    _vfs.lock_state = _vfs_lock_take(wait_ms);
    if (_vfs.lock_state == 0)
    {
        _vfs.flags |= VFFT_SCOPE_CONTENDED;
        if (!said_contended)
        {
            said_contended = 1;
            fprintf(stderr, "[measure] another VectorFFT measurement still holds this machine after %d ms: "
                            "this race runs beside it, and its winner is kept for this process only\n", wait_ms);
        }
    }
    _vfs_save_affinity();
    _vfs.core = _vfs_pin_pick();
    if (!_vfs_is_pcore(_vfs.core))
    {
        _vfs.flags |= VFFT_SCOPE_UNPINNED;
        if (!said_unpinned)
        {
            said_unpinned = 1;
            fprintf(stderr, "[measure] this thread is not pinned to a P-core (%s): its races run there, "
                            "and their winners are kept for this process only\n",
                    _vfs_cfg.pin == 1 ? "the pin is turned off"
                    : _vfs.core >= 0 ? "the configured core is not one" : "none is allowed to this process");
        }
    }
    _vfs_priority_raise();
    if (_vfs.core >= 0)
        _vfs_guard_start(_vfs.core);
    _vfs_note();
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
    _vfs.core = -1;
    _vfs.flags = 0;
}

/* the pool pinned the caller to `core` inside a scope (vfft.c): the guard
 * moves to that core's sibling and the pin stays when the scope ends. A scope
 * that began UNPINNED stays so: its earlier races ran that way. */
static void _vfs_rehome(int core)
{
    if (_vfs.depth <= 0)
        return;
    _vfs.pool_pinned = 1;
    if (_vfs.core != core)
    {
        _vfs_guard_stop();
        _vfs.core = core;
        _vfs_guard_start(core);
    }
    _vfs_note();
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
/* 1 = the scope this thread is in measures under conditions the library
 * knows are bad: its winners are served, not saved */
static inline int _vfft_scope_nosave(void)
{
    return _vfs.depth > 0 && _vfs.flags != 0;
}

/* EVERY CLOCK READ OF THE LIBRARY PASSES THE SCOPE. A probe that watches the
 * library's clock reads defines VFFT_CLOCK_PROBE as the expression to
 * evaluate at each one, after the scope was touched. */
#ifdef VFFT_CLOCK_PROBE
#  define vfft_now_ns() (_vfft_scope_touch(), (void)(VFFT_CLOCK_PROBE), vfft_now_ns())
#else
#  define vfft_now_ns() (_vfft_scope_touch(), vfft_now_ns())
#endif

#endif /* VFFT_SUPPORT_RACE_SCOPE_H */
