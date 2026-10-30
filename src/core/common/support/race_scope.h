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
 *   THE SIBLING GUARDS  a waiting thread on the pinned core's hyperthread
 *       sibling for the scope, so the OS cannot park another thread there (a
 *       high-IPC kernel sharing its core runs at 60%; measured as a 1.1-1.5x
 *       two-speed lottery, 2026-09-21); and once the scope uses the pool (a
 *       race in it forks onto the pool or grows it; a caller's own scope
 *       begun while the pool lives), the siblings of every core the pool
 *       uses, to the scope's end (a contended worker ran whole processes 2.4x
 *       slower, 2026-10-30; docs/design/pool_sibling_guard.md). The guards are
 *       sibling_guard.h's, shared with the pool: they wait with the host's
 *       free instruction -- TPAUSE into C0.2 (WAITPKG, Intel), MONITORX/MWAITX
 *       (AMD, measured on Zen 4) -- which costs the timed threads nothing
 *       measurable. A PAUSE spinner costs the racing thread ~12% (a level
 *       shift on every arm, so ratios hold); it is used only when asked for,
 *       on the racing core's sibling alone. With the guard turned off, the
 *       pool's own guards park for the scope. No guard starts inside a
 *       virtual machine or on a CPU outside the set the thread was confined
 *       to, and one whose own pin is refused ends at once: floating, it
 *       would land on a core the measurement uses.
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
 * were: the pool never pins its caller, so a create leaves the thread as it
 * found it. (An EXECUTE that forks pins the calling thread to the core the pool
 * left to it, from its first fork-join to the end of the call, and puts it back:
 * threads.h, thread_pool_caller_arm.)
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
#include "common/support/sibling_guard.h" /* the guards, their waits, the set this process may run on */

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

/* this thread's scope */
static _Thread_local struct
{
    int depth;        /* begin/enter nesting */
    int auto_on;      /* entered by a create's first clock read; left by the outermost create */
    int flags;        /* VFFT_SCOPE_CONTENDED | VFFT_SCOPE_UNPINNED */
    int core;         /* the pinned logical CPU, -1 = none */
    int prio_raised;  /* 1 = the thread, 2 = the process */
    int lock_state;   /* 1 held, 0 not obtained, -1 the system has none */
    int guard_cpu;    /* the sibling its own guard holds, -1 = none */
    int guard_kind;   /* that guard's wait: 1 TPAUSE, 2 MWAITX, 3 PAUSE */
    int suppressed;   /* 1 = the guard is turned off: the pool's guards park for the scope */
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

/* ── the sibling guard (sibling_guard.h's table, shared with the pool) ──── */

/* hold the sibling of `core`, the racing thread's, for the scope; returns once
 * a guard sits there. Nothing is held without a sibling, inside a virtual
 * machine, outside the set this thread was confined to, without a wait
 * instruction, or with the guard turned off. */
static void _vfs_guard_start(int core)
{
    const int sib = vfft_topo_sibling(core);
    const int kind = (_vfs_cfg.guard == 2) ? 3 : _vfft_guard_kind();
    _vfs.guard_cpu = -1;
    _vfs.guard_kind = 0;
    if (_vfs_cfg.guard == 1 || sib < 0 || sib >= VFFT_GUARD_MAX_CPU || !kind || _vfft_guard_in_vm()) return;
    if (!_vfs_allowed(sib)) return;      /* outside the set this thread was confined to: no thread of ours goes there */
    if (_vfft_guard_want(sib, kind))
    {
        _vfs.guard_cpu = sib;
        _vfs.guard_kind = _vfft_guards[sib].kind;   /* a guard the pool keeps there serves with its own wait */
    }
}
static void _vfs_guard_stop(void)
{
    if (_vfs.guard_cpu >= 0)
        _vfft_guard_unwant(_vfs.guard_cpu);
    _vfs.guard_cpu = -1;
    _vfs.guard_kind = 0;
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
    {   /* the pool's layout: the core it left to its caller (threads.h); a
         * worker holds every other physical core, so this is the one core
         * a race can run on without sharing it with a worker */
        const int c0 = thread_pool_caller_cpu();
        if (c0 >= 0 && _vfs_is_pcore(c0) && _vfs_allowed(c0) && _vfs_pin_to(c0))
            return c0;
    }
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
    _vfs_last.guard_kind = _vfs.guard_kind;
    _vfs_last.guard_cpu = _vfs.guard_cpu;
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
    _vfs.guard_cpu = -1;
    _vfs.guard_kind = 0;
    if (_vfs.core >= 0)
        _vfs_guard_start(_vfs.core);
    if (_vfs_cfg.guard == 1)
    {   /* measured unguarded: the pool's own guards park for the scope */
        _vfft_guard_suppress_add(1);
        _vfs.suppressed = 1;
    }
    else
        _thread_pool_pin.scope = 1;   /* once it uses the pool, every pool guard holds to the scope's end (threads.h) */
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
    if (_thread_pool_pin.scope == 2)
        _vfft_guard_hold_drop();   /* the pool's guards follow their workers again */
    _thread_pool_pin.scope = 0;
    if (_vfs.suppressed)
    {
        _vfft_guard_suppress_add(-1);
        _vfs.suppressed = 0;
    }
    _vfs_priority_restore();
    _vfs_restore_affinity();
    _vfs_lock_release();
    _vfs.core = -1;
    _vfs.flags = 0;
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
