/* sibling_guard.h -- THE SIBLING GUARDS: a waiting thread on the hyperthread sibling of each core the
 * library times or runs on, so no other thread shares that core.
 *
 * A thread on the other logical CPU of one of our cores shares the core's execution units, and our
 * thread there runs at about half speed. A threaded execute is a sequence of fork-joins over an equal,
 * fixed split, so every phase waits for that one thread: whole processes ran 2.4 times slower on an
 * i9-14900KF (docs/design/pool_sibling_guard.md). A guard holds the sibling: to the operating system the
 * CPU is busy, so the scheduler puts other threads elsewhere; to the core the waiting guard takes almost
 * nothing. It waits with the host's free timed wait -- TPAUSE into C0.2 (WAITPKG, Intel), MONITORX/MWAITX
 * (AMD, measured on Zen 4) -- or, only where the measurement settings ask for one, a PAUSE spinner
 * (~12% of the guarded thread's speed).
 *
 * ONE GUARD PER LOGICAL CPU, kept by whoever holds it:
 *   - the race scope (race_scope.h): the sibling of the racing thread's core for the scope; and once the
 *     scope uses the pool -- a race in it forks onto the pool or grows it, or a caller's own scope
 *     begins while the pool lives -- every pool guard until the scope ends (_vfft_guard_hold_take), so
 *     no guard parks between two arms and none is missing while another engine is timed. A one-thread
 *     race that does neither keeps the single guard. A scope with the guard turned off parks the pool's
 *     guards for the scope (_vfft_guard_suppress_add);
 *   - the pool (threads.h): the sibling of each worker's core and of the core it leaves to its caller,
 *     while that worker is awake -- working, or spinning for the block time (worker 0 for the caller's
 *     core). A worker that parks parks its guard within one wait (~35 us); a worker's wake wakes it.
 * No guard starts inside a virtual machine (a guest's sibling is not a core's sibling, and a waiting
 * guard still occupies a host CPU: in WSL2 the pool's guards made it up to 2x slower), where the core has
 * no sibling, where the sibling is outside the set the process may run on, on a CPU a pool worker uses,
 * or where the CPU has neither wait instruction (a PAUSE spinner the scope asks for aside). Measured on
 * Windows (i9-14900KF); on bare-metal Linux and on AMD they run unmeasured (support/README.md).
 *
 * Also here, because the scope and the pool both read it: the set this process may run on
 * (_vfs_allowed). Self-contained: threads.h includes it, and so does every bench that includes threads.h.
 */
#ifndef VFFT_SUPPORT_SIBLING_GUARD_H
#define VFFT_SUPPORT_SIBLING_GUARD_H

#include <immintrin.h>
#include <x86intrin.h>
#include "cpu_cache.h"     /* _vfft_cpuid */
#include "cpu_topology.h"  /* vfft_topo_sibling; windows.h */
#if defined(__linux__)
#  include <pthread.h>
#  include <sched.h>
#  include <time.h>
#endif

#define VFFT_GUARD_MAX_CPU 64   /* the CPUs a 64-bit mask names (processor group 0 on Windows) */

/* ── the host's wait instructions ───────────────────────────────────────── */

static inline int _vfs_has_waitpkg(void)
{   /* CPUID.(7,0):ECX[5] */
    unsigned r[4] = { 0, 0, 0, 0 };
#if VFFT_CPU_HAVE_CPUID
    _vfft_cpuid(7, 0, r);
#endif
    return (int)((r[2] >> 5) & 1u);
}
/* AMD's user-mode timed wait, CPUID.0x80000001:ECX[29] (measured present on
 * Zen 4, where WAITPKG reads 0) */
static inline int _vfs_has_monitorx(void)
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
static inline void _vfs_tpause_slice(void)
{
    _tpause(0, __rdtsc() + 200000ull);
}
/* MONITORX = 0f 01 fa (EAX = the address), MWAITX = 0f 01 fb (EAX = the
 * C-state hint, EBX = the TSC timeout, ECX bit 1 enables it): the raw
 * encodings, because the intrinsics' operand order has differed between
 * compilers and the register contract has not (it is the one Linux's own
 * delay loop uses). The monitored word is the guard's own stack slot, never
 * written, so the only wake is the timer; ~200k TSC ticks, the TPAUSE slice. */
static inline void _vfs_mwaitx_slice(volatile int *watch)
{
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xfa" :: "a"((void *)watch), "c"(0), "d"(0));
    __asm__ __volatile__(".byte 0x0f, 0x01, 0xfb" :: "a"(0u), "b"(200000u), "c"(2u));
}

/* ── a virtual machine ──────────────────────────────────────────────────── */

/* 1 = this process runs in a guest of a hypervisor. CPUID.1:ECX[31] says a hypervisor is present, and it
 * is set on a Windows host too, which runs as Hyper-V's root partition; there Hyper-V's privilege flags
 * (leaf 0x40000003, EBX bit 0, the right to create partitions) are 1, and 0 in a guest (WSL2, measured
 * 2026-10-30). So a guest is the bit with another hypervisor's vendor (leaf 0x40000000), or Hyper-V
 * without that flag. */
static inline int _vfft_guard_in_vm(void)
{
    static int vm = -1;
    if (vm < 0)
    {
        unsigned r[4] = { 0, 0, 0, 0 };
        int v = 0;
#if VFFT_CPU_HAVE_CPUID
        _vfft_cpuid(1, 0, r);
        if ((r[2] >> 31) & 1u)
        {
            v = 1;
            _vfft_cpuid(0x40000000u, 0, r);
            if (r[1] == 0x7263694Du && r[2] == 0x666F736Fu && r[3] == 0x76482074u && r[0] >= 0x40000003u)
            {   /* "Microsoft Hv" */
                _vfft_cpuid(0x40000003u, 0, r);
                v = !(r[1] & 1u);
            }
        }
#endif
        vm = v;
    }
    return vm;
}

/* ── the set this process may run on ────────────────────────────────────── */

/* Windows keeps one (the process affinity mask). Linux does not: a thread's own
 * mask is all there is, a cgroup or taskset confines through it, and the library
 * itself narrows a thread's (the scope's pin of the racing thread, a worker's pin
 * of itself). So on Linux the set is read ONCE, from the first thread that
 * reaches the library's pinning code, before anything is pinned (race_scope.h,
 * vfft.c's pool sites, the pool's guards); vfft_measure_confine replaces it. */
#if defined(__linux__)
static cpu_set_t _vfs_proc_set;
static int _vfs_proc_set_state;   /* 0 unread, 1 read, -1 unreadable */
#endif
static inline void _vfs_proc_set_read(void)
{
#if defined(__linux__)
    if (_vfs_proc_set_state == 0)
        _vfs_proc_set_state = (sched_getaffinity(0, sizeof _vfs_proc_set, &_vfs_proc_set) == 0) ? 1 : -1;
#endif
}
/* may a thread of this process run on logical CPU c */
static inline int _vfs_allowed(int c)
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

/* ── the guards ─────────────────────────────────────────────────────────── */

/* the guard of logical CPU c is _vfft_guards[c] */
typedef struct
{
    volatile int stop;          /* 1 = end */
    volatile int placed;        /* 1 = pinned to its CPU; -1 = the pin was refused and it ended */
    volatile int parked;        /* 1 = waiting on its event (the guard sets it, a waker reads it) */
    volatile int kind;          /* its wait: 1 TPAUSE, 2 MWAITX, 3 PAUSE */
    volatile int want;          /* the measurement scopes holding this CPU */
    volatile int pool;          /* 1 = the pool holds it */
    const volatile int *volatile link;   /* the pool worker's `parked` it follows: awake while that is 0 */
    int live;                   /* a thread was started and is not joined */
    int barred;                 /* 1 = a pool worker runs on this CPU: no guard goes there */
#if defined(_WIN32)
    HANDLE th, ev;              /* ev: auto-reset, a wake posted before the wait is kept */
#elif defined(__linux__)
    pthread_t th;
    pthread_mutex_t mu;
    pthread_cond_t cv;
    int sync;                   /* mu and cv initialised */
    volatile int wake;          /* a wake posted before the wait is kept */
#endif
} _vfft_guard_t;

static _vfft_guard_t _vfft_guards[VFFT_GUARD_MAX_CPU];
static volatile int _vfft_guard_hold;       /* scopes whose race forked onto the pool: every pool guard awake */
static volatile int _vfft_guard_suppress;   /* scopes measuring with the guard turned off: the pool's guards park */

/* the table's lock: starting, ending and assigning guards (the pool's lock may be held around it, never
 * the other way round); waking one takes no lock */
#if defined(_WIN32)
static CRITICAL_SECTION _vfft_guard_cs;
static volatile LONG _vfft_guard_cs_init = 0;
static inline void _vfft_guard_lock(void)
{
    if (_vfft_guard_cs_init != 2)
    {
        if (InterlockedCompareExchange(&_vfft_guard_cs_init, 1, 0) == 0)
        {
            InitializeCriticalSection(&_vfft_guard_cs);
            InterlockedExchange(&_vfft_guard_cs_init, 2);
        }
        else
            while (_vfft_guard_cs_init != 2) _mm_pause();
    }
    EnterCriticalSection(&_vfft_guard_cs);
}
static inline void _vfft_guard_unlock(void) { LeaveCriticalSection(&_vfft_guard_cs); }
/* sequentially consistent (XCHG / LOCK XADD are full fences on x86): every park/wake handshake below
 * needs the store ordered before the load that follows it */
static inline void _vfft_guard_store(volatile int *p, int v) { InterlockedExchange((volatile LONG *)p, (LONG)v); }
static inline void _vfft_guard_add(volatile int *p, int d) { InterlockedExchangeAdd((volatile LONG *)p, (LONG)d); }
#elif defined(__linux__)
static pthread_mutex_t _vfft_guard_cs = PTHREAD_MUTEX_INITIALIZER;
static inline void _vfft_guard_lock(void) { pthread_mutex_lock(&_vfft_guard_cs); }
static inline void _vfft_guard_unlock(void) { pthread_mutex_unlock(&_vfft_guard_cs); }
static inline void _vfft_guard_store(volatile int *p, int v) { __atomic_store_n(p, v, __ATOMIC_SEQ_CST); }
static inline void _vfft_guard_add(volatile int *p, int d) { (void)__atomic_add_fetch(p, d, __ATOMIC_SEQ_CST); }
#else
static inline void _vfft_guard_lock(void) {}
static inline void _vfft_guard_unlock(void) {}
static inline void _vfft_guard_store(volatile int *p, int v) { *p = v; }
static inline void _vfft_guard_add(volatile int *p, int d) { *p += d; }
#endif

/* the wait a guard uses on this host: 1 TPAUSE, 2 MWAITX; 0 = no guard (neither instruction, or a
 * virtual machine) */
static inline int _vfft_guard_kind(void)
{
    static int k = -1;
    if (k < 0)
        k = _vfft_guard_in_vm() ? 0 : _vfs_has_waitpkg() ? 1 : _vfs_has_monitorx() ? 2 : 0;
    return k;
}

/* 1 = guard g holds its CPU now: a scope wants it, or the pool does while the scopes do not park its
 * guards and either a scope's race forked onto the pool or the worker it follows is awake */
static inline int _vfft_guard_awake(const _vfft_guard_t *g)
{
    if (g->want > 0)
        return 1;
    return g->pool && _vfft_guard_suppress <= 0 && (_vfft_guard_hold > 0 || (g->link && !*g->link));
}

/* wake guard g if it is parked; called after a fenced store that may have made it awake. The handshake
 * is Dekker's, as the pool's park: the guard stores parked = 1 then re-reads what keeps it awake, the
 * waker stores that then reads parked, so one of the two always sees the other. */
static inline void _vfft_guard_poke(_vfft_guard_t *g)
{
    if (!g->parked)
        return;
#if defined(_WIN32)
    SetEvent(g->ev);
#elif defined(__linux__)
    pthread_mutex_lock(&g->mu);
    g->wake = 1;
    pthread_cond_signal(&g->cv);
    pthread_mutex_unlock(&g->mu);
#endif
}

static inline void _vfft_guard_park(_vfft_guard_t *g)
{
#if defined(_WIN32)
    _vfft_guard_store(&g->parked, 1);
    if (!g->stop && !_vfft_guard_awake(g))
        WaitForSingleObject(g->ev, INFINITE);
    _vfft_guard_store(&g->parked, 0);
#elif defined(__linux__)
    pthread_mutex_lock(&g->mu);
    _vfft_guard_store(&g->parked, 1);
    while (!g->stop && !g->wake && !_vfft_guard_awake(g))
        pthread_cond_wait(&g->cv, &g->mu);
    g->wake = 0;
    _vfft_guard_store(&g->parked, 0);
    pthread_mutex_unlock(&g->mu);
#else
    (void)g;
#endif
}

/* the guard's loop: wait while it holds its CPU, park while it does not */
static inline void _vfft_guard_body(_vfft_guard_t *g)
{
    volatile int watch = 0;
    int low = 0;   /* 1 = the thread runs at the lowest priority (a PAUSE spinner) */
    while (!g->stop)
    {
        const int k = g->kind;
#if defined(_WIN32)
        if ((k == 3) != low)
        {
            low = (k == 3);
            SetThreadPriority(GetCurrentThread(), low ? THREAD_PRIORITY_LOWEST : THREAD_PRIORITY_NORMAL);
        }
#endif
        if (!_vfft_guard_awake(g))
            _vfft_guard_park(g);
        else if (k == 1)
            _vfs_tpause_slice();
        else if (k == 2)
            _vfs_mwaitx_slice(&watch);
        else
            _mm_pause();
    }
    (void)low;
}
#if defined(_WIN32)
static inline DWORD WINAPI _vfft_guard_thread(LPVOID arg)
{
    _vfft_guard_t *g = (_vfft_guard_t *)arg;
    if (!SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << (int)(g - _vfft_guards)))
    {
        g->placed = -1;
        return 0;
    }
    g->placed = 1;
    _vfft_guard_body(g);
    return 0;
}
#elif defined(__linux__)
static inline void *_vfft_guard_thread(void *arg)
{
    _vfft_guard_t *g = (_vfft_guard_t *)arg;
    cpu_set_t s;
    CPU_ZERO(&s);
    CPU_SET((int)(g - _vfft_guards), &s);
    if (pthread_setaffinity_np(pthread_self(), sizeof s, &s) != 0)
    {
        g->placed = -1;
        return NULL;
    }
    g->placed = 1;
    _vfft_guard_body(g);
    return NULL;
}
#endif

/* start g's thread with wait `kind` (under the table's lock) */
static inline void _vfft_guard_launch(_vfft_guard_t *g, int kind)
{
    g->stop = 0;
    g->placed = 0;
    g->parked = 0;
    g->kind = kind;
#if defined(_WIN32)
    if (!g->ev)
        g->ev = CreateEvent(NULL, FALSE, FALSE, NULL);
    else
        ResetEvent(g->ev);
    g->th = g->ev ? CreateThread(NULL, 0, _vfft_guard_thread, g, 0, NULL) : NULL;
    g->live = (g->th != NULL);
#elif defined(__linux__)
    if (!g->sync)
    {
        pthread_mutex_init(&g->mu, NULL);
        pthread_cond_init(&g->cv, NULL);
        g->sync = 1;
    }
    g->wake = 0;
    g->live = (pthread_create(&g->th, NULL, _vfft_guard_thread, g) == 0);
#else
    (void)kind;
    g->live = 0;
#endif
}
/* end g's thread and join it (under the table's lock) */
static inline void _vfft_guard_end(_vfft_guard_t *g)
{
    if (!g->live)
        return;
    _vfft_guard_store(&g->stop, 1);
#if defined(_WIN32)
    SetEvent(g->ev);   /* parked, or about to park: the event keeps the wake */
    WaitForSingleObject(g->th, INFINITE);
    CloseHandle(g->th);
    g->th = NULL;
#elif defined(__linux__)
    pthread_mutex_lock(&g->mu);
    g->wake = 1;
    pthread_cond_signal(&g->cv);
    pthread_mutex_unlock(&g->mu);
    pthread_join(g->th, NULL);
#endif
    g->live = 0;
    g->placed = 0;
    g->parked = 0;
}

/* until guard g, told to hold its CPU, is up: a parked guard takes microseconds to wake, and up to
 * several milliseconds when another program's thread runs on its CPU (6.8 ms seen on a busy desktop,
 * 2026-10-30). Bounded: 10^8 TSC ticks from t0, some tens of ms; past it the caller goes on. */
static inline void _vfft_guard_wait_up(const _vfft_guard_t *g, unsigned long long t0)
{
    while (g->live && g->placed > 0 && g->parked && _vfft_guard_awake(g) && __rdtsc() - t0 < 100000000ull)
        _mm_pause();
}

/* ── for the race scope (race_scope.h) ──────────────────────────────────── */

/* the scope holds `cpu`, its racing core's sibling, with wait `kind` (3 = a PAUSE spinner); a guard the
 * pool keeps there already serves, with its own wait. Returns once a guard sits there (a thread that
 * takes longer than a second to start is left to arrive): 1 = held. */
static inline int _vfft_guard_want(int cpu, int kind)
{
    _vfft_guard_t *g;
    int fresh = 0;
    if (cpu < 0 || cpu >= VFFT_GUARD_MAX_CPU || !kind || _vfft_guard_in_vm())
        return 0;
    g = &_vfft_guards[cpu];
    _vfft_guard_lock();
    if (g->barred)
    {
        _vfft_guard_unlock();
        return 0;
    }
    _vfft_guard_add(&g->want, 1);
    if (g->live && g->placed < 0)
        _vfft_guard_end(g);   /* an earlier pin there was refused: that thread has ended */
    if (!g->live)
    {
        _vfft_guard_launch(g, kind);
        fresh = 1;
    }
    else
        _vfft_guard_poke(g);
    _vfft_guard_unlock();
    if (!fresh)
        _vfft_guard_wait_up(g, __rdtsc());   /* the pool's, parked with its worker */
    else
    {
#if defined(_WIN32)
        const ULONGLONG t0 = GetTickCount64();
        while (g->live && !g->placed && GetTickCount64() - t0 < 1000) Sleep(0);
#elif defined(__linux__)
        struct timespec a, n;
        clock_gettime(CLOCK_MONOTONIC, &a);
        n = a;
        while (g->live && !g->placed && (n.tv_sec - a.tv_sec) * 1000L + (n.tv_nsec - a.tv_nsec) / 1000000L < 1000)
        {
            sched_yield();
            clock_gettime(CLOCK_MONOTONIC, &n);
        }
#endif
    }
    if (g->live && g->placed >= 0)
        return 1;
    _vfft_guard_lock();   /* refused: it ended, or it never started */
    _vfft_guard_add(&g->want, -1);
    if (!g->pool)
        _vfft_guard_end(g);
    _vfft_guard_unlock();
    return 0;
}
/* the scope lets `cpu` go: its guard ends unless the pool or another scope holds it */
static inline void _vfft_guard_unwant(int cpu)
{
    _vfft_guard_t *g;
    if (cpu < 0 || cpu >= VFFT_GUARD_MAX_CPU)
        return;
    g = &_vfft_guards[cpu];
    _vfft_guard_lock();
    if (g->want > 0)
        _vfft_guard_add(&g->want, -1);
    if (g->want <= 0 && !g->pool)
        _vfft_guard_end(g);
    _vfft_guard_unlock();
}
/* a scope uses the pool (threads.h, _thread_pool_scope_hold): every pool guard stays awake until the
 * scope ends, so none parks between two arms; returns once they are up, so the first arm runs guarded
 * too (a race that forks after its one-thread arms finds its workers and their guards parked) */
static inline void _vfft_guard_hold_take(void)
{
    int c;
    unsigned long long t0;
    _vfft_guard_add(&_vfft_guard_hold, 1);
    for (c = 0; c < VFFT_GUARD_MAX_CPU; c++)
        if (_vfft_guards[c].pool && _vfft_guards[c].live)
            _vfft_guard_poke(&_vfft_guards[c]);
    t0 = __rdtsc();   /* one deadline for all of them */
    for (c = 0; c < VFFT_GUARD_MAX_CPU; c++)
        if (_vfft_guards[c].pool)
            _vfft_guard_wait_up(&_vfft_guards[c], t0);
}
static inline void _vfft_guard_hold_drop(void)
{
    _vfft_guard_add(&_vfft_guard_hold, -1);   /* the guards park when their workers do */
}
/* a scope measures with the guard turned off: the pool's guards park for the scope (d = +1) and follow
 * their workers again after it (d = -1) */
static inline void _vfft_guard_suppress_add(int d)
{
    int c;
    _vfft_guard_add(&_vfft_guard_suppress, d);
    if (d < 0)
        for (c = 0; c < VFFT_GUARD_MAX_CPU; c++)
            if (_vfft_guards[c].pool && _vfft_guards[c].live)
                _vfft_guard_poke(&_vfft_guards[c]);
}

/* ── for the pool (threads.h, under the pool's lock) ────────────────────── */

/* a worker now runs on `cpu`: no guard stays there */
static inline void _vfft_guard_bar(int cpu)
{
    _vfft_guard_t *g;
    if (cpu < 0 || cpu >= VFFT_GUARD_MAX_CPU)
        return;
    g = &_vfft_guards[cpu];
    _vfft_guard_lock();
    g->barred = 1;
    g->pool = 0;
    g->link = NULL;
    _vfft_guard_end(g);
    _vfft_guard_unlock();
}
/* hold the sibling of `cpu` (a worker's core, or the core the pool leaves to its caller) while the worker
 * whose `parked` flag is `link` is awake; `busy` = the CPUs the pool's threads use (bit c = logical CPU
 * c). Returns the guard's CPU, -1 = none. */
static inline int _vfft_guard_pool_add(int cpu, const volatile int *link, unsigned long long busy)
{
    const int s = vfft_topo_sibling(cpu), k = _vfft_guard_kind();
    _vfft_guard_t *g;
    if (!k || s < 0 || s >= VFFT_GUARD_MAX_CPU || ((busy >> s) & 1ull) || !_vfs_allowed(s))
        return -1;
    g = &_vfft_guards[s];
    _vfft_guard_lock();
    if (g->barred)
    {
        _vfft_guard_unlock();
        return -1;
    }
    g->link = link;
    _vfft_guard_store(&g->pool, 1);
    if (g->live && g->placed < 0)
        _vfft_guard_end(g);
    if (!g->live)
        _vfft_guard_launch(g, k);
    else
    {
        g->kind = k;   /* a scope's PAUSE spinner there takes the host's wait */
        _vfft_guard_poke(g);
    }
    _vfft_guard_unlock();
    return s;
}
/* the pool stopped its workers: its guards end, but those a scope still holds */
static inline void _vfft_guard_pool_drop(void)
{
    int c;
    _vfft_guard_lock();
    for (c = 0; c < VFFT_GUARD_MAX_CPU; c++)
    {
        _vfft_guard_t *g = &_vfft_guards[c];
        g->barred = 0;
        if (!g->pool)
            continue;
        g->pool = 0;
        g->link = NULL;
        if (g->want <= 0)
            _vfft_guard_end(g);
    }
    _vfft_guard_unlock();
}
/* the pool's guards that hold a sibling now (vfft_measure_describe) */
static inline int _vfft_guard_pool_count(void)
{
    int c, n = 0;
    for (c = 0; c < VFFT_GUARD_MAX_CPU; c++)
        if (_vfft_guards[c].pool && _vfft_guards[c].live && _vfft_guards[c].placed > 0)
            n++;
    return n;
}

#endif /* VFFT_SUPPORT_SIBLING_GUARD_H */
