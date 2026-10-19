/* caller_pin_gate.c — AN EXECUTE THAT FORKS PINS ITS CALLER TO THE POOL'S RESERVED CORE AND PUTS IT BACK;
 * ONE THAT NEVER FORKS PAYS NOTHING (src/core/common/support/threads.h, thread_pool_caller_arm: the public
 * entries arm the calling thread, the call's first fork-join pins it, the entry puts it back).
 *
 * The pool pins its workers, one per P-core, and leaves one P-core to its caller; the calling thread
 * runs a 1/T share of every phase, so a threaded execute ends at the caller's pace: on an E-core 2.2x
 * the pinned time before the door pinned the caller (caller_probe, 2026-10-19). Cells: 1D C2C 65536 at
 * T=8 and at T=1, through the public API on a cold store (the gate's thread is unpinned at the creates,
 * so the pool's reserved core is the P-core no worker holds), and a second T=8 plan of the same cell
 * (served from the first's banked wisdom) for the two-thread check.
 *   R  the reference: the T=8 plan executed with the thread pinned to the reserved core, its output
 *      kept, the median of 60 executes (the best case; nothing paced);
 *   E1 the thread pinned to an E-core ALONE: a mask without the reserved core, which the door respects:
 *      the mask after an execute equals the mask before, the output is bitwise the reference, and the
 *      median of 60 executes is over 1.3x R's (the thread stayed on the E-core);
 *   E2 the thread put on the E-core by the OS, its mask holding every E-core and the reserved core (the
 *      thread is moved onto the E-core before every execute, outside the timing): the mask after equals
 *      the mask before (the pin is put back), the output is bitwise the reference, and the median of 60
 *      executes is within 1.3x of R's: the proof the pin engaged. The control: the T=1 plan under the
 *      same placement runs well over 1.3x its reserved-core time (the E-core is the slower core; the
 *      door leaves a serial plan where it is);
 *   N  a narrower mask that excludes the reserved core (a worker's own CPU): the mask is unchanged
 *      after the execute and the output is the reference;
 *   S  the serial plan: its execute leaves the mask untouched, on the E-core and unpinned;
 *   W  two threads of the gate executing the two T=8 plans alternately, then at the same time: both
 *      outputs are the reference, both masks as they were;
 *   M  plans created with T=8 whose raced verdict is serial (N = 64, a pair; N = 1024, ZTURN-T whose
 *      threaded arm loses): no threaded pass is counted, and a call costs within 20 ns or 5% of the
 *      T=1 plan's, whichever is larger (the two plans are raced apart and may differ by a form), timed
 *      in alternating batches with the thread on the reserved core (pinning at the door cost such a
 *      call ~115 ns, 2026-10-20).
 * The pin is built for Windows: elsewhere the gate passes with a note. Usage: caller_pin_gate --wisdir <cold scratch dir> */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_diagnostics.h"

#ifndef _WIN32
int main(int argc, char **argv)
{
    (void)argc; (void)argv;
    printf("CALLER PIN GATE: the caller's pin is built for Windows only (the Linux twin is not built)\nCALLER PIN GATE ALL PASS\n");
    return 0;
}
#else
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <tlhelp32.h>

#define CELL_N 65536
#define CELL_T 8
#define NWARM 20
#define NTIME 60

static int g_checks = 0, g_fail = 0;
static const char *g_W = NULL;
#define CHECK(c, ...) do { g_checks++; if (c) printf("  ok  " __VA_ARGS__); else { g_fail++; printf("  *** FAIL: " __VA_ARGS__); } printf("\n"); } while (0)

static double now_us(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e6 * (double)c.QuadPart / (double)f.QuadPart;
}
static int cmpd(const void *a, const void *b)
{
    const double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

/* ── the machine: P-cores, E-cores, siblings (the OS's own list) ─────────────────────────────── */
static int g_ncpu, g_core[64], g_pcore[64];
static void topo(void)
{
    DWORD len = 0;
    char *buf;
    int ncores = 0, maxcls = 0, cls[64], c;
    SYSTEM_INFO si;
    GetSystemInfo(&si);
    g_ncpu = (int)si.dwNumberOfProcessors;
    if (g_ncpu > 64) g_ncpu = 64;
    for (c = 0; c < 64; c++) { g_core[c] = -1; cls[c] = 0; g_pcore[c] = 0; }
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    buf = (char *)malloc(len);
    if (buf && GetLogicalProcessorInformationEx(RelationProcessorCore, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len))
    {
        DWORD off;
        for (off = 0; off < len;)
        {
            SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *x = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf + off);
            if (x->Relationship == RelationProcessorCore && x->Processor.GroupMask[0].Group == 0)
            {
                KAFFINITY m = x->Processor.GroupMask[0].Mask;
                for (c = 0; c < g_ncpu; c++)
                    if (m & ((KAFFINITY)1 << c)) { g_core[c] = ncores; cls[c] = x->Processor.EfficiencyClass; if (cls[c] > maxcls) maxcls = cls[c]; }
                ncores++;
            }
            off += x->Size;
        }
    }
    free(buf);
    for (c = 0; c < g_ncpu; c++) g_pcore[c] = (g_core[c] >= 0 && cls[c] == maxcls);
}
static int sibling_of(int cpu)
{
    int i;
    for (i = 0; i < g_ncpu; i++)
        if (i != cpu && g_core[i] == g_core[cpu]) return i;
    return -1;
}
static const char *kind(int cpu) { return cpu < 0 ? "?" : g_pcore[cpu] ? "P" : "E"; }

/* the CPUs some other thread of this process is pinned to alone: the pool's workers, once the
 * creates are done (Toolhelp; a thread's mask is read with the Windows 7 getter) */
static DWORD_PTR worker_cpus(void)
{
    HANDLE snap = CreateToolhelp32Snapshot(TH32CS_SNAPTHREAD, 0);
    THREADENTRY32 te;
    DWORD_PTR cpus = 0;
    if (snap == INVALID_HANDLE_VALUE) return 0;
    te.dwSize = sizeof te;
    if (Thread32First(snap, &te))
    {
        do
            if (te.th32OwnerProcessID == GetCurrentProcessId() && te.th32ThreadID != GetCurrentThreadId())
            {
                HANDLE h = OpenThread(THREAD_QUERY_INFORMATION, FALSE, te.th32ThreadID);
                if (h)
                {
                    GROUP_AFFINITY ga;
                    if (GetThreadGroupAffinity(h, &ga) && ga.Mask && !(ga.Mask & (ga.Mask - 1)))
                        cpus |= (DWORD_PTR)ga.Mask;
                    CloseHandle(h);
                }
            }
        while (Thread32Next(snap, &te));
    }
    CloseHandle(snap);
    return cpus;
}
/* the first P-core (its first logical CPU) with no thread of ours pinned on either sibling: the
 * core the pool left to its caller */
static int free_pcore(DWORD_PTR w)
{
    int c;
    for (c = 0; c < g_ncpu; c++)
    {
        const int s = sibling_of(c);
        if (!g_pcore[c] || g_core[c] < 0) continue;
        if (((w >> c) & 1) || (s >= 0 && ((w >> s) & 1))) continue;
        return c;
    }
    return -1;
}
static int first_set(DWORD_PTR w)
{
    int c;
    for (c = 0; c < g_ncpu; c++) if ((w >> c) & 1) return c;
    return -1;
}
static int first_ecore(void)
{
    int c;
    for (c = 0; c < g_ncpu; c++) if (g_core[c] >= 0 && !g_pcore[c]) return c;
    return -1;
}
static DWORD_PTR my_mask(void)
{
    GROUP_AFFINITY ga;
    return GetThreadGroupAffinity(GetCurrentThread(), &ga) ? (DWORD_PTR)ga.Mask : 0;
}
static int pin_self(int cpu)
{
    return SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << cpu) != 0;
}

/* ── the cells ──────────────────────────────────────────────────────────────────────────────── */
typedef struct
{
    vfft_plan p;
    double *in, *out;
} cell_t;
static void fill(double *x, size_t n, unsigned s)
{
    size_t i;
    for (i = 0; i < n; i++)
    {
        s = s * 1664525u + 1013904223u;
        x[i] = (double)(s >> 8) / 16777216.0 - 0.5;
    }
}
static int make_cell(cell_t *c, int T, const char *label)
{
    vfft_config_t cfg;
    double t0;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.rank = 1;
    cfg.n[0] = CELL_N;
    cfg.nthreads = T;
    cfg.wisdom_dir = g_W;
    c->in = (double *)vfft_malloc(2 * CELL_N * sizeof(double));
    c->out = (double *)vfft_malloc(2 * CELL_N * sizeof(double));
    fill(c->in, 2 * CELL_N, 12345u);
    memset(c->out, 0, 2 * CELL_N * sizeof(double));
    t0 = now_us();
    c->p = vfft_create(&cfg);
    printf("  create %s: 1D C2C %d T=%d %s (%.1f ms) route=%s\n", label, CELL_N, T, c->p ? "ok" : "REFUSED",
           (now_us() - t0) / 1000.0, c->p ? vfft_plan_route(c->p) : vfft_error());
    return c->p != NULL;
}
static void exec_cell(cell_t *c) { vfft_execute(c->p, VFFT_FORWARD, c->in, c->out); }
/* the median and the minimum of NTIME back-to-back executes after NWARM, nothing paced */
static void time_cell(cell_t *c, double *med, double *mn)
{
    double t[NTIME];
    int i;
    for (i = 0; i < NWARM; i++) exec_cell(c);
    for (i = 0; i < NTIME; i++)
    {
        const double t0 = now_us();
        exec_cell(c);
        t[i] = now_us() - t0;
    }
    qsort(t, NTIME, sizeof t[0], cmpd);
    *med = t[NTIME / 2];
    *mn = t[0];
}
/* M's plans: a small cell at T threads, and the per-call time of two plans in alternating batches */
typedef struct { vfft_plan p; double *in, *out; int n; } small_t;
static int make_small(small_t *s, int n, int T)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.rank = 1;
    cfg.n[0] = n;
    cfg.nthreads = T;
    cfg.wisdom_dir = g_W;
    s->n = n;
    s->in = (double *)vfft_malloc(2 * (size_t)n * sizeof(double));
    s->out = (double *)vfft_malloc(2 * (size_t)n * sizeof(double));
    fill(s->in, 2 * (size_t)n, 777u);
    s->p = vfft_create(&cfg);
    return s->p != NULL;
}
static void free_small(small_t *s)
{
    if (s->p) vfft_destroy(s->p);
    vfft_free(s->in);
    vfft_free(s->out);
}
static double batch_ns(small_t *s, int calls)
{
    const double t0 = now_us();
    int i;
    for (i = 0; i < calls; i++) vfft_execute(s->p, VFFT_FORWARD, s->in, s->out);
    return (now_us() - t0) * 1000.0 / calls;
}
static void time_pair(small_t *a, small_t *b, double *na, double *nb)
{
    enum { NS = 21 };
    double ta[NS], tb[NS];
    const int calls = a->n <= 64 ? 200000 : 20000;
    int i;
    batch_ns(a, calls); batch_ns(b, calls);
    for (i = 0; i < NS; i++) { ta[i] = batch_ns(a, calls); tb[i] = batch_ns(b, calls); }
    qsort(ta, NS, sizeof ta[0], cmpd);
    qsort(tb, NS, sizeof tb[0], cmpd);
    *na = ta[NS / 2];
    *nb = tb[NS / 2];
}
static double max_err(const double *a, const double *b, size_t n, double *mag)
{
    size_t i;
    double e = 0.0, m = 0.0;
    for (i = 0; i < n; i++)
    {
        const double d = a[i] > b[i] ? a[i] - b[i] : b[i] - a[i];
        const double v = b[i] < 0 ? -b[i] : b[i];
        if (d > e) e = d;
        if (v > m) m = v;
    }
    *mag = m;
    return e;
}

/* ── W: the second thread ───────────────────────────────────────────────────────────────────── */
typedef struct
{
    cell_t *c;
    HANDLE go, done;        /* the alternating handoff */
    volatile LONG rounds;   /* the concurrent run: executes to do once started */
    volatile LONG start, finished;
    DWORD_PTR mask0, mask1; /* the thread's mask before and after its executes */
    HANDLE th;
} other_t;
static DWORD WINAPI other_main(LPVOID a)
{
    other_t *o = (other_t *)a;
    int i;
    o->mask0 = my_mask();
    for (i = 0; i < 20; i++)
    {   /* alternately: wait for the turn, execute, hand it back */
        WaitForSingleObject(o->go, INFINITE);
        exec_cell(o->c);
        SetEvent(o->done);
    }
    while (!o->start) Sleep(0);
    for (i = 0; i < (int)o->rounds; i++)   /* at the same time as the main thread */
        exec_cell(o->c);
    o->mask1 = my_mask();
    InterlockedExchange(&o->finished, 1);
    return 0;
}

int main(int argc, char **argv)
{
    cell_t A, B, S;
    double *ref;
    DWORD_PTR m0, m1, w, mref;
    int a, i, res, ecore, wcpu;
    double medP, minP, medE, minE, medS_P, minS_P, medS_E, minS_E, medN, minN, err, mag;
    for (a = 1; a + 1 < argc; a++)
        if (!strcmp(argv[a], "--wisdir"))
            g_W = argv[a + 1];
    if (!g_W)
    {
        printf("usage: caller_pin_gate --wisdir <cold scratch dir>\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IONBF, 0);
    topo();
    mref = my_mask();
    printf("CALLER PIN GATE: an execute that forks pins its caller to the pool's reserved core; 1D C2C %d, store %s\n", CELL_N, g_W);
    printf("  topology: %d logical CPUs; P:", g_ncpu);
    for (i = 0; i < g_ncpu; i++) if (g_pcore[i]) printf(" %d", i);
    printf("; E:");
    for (i = 0; i < g_ncpu; i++) if (g_core[i] >= 0 && !g_pcore[i]) printf(" %d", i);
    printf("; this thread's mask 0x%llx\n", (unsigned long long)mref);

    /* the creates: the threaded ones first (the pool's layout is fixed at its first growth), this
     * thread unpinned */
    if (!make_cell(&A, CELL_T, "A") || !make_cell(&B, CELL_T, "B") || !make_cell(&S, 1, "S"))
    {
        printf("*** CALLER PIN GATE FAILED: a create was refused\n");
        return 1;
    }
    w = worker_cpus();
    res = free_pcore(w);
    wcpu = first_set(w);
    ecore = first_ecore();
    printf("  pool threads %d; workers pinned on 0x%llx; the core left to the caller: CPU %d (%s); a worker's: CPU %d; "
           "first E-core: CPU %d\n", vfft_pool_threads(), (unsigned long long)w, res, kind(res), wcpu, ecore);
    CHECK(vfft_pool_threads() >= CELL_T && w != 0 && res >= 0 && wcpu >= 0,
          "the pool has %d threads, workers pinned, a reserved P-core found (CPU %d)", vfft_pool_threads(), res);
    if (res < 0 || wcpu < 0)
    {
        printf("*** CALLER PIN GATE FAILED: no pool layout to test against\n");
        return 1;
    }

    /* R: the reference, the thread pinned to the reserved core */
    CHECK(pin_self(res), "R  this thread pinned to the reserved core, CPU %d", res);
    time_cell(&A, &medP, &minP);
    ref = (double *)vfft_malloc(2 * CELL_N * sizeof(double));
    memcpy(ref, A.out, 2 * CELL_N * sizeof(double));
    time_cell(&S, &medS_P, &minS_P);
    err = max_err(S.out, ref, 2 * CELL_N, &mag);
    CHECK(err <= 1e-12 * mag, "R  T=8 on CPU %d: median %.1f us (min %.1f); T=1 there: median %.1f us (min %.1f); "
          "T=1 against T=8 max error %.1e of %.1e", res, medP, minP, medS_P, minS_P, err, mag);
    SetThreadAffinityMask(GetCurrentThread(), mref);

    /* E: the thread on an E-core. Two ways to be there: pinned to it ALONE (a mask without the
     * reserved core: the door leaves the thread where it is, E1), and put there by the OS with a
     * mask that still holds the reserved core (E2: every E-core plus the reserved core, the thread
     * moved onto the E-core before every execute, outside the timing: the door pins it). */
    if (ecore < 0)
        printf("  --  E  no E-core on this machine: the E-core checks are skipped\n");
    else
    {
        DWORD_PTR wide = (DWORD_PTR)1 << res;
        for (i = 0; i < g_ncpu; i++) if (g_core[i] >= 0 && !g_pcore[i]) wide |= (DWORD_PTR)1 << i;
        CHECK(pin_self(ecore), "E1 this thread pinned to an E-core alone, CPU %d (a mask without the reserved core)", ecore);
        m0 = my_mask();
        memset(A.out, 0, 2 * CELL_N * sizeof(double));
        time_cell(&A, &medE, &minE);
        m1 = my_mask();
        CHECK(m1 == m0 && m0 == ((DWORD_PTR)1 << ecore), "E1 the mask after the T=8 executes 0x%llx equals the mask before 0x%llx (respected)",
              (unsigned long long)m1, (unsigned long long)m0);
        CHECK(memcmp(A.out, ref, 2 * CELL_N * sizeof(double)) == 0, "E1 the output is bitwise the reference");
        CHECK(medE >= 1.3 * medP, "E1 T=8 from the E-core alone: median %.1f us (min %.1f) against %.1f pinned: %.2fx (the thread stayed there)",
              medE, minE, medP, medE / medP);
        SetThreadAffinityMask(GetCurrentThread(), mref);

        CHECK(SetThreadAffinityMask(GetCurrentThread(), wide) != 0, "E2 this thread's mask 0x%llx: every E-core and the reserved core, CPU %d",
              (unsigned long long)wide, res);
        m0 = my_mask();
        memset(A.out, 0, 2 * CELL_N * sizeof(double));
        {   /* every execute starts with the thread on the E-core: the OS's placement, imitated */
            double t[NTIME];
            int on_e = 0;
            for (i = 0; i < NWARM + NTIME; i++)
            {
                double t0;
                pin_self(ecore);                                  /* moved onto the E-core (immediate) */
                SetThreadAffinityMask(GetCurrentThread(), wide);  /* the wide mask again: the thread stays where it is */
                if (!g_pcore[GetCurrentProcessorNumber() & 63]) on_e++;
                t0 = now_us();
                exec_cell(&A);
                if (i >= NWARM) t[i - NWARM] = now_us() - t0;
            }
            qsort(t, NTIME, sizeof t[0], cmpd);
            medE = t[NTIME / 2];
            minE = t[0];
            CHECK(on_e >= (NWARM + NTIME) * 9 / 10, "E2 the thread was on an E-core before %d of %d executes", on_e, NWARM + NTIME);
        }
        m1 = my_mask();
        CHECK(m1 == m0 && m0 == wide, "E2 the mask after the T=8 executes 0x%llx equals the mask before 0x%llx (put back)",
              (unsigned long long)m1, (unsigned long long)m0);
        CHECK(memcmp(A.out, ref, 2 * CELL_N * sizeof(double)) == 0, "E2 the output is bitwise the reference");
        CHECK(medE <= 1.3 * medP, "E2 T=8 from the E-core, the reserved core in the mask: median %.1f us (min %.1f) against %.1f pinned: "
              "%.2fx (the limit 1.30x: the pin engaged)", medE, minE, medP, medE / medP);
        {   /* the control: the serial plan under the same placement is left on the E-core */
            double t[NTIME];
            for (i = 0; i < NWARM + NTIME; i++)
            {
                double t0;
                pin_self(ecore);
                SetThreadAffinityMask(GetCurrentThread(), wide);
                t0 = now_us();
                exec_cell(&S);
                if (i >= NWARM) t[i - NWARM] = now_us() - t0;
            }
            qsort(t, NTIME, sizeof t[0], cmpd);
            medS_E = t[NTIME / 2];
            minS_E = t[0];
        }
        m1 = my_mask();
        CHECK(m1 == m0, "S  the mask after the T=1 executes on the E-core 0x%llx equals the mask before 0x%llx (untouched)",
              (unsigned long long)m1, (unsigned long long)m0);
        CHECK(medS_E >= 1.3 * medS_P, "E2 the control, T=1 from the E-core, the same mask: median %.1f us (min %.1f) against %.1f on CPU %d: %.2fx "
              "(the E-core is the slower core; a serial plan is left there)", medS_E, minS_E, medS_P, res, medS_E / medS_P);
        SetThreadAffinityMask(GetCurrentThread(), mref);
    }

    /* N: a narrower mask without the reserved core, a worker's own CPU */
    CHECK(pin_self(wcpu), "N  this thread pinned to a worker's CPU %d (the reserved core excluded)", wcpu);
    m0 = my_mask();
    memset(A.out, 0, 2 * CELL_N * sizeof(double));
    {
        double t[5];
        for (i = 0; i < 5; i++)
        {
            const double t0 = now_us();
            exec_cell(&A);
            t[i] = now_us() - t0;
        }
        qsort(t, 5, sizeof t[0], cmpd);
        medN = t[2];
        minN = t[0];
    }
    m1 = my_mask();
    CHECK(m1 == m0 && m0 == ((DWORD_PTR)1 << wcpu), "N  the mask after the executes 0x%llx equals the mask before 0x%llx (respected)",
          (unsigned long long)m1, (unsigned long long)m0);
    CHECK(memcmp(A.out, ref, 2 * CELL_N * sizeof(double)) == 0, "N  the output is bitwise the reference (5 executes, median %.1f us, min %.1f)", medN, minN);
    SetThreadAffinityMask(GetCurrentThread(), mref);

    /* S: the serial plan, this thread unpinned */
    m0 = my_mask();
    for (i = 0; i < 5; i++) exec_cell(&S);
    m1 = my_mask();
    CHECK(m1 == m0 && m0 == mref, "S  the mask after the T=1 executes, unpinned, 0x%llx equals the mask before 0x%llx (untouched)",
          (unsigned long long)m1, (unsigned long long)m0);

    /* W: two threads, the two T=8 plans, alternately then at the same time */
    {
        other_t o;
        memset(&o, 0, sizeof o);
        o.c = &B;
        o.go = CreateEvent(NULL, FALSE, FALSE, NULL);
        o.done = CreateEvent(NULL, FALSE, FALSE, NULL);
        o.rounds = 20;
        o.th = CreateThread(NULL, 0, other_main, &o, 0, NULL);
        memset(A.out, 0, 2 * CELL_N * sizeof(double));
        memset(B.out, 0, 2 * CELL_N * sizeof(double));
        m0 = my_mask();
        for (i = 0; i < 20; i++)
        {
            exec_cell(&A);
            SetEvent(o.go);
            WaitForSingleObject(o.done, INFINITE);
        }
        CHECK(memcmp(A.out, ref, 2 * CELL_N * sizeof(double)) == 0, "W  alternately, 20 rounds: this thread's T=8 plan A is bitwise the reference");
        err = max_err(B.out, ref, 2 * CELL_N, &mag);
        CHECK(err <= 1e-12 * mag, "W  alternately: the other thread's T=8 plan B against the reference, max error %.1e of %.1e", err, mag);
        memset(A.out, 0, 2 * CELL_N * sizeof(double));
        memset(B.out, 0, 2 * CELL_N * sizeof(double));
        InterlockedExchange(&o.start, 1);
        for (i = 0; i < 20; i++) exec_cell(&A);
        WaitForSingleObject(o.th, INFINITE);
        CloseHandle(o.th);
        CloseHandle(o.go);
        CloseHandle(o.done);
        m1 = my_mask();
        CHECK(memcmp(A.out, ref, 2 * CELL_N * sizeof(double)) == 0, "W  at the same time, 20 each: plan A is bitwise the reference");
        err = max_err(B.out, ref, 2 * CELL_N, &mag);
        CHECK(o.finished && err <= 1e-12 * mag, "W  at the same time: plan B against the reference, max error %.1e of %.1e", err, mag);
        CHECK(m1 == m0 && o.mask1 == o.mask0, "W  both masks as they were: this thread 0x%llx -> 0x%llx, the other 0x%llx -> 0x%llx",
              (unsigned long long)m0, (unsigned long long)m1, (unsigned long long)o.mask0, (unsigned long long)o.mask1);
    }

    /* M: plans created with T=8 whose raced verdict is serial never fork, so a call costs what the T=1
     * plan's does: the entry arms the pin, and only a fork-join places it */
    {
        static const int MN[2] = { 64, 1024 };
        int k;
        pin_self(res);
        for (k = 0; k < 2; k++)
        {
            small_t s1, s8;
            double d1, d8;
            long z0;
            memset(&s1, 0, sizeof s1);
            memset(&s8, 0, sizeof s8);
            if (!make_small(&s1, MN[k], 1) || !make_small(&s8, MN[k], CELL_T))
            {
                CHECK(0, "M  N=%d: a plan was refused", MN[k]);
                free_small(&s1); free_small(&s8);
                continue;
            }
            z0 = vfft_ztt_mt_passes();
            time_pair(&s1, &s8, &d1, &d8);
            {
                const double lim = d1 * 0.05 > 20.0 ? d1 * 0.05 : 20.0;
                CHECK(vfft_ztt_mt_passes() == z0 && d8 <= d1 + lim,
                      "M  N=%d created at T=8, route %s, served serially (%ld threaded passes): %.1f ns a call against %.1f at T=1 "
                      "(limit +%.0f ns)", MN[k], vfft_plan_route(s8.p), vfft_ztt_mt_passes() - z0, d8, d1, lim);
            }
            free_small(&s1); free_small(&s8);
        }
        SetThreadAffinityMask(GetCurrentThread(), mref);
    }

    vfft_free(ref);
    vfft_destroy(A.p); vfft_free(A.in); vfft_free(A.out);
    vfft_destroy(B.p); vfft_free(B.in); vfft_free(B.out);
    vfft_destroy(S.p); vfft_free(S.in); vfft_free(S.out);
    printf("%d checks, %d failed\n%s\n", g_checks, g_fail, g_fail ? "*** CALLER PIN GATE FAILED ***" : "CALLER PIN GATE ALL PASS");
    return g_fail ? 1 : 0;
}
#endif /* _WIN32 */
