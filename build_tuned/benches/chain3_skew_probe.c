/* chain3_skew_probe.c — the chain3 engine's TWO SPEEDS, isolated (2026-09-21)
 *
 * The gauntlet benched the same banked chain3 plan 1.1-1.5x slower than the
 * planner had raced it (chain3 only; flat/pair/prime agree with the race),
 * and 47 chain3 cells ran slower when timed AFTER MKL than before it.
 *
 * Pass 1 (placement): every buffer at a chosen page offset in one page-aligned
 * arena — the page offsets of mid1/mid2/zin/zout change NOTHING (six
 * placements within 1% at 1782, 1053, 1650): 4K aliasing is REFUTED.
 * Pass 2 (sleep): no sleep between rounds = every round fast (1650: 2.47 us,
 * 40 blocks back to back); a sleep of 20 or 300 ms before a round = the whole
 * round 1.66x slower, sticky for the next 10 ms of continuous execution too.
 * Pass 3 (canaries): a dependent-FP clock canary reads full clock BEFORE and
 * AFTER a slow round, and a plain AVX stream over the same bytes is unmoved:
 * not the core clock, not the memory. A busy loop pinned to CPU 2's SIBLING
 * thread slows chain3 1.47x (and the stream 1.23x); on another core, nothing.
 *
 * Pass 4: idle-cycle sampling of CPU 2 and its SMT sibling per round (the
 * syscalls themselves perturbed the mode; PROBE_NO_IDLE_SAMPLING=1 skips them),
 * an L2 pointer-chase canary (unchanged in slow rounds: the L2 serves), and
 * THE VERDICT: a thread of our own holding the sibling CPU removes the slow
 * mode entirely. PROBE_SPIN_SIBLING=1 PAUSE spin (costs the timed thread 12%),
 * =2 TPAUSE C0.1 (10%), =3 TPAUSE C0.2 (~0%, and the mode is gone: 1650 with
 * sleeps 2.58 us median vs 4.17 unguarded), =4 C0.2 during the idle phases only
 * (also works). So: after the timed thread idles, the OS parks a foreign
 * thread on the sibling and leaves it there; the bench now holds the sibling
 * with a TPAUSE-C0.2 guard for the process's life (bench_pin_one_thread) and
 * times every engine in two separated windows. Nothing here is banked.
 *
 * Run:   chain3_skew_probe.exe N R2 A B [rounds] [sleep_ms] [flat chain | -]
 *        env PROBE_SPIN_SIBLING=1|2|3|4, PROBE_NO_IDLE_SAMPLING=1
 * Build: python build.py --compile --vfft --src benches/chain3_skew_probe.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <immintrin.h>
#include "il2p.h"       /* the chain3 plan: vfft_il3p_create / execute_fwd / destroy */
#include "il_flatdit.h" /* the flat DIT: the comparison arm */

#define PIN_CPU 2

static double now_ns(void)
{
    static LARGE_INTEGER f;
    LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}

static volatile double g_sink;
static double canary_us(void)
{   /* 500k dependent fmadds = 2M cycles: pure core clock */
    double x = 1.0;
    const double a = 1.0000001, b = 1e-9;
    double t0 = now_ns();
    for (int i = 0; i < 500000; i++) x = x * a + b;
    double t = now_ns() - t0;
    g_sink = x;
    return t / 1000.0;
}

/* L2 LATENCY canary: a random cyclic pointer chase over 256 KB (L2-resident on a
 * 2 MB L2, never L1-resident): ns per hop. ~3 ns from L2, ~12+ ns from L3. If a
 * post-idle round shows L3 latency here, the L2 is not serving its capacity. */
static int *g_chase = NULL;
static double l2_chase_ns(void)
{
    const int n = 256 * 1024 / (int)sizeof(int);
    if (!g_chase)
    {   /* one random cycle through all n slots (Sattolo) */
        g_chase = (int *)malloc((size_t)n * sizeof(int));
        for (int i = 0; i < n; i++) g_chase[i] = i;
        unsigned s = 12345u;
        for (int i = n - 1; i > 0; i--) { s = s * 1664525u + 1013904223u; int j = (int)(s % (unsigned)i); int t = g_chase[i]; g_chase[i] = g_chase[j]; g_chase[j] = t; }
    }
    int k = 0; const int hops = 100000;
    double t0 = now_ns();
    for (int i = 0; i < hops; i++) k = g_chase[k];
    double t = now_ns() - t0;
    g_sink = (double)k;
    return t / hops;
}

static void stream_avx(const double *zin, double *zout, int N)
{
    const __m256d a = _mm256_set1_pd(1.0001), b = _mm256_set1_pd(0.5);
    for (int i = 0; i + 4 <= 2 * N; i += 4)
        _mm256_storeu_pd(zout + i, _mm256_fmadd_pd(_mm256_loadu_pd(zin + i), a, b));
}

/* the SMT sibling of PIN_CPU from the topology (group 0) */
static int sibling_of(int cpu)
{
    DWORD len = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, NULL, &len);
    char *buf = (char *)malloc(len);
    if (!buf || !GetLogicalProcessorInformationEx(RelationProcessorCore, (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)buf, &len)) { free(buf); return -1; }
    int sib = -1;
    for (DWORD off = 0; off < len;)
    {
        SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *e = (SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX *)(buf + off);
        if (e->Relationship == RelationProcessorCore && e->Processor.GroupCount >= 1)
        {
            KAFFINITY m = e->Processor.GroupMask[0].Mask;
            if (m & ((KAFFINITY)1 << cpu))
                for (int c = 0; c < 64; c++) if ((m & ((KAFFINITY)1 << c)) && c != cpu) sib = c;
        }
        off += e->Size;
    }
    free(buf);
    return sib;
}

/* PROBE_SPIN_SIBLING=1: a companion thread of our own on the sibling CPU, spinning
 * on PAUSE (yields the core's front end to us). Measures what reserving the
 * sibling against the OS's other threads would cost the timed thread. */
static volatile int g_spin_on = 1;
static volatile int g_guard_active = 1;   /* mode 4: the guard holds the sibling only while this is set (the idle phases) */
static int g_spin_mode = 1;   /* 1 = PAUSE loop; 2 = TPAUSE C0.1; 3 = TPAUSE C0.2; 4 = TPAUSE C0.2 during the idle phases only */
__attribute__((target("waitpkg")))
static void tpause_loop(unsigned ctrl, int idle_only)
{
    while (g_spin_on)
    {
        if (idle_only && !g_guard_active) { Sleep(1); continue; }   /* the timed window: the sibling is free */
        unsigned long long deadline = __rdtsc() + 200000ull;   /* ~35 us at the TSC rate; the OS caps it anyway */
        _tpause(ctrl, deadline);                                /* ctrl 1 = C0.1 (fast wake); 0 = C0.2 (deeper) */
    }
}
static DWORD WINAPI spin_sibling(LPVOID arg)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << (int)(intptr_t)arg);
    if (g_spin_mode == 2) { tpause_loop(1, 0); return 0; }
    if (g_spin_mode == 3) { tpause_loop(0, 0); return 0; }
    if (g_spin_mode == 4) { tpause_loop(0, 1); return 0; }
    while (g_spin_on) _mm_pause();
    return 0;
}

static ULONG64 idle_cycles(int cpu)
{
    static ULONG64 buf[64];
    ULONG len = sizeof buf;
    if (!QueryIdleProcessorCycleTimeEx(0, &len, buf)) return 0;
    return buf[cpu];
}

static double med(double *v, int n)
{
    double s[256]; if (n > 256) n = 256;
    memcpy(s, v, (size_t)n * sizeof(double));
    for (int i = 1; i < n; i++) for (int j = i; j > 0 && s[j - 1] > s[j]; j--) { double x = s[j]; s[j] = s[j - 1]; s[j - 1] = x; }
    return s[n / 2];
}
static double mn(double *v, int n) { double m = v[0]; for (int i = 1; i < n; i++) if (v[i] < m) m = v[i]; return m; }

int main(int argc, char **argv)
{
    if (argc < 5) { fprintf(stderr, "usage: %s N R2 A B [rounds] [sleep_ms] [flat chain | -]\n", argv[0]); return 2; }
    const int N = atoi(argv[1]), R2 = atoi(argv[2]), A = atoi(argv[3]), B = atoi(argv[4]);
    const int rounds = argc > 5 ? atoi(argv[5]) : 20;
    const int sleep_ms = argc > 6 ? atoi(argv[6]) : 20;
    vfft_il3p_plan_t *p = vfft_il3p_create(N, R2, A, B);
    if (!p) { fprintf(stderr, "chain %d.%d.%d does not build at N=%d\n", R2, A, B, N); return 1; }
    vfft_ilfd_plan_t *fp = NULL;
    if (argc > 7 && strcmp(argv[7], "-"))
    {
        int R[10], K = 0; char *s = strdup(argv[7]);
        for (char *t = strtok(s, "."); t && K < 10; t = strtok(NULL, ".")) R[K++] = atoi(t);
        fp = vfft_ilfd_create_chain(N, R, K);
        if (!fp) fprintf(stderr, "flat chain %s does not build at N=%d (arm skipped)\n", argv[7], N);
        free(s);
    }
    const int sib = sibling_of(PIN_CPU);
    if (getenv("PROBE_SPIN_SIBLING") && sib >= 0)
    {
        g_spin_mode = atoi(getenv("PROBE_SPIN_SIBLING"));
        CreateThread(NULL, 0, spin_sibling, (LPVOID)(intptr_t)sib, 0, NULL);
        printf("  (a %s guard thread of our own occupies CPU %d)\n", g_spin_mode == 2 ? "TPAUSE" : "PAUSE", sib);
    }
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << PIN_CPU);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    const size_t bytes = (size_t)N * 2u * sizeof(double);
    double *zin = (double *)VFFT_IL2P_ALLOC(bytes), *zout = (double *)VFFT_IL2P_ALLOC(bytes);
    for (int i = 0; i < 2 * N; i++) zin[i] = 1.0 + 1e-6 * (double)(i & 255);
    const int reps = N <= 512 ? 2000 : N <= 2048 ? 400 : 100;
    double c3[256], fl[256], st[256], ca0[256], sibbusy[256], l2[256];
    const int sample = getenv("PROBE_NO_IDLE_SAMPLING") == NULL;
    printf("N=%d chain3 %d.%d.%d%s: %d rounds x %d reps, sleep %d ms before each round, pinned CPU %d (SMT sibling = CPU %d), HIGH\n",
           N, R2, A, B, fp ? " vs flat" : "", rounds, reps, sleep_ms, PIN_CPU, sib);
    printf("  sibling busy%% = share of the chain3 window in which CPU %d accrued NO idle cycles (0 = idle sibling, 100 = someone ran there)\n", sib);
    printf("  round  canary0 us  chain3 ns  sibling busy%%  self idle%%   flat ns  stream ns\n");
    for (int r = 0; r < rounds && r < 256; r++)
    {
        g_guard_active = 1;                       /* mode 4: hold the sibling through the idle + warm-up */
        if (sleep_ms > 0) Sleep((DWORD)sleep_ms);
        ca0[r] = canary_us();
        l2[r] = l2_chase_ns();
        for (int w = 0; w < 10; w++) vfft_il3p_execute_fwd(p, zin, zout);
        g_guard_active = 0;                       /* mode 4: release it for the timed reps */
        ULONG64 i0s = (sample && sib >= 0) ? idle_cycles(sib) : 0, i0m = sample ? idle_cycles(PIN_CPU) : 0;
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) vfft_il3p_execute_fwd(p, zin, zout);
        double dt = now_ns() - t0;
        ULONG64 i1s = (sample && sib >= 0) ? idle_cycles(sib) : 0, i1m = sample ? idle_cycles(PIN_CPU) : 0;
        c3[r] = dt / reps;
        /* idle cycles run at the (unknown) core clock; ~5.5e9/s here. Report
         * the busy share as 1 - idle_cycles / (dt * 5.5e9), clamped. */
        double idle_s = (double)(i1s - i0s) / (dt * 5.5), idle_m = (double)(i1m - i0m) / (dt * 5.5);
        sibbusy[r] = 100.0 * (1.0 - (idle_s > 1.0 ? 1.0 : idle_s));
        fl[r] = st[r] = 0;
        if (fp)
        {
            for (int w = 0; w < 10; w++) vfft_ilfd_execute_fwd(fp, zin, zout);
            t0 = now_ns();
            for (int i = 0; i < reps; i++) vfft_ilfd_execute_fwd(fp, zin, zout);
            fl[r] = (now_ns() - t0) / reps;
        }
        for (int w = 0; w < 10; w++) stream_avx(zin, zout, N);
        t0 = now_ns();
        for (int i = 0; i < reps * 4; i++) stream_avx(zin, zout, N);
        st[r] = (now_ns() - t0) / (reps * 4);
        printf("  %5d  %10.0f %10.0f %13.0f %11.0f %9.0f %10.0f\n", r, ca0[r], c3[r], sibbusy[r], 100.0 * (idle_m > 1.0 ? 1.0 : idle_m), fl[r], st[r]);
    }
    const int n = rounds < 256 ? rounds : 256;
    printf("  chain3:  min %.0f  median %.0f  (median/min %.2fx)\n", mn(c3, n), med(c3, n), med(c3, n) / mn(c3, n));
    if (fp) printf("  flat:    min %.0f  median %.0f  (median/min %.2fx)\n", mn(fl, n), med(fl, n), med(fl, n) / mn(fl, n));
    printf("  stream:  min %.0f  median %.0f  (median/min %.2fx)\n", mn(st, n), med(st, n), med(st, n) / mn(st, n));
    {   /* the correlation that decides it: sibling busy share in slow rounds vs fast rounds */
        double thr = mn(c3, n) * 1.25; double bs = 0, bf = 0; int ns_ = 0, nf = 0;
        for (int r = 0; r < n; r++) { if (c3[r] > thr) { bs += sibbusy[r]; ns_++; } else { bf += sibbusy[r]; nf++; } }
        printf("  rounds slower than 1.25x the min: %d (sibling busy %.0f%% on average); fast rounds: %d (sibling busy %.0f%%)\n",
               ns_, ns_ ? bs / ns_ : 0, nf, nf ? bf / nf : 0);
    }
    VFFT_IL2P_FREE(zin); VFFT_IL2P_FREE(zout);
    vfft_il3p_destroy(p);
    if (fp) vfft_ilfd_destroy(fp);
    return 0;
}
