/* vtune_codelet.c — VTune microarchitecture harness for ONE in-place c2c codelet.
 *
 * Purpose: answer "do the gcc-inserted reg->reg vmovapd (FMA operand-form copies)
 * actually cost port cycles, or are they move-eliminated / off the critical path?"
 * It runs a SINGLE codelet symbol in a tight, L1-resident loop for a fixed wall-time,
 * pinned to one P-core, so the process is ~100% that codelet. Then VTune's
 * uarch-exploration attributes the top-down breakdown + port utilization to it.
 *
 * READING THE RESULT (vtune -report summary):
 *   - High Retiring + low "Core Bound / Port Utilization"  => the moves are free
 *     (move-eliminated in rename and/or off the critical path). The move count is cosmetic.
 *   - Core Bound high with Port 5 (or 0/1) saturated         => the moves cost cycles;
 *     the operand-death scheduling pass in the emitter is then worth prototyping.
 *
 * L1-resident on purpose (me=8, ios=8 => 256 doubles re/im): this ISOLATES compute +
 * port pressure. The K=256 memory-bound regime (docs/vtune-profiles/vtune_r32_codelet_k256.md)
 * MASKS port effects behind DRAM stalls, so it is the wrong lens for this specific question.
 *
 * Per-codelet build (driver supplies the symbol):
 *   gcc -O3 -g -mavx2 -mfma -march=native -DRN=32 -DFN=radix32_t1_dit_fwd_avx2 -DT1S=0 \
 *       vtune_codelet.c <codelets/inplace/avx2/r32_t1_dit_fwd.c> -lm -o vtune_codelet.exe
 *
 * Profile (run_vtune.ps1 wraps this):
 *   vtune -collect uarch-exploration -result-dir vt_r32 -- vtune_codelet.exe 8 2
 *   vtune -report summary -result-dir vt_r32
 *   vtune-gui vt_r32        # per-source-line drill-down (which vmovapd lines, if any, are hot)
 *
 * 6-arg in-place rio ABI shared by n1/t1/t1s/log3:
 *   void FN(double* re, double* im, const double* tw_re, const double* tw_im, size_t ios, size_t me)
 */
#ifndef _WIN32
#  define _POSIX_C_SOURCE 200809L
#endif
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <immintrin.h>
#ifdef _WIN32
#  include <windows.h>
#else
#  include <time.h>
#  include <pthread.h>
#  include <sched.h>
#endif

#define MPI 3.14159265358979323846
#define XSTR_(x) #x
#define XSTR(x)  XSTR_(x)

#ifndef RN
#define RN 32
#endif
#ifndef FN
#define FN radix32_t1_dit_fwd_avx2
#endif
#ifndef T1S
#define T1S 0
#endif

extern void FN(double *, double *, const double *, const double *, size_t, size_t);

static void *xaligned(size_t align, size_t sz) {
#ifdef _WIN32
    return _aligned_malloc(sz, align);
#else
    void *p = NULL;
    if (posix_memalign(&p, align, sz) != 0) return NULL;
    return p;
#endif
}
static double now_ns(void) {
#ifdef _WIN32
    LARGE_INTEGER c, f;
    QueryPerformanceCounter(&c);
    QueryPerformanceFrequency(&f);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e9 + (double)ts.tv_nsec;
#endif
}
static void pin_core(int core) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << core);
    SetThreadPriority(GetCurrentThread(), THREAD_PRIORITY_HIGHEST);
#else
    cpu_set_t set; CPU_ZERO(&set); CPU_SET(core, &set);
    pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
#endif
}

int main(int argc, char **argv) {
    double seconds = (argc > 1) ? atof(argv[1]) : 8.0;
    int    core    = (argc > 2) ? atoi(argv[2]) : 2;   /* a P-core thread on the 14900KF */
    pin_core(core);

    /* Remove the only memory-independent confound (denormal assist). */
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);

    const int    N   = RN;
    const size_t me  = 8, ios = 8;
    const size_t buf = (size_t)N * ios;

    double *re    = (double *)xaligned(64, buf * sizeof(double));
    double *im    = (double *)xaligned(64, buf * sizeof(double));
    double *tw_re = (double *)xaligned(64, ((size_t)N * me + 128) * sizeof(double));
    double *tw_im = (double *)xaligned(64, ((size_t)N * me + 128) * sizeof(double));
    if (!re || !im || !tw_re || !tw_im) { fprintf(stderr, "alloc failed\n"); return 2; }

    for (size_t i = 0; i < buf; i++) {
        re[i] = (double)((i * 2654435761u) & 0xffff) / 65535.0 - 0.5;
        im[i] = (double)((i * 40503u)      & 0xffff) / 65535.0 - 0.5;
    }
    for (int s = 0; s < N; s++) {
        double ang = -2.0 * MPI * (double)(s + 1) / (double)N;
        double c = cos(ang), sn = sin(ang);
        if (T1S) { tw_re[s] = c; tw_im[s] = sn; }
        else for (size_t b = 0; b < me; b++) {
            tw_re[(size_t)s * me + b] = c;
            tw_im[(size_t)s * me + b] = sn;
        }
    }

    /* warmup (pages in, warms caches/predictors, settles turbo) */
    for (long w = 0; w < 200000; w++) FN(re, im, tw_re, tw_im, ios, me);

    /* fixed wall-time hot loop; batch the calls so now_ns() overhead is negligible */
    const int BATCH = 20000;
    long calls = 0;
    double t0 = now_ns(), tend = t0 + seconds * 1e9;
    while (now_ns() < tend) {
        for (int b = 0; b < BATCH; b++) FN(re, im, tw_re, tw_im, ios, me);
        calls += BATCH;
    }
    double el = now_ns() - t0;

    /* keep the buffer live so the optimizer can't drop the loop (it can't anyway —
     * FN is an opaque external symbol — but make the dependency explicit). */
    volatile double sink = re[0] + im[buf - 1];
    (void)sink;

    printf("FN=%s N=%d core=%d calls=%ld wall=%.3fs ns/call=%.3f\n",
           XSTR(FN), N, core, calls, el / 1e9, el / (double)calls);
    return 0;
}
