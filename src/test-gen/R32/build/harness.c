
#define _POSIX_C_SOURCE 199309L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>

#ifdef _WIN32
  #include <windows.h>
  #include <malloc.h>
#endif

/* CPUID via the compiler's preferred intrinsic header */
#if defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
  #include <intrin.h>
  static void _cpuid_count(int leaf, int subleaf, int *eax, int *ebx, int *ecx, int *edx) {
      int r[4]; __cpuidex(r, leaf, subleaf);
      *eax = r[0]; *ebx = r[1]; *ecx = r[2]; *edx = r[3];
  }
#else
  #include <cpuid.h>
  static void _cpuid_count(int leaf, int subleaf, int *eax, int *ebx, int *ecx, int *edx) {
      unsigned int a, b, c, d;
      __cpuid_count(leaf, subleaf, a, b, c, d);
      *eax = (int)a; *ebx = (int)b; *ecx = (int)c; *edx = (int)d;
  }
#endif

/* ─── timing: portable high-resolution monotonic clock in ns ─── */
#ifdef _WIN32
static double now_ns(void) {
    static LARGE_INTEGER freq;
    static int init = 0;
    if (!init) { QueryPerformanceFrequency(&freq); init = 1; }
    LARGE_INTEGER t; QueryPerformanceCounter(&t);
    return (double)t.QuadPart * 1e9 / (double)freq.QuadPart;
}
#else
static double now_ns(void) {
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e9 + ts.tv_nsec;
}
#endif

/* ─── aligned alloc / free ─── */
#ifdef _WIN32
static void *aalloc(size_t b) {
    void *p = _aligned_malloc(b, 64);
    if (p) memset(p, 0, b);
    return p;
}
static void afree(void *p) { _aligned_free(p); }
#else
static void *aalloc(size_t b) {
    void *p = NULL;
    if (posix_memalign(&p, 64, b) != 0) return NULL;
    memset(p, 0, b); return p;
}
static void afree(void *p) { free(p); }
#endif

static int _dcmp(const void *a, const void *b) {
    double x = *(const double*)a, y = *(const double*)b;
    return (x > y) - (x < y);
}

static int have_avx512(void) {
    int a, b, c, d;
    /* OSXSAVE required for AVX/AVX-512 state save */
    _cpuid_count(1, 0, &a, &b, &c, &d);
    if (!(c & (1 << 27))) return 0;
    /* XGETBV to check XCR0 bits for ZMM state saving */
    unsigned long long xcr0;
#if defined(_MSC_VER) || defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
    xcr0 = _xgetbv(0);
#else
    unsigned int lo, hi;
    __asm__ volatile ("xgetbv" : "=a"(lo), "=d"(hi) : "c"(0));
    xcr0 = ((unsigned long long)hi << 32) | lo;
#endif
    /* Bits 1(SSE) 2(AVX) 5(opmask) 6(zmm hi) 7(zmm16-31) */
    if ((xcr0 & 0xE6) != 0xE6) return 0;
    _cpuid_count(7, 0, &a, &b, &c, &d);
    return ((b & (1<<16)) != 0) && ((b & (1<<17)) != 0);
}

typedef void (*t1_fn)(double*, double*, const double*, const double*, size_t, size_t);
#define R 32
#define PI 3.14159265358979323846

static void fill_twiddles(double *Wr, double *Wi, size_t me) {
    for (size_t n = 0; n < R-1; n++) for (size_t m = 0; m < me; m++) {
        double g = -2.0 * PI * (double)(n+1) * (double)m / (double)(R*me);
        Wr[n*me+m] = cos(g); Wi[n*me+m] = sin(g);
    }
}

static double measure(t1_fn fn, size_t ios, size_t me) {
    size_t an = R * (ios > me ? ios : me);
    double *rio_re = aalloc(an*8), *rio_im = aalloc(an*8);
    double *src_re = aalloc(an*8), *src_im = aalloc(an*8);
    double *Wr = aalloc((R-1)*me*8), *Wi = aalloc((R-1)*me*8);
    unsigned s = 12345;
    for (size_t i = 0; i < an; i++) {
        s = s*1103515245u + 12345u;
        src_re[i] = ((double)(s>>16) / 32768.0) - 1.0;
        s = s*1103515245u + 12345u;
        src_im[i] = ((double)(s>>16) / 32768.0) - 1.0;
    }
    fill_twiddles(Wr, Wi, me);
    size_t work = R * me;
    int reps = (int)(1000000.0 / ((double)work + 1));
    if (reps < 20) reps = 20;
    if (reps > 10000) reps = 10000;
    for (int i = 0; i < 100; i++) {
        memcpy(rio_re, src_re, an*8); memcpy(rio_im, src_im, an*8);
        fn(rio_re, rio_im, Wr, Wi, ios, me);
    }
    double s1[21], s2[21];
    for (int i = 0; i < 21; i++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(rio_re, src_re, an*8); memcpy(rio_im, src_im, an*8);
            fn(rio_re, rio_im, Wr, Wi, ios, me);
        }
        s1[i] = (now_ns() - t0) / reps;
    }
    for (int i = 0; i < 21; i++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            memcpy(rio_re, src_re, an*8); memcpy(rio_im, src_im, an*8);
        }
        s2[i] = (now_ns() - t0) / reps;
    }
    qsort(s1, 21, sizeof(double), _dcmp);
    qsort(s2, 21, sizeof(double), _dcmp);
    double net = s1[10] - s2[10];
    if (net < 0) net = s1[10];
    afree(rio_re); afree(rio_im); afree(src_re); afree(src_im); afree(Wr); afree(Wi);
    return net;
}
extern void radix32_t1_dit_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf4r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf4r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf8r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf8r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf8r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf8r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf16r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf16r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf16r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf16r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf32r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf32r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf32r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf32r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf4r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf4r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf8r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf8r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf8r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf8r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf16r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf16r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf16r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf16r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf32r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf32r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf32r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf32r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf4r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf4r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf8r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf8r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf8r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf8r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf16r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf16r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf16r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf16r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf32r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf32r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf32r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf32r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf4r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf4r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf4r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf4r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf8r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf8r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf8r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf8r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf16r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf16r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf16r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf16r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf32r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf32r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf32r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf32r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf4r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf4r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf4r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf4r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf8r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf8r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf8r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf8r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf16r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf16r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf16r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf16r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf32r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf32r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf32r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf32r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf4r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf4r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf4r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf4r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf8r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf8r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf8r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf8r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf16r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf16r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf16r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf16r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf32r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf32r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf32r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf32r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf4r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf4r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r1_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r1_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r2_fwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r2_bwd_avx2(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf4r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf4r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf8r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf8r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf8r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf8r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf16r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf16r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf16r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf16r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf32r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf32r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf32r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_tpf32r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf4r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf4r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf8r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf8r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf8r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf8r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf16r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf16r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf16r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf16r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf32r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf32r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf32r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_dit_log3_tpf32r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_ladder_dit_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_ladder_dit_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf4r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf4r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf8r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf8r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf8r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf8r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf16r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf16r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf16r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf16r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf32r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf32r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf32r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_tpf32r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf4r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf4r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf4r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf4r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf8r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf8r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf8r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf8r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf16r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf16r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf16r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf16r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf32r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf32r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf32r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_tpf32r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf4r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf4r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf4r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf4r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf8r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf8r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf8r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf8r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf16r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf16r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf16r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf16r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf32r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf32r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf32r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_tpf32r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf4r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf4r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf4r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf4r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf8r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf8r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf8r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf8r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf16r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf16r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf16r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf16r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf32r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf32r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf32r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_tpf32r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf4r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf4r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r1_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r1_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r2_fwd_avx512(double*, double*, const double*, const double*, size_t, size_t);
extern void radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r2_bwd_avx512(double*, double*, const double*, const double*, size_t, size_t);

typedef struct { const char *id; const char *isa; t1_fn fwd, bwd; int req512; } cand_t;
static cand_t CANDS[] = {
    {"ct_t1_dit__avx2", "avx2", radix32_t1_dit_fwd_avx2, radix32_t1_dit_bwd_avx2, 0},
    {"ct_t1_dit__avx2__tpf4r1", "avx2", radix32_t1_dit_tpf4r1_fwd_avx2, radix32_t1_dit_tpf4r1_bwd_avx2, 0},
    {"ct_t1_dit__avx2__tpf8r1", "avx2", radix32_t1_dit_tpf8r1_fwd_avx2, radix32_t1_dit_tpf8r1_bwd_avx2, 0},
    {"ct_t1_dit__avx2__tpf8r2", "avx2", radix32_t1_dit_tpf8r2_fwd_avx2, radix32_t1_dit_tpf8r2_bwd_avx2, 0},
    {"ct_t1_dit__avx2__tpf16r1", "avx2", radix32_t1_dit_tpf16r1_fwd_avx2, radix32_t1_dit_tpf16r1_bwd_avx2, 0},
    {"ct_t1_dit__avx2__tpf16r2", "avx2", radix32_t1_dit_tpf16r2_fwd_avx2, radix32_t1_dit_tpf16r2_bwd_avx2, 0},
    {"ct_t1_dit__avx2__tpf32r1", "avx2", radix32_t1_dit_tpf32r1_fwd_avx2, radix32_t1_dit_tpf32r1_bwd_avx2, 0},
    {"ct_t1_dit__avx2__tpf32r2", "avx2", radix32_t1_dit_tpf32r2_fwd_avx2, radix32_t1_dit_tpf32r2_bwd_avx2, 0},
    {"ct_t1_dit_log3__avx2", "avx2", radix32_t1_dit_log3_fwd_avx2, radix32_t1_dit_log3_bwd_avx2, 0},
    {"ct_t1_dit_log3__avx2__tpf4r1", "avx2", radix32_t1_dit_log3_tpf4r1_fwd_avx2, radix32_t1_dit_log3_tpf4r1_bwd_avx2, 0},
    {"ct_t1_dit_log3__avx2__tpf8r1", "avx2", radix32_t1_dit_log3_tpf8r1_fwd_avx2, radix32_t1_dit_log3_tpf8r1_bwd_avx2, 0},
    {"ct_t1_dit_log3__avx2__tpf8r2", "avx2", radix32_t1_dit_log3_tpf8r2_fwd_avx2, radix32_t1_dit_log3_tpf8r2_bwd_avx2, 0},
    {"ct_t1_dit_log3__avx2__tpf16r1", "avx2", radix32_t1_dit_log3_tpf16r1_fwd_avx2, radix32_t1_dit_log3_tpf16r1_bwd_avx2, 0},
    {"ct_t1_dit_log3__avx2__tpf16r2", "avx2", radix32_t1_dit_log3_tpf16r2_fwd_avx2, radix32_t1_dit_log3_tpf16r2_bwd_avx2, 0},
    {"ct_t1_dit_log3__avx2__tpf32r1", "avx2", radix32_t1_dit_log3_tpf32r1_fwd_avx2, radix32_t1_dit_log3_tpf32r1_bwd_avx2, 0},
    {"ct_t1_dit_log3__avx2__tpf32r2", "avx2", radix32_t1_dit_log3_tpf32r2_fwd_avx2, radix32_t1_dit_log3_tpf32r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal", "avx2", radix32_t1_buf_dit_tile32_temporal_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__tpf4r1", "avx2", radix32_t1_buf_dit_tile32_temporal_tpf4r1_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_tpf4r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__tpf8r1", "avx2", radix32_t1_buf_dit_tile32_temporal_tpf8r1_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_tpf8r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__tpf8r2", "avx2", radix32_t1_buf_dit_tile32_temporal_tpf8r2_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_tpf8r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__tpf16r1", "avx2", radix32_t1_buf_dit_tile32_temporal_tpf16r1_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_tpf16r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__tpf16r2", "avx2", radix32_t1_buf_dit_tile32_temporal_tpf16r2_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_tpf16r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__tpf32r1", "avx2", radix32_t1_buf_dit_tile32_temporal_tpf32r1_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_tpf32r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__tpf32r2", "avx2", radix32_t1_buf_dit_tile32_temporal_tpf32r2_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_tpf32r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__prefw", "avx2", radix32_t1_buf_dit_tile32_temporal_prefw_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_prefw_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__prefw__tpf4r1", "avx2", radix32_t1_buf_dit_tile32_temporal_prefw_tpf4r1_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_prefw_tpf4r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__prefw__tpf8r1", "avx2", radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r1_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__prefw__tpf8r2", "avx2", radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r2_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__prefw__tpf16r1", "avx2", radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r1_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__prefw__tpf16r2", "avx2", radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r2_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__prefw__tpf32r1", "avx2", radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r1_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile32__draintemporal__prefw__tpf32r2", "avx2", radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r2_fwd_avx2, radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal", "avx2", radix32_t1_buf_dit_tile64_temporal_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__tpf4r1", "avx2", radix32_t1_buf_dit_tile64_temporal_tpf4r1_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_tpf4r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__tpf8r1", "avx2", radix32_t1_buf_dit_tile64_temporal_tpf8r1_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_tpf8r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__tpf8r2", "avx2", radix32_t1_buf_dit_tile64_temporal_tpf8r2_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_tpf8r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__tpf16r1", "avx2", radix32_t1_buf_dit_tile64_temporal_tpf16r1_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_tpf16r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__tpf16r2", "avx2", radix32_t1_buf_dit_tile64_temporal_tpf16r2_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_tpf16r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__tpf32r1", "avx2", radix32_t1_buf_dit_tile64_temporal_tpf32r1_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_tpf32r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__tpf32r2", "avx2", radix32_t1_buf_dit_tile64_temporal_tpf32r2_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_tpf32r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__prefw", "avx2", radix32_t1_buf_dit_tile64_temporal_prefw_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_prefw_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__prefw__tpf4r1", "avx2", radix32_t1_buf_dit_tile64_temporal_prefw_tpf4r1_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_prefw_tpf4r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__prefw__tpf8r1", "avx2", radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r1_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__prefw__tpf8r2", "avx2", radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r2_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__prefw__tpf16r1", "avx2", radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r1_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__prefw__tpf16r2", "avx2", radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r2_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__prefw__tpf32r1", "avx2", radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r1_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile64__draintemporal__prefw__tpf32r2", "avx2", radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r2_fwd_avx2, radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal", "avx2", radix32_t1_buf_dit_tile128_temporal_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__tpf4r1", "avx2", radix32_t1_buf_dit_tile128_temporal_tpf4r1_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_tpf4r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__tpf8r1", "avx2", radix32_t1_buf_dit_tile128_temporal_tpf8r1_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_tpf8r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__tpf8r2", "avx2", radix32_t1_buf_dit_tile128_temporal_tpf8r2_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_tpf8r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__tpf16r1", "avx2", radix32_t1_buf_dit_tile128_temporal_tpf16r1_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_tpf16r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__tpf16r2", "avx2", radix32_t1_buf_dit_tile128_temporal_tpf16r2_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_tpf16r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__tpf32r1", "avx2", radix32_t1_buf_dit_tile128_temporal_tpf32r1_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_tpf32r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__tpf32r2", "avx2", radix32_t1_buf_dit_tile128_temporal_tpf32r2_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_tpf32r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__prefw", "avx2", radix32_t1_buf_dit_tile128_temporal_prefw_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_prefw_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__prefw__tpf4r1", "avx2", radix32_t1_buf_dit_tile128_temporal_prefw_tpf4r1_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_prefw_tpf4r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__prefw__tpf8r1", "avx2", radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r1_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__prefw__tpf8r2", "avx2", radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r2_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__prefw__tpf16r1", "avx2", radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r1_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__prefw__tpf16r2", "avx2", radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r2_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__prefw__tpf32r1", "avx2", radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r1_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile128__draintemporal__prefw__tpf32r2", "avx2", radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r2_fwd_avx2, radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal", "avx2", radix32_t1_buf_dit_tile256_temporal_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__tpf4r1", "avx2", radix32_t1_buf_dit_tile256_temporal_tpf4r1_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_tpf4r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__tpf8r1", "avx2", radix32_t1_buf_dit_tile256_temporal_tpf8r1_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_tpf8r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__tpf8r2", "avx2", radix32_t1_buf_dit_tile256_temporal_tpf8r2_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_tpf8r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__tpf16r1", "avx2", radix32_t1_buf_dit_tile256_temporal_tpf16r1_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_tpf16r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__tpf16r2", "avx2", radix32_t1_buf_dit_tile256_temporal_tpf16r2_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_tpf16r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__tpf32r1", "avx2", radix32_t1_buf_dit_tile256_temporal_tpf32r1_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_tpf32r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__tpf32r2", "avx2", radix32_t1_buf_dit_tile256_temporal_tpf32r2_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_tpf32r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__prefw", "avx2", radix32_t1_buf_dit_tile256_temporal_prefw_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_prefw_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf4r1", "avx2", radix32_t1_buf_dit_tile256_temporal_prefw_tpf4r1_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_prefw_tpf4r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf8r1", "avx2", radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r1_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf8r2", "avx2", radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r2_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf16r1", "avx2", radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r1_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf16r2", "avx2", radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r2_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r2_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf32r1", "avx2", radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r1_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r1_bwd_avx2, 0},
    {"ct_t1_buf_dit__avx2__tile256__draintemporal__prefw__tpf32r2", "avx2", radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r2_fwd_avx2, radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r2_bwd_avx2, 0},
    {"ct_t1_dit__avx512", "avx512", radix32_t1_dit_fwd_avx512, radix32_t1_dit_bwd_avx512, 1},
    {"ct_t1_dit__avx512__tpf4r1", "avx512", radix32_t1_dit_tpf4r1_fwd_avx512, radix32_t1_dit_tpf4r1_bwd_avx512, 1},
    {"ct_t1_dit__avx512__tpf8r1", "avx512", radix32_t1_dit_tpf8r1_fwd_avx512, radix32_t1_dit_tpf8r1_bwd_avx512, 1},
    {"ct_t1_dit__avx512__tpf8r2", "avx512", radix32_t1_dit_tpf8r2_fwd_avx512, radix32_t1_dit_tpf8r2_bwd_avx512, 1},
    {"ct_t1_dit__avx512__tpf16r1", "avx512", radix32_t1_dit_tpf16r1_fwd_avx512, radix32_t1_dit_tpf16r1_bwd_avx512, 1},
    {"ct_t1_dit__avx512__tpf16r2", "avx512", radix32_t1_dit_tpf16r2_fwd_avx512, radix32_t1_dit_tpf16r2_bwd_avx512, 1},
    {"ct_t1_dit__avx512__tpf32r1", "avx512", radix32_t1_dit_tpf32r1_fwd_avx512, radix32_t1_dit_tpf32r1_bwd_avx512, 1},
    {"ct_t1_dit__avx512__tpf32r2", "avx512", radix32_t1_dit_tpf32r2_fwd_avx512, radix32_t1_dit_tpf32r2_bwd_avx512, 1},
    {"ct_t1_dit_log3__avx512", "avx512", radix32_t1_dit_log3_fwd_avx512, radix32_t1_dit_log3_bwd_avx512, 1},
    {"ct_t1_dit_log3__avx512__tpf4r1", "avx512", radix32_t1_dit_log3_tpf4r1_fwd_avx512, radix32_t1_dit_log3_tpf4r1_bwd_avx512, 1},
    {"ct_t1_dit_log3__avx512__tpf8r1", "avx512", radix32_t1_dit_log3_tpf8r1_fwd_avx512, radix32_t1_dit_log3_tpf8r1_bwd_avx512, 1},
    {"ct_t1_dit_log3__avx512__tpf8r2", "avx512", radix32_t1_dit_log3_tpf8r2_fwd_avx512, radix32_t1_dit_log3_tpf8r2_bwd_avx512, 1},
    {"ct_t1_dit_log3__avx512__tpf16r1", "avx512", radix32_t1_dit_log3_tpf16r1_fwd_avx512, radix32_t1_dit_log3_tpf16r1_bwd_avx512, 1},
    {"ct_t1_dit_log3__avx512__tpf16r2", "avx512", radix32_t1_dit_log3_tpf16r2_fwd_avx512, radix32_t1_dit_log3_tpf16r2_bwd_avx512, 1},
    {"ct_t1_dit_log3__avx512__tpf32r1", "avx512", radix32_t1_dit_log3_tpf32r1_fwd_avx512, radix32_t1_dit_log3_tpf32r1_bwd_avx512, 1},
    {"ct_t1_dit_log3__avx512__tpf32r2", "avx512", radix32_t1_dit_log3_tpf32r2_fwd_avx512, radix32_t1_dit_log3_tpf32r2_bwd_avx512, 1},
    {"ct_t1_ladder_dit__avx512", "avx512", radix32_t1_ladder_dit_fwd_avx512, radix32_t1_ladder_dit_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal", "avx512", radix32_t1_buf_dit_tile32_temporal_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__tpf4r1", "avx512", radix32_t1_buf_dit_tile32_temporal_tpf4r1_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_tpf4r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__tpf8r1", "avx512", radix32_t1_buf_dit_tile32_temporal_tpf8r1_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_tpf8r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__tpf8r2", "avx512", radix32_t1_buf_dit_tile32_temporal_tpf8r2_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_tpf8r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__tpf16r1", "avx512", radix32_t1_buf_dit_tile32_temporal_tpf16r1_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_tpf16r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__tpf16r2", "avx512", radix32_t1_buf_dit_tile32_temporal_tpf16r2_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_tpf16r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__tpf32r1", "avx512", radix32_t1_buf_dit_tile32_temporal_tpf32r1_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_tpf32r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__tpf32r2", "avx512", radix32_t1_buf_dit_tile32_temporal_tpf32r2_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_tpf32r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__prefw", "avx512", radix32_t1_buf_dit_tile32_temporal_prefw_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_prefw_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__prefw__tpf4r1", "avx512", radix32_t1_buf_dit_tile32_temporal_prefw_tpf4r1_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_prefw_tpf4r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__prefw__tpf8r1", "avx512", radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r1_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__prefw__tpf8r2", "avx512", radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r2_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_prefw_tpf8r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__prefw__tpf16r1", "avx512", radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r1_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__prefw__tpf16r2", "avx512", radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r2_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_prefw_tpf16r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__prefw__tpf32r1", "avx512", radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r1_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile32__draintemporal__prefw__tpf32r2", "avx512", radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r2_fwd_avx512, radix32_t1_buf_dit_tile32_temporal_prefw_tpf32r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal", "avx512", radix32_t1_buf_dit_tile64_temporal_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__tpf4r1", "avx512", radix32_t1_buf_dit_tile64_temporal_tpf4r1_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_tpf4r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__tpf8r1", "avx512", radix32_t1_buf_dit_tile64_temporal_tpf8r1_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_tpf8r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__tpf8r2", "avx512", radix32_t1_buf_dit_tile64_temporal_tpf8r2_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_tpf8r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__tpf16r1", "avx512", radix32_t1_buf_dit_tile64_temporal_tpf16r1_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_tpf16r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__tpf16r2", "avx512", radix32_t1_buf_dit_tile64_temporal_tpf16r2_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_tpf16r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__tpf32r1", "avx512", radix32_t1_buf_dit_tile64_temporal_tpf32r1_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_tpf32r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__tpf32r2", "avx512", radix32_t1_buf_dit_tile64_temporal_tpf32r2_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_tpf32r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__prefw", "avx512", radix32_t1_buf_dit_tile64_temporal_prefw_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_prefw_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__prefw__tpf4r1", "avx512", radix32_t1_buf_dit_tile64_temporal_prefw_tpf4r1_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_prefw_tpf4r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__prefw__tpf8r1", "avx512", radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r1_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__prefw__tpf8r2", "avx512", radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r2_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_prefw_tpf8r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__prefw__tpf16r1", "avx512", radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r1_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__prefw__tpf16r2", "avx512", radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r2_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_prefw_tpf16r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__prefw__tpf32r1", "avx512", radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r1_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile64__draintemporal__prefw__tpf32r2", "avx512", radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r2_fwd_avx512, radix32_t1_buf_dit_tile64_temporal_prefw_tpf32r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal", "avx512", radix32_t1_buf_dit_tile128_temporal_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__tpf4r1", "avx512", radix32_t1_buf_dit_tile128_temporal_tpf4r1_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_tpf4r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__tpf8r1", "avx512", radix32_t1_buf_dit_tile128_temporal_tpf8r1_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_tpf8r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__tpf8r2", "avx512", radix32_t1_buf_dit_tile128_temporal_tpf8r2_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_tpf8r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__tpf16r1", "avx512", radix32_t1_buf_dit_tile128_temporal_tpf16r1_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_tpf16r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__tpf16r2", "avx512", radix32_t1_buf_dit_tile128_temporal_tpf16r2_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_tpf16r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__tpf32r1", "avx512", radix32_t1_buf_dit_tile128_temporal_tpf32r1_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_tpf32r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__tpf32r2", "avx512", radix32_t1_buf_dit_tile128_temporal_tpf32r2_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_tpf32r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__prefw", "avx512", radix32_t1_buf_dit_tile128_temporal_prefw_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_prefw_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__prefw__tpf4r1", "avx512", radix32_t1_buf_dit_tile128_temporal_prefw_tpf4r1_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_prefw_tpf4r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__prefw__tpf8r1", "avx512", radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r1_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__prefw__tpf8r2", "avx512", radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r2_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_prefw_tpf8r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__prefw__tpf16r1", "avx512", radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r1_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__prefw__tpf16r2", "avx512", radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r2_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_prefw_tpf16r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__prefw__tpf32r1", "avx512", radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r1_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile128__draintemporal__prefw__tpf32r2", "avx512", radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r2_fwd_avx512, radix32_t1_buf_dit_tile128_temporal_prefw_tpf32r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal", "avx512", radix32_t1_buf_dit_tile256_temporal_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__tpf4r1", "avx512", radix32_t1_buf_dit_tile256_temporal_tpf4r1_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_tpf4r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__tpf8r1", "avx512", radix32_t1_buf_dit_tile256_temporal_tpf8r1_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_tpf8r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__tpf8r2", "avx512", radix32_t1_buf_dit_tile256_temporal_tpf8r2_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_tpf8r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__tpf16r1", "avx512", radix32_t1_buf_dit_tile256_temporal_tpf16r1_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_tpf16r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__tpf16r2", "avx512", radix32_t1_buf_dit_tile256_temporal_tpf16r2_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_tpf16r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__tpf32r1", "avx512", radix32_t1_buf_dit_tile256_temporal_tpf32r1_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_tpf32r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__tpf32r2", "avx512", radix32_t1_buf_dit_tile256_temporal_tpf32r2_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_tpf32r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__prefw", "avx512", radix32_t1_buf_dit_tile256_temporal_prefw_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_prefw_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__prefw__tpf4r1", "avx512", radix32_t1_buf_dit_tile256_temporal_prefw_tpf4r1_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_prefw_tpf4r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__prefw__tpf8r1", "avx512", radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r1_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__prefw__tpf8r2", "avx512", radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r2_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_prefw_tpf8r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__prefw__tpf16r1", "avx512", radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r1_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__prefw__tpf16r2", "avx512", radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r2_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_prefw_tpf16r2_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__prefw__tpf32r1", "avx512", radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r1_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r1_bwd_avx512, 1},
    {"ct_t1_buf_dit__avx512__tile256__draintemporal__prefw__tpf32r2", "avx512", radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r2_fwd_avx512, radix32_t1_buf_dit_tile256_temporal_prefw_tpf32r2_bwd_avx512, 1},
    {NULL,NULL,NULL,NULL,0}
};
typedef struct { size_t ios, me; } sp_t;
static sp_t SW[] = {
    {64, 64},
    {72, 64},
    {128, 64},
    {128, 128},
    {136, 128},
    {192, 128},
    {256, 256},
    {264, 256},
    {320, 256},
    {512, 512},
    {520, 512},
    {576, 512},
    {1024, 1024},
    {1032, 1024},
    {1088, 1024},
    {2048, 2048},
    {2056, 2048},
    {2112, 2048},
    {0,0}
};

int main(int argc, char **argv) {
    int skip512 = !have_avx512();
    int n_cand = sizeof(CANDS)/sizeof(CANDS[0]) - 1;
    int n_sp = sizeof(SW)/sizeof(SW[0]) - 1;
    const char *outpath = argc > 1 ? argv[1] : "measurements.jsonl";

    /* Read existing measurements (resume support) */
    char done_set[161][2] = {{0}};  /* [cand_idx][dir] */
    /* Actually use a set by parsing existing jsonl */
    FILE *rd = fopen(outpath, "r");
    if (rd) {
        char line[1024];
        while (fgets(line, sizeof(line), rd)) {
            /* parse: {"id":"...", "ios":X, "me":Y, "dir":"fwd", ...} */
            /* simple: just skip — we'll count lines and assume in-order */
            /* Better: we append so re-runs just keep going */
        }
        fclose(rd);
    }

    /* Count already-done measurements to skip */
    int already = 0;
    FILE *cnt = fopen(outpath, "r");
    if (cnt) {
        int ch;
        while ((ch = fgetc(cnt)) != EOF) if (ch == '\n') already++;
        fclose(cnt);
    }
    fprintf(stderr, "harness: %d candidates, %d sweeps, %d dirs; skipping %d done\n",
            n_cand, n_sp, 2, already);

    FILE *out = fopen(outpath, "a");
    if (!out) { perror("fopen"); return 1; }

    int total_emitted = 0;
    for (int ci = 0; ci < n_cand; ci++) {
        cand_t *c = &CANDS[ci];
        if (c->req512 && skip512) continue;
        fprintf(stderr, "[%d/%d] %s\n", ci+1, n_cand, c->id);
        for (int sp = 0; sp < n_sp; sp++) {
            size_t ios = SW[sp].ios, me = SW[sp].me;
            for (int d = 0; d < 2; d++) {
                if (total_emitted < already) { total_emitted++; continue; }
                t1_fn fn = (d == 0) ? c->fwd : c->bwd;
                double ns = measure(fn, ios, me);
                fprintf(out, "{\"id\":\"%s\",\"ios\":%zu,\"me\":%zu,\"dir\":\"%s\",\"ns\":%.2f}\n",
                        c->id, ios, me, d == 0 ? "fwd" : "bwd", ns);
                fflush(out);
                total_emitted++;
            }
        }
    }
    fclose(out);
    return 0;
}
