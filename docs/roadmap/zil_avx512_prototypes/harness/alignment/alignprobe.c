/* alignprobe.c — INDICATIVE (VM, 4 vCPU): what does 64-byte misalignment cost?
 *
 * PART A: a streaming IL kernel (BYTW2 twiddle apply + butterfly-ish add, the
 * shape of every IL pass: out = c*x + s*cflip(x), in place), over N complex,
 * base offset 0/16/32/48 bytes mod 64, at zmm (4 complex) and ymm (2 complex).
 * A zmm access at offset != 0 splits a cache line on EVERY access; a ymm access
 * splits on 1 of 2 (offset 32) or 1 of 2 (16, 48) accesses... measured, not assumed.
 * Working sets: L1 (16 KiB), L2 (512 KiB), DRAM-ish (64 MiB).
 *
 * PART B: masked zmm loads/stores with only 16 bytes active (rem = 1 complex),
 * at offsets 0/16/32/48: does the masked-OFF part still pay the line split?
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>

static double now_ns(void)
{ struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }

__attribute__((target("avx512f,avx512dq,avx512vl,fma"), noinline))
static void pass_z(double *x, size_t n, const double *tw)
{
    const __m512d c = _mm512_loadu_pd(tw), s = _mm512_loadu_pd(tw + 8);
    for (size_t i = 0; i + 4 <= n; i += 4) {
        __m512d v = _mm512_loadu_pd(x + 2 * i);
        v = _mm512_fmadd_pd(c, v, _mm512_mul_pd(s, _mm512_permute_pd(v, 0x55)));
        _mm512_storeu_pd(x + 2 * i, v);
    }
}
__attribute__((target("avx512f,avx512dq,avx512vl,fma"), noinline))
static void pass_y(double *x, size_t n, const double *tw)
{
    const __m256d c = _mm256_loadu_pd(tw), s = _mm256_loadu_pd(tw + 8);
    for (size_t i = 0; i + 2 <= n; i += 2) {
        __m256d v = _mm256_loadu_pd(x + 2 * i);
        v = _mm256_fmadd_pd(c, v, _mm256_mul_pd(s, _mm256_permute_pd(v, 0x5)));
        _mm256_storeu_pd(x + 2 * i, v);
    }
}
/* read-only variant (loads dominate: sum), to separate load splits from store splits */
__attribute__((target("avx512f,avx512dq,avx512vl,fma"), noinline))
static double read_z(const double *x, size_t n)
{
    __m512d a0 = _mm512_setzero_pd(), a1 = a0;
    for (size_t i = 0; i + 8 <= n; i += 8) {
        a0 = _mm512_add_pd(a0, _mm512_loadu_pd(x + 2 * i));
        a1 = _mm512_add_pd(a1, _mm512_loadu_pd(x + 2 * i + 8));
    }
    return _mm512_reduce_add_pd(_mm512_add_pd(a0, a1));
}
__attribute__((target("avx512f,avx512dq,avx512vl,fma"), noinline))
static double read_y(const double *x, size_t n)
{
    __m256d a0 = _mm256_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i + 8 <= n; i += 8) {
        a0 = _mm256_add_pd(a0, _mm256_loadu_pd(x + 2 * i));
        a1 = _mm256_add_pd(a1, _mm256_loadu_pd(x + 2 * i + 4));
        a2 = _mm256_add_pd(a2, _mm256_loadu_pd(x + 2 * i + 8));
        a3 = _mm256_add_pd(a3, _mm256_loadu_pd(x + 2 * i + 12));
    }
    __m256d s = _mm256_add_pd(_mm256_add_pd(a0, a1), _mm256_add_pd(a2, a3));
    double o[4]; _mm256_storeu_pd(o, s); return o[0] + o[1] + o[2] + o[3];
}

/* PART B: stride-64B masked accesses, one per line, with 16 active bytes at the
 * START of the vector; offset controls whether the FULL 64-byte vector crosses. */
__attribute__((target("avx512f,avx512dq,avx512vl,fma"), noinline))
static double mload_z(const double *x, size_t nlines, __mmask8 m)
{
    __m512d a0 = _mm512_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i + 4 <= nlines; i += 4) {
        a0 = _mm512_add_pd(a0, _mm512_maskz_loadu_pd(m, x + 8 * i));
        a1 = _mm512_add_pd(a1, _mm512_maskz_loadu_pd(m, x + 8 * i + 8));
        a2 = _mm512_add_pd(a2, _mm512_maskz_loadu_pd(m, x + 8 * i + 16));
        a3 = _mm512_add_pd(a3, _mm512_maskz_loadu_pd(m, x + 8 * i + 24));
    }
    return _mm512_reduce_add_pd(_mm512_add_pd(_mm512_add_pd(a0, a1), _mm512_add_pd(a2, a3)));
}
__attribute__((target("avx512f,avx512dq,avx512vl,fma"), noinline))
static void mstore_z(double *x, size_t nlines, __mmask8 m, __m512d v)
{
    for (size_t i = 0; i < nlines; i++) _mm512_mask_storeu_pd(x + 8 * i, m, v);
}
__attribute__((target("avx512f,avx512dq,avx512vl,fma"), noinline))
static double load_x(const double *x, size_t nlines)
{
    __m128d a0 = _mm_setzero_pd(), a1 = a0, a2 = a0, a3 = a0;
    for (size_t i = 0; i + 4 <= nlines; i += 4) {
        a0 = _mm_add_pd(a0, _mm_loadu_pd(x + 8 * i));
        a1 = _mm_add_pd(a1, _mm_loadu_pd(x + 8 * i + 8));
        a2 = _mm_add_pd(a2, _mm_loadu_pd(x + 8 * i + 16));
        a3 = _mm_add_pd(a3, _mm_loadu_pd(x + 8 * i + 24));
    }
    __m128d s = _mm_add_pd(_mm_add_pd(a0, a1), _mm_add_pd(a2, a3));
    return s[0] + s[1];
}
__attribute__((target("avx512f,avx512dq,avx512vl,fma"), noinline))
static void store_x(double *x, size_t nlines, __m128d v)
{
    for (size_t i = 0; i < nlines; i++) _mm_storeu_pd(x + 8 * i, v);
}

volatile double sink;

int main(int argc, char **argv)
{
    size_t sizes[] = { 16u << 10, 512u << 10, 64u << 20 };   /* bytes of complex data */
    const char *sname[] = { "L1 16KiB", "L2 512KiB", "DRAM 64MiB" };
    double tw[16]; for (int i = 0; i < 16; i++) tw[i] = 0.5 + 0.01 * i;
    char *raw = aligned_alloc(4096, (64u << 20) + 8192);
    memset(raw, 0, (64u << 20) + 8192);
    printf("PART A: streaming in-place IL twiddle pass, ns per complex (min of trials)\n");
    printf("%-11s %-5s %8s %8s %8s %8s   | read-only: %8s %8s %8s %8s\n", "set", "width",
           "off0", "off16", "off32", "off48", "off0", "off16", "off32", "off48");
    for (int s = 0; s < 3; s++) {
        size_t n = sizes[s] / 16;
        long reps = (long)(64e6 / n); if (reps < 3) reps = 3;
        for (int w = 0; w < 2; w++) {
            double r[4], rr[4];
            for (int o = 0; o < 4; o++) {
                double *x = (double *)(raw + 16 * o);
                double best = 1e30, bestr = 1e30;
                for (int t = 0; t < 5; t++) {
                    double t0 = now_ns();
                    for (long k = 0; k < reps; k++) (w ? pass_y : pass_z)(x, n, tw);
                    double dt = (now_ns() - t0) / reps / n; if (dt < best) best = dt;
                    t0 = now_ns();
                    for (long k = 0; k < reps; k++) sink = (w ? read_y : read_z)(x, n);
                    dt = (now_ns() - t0) / reps / n; if (dt < bestr) bestr = dt;
                }
                r[o] = best; rr[o] = bestr;
            }
            printf("%-11s %-5s %8.3f %8.3f %8.3f %8.3f   |            %8.3f %8.3f %8.3f %8.3f\n",
                   sname[s], w ? "ymm" : "zmm", r[0], r[1], r[2], r[3], rr[0], rr[1], rr[2], rr[3]);
        }
    }
    printf("\nPART B: one access per 64B line, L1-resident (256 lines), ns per access\n");
    printf("%-34s %8s %8s %8s %8s\n", "access", "off0", "off16", "off32", "off48");
    {
        size_t nl = 256; long reps = 200000;
        const char *nm[] = { "maskz_loadu zmm, 16B active (0x03)", "maskz_loadu zmm, 64B active (0xFF)",
                             "mask_storeu zmm, 16B active (0x03)", "mask_storeu zmm, 64B active (0xFF)",
                             "loadu xmm (16B)", "storeu xmm (16B)" };
        for (int kind = 0; kind < 6; kind++) {
            double r[4];
            for (int o = 0; o < 4; o++) {
                double *x = (double *)(raw + 16 * o);
                double best = 1e30;
                for (int t = 0; t < 5; t++) {
                    double t0 = now_ns();
                    for (long k = 0; k < reps; k++) {
                        switch (kind) {
                        case 0: sink = mload_z(x, nl, 0x03); break;
                        case 1: sink = mload_z(x, nl, 0xFF); break;
                        case 2: mstore_z(x, nl, 0x03, _mm512_set1_pd(1.0)); break;
                        case 3: mstore_z(x, nl, 0xFF, _mm512_set1_pd(1.0)); break;
                        case 4: sink = load_x(x, nl); break;
                        case 5: store_x(x, nl, _mm_set1_pd(1.0)); break;
                        }
                    }
                    double dt = (now_ns() - t0) / reps / nl; if (dt < best) best = dt;
                }
                r[o] = best;
            }
            printf("%-34s %8.3f %8.3f %8.3f %8.3f\n", nm[kind], r[0], r[1], r[2], r[3]);
        }
    }
    return 0;
}
