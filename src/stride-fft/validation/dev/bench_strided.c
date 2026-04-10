/*
 * bench_strided.c -- A/B test: strided codelet vs tile buffer
 *
 * Test 1 (R2C fused pack):
 *   A: n1_fwd(is=2K, os=B) — fused stride change in butterfly
 *   B: SIMD copy(is=2K → B) + n1_fwd(is=B, os=B) — separate pack + butterfly
 *
 * Test 2 (2D row FFT tile):
 *   A: AVX2 4x4 tile transpose + standard codelet on dense data
 *   B: Manual gather (4 scalar loads → _mm256_set_pd) + butterfly + dense store
 *
 * Both tests isolate the first-stage cost. Middle stages (t1/t1s) run on
 * dense scratch in all cases, so they're the same and not benchmarked.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#include "../core/compat.h"
#include "../core/env.h"
#include "../core/planner.h"

#define ALIGN64 __attribute__((aligned(64)))


/* ═══════════════════════════════════════════════════════════════
 * TEST 1: R2C FUSED PACK
 *
 * Simulates first stage of R2C inner FFT.
 * Input: real data at re[2n*K + k], packed as even=re, odd=im.
 * Output: scratch at sr[n*B + k], si[n*B + k].
 *
 * The first stage is radix-4 n1 on the first group (4 elements).
 * We test processing ALL groups of the first stage (halfN/4 groups).
 * ═══════════════════════════════════════════════════════════════ */

static void test_r2c_fused_pack(void) {
    printf("=== Test 1: R2C Fused Pack (radix-4 first stage) ===\n\n");

    const int sizes[] = {64, 256, 512, 1024};
    const int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    const size_t K = 256;
    const size_t B = 32;  /* block size */

    printf("%-8s %12s %12s %8s\n", "halfN", "fused_ns", "copy+n1_ns", "speedup");
    printf("--------+-------------+-------------+--------\n");

    for (int si = 0; si < nsizes; si++) {
        const int halfN = sizes[si];
        const int num_groups = halfN / 4;  /* radix-4 groups */

        /* Allocate input: N real values in stride-K layout */
        size_t input_sz = (size_t)(2 * halfN) * K;
        double *input_re = (double *)_aligned_malloc(input_sz * sizeof(double), 64);

        /* Scratch: halfN * B */
        size_t scratch_sz = (size_t)halfN * B;
        double *sr = (double *)_aligned_malloc(scratch_sz * sizeof(double), 64);
        double *si_buf = (double *)_aligned_malloc(scratch_sz * sizeof(double), 64);

        /* Fill input with random data */
        for (size_t i = 0; i < input_sz; i++)
            input_re[i] = (double)rand() / RAND_MAX;

        /* Warmup */
        for (int w = 0; w < 50; w++) {
            for (int g = 0; g < num_groups; g++) {
                size_t in_off = (size_t)(4 * g) * K;  /* skip 4 pairs per group */
                size_t out_off = (size_t)(4 * g) * B;
                radix4_n1_fwd_avx2(
                    input_re + in_off, input_re + K + in_off,
                    sr + out_off, si_buf + out_off,
                    2 * K, B, B);
            }
        }

        int reps = 5000;

        /* ── Approach A: fused n1_fwd(is=2K, os=B) ── */
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (int g = 0; g < num_groups; g++) {
                size_t in_off = (size_t)(4 * g) * K;
                size_t out_off = (size_t)(4 * g) * B;
                radix4_n1_fwd_avx2(
                    input_re + in_off, input_re + K + in_off,
                    sr + out_off, si_buf + out_off,
                    2 * K, B, B);
            }
        }
        double fused_ns = (now_ns() - t0) / reps;

        /* ── Approach B: SIMD copy + n1_fwd(is=B, os=B) ── */
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            /* Pack: copy even/odd rows to scratch */
            for (int n = 0; n < halfN; n++) {
                const double *even = input_re + (size_t)(2 * n) * K;
                const double *odd  = input_re + (size_t)(2 * n + 1) * K;
                double *dst_r = sr + (size_t)n * B;
                double *dst_i = si_buf + (size_t)n * B;
                size_t k = 0;
                for (; k + 4 <= B; k += 4) {
                    _mm256_store_pd(dst_r + k, _mm256_loadu_pd(even + k));
                    _mm256_store_pd(dst_i + k, _mm256_loadu_pd(odd + k));
                }
                for (; k < B; k++) {
                    dst_r[k] = even[k];
                    dst_i[k] = odd[k];
                }
            }
            /* First stage: in-place n1 on dense scratch */
            for (int g = 0; g < num_groups; g++) {
                size_t off = (size_t)(4 * g) * B;
                radix4_n1_fwd_avx2(
                    sr + off, si_buf + off,
                    sr + off, si_buf + off,
                    B, B, B);
            }
        }
        double copy_ns = (now_ns() - t0) / reps;

        printf("%-8d %12.0f %12.0f %7.2fx\n",
               halfN, fused_ns, copy_ns, copy_ns / fused_ns);

        _aligned_free(input_re);
        _aligned_free(sr);
        _aligned_free(si_buf);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * TEST 1b: PACK COST AS FRACTION OF FULL R2C
 *
 * Measures: (1) pack alone, (2) inner FFT alone, (3) full R2C.
 * Shows how much of the R2C pipeline the pack consumes.
 * Also tests: fused n1 for group 0 + pack rest vs full pack.
 * ═══════════════════════════════════════════════════════════════ */

/* Need full planner for this test */
#include "../core/planner.h"

static void test_r2c_pack_fraction(void) {
    printf("\n=== Test 1b: Pack Cost as Fraction of Full R2C ===\n\n");

    const int sizes[] = {256, 1000, 4096};
    const int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    const size_t K = 256;

    /* Build registry */
    stride_registry_t reg;
    stride_registry_init(&reg);

    printf("%-8s %10s %10s %10s %10s %8s\n",
           "N", "pack_ns", "fft_ns", "total_ns", "fused_ns", "pack%");
    printf("--------+----------+----------+----------+----------+--------\n");

    for (int si = 0; si < nsizes; si++) {
        int N = sizes[si];
        int halfN = N / 2;

        /* Create R2C plan (uses inner N/2-point plan with K=B) */
        stride_plan_t *r2c_plan = stride_r2c_auto_plan(N, K, &reg);
        if (!r2c_plan) { printf("%-8d  PLAN FAILED\n", N); continue; }

        stride_r2c_data_t *d = (stride_r2c_data_t *)r2c_plan->override_data;
        size_t B = d->B;

        /* Allocate buffers */
        size_t total_sz = (size_t)N * K;
        double *re = (double *)_aligned_malloc(total_sz * sizeof(double), 64);
        double *im = (double *)_aligned_malloc((size_t)(halfN + 1) * K * sizeof(double), 64);
        double *sr = (double *)_aligned_malloc((size_t)halfN * B * sizeof(double), 64);
        double *si_buf = (double *)_aligned_malloc((size_t)halfN * B * sizeof(double), 64);

        for (size_t i = 0; i < total_sz; i++) re[i] = (double)rand() / RAND_MAX;
        memset(im, 0, (size_t)(halfN + 1) * K * sizeof(double));

        int reps = 2000;

        /* Warmup */
        for (int w = 0; w < 20; w++)
            stride_execute_fwd(r2c_plan, re, im);

        /* ── Measure pack alone ── */
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (size_t b0 = 0; b0 < K; b0 += B) {
                for (int n = 0; n < halfN; n++) {
                    const double *even = re + (size_t)(2*n) * K + b0;
                    const double *odd  = re + (size_t)(2*n+1) * K + b0;
                    double *dst_r = sr + (size_t)n * B;
                    double *dst_i = si_buf + (size_t)n * B;
                    size_t k = 0;
                    for (; k + 4 <= B; k += 4) {
                        _mm256_store_pd(dst_r + k, _mm256_loadu_pd(even + k));
                        _mm256_store_pd(dst_i + k, _mm256_loadu_pd(odd + k));
                    }
                    for (; k < B; k++) { dst_r[k] = even[k]; dst_i[k] = odd[k]; }
                }
            }
        }
        double pack_ns = (now_ns() - t0) / reps;

        /* ── Measure inner FFT alone (on pre-packed scratch) ── */
        /* Pack once for the FFT measurement */
        for (int n = 0; n < halfN; n++) {
            const double *even = re + (size_t)(2*n) * K;
            const double *odd  = re + (size_t)(2*n+1) * K;
            double *dst_r = sr + (size_t)n * B;
            double *dst_i = si_buf + (size_t)n * B;
            for (size_t k = 0; k < B; k++) { dst_r[k] = even[k]; dst_i[k] = odd[k]; }
        }
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            stride_execute_fwd(d->inner, sr, si_buf);
        }
        double fft_ns = (now_ns() - t0) / reps;

        /* ── Measure full R2C ── */
        t0 = now_ns();
        for (int r = 0; r < reps; r++)
            stride_execute_fwd(r2c_plan, re, im);
        double total_ns = (now_ns() - t0) / reps;

        /* ── Measure fused first-group approach ── */
        /* Fuse group 0 of first stage via n1_fwd(is=2K, os=B),
         * pack the rest, then run full inner FFT.
         * This is the realistic "partial fusion" path. */
        stride_n1_fn n1f = d->inner->stages[0].n1_fwd;
        int R0 = d->inner->stages[0].radix;
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (size_t b0 = 0; b0 < K; b0 += B) {
                /* Fused: group 0 (R0 elements) via n1 with is=2K, os=B */
                if (n1f) {
                    n1f(re + b0, re + K + b0, sr, si_buf, 2*K, B, B);
                }
                /* Pack remaining elements (R0..halfN-1) */
                for (int n = R0; n < halfN; n++) {
                    const double *even = re + (size_t)(2*n) * K + b0;
                    const double *odd  = re + (size_t)(2*n+1) * K + b0;
                    double *dst_r = sr + (size_t)n * B;
                    double *dst_i = si_buf + (size_t)n * B;
                    size_t k = 0;
                    for (; k + 4 <= B; k += 4) {
                        _mm256_store_pd(dst_r + k, _mm256_loadu_pd(even + k));
                        _mm256_store_pd(dst_i + k, _mm256_loadu_pd(odd + k));
                    }
                    for (; k < B; k++) { dst_r[k] = even[k]; dst_i[k] = odd[k]; }
                }
                /* Run full inner FFT (first stage will re-process group 0 but it's in scratch now) */
                stride_execute_fwd(d->inner, sr, si_buf);
                /* Post-process omitted — same cost in both paths */
            }
        }
        double fused_ns = (now_ns() - t0) / reps;

        double pack_pct = 100.0 * pack_ns / total_ns;
        printf("%-8d %10.0f %10.0f %10.0f %10.0f %7.1f%%\n",
               N, pack_ns, fft_ns, total_ns, fused_ns, pack_pct);

        stride_plan_destroy(r2c_plan);
        _aligned_free(re);
        _aligned_free(im);
        _aligned_free(sr);
        _aligned_free(si_buf);
    }
}


/* ═══════════════════════════════════════════════════════════════
 * TEST C2R: OPTION B (pre-scale + strided n1_bwd) vs
 *           OPTION C (n1_bwd with ×2 baked into stores)
 *           vs BASELINE (n1_bwd on scratch + separate unpack)
 *
 * Stage 0 backward (DIF, last stage) is twiddle-free.
 * Current: n1_bwd in-place on scratch → separate ×2 strided unpack
 * Opt B:   ×2 scale scratch (dense) → n1_bwd(is=B, os=2K) strided write
 * Opt C:   n1_bwd_scaled — butterfly with ×2 on stores, strided write
 * ═══════════════════════════════════════════════════════════════ */

/* Manual radix-4 n1_bwd_scaled: butterfly + ×2 + strided write */
__attribute__((target("avx2,fma")))
static void radix4_n1_bwd_scaled_avx2(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    size_t is, size_t os, size_t vl)
{
    __m256d two = _mm256_set1_pd(2.0);
    for (size_t k = 0; k < vl; k += 4) {
        __m256d x0r = _mm256_load_pd(&in_re[0*is+k]), x0i = _mm256_load_pd(&in_im[0*is+k]);
        __m256d x1r = _mm256_load_pd(&in_re[1*is+k]), x1i = _mm256_load_pd(&in_im[1*is+k]);
        __m256d x2r = _mm256_load_pd(&in_re[2*is+k]), x2i = _mm256_load_pd(&in_im[2*is+k]);
        __m256d x3r = _mm256_load_pd(&in_re[3*is+k]), x3i = _mm256_load_pd(&in_im[3*is+k]);
        /* Radix-4 DIF butterfly (bwd) */
        __m256d t0r = _mm256_add_pd(x0r, x2r), t0i = _mm256_add_pd(x0i, x2i);
        __m256d t1r = _mm256_sub_pd(x0r, x2r), t1i = _mm256_sub_pd(x0i, x2i);
        __m256d t2r = _mm256_add_pd(x1r, x3r), t2i = _mm256_add_pd(x1i, x3i);
        __m256d t3r = _mm256_sub_pd(x1r, x3r), t3i = _mm256_sub_pd(x1i, x3i);
        /* Store with ×2 fused */
        _mm256_storeu_pd(&out_re[0*os+k], _mm256_mul_pd(two, _mm256_add_pd(t0r, t2r)));
        _mm256_storeu_pd(&out_im[0*os+k], _mm256_mul_pd(two, _mm256_add_pd(t0i, t2i)));
        _mm256_storeu_pd(&out_re[2*os+k], _mm256_mul_pd(two, _mm256_sub_pd(t0r, t2r)));
        _mm256_storeu_pd(&out_im[2*os+k], _mm256_mul_pd(two, _mm256_sub_pd(t0i, t2i)));
        _mm256_storeu_pd(&out_re[1*os+k], _mm256_mul_pd(two, _mm256_sub_pd(t1r, t3i)));
        _mm256_storeu_pd(&out_im[1*os+k], _mm256_mul_pd(two, _mm256_add_pd(t1i, t3r)));
        _mm256_storeu_pd(&out_re[3*os+k], _mm256_mul_pd(two, _mm256_add_pd(t1r, t3i)));
        _mm256_storeu_pd(&out_im[3*os+k], _mm256_mul_pd(two, _mm256_sub_pd(t1i, t3r)));
    }
}

static void test_c2r_unpack_options(void) {
    printf("\n=== Test C2R: Option B (pre-scale+strided) vs C (scaled butterfly) vs Baseline ===\n\n");

    const int halfN_sizes[] = {64, 250, 500, 2048};
    const int nsizes = sizeof(halfN_sizes) / sizeof(halfN_sizes[0]);
    const size_t K = 256;
    const size_t B = 32;

    printf("%-8s %10s %10s %10s  %8s %8s\n",
           "halfN", "base_ns", "optB_ns", "optC_ns", "B/base", "C/base");
    printf("--------+----------+----------+----------+--------+--------\n");

    for (int si = 0; si < nsizes; si++) {
        const int halfN = halfN_sizes[si];
        const int R = 4;
        const int num_groups = halfN / R;

        /* Scratch: halfN * B (dense, post-IFFT data) */
        size_t scratch_sz = (size_t)halfN * B;
        double *sr = (double *)_aligned_malloc(scratch_sz * sizeof(double), 64);
        double *si_buf = (double *)_aligned_malloc(scratch_sz * sizeof(double), 64);

        /* Output: N * K (strided, stride 2K between even/odd) */
        size_t out_sz = (size_t)(2 * halfN) * K;
        double *out_re = (double *)_aligned_malloc(out_sz * sizeof(double), 64);

        /* Fill scratch with random data */
        for (size_t i = 0; i < scratch_sz; i++) {
            sr[i] = (double)rand() / RAND_MAX;
            si_buf[i] = (double)rand() / RAND_MAX;
        }

        int reps = 3000;
        size_t num_blocks = K / B;

        /* Warmup */
        for (int w = 0; w < 30; w++) {
            for (size_t blk = 0; blk < num_blocks; blk++) {
                size_t b0 = blk * B;
                /* Baseline: in-place n1_bwd + unpack */
                for (int g = 0; g < num_groups; g++) {
                    size_t off = (size_t)(R * g) * B;
                    radix4_n1_bwd_avx2(sr + off, si_buf + off, sr + off, si_buf + off, B, B, B);
                }
                for (int n = 0; n < halfN; n++) {
                    double *even = out_re + (size_t)(2 * n) * K + b0;
                    double *odd  = out_re + (size_t)(2 * n + 1) * K + b0;
                    for (size_t k = 0; k + 4 <= B; k += 4) {
                        _mm256_storeu_pd(even + k, _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_load_pd(sr + (size_t)n * B + k)));
                        _mm256_storeu_pd(odd + k, _mm256_mul_pd(_mm256_set1_pd(2.0), _mm256_load_pd(si_buf + (size_t)n * B + k)));
                    }
                }
            }
        }

        /* ── Baseline: in-place n1_bwd on scratch + separate ×2 unpack ── */
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (size_t blk = 0; blk < num_blocks; blk++) {
                size_t b0 = blk * B;
                for (int g = 0; g < num_groups; g++) {
                    size_t off = (size_t)(R * g) * B;
                    radix4_n1_bwd_avx2(sr + off, si_buf + off, sr + off, si_buf + off, B, B, B);
                }
                __m256d two = _mm256_set1_pd(2.0);
                for (int n = 0; n < halfN; n++) {
                    double *even = out_re + (size_t)(2 * n) * K + b0;
                    double *odd  = out_re + (size_t)(2 * n + 1) * K + b0;
                    for (size_t k = 0; k + 4 <= B; k += 4) {
                        _mm256_storeu_pd(even + k, _mm256_mul_pd(two, _mm256_load_pd(sr + (size_t)n * B + k)));
                        _mm256_storeu_pd(odd + k, _mm256_mul_pd(two, _mm256_load_pd(si_buf + (size_t)n * B + k)));
                    }
                }
            }
        }
        double base_ns = (now_ns() - t0) / reps;

        /* ── Option B: ×2 scale scratch (dense) + n1_bwd(is=B, os=2K) ── */
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (size_t blk = 0; blk < num_blocks; blk++) {
                size_t b0 = blk * B;
                /* Pre-scale scratch ×2 (dense, cache-friendly) */
                __m256d two = _mm256_set1_pd(2.0);
                for (size_t i = 0; i + 4 <= scratch_sz; i += 4) {
                    _mm256_store_pd(sr + i, _mm256_mul_pd(two, _mm256_load_pd(sr + i)));
                    _mm256_store_pd(si_buf + i, _mm256_mul_pd(two, _mm256_load_pd(si_buf + i)));
                }
                /* n1_bwd with strided output: is=B, os=2K */
                for (int g = 0; g < num_groups; g++) {
                    size_t scratch_base = (size_t)(R * g) * B;
                    size_t elem_idx = scratch_base / B;
                    size_t out_off = elem_idx * 2 * K + b0;
                    size_t input_leg_stride = (B / B) * 2 * K;  /* elem_per_leg * 2K */
                    radix4_n1_bwd_avx2(sr + scratch_base, si_buf + scratch_base,
                                       out_re + out_off, out_re + K + out_off,
                                       B, input_leg_stride, B);
                }
            }
        }
        double optB_ns = (now_ns() - t0) / reps;

        /* ── Option C: scaled n1_bwd (×2 baked into butterfly stores) ── */
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (size_t blk = 0; blk < num_blocks; blk++) {
                size_t b0 = blk * B;
                for (int g = 0; g < num_groups; g++) {
                    size_t scratch_base = (size_t)(R * g) * B;
                    size_t elem_idx = scratch_base / B;
                    size_t out_off = elem_idx * 2 * K + b0;
                    size_t input_leg_stride = (B / B) * 2 * K;
                    radix4_n1_bwd_scaled_avx2(sr + scratch_base, si_buf + scratch_base,
                                              out_re + out_off, out_re + K + out_off,
                                              B, input_leg_stride, B);
                }
            }
        }
        double optC_ns = (now_ns() - t0) / reps;

        printf("%-8d %10.0f %10.0f %10.0f  %7.2fx %7.2fx\n",
               halfN, base_ns, optB_ns, optC_ns,
               base_ns / optB_ns, base_ns / optC_ns);

        _aligned_free(sr);
        _aligned_free(si_buf);
        _aligned_free(out_re);
    }

    printf("\n  B/base, C/base: >1 means option is faster than baseline\n");
}


/* ═══════════════════════════════════════════════════════════════
 * TEST 3: OPTION 1 (t1_oop) vs OPTION 2 (twiddle-free first stage)
 *
 * Option 1: t1_oop — strided read + twiddle multiply + butterfly + dense write
 *           (twiddle happens during strided access = more register pressure)
 * Option 2: n1_fwd(is=2K, os=B) — strided read + butterfly + dense write
 *           (no twiddle in first stage; twiddles absorbed into stage 2 = free)
 *
 * We also compare against the current baseline: explicit SIMD pack + full pipeline.
 * Tested across multiple K values.
 * ═══════════════════════════════════════════════════════════════ */

/* Manual radix-4 t1_oop: strided read + twiddle + butterfly + dense write.
 * Simulates what a generated t1_oop codelet would do. */
__attribute__((target("avx2,fma")))
static void radix4_t1_oop_fwd_manual(
    const double * __restrict__ in_re, const double * __restrict__ in_im,
    double * __restrict__ out_re, double * __restrict__ out_im,
    const double * __restrict__ W_re, const double * __restrict__ W_im,
    size_t is, size_t os, size_t me)
{
    /* Like t1_dit but out-of-place: reads at stride is, writes at stride os.
     * W layout: W[(j-1)*me + m] for j=1..3. */
    for (size_t m = 0; m < me; m += 4) {
        /* Strided reads */
        __m256d x0r = _mm256_loadu_pd(in_re + m + 0 * is);
        __m256d x0i = _mm256_loadu_pd(in_im + m + 0 * is);
        __m256d r1r = _mm256_loadu_pd(in_re + m + 1 * is);
        __m256d r1i = _mm256_loadu_pd(in_im + m + 1 * is);
        __m256d r2r = _mm256_loadu_pd(in_re + m + 2 * is);
        __m256d r2i = _mm256_loadu_pd(in_im + m + 2 * is);
        __m256d r3r = _mm256_loadu_pd(in_re + m + 3 * is);
        __m256d r3i = _mm256_loadu_pd(in_im + m + 3 * is);

        /* Twiddle multiply: x_j = W_j * r_j for j=1,2,3 */
        __m256d w1r = _mm256_load_pd(W_re + 0 * me + m);
        __m256d w1i = _mm256_load_pd(W_im + 0 * me + m);
        __m256d x1r = _mm256_fmsub_pd(w1r, r1r, _mm256_mul_pd(w1i, r1i));
        __m256d x1i = _mm256_fmadd_pd(w1r, r1i, _mm256_mul_pd(w1i, r1r));

        __m256d w2r = _mm256_load_pd(W_re + 1 * me + m);
        __m256d w2i = _mm256_load_pd(W_im + 1 * me + m);
        __m256d x2r = _mm256_fmsub_pd(w2r, r2r, _mm256_mul_pd(w2i, r2i));
        __m256d x2i = _mm256_fmadd_pd(w2r, r2i, _mm256_mul_pd(w2i, r2r));

        __m256d w3r = _mm256_load_pd(W_re + 2 * me + m);
        __m256d w3i = _mm256_load_pd(W_im + 2 * me + m);
        __m256d x3r = _mm256_fmsub_pd(w3r, r3r, _mm256_mul_pd(w3i, r3i));
        __m256d x3i = _mm256_fmadd_pd(w3r, r3i, _mm256_mul_pd(w3i, r3r));

        /* Butterfly */
        __m256d t0r = _mm256_add_pd(x0r, x2r), t0i = _mm256_add_pd(x0i, x2i);
        __m256d t1r = _mm256_sub_pd(x0r, x2r), t1i = _mm256_sub_pd(x0i, x2i);
        __m256d t2r = _mm256_add_pd(x1r, x3r), t2i = _mm256_add_pd(x1i, x3i);
        __m256d t3r = _mm256_sub_pd(x1r, x3r), t3i = _mm256_sub_pd(x1i, x3i);

        /* Dense writes */
        _mm256_store_pd(out_re + m + 0 * os, _mm256_add_pd(t0r, t2r));
        _mm256_store_pd(out_im + m + 0 * os, _mm256_add_pd(t0i, t2i));
        _mm256_store_pd(out_re + m + 2 * os, _mm256_sub_pd(t0r, t2r));
        _mm256_store_pd(out_im + m + 2 * os, _mm256_sub_pd(t0i, t2i));
        _mm256_store_pd(out_re + m + 1 * os, _mm256_add_pd(t1r, t3i));
        _mm256_store_pd(out_im + m + 1 * os, _mm256_sub_pd(t1i, t3r));
        _mm256_store_pd(out_re + m + 3 * os, _mm256_sub_pd(t1r, t3i));
        _mm256_store_pd(out_im + m + 3 * os, _mm256_add_pd(t1i, t3r));
    }
}

static void test_option1_vs_option2(void) {
    printf("\n=== Test 3: Option 1 (t1_oop) vs Option 2 (twiddle-free) vs Baseline ===\n");
    printf("  Opt1: strided read + twiddle + butterfly + dense write (t1_oop)\n");
    printf("  Opt2: strided read + butterfly + dense write (n1_fwd, twiddles absorbed into stage 2)\n");
    printf("  Base: SIMD pack + in-place n1/t1 on dense scratch\n\n");

    const int halfN = 500;  /* N=1000 */
    const int R = 4;
    const int num_groups = halfN / R;
    const size_t B = 32;
    const size_t K_values[] = {32, 64, 128, 256, 512, 1024};
    const int nK = sizeof(K_values) / sizeof(K_values[0]);

    printf("%-8s %10s %10s %10s  %8s %8s\n",
           "K", "opt1_ns", "opt2_ns", "base_ns", "opt1/b", "opt2/b");
    printf("--------+----------+----------+----------+--------+--------\n");

    for (int ki = 0; ki < nK; ki++) {
        size_t K = K_values[ki];
        if (B > K) continue;  /* B must fit in K */

        /* Input buffer: N real values at stride K */
        size_t input_sz = (size_t)(2 * halfN) * K;
        double *input_re = (double *)_aligned_malloc(input_sz * sizeof(double), 64);

        /* Scratch */
        size_t scratch_sz = (size_t)halfN * B;
        double *sr = (double *)_aligned_malloc(scratch_sz * sizeof(double), 64);
        double *si_buf = (double *)_aligned_malloc(scratch_sz * sizeof(double), 64);

        /* Dummy twiddle table for t1_oop (3 * me doubles per group) */
        double *tw_re = (double *)_aligned_malloc(3 * B * sizeof(double), 64);
        double *tw_im = (double *)_aligned_malloc(3 * B * sizeof(double), 64);
        for (size_t i = 0; i < 3 * B; i++) {
            tw_re[i] = cos(2.0 * M_PI * (double)i / (double)(halfN));
            tw_im[i] = sin(2.0 * M_PI * (double)i / (double)(halfN));
        }

        /* Fill input */
        for (size_t i = 0; i < input_sz; i++)
            input_re[i] = (double)rand() / RAND_MAX;

        int reps = 3000;
        size_t num_blocks = K / B;

        /* Warmup all paths */
        for (int w = 0; w < 30; w++) {
            for (size_t blk = 0; blk < num_blocks; blk++) {
                size_t b0 = blk * B;
                for (int g = 0; g < num_groups; g++) {
                    size_t in_off = (size_t)(R * g) * K + b0;
                    size_t out_off = (size_t)(R * g) * B;
                    radix4_n1_fwd_avx2(input_re + in_off, input_re + K + in_off,
                                       sr + out_off, si_buf + out_off, 2*K, B, B);
                }
            }
        }

        /* ── Option 1: t1_oop (strided + twiddle + butterfly) ── */
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (size_t blk = 0; blk < num_blocks; blk++) {
                size_t b0 = blk * B;
                /* Group 0: n1 (no twiddle) */
                radix4_n1_fwd_avx2(input_re + b0, input_re + K + b0,
                                   sr, si_buf, 2*K, B, B);
                /* Groups 1+: t1_oop (with twiddle) */
                for (int g = 1; g < num_groups; g++) {
                    size_t in_off = (size_t)(R * g) * K + b0;
                    size_t out_off = (size_t)(R * g) * B;
                    radix4_t1_oop_fwd_manual(
                        input_re + in_off, input_re + K + in_off,
                        sr + out_off, si_buf + out_off,
                        tw_re, tw_im, 2*K, B, B);
                }
            }
        }
        double opt1_ns = (now_ns() - t0) / reps;

        /* ── Option 2: n1_fwd for ALL groups (no twiddle, twiddles absorbed) ── */
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (size_t blk = 0; blk < num_blocks; blk++) {
                size_t b0 = blk * B;
                for (int g = 0; g < num_groups; g++) {
                    size_t in_off = (size_t)(R * g) * K + b0;
                    size_t out_off = (size_t)(R * g) * B;
                    radix4_n1_fwd_avx2(input_re + in_off, input_re + K + in_off,
                                       sr + out_off, si_buf + out_off, 2*K, B, B);
                }
            }
        }
        double opt2_ns = (now_ns() - t0) / reps;

        /* ── Baseline: SIMD pack + in-place first stage ── */
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (size_t blk = 0; blk < num_blocks; blk++) {
                size_t b0 = blk * B;
                /* Pack all halfN elements */
                for (int n = 0; n < halfN; n++) {
                    const double *even = input_re + (size_t)(2*n) * K + b0;
                    const double *odd  = input_re + (size_t)(2*n+1) * K + b0;
                    double *dst_r = sr + (size_t)n * B;
                    double *dst_i = si_buf + (size_t)n * B;
                    for (size_t k = 0; k + 4 <= B; k += 4) {
                        _mm256_store_pd(dst_r + k, _mm256_loadu_pd(even + k));
                        _mm256_store_pd(dst_i + k, _mm256_loadu_pd(odd + k));
                    }
                }
                /* In-place first stage: n1 group 0, then t1 groups 1+ */
                radix4_n1_fwd_avx2(sr, si_buf, sr, si_buf, B, B, B);
                for (int g = 1; g < num_groups; g++) {
                    size_t off = (size_t)(R * g) * B;
                    radix4_t1_dit_fwd_avx2(sr + off, si_buf + off,
                                           tw_re, tw_im, B, B);
                }
            }
        }
        double base_ns = (now_ns() - t0) / reps;

        printf("%-8zu %10.0f %10.0f %10.0f  %7.2fx %7.2fx\n",
               K, opt1_ns, opt2_ns, base_ns,
               base_ns / opt1_ns, base_ns / opt2_ns);

        _aligned_free(input_re);
        _aligned_free(sr);
        _aligned_free(si_buf);
        _aligned_free(tw_re);
        _aligned_free(tw_im);
    }

    printf("\n  opt1/b = baseline/option1 (>1 = option1 faster)\n");
    printf("  opt2/b = baseline/option2 (>1 = option2 faster)\n");
}


/* ═══════════════════════════════════════════════════════════════
 * TEST 4: 2D ROW FFT TILE STRATEGIES
 *
 * Simulates processing B rows of an N1×N2 array for axis-1 FFT.
 * Input: row-major re[row * N2 + col], batch elements at stride N2.
 *
 * A: AVX2 4x4 tile transpose (B rows → N2×B column-major scratch),
 *    then standard radix-4 n1_fwd on dense scratch.
 * B: Manual gather (load 4 rows' element j via _mm256_set_pd),
 *    radix-4 butterfly in registers, store dense to scratch.
 * ═══════════════════════════════════════════════════════════════ */

/* AVX2 4×4 double transpose: 4 rows → 4 columns */
__attribute__((target("avx2,fma")))
static inline void transpose_4x4_store(
    __m256d r0, __m256d r1, __m256d r2, __m256d r3,
    double *dst, size_t dst_stride)
{
    __m256d lo01 = _mm256_unpacklo_pd(r0, r1);
    __m256d hi01 = _mm256_unpackhi_pd(r0, r1);
    __m256d lo23 = _mm256_unpacklo_pd(r2, r3);
    __m256d hi23 = _mm256_unpackhi_pd(r2, r3);
    _mm256_store_pd(dst + 0 * dst_stride, _mm256_permute2f128_pd(lo01, lo23, 0x20));
    _mm256_store_pd(dst + 1 * dst_stride, _mm256_permute2f128_pd(hi01, hi23, 0x20));
    _mm256_store_pd(dst + 2 * dst_stride, _mm256_permute2f128_pd(lo01, lo23, 0x31));
    _mm256_store_pd(dst + 3 * dst_stride, _mm256_permute2f128_pd(hi01, hi23, 0x31));
}

/* Tile transpose: B rows × N2 cols (row-major) → N2 × B (column-major) */
__attribute__((target("avx2,fma")))
static void tile_transpose_to_scratch(
    const double *src, double *dst,
    size_t N2, size_t B, size_t src_stride)
{
    /* Process 4 rows × 4 cols at a time */
    for (size_t b = 0; b < B; b += 4) {
        const double *r0 = src + b * src_stride;
        const double *r1 = src + (b + 1) * src_stride;
        const double *r2 = src + (b + 2) * src_stride;
        const double *r3 = src + (b + 3) * src_stride;

        for (size_t j = 0; j < N2; j += 4) {
            __m256d a = _mm256_loadu_pd(r0 + j);
            __m256d b_v = _mm256_loadu_pd(r1 + j);
            __m256d c = _mm256_loadu_pd(r2 + j);
            __m256d d = _mm256_loadu_pd(r3 + j);
            transpose_4x4_store(a, b_v, c, d, dst + j * B + b, B);
        }
    }
}

/* Tile transpose back: N2 × B (column-major) → B rows × N2 cols (row-major) */
__attribute__((target("avx2,fma")))
static void tile_transpose_from_scratch(
    const double *src, double *dst,
    size_t N2, size_t B, size_t dst_stride)
{
    for (size_t b = 0; b < B; b += 4) {
        double *r0 = dst + b * dst_stride;
        double *r1 = dst + (b + 1) * dst_stride;
        double *r2 = dst + (b + 2) * dst_stride;
        double *r3 = dst + (b + 3) * dst_stride;

        for (size_t j = 0; j < N2; j += 4) {
            __m256d a = _mm256_load_pd(src + j * B + b);
            __m256d b_v = _mm256_load_pd(src + (j + 1) * B + b);
            __m256d c = _mm256_load_pd(src + (j + 2) * B + b);
            __m256d d = _mm256_load_pd(src + (j + 3) * B + b);
            transpose_4x4_store(a, b_v, c, d, r0 + j, 1);
            /* Ugh, dst_stride != 1. Store row by row. */
            /* Actually we need a different approach for non-unit dst_stride. */
            /* Let's just store contiguously within each row — rows are contiguous. */
        }
    }
    /* Simpler: just reverse the forward transpose logic */
    /* For benchmarking, we only measure forward transpose + FFT */
}

/* Manual gather: load element j from B=4 rows at stride N2 */
__attribute__((target("avx2,fma")))
static inline __m256d gather_4rows(const double *base, size_t j, size_t N2) {
    return _mm256_set_pd(
        base[3 * N2 + j],
        base[2 * N2 + j],
        base[1 * N2 + j],
        base[0 * N2 + j]);
}

/* Radix-4 butterfly with gathered inputs, dense output */
__attribute__((target("avx2,fma")))
static inline void radix4_gather_fwd(
    const double *in_re, const double *in_im,
    double *out_re, double *out_im,
    size_t N2, size_t num_groups, size_t B)
{
    /* Process 4 rows simultaneously, 4 FFT elements per radix-4 group */
    for (size_t g = 0; g < num_groups; g++) {
        size_t j0 = g * 4;  /* starting column for this group */

        /* Gather 4 elements from 4 rows (B=4 here) */
        __m256d x0r = gather_4rows(in_re, j0 + 0, N2);
        __m256d x0i = gather_4rows(in_im, j0 + 0, N2);
        __m256d x1r = gather_4rows(in_re, j0 + 1, N2);
        __m256d x1i = gather_4rows(in_im, j0 + 1, N2);
        __m256d x2r = gather_4rows(in_re, j0 + 2, N2);
        __m256d x2i = gather_4rows(in_im, j0 + 2, N2);
        __m256d x3r = gather_4rows(in_re, j0 + 3, N2);
        __m256d x3i = gather_4rows(in_im, j0 + 3, N2);

        /* Radix-4 DIT butterfly */
        __m256d t0r = _mm256_add_pd(x0r, x2r), t0i = _mm256_add_pd(x0i, x2i);
        __m256d t1r = _mm256_sub_pd(x0r, x2r), t1i = _mm256_sub_pd(x0i, x2i);
        __m256d t2r = _mm256_add_pd(x1r, x3r), t2i = _mm256_add_pd(x1i, x3i);
        __m256d t3r = _mm256_sub_pd(x1r, x3r), t3i = _mm256_sub_pd(x1i, x3i);

        /* Store dense to scratch: out[bin * B + batch] */
        size_t off = j0 * B;
        _mm256_store_pd(out_re + off + 0 * B, _mm256_add_pd(t0r, t2r));
        _mm256_store_pd(out_im + off + 0 * B, _mm256_add_pd(t0i, t2i));
        _mm256_store_pd(out_re + off + 2 * B, _mm256_sub_pd(t0r, t2r));
        _mm256_store_pd(out_im + off + 2 * B, _mm256_sub_pd(t0i, t2i));
        _mm256_store_pd(out_re + off + 1 * B, _mm256_add_pd(t1r, t3i));
        _mm256_store_pd(out_im + off + 1 * B, _mm256_sub_pd(t1i, t3r));
        _mm256_store_pd(out_re + off + 3 * B, _mm256_sub_pd(t1r, t3i));
        _mm256_store_pd(out_im + off + 3 * B, _mm256_add_pd(t1i, t3r));
    }
}


static void test_2d_tile(void) {
    printf("\n=== Test 2: 2D Row FFT Tile Strategies (radix-4 first stage) ===\n\n");

    /* Fixed B=4 for gather (4 rows fit one YMM register) */
    const size_t B = 4;
    const int col_sizes[] = {64, 256, 512, 1024};
    const int nsizes = sizeof(col_sizes) / sizeof(col_sizes[0]);
    const size_t N1 = 256;  /* total rows (for realistic buffer sizes) */

    printf("%-8s %12s %12s %8s\n", "N2", "tile_ns", "gather_ns", "speedup");
    printf("--------+-------------+-------------+--------\n");

    for (int ci = 0; ci < nsizes; ci++) {
        const size_t N2 = (size_t)col_sizes[ci];
        const size_t num_groups = N2 / 4;

        /* Row-major 2D array: N1 rows × N2 cols */
        double *arr_re = (double *)_aligned_malloc(N1 * N2 * sizeof(double), 64);
        double *arr_im = (double *)_aligned_malloc(N1 * N2 * sizeof(double), 64);

        /* Scratch: N2 × B (column-major tile) */
        double *sr = (double *)_aligned_malloc(N2 * B * sizeof(double), 64);
        double *si_buf = (double *)_aligned_malloc(N2 * B * sizeof(double), 64);

        /* Fill with random data */
        for (size_t i = 0; i < N1 * N2; i++) {
            arr_re[i] = (double)rand() / RAND_MAX;
            arr_im[i] = (double)rand() / RAND_MAX;
        }

        int reps = 5000;
        /* Process all row-tiles (N1/B tiles of B rows each) */
        size_t num_tiles = N1 / B;

        /* Warmup */
        for (int w = 0; w < 50; w++) {
            for (size_t t = 0; t < num_tiles; t++) {
                tile_transpose_to_scratch(arr_re + t * B * N2, sr, N2, B, N2);
                tile_transpose_to_scratch(arr_im + t * B * N2, si_buf, N2, B, N2);
                for (size_t g = 0; g < num_groups; g++) {
                    size_t off = g * 4 * B;
                    radix4_n1_fwd_avx2(sr + off, si_buf + off, sr + off, si_buf + off, B, B, B);
                }
            }
        }

        /* ── Approach A: tile transpose + standard n1 ── */
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (size_t t = 0; t < num_tiles; t++) {
                const double *src_re = arr_re + t * B * N2;
                const double *src_im = arr_im + t * B * N2;
                /* Transpose B rows × N2 cols → N2 × B scratch */
                tile_transpose_to_scratch(src_re, sr, N2, B, N2);
                tile_transpose_to_scratch(src_im, si_buf, N2, B, N2);
                /* Radix-4 first stage on dense scratch */
                for (size_t g = 0; g < num_groups; g++) {
                    size_t off = g * 4 * B;
                    radix4_n1_fwd_avx2(sr + off, si_buf + off, sr + off, si_buf + off, B, B, B);
                }
            }
        }
        double tile_ns = (now_ns() - t0) / reps;

        /* ── Approach B: gather codelet (fused gather + butterfly) ── */
        t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            for (size_t t = 0; t < num_tiles; t++) {
                const double *src_re = arr_re + t * B * N2;
                const double *src_im = arr_im + t * B * N2;
                /* Gather + butterfly → dense scratch */
                radix4_gather_fwd(src_re, src_im, sr, si_buf, N2, num_groups, B);
            }
        }
        double gather_ns = (now_ns() - t0) / reps;

        printf("%-8d %12.0f %12.0f %7.2fx\n",
               (int)N2, tile_ns, gather_ns, tile_ns / gather_ns);

        _aligned_free(arr_re);
        _aligned_free(arr_im);
        _aligned_free(sr);
        _aligned_free(si_buf);
    }
}


int main(void) {
    stride_env_init();
    stride_pin_thread(0);

    test_r2c_fused_pack();
    test_r2c_pack_fraction();
    test_c2r_unpack_options();
    test_option1_vs_option2();
    test_2d_tile();

    printf("\nDone.\n");
    return 0;
}
