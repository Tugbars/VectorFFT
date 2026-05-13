/* bench_strided_2d.c — validate + bench Design C strided codelets.
 *
 * For each (B, N2):
 *   1. Run strided codelet on a B×N2 matrix in-place (matrix→regs→matrix).
 *   2. Run gather + standard codelet + scatter as reference (current 2D path).
 *   3. Verify outputs match to FP noise.
 *   4. Time both, report ratio.
 *
 * "Strided wins" means Design C's load-fused transpose actually pays off
 * vs the existing separate gather/scatter passes. */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stddef.h>
#include <immintrin.h>

/* Strided codelets (Design C). */
__attribute__((target("avx2,fma"))) void radix16_n1_fwd_avx2_gen_strided(
    double *rio_re, double *rio_im,
    const double *tw_re, const double *tw_im,
    size_t row_stride, size_t me);
__attribute__((target("avx2,fma"))) void radix32_n1_fwd_avx2_gen_strided(
    double *rio_re, double *rio_im,
    const double *tw_re, const double *tw_im,
    size_t row_stride, size_t me);
__attribute__((target("avx2,fma"))) void radix64_n1_fwd_avx2_gen_strided(
    double *rio_re, double *rio_im,
    const double *tw_re, const double *tw_im,
    size_t row_stride, size_t me);

/* Standard OOP codelets (reference path). */
__attribute__((target("avx2,fma"))) void radix16_n1_fwd_avx2_gen(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix32_n1_fwd_avx2_gen(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);
__attribute__((target("avx2,fma"))) void radix64_n1_fwd_avx2_gen(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t);

typedef void (*strided_fn_t)(double *, double *,
                             const double *, const double *,
                             size_t, size_t);
typedef void (*std_fn_t)(const double *, const double *, double *, double *,
                         const double *, const double *, size_t);

static double *aa(size_t n) {
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double)) != 0) exit(1);
    return (double *)p;
}
static void fr(double *p, size_t n, unsigned s) {
    for (size_t i = 0; i < n; i++) {
        s = s * 1103515245u + 12345u;
        p[i] = (double)((int)(s >> 8) & 0x7fffff) / (double)0x800000 - 0.5;
    }
}
static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

/* Gather B rows from matrix (B×N at stride N) into scratch (N×B layout). */
__attribute__((target("avx2,fma")))
static void gather_b_x_n(const double *src_re, const double *src_im,
                         double *dst_re, double *dst_im,
                         size_t N, size_t B) {
    /* src is B rows × N cols at row stride N (contiguous).
     * dst is N rows × B cols at row stride B (codelet's K=B layout). */
    for (size_t b = 0; b < B; b += 4) {
        for (size_t k = 0; k < N; k += 4) {
            /* 4×4 transpose block */
            __m256d r0 = _mm256_loadu_pd(&src_re[(b+0)*N + k]);
            __m256d r1 = _mm256_loadu_pd(&src_re[(b+1)*N + k]);
            __m256d r2 = _mm256_loadu_pd(&src_re[(b+2)*N + k]);
            __m256d r3 = _mm256_loadu_pd(&src_re[(b+3)*N + k]);
            __m256d t0 = _mm256_unpacklo_pd(r0, r1);
            __m256d t1 = _mm256_unpackhi_pd(r0, r1);
            __m256d t2 = _mm256_unpacklo_pd(r2, r3);
            __m256d t3 = _mm256_unpackhi_pd(r2, r3);
            _mm256_storeu_pd(&dst_re[(k+0)*B + b], _mm256_permute2f128_pd(t0, t2, 0x20));
            _mm256_storeu_pd(&dst_re[(k+1)*B + b], _mm256_permute2f128_pd(t1, t3, 0x20));
            _mm256_storeu_pd(&dst_re[(k+2)*B + b], _mm256_permute2f128_pd(t0, t2, 0x31));
            _mm256_storeu_pd(&dst_re[(k+3)*B + b], _mm256_permute2f128_pd(t1, t3, 0x31));
            r0 = _mm256_loadu_pd(&src_im[(b+0)*N + k]);
            r1 = _mm256_loadu_pd(&src_im[(b+1)*N + k]);
            r2 = _mm256_loadu_pd(&src_im[(b+2)*N + k]);
            r3 = _mm256_loadu_pd(&src_im[(b+3)*N + k]);
            t0 = _mm256_unpacklo_pd(r0, r1);
            t1 = _mm256_unpackhi_pd(r0, r1);
            t2 = _mm256_unpacklo_pd(r2, r3);
            t3 = _mm256_unpackhi_pd(r2, r3);
            _mm256_storeu_pd(&dst_im[(k+0)*B + b], _mm256_permute2f128_pd(t0, t2, 0x20));
            _mm256_storeu_pd(&dst_im[(k+1)*B + b], _mm256_permute2f128_pd(t1, t3, 0x20));
            _mm256_storeu_pd(&dst_im[(k+2)*B + b], _mm256_permute2f128_pd(t0, t2, 0x31));
            _mm256_storeu_pd(&dst_im[(k+3)*B + b], _mm256_permute2f128_pd(t1, t3, 0x31));
        }
    }
}

/* Inverse of gather: scatter scratch (N×B layout) back to matrix (B×N). */
__attribute__((target("avx2,fma")))
static void scatter_n_x_b(const double *src_re, const double *src_im,
                          double *dst_re, double *dst_im,
                          size_t N, size_t B) {
    for (size_t k = 0; k < N; k += 4) {
        for (size_t b = 0; b < B; b += 4) {
            __m256d r0 = _mm256_loadu_pd(&src_re[(k+0)*B + b]);
            __m256d r1 = _mm256_loadu_pd(&src_re[(k+1)*B + b]);
            __m256d r2 = _mm256_loadu_pd(&src_re[(k+2)*B + b]);
            __m256d r3 = _mm256_loadu_pd(&src_re[(k+3)*B + b]);
            __m256d t0 = _mm256_unpacklo_pd(r0, r1);
            __m256d t1 = _mm256_unpackhi_pd(r0, r1);
            __m256d t2 = _mm256_unpacklo_pd(r2, r3);
            __m256d t3 = _mm256_unpackhi_pd(r2, r3);
            _mm256_storeu_pd(&dst_re[(b+0)*N + k], _mm256_permute2f128_pd(t0, t2, 0x20));
            _mm256_storeu_pd(&dst_re[(b+1)*N + k], _mm256_permute2f128_pd(t1, t3, 0x20));
            _mm256_storeu_pd(&dst_re[(b+2)*N + k], _mm256_permute2f128_pd(t0, t2, 0x31));
            _mm256_storeu_pd(&dst_re[(b+3)*N + k], _mm256_permute2f128_pd(t1, t3, 0x31));
            r0 = _mm256_loadu_pd(&src_im[(k+0)*B + b]);
            r1 = _mm256_loadu_pd(&src_im[(k+1)*B + b]);
            r2 = _mm256_loadu_pd(&src_im[(k+2)*B + b]);
            r3 = _mm256_loadu_pd(&src_im[(k+3)*B + b]);
            t0 = _mm256_unpacklo_pd(r0, r1);
            t1 = _mm256_unpackhi_pd(r0, r1);
            t2 = _mm256_unpacklo_pd(r2, r3);
            t3 = _mm256_unpackhi_pd(r2, r3);
            _mm256_storeu_pd(&dst_im[(b+0)*N + k], _mm256_permute2f128_pd(t0, t2, 0x20));
            _mm256_storeu_pd(&dst_im[(b+1)*N + k], _mm256_permute2f128_pd(t1, t3, 0x20));
            _mm256_storeu_pd(&dst_im[(b+2)*N + k], _mm256_permute2f128_pd(t0, t2, 0x31));
            _mm256_storeu_pd(&dst_im[(b+3)*N + k], _mm256_permute2f128_pd(t1, t3, 0x31));
        }
    }
}

static int run_one(const char *name, int N, size_t B,
                   strided_fn_t fn_strided, std_fn_t fn_std) {
    size_t MNK = B * (size_t)N;
    double *mat_re_s = aa(MNK);  /* strided version's matrix */
    double *mat_im_s = aa(MNK);
    double *mat_re_r = aa(MNK);  /* reference version's matrix */
    double *mat_im_r = aa(MNK);
    double *scratch_re = aa(MNK);
    double *scratch_im = aa(MNK);
    double *dummy = aa(MNK);

    fr(mat_re_s, MNK, 0xB1 + (unsigned)N);
    fr(mat_im_s, MNK, 0xB2 + (unsigned)N);
    memcpy(mat_re_r, mat_re_s, MNK * sizeof(double));
    memcpy(mat_im_r, mat_im_s, MNK * sizeof(double));
    memset(dummy, 0, MNK * sizeof(double));

    /* Strided: in-place on the matrix. */
    fn_strided(mat_re_s, mat_im_s, dummy, dummy, (size_t)N, B);

    /* Reference: gather → codelet (OOP) → scatter. */
    gather_b_x_n(mat_re_r, mat_im_r, scratch_re, scratch_im, (size_t)N, B);
    fn_std(scratch_re, scratch_im, scratch_re, scratch_im,
           dummy, dummy, B);
    scatter_n_x_b(scratch_re, scratch_im, mat_re_r, mat_im_r, (size_t)N, B);

    /* Compare */
    double err = 0;
    for (size_t i = 0; i < MNK; i++) {
        double d = fabs(mat_re_s[i] - mat_re_r[i]);
        if (d > err) err = d;
        d = fabs(mat_im_s[i] - mat_im_r[i]);
        if (d > err) err = d;
    }
    int correct = (err < 1e-10);

    /* Bench: time strided vs (gather + codelet + scatter) */
    int reps   = (B <= 32) ? 100000 : (B <= 128) ? 20000 : 5000;
    int trials = 7;
    double best_s = 1e18, best_r = 1e18;
    for (int t = 0; t < trials; t++) {
        memcpy(mat_re_s, mat_re_r, MNK * sizeof(double));  /* reset */
        memcpy(mat_im_s, mat_im_r, MNK * sizeof(double));
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            fn_strided(mat_re_s, mat_im_s, dummy, dummy, (size_t)N, B);
        }
        double dt = (now_ns() - t0) / (double)reps;
        if (dt < best_s) best_s = dt;
    }
    for (int t = 0; t < trials; t++) {
        double t0 = now_ns();
        for (int r = 0; r < reps; r++) {
            gather_b_x_n(mat_re_r, mat_im_r, scratch_re, scratch_im, (size_t)N, B);
            fn_std(scratch_re, scratch_im, scratch_re, scratch_im,
                   dummy, dummy, B);
            scatter_n_x_b(scratch_re, scratch_im, mat_re_r, mat_im_r, (size_t)N, B);
        }
        double dt = (now_ns() - t0) / (double)reps;
        if (dt < best_r) best_r = dt;
    }
    double ratio = best_s / best_r;
    const char *verdict;
    if (ratio < 0.95)      verdict = "strided WINS";
    else if (ratio < 1.05) verdict = "TIE";
    else if (ratio < 1.15) verdict = "strided SLOWER";
    else                   verdict = "REGRESSION";
    printf("%-7s N=%-3d B=%-4zu  strided=%7.1f ns  ref=%7.1f ns  ratio=%5.3f  err=%.1e  %s\n",
           name, N, B, best_s, best_r, ratio, err, correct ? verdict : "CORRECTNESS FAIL");
    free(mat_re_s); free(mat_im_s); free(mat_re_r); free(mat_im_r);
    free(scratch_re); free(scratch_im); free(dummy);
    return correct ? 0 : 1;
}

int main(void) {
    printf("================================================================\n");
    printf("  Design C strided codelet vs (gather + codelet + scatter)\n");
    printf("  in-codelet load-fused 4x4 transpose, no scratch traffic\n");
    printf("================================================================\n");
    struct { const char *name; int N; strided_fn_t fs; std_fn_t fr; } rows[] = {
        {"R16", 16, radix16_n1_fwd_avx2_gen_strided, radix16_n1_fwd_avx2_gen},
        {"R32", 32, radix32_n1_fwd_avx2_gen_strided, radix32_n1_fwd_avx2_gen},
        {"R64", 64, radix64_n1_fwd_avx2_gen_strided, radix64_n1_fwd_avx2_gen},
    };
    size_t Bs[] = {8, 16, 32, 64, 128, 256, 512, 1024, 0};
    int fails = 0;
    for (size_t i = 0; i < sizeof(rows)/sizeof(rows[0]); i++) {
        for (int bi = 0; Bs[bi] != 0; bi++) {
            fails += run_one(rows[i].name, rows[i].N, Bs[bi],
                             rows[i].fs, rows[i].fr);
        }
        printf("\n");
    }
    printf("%s\n", fails == 0 ? "All correctness PASS" : "Some FAIL");
    return fails;
}
