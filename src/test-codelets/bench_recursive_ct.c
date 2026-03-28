/**
 * bench_recursive_ct.c — Permutation-free recursive CT executor
 *
 * Uses FFTW-style n1 (separate is/os) + t1 (in-place twiddle) codelets.
 * The stride change in n1 eliminates permutation entirely.
 *
 * DIT recursive: for N = R * M
 *   Step 1: n1 child — R DFTs of size M
 *           reads at is = R * parent_is, writes at os = parent_os
 *           vl = M (batch count)
 *   Step 2: t1 twiddle — in-place on output
 *           ios = M * parent_os, me = M
 *
 * Test sizes: N = 256 (16x16), 2048 (16x8x16), 32768 (16x16x8x16)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fftw3.h>
#include "bench_compat.h"

/* R=4, R=8 CT codelets are in the main avx2 headers */
#include "fft_radix4_avx2.h"
#include "fft_radix8_avx2.h"

/* R=16 CT codelets in separate headers */
#include "fft_radix16_avx2_ct_n1.h"
#include "fft_radix16_avx2_ct_t1_dit.h"

/* ================================================================
 * Twiddle table for t1 codelets
 *
 * Layout: W_re[(n-1)*me + m] for n=1..R-1, m=0..me-1
 * Values: W_N^(n*m) = exp(-2*pi*i*n*m / N)
 * where N = R * me (the total size at this CT level)
 * ================================================================ */

static void init_t1_tw(double *W_re, double *W_im, size_t R, size_t me) {
    size_t N = R * me;
    for (size_t n = 1; n < R; n++)
        for (size_t m = 0; m < me; m++) {
            double a = -2.0 * M_PI * (double)(n * m) / (double)N;
            W_re[(n-1)*me + m] = cos(a);
            W_im[(n-1)*me + m] = sin(a);
        }
}

/* ================================================================
 * Recursive CT plan tree
 * ================================================================ */

typedef void (*n1_fn)(const double*, const double*, double*, double*,
                      size_t, size_t, size_t);  /* in, out, is, os, vl */
typedef void (*t1_fn)(double*, double*, const double*, const double*,
                      size_t, size_t);  /* rio, W, ios, me */

typedef enum { CT_LEAF, CT_DIT } node_type;

typedef struct ct_node {
    node_type type;
    size_t R;           /* radix at this level */
    size_t M;           /* child size = N/R */
    n1_fn n1;           /* out-of-place child DFT (stride-changing) */
    t1_fn t1;           /* in-place twiddle + butterfly */
    double *W_re, *W_im; /* twiddle table for t1: (R-1)*M entries */
    struct ct_node *child; /* child plan (for CT_DIT) */
} ct_node;

/* ================================================================
 * Recursive executor — NO PERMUTATION
 *
 * For DIT with N = R * M:
 *   1. Execute R child DFTs of size M
 *      n1(input, output, is=R*parent_is, os=parent_os, vl=M)
 *      The n1 codelet reads at stride R*parent_is (decimated)
 *      and writes at stride parent_os (contiguous within this level)
 *
 *   2. Apply twiddle + radix-R butterfly in-place
 *      t1(output, W, ios=M*parent_os, me=M)
 *      ios = distance between butterfly legs
 *      me = number of sub-transforms
 *
 * The input pointer advances by parent_is for each of the R sub-DFTs.
 * The output pointer advances by M*parent_os for each sub-DFT's output block.
 *
 * For a leaf node (M=1 or small enough for a single n1 call):
 *   Just call n1 directly with the given strides.
 * ================================================================ */

static void ct_execute(const ct_node *node,
                       const double *in_re, const double *in_im,
                       double *out_re, double *out_im,
                       size_t is, size_t os, size_t vl)
{
    size_t R = node->R;

    if (node->type == CT_LEAF) {
        /* Leaf: n1 processes vl batch elements at the given strides.
         * Reads in_re[n*is + k] for n=0..R-1, k=0..vl-1
         * Writes out_re[m*os + k] for m=0..R-1, k=0..vl-1 */
        node->n1(in_re, in_im, out_re, out_im, is, os, vl);
        return;
    }

    /* CT_DIT: N = R * M at this level.
     * We have vl batch elements to process.
     * M = node->M = size of each child sub-DFT.
     *
     * Step 1: For each of R sub-sequences, execute child DFT of size M.
     *   Sub-sequence r starts at in + r*is.
     *   Output block r goes to out + r*M*os.
     *   Child sees: is_child = R*is, os_child = os, vl_child = vl
     *   (vl propagates unchanged — all levels process the same batch count)
     *
     * BUT WAIT: the n1 codelet signature is n1(in, out, is, os, vl).
     * At the leaf, vl = number of independent k-values to process.
     * At a CT level, the R children each process M sub-transform points.
     * The "vl" at the leaf should be M (the innermost sub-DFT batch count),
     * NOT the top-level vl.
     *
     * Actually: vl at the top level is always 1 (one FFT). The "batch"
     * dimension of the n1 codelet is the M parameter from the parent CT.
     * So: leaf vl = parent's M, and we recurse without explicit vl. */

    size_t M = node->M;

    /* Step 1: R child DFTs */
    for (size_t r = 0; r < R; r++) {
        ct_execute(node->child,
                   in_re + r * is, in_im + r * is,
                   out_re + r * M * os, out_im + r * M * os,
                   R * is, os, M);
    }

    /* Step 2: in-place twiddle + radix-R butterfly
     * ios = M*os (distance between butterfly legs)
     * me = M (number of sub-transforms) */
    node->t1(out_re, out_im, node->W_re, node->W_im, M * os, M);
}

/* ================================================================
 * Plan builder
 * ================================================================ */

static ct_node *make_leaf(size_t R, n1_fn n1) {
    ct_node *n = (ct_node*)calloc(1, sizeof(ct_node));
    n->type = CT_LEAF;
    n->R = R;
    n->M = 1;  /* will be set by parent based on vl */
    n->n1 = n1;
    return n;
}

static ct_node *make_ct(size_t R, n1_fn n1, t1_fn t1, size_t M, ct_node *child) {
    ct_node *n = (ct_node*)calloc(1, sizeof(ct_node));
    n->type = CT_DIT;
    n->R = R;
    n->M = M;
    n->n1 = n1;
    n->t1 = t1;
    n->child = child;
    n->W_re = (double*)aligned_alloc(32, (R-1) * M * sizeof(double));
    n->W_im = (double*)aligned_alloc(32, (R-1) * M * sizeof(double));
    init_t1_tw(n->W_re, n->W_im, R, M);
    return n;
}

static void free_plan(ct_node *n) {
    if (!n) return;
    if (n->child) free_plan(n->child);
    if (n->W_re) aligned_free(n->W_re);
    if (n->W_im) aligned_free(n->W_im);
    free(n);
}

/* ================================================================
 * FFTW reference
 * ================================================================ */

static double bench_fftw(size_t N, int reps) {
    double *ri = fftw_malloc(N*8), *ii = fftw_malloc(N*8);
    double *ro = fftw_malloc(N*8), *io = fftw_malloc(N*8);
    for (size_t i = 0; i < N; i++) { ri[i] = (double)rand()/RAND_MAX; ii[i] = (double)rand()/RAND_MAX; }
    fftw_iodim d = {.n = (int)N, .is = 1, .os = 1};
    fftw_iodim h = {.n = 1, .is = (int)N, .os = (int)N};
    fftw_plan p = fftw_plan_guru_split_dft(1, &d, 1, &h, ri, ii, ro, io, FFTW_MEASURE);
    if (!p) { fftw_free(ri); fftw_free(ii); fftw_free(ro); fftw_free(io); return -1; }
    for (int i = 0; i < 20; i++) fftw_execute(p);
    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++) fftw_execute_split_dft(p, ri, ii, ro, io);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }
    fftw_destroy_plan(p); fftw_free(ri); fftw_free(ii); fftw_free(ro); fftw_free(io);
    return best;
}

/* ================================================================
 * Bench + correctness for one plan
 * ================================================================ */

static void test_plan(const char *label, size_t N, ct_node *plan, int reps) {
    double *in_re  = (double*)aligned_alloc(32, N*8);
    double *in_im  = (double*)aligned_alloc(32, N*8);
    double *out_re = (double*)aligned_alloc(32, N*8);
    double *out_im = (double*)aligned_alloc(32, N*8);

    srand(42);
    for (size_t i = 0; i < N; i++) {
        in_re[i] = (double)rand()/RAND_MAX - 0.5;
        in_im[i] = (double)rand()/RAND_MAX - 0.5;
    }

    /* Correctness: compare against FFTW */
    double *fre = fftw_malloc(N*8), *fim = fftw_malloc(N*8);
    double *fro = fftw_malloc(N*8), *fio = fftw_malloc(N*8);
    memcpy(fre, in_re, N*8); memcpy(fim, in_im, N*8);
    fftw_iodim d = {.n = (int)N, .is = 1, .os = 1};
    fftw_iodim h = {.n = 1, .is = (int)N, .os = (int)N};
    fftw_plan fp = fftw_plan_guru_split_dft(1, &d, 1, &h, fre, fim, fro, fio, FFTW_ESTIMATE);
    fftw_execute(fp);

    ct_execute(plan, in_re, in_im, out_re, out_im, 1, 1, 0);

    double max_err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fabs(out_re[i] - fro[i]) + fabs(out_im[i] - fio[i]);
        if (e > max_err) max_err = e;
    }

    printf("  %-30s N=%-6zu err=%.2e %s",
           label, N, max_err, max_err < 1e-10 ? "OK" : "FAIL");

    if (max_err > 1e-10) {
        printf("\n");
        fftw_destroy_plan(fp);
        fftw_free(fre); fftw_free(fim); fftw_free(fro); fftw_free(fio);
        aligned_free(in_re); aligned_free(in_im); aligned_free(out_re); aligned_free(out_im);
        return;
    }

    /* Benchmark */
    for (int i = 0; i < 20; i++)
        ct_execute(plan, in_re, in_im, out_re, out_im, 1, 1, 0);
    double best = 1e18;
    for (int t = 0; t < 7; t++) {
        double t0 = now_ns();
        for (int i = 0; i < reps; i++)
            ct_execute(plan, in_re, in_im, out_re, out_im, 1, 1, 0);
        double ns = (now_ns() - t0) / reps;
        if (ns < best) best = ns;
    }

    double fftw_ns = bench_fftw(N, reps);
    printf("  %8.0f ns  (%.2fx vs FFTW %.0f ns)\n", best, fftw_ns/best, fftw_ns);

    fftw_destroy_plan(fp);
    fftw_free(fre); fftw_free(fim); fftw_free(fro); fftw_free(fio);
    aligned_free(in_re); aligned_free(in_im); aligned_free(out_re); aligned_free(out_im);
}

/* ================================================================
 * Main
 * ================================================================ */

int main(void) {
    printf("================================================================\n");
    printf("  Permutation-free recursive CT executor\n");
    printf("  n1 (is/os) + t1 (in-place) — zero explicit permutation\n");
    printf("================================================================\n\n");

    /* N=64 = 8 x 8 */
    {
        ct_node *leaf = make_leaf(8, (n1_fn)radix8_n1_fwd_avx2);
        ct_node *plan = make_ct(8, (n1_fn)radix8_n1_fwd_avx2,
                                (t1_fn)radix8_t1_dit_fwd_avx2, 8, leaf);
        test_plan("8x8", 64, plan, 200000);
        free_plan(plan);
    }

    /* N=128 = 16 x 8 */
    {
        ct_node *leaf = make_leaf(8, (n1_fn)radix8_n1_fwd_avx2);
        ct_node *plan = make_ct(16, (n1_fn)radix16_n1_fwd_avx2,
                                (t1_fn)radix16_t1_dit_fwd_avx2, 8, leaf);
        test_plan("16x8", 128, plan, 100000);
        free_plan(plan);
    }

    /* N=256 = 16 x 16 */
    {
        ct_node *leaf = make_leaf(16, (n1_fn)radix16_n1_fwd_avx2);
        ct_node *plan = make_ct(16, (n1_fn)radix16_n1_fwd_avx2,
                                (t1_fn)radix16_t1_dit_fwd_avx2, 16, leaf);
        test_plan("16x16", 256, plan, 50000);
        free_plan(plan);
    }

    /* N=2048 = 16 x 8 x 16 */
    {
        ct_node *leaf = make_leaf(16, (n1_fn)radix16_n1_fwd_avx2);
        ct_node *mid = make_ct(8, (n1_fn)radix8_n1_fwd_avx2,
                               (t1_fn)radix8_t1_dit_fwd_avx2, 16, leaf);
        ct_node *plan = make_ct(16, (n1_fn)radix16_n1_fwd_avx2,
                                (t1_fn)radix16_t1_dit_fwd_avx2, 128, mid);
        test_plan("16x8x16", 2048, plan, 5000);
        free_plan(plan);
    }

    /* N=4096 = 16 x 16 x 16 */
    {
        ct_node *leaf = make_leaf(16, (n1_fn)radix16_n1_fwd_avx2);
        ct_node *mid = make_ct(16, (n1_fn)radix16_n1_fwd_avx2,
                               (t1_fn)radix16_t1_dit_fwd_avx2, 16, leaf);
        ct_node *plan = make_ct(16, (n1_fn)radix16_n1_fwd_avx2,
                                (t1_fn)radix16_t1_dit_fwd_avx2, 256, mid);
        test_plan("16x16x16", 4096, plan, 2000);
        free_plan(plan);
    }

    /* N=32768 = 16 x 16 x 8 x 16 */
    {
        ct_node *leaf = make_leaf(16, (n1_fn)radix16_n1_fwd_avx2);
        ct_node *mid1 = make_ct(8, (n1_fn)radix8_n1_fwd_avx2,
                                (t1_fn)radix8_t1_dit_fwd_avx2, 16, leaf);
        ct_node *mid2 = make_ct(16, (n1_fn)radix16_n1_fwd_avx2,
                                (t1_fn)radix16_t1_dit_fwd_avx2, 128, mid1);
        ct_node *plan = make_ct(16, (n1_fn)radix16_n1_fwd_avx2,
                                (t1_fn)radix16_t1_dit_fwd_avx2, 2048, mid2);
        test_plan("16x16x8x16", 32768, plan, 200);
        free_plan(plan);
    }

    printf("\n");
    return 0;
}
