/**
 * @file fft_fourstep.c
 * @brief Cache-aware four-step FFT with SIMD-optimized 2D twiddles
 *
 * Four-Step Algorithm:
 * 1. Column FFTs (strided - poor cache)
 * 2. 2D twiddle multiply (SIMD-optimized with blocked layout)
 * 3. TRANSPOSE (critical for cache efficiency!)
 * 4. Row FFTs (contiguous - excellent cache)
 */

#include "fft_fourstep.h"
#include "fft_planning.h"
#include "fft_planning_types.h"
#include "fft_execute.h"
#include "fft_transpose.h"
#include "fft_twiddles_planner_api.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdio.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_alloc(alignment, size) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_free(ptr) free(ptr)
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288419716939937510
#endif

//==============================================================================
// 2D TWIDDLE BLOCKED LAYOUT STRUCTURES
//==============================================================================

typedef struct
{
    double re[8];
    double im[8];
} __attribute__((aligned(64))) twiddle_2d_block_avx512_t;

typedef struct
{
    double re[4];
    double im[4];
} __attribute__((aligned(64))) twiddle_2d_block_avx2_t;

typedef struct
{
    int N1;
    int N2;
    int simd_width;
    int num_k2_blocks;
    int num_k1_values;
    void *blocks;
} twiddle_2d_blocked_layout_t;

//==============================================================================
// STRATEGY SELECTION
//==============================================================================

bool fft_should_use_fourstep(int N, int stride)
{
    if (N < FFT_FOURSTEP_MIN_SIZE)
    {
        return false;
    }

    if (N >= 65536)
    {
        return true;
    }

    size_t working_set = (size_t)N * sizeof(fft_data);
    const size_t L1_CACHE = 48 * 1024;

    if (working_set > L1_CACHE && stride > 8)
    {
        return true;
    }

    return false;
}

//==============================================================================
// 2D TWIDDLE BLOCKED LAYOUT CREATION
//==============================================================================

#ifdef __AVX512F__
static twiddle_2d_blocked_layout_t *create_blocked_2d_twiddles_avx512(
    const twiddle_handle_t *tw_handle,
    int N1,
    int N2)
{
    const int simd_width = 8;
    const int num_k2_blocks = (N1 - 1 + simd_width - 1) / simd_width;
    const int num_k1_values = N2 - 1;

    twiddle_2d_blocked_layout_t *layout =
        (twiddle_2d_blocked_layout_t *)malloc(sizeof(twiddle_2d_blocked_layout_t));
    if (!layout)
        return NULL;

    const size_t total_blocks = (size_t)num_k2_blocks * num_k1_values;
    twiddle_2d_block_avx512_t *blocks =
        (twiddle_2d_block_avx512_t *)aligned_alloc(
            64, total_blocks * sizeof(twiddle_2d_block_avx512_t));

    if (!blocks)
    {
        free(layout);
        return NULL;
    }

    fft_twiddles_soa_view tw_view;
    if (twiddle_get_soa_view(tw_handle, &tw_view) != 0)
    {
        aligned_free(blocks);
        free(layout);
        return NULL;
    }

    // Reorganize: Read from standard handle, write to 2D-optimized blocks
    for (int k2_block = 0; k2_block < num_k2_blocks; k2_block++)
    {
        const int k2_base = k2_block * simd_width + 1;

        for (int k1 = 1; k1 < N2; k1++)
        {
            twiddle_2d_block_avx512_t *block =
                &blocks[k2_block * num_k1_values + (k1 - 1)];

            for (int lane = 0; lane < simd_width; lane++)
            {
                const int k2 = k2_base + lane;

                if (k2 < N1)
                {
                    const int tw_idx = (k2 - 1) * N2 + k1;
                    block->re[lane] = tw_view.re[tw_idx];
                    block->im[lane] = tw_view.im[tw_idx];
                }
                else
                {
                    block->re[lane] = 1.0;
                    block->im[lane] = 0.0;
                }
            }
        }
    }

    layout->N1 = N1;
    layout->N2 = N2;
    layout->simd_width = simd_width;
    layout->num_k2_blocks = num_k2_blocks;
    layout->num_k1_values = num_k1_values;
    layout->blocks = blocks;

    return layout;
}
#endif

#ifdef __AVX2__
static twiddle_2d_blocked_layout_t *create_blocked_2d_twiddles_avx2(
    const twiddle_handle_t *tw_handle,
    int N1,
    int N2)
{
    const int simd_width = 4;
    const int num_k2_blocks = (N1 - 1 + simd_width - 1) / simd_width;
    const int num_k1_values = N2 - 1;

    twiddle_2d_blocked_layout_t *layout =
        (twiddle_2d_blocked_layout_t *)malloc(sizeof(twiddle_2d_blocked_layout_t));
    if (!layout)
        return NULL;

    const size_t total_blocks = (size_t)num_k2_blocks * num_k1_values;
    twiddle_2d_block_avx2_t *blocks =
        (twiddle_2d_block_avx2_t *)aligned_alloc(
            64, total_blocks * sizeof(twiddle_2d_block_avx2_t));

    if (!blocks)
    {
        free(layout);
        return NULL;
    }

    fft_twiddles_soa_view tw_view;
    if (twiddle_get_soa_view(tw_handle, &tw_view) != 0)
    {
        aligned_free(blocks);
        free(layout);
        return NULL;
    }

    for (int k2_block = 0; k2_block < num_k2_blocks; k2_block++)
    {
        const int k2_base = k2_block * simd_width + 1;

        for (int k1 = 1; k1 < N2; k1++)
        {
            twiddle_2d_block_avx2_t *block =
                &blocks[k2_block * num_k1_values + (k1 - 1)];

            for (int lane = 0; lane < simd_width; lane++)
            {
                const int k2 = k2_base + lane;

                if (k2 < N1)
                {
                    const int tw_idx = (k2 - 1) * N2 + k1;
                    block->re[lane] = tw_view.re[tw_idx];
                    block->im[lane] = tw_view.im[tw_idx];
                }
                else
                {
                    block->re[lane] = 1.0;
                    block->im[lane] = 0.0;
                }
            }
        }
    }

    layout->N1 = N1;
    layout->N2 = N2;
    layout->simd_width = simd_width;
    layout->num_k2_blocks = num_k2_blocks;
    layout->num_k1_values = num_k1_values;
    layout->blocks = blocks;

    return layout;
}
#endif

static void destroy_blocked_2d_twiddles(twiddle_2d_blocked_layout_t *layout)
{
    if (!layout)
        return;

    if (layout->blocks)
    {
        aligned_free(layout->blocks);
    }

    free(layout);
}

//==============================================================================
// SIMD 2D TWIDDLE APPLICATION
//==============================================================================

#ifdef __AVX512F__
static int apply_twiddles_2d_blocked_avx512(
    fft_data *matrix,
    const twiddle_2d_blocked_layout_t *layout,
    int N1,
    int N2)
{
    if (!layout || !matrix)
        return -1;

    const twiddle_2d_block_avx512_t *blocks =
        (const twiddle_2d_block_avx512_t *)layout->blocks;

    for (int k1 = 1; k1 < N2; k1++)
    {
        const int row_offset = k1 * N1;
        const int num_full_blocks = (N1 - 1) / 8;

        for (int block_idx = 0; block_idx < num_full_blocks; block_idx++)
        {
            const twiddle_2d_block_avx512_t *block =
                &blocks[block_idx * layout->num_k1_values + (k1 - 1)];

            // Load twiddles with UNIT STRIDE (optimal!)
            __m512d tw_re = _mm512_load_pd(block->re);
            __m512d tw_im = _mm512_load_pd(block->im);

            const int k2_start = block_idx * 8 + 1;

            // Load data (gather)
            __m512d data_re = _mm512_setr_pd(
                matrix[row_offset + k2_start + 0].re,
                matrix[row_offset + k2_start + 1].re,
                matrix[row_offset + k2_start + 2].re,
                matrix[row_offset + k2_start + 3].re,
                matrix[row_offset + k2_start + 4].re,
                matrix[row_offset + k2_start + 5].re,
                matrix[row_offset + k2_start + 6].re,
                matrix[row_offset + k2_start + 7].re);

            __m512d data_im = _mm512_setr_pd(
                matrix[row_offset + k2_start + 0].im,
                matrix[row_offset + k2_start + 1].im,
                matrix[row_offset + k2_start + 2].im,
                matrix[row_offset + k2_start + 3].im,
                matrix[row_offset + k2_start + 4].im,
                matrix[row_offset + k2_start + 5].im,
                matrix[row_offset + k2_start + 6].im,
                matrix[row_offset + k2_start + 7].im);

            // Complex multiply
            __m512d result_re = _mm512_fmsub_pd(data_re, tw_re,
                                                _mm512_mul_pd(data_im, tw_im));
            __m512d result_im = _mm512_fmadd_pd(data_re, tw_im,
                                                _mm512_mul_pd(data_im, tw_re));

            // Store (scatter)
            double temp_re[8], temp_im[8];
            _mm512_storeu_pd(temp_re, result_re);
            _mm512_storeu_pd(temp_im, result_im);

            for (int lane = 0; lane < 8; lane++)
            {
                matrix[row_offset + k2_start + lane].re = temp_re[lane];
                matrix[row_offset + k2_start + lane].im = temp_im[lane];
            }
        }

        // Scalar cleanup
        int k2 = num_full_blocks * 8 + 1;
        for (; k2 < N1; k2++)
        {
            const int data_idx = row_offset + k2;
            const int block_idx = (k2 - 1) / 8;
            const int lane = (k2 - 1) % 8;
            const twiddle_2d_block_avx512_t *block =
                &blocks[block_idx * layout->num_k1_values + (k1 - 1)];

            const double tw_re = block->re[lane];
            const double tw_im = block->im[lane];
            const double data_re = matrix[data_idx].re;
            const double data_im = matrix[data_idx].im;

            matrix[data_idx].re = data_re * tw_re - data_im * tw_im;
            matrix[data_idx].im = data_re * tw_im + data_im * tw_re;
        }
    }

    return 0;
}
#endif

#ifdef __AVX2__
static int apply_twiddles_2d_blocked_avx2(
    fft_data *matrix,
    const twiddle_2d_blocked_layout_t *layout,
    int N1,
    int N2)
{
    if (!layout || !matrix)
        return -1;

    const twiddle_2d_block_avx2_t *blocks =
        (const twiddle_2d_block_avx2_t *)layout->blocks;

    for (int k1 = 1; k1 < N2; k1++)
    {
        const int row_offset = k1 * N1;
        const int num_full_blocks = (N1 - 1) / 4;

        for (int block_idx = 0; block_idx < num_full_blocks; block_idx++)
        {
            const twiddle_2d_block_avx2_t *block =
                &blocks[block_idx * layout->num_k1_values + (k1 - 1)];

            __m256d tw_re = _mm256_load_pd(block->re);
            __m256d tw_im = _mm256_load_pd(block->im);

            const int k2_start = block_idx * 4 + 1;
            __m256d data_re = _mm256_setr_pd(
                matrix[row_offset + k2_start + 0].re,
                matrix[row_offset + k2_start + 1].re,
                matrix[row_offset + k2_start + 2].re,
                matrix[row_offset + k2_start + 3].re);

            __m256d data_im = _mm256_setr_pd(
                matrix[row_offset + k2_start + 0].im,
                matrix[row_offset + k2_start + 1].im,
                matrix[row_offset + k2_start + 2].im,
                matrix[row_offset + k2_start + 3].im);

            __m256d result_re = _mm256_fmsub_pd(data_re, tw_re,
                                                _mm256_mul_pd(data_im, tw_im));
            __m256d result_im = _mm256_fmadd_pd(data_re, tw_im,
                                                _mm256_mul_pd(data_im, tw_re));

            double temp_re[4], temp_im[4];
            _mm256_storeu_pd(temp_re, result_re);
            _mm256_storeu_pd(temp_im, result_im);

            for (int lane = 0; lane < 4; lane++)
            {
                matrix[row_offset + k2_start + lane].re = temp_re[lane];
                matrix[row_offset + k2_start + lane].im = temp_im[lane];
            }
        }

        // Scalar cleanup
        int k2 = num_full_blocks * 4 + 1;
        for (; k2 < N1; k2++)
        {
            const int data_idx = row_offset + k2;
            const int block_idx = (k2 - 1) / 4;
            const int lane = (k2 - 1) % 4;
            const twiddle_2d_block_avx2_t *block =
                &blocks[block_idx * layout->num_k1_values + (k1 - 1)];

            const double tw_re = block->re[lane];
            const double tw_im = block->im[lane];
            const double data_re = matrix[data_idx].re;
            const double data_im = matrix[data_idx].im;

            matrix[data_idx].re = data_re * tw_re - data_im * tw_im;
            matrix[data_idx].im = data_re * tw_im + data_im * tw_re;
        }
    }

    return 0;
}
#endif

// Scalar fallback
static int apply_twiddles_2d_scalar(
    fft_data *matrix,
    const twiddle_handle_t *tw_handle,
    int N1,
    int N2)
{
    fft_twiddles_soa_view tw_view;
    if (twiddle_get_soa_view(tw_handle, &tw_view) != 0)
    {
        return -1;
    }

    for (int k1 = 1; k1 < N2; k1++)
    {
        const int row_offset = k1 * N1;

        for (int k2 = 1; k2 < N1; k2++)
        {
            const int data_idx = row_offset + k2;
            const int tw_idx = (k2 - 1) * N2 + k1;

            const double tw_re = tw_view.re[tw_idx];
            const double tw_im = tw_view.im[tw_idx];
            const double data_re = matrix[data_idx].re;
            const double data_im = matrix[data_idx].im;

            matrix[data_idx].re = data_re * tw_re - data_im * tw_im;
            matrix[data_idx].im = data_re * tw_im + data_im * tw_re;
        }
    }

    return 0;
}

//==============================================================================
// TRANSPOSE
//==============================================================================

static void transpose_rectangular_cached(
    fft_data *dst,
    const fft_data *src,
    int rows,
    int cols)
{
    if (rows == cols)
    {
        if (dst != src)
        {
            memcpy(dst, src, rows * cols * sizeof(fft_data));
        }
        fft_transpose_square((fft_complex *)dst, rows);
        return;
    }

    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

//==============================================================================
// SUB-FFT EXECUTION
//==============================================================================

static int execute_1d_fft_with_plan(
    fft_data *data,
    int size,
    int stride,
    fft_object sub_plan,
    fft_data *workspace)
{
    if (!sub_plan || size != sub_plan->n_fft)
    {
        return -1;
    }

    fft_data *temp_in = workspace;
    for (int i = 0; i < size; i++)
    {
        temp_in[i] = data[i * stride];
    }

    fft_data *temp_out = workspace + size;
    fft_data *sub_workspace = workspace + 2 * size;

    int result = fft_exec_dft(sub_plan, temp_in, temp_out, sub_workspace);
    if (result != 0)
    {
        return -1;
    }

    for (int i = 0; i < size; i++)
    {
        data[i * stride] = temp_out[i];
    }

    return 0;
}

//==============================================================================
// PLAN INITIALIZATION
//==============================================================================

int fft_fourstep_init_plan(fft_object plan, int parent_depth)
{
    if (!plan || !plan->fourstep)
    {
        fprintf(stderr, "ERROR: Invalid plan or fourstep structure\n");
        return -1;
    }

    const int N = plan->n_fft;
    const int N1 = plan->fourstep->N1;
    const int N2 = plan->fourstep->N2;
    const fft_direction_t direction = plan->direction;

    if (N1 * N2 != N)
    {
        fprintf(stderr, "ERROR: Invalid partition %d × %d != %d\n", N1, N2, N);
        return -1;
    }

    // Forward declare internal planner API
    extern fft_object fft_init_extended(int N, fft_direction_t direction,
                                        int depth, int flags);

    const int FFT_PLAN_FLAG_NONE = 0;

    // Create sub-plans with depth tracking (prevents four-step recursion)
    plan->fourstep->plan_N1 = fft_init_extended(N1, direction, parent_depth + 1, FFT_PLAN_FLAG_NONE);
    plan->fourstep->plan_N2 = fft_init_extended(N2, direction, parent_depth + 1, FFT_PLAN_FLAG_NONE);

    if (!plan->fourstep->plan_N1 || !plan->fourstep->plan_N2)
    {
        if (plan->fourstep->plan_N1)
            free_fft(plan->fourstep->plan_N1);
        if (plan->fourstep->plan_N2)
            free_fft(plan->fourstep->plan_N2);
        return -1;
    }

    // Sanity check: No four-step recursion
    if (plan->fourstep->plan_N1->strategy == FFT_EXEC_FOURSTEP ||
        plan->fourstep->plan_N2->strategy == FFT_EXEC_FOURSTEP)
    {
        fprintf(stderr, "ERROR: Four-step recursion detected!\n");
        free_fft(plan->fourstep->plan_N1);
        free_fft(plan->fourstep->plan_N2);
        return -1;
    }

    // Create 2D twiddle handle (standard format)
    plan->fourstep->twiddles_2d = twiddle_create(N, N1, direction);
    if (!plan->fourstep->twiddles_2d)
    {
        free_fft(plan->fourstep->plan_N1);
        free_fft(plan->fourstep->plan_N2);
        return -1;
    }

    if (twiddle_materialize(plan->fourstep->twiddles_2d) != 0)
    {
        twiddle_destroy(plan->fourstep->twiddles_2d);
        free_fft(plan->fourstep->plan_N1);
        free_fft(plan->fourstep->plan_N2);
        return -1;
    }

// Create 2D-specific blocked layout for SIMD
#ifdef __AVX512F__
    plan->fourstep->blocked_twiddles =
        create_blocked_2d_twiddles_avx512(plan->fourstep->twiddles_2d, N1, N2);
#elif defined(__AVX2__)
    plan->fourstep->blocked_twiddles =
        create_blocked_2d_twiddles_avx2(plan->fourstep->twiddles_2d, N1, N2);
#else
    plan->fourstep->blocked_twiddles = NULL;
#endif

    if (!plan->fourstep->blocked_twiddles && (defined(__AVX512F__) || defined(__AVX2__)))
    {
        twiddle_destroy(plan->fourstep->twiddles_2d);
        free_fft(plan->fourstep->plan_N1);
        free_fft(plan->fourstep->plan_N2);
        return -1;
    }

    return 0;
}

//==============================================================================
// PLAN CLEANUP
//==============================================================================

void fft_fourstep_free_plan(fft_object plan)
{
    if (!plan || !plan->fourstep)
        return;

    if (plan->fourstep->plan_N1)
    {
        free_fft(plan->fourstep->plan_N1);
        plan->fourstep->plan_N1 = NULL;
    }

    if (plan->fourstep->plan_N2)
    {
        free_fft(plan->fourstep->plan_N2);
        plan->fourstep->plan_N2 = NULL;
    }

    if (plan->fourstep->blocked_twiddles)
    {
        destroy_blocked_2d_twiddles(plan->fourstep->blocked_twiddles);
        plan->fourstep->blocked_twiddles = NULL;
    }

    if (plan->fourstep->twiddles_2d)
    {
        twiddle_destroy(plan->fourstep->twiddles_2d);
        plan->fourstep->twiddles_2d = NULL;
    }

    free(plan->fourstep);
    plan->fourstep = NULL;
}

//==============================================================================
// WORKSPACE SIZE
//==============================================================================

size_t fft_fourstep_workspace_size(const fft_object plan)
{
    if (!plan || plan->strategy != FFT_EXEC_FOURSTEP || !plan->fourstep)
    {
        return 0;
    }

    const int N1 = plan->fourstep->N1;
    const int N2 = plan->fourstep->N2;
    const int N = N1 * N2;

    int max_dim = (N1 > N2) ? N1 : N2;
    return (size_t)(N + 3 * max_dim);
}

//==============================================================================
// MAIN EXECUTION
//==============================================================================

int fft_exec_fourstep(
    fft_object plan,
    const fft_data *input,
    fft_data *output,
    fft_data *workspace)
{
    if (!plan || !input || !output || !workspace)
    {
        return -1;
    }

    if (plan->strategy != FFT_EXEC_FOURSTEP)
    {
        return -1;
    }

    const int N = plan->n_fft;
    const int N1 = plan->fourstep->N1;
    const int N2 = plan->fourstep->N2;
    fft_object plan_N1 = plan->fourstep->plan_N1;
    fft_object plan_N2 = plan->fourstep->plan_N2;

    if (!plan_N1 || !plan_N2)
    {
        return -1;
    }

    fft_data *matrix = output;
    fft_data *transpose_buf = workspace;
    fft_data *sub_workspace = workspace + N;

    memcpy(matrix, input, N * sizeof(fft_data));

    // STEP 1: Column FFTs (N1 columns of size N2, stride N1)
    for (int col = 0; col < N1; col++)
    {
        if (execute_1d_fft_with_plan(matrix + col, N2, N1, plan_N2, sub_workspace) != 0)
        {
            return -1;
        }
    }

// STEP 2: Apply 2D twiddles using SIMD-optimized blocked layout
#ifdef __AVX512F__
    if (apply_twiddles_2d_blocked_avx512(matrix, plan->fourstep->blocked_twiddles, N1, N2) != 0)
    {
        return -1;
    }
#elif defined(__AVX2__)
    if (apply_twiddles_2d_blocked_avx2(matrix, plan->fourstep->blocked_twiddles, N1, N2) != 0)
    {
        return -1;
    }
#else
    if (apply_twiddles_2d_scalar(matrix, plan->fourstep->twiddles_2d, N1, N2) != 0)
    {
        return -1;
    }
#endif

    // STEP 3: Transpose (N2×N1 → N1×N2) - KEY for cache efficiency!
    transpose_rectangular_cached(transpose_buf, matrix, N2, N1);
    memcpy(matrix, transpose_buf, N * sizeof(fft_data));

    // STEP 4: Row FFTs (N2 rows of size N1, contiguous after transpose!)
    for (int row = 0; row < N2; row++)
    {
        if (execute_1d_fft_with_plan(matrix + row * N1, N1, 1, plan_N1, sub_workspace) != 0)
        {
            return -1;
        }
    }

    return 0;
}