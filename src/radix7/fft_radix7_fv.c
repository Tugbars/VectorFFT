/**
 * @file  fft_radix7_fv.c
 * @brief Radix-7 forward stage visitor
 *
 * The centralized planner calls fft_radix7_visit_forward() once per DIT stage.
 * This function iterates over butterfly groups and dispatches to the
 * ISA-specific kernel via the vtable.
 *
 * The planner is responsible for:
 *   - Digit-reversal permutation (before stage 0)
 *   - Stage ordering (s = 0, 1, ..., L-1 for DIT forward)
 *   - Twiddle precomputation (BLOCKED3: tw1, tw2, tw3)
 *   - Passing K = 7^s and num_groups = N / (7·K) per stage
 *
 * For stage 0 (K=1): pass tw pointers as NULL → uses N1 butterfly.
 * For stage s>0:      pass BLOCKED3 twiddle arrays → uses twiddled butterfly.
 */

/*
 * GCC -O3 -ftree-loop-vectorize can miscompile these stage loops when
 * inlined butterfly calls use restrict-qualified parameters in-place.
 * The R7_PRAGMA_NO_AUTOVEC macro handles this portably (no-op on ICX/Clang).
 */
#include "fft_r7_platform.h"
R7_PRAGMA_NO_AUTOVEC

#include "fft_radix7.h"

/*
 * Include butterfly implementations here.
 * This is the single compilation unit that produces all butterfly symbols.
 * Other files (bv.c, planner) call through vtable function pointers.
 *
 * Override R7_BUTTERFLY_API: headers default to static inline for
 * direct-include use (tests, benchmarks). Here we need extern linkage
 * so the linker can resolve vtable function pointers.
 */
#define R7_BUTTERFLY_API /* extern linkage for library build */

#include "scalar/fft_radix7_scalar.h"

#ifdef __AVX2__
#include "avx2/fft_radix7_avx2.h"
#endif

#ifdef __AVX512F__
#include "avx512/fft_radix7_avx512.h"
#endif

void fft_radix7_visit_forward(
    const fft_r7_vtable_t *vt,
    double *re, double *im,
    int K, int num_groups,
    const double *tw1_re, const double *tw1_im,
    const double *tw2_re, const double *tw2_im,
    const double *tw3_re, const double *tw3_im)
{
    const int full_size = 7 * K;

    if (tw1_re == NULL)
    {
        /* Stage 0: K=1, N1 butterfly, 7 adjacent elements per group */
        for (int g = 0; g < num_groups; g++)
        {
            int b = g * 7;
            vt->fwd_n1(
                &re[b + 0], &im[b + 0],
                &re[b + 1], &im[b + 1],
                &re[b + 2], &im[b + 2],
                &re[b + 3], &im[b + 3],
                &re[b + 4], &im[b + 4],
                &re[b + 5], &im[b + 5],
                &re[b + 6], &im[b + 6],
                &re[b + 0], &im[b + 0],
                &re[b + 1], &im[b + 1],
                &re[b + 2], &im[b + 2],
                &re[b + 3], &im[b + 3],
                &re[b + 4], &im[b + 4],
                &re[b + 5], &im[b + 5],
                &re[b + 6], &im[b + 6],
                1);
        }
    }
    else
    {
        /* Stage s > 0: twiddled butterfly, K contiguous per leg */
        for (int g = 0; g < num_groups; g++)
        {
            int b = g * full_size;
            vt->fwd_tw(
                &re[b + 0 * K], &im[b + 0 * K],
                &re[b + 1 * K], &im[b + 1 * K],
                &re[b + 2 * K], &im[b + 2 * K],
                &re[b + 3 * K], &im[b + 3 * K],
                &re[b + 4 * K], &im[b + 4 * K],
                &re[b + 5 * K], &im[b + 5 * K],
                &re[b + 6 * K], &im[b + 6 * K],
                &re[b + 0 * K], &im[b + 0 * K],
                &re[b + 1 * K], &im[b + 1 * K],
                &re[b + 2 * K], &im[b + 2 * K],
                &re[b + 3 * K], &im[b + 3 * K],
                &re[b + 4 * K], &im[b + 4 * K],
                &re[b + 5 * K], &im[b + 5 * K],
                &re[b + 6 * K], &im[b + 6 * K],
                tw1_re, tw1_im,
                tw2_re, tw2_im,
                tw3_re, tw3_im,
                K);
        }
    }
}