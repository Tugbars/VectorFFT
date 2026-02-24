/**
 * @file  fft_radix7_bv.c
 * @brief Radix-7 backward stage visitor
 *
 * The centralized planner calls fft_radix7_visit_backward() once per stage.
 * For the inverse DIT FFT, the planner must:
 *   - Call stages in REVERSE order: s = L-1, L-2, ..., 0
 *   - Apply digit-reversal permutation AFTER all stages
 *   - Divide output by N to normalize
 *
 * The backward butterfly uses conjugate twiddles internally (applied after
 * the inverse Rader convolution), so the planner passes the SAME twiddle
 * tables as forward — no separate conjugate tables needed.
 */

#include "fft_r7_platform.h"
R7_PRAGMA_NO_AUTOVEC

#include "fft_radix7.h"

void fft_radix7_visit_backward(
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
        /* Stage 0: K=1, N1 backward butterfly */
        for (int g = 0; g < num_groups; g++)
        {
            int b = g * 7;
            vt->bwd_n1(
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
        /* Stage s > 0: twiddled backward butterfly */
        for (int g = 0; g < num_groups; g++)
        {
            int b = g * full_size;
            vt->bwd_tw(
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