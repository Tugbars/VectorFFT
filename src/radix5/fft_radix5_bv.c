/**
 * @file  fft_radix5_bv.c
 * @brief Radix-5 backward stage visitor
 *
 * The centralized planner calls fft_radix5_visit_backward() once per stage.
 * For the inverse DIT FFT, the planner must:
 *   - Call stages in REVERSE order: s = L-1, L-2, ..., 0
 *   - Apply digit-reversal permutation AFTER all stages
 *   - Divide output by N to normalize
 *
 * The backward butterfly uses conjugate twiddles internally (applied after
 * the inverse WFTA), so the planner passes the SAME twiddle tables as
 * forward — no separate conjugate tables needed.
 */

#include "fft_r5_platform.h"
R5_PRAGMA_NO_AUTOVEC

#include "fft_radix5.h"

void fft_radix5_visit_backward(
    const fft_r5_vtable_t *vt,
    double *re, double *im,
    int K, int num_groups,
    const double *tw1_re, const double *tw1_im,
    const double *tw2_re, const double *tw2_im)
{
    const int full_size = 5 * K;

    if (tw1_re == NULL) {
        /* Stage 0: K=1, N1 backward butterfly */
        for (int g = 0; g < num_groups; g++) {
            int b = g * 5;
            vt->bwd_n1(
                &re[b+0], &im[b+0],
                &re[b+1], &im[b+1],
                &re[b+2], &im[b+2],
                &re[b+3], &im[b+3],
                &re[b+4], &im[b+4],
                &re[b+0], &im[b+0],
                &re[b+1], &im[b+1],
                &re[b+2], &im[b+2],
                &re[b+3], &im[b+3],
                &re[b+4], &im[b+4],
                1);
        }
    } else {
        /* Stage s > 0: twiddled backward butterfly */
        for (int g = 0; g < num_groups; g++) {
            int b = g * full_size;
            vt->bwd_tw(
                &re[b + 0*K], &im[b + 0*K],
                &re[b + 1*K], &im[b + 1*K],
                &re[b + 2*K], &im[b + 2*K],
                &re[b + 3*K], &im[b + 3*K],
                &re[b + 4*K], &im[b + 4*K],
                &re[b + 0*K], &im[b + 0*K],
                &re[b + 1*K], &im[b + 1*K],
                &re[b + 2*K], &im[b + 2*K],
                &re[b + 3*K], &im[b + 3*K],
                &re[b + 4*K], &im[b + 4*K],
                tw1_re, tw1_im,
                tw2_re, tw2_im,
                K);
        }
    }
}