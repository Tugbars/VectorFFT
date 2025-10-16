#include "fft_radix7.h"   // ✅ Gets highSpeedFFT.h → fft_types.h
#include "simd_math.h"    // ✅ Gets complex math operations

// --- Radix-7 constants ---
const double C1 = 0.6234898018587336;   // cos(2π/7)
const double C2 = -0.22252093395631440; // cos(4π/7)
const double C3 = -0.90096886790241915; // cos(6π/7)
const double S1 = 0.78183148246802981;  // sin(2π/7)
const double S2 = 0.97492791218182360;  // sin(4π/7)
const double S3 = 0.43388373911755806;  // sin(6π/7)

void fft_radix7_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign)
{
       //======================================================================
        // RADIX-7 BUTTERFLY (Rader DIT), FFTW-style optimized, pure AoS
        //  - AVX2: 8x unrolled main loop, 2x cleanup
        //  - scalar tail for 0..1 leftover
        //  - per-stage twiddles: stage_tw[6*k + 0..5] for x1..x6 (DIT)
        //======================================================================
        const int seventh = sub_len;
        int k = 0;

        // Rader permutations for N=7 (generator g=3, inverse 5):
        //   perm_in  = [1,3,2,6,4,5]  (reorder inputs x1..x6)
        //   out_perm = [1,5,4,6,2,3]  (where each conv[q] lands)
        // We encode by wiring directly below (no explicit arrays needed)

        // Build convolution twiddles tw[q] = exp( sgn * j*2π*out_perm[q] / 7 )
        // sgn: forward(+1) uses minus angle convention in init; here we follow Rader directly:
        const double base_angle = (transform_sign == 1 ? -2.0 : +2.0) * M_PI / 7.0;

#ifdef __AVX2__
        //----------------------------------------------------------------------
        // AVX2 PATH
        //----------------------------------------------------------------------
        __m256d tw_brd[6]; // [wr,wi, wr,wi] broadcast for AoS pair multiply
        {
            // out_perm = [1,5,4,6,2,3]
            const int op[6] = {1, 5, 4, 6, 2, 3};
            for (int q = 0; q < 6; ++q)
            {
                double a = op[q] * base_angle;
                double wr, wi;
#ifdef __GNUC__
                sincos(a, &wi, &wr);
#else
                wr = cos(a);
                wi = sin(a);
#endif
                // lanes (hi..lo): [im0, re0, im1, re1] expected by cmul_avx2_aos as [br,bi, br,bi]
                tw_brd[q] = _mm256_set_pd(wi, wr, wi, wr);
            }
        }

        // -----------------------------
        // 8x unrolled main loop
        // -----------------------------
        for (; k + 7 < seventh; k += 8)
        {
            // Prefetch a bit ahead
            if (k + 16 < seventh)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 16].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + seventh].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 2 * seventh].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 3 * seventh].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 4 * seventh].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 5 * seventh].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&sub_outputs[k + 16 + 6 * seventh].re, _MM_HINT_T0);
                if (seventh > 1)
                    _mm_prefetch((const char *)&stage_tw[6 * (k + 16)].re, _MM_HINT_T0);
            }

            // Load x0..x6 for 8 butterflies: 4 AVX loads per "lane group"
            __m256d x0_0 = load2_aos(&sub_outputs[k + 0 * seventh + 0], &sub_outputs[k + 0 * seventh + 1]);
            __m256d x0_1 = load2_aos(&sub_outputs[k + 0 * seventh + 2], &sub_outputs[k + 0 * seventh + 3]);
            __m256d x0_2 = load2_aos(&sub_outputs[k + 0 * seventh + 4], &sub_outputs[k + 0 * seventh + 5]);
            __m256d x0_3 = load2_aos(&sub_outputs[k + 0 * seventh + 6], &sub_outputs[k + 0 * seventh + 7]);

            __m256d x1_0 = load2_aos(&sub_outputs[k + 1 * seventh + 0], &sub_outputs[k + 1 * seventh + 1]);
            __m256d x1_1 = load2_aos(&sub_outputs[k + 1 * seventh + 2], &sub_outputs[k + 1 * seventh + 3]);
            __m256d x1_2 = load2_aos(&sub_outputs[k + 1 * seventh + 4], &sub_outputs[k + 1 * seventh + 5]);
            __m256d x1_3 = load2_aos(&sub_outputs[k + 1 * seventh + 6], &sub_outputs[k + 1 * seventh + 7]);

            __m256d x2_0 = load2_aos(&sub_outputs[k + 2 * seventh + 0], &sub_outputs[k + 2 * seventh + 1]);
            __m256d x2_1 = load2_aos(&sub_outputs[k + 2 * seventh + 2], &sub_outputs[k + 2 * seventh + 3]);
            __m256d x2_2 = load2_aos(&sub_outputs[k + 2 * seventh + 4], &sub_outputs[k + 2 * seventh + 5]);
            __m256d x2_3 = load2_aos(&sub_outputs[k + 2 * seventh + 6], &sub_outputs[k + 2 * seventh + 7]);

            __m256d x3_0 = load2_aos(&sub_outputs[k + 3 * seventh + 0], &sub_outputs[k + 3 * seventh + 1]);
            __m256d x3_1 = load2_aos(&sub_outputs[k + 3 * seventh + 2], &sub_outputs[k + 3 * seventh + 3]);
            __m256d x3_2 = load2_aos(&sub_outputs[k + 3 * seventh + 4], &sub_outputs[k + 3 * seventh + 5]);
            __m256d x3_3 = load2_aos(&sub_outputs[k + 3 * seventh + 6], &sub_outputs[k + 3 * seventh + 7]);

            __m256d x4_0 = load2_aos(&sub_outputs[k + 4 * seventh + 0], &sub_outputs[k + 4 * seventh + 1]);
            __m256d x4_1 = load2_aos(&sub_outputs[k + 4 * seventh + 2], &sub_outputs[k + 4 * seventh + 3]);
            __m256d x4_2 = load2_aos(&sub_outputs[k + 4 * seventh + 4], &sub_outputs[k + 4 * seventh + 5]);
            __m256d x4_3 = load2_aos(&sub_outputs[k + 4 * seventh + 6], &sub_outputs[k + 4 * seventh + 7]);

            __m256d x5_0 = load2_aos(&sub_outputs[k + 5 * seventh + 0], &sub_outputs[k + 5 * seventh + 1]);
            __m256d x5_1 = load2_aos(&sub_outputs[k + 5 * seventh + 2], &sub_outputs[k + 5 * seventh + 3]);
            __m256d x5_2 = load2_aos(&sub_outputs[k + 5 * seventh + 4], &sub_outputs[k + 5 * seventh + 5]);
            __m256d x5_3 = load2_aos(&sub_outputs[k + 5 * seventh + 6], &sub_outputs[k + 5 * seventh + 7]);

            __m256d x6_0 = load2_aos(&sub_outputs[k + 6 * seventh + 0], &sub_outputs[k + 6 * seventh + 1]);
            __m256d x6_1 = load2_aos(&sub_outputs[k + 6 * seventh + 2], &sub_outputs[k + 6 * seventh + 3]);
            __m256d x6_2 = load2_aos(&sub_outputs[k + 6 * seventh + 4], &sub_outputs[k + 6 * seventh + 5]);
            __m256d x6_3 = load2_aos(&sub_outputs[k + 6 * seventh + 6], &sub_outputs[k + 6 * seventh + 7]);

            // Apply per-stage DIT twiddles (if multi-stage)
            if (seventh > 1)
            {
                __m256d w1_0 = load2_aos(&stage_tw[6 * (k + 0) + 0], &stage_tw[6 * (k + 1) + 0]);
                __m256d w1_1 = load2_aos(&stage_tw[6 * (k + 2) + 0], &stage_tw[6 * (k + 3) + 0]);
                __m256d w1_2 = load2_aos(&stage_tw[6 * (k + 4) + 0], &stage_tw[6 * (k + 5) + 0]);
                __m256d w1_3 = load2_aos(&stage_tw[6 * (k + 6) + 0], &stage_tw[6 * (k + 7) + 0]);

                __m256d w2_0 = load2_aos(&stage_tw[6 * (k + 0) + 1], &stage_tw[6 * (k + 1) + 1]);
                __m256d w2_1 = load2_aos(&stage_tw[6 * (k + 2) + 1], &stage_tw[6 * (k + 3) + 1]);
                __m256d w2_2 = load2_aos(&stage_tw[6 * (k + 4) + 1], &stage_tw[6 * (k + 5) + 1]);
                __m256d w2_3 = load2_aos(&stage_tw[6 * (k + 6) + 1], &stage_tw[6 * (k + 7) + 1]);

                __m256d w3_0 = load2_aos(&stage_tw[6 * (k + 0) + 2], &stage_tw[6 * (k + 1) + 2]);
                __m256d w3_1 = load2_aos(&stage_tw[6 * (k + 2) + 2], &stage_tw[6 * (k + 3) + 2]);
                __m256d w3_2 = load2_aos(&stage_tw[6 * (k + 4) + 2], &stage_tw[6 * (k + 5) + 2]);
                __m256d w3_3 = load2_aos(&stage_tw[6 * (k + 6) + 2], &stage_tw[6 * (k + 7) + 2]);

                __m256d w4_0 = load2_aos(&stage_tw[6 * (k + 0) + 3], &stage_tw[6 * (k + 1) + 3]);
                __m256d w4_1 = load2_aos(&stage_tw[6 * (k + 2) + 3], &stage_tw[6 * (k + 3) + 3]);
                __m256d w4_2 = load2_aos(&stage_tw[6 * (k + 4) + 3], &stage_tw[6 * (k + 5) + 3]);
                __m256d w4_3 = load2_aos(&stage_tw[6 * (k + 6) + 3], &stage_tw[6 * (k + 7) + 3]);

                __m256d w5_0 = load2_aos(&stage_tw[6 * (k + 0) + 4], &stage_tw[6 * (k + 1) + 4]);
                __m256d w5_1 = load2_aos(&stage_tw[6 * (k + 2) + 4], &stage_tw[6 * (k + 3) + 4]);
                __m256d w5_2 = load2_aos(&stage_tw[6 * (k + 4) + 4], &stage_tw[6 * (k + 5) + 4]);
                __m256d w5_3 = load2_aos(&stage_tw[6 * (k + 6) + 4], &stage_tw[6 * (k + 7) + 4]);

                __m256d w6_0 = load2_aos(&stage_tw[6 * (k + 0) + 5], &stage_tw[6 * (k + 1) + 5]);
                __m256d w6_1 = load2_aos(&stage_tw[6 * (k + 2) + 5], &stage_tw[6 * (k + 3) + 5]);
                __m256d w6_2 = load2_aos(&stage_tw[6 * (k + 4) + 5], &stage_tw[6 * (k + 5) + 5]);
                __m256d w6_3 = load2_aos(&stage_tw[6 * (k + 6) + 5], &stage_tw[6 * (k + 7) + 5]);

                x1_0 = cmul_avx2_aos(x1_0, w1_0);
                x1_1 = cmul_avx2_aos(x1_1, w1_1);
                x1_2 = cmul_avx2_aos(x1_2, w1_2);
                x1_3 = cmul_avx2_aos(x1_3, w1_3);

                x2_0 = cmul_avx2_aos(x2_0, w2_0);
                x2_1 = cmul_avx2_aos(x2_1, w2_1);
                x2_2 = cmul_avx2_aos(x2_2, w2_2);
                x2_3 = cmul_avx2_aos(x2_3, w2_3);

                x3_0 = cmul_avx2_aos(x3_0, w3_0);
                x3_1 = cmul_avx2_aos(x3_1, w3_1);
                x3_2 = cmul_avx2_aos(x3_2, w3_2);
                x3_3 = cmul_avx2_aos(x3_3, w3_3);

                x4_0 = cmul_avx2_aos(x4_0, w4_0);
                x4_1 = cmul_avx2_aos(x4_1, w4_1);
                x4_2 = cmul_avx2_aos(x4_2, w4_2);
                x4_3 = cmul_avx2_aos(x4_3, w4_3);

                x5_0 = cmul_avx2_aos(x5_0, w5_0);
                x5_1 = cmul_avx2_aos(x5_1, w5_1);
                x5_2 = cmul_avx2_aos(x5_2, w5_2);
                x5_3 = cmul_avx2_aos(x5_3, w5_3);

                x6_0 = cmul_avx2_aos(x6_0, w6_0);
                x6_1 = cmul_avx2_aos(x6_1, w6_1);
                x6_2 = cmul_avx2_aos(x6_2, w6_2);
                x6_3 = cmul_avx2_aos(x6_3, w6_3);
            }

            // y0 = sum(x0..x6) per 2-butterfly pair
            __m256d y0_0 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0_0, x1_0), _mm256_add_pd(x2_0, x3_0)),
                                         _mm256_add_pd(_mm256_add_pd(x4_0, x5_0), x6_0));
            __m256d y0_1 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0_1, x1_1), _mm256_add_pd(x2_1, x3_1)),
                                         _mm256_add_pd(_mm256_add_pd(x4_1, x5_1), x6_1));
            __m256d y0_2 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0_2, x1_2), _mm256_add_pd(x2_2, x3_2)),
                                         _mm256_add_pd(_mm256_add_pd(x4_2, x5_2), x6_2));
            __m256d y0_3 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0_3, x1_3), _mm256_add_pd(x2_3, x3_3)),
                                         _mm256_add_pd(_mm256_add_pd(x4_3, x5_3), x6_3));

            // Rader input permute: tx = [x1, x3, x2, x6, x4, x5]
            __m256d tx0_0 = x1_0, tx1_0 = x3_0, tx2_0 = x2_0, tx3_0 = x6_0, tx4_0 = x4_0, tx5_0 = x5_0;
            __m256d tx0_1 = x1_1, tx1_1 = x3_1, tx2_1 = x2_1, tx3_1 = x6_1, tx4_1 = x4_1, tx5_1 = x5_1;
            __m256d tx0_2 = x1_2, tx1_2 = x3_2, tx2_2 = x2_2, tx3_2 = x6_2, tx4_2 = x4_2, tx5_2 = x5_2;
            __m256d tx0_3 = x1_3, tx1_3 = x3_3, tx2_3 = x2_3, tx3_3 = x6_3, tx4_3 = x4_3, tx5_3 = x5_3;

            // 6-pt cyclic convolution conv[q] = Σ_l tx[l] * tw[(q-l) mod 6]
            __m256d c0_0 = cmul_avx2_aos(tx0_0, tw_brd[0]);
            __m256d c0_1 = cmul_avx2_aos(tx0_1, tw_brd[0]);
            __m256d c0_2 = cmul_avx2_aos(tx0_2, tw_brd[0]);
            __m256d c0_3 = cmul_avx2_aos(tx0_3, tw_brd[0]);

            // q=0
            __m256d v0_0 = c0_0;
            __m256d v0_1 = c0_1;
            __m256d v0_2 = c0_2;
            __m256d v0_3 = c0_3;
            v0_0 = _mm256_add_pd(v0_0, cmul_avx2_aos(tx1_0, tw_brd[5]));
            v0_1 = _mm256_add_pd(v0_1, cmul_avx2_aos(tx1_1, tw_brd[5]));
            v0_2 = _mm256_add_pd(v0_2, cmul_avx2_aos(tx1_2, tw_brd[5]));
            v0_3 = _mm256_add_pd(v0_3, cmul_avx2_aos(tx1_3, tw_brd[5]));
            v0_0 = _mm256_add_pd(v0_0, cmul_avx2_aos(tx2_0, tw_brd[4]));
            v0_1 = _mm256_add_pd(v0_1, cmul_avx2_aos(tx2_1, tw_brd[4]));
            v0_2 = _mm256_add_pd(v0_2, cmul_avx2_aos(tx2_2, tw_brd[4]));
            v0_3 = _mm256_add_pd(v0_3, cmul_avx2_aos(tx2_3, tw_brd[4]));
            v0_0 = _mm256_add_pd(v0_0, cmul_avx2_aos(tx3_0, tw_brd[3]));
            v0_1 = _mm256_add_pd(v0_1, cmul_avx2_aos(tx3_1, tw_brd[3]));
            v0_2 = _mm256_add_pd(v0_2, cmul_avx2_aos(tx3_2, tw_brd[3]));
            v0_3 = _mm256_add_pd(v0_3, cmul_avx2_aos(tx3_3, tw_brd[3]));
            v0_0 = _mm256_add_pd(v0_0, cmul_avx2_aos(tx4_0, tw_brd[2]));
            v0_1 = _mm256_add_pd(v0_1, cmul_avx2_aos(tx4_1, tw_brd[2]));
            v0_2 = _mm256_add_pd(v0_2, cmul_avx2_aos(tx4_2, tw_brd[2]));
            v0_3 = _mm256_add_pd(v0_3, cmul_avx2_aos(tx4_3, tw_brd[2]));
            v0_0 = _mm256_add_pd(v0_0, cmul_avx2_aos(tx5_0, tw_brd[1]));
            v0_1 = _mm256_add_pd(v0_1, cmul_avx2_aos(tx5_1, tw_brd[1]));
            v0_2 = _mm256_add_pd(v0_2, cmul_avx2_aos(tx5_2, tw_brd[1]));
            v0_3 = _mm256_add_pd(v0_3, cmul_avx2_aos(tx5_3, tw_brd[1]));

            // q=1
            __m256d v1_0 = cmul_avx2_aos(tx0_0, tw_brd[1]);
            __m256d v1_1 = cmul_avx2_aos(tx0_1, tw_brd[1]);
            __m256d v1_2 = cmul_avx2_aos(tx0_2, tw_brd[1]);
            __m256d v1_3 = cmul_avx2_aos(tx0_3, tw_brd[1]);
            v1_0 = _mm256_add_pd(v1_0, cmul_avx2_aos(tx1_0, tw_brd[0]));
            v1_1 = _mm256_add_pd(v1_1, cmul_avx2_aos(tx1_1, tw_brd[0]));
            v1_2 = _mm256_add_pd(v1_2, cmul_avx2_aos(tx1_2, tw_brd[0]));
            v1_3 = _mm256_add_pd(v1_3, cmul_avx2_aos(tx1_3, tw_brd[0]));
            v1_0 = _mm256_add_pd(v1_0, cmul_avx2_aos(tx2_0, tw_brd[5]));
            v1_1 = _mm256_add_pd(v1_1, cmul_avx2_aos(tx2_1, tw_brd[5]));
            v1_2 = _mm256_add_pd(v1_2, cmul_avx2_aos(tx2_2, tw_brd[5]));
            v1_3 = _mm256_add_pd(v1_3, cmul_avx2_aos(tx2_3, tw_brd[5]));
            v1_0 = _mm256_add_pd(v1_0, cmul_avx2_aos(tx3_0, tw_brd[4]));
            v1_1 = _mm256_add_pd(v1_1, cmul_avx2_aos(tx3_1, tw_brd[4]));
            v1_2 = _mm256_add_pd(v1_2, cmul_avx2_aos(tx3_2, tw_brd[4]));
            v1_3 = _mm256_add_pd(v1_3, cmul_avx2_aos(tx3_3, tw_brd[4]));
            v1_0 = _mm256_add_pd(v1_0, cmul_avx2_aos(tx4_0, tw_brd[3]));
            v1_1 = _mm256_add_pd(v1_1, cmul_avx2_aos(tx4_1, tw_brd[3]));
            v1_2 = _mm256_add_pd(v1_2, cmul_avx2_aos(tx4_2, tw_brd[3]));
            v1_3 = _mm256_add_pd(v1_3, cmul_avx2_aos(tx4_3, tw_brd[3]));
            v1_0 = _mm256_add_pd(v1_0, cmul_avx2_aos(tx5_0, tw_brd[2]));
            v1_1 = _mm256_add_pd(v1_1, cmul_avx2_aos(tx5_1, tw_brd[2]));
            v1_2 = _mm256_add_pd(v1_2, cmul_avx2_aos(tx5_2, tw_brd[2]));
            v1_3 = _mm256_add_pd(v1_3, cmul_avx2_aos(tx5_3, tw_brd[2]));

            // q=2
            __m256d v2_0 = cmul_avx2_aos(tx0_0, tw_brd[2]);
            __m256d v2_1 = cmul_avx2_aos(tx0_1, tw_brd[2]);
            __m256d v2_2 = cmul_avx2_aos(tx0_2, tw_brd[2]);
            __m256d v2_3 = cmul_avx2_aos(tx0_3, tw_brd[2]);
            v2_0 = _mm256_add_pd(v2_0, cmul_avx2_aos(tx1_0, tw_brd[1]));
            v2_1 = _mm256_add_pd(v2_1, cmul_avx2_aos(tx1_1, tw_brd[1]));
            v2_2 = _mm256_add_pd(v2_2, cmul_avx2_aos(tx1_2, tw_brd[1]));
            v2_3 = _mm256_add_pd(v2_3, cmul_avx2_aos(tx1_3, tw_brd[1]));
            v2_0 = _mm256_add_pd(v2_0, cmul_avx2_aos(tx2_0, tw_brd[0]));
            v2_1 = _mm256_add_pd(v2_1, cmul_avx2_aos(tx2_1, tw_brd[0]));
            v2_2 = _mm256_add_pd(v2_2, cmul_avx2_aos(tx2_2, tw_brd[0]));
            v2_3 = _mm256_add_pd(v2_3, cmul_avx2_aos(tx2_3, tw_brd[0]));
            v2_0 = _mm256_add_pd(v2_0, cmul_avx2_aos(tx3_0, tw_brd[5]));
            v2_1 = _mm256_add_pd(v2_1, cmul_avx2_aos(tx3_1, tw_brd[5]));
            v2_2 = _mm256_add_pd(v2_2, cmul_avx2_aos(tx3_2, tw_brd[5]));
            v2_3 = _mm256_add_pd(v2_3, cmul_avx2_aos(tx3_3, tw_brd[5]));
            v2_0 = _mm256_add_pd(v2_0, cmul_avx2_aos(tx4_0, tw_brd[4]));
            v2_1 = _mm256_add_pd(v2_1, cmul_avx2_aos(tx4_1, tw_brd[4]));
            v2_2 = _mm256_add_pd(v2_2, cmul_avx2_aos(tx4_2, tw_brd[4]));
            v2_3 = _mm256_add_pd(v2_3, cmul_avx2_aos(tx4_3, tw_brd[4]));
            v2_0 = _mm256_add_pd(v2_0, cmul_avx2_aos(tx5_0, tw_brd[3]));
            v2_1 = _mm256_add_pd(v2_1, cmul_avx2_aos(tx5_1, tw_brd[3]));
            v2_2 = _mm256_add_pd(v2_2, cmul_avx2_aos(tx5_2, tw_brd[3]));
            v2_3 = _mm256_add_pd(v2_3, cmul_avx2_aos(tx5_3, tw_brd[3]));

            // q=3
            __m256d v3_0 = cmul_avx2_aos(tx0_0, tw_brd[3]);
            __m256d v3_1 = cmul_avx2_aos(tx0_1, tw_brd[3]);
            __m256d v3_2 = cmul_avx2_aos(tx0_2, tw_brd[3]);
            __m256d v3_3 = cmul_avx2_aos(tx0_3, tw_brd[3]);
            v3_0 = _mm256_add_pd(v3_0, cmul_avx2_aos(tx1_0, tw_brd[2]));
            v3_1 = _mm256_add_pd(v3_1, cmul_avx2_aos(tx1_1, tw_brd[2]));
            v3_2 = _mm256_add_pd(v3_2, cmul_avx2_aos(tx1_2, tw_brd[2]));
            v3_3 = _mm256_add_pd(v3_3, cmul_avx2_aos(tx1_3, tw_brd[2]));
            v3_0 = _mm256_add_pd(v3_0, cmul_avx2_aos(tx2_0, tw_brd[1]));
            v3_1 = _mm256_add_pd(v3_1, cmul_avx2_aos(tx2_1, tw_brd[1]));
            v3_2 = _mm256_add_pd(v3_2, cmul_avx2_aos(tx2_2, tw_brd[1]));
            v3_3 = _mm256_add_pd(v3_3, cmul_avx2_aos(tx2_3, tw_brd[1]));
            v3_0 = _mm256_add_pd(v3_0, cmul_avx2_aos(tx3_0, tw_brd[0]));
            v3_1 = _mm256_add_pd(v3_1, cmul_avx2_aos(tx3_1, tw_brd[0]));
            v3_2 = _mm256_add_pd(v3_2, cmul_avx2_aos(tx3_2, tw_brd[0]));
            v3_3 = _mm256_add_pd(v3_3, cmul_avx2_aos(tx3_3, tw_brd[0]));
            v3_0 = _mm256_add_pd(v3_0, cmul_avx2_aos(tx4_0, tw_brd[5]));
            v3_1 = _mm256_add_pd(v3_1, cmul_avx2_aos(tx4_1, tw_brd[5]));
            v3_2 = _mm256_add_pd(v3_2, cmul_avx2_aos(tx4_2, tw_brd[5]));
            v3_3 = _mm256_add_pd(v3_3, cmul_avx2_aos(tx4_3, tw_brd[5]));
            v3_0 = _mm256_add_pd(v3_0, cmul_avx2_aos(tx5_0, tw_brd[4]));
            v3_1 = _mm256_add_pd(v3_1, cmul_avx2_aos(tx5_1, tw_brd[4]));
            v3_2 = _mm256_add_pd(v3_2, cmul_avx2_aos(tx5_2, tw_brd[4]));
            v3_3 = _mm256_add_pd(v3_3, cmul_avx2_aos(tx5_3, tw_brd[4]));

            // q=4
            __m256d v4_0 = cmul_avx2_aos(tx0_0, tw_brd[4]);
            __m256d v4_1 = cmul_avx2_aos(tx0_1, tw_brd[4]);
            __m256d v4_2 = cmul_avx2_aos(tx0_2, tw_brd[4]);
            __m256d v4_3 = cmul_avx2_aos(tx0_3, tw_brd[4]);
            v4_0 = _mm256_add_pd(v4_0, cmul_avx2_aos(tx1_0, tw_brd[3]));
            v4_1 = _mm256_add_pd(v4_1, cmul_avx2_aos(tx1_1, tw_brd[3]));
            v4_2 = _mm256_add_pd(v4_2, cmul_avx2_aos(tx1_2, tw_brd[3]));
            v4_3 = _mm256_add_pd(v4_3, cmul_avx2_aos(tx1_3, tw_brd[3]));
            v4_0 = _mm256_add_pd(v4_0, cmul_avx2_aos(tx2_0, tw_brd[2]));
            v4_1 = _mm256_add_pd(v4_1, cmul_avx2_aos(tx2_1, tw_brd[2]));
            v4_2 = _mm256_add_pd(v4_2, cmul_avx2_aos(tx2_2, tw_brd[2]));
            v4_3 = _mm256_add_pd(v4_3, cmul_avx2_aos(tx2_3, tw_brd[2]));
            v4_0 = _mm256_add_pd(v4_0, cmul_avx2_aos(tx3_0, tw_brd[1]));
            v4_1 = _mm256_add_pd(v4_1, cmul_avx2_aos(tx3_1, tw_brd[1]));
            v4_2 = _mm256_add_pd(v4_2, cmul_avx2_aos(tx3_2, tw_brd[1]));
            v4_3 = _mm256_add_pd(v4_3, cmul_avx2_aos(tx3_3, tw_brd[1]));
            v4_0 = _mm256_add_pd(v4_0, cmul_avx2_aos(tx4_0, tw_brd[0]));
            v4_1 = _mm256_add_pd(v4_1, cmul_avx2_aos(tx4_1, tw_brd[0]));
            v4_2 = _mm256_add_pd(v4_2, cmul_avx2_aos(tx4_2, tw_brd[0]));
            v4_3 = _mm256_add_pd(v4_3, cmul_avx2_aos(tx4_3, tw_brd[0]));
            v4_0 = _mm256_add_pd(v4_0, cmul_avx2_aos(tx5_0, tw_brd[5]));
            v4_1 = _mm256_add_pd(v4_1, cmul_avx2_aos(tx5_1, tw_brd[5]));
            v4_2 = _mm256_add_pd(v4_2, cmul_avx2_aos(tx5_2, tw_brd[5]));
            v4_3 = _mm256_add_pd(v4_3, cmul_avx2_aos(tx5_3, tw_brd[5]));

            // q=5
            __m256d v5_0 = cmul_avx2_aos(tx0_0, tw_brd[5]);
            __m256d v5_1 = cmul_avx2_aos(tx0_1, tw_brd[5]);
            __m256d v5_2 = cmul_avx2_aos(tx0_2, tw_brd[5]);
            __m256d v5_3 = cmul_avx2_aos(tx0_3, tw_brd[5]);
            v5_0 = _mm256_add_pd(v5_0, cmul_avx2_aos(tx1_0, tw_brd[4]));
            v5_1 = _mm256_add_pd(v5_1, cmul_avx2_aos(tx1_1, tw_brd[4]));
            v5_2 = _mm256_add_pd(v5_2, cmul_avx2_aos(tx1_2, tw_brd[4]));
            v5_3 = _mm256_add_pd(v5_3, cmul_avx2_aos(tx1_3, tw_brd[4]));
            v5_0 = _mm256_add_pd(v5_0, cmul_avx2_aos(tx2_0, tw_brd[3]));
            v5_1 = _mm256_add_pd(v5_1, cmul_avx2_aos(tx2_1, tw_brd[3]));
            v5_2 = _mm256_add_pd(v5_2, cmul_avx2_aos(tx2_2, tw_brd[3]));
            v5_3 = _mm256_add_pd(v5_3, cmul_avx2_aos(tx2_3, tw_brd[3]));
            v5_0 = _mm256_add_pd(v5_0, cmul_avx2_aos(tx3_0, tw_brd[2]));
            v5_1 = _mm256_add_pd(v5_1, cmul_avx2_aos(tx3_1, tw_brd[2]));
            v5_2 = _mm256_add_pd(v5_2, cmul_avx2_aos(tx3_2, tw_brd[2]));
            v5_3 = _mm256_add_pd(v5_3, cmul_avx2_aos(tx3_3, tw_brd[2]));
            v5_0 = _mm256_add_pd(v5_0, cmul_avx2_aos(tx4_0, tw_brd[1]));
            v5_1 = _mm256_add_pd(v5_1, cmul_avx2_aos(tx4_1, tw_brd[1]));
            v5_2 = _mm256_add_pd(v5_2, cmul_avx2_aos(tx4_2, tw_brd[1]));
            v5_3 = _mm256_add_pd(v5_3, cmul_avx2_aos(tx4_3, tw_brd[1]));
            v5_0 = _mm256_add_pd(v5_0, cmul_avx2_aos(tx5_0, tw_brd[0]));
            v5_1 = _mm256_add_pd(v5_1, cmul_avx2_aos(tx5_1, tw_brd[0]));
            v5_2 = _mm256_add_pd(v5_2, cmul_avx2_aos(tx5_2, tw_brd[0]));
            v5_3 = _mm256_add_pd(v5_3, cmul_avx2_aos(tx5_3, tw_brd[0]));

            // y[m] = x0 + conv[q], with m = out_perm[q] = [1,5,4,6,2,3]
            __m256d y1_0 = _mm256_add_pd(x0_0, v0_0);
            __m256d y1_1 = _mm256_add_pd(x0_1, v0_1);
            __m256d y1_2 = _mm256_add_pd(x0_2, v0_2);
            __m256d y1_3 = _mm256_add_pd(x0_3, v0_3);

            __m256d y5_0 = _mm256_add_pd(x0_0, v1_0);
            __m256d y5_1 = _mm256_add_pd(x0_1, v1_1);
            __m256d y5_2 = _mm256_add_pd(x0_2, v1_2);
            __m256d y5_3 = _mm256_add_pd(x0_3, v1_3);

            __m256d y4_0 = _mm256_add_pd(x0_0, v2_0);
            __m256d y4_1 = _mm256_add_pd(x0_1, v2_1);
            __m256d y4_2 = _mm256_add_pd(x0_2, v2_2);
            __m256d y4_3 = _mm256_add_pd(x0_3, v2_3);

            __m256d y6_0 = _mm256_add_pd(x0_0, v3_0);
            __m256d y6_1 = _mm256_add_pd(x0_1, v3_1);
            __m256d y6_2 = _mm256_add_pd(x0_2, v3_2);
            __m256d y6_3 = _mm256_add_pd(x0_3, v3_3);

            __m256d y2_0 = _mm256_add_pd(x0_0, v4_0);
            __m256d y2_1 = _mm256_add_pd(x0_1, v4_1);
            __m256d y2_2 = _mm256_add_pd(x0_2, v4_2);
            __m256d y2_3 = _mm256_add_pd(x0_3, v4_3);

            __m256d y3_0 = _mm256_add_pd(x0_0, v5_0);
            __m256d y3_1 = _mm256_add_pd(x0_1, v5_1);
            __m256d y3_2 = _mm256_add_pd(x0_2, v5_2);
            __m256d y3_3 = _mm256_add_pd(x0_3, v5_3);

            // Store (pure AoS, like radix-5)
            STOREU_PD(&output_buffer[k + 0 * seventh + 0].re, y0_0);
            STOREU_PD(&output_buffer[k + 0 * seventh + 2].re, y0_1);
            STOREU_PD(&output_buffer[k + 0 * seventh + 4].re, y0_2);
            STOREU_PD(&output_buffer[k + 0 * seventh + 6].re, y0_3);

            STOREU_PD(&output_buffer[k + 1 * seventh + 0].re, y1_0);
            STOREU_PD(&output_buffer[k + 1 * seventh + 2].re, y1_1);
            STOREU_PD(&output_buffer[k + 1 * seventh + 4].re, y1_2);
            STOREU_PD(&output_buffer[k + 1 * seventh + 6].re, y1_3);

            STOREU_PD(&output_buffer[k + 2 * seventh + 0].re, y2_0);
            STOREU_PD(&output_buffer[k + 2 * seventh + 2].re, y2_1);
            STOREU_PD(&output_buffer[k + 2 * seventh + 4].re, y2_2);
            STOREU_PD(&output_buffer[k + 2 * seventh + 6].re, y2_3);

            STOREU_PD(&output_buffer[k + 3 * seventh + 0].re, y3_0);
            STOREU_PD(&output_buffer[k + 3 * seventh + 2].re, y3_1);
            STOREU_PD(&output_buffer[k + 3 * seventh + 4].re, y3_2);
            STOREU_PD(&output_buffer[k + 3 * seventh + 6].re, y3_3);

            STOREU_PD(&output_buffer[k + 4 * seventh + 0].re, y4_0);
            STOREU_PD(&output_buffer[k + 4 * seventh + 2].re, y4_1);
            STOREU_PD(&output_buffer[k + 4 * seventh + 4].re, y4_2);
            STOREU_PD(&output_buffer[k + 4 * seventh + 6].re, y4_3);

            STOREU_PD(&output_buffer[k + 5 * seventh + 0].re, y5_0);
            STOREU_PD(&output_buffer[k + 5 * seventh + 2].re, y5_1);
            STOREU_PD(&output_buffer[k + 5 * seventh + 4].re, y5_2);
            STOREU_PD(&output_buffer[k + 5 * seventh + 6].re, y5_3);

            STOREU_PD(&output_buffer[k + 6 * seventh + 0].re, y6_0);
            STOREU_PD(&output_buffer[k + 6 * seventh + 2].re, y6_1);
            STOREU_PD(&output_buffer[k + 6 * seventh + 4].re, y6_2);
            STOREU_PD(&output_buffer[k + 6 * seventh + 6].re, y6_3);
        }

        // -----------------------------
        // 2x AVX2 cleanup
        // -----------------------------
        for (; k + 1 < seventh; k += 2)
        {
            __m256d x0 = load2_aos(&sub_outputs[k + 0 * seventh], &sub_outputs[k + 1 + 0 * seventh]);
            __m256d x1 = load2_aos(&sub_outputs[k + 1 * seventh], &sub_outputs[k + 1 + 1 * seventh]);
            __m256d x2 = load2_aos(&sub_outputs[k + 2 * seventh], &sub_outputs[k + 1 + 2 * seventh]);
            __m256d x3 = load2_aos(&sub_outputs[k + 3 * seventh], &sub_outputs[k + 1 + 3 * seventh]);
            __m256d x4 = load2_aos(&sub_outputs[k + 4 * seventh], &sub_outputs[k + 1 + 4 * seventh]);
            __m256d x5 = load2_aos(&sub_outputs[k + 5 * seventh], &sub_outputs[k + 1 + 5 * seventh]);
            __m256d x6 = load2_aos(&sub_outputs[k + 6 * seventh], &sub_outputs[k + 1 + 6 * seventh]);

            if (seventh > 1)
            {
                __m256d w1 = load2_aos(&stage_tw[6 * k + 0], &stage_tw[6 * (k + 1) + 0]);
                __m256d w2 = load2_aos(&stage_tw[6 * k + 1], &stage_tw[6 * (k + 1) + 1]);
                __m256d w3 = load2_aos(&stage_tw[6 * k + 2], &stage_tw[6 * (k + 1) + 2]);
                __m256d w4 = load2_aos(&stage_tw[6 * k + 3], &stage_tw[6 * (k + 1) + 3]);
                __m256d w5 = load2_aos(&stage_tw[6 * k + 4], &stage_tw[6 * (k + 1) + 4]);
                __m256d w6 = load2_aos(&stage_tw[6 * k + 5], &stage_tw[6 * (k + 1) + 5]);
                x1 = cmul_avx2_aos(x1, w1);
                x2 = cmul_avx2_aos(x2, w2);
                x3 = cmul_avx2_aos(x3, w3);
                x4 = cmul_avx2_aos(x4, w4);
                x5 = cmul_avx2_aos(x5, w5);
                x6 = cmul_avx2_aos(x6, w6);
            }

            __m256d y0 = _mm256_add_pd(_mm256_add_pd(_mm256_add_pd(x0, x1), _mm256_add_pd(x2, x3)),
                                       _mm256_add_pd(_mm256_add_pd(x4, x5), x6));

            __m256d t0 = x1, t1 = x3, t2 = x2, t3 = x6, t4 = x4, t5 = x5;

            __m256d u0 = cmul_avx2_aos(t0, tw_brd[0]); // q=0
            u0 = _mm256_add_pd(u0, cmul_avx2_aos(t1, tw_brd[5]));
            u0 = _mm256_add_pd(u0, cmul_avx2_aos(t2, tw_brd[4]));
            u0 = _mm256_add_pd(u0, cmul_avx2_aos(t3, tw_brd[3]));
            u0 = _mm256_add_pd(u0, cmul_avx2_aos(t4, tw_brd[2]));
            u0 = _mm256_add_pd(u0, cmul_avx2_aos(t5, tw_brd[1]));

            __m256d u1 = cmul_avx2_aos(t0, tw_brd[1]); // q=1
            u1 = _mm256_add_pd(u1, cmul_avx2_aos(t1, tw_brd[0]));
            u1 = _mm256_add_pd(u1, cmul_avx2_aos(t2, tw_brd[5]));
            u1 = _mm256_add_pd(u1, cmul_avx2_aos(t3, tw_brd[4]));
            u1 = _mm256_add_pd(u1, cmul_avx2_aos(t4, tw_brd[3]));
            u1 = _mm256_add_pd(u1, cmul_avx2_aos(t5, tw_brd[2]));

            __m256d u2 = cmul_avx2_aos(t0, tw_brd[2]); // q=2
            u2 = _mm256_add_pd(u2, cmul_avx2_aos(t1, tw_brd[1]));
            u2 = _mm256_add_pd(u2, cmul_avx2_aos(t2, tw_brd[0]));
            u2 = _mm256_add_pd(u2, cmul_avx2_aos(t3, tw_brd[5]));
            u2 = _mm256_add_pd(u2, cmul_avx2_aos(t4, tw_brd[4]));
            u2 = _mm256_add_pd(u2, cmul_avx2_aos(t5, tw_brd[3]));

            __m256d u3 = cmul_avx2_aos(t0, tw_brd[3]); // q=3
            u3 = _mm256_add_pd(u3, cmul_avx2_aos(t1, tw_brd[2]));
            u3 = _mm256_add_pd(u3, cmul_avx2_aos(t2, tw_brd[1]));
            u3 = _mm256_add_pd(u3, cmul_avx2_aos(t3, tw_brd[0]));
            u3 = _mm256_add_pd(u3, cmul_avx2_aos(t4, tw_brd[5]));
            u3 = _mm256_add_pd(u3, cmul_avx2_aos(t5, tw_brd[4]));

            __m256d u4 = cmul_avx2_aos(t0, tw_brd[4]); // q=4
            u4 = _mm256_add_pd(u4, cmul_avx2_aos(t1, tw_brd[3]));
            u4 = _mm256_add_pd(u4, cmul_avx2_aos(t2, tw_brd[2]));
            u4 = _mm256_add_pd(u4, cmul_avx2_aos(t3, tw_brd[1]));
            u4 = _mm256_add_pd(u4, cmul_avx2_aos(t4, tw_brd[0]));
            u4 = _mm256_add_pd(u4, cmul_avx2_aos(t5, tw_brd[5]));

            __m256d u5 = cmul_avx2_aos(t0, tw_brd[5]); // q=5
            u5 = _mm256_add_pd(u5, cmul_avx2_aos(t1, tw_brd[4]));
            u5 = _mm256_add_pd(u5, cmul_avx2_aos(t2, tw_brd[3]));
            u5 = _mm256_add_pd(u5, cmul_avx2_aos(t3, tw_brd[2]));
            u5 = _mm256_add_pd(u5, cmul_avx2_aos(t4, tw_brd[1]));
            u5 = _mm256_add_pd(u5, cmul_avx2_aos(t5, tw_brd[0]));

            __m256d y1 = _mm256_add_pd(x0, u0); // m = 1
            __m256d y5 = _mm256_add_pd(x0, u1); // m = 5
            __m256d y4 = _mm256_add_pd(x0, u2); // m = 4
            __m256d y6 = _mm256_add_pd(x0, u3); // m = 6
            __m256d y2 = _mm256_add_pd(x0, u4); // m = 2
            __m256d y3 = _mm256_add_pd(x0, u5); // m = 3

            STOREU_PD(&output_buffer[k + 0 * seventh].re, y0);
            STOREU_PD(&output_buffer[k + 1 * seventh].re, y1);
            STOREU_PD(&output_buffer[k + 2 * seventh].re, y2);
            STOREU_PD(&output_buffer[k + 3 * seventh].re, y3);
            STOREU_PD(&output_buffer[k + 4 * seventh].re, y4);
            STOREU_PD(&output_buffer[k + 5 * seventh].re, y5);
            STOREU_PD(&output_buffer[k + 6 * seventh].re, y6);
        }
#endif // __AVX2__

        // -----------------------------
        // Scalar tail (0..1 leftover)
        // -----------------------------
        for (; k < seventh; ++k)
        {
            // scalar Rader
            // load
            fft_data x0 = sub_outputs[k + 0 * seventh];
            fft_data x1 = sub_outputs[k + 1 * seventh];
            fft_data x2 = sub_outputs[k + 2 * seventh];
            fft_data x3 = sub_outputs[k + 3 * seventh];
            fft_data x4 = sub_outputs[k + 4 * seventh];
            fft_data x5 = sub_outputs[k + 5 * seventh];
            fft_data x6 = sub_outputs[k + 6 * seventh];

            if (seventh > 1)
            {
                const int base = 6 * k;
                fft_data w1 = stage_tw[base + 0], w2 = stage_tw[base + 1], w3 = stage_tw[base + 2];
                fft_data w4 = stage_tw[base + 3], w5 = stage_tw[base + 4], w6 = stage_tw[base + 5];
                // DIT twiddles
                fft_data t;
                t = x1;
                x1.re = t.re * w1.re - t.im * w1.im;
                x1.im = t.re * w1.im + t.im * w1.re;
                t = x2;
                x2.re = t.re * w2.re - t.im * w2.im;
                x2.im = t.re * w2.im + t.im * w2.re;
                t = x3;
                x3.re = t.re * w3.re - t.im * w3.im;
                x3.im = t.re * w3.im + t.im * w3.re;
                t = x4;
                x4.re = t.re * w4.re - t.im * w4.im;
                x4.im = t.re * w4.im + t.im * w4.re;
                t = x5;
                x5.re = t.re * w5.re - t.im * w5.im;
                x5.im = t.re * w5.im + t.im * w5.re;
                t = x6;
                x6.re = t.re * w6.re - t.im * w6.im;
                x6.im = t.re * w6.im + t.im * w6.re;
            }

            // y0
            fft_data y0 = x0;
            y0.re += x1.re + x2.re + x3.re + x4.re + x5.re + x6.re;
            y0.im += x1.im + x2.im + x3.im + x4.im + x5.im + x6.im;

            // Rader tx = [x1, x3, x2, x6, x4, x5]
            fft_data tx[6] = {x1, x3, x2, x6, x4, x5};

            // tw[q] @ out_perm = [1,5,4,6,2,3]
            fft_data tw[6];
            {
                const int op[6] = {1, 5, 4, 6, 2, 3};
                for (int q = 0; q < 6; ++q)
                {
                    double a = op[q] * base_angle;
#ifdef __GNUC__
                    sincos(a, &tw[q].im, &tw[q].re);
#else
                    tw[q].re = cos(a);
                    tw[q].im = sin(a);
#endif
                }
            }

            // conv
            fft_data v[6] = {{0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}, {0, 0}};
            for (int q = 0; q < 6; ++q)
            {
                int idx;
                // l=0..5 unrolled
                idx = q;
                v[q].re += tx[0].re * tw[idx].re - tx[0].im * tw[idx].im;
                v[q].im += tx[0].re * tw[idx].im + tx[0].im * tw[idx].re;
                idx = q - 1;
                if (idx < 0)
                    idx += 6;
                v[q].re += tx[1].re * tw[idx].re - tx[1].im * tw[idx].im;
                v[q].im += tx[1].re * tw[idx].im + tx[1].im * tw[idx].re;
                idx = q - 2;
                if (idx < 0)
                    idx += 6;
                v[q].re += tx[2].re * tw[idx].re - tx[2].im * tw[idx].im;
                v[q].im += tx[2].re * tw[idx].im + tx[2].im * tw[idx].re;
                idx = q - 3;
                if (idx < 0)
                    idx += 6;
                v[q].re += tx[3].re * tw[idx].re - tx[3].im * tw[idx].im;
                v[q].im += tx[3].re * tw[idx].im + tx[3].im * tw[idx].re;
                idx = q - 4;
                if (idx < 0)
                    idx += 6;
                v[q].re += tx[4].re * tw[idx].re - tx[4].im * tw[idx].im;
                v[q].im += tx[4].re * tw[idx].im + tx[4].im * tw[idx].re;
                idx = q - 5;
                if (idx < 0)
                    idx += 6;
                v[q].re += tx[5].re * tw[idx].re - tx[5].im * tw[idx].im;
                v[q].im += tx[5].re * tw[idx].im + tx[5].im * tw[idx].re;
            }

            // out_perm: [1,5,4,6,2,3]
            fft_data y1 = {x0.re + v[0].re, x0.im + v[0].im};
            fft_data y5 = {x0.re + v[1].re, x0.im + v[1].im};
            fft_data y4 = {x0.re + v[2].re, x0.im + v[2].im};
            fft_data y6 = {x0.re + v[3].re, x0.im + v[3].im};
            fft_data y2 = {x0.re + v[4].re, x0.im + v[4].im};
            fft_data y3 = {x0.re + v[5].re, x0.im + v[5].im};

            output_buffer[k + 0 * seventh] = y0;
            output_buffer[k + 1 * seventh] = y1;
            output_buffer[k + 2 * seventh] = y2;
            output_buffer[k + 3 * seventh] = y3;
            output_buffer[k + 4 * seventh] = y4;
            output_buffer[k + 5 * seventh] = y5;
            output_buffer[k + 6 * seventh] = y6;
        }
}