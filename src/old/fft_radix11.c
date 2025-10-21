#include "fft_radix11.h"   // ✅ Gets highSpeedFFT.h → fft_types.h
#include "simd_math.h"    // ✅ Gets complex math operations

// --- Radix-11 constants ---
static const double C11_1 = 0.8412535328311812;   // cos(2π/11)
static const double C11_2 = 0.4154150130018864;   // cos(4π/11)
static const double C11_3 = -0.14231483827328514; // cos(6π/11)
static const double C11_4 = -0.6548607339452850;  // cos(8π/11)
static const double C11_5 = -0.9594929736144974;  // cos(10π/11)
static const double S11_1 = 0.5406408174555976;   // sin(2π/11)
static const double S11_2 = 0.9096319953545184;   // sin(4π/11)
static const double S11_3 = 0.9898214418809327;   // sin(6π/11)
static const double S11_4 = 0.7557495743542583;   // sin(8π/11)
static const double S11_5 = 0.28173255684142967;  // sin(10π/11)

void fft_radix11_butterfly(
    fft_data *output_buffer,
    fft_data *sub_outputs,
    const fft_data *stage_tw,
    int sub_len,
    int transform_sign)
{
        //==========================================================================
        // RADIX-11 BUTTERFLY (Rader DIT with 5 symmetric pairs)
        //
        // Uses Rader's algorithm: maps 11-point DFT to cyclic convolution
        // Exploits symmetry: 5 pairs (k, 11-k) for k=1..5
        //
        // Input:  sub_outputs[0..sub_len-1] through [10*sub_len..11*sub_len-1]
        //         stage_tw[10*k..10*k+9] (W^k through W^{10k}) k-major
        // Output: output_buffer in 11 lanes (Y_0..Y_10)
        //==========================================================================

        const int eleventh = sub_len;
        int k = 0;

#ifdef __AVX2__
        //----------------------------------------------------------------------
        // AVX2 PATH: Process 4 butterflies per iteration (SoA with FMA)
        //----------------------------------------------------------------------
        const __m256d vc1 = _mm256_set1_pd(C11_1); // cos(2π/11)
        const __m256d vc2 = _mm256_set1_pd(C11_2); // cos(4π/11)
        const __m256d vc3 = _mm256_set1_pd(C11_3); // cos(6π/11)
        const __m256d vc4 = _mm256_set1_pd(C11_4); // cos(8π/11)
        const __m256d vc5 = _mm256_set1_pd(C11_5); // cos(10π/11)
        const __m256d vs1 = _mm256_set1_pd(S11_1); // sin(2π/11)
        const __m256d vs2 = _mm256_set1_pd(S11_2); // sin(4π/11)
        const __m256d vs3 = _mm256_set1_pd(S11_3); // sin(6π/11)
        const __m256d vs4 = _mm256_set1_pd(S11_4); // sin(8π/11)
        const __m256d vs5 = _mm256_set1_pd(S11_5); // sin(10π/11)

        for (; k + 3 < eleventh; k += 4)
        {
            if (k + 8 < eleventh)
            {
                _mm_prefetch((const char *)&sub_outputs[k + 8].re, _MM_HINT_T0);
                _mm_prefetch((const char *)&stage_tw[10 * (k + 8)].re, _MM_HINT_T0);
            }

            // AoS -> SoA: Load 11 lanes × 4 points
            double aR[4], aI[4], bR[4], bI[4], cR[4], cI[4], dR[4], dI[4];
            double eR[4], eI[4], fR[4], fI[4], gR[4], gI[4], hR[4], hI[4];
            double iR[4], iI[4], jR[4], jI[4], kR[4], kI[4];

            deinterleave4_aos_to_soa(&sub_outputs[k], aR, aI);
            deinterleave4_aos_to_soa(&sub_outputs[k + eleventh], bR, bI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 2 * eleventh], cR, cI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 3 * eleventh], dR, dI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 4 * eleventh], eR, eI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 5 * eleventh], fR, fI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 6 * eleventh], gR, gI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 7 * eleventh], hR, hI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 8 * eleventh], iR, iI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 9 * eleventh], jR, jI);
            deinterleave4_aos_to_soa(&sub_outputs[k + 10 * eleventh], kR, kI);

            __m256d Ar = _mm256_loadu_pd(aR), Ai = _mm256_loadu_pd(aI);
            __m256d Br = _mm256_loadu_pd(bR), Bi = _mm256_loadu_pd(bI);
            __m256d Cr = _mm256_loadu_pd(cR), Ci = _mm256_loadu_pd(cI);
            __m256d Dr = _mm256_loadu_pd(dR), Di = _mm256_loadu_pd(dI);
            __m256d Er = _mm256_loadu_pd(eR), Ei = _mm256_loadu_pd(eI);
            __m256d Fr = _mm256_loadu_pd(fR), Fi = _mm256_loadu_pd(fI);
            __m256d Gr = _mm256_loadu_pd(gR), Gi = _mm256_loadu_pd(gI);
            __m256d Hr = _mm256_loadu_pd(hR), Hi = _mm256_loadu_pd(hI);
            __m256d Ir = _mm256_loadu_pd(iR), Ii = _mm256_loadu_pd(iI);
            __m256d Jr = _mm256_loadu_pd(jR), Ji = _mm256_loadu_pd(jI);
            __m256d Kr = _mm256_loadu_pd(kR), Ki = _mm256_loadu_pd(kI);

            // Load twiddles (k-major: W^k..W^{10k} at 10*k)
            fft_data w1a[4], w2a[4], w3a[4], w4a[4], w5a[4];
            fft_data w6a[4], w7a[4], w8a[4], w9a[4], w10a[4];
            for (int p = 0; p < 4; ++p)
            {
                w1a[p] = stage_tw[10 * (k + p)];
                w2a[p] = stage_tw[10 * (k + p) + 1];
                w3a[p] = stage_tw[10 * (k + p) + 2];
                w4a[p] = stage_tw[10 * (k + p) + 3];
                w5a[p] = stage_tw[10 * (k + p) + 4];
                w6a[p] = stage_tw[10 * (k + p) + 5];
                w7a[p] = stage_tw[10 * (k + p) + 6];
                w8a[p] = stage_tw[10 * (k + p) + 7];
                w9a[p] = stage_tw[10 * (k + p) + 8];
                w10a[p] = stage_tw[10 * (k + p) + 9];
            }

            double w1R[4], w1I[4], w2R[4], w2I[4], w3R[4], w3I[4], w4R[4], w4I[4];
            double w5R[4], w5I[4], w6R[4], w6I[4], w7R[4], w7I[4], w8R[4], w8I[4];
            double w9R[4], w9I[4], w10R[4], w10I[4];

            deinterleave4_aos_to_soa(w1a, w1R, w1I);
            deinterleave4_aos_to_soa(w2a, w2R, w2I);
            deinterleave4_aos_to_soa(w3a, w3R, w3I);
            deinterleave4_aos_to_soa(w4a, w4R, w4I);
            deinterleave4_aos_to_soa(w5a, w5R, w5I);
            deinterleave4_aos_to_soa(w6a, w6R, w6I);
            deinterleave4_aos_to_soa(w7a, w7R, w7I);
            deinterleave4_aos_to_soa(w8a, w8R, w8I);
            deinterleave4_aos_to_soa(w9a, w9R, w9I);
            deinterleave4_aos_to_soa(w10a, w10R, w10I);

            __m256d W1r = _mm256_loadu_pd(w1R), W1i = _mm256_loadu_pd(w1I);
            __m256d W2r = _mm256_loadu_pd(w2R), W2i = _mm256_loadu_pd(w2I);
            __m256d W3r = _mm256_loadu_pd(w3R), W3i = _mm256_loadu_pd(w3I);
            __m256d W4r = _mm256_loadu_pd(w4R), W4i = _mm256_loadu_pd(w4I);
            __m256d W5r = _mm256_loadu_pd(w5R), W5i = _mm256_loadu_pd(w5I);
            __m256d W6r = _mm256_loadu_pd(w6R), W6i = _mm256_loadu_pd(w6I);
            __m256d W7r = _mm256_loadu_pd(w7R), W7i = _mm256_loadu_pd(w7I);
            __m256d W8r = _mm256_loadu_pd(w8R), W8i = _mm256_loadu_pd(w8I);
            __m256d W9r = _mm256_loadu_pd(w9R), W9i = _mm256_loadu_pd(w9I);
            __m256d W10r = _mm256_loadu_pd(w10R), W10i = _mm256_loadu_pd(w10I);

            // Twiddle multiply
            __m256d b2r, b2i, c2r, c2i, d2r, d2i, e2r, e2i, f2r, f2i;
            __m256d g2r, g2i, h2r, h2i, i2r, i2i, j2r, j2i, k2r, k2i;

            cmul_soa_avx(Br, Bi, W1r, W1i, &b2r, &b2i);
            cmul_soa_avx(Cr, Ci, W2r, W2i, &c2r, &c2i);
            cmul_soa_avx(Dr, Di, W3r, W3i, &d2r, &d2i);
            cmul_soa_avx(Er, Ei, W4r, W4i, &e2r, &e2i);
            cmul_soa_avx(Fr, Fi, W5r, W5i, &f2r, &f2i);
            cmul_soa_avx(Gr, Gi, W6r, W6i, &g2r, &g2i);
            cmul_soa_avx(Hr, Hi, W7r, W7i, &h2r, &h2i);
            cmul_soa_avx(Ir, Ii, W8r, W8i, &i2r, &i2i);
            cmul_soa_avx(Jr, Ji, W9r, W9i, &j2r, &j2i);
            cmul_soa_avx(Kr, Ki, W10r, W10i, &k2r, &k2i);

            // Form 5 symmetric pairs: (b,k), (c,j), (d,i), (e,h), (f,g)
            __m256d t0r = _mm256_add_pd(b2r, k2r), t0i = _mm256_add_pd(b2i, k2i);
            __m256d t1r = _mm256_add_pd(c2r, j2r), t1i = _mm256_add_pd(c2i, j2i);
            __m256d t2r = _mm256_add_pd(d2r, i2r), t2i = _mm256_add_pd(d2i, i2i);
            __m256d t3r = _mm256_add_pd(e2r, h2r), t3i = _mm256_add_pd(e2i, h2i);
            __m256d t4r = _mm256_add_pd(f2r, g2r), t4i = _mm256_add_pd(f2i, g2i);

            __m256d s0r = _mm256_sub_pd(b2r, k2r), s0i = _mm256_sub_pd(b2i, k2i);
            __m256d s1r = _mm256_sub_pd(c2r, j2r), s1i = _mm256_sub_pd(c2i, j2i);
            __m256d s2r = _mm256_sub_pd(d2r, i2r), s2i = _mm256_sub_pd(d2i, i2i);
            __m256d s3r = _mm256_sub_pd(e2r, h2r), s3i = _mm256_sub_pd(e2i, h2i);
            __m256d s4r = _mm256_sub_pd(f2r, g2r), s4i = _mm256_sub_pd(f2i, g2i);

            // Y_0 = a + t0 + t1 + t2 + t3 + t4
            __m256d sum_t_r = _mm256_add_pd(_mm256_add_pd(t0r, t1r),
                                            _mm256_add_pd(_mm256_add_pd(t2r, t3r), t4r));
            __m256d sum_t_i = _mm256_add_pd(_mm256_add_pd(t0i, t1i),
                                            _mm256_add_pd(_mm256_add_pd(t2i, t3i), t4i));
            __m256d y0r = _mm256_add_pd(Ar, sum_t_r);
            __m256d y0i = _mm256_add_pd(Ai, sum_t_i);

            // For pairs Y_m / Y_{11-m}, m=1..5:
            // Real part: a + c1*t0 + c2*t1 + c3*t2 + c4*t3 + c5*t4
            // Imag rot:  s1*s0 + s2*s1 + s3*s2 + s4*s3 + s5*s4

            // Pair 1: Y_1, Y_10
            __m256d tmp1r = _mm256_add_pd(Ar, FMADD(vc1, t0r, FMADD(vc2, t1r, FMADD(vc3, t2r, FMADD(vc4, t3r, _mm256_mul_pd(vc5, t4r))))));
            __m256d tmp1i = _mm256_add_pd(Ai, FMADD(vc1, t0i, FMADD(vc2, t1i, FMADD(vc3, t2i, FMADD(vc4, t3i, _mm256_mul_pd(vc5, t4i))))));

            __m256d base1r = FMADD(vs1, s0r, FMADD(vs2, s1r, FMADD(vs3, s2r, FMADD(vs4, s3r, _mm256_mul_pd(vs5, s4r)))));
            __m256d base1i = FMADD(vs1, s0i, FMADD(vs2, s1i, FMADD(vs3, s2i, FMADD(vs4, s3i, _mm256_mul_pd(vs5, s4i)))));

            __m256d r1r, r1i;
            rot90_soa_avx(base1r, base1i, transform_sign, &r1r, &r1i);

            __m256d y1r = _mm256_add_pd(tmp1r, r1r), y1i = _mm256_add_pd(tmp1i, r1i);
            __m256d y10r = _mm256_sub_pd(tmp1r, r1r), y10i = _mm256_sub_pd(tmp1i, r1i);

            // Pair 2: Y_2, Y_9 (permute coefficients cyclically)
            __m256d tmp2r = _mm256_add_pd(Ar, FMADD(vc2, t0r, FMADD(vc4, t1r, FMADD(vc5, t2r, FMADD(vc3, t3r, _mm256_mul_pd(vc1, t4r))))));
            __m256d tmp2i = _mm256_add_pd(Ai, FMADD(vc2, t0i, FMADD(vc4, t1i, FMADD(vc5, t2i, FMADD(vc3, t3i, _mm256_mul_pd(vc1, t4i))))));

            __m256d base2r = FMADD(vs2, s0r, FMADD(vs4, s1r, FMADD(vs5, s2r, FMADD(vs3, s3r, _mm256_mul_pd(vs1, s4r)))));
            __m256d base2i = FMADD(vs2, s0i, FMADD(vs4, s1i, FMADD(vs5, s2i, FMADD(vs3, s3i, _mm256_mul_pd(vs1, s4i)))));

            __m256d r2r, r2i;
            rot90_soa_avx(base2r, base2i, transform_sign, &r2r, &r2i);

            __m256d y2r = _mm256_add_pd(tmp2r, r2r), y2i = _mm256_add_pd(tmp2i, r2i);
            __m256d y9r = _mm256_sub_pd(tmp2r, r2r), y9i = _mm256_sub_pd(tmp2i, r2i);

            // Pair 3: Y_3, Y_8
            __m256d tmp3r = _mm256_add_pd(Ar, FMADD(vc3, t0r, FMADD(vc5, t1r, FMADD(vc2, t2r, FMADD(vc1, t3r, _mm256_mul_pd(vc4, t4r))))));
            __m256d tmp3i = _mm256_add_pd(Ai, FMADD(vc3, t0i, FMADD(vc5, t1i, FMADD(vc2, t2i, FMADD(vc1, t3i, _mm256_mul_pd(vc4, t4i))))));

            __m256d base3r = FMADD(vs3, s0r, FMADD(vs5, s1r, FMADD(vs2, s2r, FMADD(vs1, s3r, _mm256_mul_pd(vs4, s4r)))));
            __m256d base3i = FMADD(vs3, s0i, FMADD(vs5, s1i, FMADD(vs2, s2i, FMADD(vs1, s3i, _mm256_mul_pd(vs4, s4i)))));

            __m256d r3r, r3i;
            rot90_soa_avx(base3r, base3i, transform_sign, &r3r, &r3i);

            __m256d y3r = _mm256_add_pd(tmp3r, r3r), y3i = _mm256_add_pd(tmp3i, r3i);
            __m256d y8r = _mm256_sub_pd(tmp3r, r3r), y8i = _mm256_sub_pd(tmp3i, r3i);

            // Pair 4: Y_4, Y_7
            __m256d tmp4r = _mm256_add_pd(Ar, FMADD(vc4, t0r, FMADD(vc3, t1r, FMADD(vc1, t2r, FMADD(vc5, t3r, _mm256_mul_pd(vc2, t4r))))));
            __m256d tmp4i = _mm256_add_pd(Ai, FMADD(vc4, t0i, FMADD(vc3, t1i, FMADD(vc1, t2i, FMADD(vc5, t3i, _mm256_mul_pd(vc2, t4i))))));

            __m256d base4r = FMADD(vs4, s0r, FMADD(vs3, s1r, FMADD(vs1, s2r, FMADD(vs5, s3r, _mm256_mul_pd(vs2, s4r)))));
            __m256d base4i = FMADD(vs4, s0i, FMADD(vs3, s1i, FMADD(vs1, s2i, FMADD(vs5, s3i, _mm256_mul_pd(vs2, s4i)))));

            __m256d r4r, r4i;
            rot90_soa_avx(base4r, base4i, transform_sign, &r4r, &r4i);

            __m256d y4r = _mm256_add_pd(tmp4r, r4r), y4i = _mm256_add_pd(tmp4i, r4i);
            __m256d y7r = _mm256_sub_pd(tmp4r, r4r), y7i = _mm256_sub_pd(tmp4i, r4i);

            // Pair 5: Y_5, Y_6
            __m256d tmp5r = _mm256_add_pd(Ar, FMADD(vc5, t0r, FMADD(vc1, t1r, FMADD(vc4, t2r, FMADD(vc2, t3r, _mm256_mul_pd(vc3, t4r))))));
            __m256d tmp5i = _mm256_add_pd(Ai, FMADD(vc5, t0i, FMADD(vc1, t1i, FMADD(vc4, t2i, FMADD(vc2, t3i, _mm256_mul_pd(vc3, t4i))))));

            __m256d base5r = FMADD(vs5, s0r, FMADD(vs1, s1r, FMADD(vs4, s2r, FMADD(vs2, s3r, _mm256_mul_pd(vs3, s4r)))));
            __m256d base5i = FMADD(vs5, s0i, FMADD(vs1, s1i, FMADD(vs4, s2i, FMADD(vs2, s3i, _mm256_mul_pd(vs3, s4i)))));

            __m256d r5r, r5i;
            rot90_soa_avx(base5r, base5i, transform_sign, &r5r, &r5i);

            __m256d y5r = _mm256_add_pd(tmp5r, r5r), y5i = _mm256_add_pd(tmp5i, r5i);
            __m256d y6r = _mm256_sub_pd(tmp5r, r5r), y6i = _mm256_sub_pd(tmp5i, r5i);

            // SoA -> AoS stores
            double Y0R[4], Y0I[4], Y1R[4], Y1I[4], Y2R[4], Y2I[4], Y3R[4], Y3I[4];
            double Y4R[4], Y4I[4], Y5R[4], Y5I[4], Y6R[4], Y6I[4], Y7R[4], Y7I[4];
            double Y8R[4], Y8I[4], Y9R[4], Y9I[4], Y10R[4], Y10I[4];

            _mm256_storeu_pd(Y0R, y0r);
            _mm256_storeu_pd(Y0I, y0i);
            _mm256_storeu_pd(Y1R, y1r);
            _mm256_storeu_pd(Y1I, y1i);
            _mm256_storeu_pd(Y2R, y2r);
            _mm256_storeu_pd(Y2I, y2i);
            _mm256_storeu_pd(Y3R, y3r);
            _mm256_storeu_pd(Y3I, y3i);
            _mm256_storeu_pd(Y4R, y4r);
            _mm256_storeu_pd(Y4I, y4i);
            _mm256_storeu_pd(Y5R, y5r);
            _mm256_storeu_pd(Y5I, y5i);
            _mm256_storeu_pd(Y6R, y6r);
            _mm256_storeu_pd(Y6I, y6i);
            _mm256_storeu_pd(Y7R, y7r);
            _mm256_storeu_pd(Y7I, y7i);
            _mm256_storeu_pd(Y8R, y8r);
            _mm256_storeu_pd(Y8I, y8i);
            _mm256_storeu_pd(Y9R, y9r);
            _mm256_storeu_pd(Y9I, y9i);
            _mm256_storeu_pd(Y10R, y10r);
            _mm256_storeu_pd(Y10I, y10i);

            interleave4_soa_to_aos(Y0R, Y0I, &output_buffer[k]);
            interleave4_soa_to_aos(Y1R, Y1I, &output_buffer[k + eleventh]);
            interleave4_soa_to_aos(Y2R, Y2I, &output_buffer[k + 2 * eleventh]);
            interleave4_soa_to_aos(Y3R, Y3I, &output_buffer[k + 3 * eleventh]);
            interleave4_soa_to_aos(Y4R, Y4I, &output_buffer[k + 4 * eleventh]);
            interleave4_soa_to_aos(Y5R, Y5I, &output_buffer[k + 5 * eleventh]);
            interleave4_soa_to_aos(Y6R, Y6I, &output_buffer[k + 6 * eleventh]);
            interleave4_soa_to_aos(Y7R, Y7I, &output_buffer[k + 7 * eleventh]);
            interleave4_soa_to_aos(Y8R, Y8I, &output_buffer[k + 8 * eleventh]);
            interleave4_soa_to_aos(Y9R, Y9I, &output_buffer[k + 9 * eleventh]);
            interleave4_soa_to_aos(Y10R, Y10I, &output_buffer[k + 10 * eleventh]);
        }
#endif // __AVX2__

        //======================================================================
        // SCALAR TAIL: Handle remaining 0..3 elements
        //======================================================================
        for (; k < eleventh; ++k)
        {
            // Load 11 lanes
            const fft_data a = sub_outputs[k];
            const fft_data b = sub_outputs[k + eleventh];
            const fft_data c = sub_outputs[k + 2 * eleventh];
            const fft_data d = sub_outputs[k + 3 * eleventh];
            const fft_data e = sub_outputs[k + 4 * eleventh];
            const fft_data f = sub_outputs[k + 5 * eleventh];
            const fft_data g = sub_outputs[k + 6 * eleventh];
            const fft_data h = sub_outputs[k + 7 * eleventh];
            const fft_data i = sub_outputs[k + 8 * eleventh];
            const fft_data j = sub_outputs[k + 9 * eleventh];
            const fft_data kval = sub_outputs[k + 10 * eleventh]; // 'k' is loop var

            // Load twiddles (k-major)
            const fft_data w1 = stage_tw[10 * k];
            const fft_data w2 = stage_tw[10 * k + 1];
            const fft_data w3 = stage_tw[10 * k + 2];
            const fft_data w4 = stage_tw[10 * k + 3];
            const fft_data w5 = stage_tw[10 * k + 4];
            const fft_data w6 = stage_tw[10 * k + 5];
            const fft_data w7 = stage_tw[10 * k + 6];
            const fft_data w8 = stage_tw[10 * k + 7];
            const fft_data w9 = stage_tw[10 * k + 8];
            const fft_data w10 = stage_tw[10 * k + 9];

            // Twiddle multiply
            double b2r = b.re * w1.re - b.im * w1.im, b2i = b.re * w1.im + b.im * w1.re;
            double c2r = c.re * w2.re - c.im * w2.im, c2i = c.re * w2.im + c.im * w2.re;
            double d2r = d.re * w3.re - d.im * w3.im, d2i = d.re * w3.im + d.im * w3.re;
            double e2r = e.re * w4.re - e.im * w4.im, e2i = e.re * w4.im + e.im * w4.re;
            double f2r = f.re * w5.re - f.im * w5.im, f2i = f.re * w5.im + f.im * w5.re;
            double g2r = g.re * w6.re - g.im * w6.im, g2i = g.re * w6.im + g.im * w6.re;
            double h2r = h.re * w7.re - h.im * w7.im, h2i = h.re * w7.im + h.im * w7.re;
            double i2r = i.re * w8.re - i.im * w8.im, i2i = i.re * w8.im + i.im * w8.re;
            double j2r = j.re * w9.re - j.im * w9.im, j2i = j.re * w9.im + j.im * w9.re;
            double k2r = kval.re * w10.re - kval.im * w10.im, k2i = kval.re * w10.im + kval.im * w10.re;

            // Form 5 symmetric pairs
            double t0r = b2r + k2r, t0i = b2i + k2i;
            double t1r = c2r + j2r, t1i = c2i + j2i;
            double t2r = d2r + i2r, t2i = d2i + i2i;
            double t3r = e2r + h2r, t3i = e2i + h2i;
            double t4r = f2r + g2r, t4i = f2i + g2i;

            double s0r = b2r - k2r, s0i = b2i - k2i;
            double s1r = c2r - j2r, s1i = c2i - j2i;
            double s2r = d2r - i2r, s2i = d2i - i2i;
            double s3r = e2r - h2r, s3i = e2i - h2i;
            double s4r = f2r - g2r, s4i = f2i - g2i;

            // Y_0
            fft_data y0 = {
                a.re + (t0r + t1r + t2r + t3r + t4r),
                a.im + (t0i + t1i + t2i + t3i + t4i)};

            // Pair 1: Y_1, Y_10
            double tmp1r = a.re + (C11_1 * t0r + C11_2 * t1r + C11_3 * t2r + C11_4 * t3r + C11_5 * t4r);
            double tmp1i = a.im + (C11_1 * t0i + C11_2 * t1i + C11_3 * t2i + C11_4 * t3i + C11_5 * t4i);
            double base1r = S11_1 * s0r + S11_2 * s1r + S11_3 * s2r + S11_4 * s3r + S11_5 * s4r;
            double base1i = S11_1 * s0i + S11_2 * s1i + S11_3 * s2i + S11_4 * s3i + S11_5 * s4i;
            double r1r = (transform_sign == 1) ? -base1i : base1i;
            double r1i = (transform_sign == 1) ? base1r : -base1r;
            fft_data y1 = {tmp1r + r1r, tmp1i + r1i};
            fft_data y10 = {tmp1r - r1r, tmp1i - r1i};

            // Pair 2: Y_2, Y_9
            double tmp2r = a.re + (C11_2 * t0r + C11_4 * t1r + C11_5 * t2r + C11_3 * t3r + C11_1 * t4r);
            double tmp2i = a.im + (C11_2 * t0i + C11_4 * t1i + C11_5 * t2i + C11_3 * t3i + C11_1 * t4i);
            double base2r = S11_2 * s0r + S11_4 * s1r + S11_5 * s2r + S11_3 * s3r + S11_1 * s4r;
            double base2i = S11_2 * s0i + S11_4 * s1i + S11_5 * s2i + S11_3 * s3i + S11_1 * s4i;
            double r2r = (transform_sign == 1) ? -base2i : base2i;
            double r2i = (transform_sign == 1) ? base2r : -base2r;
            fft_data y2 = {tmp2r + r2r, tmp2i + r2i};
            fft_data y9 = {tmp2r - r2r, tmp2i - r2i};

            // Pair 3: Y_3, Y_8
            double tmp3r = a.re + (C11_3 * t0r + C11_5 * t1r + C11_2 * t2r + C11_1 * t3r + C11_4 * t4r);
            double tmp3i = a.im + (C11_3 * t0i + C11_5 * t1i + C11_2 * t2i + C11_1 * t3i + C11_4 * t4i);
            double base3r = S11_3 * s0r + S11_5 * s1r + S11_2 * s2r + S11_1 * s3r + S11_4 * s4r;
            double base3i = S11_3 * s0i + S11_5 * s1i + S11_2 * s2i + S11_1 * s3i + S11_4 * s4i;
            double r3r = (transform_sign == 1) ? -base3i : base3i;
            double r3i = (transform_sign == 1) ? base3r : -base3r;
            fft_data y3 = {tmp3r + r3r, tmp3i + r3i};
            fft_data y8 = {tmp3r - r3r, tmp3i - r3i};

            // Pair 4: Y_4, Y_7
            double tmp4r = a.re + (C11_4 * t0r + C11_3 * t1r + C11_1 * t2r + C11_5 * t3r + C11_2 * t4r);
            double tmp4i = a.im + (C11_4 * t0i + C11_3 * t1i + C11_1 * t2i + C11_5 * t3i + C11_2 * t4i);
            double base4r = S11_4 * s0r + S11_3 * s1r + S11_1 * s2r + S11_5 * s3r + S11_2 * s4r;
            double base4i = S11_4 * s0i + S11_3 * s1i + S11_1 * s2i + S11_5 * s3i + S11_2 * s4i;
            double r4r = (transform_sign == 1) ? -base4i : base4i;
            double r4i = (transform_sign == 1) ? base4r : -base4r;
            fft_data y4 = {tmp4r + r4r, tmp4i + r4i};
            fft_data y7 = {tmp4r - r4r, tmp4i - r4i};

            // Pair 5: Y_5, Y_6
            double tmp5r = a.re + (C11_5 * t0r + C11_1 * t1r + C11_4 * t2r + C11_2 * t3r + C11_3 * t4r);
            double tmp5i = a.im + (C11_5 * t0i + C11_1 * t1i + C11_4 * t2i + C11_2 * t3i + C11_3 * t4i);
            double base5r = S11_5 * s0r + S11_1 * s1r + S11_4 * s2r + S11_2 * s3r + S11_3 * s4r;
            double base5i = S11_5 * s0i + S11_1 * s1i + S11_4 * s2i + S11_2 * s3i + S11_3 * s4i;
            double r5r = (transform_sign == 1) ? -base5i : base5i;
            double r5i = (transform_sign == 1) ? base5r : -base5r;
            fft_data y5 = {tmp5r + r5r, tmp5i + r5i};
            fft_data y6 = {tmp5r - r5r, tmp5i - r5i};

            // Store
            output_buffer[k] = y0;
            output_buffer[k + eleventh] = y1;
            output_buffer[k + 2 * eleventh] = y2;
            output_buffer[k + 3 * eleventh] = y3;
            output_buffer[k + 4 * eleventh] = y4;
            output_buffer[k + 5 * eleventh] = y5;
            output_buffer[k + 6 * eleventh] = y6;
            output_buffer[k + 7 * eleventh] = y7;
            output_buffer[k + 8 * eleventh] = y8;
            output_buffer[k + 9 * eleventh] = y9;
            output_buffer[k + 10 * eleventh] = y10;
        }
}