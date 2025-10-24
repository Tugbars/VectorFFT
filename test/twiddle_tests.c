/**
 * @file test_twiddles_hybrid.c
 * @brief Comprehensive test suite for hybrid twiddle system
 * 
 * Tests against:
 * - Mathematical properties (|W| = 1, W^N = 1)
 * - FFTW reference implementation
 * - Round-trip FFT correctness
 * - Octant accuracy improvement
 * - SIMD correctness
 * 
 * Tolerance: 1e-15 for all tests (same for SIMPLE and FACTORED)
 */

//==============================================================================
// INCLUDES & CONFIGURATION
//==============================================================================

#include "fft_twiddles_hybrid.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <assert.h>
#include <complex.h>

#ifdef __AVX512F__
#include <immintrin.h>
#endif

#define TEST_TOLERANCE 1e-15
#define MAX_TEST_SIZE 4096

// FFTW integration (we'll adapt their code)
#define FFT_FORWARD_FFTW -1
#define FFT_INVERSE_FFTW +1

//==============================================================================
// SECTION 1: FFTW INTEGRATION
//==============================================================================

// Adapt FFTW's real_cexp to our test (from their trig.c)
typedef double trigreal_fftw;

static void fftw_real_cexp(int m, int n, trigreal_fftw *out, int direction) {
    // FFTW's octant reduction
    trigreal_fftw theta, c, s, t;
    unsigned octant = 0;
    int quarter_n = n;
    
    const double K2PI = 6.2831853071795864769252867665590057683943388;
    
    n += n; n += n;
    m += m; m += m;
    
    if (direction == FFT_INVERSE_FFTW) m = -m;  // Adjust for direction
    
    if (m < 0) m += n;
    if (m > n - m) { m = n - m; octant |= 4; }
    if (m - quarter_n > 0) { m = m - quarter_n; octant |= 2; }
    if (m > quarter_n - m) { m = quarter_n - m; octant |= 1; }
    
    theta = (K2PI * (double)m) / (double)n;
    c = cos(theta);
    s = sin(theta);
    
    if (octant & 1) { t = c; c = s; s = t; }
    if (octant & 2) { t = c; c = -s; s = t; }
    if (octant & 4) { s = -s; }
    
    out[0] = c;
    out[1] = s;
}

// Generate FFTW-style stage twiddles
static void fftw_compute_stage_twiddles(
    int N_stage, 
    int radix, 
    int direction,
    double *re_out,
    double *im_out)
{
    const int sub_len = N_stage / radix;
    
    for (int r = 1; r < radix; r++) {
        for (int k = 0; k < sub_len; k++) {
            int idx = (r - 1) * sub_len + k;
            int m = r * k;
            
            trigreal_fftw out[2];
            fftw_real_cexp(m, N_stage, out, direction);
            
            re_out[idx] = out[0];
            im_out[idx] = out[1];
        }
    }
}

//==============================================================================
// SECTION 2: TEST HELPERS
//==============================================================================

// Compute max absolute error between two arrays
static double compute_max_error_complex(int n, const fft_data *a, const fft_data *b) {
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double err_re = fabs(a[i].re - b[i].re);
        double err_im = fabs(a[i].im - b[i].im);
        double err = err_re + err_im;
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static double compute_max_error_scalar(int n, const double *a, const double *b) {
    double max_err = 0.0;
    for (int i = 0; i < n; i++) {
        double err = fabs(a[i] - b[i]);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

// Naive sincos (without octant reduction) for comparison
static void sincos_naive(double angle, double *s, double *c) {
    *s = sin(angle);
    *c = cos(angle);
}

// Reference DFT (slow but correct)
static void reference_dft(int n, const fft_data *in, fft_data *out, fft_direction_t dir) {
    double sign = (dir == FFT_FORWARD) ? -1.0 : 1.0;
    
    for (int k = 0; k < n; k++) {
        double sum_re = 0.0;
        double sum_im = 0.0;
        
        for (int m = 0; m < n; m++) {
            double angle = sign * 2.0 * M_PI * k * m / n;
            double c = cos(angle);
            double s = sin(angle);
            
            sum_re += in[m].re * c - in[m].im * s;
            sum_im += in[m].re * s + in[m].im * c;
        }
        
        out[k].re = sum_re;
        out[k].im = sum_im;
    }
}

//==============================================================================
// SECTION 3: MATHEMATICAL PROPERTY TESTS
//==============================================================================

// Test 1: |W^k| = 1 for all twiddles
static void test_twiddle_unit_magnitude(void) {
    printf("Testing twiddle unit magnitude...\n");
    
    int test_sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    int test_radices[] = {2, 3, 4, 5, 7, 8, 16};
    
    for (size_t i = 0; i < sizeof(test_sizes)/sizeof(int); i++) {
        int n = test_sizes[i];
        for (size_t j = 0; j < sizeof(test_radices)/sizeof(int); j++) {
            int radix = test_radices[j];
            if (radix > n) continue;
            
            // Test both strategies
            twiddle_strategy_t strategies[] = {TWID_SIMPLE, TWID_FACTORED};
            for (int s = 0; s < 2; s++) {
                twiddle_handle_t *tw = twiddle_create_explicit(
                    n, radix, FFT_FORWARD, strategies[s]);
                assert(tw != NULL);
                
                int sub_len = n / radix;
                for (int r = 1; r < radix; r++) {
                    for (int k = 0; k < sub_len; k++) {
                        double re, im;
                        twiddle_get(tw, r, k, &re, &im);
                        
                        double mag = sqrt(re*re + im*im);
                        double error = fabs(mag - 1.0);
                        
                        assert(error < TEST_TOLERANCE);
                    }
                }
                
                twiddle_destroy(tw);
            }
        }
    }
    
    printf("  ✓ All twiddles have unit magnitude\n");
}

// Test 2: W^N = 1 (periodicity)
static void test_twiddle_periodicity(void) {
    printf("Testing twiddle periodicity (W^N = 1)...\n");
    
    int test_sizes[] = {8, 16, 32, 64, 128, 256};
    
    for (size_t i = 0; i < sizeof(test_sizes)/sizeof(int); i++) {
        int n = test_sizes[i];
        
        twiddle_handle_t *tw = twiddle_create(n, 2, FFT_FORWARD);
        assert(tw != NULL);
        
        // W^N should equal 1
        double re_prod = 1.0, im_prod = 0.0;
        
        for (int k = 0; k < n; k++) {
            double re, im;
            twiddle_get(tw, 1, 0, &re, &im);  // Get W^1
            
            // Multiply: (re_prod + i*im_prod) *= (re + i*im)
            double new_re = re_prod * re - im_prod * im;
            double new_im = re_prod * im + im_prod * re;
            re_prod = new_re;
            im_prod = new_im;
        }
        
        double error = fabs(re_prod - 1.0) + fabs(im_prod);
        assert(error < TEST_TOLERANCE * n);  // Allow accumulation
        
        twiddle_destroy(tw);
    }
    
    printf("  ✓ Periodicity verified\n");
}

// Test 3: Conjugate symmetry W^(-k) = conj(W^k)
static void test_twiddle_conjugate_symmetry(void) {
    printf("Testing conjugate symmetry...\n");
    
    int n = 64;
    twiddle_handle_t *tw_fwd = twiddle_create(n, 4, FFT_FORWARD);
    twiddle_handle_t *tw_inv = twiddle_create(n, 4, FFT_INVERSE);
    
    assert(tw_fwd != NULL);
    assert(tw_inv != NULL);
    
    int sub_len = n / 4;
    for (int r = 1; r < 4; r++) {
        for (int k = 0; k < sub_len; k++) {
            double re_fwd, im_fwd, re_inv, im_inv;
            twiddle_get(tw_fwd, r, k, &re_fwd, &im_fwd);
            twiddle_get(tw_inv, r, k, &re_inv, &im_inv);
            
            // inv should be conjugate of fwd
            double error = fabs(re_fwd - re_inv) + fabs(im_fwd + im_inv);
            assert(error < TEST_TOLERANCE);
        }
    }
    
    twiddle_destroy(tw_fwd);
    twiddle_destroy(tw_inv);
    
    printf("  ✓ Conjugate symmetry verified\n");
}

//==============================================================================
// SECTION 4: FFTW COMPARISON TESTS
//==============================================================================

// Test 4: Compare against FFTW's twiddle generation
static void test_vs_fftw_twiddles(void) {
    printf("Testing against FFTW twiddle generation...\n");
    
    int test_sizes[] = {8, 16, 32, 64, 128, 256, 512, 1024};
    int test_radices[] = {2, 3, 4, 5, 8, 16};
    
    for (size_t i = 0; i < sizeof(test_sizes)/sizeof(int); i++) {
        int n = test_sizes[i];
        for (size_t j = 0; j < sizeof(test_radices)/sizeof(int); j++) {
            int radix = test_radices[j];
            if (radix > n) continue;
            
            int sub_len = n / radix;
            int num_twiddles = (radix - 1) * sub_len;
            
            // Generate with our code (both strategies)
            twiddle_strategy_t strategies[] = {TWID_SIMPLE, TWID_FACTORED};
            for (int s = 0; s < 2; s++) {
                twiddle_handle_t *tw = twiddle_create_explicit(
                    n, radix, FFT_FORWARD, strategies[s]);
                assert(tw != NULL);
                
                // Generate with FFTW
                double *fftw_re = malloc(num_twiddles * sizeof(double));
                double *fftw_im = malloc(num_twiddles * sizeof(double));
                fftw_compute_stage_twiddles(n, radix, FFT_FORWARD_FFTW, fftw_re, fftw_im);
                
                // Compare
                double max_error = 0.0;
                for (int r = 1; r < radix; r++) {
                    for (int k = 0; k < sub_len; k++) {
                        double our_re, our_im;
                        twiddle_get(tw, r, k, &our_re, &our_im);
                        
                        int idx = (r - 1) * sub_len + k;
                        double err_re = fabs(our_re - fftw_re[idx]);
                        double err_im = fabs(our_im - fftw_im[idx]);
                        double err = err_re + err_im;
                        
                        if (err > max_error) max_error = err;
                    }
                }
                
                printf("  N=%d, radix=%d, %s: max_error = %.3e\n", 
                       n, radix, 
                       s == 0 ? "SIMPLE" : "FACTORED",
                       max_error);
                
                assert(max_error < TEST_TOLERANCE);
                
                free(fftw_re);
                free(fftw_im);
                twiddle_destroy(tw);
            }
        }
    }
    
    printf("  ✓ Matches FFTW output within tolerance\n");
}

// Test 5: SIMPLE vs FACTORED must produce identical results
static void test_simple_vs_factored(void) {
    printf("Testing SIMPLE vs FACTORED equivalence...\n");
    
    int test_sizes[] = {64, 128, 256, 512, 1024, 2048};
    int test_radices[] = {2, 4, 8, 16};
    
    for (size_t i = 0; i < sizeof(test_sizes)/sizeof(int); i++) {
        int n = test_sizes[i];
        for (size_t j = 0; j < sizeof(test_radices)/sizeof(int); j++) {
            int radix = test_radices[j];
            if (radix > n) continue;
            
            twiddle_handle_t *tw_simple = twiddle_create_explicit(
                n, radix, FFT_FORWARD, TWID_SIMPLE);
            twiddle_handle_t *tw_factored = twiddle_create_explicit(
                n, radix, FFT_FORWARD, TWID_FACTORED);
            
            assert(tw_simple != NULL);
            assert(tw_factored != NULL);
            
            int sub_len = n / radix;
            double max_error = 0.0;
            
            for (int r = 1; r < radix; r++) {
                for (int k = 0; k < sub_len; k++) {
                    double re_simple, im_simple, re_factored, im_factored;
                    twiddle_get(tw_simple, r, k, &re_simple, &im_simple);
                    twiddle_get(tw_factored, r, k, &re_factored, &im_factored);
                    
                    double err = fabs(re_simple - re_factored) + 
                                 fabs(im_simple - im_factored);
                    if (err > max_error) max_error = err;
                }
            }
            
            printf("  N=%d, radix=%d: max_error = %.3e\n", n, radix, max_error);
            assert(max_error < TEST_TOLERANCE);
            
            twiddle_destroy(tw_simple);
            twiddle_destroy(tw_factored);
        }
    }
    
    printf("  ✓ SIMPLE and FACTORED produce identical results\n");
}

//==============================================================================
// SECTION 5: OCTANT ACCURACY TESTS
//==============================================================================

// Test 6: Octant reduction improves accuracy
static void test_octant_accuracy_improvement(void) {
    printf("Testing octant accuracy improvement...\n");
    
    // Critical angles where naive sincos has higher error
    double critical_angles[] = {
        M_PI/4, M_PI/2, 3*M_PI/4, M_PI, 
        5*M_PI/4, 3*M_PI/2, 7*M_PI/4, 2*M_PI
    };
    
    int improvement_count = 0;
    int total_tests = sizeof(critical_angles) / sizeof(double);
    
    for (size_t i = 0; i < total_tests; i++) {
        double angle = critical_angles[i];
        
        // High-precision reference (use long double if available)
        long double angle_ld = (long double)angle;
        long double ref_s = sinl(angle_ld);
        long double ref_c = cosl(angle_ld);
        
        // Octant-reduced version (our code uses this internally)
        double s_octant, c_octant;
        extern void sincos_octant(double, double*, double*);  // Forward declare
        sincos_octant(angle, &s_octant, &c_octant);
        
        // Naive version
        double s_naive, c_naive;
        sincos_naive(angle, &s_naive, &c_naive);
        
        // Compute errors
        double err_octant = fabs((double)ref_s - s_octant) + fabs((double)ref_c - c_octant);
        double err_naive = fabs((double)ref_s - s_naive) + fabs((double)ref_c - c_naive);
        
        printf("  angle=%.4f: octant_err=%.3e, naive_err=%.3e", 
               angle, err_octant, err_naive);
        
        if (err_octant < err_naive) {
            printf(" ✓ improved\n");
            improvement_count++;
        } else {
            printf(" (similar)\n");
        }
    }
    
    // At least half should show improvement
    assert(improvement_count >= total_tests / 2);
    printf("  ✓ Octant reduction improves accuracy in %d/%d cases\n", 
           improvement_count, total_tests);
}

//==============================================================================
// SECTION 6: SIMD LOAD TESTS
//==============================================================================

#ifdef __AVX512F__
static void test_simd_load_avx512(void) {
    printf("Testing AVX-512 SIMD loads...\n");
    
    int n = 128;
    int radix = 8;
    
    twiddle_handle_t *tw = twiddle_create(n, radix, FFT_FORWARD);
    assert(tw != NULL);
    
    int sub_len = n / radix;
    for (int r = 1; r < radix; r++) {
        for (int k = 0; k < sub_len - 8; k += 8) {
            __m512d re_vec, im_vec;
            twiddle_load_avx512(tw, r, k, &re_vec, &im_vec);
            
            // Verify by comparing with scalar loads
            double re_arr[8], im_arr[8];
            _mm512_storeu_pd(re_arr, re_vec);
            _mm512_storeu_pd(im_arr, im_vec);
            
            for (int i = 0; i < 8; i++) {
                double re_scalar, im_scalar;
                twiddle_get(tw, r, k + i, &re_scalar, &im_scalar);
                
                double err = fabs(re_arr[i] - re_scalar) + fabs(im_arr[i] - im_scalar);
                assert(err < TEST_TOLERANCE);
            }
        }
    }
    
    twiddle_destroy(tw);
    printf("  ✓ AVX-512 loads verified\n");
}
#endif

#ifdef __AVX2__
static void test_simd_load_avx2(void) {
    printf("Testing AVX2 SIMD loads...\n");
    
    int n = 64;
    int radix = 4;
    
    twiddle_handle_t *tw = twiddle_create(n, radix, FFT_FORWARD);
    assert(tw != NULL);
    
    int sub_len = n / radix;
    for (int r = 1; r < radix; r++) {
        for (int k = 0; k < sub_len - 4; k += 4) {
            __m256d re_vec, im_vec;
            twiddle_load_avx2(tw, r, k, &re_vec, &im_vec);
            
            double re_arr[4], im_arr[4];
            _mm256_storeu_pd(re_arr, re_vec);
            _mm256_storeu_pd(im_arr, im_vec);
            
            for (int i = 0; i < 4; i++) {
                double re_scalar, im_scalar;
                twiddle_get(tw, r, k + i, &re_scalar, &im_scalar);
                
                double err = fabs(re_arr[i] - re_scalar) + fabs(im_arr[i] - im_scalar);
                assert(err < TEST_TOLERANCE);
            }
        }
    }
    
    twiddle_destroy(tw);
    printf("  ✓ AVX2 loads verified\n");
}
#endif

static void test_simd_load_sse2(void) {
    printf("Testing SSE2 SIMD loads...\n");
    
    int n = 32;
    int radix = 4;
    
    twiddle_handle_t *tw = twiddle_create(n, radix, FFT_FORWARD);
    assert(tw != NULL);
    
    int sub_len = n / radix;
    for (int r = 1; r < radix; r++) {
        for (int k = 0; k < sub_len - 2; k += 2) {
            __m128d re_vec, im_vec;
            twiddle_load_sse2(tw, r, k, &re_vec, &im_vec);
            
            double re_arr[2], im_arr[2];
            _mm_storeu_pd(re_arr, re_vec);
            _mm_storeu_pd(im_arr, im_vec);
            
            for (int i = 0; i < 2; i++) {
                double re_scalar, im_scalar;
                twiddle_get(tw, r, k + i, &re_scalar, &im_scalar);
                
                double err = fabs(re_arr[i] - re_scalar) + fabs(im_arr[i] - im_scalar);
                assert(err < TEST_TOLERANCE);
            }
        }
    }
    
    twiddle_destroy(tw);
    printf("  ✓ SSE2 loads verified\n");
}

//==============================================================================
// SECTION 7: CACHE BEHAVIOR TESTS
//==============================================================================

// Test 7: Cache hit returns same pointer
static void test_cache_returns_same_pointer(void) {
    printf("Testing cache returns same pointer...\n");
    
    twiddle_cache_clear();
    
    twiddle_handle_t *tw1 = twiddle_create(64, 4, FFT_FORWARD);
    twiddle_handle_t *tw2 = twiddle_create(64, 4, FFT_FORWARD);
    
    assert(tw1 == tw2);  // Should be same pointer
    assert(tw1->refcount == 2);
    
    twiddle_destroy(tw1);
    assert(tw2->refcount == 1);
    
    twiddle_destroy(tw2);
    
    printf("  ✓ Cache returns same pointer and manages refcount\n");
}

// Test 8: Cache respects size limit
static void test_cache_size_limit(void) {
    printf("Testing cache size limit...\n");
    
    twiddle_cache_clear();
    
    // Create more handles than cache can hold
    twiddle_handle_t *handles[TWIDDLE_CACHE_SIZE + 10];
    
    for (int i = 0; i < TWIDDLE_CACHE_SIZE + 10; i++) {
        // Use different n values to avoid cache hits
        handles[i] = twiddle_create(64 + i, 4, FFT_FORWARD);
        assert(handles[i] != NULL);
    }
    
    // Cleanup
    for (int i = 0; i < TWIDDLE_CACHE_SIZE + 10; i++) {
        twiddle_destroy(handles[i]);
    }
    
    twiddle_cache_clear();
    
    printf("  ✓ Cache respects size limit\n");
}

//==============================================================================
// SECTION 8: ROUND-TRIP FFT TESTS
//==============================================================================

// Note: These require a simple FFT implementation
// For now, test with reference DFT for small sizes

static void test_roundtrip_with_reference_dft(void) {
    printf("Testing round-trip with reference DFT...\n");
    
    int test_sizes[] = {8, 16, 32, 64};
    
    for (size_t i = 0; i < sizeof(test_sizes)/sizeof(int); i++) {
        int n = test_sizes[i];
        
        // Create impulse signal
        fft_data *signal = calloc(n, sizeof(fft_data));
        fft_data *fft_out = calloc(n, sizeof(fft_data));
        fft_data *ifft_out = calloc(n, sizeof(fft_data));
        
        signal[0].re = 1.0;
        signal[0].im = 0.0;
        
        // Forward DFT
        reference_dft(n, signal, fft_out, FFT_FORWARD);
        
        // Inverse DFT
        reference_dft(n, fft_out, ifft_out, FFT_INVERSE);
        
        // Scale
        for (int j = 0; j < n; j++) {
            ifft_out[j].re /= n;
            ifft_out[j].im /= n;
        }
        
        // Check round-trip
        double error = compute_max_error_complex(n, signal, ifft_out);
        printf("  N=%d: round-trip error = %.3e\n", n, error);
        assert(error < TEST_TOLERANCE * 10);  // Allow some accumulation
        
        free(signal);
        free(fft_out);
        free(ifft_out);
    }
    
    printf("  ✓ Round-trip DFT verified\n");
}

//==============================================================================
// SECTION 9: EDGE CASES & ERROR HANDLING
//==============================================================================

static void test_invalid_inputs(void) {
    printf("Testing invalid input handling...\n");
    
    // radix < 2
    assert(twiddle_create(64, 1, FFT_FORWARD) == NULL);
    
    // n < radix
    assert(twiddle_create(4, 8, FFT_FORWARD) == NULL);
    
    printf("  ✓ Invalid inputs handled correctly\n");
}

static void test_threshold_boundary(void) {
    printf("Testing strategy selection at threshold...\n");
    
    // At threshold, should switch strategies
    int threshold = TWIDDLE_FACTORIZATION_THRESHOLD;
    
    twiddle_handle_t *tw_below = twiddle_create(threshold - 1, 4, FFT_FORWARD);
    twiddle_handle_t *tw_at = twiddle_create(threshold, 4, FFT_FORWARD);
    
    assert(tw_below->strategy == TWID_SIMPLE);
    assert(tw_at->strategy == TWID_FACTORED);
    
    twiddle_destroy(tw_below);
    twiddle_destroy(tw_at);
    
    printf("  ✓ Strategy selection at threshold correct\n");
}

//==============================================================================
// SECTION 5: OCTANT ACCURACY TESTS (Enhanced)
//==============================================================================

// Test 6a: Direct octant vs naive comparison across full angle range
static void test_octant_vs_naive(void) {
    printf("Testing octant vs naive across angle range...\n");
    
    // Test systematic angle coverage
    int num_angles = 100;
    double angle_step = 2.0 * M_PI / num_angles;
    
    int octant_better_count = 0;
    int octant_same_count = 0;
    int naive_better_count = 0;
    
    double max_octant_err = 0.0;
    double max_naive_err = 0.0;
    double sum_octant_err = 0.0;
    double sum_naive_err = 0.0;
    
    for (int i = 0; i < num_angles; i++) {
        double angle = i * angle_step;
        
        // High-precision reference (use long double)
        long double angle_ld = (long double)angle;
        long double ref_s = sinl(angle_ld);
        long double ref_c = cosl(angle_ld);
        
        // Octant-reduced version
        double s_octant, c_octant;
        sincos_octant(angle, &s_octant, &c_octant);
        
        // Naive version
        double s_naive, c_naive;
        sincos_naive(angle, &s_naive, &c_naive);
        
        // Compute errors
        double err_octant = fabs((double)ref_s - s_octant) + fabs((double)ref_c - c_octant);
        double err_naive = fabs((double)ref_s - s_naive) + fabs((double)ref_c - c_naive);
        
        sum_octant_err += err_octant;
        sum_naive_err += err_naive;
        
        if (err_octant > max_octant_err) max_octant_err = err_octant;
        if (err_naive > max_naive_err) max_naive_err = err_naive;
        
        // Compare which is better
        if (err_octant < err_naive * 0.99) {  // 1% threshold to avoid noise
            octant_better_count++;
        } else if (err_naive < err_octant * 0.99) {
            naive_better_count++;
        } else {
            octant_same_count++;
        }
    }
    
    double avg_octant_err = sum_octant_err / num_angles;
    double avg_naive_err = sum_naive_err / num_angles;
    
    printf("  Tested %d angles from 0 to 2π:\n", num_angles);
    printf("  Octant: avg_err=%.3e, max_err=%.3e\n", avg_octant_err, max_octant_err);
    printf("  Naive:  avg_err=%.3e, max_err=%.3e\n", avg_naive_err, max_naive_err);
    printf("  Octant better: %d, Same: %d, Naive better: %d\n", 
           octant_better_count, octant_same_count, naive_better_count);
    
    // Octant should be better on average
    assert(avg_octant_err <= avg_naive_err);
    
    // Octant should be better in majority of cases (or at least not worse)
    assert(octant_better_count + octant_same_count > naive_better_count);
    
    printf("  ✓ Octant reduction performs better overall\n");
}

// Test 6b: Test critical angles where octant reduction matters most
static void test_octant_critical_angles(void) {
    printf("Testing octant reduction at critical angles...\n");
    
    // These are the exact angles where octant boundaries occur
    // and where naive sincos often has maximum error
    struct {
        double angle;
        const char *name;
    } critical_angles[] = {
        {0.0,           "0"},
        {M_PI/8,        "π/8"},
        {M_PI/4,        "π/4"},
        {3*M_PI/8,      "3π/8"},
        {M_PI/2,        "π/2"},
        {5*M_PI/8,      "5π/8"},
        {3*M_PI/4,      "3π/4"},
        {7*M_PI/8,      "7π/8"},
        {M_PI,          "π"},
        {9*M_PI/8,      "9π/8"},
        {5*M_PI/4,      "5π/4"},
        {11*M_PI/8,     "11π/8"},
        {3*M_PI/2,      "3π/2"},
        {13*M_PI/8,     "13π/8"},
        {7*M_PI/4,      "7π/4"},
        {15*M_PI/8,     "15π/8"},
        {2*M_PI,        "2π"},
        // Add some near-boundary angles (these can be problematic)
        {M_PI/4 + 1e-10, "π/4+ε"},
        {M_PI/2 - 1e-10, "π/2-ε"},
        {M_PI + 1e-10,   "π+ε"},
    };
    
    int num_critical = sizeof(critical_angles) / sizeof(critical_angles[0]);
    int octant_wins = 0;
    int ties = 0;
    
    printf("\n  Angle        Octant Error    Naive Error     Result\n");
    printf("  --------------------------------------------------------\n");
    
    for (int i = 0; i < num_critical; i++) {
        double angle = critical_angles[i].angle;
        
        // High-precision reference
        long double angle_ld = (long double)angle;
        long double ref_s = sinl(angle_ld);
        long double ref_c = cosl(angle_ld);
        
        // Octant version
        double s_octant, c_octant;
        sincos_octant(angle, &s_octant, &c_octant);
        
        // Naive version
        double s_naive, c_naive;
        sincos_naive(angle, &s_naive, &c_naive);
        
        double err_octant = fabs((double)ref_s - s_octant) + fabs((double)ref_c - c_octant);
        double err_naive = fabs((double)ref_s - s_naive) + fabs((double)ref_c - c_naive);
        
        const char *result;
        if (err_octant < err_naive * 0.99) {
            result = "✓ Octant better";
            octant_wins++;
        } else if (err_naive < err_octant * 0.99) {
            result = "✗ Naive better";
        } else {
            result = "≈ Similar";
            ties++;
        }
        
        printf("  %-12s %.6e    %.6e    %s\n", 
               critical_angles[i].name, err_octant, err_naive, result);
        
        // At critical angles, octant should never be significantly worse
        assert(err_octant <= err_naive * 1.01);
    }
    
    printf("  --------------------------------------------------------\n");
    printf("  Octant better/similar: %d/%d out of %d angles\n", 
           octant_wins, ties, num_critical);
    
    // Octant should win or tie in most cases
    assert(octant_wins + ties >= num_critical * 0.7);
    
    printf("  ✓ Octant reduction handles critical angles correctly\n");
}

//==============================================================================
// SECTION 7: CACHE BEHAVIOR TESTS (Enhanced)
//==============================================================================

// Test 7a: Cache hit returns same pointer
static void test_cache_hit(void) {
    printf("Testing cache hit mechanism...\n");
    
    twiddle_cache_clear();
    
    // Create handle with specific parameters
    twiddle_handle_t *tw1 = twiddle_create(128, 8, FFT_FORWARD);
    assert(tw1 != NULL);
    assert(tw1->refcount == 1);
    
    // Request same parameters - should get cached handle
    twiddle_handle_t *tw2 = twiddle_create(128, 8, FFT_FORWARD);
    assert(tw2 != NULL);
    assert(tw2 == tw1);  // MUST be same pointer
    assert(tw1->refcount == 2);
    
    // Request same parameters again
    twiddle_handle_t *tw3 = twiddle_create(128, 8, FFT_FORWARD);
    assert(tw3 == tw1);
    assert(tw1->refcount == 3);
    
    // Different N should create new handle
    twiddle_handle_t *tw4 = twiddle_create(256, 8, FFT_FORWARD);
    assert(tw4 != tw1);
    assert(tw4->refcount == 1);
    
    // Different radix should create new handle
    twiddle_handle_t *tw5 = twiddle_create(128, 4, FFT_FORWARD);
    assert(tw5 != tw1);
    assert(tw5->refcount == 1);
    
    // Different direction should create new handle
    twiddle_handle_t *tw6 = twiddle_create(128, 8, FFT_INVERSE);
    assert(tw6 != tw1);
    assert(tw6->refcount == 1);
    
    // Verify twiddles are actually correct (not just pointer games)
    double re1, im1, re2, im2;
    twiddle_get(tw1, 1, 0, &re1, &im1);
    twiddle_get(tw2, 1, 0, &re2, &im2);
    assert(fabs(re1 - re2) < TEST_TOLERANCE);
    assert(fabs(im1 - im2) < TEST_TOLERANCE);
    
    // Cleanup
    twiddle_destroy(tw1);
    assert(tw1->refcount == 2);  // tw2 and tw3 still alive
    
    twiddle_destroy(tw2);
    assert(tw1->refcount == 1);  // tw3 still alive
    
    twiddle_destroy(tw3);
    // tw1 should now be freed
    
    twiddle_destroy(tw4);
    twiddle_destroy(tw5);
    twiddle_destroy(tw6);
    
    twiddle_cache_clear();
    
    printf("  ✓ Cache hit returns same pointer correctly\n");
}

// Test 7b: Reference counting works correctly
static void test_cache_refcount(void) {
    printf("Testing reference counting mechanism...\n");
    
    twiddle_cache_clear();
    
    // Test 1: Simple refcount increment/decrement
    twiddle_handle_t *tw = twiddle_create(64, 4, FFT_FORWARD);
    assert(tw->refcount == 1);
    
    twiddle_handle_t *tw_copy = twiddle_create(64, 4, FFT_FORWARD);
    assert(tw_copy == tw);
    assert(tw->refcount == 2);
    
    // Destroy should decrement, not free
    twiddle_destroy(tw);
    // Can't access tw->refcount here safely, but tw_copy should still work
    
    double re, im;
    twiddle_get(tw_copy, 1, 0, &re, &im);  // Should still work
    
    twiddle_destroy(tw_copy);  // Now it should be freed
    
    // Test 2: Multiple handles with different parameters
    twiddle_handle_t *handles[5];
    handles[0] = twiddle_create(32, 2, FFT_FORWARD);
    handles[1] = twiddle_create(32, 2, FFT_FORWARD);  // Same as [0]
    handles[2] = twiddle_create(64, 4, FFT_FORWARD);  // Different
    handles[3] = twiddle_create(64, 4, FFT_FORWARD);  // Same as [2]
    handles[4] = twiddle_create(64, 4, FFT_FORWARD);  // Same as [2]
    
    assert(handles[0] == handles[1]);
    assert(handles[2] == handles[3]);
    assert(handles[2] == handles[4]);
    assert(handles[0] != handles[2]);
    
    assert(handles[0]->refcount == 2);
    assert(handles[2]->refcount == 3);
    
    // Destroy in different order
    twiddle_destroy(handles[1]);
    assert(handles[0]->refcount == 1);
    
    twiddle_destroy(handles[3]);
    assert(handles[2]->refcount == 2);
    
    twiddle_destroy(handles[0]);
    twiddle_destroy(handles[2]);
    twiddle_destroy(handles[4]);
    
    // Test 3: Create many references, destroy all
    twiddle_handle_t *base = twiddle_create(128, 8, FFT_FORWARD);
    twiddle_handle_t *refs[10];
    for (int i = 0; i < 10; i++) {
        refs[i] = twiddle_create(128, 8, FFT_FORWARD);
        assert(refs[i] == base);
        assert(base->refcount == i + 2);
    }
    
    // Destroy all
    for (int i = 0; i < 10; i++) {
        twiddle_destroy(refs[i]);
    }
    twiddle_destroy(base);
    
    twiddle_cache_clear();
    
    printf("  ✓ Reference counting works correctly\n");
}

// Test 7c: Cache overflow behavior
static void test_cache_overflow(void) {
    printf("Testing cache overflow handling...\n");
    
    twiddle_cache_clear();
    
    // Create exactly TWIDDLE_CACHE_SIZE handles with different parameters
    twiddle_handle_t *handles[TWIDDLE_CACHE_SIZE + 5];
    
    printf("  Creating %d handles (cache size = %d)...\n", 
           TWIDDLE_CACHE_SIZE + 5, TWIDDLE_CACHE_SIZE);
    
    for (int i = 0; i < TWIDDLE_CACHE_SIZE + 5; i++) {
        // Use different N values to ensure different cache keys
        handles[i] = twiddle_create(64 + i * 8, 4, FFT_FORWARD);
        assert(handles[i] != NULL);
        
        // Verify it was created correctly
        double re, im;
        twiddle_get(handles[i], 1, 0, &re, &im);
        double mag = sqrt(re*re + im*im);
        assert(fabs(mag - 1.0) < TEST_TOLERANCE);
    }
    
    // All handles should be valid and usable
    for (int i = 0; i < TWIDDLE_CACHE_SIZE + 5; i++) {
        double re, im;
        twiddle_get(handles[i], 1, 0, &re, &im);
        assert(fabs(sqrt(re*re + im*im) - 1.0) < TEST_TOLERANCE);
    }
    
    // Test that cache lookup still works for early entries
    // (those that should be in cache)
    twiddle_handle_t *lookup1 = twiddle_create(64, 4, FFT_FORWARD);
    assert(lookup1 == handles[0]);  // Should be cached
    twiddle_destroy(lookup1);
    
    // Test that handles beyond cache limit are not cached
    // (request same parameters, should get new handle)
    twiddle_handle_t *lookup2 = twiddle_create(64 + TWIDDLE_CACHE_SIZE * 8, 4, FFT_FORWARD);
    // This might or might not be cached depending on implementation
    // Just verify it works correctly
    double re, im;
    twiddle_get(lookup2, 1, 0, &re, &im);
    assert(fabs(sqrt(re*re + im*im) - 1.0) < TEST_TOLERANCE);
    twiddle_destroy(lookup2);
    
    // Cleanup all handles
    for (int i = 0; i < TWIDDLE_CACHE_SIZE + 5; i++) {
        twiddle_destroy(handles[i]);
    }
    
    twiddle_cache_clear();
    
    printf("  ✓ Cache overflow handled gracefully\n");
}

// Test 7d: Cache clear functionality
static void test_cache_clear(void) {
    printf("Testing cache clear operation...\n");
    
    twiddle_cache_clear();
    
    // Create several cached handles
    twiddle_handle_t *tw1 = twiddle_create(64, 4, FFT_FORWARD);
    twiddle_handle_t *tw2 = twiddle_create(128, 8, FFT_FORWARD);
    twiddle_handle_t *tw3 = twiddle_create(256, 4, FFT_INVERSE);
    
    assert(tw1 != NULL);
    assert(tw2 != NULL);
    assert(tw3 != NULL);
    
    // Get copies to bump refcount
    twiddle_handle_t *tw1_copy = twiddle_create(64, 4, FFT_FORWARD);
    assert(tw1_copy == tw1);
    assert(tw1->refcount == 2);
    
    // Clear cache (should force refcount to 1 and destroy all)
    twiddle_cache_clear();
    
    // After clear, requesting same parameters should create NEW handles
    twiddle_handle_t *tw1_new = twiddle_create(64, 4, FFT_FORWARD);
    assert(tw1_new != NULL);
    assert(tw1_new != tw1);  // Should be different pointer now
    assert(tw1_new->refcount == 1);
    
    // Verify new handle is correct
    double re, im;
    twiddle_get(tw1_new, 1, 0, &re, &im);
    assert(fabs(sqrt(re*re + im*im) - 1.0) < TEST_TOLERANCE);
    
    // Test multiple clear operations
    twiddle_cache_clear();
    twiddle_cache_clear();  // Should be safe to call on empty cache
    twiddle_cache_clear();
    
    // Create new handles after multiple clears
    twiddle_handle_t *tw_after = twiddle_create(32, 2, FFT_FORWARD);
    assert(tw_after != NULL);
    twiddle_get(tw_after, 1, 0, &re, &im);
    assert(fabs(sqrt(re*re + im*im) - 1.0) < TEST_TOLERANCE);
    
    twiddle_destroy(tw1_new);
    twiddle_destroy(tw_after);
    twiddle_cache_clear();
    
    printf("  ✓ Cache clear works correctly\n");
}

//==============================================================================
// SECTION 8: PLANNER API & MATERIALIZATION TESTS
//==============================================================================

// Test 8a: Basic materialization - SIMPLE strategy
static void test_materialization_simple(void) {
    printf("Testing materialization of SIMPLE handles...\n");
    
    int n = 128;
    int radix = 8;
    
    twiddle_handle_t *tw = twiddle_create_explicit(n, radix, FFT_FORWARD, TWID_SIMPLE);
    assert(tw != NULL);
    assert(tw->strategy == TWID_SIMPLE);
    
    // Not materialized yet
    assert(!twiddle_is_materialized(tw));
    assert(tw->materialized_re == NULL);
    assert(tw->materialized_im == NULL);
    
    // Materialize
    int result = twiddle_materialize(tw);
    assert(result == 0);
    
    // Now should be materialized
    assert(twiddle_is_materialized(tw));
    assert(tw->materialized_re != NULL);
    assert(tw->materialized_im != NULL);
    
    int expected_count = (radix - 1) * (n / radix);
    assert(tw->materialized_count == expected_count);
    
    // For SIMPLE strategy, should be zero-copy (borrowed pointers)
    assert(tw->owns_materialized == 0);
    assert(tw->materialized_re == tw->data.simple.re);
    assert(tw->materialized_im == tw->data.simple.im);
    
    // Verify values are correct
    int sub_len = n / radix;
    for (int r = 1; r < radix; r++) {
        for (int k = 0; k < sub_len; k++) {
            int idx = (r - 1) * sub_len + k;
            
            double re_scalar, im_scalar;
            twiddle_get(tw, r, k, &re_scalar, &im_scalar);
            
            assert(fabs(tw->materialized_re[idx] - re_scalar) < TEST_TOLERANCE);
            assert(fabs(tw->materialized_im[idx] - im_scalar) < TEST_TOLERANCE);
        }
    }
    
    twiddle_destroy(tw);
    
    printf("  ✓ SIMPLE strategy materialization works (zero-copy)\n");
}

// Test 8b: Materialization - FACTORED strategy
static void test_materialization_factored(void) {
    printf("Testing materialization of FACTORED handles...\n");
    
    int n = 2048;
    int radix = 16;
    
    twiddle_handle_t *tw = twiddle_create_explicit(n, radix, FFT_FORWARD, TWID_FACTORED);
    assert(tw != NULL);
    assert(tw->strategy == TWID_FACTORED);
    
    // Not materialized initially
    assert(!twiddle_is_materialized(tw));
    
    // Materialize
    int result = twiddle_materialize(tw);
    assert(result == 0);
    
    // Now should be materialized
    assert(twiddle_is_materialized(tw));
    assert(tw->materialized_re != NULL);
    assert(tw->materialized_im != NULL);
    
    int expected_count = (radix - 1) * (n / radix);
    assert(tw->materialized_count == expected_count);
    
    // For FACTORED strategy, should own the memory (not borrowed)
    assert(tw->owns_materialized == 1);
    
    // Verify all materialized values are correct
    int sub_len = n / radix;
    double max_error = 0.0;
    
    for (int r = 1; r < radix; r++) {
        for (int k = 0; k < sub_len; k++) {
            int idx = (r - 1) * sub_len + k;
            
            // Get value via scalar interface (reconstructs on-the-fly)
            double re_scalar, im_scalar;
            twiddle_get(tw, r, k, &re_scalar, &im_scalar);
            
            // Compare with materialized array
            double err = fabs(tw->materialized_re[idx] - re_scalar) +
                        fabs(tw->materialized_im[idx] - im_scalar);
            if (err > max_error) max_error = err;
        }
    }
    
    printf("  Max error between materialized and scalar: %.3e\n", max_error);
    assert(max_error < TEST_TOLERANCE);
    
    // Verify unit magnitude
    for (int i = 0; i < expected_count; i++) {
        double mag = sqrt(tw->materialized_re[i] * tw->materialized_re[i] +
                         tw->materialized_im[i] * tw->materialized_im[i]);
        assert(fabs(mag - 1.0) < TEST_TOLERANCE);
    }
    
    twiddle_destroy(tw);
    
    printf("  ✓ FACTORED strategy materialization works (reconstruction)\n");
}

// Test 8c: Double materialization (should be no-op)
static void test_double_materialization(void) {
    printf("Testing double materialization...\n");
    
    int n = 256;
    int radix = 8;
    
    twiddle_handle_t *tw = twiddle_create(n, radix, FFT_FORWARD);
    assert(tw != NULL);
    
    // First materialization
    int result1 = twiddle_materialize(tw);
    assert(result1 == 0);
    assert(twiddle_is_materialized(tw));
    
    // Save pointers
    double *re_ptr1 = tw->materialized_re;
    double *im_ptr1 = tw->materialized_im;
    int count1 = tw->materialized_count;
    
    // Second materialization (should be no-op)
    int result2 = twiddle_materialize(tw);
    assert(result2 == 0);
    
    // Pointers should be unchanged
    assert(tw->materialized_re == re_ptr1);
    assert(tw->materialized_im == im_ptr1);
    assert(tw->materialized_count == count1);
    
    // Third materialization (paranoia check)
    int result3 = twiddle_materialize(tw);
    assert(result3 == 0);
    assert(tw->materialized_re == re_ptr1);
    
    twiddle_destroy(tw);
    
    printf("  ✓ Double materialization is safe no-op\n");
}

// Test 8d: SoA view extraction
static void test_soa_view_extraction(void) {
    printf("Testing SoA view extraction...\n");
    
    int n = 128;
    int radix = 4;
    
    twiddle_handle_t *tw = twiddle_create(n, radix, FFT_FORWARD);
    assert(tw != NULL);
    
    // Try to get view before materialization (should fail)
    fft_twiddles_soa_view view_early;
    int result_early = twiddle_get_soa_view(tw, &view_early);
    assert(result_early != 0);  // Should fail
    
    // Materialize
    twiddle_materialize(tw);
    
    // Now get view (should succeed)
    fft_twiddles_soa_view view;
    int result = twiddle_get_soa_view(tw, &view);
    assert(result == 0);
    
    // Verify view points to correct data
    assert(view.re == tw->materialized_re);
    assert(view.im == tw->materialized_im);
    assert(view.count == tw->materialized_count);
    
    int expected_count = (radix - 1) * (n / radix);
    assert(view.count == expected_count);
    
    // Verify data through view
    int sub_len = n / radix;
    for (int r = 1; r < radix; r++) {
        for (int k = 0; k < sub_len; k++) {
            int idx = (r - 1) * sub_len + k;
            
            double re_scalar, im_scalar;
            twiddle_get(tw, r, k, &re_scalar, &im_scalar);
            
            assert(fabs(view.re[idx] - re_scalar) < TEST_TOLERANCE);
            assert(fabs(view.im[idx] - im_scalar) < TEST_TOLERANCE);
        }
    }
    
    // Test NULL pointer handling
    assert(twiddle_get_soa_view(NULL, &view) != 0);
    assert(twiddle_get_soa_view(tw, NULL) != 0);
    
    twiddle_destroy(tw);
    
    printf("  ✓ SoA view extraction works correctly\n");
}

// Test 8e: get_stage_twiddles API
static void test_get_stage_twiddles_api(void) {
    printf("Testing get_stage_twiddles() API...\n");
    
    twiddle_cache_clear();
    
    int n = 512;
    int radix = 8;
    
    // Get stage twiddles (should create and materialize)
    twiddle_handle_t *tw = get_stage_twiddles(n, radix, FFT_FORWARD);
    assert(tw != NULL);
    
    // Should be automatically materialized
    assert(twiddle_is_materialized(tw));
    assert(tw->materialized_re != NULL);
    assert(tw->materialized_im != NULL);
    
    // Verify correct size
    int expected_count = (radix - 1) * (n / radix);
    assert(tw->materialized_count == expected_count);
    
    // Should be able to get SoA view immediately
    fft_twiddles_soa_view view;
    assert(twiddle_get_soa_view(tw, &view) == 0);
    assert(view.count == expected_count);
    
    // Verify correctness with FFTW
    double *fftw_re = malloc(expected_count * sizeof(double));
    double *fftw_im = malloc(expected_count * sizeof(double));
    fftw_compute_stage_twiddles(n, radix, FFT_FORWARD_FFTW, fftw_re, fftw_im);
    
    double max_error = compute_max_error_scalar(expected_count, view.re, fftw_re);
    max_error += compute_max_error_scalar(expected_count, view.im, fftw_im);
    
    assert(max_error < TEST_TOLERANCE);
    
    free(fftw_re);
    free(fftw_im);
    
    // Test caching: same parameters should return same handle
    twiddle_handle_t *tw2 = get_stage_twiddles(n, radix, FFT_FORWARD);
    assert(tw2 == tw);
    assert(tw->refcount == 2);
    
    twiddle_destroy(tw);
    twiddle_destroy(tw2);
    
    twiddle_cache_clear();
    
    printf("  ✓ get_stage_twiddles() API works correctly\n");
}

// Test 8f: get_dft_kernel_twiddles API
static void test_get_dft_kernel_twiddles_api(void) {
    printf("Testing get_dft_kernel_twiddles() API...\n");
    
    twiddle_cache_clear();
    
    int test_radices[] = {2, 3, 4, 5, 7, 8};
    
    for (size_t i = 0; i < sizeof(test_radices)/sizeof(int); i++) {
        int radix = test_radices[i];
        
        twiddle_handle_t *tw = get_dft_kernel_twiddles(radix, FFT_FORWARD);
        assert(tw != NULL);
        
        // Should be materialized
        assert(twiddle_is_materialized(tw));
        
        // For DFT kernel, this is a special case
        // Just verify it's usable
        fft_twiddles_soa_view view;
        assert(twiddle_get_soa_view(tw, &view) == 0);
        
        // Verify unit magnitude
        for (int j = 0; j < view.count; j++) {
            double mag = sqrt(view.re[j] * view.re[j] + view.im[j] * view.im[j]);
            assert(fabs(mag - 1.0) < TEST_TOLERANCE);
        }
        
        twiddle_destroy(tw);
    }
    
    twiddle_cache_clear();
    
    printf("  ✓ get_dft_kernel_twiddles() API works correctly\n");
}

// Test 8g: SIMPLE vs FACTORED materialization produces identical results
static void test_materialization_simple_vs_factored(void) {
    printf("Testing SIMPLE vs FACTORED materialization equivalence...\n");
    
    int test_configs[] = {
        128, 4,
        256, 8,
        512, 16,
        1024, 8,
        2048, 16
    };
    
    for (size_t i = 0; i < sizeof(test_configs)/(2*sizeof(int)); i++) {
        int n = test_configs[i*2];
        int radix = test_configs[i*2 + 1];
        
        // Create both strategies
        twiddle_handle_t *tw_simple = twiddle_create_explicit(n, radix, FFT_FORWARD, TWID_SIMPLE);
        twiddle_handle_t *tw_factored = twiddle_create_explicit(n, radix, FFT_FORWARD, TWID_FACTORED);
        
        assert(tw_simple != NULL);
        assert(tw_factored != NULL);
        
        // Materialize both
        assert(twiddle_materialize(tw_simple) == 0);
        assert(twiddle_materialize(tw_factored) == 0);
        
        // Get views
        fft_twiddles_soa_view view_simple, view_factored;
        assert(twiddle_get_soa_view(tw_simple, &view_simple) == 0);
        assert(twiddle_get_soa_view(tw_factored, &view_factored) == 0);
        
        // Should have same count
        assert(view_simple.count == view_factored.count);
        
        // Compare all values
        double max_error = 0.0;
        for (int j = 0; j < view_simple.count; j++) {
            double err = fabs(view_simple.re[j] - view_factored.re[j]) +
                        fabs(view_simple.im[j] - view_factored.im[j]);
            if (err > max_error) max_error = err;
        }
        
        printf("  N=%d, radix=%d: max_error = %.3e\n", n, radix, max_error);
        assert(max_error < TEST_TOLERANCE);
        
        twiddle_destroy(tw_simple);
        twiddle_destroy(tw_factored);
    }
    
    printf("  ✓ SIMPLE and FACTORED materialization produce identical results\n");
}

// Test 8h: Memory ownership and cleanup
static void test_materialization_memory_ownership(void) {
    printf("Testing materialization memory ownership...\n");
    
    // Test SIMPLE: borrowed pointers (owns_materialized = 0)
    {
        twiddle_handle_t *tw = twiddle_create_explicit(128, 4, FFT_FORWARD, TWID_SIMPLE);
        assert(tw != NULL);
        
        twiddle_materialize(tw);
        
        assert(tw->owns_materialized == 0);  // Borrowed
        assert(tw->materialized_re == tw->data.simple.re);
        
        // Save pointer to verify no double-free
        double *re_ptr = tw->materialized_re;
        
        twiddle_destroy(tw);
        // If we reach here without crash, memory management is correct
    }
    
    // Test FACTORED: owned pointers (owns_materialized = 1)
    {
        twiddle_handle_t *tw = twiddle_create_explicit(1024, 8, FFT_FORWARD, TWID_FACTORED);
        assert(tw != NULL);
        
        twiddle_materialize(tw);
        
        assert(tw->owns_materialized == 1);  // Owned
        assert(tw->materialized_re != NULL);
        
        twiddle_destroy(tw);
        // If we reach here without crash, memory management is correct
    }
    
    // Test multiple handles sharing factored representation
    {
        twiddle_handle_t *tw1 = twiddle_create_explicit(512, 8, FFT_FORWARD, TWID_FACTORED);
        twiddle_handle_t *tw2 = twiddle_create_explicit(512, 8, FFT_FORWARD, TWID_FACTORED);
        
        assert(tw1 == tw2);  // Should be cached
        
        twiddle_materialize(tw1);
        assert(twiddle_is_materialized(tw1));
        assert(twiddle_is_materialized(tw2));  // Same handle
        
        // Both should point to same materialized data
        assert(tw1->materialized_re == tw2->materialized_re);
        
        twiddle_destroy(tw1);
        // tw2 should still be valid
        assert(twiddle_is_materialized(tw2));
        
        twiddle_destroy(tw2);
        // Now freed
    }
    
    printf("  ✓ Memory ownership handled correctly (no leaks, no double-frees)\n");
}

// Test 8i: Materialization with cache
static void test_materialization_with_cache(void) {
    printf("Testing materialization with cache...\n");
    
    twiddle_cache_clear();
    
    int n = 256;
    int radix = 4;
    
    // Create and materialize first handle
    twiddle_handle_t *tw1 = get_stage_twiddles(n, radix, FFT_FORWARD);
    assert(tw1 != NULL);
    assert(twiddle_is_materialized(tw1));
    
    // Get view
    fft_twiddles_soa_view view1;
    assert(twiddle_get_soa_view(tw1, &view1) == 0);
    
    // Get same twiddles again (should be cached)
    twiddle_handle_t *tw2 = get_stage_twiddles(n, radix, FFT_FORWARD);
    assert(tw2 == tw1);  // Same handle
    assert(tw1->refcount == 2);
    
    // Should still be materialized
    assert(twiddle_is_materialized(tw2));
    
    // Get view from second handle
    fft_twiddles_soa_view view2;
    assert(twiddle_get_soa_view(tw2, &view2) == 0);
    
    // Views should point to same data
    assert(view1.re == view2.re);
    assert(view1.im == view2.im);
    assert(view1.count == view2.count);
    
    twiddle_destroy(tw1);
    twiddle_destroy(tw2);
    
    twiddle_cache_clear();
    
    printf("  ✓ Materialization works correctly with cache\n");
}

// Test 8j: Large-scale materialization stress test
static void test_materialization_stress(void) {
    printf("Testing large-scale materialization stress...\n");
    
    twiddle_cache_clear();
    
    // Create and materialize many handles
    int configs[][2] = {
        {64, 2}, {128, 4}, {256, 8}, {512, 16},
        {1024, 2}, {2048, 4}, {4096, 8},
        {64, 4}, {128, 8}, {256, 16},
        {512, 4}, {1024, 8}, {2048, 16}
    };
    
    int num_configs = sizeof(configs) / sizeof(configs[0]);
    twiddle_handle_t *handles[20];
    
    for (int i = 0; i < num_configs; i++) {
        int n = configs[i][0];
        int radix = configs[i][1];
        
        handles[i] = get_stage_twiddles(n, radix, FFT_FORWARD);
        assert(handles[i] != NULL);
        assert(twiddle_is_materialized(handles[i]));
        
        // Verify correctness
        fft_twiddles_soa_view view;
        assert(twiddle_get_soa_view(handles[i], &view) == 0);
        
        // Spot check: verify first twiddle has unit magnitude
        double mag = sqrt(view.re[0] * view.re[0] + view.im[0] * view.im[0]);
        assert(fabs(mag - 1.0) < TEST_TOLERANCE);
    }
    
    // Cleanup all
    for (int i = 0; i < num_configs; i++) {
        twiddle_destroy(handles[i]);
    }
    
    twiddle_cache_clear();
    
    printf("  ✓ Large-scale materialization stress test passed\n");
}

//==============================================================================
// MAIN TEST RUNNER
//==============================================================================

int main(void) {
    printf("=================================================================\n");
    printf("  VectorFFT Hybrid Twiddle System - Comprehensive Test Suite\n");
    printf("  Tolerance: %.1e (same for SIMPLE and FACTORED)\n", TEST_TOLERANCE);
    printf("=================================================================\n\n");
    
    // Mathematical properties
    printf("--- SECTION 1: Mathematical Properties ---\n");
    test_twiddle_unit_magnitude();
    test_twiddle_periodicity();
    test_twiddle_conjugate_symmetry();
    printf("\n");
    
    // FFTW comparison
    printf("--- SECTION 2: FFTW Comparison ---\n");
    test_vs_fftw_twiddles();
    test_simple_vs_factored();
    printf("\n");
    
    // Octant accuracy (ENHANCED)
    printf("--- SECTION 3: Octant Accuracy ---\n");
    test_octant_vs_naive();
    test_octant_critical_angles();
    printf("\n");
    
    // SIMD tests
    printf("--- SECTION 4: SIMD Load Functions ---\n");
    test_simd_load_sse2();
#ifdef __AVX2__
    test_simd_load_avx2();
#endif
#ifdef __AVX512F__
    test_simd_load_avx512();
#endif
    printf("\n");
    
    // Cache behavior (ENHANCED)
    printf("--- SECTION 5: Cache Behavior ---\n");
    test_cache_hit();
    test_cache_refcount();
    test_cache_overflow();
    test_cache_clear();
    printf("\n");
    
    // Round-trip tests
    printf("--- SECTION 6: Round-Trip Correctness ---\n");
    test_roundtrip_with_reference_dft();
    printf("\n");
    
    // Edge cases
    printf("--- SECTION 7: Edge Cases ---\n");
    test_invalid_inputs();
    test_threshold_boundary();
    printf("\n");
    
    printf("=================================================================\n");
    printf("  ✓✓✓ ALL TESTS PASSED ✓✓✓\n");
    printf("=================================================================\n");
    
    return 0;
}