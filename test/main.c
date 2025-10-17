/**
 * @file comprehensive_complex_fft_tests.c
 * @brief Thorough test suite for complex FFT (fft_exec) accuracy
 * 
 * Tests the core complex FFT engine independently of R2C/C2R wrappers
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include "highspeedFFT.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// High-resolution timing (platform-specific)
#ifdef _WIN32
#include <windows.h>
static double get_time_ms(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}
#else
#include <sys/time.h>
static double get_time_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}
#endif

// Helper to compute complex magnitude
static inline double cmag(fft_data z) {
    return sqrt(z.re * z.re + z.im * z.im);
}

// Helper to compute MSE between two complex arrays
static double compute_complex_mse(fft_data *a, fft_data *b, int n) {
    double mse = 0.0;
    for (int i = 0; i < n; i++) {
        double dr = a[i].re - b[i].re;
        double di = a[i].im - b[i].im;
        mse += dr * dr + di * di;
    }
    return mse / n;
}

// ============================================================================
// TEST 1: Complex Impulse
// ============================================================================
int test_complex_impulse(void) {
    printf("\n=== TEST: Complex Impulse δ[0] ===\n");
    const int N = 16;
    const double tol = 1e-10;
    int pass = 1;
    
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Impulse at k=0
    for (int i = 0; i < N; i++) {
        input[i].re = (i == 0) ? 1.0 : 0.0;
        input[i].im = 0.0;
    }
    
    fft_object fft = fft_init(N, 1);
    fft_exec(fft, input, output);
    
    // Expected: All bins = 1.0 + 0i
    for (int k = 0; k < N; k++) {
        double mag = cmag(output[k]);
        if (fabs(mag - 1.0) > tol) {
            printf("  FAIL at k=%d: magnitude %.6f (expected 1.0)\n", k, mag);
            pass = 0;
        }
        if (fabs(output[k].im) > tol) {
            printf("  FAIL at k=%d: imaginary %.6f (expected 0.0)\n", k, output[k].im);
            pass = 0;
        }
    }
    
    if (pass) {
        printf("  PASS: All bins = 1.0 + 0i\n");
    }
    
    free_fft(fft);
    free(input);
    free(output);
    return pass;
}

// ============================================================================
// TEST 2: Complex DC (Constant Signal)
// ============================================================================
int test_complex_dc(void) {
    printf("\n=== TEST: Complex DC (All Ones) ===\n");
    const int N = 16;
    const double tol = 1e-10;
    int pass = 1;
    
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // All ones
    for (int i = 0; i < N; i++) {
        input[i].re = 1.0;
        input[i].im = 0.0;
    }
    
    fft_object fft = fft_init(N, 1);
    fft_exec(fft, input, output);
    
    // Expected: X[0] = N, X[k] = 0 for k > 0
    if (fabs(output[0].re - N) > tol || fabs(output[0].im) > tol) {
        printf("  FAIL at k=0: got (%.6f, %.6f), expected (%d, 0)\n",
               output[0].re, output[0].im, N);
        pass = 0;
    } else {
        printf("  PASS: DC bin = %d\n", N);
    }
    
    for (int k = 1; k < N; k++) {
        double mag = cmag(output[k]);
        if (mag > tol) {
            printf("  FAIL at k=%d: magnitude %.6e (expected 0)\n", k, mag);
            pass = 0;
        }
    }
    
    if (pass) {
        printf("  PASS: All non-DC bins ≈ 0\n");
    }
    
    free_fft(fft);
    free(input);
    free(output);
    return pass;
}

// ============================================================================
// TEST 3: Complex Exponential (Single Frequency)
// ============================================================================
int test_complex_exponential(void) {
    printf("\n=== TEST: Complex Exponential e^{i·2π·k0·n/N} ===\n");
    const int N = 32;
    const int k0 = 5; // Target frequency bin
    const double tol = 1e-9;
    int pass = 1;
    
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Generate e^{i·2π·k0·n/N}
    for (int n = 0; n < N; n++) {
        double angle = 2.0 * M_PI * k0 * n / N;
        input[n].re = cos(angle);
        input[n].im = sin(angle);
    }
    
    fft_object fft = fft_init(N, 1);
    fft_exec(fft, input, output);
    
    // Expected: X[k0] = N, all others = 0
    for (int k = 0; k < N; k++) {
        if (k == k0) {
            if (fabs(output[k].re - N) > tol || fabs(output[k].im) > tol) {
                printf("  FAIL at k=%d: got (%.6f, %.6f), expected (%d, 0)\n",
                       k, output[k].re, output[k].im, N);
                pass = 0;
            }
        } else {
            double mag = cmag(output[k]);
            if (mag > tol) {
                printf("  FAIL at k=%d: magnitude %.6e (expected 0)\n", k, mag);
                pass = 0;
            }
        }
    }
    
    if (pass) {
        printf("  PASS: Single spike at k=%d with amplitude %d\n", k0, N);
    }
    
    free_fft(fft);
    free(input);
    free(output);
    return pass;
}

// ============================================================================
// TEST 4: Parseval's Theorem for Complex FFT
// ============================================================================
int test_complex_parseval(void) {
    printf("\n=== TEST: Parseval's Theorem (Complex) ===\n");
    const int N = 32;
    const double tol = 1e-6;
    int pass = 1;
    
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Generate random complex signal
    for (int i = 0; i < N; i++) {
        input[i].re = cos(2.0 * M_PI * 3 * i / N) + 0.5 * sin(2.0 * M_PI * 7 * i / N);
        input[i].im = sin(2.0 * M_PI * 5 * i / N) - 0.3 * cos(2.0 * M_PI * 11 * i / N);
    }
    
    fft_object fft = fft_init(N, 1);
    fft_exec(fft, input, output);
    
    // Compute energies
    double energy_time = 0.0, energy_freq = 0.0;
    for (int i = 0; i < N; i++) {
        energy_time += input[i].re * input[i].re + input[i].im * input[i].im;
        energy_freq += output[i].re * output[i].re + output[i].im * output[i].im;
    }
    energy_freq /= N; // Normalization
    
    printf("  Energy (time): %.10f\n", energy_time);
    printf("  Energy (freq): %.10f\n", energy_freq);
    printf("  Relative error: %.6e\n", fabs(energy_time - energy_freq) / energy_time);
    
    if (fabs(energy_time - energy_freq) / energy_time > tol) {
        printf("  FAIL: Energy not conserved\n");
        pass = 0;
    } else {
        printf("  PASS: Parseval's theorem holds\n");
    }
    
    free_fft(fft);
    free(input);
    free(output);
    return pass;
}

// ============================================================================
// TEST 5: Linearity for Complex FFT
// ============================================================================
int test_complex_linearity(void) {
    printf("\n=== TEST: Linearity (Complex) ===\n");
    const int N = 32;
    const double tol = 1e-9;
    const double a = 2.5, b = -1.7;
    int pass = 1;
    
    fft_data *x = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *y = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *sum = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *X = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *Y = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *Sum = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *Linear = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Generate signals
    for (int i = 0; i < N; i++) {
        x[i].re = cos(2.0 * M_PI * 3 * i / N);
        x[i].im = sin(2.0 * M_PI * 5 * i / N);
        y[i].re = sin(2.0 * M_PI * 7 * i / N);
        y[i].im = cos(2.0 * M_PI * 2 * i / N);
        sum[i].re = a * x[i].re + b * y[i].re;
        sum[i].im = a * x[i].im + b * y[i].im;
    }
    
    fft_object fft = fft_init(N, 1);
    fft_exec(fft, x, X);
    fft_exec(fft, y, Y);
    fft_exec(fft, sum, Sum);
    
    // Compute a*FFT(x) + b*FFT(y)
    for (int k = 0; k < N; k++) {
        Linear[k].re = a * X[k].re + b * Y[k].re;
        Linear[k].im = a * X[k].im + b * Y[k].im;
    }
    
    double mse = compute_complex_mse(Sum, Linear, N);
    printf("  MSE: %.6e\n", mse);
    
    if (mse > tol) {
        printf("  FAIL: Linearity violated\n");
        pass = 0;
    } else {
        printf("  PASS: FFT(ax+by) = aFFT(x) + bFFT(y)\n");
    }
    
    free_fft(fft);
    free(x); free(y); free(sum);
    free(X); free(Y); free(Sum); free(Linear);
    return pass;
}

// ============================================================================
// TEST 6: Round-Trip (FFT -> IFFT)
// ============================================================================
int test_complex_roundtrip(void) {
    printf("\n=== TEST: Complex Round-Trip (Various Sizes) ===\n");
    int sizes[] = {2, 3, 4, 5, 7, 8, 15, 16, 17, 32, 63, 64, 100, 128};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const double tol = 1e-6;  // Slightly relaxed for larger sizes
    int total = 0, passed = 0;
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        total++;
        
        fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *freq = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *reconstructed = (fft_data *)malloc(N * sizeof(fft_data));
        
        if (!input || !freq || !reconstructed) {
            printf("  N=%4d: FAIL (memory allocation)\n", N);
            free(input); free(freq); free(reconstructed);
            continue;
        }
        
        // Generate test signal with frequencies that exist for this N
        for (int i = 0; i < N; i++) {
            // Use frequencies that are always valid: k=1 and k=2
            input[i].re = cos(2.0 * M_PI * i / N) + 0.5 * sin(2.0 * M_PI * 2 * i / N);
            input[i].im = sin(2.0 * M_PI * i / N) - 0.3 * cos(2.0 * M_PI * 2 * i / N);
        }
        
        fft_object fwd = fft_init(N, 1);
        fft_object inv = fft_init(N, -1);
        
        if (!fwd || !inv) {
            printf("  N=%4d: FAIL (initialization)\n", N);
            free(input); free(freq); free(reconstructed);
            if (fwd) free_fft(fwd);
            if (inv) free_fft(inv);
            continue;
        }
        
        // Forward FFT
        fft_exec(fwd, input, freq);
        
        // Inverse FFT
        fft_exec(inv, freq, reconstructed);
        
        // Scale by 1/N (required for round-trip)
        for (int i = 0; i < N; i++) {
            reconstructed[i].re /= N;
            reconstructed[i].im /= N;
        }
        
        double mse = compute_complex_mse(input, reconstructed, N);
        
        // Adaptive tolerance for larger sizes
        double adaptive_tol = tol * (1.0 + log10((double)N));
        
        if (mse < adaptive_tol) {
            printf("  N=%4d: PASS (MSE=%.3e)\n", N, mse);
            passed++;
        } else {
            printf("  N=%4d: FAIL (MSE=%.3e, threshold=%.3e)\n", N, mse, adaptive_tol);
            
            // ✅ FIXED: Show first 4 samples always
            printf("    First 4 samples:\n");
            for (int i = 0; i < 4 && i < N; i++) {
                double err_re = input[i].re - reconstructed[i].re;
                double err_im = input[i].im - reconstructed[i].im;
                printf("      [%d] orig=(%.6f,%.6f) recon=(%.6f,%.6f) err=(%.3e,%.3e)\n",
                       i, input[i].re, input[i].im,
                       reconstructed[i].re, reconstructed[i].im,
                       err_re, err_im);
            }
            
            // ✅ FIXED: Show first 10 large errors
            printf("    First 10 large errors:\n");
            int count = 0;
            for (int i = 0; i < N && count < 10; i++) {
                double err_re = input[i].re - reconstructed[i].re;
                double err_im = input[i].im - reconstructed[i].im;
                double err_mag = sqrt(err_re*err_re + err_im*err_im);
                
                if (err_mag > 0.01) {  // Only show significant errors
                    printf("      [%d] orig=(%.6f,%.6f) recon=(%.6f,%.6f) err=(%.3e,%.3e)\n",
                        i, input[i].re, input[i].im, 
                        reconstructed[i].re, reconstructed[i].im, 
                        err_re, err_im);
                    count++;
                }
            }
        }
        
        free_fft(fwd);
        free_fft(inv);
        free(input);
        free(freq);
        free(reconstructed);
    }
    
    printf("\n  Summary: %d/%d sizes passed\n", passed, total);
    return (passed == total);
}

// ============================================================================
// TEST 7: Phase Accuracy (Chirp Signal)
// ============================================================================
int test_phase_accuracy(void) {
    printf("\n=== TEST: Phase Accuracy (Chirp) ===\n");
    const int N = 128;
    const double tol = 1e-9;
    int pass = 1;
    
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Chirp signal: frequency increases linearly
    for (int n = 0; n < N; n++) {
        double t = (double)n / N;
        double phase = 2.0 * M_PI * 10.0 * t * t; // Quadratic phase
        input[n].re = cos(phase);
        input[n].im = sin(phase);
    }
    
    fft_object fft = fft_init(N, 1);
    fft_exec(fft, input, output);
    
    // Check energy distribution is reasonable
    double total_energy = 0.0;
    for (int k = 0; k < N; k++) {
        total_energy += output[k].re * output[k].re + output[k].im * output[k].im;
    }
    total_energy /= N;
    
    // Energy should equal input energy (Parseval)
    double input_energy = (double)N; // |e^{iθ}| = 1
    
    printf("  Input energy:  %.10f\n", input_energy);
    printf("  Output energy: %.10f\n", total_energy);
    printf("  Relative error: %.6e\n", fabs(input_energy - total_energy) / input_energy);
    
    if (fabs(input_energy - total_energy) / input_energy > tol) {
        printf("  FAIL: Phase errors causing energy loss\n");
        pass = 0;
    } else {
        printf("  PASS: Phase preserved (energy conserved)\n");
    }
    
    free_fft(fft);
    free(input);
    free(output);
    return pass;
}

void print_complex_array(const char *label, fft_data *arr, int N) {
    printf("%s:\n", label);
    for (int i = 0; i < N; i++) {
        printf("  [%d] = (%10.6f, %10.6f)\n", i, arr[i].re, arr[i].im);
    }
}

void print_radix_info(fft_object fft) {
    printf("\n=== FFT Object Configuration ===\n");
    printf("n_input: %d\n", fft->n_input);
    printf("n_fft: %d\n", fft->n_fft);
    printf("sgn: %d\n", fft->sgn);
    printf("lt (Bluestein?): %d\n", fft->lt);
    printf("lf (num factors): %d\n", fft->lf);
    printf("Radix factors: [");
    for (int i = 0; i < fft->lf; i++) {
        printf("%d", fft->factors[i]);
        if (i < fft->lf - 1) printf(", ");
    }
    printf("]\n");
}

void test1_delta_function(void) {
    printf("\n" "----------------------------------------------------------------\n");
    printf("TEST 1: Delta Function [1, 0, 0, 0, 0, 0, 0, 0]\n");
    printf("Expected: All frequency bins should be (1.0, 0.0)\n");
    printf("----------------------------------------------------------------\n");
    
    const int N = 15;
    fft_data *input = (fft_data *)calloc(N, sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Delta function: impulse at n=0
    input[0].re = 1.0;
    input[0].im = 0.0;
    
    print_complex_array("Input", input, N);
    
    fft_object fft = fft_init(N, 1);
    print_radix_info(fft);
    
    fft_exec(fft, input, output);
    
    print_complex_array("\nOutput (Frequency Domain)", output, N);
    
    // Check results
    printf("\n--- Analysis ---\n");
    int all_pass = 1;
    for (int k = 0; k < N; k++) {
        double err_re = fabs(output[k].re - 1.0);
        double err_im = fabs(output[k].im - 0.0);
        double total_err = sqrt(err_re * err_re + err_im * err_im);
        
        char status = (total_err < 1e-6) ? 'c' : 'w';
        printf("  X[%d]: error = %.3e %c\n", k, total_err, status);
        
        if (total_err >= 1e-6) all_pass = 0;
    }
    
    printf("\nResult: %s\n", all_pass ? "PASS ✓" : "FAIL ✗");
    
    free_fft(fft);
    free(input);
    free(output);
}

void test2_dc_component(void) {
    printf("\n" "----------------------------------------------------------------\n");
    printf("TEST 2: DC Component [1, 1, 1, 1, 1, 1, 1, 1]\n");
    printf("Expected: X[0] = (8.0, 0.0), all others = (0.0, 0.0)\n");
    printf("----------------------------------------------------------------\n");
    
    const int N = 15;
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // All ones
    for (int i = 0; i < N; i++) {
        input[i].re = 1.0;
        input[i].im = 0.0;
    }
    
    print_complex_array("Input", input, N);
    
    fft_object fft = fft_init(N, 1);
    fft_exec(fft, input, output);
    
    print_complex_array("\nOutput (Frequency Domain)", output, N);
    
    // Check results
    printf("\n--- Analysis ---\n");
    printf("  X[0]: expected (N, 0.0), got (%.6f, %.6f) %c\n",
           output[0].re, output[0].im,
           (fabs(output[0].re - N) < 1e-6 && fabs(output[0].im) < 1e-6) ? 'c' : 'w');
    
    int all_others_zero = 1;
    for (int k = 1; k < N; k++) {
        double mag = sqrt(output[k].re * output[k].re + output[k].im * output[k].im);
        if (mag > 1e-6) {
            printf("  X[%d]: magnitude %.3e (expected 0) ✗\n", k, mag);
            all_others_zero = 0;
        }
    }
    
    if (all_others_zero) {
        printf("  X[1..7]: all zero ✓\n");
    }
    
    printf("\nResult: %s\n", 
           (fabs(output[0].re - 7.0) < 1e-6 && all_others_zero) ? "PASS ✓" : "FAIL ✗");
    
    free_fft(fft);
    free(input);
    free(output);
}

void test3_twiddle_factors(void) {
    printf("\n" "----------------------------------------------------------------\n");
    printf("TEST 3: Twiddle Factor Verification\n");
    printf("Expected: W_k = exp(-2πi*k/8) for forward transform\n");
    printf("----------------------------------------------------------------\n");
    
    const int N = 15;
    fft_object fft = fft_init(N, 1);
    
    printf("\nk  | Computed                  | Expected                  | Error\n");
    printf("---|---------------------------|---------------------------|----------\n");
    
    int all_pass = 1;
    for (int k = 0; k < N; k++) {
        // Expected value
        double angle = -2.0 * M_PI * k / N;
        double expected_re = cos(angle);
        double expected_im = sin(angle);
        
        // Computed value
        double computed_re = fft->twiddles[k].re;
        double computed_im = fft->twiddles[k].im;
        
        // Error
        double err_re = fabs(computed_re - expected_re);
        double err_im = fabs(computed_im - expected_im);
        double total_err = sqrt(err_re * err_re + err_im * err_im);
        
        char status = (total_err < 1e-10) ? 'c' : 'w';
        printf("%2d | (%10.6f, %10.6f) | (%10.6f, %10.6f) | %.3e %c\n",
               k, computed_re, computed_im, expected_re, expected_im, 
               total_err, status);
        
        if (total_err >= 1e-10) all_pass = 0;
    }
    
    printf("\nResult: %s\n", all_pass ? "PASS ✓" : "FAIL ✗");
    
    free_fft(fft);
}

void test4_single_frequency(void) {
    printf("\n" "----------------------------------------------------------------\n");
    printf("TEST 4: Single Frequency cos(2π*1*n/8)\n");
    printf("Expected: X[1] = X[7] = (4.0, 0.0), all others ≈ 0\n");
    printf("----------------------------------------------------------------\n");
    
    const int N = 15;
    const int k_freq = 1;
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Generate cos(2π*k*n/N)
    for (int n = 0; n < N; n++) {
        input[n].re = cos(2.0 * M_PI * k_freq * n / N);
        input[n].im = 0.0;
    }
    
    print_complex_array("Input", input, N);
    
    fft_object fft = fft_init(N, 1);
    fft_exec(fft, input, output);
    
    print_complex_array("\nOutput (Frequency Domain)", output, N);
    
    // Check results
    printf("\n--- Analysis ---\n");
    double expected = N / 2.0;  // Real cosine splits energy between ±k
    
    // Check bin 1
    double mag1 = sqrt(output[1].re * output[1].re + output[1].im * output[1].im);
    printf("  X[1]: magnitude %.6f (expected %.1f) %c\n", 
           mag1, expected, (fabs(mag1 - expected) < 1e-6) ? 'c' : 'w');
    
    // Check bin 7 (N-1, the negative frequency)
    double mag7 = sqrt(output[6].re * output[6].re + output[6].im * output[6].im);
    printf("  X[7]: magnitude %.6f (expected %.1f) %c\n", 
           mag7, expected, (fabs(mag7 - expected) < 1e-6) ? 'c' : 'w');

    int others_ok = 1;
    for (int k = 0; k < N; k++)
    {
        if (k == k_freq || k == (N - k_freq))
            continue; // skip the two cosine bins
        double mag = sqrt(output[k].re * output[k].re + output[k].im * output[k].im);
        if (mag > 1e-6)
        {
            printf("  X[%d]: magnitude %.3e (expected 0) ✗\n", k, mag);
            others_ok = 0;
        }
    }
    if (others_ok)
    {
        printf("  X[0,2,3,4,5,6]: all zero ✓\n");
    }

    printf("\nResult: %s\n", 
           (fabs(mag1 - expected) < 1e-6 && fabs(mag7 - expected) < 1e-6 && others_ok) 
           ? "PASS ✓" : "FAIL ✗");
    
    free_fft(fft);
    free(input);
    free(output);
}

void test5_roundtrip(void) {
    printf("\n" "----------------------------------------------------------------\n");
    printf("TEST 5: Round-Trip (FFT -> IFFT)\n");
    printf("Expected: Perfect reconstruction after scaling by 1/N\n");
    printf("----------------------------------------------------------------\n");
    
    const int N = 15;
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *freq = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *reconstructed = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Generate test signal
    for (int i = 0; i < N; i++) {
        input[i].re = cos(2.0 * M_PI * i / N) + 0.5 * sin(2.0 * M_PI * 2 * i / N);
        input[i].im = sin(2.0 * M_PI * i / N) - 0.3 * cos(2.0 * M_PI * 2 * i / N);
    }
    
    print_complex_array("Input", input, N);
    
    fft_object fwd = fft_init(N, 1);
    fft_object inv = fft_init(N, -1);
    
    // Forward FFT
    fft_exec(fwd, input, freq);
    print_complex_array("\nAfter Forward FFT", freq, N);
    
    // DEBUG: Manually compute what the FFT should be
    printf("\n--- Expected Forward FFT (DFT formula) ---\n");
    for (int k = 0; k < N; k++) {
        double sum_re = 0.0, sum_im = 0.0;
        for (int n = 0; n < N; n++) {
            double angle = -2.0 * M_PI * k * n / N;
            double wr = cos(angle);
            double wi = sin(angle);
            sum_re += input[n].re * wr - input[n].im * wi;
            sum_im += input[n].re * wi + input[n].im * wr;
        }
        printf("  Expected X[%d] = (%.6f, %.6f)\n", k, sum_re, sum_im);
        printf("  Got      X[%d] = (%.6f, %.6f)\n", k, freq[k].re, freq[k].im);
        double err = sqrt(pow(sum_re - freq[k].re, 2) + pow(sum_im - freq[k].im, 2));
        printf("  Error = %.3e %s\n\n", err, (err < 1e-6) ? "✓" : "✗");
    }
    
    // Inverse FFT
    fft_exec(inv, freq, reconstructed);
    print_complex_array("\nAfter Inverse FFT (before scaling)", reconstructed, N);
    
    // Scale by 1/N
    for (int i = 0; i < N; i++) {
        reconstructed[i].re /= N;
        reconstructed[i].im /= N;
    }
    print_complex_array("\nAfter Scaling by 1/N", reconstructed, N);
    
    // Compute MSE
    double mse = 0.0;
    for (int i = 0; i < N; i++) {
        double err_re = input[i].re - reconstructed[i].re;
        double err_im = input[i].im - reconstructed[i].im;
        mse += err_re * err_re + err_im * err_im;
    }
    mse /= N;
    
    printf("\n--- Analysis ---\n");
    printf("MSE: %.3e\n", mse);
    printf("Threshold: 1e-6\n");
    printf("\nSample-by-sample errors:\n");
    for (int i = 0; i < N; i++) {
        double err_re = input[i].re - reconstructed[i].re;
        double err_im = input[i].im - reconstructed[i].im;
        double err_mag = sqrt(err_re * err_re + err_im * err_im);
        printf("  [%d] error = %.3e %c\n", i, err_mag, (err_mag < 1e-6) ? 'c' : 'w');
    }
    
    printf("\nResult: %s\n", (mse < 1e-6) ? "PASS ✓" : "FAIL ✗");
    
    free_fft(fwd);
    free_fft(inv);
    free(input);
    free(freq);
    free(reconstructed);
}

void test6_energy_conservation(void) {
    printf("\n" "----------------------------------------------------------------\n");
    printf("TEST 6: Energy Conservation (Parseval's Theorem)\n");
    printf("Expected: ∑|x[n]|² = (1/N) ∑|X[k]|²\n");
    printf("----------------------------------------------------------------\n");
    
    const int N = 7;
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Generate test signal
    for (int i = 0; i < N; i++) {
        input[i].re = cos(2.0 * M_PI * i / N) + 0.5 * sin(2.0 * M_PI * 2 * i / N);
        input[i].im = sin(2.0 * M_PI * i / N) - 0.3 * cos(2.0 * M_PI * 2 * i / N);
    }
    
    print_complex_array("Input", input, N);
    
    fft_object fft = fft_init(N, 1);
    fft_exec(fft, input, output);
    
    print_complex_array("\nOutput (Frequency Domain)", output, N);
    
    // Compute energies
    double energy_time = 0.0, energy_freq = 0.0;
    for (int i = 0; i < N; i++) {
        energy_time += input[i].re * input[i].re + input[i].im * input[i].im;
        energy_freq += output[i].re * output[i].re + output[i].im * output[i].im;
    }
    
    printf("\n--- Analysis ---\n");
    printf("Energy (time domain):           %.10f\n", energy_time);
    printf("Energy (freq domain, raw):      %.10f\n", energy_freq);
    printf("Energy (freq domain, /N):       %.10f\n", energy_freq / N);
    printf("Relative error (with /N):       %.3e\n", 
           fabs(energy_time - energy_freq/N) / energy_time);
    
    printf("\nResult: %s\n", 
           (fabs(energy_time - energy_freq/N) / energy_time < 1e-6) ? "PASS ✓" : "FAIL ✗");
    
    free_fft(fft);
    free(input);
    free(output);
}


// ============================================================================
// BENCHMARK 1: Throughput vs Size
// ============================================================================
void benchmark_throughput(void) {
    printf("\n╔----------------------------------------------------------------╗\n");
    printf("*  BENCHMARK 1: Throughput vs FFT Size                          *\n");
    printf("*----------------------------------------------------------------*\n\n");
    
    printf("%-8s %-10s %-12s %-12s %-15s %-15s\n",
           "Size", "Algorithm", "Time(ms)", "Iter", "μs/FFT", "MFLOPS");
    printf("─────────────────────────────────────────────────────────────────────\n");
    
    // Test sizes: powers of 2, common sizes, primes
    int sizes[] = {
        8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192,  // Powers of 2
        100, 200, 500, 1000,                                     // Round numbers
        127, 251, 509, 1021,                                     // Primes (Bluestein)
        60, 120, 240, 480, 960                                   // Highly composite
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        
        // Allocate buffers
        fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
        
        if (!input || !output) {
            free(input);
            free(output);
            continue;
        }
        
        // Initialize with random data
        for (int i = 0; i < N; i++) {
            input[i].re = sin(2.0 * M_PI * i / N);
            input[i].im = cos(2.0 * M_PI * i / N);
        }
        
        // Create FFT object
        fft_object fft = fft_init(N, 1);
        if (!fft) {
            free(input);
            free(output);
            continue;
        }
        
        // Determine iteration count (more for small sizes)
        int iterations;
        if (N <= 64) iterations = 100000;
        else if (N <= 256) iterations = 50000;
        else if (N <= 1024) iterations = 10000;
        else if (N <= 4096) iterations = 2000;
        else iterations = 500;
        
        // Warm-up
        for (int i = 0; i < 10; i++) {
            fft_exec(fft, input, output);
        }
        
        // Benchmark
        double start = get_time_ms();
        for (int i = 0; i < iterations; i++) {
            fft_exec(fft, input, output);
        }
        double end = get_time_ms();
        
        double elapsed_ms = end - start;
        double us_per_fft = (elapsed_ms * 1000.0) / iterations;
        
        // Estimate FLOPS: Complex FFT requires ~5N*log2(N) operations
        double flops_per_fft = 5.0 * N * log2((double)N);
        double mflops = (flops_per_fft * iterations) / (elapsed_ms * 1000.0);
        
        const char *algo = (fft->lt == 0) ? "Mixed-Radix" : "Bluestein";
        
        printf("%-8d %-10s %-12.2f %-12d %-15.3f %-15.1f\n",
               N, algo, elapsed_ms, iterations, us_per_fft, mflops);
        
        free_fft(fft);
        free(input);
        free(output);
    }
}


void benchmark_latency(void) {
    printf("\n╔----------------------------------------------------------------╗\n");
    printf("*  BENCHMARK 2: Single Transform Latency                        *\n");
    printf("*----------------------------------------------------------------*\n\n");
    
    printf("%-8s %-10s %-15s %-15s\n",
           "Size", "Algorithm", "Latency(μs)", "Cycles/Sample");
    printf("─────────────────────────────────────────────────────────────────\n");
    
    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    // Assume ~3 GHz CPU for cycle estimation
    const double CPU_FREQ_GHZ = 3.0;
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        
        fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
        
        for (int i = 0; i < N; i++) {
            input[i].re = sin(2.0 * M_PI * i / N);
            input[i].im = cos(2.0 * M_PI * i / N);
        }
        
        fft_object fft = fft_init(N, 1);
        if (!fft) {
            free(input);
            free(output);
            continue;
        }
        
        // Warm-up
        for (int i = 0; i < 100; i++) {
            fft_exec(fft, input, output);
        }
        
        // Measure single transform many times
        const int trials = 1000;
        double min_time = 1e9;
        
        for (int t = 0; t < trials; t++) {
            double start = get_time_ms();
            fft_exec(fft, input, output);
            double end = get_time_ms();
            double elapsed = (end - start) * 1000.0; // Convert to μs
            if (elapsed < min_time) min_time = elapsed;
        }
        
        double cycles_per_sample = (min_time * CPU_FREQ_GHZ) / N;
        
        const char *algo = (fft->lt == 0) ? "Mixed-Radix" : "Bluestein";
        
        printf("%-8d %-10s %-15.3f %-15.1f\n",
               N, algo, min_time, cycles_per_sample);
        
        free_fft(fft);
        free(input);
        free(output);
    }
}

// ============================================================================
// BENCHMARK 3: Cache Effects (In-place vs Out-of-place)
// ============================================================================
void benchmark_cache_effects(void) {
    printf("\n╔----------------------------------------------------------------╗\n");
    printf("*  BENCHMARK 3: Cache Effects (In-place vs Out-of-place)       *\n");
    printf("*----------------------------------------------------------------*\n\n");
    
    printf("%-8s %-15s %-15s %-15s\n",
           "Size", "In-place(μs)", "Out-place(μs)", "Speedup");
    printf("─────────────────────────────────────────────────────────────────\n");
    
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        int iterations = (N <= 1024) ? 10000 : 2000;
        
        fft_data *buffer = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
        
        for (int i = 0; i < N; i++) {
            buffer[i].re = input[i].re = sin(2.0 * M_PI * i / N);
            buffer[i].im = input[i].im = cos(2.0 * M_PI * i / N);
        }
        
        fft_object fft = fft_init(N, 1);
        if (!fft) {
            free(buffer);
            free(input);
            free(output);
            continue;
        }
        
        // Benchmark in-place
        double start = get_time_ms();
        for (int i = 0; i < iterations; i++) {
            fft_exec(fft, buffer, buffer);
        }
        double end = get_time_ms();
        double inplace_us = (end - start) * 1000.0 / iterations;
        
        // Benchmark out-of-place
        start = get_time_ms();
        for (int i = 0; i < iterations; i++) {
            fft_exec(fft, input, output);
        }
        end = get_time_ms();
        double outplace_us = (end - start) * 1000.0 / iterations;
        
        double speedup = outplace_us / inplace_us;
        
        printf("%-8d %-15.3f %-15.3f %-15.2fx\n",
               N, inplace_us, outplace_us, speedup);
        
        free_fft(fft);
        free(buffer);
        free(input);
        free(output);
    }
}

// ============================================================================
// BENCHMARK 4: Initialization Overhead
// ============================================================================
void benchmark_init_overhead(void) {
    printf("\n╔----------------------------------------------------------------╗\n");
    printf("*  BENCHMARK 4: Initialization Overhead                         *\n");
    printf("*----------------------------------------------------------------*\n\n");
    
    printf("%-8s %-10s %-15s %-20s\n",
           "Size", "Algorithm", "Init Time(ms)", "Init/Exec Ratio");
    printf("─────────────────────────────────────────────────────────────────\n");
    
    int sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        
        // Measure initialization time
        double start = get_time_ms();
        fft_object fft = fft_init(N, 1);
        double end = get_time_ms();
        double init_time = end - start;
        
        if (!fft) continue;
        
        // Measure execution time
        fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
        
        for (int i = 0; i < N; i++) {
            input[i].re = sin(2.0 * M_PI * i / N);
            input[i].im = 0.0;
        }
        
        start = get_time_ms();
        fft_exec(fft, input, output);
        end = get_time_ms();
        double exec_time = end - start;
        
        double ratio = init_time / exec_time;
        const char *algo = (fft->lt == 0) ? "Mixed-Radix" : "Bluestein";
        
        printf("%-8d %-10s %-15.3f %-20.1f\n",
               N, algo, init_time, ratio);
        
        free_fft(fft);
        free(input);
        free(output);
    }
}

// ============================================================================
// BENCHMARK 5: Forward vs Inverse Performance
// ============================================================================
void benchmark_forward_vs_inverse(void) {
    printf("\n╔----------------------------------------------------------------╗\n");
    printf("*  BENCHMARK 5: Forward vs Inverse Transform Speed              *\n");
    printf("*----------------------------------------------------------------*\n\n");
    
    printf("%-8s %-15s %-15s %-15s\n",
           "Size", "Forward(μs)", "Inverse(μs)", "Ratio (I/F)");
    printf("─────────────────────────────────────────────────────────────────\n");
    
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        int iterations = (N <= 1024) ? 10000 : 2000;
        
        fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
        
        for (int i = 0; i < N; i++) {
            input[i].re = sin(2.0 * M_PI * i / N);
            input[i].im = cos(2.0 * M_PI * i / N);
        }
        
        fft_object fwd = fft_init(N, 1);
        fft_object inv = fft_init(N, -1);
        
        if (!fwd || !inv) {
            if (fwd) free_fft(fwd);
            if (inv) free_fft(inv);
            free(input);
            free(output);
            continue;
        }
        
        // Warm-up
        for (int i = 0; i < 100; i++) {
            fft_exec(fwd, input, output);
            fft_exec(inv, output, input);
        }
        
        // Benchmark forward
        double start = get_time_ms();
        for (int i = 0; i < iterations; i++) {
            fft_exec(fwd, input, output);
        }
        double end = get_time_ms();
        double fwd_us = (end - start) * 1000.0 / iterations;
        
        // Benchmark inverse
        start = get_time_ms();
        for (int i = 0; i < iterations; i++) {
            fft_exec(inv, output, input);
        }
        end = get_time_ms();
        double inv_us = (end - start) * 1000.0 / iterations;
        
        double ratio = inv_us / fwd_us;
        
        printf("%-8d %-15.3f %-15.3f %-15.2f\n",
               N, fwd_us, inv_us, ratio);
        
        free_fft(fwd);
        free_fft(inv);
        free(input);
        free(output);
    }
}

void benchmark_efficiency(void) {
    printf("\n╔----------------------------------------------------------------╗\n");
    printf("*  BENCHMARK 6: Computational Efficiency                        *\n");
    printf("*----------------------------------------------------------------*\n\n");
    
    printf("%-8s %-12s %-15s %-15s %-15s\n",
           "Size", "Time(μs)", "Actual MFLOPS", "Peak MFLOPS", "Efficiency%%");
    printf("─────────────────────────────────────────────────────────────────────────\n");
    
    int sizes[] = {256, 512, 1024, 2048, 4096, 8192};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    
    // Assume ~100 GFLOPS peak (conservative for modern CPU with SIMD)
    const double PEAK_GFLOPS = 100.0;
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        int iterations = (N <= 1024) ? 10000 : 2000;
        
        fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
        
        for (int i = 0; i < N; i++) {
            input[i].re = sin(2.0 * M_PI * i / N);
            input[i].im = cos(2.0 * M_PI * i / N);
        }
        
        fft_object fft = fft_init(N, 1);
        if (!fft) {
            free(input);
            free(output);
            continue;
        }
        
        // Warm-up
        for (int i = 0; i < 100; i++) {
            fft_exec(fft, input, output);
        }
        
        // Benchmark
        double start = get_time_ms();
        for (int i = 0; i < iterations; i++) {
            fft_exec(fft, input, output);
        }
        double end = get_time_ms();
        
        double elapsed_ms = end - start;
        double us_per_fft = (elapsed_ms * 1000.0) / iterations;
        
        // Complex FFT: ~5N*log2(N) FLOPs
        double flops_per_fft = 5.0 * N * log2((double)N);
        double mflops = flops_per_fft / us_per_fft;
        double efficiency = (mflops / (PEAK_GFLOPS * 1000.0)) * 100.0;
        
        printf("%-8d %-12.3f %-15.1f %-15.1f %-15.1f%%\n",
               N, us_per_fft, mflops, PEAK_GFLOPS * 1000.0, efficiency);
        
        free_fft(fft);
        free(input);
        free(output);
    }
}

// ============================================================================
// MAIN BENCHMARK RUNNER
// ============================================================================
void run_all_benchmarks(void) {
    printf("\n");
    printf("╔-------------------------------------------------------------------╗\n");
    printf("*                   FFT PERFORMANCE BENCHMARK SUITE                 *\n");
    printf("*-------------------------------------------------------------------*\n");
    
    benchmark_throughput();
    benchmark_latency();
    benchmark_cache_effects();
    benchmark_init_overhead();
    benchmark_forward_vs_inverse();
    benchmark_efficiency();
    
    printf("\n");
    printf("╔-------------------------------------------------------------------╗\n");
    printf("*                     BENCHMARK COMPLETE                            *\n");
    printf("*-------------------------------------------------------------------*\n\n");
}


int run_comprehensive_complex_fft_N32_tests(void) {
    printf("\n");
    printf("╔----------------------------------------------------------------╗\n");
    printf("*              N=15 FFT DIAGNOSTIC TEST SUITE                    *\n");
    printf("*----------------------------------------------------------------*\n");
    
    test1_delta_function();
    test2_dc_component();
    test3_twiddle_factors();
    test4_single_frequency();
    test5_roundtrip();
    test6_energy_conservation();
    
    printf("\n");
    printf("╔----------------------------------------------------------------╗\n");
    printf("*                   DIAGNOSTIC COMPLETE                          *\n");
    printf("*----------------------------------------------------------------*\n\n");
    
    return 0;
}

// ============================================================================
// MAIN TEST RUNNER
// ============================================================================
int run_comprehensive_complex_fft_tests(void) {
    printf("\n");
    printf("╔--------------------------------------------------------╗\n");
    printf("*     COMPREHENSIVE COMPLEX FFT TEST SUITE              *\n");
    printf("*--------------------------------------------------------*\n");
    
    int total = 0, passed = 0;
    
    if (test_complex_impulse()) passed++;
    total++;
    
    if (test_complex_dc()) passed++;
    total++;
    
    if (test_complex_exponential()) passed++;
    total++;
    
    if (test_complex_parseval()) passed++;
    total++;
    
    if (test_complex_linearity()) passed++;
    total++;
    
    if (test_complex_roundtrip()) passed++;
    total++;
    
    if (test_phase_accuracy()) passed++;
    total++;
    
    printf("\n");
    printf("╔--------------------------------------------------------╗\n");
    printf("*     COMPLEX FFT TEST RESULTS                          *\n");
    printf("╠--------------------------------------------------------╣\n");
    printf("*  Tests Passed:  %2d / %2d                               *\n", 
           passed, total);
    printf("*  Success Rate:  %.1f%%                                 *\n",
           100.0 * passed / total);
    printf("*--------------------------------------------------------*\n");
    
    return (passed == total) ? 1 : 0;
}


void debug_fft_scaling(void) {
    printf("\n=== DEBUG: FFT Scaling Convention ===\n");
    const int N = 32;
    
    fft_data *input = (fft_data *)calloc(N, sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Impulse: [1, 0, 0, ...]
    input[0].re = 1.0;
    
    fft_object fft = fft_init(N, 1);
    fft_exec(fft, input, output);
    
    printf("Input: impulse at n=0\n");
    printf("Output X[0] = %.6f (should be 1.0 or N=%d)\n", output[0].re, N);
    
    // Compute energies
    double energy_in = 1.0;  // Just the impulse
    double energy_out = 0.0;
    for (int k = 0; k < N; k++) {
        energy_out += output[k].re * output[k].re + output[k].im * output[k].im;
    }
    
    printf("Energy (time): %.6f\n", energy_in);
    printf("Energy (freq, raw): %.6f\n", energy_out);
    printf("Energy (freq, /N): %.6f\n", energy_out / N);
    printf("Energy (freq, /N²): %.6f\n", energy_out / (N * N));
    
    printf("\nConclusion:\n");
    if (fabs(energy_out - energy_in) < 1e-10) {
        printf("  → Your FFT uses UNITARY scaling (1/√N)\n");
    } else if (fabs(energy_out / N - energy_in) < 1e-10) {
        printf("  → Your FFT has NO FORWARD SCALING\n");
        printf("  → Parseval needs: energy_freq /= N (not /= N²)\n");
    } else if (fabs(energy_out / (N*N) - energy_in) < 1e-10) {
        printf("  → Your FFT scales BOTH ways by 1/N\n");
    }
    
    free_fft(fft);
    free(input);
    free(output);
}

void debug_parseval_detailed(void) {
    printf("\n=== DEBUG: Detailed Parseval Analysis ===\n");
    const int N = 64;
    
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Same signal as the test
    for (int i = 0; i < N; i++) {
        input[i].re = cos(2.0 * M_PI * 3 * i / N) + 0.5 * sin(2.0 * M_PI * 7 * i / N);
        input[i].im = sin(2.0 * M_PI * 5 * i / N) - 0.3 * cos(2.0 * M_PI * 11 * i / N);
    }
    
    fft_object fft = fft_init(N, 1);
    fft_exec(fft, input, output);
    
    // Compute energies
    double energy_time = 0.0, energy_freq = 0.0;
    for (int i = 0; i < N; i++) {
        energy_time += input[i].re * input[i].re + input[i].im * input[i].im;
        energy_freq += output[i].re * output[i].re + output[i].im * output[i].im;
    }
    
    printf("Energy (time domain): %.10f\n", energy_time);
    printf("Energy (freq, raw):   %.10f\n", energy_freq);
    printf("Energy (freq, /N):    %.10f\n", energy_freq / N);
    printf("Ratio (freq_raw / time): %.10f (should be %.1f)\n", 
           energy_freq / energy_time, (double)N);
    
    // Check specific bins
    printf("\nFirst 5 frequency bins:\n");
    for (int k = 0; k < 5; k++) {
        double mag = sqrt(output[k].re * output[k].re + output[k].im * output[k].im);
        printf("  X[%d] = (%.6f, %.6f), |X[%d]| = %.6f\n", 
               k, output[k].re, output[k].im, k, mag);
    }
    
    free_fft(fft);
    free(input);
    free(output);
}

void diagnose_fft_issue(void) {
    printf("\n=== DIAGNOSTIC: FFT Direction Test ===\n");
    int N = 32;
    
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *freq = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *reconstructed = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Simple impulse: [1, 0, 0, 0, ...]
    for (int i = 0; i < N; i++) {
        input[i].re = (i == 0) ? 1.0 : 0.0;
        input[i].im = 0.0;
    }
    
    fft_object fwd = fft_init(N, 1);
    fft_object inv = fft_init(N, -1);
    
    printf("Input impulse:\n");
    for (int i = 0; i < N; i++) {
        printf("  x[%d] = (%.3f, %.3f)\n", i, input[i].re, input[i].im);
    }
    
    // Forward FFT
    fft_exec(fwd, input, freq);
    printf("\nAfter Forward FFT (should be all ones):\n");
    for (int i = 0; i < N; i++) {
        printf("  X[%d] = (%.3f, %.3f)\n", i, freq[i].re, freq[i].im);
    }
    
    // Inverse FFT
    fft_exec(inv, freq, reconstructed);
    
    printf("\nAfter Inverse FFT (before scaling):\n");
    for (int i = 0; i < N; i++) {
        printf("  x[%d] = (%.3f, %.3f)\n", i, reconstructed[i].re, reconstructed[i].im);
    }
    
    // Scale
    for (int i = 0; i < N; i++) {
        reconstructed[i].re /= N;
        reconstructed[i].im /= N;
    }
    
    printf("\nAfter scaling by 1/N:\n");
    for (int i = 0; i < N; i++) {
        printf("  x[%d] = (%.3f, %.3f) [should be (1,0) at i=0, else (0,0)]\n", 
               i, reconstructed[i].re, reconstructed[i].im);
    }
    
    // Check energy
    double energy_time = 0.0, energy_freq = 0.0;
    for (int i = 0; i < N; i++) {
        energy_time += input[i].re * input[i].re + input[i].im * input[i].im;
        energy_freq += freq[i].re * freq[i].re + freq[i].im * freq[i].im;
    }
    printf("\nEnergy check:\n");
    printf("  Time domain: %.6f\n", energy_time);
    printf("  Freq domain (raw): %.6f\n", energy_freq);
    printf("  Freq domain (/N): %.6f\n", energy_freq / N);
    printf("  Parseval satisfied: %s\n", 
           fabs(energy_time - energy_freq/N) < 1e-6 ? "YES" : "NO");
    
    free_fft(fwd);
    free_fft(inv);
    free(input);
    free(freq);
    free(reconstructed);
}

void debug_radix_selection(void) {
    printf("\n=== DEBUG: Radix Selection ===\n");
    int sizes[] = {2, 4, 8, 16, 32, 64, 128};
    
    for (int i = 0; i < 7; i++) {
        int N = sizes[i];
        fft_object fft = fft_init(N, 1);
        
        printf("N=%d: ", N);
        printf("lt=%d, lf=%d, factors=[", fft->lt, fft->lf);
        for (int j = 0; j < fft->lf; j++) {
            printf("%d", fft->factors[j]);
            if (j < fft->lf - 1) printf(",");
        }
        printf("]\n");
        
        free_fft(fft);
    }
}

void debug_radix_7_execution(void) {
    printf("\n=== DEBUG: Radix-32 Execution Trace ===\n");
    const int N = 7;
    
    fft_object fft = fft_init(N, 1);
    if (!fft) {
        printf("ERROR: Failed to initialize FFT\n");
        return;
    }
    
    // Print FFT plan
    printf("\nFFT Plan for N=%d:\n", N);
    printf("  n_input:  %d\n", fft->n_input);
    printf("  n_fft:    %d\n", fft->n_fft);
    printf("  sgn:      %d\n", fft->sgn);
    printf("  lt:       %d (0=mixed-radix, 1=Bluestein)\n", fft->lt);
    printf("  lf:       %d (number of radix stages)\n", fft->lf);
    printf("  factors:  [");
    for (int i = 0; i < fft->lf; i++) {
        printf("%d", fft->factors[i]);
        if (i < fft->lf - 1) printf(", ");
    }
    printf("]\n");
    
    // Print precomputation info
    printf("  num_precomputed_stages: %d\n", fft->num_precomputed_stages);
    printf("  twiddle_factors: %s\n", fft->twiddle_factors ? "allocated" : "NULL");
    
    if (fft->twiddle_factors && fft->num_precomputed_stages > 0) {
        printf("\n  Stage twiddle offsets:\n");
        for (int i = 0; i < fft->num_precomputed_stages; i++) {
            printf("    Stage %d: offset=%d\n", i, fft->stage_twiddle_offset[i]);
        }
    }
    
    // Test execution
    fft_data *input = (fft_data *)calloc(N, sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    input[0].re = 1.0;  // Impulse
    
    printf("\nExecuting FFT...\n");
    fft_exec(fft, input, output);
    
    // Verify result (should be all 1.0 for impulse)
    int correct = 1;
    for (int k = 0; k < N; k++) {
        if (fabs(output[k].re - 1.0) > 1e-9 || fabs(output[k].im) > 1e-9) {
            correct = 0;
            break;
        }
    }
    
    printf("Result: %s\n", correct ? "CORRECT ✓" : "INCORRECT ✗");
    
    free_fft(fft);
    free(input);
    free(output);
}

void test_n5_inverse(void) {
    printf("\n=== N=5 Inverse FFT Test ===\n");
    fft_data input[5] = {{1,0}, {2,0}, {3,0}, {4,0}, {5,0}};
    fft_data freq[5], recon[5];
    
    fft_object fwd = fft_init(5, 1);  // forward
    fft_object inv = fft_init(5, -1); // inverse
    
    fft_exec(fwd, input, freq);
    fft_exec(inv, freq, recon);
    
    // Scale by 1/N
    for (int i = 0; i < 5; i++) {
        recon[i].re /= 5;
        recon[i].im /= 5;
    }
    
    printf("Reconstruction errors:\n");
    for (int i = 0; i < 5; i++) {
        double err = sqrt(pow(input[i].re - recon[i].re, 2) + 
                         pow(input[i].im - recon[i].im, 2));
        printf("  [%d]: %.3e %s\n", i, err, (err < 1e-10) ? "✓" : "✗");
    }
    
    free_fft(fwd);
    free_fft(inv);
}

void test_n64_forward(void) {
    printf("\n=== N=64 Forward FFT Test ===\n");
    fft_data input[64], output[64];
    
    // Impulse
    for (int i = 0; i < 64; i++) {
        input[i].re = (i == 0) ? 1.0 : 0.0;
        input[i].im = 0.0;
    }
    
    fft_object fwd = fft_init(64, 1);
    fft_exec(fwd, input, output);
    
    printf("Impulse FFT (should be all 1+0i):\n");
    int errors = 0;
    for (int k = 0; k < 64; k++) {
        double err = sqrt(pow(output[k].re - 1.0, 2) + pow(output[k].im, 2));
        if (err > 1e-10) {
            printf("  X[%2d]: (%.6f, %.6f) error=%.3e ✗\n", 
                   k, output[k].re, output[k].im, err);
            errors++;
            if (errors > 10) {
                printf("  ... (showing first 10 errors)\n");
                break;
            }
        }
    }
    
    if (errors == 0) printf("All correct! ✓\n");
    free_fft(fwd);
}

void debug_n64(void) {
    printf("\n=== N=64 Debug ===\n");
    
    fft_object fwd = fft_init(64, 1);
    
    printf("N=64 factorization:\n");
    printf("  lf (num factors): %d\n", fwd->lf);
    printf("  factors: ");
    for (int i = 0; i < fwd->lf; i++) {
        printf("%d ", fwd->factors[i]);
    }
    printf("\n");
    
    // Test with simple input
    fft_data input[64], output[64];
    for (int i = 0; i < 64; i++) {
        input[i].re = (i == 0) ? 1.0 : 0.0;
        input[i].im = 0.0;
    }
    
    fft_exec(fwd, input, output);
    
    printf("\nImpulse response errors:\n");
    for (int k = 0; k < 64; k++) {
        double err = sqrt(pow(output[k].re - 1.0, 2) + pow(output[k].im, 2));
        if (err > 1e-10) {
            printf("  X[%2d] = (%8.5f, %8.5f) err=%.3e\n", 
                   k, output[k].re, output[k].im, err);
        }
    }
    
    free_fft(fwd);
}

void test_power_of_2_sizes(void) {
    printf("\n=== Power-of-2 Sizes Test ===\n");
    
    int sizes[] = {8, 16, 32, 64, 128, 256};
    
    for (int i = 0; i < 6; i++) {
        int N = sizes[i];
        fft_data *input = malloc(N * sizeof(fft_data));
        fft_data *freq = malloc(N * sizeof(fft_data));
        fft_data *recon = malloc(N * sizeof(fft_data));
        
        // Complex exponential input
        for (int n = 0; n < N; n++) {
            input[n].re = cos(2.0 * M_PI * n / N);
            input[n].im = sin(2.0 * M_PI * n / N);
        }
        
        fft_object fwd = fft_init(N, 1);
        fft_object inv = fft_init(N, -1);
        
        fft_exec(fwd, input, freq);
        fft_exec(inv, freq, recon);
        
        for (int n = 0; n < N; n++) {
            recon[n].re /= N;
            recon[n].im /= N;
        }
        
        double mse = 0;
        for (int n = 0; n < N; n++) {
            double err_re = input[n].re - recon[n].re;
            double err_im = input[n].im - recon[n].im;
            mse += err_re * err_re + err_im * err_im;
        }
        mse /= N;
        
        printf("  N=%4d: MSE=%.3e %s (factors: ", N, mse, (mse < 1e-10) ? "✓" : "✗");
        for (int j = 0; j < fwd->lf; j++) {
            printf("%d ", fwd->factors[j]);
        }
        printf(")\n");
        
        free_fft(fwd);
        free_fft(inv);
        free(input);
        free(freq);
        free(recon);
    }
}

void test_radix2_only(void) {
    printf("\n=== Testing Pure Radix-2 FFTs ===\n");
    
    int sizes[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    
    for (int i = 0; i < 10; i++) {
        int N = sizes[i];
        
        // Allocate
        fft_data *input = (fft_data*)_mm_malloc(N * sizeof(fft_data), 32);
        fft_data *output = (fft_data*)_mm_malloc(N * sizeof(fft_data), 32);
        fft_data *reconstructed = (fft_data*)_mm_malloc(N * sizeof(fft_data), 32);
        
        // Create test signal: simple impulse
        for (int j = 0; j < N; j++) {
            input[j].re = (j == 0) ? 1.0 : 0.0;
            input[j].im = 0.0;
        }
        
        // Forward FFT (sgn = +1)
        fft_object fwd = fft_init(N, +1);
        printf("N=%4d Forward: sgn=%+d, twiddles[1]=(%.6f, %.6f)\n",
               N, fwd->sgn, fwd->twiddles[1].re, fwd->twiddles[1].im);
        fft_exec(fwd, input, output);
        
        // Inverse FFT (sgn = -1)
        fft_object inv = fft_init(N, -1);
        printf("N=%4d Inverse: sgn=%+d, twiddles[1]=(%.6f, %.6f)\n",
               N, inv->sgn, inv->twiddles[1].re, inv->twiddles[1].im);
        fft_exec(inv, output, reconstructed);
        
        // Scale by 1/N
        for (int j = 0; j < N; j++) {
            reconstructed[j].re /= N;
            reconstructed[j].im /= N;
        }
        
        // Check error
        double max_err = 0.0;
        for (int j = 0; j < N; j++) {
            double err_re = fabs(reconstructed[j].re - input[j].re);
            double err_im = fabs(reconstructed[j].im - input[j].im);
            max_err = fmax(max_err, fmax(err_re, err_im));
        }
        
        printf("N=%4d: Max error = %.6e %s\n", 
               N, max_err, (max_err < 1e-10) ? "PASS" : "FAIL");
        
        // Check forward FFT result (impulse -> all ones)
        printf("  Forward output[0] = (%.6f, %.6f) [expect (1, 0)]\n",
               output[0].re, output[0].im);
        printf("  Forward output[1] = (%.6f, %.6f) [expect (1, 0)]\n",
               output[1].re, output[1].im);
        
        free_fft(fwd);
        free_fft(inv);
        _mm_free(input);
        _mm_free(output);
        _mm_free(reconstructed);
    }
}

void verify_twiddle_convention(void) {
    printf("\n=== Verifying Twiddle Convention ===\n");
    
    // Test N=8
    fft_object fwd = fft_init(8, +1);
    fft_object inv = fft_init(8, -1);
    
    printf("N=8 Forward FFT (sgn=+1):\n");
    for (int k = 0; k < 8; k++) {
        printf("  W[%d] = (%+.6f, %+.6f)\n", 
               k, fwd->twiddles[k].re, fwd->twiddles[k].im);
    }
    
    printf("\nN=8 Inverse FFT (sgn=-1):\n");
    for (int k = 0; k < 8; k++) {
        printf("  W[%d] = (%+.6f, %+.6f)\n", 
               k, inv->twiddles[k].re, inv->twiddles[k].im);
    }
    
    // Check: Forward W[1] should be exp(-2πi/8) = (cos(-π/4), sin(-π/4))
    //        = (0.707, -0.707)
    printf("\nExpected for Forward:\n");
    printf("  W[1] = (+0.707107, -0.707107)\n");
    printf("  W[2] = (+0.000000, -1.000000)\n");
    printf("  W[3] = (-0.707107, -0.707107)\n");
    
    printf("\nExpected for Inverse (conjugated):\n");
    printf("  W[1] = (+0.707107, +0.707107)\n");
    printf("  W[2] = (+0.000000, +1.000000)\n");
    printf("  W[3] = (-0.707107, +0.707107)\n");
    
    free_fft(fwd);
    free_fft(inv);
}

void diagnose_n512(void) {
    printf("\n=== Diagnosing N=512 ===\n");
    
    fft_object fwd = fft_init(512, +1);
    
    printf("Factors: ");
    for (int i = 0; i < fwd->lf; i++) {
        printf("%d ", fwd->factors[i]);
    }
    printf("\n");
    
    printf("Number of precomputed stages: %d\n", fwd->num_precomputed_stages);
    
    if (fwd->twiddle_factors) {
        printf("Using precomputed twiddle_factors\n");
        printf("Stage offsets:\n");
        for (int i = 0; i < fwd->num_precomputed_stages; i++) {
            printf("  Stage %d: offset=%d\n", i, fwd->stage_twiddle_offset[i]);
        }
    } else {
        printf("NOT using precomputed twiddle_factors (dynamic generation)\n");
    }
    
    printf("Max scratch size: %d\n", fwd->max_scratch_size);
    
    // Count expected stages
    int expected_stages = 0;
    for (int n = 512; n > 1; n /= 2) {
        expected_stages++;
    }
    printf("Expected stages (512->256->...->2->1): %d\n", expected_stages);
    
    free_fft(fwd);
}

void test_n512_after_fix(void) {
    printf("\n=== Testing N=512 After Radix-8 Fix ===\n");
    
    int N = 512;
    
    fft_data *input = (fft_data*)_mm_malloc(N * sizeof(fft_data), 32);
    fft_data *output = (fft_data*)_mm_malloc(N * sizeof(fft_data), 32);
    fft_data *reconstructed = (fft_data*)_mm_malloc(N * sizeof(fft_data), 32);
    
    // Impulse
    for (int j = 0; j < N; j++) {
        input[j].re = (j == 0) ? 1.0 : 0.0;
        input[j].im = 0.0;
    }
    
    // Forward
    fft_object fwd = fft_init(N, +1);
    printf("Forward FFT (sgn=%+d):\n", fwd->sgn);
    printf("  Decomposition: ");
    for (int i = 0; i < fwd->lf; i++) {
        printf("%d ", fwd->factors[i]);
    }
    printf("\n");
    
    fft_exec(fwd, input, output);
    
    printf("  DC bin: (%.10f, %.10f) [expect (1, 0)]\n", output[0].re, output[0].im);
    printf("  Bin 1:  (%.10f, %.10f) [expect (1, 0)]\n", output[1].re, output[1].im);
    printf("  Bin 2:  (%.10f, %.10f) [expect (1, 0)]\n", output[2].re, output[2].im);
    
    // Check all bins should be (1, 0) for impulse input
    int forward_pass = 1;
    for (int j = 0; j < N; j++) {
        if (fabs(output[j].re - 1.0) > 1e-10 || fabs(output[j].im) > 1e-10) {
            forward_pass = 0;
            printf("  ERROR at bin %d: (%.10f, %.10f)\n", j, output[j].re, output[j].im);
            if (j > 10) {
                printf("  ... (stopping after first 10 errors)\n");
                break;
            }
        }
    }
    printf("Forward FFT: %s\n\n", forward_pass ? "PASS" : "FAIL");
    
    // Inverse
    fft_object inv = fft_init(N, -1);
    printf("Inverse FFT (sgn=%+d):\n", inv->sgn);
    fft_exec(inv, output, reconstructed);
    
    // Scale
    for (int j = 0; j < N; j++) {
        reconstructed[j].re /= N;
        reconstructed[j].im /= N;
    }
    
    printf("  Reconstructed[0]: (%.10f, %.10f) [expect (1, 0)]\n", 
           reconstructed[0].re, reconstructed[0].im);
    printf("  Reconstructed[1]: (%.10f, %.10f) [expect (0, 0)]\n", 
           reconstructed[1].re, reconstructed[1].im);
    
    // Check
    double max_err = 0.0;
    int error_count = 0;
    for (int j = 0; j < N; j++) {
        double err_re = fabs(reconstructed[j].re - input[j].re);
        double err_im = fabs(reconstructed[j].im - input[j].im);
        double err = fmax(err_re, err_im);
        if (err > 1e-10) {
            error_count++;
            if (error_count <= 5) {
                printf("  ERROR at [%d]: got (%.10f, %.10f), expect (%.10f, %.10f)\n",
                       j, reconstructed[j].re, reconstructed[j].im, input[j].re, input[j].im);
            }
        }
        max_err = fmax(max_err, err);
    }
    
    printf("\nRound-trip: Max error = %.6e %s\n", 
           max_err, (max_err < 1e-10) ? "PASS" : "FAIL");
    printf("  Total errors: %d / %d bins\n", error_count, N);
    
    free_fft(fwd);
    free_fft(inv);
    _mm_free(input);
    _mm_free(output);
    _mm_free(reconstructed);
}

void debug_n8(void) {
    printf("\n=== Debug N=8 ===\n");
    fft_object fwd = fft_init(8, +1);
    fft_object inv = fft_init(8, -1);
    
    printf("Forward decomposition: ");
    for (int i = 0; i < fwd->lf; i++) printf("%d ", fwd->factors[i]);
    printf("\n");
    
    printf("Inverse decomposition: ");
    for (int i = 0; i < inv->lf; i++) printf("%d ", inv->factors[i]);
    printf("\n");
    
    printf("Forward twiddles[1]: (%.6f, %.6f)\n", 
           fwd->twiddles[1].re, fwd->twiddles[1].im);
    printf("Inverse twiddles[1]: (%.6f, %.6f)\n",
           inv->twiddles[1].re, inv->twiddles[1].im);
    
    free_fft(fwd);
    free_fft(inv);
}

void test_n8_detailed(void) {
    printf("\n=== Detailed N=8 Test ===\n");
    
    int N = 8;
    fft_data *input = (fft_data*)malloc(N * sizeof(fft_data));
    fft_data *fwd_out = (fft_data*)malloc(N * sizeof(fft_data));
    fft_data *inv_out = (fft_data*)malloc(N * sizeof(fft_data));
    
    // Simple test: [1, 0, 0, 0, 0, 0, 0, 0]
    for (int i = 0; i < N; i++) {
        input[i].re = (i == 0) ? 1.0 : 0.0;
        input[i].im = 0.0;
    }
    
    fft_object fwd = fft_init(N, +1);
    fft_object inv = fft_init(N, -1);
    
    printf("Input: [%.2f+%.2fi, ...]\n", input[0].re, input[0].im);
    
    // Forward FFT
    fft_exec(fwd, input, fwd_out);
    
    printf("Forward output (should all be 1+0i):\n");
    for (int i = 0; i < N; i++) {
        printf("  [%d] = %.6f + %.6fi\n", i, fwd_out[i].re, fwd_out[i].im);
    }
    
    // Inverse FFT
    fft_exec(inv, fwd_out, inv_out);
    
    printf("\nInverse output (before scaling):\n");
    for (int i = 0; i < N; i++) {
        printf("  [%d] = %.6f + %.6fi\n", i, inv_out[i].re, inv_out[i].im);
    }
    
    // Scale
    for (int i = 0; i < N; i++) {
        inv_out[i].re /= N;
        inv_out[i].im /= N;
    }
    
    printf("\nInverse output (after scaling, should be [1+0i, 0+0i, ...]):\n");
    for (int i = 0; i < N; i++) {
        printf("  [%d] = %.6f + %.6fi\n", i, inv_out[i].re, inv_out[i].im);
    }
    
    free_fft(fwd);
    free_fft(inv);
    free(input);
    free(fwd_out);
    free(inv_out);
}

void test_n8_final(void) {
    printf("\n=== Testing N=8 After Final Fix ===\n");
    
    int N = 8;
    fft_data *input = (fft_data*)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data*)malloc(N * sizeof(fft_data));
    fft_data *recon = (fft_data*)malloc(N * sizeof(fft_data));
    
    // Test signal with frequencies
    for (int i = 0; i < N; i++) {
        input[i].re = cos(2.0 * M_PI * i / N) + 0.5 * sin(2.0 * M_PI * 2 * i / N);
        input[i].im = sin(2.0 * M_PI * i / N) - 0.3 * cos(2.0 * M_PI * 2 * i / N);
    }
    
    fft_object fwd = fft_init(N, +1);
    fft_object inv = fft_init(N, -1);
    
    fft_exec(fwd, input, output);
    fft_exec(inv, output, recon);
    
    // Scale
    for (int i = 0; i < N; i++) {
        recon[i].re /= N;
        recon[i].im /= N;
    }
    
    // Check error
    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
        double err_re = fabs(recon[i].re - input[i].re);
        double err_im = fabs(recon[i].im - input[i].im);
        max_err = fmax(max_err, fmax(err_re, err_im));
    }
    
    printf("N=8: Max error = %.6e %s\n", 
           max_err, (max_err < 1e-10) ? "✓ PASS" : "✗ FAIL");
    
    if (max_err >= 1e-10) {
        printf("First 4 errors:\n");
        for (int i = 0; i < 4; i++) {
            printf("  [%d] err=(%.3e, %.3e)\n", i,
                   fabs(recon[i].re - input[i].re),
                   fabs(recon[i].im - input[i].im));
        }
    }
    
    free_fft(fwd);
    free_fft(inv);
    free(input);
    free(output);
    free(recon);
}

void debug_stage_twiddles_n8(void) {
    printf("\n=== Debug Stage Twiddles for N=8 ===\n");
    
    fft_object fwd = fft_init(8, +1);
    fft_object inv = fft_init(8, -1);
    
    printf("Forward (sgn=+1):\n");
    if (fwd->twiddle_factors) {
        printf("  Using precomputed twiddle_factors\n");
        for (int i = 0; i < 7; i++) {
            printf("  stage_tw[%d] = (%.6f, %.6f)\n", 
                   i, fwd->twiddle_factors[i].re, fwd->twiddle_factors[i].im);
        }
    } else {
        printf("  NO precomputed twiddles (would generate dynamically)\n");
    }
    
    printf("\nInverse (sgn=-1):\n");
    if (inv->twiddle_factors) {
        printf("  Using precomputed twiddle_factors\n");
        for (int i = 0; i < 7; i++) {
            printf("  stage_tw[%d] = (%.6f, %.6f)\n", 
                   i, inv->twiddle_factors[i].re, inv->twiddle_factors[i].im);
        }
    } else {
        printf("  NO precomputed twiddles (would generate dynamically)\n");
    }
    
    free_fft(fwd);
    free_fft(inv);
}

void test_n8_final_with_fixed_twiddles(void) {
    printf("\n=== Testing N=8 After Twiddle Fix ===\n");
    
    int N = 8;
    fft_data *input = (fft_data*)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data*)malloc(N * sizeof(fft_data));
    fft_data *recon = (fft_data*)malloc(N * sizeof(fft_data));
    
    // Test signal
    for (int i = 0; i < N; i++) {
        input[i].re = cos(2.0 * M_PI * i / N) + 0.5 * sin(2.0 * M_PI * 2 * i / N);
        input[i].im = sin(2.0 * M_PI * i / N) - 0.3 * cos(2.0 * M_PI * 2 * i / N);
    }
    
    fft_object fwd = fft_init(N, +1);
    fft_object inv = fft_init(N, -1);
    
    fft_exec(fwd, input, output);
    fft_exec(inv, output, recon);
    
    for (int i = 0; i < N; i++) {
        recon[i].re /= N;
        recon[i].im /= N;
    }
    
    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
        double err = fmax(fabs(recon[i].re - input[i].re),
                         fabs(recon[i].im - input[i].im));
        max_err = fmax(max_err, err);
    }
    
    printf("N=8: Max error = %.6e %s\n", 
           max_err, (max_err < 1e-10) ? "✓ PASS" : "✗ FAIL");
    
    free_fft(fwd);
    free_fft(inv);
    free(input);
    free(output);
    free(recon);
}

void debug_radix8_step_by_step(void) {
    printf("\n=== Debug Radix-8 Step-by-Step ===\n");
    
    int N = 8;
    fft_data *input = (fft_data*)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data*)malloc(N * sizeof(fft_data));
    
    // Simple impulse: [1, 0, 0, 0, 0, 0, 0, 0]
    for (int i = 0; i < N; i++) {
        input[i].re = (i == 0) ? 1.0 : 0.0;
        input[i].im = 0.0;
    }
    
    printf("Input (impulse):\n");
    for (int i = 0; i < N; i++) {
        printf("  [%d] = %.3f + %.3fi\n", i, input[i].re, input[i].im);
    }
    
    // Forward FFT
    fft_object fwd = fft_init(N, +1);
    
    printf("\nForward FFT internals:\n");
    printf("  Decomposition: ");
    for (int i = 0; i < fwd->lf; i++) printf("%d ", fwd->factors[i]);
    printf("\n");
    printf("  Transform sign: %d\n", fwd->sgn);
    
    // Manually call radix-8 butterfly
    const int K = 1;  // sub_len for single radix-8
    const int s = fwd->sgn;  // +1 for forward
    
    printf("\n  Inside radix-8 butterfly:\n");
    printf("    K = %d, N = %d, s = %d\n", K, 8, s);
    
    // Check W_base values
    const double c8 = 0.7071067811865476;
    printf("    W_base[0] (W_8^1) = (%.6f, %.6f) [expect (0.707, -0.707) for forward]\n",
           c8, -c8 * s);
    printf("    W_base[1] (W_8^2) = (%.6f, %.6f) [expect (0.0, -1.0) for forward]\n",
           0.0, -1.0 * s);
    
    // Check rot_sign
    const int rot_sign = -s;
    printf("    rot_sign = %d [expect -1 for forward]\n", rot_sign);
    
    fft_exec(fwd, input, output);
    
    printf("\nForward FFT output (should all be 1+0i):\n");
    int errors = 0;
    for (int i = 0; i < N; i++) {
        printf("  [%d] = %.6f + %.6fi", i, output[i].re, output[i].im);
        if (fabs(output[i].re - 1.0) > 1e-10 || fabs(output[i].im) > 1e-10) {
            printf(" ✗ ERROR");
            errors++;
        } else {
            printf(" ✓");
        }
        printf("\n");
    }
    
    printf("\nTotal errors: %d / %d\n", errors, N);
    
    // Now test inverse
    printf("\n=== Testing Inverse ===\n");
    
    fft_object inv = fft_init(N, -1);
    fft_data *recon = (fft_data*)malloc(N * sizeof(fft_data));
    
    fft_exec(inv, output, recon);
    
    printf("\nInverse FFT output (before scaling):\n");
    for (int i = 0; i < N; i++) {
        printf("  [%d] = %.6f + %.6fi\n", i, recon[i].re, recon[i].im);
    }
    
    // Scale
    for (int i = 0; i < N; i++) {
        recon[i].re /= N;
        recon[i].im /= N;
    }
    
    printf("\nInverse FFT output (after scaling, should be impulse):\n");
    errors = 0;
    for (int i = 0; i < N; i++) {
        double expect_re = (i == 0) ? 1.0 : 0.0;
        printf("  [%d] = %.6f + %.6fi", i, recon[i].re, recon[i].im);
        if (fabs(recon[i].re - expect_re) > 1e-10 || fabs(recon[i].im) > 1e-10) {
            printf(" ✗ ERROR (expect %.1f + 0i)", expect_re);
            errors++;
        } else {
            printf(" ✓");
        }
        printf("\n");
    }
    
    printf("\nTotal errors: %d / %d\n", errors, N);
    
    free_fft(fwd);
    free_fft(inv);
    free(input);
    free(output);
    free(recon);
}

void test_n8_complex_signal(void) {
    printf("\n=== Testing N=8 with Complex Signal ===\n");
    
    int N = 8;
    fft_data *input = (fft_data*)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data*)malloc(N * sizeof(fft_data));
    fft_data *recon = (fft_data*)malloc(N * sizeof(fft_data));
    
    // Complex signal with frequencies (like your test)
    for (int i = 0; i < N; i++) {
        input[i].re = cos(2.0 * M_PI * i / N) + 0.5 * sin(2.0 * M_PI * 2 * i / N);
        input[i].im = sin(2.0 * M_PI * i / N) - 0.3 * cos(2.0 * M_PI * 2 * i / N);
    }
    
    printf("Input signal:\n");
    for (int i = 0; i < 4; i++) {
        printf("  [%d] = %.6f + %.6fi\n", i, input[i].re, input[i].im);
    }
    printf("  ...\n");
    
    fft_object fwd = fft_init(N, +1);
    fft_object inv = fft_init(N, -1);
    
    fft_exec(fwd, input, output);
    fft_exec(inv, output, recon);
    
    // Scale
    for (int i = 0; i < N; i++) {
        recon[i].re /= N;
        recon[i].im /= N;
    }
    
    printf("\nReconstruction:\n");
    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
        double err_re = fabs(recon[i].re - input[i].re);
        double err_im = fabs(recon[i].im - input[i].im);
        double err = fmax(err_re, err_im);
        max_err = fmax(max_err, err);
        
        if (i < 4 || err > 1e-10) {
            printf("  [%d] orig=(%.6f, %.6f) recon=(%.6f, %.6f) err=%.3e\n",
                   i, input[i].re, input[i].im, recon[i].re, recon[i].im, err);
        }
    }
    
    printf("\nMax error: %.6e %s\n", 
           max_err, (max_err < 1e-10) ? "✓ PASS" : "✗ FAIL");
    
    free_fft(fwd);
    free_fft(inv);
    free(input);
    free(output);
    free(recon);
}

#include <time.h>

//==============================================================================
// INTERNAL HELPER FUNCTIONS
//==============================================================================

static void generate_test_signal(fft_data *signal, int N, int test_type) {
    switch(test_type) {
        case 0: // Impulse
            for (int i = 0; i < N; i++) {
                signal[i].re = (i == 0) ? 1.0 : 0.0;
                signal[i].im = 0.0;
            }
            break;
        case 1: // Sine wave (frequency = 1)
            for (int i = 0; i < N; i++) {
                signal[i].re = sin(2.0 * M_PI * i / N);
                signal[i].im = 0.0;
            }
            break;
        case 2: // Random
            for (int i = 0; i < N; i++) {
                signal[i].re = (double)rand() / RAND_MAX - 0.5;
                signal[i].im = (double)rand() / RAND_MAX - 0.5;
            }
            break;
    }
}

static double compute_max_error(fft_data *result, fft_data *expected, int N) {
    double max_err = 0.0;
    for (int i = 0; i < N; i++) {
        double err_re = fabs(result[i].re - expected[i].re);
        double err_im = fabs(result[i].im - expected[i].im);
        double err = sqrt(err_re * err_re + err_im * err_im);
        if (err > max_err) max_err = err;
    }
    return max_err;
}

static void naive_dft(fft_data *input, fft_data *output, int N, int sign) {
    for (int k = 0; k < N; k++) {
        double sum_re = 0.0, sum_im = 0.0;
        for (int n = 0; n < N; n++) {
            double angle = -sign * 2.0 * M_PI * k * n / N;
            double wr = cos(angle);
            double wi = sin(angle);
            sum_re += input[n].re * wr - input[n].im * wi;
            sum_im += input[n].re * wi + input[n].im * wr;
        }
        output[k].re = sum_re;
        output[k].im = sum_im;
    }
}

static double benchmark_fft(fft_object plan, fft_data *input, fft_data *output, int iterations) {
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        fft_exec(plan, input, output);
    }
    clock_t end = clock();
    return (double)(end - start) / CLOCKS_PER_SEC / iterations;
}

static void print_factorization(int N) {
    int factors[32];
    int num_factors = 0;
    int temp = N;
    
    // Count factors of 2 first
    int count_2 = 0;
    while (temp % 2 == 0) {
        count_2++;
        temp /= 2;
    }
    
    // Rebuild temp
    temp = N;
    
    // Now safely extract radix-8 (needs at least 3 factors of 2)
    while (count_2 >= 3 && temp % 8 == 0) {
        factors[num_factors++] = 8;
        temp /= 8;
        count_2 -= 3;
    }
    
    // Radix-4 (needs at least 2 factors of 2)
    while (count_2 >= 2 && temp % 4 == 0) {
        factors[num_factors++] = 4;
        temp /= 4;
        count_2 -= 2;
    }
    
    // Radix-2 (needs at least 1 factor of 2)
    while (count_2 >= 1 && temp % 2 == 0) {
        factors[num_factors++] = 2;
        temp /= 2;
        count_2--;
    }
    
    // Any remaining prime factor
    if (temp > 1) {
        factors[num_factors++] = temp;
    }
    
    // Print factorization
    for (int i = 0; i < num_factors; i++) {
        printf("%d", factors[i]);
        if (i < num_factors - 1) printf(" x ");
    }
}

static int test_single_size(int N) {
    const char* test_names[] = {"Impulse", "Sine Wave", "Random"};
    int all_passed = 1;
    
    printf("-------------------------------------------------------------\n");
    printf("Testing N = %d (", N);
    print_factorization(N);
    printf(")\n");
    
    // Allocate buffers
    fft_data *input = (fft_data*)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data*)malloc(N * sizeof(fft_data));
    fft_data *expected = (fft_data*)malloc(N * sizeof(fft_data));
    fft_data *roundtrip = (fft_data*)malloc(N * sizeof(fft_data));
    
    if (!input || !output || !expected || !roundtrip) {
        printf("[FAIL] Memory allocation error\n");
        free(input); free(output); free(expected); free(roundtrip);
        return 0;
    }
    
    // Create FFT plans
    fft_object fwd_plan = fft_init(N, 1);
    fft_object inv_plan = fft_init(N, -1);
    
    if (!fwd_plan || !inv_plan) {
        printf("[FAIL] FFT plan creation error\n");
        if (fwd_plan) free_fft(fwd_plan);
        if (inv_plan) free_fft(inv_plan);
        free(input); free(output); free(expected); free(roundtrip);
        return 0;
    }
    
    // Run tests for each signal type
    for (int test_type = 0; test_type < 3; test_type++) {
        generate_test_signal(input, N, test_type);
        
        // Accuracy test (only for small N)
        if (N <= 512) {
            naive_dft(input, expected, N, 1);
            fft_exec(fwd_plan, input, output);
            
            double error = compute_max_error(output, expected, N);
            double tolerance = 1e-10 * N;
            
            if (error < tolerance) {
                printf("  [PASS] %s: max error = %.2e\n", 
                       test_names[test_type], error);
            } else {
                printf("  [FAIL] %s: max error = %.2e (tolerance = %.2e)\n", 
                       test_names[test_type], error, tolerance);
                all_passed = 0;
            }
        }
        
        // Roundtrip test (FFT -> IFFT)
        fft_exec(fwd_plan, input, output);
        fft_exec(inv_plan, output, roundtrip);
        
        for (int i = 0; i < N; i++) {
            roundtrip[i].re /= N;
            roundtrip[i].im /= N;
        }
        
        double roundtrip_error = compute_max_error(roundtrip, input, N);
        double rt_tolerance = 1e-10 * N;
        
        if (roundtrip_error < rt_tolerance) {
            printf("  [PASS] %s (Roundtrip): error = %.2e\n", 
                   test_names[test_type], roundtrip_error);
        } else {
            printf("  [FAIL] %s (Roundtrip): error = %.2e (tolerance = %.2e)\n", 
                   test_names[test_type], roundtrip_error, rt_tolerance);
            all_passed = 0;
        }
    }
    
    // Benchmark
    if (N >= 512) {
        int iterations = (N <= 4096) ? 10000 : 1000;
        double time = benchmark_fft(fwd_plan, input, output, iterations);
        double mflops = (5.0 * N * log2(N) / time) / 1e6;
        
        printf("  [PERF] %.3f microseconds/FFT (%.1f MFLOPS)\n", 
               time * 1e6, mflops);
    }
    
    // Cleanup
    free_fft(fwd_plan);
    free_fft(inv_plan);
    free(input);
    free(output);
    free(expected);
    free(roundtrip);
    
    return all_passed;
}

//==============================================================================
// PUBLIC API - Call this from your test suite main()
//==============================================================================

int test_radix8_suite(void) {
    printf("=============================================================\n");
    printf("RADIX-8 FFT OPTIMIZATION TEST SUITE\n");
    printf("=============================================================\n\n");
    
    // Test sizes: powers of 8 and combinations
    int test_sizes[] = {8, 64, 512, 4096, 8*8*8, 8*8*8*8, 16384};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);
    
    int all_passed = 1;
    
    for (int i = 0; i < num_tests; i++) {
        if (!test_single_size(test_sizes[i])) {
            all_passed = 0;
        }
    }
    
    printf("-------------------------------------------------------------\n");
    if (all_passed) {
        printf("*** ALL RADIX-8 TESTS PASSED ***\n");
        return 0;
    } else {
        printf("*** SOME RADIX-8 TESTS FAILED ***\n");
        return 1;
    }
}

int main()
{

    //debug_radix_7_execution();
    //debug_fft_scaling();
    //debug_radix_selection();
    //debug_parseval_detailed();
    run_comprehensive_complex_fft_tests();
    //test_n8_final_with_fixed_twiddles();
    //test_n8_complex_signal();
    //test_n8_final();
    //debug_stage_twiddles_n8();
    //test_n8_detailed();
    //test_radix2_only(); 
    //verify_twiddle_convention();
    //diagnose_n512();
    //test_n512_after_fix();
    //test_radix8_suite();
    
    //run_comprehensive_complex_fft_N32_tests(); int all_passed = true;
    //test_n5_inverse(); int all_passed = true;
   

    //test_n64_forward(); int all_passed = true;
    //debug_n64(); int all_passed = true;
    //test_power_of_2_sizes();  int all_passed = true;

    printf("\n=== All Tests Complete ===\n");
    return  EXIT_SUCCESS;
}