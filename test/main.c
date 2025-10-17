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


#include <time.h>




void test_everything()
{
    const int K = 5;
    const int N = 3 * K; // 15
    const int transform_sign = 1;
    const double C_HALF = -0.5;
    const double S_SQRT3_2 = 0.8660254037844386467618;
    const double base_angle = -2.0 * M_PI / N * transform_sign;
    
    printf("====================================\n");
    printf("Complete Radix-3 Test for N=%d, K=%d\n", N, K);
    printf("====================================\n\n");
    
    // Create test input
    fft_data input[15];
    for (int i = 0; i < 15; i++)
    {
        input[i].re = i + 1.0;
        input[i].im = 0.0;
    }
    
    printf("Input (lane-major layout):\n");
    printf("Lane 0 (j=0): ");
    for (int k = 0; k < K; k++) printf("%.1f ", input[k].re);
    printf("\nLane 1 (j=1): ");
    for (int k = 0; k < K; k++) printf("%.1f ", input[k+K].re);
    printf("\nLane 2 (j=2): ");
    for (int k = 0; k < K; k++) printf("%.1f ", input[k+2*K].re);
    printf("\n\n");
    
    // Compute base twiddles
    fft_data W_base[2];
    for (int j = 1; j <= 2; j++)
    {
        double angle = base_angle * j;
        W_base[j-1].re = cos(angle);
        W_base[j-1].im = sin(angle);
    }
    
    printf("Base twiddles:\n");
    printf("  W_base[0] = W_15^1 = %.6f + %.6fi\n", W_base[0].re, W_base[0].im);
    printf("  W_base[1] = W_15^2 = %.6f + %.6fi\n\n", W_base[1].re, W_base[1].im);
    
    // Reference computation
    fft_data output[15];
    fft_data W_curr[2];
    W_curr[0].re = 1.0;
    W_curr[0].im = 0.0;
    W_curr[1].re = 1.0;
    W_curr[1].im = 0.0;
    
    printf("Processing each butterfly k:\n");
    printf("====================================\n");
    
    for (int k = 0; k < K; k++)
    {
        fft_data a = input[k];
        fft_data b = input[k + K];
        fft_data c = input[k + 2*K];
        
        printf("\nk=%d:\n", k);
        printf("  Input: a=%.1f, b=%.1f, c=%.1f\n", a.re, b.re, c.re);
        printf("  Twiddles: W_curr[0]=(%.6f,%.6fi), W_curr[1]=(%.6f,%.6fi)\n",
               W_curr[0].re, W_curr[0].im, W_curr[1].re, W_curr[1].im);
        
        // Apply twiddles
        double b2r = b.re * W_curr[0].re - b.im * W_curr[0].im;
        double b2i = b.re * W_curr[0].im + b.im * W_curr[0].re;
        
        double c2r = c.re * W_curr[1].re - c.im * W_curr[1].im;
        double c2i = c.re * W_curr[1].im + c.im * W_curr[1].re;
        
        printf("  After twiddle: b2=(%.4f,%.4fi), c2=(%.4f,%.4fi)\n", b2r, b2i, c2r, c2i);
        
        // Radix-3 butterfly
        double sumr = b2r + c2r;
        double sumi = b2i + c2i;
        double difr = b2r - c2r;
        double difi = b2i - c2i;
        
        output[k].re = a.re + sumr;
        output[k].im = a.im + sumi;
        
        double commonr = a.re + C_HALF * sumr;
        double commoni = a.im + C_HALF * sumi;
        
        double scaled_rotr = S_SQRT3_2 * difi;
        double scaled_roti = -S_SQRT3_2 * difr;
        
        output[k + K].re = commonr + scaled_rotr;
        output[k + K].im = commoni + scaled_roti;
        output[k + 2*K].re = commonr - scaled_rotr;
        output[k + 2*K].im = commoni - scaled_roti;
        
        printf("  Output: y0=(%.4f,%.4fi), y1=(%.4f,%.4fi), y2=(%.4f,%.4fi)\n",
               output[k].re, output[k].im,
               output[k+K].re, output[k+K].im,
               output[k+2*K].re, output[k+2*K].im);
        
        // Update twiddles for next k
        for (int j = 0; j < 2; j++)
        {
            double re = W_curr[j].re * W_base[j].re - W_curr[j].im * W_base[j].im;
            double im = W_curr[j].re * W_base[j].im + W_curr[j].im * W_base[j].re;
            W_curr[j].re = re;
            W_curr[j].im = im;
        }
    }
    
    printf("\n====================================\n");
    printf("Final Output (lane-major):\n");
    printf("====================================\n");
    printf("Lane 0: ");
    for (int k = 0; k < K; k++) printf("(%.2f,%.2fi) ", output[k].re, output[k].im);
    printf("\nLane 1: ");
    for (int k = 0; k < K; k++) printf("(%.2f,%.2fi) ", output[k+K].re, output[k+K].im);
    printf("\nLane 2: ");
    for (int k = 0; k < K; k++) printf("(%.2f,%.2fi) ", output[k+2*K].re, output[k+2*K].im);
    printf("\n\n");
    
    // Now verify forward+inverse gets back original
    printf("====================================\n");
    printf("Verification: Forward + Inverse FFT\n");
    printf("====================================\n");
    
    fft_data inverse_input[15];
    for (int i = 0; i < 15; i++)
    {
        inverse_input[i] = output[i];
    }
    
    // Inverse FFT
    const int inv_sign = -1;
    const double inv_base_angle = -2.0 * M_PI / N * inv_sign;
    
    fft_data W_base_inv[2];
    for (int j = 1; j <= 2; j++)
    {
        double angle = inv_base_angle * j;
        W_base_inv[j-1].re = cos(angle);
        W_base_inv[j-1].im = sin(angle);
    }
    
    fft_data reconstructed[15];
    W_curr[0].re = 1.0;
    W_curr[0].im = 0.0;
    W_curr[1].re = 1.0;
    W_curr[1].im = 0.0;
    
    for (int k = 0; k < K; k++)
    {
        fft_data a = inverse_input[k];
        fft_data b = inverse_input[k + K];
        fft_data c = inverse_input[k + 2*K];
        
        // Apply twiddles
        double b2r = b.re * W_curr[0].re - b.im * W_curr[0].im;
        double b2i = b.re * W_curr[0].im + b.im * W_curr[0].re;
        
        double c2r = c.re * W_curr[1].re - c.im * W_curr[1].im;
        double c2i = c.re * W_curr[1].im + c.im * W_curr[1].re;
        
        // Radix-3 butterfly
        double sumr = b2r + c2r;
        double sumi = b2i + c2i;
        double difr = b2r - c2r;
        double difi = b2i - c2i;
        
        reconstructed[k].re = a.re + sumr;
        reconstructed[k].im = a.im + sumi;
        
        double commonr = a.re + C_HALF * sumr;
        double commoni = a.im + C_HALF * sumi;
        
        double scaled_rotr = -S_SQRT3_2 * difi; // Inverse uses opposite sign
        double scaled_roti = S_SQRT3_2 * difr;
        
        reconstructed[k + K].re = commonr + scaled_rotr;
        reconstructed[k + K].im = commoni + scaled_roti;
        reconstructed[k + 2*K].re = commonr - scaled_rotr;
        reconstructed[k + 2*K].im = commoni - scaled_roti;
        
        // Update twiddles for next k
        for (int j = 0; j < 2; j++)
        {
            double re = W_curr[j].re * W_base_inv[j].re - W_curr[j].im * W_base_inv[j].im;
            double im = W_curr[j].re * W_base_inv[j].im + W_curr[j].im * W_base_inv[j].re;
            W_curr[j].re = re;
            W_curr[j].im = im;
        }
    }
    
    // Scale by 1/3 (radix) for inverse
    for (int i = 0; i < 15; i++)
    {
        reconstructed[i].re /= 3.0;
        reconstructed[i].im /= 3.0;
    }
    
    printf("Reconstructed (should match original input):\n");
    printf("Lane 0: ");
    for (int k = 0; k < K; k++) printf("%.2f ", reconstructed[k].re);
    printf("\nLane 1: ");
    for (int k = 0; k < K; k++) printf("%.2f ", reconstructed[k+K].re);
    printf("\nLane 2: ");
    for (int k = 0; k < K; k++) printf("%.2f ", reconstructed[k+2*K].re);
    printf("\n\n");
    
    printf("Error check:\n");
    double max_error = 0.0;
    for (int i = 0; i < 15; i++)
    {
        double err_re = fabs(reconstructed[i].re - input[i].re);
        double err_im = fabs(reconstructed[i].im - input[i].im);
        if (err_re > max_error) max_error = err_re;
        if (err_im > max_error) max_error = err_im;
    }
    printf("Max error: %.6e\n", max_error);
    if (max_error < 1e-10)
    {
        printf("✓ PASS: Reconstruction matches input!\n");
    }
    else
    {
        printf("✗ FAIL: Reconstruction error too large!\n");
    }
}

// Tolerance for floating point comparisons
#define EPSILON 1e-10

// Helper function to compute DFT reference (slow but correct)
void dft_reference(const fft_data *input, fft_data *output, int N, int sign) {
    for (int k = 0; k < N; k++) {
        output[k].re = 0.0;
        output[k].im = 0.0;
        for (int n = 0; n < N; n++) {
            double angle = -2.0 * M_PI * k * n / N * sign;
            double cos_angle = cos(angle);
            double sin_angle = sin(angle);
            output[k].re += input[n].re * cos_angle - input[n].im * sin_angle;
            output[k].im += input[n].re * sin_angle + input[n].im * cos_angle;
        }
    }
}

// Helper to print complex array
void print_complex_array(const char *label, const fft_data *arr, int N) {
    printf("%s:\n", label);
    for (int i = 0; i < N; i++) {
        printf("  [%2d]: %8.4f + %8.4fi\n", i, arr[i].re, arr[i].im);
    }
    printf("\n");
}

// Helper to compute max error between two arrays
double compute_max_error(const fft_data *a, const fft_data *b, int N) {
    double max_error = 0.0;
    for (int i = 0; i < N; i++) {
        double err_re = fabs(a[i].re - b[i].re);
        double err_im = fabs(a[i].im - b[i].im);
        if (err_re > max_error) max_error = err_re;
        if (err_im > max_error) max_error = err_im;
    }
    return max_error;
}

// Test 1: Basic round-trip test
void test_round_trip(void) {
    printf("========================================\n");
    printf("TEST 1: Round-Trip Test (Forward + Inverse)\n");
    printf("========================================\n\n");
    
    const int N = 15;
    
    // Allocate arrays
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *reconstructed = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Initialize input with simple pattern
    printf("Input signal:\n");
    for (int i = 0; i < N; i++) {
        input[i].re = i + 1.0;
        input[i].im = 0.0;
        printf("  x[%2d] = %.1f\n", i, input[i].re);
    }
    printf("\n");
    
    // Create FFT objects
    fft_object fft_forward = fft_init(N, 1);
    fft_object fft_inverse = fft_init(N, -1);
    
    if (!fft_forward || !fft_inverse) {
        printf("ERROR: Failed to initialize FFT objects\n");
        goto cleanup;
    }
    
    // Print factorization
    printf("Factorization: N = %d = ", N);
    for (int i = 0; i < fft_forward->lf; i++) {
        printf("%d", fft_forward->factors[i]);
        if (i < fft_forward->lf - 1) printf(" × ");
    }
    printf("\n\n");
    
    // Forward FFT
    printf("Performing forward FFT...\n");
    fft_exec(fft_forward, input, output);
    print_complex_array("Forward FFT output", output, N);
    
    // Inverse FFT
    printf("Performing inverse FFT...\n");
    fft_exec(fft_inverse, output, reconstructed);
    
    // Scale by 1/N for inverse
    printf("Scaling by 1/N = 1/%d...\n\n", N);
    for (int i = 0; i < N; i++) {
        reconstructed[i].re /= N;
        reconstructed[i].im /= N;
    }
    
    print_complex_array("Reconstructed signal", reconstructed, N);
    
    // Compute error
    double max_error = compute_max_error(input, reconstructed, N);
    printf("Maximum reconstruction error: %.6e\n", max_error);
    
    if (max_error < EPSILON) {
        printf("✓ PASS: Round-trip test successful!\n");
    } else {
        printf("✗ FAIL: Reconstruction error too large!\n");
    }
    
cleanup:
    if (fft_forward) free_fft(fft_forward);
    if (fft_inverse) free_fft(fft_inverse);
    free(input);
    free(output);
    free(reconstructed);
    printf("\n");
}

// Test 2: Compare with reference DFT
void test_against_reference(void) {
    printf("========================================\n");
    printf("TEST 2: Compare with Reference DFT\n");
    printf("========================================\n\n");
    
    const int N = 15;
    
    // Allocate arrays
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *fft_output = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *dft_output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Initialize with complex input
    printf("Complex input signal:\n");
    for (int i = 0; i < N; i++) {
        input[i].re = sin(2 * M_PI * i / N);
        input[i].im = cos(4 * M_PI * i / N);
        printf("  x[%2d] = %7.4f + %7.4fi\n", i, input[i].re, input[i].im);
    }
    printf("\n");
    
    // Create FFT object
    fft_object fft_forward = fft_init(N, 1);
    if (!fft_forward) {
        printf("ERROR: Failed to initialize FFT object\n");
        goto cleanup;
    }
    
    // Compute FFT
    printf("Computing FFT...\n");
    fft_exec(fft_forward, input, fft_output);
    
    // Compute reference DFT
    printf("Computing reference DFT...\n");
    dft_reference(input, dft_output, N, 1);
    
    // Compare outputs
    printf("\nComparison (showing first 5 and last 5):\n");
    printf("Index |      FFT Output       |    Reference DFT      | Error\n");
    printf("------+-----------------------+-----------------------+--------\n");
    
    for (int i = 0; i < N; i++) {
        if (i < 5 || i >= N - 5) {
            double err_re = fabs(fft_output[i].re - dft_output[i].re);
            double err_im = fabs(fft_output[i].im - dft_output[i].im);
            double err = sqrt(err_re * err_re + err_im * err_im);
            
            printf("  %2d  | %8.4f + %7.4fi | %8.4f + %7.4fi | %.2e\n",
                   i,
                   fft_output[i].re, fft_output[i].im,
                   dft_output[i].re, dft_output[i].im,
                   err);
        } else if (i == 5) {
            printf("  ... |         ...           |         ...           | ...\n");
        }
    }
    
    // Compute max error
    double max_error = compute_max_error(fft_output, dft_output, N);
    printf("\nMaximum error: %.6e\n", max_error);
    
    if (max_error < EPSILON) {
        printf("✓ PASS: FFT matches reference DFT!\n");
    } else {
        printf("✗ FAIL: FFT does not match reference DFT!\n");
    }
    
cleanup:
    if (fft_forward) free_fft(fft_forward);
    free(input);
    free(fft_output);
    free(dft_output);
    printf("\n");
}

// Test 3: Test specific frequency components
void test_frequency_components(void) {
    printf("========================================\n");
    printf("TEST 3: Frequency Component Test\n");
    printf("========================================\n\n");
    
    const int N = 15;
    
    // Test pure sinusoid at frequency 3
    printf("Testing pure sinusoid at frequency k=3\n");
    
    fft_data *input = (fft_data *)calloc(N, sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Create pure cosine at frequency 3
    for (int i = 0; i < N; i++) {
        input[i].re = cos(2 * M_PI * 3 * i / N);
        input[i].im = 0.0;
    }
    
    // Create FFT object
    fft_object fft_forward = fft_init(N, 1);
    if (!fft_forward) {
        printf("ERROR: Failed to initialize FFT object\n");
        goto cleanup;
    }
    
    // Compute FFT
    fft_exec(fft_forward, input, output);
    
    // Check spectrum
    printf("\nSpectrum magnitude:\n");
    for (int k = 0; k < N; k++) {
        double mag = sqrt(output[k].re * output[k].re + output[k].im * output[k].im);
        printf("  |X[%2d]| = %.4f", k, mag);
        
        // Expect peaks at k=3 and k=12 (N-3)
        if (k == 3 || k == 12) {
            printf(" <- Expected peak");
            if (mag < N/2 - 0.1 || mag > N/2 + 0.1) {
                printf(" (ERROR: Expected %.1f)", N/2.0);
            }
        } else {
            if (mag > 0.001) {
                printf(" <- Unexpected energy!");
            }
        }
        printf("\n");
    }
    
    // Verify expected peaks
    double mag3 = sqrt(output[3].re * output[3].re + output[3].im * output[3].im);
    double mag12 = sqrt(output[12].re * output[12].re + output[12].im * output[12].im);
    
    if (fabs(mag3 - N/2.0) < 0.1 && fabs(mag12 - N/2.0) < 0.1) {
        printf("\n✓ PASS: Frequency component test successful!\n");
    } else {
        printf("\n✗ FAIL: Frequency peaks not at expected values!\n");
    }
    
cleanup:
    if (fft_forward) free_fft(fft_forward);
    free(input);
    free(output);
    printf("\n");
}

// Test 4: Parseval's theorem (energy conservation)
void test_parsevals_theorem(void) {
    printf("========================================\n");
    printf("TEST 4: Parseval's Theorem Test\n");
    printf("========================================\n\n");
    
    const int N = 15;
    
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Random-ish input
    for (int i = 0; i < N; i++) {
        input[i].re = sin(i * 0.7) + cos(i * 1.3);
        input[i].im = sin(i * 1.1) - cos(i * 0.5);
    }
    
    // Create FFT object
    fft_object fft_forward = fft_init(N, 1);
    if (!fft_forward) {
        printf("ERROR: Failed to initialize FFT object\n");
        goto cleanup;
    }
    
    // Compute FFT
    fft_exec(fft_forward, input, output);
    
    // Compute time-domain energy
    double time_energy = 0.0;
    for (int i = 0; i < N; i++) {
        time_energy += input[i].re * input[i].re + input[i].im * input[i].im;
    }
    
    // Compute frequency-domain energy
    double freq_energy = 0.0;
    for (int k = 0; k < N; k++) {
        freq_energy += output[k].re * output[k].re + output[k].im * output[k].im;
    }
    freq_energy /= N; // Account for FFT scaling
    
    printf("Time-domain energy:      %.6f\n", time_energy);
    printf("Frequency-domain energy: %.6f (scaled by 1/N)\n", freq_energy);
    printf("Relative error:          %.6e\n", fabs(time_energy - freq_energy) / time_energy);
    
    if (fabs(time_energy - freq_energy) / time_energy < 1e-10) {
        printf("\n✓ PASS: Parseval's theorem satisfied!\n");
    } else {
        printf("\n✗ FAIL: Energy not conserved!\n");
    }
    
cleanup:
    if (fft_forward) free_fft(fft_forward);
    free(input);
    free(output);
    printf("\n");
}

// Test 5: Test different factorization orders (if your code supports it)
void test_factorization_invariance(void) {
    printf("========================================\n");
    printf("TEST 5: Factorization Invariance Test\n");
    printf("========================================\n\n");
    
    printf("Testing that result is independent of factorization order\n");
    printf("(This assumes your code can handle different factorizations)\n\n");
    
    const int N = 15;
    
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output1 = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output2 = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Complex input
    for (int i = 0; i < N; i++) {
        input[i].re = i * 0.5 - 3.0;
        input[i].im = (N - i) * 0.3;
    }
    
    // Method 1: Natural factorization (likely 3×5)
    fft_object fft1 = fft_init(N, 1);
    if (!fft1) {
        printf("ERROR: Failed to initialize FFT object 1\n");
        goto cleanup;
    }
    
    printf("Factorization 1: ");
    for (int i = 0; i < fft1->lf; i++) {
        printf("%d ", fft1->factors[i]);
    }
    printf("\n");
    
    fft_exec(fft1, input, output1);
    
    // Method 2: Use reference DFT as "different factorization"
    printf("Factorization 2: Reference DFT (no factorization)\n\n");
    dft_reference(input, output2, N, 1);
    
    // Compare outputs
    double max_error = compute_max_error(output1, output2, N);
    printf("Maximum difference between methods: %.6e\n", max_error);
    
    if (max_error < EPSILON) {
        printf("✓ PASS: Results are factorization-independent!\n");
    } else {
        printf("✗ FAIL: Different factorizations give different results!\n");
    }
    
cleanup:
    if (fft1) free_fft(fft1);
    free(input);
    free(output1);
    free(output2);
    printf("\n");
}

// This diagnostic test helps identify where the issue is
void debug_mixed_radix_stages(void) {
    printf("========================================\n");
    printf("DEBUG: Mixed-Radix Stage Analysis\n");
    printf("========================================\n\n");
    
    const int N = 15;
    
    // Create simple test input
    fft_data *input = (fft_data *)calloc(N, sizeof(fft_data));
    fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
    
    // Use impulse for easy debugging
    input[0].re = 1.0;
    input[0].im = 0.0;
    
    printf("Input: Impulse at position 0\n");
    printf("Expected output: All 1+0i\n\n");
    
    // Create FFT object and inspect factorization
    fft_object fft_forward = fft_init(N, 1);
    if (!fft_forward) {
        printf("ERROR: Failed to initialize FFT object\n");
        goto cleanup;
    }
    
    printf("FFT Configuration:\n");
    printf("  n_input: %d\n", fft_forward->n_input);
    printf("  n_fft: %d\n", fft_forward->n_fft);
    printf("  lt: %d (0=mixed-radix, 1=Bluestein)\n", fft_forward->lt);
    printf("  lf: %d (number of factors)\n", fft_forward->lf);
    printf("  factors: ");
    for (int i = 0; i < fft_forward->lf; i++) {
        printf("%d ", fft_forward->factors[i]);
    }
    printf("\n");
    printf("  sgn: %d\n", fft_forward->sgn);
    printf("  max_scratch_size: %d\n\n", fft_forward->max_scratch_size);
    
    // Execute FFT
    printf("Executing FFT...\n");
    fft_exec(fft_forward, input, output);
    
    // Check output
    printf("\nOutput (should all be 1+0i for impulse input):\n");
    int all_correct = 1;
    for (int i = 0; i < N; i++) {
        printf("  [%2d]: %8.4f + %8.4fi", i, output[i].re, output[i].im);
        
        double err_re = fabs(output[i].re - 1.0);
        double err_im = fabs(output[i].im - 0.0);
        
        if (err_re > 1e-10 || err_im > 1e-10) {
            printf(" <- ERROR!");
            all_correct = 0;
        }
        printf("\n");
    }
    
    if (all_correct) {
        printf("\n✓ PASS: Impulse response correct!\n");
    } else {
        printf("\n✗ FAIL: Impulse response incorrect!\n");
        printf("\nThis indicates an issue in the mixed-radix recursion.\n");
    }
    
    // Test with different patterns
    printf("\n----------------------------------------\n");
    printf("Testing with DC signal (all 1s):\n");
    
    for (int i = 0; i < N; i++) {
        input[i].re = 1.0;
        input[i].im = 0.0;
    }
    
    fft_exec(fft_forward, input, output);
    
    printf("Expected: X[0]=15, X[k]=0 for k≠0\n");
    printf("Actual:\n");
    for (int k = 0; k < N; k++) {
        double mag = sqrt(output[k].re * output[k].re + output[k].im * output[k].im);
        printf("  |X[%2d]| = %.4f", k, mag);
        
        if (k == 0) {
            if (fabs(output[k].re - 15.0) > 1e-10 || fabs(output[k].im) > 1e-10) {
                printf(" <- ERROR: Expected (15,0)");
            }
        } else {
            if (mag > 1e-10) {
                printf(" <- ERROR: Expected 0");
            }
        }
        printf("\n");
    }
    
    // Test individual radix components
    printf("\n----------------------------------------\n");
    printf("Testing N=3 and N=5 separately:\n\n");
    
    // Test N=3
    fft_object fft3 = fft_init(3, 1);
    fft_data input3[3] = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}};
    fft_data output3[3];
    
    if (fft3) {
        fft_exec(fft3, input3, output3);
        printf("N=3 FFT: ");
        for (int i = 0; i < 3; i++) {
            printf("(%.2f,%.2f) ", output3[i].re, output3[i].im);
        }
        printf("\n");
        free_fft(fft3);
    }
    
    // Test N=5
    fft_object fft5 = fft_init(5, 1);
    fft_data input5[5] = {{1.0, 0.0}, {2.0, 0.0}, {3.0, 0.0}, {4.0, 0.0}, {5.0, 0.0}};
    fft_data output5[5];
    
    if (fft5) {
        fft_exec(fft5, input5, output5);
        printf("N=5 FFT: ");
        for (int i = 0; i < 5; i++) {
            printf("(%.2f,%.2f) ", output5[i].re, output5[i].im);
        }
        printf("\n");
        free_fft(fft5);
    }
    
cleanup:
    if (fft_forward) free_fft(fft_forward);
    free(input);
    free(output);
    printf("\n");
}

// Test to verify the mixed_radix_dit_rec function specifically
void test_stage_by_stage(void) {
    printf("========================================\n");
    printf("Stage-by-Stage Analysis\n");
    printf("========================================\n\n");
    
    const int N = 15;
    
    // Create linearly increasing input
    fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output_fft = (fft_data *)malloc(N * sizeof(fft_data));
    fft_data *output_dft = (fft_data *)malloc(N * sizeof(fft_data));
    
    for (int i = 0; i < N; i++) {
        input[i].re = i + 1.0;
        input[i].im = 0.0;
    }
    
    printf("Input: 1, 2, 3, ..., 15\n\n");
    
    // Compute using your FFT
    fft_object fft_forward = fft_init(N, 1);
    if (!fft_forward) {
        printf("ERROR: Failed to initialize FFT object\n");
        goto cleanup;
    }
    
    fft_exec(fft_forward, input, output_fft);
    
    // Compute reference DFT
    for (int k = 0; k < N; k++) {
        output_dft[k].re = 0.0;
        output_dft[k].im = 0.0;
        for (int n = 0; n < N; n++) {
            double angle = -2.0 * M_PI * k * n / N;
            double cos_angle = cos(angle);
            double sin_angle = sin(angle);
            output_dft[k].re += input[n].re * cos_angle - input[n].im * sin_angle;
            output_dft[k].im += input[n].re * sin_angle + input[n].im * cos_angle;
        }
    }
    
    // Compare
    printf("Comparison of FFT vs DFT:\n");
    printf("k  |     Your FFT      |   Reference DFT   |  Error\n");
    printf("---+-------------------+-------------------+--------\n");
    
    double max_error = 0.0;
    for (int k = 0; k < N; k++) {
        double err_re = fabs(output_fft[k].re - output_dft[k].re);
        double err_im = fabs(output_fft[k].im - output_dft[k].im);
        double err = sqrt(err_re * err_re + err_im * err_im);
        
        if (err > max_error) max_error = err;
        
        printf("%2d | %7.2f,%7.2f | %7.2f,%7.2f | %.2e\n",
               k,
               output_fft[k].re, output_fft[k].im,
               output_dft[k].re, output_dft[k].im,
               err);
    }
    
    printf("\nMaximum error: %.6e\n", max_error);
    
    if (max_error < 1e-10) {
        printf("✓ PASS: FFT matches reference!\n");
    } else {
        printf("✗ FAIL: FFT has errors!\n");
        
        // Diagnostic hints
        printf("\nDiagnostic hints:\n");
        if (max_error > 1.0) {
            printf("- Very large error suggests fundamental issue in recursion\n");
            printf("- Check scratch buffer offsets in mixed_radix_dit_rec\n");
            printf("- Verify factor ordering in get_fft_execution_radices\n");
        } else if (max_error > 0.01) {
            printf("- Moderate error suggests twiddle factor issue\n");
            printf("- Check W_curr update logic in radix butterflies\n");
            printf("- Verify base angle calculation uses correct sign\n");
        }
    }
    
cleanup:
    if (fft_forward) free_fft(fft_forward);
    free(input);
    free(output_fft);
    free(output_dft);
    printf("\n");
}

int main()
{
 printf("\n");
    printf("************************************************\n");
    printf("*            FFT DEBUG DIAGNOSTICS             *\n");
    printf("************************************************\n\n");
    
    //debug_mixed_radix_stages();
    //test_stage_by_stage();
    
    printf("************************************************\n");
    printf("*          DEBUG ANALYSIS COMPLETE             *\n");
    printf("************************************************\n\n");

    run_comprehensive_complex_fft_tests();
   
  //  test_everything();


    printf("\n=== All Tests Complete ===\n");
    return  EXIT_SUCCESS;
}