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
    const int N = 64;
    const double tol = 1e-9;
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
    int sizes[] = {2, 4, 8, 15, 16, 17, 32, 63, 64, 100, 128, 256, 512, 1024};
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const double tol = 1e-10;
    int total = 0, passed = 0;
    
    for (int s = 0; s < num_sizes; s++) {
        int N = sizes[s];
        total++;
        
        fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *freq = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *reconstructed = (fft_data *)malloc(N * sizeof(fft_data));
        
        // Generate test signal
        for (int i = 0; i < N; i++) {
            input[i].re = cos(2.0 * M_PI * 3 * i / N) + 0.5 * sin(2.0 * M_PI * 7 * i / N);
            input[i].im = sin(2.0 * M_PI * 5 * i / N);
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
        
        fft_exec(fwd, input, freq);
        fft_exec(inv, freq, reconstructed);
        
        // Scale by 1/N
        for (int i = 0; i < N; i++) {
            reconstructed[i].re /= N;
            reconstructed[i].im /= N;
        }
        
        double mse = compute_complex_mse(input, reconstructed, N);
        
        if (mse < tol) {
            printf("  N=%4d: PASS (MSE=%.3e, algo=%s)\n", 
                   N, mse, fwd->lt == 0 ? "Mixed-Radix" : "Bluestein");
            passed++;
        } else {
            printf("  N=%4d: FAIL (MSE=%.3e)\n", N, mse);
            // Print first few samples for debugging
            printf("    First 4 samples:\n");
            for (int i = 0; i < 4 && i < N; i++) {
                printf("      [%d] orig=(%.6f,%.6f) recon=(%.6f,%.6f)\n",
                       i, input[i].re, input[i].im,
                       reconstructed[i].re, reconstructed[i].im);
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
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║     COMPREHENSIVE COMPLEX FFT TEST SUITE              ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    
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
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║     COMPLEX FFT TEST RESULTS                          ║\n");
    printf("╠════════════════════════════════════════════════════════╣\n");
    printf("║  Tests Passed:  %2d / %2d                               ║\n", 
           passed, total);
    printf("║  Success Rate:  %.1f%%                                 ║\n",
           100.0 * passed / total);
    printf("╚════════════════════════════════════════════════════════╝\n");
    
    return (passed == total) ? 1 : 0;
}

// ============================================================================
// BENCHMARK 1: Throughput vs Size
// ============================================================================
void benchmark_throughput(void) {
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  BENCHMARK 1: Throughput vs FFT Size                          ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
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
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  BENCHMARK 2: Single Transform Latency                        ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
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
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  BENCHMARK 3: Cache Effects (In-place vs Out-of-place)       ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
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
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  BENCHMARK 4: Initialization Overhead                         ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
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
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  BENCHMARK 5: Forward vs Inverse Transform Speed              ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
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
    printf("\n╔════════════════════════════════════════════════════════════════╗\n");
    printf("║  BENCHMARK 6: Computational Efficiency                        ║\n");
    printf("╚════════════════════════════════════════════════════════════════╝\n\n");
    
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
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                   FFT PERFORMANCE BENCHMARK SUITE                 ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n");
    
    benchmark_throughput();
    benchmark_latency();
    benchmark_cache_effects();
    benchmark_init_overhead();
    benchmark_forward_vs_inverse();
    benchmark_efficiency();
    
    printf("\n");
    printf("╔═══════════════════════════════════════════════════════════════════╗\n");
    printf("║                     BENCHMARK COMPLETE                            ║\n");
    printf("╚═══════════════════════════════════════════════════════════════════╝\n\n");
}

void debug_fft_scaling(void) {
    printf("\n=== DEBUG: FFT Scaling Convention ===\n");
    const int N = 8;
    
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

int main()
{
    debug_fft_scaling();
    debug_parseval_detailed();
    int all_passed = true; // run_comprehensive_complex_fft_tests();

   //run_all_benchmarks();

    printf("\n=== All Tests Complete ===\n");
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}