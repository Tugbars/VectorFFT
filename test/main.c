#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "highspeedFFT.h"
#include "real.h"



#define M_PI 3.14159265358979323846

// Utility functions from highSpeedFFT.c
void generate_signal(fft_data *signal, int length, double freq, double amplitude)
{
    for (int i = 0; i < length; i++)
    {
        signal[i].re = amplitude * sin(2.0 * M_PI * freq * i / length);
        signal[i].im = 0.0; // Real-valued input
    }
}

double compute_mse(fft_data *original, fft_data *reconstructed, int length)
{
    double mse = 0.0;
    for (int i = 0; i < length; i++)
    {
        double diff_re = original[i].re - reconstructed[i].re;
        double diff_im = original[i].im - reconstructed[i].im;
        mse += diff_re * diff_re + diff_im * diff_im;
    }
    return mse / length;
}

void print_complex(fft_data *data, int length, const char *label)
{
    printf("%s:\n", label);
    for (int i = 0; i < length; i++)
    {
        printf("  [%d] %.6f + %.6fi\n", i, data[i].re, data[i].im);
    }
    printf("\n");
}

// Utility functions from real.c
void generate_real_signal(fft_type *signal, int length, double freq, double amplitude)
{
    for (int i = 0; i < length; i++)
    {
        signal[i] = amplitude * sin(2.0 * M_PI * freq * i / length);
    }
}

double compute_mse_real(fft_type *original, fft_type *reconstructed, int length)
{
    double mse = 0.0;
    for (int i = 0; i < length; i++)
    {
        double diff = original[i] - reconstructed[i];
        mse += diff * diff;
    }
    return mse / length;
}

void print_real(fft_type *data, int length, const char *label)
{
    printf("%s:\n", label);
    for (int i = 0; i < length; i++)
    {
        printf("  [%d] %.6f\n", i, data[i]);
    }
    printf("\n");
}


// =============================================================================
// TEST 1: Hermitian Symmetry Validation
// =============================================================================
/**
 * @brief Tests that R2C output satisfies Hermitian symmetry properties
 * 
 * For real input x[n], the DFT X[k] must satisfy:
 * - X[0] is real (DC component)
 * - X[N/2] is real (Nyquist frequency)
 * - X[N-k] = conj(X[k]) for k=1..N/2-1 (not directly testable with N/2+1 output)
 * 
 * @return 1 if pass, 0 if fail
 */
int test_hermitian_symmetry(void) {
    printf("\n=== TEST: Hermitian Symmetry ===\n");
    int pass = 1;
    const int N = 16;
    const double tol = 1e-10;
    
    fft_type *real_input = (fft_type *)malloc(N * sizeof(fft_type));
    fft_data *complex_output = (fft_data *)malloc((N/2 + 1) * sizeof(fft_data));
    
    // Generate real signal (arbitrary)
    for (int i = 0; i < N; i++) {
        real_input[i] = cos(2.0 * M_PI * 2.0 * i / N) + 
                       0.5 * sin(2.0 * M_PI * 3.0 * i / N);
    }
    
    fft_real_object rfft = fft_real_init(N);
    fft_r2c_exec(rfft, real_input, complex_output);
    
    // Check DC (k=0) is real
    if (fabs(complex_output[0].im) > tol) {
        printf("FAIL: DC component X[0].im = %.6e (should be ~0)\n", 
               complex_output[0].im);
        pass = 0;
    } else {
        printf("PASS: DC component is real (X[0].im = %.6e)\n", 
               complex_output[0].im);
    }
    
    // Check Nyquist (k=N/2) is real
    if (fabs(complex_output[N/2].im) > tol) {
        printf("FAIL: Nyquist X[N/2].im = %.6e (should be ~0)\n", 
               complex_output[N/2].im);
        pass = 0;
    } else {
        printf("PASS: Nyquist is real (X[N/2].im = %.6e)\n", 
               complex_output[N/2].im);
    }
    
    // Note: Full symmetry X[N-k]=conj(X[k]) can't be tested without computing
    // the full N-point complex FFT for comparison
    
    fft_real_free(rfft);
    free(real_input);
    free(complex_output);
    
    return pass;
}

// =============================================================================
// TEST 2: Known Transform Pairs
// =============================================================================
/**
 * @brief Tests FFT against known mathematical transform pairs
 * 
 * Tests:
 * 1. Impulse δ[0] -> constant 1 in frequency
 * 2. DC (all ones) -> impulse at k=0
 * 3. Pure cosine -> two impulses at ±k_0
 * 
 * @return 1 if pass, 0 if fail
 */
int test_known_pairs(void) {
    printf("\n=== TEST: Known Transform Pairs ===\n");
    int pass = 1;
    const int N = 16;
    const double tol = 1e-9;
    
    fft_type *input = (fft_type *)malloc(N * sizeof(fft_type));
    fft_data *output = (fft_data *)malloc((N/2 + 1) * sizeof(fft_data));
    fft_real_object rfft = fft_real_init(N);
    
    // Test 1: Impulse -> All ones in frequency
    printf("Test 1: Impulse δ[0]\n");
    for (int i = 0; i < N; i++) input[i] = 0.0;
    input[0] = 1.0;
    
    fft_r2c_exec(rfft, input, output);
    
    int impulse_pass = 1;
    for (int k = 0; k <= N/2; k++) {
        if (fabs(output[k].re - 1.0) > tol || fabs(output[k].im) > tol) {
            printf("  FAIL at k=%d: got (%.6f, %.6f), expected (1.0, 0.0)\n",
                   k, output[k].re, output[k].im);
            impulse_pass = 0;
        }
    }
    if (impulse_pass) {
        printf("  PASS: Impulse transforms to constant\n");
    } else {
        pass = 0;
    }
    
    // Test 2: DC (all ones) -> Impulse at k=0
    printf("Test 2: DC signal (all ones)\n");
    for (int i = 0; i < N; i++) input[i] = 1.0;
    
    fft_r2c_exec(rfft, input, output);
    
    int dc_pass = 1;
    // k=0 should be N
    if (fabs(output[0].re - N) > tol || fabs(output[0].im) > tol) {
        printf("  FAIL at k=0: got (%.6f, %.6f), expected (%d, 0.0)\n",
               output[0].re, output[0].im, N);
        dc_pass = 0;
    }
    // All other bins should be ~0
    for (int k = 1; k <= N/2; k++) {
        if (fabs(output[k].re) > tol || fabs(output[k].im) > tol) {
            printf("  FAIL at k=%d: got (%.6f, %.6f), expected (0.0, 0.0)\n",
                   k, output[k].re, output[k].im);
            dc_pass = 0;
        }
    }
    if (dc_pass) {
        printf("  PASS: DC signal transforms to impulse at k=0\n");
    } else {
        pass = 0;
    }
    
    // Test 3: Pure cosine cos(2πk₀n/N) -> impulses at k=±k₀
    printf("Test 3: Pure cosine at frequency k=2\n");
    const int k0 = 2;
    for (int i = 0; i < N; i++) {
        input[i] = cos(2.0 * M_PI * k0 * i / N);
    }
    
    fft_r2c_exec(rfft, input, output);
    
    int cosine_pass = 1;
    // Expected: X[k0] = N/2, X[N-k0] = N/2 (but we only see X[k0])
    // For real cosine: X[k0].re = N/2, X[k0].im = 0
    if (fabs(output[k0].re - N/2.0) > tol || fabs(output[k0].im) > tol) {
        printf("  FAIL at k=%d: got (%.6f, %.6f), expected (%.1f, 0.0)\n",
               k0, output[k0].re, output[k0].im, N/2.0);
        cosine_pass = 0;
    }
    // All other bins should be ~0
    for (int k = 0; k <= N/2; k++) {
        if (k == k0) continue;
        if (fabs(output[k].re) > tol || fabs(output[k].im) > tol) {
            printf("  FAIL at k=%d: got (%.6f, %.6f), expected (0.0, 0.0)\n",
                   k, output[k].re, output[k].im);
            cosine_pass = 0;
        }
    }
    if (cosine_pass) {
        printf("  PASS: Cosine transforms to impulse at k=%d\n", k0);
    } else {
        pass = 0;
    }
    
    fft_real_free(rfft);
    free(input);
    free(output);
    
    return pass;
}

// =============================================================================
// TEST 3: Parseval's Theorem (Energy Conservation)
// =============================================================================
/**
 * @brief Tests energy conservation: Σ|x[n]|² = (1/N)Σ|X[k]|²
 * 
 * For real signals with Hermitian spectrum:
 * Energy_freq = (|X[0]|² + |X[N/2]|² + 2·Σ|X[k]|² for k=1..N/2-1) / N
 * 
 * @return 1 if pass, 0 if fail
 */
int test_parseval(void) {
    printf("\n=== TEST: Parseval's Theorem (Energy Conservation) ===\n");
    int pass = 1;
    const int N = 32;
    const double tol = 1e-9;
    
    fft_type *input = (fft_type *)malloc(N * sizeof(fft_type));
    fft_data *output = (fft_data *)malloc((N/2 + 1) * sizeof(fft_data));
    
    // Generate arbitrary real signal
    for (int i = 0; i < N; i++) {
        input[i] = cos(2.0 * M_PI * 2.0 * i / N) + 
                   0.7 * sin(2.0 * M_PI * 5.0 * i / N) +
                   0.3 * cos(2.0 * M_PI * 7.0 * i / N);
    }
    
    fft_real_object rfft = fft_real_init(N);
    fft_r2c_exec(rfft, input, output);
    
    // Compute time-domain energy
    double energy_time = 0.0;
    for (int i = 0; i < N; i++) {
        energy_time += input[i] * input[i];
    }
    
    // Compute frequency-domain energy
    // DC and Nyquist appear once
    double energy_freq = output[0].re * output[0].re;
    energy_freq += output[N/2].re * output[N/2].re;
    
    // Bins 1 to N/2-1 appear twice (positive and negative frequencies)
    for (int k = 1; k < N/2; k++) {
        double mag2 = output[k].re * output[k].re + 
                     output[k].im * output[k].im;
        energy_freq += 2.0 * mag2;
    }
    energy_freq /= N; // Normalization
    
    printf("Energy (time domain): %.10f\n", energy_time);
    printf("Energy (freq domain): %.10f\n", energy_freq);
    printf("Relative error: %.6e\n", fabs(energy_time - energy_freq) / energy_time);
    
    if (fabs(energy_time - energy_freq) / energy_time > tol) {
        printf("FAIL: Energy not conserved (Parseval's theorem violated)\n");
        pass = 0;
    } else {
        printf("PASS: Energy conserved (Parseval's theorem holds)\n");
    }
    
    fft_real_free(rfft);
    free(input);
    free(output);
    
    return pass;
}

// =============================================================================
// TEST 4: Linearity
// =============================================================================
/**
 * @brief Tests FFT linearity: FFT(a·x + b·y) = a·FFT(x) + b·FFT(y)
 * 
 * @return 1 if pass, 0 if fail
 */
int test_linearity(void) {
    printf("\n=== TEST: Linearity ===\n");
    int pass = 1;
    const int N = 16;
    const double tol = 1e-9;
    const double a = 2.5, b = -1.3;
    
    fft_type *x = (fft_type *)malloc(N * sizeof(fft_type));
    fft_type *y = (fft_type *)malloc(N * sizeof(fft_type));
    fft_type *sum = (fft_type *)malloc(N * sizeof(fft_type));
    fft_data *X = (fft_data *)malloc((N/2 + 1) * sizeof(fft_data));
    fft_data *Y = (fft_data *)malloc((N/2 + 1) * sizeof(fft_data));
    fft_data *Sum = (fft_data *)malloc((N/2 + 1) * sizeof(fft_data));
    fft_data *Linear = (fft_data *)malloc((N/2 + 1) * sizeof(fft_data));
    
    // Generate two different signals
    for (int i = 0; i < N; i++) {
        x[i] = cos(2.0 * M_PI * 2.0 * i / N);
        y[i] = sin(2.0 * M_PI * 3.0 * i / N);
        sum[i] = a * x[i] + b * y[i];
    }
    
    fft_real_object rfft = fft_real_init(N);
    
    // Compute FFT(x), FFT(y), FFT(a·x + b·y)
    fft_r2c_exec(rfft, x, X);
    fft_r2c_exec(rfft, y, Y);
    fft_r2c_exec(rfft, sum, Sum);
    
    // Compute a·FFT(x) + b·FFT(y)
    for (int k = 0; k <= N/2; k++) {
        Linear[k].re = a * X[k].re + b * Y[k].re;
        Linear[k].im = a * X[k].im + b * Y[k].im;
    }
    
    // Compare Sum vs Linear
    double max_error = 0.0;
    for (int k = 0; k <= N/2; k++) {
        double err_re = fabs(Sum[k].re - Linear[k].re);
        double err_im = fabs(Sum[k].im - Linear[k].im);
        if (err_re > max_error) max_error = err_re;
        if (err_im > max_error) max_error = err_im;
    }
    
    printf("Max error: %.6e\n", max_error);
    if (max_error > tol) {
        printf("FAIL: Linearity violated\n");
        pass = 0;
    } else {
        printf("PASS: FFT is linear\n");
    }
    
    fft_real_free(rfft);
    free(x); free(y); free(sum);
    free(X); free(Y); free(Sum); free(Linear);
    
    return pass;
}

// =============================================================================
// TEST 5: Edge Cases
// =============================================================================
/**
 * @brief Tests edge cases: minimum size (N=2) and large sizes
 * 
 * @return 1 if pass, 0 if fail
 */
int test_edge_cases(void) {
    printf("\n=== TEST: Edge Cases ===\n");
    int pass = 1;
    const double tol = 1e-10;
    
    // Test 1: Minimum size N=2
    printf("Test 1: Minimum size N=2\n");
    {
        const int N = 2;
        fft_type input[2] = {1.0, -1.0};
        fft_data output[2]; // N/2+1 = 2
        fft_type reconstructed[2];
        
        fft_real_object rfft = fft_real_init(N);
        if (!rfft) {
            printf("  FAIL: Could not initialize N=2\n");
            pass = 0;
        } else {
            fft_r2c_exec(rfft, input, output);
            fft_c2r_exec(rfft, output, reconstructed);
            
            // Scale
            for (int i = 0; i < N; i++) reconstructed[i] /= N;
            
            // Check round-trip
            double mse = 0.0;
            for (int i = 0; i < N; i++) {
                double diff = input[i] - reconstructed[i];
                mse += diff * diff;
            }
            mse /= N;
            
            if (mse < tol) {
                printf("  PASS: N=2 round-trip MSE = %.6e\n", mse);
            } else {
                printf("  FAIL: N=2 round-trip MSE = %.6e\n", mse);
                pass = 0;
            }
            
            fft_real_free(rfft);
        }
    }
    
    // Test 2: Large power-of-2
    printf("Test 2: Large power-of-2 N=1024\n");
    {
        const int N = 1024;
        fft_type *input = (fft_type *)malloc(N * sizeof(fft_type));
        fft_data *output = (fft_data *)malloc((N/2 + 1) * sizeof(fft_data));
        fft_type *reconstructed = (fft_type *)malloc(N * sizeof(fft_type));
        
        for (int i = 0; i < N; i++) {
            input[i] = sin(2.0 * M_PI * 10.0 * i / N);
        }
        
        fft_real_object rfft = fft_real_init(N);
        if (!rfft) {
            printf("  FAIL: Could not initialize N=1024\n");
            pass = 0;
        } else {
            fft_r2c_exec(rfft, input, output);
            fft_c2r_exec(rfft, output, reconstructed);
            
            for (int i = 0; i < N; i++) reconstructed[i] /= N;
            
            double mse = 0.0;
            for (int i = 0; i < N; i++) {
                double diff = input[i] - reconstructed[i];
                mse += diff * diff;
            }
            mse /= N;
            
            if (mse < tol) {
                printf("  PASS: N=1024 round-trip MSE = %.6e\n", mse);
            } else {
                printf("  FAIL: N=1024 round-trip MSE = %.6e\n", mse);
                pass = 0;
            }
            
            fft_real_free(rfft);
        }
        
        free(input);
        free(output);
        free(reconstructed);
    }
    
    // Test 3: Non-power-of-2 even (e.g., N=18)
    printf("Test 3: Non-power-of-2 even N=18\n");
    {
        const int N = 18;
        fft_type *input = (fft_type *)malloc(N * sizeof(fft_type));
        fft_data *output = (fft_data *)malloc((N/2 + 1) * sizeof(fft_data));
        fft_type *reconstructed = (fft_type *)malloc(N * sizeof(fft_type));
        
        for (int i = 0; i < N; i++) {
            input[i] = cos(2.0 * M_PI * 3.0 * i / N);
        }
        
        fft_real_object rfft = fft_real_init(N);
        if (!rfft) {
            printf("  FAIL: Could not initialize N=18\n");
            pass = 0;
        } else {
            fft_r2c_exec(rfft, input, output);
            fft_c2r_exec(rfft, output, reconstructed);
            
            for (int i = 0; i < N; i++) reconstructed[i] /= N;
            
            double mse = 0.0;
            for (int i = 0; i < N; i++) {
                double diff = input[i] - reconstructed[i];
                mse += diff * diff;
            }
            mse /= N;
            
            if (mse < tol) {
                printf("  PASS: N=18 round-trip MSE = %.6e\n", mse);
            } else {
                printf("  FAIL: N=18 round-trip MSE = %.6e\n", mse);
                pass = 0;
            }
            
            fft_real_free(rfft);
        }
        
        free(input);
        free(output);
        free(reconstructed);
    }
    
    return pass;
}

// =============================================================================
// TEST 6: Regression Test for C2R Conjugate Twiddle Bug
// =============================================================================
/**
 * @brief Specific test that would have FAILED with the old buggy code
 * 
 * This tests the exact bug you fixed: incorrect conjugate twiddle signs
 * in the C2R combine function.
 * 
 * @return 1 if pass, 0 if fail
 */
int test_c2r_regression(void) {
    printf("\n=== TEST: C2R Conjugate Twiddle Regression ===\n");
    int pass = 1;
    const int N = 16;
    const double tol = 1e-10;
    
    fft_type *cosine = (fft_type *)malloc(N * sizeof(fft_type));
    fft_data *spectrum = (fft_data *)malloc((N/2 + 1) * sizeof(fft_data));
    fft_type *reconstructed = (fft_type *)malloc(N * sizeof(fft_type));
    
    // Pure cosine at frequency k=2
    for (int i = 0; i < N; i++) {
        cosine[i] = cos(2.0 * M_PI * 2.0 * i / N);
    }
    
    fft_real_object rfft = fft_real_init(N);
    
    // Forward R2C
    fft_r2c_exec(rfft, cosine, spectrum);
    
    // Inverse C2R (this would fail with old buggy code)
    fft_c2r_exec(rfft, spectrum, reconstructed);
    
    // Scale
    for (int i = 0; i < N; i++) {
        reconstructed[i] /= N;
    }
    
    // Compute MSE
    double mse = 0.0;
    for (int i = 0; i < N; i++) {
        double diff = cosine[i] - reconstructed[i];
        mse += diff * diff;
    }
    mse /= N;
    
    printf("Cosine round-trip MSE: %.6e\n", mse);
    
    // Old buggy code would produce MSE >> 1e-10
    if (mse < tol) {
        printf("PASS: C2R conjugate twiddle fix verified\n");
    } else {
        printf("FAIL: C2R still has bugs (MSE too high)\n");
        pass = 0;
        
        // Print first few samples for debugging
        printf("First 8 samples:\n");
        printf("  Original    | Reconstructed | Error\n");
        for (int i = 0; i < 8; i++) {
            printf("  %+.6f   | %+.6f      | %.6e\n",
                   cosine[i], reconstructed[i], 
                   fabs(cosine[i] - reconstructed[i]));
        }
    }
    
    fft_real_free(rfft);
    free(cosine);
    free(spectrum);
    free(reconstructed);
    
    return pass;
}

// =============================================================================
// TEST 7: Performance Benchmark
// =============================================================================
/**
 * @brief Simple performance benchmark to verify AVX2 is being used
 * 
 * Not a pass/fail test, but provides timing information
 */
void benchmark_performance(void) {
    printf("\n=== BENCHMARK: Performance ===\n");
    const int N = 1024;
    const int iterations = 10000;
    
    fft_type *input = (fft_type *)malloc(N * sizeof(fft_type));
    fft_data *output = (fft_data *)malloc((N/2 + 1) * sizeof(fft_data));
    
    for (int i = 0; i < N; i++) {
        input[i] = sin(2.0 * M_PI * 10.0 * i / N);
    }
    
    fft_real_object rfft = fft_real_init(N);
    
    // Warm-up
    for (int i = 0; i < 100; i++) {
        fft_r2c_exec(rfft, input, output);
    }
    
    // Benchmark R2C
    clock_t start = clock();
    for (int i = 0; i < iterations; i++) {
        fft_r2c_exec(rfft, input, output);
    }
    clock_t end = clock();
    
    double time_r2c = 1000.0 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("R2C: %d iterations of N=%d in %.2f ms\n", 
           iterations, N, time_r2c);
    printf("     Average: %.3f µs per transform\n", 
           time_r2c * 1000.0 / iterations);
    
    // Benchmark C2R
    start = clock();
    for (int i = 0; i < iterations; i++) {
        fft_c2r_exec(rfft, output, input);
    }
    end = clock();
    
    double time_c2r = 1000.0 * (double)(end - start) / CLOCKS_PER_SEC;
    printf("C2R: %d iterations of N=%d in %.2f ms\n", 
           iterations, N, time_c2r);
    printf("     Average: %.3f µs per transform\n", 
           time_c2r * 1000.0 / iterations);
    
    fft_real_free(rfft);
    free(input);
    free(output);
}

// =============================================================================
// MAIN TEST RUNNER
// =============================================================================
/**
 * @brief Add this to your main() function to run all tests
 */
int run_comprehensive_tests(void) {
    printf("\n");
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║     COMPREHENSIVE REAL FFT TEST SUITE                 ║\n");
    printf("╚════════════════════════════════════════════════════════╝\n");
    
    int total = 0, passed = 0;
    
    // Run all tests
    if (test_hermitian_symmetry()) passed++;
    total++;
    
    if (test_known_pairs()) passed++;
    total++;
    
    if (test_parseval()) passed++;
    total++;
    
    if (test_linearity()) passed++;
    total++;
    
    if (test_edge_cases()) passed++;
    total++;
    
    if (test_c2r_regression()) passed++;
    total++;
    
    // Performance benchmark (not counted in pass/fail)
    benchmark_performance();
    
    // Final summary
    printf("\n");
    printf("╔════════════════════════════════════════════════════════╗\n");
    printf("║     FINAL RESULTS                                      ║\n");
    printf("╠════════════════════════════════════════════════════════╣\n");
    printf("║  Tests Passed:  %2d / %2d                               ║\n", 
           passed, total);
    printf("║  Success Rate:  %.1f%%                                 ║\n",
           100.0 * passed / total);
    printf("╚════════════════════════════════════════════════════════╝\n");
    
    return (passed == total) ? 1 : 0;
}

int main()
{
    // Common parameters
    const double freq = 2.0;            // Frequency of test signal
    const double amplitude = 1.0;       // Amplitude of test signal
    const double mse_tolerance = 1e-10; // MSE tolerance for passing tests

    // **Complex FFT Tests**
    int complex_lengths[] = {4, 8, 15, 20, 64}; // Mixed-radix: 4, 8, 64; Bluestein: 15, 20
    int num_complex_tests = sizeof(complex_lengths) / sizeof(complex_lengths[0]);

    printf("=== Complex FFT Tests ===\n");
    for (int test = 0; test < num_complex_tests; test++)
    {
        int N = complex_lengths[test];
        printf("Testing Complex FFT with N = %d\n", N);

        // Allocate memory
        fft_data *input = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *output = (fft_data *)malloc(N * sizeof(fft_data));
        fft_data *inverse = (fft_data *)malloc(N * sizeof(fft_data));
        if (!input || !output || !inverse)
        {
            fprintf(stderr, "Memory allocation failed for N = %d\n", N);
            free(input);
            free(output);
            free(inverse);
            continue;
        }

        // Generate test signal
        generate_signal(input, N, freq, amplitude);

        // Initialize and perform forward FFT
        fft_object fft = fft_init(N, 1);
        if (!fft)
        {
            fprintf(stderr, "FFT initialization failed for N = %d\n", N);
            free(input);
            free(output);
            free(inverse);
            continue;
        }
        fft_exec(fft, input, output);
        printf("Algorithm: %s\n", fft->lt == 0 ? "Mixed-Radix" : "Bluestein");
        print_complex(output, N, "FFT Output");

        // Initialize and perform inverse FFT
        fft_object ifft = fft_init(N, -1);
        if (!ifft)
        {
            fprintf(stderr, "Inverse FFT initialization failed for N = %d\n", N);
            free_fft(fft);
            free(input);
            free(output);
            free(inverse);
            continue;
        }
        fft_exec(ifft, output, inverse);

        // Scale inverse output
        for (int i = 0; i < N; i++)
        {
            inverse[i].re /= N;
            inverse[i].im /= N;
        }
        print_complex(inverse, N, "Inverse FFT Output");

        // Verify correctness
        double mse = compute_mse(input, inverse, N);
        printf("MSE: %.6e\n", mse);
        if (mse < mse_tolerance)
        {
            printf("Test passed\n");
        }
        else
        {
            printf("Test failed (MSE exceeds tolerance)\n");
        }

        // Cleanup
        free_fft(fft);
        free_fft(ifft);
        free(input);
        free(output);
        free(inverse);
        printf("\n");
    }

    // **Real FFT Tests** - UPDATED FOR UNIFIED API
    int real_lengths[] = {4, 8, 16, 32, 64}; // Even lengths only
    int num_real_tests = sizeof(real_lengths) / sizeof(real_lengths[0]);

    printf("=== Real FFT Tests ===\n");
    for (int test = 0; test < num_real_tests; test++)
    {
        int N = real_lengths[test];
        printf("Testing Real FFT with N = %d\n", N);

        // Allocate memory
        fft_type *real_input = (fft_type *)malloc(N * sizeof(fft_type));
        fft_data *complex_output = (fft_data *)malloc((N/2 + 1) * sizeof(fft_data)); // FIXED: N/2+1
        fft_type *real_inverse = (fft_type *)malloc(N * sizeof(fft_type));
        if (!real_input || !complex_output || !real_inverse)
        {
            fprintf(stderr, "Memory allocation failed for N = %d\n", N);
            free(real_input);
            free(complex_output);
            free(real_inverse);
            continue;
        }

        // Generate real-valued test signal
        generate_real_signal(real_input, N, freq, amplitude);
        print_real(real_input, N, "Original Real Signal");

        // FIXED: Create single unified real FFT object (no direction parameter)
        fft_real_object rfft = fft_real_init(N);
        if (!rfft)
        {
            fprintf(stderr, "Real FFT initialization failed for N = %d\n", N);
            free(real_input);
            free(complex_output);
            free(real_inverse);
            continue;
        }

        // Perform real-to-complex FFT
        if (fft_r2c_exec(rfft, real_input, complex_output) != 0)
        {
            fprintf(stderr, "R2C execution failed for N = %d\n", N);
            fft_real_free(rfft);
            free(real_input);
            free(complex_output);
            free(real_inverse);
            continue;
        }
        print_complex(complex_output, N / 2 + 1, "R2C FFT Output");

        // Perform complex-to-real FFT (uses same object!)
        if (fft_c2r_exec(rfft, complex_output, real_inverse) != 0)
        {
            fprintf(stderr, "C2R execution failed for N = %d\n", N);
            fft_real_free(rfft);
            free(real_input);
            free(complex_output);
            free(real_inverse);
            continue;
        }

        // Scale inverse output
        for (int i = 0; i < N; i++)
        {
            real_inverse[i] /= N;
        }
        print_real(real_inverse, N, "Reconstructed Real Signal");

        // Verify correctness
        double mse = compute_mse_real(real_input, real_inverse, N);
        printf("MSE: %.6e\n", mse);
        if (mse < mse_tolerance)
        {
            printf("Test passed\n");
        }
        else
        {
            printf("Test failed (MSE exceeds tolerance)\n");
        }

        // FIXED: Cleanup - single object only
        fft_real_free(rfft);
        free(real_input);
        free(complex_output);
        free(real_inverse);
        printf("\n");
    }

    // **Error Handling Tests** - UPDATED
    printf("=== Error Handling Tests ===\n");

    // Complex FFT: Invalid length
    fft_object bad_fft = fft_init(0, 1);
    if (!bad_fft)
    {
        printf("Complex FFT: Correctly handled invalid length (0)\n");
    }

    // Complex FFT: Invalid direction
    bad_fft = fft_init(8, 0);
    if (!bad_fft)
    {
        printf("Complex FFT: Correctly handled invalid direction (0)\n");
    }

    // FIXED: Real FFT error handling - no direction parameter now
    // Real FFT: Odd length
    fft_real_object bad_r2c = fft_real_init(5);
    if (!bad_r2c)
    {
        printf("Real FFT: Correctly handled odd length (5)\n");
    }

    // Real FFT: Zero length
    bad_r2c = fft_real_init(0);
    if (!bad_r2c)
    {
        printf("Real FFT: Correctly handled invalid length (0)\n");
    }

    // Real FFT: Negative length
    bad_r2c = fft_real_init(-8);
    if (!bad_r2c)
    {
        printf("Real FFT: Correctly handled negative length (-8)\n");
    }

    int all_passed = run_comprehensive_tests();

    printf("\n=== All Tests Complete ===\n");
    return all_passed ? EXIT_SUCCESS : EXIT_FAILURE;
}