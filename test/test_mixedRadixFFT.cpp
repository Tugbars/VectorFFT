#include <gtest/gtest.h>
#include <complex>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include "highspeedFFT.h"

// Helper class for FFT testing
class FFTTest : public ::testing::Test {
protected:
    static constexpr double EPSILON = 1e-10;
    static constexpr double LOOSE_EPSILON = 1e-8;
    
    void SetUp() override {
        // Seed for reproducible random tests
        rng.seed(42);
    }
    
    void TearDown() override {
        // Cleanup happens automatically via RAII
    }
    
    // Generate random complex signal
    std::vector<fft_data> generateRandomSignal(int N) {
        std::vector<fft_data> signal(N);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        
        for (int i = 0; i < N; ++i) {
            signal[i].re = dist(rng);
            signal[i].im = dist(rng);
        }
        return signal;
    }
    
    // Generate pure sine wave
    std::vector<fft_data> generateSineWave(int N, double frequency) {
        std::vector<fft_data> signal(N);
        for (int i = 0; i < N; ++i) {
            double phase = 2.0 * M_PI * frequency * i / N;
            signal[i].re = std::cos(phase);
            signal[i].im = std::sin(phase);
        }
        return signal;
    }
    
    // Generate real sine wave (imaginary part = 0)
    std::vector<fft_data> generateRealSine(int N, double frequency) {
        std::vector<fft_data> signal(N);
        for (int i = 0; i < N; ++i) {
            double phase = 2.0 * M_PI * frequency * i / N;
            signal[i].re = std::sin(phase);
            signal[i].im = 0.0;
        }
        return signal;
    }
    
    // Delta function (impulse)
    std::vector<fft_data> generateDelta(int N, int position = 0) {
        std::vector<fft_data> signal(N);
        for (int i = 0; i < N; ++i) {
            signal[i].re = (i == position) ? 1.0 : 0.0;
            signal[i].im = 0.0;
        }
        return signal;
    }
    
    // Constant signal
    std::vector<fft_data> generateConstant(int N, double value = 1.0) {
        std::vector<fft_data> signal(N);
        for (int i = 0; i < N; ++i) {
            signal[i].re = value;
            signal[i].im = 0.0;
        }
        return signal;
    }
    
    // Compute magnitude
    double magnitude(const fft_data& x) {
        return std::sqrt(x.re * x.re + x.im * x.im);
    }
    
    // Compute relative error
    double relativeError(const fft_data& a, const fft_data& b) {
        double num = std::sqrt((a.re - b.re) * (a.re - b.re) + 
                               (a.im - b.im) * (a.im - b.im));
        double den = std::sqrt(b.re * b.re + b.im * b.im);
        return (den > 1e-15) ? num / den : num;
    }
    
    // Naive DFT for reference (slow but correct)
    std::vector<fft_data> naiveDFT(const std::vector<fft_data>& input, int sign) {
        int N = input.size();
        std::vector<fft_data> output(N);
        
        for (int k = 0; k < N; ++k) {
            double sum_re = 0.0, sum_im = 0.0;
            for (int n = 0; n < N; ++n) {
                double angle = -2.0 * M_PI * sign * k * n / N;
                double wr = std::cos(angle);
                double wi = std::sin(angle);
                sum_re += input[n].re * wr - input[n].im * wi;
                sum_im += input[n].re * wi + input[n].im * wr;
            }
            output[k].re = sum_re;
            output[k].im = sum_im;
        }
        return output;
    }
    
    // Parseval's theorem: sum of |x[n]|^2 = (1/N) * sum of |X[k]|^2
    double parsevalError(const std::vector<fft_data>& time, 
                        const std::vector<fft_data>& freq) {
        double time_energy = 0.0;
        double freq_energy = 0.0;
        int N = time.size();
        
        for (int i = 0; i < N; ++i) {
            time_energy += time[i].re * time[i].re + time[i].im * time[i].im;
            freq_energy += freq[i].re * freq[i].re + freq[i].im * freq[i].im;
        }
        
        freq_energy /= N; // Normalize
        return std::abs(time_energy - freq_energy) / time_energy;
    }
    
    std::mt19937 rng;
};

//==============================================================================
// BASIC FUNCTIONALITY TESTS
//==============================================================================

TEST_F(FFTTest, InitializationAndCleanup) {
    fft_object fft = fft_init(16, 1);
    ASSERT_NE(fft, nullptr);
    EXPECT_EQ(fft->n_input, 16);
    EXPECT_EQ(fft->sgn, 1);
    free_fft(fft);
}

TEST_F(FFTTest, InvalidInitialization) {
    // Zero length
    fft_object fft1 = fft_init(0, 1);
    EXPECT_EQ(fft1, nullptr);
    
    // Negative length
    fft_object fft2 = fft_init(-5, 1);
    EXPECT_EQ(fft2, nullptr);
    
    // Invalid direction
    fft_object fft3 = fft_init(16, 2);
    EXPECT_EQ(fft3, nullptr);
}

//==============================================================================
// KNOWN TRANSFORM PAIRS
//==============================================================================

TEST_F(FFTTest, DeltaToConstant) {
    const int N = 64;
    auto input = generateDelta(N, 0);
    std::vector<fft_data> output(N);
    
    fft_object fft = fft_init(N, 1);
    ASSERT_NE(fft, nullptr);
    
    fft_exec(fft, input.data(), output.data());
    
    // Delta function transforms to constant (all 1s)
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(output[i].re, 1.0, EPSILON) << "at index " << i;
        EXPECT_NEAR(output[i].im, 0.0, EPSILON) << "at index " << i;
    }
    
    free_fft(fft);
}

TEST_F(FFTTest, ConstantToDelta) {
    const int N = 64;
    auto input = generateConstant(N, 1.0);
    std::vector<fft_data> output(N);
    
    fft_object fft = fft_init(N, 1);
    ASSERT_NE(fft, nullptr);
    
    fft_exec(fft, input.data(), output.data());
    
    // Constant transforms to delta at DC (index 0)
    EXPECT_NEAR(output[0].re, N, EPSILON);
    EXPECT_NEAR(output[0].im, 0.0, EPSILON);
    
    for (int i = 1; i < N; ++i) {
        EXPECT_NEAR(output[i].re, 0.0, EPSILON) << "at index " << i;
        EXPECT_NEAR(output[i].im, 0.0, EPSILON) << "at index " << i;
    }
    
    free_fft(fft);
}

TEST_F(FFTTest, SineWaveToPeaks) {
    const int N = 128;
    const int freq = 5; // 5 cycles
    auto input = generateRealSine(N, freq);
    std::vector<fft_data> output(N);
    
    fft_object fft = fft_init(N, 1);
    ASSERT_NE(fft, nullptr);
    
    fft_exec(fft, input.data(), output.data());
    
    // Should have peaks at indices freq and N-freq
    double mag_at_freq = magnitude(output[freq]);
    double mag_at_neg_freq = magnitude(output[N - freq]);
    
    EXPECT_GT(mag_at_freq, N * 0.4); // Significant energy at frequency
    EXPECT_GT(mag_at_neg_freq, N * 0.4); // And at negative frequency
    
    // Other bins should be near zero
    for (int i = 1; i < N; ++i) {
        if (i != freq && i != N - freq) {
            EXPECT_LT(magnitude(output[i]), 1.0) << "at index " << i;
        }
    }
    
    free_fft(fft);
}

//==============================================================================
// INVERSE TRANSFORM TESTS
//==============================================================================

TEST_F(FFTTest, ForwardInverseRoundTrip) {
    const int N = 128;
    auto input = generateRandomSignal(N);
    std::vector<fft_data> fwd_output(N);
    std::vector<fft_data> inv_output(N);
    
    fft_object fft_fwd = fft_init(N, 1);
    fft_object fft_inv = fft_init(N, -1);
    ASSERT_NE(fft_fwd, nullptr);
    ASSERT_NE(fft_inv, nullptr);
    
    fft_exec(fft_fwd, input.data(), fwd_output.data());
    fft_exec(fft_inv, fwd_output.data(), inv_output.data());
    
    // Scale by 1/N for inverse
    for (int i = 0; i < N; ++i) {
        inv_output[i].re /= N;
        inv_output[i].im /= N;
    }
    
    // Should recover original signal
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(inv_output[i].re, input[i].re, LOOSE_EPSILON) << "at index " << i;
        EXPECT_NEAR(inv_output[i].im, input[i].im, LOOSE_EPSILON) << "at index " << i;
    }
    
    free_fft(fft_fwd);
    free_fft(fft_inv);
}

//==============================================================================
// LINEARITY TESTS
//==============================================================================

TEST_F(FFTTest, LinearityProperty) {
    const int N = 64;
    auto x = generateRandomSignal(N);
    auto y = generateRandomSignal(N);
    
    const double a = 2.5;
    const double b = -1.3;
    
    std::vector<fft_data> ax_plus_by(N);
    std::vector<fft_data> fft_x(N), fft_y(N), fft_combo(N);
    
    // Compute a*x + b*y
    for (int i = 0; i < N; ++i) {
        ax_plus_by[i].re = a * x[i].re + b * y[i].re;
        ax_plus_by[i].im = a * x[i].im + b * y[i].im;
    }
    
    fft_object fft = fft_init(N, 1);
    ASSERT_NE(fft, nullptr);
    
    fft_exec(fft, x.data(), fft_x.data());
    fft_exec(fft, y.data(), fft_y.data());
    fft_exec(fft, ax_plus_by.data(), fft_combo.data());
    
    // FFT(a*x + b*y) should equal a*FFT(x) + b*FFT(y)
    for (int i = 0; i < N; ++i) {
        double expected_re = a * fft_x[i].re + b * fft_y[i].re;
        double expected_im = a * fft_x[i].im + b * fft_y[i].im;
        
        EXPECT_NEAR(fft_combo[i].re, expected_re, LOOSE_EPSILON) << "at index " << i;
        EXPECT_NEAR(fft_combo[i].im, expected_im, LOOSE_EPSILON) << "at index " << i;
    }
    
    free_fft(fft);
}

//==============================================================================
// PARSEVAL'S THEOREM
//==============================================================================

TEST_F(FFTTest, ParsevalsTheorem) {
    const int N = 256;
    auto input = generateRandomSignal(N);
    std::vector<fft_data> output(N);
    
    fft_object fft = fft_init(N, 1);
    ASSERT_NE(fft, nullptr);
    
    fft_exec(fft, input.data(), output.data());
    
    double error = parsevalError(input, output);
    EXPECT_LT(error, 1e-9) << "Parseval's theorem violated";
    
    free_fft(fft);
}

//==============================================================================
// SYMMETRY PROPERTIES
//==============================================================================

TEST_F(FFTTest, RealInputConjugateSymmetry) {
    const int N = 128;
    std::vector<fft_data> input(N);
    
    // Real input
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (int i = 0; i < N; ++i) {
        input[i].re = dist(rng);
        input[i].im = 0.0;
    }
    
    std::vector<fft_data> output(N);
    fft_object fft = fft_init(N, 1);
    ASSERT_NE(fft, nullptr);
    
    fft_exec(fft, input.data(), output.data());
    
    // X[k] = conj(X[N-k]) for real input
    for (int k = 1; k < N/2; ++k) {
        EXPECT_NEAR(output[k].re, output[N-k].re, LOOSE_EPSILON) 
            << "Real part at k=" << k;
        EXPECT_NEAR(output[k].im, -output[N-k].im, LOOSE_EPSILON) 
            << "Imaginary part at k=" << k;
    }
    
    free_fft(fft);
}

TEST_F(FFTTest, EvenRealInputRealOutput) {
    const int N = 64;
    std::vector<fft_data> input(N);
    
    // Even real input
    for (int i = 0; i < N/2; ++i) {
        double val = std::cos(2.0 * M_PI * i / N);
        input[i].re = val;
        input[i].im = 0.0;
        input[N-1-i].re = val;
        input[N-1-i].im = 0.0;
    }
    
    std::vector<fft_data> output(N);
    fft_object fft = fft_init(N, 1);
    ASSERT_NE(fft, nullptr);
    
    fft_exec(fft, input.data(), output.data());
    
    // Output should be real (imaginary part near zero)
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(output[i].im, 0.0, LOOSE_EPSILON) << "at index " << i;
    }
    
    free_fft(fft);
}

//==============================================================================
// DIFFERENT RADICES AND SIZES
//==============================================================================

TEST_F(FFTTest, PowerOfTwo) {
    std::vector<int> sizes = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096};
    
    for (int N : sizes) {
        auto input = generateRandomSignal(N);
        std::vector<fft_data> output(N);
        auto reference = naiveDFT(input, 1);
        
        fft_object fft = fft_init(N, 1);
        ASSERT_NE(fft, nullptr) << "Failed for N=" << N;
        
        fft_exec(fft, input.data(), output.data());
        
        double max_error = 0.0;
        for (int i = 0; i < N; ++i) {
            double err = relativeError(output[i], reference[i]);
            max_error = std::max(max_error, err);
        }
        
        EXPECT_LT(max_error, 1e-9) << "Failed for N=" << N;
        free_fft(fft);
    }
}

TEST_F(FFTTest, PowerOfThree) {
    std::vector<int> sizes = {3, 9, 27, 81, 243, 729};
    
    for (int N : sizes) {
        auto input = generateRandomSignal(N);
        std::vector<fft_data> output(N);
        auto reference = naiveDFT(input, 1);
        
        fft_object fft = fft_init(N, 1);
        ASSERT_NE(fft, nullptr) << "Failed for N=" << N;
        
        fft_exec(fft, input.data(), output.data());
        
        double max_error = 0.0;
        for (int i = 0; i < N; ++i) {
            double err = relativeError(output[i], reference[i]);
            max_error = std::max(max_error, err);
        }
        
        EXPECT_LT(max_error, 1e-9) << "Failed for N=" << N;
        free_fft(fft);
    }
}

TEST_F(FFTTest, PowerOfFive) {
    std::vector<int> sizes = {5, 25, 125, 625};
    
    for (int N : sizes) {
        auto input = generateRandomSignal(N);
        std::vector<fft_data> output(N);
        auto reference = naiveDFT(input, 1);
        
        fft_object fft = fft_init(N, 1);
        ASSERT_NE(fft, nullptr) << "Failed for N=" << N;
        
        fft_exec(fft, input.data(), output.data());
        
        double max_error = 0.0;
        for (int i = 0; i < N; ++i) {
            double err = relativeError(output[i], reference[i]);
            max_error = std::max(max_error, err);
        }
        
        EXPECT_LT(max_error, 1e-8) << "Failed for N=" << N;
        free_fft(fft);
    }
}

TEST_F(FFTTest, MixedRadix) {
    std::vector<int> sizes = {6, 10, 12, 15, 18, 20, 24, 30, 36, 40, 48, 60};
    
    for (int N : sizes) {
        auto input = generateRandomSignal(N);
        std::vector<fft_data> output(N);
        auto reference = naiveDFT(input, 1);
        
        fft_object fft = fft_init(N, 1);
        ASSERT_NE(fft, nullptr) << "Failed for N=" << N;
        
        fft_exec(fft, input.data(), output.data());
        
        double max_error = 0.0;
        for (int i = 0; i < N; ++i) {
            double err = relativeError(output[i], reference[i]);
            max_error = std::max(max_error, err);
        }
        
        EXPECT_LT(max_error, 1e-8) << "Failed for N=" << N;
        free_fft(fft);
    }
}

TEST_F(FFTTest, PrimeSizesBluestein) {
    std::vector<int> sizes = {7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47};
    
    for (int N : sizes) {
        auto input = generateRandomSignal(N);
        std::vector<fft_data> output(N);
        auto reference = naiveDFT(input, 1);
        
        fft_object fft = fft_init(N, 1);
        ASSERT_NE(fft, nullptr) << "Failed for N=" << N;
        
        fft_exec(fft, input.data(), output.data());
        
        double max_error = 0.0;
        for (int i = 0; i < N; ++i) {
            double err = relativeError(output[i], reference[i]);
            max_error = std::max(max_error, err);
        }
        
        EXPECT_LT(max_error, 1e-8) << "Failed for N=" << N;
        free_fft(fft);
    }
}

//==============================================================================
// EDGE CASES
//==============================================================================

TEST_F(FFTTest, Size1) {
    const int N = 1;
    std::vector<fft_data> input = {{3.14, 2.71}};
    std::vector<fft_data> output(N);
    
    fft_object fft = fft_init(N, 1);
    ASSERT_NE(fft, nullptr);
    
    fft_exec(fft, input.data(), output.data());
    
    // FFT of size 1 is identity
    EXPECT_NEAR(output[0].re, input[0].re, EPSILON);
    EXPECT_NEAR(output[0].im, input[0].im, EPSILON);
    
    free_fft(fft);
}

TEST_F(FFTTest, Size2) {
    const int N = 2;
    std::vector<fft_data> input = {{1.0, 0.0}, {2.0, 0.0}};
    std::vector<fft_data> output(N);
    
    fft_object fft = fft_init(N, 1);
    ASSERT_NE(fft, nullptr);
    
    fft_exec(fft, input.data(), output.data());
    
    // Manual calculation: X[0] = x[0] + x[1] = 3, X[1] = x[0] - x[1] = -1
    EXPECT_NEAR(output[0].re, 3.0, EPSILON);
    EXPECT_NEAR(output[0].im, 0.0, EPSILON);
    EXPECT_NEAR(output[1].re, -1.0, EPSILON);
    EXPECT_NEAR(output[1].im, 0.0, EPSILON);
    
    free_fft(fft);
}

TEST_F(FFTTest, AllZeros) {
    const int N = 64;
    std::vector<fft_data> input(N, {0.0, 0.0});
    std::vector<fft_data> output(N);
    
    fft_object fft = fft_init(N, 1);
    ASSERT_NE(fft, nullptr);
    
    fft_exec(fft, input.data(), output.data());
    
    for (int i = 0; i < N; ++i) {
        EXPECT_NEAR(output[i].re, 0.0, EPSILON) << "at index " << i;
        EXPECT_NEAR(output[i].im, 0.0, EPSILON) << "at index " << i;
    }
    
    free_fft(fft);
}

TEST_F(FFTTest, LargeSize) {
    const int N = 8192;
    auto input = generateRandomSignal(N);
    std::vector<fft_data> output(N);
    std::vector<fft_data> roundtrip(N);
    
    fft_object fft_fwd = fft_init(N, 1);
    fft_object fft_inv = fft_init(N, -1);
    ASSERT_NE(fft_fwd, nullptr);
    ASSERT_NE(fft_inv, nullptr);
    
    fft_exec(fft_fwd, input.data(), output.data());
    fft_exec(fft_inv, output.data(), roundtrip.data());
    
    for (int i = 0; i < N; ++i) {
        roundtrip[i].re /= N;
        roundtrip[i].im /= N;
    }
    
    double max_error = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = relativeError(roundtrip[i], input[i]);
        max_error = std::max(max_error, err);
    }
    
    EXPECT_LT(max_error, 1e-8);
    
    free_fft(fft_fwd);
    free_fft(fft_inv);
}

//==============================================================================
// SHIFT THEOREM
//==============================================================================

TEST_F(FFTTest, CircularShiftTheorem) {
    const int N = 64;
    const int shift = 7;
    auto input = generateRandomSignal(N);
    
    // Create circularly shifted version
    std::vector<fft_data> shifted(N);
    for (int i = 0; i < N; ++i) {
        shifted[i] = input[(i + shift) % N];
    }
    
    std::vector<fft_data> fft_orig(N), fft_shifted(N);
    
    fft_object fft = fft_init(N, 1);
    ASSERT_NE(fft, nullptr);
    
    fft_exec(fft, input.data(), fft_orig.data());
    fft_exec(fft, shifted.data(), fft_shifted.data());
    
    // Circular shift in time = phase shift in frequency
    // X_shifted[k] = X[k] * exp(-2πi*k*shift/N)
    for (int k = 0; k < N; ++k) {
        double angle = -2.0 * M_PI * k * shift / N;
        double expected_re = fft_orig[k].re * std::cos(angle) - 
                            fft_orig[k].im * std::sin(angle);
        double expected_im = fft_orig[k].re * std::sin(angle) + 
                            fft_orig[k].im * std::cos(angle);
        
        EXPECT_NEAR(fft_shifted[k].re, expected_re, LOOSE_EPSILON) << "at k=" << k;
        EXPECT_NEAR(fft_shifted[k].im, expected_im, LOOSE_EPSILON) << "at k=" << k;
    }
    
    free_fft(fft);
}

//==============================================================================
// NUMERICAL ACCURACY
//==============================================================================

TEST_F(FFTTest, NumericalPrecision) {
    const int N = 1024;
    auto input = generateRandomSignal(N);
    std::vector<fft_data> output(N);
    
    fft_object fft = fft_init(N, 1);
    ASSERT_NE(fft, nullptr);
    
    fft_exec(fft, input.data(), output.data());
    
    // Compare with naive DFT (slow but accurate)
    auto reference = naiveDFT(input, 1);
    
    double max_rel_error = 0.0;
    for (int i = 0; i < N; ++i) {
        double err = relativeError(output[i], reference[i]);
        max_rel_error = std::max(max_rel_error, err);
    }
    
    // Should be within double precision tolerance
    EXPECT_LT(max_rel_error, 1e-11);
    
    free_fft(fft);
}

//==============================================================================
// STRESS TESTS
//==============================================================================

TEST_F(FFTTest, MultipleExecutions) {
    const int N = 256;
    const int num_runs = 100;
    
    fft_object fft = fft_init(N, 1);
    ASSERT_NE(fft, nullptr);
    
    for (int run = 0; run < num_runs; ++run) {
        auto input = generateRandomSignal(N);
        std::vector<fft_data> output(N);
        
        fft_exec(fft, input.data(), output.data());
        
        // Verify Parseval's theorem
        double error = parsevalError(input, output);
        EXPECT_LT(error, 1e-9) << "Failed at run " << run;
    }
    
    free_fft(fft);
}

TEST_F(FFTTest, ConcurrentObjects) {
    const int N = 128;
    const int num_objects = 10;
    
    std::vector<fft_object> ffts;
    std::vector<std::vector<fft_data>> inputs;
    std::vector<std::vector<fft_data>> outputs;
    
    // Create multiple FFT objects
    for (int i = 0; i < num_objects; ++i) {
        fft_object fft = fft_init(N, 1);
        ASSERT_NE(fft, nullptr);
        ffts.push_back(fft);
        inputs.push_back(generateRandomSignal(N));
        outputs.push_back(std::vector<fft_data>(N));
    }
    
    // Execute all
    for (int i = 0; i < num_objects; ++i) {
        fft_exec(ffts[i], inputs[i].data(), outputs[i].data());
    }
    
    // Verify all
    for (int i = 0; i < num_objects; ++i) {
        double error = parsevalError(inputs[i], outputs[i]);
        EXPECT_LT(error, 1e-9) << "Failed for object " << i;
    }
    
    // Cleanup
    for (auto fft : ffts) {
        free_fft(fft);
    }
}

//==============================================================================
// MAIN
//==============================================================================

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}