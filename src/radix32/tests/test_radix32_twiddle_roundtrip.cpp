/**
 * @file test_radix32_twiddle_roundtrip.cpp
 * @brief Roundtrip tests for the twiddle radix-32 AVX2 codelet
 *
 * forward(x) → X, backward(X) → y  ⇒  y[i] == 32 · x[i]
 *
 * The twiddle codelet operates on [32 stripes][K samples] and requires:
 *   - Pass-1 DIT twiddles (radix4_dit_stage_twiddles_blocked2_t)
 *   - Pass-2 DIF twiddles (tw_stage8_t, BLOCKED8 for K ≤ 256)
 *   - Temp buffer: [32][K] doubles × 2 (re + im)
 *
 * Memory layout: data[stripe * K + k], stripe ∈ {0..31}, k ∈ {0..K-1}
 *
 * The roundtrip factor is 32 (not 32K) because the k-dependent twiddles
 * are unit-modulus and cancel in forward→backward.
 */

#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

extern "C" {
#include "fft_radix32_avx2.h"
}

/* ========================================================================
 * CONSTANTS
 * ======================================================================*/

static constexpr size_t N       = 32;
static constexpr double ABS_TOL = 1e-12;

/* ========================================================================
 * ALIGNED BUFFER
 * ======================================================================*/

struct AlignedBuf {
    double *ptr;
    size_t count;

    explicit AlignedBuf(size_t n)
        : ptr(static_cast<double *>(std::aligned_alloc(32, n * sizeof(double))))
        , count(n)
    {
        assert(ptr && "aligned_alloc failed");
        std::memset(ptr, 0, n * sizeof(double));
    }
    ~AlignedBuf() { std::free(ptr); }

    AlignedBuf(const AlignedBuf &) = delete;
    AlignedBuf &operator=(const AlignedBuf &) = delete;

    double &operator[](size_t i)       { return ptr[i]; }
    double  operator[](size_t i) const { return ptr[i]; }
};

/* ========================================================================
 * TWIDDLE GENERATION
 * ======================================================================*/

/**
 * @brief Generate pass-1 DIT twiddles (BLOCKED2) for a 32K-point stage
 *
 * W1[k] = exp(-2πi·k/(32K))   applied to DIT input b=1
 * W2[k] = exp(-2πi·2k/(32K))  applied to DIT input b=2
 * W3 = W1·W2 is derived on-the-fly
 */
static void gen_pass1_twiddles(size_t K, double *tw_re, double *tw_im)
{
    for (size_t k = 0; k < K; k++) {
        double ang1 = -2.0 * M_PI * (double)k / (32.0 * (double)K);
        tw_re[0 * K + k] = std::cos(ang1);
        tw_im[0 * K + k] = std::sin(ang1);
        tw_re[1 * K + k] = std::cos(2.0 * ang1);
        tw_im[1 * K + k] = std::sin(2.0 * ang1);
    }
}

/**
 * @brief Generate pass-2 DIF twiddles (BLOCKED8) for a 32K-point stage
 *
 * W[g][k] = exp(-2πi·(g+1)·k/(8K))  for g = 0..7
 */
static void gen_pass2_twiddles(size_t K,
                               double *tw_re[8], double *tw_im[8])
{
    for (int g = 0; g < 8; g++) {
        for (size_t k = 0; k < K; k++) {
            double angle = -2.0 * M_PI * (double)(g + 1) * (double)k
                         / (8.0 * (double)K);
            tw_re[g][k] = std::cos(angle);
            tw_im[g][k] = std::sin(angle);
        }
    }
}

/* ========================================================================
 * TWIDDLE TABLE WRAPPER — owns memory, builds structs for the codelet
 * ======================================================================*/

struct TwiddleTables {
    AlignedBuf p1_re, p1_im;
    std::vector<AlignedBuf *> p2_re_bufs, p2_im_bufs;
    double *p2_re_ptrs[8], *p2_im_ptrs[8];

    radix4_dit_stage_twiddles_blocked2_t pass1;
    tw_stage8_t pass2;

    explicit TwiddleTables(size_t K)
        : p1_re(2 * K), p1_im(2 * K)
    {
        /* Pass 1 */
        gen_pass1_twiddles(K, p1_re.ptr, p1_im.ptr);
        pass1.re = p1_re.ptr;
        pass1.im = p1_im.ptr;
        pass1.K  = K;

        /* Pass 2 */
        for (int j = 0; j < 8; j++) {
            p2_re_bufs.push_back(new AlignedBuf(K));
            p2_im_bufs.push_back(new AlignedBuf(K));
            p2_re_ptrs[j] = p2_re_bufs.back()->ptr;
            p2_im_ptrs[j] = p2_im_bufs.back()->ptr;
        }
        gen_pass2_twiddles(K, p2_re_ptrs, p2_im_ptrs);

        pass2.mode = TW_MODE_BLOCKED8;
        for (int j = 0; j < 8; j++) {
            pass2.b8.re[j] = p2_re_ptrs[j];
            pass2.b8.im[j] = p2_im_ptrs[j];
        }
        pass2.b8.K = K;
    }

    ~TwiddleTables() {
        for (auto *p : p2_re_bufs) delete p;
        for (auto *p : p2_im_bufs) delete p;
    }

    TwiddleTables(const TwiddleTables &) = delete;
    TwiddleTables &operator=(const TwiddleTables &) = delete;
};

/* ========================================================================
 * PARAMETERISED TEST FIXTURE
 *
 * Parameterised on K (samples per stripe).
 * Allocates: in, freq, out, temp — each [32][K] doubles.
 * ======================================================================*/

class Radix32TwiddleRoundtrip : public ::testing::TestWithParam<size_t> {
protected:
    size_t K     = 0;
    size_t total = 0;

    std::unique_ptr<AlignedBuf> in_re, in_im;
    std::unique_ptr<AlignedBuf> freq_re, freq_im;
    std::unique_ptr<AlignedBuf> out_re, out_im;
    std::unique_ptr<AlignedBuf> temp_re, temp_im;
    std::unique_ptr<TwiddleTables> tw;

    void SetUp() override {
        K     = GetParam();
        total = N * K;
        in_re   = std::make_unique<AlignedBuf>(total);
        in_im   = std::make_unique<AlignedBuf>(total);
        freq_re = std::make_unique<AlignedBuf>(total);
        freq_im = std::make_unique<AlignedBuf>(total);
        out_re  = std::make_unique<AlignedBuf>(total);
        out_im  = std::make_unique<AlignedBuf>(total);
        temp_re = std::make_unique<AlignedBuf>(total);
        temp_im = std::make_unique<AlignedBuf>(total);
        tw      = std::make_unique<TwiddleTables>(K);
    }

    /* --- fill helpers --- */

    void fill_random(uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (size_t i = 0; i < total; i++) {
            (*in_re)[i] = dist(rng);
            (*in_im)[i] = dist(rng);
        }
    }

    void fill_constant(double re_val, double im_val) {
        for (size_t i = 0; i < total; i++) {
            (*in_re)[i] = re_val;
            (*in_im)[i] = im_val;
        }
    }

    void zero_freq() {
        std::memset(freq_re->ptr, 0, total * sizeof(double));
        std::memset(freq_im->ptr, 0, total * sizeof(double));
    }

    void zero_out() {
        std::memset(out_re->ptr, 0, total * sizeof(double));
        std::memset(out_im->ptr, 0, total * sizeof(double));
    }

    /* --- codelet wrappers --- */

    void forward() {
        /* clear temp to catch uninitialised reads */
        std::memset(temp_re->ptr, 0, total * sizeof(double));
        std::memset(temp_im->ptr, 0, total * sizeof(double));
        radix32_stage_forward_avx2(
            K, in_re->ptr, in_im->ptr, freq_re->ptr, freq_im->ptr,
            &tw->pass1, &tw->pass2, temp_re->ptr, temp_im->ptr);
    }

    void backward() {
        std::memset(temp_re->ptr, 0, total * sizeof(double));
        std::memset(temp_im->ptr, 0, total * sizeof(double));
        radix32_stage_backward_avx2(
            K, freq_re->ptr, freq_im->ptr, out_re->ptr, out_im->ptr,
            &tw->pass1, &tw->pass2, temp_re->ptr, temp_im->ptr);
    }

    /* --- verification --- */

    /** Check y == 32·x for every element */
    void check_roundtrip(double tol = ABS_TOL) {
        for (size_t s = 0; s < N; s++) {
            for (size_t k = 0; k < K; k++) {
                size_t idx = s * K + k;
                double expect_re = (*in_re)[idx] * N;
                double expect_im = (*in_im)[idx] * N;
                EXPECT_NEAR((*out_re)[idx], expect_re, tol)
                    << "stripe=" << s << " k=" << k << " (re)";
                EXPECT_NEAR((*out_im)[idx], expect_im, tol)
                    << "stripe=" << s << " k=" << k << " (im)";
            }
        }
    }

    /** Collect the worst-case roundtrip error */
    double max_roundtrip_error() {
        double worst = 0.0;
        for (size_t i = 0; i < total; i++) {
            double er = std::fabs((*out_re)[i] - (*in_re)[i] * N);
            double ei = std::fabs((*out_im)[i] - (*in_im)[i] * N);
            if (er > worst) worst = er;
            if (ei > worst) worst = ei;
        }
        return worst;
    }
};

/* K values to test — all ≤ 256 → BLOCKED8 twiddle mode.
 * K must be ≥ 8 and a multiple of 4 (pass1 requires K ≥ 8). */
INSTANTIATE_TEST_SUITE_P(
    KValues,
    Radix32TwiddleRoundtrip,
    ::testing::Values(8, 16, 32, 64, 128, 256),
    [](const ::testing::TestParamInfo<size_t> &info) {
        return "K" + std::to_string(info.param);
    }
);

/* ========================================================================
 * TESTS
 * ======================================================================*/

/* 1. Basic random roundtrip */
TEST_P(Radix32TwiddleRoundtrip, RandomRoundtrip) {
    fill_random(42);
    forward();
    backward();
    check_roundtrip();
}

/* 2. Multi-seed robustness */
TEST_P(Radix32TwiddleRoundtrip, RandomRoundtripMultiSeed) {
    for (uint64_t seed = 200; seed < 210; seed++) {
        fill_random(seed);
        forward();
        backward();
        check_roundtrip();
    }
}

/* 3. DC input: all ones → forward gives something, backward recovers 32·x */
TEST_P(Radix32TwiddleRoundtrip, DcRoundtrip) {
    fill_constant(1.0, 0.0);
    forward();
    backward();
    check_roundtrip();
}

/* 4. Zero input: everything stays zero */
TEST_P(Radix32TwiddleRoundtrip, ZeroInput) {
    fill_constant(0.0, 0.0);
    forward();
    backward();

    for (size_t i = 0; i < total; i++) {
        EXPECT_EQ((*out_re)[i], 0.0) << "index=" << i;
        EXPECT_EQ((*out_im)[i], 0.0) << "index=" << i;
    }
}

/* 5. Impulse at stripe 0 — every sample k independently */
TEST_P(Radix32TwiddleRoundtrip, ImpulseRoundtrip) {
    fill_constant(0.0, 0.0);
    for (size_t k = 0; k < K; k++) {
        (*in_re)[0 * K + k] = 1.0;  /* stripe 0, sample k */
    }

    forward();
    backward();
    check_roundtrip();
}

/* 6. Pure cosine across stripes — roundtrip recovery */
TEST_P(Radix32TwiddleRoundtrip, PureCosineRoundtrip) {
    fill_constant(0.0, 0.0);
    const size_t freq = 5;
    for (size_t n = 0; n < N; n++) {
        double val = std::cos(2.0 * M_PI * freq * n / N);
        for (size_t k = 0; k < K; k++) {
            (*in_re)[n * K + k] = val;
        }
    }

    forward();
    backward();
    check_roundtrip();
}

/* 7. Purely imaginary input */
TEST_P(Radix32TwiddleRoundtrip, PurelyImaginary) {
    fill_constant(0.0, 0.0);
    std::mt19937_64 rng(9999);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < total; i++) {
        (*in_im)[i] = dist(rng);
    }

    forward();
    backward();
    check_roundtrip();
}

/* 8. Large amplitude — scaled tolerance */
TEST_P(Radix32TwiddleRoundtrip, LargeAmplitude) {
    std::mt19937_64 rng(5555);
    std::uniform_real_distribution<double> dist(-1e6, 1e6);
    for (size_t i = 0; i < total; i++) {
        (*in_re)[i] = dist(rng);
        (*in_im)[i] = dist(rng);
    }

    forward();
    backward();

    const double tol = 1e-6;
    check_roundtrip(tol);
}

/* 9. Per-sample independence: each k offset gets different data, verify
 *    no cross-contamination between k positions */
TEST_P(Radix32TwiddleRoundtrip, SampleIndependence) {
    std::mt19937_64 rng(0xDEAD);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    for (size_t s = 0; s < N; s++) {
        for (size_t k = 0; k < K; k++) {
            (*in_re)[s * K + k] = dist(rng);
            (*in_im)[s * K + k] = dist(rng);
        }
    }

    forward();
    backward();
    check_roundtrip();
}

/* 10. Linearity: F(a·x + b·y) == a·F(x) + b·F(y)
 *
 * Uses separate allocations to avoid aliasing the fixture buffers.
 */
TEST_P(Radix32TwiddleRoundtrip, Linearity) {
    const double a = 2.71828, b = -1.41421;

    AlignedBuf x_re(total), x_im(total);
    AlignedBuf y_re(total), y_im(total);
    AlignedBuf Fx_re(total), Fx_im(total);
    AlignedBuf Fy_re(total), Fy_im(total);
    AlignedBuf Fz_re(total), Fz_im(total);
    AlignedBuf t1(total), t2(total);

    std::mt19937_64 rng(31415);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < total; i++) {
        x_re[i] = dist(rng); x_im[i] = dist(rng);
        y_re[i] = dist(rng); y_im[i] = dist(rng);
    }

    /* z = a·x + b·y → in_re/in_im */
    for (size_t i = 0; i < total; i++) {
        (*in_re)[i] = a * x_re[i] + b * y_re[i];
        (*in_im)[i] = a * x_im[i] + b * y_im[i];
    }

    /* F(x) */
    std::memset(t1.ptr, 0, total * sizeof(double));
    std::memset(t2.ptr, 0, total * sizeof(double));
    radix32_stage_forward_avx2(
        K, x_re.ptr, x_im.ptr, Fx_re.ptr, Fx_im.ptr,
        &tw->pass1, &tw->pass2, t1.ptr, t2.ptr);

    /* F(y) */
    std::memset(t1.ptr, 0, total * sizeof(double));
    std::memset(t2.ptr, 0, total * sizeof(double));
    radix32_stage_forward_avx2(
        K, y_re.ptr, y_im.ptr, Fy_re.ptr, Fy_im.ptr,
        &tw->pass1, &tw->pass2, t1.ptr, t2.ptr);

    /* F(z) */
    forward();

    /* Verify: F(z)[i] == a·F(x)[i] + b·F(y)[i] */
    for (size_t i = 0; i < total; i++) {
        double exp_re = a * Fx_re[i] + b * Fy_re[i];
        double exp_im = a * Fx_im[i] + b * Fy_im[i];
        EXPECT_NEAR((*freq_re)[i], exp_re, ABS_TOL)
            << "linearity re, index=" << i;
        EXPECT_NEAR((*freq_im)[i], exp_im, ABS_TOL)
            << "linearity im, index=" << i;
    }
}

/* 11. Forward energy preservation (Parseval-like)
 *
 * For a 32-point DFT: Σ|X[k]|² = 32 · Σ|x[n]|²
 * The twiddle codelet's inter-stage factors are unit-modulus, so
 * energy scales by 32 per sample k. */
TEST_P(Radix32TwiddleRoundtrip, EnergyPreservation) {
    fill_random(77777);
    forward();

    for (size_t k = 0; k < K; k++) {
        double input_energy  = 0.0;
        double output_energy = 0.0;
        for (size_t s = 0; s < N; s++) {
            size_t idx = s * K + k;
            input_energy  += (*in_re)[idx]   * (*in_re)[idx]
                           + (*in_im)[idx]   * (*in_im)[idx];
            output_energy += (*freq_re)[idx] * (*freq_re)[idx]
                           + (*freq_im)[idx] * (*freq_im)[idx];
        }
        /* |X|² = N · |x|² */
        EXPECT_NEAR(output_energy, N * input_energy, 1e-8)
            << "Parseval at k=" << k;
    }
}

/* 12. Conjugate symmetry: if input is real, X[N-k] = conj(X[k])
 *     (per k sample, across the 32 stripes)
 *
 * Because the twiddle codelet's output permutation is bin-major
 * [b*8+d][k], we verify on the raw output directly. The DFT bin
 * for output stripe s is known only in the context of the full
 * decomposition (with twiddles), so instead we verify through
 * roundtrip: backward of forward should recover 32·x regardless.
 * This test is kept as a roundtrip variant with purely real input. */
TEST_P(Radix32TwiddleRoundtrip, RealInputRoundtrip) {
    std::mt19937_64 rng(2024);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < total; i++) {
        (*in_re)[i] = dist(rng);
        (*in_im)[i] = 0.0;
    }

    forward();
    backward();
    check_roundtrip();
}

/* 13. Alternating sign pattern: x[n] = (-1)^n shifts DFT by N/2 */
TEST_P(Radix32TwiddleRoundtrip, AlternatingSignRoundtrip) {
    fill_constant(0.0, 0.0);
    for (size_t n = 0; n < N; n++) {
        double val = (n % 2 == 0) ? 1.0 : -1.0;
        for (size_t k = 0; k < K; k++) {
            (*in_re)[n * K + k] = val;
        }
    }

    forward();
    backward();
    check_roundtrip();
}

/* 14. Single non-zero stripe — verify isolation across stripes */
TEST_P(Radix32TwiddleRoundtrip, SingleStripeIsolation) {
    for (size_t target_stripe = 0; target_stripe < N; target_stripe += 7) {
        fill_constant(0.0, 0.0);
        std::mt19937_64 rng(target_stripe + 1000);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (size_t k = 0; k < K; k++) {
            (*in_re)[target_stripe * K + k] = dist(rng);
            (*in_im)[target_stripe * K + k] = dist(rng);
        }

        forward();
        backward();
        check_roundtrip();
    }
}

/* 15. Error scaling with K — verify error stays bounded as K grows */
TEST_P(Radix32TwiddleRoundtrip, ErrorBound) {
    fill_random(12345);
    forward();
    backward();

    double worst = max_roundtrip_error();
    EXPECT_LT(worst, 1e-10)
        << "K=" << K << " worst roundtrip error: " << worst;
}
