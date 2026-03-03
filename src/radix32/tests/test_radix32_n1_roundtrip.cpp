/**
 * @file test_radix32_n1_roundtrip.cpp
 * @brief Roundtrip tests for the twiddle-less radix-32 AVX2 codelet
 *
 * forward(x) → X, backward(X) → y  ⇒  y[i] == 32 · x[i]
 *
 * The n1 codelet processes 4 independent 32-point DFTs in parallel
 * (one AVX2 vector width). Stride = 4 doubles between stripes.
 */

#include <gtest/gtest.h>

#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

/* ---------- cross-platform alloc + macros ---------- */
#include "../fft_radix32_platform.h"

/* ---------- pull in the n1 header (all-inline) ---------- */
extern "C" {
#include "fft_radix32_avx2_n1.h"
}

/* ========================================================================
 * HELPERS
 * ======================================================================*/

static constexpr size_t N       = 32;          /* DFT length            */
static constexpr size_t VW      = 4;           /* AVX2 double lanes     */
static constexpr size_t STRIDE  = VW;          /* one vector per stripe */
static constexpr size_t NELEM   = N * VW;      /* total doubles         */
static constexpr double INV_N   = 1.0 / N;
static constexpr double ABS_TOL = 1e-12;       /* ~2^{-40}, generous    */

/* 32-byte aligned allocation (portable) */
struct AlignedBuf {
    double *ptr;
    explicit AlignedBuf(size_t count)
        : ptr(static_cast<double *>(
              r32_aligned_alloc(32, count * sizeof(double))))
    {
        assert(ptr != nullptr && "r32_aligned_alloc failed");
        std::memset(ptr, 0, count * sizeof(double));
    }
    ~AlignedBuf() { r32_aligned_free(ptr); }
    double &operator[](size_t i)       { return ptr[i]; }
    double  operator[](size_t i) const { return ptr[i]; }
};

/* Brute-force single 32-point DFT for reference (forward, no scaling) */
static void dft32_reference(
    const double *xr, const double *xi, size_t x_stride,
    double *Xr, double *Xi, size_t X_stride,
    int direction)  /* +1 = forward (−2πi), −1 = backward (+2πi) */
{
    const double sign = (direction > 0) ? -1.0 : +1.0;
    for (size_t k = 0; k < N; k++) {
        double sr = 0.0, si = 0.0;
        for (size_t n = 0; n < N; n++) {
            double angle = sign * 2.0 * M_PI * (double)(k * n) / (double)N;
            double wr = std::cos(angle);
            double wi = std::sin(angle);
            sr += xr[n * x_stride] * wr - xi[n * x_stride] * wi;
            si += xr[n * x_stride] * wi + xi[n * x_stride] * wr;
        }
        Xr[k * X_stride] = sr;
        Xi[k * X_stride] = si;
    }
}

/* ========================================================================
 * TEST FIXTURE
 * ======================================================================*/

class Radix32N1Roundtrip : public ::testing::Test {
protected:
    AlignedBuf in_re{NELEM}, in_im{NELEM};
    AlignedBuf freq_re{NELEM}, freq_im{NELEM};
    AlignedBuf out_re{NELEM}, out_im{NELEM};

    void fill_random(uint64_t seed) {
        std::mt19937_64 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (size_t i = 0; i < NELEM; i++) {
            in_re[i] = dist(rng);
            in_im[i] = dist(rng);
        }
    }

    void fill_constant(double re_val, double im_val) {
        for (size_t i = 0; i < NELEM; i++) {
            in_re[i] = re_val;
            in_im[i] = im_val;
        }
    }

    void forward()  {
        fft_radix32_n1_forward_avx2(
            in_re.ptr, in_im.ptr, freq_re.ptr, freq_im.ptr,
            STRIDE, STRIDE);
    }

    void backward() {
        fft_radix32_n1_backward_avx2(
            freq_re.ptr, freq_im.ptr, out_re.ptr, out_im.ptr,
            STRIDE, STRIDE);
    }

    /* Check y == N·x for all 4 lanes × 32 stripes */
    void check_roundtrip(double tol = ABS_TOL) {
        for (size_t s = 0; s < N; s++) {
            for (size_t lane = 0; lane < VW; lane++) {
                size_t idx = s * STRIDE + lane;
                double expect_re = in_re[idx] * N;
                double expect_im = in_im[idx] * N;
                EXPECT_NEAR(out_re[idx], expect_re, tol)
                    << "stripe=" << s << " lane=" << lane << " (re)";
                EXPECT_NEAR(out_im[idx], expect_im, tol)
                    << "stripe=" << s << " lane=" << lane << " (im)";
            }
        }
    }

    /* Check forward output against brute-force DFT per lane */
    void check_forward_vs_reference(double tol = ABS_TOL) {
        double ref_re[N], ref_im[N];
        for (size_t lane = 0; lane < VW; lane++) {
            dft32_reference(
                &in_re.ptr[lane], &in_im.ptr[lane], STRIDE,
                ref_re, ref_im, 1,
                +1 /* forward */);
            for (size_t k = 0; k < N; k++) {
                size_t idx = k * STRIDE + lane;
                EXPECT_NEAR(freq_re[idx], ref_re[k], tol)
                    << "lane=" << lane << " bin=" << k << " (re)";
                EXPECT_NEAR(freq_im[idx], ref_im[k], tol)
                    << "lane=" << lane << " bin=" << k << " (im)";
            }
        }
    }
};

/* ========================================================================
 * TESTS
 * ======================================================================*/

/* Basic roundtrip: random data, fwd→bwd, check N·x */
TEST_F(Radix32N1Roundtrip, RandomRoundtrip) {
    fill_random(42);
    forward();
    backward();
    check_roundtrip();
}

/* Multiple seeds for broader coverage */
TEST_F(Radix32N1Roundtrip, RandomRoundtripMultiSeed) {
    for (uint64_t seed = 100; seed < 120; seed++) {
        fill_random(seed);
        forward();
        backward();
        check_roundtrip();
    }
}

/* DC input: all ones → forward should give 32+0j at bin 0, zeros elsewhere */
TEST_F(Radix32N1Roundtrip, DcInput) {
    fill_constant(1.0, 0.0);
    forward();

    for (size_t lane = 0; lane < VW; lane++) {
        /* Bin 0: should be N */
        EXPECT_NEAR(freq_re[0 * STRIDE + lane], (double)N, ABS_TOL)
            << "lane=" << lane << " bin 0 re";
        EXPECT_NEAR(freq_im[0 * STRIDE + lane], 0.0, ABS_TOL)
            << "lane=" << lane << " bin 0 im";

        /* Bins 1..31: should be zero */
        for (size_t k = 1; k < N; k++) {
            EXPECT_NEAR(freq_re[k * STRIDE + lane], 0.0, ABS_TOL)
                << "lane=" << lane << " bin=" << k << " re";
            EXPECT_NEAR(freq_im[k * STRIDE + lane], 0.0, ABS_TOL)
                << "lane=" << lane << " bin=" << k << " im";
        }
    }

    backward();
    check_roundtrip();
}

/* Zero input: everything should stay zero */
TEST_F(Radix32N1Roundtrip, ZeroInput) {
    fill_constant(0.0, 0.0);
    forward();
    backward();

    for (size_t i = 0; i < NELEM; i++) {
        EXPECT_EQ(out_re[i], 0.0) << "index=" << i;
        EXPECT_EQ(out_im[i], 0.0) << "index=" << i;
    }
}

/* Single impulse at position 0 → flat spectrum (all bins = 1+0j) */
TEST_F(Radix32N1Roundtrip, Impulse) {
    fill_constant(0.0, 0.0);
    /* Set stripe 0, all lanes = 1.0 */
    for (size_t lane = 0; lane < VW; lane++) {
        in_re[0 * STRIDE + lane] = 1.0;
    }

    forward();

    for (size_t lane = 0; lane < VW; lane++) {
        for (size_t k = 0; k < N; k++) {
            EXPECT_NEAR(freq_re[k * STRIDE + lane], 1.0, ABS_TOL)
                << "lane=" << lane << " bin=" << k << " re";
            EXPECT_NEAR(freq_im[k * STRIDE + lane], 0.0, ABS_TOL)
                << "lane=" << lane << " bin=" << k << " im";
        }
    }

    backward();
    check_roundtrip();
}

/* Pure cosine at frequency f → energy in bins f and N-f */
TEST_F(Radix32N1Roundtrip, PureCosine) {
    fill_constant(0.0, 0.0);
    const size_t freq = 5;
    for (size_t n = 0; n < N; n++) {
        double val = std::cos(2.0 * M_PI * freq * n / N);
        for (size_t lane = 0; lane < VW; lane++) {
            in_re[n * STRIDE + lane] = val;
        }
    }

    forward();

    for (size_t lane = 0; lane < VW; lane++) {
        for (size_t k = 0; k < N; k++) {
            double expect_re = 0.0;
            if (k == freq || k == N - freq) expect_re = N / 2.0;

            EXPECT_NEAR(freq_re[k * STRIDE + lane], expect_re, ABS_TOL)
                << "lane=" << lane << " bin=" << k << " re";
            EXPECT_NEAR(freq_im[k * STRIDE + lane], 0.0, ABS_TOL)
                << "lane=" << lane << " bin=" << k << " im";
        }
    }

    backward();
    check_roundtrip();
}

/* Forward output vs brute-force O(N²) DFT reference */
TEST_F(Radix32N1Roundtrip, ForwardVsReference) {
    fill_random(7777);
    forward();
    check_forward_vs_reference();
}

/* Per-lane independence: different data in each lane, verify no cross-talk */
TEST_F(Radix32N1Roundtrip, LaneIndependence) {
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);

    /* Fill each lane with different data */
    for (size_t s = 0; s < N; s++) {
        for (size_t lane = 0; lane < VW; lane++) {
            in_re[s * STRIDE + lane] = dist(rng);
            in_im[s * STRIDE + lane] = dist(rng);
        }
    }

    forward();

    /* Verify each lane independently against scalar reference */
    check_forward_vs_reference();

    backward();
    check_roundtrip();
}

/* Purely imaginary input */
TEST_F(Radix32N1Roundtrip, PurelyImaginary) {
    fill_constant(0.0, 0.0);
    std::mt19937_64 rng(9999);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < NELEM; i++) {
        in_im[i] = dist(rng);
    }

    forward();
    check_forward_vs_reference();
    backward();
    check_roundtrip();
}

/* Large amplitude to stress numerical precision */
TEST_F(Radix32N1Roundtrip, LargeAmplitude) {
    std::mt19937_64 rng(5555);
    std::uniform_real_distribution<double> dist(-1e6, 1e6);
    for (size_t i = 0; i < NELEM; i++) {
        in_re[i] = dist(rng);
        in_im[i] = dist(rng);
    }

    forward();
    backward();

    /* Scale tolerance with amplitude */
    const double tol = 1e-6;
    for (size_t s = 0; s < N; s++) {
        for (size_t lane = 0; lane < VW; lane++) {
            size_t idx = s * STRIDE + lane;
            EXPECT_NEAR(out_re[idx], in_re[idx] * N, tol)
                << "stripe=" << s << " lane=" << lane << " (re)";
            EXPECT_NEAR(out_im[idx], in_im[idx] * N, tol)
                << "stripe=" << s << " lane=" << lane << " (im)";
        }
    }
}
