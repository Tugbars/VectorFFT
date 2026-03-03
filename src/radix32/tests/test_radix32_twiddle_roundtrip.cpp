/**
 * @file test_radix32_twiddle_roundtrip.cpp
 * @brief Tests for the twiddle radix-32 AVX2 codelet
 *
 * The twiddle codelet (radix32_stage_{forward,backward}_avx2) operates on
 * [32 stripes][K samples] and is designed as a stage within a larger FFT.
 *
 * Key property: forward→backward roundtrip produces 32·x ONLY when the
 * stage twiddles are trivial (all 1), because the backward pass applies
 * pass1_bwd→pass2_bwd in the same order as forward, not reversed.
 *
 * Test strategy:
 *  A) ROUNDTRIP with trivial twiddles — tests DIT-4/DIF-8 butterfly cores
 *     through the twiddle codepath (BLOCKED8 loader, U=2 pipeline, etc.)
 *  B) FORWARD vs SCALAR REFERENCE — verifies twiddle application correctness
 *     by comparing against a brute-force simulation of the 4×8 decomposition
 *  C) STRUCTURAL — linearity, energy preservation, twiddle consistency
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

extern "C" {
#include "fft_radix32_avx2.h"
}

/* ========================================================================
 * CONSTANTS
 * ======================================================================*/

static constexpr size_t N       = 32;
static constexpr double ABS_TOL = 1e-12;

/* ========================================================================
 * ALIGNED BUFFER (portable)
 * ======================================================================*/

struct AlignedBuf {
    double *ptr;
    size_t count;

    explicit AlignedBuf(size_t n)
        : ptr(static_cast<double *>(r32_aligned_alloc(32, n * sizeof(double))))
        , count(n)
    {
        assert(ptr && "r32_aligned_alloc failed");
        std::memset(ptr, 0, n * sizeof(double));
    }
    ~AlignedBuf() { r32_aligned_free(ptr); }

    AlignedBuf(const AlignedBuf &) = delete;
    AlignedBuf &operator=(const AlignedBuf &) = delete;

    double &operator[](size_t i)       { return ptr[i]; }
    double  operator[](size_t i) const { return ptr[i]; }
};

/* ========================================================================
 * TWIDDLE TABLE BUILDER
 * ======================================================================*/

struct TwiddleTables {
    AlignedBuf p1_re, p1_im;
    std::vector<AlignedBuf *> p2_re_bufs, p2_im_bufs;
    double *p2_re_ptrs[8], *p2_im_ptrs[8];

    radix4_dit_stage_twiddles_blocked2_t pass1;
    tw_stage8_t pass2;

    /**
     * Build a twiddle table with an arbitrary generator.
     * @param K        samples per stripe
     * @param p1_gen   Pass-1 DIT twiddle: p1_gen(m, k, K) → complex
     *                 m ∈ {1,2}, k ∈ {0..K-1}
     * @param p2_gen   Pass-2 DIF twiddle: p2_gen(g, k, K) → complex
     *                 g ∈ {0..7} (applied to DIF-8 input g+1), k ∈ {0..K-1}
     */
    template <typename P1Gen, typename P2Gen>
    TwiddleTables(size_t K, P1Gen p1_gen, P2Gen p2_gen)
        : p1_re(2 * K), p1_im(2 * K)
    {
        /* Pass 1: W1[k], W2[k].  W3 = W1·W2 derived on-the-fly */
        for (size_t k = 0; k < K; k++) {
            auto [r1, i1] = p1_gen(1, k, K);
            auto [r2, i2] = p1_gen(2, k, K);
            p1_re[0 * K + k] = r1; p1_im[0 * K + k] = i1;
            p1_re[1 * K + k] = r2; p1_im[1 * K + k] = i2;
        }
        pass1.re = p1_re.ptr;
        pass1.im = p1_im.ptr;
        pass1.K  = K;

        /* Pass 2: 8 twiddle arrays W1..W8 (g=0..7) */
        for (int g = 0; g < 8; g++) {
            p2_re_bufs.push_back(new AlignedBuf(K));
            p2_im_bufs.push_back(new AlignedBuf(K));
            p2_re_ptrs[g] = p2_re_bufs.back()->ptr;
            p2_im_ptrs[g] = p2_im_bufs.back()->ptr;
            for (size_t k = 0; k < K; k++) {
                auto [r, i] = p2_gen(g, k, K);
                p2_re_ptrs[g][k] = r;
                p2_im_ptrs[g][k] = i;
            }
        }

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

/* Trivial twiddle generators (all 1+0j) */
static std::pair<double,double> trivial_p1(size_t, size_t, size_t)
    { return {1.0, 0.0}; }
static std::pair<double,double> trivial_p2(int, size_t, size_t)
    { return {1.0, 0.0}; }

/* Cooley-Tukey twiddle generators for a 32K-point DFT stage */
static std::pair<double,double> ct_p1(size_t m, size_t k, size_t K) {
    double a = -2.0 * M_PI * (double)(m * k) / (32.0 * (double)K);
    return {std::cos(a), std::sin(a)};
}
static std::pair<double,double> ct_p2(int g, size_t k, size_t K) {
    double a = -2.0 * M_PI * (double)(g + 1) * (double)k / (8.0 * (double)K);
    return {std::cos(a), std::sin(a)};
}

/* ========================================================================
 * SCALAR REFERENCE: simulates exact codelet decomposition
 *
 * For a SINGLE sample k, computes the 32-point transform that the codelet
 * performs, including twiddle application, matching the codelet's output
 * permutation (bin-major: output stripe = 8b + d).
 * ======================================================================*/

static void scalar_radix32_forward(
    const double *in_re, const double *in_im, size_t K, size_t k,
    const TwiddleTables &tw,
    double *out_re, double *out_im)
{
    /* Pass 1: DIT-4 per group, bin-major temp output */
    double tmp_re[32], tmp_im[32]; /* temp[bin*8 + group] */

    for (int g = 0; g < 8; g++) {
        /* Load 4 inputs at stride-8 positions */
        double x_re[4], x_im[4];
        for (int m = 0; m < 4; m++) {
            x_re[m] = in_re[(size_t)(g + m * 8) * K + k];
            x_im[m] = in_im[(size_t)(g + m * 8) * K + k];
        }

        /* Apply pass-1 stage twiddles: m=0 untwidled, m=1→W1, m=2→W2, m=3→W1·W2 */
        {
            double W1r = tw.p1_re[0 * tw.pass1.K + k];
            double W1i = tw.p1_im[0 * tw.pass1.K + k];
            double W2r = tw.p1_re[1 * tw.pass1.K + k];
            double W2i = tw.p1_im[1 * tw.pass1.K + k];
            double W3r = W1r * W2r - W1i * W2i;
            double W3i = W1r * W2i + W1i * W2r;

            double t1r = x_re[1]*W1r - x_im[1]*W1i;
            double t1i = x_re[1]*W1i + x_im[1]*W1r;
            x_re[1] = t1r; x_im[1] = t1i;

            double t2r = x_re[2]*W2r - x_im[2]*W2i;
            double t2i = x_re[2]*W2i + x_im[2]*W2r;
            x_re[2] = t2r; x_im[2] = t2i;

            double t3r = x_re[3]*W3r - x_im[3]*W3i;
            double t3i = x_re[3]*W3i + x_im[3]*W3r;
            x_re[3] = t3r; x_im[3] = t3i;
        }

        /* Radix-4 DIT butterfly (forward: −j rotation) */
        double a0r = x_re[0]+x_re[2], a0i = x_im[0]+x_im[2];
        double a1r = x_re[0]-x_re[2], a1i = x_im[0]-x_im[2];
        double a2r = x_re[1]+x_re[3], a2i = x_im[1]+x_im[3];
        double a3r = x_re[1]-x_re[3], a3i = x_im[1]-x_im[3];

        double y_re[4], y_im[4];
        y_re[0] = a0r+a2r; y_im[0] = a0i+a2i;
        y_re[1] = a1r+a3i; y_im[1] = a1i-a3r;
        y_re[2] = a0r-a2r; y_im[2] = a0i-a2i;
        y_re[3] = a1r-a3i; y_im[3] = a1i+a3r;

        for (int b = 0; b < 4; b++) {
            tmp_re[b * 8 + g] = y_re[b];
            tmp_im[b * 8 + g] = y_im[b];
        }
    }

    /* Pass 2: DIF-8 per bin with pass-2 twiddles */
    for (int b = 0; b < 4; b++) {
        double x_re[8], x_im[8];
        for (int g = 0; g < 8; g++) {
            x_re[g] = tmp_re[b * 8 + g];
            x_im[g] = tmp_im[b * 8 + g];
        }

        /* Apply pass-2 twiddles to inputs 1..7 (input 0 untwidled) */
        for (int g = 1; g < 8; g++) {
            double wr = tw.p2_re_ptrs[g - 1][k];
            double wi = tw.p2_im_ptrs[g - 1][k];
            double tr = x_re[g]*wr - x_im[g]*wi;
            double ti = x_re[g]*wi + x_im[g]*wr;
            x_re[g] = tr; x_im[g] = ti;
        }

        /* Radix-8 DIF butterfly (forward) */
        /* Stage 1: half-length butterflies */
        double a[8][2];
        for (int j = 0; j < 4; j++) {
            a[j][0]   = x_re[j] + x_re[j+4];
            a[j][1]   = x_im[j] + x_im[j+4];
            a[j+4][0] = x_re[j] - x_re[j+4];
            a[j+4][1] = x_im[j] - x_im[j+4];
        }

        /* W8 twiddles on odd half */
        /* a5 *= W8^1 = (√2/2)(1−j) → re'=c·re+c·im, im'=−c·re+c·im */
        {
            double c = 0.70710678118654752440;
            double r =  a[5][0]*c + a[5][1]*c;
            double i = -a[5][0]*c + a[5][1]*c;
            a[5][0] = r; a[5][1] = i;
        }
        /* a6 *= W8^2 = −j → re'=im, im'=−re */
        {
            double r =  a[6][1];
            double i = -a[6][0];
            a[6][0] = r; a[6][1] = i;
        }
        /* a7 *= W8^3 = (−√2/2)(1+j) → re'=−c·re+c·im, im'=−c·re−c·im */
        {
            double c = 0.70710678118654752440;
            double r = -a[7][0]*c + a[7][1]*c;
            double i = -a[7][0]*c - a[7][1]*c;
            a[7][0] = r; a[7][1] = i;
        }

        /* Stage 2: radix-4 on even and odd halves */
        double e0r = a[0][0]+a[2][0], e0i = a[0][1]+a[2][1];
        double e1r = a[0][0]-a[2][0], e1i = a[0][1]-a[2][1];
        double e2r = a[1][0]+a[3][0], e2i = a[1][1]+a[3][1];
        double e3r = a[1][0]-a[3][0], e3i = a[1][1]-a[3][1];

        double y_re[8], y_im[8];
        y_re[0] = e0r+e2r; y_im[0] = e0i+e2i;
        y_re[2] = e1r+e3i; y_im[2] = e1i-e3r;
        y_re[4] = e0r-e2r; y_im[4] = e0i-e2i;
        y_re[6] = e1r-e3i; y_im[6] = e1i+e3r;

        double o0r = a[4][0]+a[6][0], o0i = a[4][1]+a[6][1];
        double o1r = a[4][0]-a[6][0], o1i = a[4][1]-a[6][1];
        double o2r = a[5][0]+a[7][0], o2i = a[5][1]+a[7][1];
        double o3r = a[5][0]-a[7][0], o3i = a[5][1]-a[7][1];

        y_re[1] = o0r+o2r; y_im[1] = o0i+o2i;
        y_re[3] = o1r+o3i; y_im[3] = o1i-o3r;
        y_re[5] = o0r-o2r; y_im[5] = o0i-o2i;
        y_re[7] = o1r-o3i; y_im[7] = o1i+o3r;

        /* Output: bin b, DIF output d → stripe b*8+d */
        for (int d = 0; d < 8; d++) {
            out_re[b * 8 + d] = y_re[d];
            out_im[b * 8 + d] = y_im[d];
        }
    }
}

/* ========================================================================
 * PARAMETERISED TEST FIXTURE
 * ======================================================================*/

class Radix32TwiddleTest : public ::testing::TestWithParam<size_t> {
protected:
    size_t K     = 0;
    size_t total = 0;

    std::unique_ptr<AlignedBuf> in_re, in_im;
    std::unique_ptr<AlignedBuf> freq_re, freq_im;
    std::unique_ptr<AlignedBuf> out_re, out_im;
    std::unique_ptr<AlignedBuf> temp_re, temp_im;

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
    }

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

    void run_forward(const TwiddleTables &tw) {
        std::memset(temp_re->ptr, 0, total * sizeof(double));
        std::memset(temp_im->ptr, 0, total * sizeof(double));
        radix32_stage_forward_avx2(
            K, in_re->ptr, in_im->ptr, freq_re->ptr, freq_im->ptr,
            &tw.pass1, &tw.pass2, temp_re->ptr, temp_im->ptr);
    }

    void run_backward(const TwiddleTables &tw) {
        std::memset(temp_re->ptr, 0, total * sizeof(double));
        std::memset(temp_im->ptr, 0, total * sizeof(double));
        radix32_stage_backward_avx2(
            K, freq_re->ptr, freq_im->ptr, out_re->ptr, out_im->ptr,
            &tw.pass1, &tw.pass2, temp_re->ptr, temp_im->ptr);
    }

    void check_roundtrip(double tol = ABS_TOL) {
        for (size_t s = 0; s < N; s++) {
            for (size_t k = 0; k < K; k++) {
                size_t idx = s * K + k;
                EXPECT_NEAR((*out_re)[idx], (*in_re)[idx] * N, tol)
                    << "stripe=" << s << " k=" << k << " (re)";
                EXPECT_NEAR((*out_im)[idx], (*in_im)[idx] * N, tol)
                    << "stripe=" << s << " k=" << k << " (im)";
            }
        }
    }

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

    void check_forward_vs_reference(const TwiddleTables &tw, double tol = ABS_TOL) {
        double ref_re[32], ref_im[32];
        for (size_t k = 0; k < K; k++) {
            scalar_radix32_forward(in_re->ptr, in_im->ptr, K, k, tw,
                                   ref_re, ref_im);
            for (size_t s = 0; s < N; s++) {
                EXPECT_NEAR((*freq_re)[s * K + k], ref_re[s], tol)
                    << "stripe=" << s << " k=" << k << " (re)";
                EXPECT_NEAR((*freq_im)[s * K + k], ref_im[s], tol)
                    << "stripe=" << s << " k=" << k << " (im)";
            }
        }
    }
};

/* K values: all ≤ 256 → BLOCKED8 mode.  K ≥ 8, multiple of 4. */
INSTANTIATE_TEST_SUITE_P(
    KValues,
    Radix32TwiddleTest,
    ::testing::Values(8, 16, 32, 64, 128, 256),
    [](const ::testing::TestParamInfo<size_t> &info) {
        return "K" + std::to_string(info.param);
    }
);

/* ========================================================================
 * A) ROUNDTRIP WITH TRIVIAL TWIDDLES
 *
 * With all-ones twiddles, the codelet computes a pure 32-point DFT per k.
 * backward(forward(x)) = 32·x tests the entire twiddle codepath:
 * BLOCKED8 loader, U=2 pipeline, two-wave stores, prefetch machinery.
 * ======================================================================*/

TEST_P(Radix32TwiddleTest, TrivialTw_RandomRoundtrip) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);
    fill_random(42);
    run_forward(tw);
    run_backward(tw);
    check_roundtrip();
}

TEST_P(Radix32TwiddleTest, TrivialTw_MultiSeed) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);
    for (uint64_t seed = 200; seed < 210; seed++) {
        fill_random(seed);
        run_forward(tw);
        run_backward(tw);
        check_roundtrip();
    }
}

TEST_P(Radix32TwiddleTest, TrivialTw_Dc) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);
    fill_constant(1.0, 0.0);
    run_forward(tw);
    run_backward(tw);
    check_roundtrip();
}

TEST_P(Radix32TwiddleTest, TrivialTw_Zero) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);
    fill_constant(0.0, 0.0);
    run_forward(tw);
    run_backward(tw);
    for (size_t i = 0; i < total; i++) {
        EXPECT_EQ((*out_re)[i], 0.0) << "index=" << i;
        EXPECT_EQ((*out_im)[i], 0.0) << "index=" << i;
    }
}

TEST_P(Radix32TwiddleTest, TrivialTw_Impulse) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);
    fill_constant(0.0, 0.0);
    for (size_t k = 0; k < K; k++)
        (*in_re)[0 * K + k] = 1.0;
    run_forward(tw);
    run_backward(tw);
    check_roundtrip();
}

TEST_P(Radix32TwiddleTest, TrivialTw_PureCosine) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);
    fill_constant(0.0, 0.0);
    const size_t freq = 5;
    for (size_t n = 0; n < N; n++) {
        double val = std::cos(2.0 * M_PI * freq * n / N);
        for (size_t k = 0; k < K; k++)
            (*in_re)[n * K + k] = val;
    }
    run_forward(tw);
    run_backward(tw);
    check_roundtrip();
}

TEST_P(Radix32TwiddleTest, TrivialTw_PurelyImaginary) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);
    fill_constant(0.0, 0.0);
    std::mt19937_64 rng(9999);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < total; i++)
        (*in_im)[i] = dist(rng);
    run_forward(tw);
    run_backward(tw);
    check_roundtrip();
}

TEST_P(Radix32TwiddleTest, TrivialTw_LargeAmplitude) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);
    std::mt19937_64 rng(5555);
    std::uniform_real_distribution<double> dist(-1e6, 1e6);
    for (size_t i = 0; i < total; i++) {
        (*in_re)[i] = dist(rng);
        (*in_im)[i] = dist(rng);
    }
    run_forward(tw);
    run_backward(tw);
    check_roundtrip(1e-6);
}

TEST_P(Radix32TwiddleTest, TrivialTw_ErrorBound) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);
    fill_random(12345);
    run_forward(tw);
    run_backward(tw);
    EXPECT_LT(max_roundtrip_error(), 1e-10) << "K=" << K;
}

/* ========================================================================
 * B) FORWARD vs SCALAR REFERENCE
 *
 * Verifies the codelet's forward output against a scalar simulation
 * that mimics the exact 4×8 decomposition with twiddle application.
 * ======================================================================*/

TEST_P(Radix32TwiddleTest, FwdRef_TrivialTwiddles) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);
    fill_random(7777);
    run_forward(tw);
    check_forward_vs_reference(tw);
}

TEST_P(Radix32TwiddleTest, FwdRef_CTTwiddles) {
    TwiddleTables tw(K, ct_p1, ct_p2);
    fill_random(31415);
    run_forward(tw);
    check_forward_vs_reference(tw);
}

TEST_P(Radix32TwiddleTest, FwdRef_CTMultiSeed) {
    TwiddleTables tw(K, ct_p1, ct_p2);
    for (uint64_t seed = 100; seed < 105; seed++) {
        fill_random(seed);
        run_forward(tw);
        check_forward_vs_reference(tw);
    }
}

TEST_P(Radix32TwiddleTest, FwdRef_CTDc) {
    TwiddleTables tw(K, ct_p1, ct_p2);
    fill_constant(1.0, 0.0);
    run_forward(tw);
    check_forward_vs_reference(tw);
}

TEST_P(Radix32TwiddleTest, FwdRef_CTImpulse) {
    TwiddleTables tw(K, ct_p1, ct_p2);
    fill_constant(0.0, 0.0);
    for (size_t k = 0; k < K; k++)
        (*in_re)[0 * K + k] = 1.0;
    run_forward(tw);
    check_forward_vs_reference(tw);
}

/* ========================================================================
 * C) STRUCTURAL TESTS
 * ======================================================================*/

/* Linearity: F(a·x + b·y) == a·F(x) + b·F(y) with CT twiddles */
TEST_P(Radix32TwiddleTest, Linearity) {
    TwiddleTables tw(K, ct_p1, ct_p2);
    const double a = 2.71828, b = -1.41421;

    AlignedBuf x_re(total), x_im(total);
    AlignedBuf y_re(total), y_im(total);
    AlignedBuf Fx_re(total), Fx_im(total);
    AlignedBuf Fy_re(total), Fy_im(total);
    AlignedBuf t1(total), t2(total);

    std::mt19937_64 rng(31415);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t i = 0; i < total; i++) {
        x_re[i] = dist(rng); x_im[i] = dist(rng);
        y_re[i] = dist(rng); y_im[i] = dist(rng);
    }

    for (size_t i = 0; i < total; i++) {
        (*in_re)[i] = a * x_re[i] + b * y_re[i];
        (*in_im)[i] = a * x_im[i] + b * y_im[i];
    }

    std::memset(t1.ptr, 0, total * sizeof(double));
    std::memset(t2.ptr, 0, total * sizeof(double));
    radix32_stage_forward_avx2(
        K, x_re.ptr, x_im.ptr, Fx_re.ptr, Fx_im.ptr,
        &tw.pass1, &tw.pass2, t1.ptr, t2.ptr);

    std::memset(t1.ptr, 0, total * sizeof(double));
    std::memset(t2.ptr, 0, total * sizeof(double));
    radix32_stage_forward_avx2(
        K, y_re.ptr, y_im.ptr, Fy_re.ptr, Fy_im.ptr,
        &tw.pass1, &tw.pass2, t1.ptr, t2.ptr);

    run_forward(tw);

    for (size_t i = 0; i < total; i++) {
        double exp_re = a * Fx_re[i] + b * Fy_re[i];
        double exp_im = a * Fx_im[i] + b * Fy_im[i];
        EXPECT_NEAR((*freq_re)[i], exp_re, ABS_TOL) << "re, i=" << i;
        EXPECT_NEAR((*freq_im)[i], exp_im, ABS_TOL) << "im, i=" << i;
    }
}

/* Energy: Σ|X[s]|² = 32·Σ|x[s]|² per k (Parseval, trivial twiddles) */
TEST_P(Radix32TwiddleTest, EnergyPreservation) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);
    fill_random(77777);
    run_forward(tw);

    for (size_t k = 0; k < K; k++) {
        double e_in = 0.0, e_out = 0.0;
        for (size_t s = 0; s < N; s++) {
            size_t idx = s * K + k;
            e_in  += (*in_re)[idx]   * (*in_re)[idx]
                   + (*in_im)[idx]   * (*in_im)[idx];
            e_out += (*freq_re)[idx] * (*freq_re)[idx]
                   + (*freq_im)[idx] * (*freq_im)[idx];
        }
        EXPECT_NEAR(e_out, N * e_in, 1e-8) << "k=" << k;
    }
}

/* At k=0, CT twiddles are all 1 → must match trivial twiddles */
TEST_P(Radix32TwiddleTest, TwiddleConsistencyAtK0) {
    TwiddleTables tw_triv(K, trivial_p1, trivial_p2);
    TwiddleTables tw_ct(K, ct_p1, ct_p2);

    fill_random(2024);

    AlignedBuf triv_re(total), triv_im(total), t1(total), t2(total);
    std::memset(t1.ptr, 0, total * sizeof(double));
    std::memset(t2.ptr, 0, total * sizeof(double));
    radix32_stage_forward_avx2(
        K, in_re->ptr, in_im->ptr, triv_re.ptr, triv_im.ptr,
        &tw_triv.pass1, &tw_triv.pass2, t1.ptr, t2.ptr);

    run_forward(tw_ct);

    for (size_t s = 0; s < N; s++) {
        size_t idx = s * K + 0;
        EXPECT_NEAR((*freq_re)[idx], triv_re[idx], ABS_TOL)
            << "stripe=" << s << " k=0 (re)";
        EXPECT_NEAR((*freq_im)[idx], triv_im[idx], ABS_TOL)
            << "stripe=" << s << " k=0 (im)";
    }
}

/* Sample isolation: each k position is independent under trivial twiddles */
TEST_P(Radix32TwiddleTest, TrivialTw_SampleIndependence) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);

    /* Different random data per k */
    std::mt19937_64 rng(0xDEAD);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    for (size_t s = 0; s < N; s++)
        for (size_t k = 0; k < K; k++) {
            (*in_re)[s * K + k] = dist(rng);
            (*in_im)[s * K + k] = dist(rng);
        }

    run_forward(tw);
    run_backward(tw);
    check_roundtrip();
}

/* Alternating-sign pattern roundtrip */
TEST_P(Radix32TwiddleTest, TrivialTw_AlternatingSign) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);
    fill_constant(0.0, 0.0);
    for (size_t n = 0; n < N; n++) {
        double val = (n % 2 == 0) ? 1.0 : -1.0;
        for (size_t k = 0; k < K; k++)
            (*in_re)[n * K + k] = val;
    }
    run_forward(tw);
    run_backward(tw);
    check_roundtrip();
}

/* Single-stripe isolation roundtrip */
TEST_P(Radix32TwiddleTest, TrivialTw_SingleStripe) {
    TwiddleTables tw(K, trivial_p1, trivial_p2);
    for (size_t target = 0; target < N; target += 7) {
        fill_constant(0.0, 0.0);
        std::mt19937_64 rng(target + 1000);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (size_t k = 0; k < K; k++) {
            (*in_re)[target * K + k] = dist(rng);
            (*in_im)[target * K + k] = dist(rng);
        }
        run_forward(tw);
        run_backward(tw);
        check_roundtrip();
    }
}
