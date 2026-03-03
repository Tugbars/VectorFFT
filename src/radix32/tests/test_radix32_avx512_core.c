/**
 * @file test_radix32_avx512_core.c
 * @brief AVX-512 butterfly kernel verification
 *
 * Tests AVX-512 radix-4 DIT and radix-8 DIF cores against the AVX2
 * reference. Both use identical math (add/sub/mul/fma on IEEE-754 doubles)
 * so results should be BIT-EXACT when given the same lane values.
 *
 * Strategy: Fill ZMM[0..7] with the same data as two YMM pairs
 * (low half and high half), run both kernels, compare all 8 lanes.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

/* AVX2 reference */
#include "fft_radix32_avx2.h"

/* AVX-512 under test */
#include "fft_radix32_avx512_core.h"

/*==========================================================================
 * Helpers
 *=========================================================================*/

static void fill_rand(double *buf, size_t n, unsigned seed)
{
    for (size_t i = 0; i < n; i++)
    {
        seed = seed * 1103515245 + 12345;
        buf[i] = ((double)(seed >> 16) / 32768.0) - 1.0;
    }
}

/** Compare 8 doubles, return max abs diff */
static double cmp8(const double *a, const double *b)
{
    double mx = 0;
    for (int i = 0; i < 8; i++)
    {
        double e = fabs(a[i] - b[i]);
        if (e > mx) mx = e;
    }
    return mx;
}

/*==========================================================================
 * TEST 1: cmul_v512 vs cmul_v256
 *=========================================================================*/

TARGET_AVX512
static int test_cmul(void)
{
    double ALIGNAS(64) ar[8], ai[8], br[8], bi[8];
    fill_rand(ar, 8, 100);
    fill_rand(ai, 8, 101);
    fill_rand(br, 8, 102);
    fill_rand(bi, 8, 103);

    /* AVX2: process as two 4-wide halves */
    double ALIGNAS(64) avx2_cr[8], avx2_ci[8];
    {
        __m256d cr_lo, ci_lo, cr_hi, ci_hi;
        cmul_v256(
            _mm256_load_pd(ar), _mm256_load_pd(ai),
            _mm256_load_pd(br), _mm256_load_pd(bi),
            &cr_lo, &ci_lo);
        cmul_v256(
            _mm256_load_pd(ar+4), _mm256_load_pd(ai+4),
            _mm256_load_pd(br+4), _mm256_load_pd(bi+4),
            &cr_hi, &ci_hi);
        _mm256_store_pd(avx2_cr, cr_lo);
        _mm256_store_pd(avx2_cr+4, cr_hi);
        _mm256_store_pd(avx2_ci, ci_lo);
        _mm256_store_pd(avx2_ci+4, ci_hi);
    }

    /* AVX-512: single 8-wide */
    double ALIGNAS(64) avx512_cr[8], avx512_ci[8];
    {
        __m512d cr, ci;
        cmul_v512(
            _mm512_load_pd(ar), _mm512_load_pd(ai),
            _mm512_load_pd(br), _mm512_load_pd(bi),
            &cr, &ci);
        _mm512_store_pd(avx512_cr, cr);
        _mm512_store_pd(avx512_ci, ci);
    }

    double err_r = cmp8(avx2_cr, avx512_cr);
    double err_i = cmp8(avx2_ci, avx512_ci);
    double err = err_r > err_i ? err_r : err_i;

    int pass = err == 0.0;
    printf("  cmul       err=%.1e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * TEST 2: csquare_v512 vs csquare_v256
 *=========================================================================*/

TARGET_AVX512
static int test_csquare(void)
{
    double ALIGNAS(64) ar[8], ai[8];
    fill_rand(ar, 8, 200);
    fill_rand(ai, 8, 201);

    double ALIGNAS(64) avx2_cr[8], avx2_ci[8];
    {
        __m256d cr_lo, ci_lo, cr_hi, ci_hi;
        csquare_v256(
            _mm256_load_pd(ar), _mm256_load_pd(ai),
            &cr_lo, &ci_lo);
        csquare_v256(
            _mm256_load_pd(ar+4), _mm256_load_pd(ai+4),
            &cr_hi, &ci_hi);
        _mm256_store_pd(avx2_cr, cr_lo);
        _mm256_store_pd(avx2_cr+4, cr_hi);
        _mm256_store_pd(avx2_ci, ci_lo);
        _mm256_store_pd(avx2_ci+4, ci_hi);
    }

    double ALIGNAS(64) avx512_cr[8], avx512_ci[8];
    {
        __m512d cr, ci;
        csquare_v512(
            _mm512_load_pd(ar), _mm512_load_pd(ai),
            &cr, &ci);
        _mm512_store_pd(avx512_cr, cr);
        _mm512_store_pd(avx512_ci, ci);
    }

    double err = fmax(cmp8(avx2_cr, avx512_cr), cmp8(avx2_ci, avx512_ci));
    int pass = err == 0.0;
    printf("  csquare    err=%.1e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * TEST 3: radix4_dit_core forward — AVX-512 vs AVX2
 *=========================================================================*/

TARGET_AVX512
static int test_radix4_fwd(void)
{
    double ALIGNAS(64) x[8][8]; /* x[input][lane] */
    for (int i = 0; i < 8; i++)
        fill_rand(x[i], 8, 300 + (unsigned)i);

    /* AVX2: two halves */
    double ALIGNAS(64) a2_y[8][8]; /* y[output][lane] */
    {
        __m256d yr[4], yi[4];
        radix4_dit_core_forward_avx2(
            _mm256_load_pd(x[0]), _mm256_load_pd(x[1]),
            _mm256_load_pd(x[2]), _mm256_load_pd(x[3]),
            _mm256_load_pd(x[4]), _mm256_load_pd(x[5]),
            _mm256_load_pd(x[6]), _mm256_load_pd(x[7]),
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3]);
        for (int i = 0; i < 4; i++) {
            _mm256_store_pd(a2_y[2*i], yr[i]);
            _mm256_store_pd(a2_y[2*i+1], yi[i]);
        }

        __m256d yr2[4], yi2[4];
        radix4_dit_core_forward_avx2(
            _mm256_load_pd(x[0]+4), _mm256_load_pd(x[1]+4),
            _mm256_load_pd(x[2]+4), _mm256_load_pd(x[3]+4),
            _mm256_load_pd(x[4]+4), _mm256_load_pd(x[5]+4),
            _mm256_load_pd(x[6]+4), _mm256_load_pd(x[7]+4),
            &yr2[0], &yi2[0], &yr2[1], &yi2[1],
            &yr2[2], &yi2[2], &yr2[3], &yi2[3]);
        for (int i = 0; i < 4; i++) {
            _mm256_store_pd(a2_y[2*i]+4, yr2[i]);
            _mm256_store_pd(a2_y[2*i+1]+4, yi2[i]);
        }
    }

    /* AVX-512: single pass */
    double ALIGNAS(64) a5_y[8][8];
    {
        __m512d yr[4], yi[4];
        radix4_dit_core_forward_avx512(
            _mm512_load_pd(x[0]), _mm512_load_pd(x[1]),
            _mm512_load_pd(x[2]), _mm512_load_pd(x[3]),
            _mm512_load_pd(x[4]), _mm512_load_pd(x[5]),
            _mm512_load_pd(x[6]), _mm512_load_pd(x[7]),
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3]);
        for (int i = 0; i < 4; i++) {
            _mm512_store_pd(a5_y[2*i], yr[i]);
            _mm512_store_pd(a5_y[2*i+1], yi[i]);
        }
    }

    double err = 0;
    for (int i = 0; i < 8; i++)
        err = fmax(err, cmp8(a2_y[i], a5_y[i]));

    int pass = err == 0.0;
    printf("  r4_dit_fwd err=%.1e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * TEST 4: radix4_dit_core backward — AVX-512 vs AVX2
 *=========================================================================*/

TARGET_AVX512
static int test_radix4_bwd(void)
{
    double ALIGNAS(64) x[8][8];
    for (int i = 0; i < 8; i++)
        fill_rand(x[i], 8, 400 + (unsigned)i);

    double ALIGNAS(64) a2_y[8][8];
    {
        __m256d yr[4], yi[4];
        radix4_dit_core_backward_avx2(
            _mm256_load_pd(x[0]), _mm256_load_pd(x[1]),
            _mm256_load_pd(x[2]), _mm256_load_pd(x[3]),
            _mm256_load_pd(x[4]), _mm256_load_pd(x[5]),
            _mm256_load_pd(x[6]), _mm256_load_pd(x[7]),
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3]);
        for (int i = 0; i < 4; i++) {
            _mm256_store_pd(a2_y[2*i], yr[i]);
            _mm256_store_pd(a2_y[2*i+1], yi[i]);
        }

        __m256d yr2[4], yi2[4];
        radix4_dit_core_backward_avx2(
            _mm256_load_pd(x[0]+4), _mm256_load_pd(x[1]+4),
            _mm256_load_pd(x[2]+4), _mm256_load_pd(x[3]+4),
            _mm256_load_pd(x[4]+4), _mm256_load_pd(x[5]+4),
            _mm256_load_pd(x[6]+4), _mm256_load_pd(x[7]+4),
            &yr2[0], &yi2[0], &yr2[1], &yi2[1],
            &yr2[2], &yi2[2], &yr2[3], &yi2[3]);
        for (int i = 0; i < 4; i++) {
            _mm256_store_pd(a2_y[2*i]+4, yr2[i]);
            _mm256_store_pd(a2_y[2*i+1]+4, yi2[i]);
        }
    }

    double ALIGNAS(64) a5_y[8][8];
    {
        __m512d yr[4], yi[4];
        radix4_dit_core_backward_avx512(
            _mm512_load_pd(x[0]), _mm512_load_pd(x[1]),
            _mm512_load_pd(x[2]), _mm512_load_pd(x[3]),
            _mm512_load_pd(x[4]), _mm512_load_pd(x[5]),
            _mm512_load_pd(x[6]), _mm512_load_pd(x[7]),
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3]);
        for (int i = 0; i < 4; i++) {
            _mm512_store_pd(a5_y[2*i], yr[i]);
            _mm512_store_pd(a5_y[2*i+1], yi[i]);
        }
    }

    double err = 0;
    for (int i = 0; i < 8; i++)
        err = fmax(err, cmp8(a2_y[i], a5_y[i]));

    int pass = err == 0.0;
    printf("  r4_dit_bwd err=%.1e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * TEST 5: radix8_dif_core forward — AVX-512 vs AVX2
 *=========================================================================*/

TARGET_AVX512
static int test_radix8_fwd(void)
{
    double ALIGNAS(64) x[16][8]; /* 8 complex inputs × 8 lanes = x[0..15] */
    for (int i = 0; i < 16; i++)
        fill_rand(x[i], 8, 500 + (unsigned)i);

    /* AVX2: low 4 lanes */
    double ALIGNAS(64) a2_y[16][8];
    {
        __m256d yr[8], yi[8];
        radix8_dif_core_forward_avx2(
            _mm256_load_pd(x[0]),  _mm256_load_pd(x[1]),
            _mm256_load_pd(x[2]),  _mm256_load_pd(x[3]),
            _mm256_load_pd(x[4]),  _mm256_load_pd(x[5]),
            _mm256_load_pd(x[6]),  _mm256_load_pd(x[7]),
            _mm256_load_pd(x[8]),  _mm256_load_pd(x[9]),
            _mm256_load_pd(x[10]), _mm256_load_pd(x[11]),
            _mm256_load_pd(x[12]), _mm256_load_pd(x[13]),
            _mm256_load_pd(x[14]), _mm256_load_pd(x[15]),
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm256_store_pd(a2_y[2*i], yr[i]);
            _mm256_store_pd(a2_y[2*i+1], yi[i]);
        }

        /* AVX2: high 4 lanes */
        __m256d yr2[8], yi2[8];
        radix8_dif_core_forward_avx2(
            _mm256_load_pd(x[0]+4),  _mm256_load_pd(x[1]+4),
            _mm256_load_pd(x[2]+4),  _mm256_load_pd(x[3]+4),
            _mm256_load_pd(x[4]+4),  _mm256_load_pd(x[5]+4),
            _mm256_load_pd(x[6]+4),  _mm256_load_pd(x[7]+4),
            _mm256_load_pd(x[8]+4),  _mm256_load_pd(x[9]+4),
            _mm256_load_pd(x[10]+4), _mm256_load_pd(x[11]+4),
            _mm256_load_pd(x[12]+4), _mm256_load_pd(x[13]+4),
            _mm256_load_pd(x[14]+4), _mm256_load_pd(x[15]+4),
            &yr2[0], &yi2[0], &yr2[1], &yi2[1],
            &yr2[2], &yi2[2], &yr2[3], &yi2[3],
            &yr2[4], &yi2[4], &yr2[5], &yi2[5],
            &yr2[6], &yi2[6], &yr2[7], &yi2[7]);
        for (int i = 0; i < 8; i++) {
            _mm256_store_pd(a2_y[2*i]+4, yr2[i]);
            _mm256_store_pd(a2_y[2*i+1]+4, yi2[i]);
        }
    }

    /* AVX-512: single 8-wide pass */
    double ALIGNAS(64) a5_y[16][8];
    {
        __m512d yr[8], yi[8];
        radix8_dif_core_forward_avx512(
            _mm512_load_pd(x[0]),  _mm512_load_pd(x[1]),
            _mm512_load_pd(x[2]),  _mm512_load_pd(x[3]),
            _mm512_load_pd(x[4]),  _mm512_load_pd(x[5]),
            _mm512_load_pd(x[6]),  _mm512_load_pd(x[7]),
            _mm512_load_pd(x[8]),  _mm512_load_pd(x[9]),
            _mm512_load_pd(x[10]), _mm512_load_pd(x[11]),
            _mm512_load_pd(x[12]), _mm512_load_pd(x[13]),
            _mm512_load_pd(x[14]), _mm512_load_pd(x[15]),
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm512_store_pd(a5_y[2*i], yr[i]);
            _mm512_store_pd(a5_y[2*i+1], yi[i]);
        }
    }

    double err = 0;
    for (int i = 0; i < 16; i++)
        err = fmax(err, cmp8(a2_y[i], a5_y[i]));

    int pass = err == 0.0;
    printf("  r8_dif_fwd err=%.1e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * TEST 6: radix8_dif_core backward — AVX-512 vs AVX2
 *=========================================================================*/

TARGET_AVX512
static int test_radix8_bwd(void)
{
    double ALIGNAS(64) x[16][8];
    for (int i = 0; i < 16; i++)
        fill_rand(x[i], 8, 600 + (unsigned)i);

    double ALIGNAS(64) a2_y[16][8];
    {
        __m256d yr[8], yi[8];
        radix8_dif_core_backward_avx2(
            _mm256_load_pd(x[0]),  _mm256_load_pd(x[1]),
            _mm256_load_pd(x[2]),  _mm256_load_pd(x[3]),
            _mm256_load_pd(x[4]),  _mm256_load_pd(x[5]),
            _mm256_load_pd(x[6]),  _mm256_load_pd(x[7]),
            _mm256_load_pd(x[8]),  _mm256_load_pd(x[9]),
            _mm256_load_pd(x[10]), _mm256_load_pd(x[11]),
            _mm256_load_pd(x[12]), _mm256_load_pd(x[13]),
            _mm256_load_pd(x[14]), _mm256_load_pd(x[15]),
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm256_store_pd(a2_y[2*i], yr[i]);
            _mm256_store_pd(a2_y[2*i+1], yi[i]);
        }

        __m256d yr2[8], yi2[8];
        radix8_dif_core_backward_avx2(
            _mm256_load_pd(x[0]+4),  _mm256_load_pd(x[1]+4),
            _mm256_load_pd(x[2]+4),  _mm256_load_pd(x[3]+4),
            _mm256_load_pd(x[4]+4),  _mm256_load_pd(x[5]+4),
            _mm256_load_pd(x[6]+4),  _mm256_load_pd(x[7]+4),
            _mm256_load_pd(x[8]+4),  _mm256_load_pd(x[9]+4),
            _mm256_load_pd(x[10]+4), _mm256_load_pd(x[11]+4),
            _mm256_load_pd(x[12]+4), _mm256_load_pd(x[13]+4),
            _mm256_load_pd(x[14]+4), _mm256_load_pd(x[15]+4),
            &yr2[0], &yi2[0], &yr2[1], &yi2[1],
            &yr2[2], &yi2[2], &yr2[3], &yi2[3],
            &yr2[4], &yi2[4], &yr2[5], &yi2[5],
            &yr2[6], &yi2[6], &yr2[7], &yi2[7]);
        for (int i = 0; i < 8; i++) {
            _mm256_store_pd(a2_y[2*i]+4, yr2[i]);
            _mm256_store_pd(a2_y[2*i+1]+4, yi2[i]);
        }
    }

    double ALIGNAS(64) a5_y[16][8];
    {
        __m512d yr[8], yi[8];
        radix8_dif_core_backward_avx512(
            _mm512_load_pd(x[0]),  _mm512_load_pd(x[1]),
            _mm512_load_pd(x[2]),  _mm512_load_pd(x[3]),
            _mm512_load_pd(x[4]),  _mm512_load_pd(x[5]),
            _mm512_load_pd(x[6]),  _mm512_load_pd(x[7]),
            _mm512_load_pd(x[8]),  _mm512_load_pd(x[9]),
            _mm512_load_pd(x[10]), _mm512_load_pd(x[11]),
            _mm512_load_pd(x[12]), _mm512_load_pd(x[13]),
            _mm512_load_pd(x[14]), _mm512_load_pd(x[15]),
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm512_store_pd(a5_y[2*i], yr[i]);
            _mm512_store_pd(a5_y[2*i+1], yi[i]);
        }
    }

    double err = 0;
    for (int i = 0; i < 16; i++)
        err = fmax(err, cmp8(a2_y[i], a5_y[i]));

    int pass = err == 0.0;
    printf("  r8_dif_bwd err=%.1e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * TEST 7: radix8 fwd→bwd roundtrip (self-consistency)
 *
 * DIF_fwd(x) then DIF_bwd(y) should give back 8·x (radix-8 scaling).
 * This tests the AVX-512 kernels in isolation without AVX2 reference.
 *=========================================================================*/

TARGET_AVX512
static int test_radix8_roundtrip(void)
{
    double ALIGNAS(64) x[16][8]; /* 8 complex inputs, 8 lanes each */
    for (int i = 0; i < 16; i++)
        fill_rand(x[i], 8, 700 + (unsigned)i);

    /* Forward */
    double ALIGNAS(64) fwd[16][8];
    {
        __m512d yr[8], yi[8];
        radix8_dif_core_forward_avx512(
            _mm512_load_pd(x[0]),  _mm512_load_pd(x[1]),
            _mm512_load_pd(x[2]),  _mm512_load_pd(x[3]),
            _mm512_load_pd(x[4]),  _mm512_load_pd(x[5]),
            _mm512_load_pd(x[6]),  _mm512_load_pd(x[7]),
            _mm512_load_pd(x[8]),  _mm512_load_pd(x[9]),
            _mm512_load_pd(x[10]), _mm512_load_pd(x[11]),
            _mm512_load_pd(x[12]), _mm512_load_pd(x[13]),
            _mm512_load_pd(x[14]), _mm512_load_pd(x[15]),
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm512_store_pd(fwd[2*i], yr[i]);
            _mm512_store_pd(fwd[2*i+1], yi[i]);
        }
    }

    /* Backward */
    double ALIGNAS(64) bwd[16][8];
    {
        __m512d yr[8], yi[8];
        radix8_dif_core_backward_avx512(
            _mm512_load_pd(fwd[0]),  _mm512_load_pd(fwd[1]),
            _mm512_load_pd(fwd[2]),  _mm512_load_pd(fwd[3]),
            _mm512_load_pd(fwd[4]),  _mm512_load_pd(fwd[5]),
            _mm512_load_pd(fwd[6]),  _mm512_load_pd(fwd[7]),
            _mm512_load_pd(fwd[8]),  _mm512_load_pd(fwd[9]),
            _mm512_load_pd(fwd[10]), _mm512_load_pd(fwd[11]),
            _mm512_load_pd(fwd[12]), _mm512_load_pd(fwd[13]),
            _mm512_load_pd(fwd[14]), _mm512_load_pd(fwd[15]),
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm512_store_pd(bwd[2*i], yr[i]);
            _mm512_store_pd(bwd[2*i+1], yi[i]);
        }
    }

    /* Compare: bwd should equal 8·x */
    double err = 0;
    for (int i = 0; i < 16; i++)
        for (int j = 0; j < 8; j++)
        {
            double e = fabs(bwd[i][j] / 8.0 - x[i][j]);
            if (e > err) err = e;
        }

    int pass = err < 1e-14;
    printf("  r8_roundtrip  err=%.2e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * TEST 8: cmul_mem_v512 vs cmul_v512
 *=========================================================================*/

TARGET_AVX512
static int test_cmul_mem(void)
{
    double ALIGNAS(64) xr[8], xi[8], wr[8], wi[8];
    fill_rand(xr, 8, 800);
    fill_rand(xi, 8, 801);
    fill_rand(wr, 8, 802);
    fill_rand(wi, 8, 803);

    double ALIGNAS(64) reg_cr[8], reg_ci[8];
    {
        __m512d cr, ci;
        cmul_v512(
            _mm512_load_pd(xr), _mm512_load_pd(xi),
            _mm512_load_pd(wr), _mm512_load_pd(wi),
            &cr, &ci);
        _mm512_store_pd(reg_cr, cr);
        _mm512_store_pd(reg_ci, ci);
    }

    double ALIGNAS(64) mem_cr[8], mem_ci[8];
    {
        __m512d cr, ci;
        cmul_mem_v512(
            _mm512_load_pd(xr), _mm512_load_pd(xi),
            wr, wi, &cr, &ci);
        _mm512_store_pd(mem_cr, cr);
        _mm512_store_pd(mem_ci, ci);
    }

    double err = fmax(cmp8(reg_cr, mem_cr), cmp8(reg_ci, mem_ci));
    int pass = err == 0.0;
    printf("  cmul_mem   err=%.1e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * TEST 9: derive_w5_to_w8_512 vs AVX2 derive_w5_to_w8
 *=========================================================================*/

TARGET_AVX512
static int test_derive_w5w8(void)
{
    double ALIGNAS(64) w[8][8]; /* W1r, W1i, W2r, W2i, W3r, W3i, W4r, W4i */
    for (int i = 0; i < 8; i++)
        fill_rand(w[i], 8, 900 + (unsigned)i);

    /* AVX2: two halves */
    double ALIGNAS(64) a2[8][8]; /* W5r,W5i,W6r,W6i,W7r,W7i,W8r,W8i */
    {
        __m256d W5r_lo, W5i_lo, W6r_lo, W6i_lo, W7r_lo, W7i_lo, W8r_lo, W8i_lo;
        derive_w5_to_w8(
            _mm256_load_pd(w[0]), _mm256_load_pd(w[1]),
            _mm256_load_pd(w[2]), _mm256_load_pd(w[3]),
            _mm256_load_pd(w[4]), _mm256_load_pd(w[5]),
            _mm256_load_pd(w[6]), _mm256_load_pd(w[7]),
            &W5r_lo, &W5i_lo, &W6r_lo, &W6i_lo,
            &W7r_lo, &W7i_lo, &W8r_lo, &W8i_lo);
        _mm256_store_pd(a2[0], W5r_lo); _mm256_store_pd(a2[1], W5i_lo);
        _mm256_store_pd(a2[2], W6r_lo); _mm256_store_pd(a2[3], W6i_lo);
        _mm256_store_pd(a2[4], W7r_lo); _mm256_store_pd(a2[5], W7i_lo);
        _mm256_store_pd(a2[6], W8r_lo); _mm256_store_pd(a2[7], W8i_lo);

        __m256d W5r_hi, W5i_hi, W6r_hi, W6i_hi, W7r_hi, W7i_hi, W8r_hi, W8i_hi;
        derive_w5_to_w8(
            _mm256_load_pd(w[0]+4), _mm256_load_pd(w[1]+4),
            _mm256_load_pd(w[2]+4), _mm256_load_pd(w[3]+4),
            _mm256_load_pd(w[4]+4), _mm256_load_pd(w[5]+4),
            _mm256_load_pd(w[6]+4), _mm256_load_pd(w[7]+4),
            &W5r_hi, &W5i_hi, &W6r_hi, &W6i_hi,
            &W7r_hi, &W7i_hi, &W8r_hi, &W8i_hi);
        _mm256_store_pd(a2[0]+4, W5r_hi); _mm256_store_pd(a2[1]+4, W5i_hi);
        _mm256_store_pd(a2[2]+4, W6r_hi); _mm256_store_pd(a2[3]+4, W6i_hi);
        _mm256_store_pd(a2[4]+4, W7r_hi); _mm256_store_pd(a2[5]+4, W7i_hi);
        _mm256_store_pd(a2[6]+4, W8r_hi); _mm256_store_pd(a2[7]+4, W8i_hi);
    }

    /* AVX-512 */
    double ALIGNAS(64) a5[8][8];
    {
        __m512d W5r, W5i, W6r, W6i, W7r, W7i, W8r, W8i;
        derive_w5_to_w8_512(
            _mm512_load_pd(w[0]), _mm512_load_pd(w[1]),
            _mm512_load_pd(w[2]), _mm512_load_pd(w[3]),
            _mm512_load_pd(w[4]), _mm512_load_pd(w[5]),
            _mm512_load_pd(w[6]), _mm512_load_pd(w[7]),
            &W5r, &W5i, &W6r, &W6i, &W7r, &W7i, &W8r, &W8i);
        _mm512_store_pd(a5[0], W5r); _mm512_store_pd(a5[1], W5i);
        _mm512_store_pd(a5[2], W6r); _mm512_store_pd(a5[3], W6i);
        _mm512_store_pd(a5[4], W7r); _mm512_store_pd(a5[5], W7i);
        _mm512_store_pd(a5[6], W8r); _mm512_store_pd(a5[7], W8i);
    }

    double err = 0;
    for (int i = 0; i < 8; i++)
        err = fmax(err, cmp8(a2[i], a5[i]));

    int pass = err == 0.0;
    printf("  derive5-8  err=%.1e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * TEST 10: Fused twiddle+butterfly FORWARD — AVX-512 vs AVX2
 *
 * Runs dif8_twiddle_and_butterfly_forward on identical data with both
 * AVX2 (two 4-wide passes) and AVX-512 (one 8-wide pass). Bit-exact.
 *=========================================================================*/

TARGET_AVX512
static int test_fused_fwd(void)
{
    /* 8 complex inputs × 8 lanes */
    double ALIGNAS(64) x[16][8];
    for (int i = 0; i < 16; i++)
        fill_rand(x[i], 8, 1000 + (unsigned)i);

    /* 7 complex twiddles × 8 lanes */
    double ALIGNAS(64) tw_r[7][8], tw_i[7][8];
    for (int j = 0; j < 7; j++) {
        fill_rand(tw_r[j], 8, 1100 + (unsigned)j);
        fill_rand(tw_i[j], 8, 1200 + (unsigned)j);
    }

    /* AVX2: low half */
    double ALIGNAS(64) a2_y[16][8];
    {
        __m256d twr[7], twi[7];
        for (int j = 0; j < 7; j++) {
            twr[j] = _mm256_load_pd(tw_r[j]);
            twi[j] = _mm256_load_pd(tw_i[j]);
        }
        __m256d yr[8], yi[8];
        dif8_twiddle_and_butterfly_forward(
            _mm256_load_pd(x[0]),  _mm256_load_pd(x[1]),
            _mm256_load_pd(x[2]),  _mm256_load_pd(x[3]),
            _mm256_load_pd(x[4]),  _mm256_load_pd(x[5]),
            _mm256_load_pd(x[6]),  _mm256_load_pd(x[7]),
            _mm256_load_pd(x[8]),  _mm256_load_pd(x[9]),
            _mm256_load_pd(x[10]), _mm256_load_pd(x[11]),
            _mm256_load_pd(x[12]), _mm256_load_pd(x[13]),
            _mm256_load_pd(x[14]), _mm256_load_pd(x[15]),
            twr, twi,
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm256_store_pd(a2_y[2*i], yr[i]);
            _mm256_store_pd(a2_y[2*i+1], yi[i]);
        }

        /* AVX2: high half */
        for (int j = 0; j < 7; j++) {
            twr[j] = _mm256_load_pd(tw_r[j]+4);
            twi[j] = _mm256_load_pd(tw_i[j]+4);
        }
        __m256d yr2[8], yi2[8];
        dif8_twiddle_and_butterfly_forward(
            _mm256_load_pd(x[0]+4),  _mm256_load_pd(x[1]+4),
            _mm256_load_pd(x[2]+4),  _mm256_load_pd(x[3]+4),
            _mm256_load_pd(x[4]+4),  _mm256_load_pd(x[5]+4),
            _mm256_load_pd(x[6]+4),  _mm256_load_pd(x[7]+4),
            _mm256_load_pd(x[8]+4),  _mm256_load_pd(x[9]+4),
            _mm256_load_pd(x[10]+4), _mm256_load_pd(x[11]+4),
            _mm256_load_pd(x[12]+4), _mm256_load_pd(x[13]+4),
            _mm256_load_pd(x[14]+4), _mm256_load_pd(x[15]+4),
            twr, twi,
            &yr2[0], &yi2[0], &yr2[1], &yi2[1],
            &yr2[2], &yi2[2], &yr2[3], &yi2[3],
            &yr2[4], &yi2[4], &yr2[5], &yi2[5],
            &yr2[6], &yi2[6], &yr2[7], &yi2[7]);
        for (int i = 0; i < 8; i++) {
            _mm256_store_pd(a2_y[2*i]+4, yr2[i]);
            _mm256_store_pd(a2_y[2*i+1]+4, yi2[i]);
        }
    }

    /* AVX-512 */
    double ALIGNAS(64) a5_y[16][8];
    {
        __m512d twr[7], twi[7];
        for (int j = 0; j < 7; j++) {
            twr[j] = _mm512_load_pd(tw_r[j]);
            twi[j] = _mm512_load_pd(tw_i[j]);
        }
        __m512d yr[8], yi[8];
        dif8_twiddle_and_butterfly_forward_avx512(
            _mm512_load_pd(x[0]),  _mm512_load_pd(x[1]),
            _mm512_load_pd(x[2]),  _mm512_load_pd(x[3]),
            _mm512_load_pd(x[4]),  _mm512_load_pd(x[5]),
            _mm512_load_pd(x[6]),  _mm512_load_pd(x[7]),
            _mm512_load_pd(x[8]),  _mm512_load_pd(x[9]),
            _mm512_load_pd(x[10]), _mm512_load_pd(x[11]),
            _mm512_load_pd(x[12]), _mm512_load_pd(x[13]),
            _mm512_load_pd(x[14]), _mm512_load_pd(x[15]),
            twr, twi,
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm512_store_pd(a5_y[2*i], yr[i]);
            _mm512_store_pd(a5_y[2*i+1], yi[i]);
        }
    }

    double err = 0;
    for (int i = 0; i < 16; i++)
        err = fmax(err, cmp8(a2_y[i], a5_y[i]));

    int pass = err == 0.0;
    printf("  fused_fwd  err=%.1e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * TEST 11: Fused twiddle+butterfly BACKWARD — AVX-512 vs AVX2
 *=========================================================================*/

TARGET_AVX512
static int test_fused_bwd(void)
{
    double ALIGNAS(64) x[16][8];
    for (int i = 0; i < 16; i++)
        fill_rand(x[i], 8, 1300 + (unsigned)i);

    double ALIGNAS(64) tw_r[7][8], tw_i[7][8];
    for (int j = 0; j < 7; j++) {
        fill_rand(tw_r[j], 8, 1400 + (unsigned)j);
        fill_rand(tw_i[j], 8, 1500 + (unsigned)j);
    }

    /* AVX2 */
    double ALIGNAS(64) a2_y[16][8];
    {
        __m256d twr[7], twi[7];
        for (int j = 0; j < 7; j++) {
            twr[j] = _mm256_load_pd(tw_r[j]);
            twi[j] = _mm256_load_pd(tw_i[j]);
        }
        __m256d yr[8], yi[8];
        dif8_twiddle_and_butterfly_backward(
            _mm256_load_pd(x[0]),  _mm256_load_pd(x[1]),
            _mm256_load_pd(x[2]),  _mm256_load_pd(x[3]),
            _mm256_load_pd(x[4]),  _mm256_load_pd(x[5]),
            _mm256_load_pd(x[6]),  _mm256_load_pd(x[7]),
            _mm256_load_pd(x[8]),  _mm256_load_pd(x[9]),
            _mm256_load_pd(x[10]), _mm256_load_pd(x[11]),
            _mm256_load_pd(x[12]), _mm256_load_pd(x[13]),
            _mm256_load_pd(x[14]), _mm256_load_pd(x[15]),
            twr, twi,
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm256_store_pd(a2_y[2*i], yr[i]);
            _mm256_store_pd(a2_y[2*i+1], yi[i]);
        }

        for (int j = 0; j < 7; j++) {
            twr[j] = _mm256_load_pd(tw_r[j]+4);
            twi[j] = _mm256_load_pd(tw_i[j]+4);
        }
        __m256d yr2[8], yi2[8];
        dif8_twiddle_and_butterfly_backward(
            _mm256_load_pd(x[0]+4),  _mm256_load_pd(x[1]+4),
            _mm256_load_pd(x[2]+4),  _mm256_load_pd(x[3]+4),
            _mm256_load_pd(x[4]+4),  _mm256_load_pd(x[5]+4),
            _mm256_load_pd(x[6]+4),  _mm256_load_pd(x[7]+4),
            _mm256_load_pd(x[8]+4),  _mm256_load_pd(x[9]+4),
            _mm256_load_pd(x[10]+4), _mm256_load_pd(x[11]+4),
            _mm256_load_pd(x[12]+4), _mm256_load_pd(x[13]+4),
            _mm256_load_pd(x[14]+4), _mm256_load_pd(x[15]+4),
            twr, twi,
            &yr2[0], &yi2[0], &yr2[1], &yi2[1],
            &yr2[2], &yi2[2], &yr2[3], &yi2[3],
            &yr2[4], &yi2[4], &yr2[5], &yi2[5],
            &yr2[6], &yi2[6], &yr2[7], &yi2[7]);
        for (int i = 0; i < 8; i++) {
            _mm256_store_pd(a2_y[2*i]+4, yr2[i]);
            _mm256_store_pd(a2_y[2*i+1]+4, yi2[i]);
        }
    }

    /* AVX-512 */
    double ALIGNAS(64) a5_y[16][8];
    {
        __m512d twr[7], twi[7];
        for (int j = 0; j < 7; j++) {
            twr[j] = _mm512_load_pd(tw_r[j]);
            twi[j] = _mm512_load_pd(tw_i[j]);
        }
        __m512d yr[8], yi[8];
        dif8_twiddle_and_butterfly_backward_avx512(
            _mm512_load_pd(x[0]),  _mm512_load_pd(x[1]),
            _mm512_load_pd(x[2]),  _mm512_load_pd(x[3]),
            _mm512_load_pd(x[4]),  _mm512_load_pd(x[5]),
            _mm512_load_pd(x[6]),  _mm512_load_pd(x[7]),
            _mm512_load_pd(x[8]),  _mm512_load_pd(x[9]),
            _mm512_load_pd(x[10]), _mm512_load_pd(x[11]),
            _mm512_load_pd(x[12]), _mm512_load_pd(x[13]),
            _mm512_load_pd(x[14]), _mm512_load_pd(x[15]),
            twr, twi,
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm512_store_pd(a5_y[2*i], yr[i]);
            _mm512_store_pd(a5_y[2*i+1], yi[i]);
        }
    }

    double err = 0;
    for (int i = 0; i < 16; i++)
        err = fmax(err, cmp8(a2_y[i], a5_y[i]));

    int pass = err == 0.0;
    printf("  fused_bwd  err=%.1e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * TEST 12: Fused = manual twiddle-apply + bare core
 *
 * Verifies that dif8_twiddle_and_butterfly_forward_avx512 produces the
 * same result as: (1) manually cmul each input with its twiddle, then
 * (2) calling the bare radix8_dif_core_forward_avx512.
 *
 * Uses non-trivial twiddles (unit roots) to exercise the full path.
 *=========================================================================*/

TARGET_AVX512
static int test_fused_vs_manual_fwd(void)
{
    double ALIGNAS(64) x[16][8];
    for (int i = 0; i < 16; i++)
        fill_rand(x[i], 8, 1600 + (unsigned)i);

    /* Non-trivial twiddles: Wj = exp(-2πi·j/8) broadcast to 8 lanes */
    double ALIGNAS(64) tw_r[7][8], tw_i[7][8];
    for (int j = 0; j < 7; j++) {
        double angle = -2.0 * M_PI * (double)(j + 1) / 8.0;
        double cr = cos(angle), ci = sin(angle);
        for (int l = 0; l < 8; l++) {
            tw_r[j][l] = cr;
            tw_i[j][l] = ci;
        }
    }

    /* Method A: fused function */
    double ALIGNAS(64) fused_y[16][8];
    {
        __m512d twr[7], twi[7];
        for (int j = 0; j < 7; j++) {
            twr[j] = _mm512_load_pd(tw_r[j]);
            twi[j] = _mm512_load_pd(tw_i[j]);
        }
        __m512d yr[8], yi[8];
        dif8_twiddle_and_butterfly_forward_avx512(
            _mm512_load_pd(x[0]),  _mm512_load_pd(x[1]),
            _mm512_load_pd(x[2]),  _mm512_load_pd(x[3]),
            _mm512_load_pd(x[4]),  _mm512_load_pd(x[5]),
            _mm512_load_pd(x[6]),  _mm512_load_pd(x[7]),
            _mm512_load_pd(x[8]),  _mm512_load_pd(x[9]),
            _mm512_load_pd(x[10]), _mm512_load_pd(x[11]),
            _mm512_load_pd(x[12]), _mm512_load_pd(x[13]),
            _mm512_load_pd(x[14]), _mm512_load_pd(x[15]),
            twr, twi,
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm512_store_pd(fused_y[2*i], yr[i]);
            _mm512_store_pd(fused_y[2*i+1], yi[i]);
        }
    }

    /* Method B: manual twiddle-apply then bare core */
    double ALIGNAS(64) manual_y[16][8];
    {
        /* Twiddle x1..x7; x0 passes through */
        __m512d tx[16];
        tx[0] = _mm512_load_pd(x[0]);  /* x0r — untwidded */
        tx[1] = _mm512_load_pd(x[1]);  /* x0i */
        for (int j = 0; j < 7; j++) {
            __m512d tr, ti;
            cmul_v512(
                _mm512_load_pd(x[2*(j+1)]), _mm512_load_pd(x[2*(j+1)+1]),
                _mm512_load_pd(tw_r[j]),    _mm512_load_pd(tw_i[j]),
                &tr, &ti);
            tx[2*(j+1)]   = tr;
            tx[2*(j+1)+1] = ti;
        }

        /* Bare core */
        __m512d yr[8], yi[8];
        radix8_dif_core_forward_avx512(
            tx[0],  tx[1],  tx[2],  tx[3],
            tx[4],  tx[5],  tx[6],  tx[7],
            tx[8],  tx[9],  tx[10], tx[11],
            tx[12], tx[13], tx[14], tx[15],
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm512_store_pd(manual_y[2*i], yr[i]);
            _mm512_store_pd(manual_y[2*i+1], yi[i]);
        }
    }

    double err = 0;
    for (int i = 0; i < 16; i++)
        err = fmax(err, cmp8(fused_y[i], manual_y[i]));

    int pass = err == 0.0;
    printf("  fused=manual_fwd  err=%.1e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * TEST 13: Fused = manual twiddle-apply + bare core (BACKWARD)
 *=========================================================================*/

TARGET_AVX512
static int test_fused_vs_manual_bwd(void)
{
    double ALIGNAS(64) x[16][8];
    for (int i = 0; i < 16; i++)
        fill_rand(x[i], 8, 1700 + (unsigned)i);

    double ALIGNAS(64) tw_r[7][8], tw_i[7][8];
    for (int j = 0; j < 7; j++) {
        double angle = -2.0 * M_PI * (double)(j + 1) / 8.0;
        double cr = cos(angle), ci = sin(angle);
        for (int l = 0; l < 8; l++) {
            tw_r[j][l] = cr;
            tw_i[j][l] = ci;
        }
    }

    /* Fused */
    double ALIGNAS(64) fused_y[16][8];
    {
        __m512d twr[7], twi[7];
        for (int j = 0; j < 7; j++) {
            twr[j] = _mm512_load_pd(tw_r[j]);
            twi[j] = _mm512_load_pd(tw_i[j]);
        }
        __m512d yr[8], yi[8];
        dif8_twiddle_and_butterfly_backward_avx512(
            _mm512_load_pd(x[0]),  _mm512_load_pd(x[1]),
            _mm512_load_pd(x[2]),  _mm512_load_pd(x[3]),
            _mm512_load_pd(x[4]),  _mm512_load_pd(x[5]),
            _mm512_load_pd(x[6]),  _mm512_load_pd(x[7]),
            _mm512_load_pd(x[8]),  _mm512_load_pd(x[9]),
            _mm512_load_pd(x[10]), _mm512_load_pd(x[11]),
            _mm512_load_pd(x[12]), _mm512_load_pd(x[13]),
            _mm512_load_pd(x[14]), _mm512_load_pd(x[15]),
            twr, twi,
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm512_store_pd(fused_y[2*i], yr[i]);
            _mm512_store_pd(fused_y[2*i+1], yi[i]);
        }
    }

    /* Manual */
    double ALIGNAS(64) manual_y[16][8];
    {
        __m512d tx[16];
        tx[0] = _mm512_load_pd(x[0]);
        tx[1] = _mm512_load_pd(x[1]);
        for (int j = 0; j < 7; j++) {
            __m512d tr, ti;
            cmul_v512(
                _mm512_load_pd(x[2*(j+1)]), _mm512_load_pd(x[2*(j+1)+1]),
                _mm512_load_pd(tw_r[j]),    _mm512_load_pd(tw_i[j]),
                &tr, &ti);
            tx[2*(j+1)]   = tr;
            tx[2*(j+1)+1] = ti;
        }

        __m512d yr[8], yi[8];
        radix8_dif_core_backward_avx512(
            tx[0],  tx[1],  tx[2],  tx[3],
            tx[4],  tx[5],  tx[6],  tx[7],
            tx[8],  tx[9],  tx[10], tx[11],
            tx[12], tx[13], tx[14], tx[15],
            &yr[0], &yi[0], &yr[1], &yi[1],
            &yr[2], &yi[2], &yr[3], &yi[3],
            &yr[4], &yi[4], &yr[5], &yi[5],
            &yr[6], &yi[6], &yr[7], &yi[7]);
        for (int i = 0; i < 8; i++) {
            _mm512_store_pd(manual_y[2*i], yr[i]);
            _mm512_store_pd(manual_y[2*i+1], yi[i]);
        }
    }

    double err = 0;
    for (int i = 0; i < 16; i++)
        err = fmax(err, cmp8(fused_y[i], manual_y[i]));

    int pass = err == 0.0;
    printf("  fused=manual_bwd  err=%.1e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * TEST 13: Load/store macro round-trip
 *
 * Write known data to strided layout, load with DIF8_LOAD_INPUTS_512,
 * store with DIF8_STORE_TWO_WAVE_512, verify bit-exact.
 *=========================================================================*/

TARGET_AVX512
static int test_load_store(void)
{
    const size_t K = 16; /* 2 ZMM iterations worth */
    double ALIGNAS(64) in_re[8 * K], in_im[8 * K];
    double ALIGNAS(64) out_re[8 * K], out_im[8 * K];

    fill_rand(in_re, 8 * K, 1700);
    fill_rand(in_im, 8 * K, 1701);
    memset(out_re, 0, sizeof(out_re));
    memset(out_im, 0, sizeof(out_im));

    /* Load at k=0, store back at k=0 */
    {
        __m512d x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
        __m512d x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i;

        DIF8_LOAD_INPUTS_512(in_re, in_im, K, 0,
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i);

        DIF8_STORE_TWO_WAVE_512(_mm512_store_pd, out_re, out_im, K, 0,
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i);
    }

    /* Load at k=8, store back at k=8 */
    {
        __m512d x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
        __m512d x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i;

        DIF8_LOAD_INPUTS_512(in_re, in_im, K, 8,
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i);

        DIF8_STORE_TWO_WAVE_512(_mm512_store_pd, out_re, out_im, K, 8,
            x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i,
            x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i);
    }

    double err = 0;
    for (size_t i = 0; i < 8 * K; i++) {
        double e = fabs(out_re[i] - in_re[i]);
        if (e > err) err = e;
        e = fabs(out_im[i] - in_im[i]);
        if (e > err) err = e;
    }

    int pass = err == 0.0;
    printf("  load_store err=%.1e  %s\n", err, pass ? "PASS" : "FAIL");
    return pass;
}

/*==========================================================================
 * MAIN
 *=========================================================================*/

int main(void)
{
    printf("╔══════════════════════════════════════════════════════════╗\n");
    printf("║  AVX-512 Butterfly Kernels — Verification Suite        ║\n");
    printf("╚══════════════════════════════════════════════════════════╝\n\n");

    int total = 0, passed = 0;

    printf("── Layer 0: Core arithmetic (AVX-512 vs AVX2, bit-exact) ──\n");
    { int p = test_cmul();            total++; passed += p; }
    { int p = test_csquare();         total++; passed += p; }
    { int p = test_cmul_mem();        total++; passed += p; }
    { int p = test_radix4_fwd();      total++; passed += p; }
    { int p = test_radix4_bwd();      total++; passed += p; }
    { int p = test_radix8_fwd();      total++; passed += p; }
    { int p = test_radix8_bwd();      total++; passed += p; }
    { int p = test_radix8_roundtrip(); total++; passed += p; }

    printf("\n── Layer 1: Twiddle + butterfly (bit-exact + decomposition) ──\n");
    { int p = test_derive_w5w8();       total++; passed += p; }
    { int p = test_fused_fwd();         total++; passed += p; }
    { int p = test_fused_bwd();         total++; passed += p; }
    { int p = test_fused_vs_manual_fwd(); total++; passed += p; }
    { int p = test_fused_vs_manual_bwd(); total++; passed += p; }
    { int p = test_load_store();        total++; passed += p; }

    printf("\n══════════════════════════════════════════\n");
    printf("  %d/%d passed  %s\n", passed, total,
           (passed == total) ? "✓ ALL PASSED" : "✗ FAILURES");
    printf("══════════════════════════════════════════\n");

    return (passed == total) ? 0 : 1;
}
