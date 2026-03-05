/*
 * bench_radix32_tw_unified.c
 *
 * Comprehensive test + benchmark for unified twiddled DFT-32 dispatch.
 * Tests both packed (production) and strided (fallback) paths.
 */
#include "radix32_test_utils.h"
#include <fftw3.h>


/* ═══════════════════════════════════════════════════════════════ */
#include "fft_radix32_avx512_tw_unified.h"
/* ═══════════════════════════════════════════════════════════════ */

/* Naive reference */
static void naive_tw_dft32(size_t K, size_t k,
    const double *ir, const double *ii,
    const double *twr, const double *twi,
    double *or_, double *oi) {
    double xr[32], xi[32];
    xr[0] = ir[k]; xi[0] = ii[k];
    for (int n = 1; n < 32; n++) {
        double wr = twr[(n-1)*K + k], wi = twi[(n-1)*K + k];
        xr[n] = ir[n*K + k]*wr - ii[n*K + k]*wi;
        xi[n] = ir[n*K + k]*wi + ii[n*K + k]*wr;
    }
    for (int m = 0; m < 32; m++) {
        double sr = 0, si = 0;
        for (int n = 0; n < 32; n++) {
            double a = -2.0 * M_PI * m * n / 32.0;
            sr += xr[n]*cos(a) - xi[n]*sin(a);
            si += xr[n]*sin(a) + xi[n]*cos(a);
        }
        or_[m*K + k] = sr; oi[m*K + k] = si;
    }
}

/* ── Correctness: packed path vs naive ── */
static int test_packed(size_t K) {
    size_t N = 32 * K;
    size_t T = r32_packed_optimal_T(K);
    double *ir = aa64(N), *ii_ = aa64(N), *nr = aa64(N), *ni = aa64(N);

    /* Build twiddles */
    double *ftwr = aa64(31*K), *ftwi = aa64(31*K);
    double *ptwr = aa64(31*K), *ptwi = aa64(31*K);
    r32_build_flat_twiddles(K, -1, ftwr, ftwi);
    r32_build_packed_twiddles(K, T, ftwr, ftwi, ptwr, ptwi);

    /* Pack input */
    double *pir = aa64(N), *pii = aa64(N);
    double *por = aa64(N), *poi = aa64(N);
    fill_rand(ir, N, 1000 + (unsigned)K);
    fill_rand(ii_, N, 2000 + (unsigned)K);
    r32_pack_input(ir, ii_, pir, pii, K, T);

    /* Dispatch */
    radix32_tw_packed_dispatch_fwd(K, pir, pii, por, poi, ptwr, ptwi);

    /* Unpack output */
    double *sor = aa64(N), *soi = aa64(N);
    r32_unpack_output(por, poi, sor, soi, K, T);

    /* Naive reference */
    for (size_t k = 0; k < K; k++)
        naive_tw_dft32(K, k, ir, ii_, ftwr, ftwi, nr, ni);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(sor[i]-nr[i]), fabs(soi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr, N), max_abs(ni, N));
    double rel = mag > 0 ? err / mag : err;
    int pass = rel < 5e-13;
    printf("  packed fwd K=%-5zu T=%-3zu rel=%.2e  %s\n", K, T, rel, pass?"PASS":"FAIL");

    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(nr);r32_aligned_free(ni);
    r32_aligned_free(ftwr);r32_aligned_free(ftwi);r32_aligned_free(ptwr);r32_aligned_free(ptwi);
    r32_aligned_free(pir);r32_aligned_free(pii);r32_aligned_free(por);r32_aligned_free(poi);r32_aligned_free(sor);r32_aligned_free(soi);
    return pass;
}

/* ── Correctness: strided path vs naive ── */
static int test_strided(size_t K) {
    size_t N = 32 * K;
    double *ir = aa64(N), *ii_ = aa64(N), *or_ = aa64(N), *oi = aa64(N);
    double *nr = aa64(N), *ni = aa64(N);
    double *ftwr = aa64(31*K), *ftwi = aa64(31*K);
    double *btwr = aa64(5*K), *btwi = aa64(5*K);

    fill_rand(ir, N, 3000 + (unsigned)K);
    fill_rand(ii_, N, 4000 + (unsigned)K);
    r32_build_flat_twiddles(K, -1, ftwr, ftwi);
    r32_build_ladder_twiddles(K, -1, btwr, btwi);

    radix32_tw_strided_dispatch_fwd(K, ir, ii_, or_, oi,
        ftwr, ftwi, btwr, btwi);

    for (size_t k = 0; k < K; k++)
        naive_tw_dft32(K, k, ir, ii_, ftwr, ftwi, nr, ni);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(or_[i]-nr[i]), fabs(oi[i]-ni[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(nr, N), max_abs(ni, N));
    double rel = mag > 0 ? err / mag : err;
    int pass = rel < 5e-13;
    const char *path = K < R32_LADDER_THRESH ? "flat" :
                       K < R32_NT_THRESH     ? "ladder" : "lad+NT";
    printf("  strided fwd K=%-5zu [%-6s] rel=%.2e  %s\n", K, path, rel, pass?"PASS":"FAIL");

    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(or_);r32_aligned_free(oi);r32_aligned_free(nr);r32_aligned_free(ni);
    r32_aligned_free(ftwr);r32_aligned_free(ftwi);r32_aligned_free(btwr);r32_aligned_free(btwi);
    return pass;
}

/* ── Cross: packed vs strided (same flat twiddle derivation) ── */
static int test_cross(size_t K) {
    size_t N = 32 * K, T = r32_packed_optimal_T(K);
    double *ir = aa64(N), *ii_ = aa64(N);
    double *ftwr = aa64(31*K), *ftwi = aa64(31*K);
    double *btwr = aa64(5*K), *btwi = aa64(5*K);
    double *ptwr = aa64(31*K), *ptwi = aa64(31*K);

    fill_rand(ir, N, 7000 + (unsigned)K);
    fill_rand(ii_, N, 8000 + (unsigned)K);
    r32_build_flat_twiddles(K, -1, ftwr, ftwi);
    r32_build_ladder_twiddles(K, -1, btwr, btwi);
    r32_build_packed_twiddles(K, T, ftwr, ftwi, ptwr, ptwi);

    /* Strided path */
    double *sr = aa64(N), *si = aa64(N);
    radix32_tw_strided_dispatch_fwd(K, ir, ii_, sr, si,
        ftwr, ftwi, btwr, btwi);

    /* Packed path */
    double *pir = aa64(N), *pii = aa64(N);
    double *por = aa64(N), *poi = aa64(N);
    r32_pack_input(ir, ii_, pir, pii, K, T);
    radix32_tw_packed_dispatch_fwd(K, pir, pii, por, poi, ptwr, ptwi);
    double *pr = aa64(N), *pi_ = aa64(N);
    r32_unpack_output(por, poi, pr, pi_, K, T);

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(sr[i]-pr[i]), fabs(si[i]-pi_[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(sr, N), max_abs(si, N));
    double rel = mag > 0 ? err / mag : err;
    /* K<128 uses same flat kernel → bit-exact. K≥128 uses ladder → close but not exact. */
    int pass = (K < R32_LADDER_THRESH) ? (err == 0.0) : (rel < 1e-12);
    printf("  packed<->strided K=%-5zu T=%-3zu rel=%.2e  %s%s\n",
           K, T, rel, pass?"PASS":"FAIL",
           (K < R32_LADDER_THRESH && err == 0.0) ? " (bit-exact)" : "");

    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(sr);r32_aligned_free(si);
    r32_aligned_free(ftwr);r32_aligned_free(ftwi);r32_aligned_free(btwr);r32_aligned_free(btwi);r32_aligned_free(ptwr);r32_aligned_free(ptwi);
    r32_aligned_free(pir);r32_aligned_free(pii);r32_aligned_free(por);r32_aligned_free(poi);r32_aligned_free(pr);r32_aligned_free(pi_);
    return pass;
}

/* ── Roundtrip: fwd → bwd → scale → original ── */
static int test_roundtrip(size_t K) {
    size_t N = 32 * K, T = r32_packed_optimal_T(K);
    double *ir = aa64(N), *ii_ = aa64(N);
    double *ftwr_f = aa64(31*K), *ftwi_f = aa64(31*K);
    double *ftwr_b = aa64(31*K), *ftwi_b = aa64(31*K);
    double *ptwr_f = aa64(31*K), *ptwi_f = aa64(31*K);
    double *ptwr_b = aa64(31*K), *ptwi_b = aa64(31*K);

    fill_rand(ir, N, 5000 + (unsigned)K);
    fill_rand(ii_, N, 6000 + (unsigned)K);
    r32_build_flat_twiddles(K, -1, ftwr_f, ftwi_f);
    r32_build_flat_twiddles(K, +1, ftwr_b, ftwi_b);
    r32_build_packed_twiddles(K, T, ftwr_f, ftwi_f, ptwr_f, ptwi_f);
    r32_build_packed_twiddles(K, T, ftwr_b, ftwi_b, ptwr_b, ptwi_b);

    double *pir = aa64(N), *pii = aa64(N);
    double *pmid = aa64(N), *pmidi = aa64(N);
    double *por = aa64(N), *poi = aa64(N);
    r32_pack_input(ir, ii_, pir, pii, K, T);

    /* Forward */
    radix32_tw_packed_dispatch_fwd(K, pir, pii, pmid, pmidi, ptwr_f, ptwi_f);
    /* Backward */
    radix32_tw_packed_dispatch_bwd(K, pmid, pmidi, por, poi, ptwr_b, ptwi_b);

    /* Unpack + scale by 1/32 */
    double *rr = aa64(N), *ri = aa64(N);
    r32_unpack_output(por, poi, rr, ri, K, T);
    for (size_t i = 0; i < N; i++) { rr[i] /= 32.0; ri[i] /= 32.0; }

    double err = 0;
    for (size_t i = 0; i < N; i++) {
        double e = fmax(fabs(ir[i]-rr[i]), fabs(ii_[i]-ri[i]));
        if (e > err) err = e;
    }
    double mag = fmax(max_abs(ir, N), max_abs(ii_, N));
    double rel = mag > 0 ? err / mag : err;
    int pass = rel < 5e-14;
    printf("  roundtrip K=%-5zu T=%-3zu rel=%.2e  %s\n", K, T, rel, pass?"PASS":"FAIL");

    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(rr);r32_aligned_free(ri);
    r32_aligned_free(ftwr_f);r32_aligned_free(ftwi_f);r32_aligned_free(ftwr_b);r32_aligned_free(ftwi_b);
    r32_aligned_free(ptwr_f);r32_aligned_free(ptwi_f);r32_aligned_free(ptwr_b);r32_aligned_free(ptwi_b);
    r32_aligned_free(pir);r32_aligned_free(pii);r32_aligned_free(pmid);r32_aligned_free(pmidi);r32_aligned_free(por);r32_aligned_free(poi);
    return pass;
}

/* ── Benchmark ── */
__attribute__((target("avx512f,avx512dq,fma")))
static void run_bench(size_t K, int warm, int trials) {
    size_t N = 32 * K, T = r32_packed_optimal_T(K);
    double *ir = aa64(N), *ii_ = aa64(N), *or_ = aa64(N), *oi = aa64(N);
    double *ftwr = aa64(31*K), *ftwi = aa64(31*K);
    double *btwr = aa64(5*K), *btwi = aa64(5*K);
    double *ptwr = aa64(31*K), *ptwi = aa64(31*K);
    double *pir = aa64(N), *pii = aa64(N), *por = aa64(N), *poi = aa64(N);

    fill_rand(ir, N, 9000 + (unsigned)K);
    fill_rand(ii_, N, 9500 + (unsigned)K);
    r32_build_flat_twiddles(K, -1, ftwr, ftwi);
    r32_build_ladder_twiddles(K, -1, btwr, btwi);
    r32_build_packed_twiddles(K, T, ftwr, ftwi, ptwr, ptwi);
    r32_pack_input(ir, ii_, pir, pii, K, T);

    /* FFTW */
    fftw_complex *fin = fftw_alloc_complex(N), *fout = fftw_alloc_complex(N);
    for (size_t k = 0; k < K; k++)
        for (int n = 0; n < 32; n++) {
            fin[k*32+n][0] = ir[n*K+k]; fin[k*32+n][1] = ii_[n*K+k];
        }
    int na[1] = {32};
    fftw_plan plan = fftw_plan_many_dft(1, na, (int)K,
        fin, NULL, 1, 32, fout, NULL, 1, 32, FFTW_FORWARD, FFTW_MEASURE);
    for (int i = 0; i < warm; i++) fftw_execute(plan);
    double bfw = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns(); fftw_execute(plan);
        double dt = get_ns() - t0; if (dt < bfw) bfw = dt;
    }

    /* Packed dispatch (production) */
    for (int i = 0; i < warm; i++)
        radix32_tw_packed_dispatch_fwd(K, pir, pii, por, poi, ptwr, ptwi);
    double ns_pk = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        radix32_tw_packed_dispatch_fwd(K, pir, pii, por, poi, ptwr, ptwi);
        double dt = get_ns() - t0; if (dt < ns_pk) ns_pk = dt;
    }

    /* Strided dispatch (fallback) */
    for (int i = 0; i < warm; i++)
        radix32_tw_strided_dispatch_fwd(K, ir, ii_, or_, oi, ftwr, ftwi, btwr, btwi);
    double ns_st = 1e18;
    for (int t = 0; t < trials; t++) {
        double t0 = get_ns();
        radix32_tw_strided_dispatch_fwd(K, ir, ii_, or_, oi, ftwr, ftwi, btwr, btwi);
        double dt = get_ns() - t0; if (dt < ns_st) ns_st = dt;
    }

    const char *st_path = K < R32_LADDER_THRESH ? "flat" :
                          K < R32_NT_THRESH     ? "ladder" : "lad+NT";

    printf("  K=%-5zu T=%-3zu  FFTW=%7.0f  packed=%7.0f(%5.2fx)  strided=%7.0f(%5.2fx) [%s]\n",
           K, T, bfw, ns_pk, bfw/ns_pk, ns_st, bfw/ns_st, st_path);

    fftw_destroy_plan(plan); fftw_free(fin); fftw_free(fout);
    r32_aligned_free(ir);r32_aligned_free(ii_);r32_aligned_free(or_);r32_aligned_free(oi);
    r32_aligned_free(ftwr);r32_aligned_free(ftwi);r32_aligned_free(btwr);r32_aligned_free(btwi);
    r32_aligned_free(ptwr);r32_aligned_free(ptwi);
    r32_aligned_free(pir);r32_aligned_free(pii);r32_aligned_free(por);r32_aligned_free(poi);
}

int main(void) {
    printf("====================================================================\n");
    printf("  DFT-32 AVX-512 TWIDDLED: UNIFIED DISPATCH TEST + BENCHMARK\n");
    printf("====================================================================\n");
    printf("  Thresholds: LADDER=%d  NT=%d  PACKED_T=%d\n",
           R32_LADDER_THRESH, R32_NT_THRESH, R32_PACKED_BLOCK_T);
    printf("  All codelets do fused twiddle + DFT-32\n");
    printf("  FFTW baseline = batched DFT-32 (no twiddles)\n\n");

    int p = 0, t = 0;

    printf("-- Packed path vs naive --\n");
    { size_t Ks[] = {8,16,32,64,128,256,512,1024,2048};
      for (int i = 0; i < 9; i++) { t++; p += test_packed(Ks[i]); } }

    printf("\n-- Strided path vs naive --\n");
    { size_t Ks[] = {8,16,32,64,128,256,512,1024,2048,4096};
      for (int i = 0; i < 10; i++) { t++; p += test_strided(Ks[i]); } }

    printf("\n-- Cross: packed <-> strided --\n");
    { size_t Ks[] = {8,16,32,64,128,256,512};
      for (int i = 0; i < 7; i++) { t++; p += test_cross(Ks[i]); } }

    printf("\n======================================\n");
    printf("  %d/%d passed  %s\n", p, t, p == t ? "ALL PASSED" : "FAILURES");
    printf("======================================\n");
    if (p != t) return 1;

    printf("\n-- BENCHMARK (ns, forward) --\n");
    printf("  packed = DFT-only, data pre-packed (production)\n");
    printf("  strided = data in stride-K layout (fallback)\n\n");

    run_bench(8,     500, 5000);
    run_bench(16,    500, 5000);
    run_bench(32,    500, 3000);
    run_bench(64,    500, 3000);
    run_bench(128,   200, 2000);
    run_bench(256,   200, 2000);
    run_bench(512,   100, 1000);
    run_bench(1024,  100, 1000);
    run_bench(2048,   50, 500);
    run_bench(4096,   50, 500);
    run_bench(8192,   20, 200);

    fftw_cleanup();
    return 0;
}
