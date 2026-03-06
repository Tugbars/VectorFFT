#include "vfft_test_utils.h"
#include <fftw3.h>
#include "fft_radix11_genfft.h"

static void naive(int dir, size_t K, const double *ir, const double *ii, double *nr, double *ni)
{
    for (size_t k = 0; k < K; k++)
        for (int m = 0; m < 11; m++)
        {
            double sr = 0, si = 0;
            for (int n = 0; n < 11; n++)
            {
                double a = dir * 2.0 * M_PI * m * n / 11.0;
                sr += ir[n * K + k] * cos(a) - ii[n * K + k] * sin(a);
                si += ir[n * K + k] * sin(a) + ii[n * K + k] * cos(a);
            }
            nr[m * K + k] = sr;
            ni[m * K + k] = si;
        }
}

static int verify(const char *l, size_t K,
                  void (*fn)(const double *, const double *, double *, double *, size_t), int dir)
{
    size_t N = 11 * K;
    double *ir = aa64(N), *ii_ = aa64(N), *gr = aa64(N), *gi = aa64(N), *nr = aa64(N), *ni = aa64(N);
    fill_rand(ir, N, 1000 + (unsigned)K);
    fill_rand(ii_, N, 2000 + (unsigned)K);
    fn(ir, ii_, gr, gi, K);
    naive(dir, K, ir, ii_, nr, ni);
    double err = 0, mag = 0;
    for (size_t i = 0; i < N; i++)
    {
        double e = fmax(fabs(gr[i] - nr[i]), fabs(gi[i] - ni[i]));
        if (e > err)
            err = e;
        double m = fmax(fabs(nr[i]), fabs(ni[i]));
        if (m > mag)
            mag = m;
    }
    double rel = mag > 0 ? err / mag : err;
    int p = rel < 5e-14;
    printf("  %-24s K=%-5zu rel=%.2e %s\n", l, K, rel, p ? "PASS" : "FAIL");
    r32_aligned_free(ir);
    r32_aligned_free(ii_);
    r32_aligned_free(gr);
    r32_aligned_free(gi);
    r32_aligned_free(nr);
    r32_aligned_free(ni);
    return p;
}

static int verify_packed(const char *l, size_t K, size_t T,
                         void (*fn)(const double *, const double *, double *, double *, size_t))
{
    size_t N = 11 * K;
    double *ir = aa64(N), *ii_ = aa64(N), *nr = aa64(N), *ni = aa64(N);
    double *pir = aa64(N), *pii = aa64(N), *por = aa64(N), *poi = aa64(N), *sor = aa64(N), *soi = aa64(N);
    fill_rand(ir, N, 3000 + (unsigned)K);
    fill_rand(ii_, N, 4000 + (unsigned)K);
    r11_pack(ir, ii_, pir, pii, K, T);
    fn(pir, pii, por, poi, K);
    r11_unpack(por, poi, sor, soi, K, T);
    naive(-1, K, ir, ii_, nr, ni);
    double err = 0, mag = 0;
    for (size_t i = 0; i < N; i++)
    {
        double e = fmax(fabs(sor[i] - nr[i]), fabs(soi[i] - ni[i]));
        if (e > err)
            err = e;
        double m = fmax(fabs(nr[i]), fabs(ni[i]));
        if (m > mag)
            mag = m;
    }
    double rel = mag > 0 ? err / mag : err;
    int p = rel < 5e-14;
    printf("  %-24s K=%-5zu rel=%.2e %s\n", l, K, rel, p ? "PASS" : "FAIL");
    r32_aligned_free(ir);
    r32_aligned_free(ii_);
    r32_aligned_free(nr);
    r32_aligned_free(ni);
    r32_aligned_free(pir);
    r32_aligned_free(pii);
    r32_aligned_free(por);
    r32_aligned_free(poi);
    r32_aligned_free(sor);
    r32_aligned_free(soi);
    return p;
}

__attribute__((target("avx512f,avx512dq,fma"))) static void bench(size_t K, int warm, int trials)
{
    size_t N = 11 * K;
    double *ir = aa64(N), *ii_ = aa64(N);
    fill_rand(ir, N, 9000 + (unsigned)K);
    fill_rand(ii_, N, 9500 + (unsigned)K);

    /* FFTW */
    fftw_complex *fin = fftw_alloc_complex(N), *fout = fftw_alloc_complex(N);
    for (size_t k = 0; k < K; k++)
        for (int n = 0; n < 11; n++)
        {
            fin[k * 11 + n][0] = ir[n * K + k];
            fin[k * 11 + n][1] = ii_[n * K + k];
        }
    int na[1] = {11};
    fftw_plan plan = fftw_plan_many_dft(1, na, (int)K,
                                        fin, NULL, 1, 11, fout, NULL, 1, 11, FFTW_FORWARD, FFTW_MEASURE);
    for (int i = 0; i < warm; i++)
        fftw_execute(plan);
    double ns_fw = 1e18;
    for (int t = 0; t < trials; t++)
    {
        double t0 = get_ns();
        fftw_execute(plan);
        double dt = get_ns() - t0;
        if (dt < ns_fw)
            ns_fw = dt;
    }

    /* Packed AVX-512 */
    double ns_z = 1e18;
    {
        size_t T = 8;
        double *p1 = aa64(N), *p2 = aa64(N), *p3 = aa64(N), *p4 = aa64(N);
        r11_pack(ir, ii_, p1, p2, K, T);
        for (int i = 0; i < warm; i++)
            r11_genfft_packed_fwd_avx512(p1, p2, p3, p4, K);
        for (int t = 0; t < trials; t++)
        {
            double t0 = get_ns();
            r11_genfft_packed_fwd_avx512(p1, p2, p3, p4, K);
            double dt = get_ns() - t0;
            if (dt < ns_z)
                ns_z = dt;
        }
        r32_aligned_free(p1);
        r32_aligned_free(p2);
        r32_aligned_free(p3);
        r32_aligned_free(p4);
    }

    /* Packed AVX2 */
    double ns_a = 1e18;
    {
        size_t T = 4;
        double *p1 = aa64(N), *p2 = aa64(N), *p3 = aa64(N), *p4 = aa64(N);
        r11_pack(ir, ii_, p1, p2, K, T);
        for (int i = 0; i < warm; i++)
            r11_genfft_packed_fwd_avx2(p1, p2, p3, p4, K);
        for (int t = 0; t < trials; t++)
        {
            double t0 = get_ns();
            r11_genfft_packed_fwd_avx2(p1, p2, p3, p4, K);
            double dt = get_ns() - t0;
            if (dt < ns_a)
                ns_a = dt;
        }
        r32_aligned_free(p1);
        r32_aligned_free(p2);
        r32_aligned_free(p3);
        r32_aligned_free(p4);
    }

    printf("  K=%-5zu  FFTW=%6.0f  Z=%6.0f(%4.2fx)  A=%6.0f(%4.2fx)\n",
           K, ns_fw, ns_z, ns_fw / ns_z, ns_a, ns_fw / ns_a);

    fftw_destroy_plan(plan);
    fftw_free(fin);
    fftw_free(fout);
    r32_aligned_free(ir);
    r32_aligned_free(ii_);
}

int main(void)
{
    printf("════════════════════════════════════════════════════════════════\n");
    printf("  VectorFFT DFT-11 — FINAL (scalar + AVX2 + AVX-512)\n");
    printf("════════════════════════════════════════════════════════════════\n\n");

    int p = 0, t = 0;
    printf("── Correctness ──\n");
    size_t Ks[] = {1, 7, 8, 64, 1024};
    for (int i = 0; i < 5; i++)
    {
        t++;
        p += verify("scalar fwd", Ks[i], radix11_genfft_fwd_scalar, -1);
    }
    for (int i = 0; i < 5; i++)
    {
        t++;
        p += verify("scalar bwd", Ks[i], radix11_genfft_bwd_scalar, +1);
    }
    size_t Ka[] = {4, 8, 16, 64, 256};
    for (int i = 0; i < 5; i++)
    {
        t++;
        p += verify("avx2 fwd", Ka[i], radix11_genfft_fwd_avx2, -1);
    }
    for (int i = 0; i < 5; i++)
    {
        t++;
        p += verify("avx2 bwd", Ka[i], radix11_genfft_bwd_avx2, +1);
    }
    size_t Kz[] = {8, 16, 64, 256, 1024};
    for (int i = 0; i < 5; i++)
    {
        t++;
        p += verify("avx512 fwd", Kz[i], radix11_genfft_fwd_avx512, -1);
    }
    for (int i = 0; i < 5; i++)
    {
        t++;
        p += verify("avx512 bwd", Kz[i], radix11_genfft_bwd_avx512, +1);
    }
    for (int i = 0; i < 5; i++)
    {
        t++;
        p += verify_packed("packed avx2", Ka[i], 4, r11_genfft_packed_fwd_avx2);
    }
    for (int i = 0; i < 5; i++)
    {
        t++;
        p += verify_packed("packed avx512", Kz[i], 8, r11_genfft_packed_fwd_avx512);
    }
    /* Cross-ISA */
    {
        size_t K = 8, N = 11 * K;
        double *ir = aa64(N), *ii_ = aa64(N), *ar = aa64(N), *ai = aa64(N), *zr = aa64(N), *zi = aa64(N);
        fill_rand(ir, N, 7777);
        fill_rand(ii_, N, 8888);
        radix11_genfft_fwd_avx2(ir, ii_, ar, ai, K);
        radix11_genfft_fwd_avx512(ir, ii_, zr, zi, K);
        double err = 0;
        for (size_t i = 0; i < N; i++)
        {
            double e = fmax(fabs(ar[i] - zr[i]), fabs(ai[i] - zi[i]));
            if (e > err)
                err = e;
        }
        int ok = err < 1e-15;
        t++;
        p += ok;
        printf("  %-24s K=%-5zu err=%.2e %s\n", "cross-ISA avx2↔avx512", K, err, ok ? "PASS" : "FAIL");
        r32_aligned_free(ir);
        r32_aligned_free(ii_);
        r32_aligned_free(ar);
        r32_aligned_free(ai);
        r32_aligned_free(zr);
        r32_aligned_free(zi);
    }

    printf("\n  %d/%d %s\n", p, t, p == t ? "ALL PASSED" : "FAILURES");
    if (p != t)
        return 1;

    printf("\n── Benchmark: packed DFT-only vs FFTW ──\n");
    printf("  Z=AVX-512  A=AVX2  ratio>1 = faster than FFTW\n\n");
    bench(8, 500, 5000);
    bench(16, 500, 5000);
    bench(32, 500, 3000);
    bench(64, 500, 3000);
    bench(128, 200, 2000);
    bench(256, 200, 2000);
    bench(512, 100, 1000);
    bench(1024, 100, 1000);
    bench(2048, 50, 500);
    bench(4096, 20, 300);

    printf("\n════════════════════════════════════════════════════════════════\n");
    fftw_cleanup();
    return 0;
}
