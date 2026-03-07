/**
 * test_r32_auto.c — Test the three-tier auto dispatch:
 *   K<=512: packed table
 *   K>512:  pack+walk
 * Verifies correctness against reference DFT for both paths.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "fft_radix32_dispatch.h"

static double *alloc64(size_t n) {
    double *p=NULL; posix_memalign((void**)&p,64,n*8); memset(p,0,n*8); return p;
}

/* Scalar reference: twiddled DFT-32 */
static void ref_tw_dft32(const double *ir, const double *ii,
                         double *or_, double *oi, size_t K, int fwd) {
    const size_t NN = 32*K;
    const double sign = fwd ? -1.0 : 1.0;
    for (size_t k = 0; k < K; k++) {
        double xr[32], xi[32];
        for (int n = 0; n < 32; n++) {
            double dr = ir[n*K+k], di = ii[n*K+k];
            if (n > 0) {
                double a = sign*2.0*M_PI*(double)(n*k)/(double)NN;
                double wr = cos(a), wi = sin(a);
                double tr = dr*wr - di*wi;
                di = dr*wi + di*wr; dr = tr;
            }
            xr[n] = dr; xi[n] = di;
        }
        for (int m = 0; m < 32; m++) {
            double sr = 0, si = 0;
            for (int n = 0; n < 32; n++) {
                double a = sign*2.0*M_PI*(double)(m*n)/32.0;
                double wr = cos(a), wi = sin(a);
                sr += xr[n]*wr - xi[n]*wi;
                si += xr[n]*wi + xi[n]*wr;
            }
            or_[m*K+k] = sr; oi[m*K+k] = si;
        }
    }
}

static double maxerr(const double *ar, const double *ai,
                     const double *br, const double *bi, size_t n) {
    double mx = 0;
    for (size_t i = 0; i < n; i++) {
        double dr = fabs(ar[i]-br[i]), di = fabs(ai[i]-bi[i]);
        if (dr > mx) mx = dr; if (di > mx) mx = di;
    }
    return mx;
}

static void gen_flat_tw(double *re, double *im, size_t K) {
    const size_t NN = 32*K;
    for (int n = 1; n < 32; n++)
        for (size_t k = 0; k < K; k++) {
            double a = -2.0*M_PI*(double)(n*k)/(double)NN;
            re[(n-1)*K+k] = cos(a); im[(n-1)*K+k] = sin(a);
        }
}

__attribute__((target("avx512f")))
static void r32_pack(const double *sr, const double *si,
                     double *dr, double *di, size_t K) {
    for(size_t b=0;b<K/8;b++){
        size_t sk=b*8,dk=b*256;
        for(int n=0;n<32;n++){
            _mm512_storeu_pd(&dr[dk+n*8],_mm512_loadu_pd(&sr[n*K+sk]));
            _mm512_storeu_pd(&di[dk+n*8],_mm512_loadu_pd(&si[n*K+sk]));
        }
    }
}
__attribute__((target("avx512f")))
static void r32_unpack(const double *sr, const double *si,
                       double *dr, double *di, size_t K) {
    for(size_t b=0;b<K/8;b++){
        size_t sk=b*256,dk=b*8;
        for(int n=0;n<32;n++){
            _mm512_storeu_pd(&dr[n*K+dk],_mm512_loadu_pd(&sr[sk+n*8]));
            _mm512_storeu_pd(&di[n*K+dk],_mm512_loadu_pd(&si[sk+n*8]));
        }
    }
}
__attribute__((target("avx512f")))
static void r32_pack_tw(const double *sr, const double *si,
                        double *dr, double *di, size_t K) {
    for(size_t b=0;b<K/8;b++){
        size_t sk=b*8,dk=b*248;
        for(int n=0;n<31;n++){
            _mm512_storeu_pd(&dr[dk+n*8],_mm512_loadu_pd(&sr[n*K+sk]));
            _mm512_storeu_pd(&di[dk+n*8],_mm512_loadu_pd(&si[n*K+sk]));
        }
    }
}

int main(void) {
    int total = 0, passed = 0;
    printf("═══ Radix-32 Three-Tier Auto Dispatch Test ═══\n");
    printf("  Walk threshold = %d\n\n", RADIX32_WALK_THRESHOLD);

    size_t Ks[] = {8, 64, 128, 256, 512, 1024, 2048, 4096};
    int nK = sizeof(Ks)/sizeof(Ks[0]);

    for (int ki = 0; ki < nK; ki++) {
        size_t K = Ks[ki], NN = 32*K;
        size_t T = radix32_packed_optimal_T(K);

        double *ir = alloc64(NN), *ii = alloc64(NN);
        double *ref_r = alloc64(NN), *ref_i = alloc64(NN);
        double *out_r = alloc64(NN), *out_i = alloc64(NN);
        double *pk_ir = alloc64(NN), *pk_ii = alloc64(NN);
        double *pk_or = alloc64(NN), *pk_oi = alloc64(NN);
        double *ftwr = alloc64(31*K), *ftwi = alloc64(31*K);

        srand(42+(unsigned)K);
        for (size_t i = 0; i < NN; i++) {
            ir[i] = (double)rand()/RAND_MAX - 0.5;
            ii[i] = (double)rand()/RAND_MAX - 0.5;
        }

        gen_flat_tw(ftwr, ftwi, K);

        /* Pack data */
        r32_pack(ir, ii, pk_ir, pk_ii, K);

        /* Pack twiddles (only needed if not walking) */
        double *pk_twr = NULL, *pk_twi = NULL;
        if (!radix32_should_walk(K)) {
            pk_twr = alloc64(31*K);
            pk_twi = alloc64(31*K);
            r32_pack_tw(ftwr, ftwi, pk_twr, pk_twi, K);
        }

        /* Walk plan (only needed if walking) */
        radix32_walk_plan_t walk_plan;
        void *wp = NULL;
        if (radix32_should_walk(K)) {
            radix32_walk_plan_init(&walk_plan, K);
            wp = &walk_plan;
        }

        const char *mode = radix32_should_walk(K) ? "pk+walk" : "packed ";

        /* Forward */
        ref_tw_dft32(ir, ii, ref_r, ref_i, K, 1);
        radix32_tw_packed_auto_fwd(pk_ir, pk_ii, pk_or, pk_oi,
                                    pk_twr, pk_twi, wp, K, T);
        r32_unpack(pk_or, pk_oi, out_r, out_i, K);
        double ef = maxerr(ref_r, ref_i, out_r, out_i, NN);

        /* Backward */
        ref_tw_dft32(ir, ii, ref_r, ref_i, K, 0);
        r32_pack(ir, ii, pk_ir, pk_ii, K);  /* re-pack input */
        radix32_tw_packed_auto_bwd(pk_ir, pk_ii, pk_or, pk_oi,
                                    pk_twr, pk_twi, wp, K, T);
        r32_unpack(pk_or, pk_oi, out_r, out_i, K);
        double eb = maxerr(ref_r, ref_i, out_r, out_i, NN);

        int ok = (ef < 1e-9 && eb < 1e-9);
        total++; passed += ok;
        printf("  K=%-5zu %s  fwd=%.1e  bwd=%.1e  %s\n",
               K, mode, ef, eb, ok ? "PASS" : "FAIL");

        free(ir);free(ii);free(ref_r);free(ref_i);free(out_r);free(out_i);
        free(pk_ir);free(pk_ii);free(pk_or);free(pk_oi);
        free(ftwr);free(ftwi);
        if(pk_twr) free(pk_twr);
        if(pk_twi) free(pk_twi);
    }

    printf("\n═══ %d / %d passed ═══\n", passed, total);
    return (passed == total) ? 0 : 1;
}