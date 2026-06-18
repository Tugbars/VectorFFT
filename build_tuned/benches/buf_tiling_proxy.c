/* buf_tiling_proxy.c — DECISIVE go/no-go for the buffered-tiling lever.
 *
 * Hypothesis (the high-K real-FFT MKL gap): a radix-R hc2hc butterfly reads its R
 * legs at stride is = (N/R)*K, so at high K the R legs straddle the whole
 * L2-resident plane (prefetcher-hostile, DTLB pressure). FFTW's "buffered" answer:
 * GATHER the R legs into contiguous L1 scratch, run the SAME codelet UNIT-STRIDE
 * (legs K apart, contiguous), SCATTER back. The copy is 2 plane-passes; it wins
 * only if the strided codelet is >~2x slower than the unit-stride one.
 *
 *   A (strided)  : one stage pass, nbf=N/R codelet calls, is = (N/R)*K. No copies.
 *   B (buffered) : per butterfly — gather R*K -> contiguous scratch, codelet
 *                  unit-stride (is=os=K), scatter R*K back. Same codelet, same work.
 *
 * Both touch identical rows {bf + j*(N/R)}; only the access pattern differs. Sweep
 * K to find the crossover (expect A~=B at low K, B winning at K>=64/256). A clear
 * B-win at K=256 = GO for the buffered-tiling executor; a wash/loss = NO-GO.
 *
 * Build: build_tuned/build.py --src benches/buf_tiling_proxy.c --compile
 * Run with C:\mingw152\mingw64\bin on PATH.  Args: [N=256] [R=8]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "core/dp_planner.h"   /* vfft_proto_now_ns */

/* radix-8 forward hc2hc; ABI: (in_re,in_im,out_re,out_im,tw_re,tw_im,is,os,vl).
 * Reads R=8 legs at in_*[j*is+v]; writes R outputs at out_*[j*os+v]; vl lanes. */
extern void radix8_hc2hc_dit_fwd_avx2(const double*, const double*, double*, double*,
                                      const double*, const double*, ptrdiff_t, ptrdiff_t, size_t);

static double *ad(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void**)&p, 64, n * sizeof(double)) != 0) { fprintf(stderr,"oom\n"); exit(1); }
    return p;
}

int main(int argc, char **argv) {
    const int N = (argc > 1) ? atoi(argv[1]) : 256;
    const int R = (argc > 2) ? atoi(argv[2]) : 8;   /* radix8 codelet hardcoded below */
    const int nbf = N / R;
    const size_t Ks[] = { 8, 16, 32, 64, 128, 256, 512 };

    printf("=== buffered-tiling proxy: radix-%d stage, N=%d (nbf=%d) ===\n", R, N, nbf);
    printf("%-6s %12s %12s %12s %8s %8s  %s\n",
           "K", "strided_A", "buffered_B", "compute_C", "B/A", "C/A", "verdict");
    printf("-------+------------+------------+------------+--------+--------+----------\n");

    for (size_t ki = 0; ki < sizeof Ks/sizeof Ks[0]; ki++) {
        const size_t K = Ks[ki];
        const size_t NK = (size_t)N * K;
        const ptrdiff_t is = (ptrdiff_t)((size_t)nbf * K);   /* strided leg gap */
        double *inre = ad(NK), *inim = ad(NK), *outre = ad(NK), *outim = ad(NK);
        double *tre = ad(R), *tim = ad(R);
        double *sre = ad((size_t)R*K), *sim = ad((size_t)R*K),
               *sore = ad((size_t)R*K), *soim = ad((size_t)R*K);
        for (size_t i = 0; i < NK; i++) { inre[i] = (double)(i%7)*0.5-1; inim[i] = (double)(i%5)*0.3-0.7; }
        for (int j = 0; j < R; j++) { tre[j] = 0.92388 - 0.01*j; tim[j] = -0.38268 + 0.01*j; }

        size_t reps = (size_t)(5e7 / (double)(NK + 1)); if (reps < 50) reps = 50; if (reps > 20000) reps = 20000;

        /* ---- A: strided in-place stage pass ---- */
        for (int w = 0; w < 5; w++)
            for (int bf = 0; bf < nbf; bf++)
                radix8_hc2hc_dit_fwd_avx2(&inre[(size_t)bf*K], &inim[(size_t)bf*K],
                                          &outre[(size_t)bf*K], &outim[(size_t)bf*K],
                                          tre, tim, is, is, K);
        double abest = 1e30;
        for (int t = 0; t < 7; t++) {
            double t0 = vfft_proto_now_ns();
            for (size_t r = 0; r < reps; r++)
                for (int bf = 0; bf < nbf; bf++)
                    radix8_hc2hc_dit_fwd_avx2(&inre[(size_t)bf*K], &inim[(size_t)bf*K],
                                              &outre[(size_t)bf*K], &outim[(size_t)bf*K],
                                              tre, tim, is, is, K);
            double e = (vfft_proto_now_ns() - t0) / reps; if (e < abest) abest = e;
        }

        /* ---- B: gather -> unit-stride codelet -> scatter ---- */
        for (int w = 0; w < 5; w++)
            for (int bf = 0; bf < nbf; bf++) {
                for (int j = 0; j < R; j++) {
                    memcpy(&sre[(size_t)j*K], &inre[(size_t)(bf + j*nbf)*K], K*8);
                    memcpy(&sim[(size_t)j*K], &inim[(size_t)(bf + j*nbf)*K], K*8);
                }
                radix8_hc2hc_dit_fwd_avx2(sre, sim, sore, soim, tre, tim, (ptrdiff_t)K, (ptrdiff_t)K, K);
                for (int j = 0; j < R; j++) {
                    memcpy(&outre[(size_t)(bf + j*nbf)*K], &sore[(size_t)j*K], K*8);
                    memcpy(&outim[(size_t)(bf + j*nbf)*K], &soim[(size_t)j*K], K*8);
                }
            }
        double bbest = 1e30;
        for (int t = 0; t < 7; t++) {
            double t0 = vfft_proto_now_ns();
            for (size_t r = 0; r < reps; r++)
                for (int bf = 0; bf < nbf; bf++) {
                    for (int j = 0; j < R; j++) {
                        memcpy(&sre[(size_t)j*K], &inre[(size_t)(bf + j*nbf)*K], K*8);
                        memcpy(&sim[(size_t)j*K], &inim[(size_t)(bf + j*nbf)*K], K*8);
                    }
                    radix8_hc2hc_dit_fwd_avx2(sre, sim, sore, soim, tre, tim, (ptrdiff_t)K, (ptrdiff_t)K, K);
                    for (int j = 0; j < R; j++) {
                        memcpy(&outre[(size_t)(bf + j*nbf)*K], &sore[(size_t)j*K], K*8);
                        memcpy(&outim[(size_t)(bf + j*nbf)*K], &soim[(size_t)j*K], K*8);
                    }
                }
            double e = (vfft_proto_now_ns() - t0) / reps; if (e < bbest) bbest = e;
        }

        /* ---- C: compute-only UNIT-STRIDE (gather once outside timing) ----
         * Isolates the necessary condition: does the codelet run faster on
         * L1-resident contiguous data than strided across the L2 plane? If
         * C ~= A, unit-stride buys nothing and NO amount of copy-amortization
         * can make buffered-tiling win -> clean NO-GO. */
        for (int j = 0; j < R; j++) {
            memcpy(&sre[(size_t)j*K], &inre[(size_t)((size_t)j*nbf)*K], K*8);
            memcpy(&sim[(size_t)j*K], &inim[(size_t)((size_t)j*nbf)*K], K*8);
        }
        for (int w = 0; w < 5; w++)
            for (int bf = 0; bf < nbf; bf++)
                radix8_hc2hc_dit_fwd_avx2(sre, sim, sore, soim, tre, tim, (ptrdiff_t)K, (ptrdiff_t)K, K);
        double cbest = 1e30;
        for (int t = 0; t < 7; t++) {
            double t0 = vfft_proto_now_ns();
            for (size_t r = 0; r < reps; r++)
                for (int bf = 0; bf < nbf; bf++)
                    radix8_hc2hc_dit_fwd_avx2(sre, sim, sore, soim, tre, tim, (ptrdiff_t)K, (ptrdiff_t)K, K);
            double e = (vfft_proto_now_ns() - t0) / reps; if (e < cbest) cbest = e;
        }

        double ba = bbest / abest, ca = cbest / abest;
        printf("%-6zu %12.0f %12.0f %12.0f %8.3f %8.3f  %s\n", K, abest, bbest, cbest, ba, ca,
               ca > 0.92 ? "NO-GO (unit~=strided)" :
               (ba < 0.95 ? "B WINS" : "amortize?"));
        vfft_proto_aligned_free(inre); vfft_proto_aligned_free(inim);
        vfft_proto_aligned_free(outre); vfft_proto_aligned_free(outim);
        vfft_proto_aligned_free(tre); vfft_proto_aligned_free(tim);
        vfft_proto_aligned_free(sre); vfft_proto_aligned_free(sim);
        vfft_proto_aligned_free(sore); vfft_proto_aligned_free(soim);
    }
    return 0;
}
