/* buf_tiling_amortized.c — does AMORTIZED, L1-tiled buffering win at K=256?
 *
 * The first proxy showed: unit-stride compute is ~4x faster ONLY when the tile
 * fits cache, and a per-stage copy never amortizes. This tests the reframed
 * hypothesis: lane-tile K into Kb slabs, GATHER the [N x Kb] slab to contiguous
 * scratch ONCE (row stride K -> Kb, so the codelet's leg stride nbf*K -> nbf*Kb
 * shrinks), run S cascade passes unit-strided on the slab, SCATTER ONCE.
 *
 *   A (strided)  : S passes over the full plane, leg stride nbf*K. No copy.
 *   B (buffered) : per Kb-slab — gather [N x Kb], S passes (leg stride nbf*Kb),
 *                  scatter. One gather+scatter amortized over S*nbf codelet calls.
 *
 * Sweep S: N=256 has only ~1-2 real combine stages, so this shows whether
 * amortization is even AVAILABLE at our sizes, and the crossover S where B<A.
 * Sweep Kb: 256 (= no tiling, copy is pure overhead) vs 128/64 (smaller stride +
 * working set). Build: build_tuned/build.py --src benches/buf_tiling_amortized.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>
#include "core/dp_planner.h"   /* vfft_proto_now_ns, posix_memalign, aligned_free */

extern void radix8_hc2hc_dit_fwd_avx2(const double*, const double*, double*, double*,
                                      const double*, const double*, ptrdiff_t, ptrdiff_t, size_t);

static double *ad(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void**)&p, 64, n * sizeof(double)) != 0) { fprintf(stderr,"oom\n"); exit(1); }
    return p;
}

int main(int argc, char **argv) {
    const int N = (argc > 1) ? atoi(argv[1]) : 256;
    const int R = 8;
    const size_t K = (argc > 2) ? (size_t)atoi(argv[2]) : 256;
    const int nbf = N / R;
    const size_t NK = (size_t)N * K;

    double *inre = ad(NK), *inim = ad(NK), *outre = ad(NK), *outim = ad(NK);
    double *tre = ad(R), *tim = ad(R);
    /* slab scratch sized for the largest Kb (= K) */
    double *sinre = ad(NK), *sinim = ad(NK), *soutre = ad(NK), *soutim = ad(NK);
    for (size_t i = 0; i < NK; i++) { inre[i] = (double)(i%7)*0.5-1; inim[i] = (double)(i%5)*0.3-0.7; }
    for (int j = 0; j < R; j++) { tre[j] = 0.92388 - 0.01*j; tim[j] = -0.38268 + 0.01*j; }

    const size_t Kbs[] = { 256, 128, 64, 32 };
    const int Ss[] = { 1, 2, 3, 4 };

    printf("=== amortized L1-tiled proxy: radix-%d, N=%d, K=%zu (nbf=%d) ===\n", R, N, K, nbf);
    printf("gather=[N x Kb] once per slab, S cascade passes, scatter once.\n");
    printf("%-5s %-3s %12s %12s %8s  %s\n", "Kb", "S", "strided_A", "buffered_B", "B/A", "verdict");
    printf("------+----+------------+------------+--------+----------\n");

    for (size_t kbi = 0; kbi < sizeof Kbs/sizeof Kbs[0]; kbi++) {
        const size_t Kb = Kbs[kbi];
        if (Kb > K) continue;
        const size_t nslab = K / Kb;
        const ptrdiff_t isA = (ptrdiff_t)((size_t)nbf * K);    /* full-plane leg stride */
        const ptrdiff_t isB = (ptrdiff_t)((size_t)nbf * Kb);   /* slab leg stride */

        for (size_t si = 0; si < sizeof Ss/sizeof Ss[0]; si++) {
            const int S = Ss[si];
            size_t reps = (size_t)(2e7 / (double)(NK * S + 1)); if (reps < 20) reps = 20; if (reps > 5000) reps = 5000;

            /* ---- A: strided full-plane cascade (S passes), no copy ---- */
            for (int w = 0; w < 4; w++)
                for (int s = 0; s < S; s++)
                    for (int bf = 0; bf < nbf; bf++)
                        radix8_hc2hc_dit_fwd_avx2(&inre[(size_t)bf*K], &inim[(size_t)bf*K],
                                                  &outre[(size_t)bf*K], &outim[(size_t)bf*K],
                                                  tre, tim, isA, isA, K);
            double abest = 1e30;
            for (int t = 0; t < 7; t++) {
                double t0 = vfft_proto_now_ns();
                for (size_t r = 0; r < reps; r++)
                    for (int s = 0; s < S; s++)
                        for (int bf = 0; bf < nbf; bf++)
                            radix8_hc2hc_dit_fwd_avx2(&inre[(size_t)bf*K], &inim[(size_t)bf*K],
                                                      &outre[(size_t)bf*K], &outim[(size_t)bf*K],
                                                      tre, tim, isA, isA, K);
                double e = (vfft_proto_now_ns() - t0) / reps; if (e < abest) abest = e;
            }

            /* ---- B: per-slab gather -> S unit-strided passes -> scatter ---- */
            double bbest = 1e30;
            for (int t = 0; t < 7 + 4; t++) {   /* first 4 = warmup (best-of-7 after) */
                double t0 = vfft_proto_now_ns();
                for (size_t rr = 0; rr < reps; rr++)
                    for (size_t sl = 0; sl < nslab; sl++) {
                        const size_t b = sl * Kb;
                        for (int r = 0; r < N; r++) {        /* gather [N x Kb] contiguous */
                            memcpy(&sinre[(size_t)r*Kb], &inre[(size_t)r*K + b], Kb*8);
                            memcpy(&sinim[(size_t)r*Kb], &inim[(size_t)r*K + b], Kb*8);
                        }
                        for (int s = 0; s < S; s++)
                            for (int bf = 0; bf < nbf; bf++)
                                radix8_hc2hc_dit_fwd_avx2(&sinre[(size_t)bf*Kb], &sinim[(size_t)bf*Kb],
                                                          &soutre[(size_t)bf*Kb], &soutim[(size_t)bf*Kb],
                                                          tre, tim, isB, isB, Kb);
                        for (int r = 0; r < N; r++) {        /* scatter back */
                            memcpy(&outre[(size_t)r*K + b], &soutre[(size_t)r*Kb], Kb*8);
                            memcpy(&outim[(size_t)r*K + b], &soutim[(size_t)r*Kb], Kb*8);
                        }
                    }
                double e = (vfft_proto_now_ns() - t0) / reps;
                if (t >= 4 && e < bbest) bbest = e;
            }

            double ba = bbest / abest;
            printf("%-5zu %-3d %12.0f %12.0f %8.3f  %s\n", Kb, S, abest, bbest, ba,
                   ba < 0.95 ? "B WINS" : (ba > 1.05 ? "A wins" : "~tie"));
        }
        printf("------+----+------------+------------+--------+----------\n");
    }

    vfft_proto_aligned_free(inre); vfft_proto_aligned_free(inim);
    vfft_proto_aligned_free(outre); vfft_proto_aligned_free(outim);
    vfft_proto_aligned_free(tre); vfft_proto_aligned_free(tim);
    vfft_proto_aligned_free(sinre); vfft_proto_aligned_free(sinim);
    vfft_proto_aligned_free(soutre); vfft_proto_aligned_free(soutim);
    return 0;
}
