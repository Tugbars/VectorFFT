/* Correctness test: t1_buf_dit vs t1_dit produce identical output.
 * Both codelets are in-place CT butterflies, so given identical input
 * and twiddle table, they must produce bit-identical output (same
 * mathematical operations, same rounding order within each column).
 */
#include "t1_dit.h"
#include "t1_buf_dit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define R 64

int main() {
    /* Test a few (me, ios) combinations: includes tile-aligned and
     * misaligned sizes to exercise both the tile-full path and tail. */
    struct { size_t me, ios; } cases[] = {
        {  64,  72 },
        { 128, 136 },
        { 256, 264 },
        { 512, 520 },
        {1024, 1032},
        {2048, 2056},
        /* sizes not multiples of tile */
        {  96, 104 },
        { 200, 208 },
    };
    size_t ncases = sizeof(cases) / sizeof(cases[0]);
    size_t max_total = 64 * 2056;

    double *ref_re = aligned_alloc(64, max_total * sizeof(double));
    double *ref_im = aligned_alloc(64, max_total * sizeof(double));
    double *buf_re = aligned_alloc(64, max_total * sizeof(double));
    double *buf_im = aligned_alloc(64, max_total * sizeof(double));
    double *orig_re = aligned_alloc(64, max_total * sizeof(double));
    double *orig_im = aligned_alloc(64, max_total * sizeof(double));

    /* Twiddle table: flat layout, (R-1)*me_max scalars each re/im */
    size_t tw_max = (R - 1) * 2056;
    double *W_re = aligned_alloc(64, tw_max * sizeof(double));
    double *W_im = aligned_alloc(64, tw_max * sizeof(double));
    for (size_t i = 0; i < tw_max; i++) {
        /* Deterministic pseudo-twiddle values; not actual twiddles (we
         * aren't checking FFT correctness, only that buf matches t1). */
        double angle = (double)i / tw_max * 6.28318530717958647692;
        W_re[i] = cos(angle);
        W_im[i] = sin(angle);
    }

    srand(1234);
    for (size_t i = 0; i < max_total; i++) {
        orig_re[i] = (double)rand() / RAND_MAX - 0.5;
        orig_im[i] = (double)rand() / RAND_MAX - 0.5;
    }

    int failures = 0;

    for (size_t c = 0; c < ncases; c++) {
        size_t me = cases[c].me;
        size_t ios = cases[c].ios;
        size_t total = R * ios;

        for (int dir = 0; dir < 2; dir++) {
            /* Reset both buffers to same input */
            memcpy(ref_re, orig_re, total * sizeof(double));
            memcpy(ref_im, orig_im, total * sizeof(double));
            memcpy(buf_re, orig_re, total * sizeof(double));
            memcpy(buf_im, orig_im, total * sizeof(double));

            if (dir == 0) {
                radix64_t1_dit_fwd_avx2(ref_re, ref_im, W_re, W_im, ios, me);
                radix64_t1_buf_dit_tile64_temporal_fwd_avx2(buf_re, buf_im, W_re, W_im, ios, me);
            } else {
                radix64_t1_dit_bwd_avx2(ref_re, ref_im, W_re, W_im, ios, me);
                radix64_t1_buf_dit_tile64_temporal_bwd_avx2(buf_re, buf_im, W_re, W_im, ios, me);
            }

            /* Compare */
            double max_diff = 0.0;
            size_t bad_idx = (size_t)-1;
            for (size_t m = 0; m < me; m++) {
                for (size_t row = 0; row < R; row++) {
                    size_t idx = row * ios + m;
                    double dr = ref_re[idx] - buf_re[idx];
                    double di = ref_im[idx] - buf_im[idx];
                    double d = sqrt(dr*dr + di*di);
                    if (d > max_diff) { max_diff = d; bad_idx = idx; }
                }
            }

            const char *dname = dir == 0 ? "fwd" : "bwd";
            if (max_diff < 1e-12) {
                printf("  me=%4zu ios=%4zu %s: PASS (max_diff=%.2e)\n",
                       me, ios, dname, max_diff);
            } else {
                printf("  me=%4zu ios=%4zu %s: FAIL (max_diff=%.6e at idx %zu)\n",
                       me, ios, dname, max_diff, bad_idx);
                failures++;
            }
        }
    }

    free(ref_re); free(ref_im); free(buf_re); free(buf_im);
    free(orig_re); free(orig_im); free(W_re); free(W_im);

    if (failures == 0) {
        printf("\nAll cases PASS\n");
        return 0;
    } else {
        printf("\n%d FAILURES\n", failures);
        return 1;
    }
}
