/* Minimal crash reproducer */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "bench_compat.h"
#include "fft_radix4_avx2.h"
#include "fft_radix8_avx2.h"
#include "fft_radix16_avx2_ct_n1.h"
#include "fft_radix16_avx2_ct_t1_dit.h"

static void init_tw(double *W_re, double *W_im, size_t R, size_t me) {
    size_t N = R * me;
    for (size_t n = 1; n < R; n++)
        for (size_t m = 0; m < me; m++) {
            double a = -2.0 * M_PI * (double)(n * m) / (double)N;
            W_re[(n-1)*me + m] = cos(a);
            W_im[(n-1)*me + m] = sin(a);
        }
}

int main(void) {
    double __attribute__((aligned(32))) in[64], ii[64], out[64], oi[64];
    double __attribute__((aligned(32))) W_re[64], W_im[64];

    for(int i=0;i<64;i++){in[i]=(double)i/64.0;ii[i]=0;}

    /* Test 1: 4x16 */
    printf("Test 4x16...\n"); fflush(stdout);
    init_tw(W_re, W_im, 4, 16);
    radix16_n1_ovs_fwd_avx2(in, ii, out, oi, 4, 1, 4, 16);
    radix4_t1_dit_fwd_avx2(out, oi, W_re, W_im, 16, 16);
    printf("  out[0]=%.4f OK\n", out[0]); fflush(stdout);

    /* Test 2: 8x8 */
    printf("Test 8x8...\n"); fflush(stdout);
    init_tw(W_re, W_im, 8, 8);
    radix8_n1_ovs_fwd_avx2(in, ii, out, oi, 8, 1, 8, 8);
    printf("  n1_ovs done\n"); fflush(stdout);
    radix8_t1_dit_fwd_avx2(out, oi, W_re, W_im, 8, 8);
    printf("  out[0]=%.4f OK\n", out[0]); fflush(stdout);

    printf("All passed.\n");
    return 0;
}
