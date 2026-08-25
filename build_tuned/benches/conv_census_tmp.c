/* transient: the convert-arm census (owner-approved 2026-08-25).
 * Sweeps the IL c2c cell grid, executes each cell once per direction with
 * VFFT_CONV_LOG=1 set by the harness, and prints a cell header before each
 * execute so the [conv] stderr lines attribute to their cell.
 * usage: conv_census_tmp <wisdir>  (2>&1 | a post-pass groups lines) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

static const int NS[] = { 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096,
                          24, 48, 96, 192, 384, 768, 1536,
                          9, 27, 81, 7, 11, 13, 127 };
static const int KS[] = { 1, 2, 4, 8 };
static const int ORDS[] = { VFFT_ORDER_DEFAULT, VFFT_ORDER_SCRAMBLED,
                            VFFT_ORDER_NATURAL };
static const char *ONM[] = { "def", "nat", "scr" }; /* index by enum value */

int main(int argc, char **argv)
{
    vfft_wisdom *W = vfft_wisdom_load(argc > 1 ? argv[1] : ".");
    setvbuf(stdout, NULL, _IONBF, 0);
    setvbuf(stderr, NULL, _IONBF, 0);
    for (int pi = 0; pi < 2; pi++)
        for (int oi = 0; oi < 3; oi++)
            for (int ki = 0; ki < (int)(sizeof KS / sizeof *KS); ki++)
                for (int ni = 0; ni < (int)(sizeof NS / sizeof *NS); ni++)
                {
                    const int N = NS[ni], K = KS[ki], ord = ORDS[oi];
                    const int ip = (pi == 0);
                    vfft_config_t c;
                    memset(&c, 0, sizeof c);
                    c.transform = VFFT_C2C;
                    c.placement = ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
                    c.rigor = VFFT_MEASURE;
                    c.dims = 1;
                    c.n[0] = N;
                    c.howmany = K;
                    c.layout = VFFT_LAYOUT_INTERLEAVED;
                    c.nthreads = 1;
                    c.wisdom = W;
                    c.order = ord;
                    vfft_plan p = vfft_create(&c);
                    if (!p)
                    {
                        fprintf(stderr,
                                "cell pl=%s ord=%s N=%d K=%d REFUSED\n",
                                ip ? "ip" : "oop", ONM[ord], N, K);
                        continue;
                    }
                    double *a = malloc(2u * N * K * sizeof(double));
                    double *b = malloc(2u * N * K * sizeof(double));
                    for (int i = 0; i < 2 * N * K; i++)
                        a[i] = (double)rand() / RAND_MAX - 0.5;
                    for (int d = 0; d < 2; d++)
                    {
                        fprintf(stderr, "cell pl=%s ord=%s N=%d K=%d %s\n",
                                ip ? "ip" : "oop", ONM[ord], N, K,
                                d ? "bwd" : "fwd");
                        if (ip)
                            vfft_execute(p, d ? VFFT_BACKWARD : VFFT_FORWARD,
                                         a, NULL, a, NULL);
                        else
                            vfft_execute(p, d ? VFFT_BACKWARD : VFFT_FORWARD,
                                         a, NULL, b, NULL);
                    }
                    free(a);
                    free(b);
                    vfft_destroy(p);
                }
    fprintf(stderr, "census done\n");
    return 0;
}
