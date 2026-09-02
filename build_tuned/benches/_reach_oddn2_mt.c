/* ADVERSARIAL REACH PROBE: does an ODD-N2 2D real IL cell reach the
 * column-MT race (_il2d_real_colmt_race) and BANK an rl row?
 * The claim under test says these cells "RACE NOTHING and BANK NOTHING". */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_fingerprint.h"

int main(int argc, char **argv)
{
    static char buf[65536];
    long c[VFFT__FP_NCOUNTERS];
    vfft_config_t cfg; vfft_plan p;
    int T = (argc > 1) ? atoi(argv[1]) : 4;
    int N1 = (argc > 2) ? atoi(argv[2]) : 256;
    int N2 = (argc > 3) ? atoi(argv[3]) : 127;

    if (T > 1) vfft_set_num_threads(T);
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_R2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.layout    = VFFT_LAYOUT_INTERLEAVED;
    cfg.order     = (argc > 4 && atoi(argv[4])) ? VFFT_ORDER_NATURAL : VFFT_ORDER_DEFAULT;
    cfg.dims      = 2;
    cfg.n[0] = N1; cfg.n[1] = N2;
    cfg.howmany = 1;
    cfg.nthreads = T;
    cfg.rigor = VFFT_MEASURE;
    cfg.wisdom_write = 1;

    p = vfft_create(&cfg);
    vfft__fp_counters(c);
    printf("@@cell 2d.il.oop.r2c.%dx%d T=%d\n", N1, N2, T);
    if (!p) { printf("@@status refuse races=%ld\n", c[5]); return 0; }
    printf("@@status accept races=%ld colmtpasses=%ld\n", c[5], c[1]);
    vfft__fingerprint(p, buf, sizeof buf);
    fputs(buf, stdout);
    vfft_destroy(p);
    return 0;
}
