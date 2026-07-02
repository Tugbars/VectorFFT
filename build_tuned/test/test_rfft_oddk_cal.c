/* test_rfft_oddk_cal.c — ground-truth check: does the r2c/rfft calibrator accept ODD K?
 * (memory says "calibrator rejects K%8!=0"; rfft.h looks arbitrary-K now — settle it.)
 * Build: python build.py --src test/test_rfft_oddk_cal.c --compile */
#define VFFT_RFFT_MAX_RADIX 32
#define VFFT_RFFT_RANGED 1
#include <stdio.h>
#include <string.h>
#include "rfft_registry_avx2.h"   /* rfft_codelets_t + rfft_register_all_avx2 */
#include "c2r_registry_avx2.h"    /* c2r_register_all_avx2 (bwd codelets) */
#include "rfft_calibrate.h"       /* vfft_rfft_calibrate */
#include "wisdom_reader.h"        /* vfft_proto_wisdom_entry_t */

int main(void)
{
    rfft_codelets_t reg; memset(&reg, 0, sizeof reg);
    rfft_register_all_avx2(&reg);
    c2r_register_all_avx2(&reg);

    int N = 256;
    int Ks[] = {4, 7, 8, 11, 15, 16, 23, 32};
    printf("# vfft_rfft_calibrate(N=%d, K) — rc=0 means CALIBRATABLE, -1 means rejected\n", N);
    for (int i = 0; i < (int)(sizeof(Ks) / sizeof(Ks[0])); i++) {
        int K = Ks[i];
        vfft_proto_wisdom_entry_t out; memset(&out, 0, sizeof out);
        int rc = vfft_rfft_calibrate(N, (size_t)K, &reg, &out);
        printf("  K=%-3d (K%%8=%d): rc=%-2d", K, K % 8, rc);
        if (rc == 0) {
            printf("  factors=[");
            for (int s = 0; s < out.nf; s++) printf("%s%d", s ? "," : "", out.factors[s]);
            printf("]  ns=%.0f", out.best_ns);
        } else {
            printf("  <REJECTED>");
        }
        printf("\n");
    }
    return 0;
}
