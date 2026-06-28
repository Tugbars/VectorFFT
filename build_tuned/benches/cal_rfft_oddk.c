/* cal_rfft_oddk.c — calibrate the rfft factorization+per-stage-variant for the
 * tail-tax cells (aligned K=8,16 + odd neighbors 7,15,17 at N=256,512,1024) and
 * emit a v5 wisdom file. Lets the bench compare odd-K vs aligned on CALIBRATED
 * factorizations (not the heuristic all-flat the empty-wisdom path falls to), so
 * the tail tax is isolated from the calibration gap. Single-TU: #includes vfft.c.
 * Build: cd build_tuned && python build.py --src benches/cal_rfft_oddk.c --compile
 * Run from build_tuned/benches/ ; writes rfft_wisdom_oddk.txt in CWD.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <string.h>
#include "vfft.c"

int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    stride_env_init();
    stride_pin_thread(2);
    vfft_proto_wisdom_t W;
    memset(&W, 0, sizeof W);

    int    Ns[] = {256, 512, 1024};
    size_t Ks[] = {7, 8, 15, 16, 17};
    printf("== rfft calibration (factor + per-stage variant) for the tail-tax cells ==\n");
    for (int ni = 0; ni < (int)(sizeof Ns / sizeof Ns[0]); ni++)
        for (int ki = 0; ki < (int)(sizeof Ks / sizeof Ks[0]); ki++) {
            int N = Ns[ni]; size_t K = Ks[ki];
            vfft_proto_wisdom_entry_t e;
            memset(&e, 0, sizeof e);
            int rc = vfft_rfft_calibrate(N, K, _rfft_registry(), &e);
            if (rc == 0) {
                printf("  N=%-4d K=%-3zu nf=%d f=[", N, K, e.nf);
                for (int s = 0; s < e.nf; s++) printf("%s%d", s ? "," : "", e.factors[s]);
                printf("] v=[");
                for (int s = 0; s < e.nf; s++) printf("%s%d", s ? "," : "", e.variants[s]);
                printf("] %.0f ns\n", e.best_ns);
                vfft_proto_wisdom_add(&W, &e, 1);
            } else {
                printf("  N=%-4d K=%-3zu  CALIBRATE FAILED rc=%d\n", N, K, rc);
            }
        }

    int sv = vfft_proto_wisdom_save(&W, "rfft_wisdom_oddk.txt");
    printf("saved %zu entries (rc=%d) -> rfft_wisdom_oddk.txt\n", W.count, sv);
    return sv;
}
