/* test_wisdom_read.c — prove wisdom_reader.h loads the migrated (all-v6) spike_wisdom.txt,
 * including the exec_me verdicts folded in from the padded sweep. Build: build.py --src ... --compile */
#include <stdio.h>
#include "wisdom_reader.h"

int main(void)
{
    const char *path = "C:/Users/Tugbars/Desktop/highSpeedFFT/src/dag-fft-compiler/"
                       "generator/generated/spike_wisdom.txt";
    vfft_proto_wisdom_t w;
    if (vfft_proto_wisdom_load(&w, path) != 0) { printf("LOAD FAILED: %s\n", path); return 1; }
    printf("loaded %zu entries from spike_wisdom.txt\n", w.count);

    /* mix of: untouched original (256,4), pad-winner (256,7)+its aligned (256,8),
     * tail-winner (256,15), pad (512,7)/(1024,11-tail). */
    int cells[][2] = {{256,4},{256,7},{256,8},{256,15},{512,7},{512,31},{1024,11}};
    int fails = 0;
    for (int i = 0; i < (int)(sizeof(cells)/sizeof(cells[0])); i++) {
        const vfft_proto_wisdom_entry_t *e = vfft_proto_wisdom_lookup(&w, cells[i][0], (size_t)cells[i][1]);
        if (!e) { printf("  (%d,%d): NOT FOUND\n", cells[i][0], cells[i][1]); continue; }
        size_t Kp = ((size_t)e->K + 3) & ~(size_t)3;
        const char *verdict = e->exec_me == 0 ? "unmeasured"
                            : e->exec_me == (int)Kp ? "PAD" : "tail";
        printf("  (%d,%-3d): nf=%d exec_me=%-3d [%s]  factors=[", e->N, (int)e->K, e->nf, e->exec_me, verdict);
        for (int j = 0; j < e->nf; j++) printf("%s%d", j ? "," : "", e->factors[j]);
        printf("]  variants=[");
        for (int j = 0; j < e->nf; j++) printf("%s%d", j ? "," : "", e->variants[j]);
        printf("]\n");
        /* sanity: nf in range, factors product == N (proves fields didn't shift) */
        long long prod = 1; for (int j = 0; j < e->nf; j++) prod *= e->factors[j];
        if (e->nf <= 0 || prod != (long long)e->N) { printf("    ^ FIELD SHIFT (prod=%lld != N)\n", prod); fails++; }
    }
    vfft_proto_wisdom_free(&w);
    printf(fails ? "\nRESULT: %d malformed\n" : "\nRESULT: reader parses v6 cleanly\n", fails);
    return fails ? 1 : 0;
}
