/* wisdom_load_check.c — confirm the dag wisdom reader ingests spike_wisdom.txt
 * (incl. the grafted old-production K=256 cells) into correctly-parsed entries.
 * Build: build_tuned/build.py --src calibrator/wisdom_load_check.c --compile */
#include <stdio.h>
#include <string.h>
#include "../core/env.h"
#include "../core/planner.h"   /* vfft_proto_wisdom_t + load/lookup (wisdom_reader.h) */

int main(int argc, char **argv) {
    const char *wp = (argc > 1) ? argv[1]
        : "../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt";
    vfft_proto_wisdom_t wis; memset(&wis, 0, sizeof wis);
    int rc = vfft_proto_wisdom_load(&wis, wp);
    printf("load rc=%d  count=%zu  (%s)\n", rc, wis.count, wp);

    int grafted[][2] = {{50000,256},{60060,256},{65536,256},{78125,256},{100000,256},{117649,256}};
    int ok = 1;
    for (int i = 0; i < 6; i++) {
        const vfft_proto_wisdom_entry_t *e =
            vfft_proto_wisdom_lookup(&wis, grafted[i][0], (size_t)grafted[i][1]);
        if (!e) { printf("  MISS %d/%d\n", grafted[i][0], grafted[i][1]); ok = 0; continue; }
        printf("  %-7d/%-3d nf=%d:", e->N, (int)e->K, e->nf);
        for (int s = 0; s < e->nf; s++) printf(" %d", e->factors[s]);
        printf("  dif=%d  v=[", e->use_dif_forward);
        for (int s = 0; s < e->nf; s++) printf("%s%d", s ? "," : "", e->variants[s]);
        printf("]\n");
    }
    printf("%s\n", ok ? "ALL 6 GRAFTED CELLS PARSED OK" : "PARSE MISS — FAIL");
    return ok ? 0 : 1;
}
