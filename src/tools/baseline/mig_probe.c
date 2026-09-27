/* mig_probe.c — the OOP wisdom MIGRATOR on a synthetic legacy oop_wisdom.txt.
 *
 * Written for the kind-3 record split (layout separation phase 5): the repo
 * holds no legacy oop_wisdom.txt, so nothing exercised the migrator. This
 * writes one with every legacy kind (LEAF, BAILEY2, MODEB, kind-3 with both
 * axes + il_kv, split-only, CCOL with its chain/vars codes, mono, kind 5),
 * runs vw2_migrate_oop_gate into OUTDIR, and prints its verdict. Compare two
 * builds by the store files they write (they must be byte-identical up to the
 * date= token) and by the log.
 *
 * KNOWN, PRE-EXISTING (F4 in docs/roadmap/layout_separation_plan.md): the
 * reader-gate reports the four kind-3 cells MISSED on the pre-split code too -
 * migrated kind-3 rows are keyed place=* and the kind-3 scan matches placement
 * exactly, so the check cannot find them. The migrated rows themselves are
 * right.
 *
 * Build (any ISA; from the repo root):
 *   gcc -D_GNU_SOURCE -O1 -mavx2 -mfma -mno-avx512f -w $(include flags) \
 *       src/tools/baseline/mig_probe.c src/dag-fft-compiler/.obj/avx2/libdagcodelets.a -lm -lpthread
 * Run: mig_probe LEGACYDIR OUTDIR */
#include "vfft.c"
#include "wisdom2_migrate.h"
int main(int argc, char **argv)
{
    char path[512];
    int ch[4] = { 16, 16, 0, 0 }, cv[4] = { 2, 0, 0, 0 };
    int cc = vfft_k1_cc_chain_encode(ch, 2), ccv = vfft_k1_cc_vars_encode(cv, 2);
    FILE *f;
    snprintf(path, sizeof path, "%s/oop_wisdom.txt", argv[1]);
    f = fopen(path, "w");
    fprintf(f, "# synthetic legacy oop wisdom\n");
    fprintf(f, "64 512 0 117350.0\n");
    fprintf(f, "1024 120 1 32 32 1 185550.0\n");
    fprintf(f, "1024 256 2 5 4 4 4 4 4 0 2 2 2 2 502460.0\n");
    fprintf(f, "256 4 3 2 16 16 5 16 16 153.7 3\n");              /* both axes + il_kv */
    fprintf(f, "512 4 3 1 32 16 0 0 0 201.0\n");                  /* split only, IL none */
    fprintf(f, "16384 4 3 7 64 256 0 0 0 %d %d 900.0\n", cc, ccv);  /* CCOL */
    fprintf(f, "128 1 3 4 8 16 3 8 16 72.7 1\n");                 /* mono both */
    fprintf(f, "8192 1 5 5 12.0\n");                              /* kind 5 */
    fclose(f);
    printf("cc=%d ccv=%d\n", cc, ccv);
    int rc = vw2_migrate_oop_gate(path, argv[2]);
    printf("vw2_migrate_oop_gate rc=%d\n", rc);
    return rc;
}
