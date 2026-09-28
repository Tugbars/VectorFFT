/* wisdom2_drop_retired_nat.c — migrate a wisdom2 store directory for D2.
 *
 * Deletes the @nat-family rows of the retired interleaved modes (mode=zcasc,
 * mode=ilp, mode=conv; see vw2_migrate_drop_retired_nat in
 * src/core/wisdom2/wisdom2_migrate.h). The library already ignores them; this
 * removes them from the files. Run it once on every store directory you keep
 * (each host subtree, e.g. src/wisdom/Zen4, is its own directory). Idempotent.
 *
 * Build (from the repo root, any ISA whose codelet library is built):
 *   gcc -D_GNU_SOURCE -O1 -mavx2 -mfma -mno-avx512f -w $(include flags) \
 *       src/tools/wisdom2_drop_retired_nat.c \
 *       src/dag-fft-compiler/.obj/avx2/libdagcodelets.a -lm -lpthread
 * (include flags: python3 -c "import sys; sys.path.insert(0,'src/tools/baseline');
 *  import toolchain; print(' '.join(toolchain.include_flags()))")
 * Run: wisdom2_drop_retired_nat DIR [DIR...] */
#include "vfft.c"
#include "wisdom2_migrate.h"
int main(int argc, char **argv)
{
    int i, bad = 0;
    if (argc < 2) {
        fprintf(stderr, "usage: %s DIR [DIR...]\n", argv[0]);
        return 2;
    }
    for (i = 1; i < argc; i++) {
        int n = vw2_migrate_drop_retired_nat(argv[i]);
        printf("%s: %s\n", argv[i], n < 0 ? "FAILED" : n ? "migrated" : "nothing to do");
        if (n < 0) bad = 1;
    }
    return bad;
}
