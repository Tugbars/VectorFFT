/* wisdom2_migrate — THIN DRIVER over the module-owned migrator
 * (src/core/wisdom2/wisdom2_migrate.h). Per the thin-driver law, this file
 * holds no logic: it parses arguments and calls in.
 *
 * Usage:
 *   wisdom2_migrate <legacy_oop_wisdom.txt> <out_dir>            one-shot migrate
 *   wisdom2_migrate <legacy_oop_wisdom.txt> <out_dir> --gate     migrate + Gate A
 *                                                                (accounting,
 *                                                                idempotency,
 *                                                                field verify)
 *
 * NEVER point out_dir at the shipped generated/ tree during wave 0 —
 * scratch-wisdir law. The legacy file is never written.
 */
#include <stdio.h>
#include <string.h>
#include "../../src/core/wisdom2/wisdom2_migrate.h"

int main(int argc, char **argv)
{
    vw2_mig_stats_t st;
    if (argc < 3) {
        fprintf(stderr, "usage: wisdom2_migrate <legacy_oop_wisdom.txt> <out_dir> [--gate]\n");
        return 2;
    }
    if (argc > 3 && !strcmp(argv[3], "--gate"))
        return vw2_migrate_oop_gate(argv[1], argv[2]) ? 1 : 0;
    return vw2_migrate_oop(argv[1], argv[2], &st, 1) == VW2_OK ? 0 : 1;
}
