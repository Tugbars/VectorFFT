/* wisdom2_migrate — THIN DRIVER over the module-owned migrator
 * (src/core/wisdom2/wisdom2_migrate.h). Per the thin-driver law, this file
 * holds no logic: it parses arguments and calls in.
 *
 * Usage:
 *   wisdom2_migrate <legacy_oop_wisdom.txt> <out_dir> [--gate]
 *       wave-1 oop migration (+ Gate A: accounting, idempotency, verify)
 *   wisdom2_migrate --2d <fft2d_c2c> <fft2d_r2c> <fft2d_c2r> <out_dir> [--gate]
 *       wave-3 2D migration (all three files; pass - for an absent file)
 *
 * NEVER point out_dir at the shipped generated/ tree before promotion —
 * scratch-wisdir law. The legacy files are never written.
 */
#include <stdio.h>
#include <string.h>
#include "../../src/core/wisdom2/wisdom2_migrate.h"

int main(int argc, char **argv)
{
    vw2_mig_stats_t st;
    if (argc > 1 && !strcmp(argv[1], "--rekey-k1role")) {
        if (argc < 3) {
            fprintf(stderr, "usage: wisdom2_migrate --rekey-k1role <store_dir>\n");
            return 2;
        }
        return vw2_migrate_rekey_k1role(argv[2]) < 0 ? 1 : 0;
    }
    if (argc > 1 && !strcmp(argv[1], "--stride")) {
        const char *spike, *rfft, *padded, *out;
        if (argc < 6) {
            fprintf(stderr, "usage: wisdom2_migrate --stride <spike> <rfft> "
                            "<padded_fossil> <out_dir> [--gate]  (- = absent file)\n");
            return 2;
        }
        spike = strcmp(argv[2], "-") ? argv[2] : NULL;
        rfft = strcmp(argv[3], "-") ? argv[3] : NULL;
        padded = strcmp(argv[4], "-") ? argv[4] : NULL;
        out = argv[5];
        if (argc > 6 && !strcmp(argv[6], "--gate"))
            return vw2_migrate_stride_gate(spike, rfft, padded, out) ? 1 : 0;
        return vw2_migrate_stride(spike, rfft, padded, out, &st, 1) == 0 ? 0 : 1;
    }
    if (argc > 1 && !strcmp(argv[1], "--2d")) {
        const char *c2c, *r2c, *c2r, *out;
        if (argc < 6) {
            fprintf(stderr, "usage: wisdom2_migrate --2d <fft2d_c2c> <fft2d_r2c> "
                            "<fft2d_c2r> <out_dir> [--gate]  (- = absent file)\n");
            return 2;
        }
        c2c = strcmp(argv[2], "-") ? argv[2] : NULL;
        r2c = strcmp(argv[3], "-") ? argv[3] : NULL;
        c2r = strcmp(argv[4], "-") ? argv[4] : NULL;
        out = argv[5];
        if (argc > 6 && !strcmp(argv[6], "--gate"))
            return vw2_migrate_2d_gate(c2c, r2c, c2r, out) ? 1 : 0;
        return vw2_migrate_2d(c2c, r2c, c2r, out, &st, 1) == 0 ? 0 : 1;
    }
    if (argc < 3) {
        fprintf(stderr, "usage: wisdom2_migrate <legacy_oop_wisdom.txt> <out_dir> [--gate]\n"
                        "       wisdom2_migrate --2d <c2c> <r2c> <c2r> <out_dir> [--gate]\n");
        return 2;
    }
    if (argc > 3 && !strcmp(argv[3], "--gate"))
        return vw2_migrate_oop_gate(argv[1], argv[2]) ? 1 : 0;
    return vw2_migrate_oop(argv[1], argv[2], &st, 1) == VW2_OK ? 0 : 1;
}
