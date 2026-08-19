/* zturn_wisdom_width_gate.c — thin driver for the banked-tcut-width replay
 * gate. All gate logic lives in src/core/oop/oop_width_gate.h (module-owned);
 * this file only parses arguments and forwards them.
 *
 * 🔴 Point --wisdir at a SCRATCH copy. The gate rewrites wisdom2_oop.txt
 *    while it runs (byte-restored on exit).
 *
 * Build: python build.py --src benches/zturn_wisdom_width_gate.c --vfft
 * Run  : zturn_wisdom_width_gate.exe --wisdir <scratch wisdir> [--cell 16384]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oop_width_gate.h"

int main(int argc, char **argv)
{
    const char *wisdir = NULL;
    int N = 16384, fails, i;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--wisdir") && i + 1 < argc) wisdir = argv[++i];
        else if (!strcmp(argv[i], "--cell") && i + 1 < argc) N = atoi(argv[++i]);
    }
    if (!wisdir) { printf("usage: %s --wisdir <SCRATCH dir> [--cell N]\n", argv[0]); return 2; }
    fails = vfft_oop_width_gate_run(wisdir, N);
    if (fails < 0) return 2;
    return fails ? 1 : 0;
}
