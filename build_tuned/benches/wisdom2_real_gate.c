/* wisdom2_real_gate.c — thin driver for the wave-2 (r2c/c2r ROUTE) flip gate.
 * All logic lives in src/core/wisdom2/wisdom2_real_gate.h (module-owned).
 *
 * 🔴 Point at a SCRATCH wisdir. The gate BANKS and SAVES.
 *
 * Build: python build.py --src benches/wisdom2_real_gate.c --compile
 * Run  : wisdom2_real_gate.exe <scratch wisdir>
 */
#include <stdio.h>
#include "wisdom2_real_gate.h"

int main(int argc, char **argv)
{
    int fails;
    if (argc < 2) { printf("usage: %s <SCRATCH wisdir>\n", argv[0]); return 2; }
    fails = vfft_wisdom2_real_gate_run(argv[1]);
    if (fails < 0) return 2;
    return fails ? 1 : 0;
}
