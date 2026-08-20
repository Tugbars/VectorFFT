/* wisdom2_2d_gate.c — thin driver for the wave-3 flip gate. All logic lives
 * in src/core/wisdom2/wisdom2_2d_gate.h (module-owned).
 *
 * 🔴 Point at a SCRATCH dual dir (frozen legacy files + migrated
 *    wisdom2_2d.txt). The 3D leg banks into it.
 *
 * Build: python build.py --src benches/wisdom2_2d_gate.c --vfft --compile
 * Run  : wisdom2_2d_gate.exe <scratch wisdir>
 */
#include <stdio.h>
#include "wisdom2_2d_gate.h"

int main(int argc, char **argv)
{
    int fails;
    if (argc < 2) { printf("usage: %s <SCRATCH dual wisdir>\n", argv[0]); return 2; }
    fails = vfft_wisdom2_2d_gate_run(argv[1]);
    if (fails < 0) return 2;
    return fails ? 1 : 0;
}
