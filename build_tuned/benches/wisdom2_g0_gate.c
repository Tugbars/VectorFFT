/* wisdom2_g0_gate — THIN DRIVER over the module-owned self-test
 * (src/core/wisdom2/wisdom2_selftest.h). Per the thin-driver law, this file
 * holds no logic: it parses one argument and calls in.
 *
 * Usage: wisdom2_g0_gate [scratch_dir]   (default: ./w2_g0_scratch)
 * Exit 0 = ALL PASS.
 */
#include "../../src/core/wisdom2/wisdom2_selftest.h"

int main(int argc, char **argv)
{
    return vw2_g0_selftest(argc > 1 ? argv[1] : "w2_g0_scratch") ? 1 : 0;
}
