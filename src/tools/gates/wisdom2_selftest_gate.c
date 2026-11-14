/* wisdom2_selftest_gate.c -- the wisdom store's own self-test (src/core/common/wisdom/wisdom2_selftest.h), run on
 * a scratch directory: the grammar, the merge law, the shards, the save's merge with the disk and its lock, the
 * stamp, and the build's instruction set on every row (isa=, v1.4). Built for FP64. Built with VFFT_FLOAT defined
 * (the FP32 file, wisdom2f_oop.txt), the isa= cases (T31) pass too; the older cases assume the six FP64 shard files
 * and do not apply there. Nothing is timed.
 *
 * Build: python gauntlet/build.py --compile --src src/tools/gates/wisdom2_selftest_gate.c   (standalone)
 * Run:   wisdom2_selftest_gate --wisdir <scratch dir>      (the directory is created and wiped of store files) */
#include <stdio.h>
#include <string.h>
#if defined(_WIN32)
#include <direct.h>
#endif
#include "common/wisdom/wisdom2.h"
#include "common/wisdom/wisdom2_selftest.h"

int main(int argc, char **argv)
{
    const char *dir = NULL;
    int a, fails;
    for (a = 1; a < argc; a++)
        if (!strcmp(argv[a], "--wisdir") && a + 1 < argc) dir = argv[++a];
    if (!dir)
    {
        printf("usage: wisdom2_selftest_gate --wisdir <scratch dir>\n");
        return 2;
    }
    fails = vw2_g0_selftest(dir);
    printf("%d failure(s)\n", fails);
    printf(fails ? "WISDOM2 SELFTEST FAILED\n" : "WISDOM2 SELFTEST ALL PASS\n");
    return fails != 0;
}
