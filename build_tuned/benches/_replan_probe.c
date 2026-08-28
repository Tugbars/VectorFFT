/* TEMPORARY diagnostic: re-run the DP planner on one cell and report what it
 * picks now. Public API only. Delete after use.
 *
 * Cell under test mirrors the banked key exactly:
 *   @cell t=c2c n=256 q=256 ord=scr place=ip  -> chain=4.4.16  ns=40900
 * against the 2026-06 CSV's 73730 ns for plan 4x4x16/DIT.
 */
#include <stdio.h>
#include <stdlib.h>
#include "vfft.h"

int main(int argc, char **argv)
{
    int N = argc > 1 ? atoi(argv[1]) : 256;
    size_t K = argc > 2 ? (size_t)atol(argv[2]) : 256;

    vfft_config_t cfg = {0};
    cfg.transform    = VFFT_C2C;
    cfg.placement    = VFFT_INPLACE;
    cfg.dims         = 1;
    cfg.n[0]         = N;
    cfg.howmany      = K;
    cfg.order        = VFFT_ORDER_SCRAMBLED;
    cfg.layout       = VFFT_LAYOUT_SPLIT;
    cfg.rigor        = VFFT_MEASURE;
    cfg.recalibrate  = 1;      /* force the DP planner to run */
    cfg.wisdom_write = 1;      /* and bank what it finds */

    printf("re-planning N=%d K=%zu (c2c, in-place, split, scrambled, MEASURE)\n",
           N, (size_t)K);
    fflush(stdout);

    vfft_plan p = vfft_create(&cfg);
    if (!p) { printf("CREATE FAILED\n"); return 1; }
    printf("created; stride=%zu\n", vfft_plan_stride(p));
    vfft_destroy(p);
    printf("done - read the banked cell from the scratch wisdom dir\n");
    return 0;
}
