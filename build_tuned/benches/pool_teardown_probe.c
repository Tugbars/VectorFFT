/* pool_teardown_probe.c — does creating/executing a 2D real IL plan tear
 * down the process thread pool? (MT research, 2026-08-26: both research
 * passes claim vfft_set_num_threads(1) from a child create destroys the
 * pool for EVERY tier.) Pure observation, no library change.
 * Build: python build.py --src benches/pool_teardown_probe.c --vfft --compile */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "cpu_cache.h"   /* the detected SMT / L2 / L3 the pin map now uses */

int main(int argc, char **argv)
{
    const char *wisdir = argc > 1 ? argv[1] : ".";
    const int N1 = 512, N2 = 512;
    const size_t RN = (size_t)N1 * N2;
    const size_t CN = (size_t)N1 * (N2 / 2 + 1);
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    vfft_config_t c;
    vfft_plan p2, p1;
    double *x = (double *)malloc(RN * 8), *z = (double *)malloc(2 * CN * 8);
    setvbuf(stdout, NULL, _IONBF, 0);
    if (!x || !z) return 2;
    memset(x, 0, RN * 8);

    vfft_set_num_threads(8);
    printf("after set_num_threads(8)            : %d\n",
           vfft_get_num_threads());

    /* control: a 1D c2c plan asking for 8 */
    memset(&c, 0, sizeof c);
    c.transform = VFFT_C2C;
    c.placement = VFFT_INPLACE;
    c.rigor = VFFT_MEASURE;
    c.dims = 1; c.n[0] = 1024; c.howmany = 8;
    c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.nthreads = 8; c.wisdom = W; c.wisdom_write = 0;
    p1 = vfft_create(&c);
    printf("after 1D c2c create (nthreads=8)    : %d   <- control\n",
           vfft_get_num_threads());

    /* the subject: a 2D real IL plan asking for 8 */
    memset(&c, 0, sizeof c);
    c.transform = VFFT_R2C;
    c.placement = VFFT_OUTOFPLACE;
    c.rigor = VFFT_MEASURE;
    c.dims = 2; c.n[0] = N1; c.n[1] = N2;
    c.howmany = 1; c.nthreads = 8;
    c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.wisdom = W; c.wisdom_write = 0;
    p2 = vfft_create(&c);
    printf("after 2D real IL create (nthreads=8): %d   <- SUBJECT%s\n",
           vfft_get_num_threads(),
           vfft_get_num_threads() < 8 ? "  *** POOL DESTROYED ***" : "");
    if (!p2) { printf("2D create FAILED\n"); return 1; }

    vfft_set_num_threads(8);
    printf("re-armed to 8                       : %d\n",
           vfft_get_num_threads());
    vfft_execute(p2, VFFT_FORWARD, x, NULL, z, NULL);
    printf("after ONE 2D real execute           : %d%s\n",
           vfft_get_num_threads(),
           vfft_get_num_threads() < 8 ? "   *** POOL DESTROYED ***" : "");

    /* does the control plan still see a pool? */
    printf("1D plan's own tc workers            : %d\n",
           p1 ? vfft_plan_tc_workers(p1) : -1);
    printf("detected SMT / L2 / L3              : %d / %ld KB / %ld KB\n",
           vfft_cpu_smt(), vfft_cpu_l2_bytes() / 1024,
           vfft_cpu_l3_bytes() / 1024);
    if (p1) vfft_destroy(p1);
    vfft_destroy(p2);
    if (W) vfft_wisdom_free(W);
    free(x); free(z);
    return 0;
}
