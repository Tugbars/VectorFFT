/* il2d_mt_probe.c — READER D feasibility probe for the native IL 2D real tier.
 * Q1: does creating an IL 2D real plan tear the global pool down to 1?
 * Q2: what is h->nthreads on such a plan (via vfft_plan_tc_workers proxy / pool)?
 * Q3: can T independent plans (S2 plane-per-core) run concurrently and give
 *     bitwise-identical results to serial? (correctness only, not timing)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"
#include "vfft_diagnostics.h"

#define NT 4
#define N1 256
#define N2 256

typedef struct { vfft_plan p; double *x; double *z; } arg_t;

static DWORD WINAPI runner(LPVOID v)
{
    arg_t *a = (arg_t *)v;
    for (int r = 0; r < 20; r++)
        vfft_execute(a->p, VFFT_FORWARD, a->x, NULL, a->z, NULL);
    return 0;
}

int main(void)
{
    const size_t RN = (size_t)N1 * N2, hp1 = N2 / 2 + 1;
    const size_t CN = (size_t)N1 * hp1;
    vfft_config_t c;
    vfft_plan pl[NT];
    double *x[NT], *z[NT], *zref;
    arg_t a[NT];
    HANDLE th[NT];
    size_t i; int t;

    printf("pool at start                 : %d\n", vfft_get_num_threads());
    vfft_set_num_threads(8);
    printf("pool after set_num_threads(8) : %d\n", vfft_get_num_threads());

    memset(&c, 0, sizeof c);
    c.transform = VFFT_R2C;
    c.placement = VFFT_OUTOFPLACE;
    c.dims = 2; c.n[0] = N1; c.n[1] = N2;
    c.howmany = 1;
    c.layout = VFFT_LAYOUT_INTERLEAVED;
    c.nthreads = 8;
    c.wisdom_write = 0;

    pl[0] = vfft_create(&c);
    printf("create() returned             : %p\n", (void *)pl[0]);
    printf("pool AFTER 2D-real create     : %d   <== Q1\n",
           vfft_get_num_threads());
    if (!pl[0]) { printf("create failed\n"); return 1; }

    /* Q3: T independent plans, each on its own OS thread. */
    for (t = 1; t < NT; t++) {
        vfft_set_num_threads(8);           /* try to restore between creates */
        pl[t] = vfft_create(&c);
        printf("  plan[%d]=%p  pool now %d\n", t, (void *)pl[t],
               vfft_get_num_threads());
        if (!pl[t]) { printf("clone create %d FAILED\n", t); return 1; }
    }
    for (t = 0; t < NT; t++) {
        x[t] = (double *)malloc(RN * sizeof(double));
        z[t] = (double *)malloc((2 * CN + 8) * sizeof(double));
        for (i = 0; i < RN; i++) x[t][i] = 1.0 + 1e-6 * (double)(i & 1023);
        memset(z[t], 0, (2 * CN + 8) * sizeof(double));
        a[t].p = pl[t]; a[t].x = x[t]; a[t].z = z[t];
    }
    /* serial reference from plan 0 */
    zref = (double *)malloc((2 * CN + 8) * sizeof(double));
    vfft_execute(pl[0], VFFT_FORWARD, x[0], NULL, zref, NULL);

    for (t = 0; t < NT; t++)
        th[t] = CreateThread(NULL, 0, runner, &a[t], 0, NULL);
    WaitForMultipleObjects(NT, th, TRUE, INFINITE);
    for (t = 0; t < NT; t++) CloseHandle(th[t]);

    for (t = 0; t < NT; t++) {
        int bad = memcmp(z[t], zref, 2 * CN * sizeof(double)) != 0;
        printf("thread %d result vs serial ref : %s\n", t,
               bad ? "*** MISMATCH ***" : "bitwise identical");
    }
    printf("pool at end                   : %d\n", vfft_get_num_threads());
    for (t = 0; t < NT; t++) vfft_destroy(pl[t]);
    return 0;
}
