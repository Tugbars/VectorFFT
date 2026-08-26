/* il2d_mt_probe2.c — READER D probe #2.
 * Q4: does executing an IL 2D real plan destroy a pool the app set up?
 * Q5: is CONCURRENT vfft_create() of 2D real plans safe (S2 create-side)?
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"

#define NT 4
#define N1 256
#define N2 256

static vfft_config_t g_c;
static vfft_plan g_pl[NT];

static DWORD WINAPI creator(LPVOID v)
{
    int t = (int)(intptr_t)v;
    g_pl[t] = vfft_create(&g_c);
    return 0;
}

int main(void)
{
    const size_t RN = (size_t)N1 * N2, hp1 = N2 / 2 + 1;
    const size_t CN = (size_t)N1 * hp1;
    double *x = (double *)malloc(RN * sizeof(double));
    double *z = (double *)malloc((2 * CN + 8) * sizeof(double));
    HANDLE th[NT];
    size_t i; int t;
    vfft_plan p;

    for (i = 0; i < RN; i++) x[i] = 1.0 + 1e-6 * (double)(i & 1023);

    memset(&g_c, 0, sizeof g_c);
    g_c.transform = VFFT_R2C;
    g_c.placement = VFFT_OUTOFPLACE;
    g_c.dims = 2; g_c.n[0] = N1; g_c.n[1] = N2;
    g_c.howmany = 1;
    g_c.layout = VFFT_LAYOUT_INTERLEAVED;
    g_c.nthreads = 0;              /* "use the current pool" */
    g_c.wisdom_write = 0;

    /* ---- Q4: pool survival across a 2D real execute ---- */
    p = vfft_create(&g_c);
    if (!p) { printf("create failed\n"); return 1; }
    vfft_set_num_threads(8);
    printf("Q4 pool before execute        : %d\n", vfft_get_num_threads());
    vfft_execute(p, VFFT_FORWARD, x, NULL, z, NULL);
    printf("Q4 pool AFTER  execute        : %d   <== app pool survives?\n",
           vfft_get_num_threads());
    vfft_destroy(p);

    /* ---- Q5: concurrent creates ---- */
    vfft_set_num_threads(8);
    printf("Q5 launching %d concurrent vfft_create()...\n", NT);
    fflush(stdout);
    for (t = 0; t < NT; t++)
        th[t] = CreateThread(NULL, 0, creator, (LPVOID)(intptr_t)t, 0, NULL);
    if (WaitForMultipleObjects(NT, th, TRUE, 30000) == WAIT_TIMEOUT)
        printf("Q5 *** TIMEOUT — concurrent create hung ***\n");
    else {
        for (t = 0; t < NT; t++)
            printf("Q5   plan[%d] = %p %s\n", t, (void *)g_pl[t],
                   g_pl[t] ? "" : "  <== NULL");
        printf("Q5 survived (no crash); pool now %d\n", vfft_get_num_threads());
    }
    for (t = 0; t < NT; t++) { CloseHandle(th[t]); if (g_pl[t]) vfft_destroy(g_pl[t]); }
    printf("done\n");
    return 0;
}
