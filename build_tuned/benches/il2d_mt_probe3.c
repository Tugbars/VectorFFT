/* il2d_mt_probe3.c — READER D probe #3: isolate WHY concurrent vfft_create()
 * of 2D real plans crashes.  arm 1 = pool left at 8 (creates tear it down
 * concurrently); arm 2 = pool pre-set to 1 (stride_set_num_threads(1) then
 * early-returns inside every child create).  Each arm does R rounds of T
 * concurrent creates; a crash kills the process (detect by exit code).
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

int main(int argc, char **argv)
{
    int arm = argc > 1 ? atoi(argv[1]) : 1;
    int rounds = argc > 2 ? atoi(argv[2]) : 10;
    HANDLE th[NT];
    int r, t;

    memset(&g_c, 0, sizeof g_c);
    g_c.transform = VFFT_R2C;
    g_c.placement = VFFT_OUTOFPLACE;
    g_c.dims = 2; g_c.n[0] = N1; g_c.n[1] = N2;
    g_c.howmany = 1;
    g_c.layout = VFFT_LAYOUT_INTERLEAVED;
    g_c.nthreads = 0;
    g_c.wisdom_write = 0;
    if (arm == 3) g_c.recalibrate = 1;   /* force the create-time race + in-memory bank */
    if (arm == 4) { g_c.n[0] = 128; g_c.n[1] = 128; } /* cold cell: race + bank on miss */

    /* warm the wisdom singleton + every lazy registry ON THIS THREAD first.
     * For arms 3/4 warm with a NON-racing config so the singleton exists but
     * the racing/banking still happens on the worker threads. */
    { vfft_config_t w0 = g_c; w0.recalibrate = 0;
      vfft_plan w = vfft_create(&w0); if (w) vfft_destroy(w); }

    for (r = 0; r < rounds; r++) {
        vfft_set_num_threads(arm == 1 ? 8 : 1);
        for (t = 0; t < NT; t++) g_pl[t] = NULL;
        for (t = 0; t < NT; t++)
            th[t] = CreateThread(NULL, 0, creator, (LPVOID)(intptr_t)t, 0, NULL);
        WaitForMultipleObjects(NT, th, TRUE, INFINITE);
        for (t = 0; t < NT; t++) {
            CloseHandle(th[t]);
            if (!g_pl[t]) { printf("arm %d round %d: plan[%d] NULL\n", arm, r, t); }
            else vfft_destroy(g_pl[t]);
        }
        printf("arm %d round %d OK (pool now %d)\n", arm, r,
               vfft_get_num_threads());
        fflush(stdout);
    }
    printf("arm %d SURVIVED %d rounds\n", arm, rounds);
    return 0;
}
