/* scratch: NATURAL 2D IL c2c band widths, same run, alternated — the
 * natural axis race's verdict checked against its own candidates, plus
 * the served scrambled plan. Env pins (VFFT_IL2D_WL / VFFT_IL2D_TFUSE)
 * build the variants; env never writes wisdom. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include "vfft.h"

static double now_ns(void)
{
    static LARGE_INTEGER f; LARGE_INTEGER c;
    if (!f.QuadPart) QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return (double)c.QuadPart * 1e9 / (double)f.QuadPart;
}
static void env_set(const char *k, const char *v)
{
    static char slots[16][64];
    static int n = 0;
    char *s = slots[n++ & 15];
    snprintf(s, 64, "%s=%s", k, v ? v : "");
    putenv(s);
}
static vfft_plan mk2(vfft_wisdom *W, int N1, int N2, int nat)
{
    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_OUTOFPLACE; cfg.rigor = VFFT_MEASURE;
    cfg.dims = 2; cfg.n[0] = N1; cfg.n[1] = N2; cfg.howmany = 1;
    cfg.order = nat ? VFFT_ORDER_NATURAL : VFFT_ORDER_DEFAULT;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1;
    cfg.wisdom = W; cfg.wisdom_write = 0;
    return vfft_create(&cfg);
}
int main(int argc, char **argv)
{
    static const int C[][2] = { {243, 243}, {405, 405}, {1215, 243}, {729, 729} };
    /* the banked natural chains' stage spans L[1..]: 9.3.9 -> 27,9; 5.9.9 -> 81,9; 15.9.9 -> 81,9; 9.9.9 -> 81,9 */
    static const int SP[][3] = { {27, 9, 0}, {81, 9, 0}, {81, 9, 0}, {81, 9, 0} };
    const int nc = (int)(sizeof C / sizeof C[0]), R = 15;
    vfft_wisdom *W = vfft_wisdom_load(argc > 1 ? argv[1] : ".");
    if (!W) { printf("no wisdom\n"); return 2; }
    for (int i = 0; i < nc; i++) {
        const int N1 = C[i][0], N2 = C[i][1];
        const size_t n = (size_t)N1 * N2, nb = 2 * n * sizeof(double);
        double *x = _aligned_malloc(nb, 64), *y = _aligned_malloc(nb, 64);
        vfft_plan hs, hn, hv[6];
        char wls[6][8];
        int nv = 0, r, k;
        double t[8][16];
        for (size_t j = 0; j < 2 * n; j++) x[j] = (double)rand() / RAND_MAX - 0.5;
        env_set("VFFT_IL2D_WL", NULL); env_set("VFFT_IL2D_TFUSE", NULL);
        hs = mk2(W, N1, N2, 0);
        hn = mk2(W, N1, N2, 1);
        if (!hs || !hn) { printf("%dx%d create failed\n", N1, N2); continue; }
        printf("%dx%d\n", N1, N2);
        /* the natural variants: unbanded, and every stage span L[s] (s >= 1) with tfuse */
        {
            int cand[6], ncand = 0, s;
            cand[ncand++] = 0;
            for (s = 0; s < 3 && SP[i][s]; s++) cand[ncand++] = SP[i][s];
            for (k = 0; k < ncand; k++) {
                char v[16];
                snprintf(v, sizeof v, "%d", cand[k]);
                env_set("VFFT_IL2D_WL", v);
                env_set("VFFT_IL2D_TFUSE", "1");
                hv[nv] = mk2(W, N1, N2, 1);
                if (hv[nv]) { snprintf(wls[nv], 8, "wl%d", cand[k]); nv++; }
            }
            env_set("VFFT_IL2D_WL", NULL); env_set("VFFT_IL2D_TFUSE", NULL);
        }
        for (r = 0; r < R; r++) {
            const int narm = 2 + nv;
            for (k = 0; k < narm; k++) {
                const int arm = (k + r) % narm;
                double t0 = now_ns();
                if (arm == 0) vfft_execute(hs, VFFT_FORWARD, x, NULL, y, NULL);
                else if (arm == 1) vfft_execute(hn, VFFT_FORWARD, x, NULL, y, NULL);
                else vfft_execute(hv[arm - 2], VFFT_FORWARD, x, NULL, y, NULL);
                t[arm][r] = now_ns() - t0;
            }
        }
        {
            double mn[8];
            int a;
            for (a = 0; a < 2 + nv; a++) { mn[a] = 1e300; for (r = 0; r < R; r++) if (t[a][r] < mn[a]) mn[a] = t[a][r]; }
            printf("   scr served %8.0f | nat served %8.0f (%.3fx of scr)", mn[0], mn[1], mn[1] / mn[0]);
            for (a = 0; a < nv; a++) printf(" | nat %s %8.0f (%.3fx)", wls[a], mn[2 + a], mn[2 + a] / mn[0]);
            printf("\n");
        }
        vfft_destroy(hs); vfft_destroy(hn);
        for (k = 0; k < nv; k++) vfft_destroy(hv[k]);
        _aligned_free(x); _aligned_free(y);
    }
    vfft_wisdom_free(W);
    return 0;
}
