/* NATURAL x MT (2026-09-04): natural-order 2D IL cells at T=8 vs T=1 —
 * MT == ST bitwise, engagement proven (vfft_il2d_col_mt_passes), the
 * spectrum checked against a naive DFT at NATURAL indices, speedup. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include "vfft.h"
static double now_ns(void){LARGE_INTEGER f,t;QueryPerformanceFrequency(&f);QueryPerformanceCounter(&t);return (double)t.QuadPart*1e9/(double)f.QuadPart;}
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const struct { int n1, n2, tr; } C[] = {
        { 256, 64, 0 }, { 512, 64, 0 }, { 63, 64, 0 }, { 1024, 128, 0 },
        { 256, 64, 1 }, { 512, 128, 1 }, { 63, 64, 1 }, { 1024, 256, 1 },
    };
    vfft_set_num_threads(8);
    for (int ci = 0; ci < 8; ci++) {
        const int N1 = C[ci].n1, N2 = C[ci].n2, re = C[ci].tr;
        const size_t hp1 = (size_t)N2/2 + 1;
        const size_t SN = re ? (size_t)N1*N2 : 2*(size_t)N1*N2;
        const size_t DN = re ? 2*(size_t)N1*hp1 : 2*(size_t)N1*N2;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = re ? VFFT_R2C : VFFT_C2C;
        c.placement = VFFT_OUTOFPLACE;
        c.rigor = VFFT_MEASURE; c.dims = 2; c.n[0] = N1; c.n[1] = N2;
        c.howmany = 1; c.order = VFFT_ORDER_NATURAL;
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        c.wisdom = W; c.wisdom_write = 0;
        c.nthreads = 1;
        vfft_plan ps = vfft_create(&c);
        c.nthreads = 8;
        vfft_plan pm = vfft_create(&c);
        if (!ps || !pm) { printf("%s %4dx%-4d NAT: create FAIL\n", re?"real":"c2c ", N1, N2); continue; }
        double *x = malloc(SN*8), *zs = calloc(DN+16, 8), *zm = calloc(DN+16, 8);
        double ts = 1e300, tm = 1e300, t0, dfte = 0;
        long p0, p1;
        for (size_t i = 0; i < SN; i++) x[i] = (double)rand()/RAND_MAX - 0.5;
        vfft_execute(ps, VFFT_FORWARD, x, NULL, zs, NULL);
        p0 = vfft_il2d_col_mt_passes();
        vfft_execute(pm, VFFT_FORWARD, x, NULL, zm, NULL);
        p1 = vfft_il2d_col_mt_passes();
        int bit = memcmp(zs, zm, DN*8) == 0;
        for (int tno = 0; tno < 4; tno++) {
            const int k1 = (tno*37+1) % N1, k2 = (tno*13) % (re ? (int)hp1 : N2);
            double er = 0, ei = 0;
            for (int a = 0; a < N1; a++) for (int b = 0; b < N2; b++) {
                double an = -2.0*3.14159265358979323846*((double)k1*a/N1 + (double)k2*b/N2);
                double xr = re ? x[(size_t)a*N2+b] : x[2*((size_t)a*N2+b)];
                double xi = re ? 0.0 : x[2*((size_t)a*N2+b)+1];
                er += xr*cos(an) - xi*sin(an);
                ei += xr*sin(an) + xi*cos(an);
            }
            const size_t w = re ? hp1 : (size_t)N2;
            double d = fabs(zm[2*((size_t)k1*w+k2)] - er) + fabs(zm[2*((size_t)k1*w+k2)+1] - ei);
            if (d > dfte) dfte = d;
        }
        for (int r = 0; r < 12; r++) {
            t0 = now_ns(); vfft_execute(ps, VFFT_FORWARD, x, NULL, zs, NULL); t0 = now_ns()-t0; if (t0<ts) ts=t0;
            t0 = now_ns(); vfft_execute(pm, VFFT_FORWARD, x, NULL, zm, NULL); t0 = now_ns()-t0; if (t0<tm) tm=t0;
        }
        /* bwd bitwise too (natural input) */
        double *ys = malloc(SN*8), *ym = malloc(SN*8);
        if (re) {
            c.transform = VFFT_C2R; c.nthreads = 1; vfft_plan bs = vfft_create(&c);
            c.nthreads = 8; vfft_plan bm = vfft_create(&c);
            int bb = 0;
            if (bs && bm) { vfft_execute(bs, VFFT_BACKWARD, zs, NULL, ys, NULL); vfft_execute(bm, VFFT_BACKWARD, zs, NULL, ym, NULL); bb = memcmp(ys, ym, SN*8) == 0; }
            printf("%s %4dx%-4d NAT T=8: ST %8.0f MT %8.0f = %.2fx  passes=%ld  fwd %s  bwd %s  dft %.1e %s\n",
                   "real", N1, N2, ts, tm, ts/tm, p1-p0, bit?"BITWISE":"*** DIFF ***", bb?"BITWISE":"*** DIFF ***", dfte, dfte < 1e-6 ? "OK" : "*** WRONG ***");
            if (bs) vfft_destroy(bs); if (bm) vfft_destroy(bm);
        } else {
            vfft_execute(ps, VFFT_BACKWARD, zs, NULL, ys, NULL);
            vfft_execute(pm, VFFT_BACKWARD, zs, NULL, ym, NULL);
            int bb = memcmp(ys, ym, SN*8) == 0;
            printf("%s %4dx%-4d NAT T=8: ST %8.0f MT %8.0f = %.2fx  passes=%ld  fwd %s  bwd %s  dft %.1e %s\n",
                   "c2c ", N1, N2, ts, tm, ts/tm, p1-p0, bit?"BITWISE":"*** DIFF ***", bb?"BITWISE":"*** DIFF ***", dfte, dfte < 1e-6 ? "OK" : "*** WRONG ***");
        }
        vfft_destroy(ps); vfft_destroy(pm);
        free(x); free(zs); free(zm); free(ys); free(ym);
    }
    vfft_set_num_threads(1);
    return 0;
}
