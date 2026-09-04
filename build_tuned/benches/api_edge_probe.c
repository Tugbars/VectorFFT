/* the two matrix edges the serve/refuse sweep cannot settle by itself:
 *   (1) 2D SPLIT cells the probe saw SERVED at prime / natural — are
 *       they CORRECT? (the split 2D real prime cell used to be silently
 *       wrong; it now refuses — this checks the c2c split siblings)
 *   (2) the IL prime band: what does a prime beyond the inner's
 *       self-validation do (refuse loudly, or serve)? */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
static void dft2(const double *re, const double *im, int N1, int N2, int k1, int k2, double *er, double *ei) {
    double sr = 0, si = 0;
    for (int a = 0; a < N1; a++) for (int b = 0; b < N2; b++) {
        double an = -2.0*3.14159265358979323846*((double)k1*a/N1 + (double)k2*b/N2);
        double xr = re[(size_t)a*N2+b], xi = im[(size_t)a*N2+b];
        sr += xr*cos(an) - xi*sin(an); si += xr*sin(an) + xi*cos(an);
    }
    *er = sr; *ei = si;
}
int main(int argc, char **argv) {
    vfft_wisdom *W = vfft_wisdom_load(argv[1]);
    static const struct { int n1, n2, order; const char *nm; } C[] = {
        { 127, 64, VFFT_ORDER_DEFAULT, "2D c2c SPLIT prime 127x64 DEFAULT" },
        { 256, 64, VFFT_ORDER_NATURAL, "2D c2c SPLIT 256x64 NATURAL" },
        { 63, 63, VFFT_ORDER_DEFAULT, "2D c2c SPLIT odd 63x63 DEFAULT" },
    };
    for (int ci = 0; ci < 3; ci++) {
        const int N1 = C[ci].n1, N2 = C[ci].n2; const size_t T = (size_t)N1*N2;
        vfft_config_t c; memset(&c, 0, sizeof c);
        c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE; c.layout = VFFT_LAYOUT_SPLIT;
        c.order = C[ci].order; c.rigor = VFFT_MEASURE; c.dims = 2; c.n[0] = N1; c.n[1] = N2;
        c.howmany = 1; c.wisdom = W; c.wisdom_write = 0;
        vfft_plan p = vfft_create(&c);
        if (!p) { printf("%-40s REFUSED\n", C[ci].nm); continue; }
        double *re = malloc(T*8), *im = malloc(T*8), *ore = malloc(T*8), *oim = malloc(T*8), *bre = malloc(T*8), *bim = malloc(T*8);
        for (size_t i = 0; i < T; i++) { re[i] = (double)rand()/RAND_MAX-0.5; im[i] = (double)rand()/RAND_MAX-0.5; }
        vfft_execute(p, VFFT_FORWARD, re, im, ore, oim);
        /* natural-index check at a few bins; if it fails, search (scrambled) */
        double worst_nat = 0, worst_any = 0;
        for (int t = 0; t < 4; t++) {
            int k1 = (t*37+1) % N1, k2 = (t*13+2) % N2; double er, ei;
            dft2(re, im, N1, N2, k1, k2, &er, &ei);
            double d = fabs(ore[(size_t)k1*N2+k2]-er) + fabs(oim[(size_t)k1*N2+k2]-ei);
            if (d > worst_nat) worst_nat = d;
            double best = 1e300;
            for (size_t j = 0; j < T; j++) { double dd = fabs(ore[j]-er)+fabs(oim[j]-ei); if (dd < best) best = dd; }
            if (best > worst_any) worst_any = best;
        }
        vfft_execute(p, VFFT_BACKWARD, ore, oim, bre, bim);
        double rt = 0;
        for (size_t i = 0; i < T; i++) { double d = fabs(bre[i]/(double)T-re[i]) + fabs(bim[i]/(double)T-im[i]); if (d > rt) rt = d; }
        printf("%-40s SERVED  dft@nat %.1e  dft@any %.1e  rt %.1e  %s\n", C[ci].nm, worst_nat, worst_any, rt,
               (worst_any < 1e-7 && rt < 1e-9) ? (worst_nat < 1e-7 ? "OK (natural)" : "OK (scrambled)") : "*** WRONG ***");
        vfft_destroy(p); free(re); free(im); free(ore); free(oim); free(bre); free(bim);
    }
    /* (2) the IL prime band edge */
    {
        static const int PR[] = { 8191, 16381, 32749 };
        for (int i = 0; i < 3; i++) {
            vfft_config_t c; memset(&c, 0, sizeof c);
            c.transform = VFFT_C2C; c.placement = VFFT_OUTOFPLACE; c.layout = VFFT_LAYOUT_INTERLEAVED;
            c.rigor = VFFT_MEASURE; c.dims = 1; c.n[0] = PR[i]; c.howmany = 1; c.wisdom = W; c.wisdom_write = 0;
            fprintf(stderr, "----- 1D c2c OOP IL prime %d\n", PR[i]);
            vfft_plan p = vfft_create(&c);
            printf("%-40s %s\n", i == 0 ? "1D c2c OOP IL prime 8191" : i == 1 ? "1D c2c OOP IL prime 16381" : "1D c2c OOP IL prime 32749", p ? "SERVED" : "REFUSED");
            if (p) vfft_destroy(p);
        }
    }
    return 0;
}
