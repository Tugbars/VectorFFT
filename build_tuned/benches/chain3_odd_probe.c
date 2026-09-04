/* CHAIN3 ODD-LEGAL (2026-09-04): the three-stage K=1 IL chain at all-odd
 * factorizations, built DIRECTLY (vfft_il3p_create) and checked against
 * a naive DFT (natural indices; scrambled search as the diagnostic),
 * fwd->bwd roundtrip and the DC identity. Even chains as the control
 * (their tables must be unchanged). No timing here — correctness first. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
#include "../../src/core/oop/il2p.h"

int main(void) {
    static const struct { int N, R2, A, B; const char *nm; } C[] = {
        { 384,  8, 3, 16, "even control 8x(3x16)" },
        { 960, 32, 3, 10, "even control 32x(3x10)" },
        { 405,  5, 9,  9, "odd 5x(9x9)" },
        { 255,  5, 3, 17, "odd 5x(3x17)" },
        { 1215, 15, 9,  9, "odd 15x(9x9)" },
        { 1215, 27, 5,  9, "odd 27x(5x9)" },
        { 1215,  9, 15, 9, "odd 9x(15x9)" },
        { 2187, 27, 9,  9, "odd 27x(9x9)" },
        { 3645, 27, 15, 9, "odd 27x(15x9)" },
        { 4095, 13, 15, 21, "odd 13x(15x21)" },
        { 4095, 21, 13, 15, "odd 21x(13x15)" },
        { 6561, 27, 27, 9, "odd 27x(27x9)" },
        { 6561,  9, 27, 27, "odd 9x(27x27)" },
        { 1050, 10, 5, 21, "mixed even-leaf 10x(5x21)" },
        { 945,  9, 5, 21, "odd 9x(5x21)" },
    };
    const int n = (int)(sizeof C / sizeof C[0]);
    int bad = 0;
    for (int i = 0; i < n; i++) {
        const int N = C[i].N;
        vfft_il3p_plan_t *p = vfft_il3p_create(N, C[i].R2, C[i].A, C[i].B);
        if (!p) { printf("%-6d %-28s REFUSED\n", N, C[i].nm); bad++; continue; }
        double *x = malloc(2*(size_t)N*8), *z = malloc(2*(size_t)N*8), *y = malloc(2*(size_t)N*8);
        double s0r = 0, s0i = 0;
        for (int j = 0; j < N; j++) { x[2*j] = (double)rand()/RAND_MAX-0.5; x[2*j+1] = (double)rand()/RAND_MAX-0.5; s0r += x[2*j]; s0i += x[2*j+1]; }
        vfft_il3p_execute_fwd(p, x, z);
        double dc = fabs(z[0]-s0r) + fabs(z[1]-s0i);
        double wn = 0, wa = 0;
        for (int t = 0; t < 5; t++) {
            const int k = (t*97+3) % N; double er = 0, ei = 0;
            for (int a = 0; a < N; a++) {
                double an = -2.0*3.14159265358979323846*(double)k*a/N;
                er += x[2*a]*cos(an) - x[2*a+1]*sin(an);
                ei += x[2*a]*sin(an) + x[2*a+1]*cos(an);
            }
            double d = fabs(z[2*k]-er) + fabs(z[2*k+1]-ei);
            if (d > wn) wn = d;
            double best = 1e300;
            for (int j = 0; j < N; j++) { double dd = fabs(z[2*j]-er)+fabs(z[2*j+1]-ei); if (dd < best) best = dd; }
            if (best > wa) wa = best;
        }
        vfft_il3p_execute_bwd(p, z, y);
        double rt = 0;
        for (int j = 0; j < 2*N; j++) { double d = fabs(y[j]/N - x[j]); if (d > rt) rt = d; }
        const int ok = (wa < 1e-7 && rt < 1e-9 && dc < 1e-8);
        if (!ok) bad++;
        printf("%-6d %-28s dft@nat %.1e  dft@any %.1e  rt %.1e  dc %.1e  %s\n",
               N, C[i].nm, wn, wa, rt, dc,
               ok ? (wn < 1e-7 ? "OK (natural)" : "OK (scrambled)") : "*** WRONG ***");
        vfft_il3p_destroy(p); free(x); free(z); free(y);
    }
    printf(bad ? "=== *** %d BAD *** ===\n" : "=== ALL OK (%d chains) ===\n", bad ? bad : n);
    return bad ? 1 : 0;
}
