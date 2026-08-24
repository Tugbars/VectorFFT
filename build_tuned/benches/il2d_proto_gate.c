/* il2d_proto_gate.c — gate for the IL-2D column-pass stage maps (M0 of
 * docs/roadmap/fft2d_il_c2c_design.md; simulator = src/core/oop/il2d_proto.h).
 *
 * Proves, per cell and PER DIRECTION (roundtrip never gates a permuted
 * transform): sim[row_pos(ki)*N2 + kj] == naive2d[ki*N2 + kj] elementwise,
 * where naive2d is a separable naive DFT (natural both axes) and row_pos is
 * the simulator's own digit-reversal map — closing the algebra of the DIF
 * stage addressing, the (d,r)-only twiddle law, and the output permutation.
 *
 * Build: python build.py --src benches/il2d_proto_gate.c --compile
 * Run  : il2d_proto_gate.exe
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../../src/core/oop/il2d_proto.h"

/* naive 1D DFT along axis i (length N1, stride N2 complex), natural out */
static void naive_axis_i(const double *in, double *out, int N1, int N2,
                         double sgn)
{
    const double pi = 3.14159265358979323846;
    int k, i, j;
    for (k = 0; k < N1; k++)
        for (j = 0; j < N2; j++) {
            double sr = 0.0, si = 0.0;
            for (i = 0; i < N1; i++) {
                const double a = sgn * 2.0 * pi * (double)(k * i) / (double)N1;
                const double c = cos(a), s = sin(a);
                const size_t idx = 2 * ((size_t)i * N2 + j);
                sr += in[idx] * c - in[idx + 1] * s;
                si += in[idx] * s + in[idx + 1] * c;
            }
            out[2 * ((size_t)k * N2 + j)] = sr;
            out[2 * ((size_t)k * N2 + j) + 1] = si;
        }
}

/* naive 1D DFT along axis j (length N2, contiguous), natural out */
static void naive_axis_j(const double *in, double *out, int N1, int N2,
                         double sgn)
{
    const double pi = 3.14159265358979323846;
    int i, k, j;
    for (i = 0; i < N1; i++)
        for (k = 0; k < N2; k++) {
            double sr = 0.0, si = 0.0;
            for (j = 0; j < N2; j++) {
                const double a = sgn * 2.0 * pi * (double)(k * j) / (double)N2;
                const double c = cos(a), s = sin(a);
                const size_t idx = 2 * ((size_t)i * N2 + j);
                sr += in[idx] * c - in[idx + 1] * s;
                si += in[idx] * s + in[idx + 1] * c;
            }
            out[2 * ((size_t)i * N2 + k)] = sr;
            out[2 * ((size_t)i * N2 + k) + 1] = si;
        }
}

typedef struct {
    int N1, N2;
    int nf;
    int chain[IL2D_MAX_STAGES];
} cell_t;

static const cell_t CELLS[] = {
    /* single-stage (the M1 tier: shipped n1 kernels, Ls=N2, no emission) */
    { 4,    16, 1, { 4 } },
    { 8,     8, 1, { 8 } },
    { 16,   64, 1, { 16 } },
    { 32,    5, 1, { 32 } },        /* odd column count (count ANY >= 1) */
    { 64,  100, 1, { 64 } },
    /* multi-stage (the M2 t2c chains), incl. order variants per N1 */
    { 128,  32, 2, { 16, 8 } },
    { 128,  32, 2, { 8, 16 } },
    { 256,  16, 2, { 16, 16 } },
    { 256,  16, 2, { 4, 64 } },
    { 256,  16, 2, { 64, 4 } },
    { 512,  16, 2, { 8, 64 } },
    { 512,  16, 3, { 8, 8, 8 } },
    { 1024,  8, 2, { 16, 64 } },
    { 1024,  8, 3, { 4, 16, 16 } },
    { 4096,  8, 2, { 64, 64 } },
    /* aspect extremes */
    { 16, 1024, 1, { 16 } },
    { 1024, 16, 2, { 64, 16 } },
};

int main(void)
{
    int fails = 0, ci, dir;
    srand(12345);
    for (ci = 0; ci < (int)(sizeof CELLS / sizeof CELLS[0]); ci++) {
        const cell_t *c = &CELLS[ci];
        const size_t CN = (size_t)c->N1 * c->N2;
        double *x = malloc(2 * CN * sizeof(double));
        double *sim = malloc(2 * CN * sizeof(double));
        double *t1 = malloc(2 * CN * sizeof(double));
        double *ref = malloc(2 * CN * sizeof(double));
        if (!x || !sim || !t1 || !ref) { printf("OOM\n"); return 2; }
        for (size_t i = 0; i < 2 * CN; i++)
            x[i] = (double)rand() / RAND_MAX - 0.5;
        for (dir = 0; dir < 2; dir++) {
            const double sgn = dir ? 1.0 : -1.0;
            double maxref = 0.0, maxerr = 0.0;
            int ki, kj;
            memcpy(sim, x, 2 * CN * sizeof(double));
            if (il2d_sim_2d(sim, c->N1, c->N2, c->chain, c->nf, sgn)) {
                printf("  *** FAIL *** %dx%d: sim refused chain\n",
                       c->N1, c->N2);
                fails++;
                continue;
            }
            naive_axis_i(x, t1, c->N1, c->N2, sgn);
            naive_axis_j(t1, ref, c->N1, c->N2, sgn);
            for (ki = 0; ki < c->N1; ki++) {
                const int pi_ = il2d_sim_row_pos(ki, c->N1, c->chain, c->nf);
                for (kj = 0; kj < c->N2; kj++) {
                    const size_t ir = 2 * ((size_t)ki * c->N2 + kj);
                    const size_t is = 2 * ((size_t)pi_ * c->N2 + kj);
                    double d0 = fabs(sim[is] - ref[ir]);
                    double d1 = fabs(sim[is + 1] - ref[ir + 1]);
                    double m0 = fabs(ref[ir]), m1 = fabs(ref[ir + 1]);
                    if (m0 > maxref) maxref = m0;
                    if (m1 > maxref) maxref = m1;
                    if (d0 > maxerr) maxerr = d0;
                    if (d1 > maxerr) maxerr = d1;
                }
            }
            {
                const double rel = maxerr / (maxref > 0 ? maxref : 1.0);
                const int ok = rel < 1e-10;
                char ch[64];
                int s, off = 0;
                for (s = 0; s < c->nf; s++)
                    off += snprintf(ch + off, sizeof ch - off, "%s%d",
                                    s ? "." : "", c->chain[s]);
                printf("  %-9s %5dx%-5d chain %-10s rel %.2e  %s\n",
                       dir ? "bwd" : "fwd", c->N1, c->N2, ch, rel,
                       ok ? "PASS" : "*** FAIL ***");
                if (!ok) fails++;
            }
        }
        free(x); free(sim); free(t1); free(ref);
    }
    printf("\n%s (%d fail)\n", fails ? "*** FAIL ***" : "=== ALL PASS ===",
           fails);
    return fails ? 1 : 0;
}
