/* tail-policy bench: two-stage odd-N transforms (n1 then t2) and K=1 solos,
 * per avx512 tail policy + the avx2 kernels (real AVX2 build), alternating arms. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sched.h>
typedef void (*kfn)(const double *, const double *, double *, double *, const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);
#define POLS(X) X(narrowfix) X(ladder) X(masked) X(ladder_m3)
#define NR(X) X(43) X(37) X(41) X(8) X(16) X(32) X(64)
#define TR(X) X(23) X(27) X(25) X(29)
#define DN(R, P) void radix##R##_z_n1_fwd_avx512_##P(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#define DT(R, P) void radix##R##_z_t2_fwd_avx512_##P(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#define DP(P) NR_##P TR_##P
#define N_narrowfix(R) DN(R, narrowfix)
#define N_ladder(R) DN(R, ladder)
#define N_masked(R) DN(R, masked)
#define N_ladder_m3(R) DN(R, ladder_m3)
#define T_narrowfix(R) DT(R, narrowfix)
#define T_ladder(R) DT(R, ladder)
#define T_masked(R) DT(R, masked)
#define T_ladder_m3(R) DT(R, ladder_m3)
NR(N_narrowfix) NR(N_ladder) NR(N_masked) NR(N_ladder_m3)
TR(T_narrowfix) TR(T_ladder) TR(T_masked) TR(T_ladder_m3)
#define D2N(R) void radix##R##_z_n1_fwd_avx2(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
#define D2T(R) void radix##R##_z_t2_fwd_avx2(const double *, const double *, double *, double *, const double *, const double *, size_t, size_t, size_t, size_t, size_t);
NR(D2N) TR(D2T)

enum { NARMS = 5 };
static const char *armname[NARMS] = { "narrowfix", "ladder", "masked", "ladder_m3", "avx2" };
static kfn n1fn(int arm, int R)
{
#define CN(RR) case RR: return arm == 0 ? radix##RR##_z_n1_fwd_avx512_narrowfix : arm == 1 ? radix##RR##_z_n1_fwd_avx512_ladder : arm == 2 ? radix##RR##_z_n1_fwd_avx512_masked : arm == 3 ? radix##RR##_z_n1_fwd_avx512_ladder_m3 : radix##RR##_z_n1_fwd_avx2;
    switch (R) { NR(CN) }
    return 0;
}
static kfn t2fn(int arm, int R)
{
#define CT(RR) case RR: return arm == 0 ? radix##RR##_z_t2_fwd_avx512_narrowfix : arm == 1 ? radix##RR##_z_t2_fwd_avx512_ladder : arm == 2 ? radix##RR##_z_t2_fwd_avx512_masked : arm == 3 ? radix##RR##_z_t2_fwd_avx512_ladder_m3 : radix##RR##_z_t2_fwd_avx2;
    switch (R) { TR(CT) }
    return 0;
}
/* VTW2 stream for count columns: group g of cpv columns, leg l: record of 2*vw doubles
 * [c,c per column][-s,+s per column], from ONE logical twiddle per (leg, column) */
static double *vtw2(int R, int count, int vw, int N)
{
    const int cpv = vw / 2, groups = (count + cpv - 1) / cpv;
    double *t = aligned_alloc(64, sizeof(double) * ((size_t)groups * (R - 1) * 2 * vw + 64));
    for (int g = 0; g < groups; g++)
        for (int l = 1; l < R; l++) {
            double *rec = t + ((size_t)g * (R - 1) + (l - 1)) * 2 * vw;
            for (int j = 0; j < cpv; j++) {
                int k = g * cpv + j; if (k >= count) k = count - 1;
                double a = -2.0 * M_PI * l * k / N, c = cos(a), s = sin(a);
                rec[2 * j] = rec[2 * j + 1] = c; rec[vw + 2 * j] = -s; rec[vw + 2 * j + 1] = s;
            }
        }
    return t;
}
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e9 + t.tv_nsec; }
static int cmpd(const void *a, const void *b) { double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

typedef struct { const char *name; int N, RA, cA, RB, cB; } cs_t;   /* stage A: radix RA n1 count cA; B: radix RB t2 count cB */
int main(void)
{
    cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(2, &cs); sched_setaffinity(0, sizeof cs, &cs);
    static const cs_t C[] = {
        { "N=989  (43 x 23)", 989, 43, 23, 23, 43 }, { "N=999  (37 x 27)", 999, 37, 27, 27, 37 },
        { "N=1025 (41 x 25)", 1025, 41, 25, 25, 41 }, { "N=1073 (37 x 29)", 1073, 37, 29, 29, 37 },
        { "solo N=8   (n1, count 1)", 8, 8, 1, 0, 0 }, { "solo N=16  (n1, count 1)", 16, 16, 1, 0, 0 },
        { "solo N=32  (n1, count 1)", 32, 32, 1, 0, 0 }, { "solo N=64  (n1, count 1)", 64, 64, 1, 0, 0 },
    };
    const int ROUNDS = 21;
    printf("%-26s %-6s %-6s", "case", "remA", "remB");
    for (int a = 0; a < NARMS; a++) printf(" %12s", armname[a]);
    printf("   bitwise(all avx512 = avx2)\n");
    for (size_t ci = 0; ci < sizeof C / sizeof *C; ci++) {
        const cs_t *c = &C[ci]; const int N = c->N;
        double *x = aligned_alloc(64, 16 * N + 256), *y = aligned_alloc(64, 16 * N + 256), *z = aligned_alloc(64, 16 * N + 256);
        double *ref = malloc(16 * N), *out = malloc(16 * N);
        for (int i = 0; i < 2 * N; i++) x[i] = sin(0.37 * i) + 0.25 * cos(1.3 * i);
        double *tw[NARMS];
        for (int a = 0; a < NARMS; a++) tw[a] = c->RB ? vtw2(c->RB, c->cB, a == 4 ? 4 : 8, N) : 0;
        /* stage A: n1 radix RA over cA columns (Ls = OLs = cA); B: t2 radix RB over cB columns */
        #define RUN(a) do { n1fn(a, c->RA)(x, 0, y, 0, 0, 0, c->cA, 0, c->cA, 0, c->cA); \
                            if (c->RB) t2fn(a, c->RB)(y, 0, z, 0, tw[a], 0, c->cB, 0, c->cB, 0, c->cB); } while (0)
        int same = 1;
        for (int a = 0; a < NARMS; a++) {
            memset(y, 0, 16 * N); memset(z, 0, 16 * N); RUN(a);
            memcpy(out, c->RB ? z : y, 16 * N);
            if (a == 0) memcpy(ref, out, 16 * N); else if (memcmp(ref, out, 16 * N)) same = 0;
        }
        const int reps = (int)(3e6 / (N * 20.0)) + 50;
        double t[NARMS][64];
        for (int r = 0; r < ROUNDS; r++)
            for (int k = 0; k < NARMS; k++) {
                int a = (r & 1) ? NARMS - 1 - k : k; double best = 1e30;
                for (int b = 0; b < 5; b++) { double t0 = now(); for (int i = 0; i < reps; i++) RUN(a); double tt = (now() - t0) / reps; if (tt < best) best = tt; }
                t[a][r] = best;
            }
        printf("%-26s %-6d %-6d", c->name, c->cA % 4, c->RB ? c->cB % 4 : -1);
        for (int a = 0; a < NARMS; a++) { qsort(t[a], ROUNDS, sizeof(double), cmpd); printf(" %9.1f ns", t[a][ROUNDS / 2]); }
        printf("   %s\n", same ? "yes" : "NO");
        free(x); free(y); free(z); free(ref); free(out);
        for (int a = 0; a < NARMS; a++) free(tw[a]);
    }
    return 0;
}
