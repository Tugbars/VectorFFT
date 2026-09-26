/* twin2.c — TABLE-DRIVEN codelet twin gate (prototype of the proposed zil avx512 gate).
 *
 * One arm = (geometry class, table class, R, avx2 fn, avx512 fn). The two functions get
 * IDENTICAL data arguments; only the twiddle tables differ, each built for its own vector
 * width from ONE column-level / digit-level / group-level definition (the runtime's layout
 * contract, per VW). Every count in COUNTS is run; results are compared bitwise, then by
 * relative error (blocked forms legitimately differ at ~1e-16 where the avx2 bulk and the
 * avx512 tail cover the same column with different constructions). Canary + guard bands
 * catch unwritten outputs and out-of-region writes.
 *
 * geometry classes (derived from the gen_radix recipe by gate2.py):
 *   LEG    OOP leg-major           zin[2*(l*Ls+k)]           -> zout[2*(l*OLs+k)]
 *   LEGIP  in-place leg-major      same, zin == zout, Ls == OLs
 *   CS     column-stride in place  zin[2*(l*Ls+k*Gs)]        (Ls = 1, Gs = R+1)
 *   DIGIT  t2c: d-loop over OGs digits, zin += 2*Gs per digit, in place
 *   ROW    rowloop: rows = count/Ls rows of Ls lanes, in pitch Gs / out pitch OGs, OOP
 *   GN     t2csgn group-loop wrapper: zin_unused = obase[], Gs = groups
 * table classes: NONE, STREAM (per `per`-column group x (R-1) legs), DIGIT (per digit x
 *   (R-1) legs, broadcast), GEN2 (T1 per group + tw_im broadcast), GEN2N (T1 + tw_im per
 *   group broadcast, stride 2*VW). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

typedef void (*kfn)(const double *, const double *, double *, double *,
                    const double *, const double *, size_t, size_t, size_t, size_t, size_t);
enum { G_LEG, G_LEGIP, G_CS, G_DIGIT, G_ROW, G_GN };
enum { T_NONE, T_STREAM, T_DIGIT, T_GEN2, T_GEN2N };
typedef struct { const char *name; int geo, tab, R; kfn f2, f5; } arm_t;
#include "arms2.h"

#define GUARD 64
#define CANARY (-9.87654321e300)
static uint64_t lcg = 0x243F6A8885A308D3ull;
static double rnd(void) { lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
                          return ((double)(int64_t)(lcg >> 11)) / 4503599627370496.0; }

/* per-width builders from one definition */
static double *stream_tab(int legs, int vw, size_t cols, const double *col /*[cols+8][legs][4]*/)
{
    int per = vw / 2; size_t groups = (cols + per - 1) / per;
    size_t n = groups * (size_t)(legs - 1) * 2 * vw + 64;
    double *t = malloc(n * 8); for (size_t i = 0; i < n; i++) t[i] = rnd();
    for (size_t g = 0; g < groups; g++) for (int l = 1; l < legs; l++) {
        double *rec = t + (g * (legs - 1) + (l - 1)) * 2 * vw;
        for (int j = 0; j < per; j++) { const double *q = col + ((g * per + j) * legs + l) * 4;
            rec[2*j] = q[0]; rec[2*j+1] = q[1]; rec[vw+2*j] = q[2]; rec[vw+2*j+1] = q[3]; } }
    return t;
}
static void bcast(double *rec, int vw, const double *q)
{ for (int j = 0; j < vw / 2; j++) { rec[2*j] = q[0]; rec[2*j+1] = q[1]; rec[vw+2*j] = q[2]; rec[vw+2*j+1] = q[3]; } }

static int run(const arm_t *a, size_t count, double *worst_rel, int *exact)
{
    int R = a->R, D = 3, groups = 2, rows = 3;
    size_t Ls = 0, Gs = 0, OLs = 0, OGs = 0, nin = 0, nout = 0, cnt = count; int inplace = 0;
    size_t *obase = 0;
    switch (a->geo) {
    case G_LEG:   Ls = count + 3; OLs = count + 5; nin = 2*R*Ls; nout = 2*R*OLs; break;
    case G_LEGIP: Ls = OLs = count + 3; nin = nout = 2*R*Ls; inplace = 1; break;
    case G_CS:    Ls = OLs = 1; Gs = OGs = R + 1; nin = nout = 2*(count*Gs + R); inplace = 1; break;
    case G_DIGIT: Gs = count + 2; OGs = D; Ls = OLs = D*Gs + 1; nin = nout = 2*R*Ls; inplace = 1; break;
    case G_ROW:   Ls = OLs = count; Gs = R*Ls + 2; OGs = R*Ls + 3; cnt = rows * Ls;
                  nin = 2*(rows*Gs); nout = 2*(rows*OGs); break;
    case G_GN: {  Ls = 2; Gs = groups; OGs = Ls; OLs = count * Ls;
                  nin = 2*((size_t)groups*count*R*Ls); nout = 2*((size_t)groups*R*count*Ls);
                  obase = malloc(groups * count * sizeof(size_t));
                  for (int g = 0; g < groups; g++) for (size_t i = 0; i < count; i++) obase[g*count + i] = (size_t)g*R*count*Ls;
                  break; }
    }
    size_t cols = (a->geo == G_ROW) ? Ls : count;
    double *col = malloc((cols + 8) * R * 4 * 8); for (size_t i = 0; i < (cols + 8) * R * 4; i++) col[i] = rnd();
    double *dq = malloc((size_t)(D + groups) * R * 4 * 8); for (size_t i = 0; i < (size_t)(D + groups) * R * 4; i++) dq[i] = rnd();
    double *t2r = 0, *t5r = 0, *t2i = 0, *t5i = 0;
    if (a->tab == T_STREAM) { t2r = stream_tab(R, 4, cols, col); t5r = stream_tab(R, 8, cols, col); }
    if (a->tab == T_DIGIT) {
        size_t n2 = (size_t)D*(R-1)*8 + 64, n5 = (size_t)D*(R-1)*16 + 64;
        t2r = malloc(n2 * 8); t5r = malloc(n5 * 8);
        for (int d = 0; d < D; d++) for (int l = 1; l < R; l++) {
            bcast(t2r + ((size_t)d*(R-1) + l - 1) * 8, 4, dq + ((size_t)d*R + l) * 4);
            bcast(t5r + ((size_t)d*(R-1) + l - 1) * 16, 8, dq + ((size_t)d*R + l) * 4); }
    }
    if (a->tab == T_GEN2 || a->tab == T_GEN2N) {
        t2r = stream_tab(2, 4, cols, col); t5r = stream_tab(2, 8, cols, col);   /* legs=2 -> one record per group */
        int ng = a->tab == T_GEN2N ? groups : 1;
        t2i = malloc(ng * 8 * 8); t5i = malloc(ng * 16 * 8);
        for (int g = 0; g < ng; g++) { bcast(t2i + 8*g, 4, dq + (size_t)(D + g) * R * 4); bcast(t5i + 16*g, 8, dq + (size_t)(D + g) * R * 4); }
    }
    double *in = malloc(nin * 8); for (size_t i = 0; i < nin; i++) in[i] = rnd();
    size_t nreg = inplace ? nin : nout, nb = nreg + 2*GUARD;
    double *o2 = malloc(nb * 8), *o5 = malloc(nb * 8);
    for (size_t i = 0; i < nb; i++) o2[i] = o5[i] = CANARY;
    const double *zu = (const double *)obase;
    if (inplace) {
        memcpy(o2 + GUARD, in, nin * 8); memcpy(o5 + GUARD, in, nin * 8);
        a->f2(o2 + GUARD, zu, o2 + GUARD, 0, t2r, t2i, Ls, Gs, OLs, OGs, cnt);
        a->f5(o5 + GUARD, zu, o5 + GUARD, 0, t5r, t5i, Ls, Gs, OLs, OGs, cnt);
    } else {
        a->f2(in, zu, o2 + GUARD, 0, t2r, t2i, Ls, Gs, OLs, OGs, cnt);
        a->f5(in, zu, o5 + GUARD, 0, t5r, t5i, Ls, Gs, OLs, OGs, cnt);
    }
    size_t guard = 0, nd = 0, stale2 = 0, stale5 = 0; double w = 0, mag = 0;
    for (size_t i = 0; i < GUARD; i++) { if (o5[i] != CANARY) guard++; if (o5[nb - GUARD + i] != CANARY) guard++; }
    for (size_t i = 0; i < nreg; i++) {
        double x = o2[GUARD + i], y = o5[GUARD + i];
        if (x == CANARY) stale2++;
        if (y == CANARY) stale5++;
        if (memcmp(&x, &y, 8)) nd++;
        if (x != CANARY && y != CANARY) { if (fabs(x - y) > w) w = fabs(x - y); if (fabs(x) > mag) mag = fabs(x); }
    }
    double rel = mag > 0 ? w / mag : w;
    *exact = (nd == 0);
    if (rel > *worst_rel) *worst_rel = rel;
    int bad = guard || stale5 != stale2 || rel > 1e-13;
    free(col); free(dq); free(t2r); free(t5r); free(t2i); free(t5i); free(in); free(o2); free(o5); free(obase);
    return bad;
}

int main(int argc, char **argv)
{
    static const size_t COUNTS[] = { 1, 2, 3, 4, 5, 6, 7, 8, 9, 13, 37 };
    int nfail = 0, nexact = 0, nclose = 0;
    for (size_t i = 0; i < sizeof arms / sizeof arms[0]; i++) {
        char bad[128] = ""; int allexact = 1; double worst = 0;
        for (size_t c = 0; c < sizeof COUNTS / sizeof COUNTS[0]; c++) {
            int ex; if (run(&arms[i], COUNTS[c], &worst, &ex)) { char t[8]; snprintf(t, 8, " %zu", COUNTS[c]); strcat(bad, t); }
            allexact &= ex;
        }
        const char *v = bad[0] ? "FAIL" : allexact ? "EXACT" : "CLOSE";
        if (bad[0]) nfail++; else if (allexact) nexact++; else nclose++;
        if (bad[0] || !allexact || argc > 1)
            printf("%-5s %-44s worst_rel=%.2e%s%s\n", v, arms[i].name, worst, bad[0] ? " counts:" : "", bad);
    }
    printf("\n%zu arms: %d bitwise-EXACT, %d CLOSE (<=1e-13, blocked/tail construction), %d FAIL\n",
           sizeof arms / sizeof arms[0], nexact, nclose, nfail);
    return nfail != 0;
}
