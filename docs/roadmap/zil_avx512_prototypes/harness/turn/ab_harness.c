/* ab_harness.c -- bitwise A/B of an AVX-512 turned codelet against its AVX2
 * twin from the SAME generator (identical DAG => identical per-complex-lane
 * arithmetic => bitwise-identical outputs expected).
 *
 * -DF2=<avx2 symbol> -DF5=<avx512 symbol> -DRAD=<radix>
 * -DCLS=0 turned contiguous (n1t / t2t)     zout[2*(k*OLs + l)]
 *      =1 turned leg-strided (t2tg)          zout[2*(k*OLs + l*OGs)]
 *      =2 row loop turned (n1tr / t2tr)      rows x Ls lanes, pitch Gs / OGs
 * -DTW=1 when the kernel streams a VTW2 table (t2* kinds).
 *
 * Tables: record (group g, leg l) at (g*(R-1) + l-1) * 2*VW,
 * [c x VW][sign-folded s x VW], lane j = column k % per of group k / per —
 * each kernel gets a table in ITS OWN geometry (VW = 4 / 8) built from the
 * SAME per-(leg, column) (c, s), with a CEIL group count.
 * Output buffers end flush against a PROT_NONE page (overrun => SIGSEGV).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include <sys/mman.h>
#include <unistd.h>

#define S_(x) #x
#define S(x) S_(x)
typedef void (*fn11)(const double *, const double *, double *, double *,
                     const double *, const double *, size_t, size_t, size_t, size_t, size_t);
extern void F2(const double *, const double *, double *, double *,
               const double *, const double *, size_t, size_t, size_t, size_t, size_t);
extern void F5(const double *, const double *, double *, double *,
               const double *, const double *, size_t, size_t, size_t, size_t, size_t);

static uint64_t rs = 0x9E3779B97F4A7C15ull;
static double rnd(void) { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (double)(rs >> 11) / 9007199254740992.0 - 0.5; }

static double *table(int R, size_t cols, int VW, const double *cs /* [R][cols][2] */)
{
    const int per = VW / 2;
    size_t ng = (cols + per - 1) / per;
    size_t n = ng * (size_t)(R - 1) * 2 * VW;
    double *t = aligned_alloc(64, ((n ? n : 1) * sizeof(double) + 63) / 64 * 64);
    for (size_t i = 0; i < n; i++) t[i] = NAN;          /* unread lanes stay NaN */
    for (size_t k = 0; k < ng * per; k++)
        for (int l = 1; l < R; l++) {
            size_t g = k / per, j = k % per;
            double *rec = t + (g * (R - 1) + (l - 1)) * 2 * VW;
            size_t kk = k < cols ? k : cols - 1;        /* pad lanes: valid values */
            double c = cs[((size_t)l * cols + kk) * 2], s = cs[((size_t)l * cols + kk) * 2 + 1];
            rec[2 * j] = c; rec[2 * j + 1] = c;
            rec[VW + 2 * j] = -s; rec[VW + 2 * j + 1] = s;
        }
    return t;
}

static double *guarded(size_t ndbl, void **base, size_t *mlen)
{
    long pg = sysconf(_SC_PAGESIZE);
    size_t bytes = ndbl * sizeof(double);
    size_t npg = (bytes + pg - 1) / pg; if (!npg) npg = 1;
    *mlen = (npg + 1) * pg;
    char *m = mmap(0, *mlen, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    mprotect(m + npg * pg, pg, PROT_NONE);
    *base = m;
    return (double *)(m + npg * pg - bytes);
}

int main(void)
{
    const int R = RAD;
    long cases = 0, fails = 0, tol_cells = 0; (void)tol_cells;
    const size_t offs[] = { 0, 16, 32, 48 };
    for (size_t count = 1; count <= 13; count++)
    for (int oi = 0; oi < 3; oi++)
    for (int ai = 0; ai < 4; ai++) {
        size_t Ls, Gs = 0, OLs, OGs = 0, cnt, rows = 1, lanes;
        if (CLS == 2) {                       /* rows x lanes */
            lanes = count; rows = 3;
            Ls = lanes; Gs = (size_t)R * Ls + (size_t)ai;       /* input row pitch */
            OLs = (size_t)R + (size_t)oi;                         /* turn stride */
            OGs = lanes * OLs + (size_t)oi;                       /* output row pitch */
            cnt = rows * lanes;
        } else {
            lanes = count;
            Ls = count + (size_t)ai;
            cnt = count;
            if (CLS == 3) { OLs = count + (size_t)oi; }   /* leg-major (plain t2 / n1) */
            else if (CLS == 0) { OLs = (size_t)R + (size_t)(oi == 0 ? 0 : oi == 1 ? 3 : 8); }
            else { OGs = 2 + (size_t)oi; OLs = (size_t)R * OGs + 1; }
        }
        /* input */
        size_t nin = CLS == 2 ? 2 * (rows * Gs + 8) : 2 * ((size_t)R * Ls + 8);
        double *ib = aligned_alloc(64, ((nin + 8) * sizeof(double) + 63) / 64 * 64);
        double *zin = (double *)((char *)ib + offs[ai]);
        for (size_t i = 0; i < nin; i++) zin[i] = rnd();
        /* output extent */
        size_t nout;
        if (CLS == 3) nout = 2 * ((size_t)(R - 1) * OLs + lanes);
        else if (CLS == 0) nout = 2 * ((lanes - 1) * OLs + (size_t)R);
        else if (CLS == 1) nout = 2 * ((lanes - 1) * OLs + (size_t)(R - 1) * OGs + 1);
        else nout = 2 * ((rows - 1) * OGs + (lanes - 1) * OLs + (size_t)R);
        void *b2, *b5; size_t m2, m5;
        double *o2 = guarded(nout, &b2, &m2), *o5 = guarded(nout, &b5, &m5);
        for (size_t i = 0; i < nout; i++) { o2[i] = -777.0; o5[i] = -777.0; }
        double *t2 = 0, *t5 = 0;
#if TW
        double *cs = malloc((size_t)R * lanes * 2 * sizeof(double));
        for (size_t i = 0; i < (size_t)R * lanes * 2; i++) cs[i] = rnd() * 2.0;
        t2 = table(R, lanes, 4, cs);
        t5 = table(R, lanes, 8, cs);
        free(cs);
#endif
        F2(zin, 0, o2, 0, t2, 0, Ls, Gs, OLs, OGs, cnt);
        F5(zin, 0, o5, 0, t5, 0, Ls, Gs, OLs, OGs, cnt);
        cases++;
        int bad = memcmp(o2, o5, nout * sizeof(double)) != 0;
#if BLK
        /* BLOCKED kinds: the wide body runs the blocked construction, the
           narrow tail the monolithic one (c2c_il.ml tail note: they differ at
           ~1e-16). Columns in [2*floor(c/2), 4*floor(c/4)) are wide at avx2
           but tail at avx512, so only count % 4 in {0,1} is a same-body
           (bitwise) cell; the others are checked to 64 ulp-of-max instead. */
        if (bad && (count % 4 == 2 || count % 4 == 3)) {
            double mx = 0, md = 0;
            for (size_t i = 0; i < nout; i++) { double a = fabs(o2[i]); if (a > mx) mx = a; }
            for (size_t i = 0; i < nout; i++) { double d = fabs(o2[i] - o5[i]); if (!(d <= md)) md = d; }
            if (md <= 64.0 * 2.220446049250313e-16 * mx) { bad = 0; tol_cells++; }
        }
#endif
        /* also: every slot the contract names must have been written */
        if (!bad) {
            for (size_t r_ = 0; r_ < rows && !bad; r_++)
                for (size_t k = 0; k < lanes && !bad; k++)
                    for (int l = 0; l < R; l++) {
                        size_t idx = CLS == 3 ? 2 * ((size_t)l * OLs + k)
                                   : CLS == 0 ? 2 * (k * OLs + l)
                                   : CLS == 1 ? 2 * (k * OLs + (size_t)l * OGs)
                                              : 2 * (r_ * OGs + k * OLs + l);
                        if (o5[idx] == -777.0 || o5[idx + 1] == -777.0 || isnan(o5[idx])) { bad = 2; break; }
                    }
        }
        if (bad) {
            fails++;
            if (fails <= 5) {
                double md = 0;
                for (size_t i = 0; i < nout; i++) { double d = fabs(o2[i] - o5[i]); if (d > md || isnan(d)) md = d; }
                printf("FAIL(%d) R=%d count=%zu oi=%d ai=%d maxdiff=%g\n", bad, R, count, oi, ai, md);
            }
        }
        munmap(b2, m2); munmap(b5, m5); free(ib); free(t2); free(t5);
    }
    printf("%s R=%d cls=%d tw=%d blk=%d: cases=%ld fails=%ld tol_cells=%ld\n", S(F5), R, CLS, TW, BLK, cases, fails, tol_cells);
    return fails != 0;
}
