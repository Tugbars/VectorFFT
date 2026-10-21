/* rn1_gate.c -- the real mono kernels (radixN_z_rn1_{fwd,bwd} at the build's
 * ISA, the whole small real transform as one n1 body) against a naive real DFT,
 * fwd (r2c: N reals -> N/2+1 CCE bins) and bwd (c2r: N/2+1 bins -> N reals,
 * unnormalised), at N = 2..64; the 32 and 64 entries are the blocked bodies
 * rn1b48 / rn1b444 (2026-10-06). Each cell runs
 *   - counts 1..9 (element l of row k at zin[l*Ls + k], bin l of row k at
 *     zout[2*(l*OLs + k)]), each at the pitch Ls = OLs = count and count + 3:
 *     count 1 is the K=1 solo (the VEX-128 tail); the wide body packs two rows
 *     per vector at avx2 and four at avx512 (counts of 4 and more reach the
 *     512-bit code), and every leftover after it runs the tail. Each buffer
 *     lies between two guard bands (as long as the buffer, at least GUARD
 *     doubles); the output buffer starts as a canary (a quiet NaN no
 *     arithmetic makes), the input's gaps and bands as a second NaN: every
 *     bin or real a case owes is written, finite and right, every other double
 *     of the output buffer and of its bands still holds the canary, and the
 *     input buffer is as it was;
 *   - IN PLACE at count 1 (zin == zout, Ls = OLs = 1: zrm's call), banded the
 *     same way.
 * The kernels link as their own objects (each file declares its own static
 * constants).
 * The kernels are the build's ISA's (il_isa.h): each cell resolves through the
 * registry's RN1, RN1B48 and RN1B444 lists. An absent cell FAILS, except at
 * avx512 where the registry declares its kind absent (the kind's list defined
 * and empty: its DECLARED ABSENT block, Corpus.zil_isa_gap): reported, not
 * gated. At avx2 no kind is declared absent, so every absent cell FAILS.
 * Build: python gauntlet/build.py --compile --src src/tools/gates/rn1_gate.c (VFFT_ISA=avx512 for the avx512 kernels) */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include "il_isa.h"   /* the IL registry at the build's ISA: the kernels' declarations and radix lists */

typedef void (*kfn)(const double *, const double *, double *, double *, const double *, const double *,
                    size_t, size_t, size_t, size_t, size_t);

/* the kernel serving N in one direction at the build's ISA, 0 when the registry lists none */
static kfn rn1_fn(int n, int bwd)
{
    switch (n)
    {
#ifdef VFFT_IL_RN1_PAIR_RADICES
#define C(R) case R: return bwd ? VFFT_IL_SYM(radix##R##_z_rn1_bwd) : VFFT_IL_SYM(radix##R##_z_rn1_fwd);
    VFFT_IL_RN1_PAIR_RADICES(C)
#undef C
#endif
#ifdef VFFT_IL_RN1B48_PAIR_RADICES   /* the blocked body at 32 (4x8) */
#define C(R) case R: return bwd ? VFFT_IL_SYM(radix##R##_z_rn1b48_bwd) : VFFT_IL_SYM(radix##R##_z_rn1b48_fwd);
    VFFT_IL_RN1B48_PAIR_RADICES(C)
#undef C
#endif
#ifdef VFFT_IL_RN1B444_PAIR_RADICES   /* the blocked body at 64 (4x4x4) */
#define C(R) case R: return bwd ? VFFT_IL_SYM(radix##R##_z_rn1b444_bwd) : VFFT_IL_SYM(radix##R##_z_rn1b444_fwd);
    VFFT_IL_RN1B444_PAIR_RADICES(C)
#undef C
#endif
    default: return 0;
    }
}

/* the kind serving N is declared absent at the build's ISA: the registry defines
 * its list and leaves it empty (il_registry_<isa>.h: "every kind of the union
 * over ISAs is defined (empty = absent at this ISA)"); at avx2 the corpus is
 * whole (CMakeLists.txt's VFFT_ABSENT_avx2 is empty) and nothing is declared */
static int declared_absent(int n)
{
#if defined(VFFT_BUILD_ISA_AVX512)
#define C(R) + 1
    const int listed = n == 32 ? 0 VFFT_IL_RN1B48_PAIR_RADICES(C)
                     : n == 64 ? 0 VFFT_IL_RN1B444_PAIR_RADICES(C)
                     :           0 VFFT_IL_RN1_PAIR_RADICES(C);
#undef C
    return listed == 0;
#else
    (void)n;
    return 0;
#endif
}

static const int cells[] = {
    2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 29, 31,
    32, 37, 41, 43, 47, 53, 59, 61, 64,
};

#define MAXN 64                               /* the largest cell */
#define MAXC 9                                /* counts 1..MAXC, each at pitch count and count + 3 */
#define GUARD 64                              /* doubles: the least band on each side of a buffer */
#define CANARY_BITS 0x7ff8dead5eedbeefULL     /* the output buffer's prefill */
#define POISON_BITS 0x7ff80bad0bad0badULL     /* the input's gaps and bands */
#define MAXFAIL 4                             /* detail lines printed per cell */

static double urand(unsigned *s)
{
    *s = *s * 1664525u + 1013904223u;
    return (double)(*s >> 8) / (double)(1u << 24) - 0.5;
}
static void naive_r2c(const double *x, double *X, int n)
{
    for (int k = 0; k <= n / 2; k++)
    {
        double re = 0, im = 0;
        for (int j = 0; j < n; j++)
        {
            double a = -2.0 * 3.14159265358979323846 * (double)k * (double)j / (double)n;
            re += x[j] * cos(a); im += x[j] * sin(a);
        }
        X[2 * k] = re; X[2 * k + 1] = im;
    }
}
static void naive_c2r(const double *X, double *x, int n)
{
    for (int j = 0; j < n; j++)
    {
        double s = 0;
        for (int k = 0; k < n; k++)
        {
            double re, im;
            if (k <= n / 2) { re = X[2 * k]; im = X[2 * k + 1]; }
            else { re = X[2 * (n - k)]; im = -X[2 * (n - k) + 1]; }
            double a = 2.0 * 3.14159265358979323846 * (double)k * (double)j / (double)n;
            s += re * cos(a) - im * sin(a);
        }
        x[j] = s;
    }
}

/* ── banded buffers ── */
static void fill_bits(double *p, size_t n, uint64_t bits)
{
    for (size_t i = 0; i < n; i++) memcpy(&p[i], &bits, 8);
}
static int is_bits(double v, uint64_t bits) { uint64_t u; memcpy(&u, &v, 8); return u == bits; }
/* the band on each side of a buffer of nd doubles: as long as the buffer, at
 * least GUARD, whole 64-B lines (a store past either end by up to a leg, a
 * row or a doubled pitch lands in it) */
static size_t band_of(size_t nd)
{
    const size_t b = (nd + 7) & ~(size_t)7;
    return b > GUARD ? b : GUARD;
}
/* band + nd + band doubles, 64-B aligned, filled with bits; freed with _mm_free() */
static double *banded(size_t nd, size_t band, uint64_t bits)
{
    double *p = (double *)_mm_malloc((2 * band + nd) * sizeof(double), 64);
    if (!p) { fprintf(stderr, "rn1_gate: out of memory\n"); exit(3); }
    fill_bits(p, 2 * band + nd, bits);
    return p;
}

/* the references of one cell: MAXC rows of reals x, their CCE bins Xr (bin 0
 * and the Nyquist bin made real: the input a c2r is owed) and the unnormalised
 * c2r of those bins yr */
static double x[MAXC][MAXN], Xr[MAXC][MAXN + 2], yr[MAXC][MAXN];

/* one case: cnt rows at pitch P through f (bwd: the c2r), out of place. The
 * error relative to the reference's largest magnitude, or -1 with why filled
 * when a double outside the footprint, a band or the input moved, or an owed
 * double is unwritten or non-finite */
static double run_oop(kfn f, int bwd, int n, int cnt, size_t P, char *why, size_t whyn)
{
    const int h = n / 2;
    const size_t nin = bwd ? 2 * (size_t)(h + 1) * P : (size_t)n * P;
    const size_t nout = bwd ? (size_t)n * P : 2 * (size_t)(h + 1) * P;
    const size_t bi = band_of(nin), bo = band_of(nout);
    double *ib = banded(nin, bi, POISON_BITS), *ob = banded(nout, bo, CANARY_BITS);
    double *icopy = (double *)malloc((2 * bi + nin) * sizeof(double));
    double *in = ib + bi, *out = ob + bo, e = 0, sc = 0;
    size_t nunw = 0, nnan = 0, nstray = 0, nband = 0;
    int r, l;
    if (!icopy) { fprintf(stderr, "rn1_gate: out of memory\n"); exit(3); }
    for (r = 0; r < cnt; r++)
    {
        if (bwd)
            for (l = 0; l <= h; l++)
            {
                in[2 * ((size_t)l * P + r)] = Xr[r][2 * l];
                in[2 * ((size_t)l * P + r) + 1] = Xr[r][2 * l + 1];
            }
        else
            for (l = 0; l < n; l++) in[(size_t)l * P + r] = x[r][l];
    }
    memcpy(icopy, ib, (2 * bi + nin) * sizeof(double));
    f(in, 0, out, 0, 0, 0, P, 0, P, 0, (size_t)cnt);
    for (size_t j = 0; j < 2 * bo + nout; j++)
    {
        const int band = j < bo || j >= bo + nout;
        const size_t q = j - bo;   /* the output double; a slot q / 2 (fwd) or q (bwd) */
        const size_t s = bwd ? q : q / 2, m = s / P, k = s % P;
        const int owed = !band && k < (size_t)cnt && (bwd ? m < (size_t)n : m <= (size_t)h);
        if (!owed)
        {
            if (!is_bits(ob[j], CANARY_BITS)) { if (band) nband++; else nstray++; }
            continue;
        }
        const double ref = bwd ? yr[k][m] : Xr[k][2 * m + (q & 1)], v = ob[j];
        if (is_bits(v, CANARY_BITS)) { nunw++; continue; }
        if (!isfinite(v)) { nnan++; continue; }
        if (fabs(ref) > sc) sc = fabs(ref);
        if (fabs(v - ref) > e) e = fabs(v - ref);
    }
    const int wrote_in = memcmp(icopy, ib, (2 * bi + nin) * sizeof(double)) != 0;
    free(icopy);
    _mm_free(ib);
    _mm_free(ob);
    if (nunw || nnan || nstray || nband || wrote_in)
    {
        snprintf(why, whyn, "%zu unwritten, %zu non-finite, %zu doubles stored outside the footprint, %zu in "
                 "the bands%s", nunw, nnan, nstray, nband, wrote_in ? ", wrote its input" : "");
        return -1;
    }
    return sc > 0 ? e / sc : e;
}

/* IN PLACE at count 1 (zin == zout, Ls = OLs = 1): the buffer holds the larger
 * side (2h + 2 doubles >= N), banded with the canary. fwd: the N reals in, the
 * 2h + 2 doubles of the bins out; bwd: the bins in, the N reals out (the
 * doubles past N keep what they held). Error as run_oop */
static double run_inplace(kfn f, int bwd, int n, char *why, size_t whyn)
{
    const int h = n / 2;
    const size_t len = 2 * (size_t)(h + 1), b = band_of(len), nown = bwd ? (size_t)n : len;
    double *buf = banded(len, b, CANARY_BITS), *p = buf + b, e = 0, sc = 0;
    size_t nunw = 0, nnan = 0, nband = 0, j;
    if (bwd) memcpy(p, Xr[0], len * sizeof(double));
    else memcpy(p, x[0], (size_t)n * sizeof(double));
    f(p, 0, p, 0, 0, 0, 1, 0, 1, 0, 1);
    for (j = 0; j < 2 * b + len; j++)
    {
        if (j < b || j >= b + len) { if (!is_bits(buf[j], CANARY_BITS)) nband++; continue; }
        if (j - b >= nown) continue;
        const double ref = bwd ? yr[0][j - b] : Xr[0][j - b], v = buf[j];
        if (is_bits(v, CANARY_BITS)) { nunw++; continue; }
        if (!isfinite(v)) { nnan++; continue; }
        if (fabs(ref) > sc) sc = fabs(ref);
        if (fabs(v - ref) > e) e = fabs(v - ref);
    }
    _mm_free(buf);
    if (nunw || nnan || nband)
    {
        snprintf(why, whyn, "%zu unwritten, %zu non-finite, %zu doubles stored in the bands", nunw, nnan, nband);
        return -1;
    }
    return sc > 0 ? e / sc : e;
}

int main(void)
{
    const int ncells = (int)(sizeof cells / sizeof cells[0]);
    unsigned seed = 0x1357u;
    int fails = 0, nabsent = 0, runs = 0;
    for (int c = 0; c < ncells; c++)
    {
        const int n = cells[c], h = n / 2;
        const kfn f = rn1_fn(n, 0), b = rn1_fn(n, 1);
        double ef = 0, eb = 0, efi, ebi;
        int printed = 0, cfail = 0;
        char why[160];
        for (int r = 0; r < MAXC; r++) for (int i = 0; i < MAXN; i++) x[r][i] = urand(&seed);
        if (!f || !b)
        {
            if (!declared_absent(n))
            {   /* a cell the registry dropped from a kind it lists at this ISA */
                printf("N=%-3d absent at %s, not declared absent there  FAIL\n", n, VFFT_IL_ISA_NAME);
                fails++;
                continue;
            }
            printf("N=%-3d declared absent at %s: not gated\n", n, VFFT_IL_ISA_NAME);
            nabsent++;
            continue;
        }
        for (int r = 0; r < MAXC; r++) { naive_r2c(x[r], Xr[r], n); Xr[r][1] = 0; if (n % 2 == 0) Xr[r][2 * h + 1] = 0; naive_c2r(Xr[r], yr[r], n); }
        /* counts 1..MAXC, each at pitch count and count + 3, out of place */
        for (int cnt = 1; cnt <= MAXC; cnt++)
            for (int pv = 0; pv < 2; pv++)
                for (int bwd = 0; bwd < 2; bwd++)
                {
                    const size_t P = (size_t)cnt + (pv ? 3 : 0);
                    const double e = run_oop(bwd ? b : f, bwd, n, cnt, P, why, sizeof why);
                    runs++;
                    if (e >= 0 && e < 1e-13) { if (bwd) { if (e > eb) eb = e; } else if (e > ef) ef = e; continue; }
                    cfail = 1;
                    if (bwd) eb = INFINITY; else ef = INFINITY;
                    if (printed++ < MAXFAIL)
                    {
                        if (e < 0) printf("  N=%d %s count %d pitch %zu: %s\n", n, bwd ? "c2r" : "r2c", cnt, P, why);
                        else printf("  N=%d %s count %d pitch %zu: rel %.2e\n", n, bwd ? "c2r" : "r2c", cnt, P, e);
                    }
                }
        /* IN PLACE (zin == zout): the kind is alias-tolerant (no __restrict__, every load
         * of a loop body precedes every store); the real door's in-place placement */
        efi = run_inplace(f, 0, n, why, sizeof why);
        if (efi < 0 && printed++ < MAXFAIL) printf("  N=%d r2c in place: %s\n", n, why);
        ebi = run_inplace(b, 1, n, why, sizeof why);
        if (ebi < 0 && printed++ < MAXFAIL) printf("  N=%d c2r in place: %s\n", n, why);
        runs += 2;
        if (!(efi >= 0 && efi < 1e-13 && ebi >= 0 && ebi < 1e-13)) cfail = 1;
        if (cfail) fails++;
        printf("N=%-3d counts 1..%d at pitch count, count+3: r2c %.1e c2r %.1e | in place: r2c %.1e c2r %.1e  %s\n",
               n, MAXC, ef, eb, efi < 0 ? INFINITY : efi, ebi < 0 ? INFINITY : ebi, cfail ? "FAIL" : "PASS");
    }
    printf("%d kernel runs at %s\n", runs, VFFT_IL_ISA_NAME);
    if (nabsent)
        printf("%d of %d cells declared absent at %s: not gated\n", nabsent, ncells, VFFT_IL_ISA_NAME);
    if (nabsent == ncells)
        printf("no rn1 kernel at %s: nothing gated, every cell declared absent\n", VFFT_IL_ISA_NAME);
    else if (nabsent && !fails)
        printf("ALL PASS on the %d cells present at %s (%d declared absent, not gated)\n", ncells - nabsent, VFFT_IL_ISA_NAME, nabsent);
    else
        printf("%s\n", fails ? "FAILURES" : "ALL PASS");
    return fails ? 1 : 0;
}
