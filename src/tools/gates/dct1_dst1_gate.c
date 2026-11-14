/* dct1_dst1_gate.c -- DCT-I and DST-I (VFFT_DCT1, VFFT_DST1; src/core/il/real/dct1_dst1.h) through the public
 * door, against the definitions of include/vfft.h summed directly in long double:
 *   DCT-I   Y[k] = x[0] + (-1)^k x[N-1] + 2 sum_{n=1..N-2} x[n] cos(pi n k / (N-1))
 *   DST-I   Y[k] = 2 sum_{n=0..N-1} x[n] sin(pi (n+1)(k+1) / (N+1))
 *
 * THE CELLS: every N to 300 (DCT-I from 2, DST-I from 1) and 28 larger lengths to 2^20 + 1 (powers of two and
 * the lengths one below and one above them, odd, prime, mixed), each transform out of place and in place, on
 * one thread (and on 8 threads, or this machine's cores when it has fewer, at every seventh N, at N below 20
 * and at the larger lengths), planned by formulas (VFFT_FLAG_ESTIMATE); and 14 lengths per transform planned
 * by measuring into the scratch store (--wisdir), the three longest also threaded.
 *
 * PER CELL:
 *   - the plan's forward and its backward (the same transform) against the reference: every output up to
 *     N = 2048, above it 48 outputs;
 *   - the round trip, backward of forward = 2 (N - 1) x for DCT-I and 2 (N + 1) x for DST-I, every output;
 *   - vfft_info: N doubles in, N doubles out;
 *   - an out-of-place plan called with out == in gives the two-buffer result bit for bit;
 *   - an in-place plan called with two buffers is refused and leaves the output alone.
 * THE REFUSALS: rank 2, howmany 2, DCT-I at N = 1 (each NULL with its reason).
 * THE STORE after the measured cells: wisdom2_real.txt holds one t=dct1 row (eng=dct, its real FFT under dct_,
 * an R2C at 2 (N - 1)) and one t=dst1 row (eng=dst, its real FFT under dst_, an R2C at 2 (N + 1)) per measured
 * length and thread count (place=oop: one row serves both placements), and no other row -- neither transform
 * banks an r2c or a c2r row; no other shard is written. THE REPLAY: the measured cells created again enter no
 * race scope, and are checked like the first time.
 * The limit is 2e-13 relative to the largest output.
 *
 * Build: python gauntlet/build.py --compile --vfft --src src/tools/gates/dct1_dst1_gate.c
 * Run:   dct1_dst1_gate --wisdir <scratch dir> [quick]      (the store is WRITTEN: the two transforms' rows) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"
#include "vfft_diagnostics.h"   /* vfft_measure_scopes: the race scopes entered */

#define TOL 2e-13
#define FULL_N 2048
#define NSAMP 48
static const long double PI = 3.141592653589793238462643383279502884L;
static unsigned g_seed = 20261114u;
static long g_cells, g_bad;
static double g_worst;

/* the two transforms: sine = 1 is DST-I */
static const char *xname(int sine) { return sine ? "DST1" : "DCT1"; }
static int xnmin(int sine) { return sine ? 1 : 2; }                   /* the shortest length */
static double xscale(int sine, int N) { return sine ? 2.0 * (N + 1) : 2.0 * (N - 1); }
static int xfftlen(int sine, int N) { return sine ? 2 * (N + 1) : 2 * (N - 1); }

static double urand(void)
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return (double)(g_seed >> 8) / (double)(1u << 24) - 0.5;
}
static double ref_at(const double *x, int N, int k, int sine)
{
    long double s = 0;
    if (sine)
    {
        const long long P = 2LL * (N + 1);
        for (int n = 0; n < N; n++)
            s += x[n] * sinl(PI * (long double)(((long long)(n + 1) * (k + 1)) % P) / (long double)(N + 1));
        return (double)(2.0L * s);
    }
    {
        const long long P = 2LL * (N - 1);
        for (int n = 1; n < N - 1; n++)
            s += x[n] * cosl(PI * (long double)(((long long)n * k) % P) / (long double)(N - 1));
        return (double)((long double)x[0] + ((k & 1) ? -1.0L : 1.0L) * x[N - 1] + 2.0L * s);
    }
}
static vfft_plan mk(int sine, int rank, int N, int K, int ip, int T, int est, const char *wis)
{
    vfft_config_t c;
    memset(&c, 0, sizeof c);
    c.transform = sine ? VFFT_DST1 : VFFT_DCT1;
    c.rank = rank; c.n[0] = N; c.n[1] = rank > 1 ? N : 0; c.howmany = (size_t)K; c.nthreads = T;
    c.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    c.flags = est ? VFFT_FLAG_ESTIMATE : 0;
    c.wisdom_dir = est ? NULL : wis;
    return vfft_create(&c);
}
static void verdict(const char *what, const char *cell, double e)
{
    g_cells++;
    if (e > g_worst) g_worst = e;
    if (!(e < TOL)) { g_bad++; printf("FAIL %s %s: rel %.2e\n", cell, what, e); }
}
static double amax(const double *a, int N)
{
    double m = 0;
    for (int i = 0; i < N; i++) if (fabs(a[i]) > m) m = fabs(a[i]);
    return m;
}

/* the reference of one length and transform: the checked output indices and the transform of x at them */
typedef struct { int N, sine, cnt, *at; double *x, *r; } ref_t;
static void ref_make(ref_t *r, int N, int sine)
{
    r->N = N;
    r->sine = sine;
    r->cnt = N <= FULL_N ? N : NSAMP;
    r->at = malloc((size_t)r->cnt * sizeof(int));
    r->x = vfft_malloc((size_t)N * 8);
    r->r = malloc((size_t)r->cnt * 8);
    for (int i = 0; i < N; i++) r->x[i] = urand();
    for (int i = 0; i < r->cnt; i++)
    {
        r->at[i] = N <= FULL_N ? i : (int)(((long long)i * 2654435761u + 17) % N);
        r->r[i] = ref_at(r->x, N, r->at[i], sine);
    }
}
static void ref_free(ref_t *r)
{
    free(r->at); vfft_free(r->x); free(r->r);
}
static double ref_err(const ref_t *r, const double *got)
{
    double e = 0, sc = amax(got, r->N);
    for (int i = 0; i < r->cnt; i++)
        if (fabs(got[r->at[i]] - r->r[i]) > e) e = fabs(got[r->at[i]] - r->r[i]);
    return sc > 0 ? e / sc : e;
}

static void one(const ref_t *r, int T, int est, const char *wis)
{
    const int N = r->N, sine = r->sine;
    const double *x = r->x, sc = xscale(sine, N);
    double *y = vfft_malloc((size_t)N * 8), *z = vfft_malloc((size_t)N * 8);
    char cell[96];
    for (int ip = 0; ip < 2; ip++)
    {
        vfft_plan p = mk(sine, 1, N, 1, ip, T, est, wis);
        vfft_info_t info;
        snprintf(cell, sizeof cell, "N %d %s plan %s T %d %s", N, xname(sine), ip ? "ip" : "oop", T, est ? "estimate" : "measured");
        g_cells++;
        if (!p)
        {
            printf("FAIL %s: refused: %s\n", cell, vfft_error());
            g_bad++;
            continue;
        }
        vfft_info(p, &info);
        if (info.in_doubles != (size_t)N || info.out_doubles != (size_t)N)
        {
            g_bad++;
            printf("FAIL %s: vfft_info sizes %zu / %zu, want %d\n", cell, info.in_doubles, info.out_doubles, N);
        }
        for (int dir = 0; dir < 2; dir++)
        {   /* forward and backward: the same transform */
            const vfft_dir_t d = dir == 0 ? VFFT_FORWARD : VFFT_BACKWARD;
            if (ip) { memcpy(y, x, (size_t)N * 8); vfft_execute(p, d, y, y); }
            else { for (int i = 0; i < N; i++) y[i] = NAN; vfft_execute(p, d, x, y); }
            verdict(dir == 0 ? "forward" : "backward", cell, ref_err(r, y));
        }
        if (ip) { memcpy(z, x, (size_t)N * 8); vfft_execute(p, VFFT_FORWARD, z, z); vfft_execute(p, VFFT_BACKWARD, z, z); }
        else { vfft_execute(p, VFFT_FORWARD, x, y); vfft_execute(p, VFFT_BACKWARD, y, z); }
        {
            double e = 0;
            for (int i = 0; i < N; i++) if (fabs(z[i] - sc * x[i]) > e) e = fabs(z[i] - sc * x[i]);
            verdict("round trip", cell, e / (sc * amax(x, N)));
        }
        if (!ip)
        {
            memcpy(z, x, (size_t)N * 8);
            vfft_execute(p, VFFT_FORWARD, z, z);
            vfft_execute(p, VFFT_FORWARD, x, y);
            verdict("one-buffer call of an out-of-place plan", cell, memcmp(z, y, (size_t)N * 8) ? 1.0 : 0.0);
        }
        else
        {
            for (int i = 0; i < N; i++) y[i] = 7.0;
            vfft_execute(p, VFFT_FORWARD, x, y);
            g_cells++;
            if (!vfft_error()[0] || amax(y, N) != 7.0 || y[0] != 7.0)
            {
                g_bad++;
                printf("FAIL %s: a two-buffer call of an in-place plan was not refused\n", cell);
            }
        }
        vfft_destroy(p);
    }
    fflush(stdout);
    vfft_free(y); vfft_free(z);
}
static void both(int N, int T, int est, const char *wis)
{
    for (int sine = 0; sine < 2; sine++)
        if (N >= xnmin(sine))
        {
            ref_t r;
            ref_make(&r, N, sine);
            one(&r, T, est, wis);
            ref_free(&r);
        }
}

/* the measured lengths: the short ones, the lengths whose real FFT is a power of two, and plain ones */
static const int mlist[2][14] = {
    { 2, 3, 4, 9, 16, 17, 33, 65, 100, 129, 257, 1000, 1025, 4097 },     /* DCT-I */
    { 1, 2, 3, 7, 15, 16, 31, 63, 100, 127, 255, 1000, 1023, 4095 },     /* DST-I */
};
#define NM 14
static void measured(int TM, const char *wis)
{
    for (int sine = 0; sine < 2; sine++)
        for (int i = 0; i < NM; i++)
        {
            ref_t r;
            ref_make(&r, mlist[sine][i], sine);
            one(&r, 1, 0, wis);
            if (TM > 1 && mlist[sine][i] >= 1000) one(&r, TM, 0, wis);
            ref_free(&r);
        }
}

int main(int argc, char **argv)
{
    static const int big[] = { 511, 512, 513, 1000, 1023, 1024, 1025, 2047, 2048, 2049, 3125, 4095, 4096, 4097, 8191,
                               8193, 10007, 16385, 30030, 32767, 65535, 65537, 100000, 262143, 262145, 524289,
                               1048575, 1048577 };
    int quick = 0, TM;
    const char *wis = NULL;
    vfft_info_t mach;
    for (int a = 1; a < argc; a++)
    {
        if (!strcmp(argv[a], "quick")) quick = 1;
        else if (!strcmp(argv[a], "--wisdir") && a + 1 < argc) wis = argv[++a];
    }
    if (!wis)
    {
        printf("usage: dct1_dst1_gate --wisdir <scratch dir> [quick]\n");
        return 2;
    }
    vfft_info(NULL, &mach);
    TM = mach.cores < 8 ? mach.cores : 8;   /* the threaded cells: 8 threads, or this machine's cores */
    printf("threaded cells on %d threads%s\n", TM, TM < 2 ? " (one core: not run)" : "");
    {   /* the refusals */
        vfft_plan p;
        g_cells += 3;
        if ((p = mk(0, 2, 8, 1, 0, 1, 1, NULL)) != NULL) { g_bad++; printf("FAIL: a rank-2 DCT1 was built\n"); vfft_destroy(p); }
        else if (!vfft_error()[0]) { g_bad++; printf("FAIL: the rank-2 refusal gave no reason\n"); }
        if ((p = mk(1, 1, 8, 2, 0, 1, 1, NULL)) != NULL) { g_bad++; printf("FAIL: a DST1 batch of 2 was built\n"); vfft_destroy(p); }
        else if (!vfft_error()[0]) { g_bad++; printf("FAIL: the batch refusal gave no reason\n"); }
        if ((p = mk(0, 1, 1, 1, 0, 1, 1, NULL)) != NULL) { g_bad++; printf("FAIL: a DCT1 of one point was built\n"); vfft_destroy(p); }
        else if (!vfft_error()[0]) { g_bad++; printf("FAIL: the one-point DCT1 refusal gave no reason\n"); }
    }
    for (int N = 1; N <= (quick ? 70 : 300); N++)
    {
        both(N, 1, 1, NULL);
        if (TM > 1 && (N % 7 == 0 || N < 20)) both(N, TM, 1, NULL);
    }
    for (size_t i = 0; i < sizeof big / sizeof big[0] && !quick; i++)
    {
        both(big[i], 1, 1, NULL);
        if (TM > 1) both(big[i], TM, 1, NULL);
    }
    measured(TM, wis);
    {   /* THE STORE: the two transforms' own rows in the real shard, each with its real FFT under its prefix,
         * and nothing else -- no r2c or c2r row, no other shard */
        char path[1024], line[8192];
        long nrow[2] = { 0, 0 }, want[2] = { 0, 0 }, other = 0;
        FILE *f;
        for (int sine = 0; sine < 2; sine++)
            for (int i = 0; i < NM; i++)
                if (mlist[sine][i] > xnmin(sine)) want[sine] += 1 + (TM > 1 && mlist[sine][i] >= 1000);
        snprintf(path, sizeof path, "%s/wisdom2_real.txt", wis);
        f = fopen(path, "r");
        g_cells++;
        if (!f) { g_bad++; printf("FAIL store: %s was not written\n", path); }
        else
        {
            while (fgets(line, sizeof line, f))
            {
                int sine, N = 0, cn = 0;
                const char *ck;
                if (strncmp(line, "@cell ", 6)) continue;
                sine = !strncmp(line, "@cell t=dst1 ", 13) ? 1 : !strncmp(line, "@cell t=dct1 ", 13) ? 0 : -1;
                ck = sine < 0 ? NULL : strstr(line, sine ? " dst_k1=t=r2c,n=" : " dct_k1=t=r2c,n=");
                if (ck && sscanf(line + 13, "n=%d", &N) == 1 && sscanf(ck + 16, "%d", &cn) == 1 && cn == xfftlen(sine, N) &&
                    strstr(line, " place=oop ") && strstr(line, sine ? " eng=dst " : " eng=dct "))
                    nrow[sine]++;
                else { other++; printf("FAIL store: a row that is not a DCT-I or DST-I cell's: %.120s", line); }
            }
            fclose(f);
            if (nrow[0] != want[0] || nrow[1] != want[1] || other) g_bad++;
            printf("store: %ld dct1 rows (want %ld), %ld dst1 rows (want %ld), %ld others\n", nrow[0], want[0], nrow[1], want[1], other);
        }
        {
            static const char *none[] = { "wisdom2_oop.txt", "wisdom2_scr.txt", "wisdom2_prime.txt", "wisdom2_2d.txt", "wisdom2_3d.txt" };
            for (size_t i = 0; i < sizeof none / sizeof none[0]; i++)
            {
                snprintf(path, sizeof path, "%s/%s", wis, none[i]);
                g_cells++;
                if ((f = fopen(path, "r")) != NULL) { fclose(f); g_bad++; printf("FAIL store: %s was written\n", none[i]); }
            }
        }
    }
    {   /* THE REPLAY: the same cells created again measure nothing, and are correct */
        const long s0 = vfft_measure_scopes();
        measured(TM, wis);
        g_cells++;
        if (vfft_measure_scopes() != s0)
        {
            g_bad++;
            printf("FAIL replay: %ld race scope(s) entered while replaying the banked cells\n", vfft_measure_scopes() - s0);
        }
        else
            printf("replay: nothing measured\n");
    }
    printf("%ld checks, %ld failed; worst relative error %.2e\n", g_cells, g_bad, g_worst);
    printf(g_bad ? "DCT1 DST1 GATE FAILED\n" : "DCT1 DST1 GATE ALL PASS\n");
    return g_bad != 0;
}
