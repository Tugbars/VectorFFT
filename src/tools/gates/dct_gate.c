/* dct_gate.c -- DCT-II, DCT-III, DCT-IV and DST-IV (VFFT_DCT2, VFFT_DCT3, VFFT_DCT4, VFFT_DST4;
 * src/core/il/real/dct.h, dct4_dst4.h) through the public door, against the definitions of include/vfft.h
 * summed directly in long double:
 *   DCT-II   Y[k] = 2 sum_{n=0..N-1} x[n] cos(pi k (2n+1) / 2N)
 *   DCT-III  Y[k] = X[0] + 2 sum_{n=1..N-1} X[n] cos(pi n (2k+1) / 2N)
 *   DCT-IV   Y[k] = 2 sum_{n=0..N-1} x[n] cos(pi (2k+1)(2n+1) / 4N)
 *   DST-IV   Y[k] = 2 sum_{n=0..N-1} x[n] sin(pi (2k+1)(2n+1) / 4N)
 *
 * THE CELLS: every N in 1..300 and 25 larger lengths to 2^20 (powers of two, odd, prime, mixed), each as a
 * VFFT_DCT2, a VFFT_DCT3, a VFFT_DCT4 and a VFFT_DST4 plan, out of place and in place, on one thread (and on
 * 8 threads, or this machine's cores when it has fewer, at every seventh N, at N below 20 and at the larger
 * lengths), planned by formulas (VFFT_FLAG_ESTIMATE); and 16 lengths planned by measuring into the scratch
 * store (--wisdir), the three longest also threaded.
 *
 * PER CELL:
 *   - the plan's forward and its backward (a DCT2 plan: DCT-II forward, DCT-III backward; a DCT3 plan the
 *     other way round; DCT-IV and DST-IV are their own inverses) against the reference: every output up to
 *     N = 2048, above it 48 outputs;
 *   - the round trip, backward of forward = 2N x, every output;
 *   - vfft_info: N doubles in, N doubles out;
 *   - an out-of-place plan called with out == in gives the two-buffer result bit for bit;
 *   - an in-place plan called with two buffers is refused and leaves the output alone.
 * THE REFUSALS: rank 2 and howmany 2 of each kind, and VFFT_DHT (each NULL with its reason).
 * THE STORE after the measured cells: wisdom2_real.txt holds one row per kind, measured length and thread
 * count (place=oop: one row serves both placements) -- t=dct2 and t=dct3 with eng=dct and their real FFT
 * under dct_ (an R2C on the dct2 row, a C2R on the dct3 row), t=dct4 with eng=dct and an R2C under dct_,
 * t=dst4 with eng=dst and an R2C under dst_ -- and no other row: these cells never bank an r2c or a c2r row;
 * no other shard is written. THE REPLAY: the measured cells created again enter no race scope, and are
 * checked like the first time.
 * The limit is 2e-13 relative to the largest output.
 *
 * Build: python gauntlet/build.py --compile --vfft --src src/tools/gates/dct_gate.c
 * Run:   dct_gate --wisdir <scratch dir> [quick]      (the store is WRITTEN: the real FFTs' rows) */
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
static unsigned g_seed = 20261123u;
static long g_cells, g_bad;
static double g_worst;

enum { R_DCT2, R_DCT3, R_DCT4, R_DST4, NREF };
static const char *const kname[NREF] = { "DCT2", "DCT3", "DCT4", "DST4" };
static const vfft_transform_t kxf[NREF] = { VFFT_DCT2, VFFT_DCT3, VFFT_DCT4, VFFT_DST4 };
static const int kback[NREF] = { R_DCT3, R_DCT2, R_DCT4, R_DST4 };   /* the transform a plan runs backward */

static double urand(void)
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return (double)(g_seed >> 8) / (double)(1u << 24) - 0.5;
}
/* the definition's sum at output k; the integer product is reduced by the angle's period first */
static double ref_at(int kind, const double *x, int N, int k)
{
    long double s = 0;
    int n;
    if (kind == R_DCT2)
    {
        for (n = 0; n < N; n++)
            s += x[n] * cosl(PI * (long double)(((long long)k * (2 * n + 1)) % (4LL * N)) / (2.0L * N));
        return (double)(2.0L * s);
    }
    if (kind == R_DCT3)
    {
        for (n = 1; n < N; n++)
            s += x[n] * cosl(PI * (long double)(((long long)n * (2 * k + 1)) % (4LL * N)) / (2.0L * N));
        return (double)((long double)x[0] + 2.0L * s);
    }
    for (n = 0; n < N; n++)
    {
        const long double a = PI * (long double)(((long long)(2 * k + 1) * (2 * n + 1)) % (8LL * N)) / (4.0L * N);
        s += x[n] * (kind == R_DCT4 ? cosl(a) : sinl(a));
    }
    return (double)(2.0L * s);
}
static vfft_plan mk(int tf, int rank, int N, int K, int ip, int T, int est, const char *wis)
{
    vfft_config_t c;
    memset(&c, 0, sizeof c);
    c.transform = tf; c.rank = rank; c.n[0] = N; c.n[1] = rank > 1 ? N : 0; c.howmany = (size_t)K; c.nthreads = T;
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

/* the reference of one length: the checked output indices and the four transforms of x at them */
typedef struct { int N, cnt, *at; double *x, *r[NREF]; } ref_t;
static void ref_make(ref_t *r, int N)
{
    r->N = N;
    r->cnt = N <= FULL_N ? N : NSAMP;
    r->at = malloc((size_t)r->cnt * sizeof(int));
    r->x = vfft_malloc((size_t)N * 8);
    for (int i = 0; i < N; i++) r->x[i] = urand();
    for (int i = 0; i < r->cnt; i++)
        r->at[i] = N <= FULL_N ? i : (int)(((long long)i * 2654435761u + 17) % N);
    for (int kind = 0; kind < NREF; kind++)
    {
        r->r[kind] = malloc((size_t)r->cnt * 8);
        for (int i = 0; i < r->cnt; i++)
            r->r[kind][i] = ref_at(kind, r->x, N, r->at[i]);
    }
}
static void ref_free(ref_t *r)
{
    free(r->at); vfft_free(r->x);
    for (int kind = 0; kind < NREF; kind++) free(r->r[kind]);
}
static double ref_err(const ref_t *r, const double *got, int kind)
{
    const double *w = r->r[kind];
    double e = 0, sc = amax(got, r->N);
    for (int i = 0; i < r->cnt; i++)
        if (fabs(got[r->at[i]] - w[i]) > e) e = fabs(got[r->at[i]] - w[i]);
    return sc > 0 ? e / sc : e;
}

static void one(const ref_t *r, int T, int est, const char *wis)
{
    const int N = r->N;
    const double *x = r->x;
    double *y = vfft_malloc((size_t)N * 8), *z = vfft_malloc((size_t)N * 8);
    char cell[96];
    for (int kind = 0; kind < NREF; kind++)
        for (int ip = 0; ip < 2; ip++)
        {
            vfft_plan p = mk(kxf[kind], 1, N, 1, ip, T, est, wis);
            vfft_info_t info;
            snprintf(cell, sizeof cell, "N %d %s plan %s T %d %s", N, kname[kind], ip ? "ip" : "oop", T, est ? "estimate" : "measured");
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
            {   /* forward: the plan's own transform; backward: its inverse */
                const vfft_dir_t d = dir == 0 ? VFFT_FORWARD : VFFT_BACKWARD;
                if (ip) { memcpy(y, x, (size_t)N * 8); vfft_execute(p, d, y, y); }
                else { for (int i = 0; i < N; i++) y[i] = NAN; vfft_execute(p, d, x, y); }
                verdict(dir == 0 ? "forward" : "backward", cell, ref_err(r, y, dir == 0 ? kind : kback[kind]));
            }
            if (ip) { memcpy(z, x, (size_t)N * 8); vfft_execute(p, VFFT_FORWARD, z, z); vfft_execute(p, VFFT_BACKWARD, z, z); }
            else { vfft_execute(p, VFFT_FORWARD, x, y); vfft_execute(p, VFFT_BACKWARD, y, z); }
            {
                double e = 0;
                for (int i = 0; i < N; i++) if (fabs(z[i] - 2.0 * N * x[i]) > e) e = fabs(z[i] - 2.0 * N * x[i]);
                verdict("round trip", cell, e / (2.0 * N * amax(x, N)));
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

int main(int argc, char **argv)
{
    static const int big[] = { 512, 1000, 1009, 1024, 1536, 2047, 2048, 2310, 3125, 4096, 4099, 6144, 8191, 8192, 10007,
                               16384, 30030, 32768, 65536, 65537, 100000, 100003, 262144, 524288, 1048576 };
    static const int mlist[] = { 1, 2, 3, 8, 15, 16, 31, 64, 100, 127, 256, 257, 1000, 1001, 1024, 4096 };
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
        printf("usage: dct_gate --wisdir <scratch dir> [quick]\n");
        return 2;
    }
    vfft_info(NULL, &mach);
    TM = mach.cores < 8 ? mach.cores : 8;   /* the threaded cells: 8 threads, or this machine's cores */
    printf("threaded cells on %d threads%s\n", TM, TM < 2 ? " (one core: not run)" : "");
    {   /* the refusals */
        vfft_plan p;
        for (int kind = 0; kind < NREF; kind++)
        {
            g_cells += 2;
            if ((p = mk(kxf[kind], 2, 8, 1, 0, 1, 1, NULL)) != NULL) { g_bad++; printf("FAIL: a rank-2 %s was built\n", kname[kind]); vfft_destroy(p); }
            else if (!vfft_error()[0]) { g_bad++; printf("FAIL: the rank-2 %s refusal gave no reason\n", kname[kind]); }
            if ((p = mk(kxf[kind], 1, 8, 2, 0, 1, 1, NULL)) != NULL) { g_bad++; printf("FAIL: a %s batch of 2 was built\n", kname[kind]); vfft_destroy(p); }
            else if (!vfft_error()[0]) { g_bad++; printf("FAIL: the %s batch refusal gave no reason\n", kname[kind]); }
        }
        g_cells += 2;
        if ((p = mk(VFFT_DHT, 1, 8, 1, 0, 1, 1, NULL)) != NULL) { g_bad++; printf("FAIL: an estimate DHT was built\n"); vfft_destroy(p); }
        else if (!vfft_error()[0]) { g_bad++; printf("FAIL: the estimate DHT refusal gave no reason\n"); }
        if ((p = mk(VFFT_DHT, 1, 8, 1, 0, 1, 0, wis)) != NULL) { g_bad++; printf("FAIL: a DHT was built\n"); vfft_destroy(p); }
        else if (!vfft_error()[0]) { g_bad++; printf("FAIL: the DHT refusal gave no reason\n"); }
    }
    for (int N = 1; N <= (quick ? 70 : 300); N++)
    {
        ref_t r;
        ref_make(&r, N);
        one(&r, 1, 1, NULL);
        if (TM > 1 && (N % 7 == 0 || N < 20)) one(&r, TM, 1, NULL);
        ref_free(&r);
    }
    for (size_t i = 0; i < sizeof big / sizeof big[0] && !quick; i++)
    {
        ref_t r;
        ref_make(&r, big[i]);
        one(&r, 1, 1, NULL);
        if (TM > 1) one(&r, TM, 1, NULL);
        ref_free(&r);
    }
    for (size_t i = 0; i < sizeof mlist / sizeof mlist[0]; i++)
    {
        ref_t r;
        ref_make(&r, mlist[i]);
        one(&r, 1, 0, wis);
        if (TM > 1 && mlist[i] >= 1000) one(&r, TM, 0, wis);
        ref_free(&r);
    }
    {   /* THE STORE: the cells' own rows in the real shard, each with its real FFT under its prefix, and nothing
         * else -- no r2c or c2r row, no other shard */
        char path[1024], line[8192];
        long cnt[NREF] = { 0, 0, 0, 0 }, other = 0, want = 0;
        FILE *f;
        for (size_t i = 0; i < sizeof mlist / sizeof mlist[0]; i++)
            if (mlist[i] > 1) want += 1 + (TM > 1 && mlist[i] >= 1000);
        snprintf(path, sizeof path, "%s/wisdom2_real.txt", wis);
        f = fopen(path, "r");
        g_cells++;
        if (!f) { g_bad++; printf("FAIL store: %s was not written\n", path); }
        else
        {
            while (fgets(line, sizeof line, f))
            {
                const int oop = strstr(line, " place=oop ") != NULL;
                if (strncmp(line, "@cell ", 6)) continue;
                if (!strncmp(line, "@cell t=dct2 ", 13) && oop && strstr(line, " eng=dct ") && strstr(line, " dct_k1=t=r2c,")) cnt[R_DCT2]++;
                else if (!strncmp(line, "@cell t=dct3 ", 13) && oop && strstr(line, " eng=dct ") && strstr(line, " dct_k1=t=c2r,")) cnt[R_DCT3]++;
                else if (!strncmp(line, "@cell t=dct4 ", 13) && oop && strstr(line, " eng=dct ") && strstr(line, " dct_k1=t=r2c,")) cnt[R_DCT4]++;
                else if (!strncmp(line, "@cell t=dst4 ", 13) && oop && strstr(line, " eng=dst ") && strstr(line, " dst_k1=t=r2c,")) cnt[R_DST4]++;
                else { other++; printf("FAIL store: a row that is not one of these cells': %.120s", line); }
            }
            fclose(f);
            for (int kind = 0; kind < NREF; kind++)
                if (cnt[kind] != want) { g_bad++; printf("FAIL store: %ld %s rows, want %ld\n", cnt[kind], kname[kind], want); }
            if (other) g_bad++;
            printf("store: %ld dct2, %ld dct3, %ld dct4, %ld dst4 rows (want %ld each), %ld others\n", cnt[R_DCT2], cnt[R_DCT3],
                   cnt[R_DCT4], cnt[R_DST4], want, other);
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
        for (size_t i = 0; i < sizeof mlist / sizeof mlist[0]; i++)
        {
            ref_t r;
            ref_make(&r, mlist[i]);
            one(&r, 1, 0, wis);
            if (TM > 1 && mlist[i] >= 1000) one(&r, TM, 0, wis);
            ref_free(&r);
        }
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
    printf(g_bad ? "DCT GATE FAILED\n" : "DCT GATE ALL PASS\n");
    return g_bad != 0;
}
