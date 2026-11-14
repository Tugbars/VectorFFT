/* dst_gate.c -- DST-II and DST-III (VFFT_DST2, VFFT_DST3; src/core/il/real/dst.h) through the public door,
 * against the definitions of include/vfft.h summed directly in long double:
 *   DST-II   Y[k] = 2 sum_{n=0..N-1} x[n] sin(pi (k+1)(2n+1) / 2N)
 *   DST-III  Y[k] = (-1)^k X[N-1] + 2 sum_{n=0..N-2} X[n] sin(pi (n+1)(2k+1) / 2N)
 *
 * THE CELLS: every N in 1..300 and 23 larger lengths to 2^20 (powers of two, odd, prime, mixed), each as a
 * VFFT_DST2 plan and a VFFT_DST3 plan, out of place and in place, on one thread (and on 8 threads, or this
 * machine's cores when it has fewer, at every seventh N, at N below 20 and at the larger lengths), planned by
 * formulas (VFFT_FLAG_ESTIMATE); and 14 lengths planned by measuring into the scratch store (--wisdir), the
 * three longest also threaded.
 *
 * PER CELL:
 *   - the plan's forward and its backward (a DST2 plan: DST-II forward, DST-III backward; a DST3 plan the
 *     other way round) against the reference: every output up to N = 2048, above it 48 outputs;
 *   - the round trip, backward of forward = 2N x, every output;
 *   - vfft_info: N doubles in, N doubles out;
 *   - an out-of-place plan called with out == in gives the two-buffer result bit for bit;
 *   - an in-place plan called with two buffers is refused and leaves the output alone.
 * THE REFUSALS: rank 2, howmany 2 (each NULL with its reason).
 * THE STORE after the measured cells: wisdom2_real.txt holds one t=dst2 and one t=dst3 row per measured length
 * and thread count (place=oop: one row serves both placements), each eng=dst with its real FFT under dst_ (an
 * R2C on the dst2 row, a C2R on the dst3 row), and no other row -- a DST never banks an r2c or a c2r row;
 * no other shard is written. THE REPLAY: the measured cells created again enter no race scope, and are
 * checked like the first time.
 * The limit is 2e-13 relative to the largest output.
 *
 * Build: python gauntlet/build.py --compile --vfft --src src/tools/gates/dst_gate.c
 * Run:   dst_gate --wisdir <scratch dir> [quick]      (the store is WRITTEN: the real FFTs' rows) */
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

static double urand(void)
{
    g_seed = g_seed * 1664525u + 1013904223u;
    return (double)(g_seed >> 8) / (double)(1u << 24) - 0.5;
}
static double ref2_at(const double *x, int N, int k)
{
    long double s = 0;
    for (int n = 0; n < N; n++)
        s += x[n] * sinl(PI * (long double)(((long long)(k + 1) * (2 * n + 1)) % (4LL * N)) / (2.0L * N));
    return (double)(2.0L * s);
}
static double ref3_at(const double *X, int N, int k)
{
    long double s = 0;
    for (int n = 0; n < N - 1; n++)
        s += X[n] * sinl(PI * (long double)(((long long)(n + 1) * (2 * k + 1)) % (4LL * N)) / (2.0L * N));
    return (double)(((k & 1) ? -1.0L : 1.0L) * X[N - 1] + 2.0L * s);
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

/* the reference of one length: the checked output indices and both transforms of x at them */
typedef struct { int N, cnt, *at; double *x, *r2, *r3; } ref_t;
static void ref_make(ref_t *r, int N)
{
    r->N = N;
    r->cnt = N <= FULL_N ? N : NSAMP;
    r->at = malloc((size_t)r->cnt * sizeof(int));
    r->x = vfft_malloc((size_t)N * 8);
    r->r2 = malloc((size_t)r->cnt * 8);
    r->r3 = malloc((size_t)r->cnt * 8);
    for (int i = 0; i < N; i++) r->x[i] = urand();
    for (int i = 0; i < r->cnt; i++)
    {
        r->at[i] = N <= FULL_N ? i : (int)(((long long)i * 2654435761u + 17) % N);
        r->r2[i] = ref2_at(r->x, N, r->at[i]);
        r->r3[i] = ref3_at(r->x, N, r->at[i]);
    }
}
static void ref_free(ref_t *r)
{
    free(r->at); vfft_free(r->x); free(r->r2); free(r->r3);
}
static double ref_err(const ref_t *r, const double *got, int kind)
{
    const double *w = kind == 2 ? r->r2 : r->r3;
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
    for (int kind = 2; kind <= 3; kind++)
        for (int ip = 0; ip < 2; ip++)
        {
            vfft_plan p = mk(kind == 2 ? VFFT_DST2 : VFFT_DST3, 1, N, 1, ip, T, est, wis);
            vfft_info_t info;
            snprintf(cell, sizeof cell, "N %d DST%d plan %s T %d %s", N, kind, ip ? "ip" : "oop", T, est ? "estimate" : "measured");
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
            {   /* forward: the plan's own transform; backward: the other one */
                const vfft_dir_t d = dir == 0 ? VFFT_FORWARD : VFFT_BACKWARD;
                if (ip) { memcpy(y, x, (size_t)N * 8); vfft_execute(p, d, y, y); }
                else { for (int i = 0; i < N; i++) y[i] = NAN; vfft_execute(p, d, x, y); }
                verdict(dir == 0 ? "forward" : "backward", cell, ref_err(r, y, dir == 0 ? kind : 5 - kind));
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
    static const int big[] = { 512, 1000, 1009, 1024, 1536, 2047, 2048, 2310, 3125, 4096, 6144, 8191, 8192, 10007, 16384,
                               30030, 32768, 65536, 65537, 100000, 262144, 524288, 1048576 };
    static const int mlist[] = { 1, 2, 3, 8, 15, 16, 31, 64, 100, 127, 256, 1000, 1024, 4096 };
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
        printf("usage: dst_gate --wisdir <scratch dir> [quick]\n");
        return 2;
    }
    vfft_info(NULL, &mach);
    TM = mach.cores < 8 ? mach.cores : 8;   /* the threaded cells: 8 threads, or this machine's cores */
    printf("threaded cells on %d threads%s\n", TM, TM < 2 ? " (one core: not run)" : "");
    {   /* the refusals */
        vfft_plan p;
        g_cells += 2;
        if ((p = mk(VFFT_DST2, 2, 8, 1, 0, 1, 1, NULL)) != NULL) { g_bad++; printf("FAIL: a rank-2 DST2 was built\n"); vfft_destroy(p); }
        else if (!vfft_error()[0]) { g_bad++; printf("FAIL: the rank-2 refusal gave no reason\n"); }
        if ((p = mk(VFFT_DST3, 1, 8, 2, 0, 1, 1, NULL)) != NULL) { g_bad++; printf("FAIL: a DST3 batch of 2 was built\n"); vfft_destroy(p); }
        else if (!vfft_error()[0]) { g_bad++; printf("FAIL: the batch refusal gave no reason\n"); }
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
    {   /* THE STORE: the DST cells' own rows in the real shard, each with its real FFT under dst_, and nothing
         * else -- no r2c or c2r row, no other shard */
        char path[1024], line[8192];
        long n2 = 0, n3 = 0, other = 0, want = 0;
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
                if (strncmp(line, "@cell ", 6)) continue;
                if (!strncmp(line, "@cell t=dst2 ", 13) && strstr(line, " place=oop ") && strstr(line, " eng=dst ") &&
                    strstr(line, " dst_k1=t=r2c,")) n2++;
                else if (!strncmp(line, "@cell t=dst3 ", 13) && strstr(line, " place=oop ") && strstr(line, " eng=dst ") &&
                         strstr(line, " dst_k1=t=c2r,")) n3++;
                else { other++; printf("FAIL store: a row that is not a DST cell's: %.120s", line); }
            }
            fclose(f);
            if (n2 != want || n3 != want || other) g_bad++;
            printf("store: %ld dst2 rows, %ld dst3 rows (want %ld each), %ld others\n", n2, n3, want, other);
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
    printf(g_bad ? "DST GATE FAILED\n" : "DST GATE ALL PASS\n");
    return g_bad != 0;
}
