/* estimate_gate.c — VFFT_FLAG_ESTIMATE (include/vfft.h; il/planning/estimate_il.h,
 * il/rank1/c2c_estimate_il.h): one 1D interleaved c2c transform planned by formulas, nothing measured.
 *
 * Every create goes through the public API with the flag and names the gate's scratch store:
 *   A  correct against a long-double DFT: every bin up to N = 4096 (scrambled: the sorted multiset of the
 *      outputs), above it 32 sampled bins (scrambled: each found among the outputs, the contract being any
 *      permutation), and the forward-backward round trip. The cells:
 *        A1 natural, out of place, every N in 2..1024 and every 13th N in 1025..4096;
 *        A2 the four order x placement cells at every N in 2..256 and every power of two to 2^20;
 *        A3 the four cells at 100 lengths to 2^20 drawn with a fixed seed, 70 any N and 30 primes;
 *        A4 the four cells at T = 8, 30 lengths of every route.
 *      A cell with a prime factor above 47, in place, scrambled, one thread is the one refusal the four
 *      cells meet (its route row and its prime method row share one wisdom key): it must be refused with
 *      its reason; a smooth N past 262144 may meet it too (the unraced Bluestein). Every other cell builds.
 *   M  nothing measured at any create: the races counted (_vfft_create_race_count, vfft.c) and the race
 *      scopes entered (vfft_measure_scopes) do not move across it.
 *   R  the refusals: R2C, C2R, a real-to-real transform, rank 2, howmany 4, SPLIT, the flag with
 *      VFFT_FLAG_RECALIBRATE, with VFFT_FLAG_PATIENT, and the in-place scrambled prime cell at one thread:
 *      NULL, vfft_error() naming the reason, nothing measured.
 *   W  no wisdom store touched: the store named by wisdom_dir keeps its files (none created), and the
 *      library's own store (VFFT_WISDOM_DIR, pointed at a scratch folder before the first create) gets none.
 * Usage: estimate_gate --wisdir <cold scratch dir> */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_diagnostics.h"
#ifdef _WIN32
#include <windows.h>
#else
#include <dirent.h>
#include <sys/stat.h>
#endif

extern long _vfft_create_race_count;   /* vfft.c:255: every race a create starts (common/support/race.h:149) */

#define FULL_N 4096   /* every bin against the long-double DFT up to here */
#define NBINS 32      /* sampled bins above */
#define TOL 1e-11     /* the error a cell accepts, relative to the rms of the bins */

static int g_checks = 0, g_fail = 0, g_shown = 0;
static long g_built = 0, g_known = 0;
static const char *g_W = NULL;
static char g_D[1024];   /* the library's own store, pointed here */
#define CHECK(c, ...) do { g_checks++; if (!(c)) { g_fail++; if (g_shown++ < 40) { printf("  *** FAIL: " __VA_ARGS__); printf("\n"); } } } while (0)

/* ── the stores' files ── */
/* FNV-1a over a file's name, size and last write time */
static unsigned long long file_key(const char *name, unsigned long long size, unsigned long long wtime)
{
    unsigned long long h = 1469598103934665603ull;
    for (; *name; name++)
        h = (h ^ (unsigned char)*name) * 1099511628211ull;
    h = (h ^ size) * 1099511628211ull;
    return (h ^ wtime) * 1099511628211ull;
}
/* a directory's files: their count and the sum of their keys; a save adds a file or moves one's size or time */
static unsigned long long dir_digest(const char *dir, int *files)
{
    unsigned long long d = 0;
    *files = 0;
#ifdef _WIN32
    {
        char pat[1100];
        WIN32_FIND_DATAA f;
        HANDLE h;
        snprintf(pat, sizeof pat, "%s/*", dir);
        h = FindFirstFileA(pat, &f);
        if (h == INVALID_HANDLE_VALUE)
            return 0;
        do
            if (!(f.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY))
            {
                ++*files;
                d += file_key(f.cFileName, ((unsigned long long)f.nFileSizeHigh << 32) | f.nFileSizeLow,
                              ((unsigned long long)f.ftLastWriteTime.dwHighDateTime << 32) | f.ftLastWriteTime.dwLowDateTime);
            }
        while (FindNextFileA(h, &f));
        FindClose(h);
    }
#else
    {
        DIR *dp = opendir(dir);
        struct dirent *e;
        struct stat st;
        char path[1100];
        if (!dp)
            return 0;
        while ((e = readdir(dp)) != NULL)
        {
            snprintf(path, sizeof path, "%s/%s", dir, e->d_name);
            if (stat(path, &st) == 0 && S_ISREG(st.st_mode))
            {
                ++*files;
                d += file_key(e->d_name, (unsigned long long)st.st_size, (unsigned long long)st.st_mtime);
            }
        }
        closedir(dp);
    }
#endif
    return d;
}

/* ── the reference ── */
static const long double PI_L = 3.141592653589793238462643383279502884L;
static unsigned long long g_rng;
static double rnd(void)
{
    g_rng ^= g_rng >> 12; g_rng ^= g_rng << 25; g_rng ^= g_rng >> 27;
    return (double)((g_rng * 2685821657736338717ull) >> 11) / 9007199254740992.0 - 0.5;
}
/* one length's input and its forward: every bin (N <= FULL_N) or NBINS sampled bins */
typedef struct
{
    long N;
    double *x;            /* the input, complex interleaved */
    long double sx;       /* sum |x|^2 */
    long double *Xr, *Xi; /* N <= FULL_N: every bin */
    long k[NBINS];        /* above: the sampled bins */
    long double Kr[NBINS], Ki[NBINS];
    int nk;
} ref_t;

/* bin f of the forward, blocked: exp(theta(m f)) = exp(theta(aB f)) exp(theta(b f)) with m = aB + b, so N up
 * to 2^20 needs B + N/B long-double twiddles per bin */
static void dft_one(const double *x, long N, long f, long double *orr, long double *oi)
{
    enum { B = 1024 };
    const long nb = (N + B - 1) / B;
    long double *ic = (long double *)malloc(2 * B * sizeof *ic), *oc = (long double *)malloc((size_t)(2 * nb) * sizeof *oc);
    long double tr = 0.0L, ti = 0.0L;
    long a, b;
    if (!ic || !oc) { printf("out of memory\n"); exit(2); }
    for (b = 0; b < B; b++)
    {
        const long j = (long)(((unsigned long long)b * (unsigned long long)f) % (unsigned long long)N);
        ic[2 * b] = cosl(2.0L * PI_L * j / N);
        ic[2 * b + 1] = -sinl(2.0L * PI_L * j / N);
    }
    for (a = 0; a < nb; a++)
    {
        const long j = (long)((((unsigned long long)a * B) % (unsigned long long)N) * (unsigned long long)f % (unsigned long long)N);
        oc[2 * a] = cosl(2.0L * PI_L * j / N);
        oc[2 * a + 1] = -sinl(2.0L * PI_L * j / N);
    }
    for (a = 0; a < nb; a++)
    {
        long double sr = 0.0L, si = 0.0L;
        const long m0 = a * B, m1 = (m0 + B < N) ? m0 + B : N;
        long m;
        for (m = m0; m < m1; m++)
        {
            const long double xr = x[2 * m], xi = x[2 * m + 1], cr = ic[2 * (m - m0)], ci = ic[2 * (m - m0) + 1];
            sr += xr * cr - xi * ci;
            si += xr * ci + xi * cr;
        }
        tr += sr * oc[2 * a] - si * oc[2 * a + 1];
        ti += sr * oc[2 * a + 1] + si * oc[2 * a];
    }
    *orr = tr;
    *oi = ti;
    free(ic);
    free(oc);
}
static void ref_make(ref_t *r, long N)
{
    long i, f;
    memset(r, 0, sizeof *r);
    r->N = N;
    r->x = (double *)malloc((size_t)2 * N * sizeof(double));
    if (!r->x) { printf("out of memory\n"); exit(2); }
    g_rng = 0x9E3779B97F4A7C15ull ^ ((unsigned long long)N << 17);
    for (i = 0; i < 2 * N; i++) { r->x[i] = rnd(); r->sx += (long double)r->x[i] * r->x[i]; }
    if (N <= FULL_N)
    {
        long double *c = (long double *)malloc((size_t)N * sizeof *c), *s = (long double *)malloc((size_t)N * sizeof *s);
        r->Xr = (long double *)malloc((size_t)N * sizeof(long double));
        r->Xi = (long double *)malloc((size_t)N * sizeof(long double));
        if (!c || !s || !r->Xr || !r->Xi) { printf("out of memory\n"); exit(2); }
        for (i = 0; i < N; i++) { c[i] = cosl(2.0L * PI_L * i / N); s[i] = -sinl(2.0L * PI_L * i / N); }
        for (f = 0; f < N; f++)
        {
            long double sr = 0.0L, si = 0.0L;
            long idx = 0;
            for (i = 0; i < N; i++)
            {
                sr += r->x[2 * i] * c[idx] - r->x[2 * i + 1] * s[idx];
                si += r->x[2 * i] * s[idx] + r->x[2 * i + 1] * c[idx];
                idx += f;
                if (idx >= N) idx -= N;
            }
            r->Xr[f] = sr;
            r->Xi[f] = si;
        }
        free(c);
        free(s);
        return;
    }
    {   /* the edges, then bins drawn with a fixed seed */
        const long fixed[5] = { 0, 1, 2, N - 1, N / 2 };
        unsigned long long z = 0xD1B54A32D192ED03ull ^ (unsigned long long)N;
        int j, n = 0;
        for (i = 0; i < 5; i++) r->k[n++] = fixed[i];
        while (n < NBINS)
        {
            long v;
            int dup = 0;
            z ^= z >> 12; z ^= z << 25; z ^= z >> 27;
            v = (long)((z * 2685821657736338717ull) % (unsigned long long)N);
            for (j = 0; j < n; j++) if (r->k[j] == v) dup = 1;
            if (!dup) r->k[n++] = v;
        }
        r->nk = n;
        for (j = 0; j < n; j++) dft_one(r->x, N, r->k[j], &r->Kr[j], &r->Ki[j]);
    }
}
static void ref_free(ref_t *r)
{
    free(r->x);
    free(r->Xr);
    free(r->Xi);
    memset(r, 0, sizeof *r);
}

/* ── the checks of one plan ── */
static int cmp_cplx(const void *a, const void *b)
{
    const long double *x = (const long double *)a, *y = (const long double *)b;
    if (x[0] != y[0]) return x[0] < y[0] ? -1 : 1;
    return (x[1] > y[1]) - (x[1] < y[1]);
}
/* the forward's error against the reference, and the round trip's, both relative (a NaN wins) */
static void errors(vfft_plan p, const ref_t *r, int scr, int ip, double *ebin, double *ert)
{
    const long N = r->N;
    const size_t nb = (size_t)2 * N * sizeof(double);
    double *in = (double *)vfft_malloc(nb), *out = ip ? in : (double *)vfft_malloc(nb), *z = ip ? in : (double *)vfft_malloc(nb);
    const long double rms = sqrtl(r->sx);   /* the rms of |X| (Parseval) */
    long double m = 0.0L;
    long i;
    if (!in || !out || !z) { printf("out of memory\n"); exit(2); }
    memcpy(in, r->x, nb);
    vfft_execute(p, VFFT_FORWARD, in, out);
    if (N <= FULL_N && !scr)
        for (i = 0; i < N; i++)
        {
            const long double d = hypotl(out[2 * i] - r->Xr[i], out[2 * i + 1] - r->Xi[i]);
            if (!(d <= m)) m = d;
        }
    else if (N <= FULL_N)
    {   /* any permutation of the bins: the sorted multisets */
        long double *a = (long double *)malloc((size_t)N * 2 * sizeof(long double));
        long double *b = (long double *)malloc((size_t)N * 2 * sizeof(long double));
        if (!a || !b) { printf("out of memory\n"); exit(2); }
        for (i = 0; i < N; i++) { a[2 * i] = out[2 * i]; a[2 * i + 1] = out[2 * i + 1]; b[2 * i] = r->Xr[i]; b[2 * i + 1] = r->Xi[i]; }
        qsort(a, (size_t)N, 2 * sizeof(long double), cmp_cplx);
        qsort(b, (size_t)N, 2 * sizeof(long double), cmp_cplx);
        for (i = 0; i < N; i++)
        {
            const long double d = hypotl(a[2 * i] - b[2 * i], a[2 * i + 1] - b[2 * i + 1]);
            if (!(d <= m)) m = d;
        }
        free(a);
        free(b);
    }
    else if (!scr)
        for (i = 0; i < r->nk; i++)
        {
            const long k = r->k[i];
            const long double d = hypotl(out[2 * k] - r->Kr[i], out[2 * k + 1] - r->Ki[i]);
            if (!(d <= m)) m = d;
        }
    else
    {   /* each sampled bin found among the N outputs: the nearest output's distance */
        double best[NBINS];
        long at[NBINS];
        int j;
        for (j = 0; j < r->nk; j++) { best[j] = HUGE_VAL; at[j] = -1; }
        for (i = 0; i < N; i++)
            for (j = 0; j < r->nk; j++)
            {
                const double dr = out[2 * i] - (double)r->Kr[j], di = out[2 * i + 1] - (double)r->Ki[j], d = dr * dr + di * di;
                if (d < best[j]) { best[j] = d; at[j] = i; }
            }
        for (j = 0; j < r->nk; j++)
        {
            long double d;
            if (at[j] < 0) { m = NAN; break; }
            d = hypotl(out[2 * at[j]] - r->Kr[j], out[2 * at[j] + 1] - r->Ki[j]);
            if (!(d <= m)) m = d;
        }
    }
    *ebin = (double)(m / rms);
    vfft_execute(p, VFFT_BACKWARD, out, z);
    m = 0.0L;
    for (i = 0; i < 2 * N; i++)
    {
        const long double d = fabsl((long double)z[i] / N - r->x[i]);
        if (!(d <= m)) m = d;
    }
    *ert = (double)(m / sqrtl(r->sx / N));
    vfft_free(in);
    if (!ip) { vfft_free(out); vfft_free(z); }
}

/* the largest prime factor */
static long lpf(long n)
{
    long d, best = 1;
    for (d = 2; d * d <= n; d++)
        while (n % d == 0) { best = d; n /= d; }
    return n > 1 ? n : best;
}

/* an estimate create on the gate's store; *measured: a race or a race scope moved across it */
static vfft_plan est_create(const vfft_config_t *c0, int *measured)
{
    vfft_config_t c = *c0;
    const long r0 = _vfft_create_race_count, s0 = vfft_measure_scopes();
    vfft_plan p;
    c.flags |= VFFT_FLAG_ESTIMATE;
    c.wisdom_dir = g_W;
    p = vfft_create(&c);
    *measured = (_vfft_create_race_count != r0) || (vfft_measure_scopes() != s0);
    return p;
}

/* one cell: built and correct, or the one refusal of the four cells */
static void cell(const ref_t *r, int scr, int ip, int T)
{
    const long N = r->N;
    const int must_refuse = ip && scr && T == 1 && lpf(N) > 47;
    const int may_refuse = must_refuse || (ip && scr && T == 1 && N > 262144);
    char name[96];
    vfft_config_t c;
    vfft_plan p;
    int measured;
    snprintf(name, sizeof name, "N=%ld %s %s T=%d", N, scr ? "scr" : "nat", ip ? "ip" : "oop", T);
    memset(&c, 0, sizeof c);
    c.n[0] = (int)N;
    c.order = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    c.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    c.nthreads = T;
    p = est_create(&c, &measured);
    CHECK(!measured, "M  %s: the create MEASURED", name);
    if (!p)
    {
        const int known = may_refuse && strstr(vfft_error(), "in place, scrambled, one thread") != NULL;
        CHECK(known, "A  %s: refused: %s", name, vfft_error());
        g_known += known;
        return;
    }
    CHECK(!must_refuse, "A  %s: the in-place scrambled prime cell at one thread was served (route %s)", name,
          vfft_plan_route(p));
    {
        double eb, er;
        errors(p, r, scr, ip, &eb, &er);
        CHECK(eb <= TOL && er <= TOL, "A  %s: route %s, forward %.2e, round trip %.2e (tolerance %.0e)", name,
              vfft_plan_route(p), eb, er, TOL);
        g_built++;
    }
    vfft_destroy(p);
}
static void four(const ref_t *r, int T)
{
    int scr, ip;
    for (scr = 0; scr < 2; scr++)
        for (ip = 0; ip < 2; ip++)
            cell(r, scr, ip, T);
}

/* ── the sections ── */
static int s_checks, s_fail;
static long s_built, s_known;
static void sec_begin(void) { s_checks = g_checks; s_fail = g_fail; s_built = g_built; s_known = g_known; }
static void sec_end(const char *name)
{
    const int f = g_fail - s_fail;
    printf("SECTION %-3s: %d checks, %d failed; %ld plans built and correct, %ld refused as the in-place scrambled "
           "prime cell at one thread -> %s\n", name, g_checks - s_checks, f, g_built - s_built, g_known - s_known,
           f ? "FAIL" : "PASS");
}

/* R: NULL with the reason named, nothing measured */
static void refused(const char *tag, const vfft_config_t *c, const char *want)
{
    int measured;
    vfft_plan p = est_create(c, &measured);
    CHECK(!p && strstr(vfft_error(), want) && !measured, "R  %s: %s, vfft_error() \"%s\", %s", tag,
          p ? "SERVED" : "refused", vfft_error(), measured ? "MEASURED" : "nothing measured");
    if (!p && strstr(vfft_error(), want) && !measured)
        printf("  ok  R  %s: \"%s\"\n", tag, vfft_error());
    vfft_destroy(p);
}

static int is_prime(long n)
{
    long d;
    if (n < 2) return 0;
    if (n % 2 == 0) return n == 2;
    for (d = 3; d * d <= n; d += 2)
        if (n % d == 0) return 0;
    return 1;
}

int main(int argc, char **argv)
{
    /* A4: thirty lengths of every route at T = 8 */
    static const long T8[30] = { 64, 100, 997, 1000, 1009, 2048, 4096, 4099, 6000, 10007, 12288, 16384, 32768, 50000,
                                 65536, 65537, 100000, 131072, 196608, 262144, 300000, 499979, 524288, 655360, 786432,
                                 999983, 1000000, 1048573, 1048576, 2 * 3 * 5 * 7 * 11 * 13 * 17 };
    ref_t r;
    long N;
    int a, i, f0, f1, d0n, d1n;
    unsigned long long d0, d1, e0, e1;
    unsigned long long z = 0x2545F4914F6CDD1Dull;
    for (a = 1; a + 1 < argc; a++)
        if (!strcmp(argv[a], "--wisdir"))
            g_W = argv[a + 1];
    if (!g_W)
    {
        printf("usage: estimate_gate --wisdir <cold scratch dir>\n");
        return 2;
    }
    setvbuf(stdout, NULL, _IONBF, 0);
    /* the library's own store, pointed at a scratch folder that must stay without files */
    snprintf(g_D, sizeof g_D, "%s/library_store", g_W);
#ifdef _WIN32
    _putenv_s("VFFT_WISDOM_DIR", g_D);
#else
    setenv("VFFT_WISDOM_DIR", g_D, 1);
#endif
    d0 = dir_digest(g_W, &f0);
    e0 = dir_digest(g_D, &d0n);
    printf("ESTIMATE GATE: VFFT_FLAG_ESTIMATE, 1D C2C; store %s (%d file(s)); the library's store %s\n", g_W, f0, g_D);

    /* R first, on the cold store */
    sec_begin();
    {
        vfft_config_t c;
        memset(&c, 0, sizeof c); c.n[0] = 64; c.transform = VFFT_R2C;
        refused("R2C", &c, "plans VFFT_C2C only, not VFFT_R2C");
        memset(&c, 0, sizeof c); c.n[0] = 64; c.transform = VFFT_C2R;
        refused("C2R", &c, "plans VFFT_C2C only, not VFFT_C2R");
        memset(&c, 0, sizeof c); c.n[0] = 64; c.transform = VFFT_DCT2;
        refused("DCT2", &c, "plans VFFT_C2C only, not VFFT_DCT2");
        memset(&c, 0, sizeof c); c.n[0] = 64; c.transform = VFFT_DHT;
        refused("DHT", &c, "plans VFFT_C2C only, not VFFT_DHT");
        memset(&c, 0, sizeof c); c.rank = 2; c.n[0] = 64; c.n[1] = 64;
        refused("rank 2", &c, "plans 1D transforms only, not rank 2");
        memset(&c, 0, sizeof c); c.n[0] = 64; c.howmany = 4;
        refused("howmany 4", &c, "plans one transform, not a batch of 4");
        memset(&c, 0, sizeof c); c.n[0] = 64; c.layout = VFFT_LAYOUT_SPLIT;
        refused("SPLIT", &c, "plans the INTERLEAVED layout only");
        memset(&c, 0, sizeof c); c.n[0] = 64; c.flags = VFFT_FLAG_RECALIBRATE;
        refused("with VFFT_FLAG_RECALIBRATE", &c, "VFFT_FLAG_ESTIMATE with VFFT_FLAG_RECALIBRATE");
        memset(&c, 0, sizeof c); c.n[0] = 64; c.flags = VFFT_FLAG_PATIENT;
        refused("with VFFT_FLAG_PATIENT", &c, "VFFT_FLAG_ESTIMATE with VFFT_FLAG_PATIENT");
        memset(&c, 0, sizeof c); c.n[0] = 1009; c.order = VFFT_ORDER_SCRAMBLED; c.placement = VFFT_PLACEMENT_IN;
        refused("N=1009 in place, scrambled, one thread", &c, "in place, scrambled, one thread");
    }
    sec_end("R");

    /* A1: natural, out of place */
    sec_begin();
    for (N = 2; N <= FULL_N; N += (N < 1024 ? 1 : 13))
    {
        ref_make(&r, N);
        cell(&r, 0, 0, 1);
        ref_free(&r);
    }
    sec_end("A1");

    /* A2: the four cells, 2..256 and the powers of two */
    sec_begin();
    for (N = 2; N <= 256; N++)
    {
        ref_make(&r, N);
        four(&r, 1);
        ref_free(&r);
    }
    for (N = 512; N <= (1L << 20); N *= 2)
    {
        ref_make(&r, N);
        four(&r, 1);
        ref_free(&r);
    }
    sec_end("A2");

    /* A3: 100 lengths to 2^20, a fixed seed: 70 any N past 256, 30 primes */
    sec_begin();
    for (i = 0; i < 100; i++)
    {
        z ^= z >> 12; z ^= z << 25; z ^= z >> 27;
        N = 257 + (long)((z * 2685821657736338717ull) % (unsigned long long)((1L << 20) - 257));
        if (i >= 70)
            while (!is_prime(N)) N = N > 257 ? N - 1 : 1048573;
        ref_make(&r, N);
        four(&r, 1);
        ref_free(&r);
    }
    sec_end("A3");

    /* A4: T = 8 */
    sec_begin();
    for (i = 0; i < 30; i++)
    {
        ref_make(&r, T8[i]);
        four(&r, 8);
        ref_free(&r);
    }
    sec_end("A4");

    /* W: the stores */
    sec_begin();
    d1 = dir_digest(g_W, &f1);
    e1 = dir_digest(g_D, &d1n);
    CHECK(f1 == f0 && d1 == d0, "W  the store named by wisdom_dir: %d file(s) before, %d after, %s", f0, f1,
          d1 == d0 ? "unchanged" : "CHANGED");
    CHECK(d0n == 0 && d1n == 0 && e0 == e1, "W  the library's own store (VFFT_WISDOM_DIR): %d file(s) after", d1n);
    sec_end("W");

    printf("%d checks, %d failed; %ld plans built and correct, %ld refused as the in-place scrambled prime cell at one "
           "thread\n%s\n", g_checks, g_fail, g_built, g_known, g_fail ? "*** ESTIMATE GATE FAILED ***" : "ESTIMATE GATE ALL PASS");
    return g_fail ? 1 : 0;
}
