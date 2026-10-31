/* estimate_gate.c — VFFT_FLAG_ESTIMATE (include/vfft.h; il/estimate/estimate_il.h,
 * il/estimate/c2c_estimate_il.h, and at rank 2 and 3 il/estimate/estimate_nd_il.h,
 * il/estimate/c2c_estimate_nd_il.h): an interleaved c2c transform planned by formulas, nothing measured,
 * and a contiguous batch of them on one thread.
 *
 * Every create goes through the public API with the flag and names the gate's scratch store:
 *   A  correct against a long-double DFT: every bin up to N = 4096 (scrambled: the sorted multiset of the
 *      outputs), above it 32 sampled bins (scrambled: each found among the outputs, the contract being any
 *      permutation), and the forward-backward round trip. The cells:
 *        A1 natural, out of place, every N in 2..1024 and every 13th N in 1025..4096;
 *        A2 the four order x placement cells at every N in 2..256 and every power of two to 2^20;
 *        A3 the four cells at 100 lengths to 2^20 drawn with a fixed seed, 70 any N and 30 primes;
 *        A4 the four cells at T = 8, 30 lengths of every route.
 *      Every cell builds, the in-place scrambled cell at one thread with a prime factor above 47
 *      included (its prime method row has its own key, role=method, so the cell replays).
 *   D2 rank 2, natural, one thread, both placements (docs/design/estimate_2d3d_sketch.md): every
 *      power-of-two N1 x N2 with sides 2..8192 and at most 2^18 points, the larger ones the rules branch on (the
 *      leading stage from 2048, the band, the turn, a row plan that is a four-step), odd and composite
 *      sides, primes up to 47, Rader columns (a prime above 47 whose p - 1 chains, nested too), Bluestein
 *      columns (the turned prime pass), prime rows. Correct against a long-double DFT: every bin while
 *      P (N1 + N2) stays under ND_FULL_WORK, else 32 sampled bins; and the round trip.
 *   D2T the D2 cells at T = 8 (est_mt_2d): built with the threaded verdict the recipe writes -- the
 *      estimate's plan check refuses a plan whose served verdict (cmt, mtarm, msw, nls) is not the recipe's --
 *      and correct the same way.
 *   D3 rank 3 the same way: every power-of-two N1 x N2 x N3 with sides 2..1024 and at most 2^12 points,
 *      then powers of two over every rule (the scrambled pass's exceptions, the band, the plane child's
 *      routes), odd shapes, Rader and Bluestein on axis 0 and inside the plane, prime rows; and the two
 *      natural forms on volumes past this machine's L3: a short axis 0 (16 N1 <= L1d / 4) the strips, a
 *      long one (N1 >= 1024) the cycle walk -- the estimate's plan check refuses a plan whose served form
 *      or strip width is not the recipe's, so built is served as the recipe says.
 *   D3T the D3 cells at T = 8 (est_mt_3d): built with the threaded verdict the recipe writes -- the plan check
 *      refuses a plan whose served verdict, structure, team, form or strip width is not the recipe's -- and
 *      correct the same way; with them a tall axis 0 no strip width fits (the cycle walk at T), a flat plane at
 *      T, and two cells at T = 4 (8x8x8, once refused under the flag, and 64x64x64).
 *   B  a contiguous batch on one thread (B1 rank 1 in the four order x placement cells, B2 and B3 rank 2 and 3
 *      in both placements; K = 2 and an odd K): the batch and the one-transform plan of the same request built
 *      with nothing measured, the one-transform plan correct as in A and D, and the batch on K blocks drawn
 *      apart bitwise that plan's output on each block, both directions -- the batch is the one-transform
 *      plan, run once per transform -- with its round trip within TOL.
 *   M  nothing measured at any create: the races counted (_vfft_create_race_count, vfft.c) and the race
 *      scopes entered (vfft_measure_scopes) do not move across it.
 *   R  the refusals: R2C, C2R, a real-to-real transform, rank 4, a batch on 8 threads and a lane-major batch
 *      (rank 1, 2 and 3), SPLIT, the flag with VFFT_FLAG_RECALIBRATE and with VFFT_FLAG_PATIENT;
 *      at rank 2 and 3 scrambled order, an axis of length 1, a last axis
 *      with no 1D engine, a column whose Bluestein length passes 2^30 (refused at once: the create
 *      returns): NULL, vfft_error() naming the reason, nothing measured.
 *   E  the environment guard: a variable that steers a plan (VFFT_IL_KV at rank 1, VFFT_IL2D_WL at rank 2,
 *      VFFT_ILND_NF at rank 3, which pins the natural form, VFFT_ILND_MT at rank 3 on 8 threads, which pins
 *      the threaded verdict) refuses the create naming it; a logging one
 *      (VFFT_IL2D_LOG) does not; unset again, the same create builds.
 *   W  no wisdom store touched: the store named by wisdom_dir keeps its files (none created), and the
 *      library's own store (VFFT_WISDOM_DIR, pointed at a scratch folder before the first create) gets none.
 * Usage: estimate_gate --wisdir <cold scratch dir> */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "vfft_diagnostics.h"
#include "cpu_cache.h"   /* this machine's L1d and L3, as the estimate reads them: the natural-form cells */
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
static long g_built = 0;
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

/* one cell: built and correct */
static void cell(const ref_t *r, int scr, int ip, int T)
{
    const long N = r->N;
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
        CHECK(0, "A  %s: refused: %s", name, vfft_error());
        return;
    }
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
static long s_built;
static void sec_begin(void) { s_checks = g_checks; s_fail = g_fail; s_built = g_built; }
static void sec_end(const char *name)
{
    const int f = g_fail - s_fail;
    printf("SECTION %-3s: %d checks, %d failed; %ld plans built and correct -> %s\n", name, g_checks - s_checks, f,
           g_built - s_built, f ? "FAIL" : "PASS");
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


/* ── rank 2 and 3 ── */
#define ND_FULL_WORK 64000000L   /* every bin while P (N1 + N2 + N3) stays under this; else NBINS sampled bins */
typedef struct
{
    int rank, n[3];        /* n[2] = 1 at rank 2 */
    long P;
    double *x;             /* the input, complex interleaved, row-major, the last axis contiguous */
    long double sx;        /* sum |x|^2 */
    long double *Xr, *Xi;  /* full: every bin */
    long k[NBINS][3];      /* sampled: the bins */
    long double Kr[NBINS], Ki[NBINS];
    int nk;
} ndref_t;
/* the forward DFT along one axis of length L at stride S (the product of the later extents), in place
 * on the long-double planes */
static void nd_axis(long double *re, long double *im, long P, long L, long S)
{
    long double *c = (long double *)malloc((size_t)L * sizeof *c), *sn = (long double *)malloc((size_t)L * sizeof *sn);
    long double *tr = (long double *)malloc((size_t)L * sizeof *tr), *ti = (long double *)malloc((size_t)L * sizeof *ti);
    long o, in, j, f;
    if (!c || !sn || !tr || !ti) { printf("out of memory\n"); exit(2); }
    for (j = 0; j < L; j++) { c[j] = cosl(2.0L * PI_L * j / L); sn[j] = -sinl(2.0L * PI_L * j / L); }
    for (o = 0; o < P / (L * S); o++)
        for (in = 0; in < S; in++)
        {
            const long base = o * L * S + in;
            for (f = 0; f < L; f++)
            {
                long double sr = 0.0L, si = 0.0L;
                long idx = 0;
                for (j = 0; j < L; j++)
                {
                    const long double xr = re[base + j * S], xi = im[base + j * S];
                    sr += xr * c[idx] - xi * sn[idx];
                    si += xr * sn[idx] + xi * c[idx];
                    idx += f;
                    if (idx >= L) idx -= L;
                }
                tr[f] = sr;
                ti[f] = si;
            }
            for (f = 0; f < L; f++) { re[base + f * S] = tr[f]; im[base + f * S] = ti[f]; }
        }
    free(c); free(sn); free(tr); free(ti);
}
/* exp(-2 pi i (j k mod L) / L) for j < L into wr, wi, blocked as dft_one blocks it: j = a B + b, so a long
 * axis needs B + L / B long-double twiddles */
static void nd_twiddles(long double *wr, long double *wi, long L, long k)
{
    enum { B = 1024 };
    const long nb = (L + B - 1) / B;
    long double *ic = (long double *)malloc(2 * B * sizeof *ic), *oc = (long double *)malloc((size_t)(2 * nb) * sizeof *oc);
    long a, b;
    if (!ic || !oc) { printf("out of memory\n"); exit(2); }
    for (b = 0; b < B; b++)
    {
        const long m = (long)(((unsigned long long)b * (unsigned long long)k) % (unsigned long long)L);
        ic[2 * b] = cosl(2.0L * PI_L * m / L);
        ic[2 * b + 1] = -sinl(2.0L * PI_L * m / L);
    }
    for (a = 0; a < nb; a++)
    {
        const long m = (long)((((unsigned long long)a * B) % (unsigned long long)L) * (unsigned long long)k % (unsigned long long)L);
        oc[2 * a] = cosl(2.0L * PI_L * m / L);
        oc[2 * a + 1] = -sinl(2.0L * PI_L * m / L);
    }
    for (a = 0; a < nb; a++)
        for (b = 0; b < B && a * B + b < L; b++)
        {
            const long j = a * B + b;
            wr[j] = oc[2 * a] * ic[2 * b] - oc[2 * a + 1] * ic[2 * b + 1];
            wi[j] = oc[2 * a] * ic[2 * b + 1] + oc[2 * a + 1] * ic[2 * b];
        }
    free(ic);
    free(oc);
}
/* bin (k0, k1, k2) of the forward: the sum over the volume with one twiddle table per axis */
static void nd_bin(const ndref_t *r, const long *k, long double *orr, long double *oi)
{
    long double *w[3][2];
    long double ur = 0.0L, ui = 0.0L;
    long a, i0, i1, i2;
    for (a = 0; a < 3; a++)
    {
        const long L = r->n[a];
        w[a][0] = (long double *)malloc((size_t)L * sizeof(long double));
        w[a][1] = (long double *)malloc((size_t)L * sizeof(long double));
        if (!w[a][0] || !w[a][1]) { printf("out of memory\n"); exit(2); }
        nd_twiddles(w[a][0], w[a][1], L, k[a]);
    }
    for (i0 = 0; i0 < r->n[0]; i0++)
    {
        long double tr = 0.0L, ti = 0.0L;
        for (i1 = 0; i1 < r->n[1]; i1++)
        {
            const double *x = r->x + 2 * ((size_t)i0 * r->n[1] + i1) * r->n[2];
            long double sr = 0.0L, si = 0.0L;
            for (i2 = 0; i2 < r->n[2]; i2++)
            {
                const long double xr = x[2 * i2], xi = x[2 * i2 + 1], cr = w[2][0][i2], ci = w[2][1][i2];
                sr += xr * cr - xi * ci;
                si += xr * ci + xi * cr;
            }
            tr += sr * w[1][0][i1] - si * w[1][1][i1];
            ti += sr * w[1][1][i1] + si * w[1][0][i1];
        }
        ur += tr * w[0][0][i0] - ti * w[0][1][i0];
        ui += tr * w[0][1][i0] + ti * w[0][0][i0];
    }
    *orr = ur;
    *oi = ui;
    for (a = 0; a < 3; a++) { free(w[a][0]); free(w[a][1]); }
}
static void ndref_make(ndref_t *r, int rank, const int *n)
{
    long i;
    int a;
    memset(r, 0, sizeof *r);
    r->rank = rank;
    r->n[0] = n[0]; r->n[1] = n[1]; r->n[2] = rank == 3 ? n[2] : 1;
    r->P = (long)r->n[0] * r->n[1] * r->n[2];
    r->x = (double *)malloc((size_t)2 * r->P * sizeof(double));
    if (!r->x) { printf("out of memory\n"); exit(2); }
    g_rng = 0x9E3779B97F4A7C15ull ^ ((unsigned long long)r->n[0] << 40) ^ ((unsigned long long)r->n[1] << 20) ^ (unsigned long long)r->n[2];
    for (i = 0; i < 2 * r->P; i++) { r->x[i] = rnd(); r->sx += (long double)r->x[i] * r->x[i]; }
    if ((long long)r->P * ((long long)r->n[0] + r->n[1] + r->n[2]) <= ND_FULL_WORK)   /* long is 32 bits on MinGW */
    {
        r->Xr = (long double *)malloc((size_t)r->P * sizeof(long double));
        r->Xi = (long double *)malloc((size_t)r->P * sizeof(long double));
        if (!r->Xr || !r->Xi) { printf("out of memory\n"); exit(2); }
        for (i = 0; i < r->P; i++) { r->Xr[i] = r->x[2 * i]; r->Xi[i] = r->x[2 * i + 1]; }
        nd_axis(r->Xr, r->Xi, r->P, r->n[2], 1);
        nd_axis(r->Xr, r->Xi, r->P, r->n[1], r->n[2]);
        nd_axis(r->Xr, r->Xi, r->P, r->n[0], (long)r->n[1] * r->n[2]);
        return;
    }
    {   /* the corners, the halves, then bins drawn with a fixed seed */
        unsigned long long z = 0xD1B54A32D192ED03ull ^ (unsigned long long)r->P;
        int j;
        for (a = 0; a < 3; a++)
        {
            r->k[0][a] = 0; r->k[1][a] = a == 0; r->k[2][a] = a == 1; r->k[3][a] = a == 2;
            r->k[4][a] = r->n[a] - 1; r->k[5][a] = r->n[a] / 2;
            for (j = 0; j < 6; j++) r->k[j][a] %= r->n[a];   /* an axis of one point (rank 2's third) has bin 0 */
        }
        for (r->nk = 6; r->nk < NBINS; r->nk++)
            for (a = 0; a < 3; a++)
            {
                z ^= z >> 12; z ^= z << 25; z ^= z >> 27;
                r->k[r->nk][a] = (long)((z * 2685821657736338717ull) % (unsigned long long)r->n[a]);
            }
        for (j = 0; j < r->nk; j++)
            nd_bin(r, r->k[j], &r->Kr[j], &r->Ki[j]);
    }
}
static void ndref_free(ndref_t *r)
{
    free(r->x);
    free(r->Xr);
    free(r->Xi);
    memset(r, 0, sizeof *r);
}
/* the forward's error against the reference and the round trip's, both relative (a NaN wins) */
static void nd_errors(vfft_plan p, const ndref_t *r, int ip, double *ebin, double *ert)
{
    const long P = r->P;
    const size_t nb = (size_t)2 * P * sizeof(double);
    double *in = (double *)vfft_malloc(nb), *out = ip ? in : (double *)vfft_malloc(nb), *z = ip ? in : (double *)vfft_malloc(nb);
    const long double rms = sqrtl(r->sx);   /* the rms of |X| (Parseval) */
    long double m = 0.0L;
    long i;
    if (!in || !out || !z) { printf("out of memory\n"); exit(2); }
    memcpy(in, r->x, nb);
    vfft_execute(p, VFFT_FORWARD, in, out);
    if (r->Xr)
        for (i = 0; i < P; i++)
        {
            const long double d = hypotl(out[2 * i] - r->Xr[i], out[2 * i + 1] - r->Xi[i]);
            if (!(d <= m)) m = d;
        }
    else
        for (i = 0; i < r->nk; i++)
        {
            const long at = (r->k[i][0] * r->n[1] + r->k[i][1]) * r->n[2] + r->k[i][2];
            const long double d = hypotl(out[2 * at] - r->Kr[i], out[2 * at + 1] - r->Ki[i]);
            if (!(d <= m)) m = d;
        }
    *ebin = (double)(m / rms);
    vfft_execute(p, VFFT_BACKWARD, out, z);
    m = 0.0L;
    for (i = 0; i < 2 * P; i++)
    {
        const long double d = fabsl((long double)z[i] / P - r->x[i]);
        if (!(d <= m)) m = d;
    }
    *ert = (double)(m / sqrtl(r->sx / P));
    vfft_free(in);
    if (!ip) { vfft_free(out); vfft_free(z); }
}
static long g_nd_built = 0;
static double g_nd_eb = 0.0, g_nd_er = 0.0;   /* a section's worst forward and round-trip errors */
/* one rank-2 or rank-3 cell at T threads in both placements: built, nothing measured, correct */
static void nd_cell(int rank, int n0, int n1, int n2, int T)
{
    const int n[3] = { n0, n1, n2 };
    ndref_t r;
    int ip;
    ndref_make(&r, rank, n);
    for (ip = 0; ip < 2; ip++)
    {
        char name[96];
        vfft_config_t c;
        vfft_plan p;
        int measured;
        if (rank == 3) snprintf(name, sizeof name, "%dx%dx%d %s T=%d", n0, n1, n2, ip ? "ip" : "oop", T);
        else snprintf(name, sizeof name, "%dx%d %s T=%d", n0, n1, ip ? "ip" : "oop", T);
        memset(&c, 0, sizeof c);
        c.rank = rank;
        c.n[0] = n0; c.n[1] = n1; c.n[2] = rank == 3 ? n2 : 0;
        c.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
        c.nthreads = T;
        p = est_create(&c, &measured);
        CHECK(!measured, "M  %s: the create MEASURED", name);
        CHECK(p != NULL, "D  %s: refused: %s", name, vfft_error());
        if (!p)
            continue;
        {
            double eb, er;
            nd_errors(p, &r, ip, &eb, &er);
            CHECK(eb <= TOL && er <= TOL, "D  %s: forward %.2e, round trip %.2e (tolerance %.0e, %s)", name, eb, er, TOL,
                  r.Xr ? "every bin" : "sampled bins");
            if (!(eb <= g_nd_eb)) g_nd_eb = eb;
            if (!(er <= g_nd_er)) g_nd_er = er;
            g_nd_built++;
        }
        vfft_destroy(p);
    }
    ndref_free(&r);
}
static void sec_end_nd(const char *name, long b0)
{
    const int f = g_fail - s_fail;
    printf("SECTION %-3s: %d checks, %d failed; %ld plans built and correct, worst forward %.1e, worst round trip %.1e "
           "-> %s\n", name, g_checks - s_checks, f, g_nd_built - b0, g_nd_eb, g_nd_er, f ? "FAIL" : "PASS");
    g_nd_eb = g_nd_er = 0.0;
}
/* ── B: contiguous batches on one thread ── */
static long g_b_built = 0;
/* a batch of K transforms of len complex points (pb) against the one-transform plan of the same request (p1):
 * K blocks drawn with a seed of their own, the one-transform plan run on an aligned copy of each block; both
 * directions bitwise its output (the batch is that plan, run once per transform), the round trip within TOL */
static void batch_check(const char *name, vfft_plan pb, vfft_plan p1, long len, int K, int ip)
{
    const size_t nd = (size_t)2 * len, nb = nd * sizeof(double), all = nb * (size_t)K;
    double *x = (double *)vfft_malloc(all), *y = (double *)vfft_malloc(all), *z = ip ? y : (double *)vfft_malloc(all);
    double *w = (double *)vfft_malloc(all), *a = (double *)vfft_malloc(nb), *b = ip ? a : (double *)vfft_malloc(nb);
    long double sx = 0.0L, m = 0.0L;
    double er;
    size_t i;
    int t, fsame, bsame;
    if (!x || !y || !z || !w || !a || !b) { printf("out of memory\n"); exit(2); }
    g_rng = 0xA0761D6478BD642Full ^ ((unsigned long long)len << 8) ^ (unsigned long long)K;
    for (i = 0; i < nd * K; i++) { x[i] = rnd(); sx += (long double)x[i] * x[i]; }
    for (t = 0; t < K; t++)   /* the one-transform plan's forward, block by block */
    {
        memcpy(a, x + nd * t, nb);
        vfft_execute(p1, VFFT_FORWARD, a, b);
        memcpy(w + nd * t, b, nb);
    }
    if (ip) memcpy(y, x, all);
    vfft_execute(pb, VFFT_FORWARD, ip ? y : x, y);
    fsame = !memcmp(w, y, all);
    for (t = 0; t < K; t++)   /* its backward of the batch's forward */
    {
        memcpy(a, y + nd * t, nb);
        vfft_execute(p1, VFFT_BACKWARD, a, b);
        memcpy(w + nd * t, b, nb);
    }
    vfft_execute(pb, VFFT_BACKWARD, y, z);
    bsame = !memcmp(w, z, all);
    for (i = 0; i < nd * K; i++)
    {
        const long double d = fabsl((long double)z[i] / len - x[i]);
        if (!(d <= m)) m = d;
    }
    er = (double)(m / sqrtl(sx / ((long double)len * K)));
    CHECK(fsame && bsame && er <= TOL, "B  %s: the batch's forward %s and its backward %s the one-transform plan's, "
          "block by block; round trip %.2e", name, fsame ? "is" : "is NOT", bsame ? "is" : "is NOT", er);
    g_b_built += fsame && bsame && er <= TOL;
    vfft_free(x);
    vfft_free(y);
    vfft_free(w);
    vfft_free(a);
    if (!ip) { vfft_free(z); vfft_free(b); }
}
/* the one-transform plan against its reference: errors() at rank 1, nd_errors() at rank 2 and 3 */
typedef void (*correct_fn)(vfft_plan p, const void *ref, int scr, int ip, double *eb, double *er);
static void b1_correct(vfft_plan p, const void *r, int scr, int ip, double *eb, double *er)
{
    errors(p, (const ref_t *)r, scr, ip, eb, er);
}
static void bnd_correct(vfft_plan p, const void *r, int scr, int ip, double *eb, double *er)
{
    (void)scr;
    nd_errors(p, (const ndref_t *)r, ip, eb, er);
}
/* the request c as one transform and as a batch of K, both built with nothing measured; the one-transform plan
 * correct against the reference, the batch against it */
static void batch_cell(const char *name, vfft_config_t c, long len, int K, int scr, int ip, correct_fn correct,
                       const void *ref)
{
    vfft_plan p1, pb;
    int m1, mb;
    c.howmany = 1;
    p1 = est_create(&c, &m1);
    CHECK(p1 && !m1, "B  %s: the one-transform plan %s", name, m1 ? "MEASURED" : vfft_error());
    c.howmany = (size_t)K;
    pb = est_create(&c, &mb);
    CHECK(pb && !mb, "B  %s: the batch %s", name, mb ? "MEASURED" : vfft_error());
    if (p1 && pb && !m1 && !mb)
    {
        double eb, er;
        correct(p1, ref, scr, ip, &eb, &er);
        CHECK(eb <= TOL && er <= TOL, "B  %s: the one-transform plan: forward %.2e, round trip %.2e (tolerance %.0e)", name,
              eb, er, TOL);
        batch_check(name, pb, p1, len, K, ip);
    }
    vfft_destroy(pb);
    vfft_destroy(p1);
}
/* B1: one length in the four cells at K = 2 and K = 5 (K = 2 alone past 2^18 points) */
static void b1_len(long N)
{
    ref_t r;
    int scr, ip, k;
    static const int Ks[2] = { 2, 5 };
    ref_make(&r, N);
    for (scr = 0; scr < 2; scr++)
        for (ip = 0; ip < 2; ip++)
            for (k = 0; k < (N <= (1L << 18) ? 2 : 1); k++)
            {
                char name[96];
                vfft_config_t c;
                snprintf(name, sizeof name, "N=%ld K=%d %s %s", N, Ks[k], scr ? "scr" : "nat", ip ? "ip" : "oop");
                memset(&c, 0, sizeof c);
                c.n[0] = (int)N;
                c.order = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
                c.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
                batch_cell(name, c, N, Ks[k], scr, ip, b1_correct, &r);
            }
    ref_free(&r);
}
/* B2, B3: one shape in both placements at K = 2 and K = 3 */
static void bnd_shape(int rank, int n0, int n1, int n2)
{
    const int n[3] = { n0, n1, n2 };
    ndref_t r;
    int ip, K;
    ndref_make(&r, rank, n);
    for (ip = 0; ip < 2; ip++)
        for (K = 2; K <= 3; K++)
        {
            char name[96];
            vfft_config_t c;
            if (rank == 3) snprintf(name, sizeof name, "%dx%dx%d K=%d %s", n0, n1, n2, K, ip ? "ip" : "oop");
            else snprintf(name, sizeof name, "%dx%d K=%d %s", n0, n1, K, ip ? "ip" : "oop");
            memset(&c, 0, sizeof c);
            c.rank = rank;
            c.n[0] = n0; c.n[1] = n1; c.n[2] = rank == 3 ? n2 : 0;
            c.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
            batch_cell(name, c, r.P, K, 0, ip, bnd_correct, &r);
        }
    ndref_free(&r);
}
static void sec_end_b(const char *name, long b0)
{
    const int f = g_fail - s_fail;
    printf("SECTION %-3s: %d checks, %d failed; %ld batches built, each the one-transform plan run once per transform "
           "-> %s\n", name, g_checks - s_checks, f, g_b_built - b0, f ? "FAIL" : "PASS");
}

/* the environment, set or cleared */
static void env_set(const char *name, const char *val)
{
#ifdef _WIN32
    _putenv_s(name, val ? val : "");
#else
    if (val) setenv(name, val, 1);
    else unsetenv(name);
#endif
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
    int a, i, f0, f1, d0n, d1n, T2, T3;
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
    printf("ESTIMATE GATE: VFFT_FLAG_ESTIMATE, C2C at rank 1, 2 and 3; store %s (%d file(s)); the library's store %s\n",
           g_W, f0, g_D);

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
        memset(&c, 0, sizeof c); c.rank = 4; c.n[0] = 8; c.n[1] = 8; c.n[2] = 8; c.n[3] = 8;
        refused("rank 4", &c, "plans ranks 1 to 3, not rank 4");
        memset(&c, 0, sizeof c); c.rank = 2; c.n[0] = 64; c.n[1] = 64; c.order = VFFT_ORDER_SCRAMBLED;
        refused("rank 2 scrambled", &c, "plans rank 2 and 3 in natural order only, not VFFT_ORDER_SCRAMBLED");
        memset(&c, 0, sizeof c); c.rank = 3; c.n[0] = 8; c.n[1] = 8; c.n[2] = 8; c.order = VFFT_ORDER_SCRAMBLED;
        refused("rank 3 scrambled", &c, "plans rank 2 and 3 in natural order only, not VFFT_ORDER_SCRAMBLED");
        memset(&c, 0, sizeof c); c.rank = 2; c.n[0] = 64; c.n[1] = 64; c.howmany = 4; c.nthreads = 8;
        refused("rank 2 howmany 4 T=8", &c, "plans a batch on one thread only, not a batch of 4 on 8 threads");
        memset(&c, 0, sizeof c); c.rank = 3; c.n[0] = 8; c.n[1] = 8; c.n[2] = 8; c.howmany = 2; c.nthreads = 8;
        refused("rank 3 howmany 2 T=8", &c, "plans a batch on one thread only, not a batch of 2 on 8 threads");
        memset(&c, 0, sizeof c); c.rank = 2; c.n[0] = 64; c.n[1] = 64; c.howmany = 4; c.batch = VFFT_BATCH_LANE_MAJOR;
        refused("rank 2 howmany 4 lane-major", &c, "plans a contiguous batch only, not VFFT_BATCH_LANE_MAJOR");
        memset(&c, 0, sizeof c); c.rank = 3; c.n[0] = 8; c.n[1] = 8; c.n[2] = 8; c.howmany = 2; c.batch = VFFT_BATCH_LANE_MAJOR;
        refused("rank 3 howmany 2 lane-major", &c, "plans a contiguous batch only, not VFFT_BATCH_LANE_MAJOR");
        memset(&c, 0, sizeof c); c.rank = 2; c.n[0] = 1; c.n[1] = 64;
        refused("rank 2 1x64", &c, "every axis of at least 2 points");
        memset(&c, 0, sizeof c); c.rank = 3; c.n[0] = 8; c.n[1] = 1; c.n[2] = 8;
        refused("rank 3 8x1x8", &c, "every axis of at least 2 points");
        memset(&c, 0, sizeof c); c.rank = 2; c.n[0] = 2; c.n[1] = 16777218;
        refused("rank 2 2x16777218", &c, "have no 1D engine");
        /* 53 * 59 * 2^18: no column chain, its Bluestein length 2^31 (the create used to loop forever) */
        memset(&c, 0, sizeof c); c.rank = 2; c.n[0] = 819724288; c.n[1] = 2;
        refused("rank 2 819724288x2", &c, "N = 819724288 has no column chain and its Bluestein length 2147483648 none either");
        memset(&c, 0, sizeof c); c.n[0] = 64; c.howmany = 4; c.nthreads = 8;
        refused("howmany 4 T=8", &c, "plans a batch on one thread only, not a batch of 4 on 8 threads");
        memset(&c, 0, sizeof c); c.n[0] = 64; c.howmany = 4; c.batch = VFFT_BATCH_LANE_MAJOR;
        refused("howmany 4 lane-major", &c, "plans a contiguous batch only, not VFFT_BATCH_LANE_MAJOR");
        memset(&c, 0, sizeof c); c.n[0] = 64; c.layout = VFFT_LAYOUT_SPLIT;
        refused("SPLIT", &c, "plans the INTERLEAVED layout only");
        memset(&c, 0, sizeof c); c.n[0] = 64; c.flags = VFFT_FLAG_RECALIBRATE;
        refused("with VFFT_FLAG_RECALIBRATE", &c, "VFFT_FLAG_ESTIMATE with VFFT_FLAG_RECALIBRATE");
        memset(&c, 0, sizeof c); c.n[0] = 64; c.flags = VFFT_FLAG_PATIENT;
        refused("with VFFT_FLAG_PATIENT", &c, "VFFT_FLAG_ESTIMATE with VFFT_FLAG_PATIENT");
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

    /* D2: rank 2, natural, one thread, both placements; D2T the same cells at T = 8 */
    for (T2 = 1; T2 <= 8; T2 += 7)
    {
        sec_begin();
        static const int D2[][2] = {
            /* the larger powers of two: the leading stage, the band, the turn, a four-step row plan */
            { 512, 512 }, { 2048, 4 }, { 2048, 64 }, { 4096, 16 }, { 8192, 8 }, { 1024, 1024 }, { 64, 8192 },
            { 4096, 256 }, { 2, 262144 }, { 2, 1048576 },
            /* odd and composite sides */
            { 2, 3 }, { 5, 2 }, { 3, 5 }, { 6, 10 }, { 9, 9 }, { 12, 12 }, { 15, 27 }, { 27, 32 }, { 45, 16 }, { 63, 63 },
            { 96, 96 }, { 100, 100 }, { 135, 64 }, { 1000, 8 }, { 7, 128 }, { 30, 30 }, { 24, 40 }, { 2, 786432 },
            /* primes up to 47 */
            { 47, 64 }, { 31, 31 }, { 2, 47 }, { 17, 17 }, { 64, 47 }, { 43, 128 },
            /* Rader columns: a prime above 47 whose p - 1 chains, alone, with a rest, nested (107: 53.2) */
            { 53, 64 }, { 131, 16 }, { 107, 8 }, { 191, 4 }, { 262, 32 }, { 524, 64 }, { 2101, 16 }, { 101, 129 },
            { 53, 53 }, { 127, 64 }, { 139, 48 },
            /* Rader primes whose p - 1 is a multiple of 64 (192, 256, 12288): the engine's pick puts a smaller radix
             * first (3.64, 4.64, 3.64.64) */
            { 193, 8 }, { 257, 4 }, { 771, 2 }, { 12289, 2 },
            /* Bluestein columns, the turned prime pass: 643 (642 = 2.3.107, 106 = 2.53 does not chain), 53^2, 53.59 */
            { 643, 8 }, { 2809, 4 }, { 3127, 2 }, { 643, 100 },
            /* prime rows */
            { 8, 1009 }, { 16, 4099 }, { 4, 65537 },
        };
        const long b0 = g_nd_built;
        int x, y;
        for (x = 1; x <= 13; x++)   /* every power-of-two N1 x N2, sides 2..8192, at most 2^18 points */
            for (y = 1; y <= 13; y++)
                if (x + y <= 18)
                    nd_cell(2, 1 << x, 1 << y, 0, T2);
        for (i = 0; i < (int)(sizeof D2 / sizeof D2[0]); i++)
            nd_cell(2, D2[i][0], D2[i][1], 0, T2);
        sec_end_nd(T2 > 1 ? "D2T" : "D2", b0);
    }

    /* D3: rank 3, natural, one thread, both placements; D3T the same cells at T = 8 */
    for (T3 = 1; T3 <= 8; T3 += 7)
    {
        sec_begin();
        static const int D3[][3] = {
            /* powers of two: the scrambled pass's exceptions (512 narrow, 1024 / 4096 / 8192 wide, 8192 narrow),
             * the lead (2048), the band, the plane child's routes, a large plane, long rows */
            { 2, 2, 2 }, { 2, 4, 8 }, { 4, 8, 16 }, { 8, 8, 8 }, { 16, 16, 16 }, { 32, 32, 32 }, { 64, 64, 64 }, { 128, 8, 8 },
            { 256, 16, 16 }, { 512, 4, 4 }, { 1024, 16, 16 }, { 2048, 2, 2 }, { 4096, 16, 16 }, { 8192, 4, 4 },
            { 8192, 16, 16 }, { 2, 512, 512 }, { 16, 128, 128 }, { 4, 4, 4096 }, { 8, 2, 2048 },
            /* odd shapes */
            { 9, 15, 27 }, { 15, 15, 15 }, { 63, 8, 8 }, { 5, 7, 9 }, { 3, 3, 3 }, { 6, 10, 14 }, { 100, 3, 5 },
            /* Rader and Bluestein on axis 0 */
            { 53, 8, 8 }, { 131, 4, 4 }, { 107, 4, 4 }, { 262, 4, 8 }, { 101, 16, 16 }, { 257, 2, 4 }, { 643, 4, 4 },
            { 2809, 2, 2 },
            /* Rader and Bluestein inside the plane, prime rows */
            { 4, 53, 8 }, { 4, 643, 2 }, { 2, 101, 129 }, { 8, 8, 131 }, { 2, 131, 256 },
        };
        const long b0 = g_nd_built;
        int x, y, zz;
        for (x = 1; x <= 10; x++)   /* every power-of-two N1 x N2 x N3, sides 2..1024, at most 2^12 points */
            for (y = 1; y <= 10; y++)
                for (zz = 1; zz <= 10; zz++)
                    if (x + y + zz <= 12)
                        nd_cell(3, 1 << x, 1 << y, 1 << zz, T3);
        for (i = 0; i < (int)(sizeof D3 / sizeof D3[0]); i++)
            nd_cell(3, D3[i][0], D3[i][1], D3[i][2], T3);
        if (T3 > 1)
        {   /* at T: a tall axis 0 no strip width fits (16 N1 8 past the L2: the cycle walk), a flat plane (s=2);
             * at T = 4 the cell the flag once refused (8x8x8) and one that threads */
            nd_cell(3, 32768, 2, 2, T3);
            nd_cell(3, 8, 240, 240, T3);
            nd_cell(3, 8, 8, 8, 4);
            nd_cell(3, 64, 64, 64, 4);
        }
        {   /* the natural form (sketch §9) on volumes past this machine's L3 (none reported: past 32 MiB):
             * axis 0 short, 16 N1 <= L1d / 4, the strips; axis 0 long, N1 >= 1024 and 16 N1 > L1d / 4, the cycle walk
             * at one thread, the strips at T (the door's form wherever its strip pool is not empty) */
            const long l1d = vfft_cpu_l1d_bytes(), l3 = vfft_cpu_l3_bytes();
            const long long past = l3 > 0 ? (long long)l3 : 32LL << 20;
            int s3 = 2, c1 = 1024, c3 = 2;
            while (16LL * 64 * 512 * s3 <= past) s3 *= 2;
            while (16LL * c1 <= (long long)l1d / 4) c1 *= 2;
            while (16LL * c1 * 64 * c3 <= past) c3 *= 2;
            printf("  D3%s the natural form past the L3 (%ld bytes; L1d %ld): 64x512x%d (%lld MiB) the strips, %dx64x%d "
                   "(%lld MiB) %s\n", T3 > 1 ? "T" : "", l3, l1d, s3, (16LL * 64 * 512 * s3) >> 20, c1, c3,
                   (16LL * c1 * 64 * c3) >> 20, T3 > 1 ? "the strips (the cycle walk at one thread)" : "the cycle walk");
            nd_cell(3, 64, 512, s3, T3);
            nd_cell(3, c1, 64, c3, T3);
        }
        sec_end_nd(T3 > 1 ? "D3T" : "D3", b0);
    }

    /* B1, B2, B3: contiguous batches on one thread, over the routes of A and the shapes of D */
    {
        static const long B1[] = { 2, 3, 16, 64, 100, 997, 1000, 1009, 2048, 4096, 4099, 6000, 12288, 50000, 65536, 65537,
                                   131072, 300000, 1048576 };
        static const int B2[][2] = { { 2, 2 }, { 8, 8 }, { 16, 16 }, { 64, 64 }, { 512, 512 }, { 2048, 4 }, { 4096, 16 },
                                     { 64, 8192 }, { 2, 262144 }, { 9, 9 }, { 15, 27 }, { 100, 100 }, { 47, 64 }, { 53, 64 },
                                     { 131, 16 }, { 643, 8 }, { 2809, 4 }, { 8, 1009 }, { 16, 4099 } };
        static const int B3[][3] = { { 2, 2, 2 }, { 4, 8, 16 }, { 8, 8, 8 }, { 16, 16, 16 }, { 64, 64, 64 }, { 128, 8, 8 },
                                     { 1024, 16, 16 }, { 2048, 2, 2 }, { 2, 512, 512 }, { 8, 2, 2048 }, { 9, 15, 27 },
                                     { 5, 7, 9 }, { 100, 3, 5 }, { 53, 8, 8 }, { 643, 4, 4 }, { 4, 53, 8 }, { 8, 8, 131 },
                                     { 2, 131, 256 } };
        long b0 = g_b_built;
        sec_begin();
        for (i = 0; i < (int)(sizeof B1 / sizeof B1[0]); i++)
            b1_len(B1[i]);
        sec_end_b("B1", b0);
        b0 = g_b_built;
        sec_begin();
        for (i = 0; i < (int)(sizeof B2 / sizeof B2[0]); i++)
            bnd_shape(2, B2[i][0], B2[i][1], 0);
        sec_end_b("B2", b0);
        b0 = g_b_built;
        sec_begin();
        for (i = 0; i < (int)(sizeof B3 / sizeof B3[0]); i++)
            bnd_shape(3, B3[i][0], B3[i][1], B3[i][2]);
        sec_end_b("B3", b0);
    }

    /* E: the environment guard; each variable unset again before the next create */
    sec_begin();
    {
        vfft_config_t c;
        vfft_plan p;
        int measured;
        env_set("VFFT_IL_KV", "51");
        memset(&c, 0, sizeof c); c.n[0] = 64;
        refused("1D with VFFT_IL_KV", &c, "the developer switch VFFT_IL_KV is set");
        env_set("VFFT_IL_KV", NULL);
        env_set("VFFT_IL2D_WL", "8");
        memset(&c, 0, sizeof c); c.rank = 2; c.n[0] = 64; c.n[1] = 64;
        refused("2D with VFFT_IL2D_WL", &c, "the developer switch VFFT_IL2D_WL is set");
        env_set("VFFT_IL2D_WL", NULL);
        env_set("VFFT_ILND_NF", "2");   /* would pin the strips over the cycle walk the estimate plans here */
        memset(&c, 0, sizeof c); c.rank = 3; c.n[0] = 64; c.n[1] = 64; c.n[2] = 64;
        refused("3D with VFFT_ILND_NF", &c, "the developer switch VFFT_ILND_NF is set");
        env_set("VFFT_ILND_NF", NULL);
        env_set("VFFT_ILND_MT", "0");   /* would pin serial over the plane arm the estimate plans here at T = 8 */
        memset(&c, 0, sizeof c); c.rank = 3; c.n[0] = 64; c.n[1] = 64; c.n[2] = 64; c.nthreads = 8;
        refused("3D T=8 with VFFT_ILND_MT", &c, "the developer switch VFFT_ILND_MT is set");
        env_set("VFFT_ILND_MT", NULL);
        env_set("VFFT_IL2D_LOG", "1");   /* logging only: served */
        memset(&c, 0, sizeof c); c.rank = 2; c.n[0] = 16; c.n[1] = 16;
        p = est_create(&c, &measured);
        CHECK(p && !measured, "E  2D with VFFT_IL2D_LOG (logging only): %s", p ? "MEASURED" : vfft_error());
        vfft_destroy(p);
        env_set("VFFT_IL2D_LOG", NULL);
        memset(&c, 0, sizeof c); c.rank = 2; c.n[0] = 64; c.n[1] = 64;   /* unset again: served */
        p = est_create(&c, &measured);
        CHECK(p && !measured, "E  2D after the variables were unset: %s", p ? "MEASURED" : vfft_error());
        vfft_destroy(p);
    }
    sec_end("E");

    /* W: the stores */
    sec_begin();
    d1 = dir_digest(g_W, &f1);
    e1 = dir_digest(g_D, &d1n);
    CHECK(f1 == f0 && d1 == d0, "W  the store named by wisdom_dir: %d file(s) before, %d after, %s", f0, f1,
          d1 == d0 ? "unchanged" : "CHANGED");
    CHECK(d0n == 0 && d1n == 0 && e0 == e1, "W  the library's own store (VFFT_WISDOM_DIR): %d file(s) after", d1n);
    sec_end("W");

    printf("%d checks, %d failed; %ld 1D and %ld rank-2/3 plans built and correct, %ld batches on one thread\n%s\n",
           g_checks, g_fail, g_built, g_nd_built, g_b_built, g_fail ? "*** ESTIMATE GATE FAILED ***" : "ESTIMATE GATE ALL PASS");
    return g_fail ? 1 : 0;
}
