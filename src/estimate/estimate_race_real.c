/* estimate_race_real.c -- the estimate method's 2D and 3D REAL plans (R2C, C2R) against the banked plans, TIMED and
 * CHECKED (tracked, not wired into the library; the complex twin is estimate_race_nd.c).
 *
 * For each cell (2D N1xN2 or 3D N1xN2xN3, one direction, natural order, out of place, one thread or T threads):
 *   the check  the estimate plan (VFFT_FLAG_ESTIMATE) against a long-double DFT: R2C on 48 sampled bins of the
 *              half spectrum, C2R by the round trip (its input the estimate R2C's spectrum of random reals, its
 *              output N x those reals); relative to the largest value, PASS below 1e-11;
 *   B          the banked plan: vfft_create from a scratch copy of the store (a complete row replays)
 *   E          the estimate plan: vfft_create with VFFT_FLAG_ESTIMATE (a private in-memory store, nothing measured,
 *              no store read; il/estimate/real_estimate_nd_il.h)
 *   B2         the banked plan created a second time: the control (B against itself = the noise floor)
 * Neither create may read the library's clock (VFFT_CLOCK_PROBE counts every read): a create that read it measured,
 * and its cell is reported and skipped (its row is banked into the store named, so a second run times it). A refused
 * estimate create is reported with the library's reason. The three outputs must agree.
 * Protocol (estimate_race_nd.c's): vfft_measure_confine(0); one thread: the caller pinned to logical CPU 2, HIGH
 * priority, R rounds, the arms in a rotated order, 200 ms apart, each block = warm executes then REPS timed executes
 * with REPS sized to ~80 ms; T > 1: the caller on logical CPU 0 in one measurement scope with the library's guard,
 * every arm warmed at least 5 ms, REPS sized to ~10 ms, the order shuffled each round (seeded), never paused.
 * Reported per cell: each arm's median over the rounds, the per-round ratios E/B and B2/B with their range, the
 * agreement and the check; at the end the geometric mean of E/B.
 *
 *   estimate_race_real <store dir> [rounds=15] [T=n] [r2c|c2r] cell...   (cell = N1xN2 or N1xN2xN3; none: the list)
 * Build: python gauntlet/build.py --compile --src src/estimate/estimate_race_real.c   (no --vfft)
 * Run it on a quiet machine only, against a scratch COPY of the store. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static long est_clock_reads;
static void est_clock_probe(void);
#define VFFT_CLOCK_PROBE est_clock_probe()
#include "vfft.c"                    /* FIRST: the whole library in its own order; it carries the library's real
                                      * estimate (il/estimate/real_estimate_nd_il.h) */
static void est_clock_probe(void) { est_clock_reads++; }

#define NARM 3
#define MAXCELL 64

/* the default cells: powers of two, planes and volumes over the real estimate's rules (the tall column, the
 * one-stage column, the 128-point leaf, the fused walk, the strips, the one-stage axis 0) */
static const int DEF[][3] = {
    { 32, 32, 0 }, { 64, 64, 0 }, { 128, 128, 0 }, { 256, 256, 0 }, { 512, 512, 0 }, { 1024, 1024, 0 }, { 2048, 2048, 0 },
    { 64, 256, 0 }, { 256, 64, 0 }, { 16, 1024, 0 }, { 1024, 16, 0 }, { 32, 512, 0 }, { 512, 32, 0 }, { 128, 2048, 0 },
    { 2048, 128, 0 }, { 64, 4096, 0 }, { 4096, 64, 0 }, { 8, 8192, 0 }, { 8192, 8, 0 }, { 1024, 4096, 0 }, { 4096, 1024, 0 },
    { 8, 8, 8 }, { 16, 16, 16 }, { 32, 32, 32 }, { 64, 64, 64 }, { 128, 128, 128 }, { 4, 256, 256 }, { 256, 4, 256 },
    { 256, 256, 4 }, { 16, 128, 128 }, { 128, 16, 128 }, { 128, 128, 16 }, { 32, 64, 128 }, { 128, 64, 32 },
    { 1024, 16, 16 }, { 16, 16, 1024 }, { 512, 4, 4 },
};

static double qpc_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
static int cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

/* the banked row's verdicts for one cell (the 2D or 3D shard), the children's tokens left out */
static char *g_shard[2];
static void banked_tokens(const char *store, const int *n, int rank, int T, char *out, size_t cap)
{
    char thr[24] = "", key[160];
    char **sh = &g_shard[rank == 3];
    const char *l, *e, *t;
    size_t len = 0;
    out[0] = 0;
    if (!*sh)
    {
        char path[1024];
        FILE *f;
        long sz;
        snprintf(path, sizeof path, "%s/%s", store, rank == 3 ? "wisdom2_3d.txt" : "wisdom2_2d.txt");
        if (!(f = fopen(path, "rb"))) { snprintf(out, cap, "(no %dD shard)", rank); return; }
        fseek(f, 0, SEEK_END); sz = ftell(f); fseek(f, 0, SEEK_SET);
        *sh = (char *)malloc((size_t)sz + 1);
        sz = (long)fread(*sh, 1, (size_t)sz, f);
        (*sh)[sz] = 0;
        fclose(f);
    }
    if (T > 1)
        snprintf(thr, sizeof thr, " nthreads=%d", T);
    if (rank == 3)
        snprintf(key, sizeof key, "@cell t=r2c n=%dx%dx%d q=1 ord=nat place=oop lay=il%s |", n[0], n[1], n[2], thr);
    else
        snprintf(key, sizeof key, "@cell t=r2c n=%dx%d q=1 ord=nat place=oop lay=il%s |", n[0], n[1], thr);
    if (!(l = strstr(*sh, key))) { snprintf(out, cap, "(no row)"); return; }
    e = strstr(l + strlen(key), " | ");
    if (!e) e = strchr(l, '\n');
    for (t = l + strlen(key); t && *t && (!e || t < e) && len + 2 < cap;)
    {
        const char *te;
        while (*t == ' ') t++;
        te = t;
        while (*te && *te != ' ' && *te != '\n' && *te != '\r') te++;
        if (te > t && strncmp(t, "rp_", 3) && strncmp(t, "rx_", 3) && strncmp(t, "rax_", 4) && strncmp(t, "raxr_", 5) &&
            strncmp(t, "plane_", 6))
            len += (size_t)snprintf(out + len, cap - len, "%s%.*s", len ? " " : "", (int)(te - t), t);
        if (!*te || *te == '\n') break;
        t = te;
    }
}
static void chain_s(char *o, size_t cap, const int *R, int n)
{
    size_t l = 0;
    o[0] = 0;
    for (int i = 0; i < n && l < cap; i++) l += (size_t)snprintf(o + l, cap - l, "%s%d", i ? "." : "", R[i]);
}
static void rec2(const est_plan2dr_t *p, char *b, size_t cap)
{
    char c[64];
    int l;
    chain_s(c, sizeof c, p->col.R, p->col.nst);
    l = snprintf(b, cap, "chain=%s forms=%s blu=%d rx=%s cx=%s tf=%d", c, p->col.forms[0] ? p->col.forms : "-",
                 p->col.blu, p->rx[0] ? p->rx : "-", p->cx[0] ? p->cx : "-", p->tf);
    if (p->T > 1 && l > 0 && (size_t)l < cap)
        snprintf(b + l, cap - (size_t)l, " cmt=%d mtarm=%d msw=%d tcmt=%d", p->cmt, p->mtarm, p->msw, p->tcmt);
}
/* the estimate's recipe from the library's formulas (est_plan_2d_real / est_plan_3d_real) */
static int est_tokens(const est_machine_t *m, const int *n, int rank, int c2r, int T, char *out, size_t cap, char *why,
                      size_t whylen)
{
    if (rank == 2)
    {
        est_plan2dr_t *p = (est_plan2dr_t *)malloc(sizeof *p);
        if (!p || est_plan_2d_real(m, n[0], n[1], c2r, 0, T, p, why, whylen)) { free(p); return -1; }
        rec2(p, out, cap);
        free(p);
        return 0;
    }
    {
        est_plan3dr_t *p = (est_plan3dr_t *)malloc(sizeof *p);
        char c0[64], pl[384];
        int l;
        if (!p || est_plan_3d_real(m, n[0], n[1], n[2], c2r, 0, T, p, why, whylen)) { free(p); return -1; }
        chain_s(c0, sizeof c0, p->col.R, p->col.nst);
        rec2(&p->plane, pl, sizeof pl);
        l = snprintf(out, cap, "chain=%s blu=%d s=%d nf=%d nsw=%d", c0, p->col.blu, p->s, p->nf, p->nsw);
        if (T > 1 && l > 0 && (size_t)l < cap)
            l += snprintf(out + l, cap - (size_t)l, " cmt=%d cmts=%d cmtf=%d", p->cmt, p->cmts, p->cmtf);
        if (l > 0 && (size_t)l < cap)
            snprintf(out + l, cap - (size_t)l, " | plane %s", pl);
        free(p);
        return 0;
    }
}

static vfft_plan create(const int *n, int rank, int c2r, int T, const char *store)
{
    vfft_config_t c;
    memset(&c, 0, sizeof c);
    c.transform = c2r ? VFFT_C2R : VFFT_R2C;
    c.placement = VFFT_PLACEMENT_OUT;
    c.rank = rank;
    memcpy(c.n, n, sizeof(int) * 3);
    c.nthreads = T;
    if (store) c.wisdom_dir = store;
    else c.flags = VFFT_FLAG_ESTIMATE;
    return vfft_create(&c);
}

/* an R2C output against the long-double DFT of x on 48 sampled bins of the half spectrum (A rows of the real axis B,
 * the rows the outer extent N1 or N1 N2); the worst error over the largest bin */
static double check_r2c(const int *n, int rank, const double *x, const double *spec)
{
    const int A = rank == 2 ? n[0] : n[0] * n[1], B = rank == 2 ? n[1] : n[2];
    const size_t hp1 = (size_t)B / 2 + 1;
    const long double TWO_PI = 6.28318530717958647692528676655900577L;
    long double *wr = (long double *)malloc(sizeof(long double) * (size_t)B), *wi = (long double *)malloc(sizeof(long double) * (size_t)B);
    double err = 0, mag = 0;
    for (int s = 0; s < 48; s++)
    {
        const int k0 = s == 0 ? 0 : (int)(((unsigned)s * 2654435761u) % (unsigned)n[0]);
        const int k1 = rank == 2 ? 0 : (s == 0 ? 0 : (int)(((unsigned)s * 97u + 13u) % (unsigned)n[1]));
        const int kb = s == 1 ? (int)(hp1 - 1) : (int)(((unsigned)s * 40503u) % (unsigned)hp1);
        long double re = 0, im = 0;
        for (int k = 0; k < B; k++)
        {
            const long double a = -TWO_PI * (long double)((long long)kb * k % B) / B;
            wr[k] = cosl(a);
            wi[k] = sinl(a);
        }
        for (int r = 0; r < A; r++)
        {
            const int i0 = rank == 2 ? r : r / n[1], i1 = rank == 2 ? 0 : r % n[1];
            const long double ph = -TWO_PI * ((long double)((long long)k0 * i0 % n[0]) / n[0] +
                                              (rank == 3 ? (long double)((long long)k1 * i1 % n[1]) / n[1] : 0.0L));
            const double *xr = x + (size_t)r * B;
            long double sr = 0, si = 0;
            for (int k = 0; k < B; k++) { sr += xr[k] * wr[k]; si += xr[k] * wi[k]; }
            const long double c = cosl(ph), sn = sinl(ph);
            re += sr * c - si * sn;
            im += sr * sn + si * c;
        }
        {
            const size_t row = rank == 2 ? (size_t)k0 : (size_t)k0 * n[1] + k1;
            const double *o = spec + 2 * (row * hp1 + kb);
            const double d = hypot(o[0] - (double)re, o[1] - (double)im), m = hypot((double)re, (double)im);
            if (d > err) err = d;
            if (m > mag) mag = m;
        }
    }
    free(wr);
    free(wi);
    return mag > 0 ? err / mag : err;
}

int main(int argc, char **argv)
{
    static int cells[MAXCELL][3];
    static char line[MAXCELL][400];
    if (argc > 1 && !strcmp(argv[1], "--where"))
    {   /* this CPU's folder of the library's own store, as recal_1d_probe --where prints it */
        printf("folder=%s\nidentity=%s\nbuild=%s\n", vfft_wisdom_folder(), vfft_wisdom_identity(), vfft_wisdom_build());
        return 0;
    }
    static double t[NARM][128];
    int nc = 0, T = 1, c2r = 0, fails = 0;
    if (argc < 2) { printf("usage: estimate_race_real <store dir> [rounds] [T=n] [r2c|c2r] cell...   (cell = N1xN2 or N1xN2xN3)\n"); return 2; }
    const char *store = argv[1];
    const int R = argc > 2 ? atoi(argv[2]) : 15;
    if (R < 1 || R > 128) { printf("rounds: 1..128\n"); return 2; }
    for (int i = 3; i < argc && nc < MAXCELL; i++)
    {
        int a = 0, b = 0, c = 0;
        if (!strncmp(argv[i], "T=", 2))
        {
            T = atoi(argv[i] + 2);
            if (T < 1 || T > 1024) { printf("threads: 1..1024\n"); return 2; }
            continue;
        }
        if (!strcmp(argv[i], "r2c") || !strcmp(argv[i], "c2r")) { c2r = !strcmp(argv[i], "c2r"); continue; }
        const int k = sscanf(argv[i], "%dx%dx%d", &a, &b, &c);
        if (k < 2) { printf("bad cell %s\n", argv[i]); return 2; }
        cells[nc][0] = a; cells[nc][1] = b; cells[nc][2] = k == 3 ? c : 0;
        nc++;
    }
    if (!nc)
        for (size_t i = 0; i < sizeof DEF / sizeof DEF[0]; i++, nc++)
            memcpy(cells[nc], DEF[i], sizeof cells[nc]);

    est_machine_t m;
    memset(&m, 0, sizeof m);
    m.lanes = VFFT_IL_VW;
    m.l1d = vfft_cpu_l1d_bytes();
    m.l2 = vfft_cpu_l2_bytes();
    m.l3 = vfft_cpu_l3_bytes();
    m.cores = vfft_topology()->ncores;
    m.l1d_ways = (int)vfft_cpu_l1d_ways();
    printf("# %s; machine: lanes %d, L1d %ld (%d-way), L2 %ld, L3 %ld, cores %d; store %s; rounds %d; threads %d\n",
           c2r ? "C2R" : "R2C", m.lanes, m.l1d, m.l1d_ways, m.l2, m.l3, m.cores, store, R, T);
    printf("# confine -> 0x%llx\n", (unsigned long long)vfft_measure_confine(0));
    if (T > 1)
    {   /* the threaded protocol: the caller on logical CPU 0 in one measurement scope, the library's guard */
        vfft_measure_config_t mc;
        char d[320];
        memset(&mc, 0, sizeof mc);
        mc.pin = VFFT_MEASURE_PIN_CORE;
        mc.pin_core = 0;
        mc.guard = VFFT_MEASURE_GUARD_DEFAULT;
        mc.priority = VFFT_MEASURE_PRIORITY_LEAVE;
        vfft_measure_configure(&mc);
        const int f = vfft_measure_begin();
        printf("# measurement scope: %s%s\n", vfft_measure_describe(d, sizeof d), f ? " (not clean)" : "");
    }
    else
    {
        SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << 2);
        printf("# HIGH priority: %s; caller on CPU %lu\n", SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS) ? "ok" : "REFUSED",
               GetCurrentProcessorNumber());
    }

    double sumB = 0, sumE = 0;
    int nTimed = 0;
    for (int k = 0; k < nc; k++)
    {
        const int *n = cells[k], rank = n[2] ? 3 : 2;
        const int A = rank == 2 ? n[0] : n[0] * n[1], B = rank == 2 ? n[1] : n[2];
        const size_t hp1 = (size_t)B / 2 + 1, nreal = (size_t)A * B, ncpx = (size_t)A * hp1;
        char name[48], bt[512], et[768], why[768];
        vfft_plan p[NARM] = { 0 };
        double chk = -1;
        if (rank == 3) snprintf(name, sizeof name, "%dx%dx%d", n[0], n[1], n[2]);
        else snprintf(name, sizeof name, "%dx%d", n[0], n[1]);
        line[k][0] = 0;
        banked_tokens(store, n, rank, T, bt, sizeof bt);
        for (int a = 0; a < NARM; a++)
        {
            if (a == 1 && est_tokens(&m, n, rank, c2r, T, et, sizeof et, why, sizeof why))
            {
                printf("%-12s estimate refused (formula): %s\n", name, why);
                fails++;
                goto next;
            }
            const long r0 = est_clock_reads;
            p[a] = create(n, rank, c2r, T, a == 1 ? NULL : store);
            if (est_clock_reads != r0)
            {
                printf("%-12s %s create MEASURED (clock read %ld times): skipped\n", name, a == 1 ? "estimate" : "banked",
                       est_clock_reads - r0);
                if (a == 1) fails++;
                goto next;
            }
            if (!p[a])
            {
                printf("%-12s %s create refused: %s\n", name, a == 1 ? "estimate" : "banked", vfft_error());
                if (a == 1) fails++;
                goto next;
            }
        }
        {
            double *x = vfft_malloc(sizeof(double) * nreal + 64), *spec = vfft_malloc(sizeof(double) * 2 * ncpx + 64);
            double *in, *out[NARM];
            const size_t nin = c2r ? 2 * ncpx : nreal, nout = c2r ? nreal : 2 * ncpx;
            const int dir = c2r ? VFFT_BACKWARD : VFFT_FORWARD;
            unsigned s = 12345u;
            memset(spec, 0, sizeof(double) * 2 * ncpx);
            for (size_t i = 0; i < nreal; i++) { s = s * 1664525u + 1013904223u; x[i] = (double)(s >> 8) / 16777216.0 - 0.5; }
            /* the check: the estimate R2C (the plan itself, or for C2R a second estimate plan) on x */
            {
                vfft_plan f = c2r ? create(n, rank, 0, T, NULL) : p[1];
                if (f)
                {
                    vfft_execute(f, VFFT_FORWARD, x, spec);
                    if (!c2r)
                        chk = check_r2c(n, rank, x, spec);
                    else
                    {   /* the round trip: C2R of the R2C spectrum is N x */
                        double *back = vfft_malloc(sizeof(double) * nreal + 64), err = 0, mag = 0;
                        const double NN = (double)n[0] * n[1] * (rank == 3 ? n[2] : 1);
                        vfft_execute(p[1], VFFT_BACKWARD, spec, back);
                        for (size_t i = 0; i < nreal; i++)
                        {
                            const double w = NN * x[i];
                            if (fabs(back[i] - w) > err) err = fabs(back[i] - w);
                            if (fabs(w) > mag) mag = fabs(w);
                        }
                        chk = mag > 0 ? err / mag : err;
                        vfft_free(back);
                        vfft_destroy(f);
                    }
                }
            }
            if (chk < 0 || chk >= 1e-11) fails++;
            in = c2r ? spec : x;
            for (int a = 0; a < NARM; a++) { out[a] = vfft_malloc(sizeof(double) * nout + 64); vfft_execute(p[a], dir, in, out[a]); }
            double emax = 0, mag = 0;
            for (int a = 1; a < NARM; a++)
                for (size_t i = 0; i < nout; i++) { const double e = fabs(out[a][i] - out[0][i]); if (e > emax) emax = e; if (fabs(out[0][i]) > mag) mag = fabs(out[0][i]); }
            (void)nin;
            /* size the block to ~80 ms (T > 1: ~10 ms) from a short probe of B */
            double t0 = qpc_ns();
            for (int i = 0; i < 3; i++) vfft_execute(p[0], dir, in, out[0]);
            const double one = (qpc_ns() - t0) / 3;
            int reps = (int)((T > 1 ? 10e6 : 80e6) / one); if (reps < 3) reps = 3; if (reps > 200000) reps = 200000;
            int warm = reps / 10; if (warm < 2) warm = 2;
            if (T > 1)
            {   /* every arm warmed at least 5 ms, then the blocks back to back, the order shuffled each round (seeded) */
                unsigned rs = ((unsigned)n[0] * 73856093u ^ (unsigned)n[1] * 19349663u ^ (unsigned)n[2] * 83492791u ^ (unsigned)c2r) | 1u;
                for (int a = 0; a < NARM; a++)
                {
                    int w = 0;
                    t0 = qpc_ns();
                    while (qpc_ns() - t0 < 5e6 || w < 2) { vfft_execute(p[a], dir, in, out[a]); w++; }
                }
                for (int r = 0; r < R; r++)
                {
                    int ord[NARM];
                    for (int j = 0; j < NARM; j++) ord[j] = j;
                    for (int j = NARM - 1; j > 0; j--)
                    {
                        rs = rs * 1664525u + 1013904223u;
                        const int xj = (int)((rs >> 8) % (unsigned)(j + 1)), y = ord[j];
                        ord[j] = ord[xj];
                        ord[xj] = y;
                    }
                    for (int j = 0; j < NARM; j++)
                    {
                        const int a = ord[j];
                        t0 = qpc_ns();
                        for (int i = 0; i < reps; i++) vfft_execute(p[a], dir, in, out[a]);
                        t[a][r] = (qpc_ns() - t0) / reps;
                    }
                }
            }
            else
                for (int r = 0; r < R; r++)
                    for (int j = 0; j < NARM; j++)
                    {
                        const int a = (j + r) % NARM;
                        for (int w = 0; w < warm; w++) vfft_execute(p[a], dir, in, out[a]);
                        t0 = qpc_ns();
                        for (int i = 0; i < reps; i++) vfft_execute(p[a], dir, in, out[a]);
                        t[a][r] = (qpc_ns() - t0) / reps;
                        Sleep(200);
                    }
            double med[NARM], re[128], rc2[128];
            for (int a = 0; a < NARM; a++) { double v[128]; memcpy(v, t[a], sizeof(double) * R); qsort(v, R, sizeof(double), cmpd); med[a] = v[R / 2]; }
            for (int r = 0; r < R; r++) { re[r] = t[1][r] / t[0][r]; rc2[r] = t[2][r] / t[0][r]; }
            qsort(re, R, sizeof(double), cmpd); qsort(rc2, R, sizeof(double), cmpd);
            snprintf(line[k], sizeof line[k], "%-12s  B %11.1f  E %11.1f  B2 %11.1f   E/B %.3f [%.3f..%.3f]   B2/B %.3f [%.3f..%.3f]   agree %.0e   check %.0e %s",
                     name, med[0], med[1], med[2], re[R / 2], re[0], re[R - 1], rc2[R / 2], rc2[0], rc2[R - 1],
                     mag > 0 ? emax / mag : emax, chk, chk >= 0 && chk < 1e-11 ? "PASS" : "FAIL");
            printf("%s\n              banked: %s\n              estimate: %s\n", line[k], bt, et);
            fflush(stdout);
            sumB += log(med[0]); sumE += log(med[1]); nTimed++;
            for (int a = 0; a < NARM; a++) vfft_free(out[a]);
            vfft_free(x);
            vfft_free(spec);
        }
    next:
        for (int a = 0; a < NARM; a++) if (p[a]) vfft_destroy(p[a]);
    }
    printf("\n# summary (%s, ns per execute; E/B < 1 = estimate faster):\n", c2r ? "C2R" : "R2C");
    for (int k = 0; k < nc; k++) if (line[k][0]) printf("%s\n", line[k]);
    if (nTimed) printf("# geometric mean E/B over %d cells: %.3f\n", nTimed, exp((sumE - sumB) / nTimed));
    printf("# estimate failures (refused, measured, or check failed): %d\n", fails);
    if (T > 1)
        vfft_measure_end();
    return 0;
}
