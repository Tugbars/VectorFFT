/* estimate_race_nd.c -- the estimate method's 2D and 3D plans against the banked plans, TIMED (tracked, not
 * wired into the library; the 1D twin is estimate_race.c).
 *
 * For each cell (2D N1xN2 or 3D N1xN2xN3, c2c, natural order, out of place, one thread) three plans in one
 * process:
 *   B   the banked plan: vfft_create from a scratch copy of the store (a complete row replays)
 *   E   the estimate plan: vfft_create with VFFT_FLAG_ESTIMATE (the library's formulas and build: a private
 *       in-memory store, no measuring, no store read; docs/design/estimate_2d3d_sketch.md)
 *   B2  the banked plan created a second time: the control (B against itself = the noise floor)
 * Neither create may read the library's clock (VFFT_CLOCK_PROBE counts every read): a create that read it
 * measured, and its cell is reported and skipped. The three outputs must agree.
 * Protocol (the repo's single-thread law): vfft_measure_confine(0), the caller pinned to logical CPU 2,
 * HIGH priority; R rounds, the arms in a rotated order, 200 ms apart; each block = warm executes, then
 * REPS timed executes with REPS sized to ~80 ms. Reported per cell: each arm's median over the rounds, and
 * the per-round ratios E/B and B2/B with their range; at the end the geometric mean of E/B.
 *
 *   estimate_race_nd <store dir> [rounds=15] cell...     (cell = N1xN2 or N1xN2xN3; none: the list below)
 * Build: python gauntlet/build.py --compile --src src/estimate/estimate_race_nd.c   (no --vfft)
 * Run it on a quiet machine only, against a scratch COPY of the store (a cell whose row is incomplete races
 * at its banked create and saves into the directory named). */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static long est_clock_reads;
static void est_clock_probe(void);
#define VFFT_CLOCK_PROBE est_clock_probe()
#include "vfft.c"                    /* FIRST: the whole library in its own order; it carries the library's
                                      * estimate (il/estimate/estimate_nd_il.h) */
static void est_clock_probe(void) { est_clock_reads++; }

#define NARM 3
#define MAXCELL 64

/* the default cells: powers of two over the 2D rules (narrow and wide rows, the turn, the two-pass rows, the
 * band, the leading stage) and the 3D rules (the scrambled pass's exceptions, the band, the plane child) */
static const int DEF[][3] = {
    { 64, 64, 0 }, { 128, 128, 0 }, { 256, 256, 0 }, { 512, 512, 0 }, { 1024, 1024, 0 }, { 2048, 2048, 0 },
    { 16, 1024, 0 }, { 1024, 16, 0 }, { 512, 16, 0 }, { 1024, 8, 0 }, { 2048, 4, 0 }, { 4096, 256, 0 },
    { 32, 4096, 0 }, { 8, 8192, 0 }, { 256, 4096, 0 },
    { 16, 16, 16 }, { 32, 32, 32 }, { 64, 64, 64 }, { 128, 128, 128 }, { 8, 64, 64 }, { 64, 8, 64 },
    { 512, 4, 4 }, { 1024, 16, 16 }, { 4096, 16, 16 }, { 8192, 4, 4 }, { 4, 256, 256 }, { 32, 64, 128 },
};

static double qpc_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
static int cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

/* the banked row's verdicts for one cell (the 2D or 3D shard of the store), as its tokens */
static char *g_shard[2];
static void banked_tokens(const char *store, const int *n, int rank, char *out, size_t cap)
{
    static const char *want2[] = { "chain=", "forms=", "blu=", "tpc=", "rd=", "rdw=", "wl=", "ro=", "sw=", "rbk=", "turn=",
                                   "csk=", "rp_1_il_route=", NULL };
    static const char *want3[] = { "chain=", "forms=", "blu=", "tpc=", "rd=", "wl=", "s=", "nf=", "nsw=", "plane_1_chain=",
                                   "plane_1_wl=", "plane_1_ro=", "plane_1_turn=", "plane_1_csk=", NULL };
    const char **want = rank == 3 ? want3 : want2;
    char key[160];
    char **sh = &g_shard[rank == 3];
    const char *l, *e, *p;
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
    if (rank == 3)
        snprintf(key, sizeof key, "@cell t=c2c n=%dx%dx%d q=1 ord=nat place=oop lay=il |", n[0], n[1], n[2]);
    else
        snprintf(key, sizeof key, "@cell t=c2c n=%dx%d q=1 ord=nat place=oop lay=il |", n[0], n[1]);
    if (!(l = strstr(*sh, key))) { snprintf(out, cap, "(no row)"); return; }
    e = strchr(l, '\n');
    p = strchr(l, '|') + 1;
    for (size_t w = 0; want[w]; w++)
    {
        const char *t = p;
        while ((t = strstr(t, want[w])) != NULL && (!e || t < e) && t[-1] != ' ')
            t++;   /* a whole token: preceded by a space */
        if (!t || (e && t > e)) continue;
        const char *te = t;
        while (*te && *te != ' ' && *te != '\n' && *te != '\r') te++;
        size_t len = strlen(out);
        snprintf(out + len, cap - len, "%s%.*s", len ? " " : "", (int)(te - t), t);
    }
}
static void chain_s(char *o, size_t cap, const int *R, int n)
{
    size_t l = 0;
    o[0] = 0;
    for (int i = 0; i < n && l < cap; i++) l += snprintf(o + l, cap - l, "%s%d", i ? "." : "", R[i]);
}
/* the estimate's recipe, from the library's formulas */
static int est_tokens(const est_machine_t *m, const int *n, int rank, char *out, size_t cap, char *why, size_t whylen)
{
    char c0[64], c1[64];
    if (rank == 2)
    {
        est_plan2d_t *p = (est_plan2d_t *)malloc(sizeof *p);
        if (!p || est_plan_2d(m, n[0], n[1], p, why, whylen)) { free(p); return -1; }
        chain_s(c0, sizeof c0, p->col.R, p->col.nst);
        snprintf(out, cap, "chain=%s%s%s blu=%d tpc=%d wl=%d ro=%d sw=%d rbk=%d turn=%d csk=%d rp=%s", c0,
                 p->col.forms[0] ? " forms=" : "", p->col.forms, p->col.blu, p->col.tpc, p->col.wl, p->ro, p->sw, p->rbk,
                 p->turn, p->csk, vw2_oop_il_name[p->rp.k1_il_route]);
        free(p);
        return 0;
    }
    {
        est_plan3d_t *p = (est_plan3d_t *)malloc(sizeof *p);
        if (!p || est_plan_3d(m, n[0], n[1], n[2], p, why, whylen)) { free(p); return -1; }
        chain_s(c0, sizeof c0, p->col.R, p->col.nst);
        chain_s(c1, sizeof c1, p->plane.col.R, p->plane.col.nst);
        snprintf(out, cap, "chain=%s%s%s blu=%d wl=%d s=%d nf=%d | plane chain=%s wl=%d ro=%d turn=%d csk=%d", c0,
                 p->col.forms[0] ? " forms=" : "", p->col.forms, p->col.blu, p->col.wl, p->s, p->nf, c1, p->plane.col.wl,
                 p->plane.ro, p->plane.turn, p->plane.csk);
        free(p);
        return 0;
    }
}

int main(int argc, char **argv)
{
    static int cells[MAXCELL][3];
    static char line[MAXCELL][360];
    static double t[NARM][128];
    int nc = 0;
    if (argc < 2) { printf("usage: estimate_race_nd <store dir> [rounds] cell...   (cell = N1xN2 or N1xN2xN3)\n"); return 2; }
    const char *store = argv[1];
    const int R = argc > 2 ? atoi(argv[2]) : 15;
    if (R < 1 || R > 128) { printf("rounds: 1..128\n"); return 2; }
    for (int i = 3; i < argc && nc < MAXCELL; i++)
    {
        int a = 0, b = 0, c = 0;
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
    printf("# machine: lanes %d, L1d %ld, L2 %ld, L3 %ld, cores %d; store %s; rounds %d\n", m.lanes, m.l1d, m.l2, m.l3, m.cores,
           store, R);
    printf("# confine -> 0x%llx\n", (unsigned long long)vfft_measure_confine(0));
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << 2);
    printf("# HIGH priority: %s; caller on CPU %lu\n", SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS) ? "ok" : "REFUSED",
           GetCurrentProcessorNumber());

    double sumB = 0, sumE = 0;
    int nTimed = 0;
    for (int k = 0; k < nc; k++)
    {
        const int *n = cells[k], rank = n[2] ? 3 : 2;
        const long P = (long)n[0] * n[1] * (rank == 3 ? n[2] : 1);
        char name[48], bt[512], et[512], why[768];
        vfft_plan p[NARM] = { 0 };
        if (rank == 3) snprintf(name, sizeof name, "%dx%dx%d", n[0], n[1], n[2]);
        else snprintf(name, sizeof name, "%dx%d", n[0], n[1]);
        line[k][0] = 0;
        banked_tokens(store, n, rank, bt, sizeof bt);
        for (int a = 0; a < NARM; a++)
        {
            vfft_config_t c;
            memset(&c, 0, sizeof c);
            c.rank = rank;
            memcpy(c.n, n, sizeof(int) * 3);
            c.nthreads = 1;
            if (a == 1)
            {   /* E: the estimate, through the public flag; the recipe printed from the library's formulas */
                if (est_tokens(&m, n, rank, et, sizeof et, why, sizeof why)) { printf("%-12s estimate refused (formula): %s\n", name, why); goto next; }
                c.flags = VFFT_FLAG_ESTIMATE;
            }
            else
                c.wisdom_dir = store;   /* B and B2: the public create from the scratch store */
            const long r0 = est_clock_reads;
            p[a] = vfft_create(&c);
            if (est_clock_reads != r0)
            {
                printf("%-12s %s create MEASURED (clock read %ld times): skipped\n", name, a == 1 ? "estimate" : "banked",
                       est_clock_reads - r0);
                goto next;
            }
            if (!p[a]) { printf("%-12s %s create refused: %s\n", name, a == 1 ? "estimate" : "banked", vfft_error()); goto next; }
        }
        {
            const size_t bytes = (size_t)P * 16;
            double *in = vfft_malloc(bytes), *out[NARM];
            unsigned s = 12345u;
            for (long i = 0; i < 2L * P; i++) { s = s * 1664525u + 1013904223u; in[i] = (double)(s >> 8) / 16777216.0 - 0.5; }
            for (int a = 0; a < NARM; a++) { out[a] = vfft_malloc(bytes); vfft_execute(p[a], VFFT_FORWARD, in, out[a]); }
            double emax = 0, mag = 0;
            for (int a = 1; a < NARM; a++)
                for (long i = 0; i < 2L * P; i++) { const double e = fabs(out[a][i] - out[0][i]); if (e > emax) emax = e; if (fabs(out[0][i]) > mag) mag = fabs(out[0][i]); }
            /* size the block to ~80 ms from a short probe of B */
            double t0 = qpc_ns();
            for (int i = 0; i < 3; i++) vfft_execute(p[0], VFFT_FORWARD, in, out[0]);
            const double one = (qpc_ns() - t0) / 3;
            int reps = (int)(80e6 / one); if (reps < 3) reps = 3; if (reps > 200000) reps = 200000;
            int warm = reps / 10; if (warm < 2) warm = 2;
            for (int r = 0; r < R; r++)
                for (int j = 0; j < NARM; j++)
                {
                    const int a = (j + r) % NARM;
                    for (int w = 0; w < warm; w++) vfft_execute(p[a], VFFT_FORWARD, in, out[a]);
                    t0 = qpc_ns();
                    for (int i = 0; i < reps; i++) vfft_execute(p[a], VFFT_FORWARD, in, out[a]);
                    t[a][r] = (qpc_ns() - t0) / reps;
                    Sleep(200);
                }
            double med[NARM], re[128], rc2[128];
            for (int a = 0; a < NARM; a++) { double v[128]; memcpy(v, t[a], sizeof(double) * R); qsort(v, R, sizeof(double), cmpd); med[a] = v[R / 2]; }
            for (int r = 0; r < R; r++) { re[r] = t[1][r] / t[0][r]; rc2[r] = t[2][r] / t[0][r]; }
            qsort(re, R, sizeof(double), cmpd); qsort(rc2, R, sizeof(double), cmpd);
            snprintf(line[k], sizeof line[k], "%-12s  B %11.1f  E %11.1f  B2 %11.1f   E/B %.3f [%.3f..%.3f]   B2/B %.3f [%.3f..%.3f]   agree %.0e",
                     name, med[0], med[1], med[2], re[R / 2], re[0], re[R - 1], rc2[R / 2], rc2[0], rc2[R - 1], mag > 0 ? emax / mag : emax);
            printf("%s\n              banked: %s\n              estimate: %s\n", line[k], bt, et);
            fflush(stdout);
            sumB += log(med[0]); sumE += log(med[1]); nTimed++;
            for (int a = 0; a < NARM; a++) vfft_free(out[a]);
            vfft_free(in);
        }
    next:
        for (int a = 0; a < NARM; a++) if (p[a]) vfft_destroy(p[a]);
    }
    printf("\n# summary (ns per forward execute; E/B < 1 = estimate faster):\n");
    for (int k = 0; k < nc; k++) if (line[k][0]) printf("%s\n", line[k]);
    if (nTimed) printf("# geometric mean E/B over %d cells: %.3f\n", nTimed, exp((sumE - sumB) / nTimed));
    return 0;
}
