/* estimate_race.c -- the estimate method's plans against the banked plans, TIMED (git-ignored, unreleased).
 *
 * For each length N (1D c2c, natural order, out of place, one thread) three plans in one process:
 *   B   the banked plan: vfft_create from a scratch copy of the store (a store hit replays its row)
 *   E   the estimate plan: est_plan_1d + est_build_1d (formulas, a private in-memory store, no measuring)
 *   B2  the banked plan created a second time: the control (B against itself = the noise floor)
 * Neither create may read the library's clock (VFFT_CLOCK_PROBE counts every read): a create that read it
 * measured, and its N is reported and skipped. The three outputs must agree.
 * Protocol (the repo's single-thread law): vfft_measure_confine(0), the caller pinned to logical CPU 2,
 * HIGH priority; R rounds, the arms in a rotated order, 200 ms apart; each block = warm executes, then
 * REPS timed executes with REPS sized to ~80 ms. Reported per N: each arm's median over the rounds, and
 * the per-round ratios E/B and B2/B with their range.
 *
 *   estimate_race <store dir> [rounds=15] N...      (no N: every power of two 64..2^23)
 * Build: python gauntlet/build.py --compile --src src/estimate/estimate_race.c   (no --vfft) */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static long est_clock_reads;
static void est_clock_probe(void);
#define VFFT_CLOCK_PROBE est_clock_probe()
#include "vfft.c"                    /* FIRST: the whole library in its own order (form_slot_gate.c:23) */
#include "estimate_1d.h"
#include "estimate_build.h"
static void est_clock_probe(void) { est_clock_reads++; }

#define NARM 3
static const char *ARM[NARM] = { "B  banked", "E  estimate", "B2 control" };

static double qpc_ns(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
}
static int cmpd(const void *a, const void *b) { const double x = *(const double *)a, y = *(const double *)b; return x < y ? -1 : x > y; }

/* the stored row's recipe for one N, as its tokens (route, pair, ztt, flat, chain3, tile, four-step) */
static char *g_oop;   /* wisdom2_oop.txt, loaded once */
static void banked_tokens(const char *store, int N, char *out, size_t cap)
{
    char key[160];
    out[0] = 0;
    if (!g_oop)
    {
        char p[1024];
        FILE *f;
        long n;
        snprintf(p, sizeof p, "%s/wisdom2_oop.txt", store);
        if (!(f = fopen(p, "rb"))) { snprintf(out, cap, "(no oop shard)"); return; }
        fseek(f, 0, SEEK_END); n = ftell(f); fseek(f, 0, SEEK_SET);
        g_oop = (char *)malloc((size_t)n + 1);
        n = (long)fread(g_oop, 1, (size_t)n, f);
        g_oop[n] = 0;
        fclose(f);
    }
    snprintf(key, sizeof key, "@cell t=c2c n=%d q=1 ord=nat place=oop role=comp lay=il |", N);
    const char *l = strstr(g_oop, key);
    if (!l) { snprintf(out, cap, "(no row)"); return; }
    const char *e = strchr(l, '\n');
    const char *p = strchr(l, '|') + 1;
    const char *want[] = { "il_route=", "il_pair=", "il_ztt=", "il_tw=", "il_chain=", "il_flat=", "fs_chain=", "fs_wl=", "fs_row_il_route=", "fs_row_il_ztt=", "fs_row_il_pair=" };
    for (size_t w = 0; w < sizeof want / sizeof want[0]; w++)
    {
        const char *t = strstr(p, want[w]);
        if (!t || (e && t > e)) continue;
        const char *te = t;
        while (*te && *te != ' ' && *te != '\n' && *te != '\r') te++;
        size_t len = strlen(out);
        snprintf(out + len, cap - len, "%s%.*s", len ? " " : "", (int)(te - t), t);
    }
}
static void est_tokens(const est_recipe_t *rc, char *out, size_t cap)
{
    char s[96] = "";
    switch (rc->k1_il_route)
    {
    case VFFT_K1_IL_MONO: snprintf(out, cap, "mono"); return;
    case VFFT_K1_IL_2P_PURE: snprintf(out, cap, "2p %d.%d kv=0x%x", rc->il_R1, rc->il_R2, rc->il_kv); return;
    case VFFT_K1_IL_CHAIN3: snprintf(out, cap, "chain3 %d.%d.%d", rc->il_c3[0], rc->il_c3[1], rc->il_c3[2]); return;
    case VFFT_K1_IL_ZTT:
        for (int i = 0; i < rc->il_zt_n; i++) snprintf(s + strlen(s), sizeof s - strlen(s), "%s%d", i ? "." : "", rc->il_zt[i]);
        snprintf(out, cap, "ztt %s tw=%d", s, rc->il_tw);
        return;
    case VFFT_K1_IL_FS: snprintf(out, cap, "fs %dx%d %.60s", rc->il_R1, rc->il_R2, rc->fs); return;
    case VFFT_K1_IL_FLAT:
        for (int i = 0; i < rc->il_fl_n; i++) snprintf(s + strlen(s), sizeof s - strlen(s), "%s%d", i ? "." : "", rc->il_fl[i]);
        snprintf(out, cap, "flat %s tw=%d", s, rc->il_tw);
        return;
    default: snprintf(out, cap, "route %d", rc->k1_il_route); return;
    }
}

int main(int argc, char **argv)
{
    if (argc < 2) { printf("usage: estimate_race <store dir> [rounds] N...\n"); return 2; }
    const char *store = argv[1];
    const int R = argc > 2 ? atoi(argv[2]) : 15;
    int Ns[64], nN = 0;
    for (int i = 3; i < argc && nN < 64; i++) Ns[nN++] = atoi(argv[i]);
    if (!nN) for (int n = 64; n <= (1 << 23); n *= 2) Ns[nN++] = n;

    est_machine_t m;
    memset(&m, 0, sizeof m);
    m.lanes = VFFT_IL_VW;
    m.l1d = vfft_cpu_l1d_bytes();
    m.l2 = vfft_cpu_l2_bytes();
    m.l3 = vfft_cpu_l3_bytes();
    m.cores = vfft_topology()->ncores;
    printf("# machine: lanes %d, L1d %ld, L2 %ld, L3 %ld, cores %d; store %s; rounds %d\n", m.lanes, m.l1d, m.l2, m.l3, m.cores, store, R);
    printf("# confine -> 0x%llx\n", (unsigned long long)vfft_measure_confine(0));
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)1 << 2);
    printf("# HIGH priority: %s; caller on CPU %lu\n", SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS) ? "ok" : "REFUSED",
           GetCurrentProcessorNumber());

    double sumB = 0, sumE = 0;
    int nTimed = 0;
    static char line[64][320];
    for (int k = 0; k < nN; k++)
    {
        const int N = Ns[k];
        char bt[256], et[256], why[768];
        banked_tokens(store, N, bt, sizeof bt);
        vfft_plan p[NARM] = { 0 };
        /* B and B2: the public create from the scratch store */
        for (int a = 0; a < NARM; a += 2)
        {
            vfft_config_t c;
            memset(&c, 0, sizeof c);
            c.n[0] = N; c.nthreads = 1; c.wisdom_dir = store;
            const long r0 = est_clock_reads;
            p[a] = vfft_create(&c);
            if (est_clock_reads != r0) { printf("N=%-8d banked create MEASURED (clock read %ld times): skipped\n", N, est_clock_reads - r0); goto next; }
            if (!p[a]) { printf("N=%-8d banked create refused: %s\n", N, vfft_error()); goto next; }
        }
        /* E: the estimate */
        {
            est_request_t rq = { N, VFFT_C2C, VFFT_ORDER_NATURAL, VFFT_PLACEMENT_OUT, 1 };
            est_recipe_t rc;
            est_build_report_t rep;
            if (est_plan_1d(&m, &rq, &rc, why, sizeof why)) { printf("N=%-8d estimate refused (formula): %s\n", N, why); goto next; }
            est_tokens(&rc, et, sizeof et);
            const long r0 = est_clock_reads;
            p[1] = (vfft_plan)est_build_1d_ex(&rc, &rq, why, sizeof why, &rep);
            if (est_clock_reads != r0) { printf("N=%-8d estimate build MEASURED: skipped\n", N); goto next; }
            if (!p[1]) { printf("N=%-8d estimate build refused: %s\n", N, why); goto next; }
        }
        {
            const size_t bytes = (size_t)N * 16;
            double *in = vfft_malloc(bytes), *out[NARM];
            unsigned s = 12345u;
            for (long i = 0; i < 2L * N; i++) { s = s * 1664525u + 1013904223u; in[i] = (double)(s >> 8) / 16777216.0 - 0.5; }
            for (int a = 0; a < NARM; a++) { out[a] = vfft_malloc(bytes); vfft_execute(p[a], VFFT_FORWARD, in, out[a]); }
            double emax = 0, mag = 0;
            for (int a = 1; a < NARM; a++)
                for (long i = 0; i < 2L * N; i++) { const double e = fabs(out[a][i] - out[0][i]); if (e > emax) emax = e; if (fabs(out[0][i]) > mag) mag = fabs(out[0][i]); }
            /* size the block to ~80 ms from a short probe of B */
            double t0 = qpc_ns();
            for (int i = 0; i < 3; i++) vfft_execute(p[0], VFFT_FORWARD, in, out[0]);
            const double one = (qpc_ns() - t0) / 3;
            int reps = (int)(80e6 / one); if (reps < 3) reps = 3; if (reps > 200000) reps = 200000;
            int warm = reps / 10; if (warm < 2) warm = 2;
            static double t[NARM][128];
            for (int r = 0; r < R && r < 128; r++)
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
            const int same = strstr(bt, "il_route=ztt") && !strncmp(et, "ztt", 3) ? 0 : 0;
            (void)same;
            snprintf(line[k], sizeof line[k], "%8d  B %10.1f  E %10.1f  B2 %10.1f   E/B %.3f [%.3f..%.3f]   B2/B %.3f [%.3f..%.3f]   agree %.0e",
                     N, med[0], med[1], med[2], re[R / 2], re[0], re[R - 1], rc2[R / 2], rc2[0], rc2[R - 1], emax / mag);
            printf("%s\n          banked: %s\n          estimate: %s\n", line[k], bt, et);
            fflush(stdout);
            sumB += log(med[0]); sumE += log(med[1]); nTimed++;
            for (int a = 0; a < NARM; a++) vfft_free(out[a]);
            vfft_free(in);
        }
    next:
        for (int a = 0; a < NARM; a++) if (p[a]) vfft_destroy(p[a]);
    }
    printf("\n# summary (ns per forward execute; E/B < 1 = estimate faster):\n");
    for (int k = 0; k < nN; k++) if (line[k][0]) printf("%s\n", line[k]);
    if (nTimed) printf("# geometric mean E/B over %d lengths: %.3f\n", nTimed, exp((sumE - sumB) / nTimed));
    return 0;
}
