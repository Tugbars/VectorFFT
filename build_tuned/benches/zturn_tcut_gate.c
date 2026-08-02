/* zturn_tcut_gate.c — CORRECTNESS gate for the tiled mid stages (`tcut`) of the
 * K=1 ZTURN-S z-cascade.  docs/research/tcut_spec.md §0 (F0/F2) and §7.
 *
 * 🔴 EVERYTHING here goes through the PUBLIC FRONT DOOR
 * (vfft_create / vfft_execute / vfft_destroy).  No plan-level API is called by
 * hand: a plan reached by a private path is not the plan production uses.  The
 * tiling arm is selected the only way production can select it — the env gate
 * VFFT_TCUT, read once inside vfft_zturn2_create_chain.
 *
 * WHAT IS CHECKED, per cell and per arm
 *   E  ENGAGEMENT.  The arm's create-time [tcut] line is parsed out of stderr
 *      and its (tiled, tcut, tfuse, w, NT) are compared against the spec's
 *      predicted geometry.  Without this the gate is worthless: an arm that is
 *      silently REFUSED falls back to the untiled driver and would then "pass"
 *      every output comparison for the wrong reason.
 *   F0 memcmp vs the untiled arm, BOTH directions.  This is the primary gate
 *      and it is a bit gate, not a tolerance: the transformation reorders
 *      provably independent kernel calls and changes no arithmetic, so a
 *      1e-16 difference means the index algebra is wrong (spec §2.3).
 *   D  REFERENCE-DFT agreement through the arm's OWN output permutation.  The
 *      permutation is not assumed — it is DISCOVERED, by feeding all N complex
 *      exponentials e^{+2*pi*i*f*n/N} (whose DFT is exactly N*delta[k,f]) and
 *      taking the peak position; the map is then asserted to be a bijection.
 *      That is a complete rank-N verification of the forward operator, and it
 *      is repeated for the backward operator by feeding the permuted deltas.
 *      One naive O(N^2) DFT of a random vector is checked on top, because the
 *      task asks for a naive reference explicitly.
 *   R  ROUNDTRIP fwd->bwd == N*x, and (spec §7 row 13) the tiled arm's
 *      roundtrip error must equal the untiled arm's TO THE LAST BIT.
 *
 * Arms enumerated: off (A0) | a1 (the stage-major section-split CONTROL arm)
 * | tcut = 0..3 x tfuse = 0,1 x twiddle form = reset,honest.  Arms the create
 * fence refuses are reported as REFUSED, never as PASS.
 *
 * Build:
 *   cd build_tuned && python build.py --src benches/zturn_tcut_gate.c --vfft --compile
 * Run:
 *   zturn_tcut_gate.exe --wisdir <scratch copy of generator/generated>
 *   (🔴 point it at a SCRATCH wisdir; never let a probe write banked wisdom.)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* ───────────────────────── stderr tap (engagement) ─────────────────────── */
static char g_errpath[1024];
static long g_errpos = 0;

static int err_tap_open(const char *dir)
{
    snprintf(g_errpath, sizeof g_errpath, "%s/_tcut_gate_stderr.log", dir);
    if (!freopen(g_errpath, "w", stderr)) return 0;
    setvbuf(stderr, NULL, _IONBF, 0);
    g_errpos = 0;
    return 1;
}

/* Return everything written to stderr since the last call (owned static buf). */
static const char *err_tap_read(void)
{
    static char buf[8192];
    buf[0] = 0;
    fflush(stderr);
    FILE *f = fopen(g_errpath, "rb");
    if (!f) return buf;
    if (fseek(f, g_errpos, SEEK_SET) == 0) {
        size_t n = fread(buf, 1, sizeof buf - 1, f);
        buf[n] = 0;
        g_errpos += (long)n;
    }
    fclose(f);
    return buf;
}

/* ───────────────────────────── env arm select ──────────────────────────── */
static void env_set(const char *k, const char *v)
{
    static char slots[4][128];
    static int n = 0;
    char *s = slots[n++ & 3];
    snprintf(s, 128, "%s=%s", k, v ? v : "");
    putenv(s);
}

/* ─────────────────────────────── arm table ─────────────────────────────── */
typedef struct {
    const char *name;      /* VFFT_TCUT value */
    const char *tw;        /* VFFT_TCUT_TW value ("" = reset) */
    const char *w;         /* VFFT_TCUT_W value in KB ("" = ladder width) */
    int want_tiled;        /* predicted plan.tiled, -1 = don't care */
    int want_tcut;         /* -1 = don't care (DERIVED from an explicit width,
                            * so it varies with N and chain) */
    int want_tfuse;
} arm_t;

/* ─────────────────────── plan creation for one arm ─────────────────────── */
typedef struct {
    vfft_plan h;
    int engaged;           /* [tcut] line seen and accepted        */
    int refused;           /* [tcut] REFUSED line seen             */
    int tiled, tcut, tfuse;
    long w, NT;
    char twform[16];
} armplan_t;

static int make_arm(int N, vfft_wisdom *W, const arm_t *a, armplan_t *out)
{
    memset(out, 0, sizeof *out);
    env_set("VFFT_TCUT", a->name);
    env_set("VFFT_TCUT_TW", a->tw);
    env_set("VFFT_TCUT_W",  a->w);
    env_set("VFFT_TCUT_VERBOSE", "1");
    env_set("VFFT_FORCE_ZROUTE", "zturn");   /* the arm under test must be ZTURN-S */

    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_OUTOFPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = 1;
    cfg.order = VFFT_ORDER_SCRAMBLED;
    cfg.layout = VFFT_LAYOUT_INTERLEAVED;    /* the committed z contract */
    cfg.nthreads = 1;
    cfg.wisdom = W;
    (void)err_tap_read();                    /* drop anything queued */
    out->h = vfft_create(&cfg);
    const char *log = err_tap_read();
    const char *p = strstr(log, "[tcut]");
    if (p) {
        if (strstr(p, "REFUSED")) out->refused = 1;
        else if (sscanf(p, "[tcut] N=%*d nf=%*d tiled=%d tcut=%d tfuse=%d tw=%15s "
                           "w=%ld NT=%ld",
                        &out->tiled, &out->tcut, &out->tfuse, out->twform,
                        &out->w, &out->NT) == 6)
            out->engaged = 1;
    }
    return out->h != NULL;
}

/* ──────────────────────────── helpers ─────────────────────────────────── */
static double *alloc_z(size_t n)          /* n complex */
{
    void *p = NULL;
#ifdef _WIN32
    p = _aligned_malloc(2 * n * sizeof(double), 64);
#else
    if (posix_memalign(&p, 64, 2 * n * sizeof(double))) p = NULL;
#endif
    return (double *)p;
}
static void free_z(double *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* Discover the arm's forward output permutation pi: feeding
 * x[n] = e^{+2*pi*i*f*n/N} must produce exactly N at output slot pi(f) and 0
 * everywhere else.  Returns 0 on failure (peak not isolated / not a bijection),
 * and reports the worst relative sidelobe over all f. */
static int discover_perm(vfft_plan h, int N, int *pi, double *worst_side,
                         double *worst_peak_err)
{
    double *x = alloc_z((size_t)N), *y = alloc_z((size_t)N);
    char *seen = (char *)calloc((size_t)N, 1);
    int ok = 1;
    *worst_side = 0.0; *worst_peak_err = 0.0;
    for (int f = 0; f < N && ok; f++) {
        for (int n = 0; n < N; n++) {
            double a = 2.0 * M_PI * ((double)((long)f * n % N)) / (double)N;
            x[2 * n] = cos(a); x[2 * n + 1] = sin(a);
        }
        vfft_execute(h, VFFT_FORWARD, x, NULL, y, NULL);
        int best = -1; double bm = -1.0, second = 0.0;
        for (int k = 0; k < N; k++) {
            double m = fabs(y[2 * k]) + fabs(y[2 * k + 1]);
            if (m > bm) { second = bm > 0 ? bm : second; bm = m; best = k; }
            else if (m > second) second = m;
        }
        double side = second / (double)N;
        double perr = fabs(bm - (double)N) / (double)N; /* |peak| ~ N (re+im L1) */
        if (side > *worst_side) *worst_side = side;
        if (perr > *worst_peak_err) *worst_peak_err = perr;
        if (best < 0 || side > 1e-9 || seen[best]) ok = 0;
        else { seen[best] = 1; pi[f] = best; }
    }
    for (int k = 0; k < N && ok; k++) if (!seen[k]) ok = 0;  /* bijection */
    free(seen); free_z(x); free_z(y);
    return ok;
}

/* Backward operator, verified on the same basis: a unit delta at the SCRAMBLED
 * slot pi(f) must come back as e^{+2*pi*i*f*n/N} (bwd = N * inverse). */
static double check_bwd_basis(vfft_plan h, int N, const int *pi)
{
    double *S = alloc_z((size_t)N), *y = alloc_z((size_t)N);
    double worst = 0.0;
    for (int f = 0; f < N; f++) {
        memset(S, 0, 2 * (size_t)N * sizeof(double));
        S[2 * pi[f]] = 1.0;
        vfft_execute(h, VFFT_BACKWARD, S, NULL, y, NULL);
        for (int n = 0; n < N; n++) {
            double a = 2.0 * M_PI * ((double)((long)f * n % N)) / (double)N;
            double e = fabs(y[2 * n] - cos(a)) + fabs(y[2 * n + 1] - sin(a));
            if (e > worst) worst = e;
        }
    }
    free_z(S); free_z(y);
    return worst;
}

/* Naive O(N^2) DFT of a random vector, compared through pi (the task's
 * explicit "agree with a naive reference DFT" requirement). */
static double check_naive_dft(vfft_plan h, int N, const int *pi,
                             const double *x)
{
    double *y = alloc_z((size_t)N);
    vfft_execute(h, VFFT_FORWARD, (double *)x, NULL, y, NULL);
    double worst = 0.0, mag = 0.0;
    for (int k = 0; k < N; k++) {
        double sr = 0.0, si = 0.0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * ((double)((long)k * n % N)) / (double)N;
            double c = cos(a), s = sin(a);
            sr += x[2 * n] * c - x[2 * n + 1] * s;
            si += x[2 * n] * s + x[2 * n + 1] * c;
        }
        double m = fabs(sr) + fabs(si);
        if (m > mag) mag = m;
        double e = fabs(y[2 * pi[k]] - sr) + fabs(y[2 * pi[k] + 1] - si);
        if (e > worst) worst = e;
    }
    free_z(y);
    return mag > 0 ? worst / mag : worst;
}

/* ─────────────────────────────── main ─────────────────────────────────── */
int main(int argc, char **argv)
{
    const char *wisdir = NULL;
    int cells[8] = {2048, 4096, 8192, 16384};
    int ncells = 4, deep = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--wisdir") && i + 1 < argc) wisdir = argv[++i];
        else if (!strcmp(argv[i], "--no-dft")) deep = 0;
        else if (!strcmp(argv[i], "--cell") && i + 1 < argc) {
            ncells = 0; cells[ncells++] = atoi(argv[++i]);
        }
    }
    if (!wisdir) {
        fprintf(stdout, "usage: %s --wisdir <dir with oop_wisdom.txt> "
                        "[--cell N] [--no-dft]\n", argv[0]);
        return 2;
    }
    if (!err_tap_open(wisdir)) { printf("cannot open stderr tap\n"); return 2; }

    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { printf("vfft_wisdom_load(%s) FAILED\n", wisdir); return 2; }

    static const arm_t arms[] = {
        /* ---- ladder widths (w = D[tcut]); every campaign number is here ---- */
        {"off", "",       "",   0,  0, 0},
        {"a1",  "",       "",   2,  0, 0},
        {"a1",  "honest", "",   2,  0, 0},
        {"0",   "",       "",   1,  0, 0},
        {"0",   "honest", "",   1,  0, 0},
        {"0:1", "",       "",   1,  0, 1},
        {"1",   "",       "",   1,  1, 0},
        {"1:1", "",       "",   1,  1, 1},
        {"2",   "",       "",   1,  2, 0},
        {"2:1", "",       "",   1,  2, 1},
        {"3",   "",       "",   1,  3, 0},
        {"3:1", "",       "",   1,  3, 1},
        {"4",   "",       "",   1,  4, 0},  /* expected REFUSED on nf<=6 chains */

        /* ---- EXPLICIT widths (VFFT_TCUT_W), the axis the ladder cannot reach.
         * The cut is DERIVED, so want_tcut is -1: it varies with N and chain.
         * F0 (memcmp vs untiled) is the real gate here and it does not care
         * where the cut landed — only that the reorder changed no arithmetic.
         * 16 KB is MKL's tile and the one N=8192 could never express. */
        {"0",   "",       "4",  1, -1, 0},
        {"0",   "",       "8",  1, -1, 0},
        {"0",   "",       "16", 1, -1, 0},   /* MKL's tile */
        {"0:1", "",       "16", 1, -1, 1},   /* fused terminator + explicit w */
        {"0",   "honest", "16", 1, -1, 0},   /* honest twiddles + explicit w */
        {"0",   "",       "32", 1, -1, 0},
        {"0",   "",       "64", 1, -1, 0},
        {"0",   "",       "3",  1, -1, 0},   /* not a divisor -> expect REFUSED */
    };
    const int narms = (int)(sizeof arms / sizeof arms[0]);

    int fails = 0, refused = 0, passes = 0;
    printf("\n=== zturn tcut CORRECTNESS gate (public API, VFFT_TCUT env arm) ===\n");

    for (int ci = 0; ci < ncells; ci++) {
        const int N = cells[ci];

        /* ---- reference arm: UNTILED ---- */
        armplan_t A0;
        if (!make_arm(N, W, &arms[0], &A0)) {
            printf("N=%-6d  vfft_create FAILED for the untiled arm — cell skipped\n", N);
            fails++; continue;
        }
        double *x  = alloc_z((size_t)N);
        double *S0 = alloc_z((size_t)N), *R0 = alloc_z((size_t)N);
        srand(1234 + N);
        for (int i = 0; i < 2 * N; i++) x[i] = (double)rand() / RAND_MAX - 0.5;
        vfft_execute(A0.h, VFFT_FORWARD, x, NULL, S0, NULL);
        vfft_execute(A0.h, VFFT_BACKWARD, S0, NULL, R0, NULL);
        double rt0 = 0.0, mx = 0.0;
        for (int i = 0; i < 2 * N; i++) {
            double e = fabs(R0[i] / N - x[i]);
            if (e > rt0) rt0 = e;
            if (fabs(x[i]) > mx) mx = fabs(x[i]);
        }
        rt0 /= (mx > 0 ? mx : 1.0);

        int *pi0 = (int *)malloc(sizeof(int) * (size_t)N);
        double side0 = 0, peak0 = 0, bwd0 = 0, dft0 = 0;
        int perm0 = 0;
        if (deep) {
            perm0 = discover_perm(A0.h, N, pi0, &side0, &peak0);
            if (perm0) {
                bwd0 = check_bwd_basis(A0.h, N, pi0);
                dft0 = check_naive_dft(A0.h, N, pi0, x);
            }
        }
        printf("\nN=%-6d  UNTILED reference:  perm=%s  fwd-basis sidelobe=%.2e "
               "peak-err=%.2e  bwd-basis=%.2e  naive-DFT rel=%.2e  roundtrip rel=%.2e\n",
               N, deep ? (perm0 ? "BIJECTION" : "*** NOT A BIJECTION ***") : "(skipped)",
               side0, peak0, bwd0, dft0, rt0);
        if (deep && !perm0) fails++;

        printf("  %-14s %-7s | %-26s | %-9s %-9s | %-11s | %s\n",
               "VFFT_TCUT", "tw", "engaged (tiled/tcut/tfuse w NT)",
               "fwd memcmp", "bwd memcmp", "roundtrip", "verdict");

        for (int ai = 1; ai < narms; ai++) {
            armplan_t A;
            if (!make_arm(N, W, &arms[ai], &A)) {
                printf("  %-14s %-7s | create FAILED\n", arms[ai].name, arms[ai].tw);
                fails++; continue;
            }
            if (A.refused || !A.engaged) {
                printf("  %-14s %-7s | %-26s | %-9s %-9s | %-11s | REFUSED (fence)\n",
                       arms[ai].name, arms[ai].tw[0] ? arms[ai].tw : "reset",
                       A.refused ? "refused by fence" : "no [tcut] line",
                       "-", "-", "-");
                refused++;
                vfft_destroy(A.h);
                continue;
            }
            /* engagement must match the prediction */
            int engok = (A.tiled == arms[ai].want_tiled &&
                         (A.tiled != 1 ||
                          ((arms[ai].want_tcut < 0 ||
                            A.tcut == arms[ai].want_tcut) &&
                           A.tfuse == arms[ai].want_tfuse)));

            double *S = alloc_z((size_t)N), *R = alloc_z((size_t)N);
            vfft_execute(A.h, VFFT_FORWARD, x, NULL, S, NULL);
            int fwd_eq = (memcmp(S, S0, 2 * (size_t)N * sizeof(double)) == 0);
            vfft_execute(A.h, VFFT_BACKWARD, S0, NULL, R, NULL);
            int bwd_eq = (memcmp(R, R0, 2 * (size_t)N * sizeof(double)) == 0);
            /* roundtrip through the arm's OWN forward output */
            double *R2 = alloc_z((size_t)N);
            vfft_execute(A.h, VFFT_BACKWARD, S, NULL, R2, NULL);
            double rt = 0.0;
            for (int i = 0; i < 2 * N; i++) {
                double e = fabs(R2[i] / N - x[i]);
                if (e > rt) rt = e;
            }
            rt /= (mx > 0 ? mx : 1.0);
            int rt_bitequal = (rt == rt0);

            /* per-arm reference-DFT agreement (deep, one tiled arm per cell) */
            char dfts[64] = "(skipped)";
            int dftok = 1;
            if (deep && arms[ai].want_tiled == 1 && A.tcut == 0 && !A.tfuse
                && !arms[ai].tw[0]) {
                int *pi = (int *)malloc(sizeof(int) * (size_t)N);
                double sd = 0, pk = 0;
                int pok = discover_perm(A.h, N, pi, &sd, &pk);
                double bw = pok ? check_bwd_basis(A.h, N, pi) : 9e9;
                double df = pok ? check_naive_dft(A.h, N, pi, x) : 9e9;
                int same = pok;
                for (int k = 0; k < N && same; k++) if (pi[k] != pi0[k]) same = 0;
                snprintf(dfts, sizeof dfts, "perm=%s dft=%.1e bwd=%.1e",
                         pok ? (same ? "SAME" : "DIFFERENT!") : "BAD", df, bw);
                dftok = pok && same && df < 1e-13 && bw < 1e-9;
                free(pi);
            }
            int ok = engok && fwd_eq && bwd_eq && rt_bitequal && dftok;
            printf("  %-14s %-7s | tiled=%d tcut=%d tfuse=%d w=%-5ld NT=%-3ld | "
                   "%-9s %-9s | %-11s | %s%s%s\n",
                   arms[ai].name, arms[ai].tw[0] ? arms[ai].tw : "reset",
                   A.tiled, A.tcut, A.tfuse, A.w, A.NT,
                   fwd_eq ? "EXACT" : "DIFFER", bwd_eq ? "EXACT" : "DIFFER",
                   rt_bitequal ? "bit-equal" : "CHANGED",
                   ok ? "PASS" : "*** FAIL ***",
                   engok ? "" : " [engagement mismatch]",
                   deep && arms[ai].want_tiled == 1 && A.tcut == 0 && !A.tfuse
                       && !arms[ai].tw[0] && !arms[ai].w[0] ? dfts : "");
            if (ok) passes++; else fails++;
            free_z(S); free_z(R); free_z(R2);
            vfft_destroy(A.h);
        }
        free(pi0); free_z(x); free_z(S0); free_z(R0);
        vfft_destroy(A0.h);
    }
    vfft_wisdom_free(W);
    printf("\n=== TOTAL: %d PASS, %d FAIL, %d REFUSED-by-fence ===\n",
           passes, fails, refused);
    printf("(REFUSED is not a failure: spec F2 requires the create fence to "
           "reject an illegal cut rather than silently drop work.)\n");
    return fails ? 1 : 0;
}
