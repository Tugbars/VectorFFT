/* test_padded_dispatch.c — end-to-end public-API test of the PADDED c2c in-place path
 * (docs/roadmap/tail_handling/padding_design_decision.md, Phase 1 Step D + B). Drives ONLY
 * vfft.h: alloc a padded batch, set config.batch, vfft_create, vfft_execute — proving the
 * dispatch (build Kp-plan + run the padded wisdom's exec_me) is wired correctly.
 *
 * UNIFIED wisdom (single spike_wisdom.txt): the padded verdict is the (N,K) entry's exec_me, and
 * the pad plan IS the aligned (N,Kp) entry. Exercises BOTH branches of the padded create:
 *   PAD  cell (256,7):  (256,7).exec_me=Kp=8 -> use the aligned (256,8) entry, run me=8 (full-SIMD).
 *   TAIL cell (256,11): (256,11).exec_me=K=11 -> use (256,11)'s own factorization, run me=11 tail.
 * Correctness oracles (in-place c2c emits SCRAMBLED/digit-reversed bin order, so a natural-order
 * DFT can't be compared elementwise): (1) BIT-EXACT vs a tight (config.batch=NULL) reference plan
 * built at the SAME factorization — the STEP-E guarantee through the front door; (2) fwd->bwd
 * roundtrip recovers N*x on the K real lanes (invertibility).
 *
 * Self-contained: seeds the SINGLE spike_wisdom.txt into a scratch dir, points VFFT_WISDOM_DIR at
 * it BEFORE the first vfft_create (the default bundle is lazy-loaded once).
 *
 * Build: python build.py --src test/test_padded_dispatch.c --vfft
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <direct.h>   /* _mkdir (mingw) */
#include "vfft.h"

#define WDIR "padtest_wisdom"
#define VW 4
static size_t roundup_vw(size_t k) { return (k + (VW - 1)) & ~(size_t)(VW - 1); }

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("    FAIL: %s\n", msg); fails++; } } while (0)

/* seed one v6 wisdom line: N K nf factors... best_ns 0 0 0 0 variants... exec_me */
static void seed_line(FILE *f, int N, int K, const int *fac, const int *var, int nf, double ns, int exec_me)
{
    fprintf(f, "%d %d %d", N, K, nf);
    for (int i = 0; i < nf; i++) fprintf(f, " %d", fac[i]);
    fprintf(f, " %.1f 0 0 0 0", ns);
    for (int i = 0; i < nf; i++) fprintf(f, " %d", var[i]);
    fprintf(f, " %d\n", exec_me);
}

static void seed_wisdom(void)
{
    _mkdir(WDIR);
    int fac[3] = {4, 8, 8};   /* product 256 — buildable (radix 4/8) */
    int var[3] = {0, 2, 2};   /* n1, T1S, T1S */

    /* UNIFIED single file: padded verdict = the (N,K) entry's exec_me; pad plan = the aligned
     * (N,Kp) entry. Seed both with the SAME factorization so pad@me=Kp is bit-exact vs tight@me=K. */
    FILE *ft = fopen(WDIR "/spike_wisdom.txt", "w");
    fprintf(ft, "@version 6\n");
    seed_line(ft, 256, 7,  fac, var, 3, 90.0, 8);     /* (256,7): PAD verdict + (256,8) present */
    seed_line(ft, 256, 8,  fac, var, 3, 90.0, 0);     /* (256,8): aligned pad plan (exec_me=0) */
    seed_line(ft, 256, 11, fac, var, 3, 100.0, 11);   /* (256,11): TAIL verdict exec_me=K=11 */
    seed_line(ft, 256, 19, fac, var, 3, 90.0, 20);    /* (256,19): PAD verdict but NO (256,20) ->
                                                       * exercises ON-DEMAND aligned calibration */
    fclose(ft);
}

/* does WDIR/spike_wisdom.txt now contain an (N,K) entry? (proves on-demand calibration saved it) */
static int wisdom_has(int N, int K)
{
    FILE *f = fopen(WDIR "/spike_wisdom.txt", "r");
    if (!f) return 0;
    char line[512]; int found = 0, n, k;
    while (fgets(line, sizeof line, f))
        if (line[0] != '#' && line[0] != '@' && sscanf(line, "%d %d", &n, &k) == 2 && n == N && k == K) { found = 1; break; }
    fclose(f);
    return found;
}

/* run one (N,K) cell padded: roundtrip always; bit-exact vs tight reference only when `bitexact`
 * (the on-demand-aligned case calibrates a fresh (N,Kp) plan != the seeded factorization). */
static void run_cell(int N, int K, const char *label, int bitexact)
{
    size_t Kp = roundup_vw((size_t)K);
    printf("  cell N=%d K=%d Kp=%zu  [%s]\n", N, K, Kp, label);

    vfft_batch b = vfft_alloc_batch(N, (size_t)K);
    CHECK(b != NULL, "alloc_batch");
    if (!b) return;
    CHECK(vfft_batch_stride(b) == Kp, "batch stride == Kp");
    double *pre = vfft_batch_re(b), *pim = vfft_batch_im(b);

    /* fill the K real lanes (random), pad lanes already zeroed by the allocator. */
    srand(1234 + N + K);
    double *xr = malloc((size_t)N * K * sizeof(double)), *xi = malloc((size_t)N * K * sizeof(double));
    for (int e = 0; e < N; e++)
        for (int l = 0; l < K; l++) {
            double a = (double)rand() / RAND_MAX - 0.5, c = (double)rand() / RAND_MAX - 0.5;
            xr[e * K + l] = a; xi[e * K + l] = c;
            pre[(size_t)e * Kp + l] = a; pim[(size_t)e * Kp + l] = c;
        }

    vfft_config_t cfg; memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C; cfg.placement = VFFT_INPLACE; cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1; cfg.n[0] = N; cfg.howmany = (size_t)K; cfg.batch = b;
    vfft_plan p = vfft_create(&cfg);
    CHECK(p != NULL, "vfft_create (padded)");
    if (!p) { free(xr); free(xi); vfft_free_batch(b); return; }

    /* forward, in-place on the padded buffer */
    vfft_execute(p, VFFT_FORWARD, pre, pim, pre, pim);

    /* bit-exact vs a tight (config.batch=NULL) reference at the SAME factorization */
    if (bitexact)
    {
        double *tre = malloc((size_t)N * K * sizeof(double)), *tim = malloc((size_t)N * K * sizeof(double));
        for (int i = 0; i < N * K; i++) { tre[i] = xr[i]; tim[i] = xi[i]; }
        vfft_config_t rc; memset(&rc, 0, sizeof rc);
        rc.transform = VFFT_C2C; rc.placement = VFFT_INPLACE; rc.rigor = VFFT_MEASURE;
        rc.dims = 1; rc.n[0] = N; rc.howmany = (size_t)K; rc.batch = NULL;
        vfft_plan pr = vfft_create(&rc);
        CHECK(pr != NULL, "vfft_create (tight reference)");
        if (pr) {
            vfft_execute(pr, VFFT_FORWARD, tre, tim, tre, tim);
            double be = 0;
            for (int e = 0; e < N; e++)
                for (int l = 0; l < K; l++) {
                    double dr = fabs(pre[(size_t)e * Kp + l] - tre[e * K + l]);
                    double di = fabs(pim[(size_t)e * Kp + l] - tim[e * K + l]);
                    if (dr > be) be = dr; if (di > be) be = di;
                }
            CHECK(be == 0.0, "padded == tight reference (BIT-EXACT, same factorization)");
            printf("    padded vs tight ref: max diff %.2e %s\n", be, be == 0.0 ? "(bit-exact)" : "");
            vfft_destroy(pr);
        }
        free(tre); free(tim);
    }

    /* roundtrip: bwd recovers N*x on the K real lanes */
    vfft_execute(p, VFFT_BACKWARD, pre, pim, pre, pim);
    double rt = 0, inv = 1.0 / (double)N;
    for (int e = 0; e < N; e++)
        for (int l = 0; l < K; l++) {
            double dr = fabs(pre[(size_t)e * Kp + l] * inv - xr[e * K + l]);
            double di = fabs(pim[(size_t)e * Kp + l] * inv - xi[e * K + l]);
            if (dr > rt) rt = dr; if (di > rt) rt = di;
        }
    CHECK(rt < 1e-10, "fwd->bwd roundtrip recovers N*x");
    printf("    roundtrip err: %.2e\n", rt);

    free(xr); free(xi);
    vfft_destroy(p);
    vfft_free_batch(b);
}

int main(void)
{
    seed_wisdom();
    putenv("VFFT_WISDOM_DIR=" WDIR);   /* must precede the first vfft_create (lazy default bundle) */

    printf("# padded c2c in-place dispatch test (Step D, through vfft.h)\n");
    printf("# wisdom dir: %s\n\n", WDIR);
    run_cell(256, 7,  "PAD: (256,7).exec_me=8 -> aligned (256,8) plan @me=8", 1);
    run_cell(256, 11, "TAIL: (256,11).exec_me=11 -> own factorization @me=11", 1);

    /* ON-DEMAND: (256,19).exec_me=20 but (256,20) NOT seeded -> dispatch must calibrate the
     * aligned (256,20) plan itself, then run pad. Roundtrip-only (fresh plan != seed), and the
     * (256,20) entry must now exist in the wisdom file (proves the on-demand calibration fired). */
    CHECK(!wisdom_has(256, 20), "precondition: (256,20) absent before the on-demand cell");
    run_cell(256, 19, "ON-DEMAND: (256,19).exec_me=20, (256,20) built on the fly", 0);
    CHECK(wisdom_has(256, 20), "on-demand: (256,20) aligned plan calibrated + saved");

    printf(fails ? "\nRESULT: %d CHECK(s) FAILED\n" : "\nRESULT: all checks passed\n", fails);
    return fails ? 1 : 0;
}
