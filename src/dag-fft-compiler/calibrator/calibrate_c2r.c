/* calibrate_c2r.c — single-cell calibrator for the c2r (backward real FFT) path.
 *
 * Mirror of calibrate_r2c.c for the inverse transform. c2r is calibrated
 * SEPARATELY from r2c: it uses the DIF-backward hc2hc family (different codelets),
 * so its best per-stage variant can differ. We brute-force (factorization ×
 * per-stage variant FLAT/LOG3/T1S) inline, build via c2r_plan_create_ex, gate the
 * real output against the trusted DEFAULT-policy c2r plan (all valid combos compute
 * the same inverse DFT), time it, and keep the fastest. The winner is written to a
 * SEPARATE c2r_wisdom.txt (reusing the v5 wisdom_reader.h verbatim).
 *
 * The c2r INPUT is a valid packed halfcomplex spectrum, produced here by an rfft
 * FORWARD plan over a random real signal — so the calibrator also sanity-checks the
 * full r2c->c2r roundtrip (c2r(rfft(x)) == N*x) of the winner.
 *
 * Coverage: the LEAF (last factor) needs r2cb[r]; combine STAGES need r2cb[r] AND
 * hc2hc_dif_bwd[r] (so radix 32 is leaf-only, exactly as in r2c).
 *
 * Build: build_tuned/build.py --src calibrator/calibrate_c2r.c --compile
 * Usage: calibrate_c2r N K [core=2] [verbose=1]
 * Env:   VFFT_PROTO_C2R_WIS  (output file; default generated/c2r_wisdom.txt)
 */
#define VFFT_RFFT_MAX_RADIX 32     /* allow 32 as a LEAF (r2cb[32] exists) */
#define VFFT_RFFT_RANGED 1         /* enable the T1S (ranged) variant axis */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "../core/env.h"
#include "../core/planner.h"      /* vfft_proto_wisdom_t + load/add/save */
#include "../core/dp_planner.h"   /* vfft_proto_now_ns */
#include "../core/rfft.h"
#include "../core/c2r.h"
#include "../generator/generated/rfft_registry_avx2.h"  /* forward family (build the spectrum) */
#include "../generator/generated/c2r_registry_avx2.h"   /* backward family (r2cb + hc2hc_dif_bwd) */

#ifndef C2R_WIS
#define C2R_WIS "../../src/dag-fft-compiler/generator/generated/c2r_wisdom.txt"
#endif

static int g_leaf_ok[VFFT_RFFT_MAX_RADIX + 1];   /* r usable as leaf  (r2cb) */
static int g_stage_ok[VFFT_RFFT_MAX_RADIX + 1];  /* r usable as stage (r2cb+hc2hc_dif_bwd) */

typedef struct { int nf; int factors[STRIDE_MAX_STAGES]; } facz_t;
static facz_t g_fz[4096]; static int g_nfz;

static void enum_stages(int rem, int *pre, int depth, int leaf, int maxnf) {
    if (depth + 1 >= maxnf) return;
    if (rem == 1) {
        if (g_nfz >= (int)(sizeof g_fz/sizeof g_fz[0])) return;
        facz_t *f = &g_fz[g_nfz++];
        f->nf = depth + 1;
        for (int i = 0; i < depth; i++) f->factors[i] = pre[i];
        f->factors[depth] = leaf;
        return;
    }
    for (int r = 2; r <= VFFT_RFFT_MAX_RADIX; r++) {
        if (!g_stage_ok[r] || rem % r != 0) continue;
        pre[depth] = r;
        enum_stages(rem / r, pre, depth + 1, leaf, maxnf);
    }
}
static void enumerate(int N, int maxnf) {
    int pre[STRIDE_MAX_STAGES];
    g_nfz = 0;
    for (int leaf = 2; leaf <= VFFT_RFFT_MAX_RADIX; leaf++) {
        if (!g_leaf_ok[leaf] || N % leaf != 0) continue;
        enum_stages(N / leaf, pre, 0, leaf, maxnf);
    }
}

int main(int argc, char **argv) {
    stride_env_init();
    if (argc < 3) { fprintf(stderr, "usage: calibrate_c2r N K [core] [verbose]\n"); return 2; }
    int N = atoi(argv[1]); size_t K = (size_t)atoll(argv[2]);
    int core = (argc > 3) ? atoi(argv[3]) : 2;
    int verbose = (argc > 4) ? atoi(argv[4]) : 1;
    if (K == 0 || K % 8 != 0) { fprintf(stderr, "K must be a multiple of 8\n"); return 1; }
    if (stride_pin_thread(core) != 0) fprintf(stderr, "warn: pin cpu%d\n", core);

    rfft_codelets_t reg; memset(&reg, 0, sizeof reg);
    rfft_register_all_avx2(&reg);   /* forward: r2cf + hc2hc_dit_fwd (spectrum) */
    c2r_register_all_avx2(&reg);    /* backward: r2cb + hc2hc_dif_bwd          */
    for (int r = 2; r <= VFFT_RFFT_MAX_RADIX; r++) {
        g_leaf_ok[r]  = (reg.r2cb[r] != NULL);
        g_stage_ok[r] = (reg.r2cb[r] != NULL && reg.hc2hc_dif_bwd[r] != NULL);
    }

    enumerate(N, 6);
    if (g_nfz == 0) { fprintf(stderr, "N=%d not coverable by the c2r radix set\n", N); return 1; }

    size_t total = (size_t)N * K;
    double *x = NULL, *spec = NULL, *out = NULL, *ref = NULL;
    vfft_proto_posix_memalign((void**)&x,    64, total * sizeof(double));        /* real input    */
    vfft_proto_posix_memalign((void**)&spec, 64, 2 * total * sizeof(double));    /* halfcomplex   */
    vfft_proto_posix_memalign((void**)&out,  64, total * sizeof(double));        /* real output   */
    vfft_proto_posix_memalign((void**)&ref,  64, total * sizeof(double));        /* real ref      */
    srand(7);
    for (size_t i = 0; i < total; i++) x[i] = (double)rand()/RAND_MAX*2.0 - 1.0;

    /* Build a valid packed halfcomplex spectrum: rfft FORWARD of x. */
    rfft_plan_t *fp = rfft_plan_create(N, K, g_fz[0].factors, g_fz[0].nf, &reg);
    if (!fp) { fprintf(stderr, "forward plan NULL for the seed factorization\n"); return 1; }
    rfft_execute_fwd_packed(fp, x, spec);
    rfft_plan_destroy(fp);

    /* trusted reference: the default-policy c2r plan (the shipping path). */
    c2r_plan_t *dp = c2r_plan_create(N, K, g_fz[0].factors, g_fz[0].nf, &reg);
    if (!dp) { fprintf(stderr, "default c2r plan NULL for the seed factorization\n"); return 1; }
    c2r_execute_packed(dp, spec, ref);
    c2r_plan_destroy(dp);

    int best_nf = 0, best_factors[STRIDE_MAX_STAGES], best_variants[STRIDE_MAX_STAGES];
    double best_ns = 1e18; int benched = 0, rejected = 0;

    for (int fi = 0; fi < g_nfz; fi++) {
        const facz_t *fz = &g_fz[fi];
        int ncomb = fz->nf - 1;                 /* DIF-bwd combine stages d=0..nf-2 */
        int nvar = 1; for (int i = 0; i < ncomb; i++) nvar *= 3;
        for (int vc = 0; vc < nvar; vc++) {
            int variant[STRIDE_MAX_STAGES]; memset(variant, 0, sizeof variant);
            int t = vc;
            for (int d = 0; d < ncomb; d++) { variant[d] = t % 3; t /= 3; }
            int ok = 1;
            for (int d = 0; d < ncomb; d++) {
                int r = fz->factors[d];
                if (variant[d] == 1 && !reg.hc2hc_dif_bwd_log3[r]) { ok = 0; break; }
                if (variant[d] == 2 && !reg.hc2hc_dif_rng_bwd[r])  { ok = 0; break; }
            }
            if (!ok) continue;

            c2r_plan_t *p = c2r_plan_create_ex(N, K, fz->factors, fz->nf, variant, &reg);
            if (!p) continue;

            memset(out, 0, total * sizeof(double));
            c2r_execute_packed(p, spec, out);
            double md = 0; for (size_t i = 0; i < total; i++) { double e = fabs(out[i]-ref[i]); if (e>md) md=e; }
            if (md > 1e-9) { rejected++; c2r_plan_destroy(p); continue; }

            int reps = (int)(2e6/(double)(total+1)); if (reps < 20) reps = 20; if (reps > 100000) reps = 100000;
            for (int w = 0; w < 10; w++) c2r_execute_packed(p, spec, out);
            double ns = 1e18;
            for (int tr = 0; tr < 5; tr++) {
                double t0 = vfft_proto_now_ns();
                for (int i = 0; i < reps; i++) c2r_execute_packed(p, spec, out);
                double e = (vfft_proto_now_ns() - t0)/reps; if (e < ns) ns = e;
            }
            c2r_plan_destroy(p);
            benched++;
            if (ns < best_ns) {
                best_ns = ns; best_nf = fz->nf;
                memcpy(best_factors, fz->factors, sizeof best_factors);
                memcpy(best_variants, variant, sizeof best_variants);
            }
        }
    }
    if (best_nf == 0) { fprintf(stderr, "N=%d K=%zu: no candidate survived the gate\n", N, K); return 1; }

    /* roundtrip sanity of the winner: c2r(rfft(x)) ?= N*x */
    c2r_plan_t *bp = c2r_plan_create_ex(N, K, best_factors, best_nf, best_variants, &reg);
    double rt = 0;
    if (bp) {
        memset(out, 0, total * sizeof(double));
        c2r_execute_packed(bp, spec, out);
        for (size_t i = 0; i < total; i++) { double e = fabs(out[i] - (double)N * x[i]); if (e>rt) rt=e; }
        c2r_plan_destroy(bp);
    }

    if (verbose)
        printf("  [c2r] %d factorizations, %d benched, %d gate-rejected\n", g_nfz, benched, rejected);

    const char *wp = getenv("VFFT_PROTO_C2R_WIS"); if (!wp) wp = C2R_WIS;
    vfft_proto_wisdom_t wis; memset(&wis, 0, sizeof wis);
    vfft_proto_wisdom_load(&wis, wp);
    vfft_proto_wisdom_entry_t e; memset(&e, 0, sizeof e);
    e.N = N; e.K = K; e.nf = best_nf; e.best_ns = best_ns; e.use_dif_forward = 0;
    for (int s = 0; s < best_nf; s++) { e.factors[s] = best_factors[s]; e.variants[s] = best_variants[s]; }
    int rc = vfft_proto_wisdom_add(&wis, &e, /*overwrite=*/1);
    int sv = vfft_proto_wisdom_save(&wis, wp);

    printf("=== CALIBRATED c2r N=%d K=%zu: ", N, (size_t)K);
    for (int s = 0; s < best_nf; s++) printf("%s%d", s ? "x" : "", best_factors[s]);
    printf("  v=[");
    for (int s = 0; s < best_nf; s++) printf("%s%d", s ? "," : "", best_variants[s]);
    printf("]  %.1f ns  (roundtrip err %.1e, wisdom %s)\n",
           best_ns, rt, (rc != 0 && sv == 0) ? "written" : "WRITE FAILED");
    return 0;
}
