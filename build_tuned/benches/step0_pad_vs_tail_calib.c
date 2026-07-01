/* step0_pad_vs_tail_calib.c — STEP 0 of the padding plan (docs/roadmap/tail_handling/
 * padding_design_decision.md). For each misaligned-K cell (K % VW != 0), find the BEST
 * factorization at K (tail) AND at Kp=roundup(K,VW) (pad) via the production DP planner —
 * the "joint search" (wrinkle A) the standalone bench skipped — then time, on the padded
 * caller's Kp-wide buffer:
 *     tail_ns = best-K-factorization run at me=K  (SSE2/scalar tail)
 *     pad_ns  = best-Kp-factorization run at me=Kp (pure full-SIMD, junk lanes discarded)
 * with a tight interleaved-median A/B (per-round order-flip cancels thermal drift), 3%
 * hysteresis toward the tail on near-ties. Records exec_me (Kp=PAD / K=tail) into a v6
 * wisdom file and prints the per-cell verdict.
 *
 * Two DP contexts (one at K, one at Kp) — each sized for its own width, so the pad leg
 * does NOT overflow a K-sized buffer (the verified calibrator gotcha). variants default to
 * T1S (the DP default), which sidesteps the log3/t1p batch-table sizing subtlety.
 *
 * Does NOT modify the live _calibrate_c2c — this is the isolated measurement; folding it in
 * is the next step once the numbers are trusted.
 *
 * Build: build_tuned/build.py --src benches/step0_pad_vs_tail_calib.c --compile   (no --mkl)
 * Run from build_tuned/benches/ ; writes step0_exec_me_wisdom.txt (v6).
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "executor.h"
#include "planner.h"
#include "dp_planner.h"
#include "wisdom_reader.h"
#ifdef VFFT_USE_JIT
#include "jit/jit_runtime.h"   /* wrinkle C: bench the pad leg on the baked/JIT fast path */
#endif

#define VW 4
static size_t roundup_vw(size_t k) { return (k + (VW - 1)) & ~(size_t)(VW - 1); }

static double *ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) { fprintf(stderr, "alloc\n"); exit(1); }
    return p;
}
static void afree(double *p) { vfft_proto_aligned_free(p); }

/* K real columns random, (Kp-K) pad columns ZERO (no denormals in junk lanes). */
static void fillK(double *re, double *im, int N, size_t K, size_t Kp)
{
    srand(7 + N + (int)K);
    for (size_t e = 0; e < (size_t)N; e++)
        for (size_t b = 0; b < Kp; b++) {
            re[e * Kp + b] = (b < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0;
            im[e * Kp + b] = (b < K) ? (double)rand() / RAND_MAX - 0.5 : 0.0;
        }
}
/* jf != NULL -> bench on the baked/JIT executor (the pad leg, aligned me); NULL -> generic
 * (the tail leg — odd me, which the baked/JIT path is gated out of). Same call convention
 * as fft2d.h: fn(plan, re, im, slice_K, plan->K, is_bwd=0). */
static double burst(stride_plan_t *p, vfft_proto_exec_fn jf, double *re, double *im, size_t me, int reps)
{
    double t0 = vfft_proto_now_ns();
    if (jf) for (int i = 0; i < reps; i++) jf(p, re, im, me, p->K, 0);
    else    for (int i = 0; i < reps; i++) vfft_proto_execute_fwd(p, re, im, me);
    return vfft_proto_now_ns() - t0;
}
static int dcmp(const void *a, const void *b) { double d = *(const double *)a - *(const double *)b; return d < 0 ? -1 : d > 0 ? 1 : 0; }
static double med(double *v, int n) { qsort(v, n, sizeof(double), dcmp); return n & 1 ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]); }
static void facstr(const vfft_proto_factorization_t *ft, char *o, size_t n)
{
    int p = 0;
    for (int i = 0; i < ft->nfactors; i++) p += snprintf(o + p, n - p, "%s%d", i ? "." : "", ft->factors[i]);
    if (ft->nfactors == 0) snprintf(o, n, "-");
}

int main(void)
{
    setvbuf(stdout, NULL, _IONBF, 0);   /* unbuffered: see per-cell progress live + survive a kill */
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);

    int Ns[] = {256, 512, 1024, 2048, 4096};
    size_t Ks[] = {7, 11, 15, 19, 23, 27, 31};   /* misaligned (K%4 != 0) */
    int nN = 5, nK = 7, rounds = 151;

    vfft_proto_wisdom_t W;
    memset(&W, 0, sizeof W);

    printf("# STEP 0 — joint-search pad-vs-tail (best factorization PER LEG). exec_me=Kp -> PAD, =K -> tail.\n");
    printf("# tail = best-K-factorization @ me=K ; pad = best-Kp-factorization @ me=Kp ; both on the Kp buffer.\n");
    printf("# interleaved-median A/B, 3%% hysteresis toward tail. ratio = pad/tail (<1 => pad faster).\n");
    printf("# pad leg runs on the baked/JIT path when built --jit (aligned me=Kp); tail leg always generic.\n");
    printf("#  N     K   Kp   factK(tail)   factKp(pad)   tail_ns    pad_ns   pad/tail  padexec exec_me  winner\n");

    for (int ni = 0; ni < nN; ni++) {
        int N = Ns[ni];
        for (int c = 0; c < nK; c++) {
            size_t K = Ks[c], Kp = roundup_vw(K);

            /* DP search at K — best TAIL factorization (tail penalty biases stage count). */
            vfft_proto_dp_context_t ctxK;
            vfft_proto_dp_init(&ctxK, K, N);
            vfft_proto_factorization_t factK; memset(&factK, 0, sizeof factK);
            (void)vfft_proto_dp_plan(&ctxK, N, &reg, &factK, 0);
            vfft_proto_dp_destroy(&ctxK);

            /* DP search at Kp — best PAD factorization (no tail penalty; own context, no overflow). */
            vfft_proto_dp_context_t ctxKp;
            vfft_proto_dp_init(&ctxKp, Kp, N);
            vfft_proto_factorization_t factKp; memset(&factKp, 0, sizeof factKp);
            (void)vfft_proto_dp_plan(&ctxKp, N, &reg, &factKp, 0);
            vfft_proto_dp_destroy(&ctxKp);

            if (factK.nfactors <= 0 || factKp.nfactors <= 0) { printf("  %-5d %-3zu  DP failed\n", N, K); continue; }

            /* Build BOTH plans at Kp-stride (the padded caller's buffer). variants=NULL -> T1S. */
            stride_plan_t *pT = vfft_proto_plan_create_ex(N, Kp, factK.factors, NULL, factK.nfactors, 0, &reg);
            stride_plan_t *pP = vfft_proto_plan_create_ex(N, Kp, factKp.factors, NULL, factKp.nfactors, 0, &reg);
            if (!pT || !pP) {
                printf("  %-5d %-3zu  plan build failed\n", N, K);
                if (pT) vfft_proto_plan_destroy(pT);
                if (pP) vfft_proto_plan_destroy(pP);
                continue;
            }

            /* Wrinkle C: resolve the pad plan's baked/JIT executor (me=Kp is VW-aligned, so it
             * is eligible). The tail leg stays generic — odd me=K is gated out of baked/JIT.
             * Under a non-JIT build jfP stays NULL and both legs are generic (the floor). */
            vfft_proto_exec_fn jfP = NULL;
            const char *padpath = "generic";
#ifdef VFFT_USE_JIT
            int bakedP = (vfft_proto_lookup_fwd_avx2(pP) != NULL);
            jfP = vfft_proto_plan_jit_fwd(pP);
            padpath = jfP ? (bakedP ? "baked" : "JIT") : "generic";
#endif

            double *rT = ad((size_t)N * Kp), *iT = ad((size_t)N * Kp);
            double *rP = ad((size_t)N * Kp), *iP = ad((size_t)N * Kp);
            fillK(rT, iT, N, K, Kp);
            fillK(rP, iP, N, K, Kp);

            int reps = (int)(8000000ull / ((size_t)N * Kp));
            if (reps < 40) reps = 40;
            for (int w = 0; w < 8; w++) { burst(pT, NULL, rT, iT, K, reps); burst(pP, jfP, rP, iP, Kp, reps); }

            static double rt[256], rpd[256];
            int RR = rounds; if (RR > 256) RR = 256;
            for (int r = 0; r < RR; r++) {
                double t, p;
                if (r & 1) { t = burst(pT, NULL, rT, iT, K, reps);  p = burst(pP, jfP, rP, iP, Kp, reps); }
                else       { p = burst(pP, jfP, rP, iP, Kp, reps); t = burst(pT, NULL, rT, iT, K, reps); }
                rt[r] = t / reps; rpd[r] = p / reps;   /* per-call ns */
            }
            double tail_ns = med(rt, RR), pad_ns = med(rpd, RR);
            int exec_me = (pad_ns < tail_ns * 0.97) ? (int)Kp : (int)K;   /* 3% hysteresis toward tail */

            char fk[64], fkp[64]; facstr(&factK, fk, sizeof fk); facstr(&factKp, fkp, sizeof fkp);
            printf("  %-5d %-3zu %-3zu  %-13s %-13s %8.0f  %8.0f  %.3f   %-7s %-6d  %s\n",
                   N, K, Kp, fk, fkp, tail_ns, pad_ns, pad_ns / tail_ns, padpath, exec_me,
                   exec_me == (int)Kp ? "PAD" : "tail");

            /* Wisdom entry: store the TIGHT (K-optimal) factorization + exec_me. (The pad
             * factorization is printed for analysis; persisting it is a Step-1 refinement.) */
            vfft_proto_wisdom_entry_t ent; memset(&ent, 0, sizeof ent);
            ent.N = N; ent.K = K; ent.nf = factK.nfactors;
            for (int i = 0; i < factK.nfactors; i++) { ent.factors[i] = factK.factors[i]; ent.variants[i] = 2; }
            ent.use_dif_forward = 0;
            ent.best_ns = (exec_me == (int)Kp) ? pad_ns : tail_ns;
            ent.exec_me = exec_me;
            vfft_proto_wisdom_set(&W, &ent);

            vfft_proto_plan_destroy(pT); vfft_proto_plan_destroy(pP);
            afree(rT); afree(iT); afree(rP); afree(iP);
        }
    }

    vfft_proto_wisdom_save(&W, "step0_exec_me_wisdom.txt");
    vfft_proto_wisdom_free(&W);
    printf("\nwrote step0_exec_me_wisdom.txt (v6, exec_me populated for the cells above)\n");
    return 0;
}
