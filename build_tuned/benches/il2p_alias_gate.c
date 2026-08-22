/* il2p_alias_gate.c — can the OUT-OF-PLACE il2p path skip the mid scratch?
 *
 * THE PROPOSED CHANGE. vfft_il2p_execute_fwd always routes through the
 * plan-owned scratch:
 *     leaf_f(zin,  0, p->mid, 0, ...);      stage 1: corner turn (a SCATTER)
 *     mid_f (mid,  0, zout,   0, ...);      stage 2: identity map (Ls == OLs)
 * Only stage 1 scatters. So when zin != zout the scratch is unnecessary — the
 * leaf could write straight into zout and the mid could run IN PLACE on it,
 * removing one 2N-double plane from the working set (~16 KB at N=2048,
 * against this machine's 48 KB L1d). The backward path has the same shape:
 * t2t_b scatters, n1_b_r2 is the identity map.
 *
 * WHY A GATE AND NOT AN ARGUMENT. Stage 2 running aliased is safe only if the
 * kernel issues ALL its loads before ANY of its stores. That holds by
 * inspection for the monolithic forms, but the BLOCKED and TANGENT variants
 * restructure the math into sub-blocks, and a sub-block that stores before a
 * later one loads would corrupt silently — wrong numbers, no crash. vfft.c's
 * own rule is that IL alias-safety must be "verified per family".
 *
 * WHAT THIS GATE PROVES, per (R1, R2) pair x kernel-form variant x direction:
 *   run the shipped scratch path and the proposed aliased path on identical
 *   input and compare BITWISE. Not a tolerance — the two must compute the
 *   same thing or the optimization is not equivalent.
 *
 * ZERO timing. Deterministic. Safe on a noisy machine.
 *
 * Build: python build.py --src benches/il2p_alias_gate.c --compile
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

#include "il2p.h"

static int g_fail = 0, g_pairs = 0, g_arms = 0, g_skipped = 0;

static uint64_t lcg = 0x243F6A8885A308D3ull;
static double rnd(void)
{
    lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
    return ((double)(int64_t)(lcg >> 11)) / 4503599627370496.0;
}

/* The proposed out-of-place staging: leaf straight into zout, mid in place. */
static void exec_fwd_noscratch(const vfft_il2p_plan_t *p,
                               const double *zin, double *zout)
{
    const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
    p->leaf_f(zin,  0, zout, 0, 0,      0, R1, 0, R2, 0, R1);
    p->mid_f (zout, 0, zout, 0, p->tw,  0, R2, 0, R2, 0, R2);
}
static int exec_bwd_noscratch(const vfft_il2p_plan_t *p,
                              const double *zin, double *zout)
{
    const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
    if (!p->t2t_b || !p->n1_b_r2) return -1;
    p->t2t_b   (zin,  0, zout, 0, p->twb, 0, R2, 0, R1, 0, R2);
    p->n1_b_r2 (zout, 0, zout, 0, 0,      0, R1, 0, R1, 0, R1);
    return 0;
}

/* One arm: same input, both stagings, bitwise compare. */
static void arm(int R1, int R2, int kv, int bkv, int bwd)
{
    const int N = R1 * R2;
    const size_t nd = (size_t)N * 2u;
    vfft_il2p_plan_t *p = vfft_il2p_create(N, R1, R2);
    double *zin, *a, *b;
    int differs;

    if (!p) return;
    if (vfft_il2p_apply_kv_forms(p, kv) != 0 ||
        vfft_il2p_apply_kv_forms_bwd(p, bkv) != 0) {
        vfft_il2p_destroy(p);          /* variant has no kernel here */
        g_skipped++;
        return;
    }

    zin = (double *)malloc(nd * sizeof(double));
    a   = (double *)malloc(nd * sizeof(double));
    b   = (double *)malloc(nd * sizeof(double));
    if (!zin || !a || !b) { free(zin); free(a); free(b); vfft_il2p_destroy(p); return; }
    for (size_t i = 0; i < nd; i++) zin[i] = rnd();
    memset(a, 0, nd * sizeof(double));
    memset(b, 0, nd * sizeof(double));

    if (bwd) {
        if (vfft_il2p_execute_bwd(p, zin, a) != 0 ||
            exec_bwd_noscratch(p, zin, b) != 0) {
            free(zin); free(a); free(b); vfft_il2p_destroy(p);
            g_skipped++;
            return;
        }
    } else {
        vfft_il2p_execute_fwd(p, zin, a);
        exec_fwd_noscratch(p, zin, b);
    }

    g_arms++;
    differs = (memcmp(a, b, nd * sizeof(double)) != 0);
    if (differs) {
        /* report HOW wrong, so a rounding difference is distinguishable from
         * a corruption -- they mean different things for this decision */
        double worst = 0.0, mag = 0.0;
        size_t first = (size_t)-1;
        for (size_t i = 0; i < nd; i++) {
            double d = fabs(a[i] - b[i]);
            if (d > worst) worst = d;
            if (fabs(a[i]) > mag) mag = fabs(a[i]);
            if (d != 0.0 && first == (size_t)-1) first = i;
        }
        printf("  %-4s %2dx%-2d kv=0x%02x bkv=0x%02x  *** DIFFERS *** "
               "first@%zu rel %.3e\n",
               bwd ? "bwd" : "fwd", R1, R2, kv, bkv, first,
               mag > 0 ? worst / mag : worst);
        g_fail = 1;
    }
    free(zin); free(a); free(b);
    vfft_il2p_destroy(p);
}

/* NEGATIVE CONTROL. A gate that has never failed is unproven: 237 identical
 * arms could equally mean the two paths are accidentally the same code. This
 * runs the proposed staging with a DELIBERATELY wrong mid stride, which must
 * be detected. If this reports IDENTICAL, the comparison itself is broken and
 * every green result above it is meaningless. */
static int selftest(void)
{
    const int R1 = 16, R2 = 16, N = R1 * R2;
    const size_t nd = (size_t)N * 2u;
    vfft_il2p_plan_t *p = vfft_il2p_create(N, R1, R2);
    double *zin, *a, *b;
    int caught;
    if (!p) { printf("  selftest: could not build 16x16\n"); return 1; }
    zin = (double *)malloc(nd * sizeof(double));
    a   = (double *)malloc(nd * sizeof(double));
    b   = (double *)malloc(nd * sizeof(double));
    if (!zin || !a || !b) { free(zin); free(a); free(b); vfft_il2p_destroy(p); return 1; }
    for (size_t i = 0; i < nd; i++) zin[i] = rnd();
    vfft_il2p_execute_fwd(p, zin, a);
    /* the injected fault: mid told the WRONG output lattice */
    p->leaf_f(zin, 0, b, 0, 0,     0, (size_t)R1, 0, (size_t)R2, 0, (size_t)R1);
    p->mid_f (b,   0, b, 0, p->tw, 0, (size_t)R2, 0, (size_t)R2, 0, (size_t)R2 / 2);
    caught = (memcmp(a, b, nd * sizeof(double)) != 0);
    printf("  negative control (wrong mid stride): %s\n",
           caught ? "DETECTED -- the comparison has teeth"
                  : "*** NOT DETECTED -- THIS GATE PROVES NOTHING ***");
    free(zin); free(a); free(b);
    vfft_il2p_destroy(p);
    return caught ? 0 : 1;
}

int main(void)
{
    static const int RAD[] = { 4, 8, 16, 32, 64 };
    setvbuf(stdout, NULL, _IONBF, 0);
    printf("il2p ALIAS gate — can out-of-place skip the mid scratch?\n");
    printf("  (bitwise: shipped scratch staging vs leaf->zout + mid in place)\n\n");

    for (int Ni = 0; Ni < 5; Ni++)
        for (int i = 0; i < 5; i++) {
            const int R2 = RAD[i];
            const int R1 = RAD[Ni];
            if (R1 < 4 || R1 > 64) continue;
            if (!vfft_il2p_leaf_fn(R2, 0) || !vfft_il2p_mid_fn(R1, 0)) continue;
            g_pairs++;
            /* every expressible form combination, both directions */
            for (int mv = 0; mv <= 4; mv++)
                for (int lv = 0; lv <= 4; lv++) {
                    const int kv = VFFT_IL_KV_PACK(mv, lv);
                    arm(R1, R2, kv, 0, 0);
                    arm(R1, R2, 0, kv, 1);
                }
        }

    printf("\n  %d pair(s), %d arm(s) compared, %d skipped (no such kernel)\n",
           g_pairs, g_arms, g_skipped);
    if (selftest() != 0) g_fail = 1;
    if (!g_arms) { printf("  *** FAIL *** no arms ran — the gate proved nothing\n"); g_fail = 1; }
    printf("\n%s\n", g_fail ? "IL2P ALIAS GATE: NOT SAFE TO SKIP THE SCRATCH"
                            : "IL2P ALIAS GATE: BITWISE IDENTICAL — safe to skip");
    return g_fail;
}
