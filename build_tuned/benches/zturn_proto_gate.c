/* zturn_proto_gate.c — correctness gates for the ZTURN-S full-cascade C
 * prototype (src/core/oop/zturn_proto.h) against PRODUCTION zsplit
 * (src/core/oop/zsplit.h), per the canonical map spec (Phase 2), HARDENED
 * with the adversarial attack pass's recommendations (docs/research/
 * zturns_consensus_and_prototype.md §8a).
 *
 * GATE 1 (fwd):  vfft_zturn_execute_fwd == vfft_zsplit_execute_fwd under
 *   Gamma: out_z[l*(N/8)+4k'+j] == out_legacy[l*(N/8)+j*(N/32)+k'].
 *   EXACT (memcmp per double). Why exact: the ingest's sums are COMMUTED
 *   twins of radix4_z_s0s_fwd's (IEEE commutativity, proven by simulator
 *   gate A); the mids are the production kernel bytes on bit-equal table
 *   entries (simulator gates D+F); stf transcribes radix8_z_sterm_fwd's
 *   op graph 1:1 (FMA placement preserved, intrinsics only).
 *   Runs over --seeds=K random vectors (default 1; gate runs use 5).
 * GATE 2 (bwd):  same, backward, two combs: (a) the fwd outputs,
 *   (b) an independent random comb fed through Gamma. EXACT (memcmp),
 *   same justification (stfb/s0tb transcribe sterm_bwd/s0s_bwd graphs).
 *   Runs over the same K seeds.
 * GATE 3 (roundtrip): zturn_bwd(zturn_fwd(x)) == N*x (zsplit convention:
 *   "Roundtrip = N*x, no 1/N in-kernel") at 1e-15 class — the ONLY
 *   tolerance gate: association order legitimately differs across the
 *   full transform (fwd stage tree vs bwd stage tree do not cancel
 *   term-by-term). PASS bar 8e-15; simulator measured <= 1.332e-15.
 *   Reported as the MAX over all K seeds.
 * GATE 4 (matched pair): P_bwd(P_fwd) == identity as INDEX SETS (pure
 *   integer): rho_inv(rho(k')) == k' with rho_inv = digit reversal over
 *   the REVERSED middle radices, plus full-index bijection via flags.
 * GATE 5 (static census) runs outside this driver (objdump script).
 * GATE 6 (naive-DFT data anchor, N=2048 AND 4096): O(N^2) direct DFT of
 *   the seed-0 input vs out_z through P_fwd:
 *   out_z[l*(N/8)+4k'+j] == X[l*(N/8)+4*rho(k')+j], rel bar 1e-13 (scalar
 *   libm anchor, association differs; attack pass measured <= 7.3e-15).
 *   4096's middle chain (4.4.8) is NON-PALINDROMIC — this pins the digit-
 *   reversal ORIENTATION by data, not by shared convention (2048's 8.8.8
 *   middle chain is palindromic and constitutionally blind to it).
 *   n/a at 8192/16384 (small-N anchor; orientation already pinned by 4096).
 * GATE 7 (in-place): vfft_zturn_execute_{fwd,bwd}(zin==zout) EXACT
 *   (memcmp) vs the OOP outputs — the header's "zin == zout OK" claim,
 *   previously ungated in this driver.
 *
 * A MAP failure prints the first differing element index and its digit
 * decomposition and exits nonzero — no address fiddling in here.
 *
 * USAGE: zturn_proto_gate [--seeds=K]     (default K=1; gate runs use 5)
 * EXIT:  0 all gates pass, 1 gate failure, 2 create/alloc failure.
 *
 * BUILD (gcc 15.2, from anywhere; KDIR = codelets/zil/avx2):
 *   gcc -O3 -mavx2 -mfma -c KDIR/radix{4,8}_z_{s0s,msg}[_bwd]_avx2.c
 *                            KDIR/radix8_z_sterm{,2,_bwd}_avx2.c
 *   gcc -O3 -mavx2 -mfma -I src/core/oop -c zturn_proto_gate.c
 *   gcc -O3 -mavx2 -mfma zturn_proto_gate.o <11 kernel .o> -o gate.exe -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <math.h>
#include "zsplit.h"
#include "zturn_proto.h"

static uint64_t g_rng;
static double rnd(void)
{
    g_rng = g_rng * 6364136223846793005ULL + 1442695040888963407ULL;
    return 2.0 * ((double)(g_rng >> 11) * 0x1p-53) - 1.0;
}

typedef struct { long bad; size_t first; double a, b; int seed; } cmp_t;
static void cmp_acc(cmp_t *c, double a, double b, size_t idx, int seed)
{
    if (memcmp(&a, &b, 8) != 0) {
        if (c->bad == 0) { c->first = idx; c->a = a; c->b = b; c->seed = seed; }
        c->bad++;
    }
}

/* digit decomposition of a zturn comb double-index (first-fail forensics) */
static void decomp(int N, int nf, const int *chain, size_t d)
{
    const size_t o = d / 2, im = d & 1;
    const size_t l = o / (size_t)(N / 8), rem = o % (size_t)(N / 8);
    const size_t k2 = rem / 4, j = rem % 4;
    printf("      first diff: double %zu = complex %zu (%s) -> l=%zu k'=%zu j=%zu; k' digits (r1..r%d):",
           d, o, im ? "im" : "re", l, k2, j, nf - 2);
    long v = (long)k2;
    long dig[VFFT_ZSPLIT_MAX_NF];
    for (int s = nf - 3; s >= 0; s--) { dig[s] = v % chain[1 + s]; v /= chain[1 + s]; }
    for (int s = 0; s < nf - 2; s++) printf(" %ld", dig[s]);
    printf("\n");
}

int main(int argc, char **argv)
{
    static const int NS[4] = { 2048, 4096, 8192, 16384 };
    int seeds = 1;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "--seeds=", 8) == 0) {
            seeds = atoi(argv[i] + 8);
            if (seeds < 1 || seeds > 1000) {
                fprintf(stderr, "FATAL: --seeds=K, K in 1..1000\n"); return 2;
            }
        } else {
            fprintf(stderr, "usage: zturn_proto_gate [--seeds=K]\n"); return 2;
        }
    }
    int g1f = 0, g2f = 0, g3f = 0, g4f = 0, g6f = 0, g7f = 0;
    printf("== ZTURN-S prototype gates vs production zsplit (seeds=%d) ==\n",
           seeds);
    printf("gate 1/2 EXACT (memcmp): commuted-twin ingest sums, byte-identical\n"
           "mid kernels on bit-equal tables, 1:1 sterm/sterm_bwd/s0s_bwd op-graph\n"
           "transcriptions. gate 3 tolerance (8e-15): association order across\n"
           "the full fwd+bwd transform legitimately differs. gate 4 integer.\n"
           "gate 6 naive-DFT anchor (rel bar 1e-13, 2048+4096; 4096 middle chain\n"
           "non-palindromic => orientation pinned by data). gate 7 in-place EXACT.\n\n");
    for (int ni = 0; ni < 4; ni++) {
        const int N = NS[ni];
        int chain[VFFT_ZSPLIT_MAX_NF];
        const int nf = vfft_zsplit_default_chain(N, chain);
        vfft_zsplit_plan_t *pp = vfft_zsplit_create(N, chain, nf);
        vfft_zturn_plan_t  *zp = vfft_zturn_create(N);
        if (!pp || !zp) { printf("N=%d create FAIL\n", N); return 2; }

        const size_t nd = 2 * (size_t)N, K2 = (size_t)N / 32, OL = (size_t)N / 8;
        double *x       = (double *)malloc(nd * 8);
        double *y       = (double *)malloc(nd * 8);
        double *y_z     = (double *)malloc(nd * 8);
        double *out_ref = (double *)malloc(nd * 8);
        double *out_z   = (double *)malloc(nd * 8);
        double *back_ref= (double *)malloc(nd * 8);
        double *back_z  = (double *)malloc(nd * 8);
        double *by_ref  = (double *)malloc(nd * 8);
        double *by_z    = (double *)malloc(nd * 8);
        double *x0      = (double *)malloc(nd * 8);   /* seed-0 stash: input */
        double *oz0     = (double *)malloc(nd * 8);   /* seed-0 zturn fwd    */
        double *bz0     = (double *)malloc(nd * 8);   /* seed-0 zturn bwd    */
        double *ip      = (double *)malloc(nd * 8);   /* in-place buffer     */
        if (!x || !y || !y_z || !out_ref || !out_z || !back_ref || !back_z
            || !by_ref || !by_z || !x0 || !oz0 || !bz0 || !ip) {
            printf("N=%d alloc FAIL\n", N); return 2;
        }

        cmp_t g1 = { 0, 0, 0, 0, 0 }, g2a = { 0, 0, 0, 0, 0 },
              g2b = { 0, 0, 0, 0, 0 };
        double rmax_all = 0.0;

        for (int sd = 0; sd < seeds; sd++) {
            /* sd==0 reproduces the original single-seed gate run exactly */
            g_rng = 0x5EED0000 + (uint64_t)N
                  + 0x9E3779B97F4A7C15ULL * (uint64_t)sd;
            for (size_t i = 0; i < nd; i++) x[i] = rnd();
            for (size_t i = 0; i < nd; i++) y[i] = rnd();

            /* ------------- GATE 1: forward (per seed) ------------- */
            vfft_zsplit_execute_fwd(pp, x, out_ref);
            vfft_zturn_execute_fwd(zp, x, out_z);
            for (size_t l = 0; l < 8; l++)
                for (size_t k2 = 0; k2 < K2; k2++)
                    for (size_t j = 0; j < 4; j++) {
                        const size_t zi = l * OL + 4 * k2 + j;
                        const size_t li = l * OL + j * K2 + k2;
                        cmp_acc(&g1, out_z[2 * zi],     out_ref[2 * li],     2 * zi,     sd);
                        cmp_acc(&g1, out_z[2 * zi + 1], out_ref[2 * li + 1], 2 * zi + 1, sd);
                    }

            /* ------------- GATE 2: backward (per seed) ------------ */
            vfft_zsplit_execute_bwd(pp, out_ref, back_ref);
            vfft_zturn_execute_bwd(zp, out_z, back_z);
            for (size_t i = 0; i < nd; i++)
                cmp_acc(&g2a, back_z[i], back_ref[i], i, sd);
            /* independent comb through Gamma */
            for (size_t l = 0; l < 8; l++)
                for (size_t k2 = 0; k2 < K2; k2++)
                    for (size_t j = 0; j < 4; j++) {
                        const size_t zi = l * OL + 4 * k2 + j;
                        const size_t li = l * OL + j * K2 + k2;
                        y_z[2 * zi]     = y[2 * li];
                        y_z[2 * zi + 1] = y[2 * li + 1];
                    }
            vfft_zsplit_execute_bwd(pp, y, by_ref);
            vfft_zturn_execute_bwd(zp, y_z, by_z);
            for (size_t i = 0; i < nd; i++)
                cmp_acc(&g2b, by_z[i], by_ref[i], i, sd);

            /* ------------- GATE 3: roundtrip (per seed) ----------- */
            for (size_t i = 0; i < nd; i++) {
                const double d = fabs(back_z[i] - (double)N * x[i]) / (double)N;
                if (d > rmax_all) rmax_all = d;
            }

            if (sd == 0) {                       /* stash for gates 6/7 */
                memcpy(x0,  x,      nd * 8);
                memcpy(oz0, out_z,  nd * 8);
                memcpy(bz0, back_z, nd * 8);
            }
        }

        printf("N=%5d  GATE1 fwd  (EXACT, seeds=%d): %ld/%zu mismatched  %s\n",
               N, seeds, g1.bad, (size_t)seeds * nd, g1.bad ? "FAIL" : "PASS");
        if (g1.bad) { decomp(N, nf, chain, g1.first);
                      printf("      seed=%d zturn=%.17g legacy=%.17g\n",
                             g1.seed, g1.a, g1.b); g1f++; }

        printf("N=%5d  GATE2 bwd  (EXACT, seeds=%d): fwd-comb %ld/%zu, "
               "indep-comb %ld/%zu  %s\n",
               N, seeds, g2a.bad, (size_t)seeds * nd, g2b.bad,
               (size_t)seeds * nd, (g2a.bad || g2b.bad) ? "FAIL" : "PASS");
        if (g2a.bad || g2b.bad) {
            const cmp_t *w = g2a.bad ? &g2a : &g2b;
            printf("      first diff: natural double %zu (complex %zu %s) "
                   "seed=%d: zturn=%.17g legacy=%.17g\n",
                   w->first, w->first / 2, (w->first & 1) ? "im" : "re",
                   w->seed, w->a, w->b);
            g2f++;
        }

        printf("N=%5d  GATE3 roundtrip |bwd(fwd(x))/N - x| max over %d seeds "
               "= %.3e  %s\n",
               N, seeds, rmax_all, (rmax_all <= 8e-15) ? "PASS" : "FAIL");
        if (rmax_all > 8e-15) g3f++;

        /* ---------------- GATE 4: matched pair (integer) ---------------- */
        {
            int revmid[VFFT_ZSPLIT_MAX_NF];
            for (int i = 0; i < nf - 2; i++) revmid[i] = chain[nf - 2 - i];
            long bad = 0;
            unsigned char *seen = (unsigned char *)calloc((size_t)N, 1);
            for (size_t k2 = 0; k2 < K2; k2++) {
                const long r = _vfft_zs_brev((long)k2, nf - 2, chain + 1);
                if (_vfft_zs_brev(r, nf - 2, revmid) != (long)k2) bad++;
            }
            for (size_t l = 0; l < 8; l++)
                for (size_t k2 = 0; k2 < K2; k2++)
                    for (size_t j = 0; j < 4; j++) {
                        const size_t o = l * OL + 4 * k2 + j;
                        /* P_fwd: out[o] = X[t] */
                        const size_t t = l * OL
                            + 4 * (size_t)_vfft_zs_brev((long)k2, nf - 2, chain + 1) + j;
                        if (seen[t]) bad++;
                        seen[t] = 1;
                        /* P_inv formula (spec section 5) must return o */
                        const size_t jj = t % 4, v = (t % OL) / 4, ll = t / OL;
                        const size_t oi = ll * OL
                            + 4 * (size_t)_vfft_zs_brev((long)v, nf - 2, revmid) + jj;
                        if (oi != o) bad++;
                    }
            for (int i = 0; i < N; i++) if (!seen[i]) bad++;
            free(seen);
            printf("N=%5d  GATE4 P_fwd/P_inv bijection+inverse (integer): %ld bad  %s\n",
                   N, bad, bad ? "FAIL" : "PASS");
            if (bad) g4f++;
        }

        /* -------- GATE 6: naive-DFT data anchor (2048 AND 4096) --------- */
        if (N == 2048 || N == 4096) {
            const double TAU = 2.0 * M_PI;
            double *wr = (double *)malloc((size_t)N * 8);
            double *wi = (double *)malloc((size_t)N * 8);
            double *Xr = (double *)malloc((size_t)N * 8);
            double *Xi = (double *)malloc((size_t)N * 8);
            if (!wr || !wi || !Xr || !Xi) { printf("anchor alloc FAIL\n"); return 2; }
            for (int m = 0; m < N; m++) {          /* reduced-angle twiddles */
                const double a = -TAU * (double)m / (double)N;
                wr[m] = cos(a); wi[m] = sin(a);
            }
            double maxmag = 0.0;
            for (int t = 0; t < N; t++) {          /* O(N^2), fine here */
                double sr = 0.0, si = 0.0;
                for (int n = 0; n < N; n++) {
                    const size_t m = ((size_t)t * (size_t)n) % (size_t)N;
                    sr += x0[2*n] * wr[m] - x0[2*n + 1] * wi[m];
                    si += x0[2*n] * wi[m] + x0[2*n + 1] * wr[m];
                }
                Xr[t] = sr; Xi[t] = si;
                const double mag = fabs(sr) + fabs(si);
                if (mag > maxmag) maxmag = mag;
            }
            double relmax = 0.0;
            for (size_t l = 0; l < 8; l++)
                for (size_t k2 = 0; k2 < K2; k2++)
                    for (size_t j = 0; j < 4; j++) {
                        const size_t o = l * OL + 4 * k2 + j;
                        const size_t t = l * OL
                            + 4 * (size_t)_vfft_zs_brev((long)k2, nf - 2, chain + 1) + j;
                        const double dr = fabs(oz0[2*o]     - Xr[t]);
                        const double di = fabs(oz0[2*o + 1] - Xi[t]);
                        const double rel = (dr > di ? dr : di) / maxmag;
                        if (rel > relmax) relmax = rel;
                    }
            printf("N=%5d  GATE6 naive-DFT anchor (P_fwd, seed 0): relmax=%.3e  %s\n",
                   N, relmax, (relmax <= 1e-13) ? "PASS" : "FAIL");
            if (relmax > 1e-13) g6f++;
            free(wr); free(wi); free(Xr); free(Xi);
        } else {
            printf("N=%5d  GATE6 naive-DFT anchor: n/a (small-N gate; "
                   "orientation pinned at 4096)\n", N);
        }

        /* ------------- GATE 7: in-place (zin == zout), EXACT ------------ */
        {
            memcpy(ip, x0, nd * 8);
            vfft_zturn_execute_fwd(zp, ip, ip);
            long badf = 0;
            for (size_t i = 0; i < nd; i++)
                if (memcmp(&ip[i], &oz0[i], 8)) badf++;
            memcpy(ip, oz0, nd * 8);
            vfft_zturn_execute_bwd(zp, ip, ip);
            long badb = 0;
            for (size_t i = 0; i < nd; i++)
                if (memcmp(&ip[i], &bz0[i], 8)) badb++;
            printf("N=%5d  GATE7 in-place vs OOP (EXACT): fwd %ld/%zu, "
                   "bwd %ld/%zu  %s\n",
                   N, badf, nd, badb, nd, (badf || badb) ? "FAIL" : "PASS");
            if (badf || badb) g7f++;
        }
        printf("\n");

        free(x); free(y); free(y_z); free(out_ref); free(out_z);
        free(back_ref); free(back_z); free(by_ref); free(by_z);
        free(x0); free(oz0); free(bz0); free(ip);
        vfft_zturn_destroy(zp);
        vfft_zsplit_destroy(pp);
    }
    const int fails = g1f + g2f + g3f + g4f + g6f + g7f;
    printf(fails ? "== ZTURN-S PROTOTYPE: GATES FAILED (g1=%d g2=%d g3=%d g4=%d g6=%d g7=%d) ==\n"
                 : "== ZTURN-S PROTOTYPE: ALL GATES PASS (g1=%d g2=%d g3=%d g4=%d g6=%d g7=%d fails) ==\n",
           g1f, g2f, g3f, g4f, g6f, g7f);
    return fails ? 1 : 0;
}
