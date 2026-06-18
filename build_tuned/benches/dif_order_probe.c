/* dif_order_probe.c — GATING EXPERIMENT for the recombine push.
 *
 * Two questions, answered empirically (no reasoning about DIT/DIF conventions):
 *   Q1 (DECISIVE): what output ORDERING does a DIF-forward inner c2c produce?
 *        - DIT-forward is known digit-reversed: out[perm[f]] ~= DFT[f].
 *        - If DIF-forward is NATURAL (out[f] ~= DFT[f]) -> the Hermitian recombine
 *          can read Z[k]/Z[N/2-k] at contiguous +-k offsets (no scattered perm
 *          gather), MKL-func2-style L1-blocked. THAT is the last lever.
 *        - If DIF-forward is ALSO digit-reversed (or some other order) -> the
 *          lever is dead; bank 0.89x.
 *   Q2 (tradeoff): is DIF-FLAT inner c2c slower than DIT (wisdom can pick T1S)?
 *        DIF only supports FLAT/LOG3. If DIF-FLAT >> DIT, the natural-order win
 *        may be eaten by a slower inner. Apples-to-apples: DIT-FLAT vs DIF-FLAT,
 *        same factorization.
 *
 * Pure c2c on N=128 COMPLEX input (re & im both random -> all 128 DFT bins
 * distinct, no conjugate symmetry -> unambiguous slot<->freq matching).
 *
 * Build: cd build_tuned && python build.py --src benches/dif_order_probe.c --compile
 * Run  : PATH += C:\mingw152\mingw64\bin, then run the .exe.
 */
#define _GNU_SOURCE 1
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "core/executor.h"
#include "core/env.h"
#include "core/planner.h"
#include "core/dp_planner.h"            /* vfft_proto_now_ns */
#include "core/proto_stride_compat.h"   /* vfft_proto_posix_memalign / _aligned_free */
#include "generator/generated/registry.h"

#define PIN_CORE 2

static double *alloc_d(size_t n) {
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0) {
        fprintf(stderr, "alloc failed\n"); exit(1);
    }
    return p;
}

/* digit-reversal perm (same as bench/core): DIT freq f lives at slot perm[f]. */
static void compute_perm(const int *factors, int nf, int N, int *perm) {
    for (int n = 0; n < N; n++) {
        int idx = n, rev = 0, radix_product = 1;
        for (int s = 0; s < nf; s++) {
            int R = factors[s]; int digit = idx % R; idx /= R;
            rev += digit * (N / (radix_product * R)); radix_product *= R;
        }
        perm[n] = rev;
    }
}

/* For each output SLOT s, find the frequency f whose reference DFT best matches
 * out[s] (lane 0). Returns slot->freq map + the worst match error. */
static double identify_order(const double *out_re, const double *out_im,
                             const double *ref_re, const double *ref_im,
                             int N, size_t K, int *slot2freq) {
    double worst = 0.0;
    for (int s = 0; s < N; s++) {
        double gr = out_re[(size_t)s * K + 0], gi = out_im[(size_t)s * K + 0];
        int bestf = -1; double bestd = 1e300;
        for (int f = 0; f < N; f++) {
            double dr = gr - ref_re[f], di = gi - ref_im[f];
            double d = dr*dr + di*di;
            if (d < bestd) { bestd = d; bestf = f; }
        }
        slot2freq[s] = bestf;
        double e = sqrt(bestd);
        if (e > worst) worst = e;
    }
    return worst;
}

static int is_identity(const int *map, int N) {
    for (int i = 0; i < N; i++) if (map[i] != i) return 0;
    return 1;
}
static int equals_perm(const int *map, const int *perm, int N) {
    for (int i = 0; i < N; i++) if (map[i] != perm[i]) return 0;
    return 1;
}
/* is the map a bijection (valid permutation)? */
static int is_bijection(const int *map, int N) {
    int *seen = (int *)calloc(N, sizeof(int));
    int ok = 1;
    for (int i = 0; i < N; i++) {
        if (map[i] < 0 || map[i] >= N || seen[map[i]]) { ok = 0; break; }
        seen[map[i]] = 1;
    }
    free(seen);
    return ok;
}

static double time_c2c(stride_plan_t *p, const double *master_re, const double *master_im,
                       double *re, double *im, size_t K, int N) {
    size_t bytes = (size_t)N * K * sizeof(double);
    /* warm */
    for (int w = 0; w < 10; w++) {
        memcpy(re, master_re, bytes); memcpy(im, master_im, bytes);
        vfft_proto_execute_fwd(p, re, im, K);
    }
    int reps = (int)(4e6 / ((size_t)N * K + 1)); if (reps < 50) reps = 50; if (reps > 50000) reps = 50000;
    double best = 1e18;
    for (int t = 0; t < 15; t++) {
        double t0 = vfft_proto_now_ns();
        for (int i = 0; i < reps; i++) {
            memcpy(re, master_re, bytes); memcpy(im, master_im, bytes);  /* avoid denormal blowup; equal for both */
            vfft_proto_execute_fwd(p, re, im, K);
        }
        double ns = (vfft_proto_now_ns() - t0) / reps; if (ns < best) best = ns;
    }
    return best;  /* includes the 2 memcpys, identical for DIT and DIF -> delta is pure c2c */
}

int main(void) {
    stride_env_init();
    if (stride_pin_thread(PIN_CORE) != 0) fprintf(stderr, "warn: pin cpu%d failed\n", PIN_CORE);

    const int N = 128;
    vfft_proto_registry_t reg; vfft_proto_registry_init(&reg);

    /* Same factorization for both orientations (FLAT both, apples-to-apples).
     * (4,4,8) is the wisdom-best inner at K=256; FLAT exists for 4 and 8. */
    int factors[] = {4, 4, 8};
    int nf = (int)(sizeof factors / sizeof factors[0]);
    int vflat[] = {0, 0, 0};   /* FLAT = 0 (DIF supports FLAT/LOG3 only) */

    int perm[128]; compute_perm(factors, nf, N, perm);

    printf("=== DIF-forward output ORDER probe (N=%d, factors=(4,4,8) FLAT) ===\n", N);

    /* ---- ordering at small K=8 (unambiguous, cheap) ---- */
    {
        const size_t K = 8;
        double *mr = alloc_d((size_t)N*K), *mi = alloc_d((size_t)N*K);
        double *re = alloc_d((size_t)N*K), *im = alloc_d((size_t)N*K);
        srand(12345);
        for (size_t i = 0; i < (size_t)N*K; i++) { mr[i] = (double)rand()/RAND_MAX*2-1; mi[i] = (double)rand()/RAND_MAX*2-1; }

        /* reference DFT (lane 0), natural order */
        double *ref_re = alloc_d(N), *ref_im = alloc_d(N);
        for (int f = 0; f < N; f++) {
            double sr = 0, si = 0;
            for (int n = 0; n < N; n++) {
                double xr = mr[(size_t)n*K+0], xi = mi[(size_t)n*K+0];
                double ang = -2.0*M_PI*(double)f*(double)n/(double)N;
                double c = cos(ang), s = sin(ang);
                sr += xr*c - xi*s; si += xr*s + xi*c;
            }
            ref_re[f] = sr; ref_im[f] = si;
        }

        int s2f_dit[128], s2f_dif[128];

        /* DIT-forward */
        stride_plan_t *pdit = vfft_proto_plan_create_ex(N, K, factors, vflat, nf, 0, &reg);
        if (!pdit) { printf("DIT plan NULL\n"); return 1; }
        memcpy(re, mr, (size_t)N*K*sizeof(double)); memcpy(im, mi, (size_t)N*K*sizeof(double));
        vfft_proto_execute_fwd(pdit, re, im, K);
        double edit = identify_order(re, im, ref_re, ref_im, N, K, s2f_dit);

        /* DIF-forward */
        stride_plan_t *pdif = vfft_proto_plan_create_ex(N, K, factors, vflat, nf, 1, &reg);
        if (!pdif) { printf("DIF plan NULL (DIF-forward FLAT not buildable for (4,4,8))\n"); return 1; }
        memcpy(re, mr, (size_t)N*K*sizeof(double)); memcpy(im, mi, (size_t)N*K*sizeof(double));
        vfft_proto_execute_fwd(pdif, re, im, K);
        double edif = identify_order(re, im, ref_re, ref_im, N, K, s2f_dif);

        printf("\n[DIT] match err=%.2e  bijection=%d  identity(natural)=%d  ==digitreverse=%d\n",
               edit, is_bijection(s2f_dit,N), is_identity(s2f_dit,N), equals_perm(s2f_dit,perm,N));
        printf("[DIF] match err=%.2e  bijection=%d  identity(natural)=%d  ==digitreverse=%d\n",
               edif, is_bijection(s2f_dif,N), is_identity(s2f_dif,N), equals_perm(s2f_dif,perm,N));

        /* show the first 16 slot->freq for DIF so we can eyeball the structure */
        printf("[DIF] slot->freq (first 16): ");
        for (int s = 0; s < 16; s++) printf("%d ", s2f_dif[s]);
        printf("\n");
        printf("[perm] digit-rev   (first 16): ");
        for (int s = 0; s < 16; s++) printf("%d ", perm[s]);
        printf("\n");

        if (edif > 1e-7)
            printf("\n*** DIF-forward FLAT is NOT a correct DFT (err %.2e) — orientation unusable.\n", edif);
        else if (is_identity(s2f_dif, N))
            printf("\n*** RESULT: DIF-forward = NATURAL ORDER. Contiguous recombine is GO. ***\n");
        else if (equals_perm(s2f_dif, perm, N))
            printf("\n*** RESULT: DIF-forward = digit-reversed (same as DIT). Lever DEAD. ***\n");
        else
            printf("\n*** RESULT: DIF-forward = some OTHER permutation (see map). Needs analysis. ***\n");

        stride_plan_destroy(pdit); stride_plan_destroy(pdif);
        vfft_proto_aligned_free(mr); vfft_proto_aligned_free(mi);
        vfft_proto_aligned_free(re); vfft_proto_aligned_free(im);
        vfft_proto_aligned_free(ref_re); vfft_proto_aligned_free(ref_im);
    }

    /* ---- Q2: DIT-FLAT vs DIF-FLAT inner c2c cost at K=256 ---- */
    {
        const size_t K = 256;
        double *mr = alloc_d((size_t)N*K), *mi = alloc_d((size_t)N*K);
        double *re = alloc_d((size_t)N*K), *im = alloc_d((size_t)N*K);
        srand(999);
        for (size_t i = 0; i < (size_t)N*K; i++) { mr[i] = (double)rand()/RAND_MAX*2-1; mi[i] = (double)rand()/RAND_MAX*2-1; }

        stride_plan_t *pdit = vfft_proto_plan_create_ex(N, K, factors, vflat, nf, 0, &reg);
        stride_plan_t *pdif = vfft_proto_plan_create_ex(N, K, factors, vflat, nf, 1, &reg);
        double tdit = time_c2c(pdit, mr, mi, re, im, K, N);
        double tdif = time_c2c(pdif, mr, mi, re, im, K, N);
        printf("\n[Q2 cost @K=256, incl 2x memcpy both] DIT-FLAT=%.1f ns  DIF-FLAT=%.1f ns  (DIF-DIT=%.1f ns)\n",
               tdit, tdif, tdif - tdit);
        printf("     -> if DIF-DIT is small/zero, natural-order recombine win is free.\n");

        stride_plan_destroy(pdit); stride_plan_destroy(pdif);
        vfft_proto_aligned_free(mr); vfft_proto_aligned_free(mi);
        vfft_proto_aligned_free(re); vfft_proto_aligned_free(im);
    }

    return 0;
}
