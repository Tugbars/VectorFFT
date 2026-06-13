/* bench_oop_vs_mkl.c — phase 5: the OOP engine vs MKL DFTI, both
 * out-of-place, split storage, column layout (element e at e*K + t),
 * single thread, same binary, round-robin, min-of-rounds.
 *
 * Per cell: plan via the full phase-4 pipeline (pair tuner -> hint ->
 * vfft_oop_plan_create_auto with wisdom fallback). MKL: DFTI_REAL_REAL,
 * DFTI_NOT_INPLACE, strides {0, K}, distance 1, K transforms.
 *
 * Verification: LEAF/BAILEY2 are natural order and gate elementwise vs
 * MKL at <=1e-9. MODEB output is scrambled order (digit-permuted), so its
 * elementwise gate vs MKL is not applicable; correctness is the bit-exact
 * equivalence to the in-place dataflow (test_oop_execute/test_oop_sweep)
 * and the ORDER CAVEAT is printed with the row. Apples note: MKL delivers
 * natural order on MODEB cells; a caller needing natural order from MODEB
 * would pay a reorder pass not measured here.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <x86intrin.h>
#include <mkl_dfti.h>
#include <mkl_service.h>
#include "../core/executor.h"
#include "../core/oop_auto.h"

static unsigned long long mn2(unsigned long long a,unsigned long long b){return a<b?a:b;}

static void run_cell(int N, size_t K,
                     const vfft_proto_wisdom_t *wis,
                     vfft_proto_registry_t *reg)
{
    /* phase-4 pipeline: tune pairs if multiple, then auto-create */
    vfft_oop_pair_hint_t h = {0, 0, 0, 0};
    int nh = 0, r1 = 0, r2 = 0;
    if (vfft_oop_tune_pairs(N, K, &r1, &r2, 0) >= 2 && r1 > 0)
    {
        h.N = N; h.K = K; h.R1 = r1; h.R2 = r2; nh = 1;
    }
    vfft_oop_plan_t *p = vfft_oop_plan_create_auto(N, K, wis, &h, nh, reg);
    if (!p)
    {
        printf("  N=%-5d K=%-4zu  no plan (skipped)\n", N, K);
        return;
    }
    size_t T = (size_t)N * K;
    double *sr = aligned_alloc(64, T * 8), *si = aligned_alloc(64, T * 8);
    double *dr = aligned_alloc(64, T * 8), *di = aligned_alloc(64, T * 8);
    double *mr_ = aligned_alloc(64, T * 8), *mi_ = aligned_alloc(64, T * 8);
    srand(53 + N);
    for (size_t i = 0; i < T; i++)
    {
        sr[i] = (double)rand() / RAND_MAX - 0.5;
        si[i] = (double)rand() / RAND_MAX - 0.5;
    }

    DFTI_DESCRIPTOR_HANDLE d = 0;
    MKL_LONG str[2] = {0, (MKL_LONG)K};
    if (DftiCreateDescriptor(&d, DFTI_DOUBLE, DFTI_COMPLEX, 1, (MKL_LONG)N)
        != DFTI_NO_ERROR)
    {
        printf("  N=%-5d K=%-4zu  MKL descriptor FAIL\n", N, K);
        return;
    }
    DftiSetValue(d, DFTI_COMPLEX_STORAGE, DFTI_REAL_REAL);
    DftiSetValue(d, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    DftiSetValue(d, DFTI_NUMBER_OF_TRANSFORMS, (MKL_LONG)K);
    DftiSetValue(d, DFTI_INPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_OUTPUT_DISTANCE, 1);
    DftiSetValue(d, DFTI_INPUT_STRIDES, str);
    DftiSetValue(d, DFTI_OUTPUT_STRIDES, str);
    if (DftiCommitDescriptor(d) != DFTI_NO_ERROR)
    {
        printf("  N=%-5d K=%-4zu  MKL commit FAIL\n", N, K);
        DftiFreeDescriptor(&d);
        return;
    }

    /* correctness */
    vfft_oop_execute_fwd(p, sr, si, dr, di);
    DftiComputeForward(d, sr, si, mr_, mi_);
    const char *order;
    char gate[32];
    if (p->kind != VFFT_OOP_KIND_MODEB)
    {
        double me = 0, mm = 0;
        for (size_t i = 0; i < T; i++)
        {
            double a = dr[i] - mr_[i], b = di[i] - mi_[i];
            double e2 = sqrt(a * a + b * b), m = hypot(mr_[i], mi_[i]);
            if (e2 > me) me = e2;
            if (m > mm) mm = m;
        }
        double r = mm > 0 ? me / mm : 0;
        snprintf(gate, sizeof gate, "%.0e %s", r, r < 1e-9 ? "OK" : "FAIL");
        order = "natural";
        if (r >= 1e-9)
        {
            printf("  N=%-5d K=%-4zu  GATE FAIL %s\n", N, K, gate);
            goto done;
        }
    }
    else
    {
        snprintf(gate, sizeof gate, "bit-exact*");
        order = "scrambled";
    }

    {
        enum { ROUNDS = 20 };
        unsigned long long mv = ~0ULL, mk = ~0ULL, c;
        for (int w = 0; w < 3; w++)
        {
            vfft_oop_execute_fwd(p, sr, si, dr, di);
            DftiComputeForward(d, sr, si, mr_, mi_);
        }
        for (int r = 0; r < ROUNDS; r++)
        {
            c = __rdtsc();
            vfft_oop_execute_fwd(p, sr, si, dr, di);
            mv = mn2(mv, __rdtsc() - c);
            c = __rdtsc();
            DftiComputeForward(d, sr, si, mr_, mi_);
            mk = mn2(mk, __rdtsc() - c);
        }
        const char *kn = p->kind == VFFT_OOP_KIND_LEAF ? "LEAF" :
                         p->kind == VFFT_OOP_KIND_BAILEY2 ? "BAILEY2" : "MODEB";
        char pair[16] = "";
        if (p->kind == VFFT_OOP_KIND_BAILEY2)
            snprintf(pair, sizeof pair, " %dx%d", p->R1, p->R2);
        printf("  N=%-5d K=%-4zu %-7s%-7s gate %-10s order %-9s | "
               "vfft %9llu | mkl %9llu | speed vs MKL %.3f\n",
               N, K, kn, pair, gate, order, mv, mk, (double)mk / mv);
    }
done:
    DftiFreeDescriptor(&d);
    free(sr); free(si); free(dr); free(di); free(mr_); free(mi_);
    vfft_oop_plan_destroy(p);
}

int main(void)
{
    mkl_set_num_threads(1);
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    vfft_proto_wisdom_t wis;
    if (vfft_proto_wisdom_load(&wis, "/tmp/wis/wisdom_v198.txt") != 0)
    {
        printf("wisdom load FAIL\n");
        return 1;
    }
    printf("== OOP engine vs MKL DFTI NOT_INPLACE, split, 1 thread, "
           "higher is faster ==\n");
    run_cell(64, 512, &wis, &reg);
    run_cell(128, 512, &wis, &reg);
    run_cell(169, 512, &wis, &reg);
    run_cell(512, 120, &wis, &reg);
    run_cell(1024, 120, &wis, &reg);
    run_cell(1024, 256, &wis, &reg);
    run_cell(4096, 256, &wis, &reg);
    run_cell(2310, 32, &wis, &reg);
    printf("(* MODEB correctness = bit-exact vs in-place dataflow per "
           "test_oop_sweep; scrambled order, reorder to natural not "
           "measured)\n");
    vfft_proto_wisdom_free(&wis);
    return 0;
}
