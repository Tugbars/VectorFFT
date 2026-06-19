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
#include "../core/env.h"        /* stride_pin_thread */
#include "../core/oop_auto.h"

/* mingw lacks C11 aligned_alloc; _aligned_malloc needs _aligned_free (not free). */
#if defined(_WIN32)
#include <malloc.h>
#define AALLOC(n) _aligned_malloc((n), 64)
#define AFREE(p)  _aligned_free(p)
#else
#include <stdlib.h>
#define AALLOC(n) aligned_alloc(64, (n))
#define AFREE(p)  free(p)
#endif

static unsigned long long mn2(unsigned long long a,unsigned long long b){return a<b?a:b;}

static void run_cell(int N, size_t K,
                     const vfft_proto_wisdom_t *wis,
                     vfft_proto_registry_t *reg)
{
    /* rule-spine auto-create (tuner skipped for this baseline — it adds a
     * residual 6-8% on BAILEY2 cells; the rule's balanced-first pick is the
     * default). vfft_oop_tune_pairs uses aligned_alloc, dropped on mingw. */
    vfft_oop_plan_t *p = vfft_oop_plan_create_auto(N, K, wis, NULL, 0, reg);
    if (!p)
    {
        printf("  N=%-5d K=%-4zu  no plan (skipped)\n", N, K);
        return;
    }
    size_t T = (size_t)N * K;
    double *sr = AALLOC(T * 8), *si = AALLOC(T * 8);
    double *dr = AALLOC(T * 8), *di = AALLOC(T * 8);
    double *mr_ = AALLOC(T * 8), *mi_ = AALLOC(T * 8);
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
    AFREE(sr); AFREE(si); AFREE(dr); AFREE(di); AFREE(mr_); AFREE(mi_);
    vfft_oop_plan_destroy(p);
}

int main(void)
{
    mkl_set_num_threads(1);
    if (stride_pin_thread(2) != 0) fprintf(stderr, "warn: pin cpu2 failed\n");
    vfft_proto_registry_t reg;
    vfft_proto_registry_init(&reg);
    /* c2c wisdom is only needed for the MODEB fallback; LEAF/BAILEY2 run without
     * it. Try the build_tuned-relative paths, else continue (MODEB cells skip). */
    vfft_proto_wisdom_t wis;
    int have_wis =
        (vfft_proto_wisdom_load(&wis, "../src/dag-fft-compiler/generator/generated/spike_wisdom.txt") == 0) ||
        (vfft_proto_wisdom_load(&wis, "../../src/dag-fft-compiler/generator/generated/spike_wisdom.txt") == 0) ||
        (vfft_proto_wisdom_load(&wis, "../generator/generated/spike_wisdom.txt") == 0);
    if (!have_wis)
        printf("# wisdom load FAIL — MODEB cells will skip; LEAF/BAILEY2 still run\n");
    const vfft_proto_wisdom_t *wisp = have_wis ? &wis : NULL;
    printf("== OOP engine vs MKL DFTI NOT_INPLACE, split, 1 thread, "
           "higher is faster ==\n");
    run_cell(64, 512, wisp, &reg);
    run_cell(128, 512, wisp, &reg);
    run_cell(169, 512, wisp, &reg);
    run_cell(512, 120, wisp, &reg);
    run_cell(1024, 120, wisp, &reg);
    run_cell(1024, 256, wisp, &reg);
    run_cell(4096, 256, wisp, &reg);
    run_cell(2310, 32, wisp, &reg);
    printf("(* MODEB correctness = bit-exact vs in-place dataflow per "
           "test_oop_sweep; scrambled order, reorder to natural not "
           "measured)\n");
    if (have_wis) vfft_proto_wisdom_free(&wis);
    return 0;
}
