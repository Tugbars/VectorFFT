/* test_padded_batch.c — Step A smoke test for the opt-in padded-batch allocator
 * (docs/roadmap/tail_handling/padding_design_decision.md, Phase 1 Step A).
 *
 * The handle is INERT today (no execute path yet); this just proves the allocator
 * contract: Kp = roundup(K,VW), full N*Kp doubles addressable, re+im ZEROED, pad
 * columns zero, opaque stride reported, and free is clean.
 *
 * Build: build_tuned/build.py --src test/test_padded_batch.c --vfft
 * Run  : from anywhere (no wisdom/data files needed).
 */
#include <stdio.h>
#include <stddef.h>
#include "vfft.h"

#define VW 4
static size_t roundup_vw(size_t k) { return (k + (VW - 1)) & ~(size_t)(VW - 1); }

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } } while (0)

static void one_cell(int N, size_t K)
{
    size_t Kp = roundup_vw(K);
    vfft_batch b = vfft_alloc_batch(N, K);
    CHECK(b != NULL, "alloc returned non-NULL");
    if (!b) return;

    double *re = vfft_batch_re(b), *im = vfft_batch_im(b);
    size_t st = vfft_batch_stride(b);
    CHECK(re != NULL && im != NULL, "re/im non-NULL");
    CHECK(st == Kp, "stride == roundup(K,VW)");

    /* every one of the N*Kp doubles must read back as zero (allocator zeroed both) */
    int nonzero = 0;
    for (size_t i = 0; i < (size_t)N * Kp; i++)
        if (re[i] != 0.0 || im[i] != 0.0) nonzero++;
    CHECK(nonzero == 0, "re+im fully zeroed on alloc");

    /* full N*Kp span is writable (no OOB): stamp the last physical element */
    re[(size_t)N * Kp - 1] = 1.0;
    im[(size_t)N * Kp - 1] = 2.0;
    CHECK(re[(size_t)N * Kp - 1] == 1.0 && im[(size_t)N * Kp - 1] == 2.0, "last element R/W ok");

    /* pad columns (b in [K,Kp)) sit between real lanes at stride Kp and start zeroed */
    if (Kp > K) {
        double *r2 = vfft_batch_re(b);
        int padnz = 0;
        for (int e = 0; e < N; e++)
            for (size_t p = K; p < Kp; p++)
                if (r2[(size_t)e * Kp + p] != 0.0 && !((size_t)e * Kp + p == (size_t)N * Kp - 1)) padnz++;
        CHECK(padnz == 0, "interior pad columns zero");
    }

    printf("  N=%-5d K=%-3zu Kp=%-3zu stride=%-3zu  ok\n", N, K, Kp, st);
    vfft_free_batch(b);
}

int main(void)
{
    printf("# padded-batch allocator smoke test (Step A)\n");
    one_cell(256, 7);
    one_cell(256, 8);     /* already aligned: Kp==K, no pad */
    one_cell(512, 11);
    one_cell(1024, 15);
    one_cell(2048, 31);
    one_cell(64, 1);      /* degenerate K=1 -> Kp=4 */

    /* invalid args -> NULL, and free(NULL) is a no-op */
    CHECK(vfft_alloc_batch(0, 8) == NULL, "N<1 -> NULL");
    CHECK(vfft_alloc_batch(256, 0) == NULL, "K<1 -> NULL");
    vfft_free_batch(NULL);
    CHECK(vfft_batch_re(NULL) == NULL && vfft_batch_stride(NULL) == 0, "NULL-handle accessors safe");

    printf(fails ? "\nRESULT: %d CHECK(s) FAILED\n" : "\nRESULT: all checks passed\n", fails);
    return fails ? 1 : 0;
}
