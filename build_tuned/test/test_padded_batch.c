/* test_padded_batch.c — Step A smoke test for the opt-in padded-batch allocator
 * (docs/roadmap/tail_handling/padding_design_decision.md, Phase 1 Step A),
 * post batch-API consolidation: the batch is BORN FROM THE CONFIG via
 * vfft_alloc_batch_for(cfg) and the planes come back in execute roles via
 * vfft_batch_planes (in-place c2c: sre/dre = re plane, sim/dim = im plane).
 *
 * Proves the allocator contract: Kp = roundup(K,VW), full N*Kp doubles
 * addressable, re+im ZEROED, pad columns zero, opaque stride reported, loud
 * NULL on misuse, and free is clean.
 *
 * Build: build_tuned/build.py --src test/test_padded_batch.c --vfft
 * Run  : from anywhere (no wisdom/data files needed).
 */
#include <stdio.h>
#include <stddef.h>
#include <string.h>
#include "vfft.h"

#define VW 4
static size_t roundup_vw(size_t k) { return (k + (VW - 1)) & ~(size_t)(VW - 1); }

static int fails = 0;
#define CHECK(cond, msg) do { if (!(cond)) { printf("  FAIL: %s\n", msg); fails++; } } while (0)

static void mkcfg(vfft_config_t *c, int N, size_t K)
{
    memset(c, 0, sizeof *c);
    c->transform = VFFT_C2C;
    c->placement = VFFT_INPLACE;
    c->rigor = VFFT_MEASURE;
    c->dims = 1;
    c->n[0] = N;
    c->howmany = K;
}

static void one_cell(int N, size_t K)
{
    size_t Kp = roundup_vw(K);
    vfft_config_t cfg;
    mkcfg(&cfg, N, K);
    vfft_batch b = vfft_alloc_batch_for(&cfg);
    CHECK(b != NULL, "alloc returned non-NULL");
    if (!b) return;

    double *re, *im, *dre, *dim;
    vfft_batch_planes(b, &re, &im, &dre, &dim);
    size_t st = vfft_batch_stride(b);
    CHECK(re != NULL && im != NULL, "re/im non-NULL");
    CHECK(dre == re && dim == im, "in-place roles: dst planes == src planes");
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
        int padnz = 0;
        for (int e = 0; e < N; e++)
            for (size_t p = K; p < Kp; p++)
                if (re[(size_t)e * Kp + p] != 0.0 && !((size_t)e * Kp + p == (size_t)N * Kp - 1)) padnz++;
        CHECK(padnz == 0, "interior pad columns zero");
    }

    printf("  N=%-5d K=%-3zu Kp=%-3zu stride=%-3zu  ok\n", N, K, Kp, st);
    vfft_free_batch(b);
}

int main(void)
{
    printf("# padded-batch allocator smoke test (Step A, consolidated API)\n");
    one_cell(256, 7);
    one_cell(256, 8);     /* already aligned: Kp==K, no pad */
    one_cell(512, 11);
    one_cell(1024, 15);
    one_cell(2048, 31);
    one_cell(64, 1);      /* degenerate K=1 -> Kp=4 */

    /* invalid args -> loud NULL, and free(NULL) is a no-op */
    {
        vfft_config_t c;
        mkcfg(&c, 0, 8);
        CHECK(vfft_alloc_batch_for(&c) == NULL, "N<1 -> NULL");
        mkcfg(&c, 256, 0);
        CHECK(vfft_alloc_batch_for(&c) == NULL, "K<1 -> NULL");
        mkcfg(&c, 256, 8);
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        CHECK(vfft_alloc_batch_for(&c) == NULL, "layout=INTERLEAVED -> NULL (split-only)");
        mkcfg(&c, 32, 8);
        c.dims = 2; c.n[1] = 32;
        CHECK(vfft_alloc_batch_for(&c) == NULL, "dims=2 -> NULL (1D only)");
        CHECK(vfft_alloc_batch_for(NULL) == NULL, "NULL config -> NULL");
    }
    vfft_free_batch(NULL);
    {
        double *sre = (double *)1, *sim = (double *)1, *dre = (double *)1, *dim = (double *)1;
        vfft_batch_planes(NULL, &sre, &sim, &dre, &dim);
        CHECK(sre == NULL && sim == NULL && dre == NULL && dim == NULL,
              "NULL-handle planes() nulls all out-params");
        CHECK(vfft_batch_stride(NULL) == 0, "NULL-handle stride safe");
    }

    printf(fails ? "\nRESULT: %d CHECK(s) FAILED\n" : "\nRESULT: all checks passed\n", fails);
    return fails ? 1 : 0;
}
