/* test_padded_batch.c — Step A smoke test for the opt-in padded-batch allocator
 * (docs/roadmap/tail_handling/padding_design_decision.md, Phase 1 Step A),
 * post plan-owns-its-buffers consolidation: the planes are BORN WITH THE PLAN
 * via config.owned_buffers = 1 + vfft_create(cfg) and come back in execute
 * roles via vfft_plan_planes (in-place c2c: sre/dre = re plane, sim/dim = im
 * plane); vfft_destroy frees them.
 *
 * Proves the allocator contract: Kp = roundup(K,VW), full N*Kp doubles
 * addressable, re+im ZEROED, pad columns zero, opaque stride reported, loud
 * NULL on misuse, and destroy is clean.
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
    c->owned_buffers = 1; /* the plan allocates + owns + frees the planes */
}

static void one_cell(int N, size_t K)
{
    size_t Kp_pad = roundup_vw(K); /* the padded width, IF padding wins the race */
    vfft_config_t cfg;
    mkcfg(&cfg, N, K);
    vfft_plan p = vfft_create(&cfg); /* create FIRST: the plan owns the planes */
    CHECK(p != NULL, "create(owned_buffers) returned non-NULL");
    if (!p) return;

    double *re, *im, *dre, *dim;
    vfft_plan_planes(p, &re, &im, &dre, &dim);
    CHECK(re != NULL && im != NULL, "re/im non-NULL");
    CHECK(dre == re && dim == im, "in-place roles: dst planes == src planes");
    /* 2026-07-28: the allocator DECIDES padded-vs-tight from a measured verdict, so the
     * stride is whatever the plan says — K (tight) or the padded width. Every span and
     * index below MUST come from it; assuming roundup() here was an out-of-bounds write
     * on a tight buffer (the STATUS_HEAP_CORRUPTION shape that hit test_padded_dispatch). */
    size_t Kp = vfft_plan_stride(p);
    CHECK(Kp == K || Kp == Kp_pad, "stride is tight K or padded roundup(K,VW)");

    /* every one of the N*Kp doubles must read back as zero (allocator zeroed both) */
    int nonzero = 0;
    for (size_t i = 0; i < (size_t)N * Kp; i++)
        if (re[i] != 0.0 || im[i] != 0.0) nonzero++;
    CHECK(nonzero == 0, "re+im fully zeroed on alloc");

    /* full N*Kp span is writable (no OOB): stamp the last physical element */
    re[(size_t)N * Kp - 1] = 1.0;
    im[(size_t)N * Kp - 1] = 2.0;
    CHECK(re[(size_t)N * Kp - 1] == 1.0 && im[(size_t)N * Kp - 1] == 2.0, "last element R/W ok");

    /* pad columns (b in [K,Kp)) sit between real lanes at stride Kp and start zeroed.
     * A TIGHT verdict has none, and this block correctly does not run. */
    if (Kp > K) {
        int padnz = 0;
        for (int e = 0; e < N; e++)
            for (size_t pc = K; pc < Kp; pc++) /* not `p`: that is the plan now */
                if (re[(size_t)e * Kp + pc] != 0.0 && !((size_t)e * Kp + pc == (size_t)N * Kp - 1)) padnz++;
        CHECK(padnz == 0, "interior pad columns zero");
    }

    printf("  N=%-5d K=%-3zu stride=%-3zu (%s)  ok\n", N, K, Kp,
           Kp == K ? "TIGHT" : "PADDED");
    vfft_destroy(p); /* frees the planes it owns */
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

    /* invalid args -> loud NULL, and destroy(NULL) is a no-op.
     * mkcfg() sets owned_buffers = 1, so each of these exercises the owned-plane
     * allocator's refusals through vfft_create. */
    {
        vfft_config_t c;
        mkcfg(&c, 0, 8);
        CHECK(vfft_create(&c) == NULL, "N<1 -> NULL");
        mkcfg(&c, 256, 0);
        CHECK(vfft_create(&c) == NULL, "K<1 -> NULL");
        mkcfg(&c, 256, 8);
        c.layout = VFFT_LAYOUT_INTERLEAVED;
        CHECK(vfft_create(&c) == NULL, "layout=INTERLEAVED -> NULL (split-only)");
        mkcfg(&c, 32, 8);
        c.dims = 2; c.n[1] = 32;
        CHECK(vfft_create(&c) == NULL, "dims=2 -> NULL (1D only)");
        CHECK(vfft_create(NULL) == NULL, "NULL config -> NULL");
    }
    vfft_destroy(NULL);
    {
        double *sre = (double *)1, *sim = (double *)1, *dre = (double *)1, *dim = (double *)1;
        vfft_plan_planes(NULL, &sre, &sim, &dre, &dim);
        CHECK(sre == NULL && sim == NULL && dre == NULL && dim == NULL,
              "NULL-handle planes() nulls all out-params");
        CHECK(vfft_plan_stride(NULL) == 0, "NULL-handle stride safe");
    }

    printf(fails ? "\nRESULT: %d CHECK(s) FAILED\n" : "\nRESULT: all checks passed\n", fails);
    return fails ? 1 : 0;
}
