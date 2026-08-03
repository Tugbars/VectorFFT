/* mt_c2c_gate.c — consolidated MT correctness gate for in-place c2c.
 *
 * Re-validates the two 2026-07-06 MT bug classes on the CURRENT tree (the
 * original scratch-era gates — natmt_test/mt_slab_diag2/nat_mtfirst/… —
 * are gone; this replaces them as a permanent front-door gate):
 *
 *   BUG 1 (16·8): radix-8 LOG3 last stage ignored `me` — the 128/32 cell
 *          gave a deterministic 40.0 error at T=8. Codegen fix =
 *          block-broadcast twiddles (VFFT_PROTO_TW_BLOCK_K).
 *   BUG 2: DIF chains wrong on a PARTIAL batch with ASYMMETRIC input —
 *          a symmetric/periodic probe MASKS it. Hence: well-mixed random
 *          input here, never index hashes.
 *   plus the natural MT-FIRST trap: creating the natural plan MT-first
 *          (no ST warmup) raced a lazily-built shared state. Hence: the
 *          MT plan here is created FIRST, fresh, before any ST create.
 *
 * ARMS per cell × order ∈ {DEFAULT, NATURAL}:
 *   1. MT (T=8) create FIRST → execute fwd on random lanes;
 *   2. ST (T=1) same input → fwd;
 *   3. MT vs ST memcmp-EXACT (K-split must not change per-lane math);
 *   4. ST vs naive per-lane DFT (tolerance) — the independent reference
 *      (natural: naive IN ORDER; scrambled: compared via ST-vs-MT only,
 *      the reference arm uses roundtrip/N instead);
 *   5. MT roundtrip: bwd(fwd(x)) == N·x (legal here: same-plan matched
 *      permutation — this is NOT an ordering gate, orders are gated by
 *      arm 4 on the natural side).
 *
 * NOTE: vfft.c's `_c2c_mt_safe` slab-replay net means a codegen REGRESSION
 * would still produce correct output (whole-batch fallback) — this gate
 * proves SHIPPING CORRECTNESS, which is the contract; split-vs-fallback is
 * not externally observable and not gated here.
 *
 * Run:   mt_c2c_gate.exe --wisdir <scratch dir>
 * Build: python build.py --src benches/mt_c2c_gate.c --vfft --compile
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static double *ad(size_t n)
{
#ifdef _WIN32
    return (double *)_aligned_malloc(n * sizeof(double), 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, n * sizeof(double))) p = NULL;
    return (double *)p;
#endif
}
static void fd(double *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

/* xorshift64 — well-mixed, non-periodic (the BUG-2 lesson: rand() is OK
 * but an index hash is NOT; use a real PRNG and never a symmetric probe) */
static unsigned long long g_s = 0x9E3779B97F4A7C15ull;
static double xr(void)
{
    g_s ^= g_s << 13;
    g_s ^= g_s >> 7;
    g_s ^= g_s << 17;
    return (double)(g_s >> 11) / (double)(1ull << 53) - 0.5;
}

static vfft_plan mk(vfft_wisdom *W, int N, int K, int natural, int nthreads)
{
    vfft_config_t cfg;
    memset(&cfg, 0, sizeof cfg);
    cfg.transform = VFFT_C2C;
    cfg.placement = VFFT_INPLACE;
    cfg.rigor = VFFT_MEASURE;
    cfg.dims = 1;
    cfg.n[0] = N;
    cfg.howmany = (size_t)K;
    cfg.order = natural ? VFFT_ORDER_NATURAL : VFFT_ORDER_DEFAULT;
    cfg.layout = VFFT_LAYOUT_SPLIT;
    cfg.nthreads = nthreads;
    cfg.wisdom = W;
    return vfft_create(&cfg);
}

/* naive DFT of lane k (K-strided split storage), natural order */
static void naive_lane(const double *re, const double *im, int N, int K,
                       int lane, double *Xr, double *Xi)
{
    for (int k = 0; k < N; k++)
    {
        double sr = 0, si = 0;
        for (int j = 0; j < N; j++)
        {
            const double a = -2.0 * M_PI * (double)((long)j * k % N) / N;
            const double c = cos(a), s = sin(a);
            const double x = re[(size_t)j * K + lane],
                         y = im[(size_t)j * K + lane];
            sr += x * c - y * s;
            si += x * s + y * c;
        }
        Xr[k] = sr;
        Xi[k] = si;
    }
}

typedef struct { int N, K; const char *why; } cell_t;
static const cell_t CELLS[] = {
    { 128,  32,  "16*8 — BUG 1's cell (radix-8 LOG3 last stage)" },
    { 256,  32,  "4*8*8 control (passed originally)" },
    { 512,  32,  "4*4*32 — DIF class (BUG 2's cell)" },
    { 1024, 32,  "deep chain control" },
    { 128,  256, "16*8 at 8-slab K (full T=8 split)" },
    { 512,  256, "DIF class at 8-slab K" },
};

int main(int argc, char **argv)
{
    const char *wisdir = NULL;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--wisdir") && i + 1 < argc) wisdir = argv[++i];
    if (!wisdir) { printf("usage: %s --wisdir <SCRATCH dir>\n", argv[0]); return 2; }
    vfft_wisdom *W = vfft_wisdom_load(wisdir);
    if (!W) { printf("vfft_wisdom_load FAILED\n"); return 2; }

    printf("\n=== MT in-place c2c gate (T=8 vs ST vs naive; MT-first; "
           "asymmetric input) ===\n");
    printf("%-10s %-8s | %-6s %-10s %-10s | note\n",
           "N/K", "order", "MT==ST", "ST vs ref", "MT rt");
    int fails = 0;

    for (size_t ci = 0; ci < sizeof CELLS / sizeof CELLS[0]; ci++)
        for (int nat = 0; nat <= 1; nat++)
        {
            const int N = CELLS[ci].N, K = CELLS[ci].K;
            const size_t tot = (size_t)N * K;
            char nk[16];
            snprintf(nk, sizeof nk, "%d/%d", N, K);

            /* MT plan FIRST — the nat_mtfirst lesson (no ST warmup). */
            vfft_plan hm = mk(W, N, K, nat, 8);
            vfft_plan hs = mk(W, N, K, nat, 1);
            if (!hm || !hs)
            {
                printf("%-10s %-8s create FAILED\n", nk, nat ? "NATURAL" : "default");
                fails++;
                if (hm) vfft_destroy(hm);
                if (hs) vfft_destroy(hs);
                continue;
            }

            g_s = 0x9E3779B97F4A7C15ull + (unsigned long long)(N * 131 + K);
            double *re0 = ad(tot), *im0 = ad(tot);
            for (size_t i = 0; i < tot; i++) { re0[i] = xr(); im0[i] = xr(); }

            double *mr = ad(tot), *mi = ad(tot), *sr = ad(tot), *si = ad(tot);

            /* arm 1+2: fwd on both plans, same input */
            memcpy(mr, re0, tot * sizeof(double));
            memcpy(mi, im0, tot * sizeof(double));
            vfft_execute(hm, VFFT_FORWARD, mr, mi, mr, mi);
            memcpy(sr, re0, tot * sizeof(double));
            memcpy(si, im0, tot * sizeof(double));
            vfft_execute(hs, VFFT_FORWARD, sr, si, sr, si);

            /* arm 3: MT == ST, bitwise */
            const int eq = memcmp(mr, sr, tot * sizeof(double)) == 0 &&
                           memcmp(mi, si, tot * sizeof(double)) == 0;

            /* arm 4: ST vs independent reference.
             * natural: naive DFT IN ORDER on 3 spread lanes.
             * default (scrambled/unspecified order): roundtrip/N vs input
             * on the ST plan (matched-permutation; ordering not asserted). */
            double refe = 0.0;
            if (nat)
            {
                double *Xr = ad((size_t)N), *Xi = ad((size_t)N);
                const int lanes[3] = { 0, K / 2, K - 1 };
                for (int li = 0; li < 3; li++)
                {
                    naive_lane(re0, im0, N, K, lanes[li], Xr, Xi);
                    double m = 0, e = 0;
                    for (int k = 0; k < N; k++)
                    {
                        const double ar = sr[(size_t)k * K + lanes[li]],
                                     ai = si[(size_t)k * K + lanes[li]];
                        if (fabs(Xr[k]) > m) m = fabs(Xr[k]);
                        if (fabs(Xi[k]) > m) m = fabs(Xi[k]);
                        if (fabs(ar - Xr[k]) > e) e = fabs(ar - Xr[k]);
                        if (fabs(ai - Xi[k]) > e) e = fabs(ai - Xi[k]);
                    }
                    if (e / m > refe) refe = e / m;
                }
                fd(Xr);
                fd(Xi);
            }
            else
            {
                double *rr = ad(tot), *ri = ad(tot);
                memcpy(rr, sr, tot * sizeof(double));
                memcpy(ri, si, tot * sizeof(double));
                vfft_execute(hs, VFFT_BACKWARD, rr, ri, rr, ri);
                double m = 0, e = 0;
                const double inv = 1.0 / N;
                for (size_t i = 0; i < tot; i++)
                {
                    if (fabs(re0[i]) > m) m = fabs(re0[i]);
                    if (fabs(rr[i] * inv - re0[i]) > e) e = fabs(rr[i] * inv - re0[i]);
                    if (fabs(ri[i] * inv - im0[i]) > e) e = fabs(ri[i] * inv - im0[i]);
                }
                refe = e / m;
                fd(rr);
                fd(ri);
            }

            /* arm 5: MT roundtrip on the MT plan (bwd also K-split) */
            vfft_execute(hm, VFFT_BACKWARD, mr, mi, mr, mi);
            double rte = 0.0;
            {
                double m = 0, e = 0;
                const double inv = 1.0 / N;
                for (size_t i = 0; i < tot; i++)
                {
                    if (fabs(re0[i]) > m) m = fabs(re0[i]);
                    if (fabs(mr[i] * inv - re0[i]) > e) e = fabs(mr[i] * inv - re0[i]);
                    if (fabs(mi[i] * inv - im0[i]) > e) e = fabs(mi[i] * inv - im0[i]);
                }
                rte = e / m;
            }

            const int ok = eq && refe < 1e-9 && rte < 1e-9;
            if (!ok) fails++;
            printf("%-10s %-8s | %-6s %.2e   %.2e | %s%s\n",
                   nk, nat ? "NATURAL" : "default",
                   eq ? "EXACT" : "DIFF!", refe, rte,
                   CELLS[ci].why, ok ? "" : "   *** FAIL ***");

            fd(re0); fd(im0); fd(mr); fd(mi); fd(sr); fd(si);
            vfft_destroy(hm);
            vfft_destroy(hs);
        }

    vfft_wisdom_free(W);
    printf("\n=== %s ===\n", fails ? "*** FAIL ***" : "ALL PASS");
    return fails ? 1 : 0;
}
