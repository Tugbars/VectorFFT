/* zturn_inplace_probe.c — P0a: is the cascade REALLY alias-safe in-place?
 *
 * zturn.h's header CLAIMS "In-place (zin == zout) OK both directions" (fwd
 * zout is written only by the terminator, which reads only the plane; bwd zin
 * is read only by the first stage). The in-place front-door work (Phase A of
 * docs/roadmap/cascade_natural_inplace_plan.md) ships on that property, so it
 * gets verified EMPIRICALLY first — nothing ships on a comment.
 *
 * PLAN-LEVEL on purpose: the front door refuses VFFT_INPLACE today (that gate
 * is what Phase A relaxes), so a pre-A1 ground-truth probe cannot go through
 * it. The Phase-A2 gate re-proves everything through vfft_create once the
 * door exists. This is a correctness probe, not a benchmark — no timing.
 *
 * MATRIX: banked-shape chains (2048..32768, from the 2026-08-02 calibration)
 *   x engine {zturn, legacy zsplit}
 *   x arms {untiled, tiled@banked-width, tiled+tfuse}   (tile arms zturn-only)
 *   x direction {fwd, bwd}
 * Each cell: OOP reference run, then an ALIASED run (in==out), memcmp over
 * all 2N doubles. Anything but EXACT is a FAIL; a refused tile arm is
 * REFUSED (reported, not counted as pass or fail).
 *
 * Build: python build_tuned/build.py --src build_tuned/benches/zturn_inplace_probe.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "zturn.h"

static double *az(size_t n)                       /* n complex, 64B aligned */
{
#ifdef _WIN32
    return (double *)_aligned_malloc(2 * n * sizeof(double), 64);
#else
    void *p = NULL;
    if (posix_memalign(&p, 64, 2 * n * sizeof(double))) p = NULL;
    return (double *)p;
#endif
}
static void fz(double *p)
{
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}

typedef struct { int N, nf, chain[8]; long tw; int zsnf, zschain[8]; } cell_t;

/* Chains + widths exactly as banked by the 2026-08-02 calibration sweep
 * (wisfull). 2048 banked UNTILED — its tile arm uses the section width so the
 * tiled machinery is still exercised at that N. */
/* zschain = a last==8 sibling for the LEGACY engine (its fence refuses
 * last==4), because zsplit is the front door's FALLBACK route — if the door
 * opens for in-place, the fallback must be proven alias-safe too. */
static const cell_t CELLS[] = {
    { 2048,  5, {4,8,4,4,4},       512,  5, {4,4,4,4,8}   },
    { 4096,  6, {4,4,4,4,4,4},     1024, 5, {4,4,4,8,8}   },
    { 8192,  6, {4,8,4,4,4,4},     2048, 6, {4,4,4,4,4,8} },
    { 16384, 7, {4,4,4,4,4,4,4},   2048, 6, {4,8,4,4,4,8} },
    { 32768, 7, {4,8,4,4,4,4,4},   1024, 6, {4,8,4,4,8,8} },
};

static int g_pass = 0, g_fail = 0, g_ref = 0;

static const char *check(const double *want, const double *got, int N)
{
    if (memcmp(want, got, 2 * (size_t)N * sizeof(double)) == 0)
    { g_pass++; return "EXACT"; }
    g_fail++;
    /* first differing index, for the record */
    static char buf[48];
    for (int i = 0; i < 2 * N; i++)
        if (want[i] != got[i])
        { snprintf(buf, sizeof buf, "*** DIFFER @%d ***", i); return buf; }
    snprintf(buf, sizeof buf, "*** DIFFER ***");
    return buf;
}

int main(void)
{
    printf("\n=== P0a: cascade in-place ground truth (plan level, memcmp) ===\n");
    printf("%-7s %-16s %-9s %-22s %-22s\n",
           "N", "chain", "engine", "fwd (aliased vs OOP)", "bwd (aliased vs OOP)");

    for (size_t ci = 0; ci < sizeof CELLS / sizeof CELLS[0]; ci++)
    {
        const cell_t *c = &CELLS[ci];
        const int N = c->N;
        char cs[32] = {0};
        for (int i = 0, o = 0; i < c->nf; i++)
            o += snprintf(cs + o, sizeof cs - (size_t)o, i ? ".%d" : "%d",
                          c->chain[i]);

        double *x  = az((size_t)N);   /* pristine input                    */
        double *r  = az((size_t)N);   /* OOP reference output              */
        double *a  = az((size_t)N);   /* aliased buffer                    */
        srand(11 + N);
        for (int i = 0; i < 2 * N; i++) x[i] = (double)rand() / RAND_MAX - 0.5;

        /* ---- ZTURN: untiled, tiled, tiled+tfuse ---- */
        static const char *ARM[3] = { "untiled", "tiled", "tfuse" };
        for (int arm = 0; arm < 3; arm++)
        {
            vfft_zturn2_plan_t *p =
                vfft_zturn2_create_chain(N, (int *)c->chain, c->nf);
            if (!p) { printf("%-7d %-16s zturn/%-7s create REFUSED\n",
                             N, cs, ARM[arm]); continue; }
            if (arm > 0 &&
                !vfft_zturn2_set_tile_w(p, 1, c->tw, arm == 2 ? 1 : 0, 0))
            {
                printf("%-7d %-16s zturn/%-7s tile w=%ld%s REFUSED by fence\n",
                       N, cs, ARM[arm], c->tw, arm == 2 ? "+tfuse" : "");
                g_ref++;
                vfft_zturn2_destroy(p);
                continue;
            }
            /* fwd: reference then aliased */
            vfft_zturn2_execute_fwd(p, x, r);
            memcpy(a, x, 2 * (size_t)N * sizeof(double));
            vfft_zturn2_execute_fwd(p, a, a);
            const char *vf = check(r, a, N);
            /* bwd: feed the OOP spectrum; reference then aliased */
            double *rb = az((size_t)N);
            vfft_zturn2_execute_bwd(p, r, rb);
            memcpy(a, r, 2 * (size_t)N * sizeof(double));
            vfft_zturn2_execute_bwd(p, a, a);
            const char *vb = check(rb, a, N);
            char lab[24];
            snprintf(lab, sizeof lab, "zturn/%s", ARM[arm]);
            printf("%-7d %-16s %-9s %-22s %-22s\n", N, cs,
                   arm ? (arm == 2 ? "z/tfuse" : "z/tiled") : "zturn",
                   vf, vb);
            (void)lab;
            fz(rb);
            vfft_zturn2_destroy(p);
        }

        /* ---- legacy zsplit, untiled (it has no tiled path) ---- */
        {
            char zcs[32] = {0};
            for (int i = 0, o = 0; i < c->zsnf; i++)
                o += snprintf(zcs + o, sizeof zcs - (size_t)o, i ? ".%d" : "%d",
                              c->zschain[i]);
            vfft_zsplit_plan_t *p =
                vfft_zsplit_create(N, (int *)c->zschain, c->zsnf);
            if (!p)
                printf("%-7d %-16s %-9s (chain outside the legacy fence — "
                       "SKIP)\n", N, zcs, "zsplit");
            else
            {
                vfft_zsplit_execute_fwd(p, x, r);
                memcpy(a, x, 2 * (size_t)N * sizeof(double));
                vfft_zsplit_execute_fwd(p, a, a);
                const char *vf = check(r, a, N);
                double *rb = az((size_t)N);
                vfft_zsplit_execute_bwd(p, r, rb);
                memcpy(a, r, 2 * (size_t)N * sizeof(double));
                vfft_zsplit_execute_bwd(p, a, a);
                const char *vb = check(rb, a, N);
                printf("%-7d %-16s %-9s %-22s %-22s\n", N, zcs, "zsplit", vf, vb);
                fz(rb);
                vfft_zsplit_destroy(p);
            }
        }

        fz(x); fz(r); fz(a);
    }

    printf("\n=== %d EXACT, %d FAIL, %d tile-arm refusals ===\n",
           g_pass, g_fail, g_ref);
    printf("EXACT everywhere = the header's alias-safety claim holds and "
           "Phase A may ship on it.\nAny FAIL = the claim is false for that "
           "arm; the front door must NOT open for it.\n");
    return g_fail ? 1 : 0;
}
