/* oop_auto.h — phase 4 of docs section 16: wisdom-backed auto plan
 * creation and the divisor-pair tuner.
 *
 * vfft_oop_plan_create_auto(N, K, wis, hints, nhints, reg):
 *   1. tuned-pair hint for (N, K) -> pair-explicit BAILEY2 (the tuner's
 *      output overrides the static preference; section 14 measured the
 *      pair-order residue at 6-8 percent).
 *   2. rule spine (oop_plan.h): LEAF, then preferred unmasked BAILEY2.
 *   3. wisdom lookup for (N, K) -> MODEB built from the entry's factors
 *      and variant codes. Plans are built DIT regardless of the entry's
 *      use_dif_forward (OOP requires DIT; fidelity caveat: DIF-preferring
 *      cells run their DIT variant here).
 *
 * vfft_oop_tune_pairs(N, K, &bestR1, &bestR2, verbose):
 *   Enumerates unmasked divisor pairs, times each finished plan
 *   same-binary round-robin (min-of-rounds), returns the count of
 *   candidates and the winner. This is the entire searched residue for
 *   BAILEY2 cells; everything else is rule.
 */
#ifndef VFFT_OOP_AUTO_H
#define VFFT_OOP_AUTO_H

#include <x86intrin.h>
#include "oop_plan.h"
#include "wisdom_reader.h"

/* Portable 64B-aligned alloc/free for the pair tuner. mingw lacks C11
 * aligned_alloc and pairs _aligned_malloc with _aligned_free (not free). */
#if defined(_WIN32)
#include <malloc.h>
#define VFFT_OOP_AALLOC(n) _aligned_malloc((n), 64)
#define VFFT_OOP_AFREE(p)  _aligned_free(p)
#else
#include <stdlib.h>
#define VFFT_OOP_AALLOC(n) aligned_alloc(64, (n))
#define VFFT_OOP_AFREE(p)  free(p)
#endif

typedef struct
{
    int N;
    size_t K;
    int R1, R2;
} vfft_oop_pair_hint_t;

static inline vfft_oop_plan_t *vfft_oop_plan_create_auto(
    int N, size_t K,
    const vfft_proto_wisdom_t *wis,
    const vfft_oop_pair_hint_t *hints, int nhints,
    const vfft_proto_registry_t *reg)
{
    if (K == 0 || (K % 8u) != 0)
        return NULL;

    for (int i = 0; i < nhints; i++)
        if (hints[i].N == N && hints[i].K == K)
        {
            vfft_oop_plan_t *p =
                vfft_oop_plan_create_pair(N, K, hints[i].R1, hints[i].R2);
            if (p)
                return p;
            break; /* stale hint: fall through to rules */
        }

    vfft_oop_plan_t *p = vfft_oop_plan_create(N, K, NULL, 0, reg);
    if (p)
        return p;

    if (wis && reg)
    {
        const vfft_proto_wisdom_entry_t *e =
            vfft_proto_wisdom_lookup(wis, N, K);
        if (e)
        {
            p = (vfft_oop_plan_t *)calloc(1, sizeof(*p));
            if (!p)
                return NULL;
            p->N = N;
            p->K = K;
            p->mb = vfft_proto_plan_create(N, K, e->factors,
                                           (int *)e->variants, e->nf,
                                           (vfft_proto_registry_t *)reg);
            if (p->mb && !p->mb->use_dif_forward)
            {
                p->kind = VFFT_OOP_KIND_MODEB;
                return p;
            }
            free(p);
        }
    }
    return NULL;
}

/* Same-binary round-robin pair tuner. Returns number of candidates
 * (0 = none unmasked), winner in *bestR1/*bestR2. */
static inline int vfft_oop_tune_pairs(int N, size_t K,
                                      int *bestR1, int *bestR2, int verbose)
{
    enum { MAXC = 17, ROUNDS = 15 };
    vfft_oop_plan_t *cand[MAXC];
    int nc = 0;
    /* the direct leaf competes too at N <= 128 (a pair hint must never
     * shadow a faster leaf) */
    if (N <= 128 && (K % 8u) == 0 && vfft_oop_leaf_fn(N))
    {
        vfft_oop_plan_t *p =
            (vfft_oop_plan_t *)calloc(1, sizeof(*p));
        if (p)
        {
            p->kind = VFFT_OOP_KIND_LEAF;
            p->N = N; p->K = K;
            p->leaf = vfft_oop_leaf_fn(N);
            cand[nc++] = p;
        }
    }
    for (int R2 = N < 128 ? N : 128; R2 >= 2 && nc < MAXC; R2--)
    {
        if (N % R2)
            continue;
        vfft_oop_plan_t *p = vfft_oop_plan_create_pair(N, K, N / R2, R2);
        if (p)
            cand[nc++] = p;
    }
    if (nc == 0)
        return 0;

    size_t T = (size_t)N * K;
    double *sr = (double *)VFFT_OOP_AALLOC(T * 8);
    double *si = (double *)VFFT_OOP_AALLOC(T * 8);
    double *dr = (double *)VFFT_OOP_AALLOC(T * 8);
    double *di = (double *)VFFT_OOP_AALLOC(T * 8);
    for (size_t i = 0; i < T; i++)
    {
        sr[i] = (double)(i % 251) * 0.013 - 1.6;
        si[i] = (double)(i % 257) * 0.011 - 1.4;
    }
    unsigned long long best[MAXC];
    for (int c = 0; c < nc; c++)
    {
        best[c] = ~0ULL;
        vfft_oop_execute_fwd(cand[c], sr, si, dr, di); /* warm */
    }
    for (int r = 0; r < ROUNDS; r++)
        for (int c = 0; c < nc; c++)
        {
            unsigned long long t0 = __rdtsc();
            vfft_oop_execute_fwd(cand[c], sr, si, dr, di);
            unsigned long long dt = __rdtsc() - t0;
            if (dt < best[c])
                best[c] = dt;
        }
    int win = 0;
    for (int c = 1; c < nc; c++)
        if (best[c] < best[win])
            win = c;
    if (verbose)
    {
        printf("  tune N=%d K=%zu: %d candidate(s)\n", N, K, nc);
        for (int c = 0; c < nc; c++)
        {
            if (cand[c]->kind == VFFT_OOP_KIND_LEAF)
                printf("    leaf   %10llu cyc  speed vs winner %.3f%s\n",
                       best[c], (double)best[win] / best[c],
                       c == win ? "  <- winner" : "");
            else
                printf("    %2dx%-3d %10llu cyc  speed vs winner %.3f%s\n",
                       cand[c]->R1, cand[c]->R2, best[c],
                       (double)best[win] / best[c],
                       c == win ? "  <- winner" : "");
        }
    }
    if (cand[win]->kind == VFFT_OOP_KIND_LEAF)
    {
        *bestR1 = 0; /* leaf won: no pair hint; the rule spine handles it */
        *bestR2 = 0;
    }
    else
    {
        *bestR1 = cand[win]->R1;
        *bestR2 = cand[win]->R2;
    }
    for (int c = 0; c < nc; c++)
        vfft_oop_plan_destroy(cand[c]);
    VFFT_OOP_AFREE(sr); VFFT_OOP_AFREE(si); VFFT_OOP_AFREE(dr); VFFT_OOP_AFREE(di);
    return nc;
}

#endif /* VFFT_OOP_AUTO_H */
