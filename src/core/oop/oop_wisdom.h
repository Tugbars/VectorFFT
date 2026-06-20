/* oop_wisdom.h — OOP c2c wisdom: the 2-axis decision {kind, factorization}
 * per (N, K), persisted and looked up.
 *
 * Why a SEPARATE file (not the c2c spike_wisdom): an OOP entry must encode BOTH
 * axes — the execution KIND (LEAF/BAILEY2/MODEB, axis 1) and its FACTORIZATION
 * (axis 2: BAILEY2 pair, or MODEB multi-factor). The c2c wisdom format is
 * MODEB-shaped only (factors+variants) and is shared with the in-place path
 * (different optima). So OOP gets its own store, mirroring rfft/c2r/c2c.
 *
 * File format (v2) — one entry per line, '#' comments and blanks ignored:
 *     N K kind [params...] ns
 *   kind 0 = LEAF    :  (no params)                     e.g.  64 512 0 117350.0
 *   kind 1 = BAILEY2 :  R1 R2 t1p                       e.g.  1024 120 1 32 32 1 185550.0
 *   kind 2 = MODEB   :  nf f0..f(nf-1) v0..v(nf-1)      e.g.  1024 256 2 5 4 4 4 4 4 0 2 2 2 2 502460.0
 *   t1p = BAILEY2 s2 twiddle variant (0=flat 1=log3).
 *   v0..v(nf-1) = MODEB per-stage twiddle variant (0=FLAT 1=LOG3 2=T1S).
 *   ns = measured wall time (informational; the dispatcher ignores it).
 *
 * v2 (this format) PERSISTS per-stage variants: MODEB is built from the in-place
 * c2c wisdom's variant-rich (factors, variants) — FLAT/T1S/LOG3 mixed per stage,
 * matching the in-place path — and BAILEY2 stores the tuner's flat-vs-log3 pick.
 * (v1 dropped variants and rebuilt all-T1S; that left per-stage tuning on the
 * table relative to the in-place engine.) MODEB is DIT-only (OOP stage 0 must be
 * untwiddled), so DIF-preferring in-place cells fall back to native LEAF/BAILEY2.
 *
 * Lifecycle: offline calibrator (vfft_oop_plan_create_dp_best) writes this file;
 * runtime vfft_oop_plan_create_wisdom() does a pure lookup + build, no measure.
 */
#ifndef VFFT_OOP_WISDOM_H
#define VFFT_OOP_WISDOM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oop_plan.h"   /* kinds, vfft_oop_plan_t, leaf/pair create, proto_plan_create */

#ifndef VFFT_OOP_WISDOM_MAX
#define VFFT_OOP_WISDOM_MAX 1024
#endif

typedef struct {
    int    N;
    size_t K;
    int    kind;                          /* VFFT_OOP_KIND_{LEAF,BAILEY2,MODEB} */
    int    R1, R2;                        /* BAILEY2 */
    int    t1p_variant;                   /* BAILEY2 s2: 0=flat 1=log3 */
    int    nf;                            /* MODEB */
    int    factors[STRIDE_MAX_STAGES];    /* MODEB */
    int    variants[STRIDE_MAX_STAGES];   /* MODEB per-stage 0=FLAT 1=LOG3 2=T1S */
    double ns;                            /* measured (informational) */
} vfft_oop_wisdom_entry_t;

typedef struct {
    vfft_oop_wisdom_entry_t e[VFFT_OOP_WISDOM_MAX];
    int count;
} vfft_oop_wisdom_t;

/* Load. Returns 0 on success, -1 if the file can't be opened. */
static inline int vfft_oop_wisdom_load(vfft_oop_wisdom_t *w, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    w->count = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (*s == '#' || *s == '\n' || *s == '\r' || *s == '\0') continue;
        if (w->count >= VFFT_OOP_WISDOM_MAX) break;
        vfft_oop_wisdom_entry_t *e = &w->e[w->count];
        memset(e, 0, sizeof *e);
        char *tok = strtok(s, " \t\n\r");      if (!tok) continue; e->N = atoi(tok);
        tok = strtok(NULL, " \t\n\r");          if (!tok) continue; e->K = (size_t)strtoull(tok, NULL, 10);
        tok = strtok(NULL, " \t\n\r");          if (!tok) continue; e->kind = atoi(tok);
        int ok = 1;
        if (e->kind == VFFT_OOP_KIND_BAILEY2) {
            tok = strtok(NULL, " \t\n\r"); if (tok) e->R1 = atoi(tok); else ok = 0;
            tok = strtok(NULL, " \t\n\r"); if (tok) e->R2 = atoi(tok); else ok = 0;
            tok = strtok(NULL, " \t\n\r"); if (tok) e->t1p_variant = atoi(tok); else ok = 0;
        } else if (e->kind == VFFT_OOP_KIND_MODEB) {
            tok = strtok(NULL, " \t\n\r"); if (tok) e->nf = atoi(tok); else ok = 0;
            if (ok && (e->nf <= 0 || e->nf > STRIDE_MAX_STAGES)) ok = 0;
            for (int i = 0; ok && i < e->nf; i++) {
                tok = strtok(NULL, " \t\n\r");
                if (tok) e->factors[i] = atoi(tok); else ok = 0;
            }
            /* v2: per-stage variant codes follow the factors */
            for (int i = 0; ok && i < e->nf; i++) {
                tok = strtok(NULL, " \t\n\r");
                if (tok) e->variants[i] = atoi(tok); else ok = 0;
            }
        }
        if (!ok) continue;
        tok = strtok(NULL, " \t\n\r");          /* ns (optional) */
        e->ns = tok ? atof(tok) : 0.0;
        w->count++;
    }
    fclose(f);
    return 0;
}

static inline const vfft_oop_wisdom_entry_t *
vfft_oop_wisdom_lookup(const vfft_oop_wisdom_t *w, int N, size_t K)
{
    if (!w) return NULL;
    for (int i = 0; i < w->count; i++)
        if (w->e[i].N == N && w->e[i].K == K)
            return &w->e[i];
    return NULL;
}

/* Build the exact plan a wisdom entry names — PURE LOOKUP, no measurement.
 * Returns NULL if (N,K) has no entry or the entry's codelets are unavailable
 * (caller then falls back to the rule spine / DP). */
static inline vfft_oop_plan_t *
vfft_oop_plan_create_wisdom(int N, size_t K, const vfft_oop_wisdom_t *w,
                            const vfft_proto_registry_t *reg)
{
    const vfft_oop_wisdom_entry_t *e = vfft_oop_wisdom_lookup(w, N, K);
    if (!e) return NULL;
    if (K == 0 || (K % 8u) != 0) return NULL;

    if (e->kind == VFFT_OOP_KIND_LEAF) {
        vfft_oop11_fn fn = vfft_oop_leaf_fn(N);
        if (!fn) return NULL;
        vfft_oop_plan_t *p = (vfft_oop_plan_t *)calloc(1, sizeof *p);
        if (!p) return NULL;
        p->kind = VFFT_OOP_KIND_LEAF; p->N = N; p->K = K; p->leaf = fn;
        return p;
    }
    if (e->kind == VFFT_OOP_KIND_BAILEY2)
        /* validates pair + mask; t1p variant is the tuner's persisted pick */
        return vfft_oop_plan_create_pair_v(N, K, e->R1, e->R2, e->t1p_variant);
    if (e->kind == VFFT_OOP_KIND_MODEB)
        /* v2: rebuild with the PERSISTED per-stage variants (the variant-rich
         * mix inherited from the in-place c2c wisdom), not all-T1S. Helper owns
         * construction + inner-plan teardown on failure. */
        return _vfft_oop_make_modeb(N, K, e->factors, e->variants, e->nf, reg);
    return NULL;
}

/* Fill a wisdom entry from a finished plan (calibrator helper). ns is the
 * caller's measured time for the winner. */
static inline void vfft_oop_wisdom_entry_from_plan(vfft_oop_wisdom_entry_t *e,
                                                   const vfft_oop_plan_t *p,
                                                   int N, size_t K, double ns)
{
    memset(e, 0, sizeof *e);
    e->N = N; e->K = K; e->kind = p->kind; e->ns = ns;
    if (p->kind == VFFT_OOP_KIND_BAILEY2) {
        e->R1 = p->R1; e->R2 = p->R2; e->t1p_variant = p->t1p_variant;
    }
    else if (p->kind == VFFT_OOP_KIND_MODEB && p->mb) {
        e->nf = p->mb->num_stages;
        for (int s = 0; s < e->nf && s < STRIDE_MAX_STAGES; s++) {
            e->factors[s]  = p->mb->factors[s];
            e->variants[s] = p->mb->variants[s];  /* recorded by plan_create_ex */
        }
    }
}

static inline void vfft_oop_wisdom_write_entry(FILE *f,
                                               const vfft_oop_wisdom_entry_t *e)
{
    fprintf(f, "%d %zu %d", e->N, e->K, e->kind);
    if (e->kind == VFFT_OOP_KIND_BAILEY2)
        fprintf(f, " %d %d %d", e->R1, e->R2, e->t1p_variant);
    else if (e->kind == VFFT_OOP_KIND_MODEB) {
        fprintf(f, " %d", e->nf);
        for (int s = 0; s < e->nf; s++) fprintf(f, " %d", e->factors[s]);
        for (int s = 0; s < e->nf; s++) fprintf(f, " %d", e->variants[s]);
    }
    fprintf(f, " %.1f\n", e->ns);
}

#endif /* VFFT_OOP_WISDOM_H */
