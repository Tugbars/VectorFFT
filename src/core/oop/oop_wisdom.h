/* oop_wisdom.h — OOP c2c wisdom: the 2-axis decision {kind, factorization}
 * per (N, K), persisted and looked up.
 *
 * Why a SEPARATE file (not the c2c spike_wisdom): an OOP entry must encode BOTH
 * axes — the execution KIND (LEAF/BAILEY2/MODEB, axis 1) and its FACTORIZATION
 * (axis 2: BAILEY2 pair, or MODEB multi-factor). The c2c wisdom format is
 * MODEB-shaped only (factors+variants) and is shared with the in-place path
 * (different optima). So OOP gets its own store, mirroring rfft/c2r/c2c.
 *
 * File format — one entry per line, '#' comments and blanks ignored:
 *     N K kind [params...] ns
 *   kind 0 = LEAF    :  (no params)            e.g.  64   512  0            117350.0
 *   kind 1 = BAILEY2 :  R1 R2                   e.g.  1024 120  1  32 32     185550.0
 *   kind 2 = MODEB   :  nf f0 f1 ... f(nf-1)    e.g.  1024 256  2  5 4 4 4 4 4  502460.0
 *   ns = measured wall time (informational; the dispatcher ignores it).
 *
 * MODEB variants are not stored: the DP planner is all-T1S today, and MODEB is
 * rebuilt with variants=NULL (= T1S default in vfft_proto_plan_create). When a
 * variant-aware DP lands, add a variants column and a format version bump.
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
    int    nf;                            /* MODEB */
    int    factors[STRIDE_MAX_STAGES];    /* MODEB */
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
        } else if (e->kind == VFFT_OOP_KIND_MODEB) {
            tok = strtok(NULL, " \t\n\r"); if (tok) e->nf = atoi(tok); else ok = 0;
            if (ok && (e->nf <= 0 || e->nf > STRIDE_MAX_STAGES)) ok = 0;
            for (int i = 0; ok && i < e->nf; i++) {
                tok = strtok(NULL, " \t\n\r");
                if (tok) e->factors[i] = atoi(tok); else ok = 0;
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
        return vfft_oop_plan_create_pair(N, K, e->R1, e->R2);  /* validates pair + mask */
    if (e->kind == VFFT_OOP_KIND_MODEB)
        /* OOP wisdom has NO variants column (the format drops them; the
         * calibrator's DP is all-T1S today) → rebuild NULL = T1S. If a
         * variant-aware DP lands, add a column, bump the format, pass them here.
         * Helper owns construction + inner-plan teardown on failure. */
        return _vfft_oop_make_modeb(N, K, e->factors, /*variants=*/NULL, e->nf, reg);
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
    if (p->kind == VFFT_OOP_KIND_BAILEY2) { e->R1 = p->R1; e->R2 = p->R2; }
    else if (p->kind == VFFT_OOP_KIND_MODEB && p->mb) {
        e->nf = p->mb->num_stages;
        for (int s = 0; s < e->nf && s < STRIDE_MAX_STAGES; s++)
            e->factors[s] = p->mb->factors[s];
    }
}

static inline void vfft_oop_wisdom_write_entry(FILE *f,
                                               const vfft_oop_wisdom_entry_t *e)
{
    fprintf(f, "%d %zu %d", e->N, e->K, e->kind);
    if (e->kind == VFFT_OOP_KIND_BAILEY2)
        fprintf(f, " %d %d", e->R1, e->R2);
    else if (e->kind == VFFT_OOP_KIND_MODEB) {
        fprintf(f, " %d", e->nf);
        for (int s = 0; s < e->nf; s++) fprintf(f, " %d", e->factors[s]);
    }
    fprintf(f, " %.1f\n", e->ns);
}

#endif /* VFFT_OOP_WISDOM_H */
