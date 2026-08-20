/**
 * fft2d_c2c_wisdom.h -- dedicated 2D C2C wisdom (separate namespace).
 *
 * Mirror of fft2d_r2c_wisdom.h for the complex 2D transform (fft2d.h). Same
 * rationale: 2D plans are found by a dedicated 2D planner that MEASURES the
 * end-to-end 2D transform, NOT derived from 1D c2c wisdom (different memory
 * regime — tiled row pass + strided column pass). One entry per (N1,N2), each
 * storing BOTH inner sub-plans:
 *   - row c2c : N = N2, K = B
 *   - col c2c : N = N1, K = N2
 * with factors + per-stage variants + DIT/DIF orientation each.
 */
#ifndef VFFT_FFT2D_C2C_WISDOM_H
#define VFFT_FFT2D_C2C_WISDOM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "fft2d.h"   /* stride_plan_2d_from / stride_plan_2d / _fft2d_choose_tile +
                      * (transitively) planner.h (plan_create_ex, auto_plan), plan.h */

#define VFFT_FFT2D_C2C_WISDOM_VERSION 3

typedef struct {
    int    N1, N2;                          /* 2D cell key */
    int    B;                               /* row tile height (= row c2c batch K) */
    /* row c2c (N=N2, K=B) */
    int    row_nf;
    int    row_factors [STRIDE_MAX_STAGES];
    int    row_variants[STRIDE_MAX_STAGES]; /* 0=FLAT 1=LOG3 2=T1S 3=BUF */
    int    row_use_dif;
    /* col c2c (N=N1, K=N2) */
    int    col_nf;
    int    col_factors [STRIDE_MAX_STAGES];
    int    col_variants[STRIDE_MAX_STAGES];
    int    col_use_dif;
    double best_ns;                         /* measured end-to-end 2D c2c fwd time (SCRAMBLED) */
} vfft_fft2d_c2c_wisdom_entry_t;

/* SELF-CONTAINED natural-order record (order=VFFT_ORDER_NATURAL): its OWN natural-optimal (row,col,nat_B)
 * factorization — the one minimizing the NATURAL total (FFT + dim1/dim2 reorder), which may differ from
 * the scrambled winner (e.g. a palindromic col chain with a cheaper pair reorder). Keyed (N1,N2) in a
 * SEPARATE table, loaded from the SAME file via @nat2d lines (invisible to @/#-skipping external readers).
 * The 2D natural create reads ONLY this — the scrambled cal_ns<fb gate no longer governs its banking.
 * Design pivot 2026-07-06: scrambled and natural are different objectives + memory regimes. */
typedef struct {
    int    N1, N2, nat_B;
    int    row_nf;
    int    row_factors [STRIDE_MAX_STAGES];
    int    row_variants[STRIDE_MAX_STAGES];
    int    row_use_dif;
    int    col_nf;
    int    col_factors [STRIDE_MAX_STAGES];
    int    col_variants[STRIDE_MAX_STAGES];
    int    col_use_dif;
    double nat_ns;                          /* measured NATURAL total */
} vfft_fft2d_c2c_nat_entry_t;

typedef struct {
    vfft_fft2d_c2c_wisdom_entry_t *entries;
    size_t                         count;
    size_t                         capacity;
    vfft_fft2d_c2c_nat_entry_t    *nat;     /* second table, SAME file (@nat2d lines) */
    size_t                         nat_count;
    size_t                         nat_capacity;
} vfft_fft2d_c2c_wisdom_t;

/* Load: blank/#/@ skipped; token order:
 *   N1 N2 B  row_nf rf[..] rv[..] row_dif  col_nf cf[..] cv[..] col_dif  best_ns */
static inline int vfft_fft2d_c2c_wisdom_load(vfft_fft2d_c2c_wisdom_t *w,
                                             const char *path)
{
    memset(w, 0, sizeof(*w));
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[4096];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;
        if (*p == '@') {
            /* @nat2d = self-contained natural record (SEPARATE nat table); other @ lines are headers. */
            char *nt = strtok(p, " \t\r\n");
            if (nt && strcmp(nt, "@nat2d") == 0) {
                vfft_fft2d_c2c_nat_entry_t ne;
                memset(&ne, 0, sizeof(ne));
                char *t;
#define NN_INT(dst) do { t = strtok(NULL, " \t\r\n"); if (!t) goto skip; (dst) = atoi(t); } while (0)
                NN_INT(ne.N1); NN_INT(ne.N2); NN_INT(ne.nat_B);
                NN_INT(ne.row_nf);
                if (ne.row_nf <= 0 || ne.row_nf >= STRIDE_MAX_STAGES) continue;
                for (int i = 0; i < ne.row_nf; i++) NN_INT(ne.row_factors[i]);
                for (int i = 0; i < ne.row_nf; i++) NN_INT(ne.row_variants[i]);
                NN_INT(ne.row_use_dif);
                NN_INT(ne.col_nf);
                if (ne.col_nf <= 0 || ne.col_nf >= STRIDE_MAX_STAGES) continue;
                for (int i = 0; i < ne.col_nf; i++) NN_INT(ne.col_factors[i]);
                for (int i = 0; i < ne.col_nf; i++) NN_INT(ne.col_variants[i]);
                NN_INT(ne.col_use_dif);
#undef NN_INT
                t = strtok(NULL, " \t\r\n"); ne.nat_ns = t ? atof(t) : 0.0;
                if (w->nat_count >= w->nat_capacity) {
                    w->nat_capacity = w->nat_capacity ? w->nat_capacity * 2 : 32;
                    w->nat = (vfft_fft2d_c2c_nat_entry_t *)realloc(w->nat, w->nat_capacity * sizeof(*w->nat));
                }
                w->nat[w->nat_count++] = ne;
            }
            continue;
        }

        vfft_fft2d_c2c_wisdom_entry_t e;
        memset(&e, 0, sizeof(e));
        char *tok = strtok(p, " \t\r\n");
#define NEXT_INT(dst) do { tok = strtok(NULL, " \t\r\n"); if (!tok) goto skip; (dst) = atoi(tok); } while (0)
        if (!tok) continue;
        e.N1 = atoi(tok);
        NEXT_INT(e.N2); NEXT_INT(e.B);
        NEXT_INT(e.row_nf);
        if (e.row_nf <= 0 || e.row_nf >= STRIDE_MAX_STAGES) continue;
        for (int i = 0; i < e.row_nf; i++) NEXT_INT(e.row_factors[i]);
        for (int i = 0; i < e.row_nf; i++) NEXT_INT(e.row_variants[i]);
        NEXT_INT(e.row_use_dif);
        NEXT_INT(e.col_nf);
        if (e.col_nf <= 0 || e.col_nf >= STRIDE_MAX_STAGES) continue;
        for (int i = 0; i < e.col_nf; i++) NEXT_INT(e.col_factors[i]);
        for (int i = 0; i < e.col_nf; i++) NEXT_INT(e.col_variants[i]);
        NEXT_INT(e.col_use_dif);
        tok = strtok(NULL, " \t\r\n"); if (!tok) goto skip;
        e.best_ns = atof(tok);
#undef NEXT_INT
        /* Scrambled line ends at best_ns; any trailing tokens (a stray v2 embedded nat block from a
         * disposable staging file) are IGNORED — natural verdicts live in @nat2d lines / the nat table. */

        if (w->count >= w->capacity) {
            w->capacity = w->capacity ? w->capacity * 2 : 32;
            w->entries = (vfft_fft2d_c2c_wisdom_entry_t *)realloc(
                w->entries, w->capacity * sizeof(*w->entries));
        }
        w->entries[w->count++] = e;
        continue;
    skip:
        continue;
    }
    fclose(f);
    return 0;
}

static inline const vfft_fft2d_c2c_wisdom_entry_t *
vfft_fft2d_c2c_wisdom_lookup(const vfft_fft2d_c2c_wisdom_t *w, int N1, int N2)
{
    if (!w) return NULL;
    for (size_t i = 0; i < w->count; i++)
        if (w->entries[i].N1 == N1 && w->entries[i].N2 == N2)
            return &w->entries[i];
    return NULL;
}

/* vfft_fft2d_c2c_wisdom_add / _wisdom_save: DELETED at the wisdom2 wave-3
 * close (2026-08-20). fft2d_c2c_wisdom.txt is FROZEN — banks go through
 * vw2_2d_c2c_bank_entry/_bank_nat into the wisdom2 store (the ONE family
 * codec, wisdom2_2d_reader.h); this loader survives for the kill-switch
 * bake window, then migrator-only until v1.0. */

/* ── Natural table (order=VFFT_ORDER_NATURAL) lookup/upsert — keyed (N1,N2) on the SEPARATE nat table. ── */
static inline const vfft_fft2d_c2c_nat_entry_t *
vfft_fft2d_c2c_nat_lookup(const vfft_fft2d_c2c_wisdom_t *w, int N1, int N2)
{
    if (!w) return NULL;
    for (size_t i = 0; i < w->nat_count; i++)
        if (w->nat[i].N1 == N1 && w->nat[i].N2 == N2) return &w->nat[i];
    return NULL;
}

/* vfft_fft2d_c2c_nat_add: DELETED at the wisdom2 wave-3 close (2026-08-20)
 * — natural verdicts bank via vw2_2d_c2c_bank_nat. */

static inline void vfft_fft2d_c2c_wisdom_free(vfft_fft2d_c2c_wisdom_t *w)
{
    free(w->entries);
    free(w->nat);
    memset(w, 0, sizeof(*w));
}

/* Build straight from ONE entry (the shared body of both creators; also
 * the wisdom2 flip's constructor — the vw2 twins fill entries, this turns
 * them into plans). NULL on any build failure; the caller owns fallback. */
static inline stride_plan_t *vfft_fft2d_c2c_plan_from_fields(
    int N1, int N2, int B,
    const int *rf, const int *rv, int rnf, int rdif,
    const int *cf, const int *cv, int cnf, int cdif,
    const vfft_proto_registry_t *reg)
{
    size_t eB = (size_t)B;
    if (rnf <= 0 || cnf <= 0 || eB < 1 || eB > (size_t)N1) return NULL;
    {
        stride_plan_t *plan_row = vfft_proto_plan_create_ex(
            N2, eB, rf, rv, rnf, rdif, reg);
        if (plan_row) {
            stride_plan_t *plan_col = vfft_proto_plan_create_ex(
                N1, (size_t)N2, cf, cv, cnf, cdif, reg);
            if (plan_col) {
                stride_plan_t *p = stride_plan_2d_from(
                    N1, N2, eB, plan_col, plan_row); /* owns both */
                if (p) return p;
            } else {
                stride_plan_destroy(plan_row);
            }
        }
    }
    return NULL;
}

static inline stride_plan_t *vfft_fft2d_c2c_plan_from_entry(
    const vfft_fft2d_c2c_wisdom_entry_t *e, const vfft_proto_registry_t *reg)
{
    return vfft_fft2d_c2c_plan_from_fields(
        e->N1, e->N2, e->B,
        e->row_factors, e->row_variants, e->row_nf, e->row_use_dif,
        e->col_factors, e->col_variants, e->col_nf, e->col_use_dif, reg);
}

static inline stride_plan_t *vfft_fft2d_c2c_plan_from_nat_entry(
    const vfft_fft2d_c2c_nat_entry_t *e, const vfft_proto_registry_t *reg)
{
    return vfft_fft2d_c2c_plan_from_fields(
        e->N1, e->N2, e->nat_B,
        e->row_factors, e->row_variants, e->row_nf, e->row_use_dif,
        e->col_factors, e->col_variants, e->col_nf, e->col_use_dif, reg);
}

/* Wisdom-aware create. Calibrated plan if present, else the greedy default
 * (stride_plan_2d, which does its own exhaustive/auto inner search). */
static inline stride_plan_t *vfft_fft2d_c2c_plan_create_wisdom(
    int N1, int N2, const vfft_fft2d_c2c_wisdom_t *w,
    const vfft_proto_registry_t *reg)
{
    const vfft_fft2d_c2c_wisdom_entry_t *e = vfft_fft2d_c2c_wisdom_lookup(w, N1, N2);
    if (e) {
        stride_plan_t *p = vfft_fft2d_c2c_plan_from_entry(e, reg);
        if (p) return p;
    }
    /* greedy fallback (exhaustive/auto inner search inside stride_plan_2d) */
    return stride_plan_2d(N1, N2, reg);
}

/* NATURAL-aware create: build from the SELF-CONTAINED natural record (@nat2d) when the calibrator banked
 * one — the (row,col,nat_B) minimizing the NATURAL total, which may differ from the scrambled winner. Reads
 * ONLY the nat table. No natural record => fall back to the scrambled chain (runtime bolts the reorder on). */
static inline stride_plan_t *vfft_fft2d_c2c_plan_create_wisdom_natural(
    int N1, int N2, const vfft_fft2d_c2c_wisdom_t *w,
    const vfft_proto_registry_t *reg)
{
    const vfft_fft2d_c2c_nat_entry_t *e = vfft_fft2d_c2c_nat_lookup(w, N1, N2);
    if (e) {
        stride_plan_t *p = vfft_fft2d_c2c_plan_from_nat_entry(e, reg);
        if (p) return p;
    }
    return vfft_fft2d_c2c_plan_create_wisdom(N1, N2, w, reg);   /* no natural record -> scrambled chain */
}

#endif /* VFFT_FFT2D_C2C_WISDOM_H */
