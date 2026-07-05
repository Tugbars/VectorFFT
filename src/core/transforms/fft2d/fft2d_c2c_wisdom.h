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

#define VFFT_FFT2D_C2C_WISDOM_VERSION 2

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
    /* v2 NATURAL-order block: the (row,col,B) factorization minimizing the NATURAL total (FFT + dim1/
     * dim2 reorder) — may differ from the scrambled winner (e.g. a palindromic col chain, cheaper pair
     * reorder). Absent in v1 files => nat_present=0 => order=NATURAL falls back to the scrambled chain +
     * bolt-on reorder (current behavior). The joint FFT+reorder scoring is what "natural-aware" means. */
    int    nat_present;
    int    nat_B;
    int    nat_row_nf;
    int    nat_row_factors [STRIDE_MAX_STAGES];
    int    nat_row_variants[STRIDE_MAX_STAGES];
    int    nat_row_use_dif;
    int    nat_col_nf;
    int    nat_col_factors [STRIDE_MAX_STAGES];
    int    nat_col_variants[STRIDE_MAX_STAGES];
    int    nat_col_use_dif;
    double nat_ns;                          /* measured NATURAL total for the natural chain */
} vfft_fft2d_c2c_wisdom_entry_t;

typedef struct {
    vfft_fft2d_c2c_wisdom_entry_t *entries;
    size_t                         count;
    size_t                         capacity;
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
        if (*p == '\0' || *p == '#' || *p == '@') continue;

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

        /* v2 optional trailing NATURAL block: nat_present [nat_B nat_row.. nat_col.. nat_ns]. Absent
         * (v1 file) or malformed => nat_present stays 0; NEVER skips the entry (scrambled stays valid). */
#define NAT_INT(dst) do { tok = strtok(NULL, " \t\r\n"); if (!tok) goto nat_fail; (dst) = atoi(tok); } while (0)
        tok = strtok(NULL, " \t\r\n");
        if (tok && atoi(tok) == 1) {
            NAT_INT(e.nat_B);
            NAT_INT(e.nat_row_nf);
            if (e.nat_row_nf <= 0 || e.nat_row_nf >= STRIDE_MAX_STAGES) goto nat_fail;
            for (int i = 0; i < e.nat_row_nf; i++) NAT_INT(e.nat_row_factors[i]);
            for (int i = 0; i < e.nat_row_nf; i++) NAT_INT(e.nat_row_variants[i]);
            NAT_INT(e.nat_row_use_dif);
            NAT_INT(e.nat_col_nf);
            if (e.nat_col_nf <= 0 || e.nat_col_nf >= STRIDE_MAX_STAGES) goto nat_fail;
            for (int i = 0; i < e.nat_col_nf; i++) NAT_INT(e.nat_col_factors[i]);
            for (int i = 0; i < e.nat_col_nf; i++) NAT_INT(e.nat_col_variants[i]);
            NAT_INT(e.nat_col_use_dif);
            tok = strtok(NULL, " \t\r\n"); e.nat_ns = tok ? atof(tok) : 0.0;
            e.nat_present = 1;
        }
        goto nat_store;
    nat_fail:
        e.nat_present = 0;
    nat_store:
#undef NAT_INT

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

static inline int vfft_fft2d_c2c_wisdom_add(vfft_fft2d_c2c_wisdom_t *w,
                                            const vfft_fft2d_c2c_wisdom_entry_t *e,
                                            int overwrite)
{
    for (size_t i = 0; i < w->count; i++) {
        if (w->entries[i].N1 == e->N1 && w->entries[i].N2 == e->N2) {
            if (!overwrite) return 0;
            w->entries[i] = *e;
            return 2;
        }
    }
    if (w->count >= w->capacity) {
        w->capacity = w->capacity ? w->capacity * 2 : 32;
        w->entries = (vfft_fft2d_c2c_wisdom_entry_t *)realloc(
            w->entries, w->capacity * sizeof(*w->entries));
    }
    w->entries[w->count++] = *e;
    return 1;
}

static inline int vfft_fft2d_c2c_wisdom_save(const vfft_fft2d_c2c_wisdom_t *w,
                                             const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "@fft2d_c2c_version %d\n", VFFT_FFT2D_C2C_WISDOM_VERSION);
    fprintf(f, "# N1 N2 B | row: nf factors.. variants.. dif | "
               "col: nf factors.. variants.. dif | best_ns "
               "[| nat_present nat_B nat_row.. nat_col.. nat_ns]\n");
    for (size_t i = 0; i < w->count; i++) {
        const vfft_fft2d_c2c_wisdom_entry_t *e = &w->entries[i];
        fprintf(f, "%d %d %d  %d", e->N1, e->N2, e->B, e->row_nf);
        for (int s = 0; s < e->row_nf; s++) fprintf(f, " %d", e->row_factors[s]);
        for (int s = 0; s < e->row_nf; s++) fprintf(f, " %d", e->row_variants[s]);
        fprintf(f, " %d  %d", e->row_use_dif, e->col_nf);
        for (int s = 0; s < e->col_nf; s++) fprintf(f, " %d", e->col_factors[s]);
        for (int s = 0; s < e->col_nf; s++) fprintf(f, " %d", e->col_variants[s]);
        fprintf(f, " %d  %.1f", e->col_use_dif, e->best_ns);
        if (e->nat_present) {                       /* v2 natural block */
            fprintf(f, "  1 %d  %d", e->nat_B, e->nat_row_nf);
            for (int s = 0; s < e->nat_row_nf; s++) fprintf(f, " %d", e->nat_row_factors[s]);
            for (int s = 0; s < e->nat_row_nf; s++) fprintf(f, " %d", e->nat_row_variants[s]);
            fprintf(f, " %d  %d", e->nat_row_use_dif, e->nat_col_nf);
            for (int s = 0; s < e->nat_col_nf; s++) fprintf(f, " %d", e->nat_col_factors[s]);
            for (int s = 0; s < e->nat_col_nf; s++) fprintf(f, " %d", e->nat_col_variants[s]);
            fprintf(f, " %d  %.1f", e->nat_col_use_dif, e->nat_ns);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return 0;
}

static inline void vfft_fft2d_c2c_wisdom_free(vfft_fft2d_c2c_wisdom_t *w)
{
    free(w->entries);
    memset(w, 0, sizeof(*w));
}

/* Wisdom-aware create. Calibrated plan if present, else the greedy default
 * (stride_plan_2d, which does its own exhaustive/auto inner search). */
static inline stride_plan_t *vfft_fft2d_c2c_plan_create_wisdom(
    int N1, int N2, const vfft_fft2d_c2c_wisdom_t *w,
    const vfft_proto_registry_t *reg)
{
    const vfft_fft2d_c2c_wisdom_entry_t *e = vfft_fft2d_c2c_wisdom_lookup(w, N1, N2);
    if (e && e->row_nf > 0 && e->col_nf > 0) {
        size_t eB = (size_t)e->B;
        if (eB >= 1 && eB <= (size_t)N1) {
            stride_plan_t *plan_row = vfft_proto_plan_create_ex(
                N2, eB, e->row_factors, e->row_variants, e->row_nf, e->row_use_dif, reg);
            if (plan_row) {
                stride_plan_t *plan_col = vfft_proto_plan_create_ex(
                    N1, (size_t)N2, e->col_factors, e->col_variants, e->col_nf, e->col_use_dif, reg);
                if (plan_col) {
                    stride_plan_t *p = stride_plan_2d_from(
                        N1, N2, eB, plan_col, plan_row); /* owns both */
                    if (p) return p;
                } else {
                    stride_plan_destroy(plan_row);
                }
            }
        }
    }
    /* greedy fallback (exhaustive/auto inner search inside stride_plan_2d) */
    return stride_plan_2d(N1, N2, reg);
}

/* NATURAL-aware create: build from the natural-optimal (row,col,B) chain (v2 nat block) when the
 * calibrator banked one — this is the factorization that minimizes the NATURAL total, which may differ
 * from the scrambled winner. No natural entry (v1 wisdom / uncalibrated) => fall back to the scrambled
 * chain (the runtime then bolts the reorder on, current behavior). */
static inline stride_plan_t *vfft_fft2d_c2c_plan_create_wisdom_natural(
    int N1, int N2, const vfft_fft2d_c2c_wisdom_t *w,
    const vfft_proto_registry_t *reg)
{
    const vfft_fft2d_c2c_wisdom_entry_t *e = vfft_fft2d_c2c_wisdom_lookup(w, N1, N2);
    if (e && e->nat_present && e->nat_row_nf > 0 && e->nat_col_nf > 0) {
        size_t eB = (size_t)e->nat_B;
        if (eB >= 1 && eB <= (size_t)N1) {
            stride_plan_t *plan_row = vfft_proto_plan_create_ex(
                N2, eB, e->nat_row_factors, e->nat_row_variants, e->nat_row_nf, e->nat_row_use_dif, reg);
            if (plan_row) {
                stride_plan_t *plan_col = vfft_proto_plan_create_ex(
                    N1, (size_t)N2, e->nat_col_factors, e->nat_col_variants, e->nat_col_nf, e->nat_col_use_dif, reg);
                if (plan_col) {
                    stride_plan_t *p = stride_plan_2d_from(N1, N2, eB, plan_col, plan_row);
                    if (p) return p;
                } else {
                    stride_plan_destroy(plan_row);
                }
            }
        }
    }
    return vfft_fft2d_c2c_plan_create_wisdom(N1, N2, w, reg);   /* scrambled chain fallback */
}

#endif /* VFFT_FFT2D_C2C_WISDOM_H */
