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

#define VFFT_FFT2D_C2C_WISDOM_VERSION 1

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
    double best_ns;                         /* measured end-to-end 2D c2c fwd time */
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
               "col: nf factors.. variants.. dif | best_ns\n");
    for (size_t i = 0; i < w->count; i++) {
        const vfft_fft2d_c2c_wisdom_entry_t *e = &w->entries[i];
        fprintf(f, "%d %d %d  %d", e->N1, e->N2, e->B, e->row_nf);
        for (int s = 0; s < e->row_nf; s++) fprintf(f, " %d", e->row_factors[s]);
        for (int s = 0; s < e->row_nf; s++) fprintf(f, " %d", e->row_variants[s]);
        fprintf(f, " %d  %d", e->row_use_dif, e->col_nf);
        for (int s = 0; s < e->col_nf; s++) fprintf(f, " %d", e->col_factors[s]);
        for (int s = 0; s < e->col_nf; s++) fprintf(f, " %d", e->col_variants[s]);
        fprintf(f, " %d  %.1f\n", e->col_use_dif, e->best_ns);
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

#endif /* VFFT_FFT2D_C2C_WISDOM_H */
