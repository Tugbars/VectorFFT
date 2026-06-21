/**
 * fft2d_r2c_wisdom.h -- dedicated 2D R2C wisdom (separate namespace from 1D c2c).
 *
 * WHY ITS OWN WISDOM: the inner FFTs in a 2D r2c run in a different memory
 * regime than a standalone 1D batch (row pass = tile-local L1-resident with a
 * transpose each side; col pass = over the padded half-spectrum K_pad). The
 * optimal factorization AND per-stage variant differ, so 2D plans are found by
 * a dedicated 2D planner that MEASURES the end-to-end 2D transform — they are
 * NOT derived from 1D c2c wisdom. See fft2d_r2c_planner.h.
 *
 * One entry per (N1,N2). Each entry stores BOTH inner sub-plans:
 *   - row inner c2c   : N = N2/2, K = B          (wrapped by stride_r2c_plan)
 *   - col c2c         : N = N1,   K = K_pad
 * with each sub-plan's factors + per-stage variants + DIT/DIF orientation.
 *
 * Mirrors wisdom_reader.h (load/lookup/add(overwrite)/save/free) but with the
 * 2D key + two-subplan schema. v1 text format, one entry per line.
 */
#ifndef VFFT_FFT2D_R2C_WISDOM_H
#define VFFT_FFT2D_R2C_WISDOM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "fft2d_r2c.h"   /* stride_plan_2d_r2c_from + (transitively) planner.h (plan_create_ex,
                          * auto_plan, registry type), r2c.h (stride_r2c_plan), plan.h (STRIDE_MAX_STAGES) */

#define VFFT_FFT2D_R2C_WISDOM_VERSION 1

typedef struct {
    int    N1, N2;                          /* 2D cell key */
    int    B;                               /* row tile height */
    int    K_pad;                           /* col batch (mult of 4, >= N2/2+1) */
    /* row inner c2c (N=N2/2, K=B) */
    int    row_nf;
    int    row_factors [STRIDE_MAX_STAGES];
    int    row_variants[STRIDE_MAX_STAGES]; /* 0=FLAT 1=LOG3 2=T1S 3=BUF */
    int    row_use_dif;
    /* col c2c (N=N1, K=K_pad) */
    int    col_nf;
    int    col_factors [STRIDE_MAX_STAGES];
    int    col_variants[STRIDE_MAX_STAGES];
    int    col_use_dif;
    double best_ns;                         /* measured end-to-end 2D r2c fwd time */
} vfft_fft2d_r2c_wisdom_entry_t;

typedef struct {
    vfft_fft2d_r2c_wisdom_entry_t *entries;
    size_t                         count;
    size_t                         capacity;
} vfft_fft2d_r2c_wisdom_t;

/* ── load ──────────────────────────────────────────────────────────────────
 * Returns 0 on success, -1 on file-not-found. *w owns its entries (free with
 * vfft_fft2d_r2c_wisdom_free). Lines: blank/#/@ skipped; entry token order:
 *   N1 N2 B K_pad  row_nf rf[..] rv[..] row_dif  col_nf cf[..] cv[..] col_dif  best_ns
 */
static inline int vfft_fft2d_r2c_wisdom_load(vfft_fft2d_r2c_wisdom_t *w,
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

        vfft_fft2d_r2c_wisdom_entry_t e;
        memset(&e, 0, sizeof(e));
        char *tok = strtok(p, " \t\r\n");
#define NEXT_INT(dst) do { tok = strtok(NULL, " \t\r\n"); if (!tok) goto skip; (dst) = atoi(tok); } while (0)
        if (!tok) continue;
        e.N1 = atoi(tok);
        NEXT_INT(e.N2); NEXT_INT(e.B); NEXT_INT(e.K_pad);
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
            w->entries = (vfft_fft2d_r2c_wisdom_entry_t *)realloc(
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

static inline const vfft_fft2d_r2c_wisdom_entry_t *
vfft_fft2d_r2c_wisdom_lookup(const vfft_fft2d_r2c_wisdom_t *w, int N1, int N2)
{
    if (!w) return NULL;
    for (size_t i = 0; i < w->count; i++)
        if (w->entries[i].N1 == N1 && w->entries[i].N2 == N2)
            return &w->entries[i];
    return NULL;
}

/* Insert or replace (overwrite!=0) the (N1,N2) entry. Returns 1 (new), 2
 * (replaced), 0 (skipped — existing kept when overwrite==0). */
static inline int vfft_fft2d_r2c_wisdom_add(vfft_fft2d_r2c_wisdom_t *w,
                                            const vfft_fft2d_r2c_wisdom_entry_t *e,
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
        w->entries = (vfft_fft2d_r2c_wisdom_entry_t *)realloc(
            w->entries, w->capacity * sizeof(*w->entries));
    }
    w->entries[w->count++] = *e;
    return 1;
}

/* Write to path (v1 format, round-trips with load). Returns 0 / -1. */
static inline int vfft_fft2d_r2c_wisdom_save(const vfft_fft2d_r2c_wisdom_t *w,
                                             const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "@fft2d_r2c_version %d\n", VFFT_FFT2D_R2C_WISDOM_VERSION);
    fprintf(f, "# N1 N2 B K_pad | row: nf factors.. variants.. dif | "
               "col: nf factors.. variants.. dif | best_ns\n");
    for (size_t i = 0; i < w->count; i++) {
        const vfft_fft2d_r2c_wisdom_entry_t *e = &w->entries[i];
        fprintf(f, "%d %d %d %d  %d", e->N1, e->N2, e->B, e->K_pad, e->row_nf);
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

static inline void vfft_fft2d_r2c_wisdom_free(vfft_fft2d_r2c_wisdom_t *w)
{
    free(w->entries);
    memset(w, 0, sizeof(*w));
}

/* ── wisdom-aware create ────────────────────────────────────────────────────
 * Build a 2D r2c plan for (N1,N2). If 2D wisdom has the cell, build both inner
 * plans from the stored factors+variants+orientation (the calibrated choice);
 * otherwise fall back to the greedy auto_plan path (current default behavior —
 * estimate-mode is experimental/unwired, so the fallback is plain greedy).
 * Returns NULL only if even the fallback can't build. */
static inline stride_plan_t *vfft_fft2d_r2c_plan_create_wisdom(
    int N1, int N2, const vfft_fft2d_r2c_wisdom_t *w,
    const vfft_proto_registry_t *reg)
{
    const size_t hp1   = (size_t)(N2 / 2 + 1);
    size_t       B     = 8; if (B > (size_t)N1) B = (size_t)N1;
    size_t       K_pad = ((hp1 + 3) / 4) * 4;

    const vfft_fft2d_r2c_wisdom_entry_t *e = vfft_fft2d_r2c_wisdom_lookup(w, N1, N2);
    if (e && e->row_nf > 0 && e->col_nf > 0) {
        size_t eB = (size_t)e->B, eKpad = (size_t)e->K_pad;
        /* validate the stored knobs against the 2D-create invariants */
        if (eB >= 2 && eB <= (size_t)N1 && (eKpad & 3) == 0 && eKpad >= hp1) {
            stride_plan_t *inner = vfft_proto_plan_create_ex(
                N2 / 2, eB, e->row_factors, e->row_variants, e->row_nf, e->row_use_dif, reg);
            if (inner) {
                stride_plan_t *plan_r2c = stride_r2c_plan(N2, eB, eB, inner); /* owns inner */
                if (plan_r2c) {
                    stride_plan_t *plan_col = vfft_proto_plan_create_ex(
                        N1, eKpad, e->col_factors, e->col_variants, e->col_nf, e->col_use_dif, reg);
                    if (plan_col) {
                        stride_plan_t *p = stride_plan_2d_r2c_from(
                            N1, N2, eB, eKpad, plan_r2c, plan_col); /* owns both */
                        if (p) return p;
                        /* on failure stride_plan_2d_r2c_from already freed both */
                    } else {
                        stride_plan_destroy(plan_r2c); /* frees inner too */
                    }
                }
                /* plan_r2c NULL => stride_r2c_plan already freed inner */
            }
            /* any failure above => fall through to greedy fallback */
        }
    }

    /* greedy fallback (no calibrated wisdom for this cell) */
    {
        stride_plan_t *inner = vfft_proto_auto_plan(N2 / 2, B, reg, NULL);
        if (!inner) return NULL;
        stride_plan_t *plan_r2c = stride_r2c_plan(N2, B, B, inner);
        if (!plan_r2c) return NULL;
        stride_plan_t *plan_col = vfft_proto_auto_plan(N1, K_pad, reg, NULL);
        if (!plan_col) { stride_plan_destroy(plan_r2c); return NULL; }
        return stride_plan_2d_r2c_from(N1, N2, B, K_pad, plan_r2c, plan_col);
    }
}

#endif /* VFFT_FFT2D_R2C_WISDOM_H */
