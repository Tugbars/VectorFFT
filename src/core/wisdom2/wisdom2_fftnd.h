/* wisdom2_fftnd.h — the rank>=2 (2D/3D) transform-recipe home: entry
 * structs, recipe->plan builders, the 3D scratch table, and the LEGACY
 * file machinery — consolidated here at the owner's folder-structure
 * directive (2026-08-20) from transforms/fft2d/fft2d_c2c_wisdom.h,
 * fft2d_r2c_wisdom.h, and transforms/fft3d/fft3d_wisdom.h (deleted).
 * (Named wisdom2_fftnd.h, not fftnd_wisdom.h: transforms/fftnd owns a
 * live same-named header — the ND module's own file, wave-2 scope.)
 *
 * LIFETIME TIERS inside this file:
 *   PERMANENT — entry structs, plan_from_entry builders, the 3D scratch
 *     table + extract + create (the live wisdom2 serving path: the vw2
 *     twins in wisdom2_2d_reader.h fill the structs, these build plans).
 *   LEGACY (bake-window) — the fft2d file loaders, table lookups, frees,
 *     and plan_create_wisdom creators: alive only while the
 *     VFFT_WISDOM2_OFF=2d kill switch exists; they die in one commit at
 *     the 2D bake close, then the loaders survive migrator-only until
 *     v1.0. The three legacy files are FROZEN (stamped 2026-08-20).
 */
#ifndef VFFT_WISDOM2_FFTND_H
#define VFFT_WISDOM2_FFTND_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 1 — fft2d c2c (structs + builders PERMANENT; file machinery
 * LEGACY, see tier banner above)
 * ══════════════════════════════════════════════════════════════════════ */
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
#include "../transforms/fft2d/fft2d.h"   /* stride_plan_2d_from / stride_plan_2d / _fft2d_choose_tile +
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

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 2 — fft2d r2c/c2r (one struct serves both directions; direction
 * is the wisdom2 t= key)
 * ══════════════════════════════════════════════════════════════════════ */
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "../transforms/fft2d/fft2d_r2c.h"   /* stride_plan_2d_r2c_from + (transitively) planner.h (plan_create_ex,
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

/* vfft_fft2d_r2c_wisdom_add / _wisdom_save: DELETED at the wisdom2 wave-3
 * close (2026-08-20). Both fft2d real files are FROZEN — banks go through
 * vw2_2d_r2c_bank_entry (direction = the t= key, wisdom2_2d_reader.h);
 * this loader survives for the kill-switch bake window, then
 * migrator-only until v1.0. */

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
/* Build straight from ONE entry (the creator's body; also the wisdom2
 * flip's constructor — the vw2 twin fills an entry, this turns it into a
 * plan). NULL on invalid knobs or any build failure; caller owns fallback. */
static inline stride_plan_t *vfft_fft2d_r2c_plan_from_entry(
    const vfft_fft2d_r2c_wisdom_entry_t *e, const vfft_proto_registry_t *reg)
{
    const int N1 = e->N1, N2 = e->N2;
    const size_t hp1 = (size_t)(N2 / 2 + 1);
    size_t eB = (size_t)e->B, eKpad = (size_t)e->K_pad;
    if (e->row_nf <= 0 || e->col_nf <= 0) return NULL;
    /* validate the stored knobs against the 2D-create invariants */
    if (!(eB >= 2 && eB <= (size_t)N1 && (eKpad & 3) == 0 && eKpad >= hp1))
        return NULL;
    {
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
    }
    return NULL;
}

static inline stride_plan_t *vfft_fft2d_r2c_plan_create_wisdom(
    int N1, int N2, const vfft_fft2d_r2c_wisdom_t *w,
    const vfft_proto_registry_t *reg)
{
    size_t       B     = 8; if (B > (size_t)N1) B = (size_t)N1;
    const size_t hp1   = (size_t)(N2 / 2 + 1);
    size_t       K_pad = ((hp1 + 7) / 8) * 8;  /* §6a54: pad-to-8 — avx512 col pass full-width, no anyk tail (tail_handling doctrine) */

    const vfft_fft2d_r2c_wisdom_entry_t *e = vfft_fft2d_r2c_wisdom_lookup(w, N1, N2);
    if (e) {
        stride_plan_t *p = vfft_fft2d_r2c_plan_from_entry(e, reg);
        if (p) return p;
        /* any failure above => fall through to greedy fallback */
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

/* ════════════════════════════════════════════════════════════════════════
 * SECTION 3 — fft3d (fully PERMANENT: born in wisdom2, no legacy file
 * ever existed; the table is the in-process extraction scratch)
 * ══════════════════════════════════════════════════════════════════════ */
/**
 * fft3d_wisdom.h — dedicated 3D C2C wisdom (separate namespace).
 *
 * Mirror of fft2d_c2c_wisdom.h for the 3D transform (fft3d.h), fulfilling that
 * header's own request: "the vfft wisdom path should use stride_plan_3d_from
 * with calibrated inners". One entry per (N1,N2,N3) storing B + a_block + all
 * THREE inner sub-plans (factors, per-stage variants, DIT/DIF orientation):
 *   - axis0 c2c : N = N1, K = N2*N3
 *   - axis1 c2c : N = N2, K = N3
 *   - row   c2c : N = N3, K = B
 *
 * v1 banking: entries are extracted from the plans the exhaustive-at-create
 * builder produced (vfft.c _build_3d), so a cell pays the slow per-axis search
 * once and re-creates via stride_plan_3d_from thereafter. Variant extraction
 * reads stage flags (use_log3 -> LOG3, t1s_fwd -> T1S, else FLAT); the BUF
 * variant is not round-tripped (banked as FLAT — correctness-identical,
 * calibration may differ). best_ns is 0 for extraction-banked entries; a
 * dedicated end-to-end 3D (B, a_block) sweep is the 2D-planner-style
 * follow-up. Override (prime-axis Rader/Bluestein) plans are never banked —
 * their chains aren't expressible as factor lists; creates still succeed via
 * the greedy path.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "../transforms/fft3d/fft3d.h"   /* stride_plan_3d / stride_plan_3d_from + plan.h chain */

#define VFFT_FFT3D_WISDOM_VERSION 1

typedef struct {
    int N1, N2, N3;                             /* 3D cell key */
    int B;                                      /* row tile height (row c2c K) */
    int a_block;                                /* pass-A lane block; -1 = builder heuristic */
    int ax0_nf, ax0_factors[STRIDE_MAX_STAGES], ax0_variants[STRIDE_MAX_STAGES], ax0_dif;
    int ax1_nf, ax1_factors[STRIDE_MAX_STAGES], ax1_variants[STRIDE_MAX_STAGES], ax1_dif;
    int row_nf, row_factors[STRIDE_MAX_STAGES], row_variants[STRIDE_MAX_STAGES], row_dif;
    double best_ns;                             /* 0 = extraction-banked (unmeasured) */
} vfft_fft3d_wisdom_entry_t;

typedef struct {
    vfft_fft3d_wisdom_entry_t *entries;
    int count, cap;
} vfft_fft3d_wisdom_t;

static inline const vfft_fft3d_wisdom_entry_t *
vfft_fft3d_wisdom_lookup(const vfft_fft3d_wisdom_t *w, int N1, int N2, int N3)
{
    for (int i = 0; i < w->count; i++)
        if (w->entries[i].N1 == N1 && w->entries[i].N2 == N2 && w->entries[i].N3 == N3)
            return &w->entries[i];
    return NULL;
}

static inline int
vfft_fft3d_wisdom_put(vfft_fft3d_wisdom_t *w, const vfft_fft3d_wisdom_entry_t *e)
{
    for (int i = 0; i < w->count; i++)
        if (w->entries[i].N1 == e->N1 && w->entries[i].N2 == e->N2 &&
            w->entries[i].N3 == e->N3) { w->entries[i] = *e; return 0; }
    if (w->count == w->cap) {
        int nc = w->cap ? 2 * w->cap : 16;
        void *p = realloc(w->entries, (size_t)nc * sizeof(*w->entries));
        if (!p) return -1;
        w->entries = (vfft_fft3d_wisdom_entry_t *)p; w->cap = nc;
    }
    w->entries[w->count++] = *e;
    return 0;
}

static inline void vfft_fft3d_wisdom_free(vfft_fft3d_wisdom_t *w)
{
    free(w->entries);
    memset(w, 0, sizeof(*w));
}

/* vfft_fft3d_wisdom_save / _wisdom_load: DELETED at the wisdom2 wave-3
 * close (2026-08-20). The legacy 3D grammar NEVER materialized on disk —
 * 3D is born in wisdom2 (wisdom2_3d.txt via vw2_3d_bank_entry). This
 * table survives only as the in-process scratch the greedy creator's
 * extraction lands in before the harvest (vfft.c dims=3). */

/* Extract a bankable record from a built (non-override) stride plan. Returns
 * 0 on success, -1 for override/oversized chains (caller skips banking). */
static inline int _vfft_fft3d_extract(const stride_plan_t *p,
                                      int *nf, int *factors, int *variants, int *dif)
{
    if (!p || p->override_fwd || p->override_bwd) return -1;
    if (p->num_stages < 1 || p->num_stages > STRIDE_MAX_STAGES) return -1;
    *nf = p->num_stages;
    for (int s = 0; s < p->num_stages; s++) {
        factors[s]  = p->stages[s].radix;
        variants[s] = p->stages[s].use_log3 ? 1 : (p->stages[s].t1s_fwd ? 2 : 0);
    }
    *dif = p->use_dif_forward;
    return 0;
}

/* Wisdom-aware create. HIT -> create_ex x3 + stride_plan_3d_from (fast, the
 * fft3d.h-requested path). MISS -> replicate the greedy per-axis exhaustive
 * search with the inners VISIBLE, bank what is expressible, then _from. */
/* Build straight from ONE entry (the creator's body; also the wisdom2
 * flip's constructor). NULL on invalid/incompatible entry or build fail. */
static inline stride_plan_t *vfft_fft3d_plan_from_entry(
    const vfft_fft3d_wisdom_entry_t *e, const vfft_proto_registry_t *reg)
{
    const int N1 = e->N1, N2 = e->N2, N3 = e->N3;
    if (!(e->ax0_nf > 0 && e->ax1_nf > 0 && e->row_nf > 0 && e->B >= 1))
        return NULL;
    {
        stride_plan_t *p0 = vfft_proto_plan_create_ex(
            N1, (size_t)N2 * (size_t)N3, e->ax0_factors, e->ax0_variants,
            e->ax0_nf, e->ax0_dif, reg);
        stride_plan_t *p1 = p0 ? vfft_proto_plan_create_ex(
            N2, (size_t)N3, e->ax1_factors, e->ax1_variants,
            e->ax1_nf, e->ax1_dif, reg) : NULL;
        stride_plan_t *pr = p1 ? vfft_proto_plan_create_ex(
            N3, (size_t)e->B, e->row_factors, e->row_variants,
            e->row_nf, e->row_dif, reg) : NULL;
        if (pr) {
            stride_plan_t *p = stride_plan_3d_from(
                N1, N2, N3, (size_t)e->B,
                e->a_block < 0 ? (size_t)-1 : (size_t)e->a_block,
                p0, p1, pr);            /* owns all three */
            if (p) return p;
            p0 = p1 = pr = NULL;        /* _from freed them on failure */
        }
        if (pr) stride_plan_destroy(pr);
        if (p1) stride_plan_destroy(p1);
        if (p0) stride_plan_destroy(p0);
    }
    return NULL;
}

static inline stride_plan_t *vfft_fft3d_plan_create_wisdom(
    int N1, int N2, int N3, vfft_fft3d_wisdom_t *w,
    const vfft_proto_registry_t *reg, int *banked)
{
    if (banked) *banked = 0;
    const vfft_fft3d_wisdom_entry_t *e = vfft_fft3d_wisdom_lookup(w, N1, N2, N3);
    if (e) {
        stride_plan_t *p = vfft_fft3d_plan_from_entry(e, reg);
        if (p) return p;
        /* corrupt/incompatible entry: fall through to greedy */
    }
    /* greedy (stride_plan_3d body, inners kept visible for banking) */
    const size_t K0 = (size_t)N2 * (size_t)N3;
    const size_t NR = (size_t)N1 * (size_t)N2;
    size_t B = _fft3d_choose_tile(N3, NR);
    stride_plan_t *p0 = vfft_proto_exhaustive_plan(N1, K0, reg, 0);
    if (!p0) p0 = vfft_proto_auto_plan_dispatch(N1, K0, reg, NULL);
    if (!p0) return NULL;
    stride_plan_t *p1 = vfft_proto_exhaustive_plan(N2, (size_t)N3, reg, 0);
    if (!p1) p1 = vfft_proto_auto_plan_dispatch(N2, (size_t)N3, reg, NULL);
    if (!p1) { stride_plan_destroy(p0); return NULL; }
    stride_plan_t *pr = vfft_proto_exhaustive_plan(N3, B, reg, 0);
    if (!pr) pr = vfft_proto_auto_plan_dispatch(N3, B, reg, NULL);
    if (!pr) { stride_plan_destroy(p0); stride_plan_destroy(p1); return NULL; }

    vfft_fft3d_wisdom_entry_t ne; memset(&ne, 0, sizeof ne);
    ne.N1 = N1; ne.N2 = N2; ne.N3 = N3; ne.B = (int)B; ne.a_block = -1;
    if (_vfft_fft3d_extract(p0, &ne.ax0_nf, ne.ax0_factors, ne.ax0_variants, &ne.ax0_dif) == 0 &&
        _vfft_fft3d_extract(p1, &ne.ax1_nf, ne.ax1_factors, ne.ax1_variants, &ne.ax1_dif) == 0 &&
        _vfft_fft3d_extract(pr, &ne.row_nf, ne.row_factors, ne.row_variants, &ne.row_dif) == 0) {
        if (vfft_fft3d_wisdom_put(w, &ne) == 0 && banked) *banked = 1;
    }
    return stride_plan_3d_from(N1, N2, N3, B, (size_t)-1, p0, p1, pr);
}

#endif /* VFFT_WISDOM2_FFTND_H */
