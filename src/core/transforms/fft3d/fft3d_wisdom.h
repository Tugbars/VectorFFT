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
#ifndef VFFT_FFT3D_WISDOM_H
#define VFFT_FFT3D_WISDOM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fft3d.h"   /* stride_plan_3d / stride_plan_3d_from + plan.h chain */

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

/* One line per entry:
 *   N1 N2 N3 B a_block  nf f.. v.. dif  nf f.. v.. dif  nf f.. v.. dif  best_ns
 * '#' comment lines skipped; "@fft3dv" version guard first. */
static inline int vfft_fft3d_wisdom_save(const vfft_fft3d_wisdom_t *w, const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "@fft3dv %d\n", VFFT_FFT3D_WISDOM_VERSION);
    for (int i = 0; i < w->count; i++) {
        const vfft_fft3d_wisdom_entry_t *e = &w->entries[i];
        fprintf(f, "%d %d %d %d %d", e->N1, e->N2, e->N3, e->B, e->a_block);
        #define _W3AX(pre) do { \
            fprintf(f, "  %d", e->pre##_nf); \
            for (int j = 0; j < e->pre##_nf; j++) fprintf(f, " %d", e->pre##_factors[j]); \
            for (int j = 0; j < e->pre##_nf; j++) fprintf(f, " %d", e->pre##_variants[j]); \
            fprintf(f, " %d", e->pre##_dif); } while (0)
        _W3AX(ax0); _W3AX(ax1); _W3AX(row);
        #undef _W3AX
        fprintf(f, "  %.3f\n", e->best_ns);
    }
    fclose(f);
    return 0;
}

static inline int vfft_fft3d_wisdom_load(vfft_fft3d_wisdom_t *w, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    char line[2048];
    while (fgets(line, sizeof line, f)) {
        if (line[0] == '#' || line[0] == '\n') continue;
        if (line[0] == '@') continue;           /* version guard (v1: tolerant) */
        vfft_fft3d_wisdom_entry_t e; memset(&e, 0, sizeof e);
        char *t = strtok(line, " \t\r\n");
        #define _R3(dst) do { if (!t) goto skip; (dst) = atoi(t); t = strtok(NULL, " \t\r\n"); } while (0)
        _R3(e.N1); _R3(e.N2); _R3(e.N3); _R3(e.B); _R3(e.a_block);
        #define _R3AX(pre) do { \
            _R3(e.pre##_nf); \
            if (e.pre##_nf < 1 || e.pre##_nf > STRIDE_MAX_STAGES) goto skip; \
            for (int j = 0; j < e.pre##_nf; j++) _R3(e.pre##_factors[j]); \
            for (int j = 0; j < e.pre##_nf; j++) _R3(e.pre##_variants[j]); \
            _R3(e.pre##_dif); } while (0)
        _R3AX(ax0); _R3AX(ax1); _R3AX(row);
        #undef _R3AX
        if (t) e.best_ns = atof(t);
        #undef _R3
        vfft_fft3d_wisdom_put(w, &e);
        continue;
    skip:;
    }
    fclose(f);
    return 0;
}

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

#endif /* VFFT_FFT3D_WISDOM_H */
