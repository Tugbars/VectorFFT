/* wisdom2_2d_reader.h — the rank>=2 (2D/3D) family codec: wisdom2 records
 * <-> the EXACT legacy entry structs the plan constructors consume.
 *
 * READ twins (fill legacy structs verbatim; creators unchanged):
 *   legacy vfft_fft2d_c2c_wisdom_lookup(N1,N2)  -> vw2_2d_c2c_lookup_scr
 *   legacy vfft_fft2d_c2c_nat_lookup(N1,N2)     -> vw2_2d_c2c_lookup_nat
 *   legacy vfft_fft2d_r2c_wisdom_lookup(N1,N2)  -> vw2_2d_r2c_lookup (t= by
 *                                                  is_c2r — direction was
 *                                                  file membership, now the
 *                                                  transform tag)
 *   legacy vfft_fft3d_wisdom_lookup(N1,N2,N3)   -> vw2_3d_lookup
 *
 * CANONICAL-KEY LAW (dissolves design Q1): the 2D/3D plans are
 * placement-blind (_build_2d takes no placement; one legacy row served ip
 * AND oop), and the real-2D branch ignores order. Wildcards are
 * migration-only, so FRESH banks stamp the canonical concrete axes —
 * place=oop always; ord=nat for the order-blind real families — and the
 * lookups request the same canonical key. One row per cell serves every
 * consumer (the wave-1 kind-3/kind-4 precedent); migrated place=* / ord=*
 * rows serve the canonical request through the wildcard tier and sunset
 * as cells re-race.
 *
 * ns law: metric=fwd1 units=ns = one call of the KEYED transform (a c2r
 * row's ns is a c2r call — the legacy same-column/different-meaning trap
 * dissolves into the t= split). 3D extraction banks are MEASURE-LESS
 * (legacy best_ns==0.0 is a dead field, never encoded as ns=0).
 *
 * Variant vocabulary: flat/log3/t1s/buf (the rank>=2 chains carry BUF=3,
 * which the 1D kind-2 table never had).
 */
#ifndef VFFT_WISDOM2_2D_READER_H
#define VFFT_WISDOM2_2D_READER_H

#include <time.h>
#include "wisdom2.h"
#include "wisdom2_fftnd.h"   /* rank>=2 entry structs + builders (+ legacy tier) */

/* ------------------------------------------------------------ name maps */

static const char *vw2_2d_var_name[4] = { "flat", "log3", "t1s", "buf" };

static inline int vw2__2d_var_idx(const char *v)
{
    int i;
    if (!v) return -1;
    for (i = 0; i < 4; i++)
        if (!strcmp(vw2_2d_var_name[i], v)) return i;
    return -1;
}

/* "a.b.c" -> ints; returns count or 0 on any malformed piece */
static inline int vw2__2d_split_ints(const char *s, int *out, int cap)
{
    int n = 0;
    if (!s) return 0;
    while (*s) {
        char *end;
        long v = strtol(s, &end, 10);
        if (end == s || v <= 0 || n >= cap) return 0;
        out[n++] = (int)v;
        if (*end == '\0') break;
        if (*end != '.') return 0;
        s = end + 1;
    }
    return n;
}

/* "flat.t1s.buf" -> variant indexes; returns count or 0 on any unknown */
static inline int vw2__2d_split_vars(const char *s, int *out, int cap)
{
    char buf[24];
    int n = 0;
    if (!s) return 0;
    while (*s) {
        const char *dot = strchr(s, '.');
        size_t len = dot ? (size_t)(dot - s) : strlen(s);
        int idx;
        if (len == 0 || len >= sizeof buf || n >= cap) return 0;
        memcpy(buf, s, len); buf[len] = 0;
        idx = vw2__2d_var_idx(buf);
        if (idx < 0) return 0;
        out[n++] = idx;
        if (!dot) break;
        s = dot + 1;
    }
    return n;
}

static inline int vw2__2d_geti(const vw2_rec_t *r, const char *f, int dflt)
{
    const char *v = vw2_rec_get(r, f);
    return v ? atoi(v) : dflt;
}

/* one chain leg (plan/vars/dif prefix -> nf+factors+variants+dif); 1 = ok */
static inline int vw2__2d_leg(const vw2_rec_t *r, const char *plan_f,
                              const char *vars_f, const char *dif_f,
                              int *nf, int *factors, int *variants, int *dif)
{
    int nv;
    *nf = vw2__2d_split_ints(vw2_rec_get(r, plan_f), factors, STRIDE_MAX_STAGES);
    if (*nf <= 0 || *nf >= STRIDE_MAX_STAGES) return 0;
    nv = vw2__2d_split_vars(vw2_rec_get(r, vars_f), variants, STRIDE_MAX_STAGES);
    if (nv != *nf) return 0;                  /* vars/plan nf mismatch: refuse */
    *dif = vw2__2d_geti(r, dif_f, 0);
    return 1;
}

/* canonical request key (see the header law)
 *
 * lay= (v1.2, 2026-08-25): 2D/3D have ONE engine — the split stride
 * machinery; the interleaved caller is a convert wrap (c2c) or a fused
 * output veneer (r2c/c2r z doors, 3D il_out) around the SAME plan, so
 * every verdict shipped today is the shared interior's recipe and the
 * WRITERS stamp VW2_LAY_ANY deliberately. REQUESTS carry the CALLER's
 * layout: when the native IL 2D engine lands (measurement-first
 * campaign), its racer banks lay=il cells through its own doors and they
 * serve via lookup phase 1 with zero schema work; the ANY rows remain
 * serving vintage for both callers. */
static inline void vw2__2d_key(vw2_key_t *k, int t, int rank,
                               int n0, int n1, int n2, int ord, uint8_t lay)
{
    memset(k, 0, sizeof *k);
    k->t = (uint8_t)t;
    k->rank = (uint8_t)rank;
    k->n[0] = n0; k->n[1] = n1; k->n[2] = n2;
    k->q = 1;
    k->ord = (int8_t)ord;
    k->pl = VW2_PL_OOP;                       /* canonical: placement-blind */
    k->lay = lay;
}

/* ================================================================ READ */

static inline int vw2_2d_c2c_lookup_scr(const vw2_store_t *s, int N1, int N2,
                                        uint8_t lay,
                                        vfft_fft2d_c2c_wisdom_entry_t *e)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    vw2__2d_key(&k, VW2_T_C2C, 2, N1, N2, 0, VW2_ORD_SCR, lay);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    memset(e, 0, sizeof *e);
    e->N1 = N1; e->N2 = N2;
    e->B = vw2__2d_geti(r, "b", 0);
    if (e->B < 1) return 0;
    if (!vw2__2d_leg(r, "rowplan", "rowvars", "rowdif",
                     &e->row_nf, e->row_factors, e->row_variants, &e->row_use_dif))
        return 0;
    if (!vw2__2d_leg(r, "colplan", "colvars", "coldif",
                     &e->col_nf, e->col_factors, e->col_variants, &e->col_use_dif))
        return 0;
    {
        const char *ns = vw2_rec_get(r, "ns");
        e->best_ns = ns ? atof(ns) : 0.0;
    }
    return 1;
}

static inline int vw2_2d_c2c_lookup_nat(const vw2_store_t *s, int N1, int N2,
                                        uint8_t lay,
                                        vfft_fft2d_c2c_nat_entry_t *e)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    vw2__2d_key(&k, VW2_T_C2C, 2, N1, N2, 0, VW2_ORD_NAT, lay);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    memset(e, 0, sizeof *e);
    e->N1 = N1; e->N2 = N2;
    e->nat_B = vw2__2d_geti(r, "b", 0);
    if (e->nat_B < 1) return 0;
    if (!vw2__2d_leg(r, "rowplan", "rowvars", "rowdif",
                     &e->row_nf, e->row_factors, e->row_variants, &e->row_use_dif))
        return 0;
    if (!vw2__2d_leg(r, "colplan", "colvars", "coldif",
                     &e->col_nf, e->col_factors, e->col_variants, &e->col_use_dif))
        return 0;
    {
        const char *ns = vw2_rec_get(r, "ns");
        e->nat_ns = ns ? atof(ns) : 0.0;
    }
    return 1;
}

static inline int vw2_2d_r2c_lookup(const vw2_store_t *s, int is_c2r,
                                    int N1, int N2, uint8_t lay,
                                    vfft_fft2d_r2c_wisdom_entry_t *e)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    vw2__2d_key(&k, is_c2r ? VW2_T_C2R : VW2_T_R2C, 2, N1, N2, 0, VW2_ORD_NAT,
                lay);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    memset(e, 0, sizeof *e);
    e->N1 = N1; e->N2 = N2;
    e->B = vw2__2d_geti(r, "b", 0);
    /* k_pad VERBATIM, never re-derived (three pad conventions coexist);
     * serve-side validity stays the caller's (k_pad&3)==0 && >=hp1 check. */
    e->K_pad = vw2__2d_geti(r, "k_pad", 0);
    if (e->B < 1 || e->K_pad < 1) return 0;
    if (!vw2__2d_leg(r, "rowplan", "rowvars", "rowdif",
                     &e->row_nf, e->row_factors, e->row_variants, &e->row_use_dif))
        return 0;
    if (!vw2__2d_leg(r, "colplan", "colvars", "coldif",
                     &e->col_nf, e->col_factors, e->col_variants, &e->col_use_dif))
        return 0;
    {
        const char *ns = vw2_rec_get(r, "ns");
        e->best_ns = ns ? atof(ns) : 0.0;
    }
    return 1;
}

static inline int vw2_3d_lookup(const vw2_store_t *s, int N1, int N2, int N3,
                                uint8_t lay,
                                vfft_fft3d_wisdom_entry_t *e)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    vw2__2d_key(&k, VW2_T_C2C, 3, N1, N2, N3, VW2_ORD_SCR, lay);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    memset(e, 0, sizeof *e);
    e->N1 = N1; e->N2 = N2; e->N3 = N3;
    e->B = vw2__2d_geti(r, "b", 0);
    if (e->B < 1) return 0;
    e->a_block = vw2__2d_geti(r, "ablock", -1);   /* absent = heuristic */
    if (!vw2__2d_leg(r, "ax0plan", "ax0vars", "ax0dif",
                     &e->ax0_nf, e->ax0_factors, e->ax0_variants, &e->ax0_dif))
        return 0;
    if (!vw2__2d_leg(r, "ax1plan", "ax1vars", "ax1dif",
                     &e->ax1_nf, e->ax1_factors, e->ax1_variants, &e->ax1_dif))
        return 0;
    if (!vw2__2d_leg(r, "rowplan", "rowvars", "rowdif",
                     &e->row_nf, e->row_factors, e->row_variants, &e->row_dif))
        return 0;
    e->best_ns = 0.0;   /* dead legacy field; never carried (header law) */
    return 1;
}

/* ================================================================ WRITE
 * One family constructor per verdict shape. src = "race" (fresh bank:
 * canonical concrete key) | "migrated" (wildcard axes per the design;
 * from= required). All encoding goes through the name tables above. */

static inline void vw2__2d_stamp_date(vw2_rec_t *r)
{
    char d[16];
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    if (tm && strftime(d, sizeof d, "%Y-%m-%d", tm))
        vw2_rec_set(r, 2, "date", d);
}

#define VW2__2D_SET(sect, n, v) do { \
    if (vw2_rec_set(r, sect, n, v) != VW2_OK) { vw2_rec_free(r); *why = "token-refused"; return -1; } \
} while (0)

/* dot-join one chain leg into the record; 0 = ok, -1 = refused (*why set) */
static inline int vw2__2d_emit_leg(vw2_rec_t *r, const char *plan_f,
                                   const char *vars_f, const char *dif_f,
                                   int nf, const int *factors,
                                   const int *variants, int dif,
                                   const char **why)
{
    char buf[192];
    size_t off;
    int i;
    if (nf < 1 || nf >= STRIDE_MAX_STAGES) { vw2_rec_free(r); *why = "leg-nf-out-of-range"; return -1; }
    for (i = 0, off = 0; i < nf; i++) {
        int rr = snprintf(buf + off, sizeof buf - off, "%s%d", i ? "." : "", factors[i]);
        if (rr < 0 || (size_t)rr >= sizeof buf - off) { vw2_rec_free(r); *why = "leg-overflow"; return -1; }
        off += (size_t)rr;
    }
    VW2__2D_SET(1, plan_f, buf);
    for (i = 0, off = 0; i < nf; i++) {
        int rr;
        if (variants[i] < 0 || variants[i] > 3) { vw2_rec_free(r); *why = "garbage-variant-token"; return -1; }
        rr = snprintf(buf + off, sizeof buf - off, "%s%s", i ? "." : "",
                      vw2_2d_var_name[variants[i]]);
        if (rr < 0 || (size_t)rr >= sizeof buf - off) { vw2_rec_free(r); *why = "leg-overflow"; return -1; }
        off += (size_t)rr;
    }
    VW2__2D_SET(1, vars_f, buf);
    VW2__2D_SET(1, dif_f, dif ? "1" : "0");
    return 0;
}

/* shared tail: MEASURE section + provenance. ns<=0 => measure-less. */
static inline int vw2__2d_tail(vw2_rec_t *r, double ns, const char *src,
                               const char *from, const char **why)
{
    VW2__2D_SET(2, "ran", "1");
    if (ns > 0.0) {
        char nsbuf[48];
        snprintf(nsbuf, sizeof nsbuf, "%.1f", ns);
        VW2__2D_SET(2, "ns", nsbuf);
        VW2__2D_SET(2, "metric", "fwd1");
        VW2__2D_SET(2, "units", "ns");
    }
    VW2__2D_SET(2, "src", src);
    if (from) VW2__2D_SET(2, "from", from);
    else vw2__2d_stamp_date(r);
    return 0;
}

/* shared key emit; migrated rows wildcard the axes the legacy tables never
 * keyed (place always; ord too for the order-blind real families). */
static inline void vw2__2d_rec_key(vw2_rec_t *r, int t, int rank,
                                   int n0, int n1, int n2, int ord,
                                   int migrated, int ord_blind, uint8_t lay)
{
    memset(&r->key, 0, sizeof r->key);
    r->key.t = (uint8_t)t;
    r->key.rank = (uint8_t)rank;
    r->key.n[0] = n0; r->key.n[1] = n1; r->key.n[2] = n2;
    r->key.q = 1;
    if (migrated) {
        r->key.ord = ord_blind ? VW2_ORD_ANY : (int8_t)ord;
        r->key.pl = VW2_PL_ANY;
        /* legacy files never recorded layout: VW2_LAY_ANY vintage */
    } else {
        r->key.ord = (int8_t)ord;             /* canonical concrete */
        r->key.pl = VW2_PL_OOP;
        r->key.lay = lay;
    }
}

static inline int vw2_2d_c2c_rec_from_entry(vw2_rec_t *r,
                                            const vfft_fft2d_c2c_wisdom_entry_t *e,
                                            uint8_t lay,
                                            const char *src, const char *from,
                                            const char **why)
{
    char b[16];
    int migrated = src && !strcmp(src, "migrated");
    *why = NULL;
    memset(r, 0, sizeof *r);
    vw2__2d_rec_key(r, VW2_T_C2C, 2, e->N1, e->N2, 0, VW2_ORD_SCR, migrated, 0,
                    lay);
    if (vw2__2d_emit_leg(r, "rowplan", "rowvars", "rowdif",
                         e->row_nf, e->row_factors, e->row_variants,
                         e->row_use_dif, why)) return -1;
    if (vw2__2d_emit_leg(r, "colplan", "colvars", "coldif",
                         e->col_nf, e->col_factors, e->col_variants,
                         e->col_use_dif, why)) return -1;
    snprintf(b, sizeof b, "%d", e->B);
    VW2__2D_SET(1, "b", b);
    return vw2__2d_tail(r, e->best_ns, src, from, why);
}

static inline int vw2_2d_c2c_rec_from_nat(vw2_rec_t *r,
                                          const vfft_fft2d_c2c_nat_entry_t *e,
                                          uint8_t lay,
                                          const char *src, const char *from,
                                          const char **why)
{
    char b[16];
    int migrated = src && !strcmp(src, "migrated");
    *why = NULL;
    memset(r, 0, sizeof *r);
    vw2__2d_rec_key(r, VW2_T_C2C, 2, e->N1, e->N2, 0, VW2_ORD_NAT, migrated, 0,
                    lay);
    if (vw2__2d_emit_leg(r, "rowplan", "rowvars", "rowdif",
                         e->row_nf, e->row_factors, e->row_variants,
                         e->row_use_dif, why)) return -1;
    if (vw2__2d_emit_leg(r, "colplan", "colvars", "coldif",
                         e->col_nf, e->col_factors, e->col_variants,
                         e->col_use_dif, why)) return -1;
    snprintf(b, sizeof b, "%d", e->nat_B);
    VW2__2D_SET(1, "b", b);
    return vw2__2d_tail(r, e->nat_ns, src, from, why);
}

static inline int vw2_2d_r2c_rec_from_entry(vw2_rec_t *r,
                                            const vfft_fft2d_r2c_wisdom_entry_t *e,
                                            int is_c2r, uint8_t lay,
                                            const char *src, const char *from,
                                            const char **why)
{
    char b[16];
    int migrated = src && !strcmp(src, "migrated");
    *why = NULL;
    memset(r, 0, sizeof *r);
    vw2__2d_rec_key(r, is_c2r ? VW2_T_C2R : VW2_T_R2C, 2, e->N1, e->N2, 0,
                    VW2_ORD_NAT, migrated, /*ord_blind=*/1, lay);
    if (vw2__2d_emit_leg(r, "rowplan", "rowvars", "rowdif",
                         e->row_nf, e->row_factors, e->row_variants,
                         e->row_use_dif, why)) return -1;
    if (vw2__2d_emit_leg(r, "colplan", "colvars", "coldif",
                         e->col_nf, e->col_factors, e->col_variants,
                         e->col_use_dif, why)) return -1;
    snprintf(b, sizeof b, "%d", e->B);
    VW2__2D_SET(1, "b", b);
    snprintf(b, sizeof b, "%d", e->K_pad);
    VW2__2D_SET(1, "k_pad", b);
    return vw2__2d_tail(r, e->best_ns, src, from, why);
}

static inline int vw2_3d_rec_from_entry(vw2_rec_t *r,
                                        const vfft_fft3d_wisdom_entry_t *e,
                                        uint8_t lay,
                                        const char *src, const char *from,
                                        const char **why)
{
    char b[16];
    int migrated = src && !strcmp(src, "migrated");
    *why = NULL;
    memset(r, 0, sizeof *r);
    vw2__2d_rec_key(r, VW2_T_C2C, 3, e->N1, e->N2, e->N3, VW2_ORD_SCR, migrated,
                    0, lay);
    if (vw2__2d_emit_leg(r, "ax0plan", "ax0vars", "ax0dif",
                         e->ax0_nf, e->ax0_factors, e->ax0_variants,
                         e->ax0_dif, why)) return -1;
    if (vw2__2d_emit_leg(r, "ax1plan", "ax1vars", "ax1dif",
                         e->ax1_nf, e->ax1_factors, e->ax1_variants,
                         e->ax1_dif, why)) return -1;
    if (vw2__2d_emit_leg(r, "rowplan", "rowvars", "rowdif",
                         e->row_nf, e->row_factors, e->row_variants,
                         e->row_dif, why)) return -1;
    snprintf(b, sizeof b, "%d", e->B);
    VW2__2D_SET(1, "b", b);
    if (e->a_block > 0) {                     /* -1/0 = heuristic: absent */
        snprintf(b, sizeof b, "%d", e->a_block);
        VW2__2D_SET(1, "ablock", b);
    }
    /* 3D best_ns is the dead always-0.0 extraction field: measure-less by
     * construction (the tail's ns>0 gate makes that automatic). */
    return vw2__2d_tail(r, e->best_ns, src, from, why);
}

#undef VW2__2D_SET

/* ================================================================ BANK
 * Memory-only (process coherence); persistence is the caller's guarded
 * vw2_save (config.wisdom_write). fill_only mirrors the legacy nat
 * regime-separation overwrite=0: fill a cold cell, never clobber a warm
 * one (a natural create must not degrade calibrated scrambled wisdom). */

static inline int vw2__2d_bank(vw2_store_t *st, vw2_rec_t *rec, int fill_only)
{
    int rc;
    if (fill_only && vw2_lookup(st, &rec->key)) {
        vw2_rec_free(rec);
        return VW2_OK;                        /* warm cell: keep it */
    }
    rc = vw2_bank(st, rec);
    if (rc != VW2_OK) vw2_rec_free(rec);      /* tokens move only on success */
    return rc;
}

static inline int vw2_2d_c2c_bank_entry(vw2_store_t *st,
                                        const vfft_fft2d_c2c_wisdom_entry_t *e,
                                        int fill_only, uint8_t lay)
{
    vw2_rec_t rec;
    const char *why = NULL;
    if (vw2_2d_c2c_rec_from_entry(&rec, e, lay, "race", NULL, &why)) {
        fprintf(stderr, "[wisdom2] 2d bank refused (%s)\n", why ? why : "?");
        return -1;
    }
    return vw2__2d_bank(st, &rec, fill_only);
}

static inline int vw2_2d_c2c_bank_nat(vw2_store_t *st,
                                      const vfft_fft2d_c2c_nat_entry_t *e,
                                      uint8_t lay)
{
    vw2_rec_t rec;
    const char *why = NULL;
    if (vw2_2d_c2c_rec_from_nat(&rec, e, lay, "race", NULL, &why)) {
        fprintf(stderr, "[wisdom2] 2d nat bank refused (%s)\n", why ? why : "?");
        return -1;
    }
    return vw2__2d_bank(st, &rec, 0);
}

static inline int vw2_2d_r2c_bank_entry(vw2_store_t *st,
                                        const vfft_fft2d_r2c_wisdom_entry_t *e,
                                        int is_c2r, uint8_t lay)
{
    vw2_rec_t rec;
    const char *why = NULL;
    if (vw2_2d_r2c_rec_from_entry(&rec, e, is_c2r, lay, "race", NULL, &why)) {
        fprintf(stderr, "[wisdom2] 2d %s bank refused (%s)\n",
                is_c2r ? "c2r" : "r2c", why ? why : "?");
        return -1;
    }
    return vw2__2d_bank(st, &rec, 0);
}

static inline int vw2_3d_bank_entry(vw2_store_t *st,
                                    const vfft_fft3d_wisdom_entry_t *e,
                                    uint8_t lay)
{
    vw2_rec_t rec;
    const char *why = NULL;
    if (vw2_3d_rec_from_entry(&rec, e, lay, "race", NULL, &why)) {
        fprintf(stderr, "[wisdom2] 3d bank refused (%s)\n", why ? why : "?");
        return -1;
    }
    return vw2__2d_bank(st, &rec, 0);
}

/* ═══ native IL 2D c2c tier cells (lay=il — fft2d_il_c2c_design.md M3) ═══
 * The tier's raced verdicts live in their OWN lay=il cells: key {t=c2c
 * rank=2 n=N1xN2 q=1 ord=scr pl=OOP lay=il}, payload chain= (dot-separated
 * radices, the COLUMN-pass factorization — measured ALIVE 2026-08-25:
 * 1.30x at 4096x64, 1.18x at 1024x1024 over the greedy default). ord=scr:
 * the multi-stage tier serves i digit-reversed, and the verdict is
 * order-independent (the chain feeds both directions). The lookup goes
 * through vw2_lookup, so a lay-less/split row can come back on the ANY
 * fallback phase — the chain-token check refuses it (split rows carry
 * rowplan/colplan, never chain=). Old binaries: cells invisible + opaque
 * carry (v1.2 architecture, proven). */
static inline int vw2_2d_il_chain_lookup(const vw2_store_t *s, int N1,
                                         int N2, int *Rs, int *nst,
                                         int *wl, int *tf, int *ro)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *cv;
    int m = 0;
    vw2__2d_key(&k, VW2_T_C2C, 2, N1, N2, 0, VW2_ORD_SCR, VW2_LAY_IL);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    cv = vw2_rec_get(r, "chain");
    if (!cv) return 0;                       /* an ANY/split row: refuse */
    /* the axis verdicts; ABSENT = -1 (chain-only vintage / unraced) */
    if (wl) { const char *v = vw2_rec_get(r, "wl"); *wl = v ? atoi(v) : -1; }
    if (tf) { const char *v = vw2_rec_get(r, "tf"); *tf = v ? atoi(v) : -1; }
    if (ro) { const char *v = vw2_rec_get(r, "ro"); *ro = v ? atoi(v) : -1; }
    while (*cv && m < 8) {
        int v = 0;
        if (*cv < '0' || *cv > '9') return 0;
        while (*cv >= '0' && *cv <= '9') v = v * 10 + (*cv++ - '0');
        if (v < 2) return 0;
        Rs[m++] = v;
        if (*cv == '.') cv++;
        else break;
    }
    if (!m || *cv) return 0;
    *nst = m;
    return 1;
}

static inline int vw2_2d_il_chain_bank(vw2_store_t *st, int N1, int N2,
                                       const int *Rs, int nst,
                                       int wl, int tf, int ro, double ns)
{
    vw2_rec_t rec;
    vw2_rec_t *r = &rec;
    const char *why = NULL;
    char b[64];
    int i, off = 0;
    memset(r, 0, sizeof *r);
    vw2__2d_rec_key(r, VW2_T_C2C, 2, N1, N2, 0, VW2_ORD_SCR,
                    /*migrated=*/0, /*ord_blind=*/0, VW2_LAY_IL);
    for (i = 0; i < nst && off < (int)sizeof b - 8; i++)
        off += snprintf(b + off, sizeof b - off, "%s%d", i ? "." : "",
                        Rs[i]);
    if (vw2_rec_set(r, 1, "chain", b) != VW2_OK) {
        vw2_rec_free(r);
        fprintf(stderr, "[wisdom2] il2d chain bank refused (token)\n");
        return -1;
    }
    /* the axis verdicts (negative = unraced: token not emitted) */
    if (wl >= 0) {
        snprintf(b, sizeof b, "%d", wl);
        if (vw2_rec_set(r, 1, "wl", b) != VW2_OK) goto tokfail;
    }
    if (tf >= 0) {
        snprintf(b, sizeof b, "%d", tf);
        if (vw2_rec_set(r, 1, "tf", b) != VW2_OK) goto tokfail;
    }
    if (ro >= 0) {
        snprintf(b, sizeof b, "%d", ro);
        if (vw2_rec_set(r, 1, "ro", b) != VW2_OK) goto tokfail;
    }
    if (0) {
    tokfail:
        vw2_rec_free(r);
        fprintf(stderr, "[wisdom2] il2d axis bank refused (token)\n");
        return -1;
    }
    if (vw2__2d_tail(r, ns, "race", NULL, &why)) {
        fprintf(stderr, "[wisdom2] il2d chain bank refused (%s)\n",
                why ? why : "?");
        return -1;
    }
    return vw2__2d_bank(st, r, 0);
}

/* ═══ native IL 2D REAL tier cells (lay=il ord=scr —
 * fft2d_real_il_design.md M3). Key {t=r2c rank=2 n=N1xN2 q=1 ord=scr
 * pl=OOP lay=il} — DIRECTION-SHARED: the c2r create reads the
 * r2c-keyed row (the pair law requires ONE chain for both directions;
 * the kind-5 zr_kv REAL-N-keyed precedent). COLLISION-FREE with the
 * veneer's real cells: those key ord=nat (vw2_2d_r2c_lookup) and carry
 * rowplan/colplan, never chain= — the chain-token check also refuses
 * them on any fallback phase. Payload: chain= (the column-pass
 * factorization, deployed greedy until the M3 chain race) + rw= (the
 * ROW ROUTE verdict: 0 = the per-row TC door, W>0 = the ROWSPLIT band
 * width) + wl= (the banded column walk's band width in ROWS; 0 =
 * unbanded — rows sit OUTSIDE the walk per §2.5, tfuse structurally
 * absent for real). ABSENT axis -> -1 = unraced. */
static inline int vw2_2d_rl_lookup(const vw2_store_t *s, int N1, int N2,
                                   int *Rs, int *nst, int *rw, int *wl,
                                   int *cmt, int *cmtt)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const char *cv;
    int m = 0;
    vw2__2d_key(&k, VW2_T_R2C, 2, N1, N2, 0, VW2_ORD_SCR, VW2_LAY_IL);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    cv = vw2_rec_get(r, "chain");
    if (!cv) return 0;                       /* a veneer/ANY row: refuse */
    if (rw) { const char *v = vw2_rec_get(r, "rw"); *rw = v ? atoi(v) : -1; }
    if (wl) { const char *v = vw2_rec_get(r, "wl"); *wl = v ? atoi(v) : -1; }
    /* cmt = the COLUMN-PASS MT verdict (1 = thread it, 0 = serial), and
     * cmtt = the thread count it was RACED AT. A verdict raced at T=4
     * must never serve a T=8 request, so the caller compares cmtt to its
     * own pool and re-races on a mismatch (the nthreads key axis
     * expressed as payload + validity, without disturbing the key
     * format every reader/writer/gate shares). */
    if (cmt) { const char *v = vw2_rec_get(r, "cmt"); *cmt = v ? atoi(v) : -1; }
    if (cmtt) { const char *v = vw2_rec_get(r, "cmtt"); *cmtt = v ? atoi(v) : -1; }
    while (*cv && m < 8) {
        int v = 0;
        if (*cv < '0' || *cv > '9') return 0;
        while (*cv >= '0' && *cv <= '9') v = v * 10 + (*cv++ - '0');
        if (v < 2) return 0;
        Rs[m++] = v;
        if (*cv == '.') cv++;
        else break;
    }
    if (!m || *cv) return 0;
    *nst = m;
    return 1;
}

static inline int vw2_2d_rl_bank(vw2_store_t *st, int N1, int N2,
                                 const int *Rs, int nst, int rw, int wl,
                                 int cmt, int cmtt, double ns)
{
    vw2_rec_t rec;
    vw2_rec_t *r = &rec;
    const char *why = NULL;
    char b[64];
    int i, off = 0;
    memset(r, 0, sizeof *r);
    vw2__2d_rec_key(r, VW2_T_R2C, 2, N1, N2, 0, VW2_ORD_SCR,
                    /*migrated=*/0, /*ord_blind=*/0, VW2_LAY_IL);
    for (i = 0; i < nst && off < (int)sizeof b - 8; i++)
        off += snprintf(b + off, sizeof b - off, "%s%d", i ? "." : "",
                        Rs[i]);
    if (vw2_rec_set(r, 1, "chain", b) != VW2_OK) {
        vw2_rec_free(r);
        fprintf(stderr, "[wisdom2] il2d real bank refused (token)\n");
        return -1;
    }
    if (rw >= 0) {
        snprintf(b, sizeof b, "%d", rw);
        if (vw2_rec_set(r, 1, "rw", b) != VW2_OK) {
            vw2_rec_free(r);
            fprintf(stderr, "[wisdom2] il2d real rw bank refused (token)\n");
            return -1;
        }
    }
    if (wl >= 0) {
        snprintf(b, sizeof b, "%d", wl);
        if (vw2_rec_set(r, 1, "wl", b) != VW2_OK) {
            vw2_rec_free(r);
            fprintf(stderr, "[wisdom2] il2d real wl bank refused (token)\n");
            return -1;
        }
    }
    if (cmt >= 0 && cmtt > 0) {   /* the column-MT verdict + its T */
        snprintf(b, sizeof b, "%d", cmt);
        if (vw2_rec_set(r, 1, "cmt", b) != VW2_OK) {
            vw2_rec_free(r);
            fprintf(stderr, "[wisdom2] il2d real cmt bank refused (token)\n");
            return -1;
        }
        snprintf(b, sizeof b, "%d", cmtt);
        if (vw2_rec_set(r, 1, "cmtt", b) != VW2_OK) {
            vw2_rec_free(r);
            fprintf(stderr, "[wisdom2] il2d real cmtt bank refused (token)\n");
            return -1;
        }
    }
    if (vw2__2d_tail(r, ns, "race", NULL, &why)) {
        fprintf(stderr, "[wisdom2] il2d real bank refused (%s)\n",
                why ? why : "?");
        return -1;
    }
    return vw2__2d_bank(st, r, 0);
}


#endif /* VFFT_WISDOM2_2D_READER_H */
