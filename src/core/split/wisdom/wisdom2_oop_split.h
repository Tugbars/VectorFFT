/* wisdom2_oop_split.h — the SPLIT library's OOP wisdom: its record and codec.
 *
 * vfft_oop_sp_entry_t carries what a split request is served from: the
 * classic OOP kinds 0-2 (LEAF / BAILEY2 / MODEB, per (N, K)) and the SPLIT
 * axis of the kind-3 K=1 cell (sp_route, the pair, the CCOL column chain and
 * its variants). The codec reads and writes only split rows (lay=split, the
 * classic eng=classic rows) and, for the kind-3 cell, the SPLIT fields of the
 * lay-less vintage row.
 *
 * Layout separation phase 5, the kind-3 record split (owner's option (a),
 * 2026-09-27): before it one vfft_oop_wisdom_entry_t and one reader carried
 * both layouts. Every function here is the split half of the pre-split
 * wisdom2_oop_reader.h / wisdom2_oop.h code, verbatim but for the types. The
 * interleaved half is il/wisdom/wisdom2_oop_il.h; the frozen legacy format
 * and its dual entry are common/wisdom/wisdom2_oop_legacy.h. */
#ifndef VFFT_WISDOM2_OOP_SPLIT_H
#define VFFT_WISDOM2_OOP_SPLIT_H

#include "wisdom2_oop_rows.h"     /* the row mechanics (common) */
#include "wisdom2_oop_legacy.h"   /* the frozen legacy entry (common): the converter below */
#include "oop_plan.h"             /* STRIDE_MAX_STAGES, the cc codecs, vfft_oop_plan_t */

typedef char vw2__oop_legacy_stages_match[
    (VW2_OOP_LEGACY_MAX_STAGES == STRIDE_MAX_STAGES) ? 1 : -1];

/* ---------------------------------------------------------------- the record */
typedef struct {
    int    N;
    size_t K;                             /* classic: the batch; kind 3: the split
                                           * lane-batch the verdict ran (ran=) */
    int    kind;                          /* VFFT_OOP_KIND_{LEAF,BAILEY2,MODEB,BAILEY2V} */
    int    R1, R2;                        /* BAILEY2 pair; kind 3: the split pair */
    int    t1p_variant;                   /* BAILEY2 s2: 0=flat 1=log3 */
    int    nf;                            /* MODEB */
    int    factors[STRIDE_MAX_STAGES];    /* MODEB */
    int    variants[STRIDE_MAX_STAGES];   /* MODEB per-stage 0=FLAT 1=LOG3 2=T1S */
    int    k1_sp_route;                   /* kind 3: VFFT_K1_SP_*; -1 = unraced */
    int    cc_chain;                      /* kind 3, CCOL: encoded column chain */
    int    cc_vars;                       /* kind 3, CCOL: encoded column variants */
    int    place_ip;                      /* kind 3: the place=ip cell's own row */
    int    nthreads;                      /* kind 3: the plan's thread count (v1.3) */
    int    ord_scr;                       /* kind 3: the ord=scr cell's own row */
    int    role;
    double ns;                            /* measured (informational) */
} vfft_oop_sp_entry_t;

/* a legacy (dual) entry's SPLIT fields */
static inline void vfft_oop_sp_from_legacy(vfft_oop_sp_entry_t *o,
                                           const vfft_oop_wisdom_entry_t *e)
{
    int i;
    memset(o, 0, sizeof *o);
    o->N = e->N; o->K = e->K; o->kind = e->kind;
    o->R1 = e->R1; o->R2 = e->R2; o->t1p_variant = e->t1p_variant;
    o->nf = e->nf;
    for (i = 0; i < STRIDE_MAX_STAGES; i++) { o->factors[i] = e->factors[i]; o->variants[i] = e->variants[i]; }
    o->k1_sp_route = e->k1_sp_route; o->cc_chain = e->cc_chain; o->cc_vars = e->cc_vars;
    o->place_ip = e->place_ip; o->nthreads = e->nthreads; o->ord_scr = e->ord_scr;
    o->role = e->role; o->ns = e->ns;
}

/* ------------------------------------------------------------ name maps */
static const char *vw2_oop_sp_name[8] = {
    "3p", "2pa", "2pb", "twl", "mono", "2pa_l3", "3p_l3", "ccol"
};
static const char *vw2_oop_var_name[3] = { "flat", "log3", "t1s" };

/* "flat.t1s.log3" -> variant ints; returns count or 0 */
static inline int vw2__oop_split_vars(const char *s, int *out, int cap)
{
    char tok[16];
    int n = 0;
    if (!s) return 0;
    while (*s) {
        size_t l = 0;
        while (s[l] && s[l] != '.') l++;
        if (l == 0 || l >= sizeof tok) return 0;
        memcpy(tok, s, l);
        tok[l] = 0;
        {
            int v = vw2__oop_name_idx(vw2_oop_var_name, 3, tok);
            if (v < 0 || n >= cap) return 0;
            out[n++] = v;
        }
        s += l;
        if (*s == '.') s++;
    }
    return n;
}

/* --------------------------------------------------------- kind-3 (k1) */
/* the SPLIT axis of the kind-3 (K=1) cell: the lay=split row first, the
 * lay-less vintage row second, tiered by DECODE SUCCESS (a present-but-
 * undecodable cell must not shadow a still-decodable legacy verdict).
 * Returns 1 and fills e when the split axis decoded; 0 (e->k1_sp_route = -1,
 * UNRACED - 0 is VALID, 3P) when it did not. The pre-split reader's SP loop,
 * verbatim; K and ns are this axis's own. */
static inline int vw2_oop_lookup_k1_sp_cell(const vw2_store_t *s, int N, int want_scr,
                                            int inplace, int T, vfft_oop_sp_entry_t *e)
{
    const int pl = inplace ? VW2_PL_IP : VW2_PL_OOP;
    const vw2_rec_t *rsp = vw2__oop_k1_scan_pl(s, N, VW2_LAY_SPLIT, want_scr, pl, T);
    const vw2_rec_t *rlg = vw2__oop_k1_scan_pl(s, N, VW2_LAY_ANY, want_scr, pl, T);
    int pair[2], np, got = 0, si;
    memset(e, 0, sizeof *e);
    e->place_ip = inplace;
    e->nthreads = T > 1 ? T : 0;
    e->ord_scr = want_scr;
    e->kind = VFFT_OOP_KIND_BAILEY2V;
    e->N = N;
    e->K = 1;
    e->k1_sp_route = -1;                       /* UNRACED — 0 is VALID (3P) */
    /* SP axis: per-layout cell first, legacy row second — and the tier is
     * decided by DECODE SUCCESS, not record presence (a present-but-
     * undecodable cell — a future route token, a vars/chain mismatch —
     * must not shadow a still-decodable legacy verdict).
     * Decode into locals; commit only on full success. */
    for (si = 0; si < 2 && e->k1_sp_route < 0; si++) {
        const vw2_rec_t *rs = (si == 0) ? rsp : rlg;
        int spr, ccch = 0, ccv = 0;
        int ch[VFFT_K1_CC_MAX_NF], cv[VFFT_K1_CC_MAX_NF], nf, nv;
        if (!rs) continue;
        spr = vw2__oop_name_idx(vw2_oop_sp_name, 8, vw2_rec_get(rs, "sp_route"));
        if (spr < 0) continue;                 /* undecodable: next tier    */
        nf = vw2__oop_split_ints(vw2_rec_get(rs, "chain"), ch, VFFT_K1_CC_MAX_NF);
        if (nf > 0) {
            ccch = vfft_k1_cc_chain_encode(ch, nf);
            nv = vw2__oop_split_vars(vw2_rec_get(rs, "vars"), cv, VFFT_K1_CC_MAX_NF);
            if (nv == nf) ccv = vfft_k1_cc_vars_encode(cv, nv);
            else if (nv != 0) continue;        /* vars/chain nf mismatch    */
        }
        e->k1_sp_route = spr;
        e->cc_chain = ccch;
        e->cc_vars = ccv;
        np = vw2__oop_split_ints(vw2_rec_get(rs, "sp_pair"), pair, 2);
        if (np == 2) { e->R1 = pair[0]; e->R2 = pair[1]; }
        e->K = (size_t)vw2__oop_geti(rs, "ran", 1);
        { const char *ns = vw2_rec_get(rs, "ns"); e->ns = ns ? atof(ns) : 0.0; }
        got = 1;
    }
    return got;
}

/* --------------------------------------------------- kinds 0/1/2 (classic) */
/* Mirrors legacy lookup_ord for one order class (1 = natural, 2 = scrambled).
 * ord 0 (DEFAULT) = min-ns across the two classes when both are measured in
 * the same metric/units (the only shape the migrated data has). */
static inline int vw2__oop_classic_at(const vw2_store_t *s, int N, size_t K,
                                      int want_scr, vfft_oop_sp_entry_t *e)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    memset(&k, 0, sizeof k);
    k.t = VW2_T_C2C; k.rank = 1; k.n[0] = N;
    k.q = (int64_t)K; k.ord = want_scr ? VW2_ORD_SCR : VW2_ORD_NAT; k.pl = VW2_PL_OOP;
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    if (strcmp(vw2__oop_eng(r), "classic")) return 0;
    {
        const char *route = vw2_rec_get(r, "route");
        int f[STRIDE_MAX_STAGES], v[STRIDE_MAX_STAGES], nf, nv;
        memset(e, 0, sizeof *e);
        e->N = N; e->K = K;
        if (route && !strcmp(route, "leaf")) {
            e->kind = VFFT_OOP_KIND_LEAF;
        } else if (route && !strcmp(route, "bailey2")) {
            e->kind = VFFT_OOP_KIND_BAILEY2;
            nf = vw2__oop_split_ints(vw2_rec_get(r, "chain"), f, 2);
            if (nf != 2) return 0;
            e->R1 = f[0]; e->R2 = f[1];
            {
                const char *t1p = vw2_rec_get(r, "t1p");
                e->t1p_variant = (t1p && !strcmp(t1p, "log3")) ? 1 : 0;
            }
        } else if (route && !strcmp(route, "modeb")) {
            e->kind = VFFT_OOP_KIND_MODEB;
            nf = vw2__oop_split_ints(vw2_rec_get(r, "chain"), f, STRIDE_MAX_STAGES);
            if (nf < 1) return 0;
            nv = vw2__oop_split_vars(vw2_rec_get(r, "vars"), v, STRIDE_MAX_STAGES);
            if (nv != nf) return 0;
            e->nf = nf;
            { int i; for (i = 0; i < nf; i++) { e->factors[i] = f[i]; e->variants[i] = v[i]; } }
        } else {
            return 0;
        }
        {
            const char *ns = vw2_rec_get(r, "ns");
            e->ns = ns ? atof(ns) : 0.0;
        }
    }
    return 1;
}

static inline int vw2_oop_lookup_ord(const vw2_store_t *s, int N, size_t K,
                                     int ord, vfft_oop_sp_entry_t *e)
{
    if (ord == 1) return vw2__oop_classic_at(s, N, K, 0, e);
    if (ord == 2) return vw2__oop_classic_at(s, N, K, 1, e);
    {
        vfft_oop_sp_entry_t a, b;
        int ha = vw2__oop_classic_at(s, N, K, 0, &a);
        int hb = vw2__oop_classic_at(s, N, K, 1, &b);
        if (ha && hb) { *e = (b.ns > 0.0 && (a.ns <= 0.0 || b.ns < a.ns)) ? b : a; return 1; }
        if (ha) { *e = a; return 1; }
        if (hb) { *e = b; return 1; }
    }
    return 0;
}

/* ================================================================ WRITE */
/* the CLASSIC (kinds 0-2) record: the kinds-0-2 half of the pre-split
 * vw2_oop_rec_from_entry, verbatim. src = "race" (fresh bank) | "migrated" |
 * "seed"; from = lineage (NULL for fresh banks). On refusal returns -1 with
 * *why ("unknown-kind" for any other kind). */
static inline int vw2_oop_rec_classic(vw2_rec_t *r,
                                      const vfft_oop_sp_entry_t *e,
                                         const char *src, const char *from,
                                         const char **why)
{
    char nsbuf[48], pair[48], vars[192];
    int i;
    *why = NULL;
    memset(r, 0, sizeof *r);
    snprintf(nsbuf, sizeof nsbuf, "%.1f", e->ns);

#define VW2__OB_SET(sect, n, v) do { \
    if (vw2_rec_set(r, sect, n, v) != VW2_OK) { vw2_rec_free(r); *why = "token-refused"; return -1; } \
} while (0)

    if (e->kind == VFFT_OOP_KIND_LEAF || e->kind == VFFT_OOP_KIND_BAILEY2 ||
        e->kind == VFFT_OOP_KIND_MODEB) {
        r->key.t = VW2_T_C2C; r->key.rank = 1; r->key.n[0] = e->N;
        r->key.q = (int64_t)e->K;
        r->key.ord = (e->kind == VFFT_OOP_KIND_MODEB) ? VW2_ORD_SCR : VW2_ORD_NAT;
        r->key.pl = VW2_PL_OOP;
        VW2__OB_SET(1, "eng", "classic");
        if (e->kind == VFFT_OOP_KIND_LEAF) {
            VW2__OB_SET(1, "route", "leaf");
        } else if (e->kind == VFFT_OOP_KIND_BAILEY2) {
            VW2__OB_SET(1, "route", "bailey2");
            snprintf(pair, sizeof pair, "%d.%d", e->R1, e->R2);
            VW2__OB_SET(1, "chain", pair);
            VW2__OB_SET(1, "t1p", e->t1p_variant ? "log3" : "flat");
        } else {
            char joined[192];
            size_t off = 0;
            if (e->nf < 1 || e->nf > STRIDE_MAX_STAGES) { vw2_rec_free(r); *why = "modeb-nf-out-of-range"; return -1; }
            for (i = 0; i < e->nf; i++)
                if (e->variants[i] < 0 || e->variants[i] > 2) {
                    vw2_rec_free(r); *why = "garbage-variant-token"; return -1;
                }
            VW2__OB_SET(1, "route", "modeb");
            for (i = 0, off = 0; i < e->nf; i++) {
                int rr = snprintf(joined + off, sizeof joined - off, "%s%d", i ? "." : "", e->factors[i]);
                if (rr < 0 || (size_t)rr >= sizeof joined - off) break;
                off += (size_t)rr;
            }
            VW2__OB_SET(1, "chain", joined);
            for (i = 0, off = 0; i < e->nf; i++) {
                int rr = snprintf(vars + off, sizeof vars - off, "%s%s", i ? "." : "",
                                  vw2_oop_var_name[e->variants[i]]);
                if (rr < 0 || (size_t)rr >= sizeof vars - off) break;
                off += (size_t)rr;
            }
            VW2__OB_SET(1, "vars", vars);
        }
        {
            char ranb[24];
            snprintf(ranb, sizeof ranb, "%lld", (long long)e->K);
            VW2__OB_SET(2, "ran", ranb);
        }
        if (e->ns > 0.0) {
            VW2__OB_SET(2, "ns", nsbuf);
            VW2__OB_SET(2, "metric", "fwd1");
            VW2__OB_SET(2, "units", "cyc");    /* kinds 0-2 bank rdtsc cycles */
        }
    }
    else {
        *why = "unknown-kind";
        return -1;
    }

    VW2__OB_SET(2, "src", src);
    if (from) VW2__OB_SET(2, "from", from);
#undef VW2__OB_SET
    return VW2_OK;
}

/* bank a CLASSIC verdict (kinds 0-2) in memory; persistence is the caller's
 * guarded vw2_save. The pre-split vw2_oop_bank_entry for these kinds. */
static inline int vw2_oop_bank_classic(vw2_store_t *s, const vfft_oop_sp_entry_t *e)
{
    vw2_rec_t r;
    const char *why = NULL;
    int rc;
    if (vw2_oop_rec_classic(&r, e, "race", NULL, &why) != VW2_OK) {
        fprintf(stderr, "[wisdom2] oop bank refused (%s)\n", why ? why : "?");
        return -1;
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
}

/* Build ONE per-layout kind-3 record. Wisdom2-NATIVE, like the bwd builder:
 * fresh banks only — no legacy text line has per-layout shape. lay names
 * the CALLER LAYOUT this verdict serves (layout is an integration
 * property, AoS/SoA, never a strategy output). Each record carries ONLY
 * its own layout's fields, so neither layout's re-race can erase the
 * other's verdict, and neither layout's planning failure can veto the
 * other's bank — the two collision classes of the pre-1.2 dual line. */
static inline int vw2_oop_rec_k1_split(vw2_rec_t *r,
                                       const vfft_oop_sp_entry_t *e,
                                       const char **why)
{
    char nsbuf[48], pair[48], chain[192], vars[192];
    int i;
    *why = NULL;
    memset(r, 0, sizeof *r);
    snprintf(nsbuf, sizeof nsbuf, "%.1f", e->ns);
    r->key.t = VW2_T_C2C; r->key.rank = 1; r->key.n[0] = e->N;
    r->key.q = 1;
    r->key.pl = e->place_ip ? VW2_PL_IP : VW2_PL_OOP;   /* the in-place cell's own row */
    r->key.ord = e->ord_scr ? VW2_ORD_SCR : VW2_ORD_NAT;   /* the scrambled class's own cell */
    r->key.role = VW2_ROLE_COMP;
    r->key.lay  = VW2_LAY_SPLIT;
    r->key.nthreads = (uint8_t)(e->nthreads > 1 ? e->nthreads : 0);   /* the threaded plan's own row (v1.3) */
#define VW2__OB_SET(sect, n, v) do { \
    if (vw2_rec_set(r, sect, n, v) != VW2_OK) { vw2_rec_free(r); *why = "token-refused"; return -1; } \
} while (0)
    VW2__OB_SET(1, "eng", "k1");
    if (e->k1_sp_route < 0 || e->k1_sp_route > 7) { vw2_rec_free(r); *why = "sp-route-out-of-range"; return -1; }
    VW2__OB_SET(1, "sp_route", vw2_oop_sp_name[e->k1_sp_route]);
    snprintf(pair, sizeof pair, "%d.%d", e->R1, e->R2);
    VW2__OB_SET(1, "sp_pair", pair);
    if (e->k1_sp_route == VFFT_K1_SP_CCOL && e->cc_chain) {
        int ch[VFFT_K1_CC_MAX_NF], nf;
        nf = vfft_k1_cc_chain_decode(e->cc_chain, ch);
        if (nf <= 0) { vw2_rec_free(r); *why = "ccchain-decode-refused"; return -1; }
        {
            size_t off = 0;
            for (i = 0; i < nf; i++) {
                int rr = snprintf(chain + off, sizeof chain - off, "%s%d", i ? "." : "", ch[i]);
                if (rr < 0 || (size_t)rr >= sizeof chain - off) break;
                off += (size_t)rr;
            }
        }
        VW2__OB_SET(1, "chain", chain);
        if (e->cc_vars) {
            int cv[VFFT_K1_CC_MAX_NF];
            size_t off = 0;
            if (!vfft_k1_cc_vars_decode(e->cc_vars, nf, cv)) {
                vw2_rec_free(r); *why = "ccvars-decode-refused"; return -1;
            }
            for (i = 0; i < nf; i++) {
                int rr = snprintf(vars + off, sizeof vars - off, "%s%s", i ? "." : "",
                                  vw2_oop_var_name[cv[i]]);
                if (rr < 0 || (size_t)rr >= sizeof vars - off) break;
                off += (size_t)rr;
            }
            VW2__OB_SET(1, "vars", vars);
        }
    }
    {
        char ranb[24];
        snprintf(ranb, sizeof ranb, "%lld", (long long)e->K);
        VW2__OB_SET(2, "ran", ranb);
    }
    if (e->ns > 0.0) {
        VW2__OB_SET(2, "ns", nsbuf);
        VW2__OB_SET(2, "metric", "fwd1");
        VW2__OB_SET(2, "units", "ns");
    }
    VW2__OB_SET(2, "src", "race");
#undef VW2__OB_SET
    return VW2_OK;
}

static inline int vw2_oop_bank_k1_split(vw2_store_t *s,
                                        const vfft_oop_sp_entry_t *e)
{
    vw2_rec_t r;
    const char *why = NULL;
    int rc;
    if (vw2_oop_rec_k1_split(&r, e, &why) != VW2_OK) {
        fprintf(stderr, "[wisdom2] oop k1 %s bank refused (%s)\n",
                "lay=split", why ? why : "?");
        return -1;
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
}

/* ------------------------------------------------------------ plan builders */

/* Output-order class of an OOP kind: 1 = NATURAL (LEAF/BAILEY2), 0 = SCRAMBLED (MODEB). */
static inline int vfft_oop_sp_kind_natural(int kind) { return kind != VFFT_OOP_KIND_MODEB; }

/* Build the plan a SPECIFIC entry names (the order-aware path uses lookup_ord then this, so it can
 * pick the natural or the MODEB champion of a two-entry cell). Same build as create_wisdom's tail. */
static inline vfft_oop_plan_t *
vfft_oop_plan_from_entry(const vfft_oop_sp_entry_t *e, const vfft_proto_registry_t *reg)
{
    if (!e) return NULL;
    int N = e->N; size_t K = e->K;
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
        return vfft_oop_plan_create_pair_v(N, K, e->R1, e->R2, e->t1p_variant);
    if (e->kind == VFFT_OOP_KIND_MODEB)
        return _vfft_oop_make_modeb(N, K, e->factors, e->variants, e->nf, reg);
    return NULL;
}

/* Fill a wisdom entry from a finished plan (calibrator helper). ns is the
 * caller's measured time for the winner. */
static inline void vfft_oop_sp_entry_from_plan(vfft_oop_sp_entry_t *e,
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

#endif /* VFFT_WISDOM2_OOP_SPLIT_H */
