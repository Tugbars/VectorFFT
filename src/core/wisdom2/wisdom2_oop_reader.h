/* wisdom2_oop_reader.h — the READ side of the oop family: resolves wisdom2
 * records back into the EXACT legacy entry struct (vfft_oop_wisdom_entry_t)
 * the existing plan constructors consume. The constructors never change —
 * only the storage seam swaps (the decode-into-existing-structs move).
 * This is what vfft.c's create path calls at the wave-1 flip, mirroring the
 * legacy lookups one for one:
 *
 *   legacy vfft_oop_wisdom_lookup_k1(N)      -> vw2_oop_lookup_k1
 *   legacy vfft_oop_wisdom_lookup_zsplit(N)  -> vw2_oop_lookup_zsplit
 *   legacy vfft_oop_wisdom_lookup_ord(N,K,o) -> vw2_oop_lookup_ord
 *   legacy vfft_oop_wisdom_lookup_zr2c(N)    -> vw2_oop_lookup_zr2c
 *
 * Field encodings go BACK through the SHIPPED codecs
 * (vfft_k1_cc_chain_encode / vfft_k1_cc_vars_encode / vfft_zr2c_kv_set) —
 * one definition point, bit-exact round trips by construction.
 *
 * Legacy behaviors mirrored deliberately:
 *   - seeds (the K%8!=0 bank-only warts) never resolve — the legacy
 *     from_entry K%8 gate made those rows unservable too;
 *   - one verdict per key; the reader never guesses across records.
 *
 * Also owns the name tables (the migrator includes this header and uses
 * the same tables — forward and inverse maps cannot drift).
 */
#ifndef VFFT_WISDOM2_OOP_READER_H
#define VFFT_WISDOM2_OOP_READER_H

#include <time.h>
#include "wisdom2.h"
#include "../oop/oop_wisdom.h"   /* entry struct + THE codecs, verbatim */

/* ------------------------------------------------------------ name maps */

static const char *vw2_oop_sp_name[8] = {
    "3p", "2pa", "2pb", "twl", "mono", "2pa_l3", "3p_l3", "ccol"
};
static const char *vw2_oop_il_name[8] = {
    "none", "legacy3p", "legacy2p", "mono", "cascade", "2p", "chain3", "prime"
};
static const char *vw2_oop_var_name[3] = { "flat", "log3", "t1s" };

static inline int vw2__oop_name_idx(const char **tab, int n, const char *v)
{
    int i;
    if (!v) return -1;
    for (i = 0; i < n; i++)
        if (!strcmp(tab[i], v)) return i;
    return -1;
}

/* "a.b.c" -> ints; returns count or 0 on any malformed piece */
static inline int vw2__oop_split_ints(const char *s, int *out, int cap)
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

static inline int vw2__oop_geti(const vw2_rec_t *r, const char *name, int dflt)
{
    const char *v = vw2_rec_get(r, name);
    return v ? atoi(v) : dflt;
}

/* record -> is this the k1-engine family / cascade family / classic family?
 * The engine field IS the verdict; reading it to recognize the record's
 * family is reading the verdict, not smuggling semantics into sharding. */
static inline const char *vw2__oop_eng(const vw2_rec_t *r)
{
    const char *e = vw2_rec_get(r, "eng");
    return e ? e : "";
}

/* --------------------------------------------------------- kind-3 (k1) */

/* Mirrors legacy lookup_k1: one k1-engine row per N, axis-agnostic.
 * Returns 1 + fills e, or 0 on miss. */
static inline int vw2_oop_lookup_k1(const vw2_store_t *s, int N,
                                    vfft_oop_wisdom_entry_t *e)
{
    int i;
    for (i = 0; i < s->nrec; i++) {
        const vw2_rec_t *r = &s->rec[i];
        int pair[2], np;
        if (r->key.t != VW2_T_C2C || r->key.rank != 1 || r->key.n[0] != N) continue;
        if (strcmp(vw2__oop_eng(r), "k1")) continue;
        if (vw2__is_seed(r)) continue;
        memset(e, 0, sizeof *e);
        e->kind = VFFT_OOP_KIND_BAILEY2V;
        e->N = N;
        e->K = (size_t)vw2__oop_geti(r, "ran", 1);
        e->k1_sp_route = vw2__oop_name_idx(vw2_oop_sp_name, 8, vw2_rec_get(r, "sp_route"));
        if (e->k1_sp_route < 0) return 0;          /* unknown route: refuse  */
        np = vw2__oop_split_ints(vw2_rec_get(r, "sp_pair"), pair, 2);
        if (np == 2) { e->R1 = pair[0]; e->R2 = pair[1]; }
        {
            const char *il = vw2_rec_get(r, "il_route");
            e->k1_il_route = il ? vw2__oop_name_idx(vw2_oop_il_name, 8, il) : VFFT_K1_IL_NONE;
            if (e->k1_il_route < 0) return 0;
        }
        np = vw2__oop_split_ints(vw2_rec_get(r, "il_pair"), pair, 2);
        if (np == 2) { e->il_R1 = pair[0]; e->il_R2 = pair[1]; }
        e->il_kv = vw2__oop_geti(r, "il_kv", 0);
        {
            int ch[VFFT_K1_CC_MAX_NF], cv[VFFT_K1_CC_MAX_NF], nf, nv;
            nf = vw2__oop_split_ints(vw2_rec_get(r, "chain"), ch, VFFT_K1_CC_MAX_NF);
            if (nf > 0) {
                e->cc_chain = vfft_k1_cc_chain_encode(ch, nf);
                nv = vw2__oop_split_vars(vw2_rec_get(r, "vars"), cv, VFFT_K1_CC_MAX_NF);
                if (nv == nf) e->cc_vars = vfft_k1_cc_vars_encode(cv, nv);
                else if (nv != 0) return 0;        /* vars/chain nf mismatch */
            }
        }
        {
            const char *ns = vw2_rec_get(r, "ns");
            e->ns = ns ? atof(ns) : 0.0;
        }
        return 1;
    }
    return 0;
}

/* ------------------------------------------------------ kind-4 (cascade) */

static inline int vw2_oop_lookup_zsplit(const vw2_store_t *s, int N,
                                        vfft_oop_wisdom_entry_t *e)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    memset(&k, 0, sizeof k);
    k.t = VW2_T_C2C; k.rank = 1; k.n[0] = N;
    k.q = 1; k.ord = VW2_ORD_SCR; k.pl = VW2_PL_OOP;
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    {
        const char *eng = vw2__oop_eng(r);
        int zs_route;
        if (!strcmp(eng, "zturn")) zs_route = 1;
        else if (!strcmp(eng, "zsplit")) zs_route = 0;
        else return 0;                             /* a different engine won */
        memset(e, 0, sizeof *e);
        e->kind = VFFT_OOP_KIND_ZSPLIT;
        e->N = N; e->K = 1;
        e->zs_route = zs_route;
        e->zs_t2q = vw2__oop_geti(r, "zs_t2q", 0);
        e->zt_t2q = vw2__oop_geti(r, "zt_t2q", 0);
        e->zt_tw  = vw2__oop_geti(r, "zt_tw", 0);
        e->zt_l1  = vw2__oop_geti(r, "zt_l1", 0);
        {
            int ch[VFFT_K1_CC_MAX_NF], nf;
            nf = vw2__oop_split_ints(vw2_rec_get(r, "chain"), ch, VFFT_K1_CC_MAX_NF);
            if (nf <= 0) return 0;                 /* a cascade needs a chain */
            e->cc_chain = vfft_k1_cc_chain_encode(ch, nf);
        }
        {
            const char *ns = vw2_rec_get(r, "ns");
            e->ns = ns ? atof(ns) : 0.0;
        }
    }
    return 1;
}

/* --------------------------------------------------- kinds 0/1/2 (classic) */

/* Mirrors legacy lookup_ord for one order class (1 = natural, 2 = scrambled).
 * ord 0 (DEFAULT) = min-ns across the two classes when both are measured in
 * the same metric/units (the only shape the migrated data has). */
static inline int vw2__oop_classic_at(const vw2_store_t *s, int N, size_t K,
                                      int want_scr, vfft_oop_wisdom_entry_t *e)
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
                                     int ord, vfft_oop_wisdom_entry_t *e)
{
    if (ord == 1) return vw2__oop_classic_at(s, N, K, 0, e);
    if (ord == 2) return vw2__oop_classic_at(s, N, K, 1, e);
    {
        vfft_oop_wisdom_entry_t a, b;
        int ha = vw2__oop_classic_at(s, N, K, 0, &a);
        int hb = vw2__oop_classic_at(s, N, K, 1, &b);
        if (ha && hb) { *e = (b.ns > 0.0 && (a.ns <= 0.0 || b.ns < a.ns)) ? b : a; return 1; }
        if (ha) { *e = a; return 1; }
        if (hb) { *e = b; return 1; }
    }
    return 0;
}

/* ================================================================ WRITE
 * The other half of the family codec: legacy entry -> wisdom2 record.
 * ONE definition — the migrator, the runtime bank sites in vfft.c, and the
 * offline planners all construct records HERE (the four-constructor drift
 * of the old system is unrepresentable). */

/* Build the record for kinds 0-4. src = "race" (fresh bank) | "migrated" |
 * "seed"; from = lineage (required for migrated/seed wildcards, NULL for
 * fresh banks). Migrated kind-3 records carry the axis-agnostic wildcards;
 * FRESH kind-3 banks stamp the concrete canonical axes (q=1 ord=nat
 * place=oop — the k1-engine reader recognizes the family by eng=k1, so the
 * canonical key serves every consumer). On refusal returns -1 with *why. */
static inline int vw2_oop_rec_from_entry(vw2_rec_t *r,
                                         const vfft_oop_wisdom_entry_t *e,
                                         const char *src, const char *from,
                                         const char **why)
{
    char nsbuf[48], pair[48], chain[192], vars[192];
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
    else if (e->kind == VFFT_OOP_KIND_BAILEY2V) {
        r->key.t = VW2_T_C2C; r->key.rank = 1; r->key.n[0] = e->N;
        if (from) { r->key.q = -1; r->key.ord = VW2_ORD_ANY; r->key.pl = VW2_PL_ANY; }
        else      { r->key.q = 1;  r->key.ord = VW2_ORD_NAT; r->key.pl = VW2_PL_OOP; }
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
        if (e->k1_il_route < 0 || e->k1_il_route > 7) { vw2_rec_free(r); *why = "il-route-out-of-range"; return -1; }
        if (e->k1_il_route != VFFT_K1_IL_NONE) {
            VW2__OB_SET(1, "il_route", vw2_oop_il_name[e->k1_il_route]);
            if (e->il_R1 || e->il_R2) {
                snprintf(pair, sizeof pair, "%d.%d", e->il_R1, e->il_R2);
                VW2__OB_SET(1, "il_pair", pair);
            }
            if (e->k1_il_route == VFFT_K1_IL_CASCADE) {
                char ref[96];   /* the signpost: recipe lives in the cascade cell */
                snprintf(ref, sizeof ref, "cell(t=c2c,n=%d,q=1,ord=scr,place=oop)", e->N);
                VW2__OB_SET(1, "ref", ref);
            }
        }
        if (e->il_kv) {
            char kvb[16];
            snprintf(kvb, sizeof kvb, "%d", e->il_kv);
            VW2__OB_SET(1, "il_kv", kvb);
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
    }
    else if (e->kind == VFFT_OOP_KIND_ZSPLIT) {
        int ch[VFFT_K1_CC_MAX_NF], nf;
        if (e->N < 2048) { *why = "sub2048-wrong-slot"; return -1; }
        r->key.t = VW2_T_C2C; r->key.rank = 1; r->key.n[0] = e->N;
        r->key.q = 1; r->key.ord = VW2_ORD_SCR; r->key.pl = VW2_PL_OOP;
        VW2__OB_SET(1, "eng", e->zs_route == 1 ? "zturn" : "zsplit");
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
        {
            char tb[16];
            snprintf(tb, sizeof tb, "%d", e->zs_t2q);
            VW2__OB_SET(1, "zs_t2q", tb);
            if (e->zs_route == 1) {
                snprintf(tb, sizeof tb, "%d", e->zt_t2q);
                VW2__OB_SET(1, "zt_t2q", tb);
            }
            if (e->zt_tw > 0) {
                snprintf(tb, sizeof tb, "%d", e->zt_tw);
                VW2__OB_SET(1, "zt_tw", tb);
                snprintf(tb, sizeof tb, "%d", e->zt_l1);
                VW2__OB_SET(1, "zt_l1", tb);
            }
        }
        VW2__OB_SET(2, "ran", "1");
        if (e->ns > 0.0) {
            VW2__OB_SET(2, "ns", nsbuf);
            VW2__OB_SET(2, "metric", e->zs_route == 1 ? "joint2" : "fwd1");
            VW2__OB_SET(2, "units", "ns");
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

/* Build the up-to-4 per-slot records of a kind-5 packed verdict. Fills
 * out[] (caller-provided, size 4), returns the count (0 = all unmeasured). */
static inline int vw2_oop_recs_from_kind5(const vfft_oop_wisdom_entry_t *e,
                                          const char *src, const char *from,
                                          vw2_rec_t out[4], const char **why)
{
    int slot, n = 0;
    *why = NULL;
    for (slot = 0; slot < 4; slot++) {
        int v = vfft_zr2c_kv_get(e->zr_kv, slot);
        vw2_rec_t *s;
        if (!v) continue;
        s = &out[n];
        memset(s, 0, sizeof *s);
        s->key.t = (slot >> 1) ? VW2_T_C2R : VW2_T_R2C;
        s->key.rank = 1; s->key.n[0] = e->N;
        s->key.q = 1; s->key.ord = VW2_ORD_NAT;
        s->key.pl = (slot & 1) ? VW2_PL_IP : VW2_PL_OOP;
        if (vw2_rec_set(s, 1, "eng", "zr2c") != VW2_OK ||
            vw2_rec_set(s, 1, "route", v == 1 ? "child_oop_il" : "child_nat_ip") != VW2_OK ||
            vw2_rec_set(s, 2, "ran", "1") != VW2_OK ||
            vw2_rec_set(s, 2, "src", src) != VW2_OK ||
            (from && vw2_rec_set(s, 2, "from", from) != VW2_OK)) {
            int t;
            for (t = 0; t <= n; t++) vw2_rec_free(&out[t]);
            *why = "token-refused";
            return -1;
        }
        n++;
    }
    if (!n) { *why = "zr-kv-all-unmeasured"; return -1; }
    return n;
}

/* --------------------------------------------------- runtime bank helpers
 * What the vfft.c create-time bank sites call at the wave-1 flip. Fresh
 * banks: src=race, dated (merge tie-breaks need it). These helpers bank IN
 * MEMORY only (process coherence — README §2.2); DISK persistence is the
 * caller's guarded step (vw2_save under config.wisdom_write; tools save
 * explicitly). All refusals are loud but non-fatal — a failed bank never
 * fails a create. */

static inline void vw2__oop_stamp_date(vw2_rec_t *r)
{
    /* create-time banking timestamp (never on an execute path) */
    time_t t = time(NULL);
    struct tm *tmv = localtime(&t);
    char d[16];
    if (tmv && strftime(d, sizeof d, "%Y-%m-%d", tmv) > 0)
        vw2_rec_set(r, 2, "date", d);
}

static inline int vw2_oop_bank_entry(vw2_store_t *s, const vfft_oop_wisdom_entry_t *e)
{
    vw2_rec_t r;
    const char *why = NULL;
    int rc;
    if (e->kind == VFFT_OOP_KIND_ZR2C) {
        vw2_rec_t slots[4];
        int n = vw2_oop_recs_from_kind5(e, "race", NULL, slots, &why), i;
        if (n <= 0) {
            fprintf(stderr, "[wisdom2] oop bank refused (%s)\n", why ? why : "?");
            return -1;
        }
        for (i = 0; i < n; i++) {
            vw2__oop_stamp_date(&slots[i]);
            if (vw2_bank(s, &slots[i]) != VW2_OK) vw2_rec_free(&slots[i]);
        }
        return VW2_OK;
    }
    if (vw2_oop_rec_from_entry(&r, e, "race", NULL, &why) != VW2_OK) {
        fprintf(stderr, "[wisdom2] oop bank refused (%s)\n", why ? why : "?");
        return -1;
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
}

/* zr2c slot bank (replaces the legacy packed read-modify-write): banks ONE
 * (transform, placement) slot verdict directly — per-slot records need no
 * RMW, the other slots' records are untouched by construction. ns = the
 * slot's own race median (attributable here, unlike the legacy packed
 * line); <= 0 omits the measurement. */
static inline int vw2_oop_bank_zr2c_slot(vw2_store_t *s, int realN,
                                         int is_c2r, int is_inplace, int route,
                                         double ns)
{
    vw2_rec_t r;
    char b[48];
    int rc;
    memset(&r, 0, sizeof r);
    r.key.t = is_c2r ? VW2_T_C2R : VW2_T_R2C;
    r.key.rank = 1; r.key.n[0] = realN;
    r.key.q = 1; r.key.ord = VW2_ORD_NAT;
    r.key.pl = is_inplace ? VW2_PL_IP : VW2_PL_OOP;
    if (vw2_rec_set(&r, 1, "eng", "zr2c") != VW2_OK ||
        vw2_rec_set(&r, 1, "route", route ? "child_nat_ip" : "child_oop_il") != VW2_OK ||
        vw2_rec_set(&r, 2, "ran", "1") != VW2_OK ||
        vw2_rec_set(&r, 2, "src", "race") != VW2_OK) { vw2_rec_free(&r); return -1; }
    if (ns > 0.0) {
        snprintf(b, sizeof b, "%.1f", ns);
        if (vw2_rec_set(&r, 2, "ns", b) != VW2_OK ||
            vw2_rec_set(&r, 2, "metric", "fwd1") != VW2_OK ||
            vw2_rec_set(&r, 2, "units", "ns") != VW2_OK) { vw2_rec_free(&r); return -1; }
    }
    vw2__oop_stamp_date(&r);
    rc = vw2_bank(s, &r);
    if (rc != VW2_OK) { vw2_rec_free(&r); return rc; }
    return VW2_OK;
}

/* ------------------------------------------------------- kind-5 (zr2c) */

/* Reassembles the packed 4-slot verdict from the per-slot real records via
 * the SHIPPED kv codec. Returns 1 when any slot is measured. */
static inline int vw2_oop_lookup_zr2c(const vw2_store_t *s, int realN, int *zr_kv)
{
    int slot, any = 0, kv = 0;
    for (slot = 0; slot < 4; slot++) {
        vw2_key_t k;
        const vw2_rec_t *r;
        memset(&k, 0, sizeof k);
        k.t = (slot >> 1) ? VW2_T_C2R : VW2_T_R2C;
        k.rank = 1; k.n[0] = realN;
        k.q = 1; k.ord = VW2_ORD_NAT;
        k.pl = (slot & 1) ? VW2_PL_IP : VW2_PL_OOP;
        r = vw2_lookup(s, &k);
        if (!r) continue;
        if (strcmp(vw2__oop_eng(r), "zr2c")) continue;
        {
            const char *route = vw2_rec_get(r, "route");
            if (!route) continue;
            if (!strcmp(route, "child_oop_il"))      kv = vfft_zr2c_kv_set(kv, slot, 0);
            else if (!strcmp(route, "child_nat_ip")) kv = vfft_zr2c_kv_set(kv, slot, 1);
            else continue;
            any = 1;
        }
    }
    if (any) *zr_kv = kv;
    return any;
}

#endif /* VFFT_WISDOM2_OOP_READER_H */
