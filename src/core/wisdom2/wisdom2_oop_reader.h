/* wisdom2_oop_reader.h — the LEGACY (dual) half of the OOP-family codec, for
 * the migrator and the vw2_off_oop kill switch only.
 *
 * Layout separation phase 5, the kind-3 record split (owner's option (a),
 * 2026-09-27). The runtime reads and banks through the per-layout codecs
 * (split/wisdom/wisdom2_oop_split.h, il/wisdom/wisdom2_oop_il.h). What stays
 * here is what only the FROZEN legacy format needs:
 *   vw2_oop_rec_from_entry   a legacy line -> its wisdom2 record: classic kinds
 *                            through the split constructor, a kind-3 line as
 *                            the lay-less DUAL row (both axes, as legacy wrote
 *                            it; the per-layout readers read its halves)
 *   vw2_oop_lookup_k1_dual   the two per-layout lookups composed back into one
 *                            dual entry, the shape the migrator's round-trip
 *                            check compares against a legacy line
 * The name tables (split: sp/var, interleaved: il) come from the codecs. */
#ifndef VFFT_WISDOM2_OOP_READER_H
#define VFFT_WISDOM2_OOP_READER_H

#include <time.h>
#include "wisdom2.h"
#include "wisdom2_oop.h"         /* the legacy format + both per-layout codecs */

/* ================================================================ WRITE
 * legacy entry -> wisdom2 record (the migrator's constructor). */

/* Build the record for kinds 0-3. src = "race" (fresh bank) | "migrated" |
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
        /* the classic kinds are the split library's: one constructor */
        vfft_oop_sp_entry_t sp;
        vfft_oop_sp_from_legacy(&sp, e);
        return vw2_oop_rec_classic(r, &sp, src, from, why);
    }
    else if (e->kind == VFFT_OOP_KIND_BAILEY2V) {
        r->key.t = VW2_T_C2C; r->key.rank = 1; r->key.n[0] = e->N;
        if (from) { r->key.q = -1; r->key.ord = VW2_ORD_ANY; r->key.pl = VW2_PL_ANY; }
        else      { r->key.q = 1;  r->key.ord = VW2_ORD_NAT; r->key.pl = VW2_PL_OOP; }
        /* v1.1: kind-3 is the K=1 ENGINE'S COMPONENT RECIPE (deliberately
         * order-agnostic) — role=comp keeps it off the problem-verdict key
         * the stride family's @natoop pick owns. */
        r->key.role = VW2_ROLE_COMP;
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
        if (e->k1_il_route < 0 || e->k1_il_route > VW2_OOP_IL_ROUTE_MAX) { vw2_rec_free(r); *why = "il-route-out-of-range"; return -1; }
        if (e->k1_il_route != VFFT_K1_IL_NONE) {
            VW2__OB_SET(1, "il_route", vw2_oop_il_name[e->k1_il_route]);
            if (e->il_R1 || e->il_R2) {
                snprintf(pair, sizeof pair, "%d.%d", e->il_R1, e->il_R2);
                VW2__OB_SET(1, "il_pair", pair);
            }
            if (e->k1_il_route == VFFT_K1_IL_CHAIN3 && e->il_c3[0]) {
                char c3b[48];              /* the chain IS the verdict */
                snprintf(c3b, sizeof c3b, "%d.%d.%d", e->il_c3[0], e->il_c3[1], e->il_c3[2]);
                VW2__OB_SET(1, "il_chain", c3b);
            }
            if (e->k1_il_route == VFFT_K1_IL_FLAT && e->il_fl_n >= 2) {
                char flb[64];              /* the flat chain + its per-stage forms ARE the verdict */
                size_t off = 0;
                for (i = 0; i < e->il_fl_n; i++) {
                    int rr = snprintf(flb + off, sizeof flb - off, "%s%d", i ? "." : "", e->il_fl[i]);
                    if (rr < 0 || (size_t)rr >= sizeof flb - off) break;
                    off += (size_t)rr;
                }
                VW2__OB_SET(1, "il_flat", flb);
                if (e->il_flf[0]) VW2__OB_SET(1, "il_forms", e->il_flf);
                if (e->il_tw > 0) { char twb[16]; snprintf(twb, sizeof twb, "%d", e->il_tw); VW2__OB_SET(1, "il_tw", twb); }
            }
            if (e->k1_il_route == VFFT_K1_IL_ZTT && e->il_zt_n >= 2) {
                char ztb[48];              /* ZTURN-T: the chain IS the verdict */
                size_t off = 0;
                for (i = 0; i < e->il_zt_n; i++) {
                    int rr = snprintf(ztb + off, sizeof ztb - off, "%s%d", i ? "." : "", e->il_zt[i]);
                    if (rr < 0 || (size_t)rr >= sizeof ztb - off) break;
                    off += (size_t)rr;
                }
                VW2__OB_SET(1, "il_ztt", ztb);
                if (e->il_tw > 0) { char twb[16]; snprintf(twb, sizeof twb, "%d", e->il_tw); VW2__OB_SET(1, "il_tw", twb); }
            }
        }
        if (e->il_kv || e->il_kv_raced) {   /* a raced verdict emits il_kv even when 0 */
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
    else {
        *why = "unknown-kind";
        return -1;
    }

    VW2__OB_SET(2, "src", src);
    if (from) VW2__OB_SET(2, "from", from);
#undef VW2__OB_SET
    return VW2_OK;
}

/* the pre-split dual K=1 read, composed from the two per-layout lookups at
 * the one-thread, out-of-place, natural cell: the split axis's fields from
 * the split codec, the IL axis's from the interleaved one; K and ns follow the
 * pre-1.2 dual-line convention - the IL champion's numbers when an IL route
 * (> NONE) decoded, else the split verdict's. 1 when either axis decided. */
static inline int vw2_oop_lookup_k1_dual(const vw2_store_t *s, int N,
                                         vfft_oop_wisdom_entry_t *e)
{
    vfft_oop_sp_entry_t sp;
    vfft_oop_il_entry_t il;
    const int hs = vw2_oop_lookup_k1_sp_cell(s, N, 0, 0, 1, &sp);
    const int hi = vw2_oop_lookup_k1_il_cell(s, N, 0, 0, 1, &il);
    memset(e, 0, sizeof *e);
    e->N = N;
    e->kind = VFFT_OOP_KIND_BAILEY2V;
    e->K = 1;
    e->k1_sp_route = sp.k1_sp_route;
    e->R1 = sp.R1; e->R2 = sp.R2;
    e->cc_chain = sp.cc_chain; e->cc_vars = sp.cc_vars;
    if (hs) { e->K = sp.K; e->ns = sp.ns; }
    e->k1_il_route = il.k1_il_route;
    e->il_R1 = il.il_R1; e->il_R2 = il.il_R2;
    memcpy(e->il_c3, il.il_c3, sizeof e->il_c3);
    memcpy(e->il_fl, il.il_fl, sizeof e->il_fl); e->il_fl_n = il.il_fl_n;
    memcpy(e->il_flf, il.il_flf, sizeof e->il_flf); e->il_tw = il.il_tw;
    memcpy(e->il_zt, il.il_zt, sizeof e->il_zt); e->il_zt_n = il.il_zt_n;
    e->il_kv = il.il_kv; e->il_kv_raced = il.il_kv_raced;
    if (il.k1_il_route > VFFT_K1_IL_NONE) { e->K = il.K; e->ns = il.ns; }
    return hs || hi;
}

#endif /* VFFT_WISDOM2_OOP_READER_H */
