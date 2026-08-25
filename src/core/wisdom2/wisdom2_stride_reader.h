/* wisdom2_stride_reader.h — the stride (spike/rfft) family codec: wisdom2
 * records <-> the EXACT legacy entry structs (wisdom_reader.h) the plan
 * constructors consume.
 *
 * READ twins (fill legacy structs verbatim):
 *   legacy vfft_proto_wisdom_lookup(N,K)  -> vw2_stride_lookup      (t= by
 *                                            is_rfft — the legacy encoding
 *                                            was WHICH FILE; now the key)
 *   legacy vfft_proto_nat_lookup(N,K)     -> vw2_stride_lookup_nat
 *   legacy vfft_proto_natoop_lookup(N,K)  -> vw2_stride_lookup_natoop
 *
 * CANONICAL keys: scrambled + @nat rows are place=ip (the stride family
 * identity — the in-place strided engine); @natoop rows are place=oop.
 * rfft rows are t=r2c (the router puts them in the REAL shard — the key
 * decides the shard, never the file).
 *
 * SIGNPOST law (owner #7): a dummy-chain natural row (nf==1 &&
 * factors[0]==N — the @natoop zcasc placeholder) carries
 * `ref=cell(t=c2c,n=N,q=1,ord=scr,place=oop)` instead of the dummy; the
 * READ twin reconstructs the dummy deterministically when filling the
 * legacy struct. Real chains migrate/bank VERBATIM.
 *
 * pad_me (legacy exec_me) / il_me: emitted only when nonzero — absent =
 * not measured, exactly the legacy trailing-field law.
 */
#ifndef VFFT_WISDOM2_STRIDE_READER_H
#define VFFT_WISDOM2_STRIDE_READER_H

#include <time.h>
#include "wisdom2.h"
#include "../planning/wisdom_reader.h"   /* entry structs + THE legacy codec */

/* ------------------------------------------------------------ name maps */

static const char *vw2_stride_var_name[4] = { "flat", "log3", "t1s", "buf" };

/* @nat mode names, indexed by VFFT_NAT_* (0 = unset, never emitted).
 * leafip is RETIRED but old files may carry it — migrated verbatim,
 * never reused for a new meaning. */
static const char *vw2_stride_mode_name[8] = {
    "unset", "free", "leafip", "scr", "pcyc", "pswap", "zcasc", "ilp", "conv"
};

static inline int vw2__stride_mode_idx(const char *v)
{
    int i;
    if (!v) return -1;
    for (i = 1; i < 9; i++)
        if (!strcmp(vw2_stride_mode_name[i], v)) return i;
    return -1;
}

static inline int vw2__stride_var_idx(const char *v)
{
    int i;
    if (!v) return -1;
    for (i = 0; i < 4; i++)
        if (!strcmp(vw2_stride_var_name[i], v)) return i;
    return -1;
}

/* "a.b.c" -> ints; count or 0 on malformed */
static inline int vw2__stride_split_ints(const char *s, int *out, int cap)
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

/* "flat.t1s" -> variant indexes; count or 0 on unknown */
static inline int vw2__stride_split_vars(const char *s, int *out, int cap)
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
        idx = vw2__stride_var_idx(buf);
        if (idx < 0) return 0;
        out[n++] = idx;
        if (!dot) break;
        s = dot + 1;
    }
    return n;
}

static inline int vw2__stride_geti(const vw2_rec_t *r, const char *f, int dflt)
{
    const char *v = vw2_rec_get(r, f);
    return v ? atoi(v) : dflt;
}

static inline void vw2__stride_key(vw2_key_t *k, int t, int N, size_t K,
                                   int ord, int pl)
{
    memset(k, 0, sizeof *k);
    k->t = (uint8_t)t;
    k->rank = 1;
    k->n[0] = N;
    k->q = (int64_t)K;
    k->ord = (int8_t)ord;
    k->pl = (int8_t)pl;
}

/* ================================================================ READ */

/* TRIG HELPER CELLS (owner override 2026-08-19): a trig transform's inner
 * complex FFT is keyed under its OWNING transform, not as a plain c2c —
 * a DCT-I of N drives an inner c2c of N-1, which would otherwise collide
 * with a genuine c2c request at N-1. The record's n= is the OUTER size;
 * the inner size is derived HERE, in one place, so the read and write
 * sides cannot drift. Returns 0 for non-trig tags. */
static inline int vw2_stride_trig_inner_n(int t, int N)
{
    switch (t) {
    case VW2_T_DCT1: return N - 1;
    case VW2_T_DST1: return N + 1;
    case VW2_T_DCT2: case VW2_T_DCT3: case VW2_T_DCT4:
    case VW2_T_DST2: case VW2_T_DST3: case VW2_T_DHT:
        return N / 2;
    default: return 0;
    }
}

/* scrambled row for ANY transform tag: c2c = spike, r2c = rfft, a trig tag
 * = that transform's inner-cell verdict. 1 + fills e, or 0 on miss.
 * For a trig tag, e->N is the INNER size (what the plan is built for) while
 * the KEY carries the outer size. */
static inline int vw2_stride_lookup_t(const vw2_store_t *s, int t,
                                      int N, size_t K,
                                      vfft_proto_wisdom_entry_t *e)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    const int inner = vw2_stride_trig_inner_n(t, N);
    vw2__stride_key(&k, t, N, K, VW2_ORD_SCR, VW2_PL_IP);
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    if (!vw2_rec_get(r, "eng") || strcmp(vw2_rec_get(r, "eng"), "stride"))
        return 0;
    memset(e, 0, sizeof *e);
    e->N = inner ? inner : N;   /* trig: the plan is for the INNER size */
    e->K = K;
    e->nf = vw2__stride_split_ints(vw2_rec_get(r, "chain"), e->factors,
                                   STRIDE_MAX_STAGES);
    if (e->nf <= 0 || e->nf >= STRIDE_MAX_STAGES) return 0;
    {
        int nv = vw2__stride_split_vars(vw2_rec_get(r, "vars"), e->variants,
                                        STRIDE_MAX_STAGES);
        if (nv != e->nf) return 0;
    }
    e->use_dif_forward = vw2__stride_geti(r, "dif", 0);
    e->use_blocked = vw2__stride_geti(r, "blocked", 0);
    e->split_stage = vw2__stride_geti(r, "bsplit", 0);
    e->block_groups = vw2__stride_geti(r, "bgroups", 0);
    e->exec_me = vw2__stride_geti(r, "pad_me", 0);   /* absent = unmeasured */
    e->il_me = vw2__stride_geti(r, "il_me", 0);
    {
        const char *ns = vw2_rec_get(r, "ns");
        e->best_ns = ns ? atof(ns) : 0.0;
    }
    return 1;
}

/* the two long-standing callers: spike (c2c) and rfft (r2c) */
static inline int vw2_stride_lookup(const vw2_store_t *s, int is_rfft,
                                    int N, size_t K,
                                    vfft_proto_wisdom_entry_t *e)
{
    return vw2_stride_lookup_t(s, is_rfft ? VW2_T_R2C : VW2_T_C2C, N, K, e);
}

/* natural row (@nat: place=ip; @natoop: place=oop). Shared body.
 * lay (v1.2): the CALLER's layout. The mode payload is layout-gated
 * (ZCASC/ILP are interleaved-only candidates), so each layout owns its own
 * cell — the pre-1.2 shared cell made alternating-layout callers re-race
 * and erase each other's verdict on every flip (the @nat ping-pong).
 * vw2_lookup's two-phase resolution serves the caller's lay cell first and
 * falls back to the pre-1.2 lay-less row; VW2_LAY_ANY requests (the
 * migrate gate) match exactly the lay-less rows, byte-for-byte pre-1.2. */
static inline int vw2__stride_lookup_natx(const vw2_store_t *s, int pl,
                                          uint8_t lay, int ord, int N,
                                          size_t K,
                                          vfft_proto_nat_entry_t *e)
{
    vw2_key_t k;
    const vw2_rec_t *r;
    vw2__stride_key(&k, VW2_T_C2C, N, K, ord, pl);
    k.lay = lay;
    r = vw2_lookup(s, &k);
    if (!r) return 0;
    if (!vw2_rec_get(r, "eng") || strcmp(vw2_rec_get(r, "eng"), "stride"))
        return 0;
    memset(e, 0, sizeof *e);
    e->N = N; e->K = K;
    e->mode = vw2__stride_mode_idx(vw2_rec_get(r, "mode"));
    if (e->mode <= 0) return 0;
    if (vw2_rec_get(r, "chain")) {
        int nv;
        e->nf = vw2__stride_split_ints(vw2_rec_get(r, "chain"), e->factors,
                                       STRIDE_MAX_STAGES);
        if (e->nf <= 0 || e->nf >= STRIDE_MAX_STAGES) return 0;
        nv = vw2__stride_split_vars(vw2_rec_get(r, "vars"), e->variants,
                                    STRIDE_MAX_STAGES);
        if (nv != e->nf) return 0;
    } else if (vw2_rec_get(r, "ref")) {
        /* signpost row: reconstruct the legacy dummy chain (nf=1,
         * factors[0]=N, flat) — deterministic, reader-gate-checkable */
        e->nf = 1;
        e->factors[0] = N;
        e->variants[0] = 0;
    } else if (ord == VW2_ORD_SCR) {
        /* BARE mode row (ord=scr only): a self-contained verdict — the
         * prime/no-chain cells whose engine rebuilds from N alone. @nat
         * rows stay strict below. */
        e->nf = 1;
        e->factors[0] = N;
        e->variants[0] = 0;
    } else {
        return 0;                              /* a nat row needs one or the other */
    }
    e->use_dif = vw2__stride_geti(r, "dif", 0);
    {
        const char *ns = vw2_rec_get(r, "ns");
        e->nat_ns = ns ? atof(ns) : 0.0;
    }
    return 1;
}

static inline int vw2_stride_lookup_nat(const vw2_store_t *s, uint8_t lay,
                                        int N, size_t K,
                                        vfft_proto_nat_entry_t *e)
{
    return vw2__stride_lookup_natx(s, VW2_PL_IP, lay, VW2_ORD_NAT, N, K,
                                   e);
}

static inline int vw2_stride_lookup_natoop(const vw2_store_t *s, uint8_t lay,
                                           int N, size_t K,
                                           vfft_proto_nat_entry_t *e)
{
    return vw2__stride_lookup_natx(s, VW2_PL_OOP, lay, VW2_ORD_NAT, N, K,
                                   e);
}

/* the SCRAMBLED in-place mode cell — the ord=scr twin of @nat (2026-08-25,
 * the ILP-attach fix): a DEFAULT-order in-place IL create races ILP vs
 * ITS OWN convert incumbent and banks here (mode=ilp | mode=conv, the
 * banked loss). Its OWN key (ord differs from @nat) because the race
 * incumbents differ by order — verdicts CAN diverge. */
static inline int vw2_stride_lookup_scrmode(const vw2_store_t *s,
                                            uint8_t lay, int N, size_t K,
                                            vfft_proto_nat_entry_t *e)
{
    return vw2__stride_lookup_natx(s, VW2_PL_IP, lay, VW2_ORD_SCR, N, K,
                                   e);
}

/* ================================================================ WRITE */

static inline void vw2__stride_stamp_date(vw2_rec_t *r)
{
    char d[16];
    time_t t = time(NULL);
    struct tm *tm = localtime(&t);
    if (tm && strftime(d, sizeof d, "%Y-%m-%d", tm))
        vw2_rec_set(r, 2, "date", d);
}

#define VW2__SB_SET(sect, n, v) do { \
    if (vw2_rec_set(r, sect, n, v) != VW2_OK) { vw2_rec_free(r); *why = "token-refused"; return -1; } \
} while (0)

static inline int vw2__stride_emit_chain(vw2_rec_t *r, int nf,
                                         const int *factors, const int *variants,
                                         const char **why)
{
    char buf[192];
    size_t off;
    int i;
    if (nf < 1 || nf >= STRIDE_MAX_STAGES) { vw2_rec_free(r); *why = "chain-nf-out-of-range"; return -1; }
    for (i = 0, off = 0; i < nf; i++) {
        int rr = snprintf(buf + off, sizeof buf - off, "%s%d", i ? "." : "", factors[i]);
        if (rr < 0 || (size_t)rr >= sizeof buf - off) { vw2_rec_free(r); *why = "chain-overflow"; return -1; }
        off += (size_t)rr;
    }
    VW2__SB_SET(1, "chain", buf);
    for (i = 0, off = 0; i < nf; i++) {
        int rr;
        if (variants[i] < 0 || variants[i] > 3) { vw2_rec_free(r); *why = "garbage-variant-token"; return -1; }
        rr = snprintf(buf + off, sizeof buf - off, "%s%s", i ? "." : "",
                      vw2_stride_var_name[variants[i]]);
        if (rr < 0 || (size_t)rr >= sizeof buf - off) { vw2_rec_free(r); *why = "chain-overflow"; return -1; }
        off += (size_t)rr;
    }
    VW2__SB_SET(1, "vars", buf);
    return 0;
}

static inline int vw2__stride_tail(vw2_rec_t *r, size_t ran, double ns,
                                   const char *src, const char *from,
                                   const char **why)
{
    char b[32];
    snprintf(b, sizeof b, "%zu", ran);
    VW2__SB_SET(2, "ran", b);
    if (ns > 0.0) {
        snprintf(b, sizeof b, "%.2f", ns);
        VW2__SB_SET(2, "ns", b);
        VW2__SB_SET(2, "metric", "fwd1");
        VW2__SB_SET(2, "units", "ns");
    }
    VW2__SB_SET(2, "src", src);
    if (from) VW2__SB_SET(2, "from", from);
    else vw2__stride_stamp_date(r);
    return 0;
}

/* scrambled entry -> record, for ANY transform tag. For a TRIG tag the
 * caller passes the OUTER size in key_n (the entry itself describes the
 * inner plan); for c2c/r2c key_n is just e->N. */
static inline int vw2_stride_rec_from_entry_t(vw2_rec_t *r,
                                              const vfft_proto_wisdom_entry_t *e,
                                              int t, int key_n,
                                              const char *src, const char *from,
                                              const char **why)
{
    char b[32];
    *why = NULL;
    memset(r, 0, sizeof *r);
    /* the shipped files carry N=0/N=2,K=0 junk rows (wave-0 census) —
     * unservable garbage, refused here so migration quarantines them and
     * a fresh bank can never create the class */
    if (e->N < 2 || e->K < 1 || key_n < 2) { *why = "junk-cell"; return -1; }
    /* a trig key must describe THIS entry: the derived inner size has to
     * match the plan we are banking, or the pair is incoherent */
    {
        int inner = vw2_stride_trig_inner_n(t, key_n);
        if (inner && inner != e->N) { *why = "trig-inner-size-mismatch"; return -1; }
    }
    vw2__stride_key(&r->key, t, key_n, e->K, VW2_ORD_SCR, VW2_PL_IP);
    VW2__SB_SET(1, "eng", "stride");
    if (vw2__stride_emit_chain(r, e->nf, e->factors, e->variants, why)) return -1;
    VW2__SB_SET(1, "dif", e->use_dif_forward ? "1" : "0");
    if (e->use_blocked) {
        VW2__SB_SET(1, "blocked", "1");
        snprintf(b, sizeof b, "%d", e->split_stage);
        VW2__SB_SET(1, "bsplit", b);
        snprintf(b, sizeof b, "%d", e->block_groups);
        VW2__SB_SET(1, "bgroups", b);
    }
    /* pad-vs-tail verdicts. The NUMBER is what executes (K = run as-is with
     * the narrow lane filler; Kp = pad up so every lane is full width). The
     * `_arm` token names the winner in words, so a banked line reads back as
     * a verdict instead of a number you have to decode against q= — INFO
     * class, derived, never decision-load-bearing. */
    if (e->exec_me) {
        snprintf(b, sizeof b, "%d", e->exec_me);
        VW2__SB_SET(1, "pad_me", b);
        VW2__SB_SET(2, "pad_arm", e->exec_me == (int)e->K ? "tail" : "pad");
    }
    if (e->il_me) {
        snprintf(b, sizeof b, "%d", e->il_me);
        VW2__SB_SET(1, "il_me", b);
        VW2__SB_SET(2, "il_arm", e->il_me == (int)e->K ? "tail" : "pad");
    }
    return vw2__stride_tail(r, e->K, e->best_ns, src, from, why);
}

static inline int vw2_stride_rec_from_entry(vw2_rec_t *r,
                                            const vfft_proto_wisdom_entry_t *e,
                                            int is_rfft,
                                            const char *src, const char *from,
                                            const char **why)
{
    return vw2_stride_rec_from_entry_t(r, e, is_rfft ? VW2_T_R2C : VW2_T_C2C,
                                       e->N, src, from, why);
}

/* natural entry -> record. pl selects @nat (ip) vs @natoop (oop); lay is
 * the CALLER LAYOUT the verdict serves — fresh banks stamp it concrete, so
 * a bank never lands on (and never erases) the other layout's cell or the
 * pre-1.2 lay-less row. Migration keeps writing lay-less vintage. */
static inline int vw2_stride_rec_from_nat(vw2_rec_t *r,
                                          const vfft_proto_nat_entry_t *e,
                                          int pl, uint8_t lay, int ord,
                                          const char *src, const char *from,
                                          const char **why)
{
    *why = NULL;
    memset(r, 0, sizeof *r);
    if (e->N < 2 || e->K < 1) { *why = "junk-cell"; return -1; }
    if (e->mode <= 0 || e->mode >= 9) { *why = "unknown-nat-mode"; return -1; }
    vw2__stride_key(&r->key, VW2_T_C2C, e->N, e->K, ord, pl);
    r->key.lay = lay;
    VW2__SB_SET(1, "eng", "stride");
    VW2__SB_SET(1, "mode", vw2_stride_mode_name[e->mode]);
    if (e->nf == 1 && e->factors[0] == e->N && ord == VW2_ORD_NAT) {
        /* the dummy-chain placeholder: SIGNPOST instead (owner #7) — @nat
         * rows only, where the referenced kind-4/oop cell exists; ref_ok
         * keeps the verdict honest if its recipe vanishes. */
        char refbuf[96];
        snprintf(refbuf, sizeof refbuf,
                 "cell(t=c2c,n=%d,q=1,ord=scr,place=oop)", e->N);
        VW2__SB_SET(1, "ref", refbuf);
    } else if (e->nf == 0 || (e->nf == 1 && e->factors[0] == e->N)) {
        /* ord=scr MODE cell with no real chain (prime / Rader nf=0): the
         * verdict is SELF-CONTAINED (the engine rebuilds from N alone) —
         * emit NEITHER chain nor ref. A dangling signpost here made the
         * row invisible forever (ref_ok filtered it; the prime cell has
         * no oop cascade row to point at) — caught 2026-08-25. */
    } else {
        if (vw2__stride_emit_chain(r, e->nf, e->factors, e->variants, why)) return -1;
    }
    VW2__SB_SET(1, "dif", e->use_dif ? "1" : "0");
    return vw2__stride_tail(r, e->K, e->nat_ns, src, from, why);
}

#undef VW2__SB_SET

/* ================================================================ BANK
 * Memory-only; persistence = the caller's guarded vw2_save. */

static inline int vw2__stride_bank(vw2_store_t *st, vw2_rec_t *rec)
{
    int rc = vw2_bank(st, rec);
    if (rc != VW2_OK) vw2_rec_free(rec);
    return rc;
}

static inline int vw2_stride_bank_entry_t(vw2_store_t *st,
                                          const vfft_proto_wisdom_entry_t *e,
                                          int t, int key_n)
{
    vw2_rec_t rec;
    const char *why = NULL;
    if (vw2_stride_rec_from_entry_t(&rec, e, t, key_n, "race", NULL, &why)) {
        fprintf(stderr, "[wisdom2] stride bank refused (%s)\n", why ? why : "?");
        return -1;
    }
    return vw2__stride_bank(st, &rec);
}

static inline int vw2_stride_bank_entry(vw2_store_t *st,
                                        const vfft_proto_wisdom_entry_t *e,
                                        int is_rfft)
{
    return vw2_stride_bank_entry_t(st, e, is_rfft ? VW2_T_R2C : VW2_T_C2C, e->N);
}

static inline int vw2_stride_bank_nat(vw2_store_t *st,
                                      const vfft_proto_nat_entry_t *e,
                                      int is_oop, uint8_t lay)
{
    vw2_rec_t rec;
    const char *why = NULL;
    if (vw2_stride_rec_from_nat(&rec, e, is_oop ? VW2_PL_OOP : VW2_PL_IP,
                                lay, VW2_ORD_NAT, "race", NULL, &why)) {
        fprintf(stderr, "[wisdom2] stride nat bank refused (%s)\n", why ? why : "?");
        return -1;
    }
    return vw2__stride_bank(st, &rec);
}

/* migration alias: legacy @nat lines are ORD_NAT by definition */
static inline int vw2_stride_rec_from_nat_mig(vw2_rec_t *r,
                                              const vfft_proto_nat_entry_t *e,
                                              int pl, uint8_t lay,
                                              const char *src,
                                              const char *from,
                                              const char **why)
{
    return vw2_stride_rec_from_nat(r, e, pl, lay, VW2_ORD_NAT, src, from,
                                   why);
}

static inline int vw2_stride_bank_scrmode(vw2_store_t *st,
                                          const vfft_proto_nat_entry_t *e,
                                          uint8_t lay)
{
    vw2_rec_t rec;
    const char *why = NULL;
    if (vw2_stride_rec_from_nat(&rec, e, VW2_PL_IP, lay, VW2_ORD_SCR,
                                "race", NULL, &why)) {
        fprintf(stderr, "[wisdom2] stride scrmode bank refused (%s)\n",
                why ? why : "?");
        return -1;
    }
    return vw2__stride_bank(st, &rec);
}

#endif /* VFFT_WISDOM2_STRIDE_READER_H */
