/* oop_wisdom.h — OOP c2c wisdom: the 2-axis decision {kind, factorization}
 * per (N, K), persisted and looked up.
 *
 * Why a SEPARATE file (not the c2c spike_wisdom): an OOP entry must encode BOTH
 * axes — the execution KIND (LEAF/BAILEY2/MODEB, axis 1) and its FACTORIZATION
 * (axis 2: BAILEY2 pair, or MODEB multi-factor). The c2c wisdom format is
 * MODEB-shaped only (factors+variants) and is shared with the in-place path
 * (different optima). So OOP gets its own store, mirroring rfft/c2r/c2c.
 *
 * File format (v2) — one entry per line, '#' comments and blanks ignored:
 *     N K kind [params...] ns
 *   kind 0 = LEAF    :  (no params)                     e.g.  64 512 0 117350.0
 *   kind 1 = BAILEY2 :  R1 R2 t1p                       e.g.  1024 120 1 32 32 1 185550.0
 *   kind 2 = MODEB   :  nf f0..f(nf-1) v0..v(nf-1)      e.g.  1024 256 2 5 4 4 4 4 4 0 2 2 2 2 502460.0
 *   t1p = BAILEY2 s2 twiddle variant (0=flat 1=log3).
 *   v0..v(nf-1) = MODEB per-stage twiddle variant (0=FLAT 1=LOG3 2=T1S).
 *   ns = measured wall time (informational; the dispatcher ignores it).
 *
 * v2 (this format) PERSISTS per-stage variants: MODEB is built from the in-place
 * c2c wisdom's variant-rich (factors, variants) — FLAT/T1S/LOG3 mixed per stage,
 * matching the in-place path — and BAILEY2 stores the tuner's flat-vs-log3 pick.
 * (v1 dropped variants and rebuilt all-T1S; that left per-stage tuning on the
 * table relative to the in-place engine.) MODEB is DIT-only (OOP stage 0 must be
 * untwiddled), so DIF-preferring in-place cells fall back to native LEAF/BAILEY2.
 *
 * Lifecycle: offline calibrator (vfft_oop_plan_create_dp_best) writes this file;
 * runtime vfft_oop_plan_create_wisdom() does a pure lookup + build, no measure.
 */
#ifndef VFFT_OOP_WISDOM_H
#define VFFT_OOP_WISDOM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "oop_plan.h"   /* kinds, vfft_oop_plan_t, leaf/pair create, proto_plan_create */

#ifndef VFFT_OOP_WISDOM_MAX
#define VFFT_OOP_WISDOM_MAX 1024
#endif

typedef struct {
    int    N;
    size_t K;
    int    kind;                          /* VFFT_OOP_KIND_{LEAF,BAILEY2,MODEB,BAILEY2V} */
    int    R1, R2;                        /* BAILEY2 pair; K1: the SPLIT-axis pair */
    int    t1p_variant;                   /* BAILEY2 s2: 0=flat 1=log3 */
    int    nf;                            /* MODEB */
    int    factors[STRIDE_MAX_STAGES];    /* MODEB */
    int    variants[STRIDE_MAX_STAGES];   /* MODEB per-stage 0=FLAT 1=LOG3 2=T1S */
    /* kind 3 (BAILEY2V / K=1 engine): per-axis verdicts. Line format:
     *   N 1 3 sp_route sp_R1 sp_R2 il_route il_R1 il_R2 ns
     * routes = VFFT_K1_SP_* / VFFT_K1_IL_* (oop_plan.h). One entry carries
     * BOTH axes because the buffer layout is an EXECUTE-time contract
     * (sim==dim==NULL => interleaved), unknown at plan create. */
    int    k1_sp_route, k1_il_route, il_R1, il_R2;
    /* kind 3, sp_route == VFFT_K1_SP_CCOL only: encoded column chain
     * (vfft_k1_cc_chain_encode; one extra token before ns on the line).
     * kind 4 (ZSPLIT) reuses cc_chain for the cascade chain. */
    int    cc_chain;
    /* kind 4, tcut WIDTH axis (2026-08-02). OPTIONAL trailing pair AFTER
     * "zs_route zt_t2q":
     *   N 1 4 zs_t2q cc_chain ns [zs_route zt_t2q [zt_tw zt_l1]]
     *
     * 🔴 zt_tw == 0 MEANS UNTILED, and that is the whole back-compat story:
     * every line banked before this axis existed parses with zt_tw = 0 and
     * therefore replays as exactly today's untiled driver. There is no
     * sentinel to forget.
     *
     * zt_tw = tile width in COMPLEX POINTS (not bytes, not a cut index — the
     * width is the input and the cut is derived from it, see zturn.h).
     *
     * 🔴 zt_l1 = the L1 DATA CACHE SIZE IN BYTES the width was tuned against,
     * and it is not decoration. This is the first CACHE-OCCUPANCY quantity the
     * library banks. A chain or a radix is a property of the transform and
     * ports anywhere; a width is a property of one machine's L1, and on the
     * wrong machine it fails as a mild slowdown rather than an error — the
     * worst thing to inherit silently. Replay compares it
     * (vfft_cpu_l1d_matches) and falls back to UNTILED on a mismatch rather
     * than using a width tuned for a cache that isn't there. Relevant today:
     * this CPU is hybrid, P-core L1d 48 KB vs E-core 32 KB. */
    int    zt_tw;
    int    zt_l1;
    /* kind 4 (ZSPLIT / K=1 SCRAMBLED cascade): measured fwd terminator pick,
     * 0 = sterm (single-quad), 1 = sterm2 (2-quad unroll-and-jam). Line:
     *   N 1 4 zs_t2q cc_chain ns [zs_route zt_t2q]
     * The pick is placement-order-sensitive (§4.9993), so it is measured on
     * the installed binary by the create-time race, never hand-set.
     *
     * ROUTE AXIS (Phase 5 tranche 2, cascade_load_path_restructure §6.4) —
     * APPEND-style extension AFTER ns, so every pre-route banked line parses
     * unchanged as route 0 = legacy zsplit (ns was already the last, optional
     * token; the writer emits the trailing pair only for route!=0, keeping
     * legacy lines byte-identical):
     *   zs_route: 0 = legacy zsplit cascade, 1 = ZTURN-S (zturn.h).
     *   zt_t2q:   zturn fwd terminator pick, 0 = stf, 1 = stf2 (the stf/stf2
     *             analog of zs_t2q; measured by the same create race).
     * A route line's ns is the route race's JOINT fwd+bwd median (the route
     * verdict is joint by cutover atomicity); a legacy line's ns stays the
     * fwd-only t2q-race median (informational either way). */
    int    zs_t2q;
    int    zs_route, zt_t2q;
    double ns;                            /* measured (informational) */
} vfft_oop_wisdom_entry_t;

typedef struct {
    vfft_oop_wisdom_entry_t e[VFFT_OOP_WISDOM_MAX];
    int count;
} vfft_oop_wisdom_t;

/* Load. Returns 0 on success, -1 if the file can't be opened. */
static inline int vfft_oop_wisdom_load(vfft_oop_wisdom_t *w, const char *path)
{
    FILE *f = fopen(path, "r");
    if (!f) return -1;
    w->count = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *s = line;
        while (*s == ' ' || *s == '\t') s++;
        if (*s == '#' || *s == '\n' || *s == '\r' || *s == '\0') continue;
        if (w->count >= VFFT_OOP_WISDOM_MAX) break;
        vfft_oop_wisdom_entry_t *e = &w->e[w->count];
        memset(e, 0, sizeof *e);
        char *tok = strtok(s, " \t\n\r");      if (!tok) continue; e->N = atoi(tok);
        tok = strtok(NULL, " \t\n\r");          if (!tok) continue; e->K = (size_t)strtoull(tok, NULL, 10);
        tok = strtok(NULL, " \t\n\r");          if (!tok) continue; e->kind = atoi(tok);
        int ok = 1;
        if (e->kind == VFFT_OOP_KIND_BAILEY2) {
            tok = strtok(NULL, " \t\n\r"); if (tok) e->R1 = atoi(tok); else ok = 0;
            tok = strtok(NULL, " \t\n\r"); if (tok) e->R2 = atoi(tok); else ok = 0;
            tok = strtok(NULL, " \t\n\r"); if (tok) e->t1p_variant = atoi(tok); else ok = 0;
        } else if (e->kind == VFFT_OOP_KIND_MODEB) {
            tok = strtok(NULL, " \t\n\r"); if (tok) e->nf = atoi(tok); else ok = 0;
            if (ok && (e->nf <= 0 || e->nf > STRIDE_MAX_STAGES)) ok = 0;
            for (int i = 0; ok && i < e->nf; i++) {
                tok = strtok(NULL, " \t\n\r");
                if (tok) e->factors[i] = atoi(tok); else ok = 0;
            }
            /* v2: per-stage variant codes follow the factors */
            for (int i = 0; ok && i < e->nf; i++) {
                tok = strtok(NULL, " \t\n\r");
                if (tok) e->variants[i] = atoi(tok); else ok = 0;
            }
        } else if (e->kind == VFFT_OOP_KIND_BAILEY2V) {
            /* kind 3 = K=1 engine: sp_route sp_R1 sp_R2 il_route il_R1 il_R2 */
            tok = strtok(NULL, " \t\n\r"); if (tok) e->k1_sp_route = atoi(tok); else ok = 0;
            tok = strtok(NULL, " \t\n\r"); if (tok) e->R1 = atoi(tok); else ok = 0;
            tok = strtok(NULL, " \t\n\r"); if (tok) e->R2 = atoi(tok); else ok = 0;
            tok = strtok(NULL, " \t\n\r"); if (tok) e->k1_il_route = atoi(tok); else ok = 0;
            tok = strtok(NULL, " \t\n\r"); if (tok) e->il_R1 = atoi(tok); else ok = 0;
            tok = strtok(NULL, " \t\n\r"); if (tok) e->il_R2 = atoi(tok); else ok = 0;
            /* CCOL lines carry the encoded column chain before ns */
            if (ok && e->k1_sp_route == VFFT_K1_SP_CCOL) {
                tok = strtok(NULL, " \t\n\r");
                if (tok) e->cc_chain = atoi(tok); else ok = 0;
            }
            if (ok && e->K != 1) ok = 0;
        } else if (e->kind == VFFT_OOP_KIND_ZSPLIT) {
            /* kind 4 = K=1 SCRAMBLED cascade: zs_t2q cc_chain */
            tok = strtok(NULL, " \t\n\r"); if (tok) e->zs_t2q = atoi(tok); else ok = 0;
            tok = strtok(NULL, " \t\n\r"); if (tok) e->cc_chain = atoi(tok); else ok = 0;
            if (ok && e->K != 1) ok = 0;
        }
        if (!ok) continue;
        tok = strtok(NULL, " \t\n\r");          /* ns (optional) */
        e->ns = tok ? atof(tok) : 0.0;
        /* kind-4 route axis: OPTIONAL trailing "zs_route zt_t2q" after ns.
         * Old-format lines end at ns -> both stay 0 = legacy zsplit route
         * with the stf pick — backward compatible by construction. */
        if (e->kind == VFFT_OOP_KIND_ZSPLIT && tok) {
            tok = strtok(NULL, " \t\n\r");
            if (tok) {
                e->zs_route = atoi(tok);
                tok = strtok(NULL, " \t\n\r");
                e->zt_t2q = tok ? atoi(tok) : 0;
                /* tcut width axis: OPTIONAL pair after zt_t2q. Absent -> both
                 * stay 0 -> zt_tw == 0 -> UNTILED, which is what every line
                 * banked before this axis existed must keep meaning. */
                if (tok) {
                    tok = strtok(NULL, " \t\n\r");
                    if (tok) {
                        e->zt_tw = atoi(tok);
                        tok = strtok(NULL, " \t\n\r");
                        e->zt_l1 = tok ? atoi(tok) : 0;
                    }
                }
            }
        }
        w->count++;
    }
    fclose(f);
    return 0;
}

static inline const vfft_oop_wisdom_entry_t *
vfft_oop_wisdom_lookup(const vfft_oop_wisdom_t *w, int N, size_t K)
{
    if (!w) return NULL;
    for (int i = 0; i < w->count; i++)
        if (w->e[i].N == N && w->e[i].K == K)
            return &w->e[i];
    return NULL;
}

/* K=1 engine lookup: the kind-3 entry for N (K==1 by definition). */
static inline const vfft_oop_wisdom_entry_t *
vfft_oop_wisdom_lookup_k1(const vfft_oop_wisdom_t *w, int N)
{
    if (!w) return NULL;
    for (int i = 0; i < w->count; i++)
        if (w->e[i].N == N && w->e[i].K == 1 &&
            w->e[i].kind == VFFT_OOP_KIND_BAILEY2V)
            return &w->e[i];
    return NULL;
}

/* K=1 SCRAMBLED cascade lookup: the kind-4 entry for N (K==1 by definition). */
static inline const vfft_oop_wisdom_entry_t *
vfft_oop_wisdom_lookup_zsplit(const vfft_oop_wisdom_t *w, int N)
{
    if (!w) return NULL;
    for (int i = 0; i < w->count; i++)
        if (w->e[i].N == N && w->e[i].K == 1 &&
            w->e[i].kind == VFFT_OOP_KIND_ZSPLIT)
            return &w->e[i];
    return NULL;
}

/* Output-order class of an OOP kind: 1 = NATURAL (LEAF/BAILEY2), 0 = SCRAMBLED (MODEB). This is
 * what lets one (N,K) cell cache both a natural and a scrambled champion as SEPARATE entries. */
static inline int vfft_oop_kind_natural(int kind) { return kind != VFFT_OOP_KIND_MODEB; }

/* Order-aware lookup (config.order): ord 1=NATURAL (LEAF/BAILEY2), 2=SCRAMBLED (MODEB), 0=DEFAULT
 * (the faster of whatever is cached, by ns). Returns the matching-class entry, or the min-ns entry
 * for DEFAULT, or NULL. With per-(N,K,class) persistence a cell holds up to two entries, so an order
 * request is served from wisdom instead of re-tuning. Values match include/vfft.h VFFT_ORDER_*. */
static inline const vfft_oop_wisdom_entry_t *
vfft_oop_wisdom_lookup_ord(const vfft_oop_wisdom_t *w, int N, size_t K, int ord)
{
    if (!w) return NULL;
    const vfft_oop_wisdom_entry_t *best = NULL;
    for (int i = 0; i < w->count; i++) {
        const vfft_oop_wisdom_entry_t *e = &w->e[i];
        if (e->N != N || e->K != K) continue;
        if (e->kind == VFFT_OOP_KIND_BAILEY2V) continue; /* K1 entries: lookup_k1 only */
        if (e->kind == VFFT_OOP_KIND_ZSPLIT) continue;   /* cascade cells: lookup_zsplit only */
        int nat = vfft_oop_kind_natural(e->kind);
        if (ord == 1 && !nat) continue;                  /* NATURAL wanted; skip MODEB */
        if (ord == 2 && nat)  continue;                  /* SCRAMBLED wanted; skip native */
        if (!best || e->ns < best->ns) best = e;         /* ord 0: fastest; ord 1/2: the sole class */
    }
    return best;
}

/* Build the exact plan a wisdom entry names — PURE LOOKUP, no measurement.
 * Returns NULL if (N,K) has no entry or the entry's codelets are unavailable
 * (caller then falls back to the rule spine / DP). */
static inline vfft_oop_plan_t *
vfft_oop_plan_create_wisdom(int N, size_t K, const vfft_oop_wisdom_t *w,
                            const vfft_proto_registry_t *reg)
{
    const vfft_oop_wisdom_entry_t *e = vfft_oop_wisdom_lookup(w, N, K);
    if (!e) return NULL;
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
        /* validates pair + mask; t1p variant is the tuner's persisted pick */
        return vfft_oop_plan_create_pair_v(N, K, e->R1, e->R2, e->t1p_variant);
    if (e->kind == VFFT_OOP_KIND_MODEB)
        /* v2: rebuild with the PERSISTED per-stage variants (the variant-rich
         * mix inherited from the in-place c2c wisdom), not all-T1S. Helper owns
         * construction + inner-plan teardown on failure. */
        return _vfft_oop_make_modeb(N, K, e->factors, e->variants, e->nf, reg);
    return NULL;
}

/* Build the plan a SPECIFIC entry names (the order-aware path uses lookup_ord then this, so it can
 * pick the natural or the MODEB champion of a two-entry cell). Same build as create_wisdom's tail. */
static inline vfft_oop_plan_t *
vfft_oop_plan_from_entry(const vfft_oop_wisdom_entry_t *e, const vfft_proto_registry_t *reg)
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
static inline void vfft_oop_wisdom_entry_from_plan(vfft_oop_wisdom_entry_t *e,
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

static inline void vfft_oop_wisdom_write_entry(FILE *f,
                                               const vfft_oop_wisdom_entry_t *e)
{
    fprintf(f, "%d %zu %d", e->N, e->K, e->kind);
    if (e->kind == VFFT_OOP_KIND_BAILEY2)
        fprintf(f, " %d %d %d", e->R1, e->R2, e->t1p_variant);
    else if (e->kind == VFFT_OOP_KIND_MODEB) {
        fprintf(f, " %d", e->nf);
        for (int s = 0; s < e->nf; s++) fprintf(f, " %d", e->factors[s]);
        for (int s = 0; s < e->nf; s++) fprintf(f, " %d", e->variants[s]);
    }
    else if (e->kind == VFFT_OOP_KIND_BAILEY2V) {
        fprintf(f, " %d %d %d %d %d %d", e->k1_sp_route, e->R1, e->R2,
                e->k1_il_route, e->il_R1, e->il_R2);
        if (e->k1_sp_route == VFFT_K1_SP_CCOL)
            fprintf(f, " %d", e->cc_chain);
    }
    else if (e->kind == VFFT_OOP_KIND_ZSPLIT)
        fprintf(f, " %d %d", e->zs_t2q, e->cc_chain);
    fprintf(f, " %.1f", e->ns);
    /* kind-4 route axis: trailing "zs_route zt_t2q" AFTER ns, and only for
     * route!=0 — a legacy verdict re-banks in the exact pre-route format
     * (old readers, diffs, and banked files stay byte-stable). */
    if (e->kind == VFFT_OOP_KIND_ZSPLIT && e->zs_route) {
        fprintf(f, " %d %d", e->zs_route, e->zt_t2q);
        /* Width pair only when a width was actually banked, so a verdict that
         * did not use tiling re-banks byte-identically to the pre-width format.
         * The width is a ZTURN-only concept, so it can never appear without
         * zs_route — that is why it nests inside this branch rather than
         * beside it. */
        if (e->zt_tw > 0)
            fprintf(f, " %d %d", e->zt_tw, e->zt_l1);
    }
    fprintf(f, "\n");
}

#endif /* VFFT_OOP_WISDOM_H */
