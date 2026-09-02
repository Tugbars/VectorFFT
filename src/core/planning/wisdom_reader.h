/* wisdom_reader.h — parse production's wisdom file.
 *
 * Format: header lines (starting with @ or #) followed by entries:
 *
 *   N K nf factor_1 ... factor_nf best_ns use_blocked split_stage \
 *     block_groups use_dif_forward variant_1 ... variant_nf
 *
 * Variant codes: 0=FLAT, 1=LOG3, 2=T1S, 3=BUF (unused in current wisdom).
 *
 * In-memory table with linear (N, K) lookup. Provides BOTH read and write:
 * load() + lookup() consume wisdom; set() + save() produce it, so the
 * dag-fft-compiler core can close the loop itself (calibrator: search a
 * cell -> fill an entry -> set() -> ... -> save() -> regen plan_executors.h).
 * save() round-trips with load(). Ported from production src/core/planner.h
 * (stride_wisdom_load / stride_wisdom_save), standalone (no src/core/ include).
 */
#ifndef VFFT_PROTO_CORE_WISDOM_READER_H
#define VFFT_PROTO_CORE_WISDOM_READER_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include "plan.h"  /* STRIDE_MAX_STAGES */

typedef struct {
    int     N;
    size_t  K;
    int     nf;
    int     factors  [STRIDE_MAX_STAGES];
    int     variants [STRIDE_MAX_STAGES];   /* 0=FLAT 1=LOG3 2=T1S 3=BUF */
    int     use_dif_forward;
    int     use_blocked;
    int     split_stage;
    int     block_groups;
    double  best_ns;
    /* exec_me — the padded pad-vs-tail VERDICT for a misaligned-K cell (only ever read by
     * the padded dispatch; the tight path ignores it). Three states:
     *   0  = NOT pad-measured yet (default; also every aligned/tight cell) -> the padded
     *        planner runs its one-time A/B on a MISS, then stamps K or Kp here.
     *   K  = TAIL won: run me=K with the SSE2/scalar tail on the Kp buffer (this cell's factors).
     *   Kp = PAD  won: run me=Kp full-SIMD -> use the aligned (N,Kp) entry's plan instead.
     * Trailing v6 field; absent (v5) loads as 0. This keeps padding in the SINGLE c2c wisdom
     * file (no separate padded file). See docs/roadmap/tail_handling/padding_design_decision.md. */
    int     exec_me;
    /* il_me — the IL fused-vs-padded VERDICT for a misaligned-K cell (§6a59;
     * only read by the interleaved-z decision). Same three states as exec_me:
     * 0 = not IL-measured; K = fused won (tight folds + hybrid tails);
     * Kp = padded won (unfused, full-width interior at Kp). Trailing v7
     * field; absent (v5/v6) loads as 0. Stamped at the first-execute A/B,
     * persists with the bundle save. */
    int     il_me;
} vfft_proto_wisdom_entry_t;

enum { VFFT_NAT_UNSET = 0, VFFT_NAT_FREE = 1, VFFT_NAT_LEAF_IP = 2,
       VFFT_NAT_SCR = 3, VFFT_NAT_PURE_CYCLE = 4, VFFT_NAT_PSWAP = 5,
       /* ZCASC (B5, 2026-08-03): the K=1 interleaved zturn cascade with the
        * stfn NATURAL terminator — no reorder pass at all. Raced end-to-end
        * against the tape incumbent at create (vfft.c natural block) and
        * banked like every other mode. Replay pulls the CHAIN from the
        * kind-4 oop line (the scrambled cascade verdict; order-agnostic
        * plan data) — the @nat entry stores only the VERDICT. An old binary
        * reading mode=6 falls into its MEASURE branch and re-races: degraded,
        * never wrong. 2 (LEAF_IP) is retired but NEVER reused — old files
        * may still carry it with the old meaning. */
       VFFT_NAT_ZCASC = 6,
       /* ILP (il_coverage_plan.md Phase B, 2026-08-03): the sub-2048 K=1
        * interleaved IN-PLACE cells served by the native IL engines
        * (il2p/il3p, alias-gated; mono structurally refuses aliasing) —
        * natural output, no tape, no layout conversion. Raced end-to-end
        * vs the convert incumbent at a NATURAL create only; an explicit-
        * SCRAMBLED in-place create attaches HIT-ONLY on this verdict
        * (identity permutation — same contract note as Phase A; hit-only
        * keeps @nat single-writer). Old binaries re-measure, never wrong. */
       VFFT_NAT_ILP = 7,
       /* CONV (2026-08-25): the banked LOSS of the scrambled in-place IL
        * race — "raced, the convert incumbent won" — in the ord=scr mode
        * cell only (the @nat natural cells never carry it). Exists so a
        * losing race is not re-run on every create (the kind-3 IL_NONE
        * law). Old binaries: unknown mode -> re-measure, never wrong. */
       VFFT_NAT_CONV = 8 };

/* ── SELF-CONTAINED natural-order record (order=VFFT_ORDER_NATURAL). Its own DEPLOYED FFT
 * chain (nf/factors/variants/use_dif — the plan the reorder tape follows and that runs
 * forward) + reorder mode + measured natural total. Keyed (N,K) in a SEPARATE table, loaded
 * from the SAME file via `@nat`-tagged lines (invisible to every external @/#-skipping
 * reader — OCaml codegen, python, bootstrap.sh). Natural create/consume reads ONLY this,
 * NEVER the scrambled entry. The old opportunistic-vs-injected PSWAP distinction is gone: a
 * record just stores the deployed chain + mode=PSWAP. Design pivot 2026-07-06 (scrambled and
 * natural are different objectives + different memory-pass counts). */
typedef struct {
    int     N;
    size_t  K;
    int     mode;                            /* VFFT_NAT_FREE/SCR/PURE_CYCLE/PSWAP (LEAF_IP ditched) */
    int     nf;
    int     factors [STRIDE_MAX_STAGES];     /* the deployed natural chain */
    int     variants[STRIDE_MAX_STAGES];
    int     use_dif;
    double  nat_ns;                          /* measured natural total (margin/info) */
    int     raced;                           /* zr=1: the ZCASC/ILP challenger raced
                                              * this cell and LOST — do not rebuild the
                                              * candidate or re-race (the banked-loss
                                              * law, VFFT_NAT_CONV's comment). Absent
                                              * on old lines = 0 = race once and mark.
                                              * wisdom2-only; the legacy reader never
                                              * sets it (callers zero it on that path). */
    int     ref_ilp;                         /* mode=ilp bank: which recipe row the signpost
                                              * names — 0 none (mono), 1 the kind-3 row
                                              * lay=il, 2 kind-3 lay=split, 3 kind-3 lay-less,
                                              * 4 the PRIME shard row (2026-09-02). */
    int     ref_comp;                        /* mode=zcasc bank: 1 = the signpost names
                                              * the role=comp kind-4 RECIPE (banked by an
                                              * in-place / odd race); 0 = the OOP problem
                                              * verdict. The bank helpers set it from what
                                              * the store holds at bank time. */
} vfft_proto_nat_entry_t;

typedef struct {
    vfft_proto_wisdom_entry_t *entries;
    size_t                     count;
    size_t                     capacity;
    vfft_proto_nat_entry_t    *nat;          /* second table, SAME file (@nat lines) */
    size_t                     nat_count;
    size_t                     nat_capacity;
    /* Third table, SAME file (@natoop lines): the OOP-NATURAL verdict —
     * same entry shape and (N,K) key as @nat, SEPARATE table because the
     * two regimes have different incumbents (in-place: tape/ILP/ZCASC;
     * OOP: the K=1 engine handle vs the natord cascade) and a shared
     * (N,K) slot would make each regime's bank clobber the other's
     * (the @nat single-writer rule, extended per-placement). Unknown-@
     * lines are skipped by every shipped reader, so old binaries ignore
     * these and simply re-measure — never wrong. */
    vfft_proto_nat_entry_t    *natoop;
    size_t                     natoop_count;
    size_t                     natoop_capacity;
} vfft_proto_wisdom_t;

/* Load wisdom from path. Returns 0 on success, -1 on file-not-found or
 * parse error. On success, *wis owns its entries array; free with
 * vfft_proto_wisdom_free. */
static inline int vfft_proto_wisdom_load(vfft_proto_wisdom_t *wis,
                                         const char *path)
{
    memset(wis, 0, sizeof(*wis));
    FILE *f = fopen(path, "r");
    if (!f) return -1;

    char line[2048];
    while (fgets(line, sizeof(line), f)) {
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#') continue;
        if (*p == '@') {
            /* @nat = self-contained natural record (parsed into the SEPARATE nat table);
             * @natoop = the OOP-NATURAL verdict, same line shape, its own table. Every
             * other @ line (@version / headers) is skipped — exactly as external @/#-skipping
             * readers do, so @nat/@natoop lines never reach codegen/python/bootstrap. */
            char *nt = strtok(p, " \t\r\n");
            int is_nat = (nt && strcmp(nt, "@nat") == 0);
            int is_natoop = (nt && strcmp(nt, "@natoop") == 0);
            if (is_nat || is_natoop) {
                vfft_proto_nat_entry_t ne;
                memset(&ne, 0, sizeof(ne));
                char *t;
                t = strtok(NULL, " \t\r\n"); if (!t) continue; ne.N = atoi(t);
                t = strtok(NULL, " \t\r\n"); if (!t) continue; ne.K = (size_t)atoll(t);
                t = strtok(NULL, " \t\r\n"); if (!t) continue; ne.mode = atoi(t);
                t = strtok(NULL, " \t\r\n"); if (!t) continue; ne.nf = atoi(t);
                if (ne.nf <= 0 || ne.nf >= STRIDE_MAX_STAGES) continue;
                for (int i = 0; i < ne.nf; i++) { t = strtok(NULL, " \t\r\n"); if (!t) goto skip; ne.factors[i]  = atoi(t); }
                for (int i = 0; i < ne.nf; i++) { t = strtok(NULL, " \t\r\n"); if (!t) goto skip; ne.variants[i] = atoi(t); }
                t = strtok(NULL, " \t\r\n"); if (!t) continue; ne.use_dif = atoi(t);
                t = strtok(NULL, " \t\r\n"); ne.nat_ns = t ? atof(t) : 0.0;
                if (is_nat) {
                    if (wis->nat_count >= wis->nat_capacity) {
                        wis->nat_capacity = wis->nat_capacity ? wis->nat_capacity * 2 : 32;
                        wis->nat = realloc(wis->nat, wis->nat_capacity * sizeof(*wis->nat));
                    }
                    wis->nat[wis->nat_count++] = ne;
                } else {
                    if (wis->natoop_count >= wis->natoop_capacity) {
                        wis->natoop_capacity = wis->natoop_capacity ? wis->natoop_capacity * 2 : 32;
                        wis->natoop = realloc(wis->natoop, wis->natoop_capacity * sizeof(*wis->natoop));
                    }
                    wis->natoop[wis->natoop_count++] = ne;
                }
            }
            continue;
        }

        /* Parse: N K nf factors[nf] best_ns use_blocked split_stage \
         *        block_groups use_dif_forward variants[nf] */
        vfft_proto_wisdom_entry_t e;
        memset(&e, 0, sizeof(e));
        char *tok = strtok(p, " \t\r\n");
        if (!tok) continue;
        e.N = atoi(tok);
        tok = strtok(NULL, " \t\r\n"); if (!tok) continue;
        e.K = (size_t)atoll(tok);
        tok = strtok(NULL, " \t\r\n"); if (!tok) continue;
        e.nf = atoi(tok);
        if (e.nf <= 0 || e.nf >= STRIDE_MAX_STAGES) continue;
        for (int i = 0; i < e.nf; i++) {
            tok = strtok(NULL, " \t\r\n"); if (!tok) goto skip;
            e.factors[i] = atoi(tok);
        }
        tok = strtok(NULL, " \t\r\n"); if (!tok) continue;
        e.best_ns = atof(tok);
        tok = strtok(NULL, " \t\r\n"); if (!tok) continue;
        e.use_blocked = atoi(tok);
        tok = strtok(NULL, " \t\r\n"); if (!tok) continue;
        e.split_stage = atoi(tok);
        tok = strtok(NULL, " \t\r\n"); if (!tok) continue;
        e.block_groups = atoi(tok);
        tok = strtok(NULL, " \t\r\n"); if (!tok) continue;
        e.use_dif_forward = atoi(tok);
        for (int i = 0; i < e.nf; i++) {
            tok = strtok(NULL, " \t\r\n"); if (!tok) goto skip;
            e.variants[i] = atoi(tok);
        }
        /* Trailing v6 field: exec_me (padded verdict). Missing (v5 file) -> 0 = not
         * pad-measured. Old binaries stop tokenizing after the variants (forward compatible). */
        tok = strtok(NULL, " \t\r\n");
        e.exec_me = tok ? atoi(tok) : 0;
        tok = strtok(NULL, " \t");
        e.il_me = tok ? atoi(tok) : 0;   /* v7 trailing (absent -> 0) */
        /* Scrambled line ends at exec_me. Any trailing tokens (e.g. a stray embedded v7 nat block
         * from a disposable staging file) are IGNORED — natural verdicts live in @nat lines now. */

        /* Append. */
        if (wis->count >= wis->capacity) {
            wis->capacity = wis->capacity ? wis->capacity * 2 : 64;
            wis->entries = realloc(wis->entries,
                                   wis->capacity * sizeof(*wis->entries));
        }
        wis->entries[wis->count++] = e;
        continue;
    skip:
        continue;
    }
    fclose(f);
    return 0;
}

/* Look up a wisdom entry for (N, K). Returns NULL if not found. */
static inline const vfft_proto_wisdom_entry_t *
vfft_proto_wisdom_lookup(const vfft_proto_wisdom_t *wis,
                         int N, size_t K)
{
    for (size_t i = 0; i < wis->count; i++) {
        if (wis->entries[i].N == N && wis->entries[i].K == K)
            return &wis->entries[i];
    }
    return NULL;
}

/* Insert or replace the entry for (N, K). Returns 1 if a new entry was
 * appended, 0 if an existing (N,K) entry was overwritten. This is the
 * accumulate step a calibrator uses: search a cell -> fill an entry -> set().
 * (Pointers from a prior lookup() may be invalidated by the realloc here.) */
static inline int vfft_proto_wisdom_set(vfft_proto_wisdom_t *wis,
                                        const vfft_proto_wisdom_entry_t *e)
{
    for (size_t i = 0; i < wis->count; i++) {
        if (wis->entries[i].N == e->N && wis->entries[i].K == e->K) {
            wis->entries[i] = *e;
            return 0;
        }
    }
    if (wis->count >= wis->capacity) {
        wis->capacity = wis->capacity ? wis->capacity * 2 : 64;
        wis->entries = realloc(wis->entries,
                               wis->capacity * sizeof(*wis->entries));
    }
    wis->entries[wis->count++] = *e;
    return 1;
}

/* Calibrator / planner write primitive. Enforces the production invariant of
 * EXACTLY ONE entry per (N,K): the cell's winner is the sole entry — multiple
 * entries for one cell are not allowed. The `overwrite` flag decides what
 * happens when (N,K) is already present:
 *   overwrite == 0 : leave the existing cell untouched, return 0 (skip). Used by
 *                    incremental sweeps that only fill in missing cells and must
 *                    not clobber an already-calibrated result.
 *   overwrite != 0 : drop EVERY existing (N,K) entry (collapsing any stale
 *                    duplicates) and write `e` as the sole entry — "whatever won
 *                    now is the only entry". Returns 2.
 * When (N,K) is absent, `e` is appended in either mode (return 1). The collapse
 * is what reconciles any pre-existing multi-entry cells back to one-per-cell on
 * the first overwrite pass. */
static inline int vfft_proto_wisdom_add(vfft_proto_wisdom_t *wis,
                                        const vfft_proto_wisdom_entry_t *e,
                                        int overwrite)
{
    size_t matches = 0;
    for (size_t i = 0; i < wis->count; i++)
        if (wis->entries[i].N == e->N && wis->entries[i].K == e->K) matches++;

    if (matches > 0 && !overwrite) return 0;          /* keep existing cell */

    if (matches > 0) {                                /* collapse all (N,K) */
        size_t w = 0;
        for (size_t i = 0; i < wis->count; i++)
            if (!(wis->entries[i].N == e->N && wis->entries[i].K == e->K))
                wis->entries[w++] = wis->entries[i];
        wis->count = w;
    }
    if (wis->count >= wis->capacity) {
        wis->capacity = wis->capacity ? wis->capacity * 2 : 64;
        wis->entries = realloc(wis->entries,
                               wis->capacity * sizeof(*wis->entries));
    }
    wis->entries[wis->count++] = *e;
    return matches > 0 ? 2 : 1;
}

/* Write the table to path in the same v5 format vfft_proto_wisdom_load reads
 * (round-trips). Returns 0 on success, -1 on open failure. Ported from
 * production src/core/planner.h:stride_wisdom_save, adapted to this tree's
 * vfft_proto_wisdom_entry_t (which always carries variant codes, so no -1
 * placeholders are needed). */
static inline int vfft_proto_wisdom_save(const vfft_proto_wisdom_t *wis,
                                         const char *path)
{
    FILE *f = fopen(path, "w");
    if (!f) return -1;
    fprintf(f, "@version 8\n");
    fprintf(f, "# VectorFFT stride wisdom: %zu scrambled + %zu natural entries\n",
            wis->count, wis->nat_count);
    fprintf(f, "# scrambled: N K nf factors... best_ns use_blocked split_stage block_groups "
               "use_dif_forward variant_codes... exec_me (v=0:FLAT 1:LOG3 2:T1S 3:BUF)\n");
    fprintf(f, "# natural (self-contained, @nat-tagged -> invisible to @/#-skipping external "
               "readers): @nat N K mode nf factors... variants... use_dif nat_ns "
               "(mode 1=FREE 3=SCR 4=PURE_CYCLE 5=PSWAP)\n");
    for (size_t i = 0; i < wis->count; i++) {
        const vfft_proto_wisdom_entry_t *e = &wis->entries[i];
        fprintf(f, "%d %zu %d", e->N, e->K, e->nf);
        for (int j = 0; j < e->nf; j++)
            fprintf(f, " %d", e->factors[j]);
        fprintf(f, " %.2f %d %d %d %d", e->best_ns,
                e->use_blocked, e->split_stage, e->block_groups,
                e->use_dif_forward);
        for (int j = 0; j < e->nf; j++)
            fprintf(f, " %d", e->variants[j]);
        /* v6 trailing field: exec_me (padded verdict; 0 = not pad-measured). Scrambled line ENDS
         * here — natural verdicts are emitted below as @nat lines (regime-exclusive records). */
        fprintf(f, " %d %d\n", e->exec_me, e->il_me);   /* v6 + v7 trailing */
    }
    /* Natural table: one self-contained @nat line per entry. */
    for (size_t i = 0; i < wis->nat_count; i++) {
        const vfft_proto_nat_entry_t *n = &wis->nat[i];
        fprintf(f, "@nat %d %zu %d %d", n->N, n->K, n->mode, n->nf);
        for (int j = 0; j < n->nf; j++) fprintf(f, " %d", n->factors[j]);
        for (int j = 0; j < n->nf; j++) fprintf(f, " %d", n->variants[j]);
        fprintf(f, " %d %.2f\n", n->use_dif, n->nat_ns);
    }
    /* OOP-natural table: same line shape under the @natoop tag (skipped by
     * every pre-@natoop reader, which then just re-measures the verdict). */
    for (size_t i = 0; i < wis->natoop_count; i++) {
        const vfft_proto_nat_entry_t *n = &wis->natoop[i];
        fprintf(f, "@natoop %d %zu %d %d", n->N, n->K, n->mode, n->nf);
        for (int j = 0; j < n->nf; j++) fprintf(f, " %d", n->factors[j]);
        for (int j = 0; j < n->nf; j++) fprintf(f, " %d", n->variants[j]);
        fprintf(f, " %d %.2f\n", n->use_dif, n->nat_ns);
    }
    fclose(f);
    return 0;
}

/* ── Natural table (order=VFFT_ORDER_NATURAL) lookup/upsert — mirror of the scrambled
 * lookup/add but keyed (N,K) on the SEPARATE nat table. The natural create path uses ONLY
 * these; it never touches the scrambled entries. ── */
static inline const vfft_proto_nat_entry_t *
vfft_proto_nat_lookup(const vfft_proto_wisdom_t *wis, int N, size_t K)
{
    for (size_t i = 0; i < wis->nat_count; i++)
        if (wis->nat[i].N == N && wis->nat[i].K == K) return &wis->nat[i];
    return NULL;
}

/* One-entry-per-(N,K) upsert on the nat table. overwrite==0: keep existing (return 0);
 * else collapse all (N,K) and append e (return 2), or append if absent (return 1). */
static inline int vfft_proto_nat_add(vfft_proto_wisdom_t *wis,
                                     const vfft_proto_nat_entry_t *e, int overwrite)
{
    size_t matches = 0;
    for (size_t i = 0; i < wis->nat_count; i++)
        if (wis->nat[i].N == e->N && wis->nat[i].K == e->K) matches++;
    if (matches > 0 && !overwrite) return 0;
    if (matches > 0) {
        size_t w = 0;
        for (size_t i = 0; i < wis->nat_count; i++)
            if (!(wis->nat[i].N == e->N && wis->nat[i].K == e->K))
                wis->nat[w++] = wis->nat[i];
        wis->nat_count = w;
    }
    if (wis->nat_count >= wis->nat_capacity) {
        wis->nat_capacity = wis->nat_capacity ? wis->nat_capacity * 2 : 32;
        wis->nat = realloc(wis->nat, wis->nat_capacity * sizeof(*wis->nat));
    }
    wis->nat[wis->nat_count++] = *e;
    return matches > 0 ? 2 : 1;
}

/* ── OOP-natural table (order=NATURAL, placement=OUTOFPLACE) lookup/upsert —
 * the @nat pair re-keyed onto the natoop table. The OOP-natural create path
 * uses ONLY these (single writer per table, same as @nat). ── */
static inline const vfft_proto_nat_entry_t *
vfft_proto_natoop_lookup(const vfft_proto_wisdom_t *wis, int N, size_t K)
{
    for (size_t i = 0; i < wis->natoop_count; i++)
        if (wis->natoop[i].N == N && wis->natoop[i].K == K) return &wis->natoop[i];
    return NULL;
}

static inline int vfft_proto_natoop_add(vfft_proto_wisdom_t *wis,
                                        const vfft_proto_nat_entry_t *e, int overwrite)
{
    size_t matches = 0;
    for (size_t i = 0; i < wis->natoop_count; i++)
        if (wis->natoop[i].N == e->N && wis->natoop[i].K == e->K) matches++;
    if (matches > 0 && !overwrite) return 0;
    if (matches > 0) {
        size_t w = 0;
        for (size_t i = 0; i < wis->natoop_count; i++)
            if (!(wis->natoop[i].N == e->N && wis->natoop[i].K == e->K))
                wis->natoop[w++] = wis->natoop[i];
        wis->natoop_count = w;
    }
    if (wis->natoop_count >= wis->natoop_capacity) {
        wis->natoop_capacity = wis->natoop_capacity ? wis->natoop_capacity * 2 : 32;
        wis->natoop = realloc(wis->natoop, wis->natoop_capacity * sizeof(*wis->natoop));
    }
    wis->natoop[wis->natoop_count++] = *e;
    return matches > 0 ? 2 : 1;
}

static inline void vfft_proto_wisdom_free(vfft_proto_wisdom_t *wis) {
    free(wis->entries);
    free(wis->nat);
    free(wis->natoop);
    memset(wis, 0, sizeof(*wis));
}

#endif /* VFFT_PROTO_CORE_WISDOM_READER_H */
