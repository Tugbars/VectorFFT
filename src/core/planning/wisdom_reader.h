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
    /* ── v7 trailing block: NATURAL-ORDER verdict (order=VFFT_ORDER_NATURAL only; the
     * scrambled path never reads these). Same optional-trailing mechanism as exec_me:
     * absent (v5/v6 file) loads as nat_mode=0. The OCaml reader (emit_executor_h.ml
     * parse_wisdom_line) enforces only a MINIMUM token count and indexes from the front,
     * so v7 lines pass through codegen untouched (verified). Token order is FROZEN as
     * `exec_me nat_mode nat_ns [nat_nf nat_factors... nat_prof]` — any future padding
     * fields append AFTER the natural block. natural_order_inplace_design.md §2e. */
    int     nat_mode;                        /* 0=UNSET 1=FREE 2=LEAF_IP 3=SCR 4=PURE_CYCLE 5=PSWAP */
    double  nat_ns;                          /* measured natural-order total (margin rule input) */
    int     nat_nf;                          /* PSWAP only: injected chain length (else 0)  */
    int     nat_factors[STRIDE_MAX_STAGES];  /* PSWAP only: the injected palindromic chain  */
    int     nat_prof;                        /* PSWAP only: uniform variant profile 0/1/2   */
} vfft_proto_wisdom_entry_t;

enum { VFFT_NAT_UNSET = 0, VFFT_NAT_FREE = 1, VFFT_NAT_LEAF_IP = 2,
       VFFT_NAT_SCR = 3, VFFT_NAT_PURE_CYCLE = 4, VFFT_NAT_PSWAP = 5 };

typedef struct {
    vfft_proto_wisdom_entry_t *entries;
    size_t                     count;
    size_t                     capacity;
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
        /* Skip blank / comment / version header lines. */
        char *p = line;
        while (isspace((unsigned char)*p)) p++;
        if (*p == '\0' || *p == '#' || *p == '@') continue;

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
        /* Trailing v7 block: natural-order verdict. Missing (v5/v6) -> nat_mode UNSET. */
        e.nat_mode = 0; e.nat_ns = 0.0; e.nat_nf = 0; e.nat_prof = 0;
        tok = strtok(NULL, " \t\r\n");
        if (tok) {
            e.nat_mode = atoi(tok);
            tok = strtok(NULL, " \t\r\n");
            e.nat_ns = tok ? atof(tok) : 0.0;
            if (e.nat_mode == VFFT_NAT_PSWAP) {
                /* INJECTED PSWAP carries a trailing nf-block (nf factors... prof); OPPORTUNISTIC
                 * PSWAP (calibrated plan + pairs from its own involution perm) writes none, so its
                 * line ends after nat_ns -> nat_nf stays 0 and we KEEP nat_mode=PSWAP (do NOT
                 * downgrade to UNSET, or the verdict is silently lost on reload). */
                tok = strtok(NULL, " \t\r\n");
                if (tok) {
                    e.nat_nf = atoi(tok);
                    if (e.nat_nf < 1 || e.nat_nf > STRIDE_MAX_STAGES) { e.nat_mode = 0; e.nat_nf = 0; goto natdone; }
                    for (int i = 0; i < e.nat_nf; i++) {
                        tok = strtok(NULL, " \t\r\n");
                        if (!tok) { e.nat_mode = 0; e.nat_nf = 0; goto natdone; }
                        e.nat_factors[i] = atoi(tok);
                    }
                    tok = strtok(NULL, " \t\r\n");
                    e.nat_prof = tok ? atoi(tok) : 0;
                }
            }
        }
    natdone:;

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
    fprintf(f, "@version 7\n");
    fprintf(f, "# VectorFFT stride wisdom: %zu entries\n", wis->count);
    fprintf(f, "# N K nf factors... best_ns use_blocked split_stage block_groups "
               "use_dif_forward variant_codes... exec_me (v=0:FLAT 1:LOG3 2:T1S 3:BUF)\n");
    fprintf(f, "# v7 trailing: nat_mode nat_ns [nat_nf nat_factors... nat_prof if PSWAP] "
               "(nat_mode 0=UNSET 1=FREE 2=LEAF_IP 3=SCR 4=PURE_CYCLE 5=PSWAP); "
               "future padding fields append AFTER the natural block\n");
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
        /* v6 trailing field: exec_me (padded verdict; 0 = not pad-measured, written as-is). */
        fprintf(f, " %d", e->exec_me);
        /* v7 trailing block: natural-order verdict (mandatory two tokens keep columns regular;
         * PSWAP appends its self-describing injected-chain block). */
        fprintf(f, " %d %.2f", e->nat_mode, e->nat_ns);
        if (e->nat_mode == VFFT_NAT_PSWAP && e->nat_nf > 0) {
            fprintf(f, " %d", e->nat_nf);
            for (int j = 0; j < e->nat_nf; j++)
                fprintf(f, " %d", e->nat_factors[j]);
            fprintf(f, " %d", e->nat_prof);
        }
        fprintf(f, "\n");
    }
    fclose(f);
    return 0;
}

static inline void vfft_proto_wisdom_free(vfft_proto_wisdom_t *wis) {
    free(wis->entries);
    memset(wis, 0, sizeof(*wis));
}

#endif /* VFFT_PROTO_CORE_WISDOM_READER_H */
