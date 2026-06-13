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
} vfft_proto_wisdom_entry_t;

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
    fprintf(f, "@version 5\n");
    fprintf(f, "# VectorFFT stride wisdom: %zu entries\n", wis->count);
    fprintf(f, "# N K nf factors... best_ns use_blocked split_stage block_groups "
               "use_dif_forward variant_codes... (v=0:FLAT 1:LOG3 2:T1S 3:BUF)\n");
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
