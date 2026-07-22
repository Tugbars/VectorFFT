/**
 * fftnd_wisdom.h -- persisted calibration verdicts for rank-general plans.
 *
 * One text line per cell, human-readable/greppable like the other wisdom
 * files. Cell key = (rank, N[0..rank-1]). A line stores everything needed
 * to rebuild the winning plan without re-measuring:
 *
 *   nd r=3 n=128,128,128 s=1 B=8 blk=0,512 ns=4.61e+06 \
 *      ax0=T:8v0,4v2,4v0 ax1=T:4v0,4v1,8v0 ax2=F:16v0,8v0
 *
 *   s    -- split point (axes < s unfused, >= s fused per block)
 *   blk  -- lane_block per axis 0..rank-2 (0 = flat)
 *   B    -- last-axis tile height
 *   axM  -- inner recipe: 'T'=DIT / 'F'=DIF, then factor 'v' variant list;
 *           or the literal 'auto' (prime / non-smooth axes rebuilt through
 *           vfft_proto_auto_plan_dispatch -> Rader/Bluestein).
 *   ns   -- banked end-to-end fwd time (informational).
 *
 * Loader is line-oriented and tolerant: unknown lines are skipped, first
 * matching cell wins. Appending is the only write mode (calibrate-on-miss).
 */
#ifndef STRIDE_FFTND_WISDOM_H
#define STRIDE_FFTND_WISDOM_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "fftnd.h"

#ifndef FFTND_WIS_MAX_F
#define FFTND_WIS_MAX_F 12
#endif

typedef struct {
    int rank;
    int N[FFTND_MAX_RANK];
    int T;                                    /* thread count the verdict was
                                                 measured at (part of the key:
                                                 s/blocking optima are
                                                 T-dependent) */
    int split;
    size_t B;
    size_t lane_block[FFTND_MAX_RANK];        /* [0..rank-2] used */
    /* per-axis inner recipe */
    int ax_auto[FFTND_MAX_RANK];              /* 1 = auto_plan_dispatch */
    int ax_dif[FFTND_MAX_RANK];
    int ax_nf[FFTND_MAX_RANK];
    int ax_f[FFTND_MAX_RANK][FFTND_WIS_MAX_F];
    int ax_v[FFTND_MAX_RANK][FFTND_WIS_MAX_F];
    double ns;                                 /* banked fwd ns */
} fftnd_wis_entry_t;

/* ── format one entry as a wisdom line ── */
static void fftnd_wis_format(const fftnd_wis_entry_t *e, char *buf, size_t sz) {
    int off = snprintf(buf, sz, "nd r=%d T=%d n=", e->rank,
                       e->T > 0 ? e->T : 1);
    for (int m = 0; m < e->rank; m++)
        off += snprintf(buf + off, sz - off, "%d%s", e->N[m],
                        m + 1 < e->rank ? "," : "");
    off += snprintf(buf + off, sz - off, " s=%d B=%zu blk=", e->split, e->B);
    for (int m = 0; m < e->rank - 1; m++)
        off += snprintf(buf + off, sz - off, "%zu%s", e->lane_block[m],
                        m + 2 < e->rank ? "," : "");
    off += snprintf(buf + off, sz - off, " ns=%.3e", e->ns);
    for (int m = 0; m < e->rank; m++) {
        if (e->ax_auto[m]) {
            off += snprintf(buf + off, sz - off, " ax%d=auto", m);
        } else {
            off += snprintf(buf + off, sz - off, " ax%d=%c:", m,
                            e->ax_dif[m] ? 'F' : 'T');
            for (int s = 0; s < e->ax_nf[m]; s++)
                off += snprintf(buf + off, sz - off, "%dv%d%s",
                                e->ax_f[m][s], e->ax_v[m][s],
                                s + 1 < e->ax_nf[m] ? "," : "");
        }
    }
    snprintf(buf + off, sz - off, "\n");
}

/* ── parse helpers ── */
static int _fftnd_wis_parse_ints(const char *s, char sep, int *out, int max) {
    int n = 0;
    while (*s && n < max) {
        out[n++] = (int)strtol(s, (char **)&s, 10);
        if (*s == sep) s++;
        else break;
    }
    return n;
}

/* Parse one line into *e. Returns 1 on success. */
static int fftnd_wis_parse(const char *line, fftnd_wis_entry_t *e) {
    memset(e, 0, sizeof(*e));
    const char *p = strstr(line, "nd r=");
    if (!p) return 0;
    e->rank = atoi(p + 5);
    if (e->rank < 2 || e->rank > FFTND_MAX_RANK) return 0;
    p = strstr(line, " T=");
    e->T = p ? atoi(p + 3) : 1;               /* absent in old lines -> T=1 */
    p = strstr(line, " n=");   if (!p) return 0;
    if (_fftnd_wis_parse_ints(p + 3, ',', e->N, e->rank) != e->rank) return 0;
    p = strstr(line, " s=");   if (!p) return 0;
    e->split = atoi(p + 3);
    p = strstr(line, " B=");   if (!p) return 0;
    e->B = (size_t)strtoull(p + 3, NULL, 10);
    p = strstr(line, " blk="); if (!p) return 0;
    {
        int tmp[FFTND_MAX_RANK] = {0};
        int got = _fftnd_wis_parse_ints(p + 5, ',', tmp, e->rank - 1);
        if (got != e->rank - 1) return 0;
        for (int m = 0; m < e->rank - 1; m++) e->lane_block[m] = (size_t)tmp[m];
    }
    p = strstr(line, " ns=");
    if (p) e->ns = strtod(p + 4, NULL);
    for (int m = 0; m < e->rank; m++) {
        char tag[8]; snprintf(tag, sizeof tag, " ax%d=", m);
        p = strstr(line, tag);
        if (!p) return 0;
        p += strlen(tag);
        if (!strncmp(p, "auto", 4)) { e->ax_auto[m] = 1; continue; }
        if (*p != 'T' && *p != 'F') return 0;
        e->ax_dif[m] = (*p == 'F');
        p += 2;                                   /* skip 'T:' / 'F:' */
        int nf = 0;
        while (*p && *p != ' ' && *p != '\n' && nf < FFTND_WIS_MAX_F) {
            e->ax_f[m][nf] = (int)strtol(p, (char **)&p, 10);
            if (*p != 'v') return 0;
            p++;
            e->ax_v[m][nf] = (int)strtol(p, (char **)&p, 10);
            nf++;
            if (*p == ',') p++;
        }
        if (nf < 1) return 0;
        e->ax_nf[m] = nf;
    }
    return 1;
}

/* Find a cell in the wisdom file. Returns 1 on hit. */
static int fftnd_wis_lookup(const char *path, int rank, const int *N,
                            int T, fftnd_wis_entry_t *out) {
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    char line[1024];
    while (fgets(line, sizeof line, f)) {
        fftnd_wis_entry_t e;
        if (!fftnd_wis_parse(line, &e)) continue;
        if (e.rank != rank || e.T != (T > 0 ? T : 1)) continue;
        int same = 1;
        for (int m = 0; m < rank; m++) if (e.N[m] != N[m]) { same = 0; break; }
        if (same) { *out = e; fclose(f); return 1; }
    }
    fclose(f);
    return 0;
}

/* Append an entry (calibrate-on-miss write path). Returns 1 on success. */
static int fftnd_wis_append(const char *path, const fftnd_wis_entry_t *e) {
    FILE *f = fopen(path, "a");
    if (!f) return 0;
    char buf[1024];
    fftnd_wis_format(e, buf, sizeof buf);
    fputs(buf, f);
    fclose(f);
    return 1;
}

/* Rebuild the winning plan from an entry (no measuring). NULL on failure. */
static stride_plan_t *fftnd_wis_build(const fftnd_wis_entry_t *e,
                                      const vfft_proto_registry_t *reg) {
    stride_fftnd_data_t tmp; memset(&tmp, 0, sizeof tmp);
    tmp.rank = e->rank;
    for (int m = 0; m < e->rank; m++) tmp.N[m] = e->N[m];
    _fftnd_fill_ok(&tmp);
    stride_plan_t *plans[FFTND_MAX_RANK] = { 0 };
    for (int m = 0; m < e->rank; m++) {
        size_t Kp = (m == e->rank - 1) ? e->B : tmp.K[m];
        if (e->ax_auto[m])
            plans[m] = vfft_proto_auto_plan_dispatch(e->N[m], Kp, reg, NULL);
        else
            plans[m] = vfft_proto_plan_create_ex(e->N[m], Kp,
                            (int *)e->ax_f[m], (int *)e->ax_v[m],
                            e->ax_nf[m], e->ax_dif[m], reg);
        if (!plans[m]) {
            for (int i = 0; i < m; i++) stride_plan_destroy(plans[i]);
            return NULL;
        }
    }
    return stride_plan_nd_from(e->rank, e->N, e->B, e->split,
                               e->lane_block, plans);
}

#endif /* STRIDE_FFTND_WISDOM_H */
