/* wisdom2_migrate.h — the one-shot LOSSLESS migrator: legacy wisdom files
 * -> the wisdom2 store (campaign items 0.9 + wave migrations). OWNED BY THE
 * MODULE per the thin-driver law: the bench driver only parses arguments
 * and calls in.
 *
 * Laws implemented here (campaign RITUAL + README §6):
 *   - LOSSLESS, machine-checked: every source LINE is exactly one of
 *     {skipped (non-record: comments/blanks), migrated, quarantined};
 *     total == skipped + migrated + quarantined or the gate fails.
 *   - PLAN-PRESERVING, zero re-timing: numbers are carried as data with
 *     from=<file>:<line> lineage; nothing is measured.
 *   - The LEGACY parser is the SHIPPED one, linked verbatim
 *     (vfft_oop_wisdom_load + the oop_plan.h codecs) — never re-implemented.
 *     Each line is parsed through a one-line probe file so line numbers are
 *     exact and the legacy loader's silent drops become visible
 *     (quarantined as legacy-parse-drop instead of vanishing).
 *   - QUARANTINE, never delete, never guess: garbage rows, shadowed
 *     duplicates, sub-2048 kind-4 wrong-slot rows, and undecodable codec
 *     fields go to wisdom2_quarantine.txt verbatim with reasons.
 *   - IDEMPOTENT: re-running over existing output is byte-identical
 *     (equal records re-bank as equals; quarantine entries dedup on raw).
 *
 * Wave-1 mapping implemented: oop_wisdom.txt kinds 0-5. Other legacy
 * families land at their waves as further vw2_migrate_* functions here.
 */
#ifndef VFFT_WISDOM2_MIGRATE_H
#define VFFT_WISDOM2_MIGRATE_H

#include "wisdom2.h"
#include "wisdom2_oop_reader.h"  /* the name tables + the production read
                                    side this migrator must round-trip with
                                    (one definition — forward and inverse
                                    maps cannot drift) */
#include "wisdom2_oop.h"         /* THE legacy reader + codecs, verbatim */

#define vw2__mig_sp_name  vw2_oop_sp_name
#define vw2__mig_il_name  vw2_oop_il_name
#define vw2__mig_var_name vw2_oop_var_name

typedef struct {
    int lines_total, skipped, migrated, quarantined;
    int records_out;
    int io_errors;                /* persist failures — any nonzero fails the run */
    int per_kind[6];
} vw2_mig_stats_t;

#if defined(_WIN32)
#  define VW2__MIG_MKDIR(d) _mkdir(d)
#else
#  define VW2__MIG_MKDIR(d) mkdir(d, 0755)
#endif

/* ----------------------------------------------------------- helpers */

static inline int vw2__mig_quar_has(const vw2_store_t *st, const char *raw)
{
    char path[640];
    char *line;
    FILE *f;
    int oom = 0, found = 0;
    snprintf(path, sizeof path, "%s/wisdom2_quarantine.txt", st->dir);
    f = fopen(path, "rb");
    if (!f) return 0;
    while ((line = vw2__readline(f, &oom)) != NULL) {
        const char *r = strstr(line, "raw=");
        if (r && !strcmp(r + 4, raw)) { free(line); found = 1; break; }
        free(line);
    }
    fclose(f);
    return found;
}

static inline int vw2__mig_quar(vw2_store_t *st, vw2_mig_stats_t *stats,
                                const char *reason, const char *from, const char *raw)
{
    int rc;
    stats->quarantined++;
    if (vw2__mig_quar_has(st, raw)) return VW2_OK;   /* idempotent re-run */
    rc = vw2_quarantine_append(st, reason, from, raw);
    if (rc != VW2_OK) {
        stats->io_errors++;
        fprintf(stderr, "[wisdom2_migrate] QUARANTINE WRITE FAILED (%d) for %s — "
                        "a quarantined row must PERSIST, not just count\n", rc, from);
    }
    return rc;
}

static inline const vw2_rec_t *vw2__mig_find(const vw2_store_t *st, const vw2_key_t *k)
{
    int i;
    for (i = 0; i < st->nrec; i++)
        if (vw2_key_eq(&st->rec[i].key, k)) return &st->rec[i];
    return NULL;
}

/* keys banked THIS RUN — distinguishes a true intra-file duplicate (first
 * line wins, later line quarantined per the first-match-wins legacy law)
 * from a previous run's record (idempotent re-bank) and from a live verdict
 * outranking a wart seed. */
typedef struct { vw2_key_t *k; int n, cap; } vw2__mig_seen_t;

static inline int vw2__mig_seen(const vw2__mig_seen_t *s, const vw2_key_t *k)
{
    int i;
    for (i = 0; i < s->n; i++)
        if (vw2_key_eq(&s->k[i], k)) return 1;
    return 0;
}

static inline int vw2__mig_seen_add(vw2__mig_seen_t *s, const vw2_key_t *k)
{
    if (s->n == s->cap) {
        int nc = s->cap ? s->cap * 2 : 64;
        vw2_key_t *nk = (vw2_key_t *)realloc(s->k, (size_t)nc * sizeof *nk);
        if (!nk) { vw2__oom(); return VW2_ENOMEM; }
        s->k = nk; s->cap = nc;
    }
    s->k[s->n++] = *k;
    return VW2_OK;
}

/* Bank one migrated record under the collision law:
 *   - duplicate key ALREADY BANKED THIS RUN and not outranked -> quarantine
 *     (first match wins, like the legacy linear lookup did);
 *   - key from a PREVIOUS run -> re-bank (equal records replace byte-stably);
 *   - rank differences resolve by the merge law (a live race verdict
 *     replaces a wart seed; a seed never displaces a verdict — the refusal
 *     becomes a quarantine with its reason).
 * Takes ownership of rec either way. */
static inline int vw2__mig_bank(vw2_store_t *st, vw2__mig_seen_t *seen,
                                vw2_rec_t *rec, const char **why)
{
    const vw2_rec_t *inc = vw2__mig_find(st, &rec->key);
    int was_new = (inc == NULL);
    if (inc && vw2__mig_seen(seen, &rec->key) &&
        vw2__src_rank(rec) <= vw2__src_rank(inc)) {
        *why = (vw2__src_rank(rec) < vw2__src_rank(inc))
                   ? "shadowed-by-live-verdict"     /* seed vs a real verdict */
                   : "shadowed-duplicate";          /* true intra-file duplicate */
        vw2_rec_free(rec);
        return -1;
    }
    {
        vw2_key_t k = rec->key;
        int rc = vw2_bank(st, rec);
        if (rc != VW2_OK) {
            vw2_rec_free(rec);
            *why = (rc == VW2_ERANK) ? "shadowed-by-live-verdict" : "bank-refused";
            return -1;
        }
        if (vw2__mig_seen_add(seen, &k) != VW2_OK) { *why = "oom"; return -1; }
    }
    return was_new ? 1 : 0;   /* 1 = new key, 0 = replaced existing */
}

/* parse ONE legacy oop line through the SHIPPED reader via a probe file.
 * 1 = parsed; 0 = the legacy loader dropped it (a real legacy silent-drop);
 * -1 = probe I/O failure (FATAL — never misreported as a parse drop). */
static inline int vw2__mig_parse_line(const char *outdir, const char *line,
                                      vfft_oop_wisdom_entry_t *e)
{
    char probe[640];
    FILE *f;
    static vfft_oop_wisdom_t w;   /* 1024-entry table — keep off the stack */
    snprintf(probe, sizeof probe, "%s/mig_probe.tmp", outdir);
    f = fopen(probe, "wb");
    if (!f) return -1;
    if (fprintf(f, "%s\n", line) < 0) { fclose(f); remove(probe); return -1; }
    if (fclose(f) != 0) { remove(probe); return -1; }
    if (vfft_oop_wisdom_load(&w, probe) != 0) { remove(probe); return -1; }
    remove(probe);
    if (w.count == 1) { *e = w.e[0]; return 1; }
    return 0;
}

static inline void vw2__mig_join(char *out, size_t cap, const int *v, int n)
{
    size_t off = 0; int i, r;
    out[0] = 0;
    for (i = 0; i < n; i++) {
        r = snprintf(out + off, cap - off, "%s%d", i ? "." : "", v[i]);
        if (r < 0 || (size_t)r >= cap - off) return;
        off += (size_t)r;
    }
}

static inline void vw2__mig_join_vars(char *out, size_t cap, const int *v, int n)
{
    size_t off = 0; int i, r;
    out[0] = 0;
    for (i = 0; i < n; i++) {
        const char *nm = (v[i] >= 0 && v[i] <= 2) ? vw2__mig_var_name[v[i]] : "?";
        r = snprintf(out + off, cap - off, "%s%s", i ? "." : "", nm);
        if (r < 0 || (size_t)r >= cap - off) return;
        off += (size_t)r;
    }
}

/* ---------------------------------------------------- kind -> record(s) */

/* Emits the wisdom2 record(s) for one parsed legacy entry, via the SHARED
 * family codec (vw2_oop_rec_from_entry / vw2_oop_recs_from_kind5 in
 * wisdom2_oop_reader.h — one constructor, used by the runtime bank sites
 * too). This wrapper adds only migration policy: src selection (warts ->
 * seeds), the seen-set collision law, and stats. Returns VW2_OK or a
 * negative refusal with *why for the quarantine. */
static inline int vw2__mig_oop_entry(vw2_store_t *st, vw2__mig_seen_t *seen,
                                     vw2_mig_stats_t *stats,
                                     const vfft_oop_wisdom_entry_t *e,
                                     const char *from, const char **why)
{
    /* kinds 0-2 at K%8!=0 are the legacy BANK-ONLY warts (unreplayable
     * today, D5): they migrate as SEEDS -- race proposals, never verdicts --
     * so a live kind-4 verdict at the same cell outranks them by the merge
     * law instead of colliding by file order. */
    const int is_seed = (e->kind <= VFFT_OOP_KIND_MODEB) &&
                        (e->K == 0 || (e->K % 8u) != 0);
    const char *src = is_seed ? "seed" : "migrated";

    *why = NULL;
    if (e->kind == VFFT_OOP_KIND_ZR2C) {
        vw2_rec_t slots[4];
        int n = vw2_oop_recs_from_kind5(e, "migrated", from, slots, why), i, b;
        if (n <= 0) return -1;
        for (i = 0; i < n; i++) {
            b = vw2__mig_bank(st, seen, &slots[i], why);
            if (b < 0) {
                int t;
                for (t = i + 1; t < n; t++) vw2_rec_free(&slots[t]);
                return -1;
            }
            if (b == 1) stats->records_out++;
        }
        stats->per_kind[e->kind]++;
        return VW2_OK;
    }
    {
        vw2_rec_t r;
        int b;
        if (vw2_oop_rec_from_entry(&r, e, src, from, why) != VW2_OK) return -1;
        b = vw2__mig_bank(st, seen, &r, why);
        if (b < 0) return -1;
        if (b == 1) stats->records_out++;
    }
    stats->per_kind[e->kind]++;
    return VW2_OK;
}


/* ------------------------------------------------------- the oop family */

/* Migrate one legacy oop_wisdom.txt into the wisdom2 store at outdir.
 * NEVER touches the legacy file. Returns VW2_OK when the row-conservation
 * identity holds (total == skipped + migrated + quarantined). */
static inline int vw2_migrate_oop(const char *legacy_path, const char *outdir,
                                  vw2_mig_stats_t *stats, int verbose)
{
    vw2_store_t st;
    vw2__mig_seen_t seen;
    FILE *f;
    char *line;
    char from[256];
    const char *base;
    int oom = 0, lineno = 0, rc;

    memset(&seen, 0, sizeof seen);
    memset(stats, 0, sizeof *stats);
    f = fopen(legacy_path, "rb");
    if (!f) {
        fprintf(stderr, "[wisdom2_migrate] cannot open legacy file %s\n", legacy_path);
        return VW2_EOPEN;
    }
    VW2__MIG_MKDIR(outdir);   /* out dir must exist before probes/quarantine */
    rc = vw2_open(&st, outdir, 1);
    if (rc == VW2_EVERSION) { fclose(f); vw2_close(&st); return VW2_EPOISON; }

    base = strrchr(legacy_path, '/');
#if defined(_WIN32)
    { const char *b2 = strrchr(legacy_path, '\\'); if (b2 && (!base || b2 > base)) base = b2; }
#endif
    base = base ? base + 1 : legacy_path;

    /* TWO PASSES for determinism: pass 0 banks VERDICTS (race-class rows),
     * pass 1 banks SEEDS (the K%8!=0 bank-only warts). A seed whose key is
     * owned by a live verdict quarantines identically regardless of file
     * order or re-runs — lossless (the row survives verbatim) and
     * idempotent. Lines are read once into memory. */
    {
        char **lines = NULL; int nlines = 0, caplines = 0, pass;
        while ((line = vw2__readline(f, &oom)) != NULL) {
            if (nlines == caplines) {
                int nc = caplines ? caplines * 2 : 128;
                char **nl = (char **)realloc(lines, (size_t)nc * sizeof *nl);
                if (!nl) { vw2__oom(); oom = 1; free(line); break; }
                lines = nl; caplines = nc;
            }
            lines[nlines++] = line;
        }
        fclose(f);
        if (oom) {
            for (lineno = 0; lineno < nlines; lineno++) free(lines[lineno]);
            free(lines); free(seen.k); vw2_close(&st);
            return VW2_ENOMEM;
        }
        for (pass = 0; pass < 2; pass++)
            for (lineno = 0; lineno < nlines; lineno++) {
                vfft_oop_wisdom_entry_t e;
                int n, is_seed;
                line = lines[lineno];
                if (!line[0] || line[0] == '#' || line[0] == '@') {
                    if (pass == 0) { stats->lines_total++; stats->skipped++; }
                    continue;
                }
                if (pass == 0) stats->lines_total++;
                snprintf(from, sizeof from, "%s:%d", base, lineno + 1);
                n = vw2__mig_parse_line(outdir, line, &e);
                if (n < 0) {
                    fprintf(stderr, "[wisdom2_migrate] FATAL: probe I/O failed at %s — aborting "
                                    "(never misreported as a parse drop)\n", from);
                    for (lineno = 0; lineno < nlines; lineno++) free(lines[lineno]);
                    free(lines); free(seen.k); vw2_close(&st);
                    return VW2_EIO;
                }
                if (n == 0) {
                    if (pass == 0) {
                        vw2__mig_quar(&st, stats, "legacy-parse-drop", from, line);
                        if (verbose) fprintf(stderr, "[wisdom2_migrate] %s: legacy parser dropped: %s\n", from, line);
                    }
                    continue;
                }
                is_seed = (e.kind <= VFFT_OOP_KIND_MODEB) && (e.K == 0 || (e.K % 8u) != 0);
                if (is_seed != pass) continue;      /* verdicts pass 0, seeds pass 1 */
                {
                    const char *why = NULL;
                    if (vw2__mig_oop_entry(&st, &seen, stats, &e, from, &why) == VW2_OK) {
                        stats->migrated++;
                    } else {
                        vw2__mig_quar(&st, stats, why ? why : "refused", from, line);
                        if (verbose) fprintf(stderr, "[wisdom2_migrate] %s: quarantined (%s)\n", from, why);
                    }
                }
            }
        for (lineno = 0; lineno < nlines; lineno++) free(lines[lineno]);
        free(lines);
    }
    free(seen.k);

    rc = vw2_save(&st);
    vw2_close(&st);

    fprintf(stderr, "[wisdom2_migrate] %s: %d lines = %d skipped + %d migrated + %d quarantined"
                    " -> %d records (k0..k5: %d/%d/%d/%d/%d/%d)\n",
            base, stats->lines_total, stats->skipped, stats->migrated, stats->quarantined,
            stats->records_out, stats->per_kind[0], stats->per_kind[1], stats->per_kind[2],
            stats->per_kind[3], stats->per_kind[4], stats->per_kind[5]);
    if (stats->lines_total != stats->skipped + stats->migrated + stats->quarantined) {
        fprintf(stderr, "[wisdom2_migrate] ROW CONSERVATION VIOLATED\n");
        return VW2_EKEY;
    }
    if (stats->io_errors) {
        fprintf(stderr, "[wisdom2_migrate] %d persist failure(s) — run is NOT lossless\n",
                stats->io_errors);
        return VW2_EIO;
    }
    return rc;
}

/* ------------------------------------------------- verify (Gate A leg) */

/* Field-level equivalence: re-read the legacy file per line and the SAVED
 * wisdom2 store, and compare every migrated verdict field-by-field through
 * the codecs. Zero timing. Returns number of mismatches. */
static inline int vw2_migrate_oop_verify(const char *legacy_path, const char *outdir)
{
    vw2_store_t st;
    FILE *f;
    char *line;
    int oom = 0, bad = 0;

    if (vw2_open(&st, outdir, 0) == VW2_EVERSION) { vw2_close(&st); return 1; }
    f = fopen(legacy_path, "rb");
    if (!f) { vw2_close(&st); return 1; }

    while ((line = vw2__readline(f, &oom)) != NULL) {
        vfft_oop_wisdom_entry_t e;
        if (line[0] && line[0] != '#' && line[0] != '@' &&
            vw2__mig_parse_line(outdir, line, &e) == 1) {
            vw2_key_t k;
            const vw2_rec_t *hit = NULL;
            int i;
            const int is_seed = (e.kind <= VFFT_OOP_KIND_MODEB) &&
                                (e.K == 0 || (e.K % 8u) != 0);
            memset(&k, 0, sizeof k);
            k.rank = 1; k.n[0] = e.N;
            if (e.kind <= VFFT_OOP_KIND_MODEB) {
                k.t = VW2_T_C2C; k.q = (int64_t)e.K;
                k.ord = (e.kind == VFFT_OOP_KIND_MODEB) ? VW2_ORD_SCR : VW2_ORD_NAT;
                k.pl = VW2_PL_OOP;
            } else if (e.kind == VFFT_OOP_KIND_BAILEY2V) {
                k.t = VW2_T_C2C; k.q = -1; k.ord = VW2_ORD_ANY; k.pl = VW2_PL_ANY;
            } else if (e.kind == VFFT_OOP_KIND_ZSPLIT) {
                k.t = VW2_T_C2C; k.q = 1; k.ord = VW2_ORD_SCR; k.pl = VW2_PL_OOP;
            } else {
                /* kind-5: check each measured slot */
                int slot;
                for (slot = 0; slot < 4; slot++) {
                    int v = vfft_zr2c_kv_get(e.zr_kv, slot);
                    if (!v) continue;
                    memset(&k, 0, sizeof k);
                    k.rank = 1; k.n[0] = e.N; k.q = 1; k.ord = VW2_ORD_NAT;
                    k.t = (slot >> 1) ? VW2_T_C2R : VW2_T_R2C;
                    k.pl = (slot & 1) ? VW2_PL_IP : VW2_PL_OOP;
                    hit = NULL;
                    for (i = 0; i < st.nrec; i++)
                        if (vw2_key_eq(&st.rec[i].key, &k)) { hit = &st.rec[i]; break; }
                    if (!hit) { bad++; fprintf(stderr, "[verify] kind-5 slot missing: N=%d slot=%d\n", e.N, slot); continue; }
                    {
                        const char *rt = vw2_rec_get(hit, "route");
                        if (!rt || strcmp(rt, v == 1 ? "child_oop_il" : "child_nat_ip") != 0) {
                            bad++; fprintf(stderr, "[verify] kind-5 route mismatch N=%d slot=%d\n", e.N, slot);
                        }
                    }
                }
                free(line);
                continue;
            }
            for (i = 0; i < st.nrec; i++)
                if (vw2_key_eq(&st.rec[i].key, &k)) { hit = &st.rec[i]; break; }
            if (!hit) { free(line); continue; }   /* quarantined lines have no record */
            if (is_seed) {
                /* a seed row's key may legitimately be owned by a live
                 * verdict (shadowed at migration); only check when the
                 * record really is this seed */
                const char *sv = vw2_rec_get(hit, "src");
                if (!sv || strcmp(sv, "seed")) { free(line); continue; }
            }
            {
                const char *v;
                char want[192];
                int ch[VFFT_K1_CC_MAX_NF], nf;
                switch (e.kind) {
                case VFFT_OOP_KIND_LEAF:
                    v = vw2_rec_get(hit, "route");
                    if (!v || strcmp(v, "leaf")) { bad++; fprintf(stderr, "[verify] kind-0 N=%d route mismatch\n", e.N); }
                    break;
                case VFFT_OOP_KIND_BAILEY2:
                    snprintf(want, sizeof want, "%d.%d", e.R1, e.R2);
                    v = vw2_rec_get(hit, "chain");
                    if (!v || strcmp(v, want)) { bad++; fprintf(stderr, "[verify] kind-%d N=%d chain: want %s got %s\n", (int)e.kind, e.N, want, v ? v : "(absent)"); }
                    v = vw2_rec_get(hit, "t1p");
                    if (!v || strcmp(v, e.t1p_variant ? "log3" : "flat")) { bad++; fprintf(stderr, "[verify] kind-1 N=%d t1p mismatch\n", e.N); }
                    break;
                case VFFT_OOP_KIND_MODEB:
                    vw2__mig_join(want, sizeof want, e.factors, e.nf);
                    v = vw2_rec_get(hit, "chain");
                    if (!v || strcmp(v, want)) { bad++; fprintf(stderr, "[verify] kind-%d N=%d chain: want %s got %s\n", (int)e.kind, e.N, want, v ? v : "(absent)"); }
                    break;
                case VFFT_OOP_KIND_BAILEY2V:
                    if (e.k1_sp_route >= 0 && e.k1_sp_route <= 7) {
                        v = vw2_rec_get(hit, "sp_route");
                        if (!v || strcmp(v, vw2__mig_sp_name[e.k1_sp_route])) { bad++; fprintf(stderr, "[verify] kind-3 N=%d sp_route mismatch\n", e.N); }
                    }
                    snprintf(want, sizeof want, "%d.%d", e.R1, e.R2);
                    v = vw2_rec_get(hit, "sp_pair");
                    if (!v || strcmp(v, want)) { bad++; fprintf(stderr, "[verify] kind-3 N=%d sp_pair: want %s got %s\n", e.N, want, v ? v : "(absent)"); }
                    v = vw2_rec_get(hit, "ran");
                    snprintf(want, sizeof want, "%lld", (long long)e.K);
                    if (!v || strcmp(v, want)) { bad++; fprintf(stderr, "[verify] kind-3 N=%d ran: want %s got %s\n", e.N, want, v ? v : "(absent)"); }
                    if (e.il_kv) {
                        v = vw2_rec_get(hit, "il_kv");
                        snprintf(want, sizeof want, "%d", e.il_kv);
                        if (!v || strcmp(v, want)) { bad++; fprintf(stderr, "[verify] kind-3 N=%d il_kv mismatch\n", e.N); }
                    }
                    break;
                case VFFT_OOP_KIND_ZSPLIT:
                    nf = vfft_k1_cc_chain_decode(e.cc_chain, ch);
                    if (nf > 0) {
                        vw2__mig_join(want, sizeof want, ch, nf);
                        v = vw2_rec_get(hit, "chain");
                        if (!v || strcmp(v, want)) bad++;
                    }
                    if (e.zt_tw > 0) {
                        v = vw2_rec_get(hit, "zt_tw");
                        snprintf(want, sizeof want, "%d", e.zt_tw);
                        if (!v || strcmp(v, want)) { bad++; fprintf(stderr, "[verify] kind-4 N=%d zt_tw mismatch\n", e.N); }
                    }
                    break;
                default: break;
                }
                if (e.ns > 0.0 && e.kind != VFFT_OOP_KIND_ZR2C) {
                    v = vw2_rec_get(hit, "ns");
                    if (!v || (e.ns - atof(v) > 0.05) || (atof(v) - e.ns > 0.05)) { bad++; fprintf(stderr, "[verify] kind-%d N=%d ns: want %.1f got %s\n", (int)e.kind, e.N, e.ns, v ? v : "(absent)"); }
                }
            }
        }
        free(line);
    }
    fclose(f);
    vw2_close(&st);
    fprintf(stderr, "[wisdom2_migrate] verify: %d mismatch(es)\n", bad);
    return bad;
}

/* --------------------------------------------- reader-equivalence gate */

/* THE equivalence proof for the read side (the data half of Gate B): for
 * every verdict the LEGACY lookups would serve from the legacy file, the
 * wisdom2 reader (wisdom2_oop_reader.h) must produce a field-identical
 * legacy entry from the MIGRATED store — and for every row the legacy
 * machinery could NOT serve (K%8 warts, sub-2048 kind-4, quarantined
 * garbage), the wisdom2 reader must MISS. Zero timing. Returns mismatches. */
static inline int vw2_migrate_oop_reader_gate(const char *legacy_path, const char *outdir)
{
    static vfft_oop_wisdom_t w;      /* 1024-entry table — off the stack   */
    vw2_store_t st;
    int i, j, bad = 0, checked = 0;

    if (vfft_oop_wisdom_load(&w, legacy_path) != 0) {
        fprintf(stderr, "[reader-gate] cannot load legacy file\n");
        return 1;
    }
    if (vw2_open(&st, outdir, 0) == VW2_EVERSION) { vw2_close(&st); return 1; }

#define VW2__RG_BAD(fmt, ...) do { bad++; fprintf(stderr, "[reader-gate] " fmt "\n", __VA_ARGS__); } while (0)

    for (i = 0; i < w.count; i++) {
        const vfft_oop_wisdom_entry_t *e = &w.e[i];
        vfft_oop_wisdom_entry_t got;
        int first = 1;

        /* legacy lookups serve the FIRST row of a cell; later duplicates
         * were never servable — skip them (the migrator quarantined them) */
        for (j = 0; j < i; j++) {
            const vfft_oop_wisdom_entry_t *p = &w.e[j];
            if (p->N != e->N) continue;
            if (e->kind >= VFFT_OOP_KIND_BAILEY2V) {
                if (p->kind == e->kind) { first = 0; break; }
            } else if (p->kind <= VFFT_OOP_KIND_MODEB && p->K == e->K &&
                       (p->kind == VFFT_OOP_KIND_MODEB) == (e->kind == VFFT_OOP_KIND_MODEB)) {
                first = 0; break;
            }
        }
        if (!first) continue;

        if (e->kind <= VFFT_OOP_KIND_MODEB) {
            int ord = (e->kind == VFFT_OOP_KIND_MODEB) ? 2 : 1;
            int servable = (e->K != 0 && (e->K % 8u) == 0);
            int garbage = 0, t;
            if (e->kind == VFFT_OOP_KIND_MODEB)
                for (t = 0; t < e->nf; t++)
                    if (e->variants[t] < 0 || e->variants[t] > 2) garbage = 1;
            if (!servable || garbage) {
                if (vw2_oop_lookup_ord(&st, e->N, e->K, ord, &got))
                    VW2__RG_BAD("kind-%d N=%d K=%lld: unservable legacy row RESOLVED (must miss)",
                                (int)e->kind, e->N, (long long)e->K);
                checked++;
                continue;
            }
            if (!vw2_oop_lookup_ord(&st, e->N, e->K, ord, &got)) {
                VW2__RG_BAD("kind-%d N=%d K=%lld: servable row MISSED", (int)e->kind, e->N, (long long)e->K);
                continue;
            }
            if (got.kind != e->kind) VW2__RG_BAD("N=%d K=%lld: kind %d != %d", e->N, (long long)e->K, (int)got.kind, (int)e->kind);
            if (e->kind == VFFT_OOP_KIND_BAILEY2 &&
                (got.R1 != e->R1 || got.R2 != e->R2 || got.t1p_variant != e->t1p_variant))
                VW2__RG_BAD("bailey2 N=%d K=%lld: pair/t1p mismatch", e->N, (long long)e->K);
            if (e->kind == VFFT_OOP_KIND_MODEB) {
                if (got.nf != e->nf) VW2__RG_BAD("modeb N=%d K=%lld: nf %d != %d", e->N, (long long)e->K, got.nf, e->nf);
                else for (t = 0; t < e->nf; t++)
                    if (got.factors[t] != e->factors[t] || got.variants[t] != e->variants[t])
                        { VW2__RG_BAD("modeb N=%d K=%lld: stage %d mismatch", e->N, (long long)e->K, t); break; }
            }
            if (e->ns > 0.0 && (got.ns - e->ns > 0.05 || e->ns - got.ns > 0.05))
                VW2__RG_BAD("kind-%d N=%d K=%lld: ns %.1f != %.1f", (int)e->kind, e->N, (long long)e->K, got.ns, e->ns);
            checked++;
        }
        else if (e->kind == VFFT_OOP_KIND_BAILEY2V) {
            if (!vw2_oop_lookup_k1(&st, e->N, &got)) { VW2__RG_BAD("kind-3 N=%d: MISSED", e->N); continue; }
            if (got.k1_sp_route != e->k1_sp_route || got.R1 != e->R1 || got.R2 != e->R2)
                VW2__RG_BAD("kind-3 N=%d: sp mismatch (route %d/%d pair %d.%d/%d.%d)",
                            e->N, got.k1_sp_route, e->k1_sp_route, got.R1, got.R2, e->R1, e->R2);
            if (got.k1_il_route != e->k1_il_route || got.il_R1 != e->il_R1 || got.il_R2 != e->il_R2)
                VW2__RG_BAD("kind-3 N=%d: il mismatch", e->N);
            if (got.il_kv != e->il_kv) VW2__RG_BAD("kind-3 N=%d: il_kv %d != %d", e->N, got.il_kv, e->il_kv);
            if (got.cc_chain != e->cc_chain) VW2__RG_BAD("kind-3 N=%d: cc_chain %d != %d", e->N, got.cc_chain, e->cc_chain);
            if (got.cc_vars != e->cc_vars) VW2__RG_BAD("kind-3 N=%d: cc_vars %d != %d", e->N, got.cc_vars, e->cc_vars);
            if (got.K != e->K) VW2__RG_BAD("kind-3 N=%d: ran %lld != K %lld", e->N, (long long)got.K, (long long)e->K);
            if (e->ns > 0.0 && (got.ns - e->ns > 0.05 || e->ns - got.ns > 0.05))
                VW2__RG_BAD("kind-3 N=%d: ns mismatch", e->N);
            checked++;
        }
        else if (e->kind == VFFT_OOP_KIND_ZSPLIT) {
            if (e->N < 2048) {   /* reader-law inert rows must miss */
                if (vw2_oop_lookup_zsplit(&st, e->N, &got))
                    VW2__RG_BAD("kind-4 N=%d: sub-2048 wrong-slot row RESOLVED", e->N);
                checked++;
                continue;
            }
            if (!vw2_oop_lookup_zsplit(&st, e->N, &got)) { VW2__RG_BAD("kind-4 N=%d: MISSED", e->N); continue; }
            if (got.zs_route != e->zs_route || got.zs_t2q != e->zs_t2q || got.zt_t2q != e->zt_t2q)
                VW2__RG_BAD("kind-4 N=%d: route/t2q mismatch", e->N);
            if (got.zt_tw != e->zt_tw || (e->zt_tw > 0 && got.zt_l1 != e->zt_l1))
                VW2__RG_BAD("kind-4 N=%d: width pair mismatch", e->N);
            if (got.cc_chain != e->cc_chain) VW2__RG_BAD("kind-4 N=%d: cc_chain %d != %d", e->N, got.cc_chain, e->cc_chain);
            if (e->ns > 0.0 && (got.ns - e->ns > 0.05 || e->ns - got.ns > 0.05))
                VW2__RG_BAD("kind-4 N=%d: ns mismatch", e->N);
            checked++;
        }
        else if (e->kind == VFFT_OOP_KIND_ZR2C) {
            int kv = 0;
            if (!vw2_oop_lookup_zr2c(&st, e->N, &kv)) {
                if (e->zr_kv) VW2__RG_BAD("kind-5 N=%d: MISSED", e->N);
            } else if (kv != e->zr_kv) {
                VW2__RG_BAD("kind-5 N=%d: zr_kv %d != %d", e->N, kv, e->zr_kv);
            }
            checked++;
        }
    }
#undef VW2__RG_BAD
    vw2_close(&st);
    fprintf(stderr, "[reader-gate] %d cell(s) checked, %d mismatch(es) — %s\n",
            checked, bad, bad ? "FAIL" : "ALL PASS");
    if (checked == 0) { fprintf(stderr, "[reader-gate] VACUOUS (0 cells) — FAIL\n"); return 1; }
    return bad;
}

/* ------------------------------------------------------ the wave-0 gate */

/* migrate -> accounting -> re-migrate -> byte-identity -> verify.
 * Returns 0 = ALL PASS. */
static inline int vw2_migrate_oop_gate(const char *legacy_path, const char *outdir)
{
    vw2_mig_stats_t s1, s2;
    static char b1[262144], b2[262144];
    char path[700];
    int i, fail = 0;
    long n1, n2;

    if (vw2_migrate_oop(legacy_path, outdir, &s1, 1) != VW2_OK) {
        fprintf(stderr, "[mig-gate] FIRST RUN FAILED\n");
        return 1;
    }
    /* non-vacuous: the persisted store must hold exactly what was counted */
    {
        vw2_store_t chk;
        vw2_open(&chk, outdir, 0);
        if (chk.nrec != s1.records_out) {
            fprintf(stderr, "[mig-gate] PERSISTENCE MISMATCH: %d records counted, %d on disk\n",
                    s1.records_out, chk.nrec);
            fail++;
        }
        if (s1.migrated == 0 && s1.lines_total > s1.skipped) {
            fprintf(stderr, "[mig-gate] SUSPICIOUS: zero rows migrated from a non-empty file\n");
            fail++;
        }
        vw2_close(&chk);
    }
    /* quarantine really persisted: count @quarantined lines */
    {
        FILE *qf;
        char *ql;
        int oomq = 0, nq = 0;
        snprintf(path, sizeof path, "%s/wisdom2_quarantine.txt", outdir);
        qf = fopen(path, "rb");
        if (qf) {
            while ((ql = vw2__readline(qf, &oomq)) != NULL) {
                if (!strncmp(ql, "@quarantined ", 13)) nq++;
                free(ql);
            }
            fclose(qf);
        }
        if (nq != s1.quarantined) {
            fprintf(stderr, "[mig-gate] QUARANTINE MISMATCH: %d counted, %d persisted\n",
                    s1.quarantined, nq);
            fail++;
        }
    }
    if (vw2_migrate_oop(legacy_path, outdir, &s2, 0) != VW2_OK) {
        fprintf(stderr, "[mig-gate] SECOND RUN FAILED\n");
        return 1;
    }
    for (i = 0; i < VW2_NSHARDS; i++) {
        FILE *f;
        snprintf(path, sizeof path, "%s/%s", outdir, vw2_shard_name[i]);
        f = fopen(path, "rb");
        if (!f) continue;
        n1 = (long)fread(b1, 1, sizeof b1 - 1, f); b1[n1] = 0; fclose(f);
        /* second run already happened; compare against a third save cycle
         * is unnecessary — idempotency == run2 output equals run1 output,
         * and run2 wrote over run1, so re-read equals run2. Instead compare
         * run1 vs run2 by re-running the accounting identity: */
        (void)n1;
    }
    /* run2 re-banks over existing keys, so records_out (NEW keys) is 0 by
     * design there; the idempotency claim is on line classification. */
    if (s1.migrated != s2.migrated || s1.quarantined != s2.quarantined) {
        fprintf(stderr, "[mig-gate] IDEMPOTENCY FAILED: run1 %d/%d vs run2 %d/%d\n",
                s1.migrated, s1.quarantined, s2.migrated, s2.quarantined);
        fail++;
    }
    /* byte idempotency: snapshot after run2, run a third time, compare */
    {
        static char snap[VW2_NSHARDS][262144];
        long len[VW2_NSHARDS];
        vw2_mig_stats_t s3;
        for (i = 0; i < VW2_NSHARDS; i++) {
            FILE *f;
            len[i] = -1;
            snprintf(path, sizeof path, "%s/%s", outdir, vw2_shard_name[i]);
            f = fopen(path, "rb");
            if (!f) continue;
            len[i] = (long)fread(snap[i], 1, sizeof snap[i] - 1, f);
            snap[i][len[i]] = 0;
            fclose(f);
        }
        if (vw2_migrate_oop(legacy_path, outdir, &s3, 0) != VW2_OK) { fprintf(stderr, "[mig-gate] THIRD RUN FAILED\n"); return 1; }
        for (i = 0; i < VW2_NSHARDS; i++) {
            FILE *f;
            if (len[i] < 0) continue;
            snprintf(path, sizeof path, "%s/%s", outdir, vw2_shard_name[i]);
            f = fopen(path, "rb");
            if (!f) { fail++; continue; }
            n2 = (long)fread(b2, 1, sizeof b2 - 1, f); b2[n2] = 0; fclose(f);
            if (n2 != len[i] || memcmp(snap[i], b2, (size_t)n2)) {
                fprintf(stderr, "[mig-gate] BYTE IDEMPOTENCY FAILED: %s\n", vw2_shard_name[i]);
                fail++;
            }
        }
    }
    if (vw2_migrate_oop_verify(legacy_path, outdir) != 0) fail++;
    if (vw2_migrate_oop_reader_gate(legacy_path, outdir) != 0) fail++;
    fprintf(stderr, "[mig-gate] %s\n", fail ? "FAIL" : "ALL PASS");
    return fail;
}

/* ════════════════════════════════════════════════════════════════════════
 * WAVE 3 — the 2D families (fft2d c2c incl. @nat2d, r2c, c2r).
 * Same machinery: probe-parse every data line through the SHIPPED loaders
 * (exact line attribution; legacy silent drops become quarantine rows),
 * records built by the SHARED family codec (wisdom2_2d_reader.h), banked
 * through the generic seen-set/merge law. No seed class in this family;
 * fft3d migrates NOTHING (born in wisdom2 — zero disk instances).
 * Line classing: blank/'#' = skipped; '@'-lines are HEADERS unless they
 * begin with "@nat2d" (the natural table's data rows).
 * ══════════════════════════════════════════════════════════════════════ */

#include "wisdom2_2d_reader.h"

/* probe-parse one fft2d_c2c line. 1 = scrambled entry (*e), 2 = natural
 * entry (*ne), 0 = legacy loader dropped it, -1 = probe IO failure. */
static inline int vw2__mig2d_parse_c2c(const char *outdir, const char *line,
                                       vfft_fft2d_c2c_wisdom_entry_t *e,
                                       vfft_fft2d_c2c_nat_entry_t *ne)
{
    char probe[640];
    FILE *f;
    vfft_fft2d_c2c_wisdom_t w;
    int rc = 0;
    snprintf(probe, sizeof probe, "%s/mig_probe.tmp", outdir);
    f = fopen(probe, "wb");
    if (!f) return -1;
    if (fprintf(f, "%s\n", line) < 0) { fclose(f); remove(probe); return -1; }
    if (fclose(f) != 0) { remove(probe); return -1; }
    if (vfft_fft2d_c2c_wisdom_load(&w, probe) != 0) { remove(probe); return -1; }
    remove(probe);
    if (w.count == 1)          { *e  = w.entries[0]; rc = 1; }
    else if (w.nat_count == 1) { *ne = w.nat[0];     rc = 2; }
    vfft_fft2d_c2c_wisdom_free(&w);
    return rc;
}

/* probe-parse one fft2d_r2c/c2r line. 1 = entry, 0 = dropped, -1 = IO. */
static inline int vw2__mig2d_parse_r2c(const char *outdir, const char *line,
                                       vfft_fft2d_r2c_wisdom_entry_t *e)
{
    char probe[640];
    FILE *f;
    vfft_fft2d_r2c_wisdom_t w;
    int rc = 0;
    snprintf(probe, sizeof probe, "%s/mig_probe.tmp", outdir);
    f = fopen(probe, "wb");
    if (!f) return -1;
    if (fprintf(f, "%s\n", line) < 0) { fclose(f); remove(probe); return -1; }
    if (fclose(f) != 0) { remove(probe); return -1; }
    if (vfft_fft2d_r2c_wisdom_load(&w, probe) != 0) { remove(probe); return -1; }
    remove(probe);
    if (w.count == 1) { *e = w.entries[0]; rc = 1; }
    vfft_fft2d_r2c_wisdom_free(&w);
    return rc;
}

/* skipped = blank, '#', or an '@' header that is not @nat2d data */
static inline int vw2__mig2d_skipped(const char *line)
{
    const char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') return 1;
    if (*p == '@') return strncmp(p, "@nat2d", 6) != 0;
    return 0;
}

/* migrate ONE 2D file. kind: 0 = c2c (both tables), 1 = r2c, 2 = c2r. */
static inline int vw2__mig2d_file(vw2_store_t *st, vw2__mig_seen_t *seen,
                                  vw2_mig_stats_t *stats,
                                  const char *legacy_path, int kind,
                                  const char *outdir, int verbose)
{
    FILE *f = fopen(legacy_path, "rb");
    char line[4096], base[128];
    int lineno = 0;
    const char *bn = strrchr(legacy_path, '/');
    const char *bn2 = strrchr(legacy_path, '\\');
    if (bn2 > bn) bn = bn2;
    snprintf(base, sizeof base, "%s", bn ? bn + 1 : legacy_path);
    if (!f) {
        fprintf(stderr, "[wisdom2_migrate] cannot open %s\n", legacy_path);
        return -1;
    }
    while (fgets(line, sizeof line, f)) {
        char from[192];
        const char *why = NULL;
        vw2_rec_t rec;
        int pr;
        size_t L = strlen(line);
        lineno++;
        while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
        if (vw2__mig2d_skipped(line)) { stats->skipped++; continue; }
        snprintf(from, sizeof from, "%s:%d", base, lineno);
        if (kind == 0) {
            vfft_fft2d_c2c_wisdom_entry_t e;
            vfft_fft2d_c2c_nat_entry_t ne;
            pr = vw2__mig2d_parse_c2c(outdir, line, &e, &ne);
            if (pr < 0) { fclose(f); return -1; }             /* IO = FATAL */
            if (pr == 0) {
                if (vw2__mig_quar(st, stats, "legacy-silent-drop", from, line)) { fclose(f); return -1; }
                continue;
            }
            if (pr == 1) {
                if (vw2_2d_c2c_rec_from_entry(&rec, &e, "migrated", from, &why)) {
                    if (vw2__mig_quar(st, stats, why ? why : "codec-refused", from, line)) { fclose(f); return -1; }
                    continue;
                }
            } else {
                if (vw2_2d_c2c_rec_from_nat(&rec, &ne, "migrated", from, &why)) {
                    if (vw2__mig_quar(st, stats, why ? why : "codec-refused", from, line)) { fclose(f); return -1; }
                    continue;
                }
            }
        } else {
            vfft_fft2d_r2c_wisdom_entry_t e;
            pr = vw2__mig2d_parse_r2c(outdir, line, &e);
            if (pr < 0) { fclose(f); return -1; }
            if (pr == 0) {
                if (vw2__mig_quar(st, stats, "legacy-silent-drop", from, line)) { fclose(f); return -1; }
                continue;
            }
            if (vw2_2d_r2c_rec_from_entry(&rec, &e, kind == 2, "migrated", from, &why)) {
                if (vw2__mig_quar(st, stats, why ? why : "codec-refused", from, line)) { fclose(f); return -1; }
                continue;
            }
        }
        {
            int b = vw2__mig_bank(st, seen, &rec, &why);
            if (b < 0) {
                if (vw2__mig_quar(st, stats, why ? why : "bank-refused", from, line)) { fclose(f); return -1; }
                continue;
            }
            if (b > 0) stats->records_out++;
            stats->migrated++;
            if (verbose)
                fprintf(stderr, "[wisdom2_migrate] %s migrated\n", from);
        }
    }
    fclose(f);
    return 0;
}

/* Migrate the three 2D files into outdir's store. Any path may be NULL
 * (file absent on a tree = zero lines, not an error). */
static inline int vw2_migrate_2d(const char *c2c_path, const char *r2c_path,
                                 const char *c2r_path, const char *outdir,
                                 vw2_mig_stats_t *stats, int verbose)
{
    vw2_store_t st;
    vw2__mig_seen_t seen;
    int rc = 0;
    memset(stats, 0, sizeof *stats);
    memset(&seen, 0, sizeof seen);
    VW2__MIG_MKDIR(outdir);
    vw2_open(&st, outdir, 1);
    if (c2c_path && vw2__mig2d_file(&st, &seen, stats, c2c_path, 0, outdir, verbose)) rc = -1;
    if (!rc && r2c_path && vw2__mig2d_file(&st, &seen, stats, r2c_path, 1, outdir, verbose)) rc = -1;
    if (!rc && c2r_path && vw2__mig2d_file(&st, &seen, stats, c2r_path, 2, outdir, verbose)) rc = -1;
    if (!rc && vw2_save(&st) != VW2_OK) { stats->io_errors++; rc = -1; }
    vw2_close(&st);
    free(seen.k);
    fprintf(stderr, "[wisdom2_migrate] 2d: %d skipped + %d migrated + %d quarantined"
                    " -> %d new record(s)\n",
            stats->skipped, stats->migrated, stats->quarantined, stats->records_out);
    return rc;
}

/* Reader gate, twin-level: every legacy-servable 2D cell resolves
 * FIELD-IDENTICAL through the vw2 twins (both structs memset -> whole-
 * struct memcmp is padding-safe). Non-vacuous: zero cells = FAIL. */
static inline int vw2_migrate_2d_reader_gate(const char *c2c_path,
                                             const char *r2c_path,
                                             const char *c2r_path,
                                             const char *outdir)
{
    vw2_store_t st;
    int cells = 0, bad = 0;
    vw2_open(&st, outdir, 0);
    if (c2c_path) {
        vfft_fft2d_c2c_wisdom_t w;
        size_t i;
        if (vfft_fft2d_c2c_wisdom_load(&w, c2c_path) == 0) {
            for (i = 0; i < w.count; i++) {
                vfft_fft2d_c2c_wisdom_entry_t got;
                memset(&got, 0, sizeof got);
                cells++;
                if (!vw2_2d_c2c_lookup_scr(&st, w.entries[i].N1, w.entries[i].N2, &got) ||
                    memcmp(&got, &w.entries[i], sizeof got)) {
                    fprintf(stderr, "[reader-gate-2d] c2c scr %dx%d mismatch\n",
                            w.entries[i].N1, w.entries[i].N2);
                    bad++;
                }
            }
            for (i = 0; i < w.nat_count; i++) {
                vfft_fft2d_c2c_nat_entry_t got;
                memset(&got, 0, sizeof got);
                cells++;
                if (!vw2_2d_c2c_lookup_nat(&st, w.nat[i].N1, w.nat[i].N2, &got) ||
                    memcmp(&got, &w.nat[i], sizeof got)) {
                    fprintf(stderr, "[reader-gate-2d] c2c nat %dx%d mismatch\n",
                            w.nat[i].N1, w.nat[i].N2);
                    bad++;
                }
            }
            vfft_fft2d_c2c_wisdom_free(&w);
        }
    }
    {
        const char *rp[2]; int isc2r;
        rp[0] = r2c_path; rp[1] = c2r_path;
        for (isc2r = 0; isc2r < 2; isc2r++) {
            vfft_fft2d_r2c_wisdom_t w;
            size_t i;
            if (!rp[isc2r]) continue;
            if (vfft_fft2d_r2c_wisdom_load(&w, rp[isc2r]) != 0) continue;
            for (i = 0; i < w.count; i++) {
                vfft_fft2d_r2c_wisdom_entry_t got;
                memset(&got, 0, sizeof got);
                cells++;
                if (!vw2_2d_r2c_lookup(&st, isc2r, w.entries[i].N1, w.entries[i].N2, &got) ||
                    memcmp(&got, &w.entries[i], sizeof got)) {
                    fprintf(stderr, "[reader-gate-2d] %s %dx%d mismatch\n",
                            isc2r ? "c2r" : "r2c", w.entries[i].N1, w.entries[i].N2);
                    bad++;
                }
            }
            vfft_fft2d_r2c_wisdom_free(&w);
        }
    }
    vw2_close(&st);
    if (cells == 0) { fprintf(stderr, "[reader-gate-2d] VACUOUS (0 cells) — FAIL\n"); return 1; }
    fprintf(stderr, "[reader-gate-2d] %d cell(s) checked, %d mismatch(es) — %s\n",
            cells, bad, bad ? "FAIL" : "ALL PASS");
    return bad ? 1 : 0;
}

/* Full 2D migration gate: accounting + persisted counts + byte
 * idempotency x3 + the reader gate. */
static inline int vw2_migrate_2d_gate(const char *c2c_path, const char *r2c_path,
                                      const char *c2r_path, const char *outdir)
{
    vw2_mig_stats_t st1, st2, st3;
    int fail = 0;
    long base_nrec;
    {
        vw2_store_t s;
        vw2_open(&s, outdir, 0);
        base_nrec = s.nrec;
        vw2_close(&s);
    }
    if (vw2_migrate_2d(c2c_path, r2c_path, c2r_path, outdir, &st1, 0)) fail++;
    {   /* persisted count: reload and require base + records_out */
        vw2_store_t s;
        vw2_open(&s, outdir, 0);
        if (s.nrec != base_nrec + st1.records_out) {
            fprintf(stderr, "[mig-gate-2d] PERSISTED COUNT %d != %ld+%d\n",
                    s.nrec, base_nrec, st1.records_out);
            fail++;
        }
        vw2_close(&s);
    }
    {   /* byte idempotency x3 on the 2d shard */
        char path[640], b1[65536], b2[65536];
        long n1 = 0, n2;
        FILE *f;
        int r;
        snprintf(path, sizeof path, "%s/%s", outdir, vw2_shard_name[VW2_SHARD_2D]);
        f = fopen(path, "rb");
        if (f) { n1 = (long)fread(b1, 1, sizeof b1 - 1, f); fclose(f); }
        for (r = 0; r < 2; r++) {
            vw2_mig_stats_t *s = r ? &st3 : &st2;
            if (vw2_migrate_2d(c2c_path, r2c_path, c2r_path, outdir, s, 0)) fail++;
            if (s->migrated != st1.migrated || s->quarantined != st1.quarantined) {
                fprintf(stderr, "[mig-gate-2d] RUN %d ACCOUNTING DRIFT\n", r + 2);
                fail++;
            }
            f = fopen(path, "rb");
            if (!f) { fail++; continue; }
            n2 = (long)fread(b2, 1, sizeof b2 - 1, f); fclose(f);
            if (n2 != n1 || memcmp(b1, b2, (size_t)n2)) {
                fprintf(stderr, "[mig-gate-2d] BYTE IDEMPOTENCY FAILED (run %d)\n", r + 2);
                fail++;
            }
        }
    }
    if (vw2_migrate_2d_reader_gate(c2c_path, r2c_path, c2r_path, outdir)) fail++;
    fprintf(stderr, "[mig-gate-2d] %s\n", fail ? "FAIL" : "ALL PASS");
    return fail;
}

/* ════════════════════════════════════════════════════════════════════════
 * WAVE 4 — the stride family (spike_wisdom.txt: THREE tables; rfft file;
 * the @version-6 padded fossil quarantined whole). Probe-parse through the
 * SHIPPED v8 loader; records via the SHARED family codec
 * (wisdom2_stride_reader.h). Line classing: blank/'#' skipped; '@'-lines
 * are headers unless they begin "@nat" / "@natoop".
 * ══════════════════════════════════════════════════════════════════════ */

#include "wisdom2_stride_reader.h"

/* probe-parse one spike/rfft line. 1 = scrambled (*e), 2 = @nat (*ne),
 * 3 = @natoop (*ne), 0 = legacy loader dropped it, -1 = probe IO. */
static inline int vw2__migst_parse(const char *outdir, const char *line,
                                   vfft_proto_wisdom_entry_t *e,
                                   vfft_proto_nat_entry_t *ne)
{
    char probe[640];
    FILE *f;
    vfft_proto_wisdom_t w;
    int rc = 0;
    snprintf(probe, sizeof probe, "%s/mig_probe.tmp", outdir);
    f = fopen(probe, "wb");
    if (!f) return -1;
    if (fprintf(f, "%s\n", line) < 0) { fclose(f); remove(probe); return -1; }
    if (fclose(f) != 0) { remove(probe); return -1; }
    if (vfft_proto_wisdom_load(&w, probe) != 0) { remove(probe); return -1; }
    remove(probe);
    if (w.count == 1)             { *e  = w.entries[0]; rc = 1; }
    else if (w.nat_count == 1)    { *ne = w.nat[0];     rc = 2; }
    else if (w.natoop_count == 1) { *ne = w.natoop[0];  rc = 3; }
    vfft_proto_wisdom_free(&w);
    return rc;
}

static inline int vw2__migst_skipped(const char *line)
{
    const char *p = line;
    while (*p == ' ' || *p == '\t') p++;
    if (*p == '\0' || *p == '\n' || *p == '\r' || *p == '#') return 1;
    if (*p == '@')
        return strncmp(p, "@nat", 4) != 0;   /* covers @nat AND @natoop */
    return 0;
}

/* migrate ONE stride file. is_rfft keys scrambled rows t=r2c. */
static inline int vw2__migst_file(vw2_store_t *st, vw2__mig_seen_t *seen,
                                  vw2_mig_stats_t *stats,
                                  const char *legacy_path, int is_rfft,
                                  const char *outdir, int verbose)
{
    FILE *f = fopen(legacy_path, "rb");
    char line[4096], base[128];
    int lineno = 0;
    const char *bn = strrchr(legacy_path, '/');
    const char *bn2 = strrchr(legacy_path, '\\');
    if (bn2 > bn) bn = bn2;
    snprintf(base, sizeof base, "%s", bn ? bn + 1 : legacy_path);
    if (!f) {
        fprintf(stderr, "[wisdom2_migrate] cannot open %s\n", legacy_path);
        return -1;
    }
    while (fgets(line, sizeof line, f)) {
        char from[192];
        const char *why = NULL;
        vw2_rec_t rec;
        vfft_proto_wisdom_entry_t e;
        vfft_proto_nat_entry_t ne;
        int pr, b;
        size_t L = strlen(line);
        lineno++;
        while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
        if (vw2__migst_skipped(line)) { stats->skipped++; continue; }
        snprintf(from, sizeof from, "%s:%d", base, lineno);
        pr = vw2__migst_parse(outdir, line, &e, &ne);
        if (pr < 0) { fclose(f); return -1; }               /* IO = FATAL */
        if (pr == 0) {
            if (vw2__mig_quar(st, stats, "legacy-silent-drop", from, line)) { fclose(f); return -1; }
            continue;
        }
        if (pr == 1) {
            if (vw2_stride_rec_from_entry(&rec, &e, is_rfft, "migrated", from, &why)) {
                if (vw2__mig_quar(st, stats, why ? why : "codec-refused", from, line)) { fclose(f); return -1; }
                continue;
            }
        } else {
            if (vw2_stride_rec_from_nat(&rec, &ne,
                                        pr == 3 ? VW2_PL_OOP : VW2_PL_IP,
                                        "migrated", from, &why)) {
                if (vw2__mig_quar(st, stats, why ? why : "codec-refused", from, line)) { fclose(f); return -1; }
                continue;
            }
        }
        b = vw2__mig_bank(st, seen, &rec, &why);
        if (b < 0) {
            if (vw2__mig_quar(st, stats, why ? why : "bank-refused", from, line)) { fclose(f); return -1; }
            continue;
        }
        if (b > 0) stats->records_out++;
        stats->migrated++;
        if (verbose)
            fprintf(stderr, "[wisdom2_migrate] %s migrated\n", from);
    }
    fclose(f);
    return 0;
}

/* quarantine EVERY data line of a fossil file (never parsed for records) */
static inline int vw2__migst_fossil(vw2_store_t *st, vw2_mig_stats_t *stats,
                                    const char *path)
{
    FILE *f = fopen(path, "rb");
    char line[4096], base[128], from[192];
    int lineno = 0;
    const char *bn = strrchr(path, '/');
    const char *bn2 = strrchr(path, '\\');
    if (bn2 > bn) bn = bn2;
    snprintf(base, sizeof base, "%s", bn ? bn + 1 : path);
    if (!f) return 0;                        /* absent fossil: nothing owed */
    while (fgets(line, sizeof line, f)) {
        size_t L = strlen(line);
        lineno++;
        while (L && (line[L-1] == '\n' || line[L-1] == '\r')) line[--L] = 0;
        if (vw2__migst_skipped(line)) { stats->skipped++; continue; }
        snprintf(from, sizeof from, "%s:%d", base, lineno);
        if (vw2__mig_quar(st, stats, "superseded-fossil", from, line)) { fclose(f); return -1; }
    }
    fclose(f);
    return 0;
}

/* Migrate spike + rfft (+ the padded fossil, quarantined). NULL = absent. */
static inline int vw2_migrate_stride(const char *spike_path, const char *rfft_path,
                                     const char *padded_path, const char *outdir,
                                     vw2_mig_stats_t *stats, int verbose)
{
    vw2_store_t st;
    vw2__mig_seen_t seen;
    int rc = 0;
    memset(stats, 0, sizeof *stats);
    memset(&seen, 0, sizeof seen);
    VW2__MIG_MKDIR(outdir);
    vw2_open(&st, outdir, 1);
    if (spike_path && vw2__migst_file(&st, &seen, stats, spike_path, 0, outdir, verbose)) rc = -1;
    if (!rc && rfft_path && vw2__migst_file(&st, &seen, stats, rfft_path, 1, outdir, verbose)) rc = -1;
    if (!rc && padded_path && vw2__migst_fossil(&st, stats, padded_path)) rc = -1;
    if (!rc && vw2_save(&st) != VW2_OK) { stats->io_errors++; rc = -1; }
    vw2_close(&st);
    free(seen.k);
    fprintf(stderr, "[wisdom2_migrate] stride: %d skipped + %d migrated + %d quarantined"
                    " -> %d new record(s)\n",
            stats->skipped, stats->migrated, stats->quarantined, stats->records_out);
    return rc;
}

/* Reader gate: every legacy-servable stride cell resolves FIELD-IDENTICAL
 * through the vw2 twins (memset both sides -> whole-struct memcmp). */
static inline int vw2_migrate_stride_reader_gate(const char *spike_path,
                                                 const char *rfft_path,
                                                 const char *outdir)
{
    vw2_store_t st;
    int cells = 0, bad = 0;
    vw2_open(&st, outdir, 0);
    {
        const char *fp[2]; int is_rfft;
        fp[0] = spike_path; fp[1] = rfft_path;
        for (is_rfft = 0; is_rfft < 2; is_rfft++) {
            vfft_proto_wisdom_t w;
            size_t i;
            if (!fp[is_rfft]) continue;
            if (vfft_proto_wisdom_load(&w, fp[is_rfft]) != 0) continue;
            /* THE LAW (wave-1 precedent): the twin must equal what LEGACY
             * WOULD SERVE — the first-match lookup, never the raw table
             * row (spike carries intra-file duplicates); codec-refused
             * rows (junk cells) must MISS. */
            for (i = 0; i < w.count; i++) {
                vfft_proto_wisdom_entry_t got;
                const vfft_proto_wisdom_entry_t *served =
                    vfft_proto_wisdom_lookup(&w, w.entries[i].N, w.entries[i].K);
                vw2_rec_t rtmp;
                const char *why = NULL;
                int refused =
                    vw2_stride_rec_from_entry(&rtmp, served, is_rfft,
                                              "migrated", "gate", &why) != 0;
                if (!refused) vw2_rec_free(&rtmp);
                memset(&got, 0, sizeof got);
                cells++;
                if (refused) {
                    if (vw2_stride_lookup(&st, is_rfft, served->N, served->K, &got)) {
                        fprintf(stderr, "[reader-gate-stride] %s N=%d K=%zu: refused row RESOLVED\n",
                                is_rfft ? "rfft" : "scr", served->N, served->K);
                        bad++;
                    }
                    continue;
                }
                if (!vw2_stride_lookup(&st, is_rfft, served->N, served->K, &got) ||
                    memcmp(&got, served, sizeof got)) {
                    fprintf(stderr, "[reader-gate-stride] %s N=%d K=%zu mismatch\n",
                            is_rfft ? "rfft" : "scr", served->N, served->K);
                    bad++;
                }
            }
            for (i = 0; i < w.nat_count; i++) {
                vfft_proto_nat_entry_t got;
                const vfft_proto_nat_entry_t *served =
                    vfft_proto_nat_lookup(&w, w.nat[i].N, w.nat[i].K);
                memset(&got, 0, sizeof got);
                cells++;
                if (!vw2_stride_lookup_nat(&st, served->N, served->K, &got) ||
                    memcmp(&got, served, sizeof got)) {
                    fprintf(stderr, "[reader-gate-stride] nat N=%d K=%zu mismatch\n",
                            served->N, served->K);
                    bad++;
                }
            }
            for (i = 0; i < w.natoop_count; i++) {
                vfft_proto_nat_entry_t got;
                const vfft_proto_nat_entry_t *served =
                    vfft_proto_natoop_lookup(&w, w.natoop[i].N, w.natoop[i].K);
                memset(&got, 0, sizeof got);
                cells++;
                if (!vw2_stride_lookup_natoop(&st, served->N, served->K, &got) ||
                    memcmp(&got, served, sizeof got)) {
                    fprintf(stderr, "[reader-gate-stride] natoop N=%d K=%zu mismatch\n",
                            served->N, served->K);
                    bad++;
                }
            }
            vfft_proto_wisdom_free(&w);
        }
    }
    vw2_close(&st);
    if (cells == 0) { fprintf(stderr, "[reader-gate-stride] VACUOUS (0 cells) — FAIL\n"); return 1; }
    fprintf(stderr, "[reader-gate-stride] %d cell(s) checked, %d mismatch(es) — %s\n",
            cells, bad, bad ? "FAIL" : "ALL PASS");
    return bad ? 1 : 0;
}

/* Full stride migration gate: accounting + persisted counts + byte
 * idempotency x3 + the reader gate (both touched shards watched). */
static inline int vw2_migrate_stride_gate(const char *spike_path, const char *rfft_path,
                                          const char *padded_path, const char *outdir)
{
    vw2_mig_stats_t st1, st2, st3;
    int fail = 0;
    long base_nrec;
    {
        vw2_store_t s;
        vw2_open(&s, outdir, 0);
        base_nrec = s.nrec;
        vw2_close(&s);
    }
    if (vw2_migrate_stride(spike_path, rfft_path, padded_path, outdir, &st1, 0)) fail++;
    {
        vw2_store_t s;
        vw2_open(&s, outdir, 0);
        if (s.nrec != base_nrec + st1.records_out) {
            fprintf(stderr, "[mig-gate-stride] PERSISTED COUNT %d != %ld+%d\n",
                    s.nrec, base_nrec, st1.records_out);
            fail++;
        }
        vw2_close(&s);
    }
    {
        char p1[640], p2[640], b1[262144], b2[262144];
        long n1a = 0, n1b = 0, n2;
        FILE *f;
        int r;
        snprintf(p1, sizeof p1, "%s/%s", outdir, vw2_shard_name[VW2_SHARD_STRIDE]);
        snprintf(p2, sizeof p2, "%s/%s", outdir, vw2_shard_name[VW2_SHARD_REAL]);
        f = fopen(p1, "rb");
        if (f) { n1a = (long)fread(b1, 1, sizeof b1 / 2, f); fclose(f); }
        f = fopen(p2, "rb");
        if (f) { n1b = (long)fread(b1 + n1a, 1, sizeof b1 / 2, f); fclose(f); }
        for (r = 0; r < 2; r++) {
            vw2_mig_stats_t *sr = r ? &st3 : &st2;
            long m1 = 0, m2 = 0;
            if (vw2_migrate_stride(spike_path, rfft_path, padded_path, outdir, sr, 0)) fail++;
            if (sr->migrated != st1.migrated || sr->quarantined != st1.quarantined) {
                fprintf(stderr, "[mig-gate-stride] RUN %d ACCOUNTING DRIFT\n", r + 2);
                fail++;
            }
            f = fopen(p1, "rb");
            if (f) { m1 = (long)fread(b2, 1, sizeof b2 / 2, f); fclose(f); }
            f = fopen(p2, "rb");
            if (f) { m2 = (long)fread(b2 + m1, 1, sizeof b2 / 2, f); fclose(f); }
            n2 = m1 + m2;
            if (n2 != n1a + n1b || memcmp(b1, b2, (size_t)n2)) {
                fprintf(stderr, "[mig-gate-stride] BYTE IDEMPOTENCY FAILED (run %d)\n", r + 2);
                fail++;
            }
        }
    }
    if (vw2_migrate_stride_reader_gate(spike_path, rfft_path, outdir)) fail++;
    fprintf(stderr, "[mig-gate-stride] %s\n", fail ? "FAIL" : "ALL PASS");
    return fail;
}

/* One-shot v1.0 -> v1.1 re-key: every kind-3 record (eng=k1) moves to
 * role=comp (owner decision A, 2026-08-20 — kind-3 is the K=1 engine's
 * component recipe; its old role-absent key collided with the stride
 * family's @natoop problem verdict). Idempotent: already-role=comp rows
 * are untouched. Returns the number re-keyed, or -1. */
static inline int vw2_migrate_rekey_k1role(const char *dir)
{
    vw2_store_t st;
    char path[640], bak[720];
    int i, n = 0, rc;
    vw2_open(&st, dir, 1);
    for (i = 0; i < st.nrec; i++) {
        const char *eng = vw2_rec_get(&st.rec[i], "eng");
        if (!eng || strcmp(eng, "k1")) continue;
        if (st.rec[i].key.role == VW2_ROLE_COMP) continue;
        st.rec[i].key.role = VW2_ROLE_COMP;
        st.dirty[st.rec[i].shard] = 1;
        n++;
    }
    if (!n) {
        vw2_close(&st);
        fprintf(stderr, "[wisdom2_migrate] rekey-k1role: nothing to re-key in %s\n", dir);
        return 0;
    }
    /* A rekey moves record IDENTITY, which merge-on-save cannot see (the
     * old-key disk row would be carried back in beside the new one). One-
     * shot discipline: back the shard up, remove it so the merge base is
     * empty, save pure memory state (atomic tmp+rename), then drop the
     * backup — or restore it on any failure. */
    snprintf(path, sizeof path, "%s/%s", dir, vw2_shard_name[VW2_SHARD_OOP]);
    snprintf(bak, sizeof bak, "%s.rekey.bak", path);
    remove(bak);
    if (rename(path, bak) != 0) { vw2_close(&st); return -1; }
    st.dirty[VW2_SHARD_OOP] = 1;
    rc = vw2_save(&st);
    if (rc != VW2_OK) {
        remove(path);
        if (rename(bak, path) != 0)
            fprintf(stderr, "[wisdom2_migrate] rekey-k1role: RESTORE FAILED — "
                            "recover %s manually\n", bak);
        vw2_close(&st);
        return -1;
    }
    remove(bak);
    vw2_close(&st);
    fprintf(stderr, "[wisdom2_migrate] rekey-k1role: %d record(s) -> role=comp in %s\n",
            n, dir);
    return n;
}

#endif /* VFFT_WISDOM2_MIGRATE_H */
