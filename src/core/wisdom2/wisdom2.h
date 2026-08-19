/* wisdom2.h — THE VectorFFT wisdom store: the only parser and the only
 * writer of the @vw2 grammar, everywhere (library, calibrators, benches,
 * gates). Read src/core/wisdom2/README.md before changing anything here;
 * the README is the declaration of record and co-evolves with this file in
 * the same change.
 *
 * Layering: this module is STORAGE ONLY. It owns key/record/file mechanics
 * (parse, emit, lookup, bank, merge, atomic save). It never measures, never
 * builds plans, never reads env route-forces — those live in the planners
 * and the create path. The env-law TABLE below is data the create path
 * consults; the module itself takes no env-dependent decisions except the
 * wisdom-directory resolution at open.
 *
 * Grammar invariants enforced here (per README §3):
 *   - version header checked: bad magic / major > supported => file refused,
 *     one loud stderr line, per-file POISON (banking/saving disabled so a
 *     save can never clobber a file we could not read). Never silently empty.
 *   - unknown tokens and unknown records are carried VERBATIM through
 *     load -> bank -> save (no code path parses-and-truncates).
 *   - wildcards (q=* / ord=* / place=*) are legal only on migrated records
 *     (from= required); the writer refuses a fresh wildcard bank.
 *   - one dedup (full key tuple), merge rank race/migrated > env > seed,
 *     newer date among equals, cross-metric replacement refused.
 *   - saves are dirty-only, merge-on-save, atomic (tmp + MoveFileEx /
 *     rename). Stale .tmp swept at load.
 */
#ifndef VFFT_WISDOM2_H
#define VFFT_WISDOM2_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#if defined(_WIN32)
#  include <windows.h>
#  include <io.h>
#else
#  include <unistd.h>
#endif

/* ---------------------------------------------------------------- version */

#define VW2_MAGIC        "@vw2"
#define VW2_MAJOR        1
#define VW2_MINOR        0

/* ------------------------------------------------------------ enumerations */

typedef enum { /* transforms: the KEY vocabulary. Never a route (README §3.1). */
    VW2_T_NONE = 0,
    VW2_T_C2C, VW2_T_R2C, VW2_T_C2R,
    VW2_T_DCT1, VW2_T_DCT2, VW2_T_DCT3, VW2_T_DCT4,
    VW2_T_DST1, VW2_T_DST2, VW2_T_DST3, VW2_T_DST4,
    VW2_T_DHT,
    VW2_T_COUNT
} vw2_transform_t;

typedef enum { VW2_ORD_ANY = -1, VW2_ORD_NAT = 0, VW2_ORD_SCR = 1 } vw2_ord_t;
typedef enum { VW2_PL_ANY  = -1, VW2_PL_IP   = 0, VW2_PL_OOP  = 1 } vw2_pl_t;
typedef enum { VW2_DIR_NONE = 0, VW2_DIR_FWD = 1, VW2_DIR_BWD = 2 } vw2_dir_t;

/* src= provenance rank (merge law, README §4.2): race/migrated > env > seed.
 * migrated serves like race (the number WAS raced); a fresh same-metric race
 * outranks it via the tie-break below. */
typedef enum { VW2_SRC_SEED = 0, VW2_SRC_ENV = 1, VW2_SRC_MIGRATED = 2,
               VW2_SRC_RACE = 3 } vw2_src_t;

/* Field portability classes (README §4.3). */
typedef enum { VW2_FC_STRUCTURAL = 0, VW2_FC_LOCAL = 1, VW2_FC_INFO = 2 } vw2_fclass_t;

/* Error codes. 0 = ok. */
enum {
    VW2_OK = 0,
    VW2_EOPEN      = -1,  /* cannot open for read (missing file is NOT an error) */
    VW2_EVERSION   = -2,  /* bad magic / unsupported major -> file poisoned      */
    VW2_EREADONLY  = -3,  /* bank/save refused: store not writable               */
    VW2_EPOISON    = -4,  /* bank/save refused: file failed to load              */
    VW2_EWILDCARD  = -5,  /* fresh bank carried a wildcard without from=         */
    VW2_EMETRIC    = -6,  /* cross-metric/units replacement refused              */
    VW2_ERANK      = -7,  /* lower-rank record refused to replace incumbent      */
    VW2_EKEY       = -8,  /* malformed key                                       */
    VW2_EIO        = -9,  /* write/replace failure (old file left intact)        */
    VW2_ENOMEM     = -10
};

/* ------------------------------------------------------------------ key */

#define VW2_MAX_DIMS 4

typedef struct {
    uint8_t t;                    /* vw2_transform_t                          */
    uint8_t rank;                 /* 1..4 = dims of n                         */
    int32_t n[VW2_MAX_DIMS];      /* ordered extents (anisotropic law)        */
    int64_t q;                    /* requested quantity; -1 = '*' (migrated)  */
    int8_t  ord;                  /* vw2_ord_t; -1 = '*'                      */
    int8_t  pl;                   /* vw2_pl_t;  -1 = '*'                      */
    uint8_t dir;                  /* vw2_dir_t; 0 = absent (reserved axis)    */
} vw2_key_t;

/* --------------------------------------------------------------- record */

/* A record keeps its payload+measure+provenance as ordered (name,value)
 * string tokens. Known-ness matters only to the field registry; unknown
 * tokens ride along verbatim (strip-cycle armor). Records the parser cannot
 * OWN (unknown t=, unknown key token, unknown @-directive) are kept as
 * opaque raw lines and re-emitted untouched. */
typedef struct { char *name; char *val; uint8_t sect; /* 1=payload 2=meas */ } vw2_tok_t;

typedef struct {
    vw2_key_t  key;
    vw2_tok_t *tok;  int ntok, captok;
    int        shard;            /* which wisdom2_*.txt it lives in          */
} vw2_rec_t;

/* ------------------------------------------------------------------ store */

#define VW2_SHARD_OOP    0
#define VW2_SHARD_STRIDE 1
#define VW2_SHARD_REAL   2
#define VW2_SHARD_PRIME  3
#define VW2_SHARD_2D     4
#define VW2_SHARD_3D     5
#define VW2_NSHARDS      6
/* wisdom2_quarantine.txt is append-only and never loaded into the table. */

static const char *vw2_shard_name[VW2_NSHARDS] = {
    "wisdom2_oop.txt", "wisdom2_stride.txt", "wisdom2_real.txt",
    "wisdom2_prime.txt", "wisdom2_2d.txt", "wisdom2_3d.txt"
};

typedef struct {
    char       dir[512];
    vw2_rec_t *rec;  int nrec, caprec;
    char     **opaque[VW2_NSHARDS]; int nopq[VW2_NSHARDS], capopq[VW2_NSHARDS];
    uint8_t    poisoned[VW2_NSHARDS];  /* per-file: load failed => no save   */
    uint8_t    dirty[VW2_NSHARDS];
    uint8_t    present[VW2_NSHARDS];   /* file existed at load               */
    uint8_t    writable;               /* the write guard (README §2.2)      */
    char       meta[256];              /* "host=... isa=... l1d=..." or ""   */
} vw2_store_t;

/* ------------------------------------------------------- string tables */

static const char *vw2_t_name[VW2_T_COUNT] = {
    "?", "c2c", "r2c", "c2r", "dct1", "dct2", "dct3", "dct4",
    "dst1", "dst2", "dst3", "dst4", "dht"
};

/* @legend: ONE line per rule, module rule-table order, fixed string
 * constants (README §3.5). The G0 gate byte-compares regeneration. */
static const char *vw2_legend[] = {
    "signpost: a verdict references its component recipe (ref=), never copies it",
    "q-vs-ran: q= is the REQUESTED quantity; ran= is the EXECUTED batch of the timing run",
    "metric: ns= is comparable only within identical metric= and units=; absent ns = informational",
    "key: layout (split/il) is never a key token - it is a strategy output in the payload",
    "wildcards: q=*/ord=*/place=* are migration-vintage only (from= required); fresh banks stamp concrete axes",
    "evolution: new fields/axes land here as additive minor versions, never in frozen legacy files; reserved: sp_kv",
    "see src/core/wisdom2/README.md",
};
#define VW2_NLEGEND ((int)(sizeof vw2_legend / sizeof vw2_legend[0]))

/* Field registry (README §4.3): name -> portability class. Unregistered
 * fields default to INFO (safe: never decision-load-bearing until
 * registered). Retired names are never reused; dead env names live in the
 * env table below. */
typedef struct { const char *name; uint8_t fclass; } vw2_field_t;
static const vw2_field_t vw2_fields[] = {
    { "eng",     VW2_FC_STRUCTURAL }, { "route",   VW2_FC_STRUCTURAL },
    { "chain",   VW2_FC_STRUCTURAL }, { "vars",    VW2_FC_STRUCTURAL },
    { "rowplan", VW2_FC_STRUCTURAL }, { "colplan", VW2_FC_STRUCTURAL },
    { "mode",    VW2_FC_STRUCTURAL }, { "ref",     VW2_FC_STRUCTURAL },
    { "path",    VW2_FC_STRUCTURAL }, { "b",       VW2_FC_STRUCTURAL },
    { "k_pad",   VW2_FC_STRUCTURAL }, { "m",       VW2_FC_STRUCTURAL },
    { "pad_me",  VW2_FC_STRUCTURAL }, { "il_me",   VW2_FC_STRUCTURAL },
    { "zr_kv",   VW2_FC_STRUCTURAL },
    { "t2q",     VW2_FC_LOCAL }, { "kv",     VW2_FC_LOCAL },
    { "il_kv",   VW2_FC_LOCAL }, { "sp_kv",  VW2_FC_LOCAL }, /* reserved (D9) */
    { "zt_tw",   VW2_FC_LOCAL }, { "zt_l1",  VW2_FC_LOCAL },
    { "ran",     VW2_FC_INFO }, { "ns",   VW2_FC_INFO }, { "metric", VW2_FC_INFO },
    { "units",   VW2_FC_INFO }, { "arms", VW2_FC_INFO }, { "src",    VW2_FC_INFO },
    { "bin",     VW2_FC_INFO }, { "date", VW2_FC_INFO }, { "host",   VW2_FC_INFO },
    { "l1d",     VW2_FC_INFO }, { "from", VW2_FC_INFO },
};
#define VW2_NFIELDS ((int)(sizeof vw2_fields / sizeof vw2_fields[0]))

static inline vw2_fclass_t vw2_field_class(const char *name)
{
    int i;
    for (i = 0; i < VW2_NFIELDS; i++)
        if (!strcmp(vw2_fields[i].name, name)) return (vw2_fclass_t)vw2_fields[i].fclass;
    return VW2_FC_INFO;
}

/* Env-law table (README §5): DATA for the create path; the module never
 * applies these itself. shape: 1=force-never-bank, 2=force+bank-stamped,
 * 3=suppress-field, 4=wisdom-beats-env (the one inversion). */
typedef struct { const char *env; int shape; const char *field; } vw2_envlaw_t;
static const vw2_envlaw_t vw2_env_law[] = {
    { "VFFT_ZR2C_ROUTE",  1, "route" },
    { "VFFT_IL_PAD",      1, "il_me" },
    { "VFFT_SP_ROUTE",    1, "route" },
    { "VFFT_FORCE_ZROUTE",1, "route" },  /* demoted to debug switch (owner) */
    { "VFFT_NO_ZTURN",    1, "route" },  /* demoted to debug switch (owner) */
    { "VFFT_TCUT",        2, "zt_tw" },  /* also shape 3 for the same field  */
    { "VFFT_TCUT",        3, "zt_tw" },
    { "VFFT_NO_ILBLK",    4, "il_kv" },
};
#define VW2_NENVLAW ((int)(sizeof vw2_env_law / sizeof vw2_env_law[0]))
/* Dead env names — never reuse: VFFT_PROTO_WIS, VFFT_PROTO_PAD_WIS,
 * VFFT_WISDOM, VFFT_WARM, VFFT_NO_K1, VFFT_PROTO_WISDOM_OVERWRITE. */

/* --------------------------------------------------------- tiny helpers */

static inline char *vw2__strdup(const char *s)
{
    size_t n = strlen(s) + 1;
    char *p = (char *)malloc(n);
    if (p) memcpy(p, s, n);
    return p;
}

/* self-contained tokenizer (strtok_r is not portably declared on mingw):
 * splits *sp on spaces/tabs, NUL-terminates the token, advances *sp. */
static inline char *vw2__tok(char **sp)
{
    char *s = *sp, *t;
    while (*s == ' ' || *s == '\t') s++;
    if (!*s) { *sp = s; return NULL; }
    t = s;
    while (*s && *s != ' ' && *s != '\t') s++;
    if (*s) *s++ = 0;
    *sp = s;
    return t;
}

static inline void vw2__oom(void)
{
    fprintf(stderr, "[wisdom2] OUT OF MEMORY — store operation aborted\n");
}

static inline int vw2__t_parse(const char *s)
{
    int i;
    for (i = 1; i < VW2_T_COUNT; i++)
        if (!strcmp(s, vw2_t_name[i])) return i;
    return VW2_T_NONE;
}

/* n= parse: "1024" / "64x64" / "32x32x32" / "AxBxCxD". */
static inline int vw2__n_parse(const char *s, vw2_key_t *k)
{
    int rank = 0;
    const char *p = s;
    while (*p) {
        char *end;
        long v = strtol(p, &end, 10);
        if (end == p || v <= 0 || rank >= VW2_MAX_DIMS) return 0;
        k->n[rank++] = (int32_t)v;
        if (*end == 'x') p = end + 1;
        else if (*end == '\0') { p = end; }
        else return 0;
    }
    if (!rank) return 0;
    k->rank = (uint8_t)rank;
    return 1;
}

static inline void vw2__n_format(const vw2_key_t *k, char *out, size_t cap)
{
    size_t off = 0; int i;
    out[0] = 0;
    for (i = 0; i < k->rank; i++)
        off += (size_t)snprintf(out + off, cap - off, "%s%d", i ? "x" : "", k->n[i]);
}

static inline int vw2_key_eq(const vw2_key_t *a, const vw2_key_t *b)
{
    int i;
    if (a->t != b->t || a->rank != b->rank) return 0;
    for (i = 0; i < a->rank; i++) if (a->n[i] != b->n[i]) return 0;
    return a->q == b->q && a->ord == b->ord && a->pl == b->pl && a->dir == b->dir;
}

/* Does record key R serve request key REQ, allowing R's wildcards?
 * (A request never carries wildcards.) */
static inline int vw2_key_serves(const vw2_key_t *r, const vw2_key_t *req)
{
    int i;
    if (r->t != req->t || r->rank != req->rank) return 0;
    for (i = 0; i < r->rank; i++) if (r->n[i] != req->n[i]) return 0;
    if (r->q   != -1 && r->q   != req->q)   return 0;
    if (r->ord != -1 && r->ord != req->ord) return 0;
    if (r->pl  != -1 && r->pl  != req->pl)  return 0;
    if (r->dir != 0  && r->dir != req->dir) return 0;
    return 1;
}

static inline int vw2_key_has_wildcard(const vw2_key_t *k)
{
    return k->q == -1 || k->ord == -1 || k->pl == -1;
}

/* ------------------------------------------------------ record helpers */

static inline const char *vw2_rec_get(const vw2_rec_t *r, const char *name)
{
    int i;
    for (i = 0; i < r->ntok; i++)
        if (!strcmp(r->tok[i].name, name)) return r->tok[i].val;
    return NULL;
}

static inline int vw2_rec_set(vw2_rec_t *r, int sect, const char *name, const char *val)
{
    int i;
    for (i = 0; i < r->ntok; i++)
        if (!strcmp(r->tok[i].name, name)) {
            char *nv = vw2__strdup(val);
            if (!nv) { vw2__oom(); return VW2_ENOMEM; }
            free(r->tok[i].val);
            r->tok[i].val = nv;
            return VW2_OK;
        }
    if (r->ntok == r->captok) {
        int nc = r->captok ? r->captok * 2 : 8;
        vw2_tok_t *nt = (vw2_tok_t *)realloc(r->tok, (size_t)nc * sizeof *nt);
        if (!nt) { vw2__oom(); return VW2_ENOMEM; }
        r->tok = nt; r->captok = nc;
    }
    r->tok[r->ntok].name = vw2__strdup(name);
    r->tok[r->ntok].val  = vw2__strdup(val);
    r->tok[r->ntok].sect = (uint8_t)sect;
    if (!r->tok[r->ntok].name || !r->tok[r->ntok].val) { vw2__oom(); return VW2_ENOMEM; }
    r->ntok++;
    return VW2_OK;
}

static inline void vw2_rec_free(vw2_rec_t *r)
{
    int i;
    for (i = 0; i < r->ntok; i++) { free(r->tok[i].name); free(r->tok[i].val); }
    free(r->tok);
    r->tok = NULL; r->ntok = r->captok = 0;
}

static inline int vw2__src_rank(const vw2_rec_t *r)
{
    const char *s = vw2_rec_get(r, "src");
    if (!s) return VW2_SRC_RACE;              /* absent = fresh bank = race  */
    if (!strncmp(s, "env", 3))       return VW2_SRC_ENV;
    if (!strcmp(s, "seed"))          return VW2_SRC_SEED;
    if (!strcmp(s, "migrated"))      return VW2_SRC_MIGRATED;
    return VW2_SRC_RACE;
}

/* Serving rank: race and migrated serve as equals (README §3.4). */
static inline int vw2__merge_rank(const vw2_rec_t *r)
{
    int s = vw2__src_rank(r);
    return (s == VW2_SRC_MIGRATED) ? VW2_SRC_RACE : s;
}

/* ------------------------------------------------------- key parse/emit */

/* Parses the KEY section tokens. Returns 1 = owned, 0 = record must be
 * carried opaque (unknown t= or unknown key token: refuse-don't-guess,
 * README §3.1), -1 = malformed (also opaque). */
static inline int vw2__key_parse(char *sect, vw2_key_t *k)
{
    char *tok, *p = sect;
    int have_t = 0, have_n = 0, have_q = 0, have_ord = 0, have_pl = 0;
    memset(k, 0, sizeof *k);
    k->dir = VW2_DIR_NONE;
    while ((tok = vw2__tok(&p)) != NULL) {
        char *eq = strchr(tok, '=');
        if (!eq) return -1;
        *eq = 0;
        {
            const char *v = eq + 1;
            if (!strcmp(tok, "t")) {
                int t = vw2__t_parse(v);
                if (t == VW2_T_NONE) return 0;         /* unknown transform */
                k->t = (uint8_t)t; have_t = 1;
            } else if (!strcmp(tok, "n")) {
                if (!vw2__n_parse(v, k)) return -1;
                have_n = 1;
            } else if (!strcmp(tok, "q")) {
                if (!strcmp(v, "*")) k->q = -1;
                else { k->q = strtoll(v, NULL, 10); if (k->q <= 0) return -1; }
                have_q = 1;
            } else if (!strcmp(tok, "ord")) {
                if (!strcmp(v, "*")) k->ord = VW2_ORD_ANY;
                else if (!strcmp(v, "nat")) k->ord = VW2_ORD_NAT;
                else if (!strcmp(v, "scr")) k->ord = VW2_ORD_SCR;
                else return -1;
                have_ord = 1;
            } else if (!strcmp(tok, "place")) {
                if (!strcmp(v, "*")) k->pl = VW2_PL_ANY;
                else if (!strcmp(v, "ip"))  k->pl = VW2_PL_IP;
                else if (!strcmp(v, "oop")) k->pl = VW2_PL_OOP;
                else return -1;
                have_pl = 1;
            } else if (!strcmp(tok, "dir")) {
                if (!strcmp(v, "fwd")) k->dir = VW2_DIR_FWD;
                else if (!strcmp(v, "bwd")) k->dir = VW2_DIR_BWD;
                else return -1;
            } else {
                return 0;   /* unknown KEY token => invisible to lookup */
            }
        }
    }
    return (have_t && have_n && have_q && have_ord && have_pl) ? 1 : -1;
}

static inline void vw2__key_format(const vw2_key_t *k, char *out, size_t cap)
{
    char nb[64];
    size_t off;
    vw2__n_format(k, nb, sizeof nb);
    off = (size_t)snprintf(out, cap, "t=%s n=%s ", vw2_t_name[k->t], nb);
    if (k->q == -1) off += (size_t)snprintf(out + off, cap - off, "q=* ");
    else            off += (size_t)snprintf(out + off, cap - off, "q=%lld ", (long long)k->q);
    off += (size_t)snprintf(out + off, cap - off, "ord=%s ",
                            k->ord == VW2_ORD_ANY ? "*" : (k->ord == VW2_ORD_NAT ? "nat" : "scr"));
    off += (size_t)snprintf(out + off, cap - off, "place=%s",
                            k->pl == VW2_PL_ANY ? "*" : (k->pl == VW2_PL_IP ? "ip" : "oop"));
    if (k->dir != VW2_DIR_NONE)
        snprintf(out + off, cap - off, " dir=%s", k->dir == VW2_DIR_FWD ? "fwd" : "bwd");
}

/* ref= helper (README §3.3): "cell(t=c2c,n=4096,q=1,ord=scr,place=oop)".
 * Always the complete key. Returns 1 on parse success. */
static inline int vw2_ref_parse(const char *val, vw2_key_t *k)
{
    char buf[192];
    size_t n = strlen(val);
    if (n < 7 || n >= sizeof buf) return 0;
    if (strncmp(val, "cell(", 5) || val[n - 1] != ')') return 0;
    memcpy(buf, val + 5, n - 6);
    buf[n - 6] = 0;
    { char *c; for (c = buf; *c; c++) if (*c == ',') *c = ' '; }
    return vw2__key_parse(buf, k) == 1;
}

/* --------------------------------------------------------- shard routing */

/* key -> file. Sharding is a WRITE-side choice only (lookup reads the union
 * of all shards); it may peek at the eng= payload for the prime shard —
 * sharding is never semantics (README §2.1). */
static inline int vw2_shard_route(const vw2_key_t *k, const char *eng)
{
    if (k->rank == 2) return VW2_SHARD_2D;
    if (k->rank >= 3) return VW2_SHARD_3D;
    if (eng && (!strcmp(eng, "bluestein") || !strcmp(eng, "rader"))) return VW2_SHARD_PRIME;
    if (k->t == VW2_T_R2C || k->t == VW2_T_C2R) return VW2_SHARD_REAL;
    if (k->t == VW2_T_C2C)
        return (k->pl == VW2_PL_IP) ? VW2_SHARD_STRIDE : VW2_SHARD_OOP;
    return VW2_SHARD_STRIDE;   /* trig */
}

/* ------------------------------------------------------------- opaque */

static inline int vw2__opaque_push(vw2_store_t *s, int shard, const char *line)
{
    if (s->nopq[shard] == s->capopq[shard]) {
        int nc = s->capopq[shard] ? s->capopq[shard] * 2 : 8;
        char **np = (char **)realloc(s->opaque[shard], (size_t)nc * sizeof *np);
        if (!np) { vw2__oom(); return VW2_ENOMEM; }
        s->opaque[shard] = np; s->capopq[shard] = nc;
    }
    s->opaque[shard][s->nopq[shard]] = vw2__strdup(line);
    if (!s->opaque[shard][s->nopq[shard]]) { vw2__oom(); return VW2_ENOMEM; }
    s->nopq[shard]++;
    return VW2_OK;
}

/* ------------------------------------------------------------ line parse */

/* Parses one @cell line into rec (which the caller zeroed). Sections are
 * separated by " | " exactly; the KEY may refuse ownership (opaque carry). */
static inline int vw2__cell_parse(const char *line, vw2_rec_t *rec)
{
    char *body = vw2__strdup(line + 6);            /* skip "@cell "          */
    char *p1, *p2, *sect, *tok;
    int owned, sectid;
    if (!body) { vw2__oom(); return VW2_ENOMEM; }
    p1 = strstr(body, " | ");
    if (!p1) { free(body); return 0; }
    *p1 = 0;
    p2 = strstr(p1 + 3, " | ");
    if (p2) *p2 = 0;

    owned = vw2__key_parse(body, &rec->key);
    if (owned != 1) { free(body); return 0; }

    for (sectid = 1; sectid <= 2; sectid++) {
        sect = (sectid == 1) ? p1 + 3 : (p2 ? p2 + 3 : NULL);
        if (!sect) break;
        while ((tok = vw2__tok(&sect)) != NULL) {
            char *eq = strchr(tok, '=');
            if (!eq) continue;                     /* tolerate, keep going    */
            *eq = 0;
            if (vw2_rec_set(rec, sectid, tok, eq + 1) != VW2_OK) { free(body); return VW2_ENOMEM; }
        }
    }
    free(body);
    return 1;
}

static inline void vw2__cell_emit(FILE *f, const vw2_rec_t *r)
{
    char kb[192];
    int i, first;
    vw2__key_format(&r->key, kb, sizeof kb);
    fprintf(f, "@cell %s", kb);
    for (i = 0, first = 1; i < r->ntok; i++)
        if (r->tok[i].sect == 1) { fprintf(f, first ? " | %s=%s" : " %s=%s", r->tok[i].name, r->tok[i].val); first = 0; }
    if (first) fprintf(f, " |");
    for (i = 0, first = 1; i < r->ntok; i++)
        if (r->tok[i].sect == 2) { fprintf(f, first ? " | %s=%s" : " %s=%s", r->tok[i].name, r->tok[i].val); first = 0; }
    if (first) fprintf(f, " |");
    fputc('\n', f);
}

/* ------------------------------------------------------------------ load */

static inline void vw2__path(const vw2_store_t *s, int shard, char *out, size_t cap)
{
    snprintf(out, cap, "%s/%s", s->dir, vw2_shard_name[shard]);
}

/* Loads one shard file into the store (records + opaque). Returns VW2_OK,
 * VW2_EVERSION (poisoned), or VW2_OK with present=0 when missing. */
static inline int vw2__load_shard(vw2_store_t *s, int shard)
{
    char path[640], tmp[700];
    char line[4096];
    FILE *f;
    int first = 1;

    vw2__path(s, shard, path, sizeof path);
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
    remove(tmp);                                    /* sweep stale .tmp       */

    f = fopen(path, "rb");
    if (!f) return VW2_OK;                          /* missing = empty, fine  */
    s->present[shard] = 1;

    while (fgets(line, sizeof line, f)) {
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r')) line[--n] = 0;
        if (first) {
            int maj = -1, min = -1;
            first = 0;
            if (sscanf(line, VW2_MAGIC " %d.%d", &maj, &min) != 2 || maj != VW2_MAJOR) {
                fprintf(stderr, "[wisdom2] REFUSED %s: bad or unsupported header '%s' "
                                "(want " VW2_MAGIC " %d.x) — file poisoned, no banking/saving to it\n",
                        path, line, VW2_MAJOR);
                s->poisoned[shard] = 1;
                fclose(f);
                return VW2_EVERSION;
            }
            continue;
        }
        if (!line[0]) continue;
        if (line[0] == '#') continue;               /* comments not preserved: the
                                                       writer re-emits the header */
        if (!strncmp(line, "@legend", 7) || !strncmp(line, "@meta", 5)) continue;
        if (!strncmp(line, "@cell ", 6)) {
            vw2_rec_t rec;
            int r;
            memset(&rec, 0, sizeof rec);
            r = vw2__cell_parse(line, &rec);
            if (r == 1) {
                if (s->nrec == s->caprec) {
                    int nc = s->caprec ? s->caprec * 2 : 64;
                    vw2_rec_t *nr = (vw2_rec_t *)realloc(s->rec, (size_t)nc * sizeof *nr);
                    if (!nr) { vw2__oom(); vw2_rec_free(&rec); fclose(f); return VW2_ENOMEM; }
                    s->rec = nr; s->caprec = nc;
                }
                rec.shard = shard;
                s->rec[s->nrec++] = rec;
            } else if (r == 0) {
                vw2_rec_free(&rec);
                if (vw2__opaque_push(s, shard, line) != VW2_OK) { fclose(f); return VW2_ENOMEM; }
            } else { vw2_rec_free(&rec); fclose(f); return VW2_ENOMEM; }
        } else {
            /* unknown @-directive / future record kind: opaque carry */
            if (vw2__opaque_push(s, shard, line) != VW2_OK) { fclose(f); return VW2_ENOMEM; }
        }
    }
    fclose(f);
    return VW2_OK;
}

/* Open the store. dir==NULL resolves $VFFT_WISDOM_DIR else "." — and an
 * unset env FORCES read-only (the wrong-cwd colony killer, README §2.2).
 * `writable` is the measurement-mode guard: tools pass 1 deliberately.
 * (The config-field/env shape of the guard is an OPEN owner decision; until
 * pinned, this explicit flag is the only switch.) */
static inline int vw2_open(vw2_store_t *s, const char *dir, int writable)
{
    int i, worst = VW2_OK;
    memset(s, 0, sizeof *s);
    if (!dir || !dir[0]) {
        const char *e = getenv("VFFT_WISDOM_DIR");
        if (e && e[0]) dir = e;
        else {
            dir = ".";
            if (writable) {
                fprintf(stderr, "[wisdom2] VFFT_WISDOM_DIR unset — store at '.' opened READ-ONLY "
                                "(explicit dir required to bank)\n");
                writable = 0;
            }
        }
    }
    snprintf(s->dir, sizeof s->dir, "%s", dir);
    s->writable = (uint8_t)(writable ? 1 : 0);
    for (i = 0; i < VW2_NSHARDS; i++) {
        int r = vw2__load_shard(s, i);
        if (r != VW2_OK && worst == VW2_OK) worst = r;
    }
    fprintf(stderr, "[wisdom2] %s: %d record(s) loaded%s%s\n", s->dir, s->nrec,
            s->writable ? ", writable" : ", read-only",
            worst == VW2_EVERSION ? ", SOME FILES POISONED" : "");
    return worst;
}

static inline void vw2_close(vw2_store_t *s)
{
    int i, j;
    for (i = 0; i < s->nrec; i++) vw2_rec_free(&s->rec[i]);
    free(s->rec);
    for (j = 0; j < VW2_NSHARDS; j++) {
        for (i = 0; i < s->nopq[j]; i++) free(s->opaque[j][i]);
        free(s->opaque[j]);
    }
    memset(s, 0, sizeof *s);
}

/* ---------------------------------------------------------------- lookup */

/* Resolution (README §4.1 steps 1-2). Force-shaped env preemption is the
 * CALLER's step 0 — this module stores, it does not decide routes. */
static inline const vw2_rec_t *vw2_lookup(const vw2_store_t *s, const vw2_key_t *req)
{
    int i;
    for (i = 0; i < s->nrec; i++)                          /* 1: exact       */
        if (!vw2_key_has_wildcard(&s->rec[i].key) && vw2_key_eq(&s->rec[i].key, req))
            return &s->rec[i];
    for (i = 0; i < s->nrec; i++)                          /* 2: wildcard    */
        if (vw2_key_has_wildcard(&s->rec[i].key) && vw2_key_serves(&s->rec[i].key, req)) {
            const char *src = vw2_rec_get(&s->rec[i], "src");
            if (src && !strcmp(src, "seed")) continue;     /* seeds never serve */
            return &s->rec[i];
        }
    return NULL;                                           /* 3: MISS        */
}

/* Seed iterator: neighbor/seed records as race PROPOSALS only (never
 * verdicts). Caller filters with its own predicate. */
static inline const vw2_rec_t *vw2_scan(const vw2_store_t *s, int *cursor)
{
    if (*cursor < 0 || *cursor >= s->nrec) return NULL;
    return &s->rec[(*cursor)++];
}

/* ------------------------------------------------------------------ bank */

/* Upsert by full key tuple. Enforces: write guard, poison, the wildcard law,
 * merge rank, date tie-break, cross-metric refusal. On success the record's
 * tokens are MOVED into the store (caller must not free rec's tokens). */
static inline int vw2_bank(vw2_store_t *s, vw2_rec_t *rec)
{
    int i, shard;
    if (!s->writable) { fprintf(stderr, "[wisdom2] bank refused: store is read-only\n"); return VW2_EREADONLY; }
    if (vw2_key_has_wildcard(&rec->key) && !vw2_rec_get(rec, "from")) {
        fprintf(stderr, "[wisdom2] bank refused: wildcard key without from= "
                        "(wildcards are migration-vintage only)\n");
        return VW2_EWILDCARD;
    }
    shard = vw2_shard_route(&rec->key, vw2_rec_get(rec, "eng"));
    if (s->poisoned[shard]) {
        fprintf(stderr, "[wisdom2] bank refused: %s is poisoned (unreadable header)\n",
                vw2_shard_name[shard]);
        return VW2_EPOISON;
    }
    for (i = 0; i < s->nrec; i++) {
        if (!vw2_key_eq(&s->rec[i].key, &rec->key)) continue;
        {
            const vw2_rec_t *inc = &s->rec[i];
            const char *im = vw2_rec_get(inc, "metric"), *iu = vw2_rec_get(inc, "units");
            const char *nm = vw2_rec_get(rec, "metric"), *nu = vw2_rec_get(rec, "units");
            if (im && nm && (strcmp(im, nm) || (iu && nu && strcmp(iu, nu)))) {
                fprintf(stderr, "[wisdom2] bank refused: metric/units mismatch vs incumbent "
                                "(%s/%s vs %s/%s) — never compare across metrics\n",
                        im, iu ? iu : "?", nm, nu ? nu : "?");
                return VW2_EMETRIC;
            }
            if (vw2__merge_rank(rec) < vw2__merge_rank(inc)) {
                fprintf(stderr, "[wisdom2] bank refused: lower-rank source would replace "
                                "a raced verdict\n");
                return VW2_ERANK;
            }
            if (vw2__merge_rank(rec) == vw2__merge_rank(inc)) {
                const char *id = vw2_rec_get(inc, "date"), *nd = vw2_rec_get(rec, "date");
                if (id && nd && strcmp(nd, id) < 0) {
                    fprintf(stderr, "[wisdom2] bank refused: older record (date %s < %s)\n", nd, id);
                    return VW2_ERANK;
                }
            }
        }
        vw2_rec_free(&s->rec[i]);
        s->rec[i] = *rec;
        s->rec[i].shard = shard;
        s->dirty[shard] = 1;
        memset(rec, 0, sizeof *rec);
        return VW2_OK;
    }
    if (s->nrec == s->caprec) {
        int nc = s->caprec ? s->caprec * 2 : 64;
        vw2_rec_t *nr = (vw2_rec_t *)realloc(s->rec, (size_t)nc * sizeof *nr);
        if (!nr) { vw2__oom(); return VW2_ENOMEM; }
        s->rec = nr; s->caprec = nc;
    }
    rec->shard = shard;
    s->rec[s->nrec++] = *rec;
    s->dirty[shard] = 1;
    memset(rec, 0, sizeof *rec);
    return VW2_OK;
}

/* Field-scoped promotion (README §4.2): set one payload field on the record
 * at `key`. Replaces wisdom-file line surgery. */
static inline int vw2_update_field(vw2_store_t *s, const vw2_key_t *key,
                                   const char *name, const char *val)
{
    int i;
    if (!s->writable) { fprintf(stderr, "[wisdom2] update refused: read-only\n"); return VW2_EREADONLY; }
    for (i = 0; i < s->nrec; i++)
        if (vw2_key_eq(&s->rec[i].key, key)) {
            int r = vw2_rec_set(&s->rec[i], 1, name, val);
            if (r == VW2_OK) s->dirty[s->rec[i].shard] = 1;
            return r;
        }
    return VW2_EKEY;
}

/* ------------------------------------------------------------------ save */

static inline void vw2__emit_header(FILE *f, const vw2_store_t *s)
{
    int i;
    fprintf(f, VW2_MAGIC " %d.%d\n", VW2_MAJOR, VW2_MINOR);
    for (i = 0; i < VW2_NLEGEND; i++) fprintf(f, "@legend %s\n", vw2_legend[i]);
    if (s->meta[0]) fprintf(f, "@meta %s\n", s->meta);
}

static inline int vw2__replace_file(const char *tmp, const char *path)
{
#if defined(_WIN32)
    if (!MoveFileExA(tmp, path, MOVEFILE_REPLACE_EXISTING)) return VW2_EIO;
#else
    if (rename(tmp, path) != 0) return VW2_EIO;
#endif
    return VW2_OK;
}

/* Dirty-only, merge-on-save, atomic (README §4.2). Merge-on-save: the disk
 * file is re-read into a fresh store and THIS store's records for the shard
 * are upserted over it under the same merge law — two processes banking
 * different cells both survive; same-cell collisions resolve by rank/date. */
static inline int vw2_save(vw2_store_t *s)
{
    int shard, i, rc = VW2_OK;
    if (!s->writable) { fprintf(stderr, "[wisdom2] save refused: read-only\n"); return VW2_EREADONLY; }
    for (shard = 0; shard < VW2_NSHARDS; shard++) {
        char path[640], tmp[700];
        FILE *f;
        vw2_store_t disk;
        if (!s->dirty[shard]) continue;
        if (s->poisoned[shard]) { rc = VW2_EPOISON; continue; }

        /* merge base = current on-disk content (may have moved under us) */
        memset(&disk, 0, sizeof disk);
        snprintf(disk.dir, sizeof disk.dir, "%s", s->dir);
        disk.writable = 1;
        if (vw2__load_shard(&disk, shard) == VW2_EVERSION) { vw2_close(&disk); rc = VW2_EPOISON; continue; }
        for (i = 0; i < s->nrec; i++) {
            if (s->rec[i].shard != shard) continue;
            {
                /* deep-copy my record into the merge base via bank() */
                vw2_rec_t cp; int j;
                memset(&cp, 0, sizeof cp);
                cp.key = s->rec[i].key;
                for (j = 0; j < s->rec[i].ntok; j++)
                    if (vw2_rec_set(&cp, s->rec[i].tok[j].sect, s->rec[i].tok[j].name,
                                    s->rec[i].tok[j].val) != VW2_OK) { vw2_rec_free(&cp); break; }
                if (vw2_bank(&disk, &cp) != VW2_OK) vw2_rec_free(&cp);
            }
        }
        /* my opaque lines for this shard ride along (disk's own were loaded) */
        for (i = 0; i < s->nopq[shard]; i++) {
            int dup = 0, j;
            for (j = 0; j < disk.nopq[shard]; j++)
                if (!strcmp(disk.opaque[shard][j], s->opaque[shard][i])) { dup = 1; break; }
            if (!dup) vw2__opaque_push(&disk, shard, s->opaque[shard][i]);
        }

        vw2__path(s, shard, path, sizeof path);
        snprintf(tmp, sizeof tmp, "%s.tmp", path);
        f = fopen(tmp, "wb");
        if (!f) { vw2_close(&disk); rc = VW2_EIO; continue; }
        vw2__emit_header(f, s);
        for (i = 0; i < disk.nrec; i++)
            if (disk.rec[i].shard == shard) vw2__cell_emit(f, &disk.rec[i]);
        for (i = 0; i < disk.nopq[shard]; i++) fprintf(f, "%s\n", disk.opaque[shard][i]);
        fflush(f);
#if defined(_WIN32)
        _commit(_fileno(f));
#else
        fsync(fileno(f));
#endif
        fclose(f);
        if (vw2__replace_file(tmp, path) != VW2_OK) {
            fprintf(stderr, "[wisdom2] save FAILED for %s (old file left intact)\n", path);
            remove(tmp);
            rc = VW2_EIO;
        } else {
            s->dirty[shard] = 0;
        }
        vw2_close(&disk);
    }
    return rc;
}

/* ------------------------------------------------------------ quarantine */

/* Append-only by design: quarantined rows are kept forever with reasons
 * (README §2.1); raw= is the LAST token and runs to end-of-line. */
static inline int vw2_quarantine_append(const char *dir, const char *reason,
                                        const char *from, const char *raw)
{
    char path[640];
    FILE *f;
    long sz = 0;
    snprintf(path, sizeof path, "%s/wisdom2_quarantine.txt", dir);
    f = fopen(path, "ab");
    if (!f) return VW2_EIO;
    fseek(f, 0, SEEK_END);
    sz = ftell(f);
    if (sz == 0) fprintf(f, VW2_MAGIC " %d.%d\n", VW2_MAJOR, VW2_MINOR);
    fprintf(f, "@quarantined reason=%s from=%s raw=%s\n", reason, from, raw);
    fclose(f);
    return VW2_OK;
}

#endif /* VFFT_WISDOM2_H */
