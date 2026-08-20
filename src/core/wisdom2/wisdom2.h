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
 *   - version header checked: bad magic / major > supported / MISSING
 *     header (zero-byte or truncated file) => file refused, one loud stderr
 *     line, per-file POISON (banking/saving disabled so a save can never
 *     clobber a file we could not read). Never silently empty.
 *   - unknown tokens, unknown records, and unknown directives are carried
 *     VERBATIM through load -> bank -> save. A line this parser cannot
 *     fully OWN (unknown transform, unknown key token, bare or duplicate
 *     payload token) is carried opaque as a whole — no code path
 *     parses-and-truncates. Lines have no length limit (growing reader).
 *   - wildcards (q=* / ord=* / place=*) are legal only on migrated records
 *     (from= required); the writer refuses a fresh wildcard bank. Requests
 *     never carry wildcards (lookup refuses them loudly).
 *   - one dedup (full key tuple), merge law rank-first:
 *     race/migrated > env > seed > unknown-src; higher rank replaces
 *     unconditionally; at equal rank newer date wins, dated beats dateless,
 *     and cross-metric (or asymmetric metric/units presence when the
 *     incumbent is measured) replacement is refused.
 *   - records are EMITTED BY RESIDENCY: save writes a record into the shard
 *     file it lives in. Only vw2_bank re-routes (and then marks both the
 *     old and new shard dirty and scrubs the stale disk copy on save).
 *   - saves are dirty-only, merge-on-save, atomic (pid-suffixed tmp +
 *     MoveFileEx / rename), with every write checked — a failed emit
 *     removes the tmp and leaves the old file intact. Stale tmps are swept
 *     at OPEN only (never during a save's merge re-read).
 */
#ifndef VFFT_WISDOM2_H
#define VFFT_WISDOM2_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <limits.h>

#if defined(_WIN32)
#  include <windows.h>
#  include <io.h>
#  include <direct.h>
#  include <process.h>
#  define VW2__GETPID() _getpid()
#else
#  include <unistd.h>
#  include <dirent.h>
#  define VW2__GETPID() getpid()
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

/* src= merge rank (README §3.4/§4.2). Absent src = race (a fresh bank).
 * An UNKNOWN src value ranks LOWEST (refuse-don't-guess: a future source
 * kind read by this binary must not displace raced verdicts). */
typedef enum { VW2_SRC_UNKNOWN = 0, VW2_SRC_SEED = 1, VW2_SRC_ENV = 2,
               VW2_SRC_RACE = 3 } vw2_src_t;

/* Field portability classes (README §4.3). */
typedef enum { VW2_FC_STRUCTURAL = 0, VW2_FC_LOCAL = 1, VW2_FC_INFO = 2 } vw2_fclass_t;

/* Error codes. 0 = ok. */
enum {
    VW2_OK = 0,
    VW2_EOPEN      = -1,  /* cannot open for read (missing file is NOT an error) */
    VW2_EVERSION   = -2,  /* bad/missing magic or major -> file poisoned         */
    VW2_EREADONLY  = -3,  /* bank/save refused: store not writable               */
    VW2_EPOISON    = -4,  /* bank/save refused: file failed to load              */
    VW2_EWILDCARD  = -5,  /* fresh bank carried a wildcard without from=         */
    VW2_EMETRIC    = -6,  /* cross-metric/units replacement refused              */
    VW2_ERANK      = -7,  /* lower-rank/older record refused                     */
    VW2_EKEY       = -8,  /* malformed key / no such record                      */
    VW2_EIO        = -9,  /* write/replace failure (old file left intact)        */
    VW2_ENOMEM     = -10,
    VW2_EVALUE     = -11  /* token name/value violates the lexical rules         */
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
    uint8_t dir;                  /* vw2_dir_t; 0 = absent (reserved axis).
                                     dir matches by EQUALITY everywhere: an
                                     absent-dir record serves absent-dir
                                     requests only (README §3.1).            */
} vw2_key_t;

/* --------------------------------------------------------------- record */

typedef struct { char *name; char *val; uint8_t sect; /* 1=payload 2=meas */ } vw2_tok_t;

typedef struct {
    vw2_key_t  key;
    vw2_tok_t *tok;  int ntok, captok;
    int        shard;            /* RESIDENCY: which file it lives in        */
} vw2_rec_t;

/* ------------------------------------------------------------------ store */

#define VW2_SHARD_OOP    0
#define VW2_SHARD_STRIDE 1
#define VW2_SHARD_REAL   2
#define VW2_SHARD_PRIME  3
#define VW2_SHARD_2D     4
#define VW2_SHARD_3D     5
#define VW2_NSHARDS      6
/* wisdom2_quarantine.txt is append-only and never loaded into the table.
 * The reader consults exactly this shard table; the one-file collapse is a
 * one-table edit (the FORMAT needs no change) — README §2.1. */

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
    char       meta[256];              /* @meta payload; captured at load,
                                          settable via vw2_set_meta          */
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
 * registered). Retired names are never reused. */
typedef struct { const char *name; uint8_t fclass; } vw2_field_t;
static const vw2_field_t vw2_fields[] = {
    { "eng",     VW2_FC_STRUCTURAL }, { "route",   VW2_FC_STRUCTURAL },
    { "chain",   VW2_FC_STRUCTURAL }, { "vars",    VW2_FC_STRUCTURAL },
    { "rowplan", VW2_FC_STRUCTURAL }, { "colplan", VW2_FC_STRUCTURAL },
    { "mode",    VW2_FC_STRUCTURAL }, { "ref",     VW2_FC_STRUCTURAL },
    { "path",    VW2_FC_STRUCTURAL }, { "b",       VW2_FC_STRUCTURAL },
    { "k_pad",   VW2_FC_STRUCTURAL }, { "m",       VW2_FC_STRUCTURAL },
    { "pad_me",  VW2_FC_STRUCTURAL }, { "il_me",   VW2_FC_STRUCTURAL },
    { "sp_route",VW2_FC_STRUCTURAL }, { "sp_pair", VW2_FC_STRUCTURAL },
    { "il_route",VW2_FC_STRUCTURAL }, { "il_pair", VW2_FC_STRUCTURAL },
    { "t1p",     VW2_FC_STRUCTURAL },
    /* rank≥2 composite chains (wave 3): per-axis variant/orientation
     * fields ride beside rowplan/colplan; 3D adds the ax0/ax1 axes and
     * reuses row* for the innermost pass. */
    { "rowvars", VW2_FC_STRUCTURAL }, { "rowdif",  VW2_FC_STRUCTURAL },
    { "colvars", VW2_FC_STRUCTURAL }, { "coldif",  VW2_FC_STRUCTURAL },
    { "ax0plan", VW2_FC_STRUCTURAL }, { "ax0vars", VW2_FC_STRUCTURAL },
    { "ax0dif",  VW2_FC_STRUCTURAL }, { "ax1plan", VW2_FC_STRUCTURAL },
    { "ax1vars", VW2_FC_STRUCTURAL }, { "ax1dif",  VW2_FC_STRUCTURAL },
    /* the kv family is placement-luck, machine-tied: re-race on host
     * mismatch, never port (README §4.3; zr_kv included — it is a kernel
     * variant selector like its siblings). */
    { "zr_kv",   VW2_FC_LOCAL },
    { "t2q",     VW2_FC_LOCAL }, { "zs_t2q", VW2_FC_LOCAL },
    { "zt_t2q",  VW2_FC_LOCAL }, { "kv",     VW2_FC_LOCAL },
    { "il_kv",   VW2_FC_LOCAL }, { "sp_kv",  VW2_FC_LOCAL }, /* reserved (D9) */
    { "zt_tw",   VW2_FC_LOCAL }, { "zt_l1",  VW2_FC_LOCAL },
    /* 3D pass-A lane block: cache-geometry pick, absent = heuristic. */
    { "ablock",  VW2_FC_LOCAL },
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

static inline void vw2__oom(void)
{
    fprintf(stderr, "[wisdom2] OUT OF MEMORY — store operation aborted\n");
}

/* whitespace-splitting tokenizer (strtok_r is not portably declared) */
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

static inline int vw2__t_parse(const char *s)
{
    int i;
    for (i = 1; i < VW2_T_COUNT; i++)
        if (!strcmp(s, vw2_t_name[i])) return i;
    return VW2_T_NONE;
}

/* n= parse: "1024" / "64x64" / ... Rejects a dangling 'x', non-digit tails,
 * zero/negative extents, overflow past INT32_MAX, and rank > 4. */
static inline int vw2__n_parse(const char *s, vw2_key_t *k)
{
    int rank = 0;
    const char *p = s;
    for (;;) {
        char *end;
        long long v;
        if (*p < '0' || *p > '9') return 0;        /* digit required here    */
        errno = 0;
        v = strtoll(p, &end, 10);
        if (errno == ERANGE || v <= 0 || v > INT32_MAX) return 0;
        if (rank >= VW2_MAX_DIMS) return 0;
        k->n[rank++] = (int32_t)v;
        if (*end == '\0') break;
        if (*end != 'x') return 0;
        p = end + 1;                               /* loop re-requires digit */
    }
    k->rank = (uint8_t)rank;
    return 1;
}

static inline void vw2__n_format(const vw2_key_t *k, char *out, size_t cap)
{
    size_t off = 0; int i, r;
    out[0] = 0;
    for (i = 0; i < k->rank; i++) {
        r = snprintf(out + off, cap - off, "%s%d", i ? "x" : "", k->n[i]);
        if (r < 0 || (size_t)r >= cap - off) return;   /* truncation-safe    */
        off += (size_t)r;
    }
}

static inline int vw2_key_eq(const vw2_key_t *a, const vw2_key_t *b)
{
    int i;
    if (a->t != b->t || a->rank != b->rank) return 0;
    for (i = 0; i < a->rank; i++) if (a->n[i] != b->n[i]) return 0;
    return a->q == b->q && a->ord == b->ord && a->pl == b->pl && a->dir == b->dir;
}

/* Does record key R serve request key REQ, allowing R's wildcards?
 * dir matches by strict equality (README §3.1). Requests never carry
 * wildcards — vw2_lookup enforces that. */
static inline int vw2_key_serves(const vw2_key_t *r, const vw2_key_t *req)
{
    int i;
    if (r->t != req->t || r->rank != req->rank) return 0;
    for (i = 0; i < r->rank; i++) if (r->n[i] != req->n[i]) return 0;
    if (r->q   != -1 && r->q   != req->q)   return 0;
    if (r->ord != -1 && r->ord != req->ord) return 0;
    if (r->pl  != -1 && r->pl  != req->pl)  return 0;
    if (r->dir != req->dir) return 0;
    return 1;
}

static inline int vw2_key_has_wildcard(const vw2_key_t *k)
{
    return k->q == -1 || k->ord == -1 || k->pl == -1;
}

/* wildcard-tier precedence (README §4.1): q=*-only records outrank records
 * wildcarding ord/place. */
static inline int vw2__wild_q_only(const vw2_key_t *k)
{
    return k->q == -1 && k->ord != VW2_ORD_ANY && k->pl != VW2_PL_ANY;
}

/* ------------------------------------------------------ record helpers */

static inline const char *vw2_rec_get(const vw2_rec_t *r, const char *name)
{
    int i;
    for (i = 0; i < r->ntok; i++)
        if (!strcmp(r->tok[i].name, name)) return r->tok[i].val;
    return NULL;
}

/* Lexical law (README §3): names/values are bare tokens — no whitespace,
 * no pipes, no '=' in names, non-empty. */
static inline int vw2__lex_ok(const char *name, const char *val)
{
    const char *p;
    if (!name || !name[0] || !val || !val[0]) return 0;
    for (p = name; *p; p++)
        if (*p == ' ' || *p == '\t' || *p == '|' || *p == '=') return 0;
    for (p = val; *p; p++)
        if (*p == ' ' || *p == '\t' || *p == '|') return 0;
    return 1;
}

static inline int vw2_rec_set(vw2_rec_t *r, int sect, const char *name, const char *val)
{
    int i;
    char *nn, *nv;
    if (!vw2__lex_ok(name, val)) {
        fprintf(stderr, "[wisdom2] token refused: '%s=%s' violates the lexical rules "
                        "(bare tokens only)\n", name ? name : "?", val ? val : "?");
        return VW2_EVALUE;
    }
    for (i = 0; i < r->ntok; i++)
        if (!strcmp(r->tok[i].name, name)) {
            nv = vw2__strdup(val);
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
    nn = vw2__strdup(name);
    nv = vw2__strdup(val);
    if (!nn || !nv) { free(nn); free(nv); vw2__oom(); return VW2_ENOMEM; }
    r->tok[r->ntok].name = nn;
    r->tok[r->ntok].val  = nv;
    r->tok[r->ntok].sect = (uint8_t)sect;
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
    if (!strcmp(s, "race") || !strcmp(s, "migrated")) return VW2_SRC_RACE;
    if (!strcmp(s, "env") || !strncmp(s, "env:", 4))  return VW2_SRC_ENV;
    if (!strcmp(s, "seed"))                            return VW2_SRC_SEED;
    return VW2_SRC_UNKNOWN;   /* future src kinds never displace raced data */
}

static inline int vw2__is_seed(const vw2_rec_t *r)
{
    const char *s = vw2_rec_get(r, "src");
    return s && !strcmp(s, "seed");
}

/* Merge law (README §4.2), rank-first. Returns VW2_OK when `nw` may replace
 * `inc`, else the governing refusal code. */
static inline int vw2__merge_allows(const vw2_rec_t *inc, const vw2_rec_t *nw)
{
    int ri = vw2__src_rank(inc), rn = vw2__src_rank(nw);
    if (rn < ri) return VW2_ERANK;
    if (rn > ri) return VW2_OK;      /* higher rank replaces unconditionally */
    {
        /* equal rank: date rule — newer wins, dated beats dateless          */
        const char *id = vw2_rec_get(inc, "date"), *nd = vw2_rec_get(nw, "date");
        if (id && !nd) return VW2_ERANK;
        if (id && nd && strcmp(nd, id) < 0) return VW2_ERANK;
    }
    {
        /* equal rank: metric identity — mismatch or asymmetric presence
         * against a MEASURED incumbent is refused (README §3.4)             */
        const char *im = vw2_rec_get(inc, "metric"), *nm = vw2_rec_get(nw, "metric");
        const char *iu = vw2_rec_get(inc, "units"),  *nu = vw2_rec_get(nw, "units");
        if (im && (!nm || strcmp(im, nm))) return VW2_EMETRIC;
        if (iu && (!nu || strcmp(iu, nu))) return VW2_EMETRIC;
    }
    return VW2_OK;
}

/* ------------------------------------------------------- key parse/emit */

/* 1 = owned; 0 = carry the whole line opaque (unknown transform, unknown
 * key token — refuse-don't-guess, README §3.1); -1 = malformed (also
 * carried opaque by the caller). */
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
                else {
                    char *end;
                    errno = 0;
                    k->q = strtoll(v, &end, 10);
                    if (errno == ERANGE || end == v || *end != '\0' || k->q <= 0) return -1;
                }
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
                return 0;   /* unknown KEY token => invisible + opaque carry */
            }
        }
    }
    return (have_t && have_n && have_q && have_ord && have_pl) ? 1 : -1;
}

static inline void vw2__key_format(const vw2_key_t *k, char *out, size_t cap)
{
    char nb[64];
    size_t off = 0; int r;
    vw2__n_format(k, nb, sizeof nb);
#define VW2__CAT(...) do { r = snprintf(out + off, cap - off, __VA_ARGS__); \
    if (r < 0 || (size_t)r >= cap - off) { return; } off += (size_t)r; } while (0)
    VW2__CAT("t=%s n=%s ", vw2_t_name[k->t], nb);
    if (k->q == -1) VW2__CAT("q=* ");
    else            VW2__CAT("q=%lld ", (long long)k->q);
    VW2__CAT("ord=%s ", k->ord == VW2_ORD_ANY ? "*" : (k->ord == VW2_ORD_NAT ? "nat" : "scr"));
    VW2__CAT("place=%s", k->pl == VW2_PL_ANY ? "*" : (k->pl == VW2_PL_IP ? "ip" : "oop"));
    if (k->dir != VW2_DIR_NONE)
        VW2__CAT(" dir=%s", k->dir == VW2_DIR_FWD ? "fwd" : "bwd");
#undef VW2__CAT
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

/* key -> file, for NEW banks. Sharding is a WRITE-side choice only; it may
 * peek at eng= for the prime shard — sharding is never semantics
 * (README §2.1). Save emits by RESIDENCY, never by re-routing. */
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

/* 1 = owned; 0 = carry opaque; VW2_ENOMEM on allocation failure.
 * The literal token "-" is the empty-section marker (emitted by this
 * module, skipped here). Any other bare token, or a duplicated token name,
 * refuses ownership of the whole line — verbatim carry, never truncate. */
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
    if (owned != 1) { free(body); vw2_rec_free(rec); return 0; }

    for (sectid = 1; sectid <= 2; sectid++) {
        sect = (sectid == 1) ? p1 + 3 : (p2 ? p2 + 3 : NULL);
        if (!sect) break;
        while ((tok = vw2__tok(&sect)) != NULL) {
            char *eq = strchr(tok, '=');
            int rc;
            if (!eq) {
                if (!strcmp(tok, "-")) continue;   /* empty-section marker   */
                free(body); vw2_rec_free(rec); return 0;   /* bare token     */
            }
            *eq = 0;
            if (vw2_rec_get(rec, tok)) {           /* duplicate name         */
                free(body); vw2_rec_free(rec); return 0;
            }
            rc = vw2_rec_set(rec, sectid, tok, eq + 1);
            if (rc == VW2_EVALUE) { free(body); vw2_rec_free(rec); return 0; }
            if (rc != VW2_OK)     { free(body); vw2_rec_free(rec); return VW2_ENOMEM; }
        }
    }
    free(body);
    return 1;
}

/* Emit one record. Empty sections carry the "-" marker so the section
 * structure round-trips byte-stably (README §3 lexical rules). Returns 0
 * on success, VW2_EIO on any write error. */
static inline int vw2__cell_emit(FILE *f, const vw2_rec_t *r)
{
    char kb[192];
    int i, first;
    vw2__key_format(&r->key, kb, sizeof kb);
    if (fprintf(f, "@cell %s", kb) < 0) return VW2_EIO;
    for (i = 0, first = 1; i < r->ntok; i++)
        if (r->tok[i].sect == 1) {
            if (fprintf(f, first ? " | %s=%s" : " %s=%s", r->tok[i].name, r->tok[i].val) < 0)
                return VW2_EIO;
            first = 0;
        }
    if (first && fprintf(f, " | -") < 0) return VW2_EIO;
    for (i = 0, first = 1; i < r->ntok; i++)
        if (r->tok[i].sect == 2) {
            if (fprintf(f, first ? " | %s=%s" : " %s=%s", r->tok[i].name, r->tok[i].val) < 0)
                return VW2_EIO;
            first = 0;
        }
    if (first && fprintf(f, " | -") < 0) return VW2_EIO;
    if (fputc('\n', f) == EOF) return VW2_EIO;
    return 0;
}

/* ---------------------------------------------------------------- load */

static inline void vw2__path(const vw2_store_t *s, int shard, char *out, size_t cap)
{
    snprintf(out, cap, "%s/%s", s->dir, vw2_shard_name[shard]);
}

/* growing line reader: no length limit (README: no parse-and-truncate).
 * Returns the line (caller frees) without the trailing newline, or NULL at
 * EOF / OOM (*oom set). */
static inline char *vw2__readline(FILE *f, int *oom)
{
    size_t cap = 4096, len = 0;
    char *buf = (char *)malloc(cap);
    if (!buf) { *oom = 1; return NULL; }
    for (;;) {
        if (!fgets(buf + len, (int)(cap - len), f)) {
            if (len == 0) { free(buf); return NULL; }
            break;
        }
        len += strlen(buf + len);
        if (len && buf[len - 1] == '\n') break;
        if (cap - len < 2) {
            char *nb = (char *)realloc(buf, cap *= 2);
            if (!nb) { free(buf); *oom = 1; return NULL; }
            buf = nb;
        }
    }
    while (len && (buf[len - 1] == '\n' || buf[len - 1] == '\r')) buf[--len] = 0;
    return buf;
}

/* Loads one shard file into the store. VW2_OK, VW2_EVERSION (poisoned),
 * VW2_ENOMEM; a missing file is VW2_OK with present=0. Never touches tmps
 * (sweeping happens at open only). */
static inline int vw2__load_shard(vw2_store_t *s, int shard)
{
    char path[640];
    char *line;
    FILE *f;
    int first = 1, oom = 0, rc = VW2_OK;

    vw2__path(s, shard, path, sizeof path);
    f = fopen(path, "rb");
    if (!f) return VW2_OK;                          /* missing = empty, fine  */
    s->present[shard] = 1;

    while ((line = vw2__readline(f, &oom)) != NULL) {
        if (first) {
            int maj = -1, min = -1;
            first = 0;
            if (strncmp(line, VW2_MAGIC " ", sizeof VW2_MAGIC) != 0 ||
                sscanf(line + sizeof VW2_MAGIC, "%d.%d", &maj, &min) != 2 ||
                maj != VW2_MAJOR) {
                fprintf(stderr, "[wisdom2] REFUSED %s: bad or unsupported header '%s' "
                                "(want " VW2_MAGIC " %d.x) — file poisoned, no banking/saving to it\n",
                        path, line, VW2_MAJOR);
                s->poisoned[shard] = 1;
                free(line);
                fclose(f);
                return VW2_EVERSION;
            }
            free(line);
            continue;
        }
        if (!line[0] || line[0] == '#') { free(line); continue; }
        if (!strncmp(line, "@legend", 7) && (line[7] == ' ' || line[7] == '\0')) { free(line); continue; }
        if (!strncmp(line, "@meta", 5) && (line[5] == ' ' || line[5] == '\0')) {
            if (!s->meta[0] && line[5] == ' ')
                snprintf(s->meta, sizeof s->meta, "%s", line + 6);
            free(line);
            continue;
        }
        if (!strncmp(line, "@cell ", 6)) {
            vw2_rec_t rec;
            int r;
            memset(&rec, 0, sizeof rec);
            r = vw2__cell_parse(line, &rec);
            if (r == 1) {
                if (s->nrec == s->caprec) {
                    int nc = s->caprec ? s->caprec * 2 : 64;
                    vw2_rec_t *nr = (vw2_rec_t *)realloc(s->rec, (size_t)nc * sizeof *nr);
                    if (!nr) { vw2__oom(); vw2_rec_free(&rec); free(line); fclose(f); return VW2_ENOMEM; }
                    s->rec = nr; s->caprec = nc;
                }
                rec.shard = shard;
                s->rec[s->nrec++] = rec;
            } else if (r == 0) {
                if (vw2__opaque_push(s, shard, line) != VW2_OK) { free(line); fclose(f); return VW2_ENOMEM; }
            } else { free(line); fclose(f); return VW2_ENOMEM; }
        } else {
            /* unknown @-directive / future record kind: opaque carry */
            if (vw2__opaque_push(s, shard, line) != VW2_OK) { free(line); fclose(f); return VW2_ENOMEM; }
        }
        free(line);
    }
    fclose(f);
    if (oom) return VW2_ENOMEM;
    if (first) {
        /* file existed but had NO header line (zero-byte / crash-truncated):
         * never silently empty (README §3) */
        fprintf(stderr, "[wisdom2] REFUSED %s: empty or headerless file — poisoned, "
                        "no banking/saving to it\n", path);
        s->poisoned[shard] = 1;
        return VW2_EVERSION;
    }
    return rc;
}

/* stale-tmp sweep at OPEN only (never during a save's merge re-read):
 * removes every "<shard>.tmp*" left by crashed writers. */
static inline void vw2__sweep_tmps(const char *dir)
{
#if defined(_WIN32)
    char pat[640], full[900];
    WIN32_FIND_DATAA fd;
    HANDLE h;
    snprintf(pat, sizeof pat, "%s/wisdom2_*.txt.tmp*", dir);
    h = FindFirstFileA(pat, &fd);
    if (h == INVALID_HANDLE_VALUE) return;
    do {
        snprintf(full, sizeof full, "%s/%s", dir, fd.cFileName);
        remove(full);
    } while (FindNextFileA(h, &fd));
    FindClose(h);
#else
    DIR *d = opendir(dir);
    struct dirent *e;
    char full[900];
    if (!d) return;
    while ((e = readdir(d)) != NULL) {
        const char *n = e->d_name;
        if (!strncmp(n, "wisdom2_", 8) && strstr(n, ".txt.tmp")) {
            snprintf(full, sizeof full, "%s/%s", dir, n);
            remove(full);
        }
    }
    closedir(d);
#endif
}

/* load-time cross-shard dedup: duplicate full keys (the legacy of an old
 * re-route) resolve by the merge law; the loser is dropped from memory and
 * its shard marked dirty so the next writable save scrubs the stale line. */
static inline void vw2__dedup_loaded(vw2_store_t *s)
{
    int i, j;
    for (i = 0; i < s->nrec; i++)
        for (j = i + 1; j < s->nrec; j++) {
            int loser;
            if (!vw2_key_eq(&s->rec[i].key, &s->rec[j].key)) continue;
            loser = (vw2__merge_allows(&s->rec[i], &s->rec[j]) == VW2_OK) ? i : j;
            fprintf(stderr, "[wisdom2] duplicate key across shards (%s vs %s) — "
                            "keeping the merge-law winner, shard marked for scrub\n",
                    vw2_shard_name[s->rec[i].shard], vw2_shard_name[s->rec[j].shard]);
            s->dirty[s->rec[loser].shard] = 1;
            vw2_rec_free(&s->rec[loser]);
            s->rec[loser] = s->rec[s->nrec - 1];
            s->nrec--;
            i--;                                    /* re-scan the moved slot */
            break;
        }
}

/* Open the store. dir==NULL resolves $VFFT_WISDOM_DIR else "." — and an
 * unset env FORCES read-only (the wrong-cwd colony killer, README §2.2).
 * `writable` is the measurement-mode guard (README: exact config/env shape
 * of the guard is an OPEN owner decision; this explicit flag is the only
 * switch until then). */
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
    if (strlen(dir) >= sizeof s->dir) {
        fprintf(stderr, "[wisdom2] wisdom dir path too long — store opened READ-ONLY on '.'\n");
        dir = ".";
        writable = 0;
    }
    snprintf(s->dir, sizeof s->dir, "%s", dir);
    s->writable = (uint8_t)(writable ? 1 : 0);
    vw2__sweep_tmps(s->dir);
    for (i = 0; i < VW2_NSHARDS; i++) {
        int r = vw2__load_shard(s, i);
        if (r != VW2_OK && worst == VW2_OK) worst = r;
    }
    vw2__dedup_loaded(s);
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

/* host/isa/l1d stamp for the header; also captured from files at load. */
static inline void vw2_set_meta(vw2_store_t *s, const char *meta)
{
    snprintf(s->meta, sizeof s->meta, "%s", meta ? meta : "");
}

/* The measurement-mode guard, togglable by the owner of the store (the
 * create path applies the config guard here; tools pass writable at open).
 * Never overrides the unset-env forcing at open — callers that were forced
 * read-only for the wrong-cwd reason stay read-only. */
static inline void vw2_set_writable(vw2_store_t *s, int on)
{
    s->writable = (uint8_t)(on ? 1 : 0);
}

/* Repoint the store at a different directory for an explicit save-to-dir
 * (the vfft_wisdom_save(w, dir) API): every shard is marked dirty so the
 * whole in-memory table lands at the new location. Read state (records)
 * is kept; the old directory is not touched again. */
static inline void vw2_repoint(vw2_store_t *s, const char *dir)
{
    int i;
    if (!dir || !dir[0] || strlen(dir) >= sizeof s->dir) return;
    snprintf(s->dir, sizeof s->dir, "%s", dir);
    s->writable = 1;
    for (i = 0; i < VW2_NSHARDS; i++) {
        s->poisoned[i] = 0;      /* new dir: poison state belongs to old files */
        s->dirty[i] = 1;
    }
}

/* ---------------------------------------------------------------- lookup */

/* dangling-ref rule (README §3.3): a hit whose ref= target is absent from
 * the union (or unparseable) is a MISS — loud, then the normal miss path. */
static inline int vw2__ref_ok(const vw2_store_t *s, const vw2_rec_t *r)
{
    const char *v = vw2_rec_get(r, "ref");
    vw2_key_t tk;
    int i;
    if (!v) return 1;
    if (!vw2_ref_parse(v, &tk)) {
        fprintf(stderr, "[wisdom2] dangling ref (unparseable '%s') — verdict treated as MISS\n", v);
        return 0;
    }
    for (i = 0; i < s->nrec; i++)
        if (vw2_key_eq(&s->rec[i].key, &tk)) return 1;
    fprintf(stderr, "[wisdom2] dangling ref (target absent) — verdict treated as MISS\n");
    return 0;
}

/* Resolution (README §4.1 steps 1-2). Force-shaped env preemption is the
 * CALLER's step 0 — this module stores, it does not decide routes.
 * Seeds are never served from either tier; requests never carry wildcards;
 * wildcard tier prefers q=*-only records over ord/place wildcards. */
static inline const vw2_rec_t *vw2_lookup(const vw2_store_t *s, const vw2_key_t *req)
{
    int i, pass;
    if (vw2_key_has_wildcard(req)) {
        fprintf(stderr, "[wisdom2] lookup refused: request key carries a wildcard\n");
        return NULL;
    }
    for (i = 0; i < s->nrec; i++)                          /* 1: exact       */
        if (!vw2_key_has_wildcard(&s->rec[i].key) && vw2_key_eq(&s->rec[i].key, req)) {
            if (vw2__is_seed(&s->rec[i])) continue;
            if (!vw2__ref_ok(s, &s->rec[i])) continue;
            return &s->rec[i];
        }
    for (pass = 0; pass < 2; pass++)                       /* 2: wildcards   */
        for (i = 0; i < s->nrec; i++) {
            const vw2_rec_t *r = &s->rec[i];
            if (!vw2_key_has_wildcard(&r->key)) continue;
            if (pass == 0 && !vw2__wild_q_only(&r->key)) continue;
            if (pass == 1 &&  vw2__wild_q_only(&r->key)) continue;
            if (!vw2_key_serves(&r->key, req)) continue;
            if (vw2__is_seed(r)) continue;
            if (!vw2__ref_ok(s, r)) continue;
            return r;
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

/* internal upsert with a PINNED shard (used by save's merge; no guard or
 * wildcard checks — those belong to the public entry). Merge law applies.
 * On success the record's tokens are MOVED into the store. */
static inline int vw2__bank_pinned(vw2_store_t *s, vw2_rec_t *rec, int shard)
{
    int i;
    for (i = 0; i < s->nrec; i++) {
        int rc;
        if (!vw2_key_eq(&s->rec[i].key, &rec->key)) continue;
        rc = vw2__merge_allows(&s->rec[i], rec);
        if (rc != VW2_OK) return rc;
        {
            /* loud when a same-rank replacement changes the engine — the
             * migrator's dual-fold signal (README §4.2) */
            const char *ie = vw2_rec_get(&s->rec[i], "eng"), *ne = vw2_rec_get(rec, "eng");
            if (ie && ne && strcmp(ie, ne))
                fprintf(stderr, "[wisdom2] note: replacement changes eng=%s -> eng=%s "
                                "on an existing cell\n", ie, ne);
        }
        if (s->rec[i].shard != shard) s->dirty[s->rec[i].shard] = 1;  /* scrub old */
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

/* Public bank: guard + wildcard law + routing, then the pinned upsert.
 * On success the record's tokens are MOVED (caller must not free them). */
static inline int vw2_bank(vw2_store_t *s, vw2_rec_t *rec)
{
    int shard;
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
    return vw2__bank_pinned(s, rec, shard);
}

/* Field-scoped promotion (README §4.2): set one payload field on the record
 * at `key`. Residency is sticky — promotion never re-routes a record. */
static inline int vw2_update_field(vw2_store_t *s, const vw2_key_t *key,
                                   const char *name, const char *val)
{
    int i;
    if (!s->writable) { fprintf(stderr, "[wisdom2] update refused: read-only\n"); return VW2_EREADONLY; }
    for (i = 0; i < s->nrec; i++)
        if (vw2_key_eq(&s->rec[i].key, key)) {
            int r;
            if (s->poisoned[s->rec[i].shard]) return VW2_EPOISON;
            r = vw2_rec_set(&s->rec[i], 1, name, val);
            if (r == VW2_OK) s->dirty[s->rec[i].shard] = 1;
            return r;
        }
    return VW2_EKEY;
}

/* ------------------------------------------------------------------ save */

static inline int vw2__emit_header(FILE *f, const vw2_store_t *s)
{
    int i;
    if (fprintf(f, VW2_MAGIC " %d.%d\n", VW2_MAJOR, VW2_MINOR) < 0) return VW2_EIO;
    for (i = 0; i < VW2_NLEGEND; i++)
        if (fprintf(f, "@legend %s\n", vw2_legend[i]) < 0) return VW2_EIO;
    if (s->meta[0] && fprintf(f, "@meta %s\n", s->meta) < 0) return VW2_EIO;
    return 0;
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

/* Dirty-only, merge-on-save, atomic, RESIDENCY-emitted (README §4.2).
 * Every write is checked: any failure removes the tmp and leaves the old
 * file intact. Records this store holds in a DIFFERENT shard are scrubbed
 * from this shard's disk copy (the re-route cleanup). */
static inline int vw2_save(vw2_store_t *s)
{
    int shard, i, j, rc = VW2_OK;
    if (!s->writable) { fprintf(stderr, "[wisdom2] save refused: read-only\n"); return VW2_EREADONLY; }
    for (shard = 0; shard < VW2_NSHARDS; shard++) {
        char path[640], tmp[720];
        FILE *f;
        vw2_store_t disk;
        int err = 0, lrc;

        if (!s->dirty[shard]) continue;
        if (s->poisoned[shard]) { rc = VW2_EPOISON; continue; }

        /* merge base = current on-disk content (may have moved under us) */
        memset(&disk, 0, sizeof disk);
        snprintf(disk.dir, sizeof disk.dir, "%s", s->dir);
        disk.writable = 1;
        lrc = vw2__load_shard(&disk, shard);
        if (lrc != VW2_OK) { vw2_close(&disk); rc = (lrc == VW2_EVERSION) ? VW2_EPOISON : lrc; continue; }

        /* scrub: disk records whose key THIS store holds in another shard */
        for (j = 0; j < disk.nrec; j++)
            for (i = 0; i < s->nrec; i++)
                if (s->rec[i].shard != shard &&
                    vw2_key_eq(&s->rec[i].key, &disk.rec[j].key)) {
                    vw2_rec_free(&disk.rec[j]);
                    disk.rec[j] = disk.rec[disk.nrec - 1];
                    disk.nrec--;
                    j--;
                    break;
                }

        /* upsert my residents over the merge base (pinned: no re-route) */
        for (i = 0; i < s->nrec && !err; i++) {
            vw2_rec_t cp; int t, copy_ok = 1;
            if (s->rec[i].shard != shard) continue;
            memset(&cp, 0, sizeof cp);
            cp.key = s->rec[i].key;
            for (t = 0; t < s->rec[i].ntok; t++)
                if (vw2_rec_set(&cp, s->rec[i].tok[t].sect, s->rec[i].tok[t].name,
                                s->rec[i].tok[t].val) != VW2_OK) { copy_ok = 0; break; }
            if (!copy_ok) { vw2_rec_free(&cp); err = 1; break; }
            if (vw2__bank_pinned(&disk, &cp, shard) != VW2_OK) vw2_rec_free(&cp);
        }
        /* my opaque lines ride along (disk's own were loaded) */
        for (i = 0; i < s->nopq[shard] && !err; i++) {
            int dup = 0;
            for (j = 0; j < disk.nopq[shard]; j++)
                if (!strcmp(disk.opaque[shard][j], s->opaque[shard][i])) { dup = 1; break; }
            if (!dup && vw2__opaque_push(&disk, shard, s->opaque[shard][i]) != VW2_OK) err = 1;
        }
        if (err) { vw2_close(&disk); rc = VW2_ENOMEM; continue; }

        if (!disk.meta[0] && s->meta[0]) snprintf(disk.meta, sizeof disk.meta, "%s", s->meta);

        vw2__path(s, shard, path, sizeof path);
        snprintf(tmp, sizeof tmp, "%s.tmp.%d", path, (int)VW2__GETPID());
        f = fopen(tmp, "wb");
        if (!f) { vw2_close(&disk); rc = VW2_EIO; continue; }
        if (vw2__emit_header(f, &disk) != 0) err = 1;
        for (i = 0; i < disk.nrec && !err; i++)
            if (disk.rec[i].shard == shard && vw2__cell_emit(f, &disk.rec[i]) != 0) err = 1;
        for (i = 0; i < disk.nopq[shard] && !err; i++)
            if (fprintf(f, "%s\n", disk.opaque[shard][i]) < 0) err = 1;
        if (!err && (ferror(f) || fflush(f) != 0)) err = 1;
#if defined(_WIN32)
        if (!err && _commit(_fileno(f)) != 0) err = 1;
#else
        if (!err && fsync(fileno(f)) != 0) err = 1;
#endif
        if (fclose(f) != 0) err = 1;
        if (err || vw2__replace_file(tmp, path) != VW2_OK) {
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

/* Append-only by design (quarantined rows are kept forever with reasons —
 * README §2.1). Honors the write guard; reason/from obey the lexical rules;
 * raw= is the LAST token and runs to end-of-line (CR/LF stripped). */
static inline int vw2_quarantine_append(vw2_store_t *s, const char *reason,
                                        const char *from, const char *raw)
{
    char path[640];
    char rawbuf[8192];
    FILE *f;
    long sz;
    size_t n;
    if (!s->writable) {
        fprintf(stderr, "[wisdom2] quarantine refused: store is read-only\n");
        return VW2_EREADONLY;
    }
    if (!vw2__lex_ok("reason", reason) || !vw2__lex_ok("from", from)) {
        fprintf(stderr, "[wisdom2] quarantine refused: reason/from must be bare tokens\n");
        return VW2_EVALUE;
    }
    n = strlen(raw);
    if (n >= sizeof rawbuf) n = sizeof rawbuf - 1;
    memcpy(rawbuf, raw, n);
    rawbuf[n] = 0;
    { char *c; for (c = rawbuf; *c; c++) if (*c == '\n' || *c == '\r') *c = ' '; }
    snprintf(path, sizeof path, "%s/wisdom2_quarantine.txt", s->dir);
    f = fopen(path, "ab");
    if (!f) return VW2_EIO;
    if (fseek(f, 0, SEEK_END) != 0 || (sz = ftell(f)) < 0) { fclose(f); return VW2_EIO; }
    if (sz == 0 && fprintf(f, VW2_MAGIC " %d.%d\n", VW2_MAJOR, VW2_MINOR) < 0) { fclose(f); return VW2_EIO; }
    if (fprintf(f, "@quarantined reason=%s from=%s raw=%s\n", reason, from, rawbuf) < 0) { fclose(f); return VW2_EIO; }
    if (fclose(f) != 0) return VW2_EIO;
    return VW2_OK;
}

#endif /* VFFT_WISDOM2_H */
