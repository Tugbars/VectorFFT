/* estimate_build.h -- the 1D estimate's GLUE: a recipe becomes a public plan through the library's own
 * replay door, with no measurement.
 *
 * Estimate is unreleased (this folder is git-ignored, .gitignore:87). The formulas (estimate_1d.h)
 * compute a recipe -- the route and every parameter a race would have banked -- from N, the request
 * and the machine. This header turns that recipe into a plan the way a wisdom hit does:
 *   1. it writes the recipe as the row(s) a race would bank, into a PRIVATE store held in memory
 *      (vfft_child_store_new, il/wisdom/wisdom2_child.h:40), with the library's own writers
 *      (vw2_oop_bank_k1_il, vw2_update_field, vw2_prime_method_bank, vw2_oop_rec_k1_bwd and the
 *      vw2_real_il_bank_* writers) -- never the planner's vfft_il_dp_emit_wisdom, which banks a
 *      four-step's child from a race context that is NULL here (dp_planner_il.h:2191-2197);
 *   2. it calls the library's own create, _vfft_create_cfg (vfft.c:2509), against that store: the
 *      door's "banked verdict replays" branch builds the handle, binds the executor, holds the pool
 *      and allocates the real alias buffer, exactly as for a served plan.
 * The user's on-disk store is never opened, read or written: cfg.wisdom is always set (the default
 * store opens only when it is NULL, vfft.c:2276), wisdom_write is 0 and the create depth is raised
 * so the process save switch is never read (vfft.c:2523-2524). After every create the glue checks
 * that the library's default store was not loaded (_def_loaded, vfft.c:496-505) and that no
 * directory store was opened (_vfft_api_nstores, vfft_api.h:41). That detector sees a FIRST load
 * only: once the process has loaded its default store (any vfft_create with no wisdom_dir, or
 * vfft_wisdom_folder()), a nested create that fell to _default_wisdom() (vfft.c:2276) would read it
 * unseen. None does today -- every IL nested create sets .wisdom (k1_fourstep.h:403,606;
 * k1_commit.h:296,301; zfsr.h:115; zr2c_build.h:349,463; the 2D tier's) -- and the tool never loads
 * the default store, so its runs are covered; a library caller of est_build_1d relies on that
 * code fact. A stronger detector would have to swap the library's default-store state around the
 * create, which on a hit re-opens the user's folder from disk -- a cure worse than the gap.
 *
 * The plan is refused, never raced. A row the door cannot build as written is not an error to the
 * door: it marks the row stale and races the cell (c2c_oop_create_il.h:416-441,
 * c2c_ip_create_il.h:94-119, zrp_build.h:1088-1091, odd_build.h:751-754). So before the store is
 * handed over, a PRE-FLIGHT calls the door's own constructor sequence at one thread, with no clock,
 * and destroys what it built (REPLAY_API section 2). A four-step child is a nested create (the c2c
 * four-step, zfsr, a four-step inner of the prime cell or of zrb, a zr2c four-step child): it is not
 * built here, but its two plans are checked without a create -- the 2D child's column chain with
 * the 2D tier's own resolver, its row plan at N2 by this same pre-flight (est__fs_preflight). What
 * that cannot see, the detectors below catch.
 *
 * The no-measurement proof (REPLAY_API section 0.4): four counters are read before the pre-flight
 * and after the create -- est_clock_reads (the tool's VFFT_CLOCK_PROBE, race_scope.h:636-642: every
 * library clock read), _vfft_create_race_count (vfft.c:255, race.h:149), vfft_measure_scopes()
 * (vfft_measure.h:112) -- and the private store's row count. Any change means the create measured:
 * the plan is destroyed and NULL returned (EST_MEASURED). A row the door re-banked at an existing
 * key carries the process's "own" mark (wisdom2.h:189; the test vfft_child_store_raced makes,
 * wisdom2_child.h:164). The door does that without a clock read in two places (the four-step's
 * child completing its row, k1_commit.h:1558-1564; the 2D tier banking cmt=0 when its MT arm
 * cannot engage, il2d_tier.h:4413-4424), so such a row is not a measurement -- but it may no longer
 * be the recipe. Each one is compared with the row the glue wrote: the same payload is reported
 * only; a changed payload destroys the plan (EST_REWRITTEN).
 *
 * Name things as the library does: the row tokens, the VFFT_K1_IL_* routes, the engine names.
 * Specification: docs/design/estimate_1d_sketch.md; build reference:
 * docs/research/estimate_1d/impl/REPLAY_API.md. Paths are relative to src/core/ unless they start
 * with include/ or docs/.
 *
 * THE INCLUDER (REPLAY_API section 1.2): one TU that defines
 *     static long est_clock_reads;
 *     #define VFFT_CLOCK_PROBE (est_clock_reads++ ...)
 * then includes vfft.c FIRST (nothing of the library before it), then the formulas, then this. */
#ifndef EST_BUILD_H
#define EST_BUILD_H

#ifndef VFFT_API_H
#error "estimate_build.h is the glue of a library TU: #include \"vfft.c\" before it (REPLAY_API section 1.2)"
#endif
#ifndef VFFT_CLOCK_PROBE
#error "define est_clock_reads and VFFT_CLOCK_PROBE before #include \"vfft.c\" (REPLAY_API section 1.2)"
#endif

#include <stdarg.h>
#include "estimate_recipe.h"

#ifdef _WIN32
#define EST__ENVIRON _environ
#else
extern char **environ;
#define EST__ENVIRON environ
#endif

/* ---------------------------------------------------------------- the outcome */
enum
{
    EST_BUILT = 0,    /* a plan, built from the recipe with nothing measured */
    EST_REFUSED = 1,  /* the recipe does not build as written (or the request is outside estimate) */
    EST_MEASURED = 2, /* the create measured: a counter or the private store moved (plan destroyed) */
    EST_SERIAL = 3,   /* a threaded arm the recipe asked for did not take (plan destroyed) */
    EST_NOPLAN = 4,   /* the door returned no plan and measured nothing */
    EST_REWRITTEN = 5 /* the door re-banked a recipe row with another payload, measuring nothing: the
                       * plan is not the recipe (plan destroyed) */
};

typedef struct
{
    int status;                 /* EST_* */
    long clock_reads;           /* the deltas over pre-flight + create; all 0 when EST_BUILT */
    long races;
    long scopes;
    int rows_before, rows_after;/* the private store's row count around the create */
    int rewritten;              /* rows the door re-banked without measuring ("own" rows) */
    int changed;                /* of those, rows whose payload is no longer the glue's (EST_REWRITTEN) */
    char rewritten_key[192];    /* the first such row's key (the first changed one, when any) */
    int nested;                 /* 1 = the recipe holds a nested create (no pre-flight possible) */
} est_build_report_t;

/* ---------------------------------------------------------------- small helpers */
static inline void est__say(char *why, size_t whylen, const char *fmt, ...)
{
    va_list ap;
    if (!why || !whylen)
        return;
    va_start(ap, fmt);
    vsnprintf(why, whylen, fmt, ap);
    va_end(ap);
}

static inline int est__T(const est_request_t *rq) { return rq->nthreads > 1 ? rq->nthreads : 1; }

/* a.b.c of v[0..n-1]; -1 when it does not fit */
static inline int est__ints(char *b, size_t cap, const int *v, int n)
{
    size_t off = 0;
    int i;
    if (cap)
        b[0] = 0;
    for (i = 0; i < n; i++)
    {
        const int w = snprintf(b + off, cap - off, "%s%d", i ? "." : "", v[i]);
        if (w < 0 || (size_t)w >= cap - off)
            return -1;
        off += (size_t)w;
    }
    return 0;
}
static inline long est__prod(const int *v, int n)
{
    long p = 1;
    int i;
    for (i = 0; i < n; i++)
        p *= v[i];
    return p;
}

/* the real engines by their row names (wisdom2_real_il.h:7-44) */
enum { EST_ENG_K1, EST_ENG_ZRM, EST_ENG_ZRP, EST_ENG_ZTTR, EST_ENG_ZR2C, EST_ENG_ZFSR, EST_ENG_ZRF, EST_ENG_ZRB,
       EST_ENG_BAD };
static inline int est__eng(const est_recipe_t *rc, const est_request_t *rq)
{
    static const char *const names[] = { "k1", "zrm", "zrp", "zttr", "zr2c", "zfsr", "zrf", "zrb" };
    int i;
    if (rq->transform == VFFT_C2C)   /* the c2c row's engine is k1; an empty eng reads the same */
        return (!rc->eng[0] || !strcmp(rc->eng, "k1")) ? EST_ENG_K1 : EST_ENG_BAD;
    for (i = 1; i < (int)(sizeof names / sizeof names[0]); i++)
        if (!strcmp(rc->eng, names[i]))
            return i;
    return EST_ENG_BAD;
}

/* 1 = the recipe holds a nested create (a four-step child), which no pre-flight can build
 * (REPLAY_API section 0.3): the c2c four-step, the prime cell's or zrb's four-step inner, zfsr,
 * a zr2c four-step child or a zr2c prime child with a four-step inner */
static inline int est_recipe_nested(const est_recipe_t *rc, const est_request_t *rq)
{
    const int eng = est__eng(rc, rq);
    if (eng == EST_ENG_K1 || eng == EST_ENG_ZR2C)
        return rc->k1_il_route == VFFT_K1_IL_FS ||
               (rc->k1_il_route == VFFT_K1_IL_PRIME && !strcmp(rc->prime.in_kind, "fs"));
    if (eng == EST_ENG_ZFSR)
        return 1;
    if (eng == EST_ENG_ZRB)
        return !strcmp(rc->zrb.kind, "fs");
    return 0;
}

/* ---------------------------------------------------------------- the environment guard */
/* REPLAY_API section 1.3: a VFFT_* variable steers what a create builds (an env pin beats a banked
 * row, e.g. VFFT_ZRP in zrp_build.h:971-985, VFFT_IL_KV in k1_commit.h:131-140). Every VFFT_* name in
 * the process environment refuses estimate, except the ones that only log or only name the user's
 * store -- which estimate never opens (checked after each create). The scan reads the environment
 * itself, so a variable the library learns later is refused too. On Windows the names are matched
 * without case: getenv there ignores case, so "vfft_il_kv" is the library's VFFT_IL_KV. Returns the
 * first steering name, or NULL. */
static inline int est__env_name_eq(const char *a, const char *b, size_t n)
{
#ifdef _WIN32
    size_t i;
    for (i = 0; i < n; i++)
    {
        const int x = (a[i] >= 'a' && a[i] <= 'z') ? a[i] - 'a' + 'A' : a[i];
        const int y = (b[i] >= 'a' && b[i] <= 'z') ? b[i] - 'a' + 'A' : b[i];
        if (x != y || !x)
            return x == y;
    }
    return 1;
#else
    return !strncmp(a, b, n);
#endif
}
static inline const char *est_env_steering(void)
{
    static const char *const harmless[] = {
        "VFFT_NAT_LOG", "VFFT_ILPR_LOG", "VFFT_IL_DP_VERBOSE", "VFFT_ZRACE_VERBOSE", "VFFT_IL2D_LOG",
        "VFFT_RACE_LOG", "VFFT_TCMT_LOG", "VFFT_TCMT_VERBOSE", "VFFT_VERBOSE", "VFFT_BAKEOFF_DBG",
        "VFFT_ILND_PROF", "VFFT_ILNDR_PROF",
        "VFFT_WISDOM_DIR", "VFFT_ADOPT_WISDOM_DIR", "VFFT_WISDOM_WRITE", NULL };
    static char name[96];
    char **e;
    for (e = EST__ENVIRON; e && *e; e++)
    {
        const char *s = *e;
        const size_t n = strcspn(s, "=");
        int i, ok = 0;
        if (n < 5 || !est__env_name_eq(s, "VFFT_", 5))
            continue;
        for (i = 0; harmless[i]; i++)
            if (strlen(harmless[i]) == n && est__env_name_eq(s, harmless[i], n))
                ok = 1;
        if (!ok)
        {
            snprintf(name, sizeof name, "%.*s", (int)(n < 80 ? n : 80), s);
            return name;
        }
    }
    return NULL;
}

/* ---------------------------------------------------------------- the four-step child's tokens */
/* The recipe carries a four-step's child as the 2D row's tokens under fs_ (the sketch section 4.5,
 * table 5.7), its row plan under fs_row_ and that plan's backward twin under fs_row_bwd_ -- the
 * shipped rows' spelling (src/wisdom/14900KF/wisdom2_oop.txt, n=524288). A race today banks the 2D
 * row as the 2D tier writes it: the row plan rides INSIDE it as rp_* (wisdom2_child.h:5-21,
 * _k1fs_row_put_child k1_fourstep.h:615-640), so the K=1 row carries fs_rp_k1=<key> fs_rp_1_<name>=..
 * When the 2D row has no rp_ recipe, the 2D create completes it at its end (il2d_tier.h:4225,
 * vfft_child_row_update wisdom2_child.h:176-186) and its private store gains an "own" row; zfsr
 * then refuses its own replay and the door RACES the cell (zrp_build.h:330-334). So the glue writes
 * the child as a race would: fs_row_X -> fs_rp_1_X under the row plan's key, fs_row_bwd_X ->
 * fs_rp_2_X under its backward twin's key (the four-step's child keys, k1_fourstep.h:215-230: in
 * place, natural, one thread). A recipe already in the rp_ spelling is copied as it is.
 * to_rp = 0 copies the text unchanged (the compare arm: the stored rows' spelling).
 * Returns 0, or -1 when a token is malformed or the text does not fit cap. */
static inline int est__fs_text(const char *in, int n1, int n2, int to_rp, char *out, size_t cap)
{
    char buf[1024], *sp = buf, *t, kr[192], kb[192];
    size_t off = 0, j;
    int has_row = 0, has_rp = 0, has_bwd = 0, pass, w;
    vw2_key_t k2, krow, kbwd;
    if (!in || strlen(in) >= sizeof buf)
        return -1;
    out[0] = 0;
    /* what the text holds */
    memcpy(buf, in, strlen(in) + 1);
    while ((t = vw2__tok(&sp)) != NULL)
    {
        if (!strchr(t, '=') || strncmp(t, "fs_", 3))
            return -1;   /* every child token is name=value under fs_ */
        if (!strncmp(t, "fs_rp_", 6)) has_rp = 1;
        else if (!strncmp(t, "fs_row_bwd_", 11)) has_bwd = 1;
        else if (!strncmp(t, "fs_row_", 7)) has_row = 1;
    }
    if (!to_rp || has_rp || (!has_row && !has_bwd))
    {
        if (strlen(in) >= cap)
            return -1;
        memcpy(out, in, strlen(in) + 1);
        return 0;
    }
    if (!_k1fs_child_keys(n1, n2, 0, 1, &k2, &krow, &kbwd))
        return -1;
    vw2__key_format(&krow, kr, sizeof kr);
    vw2__key_format(&kbwd, kb, sizeof kb);
    for (j = 0; kr[j]; j++) if (kr[j] == ' ') kr[j] = ',';
    for (j = 0; kb[j]; j++) if (kb[j] == ' ') kb[j] = ',';
    /* pass 0: the 2D row's own tokens; 1: the row plan (key first); 2: its backward twin */
    for (pass = 0; pass < 3; pass++)
    {
        if (pass == 1 && has_row)
        {
            w = snprintf(out + off, cap - off, "%sfs_rp_k1=%s", off ? " " : "", kr);
            if (w < 0 || (size_t)w >= cap - off) return -1;
            off += (size_t)w;
        }
        if (pass == 2 && has_bwd)
        {
            w = snprintf(out + off, cap - off, "%sfs_rp_k%d=%s", off ? " " : "", has_row ? 2 : 1, kb);
            if (w < 0 || (size_t)w >= cap - off) return -1;
            off += (size_t)w;
        }
        memcpy(buf, in, strlen(in) + 1);
        sp = buf;
        while ((t = vw2__tok(&sp)) != NULL)
        {
            const int bwd = !strncmp(t, "fs_row_bwd_", 11), row = !bwd && !strncmp(t, "fs_row_", 7);
            if (pass == 0 && !bwd && !row)
                w = snprintf(out + off, cap - off, "%s%s", off ? " " : "", t);
            else if (pass == 1 && row)
                w = snprintf(out + off, cap - off, "%sfs_rp_1_%s", off ? " " : "", t + 7);
            else if (pass == 2 && bwd)
                w = snprintf(out + off, cap - off, "%sfs_rp_%d_%s", off ? " " : "", has_row ? 2 : 1, t + 11);
            else
                continue;
            if (w < 0 || (size_t)w >= cap - off) return -1;
            off += (size_t)w;
        }
    }
    return 0;
}
/* 1 = the text carries the 2D row (an fs_ token that is not the row plan's): what
 * _k1fs_row_has_child (k1_fourstep.h:526-538) asks of a row */
static inline int est__fs_has_child(const char *fs)
{
    const char *p = fs;
    while (p && *p)
    {
        while (*p == ' ') p++;
        if (!strncmp(p, "fs_", 3) && strncmp(p, "fs_row_", 7) && strchr(p, '='))
            return 1;
        p = strchr(p, ' ');
    }
    return 0;
}
/* the fs_ tokens onto the row at key, one vw2_update_field each (wisdom2.h:1494) -- the way
 * _k1fs_row_put_child puts them on a row (k1_fourstep.h:615-640) */
static inline int est__put_fs(vw2_store_t *st, const vw2_key_t *key, const char *fs)
{
    char buf[1024], *sp = buf, *t;
    if (strlen(fs) >= sizeof buf)
        return -1;
    memcpy(buf, fs, strlen(fs) + 1);
    while ((t = vw2__tok(&sp)) != NULL)
    {
        char *eq = strchr(t, '=');
        if (!eq)
            return -1;
        *eq = 0;
        if (vw2_update_field(st, key, t, eq + 1) != VW2_OK)
            return -1;
    }
    return 0;
}
/* N1.N2 of a four-step inner's shape (the prime row's in_sh=, zrb's in_sh=) */
static inline int est__split_of(const char *shape, int *n1, int *n2)
{
    return sscanf(shape, "%d.%d", n1, n2) == 2 && *n1 >= 4 && *n2 >= 4;
}
/* The length a prime cell's inner must have, checked for a four-step inner before its nested
 * create: a wrong M fails the replay and the door then races the prime pool
 * (_ilprime_create_banked, k1_commit.h:615). Rader: M = N - 1 (_ilprime_create_rader,
 * il_prime.h:299-318; _ilprime_fs_inner_build refuses R1 R2 != M, k1_commit.h:288-289).
 * Bluestein: any even M >= 2N - 1 (_ilprime_create_bluestein_m, il_prime.h:249-250). */
static inline int est__prime_inner_len_ok(int N, int method, long M)
{
    if (method == 1)
        return M == (long)N - 1;
    return M >= 2L * N - 1 && !(M & 1);
}

/* ---------------------------------------------------------------- the c2c section */
/* The c2c recipe's own legality, before any row is written: the route is one the door replays and
 * its shape multiplies to n. The recipe's il_R1/il_R2 of a chain3 is the race's il_pair: the mid
 * product A*B and the leaf (dp_planner_il.h:136-138; src/wisdom/14900KF/wisdom2_oop.txt n=250:
 * il_pair=25.10 il_chain=10.5.5). The zr2c replay builds a chain3 child from il_pair's leaf
 * (_zr2c_cand_of_child, zr2c_build.h:156-180), so it is filled here when the recipe leaves it 0. */
static inline int est__c2c_shape(const est_recipe_t *rc, int n, int *R1, int *R2, char *why, size_t whylen)
{
    *R1 = rc->il_R1;
    *R2 = rc->il_R2;
    switch (rc->k1_il_route)
    {
    case VFFT_K1_IL_MONO:
        return 0;
    case VFFT_K1_IL_2P_PURE:
        if (*R1 < 3 || *R2 < 3 || (long)*R1 * *R2 != n)
        { est__say(why, whylen, "pair %d.%d is not a split of %d with both radices >= 3", *R1, *R2, n); return -1; }
        return 0;
    case VFFT_K1_IL_CHAIN3:
        if (rc->il_c3[0] <= 0 || rc->il_c3[1] <= 0 || rc->il_c3[2] <= 0 || est__prod(rc->il_c3, 3) != n)
        { est__say(why, whylen, "chain3 %d.%d.%d does not multiply to %d", rc->il_c3[0], rc->il_c3[1], rc->il_c3[2], n); return -1; }
        if (!*R1 && !*R2)
        { *R2 = rc->il_c3[0]; *R1 = n / rc->il_c3[0]; }
        if (*R2 != rc->il_c3[0] || (long)*R1 * *R2 != n)
        { est__say(why, whylen, "chain3 il_pair %d.%d is not (A*B).leaf of %d.%d.%d", *R1, *R2, rc->il_c3[0], rc->il_c3[1], rc->il_c3[2]); return -1; }
        return 0;
    case VFFT_K1_IL_FLAT:
        if (rc->il_fl_n < 2 || rc->il_fl_n > VFFT_ILFD_MAX_K || est__prod(rc->il_fl, rc->il_fl_n) != n)
        { est__say(why, whylen, "flat chain of %d stages does not multiply to %d (2..%d stages)", rc->il_fl_n, n, VFFT_ILFD_MAX_K); return -1; }
        if (rc->il_rd_n || rc->il_rdw)   /* open decision 5, default: no Rader-stage flat DIT is emitted */
        { est__say(why, whylen, "flat recipe carries a Rader stage inner (il_rd): estimate emits none (sketch 4.4)"); return -1; }
        return 0;
    case VFFT_K1_IL_ZTT:
        if (rc->il_zt_n < 2 || rc->il_zt_n > 7 || est__prod(rc->il_zt, rc->il_zt_n) != n)
        { est__say(why, whylen, "ZTURN-T chain of %d stages does not multiply to %d (2..7 stages)", rc->il_zt_n, n); return -1; }
        return 0;
    case VFFT_K1_IL_FS:
        if (*R1 < 4 || *R2 < 4 || (long)*R1 * *R2 != n)
        { est__say(why, whylen, "four-step split %dx%d is not a split of %d", *R1, *R2, n); return -1; }
        return 0;
    case VFFT_K1_IL_PRIME:
        if ((rc->prime.method != 1 && rc->prime.method != 2) || !rc->prime.in_kind[0] || !rc->prime.in_shape[0])
        { est__say(why, whylen, "prime recipe without its method and inner"); return -1; }
        return 0;
    default:
        est__say(why, whylen, "c2c route %d is not one the door replays", rc->k1_il_route);
        return -1;
    }
}

/* the c2c K=1 record of the request's cell, as the planner fills it (dp_planner_il.h:2207-2240):
 * il_kv is always written (il_kv_raced = 1, else the door races: c2c_oop_create_il.h:85-98) */
static inline void est__entry(const est_recipe_t *rc, int n, int ip, int scr, int T, int R1, int R2,
                              vfft_oop_il_entry_t *e)
{
    memset(e, 0, sizeof *e);
    e->N = n;
    e->K = 1;
    e->kind = VFFT_OOP_KIND_BAILEY2V;
    e->place_ip = ip;
    e->ord_scr = scr;
    e->nthreads = T > 1 ? T : 0;
    e->k1_il_route = rc->k1_il_route;
    e->il_R1 = R1;
    e->il_R2 = R2;
    if (rc->k1_il_route == VFFT_K1_IL_CHAIN3)
        memcpy(e->il_c3, rc->il_c3, sizeof e->il_c3);
    if (rc->k1_il_route == VFFT_K1_IL_FLAT)
    {
        memcpy(e->il_fl, rc->il_fl, sizeof e->il_fl);
        e->il_fl_n = rc->il_fl_n;
        memcpy(e->il_flf, rc->il_flf, sizeof e->il_flf);
        e->il_flf[sizeof e->il_flf - 1] = 0;
        e->il_tw = rc->il_tw;
    }
    if (rc->k1_il_route == VFFT_K1_IL_ZTT)
    {
        memcpy(e->il_zt, rc->il_zt, sizeof e->il_zt);
        e->il_zt_n = rc->il_zt_n;
        e->il_tw = rc->il_tw;
    }
    e->il_kv = rc->il_kv;
    e->il_kv_raced = 1;
}

/* the zr2c child's record, the c2c section mapped 1:1 onto vw2_zr2c_child_t (wisdom2_real_il.h:388-406),
 * fs already in the row's spelling */
static inline void est__zr2c_child(const est_recipe_t *rc, int R1, int R2, const char *fs, vw2_zr2c_child_t *ch)
{
    memset(ch, 0, sizeof *ch);
    ch->route = rc->k1_il_route;
    ch->R1 = R1;
    ch->R2 = R2;
    if (rc->k1_il_route == VFFT_K1_IL_CHAIN3)
        memcpy(ch->c3, rc->il_c3, sizeof ch->c3);
    if (rc->k1_il_route == VFFT_K1_IL_FLAT)
    {
        memcpy(ch->fl, rc->il_fl, sizeof ch->fl);
        ch->fl_n = rc->il_fl_n;
        memcpy(ch->flf, rc->il_flf, sizeof ch->flf);
        ch->flf[sizeof ch->flf - 1] = 0;
    }
    if (rc->k1_il_route == VFFT_K1_IL_ZTT || rc->k1_il_route == VFFT_K1_IL_FS)
    {   /* ZTURN-T: il_ztt; the four-step: its super-band chain, il_sb (wisdom2_real_il.h:446-450) */
        memcpy(ch->zt, rc->il_zt, sizeof ch->zt);
        ch->zt_n = rc->il_zt_n;
    }
    ch->tw = rc->il_tw;
    ch->kv = rc->il_kv;
    ch->bkv = rc->il_bkv;
    if (rc->k1_il_route == VFFT_K1_IL_PRIME)
    {
        ch->pm = rc->prime.method;
        snprintf(ch->pin, sizeof ch->pin, "%s", rc->prime.in_kind);
        snprintf(ch->psh, sizeof ch->psh, "%s", rc->prime.in_shape);
        ch->ptw = rc->prime.in_tw;
        ch->pmt = rc->il_mt;   /* the prime child's threaded arm, on a threaded row only (zr2c writes it then) */
    }
    if (fs && fs[0])
        snprintf(ch->fs, sizeof ch->fs, "%s", fs);
}

/* the four-step child's split of a c2c/zr2c section: (n1, n2) of its 2D child. FS: the one-thread
 * split, or at T > 1 the threaded split il_mt -- open decision 4, default: the threaded four-step
 * carries the threaded split's own child (sketch section 7; the door seeds it under (il_mt, T),
 * k1_commit.h:1538-1552) -- and in place the same split as out of place (open decision 6, default);
 * a prime child's four-step inner: its shape. 0 = the section has no four-step child. */
static inline int est__fs_split(const est_recipe_t *rc, int n, int T, int *n1, int *n2)
{
    if (rc->k1_il_route == VFFT_K1_IL_FS)
    {
        if (T > 1 && rc->il_mt >= 4 && n % rc->il_mt == 0) { *n1 = rc->il_mt; *n2 = n / rc->il_mt; }
        else { *n1 = rc->il_R1; *n2 = rc->il_R2; }
        return 1;
    }
    if (rc->k1_il_route == VFFT_K1_IL_PRIME && !strcmp(rc->prime.in_kind, "fs"))
        return est__split_of(rc->prime.in_shape, n1, n2);
    return 0;
}

/* ---------------------------------------------------------------- the rows */
/* The row(s) a race would bank for this request, into st (memory only). to_rp: the four-step child
 * in the rp_ spelling a race banks today (1, the build) or as the recipe spells it (0, the compare
 * arm). Order: the prime METHOD row before the route row (vw2_oop_bank_k1_il keeps a method row,
 * wisdom2_oop_il.h:515-527); the route row; its threaded tokens; the dir=bwd sibling. Returns 0, or
 * -1 with the reason. */
static inline int est__put_rows(vw2_store_t *st, const est_recipe_t *rc, const est_request_t *rq, int to_rp,
                                char *why, size_t whylen)
{
    const int N = rq->N, T = est__T(rq), ip = (rq->placement == VFFT_PLACEMENT_IN);
    const int c2r = (rq->transform == VFFT_C2R), eng = est__eng(rc, rq);
    char b[128], fsb[1024];
    int R1, R2, n1 = 0, n2 = 0;
    fsb[0] = 0;
    if (eng == EST_ENG_K1)
    {
        const int scr = (rq->order == VFFT_ORDER_SCRAMBLED), route = rc->k1_il_route;
        vfft_oop_il_entry_t e;
        const vw2_rec_t *r;
        vw2_key_t key;
        if (est__c2c_shape(rc, N, &R1, &R2, why, whylen))
            return -1;
        if (est__fs_split(rc, N, T, &n1, &n2) && est__fs_text(rc->fs, n1, n2, to_rp, fsb, sizeof fsb))
        { est__say(why, whylen, "the four-step child's tokens do not parse"); return -1; }
        if (route == VFFT_K1_IL_PRIME)
        {   /* the method row: ord=scr place=ip role=comp lay=il, one thread (wisdom2_oop_il.h:545-551, :570) */
            if (vw2_prime_method_bank(st, N, rc->prime.method, rc->prime.in_kind, rc->prime.in_shape,
                                      rc->prime.in_tw) != VW2_OK)
            { est__say(why, whylen, "the prime method row was refused by its writer"); return -1; }
            if (!strcmp(rc->prime.in_kind, "fs"))
            {   /* a four-step inner's child rides on the METHOD row (k1_commit.h:594-596) */
                vw2_key_t pk;
                vw2__prime_method_key(N, &pk);
                if (!fsb[0] || est__put_fs(st, &pk, fsb))
                { est__say(why, whylen, "the prime cell's four-step inner has no child tokens"); return -1; }
            }
        }
        est__entry(rc, N, ip, scr, T, R1, R2, &e);
        if (vw2_oop_bank_k1_il(st, &e) != VW2_OK)
        { est__say(why, whylen, "the c2c row was refused by its writer (vw2_oop_rec_k1_il)"); return -1; }
        r = vw2__oop_k1_scan_pl(st, N, VW2_LAY_IL, scr, ip ? VW2_PL_IP : VW2_PL_OOP, T);
        if (!r)
        { est__say(why, whylen, "the c2c row is not the door's after banking (its key holds another row)"); return -1; }
        key = r->key;
        if (T > 1 && (route == VFFT_K1_IL_FLAT || route == VFFT_K1_IL_ZTT || route == VFFT_K1_IL_PRIME))
        {   /* the threaded arm on the T row: k1_commit.h:1317 (flat, with its width), 1397 (ZTURN-T), 1448 (prime) */
            snprintf(b, sizeof b, "%d", rc->il_mt);
            if (vw2_update_field(st, &key, "il_mt", b) != VW2_OK) goto tok;
            if (route == VFFT_K1_IL_FLAT)
            {
                snprintf(b, sizeof b, "%d", rc->il_mt_tw);
                if (vw2_update_field(st, &key, "il_mt_tw", b) != VW2_OK) goto tok;
            }
        }
        if (route == VFFT_K1_IL_FS)
        {
            if (T > 1)
            {   /* the threaded split and super-band (k1_commit.h:1543-1551): il_mtsb=0 when none */
                snprintf(b, sizeof b, "%d", rc->il_mt);
                if (vw2_update_field(st, &key, "il_mt", b) != VW2_OK) goto tok;
                if (rc->il_zt_n >= 2) { if (est__ints(b, sizeof b, rc->il_zt, rc->il_zt_n)) goto tok; }
                else snprintf(b, sizeof b, "0");
                if (vw2_update_field(st, &key, "il_mtsb", b) != VW2_OK) goto tok;
            }
            else if (rc->il_kv == 1 && rc->il_zt_n >= 2)
            {   /* the one-thread super-band chain beside il_pair (dp_planner_il.h:2244-2252); never emitted (sketch 4.5) */
                if (est__ints(b, sizeof b, rc->il_zt, rc->il_zt_n) || vw2_update_field(st, &key, "il_sb", b) != VW2_OK) goto tok;
            }
            if (!fsb[0] || est__put_fs(st, &key, fsb))
            { est__say(why, whylen, "the four-step row has no child tokens (fs_*): the door would race it (k1_commit.h:1573-1577)"); return -1; }
        }
        if ((route == VFFT_K1_IL_2P_PURE || route == VFFT_K1_IL_CHAIN3) && rc->il_bkv)
        {   /* the dir=bwd sibling, as the planner banks it (dp_planner_il.h:2266-2290): its own placement */
            vw2_rec_t br;
            const char *w2 = NULL;
            if (vw2_oop_rec_k1_bwd(&br, N, route, R1, R2, rc->il_bkv, 0.0, "race", &w2) != VW2_OK)
            { est__say(why, whylen, "the dir=bwd row was refused (%s)", w2 ? w2 : "?"); return -1; }
            br.key.pl = ip ? VW2_PL_IP : VW2_PL_OOP;
            if (route == VFFT_K1_IL_CHAIN3 &&
                (est__ints(b, sizeof b, rc->il_c3, 3) || vw2_rec_set(&br, 1, "il_chain", b) != VW2_OK))
            { vw2_rec_free(&br); goto tok; }
            if (vw2_bank(st, &br) != VW2_OK)
            { vw2_rec_free(&br); est__say(why, whylen, "the dir=bwd row was refused by the store"); return -1; }
        }
        return 0;
    }
    switch (eng)
    {
    case EST_ENG_ZRM:
        if (vw2_real_il_bank_zrm(st, N, c2r, ip, T, 0.0) != VW2_OK) goto real;
        return 0;
    case EST_ENG_ZRP:
        if (vw2_real_il_bank_zrp(st, N, c2r, ip, T, rc->zrp.R1, rc->zrp.R2, rc->zrp.form, 0.0) != VW2_OK) goto real;
        return 0;
    case EST_ENG_ZTTR:
        if (rc->zttr.nf < 2 || rc->zttr.nf > 8)
        { est__say(why, whylen, "zttr chain of %d stages (2..8)", rc->zttr.nf); return -1; }
        if (vw2_real_il_bank_zttr(st, N, c2r, ip, T, rc->zttr.chain, rc->zttr.nf, rc->zttr.tile, rc->zttr.stk,
                                  rc->zttr.mt, 0.0) != VW2_OK) goto real;
        return 0;
    case EST_ENG_ZRF:
        if (rc->zrf.K < 2 || rc->zrf.K > 10)
        { est__say(why, whylen, "zrf chain of %d stages (2..10, zrf.h:228)", rc->zrf.K); return -1; }
        if (vw2_real_il_bank_zrf(st, N, c2r, ip, T, rc->zrf.chain, rc->zrf.K, rc->zrf.nomsz, rc->zrf.tile,
                                 rc->zrf.mt, 0.0) != VW2_OK) goto real;
        return 0;
    case EST_ENG_ZRB:
        if (vw2_real_il_bank_zrb(st, N, c2r, ip, T, rc->zrb.M, rc->zrb.kind, rc->zrb.shape, rc->zrb.tw,
                                 rc->zrb.mt, 0.0) != VW2_OK) goto real;
        if (!strcmp(rc->zrb.kind, "fs"))
        {   /* a four-step inner's child rides on the zrb row (odd_build.h:731-734) */
            vw2_key_t rk;
            if (!est__split_of(rc->zrb.shape, &n1, &n2) || est__fs_text(rc->fs, n1, n2, to_rp, fsb, sizeof fsb) ||
                !fsb[0])
            { est__say(why, whylen, "zrb's four-step inner has no child tokens"); return -1; }
            vw2_real_il_key(&rk, N, c2r, ip, T);
            if (est__put_fs(st, &rk, fsb)) goto tok;
        }
        return 0;
    case EST_ENG_ZFSR:
    {   /* the split of N/2 and its child (wisdom2_real_il.h:613-645), the writer's own fs_ codec */
        vw2_rec_t all, c2d, crow, cbwd;
        char *sp = fsb, *t;
        int rc2 = -1;
        if (est__fs_text(rc->fs, rc->zfsr.n1, rc->zfsr.n2, to_rp, fsb, sizeof fsb) || !est__fs_has_child(fsb))
        { est__say(why, whylen, "zfsr has no child tokens (fs_*): the door would race it (zrp_build.h:330-334)"); return -1; }
        memset(&all, 0, sizeof all); memset(&c2d, 0, sizeof c2d); memset(&crow, 0, sizeof crow); memset(&cbwd, 0, sizeof cbwd);
        while ((t = vw2__tok(&sp)) != NULL)
        {
            char *eq = strchr(t, '=');
            if (!eq) break;
            *eq = 0;
            if (vw2_rec_set(&all, 1, t, eq + 1) != VW2_OK) break;
        }
        if (!t && vw2__fs_get(&c2d, &all, "fs_", "fs_row_") > 0 && vw2__fs_get(&crow, &all, "fs_row_", "fs_row_bwd_") >= 0 &&
            vw2__fs_get(&cbwd, &all, "fs_row_bwd_", NULL) >= 0)
            rc2 = vw2_real_il_bank_zfsr(st, N, c2r, ip, T, rc->zfsr.n1, rc->zfsr.n2, &c2d, crow.ntok ? &crow : NULL,
                                        cbwd.ntok ? &cbwd : NULL, 0.0);
        vw2_rec_free(&all); vw2_rec_free(&c2d); vw2_rec_free(&crow); vw2_rec_free(&cbwd);
        if (rc2 != VW2_OK) goto real;
        return 0;
    }
    case EST_ENG_ZR2C:
    {   /* the child at N/2, natural, one thread, in the c2c record's words (wisdom2_real_il.h:422-480) */
        vw2_zr2c_child_t ch;
        const int half = N / 2;
        if (N & 1)
        { est__say(why, whylen, "zr2c needs an even N"); return -1; }
        if (est__c2c_shape(rc, half, &R1, &R2, why, whylen))
            return -1;
        if (est__fs_split(rc, half, 1, &n1, &n2) &&
            (est__fs_text(rc->fs, n1, n2, to_rp, fsb, sizeof fsb) || strlen(fsb) >= sizeof ch.fs))
        { est__say(why, whylen, "the zr2c child's four-step tokens do not parse or pass the row's %zu-byte cap "
                                "(wisdom2_real_il.h:405)", sizeof ch.fs); return -1; }
        est__zr2c_child(rc, R1, R2, fsb, &ch);
        if (T < 2)
            ch.pmt = 0;
        if (vw2_real_il_bank_zr2c(st, N, c2r, ip, T, rc->zr2c.route, rc->zr2c.fold_mt, &ch, 0.0) != VW2_OK) goto real;
        return 0;
    }
    default:
        est__say(why, whylen, "engine \"%s\" is not a real engine of the door (wisdom2_real_il.h:7-44)", rc->eng);
        return -1;
    }
tok:
    est__say(why, whylen, "a recipe token was refused by the store's lexical rules (wisdom2.h:462-471)");
    return -1;
real:
    est__say(why, whylen, "the %s row was refused by its writer (wisdom2_real_il.h)", rc->eng);
    return -1;
}

/* ---------------------------------------------------------------- the pre-flight */
static inline int est__preflight(const est_recipe_t *rc, const est_request_t *rq, const vfft_cfg_t *c1,
                                 char *why, size_t whylen);

/* the value of token `name` in a space-joined token text, copied to out; 0 when absent */
static inline int est__fs_get(const char *fs, const char *name, char *out, size_t cap)
{
    const size_t ln = strlen(name);
    const char *p = fs;
    while (p && *p)
    {
        while (*p == ' ') p++;
        if (!strncmp(p, name, ln) && p[ln] == '=')
        {
            const char *v = p + ln + 1;
            const size_t n = strcspn(v, " ");
            if (n >= cap) return 0;
            memcpy(out, v, n);
            out[n] = 0;
            return 1;
        }
        p = strchr(p, ' ');
    }
    return 0;
}
/* the four-step child's row plan -- the 1D c2c cell at N2, in place, natural, one thread
 * (k1_fourstep.h:215-230) -- as a recipe, from its tokens in either spelling: fs_row_* / fs_row_bwd_*
 * (the shipped rows) or fs_rp_<i>_* under fs_rp_k<i> (the 2D tier's). 0 when it carries none. */
static inline int est__fs_row_recipe(const char *fs, int n2, est_recipe_t *rr)
{
    char buf[1024], *sp = buf, *t, rowp[24] = "fs_row_", bwdp[24] = "fs_row_bwd_";   /* "fs_rp_<int>_" fits */
    int have = 0, i;
    memset(rr, 0, sizeof *rr);
    if (strlen(fs) >= sizeof buf)
        return 0;
    memcpy(buf, fs, strlen(fs) + 1);
    while ((t = vw2__tok(&sp)) != NULL)
        if (!strncmp(t, "fs_rp_k", 7) && (i = atoi(t + 7)) > 0)
        {   /* the 2D tier's spelling: which row is the plan and which its backward twin */
            if (strstr(t, "dir=bwd")) snprintf(bwdp, sizeof bwdp, "fs_rp_%d_", i);
            else if (strstr(t, "place=ip") && strstr(t, "ord=nat")) snprintf(rowp, sizeof rowp, "fs_rp_%d_", i);
        }
    memcpy(buf, fs, strlen(fs) + 1);
    sp = buf;
    while ((t = vw2__tok(&sp)) != NULL)
    {
        char *eq = strchr(t, '=');
        const char *nm, *v;
        int a[10], n;
        if (!eq) continue;
        *eq = 0;
        v = eq + 1;
        if (!strncmp(t, bwdp, strlen(bwdp)))
        {   /* the backward twin: its forms are the plan's il_bkv */
            if (!strcmp(t + strlen(bwdp), "il_kv")) rr->il_bkv = atoi(v);
            continue;
        }
        if (strncmp(t, rowp, strlen(rowp)))
            continue;
        nm = t + strlen(rowp);
        have = 1;
        if (!strcmp(nm, "il_route")) rr->k1_il_route = vw2__oop_name_idx(vw2_oop_il_name, VW2_OOP_IL_ROUTE_MAX + 1, v);
        else if (!strcmp(nm, "il_pair") && vw2__oop_split_ints(v, a, 2) == 2) { rr->il_R1 = a[0]; rr->il_R2 = a[1]; }
        else if (!strcmp(nm, "il_chain") && vw2__oop_split_ints(v, a, 3) == 3) memcpy(rr->il_c3, a, sizeof rr->il_c3);
        else if (!strcmp(nm, "il_flat") && (n = vw2__oop_split_ints(v, a, 10)) >= 2) { memcpy(rr->il_fl, a, sizeof(int) * (size_t)n); rr->il_fl_n = n; }
        else if (!strcmp(nm, "il_forms")) snprintf(rr->il_flf, sizeof rr->il_flf, "%s", v);
        else if (!strcmp(nm, "il_tw")) rr->il_tw = atoi(v);
        else if (!strcmp(nm, "il_ztt") && (n = vw2__oop_split_ints(v, a, 7)) >= 2) { memcpy(rr->il_zt, a, sizeof(int) * (size_t)n); rr->il_zt_n = n; }
        else if (!strcmp(nm, "il_kv")) rr->il_kv = atoi(v);
    }
    snprintf(rr->eng, sizeof rr->eng, "k1");
    (void)n2;
    return have;
}
/* A four-step child cannot be built without its nested create, but its two plans can be checked
 * with no create at all: the 2D child's column chain (fs_chain) must multiply to N1 and resolve
 * with the 2D tier's own resolver -- else the tier races its column pool, or a Bluestein column
 * (il2d_tier.h:3440-3510) -- and its row plan at N2 must pass this pre-flight as the c2c cell it
 * is (in place, natural, one thread; k1_fourstep.h:215-230). Returns 0, or -1 with the reason. */
static inline int est__fs_preflight(const char *fs, int n1, int n2, const vfft_cfg_t *c1, char *why, size_t whylen)
{
    char v[96];
    int R[8], m;
    vfft_il2p_fn ff[8], fb[8];
    est_recipe_t rr;
    est_request_t q2;
    if (!est__fs_has_child(fs))
    { est__say(why, whylen, "the four-step %dx%d carries no child (fs_*)", n1, n2); return -1; }
    if (!est__fs_get(fs, "fs_chain", v, sizeof v) || (m = vw2__oop_split_ints(v, R, 8)) < 1 ||
        est__prod(R, m) != n1 || !_il2d_resolve(R, m, ff, fb))
    { est__say(why, whylen, "the four-step's column chain fs_chain=%s does not resolve at N1=%d (il2d_cols.h:93-113): "
                            "the 2D tier would race its column", est__fs_get(fs, "fs_chain", v, sizeof v) ? v : "?", n1); return -1; }
    {   /* a chain with an odd radix replays only with its N1-arm verdict on the row (blu=0: the chain
         * won); without it the tier races the chain against a Bluestein column (il2d_tier.h:3607-3650) */
        int s, odd = 0, multi = 0;
        const char *nm[VFFT_IL2P_COL_MAXFORMS];
        for (s = 0; s < m; s++)
        {
            if (R[s] & 1) odd = 1;
            if (vfft_il2p_col_forms(R[s], 0, nm) > 1) multi = 1;
        }
        if (odd && (!est__fs_get(fs, "fs_blu", v, sizeof v) || atoi(v) != 0))
        { est__say(why, whylen, "the four-step's column chain at N1=%d carries an odd radix and its 2D row no N1-arm "
                                "verdict fs_blu=0: the 2D tier would race chain against Bluestein (il2d_tier.h:3607-3650)", n1); return -1; }
        /* a stage with more than one column form replays only with the row's forms= (il2d_tier.h:3218-3250) */
        if (multi && (!est__fs_get(fs, "fs_forms", v, sizeof v) || !_il2d_apply_forms(R, m, v, ff, fb)))
        { est__say(why, whylen, "the four-step's column chain at N1=%d has a stage with several column forms and its 2D "
                                "row no fitting fs_forms: the 2D tier would race them (il2d_tier.h:3218-3250)", n1); return -1; }
    }
    if (!est__fs_row_recipe(fs, n2, &rr))
    { est__say(why, whylen, "the four-step's 2D child carries no row plan at N2=%d (fs_row_* / fs_rp_*)", n2); return -1; }
    memset(&q2, 0, sizeof q2);
    q2.N = n2;
    q2.transform = VFFT_C2C;
    q2.order = VFFT_ORDER_NATURAL;
    q2.placement = VFFT_PLACEMENT_IN;
    q2.nthreads = 1;
    if (est__preflight(&rr, &q2, c1, v, sizeof v))
    { est__say(why, whylen, "the four-step's row plan at N2=%d does not build: %s", n2, v); return -1; }
    return 0;
}

/* The door's own constructor sequence for the recipe, at one thread, no clock, destroyed again
 * (REPLAY_API section 2; the door sites cited per route). c1 is the request at T = 1 against an
 * empty private store. Returns 0 when the recipe builds as written, -1 with the reason. A nested
 * create is checked without its create: the band, the split, and its child's two plans
 * (est__fs_preflight). */
static inline int est__preflight(const est_recipe_t *rc, const est_request_t *rq, const vfft_cfg_t *c1,
                                 char *why, size_t whylen)
{
    const int N = rq->N, T = est__T(rq), ip = (rq->placement == VFFT_PLACEMENT_IN);
    const int scr = (rq->order == VFFT_ORDER_SCRAMBLED), c2r = (rq->transform == VFFT_C2R);
    const int eng = est__eng(rc, rq);
    int R1, R2;
    if (eng == EST_ENG_K1)
    {
        if (est__c2c_shape(rc, N, &R1, &R2, why, whylen))
            return -1;
        switch (rc->k1_il_route)
        {
        case VFFT_K1_IL_MONO:
            /* out of place the form-resolved pair (c2c_oop_create_il.h:332-339); in place the
             * alias-tolerant solo, form ignored (_k1_il_mono_candidate, k1_commit.h:1750-1766) */
            if (ip ? (!vfft_k1_mono_ilc_fn(N, 0) || !vfft_k1_mono_ilc_fn(N, 1))
                   : (!vfft_k1_mono_il_form_fn(N, rc->il_kv, 0) || !vfft_k1_mono_il_form_fn(N, rc->il_kv, 1)))
            { est__say(why, whylen, "no mono kernel at N=%d (form %d, %s)", N, rc->il_kv, ip ? "in place" : "out of place"); return -1; }
            return 0;
        case VFFT_K1_IL_2P_PURE:
        {   /* vfft_il2p_create + the forms (c2c_oop_create_il.h:168-180, k1_commit.h:100-156); the binder
             * requires the backward to resolve (il/il_execute.h:472) */
            vfft_il2p_plan_t *p = vfft_il2p_create(N, R1, R2);
            int bad;
            if (!p) { est__say(why, whylen, "pair %d.%d does not build (vfft_il2p_create)", R1, R2); return -1; }
            bad = vfft_il2p_apply_kv_forms(p, rc->il_kv) != 0 ||
                  (rc->il_bkv && vfft_il2p_apply_kv_forms_bwd(p, rc->il_bkv) != 0) || !vfft_il2p_bwd_resolves(p);
            vfft_il2p_destroy(p);
            if (bad) { est__say(why, whylen, "pair %d.%d: il_kv 0x%x / il_bkv 0x%x names a kernel this build lacks", R1, R2, rc->il_kv, rc->il_bkv); return -1; }
            return 0;
        }
        case VFFT_K1_IL_CHAIN3:
        {   /* vfft_il3p_create + the forms (c2c_oop_create_il.h:188-211, k1_commit.h:65-95) */
            vfft_il3p_plan_t *p = vfft_il3p_create(N, rc->il_c3[0], rc->il_c3[1], rc->il_c3[2]);
            int bad;
            if (!p) { est__say(why, whylen, "chain3 %d.%d.%d does not build (vfft_il3p_create)", rc->il_c3[0], rc->il_c3[1], rc->il_c3[2]); return -1; }
            bad = vfft_il3p_apply_kv_forms(p, rc->il_kv) != 0 || (rc->il_bkv && vfft_il3p_apply_kv_forms_bwd(p, rc->il_bkv) != 0);
            vfft_il3p_destroy(p);
            if (bad) { est__say(why, whylen, "chain3: il_kv 0x%x / il_bkv 0x%x names a kernel this build lacks", rc->il_kv, rc->il_bkv); return -1; }
            return 0;
        }
        case VFFT_K1_IL_FLAT:
        {   /* natural: create_chain, bwd_ok, forms, tile; scrambled: create_scr_of (c2c_oop_create_il.h:216-246) */
            vfft_ilfd_plan_t *p;
            if (scr)
                p = vfft_ilfd_create_scr_of(N, rc->il_fl, rc->il_fl_n, rc->il_flf, rc->il_tw);
            else
            {
                p = vfft_ilfd_create_chain(N, rc->il_fl, rc->il_fl_n);
                if (p && (!p->bwd_ok || (rc->il_flf[0] && !vfft_ilfd_apply_forms(p, rc->il_flf)) ||
                          (rc->il_tw > 0 && !vfft_ilfd_apply_tw(p, rc->il_tw))))
                { vfft_ilfd_destroy(p); p = NULL; }
            }
            if (!p) { est__say(why, whylen, "flat chain does not build as written (chain, forms \"%s\" or tile %d, %s class)", rc->il_flf, rc->il_tw, scr ? "scrambled" : "natural"); return -1; }
            vfft_ilfd_destroy(p);
            return 0;
        }
        case VFFT_K1_IL_ZTT:
        {   /* vfft_ztt_create_chain_ord + set_tile (c2c_oop_create_il.h:253-271); the create does not
             * enforce the policy floor (policy_il.h:63-67), so it is checked here (REPLAY_API 5.4).
             * open decision 3, default: the formulas clamp the tile onto the raced ladder {1024, 2048}
             * (dp_planner_il.h:1455); this checks the width the recipe names with the create's own law.
             * open decision 7, default: an 8-lane recipe is legality only -- the floor 64 and the
             * 8-lane chain laws (ztt.h:356-364) are the create's and this check's */
            vfft_ztt_plan_t *p;
            if (N < VFFT_POLICY_ZTT_MIN_N) { est__say(why, whylen, "ZTURN-T below its floor %d at %d lanes", VFFT_POLICY_ZTT_MIN_N, VFFT_IL_VW); return -1; }
            p = vfft_ztt_create_chain_ord(N, rc->il_zt, rc->il_zt_n, scr);
            if (p && rc->il_tw > 0 && !vfft_ztt_set_tile(p, (size_t)rc->il_tw)) { vfft_ztt_destroy(p); p = NULL; }
            if (!p) { est__say(why, whylen, "ZTURN-T chain / tile %d does not build (%s class)", rc->il_tw, scr ? "scrambled" : "natural"); return -1; }
            vfft_ztt_destroy(p);
            return 0;
        }
        case VFFT_K1_IL_FS:
        {   /* nested: legality only. The band and the split (k1_fourstep.h:386), the super-band form's
             * law (:387), a threaded split from the race's own ladder (k1_commit.h:1594-1601) */
            if (!vfft_k1fs_band(N)) { est__say(why, whylen, "N=%d is outside the four-step band", N); return -1; }
            if (rc->il_kv == 1 && !scr && R2 % 16) { est__say(why, whylen, "the super-band form needs N2 %% 16 == 0"); return -1; }
            if (T > 1)
            {
                int a[8], bb[8], ns = vfft_k1fs_splits(N, a, bb, 8), i, ok = 0;
                for (i = 0; i < ns; i++) if (a[i] == rc->il_mt) ok = 1;
                if (!ok) { est__say(why, whylen, "threaded split il_mt=%d is not one of the four-step's splits of %d", rc->il_mt, N); return -1; }
            }
            {
                int n1, n2;
                (void)est__fs_split(rc, N, T, &n1, &n2);
                if (!scr && rc->il_zt_n >= 2 && (T > 1 || rc->il_kv == 1))
                {   /* the super-band form's laws (k1_fourstep.h:387, 438-448): its chain multiplies to N1,
                     * resolves, ingests an even R0 in 4..16, closes on a width in 4..32 that tiles N1 */
                    vfft_il2p_fn ff[8], fb[8];
                    const int sbn = rc->il_zt_n, R0 = rc->il_zt[0], wl = rc->il_zt[sbn - 1];
                    if (sbn > 8 || est__prod(rc->il_zt, sbn) != n1 || n2 % 16 || R0 < 4 || R0 > 16 || (R0 & 1) ||
                        wl < 4 || wl > 32 || n1 % (R0 * wl) || !_il2d_resolve(rc->il_zt, sbn, ff, fb))
                    { est__say(why, whylen, "the super-band chain does not fit the split %dx%d (k1_fourstep.h:387,438-448)", n1, n2); return -1; }
                }
                return est__fs_preflight(rc->fs, n1, n2, c1, why, whylen);
            }
        }
        case VFFT_K1_IL_PRIME:
        {   /* the prime cell's admission (policy_il.h:173, :299) and its own build (_ilprime_create_banked's
             * replay, k1_commit.h:582-617: _ilprime_desc_parse then _ilprime_build_with) */
            _ilprime_inner_desc_t d;
            vfft_ilprime_plan_t *p;
            int n1, n2;
            if (!vfft_policy_prime_cell(N) || vfft_policy_prime_cell_unservable(N))
            { est__say(why, whylen, "N=%d is not a servable prime cell (policy_il.h:173,299)", N); return -1; }
            if (rc->prime.method == 1 && !vfft_policy_is_prime(N))
            { est__say(why, whylen, "Rader needs a prime N"); return -1; }
            if (!_ilprime_desc_parse(&d, rc->prime.in_kind, rc->prime.in_shape, rc->prime.in_tw))
            { est__say(why, whylen, "prime inner \"%s %s\" does not parse", rc->prime.in_kind, rc->prime.in_shape); return -1; }
            if (d.kind == 4)
            {   /* a four-step inner is a nested create: its band, the method's inner length, and its
                 * child's two plans */
                if (!est__split_of(rc->prime.in_shape, &n1, &n2) || !vfft_k1fs_band(n1 * n2))
                { est__say(why, whylen, "the four-step inner %s is outside the band", rc->prime.in_shape); return -1; }
                if (!est__prime_inner_len_ok(N, rc->prime.method, (long)n1 * n2))
                { est__say(why, whylen, "the four-step inner %dx%d is not a %s length at N=%d (%s; il_prime.h:249-250, "
                                        "299-318)", n1, n2, rc->prime.method == 1 ? "Rader" : "Bluestein", N,
                           rc->prime.method == 1 ? "M = N - 1" : "an even M >= 2N - 1"); return -1; }
                return est__fs_preflight(rc->fs, n1, n2, c1, why, whylen);
            }
            p = _ilprime_build_with(N, rc->prime.method == 1, &d);
            if (!p) { est__say(why, whylen, "%s with inner %s %s does not build", rc->prime.method == 1 ? "Rader" : "Bluestein", rc->prime.in_kind, rc->prime.in_shape); return -1; }
            vfft_ilprime_destroy(p);
            return 0;
        }
        default:
            return -1;
        }
    }
    {   /* real: the door's builder at T = 1, then vfft_destroy -- at one thread the release is a no-op
         * (vfft_execute.h:401, threads.h:553-559; REPLAY_API section 2.0) */
        struct vfft_plan_s *h = NULL;
        switch (eng)
        {
        case EST_ENG_ZRM:
            if (N > VFFT_ZRM_MAX_N || !vfft_zrm_fn(N, c2r)) { est__say(why, whylen, "no real mono kernel at N=%d", N); return -1; }
            h = _zrm_build_plan(c1, N);
            break;
        case EST_ENG_ZRP:
            h = _zrp_build_pair(c1, N, rc->zrp.R1, rc->zrp.R2, rc->zrp.form);
            break;
        case EST_ENG_ZTTR:   /* open decision 7, default: zrp and ZTT-r are 4-lane engines; at 8 lanes the
                              * builders return NULL (zttr.h:164-172) and the recipe is refused here */
            h = _zttr_build_plan(c1, N, rc->zttr.chain, rc->zttr.nf, rc->zttr.tile, rc->zttr.stk);
            break;
        case EST_ENG_ZRF:
            h = _zrf_build_plan(c1, N, rc->zrf.chain, rc->zrf.K, rc->zrf.nomsz, rc->zrf.tile);
            break;
        case EST_ENG_ZRB:
        {
            _ilprime_inner_desc_t d;
            int n1, n2;
            if (!_ilprime_desc_parse(&d, rc->zrb.kind, rc->zrb.shape, rc->zrb.tw))
            { est__say(why, whylen, "zrb inner \"%s %s\" does not parse", rc->zrb.kind, rc->zrb.shape); return -1; }
            if (d.kind == 4)
            {   /* the band, zrb's own length law M >= (3N-1)/2 (vfft_zrb_min_m, zrb.h:59; vfft_zrb_create
                 * refuses below it, :87 -- the replay then fails and odd_build.h:754 races the odd
                 * real), and the child's two plans */
                if (!est__split_of(rc->zrb.shape, &n1, &n2) || (long)n1 * n2 != rc->zrb.M || !vfft_k1fs_band(rc->zrb.M))
                { est__say(why, whylen, "zrb's four-step inner %s is not M=%d in the band", rc->zrb.shape, rc->zrb.M); return -1; }
                if (rc->zrb.M < vfft_zrb_min_m(N))
                { est__say(why, whylen, "zrb's M=%d is below (3N-1)/2 = %d at N=%d (zrb.h:59,87)", rc->zrb.M, vfft_zrb_min_m(N), N); return -1; }
                return est__fs_preflight(rc->fs, n1, n2, c1, why, whylen);
            }
            h = _zrb_build_plan(c1, N, rc->zrb.M, &d);   /* odd_build.h:296 */
            break;
        }
        case EST_ENG_ZFSR:
            /* nested: the band and the split's law (zfsr.h:86-101) and the child */
            if (!vfft_zfsr_band(N) || (long)rc->zfsr.n1 * rc->zfsr.n2 != N / 2 || rc->zfsr.n1 % VFFT_ZFSR_TB ||
                rc->zfsr.n2 % (2 * VFFT_ZFSR_TB))
            { est__say(why, whylen, "zfsr split %dx%d is outside the real four-step's band or law", rc->zfsr.n1, rc->zfsr.n2); return -1; }
            return est__fs_preflight(rc->fs, rc->zfsr.n1, rc->zfsr.n2, c1, why, whylen);
        case EST_ENG_ZR2C:
        {   /* the door's replay steps (_zr2c_replay, zr2c_build.h:674-700) up to the composite */
            vw2_zr2c_child_t ch;
            vfft_il_cand_t c;
            _zr2c_prime_t pr;
            char fsb[1024];
            int n1, n2;
            if (est__c2c_shape(rc, N / 2, &R1, &R2, why, whylen))
                return -1;
            fsb[0] = 0;
            if (est__fs_split(rc, N / 2, 1, &n1, &n2))
            {   /* a four-step child (or a prime child's four-step inner) is a nested create */
                if (rc->k1_il_route == VFFT_K1_IL_FS && !vfft_k1fs_band(N / 2))
                { est__say(why, whylen, "zr2c four-step child at %d is outside the band", N / 2); return -1; }
                if (rc->k1_il_route == VFFT_K1_IL_PRIME &&
                    (!vfft_policy_prime_cell(N / 2) || (rc->prime.method == 1 && !vfft_policy_is_prime(N / 2)) ||
                     !vfft_k1fs_band(n1 * n2) || !est__prime_inner_len_ok(N / 2, rc->prime.method, (long)n1 * n2)))
                { est__say(why, whylen, "zr2c prime child at %d: its four-step inner %dx%d is not a %s inner of a prime "
                                        "cell (il_prime.h:249, 299-318)", N / 2, n1, n2,
                           rc->prime.method == 1 ? "Rader" : "Bluestein"); return -1; }
                return est__fs_preflight(rc->fs, n1, n2, c1, why, whylen);
            }
            if (rc->k1_il_route == VFFT_K1_IL_PRIME && !vfft_policy_prime_cell(N / 2))
            { est__say(why, whylen, "zr2c prime child at %d is not a prime cell", N / 2); return -1; }
            est__zr2c_child(rc, R1, R2, fsb, &ch);
            if (!_zr2c_prime_of_child(&pr, &ch))
            { est__say(why, whylen, "zr2c prime child's inner does not parse"); return -1; }
            _zr2c_cand_of_child(&c, &ch);
            h = _zr2c_build_route(c1, N, rc->zr2c.route, &c, &pr, NULL);
            break;
        }
        default:
            est__say(why, whylen, "engine \"%s\" is not a real engine of the door", rc->eng);
            return -1;
        }
        if (!h)
        { est__say(why, whylen, "%s does not build as written at N=%d (the door would race the cell)", rc->eng, N); return -1; }
        vfft_destroy((vfft_plan)h);
        return 0;
    }
}

/* ---------------------------------------------------------------- what the door built */
/* 1 when the handle is the recipe's engine and every threaded arm the recipe asked for took
 * (REPLAY_API section 2.16: the binds decline silently and leave the plan serial).
 * open decision 2, default: at every T > 1 the recipe carries the T = 8 fitted arm (unfitted for
 * T != 8); the glue builds whatever arm it names at the request's T and checks that it bound. */
static inline int est__check_plan(const struct vfft_plan_s *h, const est_recipe_t *rc, const est_request_t *rq,
                                  int *serial, char *why, size_t whylen)
{
    const int T = est__T(rq), eng = est__eng(rc, rq);
    *serial = 0;
    if (eng == EST_ENG_K1)
    {   /* the in-place door sets the engine pointers only (c2c_ip_create_il.h:136-147), so the engine
         * is read from them in both placements */
        int want_mt = rc->il_mt, got_mt = 0, ok;
        switch (rc->k1_il_route)
        {
        case VFFT_K1_IL_MONO: ok = h->k1_mono_ilf != 0; break;
        case VFFT_K1_IL_2P_PURE: ok = h->k1il2p != NULL; break;
        case VFFT_K1_IL_CHAIN3: ok = h->k1il3p != NULL; break;
        case VFFT_K1_IL_FLAT: ok = h->k1ilfd != NULL; if (ok) got_mt = h->k1ilfd->mt; break;
        case VFFT_K1_IL_ZTT: ok = h->k1ztt != NULL; if (ok) got_mt = h->k1ztt->mt; break;
        case VFFT_K1_IL_PRIME: ok = h->k1ilpr != NULL; if (ok) got_mt = h->k1ilpr->mt; break;
        case VFFT_K1_IL_FS:
            ok = h->k1fs != NULL && h->k1fs->N1 == (T > 1 ? rc->il_mt : rc->il_R1);
            want_mt = got_mt = 0;
            break;
        default: ok = 0; break;
        }
        if (!ok)
        { est__say(why, whylen, "the door built another engine than the recipe's (route %d)", rc->k1_il_route); return 0; }
        if (T > 1 && want_mt != got_mt)
        { *serial = 1; est__say(why, whylen, "the threaded arm il_mt=%d did not take (bound %d)", want_mt, got_mt); return 0; }
        return 1;
    }
    {
        int ok = 0, want = 0, got = 0;
        switch (eng)
        {
        case EST_ENG_ZRM: ok = h->zrm != 0; break;
        case EST_ENG_ZRP: ok = h->zrp != NULL; break;
        case EST_ENG_ZTTR: ok = h->zttr != NULL; if (ok) { want = rc->zttr.mt; got = h->zttr->mt; } break;
        case EST_ENG_ZFSR: ok = h->zfsr != NULL; break;
        case EST_ENG_ZRF: ok = h->zrf != NULL; if (ok) { want = rc->zrf.mt; got = h->zrf->mt; } break;
        case EST_ENG_ZRB: ok = h->zrb != NULL; if (ok) { want = rc->zrb.mt; got = h->zrb->mt; } break;
        case EST_ENG_ZR2C:
            ok = h->zr2c_kid != NULL && h->zr2c_route == rc->zr2c.route;
            if (ok && T > 1)
            {
                want = (rc->zr2c.fold_mt != 0) * 10 + (rc->k1_il_route == VFFT_K1_IL_PRIME ? rc->il_mt : 0);
                got = (h->zr2c_fold_mt != 0) * 10 + (h->zr2c_kid->ilp ? h->zr2c_kid->ilp->mt : 0);
            }
            break;
        default: break;
        }
        if (!ok)
        { est__say(why, whylen, "the door built another engine than the recipe's (%s)", rc->eng); return 0; }
        if (T > 1 && want != got)
        { *serial = 1; est__say(why, whylen, "a threaded arm of %s did not take (asked %d, bound %d)", rc->eng, want, got); return 0; }
        return 1;
    }
}

/* ---------------------------------------------------------------- the rows the glue wrote */
/* A row's payload as one text: its payload tokens (section 1, wisdom2.h:182) "name=value", sorted
 * and space-joined, so a re-bank that only reorders tokens reads the same. Measurement tokens
 * (section 2) are not recipe. Returns a malloc'ed string, NULL when out of memory. */
static int est__cmp_str(const void *a, const void *b) { return strcmp(*(const char *const *)a, *(const char *const *)b); }
static inline char *est__payload_text(const vw2_rec_t *r)
{
    size_t len = 1, off = 0;
    int i, n = 0;
    char **v, *pool, *out;
    for (i = 0; i < r->ntok; i++)
        if (r->tok[i].sect == 1)
            len += strlen(r->tok[i].name) + strlen(r->tok[i].val) + 2;
    v = (char **)malloc(((size_t)r->ntok + 1) * sizeof *v);
    pool = (char *)malloc(len);
    out = (char *)malloc(len);
    if (!v || !pool || !out)
    { free(v); free(pool); free(out); return NULL; }
    for (i = 0; i < r->ntok; i++)
        if (r->tok[i].sect == 1)
        {
            v[n++] = pool + off;
            off += (size_t)sprintf(pool + off, "%s=%s", r->tok[i].name, r->tok[i].val) + 1;
        }
    qsort(v, (size_t)n, sizeof *v, est__cmp_str);
    out[0] = 0;
    for (i = 0, off = 0; i < n; i++)
        off += (size_t)sprintf(out + off, "%s%s", i ? " " : "", v[i]);
    free(v);
    free(pool);
    return out;
}
/* the rows of S as the glue wrote them (keys and payload texts), taken after vw2_disown */
typedef struct { vw2_key_t key; char *pay; } est__snap_t;
static inline void est__snap_free(est__snap_t *s, int n)
{
    int i;
    for (i = 0; s && i < n; i++)
        free(s[i].pay);
    free(s);
}
static inline est__snap_t *est__snap_take(const vw2_store_t *st)
{
    est__snap_t *s = (est__snap_t *)calloc((size_t)st->nrec + 1, sizeof *s);
    int i;
    if (!s)
        return NULL;
    for (i = 0; i < st->nrec; i++)
    {
        s[i].key = st->rec[i].key;
        if (!(s[i].pay = est__payload_text(&st->rec[i])))
        { est__snap_free(s, st->nrec); return NULL; }
    }
    return s;
}
/* 1 = the "own" row r still carries the payload the glue wrote at its key */
static inline int est__snap_same(const est__snap_t *s, int n, const vw2_rec_t *r)
{
    int i, same = 0;
    for (i = 0; i < n; i++)
        if (vw2_key_eq(&s[i].key, &r->key))
        {
            char *p = est__payload_text(r);
            same = p && !strcmp(p, s[i].pay);
            free(p);
            break;
        }
    return same;
}

/* ---------------------------------------------------------------- est_build_1d */
/* est_build_1d with the report: the status (EST_*) and the detectors' deltas. The contract is
 * est_build_1d's (estimate_recipe.h). */
static inline struct vfft_plan_s *est_build_1d_ex(const est_recipe_t *rc, const est_request_t *rq,
                                                  char *why, size_t whylen, est_build_report_t *rep)
{
    est_build_report_t lrep;
    vfft_config_t pub;
    vfft_cfg_t c, c1;
    struct vfft_wisdom_s *S = NULL, *P = NULL;
    struct vfft_plan_s *h = NULL;
    est__snap_t *snap = NULL;
    const char *env;
    long c0, r0, s0;
    int d0, n0;
    int T, ip, scr, serial = 0, i;
    if (!rep)
        rep = &lrep;
    memset(rep, 0, sizeof *rep);
    rep->status = EST_REFUSED;
    if (why && whylen)
        why[0] = 0;
    if (!rc || !rq)
    { est__say(why, whylen, "no recipe or no request"); return NULL; }
    T = est__T(rq);
    ip = (rq->placement == VFFT_PLACEMENT_IN);
    scr = (rq->order == VFFT_ORDER_SCRAMBLED);
    rep->nested = est_recipe_nested(rc, rq);
    /* the request estimate serves: 1D interleaved FP64 c2c / r2c / c2r, one transform
     * (open decision 1, default: howmany = 1, batches refused -- no batch field exists) */
    if (rq->transform != VFFT_C2C && rq->transform != VFFT_R2C && rq->transform != VFFT_C2R)
    { est__say(why, whylen, "transform %d: estimate plans c2c, r2c and c2r only", rq->transform); return NULL; }
    if (rq->N < 2 || rq->nthreads < 0 ||
        (rq->order != VFFT_ORDER_NATURAL && rq->order != VFFT_ORDER_SCRAMBLED) ||
        (rq->placement != VFFT_PLACEMENT_OUT && rq->placement != VFFT_PLACEMENT_IN))
    { est__say(why, whylen, "request out of range (N=%d, order %d, placement %d, nthreads %d)", rq->N, rq->order, rq->placement, rq->nthreads); return NULL; }
    if (rq->transform != VFFT_C2C && scr)
    { est__say(why, whylen, "a real plan is natural only (vfft.c:1940-1957)"); return NULL; }
    if (est__eng(rc, rq) == EST_ENG_BAD)
    { est__say(why, whylen, "recipe engine \"%s\" does not serve a %s request", rc->eng, rq->transform == VFFT_C2C ? "c2c" : "real"); return NULL; }
    /* 1. the in-place scrambled prime cell at T = 1: its route row and its prime method row share one
     *    key (wisdom2_oop_il.h:515-527, key :545-551) and the door's scan takes eng=k1 rows only
     *    (wisdom2_oop_rows.h:103), so the library would race it (k1_commit.h:993-1004) */
    if (rq->transform == VFFT_C2C && rc->k1_il_route == VFFT_K1_IL_PRIME && ip && scr && T == 1)
    { est__say(why, whylen, "the in-place scrambled prime cell at one thread cannot replay: its route row and "
                            "its prime method row share one key (wisdom2_oop_il.h:515-527,545-551), so the library "
                            "would race it"); return NULL; }
    if ((env = est_env_steering()) != NULL)
    { est__say(why, whylen, "%s is set: a VFFT_* variable steers what a create builds (REPLAY_API 1.3)", env); return NULL; }
    /* the public request, translated by the library's own front door (vfft_api.h:95) */
    memset(&pub, 0, sizeof pub);
    pub.transform = (vfft_transform_t)rq->transform;
    pub.rank = 1;
    pub.n[0] = rq->N;
    pub.howmany = 1;
    pub.layout = VFFT_LAYOUT_INTERLEAVED;
    pub.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    pub.order = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    pub.nthreads = T;   /* wisdom_dir stays NULL: c.wisdom is set below */
    if (!_vfft_api_translate(&pub, &c))
    { est__say(why, whylen, "the front door refuses the request: %s", _vfft_err_buf); return NULL; }
    vfft_env_init();   /* env.h:96: the pre-flight's tables in the create's FP mode */
    _vfft_create_lock();   /* vfft.c:2485 (recursive): the pre-flight touches the create's process statics */
    c0 = est_clock_reads;
    r0 = _vfft_create_race_count;
    s0 = vfft_measure_scopes();
    d0 = _def_loaded;        /* the library's own store (vfft.c:496-505) */
    n0 = _vfft_api_nstores;  /* the directory stores (vfft_api.h:41) */
    /* 2. the pre-flight at T = 1 against an empty private store */
    P = vfft_child_store_new();
    if (!P)
    { est__say(why, whylen, "out of memory"); goto out; }
    c1 = c;
    c1.nthreads = 1;
    c1.wisdom = (vfft_wisdom *)P;
    c1.wisdom_write = 0;
    c1.recalibrate = 0;
    if (est__preflight(rc, rq, &c1, why, whylen))
        goto out;
    /* 3. the rows, into a private in-memory store, then disowned: from here an own row is the door's */
    S = vfft_child_store_new();
    if (!S)
    { est__say(why, whylen, "out of memory"); goto out; }
    if (est__put_rows(&S->vw2, rc, rq, 1, why, whylen))
        goto out;
    vw2_disown(&S->vw2);   /* wisdom2.h:1486 */
    rep->rows_before = S->vw2.nrec;
    if (!(snap = est__snap_take(&S->vw2)))   /* the rows as written: what a re-bank must leave as it is */
    { est__say(why, whylen, "out of memory"); goto out; }
    /* 4. the library's own create, nested so wisdom_write = 0 holds (vfft.c:2523-2524) */
    c.wisdom = (vfft_wisdom *)S;
    c.wisdom_write = 0;
    c.recalibrate = 0;
    _vfft_err_buf[0] = 0;
    _vfft_create_depth++;   /* race_scope.h:106 */
    h = (struct vfft_plan_s *)_vfft_create_cfg(&c);
    if (--_vfft_create_depth == 0)
        _vfft_scope_create_done();   /* race_scope.h:621: leave a scope a clock read entered */
    /* 5. the no-measurement proof */
    rep->clock_reads = est_clock_reads - c0;
    rep->races = _vfft_create_race_count - r0;
    rep->scopes = vfft_measure_scopes() - s0;
    rep->rows_after = S->vw2.nrec;
    for (i = 0; i < S->vw2.nrec; i++)
        if (S->vw2.rec[i].own)
        {   /* a row the door re-banked (vfft_child_store_raced's test, wisdom2_child.h:164-173) */
            const int same = est__snap_same(snap, rep->rows_before, &S->vw2.rec[i]);
            if (!rep->rewritten++ || (!same && !rep->changed))
                vw2__key_format(&S->vw2.rec[i].key, rep->rewritten_key, sizeof rep->rewritten_key);
            rep->changed += !same;
        }
    if (rep->clock_reads || rep->races || rep->scopes || rep->rows_after != rep->rows_before)
    {
        rep->status = EST_MEASURED;
        est__say(why, whylen, "the create measured (clock reads %ld, races %ld, scopes %ld, rows %d -> %d): "
                              "the recipe did not replay as written", rep->clock_reads, rep->races, rep->scopes,
                 rep->rows_before, rep->rows_after);
        if (h) { vfft_destroy((vfft_plan)h); h = NULL; }
        goto out;
    }
    if (_def_loaded != d0 || _vfft_api_nstores != n0)
    {   /* the user's store was opened (vfft.c:496-505, vfft_api.h:41): never by estimate */
        rep->status = EST_MEASURED;
        est__say(why, whylen, "the create opened the user's wisdom store: estimate never reads it");
        if (h) { vfft_destroy((vfft_plan)h); h = NULL; }
        goto out;
    }
    if (rep->changed)
    {   /* re-banked with another payload and nothing measured: the plan is not the recipe */
        rep->status = EST_REWRITTEN;
        est__say(why, whylen, "the door re-wrote %d recipe row(s) with another payload, measuring nothing (first: %s): "
                              "the plan is not the recipe", rep->changed, rep->rewritten_key);
        if (h) { vfft_destroy((vfft_plan)h); h = NULL; }
        goto out;
    }
    if (!h)
    {
        rep->status = EST_NOPLAN;
        est__say(why, whylen, "the door built no plan (measuring nothing): %s", _vfft_err_buf[0] ? _vfft_err_buf : "no reason given");
        goto out;
    }
    /* 6. the recipe's engine, and every threaded arm it asked for */
    if (!est__check_plan(h, rc, rq, &serial, why, whylen))
    {
        rep->status = serial ? EST_SERIAL : EST_NOPLAN;
        vfft_destroy((vfft_plan)h);
        h = NULL;
        goto out;
    }
    if (h->nthreads > 1)   /* the caller's pin, as the public create arms it (vfft_api.h:207-213) */
        for (i = VFFT_FORWARD; i <= VFFT_BACKWARD; i++)
            if (h->k1_go[i])
            {
                h->k1_go_mt[i] = h->k1_go[i];
                h->k1_go[i] = _vfft_k1_go_mt;
            }
    rep->status = EST_BUILT;
out:
    est__snap_free(snap, rep->rows_before);
    vfft_child_store_free(S);   /* wisdom2_child.h:47: a plan keeps no pointer to it */
    vfft_child_store_free(P);
    _vfft_create_unlock();
    if (rep->status != EST_BUILT && h)
    {
        vfft_destroy((vfft_plan)h);
        h = NULL;
    }
    return h;
}

/* est_build_1d: the contract in estimate_recipe.h */
static inline struct vfft_plan_s *est_build_1d(const est_recipe_t *rc, const est_request_t *rq,
                                               char *why, size_t whylen)
{
    return est_build_1d_ex(rc, rq, why, whylen, NULL);
}

#endif /* EST_BUILD_H */
