/* estimate_build.h -- the 1D estimate's build of ANY request for this folder's tools: c2c through the library's
 * est_build_c2c (src/core/il/estimate/c2c_estimate_il.h), r2c / c2r through the library's est_build_real
 * (src/core/il/estimate/real_estimate_il.h, moved there from this file on 2026-11-01). Both take the same
 * steps: the pre-flight, the rows into a private store held in memory, the door's replay with
 * wisdom_write = 0, the no-measurement detectors (est__meter_read, est__replay) and the check of the plan the
 * door built. est_build_any_ex hands the recipe to them on the front door's own translation of the request
 * (_vfft_api_translate, vfft_api.h).
 *
 * WHAT THIS FOLDER ADDS FOR BOTH (the tool's guards, not the library's):
 * - the clock probe: est_clock_reads, fed by VFFT_CLOCK_PROBE (race_scope.h:636-642) at every library
 *   clock read. A read during a build means the create measured: the plan is destroyed and the status
 *   is EST_MEASURED. (The library's own build counts races and scopes, not clock reads.)
 * - the environment guard (REPLAY_API section 1.3): a VFFT_* variable steers what a create builds (an env
 *   pin beats a banked row, e.g. VFFT_ZRP in zrp_build.h:971-985, VFFT_IL_KV in k1_commit.h:131-140),
 *   so every VFFT_* name refuses the build, except the ones that only log or only name a store.
 * - the caller's pin of a T > 1 plan, as vfft_create arms it (vfft_api.h).
 *
 * THE INCLUDER (REPLAY_API section 1.2): one TU that defines
 *     static long est_clock_reads;
 *     #define VFFT_CLOCK_PROBE (est_clock_reads++ ...)
 * then includes vfft.c FIRST (nothing of the library before it), then estimate_1d.h, then this. */
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


/* 1 = the recipe holds a nested create (a four-step child), which no pre-flight can build
 * (REPLAY_API section 0.3): the c2c four-step, the prime cell's or zrb's four-step inner, zfsr,
 * a zr2c four-step child or a zr2c prime child with a four-step inner */
static inline int est_recipe_nested(const est_any_recipe_t *rc, const est_request_t *rq)
{
    const int eng = est__eng(rc, rq);
    if (eng == EST_ENG_K1 || eng == EST_ENG_ZR2C)
        return rc->c.k1_il_route == VFFT_K1_IL_FS ||
               (rc->c.k1_il_route == VFFT_K1_IL_PRIME && !strcmp(rc->c.prime.in_kind, "fs"));
    if (eng == EST_ENG_ZFSR)
        return 1;
    if (eng == EST_ENG_ZRB)
        return !strcmp(rc->zrb.kind, "fs");
    return 0;
}

/* ---------------------------------------------------------------- the environment guard */
/* Every VFFT_* name in the process environment refuses the build, except the ones that only log or
 * only name a store -- which the build never opens (checked after each create). The scan reads the
 * environment itself, so a variable the library learns later is refused too. On Windows the names
 * are matched without case: getenv there ignores case, so "vfft_il_kv" is the library's VFFT_IL_KV.
 * Returns the first steering name, or NULL. */
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

/* The row(s) a race would bank for any request, into st (memory only): c2c the library's est_put_rows, real the
 * library's est_put_rows_real (il/estimate/real_estimate_il.h). to_rp: the four-step child in the rp_ spelling a
 * race banks today (1, the build) or as the recipe spells it (0, the compare arm). Returns 0, or -1 with the
 * reason. */
static inline int est__put_rows_any(vw2_store_t *st, const est_any_recipe_t *rc, const est_request_t *rq, int to_rp,
                                    char *why, size_t whylen)
{
    if (est__eng(rc, rq) == EST_ENG_K1)
        return est_put_rows(st, &rc->c, rq, to_rp, why, whylen);
    return est_put_rows_real(st, rc, rq, to_rp, why, whylen);
}

/* ---------------------------------------------------------------- est_build_any */
/* the contract in estimate_recipe.h */
static inline struct vfft_plan_s *est_build_any_ex(const est_any_recipe_t *rc, const est_request_t *rq, char *why,
                                                   size_t whylen, est_build_report_t *rep)
{
    est_build_report_t lrep;
    vfft_config_t pub;
    vfft_cfg_t c;
    struct vfft_plan_s *h = NULL;
    const char *env;
    long c0;
    int T, ip, scr, i;
    if (!rep)
        rep = &lrep;
    memset(rep, 0, sizeof *rep);
    rep->status = EST_REFUSED;
    if (why && whylen)
        why[0] = 0;
    if (!rc || !rq)
    {
        est_refuse(why, whylen, "no recipe or no request");
        return NULL;
    }
    T = est__T(rq);
    ip = (rq->placement == VFFT_PLACEMENT_IN);
    scr = (rq->order == VFFT_ORDER_SCRAMBLED);
    /* the request estimate serves: 1D interleaved FP64 c2c / r2c / c2r, one transform
     * (open decision 1, default: howmany = 1, batches refused -- no batch field exists) */
    if (rq->transform != VFFT_C2C && rq->transform != VFFT_R2C && rq->transform != VFFT_C2R)
    {
        est_refuse(why, whylen, "transform %d: estimate plans c2c, r2c and c2r only", rq->transform);
        return NULL;
    }
    if (rq->N < 2 || rq->nthreads < 0 || (rq->order != VFFT_ORDER_NATURAL && rq->order != VFFT_ORDER_SCRAMBLED) ||
        (rq->placement != VFFT_PLACEMENT_OUT && rq->placement != VFFT_PLACEMENT_IN))
    {
        est_refuse(why, whylen, "request out of range (N=%d, order %d, placement %d, nthreads %d)", rq->N, rq->order,
                   rq->placement, rq->nthreads);
        return NULL;
    }
    if (rq->transform != VFFT_C2C && scr)
    {
        est_refuse(why, whylen, "a real plan is natural only (vfft.c:1940-1957)");
        return NULL;
    }
    if (est__eng(rc, rq) == EST_ENG_BAD)
    {
        est_refuse(why, whylen, "recipe engine \"%s\" does not serve a %s request", rc->eng,
                   rq->transform == VFFT_C2C ? "c2c" : "real");
        return NULL;
    }
    if ((env = est_env_steering()) != NULL)
    {
        est_refuse(why, whylen, "%s is set: a VFFT_* variable steers what a create builds (REPLAY_API 1.3)", env);
        return NULL;
    }
    /* the public request, translated by the library's own front door (vfft_api.h) */
    memset(&pub, 0, sizeof pub);
    pub.transform = (vfft_transform_t)rq->transform;
    pub.rank = 1;
    pub.n[0] = rq->N;
    pub.howmany = 1;
    pub.layout = VFFT_LAYOUT_INTERLEAVED;
    pub.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    pub.order = scr ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    pub.nthreads = T;   /* wisdom_dir stays NULL: the build sets its own store */
    if (!_vfft_api_translate(&pub, &c))
    {
        est_refuse(why, whylen, "the front door refuses the request: %s", _vfft_err_buf);
        return NULL;
    }
    c0 = est_clock_reads;
    if (rq->transform == VFFT_C2C)
        h = est_build_c2c(&rc->c, rq, &c, rep, why, whylen);   /* the library's build */
    else
        h = est_build_real(rc, rq, &c, rep, why, whylen);      /* the library's real build */
    if (est_clock_reads != c0)
    {   /* the tool's probe: a library clock read during the build is a measurement */
        if (h)
            vfft_destroy((vfft_plan)h);
        h = NULL;
        rep->status = EST_MEASURED;
        est_refuse(why, whylen, "the create read the library's clock %ld time(s): the recipe did not replay as written",
                   est_clock_reads - c0);
    }
    if (h && h->nthreads > 1)   /* the caller's pin, as the public create arms it (vfft_api.h) */
        for (i = VFFT_FORWARD; i <= VFFT_BACKWARD; i++)
            if (h->k1_go[i])
            {
                h->k1_go_mt[i] = h->k1_go[i];
                h->k1_go[i] = _vfft_k1_go_mt;
            }
    return h;
}

static inline struct vfft_plan_s *est_build_any(const est_any_recipe_t *rc, const est_request_t *rq, char *why,
                                                size_t whylen)
{
    return est_build_any_ex(rc, rq, why, whylen, NULL);
}

#endif /* EST_BUILD_H */
