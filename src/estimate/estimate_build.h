/* estimate_build.h -- the 1D estimate's build for ANY request: c2c through the LIBRARY's build, r2c / c2r
 * through this folder's real build. Tracked, not wired into the library.
 *
 * THE C2C BUILD IS THE LIBRARY'S (src/core/il/estimate/c2c_estimate_il.h, est_build_c2c): the pre-flight,
 * the rows into a private store held in memory, the door's replay with wisdom_write = 0, the
 * no-measurement detectors (est__meter_read, est__replay) and the check of the plan the door built.
 * est_build_any_ex hands a c2c recipe to it on the front door's own translation of the request
 * (_vfft_api_translate, vfft_api.h).
 *
 * THE REAL BUILD lives here, in the same shape, on the library's pieces: est__meter_read before the
 * pre-flight; the real pre-flight (the door's real builders at T = 1, then vfft_destroy -- at one
 * thread the release is a no-op, vfft_execute.h:401, threads.h:553-559; REPLAY_API section 2.0); the
 * real rows through the library's writers (vw2_real_il_bank_*); est__replay (disown, the snapshot of
 * the rows, the nested create, the detectors); the real plan's check.
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

/* the engines by their row names (wisdom2_real_il.h:7-44) */
enum { EST_ENG_K1, EST_ENG_ZRM, EST_ENG_ZRP, EST_ENG_ZTTR, EST_ENG_ZR2C, EST_ENG_ZFSR, EST_ENG_ZRF, EST_ENG_ZRB,
       EST_ENG_BAD };
static inline int est__eng(const est_any_recipe_t *rc, const est_request_t *rq)
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

/* ---------------------------------------------------------------- the real rows */
/* the zr2c child's record: the c2c section mapped 1:1 onto vw2_zr2c_child_t (wisdom2_real_il.h:388-406),
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

/* The row(s) a race would bank for any request, into st (memory only). c2c: the library's writer
 * (est_put_rows). Real: the engine's writer (wisdom2_real_il.h:100-682). to_rp: the four-step child in
 * the rp_ spelling a race banks today (1, the build) or as the recipe spells it (0, the compare arm).
 * Returns 0, or -1 with the reason. */
static inline int est__put_rows_any(vw2_store_t *st, const est_any_recipe_t *rc, const est_request_t *rq, int to_rp,
                                    char *why, size_t whylen)
{
    const int N = rq->N, T = est__T(rq), ip = (rq->placement == VFFT_PLACEMENT_IN);
    const int c2r = (rq->transform == VFFT_C2R), eng = est__eng(rc, rq);
    char fsb[1024];
    int R1, R2, n1 = 0, n2 = 0;
    fsb[0] = 0;
    if (eng == EST_ENG_K1)
        return est_put_rows(st, &rc->c, rq, to_rp, why, whylen);
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
            return est_refuse(why, whylen, "zttr chain of %d stages (2..8)", rc->zttr.nf);
        if (vw2_real_il_bank_zttr(st, N, c2r, ip, T, rc->zttr.chain, rc->zttr.nf, rc->zttr.tile, rc->zttr.stk,
                                  rc->zttr.mt, 0.0) != VW2_OK) goto real;
        return 0;
    case EST_ENG_ZRF:
        if (rc->zrf.K < 2 || rc->zrf.K > 10)
            return est_refuse(why, whylen, "zrf chain of %d stages (2..10, zrf.h:228)", rc->zrf.K);
        if (vw2_real_il_bank_zrf(st, N, c2r, ip, T, rc->zrf.chain, rc->zrf.K, rc->zrf.nomsz, rc->zrf.tile,
                                 rc->zrf.mt, 0.0) != VW2_OK) goto real;
        return 0;
    case EST_ENG_ZRB:
        if (vw2_real_il_bank_zrb(st, N, c2r, ip, T, rc->zrb.M, rc->zrb.kind, rc->zrb.shape, rc->zrb.tw,
                                 rc->zrb.mt, 0.0) != VW2_OK) goto real;
        if (!strcmp(rc->zrb.kind, "fs"))
        {   /* a four-step inner's child rides on the zrb row (odd_build.h:731-734) */
            vw2_key_t rk;
            if (!est__split_of(rc->zrb.shape, &n1, &n2) || est__fs_text(rc->c.fs, n1, n2, to_rp, fsb, sizeof fsb) ||
                !fsb[0])
                return est_refuse(why, whylen, "zrb's four-step inner has no child tokens");
            vw2_real_il_key(&rk, N, c2r, ip, T);
            if (est__put_fs(st, &rk, fsb)) goto tok;
        }
        return 0;
    case EST_ENG_ZFSR:
    {   /* the split of N/2 and its child (wisdom2_real_il.h:613-645), the writer's own fs_ codec */
        vw2_rec_t all, c2d, crow, cbwd;
        char *sp = fsb, *t;
        int rc2 = -1;
        if (est__fs_text(rc->c.fs, rc->zfsr.n1, rc->zfsr.n2, to_rp, fsb, sizeof fsb) || !est__fs_has_child(fsb))
            return est_refuse(why, whylen, "zfsr has no child tokens (fs_*): the door would race it (zrp_build.h:330-334)");
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
            return est_refuse(why, whylen, "zr2c needs an even N");
        if (est__c2c_shape(&rc->c, half, &R1, &R2, why, whylen))
            return -1;
        if (est__fs_split(&rc->c, half, 1, &n1, &n2) &&
            (est__fs_text(rc->c.fs, n1, n2, to_rp, fsb, sizeof fsb) || strlen(fsb) >= sizeof ch.fs))
            return est_refuse(why, whylen, "the zr2c child's four-step tokens do not parse or pass the row's %u-byte cap "
                              "(wisdom2_real_il.h:405)", (unsigned)sizeof ch.fs);
        est__zr2c_child(&rc->c, R1, R2, fsb, &ch);
        if (T < 2)
            ch.pmt = 0;
        if (vw2_real_il_bank_zr2c(st, N, c2r, ip, T, rc->zr2c.route, rc->zr2c.fold_mt, &ch, 0.0) != VW2_OK) goto real;
        return 0;
    }
    default:
        return est_refuse(why, whylen, "engine \"%s\" is not a real engine of the door (wisdom2_real_il.h:7-44)", rc->eng);
    }
tok:
    return est_refuse(why, whylen, "a recipe token was refused by the store's lexical rules (wisdom2.h:462-471)");
real:
    return est_refuse(why, whylen, "the %s row was refused by its writer (wisdom2_real_il.h)", rc->eng);
}

/* ---------------------------------------------------------------- the real pre-flight */
/* the door's real builders for the recipe at T = 1 against c1 (an empty private store), then
 * vfft_destroy. A nested create (zfsr, a four-step inner or child) is checked without its create: the
 * band, the split, its child's two plans (the library's est__fs_preflight). Returns 0, or -1 with the
 * reason. */
static inline int est__preflight_real(const est_any_recipe_t *rc, const est_request_t *rq, const vfft_cfg_t *c1,
                                      char *why, size_t whylen)
{
    const int N = rq->N, c2r = (rq->transform == VFFT_C2R), eng = est__eng(rc, rq);
    struct vfft_plan_s *h = NULL;
    int R1, R2;
    switch (eng)
    {
    case EST_ENG_ZRM:
        if (N > VFFT_ZRM_MAX_N || !vfft_zrm_fn(N, c2r))
            return est_refuse(why, whylen, "no real mono kernel at N=%d", N);
        h = _zrm_build_plan(c1, N);
        break;
    case EST_ENG_ZRP:
        h = _zrp_build_pair(c1, N, rc->zrp.R1, rc->zrp.R2, rc->zrp.form);
        break;
    case EST_ENG_ZTTR:   /* zrp and ZTT-r are 4-lane engines; at 8 lanes the builders return NULL (zttr.h:164-172) */
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
            return est_refuse(why, whylen, "zrb inner \"%s %s\" does not parse", rc->zrb.kind, rc->zrb.shape);
        if (d.kind == 4)
        {   /* the band, zrb's own length law M >= (3N-1)/2 (vfft_zrb_min_m, zrb.h:59; vfft_zrb_create
             * refuses below it, :87 -- the replay then fails and odd_build.h:754 races the odd real),
             * and the child's two plans */
            if (!est__split_of(rc->zrb.shape, &n1, &n2) || (long)n1 * n2 != rc->zrb.M || !vfft_k1fs_band(rc->zrb.M))
                return est_refuse(why, whylen, "zrb's four-step inner %s is not M=%d in the band", rc->zrb.shape, rc->zrb.M);
            if (rc->zrb.M < vfft_zrb_min_m(N))
                return est_refuse(why, whylen, "zrb's M=%d is below (3N-1)/2 = %d at N=%d (zrb.h:59,87)", rc->zrb.M,
                                  vfft_zrb_min_m(N), N);
            return est__fs_preflight(rc->c.fs, n1, n2, why, whylen);
        }
        h = _zrb_build_plan(c1, N, rc->zrb.M, &d);   /* odd_build.h:296 */
        break;
    }
    case EST_ENG_ZFSR:
        /* nested: the band and the split's law (zfsr.h:86-101) and the child */
        if (!vfft_zfsr_band(N) || (long)rc->zfsr.n1 * rc->zfsr.n2 != N / 2 || rc->zfsr.n1 % VFFT_ZFSR_TB ||
            rc->zfsr.n2 % (2 * VFFT_ZFSR_TB))
            return est_refuse(why, whylen, "zfsr split %dx%d is outside the real four-step's band or law", rc->zfsr.n1,
                              rc->zfsr.n2);
        return est__fs_preflight(rc->c.fs, rc->zfsr.n1, rc->zfsr.n2, why, whylen);
    case EST_ENG_ZR2C:
    {   /* the door's replay steps (_zr2c_replay, zr2c_build.h:674-700) up to the composite */
        const est_recipe_t *ch = &rc->c;
        vw2_zr2c_child_t zc;
        vfft_il_cand_t c;
        _zr2c_prime_t pr;
        int n1, n2;
        if (est__c2c_shape(ch, N / 2, &R1, &R2, why, whylen))
            return -1;
        if (est__fs_split(ch, N / 2, 1, &n1, &n2))
        {   /* a four-step child (or a prime child's four-step inner) is a nested create */
            if (ch->k1_il_route == VFFT_K1_IL_FS && !vfft_k1fs_band(N / 2))
                return est_refuse(why, whylen, "zr2c four-step child at %d is outside the band", N / 2);
            if (ch->k1_il_route == VFFT_K1_IL_PRIME &&
                (!vfft_policy_prime_cell(N / 2) || (ch->prime.method == 1 && !vfft_policy_is_prime(N / 2)) ||
                 !vfft_k1fs_band(n1 * n2) || !est__prime_inner_len_ok(N / 2, ch->prime.method, (long)n1 * n2)))
                return est_refuse(why, whylen, "zr2c prime child at %d: its four-step inner %dx%d is not a %s inner of a "
                                  "prime cell (il_prime.h:249, 299-318)", N / 2, n1, n2,
                                  ch->prime.method == 1 ? "Rader" : "Bluestein");
            return est__fs_preflight(ch->fs, n1, n2, why, whylen);
        }
        if (ch->k1_il_route == VFFT_K1_IL_PRIME && !vfft_policy_prime_cell(N / 2))
            return est_refuse(why, whylen, "zr2c prime child at %d is not a prime cell", N / 2);
        est__zr2c_child(ch, R1, R2, "", &zc);
        if (!_zr2c_prime_of_child(&pr, &zc))
            return est_refuse(why, whylen, "zr2c prime child's inner does not parse");
        _zr2c_cand_of_child(&c, &zc);
        h = _zr2c_build_route(c1, N, rc->zr2c.route, &c, &pr, NULL);
        break;
    }
    default:
        return est_refuse(why, whylen, "engine \"%s\" is not a real engine of the door", rc->eng);
    }
    if (!h)
        return est_refuse(why, whylen, "%s does not build as written at N=%d (the door would race the cell)", rc->eng, N);
    vfft_destroy((vfft_plan)h);
    return 0;
}

/* 1 when the handle is the recipe's real engine and every threaded arm the recipe asked for took
 * (REPLAY_API section 2.16: the binds decline silently and leave the plan serial) */
static inline int est__check_plan_real(const struct vfft_plan_s *h, const est_any_recipe_t *rc, const est_request_t *rq,
                                       int *serial, char *why, size_t whylen)
{
    const int T = est__T(rq), eng = est__eng(rc, rq);
    int ok = 0, want = 0, got = 0;
    *serial = 0;
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
            want = (rc->zr2c.fold_mt != 0) * 10 + (rc->c.k1_il_route == VFFT_K1_IL_PRIME ? rc->c.il_mt : 0);
            got = (h->zr2c_fold_mt != 0) * 10 + (h->zr2c_kid->ilp ? h->zr2c_kid->ilp->mt : 0);
        }
        break;
    default: break;
    }
    if (!ok)
    {
        est_refuse(why, whylen, "the door built another engine than the recipe's (%s)", rc->eng);
        return 0;
    }
    if (T > 1 && want != got)
    {
        *serial = 1;
        est_refuse(why, whylen, "a threaded arm of %s did not take (asked %d, bound %d)", rc->eng, want, got);
        return 0;
    }
    return 1;
}

/* ---------------------------------------------------------------- est_build_any */
/* the contract in estimate_recipe.h */
static inline struct vfft_plan_s *est_build_any_ex(const est_any_recipe_t *rc, const est_request_t *rq, char *why,
                                                   size_t whylen, est_build_report_t *rep)
{
    est_build_report_t lrep;
    vfft_config_t pub;
    vfft_cfg_t c, c1;
    struct vfft_wisdom_s *S = NULL, *P = NULL;
    struct vfft_plan_s *h = NULL;
    const char *env;
    est__meter_t m0;
    long c0;
    int T, ip, scr, serial = 0, i;
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
    {   /* the real build, on the library's pieces (est_build_c2c's shape) */
        vfft_env_init();          /* env.h:96: the pre-flight's tables in the create's FP mode */
        _vfft_create_lock();      /* vfft.c:2485 (recursive) */
        _vfft_create_depth++;     /* the whole build inside a create: a clock read enters a scope */
        est__meter_read(&m0);
        if (!(P = vfft_child_store_new()))
        {
            est_refuse(why, whylen, "out of memory");
            goto real_out;
        }
        c1 = c;
        c1.nthreads = 1;
        c1.wisdom = (vfft_wisdom *)P;
        c1.wisdom_write = 0;
        c1.recalibrate = 0;
        if (est__preflight_real(rc, rq, &c1, why, whylen))
            goto real_out;
        if (!(S = vfft_child_store_new()))
        {
            est_refuse(why, whylen, "out of memory");
            goto real_out;
        }
        if (est__put_rows_any(&S->vw2, rc, rq, 1, why, whylen))
            goto real_out;
        h = est__replay(S, &c, &m0, rep, why, whylen);
        if (h && !est__check_plan_real(h, rc, rq, &serial, why, whylen))
        {
            rep->status = serial ? EST_SERIAL : EST_NOPLAN;
            vfft_destroy((vfft_plan)h);
            h = NULL;
        }
        else if (h)
            rep->status = EST_BUILT;
    real_out:
        vfft_child_store_free(S);
        vfft_child_store_free(P);
        if (--_vfft_create_depth == 0)
            _vfft_scope_create_done();
        _vfft_create_unlock();
    }
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
