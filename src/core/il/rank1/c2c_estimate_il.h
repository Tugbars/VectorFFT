/* c2c_estimate_il.h — the ESTIMATE create (VFFT_FLAG_ESTIMATE): a 1D interleaved c2c plan built
 * from the formulas' recipe (il/planning/estimate_il.h) through the door's own replay, with
 * nothing measured.
 *
 * vfft_create (vfft_api.h) calls _vfft_create_estimate for a request the front door admitted under
 * the flag (_vfft_api_translate). It reads this machine, computes the recipe (est_plan_1d) and
 * builds it (est_build_c2c):
 *   1. THE PRE-FLIGHT. A row the door cannot build as written is no error to the door: it marks the
 *      row stale and races the cell (c2c_oop_create_il.h:416-441, c2c_ip_create_il.h:94-119). So the
 *      door's own constructor sequence runs first, at one thread, and what it built is destroyed.
 *      A four-step child is a nested create, so it is checked without one: its column chain with
 *      the 2D tier's own resolver, its row plan by this same pre-flight.
 *   2. THE ROWS. The recipe is written as the rows a race would bank (est_put_rows) into a store held
 *      in memory (vfft_child_store_new, wisdom2_child.h:40), then disowned (vw2_disown,
 *      wisdom2.h:1486): from there on a row marked own is one the door banked.
 *   3. THE DOOR. The library's own create (_vfft_create_cfg) runs against that store, nested so its
 *      wisdom_write = 0 holds (vfft.c:2523-2524): the replay of a banked verdict builds the handle,
 *      binds the executor and holds the pool, as for any plan served from wisdom. The caller's store
 *      is not opened (the front door skips wisdom_dir under the flag) and the library's own store is
 *      not loaded: cfg.wisdom is set (vfft.c:2276).
 *   4. NOTHING MEASURED. Read before the pre-flight and after the create: the races counted
 *      (_vfft_create_race_count, vfft.c:255), the race scopes entered (vfft_measure_scopes,
 *      vfft_measure.h:112 -- the whole build runs inside a create, so any clock read enters one,
 *      race_scope.h:612-618), the private store's row count, whether the library's store was loaded
 *      (_def_loaded, vfft.c:496) or a directory store opened (_vfft_api_nstores, vfft_api.h:41).
 *      Any change: the plan is destroyed and the create refused. A row the door re-banked at an
 *      existing key without measuring (k1_commit.h:1558-1564; il2d_tier.h:4413-4424) is compared
 *      with the row written: a changed payload means the plan is not the recipe, so it is
 *      destroyed and the create refused as well.
 *   5. THE RECIPE'S PLAN. The handle must carry the recipe's engine, and every threaded arm the
 *      recipe names must have bound (a bind that declines leaves the plan serial in silence).
 * An estimate create never hands back a plan that measured.
 *
 * The store detector sees a FIRST load of the library's store only: once the process has loaded it
 * (any create without wisdom_dir), a nested create that fell to _default_wisdom() would read it
 * unseen. None does: every IL nested create sets .wisdom (k1_fourstep.h:403, 606; k1_commit.h:296,
 * 301; the 2D tier's).
 *
 * POSITION IN vfft.c: included by vfft_api.h after the directory stores' table it reads; the
 * engines, the doors and the store are all defined above. */
#ifndef VFFT_IL_C2C_ESTIMATE_IL_H
#define VFFT_IL_C2C_ESTIMATE_IL_H

#include "il/planning/estimate_il.h"

/* the outcome of a build */
enum
{
    EST_BUILT = 0,    /* a plan, built from the recipe with nothing measured */
    EST_REFUSED = 1,  /* the recipe does not build as written */
    EST_MEASURED = 2, /* the create measured or read a wisdom store (plan destroyed) */
    EST_SERIAL = 3,   /* a threaded arm the recipe names did not bind (plan destroyed) */
    EST_NOPLAN = 4,   /* the door built no plan and measured nothing */
    EST_REWRITTEN = 5 /* the door re-banked a recipe row with another payload, measuring nothing:
                       * the plan is not the recipe (plan destroyed) */
};

/* what a build's detectors saw */
typedef struct
{
    int status;                 /* EST_* */
    long races, scopes;         /* their moves over the pre-flight and the create; 0 when EST_BUILT */
    int rows_before, rows_after;/* the private store's rows around the create */
    int rewritten;              /* rows the door re-banked without measuring */
    int changed;                /* of those, rows whose payload is no longer the recipe's */
    char rewritten_key[192];    /* the first such row's key (the first changed one, when any) */
} est_build_report_t;

/* ── the pre-flight ─────────────────────────────────────────────────────── */

static inline int est__preflight(const est_recipe_t *rc, const est_request_t *rq, char *why, size_t whylen);

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
/* the four-step child's row plan -- the c2c cell at N2, in place, natural, one thread
 * (k1_fourstep.h:215-230) -- as a recipe, from its tokens in either spelling: fs_row_* /
 * fs_row_bwd_*, or fs_rp_<i>_* under fs_rp_k<i> (the 2D tier's). 0 when it carries none. */
static inline int est__fs_row_recipe(const char *fs, est_recipe_t *rr)
{
    char buf[1024], *sp = buf, *t, rowp[24] = "fs_row_", bwdp[24] = "fs_row_bwd_";
    int have = 0, i;
    memset(rr, 0, sizeof *rr);
    if (strlen(fs) >= sizeof buf)
        return 0;
    memcpy(buf, fs, strlen(fs) + 1);
    while ((t = vw2__tok(&sp)) != NULL)
        if (!strncmp(t, "fs_rp_k", 7) && (i = atoi(t + 7)) > 0)
        {   /* the 2D tier's spelling: which row is the plan, which its backward twin */
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
        else if (!strcmp(nm, "il_flat") && (n = vw2__oop_split_ints(v, a, VFFT_ILFD_MAX_K)) >= 2) { memcpy(rr->il_fl, a, sizeof(int) * (size_t)n); rr->il_fl_n = n; }
        else if (!strcmp(nm, "il_forms")) snprintf(rr->il_flf, sizeof rr->il_flf, "%s", v);
        else if (!strcmp(nm, "il_tw")) rr->il_tw = atoi(v);
        else if (!strcmp(nm, "il_ztt") && (n = vw2__oop_split_ints(v, a, VFFT_ZTT_MAX_NF)) >= 2) { memcpy(rr->il_zt, a, sizeof(int) * (size_t)n); rr->il_zt_n = n; }
        else if (!strcmp(nm, "il_kv")) rr->il_kv = atoi(v);
    }
    return have;
}
/* A four-step child cannot be built without its nested create, but its two plans can be checked
 * with none: the 2D child's column chain fs_chain must multiply to N1 and resolve with the 2D tier's
 * own resolver -- else the tier races its column pool or a Bluestein column (il2d_tier.h:3440-3510)
 * -- and its row plan at N2 must pass this pre-flight as the c2c cell it is (in place, natural, one
 * thread; k1_fourstep.h:215-230). Returns 0, or -1 with the reason. */
static inline int est__fs_preflight(const char *fs, int n1, int n2, char *why, size_t whylen)
{
    char v[96];
    int R[8], m;
    vfft_il2p_fn ff[8], fb[8];
    est_recipe_t rr;
    est_request_t q2;
    if (!est__fs_has_child(fs))
        return est_refuse(why, whylen, "the four-step %dx%d carries no child (fs_*)", n1, n2);
    if (!est__fs_get(fs, "fs_chain", v, sizeof v) || (m = vw2__oop_split_ints(v, R, 8)) < 1 || est_prod(R, m) != n1 ||
        !_il2d_resolve(R, m, ff, fb))
        return est_refuse(why, whylen, "the four-step's column chain fs_chain=%s does not resolve at N1 = %d "
                          "(il2d_cols.h:93-113): the 2D tier would race its column",
                          est__fs_get(fs, "fs_chain", v, sizeof v) ? v : "?", n1);
    {   /* a chain with an odd radix replays only with the N1 arm's verdict on its row (fs_blu=0: the
         * chain won), else the tier races it against a Bluestein column (il2d_tier.h:3607-3650); a
         * stage with more than one column form only with the row's forms= (il2d_tier.h:3218-3250) */
        int s, odd = 0, multi = 0;
        const char *nm[VFFT_IL2P_COL_MAXFORMS];
        for (s = 0; s < m; s++)
        {
            if (R[s] & 1) odd = 1;
            if (vfft_il2p_col_forms(R[s], 0, nm) > 1) multi = 1;
        }
        if (odd && (!est__fs_get(fs, "fs_blu", v, sizeof v) || atoi(v) != 0))
            return est_refuse(why, whylen, "the four-step's column chain at N1 = %d carries an odd radix and its 2D row "
                              "no N1 arm verdict fs_blu=0: the 2D tier would race it against a Bluestein column", n1);
        if (multi && (!est__fs_get(fs, "fs_forms", v, sizeof v) || !_il2d_apply_forms(R, m, v, ff, fb)))
            return est_refuse(why, whylen, "the four-step's column chain at N1 = %d has a stage with several column "
                              "forms and its 2D row no fitting fs_forms: the 2D tier would race them", n1);
    }
    if (!est__fs_row_recipe(fs, &rr))
        return est_refuse(why, whylen, "the four-step's 2D child carries no row plan at N2 = %d", n2);
    memset(&q2, 0, sizeof q2);
    q2.N = n2;
    q2.transform = VFFT_C2C;
    q2.order = VFFT_ORDER_NATURAL;
    q2.placement = VFFT_PLACEMENT_IN;
    q2.nthreads = 1;
    if (est__preflight(&rr, &q2, v, sizeof v))
        return est_refuse(why, whylen, "the four-step's row plan at N2 = %d does not build: %s", n2, v);
    return 0;
}
/* the length a prime cell's four-step inner must have, checked before its nested create (a wrong M
 * fails the replay and the door races the prime pool, k1_commit.h:615): Rader M = N - 1
 * (il_prime.h:299-318; _ilprime_fs_inner_build, k1_commit.h:288-289), Bluestein an even
 * M >= 2N - 1 (il_prime.h:249-250) */
static inline int est__prime_inner_len_ok(int N, int method, long M)
{
    if (method == 1)
        return M == (long)N - 1;
    return M >= 2L * N - 1 && !(M & 1);
}
/* The door's own constructor sequence for the recipe at one thread, the plans destroyed again.
 * Returns 0 when the recipe builds as written, -1 with the reason. */
static inline int est__preflight(const est_recipe_t *rc, const est_request_t *rq, char *why, size_t whylen)
{
    const int N = rq->N, T = rq->nthreads > 1 ? rq->nthreads : 1, ip = (rq->placement == VFFT_PLACEMENT_IN);
    const int scr = (rq->order == VFFT_ORDER_SCRAMBLED);
    int R1, R2;
    if (est__c2c_shape(rc, N, &R1, &R2, why, whylen))
        return -1;
    switch (rc->k1_il_route)
    {
    case VFFT_K1_IL_MONO:
        /* out of place the form-resolved solo (c2c_oop_create_il.h:332-339); in place the
         * alias-tolerant solo, form ignored (_k1_il_mono_candidate, k1_commit.h:1750-1766) */
        if (ip ? (!vfft_k1_mono_ilc_fn(N, 0) || !vfft_k1_mono_ilc_fn(N, 1))
               : (!vfft_k1_mono_il_form_fn(N, rc->il_kv, 0) || !vfft_k1_mono_il_form_fn(N, rc->il_kv, 1)))
            return est_refuse(why, whylen, "no solo kernel at N = %d (form %d, %s)", N, rc->il_kv,
                              ip ? "in place" : "out of place");
        return 0;
    case VFFT_K1_IL_2P_PURE:
    {   /* vfft_il2p_create and the forms (c2c_oop_create_il.h:168-180, k1_commit.h:100-156); the binder
         * needs the backward to resolve (il/il_execute.h:472) */
        vfft_il2p_plan_t *p = vfft_il2p_create(N, R1, R2);
        int bad;
        if (!p)
            return est_refuse(why, whylen, "the pair %d.%d does not build (vfft_il2p_create)", R1, R2);
        bad = vfft_il2p_apply_kv_forms(p, rc->il_kv) != 0 ||
              (rc->il_bkv && vfft_il2p_apply_kv_forms_bwd(p, rc->il_bkv) != 0) || !vfft_il2p_bwd_resolves(p);
        vfft_il2p_destroy(p);
        if (bad)
            return est_refuse(why, whylen, "the pair %d.%d: il_kv 0x%x / backward 0x%x names a kernel this build lacks",
                              R1, R2, rc->il_kv, rc->il_bkv);
        return 0;
    }
    case VFFT_K1_IL_CHAIN3:
    {   /* vfft_il3p_create and the forms (c2c_oop_create_il.h:188-211, k1_commit.h:65-95) */
        vfft_il3p_plan_t *p = vfft_il3p_create(N, rc->il_c3[0], rc->il_c3[1], rc->il_c3[2]);
        int bad;
        if (!p)
            return est_refuse(why, whylen, "chain3 %d.%d.%d does not build (vfft_il3p_create)", rc->il_c3[0], rc->il_c3[1],
                              rc->il_c3[2]);
        bad = vfft_il3p_apply_kv_forms(p, rc->il_kv) != 0 || (rc->il_bkv && vfft_il3p_apply_kv_forms_bwd(p, rc->il_bkv) != 0);
        vfft_il3p_destroy(p);
        if (bad)
            return est_refuse(why, whylen, "chain3: il_kv 0x%x / backward 0x%x names a kernel this build lacks", rc->il_kv,
                              rc->il_bkv);
        return 0;
    }
    case VFFT_K1_IL_FLAT:
    {   /* natural: the chain, bwd_ok, the forms, the tile; scrambled: create_scr_of (c2c_oop_create_il.h:216-246) */
        vfft_ilfd_plan_t *p;
        if (scr)
            p = vfft_ilfd_create_scr_of(N, rc->il_fl, rc->il_fl_n, rc->il_flf, rc->il_tw);
        else
        {
            p = vfft_ilfd_create_chain(N, rc->il_fl, rc->il_fl_n);
            if (p && (!p->bwd_ok || (rc->il_flf[0] && !vfft_ilfd_apply_forms(p, rc->il_flf)) ||
                      (rc->il_tw > 0 && !vfft_ilfd_apply_tw(p, rc->il_tw))))
            {
                vfft_ilfd_destroy(p);
                p = NULL;
            }
        }
        if (!p)
            return est_refuse(why, whylen, "the flat chain does not build as written (forms \"%s\", tile %d, %s class)",
                              rc->il_flf, rc->il_tw, scr ? "scrambled" : "natural");
        vfft_ilfd_destroy(p);
        return 0;
    }
    case VFFT_K1_IL_ZTT:
    {   /* vfft_ztt_create_chain_ord and set_tile (c2c_oop_create_il.h:253-271); the create does not
         * enforce the policy's floor (policy_il.h:63-67), so it is checked here */
        vfft_ztt_plan_t *p;
        if (N < VFFT_POLICY_ZTT_MIN_N)
            return est_refuse(why, whylen, "ZTURN-T below its floor %d at %d lanes", VFFT_POLICY_ZTT_MIN_N, VFFT_IL_VW);
        p = vfft_ztt_create_chain_ord(N, rc->il_zt, rc->il_zt_n, scr);
        if (p && rc->il_tw > 0 && !vfft_ztt_set_tile(p, (size_t)rc->il_tw))
        {
            vfft_ztt_destroy(p);
            p = NULL;
        }
        if (!p)
            return est_refuse(why, whylen, "the ZTURN-T chain with tile %d does not build (%s class)", rc->il_tw,
                              scr ? "scrambled" : "natural");
        vfft_ztt_destroy(p);
        return 0;
    }
    case VFFT_K1_IL_FS:
    {   /* a nested create: the band and the split (k1_fourstep.h:386), the super-band form's law
         * (:387), a threaded split from the race's own ladder (k1_commit.h:1594-1601), the child */
        int n1, n2;
        if (!vfft_k1fs_band(N))
            return est_refuse(why, whylen, "N = %d is outside the four-step's band", N);
        if (rc->il_kv == 1 && !scr && R2 % 16)
            return est_refuse(why, whylen, "the four-step's super-band form needs N2 %% 16 == 0");
        if (T > 1)
        {
            int a[VFFT_K1FS_MAX_SPLITS], bb[VFFT_K1FS_MAX_SPLITS], i, ok = 0;
            const int ns = vfft_k1fs_splits(N, a, bb, VFFT_K1FS_MAX_SPLITS);
            for (i = 0; i < ns; i++)
                if (a[i] == rc->il_mt) ok = 1;
            if (!ok)
                return est_refuse(why, whylen, "the threaded split il_mt=%d is not one of the four-step's splits of %d",
                                  rc->il_mt, N);
        }
        (void)est__fs_split(rc, N, T, &n1, &n2);
        if (!scr && rc->il_zt_n >= 2 && (T > 1 || rc->il_kv == 1))
        {   /* the super-band form's laws (k1_fourstep.h:387, 438-448): its chain multiplies to N1,
             * resolves, ingests an even R0 in 4..16, closes on a width in 4..32 that tiles N1 */
            vfft_il2p_fn ff[8], fb[8];
            const int sbn = rc->il_zt_n, R0 = rc->il_zt[0], wl = rc->il_zt[sbn - 1];
            if (sbn > 8 || est_prod(rc->il_zt, sbn) != n1 || n2 % 16 || R0 < 4 || R0 > 16 || (R0 & 1) || wl < 4 ||
                wl > 32 || n1 % (R0 * wl) || !_il2d_resolve(rc->il_zt, sbn, ff, fb))
                return est_refuse(why, whylen, "the super-band chain does not fit the split %dx%d", n1, n2);
        }
        return est__fs_preflight(rc->fs, n1, n2, why, whylen);
    }
    case VFFT_K1_IL_PRIME:
    {   /* the prime cell's admission (policy_il.h:173, 299) and its own build, as the replay makes it
         * (_ilprime_create_banked, k1_commit.h:582-617: _ilprime_desc_parse, then _ilprime_build_with) */
        _ilprime_inner_desc_t d;
        vfft_ilprime_plan_t *p;
        int n1, n2;
        if (!vfft_policy_prime_cell(N) || vfft_policy_prime_cell_unservable(N))
            return est_refuse(why, whylen, "N = %d is not a prime cell the door serves", N);
        if (rc->prime.method == 1 && !vfft_policy_is_prime(N))
            return est_refuse(why, whylen, "Rader needs a prime N");
        if (!_ilprime_desc_parse(&d, rc->prime.in_kind, rc->prime.in_shape, rc->prime.in_tw))
            return est_refuse(why, whylen, "the prime cell's inner \"%s %s\" does not parse", rc->prime.in_kind,
                              rc->prime.in_shape);
        if (d.kind == 4)
        {   /* a four-step inner is a nested create: its band, the method's inner length, its child */
            if (!est__split_of(rc->prime.in_shape, &n1, &n2) || !vfft_k1fs_band(n1 * n2))
                return est_refuse(why, whylen, "the four-step inner %s is outside the band", rc->prime.in_shape);
            if (!est__prime_inner_len_ok(N, rc->prime.method, (long)n1 * n2))
                return est_refuse(why, whylen, "the four-step inner %dx%d is not a %s length at N = %d", n1, n2,
                                  rc->prime.method == 1 ? "Rader" : "Bluestein", N);
            return est__fs_preflight(rc->fs, n1, n2, why, whylen);
        }
        p = _ilprime_build_with(N, rc->prime.method == 1, &d);
        if (!p)
            return est_refuse(why, whylen, "%s with the inner %s %s does not build", rc->prime.method == 1 ? "Rader" : "Bluestein",
                              rc->prime.in_kind, rc->prime.in_shape);
        vfft_ilprime_destroy(p);
        return 0;
    }
    default:
        return est_refuse(why, whylen, "route %d is not one the door replays", rc->k1_il_route);
    }
}

/* ── the plan the door built ────────────────────────────────────────────── */

/* 1 when the handle is the recipe's engine and, at T > 1, the threaded arm the recipe names bound.
 * The in-place door sets the engine pointers only (c2c_ip_create_il.h:136-147), so the engine is
 * read from them in both placements. *serial = 1: the engine is right, its arm did not bind. */
static inline int est__check_plan(const struct vfft_plan_s *h, const est_recipe_t *rc, const est_request_t *rq,
                                  int *serial, char *why, size_t whylen)
{
    const int T = rq->nthreads > 1 ? rq->nthreads : 1;
    int want_mt = rc->il_mt, got_mt = 0, ok;
    *serial = 0;
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
    {
        est_refuse(why, whylen, "the door built another engine than the recipe's (%s)", vw2_oop_il_name[rc->k1_il_route]);
        return 0;
    }
    if (T > 1 && want_mt != got_mt)
    {
        *serial = 1;
        est_refuse(why, whylen, "the threaded arm il_mt=%d did not bind (bound %d)", want_mt, got_mt);
        return 0;
    }
    return 1;
}

/* ── the rows as written ────────────────────────────────────────────────── */

/* a row's payload as one text: its payload tokens (section 1, wisdom2.h:182) "name=value", sorted and
 * space-joined, so a re-bank that only reorders tokens reads the same; malloc'ed, NULL when out of
 * memory */
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
    {
        free(v);
        free(pool);
        free(out);
        return NULL;
    }
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
/* the store's rows as written: their keys and payload texts */
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
        {
            est__snap_free(s, st->nrec);
            return NULL;
        }
    }
    return s;
}
/* 1 = the own row r still carries the payload written at its key */
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

/* ── the replay and its detectors ───────────────────────────────────────── */

/* the detectors' readings: races counted, scopes entered, the library's store loaded, the
 * directory stores open */
typedef struct { long races, scopes; int def_loaded, nstores; } est__meter_t;
static inline void est__meter_read(est__meter_t *m)
{
    m->races = _vfft_create_race_count;
    m->scopes = vfft_measure_scopes();
    m->def_loaded = _def_loaded;
    m->nstores = _vfft_api_nstores;
}

/* The door's replay of the rows in S, run inside a create (_vfft_create_depth >= 1, the caller's):
 * S disowned and its rows kept as written, the library's own create run against it with
 * wisdom_write = 0, then the detectors against m0, read before the pre-flight. Returns the plan, or
 * NULL with rep->status and the reason. S stays the caller's to free (a plan keeps no pointer to
 * it, wisdom2_child.h:47). */
static inline struct vfft_plan_s *est__replay(struct vfft_wisdom_s *S, const vfft_cfg_t *cfg, const est__meter_t *m0,
                                              est_build_report_t *rep, char *why, size_t whylen)
{
    vfft_cfg_t c = *cfg;
    struct vfft_plan_s *h;
    est__snap_t *snap;
    est__meter_t m1;
    int i;
    vw2_disown(&S->vw2);
    rep->rows_before = S->vw2.nrec;
    if (!(snap = est__snap_take(&S->vw2)))
    {
        rep->status = EST_REFUSED;
        est_refuse(why, whylen, "out of memory");
        return NULL;
    }
    c.wisdom = (vfft_wisdom *)S;
    c.wisdom_write = 0;
    c.recalibrate = 0;
    _vfft_err_buf[0] = 0;
    h = (struct vfft_plan_s *)_vfft_create_cfg(&c);
    est__meter_read(&m1);
    rep->races = m1.races - m0->races;
    rep->scopes = m1.scopes - m0->scopes;
    rep->rows_after = S->vw2.nrec;
    for (i = 0; i < S->vw2.nrec; i++)
        if (S->vw2.rec[i].own)
        {   /* a row the door re-banked (vfft_child_store_raced's test, wisdom2_child.h:164-173) */
            const int same = est__snap_same(snap, rep->rows_before, &S->vw2.rec[i]);
            if (!rep->rewritten++ || (!same && !rep->changed))
                vw2__key_format(&S->vw2.rec[i].key, rep->rewritten_key, sizeof rep->rewritten_key);
            rep->changed += !same;
        }
    est__snap_free(snap, rep->rows_before);
    if (rep->races || rep->scopes || rep->rows_after != rep->rows_before)
    {
        rep->status = EST_MEASURED;
        est_refuse(why, whylen, "the create measured (races %ld, measuring scopes %ld, rows %d -> %d): the recipe did "
                   "not replay as written", rep->races, rep->scopes, rep->rows_before, rep->rows_after);
    }
    else if (m1.def_loaded != m0->def_loaded || m1.nstores != m0->nstores)
    {
        rep->status = EST_MEASURED;
        est_refuse(why, whylen, "the create opened a wisdom store: estimate reads none");
    }
    else if (rep->changed)
    {
        rep->status = EST_REWRITTEN;
        est_refuse(why, whylen, "the door re-wrote %d recipe row(s) with another payload, measuring nothing (first: %s): "
                   "the plan is not the recipe", rep->changed, rep->rewritten_key);
    }
    else if (!h)
    {
        rep->status = EST_NOPLAN;
        est_refuse(why, whylen, "the door built no plan, measuring nothing: %s",
                   _vfft_err_buf[0] ? _vfft_err_buf : "no reason given");
    }
    else
        return h;
    if (h)
        vfft_destroy((vfft_plan)h);
    return NULL;
}

/* ── the build ──────────────────────────────────────────────────────────── */

/* est_build_c2c: the recipe of rq -> a plan built with nothing measured, or NULL with the reason in
 * why and the outcome in rep (may be NULL). cfg is the request as the front door translated it
 * (vfft_cfg_t: 1D interleaved c2c, howmany 1); its store and save flag are not used. Holds the create
 * lock from the pre-flight to the end. The caller owns the plan (vfft_destroy); the caller's pin of a
 * threaded plan is vfft_create's to arm (vfft_api.h). */
static inline struct vfft_plan_s *est_build_c2c(const est_recipe_t *rc, const est_request_t *rq, const vfft_cfg_t *cfg,
                                                est_build_report_t *rep, char *why, size_t whylen)
{
    est_build_report_t lrep;
    struct vfft_wisdom_s *S = NULL;
    struct vfft_plan_s *h = NULL;
    est__meter_t m0;
    int serial = 0;
    if (!rep)
        rep = &lrep;
    memset(rep, 0, sizeof *rep);
    rep->status = EST_REFUSED;
    if (why && whylen)
        why[0] = 0;
    if (!rc || !rq || !cfg)
    {
        est_refuse(why, whylen, "no recipe, request or configuration");
        return NULL;
    }
    if (rq->transform != VFFT_C2C || cfg->transform != VFFT_C2C || cfg->dims > 1 || cfg->n[0] != rq->N ||
        cfg->howmany > 1 || cfg->layout != VFI_LAYOUT_IL)
    {
        est_refuse(why, whylen, "estimate builds one 1D interleaved c2c transform");
        return NULL;
    }
    /* the in-place scrambled prime cell at one thread cannot replay (est_plan_1d) */
    if (rc->k1_il_route == VFFT_K1_IL_PRIME && rq->placement == VFFT_PLACEMENT_IN && rq->order == VFFT_ORDER_SCRAMBLED &&
        rq->nthreads <= 1)
    {
        est_refuse(why, whylen, "the in-place scrambled prime cell at one thread cannot replay: its route row and its "
                   "prime method row share one key");
        return NULL;
    }
    vfft_env_init();          /* env.h:96: the pre-flight's tables in the create's FP mode */
    _vfft_create_lock();      /* vfft.c:2485, recursive: the pre-flight touches the create's process statics */
    _vfft_create_depth++;     /* the whole build inside a create: a clock read enters a scope */
    est__meter_read(&m0);
    if (est__preflight(rc, rq, why, whylen))
        goto out;
    if (!(S = vfft_child_store_new()))
    {
        est_refuse(why, whylen, "out of memory");
        goto out;
    }
    if (est_put_rows(&S->vw2, rc, rq, 1, why, whylen))
        goto out;
    h = est__replay(S, cfg, &m0, rep, why, whylen);
    if (h && !est__check_plan(h, rc, rq, &serial, why, whylen))
    {
        rep->status = serial ? EST_SERIAL : EST_NOPLAN;
        vfft_destroy((vfft_plan)h);
        h = NULL;
    }
    else if (h)
        rep->status = EST_BUILT;
out:
    vfft_child_store_free(S);
    if (--_vfft_create_depth == 0)
        _vfft_scope_create_done();   /* race_scope.h:621: leave a scope a clock read entered */
    _vfft_create_unlock();
    return h;
}

/* THE FRONT DOOR'S ESTIMATE CREATE (vfft_create, under VFFT_FLAG_ESTIMATE): this machine's facts,
 * the request, the recipe, the build. NULL with the reason on the error record (vfft_error). */
static vfft_plan _vfft_create_estimate(const vfft_cfg_t *cfg)
{
    est_machine_t m;
    est_request_t rq;
    est_recipe_t rc;
    char why[640];
    struct vfft_plan_s *h;
    memset(&m, 0, sizeof m);
    m.lanes = VFFT_IL_VW;
    m.l1d = vfft_cpu_l1d_bytes();
    m.l2 = vfft_cpu_l2_bytes();
    m.l3 = vfft_cpu_l3_bytes();
    m.cores = vfft_topology()->ncores;
    memset(&rq, 0, sizeof rq);
    rq.N = cfg->n[0];
    rq.transform = cfg->transform;
    rq.order = cfg->order == VFI_ORDER_SCRAMBLED ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    rq.placement = cfg->placement == VFFT_INPLACE ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    rq.nthreads = cfg->nthreads > 1 ? cfg->nthreads : 1;
    if (est_plan_1d(&m, &rq, &rc, why, sizeof why))
    {
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE: %s", why);
        return NULL;
    }
    h = est_build_c2c(&rc, &rq, cfg, NULL, why, sizeof why);
    if (!h)
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE: %s", why);
    return (vfft_plan)h;
}

#endif /* VFFT_IL_C2C_ESTIMATE_IL_H */
