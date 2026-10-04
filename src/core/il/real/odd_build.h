/* odd_build.h — the real door's ODD-N engine pick (K=1, INTERLEAVED, both
 * placements): env pin, the banked engine, or THE ODD REAL RACE.
 *
 * Three engines serve an odd N:
 *   zrm  the real mono (zrm.h): one rn1 kernel, N <= 64 at the rn1 radices,
 *        both placements (the kind is alias-tolerant);
 *   zrf  the real flat DIT (zrf.h): a real leaf, the c2c flat DIT's stages on
 *        the digit blocks, a mono at the bottom; every N whose factors are
 *        the leaf's radices, both placements. Its chain, its split-body
 *        switch and its tile budget are plan input: the chain pick (the
 *        shared measured search, il/planning/chain_search.h) names the
 *        finalists, each is gated and burst-timed untiled with and without
 *        the split body, the two fastest again at every budget (a
 *        threaded plan: each budget serial and in both threaded arms,
 *        zrf_mt.h), and the two fastest plans of all join the race;
 *   zrb  the real Bluestein (zrb.h): the chirp-z convolution at M >= (3N-1)/2
 *        on a matched IL inner pair; every odd N past the mono with no chain,
 *        both placements. Its length and its inner are plan input: a handful
 *        of lengths at or above the bound, at each the prime route's own
 *        inner pool, every candidate gated and burst-timed, the two fastest
 *        plans of all join the race.
 * The race has no incumbent: the fastest gated arm serves and is banked in
 * the real shard (wisdom2_real_il.h). A row naming an engine this door no
 * longer builds (the retired odd-real bridge's `oddr`) is a miss: the race
 * runs and overwrites it.
 *
 * THE GATE'S REFERENCE is the independent forward DFT the c2c planner gates
 * against (_real_il_ref, il/real/zrp_build.h: the even door's too) -- no
 * plan, no wisdom. A reference that fails its own self-check builds nothing:
 * the door returns NULL and the front door refuses.
 *
 * Admission (_real_il_odd_admits) is the front door's too: an in-place odd
 * request is admitted exactly where this door has an engine.
 *
 * VFFT_ZRM=1 pins the mono, VFFT_ZRM=0 keeps it out; VFFT_ZRF=chain (e.g.
 * 9.9.5, then "/t" = the split body off, "/w256" = the tile budget) pins the
 * flat DIT, VFFT_ZRF=0 keeps it out; VFFT_ZRB=M/kind:shape[/tW] (e.g.
 * 1152/ztt:4.4.8.9) pins the Bluestein, VFFT_ZRB=0 keeps it out.
 * VFFT_ZRACE_VERBOSE=1 logs the race.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: after zrp_build.h (the engines' plan
 * builders, the race body, _zrpr_relerr), k1_commit.h (the prime route's
 * inner pool) and dp_planner_il.h (chain_search.h); before vfft_create (the
 * admission).
 */
#ifndef VFFT_IL_REAL_ODD_BUILD_H
#define VFFT_IL_REAL_ODD_BUILD_H

/* an odd K=1 cell with an IL real engine, in either placement */
static int _real_il_odd_admits(int N, int c2r)
{
    return (N & 1) && N >= 3 &&
           ((N <= VFFT_ZRM_MAX_N && vfft_zrm_fn(N, c2r) != 0) || _zrf_has_chain(N) || _zrb_ok(N));
}

typedef struct { struct vfft_plan_s *h; const double *in; double *out; } _odd_arm_t;
static void _odd_arm_run(void *v)
{
    _odd_arm_t *c = (_odd_arm_t *)v;
    if (c->h->zrm)
        _exec_zrm(c->h, c->in, c->out);
    else if (c->h->zrf)
        _exec_zrf(c->h, c->in, c->out);
    else if (c->h->zrb)
        _exec_zrb(c->h, c->in, c->out);
    else if (c->h->zrbl)
        _exec_zrbl(c->h, c->in, c->out);
}

/* the flat DIT's sweep. One candidate: built, gated against ref, burst-timed
 * (best of five); it enters the two-slot podium or is destroyed. Returns 1
 * when it built (whatever became of it). s0 = the arm's input (b itself in
 * place). */
static int _zrf_try(const vfft_config_t *cfg, int N, const int *R, int K, int nomsz, int tile, int mt,
                    const double *a, const double *ref, double *b, const double *s0, size_t xs, size_t nchk,
                    struct vfft_plan_s *out[2], double bns[2], int *any_msz, int need_tiled)
{
    struct vfft_plan_s *h = _zrf_build_plan(cfg, N, R, K, nomsz, tile);
    double t0, est, best = 1e300;
    int reps;
    if (!h)
        return 0;
    if (need_tiled && !vfft_zrf_tiled(h->zrf))
    {   /* no level takes this budget: the untiled plan, already timed */
        vfft_destroy((vfft_plan)h);
        return 1;
    }
    if (mt && !vfft_zrf_mt_bind(h->zrf, h->nthreads, mt))
    {   /* the threaded arm declines this plan */
        vfft_destroy((vfft_plan)h);
        return 1;
    }
    if (any_msz)
        for (int j = 0; j < h->zrf->J; j++)
            for (int st = 1; st <= h->zrf->lv[j].ns; st++)
                *any_msz |= h->zrf->lv[j].fd->msz[st];
    memcpy(b, a, xs * sizeof(double));
    _exec_zrf(h, s0, b);
    {
        const double e = _zrpr_relerr(b, ref, nchk);
        if (e >= 1e-10)
        {
            char cs[48];
            vfft_zrf_chain_str(R, K, cs, sizeof cs);
            fprintf(stderr, "[zrf] N=%d chain %s%s/w%d%s FAILS the gate (rel %.2e vs the reference) -- dropped\n",
                    N, cs, nomsz ? "/t" : "", tile, mt ? "/m" : "", e);
            vfft_destroy((vfft_plan)h);
            return 1;
        }
    }
    _exec_zrf(h, s0, b); /* warm (a threaded form: its workers too) */
    t0 = vfft_now_ns();
    _exec_zrf(h, s0, b);
    est = vfft_now_ns() - t0;
    reps = (int)(1.0e5 / (est > 1.0 ? est : 1.0));
    if (reps < 2) reps = 2;
    if (reps > 1024) reps = 1024;
    for (int r = 0; r < 5; r++)
    {
        double t = vfft_now_ns();
        for (int i = 0; i < reps; i++) _exec_zrf(h, s0, b);
        t = (vfft_now_ns() - t) / reps;
        if (t < best) best = t;
    }
    if (best < bns[0])
    {
        if (out[1]) vfft_destroy((vfft_plan)out[1]);
        out[1] = out[0]; bns[1] = bns[0];
        out[0] = h; bns[0] = best;
    }
    else if (best < bns[1])
    {
        if (out[1]) vfft_destroy((vfft_plan)out[1]);
        out[1] = h; bns[1] = best;
    }
    else
        vfft_destroy((vfft_plan)h);
    return 1;
}

/* ── THE CHAIN PICK: the shared measured search (il/planning/chain_search.h:
 * every radix set in two orders -- its lowest radix as the first stage, its
 * second-lowest as the first stage -- the best three sets' orderings, the
 * finalists) with zrf's HEAT: every chain built serial and untiled, gated
 * against the reference, raced in one same-run race on the door's planes.
 * A heat's plans together stay under 512 MB. */
#if VFFT_CHAIN_MAX_K != VFFT_ILFD_MAX_K
#error "chain_search.h's VFFT_CHAIN_MAX_K must be the flat DIT's VFFT_ILFD_MAX_K"
#endif
typedef struct
{
    const vfft_config_t *cfg;
    int N;
    const double *a, *ref, *s0;
    double *b;
    size_t xs, nchk;
} _zrf_hctx_t;
typedef struct { struct vfft_plan_s *h; const double *s0; double *b; } _zrf_arm_t;
static void _zrf_arm_run(void *v)
{
    _zrf_arm_t *a = (_zrf_arm_t *)v;
    _exec_zrf(a->h, a->s0, a->b);
}
static void _zrf_arm_reset(void *v)
{   /* the input before every sample: an in-place arm walks its own output */
    const _zrf_hctx_t *c = (const _zrf_hctx_t *)v;
    memcpy(c->b, c->a, c->xs * sizeof(double));
}
static void _zrf_heat(void *hv, vfft_chain_t *f, const int *idx, int n, int fix, double *ns)
{
    _zrf_hctx_t *c = (_zrf_hctx_t *)hv;
    static _zrf_arm_t arm[VFFT_CHAIN_HEAT];   /* off the stack: creates are single-threaded */
    vfft_race_arm_t ra[VFFT_CHAIN_HEAT];
    double rns[VFFT_CHAIN_HEAT], t0, est;
    int map[VFFT_CHAIN_HEAT], na = 0, k, reps;
    const int ip = (c->s0 == c->b);
    for (k = 0; k < n; k++)
    {
        vfft_chain_t *ch = &f[idx[k]];
        int tries = 0;
        struct vfft_plan_s *h = NULL;
        ns[k] = 1e18;
        for (;;)
        {
            h = _zrf_build_plan(c->cfg, c->N, ch->R, ch->n, 0, 0);
            if (h)
            {
                double e;
                memcpy(c->b, c->a, c->xs * sizeof(double));
                _exec_zrf(h, c->s0, c->b);
                e = _zrpr_relerr(c->b, c->ref, c->nchk);
                if (e < 1e-10) break;
                {
                    char cs[64];
                    vfft_chain_str(ch, cs, sizeof cs);
                    fprintf(stderr, "[zrf] N=%d chain %s FAILS the gate (rel %.2e vs the reference) -- dropped\n",
                            c->N, cs, e);
                }
                vfft_destroy((vfft_plan)h);
                h = NULL;
            }
            if (!fix || ++tries >= 16 || !vfft_chain_next(ch)) break;
        }
        if (!h) continue;
        arm[na].h = h; arm[na].s0 = c->s0; arm[na].b = c->b;
        map[na] = k;
        ra[na].name = "zrf"; ra[na].run = _zrf_arm_run; ra[na].ctx = &arm[na];
        na++;
    }
    if (na == 0) return;
    _zrf_arm_reset(c);
    _zrf_arm_run(&arm[0]);
    _zrf_arm_reset(c);
    t0 = vfft_now_ns();
    _zrf_arm_run(&arm[0]);
    est = vfft_now_ns() - t0;
    reps = (int)(2.0e5 / (est > 1.0 ? est : 1.0));
    if (reps < 1) reps = 1;
    if (reps > (ip ? 32 : 1024)) reps = ip ? 32 : 1024;   /* in place: the data grows per pass */
    {
        const vfft_race_proto_t proto = { 3, reps, VFFT_RACE_MIN, 1, 1, _zrf_arm_reset, c, 1 }; /* single-thread arms: paced (VFFT_RACE_PACE_MS) */
        vfft_race_run(&proto, ra, na, rns);
    }
    for (k = 0; k < na; k++)
    {
        ns[map[k]] = rns[k];
        vfft_destroy((vfft_plan)arm[k].h);
    }
}
/* the chain pick's finalists x the split-body switch untiled; then the two
 * fastest at each tile budget (and, threaded, its two threaded arms). The two
 * fastest plans of all are returned as finished handles. */
static int _zrf_chain_sweep(const vfft_config_t *cfg, int N, const double *a, const double *ref, double *b,
                            const double *s0, size_t xs, size_t nchk, struct vfft_plan_s *out[2])
{
    static const int budgets[] = { 64, 128, 256, 512, 1024, 2048 };
    _zrf_hctx_t hc;
    vfft_chain_engine_t e;
    vfft_chain_t fin[VFFT_CHAIN_FIN_MAX];
    vfft_chain_stats_t st;
    double bns[2] = { 1e300, 1e300 };
    int seedR[2][VFFT_ILFD_MAX_K], seedK[2] = { 0, 0 }, seedm[2] = { 0, 0 }, nf;
    out[0] = out[1] = NULL;
    hc.cfg = cfg; hc.N = N; hc.a = a; hc.ref = ref; hc.s0 = s0; hc.b = b; hc.xs = xs; hc.nchk = nchk;
    e.pool = VFFT_ZRF_POOL;
    e.npool = VFFT_ZRF_NPOOL;
    e.lead2 = 0;
    e.heat = _zrf_heat;
    e.hctx = &hc;
    e.budget = (size_t)512 << 20;
    e.per_stage_point = 2u * 2u * sizeof(double);
    e.tag = "[zrf]";
    nf = vfft_chain_search(&e, N, fin, &st);
    if (getenv("VFFT_ZRACE_VERBOSE"))
    {
        fprintf(stderr, "[zrf] N=%d chain pick: %d radix set(s) (%d orders), %ld chain(s), heats of %d, %d heat(s) ->",
                N, st.nsets, st.nord, st.chains, st.cap, st.heats);
        for (int i = 0; i < nf; i++)
        {
            char cs[64];
            vfft_chain_str(&fin[i], cs, sizeof cs);
            fprintf(stderr, " %s", cs);
        }
        fprintf(stderr, "\n");
    }
    for (int c = 0; c < nf; c++)
    {
        int any = 0;
        if (!_zrf_try(cfg, N, fin[c].R, fin[c].n, 0, 0, 0, a, ref, b, s0, xs, nchk, out, bns, &any, 0))
            continue;
        if (any) /* a stage takes the split body: its twin without it is another plan */
            _zrf_try(cfg, N, fin[c].R, fin[c].n, 1, 0, 0, a, ref, b, s0, xs, nchk, out, bns, NULL, 0);
    }
    for (int i = 0; i < 2; i++)
        if (out[i])
        {
            seedK[i] = out[i]->zrf->K; seedm[i] = out[i]->zrf->nomsz;
            memcpy(seedR[i], out[i]->zrf->R, sizeof(int) * (size_t)seedK[i]);
        }
    for (int i = 0; i < 2; i++)
        for (int t = 0; seedK[i] && t < (int)(sizeof budgets / sizeof budgets[0]); t++)
        {
            _zrf_try(cfg, N, seedR[i], seedK[i], seedm[i], budgets[t], 0, a, ref, b, s0, xs, nchk, out, bns, NULL, 1);
            if (_vfft_plan_threads(cfg) > 1)
            {   /* the two threaded arms: FIRST and LEVELS (il/real/zrf_mt.h) */
                _zrf_try(cfg, N, seedR[i], seedK[i], seedm[i], budgets[t], 1, a, ref, b, s0, xs, nchk, out, bns, NULL, 1);
                _zrf_try(cfg, N, seedR[i], seedK[i], seedm[i], budgets[t], 2, a, ref, b, s0, xs, nchk, out, bns, NULL, 1);
            }
        }
    return (out[0] != NULL) + (out[1] != NULL);
}

/* the real Bluestein's handle: the length and the inner descriptor are plan
 * input (the descriptor is the prime route's, il/rank1/k1_commit.h). A
 * descriptor that does not build is a miss, never the structural rule's plan. */
static struct vfft_plan_s *_zrb_build_plan(const vfft_config_t *cfg, int N, int M, _ilprime_inner_desc_t *d)
{
    char kind[8], shape[64];
    vfft_zrb_plan_t *zp;
    struct vfft_plan_s *h;
    d->failed = 0;
    zp = vfft_zrb_create(N, M, _ilprime_inner_from_desc, d);
    if (zp && d->failed) { vfft_zrb_destroy(zp); zp = NULL; }
    if (!zp)
        return NULL;
    _ilprime_desc_str(d, kind, sizeof kind, shape, sizeof shape);
    vfft_zrb_name_inner(zp, kind, shape, d->tw);
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_zrb_destroy(zp);
        return NULL;
    }
    h->transform = cfg->transform;
    h->placement = cfg->placement;
    h->layout = (int)VFFT_LAYOUT_INTERLEAVED;
    h->N = N;
    h->K = 1;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zrb = zp;
    return h;
}

/* VFFT_ZRB at create: 1 = pinned at "M/kind:shape[/tW]" (e.g. 1152/ztt:4.4.8.9,
 * 2048/2p:32.64), 0 = kept out of the race, -1 = unset */
static int _zrb_env(int *M, _ilprime_inner_desc_t *d)
{
    const char *e = getenv("VFFT_ZRB");
    char kind[8], shape[64], *end;
    const char *s;
    size_t n;
    int tw = 0;
    *M = 0;
    if (!e || !e[0])
        return -1;
    if (e[0] == '0' && !e[1])
        return 0;
    *M = (int)strtol(e, &end, 10);
    if (end == e || *end != '/' || *M < 4)
        return 0;
    s = end + 1;
    n = strcspn(s, ":");
    if (!s[n] || n == 0 || n >= sizeof kind)
        return 0;
    memcpy(kind, s, n); kind[n] = 0;
    s += n + 1;
    n = strcspn(s, "/");
    if (n == 0 || n >= sizeof shape)
        return 0;
    memcpy(shape, s, n); shape[n] = 0;
    s += n;
    if (s[0] == '/' && s[1] == 't') { tw = atoi(s + 2); s += 2; while (*s >= '0' && *s <= '9') s++; }
    if (*s || tw < 0)
        return 0;
    return _ilprime_desc_parse(d, kind, shape, tw) ? 1 : 0;
}

/* THE REAL BLUESTEIN'S SWEEP. The lengths: the smallest power of two at or
 * above the bound, the smooth lengths 2^a*{3,5,7,9,15} below it (the three
 * smallest) and, within the pair engines' reach, the two smallest lengths
 * any inner builds; at each length the prime route's own inner pool
 * (_ilprime_inner_cands). Every candidate is built, gated against ref and
 * burst-timed (best of three); the two fastest plans of all are returned as
 * finished handles. */
#define VFFT_ZRB_MAX_M 6
static int _zrb_m_cands(int N, int *out)
{
    static const int q[5] = { 3, 5, 7, 9, 15 };
    static _ilprime_inner_desc_t pool[_ILPR_MAX_CANDS];   /* off the stack */
    const int mn = vfft_zrb_min_m(N);
    int sm[64], nsm = 0, n = 0, p2 = 16, i, a, k;
    while (p2 < mn) p2 <<= 1;
    for (i = 0; i < 5; i++)
        for (a = 2; (1 << a) * q[i] < p2 && nsm < 64; a++)
            if ((1 << a) * q[i] >= mn) sm[nsm++] = (1 << a) * q[i];
    for (i = 1; i < nsm; i++)   /* ascending */
        for (k = i; k > 0 && sm[k] < sm[k - 1]; k--) { const int t = sm[k]; sm[k] = sm[k - 1]; sm[k - 1] = t; }
    for (i = 0, k = 0; i < nsm && k < 3; i++)
        if (_ilprime_inner_cands(sm[i], pool, _ILPR_MAX_CANDS) > 0) { out[n++] = sm[i]; k++; }
    if (mn <= 4096)
    {   /* the pair engines' lengths: the two smallest that build, not already listed */
        int M;
        for (M = mn, k = 0; M < p2 && M <= 4096 && k < 2; M++)
        {
            int dup = 0;
            for (i = 0; i < n; i++) if (out[i] == M) dup = 1;
            if (dup) continue;
            if (_ilprime_inner_cands(M, pool, _ILPR_MAX_CANDS) > 0) { out[n++] = M; k++; }
        }
    }
    out[n++] = p2;
    return n;
}
static int _zrb_try(const vfft_config_t *cfg, int N, int M, _ilprime_inner_desc_t *d, int mt,
                    const double *a, const double *ref, double *b, const double *s0, size_t xs, size_t nchk,
                    struct vfft_plan_s *out[2], double bns[2])
{
    struct vfft_plan_s *h = _zrb_build_plan(cfg, N, M, d);
    double t0, est, best = 1e300;
    int reps;
    if (!h)
        return 0;
    if (mt && !vfft_zrb_mt_bind(h->zrb, h->nthreads, mt))
    {   /* the threaded arm declines this plan (no ZTURN-T inner, or no tiles) */
        vfft_destroy((vfft_plan)h);
        return 1;
    }
    memcpy(b, a, xs * sizeof(double));
    _exec_zrb(h, s0, b);
    {
        const double e = _zrpr_relerr(b, ref, nchk);
        if (e >= 1e-10)
        {
            char cs[96];
            vfft_zrb_str(h->zrb, cs, sizeof cs);
            fprintf(stderr, "[zrb] N=%d %s FAILS the gate (rel %.2e vs the reference) -- dropped\n", N, cs, e);
            vfft_destroy((vfft_plan)h);
            return 1;
        }
    }
    _exec_zrb(h, s0, b);
    t0 = vfft_now_ns();
    _exec_zrb(h, s0, b);
    est = vfft_now_ns() - t0;
    reps = (int)(5.0e4 / (est > 1.0 ? est : 1.0));
    if (reps < 2) reps = 2;
    if (reps > 512) reps = 512;
    for (int r = 0; r < 3; r++)
    {
        double t = vfft_now_ns();
        for (int i = 0; i < reps; i++) _exec_zrb(h, s0, b);
        t = (vfft_now_ns() - t) / reps;
        if (t < best) best = t;
    }
    if (best < bns[0])
    {
        if (out[1]) vfft_destroy((vfft_plan)out[1]);
        out[1] = out[0]; bns[1] = bns[0];
        out[0] = h; bns[0] = best;
    }
    else if (best < bns[1])
    {
        if (out[1]) vfft_destroy((vfft_plan)out[1]);
        out[1] = h; bns[1] = best;
    }
    else
        vfft_destroy((vfft_plan)h);
    return 1;
}
static int _zrb_sweep(const vfft_config_t *cfg, int N, const double *a, const double *ref, double *b,
                      const double *s0, size_t xs, size_t nchk, struct vfft_plan_s *out[2])
{
    static _ilprime_inner_desc_t pool[_ILPR_MAX_CANDS];   /* off the stack */
    int Ms[VFFT_ZRB_MAX_M + 2];
    const int nm = _zrb_m_cands(N, Ms);
    double bns[2] = { 1e300, 1e300 };
    int built = 0;
    out[0] = out[1] = NULL;
    for (int m = 0; m < nm; m++)
    {
        const int nc = _ilprime_inner_cands(Ms[m], pool, _ILPR_MAX_CANDS);
        int nb = 0;
        for (int c = 0; c < nc; c++)
            nb += _zrb_try(cfg, N, Ms[m], &pool[c], 0, a, ref, b, s0, xs, nchk, out, bns);
        built += nb;
        if (getenv("VFFT_ZRACE_VERBOSE"))
            fprintf(stderr, "[zrb] N=%d M=%d: %d of %d inner(s) built; best so far %.0f ns\n", N, Ms[m], nb, nc, bns[0]);
    }
    (void)built;
    if (_vfft_plan_threads(cfg) > 1)
    {   /* the threaded forms of the two fastest (zrb_mt.h): BLOCKS and TILES */
        _ilprime_inner_desc_t sd[2];
        int sM[2], ns2 = 0;
        for (int i = 0; i < 2; i++)
            if (out[i] && _ilprime_desc_parse(&sd[ns2], out[i]->zrb->ikind, out[i]->zrb->ishape, out[i]->zrb->itw))
                sM[ns2++] = out[i]->zrb->M;
        for (int i = 0; i < ns2; i++)
            for (int mt = 1; mt <= 2; mt++)
                _zrb_try(cfg, N, sM[i], &sd[i], mt, a, ref, b, s0, xs, nchk, out, bns);
    }
    return (out[0] != NULL) + (out[1] != NULL);
}

/* THE ODD REAL RACE (header). Returns the serving handle bare (the caller
 * finishes it), or NULL when no gated arm was built. The verdict is banked
 * unless the store is off. */
static struct vfft_plan_s *_real_il_odd_race(const vfft_config_t *cfg, int N, struct vfft_wisdom_s *W)
{
    const int c2r = cfg->transform == VFFT_C2R, ip = cfg->placement == VFFT_INPLACE;
    const int Tk = _vfft_plan_threads(cfg);   /* the verdict's thread key */
    const int mono_ok = N <= VFFT_ZRM_MAX_N && _zrm_env() != 0 && vfft_zrm_fn(N, c2r) != 0;
    int fR[VFFT_ILFD_MAX_K], fK = 0, fnomsz = 0, ftile = 0, fmt = 0;
    const int fenv = _zrf_env(fR, &fK, &fnomsz, &ftile, &fmt);
    _ilprime_inner_desc_t bd;
    int bM = 0;
    const int benv = _zrb_env(&bM, &bd);
    const int blue_ok = _zrb_ok(N) && benv != 0;
    const size_t xs = (size_t)N + 3;
    const size_t nchk = c2r ? (size_t)N : (size_t)N + 1; /* odd N: (N+1)/2 bins, N reals */
    double *a = (double *)vfft_aligned_alloc(xs * sizeof(double));
    double *b = (double *)vfft_aligned_alloc(xs * sizeof(double));
    double *ref = (double *)vfft_aligned_alloc(xs * sizeof(double));
    const double *s0 = ip ? b : a;
    struct vfft_plan_s *arm[6], *hz[2] = { NULL, NULL };
    double ns[6], t0, est;
    int narm = 0, reps, win = 0;
    if (!a || !b || !ref)
    {
        vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
        return NULL;
    }
    {
        unsigned sd = 0x9e3779b9u ^ (unsigned)N ^ (unsigned)(c2r << 8);
        for (size_t i = 0; i < xs; i++)
        {
            sd = sd * 1664525u + 1013904223u;
            a[i] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
        }
    }
    if (c2r)
        a[1] = 0.0; /* a CCE spectrum: real DC (odd N has no Nyquist bin) */
    memset(ref, 0, xs * sizeof(double));
    memset(b, 0, xs * sizeof(double));
    if (_real_il_ref(c2r, N, a, ref) != 0)
    {
        _vfft_warn("vfft_create: %s odd N=%d: the gate's reference failed its self-check; no engine is raced",
                   _vfft_tname(cfg->transform), N);
        vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
        return NULL;
    }
    if (mono_ok)
    {
        struct vfft_plan_s *hm = _zrm_build_plan(cfg, N);
        if (hm)
        {
            memcpy(b, a, xs * sizeof(double));
            _exec_zrm(hm, s0, b);
            const double e = _zrpr_relerr(b, ref, nchk);
            if (e >= 1e-10)
            {
                fprintf(stderr, "[zrm] N=%d %s %s the real mono FAILS the gate (rel %.2e vs the reference) -- dropped\n",
                        N, c2r ? "c2r" : "r2c", ip ? "ip" : "oop", e);
                vfft_destroy((vfft_plan)hm);
            }
            else
                arm[narm++] = hm;
        }
    }
    if (fenv != 0 && _zrf_chain_sweep(cfg, N, a, ref, b, s0, xs, nchk, hz) > 0)
    {
        arm[narm++] = hz[0];
        if (hz[1]) arm[narm++] = hz[1];
    }
    if (blue_ok)
    {
        struct vfft_plan_s *hb[2] = { NULL, NULL };
        if (_zrb_sweep(cfg, N, a, ref, b, s0, xs, nchk, hb) > 0)
        {
            arm[narm++] = hb[0];
            if (hb[1]) arm[narm++] = hb[1];
        }
    }
    if (narm == 0)
    {
        vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
        return NULL;
    }
    _vfft_create_race_count++;
    {
        _odd_arm_t ca[6];
        vfft_race_arm_t arms[6];
        static const char *nm[2] = { "zrf", "zrf2" }, *nb[2] = { "zrb", "zrb2" };
        int nz = 0, nbl = 0;
        for (int i = 0; i < narm; i++)
        {
            ca[i].h = arm[i]; ca[i].in = s0; ca[i].out = b;
            arms[i].name = arm[i]->zrm ? "zrm" : arm[i]->zrb ? nb[nbl++ & 1] : nm[nz++ & 1];
            arms[i].run = _odd_arm_run; arms[i].ctx = &ca[i];
        }
        memcpy(b, a, xs * sizeof(double));
        t0 = vfft_now_ns();
        _odd_arm_run(&ca[0]);
        est = vfft_now_ns() - t0;
        reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
        if (reps < 2) reps = 2;
        if (reps > 4096) reps = 4096;
        {
            /* a threaded plan's arms are never paused: two warm passes, no pacing */
            const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, Tk > 1 ? 2 : 1, NULL, NULL, Tk > 1 ? 0 : 1 };
            vfft_race_run(&proto, arms, narm, ns);
        }
    }
    for (int i = 1; i < narm; i++)
        if (ns[i] < ns[win]) win = i;
    if (getenv("VFFT_ZRACE_VERBOSE"))
    {
        fprintf(stderr, "[odd] N=%d %s %s race: reps=%d |", N, c2r ? "c2r" : "r2c", ip ? "ip" : "oop", reps);
        for (int i = 0; i < narm; i++)
        {
            char cs[96] = "", ws[16] = "";
            if (arm[i]->zrf)
            {
                vfft_zrf_chain_str(arm[i]->zrf->R, arm[i]->zrf->K, cs, sizeof cs);
                snprintf(ws, sizeof ws, "/w%d%s", arm[i]->zrf->tile, arm[i]->zrf->mt == 2 ? "/m2" : arm[i]->zrf->mt ? "/m1" : "");
            }
            else if (arm[i]->zrb)
            {
                vfft_zrb_str(arm[i]->zrb, cs, sizeof cs);
                if (arm[i]->zrb->mt) snprintf(ws, sizeof ws, "/m%d", arm[i]->zrb->mt);
            }
            fprintf(stderr, " %s%s%s%s=%.0f%s", arm[i]->zrm ? "zrm" : arm[i]->zrb ? "zrb:" : "zrf:", cs,
                    (arm[i]->zrf && arm[i]->zrf->nomsz) ? "/t" : "", ws, ns[i], i == win ? "*" : "");
        }
        fprintf(stderr, "\n");
    }
    vfft_aligned_free(a); vfft_aligned_free(b); vfft_aligned_free(ref);
    {
        struct vfft_plan_s *hw = arm[win];
        if (W && !W->vw2_off_oop)
        {
            const int rc = hw->zrf ? vw2_real_il_bank_zrf(&W->vw2, N, c2r, ip, Tk, hw->zrf->R, hw->zrf->K, hw->zrf->nomsz,
                                                          hw->zrf->tile, hw->zrf->mt, ns[win])
                         : hw->zrb ? vw2_real_il_bank_zrb(&W->vw2, N, c2r, ip, Tk, hw->zrb->M, hw->zrb->ikind, hw->zrb->ishape,
                                                          hw->zrb->itw, hw->zrb->mt, ns[win])
                                   : vw2_real_il_bank_zrm(&W->vw2, N, c2r, ip, Tk, ns[win]);
            if (rc == VW2_OK)
                _vw2_persist(W, cfg);
            else
                fprintf(stderr, "vfft: real engine verdict NOT banked at odd N=%d (rc=%d) -- the cell will re-race\n", N, rc);
        }
        for (int i = 0; i < narm; i++)
            if (i != win) vfft_destroy((vfft_plan)arm[i]);
        return hw;
    }
}

/* The door's odd engine pick: env pin, the banked engine, or the race.
 * Returns the handle bare; NULL when nothing could be built. */
static struct vfft_plan_s *_real_il_odd_build(const vfft_config_t *cfg, int N, struct vfft_wisdom_s *W)
{
    const int c2r = cfg->transform == VFFT_C2R, ip = cfg->placement == VFFT_INPLACE;
    const int Tk = _vfft_plan_threads(cfg);   /* the verdict's thread key */
    const int menv = _zrm_env();
    const int mono_ok = N <= VFFT_ZRM_MAX_N && menv != 0 && vfft_zrm_fn(N, c2r) != 0;
    int fR[VFFT_ILFD_MAX_K], fK = 0, fnomsz = 0, ftile = 0, fmt = 0;
    const int fenv = _zrf_env(fR, &fK, &fnomsz, &ftile, &fmt);
    _ilprime_inner_desc_t bd;
    int bM = 0;
    const int benv = _zrb_env(&bM, &bd);
    if (menv == 1 && mono_ok)
    {
        struct vfft_plan_s *hm = _zrm_build_plan(cfg, N);
        if (hm)
            return hm;
    }
    if (fenv == 1)
    {
        struct vfft_plan_s *hf = _zrf_build_plan(cfg, N, fR, fK, fnomsz, ftile);
        if (hf)
        {
            if (fmt) vfft_zrf_mt_bind(hf->zrf, hf->nthreads, fmt);
            return hf;
        }
    }
    if (benv == 1)
    {
        struct vfft_plan_s *hb = _zrb_build_plan(cfg, N, bM, &bd);
        if (hb)
            return hb;
    }
    if (W && !W->vw2_off_oop && !cfg->recalibrate)
    {
        int R1, R2, form;
        const char *eng = vw2_real_il_lookup(&W->vw2, N, c2r, ip, Tk, &R1, &R2, &form);
        if (eng && !strcmp(eng, "zrm") && mono_ok)
        {
            struct vfft_plan_s *hm = _zrm_build_plan(cfg, N);
            if (hm)
                return hm;
        }
        else if (eng && !strcmp(eng, "zrf") && fenv != 0)
        {
            int ch[VFFT_ILFD_MAX_K], ck = 0, cn = 0, ct = 0, cm = 0;
            if (vw2_real_il_lookup_zrf(&W->vw2, N, c2r, ip, Tk, ch, VFFT_ILFD_MAX_K, &ck, &cn, &ct, &cm))
            {
                struct vfft_plan_s *hf = _zrf_build_plan(cfg, N, ch, ck, cn, ct);
                if (hf)
                {
                    if (cm) vfft_zrf_mt_bind(hf->zrf, hf->nthreads, cm); /* the banked threaded arm, at the row's T */
                    return hf;
                }
            }
        }
        else if (eng && !strcmp(eng, "zrb") && benv != 0)
        {
            char kind[8], shape[64];
            int M = 0, tw = 0, bmt = 0;
            if (vw2_real_il_lookup_zrb(&W->vw2, N, c2r, ip, Tk, &M, kind, sizeof kind, shape, sizeof shape, &tw, &bmt) &&
                _ilprime_desc_parse(&bd, kind, shape, tw))
            {
                struct vfft_plan_s *hb = _zrb_build_plan(cfg, N, M, &bd);
                if (hb)
                {
                    if (bmt) vfft_zrb_mt_bind(hb->zrb, hb->nthreads, bmt); /* the banked threaded arm, at the row's T */
                    return hb;
                }
            }
        }
        /* any other row (an engine this door does not build, a recipe that no
         * longer builds): the race, which overwrites it */
    }
    return _real_il_odd_race(cfg, N, W);
}

#endif /* VFFT_IL_REAL_ODD_BUILD_H */
