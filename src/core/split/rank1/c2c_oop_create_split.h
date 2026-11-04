/* c2c_oop_create_split.h — the c2c OUT-OF-PLACE create, split tier.
 *
 * The split K=1 engine (the kind-3 row's split axis replayed, else the
 * structural default until calibrate_k1_split.c banks the cell), then the
 * classic OOP path: the owned-batch (padded) arm, the order-aware champion
 * lookup, calibrate-on-miss of both champions. Depends on common/ and split/
 * only. Split handles carry no threaded K=1 verdict, so the tier has no
 * post-step.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: calls file-scope statics of vfft.c, so
 * it is included (via the dispatcher) after those are defined.
 */
#ifndef VFFT_SPLIT_C2C_OOP_CREATE_SPLIT_H
#define VFFT_SPLIT_C2C_OOP_CREATE_SPLIT_H

static struct vfft_plan_s *_c2c_oop_create_k1_split(const vfft_cfg_t *cfg,
                                                    struct vfft_wisdom_s *W,
                                                    int N)
{
    int spr = VFFT_K1_SP_2PB;
    int sR1 = 0, sR2 = 0;
    /* the kind-3 cell's SPLIT axis (split/wisdom/wisdom2_oop_split.h).
     * The vw2_off_oop kill switch serves it from the one legacy line. */
    vfft_oop_sp_entry_t keb;
    const vfft_oop_wisdom_entry_t *kle =
        W->vw2_off_oop ? vfft_oop_wisdom_lookup_k1(&W->oop, N) : NULL;
    if (kle)
        vfft_oop_sp_from_legacy(&keb, kle);
    const vfft_oop_sp_entry_t *ke =
        W->vw2_off_oop ? (kle ? &keb : NULL)
                       : (vw2_oop_lookup_k1_sp_cell(&W->vw2, N, 0, 0, _vfft_plan_threads(cfg), &keb) ? &keb : NULL);
    const int sp_banked = (ke && ke->k1_sp_route >= 0);
    if (sp_banked)
    {
        spr = ke->k1_sp_route;
        sR1 = ke->R1;
        sR2 = ke->R2;
    }
    if (!sp_banked)
    {
        /* structural default (uncalibrated cell): mono when emitted,
         * else 2pb on the most balanced valid pair. The offline
         * calibrator (benches/calibrate_k1_split.c) banks the
         * measured lay=split row per cell. */
        if (vfft_k1_mono_fn(N) && N <= 64)
            spr = VFFT_K1_SP_MONO;
        for (int R2c = (N < 128 ? N : 128); R2c >= 4; R2c--)
        {
            if (N % R2c)
                continue;
            int R1c = N / R2c;
            if (R1c < 4 || R1c > 128 || (R1c % 4) || (R2c % 4))
                continue;
            if (!vfft_oop_leaf_fn(R2c) || !vfft_oop_t1_fn(R1c))
                continue;
            if (!sR1 || abs(R1c - R2c) < abs(sR1 - sR2))
            {
                sR1 = R1c;
                sR2 = R2c;
            }
        }
        if (!sR1 && (N % 64) == 0 && vfft_oop_t1_fn(64))
        {
            /* no classic pair (past the leaf/t1 reach, N >= 16384):
             * composed column is the ONLY K=1 route up there */
            int ccf_[VFFT_K1_CC_MAX_NF];
            if (vfft_k1_cc_default_chain(N / 64, ccf_))
            {
                spr = VFFT_K1_SP_CCOL;
                sR1 = 64;
                sR2 = N / 64;
            }
        }
    }
    vfft_oop_plan_t *psp = NULL;
    if (spr == VFFT_K1_SP_CCOL && sR1)
    {
        /* composed column: chain from the wisdom line, else the
         * per-R2 default. Create is self-validating (perm
         * discovery); failure falls through to the classic path. */
        int ccf[VFFT_K1_CC_MAX_NF];
        int ccn = (ke && ke->cc_chain)
                      ? vfft_k1_cc_chain_decode(ke->cc_chain, ccf)
                      : vfft_k1_cc_default_chain(N / sR1, ccf);
        /* column-plan VARIANTS from the kind-3 line's own cc_vars
         * token — the CCOL verdict is
         * SELF-CONTAINED in OOP wisdom (an OOP operation never
         * reads the in-place spike file at create). Decode must
         * match the chain's nf; absent/mismatch => NULL = the T1S
         * default. */
        const int *ccv = NULL;
        int ccv_[VFFT_K1_CC_MAX_NF];
        if (ccn && ke && ke->cc_vars &&
            vfft_k1_cc_vars_decode(ke->cc_vars, ccn, ccv_))
            ccv = ccv_;
        if (ccn)
            psp = vfft_oop_plan_create_k1_cc_v(N, sR1, ccf, ccn, ccv,
                                               _registry());
    }
    else if (spr != VFFT_K1_SP_MONO && sR1)
        psp = vfft_oop_plan_create_k1(N, sR1, sR2);
    /* availability degrade (wisdom may name routes this build lacks).
     * Runs BEFORE spr0 is captured: spr0 keys the JIT (and the TWL
     * table pick), and the JIT must never bake a route this build
     * cannot execute (e.g. 2PB 64x128: leaf_ugul stops at 64, so
     * execute degrades to 2PA and a 2PB bake could never compile).
     * The L3-missing cases degrade to their flat base here so spr0
     * never names an l3 twin this build lacks; the fold below then
     * only ever swaps pointers that exist. */
    if (spr == VFFT_K1_SP_MONO && !vfft_k1_mono_pair_fn(N, sR1))
        spr = VFFT_K1_SP_2PB;
    if (spr != VFFT_K1_SP_MONO)
    {
        if (!psp)
            spr = -1;
        else
        {
            if (spr == VFFT_K1_SP_3P_L3 && !psp->t1_l3)
                spr = VFFT_K1_SP_3P;
            if (spr == VFFT_K1_SP_2PA_L3 && !psp->t1_ul_l3)
                spr = VFFT_K1_SP_2PA;
            if (spr == VFFT_K1_SP_TWL && !psp->t1_ul_twl)
                spr = VFFT_K1_SP_2PA;
            if (spr == VFFT_K1_SP_2PB && !psp->leaf_ul)
                spr = VFFT_K1_SP_2PA;
            if (spr == VFFT_K1_SP_2PA && !psp->t1_ul)
                spr = VFFT_K1_SP_3P;
        }
    }
    int spr0 = spr; /* the EXECUTABLE wisdom route, pre-L3-fold
                     * (JIT sources + the Qlr-vs-Qr pick key on it) */
    /* log3 routes resolve to a create-time fn swap + the base route
     * (same Qr/Qi; the l3 twins are drop-in pointers — guaranteed
     * present here by the degrade above) */
    if (spr == VFFT_K1_SP_3P_L3)
    {
        psp->t1p = psp->t1_l3;
        spr = VFFT_K1_SP_3P;
    }
    if (spr == VFFT_K1_SP_2PA_L3)
    {
        psp->t1_ul = psp->t1_ul_l3;
        spr = VFFT_K1_SP_2PA;
    }
    if (spr >= 0)
    {
        struct vfft_plan_s *hk =
            (struct vfft_plan_s *)calloc(1, sizeof *hk);
        if (hk)
        {
            hk->transform = VFFT_C2C;
            hk->placement = VFFT_OUTOFPLACE;
            hk->layout = (int)cfg->layout;
            hk->N = N;
            hk->K = 1;
            hk->nthreads = _vfft_plan_threads(cfg);
            hk->k1_on = 1;
            hk->k1_sp_route = spr;
            hk->k1_il_route = VFFT_K1_IL_NONE;   /* a split handle has no IL route */
            hk->k1sp = psp;
            hk->k1_mono = vfft_k1_mono_pair_fn(N, sR1);
#ifdef VFFT_USE_JIT
            /* stride-baking JIT for the split route: compile
             * cost locked to create, cached on disk forever; NULL ->
             * the normal route fns below. TWL bakes against the
             * linear tables. */
            if (psp)
            {
                hk->k1_jit_qr = (spr0 == VFFT_K1_SP_TWL) ? psp->Qlr : psp->Qr;
                hk->k1_jit_qi = (spr0 == VFFT_K1_SP_TWL) ? psp->Qli : psp->Qi;
                if (hk->k1_jit_qr)
                    hk->k1_jit = vfft_k1_jit_resolve(N, sR1, sR2, spr0);
            }
#endif
            return hk;
        }
    }
    if (psp)
        vfft_oop_plan_destroy(psp);
    return NULL;
}

/* the incumbent arm of zturn-ts's K=1 out-of-place race: the kind-3 split
 * route as deployed (split_execute.h _exec_k1_split: the route fns or the
 * JIT), on the race's own planes */
typedef struct
{
    struct vfft_plan_s *h;
    double *ire, *iim, *ore, *oim;
} _zts_inc_oop_t;
static void _zts_inc_oop_run(void *v)
{
    _zts_inc_oop_t *c = (_zts_inc_oop_t *)v;
    _exec_k1_split(c->h, 1, c->ire, c->iim, c->ore, c->oim);
}

/* the K=1 out-of-place split cell with zturn-ts: the cell's eng=zts row
 * replayed; else the kind-3 route built as today and, on an UNBANKED cell
 * (or recalibrate), raced against zturn-ts's pool — the winner serves and
 * the cell banks (eng=zts, or eng=k1: held by the kind-3 route). A cell the
 * registry does not serve never races. */
static struct vfft_plan_s *_c2c_oop_create_k1_split_zts(const vfft_cfg_t *cfg,
                                                        struct vfft_wisdom_s *W,
                                                        int N)
{
    const int natural = cfg->order != VFI_ORDER_SCRAMBLED;
    struct vfft_plan_s *hk;
    int held = 0;
    if (natural && !cfg->recalibrate)
    {
        vfft_zts_plan_t *z = _zts_replay(W, N, 1, 0, 1, &held);
        if (z)
            return _zts_handle(cfg, VFFT_OUTOFPLACE, N, 1, z);
    }
    hk = _c2c_oop_create_k1_split(cfg, W, N);
    if (!natural || held || !vfft_zts_serves(N))
        return hk;
    {
        const size_t n = (size_t)N;
        double *ire = (double *)vfft_aligned_alloc(n * sizeof(double));
        double *iim = (double *)vfft_aligned_alloc(n * sizeof(double));
        double *ore = (double *)vfft_aligned_alloc(n * sizeof(double));
        double *oim = (double *)vfft_aligned_alloc(n * sizeof(double));
        if (ire && iim && ore && oim)
        {
            _zts_inc_oop_t ic = { hk, ire, iim, ore, oim };
            vfft_race_arm_t inc = { "k1", _zts_inc_oop_run, &ic };
            double *bufs[2] = { ire, iim };
            double wns = 0.0, ins = 0.0;
            vfft_zts_plan_t *z = vfft_zts_race(N, 1, 0, hk ? &inc : NULL, bufs, 2, &wns, &ins);
            if (z)
            {
                if (hk)
                    vfft_destroy(hk);
                hk = _zts_handle(cfg, VFFT_OUTOFPLACE, N, 1, z);
                if (hk)
                {
                    vw2_zts_bank(&W->vw2, N, 1, 0, z->chain, z->nf, z->tile, wns);
                    _vw2_persist(W, cfg);
                }
            }
            else if (hk)
            {
                vw2_zts_bank_held(&W->vw2, N, 1, 0, "k1", ins);
                _vw2_persist(W, cfg);
            }
        }
        vfft_aligned_free(ire);
        vfft_aligned_free(iim);
        vfft_aligned_free(ore);
        vfft_aligned_free(oim);
    }
    return hk;
}

static vfft_plan _vfft_create_c2c_oop_classic_split(const vfft_cfg_t *cfg,
                                                   vfft_batch ob,
                                                   struct vfft_wisdom_s *W,
                                                   const vfft_proto_registry_t *reg,
                                                   int N,
                                                   size_t K);

/* the incumbent arm of zturn-ts's lane-major out-of-place race: the classic
 * champion's handle run as deployed (the split execute) */
typedef struct
{
    struct vfft_plan_s *h;
    double *ire, *iim, *ore, *oim;
} _zts_inc_oopx_t;
static void _zts_inc_oopx_run(void *v)
{
    _zts_inc_oopx_t *c = (_zts_inc_oopx_t *)v;
    _vfft_split_execute(c->h, VFFT_FORWARD, c->ire, c->iim, c->ore, c->oim);
}

static vfft_plan _vfft_create_c2c_oop_split(const vfft_cfg_t *cfg,
                                           vfft_batch ob,
                                           struct vfft_wisdom_s *W,
                                           const vfft_proto_registry_t *reg,
                                           int N,
                                           size_t K)
{
    if (K > 1 && !ob && cfg->batch_geom == VFFT_BATCH_CONTIGUOUS)
    {
        /* a CONTIGUOUS split batch = K independent K=1 transforms, served
         * from the q=1 out-of-place cell when its engine has a contiguous
         * form (zturn-ts's _batch twin; the kind-3 routes have none). Plan
         * the K=1 cell through this same create, then run it K times. */
        vfft_cfg_t c1 = *cfg;
        struct vfft_plan_s *h1;
        c1.howmany = 1;
        c1.batch_geom = VFFT_BATCH_LANE_MAJOR; /* identical at K=1 */
        h1 = (struct vfft_plan_s *)_vfft_create_c2c_oop_split(&c1, NULL, W, reg, N, 1);
        if (!h1)
            return NULL;
        if (!h1->zts)
        {
            _vfft_warn("vfft_create: batch=VFFT_BATCH_CONTIGUOUS on a SPLIT out-of-place C2C "
                       "N=%d K=%zu: the N=%d K=1 split cell is served by its kind-3 route, "
                       "which has no contiguous batch form (zturn-ts, the only contiguous "
                       "split engine, %s) — ask VFFT_BATCH_LANE_MAJOR, or recalibrate the cell",
                       N, K, N, vfft_zts_serves(N) ? "lost the cell's race" : "has no cell at this N");
            vfft_destroy(h1);
            return NULL;
        }
        h1->K = K;
        _zts_tcmt_decide(h1, cfg, W, N, K); /* the threading verdict at T > 1 */
        return h1;
    }
    if (K == 1 && !ob)
    {
        struct vfft_plan_s *hk = _c2c_oop_create_k1_split_zts(cfg, W, N);
        if (hk)
            return hk;
        /* fall through to the classic OOP path */
    }
    if (K > 1 && K % (size_t)VFFT_ZTS_VW == 0 && K <= (size_t)VFFT_ZTS_LANES_MAX && !ob &&
        cfg->order != VFI_ORDER_SCRAMBLED && vfft_zts_serves(N))
    {
        /* the LANE-MAJOR batch out of place (K a multiple of the vector
         * width): zturn-ts's lane-major engine (ztv_*), the q=K out-of-place
         * cell. The incumbent is the classic out-of-place champion where one
         * exists (its natural kinds need K % 8 == 0: none at K = 4, so that
         * cell races zturn-ts's pool alone); a cell held by another family
         * falls through to the classic path. */
        int held = 0;
        vfft_zts_plan_t *z = cfg->recalibrate ? NULL : _zts_replay(W, N, K, 0, (int)K, &held);
        if (!z && !held)
        {
            struct vfft_plan_s *hc = (struct vfft_plan_s *)_vfft_create_c2c_oop_classic_split(cfg, ob, W, reg, N, K);
            const size_t n = (size_t)N * K;
            double *ire = (double *)vfft_aligned_alloc(n * sizeof(double));
            double *iim = (double *)vfft_aligned_alloc(n * sizeof(double));
            double *ore = (double *)vfft_aligned_alloc(n * sizeof(double));
            double *oim = (double *)vfft_aligned_alloc(n * sizeof(double));
            double wns = 0.0, ins = 0.0;
            if (ire && iim && ore && oim)
            {
                _zts_inc_oopx_t ic = { hc, ire, iim, ore, oim };
                vfft_race_arm_t inc = { "classic", _zts_inc_oopx_run, &ic };
                double *bufs[2] = { ire, iim };
                z = vfft_zts_race(N, (int)K, 0, hc ? &inc : NULL, bufs, 2, &wns, &ins);
                if (z)
                {
                    vw2_zts_bank(&W->vw2, N, K, 0, z->chain, z->nf, z->tile, wns);
                    _vw2_persist(W, cfg);
                }
                else if (hc)
                {
                    vw2_zts_bank_held(&W->vw2, N, K, 0, "classic", ins);
                    _vw2_persist(W, cfg);
                }
            }
            vfft_aligned_free(ire);
            vfft_aligned_free(iim);
            vfft_aligned_free(ore);
            vfft_aligned_free(oim);
            if (z)
            {
                if (hc)
                    vfft_destroy(hc);
            }
            else
                return hc; /* the classic champion holds, or nothing serves the cell */
        }
        if (z)
            return _zts_handle(cfg, VFFT_OUTOFPLACE, N, K, z);
    }
    return _vfft_create_c2c_oop_classic_split(cfg, ob, W, reg, N, K);
}

/* the CLASSIC out-of-place path: the owned-batch (padded) arm, the
 * order-aware champion lookup, calibrate-on-miss of both champions */
static vfft_plan _vfft_create_c2c_oop_classic_split(const vfft_cfg_t *cfg,
                                                   vfft_batch ob,
                                                   struct vfft_wisdom_s *W,
                                                   const vfft_proto_registry_t *reg,
                                                   int N,
                                                   size_t K)
{
    /* PADDED (opt-in): build at Kp so the OOP plan strides the caller's Kp-wide 4 planes
     * exactly. Pad-only (OOP bakes K, no runtime me). Kp = the handle's roundup(K,8), which
     * keeps all 3 kinds AND lets the (N,Kp) OOP wisdom cell cache (BAILEY2 + the wisdom
     * reader both hard-gate on K%8). Pad lanes [K,Kp) are zeroed junk, discarded. */
    size_t bK = K;
    int padded = 0;
    if (ob)
    {
        vfft_batch b = ob;
        if (b->xform != (int)VFFT_C2C || !b->oop || b->N != N || b->K != K)
        {
            _vfft_warn("vfft_create: config.batch does not match this out-of-place C2C "
                       "descriptor (batch: %s%s N=%d K=%zu; config: C2C out-of-place "
                       "N=%d K=%zu) — INTERNAL INVARIANT (the plan allocates its own buffers); please report",
                       _vfft_tname(b->xform), b->oop ? " out-of-place" : " in-place",
                       b->N, b->K, N, K);
            return NULL;
        }
        bK = b->Kp;
        padded = 1;
    }
    vfft_oop_plan_t *op = NULL;
    /* 1=NATURAL(LEAF/BAILEY2) 2=SCRAMBLED(MODEB); DEFAULT is natural (vfft.h; field trial V31,
     * owner 2026-10-13: the faster-of-both pick served a permuted spectrum under DEFAULT) */
    const int ord = cfg->order == VFI_ORDER_SCRAMBLED ? VFI_ORDER_SCRAMBLED : VFI_ORDER_NATURAL;
    /* Order-aware lookup: the cell can hold BOTH a natural and a MODEB champion as separate
     * (N,K,kind-class) entries, so the requested order is served straight from wisdom. */
    vfft_oop_sp_entry_t eb;
    const vfft_oop_sp_entry_t *e = NULL;
    if (W->vw2_off_oop)
    {   /* the kill switch: the legacy line's split (classic) fields */
        const vfft_oop_wisdom_entry_t *le = vfft_oop_wisdom_lookup_ord(&W->oop, N, bK, ord);
        if (le)
        {
            vfft_oop_sp_from_legacy(&eb, le);
            e = &eb;
        }
    }
    else if (vw2_oop_lookup_ord(&W->vw2, N, bK, ord, &eb))
        e = &eb;
    if (e && !cfg->recalibrate)
        op = vfft_oop_plan_from_entry(e, reg); /* the cached champion of the requested class */
    if (!op)
    {
        /* Calibrate-on-miss: build BOTH champions (native=LEAF/BAILEY2, MODEB), time each, and
         * persist BOTH as separate (N,K,kind-class) wisdom cells — so every config.order is cached
         * with no re-tune. Then return the requested order's champion (DEFAULT = the faster by ns).
         * Persisting both is exactly what makes MODEB and LEAF/BAILEY2 coexist per cell. */
        vfft_proto_dp_context_t ctx;
        vfft_proto_dp_init(&ctx, bK, N);
        if (cfg->rigor != VFFT_MEASURE)
            vfft_proto_dp_set_patient(&ctx);
        vfft_oop_plan_t *nat = NULL, *mb = NULL;
        double nns = 1e30, mns = 1e30;
        vfft_oop_plan_create_champions(N, bK, &ctx, reg, &nat, &nns, &mb, &mns);
        vfft_proto_dp_destroy(&ctx);
        /* Bank only servable cells: vfft_oop_plan_from_entry hard-gates
         * K%8, so a K%8!=0 champion row could never replay. It also
         * skips the K=1 MODEB champion, whose plan carries unraced
         * variant slots the wisdom2 codec would refuse. */
        if (bK > 0 && (bK % 8u) == 0)
        {
            if (nat)
            {
                vfft_oop_sp_entry_t ne;
                vfft_oop_sp_entry_from_plan(&ne, nat, N, bK, nns);
                vw2_oop_bank_classic(&W->vw2, &ne);
            }
            if (mb)
            {
                vfft_oop_sp_entry_t ne;
                vfft_oop_sp_entry_from_plan(&ne, mb, N, bK, mns);
                vw2_oop_bank_classic(&W->vw2, &ne);
            }
            if (nat || mb)
                _vw2_persist(W, cfg);
        }
        if (ord == VFI_ORDER_NATURAL)
        {
            op = nat;
            if (mb)
                vfft_oop_plan_destroy(mb);
        }
        else if (ord == VFI_ORDER_SCRAMBLED)
        {
            op = mb;
            if (nat)
                vfft_oop_plan_destroy(nat);
        }
        else if (nat && mb)
        {
            if (nns <= mns)
            {
                op = nat;
                vfft_oop_plan_destroy(mb);
            }
            else
            {
                op = mb;
                vfft_oop_plan_destroy(nat);
            }
        }
        else
            op = nat ? nat : mb;
    }
    if (!op)
    {
        if (ord == VFI_ORDER_NATURAL)
            _vfft_warn("vfft_create: no natural-order out-of-place C2C champion for "
                       "N=%d K=%zu (the natural kinds are gated on this cell) — ask "
                       "VFFT_ORDER_SCRAMBLED, the engine's own order, or recalibrate "
                       "the cell (VFFT_FLAG_RECALIBRATE) for a natural champion",
                       N, bK);
        else if (ob)
            _vfft_warn("vfft_create: no out-of-place C2C champion for the padded cell "
                       "N=%d Kp=%zu — drop VFFT_FLAG_OWN_PITCH or create the plan in place",
                       N, bK);
        else
            _vfft_warn("vfft_create: no out-of-place C2C engine covers N=%d K=%zu — "
                       "the OOP kinds need a radix factorization of N; prime and other "
                       "Rader/Bluestein-class sizes are served IN-PLACE only (create "
                       "with placement=VFFT_PLACEMENT_IN)",
                       N, bK);
        return NULL;
    }
    struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_oop_plan_destroy(op);
        return NULL;
    }
    h->transform = VFFT_C2C;
    h->placement = VFFT_OUTOFPLACE;
    h->layout = (int)cfg->layout;
    h->N = N;
    h->K = K;
    h->nthreads = _vfft_plan_threads(cfg);
    h->oplan = op;
    h->padded = padded;
    h->exec_me = (int)bK;
#ifdef VFFT_USE_JIT
    /* MODEB rides a staged inner plan -> JIT it (fwd: stages 1.. at start_stage=1;
     * bwd: whole in-place DIF at start_stage=0). LEAF/BAILEY2 have no staged plan. */
    if (op->kind == VFFT_OOP_KIND_MODEB && op->mb)
    {
        op->mb_jit_fwd = vfft_proto_plan_jit_fwd(op->mb);
        op->mb_jit_bwd = vfft_proto_plan_jit_bwd(op->mb);
    }
#endif
    return h;
}

#endif /* VFFT_SPLIT_C2C_OOP_CREATE_SPLIT_H */
