/* c2c_ip_create_split.h — the c2c IN-PLACE create, split tier.
 *
 * The split in-place arms: the CALLER-SUPPLIED BATCH arm (an owned-batch
 * descriptor was handed in, so the plan serves that exact handle), then the
 * general arms that allocate their own buffers (prime: Rader/Bluestein;
 * composite: the stride engine, natural rebuild). Depends on common/ and
 * split/ only; never on the IL tier. See split/split_create.h for the handle
 * and padding contracts.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: calls file-scope statics of vfft.c, so
 * it is included (via the dispatcher) after those are defined.
 */
#ifndef VFFT_SPLIT_C2C_IP_CREATE_SPLIT_H
#define VFFT_SPLIT_C2C_IP_CREATE_SPLIT_H

/* bank the in-place natural verdict (@nat, ord=nat) of the split stride
 * store: the mode, the deployed chain, the measured total. Memory only;
 * persistence behind config.wisdom_write. The order axis does not share a
 * cell with the scrambled one: a natural-order create can never perturb the
 * scrambled plan, and vice versa. */
static void _bank_nat_1d(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
                         int N, size_t K, int mode, double ns,
                         const int *fac, const int *var, int nf, int use_dif)
{
    vfft_proto_nat_entry_t nn;
    memset(&nn, 0, sizeof nn);
    nn.N = N;
    nn.K = K;
    nn.mode = mode;
    nn.nat_ns = ns;
    nn.nf = nf;
    nn.use_dif = use_dif;
    for (int s = 0; s < nf && s < STRIDE_MAX_STAGES; s++)
    {
        nn.factors[s] = fac[s];
        nn.variants[s] = var[s];
    }
    vw2_stride_bank_nat(&W->vw2, &nn, /*is_oop=*/0, _vw2_lay_of(cfg));
    _vw2_persist(W, cfg);
}

/* ── the split tier's ONE exit. Every handle this tier returns passes through
 * here, so a new early exit cannot skip the shared post-step (a skipped exit
 * ships mt_unsafe=0 — calloc's default, which spells "proven safe" — without
 * running the proof). The gate is cheap for ST creates (skipped). */
/* the incumbent arm of zturn-ts's in-place race: this handle's deployed
 * forward (vfft.c _exec_c2c_inplace: the stride plan, its JIT, the natural
 * tape), on the race's own planes */
typedef struct
{
    struct vfft_plan_s *h;
    double *re, *im;
} _zts_inc_ip_t;
static void _zts_inc_ip_run(void *v)
{
    _zts_inc_ip_t *c = (_zts_inc_ip_t *)v;
    _exec_c2c_inplace(c->h, VFFT_FORWARD, c->re, c->im);
}

/* a handle served by zturn-ts alone: no stride plan, no tapes */
static struct vfft_plan_s *_zts_handle(const vfft_cfg_t *cfg, int placement, int N, size_t K,
                                       vfft_zts_plan_t *z)
{
    struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_zts_destroy(z);
        return NULL;
    }
    h->transform = VFFT_C2C;
    h->placement = placement;
    h->layout = (int)cfg->layout;
    h->N = N;
    h->K = K;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zts = z;
    return h;
}

/* the cell's banked zturn-ts verdict rebuilt (NULL = no zts row, or the
 * build cannot serve it — the caller then treats the cell as held) */
static vfft_zts_plan_t *_zts_replay(struct vfft_wisdom_s *W, int N, size_t K, int place_ip,
                                    int lanes, int *held)
{
    vfft_zts_entry_t ze;
    vfft_zts_plan_t *z;
    *held = 0;
    if (!vw2_zts_lookup(&W->vw2, N, K, place_ip, &ze))
        return NULL;
    *held = 1;
    if (!ze.zts)
        return NULL;
    z = vfft_zts_create_chain(N, ze.chain, ze.nf, lanes);
    if (z && ze.tile && !vfft_zts_set_tile(z, ze.tile))
    {
        vfft_zts_destroy(z);
        z = NULL;
    }
    if (z && getenv("VFFT_NAT_LOG"))
    {
        char cs[48];
        vfft_zts_chain_str(z, cs, sizeof cs);
        fprintf(stderr, "[zts] N=%d q=%zu %s: replay chain %s tile=%zu src=wisdom\n",
                N, K, place_ip ? "ip" : "oop", cs, z->tile);
    }
    return z;
}

/* ── zturn-ts's CONTIGUOUS batch at T > 1: the THREADING verdict ──────
 * The one arm: the serial loop vs SLABS over the pool's slots (zts.h
 * vfft_zts_run_slab; split_execute.h), raced at create on the batch's own
 * cell and banked as eng=tcb tcmt= on its q=K row at the plan's thread
 * count (wisdom2_stride_reader.h vw2_stride_bank_tcmt with lay=split: the
 * IL contiguous wrapper's own grammar, vfft.c _tc_mt_decide, and its laws —
 * a slab is ceil(K/T) transforms, so the verdict moves with T and a lookup
 * never crosses it). No worker planes (K = 1 workers, VFFT_NO_TCMT) => no
 * arm: serial. VFFT_TCMT=0|1 pins the verdict (never replays, never banks). */
typedef struct
{
    struct vfft_plan_s *h;
    double *re, *im, *ore, *oim;
    int mt;
} _zts_tcmt_arm_t;
static void _zts_tcmt_arm_run(void *v)
{
    _zts_tcmt_arm_t *a = (_zts_tcmt_arm_t *)v;
    a->h->zts->mt = a->mt;
    _vfft_split_execute(a->h, VFFT_FORWARD, a->re, a->im, a->ore, a->oim);
}
typedef struct
{
    double *re, *im, *re0, *im0;
    size_t nb;
} _zts_tcmt_reseed_t;
static void _zts_tcmt_reseed(void *v)
{
    _zts_tcmt_reseed_t *r = (_zts_tcmt_reseed_t *)v;
    memcpy(r->re, r->re0, r->nb);
    memcpy(r->im, r->im0, r->nb);
}
static void _zts_tcmt_decide(struct vfft_plan_s *h, const vfft_cfg_t *cfg,
                             struct vfft_wisdom_s *W, int N, size_t K)
{
    const int T = h->nthreads;
    const int ip = (h->placement == VFFT_INPLACE);
    const int pl = ip ? VW2_PL_IP : VW2_PL_OOP;
    const int lg = getenv("VFFT_TCMT_VERBOSE") || getenv("VFFT_TCMT_LOG");
    const char *pin = getenv("VFFT_TCMT");
    int nw;
    h->zts->mt = 0;
    if (T <= 1 || K <= 1 || getenv("VFFT_NO_TCMT"))
        return;
    nw = T - 1;
    if ((size_t)nw > K - 1)
        nw = (int)(K - 1);
    if (nw > VFFT_ZTS_MT_MAX)
        nw = VFFT_ZTS_MT_MAX;
    if (vfft_zts_mt_planes(h->zts, nw) == 0)
        return; /* no worker planes: no arm */
    if (pin)
    {
        h->zts->mt = atoi(pin) ? 1 : 0;
        if (lg)
            fprintf(stderr, "[tcmt] zts N=%d K=%zu T=%d: pinned tcmt=%d (env; not banked)\n", N, K, T, h->zts->mt);
        return;
    }
    if (W && !cfg->recalibrate)
    {
        int v = 0;
        if (vw2_stride_lookup_tcmt(&W->vw2, VW2_T_C2C, N, K, VW2_ORD_NAT, pl, VW2_LAY_SPLIT, T, &v))
        {
            h->zts->mt = v;
            if (lg)
                fprintf(stderr, "[tcmt] zts N=%d K=%zu T=%d: replay tcmt=%d src=wisdom\n", N, K, T, v);
            return;
        }
    }
    { /* the race: serial loop vs slabs, on this cell's own planes */
        const size_t n = (size_t)N * K, nb = n * sizeof(double);
        double *re = (double *)vfft_aligned_alloc(nb), *im = (double *)vfft_aligned_alloc(nb);
        double *ore = ip ? re : (double *)vfft_aligned_alloc(nb);
        double *oim = ip ? im : (double *)vfft_aligned_alloc(nb);
        double *re0 = ip ? (double *)vfft_aligned_alloc(nb) : NULL;
        double *im0 = ip ? (double *)vfft_aligned_alloc(nb) : NULL;
        double st = 0, mt = 0;
        size_t i;
        if (!re || !im || !ore || !oim || (ip && (!re0 || !im0)))
        {
            vfft_aligned_free(re);
            vfft_aligned_free(im);
            if (!ip) { vfft_aligned_free(ore); vfft_aligned_free(oim); }
            vfft_aligned_free(re0);
            vfft_aligned_free(im0);
            return; /* no buffers: serial */
        }
        for (i = 0; i < n; i++)
        {
            re[i] = 1.0 + 1e-6 * (double)(i & 511);
            im[i] = 0.5 - 1e-6 * (double)(i & 255);
        }
        if (ip)
        {
            memcpy(re0, re, nb);
            memcpy(im0, im, nb);
        }
        {
            _zts_tcmt_arm_t a = { h, re, im, ore, oim, 0 };
            _zts_tcmt_arm_t b = { h, re, im, ore, oim, 1 };
            _zts_tcmt_reseed_t rs = { re, im, re0, im0, nb };
            const vfft_race_arm_t arms[2] = { { "serial", _zts_tcmt_arm_run, &a },
                                              { "slabs", _zts_tcmt_arm_run, &b } };
            vfft_race_proto_t proto;
            double ns[2];
            memset(&proto, 0, sizeof proto);
            proto.rounds = ip ? 9 : 7;
            proto.reps = ip ? 1 : (int)(32768u / (n ? n : 1)) + 1; /* >= ~30 us a sample */
            proto.agg = VFFT_RACE_MIN;
            proto.alternate = 1;
            proto.warm = 1;
            proto.reset = ip ? _zts_tcmt_reseed : NULL;
            proto.reset_ctx = ip ? &rs : NULL;
            vfft_race_run(&proto, arms, 2, ns); /* threaded arms: no pace */
            st = ns[0];
            mt = ns[1];
        }
        h->zts->mt = (mt < st);
        if (lg)
            fprintf(stderr, "[tcmt] zts N=%d K=%zu T=%d %s: race serial=%.0f slabs=%.0f -> %s\n",
                    N, K, T, ip ? "ip" : "oop", st, mt, h->zts->mt ? "SLABS" : "serial");
        vfft_aligned_free(re);
        vfft_aligned_free(im);
        if (!ip) { vfft_aligned_free(ore); vfft_aligned_free(oim); }
        vfft_aligned_free(re0);
        vfft_aligned_free(im0);
        if (W && vw2_stride_bank_tcmt(&W->vw2, VW2_T_C2C, N, K, VW2_ORD_NAT, pl, VW2_LAY_SPLIT,
                                      h->zts->mt, T, h->zts->mt ? mt : st) == VW2_OK)
            _vw2_persist(W, cfg);
    }
}

/* zts_try: 0 = the cell was consumed (no race); 1 = the natural verdict was
 * MEASURED this create (the @nat row absent, or recalibrate) and banked by
 * the natural block; 2 = measured, FREE (single stage), nothing banked. */
static vfft_plan _c2c_ip_finish_split(struct vfft_plan_s *h,
                                      struct vfft_wisdom_s *W,
                                      const vfft_cfg_t *cfg, int N,
                                      size_t K, int zts_try)
{
    /* MT-safety: flag plans whose codelet ignores the partial-lane count (so
     * _c2c_mt runs them whole-batch instead of K-splitting). Checked once on
     * the FINAL cplan (after any natural rebuild). A safety net (the DIF/LOG3
     * K-split twiddle defect is fixed at codegen); only MT plans K-split, so
     * single-threaded creates skip the check and its cost. */
    if (h->cplan)
        h->mt_unsafe = (h->nthreads > 1) ? !_c2c_mt_safe(h->cplan, h->exec_fwd) : 0;
    /* zturn-ts (split/rank1/zts_race.h): the cell's race, on a MEASURED
     * natural verdict only (feedback: race only when the banked cell is
     * missing, or under recalibrate), where the registry serves N in this
     * batch form — K=1 (lanes 1) or the lane-major 4 (lanes 4). The
     * incumbent = this handle's deployed forward. A win hands the handle to
     * zturn-ts (the stride plan and its tapes go) and banks eng=zts on the
     * cell — the stride row the natural block just banked gives way (same
     * key, same day, same metric); a loss leaves the stride row as the
     * record, or banks mode=free where the block banked nothing. */
    if (zts_try && h->cplan && !h->padded && vfft_zts_serves(N) &&
        (K == 1 || (K % (size_t)VFFT_ZTS_VW == 0 && K <= (size_t)VFFT_ZTS_LANES_MAX)))
    {
        const int lanes = K == 1 ? 1 : (int)K;
        const size_t n = (size_t)N * K;
        double *re = (double *)vfft_aligned_alloc(n * sizeof(double));
        double *im = (double *)vfft_aligned_alloc(n * sizeof(double));
        if (re && im)
        {
            _zts_inc_ip_t ic = { h, re, im };
            vfft_race_arm_t inc = { "stride", _zts_inc_ip_run, &ic };
            double *bufs[2] = { re, im };
            double wns = 0.0, ins = 0.0;
            vfft_zts_plan_t *z = vfft_zts_race(N, lanes, 1, &inc, bufs, 2, &wns, &ins);
            if (z)
            {
                vfft_proto_plan_destroy(h->cplan);
                h->cplan = NULL;
                h->exec_fwd = h->exec_bwd = NULL;
                free(h->nat_list);
                h->nat_list = NULL;
                vfft_aligned_free(h->nat_tmp);
                h->nat_tmp = NULL;
                free(h->nat_cyc_off);
                h->nat_cyc_off = NULL;
                if (h->nat_scr)
                {
                    natorder_scr_free(h->nat_scr);
                    free(h->nat_scr);
                    h->nat_scr = NULL;
                }
                h->nat_mode = 0;
                h->nat_ncyc = 0;
                h->mt_unsafe = 0;
                h->zts = z;
                vw2_zts_bank(&W->vw2, N, K, 1, z->chain, z->nf, z->tile, wns);
                _vw2_persist(W, cfg);
            }
            else if (zts_try == 2)
                _bank_nat_1d(W, cfg, N, K, VFFT_NAT_FREE, ins, NULL, NULL, 0, 0);
        }
        vfft_aligned_free(re);
        vfft_aligned_free(im);
    }
    return h;
}

static vfft_plan _vfft_create_c2c_ip_split(const vfft_cfg_t *cfg,
                                           vfft_batch ob,
                                           struct vfft_wisdom_s *W,
                                           const vfft_proto_registry_t *reg,
                                           int N,
                                           size_t K)
{
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE && ob)
    {
        vfft_batch b = ob;
        if (b->xform != (int)VFFT_C2C || b->oop || b->K != K || b->N != N) /* handle must match exactly */
        {                                                                  /* (an r2c handle's re/im are (N/2+1)*Kp; an OOP handle is 4-plane) */
            _vfft_warn("vfft_create: config.batch does not match this in-place C2C descriptor "
                       "(batch: %s%s N=%d K=%zu; config: C2C in-place N=%d K=%zu) — allocate "
                       "— INTERNAL INVARIANT (the plan allocates its own buffers); please report",
                       _vfft_tname(b->xform), b->oop ? " out-of-place" : "", b->N, b->K, N, K);
            return NULL;
        }
        size_t Kp = b->Kp;

        /* UNIFIED wisdom: the padded verdict is the (N,K) entry's exec_me, and
         * the pad plan IS the aligned (N,Kp) entry — both ordinary c2c cells. */
        /* The ladder is vfft.c's _pad_ladder: this arm only runs with an owned
         * batch, whose allocator ALREADY ran the ladder this same vfft_create,
         * so already_measured=1 (recalibrate fired there; never twice per
         * create) and ensure_pad_plan=1 (a PAD-verdict hit materialises the
         * aligned (N,Kp) plan cell that a verdict-only shipped row lacks). */
        const vfft_proto_wisdom_entry_t *te = NULL, *ae = NULL;
        int misaligned = (Kp != K);
        _pad_ladder(N, K, Kp, cfg, W, reg, /*ensure_pad_plan=*/1,
                    /*already_measured=*/1, &te, &ae);

        /* Select: PAD verdict -> the aligned (N,Kp) factorization @me=Kp ; else the (N,K) tight
         * factorization @me=K (tail on the padded buffer, always correct). */
        const int *facs = NULL, *vars = NULL;
        int nf = 0, use_dif = 0, exec_me = (int)K;
        if (misaligned && te && te->exec_me == (int)Kp && ae && ae->nf > 0)
        {
            facs = ae->factors;
            vars = ae->variants;
            nf = ae->nf;
            use_dif = ae->use_dif_forward;
            exec_me = (int)Kp;
        }
        else if (te && te->nf > 0)
        {
            facs = te->factors;
            vars = te->variants;
            nf = te->nf;
            use_dif = te->use_dif_forward;
            exec_me = (int)K;
        }
        else
            return NULL; /* no factorization available (e.g. prime N) */

        /* Backstop: verify the chosen factorization actually covers N (a wire-able-but-under-
         * covering factorization would silently compute the wrong-length transform). */
        {
            long long prod = 1;
            for (int i = 0; i < nf; i++)
                prod *= facs[i];
            if (prod != (long long)N)
                return NULL;
        }

        stride_plan_t *p = vfft_proto_plan_create_ex(N, Kp, facs, vars, nf, use_dif, reg);
        if (!p)
            return NULL;
        if (p->K != Kp) /* stride-match invariant: plan stride must equal buffer stride */
        {
            vfft_proto_plan_destroy(p);
            return NULL;
        }
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_proto_plan_destroy(p);
            return NULL;
        }
        h->transform = VFFT_C2C;
        h->placement = VFFT_INPLACE;
        h->layout = (int)cfg->layout; /* SPLIT by the batch+IL gate above */
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->cplan = p;
        h->padded = 1;
        h->exec_me = exec_me;
#ifdef VFFT_USE_JIT
        /* Only the ALIGNED pad leg (me=Kp) is eligible for the baked/JIT fast
         * path. The tail leg (exec_me==K, odd) MUST use the generic tail-capable executor,
         * so leave exec_*=NULL there (execute falls back to vfft_proto_execute_fwd/bwd). */
        if (exec_me == (int)Kp && p->num_stages > 0)
        {
            h->exec_fwd = vfft_proto_plan_jit_fwd(p);
            h->exec_bwd = vfft_proto_plan_jit_bwd(p);
        }
#endif
        return _c2c_ip_finish_split(h, W, cfg, N, K, 0);
    }

    /* ── c2c IN-PLACE ── */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE)
    {
        int zts_try = 0;
        if (K > 1 && cfg->batch_geom == VFFT_BATCH_CONTIGUOUS)
        {
            /* a CONTIGUOUS split batch (transform t at offset t*N in each
             * plane; owner 2026-11-04) = K independent K=1 transforms: the
             * q=1 cell's engine serves it when that engine has a contiguous
             * form — zturn-ts's _batch twin; the stride engine has none.
             * Plan the K=1 cell through this same create (its race and bank
             * included), then run that handle K times. */
            vfft_cfg_t c1 = *cfg;
            struct vfft_plan_s *h1;
            c1.howmany = 1;
            c1.batch_geom = VFFT_BATCH_LANE_MAJOR; /* identical at K=1 */
            h1 = (struct vfft_plan_s *)_vfft_create_c2c_ip_split(&c1, NULL, W, reg, N, 1);
            if (!h1)
                return NULL;
            if (!h1->zts)
            {
                _vfft_warn("vfft_create: batch=VFFT_BATCH_CONTIGUOUS on a SPLIT in-place C2C "
                           "N=%d K=%zu: the N=%d K=1 split cell is served by the stride "
                           "natural engine, which has no contiguous batch form (zturn-ts, "
                           "the only contiguous split engine, %s) — ask "
                           "VFFT_BATCH_LANE_MAJOR, or recalibrate the cell",
                           N, K, N, vfft_zts_serves(N) ? "lost the cell's race" : "has no cell at this N");
                vfft_destroy(h1);
                return NULL;
            }
            h1->K = K;
            _zts_tcmt_decide(h1, cfg, W, N, K); /* the threading verdict at T > 1 */
            return h1;
        }
        if (!cfg->recalibrate && cfg->order != VFI_ORDER_SCRAMBLED &&
            (K == 1 || (K % (size_t)VFFT_ZTS_VW == 0 && K <= (size_t)VFFT_ZTS_LANES_MAX)))
        {
            /* the cell's banked zturn-ts verdict, before the stride engine
             * plans anything (its scrambled cell, its natural tape). A
             * refusal (a build without the cell) falls through: the stride
             * reader sees the eng=zts row as absent and re-races. */
            int held;
            vfft_zts_plan_t *z = _zts_replay(W, N, K, 1, K == 1 ? 1 : (int)K, &held);
            if (z)
            {
                struct vfft_plan_s *hz = _zts_handle(cfg, VFFT_INPLACE, N, K, z);
                if (!hz)
                    return NULL;
                return _c2c_ip_finish_split(hz, W, cfg, N, K, 0);
            }
        }
        vfft_proto_dispatch_set_bluestein_wisdom(&W->bluestein);
        if (vfft_is_prime(N))
        {
            /* Prime N routes through Rader (radix-smooth N-1: M=N-1 + heuristic B,
             * no wisdom) or Bluestein (else: (M,B) FROM the bluestein wisdom). Only
             * the Bluestein cell consults wisdom, so calibrate-on-miss only there. */
            if (!vfft_is_radix_smooth(N - 1) &&
                (cfg->recalibrate || !bluestein_wisdom_lookup(&W->bluestein, N, K)))
            {
                size_t tot = (size_t)N * K;
                double *cre = (double *)vfft_aligned_alloc(tot * sizeof(double));
                double *cim = (double *)vfft_aligned_alloc(tot * sizeof(double));
                if (cre && cim)
                {
                    for (size_t i = 0; i < tot; i++)
                    {
                        cre[i] = (double)rand() / RAND_MAX - 0.5;
                        cim[i] = (double)rand() / RAND_MAX - 0.5;
                    }
                    double budget = (cfg->rigor == VFFT_MEASURE) ? 0.02 : 0.05;
                    int trials = (cfg->rigor == VFFT_MEASURE) ? 2 : 3;
                    bluestein_calibrate_one(&W->bluestein, N, K, reg, &W->c2c,
                                            cre, cim, budget, trials, NULL);
                    /* saved under the create's save flag, like every wisdom2
                     * persist, and not from a race scope the library knows
                     * measured badly; the verdict stays banked in memory
                     * either way. */
                    if (cfg->wisdom_write && !_vfft_scope_nosave() && W->path_bluestein[0])
                        bluestein_wisdom_save(&W->bluestein, W->path_bluestein);
                }
                vfft_aligned_free(cre);
                vfft_aligned_free(cim);
            }
        }
        else
        {
            const vfft_proto_wisdom_entry_t *e = vfft_proto_wisdom_lookup(&W->c2c, N, K);
            /* REGIME SEPARATION: order=NATURAL never re-measures/overwrites the SCRAMBLED entry — its
             * recalibrate governs only the natural verdict (below). It calibrates the scrambled cell just
             * once, when absent, to build the base plan p (the PURE-floor + race seed). So the natural
             * floor rides the STABLE banked scrambled chain, not a noisy fresh re-measure. order=DEFAULT/
             * SCRAMBLED keeps the old recalibrate-overwrites semantics. */
            int scr_recalib = cfg->recalibrate && cfg->order != VFI_ORDER_NATURAL;
            /* store first; the in-memory table stays the process cache
             * auto_plan_dispatch walks below. */
            {
                vfft_proto_wisdom_entry_t seb;
                if (!scr_recalib && !W->vw2_off_stride &&
                    vw2_stride_lookup(&W->vw2, 0, N, K, &seb))
                {
                    vfft_proto_wisdom_set(&W->c2c, &seb); /* store wins */
                    e = vfft_proto_wisdom_lookup(&W->c2c, N, K);
                }
            }
            if (!e || scr_recalib)
            {
                vfft_proto_wisdom_entry_t ne;
                if (_calibrate_c2c(N, K, cfg->rigor, reg, &ne) == 0)
                {
                    vfft_proto_wisdom_add(&W->c2c, &ne, 1);
                    vw2_stride_bank_entry(&W->vw2, &ne, 0);
                    _vw2_persist(W, cfg);
                }
            }
        }
        /* prime-aware: factorable -> CT/wisdom; prime -> Rader/Bluestein (override). */
        stride_plan_t *p = vfft_proto_auto_plan_dispatch(N, K, reg, &W->c2c);
        if (!p)
        {
            _vfft_warn("vfft_create: in-place C2C N=%d K=%zu — no CT "
                       "factorization, not prime, and the IL "
                       "prime/Bluestein engine cannot serve it "
                       "(K==1 INTERLEAVED only)",
                       N, K);
            return NULL;
        }
        /* (The natural block below never reads the scrambled entry; its base plan is `p` itself.) */
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            vfft_proto_plan_destroy(p);
            return NULL;
        }
        h->transform = VFFT_C2C;
        h->placement = VFFT_INPLACE;
        h->layout = (int)cfg->layout;
        h->N = N;
        h->K = K;
        h->nthreads = _vfft_plan_threads(cfg);
        h->cplan = p;
#ifdef VFFT_USE_JIT
        if (p->num_stages > 0)
        {
            /* staged plan -> resolve the whole-plan JIT/baked executor */
            h->exec_fwd = vfft_proto_plan_jit_fwd(p);
            h->exec_bwd = vfft_proto_plan_jit_bwd(p);
        }
        else
        {
            /* prime override plan (Rader/Bluestein): JIT the inner CT FFT — the override
             * executor calls inner_jit_* instead of the generic inner. exec_*=NULL keeps
             * the override-aware path. Accessors/setters are no-ops on the wrong kind. */
            stride_plan_t *in = stride_rader_inner_plan(p);
            if (!in)
                in = stride_bluestein_inner_plan(p);
            if (in)
            {
                vfft_proto_exec_fn ifwd = vfft_proto_plan_jit_fwd(in);
                vfft_proto_exec_fn ibwd = vfft_proto_plan_jit_bwd(in);
                if (ifwd && ibwd)
                {
                    stride_rader_set_inner_jit(p, ifwd, ibwd);
                    stride_bluestein_set_inner_jit(p, ifwd, ibwd);
                }
            }
        }
#endif
        /* ── VFFT_ORDER_NATURAL on the split stride engine: FREE, PURE_CYCLE, PSWAP (with
         * injected chains) or SCR, raced on a miss and banked as the @nat verdict. DEFAULT
         * is natural (vfft.h; field trial V31, owner 2026-10-13); SCRAMBLED alone skips
         * this block. */
        if (cfg->order == VFI_ORDER_NATURAL || cfg->order == VFI_ORDER_DEFAULT)
        {
            vfft_proto_nat_entry_t neb;
            const vfft_proto_nat_entry_t *ne =
                W->vw2_off_stride ? vfft_proto_nat_lookup(&W->c2c, N, K)
                                  : (vw2_stride_lookup_nat(&W->vw2, _vw2_lay_of(cfg), N, K, &neb) ? &neb : NULL);
            /* modes 6-8 name interleaved engines (retired from this enum by D2,
             * nat_modes.h): no split verdict, so the cell measures. The wisdom2
             * reader already reads such rows as absent; the frozen spike table
             * still carries them (read only under the kill switch, which
             * nothing sets today). */
            if (ne && ne->mode > VFFT_NAT_MAX)
                ne = NULL;
            int mode = (ne && !cfg->recalibrate) ? ne->mode : VFFT_NAT_UNSET;
            /* zturn-ts races the cell only where this create MEASURES it */
            zts_try = (!ne || cfg->recalibrate) ? (p->num_stages <= 1 ? 2 : 1) : 0;
            if (p->num_stages <= 1)
                mode = VFFT_NAT_FREE; /* single-stage / prime override: already natural, no tape */
            if (mode != VFFT_NAT_FREE) /* FREE needs no tape */
            {
                /* Self-contained natural — the DEPLOYED plan + its OWN chain drive everything; this path
                 * NEVER reads the scrambled entry. CONSUME (warm ne) rebuilds the deployed plan from ne's
                 * own chain (may differ from the scrambled p — a palindrome, or a single-radix leaf) and
                 * swaps it into cplan. MEASURE reuses the handle's scrambled plan p as the PURE-floor base
                 * + race seed (the PLAN object — not the wisdom entry). */
                int consume = (ne && !cfg->recalibrate);
                int dnf, ddif, dfac[STRIDE_MAX_STAGES], dvar[STRIDE_MAX_STAGES];
                if (consume)
                {
                    dnf = ne->nf;
                    ddif = ne->use_dif;
                    for (int s = 0; s < dnf && s < STRIDE_MAX_STAGES; s++)
                    {
                        dfac[s] = ne->factors[s];
                        dvar[s] = ne->variants[s];
                    }
                }
                else
                {
                    dnf = p->num_stages;
                    ddif = p->use_dif_forward;
                    for (int s = 0; s < dnf && s < STRIDE_MAX_STAGES; s++)
                    {
                        dfac[s] = p->factors[s];
                        dvar[s] = p->variants[s];
                    }
                }
                /* per-worker cycle scratch: h->nthreads slots of 2*K doubles (MT split).
                 * Sized by the PLAN'S SNAPSHOT, not the live pool: the pool is grow-only,
                 * so a live count taken here can be smaller than the one execute sees
                 * later, and the reorder slices tmp + slot*2*K per worker. The execute
                 * side clamps by the same h->nthreads (natorder_mt.h), so the two numbers
                 * cannot disagree. natorder_scratch_gate asserts this. */
                h->nat_tmp = (double *)vfft_aligned_alloc((size_t)(h->nthreads < 1 ? 1 : h->nthreads) * 2 * K * sizeof(double));
                if (!h->nat_tmp)
                {
                    vfft_destroy(h);
                    return NULL;
                }

                /* CONSUME SCR: rebuild the DIT scatter from the stored chain. */
                if (consume && mode == VFFT_NAT_SCR)
                {
                    natorder_scr_t sc;
                    stride_plan_t *sp = NULL;
                    int *scyc = NULL;
                    if (natorder_scr_build_dit(N, K, dfac, dnf, reg, &sc, &sp, &scyc))
                    {
                        h->nat_scr = (natorder_scr_t *)malloc(sizeof(natorder_scr_t));
                        if (h->nat_scr)
                        {
                            *h->nat_scr = sc;
                            vfft_proto_plan_destroy(h->cplan);
                            h->cplan = sp;
                            h->nat_list = scyc;
                            h->exec_fwd = NULL;
                            h->exec_bwd = NULL;
#ifdef VFFT_USE_JIT
                            h->exec_bwd = vfft_proto_plan_jit_bwd(h->cplan);
#endif
                        }
                        else
                        {
                            natorder_scr_free(&sc);
                            vfft_proto_plan_destroy(sp);
                            free(scyc);
                            mode = VFFT_NAT_PURE_CYCLE;
                        }
                    }
                    else
                        mode = VFFT_NAT_PURE_CYCLE;
                }
                /* CONSUME PURE/PSWAP (or SCR-demoted): rebuild the deployed plan from the stored chain. */
                if (consume && mode != VFFT_NAT_SCR && !h->nat_scr)
                {
                    stride_plan_t *dp = vfft_proto_plan_create_ex(N, K, dfac, dvar, dnf, ddif, reg);
                    if (dp)
                    {
                        vfft_proto_plan_destroy(h->cplan);
                        h->cplan = dp;
                        p = dp; /* probe + tape now follow the DEPLOYED plan */
                        h->exec_fwd = NULL;
                        h->exec_bwd = NULL;
#ifdef VFFT_USE_JIT
                        h->exec_fwd = vfft_proto_plan_jit_fwd(dp);
                        h->exec_bwd = vfft_proto_plan_jit_bwd(dp);
#endif
                    }
                    else
                    {
                        /* rebuild failed (OOM) -> honorable PURE on the ORIGINAL scrambled p. Reset the chain
                         * to p's OWN so the perm probe below detects on the plan actually deployed — otherwise
                         * detect runs the banked (possibly injected/leaf) chain dfac against p's spectrum,
                         * matches none, returns NULL, and the create hard-fails instead of degrading. */
                        mode = VFFT_NAT_PURE_CYCLE;
                        dnf = p->num_stages;
                        ddif = p->use_dif_forward;
                        for (int s = 0; s < dnf && s < STRIDE_MAX_STAGES; s++)
                        {
                            dfac[s] = p->factors[s];
                            dvar[s] = p->variants[s];
                        }
                    }
                }

                /* Perm + reorder tape from the deployed plan p (SCR already holds its cycle tape = scyc). */
                if (!h->nat_scr)
                {
                    size_t tot = (size_t)N * K;
                    double *cre = (double *)vfft_aligned_calloc(tot, sizeof(double));
                    double *cim = (double *)vfft_aligned_calloc(tot, sizeof(double));
                    int *M = NULL;
                    if (cre && cim)
                    {
                        cre[K] = 1.0; /* impulse at n0=1, lane 0 */
                        vfft_proto_execute_fwd(p, cre, cim, K);
                        M = vfft_natorder_detect(N, dfac, dnf, K, cre, cim, 1);
                    }
                    vfft_aligned_free(cre);
                    vfft_aligned_free(cim);
                    if (!M)
                    {
                        vfft_destroy(h);
                        return NULL;
                    }

                    if (mode == VFFT_NAT_PSWAP)
                        h->nat_list = vfft_natorder_mk_pairs(N, M); /* CONSUME PSWAP (single-leaf => empty tape = FREE) */
                    else if (mode == VFFT_NAT_PURE_CYCLE)
                        h->nat_list = vfft_natorder_mk_cycles(N, M); /* CONSUME PURE */
                    else                                             /* mode == VFFT_NAT_UNSET: MEASURE */
                    {
                        h->nat_list = vfft_natorder_mk_cycles(N, M); /* race PURE-floor baseline */
                        if (h->nat_list)
                        {
                            /* OPPORTUNISTIC PSWAP: p's perm is an involution => pairs beat cycles on the SAME
                             * plan (deterministic free win). GATE: if a single-stage [N] leaf exists, DON'T
                             * short-circuit — fall to the race so it also weighs the FREE-reorder single-radix
                             * candidate. The race re-injects the calibrated palindrome. */
                            int has_leaf = (N > 1 && N < VFFT_PROTO_REG_MAX_RADIX && reg->n1_fwd[N]);
                            int *opp = (!has_leaf) ? vfft_natorder_mk_pairs(N, M) : NULL;
                            if (opp)
                            {
                                free(h->nat_list);
                                h->nat_list = opp; /* deployed plan p unchanged */
                                mode = VFFT_NAT_PSWAP;
                                _bank_nat_1d(W, cfg, N, K, mode, 0.0, dfac, dvar, dnf, ddif);
                            }
                            else
                            {
                                /* RACE (PURE vs injected-palindrome/single-leaf PSWAP vs DIT-SCR; 5% margin),
                                 * seeded from the deployed chain dfac (the PLAN object, never the scr entry). */
                                vfft_natorder_verdict_t v;
                                /* the create-race counter: this racer (natorder_calibrate.h) is about to time */
                                _vfft_create_race_count++;
                                vfft_natorder_race(N, K, reg, p, h->nat_list, h->nat_tmp, dfac, dnf, &v);
                                mode = v.mode;
                                if (mode == VFFT_NAT_PSWAP)
                                {
                                    vfft_proto_plan_destroy(h->cplan);
                                    h->cplan = v.planB;
                                    free(h->nat_list);
                                    h->nat_list = v.pairs;
                                    h->exec_fwd = NULL;
                                    h->exec_bwd = NULL;
#ifdef VFFT_USE_JIT
                                    h->exec_fwd = vfft_proto_plan_jit_fwd(h->cplan);
                                    h->exec_bwd = vfft_proto_plan_jit_bwd(h->cplan);
#endif
                                    /* deployed = injected chain: v.factors, uniform-prof variants (stage0 FLAT), dif=0. */
                                    int fac2[STRIDE_MAX_STAGES], var2[STRIDE_MAX_STAGES];
                                    for (int s = 0; s < v.nf && s < STRIDE_MAX_STAGES; s++)
                                    {
                                        fac2[s] = v.factors[s];
                                        var2[s] = s ? v.prof : 0;
                                    }
                                    _bank_nat_1d(W, cfg, N, K, mode, v.ns, fac2, var2, v.nf, 0);
                                }
                                else if (mode == VFFT_NAT_SCR)
                                {
                                    h->nat_scr = (natorder_scr_t *)malloc(sizeof(natorder_scr_t));
                                    if (!h->nat_scr)
                                    {
                                        free(M);
                                        vfft_proto_plan_destroy(v.scr_plan);
                                        natorder_scr_free(&v.scr);
                                        free(v.scr_cycles);
                                        vfft_destroy(h);
                                        return NULL;
                                    }
                                    *h->nat_scr = v.scr;
                                    vfft_proto_plan_destroy(h->cplan);
                                    h->cplan = v.scr_plan;
                                    free(h->nat_list);
                                    h->nat_list = v.scr_cycles;
                                    h->exec_fwd = NULL;
                                    h->exec_bwd = NULL;
#ifdef VFFT_USE_JIT
                                    h->exec_bwd = vfft_proto_plan_jit_bwd(h->cplan);
#endif
                                    _bank_nat_1d(W, cfg, N, K, mode, v.ns, dfac, dvar, dnf, ddif); /* the DIT base chain */
                                }
                                else /* PURE floor: deployed = p, bank p's chain */
                                    _bank_nat_1d(W, cfg, N, K, VFFT_NAT_PURE_CYCLE, v.ns, dfac, dvar, dnf, ddif);
                            }
                        }
                    }
                    free(M);
                    if (!h->nat_list && mode != VFFT_NAT_SCR)
                    {
                        vfft_destroy(h);
                        return NULL;
                    }
                }
            }
            /* MT metadata for the reorder pass: PURE + SCR-backward split cycles (need offsets);
             * PSWAP splits pairs (count only). nat_list is now final for the chosen mode. */
            if (mode == VFFT_NAT_PSWAP)
                h->nat_ncyc = vfft_natorder_pair_count(h->nat_list);
            else if (mode == VFFT_NAT_PURE_CYCLE || mode == VFFT_NAT_SCR)
            {
                h->nat_cyc_off = vfft_natorder_cycle_offsets(h->nat_list, &h->nat_ncyc);
                if (!h->nat_cyc_off)
                {
                    vfft_destroy(h);
                    return NULL;
                }
            }
#ifdef VFFT_USE_JIT
            /* SCR: JIT/bake the OOP scratch-fill's stages 1.. (stage 0 is a bare n1 loop). Only
             * meaningful for nf>=3; execute_fwd_oop_jit skips the call when sub has 1 stage. */
            if (mode == VFFT_NAT_SCR && h->nat_scr && h->nat_scr->sub.num_stages > 1)
                h->nat_scr->sub_jit_fwd = vfft_proto_plan_jit_fwd(&h->nat_scr->sub);
#endif
            h->nat_mode = mode;
        }
        /* the MT-safety gate and zturn-ts's race are in _c2c_ip_finish_split,
         * the tier's one exit. */

        return _c2c_ip_finish_split(h, W, cfg, N, K, zts_try);
    }
    return NULL; /* unreachable: the one call site guards on the same
                  * condition, and every path in the block above returns. */
}

#endif /* VFFT_SPLIT_C2C_IP_CREATE_SPLIT_H */
