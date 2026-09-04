/* c2c_ip_create.h — the c2c IN-PLACE create tier (migration step 24).
 *
 * WHAT THIS IS
 * ------------
 * Both in-place c2c arms of _vfft_create_inner, in their original order:
 *
 *   1. the CALLER-SUPPLIED BATCH arm (`... && ob`) — an owned-batch descriptor
 *      was handed in, so the plan serves that exact handle;
 *   2. the general in-place arm, which allocates its own buffers.
 *
 * Arm 1 falls through to arm 2 only by not matching; each returns on every
 * path, so the pair lifts out behind one guard.
 *
 * WHY THE HANDLE IS CHECKED EXACTLY
 * ---------------------------------
 * Arm 1 refuses unless xform, oop, K and N all match. The shapes are genuinely
 * incompatible rather than merely different: an r2c handle's re/im planes are
 * (N/2+1)*Kp, and an OOP handle is 4-plane. A mismatched handle is a caller
 * error and is refused LOUDLY, per the tree's diagnostic directive.
 *
 * PADDED vs UNPADDED IS A WISDOM QUESTION, NOT A BRANCH
 * ----------------------------------------------------
 * Kp is the batch's padded lane count. The padded verdict is not a separate
 * planning path: it is the (N,K) entry's exec_me, and the pad plan IS the
 * aligned (N,Kp) entry -- both ordinary c2c cells in the one unified store.
 * `misaligned = (Kp != K)` selects between them; nothing here invents a cutoff.
 *
 * 🔴 te/ae are re-looked-up after every set: `wisdom_set` may realloc, so a
 * pointer held across a set is a dangling read. The original code does this
 * deliberately and the move preserves it.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. It calls file-scope statics that live in vfft.c, so
 * it must be included after those are defined and before _vfft_create_inner.
 *
 * The six parameters are the block's complete free-variable set, derived
 * rather than guessed: this is the first tier that genuinely needs `ob` (the
 * owned batch it serves) and `N` alongside cfg/W/reg/K.
 */
#ifndef VFFT_OOP_C2C_IP_CREATE_H
#define VFFT_OOP_C2C_IP_CREATE_H

/* ── the arms of the natural / scr-mode MEASURE races ────────────────
 * One context serves the five sites (four here, one in c2c_oop_create.h):
 * the incumbent is THIS handle's real execute (in-place door, or the OOP
 * door when oop=1); the challenger is whichever candidate the site built —
 * the natord cascade (zt/zs by zroute) or the IL engine (il2/il3/ilp). The
 * protocol constants stay at each site (support/race.h). */
typedef struct
{
    struct vfft_plan_s *h;
    int oop;                    /* 1: vfft_execute(h, FWD, r0 -> rz) */
    vfft_zturn2_plan_t *zt;     /* cascade challenger, route by zroute */
    vfft_zsplit_plan_t *zs;
    int zroute;
    vfft_il2p_plan_t *il2;      /* IL challenger: first non-NULL serves */
    vfft_il3p_plan_t *il3;
    vfft_ilprime_plan_t *ilp;
    vfft_oop11_fn mono;         /* the alias-tolerant solo (MONO verdict) */
    double *rz, *r0;            /* the aliased race buffer and its seed */
    size_t nb;                  /* bytes to re-seed per burst */
} _c2c_race_ctx_t;
static void _c2c_race_inc(void *v)
{
    _c2c_race_ctx_t *c = (_c2c_race_ctx_t *)v;
    if (c->oop)
        vfft_execute(c->h, VFFT_FORWARD, c->r0, NULL, c->rz, NULL);
    else
        vfft_execute(c->h, VFFT_FORWARD, c->rz, NULL, c->rz, NULL); /* aliased z -> z */
}
static void _c2c_race_chal(void *v)
{
    _c2c_race_ctx_t *c = (_c2c_race_ctx_t *)v;
    const double *in = c->oop ? c->r0 : c->rz;
    if (c->zt || c->zs)
    {
        if (c->zroute)
            vfft_zturn2_execute_fwd(c->zt, in, c->rz);
        else
            vfft_zsplit_execute_fwd(c->zs, in, c->rz);
    }
    else if (c->il2)
        vfft_il2p_execute_fwd(c->il2, in, c->rz);
    else if (c->il3)
        vfft_il3p_execute_fwd(c->il3, in, c->rz);
    else if (c->mono)
        c->mono(in, 0, c->rz, 0, 0, 0, 1, 0, 1, 0, 1);   /* one leg, z -> z legal */
    else
        vfft_ilprime_execute_fwd(c->ilp, in, c->rz);
}
static void _c2c_race_reseed(void *v)
{
    _c2c_race_ctx_t *c = (_c2c_race_ctx_t *)v;
    memcpy(c->rz, c->r0, c->nb);
}

/* ── the tier's ONE exit. Every handle this create returns passes through
 * here, so a new early exit cannot skip the shared post-step (the audit
 * found two that did: the padded-batch exit and the IL-prime exit both
 * shipped mt_unsafe=0 — calloc's default, which spells "proven safe" —
 * without running the proof). The gate is cheap for ST creates (skipped)
 * and engine handles (no cplan: nothing K-splits). */
/* ── IN-PLACE INTERLEAVED c2c: the IL tier's own create (2026-09-03) ──────
 * Owner: "we DO NOT see split as a fallback of IL". No split plan is built
 * for an interleaved caller. The cell is served by an IL engine — the K=1
 * engines (pair / chain3 / prime, with their banked forms) and, at
 * N >= 2048, the cascade (kind-4 recipe; natord under order=NATURAL) — and
 * the verdict between them is a raced IL-vs-IL verdict on the cell's own
 * mode row (@scrmode for DEFAULT/SCRAMBLED, @nat for NATURAL: mode=ilp |
 * mode=zcasc). A mode=conv or tape row is not an IL verdict and re-races.
 * With one legal arm it serves and banks; with none the create REFUSES —
 * there is nothing to fall back to, by design. Lane-major K>1 interleaved
 * (only an explicit VFFT_BATCH_LANE_MAJOR reaches here; DEFAULT geometry is
 * the transform-contiguous wrapper) is refused: measured 2026-09-03, it
 * lost to transform-contiguous at every cell, and its only engine was the
 * split K-lane plan behind a convert. Census before this path: 177 of 255
 * sizes below 257 executed through the convert in place; 3 of 255 out of
 * place, with the same kernels. */
static void _bank_ipmode_1d(struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                            int N, int mode, double ns)
{
    vfft_proto_nat_entry_t nn;
    if (!W || W->vw2_off_stride)
        return;
    memset(&nn, 0, sizeof nn);
    nn.N = N;
    nn.K = 1;
    nn.mode = mode;
    nn.nat_ns = ns;
    nn.raced = 1;
    nn.nf = 1;                 /* the dummy chain: mode=ilp emits no recipe,
                                * mode=zcasc emits the ref= signpost */
    nn.factors[0] = N;
    nn.ref_comp = _zcasc_ref_is_comp(W, N, mode); /* the recipe row that SERVED */
    nn.ref_ilp = _ilp_ref_of(W, N, mode);
    if (cfg->order == VFFT_ORDER_NATURAL)
        vw2_stride_bank_nat(&W->vw2, &nn, /*is_oop=*/0, _vw2_lay_of(cfg));
    else
        vw2_stride_bank_scrmode(&W->vw2, &nn, _vw2_lay_of(cfg));
    _vw2_persist(W, cfg);
}

static vfft_plan _c2c_ip_finish(struct vfft_plan_s *h,
                                struct vfft_wisdom_s *W,
                                const vfft_config_t *cfg, int N);

static vfft_plan _c2c_ip_create_il(const vfft_config_t *cfg,
                                   struct vfft_wisdom_s *W,
                                   const vfft_proto_registry_t *reg,
                                   int N, size_t K)
{
    const int nat = (cfg->order == VFFT_ORDER_NATURAL);
    struct vfft_plan_s *h;
    vfft_il2p_plan_t *il2 = NULL;
    vfft_il3p_plan_t *il3 = NULL;
    vfft_oop11_fn mono_f = 0, mono_b = 0;   /* the alias-tolerant solo (MONO verdict) */
    vfft_ilprime_plan_t *ilp = NULL;
    vfft_zturn2_plan_t *zt = NULL;
    int have_k1 = 0, mode = VFFT_NAT_UNSET, raced_row = 0;
    (void)reg;
    if (K > 1)
    {
        _vfft_warn("vfft_create: in-place C2C N=%d howmany=%zu with layout=INTERLEAVED and "
                   "batch_geom=LANE_MAJOR has no interleaved engine (lane-major lost to "
                   "transform-contiguous at every measured cell, 2026-09-03); use "
                   "VFFT_BATCH_DEFAULT / VFFT_BATCH_TRANSFORM_CONTIGUOUS",
                   N, K);
        return NULL;
    }
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
        return NULL;
    h->transform = VFFT_C2C;
    h->placement = VFFT_INPLACE;
    h->layout = (int)cfg->layout;
    h->N = N;
    h->K = 1;
    h->nthreads = _vfft_plan_threads(cfg);
    if (getenv("VFFT_NAT_LOG"))
        fprintf(stderr, "[ipil] N=%d order=%s: IL create (no split baseline)\n",
                N, nat ? "natural" : (cfg->order == VFFT_ORDER_SCRAMBLED ? "scrambled" : "default"));

    /* 1. the banked verdict for THIS cell (order-keyed rows) */
    if (W && !W->vw2_off_stride && !cfg->recalibrate)
    {
        vfft_proto_nat_entry_t eb;
        const int hit = nat
            ? vw2_stride_lookup_nat(&W->vw2, _vw2_lay_of(cfg), N, 1, &eb)
            : vw2_stride_lookup_scrmode(&W->vw2, _vw2_lay_of(cfg), N, 1, &eb);
        if (hit && (eb.mode == VFFT_NAT_ILP || eb.mode == VFFT_NAT_ZCASC))
        {
            mode = eb.mode;
            raced_row = 1;
        }
        /* mode=conv / tape / free rows: not IL verdicts — fall to the race */
    }

    /* 2. the K=1 IL engine candidate: the planned row's route — MONO (the
     *    alias-tolerant solo, 2026-09-04), pair, chain3, else prime */
    if (!getenv("VFFT_NO_NAT_ILP") && (mode != VFFT_NAT_ZCASC || N < 2048))
    {
        _k1_il_candidate(W, cfg, N, &il2, &il3);
        if (!il2 && !il3)
            (void)_k1_il_mono_candidate(W, N, &mono_f, &mono_b);
        if (!il2 && !il3 && !mono_f)
            ilp = _ilprime_create_banked(W, cfg, N);
        have_k1 = (il2 || il3 || mono_f || ilp) ? 1 : 0;
    }

    /* 3. the cascade candidate at N >= 2048 (natord under NATURAL) */
    if (N >= _vfft_zcasc_min_n() && !getenv("VFFT_NO_K1Z_IP") &&
        !getenv("VFFT_NO_NAT_ZCASC") && W && !W->vw2_off_stride &&
        (mode != VFFT_NAT_ILP || !have_k1))
    {
        vfft_config_t rcfg = *cfg;
        vfft_zsplit_plan_t *zs = NULL;
        int zr = 0;
        rcfg.recalibrate = 0;
        if (_k1z_wisdom_replay(&rcfg, W, N, &zs, &zt, &zr) ||
            _k1z_race_and_bank(&rcfg, W, N, /*ip=*/1, &zs, &zt, &zr))
        {
            if (zs)
                vfft_zsplit_destroy(zs);
            if (zt && nat && !vfft_zturn2_set_natord(zt, 1))
            {
                vfft_zturn2_destroy(zt);
                zt = NULL;
            }
        }
    }

    /* 4. replay a banked verdict when its engine built */
    if (mode == VFFT_NAT_ILP && !have_k1) mode = VFFT_NAT_UNSET;
    if (mode == VFFT_NAT_ZCASC && !zt)  mode = VFFT_NAT_UNSET;
    if (mode == VFFT_NAT_UNSET)
    {
        if (have_k1 && zt)
        {
            /* the IL-vs-IL race: this cell's K=1 engine vs the cascade, both
             * aliased z -> z on scratch, the tier's protocol (5 rounds,
             * alternated, median, re-seeded per burst) */
            double *rz = (double *)malloc(2 * (size_t)N * sizeof(double));
            double *r0 = (double *)malloc(2 * (size_t)N * sizeof(double));
            if (rz && r0)
            {
                const int reps = N <= 256 ? 200 : (N <= 1024 ? 80 : 32);
                double ns[2]; /* [0] K=1 engine, [1] cascade */
                _c2c_race_ctx_t ca = { h, 0, NULL, NULL, 0, il2, il3, ilp, mono_f, rz, r0,
                                       2 * (size_t)N * sizeof(double) };
                _c2c_race_ctx_t cb = { h, 0, zt, NULL, 1, NULL, NULL, NULL, 0, rz, r0,
                                       2 * (size_t)N * sizeof(double) };
                const vfft_race_arm_t arms[2] = {
                    { "ilp", _c2c_race_chal, &ca }, { "zcasc", _c2c_race_chal, &cb } };
                const vfft_race_proto_t proto = { 5, reps, VFFT_RACE_MEDIAN, 1, 0, _c2c_race_reseed, &ca };
                for (long i = 0; i < 2L * N; i++)
                    r0[i] = (double)rand() / RAND_MAX - 0.5;
                _vfft_pool_arm(h->nthreads);
                vfft_race_run(&proto, arms, 2, ns);
                mode = (ns[1] < ns[0]) ? VFFT_NAT_ZCASC : VFFT_NAT_ILP;
                if (getenv("VFFT_NAT_LOG"))
                    fprintf(stderr, "[ipil] N=%d race: ilp=%.0fns zcasc=%.0fns -> %s\n",
                            N, ns[0], ns[1], mode == VFFT_NAT_ZCASC ? "ZCASC" : "ILP");
                _bank_ipmode_1d(W, cfg, N, mode, mode == VFFT_NAT_ZCASC ? ns[1] : ns[0]);
            }
            free(rz);
            free(r0);
            if (mode == VFFT_NAT_UNSET) mode = have_k1 ? VFFT_NAT_ILP : VFFT_NAT_ZCASC;
        }
        else if (have_k1)
        {
            mode = VFFT_NAT_ILP;
            if (!raced_row) _bank_ipmode_1d(W, cfg, N, mode, 0.0);
        }
        else if (zt)
        {
            mode = VFFT_NAT_ZCASC;
            if (!raced_row) _bank_ipmode_1d(W, cfg, N, mode, 0.0);
        }
    }

    /* 5. attach the verdict; the loser dies here */
    if (mode == VFFT_NAT_ZCASC && zt)
    {
        h->zturn = zt;
        h->zroute = 1;
        zt = NULL;
        h->nat_mode = nat ? VFFT_NAT_ZCASC : 0;
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[ipil] N=%d: %s ZCASC%s\n", N,
                    raced_row ? "replay" : "attach", nat ? " (natord)" : "");
    }
    else if (mode == VFFT_NAT_ILP && have_k1)
    {
        h->k1il2p = il2;
        h->k1il3p = il3;
        h->k1ilpr = ilp;
        h->k1_mono_ilf = mono_f;
        h->k1_mono_ilb = mono_b;
        il2 = NULL; il3 = NULL; ilp = NULL;
        h->nat_mode = nat ? VFFT_NAT_ILP : 0;
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[ipil] N=%d: %s ILP (%s)\n", N,
                    raced_row ? "replay" : "attach",
                    h->k1il2p ? "il2p" : h->k1il3p ? "il3p"
                              : h->k1_mono_ilf ? "mono" : "ilprime");
    }
    if (il2) vfft_il2p_destroy(il2);
    if (il3) vfft_il3p_destroy(il3);
    if (ilp) vfft_ilprime_destroy(ilp);
    if (zt)  vfft_zturn2_destroy(zt);
    if (!h->zturn && !h->k1il2p && !h->k1il3p && !h->k1ilpr && !h->k1_mono_ilf)
    {
        _vfft_warn("vfft_create: in-place C2C N=%d with layout=INTERLEAVED has no "
                   "interleaved engine yet (no mono/pair/chain3/prime kernel serves "
                   "this N%s) — the IL kernel set does not cover it; nothing to fall "
                   "back to by design",
                   N, N >= 2048 ? ", and no cascade recipe built" : "");
        free(h);
        return NULL;
    }
    return _c2c_ip_finish(h, W, cfg, N);
}

static vfft_plan _c2c_ip_finish(struct vfft_plan_s *h,
                                struct vfft_wisdom_s *W,
                                const vfft_config_t *cfg, int N)
{
    /* MT-safety: flag plans whose codelet ignores the partial-lane count (so
     * _c2c_mt runs them whole-batch instead of K-splitting). Checked once on
     * the FINAL cplan (after any natural rebuild). Safety net now that the
     * DIF/LOG3 K-split twiddle bug is fixed at codegen; only MT plans
     * K-split, so single-threaded creates skip the check and its cost. */
    if (h->cplan)
        h->mt_unsafe = (h->nthreads > 1) ? !_c2c_mt_safe(h->cplan, h->exec_fwd) : 0;
    /* the cascade MT verdict (C1.9) for the IN-PLACE cascade too (owner,
     * 2026-09-02): until now only the OOP exit asked "serial or threaded?",
     * so an in-place K=1 cascade at T>1 ran on one core. Same replay-or-
     * race, aliased arms, its own per-T tokens; natord cascades cannot
     * engage and bank the "no" implicitly. */
    if (h->zroute && h->zturn && h->K == 1 && h->nthreads > 1)
        _zt_mt_replay_or_race(h, W, cfg, N);
    return h;
}

static vfft_plan _vfft_create_c2c_ip(const vfft_config_t *cfg,
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

        /* UNIFIED wisdom (single spike_wisdom.txt): the padded verdict is the (N,K) entry's
         * exec_me, and the pad plan IS the aligned (N,Kp) entry — both ordinary c2c cells. */
        /* THE ladder is one body now (vfft.c:_pad_ladder — A1): this arm only runs
         * with an owned batch, whose allocator ALREADY ran the ladder this same
         * vfft_create, so already_measured=1 (recalibrate fired there; never twice
         * per create) and ensure_pad_plan=1 (a PAD-verdict hit materialises the
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
        /* Wrinkle C: only the ALIGNED pad leg (me=Kp) is eligible for the baked/JIT fast
         * path. The tail leg (exec_me==K, odd) MUST use the generic tail-capable executor,
         * so leave exec_*=NULL there (execute falls back to vfft_proto_execute_fwd/bwd). */
        if (exec_me == (int)Kp && p->num_stages > 0)
        {
            h->exec_fwd = vfft_proto_plan_jit_fwd(p);
            h->exec_bwd = vfft_proto_plan_jit_bwd(p);
        }
#endif
        return _c2c_ip_finish(h, W, cfg, N);
    }

    /* ── c2c IN-PLACE ── */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED)
        return _c2c_ip_create_il(cfg, W, reg, N, K);   /* the IL tier's own create (2026-09-03) */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE)
    {
        vfft_proto_dispatch_set_bluestein_wisdom(&W->bluestein);
        if (_vfft_is_prime(N))
        {
            /* Prime N routes through Rader (radix-smooth N-1: M=N-1 + heuristic B,
             * no wisdom) or Bluestein (else: (M,B) FROM the bluestein wisdom). Only
             * the Bluestein cell consults wisdom, so calibrate-on-miss only there. */
            if (!_vfft_is_radix_smooth(N - 1) &&
                (cfg->recalibrate || !bluestein_wisdom_lookup(&W->bluestein, N, K)))
            {
                size_t tot = (size_t)N * K;
                double *cre = (double *)malloc(tot * sizeof(double));
                double *cim = (double *)malloc(tot * sizeof(double));
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
                    /* the create path honours the read-only default like
                     * every wisdom2 persist (vfft.h: "DEFAULT is read-only
                     * wisdom"); the verdict stays banked in memory either
                     * way, and the explicit vfft_wisdom_save API is the
                     * ungated door — that one IS user intent. */
                    if (cfg->wisdom_write && W->path_bluestein[0])
                        bluestein_wisdom_save(&W->bluestein, W->path_bluestein);
                }
                free(cre);
                free(cim);
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
            int scr_recalib = cfg->recalibrate && cfg->order != VFFT_ORDER_NATURAL;
            /* wave-4: store-first; the in-memory table stays the process
             * cache auto_plan_dispatch walks below. */
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
        /* (Self-contained natural design: the old C1 scrambled-entry bank-from-plan is GONE — the natural
         * block below no longer reads the scrambled entry, so it can't hard-fail for want of one. Its base
         * plan is `p` itself, which auto_plan_dispatch always built here.) */
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
        /* ── VFFT_ORDER_NATURAL (P1b: FREE + PURE_CYCLE + PSWAP w/ injected chains; the
         * calibrator race stamps the verdict into wisdom v7. SCR/LEAF-IP still degrade to
         * PURE until their executors land). order==DEFAULT leaves everything below
         * untouched — byte-identical kill switch. */
        if (cfg->order == VFFT_ORDER_NATURAL)
        {
            vfft_proto_nat_entry_t neb;
            const vfft_proto_nat_entry_t *ne =
                W->vw2_off_stride ? vfft_proto_nat_lookup(&W->c2c, N, K)
                                  : (vw2_stride_lookup_nat(&W->vw2, _vw2_lay_of(cfg), N, K, &neb) ? &neb : NULL);
            int mode = (ne && !cfg->recalibrate) ? ne->mode : VFFT_NAT_UNSET;
            const int nat_raced = (ne && !cfg->recalibrate &&
                                   !W->vw2_off_stride) ? ne->raced : 0;
            if (p->num_stages <= 1)
                mode = VFFT_NAT_FREE; /* single-stage / prime override: already natural, no tape */
            /* Natural-terminator cascade, built as a CANDIDATE for the race below from the kind-4
             * chain with recalibrate cleared. Kill switch: VFFT_NO_NAT_ZCASC.
             * See docs/design/vfft_front_door.md. */
            /* CONSUME ZCASC: attach and skip the whole tape build. A banked
             * ZCASC whose kind-4 line has since vanished (or been refused)
             * degrades to UNSET — re-measure, never hard-fail. */
            if (mode != VFFT_NAT_FREE && mode != VFFT_NAT_ZCASC &&
                mode != VFFT_NAT_ILP)
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
                h->nat_tmp = (double *)malloc((size_t)(h->nthreads < 1 ? 1 : h->nthreads) * 2 * K * sizeof(double));
                if (!h->nat_tmp)
                {
                    vfft_destroy(h);
                    return NULL;
                }

                /* CONSUME SCR (parked; rebuild the DIT scatter from the stored chain). */
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
                    double *cre = (double *)calloc(tot, sizeof(double));
                    double *cim = (double *)calloc(tot, sizeof(double));
                    int *M = NULL;
                    if (cre && cim)
                    {
                        cre[K] = 1.0; /* impulse at n0=1, lane 0 */
                        vfft_proto_execute_fwd(p, cre, cim, K);
                        M = vfft_natorder_detect(N, dfac, dnf, K, cre, cim, 1);
                    }
                    free(cre);
                    free(cim);
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
                             * candidate (the 2D 64x16 lesson). The race re-injects the calibrated palindrome. */
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
                                /* HARNESS: this racer is about to time. It lives in
                                 * natorder_calibrate.h, NOT in this file, which is why the
                                 * original census missed it - that census enumerated clock
                                 * calls in vfft.c plus their local callers, and a racer
                                 * DEFINED in another header is invisible to both passes.
                                 * The cost was concrete: c2c.split.ip.nat has no banked nat
                                 * entry, so it takes this branch every time and picked
                                 * nat=5/natcyc=96 in 8 of 10 runs and nat=4/natcyc=34 in the
                                 * other 2 - while reporting races=0 and therefore claiming to
                                 * be safe to diff. A fingerprint that flaps 20% of the time
                                 * under a purity flag is worse than no fingerprint. */
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
        /* the MT-safety gate moved to _c2c_ip_finish — the tier's one exit —
         * so the early exits above cannot skip it. */

        return _c2c_ip_finish(h, W, cfg, N);
    }
    return NULL; /* unreachable: the one call site guards on the same
                  * condition, and every path in the block above returns. */
}

#endif /* VFFT_OOP_C2C_IP_CREATE_H */
