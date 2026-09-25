/* c2c_ip_create.h — the c2c IN-PLACE create tier.
 *
 * The in-place c2c arms of _vfft_create_inner:
 *
 *   1. the CALLER-SUPPLIED BATCH arm (`... && ob`) — an owned-batch descriptor
 *      was handed in, so the plan serves that exact handle;
 *   2. the general in-place arms, which allocate their own buffers: the
 *      interleaved tier (_c2c_ip_create_il) and the split stride engine.
 *
 * Arm 1 falls through to arm 2 only by not matching; each returns on every
 * path.
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
 * pointer held across a set is a dangling read.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. It calls file-scope statics that live in vfft.c, so
 * it must be included after those are defined and before _vfft_create_inner.
 */
#ifndef VFFT_OOP_C2C_IP_CREATE_H
#define VFFT_OOP_C2C_IP_CREATE_H

/* ── a two-arm race context: the incumbent is a handle's real execute
 * (in-place, or the OOP door when oop=1), the challenger an IL engine plan
 * (the first non-NULL of il2/il3/ifd/ztt/mono/ilp). No call site uses it at
 * present. */
typedef struct
{
    struct vfft_plan_s *h;
    int oop;                    /* 1: vfft_execute(h, FWD, r0 -> rz) */
    vfft_il2p_plan_t *il2;      /* IL challenger: first non-NULL serves */
    vfft_il3p_plan_t *il3;
    vfft_ilprime_plan_t *ilp;
    vfft_ilfd_plan_t *ifd;      /* the flat DIT */
    vfft_ztt_plan_t *ztt;       /* ZTURN-T */
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
    if (c->il2)
        vfft_il2p_execute_fwd(c->il2, in, c->rz);
    else if (c->il3)
        vfft_il3p_execute_fwd(c->il3, in, c->rz);
    else if (c->ifd)
        vfft_ilfd_execute_fwd(c->ifd, in, c->rz);
    else if (c->ztt)
        vfft_ztt_execute_fwd(c->ztt, in, c->rz);   /* z -> z legal (plane mode) */
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

/* ── IN-PLACE INTERLEAVED c2c: the IL tier's own create ───────────────────
 * Split is not a fallback of IL: no split plan is built for an interleaved
 * caller. The cell is served by an IL engine — the K=1 engines (mono / pair /
 * chain3 / flat / ZTURN-T / four-step / prime, with their banked forms) — and
 * the verdict between them is the IN-PLACE CELL'S OWN raced verdict: the K=1
 * planner races every arm executed z -> z, exactly as this door runs it, and
 * banks the winner on the cell's kind-3 row keyed place=ip (its dir=bwd
 * sibling and ord=scr row alongside). The out-of-place cell's row is never
 * served here (one contract per request), and no mode row of the split
 * library is read. With one legal arm the race serves and banks it; with none
 * the create REFUSES — there is nothing to fall back to, by design.
 * Lane-major K>1 interleaved (only an explicit VFFT_BATCH_LANE_MAJOR reaches
 * here; DEFAULT geometry is the transform-contiguous wrapper) is refused: it
 * lost to transform-contiguous at every measured cell. */
/* DEFAULT = NATURAL: the classification is planning/policy.h's (L4). */
static inline int _ip_order_is_nat(const vfft_config_t *cfg, int N)
{
    return vfft_policy_ord_k1(cfg, N, /*inplace=*/1) == VW2_ORD_NAT;
}


static vfft_plan _c2c_ip_finish(struct vfft_plan_s *h,
                                struct vfft_wisdom_s *W,
                                const vfft_config_t *cfg, int N);

static vfft_plan _c2c_ip_create_il(const vfft_config_t *cfg,
                                   struct vfft_wisdom_s *W,
                                   const vfft_proto_registry_t *reg,
                                   int N, size_t K)
{
    const int nat = _ip_order_is_nat(cfg, N);   /* DEFAULT = NATURAL */
    struct vfft_plan_s *h;
    vfft_il2p_plan_t *il2 = NULL;
    vfft_il3p_plan_t *il3 = NULL;
    vfft_ilfd_plan_t *ifd = NULL;           /* the flat DIT */
    vfft_k1fs_plan_t *fs = NULL;            /* the four-step */
    vfft_ztt_plan_t *ztt = NULL;            /* ZTURN-T */
    vfft_oop11_fn mono_f = 0, mono_b = 0;   /* the alias-tolerant solo (MONO verdict) */
    vfft_ilprime_plan_t *ilp = NULL;
    int have_k1 = 0, mode = VFFT_NAT_UNSET;
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

    /* 1. the in-place cell's verdict is its OWN kind-3 row, place=ip, read or
     *    raced (executed in place) and banked by the K=1 candidate below; the
     *    split library's rows and the out-of-place cell's row are not read. */

    /* 2. the K=1 IL engine candidate: the planned row's route — MONO (the
     *    alias-tolerant solo), pair, chain3, flat, ZTURN-T, the four-step, or
     *    the prime cell (a raced arm; unraced only above the race ceiling) */
    if (!getenv("VFFT_NO_NAT_ILP"))
    {
        _k1_il_candidate(W, cfg, N, &il2, &il3, &ifd, &ztt, &fs, &ilp);   /* ilp: a PRIME verdict */
        if (ztt) vfft_ztt_bind(ztt, 1);   /* in place: the plane drivers */
        if (!il2 && !il3 && !ifd && !ztt && !fs && !ilp)
            (void)_k1_il_mono_candidate(W, cfg, N, &mono_f, &mono_b);
        if (!il2 && !il3 && !ifd && !ztt && !fs && !mono_f && !ilp && (N & (N - 1)) != 0)
            ilp = _ilprime_create_banked(W, cfg, N);   /* a route, never a fallback: the unraced cell above the ceiling */
        have_k1 = vfft_policy_k1_engine_present(
            /* mono   */ mono_f != 0,   /* in place the solo tier IS a resolved fn pair */
            /* pair   */ il2 != NULL, /* chain3 */ il3 != NULL,
            /* flat   */ ifd != NULL, /* ztt    */ ztt != NULL,
            /* fs     */ fs != NULL,  /* prime  */ ilp != NULL);
    }

    /* 3. the verdict IS the cell's own row: read above, or raced in place
     *    and banked by _k1_il_candidate / the prime cell */
    if (have_k1)
        mode = VFFT_NAT_ILP;

    /* 4. attach the verdict; the loser dies here */
    if (mode == VFFT_NAT_ILP && have_k1)
    {
        h->k1il2p = il2;
        h->k1il3p = il3;
        h->k1ilpr = ilp;
        h->k1ilfd = ifd;
        h->k1ztt = ztt;
        h->k1fs = fs;
        h->k1_mono_ilf = mono_f;
        h->k1_mono_ilb = mono_b;
        il2 = NULL; il3 = NULL; ilp = NULL; ifd = NULL; ztt = NULL; fs = NULL;
        h->nat_mode = nat ? VFFT_NAT_ILP : 0;
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[ipil] N=%d: %s ILP (%s)\n", N,
                    "attach",
                    h->k1il2p ? "il2p" : h->k1il3p ? "il3p" : h->k1ilfd ? "flat"
                              : h->k1ztt ? "ztt" : h->k1fs ? "fs" : h->k1_mono_ilf ? "mono" : "ilprime");
    }
    if (il2) vfft_il2p_destroy(il2);
    if (il3) vfft_il3p_destroy(il3);
    if (ilp) vfft_ilprime_destroy(ilp);
    if (ifd) vfft_ilfd_destroy(ifd);
    if (ztt) vfft_ztt_destroy(ztt);
    if (fs) vfft_k1fs_destroy(fs);
    if (!vfft_policy_k1_engine_present(
            /* mono   */ h->k1_mono_ilf != 0,
            /* pair   */ h->k1il2p != NULL, /* chain3 */ h->k1il3p != NULL,
            /* flat   */ h->k1ilfd != NULL, /* ztt    */ h->k1ztt != NULL,
            /* fs     */ h->k1fs != NULL,   /* prime  */ h->k1ilpr != NULL))
    {
        _vfft_warn("vfft_create: in-place C2C N=%d with layout=INTERLEAVED has no "
                   "interleaved engine yet (no mono/pair/chain3/prime kernel serves "
                   "this N%s) — the IL kernel set does not cover it; nothing to fall "
                   "back to by design",
                   N, "");
        free(h);
        return NULL;
    }
    return _c2c_ip_finish(h, W, cfg, N);
}

/* ── the tier's ONE exit. Every handle this create returns passes through
 * here, so a new early exit cannot skip the shared post-step (a skipped exit
 * ships mt_unsafe=0 — calloc's default, which spells "proven safe" — without
 * running the proof). The gate is cheap for ST creates (skipped) and engine
 * handles (no cplan: nothing K-splits). */
static vfft_plan _c2c_ip_finish(struct vfft_plan_s *h,
                                struct vfft_wisdom_s *W,
                                const vfft_config_t *cfg, int N)
{
    /* MT-safety: flag plans whose codelet ignores the partial-lane count (so
     * _c2c_mt runs them whole-batch instead of K-splitting). Checked once on
     * the FINAL cplan (after any natural rebuild). A safety net (the DIF/LOG3
     * K-split twiddle defect is fixed at codegen); only MT plans K-split, so
     * single-threaded creates skip the check and its cost. */
    if (h->cplan)
        h->mt_unsafe = (h->nthreads > 1) ? !_c2c_mt_safe(h->cplan, h->exec_fwd) : 0;
    if (h->k1ztt && h->K == 1 && h->nthreads > 1)
        _ztt_mt_replay_or_race(h, W, cfg, N);  /* ZTURN-T's threaded arm, in place: its own per-T pair */
    if (h->k1fs && h->K == 1 && h->nthreads > 1)
        _k1fs_mt_replay_or_race(h, W, cfg, N); /* the four-step's split at T, in place: its own per-T pair */
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
        return _c2c_ip_finish(h, W, cfg, N);
    }

    /* ── c2c IN-PLACE ── */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED)
        return _c2c_ip_create_il(cfg, W, reg, N, K);   /* the IL tier's own create */
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
         * injected chains) or SCR, raced on a miss and banked as the @nat verdict. Any
         * other order skips this block. */
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
            /* FREE needs no tape. ZCASC and ILP name interleaved engines,
             * which this split path does not build: they skip the tape build. */
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
        /* the MT-safety gate is in _c2c_ip_finish, the tier's one exit. */

        return _c2c_ip_finish(h, W, cfg, N);
    }
    return NULL; /* unreachable: the one call site guards on the same
                  * condition, and every path in the block above returns. */
}

#endif /* VFFT_OOP_C2C_IP_CREATE_H */
