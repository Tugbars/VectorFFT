/* c2c_ip_create_il.h — the c2c IN-PLACE create, interleaved tier.
 *
 * Reached from the IL create (il/il_create.h) for an interleaved
 * in-place c2c request with no caller-supplied batch (IL + batch is refused
 * upstream). Depends on common/ and il/ only; never on the split tier.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: calls file-scope statics of vfft.c, so
 * it is included (via the dispatcher) after those are defined.
 */
#ifndef VFFT_IL_C2C_IP_CREATE_IL_H
#define VFFT_IL_C2C_IP_CREATE_IL_H

/* the IL in-place natural marker on the handle (h->nat_mode): a K=1 IL engine
 * serves the cell in natural order, no tape. Its verdict lives on the cell's
 * own kind-3 lay=il row. The value is the one the handle carried as the split
 * enum's VFFT_NAT_ILP before D2, kept so plan fingerprints read the same. */
#define VFFT_IL_NAT_K1 7


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


static vfft_plan _c2c_ip_finish_il(struct vfft_plan_s *h,
                                   struct vfft_wisdom_s *W,
                                   const vfft_config_t *cfg, int N);

static vfft_plan _c2c_ip_create_il(const vfft_config_t *cfg,
                                   struct vfft_wisdom_s *W,
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
    vfft_config_t rcfg;                     /* the recalibrating config of a re-raced cell */
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
        int stale = _k1_il_candidate(W, cfg, N, &il2, &il3, &ifd, &ztt, &fs, &ilp);   /* ilp: a PRIME verdict */
        if (!il2 && !il3 && !ifd && !ztt && !fs && !ilp &&
            _k1_il_mono_candidate(W, cfg, N, &mono_f, &mono_b) < 0)
            stale = 1;
        if (stale && !cfg->recalibrate)
        {   /* THE BANKED ROW IS EMPTY (owner, 2026-10-04): it did not build as
             * written, so the cell is raced from scratch -- one recalibrating
             * pass, whose winner replaces the row. The rest of this create
             * runs under that config. */
            if (il2) vfft_il2p_destroy(il2);
            if (il3) vfft_il3p_destroy(il3);
            if (ilp) vfft_ilprime_destroy(ilp);
            if (ifd) vfft_ilfd_destroy(ifd);
            if (ztt) vfft_ztt_destroy(ztt);
            if (fs) vfft_k1fs_destroy(fs);
            il2 = NULL; il3 = NULL; ilp = NULL; ifd = NULL; ztt = NULL; fs = NULL;
            mono_f = mono_b = 0;
            fprintf(stderr, "[wisdom2] c2c N=%d in place: the banked row does not build in this "
                            "library -- the cell is raced from scratch\n", N);
            rcfg = *cfg;
            rcfg.recalibrate = 1;
            cfg = &rcfg;
            stale = _k1_il_candidate(W, cfg, N, &il2, &il3, &ifd, &ztt, &fs, &ilp);
            if (!il2 && !il3 && !ifd && !ztt && !fs && !ilp &&
                _k1_il_mono_candidate(W, cfg, N, &mono_f, &mono_b) < 0)
                stale = 1;
            if (stale)
                fprintf(stderr, "[wisdom2] c2c N=%d in place: the row still does not build after "
                                "the race\n", N);
        }
        if (ztt) vfft_ztt_bind(ztt, 1);   /* in place: the plane drivers */
        if (!il2 && !il3 && !ifd && !ztt && !fs && !mono_f && !ilp && (N & (N - 1)) != 0)
            ilp = _ilprime_create_banked(W, cfg, N, NULL);   /* a route, never a fallback: the unraced cell above the ceiling */
        have_k1 = vfft_policy_k1_engine_present(
            /* mono   */ mono_f != 0,   /* in place the solo tier IS a resolved fn pair */
            /* pair   */ il2 != NULL, /* chain3 */ il3 != NULL,
            /* flat   */ ifd != NULL, /* ztt    */ ztt != NULL,
            /* fs     */ fs != NULL,  /* prime  */ ilp != NULL);
    }

    /* 3. the verdict IS the cell's own row: read above, or raced in place
     *    and banked by _k1_il_candidate / the prime cell */
    if (have_k1)
        mode = VFFT_IL_NAT_K1;

    /* 4. attach the verdict; the loser dies here */
    if (mode == VFFT_IL_NAT_K1 && have_k1)
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
        h->nat_mode = nat ? VFFT_IL_NAT_K1 : 0;
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
    return _c2c_ip_finish_il(h, W, cfg, N);
}

/* ── the IL tier's ONE exit. Every IL handle passes through here, so a new
 * early exit cannot skip the threaded replay/race of the K=1 engines. IL
 * handles carry no cplan, so the split tier's K-split proof has nothing to do
 * here. */
static vfft_plan _c2c_ip_finish_il(struct vfft_plan_s *h,
                                   struct vfft_wisdom_s *W,
                                   const vfft_config_t *cfg, int N)
{
    if (h->k1ztt && h->K == 1 && h->nthreads > 1)
        _ztt_mt_replay_or_race(h, W, cfg, N);  /* ZTURN-T's threaded arm, in place: its own per-T pair */
    if (h->k1ilpr && h->K == 1 && h->nthreads > 1)
        _ilpr_mt_replay_or_race(h, W, cfg, N); /* the prime cell's, in place: its own per-T row */
    if (h->k1fs && h->K == 1 && h->nthreads > 1)
        _k1fs_mt_replay_or_race(h, W, cfg, N); /* the four-step's split at T, in place: its own per-T pair */
    return h;
}

#endif /* VFFT_IL_C2C_IP_CREATE_IL_H */
