/* fftnd_create.h — the rank-3 and rank-4 CREATE tiers (migration step 22).
 *
 * WHAT THIS IS
 * ------------
 * The dims==4 and dims==3 arms of _vfft_create_inner, as one helper. Both are
 * early-return blocks: every path inside them returns, so the tail of each is
 * unreachable and the pair lifts out without touching the rank-2 and rank-1
 * tiers that follow it in the dispatcher.
 *
 * CONTRACTS (unchanged by the move, restated because they are the tier's law)
 * -------------------------------------------------------------------------
 * K == 1. A batched rank>=3 call arrives as a K=1 override plan, not as a
 * howmany the engines see. Order is DEFAULT or SCRAMBLED only; rank-3 NATURAL
 * is the fftnd_natorder.h nat_col_list follow-up and is refused loudly here.
 * Real transforms are out-of-place. Trig (DCT/DST/DHT) is 1D only and is
 * refused above this helper, in the shared dims>=2 guard.
 *
 * WISDOM
 * ------
 * A dedicated (N1,N2,N3) table. HIT -> vfft_fft3d_plan_from_entry, the
 * fft3d.h-requested path. MISS -> greedy per-axis exhaustive with the inners
 * visible, banked through vw2_3d_bank_entry when the result is expressible.
 * The rank-4 arm shares that machinery at FFTND_MAX_RANK=4.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. It calls three file-scope statics that live in
 * vfft.c -- _vfft_plan_threads, _vw2_lay_of, _vw2_persist -- exactly as
 * il2d_tier.h, k1_commit.h and zr2c_build.h already do, so it must be included
 * after those are defined and before _vfft_create_inner. Its other callees
 * (stride_plan_nd, stride_plan_nd_r2c, the vfft_fft3d_* and vw2_3d_* wisdom
 * entry points) come from fftnd.h, fftnd_r2c.h and the wisdom2 readers, all
 * included far earlier.
 *
 * The four parameters are the block's complete free-variable set, derived
 * rather than guessed: cfg and reg and K and W are what the body reads from
 * the enclosing scope. N1/N2/N3 are NOT parameters -- they appear only as
 * struct field writes (h4->N2) or as locals declared inside the rank-3 arm.
 */
#ifndef VFFT_TRANSFORMS_FFTND_CREATE_H
#define VFFT_TRANSFORMS_FFTND_CREATE_H

/* Rank-3/rank-4 create. Returns the finished plan, or NULL after a loud
 * refusal (contract violation) or a quiet one (build/OOM failure).
 *
 * The trailing `return NULL` is unreachable for every call the dispatcher
 * makes: the one call site guards on dims being 3 or 4, and each arm returns
 * on every path. It exists so the function has a defined value on the path
 * the compiler must still see. */
static vfft_plan _vfft_create_rank34(const vfft_config_t *cfg,
                    struct vfft_wisdom_s *W,
                    const vfft_proto_registry_t *reg,
                    size_t K)
{
    /* 3D/4D INTERLEAVED (owner 2026-09-03): the IL feature-set for rank 3+
     * (c2c, r2c, c2r) does not exist yet and is a planned campaign. Until
     * then the front door REFUSES it loudly. Before this, c2c ACCEPTED the
     * layout and its execute computed nothing (a warning and zeros), and
     * real ran the split ND engine behind an il_out repack — a silent
     * convert. No fallback: refuse, never bridge. */
    if (cfg->layout == VFFT_LAYOUT_INTERLEAVED)
    {
        _vfft_warn("vfft_create: %dD %s with layout=INTERLEAVED is not wired yet "
                   "(the rank-3+ interleaved tier is a planned feature); use "
                   "VFFT_LAYOUT_SPLIT",
                   cfg->dims, _vfft_tname(cfg->transform));
        return NULL;
    }
    if (cfg->dims == 4)
    { /* §6a62: rank-4 exposure. The engines were rank-general all along
       * (FFTND_MAX_RANK=4; fndr's builder takes rank; fftnd's generic
       * wrap covers c2c) — the dispatch just stopped at 3. Same
       * contracts as 3D: K==1, order DEFAULT/SCRAMBLED, real = OOP with
       * even last dim. */
        if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
            !(K == 1 && (cfg->n[3] % 2) == 0))
        {
            _vfft_warn("vfft_create: 4D %s requires howmany==1 (got %zu) and an even last "
                       "dim (got %d)",
                       _vfft_tname(cfg->transform), K, cfg->n[3]);
            return NULL;
        }
        if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
            K == 1 && cfg->placement == VFFT_OUTOFPLACE &&
            (cfg->n[3] % 2) == 0)
        {
            stride_plan_t *tp = stride_plan_nd_r2c(4, cfg->n, reg);
            if (!tp)
                return NULL;
            struct vfft_plan_s *h4 = (struct vfft_plan_s *)calloc(1, sizeof *h4);
            if (!h4)
            {
                stride_plan_destroy(tp);
                return NULL;
            }
            h4->transform = cfg->transform;
            h4->placement = cfg->placement;
            h4->layout = (int)cfg->layout;
            h4->N = cfg->n[0];
            h4->N2 = cfg->n[1];
            h4->N3 = cfg->n[2];
            h4->N4 = cfg->n[3];
            h4->K = 1;
            h4->nthreads = _vfft_plan_threads(cfg);
            h4->tplan = tp;
            return h4;
        }
        if (cfg->transform != VFFT_C2C || K != 1 ||
            (cfg->order != VFFT_ORDER_DEFAULT && cfg->order != VFFT_ORDER_SCRAMBLED))
        {
            _vfft_warn("vfft_create: 4D supports C2C (howmany==1, order DEFAULT/SCRAMBLED) "
                       "and out-of-place R2C/C2R only (got %s, howmany=%zu, order=%d)",
                       _vfft_tname(cfg->transform), K, cfg->order);
            return NULL;
        }
        stride_plan_t *tp = stride_plan_nd(4, cfg->n, reg);
        if (!tp)
            return NULL;
        struct vfft_plan_s *h4 = (struct vfft_plan_s *)calloc(1, sizeof *h4);
        if (!h4)
        {
            stride_plan_destroy(tp);
            return NULL;
        }
        h4->transform = VFFT_C2C;
        h4->placement = cfg->placement;
        h4->layout = (int)cfg->layout;
        h4->N = cfg->n[0];
        h4->N2 = cfg->n[1];
        h4->N3 = cfg->n[2];
        h4->N4 = cfg->n[3];
        h4->K = 1;
        h4->nthreads = _vfft_plan_threads(cfg);
        h4->tplan = tp;
        return h4;
    }
    if (cfg->dims == 3)
    {
        if ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
            K == 1 && cfg->placement == VFFT_OUTOFPLACE &&
            (cfg->n[2] % 2) == 0)
        { /* §6a47/Q1: 3D real transforms via the ND r2c engine (strided
           * row engines + measured adoption live inside the builder). */
            stride_plan_t *tp = stride_plan_nd_r2c(3, cfg->n, reg);
            if (!tp)
                return NULL;
            struct vfft_plan_s *h3 = (struct vfft_plan_s *)calloc(1, sizeof *h3);
            if (!h3)
            {
                stride_plan_destroy(tp);
                return NULL;
            }
            h3->transform = cfg->transform;
            h3->placement = cfg->placement;
            h3->layout = (int)cfg->layout;
            h3->N = cfg->n[0];
            h3->N2 = cfg->n[1];
            h3->N3 = cfg->n[2];
            h3->K = 1;
            h3->nthreads = _vfft_plan_threads(cfg);
            h3->tplan = tp;
            return h3;
        }
        if (cfg->transform != VFFT_C2C || K != 1 ||
            (cfg->order != VFFT_ORDER_DEFAULT && cfg->order != VFFT_ORDER_SCRAMBLED))
        {
            _vfft_warn("vfft_create: 3D supports C2C (howmany==1, order DEFAULT/SCRAMBLED) and "
                       "out-of-place R2C/C2R with an even last dim only (got %s, howmany=%zu, "
                       "order=%d%s)",
                       _vfft_tname(cfg->transform), K, cfg->order,
                       (cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R)
                           ? (cfg->n[2] % 2 ? ", odd n[2]" : ", in-place?")
                           : "");
            return NULL;
        }
        int N1 = cfg->n[0], N2 = cfg->n[1], N3 = cfg->n[2];
        int banked = 0;
        stride_plan_t *tp = NULL;
        /* wave-3: 3D is BORN in wisdom2 (the legacy file never existed on
         * any tree). Serve from the store; on miss the legacy creator runs
         * its greedy+extract path against the in-process SCRATCH table and
         * the extraction is harvested into the store (measure-less
         * src=race — the extraction never measured; prime-axis cells bank
         * nothing, unchanged). No kill switch: nothing to fall back to. */
        {
            vfft_fft3d_wisdom_entry_t e3;
            if (vw2_3d_lookup(&W->vw2, N1, N2, N3, _vw2_lay_of(cfg), &e3))
                tp = vfft_fft3d_plan_from_entry(&e3, reg);
        }
        if (!tp)
        {
            tp = vfft_fft3d_plan_create_wisdom(N1, N2, N3, &W->fft3d_c2c, reg, &banked);
            if (banked)
            {
                const vfft_fft3d_wisdom_entry_t *ne =
                    vfft_fft3d_wisdom_lookup(&W->fft3d_c2c, N1, N2, N3);
                if (ne)
                    vw2_3d_bank_entry(&W->vw2, ne, VW2_LAY_ANY);
            }
        }
        if (!tp)
            return NULL;
        if (banked)
            _vw2_persist(W, cfg);
        struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
        if (!h)
        {
            stride_plan_destroy(tp);
            return NULL;
        }
        h->transform = VFFT_C2C;
        h->placement = cfg->placement;
        h->layout = (int)cfg->layout;
        h->N = N1;
        h->N2 = N2;
        h->N3 = N3;
        h->K = 1;
        h->nthreads = _vfft_plan_threads(cfg);
        h->tplan = tp;
        return h;
    }
    return NULL; /* unreachable: guarded on dims==3||dims==4 at the call site */
}

#endif /* VFFT_TRANSFORMS_FFTND_CREATE_H */
