/* fft2d_create_split.h — the rank-2 CREATE, split tier.
 *
 * The split 2D serving: _build_2d's stride plan (tiled rows + native
 * columns for c2c; the real plane against an N1 x (N2/2+1) spectrum for
 * r2c/c2r), the rfft / c2r-natural row inners measured-adopted, and the
 * ORDER_NATURAL axis reorder tapes. The front dispatcher
 * (split/split_create.h) is its only caller.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: calls file-scope statics of vfft.c
 * (_build_2d, _vw2_lay_of, _vw2_persist), so it is included (via the
 * dispatcher) after those are defined.
 */
#ifndef VFFT_SPLIT_FFT2D_CREATE_SPLIT_H
#define VFFT_SPLIT_FFT2D_CREATE_SPLIT_H

static vfft_plan _vfft_create_2d_split(const vfft_config_t *cfg,
                                      struct vfft_wisdom_s *W,
                                      const vfft_proto_registry_t *reg,
                                      size_t K)
{
    int N1 = cfg->n[0], N2 = cfg->n[1];
    stride_plan_t *tp = NULL;
    {
        tp = _build_2d(cfg->transform, N1, N2, cfg->rigor, reg, W, cfg->recalibrate,
                       cfg->order, _vw2_lay_of(cfg));
        /* _inner_c2c banks into the wisdom2 store; the guarded
         * _vw2_persist below covers disk. */
        if (!tp)
            return NULL;
        /* _build_2d banked into the wisdom2 store's memory; disk
         * persistence is the guarded save, and only after a SUCCESSFUL
         * create. */
        _vw2_persist(W, cfg);
    }
    struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        if (tp)
            stride_plan_destroy(tp);
        return NULL;
    }
    h->transform = cfg->transform;
    h->placement = cfg->placement;
    h->layout = (int)cfg->layout;
    h->N = N1;
    h->N2 = N2;
    h->K = K;
    h->nthreads = _vfft_plan_threads(cfg);
    h->tplan = tp;
    /* the IL column fields the shared commit always wrote (the plan
     * struct is shared until phase 8 of layout_separation_plan.md):
     * the same values, so a split handle is unchanged byte for byte */
    h->il2d_col.N = N1;
    h->il2d_col.rn = (cfg->transform == VFFT_C2C) ? (size_t)N2 : (size_t)N2 / 2 + 1;
    h->il2d_col.natst = 1;
    /* A/B race knob (struct comment): create-time env read only. */
    h->il2d_norowz = getenv("VFFT_IL2D_NO_ROWZ") != NULL;
    /* rfft-engine row inner for the R2C 2D row pass — the rfft path
     * wins at the tile's low K (−27%/call measured). Force the rfft
     * dispatch; adopt only if it landed (RFFT path, split, plan bound).
     * tp guard: the native IL real tier leaves tp NULL — split tier
     * only. */
    if (cfg->transform == VFFT_R2C && tp)
    {
        stride_fft2d_r2c_data_t *d2 = (stride_fft2d_r2c_data_t *)tp->override_data;
        size_t saved2 = vfft_r2c_dispatch_get_decouple_min_k();
        vfft_r2c_dispatch_set_decouple_min_k((size_t)-1);
        h->rfft_row = vfft_r2c_plan_create(N2, d2->B, VFFT_R2C_SPLIT,
                                           _rfft_registry(), NULL,
                                           (vfft_proto_registry_t *)reg);
        vfft_r2c_dispatch_set_decouple_min_k(saved2);
        if (h->rfft_row && h->rfft_row->path == VFFT_R2C_PATH_RFFT && h->rfft_row->layout == VFFT_R2C_SPLIT && h->rfft_row->rfft)
        {
            /* MEASURED adoption — "rfft wins at low K" does not survive
             * N-scaling (adopted unconditionally, (512,8) regresses
             * +66%). A/B both inners on tile scratch at create
             * (same-process, 64 reps each, sub-ms) and keep the winner. */
            double *sr0 = _fft2d_r2c_scratch_re(d2, 0);
            double *si0 = _fft2d_r2c_scratch_im(d2, 0);
            size_t tsz = d2->tile_real_sz;
            double *bak2 = (double *)malloc(tsz * sizeof(double));
            for (size_t ii = 0; ii < tsz; ii++)
                bak2[ii] = 1.0 + 1e-3 * (double)(ii & 63);
            rfft_plan_t *rp2 = h->rfft_row->rfft;
            double t0_, t1_;
            double t_str, t_rff;
            /* per-rep refill BOTH arms (unnormalized reps compound to
             * inf otherwise; equal handicap keeps the ratio honest). */
            memcpy(sr0, bak2, tsz * sizeof(double));
            _fft2d_r2c_inner_fwd(d2->plan_r2c, sr0, si0, 0); /* warm */
            t0_ = vfft_now_ns();
            for (int rr2 = 0; rr2 < 64; rr2++)
            {
                memcpy(sr0, bak2, tsz * sizeof(double));
                _fft2d_r2c_inner_fwd(d2->plan_r2c, sr0, si0, 0);
            }
            t1_ = vfft_now_ns();
            t_str = (t1_ - t0_);
            memcpy(sr0, bak2, tsz * sizeof(double));
            rfft_execute_fwd_natural(rp2, sr0, sr0, si0, NULL); /* warm */
            t0_ = vfft_now_ns();
            for (int rr2 = 0; rr2 < 64; rr2++)
            {
                memcpy(sr0, bak2, tsz * sizeof(double));
                rfft_execute_fwd_natural(rp2, sr0, sr0, si0, NULL);
            }
            t1_ = vfft_now_ns();
            t_rff = (t1_ - t0_);
            free(bak2);
            /* hysteresis — engine deltas measured <=3%, inside
             * regime-to-regime noise; create-time gates flipped winners
             * across weather regimes. The challenger must beat the
             * stride incumbent by >5% or the incumbent stays. */
            if (t_rff * 20 < t_str * 19)
                d2->rfft_row = rp2;
            else
            {
                vfft_r2c_plan_destroy(h->rfft_row);
                h->rfft_row = NULL;
            }
        }
        else if (h->rfft_row)
        {
            vfft_r2c_plan_destroy(h->rfft_row);
            h->rfft_row = NULL;
        }
    }
    /* bwd twin — c2r natural-engine row inner for the C2R 2D plan,
     * measured-adopted exactly like the fwd gate. Same tp guard: the
     * native IL real tier leaves tp NULL. */
    if (cfg->transform == VFFT_C2R && tp)
    {
        stride_fft2d_r2c_data_t *d2 = (stride_fft2d_r2c_data_t *)tp->override_data;
        h->c2r_row = vfft_c2r_disp_create(N2, d2->B, VFFT_C2R_NATURAL,
                                          _rfft_registry(),
                                          (vfft_proto_registry_t *)reg);
        if (h->c2r_row && h->c2r_row->packed && h->c2r_row->packed->nat_init)
        {
            double *sr0 = _fft2d_r2c_scratch_re(d2, 0);
            double *si0 = _fft2d_r2c_scratch_im(d2, 0);
            size_t tcz = d2->tile_complex_sz, trz = d2->tile_real_sz;
            double *bkr = (double *)malloc((tcz > trz ? tcz : trz) * sizeof(double));
            double *bki = (double *)malloc(tcz * sizeof(double));
            for (size_t ii = 0; ii < tcz; ii++)
            {
                bkr[ii] = 1.0 + 1e-3 * (double)(ii & 63);
                bki[ii] = 0.5 - 1e-3 * (double)(ii & 31);
            }
            c2r_plan_t *cp2 = h->c2r_row->packed;
            double t0_, t1_;
            double t_str, t_c2r;
            memcpy(sr0, bkr, tcz * sizeof(double));
            memcpy(si0, bki, tcz * sizeof(double));
            _fft2d_r2c_inner_bwd(d2->plan_r2c, sr0, si0, 0); /* warm */
            t0_ = vfft_now_ns();
            for (int rr2 = 0; rr2 < 64; rr2++)
            {
                memcpy(sr0, bkr, tcz * sizeof(double));
                memcpy(si0, bki, tcz * sizeof(double));
                _fft2d_r2c_inner_bwd(d2->plan_r2c, sr0, si0, 0);
            }
            t1_ = vfft_now_ns();
            t_str = (t1_ - t0_);
            memcpy(sr0, bkr, tcz * sizeof(double));
            memcpy(si0, bki, tcz * sizeof(double));
            c2r_execute_natural(cp2, sr0, si0, sr0, NULL); /* warm */
            t0_ = vfft_now_ns();
            for (int rr2 = 0; rr2 < 64; rr2++)
            {
                memcpy(sr0, bkr, tcz * sizeof(double));
                memcpy(si0, bki, tcz * sizeof(double));
                c2r_execute_natural(cp2, sr0, si0, sr0, NULL);
            }
            t1_ = vfft_now_ns();
            t_c2r = (t1_ - t0_);
            free(bkr);
            free(bki);
            if (t_c2r * 20 < t_str * 19) /* the >5% hysteresis */
                d2->c2r_row = cp2;
            else
            {
                vfft_c2r_disp_destroy(h->c2r_row);
                h->c2r_row = NULL;
            }
        }
        else if (h->c2r_row)
        {
            vfft_c2r_disp_destroy(h->c2r_row);
            h->c2r_row = NULL;
        }
    }
    /* ORDER_NATURAL (2D c2c): build the two per-axis digit-reversal reorder tapes from the inner
     * plans' chains. SCRAMBLED/DEFAULT leave nat2d==0 (byte-identical scrambled path). Refuse
     * (free + NULL) if orientation detect fails on either multi-stage axis — no silent wrong order. */
    if (cfg->transform == VFFT_C2C && cfg->order == VFFT_ORDER_NATURAL &&
        !h->il2d_row) /* always true here: the native IL tier is il/rank2 */
    {
        stride_fft2d_data_t *d = (stride_fft2d_data_t *)tp->override_data;
        int col_is_pairs = 0; /* dim2 runs cycle_pass in fft2d.h scratch -> never a pair tape */
        /* dim1 (whole-row): try PSWAP (involution) — the free latency win when the calibrated column
         * chain is palindromic (forcing a palindromic chain is a wash — its FFT slowdown offsets the
         * reorder win). dim2 (within-row): cycle only (fft2d.h scratch pass). */
        if (!d || !d->plan_col || !d->plan_row ||
            !vfft_natorder_2d_build_axis(N1, d->plan_col, &h->nat2d_row_list, &h->nat2d_row_is_pairs, 1) ||
            !vfft_natorder_2d_build_axis(N2, d->plan_row, &h->nat2d_col_list, &col_is_pairs, 0))
        {
            _vfft_warn("vfft_create: 2D %dx%d order=NATURAL — axis reorder-tape build "
                       "failed for this chain (orientation detect); the cell is "
                       "unsupported in natural order, use DEFAULT/SCRAMBLED",
                       N1, N2);
            vfft_destroy(h);
            return NULL;
        }
        /* dim1 MT bookkeeping: unit count + cycle start-offsets (for the per-worker range split),
         * mirroring the 1D natorder setup. NULL row tape = dim1 FREE (no reorder, no MT). */
        if (h->nat2d_row_list)
        {
            if (h->nat2d_row_is_pairs)
                h->nat2d_ncyc = vfft_natorder_pair_count(h->nat2d_row_list);
            else
            {
                h->nat2d_cyc_off = vfft_natorder_cycle_offsets(h->nat2d_row_list, &h->nat2d_ncyc);
                if (!h->nat2d_cyc_off)
                {
                    vfft_destroy(h);
                    return NULL;
                }
            }
        }
        /* h->nthreads slots of 2*N2 doubles: one dim1 cycle-scratch slot per worker (+ main).
         * Sized by the PLAN'S SNAPSHOT, not the live pool -- the pool is grow-only, so
         * the live count here can be smaller than the one _natorder_2d sees at execute;
         * that side clamps by the same h->nthreads (natorder_mt.h), so the slot count
         * and the slot index come from one number. */
        h->nat2d_tmp = (double *)malloc((size_t)(h->nthreads < 1 ? 1 : h->nthreads) * 2 * N2 * sizeof(double));
        if (!h->nat2d_tmp)
        {
            vfft_destroy(h);
            return NULL;
        }
        /* dim2 (within-row) is applied in the row-FFT scratch: borrow the col tape
         * into the fft2d data. h owns the malloc (freed in vfft_destroy); _fft2d_destroy must NOT
         * free it. dim1 stays a whole-row pass in _natorder_2d. */
        d->nat_col_list = h->nat2d_col_list;
        h->nat2d = 1;
    }
    return h;
}

#endif /* VFFT_SPLIT_FFT2D_CREATE_SPLIT_H */
