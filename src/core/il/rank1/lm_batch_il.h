/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* lm_batch_il.h — THE LANE-MAJOR 1D BATCH on the column pass
 * (docs/roadmap/lane_major_batch_design.md, 2026-10-14).
 *
 * K >= 2 transforms of length N in the lane-major geometry (config.batch_geom =
 * VFFT_BATCH_LANE_MAJOR; include/vfft.h): element e of transform t at
 * z[2 (e K + t)]. The plane is N rows of K complex at pitch K -- the 2D tier's
 * column plane with N1 = N, N2 = K and no row pass -- so the plan owns one
 * column-axis descriptor (il2d_col.h), built by the column builder at (N, K) in
 * THIS plan's role on THIS plan's own row (the 1D cell at q = K on the
 * lane-major layout axis, lay=lm: the transform-contiguous wrapper's q = K row
 * is another cell), and run by the column pass: the serial pass with its lane
 * tile (sw=), the threaded forms (strips over lanes, blocks over digits) under
 * the cell's raced verdict at the plan's T (cmt= mtarm= msw=). Everything the
 * 2D and 3D column axes have, the batch inherits: the chain race (here on the
 * door's serial forms: each chain at its lane tiles, in the cell's placement,
 * the winner's tile kept), the Rader stage with its inner and width race, the forms race, the
 * natural leaf redirection, the column Bluestein and the turned prime pass at
 * a prime N. Both placements (the pass runs in place on the plane), both order
 * classes (NATURAL by the leaf redirection; SCRAMBLED = the chain's comb per
 * lane, the column class's contract). c2c; the real batches are the next
 * pieces of the design.
 *
 * The geometries never race each other here: a lane-major request is served
 * lane-major or refused with its reason (the owner's rule, 2026-10-14); the
 * comparison of layouts is the planning model's max-performance mode.
 *
 * Included after the 2D tier (il2d_tier.h: the column builder, the passes, the
 * threaded forms, the races' scaffolding) and before il_create.h (the door). */
#ifndef VFFT_IL_LM_BATCH_IL_H
#define VFFT_IL_LM_BATCH_IL_H

/* the plan's column pass, one direction: the threaded forms under the cell's
 * verdict when the pool is live, the serial pass otherwise (the natural leaf
 * redirection, the Bluestein and turned axes and the banded walk through the
 * column-only serial execute; the plain chain pass with its lane tile) */
static void _lm_cols(struct vfft_plan_s *h, const vfft_real *src, vfft_real *dst, int reverse)
{
    vfft_ilcol_t *c = &h->il2d_col;   /* c->N is the chain's length: N for the c2c batch, N / 2 for the real one */
    if (h->nthreads > 1 && c->colmt &&
        _il2d_cols_mt_desc(c, c->N, src, dst, reverse, h->nthreads, c->nat ? c->natstage : NULL, -1))
    {
        _vfft_il2d_col_mt_count++;   /* the engagement (vfft.h): the caller's duty, as the 2D tiers count theirs */
        return;
    }
    if (c->wc > 0 && c->natsscr && c->nnatsscr >= 1 && (size_t)c->wc <= (size_t)c->natswcap &&
        !c->blu && !c->tpc && c->wl <= 0 && (!c->nat || c->nst >= 2))
    {   /* THE SERIAL DENSE TILES (2026-10-15): each tile of wc lanes through the dense scratch, the
         * natural strip (its leaf scatter included) or the scrambled strip -- the walk on the plane at
         * its pitch aliases in cache at a power-of-two K; the race picks the width (sw=) */
        const size_t K = c->rn, wc = (size_t)c->wc;
        vfft_il2p_fn const *fns = reverse ? c->b : c->f;
        vfft_real *const *tabs = reverse ? c->tb : c->tf;
        size_t k;
        for (k = 0; k < K; k += wc)
        {
            const size_t w = (k + wc < K) ? wc : K - k;
            if (c->nat)
                _il2d_col_pass_nat_strip(src, dst, c->N, K, k, w, c->nst, c->R, c->L, fns, tabs, reverse,
                                         c->natperm, c->natsscr[0]);
            else
                _il2d_col_pass_scr_strip(src, dst, c->N, K, k, w, c->nst, c->R, c->L, fns, tabs, reverse,
                                         c->natsscr[0]);
        }
        return;
    }
    if (c->nat || c->blu || c->wl > 0)
    {
        _il2d_col_exec_st(c, src, dst, reverse, c->natstage);
        return;
    }
    _il2d_col_pass(src, dst, c->N, c->rn, c->wc > 0 ? (size_t)c->wc : 0, c->nst, c->R, c->L,
                   reverse ? c->b : c->f, reverse ? c->tb : c->tf, reverse);
}
/* the bound executor (the lean door: it never declines) */
static int _lm_exec(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *zout)
{
    _lm_cols(h, zin, zout, dir == VFFT_BACKWARD);
    return 0;
}

/* ── THE CELL'S OWN VERDICTS BEYOND THE CHAIN: the serial lane tile and the
 * threaded form, on a plane of the plan's own placement, banked (sw= on the row;
 * cmt= / cmtt= through the chain bank, mtarm= / msw= beside them). The serial
 * tile comes with the chain when the chain race ran in this build (it timed
 * every candidate in the serial forms the door serves, il2d_tier.h
 * _il2d_race_chains; the descriptor's lmsw), from the row (sw=) when the chain
 * was replayed, and is raced here only for a replayed chain whose row has no
 * sw= or a pinned chain. At T > 1 the threaded verdict is raced here: the
 * serial form at that tile against the threaded walk, the strips over each
 * worker's whole range (c2c) and the strips at each sub-strip width; or
 * replayed from the row. Every arm runs the plan's own pass (_lm_cols, or the
 * real batch's executor) with the arm's knobs on the descriptor; a threaded arm
 * whose pass does not engage on this cell is out (the engagement counter
 * says). ── */
/* the lane-major batches' dense strip scratch (the descriptor's natsscr fields, freed with it): T blocks
 * (one at T = 1) of M rows of _il2d_scr_cap(swcap) complexes, swcap the widest ladder width at this T or a worker's whole
 * range where that fits L2, or the whole plane at one thread. The real batch's fused strips (serial tiles
 * and threaded sub-strips), the c2c batch's serial dense tiles (both orders) and its scrambled blocks run
 * in it; the natural chain's threaded blocks keep their own builder (its leaf). K is the lane count, size_t;
 * a width the L2 holds (vfft_policy_fits_l2_lanes) is far below INT_MAX, so swcap stays an int. */
static int _lm_sscr_build(vfft_ilcol_t *c, int M, size_t K, int T)
{
    int lad[6], nl, k, swcap = 0, t;
    const int Tb = T > 1 ? T : 1;
    if (c->natsscr || c->blu || c->tpc)
        return c->natsscr != NULL;
    nl = _il2d_sw_ladder(M, K, T, lad, 6);
    for (k = 0; k < nl; k++)
        if (lad[k] > swcap) swcap = lad[k];
    if (T > 1 && K / (size_t)T >= 8 && vfft_policy_fits_l2_lanes((size_t)M, K / (size_t)T + 8) &&
        K / (size_t)T + 8 > (size_t)swcap)
        swcap = (int)(K / (size_t)T + 8);   /* a worker's whole range, its bounds cut at 8 lanes */
    if (T <= 1 && K >= 2 && vfft_policy_fits_l2_lanes((size_t)M, K) && K > (size_t)swcap)
        swcap = (int)K;                      /* the whole plane as one tile */
    if (swcap < 2)
        return 0;
    c->natsscr = (vfft_real **)calloc((size_t)Tb, sizeof *c->natsscr);
    if (!c->natsscr)
        return 0;
    for (t = 0; t < Tb; t++)
    {
        c->natsscr[t] = (vfft_real *)vfft_aligned_alloc((2 * (size_t)M * _il2d_scr_cap((size_t)swcap) + 8) * sizeof(vfft_real));
        if (!c->natsscr[t])
        {
            c->nnatsscr = t;
            _il2d_nat_sscr_free(c);
            return 0;
        }
    }
    c->nnatsscr = Tb;
    c->natswcap = swcap;
    c->natdense = 1;
    return 1;
}
/* the scratch stays when a verdict runs in it: a real batch's fused strips (any threaded verdict, or a
 * serial tile), a c2c batch's blocks; freed otherwise */
static void _lm_sscr_settle(struct vfft_plan_s *h)
{
    vfft_ilcol_t *c = &h->il2d_col;
    const int real = h->transform != VFFT_C2C;
    const int keep = c->colmt ? (real || c->natarm == 1) : (real || c->wc > 0);
    if (!keep)
        _il2d_nat_sscr_free(c);
}
static int _lmr_exec(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *xin, vfft_real *zout);   /* below */
static int _lmc_exec(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *xout);
static int _lm2d_exec(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *zout);  /* rank2/lm_batch2d_il.h */
static int _lm2dr_exec_r2c(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *xin, vfft_real *zout);   /* rank2/lm_batch2dr_il.h */
static int _lm2dr_exec_c2r(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *xout);
typedef struct { struct vfft_plan_s *h; vfft_real *z, *zo; int colmt, natarm, msw, wc, ok; char name[24]; } _lm_arm_t;
static void _lm_arm_run(void *v)
{   /* a real batch's arm runs its executor (the fused strips with the arm's knobs, or the three-phase
     * form): the pack and the fold are part of what the knob decides; a c2c batch's arm is the pass */
    _lm_arm_t *a = (_lm_arm_t *)v;
    vfft_ilcol_t *c = &a->h->il2d_col;
    c->colmt = a->colmt;
    c->natarm = a->natarm;
    c->msw = a->msw;
    c->wc = a->wc;
    if (a->h->N2 > 0 && a->h->transform != VFFT_C2C)
    {   /* the 2D real batch: both axes, the blocks over the clones */
        if (a->h->transform == VFFT_R2C)
            _lm2dr_exec_r2c(a->h, VFFT_FORWARD, a->z, a->zo);
        else
            _lm2dr_exec_c2r(a->h, VFFT_BACKWARD, a->z, a->zo);
    }
    else if (a->h->transform == VFFT_R2C)
        _lmr_exec(a->h, VFFT_FORWARD, a->z, a->zo);
    else if (a->h->transform == VFFT_C2R)
        _lmc_exec(a->h, VFFT_BACKWARD, a->z, a->zo);
    else if (a->h->N2 > 0)
        _lm2d_exec(a->h, VFFT_FORWARD, a->z, a->zo);   /* the 2D batch: both axes, the rows over the clones */
    else
        _lm_cols(a->h, a->z, a->zo, 0);
}
/* Returns 0, or -1 OUT OF MEMORY (the race's planes could not be allocated; the reason recorded): the caller
 * refuses the plan */
static int _lm_race(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
                    const vw2_ilcol_key_t *key, int bcmt)
{
    vfft_ilcol_t *c = &h->il2d_col;
    /* the pass's rows and lanes: the descriptor's (a 2D batch's first axis runs N2 K lanes). In place the
     * arms run on one buffer re-laid from a seed: a c2c batch's pass, and a real batch of rank 2 or 3,
     * whose blocks go through the child's door (an in-place plan takes one buffer there); the 1D real
     * batch's executor is called directly and takes any pair, so it is timed on two */
    const int N = c->N, T = h->nthreads;
    const size_t K = c->rn;   /* the lanes the pass walks: every arm's plane and width is sized on them */
    const int ip = h->placement == VFFT_INPLACE && (h->transform == VFFT_C2C || h->N2 > 0);
    const int workers = T > 1 ? thread_pool_workers_for(T) : 1;
    const int usable = W && !W->vw2_off_2d && !cfg->recalibrate;
    const int c2c = h->transform == VFFT_C2C;
    /* the serial width race: the real batch's fused tiles at every chain; the c2c batch's dense tiles
     * under the serial-form law (policy_il.h: a Rader-stage chain walks the plane, no race) */
    const int plain = !c->blu && !c->tpc && c->wl <= 0 && (!c->nat || c->nst >= 2) &&
                      (!c2c || vfft_policy_lm_serial_tiles(c->rd0 != NULL));
    const int strips_ok = !c->blu && !c->tpc && (!c->nat || c->nst >= 2);
    /* the chain race of this build crowned the serial tile with the chain: kept, never raced again here */
    const int crowned = plain && c->lmsw >= 0;
    /* the arms' planes: the chain's M x K for a c2c batch; a real batch's arm runs its executor, whose
     * spectrum side is (M + 1) rows of K bins (2 K doubles more than the N K reals) */
    const size_t PN = 2 * (size_t)N * K + (h->transform != VFFT_C2C ? 2 * K : 0);
    _lm_arm_t ac[14];
    vfft_race_arm_t arms[14];
    double ns[14];
    vfft_real *z, *zo, *seed = NULL;
    _il2d_ip_reset_t rs;
    int lad[6], nl, na = 0, i, best = 0, bsw = -1, need_sw, need_mt;
    size_t q;
    c->wc = 0; c->colmt = 0; c->natarm = 0; c->msw = 0;
    bsw = crowned ? c->lmsw : (plain && usable) ? vw2_ilcol_tok_geti(&W->vw2, key, "sw", -1) : -1;
    need_sw = plain && bsw < 0;
    if (bsw >= 0)
        c->wc = bsw;
    /* a column through the turned prime pass runs serial, as the 2D door runs it (il2d_tier.h, _il2d_c2c_mt): the
     * threaded column pass would run the Bluestein columns over tables the turned pass dropped (a pool worker's
     * memset into freed scratch: 12005 x 32 and 4099 x 32 at 8 threads, 2026-11-01). No threaded arm, no threaded
     * replay. */
    need_mt = workers >= 2 && !c->tpc && !(usable && bcmt >= 0);
    if (workers >= 2 && !need_mt && !c->tpc)
    {   /* the threaded verdict from the row at this T */
        c->colmt = bcmt > 0;
        c->natarm = c->colmt ? vw2_ilcol_tok_geti(&W->vw2, key, "mtarm", 0) : 0;
        c->msw = c->colmt ? vw2_ilcol_tok_geti(&W->vw2, key, "msw", 0) : 0;
        if (c->colmt && c->natarm == 1)
            (void)(c->nat ? _il2d_nat_sscr_build(c, N, K, workers) : _lm_sscr_build(c, N, K, workers));
    }
    if (!need_sw && !need_mt)
    {
        _lm_sscr_settle(h);
        if (crowned && W && !W->vw2_off_2d)
        {   /* the chain race's tile onto the row (the chain bank made it) */
            if (vw2_ilcol_tok_seti(&W->vw2, key, "sw", c->wc) == VW2_OK)
                _vw2_persist(W, cfg);
        }
        if (crowned && getenv("VFFT_IL2D_LOG"))
            fprintf(stderr, "[lm] %d x %zu T=%d %s: the chain race's serial form sw=%d, no race here\n", N, K, T,
                    h->placement == VFFT_INPLACE ? "ip" : "oop", c->wc);
        return 0;
    }
    z = (vfft_real *)vfft_aligned_alloc((PN + 8) * sizeof(vfft_real));
    zo = ip ? z : (vfft_real *)vfft_aligned_alloc((PN + 8) * sizeof(vfft_real));
    if (ip) seed = (vfft_real *)vfft_aligned_alloc((PN + 8) * sizeof(vfft_real));
    if (!z || !zo || (ip && !seed))
    {   /* OUT OF MEMORY: the race's planes -- refused with that cause, never a plan served unmeasured */
        vfft_aligned_free(z); if (!ip) vfft_aligned_free(zo); vfft_aligned_free(seed);
        _vfft_warn("vfft_create: lane-major interleaved batch %d x %zu: out of memory (measuring its lane tile and "
                   "threads needs %.1f MB of planes)", N, K, 2.0 * (double)(PN + 8) * sizeof(vfft_real) / 1048576.0);
        return -1;
    }
    for (q = 0; q < PN + 8; q++)
        z[q] = 1.0 + 1e-6 * (double)(q & 1023);
    if (ip) { memcpy(seed, z, (PN + 8) * sizeof(vfft_real)); rs.p = z; rs.seed = seed; rs.n = PN + 8; }
#define _LM_ARM(NAME_, MT_, SHAPE_, SW_, WC_) do { \
        ac[na].h = h; ac[na].z = z; ac[na].zo = zo; ac[na].colmt = (MT_); ac[na].natarm = (SHAPE_); \
        ac[na].msw = (SW_); ac[na].wc = (WC_); ac[na].ok = 1; \
        snprintf(ac[na].name, sizeof ac[na].name, "%s", NAME_); \
        arms[na].name = ac[na].name; arms[na].run = _lm_arm_run; arms[na].ctx = &ac[na]; na++; } while (0)
    {   /* the serial lane tiles: the ladder's widths that divide K and keep a tile in L2. A c2c batch racing its
         * tile widths has no walk arm (the law), but where no width exists (fewer than eight lanes) the walk is the
         * serial arm: without it the race had threaded arms alone and banked one (100 x 4 on 8 threads: the
         * threaded walk, 6.7 times slower than the serial pass, 2026-11-01) */
        int wid[6], nw = 0;
        if (need_sw)
        {
            nl = _il2d_sw_ladder(N, K, 1, lad, 6);
            for (i = 0; i < nl; i++)
                if (!c2c || vfft_policy_lm_tile_ok(lad[i], K, c->nat))
                    wid[nw++] = lad[i];
        }
        if (!(c2c && need_sw && nw > 0))
            _LM_ARM("serial", 0, 0, 0, bsw >= 0 ? bsw : 0);
        for (i = 0; i < nw && na < 14; i++)
        {
            char nm[24];
            snprintf(nm, sizeof nm, "sw%d", wid[i]);
            _LM_ARM(nm, 0, 0, 0, wid[i]);
        }
    }
    if (need_mt)
    {
        _LM_ARM("threaded", 1, 0, 0, 0);
        if (strips_ok)
        {
            (void)(c->nat ? _il2d_nat_sscr_build(c, N, K, workers) : _lm_sscr_build(c, N, K, workers));
            if (h->transform == VFFT_C2C)   /* a real batch's whole-range strips are the threaded arm */
                _LM_ARM("strips", 1, 1, 0, 0);
            nl = _il2d_sw_ladder(N, K, T, lad, 6);
            for (i = 0; i < nl && na < 14; i++)
            {
                char nm[24];
                snprintf(nm, sizeof nm, "strips/sw%d", lad[i]);
                _LM_ARM(nm, 1, 1, lad[i], 0);
            }
        }
        for (i = 1; i < na; i++)
            if (ac[i].colmt)
            {   /* a threaded arm that does not engage on this cell is out */
                const long e0 = _vfft_il2d_col_mt_count;
                if (ip) memcpy(z, seed, (PN + 8) * sizeof(vfft_real));
                _lm_arm_run(&ac[i]);
                ac[i].ok = _vfft_il2d_col_mt_count != e0;
            }
    }
    if (na == 0)   /* no width for this K and nothing threaded: the walk, banked as sw = 0 */
        _LM_ARM("walk", 0, 0, 0, 0);
#undef _LM_ARM
    if (na > 1)
    {   /* the 1D batch race's rounds (_tc_mt_decide, vfft.c; the plane queue's since 2026-10-31): the minimum of 7
         * rounds, 9 in place, the order reversed every other round; a sample sized to ~0.3 ms of the first arm's
         * work (_il2d_race_reps), longer than the 1D race's 32768 points */
        const int rounds = ip ? 9 : 7;
        vfft_race_proto_t proto = need_mt
            ? (vfft_race_proto_t){ rounds, 1, VFFT_RACE_MIN, 1, 2, NULL, NULL, 0 }   /* threaded arms: two untimed passes first, never paused */
            : (vfft_race_proto_t){ rounds, 1, VFFT_RACE_MIN, 1, 0, NULL, NULL, 1 };  /* single-thread arms: paced */
        if (ip) { proto.reset = _il2d_ip_reset; proto.reset_ctx = &rs; }
        if (ip) memcpy(z, seed, (PN + 8) * sizeof(vfft_real));
        proto.reps = _il2d_race_reps(_lm_arm_run, &ac[0], ip ? (double)N : 0.0);
        _vfft_create_race_count++;
        vfft_race_run(&proto, arms, na, ns);
    }
    for (i = 1; i < na; i++)
        if (ac[i].ok && ns[i] < ns[best])
            best = i;
    c->wc = ac[best].colmt ? (bsw >= 0 ? bsw : 0) : ac[best].wc;
    if (ac[best].colmt && need_sw)
    {   /* the serial tile stays the best serial arm's (the threaded arm may fall back to it) */
        int bs = 0;
        for (i = 1; i < na; i++)
            if (!ac[i].colmt && ns[i] < ns[bs]) bs = i;
        c->wc = ac[bs].wc;
    }
    c->colmt = ac[best].colmt;
    c->natarm = ac[best].natarm;
    c->msw = ac[best].msw;
    _lm_sscr_settle(h);   /* the dense scratch stays for the verdicts that run in it */
    if (getenv("VFFT_IL2D_LOG"))
    {
        fprintf(stderr, "[lm] %d x %zu T=%d %s:", N, K, T, ip ? "ip" : "oop");
        for (i = 0; i < na; i++) fprintf(stderr, " %s=%.0f%s", ac[i].name, ns[i], ac[i].ok ? "" : "(x)");
        fprintf(stderr, " -> %s (sw=%d cmt=%d mtarm=%d msw=%d)\n", ac[best].name, c->wc, c->colmt, c->natarm, c->msw);
    }
    if (W && !W->vw2_off_2d)
    {   /* the verdicts on the row (the chain bank made it) */
        int changed = 0;
        if (need_sw || crowned) changed |= vw2_ilcol_tok_seti(&W->vw2, key, "sw", c->wc) == VW2_OK;
        if (need_mt)
        {
            changed |= vw2_ilcol_chain_bank(&W->vw2, key, c->R, c->nst, -1, -1, -1, c->colmt, T, c->blu, 0.0) >= 0;
            changed |= vw2_ilcol_tok_seti(&W->vw2, key, "mtarm", c->natarm) == VW2_OK;
            changed |= vw2_ilcol_tok_seti(&W->vw2, key, "msw", c->msw) == VW2_OK;
        }
        if (changed) _vw2_persist(W, cfg);
    }
    vfft_aligned_free(z); if (!ip) vfft_aligned_free(zo); vfft_aligned_free(seed);
    return 0;
}

/* ── THE DOOR: the lane-major 1D batch's create ─────────────────────────── */
static vfft_plan _vfft_create_lm_il(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, int N, size_t K)
{
    const int T = _vfft_plan_threads(cfg);
    const int ip = cfg->placement == VFFT_INPLACE;
    const int ord = vfft_policy_ord_k1(cfg, N, ip);
    const int nat = ord == VW2_ORD_NAT;
    vw2_ilcol_key_t ck;
    vfft_ilcol_t col;
    char forms[64];
    int bwl = -1, btf = -1, bro = -1, bcmt = -1, bcmtt = -1, bblu = -1, oom = 0;
    struct vfft_plan_s *h;
    if (!vfft_policy_batch_lane_major(cfg))
    {
        _vfft_warn("vfft_create: the lane-major interleaved batch serves 1D C2C at howmany >= 2 and N >= 2 "
                   "(got %s N=%d howmany=%zu) -- name batch=VFFT_BATCH_CONTIGUOUS for it",
                   _vfft_tname(cfg->transform), N, K);
        return NULL;
    }
    memset(&ck, 0, sizeof ck);
    ck.rank = 1; ck.n0 = N; ck.n1 = K; ck.ord = ord; ck.nthreads = T; ck.lm = 1;   /* the row at q = K, K whole */
    /* the Bluestein inner's chain provider for this create, as the 2D create sets it */
    _il2d_blu_ctx.W = W;
    _il2d_blu_ctx.cfg = cfg;
    _il2d_blu_ctx.N2 = K;
    _il2d_blu_chain_hook = _il2d_blu_m_chain;
    memset(&col, 0, sizeof col);
    forms[0] = 0;
    if (!_il2d_col_build(W, cfg, &ck, N, K, nat, &col, forms, sizeof forms,
                         &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu, &oom))
    {
        if (!oom)   /* out of memory: the builder's reason stands */
            _vfft_warn("vfft_create: lane-major interleaved C2C batch N=%d x K=%zu: no column pass serves N "
                       "(the column builder refused); unsupported", N, K);
        return NULL;
    }
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        _il2d_col_free(&col);
        return NULL;
    }
    h->transform = VFFT_C2C;
    h->placement = cfg->placement;
    h->layout = (int)VFI_LAYOUT_IL;
    h->N = N;
    h->N2 = 0;
    h->K = K;
    h->nthreads = T;
    h->lm = 1;
    h->il2d_col = col;
    h->il2d_col.N = N;
    h->il2d_col.rn = K;
    h->il2d_col.wl = 0;      /* the banded walk is the 2D plan's fused-row form; the batch runs unbanded */
    h->il2d_col.cut = 0;
    h->il2d_col.tfuse = 0;
    h->il2d_col.wc = 0;
    if (T <= 1 && !h->il2d_col.blu && !h->il2d_col.tpc)
        (void)_lm_sscr_build(&h->il2d_col, N, K, 1);   /* the serial dense tiles' scratch, raced below */
    if (_lm_race(h, W, cfg, &ck, bcmt))
    {   /* OUT OF MEMORY in the race (its reason recorded). Not retained yet (the front door retains the plan it
         * returns): its own release is skipped */
        h->nthreads = 1;
        vfft_destroy((vfft_plan)h);
        return NULL;
    }
    if (W && !W->vw2_off_2d && h->il2d_col.tpcS)
    {   /* the turned prime pass's child recipe onto the cell's row (the 2D create's last act;
         * without it the replay re-races the 1D prime cell and lands elsewhere) */
        vw2_key_t pk;
        vw2__ilcol_key(&ck, &pk);
        if (vfft_child_row_update(&W->vw2, &pk, "tpc_", h->il2d_col.tpcS))
            _vw2_persist(W, cfg);
    }
    (void)bcmtt; (void)bwl; (void)btf; (void)bro; (void)bblu;
    return (vfft_plan)h;
}

/* ── THE LANE-MAJOR REAL BATCH, r2c at an even N (the design's step 2) ─────
 * x[e K + t] real (N rows of K) -> z[2 (f K + t)], f < N / 2 + 1, three passes, the
 * real-axis form's own (il2d_real_axis.h) with its rows removed:
 *   PACK    W[m][k] = x[2 m][k] + i x[2 m + 1][k], m < M = N / 2 (a copy: the two reals
 *           of a packed sample sit K apart in lane-major data);
 *   COLUMNS the c2c column chain at M down every lane of W, in place, the scrambled
 *           class (the chain the column builder serves at (M, K), raced in this plan's
 *           role on this plan's own row; a prime M takes the column Bluestein);
 *   FOLD    per lane, the column spectrum read through the chain's permutation:
 *           A[0] = Re Z[0] + Im Z[0], A[M] = Re Z[0] - Im Z[0], and for k = 1..M/2
 *           E = (Z[k] + conj Z[M-k]) / 2, O = -i (Z[k] - conj Z[M-k]) / 2,
 *           A[k] = E + w^k O, A[M-k] = conj(E - w^k O), w = e^{-2 pi i / N}:
 *           the M + 1 rows of K bins, natural, straight into the caller's spectrum.
 * The chain threads through the column MT forms under the cell's verdict; the pack and
 * the fold split by lane ranges across the workers whenever that form serves. Both
 * placements: in place the buffer is (N + 2) K doubles, the N K reals first and the bins
 * over them, and nothing else changes, since the pack copies the whole input into W before
 * the fold writes a bin and the c2r unfold consumes every bin before the unpack writes a
 * real (each phase a whole dispatch). The c2r twin is its own piece. The three phases over the plane
 * are the form for a prime M (the Bluestein and turned axes) and for a width past the scratch; the
 * plain chain runs the FUSED STRIPS below (the dense strip arm with the pack and the fold around it). */
#if defined(__AVX2__)
/* A batch of two lanes: a block's two rows are four reals in a line, so the pack and the unpack are one
 * permute a row (a0 a1 b0 b1 <-> a0 b0 a1 b1). */
static void _lmr_pack2(const vfft_real *x, vfft_real *W, size_t Wp, size_t wk, size_t M)
{
    size_t m;
    for (m = 0; m < M; m++)
        _mm256_storeu_pd(W + 2 * (m * Wp + wk), _mm256_permute4x64_pd(_mm256_loadu_pd(x + 4 * m), 0xD8));
}
static void _lmc_unpack2(const vfft_real *W, size_t Wp, size_t wk, vfft_real *x, size_t M)
{
    size_t m;
    for (m = 0; m < M; m++)
        _mm256_storeu_pd(x + 4 * m, _mm256_permute4x64_pd(_mm256_loadu_pd(W + 2 * (m * Wp + wk)), 0xD8));
}
#endif
/* THE BLOCK KERNELS' AVX-512 STEPS (2026-11-16): each kernel takes whole 512-bit steps first -- eight lanes a
 * step in the pack and the unpack (eight reals of a row), four in the fold and the unfold (four complexes) --
 * then its AVX2 steps and its plain loop for the rest. A 512-bit step is the kernel's own AVX2 step twice as
 * wide, the same operations in the same order on every lane, so which lanes reach the plain loop does not
 * change and every number the AVX-512 build writes is bit for bit what it was. Where two NaNs meet in the
 * fold or the unfold, the NaN that comes out may carry another sign or payload than before (the compiler
 * orders an FMA's operands differently in the AVX2 remainder loop). The AVX2 build does not see these steps. */
static void _lmr_pack(const vfft_real *x, size_t K, vfft_real *W, size_t Wp, size_t wk, size_t M, size_t k0, size_t k1)
{
    size_t m, n;
#if defined(__AVX512F__)
    const __m512i i0 = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11), i1 = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
#endif
    for (m = 0; m < M; m++)
    {
        const vfft_real *a = x + 2 * m * K + k0, *b = a + K;
        vfft_real *w = W + 2 * (m * Wp + wk);
        const size_t w1 = k1 - k0;
        n = 0;
#if defined(__AVX512F__)
        for (; n + 8 <= w1; n += 8)
        {   /* a0..a7, b0..b7 -> a0 b0 a1 b1 a2 b2 a3 b3 | a4 b4 ... a7 b7 */
            const __m512d va = _mm512_loadu_pd(a + n), vb = _mm512_loadu_pd(b + n);
            _mm512_storeu_pd(w + 2 * n, _mm512_permutex2var_pd(va, i0, vb));
            _mm512_storeu_pd(w + 2 * n + 8, _mm512_permutex2var_pd(va, i1, vb));
        }
#endif
#if defined(__AVX2__)
        for (; n + 4 <= w1; n += 4)
        {
            const __m256d va = _mm256_loadu_pd(a + n), vb = _mm256_loadu_pd(b + n);
            const __m256d lo = _mm256_unpacklo_pd(va, vb), hi = _mm256_unpackhi_pd(va, vb);
            _mm256_storeu_pd(w + 2 * n, _mm256_permute2f128_pd(lo, hi, 0x20));
            _mm256_storeu_pd(w + 2 * n + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
        }
#endif
        for (; n < w1; n++)
        {
            w[2 * n] = a[n];
            w[2 * n + 1] = b[n];
        }
    }
}
static void _lmr_fold(const vfft_real *W, size_t Wp, size_t wk, vfft_real *A, size_t M, size_t K, const int *perm,
                      const vfft_real *tw, size_t k0, size_t k1)
{
    const vfft_real *Z0 = W + 2 * ((perm ? (size_t)perm[0] : 0) * Wp + wk);
    vfft_real *A0 = A, *AM = A + 2 * M * K;
    size_t k, n;
    for (n = k0; n < k1; n++)
    {
        const double r = Z0[2 * (n - k0)], i = Z0[2 * (n - k0) + 1];
        A0[2 * n] = r + i;
        A0[2 * n + 1] = 0.0;
        AM[2 * n] = r - i;
        AM[2 * n + 1] = 0.0;
    }
    for (k = 1; 2 * k <= M; k++)
    {
        const size_t rk = perm ? (size_t)perm[k] : k, rm = perm ? (size_t)perm[M - k] : M - k;
        const vfft_real *Pk = W + 2 * (rk * Wp + wk), *Pm = W + 2 * (rm * Wp + wk);
        vfft_real *Ak = A + 2 * k * K, *Am = A + 2 * (M - k) * K;
        const double wr = tw[2 * k], wi = tw[2 * k + 1];
        const int mirror = (2 * k < M);
        n = k0;
#if defined(__AVX512F__)
        {   /* the AVX2 step below, four complexes a step */
            const __m512d cj8 = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0), half8 = _mm512_set1_pd(0.5);
            const __m512d vwr8 = _mm512_set1_pd(wr), vwi8 = _mm512_setr_pd(-wi, wi, -wi, wi, -wi, wi, -wi, wi);
            for (; n + 4 <= k1; n += 4)
            {
                const __m512d P = _mm512_loadu_pd(Pk + 2 * (n - k0)), Q = _mm512_xor_pd(_mm512_loadu_pd(Pm + 2 * (n - k0)), cj8);
                const __m512d E = _mm512_mul_pd(half8, _mm512_add_pd(P, Q)), D = _mm512_mul_pd(half8, _mm512_sub_pd(P, Q));
                const __m512d O = _mm512_xor_pd(_mm512_permute_pd(D, 0x55), cj8);                              /* -i D */
                const __m512d T = _mm512_fmadd_pd(O, vwr8, _mm512_mul_pd(_mm512_permute_pd(O, 0x55), vwi8)); /* w^k O */
                _mm512_storeu_pd(Ak + 2 * n, _mm512_add_pd(E, T));
                if (mirror)
                    _mm512_storeu_pd(Am + 2 * n, _mm512_xor_pd(_mm512_sub_pd(E, T), cj8));
            }
        }
#endif
#if defined(__AVX2__)
        {
            const __m256d cj = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0), half = _mm256_set1_pd(0.5);
            const __m256d vwr = _mm256_set1_pd(wr), vwi = _mm256_setr_pd(-wi, wi, -wi, wi);
            for (; n + 2 <= k1; n += 2)
            {
                const __m256d P = _mm256_loadu_pd(Pk + 2 * (n - k0)), Q = _mm256_xor_pd(_mm256_loadu_pd(Pm + 2 * (n - k0)), cj);
                const __m256d E = _mm256_mul_pd(half, _mm256_add_pd(P, Q)), D = _mm256_mul_pd(half, _mm256_sub_pd(P, Q));
                const __m256d O = _mm256_xor_pd(_mm256_permute_pd(D, 0x5), cj);                          /* -i D */
                const __m256d T = _mm256_fmadd_pd(O, vwr, _mm256_mul_pd(_mm256_permute_pd(O, 0x5), vwi)); /* w^k O */
                _mm256_storeu_pd(Ak + 2 * n, _mm256_add_pd(E, T));
                if (mirror)
                    _mm256_storeu_pd(Am + 2 * n, _mm256_xor_pd(_mm256_sub_pd(E, T), cj));
            }
        }
#endif
        for (; n < k1; n++)
        {
            const double pr = Pk[2 * (n - k0)], pi = Pk[2 * (n - k0) + 1], qr = Pm[2 * (n - k0)], qi = -Pm[2 * (n - k0) + 1];
            const double er = 0.5 * (pr + qr), ei = 0.5 * (pi + qi), dr = 0.5 * (pr - qr), di = 0.5 * (pi - qi);
            const double orr = di, oi = -dr;   /* -i D */
            const double tr = orr * wr - oi * wi, ti = orr * wi + oi * wr;
            Ak[2 * n] = er + tr;
            Ak[2 * n + 1] = ei + ti;
            if (mirror)
            {
                Am[2 * n] = er - tr;
                Am[2 * n + 1] = -(ei - ti);
            }
        }
    }
}
/* ── THE C2R TWIN (the design's step 3): z[2 (f K + t)] (the N / 2 + 1 bins) -> x[e K + t],
 * unnormalized (N x, as every backward in the tree), the fold and the pack inverted:
 *   UNFOLD  per lane, the halves of the fold dropped (that factor 2 and the unnormalized
 *           M-point backward make the N): Z[0] = (A[0] + A[M]) + i (A[0] - A[M]), and for
 *           k = 1..M/2  E = A[k] + conj A[M-k], O = conj(w^k) (A[k] - conj A[M-k]),
 *           Z[k] = E + i O, Z[M-k] = conj(E - i O), written at the pass row the backward
 *           chain reads row k from (the chain's comb);
 *   COLUMNS the backward column chain at M down every lane of W, in place;
 *   UNPACK  x[2 m][k] = Re W[m][k], x[2 m + 1][k] = Im W[m][k]. */
static void _lmc_unfold(const vfft_real *A, size_t K, vfft_real *W, size_t Wp, size_t wk, size_t M, const int *perm,
                        const vfft_real *tw, size_t k0, size_t k1)
{
    vfft_real *Z0 = W + 2 * ((perm ? (size_t)perm[0] : 0) * Wp + wk);
    const vfft_real *A0 = A, *AM = A + 2 * M * K;
    size_t k, n;
    for (n = k0; n < k1; n++)
    {
        const double a0 = A0[2 * n], am = AM[2 * n];
        Z0[2 * (n - k0)] = a0 + am;
        Z0[2 * (n - k0) + 1] = a0 - am;
    }
    for (k = 1; 2 * k <= M; k++)
    {
        const size_t rk = perm ? (size_t)perm[k] : k, rm = perm ? (size_t)perm[M - k] : M - k;
        const vfft_real *Ak = A + 2 * k * K, *Am = A + 2 * (M - k) * K;
        vfft_real *Zk = W + 2 * (rk * Wp + wk), *Zm = W + 2 * (rm * Wp + wk);
        const double wr = tw[2 * k], wi = -tw[2 * k + 1];   /* conj(w^k) */
        const int mirror = (2 * k < M);
        n = k0;
#if defined(__AVX512F__)
        {   /* the AVX2 step below, four complexes a step */
            const __m512d cj8 = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
            const __m512d ci8 = _mm512_setr_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
            const __m512d vwr8 = _mm512_set1_pd(wr), vwi8 = _mm512_setr_pd(-wi, wi, -wi, wi, -wi, wi, -wi, wi);
            for (; n + 4 <= k1; n += 4)
            {
                const __m512d P = _mm512_loadu_pd(Ak + 2 * n), Q = _mm512_xor_pd(_mm512_loadu_pd(Am + 2 * n), cj8);
                const __m512d E = _mm512_add_pd(P, Q), D = _mm512_sub_pd(P, Q);
                const __m512d O = _mm512_fmadd_pd(D, vwr8, _mm512_mul_pd(_mm512_permute_pd(D, 0x55), vwi8));   /* conj(w^k) D */
                const __m512d iO = _mm512_xor_pd(_mm512_permute_pd(O, 0x55), ci8);                           /* i O */
                _mm512_storeu_pd(Zk + 2 * (n - k0), _mm512_add_pd(E, iO));
                if (mirror)
                    _mm512_storeu_pd(Zm + 2 * (n - k0), _mm512_xor_pd(_mm512_sub_pd(E, iO), cj8));
            }
        }
#endif
#if defined(__AVX2__)
        {
            const __m256d cj = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0);
            const __m256d vwr = _mm256_set1_pd(wr), vwi = _mm256_setr_pd(-wi, wi, -wi, wi);
            for (; n + 2 <= k1; n += 2)
            {
                const __m256d P = _mm256_loadu_pd(Ak + 2 * n), Q = _mm256_xor_pd(_mm256_loadu_pd(Am + 2 * n), cj);
                const __m256d E = _mm256_add_pd(P, Q), D = _mm256_sub_pd(P, Q);
                const __m256d O = _mm256_fmadd_pd(D, vwr, _mm256_mul_pd(_mm256_permute_pd(D, 0x5), vwi));   /* conj(w^k) D */
                const __m256d iO = _mm256_xor_pd(_mm256_permute_pd(O, 0x5), _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0)); /* i O */
                _mm256_storeu_pd(Zk + 2 * (n - k0), _mm256_add_pd(E, iO));
                if (mirror)
                    _mm256_storeu_pd(Zm + 2 * (n - k0), _mm256_xor_pd(_mm256_sub_pd(E, iO), cj));
            }
        }
#endif
        for (; n < k1; n++)
        {
            const double pr = Ak[2 * n], pi = Ak[2 * n + 1], qr = Am[2 * n], qi = -Am[2 * n + 1];
            const double er = pr + qr, ei = pi + qi, dr = pr - qr, di = pi - qi;
            const double orr = dr * wr - di * wi, oi = dr * wi + di * wr;   /* conj(w^k) D */
            const double ior = -oi, ioi = orr;                               /* i O */
            Zk[2 * (n - k0)] = er + ior;
            Zk[2 * (n - k0) + 1] = ei + ioi;
            if (mirror)
            {
                Zm[2 * (n - k0)] = er - ior;
                Zm[2 * (n - k0) + 1] = -(ei - ioi);
            }
        }
    }
}
static void _lmc_unpack(const vfft_real *W, size_t Wp, size_t wk, vfft_real *x, size_t K, size_t M, size_t k0, size_t k1)
{
    size_t m, n;
#if defined(__AVX512F__)
    const __m512i ir = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), ii = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
#endif
    for (m = 0; m < M; m++)
    {
        const vfft_real *w = W + 2 * (m * Wp + wk);
        vfft_real *a = x + 2 * m * K + k0, *b = a + K;
        const size_t w1 = k1 - k0;
        n = 0;
#if defined(__AVX512F__)
        for (; n + 8 <= w1; n += 8)
        {   /* re0 im0 .. re7 im7 -> re0..re7 | im0..im7 */
            const __m512d w0 = _mm512_loadu_pd(w + 2 * n), w1v = _mm512_loadu_pd(w + 2 * n + 8);
            _mm512_storeu_pd(a + n, _mm512_permutex2var_pd(w0, ir, w1v));
            _mm512_storeu_pd(b + n, _mm512_permutex2var_pd(w0, ii, w1v));
        }
#endif
#if defined(__AVX2__)
        for (; n + 4 <= w1; n += 4)
        {
            const __m256d w0 = _mm256_loadu_pd(w + 2 * n), w1v = _mm256_loadu_pd(w + 2 * n + 4);
            const __m256d lo = _mm256_permute2f128_pd(w0, w1v, 0x20), hi = _mm256_permute2f128_pd(w0, w1v, 0x31);
            _mm256_storeu_pd(a + n, _mm256_unpacklo_pd(lo, hi));
            _mm256_storeu_pd(b + n, _mm256_unpackhi_pd(lo, hi));
        }
#endif
        for (; n < w1; n++)
        {
            a[n] = w[2 * n];
            b[n] = w[2 * n + 1];
        }
    }
}
/* ── THE GROUP KERNELS (the 2D real batch's row pass, rank2/lm_batch2dr_il.h): the pack, the fold, the
 * unfold and the unpack over a GROUP of G row blocks at once. A block is K lanes wide; the blocks sit xs
 * (the reals) or zs (the bins) doubles apart; the group's packed plane is one dense scratch of M rows of
 * G K complexes, block g's lanes at [g K, g K + K). The row loop is outside and the blocks inside: a row's
 * twiddle and permutation are set once a group, and the chain between the kernels runs once over G K
 * lanes. A block's lanes go in whole-register steps (eight reals or four complexes at AVX-512, four or
 * two at AVX2), then a half register, then one lane: no scalar loop where the vectors exist. The
 * arithmetic is the block kernels' above, lane for lane; the fold carries its halves in the twiddle and in
 * the fused multiply-adds (a product by 0.5 is exact, so a lane's bins are those of the block kernel's
 * vector step bit for bit; a block kernel's odd lane goes through its plain loop and may differ in the
 * last bit). ── */
static void _lmr_pack_grp(const vfft_real *x, size_t xs, size_t G, size_t K, vfft_real *scr, size_t M)
{
    const size_t W = G * K;
    size_t m, g, n;
#if defined(__AVX512F__)
    const __m512i i0 = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11), i1 = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
#endif
#if defined(__AVX2__)
    if (K == 2)
    {   /* a block's two rows are four reals in a line, a0 a1 b0 b1 -> a0 b0 a1 b1: one permute */
        for (m = 0; m < M; m++)
        {
            const vfft_real *a = x + 4 * m;
            vfft_real *w = scr + 2 * m * W;
            for (g = 0; g < G; g++, a += xs, w += 4)
                _mm256_storeu_pd(w, _mm256_permute4x64_pd(_mm256_loadu_pd(a), 0xD8));
        }
        return;
    }
#endif
    for (m = 0; m < M; m++)
    {
        const vfft_real *a = x + 2 * m * K;
        vfft_real *w = scr + 2 * m * W;
        for (g = 0; g < G; g++, a += xs, w += 2 * K)
        {
            const vfft_real *b = a + K;
            n = 0;
#if defined(__AVX512F__)
            for (; n + 8 <= K; n += 8)
            {
                const __m512d va = _mm512_loadu_pd(a + n), vb = _mm512_loadu_pd(b + n);
                _mm512_storeu_pd(w + 2 * n, _mm512_permutex2var_pd(va, i0, vb));
                _mm512_storeu_pd(w + 2 * n + 8, _mm512_permutex2var_pd(va, i1, vb));
            }
#endif
#if defined(__AVX2__)
            for (; n + 4 <= K; n += 4)
            {
                const __m256d va = _mm256_loadu_pd(a + n), vb = _mm256_loadu_pd(b + n);
                const __m256d lo = _mm256_unpacklo_pd(va, vb), hi = _mm256_unpackhi_pd(va, vb);
                _mm256_storeu_pd(w + 2 * n, _mm256_permute2f128_pd(lo, hi, 0x20));
                _mm256_storeu_pd(w + 2 * n + 4, _mm256_permute2f128_pd(lo, hi, 0x31));
            }
            if (n + 2 <= K)
            {
                const __m128d va = _mm_loadu_pd(a + n), vb = _mm_loadu_pd(b + n);
                _mm_storeu_pd(w + 2 * n, _mm_unpacklo_pd(va, vb));
                _mm_storeu_pd(w + 2 * n + 2, _mm_unpackhi_pd(va, vb));
                n += 2;
            }
            if (n < K)
                _mm_storeu_pd(w + 2 * n, _mm_unpacklo_pd(_mm_load_sd(a + n), _mm_load_sd(b + n)));
#else
            for (; n < K; n++)
            {
                w[2 * n] = a[n];
                w[2 * n + 1] = b[n];
            }
#endif
        }
    }
}
static void _lmr_fold_grp(const vfft_real *scr, size_t G, size_t K, vfft_real *z, size_t zs, size_t M, const int *perm,
                          const vfft_real *tw)
{
    const size_t W = G * K;
    size_t k, g, n;
#if defined(__AVX512F__)
    const __m512d cj8 = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0), half8 = _mm512_set1_pd(0.5);
#endif
#if defined(__AVX2__)
    const __m256d cj = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0), half = _mm256_set1_pd(0.5);
    const __m256d re = _mm256_castsi256_pd(_mm256_setr_epi64x(-1, 0, -1, 0));
    const __m128d cj1 = _mm_setr_pd(0.0, -0.0), half1 = _mm_set1_pd(0.5), re1 = _mm_castsi128_pd(_mm_set_epi64x(0, -1));
#endif
    {   /* bins 0 and N / 2: Re Z[0] + Im Z[0] and Re Z[0] - Im Z[0], real */
        const vfft_real *Z0 = scr + 2 * (perm ? (size_t)perm[0] : 0) * W;
        vfft_real *A0 = z, *AM = z + 2 * M * K;
        for (g = 0; g < G; g++, Z0 += 2 * K, A0 += zs, AM += zs)
        {
            n = 0;
#if defined(__AVX512F__)
            for (; n + 4 <= K; n += 4)
            {
                const __m512d Z = _mm512_loadu_pd(Z0 + 2 * n), P = _mm512_permute_pd(Z, 0x55);
                _mm512_storeu_pd(A0 + 2 * n, _mm512_maskz_add_pd(0x55, Z, P));
                _mm512_storeu_pd(AM + 2 * n, _mm512_maskz_sub_pd(0x55, Z, P));
            }
#endif
#if defined(__AVX2__)
            for (; n + 2 <= K; n += 2)
            {
                const __m256d Z = _mm256_loadu_pd(Z0 + 2 * n), P = _mm256_permute_pd(Z, 0x5);
                _mm256_storeu_pd(A0 + 2 * n, _mm256_and_pd(_mm256_add_pd(Z, P), re));
                _mm256_storeu_pd(AM + 2 * n, _mm256_and_pd(_mm256_sub_pd(Z, P), re));
            }
            if (n < K)
            {
                const __m128d Z = _mm_loadu_pd(Z0 + 2 * n), P = _mm_permute_pd(Z, 1);
                _mm_storeu_pd(A0 + 2 * n, _mm_and_pd(_mm_add_pd(Z, P), re1));
                _mm_storeu_pd(AM + 2 * n, _mm_and_pd(_mm_sub_pd(Z, P), re1));
            }
#else
            for (; n < K; n++)
            {
                const double r = Z0[2 * n], i = Z0[2 * n + 1];
                A0[2 * n] = r + i;
                A0[2 * n + 1] = 0.0;
                AM[2 * n] = r - i;
                AM[2 * n + 1] = 0.0;
            }
#endif
        }
    }
    for (k = 1; 2 * k <= M; k++)
    {
        const size_t rk = perm ? (size_t)perm[k] : k, rm = perm ? (size_t)perm[M - k] : M - k;
        const vfft_real *Pk = scr + 2 * rk * W, *Pm = scr + 2 * rm * W;
        const int mirror = (2 * k < M);
        vfft_real *Ak = z + 2 * k * K, *Am = z + 2 * (M - k) * K;
#if defined(__AVX2__)
        const double hwr = 0.5 * tw[2 * k], hwi = 0.5 * tw[2 * k + 1];   /* w^k / 2 */
        const __m256d vwr = _mm256_setr_pd(hwr, -hwr, hwr, -hwr), vwi = _mm256_set1_pd(hwi);
        const __m128d vwr1 = _mm256_castpd256_pd128(vwr), vwi1 = _mm256_castpd256_pd128(vwi);
#if defined(__AVX512F__)
        const __m512d vwr8 = _mm512_setr_pd(hwr, -hwr, hwr, -hwr, hwr, -hwr, hwr, -hwr), vwi8 = _mm512_set1_pd(hwi);
#endif
#else
        const double wr = tw[2 * k], wi = tw[2 * k + 1];
#endif
        for (g = 0; g < G; g++, Pk += 2 * K, Pm += 2 * K, Ak += zs, Am += zs)
        {
            n = 0;
#if defined(__AVX512F__)
            for (; n + 4 <= K; n += 4)
            {
                const __m512d P = _mm512_loadu_pd(Pk + 2 * n), Q = _mm512_xor_pd(_mm512_loadu_pd(Pm + 2 * n), cj8);
                const __m512d S = _mm512_add_pd(P, Q), D = _mm512_sub_pd(P, Q);
                const __m512d T = _mm512_fmadd_pd(_mm512_permute_pd(D, 0x55), vwr8, _mm512_mul_pd(D, vwi8));
                _mm512_storeu_pd(Ak + 2 * n, _mm512_fmadd_pd(S, half8, T));
                if (mirror)
                    _mm512_storeu_pd(Am + 2 * n, _mm512_xor_pd(_mm512_fmsub_pd(S, half8, T), cj8));
            }
#endif
#if defined(__AVX2__)
            for (; n + 2 <= K; n += 2)
            {   /* S = P + Q, D = P - Q, T = w^k (-i D) / 2: A[k] = S / 2 + T, A[M-k] = conj(S / 2 - T) */
                const __m256d P = _mm256_loadu_pd(Pk + 2 * n), Q = _mm256_xor_pd(_mm256_loadu_pd(Pm + 2 * n), cj);
                const __m256d S = _mm256_add_pd(P, Q), D = _mm256_sub_pd(P, Q);
                const __m256d T = _mm256_fmadd_pd(_mm256_permute_pd(D, 0x5), vwr, _mm256_mul_pd(D, vwi));
                _mm256_storeu_pd(Ak + 2 * n, _mm256_fmadd_pd(S, half, T));
                if (mirror)
                    _mm256_storeu_pd(Am + 2 * n, _mm256_xor_pd(_mm256_fmsub_pd(S, half, T), cj));
            }
            if (n < K)
            {
                const __m128d P = _mm_loadu_pd(Pk + 2 * n), Q = _mm_xor_pd(_mm_loadu_pd(Pm + 2 * n), cj1);
                const __m128d S = _mm_add_pd(P, Q), D = _mm_sub_pd(P, Q);
                const __m128d T = _mm_fmadd_pd(_mm_permute_pd(D, 1), vwr1, _mm_mul_pd(D, vwi1));
                _mm_storeu_pd(Ak + 2 * n, _mm_fmadd_pd(S, half1, T));
                if (mirror)
                    _mm_storeu_pd(Am + 2 * n, _mm_xor_pd(_mm_fmsub_pd(S, half1, T), cj1));
            }
#else
            for (; n < K; n++)
            {
                const double pr = Pk[2 * n], pi = Pk[2 * n + 1], qr = Pm[2 * n], qi = -Pm[2 * n + 1];
                const double er = 0.5 * (pr + qr), ei = 0.5 * (pi + qi), dr = 0.5 * (pr - qr), di = 0.5 * (pi - qi);
                const double orr = di, oi = -dr;   /* -i D */
                const double tr = orr * wr - oi * wi, ti = orr * wi + oi * wr;
                Ak[2 * n] = er + tr;
                Ak[2 * n + 1] = ei + ti;
                if (mirror)
                {
                    Am[2 * n] = er - tr;
                    Am[2 * n + 1] = -(ei - ti);
                }
            }
#endif
        }
    }
}
/* the c2r twins: the bins of G blocks (zs apart) -> the scratch at the rows the backward chain reads, and the
 * scratch -> the reals of G blocks (xs apart) */
static void _lmc_unfold_grp(const vfft_real *z, size_t zs, size_t G, size_t K, vfft_real *scr, size_t M, const int *perm,
                            const vfft_real *tw)
{
    const size_t W = G * K;
    size_t k, g, n;
#if defined(__AVX512F__)
    const __m512d cj8 = _mm512_setr_pd(0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0);
    const __m512d ci8 = _mm512_setr_pd(-0.0, 0.0, -0.0, 0.0, -0.0, 0.0, -0.0, 0.0);
#endif
#if defined(__AVX2__)
    const __m256d cj = _mm256_setr_pd(0.0, -0.0, 0.0, -0.0), ci = _mm256_setr_pd(-0.0, 0.0, -0.0, 0.0);
    const __m128d cj1 = _mm_setr_pd(0.0, -0.0), ci1 = _mm_setr_pd(-0.0, 0.0);
#endif
    {   /* Z[0] = (A[0] + A[M]) + i (A[0] - A[M]), from the real parts */
        vfft_real *Z0 = scr + 2 * (perm ? (size_t)perm[0] : 0) * W;
        const vfft_real *A0 = z, *AM = z + 2 * M * K;
        for (g = 0; g < G; g++, Z0 += 2 * K, A0 += zs, AM += zs)
        {
            n = 0;
#if defined(__AVX512F__)
            for (; n + 4 <= K; n += 4)
                _mm512_storeu_pd(Z0 + 2 * n, _mm512_add_pd(_mm512_movedup_pd(_mm512_loadu_pd(A0 + 2 * n)),
                                                           _mm512_xor_pd(_mm512_movedup_pd(_mm512_loadu_pd(AM + 2 * n)), cj8)));
#endif
#if defined(__AVX2__)
            for (; n + 2 <= K; n += 2)
                _mm256_storeu_pd(Z0 + 2 * n, _mm256_add_pd(_mm256_movedup_pd(_mm256_loadu_pd(A0 + 2 * n)),
                                                           _mm256_xor_pd(_mm256_movedup_pd(_mm256_loadu_pd(AM + 2 * n)), cj)));
            if (n < K)
                _mm_storeu_pd(Z0 + 2 * n, _mm_add_pd(_mm_movedup_pd(_mm_loadu_pd(A0 + 2 * n)),
                                                     _mm_xor_pd(_mm_movedup_pd(_mm_loadu_pd(AM + 2 * n)), cj1)));
#else
            for (; n < K; n++)
            {
                const double a0 = A0[2 * n], am = AM[2 * n];
                Z0[2 * n] = a0 + am;
                Z0[2 * n + 1] = a0 - am;
            }
#endif
        }
    }
    for (k = 1; 2 * k <= M; k++)
    {
        const size_t rk = perm ? (size_t)perm[k] : k, rm = perm ? (size_t)perm[M - k] : M - k;
        const vfft_real *Ak = z + 2 * k * K, *Am = z + 2 * (M - k) * K;
        vfft_real *Zk = scr + 2 * rk * W, *Zm = scr + 2 * rm * W;
        const double wr = tw[2 * k], wi = -tw[2 * k + 1];   /* conj(w^k) */
        const int mirror = (2 * k < M);
#if defined(__AVX2__)
        const __m256d vwr = _mm256_set1_pd(wr), vwi = _mm256_setr_pd(-wi, wi, -wi, wi);
        const __m128d vwr1 = _mm256_castpd256_pd128(vwr), vwi1 = _mm256_castpd256_pd128(vwi);
#if defined(__AVX512F__)
        const __m512d vwr8 = _mm512_set1_pd(wr), vwi8 = _mm512_setr_pd(-wi, wi, -wi, wi, -wi, wi, -wi, wi);
#endif
#endif
        for (g = 0; g < G; g++, Ak += zs, Am += zs, Zk += 2 * K, Zm += 2 * K)
        {
            n = 0;
#if defined(__AVX512F__)
            for (; n + 4 <= K; n += 4)
            {
                const __m512d P = _mm512_loadu_pd(Ak + 2 * n), Q = _mm512_xor_pd(_mm512_loadu_pd(Am + 2 * n), cj8);
                const __m512d E = _mm512_add_pd(P, Q), D = _mm512_sub_pd(P, Q);
                const __m512d O = _mm512_fmadd_pd(D, vwr8, _mm512_mul_pd(_mm512_permute_pd(D, 0x55), vwi8));
                const __m512d iO = _mm512_xor_pd(_mm512_permute_pd(O, 0x55), ci8);
                _mm512_storeu_pd(Zk + 2 * n, _mm512_add_pd(E, iO));
                if (mirror)
                    _mm512_storeu_pd(Zm + 2 * n, _mm512_xor_pd(_mm512_sub_pd(E, iO), cj8));
            }
#endif
#if defined(__AVX2__)
            for (; n + 2 <= K; n += 2)
            {   /* E = A[k] + conj A[M-k], O = conj(w^k) (A[k] - conj A[M-k]): Z[k] = E + i O, Z[M-k] = conj(E - i O) */
                const __m256d P = _mm256_loadu_pd(Ak + 2 * n), Q = _mm256_xor_pd(_mm256_loadu_pd(Am + 2 * n), cj);
                const __m256d E = _mm256_add_pd(P, Q), D = _mm256_sub_pd(P, Q);
                const __m256d O = _mm256_fmadd_pd(D, vwr, _mm256_mul_pd(_mm256_permute_pd(D, 0x5), vwi));
                const __m256d iO = _mm256_xor_pd(_mm256_permute_pd(O, 0x5), ci);
                _mm256_storeu_pd(Zk + 2 * n, _mm256_add_pd(E, iO));
                if (mirror)
                    _mm256_storeu_pd(Zm + 2 * n, _mm256_xor_pd(_mm256_sub_pd(E, iO), cj));
            }
            if (n < K)
            {
                const __m128d P = _mm_loadu_pd(Ak + 2 * n), Q = _mm_xor_pd(_mm_loadu_pd(Am + 2 * n), cj1);
                const __m128d E = _mm_add_pd(P, Q), D = _mm_sub_pd(P, Q);
                const __m128d O = _mm_fmadd_pd(D, vwr1, _mm_mul_pd(_mm_permute_pd(D, 1), vwi1));
                const __m128d iO = _mm_xor_pd(_mm_permute_pd(O, 1), ci1);
                _mm_storeu_pd(Zk + 2 * n, _mm_add_pd(E, iO));
                if (mirror)
                    _mm_storeu_pd(Zm + 2 * n, _mm_xor_pd(_mm_sub_pd(E, iO), cj1));
            }
#else
            for (; n < K; n++)
            {
                const double pr = Ak[2 * n], pi = Ak[2 * n + 1], qr = Am[2 * n], qi = -Am[2 * n + 1];
                const double er = pr + qr, ei = pi + qi, dr = pr - qr, di = pi - qi;
                const double orr = dr * wr - di * wi, oi = dr * wi + di * wr;   /* conj(w^k) D */
                const double ior = -oi, ioi = orr;                               /* i O */
                Zk[2 * n] = er + ior;
                Zk[2 * n + 1] = ei + ioi;
                if (mirror)
                {
                    Zm[2 * n] = er - ior;
                    Zm[2 * n + 1] = -(ei - ioi);
                }
            }
#endif
        }
    }
}
static void _lmc_unpack_grp(const vfft_real *scr, size_t G, size_t K, vfft_real *x, size_t xs, size_t M)
{
    const size_t W = G * K;
    size_t m, g, n;
#if defined(__AVX512F__)
    const __m512i ir = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14), ii = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
#endif
#if defined(__AVX2__)
    if (K == 2)
    {   /* re0 im0 re1 im1 -> re0 re1 im0 im1, the block's two rows in a line: one permute */
        for (m = 0; m < M; m++)
        {
            const vfft_real *w = scr + 2 * m * W;
            vfft_real *a = x + 4 * m;
            for (g = 0; g < G; g++, a += xs, w += 4)
                _mm256_storeu_pd(a, _mm256_permute4x64_pd(_mm256_loadu_pd(w), 0xD8));
        }
        return;
    }
#endif
    for (m = 0; m < M; m++)
    {
        const vfft_real *w = scr + 2 * m * W;
        vfft_real *a = x + 2 * m * K;
        for (g = 0; g < G; g++, a += xs, w += 2 * K)
        {
            vfft_real *b = a + K;
            n = 0;
#if defined(__AVX512F__)
            for (; n + 8 <= K; n += 8)
            {
                const __m512d w0 = _mm512_loadu_pd(w + 2 * n), w1 = _mm512_loadu_pd(w + 2 * n + 8);
                _mm512_storeu_pd(a + n, _mm512_permutex2var_pd(w0, ir, w1));
                _mm512_storeu_pd(b + n, _mm512_permutex2var_pd(w0, ii, w1));
            }
#endif
#if defined(__AVX2__)
            for (; n + 4 <= K; n += 4)
            {
                const __m256d w0 = _mm256_loadu_pd(w + 2 * n), w1 = _mm256_loadu_pd(w + 2 * n + 4);
                const __m256d lo = _mm256_permute2f128_pd(w0, w1, 0x20), hi = _mm256_permute2f128_pd(w0, w1, 0x31);
                _mm256_storeu_pd(a + n, _mm256_unpacklo_pd(lo, hi));
                _mm256_storeu_pd(b + n, _mm256_unpackhi_pd(lo, hi));
            }
            if (n + 2 <= K)
            {
                const __m128d w0 = _mm_loadu_pd(w + 2 * n), w1 = _mm_loadu_pd(w + 2 * n + 2);
                _mm_storeu_pd(a + n, _mm_unpacklo_pd(w0, w1));
                _mm_storeu_pd(b + n, _mm_unpackhi_pd(w0, w1));
                n += 2;
            }
            if (n < K)
            {
                const __m128d v = _mm_loadu_pd(w + 2 * n);
                _mm_store_sd(a + n, v);
                _mm_storeh_pd(b + n, v);
            }
#else
            for (; n < K; n++)
            {
                a[n] = w[2 * n];
                b[n] = w[2 * n + 1];
            }
#endif
        }
    }
}
/* the lane-range dispatch of the elementwise passes across the workers: phase 0 pack, 1 fold
 * (r2c), 2 unfold, 3 unpack (c2r) */
typedef struct { struct vfft_plan_s *h; const vfft_real *x; vfft_real *z; size_t k0, k1; int phase; } _lmr_arg_t;
static void _lmr_tramp(void *v)
{
    _lmr_arg_t *a = (_lmr_arg_t *)v;
    struct vfft_plan_s *h = a->h;
    switch (a->phase)
    {
    case 0: _lmr_pack(a->x, h->K, h->lm_W, h->K, a->k0, (size_t)h->lm_M, a->k0, a->k1); break;
    case 1: _lmr_fold(h->lm_W, h->K, a->k0, a->z, (size_t)h->lm_M, h->K, h->lm_perm, h->lm_tw, a->k0, a->k1); break;
    case 2: _lmc_unfold(a->x, h->K, h->lm_W, h->K, a->k0, (size_t)h->lm_M, h->lm_perm, h->lm_tw, a->k0, a->k1); break;
    default: _lmc_unpack(h->lm_W, h->K, a->k0, a->z, h->K, (size_t)h->lm_M, a->k0, a->k1); break;
    }
}
static void _lmr_ranges(struct vfft_plan_s *h, const vfft_real *x, vfft_real *z, int phase)
{
    const int T = (h->nthreads > 1 && h->il2d_col.colmt) ? thread_pool_workers_for(h->nthreads) : 1;
    const size_t K = h->K;
    _lmr_arg_t a[THREAD_POOL_MAX_DISPATCH];
    int t, Ts = T;
    if (Ts > 1 && K < (size_t)(2 * Ts)) Ts = (int)(K / 2) > 1 ? (int)(K / 2) : 1;   /* at least two lanes a worker (a vector) */
    if (Ts <= 1)
    {
        a[0].h = h; a[0].x = x; a[0].z = z; a[0].k0 = 0; a[0].k1 = K; a[0].phase = phase;
        _lmr_tramp(&a[0]);
        return;
    }
    for (t = 0; t < Ts; t++)
    {
        a[t].h = h; a[t].x = x; a[t].z = z; a[t].phase = phase;
        a[t].k0 = (K * (size_t)t / (size_t)Ts) & ~(size_t)1;         /* even lane bounds: whole vectors */
        a[t].k1 = t + 1 == Ts ? K : (K * (size_t)(t + 1) / (size_t)Ts) & ~(size_t)1;
    }
    thread_pool_run(Ts, _lmr_tramp, a, sizeof a[0]);
}
/* ── THE FUSED STRIPS (the dense strip arm with the pack and the fold around it, 2026-10-15): a
 * strip of w lanes [k, k + w) goes through a dense scratch of M x w complexes (pitch w, in L2): the
 * pack from the caller's plane into the scratch, the chain in place in the scratch (the plain walker
 * at pitch w: the same kernels, tables and values, so the output is bitwise the plane form's), the
 * fold from the scratch into the caller's spectrum; c2r the mirror. The plane W is never walked at
 * its pitch (at a power-of-two K every row sits a power of two apart, so a sub-strip's rows share
 * one cache set group and every stage streams), the plane is read once and written once, and one
 * dispatch serves the execute. Threaded: each worker's lane range in sub-strips of msw (0 = the
 * whole range), the ranges cut at 8 lanes so no cache line is written by two workers; serial: tiles
 * of wc. The plain chain class with a scratch wide enough (natswcap); a Bluestein or turned axis at
 * a prime M works over the plane, and the three-phase form below serves those and any width past
 * the scratch. IN PLACE the bins of lane k sit where the reals of lanes 2 k and 2 k + 1 sit (two
 * doubles a bin, one a real), so a strip's fold would overwrite reals of strips not yet packed: the
 * input is consumed whole first, the pack (the unfold) over the plane W as the three-phase form does
 * it, and each strip then reads its lanes from W; one more pass over the plane than out of place. */
static void _lmr_strip(struct vfft_plan_s *h, const vfft_real *x, vfft_real *z, size_t k, size_t w, vfft_real *scr, int c2r,
                       int from_W)
{
    const vfft_ilcol_t *c = &h->il2d_col;
    const size_t M = (size_t)h->lm_M, K = h->K;
    size_t m;
    if (from_W)
    {   /* in place: the plane W holds the packed (unfolded) input, consumed whole before any output row */
        for (m = 0; m < M; m++)
            memcpy(scr + 2 * m * w, h->lm_W + 2 * (m * K + k), 2 * w * sizeof(vfft_real));
    }
    else if (!c2r)
    {
#if defined(__AVX2__)
        if (K == 2 && w == 2)
            _lmr_pack2(x, scr, w, 0, M);   /* the whole batch of two lanes: one permute a row */
        else
#endif
            _lmr_pack(x, K, scr, w, 0, M, k, k + w);
    }
    else
        _lmc_unfold(x, K, scr, w, 0, M, h->lm_perm, h->lm_tw, k, k + w);
    if (!c2r)
    {
        _il2d_col_pass_range(scr, scr, (int)M, w, 0, w, c->nst, c->R, c->L, c->f, c->tf, 0);
        _lmr_fold(scr, w, 0, z, M, K, h->lm_perm, h->lm_tw, k, k + w);
    }
    else
    {
        _il2d_col_pass_range(scr, scr, (int)M, w, 0, w, c->nst, c->R, c->L, c->b, c->tb, 1);
#if defined(__AVX2__)
        if (K == 2 && w == 2)
            _lmc_unpack2(scr, w, 0, z, M);
        else
#endif
            _lmc_unpack(scr, w, 0, z, K, M, k, k + w);
    }
}
typedef struct { struct vfft_plan_s *h; const vfft_real *x; vfft_real *z; size_t lo, hi, sw; int tid, c2r, from_W; } _lmf_arg_t;
static void _lmf_tramp(void *v)
{
    _lmf_arg_t *a = (_lmf_arg_t *)v;
    vfft_real *scr = a->h->il2d_col.natsscr[a->tid];
    size_t k;
    for (k = a->lo; k < a->hi; k += a->sw)
        _lmr_strip(a->h, a->x, a->z, k, (k + a->sw < a->hi) ? a->sw : a->hi - k, scr, a->c2r, a->from_W);
}
/* 1 = the fused strips served (a threaded run counted as an engagement), 0 = the three-phase form's turn */
static int _lmr_fused(struct vfft_plan_s *h, const vfft_real *x, vfft_real *z, int c2r)
{
    vfft_ilcol_t *c = &h->il2d_col;
    const size_t K = h->K, cap = (size_t)c->natswcap;
    const int ip = h->placement == VFFT_INPLACE;   /* the input consumed whole into W first (below) */
    _lmf_arg_t a[THREAD_POOL_MAX_DISPATCH];
    int t, T;
    if (!c->natsscr || c->nnatsscr < 1 || c->nat || c->blu || c->tpc || c->wl > 0)
        return 0;
    if (h->nthreads > 1 && c->colmt)
    {
        const size_t msw = c->msw > 0 ? (size_t)c->msw : 0;
        int ok = msw <= cap;
        T = thread_pool_workers_for(h->nthreads);
        if (T > c->nnatsscr) T = c->nnatsscr;
        if (K / 8 < (size_t)T) T = (int)(K / 8);   /* eight lanes a worker at least: whole cache lines */
        if (T < 2)
            return 0;
        for (t = 0; t < T; t++)
        {
            a[t].h = h; a[t].x = x; a[t].z = z; a[t].tid = t; a[t].c2r = c2r; a[t].from_W = ip;
            a[t].lo = (K * (size_t)t / (size_t)T) & ~(size_t)7;
            a[t].hi = t + 1 == T ? K : (K * (size_t)(t + 1) / (size_t)T) & ~(size_t)7;
            a[t].sw = msw ? msw : a[t].hi - a[t].lo;
            if (a[t].sw > cap || a[t].hi <= a[t].lo) ok = 0;
        }
        if (!ok)
            return 0;
        if (ip)
            _lmr_ranges(h, x, z, c2r ? 2 : 0);   /* the pack (the unfold) over the whole plane into W */
        thread_pool_run(T, _lmf_tramp, a, sizeof a[0]);
        _vfft_il2d_col_mt_count++;   /* the engagement (vfft_diagnostics.h): the caller's duty */
        return 1;
    }
    {   /* serial: tiles of wc; wc = 0 is the whole plane as one tile where the scratch holds it (the
         * three-pass form's work with no plane W: its one strip packs everything before any bin is
         * written, so in place needs no first pass); past the scratch the three-pass form serves */
        const size_t wc = c->wc > 0 ? (size_t)c->wc : (K <= cap ? K : 0);
        if (wc == 0 || wc > cap)
            return 0;
        if (ip && wc < K)
            _lmr_ranges(h, x, z, c2r ? 2 : 0);
        a[0].h = h; a[0].x = x; a[0].z = z; a[0].lo = 0; a[0].hi = K; a[0].sw = wc; a[0].tid = 0; a[0].c2r = c2r;
        a[0].from_W = ip && wc < K;
        _lmf_tramp(&a[0]);
        return 1;
    }
}
static int _lmr_exec(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *xin, vfft_real *zout)
{
    (void)dir;   /* an r2c plan is forward-only: the door refuses the other direction before this */
    if (_lmr_fused(h, xin, zout, 0))
        return 0;
    _lmr_ranges(h, xin, zout, 0);
    _lm_cols(h, h->lm_W, h->lm_W, 0);
    _lmr_ranges(h, xin, zout, 1);
    return 0;
}
static int _lmc_exec(struct vfft_plan_s *h, vfft_dir_t dir, const vfft_real *zin, vfft_real *xout)
{
    (void)dir;   /* a c2r plan is backward-only */
    if (_lmr_fused(h, zin, xout, 1))
        return 0;
    _lmr_ranges(h, zin, xout, 2);
    _lm_cols(h, h->lm_W, h->lm_W, 1);
    _lmr_ranges(h, zin, xout, 3);
    return 0;
}
/* the real batch's fold order for a chain at M: the inverse of the scrambled pass's row order (the fold reads row k
 * at the pass row that holds it); NULL with *ok = 1 for a one-stage chain (natural), *ok = 0 where it does not derive */
static int *_lmr_perm_of(const int *R, int nst, int M, int *ok)
{
    int *np, *p, k;
    *ok = 1;
    if (nst < 2)
        return NULL;
    np = _il2d_nat_perm(R, nst, M);
    p = np ? (int *)malloc((size_t)M * sizeof(int)) : NULL;
    if (!p)
    {
        free(np);
        *ok = 0;
        return NULL;
    }
    for (k = 0; k < M; k++)
        p[np[k]] = k;
    free(np);
    return p;
}
/* the fold's twiddles w^k = e^{-2 pi i k / N}, k = 0..M, once-rounded (common/math/tw_exact.h) */
static void _lmr_tw_fill(vfft_real *tw, int M, int N)
{
    int k;
    for (k = 0; k <= M; k++)
    {
        double c, s;
        vfft_cs2pi_exact((long long)k, (long long)N, &c, &s);
        tw[2 * k] = c;
        tw[2 * k + 1] = -s;
    }
}

/* ── THE CHAIN RACE'S LANE-MAJOR ARMS (il2d_tier.h, _il2d_race_chains / _il2d_race_heat with an _il2d_lmform_t).
 * A candidate is timed in each serial form its door serves it with, through the door's own pass on a plan that
 * carries the candidate's view (its kernels, tables and natural leaf; for the 1D real batch its fold order, the
 * plane and the twiddles), the arm setting the lane tile as _lm_race's serial arms set it: the c2c pass and the
 * first axis of a real 2D / 3D batch through _lm_cols, the 1D real batch through its executor. The dense scratch is
 * built by the builder the door uses at the cell's T, so the cap that decides between a tile and the form past it
 * is the door's. ── */
static void _lm_sscr_keep_one(vfft_ilcol_t *c)
{   /* the arms run serial: the first block is the one they use */
    int t;
    for (t = 1; t < c->nnatsscr; t++)
    {
        vfft_aligned_free(c->natsscr[t]);
        c->natsscr[t] = NULL;
    }
    if (c->nnatsscr > 1)
        c->nnatsscr = 1;
}
static int _lm_chain_open(_il2d_lmform_t *f, int N1, size_t rn, int nat_req)
{
    const size_t K = rn;
    memset(&f->sscr, 0, sizeof f->sscr);
    memset(&f->snat, 0, sizeof f->snat);
    f->W = f->tw = NULL;
    f->wc = 0;
    if (f->kind == 1)
    {   /* the 1D real batch's create: its scratch at T, the three-phase form's plane, the fold's twiddles */
        (void)_lm_sscr_build(&f->sscr, N1, K, f->T > 1 ? f->T : 1);
        f->W = (vfft_real *)vfft_aligned_alloc((2 * (size_t)N1 * rn + 8) * sizeof(vfft_real));
        f->tw = (vfft_real *)vfft_aligned_alloc(2 * ((size_t)N1 + 1) * sizeof(vfft_real));
        if (!f->W || !f->tw)
        {
            _lm_chain_close(f);
            return 0;
        }
        _lmr_tw_fill(f->tw, N1, 2 * N1);
    }
    else if (f->T <= 1)
        (void)_lm_sscr_build(&f->sscr, N1, K, 1);   /* the doors' at one thread, every class */
    else
    {   /* _lm_race's, built for its threaded arms (the serial arms run in it): over the workers, a natural chain
         * through its own builder */
        const int workers = thread_pool_workers_for(f->T);
        if (workers >= 2)
        {
            (void)_lm_sscr_build(&f->sscr, N1, K, workers);
            if (nat_req)
            {
                f->snat.nat = 1;
                (void)_il2d_nat_sscr_build(&f->snat, N1, K, workers);
            }
        }
    }
    _lm_sscr_keep_one(&f->sscr);
    _lm_sscr_keep_one(&f->snat);
    return 1;
}
static void _lm_chain_close(_il2d_lmform_t *f)
{
    _il2d_nat_sscr_free(&f->sscr);
    _il2d_nat_sscr_free(&f->snat);
    vfft_aligned_free(f->W);
    vfft_aligned_free(f->tw);
    f->W = f->tw = NULL;
}
/* the serial forms the door serves a chain with -- the widths _lm_race's serial race offered: the c2c pass under the
 * serial-form law (policy_il.h: a Rader-stage chain walks; a smooth chain runs the serial ladder's tiles the law
 * admits, the walk where there is none); the real batches, not under the law, the whole plane or the walk (0) and
 * every width of the serial ladder. Returns the count, at most 7. */
static int _lm_chain_forms(const _il2d_lmform_t *f, int N1, size_t rn, const int *R, int nat, int *wcs)
{
    int lad[6], nl, i, n = 0;
    nl = _il2d_sw_ladder(N1, rn, 1, lad, 6);
    if (f->kind != 0)
    {
        wcs[n++] = 0;
        for (i = 0; i < nl; i++)
            wcs[n++] = lad[i];
        return n;
    }
    if (vfft_policy_lm_serial_tiles(VFFT_IL2D_RADER_RADIX(R[0])))
        for (i = 0; i < nl; i++)
            if (vfft_policy_lm_tile_ok(lad[i], rn, nat))
                wcs[n++] = lad[i];
    if (n == 0)
        wcs[n++] = 0;
    return n;
}
/* the plan an arm runs: the candidate's view in the descriptor (unbanded, serial), the cell's scratch for its class;
 * the 1D real batch's fields beside it. NULL when the 1D real batch's fold order does not derive for the chain (its
 * create would refuse it) or nothing could be allocated. */
static struct vfft_plan_s *_lm_chain_plan(const _il2d_lmform_t *f, int N1, size_t rn, int nst, const int *R,
                                          const int *L, vfft_il2p_fn const *ff, vfft_il2p_fn const *fb,
                                          vfft_real *const *tf, vfft_real *const *tb, int nat, const int *perm,
                                          vfft_real *nscr)
{
    struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    /* the class's scratch: at T > 1 a natural chain's is its own builder's (none where it built none: the door then
     * walks), at one thread and in the 1D real batch every chain's is the one */
    const vfft_ilcol_t *sd = (nat && f->T > 1 && f->kind != 1) ? &f->snat : &f->sscr;
    vfft_ilcol_t *c;
    if (!h)
        return NULL;
    h->transform = f->kind == 0 ? VFFT_C2C : f->rev ? VFFT_C2R : VFFT_R2C;
    h->placement = f->ip ? VFFT_INPLACE : VFFT_OUTOFPLACE;
    h->layout = (int)VFI_LAYOUT_IL;
    h->N = f->kind == 1 ? 2 * N1 : N1;
    h->K = rn;
    h->nthreads = 1;
    h->lm = 1;
    c = &h->il2d_col;
    c->N = N1;
    c->rn = rn;
    c->nst = nst;
    memcpy(c->R, R, (size_t)nst * sizeof c->R[0]);
    memcpy(c->L, L, (size_t)nst * sizeof c->L[0]);
    memcpy(c->f, ff, (size_t)nst * sizeof c->f[0]);
    memcpy(c->b, fb, (size_t)nst * sizeof c->b[0]);
    memcpy(c->tf, tf, (size_t)nst * sizeof c->tf[0]);
    memcpy(c->tb, tb, (size_t)nst * sizeof c->tb[0]);
    c->nat = nat;
    c->natperm = (int *)perm;
    c->natscr = nscr;
    c->natsscr = sd->natsscr;
    c->nnatsscr = sd->nnatsscr;
    c->natswcap = sd->natswcap;
    c->natdense = sd->natdense;
    c->lmsw = -1;
    if (f->kind == 1)
    {
        int ok;
        h->lm_M = N1;
        h->lm_W = f->W;
        h->lm_tw = f->tw;
        h->lm_perm = _lmr_perm_of(R, nst, N1, &ok);
        if (!ok)
        {
            free(h);
            return NULL;
        }
    }
    return h;
}
static void _lm_chain_plan_free(struct vfft_plan_s *h)
{   /* the plan owns its fold order alone: the tables, the scratch and the planes are the race's */
    if (!h)
        return;
    free(h->lm_perm);
    free(h);
}
/* one arm: the door's serial form at the arm's lane tile (_il2d_race_ctx_t.h the plan, .wc the tile; .zo == .z in
 * place) */
static void _lm_chain_arm(void *v)
{
    _il2d_race_ctx_t *a = (_il2d_race_ctx_t *)v;
    struct vfft_plan_s *h = a->h;
    h->il2d_col.wc = a->wc;
    if (h->lm_M > 0)
    {   /* the 1D real batch: its executor (the fused strips, the whole plane, or the three-phase form) */
        if (h->transform == VFFT_C2R)
            _lmc_exec(h, VFFT_BACKWARD, a->z, a->zo);
        else
            _lmr_exec(h, VFFT_FORWARD, a->z, a->zo);
    }
    else   /* the c2c pass, or a real 2D / 3D batch's first axis on the spectrum (the backward for c2r) */
        _lm_cols(h, a->z, a->zo, h->transform == VFFT_C2R);
}
/* the real batch's create, r2c or c2r by the request (the same chain at M on the shared real row,
 * the direction's own executor) */
static vfft_plan _vfft_create_lmr_il(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W, int N, size_t K)
{
    const int T = _vfft_plan_threads(cfg), M = N / 2, c2r = cfg->transform == VFFT_C2R;
    vw2_ilcol_key_t ck;
    vfft_ilcol_t col;
    char forms[64];
    int bwl = -1, btf = -1, bro = -1, bcmt = -1, bcmtt = -1, bblu = -1, oom = 0;
    struct vfft_plan_s *h;
    if (!vfft_policy_batch_lane_major(cfg))
    {
        _vfft_warn("vfft_create: the lane-major interleaved real batch serves R2C and C2R at an even N >= 4 and "
                   "howmany >= 2, in either placement (got %s N=%d howmany=%zu); an odd N has no real column "
                   "engine -- name batch=VFFT_BATCH_CONTIGUOUS for it",
                   _vfft_tname(cfg->transform), N, K);
        return NULL;
    }
    memset(&ck, 0, sizeof ck);
    ck.rank = 1; ck.n0 = N; ck.n1 = K; ck.ord = VW2_ORD_NAT; ck.real = 1; ck.nthreads = T; ck.lm = 1;
    _il2d_blu_ctx.W = W;
    _il2d_blu_ctx.cfg = cfg;
    _il2d_blu_ctx.N2 = K;
    _il2d_blu_chain_hook = _il2d_blu_m_chain;
    memset(&col, 0, sizeof col);
    forms[0] = 0;
    if (!_il2d_col_build(W, cfg, &ck, M, K, /*nat_req=*/0, &col, forms, sizeof forms,
                         &bwl, &btf, &bro, &bcmt, &bcmtt, &bblu, &oom))
    {
        if (!oom)   /* out of memory: the builder's reason stands */
            _vfft_warn("vfft_create: lane-major interleaved R2C batch N=%d x K=%zu: no column pass serves the packed "
                       "length M=%d (the column builder refused); unsupported", N, K, M);
        return NULL;
    }
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        _il2d_col_free(&col);
        return NULL;
    }
    h->transform = c2r ? VFFT_C2R : VFFT_R2C;
    h->placement = cfg->placement;
    h->layout = (int)VFI_LAYOUT_IL;
    h->N = N;
    h->N2 = 0;
    h->K = K;
    h->nthreads = T;
    h->lm = 1;
    h->lm_M = M;
    h->il2d_col = col;
    h->il2d_col.N = M;
    h->il2d_col.rn = K;
    h->il2d_col.wl = 0;
    h->il2d_col.cut = 0;
    h->il2d_col.tfuse = 0;
    h->il2d_col.wc = 0;
    if (!h->il2d_col.blu)
    {   /* the scrambled pass's row order: the fold reads row k at the pass row that holds it */
        int ok;
        h->lm_perm = _lmr_perm_of(h->il2d_col.R, h->il2d_col.nst, M, &ok);
        if (!ok)
        {
            _vfft_warn("vfft_create: lane-major interleaved R2C batch N=%d x K=%zu: the chain's permutation at M=%d "
                       "does not derive; unsupported", N, K, M);
            _il2d_col_free(&h->il2d_col);
            free(h);
            return NULL;
        }
    }
    h->lm_W = (vfft_real *)vfft_aligned_alloc((2 * (size_t)M * K + 8) * sizeof(vfft_real));
    h->lm_tw = (vfft_real *)vfft_aligned_alloc(2 * ((size_t)M + 1) * sizeof(vfft_real));
    if (!h->lm_W || !h->lm_tw)
    {   /* OUT OF MEMORY: the packed plane (M x K complexes) or the fold's twiddles -- refused with that cause */
        _vfft_warn("vfft_create: lane-major interleaved %s batch N=%d x K=%zu: out of memory (its packed plane needs "
                   "%.1f MB)", _vfft_tname(cfg->transform), N, K, (2.0 * (double)M * (double)K + 8.0) * sizeof(vfft_real) / 1048576.0);
        vfft_aligned_free(h->lm_W); vfft_aligned_free(h->lm_tw); free(h->lm_perm);
        _il2d_col_free(&h->il2d_col);
        free(h);
        return NULL;
    }
    _lmr_tw_fill(h->lm_tw, M, N);
    if (!h->il2d_col.blu && !h->il2d_col.tpc)
        (void)_lm_sscr_build(&h->il2d_col, M, K, T);   /* the fused strips' scratch, raced below */
    if (_lm_race(h, W, cfg, &ck, bcmt))
    {   /* OUT OF MEMORY in the race (its reason recorded). Not retained yet (the front door retains the plan it
         * returns): its own release is skipped */
        h->nthreads = 1;
        vfft_destroy((vfft_plan)h);
        return NULL;
    }
    h->k1_exec = c2r ? _lmc_exec : _lmr_exec;   /* the lean door: the direction's own executor, the other refused by the door */
    h->k1_go[VFFT_FORWARD] = c2r ? NULL : _lmr_exec;
    h->k1_go[VFFT_BACKWARD] = c2r ? _lmc_exec : NULL;
    (void)bcmtt; (void)bwl; (void)btf; (void)bro; (void)bblu;
    return (vfft_plan)h;
}
#endif /* VFFT_IL_LM_BATCH_IL_H */
