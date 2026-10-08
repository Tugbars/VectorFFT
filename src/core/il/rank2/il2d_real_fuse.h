/* il2d_real_fuse.h -- THE FUSED WALK (2026-10-07): the interleaved 2D r2c
 * plan's rows fused into column stage 0, the suffix tiled.
 *
 * The standard walk runs the row pass over the whole plane (x -> z), then the
 * natural column pass: stage 0 z -> natscr, the mids in place, the leaf
 * natscr -> z. Past L2 that is a plane written by the rows and read again by
 * stage 0. This form interleaves them by DIGIT:
 *   for each digit d of stage 0 (d < D0 = N1/R0):
 *     ROWS   the row pass over the rows {d + j D0, j < R0} -- the legs of
 *            stage 0's digit-d butterfly -- into a dense staging of R0 rows
 *            (pitch P); the plan's row engine on that row set (the rows
 *            kernel at the set's stride, an engine per row, or the door's
 *            inner per row);
 *     STAGE0 stage 0's digit d: the staging in (legs one staging row apart),
 *            the pre-leaf plane natscr out (legs D0 rows apart, rows d + j D0:
 *            the same-slot DIF comb the standard pass writes);
 *   for each stage-0 sub-problem t (t < R0: natscr rows [t D0, (t+1) D0)):
 *     MIDS   stages 1..nst-2 in place on the tile;
 *     LEAF   the natural leaf over the tile's blocks, in the plan's form
 *            (strided or staged), scattering to z's natural rows.
 * Every row is complete before any column stage reads it: the two-phase
 * law's rationale (the Hermitian fold does not commute with the column
 * stages) holds; the law's letter -- the whole row pass before stage 0 --
 * is what the form gives up (owner, 2026-10-07). Bitwise against the
 * standard walk up to the leaf's staging form: the same kernels, the same
 * calls, another order.
 *
 * PLANNING. A candidate, never a default (vfft_policy_il2d_tfuse_ok,
 * planning/policy_il.h): raced at create against the standard walk on the
 * whole transform, gated at 1e-10, the race's margin toward the standard
 * walk; banked tf=1 (the form serves) or tf=0 on the real row at the plan's
 * T. VFFT_IL2D_TF=1|0 pins (beats wisdom, never banks). Structural
 * admission: a natural chain of two stages or more (not a one-kernel leaf,
 * not a prime column route), the pre-leaf plane present, the real axis on
 * N1 not serving (that form has its own walk). One thread. The threaded form
 * (digits and tiles across workers) and the c2r twin are their own pieces.
 *
 * Included after il2d_real_axis.h and before vfft_execute.h. */
#ifndef VFFT_IL2D_REAL_FUSE_H
#define VFFT_IL2D_REAL_FUSE_H

static int _il2d_tf_admits(const struct vfft_plan_s *h, const vfft_config_t *cfg)
{
    const vfft_ilcol_t *c = &h->il2d_col;
    if (h->transform == VFFT_C2R && (h->il2d_cxd_on || !h->il2d_rscr))
        return 0;   /* the destroying form is its own walk; the standard c2r arm needs its plane */
    return h->il2d_row && !h->il2d_rax_on && !h->il2d_cx_leaf && c->nat && c->nst >= 2 && !c->blu && !c->tpc &&
           c->wl == 0 && c->natscr && c->natperm && vfft_policy_il2d_tfuse_ok(cfg, h->nthreads);
}

static void _il2d_tf_free(struct vfft_plan_s *h)
{
    vfft_aligned_free(h->il2d_tf_stg);
    h->il2d_tf_stg = NULL;
    h->il2d_tf_P = 0;
    h->il2d_tf_on = 0;
}

/* THE FUSED WALK's body: x (N1 x N2 real) -> z (the natural CCE plane) */
static void _il2d_tf_body(const void *v, const double *x, double *z)
{
    struct vfft_plan_s *h = (struct vfft_plan_s *)v;
    const vfft_ilcol_t *c = &h->il2d_col;
    const size_t N1 = (size_t)h->N, hp1 = c->rn, P = h->il2d_tf_P;
    const int nst = c->nst, R0 = c->R[0], Rl = c->R[nst - 1];
    const size_t D0 = N1 / (size_t)R0;
    double *scr = c->natscr, *stg = h->il2d_tf_stg;
    double *stage = h->il2d_cx_st ? c->natstage : NULL;
    size_t d, t;
    for (d = 0; d < D0; d++)
    {
        _il2d_rows_fwd_set(h, x, d, D0, (size_t)R0, stg, P);
        /* stage 0's digit d: the staging's R0 rows in (legs P apart), natscr rows
         * d + j D0 out (legs D0 rows apart), the digit's own twiddle records */
        c->f[0](stg, NULL, scr + 2 * d * hp1, NULL, c->tf[0] + d * (size_t)(R0 - 1) * VFFT_IL_TWREC, NULL,
                P, P, D0 * hp1, 1, hp1);
    }
    for (t = 0; t < (size_t)R0; t++)
    {   /* the sub-problem's tile: the mids in place, the leaf to z's natural rows */
        double *tile = scr + 2 * t * D0 * hp1;
        if (nst > 2)
            _il2d_col_stages(tile, tile, (int)D0, hp1, 1, nst - 1, c->R, c->L, c->f, c->tf, 0);
        _il2d_nat_leaf_range(scr, z, (int)N1, hp1, Rl, c->f[nst - 1], c->natperm,
                             t * D0 / (size_t)Rl, (t + 1) * D0 / (size_t)Rl, 0, stage);
    }
}
/* the serving entry: at the row plan's stack state (the row engines are the
 * kernels that spill), directly when it has none */
static void _il2d_tf_exec_fwd(struct vfft_plan_s *h, const double *x, double *z)
{
    if (h->il2d_rx_on && h->il2d_rx_stk >= 0)
        _il2d_rowx_call(_il2d_tf_body, h, x, z, h->il2d_rx_stk);
    else
        _il2d_tf_body(h, x, z);
}

/* the c2r body: z (the natural CCE plane) -> y (N1 x N2 real): per tile the
 * reverse leaf's gather and the mids reversed; then per digit stage 0's
 * digit d into the staging and that digit's backward rows from it. The
 * column-inverse plane il2d_rscr is never written. */
static void _il2d_tf_body_bwd(const void *v, const double *z, double *y)
{
    struct vfft_plan_s *h = (struct vfft_plan_s *)v;
    const vfft_ilcol_t *c = &h->il2d_col;
    const size_t N1 = (size_t)h->N, hp1 = c->rn, P = h->il2d_tf_P;
    const int nst = c->nst, R0 = c->R[0], Rl = c->R[nst - 1];
    const size_t D0 = N1 / (size_t)R0;
    double *scr = c->natscr, *stg = h->il2d_tf_stg;
    size_t d, t;
    for (t = 0; t < (size_t)R0; t++)
    {
        double *tile = scr + 2 * t * D0 * hp1;
        _il2d_nat_leaf_range(z, scr, (int)N1, hp1, Rl, c->b[nst - 1], c->natperm,
                             t * D0 / (size_t)Rl, (t + 1) * D0 / (size_t)Rl, 1, NULL);
        if (nst > 2)
            _il2d_col_stages(tile, tile, (int)D0, hp1, 1, nst - 1, c->R, c->L, c->b, c->tb, 1);
    }
    for (d = 0; d < D0; d++)
    {
        c->b[0](scr + 2 * d * hp1, NULL, stg, NULL, c->tb[0] + d * (size_t)(R0 - 1) * VFFT_IL_TWREC, NULL,
                D0 * hp1, D0 * hp1, P, 1, hp1);
        _il2d_rows_bwd_set(h, stg, P, d, D0, (size_t)R0, y, 0);
    }
}
static void _il2d_tf_exec_bwd(struct vfft_plan_s *h, const double *z, double *y)
{
    if (h->il2d_rx_on && h->il2d_rx_stk >= 0)
        _il2d_rowx_call(_il2d_tf_body_bwd, h, z, y, h->il2d_rx_stk);
    else
        _il2d_tf_body_bwd(h, z, y);
}

static int _il2d_tf_build(struct vfft_plan_s *h)
{
    const size_t hp1 = h->il2d_col.rn, R0 = (size_t)h->il2d_col.R[0];
    h->il2d_tf_P = hp1;
    h->il2d_tf_stg = (double *)vfft_aligned_alloc((2 * R0 * hp1 + 8) * sizeof(double));
    if (h->il2d_tf_stg)   /* door 2: the pad columns past the bins are read by stage 0, never written by the rows */
        memset(h->il2d_tf_stg, 0, (2 * R0 * hp1 + 8) * sizeof(double));
    return h->il2d_tf_stg != NULL;
}

static void _il2d_tf_bank(struct vfft_wisdom_s *W, const vfft_config_t *cfg, struct vfft_plan_s *h,
                          int N1, int N2, int ord, int T, const char *tk, const char *val)
{
    if (!W || W->vw2_off_2d)
        return;
    if (vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, T, tk, val, h->il2d_ip) != 0)
    {
        vw2_2d_rl_bank(&W->vw2, N1, N2, h->transform == VFFT_C2R, h->il2d_col.R, h->il2d_col.nst, -1, -1, 0,
                       (N1 & (N1 - 1)) ? h->il2d_col.blu : -1, 0.0, ord, T, h->il2d_ip);
        if (vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, T, tk, val, h->il2d_ip) != 0)
        {
            fprintf(stderr, "vfft: the 2D real %s verdict NOT banked at %dx%d -- the cell will re-race\n", tk, N1, N2);
            return;
        }
    }
    _vw2_persist(W, cfg);
}

/* ── the race: the standard walk against the fused walk, whole transform,
 * in the plan's direction (r2c: x -> z; c2r: z -> y through the plan's
 * column-inverse plane, as it serves) ── */
typedef struct
{
    struct vfft_plan_s *h;
    const double *in;
    double *out;
} _il2d_tf_arm_t;
static void _il2d_tf_arm_std(void *v)
{
    _il2d_tf_arm_t *a = (_il2d_tf_arm_t *)v;
    if (a->h->transform == VFFT_C2R)
    {
        _il2d_real_cols(a->h, a->in, a->h->il2d_rscr, /*reverse=*/1);
        _il2d_real_rows_bwd(a->h, a->h->il2d_rscr, a->out);
        return;
    }
    _il2d_real_rows_fwd(a->h, a->in, a->out);
    _il2d_real_cols(a->h, a->out, a->out, /*reverse=*/0);
}
static void _il2d_tf_arm_tf(void *v)
{
    _il2d_tf_arm_t *a = (_il2d_tf_arm_t *)v;
    if (a->h->transform == VFFT_C2R)
        _il2d_tf_exec_bwd(a->h, a->in, a->out);
    else
        _il2d_tf_exec_fwd(a->h, a->in, a->out);
}

/* THE FORM'S PLAN: env pin, the banked tf= (tf_c2r= for a c2r plan), or the
 * race. After the row and column plans (the standard walk's arm runs them)
 * and, for r2c, the real axis on N1. */
static void _il2d_tf_plan(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                          int N1, int N2, int ord, int T)
{
    const int c2r = h->transform == VFFT_C2R;
    const char *log = getenv("VFFT_IL2D_LOG"), *e = getenv(c2r ? "VFFT_IL2D_TF_C2R" : "VFFT_IL2D_TF");
    char tkb[16];
    const char *tk = _il2d_tkp(h, c2r ? "tf_c2r" : "tf", tkb, sizeof tkb), *dn = c2r ? "c2r " : "";
    h->il2d_tf_on = 0;
    if (!_il2d_tf_admits(h, cfg))
        return;
    if (e && e[0])
    {   /* the pin: beats wisdom, never banks */
        if (e[0] == '0')
            return;
        if (_il2d_tf_build(h))
            h->il2d_tf_on = 1;
        else
            _il2d_tf_free(h);
        return;
    }
    if (!W || W->vw2_off_2d)
        return;
    if (!cfg->recalibrate)
    {
        const char *tok = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, tk, h->il2d_ip);
        if (tok)
        {
            if (tok[0] != '1')
                return;   /* 0: the standard walk */
            if (_il2d_tf_build(h))
            {
                h->il2d_tf_on = 1;
                return;
            }
            _il2d_tf_free(h);
        }
    }
    if (!_il2d_tf_build(h))
    {
        _il2d_tf_free(h);
        return;
    }
    {
        const int ip = h->il2d_ip;
        const size_t hp1 = (size_t)N2 / 2 + 1, cp = h->il2d_col.rn, RN = (size_t)N1 * (size_t)N2, CN = 2 * (size_t)N1 * cp;
        /* the pass's planes in the plan's direction; in place the one padded plane (in = its seed) */
        const size_t nin = ip ? CN : (c2r ? CN : RN), nout = ip ? CN : (c2r ? RN : CN);
        double *in = (double *)vfft_aligned_alloc((nin + 8) * sizeof(double));
        double *out = (double *)vfft_aligned_alloc((nout + 8) * sizeof(double));
        double *ref = (double *)vfft_aligned_alloc((nout + 8) * sizeof(double));
        _il2d_tf_arm_t as, af;
        vfft_race_arm_t arms[2];
        double ns[2], t0, est, err;
        int reps, win;
        if (!in || !out || !ref)
        {
            vfft_aligned_free(in); vfft_aligned_free(out); vfft_aligned_free(ref);
            _il2d_tf_free(h);
            return;
        }
        {
            unsigned sd = 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12);
            size_t j;
            for (j = 0; j < nin + 8; j++)
            {
                sd = sd * 1664525u + 1013904223u;
                in[j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
            }
            if (c2r)   /* a CCE plane: the DC and Nyquist bins real, as every engine's contract has them */
                for (j = 0; j < (size_t)N1; j++)
                    in[j * 2 * cp + 1] = in[j * 2 * cp + 2 * (hp1 - 1) + 1] = 0.0;
        }
        as.h = h; as.in = in; as.out = ref;
        af.h = h; af.in = in; af.out = out;
        memset(ref, 0, (nout + 8) * sizeof(double));
        memset(out, 0, (nout + 8) * sizeof(double));
        if (ip)
        {   /* in place: each arm on its own copy of the input plane */
            memcpy(ref, in, (nin + 8) * sizeof(double));
            memcpy(out, in, (nin + 8) * sizeof(double));
            as.in = ref;
            af.in = out;
        }
        _il2d_tf_arm_std(&as);
        _il2d_tf_arm_tf(&af);
        err = _zrpr_relerr(out, ref, nout);
        if (!(err < 1e-10))
        {
            fprintf(stderr, "[il2d-real] %stf %dx%d: the fused walk FAILS the gate (rel %.2e) -- dropped\n", dn, N1, N2, err);
            vfft_aligned_free(in); vfft_aligned_free(out); vfft_aligned_free(ref);
            _il2d_tf_free(h);
            return;
        }
        arms[0].name = c2r ? "columns-then-rows" : "rows-then-columns"; arms[0].run = _il2d_tf_arm_std; arms[0].ctx = &as;
        arms[1].name = "fused"; arms[1].run = _il2d_tf_arm_tf; arms[1].ctx = &af;
        _vfft_create_race_count++;
        t0 = vfft_now_ns();
        _il2d_tf_arm_std(&as);
        est = vfft_now_ns() - t0;
        reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
        if (reps < 1) reps = 1;
        if (reps > 4096) reps = 4096;
        if (ip && reps > 32) reps = 32;   /* in place the plane grows a factor N1 N2 per pass: a sample stays finite */
        {
            _il2d_ip_reset_t rs = { out, in, nin + 8 };
            vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1, 0 };
            if (ip)
            {   /* both arms on the one plane, re-laid before every sample */
                as.in = as.out = out;
                af.in = af.out = out;
                proto.reset = _il2d_ip_reset;
                proto.reset_ctx = &rs;
            }
            vfft_race_run(&proto, arms, 2, ns);
        }
        win = vfft_race_beats(ns[1], ns[0], VFFT_RACE_HYST);
        if (log)
            fprintf(stderr, "[il2d-real] %stf %dx%d race: reps=%d | %s=%.0f fused (stage 0 by digit, %d tiles of %d rows)=%.0f -> %s\n",
                    dn, N1, N2, reps, arms[0].name, ns[0], h->il2d_col.R[0], N1 / h->il2d_col.R[0], ns[1], win ? "fused" : arms[0].name);
        vfft_aligned_free(in); vfft_aligned_free(out); vfft_aligned_free(ref);
        if (win)
            h->il2d_tf_on = 1;
        else
            _il2d_tf_free(h);
        _il2d_tf_bank(W, cfg, h, N1, N2, ord, T, tk, win ? "1" : "0");
    }
}

#endif /* VFFT_IL2D_REAL_FUSE_H */
