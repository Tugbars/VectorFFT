/* il2d_real_pitch.h -- THE REAL TIER'S PITCH FORMS (2026-10-07): two planes the
 * caller never sees moved off the CCE pitch hp1, where a one-kernel column
 * pass in place at that pitch aliases itself (16 bytes past a 4 KB multiple
 * when 512 | N2: the kernel's 32-byte loads of the next column pair against
 * the stores of the pair just written; n1c_16 2.4-2.8x slower, measured
 * 2026-10-06; the law and the numbers in planning/policy_il.h).
 *
 *   cxp  (c2r) THE COLUMN-INVERSE PLANE'S PITCH. The one-kernel reverse pass
 *        writes il2d_rscr at hp1 + d instead of hp1 and the backward rows read
 *        it there (the row body and the column body are pitch-aware for the
 *        plan's own plane; il2d_real_plan.h). d = 0 (the CCE pitch) or the
 *        policy's pitch (1, or 3 where +1 is itself a 4 KB multiple); raced on
 *        the whole c2r, banked cxp_c2r=d. VFFT_IL2D_CXP_C2R=d pins.
 *   csk  (r2c) THE SKEWED PLANE. The rows land on a private N1 x (hp1 + 8)
 *        plane, the one-kernel pass runs in place there, the finished rows
 *        are copied to the caller's plane: one more read and write of the
 *        plane, which pays only where the kernel's loss was large. Raced on
 *        the whole r2c, banked csk=1|0. VFFT_IL2D_RCSK=1|0 pins.
 * Admission (planning/policy_il.h + the structural part here): the plan's
 * column pass is ONE KERNEL (the leaf at N1 = 128, or a one-stage chain), one
 * thread, not the destroying form, not the fused walk, not the real axis on
 * N1. Both forms keep every kernel and call of the plan as it stands; only an
 * address changes, so each is bitwise against the plan it races.
 *
 * Included after il2d_real_fuse.h and before vfft_execute.h. */
#ifndef VFFT_IL2D_REAL_PITCH_H
#define VFFT_IL2D_REAL_PITCH_H

/* the plan's column pass is one kernel */
static int _il2d_onek_pass(const struct vfft_plan_s *h)
{
    const vfft_ilcol_t *c = &h->il2d_col;
    return h->il2d_cx_leaf != NULL || (c->nst == 1 && !c->nat && !c->blu && !c->tpc);
}

/* ═══ cxp: the c2r column-inverse plane's pitch ═══════════════════════ */
static int _il2d_cxp_admits(const struct vfft_plan_s *h, const vfft_config_t *cfg)
{
    return h->transform == VFFT_C2R && h->il2d_row && h->il2d_rscr && !h->il2d_cxd_on && !h->il2d_tf_on &&
           _il2d_onek_pass(h) && vfft_policy_il2d_cxp_ok(cfg, h->nthreads, h->N2);
}
typedef struct
{
    struct vfft_plan_s *h;
    const double *z;
    double *y;
    size_t P;
} _il2d_cxp_arm_t;
static void _il2d_cxp_arm_run(void *v)
{   /* the whole c2r as the plan serves it, the plane at this arm's pitch */
    _il2d_cxp_arm_t *a = (_il2d_cxp_arm_t *)v;
    a->h->il2d_rscr_P = a->P;
    _il2d_real_cols(a->h, a->z, a->h->il2d_rscr, /*reverse=*/1);
    _il2d_real_rows_bwd(a->h, a->h->il2d_rscr, a->y);
}
static void _il2d_cxp_bank(struct vfft_wisdom_s *W, const vfft_config_t *cfg, struct vfft_plan_s *h,
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
/* the race: the plane at hp1 against the policy's pitch, whole transform, both
 * through the plan's own passes; the loser leaves nothing (one field) */
static void _il2d_cxp_plan(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                           int N1, int N2, int ord, int T)
{
    const size_t hp1 = (size_t)N2 / 2 + 1, Pc = vfft_policy_il2d_cxp_pitch(hp1);
    const char *log = getenv("VFFT_IL2D_LOG"), *e = getenv("VFFT_IL2D_CXP_C2R");
    h->il2d_rscr_P = hp1;
    if (!_il2d_cxp_admits(h, cfg))
        return;
    if (e && e[0])
    {   /* the pin: beats wisdom, never banks; a d the law does not give stands as given */
        const int d = atoi(e);
        if (d > 0 && d <= 3)
            h->il2d_rscr_P = hp1 + (size_t)d;
        return;
    }
    if (!W || W->vw2_off_2d)
        return;
    if (!cfg->recalibrate)
    {
        const char *tok = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, "cxp_c2r", h->il2d_ip);
        if (tok)
        {
            const int d = atoi(tok);
            if (d > 0 && d <= 3)
                h->il2d_rscr_P = hp1 + (size_t)d;
            return;
        }
    }
    {
        const size_t RN = (size_t)N1 * (size_t)N2, CN = 2 * (size_t)N1 * hp1;
        double *z = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
        double *y = (double *)vfft_aligned_alloc((RN + 8) * sizeof(double));
        double *yr = (double *)vfft_aligned_alloc((RN + 8) * sizeof(double));
        _il2d_cxp_arm_t a0, a1;
        vfft_race_arm_t arms[2];
        double ns[2], t0, est, err;
        int reps, win;
        char val[8];
        if (!z || !y || !yr)
        {
            vfft_aligned_free(z); vfft_aligned_free(y); vfft_aligned_free(yr);
            return;
        }
        {
            unsigned sd = 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12);
            size_t j;
            for (j = 0; j < CN + 8; j++)
            {
                sd = sd * 1664525u + 1013904223u;
                z[j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
            }
            for (j = 0; j < (size_t)N1; j++)
                z[j * 2 * hp1 + 1] = z[j * 2 * hp1 + 2 * (hp1 - 1) + 1] = 0.0;
        }
        a0.h = h; a0.z = z; a0.y = yr; a0.P = hp1;
        a1.h = h; a1.z = z; a1.y = y; a1.P = Pc;
        memset(yr, 0, (RN + 8) * sizeof(double));
        memset(y, 0, (RN + 8) * sizeof(double));
        _il2d_cxp_arm_run(&a0);
        _il2d_cxp_arm_run(&a1);
        err = _zrpr_relerr(y, yr, RN);
        if (!(err < 1e-10))
        {
            fprintf(stderr, "[il2d-real] c2r cxp %dx%d: the plane at hp1+%d FAILS the gate (rel %.2e) -- dropped\n", N1, N2, (int)(Pc - hp1), err);
            vfft_aligned_free(z); vfft_aligned_free(y); vfft_aligned_free(yr);
            h->il2d_rscr_P = hp1;
            return;
        }
        arms[0].name = "hp1"; arms[0].run = _il2d_cxp_arm_run; arms[0].ctx = &a0;
        arms[1].name = "hp1+d"; arms[1].run = _il2d_cxp_arm_run; arms[1].ctx = &a1;
        _vfft_create_race_count++;
        t0 = vfft_now_ns();
        _il2d_cxp_arm_run(&a0);
        est = vfft_now_ns() - t0;
        reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
        if (reps < 1) reps = 1;
        if (reps > 4096) reps = 4096;
        {
            const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1, 0 };
            vfft_race_run(&proto, arms, 2, ns);
        }
        win = vfft_race_beats(ns[1], ns[0], VFFT_RACE_HYST);
        if (log)
            fprintf(stderr, "[il2d-real] c2r cxp %dx%d race: reps=%d | plane at hp1=%.0f at hp1+%d=%.0f -> %s\n",
                    N1, N2, reps, ns[0], (int)(Pc - hp1), ns[1], win ? "hp1+d" : "hp1");
        vfft_aligned_free(z); vfft_aligned_free(y); vfft_aligned_free(yr);
        h->il2d_rscr_P = win ? Pc : hp1;
        snprintf(val, sizeof val, "%d", win ? (int)(Pc - hp1) : 0);
        _il2d_cxp_bank(W, cfg, h, N1, N2, ord, T, "cxp_c2r", val);
    }
}

/* ═══ csk: the r2c one-kernel pass on a skewed private plane ══════════ */
static int _il2d_rcsk_admits(const struct vfft_plan_s *h, const vfft_config_t *cfg)
{
    return h->transform == VFFT_R2C && h->il2d_row && !h->il2d_rax_on && !h->il2d_tf_on && _il2d_onek_pass(h) &&
           h->il2d_ip != 2 &&   /* door 2's plane is already off the alias */
           vfft_policy_il2d_rcsk_ok(cfg, h->nthreads);
}
static void _il2d_rcsk_free(struct vfft_plan_s *h)
{
    vfft_aligned_free(h->il2d_rcsk_scr);
    h->il2d_rcsk_scr = NULL;
    h->il2d_rcsk_P = 0;
    h->il2d_rcsk_on = 0;
}
static int _il2d_rcsk_build(struct vfft_plan_s *h)
{
    const size_t hp1 = (size_t)h->N2 / 2 + 1;
    h->il2d_rcsk_P = vfft_policy_il2d_rcsk_pitch(hp1);
    h->il2d_rcsk_scr = (double *)vfft_aligned_alloc((2 * (size_t)h->N * h->il2d_rcsk_P + 8) * sizeof(double));
    return h->il2d_rcsk_scr != NULL;
}
/* the walk: the rows onto the skewed plane, the one kernel in place there, the rows
 * out. Each pass is entered at ITS plan's stack state as the standard walk enters it
 * (the rows at the row plan's, the kernel at the column plan's: a kernel entered at
 * another depth is the Win64 stack lottery -- 2x on the leaf). */
static void _il2d_rcsk_rows(const void *v, const double *x, double *scr)
{
    struct vfft_plan_s *h = (struct vfft_plan_s *)v;
    _il2d_rows_fwd_set(h, x, 0, 1, (size_t)h->N, scr, h->il2d_rcsk_P);
}
static void _il2d_rcsk_kern(const void *v, const double *src, double *dst)
{
    const struct vfft_plan_s *h = (const struct vfft_plan_s *)v;
    const vfft_ilcol_t *c = &h->il2d_col;
    const size_t P = h->il2d_rcsk_P;
    if (h->il2d_cx_leaf)
        h->il2d_cx_leaf(src, NULL, dst, NULL, NULL, NULL, P, 0, P, 0, c->rn);
    else
        c->f[0](src, NULL, dst, NULL, NULL, NULL, P, 0, P, 0, c->rn);
}
static void _il2d_rcsk_exec_fwd(struct vfft_plan_s *h, const double *x, double *z)
{
    const size_t N1 = (size_t)h->N, hp1 = h->il2d_col.rn, P = h->il2d_rcsk_P;
    double *scr = h->il2d_rcsk_scr;
    const int ks = h->il2d_cx_perk ? (int)h->il2d_cx_ks[0] : h->il2d_cx_stk;
    size_t r;
    if (h->il2d_rx_on && h->il2d_rx_stk >= 0)
        _il2d_rowx_call(_il2d_rcsk_rows, h, x, scr, h->il2d_rx_stk);
    else
        _il2d_rcsk_rows(h, x, scr);
    if (ks >= 0)
        _il2d_rowx_call(_il2d_rcsk_kern, h, scr, scr, ks);
    else
        _il2d_rcsk_kern(h, scr, scr);
    for (r = 0; r < N1; r++)
        memcpy(z + 2 * r * hp1, scr + 2 * r * P, 2 * hp1 * sizeof(double));
}
typedef struct
{
    struct vfft_plan_s *h;
    const double *x;
    double *z;
} _il2d_rcsk_arm_t;
static void _il2d_rcsk_arm_std(void *v)
{
    _il2d_rcsk_arm_t *a = (_il2d_rcsk_arm_t *)v;
    _il2d_real_rows_fwd(a->h, a->x, a->z);
    _il2d_real_cols(a->h, a->z, a->z, /*reverse=*/0);
}
static void _il2d_rcsk_arm_skew(void *v)
{
    _il2d_rcsk_arm_t *a = (_il2d_rcsk_arm_t *)v;
    _il2d_rcsk_exec_fwd(a->h, a->x, a->z);
}
static void _il2d_rcsk_plan(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                            int N1, int N2, int ord, int T)
{
    const char *log = getenv("VFFT_IL2D_LOG"), *e = getenv("VFFT_IL2D_RCSK");
    h->il2d_rcsk_on = 0;
    if (!_il2d_rcsk_admits(h, cfg))
        return;
    if (e && e[0])
    {   /* the pin: beats wisdom, never banks */
        if (e[0] == '0')
            return;
        if (_il2d_rcsk_build(h))
            h->il2d_rcsk_on = 1;
        else
            _il2d_rcsk_free(h);
        return;
    }
    if (!W || W->vw2_off_2d)
        return;
    if (!cfg->recalibrate)
    {
        const char *tok = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, "csk", h->il2d_ip);
        if (tok)
        {
            if (tok[0] != '1')
                return;
            if (_il2d_rcsk_build(h))
            {
                h->il2d_rcsk_on = 1;
                return;
            }
            _il2d_rcsk_free(h);
        }
    }
    if (!_il2d_rcsk_build(h))
    {
        _il2d_rcsk_free(h);
        return;
    }
    {
        const int ip = h->il2d_ip;
        const size_t hp1 = (size_t)N2 / 2 + 1, CN = 2 * (size_t)N1 * h->il2d_col.rn;
        const size_t RN = ip ? CN : (size_t)N1 * (size_t)N2;   /* in place: x is the one plane's seed */
        double *x = (double *)vfft_aligned_alloc((RN + 8) * sizeof(double));
        double *z = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
        double *zr = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
        _il2d_rcsk_arm_t as, ak;
        vfft_race_arm_t arms[2];
        double ns[2], t0, est, err;
        int reps, win;
        if (!x || !z || !zr)
        {
            vfft_aligned_free(x); vfft_aligned_free(z); vfft_aligned_free(zr);
            _il2d_rcsk_free(h);
            return;
        }
        {
            unsigned sd = 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12);
            size_t j;
            for (j = 0; j < RN + 8; j++)
            {
                sd = sd * 1664525u + 1013904223u;
                x[j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
            }
        }
        as.h = h; as.x = x; as.z = zr;
        ak.h = h; ak.x = x; ak.z = z;
        memset(zr, 0, (CN + 8) * sizeof(double));
        memset(z, 0, (CN + 8) * sizeof(double));
        if (ip)
        {   /* in place: each arm on its own copy of the input plane */
            memcpy(zr, x, (RN + 8) * sizeof(double));
            memcpy(z, x, (RN + 8) * sizeof(double));
            as.x = zr;
            ak.x = z;
        }
        _il2d_rcsk_arm_std(&as);
        _il2d_rcsk_arm_skew(&ak);
        err = _zrpr_relerr(z, zr, CN);
        if (!(err < 1e-10))
        {
            fprintf(stderr, "[il2d-real] csk %dx%d: the skewed plane FAILS the gate (rel %.2e) -- dropped\n", N1, N2, err);
            vfft_aligned_free(x); vfft_aligned_free(z); vfft_aligned_free(zr);
            _il2d_rcsk_free(h);
            return;
        }
        arms[0].name = "in place"; arms[0].run = _il2d_rcsk_arm_std; arms[0].ctx = &as;
        arms[1].name = "skewed"; arms[1].run = _il2d_rcsk_arm_skew; arms[1].ctx = &ak;
        _vfft_create_race_count++;
        t0 = vfft_now_ns();
        _il2d_rcsk_arm_std(&as);
        est = vfft_now_ns() - t0;
        reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
        if (reps < 1) reps = 1;
        if (reps > 4096) reps = 4096;
        if (ip && reps > 32) reps = 32;   /* in place the plane grows a factor N1 N2 per pass: a sample stays finite */
        {
            _il2d_ip_reset_t rs = { z, x, RN + 8 };
            vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1, 0 };
            if (ip)
            {   /* both arms on the one plane, re-laid before every sample */
                as.x = as.z = z;
                ak.x = ak.z = z;
                proto.reset = _il2d_ip_reset;
                proto.reset_ctx = &rs;
            }
            vfft_race_run(&proto, arms, 2, ns);
        }
        win = vfft_race_beats(ns[1], ns[0], VFFT_RACE_HYST);
        if (log)
            fprintf(stderr, "[il2d-real] csk %dx%d race: reps=%d | in place=%.0f skewed (pitch hp1+%d, rows copied out)=%.0f -> %s\n",
                    N1, N2, reps, ns[0], (int)(h->il2d_rcsk_P - hp1), ns[1], win ? "skewed" : "in place");
        vfft_aligned_free(x); vfft_aligned_free(z); vfft_aligned_free(zr);
        if (win)
            h->il2d_rcsk_on = 1;
        else
            _il2d_rcsk_free(h);
        _il2d_cxp_bank(W, cfg, h, N1, N2, ord, T, "csk", win ? "1" : "0");
    }
}

#endif /* VFFT_IL2D_REAL_PITCH_H */
