/* real_estimate_nd_il.h — the ESTIMATE create of one 2D real transform (VFFT_FLAG_ESTIMATE; R2C and C2R, natural
 * order, either placement, any thread count): the 2D real door's row written from formulas into a private store,
 * then the door's own create replays it with nothing measured (docs/roadmap/real_estimate_plan.md, stage 3).
 *
 * THE ROW (il/wisdom/wisdom2_2d_il_reader.h:335-521; one t=r2c row serves both directions, each direction's
 * verdicts under its own names, C2R's with _c2r):
 *   - the column pass at (N1, N2/2 + 1), natural: chain forms blu rd rdw, est_col_plan's (estimate_nd_il.h) as in
 *     C2C; no turned prime pass on a real row (il2d_tier.h:3278-3281);
 *   - the band wl (0, unbanded) and at T > 1 the threaded column verdict cmt mtarm msw (est_mt_2d's rule: threaded
 *     past half the L1d, the strips at est__msw_2d's width);
 *   - the row engine rx (the rows kernel lm where it is offered, else the door's row child) and rxs;
 *   - the column form cx (strided on a natural chain, else plain) and cxs;
 *   - one thread: the fused walk tf (past the L2), the real axis raxis (n2), the skew csk (0), C2R's pitch cxp (0)
 *     and destroying form cxd (off);
 *   - the row child rp_ / rp_c2r_: at even N2 the 1D real estimate at N2 (the K = 1 inner of the row batch,
 *     at T) and at T > 1 the batch's verdict tcmt (est_batch_whole); at odd N2 the C2C estimate at N2 (one thread,
 *     out of place).
 * The first build carries no new rule: every verdict above is the C2C one, the 1D real one, or the store's
 * majority, for the study to judge (the plan's section 3, step 1).
 *
 * THE BUILD: est_build_real_2d, the 1D builds' steps: the rows into a private store, est__replay's detectors
 * (nothing measured, nothing re-banked), the plan check (the column chain, the row engine, the column form, the
 * one-thread forms served only where asked, the threaded verdict, the row child). An estimate create never hands
 * back a plan that measured.
 *
 * POSITION IN vfft.c: included by c2c_estimate_il.h after real_estimate_il.h. */
#ifndef VFFT_IL_REAL_ESTIMATE_ND_IL_H
#define VFFT_IL_REAL_ESTIMATE_ND_IL_H

/* the recipe of one 2D real plane */
typedef struct est_plan2dr_s
{
    int N1, N2, T, c2r, ip;
    est_col_t col;            /* [chain forms blu rd rdw] at (N1, N2/2 + 1), natural; tpc never */
    int wl;                   /* [wl | wl_c2r] the band in rows, 0 = unbanded */
    int cmt, mtarm, msw;      /* [cmt mtarm msw (_c2r)] T > 1: threaded columns, 1 = the strips, their width */
    char rx[96], rxs[8];      /* [rx rxs (_c2r)] the row engine and its stack state; "" = not written */
    char cx[16], cxs[8];      /* [cx cxs (_c2r)] the column form and its stack state; "" = not written */
    int tf, raxis, csk, cxp;  /* [tf (_c2r), raxis, csk, cxp_c2r] one thread; -1 = not written */
    int has_rxe;              /* rxe holds the 1D real estimate at N2 as the door builds a row engine */
    est_real_recipe_t rxe;    /* the row engine's recipe (one thread, out of place: il2d_real_plan.h:446, 495);
                               * rx names it by est__rx_token, a zr2c engine with its child under rx_ */
    est_col_t rax_col;        /* raxis = 1 (n1): rax_, the C2C column chain at M = N1/2 down N2 columns, scrambled */
    est_recipe_t raxr;        /* raxis = 1: raxr_, the C2C plan at N2 the folded rows' batch runs (natural, out of place) */
    int odd;                  /* odd N2: the row child is the C2C plan at N2 */
    est_real_recipe_t rp;     /* even N2: the 1D real recipe at N2, at T */
    est_recipe_t rpc;         /* odd N2: the C2C recipe at N2 */
    int tcmt;                 /* T > 1, even N2: the row batch's verdict; -1 = not written */
} est_plan2dr_t;

/* the rows kernel's lengths at this ISA (the r2zr list, il_registry: AVX2 4 .. 32, none at AVX-512) */
static inline int est__rows_lm_ok(int N2)
{
    static const int n[7] = { 4, 6, 8, 10, 12, 16, 32 };
    if (VFFT_IL_VW != 4)
        return 0;
    for (int i = 0; i < 7; i++)
        if (n[i] == N2)
            return 1;
    return 0;
}

/* a 1D real recipe as the 2D door's row-engine token (_il2d_rowx_name, il2d_real_plan.h:453-484); 0 = none */
static inline int est__rx_token(const est_real_recipe_t *r, char *b, size_t cap)
{
    char cs[64];
    int i, o = 0;
    if (!strcmp(r->eng, "zrm"))
        snprintf(b, cap, "zrm");
    else if (!strcmp(r->eng, "zrp"))
        snprintf(b, cap, "zrp_%d.%d_%c", r->zrp.R1, r->zrp.R2, r->zrp.form ? 'b' : 'a');
    else if (!strcmp(r->eng, "zttr"))
    {
        for (i = 0; i < r->zttr.nf && o < (int)sizeof cs - 4; i++)
            o += snprintf(cs + o, sizeof cs - o, "%s%d", i ? "." : "", r->zttr.chain[i]);
        snprintf(b, cap, "zttr_%s_t%zu_s%d", cs, r->zttr.tile, r->zttr.stk);
    }
    else if (!strcmp(r->eng, "zrf"))
    {
        vfft_zrf_chain_str(r->zrf.chain, r->zrf.K, cs, sizeof cs);
        snprintf(b, cap, "zrf_%s%s_w%d", cs, r->zrf.nomsz ? "_t" : "", r->zrf.tile > 0 ? r->zrf.tile : 0);
    }
    else if (!strcmp(r->eng, "zrb"))
    {
        if (r->zrb.tw > 0)
            snprintf(b, cap, "zrb_%d_%s:%s_t%d", r->zrb.M, r->zrb.kind, r->zrb.shape, r->zrb.tw);
        else
            snprintf(b, cap, "zrb_%d_%s:%s", r->zrb.M, r->zrb.kind, r->zrb.shape);
    }
    else if (!strcmp(r->eng, "zr2c"))
        snprintf(b, cap, "zr2c_r%d", r->zr2c.route);
    else
        return 0;
    return 1;
}

/* the forms token of a real plane's column pass as the door's race spells it (il2d_tier.h:3227-3262): R2C's pool
 * adds the half-store twins at the leaf, so its token is written wherever one exists; each stage the construction
 * table's default, names[0], the full-store form (the planner's leaf in the store: r16 23 of 30, r64 19 of 25,
 * r32 26 of 31, r2 9 of 14, r4 16 of 19) */
static inline void est__col_forms_real(const int *R, int nst, int c2r, char *f, size_t cap)
{
    const char *nm[VFFT_IL2P_COL_MAXFORMS];
    est_sb_t s;
    int i, any = 0;
    est_sb_init(&s, f, cap);
    for (i = 0; i < nst; i++)
        if (vfft_il2p_col_forms(R[i], !c2r && i == nst - 1, nm) > 1)
            any = 1;
    if (!any)
        return;
    for (i = 0; i < nst; i++)
    {
        (void)vfft_il2p_col_forms(R[i], !c2r && i == nst - 1, nm);
        est_sb_add(&s, "%s%s", i ? "." : "", nm[0]);
    }
    if (s.len >= cap)
        f[0] = 0;
}

/* THE TALL REAL COLUMN (2026-11-02): the real plane's column pass at a power of two from 512 to 8192 takes one chain
 * per length, both directions, at every row width, where C2C's formula leads with 32 from 2048 on narrow rows (the
 * C2C planner's habit there, 32.4.32 at 4096 x 16) and runs the radix-8 pick below. The real planner leads small.
 * One thread, timed against the planner on the banked planes (N2 4 - 8192): R2C 59 planes 1.124 -> 0.975, C2R 43
 * planes 1.043 -> 0.982; per length R2C | C2R: 512 1.066 -> 0.984 | 1.052 -> 0.994, 1024 1.039 -> 1.001 |
 * 1.011 -> 0.969, 2048 1.059 -> 0.964 | 1.034 -> 0.940, 4096 1.307 -> 0.964 | 1.059 -> 1.009, 8192 1.287 -> 0.949 |
 * 1.056 -> 1.001. No one formula names the five: each is the best of 5 - 7 chains timed on the same planes. */
static const struct { int N, nst, R[4]; } est_col_real_tall[] = {
    { 512, 2, { 16, 32, 0, 0 } },
    { 1024, 3, { 8, 4, 32, 0 } },
    { 2048, 3, { 8, 8, 32, 0 } },
    { 4096, 4, { 8, 8, 8, 8 } },
    { 8192, 3, { 8, 16, 64, 0 } },
};

/* the pool's column chain at N with the fewest stages: on two stages the most balanced (least sum of (log2 R)^2), on
 * three or more the largest radix last and then the most balanced; ties ascending, then the pool's order. The stage
 * count, 0 = none. (THE TALL PLANE AT T > 1, est_plan_2d_real.) */
static inline int est__col_fewest(int N, int *R)
{
    est_pool_t P;
    int i, best = -1;
    est__pool(N, &P);
    for (i = 0; i < P.n; i++)
    {
        int better;
        if (best < 0) { best = i; continue; }
        if (P.len[i] != P.len[best])
            better = P.len[i] < P.len[best];
        else if (P.len[i] >= 3 && P.c[i][P.len[i] - 1] != P.c[best][P.len[best] - 1])
            better = P.c[i][P.len[i] - 1] > P.c[best][P.len[best] - 1];
        else
        {
            const double a = est__sumsq(P.c[i], P.len[i]), b = est__sumsq(P.c[best], P.len[best]);
            if (a < b - 1e-9 || a > b + 1e-9) better = a < b;
            else better = est__inversions(P.c[i], P.len[i], 1) < est__inversions(P.c[best], P.len[best], 1);
        }
        if (better) best = i;
    }
    if (best < 0)
        return 0;
    memcpy(R, P.c[best], sizeof(int) * (size_t)P.len[best]);
    return P.len[best];
}

/* 1 = a stage s of the chain R over N, its R[s] elements N / (R[0] ... R[s]) rows of w complex apart, crowds an L1d set:
 * they fall into 4096 / gcd(stride, 4096) sets of L1d / 4096 lines (the one-stage axis 0's count, stage by stage) */
static inline int est__chain_crowds(const est_machine_t *m, int N, const int *R, int nst, long long w)
{
    long long L = N;
    for (int s = 0; s < nst; s++)
    {
        L /= R[s];
        const long long st = 16 * L * w, low = st & -st;
        if ((long long)R[s] * (low < 4096 ? low : 4096) > (long long)m->l1d)
            return 1;
    }
    return 0;
}

/* the real axis on N1 (raxis = n1, il/rank2/il2d_real_axis.h:212-300; R2C, one thread, out of place, even N1 >= 4, odd
 * N2): its two children's recipes, the C2C column chain at M = N1/2 down N2 columns in the scrambled class
 * (est_col_plan) and the C2C plan at N2 the batch of M + 1 folded rows runs. 0, or -1 with the reason. */
struct est_plan2dr_s;
static inline int est_plan_rax_real(const est_machine_t *m, struct est_plan2dr_s *p, char *why, size_t whylen);

/* est_plan_2d_real: the machine, the plane and the request -> the recipe. 0, or -1 with the reason. Pure. */
static inline int est_plan_2d_real(const est_machine_t *m, int N1, int N2, int c2r, int ip, int T, est_plan2dr_t *p,
                                   char *why, size_t whylen)
{
    const long long hp1 = N2 / 2 + 1, bytes = 16LL * N1 * hp1;
    const int c2r128wide = c2r && T <= 1 && N1 == 128 && bytes > (long long)m->l2 * 5 / 8;
    const int tallmt = T > 1 && N1 != 2 && bytes > m->l1d / 2 && N1 >= 4LL * N2;   /* the threaded column of a tall plane */
    char dummy[1];
    if (!why || !whylen) { why = dummy; whylen = sizeof dummy; }
    why[0] = 0;
    memset(p, 0, sizeof *p);
    if (est__machine_ok(m, why, whylen))
        return -1;
    if (N1 < 2 || N2 < 2)
        return est_refuse(why, whylen, "a %dx%d plane: every axis at least 2", N1, N2);
    p->N1 = N1; p->N2 = N2; p->T = T > 1 ? T : 1; p->c2r = c2r; p->ip = ip;
    if (est_col_plan(m, N1, hp1, 1, &p->col, why, whylen))
        return -1;
    p->col.tpc = -1;              /* no turned prime pass on a real row (il2d_tier.h:3278-3281) */
    if ((N1 == 8 || N1 == 16 || N1 == 32 || N1 == 64) && p->col.blu <= 0 && p->col.nst > 1)
    {   /* a column pass of 8 to 64 points in one stage at any width, where C2C's formula splits it on a wide row
         * (stage 3 study, 2026-11-02: N1 = 64 1.09 - 1.37 -> 1.00 - 1.05 on 4 of 8 planes, none slower; the unseen
         * 32 x 2048 1.24 - 1.38 -> 1.00 - 1.02; the planner's chain at every power-of-two N1 from 8 to 64 in the
         * store but 8 x 4) */
        p->col.R[0] = N1;
        p->col.nst = 1;
        p->col.rd_n = 0;
    }
    if (p->T == 1)
        for (int i = 0; i < EST_NELEM(est_col_real_tall); i++)
            if (est_col_real_tall[i].N == N1)
            {   /* the tall real column (est_col_real_tall) */
                memcpy(p->col.R, est_col_real_tall[i].R, sizeof(int) * (size_t)est_col_real_tall[i].nst);
                p->col.nst = est_col_real_tall[i].nst;
            }
    if (c2r128wide)
    {   /* C2R's 128-point column past 5/8 of the L2: the plain two-stage column, the planner's 16.8 on every such row
         * (2026-11-02, 10 planes 128 x 1536 - 16384: rule 3's b816 leaf 1.122 -> 0.961, each at or below it; the
         * leaf still wins at 128 x 1024 and below, 1.047 against the plain column's 1.11 - 1.18) */
        p->col.R[0] = 16;
        p->col.R[1] = 8;
        p->col.nst = 2;
    }
    if (tallmt && p->col.blu <= 0 && p->col.rd_n == 0)
    {   /* THE TALL PLANE AT T > 1 (2026-11-02): where N1 >= 4 N2 the threaded column runs plain (mtarm=0, below) over
         * the chain with the fewest stages (est__col_fewest), the planner's verdicts on such planes: mtarm=0 on all 62
         * banked 8-thread rows with N1 >= 4 N2, its chains 16.32, 32.32, 32.64, 64.64, 64.10, 3.8.64. At 8 threads,
         * 60 tall planes of 256 - 8192 rows, power of two or not, both directions: 2.071 -> 1.077 against the planner
         * (its own column verdicts 1.045), 58 faster, one slower (C2R 3000 x 64, 1.02 -> 1.11); on the near-square
         * 2048 x 1024, where the planner keeps the strips, it was slower (1.06 - 1.15 -> 1.22 - 1.41). */
        int R[8];
        const int k = est__col_fewest(N1, R);
        if (k > 0)
        {
            memcpy(p->col.R, R, sizeof(int) * (size_t)k);
            p->col.nst = k;
        }
    }
    if (p->col.blu <= 0)
        est__col_forms_real(p->col.R, p->col.nst, c2r, p->col.forms, sizeof p->col.forms);
    if (!c2r && p->T == 1 && N1 == 16 && N2 % 512 == 0 && p->col.nst == 1)
    {   /* R2C's 16-point column over rows a multiple of 512: the leaf's half-store twin (2026-11-02: 16 x 512, 1024,
         * 1536, 4096, 8192 1.06 - 1.18 -> 0.95 - 1.02, 16 x 2048 flat; at 16 x 1000, 3000, 6000, 12000 it is 1 - 3 %
         * slower, and the 8, 32 and 64-point leaves gain nothing; the planner's leaf at 1024 - 8192 and 1536) */
        const char *nm[VFFT_IL2P_COL_MAXFORMS];
        const int k = vfft_il2p_col_forms(16, 1, nm);
        for (int j = 1; j < k; j++)
            if (strchr(nm[j], 'h'))
            {
                snprintf(p->col.forms, sizeof p->col.forms, "%s", nm[j]);
                break;
            }
    }
    p->wl = 0;
    p->cmt = p->mtarm = p->msw = 0;
    if (p->T > 1 && N1 != 2 && bytes > m->l1d / 2)
    {   /* est_mt_2d's column rule at the real plane's bytes: threaded, the strips at their width */
        /* a Rader column on a plane past the L3 runs plain too (2026-11-02, one plane: 4099 x 2048 at 8 threads, R2C
         * 2.75 -> 1.17, C2R 2.62 -> 1.14; the only such plane banked -- the smaller Rader planes keep the strips, where
         * the plain column was slower, 109 x 512 0.96 -> 1.26) */
        const int plain = tallmt || (p->col.rd_n > 0 && m->l3 > 0 && bytes > (long long)m->l3);
        p->cmt = 1;
        p->mtarm = plain ? 0 : 1;   /* the tall plane's plain threaded column (above) */
        p->msw = plain ? 0 : est__msw_2d(m, N1, (int)hp1, p->T);
    }
    p->rx[0] = p->rxs[0] = p->cx[0] = p->cxs[0] = 0;
    if (!c2r || !(N2 & 1))
    {   /* C2R's row engines race at even N2 only (fft2d_create_il.h:1259) */
        /* the rows kernel out of place on one thread; at T > 1 the door's row child (the kernel 2.2 - 3.8 x slower at
         * 16 x 16 and 32 x 32 on 8 threads, the child 0.98 - 1.02) */
        snprintf(p->rx, sizeof p->rx, "%s", (!ip && p->T == 1 && est__rows_lm_ok(N2)) ? "lm" : "door");
        snprintf(p->rxs, sizeof p->rxs, "any");
        {
            char w2[160];
            p->has_rxe = N2 >= 4 && !est_real(m, N2, c2r, 0, 1, &p->rxe, w2, sizeof w2);
        }
        if ((N2 & 1) && p->has_rxe && (!strcmp(p->rxe.eng, "zrf") || !strcmp(p->rxe.eng, "zrb")))
        {   /* R2C at odd N2: the 1D real estimate's odd engine as the row engine (stage 3 study, 2026-11-02: the
             * door's C2C row child 1.46 / 1.52 against the planner, the odd chain 1.10 on 8 of 8 planes, Bluestein
             * 1.24 on 5 of 6; the planner's own row engine at odd N2 is one of the two); the small kernel at
             * N2 <= 64 stays on the child (5.3 x slower at 16x61) */
            char tk[96];
            if (est__rx_token(&p->rxe, tk, sizeof tk))
                snprintf(p->rx, sizeof p->rx, "%s", tk);
        }
        if (!c2r && !ip && p->T > 1 && N2 == 64 && N1 >= 4LL * N2 && VFFT_IL_VW == 4)
            /* R2C AT T > 1 OVER ROWS OF 64 ON A TALL PLANE (2026-11-02): the rows by the real pair 4.16, form a, the
             * planner's on every banked tall plane over rows of 64 from 512 rows (the pair engine exists at 4 lanes
             * only, as the 1D real estimate's zrp). 8 threads, 8 planes 256 - 8192 x 64: 1.181 -> 1.023, 7 faster,
             * none slower; over rows of 32 and 128, and in C2R, the door's row child stays (no gap) */
            snprintf(p->rx, sizeof p->rx, "zrp_4.16_a");
    }
    if (p->col.blu <= 0)
    {   /* no column form on a Bluestein pass (il2d_real_plan.h:1575) */
        snprintf(p->cx, sizeof p->cx, "%s", p->col.nst >= 2 ? "strided" : "plain");
        snprintf(p->cxs, sizeof p->cxs, "any");
        if (N1 == 128 && (p->T == 1 || !c2r) && !c2r128wide)
            /* the 128-point column leaf, the whole pass in one kernel call (vfft_il2p_col_leaf_forms; the planner's
             * at 128 x 128 in the store, 1.07 - 1.24 -> 0.99 - 1.08 on the study's N1 = 128 planes): R2C b448, C2R
             * b816 (C2R's b448 1.11 at 128 x 16 against b816's 1.04); at T > 1 R2C's alone (b448 1.15 - 1.33 -> 0.88 -
             * 1.01 on 3 of 4 planes at 8 threads; C2R's b816 there one faster, one slower) */
            snprintf(p->cx, sizeof p->cx, "%s", c2r ? "b816" : "b448");
    }
    p->tf = p->raxis = p->csk = p->cxp = -1;
    if (p->T == 1)
    {   /* the fused walk: R2C past the L2 (below it 1.10 -> 1.20 at L1, flat at L2), C2R at every size (1.07 -> 1.03
         * on 18 planes below the L2, none slower; past it off is slower on 28 of 30); and R2C over rows of 32 from
         * 256 rows at every size (2026-11-02, 10 of 10 planes 256 - 4096 rows, power of two or not, 1.06 - 1.26 ->
         * 0.88 - 1.08, 128 x 32 flat; at rows of 4 - 16 it costs 5 - 40 %, at 24, 48, 64 it is mixed) */
        p->tf = (c2r || bytes > m->l2 || (N2 == 32 && N1 >= 256)) ? 1 : 0;
        if (!c2r) { p->raxis = 0; p->csk = 0; }
        if (!c2r && !ip && N1 >= 4 && !(N1 & 1) && (N2 & 1) && N2 <= 2048)
        {   /* the real axis on N1 where the real axis N2 is an odd prime up to 2048 (stage 3 study, 2026-11-02: 16x61,
             * 32x127, 256x127, 64x1031, 1024x1031 1.09 - 1.68 -> 1.00 - 1.07; at 4099 and at the odd composites but
             * 24x45 it is slower; the planner's verdict on every one of the 16 odd-N2 planes but 16x105 and 24x45) */
            int q = 3, prime = N2 > 2;
            for (; prime && (long long)q * q <= N2; q += 2)
                if (N2 % q == 0) prime = 0;
            if (prime) { p->raxis = 1; p->tf = 0; }
        }
        else p->cxp = 3;  /* the column-inverse plane at hp1 + 3 (1.014 -> 0.999 on 17 planes, none slower) */
    }
    p->odd = N2 & 1;
    p->tcmt = -1;
    if (p->raxis == 1 && est_plan_rax_real(m, p, why, whylen))
        return -1;
    if (!p->odd)
    {
        if (est_real(m, N2, c2r, ip, p->T, &p->rp, why, whylen))
            return -1;
        if (p->T > 1)   /* the rows threaded as the batch rule says, but serial where the plane itself is (the column
                         * rule's bound: 8 x 256 and 48 x 48 at 8 threads 1.06 - 1.83 -> 0.52 - 0.88; above it the
                         * serial rows 1.6 - 3.5 x slower) */
            p->tcmt = bytes <= m->l1d / 2 ? 0 : est_batch_whole(m, (size_t)(16 * hp1), (size_t)N1, p->T);
    }
    else
    {
        est_request_t rq;
        est__rq1(&rq, N2, VFFT_PLACEMENT_OUT);
        if (est_plan_1d(m, &rq, &p->rpc, why, whylen))
            return -1;
    }
    return 0;
}

static inline int est_plan_rax_real(const est_machine_t *m, struct est_plan2dr_s *p, char *why, size_t whylen)
{
    est_request_t rq;
    if (p->c2r || p->ip || p->T > 1 || p->N1 < 4 || (p->N1 & 1) || !(p->N2 & 1))
        return est_refuse(why, whylen, "the real axis on N1 serves R2C out of place on one thread, even N1 >= 4, odd N2");
    if (est_col_plan(m, p->N1 / 2, p->N2, 0, &p->rax_col, why, whylen))
        return -1;
    p->rax_col.tpc = -1;
    p->rax_col.wl = 0;
    est__rq1(&rq, p->N2, VFFT_PLACEMENT_OUT);
    return est_plan_1d(m, &rq, &p->raxr, why, whylen);
}

/* est_put_rows_2d_real: the plane's row into st, the requested direction's verdicts, with the row child. 0, or -1
 * with the reason. */
static inline int est_put_rows_2d_real(vw2_store_t *st, const est_plan2dr_t *p, char *why, size_t whylen)
{
    const int N1 = p->N1, N2 = p->N2, T = p->T, ip = p->ip, ord = VW2_ORD_NAT;
    const char *sfx = p->c2r ? "_c2r" : "";
    const vw2_ilcol_key_t ck = { 2, N1, N2, 0, ord, 0, 1, T, ip };
    struct vfft_wisdom_s *S;
    vw2_key_t key;
    char nm[32], b[64];
    est_sb_t s;
    int ok;
#define EST_TOK(BASE, VAL) do { snprintf(nm, sizeof nm, "%s%s", (BASE), sfx); \
        if (vw2_2d_rl_tok_sets(st, N1, N2, ord, T, nm, (VAL), ip)) goto refused; } while (0)
#define EST_TOK_RAW(NAME, VAL) do { if (vw2_2d_rl_tok_sets(st, N1, N2, ord, T, (NAME), (VAL), ip)) goto refused; } while (0)
    if (vw2_2d_rl_bank(st, N1, N2, p->c2r, p->col.R, p->col.nst, p->wl, T > 1 ? p->cmt : -1, T > 1 ? 1 : 0,
                       est_is_pow2(N1) ? -1 : (p->col.blu > 0 ? p->col.blu : 0), 0.0, ord, T, ip) != VW2_OK)
        goto refused;
    if (p->col.forms[0] && !vw2_ilcol_forms_bank_base(st, &ck, p->col.blu > 0 ? "bluforms" : "forms", p->col.forms))
        goto refused;
    if (p->col.rd_n > 0)
    {
        est_sb_init(&s, b, sizeof b);
        est_sb_chain(&s, p->col.rd, p->col.rd_n);
        if (s.len >= sizeof b || !vw2_ilcol_forms_bank_base(st, &ck, "rd", b))
            goto refused;
        snprintf(b, sizeof b, "%d", p->col.rdw);
        if (!vw2_ilcol_forms_bank_base(st, &ck, "rdw", b))
            goto refused;
    }
    if (T > 1)
    {
        snprintf(b, sizeof b, "%d", p->mtarm); EST_TOK("mtarm", b);
        snprintf(b, sizeof b, "%d", p->msw); EST_TOK("msw", b);
    }
    if (p->rx[0]) { EST_TOK("rx", p->rx); EST_TOK("rxs", p->rxs); }
    if (p->cx[0]) { EST_TOK("cx", p->cx); EST_TOK("cxs", p->cxs); }
    if (p->tf >= 0) { snprintf(b, sizeof b, "%d", p->tf); EST_TOK("tf", b); }
    if (p->raxis >= 0) EST_TOK_RAW("raxis", p->raxis ? "n1" : "n2");
    if (p->csk >= 0) { snprintf(b, sizeof b, "%d", p->csk); EST_TOK_RAW("csk", b); }
    if (p->cxp >= 0) { snprintf(b, sizeof b, "%d", p->cxp); EST_TOK_RAW("cxp_c2r", b); }
    if (p->c2r && T == 1) EST_TOK_RAW("cxd_c2r", "off");
#undef EST_TOK
#undef EST_TOK_RAW
    if (!(S = vfft_child_store_new()))
        return est_refuse(why, whylen, "out of memory");
    if (!p->odd)
    {   /* the row batch at N2 x N1 (fft2d_create_il.h:880-925): its K = 1 inner's row at T, and at T > 1 its verdict */
        est_request_t rq;
        memset(&rq, 0, sizeof rq);
        rq.N = N2;
        rq.transform = p->c2r ? VFFT_C2R : VFFT_R2C;
        rq.order = VFFT_ORDER_NATURAL;
        rq.placement = ip ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
        rq.nthreads = T;
        rq.howmany = 1;
        if (est_put_rows_real(&S->vw2, &p->rp, &rq, 1, why, whylen))
            goto child_out;
        if (T > 1 && p->tcmt >= 0 &&
            vw2_stride_bank_tcmt(&S->vw2, p->c2r ? VW2_T_C2R : VW2_T_R2C, N2, (size_t)N1, VW2_ORD_NAT,
                                 ip ? VW2_PL_IP : VW2_PL_OOP, VW2_LAY_IL, p->tcmt, T, 0.0) != VW2_OK)
        {
            est_refuse(why, whylen, "the row batch's verdict at %d x %d was refused by its writer", N2, N1);
            goto child_out;
        }
    }
    else
    {   /* odd N2: the C2C plan at N2, out of place, natural, one thread (fft2d_create_il.h:819-879) */
        est_request_t rq;
        est__rq1(&rq, N2, VFFT_PLACEMENT_OUT);
        if (est_put_rows(&S->vw2, &p->rpc, &rq, 1, why, whylen))
            goto child_out;
    }
    vw2__2d_rl_key(&key, N1, N2, ord, T, ip);
    ok = vfft_child_row_update(st, &key, p->c2r ? "rp_c2r_" : "rp_", S);
    vfft_child_store_free(S);
    if (!ok)
        return est_refuse(why, whylen, "the row child at N2 = %d could not be put on the %dx%d row", N2, N1, N2);
    if (!strncmp(p->rx, "zr2c_r", 6))
    {   /* a zr2c row engine builds from its child's row in the engines' store (il2d_real_plan.h:495) */
        est_request_t rq;
        if (!p->has_rxe || strcmp(p->rxe.eng, "zr2c") || !(S = vfft_child_store_new()))
            return est_refuse(why, whylen, "the zr2c row engine has no recipe");
        memset(&rq, 0, sizeof rq);
        rq.N = N2;
        rq.transform = p->c2r ? VFFT_C2R : VFFT_R2C;
        rq.order = VFFT_ORDER_NATURAL;
        rq.placement = VFFT_PLACEMENT_OUT;
        rq.nthreads = 1;
        rq.howmany = 1;
        if (est_put_rows_real(&S->vw2, &p->rxe, &rq, 1, why, whylen))
            goto child_out;
        ok = vfft_child_row_update(st, &key, p->c2r ? "rx_c2r_" : "rx_", S);
        vfft_child_store_free(S);
        if (!ok)
            return est_refuse(why, whylen, "the zr2c row engine's child could not be put on the %dx%d row", N1, N2);
    }
    if (p->raxis == 1)
    {   /* the real axis on N1 (il2d_real_axis.h:221-300): rax_, the column chain at M = N1/2 down the N2 columns
         * (a C2C row of the scrambled class, one thread); raxr_, the C2C plan at N2 its folded rows' batch runs */
        vw2_ilcol_key_t rk;
        est_request_t rq;
        memset(&rk, 0, sizeof rk);
        rk.rank = 2; rk.n0 = N1 / 2; rk.n1 = N2; rk.ord = VW2_ORD_SCR; rk.nthreads = 1;
        if (!(S = vfft_child_store_new()))
            return est_refuse(why, whylen, "out of memory");
        if (est__put_col(&S->vw2, &rk, &p->rax_col, -1, 0))
        {
            vfft_child_store_free(S);
            return est_refuse(why, whylen, "the real axis's column chain at %d was refused by the store's writers", N1 / 2);
        }
        ok = vfft_child_row_update(st, &key, "rax_", S);
        vfft_child_store_free(S);
        if (!ok)
            return est_refuse(why, whylen, "the real axis's column chain could not be put on the %dx%d row", N1, N2);
        if (!(S = vfft_child_store_new()))
            return est_refuse(why, whylen, "out of memory");
        est__rq1(&rq, N2, VFFT_PLACEMENT_OUT);
        if (est_put_rows(&S->vw2, &p->raxr, &rq, 1, why, whylen))
            goto child_out;
        ok = vfft_child_row_update(st, &key, "raxr_", S);
        vfft_child_store_free(S);
        if (!ok)
            return est_refuse(why, whylen, "the real axis's row batch could not be put on the %dx%d row", N1, N2);
    }
    return 0;
child_out:
    vfft_child_store_free(S);
    return -1;
refused:
    return est_refuse(why, whylen, "the %dx%d real row was refused by the store's writers", N1, N2);
}

/* 1 when the handle serves the recipe: the column chain, the row engine, the column form, the one-thread forms only
 * where asked, the threaded verdict, the row child */
static inline int est__check_2d_real(const struct vfft_plan_s *h, const est_plan2dr_t *p, char *why, size_t whylen)
{
    const vfft_ilcol_t *c = &h->il2d_col;
    int i;
    if (h->N != p->N1 || h->N2 != p->N2 || h->ilnd)
        return !est_refuse(why, whylen, "the door built another shape than %dx%d", p->N1, p->N2);
    if (c->nst != p->col.nst)
        return !est_refuse(why, whylen, "the column pass has %d stages, the recipe %d", c->nst, p->col.nst);
    for (i = 0; i < c->nst; i++)
        if (c->R[i] != p->col.R[i])
            return !est_refuse(why, whylen, "the column chain differs from the recipe's at stage %d", i);
    if ((p->col.blu > 0) != (c->blu > 0))
        return !est_refuse(why, whylen, "the column pass's Bluestein arm differs from the recipe's");
    if (p->rx[0])
    {   /* the door's own spelling of what it bound */
        char nm[96];
        _il2d_rowx_name(h->il2d_rx_lm, h->il2d_rx_eng, nm, sizeof nm);
        if (!h->il2d_rx_on || strcmp(nm, p->rx))
            return !est_refuse(why, whylen, "the row engine is %s, the recipe's %s", h->il2d_rx_on ? nm : "unbound", p->rx);
    }
    if (p->cx[0] && !h->il2d_cx_on && !(p->c2r && !h->il2d_rscr))
        return !est_refuse(why, whylen, "the column form %s did not bind", p->cx);
    if ((h->il2d_tf_on && p->tf != 1) || (h->il2d_rax_on && p->raxis != 1) || (h->il2d_rcsk_on && p->csk != 1))
        return !est_refuse(why, whylen, "a one-thread form serves that the recipe did not ask for");
    if (p->raxis == 1 && !h->il2d_rax_on)
        return !est_refuse(why, whylen, "the real axis on N1 did not serve");
    if (p->T > 1)
    {
        if (h->nthreads != p->T)
            return !est_refuse(why, whylen, "the plan runs on %d threads, the recipe %d", h->nthreads, p->T);
        if ((c->colmt != 0) != (p->cmt != 0) || (p->cmt && (c->natarm != p->mtarm || c->msw != p->msw)))
            return !est_refuse(why, whylen, "the threaded column verdict is not the recipe's (cmt %d mtarm %d msw %d; "
                               "bound %d %d %d)", p->cmt, p->mtarm, p->msw, c->colmt, c->natarm, c->msw);
    }
    if (!h->il2d_row || (p->odd != (h->il2d_oddn2 != 0)))
        return !est_refuse(why, whylen, "the row child at N2 = %d is not the recipe's", p->N2);
    return 1;
}

/* est_build_real_2d: the recipe -> a plan built with nothing measured, or NULL with the reason in why and the
 * outcome in rep (may be NULL). cfg is the request as the front door translated it. */
static inline struct vfft_plan_s *est_build_real_2d(const est_plan2dr_t *p, const vfft_cfg_t *cfg,
                                                    est_build_report_t *rep, char *why, size_t whylen)
{
    est_build_report_t lrep;
    struct vfft_wisdom_s *S = NULL;
    struct vfft_plan_s *h = NULL;
    est__meter_t m0;
    if (!rep)
        rep = &lrep;
    memset(rep, 0, sizeof *rep);
    rep->status = EST_REFUSED;
    if (why && whylen)
        why[0] = 0;
    if (!p || !cfg || (cfg->transform != VFFT_R2C && cfg->transform != VFFT_C2R) ||
        (cfg->transform == VFFT_C2R) != (p->c2r != 0) || cfg->dims != 2 || cfg->n[0] != p->N1 || cfg->n[1] != p->N2 ||
        cfg->howmany > 1 || cfg->layout != VFI_LAYOUT_IL || (cfg->nthreads > 1 ? cfg->nthreads : 1) != p->T ||
        (cfg->placement == VFFT_INPLACE) != (p->ip != 0))
    {
        est_refuse(why, whylen, "estimate builds one 2D interleaved r2c or c2r plane at the recipe's shape, placement "
                   "and thread count");
        return NULL;
    }
    vfft_env_init();
    _vfft_create_lock();
    _vfft_create_depth++;
    est__meter_read(&m0);
    if (!(S = vfft_child_store_new()))
    {
        est_refuse(why, whylen, "out of memory");
        goto out;
    }
    if (est_put_rows_2d_real(&S->vw2, p, why, whylen))
        goto out;
    h = est__replay(S, cfg, &m0, rep, why, whylen);
    if (h && !est__check_2d_real(h, p, why, whylen))
    {
        rep->status = EST_NOPLAN;
        vfft_destroy((vfft_plan)h);
        h = NULL;
    }
    if (h)
        rep->status = EST_BUILT;
out:
    vfft_child_store_free(S);
    if (--_vfft_create_depth == 0)
        _vfft_scope_create_done();
    _vfft_create_unlock();
    return h;
}

/* ── rank 3 (il/rank3/fftnd_real_il.h: N3 the real axis, one t=r2c row for both directions, C2R's verdicts _c2r) ── */

/* the recipe of one 3D real volume */
typedef struct est_plan3dr_s
{
    int N1, N2, N3, T, c2r, ip;
    est_col_t col;            /* [chain forms blu rd rdw] axis 0 at (N1, N2 (N3/2 + 1)), the scrambled pass */
    int has_col1;             /* axis 1's pay-once chain is written (axis 0 a plain chain with a leaf kernel) */
    est_col_t col1;           /* [chain1 forms1 blu1 rd1 rdw1] axis 1 at (N2, N3/2 + 1) */
    int s, nf, nsw;           /* [s nf nsw (_c2r)] the one-thread structure (1 child) and form (1 in place, 2 strips) */
    int wl;                   /* [wl_c2r] the band's width in planes (C2R, s=3), a span of axis 0's chain */
    int cmt, cmts, cmtf;      /* [cmt cmtt cmts cmtf (_c2r)] T > 1: 0 serial | 2 the plane arm; its structure, form */
    est_plan2dr_t plane;      /* the plane child: the 2D real recipe at (N2, N3), one thread (fftnd_real_il.h:1371-1390) */
} est_plan3dr_t;

/* est_plan_3d_real: the machine, the volume and the request -> the recipe. 0, or -1 with the reason. Pure. The
 * first build carries no new rule: axis 0 and axis 1 by est_col_plan as in C2C, the child in place (the store's
 * majority, s=1 293 of 328, nf=1 282), at T > 1 est_mt_3d's bound (serial while the cube times min(N1, N2 N3, T)
 * fits 32 KiB per thread), the plane arm on the child, the strips where axis 0 is a plain chain. */
static inline int est_plan_3d_real(const est_machine_t *m, int N1, int N2, int N3, int c2r, int ip, int T,
                                   est_plan3dr_t *p, char *why, size_t whylen)
{
    const long long hpn = N3 / 2 + 1, w0 = (long long)N2 * hpn;
    const double cube = 16.0 * N1 * (double)w0;
    char dummy[1];
    if (!why || !whylen) { why = dummy; whylen = sizeof dummy; }
    why[0] = 0;
    memset(p, 0, sizeof *p);
    if (est__machine_ok(m, why, whylen))
        return -1;
    if (N1 < 2 || N2 < 2 || N3 < 2)
        return est_refuse(why, whylen, "a %dx%dx%d volume: every axis at least 2", N1, N2, N3);
    p->N1 = N1; p->N2 = N2; p->N3 = N3; p->T = T > 1 ? T : 1; p->c2r = c2r; p->ip = ip;
    if (est_col_plan(m, N1, w0, 0, &p->col, why, whylen))
        return -1;
    p->col.tpc = -1;              /* no turned prime pass on a real row (il2d_tier.h:3278-3281) */
    p->col.wl = 0;
    if (p->T == 1 && p->col.blu <= 0 && p->col.rd_n == 0 && p->col.nst >= 2)
    {   /* AXIS 0'S CHAIN BY ITS L1d SETS (2026-11-02, one thread, the banked volumes of 128 and 256 rows): the scrambled
         * pass's formula picks one chain on narrow rows and another on wide ones (est__narrow). A chain with a stage
         * that crowds an L1d set (est__chain_crowds) takes the wide rows' chain: 16.16 at 256 over 16 - 64 x 4 - 16
         * planes, 12 volumes 1.402 -> 0.977 (4.8.8), none slower; uncrowded it is right (18 volumes 0.934). On wide
         * rows C2R takes the narrow rows' chain where it crowds no set (128: 25 volumes 1.081 -> 1.007, 256: 5
         * volumes 1.098 -> 1.007, none slower), R2C at 256 only (5 volumes 1.099 -> 1.003; at 128 mixed, 8 faster
         * and 11 slower); at 128 x 256 x N its stage of 16 one row apart crowds (1.07 - 1.22 -> 1.18 - 1.51). */
        int Rn[8], Rw[8], k = 0;
        const long long wn = (long long)m->l1d / (16 * EST_ND_R16);   /* the narrow rows' bound (est__narrow) */
        const int kn = est_col_chain(m, N1, wn, 0, Rn), kw = est_col_chain(m, N1, wn + 1, 0, Rw);
        const int *R = NULL;
        if (est__chain_crowds(m, N1, p->col.R, p->col.nst, w0))
        {
            if (kw >= 2 && !est__chain_crowds(m, N1, Rw, kw, w0)) R = Rw, k = kw;
        }
        else if (!est__narrow(m, w0) && (c2r ? (N1 == 128 || N1 == 256) : N1 == 256) && kn >= 2 &&
                 !est__chain_crowds(m, N1, Rn, kn, w0))
            R = Rn, k = kn;
        if (R)
        {
            memcpy(p->col.R, R, sizeof(int) * (size_t)k);
            p->col.nst = k;
            est__col_forms(p->col.R, k, p->col.forms, sizeof p->col.forms);
        }
    }
    {   /* the pay-once pieces exist for a plain chain whose leaf is a kernel call (fftnd_real_il.h:1460) */
        const int plain = p->col.blu <= 0, rd1 = p->col.nst == 1 && p->col.rd_n > 0;
        p->has_col1 = 0;
        if (plain && !rd1 && !est_col_plan(m, N2, hpn, 0, &p->col1, why, whylen))
        {
            p->col1.tpc = -1;
            p->col1.wl = 0;
            p->has_col1 = 1;
        }
        why[0] = 0;
        p->s = 1;
        p->nf = 1;
        p->nsw = 0;
        if (plain && p->col.rd_n == 0 && 16LL * N1 <= m->l1d / 4 && cube > (c2r ? (double)m->l3 : 4.0 * m->l2))
        {   /* the strips (stage 3 study, 2026-11-02, one thread): R2C past four L2s (9 - 129 MiB, 6 volumes 1.11 - 1.34
             * -> 0.94 - 1.02; below, slower), C2R past the L3 as C2C's est_nf_3d (56 - 129 MiB 1.08 - 1.15 -> 0.98 -
             * 1.07; at 9 - 16 MiB slower); never on a Rader axis 0 (109, 331: slower); the width the door's policy
             * gives (vfft_policy_ilndr_strip_w, every banked nsw in the store) */
            p->nf = 2;
            p->nsw = vfft_policy_ilndr_strip_w(N1, (size_t)w0);
        }
        if (c2r && ip && p->T == 1 && p->has_col1 && p->nf == 1)
            /* C2R IN PLACE (2026-11-02, one thread): the pay-once structure (s=2), the planner's on all 12 in-place
             * volumes raced; 12 volumes 16^3 - 128^3 and 64 x 64 x 225 1.146 -> 0.992, 11 faster, none slower. Out of
             * place it is mixed (253 volumes, 43 faster, 59 slower) and R2C in place slower, so the child stays there */
            p->s = 2;
        if (p->T == 1 && p->nf == 1 && (N1 == 16 || N1 == 32 || N1 == 64) && p->col.nst > 1 && p->col.rd_n == 0)
        {   /* THE ONE-STAGE AXIS 0 (2026-11-02): where C2C's scrambled pass splits 16, 32 and 64, the real volume runs
             * one stage unless its N1 elements, 16 N2 (N3/2 + 1) bytes apart, crowd an L1d set: they fall into
             * 4096 / gcd(stride, 4096) of the sets, each set holds L1d / 4096 lines (12 here), so one stage while
             * N1 gcd(stride, 4096) <= L1d. The banked volumes at one thread: no crowding 16 28 volumes 1.263 -> 0.992,
             * 32 38 volumes 1.144 -> 0.974, 64 27 volumes 1.037 -> 0.906, none slower; crowding, one stage slower (16 x
             * 256 x 4 1.01 -> 1.22, 64 x 128 x 128 1.00 -> 1.25) and the planner's chain split on all 45. */
            const long long st = 16 * w0, low = st & -st;
            if ((long long)N1 * (low < 4096 ? low : 4096) <= (long long)m->l1d)
            {
                p->col.R[0] = N1;
                p->col.nst = 1;
                est__col_forms(p->col.R, 1, p->col.forms, sizeof p->col.forms);
            }
        }
        p->cmt = 0;
        p->cmts = p->s;
        p->cmtf = p->nf;
        long long mn = N1;
        if ((long long)N2 * N3 < mn) mn = (long long)N2 * N3;
        if (p->T < mn) mn = p->T;
        if (p->T > 1 && cube * (double)mn > 32768.0 * p->T)
        {   /* the plane arm on the child; R2C with a natural twin (axis 0 natural at two stages or more, or a
             * Bluestein axis) runs the child threaded only by the strips (fftnd_real_il.h:1830) */
            const int twin = !c2r && (p->col.blu > 0 || p->col.nst >= 2);
            p->cmt = 2;
            p->cmts = 1;
            /* the strips on a plain axis 0; a one-stage Rader axis 0 in place (stage 3 study, 2026-11-02, 8 threads:
             * 53, 109 and 331 both directions 1.10 - 1.27 -> 0.97 - 1.03, none slower; the serial strips' exclusion) */
            p->cmtf = (plain && !(p->col.rd_n > 0 && p->col.nst == 1)) ? 2 : 1;
            if (plain && p->col.rd_n == 0 && p->col.nst >= 2 && (N1 == 16 || N1 == 32 || N1 == 64))
            {   /* THE ONE-STAGE AXIS 0 AT T (2026-11-02): axis 0 of 16, 32 or 64 in one stage, in place (cmtf=1) --
                 * the planner's at 8 threads, where the strips have nothing to permute. 8 threads, both directions:
                 * 60 cells 1.318 -> 1.014 against the planner, 53 faster, one slower (R2C 32 x 32 x 61, 1.14 -> 1.20);
                 * with no L1d-set test, unlike one thread (crowded volumes too: 1.14 - 1.26 -> 1.00 - 1.08) */
                p->col.R[0] = N1;
                p->col.nst = 1;
                est__col_forms(p->col.R, 1, p->col.forms, sizeof p->col.forms);
                p->cmtf = 1;
            }
            else if (!c2r && plain && p->col.rd_n == 0 && p->col.nst >= 2 && p->has_col1 && m->l3 > 0 &&
                     cube <= (double)m->l3 / 3)
            {   /* THE STAGE-WISE PAY-ONCE AT T (2026-11-02): R2C over a plain axis 0 of two stages or more, the volume
                 * within a third of the L3, the pay-once structure dealt by stage (cmts=2 cmtf=3, fftnd_real_il.h nf=3),
                 * the planner's on such volumes. 8 threads: 19 volumes 1.235 -> 0.995, none slower; from 0.3 to 9 MiB it
                 * won at every size, from 16 MiB it lost (128 x 128 x 128 1.03 -> 1.40, past the L3 ~1.8) */
                p->cmts = 2;
                p->cmtf = 3;
            }
            else if (twin && p->cmtf != 2)
                p->cmt = 0, p->cmts = p->s, p->cmtf = p->nf;
        }
    }
    if (est_plan_2d_real(m, N2, N3, c2r, ip && !c2r, 1, &p->plane, why, whylen))
        return -1;
    return 0;
}

/* est_put_rows_3d_real: the volume's row into st, the requested direction's verdicts, the plane child under
 * plane_. 0, or -1 with the reason. */
static inline int est_put_rows_3d_real(vw2_store_t *st, const est_plan3dr_t *p, char *why, size_t whylen)
{
    const char *sfx = p->c2r ? "_c2r" : "";
    vw2_ilcol_key_t ck;
    vw2_key_t key;
    struct vfft_wisdom_s *S;
    char nm[32];
    int ok;
    memset(&ck, 0, sizeof ck);
    ck.rank = 3; ck.n0 = p->N1; ck.n1 = p->N2; ck.n2 = p->N3;
    ck.ord = VW2_ORD_NAT; ck.axis = 0; ck.real = 1; ck.nthreads = p->T; ck.ip = p->ip;
#define EST_TOKI(BASE, VAL) do { snprintf(nm, sizeof nm, "%s%s", (BASE), sfx); \
        if (vw2_ilcol_tok_seti(st, &ck, nm, (VAL)) != VW2_OK) goto refused; } while (0)
    if (est__put_col(st, &ck, &p->col, -1, 0))
        goto refused;
    EST_TOKI("s", p->s);
    EST_TOKI("nf", p->nf);
    if (p->nf == 2)
        EST_TOKI("nsw", p->nsw);
    if (p->s == 3)
        EST_TOKI("wl", p->wl);
    if (p->T > 1)
    {
        EST_TOKI("cmt", p->cmt);
        EST_TOKI("cmtt", p->T);
        EST_TOKI("cmts", p->cmts);
        EST_TOKI("cmtf", p->cmtf);
    }
#undef EST_TOKI
    if (p->has_col1)
    {
        vw2_ilcol_key_t ck1 = ck;
        ck1.axis = 1;
        if (est__put_col(st, &ck1, &p->col1, -1, 0))
            goto refused;
    }
    if (!(S = vfft_child_store_new()))
        return est_refuse(why, whylen, "out of memory");
    if (est_put_rows_2d_real(&S->vw2, &p->plane, why, whylen))
    {
        vfft_child_store_free(S);
        return -1;
    }
    vw2__ilcol_key(&ck, &key);
    ok = vfft_child_row_update(st, &key, "plane_", S);
    vfft_child_store_free(S);
    if (!ok)
        return est_refuse(why, whylen, "the plane child %dx%d could not be put on the %dx%dx%d row", p->N2, p->N3, p->N1,
                          p->N2, p->N3);
    return 0;
refused:
    return est_refuse(why, whylen, "the %dx%dx%d real row was refused by the store's writers", p->N1, p->N2, p->N3);
}

/* 1 when the handle serves the recipe: axis 0's chain, the structure and form, the threaded verdict, the plane child
 * as the 2D recipe */
static inline int est__check_3d_real(const struct vfft_plan_s *h, const est_plan3dr_t *p, char *why, size_t whylen)
{
    const struct vfft_ilndr_s *d = h->ilndr;
    const struct vfft_plan_s *c;
    int i;
    if (!d || d->N[0] != p->N1 || d->N[1] != p->N2 || d->N[2] != p->N3)
        return !est_refuse(why, whylen, "the door built another plan than the %dx%dx%d real volume", p->N1, p->N2, p->N3);
    if (d->ax0.nst != p->col.nst || ((p->col.blu > 0) != (d->ax0.blu > 0)))
        return !est_refuse(why, whylen, "axis 0 is not the recipe's chain");
    for (i = 0; i < d->ax0.nst; i++)
        if (d->ax0.R[i] != p->col.R[i])
            return !est_refuse(why, whylen, "axis 0's chain differs from the recipe's at stage %d", i);
    if (d->arm != p->s || d->nf != p->nf || (p->s == 3 && d->wl != p->wl))
        return !est_refuse(why, whylen, "the structure is s=%d nf=%d, the recipe's s=%d nf=%d", d->arm, d->nf, p->s, p->nf);
    if (p->T > 1 && (d->mt != p->cmt || (p->cmt == 2 && (d->mts != p->cmts || d->mtf != p->cmtf))))
        return !est_refuse(why, whylen, "the threaded verdict is cmt=%d cmts=%d cmtf=%d, the recipe's %d %d %d", d->mt,
                           d->mts, d->mtf, p->cmt, p->cmts, p->cmtf);
    c = p->c2r ? d->cb : d->cf;
    if (!c)
        return !est_refuse(why, whylen, "no plane child");
    return est__check_2d_real(c, &p->plane, why, whylen);
}

/* est_build_real_3d: the recipe -> a plan built with nothing measured, or NULL with the reason in why and the
 * outcome in rep (may be NULL). cfg is the request as the front door translated it. */
static inline struct vfft_plan_s *est_build_real_3d(const est_plan3dr_t *p, const vfft_cfg_t *cfg,
                                                    est_build_report_t *rep, char *why, size_t whylen)
{
    est_build_report_t lrep;
    struct vfft_wisdom_s *S = NULL;
    struct vfft_plan_s *h = NULL;
    est__meter_t m0;
    if (!rep)
        rep = &lrep;
    memset(rep, 0, sizeof *rep);
    rep->status = EST_REFUSED;
    if (why && whylen)
        why[0] = 0;
    if (!p || !cfg || (cfg->transform != VFFT_R2C && cfg->transform != VFFT_C2R) ||
        (cfg->transform == VFFT_C2R) != (p->c2r != 0) || cfg->dims != 3 || cfg->n[0] != p->N1 || cfg->n[1] != p->N2 ||
        cfg->n[2] != p->N3 || cfg->howmany > 1 || cfg->layout != VFI_LAYOUT_IL ||
        (cfg->nthreads > 1 ? cfg->nthreads : 1) != p->T || (cfg->placement == VFFT_INPLACE) != (p->ip != 0))
    {
        est_refuse(why, whylen, "estimate builds one 3D interleaved r2c or c2r volume at the recipe's shape, placement "
                   "and thread count");
        return NULL;
    }
    vfft_env_init();
    _vfft_create_lock();
    _vfft_create_depth++;
    est__meter_read(&m0);
    if (!(S = vfft_child_store_new()))
    {
        est_refuse(why, whylen, "out of memory");
        goto out;
    }
    if (est_put_rows_3d_real(&S->vw2, p, why, whylen))
        goto out;
    h = est__replay(S, cfg, &m0, rep, why, whylen);
    if (h && !est__check_3d_real(h, p, why, whylen))
    {
        rep->status = EST_NOPLAN;
        vfft_destroy((vfft_plan)h);
        h = NULL;
    }
    if (h)
        rep->status = EST_BUILT;
out:
    vfft_child_store_free(S);
    if (--_vfft_create_depth == 0)
        _vfft_scope_create_done();
    _vfft_create_unlock();
    return h;
}

/* THE FRONT DOOR'S ESTIMATE CREATE OF A REAL TRANSFORM AT RANK 2 OR 3 (_vfft_create_estimate): the request, the
 * recipe, the build. NULL with the reason on the error record (vfft_error). */
static vfft_plan _vfft_create_estimate_real_nd(const est_machine_t *m, const vfft_cfg_t *cfg)
{
    char why[640];
    struct vfft_plan_s *h;
    const int T = cfg->nthreads > 1 ? cfg->nthreads : 1, c2r = cfg->transform == VFFT_C2R;
    const int ip = cfg->placement == VFFT_INPLACE;
    if ((cfg->dims != 2 && cfg->dims != 3) || cfg->howmany > 1)
    {
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE plans one R2C or C2R transform at rank 1 to 3");
        return NULL;
    }
    if (cfg->owned_buffers)
    {
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE plans a 2D or 3D real transform at the caller's pitch, not with "
                   "VFFT_FLAG_OWN_PITCH (drop one of the two)");
        return NULL;
    }
    if (cfg->dims == 2)
    {
        est_plan2dr_t p;
        if (est_plan_2d_real(m, cfg->n[0], cfg->n[1], c2r, ip, T, &p, why, sizeof why))
        {
            _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE: %s", why);
            return NULL;
        }
        h = est_build_real_2d(&p, cfg, NULL, why, sizeof why);
    }
    else
    {
        est_plan3dr_t *p = (est_plan3dr_t *)malloc(sizeof *p);   /* two 1D real recipes and a 2D one: off the stack */
        if (!p)
        {
            _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE: out of memory");
            return NULL;
        }
        if (est_plan_3d_real(m, cfg->n[0], cfg->n[1], cfg->n[2], c2r, ip, T, p, why, sizeof why))
        {
            free(p);
            _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE: %s", why);
            return NULL;
        }
        h = est_build_real_3d(p, cfg, NULL, why, sizeof why);
        free(p);
    }
    if (!h)
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE: %s", why);
    return (vfft_plan)h;
}

#endif /* VFFT_IL_REAL_ESTIMATE_ND_IL_H */
