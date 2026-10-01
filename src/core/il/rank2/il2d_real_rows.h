/* il2d_real_rows.h — the 2D real tier's ROW PLAN (r2c) and its planner.
 *
 * The row pass of a real plane is the plan's own: which engine transforms a
 * row is decided by THIS plan's race, in the row role (the N1 rows of the
 * plane, read where they sit), and banked on the 2D real row -- never taken
 * from the 1D cell's verdict, which was raced on one L1-hot row.
 *
 * THE ENGINES
 *   lm     the rows kernel r2zr (codelets/zil/<isa>/real/rows/): the real
 *          N2-point DFT of every row, a lane a row, the whole pass in one
 *          call. Even N2 <= 32.
 *   zrm    the real mono per row            (il/real/zrm.h)
 *   zrp    the real pair per row            (il/real/zrp.h), a pair and a form
 *   zr2c   the zr2c composite per row       (il/real/zr2c_build.h), a route
 *   zttr   ZTT-r per row                    (il/real/zttr.h), chain/tile/stack
 *   door   the row route the tier had (the per-row door, or the rowsplit band)
 * The per-row engines are built by the real door's BUILDERS from the row's
 * token; the handle is the plan's own.
 *
 * THE STACK STATE. Win64 gives a callee a 16-B stack and mingw never
 * realigns a frame, so a spilling kernel under the row loop runs at one of
 * several speeds by the caller's rsp (measured 2026-10-01, the zr2c composite
 * at 16x1024: 534-539 ns a row at two residues, 591-601 at the other two).
 * The row pass is entered through the stack-aligning call (zttr.h's form) at
 * a residue that is plan input; the race times every engine at every state,
 * so no engine is judged at a state it would not be run at.
 *
 * WISDOM. Two tokens on the shared real IL row, r2c's own:
 *   rx=   door | lm | zrm | zrp_R1.R2_a|b | zr2c_r0|r1 | zttr_<chain>_t<tile>_s<stk>
 *   rxs=  the row pass's stack state 0..3
 * VFFT_IL2D_RX=<rx>[/s<state>] pins (beats the bank, never banks);
 * VFFT_IL2D_RX=off leaves the row route unbound (no aligned entry).
 *
 * One thread, even N2. A threaded plan keeps the per-row door's slabs; odd
 * N2 its c2c child; c2r its own row pass (the backward is a separate piece).
 *
 * Included after il/real/zrp_build.h (the builders) and il2d_tier.h (the row
 * route, and the dispatcher that calls _il2d_rowx_fwd). */
#ifndef VFFT_IL2D_REAL_ROWS_H
#define VFFT_IL2D_REAL_ROWS_H

/* the rows kernel at N2 (the generated registry's list) */
static inline vfft_il2p_fn vfft_il2d_rows_fn(int R)
{
    switch (R) {
#ifdef VFFT_IL_R2ZR_FWD_RADICES
#define C(R) case R: return VFFT_IL_SYM(radix##R##_z_r2zr_fwd);
    VFFT_IL_R2ZR_FWD_RADICES(C)
#undef C
#endif
    default: return 0;
    }
}

/* the pass itself: the rows kernel in one call, an engine row by row, or the
 * tier's row route */
static void _il2d_rowx_body(const struct vfft_plan_s *h, const double *sre, double *dre)
{
    const size_t rn2 = (size_t)h->N2, hp1 = rn2 / 2 + 1, n1 = (size_t)h->N;
    size_t r;
    if (h->il2d_rx_lm)
    {
        h->il2d_rx_lm(sre, NULL, dre, NULL, NULL, NULL, rn2, 0, hp1, 0, n1);
        return;
    }
    if (!h->il2d_rx_eng)
    {
        _il2d_real_rows_fwd_route((struct vfft_plan_s *)h, sre, dre);
        return;
    }
    for (r = 0; r < n1; r++)
        _real_il_exec_any(h->il2d_rx_eng, sre + r * rn2, dre + r * 2 * hp1);
}

/* THE STACK-ALIGNING ENTRY (zttr.h's): rsp set to the chosen residue mod 64
 * before the call, so every frame under the row pass sits at one state
 * whatever the caller's stack. Win64 call: args in rcx, rdx, r8; 32 B of
 * shadow space; r12 callee-saved. */
typedef void (*_il2d_rowx_fn)(const struct vfft_plan_s *, const double *, double *);
#if defined(_WIN64) && defined(__GNUC__) && defined(__x86_64__)
static inline void _il2d_rowx_call(_il2d_rowx_fn fn, const struct vfft_plan_s *h, const double *s, double *d, int stk)
{
    register const struct vfft_plan_s *a0 __asm__("rcx") = h;
    register const double *a1 __asm__("rdx") = s;
    register double *a2 __asm__("r8") = d;
    register _il2d_rowx_fn f __asm__("r10") = fn;
    register long ad __asm__("r11") = 32 + 16 * (long)(stk & 3);
    __asm__ volatile(
        "movq %%rsp, %%r12\n\t"
        "andq $-64, %%rsp\n\t"
        "subq %%r11, %%rsp\n\t"
        "call *%%r10\n\t"
        "movq %%r12, %%rsp\n\t"
        : "+r"(a0), "+r"(a1), "+r"(a2), "+r"(f), "+r"(ad)
        :
        : "rax", "r9", "r12", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7",
          "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15", "memory", "cc");
}
#else
static inline void _il2d_rowx_call(_il2d_rowx_fn fn, const struct vfft_plan_s *h, const double *s, double *d, int stk)
{
    (void)stk;
    fn(h, s, d);
}
#endif

static void _il2d_rowx_fwd(struct vfft_plan_s *h, const double *sre, double *dre)
{
    _il2d_rowx_call(_il2d_rowx_body, h, sre, dre, h->il2d_rx_stk);
}

/* the K = 1 request the per-row engines are built for */
static void _il2d_rowx_cfg(const vfft_config_t *cfg, int N2, vfft_config_t *c)
{
    memset(c, 0, sizeof *c);
    c->transform = VFFT_R2C;
    c->placement = VFFT_OUTOFPLACE;
    c->rigor = cfg->rigor;
    c->dims = 1;
    c->n[0] = N2;
    c->howmany = 1;
    c->layout = VFFT_LAYOUT_INTERLEAVED;
    c->nthreads = 1;
    c->wisdom = cfg->wisdom;
    c->wisdom_write = cfg->wisdom_write;
}

/* an engine's token */
static void _il2d_rowx_name(vfft_il2p_fn lm, const struct vfft_plan_s *e, char *b, size_t n)
{
    if (lm)
        snprintf(b, n, "lm");
    else if (!e)
        snprintf(b, n, "door");
    else if (e->zrm)
        snprintf(b, n, "zrm");
    else if (e->zrp)
        snprintf(b, n, "zrp_%d.%d_%c", e->zrp->R1, e->zrp->R2, e->zrp->form ? 'b' : 'a');
    else if (e->zttr)
    {
        char cs[32];
        vfft_ztt_chain_str(e->zttr->zt, cs, sizeof cs);
        snprintf(b, n, "zttr_%s_t%zu_s%d", cs, e->zttr->zt->tile, e->zttr->stk);
    }
    else
        snprintf(b, n, "zr2c_r%d", e->zr2c_route);
}

/* a token's engine: 1 = built (door: nothing to build), 0 = it does not build here */
static int _il2d_rowx_build(const vfft_config_t *cfg, int N1, int N2, const char *tok,
                            vfft_il2p_fn *lm, struct vfft_plan_s **eng)
{
    vfft_config_t c;
    *lm = NULL;
    *eng = NULL;
    if (!tok || !tok[0] || !strcmp(tok, "door"))
        return 1;
    if (!strcmp(tok, "lm"))
    {
        *lm = N1 >= 2 ? vfft_il2d_rows_fn(N2) : NULL;
        return *lm != NULL;
    }
    _il2d_rowx_cfg(cfg, N2, &c);
    if (!strcmp(tok, "zrm"))
        *eng = N2 <= VFFT_ZRM_MAX_N ? _zrm_build_plan(&c, N2) : NULL;
    else if (!strncmp(tok, "zrp_", 4))
    {
        int R1 = 0, R2 = 0;
        char f = 0;
        if (sscanf(tok + 4, "%d.%d_%c", &R1, &R2, &f) == 3 && (f == 'a' || f == 'b') &&
            R1 > 0 && R2 > 0 && R1 * R2 == N2 && vfft_zrp_pair_ok(N2, R1, R2, f == 'b'))
            *eng = _zrp_build_pair(&c, N2, R1, R2, f == 'b');
    }
    else if (!strncmp(tok, "zr2c_r", 6) && (tok[6] == '0' || tok[6] == '1') && !tok[7])
        *eng = _zr2c_build_route(&c, N2, tok[6] == '1');
    else if (!strncmp(tok, "zttr_", 5))
    {
        int chain[8], nf = 0, stk = 3;
        unsigned long tile = 0;
        const char *q = tok + 5;
        while (*q && nf < 8)
        {
            char *end;
            long v = strtol(q, &end, 10);
            if (end == q)
                break;
            chain[nf++] = (int)v;
            q = end;
            if (*q == '.')
                q++;
            else
                break;
        }
        if (nf >= 2 && q[0] == '_' && q[1] == 't')
        {
            char *end;
            tile = strtoul(q + 2, &end, 10);
            if (end[0] == '_' && end[1] == 's')
                stk = atoi(end + 2);
            *eng = _zttr_build_plan(&c, N2, chain, nf, (size_t)tile, stk);
        }
    }
    return *eng != NULL;
}

/* the race's arm: one candidate at one stack state installed on the plan, the row pass run */
typedef struct
{
    struct vfft_plan_s *h;
    const double *a;
    double *z;
    vfft_il2p_fn lm;
    struct vfft_plan_s *eng;
    int stk;
} _il2d_rowx_arm_t;
static void _il2d_rowx_arm_run(void *v)
{
    _il2d_rowx_arm_t *c = (_il2d_rowx_arm_t *)v;
    c->h->il2d_rx_lm = c->lm;
    c->h->il2d_rx_eng = c->eng;
    c->h->il2d_rx_stk = c->stk;
    c->h->il2d_rx_on = 1;
    _il2d_real_rows_fwd(c->h, c->a, c->z);
}

/* bank the verdict on the shared real IL row; the row is made when the cell
 * has none yet (the served chain, no row-axis tokens: fft2d_create_il.h's
 * forms bank does the same) */
static void _il2d_rowx_bank(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                            int N1, int N2, int ord, int T, const char *name, int stk)
{
    char sb[8];
    int ok;
    if (!W || W->vw2_off_2d)
        return;
    snprintf(sb, sizeof sb, "%d", stk & 3);
    ok = vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, T, "rx", name) == 0;
    if (!ok)
    {
        vw2_2d_rl_bank(&W->vw2, N1, N2, 0, h->il2d_col.R, h->il2d_col.nst, -1, -1, -1, 0,
                       (N1 & (N1 - 1)) ? h->il2d_col.blu : -1, 0.0, ord, T);
        ok = vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, T, "rx", name) == 0;
    }
    if (ok)
        ok = vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, T, "rxs", sb) == 0;
    if (ok)
        _vw2_persist(W, cfg);
    else
        fprintf(stderr, "vfft: the 2D real row plan NOT banked at %dx%d -- the cell will re-race\n", N1, N2);
}

/* THE ROW PLAN: env pin, the banked rx=, or the race. Runs after the plan's
 * stage arrays are committed (it executes the row pass of h). */
static void _il2d_real_rowplan(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                               int N1, int N2, int ord, int T)
{
    enum { NARMS = VFFT_RACE_MAX_ARMS / 4 };
    const size_t hp1 = (size_t)N2 / 2 + 1, RN = (size_t)N1 * (size_t)N2, CN = 2 * (size_t)N1 * hp1;
    const char *log = getenv("VFFT_IL2D_LOG");
    vfft_il2p_fn lm = NULL;
    struct vfft_plan_s *eng = NULL;
    h->il2d_rx_lm = NULL;
    h->il2d_rx_eng = NULL;
    h->il2d_rx_stk = 0;
    h->il2d_rx_on = 0;
    {   /* 1. env: the racing hook. Beats wisdom, never banks. */
        const char *e = getenv("VFFT_IL2D_RX");
        if (e && e[0])
        {
            char tok[96];
            const char *sl = strchr(e, '/');
            size_t n = sl ? (size_t)(sl - e) : strlen(e);
            if (!strcmp(e, "off"))
                return;
            if (n >= sizeof tok)
                n = sizeof tok - 1;
            memcpy(tok, e, n);
            tok[n] = 0;
            if (_il2d_rowx_build(cfg, N1, N2, tok, &lm, &eng))
            {
                h->il2d_rx_lm = lm;
                h->il2d_rx_eng = eng;
                h->il2d_rx_stk = (sl && sl[1] == 's') ? atoi(sl + 2) & 3 : 0;
                h->il2d_rx_on = 1;
            }
            else
                _vfft_warn("vfft_create: VFFT_IL2D_RX=%s does not build at %dx%d (the row route stands)", e, N1, N2);
            return;
        }
    }
    if (!W || W->vw2_off_2d)
        return;
    if (!cfg->recalibrate)
    {   /* 2. the banked verdict */
        const char *tok = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, "rx");
        if (tok)
        {
            const char *sv = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, "rxs");
            if (_il2d_rowx_build(cfg, N1, N2, tok, &lm, &eng))
            {
                h->il2d_rx_lm = lm;
                h->il2d_rx_eng = eng;
                h->il2d_rx_stk = sv ? atoi(sv) & 3 : 0;
                h->il2d_rx_on = 1;
                return;
            }
            /* a banked engine that no longer builds: the race */
        }
    }
    /* 3. the race, in the row role: every engine at every stack state */
    {
        _il2d_rowx_arm_t ctx[4 * NARMS];
        vfft_race_arm_t arms[4 * NARMS];
        char names[NARMS][48], xn[4 * NARMS][56];
        double ns[4 * NARMS];
        _il2d_rowx_arm_t cand[NARMS];
        vfft_config_t c;
        int na = 0, best = 0, bd = 0, i, s;
        double *a = (double *)vfft_aligned_alloc((RN + 8) * sizeof(double));
        double *z = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
        double *ref = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
        if (!a || !z || !ref)
        {
            vfft_aligned_free(a); vfft_aligned_free(z); vfft_aligned_free(ref);
            return;
        }
        {
            unsigned sd = 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12);
            size_t j;
            for (j = 0; j < RN + 8; j++)
            {
                sd = sd * 1664525u + 1013904223u;
                a[j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
            }
        }
        _il2d_rowx_cfg(cfg, N2, &c);
#define ROWX_ARM(LM, ENG) do { if (na < NARMS) { \
            cand[na].h = h; cand[na].a = a; cand[na].z = z; cand[na].lm = (LM); cand[na].eng = (ENG); cand[na].stk = 0; \
            _il2d_rowx_name(cand[na].lm, cand[na].eng, names[na], sizeof names[na]); na++; } \
            else if ((ENG) != NULL) vfft_destroy((vfft_plan)(ENG)); } while (0)
        ROWX_ARM(NULL, NULL);   /* arm 0: the row route the tier has */
        if (N1 >= 2 && vfft_il2d_rows_fn(N2))
            ROWX_ARM(vfft_il2d_rows_fn(N2), NULL);
        {
            struct vfft_plan_s *hz = _zr2c_build_route(&c, N2, 0), *e;
            if (hz)
                ROWX_ARM(NULL, hz);
            if ((e = _zr2c_build_route(&c, N2, 1)) != NULL)
                ROWX_ARM(NULL, e);
            if (N2 <= VFFT_ZRM_MAX_N && (e = _zrm_build_plan(&c, N2)) != NULL)
                ROWX_ARM(NULL, e);
            {
                int pa[VFFT_ZRP_MAX_ARMS][3];
                const int np = _zrp_arms(N2, pa, VFFT_ZRP_MAX_ARMS);
                for (i = 0; i < np; i++)
                    if ((e = _zrp_build_pair(&c, N2, pa[i][0], pa[i][1], pa[i][2])) != NULL)
                        ROWX_ARM(NULL, e);
            }
            if (hz && N2 >= 64)
            {   /* ZTT-r's shortlist, as the door sweeps it (gated on zr2c's row) */
                const size_t xs = (size_t)N2 + 2;
                double *rb = (double *)vfft_aligned_alloc(xs * sizeof(double));
                double *rr = (double *)vfft_aligned_alloc(xs * sizeof(double));
                if (rb && rr)
                {
                    struct vfft_plan_s *ht[VFFT_ZTTR_MAX_ARMS];
                    int nt;
                    _exec_zr2c(hz, a, rr);
                    nt = _zttr_sweep(&c, N2, a, rr, rb, a, xs, xs, ht);
                    for (i = 0; i < nt; i++)
                        ROWX_ARM(NULL, ht[i]);
                }
                vfft_aligned_free(rb); vfft_aligned_free(rr);
            }
        }
#undef ROWX_ARM
        /* the gate: every arm's plane against the tier's own row route */
        memset(ref, 0, (CN + 8) * sizeof(double));
        cand[0].z = ref;
        _il2d_rowx_arm_run(&cand[0]);
        cand[0].z = z;
        {
            int keep = 1;
            for (i = 1; i < na; i++)
            {
                double e;
                memset(z, 0, CN * sizeof(double));
                _il2d_rowx_arm_run(&cand[i]);
                e = _zrpr_relerr(z, ref, CN);
                if (e < 1e-10)
                {
                    if (keep != i)
                    {
                        cand[keep] = cand[i];
                        memcpy(names[keep], names[i], sizeof names[keep]);
                    }
                    keep++;
                }
                else
                {
                    fprintf(stderr, "[il2d-real] rows %dx%d: %s FAILS the gate (rel %.2e) -- dropped\n", N1, N2, names[i], e);
                    if (cand[i].eng)
                        vfft_destroy((vfft_plan)cand[i].eng);
                }
            }
            na = keep;
        }
        for (i = 0; i < na; i++)
            for (s = 0; s < 4; s++)
            {
                const int x = 4 * i + s;
                ctx[x] = cand[i];
                ctx[x].stk = s;
                snprintf(xn[x], sizeof xn[x], "%s/s%d", names[i], s);
                arms[x].name = xn[x]; arms[x].run = _il2d_rowx_arm_run; arms[x].ctx = &ctx[x];
            }
        {
            double t0, est;
            int reps;
            _vfft_create_race_count++;
            _il2d_rowx_arm_run(&ctx[0]);
            t0 = vfft_now_ns();
            _il2d_rowx_arm_run(&ctx[0]);
            est = vfft_now_ns() - t0;
            reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
            if (reps < 1) reps = 1;
            if (reps > 4096) reps = 4096;
            {   /* 9 rounds alternated, median */
                const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
                vfft_race_run(&proto, arms, 4 * na, ns);
            }
            for (i = 1; i < 4 * na; i++)
                if (ns[i] < ns[best]) best = i;
            for (s = 1; s < 4; s++)
                if (ns[s] < ns[bd]) bd = s;
            /* 3% hysteresis toward the tier's route at its own best state */
            if (best / 4 != 0 && !vfft_race_beats(ns[best], ns[bd], 0.97))
                best = bd;
            if (log)
            {
                fprintf(stderr, "[il2d-real] rows %dx%d row plan race: reps=%d |", N1, N2, reps);
                for (i = 0; i < na; i++)
                {
                    int bs = 0;
                    for (s = 1; s < 4; s++)
                        if (ns[4 * i + s] < ns[4 * i + bs]) bs = s;
                    fprintf(stderr, " %s=%.0f(s%d;%.0f/%.0f/%.0f/%.0f)", names[i], ns[4 * i + bs], bs,
                            ns[4 * i], ns[4 * i + 1], ns[4 * i + 2], ns[4 * i + 3]);
                }
                fprintf(stderr, " -> %s\n", xn[best]);
            }
        }
        for (i = 1; i < na; i++)
            if (i != best / 4 && cand[i].eng)
                vfft_destroy((vfft_plan)cand[i].eng);
        h->il2d_rx_lm = ctx[best].lm;
        h->il2d_rx_eng = ctx[best].eng;
        h->il2d_rx_stk = ctx[best].stk;
        h->il2d_rx_on = 1;
        _il2d_rowx_bank(h, W, cfg, N1, N2, ord, T, names[best / 4], ctx[best].stk);
        vfft_aligned_free(a); vfft_aligned_free(z); vfft_aligned_free(ref);
    }
}

#endif /* VFFT_IL2D_REAL_ROWS_H */
