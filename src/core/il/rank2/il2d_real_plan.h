/* il2d_real_plan.h — the 2D real tier's PASS PLANS and their planners, per
 * direction: the ROW PLAN (below) and the COLUMN PLAN (at the end of the
 * file), each the r2c plan and its c2r twin (2026-10-05).
 *
 * The row pass of a real plane is the plan's own: which engine transforms a
 * row is decided by THIS plan's race, in the row role (the N1 rows of the
 * plane, read where they sit), and banked on the 2D real row -- never taken
 * from the 1D cell's verdict, which was raced on one L1-hot row.
 *
 * THE ENGINES
 *   lm     the rows kernel r2zr (codelets/zil/<isa>/real/rows/): the real
 *          N2-point DFT of every row, a lane a row, the whole pass in one
 *          call; its backward for c2r (the CCE bins of every row -> its
 *          samples). Even N2 <= 32.
 *   zrm    the real mono per row            (il/real/zrm.h)
 *   zrp    the real pair per row            (il/real/zrp.h), a pair and a form
 *   zr2c   the zr2c composite per row       (il/real/zr2c_build.h), a route
 *   zttr   ZTT-r per row                    (il/real/zttr.h), chain/tile/stack
 *   door   the row route the tier had (the per-row door)
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
 *   rxs=  the row pass's stack state 0..3, or `any`: the four states raced
 *         within 3% (the kernels do not spill), the pass entered directly
 * VFFT_IL2D_RX=<rx>[/s<state>] pins (beats the bank, never banks);
 * VFFT_IL2D_RX=off leaves the row route unbound (no aligned entry).
 * THE C2R TWIN (2026-10-05) is the same plan over the backward row pass (the
 * column-inverse plane's CCE rows -> the real rows): the engines run their
 * backward per row (the same builders, a c2r request), the rows kernel is
 * r2zr's backward, the door is the c2r row route; raced in the row role on a
 * CCE plane, banked as rx_c2r= / rxs_c2r= on the same row (its own engine
 * store, rx_c2r_*), pinned by VFFT_IL2D_RX_C2R. Each direction's plan is its
 * own: an r2c verdict is never read by a c2r plan, nor the reverse.
 *
 * The column plan's forms and its per-kernel stack states are described at
 * its section; the destroying c2r (a request's destroy_input permission: the
 * one-kernel column pass in place on the caller's plane) at the last one.
 *
 * One thread, even N2. A threaded plan keeps the per-row door's slabs; odd
 * N2 its c2c child.
 *
 * Included after il/real/zrp_build.h (the builders) and il2d_tier.h (the row
 * route, and the dispatcher that calls _il2d_rowx_fwd). */
#ifndef VFFT_IL2D_REAL_PLAN_H
#define VFFT_IL2D_REAL_PLAN_H

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
/* its backward (the c2r row pass's kernel) */
static inline vfft_il2p_fn vfft_il2d_rows_bwd_fn(int R)
{
    switch (R) {
#ifdef VFFT_IL_R2ZR_BWD_RADICES
#define C(R) case R: return VFFT_IL_SYM(radix##R##_z_r2zr_bwd);
    VFFT_IL_R2ZR_BWD_RADICES(C)
#undef C
#endif
    default: return 0;
    }
}

/* the pass itself: the rows kernel in one call, an engine row by row, or the
 * tier's row route */
static void _il2d_rowx_body(const void *v, const double *sre, double *dre)
{
    const struct vfft_plan_s *h = (const struct vfft_plan_s *)v;
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
/* the c2r pass: the backward rows kernel in one call, an engine's backward
 * row by row (the engines run their plan's direction), or the c2r row route */
static void _il2d_rowx_body_bwd(const void *v, const double *zsrc, double *dre)
{
    const struct vfft_plan_s *h = (const struct vfft_plan_s *)v;
    const size_t rn2 = (size_t)h->N2, hp1 = rn2 / 2 + 1, n1 = (size_t)h->N;
    size_t r;
    if (h->il2d_rx_lm)
    {
        h->il2d_rx_lm(zsrc, NULL, dre, NULL, NULL, NULL, hp1, 0, rn2, 0, n1);
        return;
    }
    if (!h->il2d_rx_eng)
    {
        _il2d_real_rows_bwd_route((struct vfft_plan_s *)h, zsrc, dre);
        return;
    }
    for (r = 0; r < n1; r++)
        _real_il_exec_any(h->il2d_rx_eng, zsrc + r * 2 * hp1, dre + r * rn2);
}

/* THE STACK-ALIGNING ENTRY (zttr.h's): rsp set to the chosen residue mod 64
 * before the call, so every frame under the row pass sits at one state
 * whatever the caller's stack. Win64 call: args in rcx, rdx, r8; 32 B of
 * shadow space; r12 callee-saved. The first argument is the callee's own
 * context: the plan (a pass), or a stage of it (the column plan's per-kernel
 * entries). The Win64 ABI under GCC-style inline asm only, as zttr.h's;
 * elsewhere the pass is entered directly and the stack state binds nothing. */
typedef void (*_il2d_rowx_fn)(const void *, const double *, double *);
#if defined(_WIN64) && defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
static inline void _il2d_rowx_call(_il2d_rowx_fn fn, const void *h, const double *s, double *d, int stk)
{
    register const void *a0 __asm__("rcx") = h;
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
static inline void _il2d_rowx_call(_il2d_rowx_fn fn, const void *h, const double *s, double *d, int stk)
{
    (void)stk;
    fn(h, s, d);
}
#endif

/* the stack state to bind for a raced winner: its best state, or -1 ("any")
 * when its four states sit within 3%, or within 1 ns -- the entry's own cost
 * (measured 2026-10-02, 2x4), under which no state can pay for it. Nothing
 * under such a pass spills, and the pass is entered directly. */
static int _il2d_plan_stk(const double *ns4, int best)
{
    double mn = ns4[0], mx = ns4[0];
    int s;
    for (s = 1; s < 4; s++)
    {
        if (ns4[s] < mn) mn = ns4[s];
        if (ns4[s] > mx) mx = ns4[s];
    }
    return ((mx - mn) < 0.03 * mn || (mx - mn) < 1.0) ? -1 : best;
}
static int _il2d_plan_stk_parse(const char *v)
{
    if (!v || !v[0])
        return 0;
    if (!strcmp(v, "any"))
        return -1;
    return atoi(v[0] == 's' ? v + 1 : v) & 3;
}
static void _il2d_plan_stk_str(int stk, char *buf, size_t cap)
{
    if (stk < 0)
        snprintf(buf, cap, "any");
    else
        snprintf(buf, cap, "%d", stk & 3);
}

static void _il2d_rowx_fwd(struct vfft_plan_s *h, const double *sre, double *dre)
{
    if (h->il2d_rx_stk < 0)
        _il2d_rowx_body(h, sre, dre);
    else
        _il2d_rowx_call(_il2d_rowx_body, h, sre, dre, h->il2d_rx_stk);
}
static void _il2d_rowx_bwd(struct vfft_plan_s *h, const double *zsrc, double *dre)
{
    if (h->il2d_rx_stk < 0)
        _il2d_rowx_body_bwd(h, zsrc, dre);
    else
        _il2d_rowx_call(_il2d_rowx_body_bwd, h, zsrc, dre, h->il2d_rx_stk);
}

/* the K = 1 request the per-row engines are built for, in the plan's direction */
static void _il2d_rowx_cfg(const vfft_config_t *cfg, int N2, vfft_config_t *c, struct vfft_wisdom_s *S, int c2r)
{
    memset(c, 0, sizeof *c);
    c->transform = c2r ? VFFT_C2R : VFFT_R2C;
    c->placement = VFFT_OUTOFPLACE;
    c->rigor = cfg->rigor;
    c->dims = 1;
    c->n[0] = N2;
    c->howmany = 1;
    c->layout = VFFT_LAYOUT_INTERLEAVED;
    c->nthreads = 1;
    c->wisdom = (vfft_wisdom *)S;   /* the row engines' own store (wisdom2_child.h), never the library's */
    c->wisdom_write = 0;
    c->recalibrate = cfg->recalibrate;
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

/* a zr2c row engine at a route: its child from the row engines' own store
 * (the real row's rx_* tokens, wisdom2_child.h) when that banks this route,
 * else raced in the real role and banked there, so the recipe rides on the
 * 2D row. A 1D row is never read (owner, 2026-10-01). */
static struct vfft_plan_s *_il2d_rowx_zr2c(const vfft_config_t *c, struct vfft_wisdom_s *S, int N2, int route)
{
    int r = 0, fmt = 0;
    vw2_zr2c_child_t ch;
    struct vfft_plan_s *e;
    if (S && !c->recalibrate && vw2_real_il_lookup_zr2c(&S->vw2, N2, c->transform == VFFT_C2R, 0, 1, &r, &fmt, &ch) &&
        r == route)
    {
        vfft_il_cand_t kc;
        _zr2c_prime_t pr;
        struct vfft_wisdom_s *fsS = NULL;
        _zr2c_cand_of_child(&kc, &ch);
        if (_zr2c_prime_of_child(&pr, &ch) &&
            (ch.route != VFFT_K1_IL_FS || (fsS = _zr2c_fs_store(ch.fs, ch.R1, ch.R2, route)) != NULL) &&
            (e = _zr2c_build_route(c, N2, route, &kc, &pr, fsS)) != NULL)
            return e;
    }
    e = _zr2c_build_route(c, N2, route, NULL, NULL, NULL);
    if (e && S)
        (void)_zr2c_bank(S, c, N2, e, 0.0);   /* the raced child's recipe into the engines' store */
    return e;
}

/* a token's engine: 1 = built (door: nothing to build), 0 = it does not build here */
static int _il2d_rowx_build(const vfft_config_t *cfg, struct vfft_wisdom_s *S, int N1, int N2, const char *tok,
                            vfft_il2p_fn *lm, struct vfft_plan_s **eng, int c2r)
{
    vfft_config_t c;
    *lm = NULL;
    *eng = NULL;
    if (!tok || !tok[0] || !strcmp(tok, "door"))
        return 1;
    if (!strcmp(tok, "lm"))
    {
        *lm = N1 >= 2 ? (c2r ? vfft_il2d_rows_bwd_fn(N2) : vfft_il2d_rows_fn(N2)) : NULL;
        return *lm != NULL;
    }
    _il2d_rowx_cfg(cfg, N2, &c, S, c2r);
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
        *eng = _il2d_rowx_zr2c(&c, S, N2, tok[6] == '1');
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

/* the race's arm: one candidate at one stack state installed on the plan, the
 * row pass run in the plan's direction (bwd: the CCE plane z -> the real plane a) */
typedef struct
{
    struct vfft_plan_s *h;
    double *a;   /* the real plane: the r2c pass's input, the c2r pass's output */
    double *z;   /* the CCE plane */
    vfft_il2p_fn lm;
    struct vfft_plan_s *eng;
    int stk, bwd;
} _il2d_rowx_arm_t;
static void _il2d_rowx_arm_run(void *v)
{
    _il2d_rowx_arm_t *c = (_il2d_rowx_arm_t *)v;
    c->h->il2d_rx_lm = c->lm;
    c->h->il2d_rx_eng = c->eng;
    c->h->il2d_rx_stk = c->stk;
    c->h->il2d_rx_on = 1;
    if (c->bwd)
        _il2d_real_rows_bwd(c->h, c->z, c->a);
    else
        _il2d_real_rows_fwd(c->h, c->a, c->z);
}

/* bank the verdict on the shared real IL row; the row is made when the cell
 * has none yet (the served chain, no row-axis tokens: fft2d_create_il.h's
 * forms bank does the same) */
static void _il2d_real_plan_bank(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                                 int N1, int N2, int ord, int T, const char *tok, const char *name,
                                 const char *stok, const char *sb, int c2r)
{
    int ok;
    if (!W || W->vw2_off_2d)
        return;
    ok = vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, T, tok, name) == 0;
    if (!ok)
    {
        vw2_2d_rl_bank(&W->vw2, N1, N2, c2r, h->il2d_col.R, h->il2d_col.nst, -1, -1, 0,
                       (N1 & (N1 - 1)) ? h->il2d_col.blu : -1, 0.0, ord, T);
        ok = vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, T, tok, name) == 0;
    }
    if (ok)
        ok = vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, T, stok, sb) == 0;
    if (ok)
        _vw2_persist(W, cfg);
    else
        fprintf(stderr, "vfft: the 2D real %s plan NOT banked at %dx%d -- the cell will re-race\n", tok, N1, N2);
}

/* THE ROW PLAN: env pin, the banked rx=, or the race. Runs after the plan's
 * stage arrays are committed (it executes the row pass of h). One body for
 * the two directions: c2r = 1 is the backward pass's plan, with its own
 * tokens, env pin and engine store; the r2c path is the one that stood. */
static void _il2d_real_rowplan_dir(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                                   int N1, int N2, int ord, int T, int c2r)
{
    enum { NARMS = VFFT_RACE_MAX_ARMS / 4 };
    const size_t hp1 = (size_t)N2 / 2 + 1, RN = (size_t)N1 * (size_t)N2, CN = 2 * (size_t)N1 * hp1;
    const char *log = getenv("VFFT_IL2D_LOG");
    const char *tk_rx = c2r ? "rx_c2r" : "rx", *tk_rxs = c2r ? "rxs_c2r" : "rxs";
    const char *ev = c2r ? "VFFT_IL2D_RX_C2R" : "VFFT_IL2D_RX", *dn = c2r ? "c2r " : "";
    vfft_il2p_fn lm = NULL;
    struct vfft_plan_s *eng = NULL;
    h->il2d_rx_lm = NULL;
    h->il2d_rx_eng = NULL;
    h->il2d_rx_stk = 0;
    h->il2d_rx_on = 0;
    if (!h->il2d_rxS)
    {   /* THE ROW ENGINES IN ROLE: their own store, seeded from this cell's real row (rx_*, or rx_c2r_*) */
        vw2_key_t pk;
        vw2__2d_key(&pk, VW2_T_R2C, 2, N1, N2, 0, ord, VW2_LAY_IL, T);
        h->il2d_rxS = vfft_child_store_for(W ? &W->vw2 : NULL, &pk, c2r ? "rx_c2r_" : "rx_");
    }
    {   /* 1. env: the racing hook. Beats wisdom, never banks. */
        const char *e = getenv(ev);
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
            if (_il2d_rowx_build(cfg, h->il2d_rxS, N1, N2, tok, &lm, &eng, c2r))
            {
                h->il2d_rx_lm = lm;
                h->il2d_rx_eng = eng;
                h->il2d_rx_stk = sl ? _il2d_plan_stk_parse(sl + 1) : 0;
                h->il2d_rx_on = 1;
            }
            else
                _vfft_warn("vfft_create: %s=%s does not build at %dx%d (the row route stands)", ev, e, N1, N2);
            return;
        }
    }
    if (!W || W->vw2_off_2d)
        return;
    if (!cfg->recalibrate)
    {   /* 2. the banked verdict */
        const char *tok = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, tk_rx);
        if (tok)
        {
            const char *sv = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, tk_rxs);
            if (_il2d_rowx_build(cfg, h->il2d_rxS, N1, N2, tok, &lm, &eng, c2r))
            {
                h->il2d_rx_lm = lm;
                h->il2d_rx_eng = eng;
                h->il2d_rx_stk = _il2d_plan_stk_parse(sv);
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
        const size_t ON = c2r ? RN : CN;   /* the pass's output plane */
        double *a = (double *)vfft_aligned_alloc((RN + 8) * sizeof(double));
        double *z = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
        double *ref = (double *)vfft_aligned_alloc((ON + 8) * sizeof(double));
        double *const out = c2r ? a : z;
        if (!a || !z || !ref)
        {
            vfft_aligned_free(a); vfft_aligned_free(z); vfft_aligned_free(ref);
            return;
        }
        {
            unsigned sd = 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12);
            size_t j;
            if (!c2r)
                for (j = 0; j < RN + 8; j++)
                {
                    sd = sd * 1664525u + 1013904223u;
                    a[j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
                }
            else
            {   /* a CCE plane: the DC and Nyquist bins real, as every engine's contract has them */
                for (j = 0; j < CN + 8; j++)
                {
                    sd = sd * 1664525u + 1013904223u;
                    z[j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
                }
                for (j = 0; j < (size_t)N1; j++)
                    z[j * 2 * hp1 + 1] = z[j * 2 * hp1 + 2 * (hp1 - 1) + 1] = 0.0;
            }
        }
        _il2d_rowx_cfg(cfg, N2, &c, h->il2d_rxS, c2r);
#define ROWX_ARM(LM, ENG) do { if (na < NARMS) { \
            cand[na].h = h; cand[na].a = a; cand[na].z = z; cand[na].lm = (LM); cand[na].eng = (ENG); cand[na].stk = 0; \
            cand[na].bwd = c2r; _il2d_rowx_name(cand[na].lm, cand[na].eng, names[na], sizeof names[na]); na++; } \
            else if ((ENG) != NULL) vfft_destroy((vfft_plan)(ENG)); } while (0)
        ROWX_ARM(NULL, NULL);   /* arm 0: the row route the tier has */
        if (N1 >= 2 && (c2r ? vfft_il2d_rows_bwd_fn(N2) : vfft_il2d_rows_fn(N2)))
            ROWX_ARM(c2r ? vfft_il2d_rows_bwd_fn(N2) : vfft_il2d_rows_fn(N2), NULL);
        {
            struct vfft_plan_s *hz = _il2d_rowx_zr2c(&c, h->il2d_rxS, N2, 0), *e;
            if (hz)
                ROWX_ARM(NULL, hz);
            if ((e = _il2d_rowx_zr2c(&c, h->il2d_rxS, N2, 1)) != NULL)
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
            {   /* ZTT-r's shortlist, as the door sweeps it (gated on zr2c's row): the
                 * plane's first row in (real samples, or a CCE row), its transform out */
                const size_t xs = (size_t)N2 + 2;
                double *rb = (double *)vfft_aligned_alloc(xs * sizeof(double));
                double *rr = (double *)vfft_aligned_alloc(xs * sizeof(double));
                if (rb && rr)
                {
                    struct vfft_plan_s *ht[VFFT_ZTTR_MAX_ARMS];
                    const double *s0 = c2r ? z : a;
                    int nt;
                    _exec_zr2c(hz, s0, rr);
                    nt = _zttr_sweep(&c, N2, s0, rr, rb, s0, xs, c2r ? (size_t)N2 : xs, ht);
                    for (i = 0; i < nt; i++)
                        ROWX_ARM(NULL, ht[i]);
                }
                vfft_aligned_free(rb); vfft_aligned_free(rr);
            }
        }
#undef ROWX_ARM
        /* the gate: every arm's plane against the tier's own row route */
        memset(ref, 0, (ON + 8) * sizeof(double));
        if (c2r)
            cand[0].a = ref;
        else
            cand[0].z = ref;
        _il2d_rowx_arm_run(&cand[0]);
        cand[0].a = a;
        cand[0].z = z;
        {
            int keep = 1;
            for (i = 1; i < na; i++)
            {
                double e;
                memset(out, 0, ON * sizeof(double));
                _il2d_rowx_arm_run(&cand[i]);
                e = _zrpr_relerr(out, ref, ON);
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
                    fprintf(stderr, "[il2d-real] %srows %dx%d: %s FAILS the gate (rel %.2e) -- dropped\n", dn, N1, N2, names[i], e);
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
                fprintf(stderr, "[il2d-real] %srows %dx%d row plan race: reps=%d |", dn, N1, N2, reps);
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
        h->il2d_rx_stk = _il2d_plan_stk(&ns[4 * (best / 4)], ctx[best].stk);
        h->il2d_rx_on = 1;
        if (h->il2d_rx_eng && h->il2d_rx_eng->zr2c_kid && h->il2d_rxS)
            (void)_zr2c_bank(h->il2d_rxS, &c, N2, h->il2d_rx_eng, 0.0);   /* the winner's child: the recipe the row carries */
        if (log && h->il2d_rx_stk < 0)
            fprintf(stderr, "[il2d-real] %srows %dx%d: %s is stack-insensitive -> any\n", dn, N1, N2, names[best / 4]);
        {
            char sb[8];
            _il2d_plan_stk_str(h->il2d_rx_stk, sb, sizeof sb);
            _il2d_real_plan_bank(h, W, cfg, N1, N2, ord, T, tk_rx, names[best / 4], tk_rxs, sb, c2r);
        }
        vfft_aligned_free(a); vfft_aligned_free(z); vfft_aligned_free(ref);
    }
}
static void _il2d_real_rowplan(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                               int N1, int N2, int ord, int T)
{
    _il2d_real_rowplan_dir(h, W, cfg, N1, N2, ord, T, 0);
}
/* the c2r twin: the backward row pass (the column-inverse plane's CCE rows ->
 * the real rows), in the row role, banked rx_c2r= / rxs_c2r= */
static void _il2d_real_rowplan_c2r(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                                   int N1, int N2, int ord, int T)
{
    _il2d_real_rowplan_dir(h, W, cfg, N1, N2, ord, T, 1);
}

/* ═══════════════════════════════════════════════════════════════════════
 * THE COLUMN PLAN (r2c, and its c2r twin). The serial column pass has the row pass's exposure:
 * its kernels spill from radix 16, so the pass runs at one of several speeds
 * by the caller's rsp (the natural pass at 128x32: 2464 ns in one caller,
 * 1785 in another, measured 2026-10-01). And the pass has forms.
 *
 * THE FORMS
 *   plain | strided | staged   the plan's chain: without the natural leaf, or
 *          the natural leaf at its natural stride or through the staging
 *          (staged is 6-20% ahead on planes past L1, strided 2x ahead inside
 *          it).
 *   b816 | b448   a LEAF: the whole pass in ONE kernel, the N1-point blocked
 *          column leaf (N1 = 128: two-pass 8x16, three-pass 4x4x8;
 *          codelets/zil/<isa>/shared/col/blocked). In place and natural by
 *          construction: no second plane, which is what costs the natural
 *          chain 1.6-2x once plane + scratch leave L1 (measured 2026-10-01).
 *          Forward kernels for r2c, their backward twins for c2r; the
 *          chain stays the plan's chain, and the threaded walks run it.
 *          Gated against the chain's plane before it races.
 *
 * THE STACK STATE IS PER KERNEL. Which residue is the fast one belongs to a
 * kernel's own frame, and the stages of a chain disagree (a mid and a leaf):
 * one state for the pass is a compromise between them. The plan holds one
 * state per chain stage, each stage entered through its own aligned call.
 * The race: every form at the four states, one state for the whole pass;
 * then, where the winner is a chain of two stages or more (always the
 * natural one) whose pass is state-sensitive, each stage's four states
 * with the others held, and the
 * per-stage plan is kept when it beats the best single state by 3%. A stage
 * (or a pass) whose four states tie is entered directly (`any`). A leaf is
 * one kernel: one state.
 *
 * WISDOM, r2c's own tokens on the shared real IL row:
 *   cx=   plain | strided | staged | b816 | b448
 *   cxs=  one state for the pass, 0..3 | any; or one per chain stage,
 *         dot-separated: 2.any.0
 * VFFT_IL2D_CX=<cx>[/<cxs>] pins (s<state> is read too); VFFT_IL2D_CX=off
 * leaves the pass unbound. The prime-column routes (Bluestein, turned) keep
 * their own passes; the threaded column walks are untouched.
 *
 * THE C2R TWIN (2026-10-05): the same plan over the REVERSE pass (the
 * Hermitian-transpose chain, out of place from the caller's plane onto the
 * column-inverse plane): the same forms (plain | strided | staged, the
 * natural leaf's reverse gathering through the staging), the backward
 * leaves (b816 | b448: n1cb*_bwd), the per-stage states indexed by chain
 * stage as r2c's are (the stages run in reverse order). Raced on the
 * cell's own reverse pass, banked cx_c2r= / cxs_c2r=, pinned by
 * VFFT_IL2D_CX_C2R.
 * ═══════════════════════════════════════════════════════════════════════ */
static void _il2d_colx_body(const void *v, const double *src, double *dst)
{
    const struct vfft_plan_s *h = (const struct vfft_plan_s *)v;
    const vfft_ilcol_t *c = &h->il2d_col;
    if (h->il2d_cx_leaf)
    {   /* the whole pass in one kernel: the N1 legs a row pitch apart, the count = the columns */
        h->il2d_cx_leaf(src, NULL, dst, NULL, NULL, NULL, c->rn, 0, c->rn, 0, c->rn);
        return;
    }
    _il2d_col_exec_st(c, src, dst, 0, (c->nat && h->il2d_cx_st) ? c->natstage : NULL);
}
/* the c2r pass: the backward leaf, or the reverse chain in its form */
static void _il2d_colx_body_bwd(const void *v, const double *src, double *dst)
{
    const struct vfft_plan_s *h = (const struct vfft_plan_s *)v;
    const vfft_ilcol_t *c = &h->il2d_col;
    if (h->il2d_cx_leaf)
    {
        h->il2d_cx_leaf(src, NULL, dst, NULL, NULL, NULL, c->rn, 0, c->rn, 0, c->rn);
        return;
    }
    _il2d_col_exec_st(c, src, dst, 1, (c->nat && h->il2d_cx_st) ? c->natstage : NULL);
}

/* ONE CHAIN STAGE under its own entry: the natural pass (_il2d_col_pass_nat,
 * forward) a stage at a time -- same kernels, same calls, same order:
 * bitwise. A 2D real chain of two stages or more is always the natural one. */
typedef struct
{
    const struct vfft_plan_s *h;
    int s;
} _il2d_cxk_t;
static void _il2d_cxk_stage(const void *v, const double *src, double *dst)
{
    const _il2d_cxk_t *k = (const _il2d_cxk_t *)v;
    const vfft_ilcol_t *c = &k->h->il2d_col;
    _il2d_col_stages(src, dst, c->N, c->rn, k->s, k->s + 1, c->R, c->L, c->f, c->tf, 0);
}
static void _il2d_cxk_natleaf(const void *v, const double *scr, double *dst)
{
    const _il2d_cxk_t *k = (const _il2d_cxk_t *)v;
    const struct vfft_plan_s *h = k->h;
    const vfft_ilcol_t *c = &h->il2d_col;
    const int Rl = c->R[c->nst - 1];
    _il2d_nat_leaf_range(scr, dst, c->N, c->rn, Rl, c->f[c->nst - 1], c->natperm, 0, (size_t)(c->N / Rl), 0,
                         h->il2d_cx_st ? c->natstage : NULL);
}
static inline void _il2d_cxk_run(_il2d_rowx_fn fn, const _il2d_cxk_t *k, const double *s, double *d, int stk)
{
    if (stk < 0)
        fn(k, s, d);
    else
        _il2d_rowx_call(fn, k, s, d, stk);
}
/* the pass, a stage at a time: the mids src -> natscr, the leaf natscr -> dst */
static void _il2d_colx_perk(const struct vfft_plan_s *h, const double *src, double *dst)
{
    const vfft_ilcol_t *c = &h->il2d_col;
    const signed char *ks = h->il2d_cx_ks;
    const int nst = c->nst;
    _il2d_cxk_t k;
    int s;
    k.h = h;
    for (s = 0; s < nst - 1; s++)
    {
        k.s = s;
        _il2d_cxk_run(_il2d_cxk_stage, &k, s == 0 ? src : c->natscr, c->natscr, ks[s]);
    }
    k.s = nst - 1;
    _il2d_cxk_run(_il2d_cxk_natleaf, &k, c->natscr, dst, ks[nst - 1]);
}
static void _il2d_colx_fwd(struct vfft_plan_s *h, const double *src, double *dst)
{
    if (h->il2d_cx_perk)
        _il2d_colx_perk(h, src, dst);
    else if (h->il2d_cx_stk < 0)
        _il2d_colx_body(h, src, dst);
    else
        _il2d_rowx_call(_il2d_colx_body, h, src, dst, h->il2d_cx_stk);
}

/* THE C2R PER-STAGE ENTRIES: the reverse natural pass (_il2d_col_pass_nat,
 * reverse) a stage at a time -- the leaf gathers the natural src into the
 * scratch's comb, the mids run in reverse chain order in place, stage 0
 * writes the scratch -> dst; the same kernels, calls and order: bitwise. */
static void _il2d_cxk_stage_bwd(const void *v, const double *src, double *dst)
{
    const _il2d_cxk_t *k = (const _il2d_cxk_t *)v;
    const vfft_ilcol_t *c = &k->h->il2d_col;
    _il2d_col_stages(src, dst, c->N, c->rn, k->s, k->s + 1, c->R, c->L, c->b, c->tb, 1);
}
static void _il2d_cxk_natleaf_bwd(const void *v, const double *src, double *scr)
{
    const _il2d_cxk_t *k = (const _il2d_cxk_t *)v;
    const struct vfft_plan_s *h = k->h;
    const vfft_ilcol_t *c = &h->il2d_col;
    const int Rl = c->R[c->nst - 1];
    _il2d_nat_leaf_range(src, scr, c->N, c->rn, Rl, c->b[c->nst - 1], c->natperm, 0, (size_t)(c->N / Rl), 1,
                         h->il2d_cx_st ? c->natstage : NULL);
}
static void _il2d_colx_perk_bwd(const struct vfft_plan_s *h, const double *src, double *dst)
{
    const vfft_ilcol_t *c = &h->il2d_col;
    const signed char *ks = h->il2d_cx_ks;
    const int nst = c->nst;
    _il2d_cxk_t k;
    int s;
    k.h = h;
    k.s = nst - 1;
    _il2d_cxk_run(_il2d_cxk_natleaf_bwd, &k, src, c->natscr, ks[nst - 1]);
    for (s = nst - 2; s >= 0; s--)
    {
        k.s = s;
        _il2d_cxk_run(_il2d_cxk_stage_bwd, &k, c->natscr, s == 0 ? dst : c->natscr, ks[s]);
    }
}
static void _il2d_colx_bwd(struct vfft_plan_s *h, const double *src, double *dst)
{
    if (h->il2d_cx_perk)
        _il2d_colx_perk_bwd(h, src, dst);
    else if (h->il2d_cx_stk < 0)
        _il2d_colx_body_bwd(h, src, dst);
    else
        _il2d_rowx_call(_il2d_colx_body_bwd, h, src, dst, h->il2d_cx_stk);
}

static void _il2d_colx_reset(struct vfft_plan_s *h)
{
    h->il2d_cx_on = h->il2d_cx_st = h->il2d_cx_stk = h->il2d_cx_perk = 0;
    h->il2d_cx_leaf = NULL;
    memset(h->il2d_cx_ks, 0, sizeof h->il2d_cx_ks);
}
/* cxs as text: the pass's one state, or one per stage */
static void _il2d_colx_stk_str(const struct vfft_plan_s *h, char *buf, size_t cap)
{
    int s;
    size_t l = 0;
    if (!h->il2d_cx_perk)
    {
        _il2d_plan_stk_str(h->il2d_cx_stk, buf, cap);
        return;
    }
    buf[0] = 0;
    for (s = 0; s < h->il2d_col.nst && l + 6 < cap; s++)
    {
        if (s)
            buf[l++] = '.';
        _il2d_plan_stk_str(h->il2d_cx_ks[s], buf + l, cap - l);
        l += strlen(buf + l);
    }
}
/* a cxs value onto the plan (its form already set); 0 = not a state list of
 * this plan's pass, the caller races */
static int _il2d_colx_stk_set(struct vfft_plan_s *h, const char *v)
{
    const int nst = h->il2d_col.nst;
    int s = 0;
    h->il2d_cx_perk = 0;
    if (!v || !strchr(v, '.'))
    {
        h->il2d_cx_stk = _il2d_plan_stk_parse(v);
        return 1;
    }
    if (h->il2d_cx_leaf || !h->il2d_col.nat || nst < 2)
        return 0; /* a leaf, or a one-stage chain, is one kernel: one state */
    while (*v && s < nst)
    {
        char e[8];
        int l = 0;
        while (*v && *v != '.' && l < 7)
            e[l++] = *v++;
        e[l] = 0;
        if (!l || (*v && *v != '.'))
            return 0;
        if (*v == '.')
            v++;
        if (strcmp(e, "any"))
        {
            const char *d = (e[0] == 's') ? e + 1 : e;
            if (d[0] < '0' || d[0] > '3' || d[1])
                return 0;
        }
        h->il2d_cx_ks[s++] = (signed char)_il2d_plan_stk_parse(e);
    }
    if (*v || s != nst)
        return 0;
    h->il2d_cx_perk = 1;
    h->il2d_cx_stk = 0;
    return 1;
}
/* a cx name (n characters) onto the plan: a form of the chain or a leaf of
 * this cell (the backward leaf for c2r); 0 = neither */
static int _il2d_colx_form_set(struct vfft_plan_s *h, const char *name, size_t n,
                               const char *const *fname, int nf, const char *const *lname, int nl, int c2r)
{
    int f;
    h->il2d_cx_leaf = NULL;
    h->il2d_cx_st = 0;
    for (f = 0; f < nf; f++)
        if (strlen(fname[f]) == n && !strncmp(name, fname[f], n))
        {
            h->il2d_cx_st = f;
            return 1;
        }
    for (f = 0; f < nl; f++)
        if (strlen(lname[f]) == n && !strncmp(name, lname[f], n))
        {
            h->il2d_cx_leaf = c2r ? vfft_il2p_col_leaf_bwd_fn(h->il2d_col.N, lname[f])
                                  : vfft_il2p_col_leaf_fn(h->il2d_col.N, lname[f]);
            return h->il2d_cx_leaf != NULL;
        }
    return 0;
}

typedef struct
{
    struct vfft_plan_s *h;
    double *z;
    int st, stk, perk, bwd;
    signed char ks[8];
    vfft_il2p_fn leaf;
} _il2d_colx_arm_t;
static void _il2d_colx_arm_run(void *v)
{
    _il2d_colx_arm_t *c = (_il2d_colx_arm_t *)v;
    struct vfft_plan_s *h = c->h;
    h->il2d_cx_st = c->st;
    h->il2d_cx_stk = c->stk;
    h->il2d_cx_perk = c->perk;
    memcpy(h->il2d_cx_ks, c->ks, sizeof h->il2d_cx_ks);
    h->il2d_cx_leaf = c->leaf;
    h->il2d_cx_on = 1;
    if (c->bwd)
        _il2d_real_cols(h, c->z, h->il2d_rscr, 1);   /* the reverse pass, out of place onto the plan's plane */
    else
        _il2d_real_cols(h, c->z, c->z, 0);
}

#define VFFT_IL2D_CX_MAXFORMS (2 + VFFT_IL2P_COL_MAXLEAF)
/* one body for the two directions: c2r = 1 races the reverse pass (its forms,
 * the backward leaves) out of place onto the plan's column-inverse plane, as
 * the pass serves; the r2c path is the one that stood */
static void _il2d_real_colplan_pick(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                                    int N1, int N2, int ord, int T, int c2r)
{
    const size_t hp1 = (size_t)N2 / 2 + 1, CN = 2 * (size_t)N1 * hp1;
    const vfft_ilcol_t *col = &h->il2d_col;
    const int nat = col->nat != 0, nf = (nat && col->natstage) ? 2 : 1;
    const char *fname[2];
    const char *lname[VFFT_IL2P_COL_MAXLEAF];
    const char *log = getenv("VFFT_IL2D_LOG");
    const char *tk_cx = c2r ? "cx_c2r" : "cx", *tk_cxs = c2r ? "cxs_c2r" : "cxs";
    const char *ev = c2r ? "VFFT_IL2D_CX_C2R" : "VFFT_IL2D_CX", *dn = c2r ? "c2r " : "";
    int nl = 0, s;
    fname[0] = nat ? "strided" : "plain";
    fname[1] = "staged";
    _il2d_colx_reset(h);
    if (col->blu || col->tpc)
        return;
    if (c2r && !h->il2d_rscr)
        return; /* the reverse pass serves onto the column-inverse plane */
    /* a leaf writes natural order: offered where the chain's pass does */
    if (nat || col->nst == 1)
        nl = c2r ? vfft_il2p_col_leaf_bwd_forms(N1, lname) : vfft_il2p_col_leaf_forms(N1, lname);
    {   /* 1. env: the racing hook. Beats wisdom, never banks. */
        const char *e = getenv(ev);
        if (e && e[0])
        {
            const char *sl = strchr(e, '/');
            const size_t n = sl ? (size_t)(sl - e) : strlen(e);
            if (!strcmp(e, "off"))
                return;
            if (_il2d_colx_form_set(h, e, n, fname, nf, lname, nl, c2r) && _il2d_colx_stk_set(h, sl ? sl + 1 : NULL))
            {
                h->il2d_cx_on = 1;
                return;
            }
            _il2d_colx_reset(h);
            _vfft_warn("vfft_create: %s=%s is not a column form (and stack states) of %dx%d (the pass stays unbound)", ev, e, N1, N2);
            return;
        }
    }
    if (!W || W->vw2_off_2d)
        return;
    if (!cfg->recalibrate)
    {   /* 2. the banked verdict */
        const char *tok = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, tk_cx);
        if (tok && _il2d_colx_form_set(h, tok, strlen(tok), fname, nf, lname, nl, c2r) &&
            _il2d_colx_stk_set(h, vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, tk_cxs)))
        {
            h->il2d_cx_on = 1;
            return;
        }
        _il2d_colx_reset(h);
    }
    /* 3. the race: every form at every stack state, the pass in place on a scratch plane
     * (the values compound: benign, the chain race's precedent); the reverse pass out
     * of place from the scratch plane onto the plan's column-inverse plane */
    {
        _il2d_colx_arm_t cand[VFFT_IL2D_CX_MAXFORMS], ctx[4 * VFFT_IL2D_CX_MAXFORMS];
        vfft_race_arm_t arms[4 * VFFT_IL2D_CX_MAXFORMS];
        char names[VFFT_IL2D_CX_MAXFORMS][8], xn[4 * VFFT_IL2D_CX_MAXFORMS][24], sb[40];
        double ns[4 * VFFT_IL2D_CX_MAXFORMS], t0, est;
        const vfft_race_proto_t proto0 = { 9, 1, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
        vfft_race_proto_t proto = proto0; /* 9 rounds alternated, median */
        int na = 0, best = 0, bc = 0, reps, i, f, u;
        double *z = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
        double *const out = c2r ? h->il2d_rscr : z;   /* the pass's output plane */
        if (!z)
            return;
        {
            unsigned sd = 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12);
            size_t j;
            for (j = 0; j < CN + 8; j++)
            {
                sd = sd * 1664525u + 1013904223u;
                z[j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
            }
        }
        for (f = 0; f < nf; f++)
        {
            memset(&cand[na], 0, sizeof cand[na]);
            cand[na].h = h; cand[na].z = z; cand[na].st = f; cand[na].stk = -1; cand[na].bwd = c2r;
            snprintf(names[na], sizeof names[na], "%s", fname[f]);
            na++;
        }
        if (nl)
        {   /* the gate: a leaf's plane against the chain's, from one input */
            double *z0 = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
            double *ref = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
            if (z0 && ref)
            {
                memcpy(z0, z, (CN + 8) * sizeof(double));
                memcpy(ref, z0, (CN + 8) * sizeof(double));
                cand[0].z = ref;
                _il2d_colx_arm_run(&cand[0]);
                cand[0].z = z;
                if (c2r)
                    memcpy(ref, out, CN * sizeof(double));   /* the reverse pass left its plane in the scratch */
                for (f = 0; f < nl; f++)
                {
                    double e;
                    memset(&cand[na], 0, sizeof cand[na]);
                    cand[na].h = h; cand[na].z = z; cand[na].stk = -1; cand[na].bwd = c2r;
                    cand[na].leaf = c2r ? vfft_il2p_col_leaf_bwd_fn(N1, lname[f]) : vfft_il2p_col_leaf_fn(N1, lname[f]);
                    if (!cand[na].leaf)
                        continue;
                    memcpy(z, z0, (CN + 8) * sizeof(double));
                    _il2d_colx_arm_run(&cand[na]);
                    e = _zrpr_relerr(out, ref, CN);
                    if (e < 1e-10)
                    {
                        snprintf(names[na], sizeof names[na], "%s", lname[f]);
                        na++;
                    }
                    else
                        fprintf(stderr, "[il2d-real] %scols %dx%d: leaf %s FAILS the gate (rel %.2e) -- dropped\n", dn, N1, N2, lname[f], e);
                }
                memcpy(z, z0, (CN + 8) * sizeof(double));
            }
            vfft_aligned_free(z0); vfft_aligned_free(ref);
        }
        for (i = 0; i < na; i++)
            for (s = 0; s < 4; s++)
            {
                const int x = 4 * i + s;
                ctx[x] = cand[i];
                ctx[x].stk = s;
                snprintf(xn[x], sizeof xn[x], "%s/s%d", names[i], s);
                arms[x].name = xn[x]; arms[x].run = _il2d_colx_arm_run; arms[x].ctx = &ctx[x];
            }
        _vfft_create_race_count++;
        _il2d_colx_arm_run(&ctx[0]);
        t0 = vfft_now_ns();
        _il2d_colx_arm_run(&ctx[0]);
        est = vfft_now_ns() - t0;
        reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
        if (reps < 1) reps = 1;
        if (reps > 4096) reps = 4096;
        proto.reps = reps;
        vfft_race_run(&proto, arms, 4 * na, ns);
        for (i = 1; i < 4 * na; i++)
            if (ns[i] < ns[best]) best = i;
        for (i = 1; i < 4 * nf; i++)
            if (ns[i] < ns[bc]) bc = i;
        /* 3% hysteresis toward the chain: a leaf serves where it clearly wins */
        if (ctx[best].leaf && !vfft_race_beats(ns[best], ns[bc], 0.97))
            best = bc;
        if (log)
        {
            fprintf(stderr, "[il2d-real] %scols %dx%d column plan race: reps=%d |", dn, N1, N2, reps);
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
        u = _il2d_plan_stk(&ns[4 * (best / 4)], ctx[best].stk);
        ctx[best].stk = u;
        if (!ctx[best].leaf && nat && col->nst >= 2 && u >= 0)
        {   /* the per-kernel states: each stage's four with the others held, in chain order */
            _il2d_colx_arm_t kc[4], fin[2];
            vfft_race_arm_t ka[4];
            char kn[4][24];
            double kns[4];
            signed char ks[8];
            int t, differs = 0;
            memset(ks, 0, sizeof ks);
            for (s = 0; s < col->nst; s++)
                ks[s] = (signed char)u;
            for (s = 0; s < col->nst; s++)
            {
                int kb = 0;
                for (t = 0; t < 4; t++)
                {
                    kc[t] = ctx[best];
                    kc[t].perk = 1;
                    memcpy(kc[t].ks, ks, sizeof ks);
                    kc[t].ks[s] = (signed char)t;
                    snprintf(kn[t], sizeof kn[t], "stage%d/s%d", s, t);
                    ka[t].name = kn[t]; ka[t].run = _il2d_colx_arm_run; ka[t].ctx = &kc[t];
                }
                vfft_race_run(&proto, ka, 4, kns);
                for (t = 1; t < 4; t++)
                    if (kns[t] < kns[kb]) kb = t;
                if (kb != ks[s] && vfft_race_beats(kns[kb], kns[ks[s]], 0.97))
                    ks[s] = (signed char)kb; /* 3% toward the pass's state */
                if (_il2d_plan_stk(kns, ks[s]) < 0)
                    ks[s] = -1;
                if (ks[s] != u)
                    differs = 1;
                if (log)
                    fprintf(stderr, "[il2d-real] %scols %dx%d stage %d (radix %d) states: %.0f/%.0f/%.0f/%.0f -> %d\n",
                            dn, N1, N2, s, col->R[s], kns[0], kns[1], kns[2], kns[3], (int)ks[s]);
            }
            if (differs)
            {   /* the per-stage plan against the best single state */
                fin[0] = ctx[best];
                fin[1] = ctx[best];
                fin[1].perk = 1;
                memcpy(fin[1].ks, ks, sizeof ks);
                ka[0].name = "pass"; ka[0].run = _il2d_colx_arm_run; ka[0].ctx = &fin[0];
                ka[1].name = "per-stage"; ka[1].run = _il2d_colx_arm_run; ka[1].ctx = &fin[1];
                vfft_race_run(&proto, ka, 2, kns);
                if (log)
                    fprintf(stderr, "[il2d-real] %scols %dx%d one state %.0f vs per stage %.0f -> %s\n", dn, N1, N2, kns[0], kns[1],
                            vfft_race_beats(kns[1], kns[0], 0.97) ? "per stage" : "one state");
                if (vfft_race_beats(kns[1], kns[0], 0.97))
                    ctx[best] = fin[1];
            }
        }
        h->il2d_cx_st = ctx[best].st;
        h->il2d_cx_stk = ctx[best].perk ? 0 : ctx[best].stk;
        h->il2d_cx_perk = ctx[best].perk;
        memcpy(h->il2d_cx_ks, ctx[best].ks, sizeof h->il2d_cx_ks);
        h->il2d_cx_leaf = ctx[best].leaf;
        h->il2d_cx_on = 1;
        _il2d_colx_stk_str(h, sb, sizeof sb);
        if (log)
            fprintf(stderr, "[il2d-real] %scols %dx%d: %s=%s %s=%s\n", dn, N1, N2, tk_cx, names[best / 4], tk_cxs, sb);
        _il2d_real_plan_bank(h, W, cfg, N1, N2, ord, T, tk_cx, names[best / 4], tk_cxs, sb, c2r);
        vfft_aligned_free(z);
    }
}

static void _il2d_real_colplan_dir(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                                   int N1, int N2, int ord, int T, int c2r)
{
    const vfft_ilcol_t *c = &h->il2d_col;
    _il2d_real_colplan_pick(h, W, cfg, N1, N2, ord, T, c2r);
    /* a one-stage chain is one kernel: its pass is that call (what
     * _il2d_col_pass makes of it), bound as the leaf. Banded or not: a
     * one-stage chain's only legal band is the plane (wl = N1), the same
     * call -- the band race banks either spelling of it. */
    if (h->il2d_cx_on && !h->il2d_cx_leaf && !h->il2d_cx_perk && c->nst == 1 && !c->nat && !c->blu && !c->tpc &&
        (c->wl == 0 || c->wl == c->N) && c->L[0] == c->N && (c2r ? c->b[0] : c->f[0]))
        h->il2d_cx_leaf = c2r ? c->b[0] : c->f[0];
}
static void _il2d_real_colplan(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                               int N1, int N2, int ord, int T)
{
    _il2d_real_colplan_dir(h, W, cfg, N1, N2, ord, T, 0);
}
/* the c2r twin: the reverse column pass's form and states, banked cx_c2r= / cxs_c2r= */
static void _il2d_real_colplan_c2r(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                                   int N1, int N2, int ord, int T)
{
    _il2d_real_colplan_dir(h, W, cfg, N1, N2, ord, T, 1);
}

/* ═══════════════════════════════════════════════════════════════════════
 * THE DESTROYING C2R (2026-10-06). A c2r request may permit the plan to
 * overwrite its input (vfft_config_t.destroy_input: a permission). Where the
 * reverse column pass is ONE kernel call (the law: policy_il.h,
 * vfft_policy_il2d_c2r_destroy_ok), that call can run IN PLACE on the
 * caller's CCE plane and the row pass read it: the column-inverse plane
 * (il2d_rscr) is never touched. Past L1 that is a plane the transform no
 * longer writes and reads back (measured 2026-10-06, one plan both ways:
 * 1.06-1.27 on the one-kernel cells; L1-sized planes go either way, which is
 * why the form is raced and never assumed).
 *
 * THE RACE, on the WHOLE transform (the saving is the row pass's as much as
 * the column pass's: the rows read the plane the kernel just wrote): the
 * scratch form as the plan stands (its c2r column plan and row plan) against
 * every one-kernel candidate in place -- `plain` (the one-stage chain's
 * backward kernel) or the backward leaves of the cell (b816 | b448), each
 * gated against the scratch form's output -- at the four stack states. The
 * caller's plane is rewritten before every timed sample (the race's reset
 * hook: repeated in-place inverses grow by N1 a call) and a sample is short
 * enough to stay finite. 3% hysteresis toward the scratch form: the input is
 * destroyed only where that clearly pays.
 *
 * WISDOM, on the shared real IL row:
 *   cxd_c2r=   off | plain | b816 | b448     (off: the scratch form won)
 *   cxds_c2r=  the in-place kernel's stack state 0..3 | any
 * VFFT_IL2D_CXD_C2R=<cxd>[/s<state>] pins (beats the bank, never banks);
 * VFFT_IL2D_CXD_C2R=off keeps the scratch form. Read and raced only for a
 * request that carries the permission: a plan without it never looks at
 * these tokens and preserves its input as it always did.
 * ═══════════════════════════════════════════════════════════════════════ */
static void _il2d_cxd_body(const void *v, const double *z, double *zz)
{
    const struct vfft_plan_s *h = (const struct vfft_plan_s *)v;
    const size_t rn = h->il2d_col.rn;
    (void)z;
    h->il2d_cxd_leaf(zz, NULL, zz, NULL, NULL, NULL, rn, 0, rn, 0, rn);
}
/* the reverse column pass: the one kernel, in place on the caller's plane */
static void _il2d_cxd_cols(struct vfft_plan_s *h, double *z)
{
    if (h->il2d_cxd_stk < 0)
        _il2d_cxd_body(h, z, z);
    else
        _il2d_rowx_call(_il2d_cxd_body, h, z, z, h->il2d_cxd_stk);
}
/* a cxd name (n characters) onto the plan: `plain` at a one-stage chain, or a
 * backward leaf of this cell; 0 = neither */
static int _il2d_cxd_form_set(struct vfft_plan_s *h, const char *name, size_t n, int one_stage,
                              const char *const *lname, int nl)
{
    int f;
    h->il2d_cxd_leaf = NULL;
    if (n == 5 && !strncmp(name, "plain", 5))
        h->il2d_cxd_leaf = one_stage ? h->il2d_col.b[0] : NULL;
    else
        for (f = 0; f < nl; f++)
            if (strlen(lname[f]) == n && !strncmp(name, lname[f], n))
                h->il2d_cxd_leaf = vfft_il2p_col_leaf_bwd_fn(h->il2d_col.N, lname[f]);
    return h->il2d_cxd_leaf != NULL;
}

/* the race's arm: the whole c2r, the scratch form (leaf NULL) or one kernel in place */
typedef struct
{
    struct vfft_plan_s *h;
    double *z, *y;   /* the caller's CCE plane, the real plane */
    vfft_il2p_fn leaf;
    int stk;
} _il2d_cxd_arm_t;
static void _il2d_cxd_arm_run(void *v)
{
    _il2d_cxd_arm_t *c = (_il2d_cxd_arm_t *)v;
    struct vfft_plan_s *h = c->h;
    if (!c->leaf)
    {
        h->il2d_cxd_on = 0;
        _il2d_real_cols(h, c->z, h->il2d_rscr, 1);
        _il2d_real_rows_bwd(h, h->il2d_rscr, c->y);
        return;
    }
    h->il2d_cxd_leaf = c->leaf;
    h->il2d_cxd_stk = c->stk;
    h->il2d_cxd_on = 1;
    _il2d_cxd_cols(h, c->z);
    _il2d_real_rows_bwd(h, c->z, c->y);
}
/* the race's reset: the caller's plane rewritten (the same values every time) */
typedef struct
{
    double *z;
    size_t n;
    unsigned sd;
} _il2d_cxd_seed_t;
static void _il2d_cxd_seed(void *v)
{
    const _il2d_cxd_seed_t *s = (const _il2d_cxd_seed_t *)v;
    unsigned sd = s->sd;
    size_t j;
    for (j = 0; j < s->n; j++)
    {
        sd = sd * 1664525u + 1013904223u;
        s->z[j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
    }
}

static void _il2d_real_destroyplan_c2r(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                                       int N1, int N2, int ord, int T)
{
    enum { MAXC = 1 + VFFT_IL2P_COL_MAXLEAF };   /* plain, the leaves */
    const size_t hp1 = (size_t)N2 / 2 + 1, RN = (size_t)N1 * (size_t)N2, CN = 2 * (size_t)N1 * hp1;
    const vfft_ilcol_t *col = &h->il2d_col;
    const int one_stage = col->nst == 1 && !col->nat && !col->blu && !col->tpc && (col->wl == 0 || col->wl == col->N) &&
                          col->L[0] == col->N && col->b[0] != NULL;
    const char *lname[VFFT_IL2P_COL_MAXLEAF];
    const char *log = getenv("VFFT_IL2D_LOG");
    int nl = 0;
    h->il2d_cxd_on = 0;
    h->il2d_cxd_stk = 0;
    h->il2d_cxd_leaf = NULL;
    /* a leaf writes natural order: offered where the column plan offers it */
    if (!col->blu && !col->tpc && (col->nat || col->nst == 1))
        nl = vfft_il2p_col_leaf_bwd_forms(N1, lname);
    if (!h->il2d_rscr || !vfft_policy_il2d_c2r_destroy_ok(cfg, h->nthreads, one_stage, nl, col->blu || col->tpc))
        return;
    {   /* 1. env: the racing hook. Beats wisdom, never banks. */
        const char *e = getenv("VFFT_IL2D_CXD_C2R");
        if (e && e[0])
        {
            const char *sl = strchr(e, '/');
            const size_t n = sl ? (size_t)(sl - e) : strlen(e);
            if (!strcmp(e, "off"))
                return;
            if (_il2d_cxd_form_set(h, e, n, one_stage, lname, nl))
            {
                h->il2d_cxd_stk = sl ? _il2d_plan_stk_parse(sl + 1) : 0;
                h->il2d_cxd_on = 1;
            }
            else
                _vfft_warn("vfft_create: VFFT_IL2D_CXD_C2R=%s is not a one-kernel column form of %dx%d (the scratch form serves)", e, N1, N2);
            return;
        }
    }
    if (!W || W->vw2_off_2d)
        return;
    if (!cfg->recalibrate)
    {   /* 2. the banked verdict */
        const char *tok = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, "cxd_c2r");
        if (tok)
        {
            if (!strcmp(tok, "off"))
                return; /* the scratch form won this cell's race */
            if (_il2d_cxd_form_set(h, tok, strlen(tok), one_stage, lname, nl))
            {
                h->il2d_cxd_stk = _il2d_plan_stk_parse(vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, "cxds_c2r"));
                h->il2d_cxd_on = 1;
                return;
            }
            /* a banked kernel that no longer builds here: the race */
        }
    }
    /* 3. the race: the whole transform, the scratch form against every one-kernel
     * candidate in place at every stack state */
    {
        _il2d_cxd_arm_t cand[MAXC], ctx[1 + 4 * MAXC];
        vfft_race_arm_t arms[1 + 4 * MAXC];
        char names[MAXC][8], xn[1 + 4 * MAXC][24], sb[8];
        double ns[1 + 4 * MAXC];
        _il2d_cxd_seed_t seed;
        int nc = 0, na, best = 1, ci = 0, win = 0, i, s, f, reps, lg = 1;
        double *z = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
        double *y = (double *)vfft_aligned_alloc((RN + 8) * sizeof(double));
        double *ref = (double *)vfft_aligned_alloc((RN + 8) * sizeof(double));
        if (!z || !y || !ref)
        {
            vfft_aligned_free(z); vfft_aligned_free(y); vfft_aligned_free(ref);
            return;
        }
        seed.z = z;
        seed.n = CN + 8;
        seed.sd = 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12);
        /* the reference: the scratch form's plane */
        ctx[0].h = h; ctx[0].z = z; ctx[0].y = ref; ctx[0].leaf = NULL; ctx[0].stk = 0;
        _il2d_cxd_seed(&seed);
        memset(ref, 0, (RN + 8) * sizeof(double));
        _il2d_cxd_arm_run(&ctx[0]);
        ctx[0].y = y;
        /* the candidates, each gated against it */
        for (f = -1; f < nl; f++)
        {
            const char *nm = f < 0 ? "plain" : lname[f];
            vfft_il2p_fn k = f < 0 ? (one_stage ? col->b[0] : NULL) : vfft_il2p_col_leaf_bwd_fn(N1, lname[f]);
            double e;
            if (!k)
                continue;
            cand[nc].h = h; cand[nc].z = z; cand[nc].y = y; cand[nc].leaf = k; cand[nc].stk = -1;
            _il2d_cxd_seed(&seed);
            memset(y, 0, RN * sizeof(double));
            _il2d_cxd_arm_run(&cand[nc]);
            e = _zrpr_relerr(y, ref, RN);
            if (e < 1e-10)
            {
                snprintf(names[nc], sizeof names[nc], "%s", nm);
                nc++;
            }
            else
                fprintf(stderr, "[il2d-real] c2r destroy %dx%d: %s in place FAILS the gate (rel %.2e) -- dropped\n", N1, N2, nm, e);
        }
        if (!nc)
        {   /* no candidate stands: the scratch form serves, nothing is banked */
            h->il2d_cxd_on = 0; h->il2d_cxd_leaf = NULL; h->il2d_cxd_stk = 0;
            vfft_aligned_free(z); vfft_aligned_free(y); vfft_aligned_free(ref);
            return;
        }
        arms[0].name = "scratch"; arms[0].run = _il2d_cxd_arm_run; arms[0].ctx = &ctx[0];
        for (i = 0; i < nc; i++)
            for (s = 0; s < 4; s++)
            {
                const int x = 1 + 4 * i + s;
                ctx[x] = cand[i];
                ctx[x].stk = s;
                snprintf(xn[x], sizeof xn[x], "%s/s%d", names[i], s);
                arms[x].name = xn[x]; arms[x].run = _il2d_cxd_arm_run; arms[x].ctx = &ctx[x];
            }
        na = 1 + 4 * nc;
        {
            double t0, est;
            _vfft_create_race_count++;
            _il2d_cxd_seed(&seed);
            _il2d_cxd_arm_run(&ctx[0]);
            t0 = vfft_now_ns();
            _il2d_cxd_arm_run(&ctx[0]);
            est = vfft_now_ns() - t0;
            reps = (int)(3.0e5 / (est > 1.0 ? est : 1.0));
            if (reps > 4096) reps = 4096;
            /* an in-place call grows the plane by at most N1: a sample of reps calls stays finite */
            while ((1 << lg) < N1) lg++;
            if (reps > 900 / lg) reps = 900 / lg;
            if (reps < 1) reps = 1;
            {   /* 9 rounds alternated, median; the plane rewritten before every sample */
                const vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, _il2d_cxd_seed, &seed, 1 };
                vfft_race_run(&proto, arms, na, ns);
            }
        }
        for (i = 2; i < na; i++)
            if (ns[i] < ns[best]) best = i;
        ci = (best - 1) / 4;
        /* 3% hysteresis toward the scratch form: the input is destroyed where that clearly pays */
        win = vfft_race_beats(ns[best], ns[0], 0.97);
        if (log)
        {
            fprintf(stderr, "[il2d-real] c2r destroy %dx%d race: reps=%d | scratch=%.0f |", N1, N2, reps, ns[0]);
            for (i = 0; i < nc; i++)
            {
                int bs = 0;
                for (s = 1; s < 4; s++)
                    if (ns[1 + 4 * i + s] < ns[1 + 4 * i + bs]) bs = s;
                fprintf(stderr, " %s=%.0f(s%d;%.0f/%.0f/%.0f/%.0f)", names[i], ns[1 + 4 * i + bs], bs,
                        ns[1 + 4 * i], ns[2 + 4 * i], ns[3 + 4 * i], ns[4 + 4 * i]);
            }
            fprintf(stderr, " -> %s\n", win ? xn[best] : "scratch (off)");
        }
        if (win)
        {
            h->il2d_cxd_leaf = cand[ci].leaf;
            h->il2d_cxd_stk = _il2d_plan_stk(&ns[1 + 4 * ci], ctx[best].stk);
            h->il2d_cxd_on = 1;
            _il2d_plan_stk_str(h->il2d_cxd_stk, sb, sizeof sb);
        }
        else
        {
            h->il2d_cxd_on = 0;
            h->il2d_cxd_leaf = NULL;
            h->il2d_cxd_stk = 0;
            snprintf(sb, sizeof sb, "any");
        }
        _il2d_real_plan_bank(h, W, cfg, N1, N2, ord, T, "cxd_c2r", win ? names[ci] : "off", "cxds_c2r", sb, 1);
        vfft_aligned_free(z); vfft_aligned_free(y); vfft_aligned_free(ref);
    }
}

#endif /* VFFT_IL2D_REAL_PLAN_H */
