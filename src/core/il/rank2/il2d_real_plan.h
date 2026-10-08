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
 *   zrf    the real flat DIT per row        (il/real/zrf.h), odd N2: chain, split
 *                                           body, tile budget
 *   zrb    the real Bluestein per row       (il/real/zrb.h), odd N2: length, inner
 *   door   the row route the tier had (the per-row door at an even N2; at an
 *          odd N2 the promote and the c2c(N2) child)
 * The per-row engines are built by the real door's BUILDERS from the row's
 * token; the handle is the plan's own. At an odd N2 (2026-10-07) the race
 * runs the odd door's sweeps in the row role (il/real/odd_build.h: the
 * flat DIT's chain pick, the Bluestein's lengths and inners), each candidate
 * gated against the route's transform of the plane's first row.
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
 *         | zrf_<chain>[_t]_w<tile> | zrb_<M>_<kind>:<shape>[_t<tw>]   (the odd
 *         engines, in their env pins' spelling with '_' for '/')
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
 * Every N2 (odd N2 since 2026-10-07, r2c; a c2r plan at an odd N2 keeps its
 * c2c child until its own piece). A threaded plan threads the pass by row
 * ranges through an engine's worker clones; the door route keeps its own
 * threading (the per-row door's slabs at an even N2; the odd child is serial).
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

/* THE IN-PLACE RACE's reset (support/race.h's hook): an in-place arm walks its
 * own output, so the plane is re-laid from its seed before every timed
 * sample (docs/roadmap/real_inplace_design.md) */
typedef struct
{
    double *p;
    const double *seed;
    size_t n;
} _il2d_ip_reset_t;
static void _il2d_ip_reset(void *v)
{
    const _il2d_ip_reset_t *r = (const _il2d_ip_reset_t *)v;
    memcpy(r->p, r->seed, r->n * sizeof(double));
}
/* the relative error over the real rows of two planes at their pitches (the pad doubles
 * past N2 are not the transform's: a plane transformed in place keeps its bins there) */
static double _il2d_relerr_rows(const double *a, const double *b, size_t rows, size_t n, size_t pa, size_t pb)
{
    double num = 0.0, den = 0.0;
    size_t r, j;
    for (r = 0; r < rows; r++)
        for (j = 0; j < n; j++)
        {
            const double d = a[r * pa + j] - b[r * pb + j];
            num += d * d;
            den += b[r * pb + j] * b[r * pb + j];
        }
    return den > 0.0 ? sqrt(num / den) : sqrt(num);
}
/* a real-row token's name at the plan's door: door 2 (the pitch twin, il2d_ip == 2) keeps
 * its own pitch-sensitive verdicts as the _p set (real_inplace_design.md) */
static const char *_il2d_tkp(const struct vfft_plan_s *h, const char *base, char *buf, size_t sz)
{
    if (h->il2d_ip != 2) return base;
    snprintf(buf, sz, "%s_p", base);
    return buf;
}

/* the pass itself: the rows kernel in one call, an engine row by row, or the
 * tier's row route */
static void _il2d_rowx_body(const void *v, const double *sre, double *dre)
{
    const struct vfft_plan_s *h = (const struct vfft_plan_s *)v;
    const size_t rn2 = (size_t)h->N2, hp1 = rn2 / 2 + 1, n1 = (size_t)h->N, rp = _il2d_rp(h), cpd = _il2d_cpd(h);
    size_t r;
    (void)hp1;
    if (h->il2d_rx_lm)
    {
        h->il2d_rx_lm(sre, NULL, dre, NULL, NULL, NULL, rp, 0, cpd / 2, 0, n1);
        return;
    }
    if (!h->il2d_rx_eng)
    {
        _il2d_real_rows_fwd_route((struct vfft_plan_s *)h, sre, dre);
        return;
    }
    for (r = 0; r < n1; r++)
        _real_il_exec_any(h->il2d_rx_eng, sre + r * rp, dre + r * cpd);   /* the real row in, its CCE row out: each at its plane's pitch */
}
/* the c2r pass: the backward rows kernel in one call, an engine's backward
 * row by row (the engines run their plan's direction), or the c2r row route */
static void _il2d_rows_bwd_set(struct vfft_plan_s *h, const double *src, size_t P, size_t d, size_t D0, size_t R0, double *y, size_t rpy);
static void _il2d_rowx_body_bwd(const void *v, const double *zsrc, double *dre)
{
    const struct vfft_plan_s *h = (const struct vfft_plan_s *)v;
    const size_t rn2 = (size_t)h->N2, hp1 = rn2 / 2 + 1, n1 = (size_t)h->N, rp = _il2d_rp(h);
    /* the plan's own column-inverse plane may sit at its raced pitch (il2d_real_pitch.h);
     * every other source (a race's plane) is at the CCE pitch */
    const size_t P = (zsrc == h->il2d_rscr && h->il2d_rscr_P) ? h->il2d_rscr_P : h->il2d_col.rn;   /* the plane's pitch (door 2 past hp1) */
    size_t r;
    (void)hp1;
    if (h->il2d_rx_lm)
    {
        h->il2d_rx_lm(zsrc, NULL, dre, NULL, NULL, NULL, P, 0, rp, 0, n1);
        return;
    }
    if (!h->il2d_rx_eng)
    {
        if (P != hp1)
            _il2d_rows_bwd_set((struct vfft_plan_s *)h, zsrc, P, 0, 1, n1, dre, 0);   /* the route, row by row at the pitch */
        else
            _il2d_real_rows_bwd_route((struct vfft_plan_s *)h, zsrc, dre);
        return;
    }
    for (r = 0; r < n1; r++)
        _real_il_exec_any(h->il2d_rx_eng, zsrc + r * 2 * P, dre + r * rp);
}

/* THE ROW PASS OVER A ROW SET: the rows {d + j D0, j < R0} of the real plane (pitch N2)
 * to R0 consecutive CCE rows at pitch P (fwd), or back (bwd) -- the plan's row engine on
 * that set: the rows kernel at the set's stride, an engine per row, the odd-N2 route per
 * row, or the door batch's K = 1 inner per row. d = 0, D0 = 1, R0 = N1 is the whole pass
 * at pitch P. Shared by the fused walk (il2d_real_fuse.h) and the pitch forms
 * (il2d_real_pitch.h). */
static inline void _tc_one(struct vfft_plan_s *in, vfft_dir_t dir, double *s, double *d);   /* il/il_execute.h */
static void _il2d_rows_fwd_set(struct vfft_plan_s *h, const double *x, size_t d, size_t D0, size_t R0,
                               double *out, size_t P)
{
    const size_t N2 = (size_t)h->N2, hp1 = N2 / 2 + 1, rp = _il2d_rp(h);   /* the real plane's pitch */
    size_t j;
    if (h->il2d_rx_on && h->il2d_rx_lm)
    {
        h->il2d_rx_lm(x + d * rp, NULL, out, NULL, NULL, NULL, D0 * rp, 0, P, 0, R0);
        return;
    }
    if (h->il2d_rx_on && h->il2d_rx_eng)
    {
        for (j = 0; j < R0; j++)
            _real_il_exec_any(h->il2d_rx_eng, x + (d + j * D0) * rp, out + j * 2 * P);
        return;
    }
    if (h->il2d_oddn2)
    {
        double *b1 = h->il2d_orbuf, *b2 = h->il2d_orbuf + 2 * N2;
        for (j = 0; j < R0; j++)
        {
            _il2d_row_promote(x + (d + j * D0) * rp, b1, N2);
            vfft_execute((vfft_plan)h->il2d_row, VFFT_FORWARD, b1, NULL, b2, NULL);
            memcpy(out + j * 2 * P, b2, 2 * hp1 * sizeof(double));
        }
        return;
    }
    {
        struct vfft_plan_s *in = h->il2d_row->tcb0 ? h->il2d_row->tcb0 : h->il2d_row->tcb;
        for (j = 0; j < R0; j++)
            _tc_one(in, VFFT_FORWARD, (double *)(x + (d + j * D0) * rp), out + j * 2 * P);
    }
}
static void _il2d_rows_bwd_set(struct vfft_plan_s *h, const double *src, size_t P, size_t d, size_t D0, size_t R0,
                               double *y, size_t rpy)
{   /* rpy = the real plane's row pitch, 0 = the plan's own (a rank-3 caller lands rows in its volume) */
    const size_t N2 = (size_t)h->N2, hp1 = N2 / 2 + 1, rp = rpy ? rpy : _il2d_rp(h);
    size_t j;
    if (h->il2d_rx_on && h->il2d_rx_lm)
    {
        h->il2d_rx_lm(src, NULL, y + d * rp, NULL, NULL, NULL, P, 0, D0 * rp, 0, R0);
        return;
    }
    if (h->il2d_rx_on && h->il2d_rx_eng)
    {
        for (j = 0; j < R0; j++)
            _real_il_exec_any(h->il2d_rx_eng, src + j * 2 * P, y + (d + j * D0) * rp);
        return;
    }
    if (h->il2d_oddn2)
    {
        double *b1 = h->il2d_orbuf, *b2 = h->il2d_orbuf + 2 * N2;
        for (j = 0; j < R0; j++)
        {
            _il2d_row_extend(src + j * 2 * P, b1, N2, hp1);
            vfft_execute((vfft_plan)h->il2d_row, VFFT_BACKWARD, b1, NULL, b2, NULL);
            _il2d_row_re(b2, y + (d + j * D0) * rp, N2);
        }
        return;
    }
    {
        struct vfft_plan_s *in = h->il2d_row->tcb0 ? h->il2d_row->tcb0 : h->il2d_row->tcb;
        for (j = 0; j < R0; j++)
            _tc_one(in, VFFT_BACKWARD, (double *)(src + j * 2 * P), y + (d + j * D0) * rp);
    }
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

/* THE THREADED ROW PASS (2026-10-06, the thread guard lifted): rows [lo, hi)
 * per worker -- the rows kernel over its rows (a lane is one row; the grouping
 * into vectors never crosses lanes, so the split is bitwise), an engine
 * through worker t's CLONE il2d_rxw[t-1] (one plan instance per worker,
 * il2d_real_mt.md §6.2: the same token replayed from the engines' store). The
 * door route is not split here: the door's batch threads itself. Every worker
 * enters its body at the plan's stack state, as the serial pass does. */
typedef struct
{
    const struct vfft_plan_s *h;
    const double *s;
    double *d;
    size_t lo, hi;
    int tid, bwd;
} _il2d_rowx_mt_t;
static void _il2d_rowx_range(const void *v, const double *s, double *d)
{
    const _il2d_rowx_mt_t *a = (const _il2d_rowx_mt_t *)v;
    const struct vfft_plan_s *h = a->h;
    const size_t rp = _il2d_rp(h), cpd = _il2d_cpd(h);   /* the real plane's and the CCE plane's row pitches */
    size_t r;
    if (h->il2d_rx_lm)
    {
        if (a->bwd)
            h->il2d_rx_lm(s + a->lo * cpd, NULL, d + a->lo * rp, NULL, NULL, NULL, cpd / 2, 0, rp, 0, a->hi - a->lo);
        else
            h->il2d_rx_lm(s + a->lo * rp, NULL, d + a->lo * cpd, NULL, NULL, NULL, rp, 0, cpd / 2, 0, a->hi - a->lo);
        return;
    }
    {
        struct vfft_plan_s *e = a->tid > 0 ? h->il2d_rxw[a->tid - 1] : h->il2d_rx_eng;
        if (a->bwd)
            for (r = a->lo; r < a->hi; r++)
                _real_il_exec_any(e, s + r * cpd, d + r * rp);
        else
            for (r = a->lo; r < a->hi; r++)
                _real_il_exec_any(e, s + r * rp, d + r * cpd);
    }
}
static void _il2d_rowx_mt_tramp(void *v)
{
    _il2d_rowx_mt_t *a = (_il2d_rowx_mt_t *)v;
    if (a->h->il2d_rx_stk < 0)
        _il2d_rowx_range(a, a->s, a->d);
    else
        _il2d_rowx_call(_il2d_rowx_range, a, a->s, a->d, a->h->il2d_rx_stk);
}
/* the workers the row pass can use: the plan's T under the pool's clamp, every
 * worker at least two rows (the rows kernel's floor), an engine only as far as
 * its clones go; 1 = the pass runs serial (the door route always: its batch
 * threads itself) */
static int _il2d_rowx_T(const struct vfft_plan_s *h)
{
    int T;
    if (h->nthreads <= 1 || (!h->il2d_rx_lm && !h->il2d_rx_eng))
        return 1;
    T = thread_pool_workers_for(h->nthreads);
    if (h->il2d_rx_lm)
    {
        if ((size_t)T > (size_t)h->N / 2)
            T = (int)((size_t)h->N / 2);
    }
    else
    {
        if (T > h->il2d_rxw_n + 1)
            T = h->il2d_rxw_n + 1;
        if (T > h->N)
            T = h->N;
    }
    return T < 1 ? 1 : T;
}
/* 1 = the pass ran threaded */
static int _il2d_rowx_mt(struct vfft_plan_s *h, const double *s, double *d, int bwd)
{
    _il2d_rowx_mt_t a[THREAD_POOL_MAX_DISPATCH];
    const int T = _il2d_rowx_T(h);
    const size_t n1 = (size_t)h->N;
    int t;
    if (T < 2)
        return 0;
    for (t = 0; t < T; t++)
    {
        a[t].h = h; a[t].s = s; a[t].d = d; a[t].tid = t; a[t].bwd = bwd;
        a[t].lo = n1 * (size_t)t / (size_t)T;
        a[t].hi = n1 * (size_t)(t + 1) / (size_t)T;
    }
    thread_pool_run(T, _il2d_rowx_mt_tramp, a, sizeof a[0]); /* caller = a[0] */
    _vfft_il2d_row_mt_count++; /* engagement, see vfft.c */
    return 1;
}
static void _il2d_rowx_fwd(struct vfft_plan_s *h, const double *sre, double *dre)
{
    if (_il2d_rowx_mt(h, sre, dre, 0))
        return;
    if (h->il2d_rx_stk < 0)
        _il2d_rowx_body(h, sre, dre);
    else
        _il2d_rowx_call(_il2d_rowx_body, h, sre, dre, h->il2d_rx_stk);
}
static void _il2d_rowx_bwd(struct vfft_plan_s *h, const double *zsrc, double *dre)
{
    if (_il2d_rowx_mt(h, zsrc, dre, 1))
        return;
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
    c->placement = cfg->placement;   /* the plan's placement: in place, the engines' in-place forms */
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
    else if (e->zrf)
    {   /* the real flat DIT (odd N2): the VFFT_ZRF pin's spelling, '_' for its '/' */
        char cs[48];
        vfft_zrf_chain_str(e->zrf->R, e->zrf->K, cs, sizeof cs);
        snprintf(b, n, "zrf_%s%s_w%d", cs, e->zrf->nomsz ? "_t" : "", e->zrf->tile > 0 ? e->zrf->tile : 0);
    }
    else if (e->zrb)
    {   /* the real Bluestein (odd N2): the VFFT_ZRB pin's spelling, '_' for its '/' */
        if (e->zrb->itw > 0)
            snprintf(b, n, "zrb_%d_%s:%s_t%d", e->zrb->M, e->zrb->ikind, e->zrb->ishape, e->zrb->itw);
        else
            snprintf(b, n, "zrb_%d_%s:%s", e->zrb->M, e->zrb->ikind, e->zrb->ishape);
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
    {   /* OUT OF PLACE ONLY (real_inplace_design.md): its two planes are __restrict__, and a
         * lone last row re-runs the row before it -- in place that row is already overwritten */
        *lm = (N1 >= 2 && cfg->placement != VFFT_INPLACE) ? (c2r ? vfft_il2d_rows_bwd_fn(N2) : vfft_il2d_rows_fn(N2)) : NULL;
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
    else if (!strncmp(tok, "zrf_", 4))
    {   /* zrf_<chain>[_t]_w<tile>: the real flat DIT at an odd N2 (il/real/zrf.h) */
        int R[VFFT_ILFD_MAX_K], K = 0, nomsz = 0, tile = 0;
        long prod = 1;
        const char *q = tok + 4;
        while (*q && K < VFFT_ILFD_MAX_K)
        {
            char *end;
            long v = strtol(q, &end, 10);
            if (end == q || v < 3 || !(v & 1))
                break;
            R[K++] = (int)v;
            prod *= v;
            q = end;
            if (*q == '.')
                q++;
            else
                break;
        }
        if (q[0] == '_' && q[1] == 't' && q[2] == '_')
        {
            nomsz = 1;
            q += 2;
        }
        if (K >= 2 && prod == N2 && q[0] == '_' && q[1] == 'w')
        {
            char *end;
            tile = (int)strtol(q + 2, &end, 10);
            if (!*end && tile >= 0)
                *eng = _zrf_build_plan(&c, N2, R, K, nomsz, tile);
        }
    }
    else if (!strncmp(tok, "zrb_", 4))
    {   /* zrb_<M>_<kind>:<shape>[_t<tw>]: the real Bluestein at an odd N2 (il/real/zrb.h), its
         * inner in the prime route's spelling (il/rank1/k1_commit.h) */
        char kind[8], shape[64], *end;
        const char *q = tok + 4, *s;
        size_t n;
        int M = (int)strtol(q, &end, 10), tw = 0;
        if (end != q && *end == '_' && M >= vfft_zrb_min_m(N2))
        {
            s = end + 1;
            n = strcspn(s, ":");
            if (s[n] && n > 0 && n < sizeof kind)
            {
                memcpy(kind, s, n);
                kind[n] = 0;
                s += n + 1;
                n = strcspn(s, "_");
                if (n > 0 && n < sizeof shape)
                {
                    _ilprime_inner_desc_t d;
                    memcpy(shape, s, n);
                    shape[n] = 0;
                    s += n;
                    if (s[0] == '_' && s[1] == 't')
                    {
                        tw = (int)strtol(s + 2, &end, 10);
                        s = end;
                    }
                    if (!*s && tw >= 0 && _ilprime_desc_parse(&d, kind, shape, tw))
                        *eng = _zrb_build_plan(&c, N2, M, &d);
                }
            }
        }
    }
    return *eng != NULL;
}

static void _il2d_rowx_clones_drop(struct vfft_plan_s **w, int n)
{
    int t;
    if (!w)
        return;
    for (t = 0; t < n; t++)
        if (w[t])
            vfft_destroy((vfft_plan)w[t]);
    free(w);
}
/* an engine's worker clones for T workers (T-1 of them), by its own token
 * replayed from the engines' store (the recipe the primary banked; never a
 * race): one plan instance per worker. Each clone's token is checked against
 * the primary's (route equivalence); on any failure none are kept and the
 * engine runs its rows serial. 1 = done (nothing to clone counts), 0 = failed. */
static int _il2d_rowx_clones(const vfft_config_t *cfg, struct vfft_wisdom_s *S, int N1, int N2, const char *tok,
                             int T, int c2r, struct vfft_plan_s ***out, int *out_n)
{
    const int n = (T > THREAD_POOL_MAX_DISPATCH ? THREAD_POOL_MAX_DISPATCH : T) - 1;
    vfft_config_t c = *cfg;
    struct vfft_plan_s **arr;
    int t;
    *out = NULL;
    *out_n = 0;
    if (n <= 0 || !tok || !tok[0] || !strcmp(tok, "lm") || !strcmp(tok, "door"))
        return 1;
    c.recalibrate = 0;
    arr = (struct vfft_plan_s **)calloc((size_t)n, sizeof *arr);
    if (!arr)
        return 0;
    for (t = 0; t < n; t++)
    {
        vfft_il2p_fn lm = NULL;
        char nm[48];
        if (!_il2d_rowx_build(&c, S, N1, N2, tok, &lm, &arr[t], c2r) || lm || !arr[t])
        {
            _vfft_warn("il2d real rows: clone %d of %s failed to build at %dx%d -- the engine runs its rows serial", t, tok, N1, N2);
            _il2d_rowx_clones_drop(arr, t + 1);
            return 0;
        }
        _il2d_rowx_name(NULL, arr[t], nm, sizeof nm);
        if (strcmp(nm, tok))
        {
            _vfft_warn("il2d real rows: clone %d of %s is %s at %dx%d -- the engine runs its rows serial", t, tok, nm, N1, N2);
            _il2d_rowx_clones_drop(arr, t + 1);
            return 0;
        }
    }
    *out = arr;
    *out_n = n;
    return 1;
}

/* the race's arm: one candidate at one stack state installed on the plan, the
 * row pass run in the plan's direction (bwd: the CCE plane z -> the real plane a);
 * at T > 1 the candidate's clones ride with it and the pass threads */
typedef struct
{
    struct vfft_plan_s *h;
    double *a;   /* the real plane: the r2c pass's input, the c2r pass's output */
    double *z;   /* the CCE plane */
    vfft_il2p_fn lm;
    struct vfft_plan_s *eng;
    struct vfft_plan_s **w;   /* the engine's worker clones (T > 1), or NULL */
    int wn;
    int stk, bwd;
    int whole;   /* T > 1: the arm runs the WHOLE transform in serving order (the column pass
                  * as the cell serves it, under the colmt verdict), so the cross-core exchange
                  * between the passes is inside the measurement (il2d_real_mt.md §5.2) */
} _il2d_rowx_arm_t;
static void _il2d_rowx_arm_run(void *v)
{
    _il2d_rowx_arm_t *c = (_il2d_rowx_arm_t *)v;
    c->h->il2d_rx_lm = c->lm;
    c->h->il2d_rx_eng = c->eng;
    c->h->il2d_rxw = c->w;
    c->h->il2d_rxw_n = c->wn;
    c->h->il2d_rx_stk = c->stk;
    c->h->il2d_rx_on = 1;
    if (c->bwd)
    {
        if (c->whole)
        {
            _il2d_real_cols(c->h, c->z, c->h->il2d_rscr, 1);
            _il2d_real_rows_bwd(c->h, c->h->il2d_rscr, c->a);
        }
        else
            _il2d_real_rows_bwd(c->h, c->z, c->a);
    }
    else
    {
        _il2d_real_rows_fwd(c->h, c->a, c->z);
        if (c->whole)
            _il2d_real_cols(c->h, c->z, c->z, 0);
    }
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
    ok = vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, T, tok, name, h->il2d_ip) == 0;
    if (!ok)
    {
        vw2_2d_rl_bank(&W->vw2, N1, N2, c2r, h->il2d_col.R, h->il2d_col.nst, -1, -1, 0,
                       (N1 & (N1 - 1)) ? h->il2d_col.blu : -1, 0.0, ord, T, h->il2d_ip);
        ok = vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, T, tok, name, h->il2d_ip) == 0;
    }
    if (ok)
        ok = vw2_2d_rl_tok_sets(&W->vw2, N1, N2, ord, T, stok, sb, h->il2d_ip) == 0;
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
    const int ip = h->il2d_ip;
    const size_t hp1 = (size_t)N2 / 2 + 1, cp = h->il2d_col.rn, CN = 2 * (size_t)N1 * cp;   /* cp = the CCE plane's pitch */
    const size_t RN = ip ? CN : (size_t)N1 * (size_t)N2;   /* the real plane's extent: in place the one padded plane */
    const char *log = getenv("VFFT_IL2D_LOG");
    const char *tk_rx = c2r ? "rx_c2r" : "rx", *tk_rxs = c2r ? "rxs_c2r" : "rxs";
    const char *ev = c2r ? "VFFT_IL2D_RX_C2R" : "VFFT_IL2D_RX", *dn = c2r ? "c2r " : "";
    vfft_il2p_fn lm = NULL;
    struct vfft_plan_s *eng = NULL;
    h->il2d_rx_lm = NULL;
    h->il2d_rx_eng = NULL;
    h->il2d_rxw = NULL;
    h->il2d_rxw_n = 0;
    h->il2d_rx_stk = 0;
    h->il2d_rx_on = 0;
    if (!h->il2d_rxS)
    {   /* THE ROW ENGINES IN ROLE: their own store, seeded from this cell's real row (rx_*, or rx_c2r_*) */
        vw2_key_t pk;
        vw2__2d_rl_key(&pk, N1, N2, ord, T, ip);
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
                if (eng && h->nthreads > 1)
                    (void)_il2d_rowx_clones(cfg, h->il2d_rxS, N1, N2, tok, h->nthreads, c2r, &h->il2d_rxw, &h->il2d_rxw_n);
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
        const char *tok = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, tk_rx, ip);
        if (tok)
        {
            const char *sv = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, tk_rxs, ip);
            if (_il2d_rowx_build(cfg, h->il2d_rxS, N1, N2, tok, &lm, &eng, c2r))
            {
                h->il2d_rx_lm = lm;
                h->il2d_rx_eng = eng;
                h->il2d_rx_stk = _il2d_plan_stk_parse(sv);
                h->il2d_rx_on = 1;
                if (eng && h->nthreads > 1)
                    (void)_il2d_rowx_clones(cfg, h->il2d_rxS, N1, N2, tok, h->nthreads, c2r, &h->il2d_rxw, &h->il2d_rxw_n);
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
        /* T > 1: every arm runs the whole transform (the c2r reverse pass needs its plane) */
        const int whole = h->nthreads > 1 && (!c2r || h->il2d_rscr != NULL);
        double *a = (double *)vfft_aligned_alloc((RN + 8) * sizeof(double));
        /* r2c in place: the one plane; c2r in place: the rows read a CCE plane and write the real
         * plane (the landing form's call), the in-place call gated below */
        const int ipw = ip && !c2r;
        double *z = ipw ? a : (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
        double *ref = (double *)vfft_aligned_alloc((ON + 8) * sizeof(double));
        double *seed = ipw ? (double *)vfft_aligned_alloc((RN + 8) * sizeof(double)) : NULL;   /* the plane's input, re-laid before every sample */
        double *const out = c2r ? a : z;
        _il2d_ip_reset_t rs;
        if (!a || !z || !ref || (ipw && !seed))
        {
            vfft_aligned_free(a); if (!ipw) vfft_aligned_free(z); vfft_aligned_free(ref); vfft_aligned_free(seed);
            return;
        }
        rs.p = a; rs.seed = seed; rs.n = RN + 8;
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
                    z[j * 2 * cp + 1] = z[j * 2 * cp + 2 * (hp1 - 1) + 1] = 0.0;
            }
            if (ipw)
                memcpy(seed, a, (RN + 8) * sizeof(double));
        }
        _il2d_rowx_cfg(cfg, N2, &c, h->il2d_rxS, c2r);
#define ROWX_ARM(LM, ENG) do { if (na < NARMS) { \
            cand[na].h = h; cand[na].a = a; cand[na].z = z; cand[na].lm = (LM); cand[na].eng = (ENG); cand[na].stk = 0; \
            cand[na].w = NULL; cand[na].wn = 0; cand[na].whole = whole; \
            cand[na].bwd = c2r; _il2d_rowx_name(cand[na].lm, cand[na].eng, names[na], sizeof names[na]); na++; } \
            else if ((ENG) != NULL) vfft_destroy((vfft_plan)(ENG)); } while (0)
        ROWX_ARM(NULL, NULL);   /* arm 0: the row route the tier has */
        /* THE GATE'S REFERENCE: the tier's own row route over the plane (every
         * arm is gated against it below; at T > 1 it is the whole transform) */
        memset(ref, 0, (ON + 8) * sizeof(double));
        if (ipw)
        {   /* in place: the route on a copy of the input plane */
            memcpy(ref, seed, (RN + 8) * sizeof(double));
            cand[0].a = cand[0].z = ref;
        }
        else if (c2r)
            cand[0].a = ref;
        else
            cand[0].z = ref;
        _il2d_rowx_arm_run(&cand[0]);
        cand[0].a = a;
        cand[0].z = z;
        /* WHICH DOOR'S ENGINES RACE is the policy's law of N2's parity
         * (vfft_policy_il2d_rows_even_door / _odd_door, planning/policy_il.h);
         * the mono's band is the engine's own */
        /* the rows kernel: OUT OF PLACE ONLY (real_inplace_design.md: __restrict__ planes, the
         * lone last row re-run) -- never an arm of an in-place race */
        if (N1 >= 2 && !ip && vfft_policy_il2d_rows_even_door(N2) && (c2r ? vfft_il2d_rows_bwd_fn(N2) : vfft_il2d_rows_fn(N2)))
            ROWX_ARM(c2r ? vfft_il2d_rows_bwd_fn(N2) : vfft_il2d_rows_fn(N2), NULL);
        if (vfft_policy_il2d_rows_odd_door(N2))
        {   /* ODD N2 (2026-10-07): the odd door's engines in the row role -- the real
             * mono (below, N2 <= 64), the real flat DIT and the real Bluestein
             * through the odd door's own sweeps (il/real/odd_build.h: the chain
             * pick, the lengths and inners), every candidate gated against the
             * route's ROW PASS of the plane's first row (the pass alone: at T > 1
             * the gate's reference above is the whole transform) and burst-timed,
             * the two fastest of each sweep joining the race. The route (arm 0: the
             * promote and the c2c child) stays the plan's when nothing beats it. */
            const size_t xs = (size_t)N2 + 3;   /* the odd door's row buffers */
            const size_t nin = c2r ? 2 * hp1 : (size_t)N2, nchk = c2r ? (size_t)N2 : (size_t)N2 + 1;
            double *rb = (double *)vfft_aligned_alloc(xs * sizeof(double));
            double *rr = (double *)vfft_aligned_alloc(xs * sizeof(double));
            double *rt = (double *)vfft_aligned_alloc(xs * sizeof(double));
            if (rb && rr && rt)
            {
                struct vfft_plan_s *ho[2] = { NULL, NULL };
                memset(rb, 0, xs * sizeof(double));
                memset(rr, 0, xs * sizeof(double));
                memcpy(rb, c2r ? z : a, nin * sizeof(double));   /* the plane's first row */
                if (c2r)                                          /* its transform by the route's row pass */
                    _il2d_real_rows_bwd_route(h, z, a);           /* (the pass's output plane is scratch here) */
                else
                    _il2d_real_rows_fwd_route(h, a, z);
                memcpy(rr, c2r ? a : z, nchk * sizeof(double));
                if (_zrf_has_chain(N2) && _zrf_chain_sweep(&c, N2, rb, rr, rt, rb, xs, nchk, ho) > 0)
                {
                    ROWX_ARM(NULL, ho[0]);
                    if (ho[1])
                        ROWX_ARM(NULL, ho[1]);
                }
                ho[0] = ho[1] = NULL;
                if (_zrb_ok(N2) && _zrb_sweep(&c, N2, rb, rr, rt, rb, xs, nchk, ho) > 0)
                {
                    ROWX_ARM(NULL, ho[0]);
                    if (ho[1])
                        ROWX_ARM(NULL, ho[1]);
                }
            }
            vfft_aligned_free(rb);
            vfft_aligned_free(rr);
            vfft_aligned_free(rt);
            if (ipw)
                memcpy(a, seed, (RN + 8) * sizeof(double));   /* the route's run transformed the plane */
        }
        {
            /* the even door's engines (zr2c, the pairs, ZTT-r) are candidates at an
             * EVEN N2 only (the policy's law): at an odd N2 their builders still
             * build (a zr2c at N=15 raced its child and banked it into the row
             * engines' store before the gate dropped it), so they are not asked */
            const int even_door = vfft_policy_il2d_rows_even_door(N2);
            struct vfft_plan_s *hz = even_door ? _il2d_rowx_zr2c(&c, h->il2d_rxS, N2, 0) : NULL, *e;
            if (hz)
                ROWX_ARM(NULL, hz);
            if (even_door && (e = _il2d_rowx_zr2c(&c, h->il2d_rxS, N2, 1)) != NULL)
                ROWX_ARM(NULL, e);
            if (N2 <= VFFT_ZRM_MAX_N && (e = _zrm_build_plan(&c, N2)) != NULL)
                ROWX_ARM(NULL, e);
            {
                int pa[VFFT_ZRP_MAX_ARMS][3];
                const int np = even_door ? _zrp_arms(N2, pa, VFFT_ZRP_MAX_ARMS) : 0;
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
        /* the gate: every arm's plane against the tier's own row route (ref) */
        {
            int keep = 1;
            for (i = 1; i < na; i++)
            {
                double e;
                if (ipw)
                    memcpy(a, seed, (RN + 8) * sizeof(double));
                else
                    memset(out, 0, ON * sizeof(double));
                _il2d_rowx_arm_run(&cand[i]);
                e = (ip && c2r) ? _il2d_relerr_rows(out, ref, (size_t)N1, (size_t)N2, _il2d_rp(h), _il2d_rp(h)) : _zrpr_relerr(out, ref, ON);   /* in place the pad doubles are not the transform: the rows alone */
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
        if (ip && c2r)
        {   /* the in-place call too (the destroying form's rows read the plane they write, row by
             * row): every engine from a copy of the CCE plane onto itself, against the reference */
            double *w = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
            const size_t rp = _il2d_rp(h);
            int keep = 1;
            for (i = 1; w && i < na; i++)
            {
                double *sa = cand[i].a, *sz = cand[i].z, e;
                memcpy(w, z, (CN + 8) * sizeof(double));
                cand[i].a = w; cand[i].z = w;
                _il2d_rowx_arm_run(&cand[i]);
                cand[i].a = sa; cand[i].z = sz;
                e = _il2d_relerr_rows(w, ref, (size_t)N1, (size_t)N2, rp, rp);
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
                    fprintf(stderr, "[il2d-real] c2r rows %dx%d: %s in place (row by row) FAILS the gate (rel %.2e) -- dropped\n", N1, N2, names[i], e);
                    if (cand[i].eng)
                        vfft_destroy((vfft_plan)cand[i].eng);
                }
            }
            if (w)
                na = keep;
            vfft_aligned_free(w);
        }
        if (h->nthreads > 1)
        {   /* T > 1: every engine candidate gets its worker clones, and its
             * THREADED pass is gated against the reference too (the clones'
             * output is the primary's by construction; the gate says so) */
            for (i = 1; i < na; i++)
                if (cand[i].eng)
                {
                    double e;
                    if (!_il2d_rowx_clones(cfg, h->il2d_rxS, N1, N2, names[i], h->nthreads, c2r, &cand[i].w, &cand[i].wn))
                        continue;   /* no clones: the engine's rows run serial under this plan */
                    if (ipw)
                        memcpy(a, seed, (RN + 8) * sizeof(double));
                    else
                        memset(out, 0, ON * sizeof(double));
                    _il2d_rowx_arm_run(&cand[i]);
                    e = (ip && c2r) ? _il2d_relerr_rows(out, ref, (size_t)N1, (size_t)N2, _il2d_rp(h), _il2d_rp(h)) : _zrpr_relerr(out, ref, ON);
                    if (!(e < 1e-10))
                    {
                        fprintf(stderr, "[il2d-real] %srows %dx%d T=%d: %s threaded FAILS the gate (rel %.2e) -- its clones dropped\n",
                                dn, N1, N2, h->nthreads, names[i], e);
                        _il2d_rowx_clones_drop(cand[i].w, cand[i].wn);
                        cand[i].w = NULL;
                        cand[i].wn = 0;
                    }
                }
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
            if (ipw && reps > 32) reps = 32;   /* in place the plane grows a factor N2 per pass: a sample stays finite */
            {   /* 9 rounds alternated, median; threaded arms (T > 1) under the
                 * threaded protocol instead: min of 3, two untimed passes per
                 * arm first, no pacing (a paced pool parks its workers) */
                vfft_race_proto_t proto = { 9, reps, VFFT_RACE_MEDIAN, 1, 1, NULL, NULL, 1 };
                if (ipw)
                {   /* the one plane, re-laid before every sample */
                    proto.reset = _il2d_ip_reset;
                    proto.reset_ctx = &rs;
                }
                if (h->nthreads > 1)
                {
                    proto.rounds = 3; proto.agg = VFFT_RACE_MIN; proto.alternate = 0; proto.warm = 2; proto.pace = 0;
                }
                vfft_race_run(&proto, arms, 4 * na, ns);
            }
            for (i = 1; i < 4 * na; i++)
                if (ns[i] < ns[best]) best = i;
            for (s = 1; s < 4; s++)
                if (ns[s] < ns[bd]) bd = s;
            /* the race's margin toward the tier's route at its own best state */
            if (best / 4 != 0 && !vfft_race_beats(ns[best], ns[bd], VFFT_RACE_HYST))
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
            {
                _il2d_rowx_clones_drop(cand[i].w, cand[i].wn);
                vfft_destroy((vfft_plan)cand[i].eng);
            }
        h->il2d_rx_lm = ctx[best].lm;
        h->il2d_rx_eng = ctx[best].eng;
        h->il2d_rxw = ctx[best].w;
        h->il2d_rxw_n = ctx[best].wn;
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
        vfft_aligned_free(a); if (!ipw) vfft_aligned_free(z); vfft_aligned_free(ref); vfft_aligned_free(seed);
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
 * their own passes.
 *
 * AT T > 1 (2026-10-06, the thread guard lifted): the plan races and serves
 * under the cell's colmt verdict -- a leaf by column ranges, the chain through
 * the tier's MT column pass with the leaf in the plan's form and the pass's
 * one state on every dispatched body (no per-stage states). The verdict
 * banks on the T row, as every real token does.
 *
 * THE C2R TWIN (2026-10-05): the same plan over the REVERSE pass (the
 * Hermitian-transpose chain, out of place from the caller's plane onto the
 * column-inverse plane): the forms plain | strided (the natural leaf
 * gathering the natural rows at its stride into the scratch comb, the mids
 * in reverse order) and the backward leaves (b816 | b448: n1cb*_bwd), the
 * per-stage states indexed by chain stage as r2c's are. NO staged form for
 * the reverse pass: the gather is cheap at the real plane's odd pitch (no
 * set conflicts) and a staging only adds traffic -- the forward's walk with
 * a reversed scatter was raced and refuted 2026-10-06 (8-22% over strided
 * at every chain cell but one tie). Raced on the cell's own reverse pass,
 * banked cx_c2r= / cxs_c2r=, pinned by VFFT_IL2D_CX_C2R.
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
    /* the plan's own column-inverse plane may sit at its raced pitch (il2d_real_pitch.h) */
    const size_t Po = (dst == h->il2d_rscr && h->il2d_rscr_P) ? h->il2d_rscr_P : c->rn;
    if (h->il2d_cx_leaf)
    {
        h->il2d_cx_leaf(src, NULL, dst, NULL, NULL, NULL, c->rn, 0, Po, 0, c->rn);
        return;
    }
    if (Po != c->rn)
    {   /* the one-stage chain's kernel out of place at the plane's pitch (the pitch form admits one-kernel passes only) */
        c->b[0](src, NULL, dst, NULL, NULL, NULL, c->rn, 0, Po, 0, c->rn);
        return;
    }
    _il2d_col_exec_st(c, src, dst, 1, NULL);   /* the reverse leaf gathers at its stride: no staged form */
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
/* THE THREADED COLUMN PASS (2026-10-06, the thread guard lifted): under the
 * cell's colmt verdict the plan's form runs threaded -- a leaf by COLUMN
 * RANGES (a column is one lane: bitwise), a chain through the tier's MT column
 * pass (il2d_tier.h) with the natural leaf in the plan's form (strided, or
 * each worker's own staging block) and the pass's one stack state on every
 * dispatched body. The per-stage states are a serial plan's: a threaded pass
 * has one state. 0 = the pass did not thread (the serial form runs). */
typedef struct
{
    const struct vfft_plan_s *h;
    const double *src;
    double *dst;
    size_t lo, hi;
} _il2d_colx_mt_t;
static void _il2d_colx_leaf_range(const void *v, const double *src, double *dst)
{
    const _il2d_colx_mt_t *a = (const _il2d_colx_mt_t *)v;
    const vfft_ilcol_t *c = &a->h->il2d_col;
    a->h->il2d_cx_leaf(src + 2 * a->lo, NULL, dst + 2 * a->lo, NULL, NULL, NULL, c->rn, 0, c->rn, 0, a->hi - a->lo);
}
static void _il2d_colx_leaf_tramp(void *v)
{
    _il2d_colx_mt_t *a = (_il2d_colx_mt_t *)v;
    if (a->h->il2d_cx_stk < 0)
        _il2d_colx_leaf_range(a, a->src, a->dst);
    else
        _il2d_rowx_call(_il2d_colx_leaf_range, a, a->src, a->dst, a->h->il2d_cx_stk);
}
static int _il2d_colx_mt(struct vfft_plan_s *h, const double *src, double *dst, int reverse)
{
    int T;
    if (h->nthreads <= 1 || !h->il2d_col.colmt)
        return 0;
    T = thread_pool_workers_for(h->nthreads);
    if (h->il2d_cx_leaf)
    {
        _il2d_colx_mt_t a[THREAD_POOL_MAX_DISPATCH];
        const size_t cols = h->il2d_col.rn;
        int t;
        if ((size_t)T > cols)
            T = (int)cols;
        if (T < 2)
            return 0;
        for (t = 0; t < T; t++)
        {
            a[t].h = h; a[t].src = src; a[t].dst = dst;
            a[t].lo = cols * (size_t)t / (size_t)T;
            a[t].hi = cols * (size_t)(t + 1) / (size_t)T;
        }
        thread_pool_run(T, _il2d_colx_leaf_tramp, a, sizeof a[0]); /* caller = a[0] */
        _vfft_il2d_col_mt_count++;
        return 1;
    }
    if (h->il2d_cx_perk)
        return 0;
    return _il2d_real_cols_mt(h, src, dst, reverse, h->nthreads);
}
static void _il2d_colx_fwd(struct vfft_plan_s *h, const double *src, double *dst)
{
    if (_il2d_colx_mt(h, src, dst, 0))
        return;
    if (h->il2d_cx_perk)
        _il2d_colx_perk(h, src, dst);
    else if (h->il2d_cx_stk < 0)
        _il2d_colx_body(h, src, dst);
    else
        _il2d_rowx_call(_il2d_colx_body, h, src, dst, h->il2d_cx_stk);
}

/* THE C2R PER-STAGE ENTRIES: the reverse natural pass (_il2d_col_pass_nat,
 * reverse) a stage at a time -- the leaf gathers the natural src into the
 * scratch's comb at its stride, the mids run in reverse chain order in place,
 * stage 0 writes the scratch -> dst; the same kernels, calls and order: bitwise. */
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
    _il2d_nat_leaf_range(src, scr, c->N, c->rn, Rl, c->b[c->nst - 1], c->natperm, 0, (size_t)(c->N / Rl), 1, NULL);
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
    if (_il2d_colx_mt(h, src, dst, 1))
        return;
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
    double *a;   /* T > 1: the real plane the row pass runs with (the whole transform per arm) */
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
    {
        _il2d_real_cols(h, c->z, h->il2d_rscr, 1);   /* the reverse pass, out of place onto the plan's plane */
        if (c->a)
            _il2d_real_rows_bwd(h, h->il2d_rscr, c->a);   /* T > 1: the row pass after it, as the cell serves */
    }
    else
    {
        if (c->a)
            _il2d_real_rows_fwd(h, c->a, c->z);   /* T > 1: the row pass before it, as the cell serves */
        _il2d_real_cols(h, c->z, c->z, 0);
    }
}

#define VFFT_IL2D_CX_MAXFORMS (2 + VFFT_IL2P_COL_MAXLEAF)
/* one body for the two directions: c2r = 1 races the reverse pass (its forms,
 * the backward leaves) out of place onto the plan's column-inverse plane, as
 * the pass serves; the r2c path is the one that stood */
static void _il2d_real_colplan_pick(struct vfft_plan_s *h, struct vfft_wisdom_s *W, const vfft_config_t *cfg,
                                    int N1, int N2, int ord, int T, int c2r)
{
    const size_t hp1 = (size_t)N2 / 2 + 1, cp = h->il2d_col.rn, CN = 2 * (size_t)N1 * cp;   /* cp = the CCE plane's pitch */
    const vfft_ilcol_t *col = &h->il2d_col;
    const int nat = col->nat != 0, nf = (nat && col->natstage && !c2r) ? 2 : 1;   /* the reverse pass has no staged form */
    const char *fname[2];
    const char *lname[VFFT_IL2P_COL_MAXLEAF];
    const char *log = getenv("VFFT_IL2D_LOG");
    char tkb_cx[16], tkb_cxs[16];
    const char *tk_cx = _il2d_tkp(h, c2r ? "cx_c2r" : "cx", tkb_cx, sizeof tkb_cx);
    const char *tk_cxs = _il2d_tkp(h, c2r ? "cxs_c2r" : "cxs", tkb_cxs, sizeof tkb_cxs);
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
        const char *tok = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, tk_cx, h->il2d_ip);
        if (tok && _il2d_colx_form_set(h, tok, strlen(tok), fname, nf, lname, nl, c2r) &&
            _il2d_colx_stk_set(h, vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, tk_cxs, h->il2d_ip)))
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
        vfft_race_proto_t proto = proto0; /* 9 rounds alternated, median; at T > 1 the threaded protocol below */
        int na = 0, best = 0, bc = 0, reps, i, f, u;
        const size_t RN = h->il2d_ip ? CN : (size_t)N1 * (size_t)N2;   /* the real plane's extent: in place the one padded plane (the T > 1 arms run the row pass on it) */
        double *z = (double *)vfft_aligned_alloc((CN + 8) * sizeof(double));
        double *const out = c2r ? h->il2d_rscr : z;   /* the pass's output plane */
        /* T > 1: the whole transform per arm -- the row pass (bound before this plan) runs with
         * the arm in serving order, so the exchange between the passes is measured */
        double *a = (h->nthreads > 1 && h->il2d_rx_on) ? (double *)vfft_aligned_alloc((RN + 8) * sizeof(double)) : NULL;
        if (!z)
        {
            vfft_aligned_free(a);
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
            if (a)
                for (j = 0; j < RN + 8; j++)
                {
                    sd = sd * 1664525u + 1013904223u;
                    a[j] = (double)(sd >> 8) / (double)(1u << 24) - 0.5;
                }
            if (a && c2r)
                for (j = 0; j < (size_t)N1; j++)   /* a CCE plane: the DC and Nyquist bins real */
                    z[j * 2 * cp + 1] = z[j * 2 * cp + 2 * (hp1 - 1) + 1] = 0.0;
        }
        for (f = 0; f < nf; f++)
        {
            memset(&cand[na], 0, sizeof cand[na]);
            cand[na].h = h; cand[na].z = z; cand[na].a = a; cand[na].st = f; cand[na].stk = -1; cand[na].bwd = c2r;
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
                    cand[na].h = h; cand[na].z = z; cand[na].a = a; cand[na].stk = -1; cand[na].bwd = c2r;
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
        if (h->nthreads > 1)
        {   /* the pass threads under the colmt verdict: min of 3, two untimed
             * passes per arm first, no pacing (a paced pool parks its workers) */
            proto.rounds = 3; proto.agg = VFFT_RACE_MIN; proto.alternate = 0; proto.warm = 2; proto.pace = 0;
        }
        vfft_race_run(&proto, arms, 4 * na, ns);
        for (i = 1; i < 4 * na; i++)
            if (ns[i] < ns[best]) best = i;
        for (i = 1; i < 4 * nf; i++)
            if (ns[i] < ns[bc]) bc = i;
        /* the race's margin toward the chain: a leaf serves where it clearly wins */
        if (ctx[best].leaf && !vfft_race_beats(ns[best], ns[bc], VFFT_RACE_HYST))
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
        if (!ctx[best].leaf && nat && col->nst >= 2 && u >= 0 && h->nthreads <= 1)
        {   /* the per-kernel states: each stage's four with the others held, in
             * chain order (a serial pass's: a threaded pass has one state) */
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
                if (kb != ks[s] && vfft_race_beats(kns[kb], kns[ks[s]], VFFT_RACE_HYST))
                    ks[s] = (signed char)kb; /* the race's margin toward the pass's state */
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
                            vfft_race_beats(kns[1], kns[0], VFFT_RACE_HYST) ? "per stage" : "one state");
                if (vfft_race_beats(kns[1], kns[0], VFFT_RACE_HYST))
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
        vfft_aligned_free(a);
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
    const size_t hp1 = (size_t)N2 / 2 + 1, RN = (size_t)N1 * (size_t)N2, CN = 2 * (size_t)N1 * h->il2d_col.rn;
    const vfft_ilcol_t *col = &h->il2d_col;
    const int one_stage = col->nst == 1 && !col->nat && !col->blu && !col->tpc && (col->wl == 0 || col->wl == col->N) &&
                          col->L[0] == col->N && col->b[0] != NULL;
    const char *lname[VFFT_IL2P_COL_MAXLEAF];
    const char *log = getenv("VFFT_IL2D_LOG");
    const int ip = h->il2d_ip;
    const size_t RNp = ip ? CN : RN;   /* the real plane's extent: in place the one padded plane */
    char tkb_d[16], tkb_ds[16];
    const char *tk_cxd = _il2d_tkp(h, "cxd_c2r", tkb_d, sizeof tkb_d), *tk_cxds = _il2d_tkp(h, "cxds_c2r", tkb_ds, sizeof tkb_ds);
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
        const char *tok = vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, tk_cxd, h->il2d_ip);
        if (tok)
        {
            if (!strcmp(tok, "off"))
                return; /* the scratch form won this cell's race */
            if (_il2d_cxd_form_set(h, tok, strlen(tok), one_stage, lname, nl))
            {
                h->il2d_cxd_stk = _il2d_plan_stk_parse(vw2_2d_rl_tok_gets(&W->vw2, N1, N2, ord, T, tk_cxds, h->il2d_ip));
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
        double *y = ip ? z : (double *)vfft_aligned_alloc((RN + 8) * sizeof(double));   /* in place: the one plane */
        double *ref = (double *)vfft_aligned_alloc((RNp + 8) * sizeof(double));
        if (!z || !y || !ref)
        {
            vfft_aligned_free(z); if (!ip) vfft_aligned_free(y); vfft_aligned_free(ref);
            return;
        }
        seed.z = z;
        seed.n = CN + 8;
        seed.sd = 0x9e3779b9u ^ (unsigned)N1 ^ ((unsigned)N2 << 12);
        /* the reference: the scratch form's plane */
        ctx[0].h = h; ctx[0].z = z; ctx[0].y = ref; ctx[0].leaf = NULL; ctx[0].stk = 0;
        _il2d_cxd_seed(&seed);
        memset(ref, 0, (RNp + 8) * sizeof(double));
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
            if (!ip)
                memset(y, 0, RN * sizeof(double));
            _il2d_cxd_arm_run(&cand[nc]);
            e = ip ? _il2d_relerr_rows(y, ref, (size_t)N1, (size_t)N2, _il2d_rp(h), _il2d_rp(h)) : _zrpr_relerr(y, ref, RN);
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
            vfft_aligned_free(z); if (!ip) vfft_aligned_free(y); vfft_aligned_free(ref);
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
        /* the race's margin toward the scratch form: the input is destroyed where that clearly pays */
        win = vfft_race_beats(ns[best], ns[0], VFFT_RACE_HYST);
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
        _il2d_real_plan_bank(h, W, cfg, N1, N2, ord, T, tk_cxd, win ? names[ci] : "off", tk_cxds, sb, 1);
        vfft_aligned_free(z); if (!ip) vfft_aligned_free(y); vfft_aligned_free(ref);   /* in place: y is z */
    }
}

#endif /* VFFT_IL2D_REAL_PLAN_H */
