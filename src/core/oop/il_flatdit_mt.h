/* il_flatdit_mt.h — the FLAT mixed-radix DIT's intra-transform threading
 * (2026-09-07; docs/design/3D_mt_il_strategy.md declares the method).
 *
 * Every stage of the bound executor (il_flatdit.h) is a set of independent
 * UNITS — the leaf's columns, a mid stage's blocks, a tail stage's groups —
 * and the tile axis already walks the suffix stages [tlo, thi) as
 * self-contained tiles. Two partition arms, both pure loop restrictions of
 * the serving call lists (same kernels, same tables, same per-element order;
 * threaded output is bitwise the serial output, gated):
 *   BLOCKS arm (mt=1): every stage split over the workers by units, one
 *              dispatch per stage in list order. No width needed.
 *   TILES arm  (mt=2): the wide prefix stages [0, tlo) split by units, then
 *              ONE dispatch in which workers take disjoint TILE ranges and
 *              walk the suffix depth-first per tile (exchange-free inside a
 *              worker's tiles), then the wide tail stages [thi, K) split by
 *              units. The scrambled backward tiles its first K-tcut
 *              transposed records the same way. Needs tw > 0, ntile >= 2.
 * The per-worker unit records are BOUND at plan time (vfft_ilfd_mt_bind):
 * one record per (list, stage, worker) with the counts of its unit range
 * and the base steps of one unit, executed by _ilfd_call with the range
 * start as the tile index — execute is pure dispatch. The plan's own
 * staging plane is shared (workers write disjoint units); the tables are
 * read-only; there is nothing to clone.
 * The verdict — serial, blocks, or tiles at a width — is RACED at the
 * plan's T with steady-state samples (REPS executes after warm passes;
 * every tile width the chain offers is an arm of the tiles family, because
 * the width raced at one thread need not be the threaded one) and banked
 * on the cell's kind-3 IL row as il_mt= (0|1|2), il_mt_t= (the T raced
 * at), il_mt_tw= (the tiles arm's width); il_tw= stays the one-thread
 * verdict. A verdict serves only at its own T. Engagement counter:
 * vfft_ilfd_mt_passes() (vfft.c). VFFT_ILFD_MT=0|1|2 pins, never banks.
 *
 * Position: after il_flatdit.h and support/threads.h + support/race.h in
 * vfft.c; engine-only (no plan handle, no wisdom) — the front door's
 * replay-or-race lives in oop/k1_commit.h. */
#ifndef VFFT_IL_FLATDIT_MT_H
#define VFFT_IL_FLATDIT_MT_H

extern long _vfft_ilfd_mt_count; /* vfft.c: the engagement counter */

typedef struct {
    vfft_ilfd_call_t c;   /* the unit-range record: counts = the range, steps = one unit */
    size_t t;             /* the range start, passed as the tile index */
    int on;               /* 0 = an empty range (more workers than units) */
} _ilfd_mt_rec_t;

typedef struct {
    int T, K;
    _ilfd_mt_rec_t *f, *b, *tr;   /* K*T each: cf, cb, ct chunks (wide-form records) */
} _ilfd_mt_bind_t;

/* the unit view of one WIDE record: dst = src with the base steps of ONE
 * unit; *nunits = how many units the stage has; *cnt = which count field
 * the range goes into (0 count, 1 Gs, 2 OGs, 3 ngrp, 4 nb). The forms are
 * told apart exactly as _ilfd_tile_rec tells them: a leaf (columns), a
 * t2csgn (groups; with a group ORDER the order table itself is what steps),
 * msz (blocks), t2cp (blocks), the per-column tail (groups), the natural
 * BLK last stage (blocks with an absolute base table). */
static inline void _ilfd_unit_rec(const vfft_ilfd_plan_t *p, int s, int leaf,
                                  const vfft_ilfd_call_t *src, vfft_ilfd_call_t *dst,
                                  size_t *nunits, int *cnt)
{
    const size_t L = src->L;
    const size_t recs = s > 0 ? (size_t)(p->R[s] - 1) * 8 : 0;
    *dst = *src;
    dst->in_tstep = dst->out_tstep = dst->tw_tstep = dst->t2_tstep = dst->a1_tstep = dst->g_tstep = 0;
    if (leaf) {                          /* columns: in/out advance one complex */
        dst->in_tstep = dst->out_tstep = 2;
        *nunits = src->count; *cnt = 0;
        return;
    }
    if (src->op == _ILFD_ONE) {
        if (src->a1) {                   /* t2csgn: groups of count blocks */
            *nunits = src->Gs; *cnt = 1;
            if (!src->a3) {              /* block order: the bases step per group */
                dst->in_tstep = 2 * src->count * L;
                dst->a1_tstep = src->count;
                dst->t2_tstep = 8;
            }                            /* natural-base order: a3 is shifted per range (below) */
            return;
        }
        if (src->in_sel == _ILFD_NUL) {  /* msz: blocks, in place on the plane */
            dst->out_tstep = 2 * L; dst->tw_tstep = recs;
            *nunits = src->Gs; *cnt = 1;
            return;
        }
        dst->in_tstep = dst->out_tstep = 2 * L; dst->tw_tstep = recs;   /* t2cp: blocks */
        *nunits = src->OGs; *cnt = 2;
        return;
    }
    if (src->op == _ILFD_COL) {          /* the per-column tail: groups, absolute indices */
        dst->g_tstep = 1;
        *nunits = src->ngrp; *cnt = 3;
        return;
    }
    dst->in_tstep = 2 * L; dst->tw_tstep = src->tw_step;                /* BLK: blocks */
    *nunits = src->nb; *cnt = 4;
}

static inline void _ilfd_mt_chunk_list(const vfft_ilfd_plan_t *p, int T, int transposed,
                                       int bwd, _ilfd_mt_rec_t *out)
{
    int i, w;
    for (i = 0; i < p->K; i++) {
        vfft_ilfd_call_t wide, unit;
        size_t nunits = 0;
        int cnt = 0, s, leaf;
        if (transposed) {
            leaf = (i == p->K - 1);
            s = leaf ? 0 : p->K - 1 - i;
            if (leaf) {
                memset(&wide, 0, sizeof wide);
                wide.op = _ILFD_ONE; wide.fn = p->lb; wide.in_sel = _ILFD_STG; wide.out_sel = _ILFD_ZOUT;
                wide.Ls = wide.OLs = wide.count = p->D[0];
            } else
                _ilfd_bind_stage_T(p, s, &wide);
        } else {
            s = i; leaf = (i == 0);
            _ilfd_bind_stage_dir(p, s, bwd, &wide);
        }
        _ilfd_unit_rec(p, s, leaf, &wide, &unit, &nunits, &cnt);
        for (w = 0; w < T; w++) {
            _ilfd_mt_rec_t *r = &out[i * T + w];
            const size_t lo = nunits * (size_t)w / (size_t)T, hi = nunits * (size_t)(w + 1) / (size_t)T;
            r->c = unit;
            r->t = lo;
            r->on = (hi > lo);
            switch (cnt) {
            case 0: r->c.count = hi - lo; break;
            case 1: r->c.Gs = hi - lo; break;
            case 2: r->c.OGs = hi - lo; break;
            case 3: r->c.ngrp = hi - lo; break;
            default: r->c.nb = hi - lo; break;
            }
            if (unit.op == _ILFD_ONE && unit.a1 && unit.a3) {   /* the group ORDER steps */
                r->c.a3 = (double *)((const size_t *)unit.a3 + lo);
                r->t = 0;
            }
            if (unit.op == _ILFD_BLK) {                          /* the absolute base table */
                r->c.obase = unit.obase + lo;
            }
        }
    }
}

/* bind the per-worker unit records for T workers (T-1 helpers + the caller);
 * one allocation, freed by vfft_ilfd_destroy through p->mtb. Idempotent
 * for the same T. Returns 1, 0 = out of memory (MT declines). */
static inline int vfft_ilfd_mt_bind(vfft_ilfd_plan_t *p, int T)
{
    _ilfd_mt_bind_t *b;
    const size_t nrec = (size_t)p->K * (size_t)T;
    if (T < 2) return 0;
    if (p->mtb && ((_ilfd_mt_bind_t *)p->mtb)->T == T) return 1;
    free(p->mtb); p->mtb = NULL;
    b = (_ilfd_mt_bind_t *)malloc(sizeof *b + 3 * nrec * sizeof(_ilfd_mt_rec_t));
    if (!b) return 0;
    b->T = T; b->K = p->K;
    b->f = (_ilfd_mt_rec_t *)(b + 1);
    b->b = b->f + nrec;
    b->tr = b->b + nrec;
    _ilfd_mt_chunk_list(p, T, 0, 0, b->f);
    _ilfd_mt_chunk_list(p, T, 0, 1, b->b);
    if (p->scr_ok) _ilfd_mt_chunk_list(p, T, 1, 0, b->tr);
    else memset(b->tr, 0, nrec * sizeof *b->tr);
    p->mtb = b;
    return 1;
}

/* ── the dispatches ─────────────────────────────────────────────────── */
typedef struct {
    const vfft_ilfd_plan_t *p;
    const double *zin;
    double *zout;
    const _ilfd_mt_rec_t *rec;        /* stage dispatch: this worker's record */
    const vfft_ilfd_call_t *tiled;    /* tile dispatch: the tiled records */
    int ntiled;
    size_t t_lo, t_hi;
} _ilfd_mt_arg;

static void _ilfd_mt_tramp(void *v)
{
    const _ilfd_mt_arg *a = (const _ilfd_mt_arg *)v;
    if (a->rec) {
        if (a->rec->on) _ilfd_call(a->p, &a->rec->c, a->rec->t, a->zin, a->zout);
        return;
    }
    {
        size_t t;
        int i;
        for (t = a->t_lo; t < a->t_hi; t++)
            for (i = 0; i < a->ntiled; i++)
                _ilfd_call(a->p, a->tiled + i, t, a->zin, a->zout);
    }
}

/* one stage over T workers: worker w runs record (i, w) */
static inline void _ilfd_mt_stage(const vfft_ilfd_plan_t *p, const _ilfd_mt_rec_t *recs, int i,
                                  const double *zin, double *zout, int T)
{
    _ilfd_mt_arg a[STRIDE_POOL_MAX_DISPATCH];
    int w;
    for (w = 0; w < T; w++) {
        a[w].p = p; a[w].zin = zin; a[w].zout = zout;
        a[w].rec = &recs[i * T + w]; a[w].tiled = 0; a[w].ntiled = 0; a[w].t_lo = a[w].t_hi = 0;
    }
    stride_pool_run(T, _ilfd_mt_tramp, a, sizeof a[0]);
}

/* the tiled records over T workers: worker w walks tiles [lo_w, hi_w) depth-first */
static inline void _ilfd_mt_tiles(const vfft_ilfd_plan_t *p, const vfft_ilfd_call_t *tiled, int n,
                                  size_t ntile, const double *zin, double *zout, int T)
{
    _ilfd_mt_arg a[STRIDE_POOL_MAX_DISPATCH];
    int w;
    for (w = 0; w < T; w++) {
        a[w].p = p; a[w].zin = zin; a[w].zout = zout; a[w].rec = 0;
        a[w].tiled = tiled; a[w].ntiled = n;
        a[w].t_lo = ntile * (size_t)w / (size_t)T; a[w].t_hi = ntile * (size_t)(w + 1) / (size_t)T;
    }
    stride_pool_run(T, _ilfd_mt_tramp, a, sizeof a[0]);
}

/* Returns 1 when it ran threaded, 0 when the caller must run serial. The
 * binding is for the plan's own T; a live pool clamped below it declines
 * (the bound ranges would not cover the units). */
static inline int vfft_ilfd_execute_mt(const vfft_ilfd_plan_t *p, const double *zin, double *zout,
                                       int bwd)
{
    const _ilfd_mt_bind_t *b = (const _ilfd_mt_bind_t *)p->mtb;
    const int T = stride_pool_workers_for(p->mt_t);
    const _ilfd_mt_rec_t *recs;
    const vfft_ilfd_call_t *list;
    int i;
    if (!b || T < 2 || T != b->T || p->mt <= 0) return 0;
    if (bwd && p->scr) { recs = b->tr; list = p->ct; if (!p->scr_ok) return 0; }
    else if (bwd)      { recs = b->b;  list = p->cb; }
    else               { recs = b->f;  list = p->cf; }
    if (p->mt == 1) {                                 /* BLOCKS: every stage by units */
        for (i = 0; i < p->K; i++) _ilfd_mt_stage(p, recs, i, zin, zout, T);
    } else {                                          /* TILES */
        if (p->tw <= 0 || p->ntile < 2) return 0;
        if (bwd && p->scr) {
            const int nt = p->K - p->tcut;            /* the tiled transposed records come first */
            _ilfd_mt_tiles(p, list, nt, (size_t)p->ntile, zin, zout, T);
            for (i = nt; i < p->K; i++) _ilfd_mt_stage(p, recs, i, zin, zout, T);
        } else {
            for (i = 0; i < p->tlo; i++) _ilfd_mt_stage(p, recs, i, zin, zout, T);
            _ilfd_mt_tiles(p, list + p->tlo, p->thi - p->tlo, (size_t)p->ntile, zin, zout, T);
            for (i = p->thi; i < p->K; i++) _ilfd_mt_stage(p, recs, i, zin, zout, T);
        }
    }
    _vfft_ilfd_mt_count++;
    return 1;
}

/* ── the race at T: serial vs blocks vs tiles at every legal width ──── */
typedef struct { vfft_ilfd_plan_t *p; const double *zin; double *zout; int mt, tw, ok; char name[24]; } _ilfd_mt_ctx_t;
static void _ilfd_mt_arm_run(void *v)
{
    _ilfd_mt_ctx_t *c = (_ilfd_mt_ctx_t *)v;
    vfft_ilfd_plan_t *p = c->p;
    if (p->tw != c->tw) vfft_ilfd_apply_tw(p, c->tw);   /* the arm's width (rebinds the lists) */
    p->mt = c->mt;
    if (c->mt == 0) { vfft_ilfd_execute_fwd(p, c->zin, c->zout); return; }
    if (c->ok && !vfft_ilfd_execute_mt(p, c->zin, c->zout, 0)) c->ok = 0;
}
/* Leaves the plan at the WINNING (mt, tw) and bound for T. tw0 = the
 * one-thread width to restore for the serial verdict. Returns mt. */
static inline int vfft_ilfd_mt_race(vfft_ilfd_plan_t *p, int T, int tw0,
                                    const double *zin, double *zout, int *mt_tw)
{
    _ilfd_mt_ctx_t cx[VFFT_ILFD_MAX_K + 4];
    vfft_race_arm_t arms[VFFT_ILFD_MAX_K + 4];
    double ns[VFFT_ILFD_MAX_K + 4];
    int cand[VFFT_ILFD_MAX_K + 1], ncand, na = 0, a, best = 0, reps, i;
    *mt_tw = 0;
    p->mt_t = T;
    if (!vfft_ilfd_mt_bind(p, T)) { p->mt = 0; return 0; }
    {   /* reps from one serial timing: ~20 ms of serial-equivalent work per sample */
        double t0;
        vfft_ilfd_apply_tw(p, tw0);
        vfft_ilfd_execute_fwd(p, zin, zout);
        t0 = _il_ab_now(); vfft_ilfd_execute_fwd(p, zin, zout); t0 = _il_ab_now() - t0;
        reps = (int)(20e6 / (t0 > 1.0 ? t0 : 1.0));
        if (reps < 2) reps = 2;
        if (reps > 256) reps = 256;
    }
#define ILFD_ARM(MT, TW, NAME) do { \
        cx[na].p = p; cx[na].zin = zin; cx[na].zout = zout; cx[na].mt = (MT); cx[na].tw = (TW); cx[na].ok = 1; \
        snprintf(cx[na].name, sizeof cx[na].name, "%s", NAME); \
        arms[na].name = cx[na].name; arms[na].run = _ilfd_mt_arm_run; arms[na].ctx = &cx[na]; na++; \
    } while (0)
    ILFD_ARM(0, tw0, "serial");
    ILFD_ARM(1, tw0, "blocks");
    ncand = vfft_ilfd_tw_candidates(p, 0, cand, VFFT_ILFD_MAX_K + 1);
    for (i = 0; i < ncand && na < (int)(sizeof cx / sizeof cx[0]); i++) {
        if (cand[i] <= 0 || p->N / cand[i] < 2) continue;
        ILFD_ARM(2, cand[i], "tiles");
        snprintf(cx[na - 1].name, sizeof cx[na - 1].name, "tiles/tw%d", cand[i]);
    }
#undef ILFD_ARM
    {
        const vfft_race_proto_t proto = { 3, reps, VFFT_RACE_MIN, 1, 2, NULL, NULL };
        vfft_race_run(&proto, arms, na, ns);
    }
    for (a = 1; a < na; a++)
        if (cx[a].ok && ns[a] < ns[best]) best = a;
    if (getenv("VFFT_NAT_LOG") || getenv("VFFT_IL2D_LOG")) {
        fprintf(stderr, "[k1fd-mt] N=%d T=%d reps=%d", p->N, T, reps);
        for (a = 0; a < na; a++) fprintf(stderr, " %s=%.0f%s", cx[a].name, ns[a], cx[a].ok ? "" : "(no engage)");
        fprintf(stderr, " -> %s\n", cx[best].name);
    }
    p->mt = cx[best].mt;
    *mt_tw = (p->mt == 2) ? cx[best].tw : 0;
    vfft_ilfd_apply_tw(p, p->mt == 2 ? cx[best].tw : tw0);
    return p->mt;
}

#endif /* VFFT_IL_FLATDIT_MT_H */
