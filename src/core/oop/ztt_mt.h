/* ztt_mt.h — ZTURN-T's THREADED arm: the staged walk, sectioned
 * (docs/design/ztt_mt_design.md, 2026-09-15; the method is the deleted
 * cascade's, docs/design/cascade_mt_method.md).
 *
 * Every stage of the staged walk (ztt.h _ztt_staged_run) is a loop over
 * units — columns (the ingest, the terminator, the plain stage 0), groups
 * (a mid), tiles (the natural tiled prefix) or blocks (the plain suffix) —
 * whose iterations touch disjoint spans and share only read-only tables.
 * A thread takes a RANGE of units: the same kernel call with shifted
 * pointers and a shorter count. Nothing is recomputed and no order inside a
 * unit changes, so the threaded output is BITWISE the serial walk's, which
 * is what benches/ztt_mt_gate.c holds. One fork-join per stage
 * (stride_pool_run: the caller is slot 0, the workers spin).
 *
 * Two arms, the flat DIT's (il_flatdit_mt.h): mt=1 BLOCKS — every stage cut
 * by units, the tile ignored (tiling changes group order only); mt=2 TILES —
 * the tiled prefix (natural) / the per-block suffix (plain) cut by tiles or
 * blocks, each tile's whole prefix on one thread, the sweeps cut by units.
 * The arm is a RACED plan parameter, banked per T (k1_commit.h
 * _ztt_mt_replay_or_race); this header is the execution only.
 *
 * At T > 1 a pow2 cell runs this walk, not its fused codelet: the fusion is
 * worth 0-3% above 2048, the threads are worth multiples.
 *
 * Column ranges are cut at multiples of 4 (every kind's `count % 4 == 0`);
 * an empty range does nothing. The plane offset is computed once by the
 * caller and shared: the threads write disjoint spans of the plan's one
 * plane. Nesting is forbidden by the pool's law: this runs on the caller
 * thread of a plan whose own nthreads > 1, never from a worker.
 *
 * Included by vfft.c ONLY (it pulls support/threads.h, whose pool state is
 * per translation unit — a bench that included it would own a second pool). */
#ifndef VFFT_OOP_ZTT_MT_H
#define VFFT_OOP_ZTT_MT_H

#include "ztt.h"
#include "support/threads.h"
#include "support/race.h"      /* the race body and _il_ab_now, for vfft_ztt_mt_race */

extern long _vfft_ztt_mt_count;   /* vfft.c: the engagement counter */

/* one thread's work item */
typedef struct
{
    int kind;                         /* 0 = one kernel call, 1 = natural tile range,
                                       * 2 = plain block range fwd, 3 = plain block range bwd */
    vfft_ztt_kfn fn;
    const double *zin, *tw, *tw_im;
    double *zout;
    size_t Ls, Gs, OLs, count;
    const vfft_ztt_plan_t *p;         /* kinds 1-3 */
    double *W;
    const double *tws;                /* the stream base for the direction */
    const double *zin_b;              /* plain bwd: the block source (zin) */
    int bwd;
    size_t t_lo, t_hi;
} _ztt_mt_arg;

/* the natural tile's prefix / the plain block's suffix, as _ztt_staged_run runs them */
static inline void _ztt_mt_tile_natural(const vfft_ztt_plan_t *p, const vfft_ztt_kfn *st,
                                        const double *tw, double *W, size_t t)
{
    const size_t tile = p->tile;
    double *B = W + t * tile * 2;
    int s;
    for (s = 1; s < p->nf - 1; s++)
    {
        const size_t RL = (size_t)p->L[s] * (size_t)p->chain[s];
        if (tile % RL == 0)
            st[s](B, 0, B, 0, tw + p->twoff[s], 0, (size_t)p->L[s], tile / RL, 0, 0, (size_t)p->L[s]);
    }
}
static inline void _ztt_mt_block_plain(const vfft_ztt_plan_t *p, const vfft_ztt_kfn *st,
                                       const double *tw, const double *zin, double *zout,
                                       size_t t, int bwd)
{
    const size_t tile = p->tile, Rl = (size_t)p->chain[p->nf - 1];
    double *B = zout + t * tile * 2;
    int s;
    if (!bwd)
    {
        for (s = 1; s < p->nf - 1; s++)
            if (tile % (size_t)p->len[s] == 0)
                st[s](B, 0, B, 0, tw + p->twoff[s], 0,
                      (size_t)p->len[s + 1], tile / (size_t)p->len[s], 0, 0, (size_t)p->len[s + 1]);
        st[p->nf - 1](B, 0, B, 0, tw, 0, 0, 1, 0, 0, tile / Rl);
    }
    else
    {
        const double *Bi = zin + t * tile * 2;
        st[p->nf - 1](Bi, 0, B, 0, tw, 0, 0, 1, 0, 0, tile / Rl);
        for (s = p->nf - 2; s >= 1; s--)
            if (tile % (size_t)p->len[s] == 0)
                st[s](B, 0, B, 0, tw + p->twoff[s], 0,
                      (size_t)p->len[s + 1], tile / (size_t)p->len[s], 0, 0, (size_t)p->len[s + 1]);
    }
}

static void _ztt_mt_tramp(void *v)
{
    const _ztt_mt_arg *a = (const _ztt_mt_arg *)v;
    size_t t;
    switch (a->kind)
    {
    case 0:
        if (a->count) a->fn(a->zin, 0, a->zout, 0, a->tw, a->tw_im, a->Ls, a->Gs, a->OLs, 0, a->count);
        break;
    case 1:
        for (t = a->t_lo; t < a->t_hi; t++)
            _ztt_mt_tile_natural(a->p, a->bwd ? a->p->st_bwd : a->p->st_fwd, a->tws, a->W, t);
        break;
    default:
        for (t = a->t_lo; t < a->t_hi; t++)
            _ztt_mt_block_plain(a->p, a->bwd ? a->p->st_bwd : a->p->st_fwd, a->tws, a->zin_b, a->zout, t, a->bwd);
        break;
    }
}

/* ── the three dispatch shapes ──────────────────────────────────────────── */

/* a COLUMN stage: count columns from base pointers, both edges advancing one
 * complex per column, the stream one record group per column quad */
static inline void _ztt_mt_columns(int T, vfft_ztt_kfn fn, const double *zin, double *zout, int oshift,
                                   const double *tw, const double *tw_im, int twrec,
                                   size_t Ls, size_t OLs, size_t count, const size_t *rb)
{
    _ztt_mt_arg a[STRIDE_POOL_MAX_DISPATCH];
    const size_t nq = count / 4;
    int w;
    for (w = 0; w < T; w++)
    {
        const size_t lo = 4 * (nq * (size_t)w / (size_t)T), hi = 4 * (nq * (size_t)(w + 1) / (size_t)T);
        memset(&a[w], 0, sizeof a[w]);
        a[w].kind = 0; a[w].fn = fn;
        a[w].zin = zin + 2 * lo; a[w].zout = oshift ? zout + 2 * lo : zout;   /* the ingest stores at rb[c]*R0, absolute */
        a[w].tw = tw ? tw + (lo / 4) * (size_t)twrec : 0;
        a[w].tw_im = rb ? (const double *)(rb + lo) : tw_im;
        a[w].Ls = Ls; a[w].Gs = 1; a[w].OLs = OLs; a[w].count = hi - lo;
    }
    stride_pool_run(T, _ztt_mt_tramp, a, sizeof a[0]);
}

/* a GROUP stage: Gs groups at pitch doubles apart, one group-invariant stream */
static inline void _ztt_mt_groups(int T, vfft_ztt_kfn fn, double *base, size_t pitch,
                                  const double *tw, size_t Ls, size_t Gs, size_t count)
{
    _ztt_mt_arg a[STRIDE_POOL_MAX_DISPATCH];
    int w;
    for (w = 0; w < T; w++)
    {
        const size_t lo = Gs * (size_t)w / (size_t)T, hi = Gs * (size_t)(w + 1) / (size_t)T;
        memset(&a[w], 0, sizeof a[w]);
        a[w].kind = 0; a[w].fn = fn;
        a[w].zin = base + lo * pitch; a[w].zout = base + lo * pitch;
        a[w].tw = tw; a[w].Ls = Ls; a[w].Gs = hi - lo; a[w].count = count;
        if (hi == lo) a[w].count = 0;
    }
    stride_pool_run(T, _ztt_mt_tramp, a, sizeof a[0]);
}

/* the plain untiled last stage: N/R groups of R, 4 per kernel iteration */
static inline void _ztt_mt_tld(int T, vfft_ztt_kfn fn, const double *zin, double *zout,
                               const double *tw, size_t R, size_t ngroups)
{
    _ztt_mt_arg a[STRIDE_POOL_MAX_DISPATCH];
    const size_t nq = ngroups / 4;
    int w;
    for (w = 0; w < T; w++)
    {
        const size_t lo = 4 * (nq * (size_t)w / (size_t)T), hi = 4 * (nq * (size_t)(w + 1) / (size_t)T);
        memset(&a[w], 0, sizeof a[w]);
        a[w].kind = 0; a[w].fn = fn;
        a[w].zin = zin + 2 * R * lo; a[w].zout = zout + 2 * R * lo;
        a[w].tw = tw; a[w].Ls = 0; a[w].Gs = 1; a[w].count = hi - lo;
    }
    stride_pool_run(T, _ztt_mt_tramp, a, sizeof a[0]);
}

/* the TILE / BLOCK range dispatch */
static inline void _ztt_mt_tiles(int T, const vfft_ztt_plan_t *p, int kind, const double *tws,
                                 double *W, const double *zin_b, double *zout, int bwd, size_t ntile)
{
    _ztt_mt_arg a[STRIDE_POOL_MAX_DISPATCH];
    int w;
    for (w = 0; w < T; w++)
    {
        memset(&a[w], 0, sizeof a[w]);
        a[w].kind = kind; a[w].p = p; a[w].tws = tws; a[w].W = W; a[w].zin_b = zin_b;
        a[w].zout = zout; a[w].bwd = bwd;
        a[w].t_lo = ntile * (size_t)w / (size_t)T; a[w].t_hi = ntile * (size_t)(w + 1) / (size_t)T;
    }
    stride_pool_run(T, _ztt_mt_tramp, a, sizeof a[0]);
}

/* bind the arm for the plan's T; 0 = declined (T < 2 or an arm the plan
 * cannot run: TILES needs a tile and at least two of them) */
static inline int vfft_ztt_mt_bind(vfft_ztt_plan_t *p, int T, int arm)
{
    if (T < 2 || arm < 1 || arm > 2) { p->mt = 0; p->mt_t = 0; return 0; }
    if (arm == 2 && (p->tile == 0 || (size_t)p->N / p->tile < 2)) { p->mt = 0; p->mt_t = 0; return 0; }
    p->mt = arm;
    p->mt_t = T;
    return 1;
}

/* Returns 1 when it ran threaded, 0 when the caller must run serial: no arm
 * bound, T < 2, or a live pool clamped below the bound T (the cut is for
 * exactly mt_t threads: fewer would not cover the units the same way, and
 * the verdict was raced at that T). */
static inline int vfft_ztt_execute_mt(const vfft_ztt_plan_t *p, const double *zin, double *zout, int bwd)
{
    const int T = stride_pool_workers_for(p->mt_t);
    const int nf = p->nf;
    const vfft_ztt_kfn *st = bwd ? p->st_bwd : p->st_fwd;
    const double *tw = bwd ? p->twb : p->tw;
    const size_t N = (size_t)p->N, tile = p->tile;
    int s;
    if (p->mt <= 0 || T < 2 || T != p->mt_t || !st[0]) return 0;
    if (p->mt == 2 && (tile == 0 || N / tile < 2)) return 0;
    if (p->scr)
    {
        const size_t Rl = (size_t)p->chain[nf - 1], len1 = (size_t)p->len[1];
        const int R0 = p->chain[0];
        if (!bwd)
        {
            _ztt_mt_columns(T, st[0], zin, zout, 1, tw, 0, (R0 - 1) * 8, len1, 0, len1, 0);
            for (s = 1; s < nf - 1; s++)
                if (p->mt == 1 || tile % (size_t)p->len[s])
                    _ztt_mt_groups(T, st[s], zout, 2 * (size_t)p->len[s], tw + p->twoff[s],
                                   (size_t)p->len[s + 1], N / (size_t)p->len[s], (size_t)p->len[s + 1]);
            if (p->mt == 2)
                _ztt_mt_tiles(T, p, 2, tw, 0, zin, zout, 0, N / tile);
            else
                _ztt_mt_tld(T, st[nf - 1], zout, zout, tw, Rl, N / Rl);
        }
        else
        {
            if (p->mt == 2)
                _ztt_mt_tiles(T, p, 3, tw, 0, zin, zout, 1, N / tile);
            else
                _ztt_mt_tld(T, st[nf - 1], zin, zout, tw, Rl, N / Rl);
            for (s = nf - 2; s >= 1; s--)
                if (p->mt == 1 || tile % (size_t)p->len[s])
                    _ztt_mt_groups(T, st[s], zout, 2 * (size_t)p->len[s], tw + p->twoff[s],
                                   (size_t)p->len[s + 1], N / (size_t)p->len[s], (size_t)p->len[s + 1]);
            _ztt_mt_columns(T, st[0], zout, zout, 1, tw, 0, (R0 - 1) * 8, len1, len1, len1, 0);
        }
    }
    else
    {
        const int ip = (zin == zout || p->inplace);
        double *W = ip ? _ztt_plane_for(p, zout) : zout;
        const vfft_ztt_kfn last = ip ? (bwd ? p->tl_bwd_plane : p->tl_fwd_plane) : st[nf - 1];
        const size_t L = (size_t)p->L[nf - 1];
        const int Rl = p->chain[nf - 1];
        _ztt_mt_columns(T, st[0], zin, W, 0, 0, 0, 0, (size_t)p->ncol, 0, (size_t)p->ncol, p->rb);
        if (p->mt == 2)
            _ztt_mt_tiles(T, p, 1, tw, W, 0, 0, bwd, N / tile);
        for (s = 1; s < nf - 1; s++)
        {
            const size_t RL = (size_t)p->L[s] * (size_t)p->chain[s];
            if (p->mt == 1 || tile % RL)
                _ztt_mt_groups(T, st[s], W, 2 * RL, tw + p->twoff[s], (size_t)p->L[s], (size_t)p->Gs[s], (size_t)p->L[s]);
        }
        _ztt_mt_columns(T, last, W, zout, 1, tw + p->twoff[nf - 1], 0, (Rl - 1) * 8, L, L, L, 0);
    }
    _vfft_ztt_mt_count++;
    return 1;
}

/* ── the race at T: serial vs BLOCKS vs TILES on the plan as bound ──────
 * (the flat DIT's protocol: reps to ~20 ms of serial-equivalent work per
 * sample, min-of-3, 2 warm passes, unpaced — the threaded arm is measured
 * hot). An in-place plan races ALIASED arms (z -> z through the plane) with
 * the buffer re-seeded before every sample; an out-of-place plan races
 * zin -> zout. Leaves the plan at the winner, bound for T. Returns mt. */
typedef struct { vfft_ztt_plan_t *p; const double *zin; double *zout; int mt, ok; } _ztt_mt_ctx_t;
typedef struct { double *dst; const double *src; size_t nb; } _ztt_mt_rst_t;
static void _ztt_mt_arm_run(void *v)
{
    _ztt_mt_ctx_t *c = (_ztt_mt_ctx_t *)v;
    if (c->mt == 0) { c->p->mt = 0; vfft_ztt_execute_fwd(c->p, c->zin, c->zout); return; }
    c->p->mt = c->mt;
    if (c->ok && !vfft_ztt_execute_mt(c->p, c->zin, c->zout, 0)) c->ok = 0;
}
static void _ztt_mt_reseed(void *v) { _ztt_mt_rst_t *r = (_ztt_mt_rst_t *)v; memcpy(r->dst, r->src, r->nb); }
static inline int vfft_ztt_mt_race(vfft_ztt_plan_t *p, int T, const double *zin, double *zout, double *ns_out)
{
    static const char *names[3] = { "serial", "blocks", "tiles" };
    _ztt_mt_ctx_t cx[3];
    vfft_race_arm_t arms[3];
    double ns[3];
    _ztt_mt_rst_t rs;
    const int ip = (zin == zout);
    int na = 0, a, best = 0, reps;
    if (T < 2) { p->mt = 0; p->mt_t = 0; return 0; }
    p->mt_t = T;
    {   /* reps from one serial timing */
        double t0;
        p->mt = 0;
        vfft_ztt_execute_fwd(p, zin, zout);
        t0 = _il_ab_now(); vfft_ztt_execute_fwd(p, zin, zout); t0 = _il_ab_now() - t0;
        reps = (int)(20e6 / (t0 > 1.0 ? t0 : 1.0));
        if (reps < 2) reps = 2;
        if (reps > (1 << 19)) reps = 1 << 19;
    }
    for (a = 0; a < 3; a++)
    {
        if (a == 2 && (p->tile == 0 || (size_t)p->N / p->tile < 2)) break;
        cx[na].p = p; cx[na].zin = zin; cx[na].zout = zout; cx[na].mt = a; cx[na].ok = 1;
        arms[na].name = names[a]; arms[na].run = _ztt_mt_arm_run; arms[na].ctx = &cx[na];
        na++;
    }
    rs.dst = zout; rs.src = ip ? p->plane : zin; rs.nb = 0;   /* placeholder, set below */
    if (ip)
    {   /* aliased arms: re-seed the buffer from a copy before every sample */
        double *seed = (double *)VFFT_ZTT_ALLOC((size_t)2 * p->N * sizeof(double));
        if (!seed) { p->mt = 0; return 0; }
        memcpy(seed, zout, (size_t)2 * p->N * sizeof(double));
        rs.src = seed; rs.nb = (size_t)2 * p->N * sizeof(double);
        {
            const vfft_race_proto_t proto = { 3, reps, VFFT_RACE_MIN, 1, 2, _ztt_mt_reseed, &rs };
            vfft_race_run(&proto, arms, na, ns);
        }
        VFFT_ZTT_FREE(seed);
    }
    else
    {
        const vfft_race_proto_t proto = { 3, reps, VFFT_RACE_MIN, 1, 2, NULL, NULL };
        vfft_race_run(&proto, arms, na, ns);
    }
    for (a = 1; a < na; a++)
        if (cx[a].ok && ns[a] < ns[best]) best = a;
    if (getenv("VFFT_NAT_LOG") || getenv("VFFT_IL2D_LOG"))
    {
        fprintf(stderr, "[ztt-mt] N=%d T=%d %s%s reps=%d", p->N, T, p->scr ? "scr" : "nat", ip ? " ip" : "", reps);
        for (a = 0; a < na; a++) fprintf(stderr, " %s=%.0f%s", names[cx[a].mt], ns[a], cx[a].ok ? "" : "(no engage)");
        fprintf(stderr, " -> %s\n", names[cx[best].mt]);
    }
    if (ns_out) { for (a = 0; a < 3; a++) ns_out[a] = a < na ? ns[a] : -1.0; }
    p->mt = cx[best].mt;
    if (p->mt && !vfft_ztt_mt_bind(p, T, p->mt)) p->mt = 0;
    return p->mt;
}

#endif /* VFFT_OOP_ZTT_MT_H */
