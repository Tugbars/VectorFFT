/* c2c_create_split.h — the split library's 1D C2C create: zturn-ts, both
 * placements (docs/design/split_v1_0_contract.md).
 *
 * WHAT THE SPLIT SIDE SERVES: a power-of-two N the zturn-ts registry holds,
 * natural order, in place or out of place; K = 1; a CONTIGUOUS batch of any
 * K (the K = 1 plan run K times, in slabs over the pool at T > 1); a
 * LANE-MAJOR batch of K a multiple of the vector width. Everything else is
 * refused here, loudly, naming the interleaved layout: an N outside the
 * registry, a scrambled order, a caller-owned (padded) batch, a lane-major K
 * off the vector width. The split layout is for highly batched low-N work;
 * where the interleaved layout starts winning depends on the platform, the
 * task and the thread count, so nothing here draws a ceiling (owner, 2026-11-05).
 *
 * ONE ENGINE, ONE ROW. The cell is the split natural row
 *   @cell t=c2c n=N q=K ord=nat place=ip|oop lay=split | eng=zts chain= [tile=]
 * (split/wisdom/wisdom2_zts_split.h). A hit replays. A miss, a row naming an
 * engine this library no longer has, or VFFT_FLAG_RECALIBRATE races the
 * registry's chains and tiles at N (zts_race.h) with NO incumbent — nothing
 * else serves a split cell — and banks the winner. A contiguous batch's
 * threading verdict is its own row (eng=tcb tcmt= at nthreads=T,
 * common/wisdom/wisdom2_tcmt.h), raced at create at the plan's thread count;
 * so is the lane-major batch's (lane blocks over the slots), on the lane-major
 * cell (lay=slm, its own cell beside the contiguous rows at the same q).
 *
 * Depends on common/ and split/ only. POSITION IN vfft.c IS LOAD-BEARING:
 * calls file-scope statics of vfft.c (_vfft_plan_threads, _vw2_persist,
 * _vfft_warn, _vfft_tname) and _vfft_split_execute (forward-declared there).
 */
#ifndef VFFT_SPLIT_C2C_CREATE_SPLIT_H
#define VFFT_SPLIT_C2C_CREATE_SPLIT_H

#include "split/rank1/zts.h"
#include "split/rank1/zts_race.h"
#include "split/wisdom/wisdom2_zts_split.h"
#include "common/wisdom/wisdom2_tcmt.h"
#include "split/rank1/zts_bluestein.h"
#ifdef VFFT_USE_JIT
#include "jit/zts_jit_runtime.h"       /* a cell without a fused driver gets one compiled at create */
#endif

/* a handle served by zturn-ts: the plan, the geometry, nothing else */
static struct vfft_plan_s *_zts_handle(const vfft_cfg_t *cfg, int placement, int N, size_t K,
                                       vfft_zts_plan_t *z)
{
    struct vfft_plan_s *h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_zts_destroy(z);
        return NULL;
    }
    h->transform = VFFT_C2C;
    h->placement = placement;
    h->layout = (int)cfg->layout;
    h->N = N;
    h->K = K;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zts = z;
    return h;
}

/* the cell's banked zturn-ts verdict rebuilt. NULL = no usable row: the
 * cell is unbanked, or its row names an engine this library no longer has
 * (a stride, classic or kind-3 verdict of the retired engines — *stale
 * says so, and the caller races and banks over it). */
static vfft_zts_plan_t *_zts_replay(struct vfft_wisdom_s *W, int N, size_t K, int place_ip,
                                    int lanes, int *stale)
{
    vfft_zts_entry_t ze;
    vfft_zts_plan_t *z;
    *stale = 0;
    if (!W || !vw2_zts_lookup(&W->vw2, N, K, place_ip, &ze))
        return NULL;
    if (!ze.zts)
    {
        *stale = 1;
        return NULL;
    }
    z = vfft_zts_create_chain(N, ze.chain, ze.nf, lanes);
    if (z && ze.tile && !vfft_zts_set_tile(z, ze.tile))
    {
        vfft_zts_destroy(z);
        z = NULL;
    }
    if (z && ze.blk > 1 && !vfft_zts_set_blk(z, ze.blk))
    {
        vfft_zts_destroy(z);
        z = NULL;
    }
    if (z && getenv("VFFT_NAT_LOG"))
    {
        char cs[48];
        vfft_zts_chain_str(z, cs, sizeof cs);
        fprintf(stderr, "[zts] N=%d q=%zu %s: replay chain %s tile=%zu blk=%d src=wisdom\n",
                N, K, place_ip ? "ip" : "oop", cs, z->tile, z->blk);
    }
    return z;
}

/* the cell planned: replayed, else raced (the registry's pool at N in this
 * lane mode, no incumbent) and banked. NULL = nothing serves the cell. */
/* the DEPLOYED plan (never a race candidate): under VFFT_USE_JIT a cell the
 * registry holds no fused driver for (the odd band, the lane-major band) gets
 * one compiled now — the driver zts_drivers.ml emits for a registry cell,
 * written for this chain, the kernels' sources inlined, cached on disk
 * (jit/zts_jit_runtime.h). The staged walk stays on any failure. */
static vfft_zts_plan_t *_zts_deployed(vfft_zts_plan_t *z)
{
#ifdef VFFT_USE_JIT
    if (z && z->staged)
    {
        const vfft_zts_cell_t *c = vfft_zts_jit_resolve(z->N, z->chain, z->nf, z->lanes);
        if (c)
        {
            z->cell = c;
            z->staged = 0;
            z->jit = 1;
        }
    }
#endif
    return z;
}
static vfft_zts_plan_t *_zts_plan_cell(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
                                       int N, size_t K, int place_ip, int lanes)
{
    int stale = 0;
    vfft_zts_plan_t *z = cfg->recalibrate ? NULL : _zts_replay(W, N, K, place_ip, lanes, &stale);
    double wns = 0.0, ins = 0.0;
    if (z)
        return _zts_deployed(z);
    if (stale && getenv("VFFT_NAT_LOG"))
        fprintf(stderr, "[zts] N=%d q=%zu %s: the banked row names a retired engine; racing zturn-ts\n",
                N, K, place_ip ? "ip" : "oop");
    z = vfft_zts_race(N, lanes, place_ip, NULL, NULL, 0, &wns, &ins);
    if (z && lanes > 1)
    {   /* the lane-major cell's block: raced on the winner (zts_race.h) */
        double bns = 0.0;
        if (vfft_zts_race_blk(z, place_ip, &bns) > 1 && bns > 0.0)
            wns = bns;
    }
    if (z && W)
    {
        vw2_zts_bank(&W->vw2, N, K, place_ip, z->chain, z->nf, z->tile, z->blk, wns);
        _vw2_persist(W, cfg);
    }
    return _zts_deployed(z);
}

/* ── zturn-ts's CONTIGUOUS batch at T > 1: the THREADING verdict ──────
 * The one arm: the serial loop vs SLABS over the pool's slots (zts.h
 * vfft_zts_run_slab; split_execute.h), raced at create on the batch's own
 * cell and banked as eng=tcb tcmt= on its q=K row at the plan's thread
 * count (common/wisdom/wisdom2_tcmt.h, the contiguous wrapper's grammar on
 * both layouts; a slab is ceil(K/T) transforms, so the verdict moves with T
 * and a lookup never crosses it). No worker planes (K = 1 workers,
 * VFFT_NO_TCMT) => no arm: serial. VFFT_TCMT=0|1 pins the verdict (never
 * replays, never banks). */
typedef struct
{
    struct vfft_plan_s *h;
    double *re, *im, *ore, *oim;
    int mt;
} _zts_tcmt_arm_t;
static void _zts_tcmt_arm_run(void *v)
{
    _zts_tcmt_arm_t *a = (_zts_tcmt_arm_t *)v;
    a->h->zts->mt = a->mt;
    _vfft_split_execute(a->h, VFFT_FORWARD, a->re, a->im, a->ore, a->oim);
}
typedef struct
{
    double *re, *im, *re0, *im0;
    size_t nb;
} _zts_tcmt_reseed_t;
static void _zts_tcmt_reseed(void *v)
{
    _zts_tcmt_reseed_t *r = (_zts_tcmt_reseed_t *)v;
    memcpy(r->re, r->re0, r->nb);
    memcpy(r->im, r->im0, r->nb);
}
/* the arms a cell has at T: 0 serial; 1 slabs (contiguous K >= T) or lane
 * blocks (lane-major, >= 2 blocks); the sectioned walk's 1 BLOCKS and 2
 * TILES (K = 1, or K < T contiguous, or the lane-major single group). The
 * verdict is one number on the cell's threading row: 0, 1 or 2, with mt_sect
 * saying which form a 1 means (the cell's shape decides: a cell with >= 2
 * slab units never sections; one with fewer always does). */
/* the row's value names the FORM with the arm, so a row banked under one
 * form is never read under the other (a contiguous K cell is slabs at K >= T
 * and the sectioned hybrid at K < T): 0 serial; 1 slabs / lane blocks;
 * 3 the sectioned BLOCKS; 4 the sectioned TILES. */
#define VFFT_ZTS_TCMT_SECT_BLOCKS 3
#define VFFT_ZTS_TCMT_SECT_TILES 4
static int _zts_tcmt_value(int mt, int sect)
{
    return mt == 0 ? 0 : sect ? (mt == 2 ? VFFT_ZTS_TCMT_SECT_TILES : VFFT_ZTS_TCMT_SECT_BLOCKS) : 1;
}
/* the row's value read for this form: 1 = fits (mt set), 0 = the other form's row (re-race) */
static int _zts_tcmt_fits(int v, int sect, int *mt)
{
    if (v == 0) { *mt = 0; return 1; }
    if (sect && v == VFFT_ZTS_TCMT_SECT_BLOCKS) { *mt = 1; return 1; }
    if (sect && v == VFFT_ZTS_TCMT_SECT_TILES) { *mt = 2; return 1; }
    if (!sect && v == 1) { *mt = 1; return 1; }
    return 0;
}
static void _zts_tcmt_set(struct vfft_plan_s *h, int v, int sect)
{
    h->zts->mt = v;
    h->zts->mt_sect = sect;
}
static void _zts_tcmt_decide(struct vfft_plan_s *h, const vfft_cfg_t *cfg,
                             struct vfft_wisdom_s *W, int N, size_t K)
{
    const int T = h->nthreads;
    const int ip = (h->placement == VFFT_INPLACE);
    const int pl = ip ? VW2_PL_IP : VW2_PL_OOP;
    const int lm = h->zts->lanes > 1;                       /* the lane-major batch: lane blocks over the slots */
    const uint8_t lay = lm ? VW2_LAY_SLM : VW2_LAY_SPLIT;   /* its own cell beside the contiguous batch's rows */
    const int lg = getenv("VFFT_TCMT_VERBOSE") || getenv("VFFT_TCMT_LOG");
    const char *pin = getenv("VFFT_TCMT");
    size_t units;                                           /* the work units a slab holds whole: transforms, or lane blocks */
    int nw, sect, can_sect;
    h->zts->mt = 0;
    h->zts->mt_sect = 0;
    if (T <= 1 || getenv("VFFT_NO_TCMT"))
        return;
    units = lm ? ((size_t)h->zts->lanes + (size_t)h->zts->blk * VFFT_ZTS_VW - 1) / ((size_t)h->zts->blk * VFFT_ZTS_VW) : K;
    /* the sectioned walk serves where the slab form cannot fill the pool:
     * K = 1, a contiguous K < T, the lane-major single group */
    can_sect = lm ? (h->zts->lanes == VFFT_ZTS_VW && h->zts->v0f && h->zts->vlf)
                  : (h->zts->s0f && h->zts->slf && (K == 1 || (size_t)T / K >= 2));
    sect = can_sect && (units < 2 || (!lm && (size_t)T / K >= 2 && K < (size_t)T));
    if (!sect && units <= 1)
        return; /* one unit and no sectioned form: nothing to slice */
    nw = T - 1;
    if (!sect && (size_t)nw > units - 1)
        nw = (int)(units - 1);
    if (sect && !lm && K > 1 && nw > (int)K - 1)
        nw = (int)K - 1;   /* the sectioned contiguous batch: a plane per transform */
    if (nw > VFFT_ZTS_MT_MAX)
        nw = VFFT_ZTS_MT_MAX;
    if (nw > 0 && vfft_zts_mt_planes(h->zts, nw) == 0)
        return; /* no worker planes: no arm */
    if (pin)
    {   /* VFFT_TCMT = the arm for this form: 0, 1 (slabs / blocks), 2 (tiles) */
        _zts_tcmt_set(h, atoi(pin), sect);
        if (lg)
            fprintf(stderr, "[tcmt] zts N=%d K=%zu T=%d: pinned mt=%d%s (env; not banked)\n", N, K, T, h->zts->mt, sect ? " (sectioned)" : "");
        return;
    }
    if (W && !cfg->recalibrate)
    {
        int v = 0, mt = 0;
        if (vw2_tcmt_lookup(&W->vw2, VW2_T_C2C, N, K, VW2_ORD_NAT, pl, lay, T, &v))
        {
            if (_zts_tcmt_fits(v, sect, &mt))
            {
                _zts_tcmt_set(h, mt, sect);
                if (lg)
                    fprintf(stderr, "[tcmt] zts N=%d K=%zu T=%d: replay tcmt=%d -> mt=%d%s src=wisdom\n", N, K, T, v, mt, sect ? " (sectioned)" : "");
                return;
            }
            if (lg)
                fprintf(stderr, "[tcmt] zts N=%d K=%zu T=%d: the row's tcmt=%d is the other form's; racing\n", N, K, T, v);
        }
    }
    { /* the race: serial loop vs slabs, on this cell's own planes */
        const size_t n = (size_t)N * K, nb = n * sizeof(double);
        double *re = (double *)vfft_aligned_alloc(nb), *im = (double *)vfft_aligned_alloc(nb);
        double *ore = ip ? re : (double *)vfft_aligned_alloc(nb);
        double *oim = ip ? im : (double *)vfft_aligned_alloc(nb);
        double *re0 = ip ? (double *)vfft_aligned_alloc(nb) : NULL;
        double *im0 = ip ? (double *)vfft_aligned_alloc(nb) : NULL;
        double st = 0, mt = 0;
        size_t i;
        if (!re || !im || !ore || !oim || (ip && (!re0 || !im0)))
        {
            vfft_aligned_free(re);
            vfft_aligned_free(im);
            if (!ip) { vfft_aligned_free(ore); vfft_aligned_free(oim); }
            vfft_aligned_free(re0);
            vfft_aligned_free(im0);
            return; /* no buffers: serial */
        }
        for (i = 0; i < n; i++)
        {
            re[i] = 1.0 + 1e-6 * (double)(i & 511);
            im[i] = 0.5 - 1e-6 * (double)(i & 255);
        }
        if (ip)
        {
            memcpy(re0, re, nb);
            memcpy(im0, im, nb);
        }
        {
            _zts_tcmt_arm_t a = { h, re, im, ore, oim, 0 };
            _zts_tcmt_arm_t b = { h, re, im, ore, oim, 1 };
            _zts_tcmt_arm_t c = { h, re, im, ore, oim, 2 };
            _zts_tcmt_reseed_t rs = { re, im, re0, im0, nb };
            const int tiles = sect && h->zts->tile && (size_t)h->zts->N / h->zts->tile >= 2;
            const vfft_race_arm_t arms[3] = { { "serial", _zts_tcmt_arm_run, &a },
                                              { sect ? "blocks" : "slabs", _zts_tcmt_arm_run, &b },
                                              { "tiles", _zts_tcmt_arm_run, &c } };
            vfft_race_proto_t proto;
            double ns[3];
            int best = 0, i, reps;
            h->zts->mt_sect = sect;
            {   /* reps from one serial timing: ~4 ms of serial-equivalent work per
                 * sample (the interleaved ZTT-MT race's law), min of 3 rounds — one
                 * execute per sample at a small N flapped the verdict. The in-place
                 * arms repeat on the same buffer inside a sample (the values walk
                 * into inf; the timing holds, as on the interleaved side). */
                double t0;
                h->zts->mt = 0;
                _vfft_split_execute(h, VFFT_FORWARD, re, im, ore, oim);
                t0 = vfft_now_ns();
                _vfft_split_execute(h, VFFT_FORWARD, re, im, ore, oim);
                t0 = vfft_now_ns() - t0;
                reps = (int)(4e6 / (t0 > 1.0 ? t0 : 1.0));
                if (reps < 2) reps = 2;
                if (reps > 4096) reps = 4096;
                if (ip) { memcpy(re, re0, nb); memcpy(im, im0, nb); }
            }
            memset(&proto, 0, sizeof proto);
            proto.rounds = 3;
            proto.reps = reps;
            proto.agg = VFFT_RACE_MIN;
            proto.alternate = 1;
            proto.warm = 2;
            proto.reset = ip ? _zts_tcmt_reseed : NULL;
            proto.reset_ctx = ip ? &rs : NULL;
            vfft_race_run(&proto, arms, tiles ? 3 : 2, ns); /* threaded arms: no pace */
            for (i = 1; i < (tiles ? 3 : 2); i++)
                if (ns[i] < ns[best]) best = i;
            st = ns[0];
            mt = ns[best];
            _zts_tcmt_set(h, best, sect);
            if (lg)
                fprintf(stderr, "[tcmt] zts N=%d K=%zu T=%d %s: race serial=%.0f %s=%.0f%s -> %s\n",
                        N, K, T, ip ? "ip" : "oop", ns[0], sect ? "blocks" : "slabs", ns[1],
                        tiles ? "" : "", best == 0 ? "serial" : best == 1 ? (sect ? "BLOCKS" : "SLABS") : "TILES");
            if (lg && tiles)
                fprintf(stderr, "[tcmt]   tiles=%.0f\n", ns[2]);
        }
        vfft_aligned_free(re);
        vfft_aligned_free(im);
        if (!ip) { vfft_aligned_free(ore); vfft_aligned_free(oim); }
        vfft_aligned_free(re0);
        vfft_aligned_free(im0);
        if (W && vw2_tcmt_bank(&W->vw2, VW2_T_C2C, N, K, VW2_ORD_NAT, pl, lay,
                               _zts_tcmt_value(h->zts->mt, h->zts->mt_sect), T, h->zts->mt ? mt : st) == VW2_OK)
            _vw2_persist(W, cfg);
    }
}

/* ── Bluestein over zturn-ts (zts_bluestein.h): the M cell of the request's
 * form planned through the cell planner (its own replay / race / bank), plus
 * a K = 1 plan at M for the kernel spectra; the handle carries the Bluestein
 * plan, nothing else. The contiguous batch runs serially; the lane-major
 * batch is one pass with the chirp broadcast over the lanes. */
static vfft_plan _vfft_create_c2c_blu_split(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W,
                                            int N, size_t K, int lm)
{
    const int M = vfft_zts_blu_m(N);
    const int lanes = lm ? (int)K : 1;
    vfft_zts_plan_t *inner, *ks;
    vfft_zts_blu_t *b;
    struct vfft_plan_s *h;
    inner = _zts_plan_cell(W, cfg, M, lm ? K : 1, 1, lanes);        /* the M cell, in place, this form */
    if (!inner)
    {
        _vfft_warn("vfft_create: %s C2C N=%d K=%zu on the SPLIT layout (Bluestein): no zturn-ts cell at the "
                   "convolution length M=%d", cfg->placement == VFFT_INPLACE ? "in-place" : "out-of-place", N, K, M);
        return NULL;
    }
    ks = lanes == 1 ? inner : _zts_plan_cell(W, cfg, M, 1, 1, 1);   /* the kernels' K = 1 plan at M */
    b = ks ? vfft_zts_blu_create(N, lanes, inner, ks) : NULL;
    if (ks && ks != inner) vfft_zts_destroy(ks);
    if (!b)
    {
        vfft_zts_destroy(inner);
        _vfft_warn("vfft_create: %s C2C N=%d K=%zu on the SPLIT layout (Bluestein): the plan could not be built",
                   cfg->placement == VFFT_INPLACE ? "in-place" : "out-of-place", N, K);
        return NULL;
    }
    h = (struct vfft_plan_s *)calloc(1, sizeof *h);
    if (!h)
    {
        vfft_zts_blu_destroy(b);
        return NULL;
    }
    h->transform = VFFT_C2C;
    h->placement = cfg->placement;
    h->layout = (int)cfg->layout;
    h->N = N;
    h->K = K;
    h->nthreads = _vfft_plan_threads(cfg);
    h->zts_blu = b;
    if (getenv("VFFT_NAT_LOG"))
        fprintf(stderr, "[zts-blu] N=%d K=%zu %s: Bluestein over the M=%d cell\n", N, K, lm ? "lane-major" : "contiguous", M);
    return (vfft_plan)h;
}

/* the 1D C2C create, split tier: the refusals of the contract, then the
 * cell (the padded batch was retired 2026-11-06: vfft.c refuses the flag) */
static vfft_plan _vfft_create_c2c_split(const vfft_cfg_t *cfg,
                                        struct vfft_wisdom_s *W,
                                        int N,
                                        size_t K)
{
    const int ip = (cfg->placement == VFFT_INPLACE);
    const char *pname = ip ? "in-place" : "out-of-place";
    vfft_zts_plan_t *z;
    if (cfg->order == VFI_ORDER_SCRAMBLED)
    {
        _vfft_warn("vfft_create: %s C2C N=%d K=%zu on the SPLIT layout at VFFT_ORDER_SCRAMBLED: "
                   "the split engine is natural by construction and has no cheaper scrambled "
                   "form — ask VFFT_ORDER_NATURAL (or the default)",
                   pname, N, K);
        return NULL;
    }
    if (K > 1 && cfg->batch_geom != VFFT_BATCH_CONTIGUOUS ? !vfft_zts_serves_v(N) : !vfft_zts_serves(N))
    {
        /* outside the bands: Bluestein over zturn-ts (zts_bluestein.h), the
         * M = next-pow2 cell of this form inside; a lane-major K off the
         * vector width is refused below like any lane-major request */
        const int lm = K > 1 && cfg->batch_geom != VFFT_BATCH_CONTIGUOUS;
        if (vfft_zts_blu_m(N) && !(lm && (K % (size_t)VFFT_ZTS_VW != 0 || K > (size_t)VFFT_ZTS_LANES_MAX)))
            return _vfft_create_c2c_blu_split(cfg, W, N, K, lm);
        if (K > 1 && cfg->batch_geom != VFFT_BATCH_CONTIGUOUS)
            _vfft_warn("vfft_create: %s C2C N=%d K=%zu lane-major on the SPLIT layout: the lane-major "
                       "engine serves N a product of 2 to 7 radices over 3, 4, 5, 7, 8, 9, 15 (the "
                       "power of two in 4s and 8s) up to %d — use VFFT_LAYOUT_INTERLEAVED for this N",
                       pname, N, K, VFFT_ZTS_MAX_N);
        else
            _vfft_warn("vfft_create: %s C2C N=%d K=%zu on the SPLIT layout: the split engine serves "
                       "the powers of two from %d to %d and the lengths 2^a * m (a >= 4, m over 3, 5, 7, "
                       "9, 15) up to %d — use VFFT_LAYOUT_INTERLEAVED for this N",
                       pname, N, K, VFFT_ZTS_MIN_N, VFFT_ZTS_MAX_N, VFFT_ZTS_MAX_N);
        return NULL;
    }
    if (K > 1 && cfg->batch_geom == VFFT_BATCH_CONTIGUOUS)
    {
        /* a CONTIGUOUS split batch (transform t at offset t*N in each plane)
         * = K independent K=1 transforms: the q=1 cell's plan run K times
         * (zturn-ts's _batch twin), in slabs over the pool at T > 1 */
        vfft_cfg_t c1 = *cfg;
        struct vfft_plan_s *h1;
        c1.howmany = 1;
        c1.batch_geom = VFFT_BATCH_LANE_MAJOR; /* identical at K=1 */
        h1 = (struct vfft_plan_s *)_vfft_create_c2c_split(&c1, W, N, 1);
        if (!h1)
            return NULL;
        h1->K = K;
        _zts_tcmt_decide(h1, cfg, W, N, K); /* the threading verdict at T > 1 */
        return (vfft_plan)h1;
    }
    if (K > 1 && (K % (size_t)VFFT_ZTS_VW != 0 || K > (size_t)VFFT_ZTS_LANES_MAX))
    {
        _vfft_warn("vfft_create: %s C2C N=%d K=%zu lane-major on the SPLIT layout: the "
                   "lane-major engine takes K a multiple of %d (up to %d) — pad the batch "
                   "to that width, ask VFFT_BATCH_CONTIGUOUS, or use VFFT_LAYOUT_INTERLEAVED",
                   pname, N, K, VFFT_ZTS_VW, VFFT_ZTS_LANES_MAX);
        return NULL;
    }
    z = _zts_plan_cell(W, cfg, N, K, ip, K == 1 ? 1 : (int)K);
    if (!z)
    {
        _vfft_warn("vfft_create: %s C2C N=%d K=%zu on the SPLIT layout: no zturn-ts chain "
                   "built for the cell (out of memory, or a build without its drivers)",
                   pname, N, K);
        return NULL;
    }
    {
        struct vfft_plan_s *h = _zts_handle(cfg, cfg->placement, N, K, z);
        if (h)
            _zts_tcmt_decide(h, cfg, W, N, K); /* the threading verdict at T > 1: the lane-major
                                                * batch's blocks, or the K = 1 cell's sectioned walk */
        return (vfft_plan)h;
    }
}

#endif /* VFFT_SPLIT_C2C_CREATE_SPLIT_H */
