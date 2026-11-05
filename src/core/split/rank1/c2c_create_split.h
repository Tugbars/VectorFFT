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
static vfft_zts_plan_t *_zts_plan_cell(struct vfft_wisdom_s *W, const vfft_cfg_t *cfg,
                                       int N, size_t K, int place_ip, int lanes)
{
    int stale = 0;
    vfft_zts_plan_t *z = cfg->recalibrate ? NULL : _zts_replay(W, N, K, place_ip, lanes, &stale);
    double wns = 0.0, ins = 0.0;
    if (z)
        return z;
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
    return z;
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
    int nw;
    h->zts->mt = 0;
    if (T <= 1 || K <= 1 || getenv("VFFT_NO_TCMT"))
        return;
    units = lm ? ((size_t)h->zts->lanes + (size_t)h->zts->blk * VFFT_ZTS_VW - 1) / ((size_t)h->zts->blk * VFFT_ZTS_VW) : K;
    if (units <= 1)
        return; /* one unit: nothing to slice */
    nw = T - 1;
    if ((size_t)nw > units - 1)
        nw = (int)(units - 1);
    if (nw > VFFT_ZTS_MT_MAX)
        nw = VFFT_ZTS_MT_MAX;
    if (vfft_zts_mt_planes(h->zts, nw) == 0)
        return; /* no worker planes: no arm */
    if (pin)
    {
        h->zts->mt = atoi(pin) ? 1 : 0;
        if (lg)
            fprintf(stderr, "[tcmt] zts N=%d K=%zu T=%d: pinned tcmt=%d (env; not banked)\n", N, K, T, h->zts->mt);
        return;
    }
    if (W && !cfg->recalibrate)
    {
        int v = 0;
        if (vw2_tcmt_lookup(&W->vw2, VW2_T_C2C, N, K, VW2_ORD_NAT, pl, lay, T, &v))
        {
            h->zts->mt = v;
            if (lg)
                fprintf(stderr, "[tcmt] zts N=%d K=%zu T=%d: replay tcmt=%d src=wisdom\n", N, K, T, v);
            return;
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
            _zts_tcmt_reseed_t rs = { re, im, re0, im0, nb };
            const vfft_race_arm_t arms[2] = { { "serial", _zts_tcmt_arm_run, &a },
                                              { "slabs", _zts_tcmt_arm_run, &b } };
            vfft_race_proto_t proto;
            double ns[2];
            memset(&proto, 0, sizeof proto);
            proto.rounds = ip ? 9 : 7;
            proto.reps = ip ? 1 : (int)(32768u / (n ? n : 1)) + 1; /* >= ~30 us a sample */
            proto.agg = VFFT_RACE_MIN;
            proto.alternate = 1;
            proto.warm = 1;
            proto.reset = ip ? _zts_tcmt_reseed : NULL;
            proto.reset_ctx = ip ? &rs : NULL;
            vfft_race_run(&proto, arms, 2, ns); /* threaded arms: no pace */
            st = ns[0];
            mt = ns[1];
        }
        h->zts->mt = (mt < st);
        if (lg)
            fprintf(stderr, "[tcmt] zts N=%d K=%zu T=%d %s: race serial=%.0f slabs=%.0f -> %s\n",
                    N, K, T, ip ? "ip" : "oop", st, mt, h->zts->mt ? "SLABS" : "serial");
        vfft_aligned_free(re);
        vfft_aligned_free(im);
        if (!ip) { vfft_aligned_free(ore); vfft_aligned_free(oim); }
        vfft_aligned_free(re0);
        vfft_aligned_free(im0);
        if (W && vw2_tcmt_bank(&W->vw2, VW2_T_C2C, N, K, VW2_ORD_NAT, pl, lay,
                               h->zts->mt, T, h->zts->mt ? mt : st) == VW2_OK)
            _vw2_persist(W, cfg);
    }
}

/* the 1D C2C create, split tier: the refusals of the contract, then the
 * cell. ob = a caller-owned batch descriptor (padded planes): never served
 * here. */
static vfft_plan _vfft_create_c2c_split(const vfft_cfg_t *cfg,
                                        vfft_batch ob,
                                        struct vfft_wisdom_s *W,
                                        int N,
                                        size_t K)
{
    const int ip = (cfg->placement == VFFT_INPLACE);
    const char *pname = ip ? "in-place" : "out-of-place";
    vfft_zts_plan_t *z;
    if (ob)
    {
        _vfft_warn("vfft_create: %s C2C N=%d K=%zu on the SPLIT layout with a caller-owned "
                   "(padded) batch: the split layout serves plain planes only — drop "
                   "VFFT_FLAG_OWN_PITCH, or use VFFT_LAYOUT_INTERLEAVED",
                   pname, N, K);
        return NULL;
    }
    if (cfg->order == VFI_ORDER_SCRAMBLED)
    {
        _vfft_warn("vfft_create: %s C2C N=%d K=%zu on the SPLIT layout at VFFT_ORDER_SCRAMBLED: "
                   "the split engine is natural by construction and has no cheaper scrambled "
                   "form — ask VFFT_ORDER_NATURAL (or the default)",
                   pname, N, K);
        return NULL;
    }
    if (!vfft_zts_serves(N))
    {
        _vfft_warn("vfft_create: %s C2C N=%d K=%zu on the SPLIT layout: the split engine serves "
                   "the powers of two from %d to %d only — use VFFT_LAYOUT_INTERLEAVED for "
                   "this N",
                   pname, N, K, VFFT_ZTS_MIN_N, VFFT_ZTS_MAX_N);
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
        h1 = (struct vfft_plan_s *)_vfft_create_c2c_split(&c1, NULL, W, N, 1);
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
        if (h && K > 1)
            _zts_tcmt_decide(h, cfg, W, N, K); /* the lane-major batch's threading verdict at T > 1 */
        return (vfft_plan)h;
    }
}

#endif /* VFFT_SPLIT_C2C_CREATE_SPLIT_H */
