/* c2c_create_split.h — the split library's 1D C2C create: zturn-ts, both
 * placements (docs/design/split_v1_0_contract.md).
 *
 * WHAT THE SPLIT SIDE SERVES: a power-of-two N the zturn-ts registry holds,
 * natural order, in place or out of place; K = 1; a CONTIGUOUS batch of any
 * K (the K = 1 plan run K times, in slabs over the pool at T > 1); a
 * LANE-MAJOR batch of any K (the lanes past the last whole vector group run
 * the plan's internal zero-padded tail, 2026-11-07). Everything else is
 * refused here, loudly, naming the interleaved layout: an N outside the
 * bands and Bluestein's reach, a scrambled order, a caller-owned (padded)
 * batch. The split layout is for highly batched low-N work;
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

/* ── THE THREADING VERDICT of a split cell at T > 1 ──────────────────────
 * A cell has a LIST OF FORMS (split_execute.h; zts_mt.h; zts_blu_mt.h), the
 * same for a zturn-ts cell and for a Bluestein cell around its inner M cell:
 *   serial;
 *   SLABS of whole transforms (a contiguous batch of K >= T);
 *   the SECTIONED walk's BLOCKS and TILES (one transform; a contiguous K < T
 *     sharing every stage's dispatch; a lane-major batch, block by block);
 *   lane BLOCK ranges (a lane-major batch of two or more blocks) and lane
 *     GROUP ranges (two or more groups: the same slots at group granularity,
 *     each thread's row visits holding fewer groups than the banked block).
 * Raced at create (serial vs every form the cell has, the library body) and
 * banked eng=tcb tcmt= on the cell's threading row at nthreads=T
 * (common/wisdom/wisdom2_tcmt.h); the row's VALUE names the form, so a row
 * banked under one form is never read under another: 0 serial; 1 slabs /
 * lane block ranges; 3 the sectioned BLOCKS; 4 the sectioned TILES; 5 lane
 * group ranges. No worker planes => no slab / range forms. VFFT_TCMT=<value>
 * pins a form (never replays, never banks). */
#define VFFT_ZTS_TCMT_SECT_BLOCKS 3
#define VFFT_ZTS_TCMT_SECT_TILES 4
#define VFFT_ZTS_TCMT_GROUPS 5
typedef struct
{
    const char *name;
    int mt, sect, grp;
} _zts_tcmt_form_t;
static const _zts_tcmt_form_t _ZTS_FORM_SERIAL = { "serial", 0, 0, 0 };
static int _zts_tcmt_value(const _zts_tcmt_form_t *f)
{
    return f->mt == 0 ? 0 : f->sect ? (f->mt == 2 ? VFFT_ZTS_TCMT_SECT_TILES : VFFT_ZTS_TCMT_SECT_BLOCKS)
                          : f->grp ? VFFT_ZTS_TCMT_GROUPS : 1;
}
/* lanes per range unit of the lane forms: a whole block, or one cache line
 * of lanes (the line the descriptor read; two groups on a 64-byte line) */
static size_t _zts_lane_unit(int blk, int grp)
{
    if (!grp) return (size_t)blk * VFFT_ZTS_VW;
    {
        const long line = vfft_cpu_cache()->line_bytes;
        size_t u = (size_t)(line > 0 ? line : 64) / sizeof(double);
        if (u < VFFT_ZTS_VW) u = VFFT_ZTS_VW;
        return u - u % VFFT_ZTS_VW;
    }
}
/* the form onto the handle's engine: zturn-ts, or Bluestein over it */
static void _zts_tcmt_set(struct vfft_plan_s *h, const _zts_tcmt_form_t *f)
{
    if (h->zts_blu)
        _zts_blu_set_mt(h->zts_blu, f->mt, f->sect, f->grp, _zts_lane_unit(h->zts_blu->inner->blk, f->grp));
    else
    {
        h->zts->mt = f->mt;
        h->zts->mt_sect = f->sect;
        h->zts->mt_grp = f->grp;
        h->zts->mt_unit = _zts_lane_unit(h->zts->blk, f->grp);
    }
}
static int _zts_tcmt_mt(const struct vfft_plan_s *h)
{
    return h->zts_blu ? h->zts_blu->mt : h->zts->mt;
}
typedef struct
{
    struct vfft_plan_s *h;
    double *re, *im, *ore, *oim;
    const _zts_tcmt_form_t *f;
} _zts_tcmt_arm_t;
static void _zts_tcmt_arm_run(void *v)
{
    _zts_tcmt_arm_t *a = (_zts_tcmt_arm_t *)v;
    _zts_tcmt_set(a->h, a->f);
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
#define VFFT_ZTS_TCMT_FORMS 6
/* the race: serial (forms[0]) vs every other form, on this cell's own
 * planes; leaves the handle bound to the winner, banks its value */
static void _zts_tcmt_race(struct vfft_plan_s *h, const vfft_cfg_t *cfg, struct vfft_wisdom_s *W,
                           int N, size_t K, int ip, int pl, uint8_t lay,
                           const _zts_tcmt_form_t *forms, int nf, int lg, const char *eng)
{
    const int T = h->nthreads;
    const size_t n = (size_t)N * K, nb = n * sizeof(double);
    double *re = (double *)vfft_aligned_alloc(nb), *im = (double *)vfft_aligned_alloc(nb);
    double *ore = ip ? re : (double *)vfft_aligned_alloc(nb);
    double *oim = ip ? im : (double *)vfft_aligned_alloc(nb);
    double *re0 = ip ? (double *)vfft_aligned_alloc(nb) : NULL;
    double *im0 = ip ? (double *)vfft_aligned_alloc(nb) : NULL;
    size_t i;
    if (!re || !im || !ore || !oim || (ip && (!re0 || !im0)))
    {
        vfft_aligned_free(re);
        vfft_aligned_free(im);
        if (!ip) { vfft_aligned_free(ore); vfft_aligned_free(oim); }
        vfft_aligned_free(re0);
        vfft_aligned_free(im0);
        _zts_tcmt_set(h, &_ZTS_FORM_SERIAL);
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
        _zts_tcmt_arm_t ax[VFFT_ZTS_TCMT_FORMS];
        vfft_race_arm_t arms[VFFT_ZTS_TCMT_FORMS];
        _zts_tcmt_reseed_t rs = { re, im, re0, im0, nb };
        vfft_race_proto_t proto;
        double ns[VFFT_ZTS_TCMT_FORMS];
        int best = 0, k, reps;
        for (k = 0; k < nf; k++)
        {
            ax[k].h = h; ax[k].re = re; ax[k].im = im; ax[k].ore = ore; ax[k].oim = oim; ax[k].f = &forms[k];
            arms[k].name = forms[k].name; arms[k].run = _zts_tcmt_arm_run; arms[k].ctx = &ax[k];
        }
        {   /* reps from one serial timing: ~4 ms of serial-equivalent work per
             * sample (the interleaved ZTT-MT race's law), min of 3 rounds — one
             * execute per sample at a small N flapped the verdict. The in-place
             * arms repeat on the same buffer inside a sample (the values walk
             * into inf; the timing holds, as on the interleaved side). */
            double t0;
            _zts_tcmt_set(h, &forms[0]);
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
        vfft_race_run(&proto, arms, nf, ns); /* threaded arms: no pace */
        for (k = 1; k < nf; k++)
            if (ns[k] < ns[best]) best = k;
        _zts_tcmt_set(h, &forms[best]);
        if (lg)
        {
            fprintf(stderr, "[tcmt] %s N=%d K=%zu T=%d %s: race", eng, N, K, T, ip ? "ip" : "oop");
            for (k = 0; k < nf; k++) fprintf(stderr, " %s=%.0f", forms[k].name, ns[k]);
            fprintf(stderr, " -> %s (tcmt=%d)\n", forms[best].name, _zts_tcmt_value(&forms[best]));
        }
        vfft_aligned_free(re);
        vfft_aligned_free(im);
        if (!ip) { vfft_aligned_free(ore); vfft_aligned_free(oim); }
        vfft_aligned_free(re0);
        vfft_aligned_free(im0);
        if (W && vw2_tcmt_bank(&W->vw2, VW2_T_C2C, N, K, VW2_ORD_NAT, pl, lay,
                               _zts_tcmt_value(&forms[best]), T, ns[best]) == VW2_OK)
            _vw2_persist(W, cfg);
    }
}
/* the pinned form (VFFT_TCMT = its row value) or the banked row, among the
 * cell's forms: 1 = decided, 0 = race */
static int _zts_tcmt_replay(struct vfft_plan_s *h, const vfft_cfg_t *cfg, struct vfft_wisdom_s *W,
                            int N, size_t K, int pl, uint8_t lay,
                            const _zts_tcmt_form_t *forms, int nf, int lg, const char *eng)
{
    const int T = h->nthreads;
    const char *pin = getenv("VFFT_TCMT");
    int v = 0, k;
    if (pin)
    {
        v = atoi(pin);
        for (k = 0; k < nf; k++)
            if (_zts_tcmt_value(&forms[k]) == v) break;
        _zts_tcmt_set(h, k < nf ? &forms[k] : &forms[0]);
        if (lg)
            fprintf(stderr, "[tcmt] %s N=%d K=%zu T=%d: pinned tcmt=%d -> %s (env; not banked)\n", eng, N, K, T, v,
                    k < nf ? forms[k].name : "serial (no such form here)");
        return 1;
    }
    if (W && !cfg->recalibrate && vw2_tcmt_lookup(&W->vw2, VW2_T_C2C, N, K, VW2_ORD_NAT, pl, lay, T, &v))
    {
        for (k = 0; k < nf; k++)
            if (_zts_tcmt_value(&forms[k]) == v) break;
        if (k < nf)
        {
            _zts_tcmt_set(h, &forms[k]);
            if (lg)
                fprintf(stderr, "[tcmt] %s N=%d K=%zu T=%d: replay tcmt=%d -> %s src=wisdom\n", eng, N, K, T, v, forms[k].name);
            return 1;
        }
        if (lg)
            fprintf(stderr, "[tcmt] %s N=%d K=%zu T=%d: the row's tcmt=%d names a form this cell lacks; racing\n", eng, N, K, T, v);
    }
    return 0;
}
/* the forms of a cell: the engine's stage table, lanes, block, tile and the
 * batch shape decide which exist; the worker planes they need are built
 * (none => the range / slab forms drop out). Returns the count, serial first. */
static int _zts_tcmt_forms(vfft_zts_plan_t *p, int T, size_t K, int inner_only, _zts_tcmt_form_t *out, int *nw_out)
{
    const int lm = p->lanes > 1;
    int nf = 0, nw = 0;
    out[nf++] = _ZTS_FORM_SERIAL;
    if (!lm)
    {
        const int can_sect = p->s0f && p->slf && (K == 1 || (size_t)T / K >= 2);
        const int sect = can_sect && (K == 1 || (K < (size_t)T && (size_t)T / K >= 2));
        if (sect)
        {
            if (K > 1) nw = (int)K - 1;   /* the sectioned contiguous batch: a plane per transform */
            out[nf].name = "blocks"; out[nf].mt = 1; out[nf].sect = 1; out[nf].grp = 0; nf++;
            if (p->tile && (size_t)p->N / p->tile >= 2) { out[nf].name = "tiles"; out[nf].mt = 2; out[nf].sect = 1; out[nf].grp = 0; nf++; }
        }
        else if (K >= 2)
        {
            nw = T - 1;
            if ((size_t)nw > K - 1) nw = (int)K - 1;
            out[nf].name = "slabs"; out[nf].mt = 1; out[nf].sect = 0; out[nf].grp = 0; nf++;
        }
    }
    else
    {
        const size_t groups = (size_t)p->lanes / VFFT_ZTS_VW;
        const size_t blocks = (groups + (size_t)p->blk - 1) / (size_t)p->blk;
        if (groups >= 2)
        {
            nw = T - 1;
            if ((size_t)nw > groups - 1) nw = (int)(groups - 1);   /* enough for either granularity */
            if (blocks >= 2) { out[nf].name = "blocks"; out[nf].mt = 1; out[nf].sect = 0; out[nf].grp = 0; nf++; }
            out[nf].name = "groups"; out[nf].mt = 1; out[nf].sect = 0; out[nf].grp = 1; nf++;
        }
        if (p->v0f && p->vlf)
        {
            out[nf].name = "sect-blocks"; out[nf].mt = 1; out[nf].sect = 1; out[nf].grp = 0; nf++;
            if (p->tile && (size_t)p->N / p->tile >= 2) { out[nf].name = "sect-tiles"; out[nf].mt = 2; out[nf].sect = 1; out[nf].grp = 0; nf++; }
        }
    }
    (void)inner_only;
    if (nw > VFFT_ZTS_MT_MAX) nw = VFFT_ZTS_MT_MAX;
    *nw_out = nw;
    return nf;
}
/* the range / slab forms dropped from the list when no worker plane exists */
static int _zts_tcmt_drop_planes(_zts_tcmt_form_t *forms, int nf)
{
    int k, n = 0;
    for (k = 0; k < nf; k++)
        if (forms[k].mt == 0 || forms[k].sect) forms[n++] = forms[k];
    return n;
}
/* a zturn-ts cell's verdict */
static void _zts_tcmt_decide(struct vfft_plan_s *h, const vfft_cfg_t *cfg,
                             struct vfft_wisdom_s *W, int N, size_t K)
{
    const int T = h->nthreads;
    const int ip = (h->placement == VFFT_INPLACE);
    const int pl = ip ? VW2_PL_IP : VW2_PL_OOP;
    const int lm = h->zts->lanes > 1;                       /* the lane-major batch: its own cell beside the contiguous batch's rows */
    const uint8_t lay = lm ? VW2_LAY_SLM : VW2_LAY_SPLIT;
    const int lg = getenv("VFFT_TCMT_VERBOSE") || getenv("VFFT_TCMT_LOG");
    _zts_tcmt_form_t forms[VFFT_ZTS_TCMT_FORMS];
    int nf, nw = 0;
    _zts_tcmt_set(h, &_ZTS_FORM_SERIAL);
    if (T <= 1 || getenv("VFFT_NO_TCMT"))
        return;
    nf = _zts_tcmt_forms(h->zts, T, K, 0, forms, &nw);
    if (nw > 0 && vfft_zts_mt_planes(h->zts, nw) == 0)
        nf = _zts_tcmt_drop_planes(forms, nf);   /* no worker planes: no range / slab form */
    if (nf < 2)
        return;
    if (_zts_tcmt_replay(h, cfg, W, N, K, pl, lay, forms, nf, lg, "zts"))
        return;
    _zts_tcmt_race(h, cfg, W, N, K, ip, pl, lay, forms, nf, lg, "zts");
}
/* ── THE BLUESTEIN CELLS' THREADING RULE (owner 2026-11-06: the split side's
 * threading lives in the split ESTIMATE, Bluestein first — the cells with
 * the evidence; the default create takes the rule too: no race, no threading
 * row). The plane is the convolution length M by the lanes (one transform:
 * one lane; the contiguous batch: K transforms of it). THE CUT FOLLOWS THE
 * PLANE'S LONGER AXIS, each cut with its floor in work per fork-join
 * (W1 complex elements per thread; the pool's dispatch against the span):
 *   a one-dispatch form — slabs of whole transforms, lane ranges — needs
 *     M * K >= W1 * T (slabs) or M * lanes >= 4 * W1 * T (lane ranges: the
 *     strided slice costs more than a contiguous one);
 *   the sectioned walk, nf + 2 fork-joins per block, needs M >= 8 * W1 * T
 *     for one transform and M * lanes >= 16 * W1 * T for a lane-major batch,
 *     and M >= W1 * T either way;
 *   a lane cut needs a cache line per thread per row, lanes >= line / 8 * T,
 *     and its unit is the line, never the TLB block;
 *   above both floors the ratio lanes / M decides: >= 1/16 the lanes,
 *     <= 1/64 the stages, between them the stages at M >= W1 * T and the
 *     lanes below.
 * W1 = 512 on the 14900KF at eight threads: 45 of 45 measured cells within
 * 10 % of the best form, the worst 3 % (docs/research/split_odd_band_2026_
 * 11_06.md); the thresholds await the Zen 4 confirmation. The line width
 * and the thread count are read, not guessed; VFFT_TCMT=<value> still pins
 * a form (a developer's override). */
#define VFFT_ZTS_BLU_RULE_W1 512
static void _zts_blu_tcmt_decide(struct vfft_plan_s *h, const vfft_cfg_t *cfg,
                                 struct vfft_wisdom_s *W, int N, size_t K)
{
    static const _zts_tcmt_form_t SLABS = { "slabs", 1, 0, 0 }, BLOCKS = { "blocks", 1, 0, 0 },
                                  GROUPS = { "groups", 1, 0, 1 }, SECT_B = { "sect-blocks", 1, 1, 0 },
                                  SECT_T = { "sect-tiles", 2, 1, 0 };
    vfft_zts_blu_t *b = h->zts_blu;
    const int T = h->nthreads;
    const int lm = b->lanes > 1;
    const int lg = getenv("VFFT_TCMT_VERBOSE") || getenv("VFFT_TCMT_LOG");
    const char *pin = getenv("VFFT_TCMT");
    const size_t M = (size_t)b->M, lanes = lm ? (size_t)b->lanes : 1;
    const size_t line_lanes = _zts_lane_unit(1, 1);
    const size_t w1 = (size_t)VFFT_ZTS_BLU_RULE_W1 * (size_t)T;
    const int tiled = b->inner->tile && (size_t)b->inner->N / b->inner->tile >= 2;
    const int sect_s = b->inner->s0f && b->inner->slf, sect_v = b->inner->v0f && b->inner->vlf;
    const _zts_tcmt_form_t *f = &_ZTS_FORM_SERIAL;
    int nw = 0, i;
    (void)cfg; (void)W;
    _zts_tcmt_set(h, &_ZTS_FORM_SERIAL);
    if (T <= 1 || getenv("VFFT_NO_TCMT"))
        return;
    if (pin)
    {
        switch (atoi(pin))
        {
        case 1: f = lm ? (b->turned ? &SLABS : &BLOCKS) : (K > 1 ? &SLABS : &_ZTS_FORM_SERIAL); break;
        case VFFT_ZTS_TCMT_GROUPS: f = lm && !b->turned ? &GROUPS : &_ZTS_FORM_SERIAL; break;
        case VFFT_ZTS_TCMT_SECT_BLOCKS: f = (lm ? (sect_v && !b->turned) : (K == 1 && sect_s)) ? &SECT_B : &_ZTS_FORM_SERIAL; break;
        case VFFT_ZTS_TCMT_SECT_TILES: f = (lm ? (sect_v && !b->turned) : (K == 1 && sect_s)) && tiled ? &SECT_T : &_ZTS_FORM_SERIAL; break;
        default: break;
        }
    }
    else if (!lm)
    {
        if (K == 1)
        {   /* one transform: the sectioned walk from 8 W1 T */
            if (sect_s && M >= 8 * w1) f = tiled ? &SECT_T : &SECT_B;
        }
        else if (M * K >= w1)
            f = &SLABS;   /* the contiguous batch: one dispatch of whole transforms */
    }
    else if (b->turned)
    {   /* the turned form: slabs of lane blocks, the contiguous rule (one dispatch) */
        if (M * lanes >= w1 && (lanes + VFFT_ZTS_BLU_ROWS - 1) / VFFT_ZTS_BLU_ROWS >= 2) f = &SLABS;
    }
    else
    {
        const int lane_ok = lanes >= line_lanes * (size_t)T && M * lanes >= 4 * w1;
        const int sect_ok = sect_v && M * lanes >= 16 * w1 && M >= w1;
        const double r = (double)lanes / (double)M;
        if (lane_ok && (!sect_ok || r >= 1.0 / 16)) f = &GROUPS;
        else if (sect_ok && (!lane_ok || r <= 1.0 / 64)) f = tiled ? &SECT_T : &SECT_B;
        else if (lane_ok && sect_ok) f = M >= w1 ? (tiled ? &SECT_T : &SECT_B) : &GROUPS;
    }
    if (f->mt && !f->sect)
    {   /* the worker planes: one per slot beyond the caller (the slab form's own za planes too) */
        const size_t units = lm ? (b->turned ? (lanes + VFFT_ZTS_BLU_ROWS - 1) / VFFT_ZTS_BLU_ROWS
                                             : (lanes + (f->grp ? line_lanes : (size_t)b->inner->blk * VFFT_ZTS_VW) - 1) /
                                                   (f->grp ? line_lanes : (size_t)b->inner->blk * VFFT_ZTS_VW))
                                : K;
        const size_t za = (size_t)2 * (b->turned ? VFFT_ZTS_BLU_ROWS : 1) * M;   /* a slot's own row scratch (re then im) */
        int got;
        nw = T - 1;
        if ((size_t)nw > units - 1) nw = (int)(units - 1);
        if (nw > VFFT_ZTS_MT_MAX) nw = VFFT_ZTS_MT_MAX;
        got = nw > 0 ? vfft_zts_mt_planes(b->inner, nw) : 0;
        if (lm && !b->turned)
            b->mt_n = got;
        else
            for (i = b->mt_n; i < got; i++)
            {
                b->mtza[i] = (double *)vfft_aligned_alloc(za * sizeof(double));
                if (!b->mtza[i]) break;
                b->mt_n = i + 1;
            }
        if (b->mt_n < 1) f = &_ZTS_FORM_SERIAL;
    }
    _zts_tcmt_set(h, f);
    if (lg)
        fprintf(stderr, "[tcmt] zts-blu N=%d K=%zu T=%d: the rule -> %s (M=%zu lanes=%zu line=%zu lanes W1*T=%zu%s)\n",
                N, K, T, f->name, M, lanes, line_lanes, w1, pin ? "; pinned" : "");
}

/* ── Bluestein over zturn-ts (zts_bluestein.h): the M cell of the request's
 * form planned through the cell planner (its own replay / race / bank), plus
 * a K = 1 plan at M for the kernel spectra; the handle carries the Bluestein
 * plan, nothing else; its threading verdict raced and banked like any cell
 * (_zts_blu_tcmt_decide). */
static vfft_plan _vfft_create_c2c_blu_split(const vfft_cfg_t *cfg, struct vfft_wisdom_s *W,
                                            int N, size_t K, int lm)
{
    const int M = vfft_zts_blu_m(N);
    const int lanes = lm ? (int)K : 1;
    vfft_zts_plan_t *inner, *ks;
    vfft_zts_blu_t *b;
    struct vfft_plan_s *h;
    /* the M cell, in place. The S cell (K = 1) for one transform, the contiguous batch and the
     * TURNED lane-major form (2026-11-06: the convolution per row, the lanes turned by the
     * modulate / demodulate passes, slabs of line-wide lane blocks over the slots). The V cell
     * at M with K lanes (the chirp broadcast over the lanes, the sectioned walk over the pool)
     * serves the lane-major batch BELOW THE LANE FLOOR at T > 1 — fewer lanes than a cache
     * line per thread, where the turned form's slabs cannot fill the pool and a block narrower
     * than a line would have the slots sharing the output lines. The same floor as the
     * threading rule's (the line width read from the machine). */
    {
        const int T = _vfft_plan_threads(cfg);
        const int v_cell = lm && T > 1 && K < _zts_lane_unit(1, 1) * (size_t)T;
        inner = _zts_plan_cell(W, cfg, M, v_cell ? K : 1, 1, v_cell ? lanes : 1);
    }
    if (!inner)
    {
        _vfft_warn("vfft_create: %s C2C N=%d K=%zu on the SPLIT layout (Bluestein): no zturn-ts cell at the "
                   "convolution length M=%d", cfg->placement == VFFT_INPLACE ? "in-place" : "out-of-place", N, K, M);
        return NULL;
    }
    ks = inner->lanes == 1 ? inner : _zts_plan_cell(W, cfg, M, 1, 1, 1);   /* the kernels' K = 1 plan at M */
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
    _zts_blu_tcmt_decide(h, cfg, W, N, K); /* the threading verdict at T > 1 (zts_blu_mt.h) */
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
         * M = next-pow2 cell of this form inside; a lane-major K above the
         * lane bound is refused below like any lane-major request */
        const int lm = K > 1 && cfg->batch_geom != VFFT_BATCH_CONTIGUOUS;
        if (vfft_zts_blu_m(N) && !(lm && K > (size_t)VFFT_ZTS_LANES_MAX))
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
    if (K > (size_t)VFFT_ZTS_LANES_MAX)
    {   /* any lane count up to the bound: a count off the vector width runs its tail
         * lanes through a zero-padded group (zts.h _zts_run_tail, 2026-11-07) */
        _vfft_warn("vfft_create: %s C2C N=%d K=%zu lane-major on the SPLIT layout: the "
                   "lane-major engine takes K up to %d — ask VFFT_BATCH_CONTIGUOUS, or use VFFT_LAYOUT_INTERLEAVED",
                   pname, N, K, VFFT_ZTS_LANES_MAX);
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
