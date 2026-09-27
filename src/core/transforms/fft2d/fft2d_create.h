/* fft2d_create.h — the rank-2 CREATE tier.
 *
 * WHAT THIS IS
 * ------------
 * The dims==2 arm of _vfft_create_inner: the largest single tier in the
 * dispatcher, and the one that decides between the two rank-2 servings the
 * library actually has. It returns on every path, so it sits behind a guard
 * ahead of the rank-1 tiers.
 *
 * n[0]=N1 (rows), n[1]=N2 (columns).
 *
 * THE TWO SERVINGS, and how the tier chooses
 * ------------------------------------------
 * INTERLEAVED (lay=il) is the native rank-2 route, built in
 * il/rank2/fft2d_create_il.h (with the plane queue for howmany > 1). Its
 * passes, MT and racers live in il/rank2/il2d_tier.h; the IL create settles
 * the row child, the chain, and the banked axes (wl, cut, tfuse, rowoop)
 * before handing off.
 *
 * SPLIT is the other serving, built in split/rank2/fft2d_create_split.h, and
 * it is a different machine end to end -- different codelets, different
 * executor, different planner. The two do not share an interior; the choice
 * is made here, once, and never revisited. This file is the dispatcher.
 *
 * c2c is in-place (tiled-row + native-column). r2c/c2r are out-of-place: a
 * real plane against an N1 x (N2/2+1) spectrum, one plan serving both
 * directions.
 *
 * WHAT IS DECIDED HERE vs WHAT IS RACED
 * -------------------------------------
 * This tier does not invent a plan. Where a choice is open it calls a racer
 * (_il2d_real_rowrace and the rest, all in il2d_tier.h) and banks the verdict;
 * where wisdom already holds a verdict it replays it. A banked line reads back
 * as a verdict, never as a heuristic -- so nothing in this file may grow a
 * hand-written cutoff.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. Like il2d_tier.h, k1_commit.h and zr2c_build.h it
 * calls file-scope statics that live in vfft.c (_vfft_plan_threads,
 * _vw2_lay_of, _vw2_persist, _build_2d), so it must be included after those
 * are defined and before _vfft_create_inner.
 *
 * The four parameters are the block's complete free-variable set: cfg, W,
 * reg, K. N1/N2 are locals declared inside the block.
 */
#ifndef VFFT_TRANSFORMS_FFT2D_CREATE_H
#define VFFT_TRANSFORMS_FFT2D_CREATE_H

#include "fft2d_create_il.h"
#include "fft2d_create_split.h"

static vfft_plan _vfft_create_2d(const vfft_config_t *cfg,
                                 struct vfft_wisdom_s *W,
                                 const vfft_proto_registry_t *reg,
                                 size_t K)
{
    if (cfg->dims == 2)
    {
        /* the Bluestein inner chain provider for THIS create: the (M, N2)
         * chain row serves the length-M column chain */
        _il2d_blu_ctx.W = W;
        _il2d_blu_ctx.cfg = cfg;
        _il2d_blu_ctx.N2 = cfg->n[1];
        _il2d_blu_chain_hook = _il2d_blu_m_chain;
        /* the 2D executors are K-blind — howmany > 1 is served by the
         * PLANE QUEUE (plane_queue.h, sequential-plane batching): a
         * wrapper over one primary howmany=1 plan (loop
         * mode, keeps its intra-MT verdicts) + serial clones pulled by
         * an atomic plane counter (queue mode), loop-vs-queue RACED at
         * create. Contiguous planes only (the canonical dist for each
         * transform); layouts/transforms the tier cannot express keep
         * the loud refusal. */
        if (K != 1)
        {
            if (cfg->layout != VFFT_LAYOUT_INTERLEAVED ||
                (cfg->transform != VFFT_C2C &&
                 cfg->transform != VFFT_R2C &&
                 cfg->transform != VFFT_C2R))
            {
                _vfft_warn("vfft_create: dims=2 howmany=%zu is served by "
                           "the plane queue for INTERLEAVED C2C/R2C/C2R "
                           "only (got %s, layout=%d) — batch other 2D "
                           "plans sequentially",
                           K, _vfft_tname(cfg->transform),
                           (int)cfg->layout);
                return NULL;
            }
            return _vfft_create_2d_pq_il(cfg, W, K);
        }
        /* the one layout fork: the native IL tier serves IL c2c (any
         * placement) and IL real out of place; everything else is split. */
        if (cfg->layout == VFFT_LAYOUT_INTERLEAVED &&
            (cfg->transform == VFFT_C2C ||
             ((cfg->transform == VFFT_R2C || cfg->transform == VFFT_C2R) &&
              cfg->placement == VFFT_OUTOFPLACE)))
            return _vfft_create_2d_il(cfg, W, K);
        return _vfft_create_2d_split(cfg, W, reg, K);
    }
    return NULL; /* unreachable: the one call site guards on the same
                  * condition, and every path in the block above returns. */
}

#endif /* VFFT_TRANSFORMS_FFT2D_CREATE_H */
