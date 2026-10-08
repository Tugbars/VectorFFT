/* il_create.h — the INTERLEAVED create: the IL side of vfft_create's one fork.
 *
 * _vfft_create_inner validates the config, applies the layout-independent
 * refusals (IL + trig, IL + config.batch, ...), builds the transform-contiguous
 * batch wrapper (1D IL howmany > 1), routes 1D real to bridge/real_bridge.h
 * (the D1 crossing), and then forks ONCE on config.layout: INTERLEAVED
 * requests come here. Split is not a fallback of IL: every tier below serves
 * natively or refuses loudly, and includes only il/ and common/.
 *
 *   rank 3/4   il/rank3/fftnd_il.h        rank-3 c2c; the rest refused
 *   rank 2     il/rank2/fft2d_create_il.h native IL 2D; the plane queue for K>1
 *   c2c ip     il/rank1/c2c_ip_create_il.h
 *   c2c oop    il/rank1/c2c_oop_create_il.h
 *
 * The contracts of these tiers are documented in split/split_create.h (the
 * former dispatcher docs) and in each tier file.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: the tiers call file-scope statics of
 * vfft.c, so this is included after those and before _vfft_create_inner.
 */
#ifndef VFFT_IL_CREATE_H
#define VFFT_IL_CREATE_H

#include "il/rank2/fft2d_create_il.h"
#include "il/rank1/c2c_ip_create_il.h"
#include "il/rank1/c2c_oop_create_il.h"

static vfft_plan _vfft_il_create(const vfft_config_t *cfg,
                                 struct vfft_wisdom_s *W,
                                 int N,
                                 size_t K)
{
    if (cfg->dims == 3 || cfg->dims == 4)
        return _vfft_create_rank34_il(cfg, W, K);
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
         * transform). IL trig is refused before the fork. */
        if (K != 1)
            return _vfft_create_2d_pq_il(cfg, W, K);
        /* IL c2c (any placement), IL real out of place, and IL real in
         * place where the law admits it (vfft_policy_il2d_ip_ok: r2c, one
         * thread; docs/roadmap/real_inplace_design.md) */
        return _vfft_create_2d_il(cfg, W, K);
    }
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE)
        return _c2c_ip_create_il(cfg, W, N, K);   /* config.batch + IL is refused before the fork */
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_OUTOFPLACE)
        return _vfft_create_c2c_oop_il(cfg, W, N, K);
    return NULL; /* unreachable: 1D real went to the bridge and IL trig is
                  * refused before the fork */
}

#endif /* VFFT_IL_CREATE_H */
