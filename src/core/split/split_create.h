/* split_create.h — the SPLIT create: the split side of vfft_create's one fork.
 *
 * _vfft_create_inner validates the config, applies the layout-independent
 * refusals, routes 1D interleaved real to its door, and then forks ONCE on
 * config.layout: SPLIT requests come here, INTERLEAVED ones go to
 * il/il_create.h. From here on nothing tests the layout again; every tier
 * below is a split tier and includes only split/ and common/.
 *
 * THE SPLIT SIDE IN v1.0 (docs/design/split_v1_0_contract.md): C2C only, on
 * one engine, zturn-ts: rank 1 (split/rank1/c2c_create_split.h), rank 2 and 3
 * as their axes (split/rankn/c2c_nd_create_split.h). The real-to-real
 * transforms (DCT/DST/DHT), which lived on the retired stride engine alone,
 * are refused, as is rank 4. Every refusal names the interleaved layout.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: the tier calls file-scope statics of
 * vfft.c (_vfft_warn, _vfft_tname, _vfft_plan_threads, _vw2_persist), so this
 * is included after those and before _vfft_create_inner.
 */
#ifndef VFFT_SPLIT_CREATE_H
#define VFFT_SPLIT_CREATE_H

#include "split/rank1/c2c_create_split.h"
#include "split/rankn/c2c_nd_create_split.h"

static vfft_plan _vfft_split_create(const vfft_cfg_t *cfg,
                                    struct vfft_wisdom_s *W,
                                    int N,
                                    size_t K)
{
    if (cfg->dims >= 4)
    {
        _vfft_warn("vfft_create: rank %d %s on the SPLIT layout is not served (the split side serves "
                   "rank 1, 2 and 3 C2C) — use VFFT_LAYOUT_INTERLEAVED",
                   cfg->dims, _vfft_tname(cfg->transform));
        return NULL;
    }
    if (cfg->dims >= 2 && cfg->transform == VFFT_C2C)
        return _vfft_create_c2c_nd_split(cfg, W, K);
    if (_VFFT_IS_TRIG(cfg->transform))
    {
        _vfft_warn("vfft_create: %s N=%d is not served: the real-to-real transforms were "
                   "retired with the split stride engine (v1.0 serves C2C, R2C and C2R)",
                   _vfft_tname(cfg->transform), N);
        return NULL;
    }
    if (cfg->transform != VFFT_C2C)
    {
        _vfft_warn("vfft_create: %s N=%d K=%zu on the SPLIT layout: the split side serves "
                   "C2C only — use VFFT_LAYOUT_INTERLEAVED",
                   _vfft_tname(cfg->transform), N, K);
        return NULL; /* a split R2C / C2R request lands here (the bridge to the split real engines deleted 2026-11-06) */
    }
    return _vfft_create_c2c_split(cfg, W, N, K);
}

#endif /* VFFT_SPLIT_CREATE_H */
