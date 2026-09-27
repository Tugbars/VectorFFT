/* fftnd_create.h — the rank-3 and rank-4 CREATE tiers.
 *
 * WHAT THIS IS
 * ------------
 * The dims==4 and dims==3 arms of _vfft_create_inner. Layout separation
 * phase 6: this file is now the front door's DISPATCHER on the committed
 * layout - the interleaved tier is _vfft_create_rank34_il (il/rank3/fftnd_il.h),
 * the split tier _vfft_create_rank34_split (split/rank3/fftnd_create_split.h).
 *
 * CONTRACTS
 * ---------
 * K == 1. A batched rank>=3 call arrives as a K=1 override plan, not as a
 * howmany the engines see. SPLIT order is DEFAULT or SCRAMBLED only; rank-3
 * split NATURAL is refused loudly here (its planned follow-up is
 * fftnd_natorder.h's nat_col_list). Real transforms are out-of-place.
 * Trig (DCT/DST/DHT) is 1D only and is refused above this helper, in the
 * shared dims>=2 guard. INTERLEAVED rank-3 c2c goes to fftnd_il.h.
 *
 * WISDOM
 * ------
 * Rank-3 split c2c: a dedicated (N1,N2,N3) row in the wisdom2 store. HIT ->
 * vfft_fft3d_plan_from_entry. MISS -> greedy per-axis exhaustive with the
 * inners visible, banked through vw2_3d_bank_entry when the result is
 * expressible. The rank-4 c2c arm has no wisdom: stride_plan_nd's per-axis
 * search runs at every create.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. It calls three file-scope statics that live in
 * vfft.c -- _vfft_plan_threads, _vw2_lay_of, _vw2_persist -- exactly as
 * il2d_tier.h, k1_commit.h and zr2c_build.h already do, so it must be included
 * after those are defined and before _vfft_create_inner. Its other callees
 * (stride_plan_nd, stride_plan_nd_r2c, the vfft_fft3d_* and vw2_3d_* wisdom
 * entry points) come from fftnd.h, fftnd_r2c.h and the wisdom2 readers, all
 * included far earlier.
 */
#ifndef VFFT_TRANSFORMS_FFTND_CREATE_H
#define VFFT_TRANSFORMS_FFTND_CREATE_H

#include "fftnd_create_split.h"   /* the split rank-3/4 tier */

/* Rank-3/rank-4 create: one layout fork, each library its own tier. */
static vfft_plan _vfft_create_rank34(const vfft_config_t *cfg,
                    struct vfft_wisdom_s *W,
                    const vfft_proto_registry_t *reg,
                    size_t K)
{
    if (cfg->layout == VFFT_LAYOUT_INTERLEAVED)
        return _vfft_create_rank34_il(cfg, W, K);
    return _vfft_create_rank34_split(cfg, W, reg, K);
}

#endif /* VFFT_TRANSFORMS_FFTND_CREATE_H */
