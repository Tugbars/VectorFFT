/* split_create.h — the SPLIT create: the split side of vfft_create's one fork.
 *
 * _vfft_create_inner validates the config, applies the layout-independent
 * refusals, routes 1D real to bridge/real_bridge.h (the D1 crossing), and then
 * forks ONCE on config.layout: SPLIT requests come here, INTERLEAVED ones go
 * to il/il_create.h. From here on nothing tests the layout again; every tier
 * below is a split tier and includes only split/ and common/.
 *
 * The documentation of the per-tier dispatchers that phase 6 left in the front
 * folders follows; it describes the contracts these tiers keep.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: the tiers call file-scope statics of
 * vfft.c (_pad_ladder, _build_2d, _vw2_persist, ...), so this is included after
 * those and before _vfft_create_inner.
 */
#ifndef VFFT_SPLIT_CREATE_H
#define VFFT_SPLIT_CREATE_H

#include "split/rank3/fftnd_create_split.h"
#include "split/rank2/fft2d_create_split.h"
#include "split/rank1/c2c_ip_create_split.h"
#include "split/rank1/c2c_oop_create_split.h"
#include "split/trig/trig_create.h"

/* ════════════════════════════════════════════════════════════════════
 * RANK 3 / 4  (was transforms/fftnd/fftnd_create.h)
 * ════════════════════════════════════════════════════════════════════
 *
 *  fftnd_create.h — the rank-3 and rank-4 CREATE tiers.
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
 *
 */

/* ════════════════════════════════════════════════════════════════════
 * RANK 2  (was transforms/fft2d/fft2d_create.h)
 * ════════════════════════════════════════════════════════════════════
 *
 *  fft2d_create.h — the rank-2 CREATE tier.
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
 *
 */

/* ════════════════════════════════════════════════════════════════════
 * c2c IN-PLACE  (was oop/c2c_ip_create.h)
 * ════════════════════════════════════════════════════════════════════
 *
 *  c2c_ip_create.h — the c2c IN-PLACE create tier.
 *
 * The front dispatcher of the in-place c2c create. It forks on layout once:
 *
 *   - interleaved, no caller-supplied batch -> il/rank1/c2c_ip_create_il.h
 *   - everything else                       -> split/rank1/c2c_ip_create_split.h
 *
 * The split tier holds the CALLER-SUPPLIED BATCH arm (an owned-batch
 * descriptor was handed in, so the plan serves that exact handle) and the
 * general split arms. IL + batch is refused upstream, so it never gets here.
 *
 * WHY THE HANDLE IS CHECKED EXACTLY
 * ---------------------------------
 * Arm 1 refuses unless xform, oop, K and N all match. The shapes are genuinely
 * incompatible rather than merely different: an r2c handle's re/im planes are
 * (N/2+1)*Kp, and an OOP handle is 4-plane. A mismatched handle is a caller
 * error and is refused LOUDLY, per the tree's diagnostic directive.
 *
 * PADDED vs UNPADDED IS A WISDOM QUESTION, NOT A BRANCH
 * ----------------------------------------------------
 * Kp is the batch's padded lane count. The padded verdict is not a separate
 * planning path: it is the (N,K) entry's exec_me, and the pad plan IS the
 * aligned (N,Kp) entry -- both ordinary c2c cells in the one unified store.
 * `misaligned = (Kp != K)` selects between them; nothing here invents a cutoff.
 *
 * 🔴 te/ae are re-looked-up after every set: `wisdom_set` may realloc, so a
 * pointer held across a set is a dangling read.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. It calls file-scope statics that live in vfft.c, so
 * it must be included after those are defined and before _vfft_create_inner.
 *
 */

/* ════════════════════════════════════════════════════════════════════
 * c2c OUT-OF-PLACE  (was oop/c2c_oop_create.h)
 * ════════════════════════════════════════════════════════════════════
 *
 *  c2c_oop_create.h — the c2c OUT-OF-PLACE create tier.
 *
 * The c2c out-of-place arm of _vfft_create_inner; it returns on every path.
 *
 * OOP IS NOT IN-PLACE WITH A COPY BOLTED ON
 * -----------------------------------------
 * It is a separate serving with its own wisdom family (W->oop / vw2 oop
 * records) and its own K=1 route. That is why this tier is a sibling of
 * c2c_ip_create.h rather than a branch inside it: the two consult different
 * banked verdicts and build different plans.
 *
 * THE K=1 CASE
 * ------------
 * `K == 1 && !ob` is the K=1 tier: the interleaved routes (mono, pair,
 * chain3, flat, ZTURN-T, four-step, prime) replay the banked kind-3 row or
 * race and bank it (_k1_il_plan_race); the split routes replay their own
 * lay=split row. The tier does not choose an engine by rule.
 *
 * `ob` splits the same way it does in-place: a caller-supplied batch handle is
 * checked and served exactly, otherwise the plan owns its buffers.
 *
 * WISDOM, NOT HEURISTIC
 * ---------------------
 * Every interleaved choice here is replayed from a banked verdict or raced
 * and then banked. The split side's uncalibrated default is structural until
 * calibrate_k1_split.c banks the cell.
 *
 * POSITION IN vfft.c IS LOAD-BEARING
 * ----------------------------------
 * Not a standalone header. It calls file-scope statics that live in vfft.c, so
 * it must be included after those are defined and before _vfft_create_inner.
 *
 */

static vfft_plan _vfft_split_create(const vfft_config_t *cfg,
                                    vfft_batch ob,
                                    struct vfft_wisdom_s *W,
                                    const vfft_proto_registry_t *reg,
                                    int N,
                                    size_t K)
{
    if (cfg->dims == 3 || cfg->dims == 4)
        return _vfft_create_rank34_split(cfg, W, reg, K);
    if (cfg->dims == 2)
    {
        /* the 2D executors are K-blind; howmany > 1 is the PLANE QUEUE, an
         * interleaved tier (il/rank2/fft2d_create_il.h) */
        if (K != 1)
        {
            _vfft_warn("vfft_create: dims=2 howmany=%zu is served by "
                       "the plane queue for INTERLEAVED C2C/R2C/C2R "
                       "only (got %s, layout=%d) — batch other 2D "
                       "plans sequentially",
                       K, _vfft_tname(cfg->transform),
                       (int)cfg->layout);
            return NULL;
        }
        return _vfft_create_2d_split(cfg, W, reg, K);
    }
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE)
        return _vfft_create_c2c_ip_split(cfg, ob, W, reg, N, K);
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_OUTOFPLACE)
        return _vfft_create_c2c_oop_split(cfg, ob, W, reg, N, K);
    if (_VFFT_IS_TRIG(cfg->transform))
        return _vfft_create_trig(cfg, ob, W, reg, N, K);
    return NULL; /* unreachable: 1D real went to the bridge before the fork,
                  * and every other transform is dispatched above */
}

#endif /* VFFT_SPLIT_CREATE_H */
