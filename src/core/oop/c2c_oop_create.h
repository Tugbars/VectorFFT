/* c2c_oop_create.h — the c2c OUT-OF-PLACE create tier.
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
 */
#ifndef VFFT_OOP_C2C_OOP_CREATE_H
#define VFFT_OOP_C2C_OOP_CREATE_H

#include "c2c_oop_create_il.h"
#include "c2c_oop_create_split.h"

static vfft_plan _vfft_create_c2c_oop(const vfft_config_t *cfg,
                                      vfft_batch ob,
                                      struct vfft_wisdom_s *W,
                                      const vfft_proto_registry_t *reg,
                                      int N,
                                      size_t K)
{
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_OUTOFPLACE)
    {
        /* the one layout fork. config.batch + INTERLEAVED is refused
         * upstream, so the IL tier never sees an owned batch. */
        if (cfg->layout == VFFT_LAYOUT_INTERLEAVED)
            return _vfft_create_c2c_oop_il(cfg, W, N, K);
        return _vfft_create_c2c_oop_split(cfg, ob, W, reg, N, K);
    }
    return NULL; /* unreachable: the one call site guards on the same
                  * condition, and every path in the block above returns. */
}

#endif /* VFFT_OOP_C2C_OOP_CREATE_H */
