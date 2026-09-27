/* c2c_ip_create.h — the c2c IN-PLACE create tier.
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
 */
#ifndef VFFT_OOP_C2C_IP_CREATE_H
#define VFFT_OOP_C2C_IP_CREATE_H
#include "c2c_ip_create_il.h"
#include "c2c_ip_create_split.h"

static vfft_plan _vfft_create_c2c_ip(const vfft_config_t *cfg,
                                     vfft_batch ob,
                                     struct vfft_wisdom_s *W,
                                     const vfft_proto_registry_t *reg,
                                     int N,
                                     size_t K)
{
    if (cfg->transform == VFFT_C2C && cfg->placement == VFFT_INPLACE &&
        cfg->layout == VFFT_LAYOUT_INTERLEAVED && !ob)
        return _c2c_ip_create_il(cfg, W, N, K);   /* the IL tier's own create */
    return _vfft_create_c2c_ip_split(cfg, ob, W, reg, N, K);
}

#endif /* VFFT_OOP_C2C_IP_CREATE_H */
