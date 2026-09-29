/* real_create_il.h — the r2c / c2r CREATE, interleaved tier.
 *
 * Four engines serve an even N, K==1, INTERLEAVED request with no
 * caller-supplied batch (il/real/zrp_build.h, the door's engine pick):
 *   zr2c  the CCE plane reinterpreted as z[N/2], a c2c child, and the
 *         Hermitian fold (zr2c_build.h); the same route serves r2c and c2r
 *         (the c2r twin folds first);
 *   zrp   the real pair: the stock n1t leaf over the packed view and the
 *         t2h Hermitian top stage, no fold pass (zrp.h), its c2r the mirror;
 *   zttr  ZTT-r: the ZTT at N/2 with the fold fused into the terminator or
 *         the ingest (zttr.h);
 *   zrm   the real mono: the whole transform as one rn1 kernel, N <= 64
 *         (zrm.h; odd N races it in the bridge, not here).
 * The cell's engine is banked in the real shard (wisdom2_real_il.h); a miss
 * races them. The real bridge (bridge/real_bridge.h) holds the gate and
 * calls this only for a matching request.
 *
 * Returns the handle; or NULL with *refused=0 to let an out-of-place request
 * fall through to the split real engines (the D1 bridge, phase 7); or NULL
 * with *refused=1 when the request is in place and no engine could be built.
 *
 * It runs BEFORE the split-path calibrate-on-miss blocks on purpose: a
 * zr2c-served cell must not pay for (or bank) c2c(N/2, K)/rfft rows it never
 * reads — the child rides the K=1 engine tables through its own recursive
 * create.
 *
 * POSITION IN vfft.c IS LOAD-BEARING: calls file-scope statics of vfft.c, so
 * it is included (via the dispatcher) after those are defined.
 */
#ifndef VFFT_IL_REAL_CREATE_IL_H
#define VFFT_IL_REAL_CREATE_IL_H

static struct vfft_plan_s *_vfft_create_real_il(const vfft_config_t *cfg,
                                                struct vfft_wisdom_s *W,
                                                int N, int *refused)
{
    struct vfft_plan_s *hz = _real_il_build(cfg, N, W);
    *refused = 0;
    if (hz)
        return hz; /* the banked engine, or the race's winner: banks its own cell */
    /* 🔴 NO SILENT DEGRADE TO OUT-OF-PLACE. The in-place refusal in create
     * ADMITTED this shape, so falling through would stamp
     * h->placement = INPLACE onto a handle whose executor is the OOP
     * CCE path -- engines that stream an N-double real plane into an
     * N+2-double CCE plane and were never gated for aliasing. The
     * caller then makes the documented (z,NULL,z,NULL) call and gets
     * an out-of-place executor whose source aliases its destination.
     * The interleaved engines are the ONLY in-place real path, so if none
     * could be built there is no in-place plan to give: refuse loudly.
     * Out-of-place callers keep the fall-through unchanged. */
    if (cfg->placement == VFFT_INPLACE)
    {
        _vfft_warn("vfft_create: in-place %s N=%d could not build the zr2c route "
                   "(the only in-place real path); no out-of-place fallback exists "
                   "for an in-place plan -- use VFFT_OUTOFPLACE",
                   _vfft_tname(cfg->transform), N);
        *refused = 1;
    }
    return NULL;
}

#endif /* VFFT_IL_REAL_CREATE_IL_H */
