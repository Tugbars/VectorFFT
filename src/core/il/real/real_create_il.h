/* real_create_il.h — the r2c / c2r CREATE, interleaved tier.
 *
 * K==1, INTERLEAVED, no caller-supplied batch. The door's engine pick is by
 * the parity of N:
 *   even  il/real/zrp_build.h: zr2c (the CCE plane reinterpreted as z[N/2],
 *         a c2c child, and the Hermitian fold), zrp (the real pair), zttr
 *         (ZTT-r), zfsr (the real four-step) and zrm (the real mono);
 *   odd   il/real/odd_build.h: zrm (the real mono), zrf (the real flat DIT)
 *         and zrb (the real Bluestein), both placements.
 * The cell's engine is banked in the real shard (wisdom2_real_il.h); a miss
 * races. The real bridge (bridge/real_bridge.h) holds the gate and calls
 * this only for a matching request.
 *
 * Returns the handle; or NULL with *refused=0 to let an EVEN out-of-place
 * request fall through to the split real engines (the D1 bridge, phase 7);
 * or NULL with *refused=1 when no engine could be built and nothing else may
 * serve: an in-place request (the interleaved engines are the only in-place
 * real path), or any odd request (an odd cell this door admits is served by
 * its engines or refused -- never by the other library's).
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
    struct vfft_plan_s *hz = (N & 1) ? _real_il_odd_build(cfg, N, W) : _real_il_build(cfg, N, W);
    *refused = 0;
    if (hz)
        return hz; /* the banked engine, or the race's winner: banks its own cell */
    if (N & 1)
    {
        _vfft_warn("vfft_create: %s odd N=%d %s: no IL real engine built at this cell "
                   "(every arm failed to build or was gated out); unsupported",
                   _vfft_tname(cfg->transform), N,
                   cfg->placement == VFFT_INPLACE ? "in place" : "out of place");
        *refused = 1;
        return NULL;
    }
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
