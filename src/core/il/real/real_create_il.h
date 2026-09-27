/* real_create_il.h — the r2c / c2r CREATE, interleaved tier.
 *
 * The zr2c route (il/real/zr2c_build.h): even N, K==1, INTERLEAVED, no
 * caller-supplied batch — the CCE plane reinterpreted as z[N/2], a c2c child,
 * and the Hermitian fold. The same route serves r2c and c2r (the c2r twin
 * folds first). The front dispatcher (transforms/real/real_create.h) holds
 * the gate and calls this only for a matching request.
 *
 * Returns the handle; or NULL with *refused=0 to let an out-of-place request
 * fall through to the split real engines (the D1 bridge, phase 7); or NULL
 * with *refused=1 when the request is in place and zr2c could not be built.
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
    struct vfft_plan_s *hz = _zr2c_build(cfg, N, W);
    *refused = 0;
    if (hz)
        return hz; /* zr2c serving: banks its own kind-5 cell */
    /* 🔴 NO SILENT DEGRADE TO OUT-OF-PLACE. The in-place refusal in create
     * ADMITTED this shape, so falling through would stamp
     * h->placement = INPLACE onto a handle whose executor is the OOP
     * CCE path -- engines that stream an N-double real plane into an
     * N+2-double CCE plane and were never gated for aliasing. The
     * caller then makes the documented (z,NULL,z,NULL) call and gets
     * an out-of-place executor whose source aliases its destination.
     * zr2c is the ONLY in-place real path, so if it could not be
     * built there is no in-place plan to give: refuse loudly.
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
