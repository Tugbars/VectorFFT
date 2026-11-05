/* zts_stages.h — zturn-ts's STAGE KERNELS, for the staged executor: the
 * cells with no fused driver (every 2^a*odd cell of the odd band,
 * zts.h vfft_zts_odd_band) run one kernel call per stage and block. The
 * kernels are codelets/split/ztt/<isa> (gen_set quadrant ztt-<isa>): the S
 * family (zts: split planes at the edges, ZTURN-T's block plane inside) and
 * the V family (ztv: lane-major, the blocked edges). The ends are radix 4 or
 * 8 (the ingest's turn lattice, the terminator's lane transposes); the mids
 * are radix-agnostic and exist at 3, 4, 5, 7, 8, 9 and 15. A radix without
 * a kernel resolves to NULL and the create refuses the chain.
 *
 * The exported ABIs (cascade_z.ml, the "ts" wrappers):
 *   S ingest      (in_re, in_im, zout, tw, rb, Ls, Gs, OLs, count)
 *   S mid         (zin, zout, tw, rb, Ls, Gs, OLs, count)            in place, Gs groups
 *   S terminator  (zin, out_re, out_im, tw, rb, Ls, Gs, OLs, count)  tlfi: the plane form
 *   V ingest      (in_re, in_im, out_re, out_im, tw, rb, Ls, Gs, OLs, count, ep, nb, gp)
 *   V mid         (in_re, in_im, out_re, out_im, tw, rb, Ls, Gs, OLs, count)
 *   V terminator  (in_re, in_im, out_re, out_im, tw, rb, Ls, Gs, OLs, count, ep, nb, gp) */
#ifndef VFFT_ZTS_STAGES_H
#define VFFT_ZTS_STAGES_H

#include <stddef.h>

typedef void (*vfft_zts_s0_fn)(const double *, const double *, double *, const double *, const size_t *,
                               size_t, size_t, size_t, size_t);
typedef void (*vfft_zts_sm_fn)(const double *, double *, const double *, const size_t *,
                               size_t, size_t, size_t, size_t);
typedef void (*vfft_zts_sl_fn)(const double *, double *, double *, const double *, const size_t *,
                               size_t, size_t, size_t, size_t);
typedef void (*vfft_zts_v0_fn)(const double *, const double *, double *, double *, const double *, const size_t *,
                               size_t, size_t, size_t, size_t, size_t, size_t, size_t);
typedef void (*vfft_zts_vm_fn)(const double *, const double *, double *, double *, const double *, const size_t *,
                               size_t, size_t, size_t, size_t);
typedef void (*vfft_zts_vl_fn)(const double *, const double *, double *, double *, const double *, const size_t *,
                               size_t, size_t, size_t, size_t, size_t, size_t, size_t);

#if !defined(VFFT_BUILD_ISA_AVX512)

#define _ZTS_DECL_S0(R) \
    void radix##R##_zts_t0tp_fwd_avx2(const double *, const double *, double *, const double *, const size_t *, size_t, size_t, size_t, size_t); \
    void radix##R##_zts_t0tp_bwd_avx2(const double *, const double *, double *, const double *, const size_t *, size_t, size_t, size_t, size_t);
#define _ZTS_DECL_SM(R) \
    void radix##R##_zts_tmg_fwd_avx2(const double *, double *, const double *, const size_t *, size_t, size_t, size_t, size_t); \
    void radix##R##_zts_tmg_bwd_avx2(const double *, double *, const double *, const size_t *, size_t, size_t, size_t, size_t);
#define _ZTS_DECL_SL(R) \
    void radix##R##_zts_tlfi_fwd_avx2(const double *, double *, double *, const double *, const size_t *, size_t, size_t, size_t, size_t); \
    void radix##R##_zts_tlfi_bwd_avx2(const double *, double *, double *, const double *, const size_t *, size_t, size_t, size_t, size_t);
#define _ZTS_DECL_V0(R) \
    void radix##R##_ztv_t0tp_fwd_avx2(const double *, const double *, double *, double *, const double *, const size_t *, size_t, size_t, size_t, size_t, size_t, size_t, size_t); \
    void radix##R##_ztv_t0tp_bwd_avx2(const double *, const double *, double *, double *, const double *, const size_t *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);
#define _ZTS_DECL_VM(R) \
    void radix##R##_ztv_tmg_fwd_avx2(const double *, const double *, double *, double *, const double *, const size_t *, size_t, size_t, size_t, size_t); \
    void radix##R##_ztv_tmg_bwd_avx2(const double *, const double *, double *, double *, const double *, const size_t *, size_t, size_t, size_t, size_t);
#define _ZTS_DECL_VL(R) \
    void radix##R##_ztv_tlfi_fwd_avx2(const double *, const double *, double *, double *, const double *, const size_t *, size_t, size_t, size_t, size_t, size_t, size_t, size_t); \
    void radix##R##_ztv_tlfi_bwd_avx2(const double *, const double *, double *, double *, const double *, const size_t *, size_t, size_t, size_t, size_t, size_t, size_t, size_t);

_ZTS_DECL_S0(4) _ZTS_DECL_S0(8)
_ZTS_DECL_SL(4) _ZTS_DECL_SL(8)
_ZTS_DECL_SM(3) _ZTS_DECL_SM(4) _ZTS_DECL_SM(5) _ZTS_DECL_SM(7) _ZTS_DECL_SM(8) _ZTS_DECL_SM(9) _ZTS_DECL_SM(15)
_ZTS_DECL_V0(4) _ZTS_DECL_V0(8)
_ZTS_DECL_VL(4) _ZTS_DECL_VL(8)
_ZTS_DECL_VM(3) _ZTS_DECL_VM(4) _ZTS_DECL_VM(5) _ZTS_DECL_VM(7) _ZTS_DECL_VM(8) _ZTS_DECL_VM(9) _ZTS_DECL_VM(15)

#define _ZTS_PICK2(R, F, B) case R: return bwd ? B : F;
static inline vfft_zts_s0_fn _zts_s0_fn(int R, int bwd)
{
    switch (R) { _ZTS_PICK2(4, radix4_zts_t0tp_fwd_avx2, radix4_zts_t0tp_bwd_avx2)
                 _ZTS_PICK2(8, radix8_zts_t0tp_fwd_avx2, radix8_zts_t0tp_bwd_avx2) }
    return NULL;
}
static inline vfft_zts_sm_fn _zts_sm_fn(int R, int bwd)
{
    switch (R) { _ZTS_PICK2(3, radix3_zts_tmg_fwd_avx2, radix3_zts_tmg_bwd_avx2)
                 _ZTS_PICK2(4, radix4_zts_tmg_fwd_avx2, radix4_zts_tmg_bwd_avx2)
                 _ZTS_PICK2(5, radix5_zts_tmg_fwd_avx2, radix5_zts_tmg_bwd_avx2)
                 _ZTS_PICK2(7, radix7_zts_tmg_fwd_avx2, radix7_zts_tmg_bwd_avx2)
                 _ZTS_PICK2(8, radix8_zts_tmg_fwd_avx2, radix8_zts_tmg_bwd_avx2)
                 _ZTS_PICK2(9, radix9_zts_tmg_fwd_avx2, radix9_zts_tmg_bwd_avx2)
                 _ZTS_PICK2(15, radix15_zts_tmg_fwd_avx2, radix15_zts_tmg_bwd_avx2) }
    return NULL;
}
static inline vfft_zts_sl_fn _zts_sl_fn(int R, int bwd)
{
    switch (R) { _ZTS_PICK2(4, radix4_zts_tlfi_fwd_avx2, radix4_zts_tlfi_bwd_avx2)
                 _ZTS_PICK2(8, radix8_zts_tlfi_fwd_avx2, radix8_zts_tlfi_bwd_avx2) }
    return NULL;
}
static inline vfft_zts_v0_fn _zts_v0_fn(int R, int bwd)
{
    switch (R) { _ZTS_PICK2(4, radix4_ztv_t0tp_fwd_avx2, radix4_ztv_t0tp_bwd_avx2)
                 _ZTS_PICK2(8, radix8_ztv_t0tp_fwd_avx2, radix8_ztv_t0tp_bwd_avx2) }
    return NULL;
}
static inline vfft_zts_vm_fn _zts_vm_fn(int R, int bwd)
{
    switch (R) { _ZTS_PICK2(3, radix3_ztv_tmg_fwd_avx2, radix3_ztv_tmg_bwd_avx2)
                 _ZTS_PICK2(4, radix4_ztv_tmg_fwd_avx2, radix4_ztv_tmg_bwd_avx2)
                 _ZTS_PICK2(5, radix5_ztv_tmg_fwd_avx2, radix5_ztv_tmg_bwd_avx2)
                 _ZTS_PICK2(7, radix7_ztv_tmg_fwd_avx2, radix7_ztv_tmg_bwd_avx2)
                 _ZTS_PICK2(8, radix8_ztv_tmg_fwd_avx2, radix8_ztv_tmg_bwd_avx2)
                 _ZTS_PICK2(9, radix9_ztv_tmg_fwd_avx2, radix9_ztv_tmg_bwd_avx2)
                 _ZTS_PICK2(15, radix15_ztv_tmg_fwd_avx2, radix15_ztv_tmg_bwd_avx2) }
    return NULL;
}
static inline vfft_zts_vl_fn _zts_vl_fn(int R, int bwd)
{
    switch (R) { _ZTS_PICK2(4, radix4_ztv_tlfi_fwd_avx2, radix4_ztv_tlfi_bwd_avx2)
                 _ZTS_PICK2(8, radix8_ztv_tlfi_fwd_avx2, radix8_ztv_tlfi_bwd_avx2) }
    return NULL;
}
#undef _ZTS_PICK2

#else /* no split ZTURN-T kernels at 8 lanes yet: every staged chain refuses */
static inline vfft_zts_s0_fn _zts_s0_fn(int R, int bwd) { (void)R; (void)bwd; return NULL; }
static inline vfft_zts_sm_fn _zts_sm_fn(int R, int bwd) { (void)R; (void)bwd; return NULL; }
static inline vfft_zts_sl_fn _zts_sl_fn(int R, int bwd) { (void)R; (void)bwd; return NULL; }
static inline vfft_zts_v0_fn _zts_v0_fn(int R, int bwd) { (void)R; (void)bwd; return NULL; }
static inline vfft_zts_vm_fn _zts_vm_fn(int R, int bwd) { (void)R; (void)bwd; return NULL; }
static inline vfft_zts_vl_fn _zts_vl_fn(int R, int bwd) { (void)R; (void)bwd; return NULL; }
#endif

#endif /* VFFT_ZTS_STAGES_H */
