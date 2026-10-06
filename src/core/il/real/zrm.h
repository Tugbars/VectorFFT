/* zrm.h - the real MONO engine: the whole small real transform as ONE kernel.
 *
 * The rn1 kind (codelets/zil/<isa>/real/mono/, generator/lib/gen/c2c_il.ml,
 * `gen_radix N --cil-rn1 [--cil-bwd]`): the n1 body on real input. The
 * forward loads N reals (a lane is (x, 0)), computes the N-point DFT and
 * stores bins 0..N/2 only (the CCE half plane the r2c contract wants); the
 * backward loads bins 0..N/2, forms bin N-l as the conjugate of bin l (a
 * sign flip, no load) and stores N real lanes (the unnormalised c2r). One
 * call is the transform: count = 1 runs the kernel's VEX-128 tail with
 * Ls = OLs = 1 -- no child, no fold, no table, no scratch. N = 3..64 at the
 * n1 radices (VFFT_IL_RN1_PAIR_RADICES).
 *
 * AT 32 AND 64 THE BODY IS BLOCKED (2026-10-06): rn1b48 (4x8) and rn1b444
 * (4x4x4, three passes) -- the c2c blocked construction on the mono's edges,
 * the sub-DFTs parked in a function-scope S[], the remainder arm (count = 1:
 * this engine's call) the blocked passes at VEX-128. The kernel race: 7-14%
 * over the monolithic bodies at count 1 and 2, both directions, spills 443
 * -> 226 (32) and 1276 -> 761 (64); the monolithic 32/64 bodies are deleted
 * (the law of il2p.h: r32/r64 never monolithic).
 *
 * In place is legal: the kind is alias-tolerant (no __restrict__; every load
 * of a loop body precedes every store), so zin == zout reads the N reals and
 * writes the N+2 doubles over them (gated in gauntlet/rn1_gate.c).
 *
 * The real door (il/real/zrp_build.h) races it at N <= VFFT_ZRM_MAX_N
 * against zr2c (even N) and, in bridge/real_bridge.h, against the odd-real
 * routes (odd N), and banks eng=zrm in the real shard (wisdom2_real_il.h).
 * The plan carries the kernel pointer (vfft_plan_s.zrm) and the bound
 * execute (il/il_execute.h) calls it with nothing in between.
 */
#ifndef VFFT_ZRM_H
#define VFFT_ZRM_H

#include <stddef.h>

#include "common/abi/codelet_abi.h" /* vfft_oop11_fn */
#include "il_isa.h"                 /* the IL registry at the build's ISA */

#define VFFT_ZRM_MAX_N 64

/* the kernel serving N in one direction, or 0 when the kind has no such radix */
static inline vfft_oop11_fn vfft_zrm_fn(int N, int bwd)
{
    switch (N)
    {
#ifdef VFFT_IL_RN1_PAIR_RADICES
#define C(R) case R: return bwd ? VFFT_IL_SYM(radix##R##_z_rn1_bwd) : VFFT_IL_SYM(radix##R##_z_rn1_fwd);
    VFFT_IL_RN1_PAIR_RADICES(C)
#undef C
#endif
#ifdef VFFT_IL_RN1B48_PAIR_RADICES
    case 32: return bwd ? VFFT_IL_SYM(radix32_z_rn1b48_bwd) : VFFT_IL_SYM(radix32_z_rn1b48_fwd);   /* the blocked 4x8 body */
#endif
#ifdef VFFT_IL_RN1B444_PAIR_RADICES
    case 64: return bwd ? VFFT_IL_SYM(radix64_z_rn1b444_bwd) : VFFT_IL_SYM(radix64_z_rn1b444_fwd); /* the blocked 4x4x4 body */
#endif
    default: return 0;
    }
}

/* the whole transform: the K=1 solo call (Ls = OLs = 1, count = 1) */
static inline void vfft_zrm_execute(vfft_oop11_fn fn, const double *in, double *out)
{
    fn(in, NULL, out, NULL, NULL, NULL, 1, 0, 1, 0, 1);
}

#endif /* VFFT_ZRM_H */
