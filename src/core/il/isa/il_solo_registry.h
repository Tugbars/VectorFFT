/* il_solo_registry.h - the interleaved K=1 SOLO tier's kernels and resolvers.
 *
 * The mono-64 8x8 IL twins and the resolvers of the whole-transform solo
 * kernels (n1, the alias-tolerant n1c, the column-stride n1ccs) at the build's
 * ISA. Carved verbatim out of oop/oop_leaf_registry.h (layout separation
 * phase 5): they were the interleaved half of the split OOP leaf registry, so
 * every split OOP consumer pulled in the IL ISA surface through it. */
#ifndef VFFT_IL_SOLO_REGISTRY_H
#define VFFT_IL_SOLO_REGISTRY_H

#include "common/abi/codelet_abi.h"   /* vfft_oop11_fn */
/* mono-64 IL twins: z->z; bwd = fwd DAG with (im,re)-swapped
 * boundary lattices (swap identity), unnormalized inverse, output (re,im).
 * ABI: (in_z, unused, out_z, unused, ...). Split bwd needs NO codelet —
 * call the split fwd with re/im pointer pairs swapped. */
#include "il_isa.h"   /* the IL registry and kernel names at the build's ISA */
#ifndef VFFT_FLOAT   /* the 64-point mono is FP64's: the FP32 corpus has no form 1 */
VFFT_IL_DECL(VFFT_IL_SYM(vfft_k1_mono64_8x8_il_fwd))
VFFT_IL_DECL(VFFT_IL_SYM(vfft_k1_mono64_8x8_il_bwd))
#endif

/* ── MONO tier (sub-128, K=1 interleaved): the SOLO kernels ──
 * A solo kernel is the whole N-point transform in one call: natural order
 * in and out, twiddle-free, the pure-IL n1 kind on the 11-arg leg ABI
 * (one leg: Ls = OLs = 1, count = 1 -> the VEX-128 tail IS the transform).
 * Emitted at every radix the pure-IL family has (VFFT_IL_N1_PAIR_RADICES,
 * 2..64) by `gen_radix R --cil-n1 [--cil-bwd]`; the corpus lists them.
 *
 * FORMS (the planner's mono axis, banked as il_kv on the MONO row):
 *   form 0 = radixN_z_n1      the solo kind (every N in the set)
 *   form 1 = mono64_8x8_il    the fused 8x8 four-step (N = 64 only)
 * Out-of-place serves the __restrict__ n1 kernels; IN-PLACE serves the
 * alias-tolerant n1c twins (vfft_k1_mono_ilc_fn), same math, no restrict,
 * so z -> z is legal by construction (n1c exists at every N in the set). */
static inline int vfft_k1_mono_il_nforms(int N)
{
    switch (N)
    {
#define C(R) case R:
    VFFT_IL_N1_PAIR_RADICES(C)
#undef C
#ifdef VFFT_FLOAT
        return 1;
#else
        return N == 64 ? 2 : 1;
#endif
    default: return 0;
    }
}

static inline vfft_oop11_fn vfft_k1_mono_il_form_fn(int N, int form, int bwd)
{
#ifndef VFFT_FLOAT
    if (form == 1)
        return N == 64 ? (bwd ? VFFT_IL_SYM(vfft_k1_mono64_8x8_il_bwd)
                              : VFFT_IL_SYM(vfft_k1_mono64_8x8_il_fwd))
                       : 0;
#endif
    if (form != 0) return 0;
    switch (N)
    {
#define C(R) case R: return bwd ? VFFT_IL_SYM(radix##R##_z_n1_bwd) : VFFT_IL_SYM(radix##R##_z_n1_fwd);
    VFFT_IL_N1_PAIR_RADICES(C)
#undef C
    default: return 0;
    }
}

/* form 0 — the existence probe every route table and the planner use */
static inline vfft_oop11_fn vfft_k1_mono_il_fn(int N, int bwd)
{
    return vfft_k1_mono_il_form_fn(N, 0, bwd);
}

/* the IN-PLACE solo: alias-tolerant n1c, both directions */
static inline vfft_oop11_fn vfft_k1_mono_ilc_fn(int N, int bwd)
{
    switch (N)
    {
#define C(R) case R: return bwd ? VFFT_IL_SYM(radix##R##_z_n1c_bwd) : VFFT_IL_SYM(radix##R##_z_n1c_fwd);
    VFFT_IL_N1C_PAIR_RADICES(C)
#undef C
    default: return 0;
    }
}

/* the BATCHED solo: n1ccs = the n1c leaf with column-stride
 * addressing -- lane k is ONE WHOLE R-point transform at pitch Gs (a row of
 * a plane, a transform of a batch), two per vector through loadu2/storeu2
 * pairs, in place, natural order, both directions. Call:
 *   fn(z, NULL, z, NULL, NULL, NULL, 1, pitch, 1, pitch, count)
 * (Ls = OLs = 1: the legs of one transform are contiguous). NULL where the
 * corpus has no pair at R -- the registry's PAIR list is the resolver. */
static inline vfft_oop11_fn vfft_il_n1ccs_fn(int R, int bwd)
{
    switch (R)
    {
#define C(r) case r: return bwd ? VFFT_IL_SYM(radix##r##_z_n1ccs_bwd) : VFFT_IL_SYM(radix##r##_z_n1ccs_fwd);
    VFFT_IL_N1CCS_PAIR_RADICES(C)
#undef C
    default: return 0;
    }
}

/* ── THE ROW FAMILY (2026-11-08): the batched row kernel of the 2D tier's PLAIN
 * row passes (the batched rows route and the skewed pass's rows). Per radix the
 * corpus holds the winner of the 2026-11-08 pinned race at 256 bits:
 *   row       one row per iteration, a lane is a POINT of the row (stage 1 across
 *             the registers, one register transpose per block, stage 2 across the
 *             block; MKL's row form), rowb<m1><p1> = stage 1 split through S[];
 *   rowswap   lanes = rows like n1ccs, but whole-vector loads and 128-bit-lane
 *             swaps instead of the lane gather;
 *   n1ccs     the lane-load construction, where it won.
 * The folders rows/row, rows/rowswap, rows/n1ccs; the generated
 * VFFT_IL_ROW_CHOICE lists the pair per radix. Same call as n1ccs:
 *   fn(z, NULL, z, NULL, NULL, NULL, 1, pitch, 1, pitch, rows)
 * with PLAIN row stores: row and rowswap take OLs = 1 (rowswap: Ls = OLs = 1),
 * so the TURN route keeps vfft_il_n1ccs_fn for its turned stores (OLs = the
 * scratch pitch, OGs = 1). NULL where the radix has no row kernel. */
static inline vfft_oop11_fn vfft_il_row_fn(int R, int bwd)
{
    switch (R)
    {
#define C(r, tag) case r: return bwd ? VFFT_IL_SYM(radix##r##_z_##tag##_bwd) : VFFT_IL_SYM(radix##r##_z_##tag##_fwd);
    VFFT_IL_ROW_CHOICE(C)
#undef C
    default: return 0;
    }
}

/* 1 = the radix's row kernel is a SPLIT form (the tag rowb<m1><p1>: stage 1
 * through S[], the kernel holding a fraction of the row at a time), 0 = a
 * whole-row form (row, rowswap, n1ccs) or no kernel. The 2D tier's row-engine
 * admission (vfft_policy_il2d_row_fixed) reads it: a whole-row form past half
 * the register file (16 on AVX2, 64 on AVX-512) keeps its race. The tag grammar is the
 * generator's (c2c_il.ml emit_row: "swap"? ^ "row" ^ "b<m1><p1>"? ^ "t"?). */
static inline int vfft_il_row_split(int R)
{
    switch (R)
    {
#define C(r, tag) case r: return (#tag)[0] == 'r' && (#tag)[3] == 'b';
    VFFT_IL_ROW_CHOICE(C)
#undef C
    default: return 0;
    }
}

/* The TURN route's row kernel (2026-11-08): the same families with TURNED
 * stores -- point l of row k at zout[2*(l*OLs + k*OGs)], OLs = the point pitch
 * of the turn's scratch, OGs = 1 (the symbols end in t: rowswapt, rowb44t,
 * rowb48t). Per radix the corpus holds the winner of the turn-role race, else
 * n1ccs, whose store honours those pitches as it is. Call:
 *   fn(z, NULL, scratch, NULL, NULL, NULL, 1, pitch, P, 1, rows) */
static inline vfft_oop11_fn vfft_il_rowt_fn(int R, int bwd)
{
    switch (R)
    {
#define C(r, tag) case r: return bwd ? VFFT_IL_SYM(radix##r##_z_##tag##_bwd) : VFFT_IL_SYM(radix##r##_z_##tag##_fwd);
    VFFT_IL_ROWT_CHOICE(C)
#undef C
    default: return 0;
    }
}

#endif /* VFFT_IL_SOLO_REGISTRY_H */
