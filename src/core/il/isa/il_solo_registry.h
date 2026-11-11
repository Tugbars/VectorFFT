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
 *   form 1 = mono64_8x8_il    the fused 8x8 four-step (N = 64 only; FP64)
 *   FP32: forms 1.. = the FUSED kernels radixN_z_mono<A>x<M>, the whole
 *             transform in one call (N = 2 .. 1024; below)
 * Out-of-place serves the __restrict__ n1 kernels; IN-PLACE serves the
 * alias-tolerant n1c twins (vfft_k1_mono_ilc_fn), same math, no restrict,
 * so z -> z is legal by construction (n1c exists at every N in the set). */

#ifdef VFFT_FLOAT
/* ── THE FUSED FORMS (FP32) ── radixN_z_mono<A>x<M>: the whole transform in
 * one call, natural order in and out, no table to build (generator:
 * c2c_il.ml emit_k1_f32, emit_k1_f32_small). Two generated lists per ISA,
 * X(N, form, tag), the form number = the corpus's order and what the wisdom
 * banks:
 *   VFFT_IL_MONO_FUSED     the kernel of the form (out of place);
 *   VFFT_IL_MONO_FUSED_IP  the kernel the form runs IN PLACE: the same one
 *                          where it loads everything before it stores (the
 *                          small kernels, and those that hold the transform
 *                          in registers), else its twin with the work plane
 *                          on its own stack (the tag's i) -- the plain
 *                          kernel of that size uses zout as its plane and is
 *                          out of place only. */
static inline vfft_oop11_fn vfft_k1_mono_il_fused_fn(int N, int form, int bwd)
{
#define C(R, F, tag) \
    if (N == R && form == F) return bwd ? VFFT_IL_SYM(radix##R##_z_##tag##_bwd) : VFFT_IL_SYM(radix##R##_z_##tag##_fwd);
    VFFT_IL_MONO_FUSED(C)
#undef C
    (void)N; (void)form; (void)bwd;
    return 0;
}
static inline vfft_oop11_fn vfft_k1_mono_il_fused_ip_fn(int N, int form, int bwd)
{
#define C(R, F, tag) \
    if (N == R && form == F) return bwd ? VFFT_IL_SYM(radix##R##_z_##tag##_bwd) : VFFT_IL_SYM(radix##R##_z_##tag##_fwd);
    VFFT_IL_MONO_FUSED_IP(C)
#undef C
    (void)N; (void)form; (void)bwd;
    return 0;
}
/* the highest fused form at N (0 = none) */
static inline int vfft_k1_mono_il_nfused(int N)
{
    int n = 0;
#define C(R, F, tag) if (N == R && F > n) n = F;
    VFFT_IL_MONO_FUSED(C)
#undef C
    (void)N;
    return n;
}
#endif

static inline int vfft_k1_mono_il_nforms(int N)
{
#ifdef VFFT_FLOAT
    if (vfft_k1_mono_il_nfused(N))
        return 1 + vfft_k1_mono_il_nfused(N);   /* form 0 (the solo n1, where N has one) + the fused forms */
#endif
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
#else
    if (form >= 1)
        return vfft_k1_mono_il_fused_fn(N, form, bwd);
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

/* a form's OWN in-place kernel (the planner's in-place candidates): form 0 =
 * the alias-tolerant n1c; an FP32 fused form, the kernel its
 * VFFT_IL_MONO_FUSED_IP entry names; 0 = the form has none */
static inline vfft_oop11_fn vfft_k1_mono_il_ip_own_fn(int N, int form, int bwd)
{
    if (form == 0) return vfft_k1_mono_ilc_fn(N, bwd);
#ifdef VFFT_FLOAT
    return vfft_k1_mono_il_fused_ip_fn(N, form, bwd);
#else
    return 0;
#endif
}

/* the in-place kernel a banked MONO row resolves to: its form's own, else the
 * one alias-safe solo n1c(N) (where every form resolved before the fused ones) */
static inline vfft_oop11_fn vfft_k1_mono_il_ip_fn(int N, int form, int bwd)
{
    vfft_oop11_fn f = vfft_k1_mono_il_ip_own_fn(N, form, bwd);
    return f ? f : vfft_k1_mono_ilc_fn(N, bwd);
}

/* the kernel that serves a call with ONE buffer on an OUT-OF-PLACE plan bound
 * to bound_fwd (its forward kernel): the in-place kernel of the bound form --
 * an FP32 fused form's own (which may be bound_fwd's pair itself: the caller
 * then needs no second executor), else the alias-tolerant n1c twin of the
 * restrict solo kinds. 0 = none at this length. */
static inline vfft_oop11_fn vfft_k1_mono_il_alias_fn(int N, vfft_oop11_fn bound_fwd, int bwd)
{
#ifdef VFFT_FLOAT
#define C(R, F, tag) \
    if (N == R && bound_fwd == VFFT_IL_SYM(radix##R##_z_##tag##_fwd)) return vfft_k1_mono_il_fused_ip_fn(N, F, bwd);
    VFFT_IL_MONO_FUSED(C)
#undef C
#endif
    (void)bound_fwd;
    return vfft_k1_mono_ilc_fn(N, bwd);
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
