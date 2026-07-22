/* il_execute.h — interleaved-boundary execution for DIT chains.
 *
 * Sibling of oop_execute.h. Stage 0 of a DIT plan is untwiddled in every
 * group, and the backward generic executor runs stages in REVERSE order,
 * so BOTH user-memory boundaries of a DIT plan are n1 codelets:
 *   forward : stage-0 n1_fwd reads the input   -> derived il_in reads z pairs
 *   backward: stage-0 n1_bwd writes the output -> derived il_out writes z pairs
 *
 * The derived codelets come from tools/il_derive.py applied to the ENGINE'S
 * OWN inplace n1 emission (codelets/inplace/rR_n1_{fwd,bwd}.c — the same
 * bodies the registry wrappers call), so stage-0 speed and rounding match
 * production exactly; only the boundary loads/stores carry the lattice.
 * (First iteration derived from the UG_UG oop family; measured 3x slower
 * stage-0 — regalloc is not wired on that path. Wrong donor, replaced.)
 *
 * Bit-identical to (il2sp + generic execution + sp2il) by construction of
 * the derivation. Rejected with -1, mirroring oop_execute.h:
 *   - DIF-oriented plans; override plans;
 *   - plans whose stage 0 has any twiddled group;
 *   - stage-0 radix without a derived IL codelet below.
 */
#ifndef VFFT_IL_EXECUTE_H
#define VFFT_IL_EXECUTE_H

#include "executor_generic.h"
#include "executor.h" /* _vfft_proto_lookup_{fwd,bwd} for baked resume */

typedef void (*vfft_il_n1f_fn)(const double *, double *, double *,
                               const double *, const double *, size_t, size_t);
typedef void (*vfft_il_n1b_fn)(const double *, const double *, double *,
                               const double *, const double *, size_t, size_t);

/* Range-capable jit executor: stages S with start_stage <= S < stop_stage in
 * the direction's natural walk order. Identical to the typedef in
 * jit_runtime.h (C11 permits duplicate identical typedefs); duplicated so
 * this header stays independent of the jit runtime. */
typedef void (*vfft_proto_exec_range_fn)(const stride_plan_t *, double *, double *,
                                         size_t, size_t, int, int);

#define VFFT_IL_DECL(R, ISA) \
    void radix##R##_n1_fwd_##ISA##_il_in(const double *, double *, double *, \
        const double *, const double *, size_t, size_t); \
    void radix##R##_n1_bwd_##ISA##_il_out(const double *, const double *, double *, \
        const double *, const double *, size_t, size_t);

#if defined(__AVX512F__) && !defined(VFFT_IL_FORCE_AVX2)
VFFT_IL_DECL(2, avx512)  VFFT_IL_DECL(4, avx512)  VFFT_IL_DECL(8, avx512)
VFFT_IL_DECL(16, avx512) VFFT_IL_DECL(32, avx512) VFFT_IL_DECL(64, avx512)
#define VFFT_IL_F(R) radix##R##_n1_fwd_avx512_il_in
#define VFFT_IL_B(R) radix##R##_n1_bwd_avx512_il_out
#else
VFFT_IL_DECL(2, avx2)  VFFT_IL_DECL(4, avx2)  VFFT_IL_DECL(8, avx2)
VFFT_IL_DECL(16, avx2) VFFT_IL_DECL(32, avx2) VFFT_IL_DECL(64, avx2)
#define VFFT_IL_F(R) radix##R##_n1_fwd_avx2_il_in
#define VFFT_IL_B(R) radix##R##_n1_bwd_avx2_il_out
#endif

#ifndef VFFT_IL_POW2_ONLY
/* extended (non-pow2) radices, BOTH ISAs — hoisted out of the avx2 branch
 * so avx512 full-config builds see the decls too (pre-6a16 latent bug). */
VFFT_IL_DECL(3, avx2)   VFFT_IL_DECL(3, avx512)
VFFT_IL_DECL(5, avx2)   VFFT_IL_DECL(5, avx512)
VFFT_IL_DECL(6, avx2)   VFFT_IL_DECL(6, avx512)
VFFT_IL_DECL(7, avx2)   VFFT_IL_DECL(7, avx512)
VFFT_IL_DECL(10, avx2)  VFFT_IL_DECL(10, avx512)
VFFT_IL_DECL(11, avx2)  VFFT_IL_DECL(11, avx512)
VFFT_IL_DECL(12, avx2)  VFFT_IL_DECL(12, avx512)
VFFT_IL_DECL(13, avx2)  VFFT_IL_DECL(13, avx512)
VFFT_IL_DECL(17, avx2)  VFFT_IL_DECL(17, avx512)
VFFT_IL_DECL(19, avx2)  VFFT_IL_DECL(19, avx512)
VFFT_IL_DECL(20, avx2)  VFFT_IL_DECL(20, avx512)
VFFT_IL_DECL(25, avx2)  VFFT_IL_DECL(25, avx512)
#endif /* !VFFT_IL_POW2_ONLY */

static inline vfft_il_n1f_fn _vfft_il_s0_fwd(int r)
{
    switch (r) {
    case 2:  return VFFT_IL_F(2);
    case 4:  return VFFT_IL_F(4);
    case 8:  return VFFT_IL_F(8);
    case 16: return VFFT_IL_F(16);
    case 32: return VFFT_IL_F(32);
    case 64: return VFFT_IL_F(64);
#ifndef VFFT_IL_POW2_ONLY
    case 3: return VFFT_IL_F(3);
    case 5: return VFFT_IL_F(5);
    case 6: return VFFT_IL_F(6);
    case 7: return VFFT_IL_F(7);
    case 10: return VFFT_IL_F(10);
    case 11: return VFFT_IL_F(11);
    case 12: return VFFT_IL_F(12);
    case 13: return VFFT_IL_F(13);
    case 17: return VFFT_IL_F(17);
    case 19: return VFFT_IL_F(19);
    case 20: return VFFT_IL_F(20);
    case 25: return VFFT_IL_F(25);
#endif
    default: return 0;
    }
}
static inline vfft_il_n1b_fn _vfft_il_s0_bwd(int r)
{
    switch (r) {
    case 2:  return VFFT_IL_B(2);
    case 4:  return VFFT_IL_B(4);
    case 8:  return VFFT_IL_B(8);
    case 16: return VFFT_IL_B(16);
    case 32: return VFFT_IL_B(32);
    case 64: return VFFT_IL_B(64);
#ifndef VFFT_IL_POW2_ONLY
    case 3: return VFFT_IL_B(3);
    case 5: return VFFT_IL_B(5);
    case 6: return VFFT_IL_B(6);
    case 7: return VFFT_IL_B(7);
    case 10: return VFFT_IL_B(10);
    case 11: return VFFT_IL_B(11);
    case 12: return VFFT_IL_B(12);
    case 13: return VFFT_IL_B(13);
    case 17: return VFFT_IL_B(17);
    case 19: return VFFT_IL_B(19);
    case 20: return VFFT_IL_B(20);
    case 25: return VFFT_IL_B(25);
#endif
    default: return 0;
    }
}

/* ════════════════════════════════════════════════════════════════════════
 * 6a16 extension — variant-aware boundary folding.
 * Adds: DIT forward EXIT fold (t1s/t1/t1_log3 il_out by stage variant),
 * DIF forward ENTRY fold (t1_dif[_log3] il_in) + EXIT fold (n1 il_out),
 * DIF backward EXIT fold (t1_dif_bwd il_out), and backward ENTRY folds
 * (n1_bwd il_in + conj(cf_all) fixup on split — fixup order unchanged).
 * All range executors are local copies of executor_generic loops with
 * bounds; nothing outside this header is modified. Uncovered radices
 * resolve to 0 -> the wrapper returns -1 and the caller falls back.
 * ════════════════════════════════════════════════════════════════════ */
#define VFFT_IL_DECL_TWO(R, ISA) \
    void radix##R##_t1s_dit_fwd_##ISA##_il_out(const double *, const double *, double *, \
        const double *, const double *, size_t, size_t); \
    void radix##R##_t1_dit_fwd_##ISA##_il_out(const double *, const double *, double *, \
        const double *, const double *, size_t, size_t); \
    void radix##R##_t1_dit_log3_fwd_##ISA##_il_out(const double *, const double *, double *, \
        const double *, const double *, size_t, size_t); \
    void radix##R##_t1_dif_fwd_##ISA##_il_in(const double *, double *, double *, \
        const double *, const double *, size_t, size_t); \
    void radix##R##_t1_dif_log3_fwd_##ISA##_il_in(const double *, double *, double *, \
        const double *, const double *, size_t, size_t); \
    void radix##R##_t1_dif_bwd_##ISA##_il_out(const double *, const double *, double *, \
        const double *, const double *, size_t, size_t); \
    void radix##R##_t1_dif_log3_bwd_##ISA##_il_out(const double *, const double *, double *, \
        const double *, const double *, size_t, size_t);
#define VFFT_IL_DECL_T1SB(R, ISA) \
    void radix##R##_t1s_dit_bwd_##ISA##_il_in(const double *, double *, double *, \
        const double *, const double *, size_t, size_t);
#define VFFT_IL_DECL_N1X(R, ISA) \
    void radix##R##_n1_fwd_##ISA##_il_out(const double *, const double *, double *, \
        const double *, const double *, size_t, size_t); \
    void radix##R##_n1_bwd_##ISA##_il_in(const double *, double *, double *, \
        const double *, const double *, size_t, size_t);

#if defined(__AVX512F__) && !defined(VFFT_IL_FORCE_AVX2)
VFFT_IL_DECL_TWO(4, avx512)  VFFT_IL_DECL_TWO(8, avx512)
VFFT_IL_DECL_TWO(16, avx512) VFFT_IL_DECL_TWO(32, avx512)
VFFT_IL_DECL_T1SB(2, avx512)  VFFT_IL_DECL_T1SB(4, avx512)  VFFT_IL_DECL_T1SB(8, avx512)
VFFT_IL_DECL_T1SB(16, avx512) VFFT_IL_DECL_T1SB(32, avx512) VFFT_IL_DECL_T1SB(64, avx512)
VFFT_IL_DECL_N1X(2, avx512)  VFFT_IL_DECL_N1X(4, avx512)  VFFT_IL_DECL_N1X(8, avx512)
VFFT_IL_DECL_N1X(16, avx512) VFFT_IL_DECL_N1X(32, avx512) VFFT_IL_DECL_N1X(64, avx512)
#ifndef VFFT_IL_POW2_ONLY
VFFT_IL_DECL_TWO(5, avx512)  VFFT_IL_DECL_TWO(10, avx512)
VFFT_IL_DECL_TWO(20, avx512) VFFT_IL_DECL_TWO(25, avx512)
VFFT_IL_DECL_T1SB(5, avx512)  VFFT_IL_DECL_T1SB(10, avx512)
VFFT_IL_DECL_T1SB(20, avx512) VFFT_IL_DECL_T1SB(25, avx512)
VFFT_IL_DECL_N1X(3, avx512)  VFFT_IL_DECL_N1X(5, avx512)  VFFT_IL_DECL_N1X(6, avx512)
VFFT_IL_DECL_N1X(7, avx512)  VFFT_IL_DECL_N1X(10, avx512) VFFT_IL_DECL_N1X(11, avx512)
VFFT_IL_DECL_N1X(12, avx512) VFFT_IL_DECL_N1X(13, avx512) VFFT_IL_DECL_N1X(17, avx512)
VFFT_IL_DECL_N1X(19, avx512) VFFT_IL_DECL_N1X(20, avx512) VFFT_IL_DECL_N1X(25, avx512)
#endif
#define VFFT_IL_T1S_O(R)   radix##R##_t1s_dit_fwd_avx512_il_out
#define VFFT_IL_T1_O(R)    radix##R##_t1_dit_fwd_avx512_il_out
#define VFFT_IL_T1L3_O(R)  radix##R##_t1_dit_log3_fwd_avx512_il_out
#define VFFT_IL_DIF_I(R)   radix##R##_t1_dif_fwd_avx512_il_in
#define VFFT_IL_DIFL3_I(R) radix##R##_t1_dif_log3_fwd_avx512_il_in
#define VFFT_IL_DIFB_O(R)  radix##R##_t1_dif_bwd_avx512_il_out
#define VFFT_IL_DIFBL3_O(R) radix##R##_t1_dif_log3_bwd_avx512_il_out
#define VFFT_IL_N1F_O(R)   radix##R##_n1_fwd_avx512_il_out
#define VFFT_IL_N1B_I(R)   radix##R##_n1_bwd_avx512_il_in
#define VFFT_IL_T1SB_I(R)  radix##R##_t1s_dit_bwd_avx512_il_in
#else
VFFT_IL_DECL_TWO(4, avx2)  VFFT_IL_DECL_TWO(8, avx2)
VFFT_IL_DECL_TWO(16, avx2) VFFT_IL_DECL_TWO(32, avx2)
VFFT_IL_DECL_T1SB(2, avx2)  VFFT_IL_DECL_T1SB(4, avx2)  VFFT_IL_DECL_T1SB(8, avx2)
VFFT_IL_DECL_T1SB(16, avx2) VFFT_IL_DECL_T1SB(32, avx2) VFFT_IL_DECL_T1SB(64, avx2)
VFFT_IL_DECL_N1X(2, avx2)  VFFT_IL_DECL_N1X(4, avx2)  VFFT_IL_DECL_N1X(8, avx2)
VFFT_IL_DECL_N1X(16, avx2) VFFT_IL_DECL_N1X(32, avx2) VFFT_IL_DECL_N1X(64, avx2)
#ifndef VFFT_IL_POW2_ONLY
VFFT_IL_DECL_TWO(5, avx2)  VFFT_IL_DECL_TWO(10, avx2)
VFFT_IL_DECL_TWO(20, avx2) VFFT_IL_DECL_TWO(25, avx2)
VFFT_IL_DECL_T1SB(5, avx2)  VFFT_IL_DECL_T1SB(10, avx2)
VFFT_IL_DECL_T1SB(20, avx2) VFFT_IL_DECL_T1SB(25, avx2)
VFFT_IL_DECL_N1X(3, avx2)  VFFT_IL_DECL_N1X(5, avx2)  VFFT_IL_DECL_N1X(6, avx2)
VFFT_IL_DECL_N1X(7, avx2)  VFFT_IL_DECL_N1X(10, avx2) VFFT_IL_DECL_N1X(11, avx2)
VFFT_IL_DECL_N1X(12, avx2) VFFT_IL_DECL_N1X(13, avx2) VFFT_IL_DECL_N1X(17, avx2)
VFFT_IL_DECL_N1X(19, avx2) VFFT_IL_DECL_N1X(20, avx2) VFFT_IL_DECL_N1X(25, avx2)
#endif
#define VFFT_IL_T1S_O(R)   radix##R##_t1s_dit_fwd_avx2_il_out
#define VFFT_IL_T1_O(R)    radix##R##_t1_dit_fwd_avx2_il_out
#define VFFT_IL_T1L3_O(R)  radix##R##_t1_dit_log3_fwd_avx2_il_out
#define VFFT_IL_DIF_I(R)   radix##R##_t1_dif_fwd_avx2_il_in
#define VFFT_IL_DIFL3_I(R) radix##R##_t1_dif_log3_fwd_avx2_il_in
#define VFFT_IL_DIFB_O(R)  radix##R##_t1_dif_bwd_avx2_il_out
#define VFFT_IL_DIFBL3_O(R) radix##R##_t1_dif_log3_bwd_avx2_il_out
#define VFFT_IL_N1F_O(R)   radix##R##_n1_fwd_avx2_il_out
#define VFFT_IL_N1B_I(R)   radix##R##_n1_bwd_avx2_il_in
#define VFFT_IL_T1SB_I(R)  radix##R##_t1s_dit_bwd_avx2_il_in
#endif

#define _VFFT_IL_TWSW(NAME, MACRO) \
static inline vfft_il_n1b_fn NAME(int r) { \
    switch (r) { \
    case 4:  return MACRO(4); \
    case 8:  return MACRO(8); \
    case 16: return MACRO(16); \
    case 32: return MACRO(32); \
    VFFT_IL_TW_EXT(MACRO) \
    default: return 0; } }
#define _VFFT_IL_TWSW_I(NAME, MACRO) \
static inline vfft_il_n1f_fn NAME(int r) { \
    switch (r) { \
    case 4:  return MACRO(4); \
    case 8:  return MACRO(8); \
    case 16: return MACRO(16); \
    case 32: return MACRO(32); \
    VFFT_IL_TW_EXT(MACRO) \
    default: return 0; } }
#ifndef VFFT_IL_POW2_ONLY
#define VFFT_IL_TW_EXT(M) \
    case 5:  return M(5); \
    case 10: return M(10); \
    case 20: return M(20); \
    case 25: return M(25);
#else
#define VFFT_IL_TW_EXT(M)
#endif
_VFFT_IL_TWSW(_vfft_il_t1s_ilo, VFFT_IL_T1S_O)
_VFFT_IL_TWSW(_vfft_il_t1_ilo, VFFT_IL_T1_O)
_VFFT_IL_TWSW(_vfft_il_t1l3_ilo, VFFT_IL_T1L3_O)
_VFFT_IL_TWSW(_vfft_il_difb_ilo, VFFT_IL_DIFB_O)
_VFFT_IL_TWSW(_vfft_il_difbl3_ilo, VFFT_IL_DIFBL3_O)
_VFFT_IL_TWSW_I(_vfft_il_dif_ili, VFFT_IL_DIF_I)
_VFFT_IL_TWSW_I(_vfft_il_difl3_ili, VFFT_IL_DIFL3_I)

static inline vfft_il_n1b_fn _vfft_il_n1f_ilo(int r)
{
    switch (r) {
    case 2:  return VFFT_IL_N1F_O(2);
    case 4:  return VFFT_IL_N1F_O(4);
    case 8:  return VFFT_IL_N1F_O(8);
    case 16: return VFFT_IL_N1F_O(16);
    case 32: return VFFT_IL_N1F_O(32);
    case 64: return VFFT_IL_N1F_O(64);
#ifndef VFFT_IL_POW2_ONLY
    case 3: return VFFT_IL_N1F_O(3);
    case 5: return VFFT_IL_N1F_O(5);
    case 6: return VFFT_IL_N1F_O(6);
    case 7: return VFFT_IL_N1F_O(7);
    case 10: return VFFT_IL_N1F_O(10);
    case 11: return VFFT_IL_N1F_O(11);
    case 12: return VFFT_IL_N1F_O(12);
    case 13: return VFFT_IL_N1F_O(13);
    case 17: return VFFT_IL_N1F_O(17);
    case 19: return VFFT_IL_N1F_O(19);
    case 20: return VFFT_IL_N1F_O(20);
    case 25: return VFFT_IL_N1F_O(25);
#endif
    default: return 0;
    }
}
static inline vfft_il_n1f_fn _vfft_il_n1b_ili(int r)
{
    switch (r) {
    case 2:  return VFFT_IL_N1B_I(2);
    case 4:  return VFFT_IL_N1B_I(4);
    case 8:  return VFFT_IL_N1B_I(8);
    case 16: return VFFT_IL_N1B_I(16);
    case 32: return VFFT_IL_N1B_I(32);
    case 64: return VFFT_IL_N1B_I(64);
#ifndef VFFT_IL_POW2_ONLY
    case 3: return VFFT_IL_N1B_I(3);
    case 5: return VFFT_IL_N1B_I(5);
    case 6: return VFFT_IL_N1B_I(6);
    case 7: return VFFT_IL_N1B_I(7);
    case 10: return VFFT_IL_N1B_I(10);
    case 11: return VFFT_IL_N1B_I(11);
    case 12: return VFFT_IL_N1B_I(12);
    case 13: return VFFT_IL_N1B_I(13);
    case 17: return VFFT_IL_N1B_I(17);
    case 19: return VFFT_IL_N1B_I(19);
    case 20: return VFFT_IL_N1B_I(20);
    case 25: return VFFT_IL_N1B_I(25);
#endif
    default: return 0;
    }
}

/* Local bounded copies of the generic stage loops (see executor_generic.h). */
static inline void _vfft_il_fwd_range(const stride_plan_t *plan, double *re,
                                      double *im, size_t slice_K, int from, int stop)
{
    for (int s = from; s < stop; s++) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups, R = st->radix;
        for (int g = 0; g < G; g++) {
            double *br = re + st->group_base[g], *bi = im + st->group_base[g];
            if (!st->needs_tw[g]) {
                st->n1_fwd(br, bi, br, bi, st->stride, st->stride, slice_K);
                continue;
            }
            double cfr = st->cf0_re[g], cfi = st->cf0_im[g];
            if (st->use_log3) {
                if (cfr != 1.0 || cfi != 0.0)
                    for (int j = 0; j < R; j++)
                        _stride_cmul_scalar_inplace(br + (size_t)j * st->stride,
                                                    bi + (size_t)j * st->stride,
                                                    slice_K, cfr, cfi);
                const int Rm1 = R - 1;
                double twr[63 * VFFT_PROTO_TW_BLOCK_K], twi[63 * VFFT_PROTO_TW_BLOCK_K];
                for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
                    size_t tK = slice_K - kb;
                    if (tK > VFFT_PROTO_TW_BLOCK_K) tK = VFFT_PROTO_TW_BLOCK_K;
                    for (int j = 0; j < Rm1; j++)
                        _stride_broadcast_2(twr + (size_t)j * tK, twi + (size_t)j * tK, tK,
                                            st->grp_tw_re[g][(size_t)j * plan->K],
                                            st->grp_tw_im[g][(size_t)j * plan->K]);
                    st->t1_fwd(br + kb, bi + kb, twr, twi, st->stride, tK);
                }
                continue;
            }
            if (st->t1s_fwd) {
                if (cfr != 1.0 || cfi != 0.0)
                    _stride_cmul_scalar_inplace(br, bi, slice_K, cfr, cfi);
                st->t1s_fwd(br, bi, st->tw_scalar_re[g], st->tw_scalar_im[g],
                            st->stride, slice_K);
                continue;
            }
            if (cfr != 1.0 || cfi != 0.0)
                _stride_cmul_scalar_inplace(br, bi, slice_K, cfr, cfi);
            {
                const int Rm1 = R - 1;
                const double *sr = st->tw_scalar_re[g], *si = st->tw_scalar_im[g];
                double twr[63 * VFFT_PROTO_TW_BLOCK_K], twi[63 * VFFT_PROTO_TW_BLOCK_K];
                for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
                    size_t tK = slice_K - kb;
                    if (tK > VFFT_PROTO_TW_BLOCK_K) tK = VFFT_PROTO_TW_BLOCK_K;
                    for (int j = 0; j < Rm1; j++)
                        _stride_broadcast_2(twr + (size_t)j * tK, twi + (size_t)j * tK,
                                            tK, sr[j], si[j]);
                    st->t1_fwd(br + kb, bi + kb, twr, twi, st->stride, tK);
                }
            }
        }
    }
}
static inline void _vfft_il_fwd_dif_range(const stride_plan_t *plan, double *re,
                                          double *im, size_t slice_K, int from, int stop)
{
    for (int s = from; s < stop; s++) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        for (int g = 0; g < G; g++) {
            double *br = re + st->group_base[g], *bi = im + st->group_base[g];
            if (!st->needs_tw[g]) {
                st->n1_fwd(br, bi, br, bi, st->stride, st->stride, slice_K);
                continue;
            }
            const int Rm1 = st->radix - 1;
            double twr[63 * VFFT_PROTO_TW_BLOCK_K], twi[63 * VFFT_PROTO_TW_BLOCK_K];
            for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
                size_t tK = slice_K - kb;
                if (tK > VFFT_PROTO_TW_BLOCK_K) tK = VFFT_PROTO_TW_BLOCK_K;
                for (int j = 0; j < Rm1; j++)
                    _stride_broadcast_2(twr + (size_t)j * tK, twi + (size_t)j * tK, tK,
                                        st->grp_tw_re[g][(size_t)j * plan->K],
                                        st->grp_tw_im[g][(size_t)j * plan->K]);
                st->t1_fwd(br + kb, bi + kb, twr, twi, st->stride, tK);
            }
        }
    }
}
static inline void _vfft_il_bwd_range(const stride_plan_t *plan, double *re,
                                      double *im, size_t slice_K, int top_from, int until)
{
    const size_t full_K = plan->K;
    for (int s = top_from; s >= until; s--) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups, R = st->radix;
        for (int g = 0; g < G; g++) {
            double *br = re + st->group_base[g], *bi = im + st->group_base[g];
            st->n1_bwd(br, bi, br, bi, st->stride, st->stride, slice_K);
            if (!st->needs_tw[g]) continue;
            const double *cr = st->cf_all_re + (size_t)g * R * full_K;
            const double *ci = st->cf_all_im + (size_t)g * R * full_K;
            for (int j = 0; j < R; j++)
                _vfft_proto_cmul_conj_vec(br + (size_t)j * st->stride,
                                          bi + (size_t)j * st->stride,
                                          cr + (size_t)j * full_K,
                                          ci + (size_t)j * full_K, slice_K);
        }
    }
}
static inline void _vfft_il_bwd_dif_range(const stride_plan_t *plan, double *re,
                                          double *im, size_t slice_K, int top_from, int until)
{
    for (int s = top_from; s >= until; s--) {
        const stride_stage_t *st = &plan->stages[s];
        const int G = st->num_groups;
        for (int g = 0; g < G; g++) {
            double *br = re + st->group_base[g], *bi = im + st->group_base[g];
            if (!st->needs_tw[g]) {
                st->n1_bwd(br, bi, br, bi, st->stride, st->stride, slice_K);
                continue;
            }
            const int Rm1 = st->radix - 1;
            double twr[63 * VFFT_PROTO_TW_BLOCK_K], twi[63 * VFFT_PROTO_TW_BLOCK_K];
            for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
                size_t tK = slice_K - kb;
                if (tK > VFFT_PROTO_TW_BLOCK_K) tK = VFFT_PROTO_TW_BLOCK_K;
                for (int j = 0; j < Rm1; j++)
                    _stride_broadcast_2(twr + (size_t)j * tK, twi + (size_t)j * tK, tK,
                                        st->grp_tw_re[g][(size_t)j * plan->K],
                                        st->grp_tw_im[g][(size_t)j * plan->K]);
                st->t1_bwd(br + kb, bi + kb, twr, twi, st->stride, tK);
            }
        }
    }
}

static inline vfft_il_n1f_fn _vfft_il_t1sb_ili(int r)
{
    switch (r) {
    case 2:  return VFFT_IL_T1SB_I(2);
    case 4:  return VFFT_IL_T1SB_I(4);
    case 8:  return VFFT_IL_T1SB_I(8);
    case 16: return VFFT_IL_T1SB_I(16);
    case 32: return VFFT_IL_T1SB_I(32);
    case 64: return VFFT_IL_T1SB_I(64);
#ifndef VFFT_IL_POW2_ONLY
    case 5:  return VFFT_IL_T1SB_I(5);
    case 10: return VFFT_IL_T1SB_I(10);
    case 20: return VFFT_IL_T1SB_I(20);
    case 25: return VFFT_IL_T1SB_I(25);
#endif
    default: return 0;
    }
}

/* Verbatim VFFT_PROTO_BWD_LEG0_CONJ_avx2 (plan_executors.h) as a function —
 * bit parity with the jit STAGE_BWD leg-0 fixup demands the exact FMA
 * grouping, so this is a copy, not a call to _stride_cmul_scalar_inplace. */
static inline void _vfft_il_bwd_leg0_conj(double *re_p, double *im_p,
                                          double cfr, double cfi, size_t n)
{
    if (cfr != 1.0 || cfi != 0.0) {
        __m256d vcfr = _mm256_set1_pd(cfr);
        __m256d vcfi = _mm256_set1_pd(cfi);
        size_t kk = 0;
        for (; kk + 4 <= n; kk += 4) {
            __m256d vr = _mm256_loadu_pd(re_p + kk);
            __m256d vi = _mm256_loadu_pd(im_p + kk);
            __m256d nr = _mm256_fmadd_pd(vi, vcfi, _mm256_mul_pd(vr, vcfr));
            __m256d ni = _mm256_fnmadd_pd(vr, vcfi, _mm256_mul_pd(vi, vcfr));
            _mm256_storeu_pd(re_p + kk, nr);
            _mm256_storeu_pd(im_p + kk, ni);
        }
        for (; kk < n; kk++) {
            double tr = re_p[kk];
            re_p[kk] = tr * cfr + im_p[kk] * cfi;
            im_p[kk] = im_p[kk] * cfr - tr * cfi;
        }
    }
}


/* ── 6a17 fold helpers: resolve/apply pairs shared by _core and _jit tiers.
 * resolve_* never touches data (resolve-before-touch doctrine); apply_* runs
 * the boundary group loop. The fwd_ilin/bwd_ilout DIF branches above keep
 * their inline copies (accepted duplication to cap refactor blast radius). */
typedef struct { vfft_il_n1f_fn n1, tw; } vfft_il_infold_t;   /* z -> split */
typedef struct { vfft_il_n1b_fn n1, tw; } vfft_il_outfold_t;  /* split -> z */

static inline int _vfft_il_resolve_fwd_entry(const stride_plan_t *plan,
                                             vfft_il_infold_t *f)
{
    const stride_stage_t *st = &plan->stages[0];
    int tw = 0, n1 = 0;
    for (int g = 0; g < st->num_groups; g++)
        if (st->needs_tw[g]) tw = 1; else n1 = 1;
    f->n1 = f->tw = 0;
    if (n1 && !(f->n1 = _vfft_il_s0_fwd(st->radix))) return -1;
    if (tw) {
        if (!plan->use_dif_forward) return -1; /* DIT stage 0 is untwiddled */
        f->tw = st->use_log3 ? _vfft_il_difl3_ili(st->radix)
                             : _vfft_il_dif_ili(st->radix);
        if (!f->tw) return -1;
    }
    return 0;
}
static inline void _vfft_il_apply_fwd_entry(const stride_plan_t *plan,
                                            const double *z, double *re, double *im,
                                            size_t slice_K, const vfft_il_infold_t *f)
{
    const stride_stage_t *st = &plan->stages[0];
    for (int g = 0; g < st->num_groups; g++) {
        double *br = re + st->group_base[g], *bi = im + st->group_base[g];
        const double *zb = z + 2 * (size_t)st->group_base[g];
        if (!st->needs_tw[g]) {
            f->n1(zb, br, bi, 0, 0, st->stride, slice_K);
            continue;
        }
        const int Rm1 = st->radix - 1;
        double twr[63 * VFFT_PROTO_TW_BLOCK_K], twi[63 * VFFT_PROTO_TW_BLOCK_K];
        for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
            size_t tK = slice_K - kb;
            if (tK > VFFT_PROTO_TW_BLOCK_K) tK = VFFT_PROTO_TW_BLOCK_K;
            for (int j = 0; j < Rm1; j++)
                _stride_broadcast_2(twr + (size_t)j * tK, twi + (size_t)j * tK, tK,
                                    st->grp_tw_re[g][(size_t)j * plan->K],
                                    st->grp_tw_im[g][(size_t)j * plan->K]);
            f->tw(zb + 2 * kb, br + kb, bi + kb, twr, twi, st->stride, tK);
        }
    }
}
static inline int _vfft_il_resolve_fwd_exit(const stride_plan_t *plan,
                                            vfft_il_outfold_t *f)
{
    const stride_stage_t *st = &plan->stages[plan->num_stages - 1];
    int tw = 0, n1 = 0;
    for (int g = 0; g < st->num_groups; g++)
        if (st->needs_tw[g]) tw = 1; else n1 = 1;
    f->n1 = f->tw = 0;
    if (n1 && !(f->n1 = _vfft_il_n1f_ilo(st->radix))) return -1;
    if (tw) {
        if (plan->use_dif_forward) return -1; /* DIF last is untwiddled */
        f->tw = st->use_log3 ? _vfft_il_t1l3_ilo(st->radix)
              : st->t1s_fwd  ? _vfft_il_t1s_ilo(st->radix)
                             : _vfft_il_t1_ilo(st->radix);
        if (!f->tw) return -1;
    }
    return 0;
}
static inline void _vfft_il_apply_fwd_exit(const stride_plan_t *plan,
                                           double *re, double *im, double *z,
                                           size_t slice_K, const vfft_il_outfold_t *f)
{
    const stride_stage_t *st = &plan->stages[plan->num_stages - 1];
    for (int g = 0; g < st->num_groups; g++) {
        double *br = re + st->group_base[g], *bi = im + st->group_base[g];
        double *zb = z + 2 * (size_t)st->group_base[g];
        if (!st->needs_tw[g]) {
            f->n1(br, bi, zb, 0, 0, st->stride, slice_K);
            continue;
        }
        double cfr = st->cf0_re[g], cfi = st->cf0_im[g];
        if (st->use_log3) {
            if (cfr != 1.0 || cfi != 0.0)
                for (int j = 0; j < st->radix; j++)
                    _stride_cmul_scalar_inplace(br + (size_t)j * st->stride,
                                                bi + (size_t)j * st->stride,
                                                slice_K, cfr, cfi);
            const int Rm1 = st->radix - 1;
            double twr[63 * VFFT_PROTO_TW_BLOCK_K], twi[63 * VFFT_PROTO_TW_BLOCK_K];
            for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
                size_t tK = slice_K - kb;
                if (tK > VFFT_PROTO_TW_BLOCK_K) tK = VFFT_PROTO_TW_BLOCK_K;
                for (int j = 0; j < Rm1; j++)
                    _stride_broadcast_2(twr + (size_t)j * tK, twi + (size_t)j * tK, tK,
                                        st->grp_tw_re[g][(size_t)j * plan->K],
                                        st->grp_tw_im[g][(size_t)j * plan->K]);
                f->tw(br + kb, bi + kb, zb + 2 * kb, twr, twi, st->stride, tK);
            }
            continue;
        }
        if (cfr != 1.0 || cfi != 0.0)
            _stride_cmul_scalar_inplace(br, bi, slice_K, cfr, cfi);
        if (st->t1s_fwd) {
            f->tw(br, bi, zb, st->tw_scalar_re[g], st->tw_scalar_im[g],
                  st->stride, slice_K);
            continue;
        }
        {
            const int Rm1 = st->radix - 1;
            const double *sr = st->tw_scalar_re[g], *si = st->tw_scalar_im[g];
            double twr[63 * VFFT_PROTO_TW_BLOCK_K], twi[63 * VFFT_PROTO_TW_BLOCK_K];
            for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
                size_t tK = slice_K - kb;
                if (tK > VFFT_PROTO_TW_BLOCK_K) tK = VFFT_PROTO_TW_BLOCK_K;
                for (int j = 0; j < Rm1; j++)
                    _stride_broadcast_2(twr + (size_t)j * tK, twi + (size_t)j * tK,
                                        tK, sr[j], si[j]);
                f->tw(br + kb, bi + kb, zb + 2 * kb, twr, twi, st->stride, tK);
            }
        }
    }
}
static inline int _vfft_il_resolve_bwd_exit(const stride_plan_t *plan,
                                            vfft_il_outfold_t *f)
{
    const stride_stage_t *st = &plan->stages[0];
    int tw = 0, n1 = 0;
    for (int g = 0; g < st->num_groups; g++)
        if (st->needs_tw[g]) tw = 1; else n1 = 1;
    f->n1 = f->tw = 0;
    if (n1 && !(f->n1 = _vfft_il_s0_bwd(st->radix))) return -1;
    if (tw) {
        if (!plan->use_dif_forward) return -1; /* DIT stage 0 untwiddled */
        f->tw = st->use_log3 ? _vfft_il_difbl3_ilo(st->radix)
                             : _vfft_il_difb_ilo(st->radix);
        if (!f->tw) return -1;
    }
    return 0;
}
static inline void _vfft_il_apply_bwd_exit(const stride_plan_t *plan,
                                           double *re, double *im, double *z,
                                           size_t slice_K, const vfft_il_outfold_t *f)
{
    const stride_stage_t *st = &plan->stages[0];
    for (int g = 0; g < st->num_groups; g++) {
        double *br = re + st->group_base[g], *bi = im + st->group_base[g];
        double *zb = z + 2 * (size_t)st->group_base[g];
        if (!st->needs_tw[g]) {
            f->n1(br, bi, zb, 0, 0, st->stride, slice_K);
            continue;
        }
        const int Rm1 = st->radix - 1;
        double twr[63 * VFFT_PROTO_TW_BLOCK_K], twi[63 * VFFT_PROTO_TW_BLOCK_K];
        for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
            size_t tK = slice_K - kb;
            if (tK > VFFT_PROTO_TW_BLOCK_K) tK = VFFT_PROTO_TW_BLOCK_K;
            for (int j = 0; j < Rm1; j++)
                _stride_broadcast_2(twr + (size_t)j * tK, twi + (size_t)j * tK, tK,
                                    st->grp_tw_re[g][(size_t)j * plan->K],
                                    st->grp_tw_im[g][(size_t)j * plan->K]);
            f->tw(br + kb, bi + kb, zb + 2 * kb, twr, twi, st->stride, tK);
        }
    }
}
/* 6a21: the jit tier's DIF bwd is variant-bound (emit_jit v5), so the jit
 * il2il exit fold uses the same variant-bound resolver as the core tier.
 * The plain-forced twin that lived here is deleted. */
static inline int _vfft_il_resolve_bwd_entry_gen(const stride_plan_t *plan,
                                                 vfft_il_infold_t *f)
{
    f->tw = 0;
    f->n1 = _vfft_il_n1b_ili(plan->stages[plan->num_stages - 1].radix);
    return f->n1 ? 0 : -1;
}
static inline void _vfft_il_apply_bwd_entry_gen(const stride_plan_t *plan,
                                                const double *z, double *re, double *im,
                                                size_t slice_K, const vfft_il_infold_t *f)
{
    const stride_stage_t *st = &plan->stages[plan->num_stages - 1];
    const size_t full_K = plan->K;
    for (int g = 0; g < st->num_groups; g++) {
        double *br = re + st->group_base[g], *bi = im + st->group_base[g];
        const double *zb = z + 2 * (size_t)st->group_base[g];
        f->n1(zb, br, bi, 0, 0, st->stride, slice_K);
        if (!st->needs_tw[g]) continue; /* DIF top (outer) never twiddled */
        const double *cr = st->cf_all_re + (size_t)g * st->radix * full_K;
        const double *ci = st->cf_all_im + (size_t)g * st->radix * full_K;
        for (int j = 0; j < st->radix; j++)
            _vfft_proto_cmul_conj_vec(br + (size_t)j * st->stride,
                                      bi + (size_t)j * st->stride,
                                      cr + (size_t)j * full_K,
                                      ci + (size_t)j * full_K, slice_K);
    }
}
/* JIT-tier bwd entry: mirrors VFFT_PROTO_STAGE_BWD at the top stage —
 * fused t1s_dit_bwd (legs 1..R-1 conj-twiddled in-codelet) + leg0-conj on
 * split, exactly the macro's op order. DIF top is the untwiddled outer, so
 * only the n1 branch fires there. */
static inline int _vfft_il_resolve_bwd_entry_jit(const stride_plan_t *plan,
                                                 vfft_il_infold_t *f)
{
    const stride_stage_t *st = &plan->stages[plan->num_stages - 1];
    int tw = 0, n1 = 0;
    for (int g = 0; g < st->num_groups; g++)
        if (st->needs_tw[g]) tw = 1; else n1 = 1;
    f->n1 = f->tw = 0;
    if (n1 && !(f->n1 = _vfft_il_n1b_ili(st->radix))) return -1;
    if (tw && !(f->tw = _vfft_il_t1sb_ili(st->radix))) return -1;
    return 0;
}
static inline void _vfft_il_apply_bwd_entry_jit(const stride_plan_t *plan,
                                                const double *z, double *re, double *im,
                                                size_t slice_K, const vfft_il_infold_t *f)
{
    const stride_stage_t *st = &plan->stages[plan->num_stages - 1];
    for (int g = 0; g < st->num_groups; g++) {
        double *br = re + st->group_base[g], *bi = im + st->group_base[g];
        const double *zb = z + 2 * (size_t)st->group_base[g];
        if (!st->needs_tw[g]) {
            f->n1(zb, br, bi, 0, 0, st->stride, slice_K);
            continue;
        }
        f->tw(zb, br, bi, st->tw_scalar_re[g], st->tw_scalar_im[g],
              st->stride, slice_K);
        _vfft_il_bwd_leg0_conj(br, bi, st->cf0_re[g], st->cf0_im[g], slice_K);
    }
}

/* Fully-folded z -> z transforms: entry lattice + split interior + exit
 * lattice, no converter sweeps. work_re/im are scratch (clobbered).
 * z_in == z_out is safe (entry fully consumes z before exit writes).
 * Single-stage plans return -1 (entry and exit share the stage). */
static inline int vfft_proto_execute_fwd_il2il_core(const stride_plan_t *plan,
                                                    const double *z_in,
                                                    double *work_re, double *work_im,
                                                    double *z_out, size_t slice_K)
{
    if (plan->override_fwd || plan->num_stages < 2) return -1;
    vfft_il_infold_t e; vfft_il_outfold_t x;
    if (_vfft_il_resolve_fwd_entry(plan, &e) || _vfft_il_resolve_fwd_exit(plan, &x))
        return -1;
    _vfft_il_apply_fwd_entry(plan, z_in, work_re, work_im, slice_K, &e);
    if (plan->use_dif_forward)
        _vfft_il_fwd_dif_range(plan, work_re, work_im, slice_K, 1, plan->num_stages - 1);
    else
        _vfft_il_fwd_range(plan, work_re, work_im, slice_K, 1, plan->num_stages - 1);
    _vfft_il_apply_fwd_exit(plan, work_re, work_im, z_out, slice_K, &x);
    return 0;
}

static inline int vfft_proto_execute_bwd_il2il_core(const stride_plan_t *plan,
                                                    const double *z_in,
                                                    double *work_re, double *work_im,
                                                    double *z_out, size_t slice_K)
{
    if (plan->override_bwd || plan->num_stages < 2) return -1;
    vfft_il_infold_t e; vfft_il_outfold_t x;
    if (_vfft_il_resolve_bwd_entry_gen(plan, &e) || _vfft_il_resolve_bwd_exit(plan, &x))
        return -1;
    _vfft_il_apply_bwd_entry_gen(plan, z_in, work_re, work_im, slice_K, &e);
    if (plan->use_dif_forward)
        _vfft_il_bwd_dif_range(plan, work_re, work_im, slice_K, plan->num_stages - 2, 1);
    else
        _vfft_il_bwd_range(plan, work_re, work_im, slice_K, plan->num_stages - 2, 1);
    _vfft_il_apply_bwd_exit(plan, work_re, work_im, z_out, slice_K, &x);
    return 0;
}

/* Forward EXIT fold: run stages [0, n-1) on the split work arrays, then the
 * last stage writes interleaved z directly. Mid stages run GENERIC (jit
 * executors have start_stage only — a stop gate is the follow-up); for
 * 2-stage plans this costs ~nothing. re/im are clobbered (work buffers). */
static inline int vfft_proto_execute_fwd_ilout_core(const stride_plan_t *plan,
                                                    double *re, double *im,
                                                    double *out_z, size_t slice_K)
{
    if (plan->override_fwd) return -1;
    vfft_il_outfold_t x;
    if (_vfft_il_resolve_fwd_exit(plan, &x)) return -1;
    if (plan->use_dif_forward)
        _vfft_il_fwd_dif_range(plan, re, im, slice_K, 0, plan->num_stages - 1);
    else
        _vfft_il_fwd_range(plan, re, im, slice_K, 0, plan->num_stages - 1);
    _vfft_il_apply_fwd_exit(plan, re, im, out_z, slice_K, &x);
    return 0;
}
static inline int vfft_proto_execute_fwd_ilout_jit(const stride_plan_t *plan,
                                                   double *re, double *im,
                                                   double *out_z, size_t slice_K,
                                                   vfft_proto_exec_range_fn range_fn)
{
    if (!range_fn)
        return vfft_proto_execute_fwd_ilout_core(plan, re, im, out_z, slice_K);
    if (plan->override_fwd) return -1;
    vfft_il_outfold_t x;
    if (_vfft_il_resolve_fwd_exit(plan, &x)) return -1;
    range_fn(plan, re, im, slice_K, plan->K, 0, plan->num_stages - 1);
    _vfft_il_apply_fwd_exit(plan, re, im, out_z, slice_K, &x);
    return 0;
}

/* Backward ENTRY fold: first-executed stage (n-1) reads z pairs; remaining
 * stages run generic (reverse resume is not expressible with the jit
 * start-gate). conj(cf_all) fixup applies on SPLIT output, order unchanged. */
static inline int vfft_proto_execute_bwd_ilin_core(const stride_plan_t *plan,
                                                   const double *z,
                                                   double *re, double *im,
                                                   size_t slice_K)
{
    if (plan->override_bwd) return -1;
    vfft_il_infold_t e;
    if (_vfft_il_resolve_bwd_entry_gen(plan, &e)) return -1;
    _vfft_il_apply_bwd_entry_gen(plan, z, re, im, slice_K, &e);
    if (plan->use_dif_forward)
        _vfft_il_bwd_dif_range(plan, re, im, slice_K, plan->num_stages - 2, 0);
    else
        _vfft_il_bwd_range(plan, re, im, slice_K, plan->num_stages - 2, 0);
    return 0;
}

/* ── 6a17 jit-tier wrappers: interior via the range executor, boundaries via
 * the fold helpers. Tier purity: any resolve gap falls back to the WHOLE
 * _core path (never a mixed-tier pipeline), keeping bit identity with the
 * corresponding full-tier reference. ─────────────────────────────────── */
static inline int vfft_proto_execute_fwd_il2il_jit(const stride_plan_t *plan,
                                                   const double *z_in,
                                                   double *work_re, double *work_im,
                                                   double *z_out, size_t slice_K,
                                                   vfft_proto_exec_range_fn range_fn)
{
    if (!range_fn)
        return vfft_proto_execute_fwd_il2il_core(plan, z_in, work_re, work_im,
                                                 z_out, slice_K);
    if (plan->override_fwd || plan->num_stages < 2) return -1;
    vfft_il_infold_t e; vfft_il_outfold_t x;
    if (_vfft_il_resolve_fwd_entry(plan, &e) || _vfft_il_resolve_fwd_exit(plan, &x))
        return -1;
    _vfft_il_apply_fwd_entry(plan, z_in, work_re, work_im, slice_K, &e);
    range_fn(plan, work_re, work_im, slice_K, plan->K, 1, plan->num_stages - 1);
    _vfft_il_apply_fwd_exit(plan, work_re, work_im, z_out, slice_K, &x);
    return 0;
}
static inline int vfft_proto_execute_bwd_ilin_jit2(const stride_plan_t *plan,
                                                   const double *z,
                                                   double *re, double *im,
                                                   size_t slice_K,
                                                   vfft_proto_exec_range_fn range_fn)
{
    if (!range_fn || plan->num_stages < 2)
        return vfft_proto_execute_bwd_ilin_core(plan, z, re, im, slice_K);
    if (plan->override_bwd) return -1;
    vfft_il_infold_t e;
    if (_vfft_il_resolve_bwd_entry_jit(plan, &e))
        return vfft_proto_execute_bwd_ilin_core(plan, z, re, im, slice_K);
    _vfft_il_apply_bwd_entry_jit(plan, z, re, im, slice_K, &e);
    range_fn(plan, re, im, slice_K, plan->K, 0, plan->num_stages - 1);
    return 0;
}
static inline int vfft_proto_execute_bwd_il2il_jit(const stride_plan_t *plan,
                                                   const double *z_in,
                                                   double *work_re, double *work_im,
                                                   double *z_out, size_t slice_K,
                                                   vfft_proto_exec_range_fn range_fn)
{
    if (!range_fn)
        return vfft_proto_execute_bwd_il2il_core(plan, z_in, work_re, work_im,
                                                 z_out, slice_K);
    if (plan->override_bwd || plan->num_stages < 2) return -1;
    vfft_il_infold_t e; vfft_il_outfold_t x;
    if (_vfft_il_resolve_bwd_exit(plan, &x)) return -1;
    if (_vfft_il_resolve_bwd_entry_jit(plan, &e))
        return vfft_proto_execute_bwd_il2il_core(plan, z_in, work_re, work_im,
                                                 z_out, slice_K);
    _vfft_il_apply_bwd_entry_jit(plan, z_in, work_re, work_im, slice_K, &e);
    range_fn(plan, work_re, work_im, slice_K, plan->K, 1, plan->num_stages - 1);
    _vfft_il_apply_bwd_exit(plan, work_re, work_im, z_out, slice_K, &x);
    return 0;
}

static inline int _vfft_il_eligible(const stride_plan_t *plan)
{
    if (plan->use_dif_forward || plan->override_fwd)
        return 0;
    const stride_stage_t *st = &plan->stages[0];
    for (int g = 0; g < st->num_groups; g++)
        if (st->needs_tw[g])
            return 0;
    return 1;
}

/* Interleaved-in forward: z (pairs) -> dst split (natural plan output).
 * 0 on success; -1 -> caller falls back to il2sp + execute. */
static inline int vfft_proto_execute_fwd_ilin_core(const stride_plan_t *plan,
                                              const double *z,
                                              double *dst_re, double *dst_im,
                                              size_t slice_K,
                                              vfft_proto_exec_fn stages1)
{
    /* DIF entry fold: stage 0 is twiddled (t1_dif[_log3] il_in per group),
     * DIF outer is last. Remainder resumes at stage 1 (jit DIF macros are
     * start_stage-gated) or via the local dif range. */
    if (plan->use_dif_forward && !plan->override_fwd) {
        const stride_stage_t *st = &plan->stages[0];
        vfft_il_n1f_fn f_tw = 0, f_n1 = 0;
        int need_tw = 0, need_n1 = 0;
        for (int g = 0; g < st->num_groups; g++)
            if (st->needs_tw[g]) need_tw = 1; else need_n1 = 1;
        if (need_n1 && !(f_n1 = _vfft_il_s0_fwd(st->radix))) return -1;
        if (need_tw) {
            f_tw = st->use_log3 ? _vfft_il_difl3_ili(st->radix)
                                : _vfft_il_dif_ili(st->radix);
            if (!f_tw) return -1;
        }
        for (int g = 0; g < st->num_groups; g++) {
            double *br = dst_re + st->group_base[g], *bi = dst_im + st->group_base[g];
            const double *zb = z + 2 * (size_t)st->group_base[g];
            if (!st->needs_tw[g]) {
                f_n1(zb, br, bi, 0, 0, st->stride, slice_K);
                continue;
            }
            const int Rm1 = st->radix - 1;
            double twr[63 * VFFT_PROTO_TW_BLOCK_K], twi[63 * VFFT_PROTO_TW_BLOCK_K];
            for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
                size_t tK = slice_K - kb;
                if (tK > VFFT_PROTO_TW_BLOCK_K) tK = VFFT_PROTO_TW_BLOCK_K;
                for (int j = 0; j < Rm1; j++)
                    _stride_broadcast_2(twr + (size_t)j * tK, twi + (size_t)j * tK, tK,
                                        st->grp_tw_re[g][(size_t)j * plan->K],
                                        st->grp_tw_im[g][(size_t)j * plan->K]);
                f_tw(zb + 2 * kb, br + kb, bi + kb, twr, twi, st->stride, tK);
            }
        }
        if (stages1)
            stages1(plan, dst_re, dst_im, slice_K, plan->K, /*start_stage=*/1);
        else
            _vfft_il_fwd_dif_range(plan, dst_re, dst_im, slice_K, 1, plan->num_stages);
        return 0;
    }
    if (!_vfft_il_eligible(plan))
        return -1;
    const stride_stage_t *st = &plan->stages[0];
    vfft_il_n1f_fn fn = _vfft_il_s0_fwd(st->radix);
    if (!fn)
        return -1;
    for (int g = 0; g < st->num_groups; g++)
    {
        const size_t off = st->group_base[g];
        fn(z + 2 * off, dst_re + off, dst_im + off, 0, 0, st->stride, slice_K);
    }
    if (plan->num_stages > 1)
        {
            /* Baked (tier-1 JIT) resume when the plan has one: every STAGE_*
             * macro is gated `if (start_stage <= S)`, so start_stage=1 runs
             * exactly stages 1.. — closes the generic-dispatch tax
             * (~143K cyc/axis at the 64^3 shape). Generic is the fallback. */
            if (stages1)
                stages1(plan, dst_re, dst_im, slice_K, plan->K,
                        /*start_stage=*/1);
            else
                vfft_proto_execute_fwd_generic_from(plan, dst_re, dst_im,
                                                    slice_K, 1);
        }
    return 0;
}

/* Interleaved-out backward: src split spectrum (DESTROYED — stages run
 * in-place on it) -> z (pairs). Unnormalized inverse, same semantics as
 * vfft_proto_execute_bwd. 0 on success; -1 -> fallback. */

static inline int vfft_proto_execute_fwd_ilin(const stride_plan_t *plan,
                                              const double *z,
                                              double *dst_re, double *dst_im,
                                              size_t slice_K)
{
    return vfft_proto_execute_fwd_ilin_core(plan, z, dst_re, dst_im, slice_K,
                                            _vfft_proto_lookup_fwd(plan));
}

/* Recommended tier: caller resolves once at plan time via
 * vfft_proto_plan_jit_fwd() (jit_runtime.h) and passes the fn — mirrors
 * oop_execute.h's stages1_jit contract. The JIT body uses the same
 * start_stage-gated STAGE macros, so the resume contract holds. */
static inline int vfft_proto_execute_fwd_ilin_jit(const stride_plan_t *plan,
                                                  const double *z,
                                                  double *dst_re,
                                                  double *dst_im,
                                                  size_t slice_K,
                                                  vfft_proto_exec_fn stages1)
{
    return vfft_proto_execute_fwd_ilin_core(plan, z, dst_re, dst_im, slice_K,
                                            stages1);
}

static inline int vfft_proto_execute_bwd_ilout_core(const stride_plan_t *plan,
                                               double *src_re, double *src_im,
                                               double *z,
                                               size_t slice_K,
                                               vfft_proto_exec_fn stages1)
{
    /* DIF exit fold: bwd runs n-1..1 (t1_dif_bwd fused, no fixups), stage 0
     * writes z via t1_dif_bwd il_out per kb-block. jit remainder uses the
     * start_stage gate (skip stage 0). */
    if (plan->use_dif_forward && !plan->override_bwd) {
        const stride_stage_t *st = &plan->stages[0];
        vfft_il_n1b_fn f_tw = 0, f_n1 = 0;
        int need_tw = 0, need_n1 = 0;
        for (int g = 0; g < st->num_groups; g++)
            if (st->needs_tw[g]) need_tw = 1; else need_n1 = 1;
        if (need_n1 && !(f_n1 = _vfft_il_s0_bwd(st->radix))) return -1;
        if (need_tw) {
            f_tw = st->use_log3 ? _vfft_il_difbl3_ilo(st->radix)
                                : _vfft_il_difb_ilo(st->radix);
            if (!f_tw) return -1;
        }
        if (stages1)
            stages1(plan, src_re, src_im, slice_K, plan->K, /*start_stage=*/1);
        else
            _vfft_il_bwd_dif_range(plan, src_re, src_im, slice_K, plan->num_stages - 1, 1);
        for (int g = 0; g < st->num_groups; g++) {
            double *br = src_re + st->group_base[g], *bi = src_im + st->group_base[g];
            double *zb = z + 2 * (size_t)st->group_base[g];
            if (!st->needs_tw[g]) {
                f_n1(br, bi, zb, 0, 0, st->stride, slice_K);
                continue;
            }
            const int Rm1 = st->radix - 1;
            double twr[63 * VFFT_PROTO_TW_BLOCK_K], twi[63 * VFFT_PROTO_TW_BLOCK_K];
            for (size_t kb = 0; kb < slice_K; kb += VFFT_PROTO_TW_BLOCK_K) {
                size_t tK = slice_K - kb;
                if (tK > VFFT_PROTO_TW_BLOCK_K) tK = VFFT_PROTO_TW_BLOCK_K;
                for (int j = 0; j < Rm1; j++)
                    _stride_broadcast_2(twr + (size_t)j * tK, twi + (size_t)j * tK, tK,
                                        st->grp_tw_re[g][(size_t)j * plan->K],
                                        st->grp_tw_im[g][(size_t)j * plan->K]);
                f_tw(br + kb, bi + kb, zb + 2 * kb, twr, twi, st->stride, tK);
            }
        }
        return 0;
    }

    if (!_vfft_il_eligible(plan))
        return -1;
    const stride_stage_t *st = &plan->stages[0];
    vfft_il_n1b_fn fn = _vfft_il_s0_bwd(st->radix);
    if (!fn)
        return -1;
    if (plan->num_stages > 1)
        {
            /* Same gate on the reversed executor: start_stage=1 excludes
             * S=0 regardless of execution order, i.e. until(1) semantics
             * on the baked bwd path for free. */
            if (stages1)
                stages1(plan, src_re, src_im, slice_K, plan->K,
                        /*start_stage=*/1);
            else
                vfft_proto_execute_bwd_generic_until(plan, src_re, src_im,
                                                     slice_K, 1);
        }
    for (int g = 0; g < st->num_groups; g++)
    {
        const size_t off = st->group_base[g];
        fn(src_re + off, src_im + off, z + 2 * off, 0, 0, st->stride, slice_K);
    }
    return 0;
}


static inline int vfft_proto_execute_bwd_ilout(const stride_plan_t *plan,
                                               double *src_re, double *src_im,
                                               double *z, size_t slice_K)
{
    return vfft_proto_execute_bwd_ilout_core(plan, src_re, src_im, z, slice_K,
                                             _vfft_proto_lookup_bwd(plan));
}

static inline int vfft_proto_execute_bwd_ilout_jit(const stride_plan_t *plan,
                                                   double *src_re,
                                                   double *src_im, double *z,
                                                   size_t slice_K,
                                                   vfft_proto_exec_fn stages1)
{
    return vfft_proto_execute_bwd_ilout_core(plan, src_re, src_im, z, slice_K,
                                             stages1);
}

#endif /* VFFT_IL_EXECUTE_H */
