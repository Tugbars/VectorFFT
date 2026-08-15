/* il2p.h — PURE-IL two-pass K=1 route (bailey2 shape, interleaved end to end).
 *
 * z -> n1t(R2) -> z scratch -> t2(R1) -> z.  No split planes anywhere: every
 * intermediate is interleaved [re,im,re,im], 2 complex per ymm.
 *
 * ── WHY THIS REPLACES THE HYBRID 2P ROUTE ───────────────────────────────
 *
 * The incumbent VFFT_K1_IL_2P (oop_plan.h) is interleaved only at the API
 * boundary: il_leaf writes TWO SPLIT PLANES (p->col_re, p->col_im) and
 * t1_ul_il reads them back. Measured 2026-07-26, both arms gated against a
 * scalar DFT, this route beats it end to end:
 *
 *     N=64  (8x8)    hybrid 60.4 ns   pure IL 33.7 ns   0.558x
 *     N=256 (16x16)  hybrid 248.5     pure IL 190.2     0.765x
 *     N=1024(32x32)  hybrid 1796.4    pure IL 1717.3    0.956x  (wash)
 *
 * At the codelet level, with the WORKING SET HELD CONSTANT (the earlier sweep
 * confounded radix with working set), pure IL wins at every radix:
 *     R=4 0.510 | R=8 0.599 | R=16 0.657 | R=32 0.658 | R=64 0.894
 *
 * THE BOUNDARY: pure IL wins while the working chunk is L1-RESIDENT and
 * degrades past it. N=1024 is in+mid+out = 3*16 KB = 48 KB = exactly this
 * machine's L1d, and that cell measures a dead wash — the crossover sits
 * precisely where the mechanism predicts. Above it the block-split cascade
 * (zsplit.h) owns the range, which is also what MKL does and for the reason
 * its RE doc gives: "2 passes can't amortize a conversion; the high-N cascade
 * converts because log-many [passes can]".
 *
 * So the hybrid conversion was never justified at this tier: two passes cannot
 * pay a layout conversion back. It also was NOT derived from the MKL research
 * (docs/research/mkl_il_512_anatomy.md calls our split-plane staging "the exact
 * opposite" of MKL's mid-N path, which is interleaved throughout).
 *
 * ── STAGING (validated against a scalar DFT, not asserted) ───────────────
 *   n1t(R2): count=R1, Ls=R1, OLs=R2 — corner-turn fused into the stores, so
 *            element (leg p, col k) lands at mid[2*(k*R2 + p)].
 *   t2(R1) : count=R2, Ls=R2, OLs=R2 — reads that plane with leg=k, col=p
 *            (the four-step transpose), applies the streamed VTW2 twiddles.
 *   VTW2 record (col-pair pp, leg l) at tw + (pp*(R1-1) + (l-1))*8:
 *     [ c(k), c(k), c(k+1), c(k+1) ][ -s(k), +s(k), -s(k+1), +s(k+1) ],
 *   k = 2*pp, angle -2*pi*l*k/N. BYTW2 = fmadd(c, x, mul(s, cflip x)).
 *
 * BWD is TABLE-SIDE conjugated (codelet_cil.ml module card, gotcha 2): the
 * kernel's BYTW2 is bit-for-bit the forward one, only its position moves, so
 * the caller supplies a conjugated stream. Hence twb.
 */
#ifndef VFFT_IL2P_H
#define VFFT_IL2P_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifndef VFFT_IL2P_PI
#define VFFT_IL2P_PI 3.14159265358979323846
#endif

#if defined(_WIN32)
#include <malloc.h>
#define VFFT_IL2P_ALLOC(n) _aligned_malloc((n), 64)
#define VFFT_IL2P_FREE(p)  _aligned_free(p)
#else
#define VFFT_IL2P_ALLOC(n) aligned_alloc(64, (((n) + 63u) / 64u) * 64u)
#define VFFT_IL2P_FREE(p)  free(p)
#endif

typedef void (*vfft_il2p_fn)(const double *, const double *, double *, double *,
                             const double *, const double *,
                             size_t, size_t, size_t, size_t, size_t);

/* GENERATED REGISTRY (bin/emit_il_registry.ml -> generated/il_registry_avx2.h):
 * extern declarations for all 253 corpus-covered IL cells, plus the radix
 * X-macro lists the resolvers below expand.  Derived from Corpus, so "the
 * codelet exists" and "a resolver can reach it" cannot drift apart.
 * NOT covered, still declared by hand below: the 6 tangent kernels and the
 * blocked / sed-renamed variants (t2b, n1tb, n1tb44, t2b48, n1tb48) -- they
 * sit outside the corpus pending pool sunset / an emitter suffix knob. */
#include "il_registry_avx2.h"

/* t2 declarations: GENERATED (VFFT_IL_T2_{FWD,BWD}_RADICES). */

/* ── BLOCKED t2 mids (`--cil-blocked`, symbol tag `b`) — RACED CANDIDATES ─
 * Promoted 2026-08-03 from the twmem r32exp campaign (docs/research/
 * twmem_campaign/results/r32exp_blocked.md census + gate, r32exp_timing.md
 * v1-v3 timing). The blocked form splits the R1 DFT into two passes (m·p),
 * dropping peak live R1 -> max(m,p): the monolithic r32 t2's RA churn (26
 * multi-stored frame slots, 21.6% of body insns on ymm stack traffic)
 * collapses to 0-4 slots. Quiet-machine race: t2b48 [4·8] −18..−20% kernel
 * and −5..−14% through execute_fwd (WIN 3/3 both levels); t2b [2·16]
 * −25..−27% kernel, pipeline unresolved; t2b16 [2·8] confirmed over two
 * sessions. Side-finding: blocked kernels are immune to the per-process
 * stack-ASLR/4KB-alias tail-risk that inflates the spilling monolith.
 *
 * The winner is NEVER hand-set (sterm/sterm2 placement-luck lesson) — it
 * is MEASURED AT THE FRONT DOOR (bench_1d_vs_mkl.c builds handles per
 * variant and times them) and selected here via VFFT_IL2P_MID; create
 * itself does no timing. FWD-ONLY by design: execute_fwd is mid_f's
 * only consumer; the bwd path standardizes on t2t/F-DIAG and never reads a
 * mid twin (--cil-blocked could emit a bwd, but it would be an un-raced
 * orphan). CONTRACT: count % 2 == 0 — blocked kernels carry NO odd-count
 * tail (monolithic-only feature), so the race requires even R2.
 * Kill switch: VFFT_NO_T2B (VFFT_NO_IL2P precedent). */
#define VFFT_IL2P_DECL_T2B(SYM) \
  extern void SYM( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
VFFT_IL2P_DECL_T2B(radix16_z_t2b_fwd_avx2)
VFFT_IL2P_DECL_T2B(radix32_z_t2b_fwd_avx2)
VFFT_IL2P_DECL_T2B(radix32_z_t2b48_fwd_avx2)
#undef VFFT_IL2P_DECL_T2B

/* n1t declarations: GENERATED (VFFT_IL_N1T_{FWD,BWD,PAIR}_RADICES).
 * The groups below are kept as PROSE -- why each radix class exists -- but
 * the list itself is no longer written here.
 *   pow2: 4 8 16 32 64
/* even-composite leaves (2026-07-29, emitted via dft_small's mixed
 * recursion): unlock 2-stage pairs at 4·odd² N — 36=6x6, 100=10x10,
 * 144=12x12 — and even-composite chain leaves (300 = 6·(5·10)). */
 *   even composites: 6 10 12
/* odd leaves (2026-07-29, with the odd-count tail): all-odd pairs —
 * 45 = 9x5, 225 = 15x15, 675 = 27x25. Both stage counts go odd; the
 * inline VEX-128 tail carries them. */
 *   odd: 3 5 7 9 11 13 15 17 19 21 25 27 */
/* BLOCKED leaves (E9, 2026-08-05): the n1t corner-turn carried through
 * emit_blocked's pass-pairs. FWD-ONLY (leaf_b's only consumer is the F-DIAG
 * fallback) and radix-32 only for now — raced per cell at create like the
 * blocked mids; n1tb (2·16) is BITWISE-identical to n1t, n1tb48 (4·8) is
 * the tolerance class. */
extern void radix32_z_n1tb_fwd_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t, size_t, size_t, size_t, size_t);
extern void radix32_z_n1tb48_fwd_avx2(
    const double *, const double *, double *, double *,
    const double *, const double *, size_t, size_t, size_t, size_t, size_t);

/* n1t and t2 both cover 4..64 — the FULL set the K=1 IL pair search can select,
 * which is what keeps the il_in/il_out hybrid fallback unreachable.
 *
 * Radix 4 was added 2026-07-29. The old comment here read "a 4-leg corner-turn
 * leaf was never emitted", which was true but read as a limitation: the emitter
 * could always produce it (codelet_cil.ml's n1t refusal is on VECTOR WIDTH,
 * per<>2, NOT on radix) — the kernel had simply never been asked for. Its
 * absence left every pair with R2=4 (N=16 4x4, 32 8x4, 64 16x4, 128 32x4) with
 * no pure-IL plan, so execute silently used the hybrid.
 *
 * 🔴 Keep this list equal to the pair search's registry (vfft.c:3110-3124).
 * benches/il2p_bwd_gate.c asserts that equality by building all 25 pairs. */
static inline vfft_il2p_fn vfft_il2p_leaf_fn(int R, int bwd)
{
    switch (R) {
#define C(R) case R: return bwd ? radix##R##_z_n1t_bwd_avx2 : radix##R##_z_n1t_fwd_avx2;
    VFFT_IL_N1T_PAIR_RADICES(C)
#undef C
    default: return 0;
    }
}
/* Odd/prime t2 twins (conjugate-pair construction, codelet_cil.ml,
 * generated 2026-07-28): the 3-STAGE CHAIN's mid stages put odd factors
 * here as kernel RADICES (never as counts — that is the whole point of the
 * chain, docs/roadmap/il_odd_chain.md). The classic 2-stage pair search
 * never selects them (it requires both factors % 4 == 0), so extending
 * this registry does not change any pow2 route. */
/* odd / even-composite t2 declarations: GENERATED. */

static inline vfft_il2p_fn vfft_il2p_mid_fn(int R, int bwd)
{
    switch (R) {
#define C(R) case R: return bwd ? radix##R##_z_t2_bwd_avx2 : radix##R##_z_t2_fwd_avx2;
    VFFT_IL_T2_PAIR_RADICES(C)
#undef C
    default: return 0;
    }
}

/* t2tg — t2t's turned store with OGs wired as the LEG STRIDE (symbol tag
 * `tg`, emitted by `--cil-turnst-gs`): the chain BACKWARD's middle stage,
 * where leg groups from different calls interleave at stride A. */
/* t2tg declarations: GENERATED (VFFT_IL_T2TG_BWD_RADICES). */

static inline vfft_il2p_fn vfft_il2p_t2tg_bwd_fn(int R)
{
    switch (R) {
#define C(R) case R: return radix##R##_z_t2tg_bwd_avx2;
    VFFT_IL_T2TG_BWD_RADICES(C)
#undef C
    default: return 0;
    }
}

/* Plain n1 (natural in/out, TWIDDLE-FREE), radix R1 — the second stage of the
 * F-DIAG backward decomposition below. Distinct from leaf_fn (n1t, which fuses
 * the corner-turn into its stores) and from mid_fn (t2, which carries the
 * streamed VTW2 twiddle). */
/* n1 bwd declarations: GENERATED (VFFT_IL_N1_BWD_RADICES). */

static inline vfft_il2p_fn vfft_il2p_n1_bwd_fn(int R)
{
    switch (R) {
#define C(R) case R: return radix##R##_z_n1_bwd_avx2;
    VFFT_IL_N1_BWD_RADICES(C)
#undef C
    default: return 0;
    }
}

/* 🔴 t2p IS RETIRED — Tugbars, 2026-07-29: "disable t2p ... the whole tree
 * standardizes on t2t semantics." The t2p kind (PRE-twiddle + backward
 * butterfly + straight store, route A / conj-of-forward) lost the bwd race
 * at every R1 <= 32 and was kept only as a rival; to prevent the recurring
 * "which bwd arm?" confusion its registry, plan field, execute route, race
 * arm, and all 17 kernel files (pow2 + odd) were DELETED. F-DIAG below
 * remains the unfused reference of that same math. If a pre-twiddle kind is
 * ever needed again (the 3-stage odd chain's conj-of-forward composition
 * wanted one), that is a DELIBERATE decision — the sanctioned path is the
 * t2t-with-leg-stride store variant instead. */

/* t2t — POST-twiddle + backward butterfly + TURNED store: THE canonical bwd
 * flat codelet. Stage 1 of the decomposition that runs the R1 butterfly
 * FIRST. Emitted by `--cil-t2 --cil-bwd --cil-turnst`: store FORM is
 * independent of kind, which is the coupling that made this kernel
 * inexpressible.
 * All three of POST / TURNED / (Ls,OLs,count) below are FORCED by the
 * derivation, not chosen — perturbing any one gives O(1) error. */
/* t2t bwd declarations: GENERATED (VFFT_IL_T2T_BWD_RADICES). */

static inline vfft_il2p_fn vfft_il2p_t2t_bwd_fn(int R)
{
    switch (R) {
#define C(R) case R: return radix##R##_z_t2t_bwd_avx2;
    VFFT_IL_T2T_BWD_RADICES(C)
#undef C
    default: return 0;
    }
}

typedef struct {
    int N, R1, R2;
    double *mid;            /* interleaved scratch, 2N doubles */
    double *tw, *twb;       /* streamed VTW2 for t2: fwd and conjugated bwd */
    vfft_il2p_fn leaf_f, leaf_b;   /* n1t, radix R2 */
    vfft_il2p_fn mid_f,  mid_b;    /* t2,  radix R1 */
    vfft_il2p_fn n1_b;             /* plain n1 bwd, radix R1 (F-DIAG stage 2) */
    /* ⚠️ t2t's stage 2 is n1 bwd at radix R2, NOT R1. Using n1_b there is
     * a real trap — the control sweep measured it at 1.1e+00. */
    vfft_il2p_fn t2t_b;            /* post-tw + turned store, radix R1 (s1)   */
    vfft_il2p_fn n1_b_r2;          /* plain n1 bwd, radix R2        (s2)      */
} vfft_il2p_plan_t;

static inline void vfft_il2p_destroy(vfft_il2p_plan_t *p)
{
    if (!p) return;
    VFFT_IL2P_FREE(p->mid);
    VFFT_IL2P_FREE(p->tw);
    VFFT_IL2P_FREE(p->twb);
    free(p);
}

/* NULL when the pair has no pure-IL kernels, so a caller can fall back rather
 * than build a plan that cannot execute. Since 2026-07-29 there is NO parity
 * constraint: every monolithic cil kernel carries the inline VEX-128
 * odd-count tail, and the VTW2 table below ceils its pair count so an odd
 * R2's last (even-indexed) column has its record.
 *
 * 🔴 COVERAGE IS THE CONTRACT, NOT AN ACCIDENT.
 * This must succeed for EVERY (R1,R2) the caller's pair search can select —
 * otherwise execute silently falls back to the il_in/il_out hybrid route.
 * Do NOT reason about whether a given gap "can be reached in practice": that
 * answer depends on the ISA (`per`), the codelet registries and the search
 * bounds, so it is platform-specific and goes stale. Enforce coverage instead;
 * benches/il2p_bwd_gate.c asserts it exhaustively over the whole domain.
 *
 * The old `R2 < 8` bound was exactly such an accident: it had no structural
 * reason (R2=4 is even, and ntw = (R2/2)*(R1-1)*8 is well-formed), it simply
 * predated the radix-4 n1t kernel. It left N=16 (pair 4x4) on the hybrid. */
/* ── TANGENT-INTERIOR KERNELS (variant 3) ────────────────────────────────
 * 2026-08-11. Same transforms as the classic forms, different interior
 * arithmetic: rotations factored e^(-i.th) = cos(th)*(1 - i*tan(th)), the
 * shear left un-normalized and cos folded into the consuming butterfly's FMA
 * pair, so butterfly adds move off the FP-add ports onto the FMA ports.
 * Source + measured deltas:
 *   src/dag-fft-compiler/codelets/zil/avx2/pure_il/tangent/README.md
 *
 * FORWARD ONLY (no backward twins emitted yet) — same scope the blocked
 * forms already have, so apply_kv_forms/blocked_default, which only touch
 * mid_f/leaf_f, need no new guard.
 *
 * R8/R16 forms are MONOLITHIC emissions and carry the inline VEX-128
 * odd-count tail, so they are legal at any count; BOTH R32 forms are
 * blocked (split 2.16) and need the even-count gate.
 *
 * 2026-08-13 wing32 (A-1, docs/roadmap/r32_tangent_parity_plan.md): the
 * R32 mid is now radix32_z_t2bw32 (canonical-angle combine + ROTFMA;
 * supersedes t2btan216, -3.3..-5.5% both shapes) and the R32 LEAF EXISTS
 * again — radix32_z_n1tbw32 with the TURNED-128 store edge. The old
 * leaf's +32.4% kill was the paired permute2f128 store edge, not the
 * tangent interior; route (32,16) with this leaf ties the hand champion
 * (~302-305 ns at N=512, fft512_a0, 3 runs). */
extern void radix8_z_t2tan_fwd_avx2(const double *, const double *,
    double *, double *, const double *, const double *,
    size_t, size_t, size_t, size_t, size_t);
extern void radix8_z_n1ttan_fwd_avx2(const double *, const double *,
    double *, double *, const double *, const double *,
    size_t, size_t, size_t, size_t, size_t);
extern void radix16_z_t2tan_fwd_avx2(const double *, const double *,
    double *, double *, const double *, const double *,
    size_t, size_t, size_t, size_t, size_t);
extern void radix16_z_n1ttan_fwd_avx2(const double *, const double *,
    double *, double *, const double *, const double *,
    size_t, size_t, size_t, size_t, size_t);
extern void radix32_z_t2bw32_fwd_avx2(const double *, const double *,
    double *, double *, const double *, const double *,
    size_t, size_t, size_t, size_t, size_t);
extern void radix32_z_n1tbw32_fwd_avx2(const double *, const double *,
    double *, double *, const double *, const double *,
    size_t, size_t, size_t, size_t, size_t);

/* ── BLOCKED-KERNEL VARIANT REGISTRY ─────────────────────────────────────
 * 2026-08-05. Same role as vfft_il2p_leaf_fn / vfft_il2p_mid_fn above:
 * a pure (radix, variant) -> symbol lookup. NO selection policy, NO env,
 * NO timing. The VERDICT lives in wisdom (kind-3 `il_kv`, packed
 * mid | leaf<<4) and is applied by the front door after create; the
 * measurement that produced it is the bench's job.
 *
 * variant: 0 = monolithic registry kernel (return 0 -> caller keeps it)
 *          1 = blocked 2·16   2 = blocked 4·8   3 = TANGENT interior
 * Returns 0 for any (radix, variant) with no emitted kernel, so an
 * unsupported verdict degrades to the monolithic kernel — always correct.
 *
 * CONTRACT: blocked kernels carry NO odd-count tail. The mid runs at
 * count = R2 and the leaf at count = R1, so the caller must refuse a
 * blocked mid for odd R2 and a blocked leaf for odd R1 — the `count_ok`
 * argument makes that explicit at the call site rather than implicit. */
static inline vfft_il2p_fn vfft_il2p_mid_v_fn(int R1, int variant, int count_ok)
{
    if (!variant) return 0;
    if (variant == 3) {                 /* tangent interior */
        if (R1 == 8)  return radix8_z_t2tan_fwd_avx2;   /* monolithic: has  */
        if (R1 == 16) return radix16_z_t2tan_fwd_avx2;  /* the odd tail     */
        if (R1 == 32 && count_ok) return radix32_z_t2bw32_fwd_avx2; /* blocked wing32 */
        return 0;
    }
    if (!count_ok) return 0;
    if (R1 == 16 && variant == 1) return radix16_z_t2b_fwd_avx2;
    if (R1 == 32 && variant == 1) return radix32_z_t2b_fwd_avx2;
    if (R1 == 32 && variant == 2) return radix32_z_t2b48_fwd_avx2;
    return 0;
}

/* R=16 blocked leaf, 4·4 — the RACED winner (2026-08-06). All three splits
 * were emitted and benched against each other and against monolithic at
 * N=512 (pair 32x16, mid held at 4·8, 24 arms, alternating, core 2):
 *   4·4 = 362 ns  <  2·8 = 367  <  mono = 373  <  8·2 = 376  (medians)
 * 4·4's WORST arm (366) beats monolithic's BEST (371) — non-overlapping.
 * The two losers were deleted rather than kept as dead registry entries;
 * this header is the record. 🔴 8·2 is SLOWER THAN MONOLITHIC: the same
 * factorization transposed differs by 2.4%, which is why the split shape
 * is raced per ISA and never reasoned from the factorization alone.
 *
 * NOT a structural default: R=16 FITS the register file (8.6% ymm spill,
 * the census CONTROL class), so unlike R>=32 this is a wisdom-selected
 * pool candidate and MONOLITHIC n1t(16) remains the fallback. Its purpose
 * is to remove a real confound — blocked forms covered R=32 in both slots
 * but R=16 in the MID only, so the (16,32)-vs-(32,16) ordering race was
 * comparing orderings with different form coverage on each side. */
extern void radix16_z_n1tb44_fwd_avx2(const double *, const double *,
    double *, double *, const double *, const double *,
    size_t, size_t, size_t, size_t, size_t);

static inline vfft_il2p_fn vfft_il2p_leaf_v_fn(int R2, int variant, int count_ok)
{
    if (!variant) return 0;
    if (variant == 3) {                 /* tangent interior */
        if (R2 == 8)  return radix8_z_n1ttan_fwd_avx2;   /* monolithic   */
        if (R2 == 16) return radix16_z_n1ttan_fwd_avx2;  /* (odd legal)  */
        if (R2 == 32 && count_ok) return radix32_z_n1tbw32_fwd_avx2; /* blocked
            wing32, TURNED-128 store — the old kill was the store edge */
        return 0;
    }
    if (!count_ok) return 0;
    if (R2 == 32 && variant == 1) return radix32_z_n1tb_fwd_avx2;
    if (R2 == 32 && variant == 2) return radix32_z_n1tb48_fwd_avx2;
    if (R2 == 16 && variant == 1) return radix16_z_n1tb44_fwd_avx2; /* 4·4 */
    return 0;
}

/* kind-3 wisdom packing: il_kv = mid | leaf<<4.
 * Nibble 0xF = FORCE MONOLITHIC — needed since the structural default below
 * made blocked the R>=32 default: a platform where blocked measures slower
 * must stay expressible as a banked verdict (the pool stays full in BOTH
 * directions; this box is not the last word). */
#define VFFT_IL_KV_MID(kv)   ((kv) & 0xf)
#define VFFT_IL_KV_LEAF(kv)  (((kv) >> 4) & 0xf)
#define VFFT_IL_KV_PACK(m,l) (((m) & 0xf) | (((l) & 0xf) << 4))
#define VFFT_IL_KV_MONO      0xf

/* ── STRUCTURAL DEFAULT: blocked kernels ARE the R>=32 forward kernels ───
 *
 * Scope: FORWARD only (no blocked bwd twins exist yet) and even counts only
 * (blocked kernels carry NO odd-count tail — the monolithic kernel, which
 * does, remains the odd-count fallback). 4·8 forms preferred over 2·16:
 * measured dominant on the mid (pipeline -11..-21%, the only arm that
 * reproduced in every valid section of every run) and the register
 * arithmetic agrees (peak-live max(p,m): 4·8 < 2·16); 2·16 is the fallback
 * when no 4·8 form exists. R=16 is deliberately NOT in the rule — it fits
 * the register file (8.6% spill, the census control class); any r16 win is
 * cell-local and belongs to wisdom.
 *
 * VFFT_NO_ILBLK: create-time kill switch (VFFT_NO_ZTURN idiom) + the
 * bench's A/B hook through the front door. A boolean availability gate, not
 * a picker: no measurement, no timer, no verdict — the banned class stays
 * banned here. Wisdom il_kv OVERRIDES this default (vfft.c apply_kv runs
 * after create; 0xF forces monolithic). */
static inline void vfft_il2p_apply_blocked_default(vfft_il2p_plan_t *p)
{
    if (!p || getenv("VFFT_NO_ILBLK")) return;
    if (p->R1 >= 32 && (p->R2 & 1) == 0) {
        vfft_il2p_fn m = vfft_il2p_mid_v_fn(p->R1, 2, 1);   /* 4·8  */
        if (!m) m = vfft_il2p_mid_v_fn(p->R1, 1, 1);        /* 2·16 */
        if (m) p->mid_f = m;
    }
    if (p->R2 >= 32 && (p->R1 & 1) == 0) {
        vfft_il2p_fn l = vfft_il2p_leaf_v_fn(p->R2, 2, 1);  /* 4·8  */
        if (!l) l = vfft_il2p_leaf_v_fn(p->R2, 1, 1);       /* 2·16 */
        if (l) p->leaf_f = l;
    }
}

/* Apply an explicit il_kv FORM verdict onto a built plan — the ONE
 * definition of the nibble semantics, shared by vfft.c's wisdom apply and
 * the DP planner's variant-axis candidates (two copies of this logic is
 * the drift bug again). Deterministic, env-free. Nibble 0 = leave the
 * slot as create resolved it (structural default); VFFT_IL_KV_MONO (0xF)
 * = force the monolithic kernel back; else = the registry variant, parity
 * gated exactly like the default. */
static inline void vfft_il2p_apply_kv_forms(vfft_il2p_plan_t *p, int kv)
{
    if (!p || !kv) return;
    const int mv = VFFT_IL_KV_MID(kv), lv = VFFT_IL_KV_LEAF(kv);
    if (mv == VFFT_IL_KV_MONO)
        p->mid_f = vfft_il2p_mid_fn(p->R1, 0);
    else if (mv) {
        vfft_il2p_fn m = vfft_il2p_mid_v_fn(p->R1, mv, (p->R2 & 1) == 0);
        if (m) p->mid_f = m;
    }
    if (lv == VFFT_IL_KV_MONO)
        p->leaf_f = vfft_il2p_leaf_fn(p->R2, 0);
    else if (lv) {
        vfft_il2p_fn l = vfft_il2p_leaf_v_fn(p->R2, lv, (p->R1 & 1) == 0);
        if (l) p->leaf_f = l;
    }
}

static inline vfft_il2p_plan_t *vfft_il2p_create(int N, int R1, int R2)
{
    if (N <= 0 || R1 < 3 || R2 < 3 || (long)R1 * (long)R2 != (long)N) return 0;
    /* (The old (R1&1)||(R2&1) refusal is GONE — 2026-07-29, with the
     * odd-COUNT tail: every monolithic cil kernel now carries an inline
     * VEX-128 tail (il_odd_count_tail.md §3), so odd counts are legal and
     * all-odd pairs (45 = 9x5) become plans. Registry probes below remain
     * the availability filter. */
    vfft_il2p_fn lf = vfft_il2p_leaf_fn(R2, 0), lb = vfft_il2p_leaf_fn(R2, 1);
    vfft_il2p_fn mf = vfft_il2p_mid_fn(R1, 0),  mb = vfft_il2p_mid_fn(R1, 1);
    if (!lf || !lb || !mf || !mb) return 0;
    /* n1_b may be absent without invalidating the forward plan — only the
     * F-DIAG backward path needs it, and execute_bwd checks. */
    vfft_il2p_fn nb = vfft_il2p_n1_bwd_fn(R1);
    vfft_il2p_fn tt = vfft_il2p_t2t_bwd_fn(R1);
    vfft_il2p_fn nb2 = vfft_il2p_n1_bwd_fn(R2);   /* t2t stage 2 is radix R2 */

    vfft_il2p_plan_t *p = (vfft_il2p_plan_t *)calloc(1, sizeof(*p));
    if (!p) return 0;
    p->N = N; p->R1 = R1; p->R2 = R2;
    p->leaf_f = lf; p->leaf_b = lb; p->mid_f = mf; p->mid_b = mb;
    p->n1_b = nb;
    p->t2t_b = tt;
    p->n1_b_r2 = nb2;

    /* CEIL pair count: odd R2's last column (even index R2-1) reads record
     * pair (R2-1)/2 lane 0 from the tail's cursor — floor would under-
     * allocate by one record set. Lane 1 of that last record (column R2,
     * which does not exist) is filled with the k = R2 angle: valid values,
     * never read. */
    size_t npair = ((size_t)R2 + 1u) / 2u;
    size_t ntw = npair * (size_t)(R1 - 1) * 8u;
    p->mid = (double *)VFFT_IL2P_ALLOC((size_t)N * 2u * sizeof(double));
    p->tw  = (double *)VFFT_IL2P_ALLOC(ntw * sizeof(double));
    p->twb = (double *)VFFT_IL2P_ALLOC(ntw * sizeof(double));
    if (!p->mid || !p->tw || !p->twb) { vfft_il2p_destroy(p); return 0; }

    for (size_t pp = 0; pp < npair; pp++)
        for (int l = 1; l < R1; l++) {
            size_t off = (pp * (size_t)(R1 - 1) + (size_t)(l - 1)) * 8u;
            double *rf = p->tw + off, *rb = p->twb + off;
            for (int j = 0; j < 2; j++) {
                double k = (double)(2u * pp + (size_t)j);
                double a = -2.0 * VFFT_IL2P_PI * (double)l * k / (double)N;
                double c = cos(a), s = sin(a);
                rf[2 * j] = c;      rf[2 * j + 1] = c;
                rf[4 + 2 * j] = -s; rf[4 + 2 * j + 1] = s;
                /* bwd: conjugate the table, kernel arithmetic unchanged */
                rb[2 * j] = c;      rb[2 * j + 1] = c;
                rb[4 + 2 * j] = s;  rb[4 + 2 * j + 1] = -s;
            }
        }
    vfft_il2p_apply_blocked_default(p);
    return p;
}

static inline void vfft_il2p_execute_fwd(const vfft_il2p_plan_t *p,
                                         const double *zin, double *zout)
{
    const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
    p->leaf_f(zin, 0, p->mid, 0, 0, 0, R1, 0, R2, 0, R1);
    p->mid_f(p->mid, 0, zout, 0, p->tw, 0, R2, 0, R2, 0, R2);
}

/* ── F-DIAG: the unfused backward composition (reference + fallback) ─────
 *
 * Validated 2026-07-29 against the gated forward at 7 cells (1.89e-14
 * @N=128 16x8 .. 6.47e-13 @N=4096 64x64). Controls: deleting the diagonal,
 * or applying it POST instead of PRE, both give O(1) error.
 *
 * 🔴 IF THIS EVER NEEDS RE-DERIVING, DO NOT SCAN STRIDES. Eight compositions
 * were falsified that way first, all at O(1); the closest failing arm
 * differed from the correct one by ONE SEMANTIC BIT — stage-2 twiddle POST
 * vs PRE — with identical stages, radices, strides, table and order, so no
 * stride scan could ever have reached it. The old "the inverse needs an
 * un-turn and no emitted kernel un-turns" diagnosis was WRONG: it holds only
 * for the operator-inverse route, while this composition keeps the turn
 * exactly where the forward put it. Full record: memory
 * [[il2p-backward-solved]].
 *
 *   stage 1  leaf_b = n1t_bwd(R2), args IDENTICAL to forward stage 1
 *              mid[k*R2 + p] = IDFT_R2(column k)[p]
 *   diagonal PRE-multiply by e^{+2pi i * l * col / N}, legs 1..R1-1
 *   stage 2  n1_b = plain n1_bwd(R1), Ls = OLs = count = R2
 *
 * (Historical: fusing the diagonal into stage 2 was the t2p kind — bitwise
 * identical to this form at all 7 gated cells. t2p is RETIRED 2026-07-29;
 * F-DIAG stays as the unfused reference/fallback of that math.)
 *
 * ⚠️ GATE AT NON-SQUARE PAIRS. The two mirror decompositions coincide when
 * R1 == R2, so 256 (16x16) / 1024 (32x32) / 4096 (64x64) cannot adjudicate.
 * Use 128 (8x16) or 512 (16x32).
 *
 * Returns 0 on success, -1 if this build lacks the plain n1 bwd twin. */
static inline int vfft_il2p_execute_bwd_fdiag(const vfft_il2p_plan_t *p,
                                              const double *zin, double *zout)
{
    const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
    if (!p->n1_b) return -1;

    /* stage 1 — same call shape as the forward leaf, backward twin */
    p->leaf_b(zin, 0, p->mid, 0, 0, 0, R1, 0, R2, 0, R1);

    /* diagonal: mid[l*R2 + col] *= conj-twiddle, read from the SAME VTW2
     * records stage 2 would consume. Record layout (see create): per column
     * PAIR pp, per leg l in 1..R1-1, 8 doubles [c c c c][s -s s -s], lane
     * j = col & 1. BYTW2 semantics make the applied factor (c - i*s), i.e.
     * e^{+2pi i * l * col / N} for the bwd table. Leg 0 is w^0 = 1. */
    for (size_t l = 1; l < R1; l++)
        for (size_t col = 0; col < R2; col++) {
            const double *rb =
                p->twb + ((col >> 1) * (R1 - 1) + (l - 1)) * 8u;
            const size_t j = col & 1u;
            const double c = rb[2 * j], s = rb[4 + 2 * j];
            double *z = p->mid + 2 * (l * R2 + col);
            const double xr = z[0], xi = z[1];
            z[0] = c * xr + s * xi;
            z[1] = c * xi - s * xr;
        }

    /* stage 2 — plain backward butterfly, twiddle already applied */
    p->n1_b(p->mid, 0, zout, 0, 0, 0, R2, 0, R2, 0, R2);
    return 0;
}

/* (ROUTE A — the fused conj-of-forward composition via the t2p kernel — was
 * RETIRED AND DELETED 2026-07-29 with the t2p kind itself; see the
 * retirement note above the t2t registry. Its math survives as F-DIAG.) */

/* t2t — THE decomposition: run the R1 butterfly FIRST, then R2.
 * (The retired route A ran R2 first, mirroring the forward's stage order.)
 *
 * Derived 2026-07-29 by two blind derivations that produced the SAME triples —
 * nothing to adjudicate — and validated in a scalar simulator at 10 cells
 * including non-square in BOTH orders. Route A's own numbers were the control.
 *
 *   x[a*R1+b] = SUM_k e^{+2pi i ak/R2} e^{+2pi i bk/N} [ SUM_j X[j*R2+k] e^{+2pi i bj/R1} ]
 *               \____ stage 2, IDFT_R2 ___/ \_twiddle_/  \_____ stage 1, IDFT_R1 ______/
 *
 * A views the spectrum as K = a*R1 + b (R1 the fast stride); B takes the
 * OPPOSITE view on both index lines, K = alpha*R2 + beta and n = gamma*R1 + delta.
 *
 * 🔴 THREE THINGS ARE FORCED BY THE DERIVATION, NOT CHOSEN. A control sweep
 * perturbing one argument at a time gave O(1) error for EVERY perturbation
 * (0.54 .. 1.37), so this triple is pinned, not one of a family:
 *   - twiddle POST, not PRE — the factor e^{+2pi i bk/N} depends on b, the R1
 *     butterfly's OUTPUT leg; a pre-twiddle would index the input leg.
 *   - store TURNED, not straight.
 *   - (Ls,OLs,count) exactly as below; swapping counts, radices, or any stride
 *     all fail at O(1).
 *
 * ⚠️ STAGE 2 IS n1_bwd AT RADIX R2, NOT R1. Using p->n1_b (the R1 twin) here
 * measures 1.1e+00 — the control sweep flagged it explicitly as a trap.
 *
 * The table is p->twb UNCHANGED — same pointer route A's stage 2 takes, same
 * cursor convention. Consumption is exactly ntw = (R2/2)*(R1-1)*8, verified an
 * EXACT fit (no overread) under ASan at 10 cells. No new table, no new alloc.
 *
 * COVERAGE: B works where A cannot. A's stage 1 needs an n1t leaf at radix R2,
 * which does not exist at R2=4; B was validated at 128=32x4 and 64=16x4 where
 * route A is unavailable.
 *
 * Returns 0 on success, -1 if this build lacks the twins. */
static inline int vfft_il2p_execute_bwd_t2t(const vfft_il2p_plan_t *p,
                                            const double *zin, double *zout)
{
    const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
    if (!p->t2t_b || !p->n1_b_r2) return -1;
    p->t2t_b(zin, 0, p->mid, 0, p->twb, 0, R2, 0, R1, 0, R2);
    p->n1_b_r2(p->mid, 0, zout, 0, 0, 0, R1, 0, R1, 0, R1);
    return 0;
}

/* ── THE BACKWARD PATH ───────────────────────────────────────────────────
 * t2t, THE canonical bwd composition (Tugbars 2026-07-29: t2p retired
 * everywhere, "the whole tree standardizes on t2t semantics").
 *
 * The original race record (three independent runs of il2p_bwd_gate.c,
 * kept for history): the winner tracked R1 — t2t won 2-14% at R1 <= 32,
 * the retired t2p arm won 1-10% at R1 == 64 only, 32x32 unresolvable.
 * t2t's stage 1 IS the R1 butterfly (turned store), so a fat R1 makes it
 * pay early. Tugbars' call: IL plans favour many small stages, so R1=64 is
 * rare ⇒ t2t. It also covers strictly more pairs than t2p did (t2p's
 * stage 1 needed an n1t leaf that never existed at R2=4).
 *
 * ⚠️ A SINGLE RUN WOULD HAVE MISLED — the first race read 9/10 one way and
 * did not reproduce. Always repeat races before re-deciding.
 *
 * F-DIAG is the availability fallback ONLY (unfused reference of the
 * retired route-A math — correctness net for a build lacking the t2t
 * twins, never a speed arm). */
static inline int vfft_il2p_execute_bwd(const vfft_il2p_plan_t *p,
                                        const double *zin, double *zout)
{
    if (vfft_il2p_execute_bwd_t2t(p, zin, zout) == 0) return 0;
    return vfft_il2p_execute_bwd_fdiag(p, zin, zout);
}

/* ═══════════════════════════════════════════════════════════════════════
 * il3p — the 3-STAGE pure-IL chain: N = R2 · A · B (R1 = A·B), the route
 * that gives odd/prime factors a K=1 IL plan (docs/roadmap/il_odd_chain.md).
 *
 * WHY 3 STAGES: every cil kernel vectorizes 2 complex/ymm and requires
 * count % 2 == 0. In the 2-stage pair the leaf runs at count=R1 and the
 * mid at count=R2 — BOTH factors must be even, so a 2-stage plan can NEVER
 * host an odd factor. The chain pins the SIMD axis to the leaf's q columns
 * (count = R2 at both mid stages, R1 at the leaf — all even) and odd
 * factors appear only as kernel RADICES. No odd-count tail exists or is
 * needed on this route.
 *
 * FORWARD (gated 12/12 vs naive DFT, real kernels):
 *   S1  n1t(R2), 1 call:  in zin (Ls=R1), out mid1 (OLs=R2), count=R1
 *   S2a t2(B), A calls c: in mid1+2cR2 (Ls=A·R2), out mid2+2cR2
 *                         (OLs=A·R2), count=R2, tw = VTW2(B, R2, B·R2)
 *   S2b t2(A), B calls b: in mid2+2bAR2 (Ls=R2), out zout+2bR2
 *                         (OLs=B·R2), count=R2,
 *                         tw = VTW2(A, B·R2, N) + region b·R2
 *   ⚠ S2b's twiddle argument is the COMBINED index q + b·R2 — ONE big
 *   table over all B·R2 columns; dropping the ω_{R1}^{cb} factor fails
 *   O(1) at every cell including the pow2 control (recorded in the doc).
 *
 * BACKWARD (gated 13/13 vs naive IDFT; t2t semantics — t2p is retired):
 *   B1  t2_bwd(A), B calls b:  in zin+2bR2 (Ls=B·R2), out mid2+2bAR2
 *                              (OLs=R2), count=R2, tw = conj big + region b
 *   B2  t2tg_bwd(B), A calls c: in mid2+2cR2 (Ls=A·R2), out mid1+2c
 *                              (OLs=R1, OGs=A — LEG-STRIDED turn),
 *                              count=R2, tw = conj VTW2(B, R2, B·R2)
 *   B3  n1_bwd(R2), 1 call:    in mid1 (Ls=R1), out zout (OLs=R1),
 *                              count=R1 — NATURAL, unnormalized (N·x)
 *
 * zin == zout is safe both directions (each stage fully consumes its input
 * before the next writes; the boundary stages touch the caller buffers).
 *
 * 🔴 The chain (R2, A, B) is a PLAN INPUT. vfft_il3p_default_chain below is
 * a LEGAL default for uncalibrated cells only — the measured per-cell pick
 * belongs to the wisdom campaign (plans come from measured search). */
typedef struct {
    int N, R2, A, B;               /* R1 = A*B */
    double *mid1, *mid2;           /* interleaved scratch, 2N doubles each */
    double *twB, *twA;             /* fwd: S2a table; S2b BIG table         */
    double *twAc, *twBc;           /* bwd: B1 BIG conj table; B2 conj table */
    vfft_il2p_fn leaf_f, n1_b;     /* n1t(R2) fwd; n1(R2) bwd               */
    vfft_il2p_fn tA_f, tB_f;       /* t2(A), t2(B) fwd                      */
    vfft_il2p_fn tA_b, tBg_b;      /* t2(A) bwd; t2tg(B) bwd                */
} vfft_il3p_plan_t;

static inline void vfft_il3p_destroy(vfft_il3p_plan_t *p)
{
    if (!p) return;
    VFFT_IL2P_FREE(p->mid1);
    VFFT_IL2P_FREE(p->mid2);
    VFFT_IL2P_FREE(p->twB);
    VFFT_IL2P_FREE(p->twA);
    VFFT_IL2P_FREE(p->twAc);
    VFFT_IL2P_FREE(p->twBc);
    free(p);
}

/* VTW2 fill, (legs, cols, modulus)-parametric — same record convention as
 * the 2-stage create above: (pair pp, leg l) at (pp*(legs-1)+(l-1))*8,
 * [c,c,c,c][-s,+s,-s,+s], angle -2*pi*l*k/modulus. conj flips the sins. */
static inline double *_vfft_il3p_vtw2(int legs, int cols, int modulus, int conj)
{
    size_t nrec = ((size_t)cols / 2u) * (size_t)(legs - 1);
    double *tw = (double *)VFFT_IL2P_ALLOC(nrec * 8u * sizeof(double));
    if (!tw) return 0;
    for (int pp = 0; pp < cols / 2; pp++)
        for (int l = 1; l < legs; l++) {
            double *rf = tw + ((size_t)pp * (legs - 1) + (l - 1)) * 8u;
            for (int j = 0; j < 2; j++) {
                double k = (double)(2 * pp + j);
                double a = -2.0 * VFFT_IL2P_PI * (double)l * k / (double)modulus;
                double s = conj ? sin(a) : -sin(a);
                rf[2 * j] = cos(a);
                rf[2 * j + 1] = cos(a);
                rf[4 + 2 * j] = s;
                rf[4 + 2 * j + 1] = -s;
            }
        }
    return tw;
}

/* LEGAL default chain for an uncalibrated cell (⚠ default, NOT a measured
 * plan): a covered leaf R2 (pow2 preferred, then even-composite) whose
 * cofactor R1 = N/R2 is even and splits as A·B with both mid kernels
 * present — first as odd·pow2, else with an even-composite B (6/10/12),
 * which serves the single-4 cells like 200 = 4·(5·10), 300 = 6·(5·10).
 * Returns 0 when no chain exists (pure pow2 N — the pair owns it; all-odd
 * N — every count odd). */
static inline int vfft_il3p_default_chain(int N, int *R2, int *A, int *B)
{
    static const int LEAF[] = { 32, 16, 8, 4, 12, 10, 6 };
    static const int ECB[]  = { 12, 10, 6 };
    for (int i = 0; i < 7; i++) {
        int r2 = LEAF[i];
        if (N % r2) continue;
        int R1 = N / r2;
        if (R1 < 4 || (R1 & 1)) continue;
        int o = R1;
        while ((o & 1) == 0) o >>= 1;      /* odd part */
        int pb = R1 / o;                   /* pow2 part */
        if (o == 1) continue;              /* pure pow2: the pair route owns it */
        if (pb >= 4 &&
            vfft_il2p_mid_fn(o, 0) && vfft_il2p_mid_fn(o, 1) &&
            vfft_il2p_mid_fn(pb, 0) && vfft_il2p_t2tg_bwd_fn(pb)) {
            *R2 = r2; *A = o; *B = pb;
            return 1;
        }
        /* single-2 cofactor (pb == 2) or uncovered odd part: try an
         * even-composite B so the lone factor of 2 rides inside it. */
        for (int j = 0; j < 3; j++) {
            int b = ECB[j];
            if (R1 % b) continue;
            int a = R1 / b;
            if (a < 3) continue;           /* no radix-2 mids */
            if (!vfft_il2p_mid_fn(a, 0) || !vfft_il2p_mid_fn(a, 1)) continue;
            if (!vfft_il2p_mid_fn(b, 0) || !vfft_il2p_t2tg_bwd_fn(b)) continue;
            *R2 = r2; *A = a; *B = b;
            return 1;
        }
    }
    return 0;
}

/* NULL when any kernel or table is unavailable — the caller falls back
 * (route truthfulness: a chain route always names a runnable plan). */
static inline vfft_il3p_plan_t *vfft_il3p_create(int N, int R2, int A, int B)
{
    const int R1 = A * B;
    if (N <= 0 || (long)R1 * (long)R2 != (long)N) return 0;
    if ((R1 & 1) || (R2 & 1)) return 0;    /* count contracts, both stages */
    vfft_il2p_fn lf  = vfft_il2p_leaf_fn(R2, 0);
    vfft_il2p_fn nb  = vfft_il2p_n1_bwd_fn(R2);
    vfft_il2p_fn af  = vfft_il2p_mid_fn(A, 0), ab = vfft_il2p_mid_fn(A, 1);
    vfft_il2p_fn bf  = vfft_il2p_mid_fn(B, 0);
    vfft_il2p_fn btg = vfft_il2p_t2tg_bwd_fn(B);
    if (!lf || !nb || !af || !ab || !bf || !btg) return 0;

    vfft_il3p_plan_t *p = (vfft_il3p_plan_t *)calloc(1, sizeof(*p));
    if (!p) return 0;
    p->N = N; p->R2 = R2; p->A = A; p->B = B;
    p->leaf_f = lf; p->n1_b = nb;
    p->tA_f = af; p->tB_f = bf;
    p->tA_b = ab; p->tBg_b = btg;
    p->mid1 = (double *)VFFT_IL2P_ALLOC((size_t)N * 2u * sizeof(double));
    p->mid2 = (double *)VFFT_IL2P_ALLOC((size_t)N * 2u * sizeof(double));
    p->twB  = _vfft_il3p_vtw2(B, R2, B * R2, 0);
    p->twA  = _vfft_il3p_vtw2(A, B * R2, N, 0);
    p->twAc = _vfft_il3p_vtw2(A, B * R2, N, 1);
    p->twBc = _vfft_il3p_vtw2(B, R2, B * R2, 1);
    if (!p->mid1 || !p->mid2 || !p->twB || !p->twA || !p->twAc || !p->twBc) {
        vfft_il3p_destroy(p);
        return 0;
    }
    /* Structural blocked default, LEAF only (same rule + kill switch as
     * vfft_il2p_apply_blocked_default; R1 = A*B is even by the count-
     * contract guard above, so the leaf's count parity is guaranteed).
     * The mids run radices A,B — small cofactors the registry has no
     * blocked twins for; nothing to select there. */
    if (!getenv("VFFT_NO_ILBLK") && R2 >= 32) {
        vfft_il2p_fn bl = vfft_il2p_leaf_v_fn(R2, 2, 1);    /* 4·8  */
        if (!bl) bl = vfft_il2p_leaf_v_fn(R2, 1, 1);        /* 2·16 */
        if (bl) p->leaf_f = bl;
    }
    return p;
}

static inline void vfft_il3p_execute_fwd(const vfft_il3p_plan_t *p,
                                         const double *zin, double *zout)
{
    const size_t R2 = (size_t)p->R2, A = (size_t)p->A, B = (size_t)p->B;
    const size_t R1 = A * B;
    p->leaf_f(zin, 0, p->mid1, 0, 0, 0, R1, 0, R2, 0, R1);
    for (size_t c = 0; c < A; c++)
        p->tB_f(p->mid1 + 2 * c * R2, 0, p->mid2 + 2 * c * R2, 0,
                p->twB, 0, A * R2, 0, A * R2, 0, R2);
    for (size_t b = 0; b < B; b++)
        p->tA_f(p->mid2 + 2 * b * A * R2, 0, zout + 2 * b * R2, 0,
                p->twA + (b * R2 / 2u) * (A - 1) * 8u, 0,
                R2, 0, B * R2, 0, R2);
}

static inline void vfft_il3p_execute_bwd(const vfft_il3p_plan_t *p,
                                         const double *zin, double *zout)
{
    const size_t R2 = (size_t)p->R2, A = (size_t)p->A, B = (size_t)p->B;
    const size_t R1 = A * B;
    for (size_t b = 0; b < B; b++)
        p->tA_b(zin + 2 * b * R2, 0, p->mid2 + 2 * b * A * R2, 0,
                p->twAc + (b * R2 / 2u) * (A - 1) * 8u, 0,
                B * R2, 0, R2, 0, R2);
    for (size_t c = 0; c < A; c++)
        p->tBg_b(p->mid2 + 2 * c * R2, 0, p->mid1 + 2 * c, 0,
                 p->twBc, 0, A * R2, 0, R1, A, R2);
    p->n1_b(p->mid1, 0, zout, 0, 0, 0, R1, 0, R1, 0, R1);
}

#endif /* VFFT_IL2P_H */
