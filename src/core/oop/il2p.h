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

#define VFFT_IL2P_DECL(R) \
  extern void radix##R##_z_t2_fwd_avx2( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
  extern void radix##R##_z_t2_bwd_avx2( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
VFFT_IL2P_DECL(4) VFFT_IL2P_DECL(8) VFFT_IL2P_DECL(16)
VFFT_IL2P_DECL(32) VFFT_IL2P_DECL(64)
#undef VFFT_IL2P_DECL

#define VFFT_IL2P_DECL_LEAF(R) \
  extern void radix##R##_z_n1t_fwd_avx2( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
  extern void radix##R##_z_n1t_bwd_avx2( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
VFFT_IL2P_DECL_LEAF(4)
VFFT_IL2P_DECL_LEAF(8) VFFT_IL2P_DECL_LEAF(16)
VFFT_IL2P_DECL_LEAF(32) VFFT_IL2P_DECL_LEAF(64)
/* even-composite leaves (2026-07-29, emitted via dft_small's mixed
 * recursion): unlock 2-stage pairs at 4·odd² N — 36=6x6, 100=10x10,
 * 144=12x12 — and even-composite chain leaves (300 = 6·(5·10)). */
VFFT_IL2P_DECL_LEAF(6) VFFT_IL2P_DECL_LEAF(10) VFFT_IL2P_DECL_LEAF(12)
#undef VFFT_IL2P_DECL_LEAF

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
    C(4) C(8) C(16) C(32) C(64)
    C(6) C(10) C(12)
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
#define VFFT_IL2P_DECL_ODD_T2(R) \
  extern void radix##R##_z_t2_fwd_avx2( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t); \
  extern void radix##R##_z_t2_bwd_avx2( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
VFFT_IL2P_DECL_ODD_T2(3)  VFFT_IL2P_DECL_ODD_T2(5)  VFFT_IL2P_DECL_ODD_T2(7)
VFFT_IL2P_DECL_ODD_T2(9)  VFFT_IL2P_DECL_ODD_T2(11) VFFT_IL2P_DECL_ODD_T2(13)
VFFT_IL2P_DECL_ODD_T2(15) VFFT_IL2P_DECL_ODD_T2(17) VFFT_IL2P_DECL_ODD_T2(19)
VFFT_IL2P_DECL_ODD_T2(21) VFFT_IL2P_DECL_ODD_T2(25) VFFT_IL2P_DECL_ODD_T2(27)
VFFT_IL2P_DECL_ODD_T2(6)  VFFT_IL2P_DECL_ODD_T2(10) VFFT_IL2P_DECL_ODD_T2(12)
#undef VFFT_IL2P_DECL_ODD_T2

static inline vfft_il2p_fn vfft_il2p_mid_fn(int R, int bwd)
{
    switch (R) {
#define C(R) case R: return bwd ? radix##R##_z_t2_bwd_avx2 : radix##R##_z_t2_fwd_avx2;
    C(4) C(8) C(16) C(32) C(64)
    C(3) C(5) C(7) C(9) C(11) C(13) C(15) C(17) C(19) C(21) C(25) C(27)
    C(6) C(10) C(12)
#undef C
    default: return 0;
    }
}

/* t2tg — t2t's turned store with OGs wired as the LEG STRIDE (symbol tag
 * `tg`, emitted by `--cil-turnst-gs`): the chain BACKWARD's middle stage,
 * where leg groups from different calls interleave at stride A. */
#define VFFT_IL2P_DECL_T2TG(R) \
  extern void radix##R##_z_t2tg_bwd_avx2( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
VFFT_IL2P_DECL_T2TG(3)  VFFT_IL2P_DECL_T2TG(4)  VFFT_IL2P_DECL_T2TG(5)
VFFT_IL2P_DECL_T2TG(7)  VFFT_IL2P_DECL_T2TG(8)  VFFT_IL2P_DECL_T2TG(9)
VFFT_IL2P_DECL_T2TG(11) VFFT_IL2P_DECL_T2TG(13) VFFT_IL2P_DECL_T2TG(15)
VFFT_IL2P_DECL_T2TG(16) VFFT_IL2P_DECL_T2TG(17) VFFT_IL2P_DECL_T2TG(19)
VFFT_IL2P_DECL_T2TG(21) VFFT_IL2P_DECL_T2TG(25) VFFT_IL2P_DECL_T2TG(27)
VFFT_IL2P_DECL_T2TG(32) VFFT_IL2P_DECL_T2TG(64)
VFFT_IL2P_DECL_T2TG(6)  VFFT_IL2P_DECL_T2TG(10) VFFT_IL2P_DECL_T2TG(12)
#undef VFFT_IL2P_DECL_T2TG

static inline vfft_il2p_fn vfft_il2p_t2tg_bwd_fn(int R)
{
    switch (R) {
#define C(R) case R: return radix##R##_z_t2tg_bwd_avx2;
    C(3) C(4) C(5) C(7) C(8) C(9) C(11) C(13) C(15) C(16) C(17) C(19)
    C(21) C(25) C(27) C(32) C(64) C(6) C(10) C(12)
#undef C
    default: return 0;
    }
}

/* Plain n1 (natural in/out, TWIDDLE-FREE), radix R1 — the second stage of the
 * F-DIAG backward decomposition below. Distinct from leaf_fn (n1t, which fuses
 * the corner-turn into its stores) and from mid_fn (t2, which carries the
 * streamed VTW2 twiddle). */
#define C(R) \
  extern void radix##R##_z_n1_bwd_avx2( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
    C(4) C(8) C(16) C(32) C(64) C(6) C(10) C(12)
#undef C

static inline vfft_il2p_fn vfft_il2p_n1_bwd_fn(int R)
{
    switch (R) {
#define C(R) case R: return radix##R##_z_n1_bwd_avx2;
    C(4) C(8) C(16) C(32) C(64) C(6) C(10) C(12)
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
#define C(R) \
  extern void radix##R##_z_t2t_bwd_avx2( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
    C(4) C(8) C(16) C(32) C(64) C(6) C(10) C(12)
#undef C

static inline vfft_il2p_fn vfft_il2p_t2t_bwd_fn(int R)
{
    switch (R) {
#define C(R) case R: return radix##R##_z_t2t_bwd_avx2;
    C(4) C(8) C(16) C(32) C(64) C(6) C(10) C(12)
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
 * than build a plan that cannot execute. R2 must be even (the leaf's
 * count%2==0 contract is on R1; the VTW2 stream indexes column PAIRS of R2).
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
static inline vfft_il2p_plan_t *vfft_il2p_create(int N, int R1, int R2)
{
    if (N <= 0 || R1 < 4 || R2 < 4 || (long)R1 * (long)R2 != (long)N) return 0;
    if ((R1 & 1) || (R2 & 1)) return 0;
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

    size_t ntw = ((size_t)R2 / 2u) * (size_t)(R1 - 1) * 8u;
    p->mid = (double *)VFFT_IL2P_ALLOC((size_t)N * 2u * sizeof(double));
    p->tw  = (double *)VFFT_IL2P_ALLOC(ntw * sizeof(double));
    p->twb = (double *)VFFT_IL2P_ALLOC(ntw * sizeof(double));
    if (!p->mid || !p->tw || !p->twb) { vfft_il2p_destroy(p); return 0; }

    for (size_t pp = 0; pp < (size_t)R2 / 2u; pp++)
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
    return p;
}

static inline void vfft_il2p_execute_fwd(const vfft_il2p_plan_t *p,
                                         const double *zin, double *zout)
{
    const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
    p->leaf_f(zin, 0, p->mid, 0, 0, 0, R1, 0, R2, 0, R1);
    p->mid_f(p->mid, 0, zout, 0, p->tw, 0, R2, 0, R2, 0, R2);
}

/* 🔴 NOT YET CORRECT — DO NOT WIRE. Returns -1 so no caller can silently get
 * wrong data; the forward path above is gated (fwd-vs-scalar 2e-14..1.5e-12
 * at 64/128/256/512/1024/4096, square and non-square pairs alike).
 *
 * WHAT IS WRONG: the naive body below (leaf_b then mid_b, forward order,
 * conjugated table) measures a roundtrip error of ~2.0 — O(1), i.e. not a
 * conjugation slip but a structural one. The inverse of (leaf -> mid) is
 * (mid^-1 -> leaf^-1), so the STAGES MUST RUN IN REVERSE ORDER.
 *
 * WHAT STILL NEEDS DECIDING (do not guess): the forward leaf fuses the
 * corner-turn into its STORES, writing (leg p, col k) to mid[2*(k*R2 + p)].
 * Inverting that needs a corner-turn in the LOADS, and it is not established
 * whether radixR_z_n1t_bwd_avx2 does that or simply repeats the store-side
 * turn with an inverse butterfly. Read the emitted bwd source (or
 * codelet_cil.ml's N1T bwd path) before writing this.
 *
 * The per-kernel backward twins are themselves fine — build_tuned/benches/
 * cil_bwd_gate.c roundtrips t2_fwd/t2_bwd at identical strides with a
 * conjugated table. The defect is purely this route's stage composition.
 *
 * ── 8 COMPOSITIONS ALREADY FALSIFIED (2026-07-26). DO NOT RETRY. ────────
 * Measured at N=128, R1=16, R2=8 (non-square, so a radix swap is visible).
 * All roundtrip errors are O(1) — structural, not a conjugation slip:
 *   leaf_b(R2) -> mid_b(R1), fwd strides, conj table ......... 1.888
 *   mid_b(R1)  -> leaf_b(R2), reversed order ................. 2.025
 *   leaf_b(R1) -> mid_b(R2), swapped radices ................. 2.085
 * and with the PLAIN n1_bwd (no corner-turn) as stage 2, after mid_b(R1):
 *   n1_b Ls=1  OLs=R1 cnt=R1 ................................. 2.195
 *   n1_b Ls=R2 OLs=R1 cnt=R1 ................................. 1.993
 *   n1_b Ls=1  OLs=R1 cnt=R2 ................................. 1.952
 *   n1_b Ls=R2 OLs=1  cnt=R1 ................................. 2.222
 *   n1_b Ls=R1 OLs=R1 cnt=R1 ................................. 2.072
 *
 * READ THIS BEFORE THE NEXT ATTEMPT: the failure is NOT stride assignment —
 * five different stride triples on the same structure all fail alike. The
 * forward stage 1 does (DFT_R2 down columns) THEN (corner-turn in stores);
 * its inverse is (un-turn) THEN (IDFT_R2), and NO emitted kernel un-turns.
 *
 * MOST PROMISING UNEXPLORED DIRECTION: the swapped-radix arm above was tested
 * with only ONE stride triple. An inverse four-step naturally exchanges which
 * factor indexes columns, so bwd may legitimately be n1t(R1) -> t2(R2) with a
 * table built for radix R2 over R1 columns — i.e. the structure was right and
 * only the strides were wrong. Scan that arm's stride space before adding a
 * new codelet kind. Derive the index map from the fwd identity
 *   mid[2*(k*R2 + p)] = DFT_R2(column k)[p],  k in [0,R1), p in [0,R2)
 * rather than guessing triples. */
/* ── F-DIAG: the SOLVED backward composition ─────────────────────────────
 *
 * Derived 2026-07-29 by two BLIND derivations (first-principles and
 * artifact-side) that came out formula-identical on the index map, then
 * validated by a scalar simulator against the gated forward at 7 cells
 * (1.89e-14 @N=128 16x8 ... 6.47e-13 @N=4096 64x64). Controls: deleting the
 * diagonal, or applying it POST instead of PRE, both give O(1) error.
 *
 * 🔴 WHY THE OLD DIAGNOSIS WAS WRONG. The comment above says the inverse needs
 * an "un-turn" and that no emitted kernel un-turns. True only for the
 * OPERATOR-inverse route. This composition keeps the turn exactly where the
 * forward put it and needs no un-turn at all.
 *
 * 🔴 THE 8 FALSIFIED ARMS WERE ONE BIT AWAY. Arm #1 (leaf_b(R2) -> mid_b(R1),
 * fwd strides, conj table, err 1.888) differs from this ONLY in that the
 * stage-2 twiddle is applied POST (t2 bwd) instead of PRE. Same stages, same
 * radices, same strides, same table, same order, same arguments. That is why
 * no stride scan could ever have found it.
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
