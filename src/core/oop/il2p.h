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
VFFT_IL2P_DECL_LEAF(8) VFFT_IL2P_DECL_LEAF(16)
VFFT_IL2P_DECL_LEAF(32) VFFT_IL2P_DECL_LEAF(64)
#undef VFFT_IL2P_DECL_LEAF

/* n1t exists for 8..64 (a 4-leg corner-turn leaf was never emitted); t2 for
 * 4..64. The resolvers return 0 outside that reach so callers degrade. */
static inline vfft_il2p_fn vfft_il2p_leaf_fn(int R, int bwd)
{
    switch (R) {
#define C(R) case R: return bwd ? radix##R##_z_n1t_bwd_avx2 : radix##R##_z_n1t_fwd_avx2;
    C(8) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
}
static inline vfft_il2p_fn vfft_il2p_mid_fn(int R, int bwd)
{
    switch (R) {
#define C(R) case R: return bwd ? radix##R##_z_t2_bwd_avx2 : radix##R##_z_t2_fwd_avx2;
    C(4) C(8) C(16) C(32) C(64)
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
    C(4) C(8) C(16) C(32) C(64)
#undef C

static inline vfft_il2p_fn vfft_il2p_n1_bwd_fn(int R)
{
    switch (R) {
#define C(R) case R: return radix##R##_z_n1_bwd_avx2;
    C(4) C(8) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
}

/* t2p — PRE-twiddle + backward butterfly + STRAIGHT store. This is the F-DIAG
 * decomposition with the diagonal FUSED into stage 2. Emitted by
 * `--cil-t2 --cil-bwd --cil-pretw`: twiddle POSITION is independent of
 * direction, which is the coupling that made this kernel inexpressible before.
 * Fusing removes an extra read+write of the whole scratch plane and does the
 * multiply in-register; the unfused diagonal measures 26-56% of backward. */
#define C(R) \
  extern void radix##R##_z_t2p_bwd_avx2( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
    C(4) C(8) C(16) C(32) C(64)
#undef C

static inline vfft_il2p_fn vfft_il2p_t2p_bwd_fn(int R)
{
    switch (R) {
#define C(R) case R: return radix##R##_z_t2p_bwd_avx2;
    C(4) C(8) C(16) C(32) C(64)
#undef C
    default: return 0;
    }
}

/* t2t — POST-twiddle + backward butterfly + TURNED store. Stage 1 of the RIVAL
 * decomposition (route B), which runs the R1 butterfly FIRST. Emitted by
 * `--cil-t2 --cil-bwd --cil-turnst`: store FORM is independent of kind, which
 * is the coupling that made this kernel inexpressible.
 * All three of POST / TURNED / (Ls,OLs,count) below are FORCED by the
 * derivation, not chosen — perturbing any one gives O(1) error. */
#define C(R) \
  extern void radix##R##_z_t2t_bwd_avx2( \
      const double *, const double *, double *, double *, \
      const double *, const double *, size_t, size_t, size_t, size_t, size_t);
    C(4) C(8) C(16) C(32) C(64)
#undef C

static inline vfft_il2p_fn vfft_il2p_t2t_bwd_fn(int R)
{
    switch (R) {
#define C(R) case R: return radix##R##_z_t2t_bwd_avx2;
    C(4) C(8) C(16) C(32) C(64)
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
    vfft_il2p_fn t2p_b;            /* fused pre-tw bwd, radix R1 (route A)    */
    /* ⚠️ route B's stage 2 is n1 bwd at radix R2, NOT R1. Using n1_b there is
     * a real trap — the control sweep measured it at 1.1e+00. */
    vfft_il2p_fn t2t_b;            /* post-tw + turned store, radix R1 (B s1) */
    vfft_il2p_fn n1_b_r2;          /* plain n1 bwd, radix R2      (B s2)      */
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
 * count%2==0 contract is on R1; the VTW2 stream indexes column PAIRS of R2). */
static inline vfft_il2p_plan_t *vfft_il2p_create(int N, int R1, int R2)
{
    if (N <= 0 || R1 < 4 || R2 < 8 || (long)R1 * (long)R2 != (long)N) return 0;
    if ((R1 & 1) || (R2 & 1)) return 0;
    vfft_il2p_fn lf = vfft_il2p_leaf_fn(R2, 0), lb = vfft_il2p_leaf_fn(R2, 1);
    vfft_il2p_fn mf = vfft_il2p_mid_fn(R1, 0),  mb = vfft_il2p_mid_fn(R1, 1);
    if (!lf || !lb || !mf || !mb) return 0;
    /* n1_b may be absent without invalidating the forward plan — only the
     * F-DIAG backward path needs it, and execute_bwd checks. */
    vfft_il2p_fn nb = vfft_il2p_n1_bwd_fn(R1);
    vfft_il2p_fn tp = vfft_il2p_t2p_bwd_fn(R1);
    vfft_il2p_fn tt = vfft_il2p_t2t_bwd_fn(R1);
    vfft_il2p_fn nb2 = vfft_il2p_n1_bwd_fn(R2);   /* B stage 2 is radix R2 */

    vfft_il2p_plan_t *p = (vfft_il2p_plan_t *)calloc(1, sizeof(*p));
    if (!p) return 0;
    p->N = N; p->R1 = R1; p->R2 = R2;
    p->leaf_f = lf; p->leaf_b = lb; p->mid_f = mf; p->mid_b = mb;
    p->n1_b = nb;
    p->t2p_b = tp;
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
 * Fusing the diagonal into stage 2 is exactly the T2P kind (pre-twiddle +
 * backward butterfly + straight store). This F-DIAG form is BITWISE IDENTICAL
 * to it (|A - F-DIAG| = 0.000e+00 at all 7 cells), so it proves the math on
 * real hardware with NO new kernel. Keep it as the reference arm once T2P
 * exists.
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

/* ROUTE A, FUSED — identical math to F-DIAG above, with the diagonal folded
 * into stage 2 by the t2p kernel (PRE-twiddle + backward butterfly + straight
 * store). Stage 1 is untouched; stage 2 swaps n1_bwd for t2p_bwd and takes the
 * conjugated table, so the whole scratch plane is read and written ONCE
 * instead of twice, and the multiply happens in-register rather than scalar.
 *
 * Gate it against vfft_il2p_execute_bwd (F-DIAG): same c/s values, same order
 * of operations, so the results should agree to the last bit or very near it.
 * Any O(1) disagreement means the fused kernel's twiddle indexing is wrong,
 * NOT that the decomposition is wrong -- F-DIAG already proved the math.
 *
 * Returns 0 on success, -1 if this build lacks the t2p twin. */
static inline int vfft_il2p_execute_bwd_t2p(const vfft_il2p_plan_t *p,
                                            const double *zin, double *zout)
{
    const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
    if (!p->t2p_b) return -1;
    p->leaf_b(zin, 0, p->mid, 0, 0, 0, R1, 0, R2, 0, R1);
    p->t2p_b(p->mid, 0, zout, 0, p->twb, 0, R2, 0, R2, 0, R2);
    return 0;
}

/* ROUTE B — the MIRROR decomposition: run the R1 butterfly FIRST, then R2.
 * (Route A above runs R2 first, mirroring the forward's stage order.)
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

/* ── THE DEFAULT BACKWARD PATH ───────────────────────────────────────────
 * Route B (t2t). Chosen on MEASUREMENT, not argument: three independent runs
 * of build_tuned/benches/il2p_bwd_gate.c show the winner tracks R1 —
 *
 *     R1 <= 32 : B wins, 2-14%   (128 8x16, 128 16x8, 512 16x32,
 *                                 512 32x16, 1024 16x64)
 *     R1 == 64 : A wins, 1-10%   (1024 64x16, 2048 64x32, 4096 64x64)
 *     1024 32x32 : unresolvable at this precision (0.75 / 1.00 / 1.08)
 *
 * B's stage 1 IS the R1 butterfly (with a turned store), so a fat R1 makes B
 * pay early; A defers R1 to stage 2. Tugbars' call: IL plans favour many small
 * stages, so R1=64 is rare in practice ⇒ default B.
 *
 * ⚠️ A SINGLE RUN WOULD HAVE MISLED — the first race read B 9/10 and did not
 * reproduce (B 7/10, then 6/10). Always repeat this race before re-deciding.
 *
 * 🔴 This is a DEFAULT, not a plan. The per-cell pick belongs in wisdom; do NOT
 * add a hand-written `if (R1 == 64) use A` here — that is precisely the
 * hand-invented heuristic the project forbids. Fallback order below is by
 * AVAILABILITY only.
 *
 * B also covers strictly more pairs: A's stage 1 needs an n1t leaf at radix R2,
 * which does not exist at R2=4 (validated at 128=32x4, 64=16x4). */
static inline int vfft_il2p_execute_bwd(const vfft_il2p_plan_t *p,
                                        const double *zin, double *zout)
{
    if (vfft_il2p_execute_bwd_t2t(p, zin, zout) == 0) return 0;
    if (vfft_il2p_execute_bwd_t2p(p, zin, zout) == 0) return 0;
    return vfft_il2p_execute_bwd_fdiag(p, zin, zout);
}

#endif /* VFFT_IL2P_H */
