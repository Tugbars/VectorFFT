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

typedef struct {
    int N, R1, R2;
    double *mid;            /* interleaved scratch, 2N doubles */
    double *tw, *twb;       /* streamed VTW2 for t2: fwd and conjugated bwd */
    vfft_il2p_fn leaf_f, leaf_b;   /* n1t, radix R2 */
    vfft_il2p_fn mid_f,  mid_b;    /* t2,  radix R1 */
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

    vfft_il2p_plan_t *p = (vfft_il2p_plan_t *)calloc(1, sizeof(*p));
    if (!p) return 0;
    p->N = N; p->R1 = R1; p->R2 = R2;
    p->leaf_f = lf; p->leaf_b = lb; p->mid_f = mf; p->mid_b = mb;

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
 * conjugated table. The defect is purely this route's stage composition. */
static inline int vfft_il2p_execute_bwd(const vfft_il2p_plan_t *p,
                                        const double *zin, double *zout)
{
    (void)p; (void)zin; (void)zout;
    return -1;
}

#endif /* VFFT_IL2P_H */
