/* zil_sterm_pipe.c — sterm SOFTWARE-PIPELINING race (z_cascade_plan §4.9993).
 *
 * VTune (§4.9991): sterm is the stall mass — CPI 0.371, 44% retiring, store
 * latency + FB-full + L1-latency dominant; 16+ live ymm ⇒ gcc spills. This
 * bench races scheduling-only variants of the terminator inside the REAL
 * production cascade (zsplit.h plan + s0s/msg front), all gated BIT-IDENTICAL
 * against the emitted baseline before timing.
 *
 * Arms: emit  = linked emitted radix8_z_sterm_fwd_avx2 (control)
 *       copy  = same source pasted in this TU (TU/flag-parity control)
 *       rot   = cross-iteration rotation of the twiddle squaring tree
 *       phase = intra-iteration live-range minimization (anti-spill)
 *       pfw   = PREFETCHW on the 8 output RFO streams (+1 input line)
 *       uj2   = 2-quad unroll-and-jam (ILP/MLP hypothesis test)
 *       nt    = non-temporal full-line stores (informational: RFO killer;
 *               needs aligned zout — bench allocs are 64B-aligned)
 *
 * Discipline (canonical_mkl_bench): pin logical core 2 (mask 4), HIGH prio,
 * 32MB cachebust per measurement, arm order rotated per round, best-of-R,
 * Sleep pacing. Timings: FULL cascade (ground truth) + terminator-only
 * (resolution; input = plan->sp left hot by a front pass, like real life).
 *
 * Build:  python build.py --src benches/zil_sterm_pipe.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <immintrin.h>

#include "zsplit.h"

/* ---- macros: verbatim from radix8_z_sterm_avx2.c (emitter-owned) ---- */
#define DEINT(zlo, zhi, re, im) do {                                  \
    __m256d _u = _mm256_unpacklo_pd(zlo, zhi);                        \
    __m256d _v = _mm256_unpackhi_pd(zlo, zhi);                        \
    re = _mm256_permute4x64_pd(_u, 0xD8);                             \
    im = _mm256_permute4x64_pd(_v, 0xD8);                             \
} while (0)
#define REINT(re, im, zlo, zhi) do {                                  \
    __m256d _p = _mm256_permute4x64_pd(re, 0xD8);                     \
    __m256d _q = _mm256_permute4x64_pd(im, 0xD8);                     \
    zlo = _mm256_unpacklo_pd(_p, _q);                                 \
    zhi = _mm256_unpackhi_pd(_p, _q);                                 \
} while (0)
#define SPLIT_CMUL(ar,ai, ct,st, or_,oi_) do {                        \
    or_ = _mm256_fnmadd_pd(st, ai, _mm256_mul_pd(ct, ar));            \
    oi_ = _mm256_fmadd_pd(st, ar, _mm256_mul_pd(ct, ai));             \
} while (0)
#define TR4(a0,a1,a2,a3, t0,t1,t2,t3) do {                            \
    __m256d _u0 = _mm256_unpacklo_pd(a0, a1);                         \
    __m256d _u1 = _mm256_unpackhi_pd(a0, a1);                         \
    __m256d _u2 = _mm256_unpacklo_pd(a2, a3);                         \
    __m256d _u3 = _mm256_unpackhi_pd(a2, a3);                         \
    t0 = _mm256_permute2f128_pd(_u0, _u2, 0x20);                      \
    t1 = _mm256_permute2f128_pd(_u1, _u3, 0x20);                      \
    t2 = _mm256_permute2f128_pd(_u0, _u2, 0x31);                      \
    t3 = _mm256_permute2f128_pd(_u1, _u3, 0x31);                      \
} while (0)
#define WPROD(cA,sA, cB,sB, cP,sP) do {                               \
    cP = _mm256_fnmadd_pd(sA, sB, _mm256_mul_pd(cA, cB));             \
    sP = _mm256_fmadd_pd(cA, sB, _mm256_mul_pd(sA, cB));              \
} while (0)
#define SPLIT_BFLY8(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i, \
                    o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i) do { \
    const __m256d _C = _mm256_set1_pd(0.70710678118654752440);        \
    __m256d t0r=_mm256_add_pd(x0r,x4r), t0i=_mm256_add_pd(x0i,x4i);   \
    __m256d t1r=_mm256_sub_pd(x0r,x4r), t1i=_mm256_sub_pd(x0i,x4i);   \
    __m256d t2r=_mm256_add_pd(x2r,x6r), t2i=_mm256_add_pd(x2i,x6i);   \
    __m256d t3r=_mm256_sub_pd(x2r,x6r), t3i=_mm256_sub_pd(x2i,x6i);   \
    __m256d E0r=_mm256_add_pd(t0r,t2r), E0i=_mm256_add_pd(t0i,t2i);   \
    __m256d E2r=_mm256_sub_pd(t0r,t2r), E2i=_mm256_sub_pd(t0i,t2i);   \
    __m256d E1r=_mm256_add_pd(t1r,t3i), E1i=_mm256_sub_pd(t1i,t3r);   \
    __m256d E3r=_mm256_sub_pd(t1r,t3i), E3i=_mm256_add_pd(t1i,t3r);   \
    __m256d s0r=_mm256_add_pd(x1r,x5r), s0i=_mm256_add_pd(x1i,x5i);   \
    __m256d s1r=_mm256_sub_pd(x1r,x5r), s1i=_mm256_sub_pd(x1i,x5i);   \
    __m256d s2r=_mm256_add_pd(x3r,x7r), s2i=_mm256_add_pd(x3i,x7i);   \
    __m256d s3r=_mm256_sub_pd(x3r,x7r), s3i=_mm256_sub_pd(x3i,x7i);   \
    __m256d O0r=_mm256_add_pd(s0r,s2r), O0i=_mm256_add_pd(s0i,s2i);   \
    __m256d O2r=_mm256_sub_pd(s0r,s2r), O2i=_mm256_sub_pd(s0i,s2i);   \
    __m256d O1r=_mm256_add_pd(s1r,s3i), O1i=_mm256_sub_pd(s1i,s3r);   \
    __m256d O3r=_mm256_sub_pd(s1r,s3i), O3i=_mm256_add_pd(s1i,s3r);   \
    __m256d X1r=_mm256_add_pd(O1r,O1i), X1i=_mm256_sub_pd(O1i,O1r);   \
    __m256d X3r=_mm256_sub_pd(O3i,O3r), X3n=_mm256_add_pd(O3r,O3i);   \
    o0r=_mm256_add_pd(E0r,O0r); o0i=_mm256_add_pd(E0i,O0i);           \
    o4r=_mm256_sub_pd(E0r,O0r); o4i=_mm256_sub_pd(E0i,O0i);           \
    o1r=_mm256_fmadd_pd(_C,X1r,E1r); o1i=_mm256_fmadd_pd(_C,X1i,E1i); \
    o5r=_mm256_fnmadd_pd(_C,X1r,E1r); o5i=_mm256_fnmadd_pd(_C,X1i,E1i); \
    o2r=_mm256_add_pd(E2r,O2i); o2i=_mm256_sub_pd(E2i,O2r);           \
    o6r=_mm256_sub_pd(E2r,O2i); o6i=_mm256_add_pd(E2i,O2r);           \
    o3r=_mm256_fmadd_pd(_C,X3r,E3r); o3i=_mm256_fnmadd_pd(_C,X3n,E3i); \
    o7r=_mm256_fnmadd_pd(_C,X3r,E3r); o7i=_mm256_fmadd_pd(_C,X3n,E3i); \
} while (0)

typedef void (*zfn)(const double *, const double *, double *, double *,
                    const double *, const double *, unsigned long long,
                    unsigned long long, unsigned long long,
                    unsigned long long, unsigned long long);

/* ---- arm "copy": baseline body pasted (TU/flag parity control) ---- */
__attribute__((target("avx2,fma")))
static void sterm_fwd_copy(
    const double * __restrict__ zin, const double * __restrict__ zu,
    double * __restrict__ zout, double * __restrict__ zou,
    const double * tw_re, const double * tw_im,
    unsigned long long Ls, unsigned long long Gs, unsigned long long OLs,
    unsigned long long OGs, unsigned long long count)
{
    (void)zu; (void)zou; (void)tw_im; (void)Ls; (void)Gs; (void)OGs;
    for (size_t k = 0; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8];
        {
            __m256d rl0 = _mm256_loadu_pd(zin + 16*(size_t)k);
            __m256d il0 = _mm256_loadu_pd(zin + 16*(size_t)k + 4);
            __m256d rh0 = _mm256_loadu_pd(zin + 16*(size_t)k + 8);
            __m256d ih0 = _mm256_loadu_pd(zin + 16*(size_t)k + 12);
            __m256d rl1 = _mm256_loadu_pd(zin + 16*((size_t)k+1));
            __m256d il1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 4);
            __m256d rh1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 8);
            __m256d ih1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 12);
            __m256d rl2 = _mm256_loadu_pd(zin + 16*((size_t)k+2));
            __m256d il2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 4);
            __m256d rh2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 8);
            __m256d ih2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 12);
            __m256d rl3 = _mm256_loadu_pd(zin + 16*((size_t)k+3));
            __m256d il3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 4);
            __m256d rh3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 8);
            __m256d ih3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 12);
            TR4(rl0, rl1, rl2, rl3, xr[0], xr[1], xr[2], xr[3]);
            TR4(il0, il1, il2, il3, xi[0], xi[1], xi[2], xi[3]);
            TR4(rh0, rh1, rh2, rh3, xr[4], xr[5], xr[6], xr[7]);
            TR4(ih0, ih1, ih2, ih3, xi[4], xi[5], xi[6], xi[7]);
        }
        {
            __m256d c1 = _mm256_loadu_pd(tw_re + 2*(size_t)k);
            __m256d s1 = _mm256_loadu_pd(tw_re + 2*(size_t)k + 4);
            __m256d c2, s2, c3, s3, c4, s4, cw, sw, rr, ii;
            SPLIT_CMUL(xr[1], xi[1], c1, s1, rr, ii); xr[1] = rr; xi[1] = ii;
            WPROD(c1, s1, c1, s1, c2, s2);
            SPLIT_CMUL(xr[2], xi[2], c2, s2, rr, ii); xr[2] = rr; xi[2] = ii;
            WPROD(c2, s2, c1, s1, c3, s3);
            SPLIT_CMUL(xr[3], xi[3], c3, s3, rr, ii); xr[3] = rr; xi[3] = ii;
            WPROD(c2, s2, c2, s2, c4, s4);
            SPLIT_CMUL(xr[4], xi[4], c4, s4, rr, ii); xr[4] = rr; xi[4] = ii;
            WPROD(c4, s4, c1, s1, cw, sw);
            SPLIT_CMUL(xr[5], xi[5], cw, sw, rr, ii); xr[5] = rr; xi[5] = ii;
            WPROD(c4, s4, c2, s2, cw, sw);
            SPLIT_CMUL(xr[6], xi[6], cw, sw, rr, ii); xr[6] = rr; xi[6] = ii;
            WPROD(c4, s4, c3, s3, cw, sw);
            SPLIT_CMUL(xr[7], xi[7], cw, sw, rr, ii); xr[7] = rr; xi[7] = ii;
        }
        __m256d or_[8], oi_[8];
        SPLIT_BFLY8(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                    xr[4],xi[4],xr[5],xi[5],xr[6],xi[6],xr[7],xi[7],
                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3],
                    or_[4],oi_[4],or_[5],oi_[5],or_[6],oi_[6],or_[7],oi_[7]);
        for (int l = 0; l < 8; l++) {
            __m256d zlo, zhi;
            REINT(or_[l], oi_[l], zlo, zhi);
            _mm256_storeu_pd(zout + 2*((size_t)l*OLs + k), zlo);
            _mm256_storeu_pd(zout + 2*((size_t)l*OLs + k) + 4, zhi);
        }
    }
}

/* ===================== WORKFLOW ARMS PASTED BELOW ===================== */
/* ================= ARM rot (sterm_fwd_rot, OK) ================= */
/* ARM rot — cross-iteration rotation of the twiddle squaring tree.
 * Bit-identical to radix8_z_sterm_fwd_avx2: every FP op has the same operands
 * in the same order; only cross-value scheduling / liveness is restructured.
 * Pipeline: iteration k's store tail computes iteration k+4's c1/c2/c4, so the
 * serial squaring chain never gates the top of an iteration. */
__attribute__((target("avx2,fma")))
void sterm_fwd_rot(
    const double * __restrict__ zin,
    const double * __restrict__ zin_unused,
    double       * __restrict__ zout,
    double       * __restrict__ zout_unused,
    const double * tw_re, const double * tw_im,
    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)
{
    (void)zin_unused; (void)zout_unused; (void)tw_im;
    (void)Ls; (void)Gs; (void)OGs;
    if (count < 4) return;

    const __m256d C7 = _mm256_set1_pd(0.70710678118654752440);
    const size_t oL = 2 * OLs;

    /* Prologue: rotated head of the twiddle tree for k = 0. */
    __m256d cw1 = _mm256_loadu_pd(tw_re);
    __m256d sw1 = _mm256_loadu_pd(tw_re + 4);
    __m256d cw2, sw2, cw4, sw4;
    WPROD(cw1, sw1, cw1, sw1, cw2, sw2);
    WPROD(cw2, sw2, cw2, sw2, cw4, sw4);

    for (size_t k = 0; k + 4 <= count; k += 4) {
        const double *zp = zin + 16 * k;
        double *po = zout + 2 * k;
        /* Next group's twiddle index, clamped (branchless) so the final trip
         * re-reads its own valid columns instead of running off the table;
         * those recomputed values are dead after the loop. */
        const size_t kn2 = (k + 8 <= count) ? 2 * (k + 4) : 2 * k;

        /* ---- data loads + column->leg transposes: independent of twiddles,
         * they fill the front of the iteration while the pre-rotated
         * cw1/cw2/cw4 from the previous trip are already sitting ready. ---- */
        __m256d rl0 = _mm256_loadu_pd(zp + 0);
        __m256d il0 = _mm256_loadu_pd(zp + 4);
        __m256d rh0 = _mm256_loadu_pd(zp + 8);
        __m256d ih0 = _mm256_loadu_pd(zp + 12);
        __m256d rl1 = _mm256_loadu_pd(zp + 16);
        __m256d il1 = _mm256_loadu_pd(zp + 20);
        __m256d rh1 = _mm256_loadu_pd(zp + 24);
        __m256d ih1 = _mm256_loadu_pd(zp + 28);
        __m256d rl2 = _mm256_loadu_pd(zp + 32);
        __m256d il2 = _mm256_loadu_pd(zp + 36);
        __m256d rh2 = _mm256_loadu_pd(zp + 40);
        __m256d ih2 = _mm256_loadu_pd(zp + 44);
        __m256d rl3 = _mm256_loadu_pd(zp + 48);
        __m256d il3 = _mm256_loadu_pd(zp + 52);
        __m256d rh3 = _mm256_loadu_pd(zp + 56);
        __m256d ih3 = _mm256_loadu_pd(zp + 60);

        __m256d x0r, x1r, x2r, x3r, x4r, x5r, x6r, x7r;
        __m256d x0i, x1i, x2i, x3i, x4i, x5i, x6i, x7i;
        TR4(rl0, rl1, rl2, rl3, x0r, x1r, x2r, x3r);
        TR4(il0, il1, il2, il3, x0i, x1i, x2i, x3i);
        TR4(rh0, rh1, rh2, rh3, x4r, x5r, x6r, x7r);
        TR4(ih0, ih1, ih2, ih3, x4i, x5i, x6i, x7i);

        /* ---- in-iteration remainder of the tree: only c3->c7 is serial now
         * (2 WPRODs); c5/c6 hang directly off the pre-rotated c4. Ordered so
         * each cw pair dies at its last use. Same ops/operand order as
         * baseline (WPRODs and CMULs are mutually independent values). ---- */
        __m256d cw3, sw3, cw5, sw5, cw6, sw6, cw7, sw7, rr, ii;
        WPROD(cw2, sw2, cw1, sw1, cw3, sw3);
        WPROD(cw4, sw4, cw1, sw1, cw5, sw5);
        SPLIT_CMUL(x1r, x1i, cw1, sw1, rr, ii); x1r = rr; x1i = ii;
        WPROD(cw4, sw4, cw2, sw2, cw6, sw6);
        SPLIT_CMUL(x2r, x2i, cw2, sw2, rr, ii); x2r = rr; x2i = ii;
        WPROD(cw4, sw4, cw3, sw3, cw7, sw7);
        SPLIT_CMUL(x3r, x3i, cw3, sw3, rr, ii); x3r = rr; x3i = ii;
        SPLIT_CMUL(x4r, x4i, cw4, sw4, rr, ii); x4r = rr; x4i = ii;
        SPLIT_CMUL(x5r, x5i, cw5, sw5, rr, ii); x5r = rr; x5i = ii;
        SPLIT_CMUL(x6r, x6i, cw6, sw6, rr, ii); x6r = rr; x6i = ii;
        SPLIT_CMUL(x7r, x7i, cw7, sw7, rr, ii); x7r = rr; x7i = ii;

        /* ---- SPLIT_BFLY8 hand-expanded op-for-op from the macro; outputs
         * are produced and retired pairwise so o-values never accumulate. */
        __m256d t0r = _mm256_add_pd(x0r, x4r), t0i = _mm256_add_pd(x0i, x4i);
        __m256d t1r = _mm256_sub_pd(x0r, x4r), t1i = _mm256_sub_pd(x0i, x4i);
        __m256d t2r = _mm256_add_pd(x2r, x6r), t2i = _mm256_add_pd(x2i, x6i);
        __m256d t3r = _mm256_sub_pd(x2r, x6r), t3i = _mm256_sub_pd(x2i, x6i);
        __m256d E0r = _mm256_add_pd(t0r, t2r), E0i = _mm256_add_pd(t0i, t2i);
        __m256d E2r = _mm256_sub_pd(t0r, t2r), E2i = _mm256_sub_pd(t0i, t2i);
        __m256d E1r = _mm256_add_pd(t1r, t3i), E1i = _mm256_sub_pd(t1i, t3r);
        __m256d E3r = _mm256_sub_pd(t1r, t3i), E3i = _mm256_add_pd(t1i, t3r);

        __m256d u0r = _mm256_add_pd(x1r, x5r), u0i = _mm256_add_pd(x1i, x5i);
        __m256d u1r = _mm256_sub_pd(x1r, x5r), u1i = _mm256_sub_pd(x1i, x5i);
        __m256d u2r = _mm256_add_pd(x3r, x7r), u2i = _mm256_add_pd(x3i, x7i);
        __m256d u3r = _mm256_sub_pd(x3r, x7r), u3i = _mm256_sub_pd(x3i, x7i);
        __m256d O0r = _mm256_add_pd(u0r, u2r), O0i = _mm256_add_pd(u0i, u2i);
        __m256d O2r = _mm256_sub_pd(u0r, u2r), O2i = _mm256_sub_pd(u0i, u2i);
        __m256d O1r = _mm256_add_pd(u1r, u3i), O1i = _mm256_sub_pd(u1i, u3r);
        __m256d O3r = _mm256_sub_pd(u1r, u3i), O3i = _mm256_add_pd(u1i, u3r);

        /* rotated: next group's w^1 loads issue in the butterfly's shadow */
        __m256d nc1 = _mm256_loadu_pd(tw_re + kn2);
        __m256d ns1 = _mm256_loadu_pd(tw_re + kn2 + 4);

        __m256d X1r = _mm256_add_pd(O1r, O1i), X1i = _mm256_sub_pd(O1i, O1r);
        __m256d X3r = _mm256_sub_pd(O3i, O3r), X3n = _mm256_add_pd(O3r, O3i);

        __m256d zlo, zhi;

        /* pair (0,4) */
        __m256d o0r = _mm256_add_pd(E0r, O0r), o0i = _mm256_add_pd(E0i, O0i);
        __m256d o4r = _mm256_sub_pd(E0r, O0r), o4i = _mm256_sub_pd(E0i, O0i);
        REINT(o0r, o0i, zlo, zhi);
        _mm256_storeu_pd(po, zlo);
        _mm256_storeu_pd(po + 4, zhi);
        REINT(o4r, o4i, zlo, zhi);
        _mm256_storeu_pd(po + 4 * oL, zlo);
        _mm256_storeu_pd(po + 4 * oL + 4, zhi);

        /* rotated: c2 = w^2 for the next trip (FMA work interleaved with the
         * port-5-heavy REINT shuffles and the store burst) */
        __m256d nc2, ns2;
        WPROD(nc1, ns1, nc1, ns1, nc2, ns2);

        /* pair (1,5) */
        __m256d o1r = _mm256_fmadd_pd(C7, X1r, E1r),  o1i = _mm256_fmadd_pd(C7, X1i, E1i);
        __m256d o5r = _mm256_fnmadd_pd(C7, X1r, E1r), o5i = _mm256_fnmadd_pd(C7, X1i, E1i);
        REINT(o1r, o1i, zlo, zhi);
        _mm256_storeu_pd(po + oL, zlo);
        _mm256_storeu_pd(po + oL + 4, zhi);
        REINT(o5r, o5i, zlo, zhi);
        _mm256_storeu_pd(po + 5 * oL, zlo);
        _mm256_storeu_pd(po + 5 * oL + 4, zhi);

        /* rotated: c4 = w^4 for the next trip */
        __m256d nc4, ns4;
        WPROD(nc2, ns2, nc2, ns2, nc4, ns4);

        /* pair (2,6) */
        __m256d o2r = _mm256_add_pd(E2r, O2i), o2i = _mm256_sub_pd(E2i, O2r);
        __m256d o6r = _mm256_sub_pd(E2r, O2i), o6i = _mm256_add_pd(E2i, O2r);
        REINT(o2r, o2i, zlo, zhi);
        _mm256_storeu_pd(po + 2 * oL, zlo);
        _mm256_storeu_pd(po + 2 * oL + 4, zhi);
        REINT(o6r, o6i, zlo, zhi);
        _mm256_storeu_pd(po + 6 * oL, zlo);
        _mm256_storeu_pd(po + 6 * oL + 4, zhi);

        /* pair (3,7) */
        __m256d o3r = _mm256_fmadd_pd(C7, X3r, E3r),  o3i = _mm256_fnmadd_pd(C7, X3n, E3i);
        __m256d o7r = _mm256_fnmadd_pd(C7, X3r, E3r), o7i = _mm256_fmadd_pd(C7, X3n, E3i);
        REINT(o3r, o3i, zlo, zhi);
        _mm256_storeu_pd(po + 3 * oL, zlo);
        _mm256_storeu_pd(po + 3 * oL + 4, zhi);
        REINT(o7r, o7i, zlo, zhi);
        _mm256_storeu_pd(po + 7 * oL, zlo);
        _mm256_storeu_pd(po + 7 * oL + 4, zhi);

        /* rotate the pipelined twiddle head into place for k+4 */
        cw1 = nc1; sw1 = ns1;
        cw2 = nc2; sw2 = ns2;
        cw4 = nc4; sw4 = ns4;
    }
}

/* ================= ARM phase (sterm_fwd_phase, OK) ================= */
/* ARM "phase": intra-iteration live-range minimization for the sterm
 * split-input radix-8 terminator. Three phases per 4-column trip:
 *   A) lo-block loads + TR4 -> legs 0-3; twiddle chain c2/c3/c4; CMUL legs 1-3.
 *   B) chain extension c5/c6/c7 (kills c1..c3 before the hi loads, overlaps
 *      serial WPROD FMA latency with the hi loads/transposes); hi-block loads
 *      + TR4 -> legs 4-7; CMUL legs 4-7.
 *   C) SPLIT_BFLY8 hand-expanded in the macro's exact op order, with each
 *      output pair REINT+stored as soon as it is final (o0/o4, o1/o5, o2/o6,
 *      o3/o7); X1/X3 sunk to just before their consumers. No or_/oi_ arrays.
 * All FP ops, operand orders, and associations are identical to the baseline
 * macros; only independent-value ordering and staging differ. */
__attribute__((target("avx2,fma")))
void sterm_fwd_phase(
    const double * __restrict__ zin,
    const double * __restrict__ zin_unused,
    double       * __restrict__ zout,
    double       * __restrict__ zout_unused,
    const double * tw_re, const double * tw_im,
    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)
{
    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Ls; (void)Gs; (void)OGs;
    const __m256d C7 = _mm256_set1_pd(0.70710678118654752440);
    double * const q0 = zout;
    double * const q1 = zout +  2*OLs;
    double * const q2 = zout +  4*OLs;
    double * const q3 = zout +  6*OLs;
    double * const q4 = zout +  8*OLs;
    double * const q5 = zout + 10*OLs;
    double * const q6 = zout + 12*OLs;
    double * const q7 = zout + 14*OLs;
    for (size_t k = 0; k + 4 <= count; k += 4) {
        const double * const p = zin + 16*k;

        /* ---------------- Phase A: lo block -> legs 0-3 ---------------- */
        __m256d x0r, x0i, x1r, x1i, x2r, x2i, x3r, x3i;
        {
            __m256d rl0 = _mm256_loadu_pd(p +  0), il0 = _mm256_loadu_pd(p +  4);
            __m256d rl1 = _mm256_loadu_pd(p + 16), il1 = _mm256_loadu_pd(p + 20);
            __m256d rl2 = _mm256_loadu_pd(p + 32), il2 = _mm256_loadu_pd(p + 36);
            __m256d rl3 = _mm256_loadu_pd(p + 48), il3 = _mm256_loadu_pd(p + 52);
            TR4(rl0, rl1, rl2, rl3, x0r, x1r, x2r, x3r);
            TR4(il0, il1, il2, il3, x0i, x1i, x2i, x3i);
        }
        __m256d c1, s1, c2, s2, c3, s3, c4, s4;
        c1 = _mm256_loadu_pd(tw_re + 2*k);
        s1 = _mm256_loadu_pd(tw_re + 2*k + 4);
        {
            __m256d rr, ii;
            SPLIT_CMUL(x1r, x1i, c1, s1, rr, ii); x1r = rr; x1i = ii;
            WPROD(c1, s1, c1, s1, c2, s2);
            SPLIT_CMUL(x2r, x2i, c2, s2, rr, ii); x2r = rr; x2i = ii;
            WPROD(c2, s2, c1, s1, c3, s3);
            SPLIT_CMUL(x3r, x3i, c3, s3, rr, ii); x3r = rr; x3i = ii;
            WPROD(c2, s2, c2, s2, c4, s4);
        }

        /* ---------------- Phase B: hi block -> legs 4-7 ---------------- */
        __m256d x4r, x4i, x5r, x5i, x6r, x6i, x7r, x7i;
        {
            /* Extend the twiddle chain first: c1/c2/c3 die HERE, and the
             * serial FMA latency of the three WPRODs overlaps the hi-block
             * loads and transposes below. Same operands as the baseline's
             * cw/sw computations -> bit-identical values. */
            __m256d c5, s5, c6, s6, c7, s7;
            WPROD(c4, s4, c1, s1, c5, s5);
            WPROD(c4, s4, c2, s2, c6, s6);
            WPROD(c4, s4, c3, s3, c7, s7);
            {
                __m256d rh0 = _mm256_loadu_pd(p +  8), ih0 = _mm256_loadu_pd(p + 12);
                __m256d rh1 = _mm256_loadu_pd(p + 24), ih1 = _mm256_loadu_pd(p + 28);
                __m256d rh2 = _mm256_loadu_pd(p + 40), ih2 = _mm256_loadu_pd(p + 44);
                __m256d rh3 = _mm256_loadu_pd(p + 56), ih3 = _mm256_loadu_pd(p + 60);
                TR4(rh0, rh1, rh2, rh3, x4r, x5r, x6r, x7r);
                TR4(ih0, ih1, ih2, ih3, x4i, x5i, x6i, x7i);
            }
            __m256d rr, ii;
            SPLIT_CMUL(x4r, x4i, c4, s4, rr, ii); x4r = rr; x4i = ii;
            SPLIT_CMUL(x5r, x5i, c5, s5, rr, ii); x5r = rr; x5i = ii;
            SPLIT_CMUL(x6r, x6i, c6, s6, rr, ii); x6r = rr; x6i = ii;
            SPLIT_CMUL(x7r, x7i, c7, s7, rr, ii); x7r = rr; x7i = ii;
        }

        /* -------- Phase C: BFLY8, outputs stored as soon as final -------- */
        {
            __m256d t0r = _mm256_add_pd(x0r, x4r), t0i = _mm256_add_pd(x0i, x4i);
            __m256d t1r = _mm256_sub_pd(x0r, x4r), t1i = _mm256_sub_pd(x0i, x4i);
            __m256d t2r = _mm256_add_pd(x2r, x6r), t2i = _mm256_add_pd(x2i, x6i);
            __m256d t3r = _mm256_sub_pd(x2r, x6r), t3i = _mm256_sub_pd(x2i, x6i);
            __m256d E0r = _mm256_add_pd(t0r, t2r), E0i = _mm256_add_pd(t0i, t2i);
            __m256d E2r = _mm256_sub_pd(t0r, t2r), E2i = _mm256_sub_pd(t0i, t2i);
            __m256d E1r = _mm256_add_pd(t1r, t3i), E1i = _mm256_sub_pd(t1i, t3r);
            __m256d E3r = _mm256_sub_pd(t1r, t3i), E3i = _mm256_add_pd(t1i, t3r);
            __m256d s0r = _mm256_add_pd(x1r, x5r), s0i = _mm256_add_pd(x1i, x5i);
            __m256d s1r = _mm256_sub_pd(x1r, x5r), s1i = _mm256_sub_pd(x1i, x5i);
            __m256d s2r = _mm256_add_pd(x3r, x7r), s2i = _mm256_add_pd(x3i, x7i);
            __m256d s3r = _mm256_sub_pd(x3r, x7r), s3i = _mm256_sub_pd(x3i, x7i);
            __m256d O0r = _mm256_add_pd(s0r, s2r), O0i = _mm256_add_pd(s0i, s2i);
            __m256d O2r = _mm256_sub_pd(s0r, s2r), O2i = _mm256_sub_pd(s0i, s2i);
            __m256d O1r = _mm256_add_pd(s1r, s3i), O1i = _mm256_sub_pd(s1i, s3r);
            __m256d O3r = _mm256_sub_pd(s1r, s3i), O3i = _mm256_add_pd(s1i, s3r);
            __m256d zlo, zhi;
            /* legs 0 / 4: final now -> retire E0/O0 immediately */
            {
                __m256d o0r = _mm256_add_pd(E0r, O0r), o0i = _mm256_add_pd(E0i, O0i);
                REINT(o0r, o0i, zlo, zhi);
                _mm256_storeu_pd(q0 + 2*k,     zlo);
                _mm256_storeu_pd(q0 + 2*k + 4, zhi);
                __m256d o4r = _mm256_sub_pd(E0r, O0r), o4i = _mm256_sub_pd(E0i, O0i);
                REINT(o4r, o4i, zlo, zhi);
                _mm256_storeu_pd(q4 + 2*k,     zlo);
                _mm256_storeu_pd(q4 + 2*k + 4, zhi);
            }
            /* legs 1 / 5: X1 sunk here (independent), then retire E1/X1 */
            {
                __m256d X1r = _mm256_add_pd(O1r, O1i), X1i = _mm256_sub_pd(O1i, O1r);
                __m256d o1r = _mm256_fmadd_pd(C7, X1r, E1r),
                        o1i = _mm256_fmadd_pd(C7, X1i, E1i);
                REINT(o1r, o1i, zlo, zhi);
                _mm256_storeu_pd(q1 + 2*k,     zlo);
                _mm256_storeu_pd(q1 + 2*k + 4, zhi);
                __m256d o5r = _mm256_fnmadd_pd(C7, X1r, E1r),
                        o5i = _mm256_fnmadd_pd(C7, X1i, E1i);
                REINT(o5r, o5i, zlo, zhi);
                _mm256_storeu_pd(q5 + 2*k,     zlo);
                _mm256_storeu_pd(q5 + 2*k + 4, zhi);
            }
            /* legs 2 / 6: retire E2/O2 */
            {
                __m256d o2r = _mm256_add_pd(E2r, O2i), o2i = _mm256_sub_pd(E2i, O2r);
                REINT(o2r, o2i, zlo, zhi);
                _mm256_storeu_pd(q2 + 2*k,     zlo);
                _mm256_storeu_pd(q2 + 2*k + 4, zhi);
                __m256d o6r = _mm256_sub_pd(E2r, O2i), o6i = _mm256_add_pd(E2i, O2r);
                REINT(o6r, o6i, zlo, zhi);
                _mm256_storeu_pd(q6 + 2*k,     zlo);
                _mm256_storeu_pd(q6 + 2*k + 4, zhi);
            }
            /* legs 3 / 7: X3 sunk here (independent), then retire E3/X3 */
            {
                __m256d X3r = _mm256_sub_pd(O3i, O3r), X3n = _mm256_add_pd(O3r, O3i);
                __m256d o3r = _mm256_fmadd_pd(C7, X3r, E3r),
                        o3i = _mm256_fnmadd_pd(C7, X3n, E3i);
                REINT(o3r, o3i, zlo, zhi);
                _mm256_storeu_pd(q3 + 2*k,     zlo);
                _mm256_storeu_pd(q3 + 2*k + 4, zhi);
                __m256d o7r = _mm256_fnmadd_pd(C7, X3r, E3r),
                        o7i = _mm256_fmadd_pd(C7, X3n, E3i);
                REINT(o7r, o7i, zlo, zhi);
                _mm256_storeu_pd(q7 + 2*k,     zlo);
                _mm256_storeu_pd(q7 + 2*k + 4, zhi);
            }
        }
    }
}

/* ================= ARM pfw (sterm_fwd_pfw, OK) ================= */
/* ARM pfw — STORE-STREAM PREFETCH variant of radix8_z_sterm_fwd_avx2.
 * Body is the baseline verbatim; only additions are PREFETCHW on all 8
 * output-leg lines two iterations ahead (k+8 = next-but-one 64B line of
 * each of the 8 RFO streams) plus one read prefetch on the input stream.
 * Macros DEINT/REINT/SPLIT_CMUL/TR4/WPROD/SPLIT_BFLY8 are provided by the
 * bench TU. Prefetching past the buffer end on the last two iterations is
 * architecturally safe: prefetch never faults. */
__attribute__((target("avx2,fma,prfchw")))
void sterm_fwd_pfw(
    const double * __restrict__ zin,
    const double * __restrict__ zin_unused,
    double       * __restrict__ zout,
    double       * __restrict__ zout_unused,
    const double * tw_re, const double * tw_im,
    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)
{
    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Ls; (void)Gs; (void)OGs;
    for (size_t k = 0; k + 4 <= count; k += 4) {
        /* Write-intent prefetch: the 64B output line each leg will store two
         * iterations from now (each iter stores exactly one line per leg). */
        __builtin_prefetch((const void*)(zout + 2*((size_t)0*OLs + k + 8)), 1, 3);
        __builtin_prefetch((const void*)(zout + 2*((size_t)1*OLs + k + 8)), 1, 3);
        __builtin_prefetch((const void*)(zout + 2*((size_t)2*OLs + k + 8)), 1, 3);
        __builtin_prefetch((const void*)(zout + 2*((size_t)3*OLs + k + 8)), 1, 3);
        __builtin_prefetch((const void*)(zout + 2*((size_t)4*OLs + k + 8)), 1, 3);
        __builtin_prefetch((const void*)(zout + 2*((size_t)5*OLs + k + 8)), 1, 3);
        __builtin_prefetch((const void*)(zout + 2*((size_t)6*OLs + k + 8)), 1, 3);
        __builtin_prefetch((const void*)(zout + 2*((size_t)7*OLs + k + 8)), 1, 3);
        /* Read prefetch: input block two iterations ahead. */
        __builtin_prefetch(zin + 16*(k+8), 0, 3);
        __m256d xr[8], xi[8];
        {
            __m256d rl0 = _mm256_loadu_pd(zin + 16*(size_t)k);
            __m256d il0 = _mm256_loadu_pd(zin + 16*(size_t)k + 4);
            __m256d rh0 = _mm256_loadu_pd(zin + 16*(size_t)k + 8);
            __m256d ih0 = _mm256_loadu_pd(zin + 16*(size_t)k + 12);
            __m256d rl1 = _mm256_loadu_pd(zin + 16*((size_t)k+1));
            __m256d il1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 4);
            __m256d rh1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 8);
            __m256d ih1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 12);
            __m256d rl2 = _mm256_loadu_pd(zin + 16*((size_t)k+2));
            __m256d il2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 4);
            __m256d rh2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 8);
            __m256d ih2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 12);
            __m256d rl3 = _mm256_loadu_pd(zin + 16*((size_t)k+3));
            __m256d il3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 4);
            __m256d rh3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 8);
            __m256d ih3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 12);
            TR4(rl0, rl1, rl2, rl3, xr[0], xr[1], xr[2], xr[3]);
            TR4(il0, il1, il2, il3, xi[0], xi[1], xi[2], xi[3]);
            TR4(rh0, rh1, rh2, rh3, xr[4], xr[5], xr[6], xr[7]);
            TR4(ih0, ih1, ih2, ih3, xi[4], xi[5], xi[6], xi[7]);
        }
        {
            __m256d c1 = _mm256_loadu_pd(tw_re + 2*(size_t)k);
            __m256d s1 = _mm256_loadu_pd(tw_re + 2*(size_t)k + 4);
            __m256d c2, s2, c3, s3, c4, s4, cw, sw, rr, ii;
            SPLIT_CMUL(xr[1], xi[1], c1, s1, rr, ii); xr[1] = rr; xi[1] = ii;
            WPROD(c1, s1, c1, s1, c2, s2);
            SPLIT_CMUL(xr[2], xi[2], c2, s2, rr, ii); xr[2] = rr; xi[2] = ii;
            WPROD(c2, s2, c1, s1, c3, s3);
            SPLIT_CMUL(xr[3], xi[3], c3, s3, rr, ii); xr[3] = rr; xi[3] = ii;
            WPROD(c2, s2, c2, s2, c4, s4);
            SPLIT_CMUL(xr[4], xi[4], c4, s4, rr, ii); xr[4] = rr; xi[4] = ii;
            WPROD(c4, s4, c1, s1, cw, sw);
            SPLIT_CMUL(xr[5], xi[5], cw, sw, rr, ii); xr[5] = rr; xi[5] = ii;
            WPROD(c4, s4, c2, s2, cw, sw);
            SPLIT_CMUL(xr[6], xi[6], cw, sw, rr, ii); xr[6] = rr; xi[6] = ii;
            WPROD(c4, s4, c3, s3, cw, sw);
            SPLIT_CMUL(xr[7], xi[7], cw, sw, rr, ii); xr[7] = rr; xi[7] = ii;
        }
        __m256d or_[8], oi_[8];
        SPLIT_BFLY8(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                    xr[4],xi[4],xr[5],xi[5],xr[6],xi[6],xr[7],xi[7],
                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3],
                    or_[4],oi_[4],or_[5],oi_[5],or_[6],oi_[6],or_[7],oi_[7]);
        for (int l = 0; l < 8; l++) {
            __m256d zlo, zhi;
            REINT(or_[l], oi_[l], zlo, zhi);
            _mm256_storeu_pd(zout + 2*((size_t)l*OLs + k), zlo);
            _mm256_storeu_pd(zout + 2*((size_t)l*OLs + k) + 4, zhi);
        }
    }
}

/* ================= ARM uj2 (sterm_fwd_uj2, OK) ================= */
/* ARM uj2 - 2-quad unroll-and-jam of the split-input radix-8 terminator.
 * Two independent 4-column bodies per trip: A = cols k..k+3, B = cols k+4..k+7.
 * Per-column FP operation order is exactly the baseline's (TR4 / WPROD /
 * SPLIT_CMUL / SPLIT_BFLY8 / REINT, same operands, same association); only
 * cross-column scheduling, scalarized staging and loop structure differ.
 * Interleave: [loads A+B] [TR4 A] [TR4 B] [twiddle tree + CMULs alternated
 * A/B op-by-op] [BFLY A] [stores A] [BFLY B] [stores B].
 * Tail: baseline-shaped 4-column body for count % 8 == 4. */
__attribute__((target("avx2,fma")))
void sterm_fwd_uj2(
    const double * __restrict__ zin,
    const double * __restrict__ zin_unused,
    double       * __restrict__ zout,
    double       * __restrict__ zout_unused,
    const double * tw_re, const double * tw_im,
    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)
{
    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Ls; (void)Gs; (void)OGs;
    const size_t oL = 2*OLs;
    size_t k = 0;
    for (; k + 8 <= count; k += 8) {
        const double *pA = zin + 16*k;
        const double *pB = pA + 64;
        double *q = zout + 2*k;
        /* ---- phase 1: all 32 input loads up front (16 lines in flight) ---- */
        __m256d rl0A = _mm256_loadu_pd(pA +  0), il0A = _mm256_loadu_pd(pA +  4);
        __m256d rh0A = _mm256_loadu_pd(pA +  8), ih0A = _mm256_loadu_pd(pA + 12);
        __m256d rl1A = _mm256_loadu_pd(pA + 16), il1A = _mm256_loadu_pd(pA + 20);
        __m256d rh1A = _mm256_loadu_pd(pA + 24), ih1A = _mm256_loadu_pd(pA + 28);
        __m256d rl2A = _mm256_loadu_pd(pA + 32), il2A = _mm256_loadu_pd(pA + 36);
        __m256d rh2A = _mm256_loadu_pd(pA + 40), ih2A = _mm256_loadu_pd(pA + 44);
        __m256d rl3A = _mm256_loadu_pd(pA + 48), il3A = _mm256_loadu_pd(pA + 52);
        __m256d rh3A = _mm256_loadu_pd(pA + 56), ih3A = _mm256_loadu_pd(pA + 60);
        __m256d rl0B = _mm256_loadu_pd(pB +  0), il0B = _mm256_loadu_pd(pB +  4);
        __m256d rh0B = _mm256_loadu_pd(pB +  8), ih0B = _mm256_loadu_pd(pB + 12);
        __m256d rl1B = _mm256_loadu_pd(pB + 16), il1B = _mm256_loadu_pd(pB + 20);
        __m256d rh1B = _mm256_loadu_pd(pB + 24), ih1B = _mm256_loadu_pd(pB + 28);
        __m256d rl2B = _mm256_loadu_pd(pB + 32), il2B = _mm256_loadu_pd(pB + 36);
        __m256d rh2B = _mm256_loadu_pd(pB + 40), ih2B = _mm256_loadu_pd(pB + 44);
        __m256d rl3B = _mm256_loadu_pd(pB + 48), il3B = _mm256_loadu_pd(pB + 52);
        __m256d rh3B = _mm256_loadu_pd(pB + 56), ih3B = _mm256_loadu_pd(pB + 60);
        /* ---- phase 2: transposes (port-1/5 shuffle work under load latency) ---- */
        __m256d xr0A, xr1A, xr2A, xr3A, xr4A, xr5A, xr6A, xr7A;
        __m256d xi0A, xi1A, xi2A, xi3A, xi4A, xi5A, xi6A, xi7A;
        TR4(rl0A, rl1A, rl2A, rl3A, xr0A, xr1A, xr2A, xr3A);
        TR4(il0A, il1A, il2A, il3A, xi0A, xi1A, xi2A, xi3A);
        TR4(rh0A, rh1A, rh2A, rh3A, xr4A, xr5A, xr6A, xr7A);
        TR4(ih0A, ih1A, ih2A, ih3A, xi4A, xi5A, xi6A, xi7A);
        __m256d xr0B, xr1B, xr2B, xr3B, xr4B, xr5B, xr6B, xr7B;
        __m256d xi0B, xi1B, xi2B, xi3B, xi4B, xi5B, xi6B, xi7B;
        TR4(rl0B, rl1B, rl2B, rl3B, xr0B, xr1B, xr2B, xr3B);
        TR4(il0B, il1B, il2B, il3B, xi0B, xi1B, xi2B, xi3B);
        TR4(rh0B, rh1B, rh2B, rh3B, xr4B, xr5B, xr6B, xr7B);
        TR4(ih0B, ih1B, ih2B, ih3B, xi4B, xi5B, xi6B, xi7B);
        /* ---- phase 3: serial squaring tree + CMULs, A/B alternated op-by-op.
           Two independent WPROD latency chains keep FMA ports 0/1 saturated.
           Per-body op sequence identical to baseline:
           CMUL1, W2, CMUL2, W3, CMUL3, W4, CMUL4, W5, CMUL5, W6, CMUL6, W7, CMUL7. */
        __m256d c1A = _mm256_loadu_pd(tw_re + 2*k);
        __m256d s1A = _mm256_loadu_pd(tw_re + 2*k +  4);
        __m256d c1B = _mm256_loadu_pd(tw_re + 2*k +  8);
        __m256d s1B = _mm256_loadu_pd(tw_re + 2*k + 12);
        __m256d yr1A, yi1A, yr2A, yi2A, yr3A, yi3A, yr4A, yi4A,
                yr5A, yi5A, yr6A, yi6A, yr7A, yi7A;
        __m256d yr1B, yi1B, yr2B, yi2B, yr3B, yi3B, yr4B, yi4B,
                yr5B, yi5B, yr6B, yi6B, yr7B, yi7B;
        __m256d c2A, s2A, c3A, s3A, c4A, s4A, c5A, s5A, c6A, s6A, c7A, s7A;
        __m256d c2B, s2B, c3B, s3B, c4B, s4B, c5B, s5B, c6B, s6B, c7B, s7B;
        SPLIT_CMUL(xr1A, xi1A, c1A, s1A, yr1A, yi1A);
        SPLIT_CMUL(xr1B, xi1B, c1B, s1B, yr1B, yi1B);
        WPROD(c1A, s1A, c1A, s1A, c2A, s2A);
        WPROD(c1B, s1B, c1B, s1B, c2B, s2B);
        SPLIT_CMUL(xr2A, xi2A, c2A, s2A, yr2A, yi2A);
        SPLIT_CMUL(xr2B, xi2B, c2B, s2B, yr2B, yi2B);
        WPROD(c2A, s2A, c1A, s1A, c3A, s3A);
        WPROD(c2B, s2B, c1B, s1B, c3B, s3B);
        SPLIT_CMUL(xr3A, xi3A, c3A, s3A, yr3A, yi3A);
        SPLIT_CMUL(xr3B, xi3B, c3B, s3B, yr3B, yi3B);
        WPROD(c2A, s2A, c2A, s2A, c4A, s4A);
        WPROD(c2B, s2B, c2B, s2B, c4B, s4B);
        SPLIT_CMUL(xr4A, xi4A, c4A, s4A, yr4A, yi4A);
        SPLIT_CMUL(xr4B, xi4B, c4B, s4B, yr4B, yi4B);
        WPROD(c4A, s4A, c1A, s1A, c5A, s5A);
        WPROD(c4B, s4B, c1B, s1B, c5B, s5B);
        SPLIT_CMUL(xr5A, xi5A, c5A, s5A, yr5A, yi5A);
        SPLIT_CMUL(xr5B, xi5B, c5B, s5B, yr5B, yi5B);
        WPROD(c4A, s4A, c2A, s2A, c6A, s6A);
        WPROD(c4B, s4B, c2B, s2B, c6B, s6B);
        SPLIT_CMUL(xr6A, xi6A, c6A, s6A, yr6A, yi6A);
        SPLIT_CMUL(xr6B, xi6B, c6B, s6B, yr6B, yi6B);
        WPROD(c4A, s4A, c3A, s3A, c7A, s7A);
        WPROD(c4B, s4B, c3B, s3B, c7B, s7B);
        SPLIT_CMUL(xr7A, xi7A, c7A, s7A, yr7A, yi7A);
        SPLIT_CMUL(xr7B, xi7B, c7B, s7B, yr7B, yi7B);
        /* ---- phase 4: BFLY A + stores A first, so A's 16 outputs die before
           B's butterfly peaks; B's FMA work overlaps A's store drain ---- */
        __m256d o0rA,o0iA,o1rA,o1iA,o2rA,o2iA,o3rA,o3iA,
                o4rA,o4iA,o5rA,o5iA,o6rA,o6iA,o7rA,o7iA;
        SPLIT_BFLY8(xr0A,xi0A, yr1A,yi1A, yr2A,yi2A, yr3A,yi3A,
                    yr4A,yi4A, yr5A,yi5A, yr6A,yi6A, yr7A,yi7A,
                    o0rA,o0iA, o1rA,o1iA, o2rA,o2iA, o3rA,o3iA,
                    o4rA,o4iA, o5rA,o5iA, o6rA,o6iA, o7rA,o7iA);
        { __m256d zlo, zhi; REINT(o0rA,o0iA, zlo, zhi);
          _mm256_storeu_pd(q         , zlo); _mm256_storeu_pd(q          + 4, zhi); }
        { __m256d zlo, zhi; REINT(o1rA,o1iA, zlo, zhi);
          _mm256_storeu_pd(q +   oL  , zlo); _mm256_storeu_pd(q +   oL   + 4, zhi); }
        { __m256d zlo, zhi; REINT(o2rA,o2iA, zlo, zhi);
          _mm256_storeu_pd(q + 2*oL  , zlo); _mm256_storeu_pd(q + 2*oL   + 4, zhi); }
        { __m256d zlo, zhi; REINT(o3rA,o3iA, zlo, zhi);
          _mm256_storeu_pd(q + 3*oL  , zlo); _mm256_storeu_pd(q + 3*oL   + 4, zhi); }
        { __m256d zlo, zhi; REINT(o4rA,o4iA, zlo, zhi);
          _mm256_storeu_pd(q + 4*oL  , zlo); _mm256_storeu_pd(q + 4*oL   + 4, zhi); }
        { __m256d zlo, zhi; REINT(o5rA,o5iA, zlo, zhi);
          _mm256_storeu_pd(q + 5*oL  , zlo); _mm256_storeu_pd(q + 5*oL   + 4, zhi); }
        { __m256d zlo, zhi; REINT(o6rA,o6iA, zlo, zhi);
          _mm256_storeu_pd(q + 6*oL  , zlo); _mm256_storeu_pd(q + 6*oL   + 4, zhi); }
        { __m256d zlo, zhi; REINT(o7rA,o7iA, zlo, zhi);
          _mm256_storeu_pd(q + 7*oL  , zlo); _mm256_storeu_pd(q + 7*oL   + 4, zhi); }
        /* ---- phase 5: BFLY B + stores B (adjacent 64B lines to A's) ---- */
        __m256d o0rB,o0iB,o1rB,o1iB,o2rB,o2iB,o3rB,o3iB,
                o4rB,o4iB,o5rB,o5iB,o6rB,o6iB,o7rB,o7iB;
        SPLIT_BFLY8(xr0B,xi0B, yr1B,yi1B, yr2B,yi2B, yr3B,yi3B,
                    yr4B,yi4B, yr5B,yi5B, yr6B,yi6B, yr7B,yi7B,
                    o0rB,o0iB, o1rB,o1iB, o2rB,o2iB, o3rB,o3iB,
                    o4rB,o4iB, o5rB,o5iB, o6rB,o6iB, o7rB,o7iB);
        { __m256d zlo, zhi; REINT(o0rB,o0iB, zlo, zhi);
          _mm256_storeu_pd(q          + 8, zlo); _mm256_storeu_pd(q          + 12, zhi); }
        { __m256d zlo, zhi; REINT(o1rB,o1iB, zlo, zhi);
          _mm256_storeu_pd(q +   oL   + 8, zlo); _mm256_storeu_pd(q +   oL   + 12, zhi); }
        { __m256d zlo, zhi; REINT(o2rB,o2iB, zlo, zhi);
          _mm256_storeu_pd(q + 2*oL   + 8, zlo); _mm256_storeu_pd(q + 2*oL   + 12, zhi); }
        { __m256d zlo, zhi; REINT(o3rB,o3iB, zlo, zhi);
          _mm256_storeu_pd(q + 3*oL   + 8, zlo); _mm256_storeu_pd(q + 3*oL   + 12, zhi); }
        { __m256d zlo, zhi; REINT(o4rB,o4iB, zlo, zhi);
          _mm256_storeu_pd(q + 4*oL   + 8, zlo); _mm256_storeu_pd(q + 4*oL   + 12, zhi); }
        { __m256d zlo, zhi; REINT(o5rB,o5iB, zlo, zhi);
          _mm256_storeu_pd(q + 5*oL   + 8, zlo); _mm256_storeu_pd(q + 5*oL   + 12, zhi); }
        { __m256d zlo, zhi; REINT(o6rB,o6iB, zlo, zhi);
          _mm256_storeu_pd(q + 6*oL   + 8, zlo); _mm256_storeu_pd(q + 6*oL   + 12, zhi); }
        { __m256d zlo, zhi; REINT(o7rB,o7iB, zlo, zhi);
          _mm256_storeu_pd(q + 7*oL   + 8, zlo); _mm256_storeu_pd(q + 7*oL   + 12, zhi); }
    }
    /* ---- baseline-shaped 4-column tail (count % 8 == 4) ---- */
    for (; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8];
        {
            __m256d rl0 = _mm256_loadu_pd(zin + 16*k);
            __m256d il0 = _mm256_loadu_pd(zin + 16*k + 4);
            __m256d rh0 = _mm256_loadu_pd(zin + 16*k + 8);
            __m256d ih0 = _mm256_loadu_pd(zin + 16*k + 12);
            __m256d rl1 = _mm256_loadu_pd(zin + 16*(k+1));
            __m256d il1 = _mm256_loadu_pd(zin + 16*(k+1) + 4);
            __m256d rh1 = _mm256_loadu_pd(zin + 16*(k+1) + 8);
            __m256d ih1 = _mm256_loadu_pd(zin + 16*(k+1) + 12);
            __m256d rl2 = _mm256_loadu_pd(zin + 16*(k+2));
            __m256d il2 = _mm256_loadu_pd(zin + 16*(k+2) + 4);
            __m256d rh2 = _mm256_loadu_pd(zin + 16*(k+2) + 8);
            __m256d ih2 = _mm256_loadu_pd(zin + 16*(k+2) + 12);
            __m256d rl3 = _mm256_loadu_pd(zin + 16*(k+3));
            __m256d il3 = _mm256_loadu_pd(zin + 16*(k+3) + 4);
            __m256d rh3 = _mm256_loadu_pd(zin + 16*(k+3) + 8);
            __m256d ih3 = _mm256_loadu_pd(zin + 16*(k+3) + 12);
            TR4(rl0, rl1, rl2, rl3, xr[0], xr[1], xr[2], xr[3]);
            TR4(il0, il1, il2, il3, xi[0], xi[1], xi[2], xi[3]);
            TR4(rh0, rh1, rh2, rh3, xr[4], xr[5], xr[6], xr[7]);
            TR4(ih0, ih1, ih2, ih3, xi[4], xi[5], xi[6], xi[7]);
        }
        {
            __m256d c1 = _mm256_loadu_pd(tw_re + 2*k);
            __m256d s1 = _mm256_loadu_pd(tw_re + 2*k + 4);
            __m256d c2, s2, c3, s3, c4, s4, cw, sw, rr, ii;
            SPLIT_CMUL(xr[1], xi[1], c1, s1, rr, ii); xr[1] = rr; xi[1] = ii;
            WPROD(c1, s1, c1, s1, c2, s2);
            SPLIT_CMUL(xr[2], xi[2], c2, s2, rr, ii); xr[2] = rr; xi[2] = ii;
            WPROD(c2, s2, c1, s1, c3, s3);
            SPLIT_CMUL(xr[3], xi[3], c3, s3, rr, ii); xr[3] = rr; xi[3] = ii;
            WPROD(c2, s2, c2, s2, c4, s4);
            SPLIT_CMUL(xr[4], xi[4], c4, s4, rr, ii); xr[4] = rr; xi[4] = ii;
            WPROD(c4, s4, c1, s1, cw, sw);
            SPLIT_CMUL(xr[5], xi[5], cw, sw, rr, ii); xr[5] = rr; xi[5] = ii;
            WPROD(c4, s4, c2, s2, cw, sw);
            SPLIT_CMUL(xr[6], xi[6], cw, sw, rr, ii); xr[6] = rr; xi[6] = ii;
            WPROD(c4, s4, c3, s3, cw, sw);
            SPLIT_CMUL(xr[7], xi[7], cw, sw, rr, ii); xr[7] = rr; xi[7] = ii;
        }
        __m256d or_[8], oi_[8];
        SPLIT_BFLY8(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                    xr[4],xi[4],xr[5],xi[5],xr[6],xi[6],xr[7],xi[7],
                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3],
                    or_[4],oi_[4],or_[5],oi_[5],or_[6],oi_[6],or_[7],oi_[7]);
        for (int l = 0; l < 8; l++) {
            __m256d zlo, zhi;
            REINT(or_[l], oi_[l], zlo, zhi);
            _mm256_storeu_pd(zout + 2*((size_t)l*OLs + k), zlo);
            _mm256_storeu_pd(zout + 2*((size_t)l*OLs + k) + 4, zhi);
        }
    }
}

/* ================= ARM nt (sterm_fwd_nt, OK) ================= */
__attribute__((target("avx2,fma")))
void sterm_fwd_nt(
    const double * __restrict__ zin,
    const double * __restrict__ zin_unused,
    double       * __restrict__ zout,
    double       * __restrict__ zout_unused,
    const double * tw_re, const double * tw_im,
    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)
{
    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Ls; (void)Gs; (void)OGs;
    for (size_t k = 0; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8];
        {
            __m256d rl0 = _mm256_loadu_pd(zin + 16*(size_t)k);
            __m256d il0 = _mm256_loadu_pd(zin + 16*(size_t)k + 4);
            __m256d rh0 = _mm256_loadu_pd(zin + 16*(size_t)k + 8);
            __m256d ih0 = _mm256_loadu_pd(zin + 16*(size_t)k + 12);
            __m256d rl1 = _mm256_loadu_pd(zin + 16*((size_t)k+1));
            __m256d il1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 4);
            __m256d rh1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 8);
            __m256d ih1 = _mm256_loadu_pd(zin + 16*((size_t)k+1) + 12);
            __m256d rl2 = _mm256_loadu_pd(zin + 16*((size_t)k+2));
            __m256d il2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 4);
            __m256d rh2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 8);
            __m256d ih2 = _mm256_loadu_pd(zin + 16*((size_t)k+2) + 12);
            __m256d rl3 = _mm256_loadu_pd(zin + 16*((size_t)k+3));
            __m256d il3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 4);
            __m256d rh3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 8);
            __m256d ih3 = _mm256_loadu_pd(zin + 16*((size_t)k+3) + 12);
            TR4(rl0, rl1, rl2, rl3, xr[0], xr[1], xr[2], xr[3]);
            TR4(il0, il1, il2, il3, xi[0], xi[1], xi[2], xi[3]);
            TR4(rh0, rh1, rh2, rh3, xr[4], xr[5], xr[6], xr[7]);
            TR4(ih0, ih1, ih2, ih3, xi[4], xi[5], xi[6], xi[7]);
        }
        {
            __m256d c1 = _mm256_loadu_pd(tw_re + 2*(size_t)k);
            __m256d s1 = _mm256_loadu_pd(tw_re + 2*(size_t)k + 4);
            __m256d c2, s2, c3, s3, c4, s4, cw, sw, rr, ii;
            SPLIT_CMUL(xr[1], xi[1], c1, s1, rr, ii); xr[1] = rr; xi[1] = ii;
            WPROD(c1, s1, c1, s1, c2, s2);
            SPLIT_CMUL(xr[2], xi[2], c2, s2, rr, ii); xr[2] = rr; xi[2] = ii;
            WPROD(c2, s2, c1, s1, c3, s3);
            SPLIT_CMUL(xr[3], xi[3], c3, s3, rr, ii); xr[3] = rr; xi[3] = ii;
            WPROD(c2, s2, c2, s2, c4, s4);
            SPLIT_CMUL(xr[4], xi[4], c4, s4, rr, ii); xr[4] = rr; xi[4] = ii;
            WPROD(c4, s4, c1, s1, cw, sw);
            SPLIT_CMUL(xr[5], xi[5], cw, sw, rr, ii); xr[5] = rr; xi[5] = ii;
            WPROD(c4, s4, c2, s2, cw, sw);
            SPLIT_CMUL(xr[6], xi[6], cw, sw, rr, ii); xr[6] = rr; xi[6] = ii;
            WPROD(c4, s4, c3, s3, cw, sw);
            SPLIT_CMUL(xr[7], xi[7], cw, sw, rr, ii); xr[7] = rr; xi[7] = ii;
        }
        __m256d or_[8], oi_[8];
        SPLIT_BFLY8(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                    xr[4],xi[4],xr[5],xi[5],xr[6],xi[6],xr[7],xi[7],
                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3],
                    or_[4],oi_[4],or_[5],oi_[5],or_[6],oi_[6],or_[7],oi_[7]);
        for (int l = 0; l < 8; l++) {
            __m256d zlo, zhi;
            REINT(or_[l], oi_[l], zlo, zhi);
            _mm256_stream_pd(zout + 2*((size_t)l*OLs + k), zlo);
            _mm256_stream_pd(zout + 2*((size_t)l*OLs + k) + 4, zhi);
        }
    }
    _mm_sfence();
}


/* ===================== WORKFLOW ARMS PASTED ABOVE ===================== */

/* ---- BWD race: INV macros + baseline copy + uj2 twin ---- */
#define SPLIT_BFLY4_INV(i0r,i0i,i1r,i1i,i2r,i2i,i3r,i3i, o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i) do { \
    __m256d t0r=_mm256_add_pd(i0r,i2r), t0i=_mm256_add_pd(i0i,i2i);   \
    __m256d t1r=_mm256_sub_pd(i0r,i2r), t1i=_mm256_sub_pd(i0i,i2i);   \
    __m256d t2r=_mm256_add_pd(i1r,i3r), t2i=_mm256_add_pd(i1i,i3i);   \
    __m256d t3r=_mm256_sub_pd(i1r,i3r), t3i=_mm256_sub_pd(i1i,i3i);   \
    o0r=_mm256_add_pd(t0r,t2r); o0i=_mm256_add_pd(t0i,t2i);           \
    o2r=_mm256_sub_pd(t0r,t2r); o2i=_mm256_sub_pd(t0i,t2i);           \
    o1r=_mm256_sub_pd(t1r,t3i); o1i=_mm256_add_pd(t1i,t3r);  /* +(+i)t3 */ \
    o3r=_mm256_add_pd(t1r,t3i); o3i=_mm256_sub_pd(t1i,t3r);  /* -(+i)t3 */ \
} while (0)
#define SPLIT_BFLY8_INV(x0r,x0i,x1r,x1i,x2r,x2i,x3r,x3i,x4r,x4i,x5r,x5i,x6r,x6i,x7r,x7i, \
                    o0r,o0i,o1r,o1i,o2r,o2i,o3r,o3i,o4r,o4i,o5r,o5i,o6r,o6i,o7r,o7i) do { \
    const __m256d _C = _mm256_set1_pd(0.70710678118654752440);        \
    __m256d t0r=_mm256_add_pd(x0r,x4r), t0i=_mm256_add_pd(x0i,x4i);   \
    __m256d t1r=_mm256_sub_pd(x0r,x4r), t1i=_mm256_sub_pd(x0i,x4i);   \
    __m256d t2r=_mm256_add_pd(x2r,x6r), t2i=_mm256_add_pd(x2i,x6i);   \
    __m256d t3r=_mm256_sub_pd(x2r,x6r), t3i=_mm256_sub_pd(x2i,x6i);   \
    __m256d E0r=_mm256_add_pd(t0r,t2r), E0i=_mm256_add_pd(t0i,t2i);   \
    __m256d E2r=_mm256_sub_pd(t0r,t2r), E2i=_mm256_sub_pd(t0i,t2i);   \
    __m256d E1r=_mm256_sub_pd(t1r,t3i), E1i=_mm256_add_pd(t1i,t3r);   \
    __m256d E3r=_mm256_add_pd(t1r,t3i), E3i=_mm256_sub_pd(t1i,t3r);   \
    __m256d s0r=_mm256_add_pd(x1r,x5r), s0i=_mm256_add_pd(x1i,x5i);   \
    __m256d s1r=_mm256_sub_pd(x1r,x5r), s1i=_mm256_sub_pd(x1i,x5i);   \
    __m256d s2r=_mm256_add_pd(x3r,x7r), s2i=_mm256_add_pd(x3i,x7i);   \
    __m256d s3r=_mm256_sub_pd(x3r,x7r), s3i=_mm256_sub_pd(x3i,x7i);   \
    __m256d O0r=_mm256_add_pd(s0r,s2r), O0i=_mm256_add_pd(s0i,s2i);   \
    __m256d O2r=_mm256_sub_pd(s0r,s2r), O2i=_mm256_sub_pd(s0i,s2i);   \
    __m256d O1r=_mm256_sub_pd(s1r,s3i), O1i=_mm256_add_pd(s1i,s3r);   \
    __m256d O3r=_mm256_add_pd(s1r,s3i), O3i=_mm256_sub_pd(s1i,s3r);   \
    __m256d X1r=_mm256_sub_pd(O1r,O1i), X1i=_mm256_add_pd(O1i,O1r);   \
    __m256d X3r=_mm256_add_pd(O3i,O3r), X3n=_mm256_sub_pd(O3r,O3i);   \
    o0r=_mm256_add_pd(E0r,O0r); o0i=_mm256_add_pd(E0i,O0i);           \
    o4r=_mm256_sub_pd(E0r,O0r); o4i=_mm256_sub_pd(E0i,O0i);           \
    o1r=_mm256_fmadd_pd(_C,X1r,E1r); o1i=_mm256_fmadd_pd(_C,X1i,E1i); \
    o5r=_mm256_fnmadd_pd(_C,X1r,E1r); o5i=_mm256_fnmadd_pd(_C,X1i,E1i); \
    o2r=_mm256_sub_pd(E2r,O2i); o2i=_mm256_add_pd(E2i,O2r);           \
    o6r=_mm256_add_pd(E2r,O2i); o6i=_mm256_sub_pd(E2i,O2r);           \
    o3r=_mm256_fnmadd_pd(_C,X3r,E3r); o3i=_mm256_fmadd_pd(_C,X3n,E3i); \
    o7r=_mm256_fmadd_pd(_C,X3r,E3r); o7i=_mm256_fnmadd_pd(_C,X3n,E3i); \
} while (0)

__attribute__((target("avx2,fma")))
static void sterm_bwd_copy(
    const double * __restrict__ zin,
    const double * __restrict__ zin_unused,
    double       * __restrict__ zout,
    double       * __restrict__ zout_unused,
    const double * tw_re, const double * tw_im,
    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)
{
    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Ls; (void)Gs; (void)OGs;
    for (size_t k = 0; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8];
        for (int l = 0; l < 8; l++) {
            __m256d zlo = _mm256_loadu_pd(zin + 2*((size_t)l*OLs + k));
            __m256d zhi = _mm256_loadu_pd(zin + 2*((size_t)l*OLs + k) + 4);
            DEINT(zlo, zhi, xr[l], xi[l]);
        }
        __m256d or_[8], oi_[8];
        SPLIT_BFLY8_INV(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                    xr[4],xi[4],xr[5],xi[5],xr[6],xi[6],xr[7],xi[7],
                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3],
                    or_[4],oi_[4],or_[5],oi_[5],or_[6],oi_[6],or_[7],oi_[7]);
        {
            __m256d c1 = _mm256_loadu_pd(tw_re + 2*(size_t)k);
            __m256d s1 = _mm256_loadu_pd(tw_re + 2*(size_t)k + 4);
            __m256d c2, s2, c3, s3, c4, s4, cw, sw, rr, ii;
            SPLIT_CMUL(or_[1], oi_[1], c1, s1, rr, ii); or_[1] = rr; oi_[1] = ii;
            WPROD(c1, s1, c1, s1, c2, s2);
            SPLIT_CMUL(or_[2], oi_[2], c2, s2, rr, ii); or_[2] = rr; oi_[2] = ii;
            WPROD(c2, s2, c1, s1, c3, s3);
            SPLIT_CMUL(or_[3], oi_[3], c3, s3, rr, ii); or_[3] = rr; oi_[3] = ii;
            WPROD(c2, s2, c2, s2, c4, s4);
            SPLIT_CMUL(or_[4], oi_[4], c4, s4, rr, ii); or_[4] = rr; oi_[4] = ii;
            WPROD(c4, s4, c1, s1, cw, sw);
            SPLIT_CMUL(or_[5], oi_[5], cw, sw, rr, ii); or_[5] = rr; oi_[5] = ii;
            WPROD(c4, s4, c2, s2, cw, sw);
            SPLIT_CMUL(or_[6], oi_[6], cw, sw, rr, ii); or_[6] = rr; oi_[6] = ii;
            WPROD(c4, s4, c3, s3, cw, sw);
            SPLIT_CMUL(or_[7], oi_[7], cw, sw, rr, ii); or_[7] = rr; oi_[7] = ii;
        }
        {
            __m256d b0, b1, b2, b3;
            TR4(or_[0], or_[1], or_[2], or_[3], b0, b1, b2, b3);
            _mm256_storeu_pd(zout + 16*(size_t)k,        b0);
            _mm256_storeu_pd(zout + 16*((size_t)k + 1),  b1);
            _mm256_storeu_pd(zout + 16*((size_t)k + 2),  b2);
            _mm256_storeu_pd(zout + 16*((size_t)k + 3),  b3);
            TR4(oi_[0], oi_[1], oi_[2], oi_[3], b0, b1, b2, b3);
            _mm256_storeu_pd(zout + 16*(size_t)k + 4,       b0);
            _mm256_storeu_pd(zout + 16*((size_t)k + 1) + 4, b1);
            _mm256_storeu_pd(zout + 16*((size_t)k + 2) + 4, b2);
            _mm256_storeu_pd(zout + 16*((size_t)k + 3) + 4, b3);
            TR4(or_[4], or_[5], or_[6], or_[7], b0, b1, b2, b3);
            _mm256_storeu_pd(zout + 16*(size_t)k + 8,       b0);
            _mm256_storeu_pd(zout + 16*((size_t)k + 1) + 8, b1);
            _mm256_storeu_pd(zout + 16*((size_t)k + 2) + 8, b2);
            _mm256_storeu_pd(zout + 16*((size_t)k + 3) + 8, b3);
            TR4(oi_[4], oi_[5], oi_[6], oi_[7], b0, b1, b2, b3);
            _mm256_storeu_pd(zout + 16*(size_t)k + 12,       b0);
            _mm256_storeu_pd(zout + 16*((size_t)k + 1) + 12, b1);
            _mm256_storeu_pd(zout + 16*((size_t)k + 2) + 12, b2);
            _mm256_storeu_pd(zout + 16*((size_t)k + 3) + 12, b3);
        }
    }
}

/* ARM uj2b - 2-quad unroll-and-jam of the INVERSE split terminator
 * (radix8_z_sterm_bwd_avx2). Two independent 4-column bodies per trip:
 * A = cols k..k+3, B = cols k+4..k+7 -- the fwd sterm uj2 winner's
 * granularity applied to the reversed dataflow.
 * Per-column FP operation order is exactly the baseline's
 * (DEINT / SPLIT_BFLY8_INV / WPROD / SPLIT_CMUL / TR4: same operands,
 * same association order); only cross-column scheduling, scalarized
 * staging and loop structure differ.
 * Interleave: [all 32 leg-stream loads A+B -- 16 lines of load MLP]
 * [DEINT A] [DEINT B] [INV BFLY A] [INV BFLY B]
 * [conj twiddle squaring tree + CMULs alternated A/B op-by-op]
 * [TR4 + 16 contiguous stores A] [TR4 + 16 contiguous stores B].
 * Tail: baseline-shaped 4-column body for count % 8 == 4. */
__attribute__((target("avx2,fma")))
static void sterm_bwd_uj2(
    const double * __restrict__ zin,
    const double * __restrict__ zin_unused,
    double       * __restrict__ zout,
    double       * __restrict__ zout_unused,
    const double * tw_re, const double * tw_im,
    size_t Ls, size_t Gs, size_t OLs, size_t OGs, size_t count)
{
    (void)zin_unused; (void)zout_unused; (void)tw_im; (void)Ls; (void)Gs; (void)OGs;
    const size_t oL = 2*OLs;
    size_t k = 0;
    for (; k + 8 <= count; k += 8) {
        const double *p  = zin  + 2*k;   /* leg-stream base for col k; B cols are +8 */
        double       *qA = zout + 16*k;  /* contiguous output blocks, cols k..k+3   */
        double       *qB = qA + 64;      /* cols k+4..k+7 */
        /* ---- phase 1: all 32 leg-stream loads up front. The 8 leg streams
           are oL doubles (16*OLs bytes) apart -> 16 distinct lines in flight
           before any compute; this is where bwd stalls. ---- */
        __m256d zl0A = _mm256_loadu_pd(p       ), zh0A = _mm256_loadu_pd(p        + 4);
        __m256d zl1A = _mm256_loadu_pd(p +   oL), zh1A = _mm256_loadu_pd(p +   oL + 4);
        __m256d zl2A = _mm256_loadu_pd(p + 2*oL), zh2A = _mm256_loadu_pd(p + 2*oL + 4);
        __m256d zl3A = _mm256_loadu_pd(p + 3*oL), zh3A = _mm256_loadu_pd(p + 3*oL + 4);
        __m256d zl4A = _mm256_loadu_pd(p + 4*oL), zh4A = _mm256_loadu_pd(p + 4*oL + 4);
        __m256d zl5A = _mm256_loadu_pd(p + 5*oL), zh5A = _mm256_loadu_pd(p + 5*oL + 4);
        __m256d zl6A = _mm256_loadu_pd(p + 6*oL), zh6A = _mm256_loadu_pd(p + 6*oL + 4);
        __m256d zl7A = _mm256_loadu_pd(p + 7*oL), zh7A = _mm256_loadu_pd(p + 7*oL + 4);
        __m256d zl0B = _mm256_loadu_pd(p        + 8), zh0B = _mm256_loadu_pd(p        + 12);
        __m256d zl1B = _mm256_loadu_pd(p +   oL + 8), zh1B = _mm256_loadu_pd(p +   oL + 12);
        __m256d zl2B = _mm256_loadu_pd(p + 2*oL + 8), zh2B = _mm256_loadu_pd(p + 2*oL + 12);
        __m256d zl3B = _mm256_loadu_pd(p + 3*oL + 8), zh3B = _mm256_loadu_pd(p + 3*oL + 12);
        __m256d zl4B = _mm256_loadu_pd(p + 4*oL + 8), zh4B = _mm256_loadu_pd(p + 4*oL + 12);
        __m256d zl5B = _mm256_loadu_pd(p + 5*oL + 8), zh5B = _mm256_loadu_pd(p + 5*oL + 12);
        __m256d zl6B = _mm256_loadu_pd(p + 6*oL + 8), zh6B = _mm256_loadu_pd(p + 6*oL + 12);
        __m256d zl7B = _mm256_loadu_pd(p + 7*oL + 8), zh7B = _mm256_loadu_pd(p + 7*oL + 12);
        /* ---- phase 2: deinterleave (port-1/5 shuffle work under load latency) ---- */
        __m256d xr0A, xr1A, xr2A, xr3A, xr4A, xr5A, xr6A, xr7A;
        __m256d xi0A, xi1A, xi2A, xi3A, xi4A, xi5A, xi6A, xi7A;
        DEINT(zl0A, zh0A, xr0A, xi0A);
        DEINT(zl1A, zh1A, xr1A, xi1A);
        DEINT(zl2A, zh2A, xr2A, xi2A);
        DEINT(zl3A, zh3A, xr3A, xi3A);
        DEINT(zl4A, zh4A, xr4A, xi4A);
        DEINT(zl5A, zh5A, xr5A, xi5A);
        DEINT(zl6A, zh6A, xr6A, xi6A);
        DEINT(zl7A, zh7A, xr7A, xi7A);
        __m256d xr0B, xr1B, xr2B, xr3B, xr4B, xr5B, xr6B, xr7B;
        __m256d xi0B, xi1B, xi2B, xi3B, xi4B, xi5B, xi6B, xi7B;
        DEINT(zl0B, zh0B, xr0B, xi0B);
        DEINT(zl1B, zh1B, xr1B, xi1B);
        DEINT(zl2B, zh2B, xr2B, xi2B);
        DEINT(zl3B, zh3B, xr3B, xi3B);
        DEINT(zl4B, zh4B, xr4B, xi4B);
        DEINT(zl5B, zh5B, xr5B, xi5B);
        DEINT(zl6B, zh6B, xr6B, xi6B);
        DEINT(zl7B, zh7B, xr7B, xi7B);
        /* ---- phase 3: inverse radix-8 butterflies, A then B ---- */
        __m256d o0rA,o0iA,o1rA,o1iA,o2rA,o2iA,o3rA,o3iA,
                o4rA,o4iA,o5rA,o5iA,o6rA,o6iA,o7rA,o7iA;
        SPLIT_BFLY8_INV(xr0A,xi0A, xr1A,xi1A, xr2A,xi2A, xr3A,xi3A,
                        xr4A,xi4A, xr5A,xi5A, xr6A,xi6A, xr7A,xi7A,
                        o0rA,o0iA, o1rA,o1iA, o2rA,o2iA, o3rA,o3iA,
                        o4rA,o4iA, o5rA,o5iA, o6rA,o6iA, o7rA,o7iA);
        __m256d o0rB,o0iB,o1rB,o1iB,o2rB,o2iB,o3rB,o3iB,
                o4rB,o4iB,o5rB,o5iB,o6rB,o6iB,o7rB,o7iB;
        SPLIT_BFLY8_INV(xr0B,xi0B, xr1B,xi1B, xr2B,xi2B, xr3B,xi3B,
                        xr4B,xi4B, xr5B,xi5B, xr6B,xi6B, xr7B,xi7B,
                        o0rB,o0iB, o1rB,o1iB, o2rB,o2iB, o3rB,o3iB,
                        o4rB,o4iB, o5rB,o5iB, o6rB,o6iB, o7rB,o7iB);
        /* ---- phase 4: conjugated (table-side) twiddle POST-multiply; two
           independent serial squaring trees alternated op-by-op so both FMA
           ports stay fed. Per-body op sequence identical to baseline:
           CMUL1, W2, CMUL2, W3, CMUL3, W4, CMUL4, W5, CMUL5, W6, CMUL6, W7, CMUL7. */
        __m256d c1A = _mm256_loadu_pd(tw_re + 2*k);
        __m256d s1A = _mm256_loadu_pd(tw_re + 2*k +  4);
        __m256d c1B = _mm256_loadu_pd(tw_re + 2*k +  8);
        __m256d s1B = _mm256_loadu_pd(tw_re + 2*k + 12);
        __m256d y1rA,y1iA, y2rA,y2iA, y3rA,y3iA, y4rA,y4iA,
                y5rA,y5iA, y6rA,y6iA, y7rA,y7iA;
        __m256d y1rB,y1iB, y2rB,y2iB, y3rB,y3iB, y4rB,y4iB,
                y5rB,y5iB, y6rB,y6iB, y7rB,y7iB;
        __m256d c2A,s2A, c3A,s3A, c4A,s4A, c5A,s5A, c6A,s6A, c7A,s7A;
        __m256d c2B,s2B, c3B,s3B, c4B,s4B, c5B,s5B, c6B,s6B, c7B,s7B;
        SPLIT_CMUL(o1rA, o1iA, c1A, s1A, y1rA, y1iA);
        SPLIT_CMUL(o1rB, o1iB, c1B, s1B, y1rB, y1iB);
        WPROD(c1A, s1A, c1A, s1A, c2A, s2A);
        WPROD(c1B, s1B, c1B, s1B, c2B, s2B);
        SPLIT_CMUL(o2rA, o2iA, c2A, s2A, y2rA, y2iA);
        SPLIT_CMUL(o2rB, o2iB, c2B, s2B, y2rB, y2iB);
        WPROD(c2A, s2A, c1A, s1A, c3A, s3A);
        WPROD(c2B, s2B, c1B, s1B, c3B, s3B);
        SPLIT_CMUL(o3rA, o3iA, c3A, s3A, y3rA, y3iA);
        SPLIT_CMUL(o3rB, o3iB, c3B, s3B, y3rB, y3iB);
        WPROD(c2A, s2A, c2A, s2A, c4A, s4A);
        WPROD(c2B, s2B, c2B, s2B, c4B, s4B);
        SPLIT_CMUL(o4rA, o4iA, c4A, s4A, y4rA, y4iA);
        SPLIT_CMUL(o4rB, o4iB, c4B, s4B, y4rB, y4iB);
        WPROD(c4A, s4A, c1A, s1A, c5A, s5A);
        WPROD(c4B, s4B, c1B, s1B, c5B, s5B);
        SPLIT_CMUL(o5rA, o5iA, c5A, s5A, y5rA, y5iA);
        SPLIT_CMUL(o5rB, o5iB, c5B, s5B, y5rB, y5iB);
        WPROD(c4A, s4A, c2A, s2A, c6A, s6A);
        WPROD(c4B, s4B, c2B, s2B, c6B, s6B);
        SPLIT_CMUL(o6rA, o6iA, c6A, s6A, y6rA, y6iA);
        SPLIT_CMUL(o6rB, o6iB, c6B, s6B, y6rB, y6iB);
        WPROD(c4A, s4A, c3A, s3A, c7A, s7A);
        WPROD(c4B, s4B, c3B, s3B, c7B, s7B);
        SPLIT_CMUL(o7rA, o7iA, c7A, s7A, y7rA, y7iA);
        SPLIT_CMUL(o7rB, o7iB, c7B, s7B, y7rB, y7iB);
        /* ---- phase 5: TR4 + stores A first, so A's 16 values die before
           B's transpose peaks; B's shuffle work overlaps A's store drain ---- */
        {
            __m256d b0, b1, b2, b3;
            TR4(o0rA, y1rA, y2rA, y3rA, b0, b1, b2, b3);
            _mm256_storeu_pd(qA     , b0);
            _mm256_storeu_pd(qA + 16, b1);
            _mm256_storeu_pd(qA + 32, b2);
            _mm256_storeu_pd(qA + 48, b3);
            TR4(o0iA, y1iA, y2iA, y3iA, b0, b1, b2, b3);
            _mm256_storeu_pd(qA +  4, b0);
            _mm256_storeu_pd(qA + 20, b1);
            _mm256_storeu_pd(qA + 36, b2);
            _mm256_storeu_pd(qA + 52, b3);
            TR4(y4rA, y5rA, y6rA, y7rA, b0, b1, b2, b3);
            _mm256_storeu_pd(qA +  8, b0);
            _mm256_storeu_pd(qA + 24, b1);
            _mm256_storeu_pd(qA + 40, b2);
            _mm256_storeu_pd(qA + 56, b3);
            TR4(y4iA, y5iA, y6iA, y7iA, b0, b1, b2, b3);
            _mm256_storeu_pd(qA + 12, b0);
            _mm256_storeu_pd(qA + 28, b1);
            _mm256_storeu_pd(qA + 44, b2);
            _mm256_storeu_pd(qA + 60, b3);
        }
        /* ---- phase 6: TR4 + stores B (next four contiguous 128B blocks) ---- */
        {
            __m256d b0, b1, b2, b3;
            TR4(o0rB, y1rB, y2rB, y3rB, b0, b1, b2, b3);
            _mm256_storeu_pd(qB     , b0);
            _mm256_storeu_pd(qB + 16, b1);
            _mm256_storeu_pd(qB + 32, b2);
            _mm256_storeu_pd(qB + 48, b3);
            TR4(o0iB, y1iB, y2iB, y3iB, b0, b1, b2, b3);
            _mm256_storeu_pd(qB +  4, b0);
            _mm256_storeu_pd(qB + 20, b1);
            _mm256_storeu_pd(qB + 36, b2);
            _mm256_storeu_pd(qB + 52, b3);
            TR4(y4rB, y5rB, y6rB, y7rB, b0, b1, b2, b3);
            _mm256_storeu_pd(qB +  8, b0);
            _mm256_storeu_pd(qB + 24, b1);
            _mm256_storeu_pd(qB + 40, b2);
            _mm256_storeu_pd(qB + 56, b3);
            TR4(y4iB, y5iB, y6iB, y7iB, b0, b1, b2, b3);
            _mm256_storeu_pd(qB + 12, b0);
            _mm256_storeu_pd(qB + 28, b1);
            _mm256_storeu_pd(qB + 44, b2);
            _mm256_storeu_pd(qB + 60, b3);
        }
    }
    /* ---- baseline-shaped 4-column tail (count % 8 == 4) ---- */
    for (; k + 4 <= count; k += 4) {
        __m256d xr[8], xi[8];
        for (int l = 0; l < 8; l++) {
            __m256d zlo = _mm256_loadu_pd(zin + 2*((size_t)l*OLs + k));
            __m256d zhi = _mm256_loadu_pd(zin + 2*((size_t)l*OLs + k) + 4);
            DEINT(zlo, zhi, xr[l], xi[l]);
        }
        __m256d or_[8], oi_[8];
        SPLIT_BFLY8_INV(xr[0],xi[0],xr[1],xi[1],xr[2],xi[2],xr[3],xi[3],
                    xr[4],xi[4],xr[5],xi[5],xr[6],xi[6],xr[7],xi[7],
                    or_[0],oi_[0],or_[1],oi_[1],or_[2],oi_[2],or_[3],oi_[3],
                    or_[4],oi_[4],or_[5],oi_[5],or_[6],oi_[6],or_[7],oi_[7]);
        {
            __m256d c1 = _mm256_loadu_pd(tw_re + 2*(size_t)k);
            __m256d s1 = _mm256_loadu_pd(tw_re + 2*(size_t)k + 4);
            __m256d c2, s2, c3, s3, c4, s4, cw, sw, rr, ii;
            SPLIT_CMUL(or_[1], oi_[1], c1, s1, rr, ii); or_[1] = rr; oi_[1] = ii;
            WPROD(c1, s1, c1, s1, c2, s2);
            SPLIT_CMUL(or_[2], oi_[2], c2, s2, rr, ii); or_[2] = rr; oi_[2] = ii;
            WPROD(c2, s2, c1, s1, c3, s3);
            SPLIT_CMUL(or_[3], oi_[3], c3, s3, rr, ii); or_[3] = rr; oi_[3] = ii;
            WPROD(c2, s2, c2, s2, c4, s4);
            SPLIT_CMUL(or_[4], oi_[4], c4, s4, rr, ii); or_[4] = rr; oi_[4] = ii;
            WPROD(c4, s4, c1, s1, cw, sw);
            SPLIT_CMUL(or_[5], oi_[5], cw, sw, rr, ii); or_[5] = rr; oi_[5] = ii;
            WPROD(c4, s4, c2, s2, cw, sw);
            SPLIT_CMUL(or_[6], oi_[6], cw, sw, rr, ii); or_[6] = rr; oi_[6] = ii;
            WPROD(c4, s4, c3, s3, cw, sw);
            SPLIT_CMUL(or_[7], oi_[7], cw, sw, rr, ii); or_[7] = rr; oi_[7] = ii;
        }
        {
            __m256d b0, b1, b2, b3;
            TR4(or_[0], or_[1], or_[2], or_[3], b0, b1, b2, b3);
            _mm256_storeu_pd(zout + 16*(size_t)k,        b0);
            _mm256_storeu_pd(zout + 16*((size_t)k + 1),  b1);
            _mm256_storeu_pd(zout + 16*((size_t)k + 2),  b2);
            _mm256_storeu_pd(zout + 16*((size_t)k + 3),  b3);
            TR4(oi_[0], oi_[1], oi_[2], oi_[3], b0, b1, b2, b3);
            _mm256_storeu_pd(zout + 16*(size_t)k + 4,       b0);
            _mm256_storeu_pd(zout + 16*((size_t)k + 1) + 4, b1);
            _mm256_storeu_pd(zout + 16*((size_t)k + 2) + 4, b2);
            _mm256_storeu_pd(zout + 16*((size_t)k + 3) + 4, b3);
            TR4(or_[4], or_[5], or_[6], or_[7], b0, b1, b2, b3);
            _mm256_storeu_pd(zout + 16*(size_t)k + 8,       b0);
            _mm256_storeu_pd(zout + 16*((size_t)k + 1) + 8, b1);
            _mm256_storeu_pd(zout + 16*((size_t)k + 2) + 8, b2);
            _mm256_storeu_pd(zout + 16*((size_t)k + 3) + 8, b3);
            TR4(oi_[4], oi_[5], oi_[6], oi_[7], b0, b1, b2, b3);
            _mm256_storeu_pd(zout + 16*(size_t)k + 12,       b0);
            _mm256_storeu_pd(zout + 16*((size_t)k + 1) + 12, b1);
            _mm256_storeu_pd(zout + 16*((size_t)k + 2) + 12, b2);
            _mm256_storeu_pd(zout + 16*((size_t)k + 3) + 12, b3);
        }
    }
}


/* ---- cascade helpers: mirror vfft_zsplit_execute_fwd with swappable term */
static void run_front(const vfft_zsplit_plan_t *p, const double *zin)
{
    const int nf = p->nf;
    if (p->chain[0] == 8)
        radix8_z_s0s_fwd_avx2(zin, 0, p->sp, 0, 0, 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0]);
    else
        radix4_z_s0s_fwd_avx2(zin, 0, p->sp, 0, 0, 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0]);
    for (int s = 1; s <= nf - 2; s++) {
        zfn f = (p->chain[s] == 8) ? radix8_z_msg_fwd_avx2
                                   : radix4_z_msg_fwd_avx2;
        f(p->sp, 0, p->sp, 0, p->twsp[s], 0,
          (unsigned long long)p->D[s], (unsigned long long)p->G[s],
          0, 0, (unsigned long long)p->D[s]);
    }
}
static void exec_term(const vfft_zsplit_plan_t *p, double *zout, zfn term)
{
    term(p->sp, 0, zout, 0, p->twq, 0, 0, 0,
         (unsigned long long)(p->N / 8), 0, (unsigned long long)(p->N / 8));
}
static void run_back(const vfft_zsplit_plan_t *p, const double *zin, double *zout,
                     zfn term)
{
    const int nf = p->nf;
    term(zin, 0, p->sp, 0, p->twqb, 0, 0, 0,
         (unsigned long long)(p->N / 8), 0, (unsigned long long)(p->N / 8));
    for (int s = nf - 2; s >= 1; s--) {
        zfn f = (p->chain[s] == 8) ? radix8_z_msg_bwd_avx2
                                   : radix4_z_msg_bwd_avx2;
        f(p->sp, 0, p->sp, 0, p->twspb[s], 0,
          (unsigned long long)p->D[s], (unsigned long long)p->G[s],
          0, 0, (unsigned long long)p->D[s]);
    }
    if (p->chain[0] == 8)
        radix8_z_s0s_bwd_avx2(p->sp, 0, zout, 0, 0, 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0]);
    else
        radix4_z_s0s_bwd_avx2(p->sp, 0, zout, 0, 0, 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0], 0,
                              (unsigned long long)p->D[0]);
}

/* ---- harness ---- */
static double qpc_ms(void)
{
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f); QueryPerformanceCounter(&c);
    return 1000.0 * (double)c.QuadPart / (double)f.QuadPart;
}
static char *g_bust;
#define BUST_SZ (32u * 1024u * 1024u)
static void cachebust(void)
{
    for (size_t i = 0; i < BUST_SZ; i += 64) g_bust[i]++;
}

typedef struct { const char *name; zfn fn; } arm_t;

int main(void)
{
    SetThreadAffinityMask(GetCurrentThread(), (DWORD_PTR)4);
    SetPriorityClass(GetCurrentProcess(), HIGH_PRIORITY_CLASS);
    g_bust = (char *)malloc(BUST_SZ);
    memset(g_bust, 1, BUST_SZ);

    arm_t arms[] = {
        { "emit",  radix8_z_sterm_fwd_avx2 },
        { "copy",  sterm_fwd_copy },
        { "rot",   (zfn)sterm_fwd_rot },
        { "phase", (zfn)sterm_fwd_phase },
        { "pfw",   (zfn)sterm_fwd_pfw },
        { "uj2",   (zfn)sterm_fwd_uj2 },
        { "nt",    (zfn)sterm_fwd_nt },
    };
    const int NA = (int)(sizeof(arms) / sizeof(arms[0]));
    const int cells[] = { 2048, 4096, 8192, 16384 };
    const int reps_full[] = { 300, 150, 80, 40 };
    const int ROUNDS = 7;
    int rc = 0;

    for (int ci = 0; ci < 4; ci++) {
        const int N = cells[ci];
        int chain[VFFT_ZSPLIT_MAX_NF];
        int nf = vfft_zsplit_default_chain(N, chain);
        vfft_zsplit_plan_t *p = vfft_zsplit_create(N, chain, nf);
        if (!p) { printf("N=%d create FAIL\n", N); return 1; }

        double *in  = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
        double *ref = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
        double *out = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
        double *bref = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
        double *bout = (double *)_aligned_malloc((size_t)2 * N * 8, 64);
        srand(20260725 + N);
        for (int i = 0; i < 2 * N; i++)
            in[i] = (double)rand() / RAND_MAX - 0.5;

        /* gates: every arm bit-identical to the production fwd */
        vfft_zsplit_execute_fwd(p, in, ref);
        for (int a = 0; a < NA; a++) {
            memset(out, 0, (size_t)2 * N * 8);
            run_front(p, in);
            exec_term(p, out, arms[a].fn);
            if (memcmp(out, ref, (size_t)2 * N * 8) != 0) {
                printf("N=%-6d GATE %-5s BIT-MISMATCH FAIL\n", N, arms[a].name);
                rc = 1;
            } else {
                printf("N=%-6d GATE %-5s bit-identical PASS\n", N, arms[a].name);
            }
        }
        if (rc) { printf("gates failed at N=%d — not timing\n", N); return rc; }

        /* race: full cascade + terminator-only, rotated arms, best-of-ROUNDS */
        double best_full[16], best_term[16];
        for (int a = 0; a < NA; a++) { best_full[a] = 1e30; best_term[a] = 1e30; }
        const int RF = reps_full[ci], RT = RF * 3;
        for (int r = 0; r < ROUNDS; r++) {
            for (int j = 0; j < NA; j++) {
                int a = (j + r) % NA;               /* order-neutral rotation */
                cachebust();
                run_front(p, in);                    /* warm + fresh sp */
                exec_term(p, out, arms[a].fn);
                double t0 = qpc_ms();
                for (int i = 0; i < RF; i++) {
                    run_front(p, in);
                    exec_term(p, out, arms[a].fn);
                }
                double ns = (qpc_ms() - t0) * 1e6 / RF;
                if (ns < best_full[a]) best_full[a] = ns;

                cachebust();
                run_front(p, in);                    /* sp hot, like real life */
                exec_term(p, out, arms[a].fn);
                t0 = qpc_ms();
                for (int i = 0; i < RT; i++)
                    exec_term(p, out, arms[a].fn);
                ns = (qpc_ms() - t0) * 1e6 / RT;
                if (ns < best_term[a]) best_term[a] = ns;
                Sleep(60);
            }
            Sleep(150);
        }
        for (int a = 0; a < NA; a++)
            printf("N=%-6d %-5s full %9.1f ns (%+6.2f%%)   term %8.1f ns (%+6.2f%%)\n",
                   N, arms[a].name,
                   best_full[a], 100.0 * (best_full[a] / best_full[0] - 1.0),
                   best_term[a], 100.0 * (best_term[a] / best_term[0] - 1.0));
        /* ================== BWD terminator race ================== */
        {
            arm_t barms[] = {
                { "bemit", radix8_z_sterm_bwd_avx2 },
                { "bcopy", (zfn)sterm_bwd_copy },
                { "buj2",  (zfn)sterm_bwd_uj2 },
            };
            const int NB = (int)(sizeof(barms) / sizeof(barms[0]));
            vfft_zsplit_execute_bwd(p, ref, bref);      /* production reference */
            for (int a = 0; a < NB; a++) {
                memset(bout, 0, (size_t)2 * N * 8);
                run_back(p, ref, bout, barms[a].fn);
                if (memcmp(bout, bref, (size_t)2 * N * 8) != 0) {
                    printf("N=%-6d GATE %-5s BIT-MISMATCH FAIL\n", N, barms[a].name);
                    rc = 1;
                } else {
                    printf("N=%-6d GATE %-5s bit-identical PASS\n", N, barms[a].name);
                }
            }
            if (rc) { printf("bwd gates failed at N=%d\n", N); return rc; }
            double bbf[8], bbt[8];
            for (int a = 0; a < NB; a++) { bbf[a] = 1e30; bbt[a] = 1e30; }
            const int RF = reps_full[ci], RT = RF * 3;
            for (int r = 0; r < ROUNDS; r++) {
                for (int j = 0; j < NB; j++) {
                    int a = (j + r) % NB;
                    cachebust();
                    run_back(p, ref, bout, barms[a].fn);
                    double t0 = qpc_ms();
                    for (int i = 0; i < RF; i++)
                        run_back(p, ref, bout, barms[a].fn);
                    double ns = (qpc_ms() - t0) * 1e6 / RF;
                    if (ns < bbf[a]) bbf[a] = ns;

                    cachebust();
                    barms[a].fn(ref, 0, p->sp, 0, p->twqb, 0, 0, 0,
                                (unsigned long long)(N / 8), 0,
                                (unsigned long long)(N / 8));
                    t0 = qpc_ms();
                    for (int i = 0; i < RT; i++)
                        barms[a].fn(ref, 0, p->sp, 0, p->twqb, 0, 0, 0,
                                    (unsigned long long)(N / 8), 0,
                                    (unsigned long long)(N / 8));
                    ns = (qpc_ms() - t0) * 1e6 / RT;
                    if (ns < bbt[a]) bbt[a] = ns;
                    Sleep(60);
                }
                Sleep(150);
            }
            for (int a = 0; a < NB; a++)
                printf("N=%-6d %-5s bwdF %9.1f ns (%+6.2f%%)   bwdT %8.1f ns (%+6.2f%%)\n",
                       N, barms[a].name,
                       bbf[a], 100.0 * (bbf[a] / bbf[0] - 1.0),
                       bbt[a], 100.0 * (bbt[a] / bbt[0] - 1.0));
        }
        _aligned_free(in); _aligned_free(ref); _aligned_free(out);
        _aligned_free(bref); _aligned_free(bout);
        vfft_zsplit_destroy(p);
    }
    printf(rc ? "OVERALL FAIL\n" : "OVERALL PASS\n");
    return rc;
}
