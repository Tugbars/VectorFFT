/* zturn.h — ZTURN-S PRODUCTION cascade (Phase 5 tranche 1 of
 * docs/roadmap/cascade_load_path_restructure.md; GO verdict §6.4).
 *
 * The productionized ZTURN-S route: the K=1 z-cascade's corner-turn moved
 * from the terminator's LOADS into the ingest's STORES over MKL's observed
 * SECTIONED record geometry (plane = 2N doubles, 4 sections at byte offsets
 * {0, 4N, 8N, 12N}; ingest position p emits ONE 64-B [re x4][im x4] record
 * at section bitrev2(p mod 4), granule p div 4; lanes = the radix-4
 * butterfly OUTPUT digit).
 *
 * DIFFERENCE FROM src/core/oop/zturn_proto.h (the committed Phase-2
 * prototype, gated memcmp-EXACT vs production zsplit): the four boundary
 * bodies are no longer local static transcriptions — the execute paths call
 * the GENERATED first-class kernels
 *     radix4_z_s0t_r4_{fwd,bwd}_avx2, radix8_z_stf_r4_{fwd,bwd}_avx2
 * (src/dag-fft-compiler/codelets/zil/avx2, codelet_zsplit.ml ZTURN-S kinds
 * s0t/s0tb/stf/stfb), proven bit-identical to the prototype bodies by the
 * Phase-3 GATE0 (zturn_proto_gate.c -DZTURN_GEN_KERNELS, all four cells).
 * The mids remain the byte-identical PRODUCTION msg kernels; only their
 * TABLE CONTENTS repack (lane-varying, x4 section-tiled), so the production
 * arg tuple (Ls = D_s, Gs = G_s, count = D_s) is unchanged.
 *
 * Pipeline:
 *   fwd: radix4_z_s0t_r4_fwd  (fused-turn sectioned ingest, zin -> plane)
 *        -> radix{4,8}_z_msg_fwd (in-place on the plane, per mid stage)
 *        -> radix8_z_stf_r4_fwd (4 section taps, 128 B contiguous per tap,
 *           NO load shuffles; REINT comb stores, plane -> zout).
 *   bwd: radix8_z_stf_r4_bwd (DEINT comb loads, direct record stores)
 *        -> radix{4,8}_z_msg_bwd -> radix4_z_s0t_r4_bwd (un-turn in the
 *           load network, REINT natural-z stores).
 *
 * Output order (SCRAMBLED class, fwd/bwd MATCHED):
 *   out_z[l*(N/8) + 4*k' + j] = X[l*(N/8) + 4*rho(k') + j],
 *   rho = digit reversal over the MIDDLE radices (chain[1..nf-2]).
 *   Differs from legacy zsplit by a pure per-row (N/32 x 4) transpose
 *   (Gamma): out_z[l*(N/8)+4k'+j] = out_legacy[l*(N/8)+j*(N/32)+k'].
 *
 * Chains = the SAME calibrated per-cell winners as production zsplit
 * (vfft_zsplit_default_chain — no invented chains; the plan re-search is
 * Phase 5's planner tranche). Scope fence: chain[0] == 4 REQUIRED (the
 * r0 = 4 four-section geometry baked into the _r4 kernels), last == 8,
 * D[nf-2] % 4 == 0 asserted (msg count contract). Roundtrip = N*x (no 1/N
 * in-kernel), matching zsplit. In-place (zin == zout) OK both directions:
 * fwd zout is written only by stf, which reads only the plane; bwd zin is
 * read only by stfb, the first stage.
 *
 * zsplit.h stays untouched as the legacy route, permanent fallback and
 * permanent A/B control. zturn_proto.h stays linkable beside this header
 * (distinct vfft_zturn2_* symbols) for the production gate
 * (scratchpad p5/zturn_prod_gate.c: zturn2 == proto memcmp-EXACT).
 */
#ifndef VFFT_ZTURN_H
#define VFFT_ZTURN_H

#include "zsplit.h"   /* chains, _vfft_zs_brev, msg kernel decls, allocator */

/* generator-owned ZTURN-S boundary kernels (codelets/zil/avx2, the frozen
 * 11-arg z ABI as emitted by codelet_zsplit.ml — size_t widths; the _r4
 * fname tag = the baked r0=4 section geometry) */
#define VFFT_ZT_DECL(fn) extern void fn(const double *, const double *, \
    double *, double *, const double *, const double *,                 \
    size_t, size_t, size_t, size_t, size_t);
VFFT_ZT_DECL(radix4_z_s0t_r4_fwd_avx2)  VFFT_ZT_DECL(radix4_z_s0t_r4_bwd_avx2)
VFFT_ZT_DECL(radix8_z_stf_r4_fwd_avx2)  VFFT_ZT_DECL(radix8_z_stf_r4_bwd_avx2)
VFFT_ZT_DECL(radix8_z_stf2_r4_fwd_avx2) /* 2-quad unroll-and-jam stf twin
                                         * (fwd-only, mirrors sterm2's scope;
                                         * bit-identical to stf, gate-proven) */
#undef VFFT_ZT_DECL

typedef struct {
    int N, nf;
    int chain[VFFT_ZSPLIT_MAX_NF];
    long D[VFFT_ZSPLIT_MAX_NF], G[VFFT_ZSPLIT_MAX_NF];
    double *twz[VFFT_ZSPLIT_MAX_NF];   /* mid tables, lane-varying, fwd     */
    double *twzb[VFFT_ZSPLIT_MAX_NF];  /* mid tables, bwd (sin negated)     */
    double *tzq, *tzqb;                /* terminator per-(k',lane) w^1      */
    double *plane;                     /* sectioned plane, 2N doubles, 64B  */
    int t2q;                           /* fwd terminator schedule: 0 = stf
                                        * (single-quad), 1 = stf2 (2-quad
                                        * unroll-and-jam). Bit-identical pair
                                        * — the zturn analog of zsplit's
                                        * sterm/sterm2 t2q: placement-luck-
                                        * sized delta, so MEASURED per cell
                                        * (vfft.c create race), never
                                        * reasoned. bwd keeps single-quad. */
} vfft_zturn2_plan_t;

static inline void vfft_zturn2_destroy(vfft_zturn2_plan_t *p)
{
    if (!p) return;
    for (int s = 0; s < VFFT_ZSPLIT_MAX_NF; s++) {
        VFFT_ZS_FREE(p->twz[s]);
        VFFT_ZS_FREE(p->twzb[s]);
    }
    VFFT_ZS_FREE(p->tzq);
    VFFT_ZS_FREE(p->tzqb);
    VFFT_ZS_FREE(p->plane);
    free(p);
}

/* Plan-time twiddle repack per the canonical map (zturns_consensus doc §3;
 * TRANSCRIBED from the gate-proven vfft_zturn_create, zturn_proto.h — the
 * proven table builder, not re-derived): mid tables lane-varying and x4
 * section-tiled (g2 = g mod G[s]/4 keeps the production msg arg tuple
 * legal), angle -2*pi*((l*(j+4*brev'(g'))) mod M)/M with M = N/D_s;
 * terminator per-(k',lane) w^1 at angle -2*pi*((j+4*rho(k')) mod N)/N. */
static inline vfft_zturn2_plan_t *vfft_zturn2_create(int N)
{
    int chain[VFFT_ZSPLIT_MAX_NF];
    const int nf = vfft_zsplit_default_chain(N, chain);
    if (nf < 3 || nf > VFFT_ZSPLIT_MAX_NF) return NULL;
    if (chain[0] != 4) return NULL;    /* ZTURN-S fence: 4-section geometry */
    if (chain[nf - 1] != 8) return NULL;
    long prod = 1;
    for (int s = 0; s < nf; s++) {
        if (s >= 1 && s <= nf - 2 && chain[s] != 4 && chain[s] != 8) return NULL;
        prod *= chain[s];
    }
    if (prod != N || (N / 8) % 4) return NULL;

    vfft_zturn2_plan_t *p = (vfft_zturn2_plan_t *)calloc(1, sizeof(*p));
    if (!p) return NULL;
    p->N = N; p->nf = nf;
    for (int s = 0; s < nf; s++) p->chain[s] = chain[s];
    p->D[nf - 1] = 1;
    for (int i = nf - 2; i >= 0; i--) p->D[i] = p->D[i + 1] * chain[i + 1];
    p->G[0] = 1;
    for (int i = 1; i < nf; i++) p->G[i] = p->G[i - 1] * chain[i - 1];
    if (p->D[nf - 2] % 4) goto fail;   /* asserted, not assumed (spec risk) */

    {
        const double TAU = 2.0 * M_PI;
        for (int s = 1; s <= nf - 2; s++) {
            const int R = chain[s], Rm1 = R - 1;
            const long M = (long)N / p->D[s];          /* = G_{s+1}       */
            const long Gp = p->G[s] / 4;               /* section period  */
            p->twz[s]  = (double *)VFFT_ZS_ALLOC((size_t)p->G[s] * Rm1 * 8 * 8);
            p->twzb[s] = (double *)VFFT_ZS_ALLOC((size_t)p->G[s] * Rm1 * 8 * 8);
            if (!p->twz[s] || !p->twzb[s]) goto fail;
            for (long g = 0; g < p->G[s]; g++) {
                const long g2 = g % Gp;                /* x4 section tiling */
                const long br = _vfft_zs_brev(g2, s - 1, chain + 1);
                for (int l = 1; l < R; l++)
                    for (int j = 0; j < 4; j++) {
                        const double a = -TAU
                            * (double)(((long)l * (j + 4 * br)) % M) / (double)M;
                        double *f = p->twz[s]  + ((size_t)g * Rm1 + (l - 1)) * 8;
                        double *b = p->twzb[s] + ((size_t)g * Rm1 + (l - 1)) * 8;
                        f[j] = cos(a); f[4 + j] = sin(a);
                        b[j] = cos(a); b[4 + j] = -sin(a);
                    }
            }
        }
        {
            const long K2 = (long)N / 32;
            p->tzq   = (double *)VFFT_ZS_ALLOC((size_t)K2 * 8 * 8);
            p->tzqb  = (double *)VFFT_ZS_ALLOC((size_t)K2 * 8 * 8);
            p->plane = (double *)VFFT_ZS_ALLOC((size_t)2 * N * 8);
            if (!p->tzq || !p->tzqb || !p->plane) goto fail;
            for (long k2 = 0; k2 < K2; k2++) {
                const long br = _vfft_zs_brev(k2, nf - 2, chain + 1);
                for (int j = 0; j < 4; j++) {
                    const double a = -TAU
                        * (double)((j + 4 * br) % (long)N) / (double)N;
                    p->tzq[8 * k2 + j] = cos(a);  p->tzq[8 * k2 + 4 + j] = sin(a);
                    p->tzqb[8 * k2 + j] = cos(a); p->tzqb[8 * k2 + 4 + j] = -sin(a);
                }
            }
        }
    }
    return p;
fail:
    vfft_zturn2_destroy(p);
    return NULL;
}

/* natural z in -> ZTURN-S scrambled comb out. zin == zout OK (the
 * terminator is the only writer of zout and reads only the plane).
 * Kernel arg tuples = the Phase-3 GATE0-proven calls (zturn_proto_gate.c
 * zt_exec_fwd): s0t (zin, plane, Ls = count = N/4), msg production tuple
 * (Ls = D_s, Gs = G_s, count = D_s), stf (plane, zout, OLs = count = N/8). */
static inline void vfft_zturn2_execute_fwd(const vfft_zturn2_plan_t *p,
                                           const double *zin, double *zout)
{
    radix4_z_s0t_r4_fwd_avx2(zin, 0, p->plane, 0, 0, 0,
                             (size_t)p->N / 4, 0, 0, 0, (size_t)p->N / 4);
    for (int s = 1; s <= p->nf - 2; s++) {
        /* PRODUCTION msg kernels BYTE-FOR-BYTE, production arg tuple */
        void (*f)(const double *, const double *, double *, double *,
                  const double *, const double *, unsigned long long,
                  unsigned long long, unsigned long long, unsigned long long,
                  unsigned long long) =
            (p->chain[s] == 8) ? radix8_z_msg_fwd_avx2 : radix4_z_msg_fwd_avx2;
        f(p->plane, 0, p->plane, 0, p->twz[s], 0,
          (unsigned long long)p->D[s], (unsigned long long)p->G[s],
          0, 0, (unsigned long long)p->D[s]);
    }
    (p->t2q ? radix8_z_stf2_r4_fwd_avx2 : radix8_z_stf_r4_fwd_avx2)(
        p->plane, 0, zout, 0, p->tzq, 0,
        0, 0, (size_t)p->N / 8, 0, (size_t)p->N / 8);
}

/* ZTURN-S scrambled comb in -> N * natural z out. zin == zout OK (zin is
 * read only by stfb, the first stage). */
static inline void vfft_zturn2_execute_bwd(const vfft_zturn2_plan_t *p,
                                           const double *zin, double *zout)
{
    radix8_z_stf_r4_bwd_avx2(zin, 0, p->plane, 0, p->tzqb, 0,
                             0, 0, (size_t)p->N / 8, 0, (size_t)p->N / 8);
    for (int s = p->nf - 2; s >= 1; s--) {
        void (*f)(const double *, const double *, double *, double *,
                  const double *, const double *, unsigned long long,
                  unsigned long long, unsigned long long, unsigned long long,
                  unsigned long long) =
            (p->chain[s] == 8) ? radix8_z_msg_bwd_avx2 : radix4_z_msg_bwd_avx2;
        f(p->plane, 0, p->plane, 0, p->twzb[s], 0,
          (unsigned long long)p->D[s], (unsigned long long)p->G[s],
          0, 0, (unsigned long long)p->D[s]);
    }
    radix4_z_s0t_r4_bwd_avx2(p->plane, 0, zout, 0, 0, 0,
                             (size_t)p->N / 4, 0, 0, 0, (size_t)p->N / 4);
}

#endif /* VFFT_ZTURN_H */
