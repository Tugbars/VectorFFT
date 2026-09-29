/* zrp.h — the interleaved REAL PAIR (2026-09-29, docs/roadmap/
 * il_real_engine_research.md, method 1): a real transform of even N = R1 * R2
 * in two stages, no fold pass — the untangle rides in the top stage (form A)
 * or never happens (form B). Its c2r is the mirror.
 *
 * FORM A (leaf = n1t): x[N] read as z[N/2] = x[2m] + i x[2m+1] is R1/2 packed
 *   columns of R2 complex legs; the stock n1t(R2) leaf (count = R1/2, Ls =
 *   R1/2, OLs = R2) leaves packed column c's spectrum at z[c*R2 + p]; t2h(R1)
 *   untangles, twiddles, runs the radix-R1 butterfly and mirror-stores IN
 *   PLACE (Ls = OLs = R2, count = R2/2) -> the CCE half spectrum X[0..N/2].
 *   bwd: t2h bwd(R1) (Ls = R2, OLs = R1/2) -> the packed plane corner-turned,
 *   then the stock n1 bwd(R2) in place per column (count = Ls = OLs = R1/2)
 *   -> N times x in natural order.
 * FORM B (leaf = r2z): r2z(R2) reads R1 real columns four per vector (Ls =
 *   R1, count = R1) and stores each column's packed half spectrum at
 *   zout[2*(c*OLs + p)], OLs = R2 (half of every R2-wide row: the layout
 *   that makes the top in place); t2m(R1) combines the R1 half spectra
 *   (Ls = OLs = R2, count = R2/2) -> X. No untangle anywhere: the ~N/2
 *   complex multiplies the packing trick pays are gone.
 *   bwd: t2m bwd(R1) (Ls = R2, OLs = R1) -> the real legs' half spectra
 *   corner-turned (column p, leg j) at zout[2*(p*R1 + j)], then r2z bwd(R2)
 *   (Ls = R1, OLs = R1, count = R1) -> N times x.
 *
 * PLANES. fwd out of place: the leaf writes the caller's CCE plane (form A:
 *   N doubles; form B: 2N doubles of address, half of them written — the
 *   CCE plane is N+2 doubles, so form B's leaf goes to the plan's scratch
 *   and the top scratch -> X); in place: leaf -> scratch, top scratch -> X.
 *   bwd: the top's turned plane needs a plane of its own (its writes are
 *   not its reads): form A out of place uses the real output plane (the
 *   leaf then runs in place there); form B and every in-place call go
 *   through the scratch. scratch = 2N doubles, allocated when a placement
 *   needs it.
 * TABLES: tw (fwd) and twb (bwd), (R2/4 + 1) record sets of (R1-1) records
 *   of 8 doubles [c c c' c'][-s +s -s' +s'] (the t2 mid's shape): set s is
 *   the columns (2s+1, 2s+2) at the angles -2 pi j p / N, the last set the
 *   columns 0 (angle 0) and R2/2. Form A's fwd records are halved (the
 *   untangle's 1/2; leg 0's is the kernel's own constant); form B's and
 *   every bwd record are unhalved, the bwd ones conjugated. */
#ifndef VFFT_ZRP_H
#define VFFT_ZRP_H

#include <stdlib.h>
#include <string.h>

#include "il2p.h" /* vfft_il2p_fn, the leaf resolvers, tw_exact, zalloc, the registry */

/* form 0 = n1t + t2h (the packed leaf), 1 = r2z + t2m (the real leaf) */
#define VFFT_ZRP_FORM_A 0
#define VFFT_ZRP_FORM_B 1

static inline vfft_il2p_fn vfft_zrp_top_fn(int R, int form, int bwd)
{
    if (form == VFFT_ZRP_FORM_A) {
        switch (R) {
#ifdef VFFT_IL_T2H_PAIR_RADICES
#define C(R) case R: return bwd ? VFFT_IL_SYM(radix##R##_z_t2h_bwd) : VFFT_IL_SYM(radix##R##_z_t2h_fwd);
        VFFT_IL_T2H_PAIR_RADICES(C)
#undef C
#endif
        default: return 0;
        }
    }
    switch (R) {
#ifdef VFFT_IL_T2M_PAIR_RADICES
#define C(R) case R: return bwd ? VFFT_IL_SYM(radix##R##_z_t2m_bwd) : VFFT_IL_SYM(radix##R##_z_t2m_fwd);
    VFFT_IL_T2M_PAIR_RADICES(C)
#undef C
#endif
    default: return 0;
    }
}

static inline vfft_il2p_fn vfft_zrp_leaf_fn(int R, int form, int bwd)
{
    if (form == VFFT_ZRP_FORM_A)
        return bwd ? vfft_il2p_n1_bwd_fn(R) : vfft_il2p_leaf_fn(R, 0);
    switch (R) {
#ifdef VFFT_IL_R2Z_PAIR_RADICES
#define C(R) case R: return bwd ? VFFT_IL_SYM(radix##R##_z_r2z_bwd) : VFFT_IL_SYM(radix##R##_z_r2z_fwd);
    VFFT_IL_R2Z_PAIR_RADICES(C)
#undef C
#endif
    default: return 0;
    }
}

typedef struct vfft_zrp_plan_s {
    int N, R1, R2, form;
    vfft_il2p_fn leaf_f, top_f, top_b, leaf_b;
    double *tw, *twb;  /* the record sets, fwd and bwd */
    double *scratch;   /* 2N doubles, NULL when no placement needs it */
} vfft_zrp_plan_t;

static inline void vfft_zrp_destroy(vfft_zrp_plan_t *p)
{
    if (!p) return;
    vfft_aligned_free(p->tw);
    vfft_aligned_free(p->twb);
    vfft_aligned_free(p->scratch);
    free(p);
}

/* 1 when the pair builds in this form: both radices even, R2 >= 4, all four
 * kernels resolve. The pair and the form are PLAN INPUT (the door races). */
static inline int vfft_zrp_pair_ok(int N, int R1, int R2, int form)
{
    if (N <= 0 || R1 < 4 || R2 < 4 || (R1 & 1) || (R2 & 1) || (long)R1 * (long)R2 != (long)N)
        return 0;
    return vfft_zrp_leaf_fn(R2, form, 0) && vfft_zrp_leaf_fn(R2, form, 1) &&
           vfft_zrp_top_fn(R1, form, 0) && vfft_zrp_top_fn(R1, form, 1);
}

/* inplace = 1 allocates the scratch the in-place placements need; form B and
 * the c2r direction need it out of place too, so it is allocated whenever
 * the plan can be asked for it (form B always; form A in place only — its
 * c2r out of place turns into the real output plane). */
static inline vfft_zrp_plan_t *vfft_zrp_create(int N, int R1, int R2, int form, int inplace)
{
    if (!vfft_zrp_pair_ok(N, R1, R2, form)) return 0;
    vfft_zrp_plan_t *p = (vfft_zrp_plan_t *)calloc(1, sizeof *p);
    if (!p) return 0;
    p->N = N; p->R1 = R1; p->R2 = R2; p->form = form;
    p->leaf_f = vfft_zrp_leaf_fn(R2, form, 0);
    p->leaf_b = vfft_zrp_leaf_fn(R2, form, 1);
    p->top_f = vfft_zrp_top_fn(R1, form, 0);
    p->top_b = vfft_zrp_top_fn(R1, form, 1);
    const int half = R2 / 2, nsets = half / 2 + 1;
    const size_t ntw = (size_t)nsets * (size_t)(R1 - 1) * 8u;
    const int need_scratch = inplace || form == VFFT_ZRP_FORM_B;
    p->tw = (double *)vfft_aligned_alloc(ntw * sizeof(double));
    p->twb = (double *)vfft_aligned_alloc(ntw * sizeof(double));
    p->scratch = need_scratch ? (double *)vfft_aligned_alloc(2u * (size_t)N * sizeof(double)) : 0;
    if (!p->tw || !p->twb || (need_scratch && !p->scratch)) { vfft_zrp_destroy(p); return 0; }
    const double fs = form == VFFT_ZRP_FORM_A ? 0.5 : 1.0;   /* form A's fwd records carry the untangle's 1/2 */
    for (int s = 0; s < nsets; s++)
        for (int j = 1; j < R1; j++) {
            double *rf = p->tw + ((size_t)s * (size_t)(R1 - 1) + (size_t)(j - 1)) * 8u;
            double *rb = p->twb + ((size_t)s * (size_t)(R1 - 1) + (size_t)(j - 1)) * 8u;
            for (int l = 0; l < 2; l++) {
                /* lane l's column: a regular set -> 2s+1+l (the tail set's
                 * lane 1 = column R2/2, valid values never read); the last
                 * set -> 0 and R2/2 */
                const int col = (s == nsets - 1) ? (l ? half : 0) : (2 * s + 1 + l);
                double c, sn;
                vfft_cs2pi_exact((long long)j * (long long)col, (long long)N, &c, &sn);
                const double sa = -sn; /* sin(-2 pi j col / N) */
                rf[2 * l] = fs * c;       rf[2 * l + 1] = fs * c;
                rf[4 + 2 * l] = -fs * sa; rf[5 + 2 * l] = fs * sa;
                rb[2 * l] = c;            rb[2 * l + 1] = c;
                rb[4 + 2 * l] = sa;       rb[5 + 2 * l] = -sa;
            }
        }
    return p;
}

/* x[N] reals -> X[0..N/2] CCE (N+2 doubles). x == X is the in-place call. */
static inline void vfft_zrp_execute_fwd(const vfft_zrp_plan_t *p, const double *x, double *X)
{
    const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
    if (p->form == VFFT_ZRP_FORM_A) {
        if (x == (const double *)X) {
            p->leaf_f(x, 0, p->scratch, 0, 0, 0, R1 / 2, 0, R2, 0, R1 / 2);
            p->top_f(p->scratch, 0, X, 0, p->tw, 0, R2, 0, R2, 0, R2 / 2);
            return;
        }
        p->leaf_f(x, 0, X, 0, 0, 0, R1 / 2, 0, R2, 0, R1 / 2);
        p->top_f(X, 0, X, 0, p->tw, 0, R2, 0, R2, 0, R2 / 2);
        return;
    }
    /* form B: the leaf plane spans 2N doubles (half of every row), so it
     * lives in the scratch; the top reads it and writes X */
    p->leaf_f(x, 0, p->scratch, 0, 0, 0, R1, 0, R2, 0, R1);
    p->top_f(p->scratch, 0, X, 0, p->tw, 0, R2, 0, R2, 0, R2 / 2);
}

/* X[0..N/2] CCE -> x[N] = N * (the inverse). X == x is the in-place call. */
static inline void vfft_zrp_execute_bwd(const vfft_zrp_plan_t *p, const double *X, double *x)
{
    const size_t R1 = (size_t)p->R1, R2 = (size_t)p->R2;
    if (p->form == VFFT_ZRP_FORM_A) {
        if (X == (const double *)x) {
            p->top_b(X, 0, p->scratch, 0, p->twb, 0, R2, 0, R1 / 2, 0, R2 / 2);
            p->leaf_b(p->scratch, 0, x, 0, 0, 0, R1 / 2, 0, R1 / 2, 0, R1 / 2);
            return;
        }
        p->top_b(X, 0, x, 0, p->twb, 0, R2, 0, R1 / 2, 0, R2 / 2);
        p->leaf_b(x, 0, x, 0, 0, 0, R1 / 2, 0, R1 / 2, 0, R1 / 2);
        return;
    }
    p->top_b(X, 0, p->scratch, 0, p->twb, 0, R2, 0, R1, 0, R2 / 2);
    p->leaf_b(p->scratch, 0, x, 0, 0, 0, R1, 0, R1, 0, R1);
}

#endif /* VFFT_ZRP_H */
