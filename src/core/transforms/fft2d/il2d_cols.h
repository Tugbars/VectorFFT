/* il2d_cols.h - native IL 2D: the column-chain machinery.
 *
 * The chain enumerator, the twiddle-table builders, and the column-pass
 * kernels that execute and the create-time races both serve through. Extracted
 * from vfft.c as migration step 6b; see docs/design/refactor_migration_plan.md.
 *
 * WHAT IS HERE, AND WHAT DELIBERATELY IS NOT
 * ------------------------------------------
 * Here: the pieces that take their inputs as explicit arguments - chain
 * enumeration and resolution, table construction, and the column walkers
 * (wide, ranged, banded, natural, Bluestein).
 *
 * NOT here, and not by oversight:
 *   - the RACERS (_il2d_race_chains, _il2d_axis_race, _il2d_real_rowrace,
 *     the MT races). They carry the create-time protocol and the banking, and
 *     they belong with the wisdom write path, not with the kernels.
 *   - anything that dereferences a plan. _il2d_real_wl_cut reads h->N,
 *     h->il2d_nst and h->il2d_L, so it stays in vfft.c until step 15 lifts
 *     vfft_plan_s into vfft_internal.h. Ten lines, deliberately left behind.
 *
 * WHY THE FORWARD DECLARATIONS SURVIVED THE MOVE
 * ----------------------------------------------
 * _il2d_build_chain calls _il2d_build_tables, which is defined at the bottom of
 * this file, and the Bluestein builder calls back into the column pass. The
 * original file resolved that with forward declarations; they are carried over
 * verbatim rather than reordered, because reordering definitions changes what
 * the compiler sees and the migration's identity gate compares emitted symbol
 * bodies. Order preserved = the gate stays meaningful.
 *
 * ON THE CAP
 * ----------
 * VFFT_IL2D_MAXCAND bounds the candidate pool. When it bites, the drop is
 * LOGGED - the no-silent-caps law. A truncated pool is a biased pool, so a cap
 * that trims quietly would skew every verdict that followed it.
 */
#ifndef VFFT_TRANSFORMS_FFT2D_IL2D_COLS_H
#define VFFT_TRANSFORMS_FFT2D_IL2D_COLS_H

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "il2p.h"             /* vfft_il2p_fn, the t2c/n1c kernel resolvers */
#include "fft2d_real_il.h"    /* _il2d_row_cmul, used by the Bluestein pass */
#include "support/diag.h"     /* _vfft_warn - the chain builder refuses loudly */

/* ── native IL 2D c2c: column-chain builders (fft2d_il_c2c_design.md).
 * _il2d_build_chain: factor N1 greedy-largest over the t2c/n1c radix set;
 * stages 0..m-2 resolve t2c pairs, the last resolves the n1c pair. v1
 * STRUCTURAL default — the chain is a lay=il wisdom axis at M3 (raced,
 * never a shipped constant). Returns 0 if N1 is not expressible.
 * _il2d_build_tables: per t2c stage, the d-major record table — per digit
 * d in [0,D), per leg r in 1..R-1, [c x4][-s,+s,-s,+s] for
 * w = e^{sgn*2*pi*i*(d*r)/L} (fwd sgn=-1; bwd table CONJUGATED — the
 * kernels are shape-identical, conj is table-side, never text
 * derivation). Algebra simulator-proven: src/core/oop/il2d_proto.h. */
static int _il2d_build_tables(int N1, int nst, const int *Rs, int *Ls,
                              double **tf, double **tb);

static long _il2d_chain_prod(const int *Rs, int m)
{
    long p = 1;
    int i;
    for (i = 0; i < m; i++)
        p *= Rs[i];
    return p;
}

/* resolve a chain's kernel pairs: mids = t2c, last = n1c. */
static int _il2d_resolve(const int *Rs, int m, vfft_il2p_fn *ff,
                         vfft_il2p_fn *fb)
{
    int s;
    for (s = 0; s < m; s++)
    {
        const int last = (s == m - 1);
        ff[s] = last ? vfft_il2p_n1c_fn(Rs[s], 0) : vfft_il2p_t2c_fn(Rs[s], 0);
        fb[s] = last ? vfft_il2p_n1c_fn(Rs[s], 1) : vfft_il2p_t2c_fn(Rs[s], 1);
        if (!ff[s] || !fb[s])
            return 0;
    }
    return 1;
}

/* ordered compositions of N1 over the codelet radices, depth <= 4,
 * capped at 24 (no-silent-caps law: the cap is LOGGED when it bites). */
#define VFFT_IL2D_MAXCAND 24
static void _il2d_enum_rec(int L, int depth, int *cur, int (*out)[8],
                           int *lens, int *n, int *dropped)
{
    static const int POOL[] = { 64, 32, 16, 8, 4,
                                /* odd radices (2026-08-27): odd-N1
                                 * chains — emitted t2c/n1c kinds */
                                27, 25, 21, 19, 17, 15, 13, 11, 9, 7,
                                5, 3 };
    int p;
    if (L == 1)
    {
        if (depth == 0)
            return;
        if (*n >= VFFT_IL2D_MAXCAND)
        {
            (*dropped)++;
            return;
        }
        memcpy(out[*n], cur, 8 * sizeof(int));
        lens[*n] = depth;
        (*n)++;
        return;
    }
    if (depth >= 4)
        return;
    for (p = 0; p < (int)(sizeof POOL / sizeof POOL[0]); p++)
        if (L % POOL[p] == 0)
        {
            cur[depth] = POOL[p];
            _il2d_enum_rec(L / POOL[p], depth + 1, cur, out, lens, n,
                           dropped);
        }
}

/* the column pass, shared by execute and the create-time chain race
 * (component-pinned timing: the race times exactly this). */
/* the column pass, shared by execute and the create-time chain race.
 * fwd: stages 0..nst-1 (DIF, natural -> chain-digit-reversed comb).
 * bwd (reverse != 0): the HERMITIAN TRANSPOSE — stages nst-1..0, each a
 * PRE-twiddle conj stage (the t2c bwd kernels), CONSUMING the comb and
 * producing natural — the matched-roundtrip law (bwd eats the SAME
 * route's comb; any chain roundtrips, palindromic or not). */
/* run stages [s_lo, s_hi) over a ROW RANGE of nrows rows starting at the
 * given base pointers (nrows = N1 for the wide walk, = the band width for
 * the banded walk; every block of a stage in [s_lo,s_hi) fits the range
 * by the cut derivation L_s | wl). reverse = the Hermitian bwd order. */
static void _il2d_col_stages2(const double *src, double *dst, int nrows,
                              size_t pitch, size_t cnt, int s_lo,
                              int s_hi, const int *Rst, const int *Lst,
                              vfft_il2p_fn const *fns,
                              double *const *tabs, int reverse)
{
    int si;
    for (si = s_lo; si < s_hi; si++)
    {
        const int s = reverse ? s_hi - 1 - (si - s_lo) : si;
        const int R = Rst[s], D = Lst[s] / R;
        const double *s0 = (si == s_lo) ? src : dst;
        int b;
        for (b = 0; b < nrows / Lst[s]; b++)
        {
            const size_t off = 2 * (size_t)b * Lst[s] * pitch;
            if (D == 1)
                fns[s](s0 + off, NULL, dst + off, NULL, NULL, NULL,
                       pitch, 0, pitch, 0, cnt);
            else
                fns[s](s0 + off, NULL, dst + off, NULL, tabs[s], NULL,
                       (size_t)D * pitch, pitch, (size_t)D * pitch,
                       (size_t)D, cnt);
        }
    }
}

static void _il2d_col_stages(const double *src, double *dst, int nrows,
                             size_t rn, int s_lo, int s_hi,
                             const int *Rst, const int *Lst,
                             vfft_il2p_fn const *fns, double *const *tabs,
                             int reverse)
{
    _il2d_col_stages2(src, dst, nrows, rn, rn, s_lo, s_hi, Rst, Lst, fns,
                      tabs, reverse);
}

/* Column sub-range variant: run the whole chain over columns [k_lo,k_hi)
 * only. Columns are independent across EVERY stage (the strip axis the
 * single-thread walk already uses), so this is a pure loop restriction —
 * bit-identical to the full pass, and the unit of the MT strip arm. */
static void _il2d_col_pass_range(const double *src, double *dst, int N1,
                                 size_t rn, size_t k_lo, size_t k_hi,
                                 int nst, const int *Rst, const int *Lst,
                                 vfft_il2p_fn const *fns,
                                 double *const *tabs, int reverse)
{
    int si;
    const size_t w = k_hi - k_lo;
    if (!w)
        return;
    for (si = 0; si < nst; si++)
    {
        const int s = reverse ? nst - 1 - si : si;
        const int R = Rst[s], D = Lst[s] / R;
        const double *s0 = (si == 0) ? src : dst;
        int b;
        for (b = 0; b < N1 / Lst[s]; b++)
        {
            const size_t off = 2 * ((size_t)b * Lst[s] * rn + k_lo);
            if (D == 1)
                fns[s](s0 + off, NULL, dst + off, NULL, NULL, NULL,
                       rn, 0, rn, 0, w);
            else
                fns[s](s0 + off, NULL, dst + off, NULL, tabs[s], NULL,
                       (size_t)D * rn, rn, (size_t)D * rn, (size_t)D, w);
        }
    }
}

static void _il2d_col_pass(const double *src, double *dst, int N1,
                           size_t rn, size_t wc, int nst, const int *Rst,
                           const int *Lst, vfft_il2p_fn const *fns,
                           double *const *tabs, int reverse)
{
    size_t k0;
    int si;
    if (wc == 0 || wc > rn)
        wc = rn;
    for (k0 = 0; k0 < rn; k0 += wc)
    {
        const size_t w = (rn - k0 < wc) ? rn - k0 : wc;
        for (si = 0; si < nst; si++)
        {
            const int s = reverse ? nst - 1 - si : si;
            const int R = Rst[s], D = Lst[s] / R;
            const double *s0 = (si == 0) ? src : dst;
            int b;
            for (b = 0; b < N1 / Lst[s]; b++)
            {
                const size_t off = 2 * ((size_t)b * Lst[s] * rn + k0);
                if (D == 1)
                    fns[s](s0 + off, NULL, dst + off, NULL, NULL, NULL,
                           rn, 0, rn, 0, w);
                else
                    fns[s](s0 + off, NULL, dst + off, NULL, tabs[s], NULL,
                           (size_t)D * rn, rn, (size_t)D * rn, (size_t)D,
                           w);
            }
        }
    }
}

static int _il2d_build_chain(int N1, int *Rs, vfft_il2p_fn *ff,
                             vfft_il2p_fn *fb, int *nst)
{
    static const int POOL[] = { 64, 32, 16, 8, 4,
                                27, 25, 21, 19, 17, 15, 13, 11, 9, 7,
                                5, 3 };
    int L = N1, m = 0;
    /* env override first: VFFT_IL2D_CHAIN="64.16" (dot-separated radices,
     * product must equal N1) — the raced-axis escape hatch, env BEATS the
     * structural default (and, later, wisdom). Invalid spec: warn LOUDLY
     * and fall through to greedy — never a silent reinterpretation. */
    {
        const char *ce = getenv("VFFT_IL2D_CHAIN");
        if (ce && *ce)
        {
            const char *p = ce;
            long prod = 1;
            m = 0;
            while (*p && m < 8)
            {
                char *end;
                long r = strtol(p, &end, 10);
                if (end == p || r < 2)
                    break;
                Rs[m++] = (int)r;
                prod *= r;
                p = (*end == '.') ? end + 1 : end;
                if (*end != '.' && *end != '\0')
                    break;
                if (*end == '\0')
                {
                    p = end;
                    break;
                }
            }
            if (*p == '\0' && m > 0 && prod == N1
                && _il2d_resolve(Rs, m, ff, fb))
            {
                *nst = m;
                return 1;
            }
            _vfft_warn("VFFT_IL2D_CHAIN=\"%s\" invalid for N1=%d "
                       "(product/radix mismatch) — greedy default used",
                       ce, N1);
            m = 0;
            L = N1;
        }
    }
    while (L > 1)
    {
        int p, R = 0;
        if (m >= 8)
            return 0;
        for (p = 0; p < (int)(sizeof POOL / sizeof POOL[0]); p++)
        {
            const int r = POOL[p];
            if (L % r == 0 && (L / r == 1 || L / r >= 4))
            {
                R = r;
                break;
            }
        }
        if (!R)
            return 0; /* leftover factor (2, odd) — tier not expressible */
        Rs[m++] = R;
        L /= R;
    }
    if (m == 0 || !_il2d_resolve(Rs, m, ff, fb))
        return 0;
    *nst = m;
    return 1;
}

/* ── NATURAL n1 (M4-lite, struct comment at il2d_nat) ─────────────────
 * The perm builder: the chain's comb is the mixed-radix digit reversal;
 * the exact digit convention is settled EMPIRICALLY at create — both
 * peel orders are built and the one satisfying the block-affine
 * property is kept; neither fitting refuses the create LOUDLY. */
static int *_il2d_nat_perm(const int *Rs, int nst, int N1)
{
    /* SAME-SLOT DIF: stage s deposits the frequency digit
     * f_s = (f / prod_{u<s} R_u) mod R_s at row-weight D_s = L_s/R_s,
     * so scr row j = sum f_s * D_s, and inverting (the D_s are the
     * nested mixed-radix weights of j):
     *     perm[j] = sum_s ((j / D_s) mod R_s) * prod_{u<s} R_u.
     * The leaf (D=1) contributes (j mod R_leaf) * (N1/R_leaf) — the
     * block-affine property the OLs redirection needs, asserted below
     * together with bijectivity (a wrong derivation refuses create,
     * never serves silently). */
    int *perm = (int *)malloc((size_t)N1 * sizeof(int));
    int D[8], W[8];
    int s2, j, L = N1, w = 1;
    if (!perm || nst < 2 || nst > 8)
    {
        free(perm);
        return NULL;
    }
    for (s2 = 0; s2 < nst; s2++)
    {
        D[s2] = L / Rs[s2];
        W[s2] = w;
        w *= Rs[s2];
        L = D[s2];
    }
    for (j = 0; j < N1; j++)
    {
        int nat = 0;
        for (s2 = 0; s2 < nst; s2++)
            nat += ((j / D[s2]) % Rs[s2]) * W[s2];
        perm[j] = nat;
    }
    {
        const int Rl = Rs[nst - 1], stride = N1 / Rl;
        char *seen = (char *)calloc((size_t)N1, 1);
        int ok = (seen != NULL);
        for (j = 0; j < N1 && ok; j++)
        {
            if (perm[j] != perm[j - j % Rl] + (j % Rl) * stride)
                ok = 0;
            else if (perm[j] < 0 || perm[j] >= N1 || seen[perm[j]])
                ok = 0;
            else
                seen[perm[j]] = 1;
        }
        free(seen);
        if (!ok)
        {
            free(perm);
            return NULL;
        }
    }
    return perm;
}

/* the natural column pass: standard stages, the LEAF redirected — fwd
 * stores block b's rows at perm[b*R] with OLs = (N1/R)*rn; bwd (the
 * Hermitian transpose, leaf first) GATHERS its legs from the natural
 * positions via Ls, the exact mirror. Unbanded, full plane. */
static void _il2d_col_pass_nat(const double *src, double *dst, int N1,
                               size_t rn, int nst, const int *Rst,
                               const int *Lst, vfft_il2p_fn const *fns,
                               double *const *tabs, int reverse,
                               const int *perm, double *scr)
{
    /* fwd: stages 0..nst-2 run src -> scr (stage 0 is the OOP move,
     * the rest in place on scr); the LEAF reads scr and SCATTERS to
     * dst (out-base perm[b*R], OLs = (N1/R)*rn). bwd mirrors: the leaf
     * GATHERS from the natural src into scr's comb, mids run in place,
     * and stage 0 writes scr -> dst (the kernels take separate bases —
     * no extra copy pass anywhere). The scatter/gather NEVER shares a
     * plane with the stages still reading it. */
    const int Rl = Rst[nst - 1];
    const size_t nstride = (size_t)(N1 / Rl) * rn;
    int s, b;
    if (!reverse)
    {
        for (s = 0; s < nst - 1; s++)
        {
            const int R = Rst[s], D = Lst[s] / R;
            const double *s0 = (s == 0) ? src : scr;
            for (b = 0; b < N1 / Lst[s]; b++)
            {
                const size_t off = 2 * (size_t)b * Lst[s] * rn;
                fns[s](s0 + off, NULL, scr + off, NULL, tabs[s], NULL,
                       (size_t)D * rn, rn, (size_t)D * rn, (size_t)D,
                       rn);
            }
        }
        for (b = 0; b < N1 / Rl; b++)
            fns[nst - 1](scr + 2 * (size_t)b * Rl * rn, NULL,
                         dst + 2 * (size_t)perm[b * Rl] * rn, NULL,
                         NULL, NULL, rn, 0, nstride, 0, rn);
    }
    else
    {
        for (b = 0; b < N1 / Rl; b++)
            fns[nst - 1](src + 2 * (size_t)perm[b * Rl] * rn, NULL,
                         scr + 2 * (size_t)b * Rl * rn, NULL, NULL,
                         NULL, nstride, 0, rn, 0, rn);
        for (s = nst - 2; s >= 0; s--)
        {
            const int R = Rst[s], D = Lst[s] / R;
            double *out = (s == 0) ? dst : scr;
            for (b = 0; b < N1 / Lst[s]; b++)
            {
                const size_t off = 2 * (size_t)b * Lst[s] * rn;
                fns[s](scr + off, NULL, out + off, NULL, tabs[s], NULL,
                       (size_t)D * rn, rn, (size_t)D * rn, (size_t)D,
                       rn);
            }
        }
    }
}

/* ── the COLUMN-AXIS BLUESTEIN, extracted (2026-08-27) so THREE users
 * share one implementation: the c2c no-chain path, the chain-vs-blu
 * RACE (the odd chains are now emitted, so both arms exist for odd
 * N1), and the REAL tier (rn = hp1 there; the pipeline is C-linear
 * over any count). ─────────────────────────────────────────────── */

/* build the M-chain + tables + chirps + comb-order kernels + scratch
 * into the CALLER's arrays. Returns M (>0) or 0. rn = the plane's row
 * width in complex (N2 for c2c, hp1 for real). */
static int _il2d_build_chain(int N1, int *Rs, vfft_il2p_fn *ff,
                             vfft_il2p_fn *fb, int *nst);
static void _il2d_col_pass(const double *src, double *dst, int N1,
                           size_t rn, size_t wc, int nst, const int *Rst,
                           const int *Lst, vfft_il2p_fn const *fns,
                           double *const *tabs, int reverse);
static int _il2d_build_tables(int N1, int nst, const int *Rs, int *Ls,
                              double **tf, double **tb);
/* BLUESTEIN INNER CHAIN HOOK (2026-09-02): the 2D create installs a
 * provider that fills the length-M column chain from wisdom (the (M, N2)
 * chain row, raced and banked there on a miss); NULL, or a provider that
 * declines, leaves the greedy chain in charge. Set once at the 2D create's
 * entry — planning side, one create at a time. */
typedef int (*_il2d_blu_chain_fn)(int M, int *Rs, int *nst);
static _il2d_blu_chain_fn _il2d_blu_chain_hook = 0;

static int _il2d_blu_build(int N1, size_t rn, int *Rs, int *Ls,
                           vfft_il2p_fn *ff, vfft_il2p_fn *fb,
                           double **tf, double **tb, int *nst,
                           double **chf, double **chb, double **kf,
                           double **kb, double **scr)
{
    int M = 16, s2, ok = 0, served = 0;
    double *za = NULL, *zb2 = NULL;
    while (M < 2 * N1 - 1)
        M <<= 1;
    *chf = *chb = *kf = *kb = *scr = NULL;
    if (_il2d_blu_chain_hook && _il2d_blu_chain_hook(M, Rs, nst))
        served = _il2d_resolve(Rs, *nst, ff, fb);   /* the validator is the law */
    if (!served && !_il2d_build_chain(M, Rs, ff, fb, nst))
        return 0;
    if (_il2d_build_tables(M, *nst, Rs, Ls, tf, tb))
        return 0;
    *chf = (double *)malloc(2 * (size_t)N1 * sizeof(double));
    *chb = (double *)malloc(2 * (size_t)N1 * sizeof(double));
    *kf = (double *)malloc(2 * (size_t)M * sizeof(double));
    *kb = (double *)malloc(2 * (size_t)M * sizeof(double));
    *scr = (double *)malloc(2 * (size_t)M * rn * sizeof(double));
    za = (double *)calloc(2 * (size_t)M, sizeof(double));
    zb2 = (double *)malloc(2 * (size_t)M * sizeof(double));
    if (*chf && *chb && *kf && *kb && *scr && za && zb2)
    {
        int r, d2;
        for (r = 0; r < N1; r++)
        {
            const long long m2 = ((long long)r * r) % (2LL * N1);
            const double a = -VFFT_IL2P_PI * (double)m2 / (double)N1;
            (*chf)[2 * r] = cos(a);
            (*chf)[2 * r + 1] = sin(a);
            (*chb)[2 * r] = cos(a);
            (*chb)[2 * r + 1] = -sin(a);
        }
        for (d2 = 0; d2 < 2; d2++)
        {
            const double *ch = d2 ? *chb : *chf;
            double *kern = d2 ? *kb : *kf;
            const double inv = 1.0 / (double)M;
            memset(za, 0, 2 * (size_t)M * sizeof(double));
            za[0] = ch[0];
            za[1] = -ch[1];
            for (r = 1; r < N1; r++)
            {
                za[2 * r] = ch[2 * r];
                za[2 * r + 1] = -ch[2 * r + 1];
                za[2 * (M - r)] = ch[2 * r];
                za[2 * (M - r) + 1] = -ch[2 * r + 1];
            }
            _il2d_col_pass(za, zb2, M, 1, 1, *nst, Rs, Ls, ff, tf, 0);
            for (s2 = 0; s2 < 2 * M; s2++)
                kern[s2] = zb2[s2] * inv;
        }
        ok = 1;
    }
    free(za);
    free(zb2);
    if (!ok)
    {
        free(*chf); free(*chb); free(*kf); free(*kb); free(*scr);
        *chf = *chb = *kf = *kb = *scr = NULL;
        for (s2 = 0; s2 < *nst; s2++)
        {
            free(tf[s2]); tf[s2] = NULL;
            free(tb[s2]); tb[s2] = NULL;
        }
        return 0;
    }
    return M;
}

/* the blu column pipeline over an N1 x rn plane (explicit args — the
 * execute branches and the race both serve through THIS). reverse = the
 * inverse transform (conjugated chirp/kernel, the caller passes them).
 * src/dst may alias. */
/* the Bluestein pipeline over a column WINDOW [c0, c1) of an rn-wide plane:
 * every step is column-local (row-wise chirp/kernel multiplies touch each
 * column independently; the M-chain column passes take a column range), so
 * a window is an independent unit and windows share `scr` disjointly —
 * this is what the threaded column walk partitions (2026-09-02). */
static void _il2d_blu_cols_range(const double *src, double *dst, int N1,
                                 size_t rn, size_t c0, size_t c1, int M,
                                 int nst, const int *Rs, const int *Ls,
                                 vfft_il2p_fn const *ff,
                                 vfft_il2p_fn const *fb,
                                 double *const *tf, double *const *tb,
                                 const double *ch, const double *kn,
                                 double *scr)
{
    const size_t wc = c1 - c0;
    long r2;
    if (c1 <= c0) return;
    for (r2 = N1; r2 < M; r2++)              /* the zero pad, this window */
        memset(scr + 2 * ((size_t)r2 * rn + c0), 0, 2 * wc * sizeof(double));
    for (r2 = 0; r2 < N1; r2++)
        _il2d_row_cmul(scr + 2 * ((size_t)r2 * rn + c0),
                       src + 2 * ((size_t)r2 * rn + c0), ch[2 * r2],
                       ch[2 * r2 + 1], wc);
    _il2d_col_pass_range(scr, scr, M, rn, c0, c1, nst, Rs, Ls, ff, tf, 0);
    for (r2 = 0; r2 < M; r2++)
        _il2d_row_cmul(scr + 2 * ((size_t)r2 * rn + c0),
                       scr + 2 * ((size_t)r2 * rn + c0), kn[2 * r2],
                       kn[2 * r2 + 1], wc);
    _il2d_col_pass_range(scr, scr, M, rn, c0, c1, nst, Rs, Ls, fb, tb, 1);
    for (r2 = 0; r2 < N1; r2++)
        _il2d_row_cmul(dst + 2 * ((size_t)r2 * rn + c0),
                       scr + 2 * ((size_t)r2 * rn + c0), ch[2 * r2],
                       ch[2 * r2 + 1], wc);
}

static void _il2d_blu_cols(const double *src, double *dst, int N1,
                           size_t rn, int M, int nst, const int *Rs,
                           const int *Ls, vfft_il2p_fn const *ff,
                           vfft_il2p_fn const *fb, double *const *tf,
                           double *const *tb, const double *ch,
                           const double *kn, double *scr)
{
    _il2d_blu_cols_range(src, dst, N1, rn, 0, rn, M, nst, Rs, Ls, ff, fb,
                         tf, tb, ch, kn, scr);
}

static int _il2d_build_tables(int N1, int nst, const int *Rs, int *Ls,
                              double **tf, double **tb)
{
    const double pi = 3.14159265358979323846;
    int s, L = N1;
    for (s = 0; s < nst; s++)
    {
        const int R = Rs[s], D = L / R;
        Ls[s] = L;
        tf[s] = NULL;
        tb[s] = NULL;
        if (D > 1) /* the n1c leaf carries no table */
        {
            const size_t nrec = (size_t)D * (R - 1);
            double *f = (double *)malloc(nrec * 8 * sizeof(double));
            double *bt = (double *)malloc(nrec * 8 * sizeof(double));
            int d, r, lane;
            if (!f || !bt)
            {
                free(f);
                free(bt);
                return -1;
            }
            for (d = 0; d < D; d++)
                for (r = 1; r < R; r++)
                {
                    const double a =
                        -2.0 * pi * (double)((size_t)d * r % (size_t)L)
                        / (double)L;
                    const double c = cos(a), si = sin(a);
                    double *rf = f + ((size_t)d * (R - 1) + (r - 1)) * 8;
                    double *rb = bt + ((size_t)d * (R - 1) + (r - 1)) * 8;
                    for (lane = 0; lane < 4; lane++)
                    {
                        rf[lane] = c;
                        rb[lane] = c;
                        rf[4 + lane] = (lane & 1) ? si : -si;
                        rb[4 + lane] = (lane & 1) ? -si : si; /* conj */
                    }
                }
            tf[s] = f;
            tb[s] = bt;
        }
        L /= R;
    }
    return 0;
}

#endif /* VFFT_TRANSFORMS_FFT2D_IL2D_COLS_H */
