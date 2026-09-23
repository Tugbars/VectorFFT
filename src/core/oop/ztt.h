/* ztt.h — ZTURN-T: the RUN-CONTIGUOUS DIT engine for K=1 interleaved c2c,
 * 16 <= N <= 262144, route VFFT_K1_IL_ZTT (docs/design/zturn_t_ship_plan.md;
 * the probe's contract: docs/research/sub2048_mkl_method/campaign_state/
 * probes/ZT/CONTRACT.md).
 *
 *   N = R[0] * ... * R[nf-1], R[0] and R[nf-1] in {4, 8}, a mid in {4, 8} or
 *   3/5/7/9/15 (the 2^a*odd band, docs/design/ztt_odd_design.md, 2026-09-14),
 *   (N / R[0]) % 4 == 0. A pow2 cell binds its FUSED codelet; a 2^a*odd cell
 *   is STAGED (one kernel call per stage and block, vfft_ztt_odd_band).
 *   ingest  t0tp : natural packed z legs at stride N/R0 -> the PLANE, one
 *                  run of R0 per column at run rb[c] (64-B [re x4][im x4]
 *                  blocks); twiddle-free.
 *   mids    tmg  : combine R adjacent runs of length L into ONE run of R*L
 *                  IN PLACE with the column-varying pre-twiddle w_{RL}^{r*b};
 *                  ONE stream per stage serves every group.
 *   last    tlf  : the same combine, REINT packed store: NATURAL output, no
 *                  reordering pass anywhere.
 *   inverse      : the same pipeline with conjugate roots (the bwd kinds +
 *                  the s-negated streams); unnormalised, roundtrip = N * x.
 *
 * IN PLACE (2026-09-09, zturn_t_ship_plan.md 9): the `plane` drivers end in
 * tlfi, the in-place terminator — tlf with its output streams prefetched. The
 * caller's buffer goes cold under the plane, and a store into it waited on
 * its line fill (+17..25% over out of place); the prefetch issues those
 * fills ahead. Same arithmetic: in place stays bitwise the dest result.
 *
 * EXECUTION is one indirect call: the plan binds ONE FUSED DRIVER per
 * direction (generated/ztt_drivers_avx2.c — the three kind bodies inlined
 * with literal trip counts and the twiddle cursor carried in a register, zero
 * calls inside; the form that measured 9-12% over per-stage calls at N=128,
 * cascade_stage_fusion.md). Two buffer modes: `dest` runs the whole pipeline
 * in the destination (out of place), `plane` in the plan's scratch (in place,
 * or zin == zout at the call). Bound at create by placement; the execute
 * keeps one predictable compare so an aliased call on an out-of-place plan
 * is still correct.
 *
 * TILING (docs/design/zturn_t_2048plus_plan.md step 2): the mid stages whose
 * run length R*L <= tile run PER TILE — a contiguous plane span of `tile`
 * complexes holds WHOLE groups of every such stage, so the tile stays L1-hot
 * across those stages — the rest sweep the plane; the ingest is untiled (a
 * per-tile ingest would re-read every input line once per tile). The tile
 * changes group ORDER only: every width is bitwise the untiled result. It is
 * a RACED plan parameter (the planner's ladder, banked as il_tw=), never a
 * rule; vfft_ztt_tile_legal is the one law for it.
 *
 * CREATE builds (almost) no trig: up to the table's octave (RL <= 16384)
 * every stream is expanded from the baked quarter-wave (ztt_qw16384.h) by
 * index shift + reflection + sign flip. Above it (S4, zcascade_sunset_plan.md,
 * 2026-09-09) a stage's record is the TWO-LEVEL product: the angle
 * 2*pi*pw/RL splits as pw = a*2^u + b (u = log2 RL - 14, b < 2^u <= 16 at
 * 262144), w = table(a) * fine(b) with fine(b) = exp(-2*pi*i*b/RL) — one
 * 2^u-entry cos/sin table per stage at create, one complex product per
 * record, ~1 ulp. The ceiling is the cascade's, 262144. The validator is the law: an illegal
 * chain, a size out of range, or a cell without a registry driver returns
 * NULL loudly — no fallback, no default chain (the planner is the only
 * source of a chain). */
#ifndef VFFT_ZTT_H
#define VFFT_ZTT_H

#include "tw_exact.h"   /* once-rounded cos/sin(2*pi*p/n) for the create-time tables */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>              /* the fine table above the octave (S4, 2026-09-09)  */
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* FUSED CODELETS (owner's ruling 2026-09-14): the drivers this plan binds are
 * whole-transform functions, one per pow2 cell (N, chain, direction, buffer
 * mode), generated into generator/generated/fused_codelets/ with the stage
 * kernels INLINED and literal trip counts — the pow2 ZTURN-T solution's
 * executable form, and only that. The composable STAGE KERNELS (t0tp/tmg/tlf/
 * tlfi and the plain t0d/tmgd/tld/tldb in codelets/zil/avx2/boundary_split)
 * are the product every other solution is built from; a fused codelet cannot
 * be recombined. README.md beside the fused files. */
#include "ztt_registry_avx2.h"   /* the cells and their fused codelets (generated) */
#include "ztt_qw16384.h"         /* the baked quarter-wave (generated)            */
/* THE STAGE KERNELS, for the STAGED executor (docs/design/ztt_odd_design.md,
 * 2026-09-14): a cell with no fused codelet — every 2^a*odd cell in the odd
 * band, or a pow2 cell under the gate's force — runs its stage table, one
 * kernel call per (stage, block), each kind's exported function in the frozen
 * 11-arg z ABI with the group loop inside it. The radix lists come from the
 * corpus through il_registry_avx2.h, so a chain whose radix has no kernel is
 * refused at create, never at run time. The odd radices 3/5/7/9/15 exist for
 * the MIDS only (tmg, tmgb, tmgd): the ingest and the terminators are lane
 * lattices and stay 4/8 — the odd radix is always a mid. */
#include "il_registry_avx2.h"
typedef void (*vfft_ztt_kfn)(const double *, const double *, double *, double *,
                             const double *, const double *,
                             size_t, size_t, size_t, size_t, size_t);
#define _ZTT_KFN2(NAME, KIND, MF, MB) static inline vfft_ztt_kfn NAME(int R, int bwd) {     if (bwd) { switch (R) { MB(_ZTT_KB_##KIND) default: return 0; } }     switch (R) { MF(_ZTT_KF_##KIND) default: return 0; } }
#define _ZTT_KF_t0tp(R) case R: return radix##R##_z_t0tp_fwd_avx2;
#define _ZTT_KB_t0tp(R) case R: return radix##R##_z_t0tp_bwd_avx2;
#define _ZTT_KF_tmg(R)  case R: return radix##R##_z_tmg_fwd_avx2;
#define _ZTT_KB_tmg(R)  case R: return radix##R##_z_tmg_bwd_avx2;
#define _ZTT_KF_tlf(R)  case R: return radix##R##_z_tlf_fwd_avx2;
#define _ZTT_KB_tlf(R)  case R: return radix##R##_z_tlf_bwd_avx2;
#define _ZTT_KF_tlfi(R) case R: return radix##R##_z_tlfi_fwd_avx2;
#define _ZTT_KB_tlfi(R) case R: return radix##R##_z_tlfi_bwd_avx2;
#define _ZTT_KF_tld(R)  case R: return radix##R##_z_tld_fwd_avx2;
#define _ZTT_KB_tld(R)  case R: return radix##R##_z_tld_bwd_avx2;
#define _ZTT_KF_t0d(R)  case R: return radix##R##_z_t0d_fwd_avx2;
#define _ZTT_KB_t0d(R)
#define _ZTT_KF_tmgd(R) case R: return radix##R##_z_tmgd_fwd_avx2;
#define _ZTT_KB_tmgd(R)
#define _ZTT_NONE(X)
_ZTT_KFN2(_ztt_kfn_t0tp, t0tp, VFFT_IL_T0TP_FWD_RADICES, VFFT_IL_T0TP_BWD_RADICES)
_ZTT_KFN2(_ztt_kfn_tmg,  tmg,  VFFT_IL_TMG_FWD_RADICES,  VFFT_IL_TMG_BWD_RADICES)
_ZTT_KFN2(_ztt_kfn_tlf,  tlf,  VFFT_IL_TLF_FWD_RADICES,  VFFT_IL_TLF_BWD_RADICES)
_ZTT_KFN2(_ztt_kfn_tlfi, tlfi, VFFT_IL_TLFI_FWD_RADICES, VFFT_IL_TLFI_BWD_RADICES)
_ZTT_KFN2(_ztt_kfn_tld,  tld,  VFFT_IL_TLD_FWD_RADICES,  VFFT_IL_TLD_BWD_RADICES)
_ZTT_KFN2(_ztt_kfn_t0d,  t0d,  VFFT_IL_T0D_FWD_RADICES,  _ZTT_NONE)
_ZTT_KFN2(_ztt_kfn_tmgd, tmgd, VFFT_IL_TMGD_FWD_RADICES, _ZTT_NONE)

#define VFFT_ZTT_MAX_NF 7        /* == the registry's chain[7]; VFFT_ZSPLIT_MAX_NF */
#define VFFT_ZTT_MAX_N 262144    /* the cascade's ceiling; above the octave the
                                  * streams are the TWO-LEVEL product (below)     */
#define VFFT_ZTT_QW_M 16384
#define VFFT_ZTT_QW_LGM 14       /* log2 M */
#define VFFT_ZTT_QW_Q 4096       /* M / 4 */
#define VFFT_ZTT_QW_LGQ 12

#if defined(_WIN32)
#include <malloc.h>
#define VFFT_ZTT_ALLOC(sz) _aligned_malloc((sz), 64)
#define VFFT_ZTT_FREE(p) _aligned_free(p)
#else
#define VFFT_ZTT_ALLOC(sz) aligned_alloc(64, ((((sz)) + 63u) / 64u) * 64u)
#define VFFT_ZTT_FREE(p) free(p)
#endif

typedef struct
{
    int N, nf;
    int chain[VFFT_ZTT_MAX_NF];
    long ncol;                        /* N / chain[0]: ingest columns = rb entries   */
    long L[VFFT_ZTT_MAX_NF];          /* run length BEFORE stage s (s >= 1)          */
    long Gs[VFFT_ZTT_MAX_NF];         /* group count of stage s                      */
    double *tw, *twb;                 /* ONE contiguous stream each, stage order:
                                       * fwd, and bwd (the s half negated)          */
    size_t twdoubles;
    size_t *rb;                       /* the run-base table, ncol entries            */
    double *plane;                    /* 2N doubles + 4 KB of slack, 64-B: the `plane`
                                       * mode scratch; each in-place call starts it 2 KB
                                       * (mod 4 KB) from the caller's buffer (_ztt_plane_for) */
    const vfft_ztt_cell_t *cell;      /* the registry row: the four drivers          */
    vfft_ztt_fn fwd, bwd;             /* BOUND by placement (vfft_ztt_bind)          */
    int inplace;
    size_t tile;                      /* TILE WIDTH in complexes, 0 = untiled (raced,
                                       * il_tw=; vfft_ztt_set_tile)                  */
    /* THE PLAIN SCHEDULE = the scrambled order class (2026-09-14,
     * docs/design/ztt_scrambled_design.md). scr = 1 executes the cell's
     * fwd_scr / bwd_scr fused codelets (ABI zin, zout, tw, tile: no plane,
     * no run-base table, zin == zout is the same function). Then tw / twb
     * hold the PLAIN layout — stage s < nf-1 carries 2*(R_s-1)*Len_{s+1}
     * doubles of w_{Len_s}^(p*b), the last stage none — rb and plane stay
     * NULL, len[] is the block ladder, and perm[k] is the output position of
     * frequency k: built once for the gates and introspection, never read at
     * run time. A plan is ONE order class for its whole life: the two never
     * mix (design_contracts.md 8b). */
    int scr;
    long len[VFFT_ZTT_MAX_NF + 1];    /* plain: Len_s = prod_{u >= s} R_u, Len_nf = 1 */
    size_t *perm;                     /* plain: N entries, frequency -> position      */
    /* THE STAGED EXECUTOR (ztt_odd_design.md, 2026-09-14): staged = 1 when the
     * cell has no fused codelet (cell == NULL) — the stage table below runs
     * the fused driver's exact loop nest with run-time bounds, one call per
     * (stage, block). st_fwd/st_bwd[s] = stage s's kernel (natural: t0tp,
     * tmg.., tlf; plain fwd: t0d, tmgd.., tld; plain bwd: tlfb, tmgb.., tldb —
     * indexed by the FORWARD stage they invert); tl_*_plane = the natural
     * in-place terminator (tlfi); twoff[s] = stage s's offset into tw/twb. */
    int staged;
    vfft_ztt_kfn st_fwd[VFFT_ZTT_MAX_NF], st_bwd[VFFT_ZTT_MAX_NF];
    vfft_ztt_kfn tl_fwd_plane, tl_bwd_plane;
    size_t twoff[VFFT_ZTT_MAX_NF];
    /* THE THREADED ARM (ztt_mt.h, docs/design/ztt_mt_design.md): mt = the
     * raced arm (0 serial, 1 BLOCKS, 2 TILES), mt_t = the thread count it
     * was bound and banked for; at T > 1 every cell runs the sectioned
     * staged walk, the fused codelet is the serial form only. */
    int mt, mt_t;
} vfft_ztt_plan_t;

/* the registry row for (N, chain), NULL when the cell was not emitted */
static inline const vfft_ztt_cell_t *vfft_ztt_lookup(int N, const int *chain, int nf)
{
    int i, s;
    for (i = 0; i < VFFT_ZTT_NCELLS_AVX2; i++)
    {
        const vfft_ztt_cell_t *c = &vfft_ztt_cells_avx2[i];
        if (c->n != N || c->nf != nf) continue;
        for (s = 0; s < nf; s++)
            if (c->chain[s] != chain[s]) break;
        if (s == nf) return c;
    }
    return NULL;
}

static inline void vfft_ztt_destroy(vfft_ztt_plan_t *p)
{
    if (!p) return;
    VFFT_ZTT_FREE(p->tw);
    VFFT_ZTT_FREE(p->twb);
    VFFT_ZTT_FREE(p->rb);
    VFFT_ZTT_FREE(p->plane);
    VFFT_ZTT_FREE(p->perm);
    free(p);
}

/* mixed-radix digit reversal over R[0..nR-1]: n = d0 + R0*(d1 + ...), d0
 * fastest; digit d_s carries weight prod_{u>s} R[u] (CONTRACT.md 5) */
static inline long _ztt_digitrev(long n, const int *R, int nR)
{
    long out = 0;
    int s, u;
    for (s = 0; s < nR; s++)
    {
        const int d = (int)(n % R[s]);
        long wt = 1;
        n /= R[s];
        for (u = s + 1; u < nR; u++) wt *= R[u];
        out += (long)d * wt;
    }
    return out;
}

/* sin(2*pi*u/M) from the quarter wave, u in [0, M): quadrant q = u / Q,
 * rem = u % Q: q=0 T[rem], q=1 T[Q-rem] (mirror about pi/2), q>=2 negated
 * (a sign-bit flip, so the -0.0 the contract requires at r*b == 0 comes out
 * bit for bit) */
static inline double _ztt_qsin(long u)
{
    const long q = u >> VFFT_ZTT_QW_LGQ;
    const long rem = u & (VFFT_ZTT_QW_Q - 1);
    const double v = (q & 1) ? VFFT_ZTT_QW16384_SIN[VFFT_ZTT_QW_Q - rem]
                             : VFFT_ZTT_QW16384_SIN[rem];
    return (q & 2) ? -v : v;
}

static inline int _ztt_log2(long v)
{
    int k = 0;
    while ((1L << k) < v) k++;
    return k;
}

/* one stage's stream (CONTRACT.md 4): record for (column quad k, leg r>=1)
 * at tw + ((k/4)*(R-1) + (r-1))*8 doubles = [c(k..k+3)][s(k..k+3)] of
 * w_{RL}^{r*b}, b = k + lane; fwd s = -sin (times c + i s = exp(-i th)),
 * bwd s = +sin. Returns the doubles written = 2*(R-1)*L. */
static inline size_t _ztt_fill_stage(double *tw, long L, int R, long RL, int bwd)
{
    if (RL & (RL - 1))
    {   /* a 2^a*odd modulus (an odd mid, and every stage after it in the
         * chain): no octave, no quarter wave — the angle 2*pi*pw/RL from libm,
         * reduced to [-pi, pi] (the cascade's odd stages did the same). The
         * fwd s = -sin(0) = -0.0 at r*b == 0, as the record contract wants. */
        long k;
        int r, lane;
        for (k = 0; k < L; k += 4)
            for (r = 1; r < R; r++)
            {
                double *rec = tw + ((size_t)(k / 4) * (size_t)(R - 1) + (size_t)(r - 1)) * 8;
                for (lane = 0; lane < 4; lane++)
                {
                    const long b = k + lane;
                    long pw = ((long)r * b) % RL;
                    double c, sn;   /* a = 2*pi*pw/RL */
                    if (2 * pw > RL) pw -= RL;
                    vfft_cs2pi_exact((long long)pw, (long long)RL, &c, &sn);
                    rec[lane] = c;
                    rec[4 + lane] = bwd ? sn : -sn;
                }
            }
        return (size_t)2 * (size_t)(R - 1) * (size_t)L;
    }
    const int lg = _ztt_log2(RL);
    /* up = the bits of pw ABOVE the table's octave (0 within it): the fine
     * table has 2^up entries, cos/sin of 2*pi*b/RL, b < 2^up (<= 16 at the
     * ceiling) — the only trig the create ever runs */
    const int up = lg > VFFT_ZTT_QW_LGM ? lg - VFFT_ZTT_QW_LGM : 0;
    const int sh = up ? 0 : VFFT_ZTT_QW_LGM - lg;    /* M = 2^LGM */
    double fc[64], fs[64];
    long k, bf;
    int r, lane;
    if (up)
        for (bf = 0; bf < (1L << up); bf++)
        {
            vfft_cs2pi_exact((long long)bf, (long long)RL, &fc[bf], &fs[bf]);
        }
    for (k = 0; k < L; k += 4)
        for (r = 1; r < R; r++)
        {
            double *rec = tw + ((size_t)(k / 4) * (size_t)(R - 1) + (size_t)(r - 1)) * 8;
            for (lane = 0; lane < 4; lane++)
            {
                const long b = k + lane;
                const long pw = ((long)r * b) % RL;
                double c, s;
                if (!up)
                {
                    const long idx = pw << sh;
                    s = _ztt_qsin(idx);
                    c = _ztt_qsin((idx + VFFT_ZTT_QW_Q) & (VFFT_ZTT_QW_M - 1));
                }
                else
                {   /* pw = a*2^up + bb: w = table(a) * fine(bb) */
                    const long a = pw >> up, bb = pw & ((1L << up) - 1);
                    const double sa = _ztt_qsin(a);
                    const double ca = _ztt_qsin((a + VFFT_ZTT_QW_Q) & (VFFT_ZTT_QW_M - 1));
                    c = ca * fc[bb] - sa * fs[bb];
                    s = sa * fc[bb] + ca * fs[bb];
                }
                rec[lane] = c;
                rec[4 + lane] = bwd ? s : -s;
            }
        }
    return (size_t)2 * (size_t)(R - 1) * (size_t)L;
}

/* the chain grammar (ztt_odd_design.md): the ends are 4 or 8 (the ingest's
 * turn lattice and the terminators' lane transposes), a mid is 4, 8 or an odd
 * 3/5/7/9/15 (its edges are radix-agnostic) */
static inline int _ztt_radix_ok(int R, int mid)
{
    if (R == 4 || R == 8) return 1;
    return mid && (R == 3 || R == 5 || R == 7 || R == 9 || R == 15);
}

/* create: the validator is the law (NULL = not a ZTURN-T cell, loudly under
 * VFFT_NAT_LOG). Bound out of place; vfft_ztt_bind(p, 1) rebinds in place.
 * A cell with a fused codelet binds it; a cell without one (every 2^a*odd
 * cell) is STAGED — its stage table is resolved here from the kind registry
 * and refused if a radix has no kernel. force_staged builds the staged form
 * at a pow2 cell too: the gate's tool (staged == fused bitwise), nothing
 * else's. */
static inline vfft_ztt_plan_t *_ztt_create(int N, const int *chain, int nf, int scr, int force_staged)
{
    vfft_ztt_plan_t *p;
    const vfft_ztt_cell_t *cell;
    long prod = 1, j;
    int s, staged = 0;
    size_t off;
    const char *why = NULL;
    if (nf < 2 || nf > VFFT_ZTT_MAX_NF) why = "nf outside 2..7";
    else if (N < 16 || N > VFFT_ZTT_MAX_N) why = "N outside 16..262144";
    for (s = 0; !why && s < nf; s++)
    {
        if (!_ztt_radix_ok(chain[s], s > 0 && s < nf - 1))
            why = "radix not in {4, 8} at an end, or not in {4, 8, 3, 5, 7, 9, 15} in a mid";
        prod *= chain[s];
    }
    if (!why && prod != (long)N) why = "chain product != N";
    if (!why && (N / chain[0]) % 4) why = "(N / R0) % 4 != 0";
    cell = why ? NULL : vfft_ztt_lookup(N, chain, nf);
    if (!why) staged = force_staged || !cell;
    if (!why && staged && !force_staged && (N & (N - 1)) == 0)
        why = "no fused driver for this pow2 cell (ztt_registry_avx2.h)";
    if (why)
    {
        if (getenv("VFFT_NAT_LOG"))
            fprintf(stderr, "[ztt] N=%d: refused: %s\n", N, why);
        return NULL;
    }
    p = (vfft_ztt_plan_t *)calloc(1, sizeof *p);
    if (!p) return NULL;
    p->N = N;
    p->nf = nf;
    for (s = 0; s < nf; s++) p->chain[s] = chain[s];
    p->cell = cell;
    p->ncol = N / chain[0];
    p->scr = scr ? 1 : 0;
    p->staged = staged;
    {   /* the stage table: every kernel resolved now, none looked up at run
         * time — for EVERY plan, the fused cells too: the threaded arm
         * (ztt_mt.h) sections this table at any cell */
        int ok = 1;
        for (s = 0; s < nf; s++)
        {
            const int R = chain[s];
            if (p->scr)
            {
                if (s == 0)           { p->st_fwd[s] = _ztt_kfn_t0d(R, 0);  p->st_bwd[s] = _ztt_kfn_tlf(R, 1); }
                else if (s == nf - 1) { p->st_fwd[s] = _ztt_kfn_tld(R, 0);  p->st_bwd[s] = _ztt_kfn_tld(R, 1); }
                else                  { p->st_fwd[s] = _ztt_kfn_tmgd(R, 0); p->st_bwd[s] = _ztt_kfn_tmg(R, 1); }
            }
            else
            {
                if (s == 0)           { p->st_fwd[s] = _ztt_kfn_t0tp(R, 0); p->st_bwd[s] = _ztt_kfn_t0tp(R, 1); }
                else if (s == nf - 1) { p->st_fwd[s] = _ztt_kfn_tlf(R, 0);  p->st_bwd[s] = _ztt_kfn_tlf(R, 1);
                                        p->tl_fwd_plane = _ztt_kfn_tlfi(R, 0); p->tl_bwd_plane = _ztt_kfn_tlfi(R, 1);
                                        ok = ok && p->tl_fwd_plane && p->tl_bwd_plane; }
                else                  { p->st_fwd[s] = _ztt_kfn_tmg(R, 0);  p->st_bwd[s] = _ztt_kfn_tmg(R, 1); }
            }
            ok = ok && p->st_fwd[s] && p->st_bwd[s];
        }
        if (!ok)
        {
            if (getenv("VFFT_NAT_LOG"))
                fprintf(stderr, "[ztt] N=%d: refused: no stage kernel for a radix in this chain\n", N);
            free(p);
            return NULL;
        }
    }
    if (p->scr)
    {
        /* THE PLAIN SCHEDULE (ztt_scrambled_design.md): Len_0 = N,
         * Len_{s+1} = Len_s / R_s; stage s < nf-1 has Len_{s+1} columns at
         * root Len_s — the natural fill routine with (columns, R, modulus) =
         * (Len_{s+1}, R_s, Len_s); the last stage is twiddle-free. No run-base
         * table, no plane. */
        p->len[nf] = 1;
        for (s = nf - 1; s >= 0; s--) p->len[s] = p->len[s + 1] * chain[s];
        p->twdoubles = 0;
        for (s = 0; s < nf - 1; s++)
            p->twdoubles += (size_t)2 * (size_t)(chain[s] - 1) * (size_t)p->len[s + 1];
        p->tw = (double *)VFFT_ZTT_ALLOC(p->twdoubles * sizeof(double));
        p->twb = (double *)VFFT_ZTT_ALLOC(p->twdoubles * sizeof(double));
        p->perm = (size_t *)VFFT_ZTT_ALLOC((size_t)N * sizeof(size_t));
        if (!p->tw || !p->twb || !p->perm) { vfft_ztt_destroy(p); return NULL; }
        for (s = 0, off = 0; s < nf - 1; s++)
        {
            const size_t n = _ztt_fill_stage(p->tw + off, p->len[s + 1], chain[s], p->len[s], 0);
            (void)_ztt_fill_stage(p->twb + off, p->len[s + 1], chain[s], p->len[s], 1);
            p->twoff[s] = off;
            off += n;
        }
        p->twoff[nf - 1] = off;   /* the last stage carries no stream */
        /* the permutation: frequency k sits at the in-place DIF position
         * ic = digitrev(k) over the chain, then inside the last stage's
         * 4-column span at tld's unpack-only lane order [c, c+2 | c+1, c+3]:
         * R*(col & ~3) + 4*p + [0,2,1,3][col & 3], col = ic / R, p = ic % R */
        {
            static const size_t sig[4] = { 0, 2, 1, 3 };
            const long R = chain[nf - 1];
            for (j = 0; j < (long)N; j++)
            {
                const long ic = _ztt_digitrev(j, chain, nf);
                const long col = ic / R, pp = ic % R;
                p->perm[j] = (size_t)(R * (col & ~3L) + 4 * pp) + sig[col & 3];
            }
        }
        p->inplace = 0;
        p->tile = 0;
        p->fwd = NULL;   /* the plain fused codelets are reached through the cell */
        p->bwd = NULL;
        return p;
    }
    /* geometry: L[1] = R0, L[s+1] = L[s]*R[s], Gs[s] = N / (R[s]*L[s]) */
    p->L[1] = chain[0];
    p->twdoubles = 0;
    for (s = 1; s < nf; s++)
    {
        const long RL = p->L[s] * chain[s];
        p->Gs[s] = N / RL;
        p->twdoubles += (size_t)2 * (size_t)(chain[s] - 1) * (size_t)p->L[s];
        if (s + 1 < nf) p->L[s + 1] = RL;
    }
    p->tw = (double *)VFFT_ZTT_ALLOC(p->twdoubles * sizeof(double));
    p->twb = (double *)VFFT_ZTT_ALLOC(p->twdoubles * sizeof(double));
    p->rb = (size_t *)VFFT_ZTT_ALLOC((size_t)p->ncol * sizeof(size_t));
    p->plane = (double *)VFFT_ZTT_ALLOC((size_t)2 * (size_t)N * sizeof(double) + 4096u);
    if (!p->tw || !p->twb || !p->rb || !p->plane) { vfft_ztt_destroy(p); return NULL; }
    /* the streams, stage order, ONE allocation each (the fused driver's
     * carried cursor walks straight from one stage's end into the next) */
    for (s = 1, off = 0; s < nf; s++)
    {
        const size_t n = _ztt_fill_stage(p->tw + off, p->L[s], chain[s], p->L[s] * chain[s], 0);
        (void)_ztt_fill_stage(p->twb + off, p->L[s], chain[s], p->L[s] * chain[s], 1);
        p->twoff[s] = off;
        off += n;
    }
    /* the run-base table: rb[c] = the run j whose digit reversal (over the
     * middle+last radices) is column c; refused unless a bijection */
    for (j = 0; j < p->ncol; j++) p->rb[j] = (size_t)-1;
    for (j = 0; j < p->ncol; j++)
    {
        const long c = _ztt_digitrev(j, chain + 1, nf - 1);
        if (c < 0 || c >= p->ncol || p->rb[c] != (size_t)-1)
        {
            if (getenv("VFFT_NAT_LOG"))
                fprintf(stderr, "[ztt] N=%d: run-base table is not a bijection (c=%ld)\n", N, c);
            vfft_ztt_destroy(p);
            return NULL;
        }
        p->rb[c] = (size_t)j;
    }
    p->inplace = 0;
    p->tile = 0;
    p->fwd = cell ? cell->fwd_dest : NULL;   /* staged: the stage table runs */
    p->bwd = cell ? cell->bwd_dest : NULL;
    return p;
}

/* the order-class create: scr = 0 natural, 1 the plain schedule; fused where
 * the cell has a codelet, staged where it has none */
static inline vfft_ztt_plan_t *vfft_ztt_create_chain_ord(int N, const int *chain, int nf, int scr)
{
    return _ztt_create(N, chain, nf, scr, 0);
}

/* the natural-order create: the order class is a property of the plan for
 * its whole life (scr = 0 here, 1 = the plain schedule, ..._ord above) */
static inline vfft_ztt_plan_t *vfft_ztt_create_chain(int N, const int *chain, int nf)
{
    return _ztt_create(N, chain, nf, 0, 0);
}

/* plain plans: the output position of frequency k (gates, introspection) */
static inline size_t vfft_ztt_perm(const vfft_ztt_plan_t *p, long k)
{
    return p->scr ? p->perm[k] : (size_t)k;
}

/* placement binding: out of place = the `dest` drivers (the pipeline runs
 * in zout), in place = the `plane` drivers (zin is consumed by the ingest
 * before the last stage writes zout, so zin == zout is legal) */
static inline void vfft_ztt_bind(vfft_ztt_plan_t *p, int inplace)
{
    p->inplace = inplace ? 1 : 0;
    if (p->scr || p->staged) return;   /* plain: placement-free; staged: the walk reads inplace */
    p->fwd = inplace ? p->cell->fwd_plane : p->cell->fwd_dest;
    p->bwd = inplace ? p->cell->bwd_plane : p->cell->bwd_dest;
}

/* THE PURE-POW2 BAND (owner's law, design_contracts.md section 4,
 * 2026-09-09): at a power of two in 16..VFFT_ZTT_MAX_N the interleaved K=1
 * cell belongs to the solo kernels and the pairs (<= 64), the pairs and
 * ZTURN-T (128..1024) and ZTURN-T alone (2048 and up) — and NO cascade race
 * arm exists at any NATURAL or DEFAULT door, out of place or in place: "no
 * cascade race arm please, eliminate". The explicit SCRAMBLED cell (out of
 * place and in place) keeps the cascade, its only scrambled writer, until
 * the scrambled ZTURN-T class exists. Before this the natural door raced the natord cascade
 * from 128 up and had banked it at 512 (400 ns over a 300 ns pair — a door
 * clock's verdict, not the tier's). The doors consult this before building a
 * cascade candidate; outside the band (above the ceiling, any odd factor)
 * they behave as before. */
static inline int vfft_ztt_band(int N)
{
    return N >= 16 && N <= VFFT_ZTT_MAX_N && (N & (N - 1)) == 0;
}

/* THE ODD BAND (ztt_odd_design.md, 2026-09-14): N = 2^a * m, a >= 4, m > 1 a
 * product over {3, 5, 7, 9, 15} with at most five odd mids (nf <= 7 with the
 * two pow2 ends), 2048 <= N <= the ceiling — the cascade's exact odd cell
 * set. The chain grammar is the planner's (_il_dp_enumerate_odd_mids): the
 * odd part decomposed greedily largest-first, its mids at every interior
 * position, the pow2 slots over ordered {4, 8}. */
static inline int vfft_ztt_odd_band(int N)
{
    static const int OP[5] = { 15, 9, 7, 5, 3 };
    int m = N, a = 0, nm = 0, i;
    if (N < 2048 || N > VFFT_ZTT_MAX_N) return 0;
    while ((m & 1) == 0) { m >>= 1; a++; }
    if (m == 1 || a < 4) return 0;
    for (i = 0; i < 5; i++)
        while (m % OP[i] == 0) { m /= OP[i]; nm++; }
    return m == 1 && nm <= VFFT_ZTT_MAX_NF - 2;
}

/* the tile law: 0 (untiled) is always legal; else a width that DIVIDES N
 * (the tiles partition the array), that the first mid's group R0*R1 divides
 * (below it no stage tiles: a tile holds whole groups of every tiled stage,
 * and the tiled stages are the mids whose R*L divides the width) and that is
 * below N; a chain without mids (nf == 2) tiles nothing and refuses every
 * width. At a pow2 N this is exactly "a power of two, >= R0*R1, < N"; at a
 * 2^a*odd N it admits the ladder's 48 KB (3072) where the chain's products
 * allow it and refuses a pow2 width the odd mid's group does not divide.
 * Shared by the create, the planner's ladder and the gate. */
static inline int vfft_ztt_tile_legal(int N, const int *chain, int nf, size_t tile)
{
    if (tile == 0) return 1;
    if (nf < 3) return 0;
    if ((size_t)N % tile) return 0;
    if (tile % ((size_t)chain[0] * (size_t)chain[1])) return 0;
    return tile < (size_t)N;
}

/* the same law for the plain schedule, mirrored: its tiled stages are the
 * SUFFIX (Len shrinks with s), so the smallest block that tiles anything is
 * the last mid's, R_{nf-2} * R_{nf-1}, and the per-block stages are those
 * whose Len divides the width */
static inline int vfft_ztt_tile_legal_ord(int N, const int *chain, int nf, size_t tile, int scr)
{
    if (!scr) return vfft_ztt_tile_legal(N, chain, nf, tile);
    if (tile == 0) return 1;
    if (nf < 3) return 0;
    if ((size_t)N % tile) return 0;
    if (tile % ((size_t)chain[nf - 2] * (size_t)chain[nf - 1])) return 0;
    return tile < (size_t)N;
}

static inline int vfft_ztt_set_tile(vfft_ztt_plan_t *p, size_t tile)
{
    if (!vfft_ztt_tile_legal_ord(p->N, p->chain, p->nf, tile, p->scr)) return 0;
    p->tile = tile;
    return 1;
}

/* the plane's PAGE OFFSET against the caller's buffer (measured 2026-09-09,
 * probes/ZT/zt_plane_skew.c, and again with tlfi in the tree): with the plane
 * just below zout in its 4 KB slot, the terminator's stores to zout sit in the
 * same 4K slot as its loads from the plane a few column quads ahead — a
 * 4K-alias stall worth +20..30% in place at 4096..16384, SEPARATE from the
 * store-miss latency tlfi's prefetch removes (both are needed). The plane
 * carries 4 KB of slack and each in-place call starts it 2 KB (mod 4 KB) from
 * zout, 64-B aligned: the alias distance becomes 128 columns. Arithmetic is
 * untouched (the gate's in-place bitwise law holds). */
static inline double *_ztt_plane_for(const vfft_ztt_plan_t *p, const double *zout)
{
    const uintptr_t base = (uintptr_t)p->plane;
    const uintptr_t want = ((uintptr_t)zout + 2048u) & 4095u;
    const uintptr_t off = ((want - (base & 4095u)) & 4095u) & ~(uintptr_t)63u;
    return (double *)(base + off);
}

/* THE STAGED WALK (ztt_odd_design.md): the fused drivers' loop nests
 * (ztt_drivers.ml emit_driver / emit_plain_driver) with run-time bounds. A
 * mid runs per tile when its group (natural: R*L; plain: Len) divides the
 * width, else it sweeps — at a pow2 cell that is the drivers' "<= tile", and
 * the gate holds the two bitwise. Natural: ingest sweep; per tile the mids
 * whose R*L divides it; the other mids as sweeps; the terminator (tlf in the
 * destination, tlfi from the plane in place). Plain fwd: t0d sweep; the
 * sweeping mids in stage order; per block the dividing mids then tld. Plain
 * bwd: per block tldb then the dividing mids high to low; the other mids
 * high to low as sweeps; tlfb in place. */
static inline void _ztt_staged_run(const vfft_ztt_plan_t *p, const double *zin, double *zout, int bwd)
{
    const int nf = p->nf;
    const size_t N = (size_t)p->N, tile = p->tile;
    const vfft_ztt_kfn *st = bwd ? p->st_bwd : p->st_fwd;
    const double *tw = bwd ? p->twb : p->tw;
    int s;
    size_t t;
    if (p->scr)
    {
        const size_t Rl = (size_t)p->chain[nf - 1];
        if (!bwd)
        {
            st[0](zin, 0, zout, 0, tw, 0, (size_t)p->len[1], 1, 0, 0, (size_t)p->len[1]);
            for (s = 1; s < nf - 1; s++)
                if (!tile || tile % (size_t)p->len[s])
                    st[s](zout, 0, zout, 0, tw + p->twoff[s], 0,
                          (size_t)p->len[s + 1], N / (size_t)p->len[s], 0, 0, (size_t)p->len[s + 1]);
            if (tile)
                for (t = 0; t < N / tile; t++)
                {
                    double *B = zout + t * tile * 2;
                    for (s = 1; s < nf - 1; s++)
                        if (tile % (size_t)p->len[s] == 0)
                            st[s](B, 0, B, 0, tw + p->twoff[s], 0,
                                  (size_t)p->len[s + 1], tile / (size_t)p->len[s], 0, 0, (size_t)p->len[s + 1]);
                    st[nf - 1](B, 0, B, 0, tw, 0, 0, 1, 0, 0, tile / Rl);
                }
            else
                st[nf - 1](zout, 0, zout, 0, tw, 0, 0, 1, 0, 0, N / Rl);
        }
        else
        {
            if (tile)
                for (t = 0; t < N / tile; t++)
                {
                    const double *Bi = zin + t * tile * 2;
                    double *B = zout + t * tile * 2;
                    st[nf - 1](Bi, 0, B, 0, tw, 0, 0, 1, 0, 0, tile / Rl);
                    for (s = nf - 2; s >= 1; s--)
                        if (tile % (size_t)p->len[s] == 0)
                            st[s](B, 0, B, 0, tw + p->twoff[s], 0,
                                  (size_t)p->len[s + 1], tile / (size_t)p->len[s], 0, 0, (size_t)p->len[s + 1]);
                }
            else
                st[nf - 1](zin, 0, zout, 0, tw, 0, 0, 1, 0, 0, N / Rl);
            for (s = nf - 2; s >= 1; s--)
                if (!tile || tile % (size_t)p->len[s])
                    st[s](zout, 0, zout, 0, tw + p->twoff[s], 0,
                          (size_t)p->len[s + 1], N / (size_t)p->len[s], 0, 0, (size_t)p->len[s + 1]);
            st[0](zout, 0, zout, 0, tw, 0, (size_t)p->len[1], 1, (size_t)p->len[1], 0, (size_t)p->len[1]);
        }
        return;
    }
    {
        const int ip = (zin == zout || p->inplace);
        double *W = ip ? _ztt_plane_for(p, zout) : zout;
        const vfft_ztt_kfn last = ip ? (bwd ? p->tl_bwd_plane : p->tl_fwd_plane) : st[nf - 1];
        const size_t L = (size_t)p->L[nf - 1];
        st[0](zin, 0, W, 0, 0, (const double *)p->rb, (size_t)p->ncol, 0, 0, 0, (size_t)p->ncol);
        if (tile)
            for (t = 0; t < N / tile; t++)
            {
                double *B = W + t * tile * 2;
                for (s = 1; s < nf - 1; s++)
                {
                    const size_t RL = (size_t)p->L[s] * (size_t)p->chain[s];
                    if (tile % RL == 0)
                        st[s](B, 0, B, 0, tw + p->twoff[s], 0, (size_t)p->L[s], tile / RL, 0, 0, (size_t)p->L[s]);
                }
            }
        for (s = 1; s < nf - 1; s++)
        {
            const size_t RL = (size_t)p->L[s] * (size_t)p->chain[s];
            if (!tile || tile % RL)
                st[s](W, 0, W, 0, tw + p->twoff[s], 0, (size_t)p->L[s], (size_t)p->Gs[s], 0, 0, (size_t)p->L[s]);
        }
        last(W, 0, zout, 0, tw + p->twoff[nf - 1], 0, L, 1, L, 0, L);
    }
}

static inline void vfft_ztt_execute_fwd(const vfft_ztt_plan_t *p,
                                        const double *zin, double *zout)
{
    if (p->staged) { _ztt_staged_run(p, zin, zout, 0); return; }
    if (p->scr) { p->cell->fwd_scr(zin, zout, p->tw, p->tile); return; }   /* the plain schedule: one function, either placement */
    if (zin == zout || p->inplace)
        p->cell->fwd_plane(zin, zout, _ztt_plane_for(p, zout), p->tw, p->rb, p->tile);
    else
        p->fwd(zin, zout, p->plane, p->tw, p->rb, p->tile);
}

static inline void vfft_ztt_execute_bwd(const vfft_ztt_plan_t *p,
                                        const double *zin, double *zout)
{
    if (p->staged) { _ztt_staged_run(p, zin, zout, 1); return; }
    if (p->scr) { p->cell->bwd_scr(zin, zout, p->twb, p->tile); return; }   /* the plain schedule's stage-by-stage inverse */
    if (zin == zout || p->inplace)
        p->cell->bwd_plane(zin, zout, _ztt_plane_for(p, zout), p->twb, p->rb, p->tile);
    else
        p->bwd(zin, zout, p->plane, p->twb, p->rb, p->tile);
}

/* "4.4.8" for logs and the wisdom token il_ztt= */
static inline int vfft_ztt_chain_str(const vfft_ztt_plan_t *p, char *buf, size_t n)
{
    int s, off = 0;
    for (s = 0; s < p->nf && (size_t)off < n; s++)
        off += snprintf(buf + off, n - (size_t)off, "%s%d", s ? "." : "", p->chain[s]);
    return off;
}

#endif /* VFFT_ZTT_H */
