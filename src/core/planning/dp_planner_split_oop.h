/* dp_planner_split_oop.h — the SPLIT-OOP-engine planner (kind-3 sp fields).
 *
 * NAMING (owner terminology, 2026-08-18). "K4" is the MATH view of this
 * engine: split-layout SIMD has no intra-complex parallelism to exploit
 * without shuffles, so its 4 AVX2 lanes must hold 4 independent same-shaped
 * problems. This engine serves ONE caller transform per call by
 * MANUFACTURING that lane-batch from its own decomposition — the four-step
 * splits N = R1×R2 into R1 independent length-R2 sub-DFTs and runs 4 of
 * them per iteration (GROUPW; 8 on a future AVX-512 port). So: one
 * transform per call at the API, 4 batched sub-FFTs in one go in the math.
 * The wisdom cell key stays (N, K=1) — K in the wisdom counts CALLER
 * transforms, and the caller handed us one. CCOL is this same idea taken
 * wholesale: reshape the one transform into an explicit K=R1 column batch
 * and hand it to the batched engine below.
 *
 * The planner family, by engine:
 *   dp_planner.h          — the split CALLER-BATCHED proto stride engine
 *                           (lanes = the caller's own transforms, K ≡ 0
 *                           mod 8; recursive memoized DP; also the INNER
 *                           tuner the CCOL axis below delegates to);
 *   dp_planner_il.h       — the single-transform INTERLEAVED engines
 *                           (il2p + zturn cascade);
 *   dp_planner_split_oop.h — this file: the single-transform SPLIT engine
 *                           (lane-batch of its own sub-problems;
 *                           natural-order; mono/3P/2PA/2PB/TWL/L3/CCOL).
 *
 * This header OWNS the split planning phase:
 * candidate enumeration (route × pair × CCOL R1 × column chain × column
 * variants), the correctness gate, the order-rotated timing discipline,
 * winner selection, and banking through the SHIPPED writers. Bench
 * harnesses (build_tuned/benches/calibrate_k1.c) are THIN DRIVERS over
 * this header — they parse arguments and call in; they hold no planning
 * logic (owner directive 2026-08-18).
 *
 * The split race here was MIGRATED verbatim from calibrate_k1.c v2
 * (candidate table, gate-before-time, reseed-per-burst, order-rotated
 * best-of-trials, split-IP-axis banking contract). What is NEW vs v2:
 *   - CCOL is a raced AXIS, not a single default arm: R1 ∈ {8,16,32,64}
 *     (the column engine needs K=R1 ≡ 0 mod 8), with the column plan's
 *     chain + per-stage variants tuned by the EXISTING proto DP
 *     (measure.h) pinned to DIT — the OOP boundary (oop_execute.h) is
 *     DIT-only, so DIF winners are structurally unusable here.
 *   - The inner (R2, K=R1) tunings are banked as ordinary spike-wisdom v8
 *     lines under the B1 write policy: NEVER overwrite a DIF-tuned line
 *     (that is the batched product's verdict; objectives differ); write
 *     only when the cell is absent or an existing DIT line is beaten.
 *   - A CCOL split winner's chain rides the kind-3 line (cc_chain token;
 *     vfft_il_dp_emit_wisdom carries it since the B2 signature change).
 *   - The correctness reference is O(N log N) (scalar radix-2, self-checked
 *     at 8 bins by direct summation) so cells ≥ 8192 gate in milliseconds;
 *     non-pow2 N falls back to the O(N^2) direct reference.
 *
 * Duplicate spike keys: the shipped spike file carries some duplicate
 * (N,K) rows from the 2026-07-23 era. Policy here (and for any reader):
 * FIRST match wins; a duplicate is logged, never silently rewritten.
 *
 * 🔮 FUTURE (owner-chartered 2026-08-18): this planner extends to ODD and
 * PRIME N — native split RADER and BLUESTEIN routes enter THIS enumeration
 * and race on the same kind-3 sp axis, banked through the same writers.
 * The %4-only pair filter and pow2 CCOL chains below are the CURRENT
 * scope, not the design boundary. Add those routes here, never in a bench.
 */
#ifndef VFFT_DP_PLANNER_SPLIT_OOP_H
#define VFFT_DP_PLANNER_SPLIT_OOP_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#include "executor.h"
#include "planner.h"
#include "oop_plan.h"
#include "oop_wisdom.h"
#include "dp_planner.h"
#include "measure.h"
#include "wisdom_reader.h"
#include "dp_planner_il.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* QPC directly (same reason as the v2 harness: vfft_proto_now_ns is a C99
 * inline whose external definition is not guaranteed in every TU). */
static double _sp_now_ns(void)
{
#ifdef _WIN32
    LARGE_INTEGER f, c;
    QueryPerformanceFrequency(&f);
    QueryPerformanceCounter(&c);
    return 1e9 * (double)c.QuadPart / (double)f.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return 1e9 * ts.tv_sec + ts.tv_nsec;
#endif
}

/* ── split-side routes (v2's table, unchanged) ────────────────────── */
enum { VFFT_SP_R_3P = 0, VFFT_SP_R_3P_IP, VFFT_SP_R_2PA_IP, VFFT_SP_R_2PB_IP,
       VFFT_SP_R_TWL_IP, VFFT_SP_R_3PL3_IP, VFFT_SP_R_2PAL3_IP,
       VFFT_SP_R_MONO, VFFT_SP_R_MONO_ALT, VFFT_SP_R_CCOL, VFFT_SP_R_NROUTES };
static const char *VFFT_SP_RNAME[VFFT_SP_R_NROUTES] = {
    "3p", "3p-ip", "2pa-ip", "2pb-ip", "twl-ip",
    "3p-l3-ip", "2pa-l3-ip", "mono", "mono-alt", "cc" };
/* axis: 0 = split-oop, 1 = split-ip (the banked sp_route axis) */
static const int VFFT_SP_RAXIS[VFFT_SP_R_NROUTES] = { 0, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
static const int VFFT_SP_SPMAP[VFFT_SP_R_NROUTES] = {
    VFFT_K1_SP_3P, VFFT_K1_SP_3P, VFFT_K1_SP_2PA, VFFT_K1_SP_2PB,
    VFFT_K1_SP_TWL, VFFT_K1_SP_3P_L3, VFFT_K1_SP_2PA_L3,
    VFFT_K1_SP_MONO, VFFT_K1_SP_MONO, VFFT_K1_SP_CCOL };

typedef struct {
    int route, R1, R2, cc_code, cc_vcode;
    vfft_oop_plan_t *p;
    double best;
    int gated;
} vfft_sp_cand_t;

#define VFFT_SP_MAX_CAND  128
#define VFFT_SP_MAX_PLANS 32

/* per-cell bench state (no file-scope mutables in a production header) */
typedef struct {
    int N;
    double *xr, *xi;   /* pristine input */
    double *dr, *di;   /* OOP dst */
    double *wr, *wi;   /* in-place working */
    double *Rr, *Ri;   /* reference */
} _sp_bench_t;

static double *_sp_ad(size_t n)
{
    double *p = NULL;
    if (vfft_proto_posix_memalign((void **)&p, 64, n * sizeof(double)) != 0)
        return NULL;
    return p;
}

static void _sp_cachebust(void)
{
    size_t s = 32u * 1024u * 1024u / 8u;
    double *j = (double *)malloc(s * 8);
    volatile double a = 0;
    if (!j) return;
    for (size_t i = 0; i < s; i++) j[i] = (double)i * 0.5;
    for (size_t i = 0; i < s; i++) a += j[i];
    (void)a;
    free(j);
}

static void _sp_reseed(_sp_bench_t *b)
{
    memcpy(b->wr, b->xr, (size_t)b->N * 8);
    memcpy(b->wi, b->xi, (size_t)b->N * 8);
}

static void _sp_run_cand(_sp_bench_t *b, const vfft_sp_cand_t *c)
{
    switch (c->route) {
    case VFFT_SP_R_3P:       vfft_oop_execute_fwd(c->p, b->xr, b->xi, b->dr, b->di); break;
    case VFFT_SP_R_3P_IP:    vfft_oop_execute_fwd(c->p, b->wr, b->wi, b->wr, b->wi); break;
    case VFFT_SP_R_2PA_IP:   vfft_oop_execute_fwd_2pa(c->p, b->wr, b->wi, b->wr, b->wi); break;
    case VFFT_SP_R_2PB_IP:   vfft_oop_execute_fwd_2pb(c->p, b->wr, b->wi, b->wr, b->wi); break;
    case VFFT_SP_R_TWL_IP:   vfft_oop_execute_fwd_2pa_twl(c->p, b->wr, b->wi, b->wr, b->wi); break;
    case VFFT_SP_R_3PL3_IP:  vfft_oop_execute_fwd(c->p, b->wr, b->wi, b->wr, b->wi); break;
    case VFFT_SP_R_2PAL3_IP: vfft_oop_execute_fwd_2pa(c->p, b->wr, b->wi, b->wr, b->wi); break;
    case VFFT_SP_R_MONO:     vfft_k1_mono_fn(b->N)(b->xr, b->xi, b->dr, b->di, 0,0,0,0,0,0,0); break;
    case VFFT_SP_R_MONO_ALT: vfft_k1_mono_alt_fn(b->N)(b->xr, b->xi, b->dr, b->di, 0,0,0,0,0,0,0); break;
    case VFFT_SP_R_CCOL:     vfft_oop_execute_fwd_ccol(c->p, b->wr, b->wi, b->wr, b->wi); break;
    }
}

/* ── correctness reference ─────────────────────────────────────────
 * pow2: scalar radix-2 DIT (natural order out), SELF-CHECKED at 8 bins by
 * direct O(N) summation to 1e-9 — the dp_planner_il discipline. Any
 * self-check failure poisons the cell (returns -1) rather than gating
 * against a wrong reference. non-pow2: O(N^2) direct (future odd/prime
 * cells are small; log if used above 8192). */
static void _sp_ref_direct(const double *ar, const double *ai,
                           double *Rr, double *Ri, int N)
{
    for (int k = 0; k < N; k++) {
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * (double)((long long)k * n % N) / N;
            double c = cos(a), s = sin(a);
            sr += ar[n] * c - ai[n] * s;
            si += ar[n] * s + ai[n] * c;
        }
        Rr[k] = sr; Ri[k] = si;
    }
}

static int _sp_reference(const double *ar, const double *ai,
                         double *Rr, double *Ri, int N)
{
    if (N & (N - 1)) {
        if (N > 8192)
            printf("#   WARN: O(N^2) reference at non-pow2 N=%d\n", N);
        _sp_ref_direct(ar, ai, Rr, Ri, N);
        return 0;
    }
    /* iterative scalar radix-2 DIT: bit-reverse copy, then butterflies */
    int lg = 0; while ((1 << lg) < N) lg++;
    for (int i = 0; i < N; i++) {
        unsigned r = 0;
        for (int bit = 0; bit < lg; bit++) r |= (unsigned)(((i >> bit) & 1) << (lg - 1 - bit));
        Rr[r] = ar[i]; Ri[r] = ai[i];
    }
    for (int len = 2; len <= N; len <<= 1) {
        double ang = -2.0 * M_PI / (double)len;
        for (int i = 0; i < N; i += len)
            for (int j = 0; j < len / 2; j++) {
                double c = cos(ang * j), s = sin(ang * j);
                double ur = Rr[i + j], ui = Ri[i + j];
                double vr = Rr[i + j + len / 2] * c - Ri[i + j + len / 2] * s;
                double vi = Rr[i + j + len / 2] * s + Ri[i + j + len / 2] * c;
                Rr[i + j] = ur + vr;           Ri[i + j] = ui + vi;
                Rr[i + j + len / 2] = ur - vr; Ri[i + j + len / 2] = ui - vi;
            }
    }
    /* 8-bin self-check by direct summation */
    double mag = 0;
    for (int k = 0; k < N; k++) {
        double m = fabs(Rr[k]) + fabs(Ri[k]);
        if (m > mag) mag = m;
    }
    unsigned lcg = 0x2545F491u;
    for (int t = 0; t < 8; t++) {
        lcg = lcg * 1664525u + 1013904223u;
        int k = (int)(lcg % (unsigned)N);
        double sr = 0, si = 0;
        for (int n = 0; n < N; n++) {
            double a = -2.0 * M_PI * (double)((long long)k * n % N) / N;
            sr += ar[n] * cos(a) - ai[n] * sin(a);
            si += ar[n] * sin(a) + ai[n] * cos(a);
        }
        if (fabs(sr - Rr[k]) + fabs(si - Ri[k]) > 1e-9 * (mag > 1 ? mag : 1)) {
            printf("#   REF-SELFCHECK FAIL N=%d bin=%d — cell poisoned\n", N, k);
            return -1;
        }
    }
    return 0;
}

static int _sp_gate(_sp_bench_t *b, vfft_sp_cand_t *c)
{
    _sp_reseed(b);
    memset(b->dr, 0, (size_t)b->N * 8);
    memset(b->di, 0, (size_t)b->N * 8);
    _sp_run_cand(b, c);
    const double *or_ = (VFFT_SP_RAXIS[c->route] == 0 ||
                         c->route == VFFT_SP_R_MONO ||
                         c->route == VFFT_SP_R_MONO_ALT) ? b->dr : b->wr;
    const double *oi_ = (or_ == b->dr) ? b->di : b->wi;
    double e = 0, m = 0;
    for (int n = 0; n < b->N; n++) {
        double dr_ = fabs(or_[n] - b->Rr[n]), di_ = fabs(oi_[n] - b->Ri[n]);
        if (dr_ > e) e = dr_;
        if (di_ > e) e = di_;
        double mm = fabs(b->Rr[n]) > fabs(b->Ri[n]) ? fabs(b->Rr[n]) : fabs(b->Ri[n]);
        if (mm > m) m = mm;
    }
    c->gated = (m > 0 && e / m < 1e-9);
    if (!c->gated)
        printf("#   GATE-FAIL %s %dx%d relerr=%.2e — dropped, not timed\n",
               VFFT_SP_RNAME[c->route], c->R1, c->R2, m > 0 ? e / m : e);
    return c->gated;
}

/* ── CCOL inner tuning: DIT-pinned proto DP at (R2, K=R1) ──────────
 * Fills chain_out/var_out (cc-encodable factors only), returns nf (0 = no
 * usable DIT plan). Applies the B1 spike write policy when wisdir != NULL. */
static int _sp_cc_encodable(const int *f, int nf)
{
    for (int s = 0; s < nf; s++)
        if (f[s] < 4 || f[s] > 64 || (f[s] & (f[s] - 1)))
            return 0;
    return nf >= 1 && nf <= VFFT_K1_CC_MAX_NF;
}

/* B2.2 (2026-08-18): the CCOL verdict is SELF-CONTAINED in the kind-3
 * line (cc_chain + cc_vars) — this planner neither reads nor writes the
 * in-place engine's spike file. One engine, one wisdom file (the MODEB
 * kind-2 precedent). Raced == served is structural: the banked line
 * carries exactly the chain+variants the raced candidate was built with,
 * and create rebuilds from that line alone. */

/* B2.3 (2026-08-19, owner's OoO-context principle): isolated sub-plan
 * timings make PROPOSALS, never decisions — prepending a stage changes the
 * program (L1/L3 residency, seam distances), so the inner DP's job is only
 * to nominate plausible recipes. It returns up to VFFT_SP_CC_PROPOSALS
 * distinct DIT cc-encodable (chain, variants) recipes per (R2, R1); every
 * one is built as a COMPLETE CCOL candidate and the end-to-end race
 * decides everything, including the chain. This is exactly how the split
 * machinery already works for scrambled in-place (measure.h's top-K →
 * refine → whole-plan bench). */
#define VFFT_SP_CC_PROPOSALS 3

typedef struct {
    int nf;
    int chain[VFFT_K1_CC_MAX_NF];
    int vars[VFFT_K1_CC_MAX_NF];
    double inner_ns; /* proposer's isolated estimate — NEVER banked */
} _sp_cc_prop_t;

static int _sp_cc_prop_same(const _sp_cc_prop_t *a,
                            const vfft_proto_plan_decision_t *d)
{
    if (a->nf != d->nf) return 0;
    for (int s = 0; s < a->nf; s++)
        if (a->chain[s] != d->factors[s] || a->vars[s] != d->variants[s])
            return 0;
    return 1;
}

static int _sp_ccol_proposals(const vfft_proto_registry_t *reg,
                              int R2, int R1, _sp_cc_prop_t *props, int maxp,
                              int verbose)
{
    static vfft_proto_dp_context_t dctx; /* large buffers; static, reused */
    static int dctx_live = 0;
    if (dctx_live) { vfft_proto_dp_destroy(&dctx); dctx_live = 0; }
    vfft_proto_dp_init(&dctx, (size_t)R1, R2);
    dctx_live = 1;

    vfft_proto_plan_decision_t best, pool[VFFT_PROTO_MEASURE_TOPK * 2 * VFFT_PROTO_MEASURE_DEPLOY_MAX];
    int npool = 0;
    double ns = vfft_proto_dp_plan_measure(&dctx, R2, reg, &best, pool, &npool,
                                           verbose > 1);
    if (ns >= 1e17) return 0;

    int np = 0;
    /* the pool is sorted best-first; take distinct DIT cc-encodable recipes
     * (the overall best, when DIT+encodable, is in the pool too) */
    for (int i = 0; i < npool && np < maxp; i++) {
        if (pool[i].use_dif_forward || !_sp_cc_encodable(pool[i].factors, pool[i].nf))
            continue;
        int dup = 0;
        for (int j = 0; j < np && !dup; j++)
            dup = _sp_cc_prop_same(&props[j], &pool[i]);
        if (dup) continue;
        props[np].nf = pool[i].nf;
        memcpy(props[np].chain, pool[i].factors, pool[i].nf * sizeof(int));
        memcpy(props[np].vars, pool[i].variants, pool[i].nf * sizeof(int));
        props[np].inner_ns = pool[i].cost_ns;
        np++;
    }
    /* pool DIF-only (or empty): DIT variant search on the winner's factors */
    if (np == 0 && _sp_cc_encodable(best.factors, best.nf)) {
        int vout[STRIDE_MAX_STAGES];
        double dns = _vfft_proto_dp_variant_search(&dctx, R2, best.factors,
                        best.nf, /*use_dif=*/0, dctx.K, reg, vout,
                        NULL, NULL, 0, 0);
        if (dns < 1e17) {
            props[0].nf = best.nf;
            memcpy(props[0].chain, best.factors, best.nf * sizeof(int));
            memcpy(props[0].vars, vout, best.nf * sizeof(int));
            props[0].inner_ns = dns;
            np = 1;
        }
    }
    if (verbose) {
        if (!np)
            printf("#   ccol inner (%d,%d): no DIT-usable plan — arm skipped\n",
                   R2, R1);
        for (int i = 0; i < np; i++) {
            printf("#   ccol proposal %d/%d (%d,%d) DIT: ", i + 1, np, R2, R1);
            for (int s = 0; s < props[i].nf; s++)
                printf("%s%d", s ? "x" : "", props[i].chain[s]);
            printf(" = %.1f ns/col-pass (isolated — informational)\n",
                   props[i].inner_ns);
        }
    }
    return np;
}

/* ── the split race for one cell ───────────────────────────────────
 * Enumerates + gates + times all split candidates; fills cand[] and the
 * two axis winners (indices, -1 = none). Returns candidate count, or -1
 * on a poisoned cell (reference self-check failure / OOM). Caller owns
 * plan destruction via vfft_sp_dp_release(). */
static int vfft_sp_dp_plan(const vfft_proto_registry_t *reg,
                           const char *wisdir, int N, int rigor,
                           vfft_sp_cand_t *cand,
                           vfft_oop_plan_t **plans, int *np_out,
                           int *win_ip, int *win_oop, int verbose)
{
    int trials = rigor ? 5 : 3;
    (void)wisdir; /* B2.2: spike decoupled — kept for signature stability */
    _sp_bench_t b;
    memset(&b, 0, sizeof b);
    b.N = N;
    b.xr = _sp_ad(N); b.xi = _sp_ad(N); b.dr = _sp_ad(N); b.di = _sp_ad(N);
    b.wr = _sp_ad(N); b.wi = _sp_ad(N); b.Rr = _sp_ad(N); b.Ri = _sp_ad(N);
    if (!b.xr || !b.xi || !b.dr || !b.di || !b.wr || !b.wi || !b.Rr || !b.Ri)
        return -1;
    srand(77 + N);
    for (int n = 0; n < N; n++) {
        b.xr[n] = (double)rand() / RAND_MAX - 0.5;
        b.xi[n] = (double)rand() / RAND_MAX - 0.5;
    }
    if (_sp_reference(b.xr, b.xi, b.Rr, b.Ri, N) != 0)
        return -1;

    int nc = 0, np = 0;

    /* classic pairs + route twins (MIGRATED from v2, unchanged) */
    for (int R2 = (N < 128 ? N : 128); R2 >= 4; R2--) {
        if (N % R2) continue;
        int R1 = N / R2;
        if (R1 < 4 || R1 > 128 || (R1 % 4) || (R2 % 4)) continue;
        vfft_oop_plan_t *p = vfft_oop_plan_create_k1(N, R1, R2);
        if (!p || np >= VFFT_SP_MAX_PLANS - 2) { if (p) vfft_oop_plan_destroy(p); continue; }
        plans[np++] = p;
        struct { int route; int avail; } rs[] = {
            { VFFT_SP_R_3P, 1 }, { VFFT_SP_R_3P_IP, 1 },
            { VFFT_SP_R_2PA_IP, p->t1_ul != 0 }, { VFFT_SP_R_2PB_IP, p->leaf_ul != 0 },
            { VFFT_SP_R_TWL_IP, p->t1_ul_twl != 0 },
        };
        for (int i = 0; i < 5 && nc < VFFT_SP_MAX_CAND - 8; i++)
            if (rs[i].avail) {
                vfft_sp_cand_t t = { rs[i].route, R1, R2, 0, 0, p, 1e18, 0 };
                cand[nc++] = t;
            }
        if ((p->t1_l3 || p->t1_ul_l3) && np < VFFT_SP_MAX_PLANS - 2 &&
            nc < VFFT_SP_MAX_CAND - 8) {
            vfft_oop_plan_t *pl = vfft_oop_plan_create_k1(N, R1, R2);
            if (pl) {
                plans[np++] = pl;
                if (pl->t1_l3) {
                    pl->t1p = pl->t1_l3;
                    vfft_sp_cand_t t = { VFFT_SP_R_3PL3_IP, R1, R2, 0, 0, pl, 1e18, 0 };
                    cand[nc++] = t;
                }
                if (pl->t1_ul_l3 && nc < VFFT_SP_MAX_CAND - 8) {
                    pl->t1_ul = pl->t1_ul_l3;
                    vfft_sp_cand_t t = { VFFT_SP_R_2PAL3_IP, R1, R2, 0, 0, pl, 1e18, 0 };
                    cand[nc++] = t;
                }
            }
        }
    }

    /* CCOL axis (NEW in B2): R1 ∈ {8,16,32,64}, inner-tuned chain+variants */
    static const int CC_R1[] = { 8, 16, 32, 64 };
    for (int i = 0; i < 4; i++) {
        int R1 = CC_R1[i];
        if (N % R1) continue;
        int R2 = N / R1;
        if ((R2 % 4) || R2 < 16) continue;
        if (!vfft_oop_t1_fn(R1)) continue;
        if (np >= VFFT_SP_MAX_PLANS - VFFT_SP_CC_PROPOSALS ||
            nc >= VFFT_SP_MAX_CAND - VFFT_SP_CC_PROPOSALS - 2) break;
        _sp_cc_prop_t props[VFFT_SP_CC_PROPOSALS];
        int nprop = _sp_ccol_proposals(reg, R2, R1, props,
                                       VFFT_SP_CC_PROPOSALS, verbose);
        for (int pi = 0; pi < nprop; pi++) {
            if (np >= VFFT_SP_MAX_PLANS - 1 || nc >= VFFT_SP_MAX_CAND - 2)
                break;
            vfft_oop_plan_t *pc = vfft_oop_plan_create_k1_cc_v(
                N, R1, props[pi].chain, props[pi].nf, props[pi].vars, reg);
            if (!pc) continue;
            plans[np++] = pc;
            vfft_sp_cand_t t = { VFFT_SP_R_CCOL, R1, R2,
                                 vfft_k1_cc_chain_encode(props[pi].chain, props[pi].nf),
                                 vfft_k1_cc_vars_encode(props[pi].vars, props[pi].nf),
                                 pc, 1e18, 0 };
            cand[nc++] = t;
        }
    }

    if (vfft_k1_mono_fn(N) && nc < VFFT_SP_MAX_CAND)
    { vfft_sp_cand_t t = { VFFT_SP_R_MONO, 0, 0, 0, 0, NULL, 1e18, 0 }; cand[nc++] = t; }
    if (vfft_k1_mono_alt_fn(N) && nc < VFFT_SP_MAX_CAND)
    { vfft_sp_cand_t t = { VFFT_SP_R_MONO_ALT, 0, 0, 0, 0, NULL, 1e18, 0 }; cand[nc++] = t; }

    int ngated = 0;
    for (int k = 0; k < nc; k++) ngated += _sp_gate(&b, &cand[k]);

    int reps = (int)(2e6 / (double)N);
    if (reps < 100) reps = 100;
    if (reps > 400000) reps = 400000;
    if (verbose)
        printf("# N=%d split candidates=%d gated=%d trials=%d reps=%d\n",
               N, nc, ngated, trials, reps);

    /* order-rotated timing, reseed per burst (MIGRATED, unchanged) */
    for (int t = 0; t < trials; t++) {
        if (t) _sp_cachebust();
        for (int k = 0; k < nc; k++) {
            vfft_sp_cand_t *c = &cand[(k + t) % nc];
            if (!c->gated) continue;
            _sp_reseed(&b);
            for (int w = 0; w < 10; w++) _sp_run_cand(&b, c);
            _sp_reseed(&b);
            double t0 = _sp_now_ns();
            for (int i = 0; i < reps; i++) _sp_run_cand(&b, c);
            double ns = (_sp_now_ns() - t0) / reps;
            if (ns < c->best) c->best = ns;
        }
    }

    *win_ip = -1; *win_oop = -1;
    for (int k = 0; k < nc; k++) {
        if (!cand[k].gated) continue;
        int *w = VFFT_SP_RAXIS[cand[k].route] ? win_ip : win_oop;
        if (*w < 0 || cand[k].best < cand[*w].best) *w = k;
        if (verbose)
            printf("cand,%d,%s,%d,%d,%.1f\n", N, VFFT_SP_RNAME[cand[k].route],
                   cand[k].R1, cand[k].R2, cand[k].best);
    }

    vfft_proto_aligned_free(b.xr); vfft_proto_aligned_free(b.xi);
    vfft_proto_aligned_free(b.dr); vfft_proto_aligned_free(b.di);
    vfft_proto_aligned_free(b.wr); vfft_proto_aligned_free(b.wi);
    vfft_proto_aligned_free(b.Rr); vfft_proto_aligned_free(b.Ri);
    *np_out = np;
    return nc;
}

static void vfft_sp_dp_release(vfft_oop_plan_t **plans, int np)
{
    for (int i = 0; i < np; i++) vfft_oop_plan_destroy(plans[i]);
}

/* ── B5 decode-gate comparator (production logic; gates are thin drivers).
 * Does the BUILT plan serve exactly what the kind-3 CCOL line banked?
 * Chain: compared stage-for-stage against the decoded cc_chain. Variants:
 * cc_vars must decode against the same nf — create passes the decoded
 * array straight into the column-plan create (a single line of custody),
 * so a successful decode + chain/pair match IS the variants guarantee. */
static int vfft_sp_ccol_line_served(const vfft_oop_wisdom_entry_t *ke,
                                    const vfft_oop_plan_t *p)
{
    int ccf[VFFT_K1_CC_MAX_NF], ccv[VFFT_K1_CC_MAX_NF];
    if (!ke || !p) return 0;
    if (ke->k1_sp_route != VFFT_K1_SP_CCOL) return 0;
    int nf = vfft_k1_cc_chain_decode(ke->cc_chain, ccf);
    if (!nf) return 0;
    if (ke->cc_vars && !vfft_k1_cc_vars_decode(ke->cc_vars, nf, ccv))
        return 0;                         /* banked variants must decode  */
    if (p->kind != VFFT_OOP_KIND_BAILEY2V || !p->colp || !p->cc_perm)
        return 0;                         /* must be a CCOL-shaped plan   */
    if (p->R1 != ke->R1 || p->R2 != ke->R2) return 0;
    if (p->cc_nf != nf) return 0;
    for (int s = 0; s < nf; s++)
        if (p->cc_chain[s] != ccf[s]) return 0;
    return 1;
}

/* ── merge lines emitted by plan_and_bank into <wisdir>/oop_wisdom.txt
 * (MIGRATED from v2 verbatim, incl. the sub-2048 kind-4 wrong-slot filter
 * — see the 2026-08-06 note in the v2 header). Shipped reader/writer only. */
static vfft_oop_wisdom_t _sp_wmain, _sp_wnew; /* large; static, not stack */
static int _sp_merge_bank(const char *main_path, const char *tmp_path)
{
    memset(&_sp_wmain, 0, sizeof _sp_wmain);
    (void)vfft_oop_wisdom_load(&_sp_wmain, main_path);
    memset(&_sp_wnew, 0, sizeof _sp_wnew);
    if (vfft_oop_wisdom_load(&_sp_wnew, tmp_path) != 0) return 0;
    int merged = 0;
    for (int i = 0; i < _sp_wnew.count; i++) {
        if (_sp_wnew.e[i].kind == VFFT_OOP_KIND_ZSPLIT && _sp_wnew.e[i].N < 2048) {
            printf("#   skip sub-2048 kind-4 row (N=%d) — wrong-slot verdict\n",
                   _sp_wnew.e[i].N);
            continue;
        }
        int idx = -1;
        for (int j = 0; j < _sp_wmain.count; j++)
            if (_sp_wmain.e[j].N == _sp_wnew.e[i].N &&
                _sp_wmain.e[j].K == _sp_wnew.e[i].K &&
                _sp_wmain.e[j].kind == _sp_wnew.e[i].kind) { idx = j; break; }
        if (idx < 0) {
            if (_sp_wmain.count >= VFFT_OOP_WISDOM_MAX) continue;
            idx = _sp_wmain.count++;
        }
        _sp_wmain.e[idx] = _sp_wnew.e[i];
        merged++;
    }
    FILE *f = fopen(main_path, "w");
    if (!f) return -1;
    for (int j = 0; j < _sp_wmain.count; j++)
        vfft_oop_wisdom_write_entry(f, &_sp_wmain.e[j]);
    fclose(f);
    return merged;
}

/* ── the whole calibrate-and-record step for one cell ──────────────
 * Split race (this header) + IL race (delegated WHOLE to dp_planner_il)
 * + kind-3/kind-4 banking through the shipped writer. Returns lines
 * merged into <wisdir>/oop_wisdom.txt, or -1 on a poisoned cell. */
static int vfft_sp_dp_plan_and_bank(vfft_il_dp_context_t *ilctx,
                                    const vfft_proto_registry_t *reg,
                                    const char *wisdir, int N, int rigor,
                                    int verbose)
{
    static vfft_sp_cand_t cand[VFFT_SP_MAX_CAND];
    vfft_oop_plan_t *plans[VFFT_SP_MAX_PLANS];
    int np = 0, win_ip = -1, win_oop = -1;
    int nc = vfft_sp_dp_plan(reg, wisdir, N, rigor, cand, plans, &np,
                             &win_ip, &win_oop, verbose);
    if (nc < 0) return -1;

    int spr = -1, sR1 = 0, sR2 = 0, scc = 0, scv = 0;
    if (win_ip >= 0) {
        spr = VFFT_SP_SPMAP[cand[win_ip].route];
        sR1 = cand[win_ip].R1; sR2 = cand[win_ip].R2;
        scc = cand[win_ip].cc_code;
        scv = cand[win_ip].cc_vcode;
        if (cand[win_ip].route == VFFT_SP_R_MONO_ALT) { sR1 = 8; sR2 = N / 8; }
        if (verbose)
            printf("# N=%d SPLIT-IP winner: %s %dx%d %.1f ns"
                   "  (split-oop winner: %s %.1f ns)\n",
                   N, VFFT_SP_RNAME[cand[win_ip].route], sR1, sR2,
                   cand[win_ip].best,
                   win_oop >= 0 ? VFFT_SP_RNAME[cand[win_oop].route] : "-",
                   win_oop >= 0 ? cand[win_oop].best : 0.0);
    } else if (verbose)
        printf("# N=%d NO gated split candidate — kind-3 line will be "
               "SKIPPED (sp_route<0 refusal, not zero-filled)\n", N);

    char wpath[600], tpath[600];
    snprintf(wpath, sizeof wpath, "%s/oop_wisdom.txt", wisdir);
    snprintf(tpath, sizeof tpath, "%s/k1_bank_tmp.txt", wisdir);

    int merged = 0;
    FILE *tf = fopen(tpath, "w");
    if (tf) {
        double sp_ns = (win_ip >= 0) ? cand[win_ip].best : 1e18;
        int lines = vfft_il_dp_plan_and_bank(ilctx, tf, N, spr, sR1, sR2, scc,
                                             scv, sp_ns, verbose);
        fclose(tf);
        merged = _sp_merge_bank(wpath, tpath);
        if (verbose)
            printf("# N=%d banked %d line(s) (merged %d into %s)\n",
                   N, lines, merged, wpath);
    }
    vfft_sp_dp_release(plans, np);
    return merged;
}

#endif /* VFFT_DP_PLANNER_SPLIT_OOP_H */
