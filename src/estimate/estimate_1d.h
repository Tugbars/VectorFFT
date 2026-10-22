/* estimate_1d.h -- the 1D estimate's REAL-TRANSFORM formulas (r2c / c2r), on top of the library's c2c
 * formulas (src/core/il/planning/estimate_il.h), and the recipe printer. Git-ignored, unreleased.
 *
 * Implements est_plan_any and est_recipe_print (the contracts are in estimate_recipe.h) as the sketch
 * docs/design/estimate_1d_sketch.md specifies them (section 6 real, section 7 threads). The c2c
 * procedure, its helpers and the code facts it reads are the LIBRARY's: est_c2c, est_fs_child,
 * est_ztt_ok, est_inner_pair, est_blue_inner, est_inner_ztt, est_inner_fs, est_inner_mt, est_fs_split,
 * est_odd_stages, est_merge10, est_mincounts ... (estimate_il.h). This file adds only what the real
 * transforms need. Sections cited as "section n" are the sketch's; library behaviour is cited
 * file:line, relative to src/core/.
 *
 * PURE: reads only the machine and the request. No clock, no store, no allocation, no I/O.
 *
 * The sketch's open decisions (section 12) take these defaults where the real rules meet them:
 *   1  one transform only (howmany = 1).
 *   2  every T > 1 takes the T = 8 fitted rules; at T != 8 they are UNFITTED.
 *   7  the 8-lane rules: the library's formulas follow the build's lanes (est_plan_1d refuses a
 *      machine of another count), so an 8-lane recipe is planned by an AVX-512 build only.
 * Gaps of the text, reported and not filled: zfsr at T > 1 is refused (section 7 has no zfsr row);
 * the zr2c prime child's serial bound reads at its M as section 7 words it. One compromise for the
 * owner to confirm: zrf's LEVELS arm only where the library can bind it (est_real). */
#ifndef EST_1D_H
#define EST_1D_H

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "estimate_recipe.h"

/* ================================================================ real code facts */

#define EST_ZFSR_TB 16   /* VFFT_ZFSR_TB (il/real/zfsr.h:57): N1 % 16, N2 % 32 (zfsr.h:100-101) */

/* the real mono: VFFT_IL_RN1_PAIR_RADICES (il_registry_avx2.h:613) plus 32 (rn1b48) and 64
 * (rn1b444), N <= VFFT_ZRM_MAX_N = 64 (il/real/zrm.h:39,42-57); the same at avx512 */
static const int est_rn1_list[] = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21,
                                    22, 23, 25, 26, 27, 29, 31, 32, 37, 41, 43, 47, 53, 59, 61, 64 };
/* the real pair's radices: VFFT_IL_T2M / R2Z / T2H lists at avx2 (il_registry_avx2.h:525,1104,
 * 1125); EMPTY at avx512 (il_registry_avx512.h:477,1024,1029) */
static const int est_zrp_list[] = { 4, 6, 8, 10, 12, 16, 32, 64 };
/* the zrf tile budgets (il/real/odd_build.h:236) */
static const int est_zrf_budgets[] = { 64, 128, 256, 512, 1024, 2048 };

/* ================================================================ real (section 6) */

/* ZTT-r's grammar at M = N/2 (zttr.h:113-143,180): 2^a times a product over {15, 9, 7, 5, 3},
 * a >= 5 (the terminator run L = M / R_last >= 8 with L % 8 == 0 needs 32 | M), 32 <= M <= 262144 */
static inline int est_zttr_grammar(long long M)
{
    long long o;
    int i;
    if (M < 32 || M > VFFT_ZTT_MAX_N || est_v2(M) < 5) return 0;
    o = M >> est_v2(M);
    for (i = 0; i < EST_NELEM(est_odd_mids); i++)
        while (o % est_odd_mids[i] == 0) o /= est_odd_mids[i];
    return o == 1;
}
/* section 6 even rule 4, the zttr chain at M = N/2. Power-of-two M: c2r 4, then 8s, then 4s; r2c 8,
 * the 4s, then 8s, closing 4 (with no forced 4, two 8s become three 4s). 2^a odd M: the first pow2
 * stage, the odd primes 7, 5, 3, the other pow2 stages, closing 4. Tile: untiled below M = 4096,
 * else the first legal width of 2048, 3072, 1024, 512 (zrp_build.h:445-458: tile < M, M % tile
 * == 0, a multiple of R0 R1). Validated by ZTT's natural laws and zttr.h:173-180. 0 = none. */
static inline int est_zttr_chain(long long N, int c2r, int *c, size_t *tile)
{
    const long long M = N / 2;
    const int a = est_v2(M);
    long long o = M >> a;
    int k = 0, e, f, i;
    static const int widths[4] = { 2048, 3072, 1024, 512 };
    if (o == 1)
    {
        est_mincounts(a, &e, &f);
        if (f == 0 && e >= 2) { e -= 2; f += 3; }
        if (f < 1 || (!c2r && e < 1)) return 0;
        if (c2r)
        {
            c[k++] = 4;
            for (i = 0; i < e; i++) c[k++] = 8;
            for (i = 0; i < f - 1; i++) c[k++] = 4;
        }
        else
        {
            c[k++] = 8;
            for (i = 0; i < f - 1; i++) c[k++] = 4;
            for (i = 0; i < e - 1; i++) c[k++] = 8;
            c[k++] = 4;
        }
    }
    else
    {
        int mids[16], nm = 0, p2[16], np = 0;
        const int r = a - 2;
        static const int op[3] = { 7, 5, 3 };
        for (i = 0; i < 3; i++)
            while (o % op[i] == 0 && nm < 16) { mids[nm++] = op[i]; o /= op[i]; }
        if (o != 1) return 0;
        if (r % 3 == 0) { for (i = 0; i < r / 3; i++) p2[np++] = 8; }
        else if (r % 3 == 2) { p2[np++] = 4; for (i = 0; i < r / 3; i++) p2[np++] = 8; }
        else { if (r < 4) return 0; p2[np++] = 4; p2[np++] = 4; for (i = 0; i < (r - 4) / 3; i++) p2[np++] = 8; }
        if (!np || np + nm + 1 > 16) return 0;
        c[k++] = p2[0];
        for (i = 0; i < nm; i++) c[k++] = mids[i];
        for (i = 1; i < np; i++) c[k++] = p2[i];
        c[k++] = 4;
    }
    if (k > VFFT_ZTT_MAX_NF || !est_ztt_ok(M, c, k, 0)) return 0;
    if ((c[k - 1] != 4 && c[k - 1] != 8) || M / c[k - 1] < 8 || (M / c[k - 1]) % 8) return 0;
    *tile = 0;
    if (M >= 4096)
        for (i = 0; i < 4; i++)
            if (widths[i] < M && M % widths[i] == 0 && widths[i] % ((long long)c[0] * c[1]) == 0)
            {
                *tile = (size_t)widths[i];
                break;
            }
    return k;
}
/* section 6 zrf's chain: the prime factors, 3.3 -> 9, ascending; 2 <= K <= 10 (zrf.h:228), the
 * section 4.4 merges above 10 stages, re-sorted ascending (29296875 = 3.5.5.5.5.5.5.5.5.25). 0 = none. */
static inline int est_zrf_chain(long long N, int *c)
{
    int k = est_odd_stages(N, c);
    if (k < 2) return 0;
    est_sort_asc(c, k);
    if (k > VFFT_ILFD_MAX_K)
    {
        est_sort_desc(c, k);
        k = est_merge10(c, k);
        if (k < 2) return 0;
        est_sort_asc(c, k);
    }
    return k;
}
/* section 6 zrb's inner at M: M < 2048 a pair, then the il3p seed; otherwise as section 4.6 -- a pow2
 * M takes Bluestein's (the convolution ZTURN-T to 262144, the four-step above), any other M ZTURN-T's
 * odd chain in the odd band, else the four-step in its band. 1 / 0 none / -1 refused. */
static inline int est_zrb_inner(const est_machine_t *m, long long M, est_inner_t *in, char *fs, size_t cap,
                                char *why, size_t whylen)
{
    int c[3];
    if (M < 2048)
    {
        if (est_inner_pair(M, &c[0], &c[1])) { est_inner_set(in, "2p", c, 2, 0, 0); return 1; }
        if (vfft_il3p_default_chain((int)M, &c[0], &c[1], &c[2])) { est_inner_set(in, "3p", c, 3, 0, 0); return 1; }
        return 0;
    }
    if (est_is_pow2(M)) return est_blue_inner(m, M, in, fs, cap, why, whylen);
    if (est_inner_ztt(m, M, in)) return 1;
    return est_inner_fs(m, M, in, fs, cap, why, whylen);
}
/* section 6 zrb's length m: the smallest 2^a {1, 3, 5, 7, 9, 15} >= N + (N-1)/2 (vfft_zrb_min_m,
 * zrb.h:59) that an inner builds -- 2^a q with a >= 2 and the power of two >= 16, the code's
 * candidate shapes (odd_build.h:368-376) -- up to 2^24 */
static inline int est_zrb(const est_machine_t *m, long long N, est_any_recipe_t *o, char *why, size_t whylen)
{
    static const int q[6] = { 1, 3, 5, 7, 9, 15 };
    const long long mn = N + (N - 1) / 2;
    long long cand[200], M;
    int nc = 0, i, j, a, r;
    for (i = 0; i < 6; i++)
        for (a = 2; a <= 24; a++)
        {
            M = (long long)q[i] << a;
            if (M > VFFT_K1FS_MAX_N) break;
            if (M >= mn && (q[i] != 1 || M >= 16) && nc < 200) cand[nc++] = M;
        }
    for (i = 1; i < nc; i++)
        for (j = i; j > 0 && cand[j] < cand[j - 1]; j--) { const long long x = cand[j]; cand[j] = cand[j - 1]; cand[j - 1] = x; }
    for (i = 0; i < nc; i++)
    {
        est_inner_t in;
        memset(&in, 0, sizeof in);
        o->c.fs[0] = 0;
        r = est_zrb_inner(m, cand[i], &in, o->c.fs, sizeof o->c.fs, why, whylen);
        if (r < 0) return -1;
        if (r == 0) continue;
        snprintf(o->eng, sizeof o->eng, "zrb");
        o->zrb.M = (int)cand[i];
        snprintf(o->zrb.kind, sizeof o->zrb.kind, "%s", in.kind);
        snprintf(o->zrb.shape, sizeof o->zrb.shape, "%s", in.shape);
        o->zrb.tw = in.tw;
        o->zrb.mt = 0;   /* section 7: zrb has no threaded row */
        return 0;
    }
    return est_refuse(why, whylen, "real N = %lld: no zrb length up to 2^24 has an inner (sketch 6)", N);
}

/* section 6: the door picks by parity (il/real/real_create_il.h:34); a real N plans like complex N/2 */
static inline int est_real(const est_machine_t *m, long long N, int c2r, int ip, int T, est_any_recipe_t *o,
                           char *why, size_t whylen)
{
    int c[32], k, i;
    memset(o, 0, sizeof *o);
    if (N < 2) return est_refuse(why, whylen, "real N = %lld < 2 (sketch 6: the odd door admits N >= 3, odd_build.h:50-54)", N);
    if (N % 2)
    {   /* odd N */
        if (est_in(est_rn1_list, EST_NELEM(est_rn1_list), N) && (N <= 17 || vfft_policy_is_prime((int)N)))
        {
            snprintf(o->eng, sizeof o->eng, "zrm");
            return 0;
        }
        if (vfft_policy_prime_cell((int)N))
        {   /* zrb, refused when (3N - 1)/2 > 2^24 (_zrb_ok, zrp_build.h:260-264) */
            if ((3 * N - 1) / 2 > VFFT_K1FS_MAX_N)
                return est_refuse(why, whylen, "real N = %lld: zrb's (3N - 1)/2 passes 2^24 (sketch 6)", N);
            return est_zrb(m, N, o, why, whylen);
        }
        k = est_zrf_chain(N, c);
        if (!k) return est_refuse(why, whylen, "real N = %lld: no zrf chain of 2..10 stages (sketch 6)", N);
        snprintf(o->eng, sizeof o->eng, "zrf");
        memcpy(o->zrf.chain, c, sizeof(int) * (size_t)k);
        o->zrf.K = k;
        o->zrf.nomsz = 0;   /* the split body off (row msz=0) when a middle stage is in {3, 5, 7, 9, 15} */
        for (i = 1; i < k - 1; i++)
            if (c[i] == 3 || c[i] == 5 || c[i] == 7 || c[i] == 9 || c[i] == 15) o->zrf.nomsz = 1;
        o->zrf.tile = 0;   /* 0 while 8N <= L1d, else the widest budget with 16 b <= L1d/2 */
        if (8 * N > (long long)m->l1d)
            for (i = 0; i < EST_NELEM(est_zrf_budgets); i++)
                if (16LL * est_zrf_budgets[i] <= m->l1d / 2) o->zrf.tile = est_zrf_budgets[i];
        /* section 7: LEVELS above the serial bound -- where it binds. vfft_zrf_mt_bind declines LEVELS
         * unless the first level is tiled with a tile per worker (_zrf_mt_level_ok,
         * zrf_mt.h:133-156), and a race banks only an arm that bound (the stored T = 8 zrf rows
         * carry mt=2, mt=1 or no mt).
         * The first level's tile is the widest stage span R_s D_s (s in 1..K-2, D_s > 4, not a
         * tail stage) within the budget (zrf.h:255-262); its tiles are h D / tw with D = N / R_0,
         * h = R_0 / 2 (zrf.h:329). Where LEVELS would decline the row stays serial.
         * A COMPROMISE FOR THE OWNER TO CONFIRM: section 7 says mt=2 above the bound with no test on T;
         * written literally, a cell where LEVELS cannot bind gets a silently serial plan, which the
         * build refuses (EST_SERIAL). The T in the test is the library's own bind law
         * (_zrf_mt_level_ok, zrf_mt.h:133-136), not a fitted rule (TEST_LOG.md bug 3). */
        if (T > 1 && !est_serial(m, N / 2) && o->zrf.tile > 0)
        {
            const long long D = N / c[0], h = c[0] / 2;
            long long tw0 = 0;
            int s;
            for (s = 1; s <= k - 2 && !tw0; s++)
            {
                const long long Ds = est_prod(c + s + 1, k - s - 1), span = (long long)c[s] * Ds;
                if (Ds > 4 && span <= o->zrf.tile) tw0 = span;
            }
            if (tw0 > 0 && h * D / tw0 >= T) o->zrf.mt = 2;
        }
        return 0;
    }
    /* even N */
    if (N / 2 > VFFT_K1FS_MAX_N) return est_refuse(why, whylen, "real N = %lld: N/2 > 2^24 (sketch 6)", N);
    /* 1. zrm */
    if (N <= 16 && est_in(est_rn1_list, EST_NELEM(est_rn1_list), N))
    {
        snprintf(o->eng, sizeof o->eng, "zrm");
        return 0;
    }
    /* 2. zrp, 4 lanes only, N <= 160 (table 5.8), an even x even pair: the most balanced, smaller
     * factor first, form B (the r2z leaf) */
    if (VFFT_IL_VW == 4 && N <= 160)
    {
        int lo = 0, hi = 0;
        for (i = 0; i < EST_NELEM(est_zrp_list); i++)
        {
            const int a = est_zrp_list[i];
            if (N % a || !est_in(est_zrp_list, EST_NELEM(est_zrp_list), N / a)) continue;
            {
                const int b = (int)(N / a), l = a < b ? a : b, h = a < b ? b : a;
                if (!hi || h < hi) { lo = l; hi = h; }
            }
        }
        if (hi)
        {
            snprintf(o->eng, sizeof o->eng, "zrp");
            o->zrp.R1 = lo;
            o->zrp.R2 = hi;
            o->zrp.form = 1;
            return 0;
        }
    }
    /* 3. zfsr: N/2 a power of two in 2^19..2^24; the split of section 4.5 at N/2 and its child as fs_*.
     * At T > 1 the sketch has no rule for zfsr: section 7's table lists no zfsr row, and section 11
     * says only that estimate does not switch engines by T. A race at T picks the split at T
     * (_zfsr_sweep over vfft_k1fs_splits, zrp_build.h:518-530) and seeds the 2D child under T
     * (_k1fs_child_keys(n1, n2, 0, T), :316-318), so neither the split nor the child's T > 1
     * tokens follow from the text. The gap is reported, not filled: the cell is refused. */
    if (est_is_pow2(N / 2) && N / 2 >= (1LL << 19))
    {
        int n1, n2;
        if (T > 1)
            return est_refuse(why, whylen, "real N = %lld at T = %d: the sketch has no threaded rule for zfsr (section 7 "
                              "lists none; its race picks the split and the 2D child at T, zrp_build.h:316-318, "
                              "518-530) -- a gap reported, not filled", N, T);
        if (!est_fs_split(N / 2, 0, &n1, &n2) || n1 % EST_ZFSR_TB || n2 % (2 * EST_ZFSR_TB))
            return est_refuse(why, whylen, "real N = %lld: no zfsr split (sketch 6)", N);
        snprintf(o->eng, sizeof o->eng, "zfsr");
        o->zfsr.n1 = n1;
        o->zfsr.n2 = n2;
        return est_fs_child(m, n1, n2, 1, o->c.fs, sizeof o->c.fs, why, whylen);
    }
    /* 4. zttr, 4 lanes only: a pow2 N in 512..524288, or N >= 2048 with N/2 in ZTT-r's grammar */
    if (VFFT_IL_VW == 4 && ((est_is_pow2(N) && N >= 512 && N <= 524288) ||
                            (!est_is_pow2(N) && N >= 2048 && est_zttr_grammar(N / 2))))
    {
        size_t tile = 0;
        k = est_zttr_chain(N, c2r, c, &tile);
        if (k)
        {
            snprintf(o->eng, sizeof o->eng, "zttr");
            memcpy(o->zttr.chain, c, sizeof(int) * (size_t)k);
            o->zttr.nf = k;
            o->zttr.tile = tile;
            o->zttr.stk = 3;
            o->zttr.mt = 0;   /* section 7: zttr has no threaded row */
            return 0;
        }
    }
    /* 5. zr2c: the child is the library's c2c procedure at N/2, natural, one thread; route
     * child_nat_ip in place and for c2r from N = 1024 (table 5.8), else child_oop_il */
    {
        const int route = ip || (c2r && N >= 1024);
        if (est_c2c(m, N / 2, 0, route, 1, &o->c, why, whylen) != 0) return -1;
        snprintf(o->eng, sizeof o->eng, "zr2c");
        o->zr2c.route = route;
        if (c2r && o->c.k1_il_route == VFFT_K1_IL_2P_PURE)
        {   /* the c2r pair child mirrored, leaf and mid swapped, both forms recomputed for the
             * swapped slots (a c2r child runs only its backward forms, wisdom2_real_il.h:382) */
            const int t = o->c.il_R1;
            o->c.il_R1 = o->c.il_R2;
            o->c.il_R2 = t;
            o->c.il_kv = est_nib_fwd(o->c.il_R1, 0) | est_nib_fwd(o->c.il_R2, 1) << 4;
            o->c.il_bkv = est_nib_bwd(o->c.il_R1, 0) | est_nib_bwd(o->c.il_R2, 1) << 4;
        }
        o->c.il_mt = 0;   /* the child is one-thread (zr2c_build.h:38-41) ... */
        o->c.il_mt_tw = 0;
        if (T > 1 && N >= 64 && !est_serial(m, N / 2))
        {   /* section 7: fold=mt (zrp_build.h:795-818); a prime child also takes its own arm at its M */
            o->zr2c.fold_mt = 1;
            if (o->c.k1_il_route == VFFT_K1_IL_PRIME)
            {
                est_inner_t in;
                long long M = 1;
                memset(&in, 0, sizeof in);
                in.tw = o->c.prime.in_tw;
                in.ztt = !strcmp(o->c.prime.in_kind, "ztt");
                {   /* M = the inner shape's product (_ilprime_desc_len, k1_commit.h:488-496) */
                    const char *p = o->c.prime.in_shape;
                    long long v = 0;
                    for (;; p++)
                    {
                        if (*p >= '0' && *p <= '9') { v = 10 * v + (*p - '0'); continue; }
                        M *= v;
                        v = 0;
                        if (!*p) break;
                    }
                }
                /* the bound at the child's M (section 7's zr2c row: "the prime-inner rule above at the
                 * child's M"); OPEN for the owner -- the c2c cell reads it at N. At L1d = 48 KiB the two
                 * never differ for a zr2c child. */
                o->c.il_mt = est_inner_mt(m, &in, M, M, T);
            }
        }
        return 0;
    }
}

/* ================================================================ the contracts */

static inline int est_plan_any(const est_machine_t *m, const est_request_t *rq, est_any_recipe_t *out, char *why,
                               size_t whylen)
{
    const int T = rq && rq->nthreads > 1 ? rq->nthreads : 1;   /* [open decision 2, default] */
    char dummy[1];
    if (!why || !whylen) { why = dummy; whylen = sizeof dummy; }
    why[0] = 0;
    if (!m || !rq || !out) return est_refuse(why, whylen, "missing argument");
    memset(out, 0, sizeof *out);
    if (rq->transform == VFFT_C2C)
    {   /* the library's formulas */
        snprintf(out->eng, sizeof out->eng, "k1");
        return est_plan_1d(m, rq, &out->c, why, whylen);
    }
    if (rq->transform != VFFT_R2C && rq->transform != VFFT_C2R)
        return est_refuse(why, whylen, "transform %d is not c2c (0), r2c (1) or c2r (2)", rq->transform);
    /* the library's machine and request checks, as est_plan_1d makes them */
    if (m->lanes != VFFT_IL_VW)
        return est_refuse(why, whylen, "a machine of %d lanes: the formulas follow this build's %d", m->lanes, VFFT_IL_VW);
    if (m->l1d < 1024) return est_refuse(why, whylen, "L1d %ld bytes is not a cache size", m->l1d);
    if (rq->nthreads < 0) return est_refuse(why, whylen, "nthreads %d < 0", rq->nthreads);
    if (rq->placement != VFFT_PLACEMENT_OUT && rq->placement != VFFT_PLACEMENT_IN)
        return est_refuse(why, whylen, "placement %d is not out (0) or in (1)", rq->placement);
    if (rq->order != VFFT_ORDER_NATURAL)
        return est_refuse(why, whylen, "a real plan is natural order only (vfft.c:1940-1957)");
    return est_real(m, rq->N, rq->transform == VFFT_C2R, rq->placement == VFFT_PLACEMENT_IN, T, out, why, whylen);
}

/* the c2c route row: the key, then the route's tokens (the library's est_c2c_tokens), the T row's
 * threaded tokens (il_mt il_mt_tw il_mtsb, set with vw2_update_field after the bank) and the
 * four-step's fs_* */
static inline void est_print_c2c(est_sb_t *s, const est_recipe_t *rc, const est_request_t *rq, int T)
{
    const char *pl = rq->placement == VFFT_PLACEMENT_IN ? "ip" : "oop";
    if (rc->k1_il_route == VFFT_K1_IL_PRIME)
    {   /* the method row first: vw2_prime_method_bank's key and tokens (wisdom2_oop_il.h:545-596) */
        est_sb_add(s, "t=c2c n=%d q=1 ord=scr place=ip role=comp lay=il | eng=%s in=%s in_sh=%s in_tw=%d",
                   rq->N, rc->prime.method == 1 ? "rader" : "bluestein", rc->prime.in_kind, rc->prime.in_shape,
                   rc->prime.in_tw);
        if (!strcmp(rc->prime.in_kind, "fs") && rc->fs[0]) est_sb_add(s, " %s", rc->fs);
        est_sb_add(s, "\n");
    }
    est_sb_add(s, "t=c2c n=%d q=1 ord=%s place=%s role=comp lay=il", rq->N,
               rq->order == VFFT_ORDER_SCRAMBLED ? "scr" : "nat", pl);
    if (T > 1) est_sb_add(s, " nthreads=%d", T);
    est_sb_add(s, " | ");
    est_c2c_tokens(s, rc, "");
    if (T > 1)
    {
        if (rc->k1_il_route == VFFT_K1_IL_FLAT || rc->k1_il_route == VFFT_K1_IL_ZTT || rc->k1_il_route == VFFT_K1_IL_PRIME)
        {
            est_sb_add(s, " il_mt=%d", rc->il_mt);
            if (rc->k1_il_route == VFFT_K1_IL_FLAT && rc->il_mt == 2) est_sb_add(s, " il_mt_tw=%d", rc->il_mt_tw);
        }
        if (rc->k1_il_route == VFFT_K1_IL_FS)
        {
            est_sb_add(s, " il_mt=%d il_mtsb=", rc->il_mt);
            if (rc->il_zt_n) est_sb_chain(s, rc->il_zt, rc->il_zt_n);
            else est_sb_add(s, "0");
        }
    }
    if (rc->k1_il_route == VFFT_K1_IL_FS && rc->fs[0]) est_sb_add(s, " %s", rc->fs);
    est_sb_add(s, "\n");
    if ((rc->k1_il_route == VFFT_K1_IL_2P_PURE || rc->k1_il_route == VFFT_K1_IL_CHAIN3) && rc->il_bkv)
    {   /* the dir=bwd sibling, keyed ord=nat at the request's placement (dp_planner_il.h:2266-2290) */
        est_sb_add(s, "t=c2c n=%d q=1 ord=nat place=%s dir=bwd role=comp lay=il | ", rq->N, pl);
        est_bwd_tokens(s, rc, "");
        est_sb_add(s, "\n");
    }
}

static inline int est_recipe_print(const est_any_recipe_t *rc, const est_request_t *rq, char *buf, size_t cap)
{
    const int T = rq->nthreads > 1 ? rq->nthreads : 1;
    const est_recipe_t *ch = &rc->c;
    est_sb_t s;
    est_sb_init(&s, buf, cap);
    if (rq->transform == VFFT_C2C)
        est_print_c2c(&s, ch, rq, T);
    else
    {   /* the real row: vw2_real_il_key (wisdom2_real_il.h:60-68) and the engine's tokens
         * (wisdom2_real_il.h:1-44) */
        est_sb_add(&s, "t=%s n=%d q=1 ord=nat place=%s", rq->transform == VFFT_C2R ? "c2r" : "r2c", rq->N,
                   rq->placement == VFFT_PLACEMENT_IN ? "ip" : "oop");
        if (T > 1) est_sb_add(&s, " nthreads=%d", T);
        est_sb_add(&s, " | eng=%s", rc->eng);
        if (!strcmp(rc->eng, "zrp"))
            est_sb_add(&s, " pair=%d.%d leaf=%s", rc->zrp.R1, rc->zrp.R2, rc->zrp.form ? "r2z" : "n1t");
        else if (!strcmp(rc->eng, "zttr"))
        {
            est_sb_add(&s, " chain=");
            est_sb_chain(&s, rc->zttr.chain, rc->zttr.nf);
            est_sb_add(&s, " tile=%u stk=%d", (unsigned)rc->zttr.tile, rc->zttr.stk);
            if (rc->zttr.mt) est_sb_add(&s, " mt=%d", rc->zttr.mt);
        }
        else if (!strcmp(rc->eng, "zfsr"))
            est_sb_add(&s, " split=%dx%d %s", rc->zfsr.n1, rc->zfsr.n2, ch->fs);
        else if (!strcmp(rc->eng, "zrf"))
        {
            est_sb_add(&s, " chain=");
            est_sb_chain(&s, rc->zrf.chain, rc->zrf.K);
            est_sb_add(&s, " msz=%d tile=%d", rc->zrf.nomsz ? 0 : 1, rc->zrf.tile);
            if (rc->zrf.mt) est_sb_add(&s, " mt=%d", rc->zrf.mt);
        }
        else if (!strcmp(rc->eng, "zrb"))
        {
            est_sb_add(&s, " m=%d in=%s in_sh=%s in_tw=%d", rc->zrb.M, rc->zrb.kind, rc->zrb.shape, rc->zrb.tw);
            if (rc->zrb.mt) est_sb_add(&s, " mt=%d", rc->zrb.mt);
            if (!strcmp(rc->zrb.kind, "fs") && ch->fs[0]) est_sb_add(&s, " %s", ch->fs);
        }
        else if (!strcmp(rc->eng, "zr2c"))
        {   /* vw2_real_il_bank_zr2c's order (wisdom2_real_il.h:422-470) */
            est_sb_add(&s, " route=%s", rc->zr2c.route ? "child_nat_ip" : "child_oop_il");
            if (T > 1) est_sb_add(&s, " fold=%s", rc->zr2c.fold_mt ? "mt" : "st");
            est_sb_add(&s, " il_route=%s", vw2_oop_il_name[ch->k1_il_route]);
            if (ch->il_R1 || ch->il_R2) est_sb_add(&s, " il_pair=%d.%d", ch->il_R1, ch->il_R2);
            if (ch->k1_il_route == VFFT_K1_IL_CHAIN3) est_sb_add(&s, " il_chain=%d.%d.%d", ch->il_c3[0], ch->il_c3[1], ch->il_c3[2]);
            if (ch->k1_il_route == VFFT_K1_IL_FLAT)
            {
                est_sb_add(&s, " il_flat=");
                est_sb_chain(&s, ch->il_fl, ch->il_fl_n);
                if (ch->il_flf[0]) est_sb_add(&s, " il_forms=%s", ch->il_flf);
            }
            if (ch->k1_il_route == VFFT_K1_IL_ZTT || (ch->k1_il_route == VFFT_K1_IL_FS && ch->il_zt_n >= 2))
            {
                est_sb_add(&s, " %s=", ch->k1_il_route == VFFT_K1_IL_ZTT ? "il_ztt" : "il_sb");
                est_sb_chain(&s, ch->il_zt, ch->il_zt_n);
            }
            if (ch->il_tw > 0) est_sb_add(&s, " il_tw=%d", ch->il_tw);
            if (ch->k1_il_route == VFFT_K1_IL_PRIME)
            {
                est_sb_add(&s, " il_prime=%s il_prime_in=%s il_prime_sh=%s il_prime_tw=%d",
                           ch->prime.method == 1 ? "rader" : "bluestein", ch->prime.in_kind, ch->prime.in_shape,
                           ch->prime.in_tw);
                if (ch->il_mt > 0) est_sb_add(&s, " il_mt=%d", ch->il_mt);
            }
            if (ch->fs[0] && (ch->k1_il_route == VFFT_K1_IL_FS || ch->k1_il_route == VFFT_K1_IL_PRIME)) est_sb_add(&s, " %s", ch->fs);
            est_sb_add(&s, " il_kv=%d il_bkv=%d", ch->il_kv, ch->il_bkv);
        }
        est_sb_add(&s, "\n");
    }
    if (buf && cap) buf[s.len < cap ? s.len : cap - 1] = 0;
    return (int)s.len;
}

#endif /* EST_1D_H */
