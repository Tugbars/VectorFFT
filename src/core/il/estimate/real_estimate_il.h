/* real_estimate_il.h — the ESTIMATE create of a 1D REAL transform (VFFT_FLAG_ESTIMATE on R2C or C2R, rank 1,
 * one transform): the real door's row written from the formulas and replayed by the door with nothing measured
 * (docs/roadmap/real_estimate_plan.md, stage 1; the formulas are docs/design/estimate_1d_sketch.md sections 6
 * and 7). Both directions live in this file (the owner, 2026-10-28 and 2026-11-01: the real transforms have
 * their own file).
 *
 * THE DOOR (il/real/real_create_il.h): even N through _real_il_build (il/real/zrp_build.h), odd N through
 * _real_il_odd_build (il/real/odd_build.h); each looks its row up first and races only on a miss. The row is
 * the direction's own, t=r2c|c2r n=N q=1 ord=nat place=oop|ip [nthreads=T] (vw2_real_il_key,
 * il/wisdom/wisdom2_real_il.h:60), payload eng= and the engine's tokens: zrm; zrp pair= leaf=; zttr chain=
 * tile= stk= [mt=]; zr2c route= [fold=] and the half's c2c recipe; zfsr split= and the four-step child fs_*;
 * zrf chain= msz= tile= [mt=]; zrb m= in= in_sh= in_tw= [mt=].
 *
 * THE RECIPE (est_plan_real -> est_real): the engine by the sketch's section 6, the threaded arms by its
 * section 7 (fitted at T = 8). The zr2c half is the c2c estimate at N/2 (est_c2c), so the c2c rules reach it.
 * The sketch's open decisions (its section 12) take these defaults: every T > 1 takes the rules fitted at
 * T = 8; the formulas follow the build's lanes (an 8-lane recipe is planned by an AVX-512 build only).
 * Gaps of the sketch, reported and not filled: zfsr at T > 1 is refused (section 7 has no rule for it); the
 * zr2c prime child's serial bound reads at its M as section 7 words it; zrf's LEVELS arm only where the
 * library can bind it (est_real).
 *
 * THE BUILD (est_build_real) takes est_build_c2c's steps: the pre-flight (the door's real builders at one
 * thread against an empty store, then destroyed; a nested four-step checked by est__fs_preflight), the row
 * into a store held in memory (the real writers vw2_real_il_bank_*), the door's replay (est__replay and its
 * detectors), and the plan the door built checked against the recipe (est__check_plan_real): the engine, every
 * threaded arm the recipe names, and the lean door's entry (k1_go) in the transform's direction, so an estimate
 * plan is entered by the same jump as the planner's (the owner, 2026-11-01: no call may leak into execution).
 * An estimate create never hands back a plan that measured.
 *
 * POSITION IN vfft.c: included by c2c_estimate_il.h after the lane-major file; it reuses that file's pieces
 * (est__replay, est__meter_read, est__fs_preflight, est__fs_text, est__put_fs ...). The real doors and their
 * builders are defined above (vfft.c includes the il/real/ headers before vfft_api.h). */
#ifndef VFFT_IL_REAL_ESTIMATE_IL_H
#define VFFT_IL_REAL_ESTIMATE_IL_H


/* ── the real code facts ─────────────────────────────────────────────────── */

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

/* ── the recipe ─────────────────────────────────────────────────────────── */

/* The recipe of a real request: the engine, the c2c recipe as member c (the zr2c half at N/2, natural, one
 * thread; zfsr's split child and zrb's four-step inner in c.fs) and the real engines' fields. The prototype's
 * tools (src/estimate/) name it est_any_recipe_t and put a c2c recipe in c with eng "k1". */
typedef struct est_real_recipe_s
{
    char eng[8];          /* [eng] the row's engine: "k1" for c2c; zrm | zrp | zttr | zr2c | zfsr | zrf | zrb */
    est_recipe_t c;       /* c2c: the recipe (est_plan_1d). zr2c: the child at N/2, natural, T = 1, mapped
                           * 1:1 onto vw2_zr2c_child_t (wisdom2_real_il.h:388-406; c.il_mt = the prime
                           * child's threaded arm). zfsr / zrb kind 4: c.fs holds the four-step child. */

    /* ---- real: one row per (r2c|c2r, N, placement, T), ord=nat (wisdom2_real_il.h:1-44) ------- */
    struct { int R1, R2, form; } zrp;    /* [pair=R1.R2 leaf=n1t|r2z] form 0 A (n1t), 1 B (r2z);
                                          * 4 lanes only (sketch 6)                             */
    struct { int chain[8], nf; size_t tile; int stk, mt; } zttr;
                                         /* [chain= tile= stk= mt=] the chain at M = N/2, 2..8
                                          * stages; stk = 3; mt 0 = absent; 4 lanes only         */
    struct { int route, fold_mt; } zr2c; /* [route=child_oop_il|child_nat_ip fold=st|mt] route =
                                          * the CHILD's placement, 0 child_oop_il, 1 child_nat_ip;
                                          * fold_mt only on a T > 1 row                          */
    struct { int n1, n2; } zfsr;         /* [split=n1xn2] the split of N/2; its child = c.fs      */
    struct { int chain[10], K, nomsz, tile, mt; } zrf;
                                         /* [chain= msz= tile= mt=] odd radices ascending, 2..10
                                          * (zrf.h:228); nomsz = !msz; tile = the budget in
                                          * complex, 0 = untiled; mt 2 LEVELS on a T > 1 row     */
    struct { int M; char kind[8], shape[64]; int tw, mt; } zrb;
                                         /* [m= in= in_sh= in_tw= mt=] the convolution length and
                                          * its inner, the prime row's spelling; a kind-4 inner's
                                          * child goes in c.fs; mt 0 = serial                    */
} est_real_recipe_t;

/* ── the formulas (the sketch's section 6 real, section 7 threads) ──────── */

/* ZTT-r's grammar at M = N/2 (zttr.h:113-143,180): 2^a times a product over {15, 9, 7, 5, 3},
 * a >= 5 (the terminator run L = M / R_last >= 8 with L % 8 == 0 needs 32 | M), 32 <= M <= 262144 */
static inline int est_zttr_grammar(long long M)
{
    long long o;
    int i;
    if (M < 32 || M > VFFT_ZTT_MAX_N || est_v2(M) < 5) return 0;
    o = M >> est_v2(M);
    for (i = 0; i < VFFT_ZTT_ODD_MIDS_N; i++)
        while (o % VFFT_ZTT_ODD_MIDS[i] == 0) o /= VFFT_ZTT_ODD_MIDS[i];
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
static inline int est_zrb(const est_machine_t *m, long long N, est_real_recipe_t *o, char *why, size_t whylen)
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
static inline int est_real(const est_machine_t *m, long long N, int c2r, int ip, int T, est_real_recipe_t *o,
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
         * (_zrf_mt_level_ok, zrf_mt.h:133-136), not a fitted rule (src/estimate/TEST_LOG.md, bug 3). */
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

/* est_plan_real: the recipe of a 1D real request (r2c or c2r, one transform, natural order). Pure: reads only
 * the machine and the request. Returns 0, or -1 with the reason. */
static inline int est_plan_real(const est_machine_t *m, const est_request_t *rq, est_real_recipe_t *out, char *why,
                                size_t whylen)
{
    const int T = rq && rq->nthreads > 1 ? rq->nthreads : 1;
    char dummy[1];
    if (!why || !whylen) { why = dummy; whylen = sizeof dummy; }
    why[0] = 0;
    if (!m || !rq || !out) return est_refuse(why, whylen, "missing argument");
    memset(out, 0, sizeof *out);
    if (rq->transform != VFFT_R2C && rq->transform != VFFT_C2R)
        return est_refuse(why, whylen, "transform %d is not r2c (1) or c2r (2)", rq->transform);
    /* the machine and request checks est_plan_1d makes */
    if (m->lanes != VFFT_IL_VW)
        return est_refuse(why, whylen, "a machine of %d lanes: the formulas follow this build's %d", m->lanes, VFFT_IL_VW);
    if (m->l1d < 1024) return est_refuse(why, whylen, "L1d %ld bytes is not a cache size", m->l1d);
    if (rq->nthreads < 0) return est_refuse(why, whylen, "nthreads %d < 0", rq->nthreads);
    if (rq->placement != VFFT_PLACEMENT_OUT && rq->placement != VFFT_PLACEMENT_IN)
        return est_refuse(why, whylen, "placement %d is not out (0) or in (1)", rq->placement);
    if (rq->order != VFFT_ORDER_NATURAL)
        return est_refuse(why, whylen, "a real plan is natural order only (vfft.c:1941-1957)");
    if (rq->howmany > 1)
        return est_refuse(why, whylen, "one real transform, not a batch of %d", rq->howmany);
    return est_real(m, rq->N, rq->transform == VFFT_C2R, rq->placement == VFFT_PLACEMENT_IN, T, out, why, whylen);
}


/* ── the row ────────────────────────────────────────────────────────────── */

static inline int est__T(const est_request_t *rq) { return rq->nthreads > 1 ? rq->nthreads : 1; }

/* the engines by their row names (wisdom2_real_il.h:7-44) */
enum { EST_ENG_K1, EST_ENG_ZRM, EST_ENG_ZRP, EST_ENG_ZTTR, EST_ENG_ZR2C, EST_ENG_ZFSR, EST_ENG_ZRF, EST_ENG_ZRB,
       EST_ENG_BAD };
static inline int est__eng(const est_real_recipe_t *rc, const est_request_t *rq)
{
    static const char *const names[] = { "k1", "zrm", "zrp", "zttr", "zr2c", "zfsr", "zrf", "zrb" };
    int i;
    if (rq->transform == VFFT_C2C)   /* the c2c row's engine is k1; an empty eng reads the same */
        return (!rc->eng[0] || !strcmp(rc->eng, "k1")) ? EST_ENG_K1 : EST_ENG_BAD;
    for (i = 1; i < (int)(sizeof names / sizeof names[0]); i++)
        if (!strcmp(rc->eng, names[i]))
            return i;
    return EST_ENG_BAD;
}

/* the zr2c child's record: the c2c section mapped 1:1 onto vw2_zr2c_child_t (wisdom2_real_il.h:388-406),
 * fs already in the row's spelling */
static inline void est__zr2c_child(const est_recipe_t *rc, int R1, int R2, const char *fs, vw2_zr2c_child_t *ch)
{
    memset(ch, 0, sizeof *ch);
    ch->route = rc->k1_il_route;
    ch->R1 = R1;
    ch->R2 = R2;
    if (rc->k1_il_route == VFFT_K1_IL_CHAIN3)
        memcpy(ch->c3, rc->il_c3, sizeof ch->c3);
    if (rc->k1_il_route == VFFT_K1_IL_FLAT)
    {
        memcpy(ch->fl, rc->il_fl, sizeof ch->fl);
        ch->fl_n = rc->il_fl_n;
        memcpy(ch->flf, rc->il_flf, sizeof ch->flf);
        ch->flf[sizeof ch->flf - 1] = 0;
    }
    if (rc->k1_il_route == VFFT_K1_IL_ZTT || rc->k1_il_route == VFFT_K1_IL_FS)
    {   /* ZTURN-T: il_ztt; the four-step: its super-band chain, il_sb (wisdom2_real_il.h:446-450) */
        memcpy(ch->zt, rc->il_zt, sizeof ch->zt);
        ch->zt_n = rc->il_zt_n;
    }
    ch->tw = rc->il_tw;
    ch->kv = rc->il_kv;
    ch->bkv = rc->il_bkv;
    if (rc->k1_il_route == VFFT_K1_IL_PRIME)
    {
        ch->pm = rc->prime.method;
        snprintf(ch->pin, sizeof ch->pin, "%s", rc->prime.in_kind);
        snprintf(ch->psh, sizeof ch->psh, "%s", rc->prime.in_shape);
        ch->ptw = rc->prime.in_tw;
        ch->pmt = rc->il_mt;   /* the prime child's threaded arm, on a threaded row only (zr2c writes it then) */
    }
    if (fs && fs[0])
        snprintf(ch->fs, sizeof ch->fs, "%s", fs);
}

/* The row a race would bank for a real request, into st (memory only), by the engine's writer
 * (wisdom2_real_il.h:100-682). to_rp: the four-step child in the rp_ spelling a race banks today (1, the
 * build) or as the recipe spells it (0, the compare arm). Returns 0, or -1 with the reason. */
static inline int est_put_rows_real(vw2_store_t *st, const est_real_recipe_t *rc, const est_request_t *rq,
                                    int to_rp, char *why, size_t whylen)
{
    const int N = rq->N, T = est__T(rq), ip = (rq->placement == VFFT_PLACEMENT_IN);
    const int c2r = (rq->transform == VFFT_C2R), eng = est__eng(rc, rq);
    char fsb[1024];
    int R1, R2, n1 = 0, n2 = 0;
    fsb[0] = 0;
    switch (eng)
    {
    case EST_ENG_ZRM:
        if (vw2_real_il_bank_zrm(st, N, c2r, ip, T, 0.0) != VW2_OK) goto real;
        return 0;
    case EST_ENG_ZRP:
        if (vw2_real_il_bank_zrp(st, N, c2r, ip, T, rc->zrp.R1, rc->zrp.R2, rc->zrp.form, 0.0) != VW2_OK) goto real;
        return 0;
    case EST_ENG_ZTTR:
        if (rc->zttr.nf < 2 || rc->zttr.nf > 8)
            return est_refuse(why, whylen, "zttr chain of %d stages (2..8)", rc->zttr.nf);
        if (vw2_real_il_bank_zttr(st, N, c2r, ip, T, rc->zttr.chain, rc->zttr.nf, rc->zttr.tile, rc->zttr.stk,
                                  rc->zttr.mt, 0.0) != VW2_OK) goto real;
        return 0;
    case EST_ENG_ZRF:
        if (rc->zrf.K < 2 || rc->zrf.K > 10)
            return est_refuse(why, whylen, "zrf chain of %d stages (2..10, zrf.h:228)", rc->zrf.K);
        if (vw2_real_il_bank_zrf(st, N, c2r, ip, T, rc->zrf.chain, rc->zrf.K, rc->zrf.nomsz, rc->zrf.tile,
                                 rc->zrf.mt, 0.0) != VW2_OK) goto real;
        return 0;
    case EST_ENG_ZRB:
        if (vw2_real_il_bank_zrb(st, N, c2r, ip, T, rc->zrb.M, rc->zrb.kind, rc->zrb.shape, rc->zrb.tw,
                                 rc->zrb.mt, 0.0) != VW2_OK) goto real;
        if (!strcmp(rc->zrb.kind, "fs"))
        {   /* a four-step inner's child rides on the zrb row (odd_build.h:731-734) */
            vw2_key_t rk;
            if (!est__split_of(rc->zrb.shape, &n1, &n2) || est__fs_text(rc->c.fs, n1, n2, to_rp, fsb, sizeof fsb) ||
                !fsb[0])
                return est_refuse(why, whylen, "zrb's four-step inner has no child tokens");
            vw2_real_il_key(&rk, N, c2r, ip, T);
            if (est__put_fs(st, &rk, fsb)) goto tok;
        }
        return 0;
    case EST_ENG_ZFSR:
    {   /* the split of N/2 and its child (wisdom2_real_il.h:613-645), the writer's own fs_ codec */
        vw2_rec_t all, c2d, crow, cbwd;
        char *sp = fsb, *t;
        int rc2 = -1;
        if (est__fs_text(rc->c.fs, rc->zfsr.n1, rc->zfsr.n2, to_rp, fsb, sizeof fsb) || !est__fs_has_child(fsb))
            return est_refuse(why, whylen, "zfsr has no child tokens (fs_*): the door would race it (zrp_build.h:330-334)");
        memset(&all, 0, sizeof all); memset(&c2d, 0, sizeof c2d); memset(&crow, 0, sizeof crow); memset(&cbwd, 0, sizeof cbwd);
        while ((t = vw2__tok(&sp)) != NULL)
        {
            char *eq = strchr(t, '=');
            if (!eq) break;
            *eq = 0;
            if (vw2_rec_set(&all, 1, t, eq + 1) != VW2_OK) break;
        }
        if (!t && vw2__fs_get(&c2d, &all, "fs_", "fs_row_") > 0 && vw2__fs_get(&crow, &all, "fs_row_", "fs_row_bwd_") >= 0 &&
            vw2__fs_get(&cbwd, &all, "fs_row_bwd_", NULL) >= 0)
            rc2 = vw2_real_il_bank_zfsr(st, N, c2r, ip, T, rc->zfsr.n1, rc->zfsr.n2, &c2d, crow.ntok ? &crow : NULL,
                                        cbwd.ntok ? &cbwd : NULL, 0.0);
        vw2_rec_free(&all); vw2_rec_free(&c2d); vw2_rec_free(&crow); vw2_rec_free(&cbwd);
        if (rc2 != VW2_OK) goto real;
        return 0;
    }
    case EST_ENG_ZR2C:
    {   /* the child at N/2, natural, one thread, in the c2c record's words (wisdom2_real_il.h:422-480) */
        vw2_zr2c_child_t ch;
        const int half = N / 2;
        if (N & 1)
            return est_refuse(why, whylen, "zr2c needs an even N");
        if (est__c2c_shape(&rc->c, half, &R1, &R2, why, whylen))
            return -1;
        if (est__fs_split(&rc->c, half, 1, &n1, &n2) &&
            (est__fs_text(rc->c.fs, n1, n2, to_rp, fsb, sizeof fsb) || strlen(fsb) >= sizeof ch.fs))
            return est_refuse(why, whylen, "the zr2c child's four-step tokens do not parse or pass the row's %u-byte cap "
                              "(wisdom2_real_il.h:405)", (unsigned)sizeof ch.fs);
        est__zr2c_child(&rc->c, R1, R2, fsb, &ch);
        if (T < 2)
            ch.pmt = 0;
        if (vw2_real_il_bank_zr2c(st, N, c2r, ip, T, rc->zr2c.route, rc->zr2c.fold_mt, &ch, 0.0) != VW2_OK) goto real;
        return 0;
    }
    default:
        return est_refuse(why, whylen, "engine \"%s\" is not a real engine of the door (wisdom2_real_il.h:7-44)", rc->eng);
    }
tok:
    return est_refuse(why, whylen, "a recipe token was refused by the store's lexical rules (wisdom2.h:462-471)");
real:
    return est_refuse(why, whylen, "the %s row was refused by its writer (wisdom2_real_il.h)", rc->eng);
}

/* ── the pre-flight and the plan check ──────────────────────────────────── */

/* the door's real builders for the recipe at T = 1 against c1 (an empty private store), then
 * vfft_destroy. A nested create (zfsr, a four-step inner or child) is checked without its create: the
 * band, the split, its child's two plans (the library's est__fs_preflight). Returns 0, or -1 with the
 * reason. */
static inline int est__preflight_real(const est_real_recipe_t *rc, const est_request_t *rq, const vfft_cfg_t *c1,
                                      char *why, size_t whylen)
{
    const int N = rq->N, c2r = (rq->transform == VFFT_C2R), eng = est__eng(rc, rq);
    struct vfft_plan_s *h = NULL;
    int R1, R2;
    switch (eng)
    {
    case EST_ENG_ZRM:
        if (N > VFFT_ZRM_MAX_N || !vfft_zrm_fn(N, c2r))
            return est_refuse(why, whylen, "no real mono kernel at N=%d", N);
        h = _zrm_build_plan(c1, N);
        break;
    case EST_ENG_ZRP:
        h = _zrp_build_pair(c1, N, rc->zrp.R1, rc->zrp.R2, rc->zrp.form);
        break;
    case EST_ENG_ZTTR:   /* zrp and ZTT-r are 4-lane engines; at 8 lanes the builders return NULL (zttr.h:164-172) */
        h = _zttr_build_plan(c1, N, rc->zttr.chain, rc->zttr.nf, rc->zttr.tile, rc->zttr.stk);
        break;
    case EST_ENG_ZRF:
        h = _zrf_build_plan(c1, N, rc->zrf.chain, rc->zrf.K, rc->zrf.nomsz, rc->zrf.tile);
        break;
    case EST_ENG_ZRB:
    {
        _ilprime_inner_desc_t d;
        int n1, n2;
        if (!_ilprime_desc_parse(&d, rc->zrb.kind, rc->zrb.shape, rc->zrb.tw))
            return est_refuse(why, whylen, "zrb inner \"%s %s\" does not parse", rc->zrb.kind, rc->zrb.shape);
        if (d.kind == 4)
        {   /* the band, zrb's own length law M >= (3N-1)/2 (vfft_zrb_min_m, zrb.h:59; vfft_zrb_create
             * refuses below it, :87 -- the replay then fails and odd_build.h:754 races the odd real),
             * and the child's two plans */
            if (!est__split_of(rc->zrb.shape, &n1, &n2) || (long)n1 * n2 != rc->zrb.M || !vfft_k1fs_band(rc->zrb.M))
                return est_refuse(why, whylen, "zrb's four-step inner %s is not M=%d in the band", rc->zrb.shape, rc->zrb.M);
            if (rc->zrb.M < vfft_zrb_min_m(N))
                return est_refuse(why, whylen, "zrb's M=%d is below (3N-1)/2 = %d at N=%d (zrb.h:59,87)", rc->zrb.M,
                                  vfft_zrb_min_m(N), N);
            return est__fs_preflight(rc->c.fs, n1, n2, why, whylen);
        }
        h = _zrb_build_plan(c1, N, rc->zrb.M, &d);   /* odd_build.h:296 */
        break;
    }
    case EST_ENG_ZFSR:
        /* nested: the band and the split's law (zfsr.h:86-101) and the child */
        if (!vfft_zfsr_band(N) || (long)rc->zfsr.n1 * rc->zfsr.n2 != N / 2 || rc->zfsr.n1 % VFFT_ZFSR_TB ||
            rc->zfsr.n2 % (2 * VFFT_ZFSR_TB))
            return est_refuse(why, whylen, "zfsr split %dx%d is outside the real four-step's band or law", rc->zfsr.n1,
                              rc->zfsr.n2);
        return est__fs_preflight(rc->c.fs, rc->zfsr.n1, rc->zfsr.n2, why, whylen);
    case EST_ENG_ZR2C:
    {   /* the door's replay steps (_zr2c_replay, zr2c_build.h:674-700) up to the composite */
        const est_recipe_t *ch = &rc->c;
        vw2_zr2c_child_t zc;
        vfft_il_cand_t c;
        _zr2c_prime_t pr;
        int n1, n2;
        if (est__c2c_shape(ch, N / 2, &R1, &R2, why, whylen))
            return -1;
        if (est__fs_split(ch, N / 2, 1, &n1, &n2))
        {   /* a four-step child (or a prime child's four-step inner) is a nested create */
            if (ch->k1_il_route == VFFT_K1_IL_FS && !vfft_k1fs_band(N / 2))
                return est_refuse(why, whylen, "zr2c four-step child at %d is outside the band", N / 2);
            if (ch->k1_il_route == VFFT_K1_IL_PRIME &&
                (!vfft_policy_prime_cell(N / 2) || (ch->prime.method == 1 && !vfft_policy_is_prime(N / 2)) ||
                 !vfft_k1fs_band(n1 * n2) || !est__prime_inner_len_ok(N / 2, ch->prime.method, (long)n1 * n2)))
                return est_refuse(why, whylen, "zr2c prime child at %d: its four-step inner %dx%d is not a %s inner of a "
                                  "prime cell (il_prime.h:249, 299-318)", N / 2, n1, n2,
                                  ch->prime.method == 1 ? "Rader" : "Bluestein");
            return est__fs_preflight(ch->fs, n1, n2, why, whylen);
        }
        if (ch->k1_il_route == VFFT_K1_IL_PRIME && !vfft_policy_prime_cell(N / 2))
            return est_refuse(why, whylen, "zr2c prime child at %d is not a prime cell", N / 2);
        est__zr2c_child(ch, R1, R2, "", &zc);
        if (!_zr2c_prime_of_child(&pr, &zc))
            return est_refuse(why, whylen, "zr2c prime child's inner does not parse");
        _zr2c_cand_of_child(&c, &zc);
        h = _zr2c_build_route(c1, N, rc->zr2c.route, &c, &pr, NULL);
        break;
    }
    default:
        return est_refuse(why, whylen, "engine \"%s\" is not a real engine of the door", rc->eng);
    }
    if (!h)
        return est_refuse(why, whylen, "%s does not build as written at N=%d (the door would race the cell)", rc->eng, N);
    vfft_destroy((vfft_plan)h);
    return 0;
}

/* 1 when the handle is the recipe's real engine and every threaded arm the recipe asked for took
 * (a bind that declines leaves the plan serial in silence), and the lean door's entry is bound */
static inline int est__check_plan_real(const struct vfft_plan_s *h, const est_real_recipe_t *rc, const est_request_t *rq,
                                       int *serial, char *why, size_t whylen)
{
    const int T = est__T(rq), eng = est__eng(rc, rq);
    int ok = 0, want = 0, got = 0;
    *serial = 0;
    switch (eng)
    {
    case EST_ENG_ZRM: ok = h->zrm != 0; break;
    case EST_ENG_ZRP: ok = h->zrp != NULL; break;
    case EST_ENG_ZTTR: ok = h->zttr != NULL; if (ok) { want = rc->zttr.mt; got = h->zttr->mt; } break;
    case EST_ENG_ZFSR: ok = h->zfsr != NULL; break;
    case EST_ENG_ZRF: ok = h->zrf != NULL; if (ok) { want = rc->zrf.mt; got = h->zrf->mt; } break;
    case EST_ENG_ZRB: ok = h->zrb != NULL; if (ok) { want = rc->zrb.mt; got = h->zrb->mt; } break;
    case EST_ENG_ZR2C:
        ok = h->zr2c_kid != NULL && h->zr2c_route == rc->zr2c.route;
        if (ok && T > 1)
        {
            want = (rc->zr2c.fold_mt != 0) * 10 + (rc->c.k1_il_route == VFFT_K1_IL_PRIME ? rc->c.il_mt : 0);
            got = (h->zr2c_fold_mt != 0) * 10 + (h->zr2c_kid->ilp ? h->zr2c_kid->ilp->mt : 0);
        }
        break;
    default: break;
    }
    if (!ok)
    {
        est_refuse(why, whylen, "the door built another engine than the recipe's (%s)", rc->eng);
        return 0;
    }
    if (T > 1 && want != got)
    {
        *serial = 1;
        est_refuse(why, whylen, "a threaded arm of %s did not take (asked %d, bound %d)", rc->eng, want, got);
        return 0;
    }
    if (!h->k1_go[rq->transform == VFFT_C2R ? VFFT_BACKWARD : VFFT_FORWARD])
    {   /* every 1D real executor of one transform binds the lean door's entry (_vfft_real_bind_exec,
         * il/il_execute.h); a plan without it would take the full door at every execute */
        est_refuse(why, whylen, "the %s plan has no lean entry (k1_go): each execute would take the full door", rc->eng);
        return 0;
    }
    return 1;
}

/* ── the build ──────────────────────────────────────────────────────────── */

/* est_build_real: the recipe of rq -> a plan built with nothing measured, or NULL with the reason in why and the
 * outcome in rep (may be NULL). cfg is the request as the front door translated it (one 1D interleaved r2c or
 * c2r transform); its store and save flag are not used. Holds the create lock from the pre-flight to the end.
 * The caller owns the plan (vfft_destroy); the caller's pin of a threaded plan is vfft_create's to arm
 * (vfft_api.h). */
static inline struct vfft_plan_s *est_build_real(const est_real_recipe_t *rc, const est_request_t *rq,
                                                 const vfft_cfg_t *cfg, est_build_report_t *rep, char *why,
                                                 size_t whylen)
{
    est_build_report_t lrep;
    struct vfft_wisdom_s *S = NULL, *P = NULL;
    struct vfft_plan_s *h = NULL;
    est__meter_t m0;
    vfft_cfg_t c1;
    int serial = 0, T;
    if (!rep)
        rep = &lrep;
    memset(rep, 0, sizeof *rep);
    rep->status = EST_REFUSED;
    if (why && whylen)
        why[0] = 0;
    if (!rc || !rq || !cfg)
    {
        est_refuse(why, whylen, "no recipe, request or configuration");
        return NULL;
    }
    T = est__T(rq);
    if ((rq->transform != VFFT_R2C && rq->transform != VFFT_C2R) || cfg->transform != rq->transform ||
        cfg->dims > 1 || cfg->n[0] != rq->N || cfg->howmany > 1 || cfg->layout != VFI_LAYOUT_IL ||
        (cfg->nthreads > 1 ? cfg->nthreads : 1) != T ||
        (cfg->placement == VFFT_INPLACE) != (rq->placement == VFFT_PLACEMENT_IN))
    {
        est_refuse(why, whylen, "estimate builds one 1D interleaved r2c or c2r transform at the recipe's placement "
                   "and thread count");
        return NULL;
    }
    if (est__eng(rc, rq) == EST_ENG_BAD || est__eng(rc, rq) == EST_ENG_K1)
    {
        est_refuse(why, whylen, "recipe engine \"%s\" is not a real engine", rc->eng);
        return NULL;
    }
    vfft_env_init();          /* env.h:96: the pre-flight's tables in the create's FP mode */
    _vfft_create_lock();      /* vfft.c:2485, recursive: the pre-flight touches the create's process statics */
    _vfft_create_depth++;     /* the whole build inside a create: a clock read enters a scope */
    est__meter_read(&m0);
    if (!(P = vfft_child_store_new()))
    {
        est_refuse(why, whylen, "out of memory");
        goto out;
    }
    c1 = *cfg;                /* the pre-flight builds at one thread against an empty store */
    c1.nthreads = 1;
    c1.wisdom = (vfft_wisdom *)P;
    c1.wisdom_write = 0;
    c1.recalibrate = 0;
    if (est__preflight_real(rc, rq, &c1, why, whylen))
        goto out;
    if (!(S = vfft_child_store_new()))
    {
        est_refuse(why, whylen, "out of memory");
        goto out;
    }
    if (est_put_rows_real(&S->vw2, rc, rq, 1, why, whylen))
        goto out;
    h = est__replay(S, cfg, &m0, rep, why, whylen);
    if (h && !est__check_plan_real(h, rc, rq, &serial, why, whylen))
    {
        rep->status = serial ? EST_SERIAL : EST_NOPLAN;
        vfft_destroy((vfft_plan)h);
        h = NULL;
    }
    if (h)
        rep->status = EST_BUILT;
out:
    vfft_child_store_free(S);
    vfft_child_store_free(P);
    if (--_vfft_create_depth == 0)
        _vfft_scope_create_done();   /* race_scope.h:621: leave a scope a clock read entered */
    _vfft_create_unlock();
    return h;
}

/* THE FRONT DOOR'S ESTIMATE CREATE OF A REAL TRANSFORM (_vfft_create_estimate, for R2C and C2R): the request,
 * the recipe, the build. NULL with the reason on the error record (vfft_error). */
static vfft_plan _vfft_create_estimate_real(const est_machine_t *m, const vfft_cfg_t *cfg)
{
    est_request_t rq;
    est_real_recipe_t rc;
    char why[640];
    struct vfft_plan_s *h;
    if (cfg->dims > 1 || cfg->howmany > 1)
    {   /* the front door refuses these first (_vfft_api_translate) */
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE plans one R2C or C2R transform at rank 1");
        return NULL;
    }
    memset(&rq, 0, sizeof rq);
    rq.N = cfg->n[0];
    rq.transform = cfg->transform;
    rq.order = cfg->order == VFI_ORDER_SCRAMBLED ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    rq.placement = cfg->placement == VFFT_INPLACE ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    rq.nthreads = cfg->nthreads > 1 ? cfg->nthreads : 1;
    rq.howmany = 1;
    if (est_plan_real(m, &rq, &rc, why, sizeof why))
    {
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE: %s", why);
        return NULL;
    }
    h = est_build_real(&rc, &rq, cfg, NULL, why, sizeof why);
    if (!h)
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE: %s", why);
    return (vfft_plan)h;
}

#endif /* VFFT_IL_REAL_ESTIMATE_IL_H */
