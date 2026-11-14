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
 * section 7 (fitted at T = 8), with the owner's rules of 2026-11-01 (docs/roadmap/real_estimate_plan.md
 * section 7, each fitted on the estimate's own recipe changed in one place): zttr's and zrb's threaded arms by
 * the c2c ZTURN-T rule (est_inner_mt; zttr at M = N/2, zrb at its convolution length, an untiled zrb inner
 * tiled at T > 1), a zr2c prime half's bound at its inner's length, zrb's length past a three-stage inner,
 * zrf's chain led by its largest radix up to 9, zttr from N = 1024 where N is not a power of two and for r2c
 * powers of two from 16384, the zrp pair with a second factor of 10 or more. The zr2c half is the c2c
 * estimate at N/2 (est_c2c), so the c2c rules reach it.
 * AT 8 LANES (an AVX-512 build; the owner, 2026-11-13) the pair, ZTT-r and the rows kernel are named by the same
 * rules, with the differences the 8-lane engines' laws and the first AVX-512 planner rows ask for, each at its
 * rule below: a ZTT-r chain is led by an 8, c2r powers of two take ZTT-r from 2048, and at one thread the real
 * four-step starts at N = 524288. To be refined on the next AVX-512 machine.
 * The sketch's open decisions (its section 12) take these defaults: every T > 1 takes the rules fitted at
 * T = 8; the formulas follow the build's lanes (an 8-lane recipe is planned by an AVX-512 build only).
 * zfsr at T > 1 takes the c2c four-step's threaded split and child at N/2 (the owner, 2026-11-01). A gap of
 * the sketch, reported and not filled: zrf's LEVELS arm only where the library can bind it (est_real).
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
/* the real engines' lanes: the pair, ZTT-r and the rows kernel run the build ISA's doubles per vector, 4 at AVX2
 * and 8 at AVX-512 (VFFT_ZTTR_VW, il/real/zttr.h; VFFT_ZRP_VW, il/real/zrp.h) */
#define EST_REAL_VW VFFT_ZTTR_VW

/* the real mono: VFFT_IL_RN1_PAIR_RADICES (il_registry_avx2.h:613) plus 32 (rn1b48) and 64
 * (rn1b444), N <= VFFT_ZRM_MAX_N = 64 (il/real/zrm.h:39,42-57); the same at avx512 */
static const int est_rn1_list[] = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21,
                                    22, 23, 25, 26, 27, 29, 31, 32, 37, 41, 43, 47, 53, 59, 61, 64 };
/* the real pair's radices: the VFFT_IL_T2M / R2Z / T2H lists, the same at avx2 and at avx512
 * (il_registry_<isa>.h) */
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
    struct { int R1, R2, form; } zrp;    /* [pair=R1.R2 leaf=n1t|r2z] form 0 A (n1t), 1 B (r2z)
                                          * (sketch 6)                                           */
    struct { int chain[8], nf; size_t tile; int stk, mt; } zttr;
                                         /* [chain= tile= stk= mt=] the chain at M = N/2, 2..8
                                          * stages; stk = 3; mt 0 = absent                       */
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
    int tcmt;                            /* [tcmt] a contiguous batch on T threads whose door builds
                                          * copies (est__real_copies): the batch's own row (eng=tcb,
                                          * q=K, nthreads=T), 1 whole transforms per thread, 0 all
                                          * threads on each transform (est_batch_whole)          */
} est_real_recipe_t;

/* ── the formulas (the sketch's section 6 real, section 7 threads) ──────── */

/* ZTT-r's grammar at M = N/2 (zttr.h, vfft_zttr_create): 2^a times a product over {15, 9, 7, 5, 3}; the terminator
 * run L = M / R_last comes in whole pairs of vectors, L >= 2 VW and L % (2 VW) == 0, so 8 VW | M: a >= 5 at 4
 * lanes, a >= 6 at 8; 32 <= M <= 262144 */
static inline int est_zttr_grammar(long long M)
{
    long long o;
    int i;
    if (M < 32 || M > VFFT_ZTT_MAX_N || est_v2(M) < (EST_REAL_VW == 8 ? 6 : 5)) return 0;
    o = M >> est_v2(M);
    for (i = 0; i < VFFT_ZTT_ODD_MIDS_N; i++)
        while (o % VFFT_ZTT_ODD_MIDS[i] == 0) o /= VFFT_ZTT_ODD_MIDS[i];
    return o == 1;
}
/* a zttr tile of width w at M after the stages c (zrp_build.h:445-458: w < M, M % w == 0, a multiple of R0 R1) */
static inline int est__zttr_tile_ok(long long M, const int *c, long long w)
{
    return w < M && M % w == 0 && w % ((long long)c[0] * c[1]) == 0;
}
/* ZTT-r's laws on a whole chain (est_ztt_ok, natural: at 8 lanes the first stage is 8) and its terminator: the
 * last stage 4 or 8 over a run M / R_last of whole pairs of vectors, 2 VW points at least (vfft_zttr_create) */
static inline int est__zttr_legal(long long M, const int *c, int k)
{
    return k <= VFFT_ZTT_MAX_NF && est_ztt_ok(M, c, k, 0) && (c[k - 1] == 4 || c[k - 1] == 8) &&
           M / c[k - 1] >= 2 * EST_REAL_VW && (M / c[k - 1]) % (2 * EST_REAL_VW) == 0;
}
/* 2^a odd M: the first power-of-two stage, the odd stages mids, the other power-of-two stages, the closing
 * radix; 2^(a - lg close) as 8s, one 4 and 8s, or two 4s and 8s; at 8 lanes its first 8 leads (the law), and
 * with no 8 there is no chain. 0 = none within the laws. */
static inline int est__zttr_odd(long long M, int a, const int *mids, int nm, int close, int *c)
{
    const int r = a - (close == 8 ? 3 : 2);
    int p2[16], np = 0, k = 0, i;
    if (r < 0) return 0;
    if (r % 3 == 0) { for (i = 0; i < r / 3; i++) p2[np++] = 8; }
    else if (r % 3 == 2) { p2[np++] = 4; for (i = 0; i < r / 3; i++) p2[np++] = 8; }
    else { if (r < 4) return 0; p2[np++] = 4; p2[np++] = 4; for (i = 0; i < (r - 4) / 3; i++) p2[np++] = 8; }
    if (EST_REAL_VW == 8)
    {
        for (i = 0; i < np && p2[i] != 8; i++) ;
        if (i == np) return 0;
        for (; i > 0; i--) p2[i] = p2[i - 1];
        p2[0] = 8;
    }
    if (!np || np + nm + 1 > 16) return 0;
    c[k++] = p2[0];
    for (i = 0; i < nm; i++) c[k++] = mids[i];
    for (i = 1; i < np; i++) c[k++] = p2[i];
    c[k++] = close;
    return est__zttr_legal(M, c, k) ? k : 0;
}
/* ZTT-r's fewest stages at 2^a odd M: the odd part greedily largest first over {15, 9, 7, 5, 3} (the ZTT's own odd
 * grammar, dp_planner_il.h:1470-1490), the power of two in the fewest stages, closing 8 (else 4). 0 = none. */
static inline int est_zttr_fewest(long long M, int *c)
{
    int mids[16], nm = 0, a = est_v2(M), i, k, close;
    long long o = M >> a;
    for (i = 0; i < VFFT_ZTT_ODD_MIDS_N; i++)
        while (o % VFFT_ZTT_ODD_MIDS[i] == 0 && nm < 16) { mids[nm++] = VFFT_ZTT_ODD_MIDS[i]; o /= VFFT_ZTT_ODD_MIDS[i]; }
    if (o != 1 || !nm) return 0;
    for (close = 8; close >= 4; close -= 4)
        if ((k = est__zttr_odd(M, a, mids, nm, close, c)) > 0)
            return k;
    return 0;
}
/* section 6 even rule 4, the zttr chain at M = N/2. Power-of-two M: c2r 4, then 8s, then 4s; r2c 8,
 * the 4s, then 8s, closing 4 (with no forced 4, two 8s become three 4s). 2^a odd M: the first pow2
 * stage, the odd primes 7, 5, 3, the other pow2 stages, closing 4. ZTT-r holds at most VFFT_ZTT_MAX_NF = 7
 * stages; past it the fewest stages (est_zttr_fewest). Before 2026-11-04, 77 of the 294 lengths N >= 1024 with N/2
 * in the grammar, 68 of them from 100,000, had no chain and fell to zr2c, whose child runs one thread: 4.7 x the
 * planner at 400,000 and 6.7 x at 276,480 on 8 threads, 1.15 x at one thread; with the fewest stages, 16 of them
 * 1.01 x at one thread (the planner's 31,104 8.3.9.9.8, 400,000 8.5.5.5.5.5.8, 103,680 4.15.9.3.4.8). Tile: untiled
 * below M = 4096, else the first legal width of 2048, 3072, 1024, 512. Validated by ZTT's natural laws and
 * the terminator's (est__zttr_legal). 0 = none.
 * AT 8 LANES the first stage is 8 (the law). r2c's power-of-two chain leads with 8 as it stands. c2r's cannot lead
 * with its 4: two 8s, the forced 4s, then the other 8s (2026-11-13, a 2-core Xeon at AVX-512, one thread: the
 * planner's chain on 10 of its 15 c2r power-of-two ZTT-r rows, 2048 8.8.4.4, 8192 8.8.8.8, 16384 8.8.4.4.8,
 * 262144 8.8.4.8.8.8; no other one form names more). 2^a odd M: est__zttr_odd's stages, its first 8 leading. */
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
        if (EST_REAL_VW == 8 && c2r)
        {   /* 8 lanes: two 8s, the forced 4s, then the other 8s */
            if (e < 1) return 0;
            c[k++] = 8;
            if (e >= 2) c[k++] = 8;
            for (i = 0; i < f; i++) c[k++] = 4;
            for (i = 0; i < e - 2; i++) c[k++] = 8;
        }
        else
        {
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
    }
    else
    {
        int mids[16], nm = 0;
        static const int op[3] = { 7, 5, 3 };
        for (i = 0; i < 3; i++)
            while (o % op[i] == 0 && nm < 16) { mids[nm++] = op[i]; o /= op[i]; }
        if (o != 1) return 0;
        if (!(k = est__zttr_odd(M, a, mids, nm, 4, c)))
            k = est_zttr_fewest(M, c);
    }
    if (!est__zttr_legal(M, c, k)) return 0;
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
/* zttr on threads that take tiles (zttr_mt.h TILES): a tile must hold the first two stages, R0 R1 | w, so the
 * power-of-two stages lead, ascending, then the odd stages, the largest power-of-two stage last; the tile the
 * widest of 512, 1024, 2048, 3072 that is legal and whose data, 16 w bytes, fits half the L1d (1024 on 48 and on
 * 32 KiB). At M a power of two that is the chain ascending: on the 10 cells 16,384 - 262,144 at 8 threads
 * (docs/roadmap/real_estimate_plan.md section 7) ascending with the tile 1024 1.042 against the planner, the
 * one-thread order 1.104 - 1.123, ascending alone 1.075, the tile alone 1.152. Rewrites c and *tile; 0 = no
 * legal tile in that order (the one-thread chain stays). */
static inline int est_zttr_mt_order(const est_machine_t *m, long long M, int *c, int k, size_t *tile)
{
    static const int widths[4] = { 3072, 2048, 1024, 512 };
    int d[VFFT_ZTT_MAX_NF], p[VFFT_ZTT_MAX_NF], np = 0, n = 0, i;
    for (i = 0; i < k; i++)
        if (c[i] == 4 || c[i] == 8) p[np++] = c[i];
    if (np < 2 || k > VFFT_ZTT_MAX_NF) return 0;
    est_sort_asc(p, np);
    for (i = 0; i < np - 1; i++) d[n++] = p[i];
    for (i = 0; i < k; i++)
        if (c[i] != 4 && c[i] != 8) d[n++] = c[i];
    d[n++] = p[np - 1];
    if (!est__zttr_legal(M, d, n)) return 0;
    for (i = 0; i < 4; i++)
        if (16LL * widths[i] <= m->l1d / 2 && est__zttr_tile_ok(M, d, widths[i]))
        {
            memcpy(c, d, sizeof(int) * (size_t)n);
            *tile = (size_t)widths[i];
            return n;
        }
    return 0;
}
/* section 6 zrf's chain: the prime factors, 3.3 -> 9, ascending; 2 <= K <= 10 (zrf.h:228), the
 * section 4.4 merges above 10 stages, re-sorted ascending (29296875 = 3.5.5.5.5.5.5.5.5.25); then the
 * largest radix up to 9 leads, the rest ascending (the owner, 2026-11-01: the planner leads with 9 or 7,
 * 3375 9.5.3.5.5, 105 7.3.5, 945 9.3.7.5; over 22 odd composites at one thread 1.101 -> 1.041 against it,
 * where the largest first, 23 at 1035, lost and the descending order lost). 0 = none. */
static inline int est_zrf_chain(long long N, int *c)
{
    int k = est_odd_stages(N, c), b = -1, i;
    if (k < 2) return 0;
    est_sort_asc(c, k);
    if (k > VFFT_ILFD_MAX_K)
    {
        est_sort_desc(c, k);
        k = est_merge10(c, k);
        if (k < 2) return 0;
        est_sort_asc(c, k);
    }
    for (i = 0; i < k; i++)
        if (c[i] <= 9) b = i;   /* ascending: the last radix up to 9 is the largest */
    if (b > 0)
    {
        const int lead = c[b];
        for (i = b; i > 0; i--) c[i] = c[i - 1];
        c[0] = lead;
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
    /* a three-stage inner is passed over for the next length whose inner is a pair or ZTURN-T (the owner,
     * 2026-11-01: 1009 and 1031 at 1536 or 1792 with a three-stage inner, 1.10 - 1.15 against the planner, at
     * 2048 with ZTURN-T, the planner's, 1.03 - 1.06); a three-stage inner serves only when no other length has
     * an inner */
    for (j = 0; j < 2; j++)
        for (i = 0; i < nc; i++)
        {
            est_inner_t in;
            memset(&in, 0, sizeof in);
            o->c.fs[0] = 0;
            r = est_zrb_inner(m, cand[i], &in, o->c.fs, sizeof o->c.fs, why, whylen);
            if (r < 0) return -1;
            if (r == 0 || (j == 0 && !strcmp(in.kind, "3p"))) continue;
            snprintf(o->eng, sizeof o->eng, "zrb");
            o->zrb.M = (int)cand[i];
            snprintf(o->zrb.kind, sizeof o->zrb.kind, "%s", in.kind);
            snprintf(o->zrb.shape, sizeof o->zrb.shape, "%s", in.shape);
            o->zrb.tw = in.tw;
            o->zrb.mt = 0;   /* the threaded arm: est_real */
            return 0;
        }
    return est_refuse(why, whylen, "real N = %lld: no zrb length up to 2^24 has an inner (sketch 6)", N);
}

static inline int est__real_zr2c(const est_machine_t *m, long long N, int c2r, int ip, int T, est_real_recipe_t *o,
                                 char *why, size_t whylen);

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
            if (est_zrb(m, N, o, why, whylen)) return -1;
            if (T > 1 && !strcmp(o->zrb.kind, "ztt") && !est_serial(m, o->zrb.M))
            {   /* the threaded arm (the owner, 2026-11-01): the c2c prime cell's rule, the bound read at the
                 * convolution length M (est_inner_mt); an untiled inner (the odd band's) takes the first legal tile
                 * of 1024, 2048, 512 so it runs TILES. Against the planner at 8 threads: 4099 (M = 7168) TILES
                 * 0.84 - 0.89, BLOCKS 0.92 - 0.94, one thread 1.0 (the planner's row there is serial); 50021 TILES
                 * 1.11 - 1.15, BLOCKS 1.19 - 1.23, one thread 5.2 - 5.6 */
                static const int tiles[3] = { 1024, 2048, 512 };
                est_inner_t in;
                int zc[16], nz = 0, v = 0, t;
                const char *p;
                for (p = o->zrb.shape;; p++)
                {   /* the inner's chain from its shape */
                    if (*p >= '0' && *p <= '9') { v = 10 * v + (*p - '0'); continue; }
                    if (v && nz < 16) zc[nz++] = v;
                    v = 0;
                    if (!*p) break;
                }
                for (t = 0; t < 3 && !o->zrb.tw; t++)
                    if (o->zrb.M / tiles[t] >= 2 && est_ztt_tile_ok(o->zrb.M, zc, nz, tiles[t], 1))
                        o->zrb.tw = tiles[t];
                memset(&in, 0, sizeof in);
                in.ztt = 1;
                in.tw = o->zrb.tw;
                o->zrb.mt = est_inner_mt(m, &in, o->zrb.M, o->zrb.M, T);
            }
            return 0;
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
    /* 2. zrp, N <= 160 (table 5.8), an even x even pair R1.R2, form B (the r2z leaf): 10 second
     * when it divides; else the most balanced pair whose second factor is at least 10; else the most balanced,
     * smaller factor first (the owner, 2026-11-01: 32 of the store's 34 pair rows, the most balanced pair 21;
     * at one thread 48 4.12, 64 4.16, 120 12.10, 160 16.10 against 6.8, 8.8, 10.12, 10.16: 1.07 - 1.28 ->
     * 0.95 - 1.03). The same at 8 lanes (2026-11-13, a 2-core Xeon at AVX-512, one thread: its planner banked the
     * pair on 21 of its 32 rows at 32 - 160, this rule's pair on 20 of them; 128 took zr2c in all four cells), but
     * for N = 128, which stays on zr2c (2026-11-14, the same Xeon, the estimate's plans timed against the day before:
     * the pair 8.16 0.099 / 0.095 / 0.101 us against zr2c's 0.084 / 0.085 / 0.090 in R2C and C2R out of place and
     * R2C in place, the AVX2 build's times steady between the two runs; C2R in place 0.102 against 0.114) */
    if (N <= 160 && !(EST_REAL_VW == 8 && N == 128))
    {
        int lo = 0, hi = 0, pass;
        for (pass = 0; pass < 3 && !hi; pass++)
            for (i = 0; i < EST_NELEM(est_zrp_list); i++)
            {   /* pass 0: R2 = 10; pass 1: R2 >= 10; pass 2: any */
                const int a = est_zrp_list[i], b = (int)(N / a), d = a > b ? a - b : b - a, dd = lo > hi ? lo - hi : hi - lo;
                if (N % a || !est_in(est_zrp_list, EST_NELEM(est_zrp_list), b)) continue;
                if ((pass == 0 && b != 10) || (pass == 1 && b < 10)) continue;
                if (!hi || d < dd || (d == dd && a < lo)) { lo = a; hi = b; }
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
     * At T > 1 the c2c four-step's threaded recipe at N/2 (the owner, 2026-11-01): the door builds its c2c
     * four-step child at T and seeds the child's rows under T (_k1fs_child_keys(n1, n2, 0, T),
     * zrp_build.h:308-318), so the split is the c2c threaded split (est_fs_split, threaded: N/2048 up to 2^20,
     * then 4096) and the child is that split's own at T (est_fs_child), as the c2c four-step's row at T.
     * At 8 lanes and one thread the band starts an octave lower, at ZTT-r's ceiling N/2 = 2^18, as the square
     * 512 x 512 (2026-11-13, the Xeon's planner at AVX-512: zfsr at N = 524288 in all four cells, its splits
     * 1024x256 twice, 512x512 and 256x1024; 1.08 - 1.20 x faster than the zr2c the estimate named there). */
    if (EST_REAL_VW == 8 && T == 1 && N / 2 == (1LL << 18))
    {
        snprintf(o->eng, sizeof o->eng, "zfsr");
        o->zfsr.n1 = 512;
        o->zfsr.n2 = 512;
        return est_fs_child(m, 512, 512, 1, o->c.fs, sizeof o->c.fs, why, whylen);
    }
    if (est_is_pow2(N / 2) && N / 2 >= (1LL << 19))
    {
        int n1, n2;
        if (!est_fs_split(N / 2, T > 1, &n1, &n2) || n1 % EST_ZFSR_TB || n2 % (2 * EST_ZFSR_TB))
            return est_refuse(why, whylen, "real N = %lld: no zfsr split at T = %d (sketch 6)", N, T);
        snprintf(o->eng, sizeof o->eng, "zfsr");
        o->zfsr.n1 = n1;
        o->zfsr.n2 = n2;
        return est_fs_child(m, n1, n2, T, o->c.fs, sizeof o->c.fs, why, whylen);
    }
    /* 4. zttr: a pow2 N in 512..524288 (r2c from 16384), or N >= 1024 with N/2 in ZTT-r's grammar
     * (the owner, 2026-11-01: at one thread zttr wins from 1152 where N is not a power of two, 1920 1.20 ->
     * 1.005; r2c powers of two up to 8192 run faster as zr2c, 512 1.01 -> 0.90, 1024 1.07 -> 0.96, and the
     * planner's r2c rows take zttr at powers of two from 16384; c2r unchanged).
     * At 8 lanes c2r's powers of two start at 2048 (2026-11-13, the Xeon's planner at AVX-512, one thread: c2r 512
     * and 1024 on zr2c in both placements, zttr from 2048 out of place and 4096 in place; out of place at 4096 -
     * 262144 its zttr rows ran 1.13 - 1.41 x faster than the zr2c the estimate named, in place and in r2c within
     * 0.84 - 1.22; 1536, 3840, 6144 took zttr in every cell, 1920 has no 8-lane chain) */
    if ((est_is_pow2(N) && N >= (c2r ? (EST_REAL_VW == 8 ? 2048 : 512) : 16384) && N <= 524288) ||
        (!est_is_pow2(N) && N >= 1024 && est_zttr_grammar(N / 2)))
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
            {   /* the c2c ZTURN-T rule at the real data M = N/2 (the owner, 2026-11-01): serial under the bound, else
                 * TILES with at least 2 tiles, else BLOCKS (est_inner_mt; zttr_mt.h binds the same law). Against the
                 * planner at 8 threads (docs/roadmap/real_estimate_plan.md section 7): serial at 1024 and 4096 (BLOCKS
                 * there 1.6 - 8.3 x slower), TILES from 16384 (1.55 - 5.2 -> 0.96 - 1.29, BLOCKS 1.03 - 1.46) */
                est_inner_t in;
                memset(&in, 0, sizeof in);
                in.ztt = 1;
                in.tw = (int)tile;
                o->zttr.mt = est_inner_mt(m, &in, N / 2, N / 2, T);
                if (o->zttr.mt && (k = est_zttr_mt_order(m, N / 2, c, k, &tile)) > 0)
                {   /* threaded: the order and the tile that let the threads take tiles */
                    memcpy(o->zttr.chain, c, sizeof(int) * (size_t)k);
                    o->zttr.nf = k;
                    o->zttr.tile = tile;
                    in.tw = (int)tile;
                    o->zttr.mt = est_inner_mt(m, &in, N / 2, N / 2, T);
                }
                if (o->zttr.mt == 1 && (k = est_zttr_fewest(N / 2, c)) > 0)
                {   /* BLOCKS cuts every stage by units, one fork-join a stage (zttr_mt.h): the fewest stages, untiled, as
                     * the planner's threaded untiled rows take them (62,208 4.9.9.3.4.8, 103,680 4.15.9.3.4.8; the
                     * one-thread chains, 7 stages there, ran 1.17 - 1.30 x the planner at 8 threads, these 1.02) */
                    memcpy(o->zttr.chain, c, sizeof(int) * (size_t)k);
                    o->zttr.nf = k;
                    o->zttr.tile = 0;
                }
            }
            return 0;
        }
    }
    /* 5. zr2c */
    return est__real_zr2c(m, N, c2r, ip, T, o, why, whylen);
}

/* zr2c at even N: the child is the library's c2c procedure at N/2, natural, one thread; route child_nat_ip in place
 * and for c2r from N = 1024 (table 5.8), else child_oop_il. est_real's last step, and the 2D real rows' engine where
 * their batch needs copies the picked engine cannot give (est_plan_2d_real). Clears o. */
static inline int est__real_zr2c(const est_machine_t *m, long long N, int c2r, int ip, int T, est_real_recipe_t *o,
                                 char *why, size_t whylen)
{
    memset(o, 0, sizeof *o);
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
        if (T > 1 && N >= 64)
        {   /* section 7: fold=mt (zrp_build.h:795-818) above the bound at N/2; a prime child reads the bound at
             * its inner's length M, as the c2c prime cell (the owner, 2026-11-01: 8198, the half 4099, the fold
             * and the prime child threaded 0.995 against the planner, one thread 1.24 - 1.28), and takes its own
             * arm there */
            const int prime = o->c.k1_il_route == VFFT_K1_IL_PRIME;
            est_inner_t in;
            long long M = 1;
            memset(&in, 0, sizeof in);
            if (prime)
            {
                const char *p = o->c.prime.in_shape;
                long long v = 0;
                in.tw = o->c.prime.in_tw;
                in.ztt = !strcmp(o->c.prime.in_kind, "ztt");
                for (;; p++)
                {   /* M = the inner shape's product (_ilprime_desc_len, k1_commit.h:488-496) */
                    if (*p >= '0' && *p <= '9') { v = 10 * v + (*p - '0'); continue; }
                    M *= v;
                    v = 0;
                    if (!*p) break;
                }
            }
            /* R2C's fold reads the one-thread child's output in the calling core's L2: it threads once the child's
             * input and output, 16 N bytes, pass that L2 (2026-11-04: the store's 86 rows agree at 85, the bound
             * 16 N / 2 > 2 L1d at 77; the misses were R2C 28000 - 220034 the planner keeps serial, 100000 1.32 x) */
            if (prime ? !est_serial(m, M) : (c2r ? !est_serial(m, N / 2) : 16LL * N > (long long)m->l2))
            {
                o->zr2c.fold_mt = 1;
                if (prime)
                    o->c.il_mt = est_inner_mt(m, &in, M, M, T);
            }
        }
        return 0;
    }
}

/* the contiguous batch's copies on T threads (vfft.c, the wrapper's clones: each built in the slab role and kept
 * when _tc_inner_mt_safe holds): every real engine but the real four-step (its team threads on the pool) and ZTT-r
 * in place (the plan owns one scratch plane beside the caller's buffer, zttr.h:100) */
static inline int est__real_copies(const est_real_recipe_t *rc, int ip)
{
    return strcmp(rc->eng, "zfsr") && !(ip && !strcmp(rc->eng, "zttr"));
}
/* THE CONTIGUOUS REAL BATCH'S WAY ON T THREADS (the owner's rule, 2026-11-03; docs/roadmap/real_estimate_plan.md section
 * 12, fitted on the study's two ways timed at every cell, 176 cells at 8 threads, 20 lengths x K 3 - 24, both
 * directions): whole transforms per thread while one transform's N/2 + 1 bins, 16 bytes each, fit a core's L2, when
 * the batch carries EST_BATCH_FLOOR bytes or the transform is a prime one -- an odd N, or the half of an even one, with a
 * prime factor no chain radix carries (vfft_policy_prime_cell): whatever engine runs it (odd Bluestein, the small real
 * kernel at 53 - 61, the half-length engine on a prime half), its work per byte is many times the others', and 61, 131
 * or 262 at K = 3 already pay a copy each; else all threads on each transform (below the floor, the serial loop). The
 * C2C rule's L2 / 8 bound (est_batch_whole) assumes one transform on T threads uses them all; a real one mostly does not
 * (zr2c threads its fold only): it lost 3 - 7 times at 65536 and 100000. The first study, 176 cells: 1.107 -> 1.000
 * against the planner; with the calibration set (108 cells over the four layers, K 2 - 64, the banked odd lengths at
 * K = 64), 284 cells: 1.018 (1.023 against the faster way; the prime clause read off the engine 1.030; a threaded
 * engine's own bound or a K-against-T clause no better). The 30 cells drawn after it (odd lengths, 69 x 27 - 639401 x
 * 12) missed it at 1.389, two ways: an odd N (an odd engine: zrf, zrb, the small kernel) does many times an even one's
 * work per byte, so its floor is half the C2C one (69 x 27, 19 x 53, 375 x 5: whole transforms 1.5 - 3.9 times
 * faster below 16 KiB); and odd Bluestein with its own threaded arm (zrb mt) scales on its transform from
 * EST_REAL_BATCH_ZRB_MT bytes (24031 x 2, 92861 x 6, 97217 x 56: all threads on each 1.7 - 2.1 times faster). With
 * both, all 314 cells 1.049 -> 1.032 against the planner. The large transforms (2026-11-04): all threads on each pay
 * only where the one transform's plan threads its own work (real ZTURN-T, odd Bluestein or the odd flat DIT on their
 * threaded arms, a prime half whose convolution threads; est__real_threads_own), and there only once its input and
 * output, 2 x 16 (N/2 + 1) bytes, pass the L2; a zr2c whose child runs one thread threads only its fold, so whole
 * transforms win at any size (480240 x 24 2.0 x, 311722 x 5 2.3 x, 720000 x 12, 1e6 x 8, 1024000 x 4 past the L2).
 * The store's 364 rows agree at 332, the bytes-within-L2 bound for every engine at 322. */
#define EST_REAL_BATCH_ZRB_MT (128 << 10)
static inline int est__real_threads_own(const est_real_recipe_t *rc)
{
    if (!strcmp(rc->eng, "zttr")) return rc->zttr.mt > 0;
    if (!strcmp(rc->eng, "zrb")) return rc->zrb.mt > 0;
    if (!strcmp(rc->eng, "zrf")) return rc->zrf.mt > 0;
    if (!strcmp(rc->eng, "zr2c")) return rc->c.il_mt > 0;
    return 0;
}
static inline int est__real_batch_whole(const est_machine_t *m, const est_real_recipe_t *rc, long long N, size_t K)
{
    const size_t bytes = 16 * ((size_t)N / 2 + 1);
    const int prime = vfft_policy_prime_cell((int)((N & 1) ? N : N / 2));
    if (K < 2 || m->l2 <= 0)
        return 0;
    if (!strcmp(rc->eng, "zrb") && rc->zrb.mt && bytes >= EST_REAL_BATCH_ZRB_MT)
        return 0;
    if (est__real_threads_own(rc) && 2 * bytes > (size_t)m->l2)
        return 0;
    return prime || (double)K * (double)bytes >= ((N & 1) ? EST_BATCH_FLOOR / 2 : EST_BATCH_FLOOR);
}

/* est_plan_real: the recipe of a 1D real request (r2c or c2r, natural order): one transform, or a contiguous batch
 * of them (howmany > 1), each the one transform's plan, on T threads the way est__real_batch_whole picks where the
 * door builds copies. Pure: reads only the machine and the request. Returns 0, or -1 with the reason. */
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
    if (est_real(m, rq->N, rq->transform == VFFT_C2R, rq->placement == VFFT_PLACEMENT_IN, T, out, why, whylen))
        return -1;
    if (rq->howmany > 1 && T > 1)
    {
        const int ip = rq->placement == VFFT_PLACEMENT_IN;
        est_real_recipe_t zr;
        if (ip && !strcmp(out->eng, "zttr") &&
            !est__real_zr2c(m, rq->N, rq->transform == VFFT_C2R, 1, T, &zr, why, whylen) &&
            est__real_batch_whole(m, &zr, rq->N, rq->howmany))
        {   /* whole transforms per thread need the batch's copies; ZTT-r in place owns one scratch plane and gets none
             * (vfft.c _tc_inner_mt_safe), so its batch ran serially: zr2c, the planner's engine there, as the 2D real
             * rows take it (est_plan_2d_real). 2026-11-12, C2R 512 x 512 in place at 8 threads: 193.9 us against
             * 38.8 out of place */
            *out = zr;
            out->tcmt = 1;
        }
        else if (est__real_copies(out, ip))
            out->tcmt = est__real_batch_whole(m, out, rq->N, rq->howmany);
        why[0] = 0;
    }
    return 0;
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
    case EST_ENG_ZTTR:   /* the pair and ZTT-r build at both lane counts (ZTT-r where its chain is legal there:
                          * at 8 lanes the first radix is 8, zttr.h) */
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
 * c2r transform, or a contiguous batch of them: the transform-contiguous wrapper over the one transform's plan,
 * vfft.c, whose way on T threads its own row says when the door builds copies); its store and save flag are not
 * used. Holds the create lock from the pre-flight to the end. The caller owns the plan (vfft_destroy); the
 * caller's pin of a threaded plan is vfft_create's to arm (vfft_api.h). */
static inline struct vfft_plan_s *est_build_real(const est_real_recipe_t *rc, const est_request_t *rq,
                                                 const vfft_cfg_t *cfg, est_build_report_t *rep, char *why,
                                                 size_t whylen)
{
    est_build_report_t lrep;
    struct vfft_wisdom_s *S = NULL, *P = NULL;
    struct vfft_plan_s *h = NULL;
    const struct vfft_plan_s *h1;
    est__meter_t m0;
    vfft_cfg_t c1;
    int serial = 0, T, copies;
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
    copies = cfg->howmany > 1 && T > 1 && est__real_copies(rc, rq->placement == VFFT_PLACEMENT_IN);
    if ((rq->transform != VFFT_R2C && rq->transform != VFFT_C2R) || cfg->transform != rq->transform ||
        cfg->dims > 1 || cfg->n[0] != rq->N || (cfg->howmany > 1 && cfg->batch_geom != VFFT_BATCH_CONTIGUOUS) ||
        (cfg->howmany > 1 ? cfg->howmany : 1) != (rq->howmany > 1 ? rq->howmany : 1) || cfg->layout != VFI_LAYOUT_IL ||
        (cfg->nthreads > 1 ? cfg->nthreads : 1) != T ||
        (cfg->placement == VFFT_INPLACE) != (rq->placement == VFFT_PLACEMENT_IN))
    {
        est_refuse(why, whylen, "estimate builds one 1D interleaved r2c or c2r transform, or a contiguous batch of them, "
                   "at the recipe's placement, thread count and batch");
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
    c1 = *cfg;                /* the pre-flight builds one transform at one thread against an empty store */
    c1.nthreads = 1;
    c1.howmany = 1;
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
    if (copies && vw2_tcmt_bank(&S->vw2, rq->transform == VFFT_C2R ? VW2_T_C2R : VW2_T_R2C, rq->N, cfg->howmany,
                                       vfft_policy_ord_k1(cfg, rq->N, cfg->placement == VFFT_INPLACE),
                                       cfg->placement == VFFT_INPLACE ? VW2_PL_IP : VW2_PL_OOP, _vw2_lay_of(cfg), rc->tcmt,
                                       T, 0.0) != VW2_OK)
    {   /* the batch's own row as the wrapper looks it up (_tc_mt_decide, vfft.c) */
        est_refuse(why, whylen, "the batch's row (eng=tcb) was refused by its writer");
        goto out;
    }
    h = est__replay(S, cfg, &m0, rep, why, whylen);
    if (h && (est__batch_one(h, cfg, copies && rc->tcmt, &h1, why, whylen) ||
              !est__check_plan_real(h1, rc, rq, &serial, why, whylen)))
    {
        rep->status = serial ? EST_SERIAL : EST_NOPLAN;
        vfft_destroy((vfft_plan)h);
        h = NULL;
    }
    else if (h && copies && rc->tcmt)
    {   /* the first copy runs the recipe's engine in its one-thread form (_tc_slab_role) */
        est_request_t q1 = *rq;
        int s1 = 0;
        q1.nthreads = 1;
        if (!est__check_plan_real(h->tcbw[0], rc, &q1, &s1, why, whylen))
        {
            rep->status = EST_NOPLAN;
            vfft_destroy((vfft_plan)h);
            h = NULL;
        }
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

/* THE FRONT DOOR'S ESTIMATE CREATE OF A REAL TRANSFORM (_vfft_create_estimate, for R2C and C2R at rank 1): one
 * transform or a contiguous batch of them (a lane-major batch is real_estimate_lm_il.h's): the request, the recipe,
 * the build. NULL with the reason on the error record (vfft_error). */
static vfft_plan _vfft_create_estimate_real(const est_machine_t *m, const vfft_cfg_t *cfg)
{
    est_request_t rq;
    est_real_recipe_t rc;
    char why[640];
    struct vfft_plan_s *h;
    if (cfg->dims > 1 || (cfg->howmany > 1 && cfg->batch_geom != VFFT_BATCH_CONTIGUOUS))
    {   /* _vfft_create_estimate routes these elsewhere first */
        _vfft_warn("vfft_create: VFFT_FLAG_ESTIMATE plans one R2C or C2R transform at rank 1 here, or a contiguous batch");
        return NULL;
    }
    memset(&rq, 0, sizeof rq);
    rq.N = cfg->n[0];
    rq.transform = cfg->transform;
    rq.order = cfg->order == VFI_ORDER_SCRAMBLED ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
    rq.placement = cfg->placement == VFFT_INPLACE ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
    rq.nthreads = cfg->nthreads > 1 ? cfg->nthreads : 1;
    rq.howmany = cfg->howmany > 1 ? cfg->howmany : 1;
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
