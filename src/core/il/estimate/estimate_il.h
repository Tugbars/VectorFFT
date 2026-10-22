/* estimate_il.h — THE ESTIMATE: the plan of a 1D interleaved c2c transform chosen by formulas,
 * with nothing measured (VFFT_FLAG_ESTIMATE, include/vfft.h).
 *
 * A planning method the caller selects, never a fallback. From N, the request and this machine's
 * cache sizes, est_plan_1d computes the row a race would have banked for the cell: the route and
 * every parameter of it (the recipe). est_put_rows writes that recipe as rows, with the store's own
 * writers, into a store the build holds in memory (il/estimate/c2c_estimate_il.h); the door's replay of
 * a banked verdict then builds the plan. No wisdom store is read or written.
 *
 * SCOPE: one 1D interleaved c2c transform (howmany 1), either order, either placement, any thread
 * count. The front door refuses every other request under the flag (vfft_api.h).
 *
 * THE FORMULAS ARE PURE: est_plan_1d reads the machine and the request only -- no clock, no store,
 * no allocation. The facts they need are the library's own: the radix lists of the IL registry
 * (VFFT_IL_*_RADICES), the bands and ceilings of ztt.h, k1_fourstep_band.h and policy_il.h, the tile
 * laws of ztt.h, the chain3 seed of il2p.h. A law or a table that lives inside an engine's or the
 * planner's function body is mirrored here with its source. The build's pre-flight runs the door's
 * own constructors on the recipe before the door sees it, so a recipe a registry change broke is
 * refused there, never raced.
 *
 * A T > 1 request keeps the route of T = 1 and adds the threaded arm; the arms are the rules the
 * raced rows follow at T = 8, used at every T > 1.
 *
 * POSITION IN vfft.c: after the IL engines, the K=1 planner, the four-step and the store's row
 * writers (vw2_oop_bank_k1_il, vw2_prime_method_bank, _k1fs_child_keys); c2c_estimate_il.h
 * includes it. */
#ifndef VFFT_IL_ESTIMATE_IL_H
#define VFFT_IL_ESTIMATE_IL_H

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

/* ── the machine, the request, the recipe ───────────────────────────────── */

/* The machine the formulas plan for, read once per create (_vfft_create_estimate). */
typedef struct est_machine_s
{
    int lanes;  /* VFFT_IL_VW (il/isa/il_isa.h): the formulas follow this build's laws, so
                 * est_plan_1d refuses any other count */
    long l1d;   /* bytes, vfft_cpu_l1d_bytes() (cpu_cache.h:560): the P-core's L1d */
    long l2;    /* bytes, vfft_cpu_l2_bytes() (cpu_cache.h:564): the four-step's band width */
    long l3;    /* bytes, vfft_cpu_l3_bytes() (cpu_cache.h:569), shared, 0 = not reported: the
                 * threaded four-step's super-band */
    int cores;  /* vfft_topology()->ncores: reported, no rule reads it */
} est_machine_t;

/* The request: the row's key. The values are the public enums' (include/vfft.h). */
typedef struct est_request_s
{
    int N;          /* vfft_config_t.n[0] */
    int transform;  /* VFFT_C2C; est_plan_1d refuses any other */
    int order;      /* VFFT_ORDER_NATURAL | VFFT_ORDER_SCRAMBLED -> ord=nat | scr */
    int placement;  /* VFFT_PLACEMENT_OUT | VFFT_PLACEMENT_IN -> place=oop | ip */
    int nthreads;   /* T; 0 or 1 = one thread -> nthreads=T on a T > 1 row */
} est_request_t;

/* The recipe: the c2c K=1 row's payload, each field named as the record names it
 * (vfft_oop_il_entry_t, wisdom2_oop_il.h:24-49), row tokens in [brackets]. Zero = absent. */
typedef struct est_recipe_s
{
    int k1_il_route;               /* [il_route] VFFT_K1_IL_* (common/abi/route_ids.h): MONO, 2P_PURE,
                                    * CHAIN3, FLAT, ZTT, FS, PRIME */
    int il_R1, il_R2;              /* [il_pair] the pair, mid R1 x leaf R2; chain3 (A B) x leaf, as the
                                    * planner banks it (dp_planner_il.h:2219-2221); the four-step its
                                    * one-thread split N1 x N2 */
    int il_c3[3];                  /* [il_chain] chain3 {leaf, A, B} */
    int il_fl[VFFT_ILFD_MAX_K];    /* [il_flat] the flat DIT's chain, leaf first */
    int il_fl_n;
    char il_flf[24];               /* [il_forms] the forms of stages 1..K-1, t|m|n|o '.'-joined;
                                    * "" = the create's defaults */
    int il_tw;                     /* [il_tw] flat: a stage's block span; ZTURN-T: the tile;
                                    * 0 = untiled */
    int il_zt[VFFT_ZTT_MAX_NF];    /* [il_ztt] ZTURN-T's chain; the threaded four-step's super-band
                                    * chain [il_mtsb] */
    int il_zt_n;
    int il_kv;                     /* [il_kv] the forward forms, always written (il_kv_raced = 1):
                                    * pair mid | leaf<<4, chain3 A | B<<4 | leaf<<8, else 0 */
    int il_bkv;                    /* [il_kv] of the dir=bwd row: a pair's or chain3's backward
                                    * forms (pair mid | leaf<<4, chain3 leaf<<8) */
    int il_mt;                     /* [il_mt] T > 1. Flat, ZTURN-T, prime: the threaded arm, 0 serial,
                                    * 1 BLOCKS, 2 TILES (k1_commit.h:1317, 1397, 1448). The
                                    * four-step: the threaded split's N1 (k1_commit.h:1543-1551) */
    int il_mt_tw;                  /* [il_mt_tw] T > 1, the flat's TILES arm: its width */
    char fs[512];                  /* the four-step's 2D child as the 2D row's tokens, fs_*, with its row
                                    * plan at N2 as fs_row_* and that plan's backward twin as
                                    * fs_row_bwd_*: the four-step route's child, or the prime cell's
                                    * four-step inner's */
    struct
    {
        int method;                /* [eng=rader|bluestein] 1 Rader, 2 Bluestein */
        char in_kind[8];           /* [in] 2p | 3p | ztt | fs */
        char in_shape[64];         /* [in_sh] R1.R2 | R2.A.B | the ZTURN-T chain | N1.N2 */
        int in_tw;                 /* [in_tw] the ZTURN-T inner's tile, 0 = untiled */
    } prime;                       /* the prime cell's method row (vw2_prime_method_bank,
                                    * wisdom2_oop_il.h:570); its route row says il_route=prime il_kv=0 */
} est_recipe_t;

/* ── code facts ─────────────────────────────────────────────────────────── */

#define EST__R(R) R,
/* the solo kernels, the MONO route's lengths (il_solo_registry.h:38) */
static const int est_n1_list[] = { VFFT_IL_N1_PAIR_RADICES(EST__R) };
/* the pair's radices: the leaf resolver's list, which the planner's pair and chain3 enumerators draw
 * from (dp_planner_il.h:1587, 1689); the mid and the chain3 mids have the same list
 * (VFFT_IL_T2_PAIR_RADICES, VFFT_IL_T2TG_BWD_RADICES) */
static const int est_pair_list[] = { VFFT_IL_N1T_PAIR_RADICES(EST__R) };
/* the odd-composite Cooley-Tukey twins, form 5, by slot (il2p.h:786-800, 842-856, 916-923, 950-957) */
static const int est_ct_mid_fwd[] = { VFFT_IL_T2_CT_FWD_RADICES(EST__R) };
static const int est_ct_leaf_fwd[] = { VFFT_IL_N1T_CT_FWD_RADICES(EST__R) };
static const int est_ct_mid_bwd[] = { VFFT_IL_T2T_CT_BWD_RADICES(EST__R) };
static const int est_ct_leaf_bwd[] = { VFFT_IL_N1_CT_BWD_RADICES(EST__R) };
#undef EST__R
/* the pair list's primes from 23 up: a pair or chain3 keeps such a radix out of the leaf */
static const int est_large_list[] = { 23, 29, 31, 37, 41, 43, 47 };
/* ZTURN-T's odd mids in the greedy order of its odd band (ztt.h:577; dp_planner_il.h:1482) */
static const int est_odd_mids[] = { 15, 9, 7, 5, 3 };
/* the 2D tier's column pool, its odd radices in the enumerator's order (il2d_cols.h:166-178) */
static const int est_odd2d[] = { 47, 43, 41, 37, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3 };
/* the ZTURN-T tile widths the planner races beside untiled (dp_planner_il.h:1455) */
#define EST_ZTT_TILE_LO 1024
#define EST_ZTT_TILE_HI 2048

#define EST_NELEM(a) ((int)(sizeof(a) / sizeof((a)[0])))

static inline int est_in(const int *list, int n, long long R)
{
    int i;
    for (i = 0; i < n; i++)
        if (list[i] == R)
            return 1;
    return 0;
}

/* ── small arithmetic ───────────────────────────────────────────────────── */

static inline int est_is_pow2(long long n) { return n > 0 && (n & (n - 1)) == 0; }
static inline int est_lg(long long n)
{
    int k = 0;
    while (n > 1) { n >>= 1; k++; }
    return k;
}
static inline int est_v2(long long n)
{
    int a = 0;
    if (n <= 0) return 0;
    while ((n & 1) == 0) { n >>= 1; a++; }
    return a;
}
/* the largest prime factor; 1 for n < 2 */
static inline long long est_lpf(long long n)
{
    long long d, best = 1;
    if (n < 2) return 1;
    for (d = 2; d * d <= n; d++)
        while (n % d == 0) { best = d; n /= d; }
    return n > 1 ? n : best;
}
/* 2^n in the fewest stages over {8, 4}: ceil(n/3) stages, f of them radix 4 (n >= 2) */
static inline void est_mincounts(int n, int *e8, int *f4)
{
    const int k = (n + 2) / 3, f = 3 * k - n;
    *e8 = k - f;
    *f4 = f;
}
static inline long long est_prod(const int *c, int n)
{
    long long p = 1;
    int i;
    for (i = 0; i < n; i++) p *= c[i];
    return p;
}

/* ── text ───────────────────────────────────────────────────────────────── */

typedef struct { char *b; size_t cap, len; } est_sb_t;   /* len counts what would be written */
static inline void est_sb_init(est_sb_t *s, char *b, size_t cap)
{
    s->b = b; s->cap = cap; s->len = 0;
    if (b && cap) b[0] = 0;
}
static inline void est_sb_add(est_sb_t *s, const char *fmt, ...)
{
    va_list ap;
    char *dst = NULL;
    size_t room = 0;
    int n;
    if (s->b && s->len < s->cap) { dst = s->b + s->len; room = s->cap - s->len; }
    va_start(ap, fmt);
    n = vsnprintf(dst, room, fmt, ap);
    va_end(ap);
    if (n > 0) s->len += (size_t)n;
}
static inline void est_sb_chain(est_sb_t *s, const int *c, int n)   /* "a.b.c" */
{
    int i;
    for (i = 0; i < n; i++) est_sb_add(s, "%s%d", i ? "." : "", c[i]);
}
static inline int est_refuse(char *why, size_t whylen, const char *fmt, ...)
{
    if (why && whylen)
    {
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(why, whylen, fmt, ap);
        va_end(ap);
    }
    return -1;
}

/* ── the machine ────────────────────────────────────────────────────────── */

/* s1 = floor(log2(L1d / 16)): the largest power of two of complexes the L1d holds (11 at 32-63 KiB) */
static inline int est_s1(const est_machine_t *m) { return est_lg(m->l1d / 16); }
/* a tile the L1 formula puts off the planner's raced widths is clamped onto them (at an L1d of
 * 32-63 KiB the formula gives 1024 and 2048 and nothing moves); 0 (untiled) stays 0 */
static inline int est_ztt_clamp(long long t)
{
    if (t <= 0) return 0;
    return t < EST_ZTT_TILE_LO ? EST_ZTT_TILE_LO : (t > EST_ZTT_TILE_HI ? EST_ZTT_TILE_HI : (int)t);
}
/* a transform stays serial at T > 1 while its complex data fits two L1s: 16 Nc <= 2 L1d */
static inline int est_serial(const est_machine_t *m, long long Nc) { return 16LL * Nc <= 2LL * m->l1d; }
/* a ZTURN-T length at this ISA: a power of two from the policy's floor to the ceiling, or the
 * odd band (policy_il.h:63-67, 85-88) */
static inline int est_ztt_len(long long M)
{
    if (M < 2 || M > VFFT_ZTT_MAX_N) return 0;
    return (est_is_pow2(M) && M >= VFFT_POLICY_ZTT_MIN_N) || vfft_policy_ztt_odd_band((int)M);
}

/* ── the pair forms ─────────────────────────────────────────────────────── */

/* the forward nibble of a pair slot: the Cooley-Tukey twin at an odd composite (5), the tangent
 * interior at 8 and 16 (3), at 32 the tangent as the mid (3) and its T256 store edge as the leaf
 * (4), else the create's default (0); the kernels are il2p.h:768-868 */
static inline int est_nib_fwd(int R, int leaf)
{
    if (leaf ? est_in(est_ct_leaf_fwd, EST_NELEM(est_ct_leaf_fwd), R) : est_in(est_ct_mid_fwd, EST_NELEM(est_ct_mid_fwd), R))
        return 5;
    if (R == 8 || R == 16) return 3;
    if (R == 32) return leaf ? 4 : 3;
    return 0;
}
/* the backward nibble: the twin at an odd composite (5), the tangent at 8 and 16 in either slot and
 * at 32 as the leaf (3); radix 32 has no backward tangent mid (il2p.h:892-957) */
static inline int est_nib_bwd(int R, int leaf)
{
    if (leaf ? est_in(est_ct_leaf_bwd, EST_NELEM(est_ct_leaf_bwd), R) : est_in(est_ct_mid_bwd, EST_NELEM(est_ct_mid_bwd), R))
        return 5;
    if (R == 8 || R == 16) return 3;
    if (R == 32 && leaf) return 3;
    return 0;
}

/* ── ZTURN-T ────────────────────────────────────────────────────────────── */

/* the create's chain laws (_ztt_create, ztt.h:344-365): 2..7 stages, the ends in {4, 8}, the mids in
 * {4, 8, 3, 5, 7, 9, 15}, product N, (N / R0) % 4 == 0; at 8 lanes (N / R0) % 8 == 0, the natural
 * class R0 % 8 == 0, the scrambled class R_last % 8 == 0 and (N / R_last) % 8 == 0 */
static inline int est_ztt_ok(long long N, const int *c, int nf, int scr)
{
    int s;
    if (nf < 2 || nf > VFFT_ZTT_MAX_NF || N < 16 || N > VFFT_ZTT_MAX_N) return 0;
    for (s = 0; s < nf; s++)
    {
        const int R = c[s], mid = s > 0 && s < nf - 1;
        if (!(R == 4 || R == 8 || (mid && (R == 3 || R == 5 || R == 7 || R == 9 || R == 15)))) return 0;
    }
    if (est_prod(c, nf) != N || (N / c[0]) % 4) return 0;
    if (VFFT_IL_VW == 8)
    {
        if ((N / c[0]) % 8) return 0;
        if (!scr && c[0] % 8) return 0;
        if (scr && (c[nf - 1] % 8 || (N / c[nf - 1]) % 8)) return 0;
    }
    return 1;
}
/* the tile law of the order class (vfft_ztt_tile_legal_ord, ztt.h:609-620) */
static inline int est_ztt_tile_ok(long long N, const int *c, int nf, long long t, int scr)
{
    return t >= 0 && vfft_ztt_tile_legal_ord((int)N, c, nf, (size_t)t, scr);
}
/* the fewest-stage chain of 2^n: an ingest 8, the forced 4s, then 8s; 4.4 at 16 */
static inline int est_ztt_f0(int n, int *c)
{
    int e, f, k = 0, i;
    if (n < 2) return 0;
    est_mincounts(n, &e, &f);
    if (e == 0) { for (i = 0; i < f; i++) c[k++] = 4; return k; }
    c[k++] = 8;
    for (i = 0; i < f; i++) c[k++] = 4;
    for (i = 0; i < e - 1; i++) c[k++] = 8;
    return k;
}
/* fill(b): b/3 eights when 3 | b; else radix-4 stages, with one 8 second when b is odd. Returns the
 * count appended, -1 when b = 1 (no such fill). */
static inline int est_ztt_fill(int b, int *c)
{
    int k = 0, i;
    if (b <= 0) return 0;
    if (b % 3 == 0) { for (i = 0; i < b / 3; i++) c[k++] = 8; return k; }
    if (b % 2 == 0) { for (i = 0; i < b / 2; i++) c[k++] = 4; return k; }
    if (b < 5) return -1;
    c[k++] = 4;
    c[k++] = 8;
    for (i = 0; i < (b - 5) / 2; i++) c[k++] = 4;
    return k;
}
/* natural, out of place. Up to 2^s1 the fewest-stage chain, untiled. Above, the tiled head first:
 * s is the one value in {s1-2, s1-1, s1} with 3 | (n - s), the chain is 8, fill(s-3), then
 * (n-s)/3 eights, the tile 2^s (2^(s1-1) when s = s1-2), clamped. head (may be NULL) gets the
 * tiled head's stage count, the ingest 8 and fill(s-3), or 0 for the untiled shape: the scrambled
 * chain moves exactly that head (est_ztt_scr). */
static inline int est_ztt_nat_oop_h(const est_machine_t *m, int n, int *c, int *tw, int *head)
{
    const int s1 = est_s1(m);
    int s, k = 0, f, i;
    long long t;
    *tw = 0;
    if (head) *head = 0;
    if (n <= s1) return est_ztt_f0(n, c);
    for (s = s1 - 2; s <= s1; s++)
        if ((n - s) % 3 == 0) break;
    c[k++] = 8;
    f = est_ztt_fill(s - 3, c + k);
    if (f < 0) return est_ztt_f0(n, c);   /* an L1d so small that s - 3 = 1: the untiled shape */
    k += f;
    for (i = 0; i < (n - s) / 3; i++) c[k++] = 8;
    if (k > VFFT_ZTT_MAX_NF) return est_ztt_f0(n, c);
    if (head) *head = 1 + f;
    t = (s == s1 - 2) ? (1LL << (s1 - 1)) : (1LL << s);
    *tw = est_ztt_clamp(t);
    if (!est_ztt_tile_ok(1LL << n, c, k, *tw, 0)) *tw = 0;
    return k;
}
/* natural, in place: the fewest-stage chain; the tile is the longest chain prefix whose product is
 * 2^(s1-1) or 2^s1 (else 2^(s1-1)), clamped; untiled up to 2^s1 */
static inline int est_ztt_nat_ip(const est_machine_t *m, int n, int *c, int *tw)
{
    const int s1 = est_s1(m), k = est_ztt_f0(n, c);
    const long long wlo = 1LL << (s1 - 1), whi = 1LL << s1;
    long long p = 1, t = wlo;
    int i;
    *tw = 0;
    if (n <= s1) return k;
    for (i = 0; i < k - 1; i++)
    {
        p *= c[i];
        if (p == wlo || p == whi) t = p;
    }
    *tw = est_ztt_clamp(t);
    if (!est_ztt_tile_ok(1LL << n, c, k, *tw, 0)) *tw = 0;
    return k;
}
/* scrambled. Up to 2^s1 the forced 4s first, then 8s, untiled. Above, the natural out-of-place
 * chain with its tiled head moved to the end, because the plain schedule tiles the suffix
 * (ztt.h:606-620); at 8 lanes the head reversed so the chain ends in 8. The tile 2^(s1-1) up to
 * n = s1 + 2, else 2^s1 (clamped); the other width when the plain tile law refuses it, else
 * untiled. */
static inline int est_ztt_scr(const est_machine_t *m, int n, int *c, int *tw)
{
    const int s1 = est_s1(m);
    int nat[16], ntw, k, i, j, e, f, cut = 0;
    long long w[2];
    *tw = 0;
    if (n <= s1)
    {
        est_mincounts(n, &e, &f);
        k = 0;
        for (i = 0; i < f; i++) c[k++] = 4;
        for (i = 0; i < e; i++) c[k++] = 8;
        return k;
    }
    k = est_ztt_nat_oop_h(m, n, nat, &ntw, &cut);
    j = 0;
    for (i = cut; i < k; i++) c[j++] = nat[i];
    for (i = 0; i < cut; i++) c[j++] = VFFT_IL_VW == 8 ? nat[cut - 1 - i] : nat[i];
    w[0] = est_ztt_clamp(1LL << (s1 - 1));
    w[1] = est_ztt_clamp(1LL << s1);
    if (n > s1 + 2) { const long long x = w[0]; w[0] = w[1]; w[1] = x; }
    for (i = 0; i < 2; i++)
        if (est_ztt_tile_ok(1LL << n, c, k, w[i], 1)) { *tw = (int)w[i]; break; }
    return k;
}
/* the convolution form, a Rader or Bluestein inner's (the scrambled class, k1_commit.h:340-345):
 * 8s first, then the forced 4s; at 8 lanes the 4s first, then 8s (its last stage must be 8).
 * Untiled up to 2^s1, else 2^(s1-1) clamped; a width the plain tile law refuses leaves it
 * untiled. */
static inline int est_ztt_conv(const est_machine_t *m, int n, int *c, int *tw)
{
    int e, f, k = 0, i;
    est_mincounts(n, &e, &f);
    if (VFFT_IL_VW == 8)
    {
        for (i = 0; i < f; i++) c[k++] = 4;
        for (i = 0; i < e; i++) c[k++] = 8;
    }
    else
    {
        for (i = 0; i < e; i++) c[k++] = 8;
        for (i = 0; i < f; i++) c[k++] = 4;
    }
    *tw = n <= est_s1(m) ? 0 : est_ztt_clamp(1LL << (est_s1(m) - 1));
    if (!est_ztt_tile_ok(1LL << n, c, k, *tw, 1)) *tw = 0;
    return k;
}
/* 2^r as 8s then the forced 4s (r >= 2), or 4s then 8s; no stage for r < 2 */
static inline int est_bits(int r, int fours_first, int *c)
{
    int e, f, k = 0, i;
    if (r < 2) return 0;
    est_mincounts(r, &e, &f);
    if (fours_first) { for (i = 0; i < f; i++) c[k++] = 4; for (i = 0; i < e; i++) c[k++] = 8; }
    else { for (i = 0; i < e; i++) c[k++] = 8; for (i = 0; i < f; i++) c[k++] = 4; }
    return k;
}
/* the odd band. 4 lanes: the ingest 4, the odd part greedy, largest first, then the rest of the
 * power of two as 8s and the forced 4s; a chain past 7 stages ingests 8 instead. 8 lanes: the
 * natural chain ingests 8 (t0tp exists at 8 only there); the scrambled chain ends in 8 (tld), with
 * 4 as its ingest when the power of two is 2^7. Untiled. 0 = no chain within the laws. */
static inline int est_ztt_odd(long long N, int scr, int *c)
{
    int mids[8], nm = 0, a = est_v2(N), k, i, r, head, rest[16], nr;
    long long o = N >> est_v2(N);
    for (i = 0; i < EST_NELEM(est_odd_mids); i++)
        while (o % est_odd_mids[i] == 0 && nm < 8) { mids[nm++] = est_odd_mids[i]; o /= est_odd_mids[i]; }
    if (o != 1) return 0;
    if (VFFT_IL_VW == 8 && scr)
    {
        head = (a - 3 == 4) ? 4 : 8;
        nr = est_bits(a - est_lg(head), 1, rest);
    }
    else if (VFFT_IL_VW == 8)
    {
        head = 8;
        nr = est_bits(a - 3, 0, rest);
    }
    else
    {
        head = 4;
        nr = est_bits(a - 2, 0, rest);
        if (1 + nm + nr > VFFT_ZTT_MAX_NF)
        {
            head = 8;
            nr = est_bits(a - 3, 0, rest);
        }
    }
    if (nr < 1 || 1 + nm + nr > VFFT_ZTT_MAX_NF) return 0;
    k = 0;
    c[k++] = head;
    for (i = 0; i < nm; i++) c[k++] = mids[i];
    for (r = 0; r < nr; r++) c[k++] = rest[r];
    return est_ztt_ok(N, c, k, scr) ? k : 0;
}

/* ── the pair and chain3 ────────────────────────────────────────────────── */

/* the planner's pair (dp_planner_il.h:1587-1611): R2 the leaf, R1 = N / R2 the mid, both in the
 * pair list; no radix-64 slot at a power of two (the pow2 sunset, :1598-1604) */
static inline int est_pair_legal(long long N, int R1, int R2)
{
    if (R1 < 3 || R2 < 3 || R1 > 64 || R2 > 64 || (long long)R1 * R2 != N) return 0;
    if (!est_in(est_pair_list, EST_NELEM(est_pair_list), R1) ||
        !est_in(est_pair_list, EST_NELEM(est_pair_list), R2)) return 0;
    if (est_is_pow2(N) && (R1 == 64 || R2 == 64)) return 0;
    return 1;
}
static inline int est_large(int R) { return est_in(est_large_list, EST_NELEM(est_large_list), R); }
/* the pair: the most balanced legal pair (the smallest larger factor). When exactly one factor is a
 * prime from 23 up it is the mid R1; otherwise the larger factor is the leaf R2. 0 = none. */
static inline int est_pair_shape(long long N, int *R1, int *R2)
{
    int i, lo = 0, hi = 0;
    for (i = 0; i < EST_NELEM(est_pair_list); i++)
    {
        const int b = est_pair_list[i];
        int a, l, h;
        if (N % b) continue;
        if (N / b > 64) continue;
        a = (int)(N / b);
        if (!est_pair_legal(N, a, b)) continue;
        l = a < b ? a : b;
        h = a < b ? b : a;
        if (!hi || h < hi) { lo = l; hi = h; }
    }
    if (!hi) return 0;
    if (est_large(lo) != est_large(hi)) { *R1 = est_large(lo) ? lo : hi; *R2 = est_large(lo) ? hi : lo; }
    else { *R1 = lo; *R2 = hi; }
    return 1;
}
/* a convolution inner's pair at M: a power of two from 16 to 512 takes mid 4 up to 128, else 16;
 * any other M the pair above */
static inline int est_inner_pair(long long M, int *R1, int *R2)
{
    if (est_is_pow2(M) && M >= 16 && M <= 512)
    {
        *R1 = M <= 128 ? 4 : 16;
        *R2 = (int)(M / *R1);
        return est_pair_legal(M, *R1, *R2);
    }
    return est_pair_shape(M, R1, R2);
}
/* the chain3 enumerator's law (dp_planner_il.h:1674-1730; il2p.h:1576-1586): the leaf R2 in the pair
 * list, R1 = A B >= 9 and not a power of two, A and B in the pair list */
static inline int est_c3_legal(long long N, int R2, int A, int B)
{
    const long long R1 = (long long)A * B;
    if (R2 < 3 || A < 3 || B < 3 || R1 * R2 != N || R1 < 9 || est_is_pow2(R1)) return 0;
    return est_in(est_pair_list, EST_NELEM(est_pair_list), R2) &&
           est_in(est_pair_list, EST_NELEM(est_pair_list), A) &&
           est_in(est_pair_list, EST_NELEM(est_pair_list), B);
}
static inline double est_c3_spread(int a, int b, int c)   /* 3 x the variance of log radix */
{
    const double la = log((double)a), lb = log((double)b), lc = log((double)c), mu = (la + lb + lc) / 3.0;
    return (la - mu) * (la - mu) + (lb - mu) * (lb - mu) + (lc - mu) * (lc - mu);
}
/* chain3. The factors: the most balanced legal triple, the least variance of log radix (the first
 * in the enumerator's order on a tie: the leaf in list order, A ascending). The order: the leaf is
 * the largest even factor (else the largest that is not a prime from 23 up), the two mids follow
 * descending; when the enumerator refuses that order the mids swap; failing that, the legal order
 * of the same factors with the largest leaf. c3 = {R2, A, B}. */
static inline int est_chain3_shape(long long N, int c3[3])
{
    int i, A, have = 0, t[3], s[3], leaf = 0, li = -1, r0, r1;
    double best = 0;
    for (i = 0; i < EST_NELEM(est_pair_list); i++)
    {
        const int R2 = est_pair_list[i];
        long long R1;
        if (N % R2) continue;
        R1 = N / R2;
        for (A = 3; A <= R1 / 2; A++)
        {
            double sp;
            if (R1 % A || !est_c3_legal(N, R2, A, (int)(R1 / A))) continue;
            sp = est_c3_spread(R2, A, (int)(R1 / A));
            if (!have || sp < best - 1e-12) { best = sp; t[0] = R2; t[1] = A; t[2] = (int)(R1 / A); have = 1; }
        }
    }
    if (!have) return 0;
    s[0] = t[0]; s[1] = t[1]; s[2] = t[2];   /* the factors, ascending */
    for (i = 1; i < 3; i++)
        for (A = i; A > 0 && s[A] < s[A - 1]; A--) { const int x = s[A]; s[A] = s[A - 1]; s[A - 1] = x; }
    for (i = 0; i < 3; i++)
        if (s[i] % 2 == 0 && s[i] >= leaf) { leaf = s[i]; li = i; }
    if (li < 0)
    {
        for (i = 0; i < 3; i++)
            if (!est_large(s[i]) && s[i] >= leaf) { leaf = s[i]; li = i; }
        if (li < 0) { leaf = s[2]; li = 2; }
    }
    r0 = s[li == 2 ? 1 : 2];   /* the other two, descending */
    r1 = s[li == 0 ? 1 : 0];
    if (est_c3_legal(N, leaf, r0, r1)) { c3[0] = leaf; c3[1] = r0; c3[2] = r1; return 1; }
    if (est_c3_legal(N, leaf, r1, r0)) { c3[0] = leaf; c3[1] = r1; c3[2] = r0; return 1; }
    {   /* the legal order of these factors with the largest leaf, the first in the enumerator's order */
        int found = 0;
        for (i = 0; i < EST_NELEM(est_pair_list); i++)
        {
            const int R2 = est_pair_list[i];
            long long R1;
            if (N % R2) continue;
            R1 = N / R2;
            for (A = 3; A <= R1 / 2; A++)
            {
                int u[3], q, w;
                if (R1 % A || !est_c3_legal(N, R2, A, (int)(R1 / A))) continue;
                u[0] = R2; u[1] = A; u[2] = (int)(R1 / A);
                for (q = 1; q < 3; q++)
                    for (w = q; w > 0 && u[w] < u[w - 1]; w--) { const int x = u[w]; u[w] = u[w - 1]; u[w - 1] = x; }
                if (u[0] != s[0] || u[1] != s[1] || u[2] != s[2]) continue;
                if (!found || R2 > c3[0]) { c3[0] = R2; c3[1] = A; c3[2] = (int)(R1 / A); found = 1; }
            }
        }
        return found;
    }
}

/* ── the flat DIT ───────────────────────────────────────────────────────── */

/* the odd prime factors of o (o odd), 3.3 merged into 9; -1 when a factor passes the chain primes */
static inline int est_odd_stages(long long o, int *c)
{
    int k = 0, p, n3 = 0, i;
    for (p = 3; p <= VFFT_POLICY_CHAIN_MAX_PRIME; p += 2)
        while (o % p == 0)
        {
            o /= p;
            if (p == 3) n3++;
            else c[k++] = p;
        }
    if (o != 1) return -1;
    for (i = 0; i < n3 / 2; i++) c[k++] = 9;
    if (n3 % 2) c[k++] = 3;
    return k;
}
static inline void est_sort_desc(int *c, int n)
{
    int i, j;
    for (i = 1; i < n; i++)
        for (j = i; j > 0 && c[j] > c[j - 1]; j--) { const int x = c[j]; c[j] = c[j - 1]; c[j - 1] = x; }
}
static inline void est_sort_asc(int *c, int n)
{
    int i, j;
    for (i = 1; i < n; i++)
        for (j = i; j > 0 && c[j] < c[j - 1]; j--) { const int x = c[j]; c[j] = c[j - 1]; c[j - 1] = x; }
}
/* a chain past the flat DIT's 10 stages (VFFT_ILFD_MAX_K): two 5s merge into 25; failing that a 3
 * and a 5 into 15, else a 3 and a 7 into 21, the merged radix in the place of the first of its two
 * factors. Returns the new stage count, or -1. */
static inline int est_merge10(int *c, int K)
{
    static const int pr[3][3] = { { 5, 5, 25 }, { 5, 3, 15 }, { 7, 3, 21 } };
    while (K > VFFT_ILFD_MAX_K)
    {
        int q, done = 0;
        for (q = 0; q < 3 && !done; q++)
        {
            int i, j = -1, k = -1;
            for (i = 0; i < K; i++)
            {
                if (j < 0 && (c[i] == pr[q][0] || c[i] == pr[q][1])) { j = i; continue; }
                if (j >= 0 && k < 0 && c[i] == (c[j] == pr[q][0] ? pr[q][1] : pr[q][0])) k = i;
            }
            if (j < 0 || k < 0) continue;
            c[j] = pr[q][2];
            for (i = k; i < K - 1; i++) c[i] = c[i + 1];
            K--;
            done = 1;
        }
        if (!done) return -1;
    }
    return K;
}
/* the chain: a lone 2 leads when N = 2 (mod 4) (chain_search.h:239-243), otherwise the largest radix
 * leads; then the 8s and the forced 4s, then the odd primes descending with 3.3 merged into 9; at
 * most 10 stages. No Rader stage is emitted, so the scrambled class always builds
 * (il_flatdit.h:313). 0 = no chain (a factor above 47). */
static inline int est_flat_chain(long long N, int *c)
{
    int st[64], ns = 0, a = est_v2(N), k = 0, i, lead = -1, nodd;
    if (a >= 2) ns = est_bits(a, 0, st);
    nodd = est_odd_stages(N >> a, st + ns);
    if (nodd < 0) return 0;
    ns += nodd;
    if (a == 1) c[k++] = 2;
    else
    {
        for (i = 0; i < ns; i++)
            if (lead < 0 || st[i] > st[lead]) lead = i;
        if (lead < 0) return 0;
        c[k++] = st[lead];
        for (i = lead; i < ns - 1; i++) st[i] = st[i + 1];
        ns--;
    }
    {   /* the pow2 stages (8s before 4s), then the odd ones descending */
        int p2[32], np2 = 0, od[32], nod = 0;
        for (i = 0; i < ns; i++)
        {
            if (st[i] == 4 || st[i] == 8 || st[i] == 16) p2[np2++] = st[i];
            else od[nod++] = st[i];
        }
        est_sort_desc(p2, np2);
        est_sort_desc(od, nod);
        for (i = 0; i < np2; i++) c[k++] = p2[i];
        for (i = 0; i < nod; i++) c[k++] = od[i];
    }
    if (k < 2) return 0;
    k = est_merge10(c, k);
    return k < 2 ? 0 : k;
}
/* the forms: t on every middle stage; the last stage t at K = 2, otherwise n, o in the natural
 * class from K = 5. Every pool radix has the t2csgn tail (il_flatdit.h:286-289). */
static inline void est_flat_forms(int K, int scr, char *f, size_t cap)
{
    est_sb_t s;
    int i;
    est_sb_init(&s, f, cap);
    if (K < 2) return;
    if (K == 2) { est_sb_add(&s, "t"); return; }
    for (i = 0; i < K - 2; i++) est_sb_add(&s, "t.");
    est_sb_add(&s, "%s", (!scr && K >= 5) ? "o" : "n");
}
/* the tile: untiled while 16 N <= 2 L1d; otherwise the widest block span R_s D_s of a stage s in
 * [1, K-2] that is no tail (D_s > VFFT_ILFD_TAIL_D) with 16 R_s D_s <= L1d (vfft_ilfd_apply_tw,
 * il_flatdit.h:904-916) */
static inline int est_flat_tw(const est_machine_t *m, long long N, const int *c, int K)
{
    long long best = 0;
    int s;
    if (16LL * N <= 2LL * m->l1d) return 0;
    for (s = 1; s <= K - 2; s++)
    {
        const long long D = est_prod(c + s + 1, K - s - 1), w = c[s] * D;
        if (D > VFFT_ILFD_TAIL_D && 16LL * w <= m->l1d && w > best) best = w;
    }
    return (int)best;
}

/* ── the four-step ──────────────────────────────────────────────────────── */

/* the column chain at N1 over the 2D tier's pool, at most 4 stages (il2d_cols.h:164-241): 8s with
 * one 4 or 16, then the odd part largest first, then a closing radix. A lone 2 closes the chain
 * folded into the smallest odd factor p whose 2p is a closing radix (6, 10, 14, 22, 26), else as 2
 * itself (the closing-only radices, il2d_cols.h:160-163). A chain past 4 stages folds its power of
 * two into 64, 32 or 16 stages. Returns the stage count, 0 = none. */
static inline int est_fs_col_chain(long long N1, int *c)
{
    int odd[32], no = 0, pw[32], np = 0, close = 0, a = est_v2(N1), i, k, big;
    long long o = N1 >> est_v2(N1);
    for (i = 0; i < EST_NELEM(est_odd2d); i++)
        while (o % est_odd2d[i] == 0 && no < 32) { odd[no++] = est_odd2d[i]; o /= est_odd2d[i]; }
    if (o != 1) return 0;
    if (a == 1)
    {
        int bi = -1;
        for (i = 0; i < no; i++)
        {
            const int q = 2 * odd[i];
            if ((q == 6 || q == 10 || q == 14 || q == 22 || q == 26) && (bi < 0 || odd[i] < odd[bi])) bi = i;
        }
        if (bi >= 0)
        {
            close = 2 * odd[bi];
            for (i = bi; i < no - 1; i++) odd[i] = odd[i + 1];
            no--;
        }
        else close = 2;
        a = 0;
    }
    if (a > 0)
    {
        const int q = a / 3, r = a % 3;
        if (r == 1 && q < 1) return 0;
        for (i = 0; i < (r == 1 ? q - 1 : q); i++) pw[np++] = 8;
        if (r == 2) pw[np++] = 4;
        if (r == 1) pw[np++] = 16;
    }
    if (np + no + (close != 0) > 4 && np > 0)
    {   /* the power of two folded into 64, 32 or 16 stages */
        for (big = 64; big >= 16; big /= 2)
        {
            const int lb = est_lg(big), q = a / lb, r = a % lb;
            int cand[32], nc = 0;
            if (r == 1)
            {
                if (q < 1) continue;
                for (i = 0; i < q - 1; i++) cand[nc++] = big;
                cand[nc++] = big / 2;
                cand[nc++] = 4;
            }
            else
            {
                for (i = 0; i < q; i++) cand[nc++] = big;
                if (r >= 2) cand[nc++] = 1 << r;
            }
            if (nc && est_prod(cand, nc) == (1LL << a))
            {
                if (nc < np) { np = nc; memcpy(pw, cand, sizeof(int) * (size_t)nc); }
                break;
            }
        }
    }
    k = 0;
    for (i = 0; i < np; i++) c[k++] = pw[i];
    for (i = 0; i < no; i++) c[k++] = odd[i];
    if (close) c[k++] = close;
    return (k >= 1 && k <= 4 && est_prod(c, k) == N1) ? k : 0;
}
/* the split N1 x N2. A power of two: 512 x 1024 at 2^19, N1 = 4096 from 2^20 (threaded: N / 2048
 * up to 2^20, then 4096). Any other N: the most balanced split with both sides in
 * [VFFT_K1FS_SIDE_MIN, VFFT_K1FS_SIDE_MAX], N1 a column chain and N2 no prime cell
 * (k1_fourstep.h:86-158); on a tie the larger N1 (threaded: the smaller). 0 = none. */
static inline int est_fs_split(long long N, int threaded, int *N1, int *N2)
{
    long long d, best = 0;
    int c[8], b1 = 0, b2 = 0;
    if (N < VFFT_K1FS_MIN_N || N > VFFT_K1FS_MAX_N) return 0;
    if (est_is_pow2(N))
    {
        if (N < (1LL << 19)) return 0;
        if (threaded) *N1 = N <= (1LL << 20) ? (int)(N / 2048) : 4096;
        else *N1 = N == (1LL << 19) ? 512 : 4096;
        *N2 = (int)(N / *N1);
        return 1;
    }
    for (d = VFFT_K1FS_SIDE_MIN; d <= VFFT_K1FS_SIDE_MAX; d++)
    {
        const long long e = N / d, hi = d > e ? d : e;
        if (N % d || e < VFFT_K1FS_SIDE_MIN || e > VFFT_K1FS_SIDE_MAX) continue;
        if (!est_fs_col_chain(d, c) || vfft_policy_prime_cell((int)e)) continue;
        if (!best || hi < best || (hi == best && (threaded ? d < b1 : d > b1)))
        {
            best = hi;
            b1 = (int)d;
            b2 = (int)e;
        }
    }
    if (!best) return 0;
    *N1 = b1;
    *N2 = b2;
    return 1;
}
/* fs_wl, the band width: the widest column-stage span w < N1 with 16 w N2 <= L2 -- the 2D tier's L2
 * gate (il2d_tier.h:1992-2010, vfft_policy_fits_l2 policy_cache.h:33-37) over its spans, the
 * suffix products of the chain from stage 1 (il2d_cols.h:1054-1062), w >= 8
 * (vfft_policy_il2d_band_ok, policy_il.h:587-590); 0 when none qualifies */
static inline int est_fs_wl(const est_machine_t *m, const int *c, int nst, long long N1, long long N2)
{
    long long L = N1, best = 0;
    int s;
    for (s = 0; s < nst; s++)
    {
        if (s >= 1 && L >= 8 && L < N1 && m->l2 > 0 && 16LL * L * N2 <= (long long)m->l2 && L > best)
            best = L;
        L /= c[s];
    }
    return (int)best;
}

/* ── the c2c route row's tokens ─────────────────────────────────────────── */

static inline int est_c2c(const est_machine_t *m, long long N, int scr, int ip, int T, est_recipe_t *o,
                          char *why, size_t whylen);

/* the c2c K=1 row's tokens in vw2_oop_rec_k1_il's order (wisdom2_oop_il.h:417-500), each name
 * prefixed by pfx */
static inline void est_c2c_tokens(est_sb_t *s, const est_recipe_t *r, const char *pfx)
{
    est_sb_add(s, "%seng=k1 %sil_route=%s", pfx, pfx, vw2_oop_il_name[r->k1_il_route]);
    if (r->il_R1 || r->il_R2) est_sb_add(s, " %sil_pair=%d.%d", pfx, r->il_R1, r->il_R2);
    if (r->k1_il_route == VFFT_K1_IL_CHAIN3)
        est_sb_add(s, " %sil_chain=%d.%d.%d", pfx, r->il_c3[0], r->il_c3[1], r->il_c3[2]);
    if (r->k1_il_route == VFFT_K1_IL_FLAT)
    {
        est_sb_add(s, " %sil_flat=", pfx);
        est_sb_chain(s, r->il_fl, r->il_fl_n);
        if (r->il_flf[0]) est_sb_add(s, " %sil_forms=%s", pfx, r->il_flf);
        if (r->il_tw > 0) est_sb_add(s, " %sil_tw=%d", pfx, r->il_tw);
    }
    if (r->k1_il_route == VFFT_K1_IL_ZTT)
    {
        est_sb_add(s, " %sil_ztt=", pfx);
        est_sb_chain(s, r->il_zt, r->il_zt_n);
        if (r->il_tw > 0) est_sb_add(s, " %sil_tw=%d", pfx, r->il_tw);
    }
    est_sb_add(s, " %sil_kv=%d", pfx, r->il_kv);
}
/* the dir=bwd row's tokens (vw2_oop_rec_k1_bwd, wisdom2_oop_il.h:359-405), with il_chain for chain3
 * as the planner adds it (dp_planner_il.h:2276-2286) */
static inline void est_bwd_tokens(est_sb_t *s, const est_recipe_t *r, const char *pfx)
{
    est_sb_add(s, "%seng=k1 %sil_route=%s %sil_pair=%d.%d %sil_kv=%d", pfx, pfx, vw2_oop_il_name[r->k1_il_route],
               pfx, r->il_R1, r->il_R2, pfx, r->il_bkv);
    if (r->k1_il_route == VFFT_K1_IL_CHAIN3)
        est_sb_add(s, " %sil_chain=%d.%d.%d", pfx, r->il_c3[0], r->il_c3[1], r->il_c3[2]);
}

/* ── the four-step's child ──────────────────────────────────────────────── */

/* The four-step's 2D child at N1 x N2 as the 2D row's tokens: fs_chain, fs_wl, fs_tf = (fs_wl > 0)
 * (il2d_tier.h:3936), the switches fs_ro=0 fs_sw=0 fs_rbk=8 fs_turn=0 fs_csk=0, and two verdicts the
 * 2D tier races when its row lacks them (below); at T > 1 fs_cmt=1 with fs_mtarm=1 fs_msw=128 when
 * fs_wl = 0, else 0 and 0. Then the row plan at N2 -- this procedure's c2c cell, natural, in place,
 * one thread, the row plan's own key (k1_fourstep.h:226) -- as fs_row_*, and its backward twin as
 * fs_row_bwd_* (key :229) when the row plan is a pair or chain3 with backward forms. */
static inline int est_fs_child(const est_machine_t *m, int N1, int N2, int T, char *fs, size_t cap,
                               char *why, size_t whylen)
{
    est_recipe_t row;
    est_sb_t s;
    int c[8], nst, wl;
    nst = est_fs_col_chain(N1, c);
    if (!nst) return est_refuse(why, whylen, "the four-step %dx%d has no column chain at %d", N1, N2, N1);
    wl = est_fs_wl(m, c, nst, N1, N2);
    if (est_c2c(m, N2, 0, 1, 1, &row, why, whylen) != 0) return -1;
    est_sb_init(&s, fs, cap);
    est_sb_add(&s, "fs_chain=");
    est_sb_chain(&s, c, nst);
    est_sb_add(&s, " fs_wl=%d fs_tf=%d fs_ro=0", wl, wl > 0);
    {   /* The two verdicts spell the column chain fs_chain with its kernels at the construction
         * table's defaults, in the tokens a race banks when that choice wins:
         * - fs_blu=0: a chain with an odd radix races itself against a Bluestein column unless the
         *   row carries the N1 arm's verdict; 0 = the chain won (il2d_tier.h:3607-3650). A power of
         *   two never races it.
         * - fs_forms: a stage whose radix has rival column forms races them unless the row carries
         *   forms= (il2d_tier.h:3218-3250). The rivals are vfft_il2p_col_forms (il2p.h:481-487, a
         *   c2c column: radix 32 {b48, b84}, every other radix one form); the race names each stage
         *   by its first form ("b48", "b88" for 64, "-" else; il2d_tier.h:2483, 2597-2598). */
        int i, odd = 0, multi = 0;
        for (i = 0; i < nst; i++)
        {
            if (c[i] & 1) odd = 1;
            if (c[i] == 32) multi = 1;
        }
        if (odd) est_sb_add(&s, " fs_blu=0");
        if (multi)
        {
            est_sb_add(&s, " fs_forms=");
            for (i = 0; i < nst; i++)
                est_sb_add(&s, "%s%s", i ? "." : "", c[i] == 32 ? "b48" : c[i] == 64 ? "b88" : "-");
        }
    }
    est_sb_add(&s, " fs_sw=0 fs_rbk=8 fs_turn=0 fs_csk=0");
    if (T > 1)
        est_sb_add(&s, " fs_cmt=1 fs_mtarm=%d fs_msw=%d", wl == 0, wl == 0 ? 128 : 0);
    est_sb_add(&s, " ");
    est_c2c_tokens(&s, &row, "fs_row_");
    if ((row.k1_il_route == VFFT_K1_IL_2P_PURE || row.k1_il_route == VFFT_K1_IL_CHAIN3) && row.il_bkv)
    {
        est_sb_add(&s, " ");
        est_bwd_tokens(&s, &row, "fs_row_bwd_");
    }
    if (s.len >= cap)
        return est_refuse(why, whylen, "the four-step %dx%d: its child's tokens pass %u bytes", N1, N2, (unsigned)cap);
    return 0;
}
/* The four-step route: il_pair = the one-thread split (in place the same split), il_kv = 0 (the
 * transpose-sweep form). At T > 1, il_mt = the threaded split's N1 and the child is that split's own
 * (k1_commit.h:1538-1552, 1595-1620); il_mtsb = 8.64.8 when natural, 16 N passes the L3 or no L3 is
 * reported (vfft_policy_exceeds_l3, policy_cache.h:46-50), N1 = 4096 and N2 % 16 == 0
 * (k1_fourstep.h:955-958), else none. */
static inline int est_fs_route(const est_machine_t *m, long long N, int scr, int T, est_recipe_t *o,
                               char *why, size_t whylen)
{
    int N1, N2, N1T, N2T;
    if (!est_fs_split(N, 0, &N1, &N2)) return est_refuse(why, whylen, "N = %lld has no four-step split", N);
    o->k1_il_route = VFFT_K1_IL_FS;
    o->il_R1 = N1;
    o->il_R2 = N2;
    o->il_kv = 0;
    if (T <= 1) return est_fs_child(m, N1, N2, 1, o->fs, sizeof o->fs, why, whylen);
    if (!est_fs_split(N, 1, &N1T, &N2T))
        return est_refuse(why, whylen, "N = %lld has no four-step split for the threaded plan", N);
    o->il_mt = N1T;
    if (!scr && (m->l3 <= 0 || 16LL * N > (long long)m->l3) && N1T == 4096 && N2T % 16 == 0)
    {
        o->il_zt[0] = 8; o->il_zt[1] = 64; o->il_zt[2] = 8;
        o->il_zt_n = 3;
    }
    return est_fs_child(m, N1T, N2T, T, o->fs, sizeof o->fs, why, whylen);
}

/* ── the prime cell ─────────────────────────────────────────────────────── */

typedef struct { char kind[8], shape[64]; int tw, ztt; } est_inner_t;

static inline void est_inner_set(est_inner_t *in, const char *kind, const int *c, int n, int tw, int ztt)
{
    est_sb_t s;
    snprintf(in->kind, sizeof in->kind, "%s", kind);
    est_sb_init(&s, in->shape, sizeof in->shape);
    est_sb_chain(&s, c, n);
    in->tw = tw;
    in->ztt = ztt;
}
/* a four-step inner at M: the split and child of the four-step at M, the scrambled class, out of
 * place, one thread, the transpose-sweep form (k1_commit.h:276-307); the child goes in fs */
static inline int est_inner_fs(const est_machine_t *m, long long M, est_inner_t *in, char *fs, size_t cap,
                               char *why, size_t whylen)
{
    int c[2];
    if (!vfft_k1fs_band((int)M) || !est_fs_split(M, 0, &c[0], &c[1])) return 0;
    if (est_fs_child(m, c[0], c[1], 1, fs, cap, why, whylen) != 0) return -1;
    est_inner_set(in, "fs", c, 2, 0, 0);
    return 1;
}
/* a ZTURN-T inner at M, the scrambled class: the convolution chain at a power of two, the odd chain
 * in the odd band (untiled) */
static inline int est_inner_ztt(const est_machine_t *m, long long M, est_inner_t *in)
{
    int c[16], k, tw = 0;
    if (est_is_pow2(M) && M >= VFFT_POLICY_ZTT_MIN_N && M <= VFFT_ZTT_MAX_N)
        k = est_ztt_conv(m, est_lg(M), c, &tw);
    else if (M <= VFFT_ZTT_MAX_N && vfft_policy_ztt_odd_band((int)M))
        k = est_ztt_odd(M, 1, c);
    else
        return 0;
    if (!k || !est_ztt_ok(M, c, k, 1)) return 0;
    est_inner_set(in, "ztt", c, k, tw, 1);
    return 1;
}
/* Rader's inner at M = N - 1: a pair when M <= 4096 has one; else ZTURN-T; else the chain3 seed
 * (vfft_il3p_default_chain, il2p.h:1523; the prime pool holds it up to 4096, k1_commit.h:420-438);
 * else, in the four-step's band, the four-step. 1 = found, 0 = none, -1 = refused. */
static inline int est_rader_inner(const est_machine_t *m, long long M, est_inner_t *in, char *fs, size_t cap,
                                  char *why, size_t whylen)
{
    int c[3];
    if (M <= 4096 && est_inner_pair(M, &c[0], &c[1])) { est_inner_set(in, "2p", c, 2, 0, 0); return 1; }
    if (est_inner_ztt(m, M, in)) return 1;
    if (M <= 4096 && vfft_il3p_default_chain((int)M, &c[0], &c[1], &c[2])) { est_inner_set(in, "3p", c, 3, 0, 0); return 1; }
    return est_inner_fs(m, M, in, fs, cap, why, whylen);
}
/* an inner exists at M = N - 1 for Rader's second case: a pair or the chain3 seed up to 4096, or a
 * ZTURN-T length */
static inline int est_rader_inner_exists(long long M)
{
    int c[3];
    return (M <= 4096 && (est_inner_pair(M, &c[0], &c[1]) || vfft_il3p_default_chain((int)M, &c[0], &c[1], &c[2]))) ||
           est_ztt_len(M);
}
/* Bluestein's inner at M, the power of two the policy names first (vfft_policy_ilprime_blue_m,
 * policy_il.h:311-317): below 256 the pair (4.32 at 128); up to 262144 ZTURN-T in its convolution
 * form; above, the four-step at M */
static inline int est_blue_inner(const est_machine_t *m, long long M, est_inner_t *in, char *fs, size_t cap,
                                 char *why, size_t whylen)
{
    int c[2];
    if (M < 256)
    {
        if (!est_inner_pair(M, &c[0], &c[1])) return 0;
        est_inner_set(in, "2p", c, 2, 0, 0);
        return 1;
    }
    if (M <= VFFT_ZTT_MAX_N) return est_inner_ztt(m, M, in);
    return est_inner_fs(m, M, in, fs, cap, why, whylen);
}
/* the ZTURN-T inner's threaded arm: serial while the bound 16 Nc <= 2 L1d holds, else TILES (2) when
 * the inner is tiled with at least 2 tiles of its length M, else BLOCKS (1); no other inner has a
 * threaded form (il_prime_mt.h:36). The c2c prime cell reads the bound at Nc = N, the transform's
 * length. */
static inline int est_inner_mt(const est_machine_t *m, const est_inner_t *in, long long Nc, long long M, int T)
{
    if (T <= 1 || !in->ztt || est_serial(m, Nc)) return 0;
    return (in->tw > 0 && M / in->tw >= 2) ? 2 : 1;
}
/* The prime cell, and the unraced Bluestein of a smooth N past 262144 with no four-step split
 * (c2c_oop_create_il.h:292-320). Rader when N is prime and N - 1 is a ZTURN-T length at this ISA,
 * or lpf(N - 1) <= 23 and N <= L1d/16 with an inner at N - 1, or Bluestein's M would pass the inner
 * ceiling (N > 2^23) and N - 1 has a four-step split; otherwise Bluestein, refused when its M passes
 * the ceiling (the other primes there stay refused). The route row says il_route=prime il_kv=0; the
 * method row carries the inner (an inner banks no forms). At T > 1 the route row's il_mt is the
 * inner's threaded arm. */
static inline int est_prime_route(const est_machine_t *m, long long N, int T, est_recipe_t *o, char *why,
                                  size_t whylen)
{
    est_inner_t in;
    int M = 0;
    int rader = 0, r, n1, n2;
    memset(&in, 0, sizeof in);
    if (vfft_policy_is_prime((int)N))
    {
        const long long Mr = N - 1;
        if (est_ztt_len(Mr)) rader = 1;
        else if (est_lpf(Mr) <= 23 && N <= m->l1d / 16 && est_rader_inner_exists(Mr)) rader = 1;
        else if (2 * N - 1 > VFFT_POLICY_INNER_MAX_N && est_fs_split(Mr, 0, &n1, &n2)) rader = 1;
    }
    if (rader)
    {
        M = (int)(N - 1);
        r = est_rader_inner(m, M, &in, o->fs, sizeof o->fs, why, whylen);
        if (r < 0) return -1;
        if (r == 0) return est_refuse(why, whylen, "N = %lld: no inner for Rader at N - 1 = %d", N, M);
    }
    else
    {
        (void)vfft_policy_ilprime_blue_m((int)N, &M, 1);
        if (M > VFFT_POLICY_INNER_MAX_N)
            return est_refuse(why, whylen, "N = %lld is prime and Rader has no inner at N - 1 within these rules; "
                              "Bluestein's inner (%d) passes %d", N, M, VFFT_POLICY_INNER_MAX_N);
        r = est_blue_inner(m, M, &in, o->fs, sizeof o->fs, why, whylen);
        if (r < 0) return -1;
        if (r == 0) return est_refuse(why, whylen, "N = %lld: no inner for Bluestein at %d", N, M);
    }
    o->k1_il_route = VFFT_K1_IL_PRIME;
    o->il_kv = 0;
    o->prime.method = rader ? 1 : 2;
    snprintf(o->prime.in_kind, sizeof o->prime.in_kind, "%s", in.kind);
    snprintf(o->prime.in_shape, sizeof o->prime.in_shape, "%s", in.shape);
    o->prime.in_tw = in.tw;
    o->il_mt = est_inner_mt(m, &in, N, M, T);
    return 0;
}

/* ── the c2c procedure ──────────────────────────────────────────────────── */

/* The first test that holds decides the route; T > 1 keeps it and adds the threaded arm. scr = the
 * scrambled class, ip = in place, T = the thread count. */
static inline int est_c2c(const est_machine_t *m, long long N, int scr, int ip, int T, est_recipe_t *o,
                          char *why, size_t whylen)
{
    int c[64], k, tw = 0, R1, R2, c3[3];
    memset(o, 0, sizeof *o);
    /* 1. the planner's ceiling (k1_fourstep_band.h:10) */
    if (N < 2 || N > VFFT_K1FS_MAX_N)
        return est_refuse(why, whylen, "N = %lld is outside 2..%d", N, VFFT_K1FS_MAX_N);
    /* 2. a solo kernel, when N has no legal pair or N <= 10 */
    if (est_in(est_n1_list, EST_NELEM(est_n1_list), N) && (!est_pair_shape(N, &R1, &R2) || N <= 10))
    {
        o->k1_il_route = VFFT_K1_IL_MONO;
        o->il_kv = 0;   /* form 0, the solo n1 (il_solo_registry.h:45) */
        return 0;
    }
    /* 3. the prime cell (vfft_policy_prime_cell, policy_il.h:173); a composite one whose Bluestein
     * inner passes the ceiling has no engine (policy_il.h:299-302) */
    if (vfft_policy_prime_cell((int)N))
    {
        if (vfft_policy_prime_cell_unservable((int)N))
            return est_refuse(why, whylen, "N = %lld has a prime factor above %d and is not prime: its Bluestein "
                              "inner (2N - 1) passes %d", N, VFFT_POLICY_CHAIN_MAX_PRIME, VFFT_POLICY_INNER_MAX_N);
        return est_prime_route(m, N, T, o, why, whylen);
    }
    /* 4. a power of two */
    if (est_is_pow2(N))
    {
        const int n = est_lg(N);
        if (N >= (1LL << 19)) return est_fs_route(m, N, scr, T, o, why, whylen);   /* 4a: past ZTURN-T */
        if (N < VFFT_POLICY_ZTT_MIN_N || (!scr && N >= 32 && N <= 512))
        {   /* 4b: the pair, mid 4 up to 128, else 16 */
            o->k1_il_route = VFFT_K1_IL_2P_PURE;
            o->il_R1 = N <= 128 ? 4 : 16;
            o->il_R2 = (int)(N / o->il_R1);
            if (!est_pair_legal(N, o->il_R1, o->il_R2))
                return est_refuse(why, whylen, "N = %lld: the pair %d.%d is not legal", N, o->il_R1, o->il_R2);
            o->il_kv = est_nib_fwd(o->il_R1, 0) | est_nib_fwd(o->il_R2, 1) << 4;
            o->il_bkv = est_nib_bwd(o->il_R1, 0) | est_nib_bwd(o->il_R2, 1) << 4;
            return 0;
        }
        /* 4c: ZTURN-T */
        if (scr) k = est_ztt_scr(m, n, c, &tw);
        else if (!ip) k = est_ztt_nat_oop_h(m, n, c, &tw, NULL);
        else k = est_ztt_nat_ip(m, n, c, &tw);
        if (!est_ztt_ok(N, c, k, scr) || !est_ztt_tile_ok(N, c, k, tw, scr))
            return est_refuse(why, whylen, "N = %lld: no ZTURN-T chain within the create's laws", N);
        goto ztt;
    }
    /* 5. ZTURN-T's odd band at this ISA (policy_il.h:85-88) */
    if (N <= VFFT_ZTT_MAX_N && vfft_policy_ztt_odd_band((int)N))
    {
        k = est_ztt_odd(N, scr, c);
        if (!k) return est_refuse(why, whylen, "N = %lld: no ZTURN-T chain of at most %d stages", N, VFFT_ZTT_MAX_NF);
        tw = 0;
        goto ztt;
    }
    /* 6. past ZTURN-T's ceiling */
    if (N > VFFT_ZTT_MAX_N)
    {
        if (vfft_k1fs_band((int)N) && est_fs_split(N, 0, &R1, &R2))
        {
            if (!scr && ip) goto flat;   /* in place and natural the four-step pays a plane copy */
            return est_fs_route(m, N, scr, T, o, why, whylen);
        }
        /* the door's unraced Bluestein (c2c_oop_create_il.h:292-320) */
        if (2 * N - 1 > VFFT_POLICY_INNER_MAX_N)
            return est_refuse(why, whylen, "N = %lld has no four-step split (no divisor pair with both sides in "
                              "%d..%d) and its Bluestein inner (2N - 1) passes %d", N, VFFT_K1FS_SIDE_MIN,
                              VFFT_K1FS_SIDE_MAX, VFFT_POLICY_INNER_MAX_N);
        return est_prime_route(m, N, T, o, why, whylen);
    }
    /* 7. a legal pair */
    if (est_pair_shape(N, &R1, &R2))
    {
        o->k1_il_route = VFFT_K1_IL_2P_PURE;
        o->il_R1 = R1;
        o->il_R2 = R2;
        o->il_kv = est_nib_fwd(R1, 0) | est_nib_fwd(R2, 1) << 4;
        o->il_bkv = est_nib_bwd(R1, 0) | est_nib_bwd(R2, 1) << 4;
        return 0;
    }
    /* 8. a legal chain3: il_kv A | B<<4 | leaf<<8; il_bkv the leaf only (a backward mid nibble
     * refuses, il2p.h:1649-1652); il_pair = (A B).leaf as the planner banks it */
    if (est_chain3_shape(N, c3))
    {
        o->k1_il_route = VFFT_K1_IL_CHAIN3;
        memcpy(o->il_c3, c3, sizeof o->il_c3);
        o->il_R1 = c3[1] * c3[2];
        o->il_R2 = c3[0];
        o->il_kv = est_nib_fwd(c3[1], 0) | est_nib_fwd(c3[2], 0) << 4 | est_nib_fwd(c3[0], 1) << 8;
        o->il_bkv = est_nib_bwd(c3[0], 1) << 8;
        return 0;
    }
flat:
    /* 9. the flat DIT */
    k = est_flat_chain(N, c);
    if (!k) return est_refuse(why, whylen, "N = %lld: no flat chain of at most %d stages", N, VFFT_ILFD_MAX_K);
    o->k1_il_route = VFFT_K1_IL_FLAT;
    memcpy(o->il_fl, c, sizeof(int) * (size_t)k);
    o->il_fl_n = k;
    est_flat_forms(k, scr, o->il_flf, sizeof o->il_flf);
    o->il_tw = est_flat_tw(m, N, c, k);
    o->il_kv = 0;
    if (T > 1 && !est_serial(m, N))   /* TILES at the tile's width when tiled, else BLOCKS */
    {
        o->il_mt = o->il_tw ? 2 : 1;
        o->il_mt_tw = o->il_tw ? o->il_tw : 0;
    }
    return 0;
ztt:
    o->k1_il_route = VFFT_K1_IL_ZTT;
    memcpy(o->il_zt, c, sizeof(int) * (size_t)k);
    o->il_zt_n = k;
    o->il_tw = tw;
    o->il_kv = 0;
    if (T > 1 && !est_serial(m, N))   /* TILES when tiled with at least 2 tiles, else BLOCKS (ztt_mt.h:13-17) */
        o->il_mt = (tw > 0 && N / tw >= 2) ? 2 : 1;
    return 0;
}

/* est_plan_1d: the machine and the request -> the recipe. Returns 0 with *out filled, or -1 with
 * the reason in why (cut to whylen; why may be NULL when whylen is 0). Pure. Refused besides the
 * lengths est_c2c refuses: a transform other than VFFT_C2C, a machine of another lane count than
 * this build's, and the IN-PLACE SCRAMBLED PRIME CELL AT ONE THREAD. That cell cannot replay: its
 * route row and its prime method row share one key (vw2__prime_method_key, wisdom2_oop_il.h:545-551),
 * the method row is kept over the route row (vw2_oop_bank_k1_il, :515-527) and the door's scan
 * takes eng=k1 rows only (wisdom2_oop_rows.h:103), so it finds no route row and races the cell
 * (k1_commit.h:993-1004). */
static inline int est_plan_1d(const est_machine_t *m, const est_request_t *rq, est_recipe_t *out, char *why,
                              size_t whylen)
{
    int T;
    char dummy[1];
    if (!why || !whylen) { why = dummy; whylen = sizeof dummy; }
    why[0] = 0;
    if (!m || !rq || !out) return est_refuse(why, whylen, "no machine, request or recipe");
    memset(out, 0, sizeof *out);
    if (rq->transform != VFFT_C2C)
        return est_refuse(why, whylen, "transform %d: estimate plans the 1D complex transform (VFFT_C2C) only", rq->transform);
    if (m->lanes != VFFT_IL_VW)
        return est_refuse(why, whylen, "a machine of %d lanes: the formulas follow this build's %d", m->lanes, VFFT_IL_VW);
    if (m->l1d < 1024) return est_refuse(why, whylen, "an L1d of %ld bytes", m->l1d);
    if (rq->nthreads < 0) return est_refuse(why, whylen, "nthreads %d is negative", rq->nthreads);
    if (rq->order != VFFT_ORDER_NATURAL && rq->order != VFFT_ORDER_SCRAMBLED)
        return est_refuse(why, whylen, "order %d is neither natural nor scrambled", rq->order);
    if (rq->placement != VFFT_PLACEMENT_OUT && rq->placement != VFFT_PLACEMENT_IN)
        return est_refuse(why, whylen, "placement %d is neither out of place nor in place", rq->placement);
    T = rq->nthreads > 1 ? rq->nthreads : 1;
    if (est_c2c(m, rq->N, rq->order == VFFT_ORDER_SCRAMBLED, rq->placement == VFFT_PLACEMENT_IN, T, out, why, whylen) != 0)
        return -1;
    if (out->k1_il_route == VFFT_K1_IL_PRIME && rq->order == VFFT_ORDER_SCRAMBLED && rq->placement == VFFT_PLACEMENT_IN &&
        T == 1)
        return est_refuse(why, whylen, "N = %d in place, scrambled, one thread: a length with a prime factor above %d "
                          "cannot be planned here without measuring (its route row and its prime method row share "
                          "one wisdom key, so the library would race the cell); out of place, natural order or "
                          "more than one thread is served", rq->N, VFFT_POLICY_CHAIN_MAX_PRIME);
    return 0;
}

/* ── the row writer ─────────────────────────────────────────────────────── */

/* The four-step's child in the spelling a race banks today. The recipe spells it as the shipped rows
 * do: the 2D row's tokens under fs_, its row plan under fs_row_, that plan's backward twin under
 * fs_row_bwd_. A race banks the row plan INSIDE the 2D row as rp_* (wisdom2_child.h:5-21,
 * _k1fs_row_put_child k1_fourstep.h:615-640): fs_rp_k1=<its key> fs_rp_1_<name>=... Without it the
 * 2D create completes the row at its end (il2d_tier.h:4225, vfft_child_row_update
 * wisdom2_child.h:176-186). So fs_row_X becomes fs_rp_1_X under the row plan's key and fs_row_bwd_X
 * becomes fs_rp_2_X under its backward twin's (the four-step's child keys, k1_fourstep.h:215-230);
 * a text already in the rp_ spelling is copied as it is. to_rp = 0 copies every text as it is.
 * Returns 0, or -1 when a token is malformed or the text does not fit cap. */
static inline int est__fs_text(const char *in, int n1, int n2, int to_rp, char *out, size_t cap)
{
    char buf[1024], *sp = buf, *t, kr[192], kb[192];
    size_t off = 0, j;
    int has_row = 0, has_rp = 0, has_bwd = 0, pass, w;
    vw2_key_t k2, krow, kbwd;
    if (!in || strlen(in) >= sizeof buf)
        return -1;
    out[0] = 0;
    memcpy(buf, in, strlen(in) + 1);
    while ((t = vw2__tok(&sp)) != NULL)
    {
        if (!strchr(t, '=') || strncmp(t, "fs_", 3))
            return -1;   /* every child token is name=value under fs_ */
        if (!strncmp(t, "fs_rp_", 6)) has_rp = 1;
        else if (!strncmp(t, "fs_row_bwd_", 11)) has_bwd = 1;
        else if (!strncmp(t, "fs_row_", 7)) has_row = 1;
    }
    if (!to_rp || has_rp || (!has_row && !has_bwd))
    {
        if (strlen(in) >= cap)
            return -1;
        memcpy(out, in, strlen(in) + 1);
        return 0;
    }
    if (!_k1fs_child_keys(n1, n2, 0, 1, &k2, &krow, &kbwd))
        return -1;
    vw2__key_format(&krow, kr, sizeof kr);
    vw2__key_format(&kbwd, kb, sizeof kb);
    for (j = 0; kr[j]; j++) if (kr[j] == ' ') kr[j] = ',';
    for (j = 0; kb[j]; j++) if (kb[j] == ' ') kb[j] = ',';
    /* pass 0: the 2D row's own tokens; 1: the row plan, its key first; 2: its backward twin */
    for (pass = 0; pass < 3; pass++)
    {
        if (pass == 1 && has_row)
        {
            w = snprintf(out + off, cap - off, "%sfs_rp_k1=%s", off ? " " : "", kr);
            if (w < 0 || (size_t)w >= cap - off) return -1;
            off += (size_t)w;
        }
        if (pass == 2 && has_bwd)
        {
            w = snprintf(out + off, cap - off, "%sfs_rp_k%d=%s", off ? " " : "", has_row ? 2 : 1, kb);
            if (w < 0 || (size_t)w >= cap - off) return -1;
            off += (size_t)w;
        }
        memcpy(buf, in, strlen(in) + 1);
        sp = buf;
        while ((t = vw2__tok(&sp)) != NULL)
        {
            const int bwd = !strncmp(t, "fs_row_bwd_", 11), row = !bwd && !strncmp(t, "fs_row_", 7);
            if (pass == 0 && !bwd && !row)
                w = snprintf(out + off, cap - off, "%s%s", off ? " " : "", t);
            else if (pass == 1 && row)
                w = snprintf(out + off, cap - off, "%sfs_rp_1_%s", off ? " " : "", t + 7);
            else if (pass == 2 && bwd)
                w = snprintf(out + off, cap - off, "%sfs_rp_%d_%s", off ? " " : "", has_row ? 2 : 1, t + 11);
            else
                continue;
            if (w < 0 || (size_t)w >= cap - off) return -1;
            off += (size_t)w;
        }
    }
    return 0;
}
/* 1 = the text carries the 2D row: an fs_ token that is not the row plan's (what
 * _k1fs_row_has_child asks of a row, k1_fourstep.h:526-538) */
static inline int est__fs_has_child(const char *fs)
{
    const char *p = fs;
    while (p && *p)
    {
        while (*p == ' ') p++;
        if (!strncmp(p, "fs_", 3) && strncmp(p, "fs_row_", 7) && strchr(p, '='))
            return 1;
        p = strchr(p, ' ');
    }
    return 0;
}
/* the fs_ tokens onto the row at key, one vw2_update_field each (wisdom2.h:1494), as
 * _k1fs_row_put_child puts them on a row (k1_fourstep.h:615-640) */
static inline int est__put_fs(vw2_store_t *st, const vw2_key_t *key, const char *fs)
{
    char buf[1024], *sp = buf, *t;
    if (strlen(fs) >= sizeof buf)
        return -1;
    memcpy(buf, fs, strlen(fs) + 1);
    while ((t = vw2__tok(&sp)) != NULL)
    {
        char *eq = strchr(t, '=');
        if (!eq)
            return -1;
        *eq = 0;
        if (vw2_update_field(st, key, t, eq + 1) != VW2_OK)
            return -1;
    }
    return 0;
}
/* N1.N2 of a four-step inner's shape (the prime row's in_sh=) */
static inline int est__split_of(const char *shape, int *n1, int *n2)
{
    return sscanf(shape, "%d.%d", n1, n2) == 2 && *n1 >= 4 && *n2 >= 4;
}
/* The recipe's own legality before a row is written: a route the door replays, its shape a split
 * of n. A chain3's il_pair is the race's (A B).leaf (dp_planner_il.h:136-138), filled when the
 * recipe leaves it 0. R1, R2: the row's il_pair. Returns 0, or -1 with the reason. */
static inline int est__c2c_shape(const est_recipe_t *rc, int n, int *R1, int *R2, char *why, size_t whylen)
{
    *R1 = rc->il_R1;
    *R2 = rc->il_R2;
    switch (rc->k1_il_route)
    {
    case VFFT_K1_IL_MONO:
        return 0;
    case VFFT_K1_IL_2P_PURE:
        if (*R1 < 3 || *R2 < 3 || (long)*R1 * *R2 != n)
            return est_refuse(why, whylen, "the pair %d.%d is not a split of %d with both radices >= 3", *R1, *R2, n);
        return 0;
    case VFFT_K1_IL_CHAIN3:
        if (rc->il_c3[0] <= 0 || rc->il_c3[1] <= 0 || rc->il_c3[2] <= 0 || est_prod(rc->il_c3, 3) != n)
            return est_refuse(why, whylen, "chain3 %d.%d.%d does not multiply to %d", rc->il_c3[0], rc->il_c3[1], rc->il_c3[2], n);
        if (!*R1 && !*R2)
        {
            *R2 = rc->il_c3[0];
            *R1 = n / rc->il_c3[0];
        }
        if (*R2 != rc->il_c3[0] || (long)*R1 * *R2 != n)
            return est_refuse(why, whylen, "chain3's il_pair %d.%d is not (A B).leaf of %d.%d.%d", *R1, *R2, rc->il_c3[0],
                              rc->il_c3[1], rc->il_c3[2]);
        return 0;
    case VFFT_K1_IL_FLAT:
        if (rc->il_fl_n < 2 || rc->il_fl_n > VFFT_ILFD_MAX_K || est_prod(rc->il_fl, rc->il_fl_n) != n)
            return est_refuse(why, whylen, "the flat chain of %d stages does not multiply to %d (2..%d stages)", rc->il_fl_n,
                              n, VFFT_ILFD_MAX_K);
        return 0;
    case VFFT_K1_IL_ZTT:
        if (rc->il_zt_n < 2 || rc->il_zt_n > VFFT_ZTT_MAX_NF || est_prod(rc->il_zt, rc->il_zt_n) != n)
            return est_refuse(why, whylen, "the ZTURN-T chain of %d stages does not multiply to %d (2..%d stages)",
                              rc->il_zt_n, n, VFFT_ZTT_MAX_NF);
        return 0;
    case VFFT_K1_IL_FS:
        if (*R1 < 4 || *R2 < 4 || (long)*R1 * *R2 != n)
            return est_refuse(why, whylen, "the four-step split %dx%d is not a split of %d", *R1, *R2, n);
        return 0;
    case VFFT_K1_IL_PRIME:
        if ((rc->prime.method != 1 && rc->prime.method != 2) || !rc->prime.in_kind[0] || !rc->prime.in_shape[0])
            return est_refuse(why, whylen, "a prime recipe without its method and inner");
        return 0;
    default:
        return est_refuse(why, whylen, "route %d is not one the door replays", rc->k1_il_route);
    }
}
/* the four-step child's split: the route's one-thread split, or at T > 1 the threaded split il_mt
 * (the door seeds the child under (il_mt, T), k1_commit.h:1538-1552); a prime cell's four-step
 * inner: its shape. 0 = no four-step child. */
static inline int est__fs_split(const est_recipe_t *rc, int n, int T, int *n1, int *n2)
{
    if (rc->k1_il_route == VFFT_K1_IL_FS)
    {
        if (T > 1 && rc->il_mt >= 4 && n % rc->il_mt == 0) { *n1 = rc->il_mt; *n2 = n / rc->il_mt; }
        else { *n1 = rc->il_R1; *n2 = rc->il_R2; }
        return 1;
    }
    if (rc->k1_il_route == VFFT_K1_IL_PRIME && !strcmp(rc->prime.in_kind, "fs"))
        return est__split_of(rc->prime.in_shape, n1, n2);
    return 0;
}

/* est_put_rows: the rows a race would bank for the request, into st, with the store's own writers --
 * never the planner's vfft_il_dp_emit_wisdom, which banks a four-step's child from a race context
 * (dp_planner_il.h:2191-2197). In this order: the prime METHOD row before the route row (the route
 * row's writer keeps a method row, wisdom2_oop_il.h:515-527); the route row as the planner fills it
 * (dp_planner_il.h:2207-2240), il_kv always written (il_kv_raced = 1, else the door races,
 * c2c_oop_create_il.h:85-98); the T row's threaded tokens; the four-step's child; the dir=bwd row.
 * to_rp: the four-step's child in the spelling a race banks today (1) or as the recipe spells it
 * (0). Returns 0, or -1 with the reason. */
static inline int est_put_rows(vw2_store_t *st, const est_recipe_t *rc, const est_request_t *rq, int to_rp, char *why,
                               size_t whylen)
{
    const int N = rq->N, T = rq->nthreads > 1 ? rq->nthreads : 1, ip = (rq->placement == VFFT_PLACEMENT_IN);
    const int scr = (rq->order == VFFT_ORDER_SCRAMBLED), route = rc->k1_il_route;
    char b[128], fsb[1024];
    int R1, R2, n1 = 0, n2 = 0;
    vfft_oop_il_entry_t e;
    const vw2_rec_t *r;
    vw2_key_t key;
    fsb[0] = 0;
    if (rq->transform != VFFT_C2C)
        return est_refuse(why, whylen, "the row writer writes c2c rows only");
    if (est__c2c_shape(rc, N, &R1, &R2, why, whylen))
        return -1;
    if (est__fs_split(rc, N, T, &n1, &n2) && est__fs_text(rc->fs, n1, n2, to_rp, fsb, sizeof fsb))
        return est_refuse(why, whylen, "the four-step child's tokens do not parse");
    if (route == VFFT_K1_IL_PRIME)
    {   /* the method row: ord=scr place=ip role=comp lay=il, one thread (wisdom2_oop_il.h:545-551, :570) */
        if (vw2_prime_method_bank(st, N, rc->prime.method, rc->prime.in_kind, rc->prime.in_shape, rc->prime.in_tw) != VW2_OK)
            return est_refuse(why, whylen, "the prime method row was refused by its writer");
        if (!strcmp(rc->prime.in_kind, "fs"))
        {   /* a four-step inner's child rides on the METHOD row (k1_commit.h:594-596) */
            vw2_key_t pk;
            vw2__prime_method_key(N, &pk);
            if (!fsb[0] || est__put_fs(st, &pk, fsb))
                return est_refuse(why, whylen, "the prime cell's four-step inner has no child tokens");
        }
    }
    memset(&e, 0, sizeof e);
    e.N = N;
    e.K = 1;
    e.kind = VFFT_OOP_KIND_BAILEY2V;
    e.place_ip = ip;
    e.ord_scr = scr;
    e.nthreads = T > 1 ? T : 0;
    e.k1_il_route = route;
    e.il_R1 = R1;
    e.il_R2 = R2;
    if (route == VFFT_K1_IL_CHAIN3)
        memcpy(e.il_c3, rc->il_c3, sizeof e.il_c3);
    if (route == VFFT_K1_IL_FLAT)
    {
        memcpy(e.il_fl, rc->il_fl, sizeof e.il_fl);
        e.il_fl_n = rc->il_fl_n;
        memcpy(e.il_flf, rc->il_flf, sizeof e.il_flf);
        e.il_flf[sizeof e.il_flf - 1] = 0;
        e.il_tw = rc->il_tw;
    }
    if (route == VFFT_K1_IL_ZTT)
    {
        memcpy(e.il_zt, rc->il_zt, sizeof e.il_zt);
        e.il_zt_n = rc->il_zt_n;
        e.il_tw = rc->il_tw;
    }
    e.il_kv = rc->il_kv;
    e.il_kv_raced = 1;
    if (vw2_oop_bank_k1_il(st, &e) != VW2_OK)
        return est_refuse(why, whylen, "the c2c row was refused by its writer (vw2_oop_rec_k1_il)");
    r = vw2__oop_k1_scan_pl(st, N, VW2_LAY_IL, scr, ip ? VW2_PL_IP : VW2_PL_OOP, T);
    if (!r)
        return est_refuse(why, whylen, "the c2c row is not the door's after banking (its key holds another row)");
    key = r->key;
    if (T > 1 && (route == VFFT_K1_IL_FLAT || route == VFFT_K1_IL_ZTT || route == VFFT_K1_IL_PRIME))
    {   /* the threaded arm on the T row: k1_commit.h:1317 (flat, with its width), 1397 (ZTURN-T), 1448 (prime) */
        snprintf(b, sizeof b, "%d", rc->il_mt);
        if (vw2_update_field(st, &key, "il_mt", b) != VW2_OK)
            goto tok;
        if (route == VFFT_K1_IL_FLAT)
        {
            snprintf(b, sizeof b, "%d", rc->il_mt_tw);
            if (vw2_update_field(st, &key, "il_mt_tw", b) != VW2_OK)
                goto tok;
        }
    }
    if (route == VFFT_K1_IL_FS)
    {
        if (T > 1)
        {   /* the threaded split and the super-band, il_mtsb=0 when none (k1_commit.h:1543-1551) */
            est_sb_t s;
            snprintf(b, sizeof b, "%d", rc->il_mt);
            if (vw2_update_field(st, &key, "il_mt", b) != VW2_OK)
                goto tok;
            est_sb_init(&s, b, sizeof b);
            if (rc->il_zt_n >= 2) est_sb_chain(&s, rc->il_zt, rc->il_zt_n);
            else est_sb_add(&s, "0");
            if (s.len >= sizeof b || vw2_update_field(st, &key, "il_mtsb", b) != VW2_OK)
                goto tok;
        }
        else if (rc->il_kv == 1 && rc->il_zt_n >= 2)
        {   /* the one-thread super-band form's chain beside il_pair (dp_planner_il.h:2244-2252) */
            est_sb_t s;
            est_sb_init(&s, b, sizeof b);
            est_sb_chain(&s, rc->il_zt, rc->il_zt_n);
            if (s.len >= sizeof b || vw2_update_field(st, &key, "il_sb", b) != VW2_OK)
                goto tok;
        }
        if (!fsb[0] || est__put_fs(st, &key, fsb))
            return est_refuse(why, whylen, "the four-step row has no child tokens (fs_*): the door would race it "
                              "(k1_commit.h:1573-1577)");
    }
    if ((route == VFFT_K1_IL_2P_PURE || route == VFFT_K1_IL_CHAIN3) && rc->il_bkv)
    {   /* the dir=bwd row as the planner banks it, at the request's placement (dp_planner_il.h:2266-2290) */
        vw2_rec_t br;
        const char *w2 = NULL;
        if (vw2_oop_rec_k1_bwd(&br, N, route, R1, R2, rc->il_bkv, 0.0, "race", &w2) != VW2_OK)
            return est_refuse(why, whylen, "the dir=bwd row was refused (%s)", w2 ? w2 : "?");
        br.key.pl = ip ? VW2_PL_IP : VW2_PL_OOP;
        if (route == VFFT_K1_IL_CHAIN3)
        {
            est_sb_t s;
            est_sb_init(&s, b, sizeof b);
            est_sb_chain(&s, rc->il_c3, 3);
            if (s.len >= sizeof b || vw2_rec_set(&br, 1, "il_chain", b) != VW2_OK)
            {
                vw2_rec_free(&br);
                goto tok;
            }
        }
        if (vw2_bank(st, &br) != VW2_OK)
        {
            vw2_rec_free(&br);
            return est_refuse(why, whylen, "the dir=bwd row was refused by the store");
        }
    }
    return 0;
tok:
    return est_refuse(why, whylen, "a recipe token was refused by the store's lexical rules (wisdom2.h:462-471)");
}

#endif /* VFFT_IL_ESTIMATE_IL_H */
