/* estimate_1d.h -- the 1D estimate's FORMULAS: machine + request -> the recipe a race would bank.
 *
 * Implements est_plan_1d and est_recipe_print (the contracts are in estimate_recipe.h) as the
 * sketch docs/design/estimate_1d_sketch.md specifies them. The sketch's TEXT is the design; the
 * Python model under docs/research/estimate_1d/synth/ predates its corrections and is followed
 * only where the text is silent (each such place says so). Sections cited as "§n" are the
 * sketch's; library behaviour is cited file:line, relative to src/core/.
 *
 * PURE: reads only the machine and the request. No library header, no clock, no store, no
 * allocation, no I/O. The code facts it needs -- the kernel radix lists, ceilings, bands, the
 * legality laws of the creates -- are copied below with their source. Sketch §5 does not count
 * them as tuning ("code facts the formulas read are not tuning"); a registry change must be
 * mirrored here, and the glue's clock-free pre-flight (REPLAY_API §2) is the check that it was.
 *
 * The open decisions of sketch §12 take these defaults, marked "open decision N, default" where
 * they apply:
 *   1  one transform only (howmany = 1); a batch is not a request this header takes.
 *   2  every T > 1 takes the T = 8 fitted rules; at T != 8 they are UNFITTED.
 *   3  a ZTURN-T tile the L1 formula puts off the planner's raced ladder {1024, 2048}
 *      (dp_planner_il.h:1455) is clamped onto it.
 *   4  the threaded four-step carries the threaded split's own child (§7).
 *   5  never emits the prime-pool members §12.5 lists (Bluestein's smooth lengths, the
 *      Rader-stage flat inner, Rader on a four-step inner below 2^23); the primes §4.6
 *      refuses stay refused.
 *   6  the in-place four-step reuses the out-of-place split.
 *   7  the 8-lane rules are written as specified: legal, fitted to no row.
 * And: the in-place scrambled prime cell at T = 1 is refused (its route row and its prime
 * method row share one key, wisdom2_oop_il.h:515-527, 545-551; the door would race it).
 * Gaps of the text, reported and not filled: zfsr at T > 1 is refused (§7 has no zfsr row); the
 * zr2c prime child's serial bound reads at its M as §7 words it (est_inner_mt). One compromise
 * for the owner to confirm: zrf's LEVELS arm only where the library can bind it (est_real). */
#ifndef EST_1D_H
#define EST_1D_H

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <string.h>

#include "estimate_recipe.h"

/* ================================================================ code facts */

#define EST_MAX_N        16777216LL /* 2^24: VFFT_K1FS_MAX_N (il/rank1/k1_fourstep_band.h:9), the
                                     * planner's ceiling (§3.1) and the inner pool's
                                     * (VFFT_POLICY_INNER_MAX_N, policy_il.h:296) */
#define EST_ZTT_MAX_N    262144     /* VFFT_ZTT_MAX_N (il/rank1/ztt.h:121) */
#define EST_ZTT_MAX_NF   7          /* VFFT_ZTT_MAX_NF (ztt.h:120) */
#define EST_FS_MIN_N     262144     /* VFFT_K1FS_MIN_N (k1_fourstep_band.h:8) */
#define EST_FS_SIDE_MIN  64         /* VFFT_K1FS_SIDE_MIN (k1_fourstep_band.h:25) */
#define EST_FS_SIDE_MAX  16384      /* VFFT_K1FS_SIDE_MAX (k1_fourstep_band.h:26) */
#define EST_ILFD_MAX_K   10         /* VFFT_ILFD_MAX_K (il/rank1/il_flatdit.h:29); zrf too (il/real/zrf.h:228) */
#define EST_CHAIN_MAX_P  47         /* VFFT_POLICY_CHAIN_MAX_PRIME (policy_il.h:169) */
#define EST_ZFSR_TB      16         /* VFFT_ZFSR_TB (il/real/zfsr.h:57): N1 % 16, N2 % 32 (zfsr.h:100-101) */

/* the c2c K=1 routes: VFFT_K1_IL_* (common/abi/route_ids.h:40-83) */
#define EST_R_MONO   3
#define EST_R_PAIR   5
#define EST_R_CHAIN3 6
#define EST_R_PRIME  7
#define EST_R_FLAT   8
#define EST_R_ZTT    9
#define EST_R_FS     10

/* the solo kernels: VFFT_IL_N1_PAIR_RADICES (dag-fft-compiler/generator/generated/
 * il_registry_avx2.h:122; the same at avx512, il_registry_avx512.h:122). MONO serves these N. */
static const int est_n1_list[] = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21,
                                   22, 23, 25, 26, 27, 29, 31, 32, 37, 41, 43, 47, 64 };
/* the pair radices: VFFT_IL_N1T_PAIR_RADICES (the leaf, il_registry_avx2.h:428) = the
 * VFFT_IL_T2_PAIR_RADICES (the mid, :700) = VFFT_IL_T2TG_BWD_RADICES; the same at avx512.
 * The pair and chain3 enumerators draw their radices from it (dp_planner_il.h:1587-1611,
 * 1674-1730); il2p.h:1180-1186 and :1576-1586 refuse anything else. */
static const int est_pair_list[] = { 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 21, 23, 25,
                                     27, 29, 31, 32, 37, 41, 43, 47, 64 };
/* the real mono: VFFT_IL_RN1_PAIR_RADICES (il_registry_avx2.h:613) plus 32 (rn1b48) and 64
 * (rn1b444), N <= VFFT_ZRM_MAX_N = 64 (il/real/zrm.h:39,42-57); the same at avx512 */
static const int est_rn1_list[] = { 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 19, 21,
                                    22, 23, 25, 26, 27, 29, 31, 32, 37, 41, 43, 47, 53, 59, 61, 64 };
/* the real pair's radices: VFFT_IL_T2M / R2Z / T2H lists at avx2 (il_registry_avx2.h:525,1104,
 * 1125); EMPTY at avx512 (il_registry_avx512.h:477,1024,1029) */
static const int est_zrp_list[] = { 4, 6, 8, 10, 12, 16, 32, 64 };
/* the odd-composite Cooley-Tukey twins (form 5): VFFT_IL_T2_CT_FWD = N1T_CT_FWD = T2T_CT_BWD =
 * N1_CT_BWD = {15, 21, 25, 27}, both ISAs */
static const int est_ct_list[] = { 15, 21, 25, 27 };
/* the primes >= 23 of the pair list: "large" for the slot order and the chain3 leaf (table 5.5) */
static const int est_large_list[] = { 23, 29, 31, 37, 41, 43, 47 };
/* ZTURN-T's odd mids, the greedy order (dp_planner_il.h:1482-1495; ztt.h:575-585) */
static const int est_odd_mids[] = { 15, 9, 7, 5, 3 };
/* the 2D tier's column pool, its odd part in the enumerator's order (il/rank2/il2d_cols.h:166-178) */
static const int est_odd2d[] = { 47, 43, 41, 37, 31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3 };
/* vfft_il3p_default_chain's leaf order (il2p.h:1527-1528) */
static const int est_il3p_leaf[] = { 32, 16, 8, 4, 12, 10, 6, 47, 43, 41, 37, 31, 29, 27, 25, 23,
                                     21, 19, 17, 15, 13, 11, 9, 7, 5, 3 };
/* the zrf tile budgets (il/real/odd_build.h:236) */
static const int est_zrf_budgets[] = { 64, 128, 256, 512, 1024, 2048 };

#define EST_NELEM(a) ((int)(sizeof(a) / sizeof((a)[0])))

static inline int est_in(const int *list, int n, long long R)
{
    int i;
    for (i = 0; i < n; i++)
        if (list[i] == R)
            return 1;
    return 0;
}

/* ================================================================ small arithmetic */

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
static inline int est_is_prime(long long n)
{
    long long d;
    if (n < 2) return 0;
    if (n % 2 == 0) return n == 2;
    for (d = 3; d * d <= n; d += 2)
        if (n % d == 0) return 0;
    return 1;
}
/* lpf: the largest prime factor (§4.6); 1 for n < 2 */
static inline long long est_lpf(long long n)
{
    long long d, best = 1;
    if (n < 2) return 1;
    for (d = 2; d * d <= n; d++)
        while (n % d == 0) { best = d; n /= d; }
    return n > 1 ? n : best;
}
/* THE PRIME CELL: a prime factor above 47 (vfft_policy_prime_cell, policy_il.h:173-182) */
static inline int est_prime_cell(long long N)
{
    long long m = N;
    int p;
    if (N < 2) return 0;
    for (p = 2; p <= EST_CHAIN_MAX_P; p++)
        while (m % p == 0) m /= p;
    return m > 1;
}
static inline long long est_next_pow2(long long x)
{
    long long p = 1;
    while (p < x) p <<= 1;
    return p;
}
/* (e8, f4): the fewest stages over {8, 4} for 2^n -- ceil(n/3) stages, f of them radix 4
 * (the "forced 4s", §4.1: 0, 2 or 1 by (n-3) mod 3); n >= 2 */
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

/* ================================================================ text helpers */

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
static inline const char *est_route_name(int r)   /* vw2_oop_il_name (wisdom2_oop_il.h:95-97) */
{
    switch (r)
    {
    case EST_R_MONO: return "mono";
    case EST_R_PAIR: return "2p";
    case EST_R_CHAIN3: return "chain3";
    case EST_R_PRIME: return "prime";
    case EST_R_FLAT: return "flat";
    case EST_R_ZTT: return "ztt";
    case EST_R_FS: return "fs";
    default: return "none";
    }
}

/* ================================================================ the machine (§2, §3 preamble) */

/* s1 = floor(log2(L1d / 16)): 11 for any L1d of 32-63 KiB (§3) */
static inline int est_s1(const est_machine_t *m) { return est_lg(m->l1d / 16); }
/* the ZTURN-T floor: 16 at 4 lanes, 64 at 8 (VFFT_POLICY_ZTT_MIN_N, policy_il.h:63-67) */
static inline int est_ztt_floor(const est_machine_t *m) { return m->lanes == 8 ? 64 : 16; }
/* [open decision 3, default] a ZTURN-T tile off the raced ladder {1024, 2048}
 * (dp_planner_il.h:1455) is clamped onto it; 0 (untiled) stays 0. At s1 = 11 (L1d 32-63 KiB)
 * W_lo = 1024 and W_hi = 2048 are the ladder and nothing moves. */
static inline int est_ztt_clamp(long long t)
{
    if (t <= 0) return 0;
    return t < 1024 ? 1024 : (t > 2048 ? 2048 : (int)t);
}
/* §7: one transform stays serial while its complex data fits two L1s, 16 N_c <= 2 L1d */
static inline int est_serial(const est_machine_t *m, long long Nc) { return 16LL * Nc <= 2LL * m->l1d; }

/* ================================================================ bands (policy twins) */

/* ZTURN-T's odd band at this ISA: vfft_ztt_odd_band (ztt.h:575-585) narrowed to a pow2 part of
 * VFFT_POLICY_ZTT_MIN_N or more (vfft_policy_ztt_odd_band, policy_il.h:85-88): a >= 4, a >= 6
 * at 8 lanes; at most five odd mids, greedy over {15, 9, 7, 5, 3} (§3.5) */
static inline int est_odd_band(long long N, int lanes)
{
    long long m;
    int a, nm = 0, i;
    if (N < 2048 || N > EST_ZTT_MAX_N) return 0;
    a = est_v2(N);
    m = N >> a;
    if (m == 1 || a < 4 || (1LL << a) < (lanes == 8 ? 64 : 16)) return 0;
    for (i = 0; i < EST_NELEM(est_odd_mids); i++)
        while (m % est_odd_mids[i] == 0) { m /= est_odd_mids[i]; nm++; }
    return m == 1 && nm <= EST_ZTT_MAX_NF - 2;
}
/* the four-step's band: every pow2 in [262144, 2^24], and any other N in it with a divisor
 * pair in [64, 16384] (vfft_k1fs_band, k1_fourstep_band.h:30-49) */
static inline int est_fs_band(long long N)
{
    long long d;
    if (N < EST_FS_MIN_N || N > EST_MAX_N) return 0;
    if (est_is_pow2(N)) return 1;
    for (d = EST_FS_SIDE_MIN; d <= EST_FS_SIDE_MAX && d * d <= N; d++)
        if (N % d == 0 && N / d >= EST_FS_SIDE_MIN && N / d <= EST_FS_SIDE_MAX)
            return 1;
    return 0;
}
/* N - 1 (or any M) a ZTURN-T length at VW (§4.6): a pow2 in [floor, 262144], or the odd band */
static inline int est_ztt_len(const est_machine_t *m, long long M)
{
    return (est_is_pow2(M) && M >= est_ztt_floor(m) && M <= EST_ZTT_MAX_N) || est_odd_band(M, m->lanes);
}

/* ================================================================ forms (table 5.1) */

/* forward il_kv nibble: {15,21,25,27} -> 5; {8,16} -> 3; 32 -> 3 as mid, 4 as leaf; else 0 (the
 * create's default). Kernels: il2p.h:768-868 (mid_v_fn / leaf_v_fn). */
static inline int est_nib_fwd(int R, int leaf)
{
    if (est_in(est_ct_list, EST_NELEM(est_ct_list), R)) return 5;
    if (R == 8 || R == 16) return 3;
    if (R == 32) return leaf ? 4 : 3;
    return 0;
}
/* backward il_bkv nibble: mid {8,16} -> 3, leaf {8,16,32} -> 3, {15,21,25,27} -> 5 in either slot;
 * else 0. Radix 32 has no backward mid twin and no backward variant 4 (il2p.h:892-965). */
static inline int est_nib_bwd(int R, int leaf)
{
    if (est_in(est_ct_list, EST_NELEM(est_ct_list), R)) return 5;
    if (R == 8 || R == 16) return 3;
    if (R == 32 && leaf) return 3;
    return 0;
}

/* ================================================================ ZTURN-T (§4.1, §4.2) */

/* the create's laws (ztt.h:344-365, 393-423): 2..7 stages, ends in {4, 8}, mids in
 * {4, 8, 3, 5, 7, 9, 15}, product N, (N / R0) % 4 == 0; at 8 lanes (N / R0) % 8 == 0, the
 * natural class R0 % 8 == 0, the plain class R_last % 8 == 0 and (N / R_last) % 8 == 0 (t0tp and
 * tld exist at 8 only there, il_registry_avx512.h:575,1123) */
static inline int est_ztt_ok(long long N, const int *c, int nf, int scr, int lanes)
{
    int s;
    if (nf < 2 || nf > EST_ZTT_MAX_NF || N < 16 || N > EST_ZTT_MAX_N) return 0;
    for (s = 0; s < nf; s++)
    {
        const int R = c[s], mid = s > 0 && s < nf - 1;
        if (!(R == 4 || R == 8 || (mid && (R == 3 || R == 5 || R == 7 || R == 9 || R == 15)))) return 0;
    }
    if (est_prod(c, nf) != N || (N / c[0]) % 4) return 0;
    if (lanes == 8)
    {
        if ((N / c[0]) % 8) return 0;
        if (!scr && c[0] % 8) return 0;
        if (scr && (c[nf - 1] % 8 || (N / c[nf - 1]) % 8)) return 0;
    }
    return 1;
}
/* the tile laws: natural vfft_ztt_tile_legal (ztt.h:596-603), plain vfft_ztt_tile_legal_ord
 * (ztt.h:609-620) */
static inline int est_ztt_tile_ok(long long N, const int *c, int nf, long long t, int scr, int lanes)
{
    if (t == 0) return 1;
    if (nf < 3 || N % t || t >= N) return 0;
    if (!scr) return t % ((long long)c[0] * c[1]) == 0;
    if (t % ((long long)c[nf - 2] * c[nf - 1])) return 0;
    if (lanes == 8 && (t / c[nf - 1]) % 8) return 0;
    return 1;
}
/* the fewest-stage chain (the untiled shape, §4.1): an ingest 8, the forced 4s, then 8s; 4.4 at 16 */
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
/* fill(b) (§4.1): b/3 eights when 3 | b; else radix-4 stages, with one 8 in second place when
 * b is odd. Returns the count appended, -1 when b = 1 (no such fill). */
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
/* §4.1 natural, out of place. n <= s1: the fewest-stage chain, untiled. Above: s is the one value
 * in {s1-2, s1-1, s1} with 3 | (n - s); the chain is 8, fill(s-3), then (n-s)/3 eights;
 * il_tw = 2^s, W_lo when s = s1-2 (open decision 3 then clamps it). head (may be NULL) gets the
 * stage count of the tiled part -- the ingest 8 and fill(s-3), product 2^s -- or 0 when the
 * untiled shape is returned; the scrambled chain moves exactly that part (est_ztt_scr). */
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
    if (k > EST_ZTT_MAX_NF) return est_ztt_f0(n, c);
    if (head) *head = 1 + f;
    t = (s == s1 - 2) ? (1LL << (s1 - 1)) : (1LL << s);
    *tw = est_ztt_clamp(t);
    if (!est_ztt_tile_ok(1LL << n, c, k, *tw, 0, m->lanes)) *tw = 0;
    return k;
}
static inline int est_ztt_nat_oop(const est_machine_t *m, int n, int *c, int *tw)
{
    return est_ztt_nat_oop_h(m, n, c, tw, NULL);
}
/* §4.1 natural, in place: the fewest-stage chain; il_tw is the longest chain prefix whose product
 * is W_lo or W_hi (else W_lo); untiled while n <= s1 */
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
    if (!est_ztt_tile_ok(1LL << n, c, k, *tw, 0, m->lanes)) *tw = 0;
    return k;
}
/* §4.1 scrambled (an explicit request). n <= s1: the forced 4s first, then 8s, untiled. Above:
 * the natural out-of-place chain with its tiled part -- the ingest 8 and fill(s-3), product 2^s,
 * whatever width the tile then takes (W_lo at s = s1-2, or a width open decision 3 clamped) --
 * moved to the end, because the plain schedule tiles the suffix (ztt.h:606-620); at 8 lanes the
 * tiled part reversed so the chain ends in 8. il_tw = W_lo up to n = s1 + 2, else W_hi; if the
 * plain tile law refuses it the other width, else untiled. */
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
    for (i = 0; i < cut; i++) c[j++] = m->lanes == 8 ? nat[cut - 1 - i] : nat[i];
    w[0] = est_ztt_clamp(1LL << (s1 - 1));
    w[1] = est_ztt_clamp(1LL << s1);
    if (n > s1 + 2) { const long long x = w[0]; w[0] = w[1]; w[1] = x; }
    for (i = 0; i < 2; i++)
        if (est_ztt_tile_ok(1LL << n, c, k, w[i], 1, m->lanes)) { *tw = (int)w[i]; break; }
    return k;
}
/* §4.6 the convolution form (a Bluestein or Rader inner, the scrambled class, k1_commit.h:340-345):
 * 8s first, then the forced 4s at 4 lanes; the 4s first, then 8s at 8 lanes (its last stage
 * must be 8). Untiled while M <= 2^s1, else W_lo (open decision 3 clamps it; a width the plain
 * tile law refuses leaves it untiled). */
static inline int est_ztt_conv(const est_machine_t *m, int n, int *c, int *tw)
{
    int e, f, k = 0, i;
    est_mincounts(n, &e, &f);
    if (m->lanes == 8)
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
    if (!est_ztt_tile_ok(1LL << n, c, k, *tw, 1, m->lanes)) *tw = 0;
    return k;
}
/* 2^r as 8s then the forced 4s (r >= 2), or 4s then 8s; 0 stages for r < 2 */
static inline int est_bits(int r, int fours_first, int *c)
{
    int e, f, k = 0, i;
    if (r < 2) return 0;
    est_mincounts(r, &e, &f);
    if (fours_first) { for (i = 0; i < f; i++) c[k++] = 4; for (i = 0; i < e; i++) c[k++] = 8; }
    else { for (i = 0; i < e; i++) c[k++] = 8; for (i = 0; i < f; i++) c[k++] = 4; }
    return k;
}
/* §4.2 the odd band. 4 lanes: [4] + the odd part greedy, largest first, over {15, 9, 7, 5, 3}
 * (dp_planner_il.h:1482-1495) + the rest of the power of two as 8s and the forced 4s; a chain
 * past 7 stages ingests 8 instead (only 200000 = 8.5.5.5.5.5.8). 8 lanes: the natural chain
 * ingests 8 (t0tp exists at 8 only, il_registry_avx512.h:575); the scrambled chain ends in 8
 * (tld, :1123), with [4] as its ingest when a = 7. Untiled. 0 = no chain within the laws. */
static inline int est_ztt_odd(const est_machine_t *m, long long N, int scr, int *c)
{
    int mids[8], nm = 0, a = est_v2(N), k, i, r, head, rest[16], nr;
    long long o = N >> est_v2(N);
    for (i = 0; i < EST_NELEM(est_odd_mids); i++)
        while (o % est_odd_mids[i] == 0 && nm < 8) { mids[nm++] = est_odd_mids[i]; o /= est_odd_mids[i]; }
    if (o != 1) return 0;
    if (m->lanes == 8 && scr)
    {
        head = (a - 3 == 4) ? 4 : 8;
        nr = est_bits(a - est_lg(head), 1, rest);
    }
    else if (m->lanes == 8)
    {
        head = 8;
        nr = est_bits(a - 3, 0, rest);
    }
    else
    {
        head = 4;
        nr = est_bits(a - 2, 0, rest);
        if (1 + nm + nr > EST_ZTT_MAX_NF)
        {   /* the 4-lane chain past 7 stages ingests 8 instead (§4.2) */
            head = 8;
            nr = est_bits(a - 3, 0, rest);
        }
    }
    if (nr < 1 || 1 + nm + nr > EST_ZTT_MAX_NF) return 0;
    k = 0;
    c[k++] = head;
    for (i = 0; i < nm; i++) c[k++] = mids[i];
    for (r = 0; r < nr; r++) c[k++] = rest[r];
    return est_ztt_ok(N, c, k, scr, m->lanes) ? k : 0;
}

/* ================================================================ pair and chain3 (§4.3) */

/* the planner's pair (dp_planner_il.h:1587-1611): R2 the leaf and R1 = N / R2 the mid, both in
 * the pair list (3..64); no radix-64 slot at a power of two (the pow2 sunset, :1598-1604) */
static inline int est_pair_legal(long long N, int R1, int R2)
{
    if (R1 < 3 || R2 < 3 || R1 > 64 || R2 > 64 || (long long)R1 * R2 != N) return 0;
    if (!est_in(est_pair_list, EST_NELEM(est_pair_list), R1) ||
        !est_in(est_pair_list, EST_NELEM(est_pair_list), R2)) return 0;
    if (est_is_pow2(N) && (R1 == 64 || R2 == 64)) return 0;
    return 1;
}
static inline int est_large(int R) { return est_in(est_large_list, EST_NELEM(est_large_list), R); }
/* §4.3 pair factors: the most balanced legal pair, min |log(R1/R2)| (= the smallest larger
 * factor at a fixed product). Order: when exactly one factor is a prime >= 23 (table 5.5) it is
 * the mid R1; otherwise the larger factor is the leaf R2. 0 = no legal pair. */
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
/* §4.6 / §6 a convolution inner's pair at M: a pow2 M <= 512 takes table 5.3's pow2 pair (mid 4
 * up to 128, then 16: the row at N = 59 is 4.32, wisdom2_prime.txt:32), any other M the §4.3 pair */
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

/* the chain3 enumerator's law (dp_planner_il.h:1674-1730; il2p.h:1576-1586): the leaf R2 in the
 * pair list (n1t and n1 bwd), R1 = A B >= 9 and not a pure power of two, A and B in the pair
 * list (t2 fwd/bwd, t2tg bwd), A from 3 to R1/2 (so both orders of A, B enumerate) */
static inline int est_c3_legal(long long N, int R2, int A, int B)
{
    const long long R1 = (long long)A * B;
    if (R2 < 3 || A < 3 || B < 3 || R1 * R2 != N || R1 < 9 || est_is_pow2(R1)) return 0;
    return est_in(est_pair_list, EST_NELEM(est_pair_list), R2) &&
           est_in(est_pair_list, EST_NELEM(est_pair_list), A) &&
           est_in(est_pair_list, EST_NELEM(est_pair_list), B);
}
static inline double est_c3_spread(int a, int b, int c)   /* the variance of log radix, x 3 */
{
    const double la = log((double)a), lb = log((double)b), lc = log((double)c), mu = (la + lb + lc) / 3.0;
    return (la - mu) * (la - mu) + (lb - mu) * (lb - mu) + (lc - mu) * (lc - mu);
}
/* §4.3 chain3. Factors: the most balanced legal triple, min variance of log radix (the first in
 * the enumerator's order on a tie: leaf in list order, A ascending). Order: the leaf is the
 * largest even factor (else the largest that is not a prime >= 23); the two mids follow in
 * descending order; if the enumerator refuses that order the mids swap; failing that the legal
 * order with the largest leaf (design-evidence-fit/scripts/estimate.py:105-109). c3 = {R2, A, B}. */
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
    s[0] = t[0]; s[1] = t[1]; s[2] = t[2];   /* the multiset, ascending */
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
    {   /* the legal order of this multiset with the largest leaf, first in the enumerator's order */
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
/* vfft_il3p_default_chain (il2p.h:1523-1572), the code's own seed, mirrored: c3 = {R2, A, B} */
static inline int est_il3p_default(long long M, int c3[3])
{
    static const int ecb[3] = { 12, 10, 6 };
    const int np = EST_NELEM(est_pair_list);
    int i, j;
    for (i = 0; i < EST_NELEM(est_il3p_leaf); i++)
    {
        const int r2 = est_il3p_leaf[i];
        long long R1, o, pb, a;
        if (M % r2) continue;
        R1 = M / r2;
        if (R1 < 9) continue;
        o = R1;
        while ((o & 1) == 0) o >>= 1;
        pb = R1 / o;
        if (o == 1) continue;
        if (pb == 1)
        {
            for (a = 3; a * a <= R1; a += 2)
            {
                const long long b = R1 / a;
                if (R1 % a || b < 3) continue;
                if (!est_in(est_pair_list, np, a) || !est_in(est_pair_list, np, b)) continue;
                c3[0] = r2; c3[1] = (int)a; c3[2] = (int)b;
                return 1;
            }
            continue;
        }
        if (pb >= 4 && est_in(est_pair_list, np, o) && est_in(est_pair_list, np, pb))
        {
            c3[0] = r2; c3[1] = (int)o; c3[2] = (int)pb;
            return 1;
        }
        for (j = 0; j < 3; j++)
        {
            const int b = ecb[j];
            if (R1 % b) continue;
            a = R1 / b;
            if (a < 3 || !est_in(est_pair_list, np, a) || !est_in(est_pair_list, np, b)) continue;
            c3[0] = r2; c3[1] = (int)a; c3[2] = b;
            return 1;
        }
    }
    return 0;
}

/* ================================================================ flat DIT (§4.4) */

/* the odd prime factors of o (o odd), with 3.3 merged into 9; 0 when a factor exceeds 47 */
static inline int est_odd_stages(long long o, int *c)
{
    int k = 0, p, n3 = 0, i;
    for (p = 3; p <= EST_CHAIN_MAX_P; p += 2)
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
/* the >10-stage merge (§4.4, also zrf's in §6): while K > 10 merge two 5s into 25; failing that
 * a 3 and a 5 into 15, else a 3 and a 7 into 21. Here the merged radix takes the place of the
 * first of its two factors, which is what the sketch's two examples show (11718750 =
 * 2.25.5.5.5.5.5.5.5.3, 16406250 = 2.7.25.5.5.5.5.5.5.3). Returns the new K, or -1. */
static inline int est_merge10(int *c, int K)
{
    static const int pr[3][3] = { { 5, 5, 25 }, { 5, 3, 15 }, { 7, 3, 21 } };
    while (K > EST_ILFD_MAX_K)
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
/* §4.4 the chain: a lone 2 leads when N = 2 (mod 4) (chain_search.h:239-243), otherwise the
 * largest radix leads; then the 8s and the forced 4s, then the odd primes descending with 3.3
 * merged into 9; at most 10 stages (VFFT_ILFD_MAX_K). No Rader stage is ever emitted, so the
 * scrambled class always builds (il_flatdit.h:313). 0 = no chain (a factor above 47). */
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
    {   /* the pow2 stages (descending: 8s before 4s), then the odd ones descending */
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
/* §4.4 il_forms: t on every middle stage; the last is t at K = 2, otherwise n, and o in the
 * natural class from K = 5 (table 5.6). Every pool radix has the t2csgn tail (il_flatdit.h:286-289). */
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
/* §4.4 il_tw: untiled while 16 N <= 2 L1d; otherwise the widest block span R_s D_s of a non-tail
 * stage s in [1, K-2] (D_s > 4, VFFT_ILFD_TAIL_D = 4) with 16 span <= L1d (il_flatdit.h:904-916) */
static inline int est_flat_tw(const est_machine_t *m, long long N, const int *c, int K)
{
    long long best = 0;
    int s;
    if (16LL * N <= 2LL * m->l1d) return 0;
    for (s = 1; s <= K - 2; s++)
    {
        const long long D = est_prod(c + s + 1, K - s - 1), w = c[s] * D;
        if (D > 4 && 16LL * w <= m->l1d && w > best) best = w;
    }
    return (int)best;
}

/* ================================================================ four-step (§4.5) */

/* fs_chain (§4.5): the column chain at N1 over the 2D tier's pool, depth <= 4
 * (il/rank2/il2d_cols.h:164-241): 8s with one 4 or 16, then the odd part largest first, with a
 * closing radix last. A lone 2 closes the chain folded into the smallest odd factor p with 2p a
 * closing radix (6, 10, 14, 22, 26), else as 2 itself (the closing-only radices, :160-163).
 * Where the text's shape passes 4 stages the pow2 part folds into 64/32/16 stages (the model's
 * fold, design-evidence-fit/scripts/estimate.py:274-282; the text is silent there). Returns nst. */
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
    {   /* fold the pow2 part into 64/32/16 stages */
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
/* the split (§4.5). Powers of two: 2^19 = 512.1024 (table 5.7), N1 = 4096 and N2 = N/4096 from
 * 2^20. Other N: the most balanced legal split, both sides in [64, 16384], N1 with a column
 * chain (est_fs_col_chain) and N2 not a prime cell (k1_fourstep.h:86-158), N1 >= N2 on a tie.
 * threaded (§7): N/2048 up to 2^20, then 4096; other N the smaller column side on a tie. */
static inline int est_fs_split(long long N, int threaded, int *N1, int *N2)
{
    long long d, best = 0;
    int c[8], b1 = 0, b2 = 0;
    if (N < EST_FS_MIN_N || N > EST_MAX_N) return 0;
    if (est_is_pow2(N))
    {
        if (N < (1LL << 19)) return 0;
        if (threaded) *N1 = N <= (1LL << 20) ? (int)(N / 2048) : 4096;
        else *N1 = N == (1LL << 19) ? 512 : 4096;
        *N2 = (int)(N / *N1);
        return 1;
    }
    for (d = EST_FS_SIDE_MIN; d <= EST_FS_SIDE_MAX; d++)
    {
        const long long e = N / d, hi = d > e ? d : e;
        if (N % d || e < EST_FS_SIDE_MIN || e > EST_FS_SIDE_MAX) continue;
        if (!est_fs_col_chain(d, c) || est_prime_cell(e)) continue;
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
/* fs_wl (§4.5): the widest column stage span w < N1 with 16 w N2 <= L2 -- the 2D tier's own L2
 * gate (il2d_tier.h:1992-2010, vfft_policy_fits_l2 policy_cache.h:33-37); the spans are the
 * tier's L[s], s >= 1 (the suffix products, il2d_cols.h:1054-1062), w >= 8
 * (vfft_policy_il2d_band_ok, policy_il.h:584-590); 0 when none qualifies */
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

/* ================================================================ the c2c procedure (§3) */

static inline int est_c2c(const est_machine_t *m, long long N, int scr, int ip, int T, est_recipe_t *o,
                          char *why, size_t whylen);

/* the c2c K=1 route row's tokens, in vw2_oop_rec_k1_il's order (wisdom2_oop_il.h:417-500),
 * each name prefixed by pfx */
static inline void est_c2c_tokens(est_sb_t *s, const est_recipe_t *r, const char *pfx)
{
    est_sb_add(s, "%seng=k1 %sil_route=%s", pfx, pfx, est_route_name(r->k1_il_route));
    if (r->il_R1 || r->il_R2) est_sb_add(s, " %sil_pair=%d.%d", pfx, r->il_R1, r->il_R2);
    if (r->k1_il_route == EST_R_CHAIN3)
        est_sb_add(s, " %sil_chain=%d.%d.%d", pfx, r->il_c3[0], r->il_c3[1], r->il_c3[2]);
    if (r->k1_il_route == EST_R_FLAT)
    {
        est_sb_add(s, " %sil_flat=", pfx);
        est_sb_chain(s, r->il_fl, r->il_fl_n);
        if (r->il_flf[0]) est_sb_add(s, " %sil_forms=%s", pfx, r->il_flf);
        if (r->il_tw > 0) est_sb_add(s, " %sil_tw=%d", pfx, r->il_tw);
    }
    if (r->k1_il_route == EST_R_ZTT)
    {
        est_sb_add(s, " %sil_ztt=", pfx);
        est_sb_chain(s, r->il_zt, r->il_zt_n);
        if (r->il_tw > 0) est_sb_add(s, " %sil_tw=%d", pfx, r->il_tw);
    }
    est_sb_add(s, " %sil_kv=%d", pfx, r->il_kv);
}
/* the dir=bwd sibling's tokens (vw2_oop_rec_k1_bwd, wisdom2_oop_il.h:359-405, plus il_chain for
 * chain3 as dp_planner_il.h:2276-2286 adds it) */
static inline void est_bwd_tokens(est_sb_t *s, const est_recipe_t *r, const char *pfx)
{
    est_sb_add(s, "%seng=k1 %sil_route=%s %sil_pair=%d.%d %sil_kv=%d", pfx, pfx, est_route_name(r->k1_il_route),
               pfx, r->il_R1, r->il_R2, pfx, r->il_bkv);
    if (r->k1_il_route == EST_R_CHAIN3)
        est_sb_add(s, " %sil_chain=%d.%d.%d", pfx, r->il_c3[0], r->il_c3[1], r->il_c3[2]);
}

/* §4.5 the four-step's 2D child at (N1, N2) as the row's fs_* tokens -- the shipped rows' shape
 * (src/wisdom/14900KF/wisdom2_oop.txt:203, :8387): fs_chain, fs_wl, fs_tf = (fs_wl > 0)
 * (il2d_tier.h:3936), the constant switches fs_ro=0 fs_sw=0 fs_rbk=8 fs_turn=0 fs_csk=0
 * (table 5.7), fs_blu=0 / fs_forms when the chain needs them (below; NOT in table 5.7: an addition
 * the sketch should record), at T > 1 (the c2c four-step only, §7) fs_cmt=1 and fs_mtarm=1
 * fs_msw=128 when fs_wl = 0, else 0 and 0 (table 5.7), then the row plan at N2 -- this procedure,
 * c2c natural IN PLACE at T = 1, the row plan's own key (k1_fourstep.h:226) -- as fs_row_*, and its
 * backward twin fs_row_bwd_* (key :229) when the row is a pair or chain3 with nonzero il_bkv (no
 * other row route has one). */
static inline int est_fs_child(const est_machine_t *m, int N1, int N2, int T, char *fs, size_t cap,
                               char *why, size_t whylen)
{
    est_recipe_t row;
    est_sb_t s;
    int c[8], nst, wl;
    nst = est_fs_col_chain(N1, c);
    if (!nst) return est_refuse(why, whylen, "four-step %dx%d: no column chain at N1 = %d (sketch 4.5)", N1, N2, N1);
    wl = est_fs_wl(m, c, nst, N1, N2);
    if (est_c2c(m, N2, 0, 1, 1, &row, why, whylen) != 0) return -1;
    est_sb_init(&s, fs, cap);
    est_sb_add(&s, "fs_chain=");
    est_sb_chain(&s, c, nst);
    est_sb_add(&s, " fs_wl=%d fs_tf=%d fs_ro=0", wl, wl > 0);
    {   /* Two verdicts the 2D tier races when its row lacks them, which table 5.7 does not list
         * (found by the test sweeps, src/estimate/TEST_LOG.md). Both spell the sketch's own choice
         * -- the column chain fs_chain, its kernels at the construction-table default -- in the
         * tokens a race banks when that choice wins:
         * - fs_blu=0: a chain with an odd radix races itself against a Bluestein column unless the
         *   row carries the N1-arm verdict; 0 = the chain won (il2d_tier.h:3607-3650; the 2D rows
         *   in the store, e.g. "chain=5.3.9 ... blu=0"). A power-of-two chain never races it.
         * - fs_forms: a stage whose radix has rival column forms races them unless the row
         *   carries forms= (il2d_tier.h:3218-3250). The rivals are vfft_il2p_col_forms
         *   (il2p.h:481-487; a c2c column, so no half-store twins): radix 32 {b48, b84}, every
         *   other radix one form. The race spells every stage by its first name, the incumbent
         *   ("b48", "b88" for 64, "-" else; il2d_tier.h:2483, 2597-2598), so that is written. */
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
    if (T > 1)   /* [open decision 2, default] the T = 8 rule at every T > 1 */
        est_sb_add(&s, " fs_cmt=1 fs_mtarm=%d fs_msw=%d", wl == 0, wl == 0 ? 128 : 0);
    est_sb_add(&s, " ");
    est_c2c_tokens(&s, &row, "fs_row_");
    if ((row.k1_il_route == EST_R_PAIR || row.k1_il_route == EST_R_CHAIN3) && row.il_bkv)
    {
        est_sb_add(&s, " ");
        est_bwd_tokens(&s, &row, "fs_row_bwd_");
    }
    if (s.len >= cap) return est_refuse(why, whylen, "four-step %dx%d: the child's tokens pass %u bytes", N1, N2, (unsigned)cap);
    return 0;
}
/* §4.5 the four-step route: il_pair = the one-thread split (open decision 6, default: in place
 * reuses it), il_kv = 0 (form 0 at T = 1; the super-band form 1 is not emitted). At T > 1 (§7,
 * [open decision 4, default]): il_mt = the threaded split's N1_T, the child is that split's own
 * (k1_commit.h:1538-1552, 1595-1620), and il_mtsb = 8.64.8 when natural, 16 N > L3 (or no L3:
 * vfft_policy_exceeds_l3, policy_cache.h:46-50), N1_T = 4096 and N2_T % 16 == 0
 * (k1_fourstep.h:955-958), else 0. */
static inline int est_fs_route(const est_machine_t *m, long long N, int scr, int T, est_recipe_t *o,
                               char *why, size_t whylen)
{
    int N1, N2, N1T, N2T;
    if (!est_fs_split(N, 0, &N1, &N2)) return est_refuse(why, whylen, "N = %lld: no four-step split (sketch 4.5)", N);
    o->k1_il_route = EST_R_FS;
    o->il_R1 = N1;
    o->il_R2 = N2;
    o->il_kv = 0;
    if (T <= 1) return est_fs_child(m, N1, N2, 1, o->fs, sizeof o->fs, why, whylen);
    if (!est_fs_split(N, 1, &N1T, &N2T)) return est_refuse(why, whylen, "N = %lld: no threaded four-step split (sketch 7)", N);
    o->il_mt = N1T;
    if (!scr && (m->l3 <= 0 || 16LL * N > (long long)m->l3) && N1T == 4096 && N2T % 16 == 0)
    {
        o->il_zt[0] = 8; o->il_zt[1] = 64; o->il_zt[2] = 8;
        o->il_zt_n = 3;
    }
    return est_fs_child(m, N1T, N2T, T, o->fs, sizeof o->fs, why, whylen);
}

/* ================================================================ the prime cell (§4.6) */

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
/* a four-step inner at M (kind 4): the split and child of §4.5 at M, scrambled class, out of
 * place, T = 1, form 0 (k1_commit.h:276-307); the child goes in fs */
static inline int est_inner_fs(const est_machine_t *m, long long M, est_inner_t *in, char *fs, size_t cap,
                               char *why, size_t whylen)
{
    int c[2];
    if (!est_fs_band(M) || !est_fs_split(M, 0, &c[0], &c[1])) return 0;
    if (est_fs_child(m, c[0], c[1], 1, fs, cap, why, whylen) != 0) return -1;
    est_inner_set(in, "fs", c, 2, 0, 0);
    return 1;
}
/* a ZTURN-T inner at M, the scrambled class: the convolution chain at a pow2, the odd chain in
 * the odd band (untiled) */
static inline int est_inner_ztt(const est_machine_t *m, long long M, est_inner_t *in)
{
    int c[16], k, tw = 0;
    if (est_is_pow2(M) && M >= est_ztt_floor(m) && M <= EST_ZTT_MAX_N)
        k = est_ztt_conv(m, est_lg(M), c, &tw);
    else if (est_odd_band(M, m->lanes))
        k = est_ztt_odd(m, M, 1, c);
    else
        return 0;
    if (!k || !est_ztt_ok(M, c, k, 1, m->lanes)) return 0;
    est_inner_set(in, "ztt", c, k, tw, 1);
    return 1;
}
/* §4.6 the Rader inner at M = N - 1: a pair if M <= 4096 has one; otherwise ZTURN-T (the
 * convolution chain, or the odd chain); otherwise the code's own vfft_il3p_default_chain (the
 * pool holds it at M <= 4096, k1_commit.h:420-438); otherwise, in the four-step band, the
 * four-step at M. 1 = found, 0 = none, -1 = refused. */
static inline int est_rader_inner(const est_machine_t *m, long long M, est_inner_t *in, char *fs, size_t cap,
                                  char *why, size_t whylen)
{
    int c[3];
    if (M <= 4096 && est_inner_pair(M, &c[0], &c[1])) { est_inner_set(in, "2p", c, 2, 0, 0); return 1; }
    if (est_inner_ztt(m, M, in)) return 1;
    if (M <= 4096 && est_il3p_default(M, c)) { est_inner_set(in, "3p", c, 3, 0, 0); return 1; }
    return est_inner_fs(m, M, in, fs, cap, why, whylen);
}
/* an inner exists at N - 1 for §4.6's second Rader case: a pair or the il3p seed at M <= 4096,
 * or a ZTURN-T length (the model's rader_inner_exists, synth/synth_estimate.py:86-87) */
static inline int est_rader_inner_exists(const est_machine_t *m, long long M)
{
    int c[3];
    return (M <= 4096 && (est_inner_pair(M, &c[0], &c[1]) || est_il3p_default(M, c))) || est_ztt_len(m, M);
}
/* §4.6 the Bluestein inner at M (the next power of two >= 2N - 1, at least 16:
 * vfft_policy_ilprime_blue_m, policy_il.h:311-317): M = 128, the pair 4.32 (table 5.3); M <=
 * 262144, ZTURN-T in its convolution form; above, the four-step at M */
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
    if (M <= EST_ZTT_MAX_N) return est_inner_ztt(m, M, in);
    return est_inner_fs(m, M, in, fs, cap, why, whylen);
}
/* the ZTURN-T inner's threaded arm (§7): serial while 16 Nc <= 2 L1d, else TILES (2) when the
 * inner is tiled with >= 2 tiles of its length M, else BLOCKS (1); a non-ZTURN-T inner has no
 * threaded form (il_prime_mt.h:36). Nc is the length the serial bound reads:
 * - the c2c prime cell: Nc = N, the transform's own length (§7: "16·N_c <= 2·L1d (N_c = N for
 *   c2c ...)"), not the inner's M -- a Bluestein N in 2049..6144 (M = 8192 or 16384) stays serial;
 * - a zr2c prime child: Nc = the child's M, as §7's zr2c row words it ("the prime-inner rule
 *   above at the child's M"). OPEN for the owner (the text reads both ways). At L1d = 48 KiB the
 *   two never differ there: the fold's own gate already put N/2 above the bound, a Bluestein M is
 *   larger still, and a Rader M = N/2 - 1 falls back under it only at N/2 = 6145, not a prime. */
static inline int est_inner_mt(const est_machine_t *m, const est_inner_t *in, long long Nc, long long M, int T)
{
    if (T <= 1 || !in->ztt || est_serial(m, Nc)) return 0;
    return (in->tw > 0 && M / in->tw >= 2) ? 2 : 1;
}
/* §4.6 the prime cell, and §3.6's unraced Bluestein at a smooth N past 262144 with no split.
 * Method: Rader iff N is prime and N - 1 is a ZTURN-T length at VW, or lpf(N - 1) <= 23 (table
 * 5.5) and N <= L1d/16 with an inner at N - 1, or the Bluestein M would pass 2^24 (N > 2^23) and
 * N - 1 has a four-step split. Otherwise Bluestein, refused when M > 2^24 ([open decision 5,
 * default]: the other primes in (2^23, 2^24] stay refused). The route row says il_route=prime
 * il_kv=0; the method row carries the inner (no il_kv / il_bkv on an inner,
 * wisdom2_prime.txt:30-33). At T > 1 the route row's il_mt is the inner's arm (est_inner_mt),
 * the serial bound read at N. */
static inline int est_prime_route(const est_machine_t *m, long long N, int T, est_recipe_t *o, char *why,
                                  size_t whylen)
{
    est_inner_t in;
    long long M;
    int rader = 0, r, n1, n2;
    memset(&in, 0, sizeof in);
    if (est_is_prime(N))
    {
        const long long Mr = N - 1;
        if (est_ztt_len(m, Mr)) rader = 1;
        else if (est_lpf(Mr) <= 23 && N <= m->l1d / 16 && est_rader_inner_exists(m, Mr)) rader = 1;
        else if (2 * N - 1 > EST_MAX_N && est_fs_split(Mr, 0, &n1, &n2)) rader = 1;
    }
    if (rader)
    {
        M = N - 1;
        r = est_rader_inner(m, M, &in, o->fs, sizeof o->fs, why, whylen);
        if (r < 0) return -1;
        if (r == 0) return est_refuse(why, whylen, "N = %lld: Rader has no inner at M = %lld (sketch 4.6)", N, M);
    }
    else
    {
        M = est_next_pow2(2 * N - 1 > 16 ? 2 * N - 1 : 16);
        if (M > EST_MAX_N)
            return est_refuse(why, whylen, "N = %lld: Bluestein's M = %lld passes 2^24 and no Rader case holds "
                              "(sketch 4.6; open decision 5)", N, M);
        r = est_blue_inner(m, M, &in, o->fs, sizeof o->fs, why, whylen);
        if (r < 0) return -1;
        if (r == 0) return est_refuse(why, whylen, "N = %lld: Bluestein has no inner at M = %lld (sketch 4.6)", N, M);
    }
    o->k1_il_route = EST_R_PRIME;
    o->il_kv = 0;
    o->prime.method = rader ? 1 : 2;
    snprintf(o->prime.in_kind, sizeof o->prime.in_kind, "%s", in.kind);
    snprintf(o->prime.in_shape, sizeof o->prime.in_shape, "%s", in.shape);
    o->prime.in_tw = in.tw;
    o->il_mt = est_inner_mt(m, &in, N, M, T);
    return 0;
}

/* ================================================================ the c2c procedure (§3) */

/* §3: the first test that holds decides the route; T > 1 keeps it and adds the threaded arm (§7).
 * scr = the order class, ip = the placement, T = the thread count (1 = one thread). */
static inline int est_c2c(const est_machine_t *m, long long N, int scr, int ip, int T, est_recipe_t *o,
                          char *why, size_t whylen)
{
    const int vw = m->lanes;
    int c[64], k, tw = 0, R1, R2, c3[3];
    memset(o, 0, sizeof *o);
    snprintf(o->eng, sizeof o->eng, "k1");
    /* 1. the planner's ceiling */
    if (N < 2 || N > EST_MAX_N) return est_refuse(why, whylen, "c2c N = %lld outside 2..2^24 (sketch 3.1)", N);
    /* 2. a solo kernel, and no legal pair or N <= 10 (table 5.4) */
    if (est_in(est_n1_list, EST_NELEM(est_n1_list), N) && (!est_pair_shape(N, &R1, &R2) || N <= 10))
    {
        o->k1_il_route = EST_R_MONO;
        o->il_kv = 0;   /* form 0, the solo n1 (il_solo_registry.h:45); mono64 is never emitted */
        return 0;
    }
    /* 3. the prime cell (policy_il.h:169-182); a composite with 2N - 1 > 2^24 is refused
     * (vfft_policy_prime_cell_unservable, policy_il.h:298-302) */
    if (est_prime_cell(N))
    {
        if (!est_is_prime(N) && 2 * N - 1 > EST_MAX_N)
            return est_refuse(why, whylen, "N = %lld: a composite prime cell with 2N - 1 > 2^24 (sketch 3.3)", N);
        return est_prime_route(m, N, T, o, why, whylen);
    }
    /* 4. a power of two */
    if (est_is_pow2(N))
    {
        const int n = est_lg(N);
        if (N >= (1LL << 19)) return est_fs_route(m, N, scr, T, o, why, whylen);   /* 4a: ZTURN-T ends at 262144 */
        if (N < est_ztt_floor(m) || (!scr && N >= 32 && N <= 512))
        {   /* 4b: the pair, mid 4 up to 128, else 16 (table 5.2, 5.3) */
            o->k1_il_route = EST_R_PAIR;
            o->il_R1 = N <= 128 ? 4 : 16;
            o->il_R2 = (int)(N / o->il_R1);
            if (!est_pair_legal(N, o->il_R1, o->il_R2))
                return est_refuse(why, whylen, "N = %lld: the pow2 pair %d.%d is not legal (sketch 3.4b)", N, o->il_R1, o->il_R2);
            o->il_kv = est_nib_fwd(o->il_R1, 0) | est_nib_fwd(o->il_R2, 1) << 4;
            o->il_bkv = est_nib_bwd(o->il_R1, 0) | est_nib_bwd(o->il_R2, 1) << 4;
            return 0;
        }
        /* 4c: ZTURN-T */
        if (scr) k = est_ztt_scr(m, n, c, &tw);
        else if (!ip) k = est_ztt_nat_oop(m, n, c, &tw);
        else k = est_ztt_nat_ip(m, n, c, &tw);
        if (!est_ztt_ok(N, c, k, scr, vw) || !est_ztt_tile_ok(N, c, k, tw, scr, vw))
            return est_refuse(why, whylen, "N = %lld: the ZTURN-T chain breaks the create's laws (sketch 4.1)", N);
        goto ztt;
    }
    /* 5. ZTURN-T's odd band */
    if (est_odd_band(N, vw))
    {
        k = est_ztt_odd(m, N, scr, c);
        if (!k) return est_refuse(why, whylen, "N = %lld: no odd-band ZTURN-T chain within 7 stages (sketch 4.2)", N);
        tw = 0;
        goto ztt;
    }
    /* 6. past ZTURN-T's ceiling */
    if (N > EST_ZTT_MAX_N)
    {
        if (est_fs_band(N) && est_fs_split(N, 0, &R1, &R2))
        {
            if (!scr && ip) goto flat;   /* the four-step pays a plane copy in place, natural */
            return est_fs_route(m, N, scr, T, o, why, whylen);
        }
        /* the door's unraced Bluestein (c2c_oop_create_il.h:292-320), refused when 2N - 1 > 2^24 */
        if (2 * N - 1 > EST_MAX_N)
            return est_refuse(why, whylen, "N = %lld: no four-step split and 2N - 1 > 2^24 (sketch 3.6)", N);
        return est_prime_route(m, N, T, o, why, whylen);
    }
    /* 7. a legal pair */
    if (est_pair_shape(N, &R1, &R2))
    {
        o->k1_il_route = EST_R_PAIR;
        o->il_R1 = R1;
        o->il_R2 = R2;
        o->il_kv = est_nib_fwd(R1, 0) | est_nib_fwd(R2, 1) << 4;
        o->il_bkv = est_nib_bwd(R1, 0) | est_nib_bwd(R2, 1) << 4;
        return 0;
    }
    /* 8. a legal chain3: il_kv A | B<<4 | leaf<<8; il_bkv the leaf only, leaf<<8 (a nonzero mid
     * nibble refuses, il2p.h:1649-1652). il_pair = (A B).R2 as the race banks it
     * (dp_planner_il.h:2219-2221; src/wisdom/14900KF/wisdom2_oop.txt "n=250 ... il_pair=25.10"). */
    if (est_chain3_shape(N, c3))
    {
        o->k1_il_route = EST_R_CHAIN3;
        memcpy(o->il_c3, c3, sizeof o->il_c3);
        o->il_R1 = c3[1] * c3[2];
        o->il_R2 = c3[0];
        o->il_kv = est_nib_fwd(c3[1], 0) | est_nib_fwd(c3[2], 0) << 4 | est_nib_fwd(c3[0], 1) << 8;
        o->il_bkv = est_nib_bwd(c3[0], 1) << 8;
        return 0;
    }
flat:
    /* 9. the flat DIT (§4.4) */
    k = est_flat_chain(N, c);
    if (!k) return est_refuse(why, whylen, "N = %lld: no flat chain within 10 stages (sketch 4.4)", N);
    o->k1_il_route = EST_R_FLAT;
    memcpy(o->il_fl, c, sizeof(int) * (size_t)k);
    o->il_fl_n = k;
    est_flat_forms(k, scr, o->il_flf, sizeof o->il_flf);
    o->il_tw = est_flat_tw(m, N, c, k);
    o->il_kv = 0;
    if (T > 1 && !est_serial(m, N))   /* §7: TILES with il_mt_tw = il_tw when tiled, else BLOCKS */
    {
        o->il_mt = o->il_tw ? 2 : 1;
        o->il_mt_tw = o->il_tw ? o->il_tw : 0;
    }
    return 0;
ztt:
    o->k1_il_route = EST_R_ZTT;
    memcpy(o->il_zt, c, sizeof(int) * (size_t)k);
    o->il_zt_n = k;
    o->il_tw = tw;
    o->il_kv = 0;
    if (T > 1 && !est_serial(m, N))   /* §7: TILES when tiled with >= 2 tiles, else BLOCKS (ztt_mt.h:13-17) */
        o->il_mt = (tw > 0 && N / tw >= 2) ? 2 : 1;
    return 0;
}

/* ================================================================ real (§6) */

/* ZTT-r's grammar at M = N/2 (zttr.h:113-143,180): 2^a times a product over {15, 9, 7, 5, 3},
 * a >= 5 (the terminator run L = M / R_last >= 8 with L % 8 == 0 needs 32 | M), 32 <= M <= 262144 */
static inline int est_zttr_grammar(long long M)
{
    long long o;
    int i;
    if (M < 32 || M > EST_ZTT_MAX_N || est_v2(M) < 5) return 0;
    o = M >> est_v2(M);
    for (i = 0; i < EST_NELEM(est_odd_mids); i++)
        while (o % est_odd_mids[i] == 0) o /= est_odd_mids[i];
    return o == 1;
}
/* §6 even rule 4, the zttr chain at M = N/2. Power-of-two M: c2r 4, then 8s, then 4s; r2c 8, the
 * 4s, then 8s, closing 4 (with no forced 4, two 8s become three 4s). 2^a odd M: the first pow2
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
    if (k > EST_ZTT_MAX_NF || !est_ztt_ok(M, c, k, 0, 4)) return 0;
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
/* §6 zrf's chain: the prime factors, 3.3 -> 9, ascending; 2 <= K <= 10 (zrf.h:228), the §4.4
 * merges above 10 stages, re-sorted ascending (29296875 = 3.5.5.5.5.5.5.5.5.25). 0 = none. */
static inline int est_zrf_chain(long long N, int *c)
{
    int k = est_odd_stages(N, c);
    if (k < 2) return 0;
    est_sort_asc(c, k);
    if (k > EST_ILFD_MAX_K)
    {
        est_sort_desc(c, k);
        k = est_merge10(c, k);
        if (k < 2) return 0;
        est_sort_asc(c, k);
    }
    return k;
}
/* §6 zrb's inner at M: M < 2048 a pair, then the il3p seed; otherwise as §4.6 -- a pow2 M takes
 * Bluestein's (the convolution ZTURN-T to 262144, the four-step above), any other M ZTURN-T's
 * odd chain in the odd band, else the four-step in its band. 1 / 0 none / -1 refused. */
static inline int est_zrb_inner(const est_machine_t *m, long long M, est_inner_t *in, char *fs, size_t cap,
                                char *why, size_t whylen)
{
    int c[3];
    if (M < 2048)
    {
        if (est_inner_pair(M, &c[0], &c[1])) { est_inner_set(in, "2p", c, 2, 0, 0); return 1; }
        if (est_il3p_default(M, c)) { est_inner_set(in, "3p", c, 3, 0, 0); return 1; }
        return 0;
    }
    if (est_is_pow2(M)) return est_blue_inner(m, M, in, fs, cap, why, whylen);
    if (est_inner_ztt(m, M, in)) return 1;
    return est_inner_fs(m, M, in, fs, cap, why, whylen);
}
/* §6 zrb's length m: the smallest 2^a {1, 3, 5, 7, 9, 15} >= N + (N-1)/2 (vfft_zrb_min_m,
 * zrb.h:59) that an inner builds -- 2^a q with a >= 2 and the power of two >= 16, the code's
 * candidate shapes (odd_build.h:368-376) -- up to 2^24 */
static inline int est_zrb(const est_machine_t *m, long long N, est_recipe_t *o, char *why, size_t whylen)
{
    static const int q[6] = { 1, 3, 5, 7, 9, 15 };
    const long long mn = N + (N - 1) / 2;
    long long cand[200], M;
    int nc = 0, i, j, a, r;
    for (i = 0; i < 6; i++)
        for (a = 2; a <= 24; a++)
        {
            M = (long long)q[i] << a;
            if (M > EST_MAX_N) break;
            if (M >= mn && (q[i] != 1 || M >= 16) && nc < 200) cand[nc++] = M;
        }
    for (i = 1; i < nc; i++)
        for (j = i; j > 0 && cand[j] < cand[j - 1]; j--) { const long long x = cand[j]; cand[j] = cand[j - 1]; cand[j - 1] = x; }
    for (i = 0; i < nc; i++)
    {
        est_inner_t in;
        memset(&in, 0, sizeof in);
        o->fs[0] = 0;
        r = est_zrb_inner(m, cand[i], &in, o->fs, sizeof o->fs, why, whylen);
        if (r < 0) return -1;
        if (r == 0) continue;
        snprintf(o->eng, sizeof o->eng, "zrb");
        o->zrb.M = (int)cand[i];
        snprintf(o->zrb.kind, sizeof o->zrb.kind, "%s", in.kind);
        snprintf(o->zrb.shape, sizeof o->zrb.shape, "%s", in.shape);
        o->zrb.tw = in.tw;
        o->zrb.mt = 0;   /* §7: zrb has no threaded row */
        return 0;
    }
    return est_refuse(why, whylen, "real N = %lld: no zrb length up to 2^24 has an inner (sketch 6)", N);
}

/* §6: the door picks by parity (il/real/real_create_il.h:34); a real N plans like complex N/2 */
static inline int est_real(const est_machine_t *m, long long N, int c2r, int ip, int T, est_recipe_t *o,
                           char *why, size_t whylen)
{
    const int vw = m->lanes;
    int c[32], k, i;
    memset(o, 0, sizeof *o);
    if (N < 2) return est_refuse(why, whylen, "real N = %lld < 2 (sketch 6: the odd door admits N >= 3, odd_build.h:50-54)", N);
    if (N % 2)
    {   /* odd N */
        if (est_in(est_rn1_list, EST_NELEM(est_rn1_list), N) && (N <= 17 || est_is_prime(N)))
        {
            snprintf(o->eng, sizeof o->eng, "zrm");
            return 0;
        }
        if (est_prime_cell(N))
        {   /* zrb, refused when (3N - 1)/2 > 2^24 (_zrb_ok, zrp_build.h:260-264) */
            if ((3 * N - 1) / 2 > EST_MAX_N)
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
        /* §7: LEVELS above the serial bound -- where it binds. vfft_zrf_mt_bind declines LEVELS
         * unless the first level is tiled with a tile per worker (_zrf_mt_level_ok,
         * zrf_mt.h:133-156), and a race banks only an arm that bound (the stored T = 8 zrf rows
         * carry mt=2, mt=1 or no mt).
         * The first level's tile is the widest stage span R_s D_s (s in 1..K-2, D_s > 4, not a
         * tail stage) within the budget (zrf.h:255-262); its tiles are h D / tw with D = N / R_0,
         * h = R_0 / 2 (zrf.h:329). Where LEVELS would decline the row stays serial.
         * A COMPROMISE FOR THE OWNER TO CONFIRM: §7 says mt=2 above the bound with no test on T;
         * written literally, a cell where LEVELS cannot bind gets a silently serial plan, which the
         * glue refuses (EST_SERIAL). The T in the test is the library's own bind law
         * (_zrf_mt_level_ok, zrf_mt.h:133-136), not a fitted rule (src/estimate/TEST_LOG.md bug 3). */
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
    if (N / 2 > EST_MAX_N) return est_refuse(why, whylen, "real N = %lld: N/2 > 2^24 (sketch 6)", N);
    /* 1. zrm */
    if (N <= 16 && est_in(est_rn1_list, EST_NELEM(est_rn1_list), N))
    {
        snprintf(o->eng, sizeof o->eng, "zrm");
        return 0;
    }
    /* 2. zrp, 4 lanes only, N <= 160 (table 5.8), an even x even pair: the most balanced, smaller
     * factor first, form B (the r2z leaf) */
    if (vw == 4 && N <= 160)
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
    /* 3. zfsr: N/2 a power of two in 2^19..2^24; the split of §4.5 at N/2 and its child as fs_*.
     * At T > 1 the sketch has no rule for zfsr: §7's table lists no zfsr row, and §11 says only
     * that estimate does not switch engines by T. A race at T picks the split at T
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
        return est_fs_child(m, n1, n2, 1, o->fs, sizeof o->fs, why, whylen);
    }
    /* 4. zttr, 4 lanes only: a pow2 N in 512..524288, or N >= 2048 with N/2 in ZTT-r's grammar */
    if (vw == 4 && ((est_is_pow2(N) && N >= 512 && N <= 524288) ||
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
            o->zttr.mt = 0;   /* §7: zttr has no threaded row */
            return 0;
        }
    }
    /* 5. zr2c: the child is this c2c procedure at N/2, natural, one thread; route child_nat_ip in
     * place and for c2r from N = 1024 (table 5.8), else child_oop_il */
    {
        const int route = ip || (c2r && N >= 1024);
        est_recipe_t ch;
        if (est_c2c(m, N / 2, 0, route, 1, &ch, why, whylen) != 0) return -1;
        *o = ch;
        snprintf(o->eng, sizeof o->eng, "zr2c");
        o->zr2c.route = route;
        if (c2r && o->k1_il_route == EST_R_PAIR)
        {   /* the c2r pair child mirrored, leaf and mid swapped, both forms recomputed for the
             * swapped slots (a c2r child runs only its backward forms, wisdom2_real_il.h:382) */
            const int t = o->il_R1;
            o->il_R1 = o->il_R2;
            o->il_R2 = t;
            o->il_kv = est_nib_fwd(o->il_R1, 0) | est_nib_fwd(o->il_R2, 1) << 4;
            o->il_bkv = est_nib_bwd(o->il_R1, 0) | est_nib_bwd(o->il_R2, 1) << 4;
        }
        o->il_mt = 0;   /* the child is one-thread (zr2c_build.h:38-41) ... */
        o->il_mt_tw = 0;
        if (T > 1 && N >= 64 && !est_serial(m, N / 2))
        {   /* §7: fold=mt (zrp_build.h:795-818); a prime child also takes its own arm at its M */
            o->zr2c.fold_mt = 1;
            if (o->k1_il_route == EST_R_PRIME)
            {
                est_inner_t in;
                long long M = 1;
                memset(&in, 0, sizeof in);
                in.tw = o->prime.in_tw;
                in.ztt = !strcmp(o->prime.in_kind, "ztt");
                {   /* M = the inner shape's product (_ilprime_desc_len, k1_commit.h:488-496) */
                    const char *p = o->prime.in_shape;
                    long long v = 0;
                    for (;; p++)
                    {
                        if (*p >= '0' && *p <= '9') { v = 10 * v + (*p - '0'); continue; }
                        M *= v;
                        v = 0;
                        if (!*p) break;
                    }
                }
                o->il_mt = est_inner_mt(m, &in, M, M, T);   /* the bound at the child's M (§7, above) */
            }
        }
        return 0;
    }
}

/* ================================================================ the contracts */

static inline int est_plan_1d(const est_machine_t *m, const est_request_t *rq, est_recipe_t *out, char *why,
                              size_t whylen)
{
    const int T = rq && rq->nthreads > 1 ? rq->nthreads : 1;   /* [open decision 2, default] */
    char dummy[1];
    if (!why || !whylen) { why = dummy; whylen = sizeof dummy; }
    why[0] = 0;
    if (!m || !rq || !out) return est_refuse(why, whylen, "missing argument");
    memset(out, 0, sizeof *out);
    if (m->lanes != 4 && m->lanes != 8) return est_refuse(why, whylen, "lanes %d is not 4 or 8", m->lanes);
    if (m->l1d < 1024) return est_refuse(why, whylen, "L1d %ld bytes is not a cache size", m->l1d);
    if (rq->nthreads < 0) return est_refuse(why, whylen, "nthreads %d < 0", rq->nthreads);
    if (rq->order != 0 && rq->order != 1) return est_refuse(why, whylen, "order %d is not natural (0) or scrambled (1)", rq->order);
    if (rq->placement != 0 && rq->placement != 1) return est_refuse(why, whylen, "placement %d is not out (0) or in (1)", rq->placement);
    if (rq->transform == 0)
    {
        if (est_c2c(m, rq->N, rq->order == 1, rq->placement == 1, T, out, why, whylen) != 0) return -1;
        if (out->k1_il_route == EST_R_PRIME && rq->order == 1 && rq->placement == 1 && T == 1)
            return est_refuse(why, whylen, "N = %d: the in-place scrambled prime cell at T = 1 is refused: its route "
                              "row and its prime method row share one key (wisdom2_oop_il.h:515-527, 545-551), so the "
                              "door finds no route row and races (k1_commit.h:993-1004)", rq->N);
        return 0;
    }
    if (rq->transform == 1 || rq->transform == 2)
    {
        if (rq->order != 0)
            return est_refuse(why, whylen, "a real plan is natural order only (vfft.c:1940-1957)");
        return est_real(m, rq->N, rq->transform == 2, rq->placement == 1, T, out, why, whylen);
    }
    return est_refuse(why, whylen, "transform %d is not c2c (0), r2c (1) or c2r (2)", rq->transform);
}

/* the c2c route row: the key, then the route's tokens, the T row's threaded tokens (il_mt
 * il_mt_tw il_mtsb, set with vw2_update_field after the bank) and the four-step's fs_* */
static inline void est_print_c2c(est_sb_t *s, const est_recipe_t *rc, const est_request_t *rq, int T)
{
    const char *pl = rq->placement == 1 ? "ip" : "oop";
    if (rc->k1_il_route == EST_R_PRIME)
    {   /* the method row first: vw2_prime_method_bank's key and tokens (wisdom2_oop_il.h:545-596) */
        est_sb_add(s, "t=c2c n=%d q=1 ord=scr place=ip role=comp lay=il | eng=%s in=%s in_sh=%s in_tw=%d",
                   rq->N, rc->prime.method == 1 ? "rader" : "bluestein", rc->prime.in_kind, rc->prime.in_shape,
                   rc->prime.in_tw);
        if (!strcmp(rc->prime.in_kind, "fs") && rc->fs[0]) est_sb_add(s, " %s", rc->fs);
        est_sb_add(s, "\n");
    }
    est_sb_add(s, "t=c2c n=%d q=1 ord=%s place=%s role=comp lay=il", rq->N, rq->order == 1 ? "scr" : "nat", pl);
    if (T > 1) est_sb_add(s, " nthreads=%d", T);
    est_sb_add(s, " | ");
    est_c2c_tokens(s, rc, "");
    if (T > 1)
    {
        if (rc->k1_il_route == EST_R_FLAT || rc->k1_il_route == EST_R_ZTT || rc->k1_il_route == EST_R_PRIME)
        {
            est_sb_add(s, " il_mt=%d", rc->il_mt);
            if (rc->k1_il_route == EST_R_FLAT && rc->il_mt == 2) est_sb_add(s, " il_mt_tw=%d", rc->il_mt_tw);
        }
        if (rc->k1_il_route == EST_R_FS)
        {
            est_sb_add(s, " il_mt=%d il_mtsb=", rc->il_mt);
            if (rc->il_zt_n) est_sb_chain(s, rc->il_zt, rc->il_zt_n);
            else est_sb_add(s, "0");
        }
    }
    if (rc->k1_il_route == EST_R_FS && rc->fs[0]) est_sb_add(s, " %s", rc->fs);
    est_sb_add(s, "\n");
    if ((rc->k1_il_route == EST_R_PAIR || rc->k1_il_route == EST_R_CHAIN3) && rc->il_bkv)
    {   /* the dir=bwd sibling, keyed ord=nat at the request's placement (dp_planner_il.h:2266-2290) */
        est_sb_add(s, "t=c2c n=%d q=1 ord=nat place=%s dir=bwd role=comp lay=il | ", rq->N, pl);
        est_bwd_tokens(s, rc, "");
        est_sb_add(s, "\n");
    }
}

static inline int est_recipe_print(const est_recipe_t *rc, const est_request_t *rq, char *buf, size_t cap)
{
    const int T = rq->nthreads > 1 ? rq->nthreads : 1;
    est_sb_t s;
    est_sb_init(&s, buf, cap);
    if (rq->transform == 0)
        est_print_c2c(&s, rc, rq, T);
    else
    {   /* the real row: vw2_real_il_key (wisdom2_real_il.h:60-68) and the engine's tokens
         * (wisdom2_real_il.h:1-44) */
        est_sb_add(&s, "t=%s n=%d q=1 ord=nat place=%s", rq->transform == 2 ? "c2r" : "r2c", rq->N,
                   rq->placement == 1 ? "ip" : "oop");
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
            est_sb_add(&s, " split=%dx%d %s", rc->zfsr.n1, rc->zfsr.n2, rc->fs);
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
            if (!strcmp(rc->zrb.kind, "fs") && rc->fs[0]) est_sb_add(&s, " %s", rc->fs);
        }
        else if (!strcmp(rc->eng, "zr2c"))
        {   /* vw2_real_il_bank_zr2c's order (wisdom2_real_il.h:422-470) */
            est_sb_add(&s, " route=%s", rc->zr2c.route ? "child_nat_ip" : "child_oop_il");
            if (T > 1) est_sb_add(&s, " fold=%s", rc->zr2c.fold_mt ? "mt" : "st");
            est_sb_add(&s, " il_route=%s", est_route_name(rc->k1_il_route));
            if (rc->il_R1 || rc->il_R2) est_sb_add(&s, " il_pair=%d.%d", rc->il_R1, rc->il_R2);
            if (rc->k1_il_route == EST_R_CHAIN3) est_sb_add(&s, " il_chain=%d.%d.%d", rc->il_c3[0], rc->il_c3[1], rc->il_c3[2]);
            if (rc->k1_il_route == EST_R_FLAT)
            {
                est_sb_add(&s, " il_flat=");
                est_sb_chain(&s, rc->il_fl, rc->il_fl_n);
                if (rc->il_flf[0]) est_sb_add(&s, " il_forms=%s", rc->il_flf);
            }
            if (rc->k1_il_route == EST_R_ZTT || (rc->k1_il_route == EST_R_FS && rc->il_zt_n >= 2))
            {
                est_sb_add(&s, " %s=", rc->k1_il_route == EST_R_ZTT ? "il_ztt" : "il_sb");
                est_sb_chain(&s, rc->il_zt, rc->il_zt_n);
            }
            if (rc->il_tw > 0) est_sb_add(&s, " il_tw=%d", rc->il_tw);
            if (rc->k1_il_route == EST_R_PRIME)
            {
                est_sb_add(&s, " il_prime=%s il_prime_in=%s il_prime_sh=%s il_prime_tw=%d",
                           rc->prime.method == 1 ? "rader" : "bluestein", rc->prime.in_kind, rc->prime.in_shape,
                           rc->prime.in_tw);
                if (rc->il_mt > 0) est_sb_add(&s, " il_mt=%d", rc->il_mt);
            }
            if (rc->fs[0] && (rc->k1_il_route == EST_R_FS || rc->k1_il_route == EST_R_PRIME)) est_sb_add(&s, " %s", rc->fs);
            est_sb_add(&s, " il_kv=%d il_bkv=%d", rc->il_kv, rc->il_bkv);
        }
        est_sb_add(&s, "\n");
    }
    if (buf && cap) buf[s.len < cap ? s.len : cap - 1] = 0;
    return (int)s.len;
}

#endif /* EST_1D_H */
