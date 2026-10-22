/* estimate_selftest.c -- print the 1D estimate's recipes for a list of N, and an independent law pass
 * over a sweep (git-ignored). No plan is built, nothing is timed.
 *
 * The c2c formulas are the LIBRARY's (src/core/il/planning/estimate_il.h), and they read the library's
 * own facts (the IL registry's radix lists, the bands of ztt.h, k1_fourstep_band.h and policy_il.h), so
 * this program is a library TU: it includes vfft.c, like the other tools here. It calls no create.
 *
 * Build (no --vfft: the TU includes vfft.c; about 90 s):
 *   python gauntlet/build.py --compile --src src/estimate/estimate_selftest.c
 * Usage (from the Bash tool, where the mingw DLLs are on PATH):
 *   estimate_selftest [options] N...
 *     -t c2c|r2c|c2r   transform (default c2c)
 *     -o nat|scr       order (default nat)
 *     -p oop|ip        placement (default oop)
 *     -T n             threads (default 1)
 *     -vw 4|8          lanes: must be this build's (VFFT_IL_VW) -- the formulas follow its laws, so an
 *                      8-lane law pass needs an AVX-512 build (VFFT_ISA=avx512) and an AVX-512 host or SDE
 *     -l1d B -l2 B -l3 B   cache bytes (default the i9's P-core: 49152 / 2097152 / 37748736,
 *                      the stamp of src/wisdom/14900KF)
 *     -all             each N in every c2c cell (nat/scr x oop/ip) and r2c/c2r oop
 *     -sweep A B       every N in [A, B] for the request: route counts, refusals, and every
 *                      recipe that breaks a law re-checked here (products, stage counts,
 *                      radix lists, form nibbles) -- a quick sanity pass, not the pre-flight
 * Each recipe prints in the store's own text, one line per row the build would bank. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#pragma GCC diagnostic ignored "-Wunused-function"
#include "vfft.c"                    /* FIRST: the whole library in its own order; it carries the library's estimate */
#include "estimate_1d.h"             /* the real formulas, est_plan_any, est_recipe_print */

/* ---------------------------------------------------------------- an independent law pass */
/* the form-nibble resolvers, transcribed from il2p.h:768-965 (laws.py MID_V / LEAF_V / T2T_BWD_V /
 * N1_BWD_V): variant -> the radices with that kernel */
static int st_nib_ok(int R, int v, int leaf, int bwd)
{
    if (v == 0 || v == 0xF) return 1;
    if (!bwd)
    {
        if (v == 3) return R == 8 || R == 16 || R == 32;
        if (v == 4) return leaf ? R == 32 : (R == 16 || R == 32);
        if (v == 5) return R == 15 || R == 21 || R == 25 || R == 27;
        if (v == 1) return leaf ? (R == 16 || R == 32 || R == 64) : (R == 16 || R == 32 || R == 64);
        if (v == 2) return R == 32 || R == 64;
        return 0;
    }
    if (v == 3) return R == 8 || R == 16 || (leaf && R == 32);
    if (v == 5) return R == 15 || R == 21 || R == 25 || R == 27;
    if (v == 1 || v == 2) return R == 32 || R == 64;
    return 0;
}
static long long st_prod_shape(const char *p)
{
    long long m = 1, v = 0;
    for (;; p++)
    {
        if (*p >= '0' && *p <= '9') { v = 10 * v + (*p - '0'); continue; }
        m *= v;
        v = 0;
        if (!*p) break;
    }
    return m;
}
static int st_in_pair(int R) { return est_in(est_pair_list, EST_NELEM(est_pair_list), R); }

/* the c2c section of a recipe (also a zr2c child): returns a reason, or NULL */
static const char *st_check_c2c(long long N, const est_recipe_t *r, int scr)
{
    int i;
    switch (r->k1_il_route)
    {
    case VFFT_K1_IL_MONO:
        if (!est_in(est_n1_list, EST_NELEM(est_n1_list), N)) return "mono: N not in the n1 list";
        return r->il_kv ? "mono: form != 0" : NULL;
    case VFFT_K1_IL_2P_PURE:
        if ((long long)r->il_R1 * r->il_R2 != N) return "pair: product";
        if (!st_in_pair(r->il_R1) || !st_in_pair(r->il_R2)) return "pair: radix not in the pair list";
        if (est_is_pow2(N) && (r->il_R1 == 64 || r->il_R2 == 64)) return "pair: 64 slot at a pow2";
        if (!st_nib_ok(r->il_R1, r->il_kv & 15, 0, 0) || !st_nib_ok(r->il_R2, (r->il_kv >> 4) & 15, 1, 0)) return "pair: il_kv nibble";
        if (!st_nib_ok(r->il_R1, r->il_bkv & 15, 0, 1) || !st_nib_ok(r->il_R2, (r->il_bkv >> 4) & 15, 1, 1)) return "pair: il_bkv nibble";
        return NULL;
    case VFFT_K1_IL_CHAIN3:
    {
        const long long R1 = (long long)r->il_c3[1] * r->il_c3[2];
        if (R1 * r->il_c3[0] != N) return "chain3: product";
        if (R1 < 9 || est_is_pow2(R1)) return "chain3: pure pow2 / small cofactor";
        for (i = 0; i < 3; i++) if (!st_in_pair(r->il_c3[i])) return "chain3: radix not in the list";
        if (!st_nib_ok(r->il_c3[1], r->il_kv & 15, 0, 0) || !st_nib_ok(r->il_c3[2], (r->il_kv >> 4) & 15, 0, 0) ||
            !st_nib_ok(r->il_c3[0], (r->il_kv >> 8) & 15, 1, 0)) return "chain3: il_kv nibble";
        if (r->il_bkv & 0xFF) return "chain3: il_bkv mid nibble";
        if (!st_nib_ok(r->il_c3[0], (r->il_bkv >> 8) & 15, 1, 1)) return "chain3: il_bkv leaf nibble";
        return NULL;
    }
    case VFFT_K1_IL_FLAT:
    {
        static const int t2cp[] = { 3, 4, 5, 7, 8, 9, 11, 13, 15, 16, 17, 19, 21, 23, 25, 27, 29, 31, 37, 41, 43, 47 };
        if (r->il_fl_n < 2 || r->il_fl_n > 10) return "flat: K outside 2..10";
        if (est_prod(r->il_fl, r->il_fl_n) != N) return "flat: product";
        if (r->il_fl[0] == 2 && N % 4 != 2) return "flat: lone 2 with 4 | N";
        if (!est_in(est_n1_list, EST_NELEM(est_n1_list), r->il_fl[0])) return "flat: leaf has no n1c";
        for (i = 1; i < r->il_fl_n; i++) if (!est_in(t2cp, EST_NELEM(t2cp), r->il_fl[i])) return "flat: stage has no t2cp";
        if (est_is_pow2(N) || (N <= VFFT_ZTT_MAX_N && vfft_policy_ztt_odd_band((int)N))) return "flat: outside its pool";
        if (r->il_flf[0] && (int)strlen(r->il_flf) != 2 * (r->il_fl_n - 1) - 1) return "flat: forms length";
        if (scr && strchr(r->il_flf, 'o')) return "flat: o in the scrambled class";
        if (r->il_tw)
        {
            int s, ok = 0;
            for (s = 1; s <= r->il_fl_n - 2; s++)
            {
                const long long D = est_prod(r->il_fl + s + 1, r->il_fl_n - s - 1);
                if (D > 4 && r->il_fl[s] * D == r->il_tw) ok = 1;
            }
            if (!ok) return "flat: il_tw not a non-tail span";
        }
        return NULL;
    }
    case VFFT_K1_IL_ZTT:
        if (!est_ztt_ok(N, r->il_zt, r->il_zt_n, scr)) return "ztt: chain laws";
        if (!est_ztt_tile_ok(N, r->il_zt, r->il_zt_n, r->il_tw, scr)) return "ztt: tile law";
        if (r->il_tw && r->il_tw != 1024 && r->il_tw != 2048 && est_is_pow2(N)) return "ztt: tile off the ladder";
        if (est_is_pow2(N) && N < VFFT_POLICY_ZTT_MIN_N) return "ztt: below the floor";
        if (r->il_mt == 2 && !(r->il_tw && N / r->il_tw >= 2)) return "ztt: TILES without 2 tiles";
        return NULL;
    case VFFT_K1_IL_FS:
        if ((long long)r->il_R1 * r->il_R2 != N || r->il_R1 < 64 || r->il_R2 < 64) return "fs: split";
        if (!vfft_k1fs_band((int)N)) return "fs: outside the band";
        if (!strstr(r->fs, "fs_chain=") || !strstr(r->fs, "fs_row_eng=k1")) return "fs: child tokens missing";
        if (r->il_mt && N % r->il_mt) return "fs: il_mt does not divide N";
        return NULL;
    case VFFT_K1_IL_PRIME:
    {
        const long long M = st_prod_shape(r->prime.in_shape);
        if (r->prime.method == 1 && (!vfft_policy_is_prime((int)N) || M != N - 1)) return "prime: Rader at a composite / M != N-1";
        if (r->prime.method == 2 && (M < 2 * N - 1 || M % 2 || M > VFFT_K1FS_MAX_N)) return "prime: Bluestein M";
        if (!strcmp(r->prime.in_kind, "ztt"))
        {
            int c[8], k = 0;
            const char *p = r->prime.in_shape;
            while (*p && k < 8) { c[k++] = atoi(p); while (*p && *p != '.') p++; if (*p) p++; }
            if (!est_ztt_ok(M, c, k, 1) || !est_ztt_tile_ok(M, c, k, r->prime.in_tw, 1)) return "prime: ZTT inner laws";
        }
        else if (r->il_mt) return "prime: il_mt with a non-ZTT inner";
        if (!strcmp(r->prime.in_kind, "fs") && !strstr(r->fs, "fs_chain=")) return "prime: fs inner without a child";
        return NULL;
    }
    default:
        return "unknown route";
    }
}
static const char *st_check(const est_machine_t *m, const est_request_t *rq, const est_any_recipe_t *r)
{
    const long long N = rq->N;
    int i;
    if (rq->transform == VFFT_C2C) return st_check_c2c(N, &r->c, rq->order == VFFT_ORDER_SCRAMBLED);
    if (!strcmp(r->eng, "zrm")) return (N <= 64 && est_in(est_rn1_list, EST_NELEM(est_rn1_list), N)) ? NULL : "zrm: list";
    if (!strcmp(r->eng, "zrp"))
        return (m->lanes == 4 && (long long)r->zrp.R1 * r->zrp.R2 == N && r->zrp.R1 % 2 == 0 && r->zrp.R2 % 2 == 0 &&
                est_in(est_zrp_list, EST_NELEM(est_zrp_list), r->zrp.R1) && est_in(est_zrp_list, EST_NELEM(est_zrp_list), r->zrp.R2))
                   ? NULL : "zrp: pair";
    if (!strcmp(r->eng, "zttr"))
    {
        const long long M = N / 2;
        if (m->lanes != 4) return "zttr at 8 lanes";
        if (!est_ztt_ok(M, r->zttr.chain, r->zttr.nf, 0)) return "zttr: ZTT laws";
        if (!est_ztt_tile_ok(M, r->zttr.chain, r->zttr.nf, (long long)r->zttr.tile, 0)) return "zttr: tile";
        if (M / r->zttr.chain[r->zttr.nf - 1] < 8 || (M / r->zttr.chain[r->zttr.nf - 1]) % 8) return "zttr: L law";
        return NULL;
    }
    if (!strcmp(r->eng, "zfsr"))
        return ((long long)r->zfsr.n1 * r->zfsr.n2 == N / 2 && r->zfsr.n1 % 16 == 0 && r->zfsr.n2 % 32 == 0 &&
                strstr(r->c.fs, "fs_chain=")) ? NULL : "zfsr: split";
    if (!strcmp(r->eng, "zrf"))
    {
        static const int pool[] = { 9, 7, 5, 3, 25, 27, 21, 23, 19, 17, 15, 13, 11, 29, 31, 37, 41, 43, 47 };
        if (N % 2 == 0 || r->zrf.K < 2 || r->zrf.K > 10 || est_prod(r->zrf.chain, r->zrf.K) != N) return "zrf: chain";
        for (i = 0; i < r->zrf.K; i++) if (!est_in(pool, EST_NELEM(pool), r->zrf.chain[i])) return "zrf: radix";
        return NULL;
    }
    if (!strcmp(r->eng, "zrb"))
    {
        const long long M = st_prod_shape(r->zrb.shape);
        if (N % 2 == 0 || N <= 64 || r->zrb.M < N + (N - 1) / 2 || r->zrb.M > VFFT_K1FS_MAX_N || M != r->zrb.M) return "zrb: M";
        return NULL;
    }
    if (!strcmp(r->eng, "zr2c"))
    {
        if (N % 2) return "zr2c: odd";
        return st_check_c2c(N / 2, &r->c, 0);
    }
    return "unknown engine";
}

/* ---------------------------------------------------------------- the driver */
static const char *st_engs[] = { "zrm", "zrp", "zttr", "zr2c", "zfsr", "zrf", "zrb" };

/* one request: print its recipe (quiet = 0), *bad = a law it breaks; returns the c2c route, or
 * 16 + the real engine's index in st_engs, or -1 for a refusal */
static int st_one(const est_machine_t *m, const est_request_t *rq, int quiet, const char **bad)
{
    est_any_recipe_t r;
    char why[512], buf[4096];
    *bad = NULL;
    if (est_plan_any(m, rq, &r, why, sizeof why) != 0)
    {
        if (!quiet) printf("N=%d refused: %s\n", rq->N, why);
        return -1;
    }
    *bad = st_check(m, rq, &r);
    if (!quiet)
    {
        est_recipe_print(&r, rq, buf, sizeof buf);
        fputs(buf, stdout);
        if (*bad) printf("  !! %s\n", *bad);
    }
    if (rq->transform == VFFT_C2C) return r.c.k1_il_route;
    {
        int e;
        for (e = 0; e < EST_NELEM(st_engs); e++)
            if (!strcmp(r.eng, st_engs[e])) return 16 + e;
    }
    return 0;
}

int main(int argc, char **argv)
{
    est_machine_t m = { VFFT_IL_VW, 49152, 2097152, 37748736, 8 };
    est_request_t rq = { 0, VFFT_C2C, VFFT_ORDER_NATURAL, VFFT_PLACEMENT_OUT, 1 };
    int i, all = 0, sweep = 0;
    long long a = 0, b = -1;
    setvbuf(stdout, NULL, _IOFBF, 1 << 16);
    for (i = 1; i < argc; i++)
    {
        const char *s = argv[i];
        if (!strcmp(s, "-t") && i + 1 < argc) { s = argv[++i]; rq.transform = !strcmp(s, "r2c") ? VFFT_R2C : !strcmp(s, "c2r") ? VFFT_C2R : VFFT_C2C; }
        else if (!strcmp(s, "-o") && i + 1 < argc) rq.order = !strcmp(argv[++i], "scr") ? VFFT_ORDER_SCRAMBLED : VFFT_ORDER_NATURAL;
        else if (!strcmp(s, "-p") && i + 1 < argc) rq.placement = !strcmp(argv[++i], "ip") ? VFFT_PLACEMENT_IN : VFFT_PLACEMENT_OUT;
        else if (!strcmp(s, "-T") && i + 1 < argc) rq.nthreads = atoi(argv[++i]);
        else if (!strcmp(s, "-vw") && i + 1 < argc)
        {
            m.lanes = atoi(argv[++i]);
            if (m.lanes != VFFT_IL_VW)
            {
                fprintf(stderr, "-vw %d: this build is %d lanes (%s); the formulas follow the build's laws -- build with "
                        "VFFT_ISA=%s for %d lanes\n", m.lanes, VFFT_IL_VW, VFFT_ISA_NAME, m.lanes == 8 ? "avx512" : "avx2",
                        m.lanes);
                return 2;
            }
        }
        else if (!strcmp(s, "-l1d") && i + 1 < argc) m.l1d = atol(argv[++i]);
        else if (!strcmp(s, "-l2") && i + 1 < argc) m.l2 = atol(argv[++i]);
        else if (!strcmp(s, "-l3") && i + 1 < argc) m.l3 = atol(argv[++i]);
        else if (!strcmp(s, "-all")) all = 1;
        else if (!strcmp(s, "-sweep") && i + 2 < argc) { sweep = 1; a = atoll(argv[i + 1]); b = atoll(argv[i + 2]); i += 2; }
        else if (s[0] == '-' ) { fprintf(stderr, "unknown option %s\n", s); return 2; }
    }
    if (sweep)
    {
        long long n, refused = 0, bad = 0, counted = 0;
        long cnt[256];
        memset(cnt, 0, sizeof cnt);
        for (n = a; n <= b; n++)
        {
            const char *why;
            int r;
            rq.N = (int)n;
            r = st_one(&m, &rq, 1, &why);
            counted++;
            if (r < 0) { refused++; if (refused <= 20) { char w[512]; est_any_recipe_t t; est_plan_any(&m, &rq, &t, w, sizeof w); printf("refused N=%lld: %s\n", n, w); } continue; }
            cnt[r & 255]++;
            if (why) { bad++; if (bad <= 40) { printf("LAW N=%lld: %s\n", n, why); st_one(&m, &rq, 0, &why); } }
        }
        printf("swept %lld N (vw=%d l1d=%ld t=%d ord=%d place=%d T=%d): refused %lld, law breaks %lld\n", counted, m.lanes,
               m.l1d, rq.transform, rq.order, rq.placement, rq.nthreads, refused, bad);
        for (i = 0; i < 256; i++)
            if (cnt[i])
            {
                if (rq.transform == VFFT_C2C) printf("  %-6s %ld\n", i <= VW2_OOP_IL_ROUTE_MAX ? vw2_oop_il_name[i] : "?", cnt[i]);
                else printf("  %-6s %ld\n", i >= 16 && i < 16 + EST_NELEM(st_engs) ? st_engs[i - 16] : "?", cnt[i]);
            }
        fflush(stdout);
        return bad ? 1 : 0;
    }
    for (i = 1; i < argc; i++)
    {
        const char *s = argv[i], *bad;
        if (s[0] == '-')
        {
            if (!strcmp(s, "-all")) continue;
            i += !strcmp(s, "-sweep") ? 2 : 1;
            continue;
        }
        rq.N = atoi(s);
        if (!all) { st_one(&m, &rq, 0, &bad); continue; }
        {
            est_request_t q = rq;
            int o, p;
            q.transform = VFFT_C2C;
            for (o = 0; o < 2; o++)
                for (p = 0; p < 2; p++) { q.order = o; q.placement = p; st_one(&m, &q, 0, &bad); }
            q.order = VFFT_ORDER_NATURAL; q.placement = VFFT_PLACEMENT_OUT;
            q.transform = VFFT_R2C; st_one(&m, &q, 0, &bad);
            q.transform = VFFT_C2R; st_one(&m, &q, 0, &bad);
        }
    }
    fflush(stdout);
    return 0;
}
