/* policy_gate.c — the planning policy module answers exactly what the
 * scattered sites answered before it (docs/design/planning_policy_design.md,
 * 2026-09-16).
 *
 * THE REFERENCES BELOW ARE FROZEN. Each `_ref_` function is the law as it
 * was written at its call site immediately before the migration step that
 * moved it, copied verbatim. They are never "fixed" to match the module:
 * if the module and a reference disagree, the MIGRATION is wrong (or a
 * defect has been found and must be ruled on), and this gate fails. They
 * are deleted only with the migration's last step.
 *
 * Step 1 covers L4 (order) and L9 (ceilings); step 2 adds L1/L2, the
 * BAND MAP (which families race in a cell) against the two pools:
 * sampled at every band boundary AND swept exhaustively over N = 2..2^23.
 * Later steps add their own reference arms here.
 *
 * Run:   policy_gate.exe
 * Build: python build.py --compile --vfft --src benches/policy_gate.c */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "vfft.h"
#include "wisdom2.h"
#include "oop/ztt.h"
#include "oop/k1_fourstep_band.h"
#include "ztt_registry_avx2.h"   /* ground truth for ZTURN-T's band */
#include "planning/policy.h"

static int g_fail = 0;
#define CHECK(cond, ...) do { if (!(cond)) { g_fail++; printf("  *** FAIL: "); printf(__VA_ARGS__); printf("\n"); } } while (0)

/* ── L4 references, verbatim from their pre-migration sites ─────────── */
/* oop/c2c_ip_create.h `_ip_order_is_nat` (rank 1, in place) */
static int _ref_ip_order_is_nat(const vfft_config_t *cfg, int N)
{
    return cfg->order == VFFT_ORDER_NATURAL ||
           (cfg->order == VFFT_ORDER_DEFAULT && (N & (N - 1)) == 0);
}
/* oop/k1_commit.h `scr_req` (rank 1, the K=1 engine row) */
static int _ref_k1_scr_req(const vfft_config_t *cfg)
{
    return (cfg->order == VFFT_ORDER_SCRAMBLED);
}
/* transforms/fft2d/fft2d_create.h `il2d_ord`, il2d_tier.h x8, plane_queue.h */
static int _ref_rankn_ord(const vfft_config_t *cfg)
{
    return (cfg->order == VFFT_ORDER_NATURAL) ? VW2_ORD_NAT : VW2_ORD_SCR;
}
/* ── L9 reference, verbatim from oop/k1_commit.h's race gate ─────────── */
#define REF_K1_IL_PLAN_MAX_N 16384
#define REF_K1_IL_PLAN_ODD_MAX_N 262144
static long _ref_race_max_n(int N)
{
    const int pow2 = (N & (N - 1)) == 0;
    const int oddband = vfft_ztt_odd_band(N);
    return (long)(pow2 ? VFFT_K1FS_MAX_N : oddband ? VFFT_ZTT_MAX_N
                       : ((N & 3) ? REF_K1_IL_PLAN_ODD_MAX_N : REF_K1_IL_PLAN_MAX_N));
}


/* ── the scrambled NO-FALLBACK line, as k1_commit.h spelled it ───────── */
static int _ref_scr_writer_band(int N)
{
    return ((N & (N - 1)) == 0 || vfft_ztt_odd_band(N));
}
/* ── the BENCH's own copy of the admission (bench_1d_vs_mkl.c:5643) ───── */
static int _ref_bench_direct_cell(int N)
{
    return (N < 2048 || (N & 3) || vfft_ztt_odd_band(N) ||
            vfft_ztt_band(N) || vfft_k1fs_band(N));
}
/* ── L1/L2 reference: the two pools as dp_planner_il.h spelled them on
 * 2026-09-16, before the band map moved into policy.h. Copied structure for
 * structure. The unconditional `_il_dp_enumerate_ztt(N, s)` at the end of
 * the natural pool's small branch is modelled by its REAL effect — the
 * enumerator walks the registry and emits only where a cell matches N — so
 * this reference asks the registry, and the module asks the band. The arm
 * below proves those two predicates are the same function. */
static int _ref_ztt_registry_has(int N)
{
    int i;
    for (i = 0; i < VFFT_ZTT_NCELLS_AVX2; i++)
        if (vfft_ztt_cells_avx2[i].n == N) return 1;
    return 0;
}
/* ── L1/L9 reference: `_k1_il_plan_race`'s own gate as it stood before
 * 2026-09-16 — the ownership refusal and the ceiling, spelled inline. */
static int _ref_races(int N)
{
    const int pow2 = (N & (N - 1)) == 0;
    const int oddband = vfft_ztt_odd_band(N);
    if (!pow2 && !oddband && N >= 2048 && !(N & 3)) return 0;
    if ((long)N > _ref_race_max_n(N)) return 0;
    return 1;
}
#define REF_PUSH(f) do { if (n < max) out[n] = (f); n++; } while (0)
static int _ref_pool_natural(int N, int with_flat, vfft_fam_t *out, int max)
{
    const int pow2 = (N & (N - 1)) == 0;
    int n = 0;
    if (pow2 && N >= 2048)
    {
        if (N <= VFFT_ZTT_MAX_N) REF_PUSH(VFFT_FAM_ZTT);
        if (vfft_k1fs_band(N))   REF_PUSH(VFFT_FAM_FS);
        return n;
    }
    if (vfft_ztt_odd_band(N)) REF_PUSH(VFFT_FAM_ZTT_ODD);
    REF_PUSH(VFFT_FAM_MONO);
    REF_PUSH(VFFT_FAM_PAIR);
    REF_PUSH(VFFT_FAM_CHAIN3);
    if (with_flat && !pow2 && (N < 2048 || (N & 3))) REF_PUSH(VFFT_FAM_FLAT);
    if (_ref_ztt_registry_has(N)) REF_PUSH(VFFT_FAM_ZTT);
    return n;
}
static int _ref_pool(int N, int ord, vfft_fam_t *out, int max)
{
    const int pow2 = (N & (N - 1)) == 0;
    int n = 0;
    if (ord == VW2_ORD_NAT)
        return _ref_pool_natural(N, 1, out, max);
    if (vfft_ztt_band(N))          { REF_PUSH(VFFT_FAM_ZTT);     return n; }
    if (pow2 && vfft_k1fs_band(N)) { REF_PUSH(VFFT_FAM_FS);      return n; }
    if (vfft_ztt_odd_band(N))      { REF_PUSH(VFFT_FAM_ZTT_ODD); return n; }
    if (N < 2048 || (N & 3))
    {
        n = _ref_pool_natural(N, 0, out, max);
        if (!pow2) { if (n < max) out[n] = VFFT_FAM_FLAT; n++; }
    }
    return n;
}
#undef REF_PUSH
static unsigned _fam_mask(const vfft_fam_t *v, int n)
{
    unsigned m = 0;
    int i;
    for (i = 0; i < n && i < VFFT_FAM_NFAM; i++) m |= 1u << (unsigned)v[i];
    return m;
}

/* ── steps 4-6 (2026-09-16): L3's fence, L6's presence, L8's two ladders ──
 * Frozen the same way: each is the predicate as the call sites spelled it
 * the moment before the migration touched them. */

/* L3: seven sites, five token spellings, one comparison. */
static int _ref_replays_at_T(int banked_T, int T)
{
    return banked_T == T;
}

/* L6: the three doors' lists. The out-of-place door carried its layout
 * gates inside the OR; they are passed in already folded, exactly as the
 * migrated call does, so this is the list itself. */
static int _ref_engine_present_oop(int mono_route_lay, int pair_lay, int il3p,
                                   int ilpr, int ilfd, int ztt, int fs)
{
    return (pair_lay || il3p || ilpr || ilfd || ztt || fs || mono_route_lay) ? 1 : 0;
}
static int _ref_have_k1_ip(int il2, int il3, int ifd, int ztt, int fs,
                           int mono_f, int ilp)
{
    return (il2 || il3 || ifd || ztt || fs || mono_f || ilp) ? 1 : 0;
}
static int _ref_ip_refusal(int il2p, int il3p, int ilpr, int ilfd, int ztt,
                           int fs, int mono_ilf)
{   /* the refusal fired when NOTHING was present */
    return (!il2p && !il3p && !ilpr && !ilfd && !ztt && !fs && !mono_ilf) ? 1 : 0;
}

/* L8: the ladders AS WRITTEN -- note neither carried a cache-unknown term.
 * That term is the module's addition and the sweep below is what proves it
 * changes no answer for a positive working set. */
static int _ref_fits_l2(long bytes)
{
    return bytes <= vfft_cpu_l2_bytes();
}
static int _ref_exceeds_l3(long bytes)
{
    const long l3 = vfft_cpu_l3_bytes();
    return l3 <= 0 || bytes > l3;
}

/* ── rank >= 2 (2026-09-17): R3's two laws and R7, frozen as the sites spelled them ──
 * R3a: _il2d_real_wl_cut / _ilnd_wl_cut, the legality + cut. */
static int _ref_wl_cut(int N, int nst, const int *L, int wl)
{
    int s2;
    if (wl <= 0 || wl > N || N % wl != 0)
        return -1;
    for (s2 = 0; s2 < nst; s2++)
        if (wl % L[s2] == 0)
            return s2;
    return -1;
}
/* R3b: the c2c axis race's recovery loop -- cut = 0 unless a stage divides */
static int _ref_cut_recover(int nst, const int *L, int wl)
{
    int s2, cut = 0;
    if (wl > 0)
        for (s2 = 0; s2 < nst; s2++)
            if (wl % L[s2] == 0)
            {
                cut = s2;
                break;
            }
    return cut;
}
/* R7: the three call sites' literals. rank 2 axis 0: il2d_ord == NAT;
 * rank 3 axis 1: d->nat; rank 3 axis 0: 0. */
static int _ref_axis_nat(int rank, int axis, int ord)
{
    if (rank == 2) return ord == VW2_ORD_NAT;
    if (axis == 1) return ord == VW2_ORD_NAT;
    return 0;
}
/* R2: the c2c cascade loop's spelling (its domain is stage spans) and the 3D one's */
static int _ref_c2c_band(int N1, int w) { return !(w > N1 || N1 % w || w < 8); }
static int _ref_3d_band(int N, int nst, const int *L, int w) { return !(w < 8 || _ref_wl_cut(N, nst, L, w) < 0); }
/* every ordered composition of N over the column radix pool, depth <= 4 --
 * the enumerator's shape, re-spelled here so the gate needs no engine header */
static int _ref_chains(int L, int depth, int *cur, int (*out)[8], int *lens, int *n)
{
    static const int POOL[] = { 64, 32, 16, 8, 4, 27, 25, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3 };
    int p;
    if (L == 1)
    {
        if (depth == 0 || *n >= 64) return 0;
        memcpy(out[*n], cur, 8 * sizeof(int));
        lens[*n] = depth;
        (*n)++;
        return 0;
    }
    if (depth >= 4) return 0;
    for (p = 0; p < (int)(sizeof POOL / sizeof POOL[0]); p++)
        if (L % POOL[p] == 0)
        {
            cur[depth] = POOL[p];
            _ref_chains(L / POOL[p], depth + 1, cur, out, lens, n);
        }
    return 0;
}

int main(void)
{
    /* every band boundary and a spread inside each: pow2 from the solos to
     * the four-step's ceiling, 2^a*odd in and out of ZTURN-T's odd band,
     * pure odd, prime, and the awkward composites */
    static const int NS[] = {
        2, 3, 4, 5, 7, 8, 16, 32, 50, 64, 100, 126, 128, 130, 192, 250, 256, 320,
        384, 500, 512, 768, 1000, 1024, 1536, 2048, 3072, 4096, 6561, 8192, 12288,
        15625, 16384, 16807, 16000, 32768, 49152, 65536, 98415, 131072, 177147,
        196608, 245760, 262144, 393216, 524288, 1048576, 2097152, 4194304,
        8388608, 999, 1001, 2187, 3125, 9973, 65537
    };
    static const int ORDS[3] = { VFFT_ORDER_DEFAULT, VFFT_ORDER_NATURAL, VFFT_ORDER_SCRAMBLED };
    static const char *ONM[3] = { "DEFAULT", "NATURAL", "SCRAMBLED" };
    const int nn = (int)(sizeof NS / sizeof NS[0]);
    int i, o, nchk = 0;

    printf("PLANNING POLICY gate: the module vs the pre-migration sites "
           "(L4 order, L9 ceilings, L1/L2 the band map, L3 the per-T fence, "
           "L6 engine presence, L8 the ladders; rank>=2: R3 the tcut laws, R7 the axis pass), "
           "%d N x %d order classes\n", nn, 3);

    for (i = 0; i < nn; i++)
    {
        const int N = NS[i];
        /* L9: the race ceiling is order- and placement-free */
        CHECK(vfft_policy_race_max_n(N) == _ref_race_max_n(N),
              "N=%d race ceiling: policy %ld != site %ld",
              N, vfft_policy_race_max_n(N), _ref_race_max_n(N));
        nchk++;
        for (o = 0; o < 3; o++)
        {
            vfft_config_t cfg;
            memset(&cfg, 0, sizeof cfg);
            cfg.transform = VFFT_C2C; cfg.dims = 1; cfg.n[0] = N; cfg.howmany = 1;
            cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1;
            cfg.order = ORDS[o];

            /* L4, rank 1 in place: the in-place door's own mode row */
            CHECK((vfft_policy_ord_k1(&cfg, N, 1) == VW2_ORD_NAT) == (_ref_ip_order_is_nat(&cfg, N) != 0),
                  "N=%d ord=%s rank1 ip: policy %s != site %s", N, ONM[o],
                  vfft_policy_ord_k1(&cfg, N, 1) == VW2_ORD_NAT ? "nat" : "scr",
                  _ref_ip_order_is_nat(&cfg, N) ? "nat" : "scr");
            /* L4, rank 1 out of place: the K=1 engine row */
            CHECK((vfft_policy_ord_k1(&cfg, N, 0) == VW2_ORD_SCR) == (_ref_k1_scr_req(&cfg) != 0),
                  "N=%d ord=%s rank1 oop: policy %s != site %s", N, ONM[o],
                  vfft_policy_ord_k1(&cfg, N, 0) == VW2_ORD_SCR ? "scr" : "nat",
                  _ref_k1_scr_req(&cfg) ? "scr" : "nat");
            /* L4, rank >= 2: the 2D/3D tier's order axis */
            CHECK(vfft_policy_ord_rankn(&cfg) == _ref_rankn_ord(&cfg),
                  "N=%d ord=%s rankN: policy %d != site %d", N, ONM[o],
                  vfft_policy_ord_rankn(&cfg), _ref_rankn_ord(&cfg));
            /* the split layout takes the same order law (the cell's layout
             * axis is separate); check it is not accidentally IL-gated */
            cfg.layout = VFFT_LAYOUT_SPLIT;
            CHECK(vfft_policy_ord_rankn(&cfg) == _ref_rankn_ord(&cfg),
                  "N=%d ord=%s rankN split: policy %d != site %d", N, ONM[o],
                  vfft_policy_ord_rankn(&cfg), _ref_rankn_ord(&cfg));
            cfg.layout = VFFT_LAYOUT_INTERLEAVED;
            nchk += 4;

            /* the CELL carries the same answers it was built from */
            {
                const vfft_cell_t c1 = vfft_policy_cell(&cfg, N, 1, 1, 0, 4);
                const vfft_cell_t c2 = vfft_policy_cell(&cfg, N, 1, 2, 0, 8);
                CHECK(c1.ord == vfft_policy_ord_k1(&cfg, N, 0) && c1.N == N && c1.T == 4 &&
                      c1.layout == VW2_LAY_IL && c1.rank == 1,
                      "N=%d ord=%s: cell(rank1) disagrees with its own classifier", N, ONM[o]);
                CHECK(c2.ord == vfft_policy_ord_rankn(&cfg) && c2.rank == 2 && c2.T == 8,
                      "N=%d ord=%s: cell(rank2) disagrees with its own classifier", N, ONM[o]);
                nchk += 2;
            }
        }
    }
    /* ── EXHAUSTIVE arm: every N the library can be asked for, not a
     * sample. The order rules and the race ceiling are pure integer
     * functions of (N, order), so the whole served domain is checkable —
     * and a sampled agreement is not equality. The first disagreement per
     * law is printed; the rest are counted. */
    {
        const int NMAX = 1 << 23;
        long bad_ceil = 0, bad_ip = 0, bad_oop = 0, bad_rankn = 0, n;
        vfft_config_t cfg;
        memset(&cfg, 0, sizeof cfg);
        cfg.transform = VFFT_C2C; cfg.dims = 1; cfg.howmany = 1;
        cfg.layout = VFFT_LAYOUT_INTERLEAVED; cfg.nthreads = 1;
        for (n = 2; n <= NMAX; n++)
        {
            const int Nx = (int)n;
            if (vfft_policy_race_max_n(Nx) != _ref_race_max_n(Nx))
            {
                if (!bad_ceil)
                    printf("  *** FAIL: first ceiling disagreement at N=%d: policy %ld != site %ld\n",
                           Nx, vfft_policy_race_max_n(Nx), _ref_race_max_n(Nx));
                bad_ceil++;
            }
            for (o = 0; o < 3; o++)
            {
                cfg.n[0] = Nx;
                cfg.order = ORDS[o];
                if ((vfft_policy_ord_k1(&cfg, Nx, 1) == VW2_ORD_NAT) != (_ref_ip_order_is_nat(&cfg, Nx) != 0))
                {
                    if (!bad_ip) printf("  *** FAIL: first rank1-ip disagreement at N=%d ord=%s\n", Nx, ONM[o]);
                    bad_ip++;
                }
                if ((vfft_policy_ord_k1(&cfg, Nx, 0) == VW2_ORD_SCR) != (_ref_k1_scr_req(&cfg) != 0))
                {
                    if (!bad_oop) printf("  *** FAIL: first rank1-oop disagreement at N=%d ord=%s\n", Nx, ONM[o]);
                    bad_oop++;
                }
                if (vfft_policy_ord_rankn(&cfg) != _ref_rankn_ord(&cfg))
                {
                    if (!bad_rankn) printf("  *** FAIL: first rankN disagreement at N=%d ord=%s\n", Nx, ONM[o]);
                    bad_rankn++;
                }
            }
        }
        /* L1/L2: the BAND MAP equals the two pools it replaces, for every N
         * and both order classes — as SETS of admitted families. */
        {
            long bad_pool = 0, bad_reg = 0, bad_races = 0, bad_wb = 0, bad_bench = 0;
            for (n = 2; n <= NMAX; n++)
            {
                const int Nx = (int)n;
                vfft_fam_t pm[VFFT_FAM_NFAM], pr[VFFT_FAM_NFAM];
                int oi;
                /* ZTURN-T's band IS its registry — the equivalence the map
                 * relies on to gate the family by the band */
                if ((vfft_ztt_band(Nx) != 0) != (_ref_ztt_registry_has(Nx) != 0))
                {
                    if (!bad_reg) printf("  *** FAIL: ztt band != registry at N=%d (band %d, registry %d)\n",
                                         Nx, vfft_ztt_band(Nx), _ref_ztt_registry_has(Nx));
                    bad_reg++;
                }
                {   /* L1/L9: "may this cell race at all?" — one question now */
                    vfft_cell_t rc2;
                    memset(&rc2, 0, sizeof rc2);
                    rc2.N = Nx; rc2.K = 1; rc2.rank = 1; rc2.T = 1;
                    rc2.layout = VW2_LAY_IL; rc2.ord = VW2_ORD_NAT;
                    if ((vfft_policy_races(&rc2) != 0) != (_ref_races(Nx) != 0))
                    {
                        if (!bad_races)
                            printf("  *** FAIL: first race-gate disagreement at N=%d: policy %d != site %d\n",
                                   Nx, vfft_policy_races(&rc2), _ref_races(Nx));
                        bad_races++;
                    }
                }
                {   /* the writer-band law, and the bench's admission — the
                     * bench's copy is the SAME SET as "the planner races here",
                     * which is why it can ask instead of keeping a copy */
                    vfft_cell_t wc;
                    memset(&wc, 0, sizeof wc);
                    wc.N = Nx; wc.K = 1; wc.rank = 1; wc.T = 1;
                    wc.layout = VW2_LAY_IL; wc.ord = VW2_ORD_SCR;
                    if ((vfft_policy_scr_writer_band(&wc) != 0) != (_ref_scr_writer_band(Nx) != 0))
                    {
                        if (!bad_wb) printf("  *** FAIL: first writer-band disagreement at N=%d\n", Nx);
                        bad_wb++;
                    }
                    wc.ord = VW2_ORD_NAT;
                    if ((vfft_policy_k1_direct_cell(&wc) != 0) != (_ref_bench_direct_cell(Nx) != 0))
                    {
                        if (!bad_bench)
                            printf("  *** FAIL: first bench-admission disagreement at N=%d: policy %d != bench %d\n",
                                   Nx, vfft_policy_k1_direct_cell(&wc), _ref_bench_direct_cell(Nx));
                        bad_bench++;
                    }
                }
                for (oi = 0; oi < 2; oi++)
                {
                    const int ordv = oi ? VW2_ORD_SCR : VW2_ORD_NAT;
                    vfft_cell_t cc;
                    int nm2, nr2;
                    memset(&cc, 0, sizeof cc);
                    cc.N = Nx; cc.K = 1; cc.rank = 1; cc.T = 1;
                    cc.layout = VW2_LAY_IL; cc.ord = ordv; cc.inplace = 0;
                    nm2 = vfft_policy_pool(&cc, pm, VFFT_FAM_NFAM);
                    nr2 = _ref_pool(Nx, ordv, pr, VFFT_FAM_NFAM);
                    if (nm2 != nr2 || _fam_mask(pm, nm2) != _fam_mask(pr, nr2))
                    {
                        if (!bad_pool)
                        {
                            int q;
                            printf("  *** FAIL: first pool disagreement at N=%d ord=%s\n    policy:",
                                   Nx, ordv == VW2_ORD_NAT ? "nat" : "scr");
                            for (q = 0; q < nm2; q++) printf(" %s", vfft_policy_fam_name(pm[q]));
                            printf("\n    site  :");
                            for (q = 0; q < nr2; q++) printf(" %s", vfft_policy_fam_name(pr[q]));
                            printf("\n");
                        }
                        bad_pool++;
                    }
                }
            }
            if (bad_pool || bad_reg || bad_races || bad_wb || bad_bench) g_fail++;
            printf("exhaustive N=2..%d: band map vs the two pools %s (%ld differ), "
                   "ztt band==registry %s, race gate %s\n", NMAX,
                   bad_pool ? "DIFFERS" : "equal", bad_pool, bad_reg ? "DIFFERS" : "equal",
                   bad_races ? "DIFFERS" : "equal");
            printf("exhaustive N=2..%d: scrambled writer band %s, the bench's admission %s\n",
                   NMAX, bad_wb ? "DIFFERS" : "equal", bad_bench ? "DIFFERS" : "equal");
            nchk += (NMAX - 1) * 6;
        }
        if (bad_ceil || bad_ip || bad_oop || bad_rankn) g_fail++;
        printf("exhaustive N=2..%d x 3 orders: ceiling %s, rank1-ip %s, rank1-oop %s, rankN %s\n",
               NMAX, bad_ceil ? "DIFFERS" : "equal", bad_ip ? "DIFFERS" : "equal",
               bad_oop ? "DIFFERS" : "equal", bad_rankn ? "DIFFERS" : "equal");
        nchk += (NMAX - 1) * 4;
    }
    /* ── steps 4-6: the three narrowed laws ─────────────────────────────
     * L3, exhaustive over every (banked, requested) pair a plan can carry,
     * the unraced sentinels included. */
    {
        long bad_t = 0, bad_e = 0, bad_l2 = 0, bad_l3 = 0;
        int bt, rt, m;
        for (bt = -2; bt <= 64; bt++)
            for (rt = -2; rt <= 64; rt++)
                if ((vfft_policy_replays_at_T(bt, rt) != 0) != (_ref_replays_at_T(bt, rt) != 0))
                    bad_t++;
        CHECK(bad_t == 0, "L3 per-T fence: %ld of %d pairs differ", bad_t, 67 * 67);
        nchk += 67 * 67;

        /* L6, every one of the 2^7 presence combinations, against all THREE
         * doors' spellings -- including the refusal, which is the negation */
        for (m = 0; m < 128; m++)
        {
            const int a = (m >> 0) & 1, b = (m >> 1) & 1, c = (m >> 2) & 1,
                      d = (m >> 3) & 1, e = (m >> 4) & 1, f = (m >> 5) & 1,
                      g = (m >> 6) & 1;
            const int pol = vfft_policy_k1_engine_present(a, b, c, d, e, f, g);
            if (pol != _ref_engine_present_oop(a, b, c, g, d, e, f)) bad_e++;
            if (pol != _ref_have_k1_ip(b, c, d, e, f, a, g)) bad_e++;
            if ((pol == 0) != (_ref_ip_refusal(b, c, g, d, e, f, a) != 0)) bad_e++;
        }
        CHECK(bad_e == 0, "L6 engine presence: %ld of %d door checks differ", bad_e, 128 * 3);
        nchk += 128 * 3;

        /* L8, the two ladders over every positive working set that matters:
         * powers of two to 1 GiB and the two cache capacities +/- 1 byte */
        {
            long probes[96];
            int np = 0, q;
            const long l2 = vfft_cpu_l2_bytes(), l3 = vfft_cpu_l3_bytes();
            long v;
            for (v = 1; v > 0 && v <= (1L << 30); v <<= 1) probes[np++] = v;
            probes[np++] = l2 - 1; probes[np++] = l2; probes[np++] = l2 + 1;
            probes[np++] = l3 - 1; probes[np++] = l3; probes[np++] = l3 + 1;
            for (q = 0; q < np; q++)
            {
                if (probes[q] <= 0) continue;   /* the contract: positive only */
                if ((vfft_policy_fits_l2(probes[q]) != 0) != (_ref_fits_l2(probes[q]) != 0))
                    bad_l2++;
                if ((vfft_policy_exceeds_l3(probes[q]) != 0) != (_ref_exceeds_l3(probes[q]) != 0))
                    bad_l3++;
            }
            CHECK(bad_l2 == 0, "L8 fits_l2: %ld of %d working sets differ (L2=%ld)",
                  bad_l2, np, l2);
            CHECK(bad_l3 == 0, "L8 exceeds_l3: %ld of %d working sets differ (L3=%ld)",
                  bad_l3, np, l3);
            /* and the polarity itself, which is the whole reason there are
             * two functions: at a working set BETWEEN L2 and L3 they must
             * BOTH say no -- it does not fit L2, and it does not exceed L3 */
            if (l2 > 0 && l3 > l2)
            {
                const long mid = l2 + (l3 - l2) / 2;
                CHECK(!vfft_policy_fits_l2(mid) && !vfft_policy_exceeds_l3(mid),
                      "L8 polarity: a working set between L2 and L3 must fail BOTH "
                      "(fits_l2=%d exceeds_l3=%d at %ld)",
                      vfft_policy_fits_l2(mid), vfft_policy_exceeds_l3(mid), mid);
                nchk++;
            }
            nchk += 2 * np;
        }
    }

    /* ── rank >= 2: R3 (both laws) over every chain to 4096, R7 over its table ── */
    {
        long bad_a = 0, bad_b = 0, bad_7 = 0, bad_r2 = 0, nchains = 0, nwl = 0;
        int N1, rk, ax, oi;
        for (N1 = 2; N1 <= 4096; N1++)
        {
            int cand[64][8], lens[64], cur[8], nc = 0, ci;
            _ref_chains(N1, 0, cur, cand, lens, &nc);
            for (ci = 0; ci < nc; ci++)
            {
                /* the stage spans as the tier lays them: L[s] = N1 / prod_{u<s} R_u */
                int L[8], u, acc = N1, wl;
                for (u = 0; u < lens[ci]; u++) { L[u] = acc; acc /= cand[ci][u]; }
                nchains++;
                for (u = 1; u < lens[ci]; u++)   /* R2 vs c2c, over the spans the loop visits */
                    if (vfft_policy_il2d_band_ok(N1, lens[ci], L, L[u]) != _ref_c2c_band(N1, L[u])) bad_r2++;
                for (wl = 0; wl <= N1 + 1; wl++)
                {
                    if (vfft_policy_il2d_band_ok(N1, lens[ci], L, wl) != _ref_3d_band(N1, lens[ci], L, wl)) bad_r2++;
                    const int a = vfft_policy_il2d_wl_cut(N1, lens[ci], L, wl);
                    const int b = _ref_wl_cut(N1, lens[ci], L, wl);
                    int r, c;
                    if (a != b) bad_a++;
                    r = vfft_policy_il2d_cut_of(lens[ci], L, wl);
                    c = ((wl > 0 && r >= 0) ? r : 0);      /* the sites' exact spelling */
                    if (c != _ref_cut_recover(lens[ci], L, wl)) bad_b++;
                    nwl++;
                }
            }
        }
        CHECK(bad_a == 0, "R3a wl_cut: %ld of %ld (chain, wl) pairs differ", bad_a, nwl);
        CHECK(bad_b == 0, "R3b cut_of: %ld of %ld (chain, wl) pairs differ", bad_b, nwl);
        CHECK(bad_r2 == 0, "R2 band_ok: %ld disagreements with the c2c / 3D spellings", bad_r2);
        nchk += (int)(2 * nwl);
        for (rk = 2; rk <= 3; rk++)
            for (ax = 0; ax <= 1; ax++)
                for (oi = 0; oi < 2; oi++)
                {
                    const int ord = oi ? VW2_ORD_SCR : VW2_ORD_NAT;
                    if (vfft_policy_rankn_axis_nat(rk, ax, ord) != _ref_axis_nat(rk, ax, ord)) bad_7++;
                    nchk++;
                }
        CHECK(bad_7 == 0, "R7 axis pass: %ld of 8 (rank, axis, ord) cases differ", bad_7);
        printf("rank>=2: %ld chains to 4096, R3 both laws %s over %ld widths, R2 band_ok %s, R7 %s\n",
               nchains, (bad_a || bad_b) ? "DIFFER" : "equal", nwl, bad_r2 ? "DIFFERS" : "equal",
               bad_7 ? "DIFFERS" : "equal");
    }

    printf("%d checks, %s\n", nchk, g_fail ? "*** GATE FAILED ***" : "ALL PASS");
    return g_fail ? 1 : 0;
}
