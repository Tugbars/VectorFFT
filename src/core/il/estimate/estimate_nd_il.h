/* estimate_nd_il.h — THE ESTIMATE at rank 2 and 3: the plan of a 2D or 3D interleaved c2c transform
 * chosen by formulas, with nothing measured (VFFT_FLAG_ESTIMATE, include/vfft.h; the design:
 * docs/design/estimate_2d3d_sketch.md).
 *
 * The 2D door (il/rank2/fft2d_create_il.h) and the 3D door (il/rank3/fftnd_il.h) build a plan from a
 * complete column row with nothing timed: every race sits behind a missing token.
 * est_plan_2d / est_plan_3d compute that row -- the column pass's verdicts (the chain and its forms;
 * the N-arm: the chain, a Rader stage, or the column Bluestein with the turned prime pass), the band
 * width, the 2D row route, the 3D structure and natural form -- and the recipe of every 1D child the
 * door builds, by the 1D estimate's est_plan_1d unchanged. est_put_rows_2d / est_put_rows_3d write it
 * with the store's own writers; the build is c2c_estimate_nd_il.h's.
 *
 * SCOPE: one transform (howmany 1), natural order, either placement (the 2D and 3D rows are
 * placement-blind: one recipe serves both), every axis >= 2, any thread count (est_mt_2d and est_mt_3d
 * thread the one-thread plan); a contiguous batch of them on one thread is that transform's plan, run
 * once per transform (the plane or the volume queue, c2c_estimate_il.h, est__batch_one). Rank 3 is the
 * 2D plan per plane then
 * axis 0 (s=1), or on the even planes near the L2 and the planes with an odd side the flat tier (s=2,
 * est__flat_plane). The front door
 * refuses the rest under the flag (vfft_api.h).
 *
 * THE FORMULAS ARE PURE, as at rank 1: the machine and the request only -- no clock, no store, no
 * allocation. The facts they read are the library's: the column pool the door races
 * (_il2d_enum_rec, il2d_cols.h:210), the Rader stage's admission, the engine's own inner pick and its
 * candidates (vfft_policy_il2d_rader_prime, _ilrd_pick_chain, _ilrd_enum_inners), the kernel resolvers
 * (vfft_il2p_col_forms, vfft_k1_mono_ilc_fn, vfft_il_n1ccs_fn), the band laws of policy_il.h, the 3D
 * door's strip widths (_ilnd_sw_pool, whose L2 bound is the host's: at create the machine's). The
 * build's pre-flight runs the door's own constructors on the recipe, so a recipe a registry change
 * broke is refused there, never raced.
 *
 * POSITION IN vfft.c: after estimate_il.h, which it extends, and after the 2D and 3D tiers whose
 * statics it reads; c2c_estimate_nd_il.h includes it. */
#ifndef VFFT_IL_ESTIMATE_ND_IL_H
#define VFFT_IL_ESTIMATE_ND_IL_H

#include "il/estimate/estimate_il.h"

/* ── the recipe ─────────────────────────────────────────────────────────── */

/* One column pass's verdicts: the column row's axis tokens (il2d_tier.h:3425-3912), tokens in
 * [brackets]. The 2D plan's column pass runs the NATURAL pass at (N1, w = N2); the 3D plan's axis 0
 * the SCRAMBLED pass at (N1, w = N2 N3) for both order classes (vfft_policy_rankn_axis_nat,
 * policy_il.h:787). */
typedef struct est_col_s
{
    int N;                 /* the axis length */
    long long w;           /* the row width in complex: N2 (2D), N2 N3 (3D axis 0) */
    int nat;               /* 1 = the natural pass, 0 = the scrambled pass */
    int R[8], nst;         /* [chain] the chain that serves: N's, or the Bluestein M chain */
    char forms[64];        /* [forms] ([bluforms] when blu > 0): the race's spelling, "" = not written */
    int blu;               /* [blu] -1 = not written (N a power of two), 0 = the chain, M = Bluestein */
    int tpc;               /* [tpc] 1 = the turned prime pass, -1 = not written (no Bluestein) */
    int rd[8], rd_n;       /* [rd] the Rader stage's inner chain of p - 1; rd_n = 0: no Rader stage */
    int rdw;               /* [rdw] its window in columns */
    int wl;                /* [wl] the band width in rows, 0 = unbanded; [tf] = (wl > 0) */
} est_col_t;

/* The 2D plan's row: the column pass, the row route and the children the door builds. */
typedef struct est_plan2d_s
{
    int N1, N2;
    est_col_t col;
    int ro;                /* [ro] 0 the per-row child, 2 the batched rows, 3 the batched two-pass rows */
    int sw;                /* [sw] the unbanded walk's column tile, 0 = untiled */
    int rbk;               /* [rbk] the two-pass rows' tile, KB */
    int turn;              /* [turn] the whole plane through the 1D plan at N1 */
    int csk;               /* [csk] the skewed column pass */
    int has_turn;          /* the door builds the turn plan: natural, no Bluestein, N2 has an n1ccs
                            * kernel (fft2d_create_il.h:544) */
    int has_csk;           /* the door builds the skewed pass's row plan: no Bluestein, N1 has a mono
                            * kernel (fft2d_create_il.h:584-588) */
    int T;                 /* the threads (est_mt_2d): 0 or 1 = one thread; T > 1 = the row's key nthreads=T and
                            * the threaded verdict below (fft2d_create_il.h:1144-1152) */
    int cmt;               /* [cmt] T > 1: the column pass threaded (1) or serial (0) */
    int mtarm;             /* [mtarm] T > 1, threaded: 0 the one threaded walk (the skew's, the turn's), 1 the strips */
    int msw;               /* [msw] the strips' width in columns, 0 = each worker's whole range */
    int nls;               /* [nls] T > 1: the natural leaf staged (1) or strided (0); -1 = not written (no natural
                            * leaf: the door banks it only beside one, il2d_tier.h:4517) */
    est_recipe_t rp;       /* [rp_]   N2, in place, natural, T = 1: always */
    est_recipe_t turnp;    /* [turn_] N1, in place, natural: has_turn */
    est_recipe_t cskp;     /* [csk_]  N2, out of place, natural: has_csk */
    est_recipe_t tpcp;     /* [tpc_]  N1, in place, natural: col.tpc == 1 */
} est_plan2d_t;

/* The 3D plan's row: axis 0, the structure, the natural form, the plane child. */
typedef struct est_plan3d_s
{
    int N1, N2, N3;
    est_col_t col;         /* axis 0; with the strips its band is 0 and [wl] [tf] are not written */
    int s;                 /* [s] 1 = the 2D plan per plane, 2 = the flat tier (est__flat_plane) */
    int nf;                /* [nf] 1 = the cycle walk, 2 = the strips */
    int nsw;               /* [nsw] the strip width in columns (nf = 2); 0 = not written (nf = 1) */
    est_plan2d_t plane;    /* [plane_] the 2D plan at N2 x N3, natural, with its own children; s = 2 writes
                            * only its row plan, under rp_ */
    est_col_t col1;        /* [chain1 forms1 blu1] s = 2: axis 1, the plane plan's column pass unbanded (the
                            * door bands axis 0 alone, fftnd_il.h:1752) */
    est_recipe_t tpcp;     /* [tpc_] N1, in place, natural: col.tpc == 1 */
    int T;                 /* the threads (est_mt_3d): 0 or 1 = one thread; T > 1 = the row's key nthreads=T and
                            * the threaded verdict below (fftnd_il.h:1778-1802) */
    int cmt;               /* [cmt] T > 1: 0 serial (the one-thread plan above), 2 the plane arm */
    int cmts;              /* [cmts] T > 1: the structure the threaded verdict runs with */
    int cmtf;              /* [cmtf] T > 1: the natural form at T, 1 the cycle walk, 2 the strips */
    int mtnsw;             /* [nsw] cmt = 2 with the strips: their width at T, written over the one-thread nsw
                            * (the row carries one nsw; fftnd_il.h:1785-1786); 0 otherwise */
} est_plan3d_t;

/* ── the estimate's constants (docs/design/estimate_2d3d_sketch.md) ────── */

/* the column chain's regimes: a row is NARROW while a radix-16 butterfly across it fits the L1d,
 * 16 x 16 w bytes <= L1d (sketch §4) */
#define EST_ND_R16 16
/* the scrambled pass's one-stage bound: above N = 16, every radix (not only the blocked radix-32
 * and radix-64 kernels), the plane within this fraction of the L1d (sketch §4 step 1; which radices
 * it should cover is open, sketch §14) */
#define EST_ND_BLOCKED_L1_DIV 4
/* the leading stage of a power of two from 2^s1: 32 on narrow rows, 16 on wide ones (sketch §4 step 3) */
#define EST_ND_LEAD_NARROW 32
#define EST_ND_LEAD_WIDE 16
/* the 2D band: none while the plane fits L2 / EST_ND_2D_PLANE_DIV (the 3D plan's plane child:
 * L2 / EST_ND_3D_PLANE_DIV); the band within L2 / EST_ND_2D_BAND_DIV (sketch §7) */
#define EST_ND_2D_PLANE_DIV 4
#define EST_ND_3D_PLANE_DIV 2
#define EST_ND_2D_BAND_DIV 4
/* the 2D row route (sketch §8): the skewed pass up to N1 = 16, and at the odd N1 up to 31 on rows of at
 * least 128 but at the kernels that spill (est__spills_single); the turn from N1 = 128 at N2 <= 8; the
 * batched rows up to N2 = 16, the batched two-pass rows up to N2 = 64; the two-pass tile 8 KB. The odd
 * band is of 2026-10-26: the planner banks the skewed pass at the columns 17, 19, 21, 23, 29, 31 on rows of
 * 128-2048 (the estimate without it 1.18-1.51 x the planner in 13 of 13 cells) and not on narrow rows
 * (29x64, 21x21, the 21x64 and 21x21 planes of 21x21x64 and 21x21x21); over the 226 rows of the store
 * that carry csk this agrees in 208 against 195 for "N1 <= 16" alone. Even N1 in 17..31 were not
 * measured and keep the old bound. */
#define EST_ND_CSK_MAX 16
#define EST_ND_CSK_ODD_MAX 31
#define EST_ND_CSK_ODD_N2 128
#define EST_ND_TURN_N1 128
#define EST_ND_TURN_N2 8
#define EST_ND_ROWB_MAX 16
#define EST_ND_ROWB2_MAX 64
#define EST_ND_RBK 8
/* the Rader stage (est__rader, 2026-10-26): the window EST_ND_RD_W columns; the inner chain the engine's
 * pick unless a radix passes EST_ND_RD_MAX_RADIX, an odd radix passes EST_ND_RD_ODD_MAX, or the first
 * radix is EST_ND_RD_MAX_RADIX or more. Every inner chain x the windows 8-256 timed on 26 2D cells (primes
 * 53-509, 64-2048 columns; docs/research/estimate_threaded/evidence/nd/rader2d/): W32 the best or within
 * 1 % at 23 of 26; the chains failing a check lose 6-86 % (4.15, 4.25, 21.6, 25.10: odd kernels that
 * spill; 4.64, 3.64; 32.3). Against the planner 1.20 -> 1.00 (geomean), the worst 1.86 -> 1.08 (193x256,
 * whose planner takes the three-stage 4.4.12). */
#define EST_ND_RD_W 32
#define EST_ND_RD_MAX_RADIX 32
#define EST_ND_RD_ODD_MAX 13
/* the 3D natural form (sketch §9): the strips when the volume passes the L3 and axis 0's column is
 * short, 16 N1 bytes within L1d / EST_ND_NF_L1_DIV; the strip width the widest of the door's whose
 * strip scratch, 16 N1 w bytes, fits L2 / EST_ND_NSW_L2_DIV */
#define EST_ND_NF_L1_DIV 4
#define EST_ND_NSW_L2_DIV 2
/* the 3D structure (est__flat_plane): the flat tier on the even planes of EST_ND_FLAT_LO_PCT to
 * EST_ND_FLAT_HI_PCT percent of the L2 (2026-10-26, s=1 against s=2 timed on 64 cells, 15 rounds, one
 * thread; docs/research/estimate_threaded/evidence/nd/s_confirm/). The even planes in that band: s=2
 * 0.82-1.00 x s=1 in 15 of 15 (240^2 at N1 16 32, 150x400, 280^2, 288^2, 300^2 at N1 8 16 32, 320^2,
 * 336^2, 352^2, 360^2 at N1 8 16, 384^2, 400^2); below it 0.96-1.06 (100^2 1.06, 192^2 1.03), above it
 * 448^2 1.04.
 * Not the powers of two (1 MB 0.95-1.05, 2 MB 1.05-1.11). A cube past the L3 keeps the planes up to
 * EST_ND_FLAT_HI_L3_PCT: the estimate's own s=2 against its s=1 over 27 cells (e_flat2.c, 21 rounds) --
 * within the L3 0.90-0.98 in 6 of 6; past it, planes below 1.5 MB 0.92-1.00 in 14 of 14, planes of 1.5 MB
 * and more 0.95-1.03 (320^2 1.01, 400^2 1.01 and 1.03, 384^2 1.02).
 * THE PLANES WITH AN ODD SIDE (2026-10-27) take the flat tier at any size up to the same top, once their
 * chains carry no 21, 25 or 27 (est__pool): the estimate's own s=2 against its s=1, same run, 15 rounds,
 * 76 cells of 7 KB to 2.5 MB (N1 3-301). Where the plane plan runs no skewed column pass, 63 of 66 at
 * 0.80-1.02 (64x27^2 0.80, 9x21^2 0.83, 25^3 0.86, 27^3 0.85, 105^2 at N1 32 0.88, 275^2 0.87, 145^3
 * 0.95, 301^3 0.97), 3 at 1.05-1.10 (31^3, 243^2 and 297^2 at N1 8); where it skews it (a one-stage
 * column, has_csk) 7 of 10 lose 1.10-1.46 (every plane of 15 rows: 4x15x256 1.46, 15^2x4096 1.20; 9^3
 * 1.37, 7x11x13 1.26), the flat tier having no skewed pass. The planner banks s=2 at 18 of the 20 odd-side
 * cubes in the store (s=1 at 75^3, a tie, and 15^2x4096). With 25 and 27 in their chains the same planes
 * had lost (105-405: 1.04-1.23 in 15 of 17). Nor when the rows are one odd solo kernel (N3 odd in the
 * solo list, not spilling): 9^3, 15^3, 31^3 and 7x11x13 lose 5-37 %, and the planner banks s=1 at all
 * four. */
#define EST_ND_FLAT_LO_PCT 40
#define EST_ND_FLAT_HI_PCT 125
#define EST_ND_FLAT_HI_L3_PCT 75
/* THE 2D PLAN AT T > 1 (est_mt_2d, 2026-10-29): the one-thread plan, then how it threads, from the data's
 * size against the caches and the count of columns only -- how a thread count scales is the machine's, so
 * no rule is fitted to it. Serial while the plane fits L1d / EST_ND_MT_SERIAL_L1_DIV, or on two rows; the
 * turn (where the door builds one) on at most EST_ND_MT_TURN_PER_T x T columns, too few to share; the
 * skewed pass and the turn their one threaded walk; the chain walk as strips (est__msw_2d), which the door
 * runs unbanded (il2d_tier.h:1643), so with the unbanded walk's chain: no band, est__pick's radix-8 regime
 * as est_blu_chain takes it (a radix-32 stage of an unbanded pass past the L2 runs 64 streams 16 KB apart,
 * il2d_tier.h:1898-1903). The strips' width: an 8-column strip (EST_ND_MT_SW_MIN, the door's narrowest)
 * within L1d / EST_ND_MT_SW_L1_DIV while the transform's in and out fit the L2 -> the widest within that
 * fraction of the L1d (the column passes stay in the L1); else the widest whose strip's in and out fit the
 * L2 (past the L2 an 8-column strip reads 128 B of each row and jumps to a new 4 KB page). Timed against
 * the planner's banked plan at T = 8 on the 159 power-of-two planes (sides 2-8192, at most 2^22 points),
 * one process, 15 rounds (docs/research/estimate_threaded/evidence/nd/mt2d/): 1.013 (geomean, median
 * 1.005, control 1.000); the planner's own 8-thread recipe under the same rules 1.015. Each rule's share:
 * the turn (2-16 columns from 256 rows: 1.56-2.50 -> 0.96-1.02), the cache width against the widest rung
 * 1.079 -> 1.039 overall, the L2 clause 1.153 -> 0.993 on the 9 planes it moves (32-128 rows, 1024-8192
 * columns), the unbanded chain 1.075 -> 1.031 on the 62 planes it moves (the leftover factor in a larger
 * last stage instead: 1.070). Left above 1.10: the planes the planner tiles or blocks (8192x32-256,
 * 2048x256, 256x256, 1024x64, 512x128: 1.10-1.23, the planner's recipe under these rules too), 256x128
 * (1.44: strips of 16, the planner's 8) and 32 rows over 128-1024 columns (1.06-1.19: the planner's skewed
 * pass at 8 threads). Past the powers of two, 24 planes (composite, odd, Rader and Bluestein columns; the
 * planner raced at T = 8 into the store): 1.068 (median 1.00; the unbanded chain on their chains against
 * the one-thread chain 1.004). Above 1.10: 6000x500 2.16 (no rung divides 500, so each worker's whole range,
 * 6 MB, past the L2; the planner tiles), the tiled 100x100, 3000x200, 1920x1080, 3024x512 (1.11-1.38), and
 * 643x256 1.46 (the turned prime pass, serial in both plans: its Bluestein chain).
 * THE SKEWED PASS THE DOOR RUNS SERIAL (2026-10-30): the door threads the skewed pass on whole groups of
 * EST_ND_MT_CSK_COLS columns only (_il2d_csk_exec_mt, il2d_tier.h:1432), so where N2 is not a multiple of
 * it the skew is dropped and the plain chain walk threads in blocks (mtarm 0, the planner's split there),
 * one stage of N1 where the pool has it (the planner's block plans keep it: 16x4050 with the one-thread
 * 4.4 3.06, with 16 0.96). Timed at T = 8 on the 77 such planes of 3-16 rows (L1d to L3), one process, 15
 * rounds: 2.105 -> 0.876 (median 1.00, control 0.994); where the planner banked serial (20) 0.50 (16x4093
 * 0.15, 8x8100 0.20; 6x360 and 16x147 lose 1.28 and 1.36), where it blocks 1.07. Blocks do not fix the
 * turn the door runs serial (on N1 not a multiple of EST_ND_MT_TURN_N1_DIV, :1458; 243, 625, 2187 x 12:
 * 1.49-1.65 in blocks, 1.22-2.17 serial): the tile below does.
 * THE TILE ON A TALL PLANE (2026-10-30): where N1 >= EST_ND_MT_TILE_N1_PER_N2 x N2 the chain walk takes the
 * door's tile (mtarm 2: the first stage across the plane, then each worker one sub-problem with the leaf's
 * rows fused, il2d_tier.h:1547) instead of the strips, no band, the unbanded walk's chain with its first
 * stage the pool's smallest first radix of at least T where the chain leads with less (the tile runs
 * min(R0, T) workers: 1500x20 strips 3.30, the tile with 5.5.5.12 1.00, with 15.5.5.4 0.94); so does the
 * turn the door runs serial. Tile against strips at T = 8, one process, 15 rounds, by N1 / N2 (236 planes
 * of the power-of-two and grid studies): 64 and up 0.57, 16-64 0.70, 4-16 0.85, 1-4 1.05, below 1 1.25-1.83
 * (the strip of a worker past the L2: the tile 14 of 14). The rule's threshold read off those planes (1.153
 * -> 1.039 against the planner; per plane the better of the two 1.000), then held on 84 planes no study
 * had used, raced into the store first (14 per N1 / N2 band, 128 KB to 48 MB): 1.433 -> 1.203, the better
 * of the two per plane 1.146; tile against strips 0.57, 0.71, 0.87, 0.98, 1.31-1.51 by the same bands; a
 * rule on the strip's bytes against the caches held worse (1.236). Then 76 more tall planes no study had
 * used (N1 >= 4 N2, raced first): the plan before the tile 2.047 -> 1.212, except a chain with a Rader stage,
 * where the tile ran 1.32-4.03 x the strips on 7 of 7 (1282x84 641.2, 2307x75 769.3; the planner strips
 * all 7, the strips 1.01 of it), so a Rader chain keeps the strips, and so does the turn the door runs
 * serial on one (579x15 193.3: the tile 1.57, the serial turn 3.04, the planner's strips 1.00). The other
 * 69: 2.199 -> 1.133 (N1 / N2 4-8: better on 12 of 18, worse on none).
 * THE TILE'S CHAIN (2026-10-30): R0 sub-problems on T workers take ceil(R0 / T) rounds, so a first radix
 * that is not a multiple of T leaves workers idle (9 at T = 8: two rounds, 0.56 of the workers busy); there
 * the first radix of at most EST_ND_MT_TILE_R0_MAX that shares best, the larger on a tie, and the rest in
 * the one-thread regime (fewer, larger stages: 1440x160 9.8.5.4 -> 32.5.9, the planner's). Read off the 192
 * tiled planes of the studies above (1.111 against the planner, median 1.081), then held on 50 tall planes
 * no study had used, raced into the store first: 1.202 -> 1.062 (median 1.195 -> 1.066, control 0.997),
 * faster by more than 10% on 25, slower on 2 (693x66, 768x52); 9 -> 15 0.79 (16 planes), 9 -> 16 0.74,
 * 9 -> 8 0.77, a first radix of 8 kept (the rest only) 0.98. Left: on 8-led planes of N1 not a power of two
 * the planner leads with 16 or 32 (1120x90 32.5.7 against 8.4.5.7: 1.20 over 17 such planes); 8 -> 32
 * there lost on the powers of two (1.19) and below 512 rows (1.35). */
#define EST_ND_MT_SERIAL_L1_DIV 2
#define EST_ND_MT_TURN_PER_T 2
#define EST_ND_MT_SW_MIN 8
#define EST_ND_MT_SW_L1_DIV 2
#define EST_ND_MT_CSK_COLS 16
#define EST_ND_MT_TURN_N1_DIV 4
#define EST_ND_MT_TILE_N1_PER_N2 4
#define EST_ND_MT_TILE_R0_MAX 32
/* THE 3D PLAN AT T > 1 (est_mt_3d, 2026-10-31): the one-thread plan -- its structure and plane plan unchanged,
 * its axis-0 chain too but on the strips below -- threaded with the door's own partition: one thread while the
 * cube, 16 N1 N2 N3 bytes, times
 * the workers that can take part, min(N1, N2 N3, T), fits EST_ND_MT3_SERIAL_KB x T (the bound grows where
 * fewer than T take part: twice it at 4 workers of 8, four times at 2); else the plane arm (axis 0 in column
 * strips on min(N2 N3, T) workers, then the planes on min(N1, T); the door never races the band arm,
 * fftnd_il.h:80) on the full team, with the one-thread structure; the natural form the door's own, the strips
 * wherever its strip pool is not empty and the cycle walk where it is (fftnd_il.h:83), the strips' width the
 * widest of the pool whose scratch, 16 N1 w bytes, fits L2 / EST_ND_MT3_NSW_L2_DIV, else the narrowest.
 * Timed at T = 8 against the planner's plans on 299 cells raced into the store first (1.4 KB to 256 MB in five
 * regions by the cube against the L1d, L2 and L3; cubes, flat, tall and other shapes, half not powers of two,
 * Rader and Bluestein axis 0), one process, 15 rounds, the hyperthread siblings held
 * (docs/research/estimate_threaded/evidence/nd/3d_night/): the threading rules on the planner's own one-thread
 * plans 1.004 (15 cells above 1.10), the estimate before the strips' chain below 1.019 (each region at most
 * 1.047; 55 above 1.10, 41 of them the plan it keeps from one thread: 29 the axis-0 chain, which the planner
 * takes otherwise at 8 threads
 * (on powers of two more, smaller stages with the largest radix last: 2048x8x8 8.8.4.8 against 32.8.8, 1.25,
 * where its one-thread row banks the estimate's 32.8.8); 8 the plane plan on flat cells, 8x8x16384 1.85; 4 the
 * structure). Each rule against its rivals: one thread or threads on 94 cells timed both ways (8-512 KB): the
 * cube within 24 KB (L1d / 2, the 2D rule) 1.051 against the faster of the two, the
 * bound above 1.011 (cells of 34-37 KB stay mixed whatever the bound: 52x7x6 1.41, 63x6x6 1.28); the full
 * team: the half team where a plane passes the L2 cost 8-11 % past 16 MB (7x32x8819 2.4x), it won 3 of 63
 * races, and the old rows that banked it, timed again, 0.995 of the full team; the one-thread structure
 * against always the child 1.009 and 1.046 at 2-16 MB (15x256x256 1.73); every width of the pool timed on 70
 * cells: L2 / 4 1.019 against the best on the 30 it was read off and 1.024 on 40 fresh cells, the one-thread
 * width 1.048, L2 / 2 1.049 (1024x64x32: 16 columns 0.71 of the planner's 128).
 * THE STRIPS' CHAIN (2026-10-31): where every side is a power of two and axis 0 runs threaded strips with more
 * columns than workers (N2 N3 > T), its chain is est__pick's radix-8 regime, the chain the 2D estimate gives
 * its threaded strips (est__unbanded_chain_2d); the planner's 8-thread rows take more, smaller stages there,
 * while their one-thread rows bank the one-thread chain. Timed on 108 power-of-two cells (the night's 88 and
 * 20 of 8-32 KB raced fresh), one process, 15 rounds, control 1.002
 * (docs/research/estimate_threaded/evidence/nd/3d_night/chain_study/): 1.035 -> 1.001 against the planner
 * (median 1.004; above 1.10 26 -> 15), 48 KB-2 MB 1.085 -> 1.013, tall cells 1.046 -> 0.989; of the 41 cells
 * it changes 33 faster (2048x8x8 32.8.8 -> 4.8.8.8 0.75), 1 slower. Not with one column per worker (N2 N3 <=
 * T: 3-10 % slower on 5 of 5, the column pass a 1D transform per worker), nor on the cycle walk at T (1.167 ->
 * 1.338 on 11 cells), nor past the powers of two (10 cells of the tall shapes it would change, among them
 * 128x6x5: 5 faster, 4 slower, 0.998).
 * THE CYCLE WALK'S CHAIN (2026-10-31): where every side is a power of two and the threaded plan runs the cycle
 * walk on a chain of two stages or more (no strip width fits: N1 above 16384), axis 0 takes the pool's chain
 * ending in EST_ND_MT3_CYCLE_LAST, the door's largest column radix, the stages before it the most balanced,
 * ascending (32768: 8.8.8.64, the planner's own; 65536: 8.8.16.64). Timed on 57 such cells, N1 17500-262144,
 * 2-256 MB (40 raced fresh for it), one process, 15 rounds, control 1.006 (.../3d_night/chain_study/): the 27
 * whose sides are all powers of two 1.118 -> 0.986 against the planner (the planner's own chains 1.003),
 * faster on 24, slower by more than 2 % on 1 (262144x2x8 0.953 -> 0.979); 2-16 MB 1.158 -> 0.965, 16-36 MB
 * 1.077 -> 0.993, past 36 MB 1.118 -> 1.003. The radix-8 regime of the strips loses there (1.147), as do the
 * one-thread stages ascending (1.058) and descending (1.185). The other 30 stand at 0.993 with the one-thread
 * chain; the largest radix first, the planner's habit on them, is mixed by region, so they keep it. */
#define EST_ND_MT3_SERIAL_KB 32
#define EST_ND_MT3_NSW_L2_DIV 4
#define EST_ND_MT3_CYCLE_LAST 64

/* the scrambled pass's exceptions (sketch §4 step 5): the races repeat these where the formula names
 * another chain; keyed by N and the row regime (1 = narrow) */
static const struct { int N, narrow, nst, R[4]; } est_col_exc[] = {
    { 512, 1, 3, { 4, 8, 16, 0 } },
    { 1024, 0, 4, { 8, 8, 4, 4 } },
    { 4096, 0, 3, { 16, 32, 8, 0 } },
    { 8192, 1, 3, { 32, 8, 32, 0 } },
    { 8192, 0, 4, { 32, 8, 4, 8 } },
};

/* ── the column chain ───────────────────────────────────────────────────── */

/* the door's column pool at N (il2d_cols.h:160-253): every ordered composition over the column
 * radices, at most 4 stages, a closing-only radix last, a Rader prime first */
typedef struct
{
    int c[VFFT_IL2D_POOL_MAX][8];
    int len[VFFT_IL2D_POOL_MAX];
    int n;
} est_pool_t;
static inline void est__pool(int N, est_pool_t *P)
{
    int cur[8], dropped = 0, i, j, k = 0, bad;
    P->n = 0;
    memset(cur, 0, sizeof cur);
    if (N >= 2)
        _il2d_enum_rec(N, 0, cur, P->c, P->len, &P->n, &dropped);
    /* no chain with a spilling radix (est_radix_spills: 21, 25, 27, ...) while another chain exists */
    for (i = 0; i < P->n; i++)
    {
        for (bad = 0, j = 0; j < P->len[i]; j++)
            bad |= est_radix_spills(P->c[i][j]);
        if (bad) continue;
        if (k != i)
        {
            memcpy(P->c[k], P->c[i], sizeof P->c[k]);
            P->len[k] = P->len[i];
        }
        k++;
    }
    if (k)
        P->n = k;
}
static inline int est__in_pool(const est_pool_t *P, const int *R, int nst)
{
    int i;
    for (i = 0; i < P->n; i++)
        if (P->len[i] == nst && !memcmp(P->c[i], R, sizeof(int) * (size_t)nst))
            return 1;
    return 0;
}
/* a narrow row: a radix-16 butterfly across the row fits the L1d */
static inline int est__narrow(const est_machine_t *m, long long w)
{
    return (long long)EST_ND_R16 * 16 * w <= (long long)m->l1d;
}
/* THE ODD COMPOSITE KERNELS THAT SPILL (2026-10-26): 25 and 27 run as one whole-axis kernel lose to
 * their split (5.5, 3.9 / 9.3) -- as a single column or plane stage 1.15-1.82 x the planner's plan in 7 of
 * 7 cells (25x25, 27x27, 25x25x25, 27x27x27, 25x25x64, 64x25x25, 27x27x64, 64x27x27), and no planner row
 * of the 14900KF store banks either as a single stage; the skewed pass, which runs the column as one such
 * kernel, is not banked at N1 = 25 or 27 either. A single 21 is not among them (21x21 0.99, 21x21x64
 * 1.01). */
static inline int est__spills_single(int N) { return N == 25 || N == 27; }
/* one stage at N may serve: below 16 always; else the plane within the L1d, on the scrambled pass
 * every N above 16 within L1d / EST_ND_BLOCKED_L1_DIV (the caller checks the pool) */
static inline int est__single_ok(const est_machine_t *m, int N, long long w, int nat)
{
    const long long plane = 16LL * N * w;
    if (N < 16)
        return 1;
    return plane <= ((nat || N <= 16) ? (long long)m->l1d : (long long)m->l1d / EST_ND_BLOCKED_L1_DIV);
}
static inline int est__ceil_lg(long long n) { return est_lg(n) + !est_is_pow2(n); }
/* the balance of a chain: the sum of (log2 R)^2 over its stages (the smaller, the more balanced) */
static inline double est__sumsq(const int *R, int n)
{
    double s = 0.0;
    int i;
    for (i = 0; i < n; i++)
    {
        const double l = log((double)R[i]) / log(2.0);
        s += l * l;
    }
    return s;
}
/* the stages out of the order (asc = the largest last), a closing-only last radix aside */
static inline int est__inversions(const int *R, int n, int asc)
{
    int i, j, k = 0;
    if (n > 0 && _il2d_pool_closing_only(R[n - 1]))
        n--;
    for (i = 0; i < n; i++)
        for (j = i + 1; j < n; j++)
            if (asc ? R[i] > R[j] : R[i] < R[j])
                k++;
    return k;
}
/* one candidate's rank in the pick below, the smaller first: the distance of its stage count from
 * the target, its stage count, its balance, its stages out of the order, its place in the pool */
typedef struct { int d, k; double s; int inv, i; } est__rank_t;
static inline int est__rank_lt(const est__rank_t *a, const est__rank_t *b)
{
    if (a->d != b->d) return a->d < b->d;
    if (a->k != b->k) return a->k < b->k;
    if (a->s < b->s - 1e-9 || a->s > b->s + 1e-9) return a->s < b->s;
    if (a->inv != b->inv) return a->inv < b->inv;
    return a->i < b->i;
}
/* The pool's chain at N that begins with pre[0..np): the rest (product Nrest) one stage when the
 * pool holds it and it may serve (single, est__single_ok), else the chain of two stages or more first
 * by: the stage count nearest k* (ceil(log2 Nrest / 4) on narrow rows, / 3 on wide ones or when
 * wide is forced; an odd rest: the fewest stages a chain without a 15 reaches when that is fewer, and
 * then no chain with a 15), the fewer on a tie; the least sum of (log2 R)^2; the order (ascending,
 * except the natural pass on wide rows: descending); the pool's order. A rest with no such chain takes
 * its one stage when the pool holds it. Returns the stage count into R, 0 = none.
 * The odd rest (2026-10-27): log2 / 3 counts an odd length's stages as if they were radix 8, so wide
 * rows took one stage too many (81 -> 3.3.9, 99 -> 3.3.11) where the planner runs 9.9 and 11.9 or
 * 9.11. Never through a 15: the planner splits 3 x 5 in its column chains (75 -> 5.3.5, 105 -> 7.3.5,
 * 135 -> 5.3.9), and the fewest stages through a 15 lost 1.06-2.09 x the old chain (225x16x16 15.15,
 * 135x16x16 9.15, 105x16x64 7.15, 75x32x32 5.15, 225x256 15.15). Same process, the old rule against
 * this one at the 21 of 39 timed shapes it changes: 0.964 x the old time, 1.074 -> 1.033 against the
 * planner (81^3, 99^3, 81x81x256, 99x99x243, 3x81x256 0.85-0.97); rows of 243 lose (81x243 1.19,
 * 99x243 1.13). */
static inline int est__has15(const int *R, int n)
{
    int i;
    for (i = 0; i < n; i++)
        if (R[i] == 15)
            return 1;
    return 0;
}
static inline int est__pick(const est_pool_t *P, const int *pre, int np, int Nrest, const est_machine_t *m,
                            long long w, int nat, int wide, int single, int *R)
{
    const int narrow = !wide && est__narrow(m, w);
    const int bits = est__ceil_lg(Nrest), asc = !(nat && !narrow);
    int kt = narrow ? (bits + 3) / 4 : (bits + 2) / 3, no15 = 0;
    int i, best = -1, one = -1;
    est__rank_t b, r;
    memset(&b, 0, sizeof b);
    if (Nrest & 1)   /* an odd rest: the fewest stages a chain without a 15 reaches */
        for (i = 0; i < P->n; i++)
        {
            const int k = P->len[i] - np;
            if (k < 2 || k >= kt || (np && memcmp(P->c[i], pre, sizeof(int) * (size_t)np)) || est__has15(P->c[i] + np, k))
                continue;
            kt = k;
            no15 = 1;
        }
    for (i = 0; i < P->n; i++)
    {
        const int k = P->len[i] - np;
        if (k < 1 || (np && memcmp(P->c[i], pre, sizeof(int) * (size_t)np)))
            continue;
        if (k == 1)
        {
            if (P->c[i][np] == Nrest)
                one = i;
            continue;
        }
        if (no15 && est__has15(P->c[i] + np, k))
            continue;
        r.d = k > kt ? k - kt : kt - k;
        r.k = k;
        r.s = est__sumsq(P->c[i] + np, k);
        r.inv = est__inversions(P->c[i] + np, k, asc);
        r.i = i;
        if (best < 0 || est__rank_lt(&r, &b))
        {
            best = i;
            b = r;
        }
    }
    if (one >= 0 && (best < 0 || (single && est__single_ok(m, Nrest, w, nat) && !est__spills_single(Nrest))))
        best = one;
    if (best < 0)
        return 0;
    memcpy(R, P->c[best], sizeof(int) * (size_t)P->len[best]);
    return P->len[best];
}
/* The column chain at N over rows of w complex on the natural (nat = 1) or scrambled pass
 * (sketch §4): one stage when it may serve; a Rader prime first, then the rest; the scrambled pass's
 * exceptions; a power of two from 2^s1 led by 32 (narrow rows) or 16; else the pick above. 0 = the
 * pool holds no chain at N (the door's column Bluestein). */
static inline int est_col_chain(const est_machine_t *m, int N, long long w, int nat, int *R)
{
    est_pool_t P;
    int k, i, pre[1] = { 0 }, one = 0;
    est__pool(N, &P);
    if (P.n == 0)
        return 0;
    if (VFFT_IL2D_RADER_RADIX(P.c[0][0]))
    {   /* the Rader leaf: every candidate is [p] + a composition of N / p (il2d_cols.h:217-232) */
        pre[0] = P.c[0][0];
        if (N == pre[0])
        {
            R[0] = N;
            return 1;
        }
        return est__pick(&P, pre, 1, N / pre[0], m, w, nat, 0, 1, R);
    }
    for (i = 0; i < P.n; i++)
        if (P.len[i] == 1)
            one = 1;
    if (one && est__single_ok(m, N, w, nat) && !est__spills_single(N))
    {
        R[0] = N;
        return 1;
    }
    if (!nat && est_is_pow2(N))
        for (i = 0; i < EST_NELEM(est_col_exc); i++)
            if (est_col_exc[i].N == N && est_col_exc[i].narrow == est__narrow(m, w) &&
                est__in_pool(&P, est_col_exc[i].R, est_col_exc[i].nst))
            {
                memcpy(R, est_col_exc[i].R, sizeof(int) * (size_t)est_col_exc[i].nst);
                return est_col_exc[i].nst;
            }
    if (est_is_pow2(N) && N >= (1 << est_s1(m)))
    {
        pre[0] = est__narrow(m, w) ? EST_ND_LEAD_NARROW : EST_ND_LEAD_WIDE;
        if ((k = est__pick(&P, pre, 1, N / pre[0], m, w, nat, 0, 1, R)) > 0)
            return k;
    }
    return est__pick(&P, pre, 0, N, m, w, nat, 0, 1, R);
}
/* the Bluestein inner chain at M: raced unbanded on the scrambled pass (il2d_tier.h:2405), so the
 * radix-8 regime's chain whatever the row width, two stages or more (sketch §5) */
static inline int est_blu_chain(const est_machine_t *m, int M, long long w, int *R)
{
    est_pool_t P;
    est__pool(M, &P);
    return est__pick(&P, NULL, 0, M, m, w, 0, 1, 0, R);
}
/* the forms token as the race spells it (il2d_tier.h:2483, 2597): written only when a stage has rival
 * column forms (vfft_il2p_col_forms, il2p.h:481; at c2c radix 32), one name per stage, each the
 * construction table's default, names[0] */
static inline void est__col_forms(const int *R, int nst, char *f, size_t cap)
{
    const char *nm[VFFT_IL2P_COL_MAXFORMS];
    est_sb_t s;
    int i, any = 0;
    est_sb_init(&s, f, cap);
    for (i = 0; i < nst; i++)
        if (vfft_il2p_col_forms(R[i], 0, nm) > 1)
            any = 1;
    if (!any)
        return;
    for (i = 0; i < nst; i++)
    {
        (void)vfft_il2p_col_forms(R[i], 0, nm);
        est_sb_add(&s, "%s%s", i ? "." : "", nm[0]);
    }
    if (s.len >= cap)
        f[0] = 0;
}
/* the column pass's spans L[s] = N / (R[0] ... R[s-1]) (il2d_cols.h:1054-1062) */
static inline void est__spans(int N, const int *R, int nst, int *L)
{
    int s, l = N;
    for (s = 0; s < nst; s++)
    {
        L[s] = l;
        l /= R[s];
    }
}
/* the door's band-width pool below N (2D: il2d_tier.h:4009-4046; 3D: fftnd_il.h:1133-1157): the ladder
 * under the tcut law, the stage spans of 8 rows or more whose band fits the L2 */
static inline int est__wl_pool(const est_machine_t *m, const est_col_t *c, int *out)
{
    int L[8], n = 0, p, s, q, dup;
    est__spans(c->N, c->R, c->nst, L);
    for (p = 0; p < VFFT_IL2D_WL_LADDER_N; p++)
        if (VFFT_IL2D_WL_LADDER[p] < c->N && vfft_policy_il2d_wl_cut(c->N, c->nst, L, VFFT_IL2D_WL_LADDER[p]) >= 0)
            out[n++] = VFFT_IL2D_WL_LADDER[p];
    for (s = 1; s < c->nst; s++)
    {
        const int w = L[s];
        if (w >= c->N || !vfft_policy_il2d_band_ok(c->N, c->nst, L, w) || 16LL * w * c->w > (long long)m->l2)
            continue;
        for (q = 0, dup = 0; q < n; q++)
            if (out[q] == w) dup = 1;
        if (!dup)
            out[n++] = w;
    }
    return n;
}

/* ── the N-arm and the column pass's verdicts ──────────────────────────── */

/* an inner chain the Rader stage serves badly: a radix above EST_ND_RD_MAX_RADIX, an odd radix above
 * EST_ND_RD_ODD_MAX, or a first radix of EST_ND_RD_MAX_RADIX or more */
static inline int est__rd_inner_bad(const int *Rs, int nst)
{
    int i;
    if (Rs[0] >= EST_ND_RD_MAX_RADIX)
        return 1;
    for (i = 0; i < nst; i++)
        if (Rs[i] > EST_ND_RD_MAX_RADIX || ((Rs[i] & 1) && Rs[i] > EST_ND_RD_ODD_MAX))
            return 1;
    return 0;
}
/* the Rader stage at stage 0 (sketch §6): the inner chain of p - 1 the engine's own pick (_ilrd_pick_chain,
 * il_rader_stage.h:184; never led by radix 64, so the request rebuilds it, :392) unless est__rd_inner_bad;
 * then, of the engine's candidates (_ilrd_enum_inners, :705, the planner's race pool), the fewest stages
 * that pass, the smallest radices first (the pick when none passes); the window EST_ND_RD_W within the
 * column space (N / p) w, as the engine takes a requested width (_ilrd_create, :433-438). */
static inline int est__rader(est_col_t *c, int p)
{
    const size_t ncol = (size_t)(c->N / p) * (size_t)c->w;
    _ilrd_inner_t cand[VFFT_ILRD_MAX_INNERS];
    size_t W;
    int nestp = 0, n, i, best = -1;
    if (!_ilrd_pick_chain(p - 1, c->rd, &c->rd_n, &nestp))
        return -1;
    if (est__rd_inner_bad(c->rd, c->rd_n))
    {
        n = _ilrd_enum_inners(p - 1, cand, VFFT_ILRD_MAX_INNERS);
        for (i = 0; i < n; i++)
            if (!est__rd_inner_bad(cand[i].Rs, cand[i].nst) &&
                (best < 0 || cand[i].nst < cand[best].nst ||
                 (cand[i].nst == cand[best].nst &&
                  memcmp(cand[i].Rs, cand[best].Rs, sizeof(int) * (size_t)cand[i].nst) < 0)))
                best = i;
        if (best >= 0)
        {
            memcpy(c->rd, cand[best].Rs, sizeof(int) * (size_t)cand[best].nst);
            c->rd_n = cand[best].nst;
        }
    }
    W = EST_ND_RD_W;
    W &= ~(size_t)1;
    if (W > ncol)
        W = ncol & ~(size_t)1;
    if (W < VFFT_ILRD_W_MIN)
        W = VFFT_ILRD_W_MIN;
    c->rdw = (int)W;
    return 0;
}
/* The column pass at N over rows of w (sketch §4-§6): the chain when the pool holds one -- blu=0 where
 * N is not a power of two (the chain beats Bluestein; the race banks blu only then,
 * il2d_tier.h:4200), with the Rader stage's inner and window where it leads; else the column
 * Bluestein at M = the next power of two >= 2N - 1, at least 16 (il2d_cols.h:928-938), its M chain,
 * and the turned prime pass, tpc=1. M is computed in 64 bits: an M the pool cannot chain (above
 * 2^24, four stages of at most 64) is refused -- one past an int at once, where the door's own int
 * loop (il2d_cols.h:937-938) would never end or overflow. The band width is the caller's. */
/* THE COLUMN UNDER PRIME ROWS (2026-10-26): when the rows are the 1D prime cell, a power-of-two column the
 * pool holds as one kernel stays one stage, whatever the plane's size. Each row is then its own 1D plan,
 * with no row walk fused into the column pass, so the column pass is a pass over the plane of its own and
 * one pass beats two. The store: 16 and 64 columns over prime rows bank the single stage at 9 of 9 rows
 * (16 x 131, 521, 2053, 4099; 64 x 131, 257, 521, 1031, 2053), while over power-of-two rows of 256 and more
 * they split at 10 of 12; the split cost 1.08-2.07 x the planner at 10 of 10 timed shapes. */
static inline void est__col_single_under_prime_rows(est_col_t *c)
{
    est_pool_t P;
    int i;
    if (c->nst < 2 || !est_is_pow2(c->N))
        return;
    est__pool(c->N, &P);
    for (i = 0; i < P.n; i++)
        if (P.len[i] == 1 && P.c[i][0] == c->N)
        {
            c->R[0] = c->N;
            c->nst = 1;
            est__col_forms(c->R, 1, c->forms, sizeof c->forms);
            return;
        }
}
static inline int est_col_plan(const est_machine_t *m, int N, long long w, int nat, est_col_t *c, char *why,
                               size_t whylen)
{
    memset(c, 0, sizeof *c);
    c->N = N;
    c->w = w;
    c->nat = nat;
    c->blu = -1;
    c->tpc = -1;
    if (N < 2 || w < 1)
        return est_refuse(why, whylen, "a column of %d rows over %lld lanes", N, w);
    c->nst = est_col_chain(m, N, w, nat, c->R);
    if (c->nst > 0)
    {
        c->blu = est_is_pow2(N) ? -1 : 0;
        if (VFFT_IL2D_RADER_RADIX(c->R[0]) && est__rader(c, c->R[0]))
            return est_refuse(why, whylen, "N = %d: the Rader stage at %d has no inner chain of %d", N, c->R[0],
                              c->R[0] - 1);
    }
    else
    {
        long long M = 16;
        while (M < 2LL * N - 1)
            M <<= 1;
        if (M > INT_MAX || (c->nst = est_blu_chain(m, (int)M, w, c->R)) < 1)
            return est_refuse(why, whylen, "N = %d has no column chain and its Bluestein length %lld none either", N, M);
        c->blu = (int)M;
        c->tpc = 1;
    }
    est__col_forms(c->R, c->nst, c->forms, sizeof c->forms);
    return 0;
}
/* the 2D band (sketch §7): none beside the skew or the turn, on a one-stage or Bluestein chain, or
 * while the plane fits L2 / plane_div (EST_ND_2D_PLANE_DIV, or EST_ND_3D_PLANE_DIV for the 3D plan's plane
 * child); else the widest pool width whose band fits L2 / 4, 0 when none does */
static inline int est_wl_2d(const est_machine_t *m, const est_col_t *c, int N2, int csk, int turn, int plane_div)
{
    int pool[16], n, i, best = 0;
    if (csk || turn || c->blu > 0 || c->nst < 2 || 16LL * c->N * N2 <= (long long)m->l2 / plane_div)
        return 0;
    n = est__wl_pool(m, c, pool);
    for (i = 0; i < n; i++)
        if (16LL * pool[i] * N2 <= (long long)m->l2 / EST_ND_2D_BAND_DIV && pool[i] > best)
            best = pool[i];
    return best;
}
/* the 3D axis-0 band (sketch §9): none on a one-stage or Bluestein axis 0; else the widest pool width
 * whose band fits the L2, else the narrowest pool width */
static inline int est_wl_3d(const est_machine_t *m, const est_col_t *c)
{
    int pool[16], n, i, best = 0, least = 0;
    if (c->blu > 0 || c->nst < 2)
        return 0;
    n = est__wl_pool(m, c, pool);
    for (i = 0; i < n; i++)
    {
        if (16LL * pool[i] * c->w <= (long long)m->l2 && pool[i] > best)
            best = pool[i];
        if (!least || pool[i] < least)
            least = pool[i];
    }
    return best ? best : least;
}
/* the door's strip widths at this axis 0, read off its own pool (_ilnd_sw_pool, fftnd_il.h:826-836):
 * the powers of two from 8 to 1024 whose strip scratch, 16 N1 w bytes, fits the L2 (the host's) and
 * no wider than the plane; none on a one-stage or Bluestein axis 0, which has nothing to permute */
static inline int est__sw_pool(const est_col_t *c, int N2, int N3, int *out, int max)
{
    vfft_ilnd_t d;
    memset(&d, 0, sizeof d);
    d.N[0] = c->N;
    d.plane = (size_t)N2 * (size_t)N3;
    d.ax0.blu = c->blu > 0 ? c->blu : 0;
    d.ax0.nst = c->nst;
    return _ilnd_sw_pool(&d, out, max);
}
/* the 3D natural form (sketch §9): the strips, nf = 2, when the volume passes the L3 (16 N1 N2 N3
 * bytes; no L3 reported counts as passed, as the 1D estimate reads it) and axis 0's column is short,
 * 16 N1 <= L1d / EST_ND_NF_L1_DIV -- at the widest width of the door's pool whose strip scratch fits
 * L2 / EST_ND_NSW_L2_DIV; else the cycle walk, nf = 1, also where no width of the pool fits. Returns nf,
 * the width into *nsw (0 with the cycle walk). */
static inline int est_nf_3d(const est_machine_t *m, const est_col_t *c, int N2, int N3, int *nsw)
{
    const long long plane = (long long)N2 * N3;
    int sp[8], n, i;
    *nsw = 0;
    if (m->l3 > 0 && plane <= (long long)m->l3 / (16LL * c->N))   /* 16 N1 N2 N3 <= L3, without the product */
        return 1;
    if (16LL * c->N > (long long)m->l1d / EST_ND_NF_L1_DIV)
        return 1;
    n = est__sw_pool(c, N2, N3, sp, 8);
    for (i = 0; i < n; i++)
        if (16LL * c->N * sp[i] <= (long long)m->l2 / EST_ND_NSW_L2_DIV && sp[i] > *nsw)
            *nsw = sp[i];
    return *nsw > 0 ? 2 : 1;
}

/* ── the 2D row route ───────────────────────────────────────────────────── */

/* the two-pass rows' kernels have row-loop twins: the row plan a pair of radices 4, 8 and 16 in
 * their plain or tangent bodies, the row-loop twin table of il2d_tier.h:889-912 (the pre-flight asks
 * the table itself) */
static inline int est__rowb2_form(const est_recipe_t *rp)
{
    int s;
    if (rp->k1_il_route != VFFT_K1_IL_2P_PURE)
        return 0;
    if ((rp->il_R1 != 4 && rp->il_R1 != 8 && rp->il_R1 != 16) || (rp->il_R2 != 4 && rp->il_R2 != 8 && rp->il_R2 != 16))
        return 0;
    for (s = 0; s < 2; s++)
    {
        const int f = (rp->il_kv >> (4 * s)) & 15, b = (rp->il_bkv >> (4 * s)) & 15;
        if ((f != 0 && f != 3) || (b != 0 && b != 3))
            return 0;
    }
    return 1;
}
/* the 1D child's recipe: est_plan_1d at N, natural, one thread, in place or out of place */
static inline int est__child_1d(const est_machine_t *m, int N, int placement, est_recipe_t *rc, const char *what,
                                char *why, size_t whylen)
{
    est_request_t rq;
    char w2[400];
    memset(&rq, 0, sizeof rq);
    rq.N = N;
    rq.transform = VFFT_C2C;
    rq.order = VFFT_ORDER_NATURAL;
    rq.placement = placement;
    rq.nthreads = 1;
    if (est_plan_1d(m, &rq, rc, w2, sizeof w2))
        return est_refuse(why, whylen, "%s at N = %d: %s", what, N, w2);
    rc->il_rd_n = 0;   /* a child's Rader stage keeps the engine's pick and width: the 1D window rule
                        * (est_rader_window) was measured on 1D requests only */
    rc->il_rdw = 0;
    return 0;
}
static inline int est__machine_ok(const est_machine_t *m, char *why, size_t whylen)
{
    if (!m)
        return est_refuse(why, whylen, "no machine");
    if (m->lanes != VFFT_IL_VW)
        return est_refuse(why, whylen, "a machine of %d lanes: the formulas follow this build's %d", m->lanes, VFFT_IL_VW);
    if (m->l1d < 1024 || m->l2 < 1024)
        return est_refuse(why, whylen, "an L1d of %ld bytes and an L2 of %ld", m->l1d, m->l2);
    return 0;
}

/* the row route of a plane without the turn (sketch §8): p's N2, column and row plan set */
static inline int est__ro_2d(const est_plan2d_t *p)
{
    const int N2 = p->N2, n1ccs = vfft_il_n1ccs_fn(N2, 0) && vfft_il_n1ccs_fn(N2, 1);
    if (p->col.blu > 0)
        return 0;
    if (N2 <= EST_ND_ROWB_MAX)
        return n1ccs ? 2 : 0;
    if (N2 <= EST_ND_ROWB2_MAX && !est_is_pow2(N2) && (N2 > 32 || est__spills_single(N2)))
        return 0;   /* a row of 25, 27 or a non-power of two above 32 (up to 64): the per-row child
                     * (2026-10-26). The batched rows run each row as one kernel of N2, and those kernels
                     * spill (est__spills_single): the planner banks the per-row child at the rows of 25, 27,
                     * 35, 45, 49, 60, 63 (2D 25x25, 27x27, 45x60, 63x63; the planes of 9x15x27, 25^3, 27^3,
                     * 35^3, 45^3, 49^3, 63^3, 64x25x25, 64x27x27) and the batched rows at the rows of 21 and
                     * 31 (the planes of 21^3, 31^3; 2D 21x21 is the one exception, 14 of 15 agree). The
                     * estimate's batched rows cost 25x25 and 27x27 1.22-1.69 x the planner; the per-row child
                     * at 21^3 and 31^3 cost 1.34 and 1.07 */
    if (N2 <= EST_ND_ROWB2_MAX)
        return est__rowb2_form(&p->rp) ? 3 : n1ccs ? 2 : 0;
    return 0;
}

/* est_plan_2d: the 2D c2c plan at N1 x N2, natural, one thread (sketch §3-§8, §10). Returns 0 with
 * *out filled, or -1 with the reason. Pure. */
static inline int est_plan_2d(const est_machine_t *m, int N1, int N2, est_plan2d_t *out, char *why, size_t whylen)
{
    const char *rowno;
    int mono, n1ccs;
    char dummy[1];
    if (!why || !whylen) { why = dummy; whylen = sizeof dummy; }
    why[0] = 0;
    if (!out)
        return est_refuse(why, whylen, "no plan");
    memset(out, 0, sizeof *out);
    if (est__machine_ok(m, why, whylen))
        return -1;
    if (N1 < 2 || N2 < 2)
        return est_refuse(why, whylen, "%dx%d: estimate plans every axis of at least 2 points", N1, N2);
    if ((rowno = vfft_policy_c2c_row_refused(N2)) != NULL)
        return est_refuse(why, whylen, "%dx%d: its rows at the last axis N = %d have no 1D engine: %s", N1, N2, N2, rowno);
    out->N1 = N1;
    out->N2 = N2;
    if (est_col_plan(m, N1, N2, 1, &out->col, why, whylen))
        return -1;
    if (vfft_policy_prime_cell(N2))   /* the rows are the 1D prime cell (est__child_1d below, in place) */
        est__col_single_under_prime_rows(&out->col);
    mono = vfft_k1_mono_ilc_fn(N1, 0) && vfft_k1_mono_ilc_fn(N1, 1);
    n1ccs = vfft_il_n1ccs_fn(N2, 0) && vfft_il_n1ccs_fn(N2, 1);
    out->has_turn = out->col.blu <= 0 && n1ccs;
    out->has_csk = out->col.blu <= 0 && mono;
    if (est__child_1d(m, N2, VFFT_PLACEMENT_IN, &out->rp, "the row plan", why, whylen))
        return -1;
    if (out->has_turn && est__child_1d(m, N1, VFFT_PLACEMENT_IN, &out->turnp, "the turn plan", why, whylen))
        return -1;
    if (out->has_csk && est__child_1d(m, N2, VFFT_PLACEMENT_OUT, &out->cskp, "the skewed pass's row plan", why, whylen))
        return -1;
    if (out->col.tpc == 1 && est__child_1d(m, N1, VFFT_PLACEMENT_IN, &out->tpcp, "the turned prime pass", why, whylen))
        return -1;
    /* the row route (sketch §8); a Bluestein cell has no axis race and serves the per-row child */
    out->csk = out->has_csk && (N1 <= EST_ND_CSK_MAX || ((N1 & 1) && N1 <= EST_ND_CSK_ODD_MAX && N2 >= EST_ND_CSK_ODD_N2 &&
                                                            !est__spills_single(N1)));
    out->turn = out->has_turn && !out->csk && N1 >= EST_ND_TURN_N1 && N2 <= EST_ND_TURN_N2;
    out->ro = out->turn ? 0 : est__ro_2d(out);
    out->rbk = EST_ND_RBK;
    out->sw = 0;
    out->col.wl = est_wl_2d(m, &out->col, N2, out->csk, out->turn, EST_ND_2D_PLANE_DIV);
    return 0;
}
/* the strips' width at T (EST_ND_MT_*): a rung of the door's ladder (_il2d_sw_ladder: w at most a worker's
 * column block N2 / T, the last sub-strip narrower where w does not divide it, the strip N1 w 16 B within
 * the L2) -- while an 8-column strip fits L1d / EST_ND_MT_SW_L1_DIV and
 * the transform's in and out fit the L2, the widest within that fraction of the L1d; else the widest whose
 * strip's in and out, 32 N1 w B, fit the L2; the narrowest when none does; 0 (each worker's whole range) on
 * an empty ladder */
static inline int est__msw_2d(const est_machine_t *m, int N1, int N2, int T)
{
    int lad[6], n, i, w = 0;
    const int l1 = 16LL * N1 * EST_ND_MT_SW_MIN <= m->l1d / EST_ND_MT_SW_L1_DIV && 32LL * N1 * N2 <= m->l2;
    if ((n = _il2d_sw_ladder(N1, N2, T, lad, 6)) < 1)
        return 0;
    for (i = 0; i < n; i++)
        if (l1 ? 16LL * N1 * lad[i] <= m->l1d / EST_ND_MT_SW_L1_DIV : 32LL * N1 * lad[i] <= m->l2)
            w = lad[i];
    return w ? w : lad[0];
}
/* the unbanded walk's chain (est_blu_chain's pick); a Rader stage, a Bluestein column or a chain the pool
 * lacks keeps the plan's */
static inline void est__unbanded_chain_2d(const est_machine_t *m, est_plan2d_t *p)
{
    est_pool_t P;
    int R[8], k;
    if (p->col.blu > 0 || p->col.rd_n)
        return;
    est__pool(p->N1, &P);
    if ((k = est__pick(&P, NULL, 0, p->N1, m, p->N2, 1, /*wide=*/1, /*single=*/0, R)) > 0)
    {
        memcpy(p->col.R, R, sizeof(int) * (size_t)k);
        p->col.nst = k;
        est__col_forms(p->col.R, p->col.nst, p->col.forms, sizeof p->col.forms);
    }
}
/* the tile's chain at T: 1 when the door tiles p's chain (a natural chain of two stages or more, no
 * Bluestein; il2d_tier.h:1545) and the chain has no Rader stage (the strips run it faster, EST_ND_MT_*),
 * its first stage then the pool's smallest first radix of at least T where the chain leads with less, and
 * where that radix is not a multiple of T the pool's first radix of at most EST_ND_MT_TILE_R0_MAX that
 * shares best among the workers, R0 / (T ceil(R0 / T)), the larger on a tie (the tile runs min(R0, T)
 * workers on R0 sub-problems, il2d_tier.h:1555); the rest est__pick's one-thread regime; 0 leaves p as it was */
static inline int est__tile_chain_2d(const est_machine_t *m, int T, est_plan2d_t *p)
{
    est_pool_t P;
    int R[8], k, r0 = 0, i;
    double bb = -1;
    if (!p->col.nat || p->col.blu > 0 || p->col.nst < 2 || p->col.rd_n)
        return 0;
    est__pool(p->N1, &P);
    if (p->col.R[0] < T)
        for (i = 0; i < P.n; i++)
            if (P.c[i][0] >= T && P.len[i] >= 2 && (!r0 || P.c[i][0] < r0))
                r0 = P.c[i][0];
    if (!r0)
        r0 = p->col.R[0];
    if (r0 % T)
        for (i = 0; i < P.n; i++)
        {
            const int c = P.c[i][0];
            const double b = (double)c / (T * ((c + T - 1) / T));
            if (P.len[i] >= 2 && c <= EST_ND_MT_TILE_R0_MAX && (b > bb + 1e-9 || (b > bb - 1e-9 && c > r0)))
            {
                bb = b;
                r0 = c;
            }
        }
    if ((k = est__pick(&P, &r0, 1, p->N1 / r0, m, p->N2, 1, /*wide=*/0, /*single=*/0, R)) > 0)
    {
        memcpy(p->col.R, R, sizeof(int) * (size_t)k);
        p->col.nst = k;
        est__col_forms(p->col.R, p->col.nst, p->col.forms, sizeof p->col.forms);
    }
    return 1;
}
/* est_mt_2d: the one-thread plan p (est_plan_2d) threaded at T (EST_ND_MT_*): the threaded verdict and,
 * where it threads, the turn on few columns, the blocks where the door would run the skewed pass serial,
 * the tile on a tall plane, or the strips with the unbanded walk's chain. A column through the turned
 * prime pass stays serial: the door runs that pass on one thread (il2d_tier.h:1503). T < 2 leaves p
 * one-thread. */
static inline void est_mt_2d(const est_machine_t *m, int T, est_plan2d_t *p)
{
    const int nat = p->col.nat && p->col.blu <= 0;   /* the natural leaf, once the chain has two stages */
    p->T = T > 1 ? T : 1;
    p->cmt = p->mtarm = p->msw = 0;
    p->nls = -1;
    if (T < 2)
        return;
    if (16LL * p->N1 * p->N2 > m->l1d / EST_ND_MT_SERIAL_L1_DIV && p->N1 != 2 && p->col.tpc != 1)
    {
        const int tall = p->N1 >= (long long)EST_ND_MT_TILE_N1_PER_N2 * p->N2;
        int tile = 0;
        p->cmt = 1;
        if (p->N2 <= EST_ND_MT_TURN_PER_T * T && p->has_turn && !p->turn)
        {
            p->turn = 1;
            p->csk = 0;
            p->ro = 0;
            p->col.wl = est_wl_2d(m, &p->col, p->N2, p->csk, p->turn, EST_ND_2D_PLANE_DIV);
        }
        if (p->turn && p->N1 % EST_ND_MT_TURN_N1_DIV && tall)
        {   /* the door runs this turn serial: the tile where the chain takes one, the strips on a Rader chain */
            const est_col_t keep = p->col;
            est__unbanded_chain_2d(m, p);
            if ((tile = est__tile_chain_2d(m, T, p)) != 0 || p->col.rd_n)
            {
                p->turn = 0;
                p->ro = est__ro_2d(p);
            }
            else
                p->col = keep;
        }
        if (p->csk && p->N2 % EST_ND_MT_CSK_COLS)
        {   /* the door runs this skew serial: the chain walk in blocks, one stage of N1 where the pool has it */
            est_pool_t P;
            int R1[1] = { p->N1 };
            p->csk = 0;
            est__pool(p->N1, &P);
            if (p->col.blu <= 0 && !p->col.rd_n && est__in_pool(&P, R1, 1))
            {
                p->col.R[0] = p->N1;
                p->col.nst = 1;
                est__col_forms(p->col.R, p->col.nst, p->col.forms, sizeof p->col.forms);
            }
            p->col.wl = est_wl_2d(m, &p->col, p->N2, p->csk, p->turn, EST_ND_2D_PLANE_DIV);
        }
        else if (!p->csk && !p->turn)
        {
            if (!tile)
            {
                est__unbanded_chain_2d(m, p);
                tile = tall && est__tile_chain_2d(m, T, p);
            }
            p->col.wl = 0;
            p->mtarm = tile ? 2 : 1;
            p->msw = tile ? 0 : est__msw_2d(m, p->N1, p->N2, T);
        }
    }
    if (nat && p->col.nst > 1)
        p->nls = (p->cmt && p->mtarm == 1) ? 0 : 1;   /* the strips strided; else the door's staged default */
}
/* the 3D structure: the flat tier (s=2) on a plane of at most EST_ND_FLAT_HI_PCT percent of the L2
 * (EST_ND_FLAT_HI_L3_PCT when the cube passes the L3) whose column pass is a plain chain (no Bluestein, no
 * Rader stage) and that is either even -- N2 and N3 even, not both powers of two, at least
 * EST_ND_FLAT_LO_PCT percent of the L2 -- or has an odd side, a plane plan without the skewed column
 * pass and rows that are not one odd solo kernel; else the 2D plan per plane (s=1) */
static inline int est__flat_plane(const est_machine_t *m, int N1, int N2, int N3, const est_plan2d_t *pl)
{
    const est_col_t *c = &pl->col;
    const long long b = 16LL * N2 * N3;
    const int hi = b * N1 > (long long)m->l3 ? EST_ND_FLAT_HI_L3_PCT : EST_ND_FLAT_HI_PCT;
    if (c->blu > 0 || c->rd_n || 100 * b > hi * (long long)m->l2)
        return 0;
    if ((N2 & 1) || (N3 & 1))   /* a plane with an odd side: any size, unless the plane plan skews its column
                                 * pass (the flat tier has no skewed pass) or its rows are one odd solo kernel */
        return !pl->csk && !((N3 & 1) && est_in(est_n1_list, EST_NELEM(est_n1_list), N3) && !est_radix_spills(N3));
    return !(est_is_pow2(N2) && est_is_pow2(N3)) && 100 * b >= EST_ND_FLAT_LO_PCT * (long long)m->l2;
}
/* est_plan_3d: the 3D c2c plan at N1 x N2 x N3, natural, one thread (sketch §9): the structure
 * (est__flat_plane: the 2D plan per plane, s=1, or the flat tier, s=2, with the 2D plan's column chain
 * and row plan), axis 0 on the scrambled pass over the planes, the natural form (the strips or the cycle
 * walk; axis 0's band only with the cycle walk). Pure. */
static inline int est_plan_3d(const est_machine_t *m, int N1, int N2, int N3, est_plan3d_t *out, char *why,
                              size_t whylen)
{
    const char *rowno;
    char dummy[1];
    if (!why || !whylen) { why = dummy; whylen = sizeof dummy; }
    why[0] = 0;
    if (!out)
        return est_refuse(why, whylen, "no plan");
    memset(out, 0, sizeof *out);
    if (est__machine_ok(m, why, whylen))
        return -1;
    if (N1 < 2 || N2 < 2 || N3 < 2)
        return est_refuse(why, whylen, "%dx%dx%d: estimate plans every axis of at least 2 points", N1, N2, N3);
    if ((rowno = vfft_policy_c2c_row_refused(N3)) != NULL)
        return est_refuse(why, whylen, "%dx%dx%d: its rows at the last axis N = %d have no 1D engine: %s", N1, N2, N3,
                          N3, rowno);
    out->N1 = N1;
    out->N2 = N2;
    out->N3 = N3;
    if (est_plan_2d(m, N2, N3, &out->plane, why, whylen))
        return -1;
    out->plane.col.wl = est_wl_2d(m, &out->plane.col, N3, out->plane.csk, out->plane.turn, EST_ND_3D_PLANE_DIV);
    if (est_col_plan(m, N1, (long long)N2 * N3, 0, &out->col, why, whylen))
        return -1;
    if (out->col.tpc == 1 && est__child_1d(m, N1, VFFT_PLACEMENT_IN, &out->tpcp, "the turned prime pass", why, whylen))
        return -1;
    out->s = est__flat_plane(m, N1, N2, N3, &out->plane) ? 2 : 1;
    if (out->s == 2)
    {
        out->col1 = out->plane.col;
        out->col1.wl = 0;
    }
    out->nf = est_nf_3d(m, &out->col, N2, N3, &out->nsw);
    out->col.wl = out->nf == 2 ? 0 : est_wl_3d(m, &out->col);   /* the strips serve axis 0 unbanded (fftnd_il.h:1619) */
    return 0;
}
/* est_mt_3d: the one-thread plan p (est_plan_3d) threaded at T (EST_ND_MT3_*): serial while the cube times the
 * workers that can take part fits EST_ND_MT3_SERIAL_KB x T; else the plane arm on the full team with the
 * one-thread structure, the strips wherever the door's strip pool at axis 0 is not empty (at the widest width
 * whose scratch fits L2 / EST_ND_MT3_NSW_L2_DIV, else the narrowest), the cycle walk where it is. The recipe
 * is kept, but for the axis-0 chain of a cube whose sides are all powers of two: on the strips with more
 * columns than workers est__pick's radix-8 regime, on the cycle walk the pool's chain ending in
 * EST_ND_MT3_CYCLE_LAST with the stages before it the most balanced, ascending; the band beside the one-thread
 * cycle walk taken again for either. T < 2 leaves p one-thread. */
static inline void est_mt_3d(const est_machine_t *m, int T, est_plan3d_t *p)
{
    const long long cube = 16LL * p->N1 * p->N2 * p->N3, cols = (long long)p->N2 * p->N3;
    const int pow2 = est_is_pow2(p->N1) && est_is_pow2(p->N2) && est_is_pow2(p->N3);
    long long w = T;
    int sp[8], n, i;
    p->T = T > 1 ? T : 1;
    p->cmt = 0;
    p->cmts = p->s;
    p->cmtf = p->nf;
    p->mtnsw = 0;
    if (T < 2)
        return;
    if (p->N1 < w)
        w = p->N1;
    if (cols < w)
        w = cols;
    if (cube * w <= (long long)EST_ND_MT3_SERIAL_KB * 1024 * T)
        return;
    p->cmt = 2;
    n = est__sw_pool(&p->col, p->N2, p->N3, sp, 8);
    p->cmtf = n > 0 ? 2 : 1;
    for (i = 0; i < n; i++)
        if (16LL * p->N1 * sp[i] <= (long long)m->l2 / EST_ND_MT3_NSW_L2_DIV && sp[i] > p->mtnsw)
            p->mtnsw = sp[i];
    if (n > 0 && !p->mtnsw)
        p->mtnsw = sp[0];
    if (p->cmtf == 2 && pow2 && cols > T)
    {   /* the strips' chain (EST_ND_MT3_*): the radix-8 regime, as the 2D strips take it */
        est_pool_t P;
        int R[8], k;
        est__pool(p->N1, &P);
        if ((k = est__pick(&P, NULL, 0, p->N1, m, p->col.w, 0, /*wide=*/1, /*single=*/0, R)) >= 2)
        {
            memcpy(p->col.R, R, sizeof(int) * (size_t)k);
            p->col.nst = k;
            est__col_forms(p->col.R, p->col.nst, p->col.forms, sizeof p->col.forms);
            p->col.wl = p->nf == 2 ? 0 : est_wl_3d(m, &p->col);
        }
    }
    else if (p->cmtf == 1 && pow2 && p->col.nst >= 2)
    {   /* the cycle walk's chain (EST_ND_MT3_*): the pool's chain ending in EST_ND_MT3_CYCLE_LAST, the stages
         * before it the least sum of (log2 R)^2, then the fewest out of ascending order, then the pool's order */
        est_pool_t P;
        int best = -1, bi = 0;
        double bs = 0.0;
        est__pool(p->N1, &P);
        for (i = 0; i < P.n; i++)
        {
            const int k = P.len[i];
            double s;
            int inv;
            if (k < 2 || P.c[i][k - 1] != EST_ND_MT3_CYCLE_LAST)
                continue;
            s = est__sumsq(P.c[i], k - 1);
            inv = est__inversions(P.c[i], k - 1, 1);
            if (best < 0 || s < bs - 1e-9 || (s < bs + 1e-9 && inv < bi))
            {
                best = i;
                bs = s;
                bi = inv;
            }
        }
        if (best >= 0)
        {
            memcpy(p->col.R, P.c[best], sizeof(int) * (size_t)P.len[best]);
            p->col.nst = P.len[best];
            est__col_forms(p->col.R, p->col.nst, p->col.forms, sizeof p->col.forms);
            p->col.wl = p->nf == 2 ? 0 : est_wl_3d(m, &p->col);
        }
    }
}

/* ── the row writer ─────────────────────────────────────────────────────── */

/* the column row's key: rank 2 or 3, natural, axis 0, c2c, one thread (vw2__ilcol_key; the doors'
 * keys at fft2d_create_il.h:404 and fftnd_il.h:1474-1478); the writers key a threaded plan's row at its T */
static inline void est__colkey(vw2_ilcol_key_t *ck, int rank, int n0, int n1, int n2)
{
    memset(ck, 0, sizeof *ck);
    ck->rank = rank;
    ck->n0 = n0;
    ck->n1 = n1;
    ck->n2 = n2;
    ck->ord = VW2_ORD_NAT;
    ck->nthreads = 1;
}
/* the column pass's tokens: chain wl tf ro blu with the chain bank (reader.h:189), then forms or
 * bluforms, rd rdw, tpc as the door's own banks write them (il2d_tier.h:3253, 2425, 3082-3083, 3386);
 * ro < 0 = not written (rank 3); wlw = 0: wl and tf not written (the 3D strips: the door banks the
 * band only where the cycle walk won, fftnd_il.h:1733) */
static inline int est__put_col(vw2_store_t *st, const vw2_ilcol_key_t *ck, const est_col_t *c, int ro, int wlw)
{
    char b[64];
    est_sb_t s;
    if (vw2_ilcol_chain_bank(st, ck, c->R, c->nst, wlw ? c->wl : -1, wlw ? c->wl > 0 : -1, ro, -1, -1, c->blu, 0.0) !=
        VW2_OK)
        return -1;
    if (c->forms[0] && !vw2_ilcol_forms_bank_base(st, ck, c->blu > 0 ? "bluforms" : "forms", c->forms))
        return -1;
    if (c->rd_n > 0)
    {
        est_sb_init(&s, b, sizeof b);
        est_sb_chain(&s, c->rd, c->rd_n);
        if (s.len >= sizeof b || !vw2_ilcol_forms_bank_base(st, ck, "rd", b))
            return -1;
        snprintf(b, sizeof b, "%d", c->rdw);
        if (!vw2_ilcol_forms_bank_base(st, ck, "rdw", b))
            return -1;
    }
    if (c->tpc >= 0 && vw2_ilcol_tok_seti(st, ck, "tpc", c->tpc) != VW2_OK)
        return -1;
    return 0;
}
/* a 1D child onto the parent row at key under pre: its rows written by est_put_rows into a store of
 * its own, moved with the door's vfft_child_row_update (wisdom2_child.h:179) */
static inline int est__put_child(vw2_store_t *st, const vw2_key_t *key, const char *pre, const est_recipe_t *rc, int N,
                                 int placement, char *why, size_t whylen)
{
    struct vfft_wisdom_s *T = vfft_child_store_new();
    est_request_t rq;
    int ok;
    if (!T)
        return est_refuse(why, whylen, "out of memory");
    memset(&rq, 0, sizeof rq);
    rq.N = N;
    rq.transform = VFFT_C2C;
    rq.order = VFFT_ORDER_NATURAL;
    rq.placement = placement;
    rq.nthreads = 1;
    if (est_put_rows(&T->vw2, rc, &rq, 1, why, whylen))
    {
        vfft_child_store_free(T);
        return -1;
    }
    ok = vfft_child_row_update(st, key, pre, T);
    vfft_child_store_free(T);
    if (!ok)
        return est_refuse(why, whylen, "the child %s* at N = %d could not be put on its parent's row", pre, N);
    return 0;
}
/* est_put_rows_2d: the 2D row into st -- the column tokens, sw rbk turn csk (il2d_tier.h:4201-4204), at
 * T > 1 under the key nthreads=T with the threaded verdict cmt mtarm msw nls (il2d_tier.h:4511-4517), then
 * the children rp_ turn_ csk_ tpc_ (il2d_tier.h:4225-4228), one-thread rows as the door's. Returns 0, or -1
 * with the reason. */
static inline int est_put_rows_2d(vw2_store_t *st, const est_plan2d_t *p, char *why, size_t whylen)
{
    vw2_ilcol_key_t ck;
    vw2_key_t key;
    est__colkey(&ck, 2, p->N1, p->N2, 0);
    ck.nthreads = p->T > 1 ? p->T : 1;
    if (est__put_col(st, &ck, &p->col, p->ro, 1) || vw2_ilcol_tok_seti(st, &ck, "sw", p->sw) != VW2_OK ||
        vw2_ilcol_tok_seti(st, &ck, "rbk", p->rbk) != VW2_OK || vw2_ilcol_tok_seti(st, &ck, "turn", p->turn) != VW2_OK ||
        vw2_ilcol_tok_seti(st, &ck, "csk", p->csk) != VW2_OK)
        return est_refuse(why, whylen, "the %dx%d row was refused by the store's writers", p->N1, p->N2);
    if (p->T > 1 &&
        (vw2_ilcol_tok_seti(st, &ck, "cmt", p->cmt) != VW2_OK || vw2_ilcol_tok_seti(st, &ck, "mtarm", p->mtarm) != VW2_OK ||
         vw2_ilcol_tok_seti(st, &ck, "msw", p->msw) != VW2_OK ||
         (p->nls >= 0 && vw2_ilcol_tok_seti(st, &ck, "nls", p->nls) != VW2_OK)))
        return est_refuse(why, whylen, "the %dx%d row's threaded verdict was refused by the store's writers", p->N1, p->N2);
    vw2__ilcol_key(&ck, &key);
    if (est__put_child(st, &key, "rp_", &p->rp, p->N2, VFFT_PLACEMENT_IN, why, whylen))
        return -1;
    if (p->has_turn && est__put_child(st, &key, "turn_", &p->turnp, p->N1, VFFT_PLACEMENT_IN, why, whylen))
        return -1;
    if (p->has_csk && est__put_child(st, &key, "csk_", &p->cskp, p->N2, VFFT_PLACEMENT_OUT, why, whylen))
        return -1;
    if (p->col.tpc == 1 && est__put_child(st, &key, "tpc_", &p->tpcp, p->N1, VFFT_PLACEMENT_IN, why, whylen))
        return -1;
    return 0;
}
/* est_put_rows_3d: the 3D row into st -- the axis-0 tokens (with the strips no wl, no tf), s, nf and,
 * with the strips, nsw (fftnd_il.h:1731-1738), at T > 1 under the key nthreads=T with the threaded verdict
 * cmt cmts cmtf and, beside the threaded strips, their width over nsw (fftnd_il.h:1826-1839; no cmtp: the
 * full team), then s=1: the plane child as a whole 2D row with its own
 * children under plane_; s=2: axis 1's tokens on the same row (chain1 forms1 blu1, no band) and the row
 * plan under rp_ (_ilnd_build_flat, fftnd_il.h:1192-1216); then tpc_ (fftnd_il.h:1933-1935). The threaded
 * structure is the one-thread one, so the row carries that structure's children alone. Returns 0, or -1
 * with the reason. */
static inline int est_put_rows_3d(vw2_store_t *st, const est_plan3d_t *p, char *why, size_t whylen)
{
    vw2_ilcol_key_t ck;
    vw2_key_t key;
    struct vfft_wisdom_s *T;
    int ok;
    est__colkey(&ck, 3, p->N1, p->N2, p->N3);
    ck.nthreads = p->T > 1 ? p->T : 1;
    if (est__put_col(st, &ck, &p->col, -1, p->nf != 2) || vw2_ilcol_tok_seti(st, &ck, "s", p->s) != VW2_OK ||
        vw2_ilcol_tok_seti(st, &ck, "nf", p->nf) != VW2_OK ||
        (p->nf == 2 && vw2_ilcol_tok_seti(st, &ck, "nsw", p->nsw) != VW2_OK))
        return est_refuse(why, whylen, "the %dx%dx%d row was refused by the store's writers", p->N1, p->N2, p->N3);
    if (p->T > 1 &&
        (vw2_ilcol_tok_seti(st, &ck, "cmt", p->cmt) != VW2_OK || vw2_ilcol_tok_seti(st, &ck, "cmts", p->cmts) != VW2_OK ||
         vw2_ilcol_tok_seti(st, &ck, "cmtf", p->cmtf) != VW2_OK ||
         (p->cmt == 2 && p->cmtf == 2 && vw2_ilcol_tok_seti(st, &ck, "nsw", p->mtnsw) != VW2_OK)))
        return est_refuse(why, whylen, "the %dx%dx%d row's threaded verdict was refused by the store's writers", p->N1,
                          p->N2, p->N3);
    vw2__ilcol_key(&ck, &key);
    if (p->s == 2)
    {
        vw2_ilcol_key_t ck1 = ck;
        ck1.axis = 1;
        if (est__put_col(st, &ck1, &p->col1, -1, 0))
            return est_refuse(why, whylen, "the %dx%dx%d row's axis 1 was refused by the store's writers", p->N1, p->N2,
                              p->N3);
        if (est__put_child(st, &key, "rp_", &p->plane.rp, p->N3, VFFT_PLACEMENT_IN, why, whylen))
            return -1;
        if (p->col.tpc == 1 && est__put_child(st, &key, "tpc_", &p->tpcp, p->N1, VFFT_PLACEMENT_IN, why, whylen))
            return -1;
        return 0;
    }
    if (!(T = vfft_child_store_new()))
        return est_refuse(why, whylen, "out of memory");
    if (est_put_rows_2d(&T->vw2, &p->plane, why, whylen))
    {
        vfft_child_store_free(T);
        return -1;
    }
    ok = vfft_child_row_update(st, &key, "plane_", T);
    vfft_child_store_free(T);
    if (!ok)
        return est_refuse(why, whylen, "the plane child %dx%d could not be put on the %dx%dx%d row", p->N2, p->N3,
                          p->N1, p->N2, p->N3);
    if (p->col.tpc == 1 && est__put_child(st, &key, "tpc_", &p->tpcp, p->N1, VFFT_PLACEMENT_IN, why, whylen))
        return -1;
    return 0;
}

#endif /* VFFT_IL_ESTIMATE_ND_IL_H */
