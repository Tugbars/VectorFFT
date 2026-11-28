/* SPDX-License-Identifier: MIT
 * Copyright (c) 2026 Tuğbars Heptaşkın */
/* il2d_col.h — THE COLUMN-AXIS PASS DESCRIPTOR.
 *
 * Everything the interleaved column pass over an N-row plane of rn complex
 * per row needs: the chain, its kernels and tables, the banded walk, the
 * natural leaf redirection, the column-axis Bluestein and the column-MT
 * verdict. The 2D plan owns one instance (il2d_col: axis N1 over rows of
 * N2, or hp1 for the real tier); a rank-N IL plan owns one per column axis
 * (fftnd_il.h: axis 0 over N1 rows of N2*N3, axis 1 per plane over N2 rows
 * of N3). The ROW pass is not here: rows belong to the plan that owns them.
 * The column-pass functions (il2d_cols.h, il2d_tier.h) take a pointer to
 * this; nothing in it depends on the plan struct. */
#ifndef VFFT_TRANSFORMS_FFT2D_IL2D_COL_H
#define VFFT_TRANSFORMS_FFT2D_IL2D_COL_H

#include <stddef.h>
#include "il2p.h"   /* vfft_il2p_fn */

struct vfft_plan_s;   /* the turned prime pass's 1D plan */

typedef struct {
    int N;                    /* rows: the axis length */
    size_t rn;                /* row length in complex: N2 (c2c), hp1 = N2/2+1 (real) */
    /* the column chain: stage s has radix R[s] over sub-length L[s] (D =
     * L/R); stages 0..nst-2 are t2c with driver-built d-major record tables
     * (tf fwd / tb conjugated bwd), the leaf n1c; kernels f fwd / b bwd */
    int nst;
    int R[8], L[8];
    vfft_il2p_fn f[8], b[8];
    vfft_real *tf[8], *tb[8];
    /* a RADER STAGE at stage 0 (il_rader_stage.h; docs/roadmap/rader_leaf_design.md
     * section 13): R[0] a prime above the kernel pool, f[0] = b[0] = NULL, and tf[0] /
     * tb[0] the engine's two direction tags, so every walker finds the engine in the
     * stage-0 table slot and runs it through _il2d_rd0_hook; rd0 owns it */
    void *rd0;
    int wc;                   /* column-tile width (complex); 0 = full rn (untiled) */
    /* the BANDED walk: wl = band width in
     * ROWS (0 = unbanded); cut = DERIVED from wl (the first stage with
     * L_s | wl; wide prefix stages run first); tfuse folds the ROW pass per
     * band (c2c only — the owning plan runs the rows) */
    int wl, cut, tfuse;
    /* the staged band route: copy each band into scratch at pitch
     * (a skew that kills the 4KB set-group aliasing), suffix + rows there,
     * copy back. Requires wl > 0. c2c only. */
    int staged, pitch;
    vfft_real *bandscr;          /* 2 * wl * pitch doubles */
    int colmt;                /* the RACED column-MT verdict for this cell at the plan's T */
    /* NATURAL n1: the leaf call for block b writes its R rows at
     * out-base natperm[b*R] with OLs = (N/R)*rn — natural order as a driver
     * redirection, any chain; bwd gathers from the natural positions.
     * natperm is block-affine by construction (asserted at create). */
    int nat;
    int natarm;               /* the THREADED partition, RACED at create: 0 = bands (or the
                               * natural block partition), 1 = strips — every class
                               * (il2d_large_plane_design.md) */
    int msw;                  /* the threaded strips' sub-strip width in columns (0 = the
                               * worker's whole range); banked msw= beside cmt on the plan's row (v1.3) */
    int *natperm;             /* N entries, scr row -> natural row */
    vfft_real *natscr;           /* the pre-leaf plane: N rows sized VFFT_IL2D_NAT_CAP(rn) complexes (below) */
    int natst;                /* the leaf STAGED (1) or at its natural stride (0): the serial
                               * walk's form is staged (dominant) unless its row names nls= (the
                               * estimate's LEAF rule, estimate_nd_il.h); the threaded arms race both,
                               * banked nls= beside cmt on the plan's row (il2d_natural_leaf_design.md) */
    vfft_real *natstage;         /* the leaf's STAGING: T slots of R_last rows sized VFFT_IL2D_NAT_CAP(rn)
                               * complexes, one a worker (_il2d_stage_alloc / _il2d_stage_slot below;
                               * il2d_natural_leaf_design.md);
                               * NULL = the leaf stores at its natural stride */
    vfft_real **natsscr;         /* the threaded strips' DENSE per-worker scratch:
                               * nnatsscr blocks of N rows of _il2d_scr_cap(natswcap) complexes;
                               * the natural strip's pitch is _il2d_scr_pitch(the strip width), the
                               * others' the width (the 3D tier's strip form, _il2d_col_pass_nat_strip); the
                               * lane-major batches build it for the scrambled chain too (_lm_sscr_build:
                               * the real batch's fused strips, the c2c batch's blocks) */
    int nnatsscr, natswcap;
    int natdense;             /* the dense strip form is on (bound at build: VFFT_IL2D_DENSE=0 keeps the shared scratch; no getenv on the execute path) */
    /* column-axis BLUESTEIN: chirp convolution at M = next pow2 >= 2N-1 over
     * an M x rn scratch plane; blu = M (0 = off); R/L/f/b/tf/tb hold the
     * M-chain */
    int blu;
    vfft_real *bluchf, *bluchb;  /* chirp, 2*N each, fwd/bwd */
    vfft_real *blukf, *blukb;    /* comb-order kernels, 2*M */
    vfft_real *bluscr;           /* the M x rn plane, 2*M*rn */
    /* THE TURNED PRIME COLUMN PASS (tpc): at a prime N the axis
     * runs through the 1D prime route instead of the column Bluestein -- the
     * lanes transposed into tpcscr (pitch N + 8), the in-place natural K=1
     * plan at N on every row of it, transposed back. blu stays = M as the
     * prime-column marker (natural out, no chain); the Bluestein tables are
     * not kept when tpc serves. Raced at create, banked tpc= on the column
     * row. Threaded: the turns by rows on every worker, the plan by lanes
     * through each worker's own clone (_il2d_tpc_cols_mt), the clones built at
     * the cell's T. */
    int tpc;
    struct vfft_plan_s *tpcplan;
    vfft_real *tpcscr;           /* rn x VFFT_IL2D_TPC_PITCH(N) complex */
    struct vfft_wisdom_s *tpcS;   /* the plan's private store: its recipe rides on the column row as tpc_* */
    struct vfft_plan_s **tpcw;    /* the clones of tpcplan for workers 1 .. tpcw_n (the cell's T - 1), replayed from tpcS */
    int tpcw_n;
    /* a LANE-MAJOR cell's serial lane tile as its chain race crowned it with the chain (il2d_tier.h,
     * _il2d_race_chains: every candidate timed in the serial forms its door serves; 0 = no tile: the walk, or the
     * real batch's whole plane); -1 = no chain race in this build (a replayed or pinned chain: the lane-major race
     * replays sw= or races it). Read once, by the lane-major race (il/rank1/lm_batch_il.h, _lm_race). */
    int lmsw;
} vfft_ilcol_t;

/* THE NATURAL PASS'S OWN ROW PITCH (2026-11-24), in complexes: the row pitch of the two buffers the natural column
 * pass owns, the pre-leaf plane (natscr) and the leaf's staging block, as the walks named below address them -- rn,
 * and one cache line more where a row of rn complexes is a multiple of 512 B.
 * WHY: a stage's legs are whole rows apart. Where a row is a multiple of 4 KiB (rn = 256: one L1d way on the 14900KF
 * and on Zen 4) every leg of a stage meets in one L1d set, and a set holds 12 lines (8 on Zen 4): a stage of 15, 21,
 * 25 or 27 legs throws out lines it is still reading, and at the plane's pitch the pre-leaf plane and the staging
 * block inherit that from the caller's plane. With the line, legs D rows apart sit D lines apart in the set index:
 * no two in one set for D < 64. Legs 64 rows apart (and its multiples) meet again whatever the line: the early stages
 * of a long power-of-two column gain nothing from it. The caller's planes keep their pitch: the first stage's loads
 * and a strided leaf's stores are theirs.
 * WHO LAYS THEM AT IT: the c2c 2D plan's walks -- on one thread both directions, unbanded and banded (il_execute.h);
 * threaded, the block and the tile partition (il2d_tier.h _il2d_c2c_mt; the strip partition has its own dense
 * scratch at _il2d_scr_pitch, its wide-strip fallback on the shared plane stays at rn) -- and the serial natural arms
 * of that cell's races (il2d_tier.h _il2d_key_np: the arm is the serving walk). A 3D plan's planes are such 2D plans;
 * its flat tier runs the middle axis's stages inside the caller's volume and owns no buffer to lay out. Every other
 * walker of a natural chain -- the real tier, the lane-major batches -- lays them at rn: not measured there.
 * EVIDENCE (14900KF, AVX2, one thread; the same build with this defined (rn) beside):
 *  - the column pass alone, forward, out of place, every chain of the pool within one stage of the fewest, five
 *    planes of 75 to 525 columns on rows of 256, us at the plane's pitch -> at this one: 225x256 15.15 141.7 -> 65.2,
 *    9.25 108.8 -> 69.7, 9.5.5 80.9 -> 63.9; 75x256 15.5 27.1 -> 15.8, 5.5.3 17.3 -> 16.0; 135x256 9.15 59.7 ->
 *    31.5; 315x256 15.7.3 148.7 -> 99.5; 525x256 15.7.5 271.9 -> 176.8. Both buffers: the staging left at the
 *    plane's pitch keeps a large leaf slow (75x256 5.15 29.9 against 16.3); N2 + 8 runs as N2 + 4.
 *  - the whole transform on the estimate's plan, the plan the same in both builds, 37 planes, five alternated runs:
 *    forward 0.981, backward 0.979 (geometric means of this over the plane's pitch; 45x256 0.93, 64x256 backward
 *    0.92, 256x256 0.96); the three planes whose rows are no multiple of 512 B 0.98 to 1.01 (the pitch is rn there);
 *    planes past the L2 (2187x256, 1000x512, 2048x1024) move 10 to 25 % run to run in either build, their medians
 *    within 6 %. Bitwise the same output on all 37.
 *  - the measuring planner from cold, each build its own store, 8 planes: forward 0.940, backward 0.947; the chain
 *    changes at 6 (75x256 5.5.3 -> 15.5 0.92, 225x256 5.5.3.3 -> 9.5.5 0.89, 525x256 7.5.5.3 -> 15.7.5 0.89, 64x256
 *    4.8.2 -> 16.4 0.91). 225x512 takes 15.15 where it took 9.5.5: forward 1.03 to 1.04, backward 0.89 to 0.92 (three
 *    cold plans); 256x256's plan differs from one cold plan to the next (forward 0.96, 1.04, 1.07; the plan of
 *    before every time at the plane's pitch) -- the chain race times the column walk without the rows.
 *  - every chain of 427 planes (61 non-power-of-two columns over rows of 16 to 1050), the plain pass and the rows:
 *    on the rows that get this pitch 1346 chains, 614 faster by more than 3 %, 720 within, 12 slower by 3 to 5 %
 *    (rows of 256: 0.878, none slower; 1024: 0.906; 64: 0.973).
 *  - eight threads, the estimate's plans, 22 planes, five alternated runs: 0.981 by the median, none slower by more
 *    than 3 %; the tile partition with a radix of 15 or 25 in its chain 1050x256 0.81, 4375x256 0.82; the block
 *    partition forced: 0.980, five planes faster by more than 3 %, none slower. 15 volumes at one and at eight
 *    threads: 0.997 and 0.992. The same bits as at the plane's pitch in every case.
 * A build may define it (a probe's A/B: (rn) is the plane's pitch); VFFT_IL2D_NAT_CAP is what the buffers' rows are
 * sized for. */
#define VFFT_IL_SCR_PAD ((size_t)(64 / (2 * sizeof(vfft_real))))   /* one cache line, in complexes */
#ifndef VFFT_IL2D_NAT_PITCH
#define VFFT_IL2D_NAT_PITCH(rn) \
    ((((size_t)(rn)) * 2 * sizeof(vfft_real)) % 512 == 0 ? (size_t)(rn) + VFFT_IL_SCR_PAD : (size_t)(rn))
#endif
#define VFFT_IL2D_NAT_CAP(rn) ((size_t)(rn) + VFFT_IL_SCR_PAD)

/* THE LEAF'S STAGING, ONE SLOT A WORKER: the leaf writes a block's R_last rows of rn complexes into its worker's slot
 * and the rows run there, so every worker walks its slot again and again beside its neighbours'. The slots are the
 * allocator's (common/support/zalloc.h, vfft_apart_slots_alloc: each on cache lines of its own, 8 KiB after the one
 * before ends -- the measurements are there). A slot's address follows from the chain's last radix and the lane
 * count alone: every walker of a chain finds the same slot, and a block built for the largest last radix serves a
 * chain with a smaller one. Its rows are sized for the natural pass's own pitch (above): a walker lays them at that
 * pitch or at rn. */
static inline size_t _il2d_stage_slot_bytes(int Rl, size_t rn)
{
    return 2 * (size_t)Rl * VFFT_IL2D_NAT_CAP(rn) * sizeof(vfft_real);
}
static inline vfft_real *_il2d_stage_alloc(int T, int Rl, size_t rn)   /* T slots (one when T < 1); NULL = out of memory */
{
    return (vfft_real *)vfft_apart_slots_alloc((size_t)(T > 0 ? T : 1), _il2d_stage_slot_bytes(Rl, rn));
}
static inline vfft_real *_il2d_stage_slot(vfft_real *stage, int tid, int Rl, size_t rn)   /* worker tid's (the caller: 0) */
{
    return (vfft_real *)vfft_apart_slot(stage, (size_t)(tid > 0 ? tid : 0), _il2d_stage_slot_bytes(Rl, rn));
}
static inline void _il2d_stage_free(vfft_real *stage)
{
    vfft_apart_free(stage);
}

#endif /* VFFT_TRANSFORMS_FFT2D_IL2D_COL_H */
