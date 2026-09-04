/* vfft_internal.h - the three private handle types.
 *
 * vfft_wisdom_s, vfft_plan_s and vfft_batch_s: the structs behind the opaque
 * pointers the public API hands out. Lifted from vfft.c as migration step 15;
 * see docs/design/refactor_migration_plan.md.
 *
 * WHY THIS STEP IS THE GATE
 * -------------------------
 * 67 of the 137 function definitions in vfft.c take one of these three types -
 * 8,520 lines, 71% of the file, including _vfft_create_inner (4,006 lines) and
 * vfft_execute (656). While the definitions lived inside vfft.c, none of that
 * code could move anywhere: a function that dereferences a plan cannot be
 * extracted into a module header while the plan's layout is private to one
 * translation unit.
 *
 * Everything before this step was deliberately a rehearsal on struct-free code -
 * kernels, converters, racers that take their inputs explicitly. Ten such moves
 * ran green before this one was attempted, which is the point: the harness was
 * exercised end to end on work that reverts with one git checkout, so that when
 * it says EQUIVALENT here it is being trusted rather than tested.
 *
 * THE CASE FOR IT WAS ALREADY IN THE TREE
 * ---------------------------------------
 * Four bench translation units textually #include "vfft.c" - compiling all
 * 12,000-odd lines - for no reason other than needing these struct layouts.
 * That is the symptom this header exists to cure.
 *
 * INCLUSION CONTRACT
 * ------------------
 * This header describes LAYOUTS, not behaviour: its fields name types owned by
 * the engine headers (stride_plan_t, vfft_zturn2_plan_t, vfft_il2p_plan_t,
 * natorder_scr_t, vw2_store_t and others). It must therefore be included AFTER
 * the engine prelude, exactly as vfft.c does. It deliberately does not
 * replicate that prelude: the prelude interleaves ISA-conditional macro
 * definitions with its includes, and duplicating it would mean two places that
 * must agree about which registry is selected.
 *
 * Giving the four bench TUs a standalone spelling is step 21's business, not
 * this one's.
 *
 * NOTHING HERE IS A DECISION
 * --------------------------
 * These are field layouts. No function, no state, no macro that selects
 * behaviour - so this header can be included from anywhere in the internal DAG
 * without creating an edge that constrains anything.
 */
#ifndef VFFT_INTERNAL_H
#define VFFT_INTERNAL_H

/* ════════════════════════════════════════════════════════════════════════
 * OPAQUE TYPES
 * ════════════════════════════════════════════════════════════════════════ */

struct vfft_wisdom_s
{
    char path_c2c[640];       /* spike_wisdom.txt   */
    vfft_proto_wisdom_t c2c;  /* c2c inner / decoupled-r2c inner format. Also holds the padded
                               * pad-vs-tail verdict per cell in each entry's exec_me field, and
                               * the aligned (N,Kp) entries pad reuses — no separate padded file. */
    vfft_oop_wisdom_t oop;    /* OOP 2-axis format   */
    vfft_proto_wisdom_t rfft; /* r2c rfft-path factorization+variant   */
    /* Dedicated 2D wisdom (end-to-end-2D measured, independent of 1D c2c). One
     * entry per (N1,N2), two sub-plans each. r2c and c2r have separate tables
     * (different optima, same bidirectional plan structure). */
    vfft_fft2d_c2c_wisdom_t fft2d_c2c;
    vfft_fft2d_r2c_wisdom_t fft2d_r2c;
    vfft_fft2d_r2c_wisdom_t fft2d_c2r; /* shared struct, c2r-tuned plans */
    vfft_fft3d_wisdom_t fft3d_c2c;     /* dedicated 3D table (B + a_block + 3 axis chains) */
    char path_bluestein[640];          /* bluestein_wisdom.txt */
    bluestein_wisdom_t bluestein;      /* prime-N (M,B) for Bluestein cells (Rader needs none) */
    /* 1D c2r NATURAL-vs-STRIDE path decision (c2r_path.txt; "N K path", 0=natural,
     * 1=stride). Loaded into the file-static _vfft_c2r_paths table (c2r_dispatch.h)
     * for the non-bakeoff (MEASURE / high-K) dispatch; high rigor measures instead. */
    char path_c2r_path[640]; /* c2r_path.txt */

    /* wisdom2 (the new store, src/core/wisdom2/README.md). Wave 1 flips the
     * OOP family here: reads via the vw2_oop_* twins, banks via
     * vw2_oop_bank_entry (memory) + guarded vw2_save (config.wisdom_write).
     * Legacy oop_wisdom.txt stays loaded ONLY as the kill-switch fallback
     * (VFFT_WISDOM2_OFF containing "oop" flips READS back to it during the
     * bake window; writes go to wisdom2 either way — banking never mutates
     * a frozen file). */
    vw2_store_t vw2;
    int vw2_off_oop;   /* kill switch, cached at bundle load */
    int vw2_off_2d;    /* kill switch: 2D reads fall back to the legacy
                          tables (3D has no legacy fallback — born in
                          wisdom2) */
    int vw2_off_stride;/* kill switch: stride/spike reads fall back */
    char dir[512];     /* the bundle's directory (wisdom2 opens from it) */
};

struct vfft_plan_s
{
    vfft_transform_t transform;
    vfft_placement_t placement;
    /* Committed layout axis (vfft_layout_t, stamped at create). Execute
     * dispatches on THIS — never on the pointer signature (the historical
     * NULL-inference is removed); a signature that contradicts it is a loud
     * refused error. Construction itself is layout-independent (split-default
     * plans are untouched by the axis; an INTERLEAVED commitment only selects
     * the z dispatch + enables the convert fallbacks). */
    int layout;
    int N;
    int N2; /* 2D second dim (0 = 1D)    */
    int N3; /* 3D third dim  (0 = 1D/2D) */
    int N4; /* 4D fourth dim (0 = rank<4)  — §6a62 */
    size_t K;
    int nthreads;
    stride_plan_t *cplan;   /* c2c in-place (owned)      */
    vfft_oop_plan_t *oplan; /* c2c out-of-place (owned)  */
    /* K=1 engine (row_major_engine.md §13; c2c OOP, howmany==1, natural).
     * Route per axis from kind-3 wisdom (or the default heuristic); the axis
     * is the plan's COMMITTED layout (h->layout, stamped at create — the old
     * execute-time buffer-contract inference is gone). k1sp is the BAILEY2V
     * plan for the SPLIT pair (owned); the IL axis is k1il2p below — the
     * hybrid k1il plan and its _sw entry points were deleted 2026-07-29.
     * Split bwd = pointer-swap identity. Kill-switch: env VFFT_NO_K1 at
     * create. */
    int k1_on;
    int k1_sp_route, k1_il_route;
    vfft_oop_plan_t *k1sp;
    /* K=1 SCRAMBLED interleaved z->z: the block-split cascade (zsplit.h;
     * ≥2048 cells, default chains = calibrated winners). Serves ONLY plans
     * committed to layout=INTERLEAVED; split-layout plans and uncovered
     * cells go through the classic path (uncovered IL cells convert). Owned. */
    vfft_zsplit_plan_t *zsplit;
    /* ROUTE AXIS for that cascade (cascade_load_path_restructure §6.4/§2.6):
     * zroute is the ONE field BOTH execute directions dispatch on (0 = legacy
     * zsplit, 1 = ZTURN-S). Cutover atomicity is STRUCTURAL: create keeps
     * exactly one cascade plan (the loser is destroyed before the handle
     * exists) and _exec_zcascade is the single consumer, so a mixed
     * fwd-legacy/bwd-zturn pairing is inexpressible, not just unlikely.
     * Invariant: zroute==1 <=> zturn!=NULL && zsplit==NULL. The SCRAMBLED
     * contract permits the routes' different output permutations (§2.6) —
     * a route's OWN bwd always consumes its OWN fwd comb. ZTURN is the
     * DEFAULT route on a wisdom miss (2026-07-27 cutover; banked route
     * verdicts — including old-format = legacy — are honored). Kill switch:
     * env VFFT_NO_ZTURN at create pins legacy (VFFT_NO_IL2P precedent);
     * VFFT_FORCE_ZROUTE=legacy|zturn is the gate/test forcing hook. */
    /* ── 2D PLANE QUEUE (howmany > 1, sequential contiguous planes —
     * the designed 2D batching feature; docs/design/il2d_real_mt.md
     * increment 5/M6). The handle is a thin wrapper: pq_inner is the
     * PRIMARY howmany=1 2D plan built with the caller's thread budget
     * (so the serial plane loop still intra-MTs per its own banked
     * verdicts — manual batching is never beaten by building this),
     * and pq_w[0..pq_wn) are SERIAL clone plans for the queue (a queue
     * worker must not nest-dispatch into the pool; the CALLER also
     * runs a serial clone in queue mode, slot 0). Loop-vs-queue is
     * RACED at create (pq_mt); clones are BITWISE-verified against the
     * primary on a probe plane or the queue declines. */
    struct vfft_plan_s *pq_inner;
    struct vfft_plan_s **pq_w;
    int pq_wn;
    size_t pq_n;              /* plane count (howmany) */
    size_t pq_sdist, pq_ddist; /* plane strides, DOUBLES (contiguous) */
    int pq_mt;                /* raced verdict: 1 = queue, 0 = loop */
    int zroute;
    vfft_zturn2_plan_t *zturn;
    /* K=1 cascade MT verdict (INC-Z, the 2D design ported to 1D): 1 =
     * thread the zturn walk. RACED at the OOP scrambled commit when the
     * pool is live (serial default everywhere else — natural/in-place
     * commits inherit later). The cascade needs NO clones: one read-only
     * plan, and every phase partitions the sectioned plane disjointly. */
    int zt_mt;
    /* K=1 NATURAL interleaved z->z, PURE IL (il2p.h): n1t -> z scratch -> t2,
     * no split planes, BOTH directions (bwd = t2t then n1_bwd(R2), solved
     * 2026-07-29). THE IL 2-pass plan — the hybrid it displaced measured
     * 0.558x @N=64 / 0.765x @256 / 0.956x @1024 against it (scalar-DFT
     * gated) and was deleted. Owned; NULL <=> k1_il_route != 2P_PURE/MONO. */
    vfft_il2p_plan_t *k1il2p;
    /* K=1 NATURAL interleaved z->z, PURE-IL 3-STAGE CHAIN (il2p.h il3p):
     * odd·2^k N in the Bailey band (route VFFT_K1_IL_CHAIN3; both dirs,
     * gated — docs/roadmap/il_odd_chain.md). Such cells have NO split K=1
     * route, so the handle may exist with k1_sp_route == -1; that is legal
     * ONLY for INTERLEAVED-committed plans (create guards it — the split
     * dispatch can never reach an IL-only handle). Owned. */
    vfft_il3p_plan_t *k1il3p;
    /* K=1 PRIME N on the IL machinery (il_prime.h; route VFFT_K1_IL_PRIME):
     * Rader or Bluestein over il2p/il3p inner plans, both dirs, natural.
     * Same IL-only-handle rules as k1il3p. Owned. */
    vfft_ilprime_plan_t *k1ilpr;
    /* TRANSFORM-CONTIGUOUS batch (config.batch_geom, 1D C2C interleaved,
     * K>1): this handle is a thin WRAPPER — `tcb` is a fully-built K=1
     * handle and execute simply runs it K times at 2*N-double strides.
     * Non-NULL <=> this is a wrapper handle, and then NOTHING else on the
     * struct is live except transform/placement/layout/N/K/nthreads and the
     * clone set below. Owned; destroy frees it. Serving a batch as K
     * independent transforms is why this geometry needs no batched
     * machinery, no layout conversion, and inherits every K=1 improvement
     * for free. */
    struct vfft_plan_s *tcb;
    /* TC MT (split's per-lane trick, one level up: per-TRANSFORM). The K=1
     * IL engines are NOT reentrant — il2p/il3p own `mid` scratch, zturn owns
     * its sectioned `plane` — so worker t runs its slab of transforms on its
     * OWN identically-created K=1 handle, never on a shared one. Clones are
     * built at create ONLY when the inner route is provably pool-free both
     * directions (_tc_inner_mt_safe) and each clone is verified
     * output-equivalent to the primary (_tc_clone_equiv) — a wisdom-absent
     * cascade cell can re-race at create, and two clones with different
     * chains would emit different scrambled combs inside ONE batch. A clone
     * that fails the check is destroyed and the worker set stops growing
     * (degrade = fewer workers / serial, never a mixed batch). tcbw_n == 0
     * <=> tcbw == NULL <=> serial loop (today's path, byte-identical). */
    struct vfft_plan_s **tcbw;
    int tcbw_n;
    /* The THREADING verdict for the wrapper: 1 = slabs over the clones,
     * 0 = the serial loop. Raced at create (serial vs slabs on this cell)
     * or replayed from its eng=tcb row; T-free (one transform per core).
     * Without clones it is 0 by construction. See _tc_mt_decide. */
    int tc_mt;
    /* Per-transform block strides IN DOUBLES for the wrapper loop. C2C has
     * one stride (2*N both ends), but the REAL transforms do NOT: r2c reads
     * N reals and writes 2*(N/2+1) CCE doubles, c2r is the mirror, and the
     * in-place real shape is 2*(N/2+1) at BOTH ends. So the stride is a
     * property of (transform, placement), computed once at create and stored
     * -- never re-derived at execute, and never assumed equal. */
    size_t tcb_sn, tcb_dn;
    vfft_oop11_fn k1_mono, k1_mono_ilf, k1_mono_ilb;
#ifdef VFFT_USE_JIT
    /* K=1 stride-baking JIT (§13.3): the winner split route compiled at plan
     * time with the cell's exact geometry as literal constants. NULL = the
     * normal route fns (JIT is a speed cache, never a correctness dep).
     * k1_jit_qr/qi = the table the baked stage-2 expects (Qlr/Qli for TWL,
     * Qr/Qi otherwise); scratch comes from k1sp->col_re/col_im. */
    vfft_k1_jit_fn k1_jit;
    const double *k1_jit_qr, *k1_jit_qi;
#endif
    vfft_r2c_plan_t *rplan;    /* r2c fwd (owned)           */
    vfft_c2r_disp_t *c2rdisp;  /* 1D c2r 2-axis: NATURAL/STRIDE (owned) */
    /* §D2 zr2c (2026-08-13, DESIGN_interleaved_r2c.md Phase 2): 1D
     * INTERLEAVED-CCE real transforms as reinterpret + CHILD c2c(N/2) +
     * the zr2c.h fold. zr2c_child != NULL selects this route over
     * rplan/c2rdisp at execute. route 0 = OOP-IL child (natural OOP c2c);
     * route 1 = NAT-IP cascade child (natural in-place c2c — MKL's own
     * regime routing, validated 2026-08-13). Verdicts belong to the
     * zr2c-owned wisdom kind (owner directive); until the calibrator
     * lands, create uses the placement-matched STRUCTURAL default and
     * the VFFT_ZR2C_ROUTE env override (env beats wisdom, house rule). */
    struct vfft_plan_s *zr2c_child; /* c2c(N/2) plan (owned)              */
    /* ── the ODD-REAL BRIDGE (2026-08-27, 1D K==1): real <-> CCE through
     * the c2c engine — fwd: promote real -> complex, c2c fwd, keep the
     * hp1 bins; bwd: Hermitian-extend hp1 -> N (no Nyquist at odd N —
     * the mirror is exact), c2c bwd, take Re. The 2D odd-N2 row
     * primitives lifted to the 1D front door. Serves BOTH layouts (the
     * split spellings pack/unpack around the same child). Closes the
     * two 1D real holes: c2r at ANY odd N (nothing else exists), and
     * r2c at non-radix-smooth odd N (prime/awkward — the c2c child
     * rides the pair/chain/prime engines). Smooth-odd r2c keeps its
     * native rfft route; racing the two is the sweep's. */
    struct vfft_plan_s *oddr_child; /* c2c(N) K=1 NATURAL OOP IL (owned) */
    double *oddr_buf;               /* 2 x 2N doubles: the row pair      */
    int zr2c_route;                 /* 0 = OOP-IL child, 1 = NAT-IP child */
    double *zr2c_aff;               /* affS ++ affC (one allocation)      */
    double *zr2c_scratch;           /* N+2 dbl, route-0 placements only   */
    stride_plan_t *tplan;      /* trig DCT/DST/DHT (owned)  */
    vfft_r2c_plan_t *rfft_row; /* §6a31: 2D row-pass rfft inner (owned)   */
    vfft_c2r_disp_t *c2r_row;  /* §6a32: 2D bwd row-pass c2r inner (owned) */
    /* config.owned_buffers: the planes THIS plan allocated and will free.
     * NULL when the caller brings their own buffers (the drop-in default). */
    struct vfft_batch_s *own_batch;
    /* Transparent JIT/baked-resolved c2c in-place executor (NULL = generic). Resolved
     * once at create; execute calls it directly (zero JIT overhead in the hot path). */
    vfft_proto_exec_fn exec_fwd, exec_bwd;
    /* Padded c2c in-place (config.batch != NULL): cplan is built at Kp = the batch stride,
     * and execute runs `exec_me` batch lanes (Kp = full-SIMD pad, or K = SSE2/scalar tail
     * on the padded buffer — the padded wisdom's per-cell verdict). padded==0 => tight, the
     * default; exec_me is then unused (tight runs p->K via _c2c_mt). See padding_design_decision.md. */
    int padded;
    int exec_me;
    /* ── native IL 2D c2c tier (docs/roadmap/fft2d_il_c2c_design.md):
     * n1c/t2c column chain + per-row K=1 IL child (order NATURAL). THE
     * serving for IL 2D c2c — OWNER LAW 2026-08-25: split is NOT a
     * fallback of IL, the convert wrapper is GONE; inexpressible cells
     * REFUSE at create. Cold cells race + bank (lay=il) + serve. */
    struct vfft_plan_s *il2d_row;
    /* the column chain: stage s has radix il2d_R[s] over sub-length
     * il2d_L[s] (D = L/R); stages 0..nst-2 are t2c with driver-built
     * d-major record tables (il2d_tf fwd / il2d_tb conjugated bwd), the
     * last stage is the twiddle-free n1c. nst == 1 = the M1 single-stage
     * tier (identity order along i); nst > 1 leaves i digit-reversed by
     * the chain (the scrambled contract; natural = M4 rho tables). */
    int il2d_nst;
    int il2d_wc; /* column-tile width (complex); 0 = full N2 (untiled) */
    /* banded walk (the cascade's tcut mapped to 2D — zturn.h §TILING AXIS):
     * il2d_wl = band width in ROWS (0 = unbanded); il2d_cut = DERIVED
     * (stages cut..nst-1 run depth-first per band — the suffix whose
     * L_s | wl; wide prefix stages run first); il2d_tfuse folds the ROW
     * PASS per band (the terminator analog). F0 law: banding changes only
     * loop order + base pointers — output memcmp-identical to unbanded. */
    int il2d_wl, il2d_cut, il2d_tfuse;
    /* row route (the small-N2 lever): 0 = in-place NATURAL child (the
     * default; pays the 1D in-place service floor ~180ns/row at tiny N);
     * 1 = OOP NATURAL child + L1-hot row scratch + memcpy back (the mono
     * route). Env-raced (VFFT_IL2D_ROWOOP=1); banked with §10. */
    int il2d_rowoop;
    struct vfft_plan_s *il2d_rowo; /* the OOP child (route 1) */
    /* ── c2c MT (INC-C, docs/design/il2d_real_mt.md ported): per-worker
     * row state. The serving row path is ONE shared child through ONE
     * shared rowscr — two concurrent bands would interleave plan state
     * and produce garbage, so worker t>0 gets clone slot t-1, verified
     * route-equivalent at build (_tc_clone_equiv). Worker 0 (the
     * caller) keeps the primary child/scratch. roww_n == 0 => MT
     * declines (the engagement counter shows it). */
    struct vfft_plan_s **il2d_roww; /* clone row children, T-1 slots */
    int il2d_roww_n;
    double *il2d_rowscr_w;          /* rowoop: T-1 slots x 2*N2 */
    double *il2d_rowscr;           /* 2*N2 doubles */
    /* staged band route (§10b): copy each band into scratch at il2d_pitch
     * (skew-selected so every suffix-stage leg stride and the leaf stride
     * are non-0 mod 4096 — the priced 2.4-3x aliasing cure), run the
     * suffix + rows there, copy back. Requires il2d_wl > 0. */
    int il2d_staged, il2d_pitch;
    double *il2d_bandscr;          /* 2 * wl * pitch doubles */
    int il2d_R[8], il2d_L[8];
    vfft_il2p_fn il2d_f[8], il2d_b[8];
    double *il2d_tf[8], *il2d_tb[8];
    /* native IL 2D REAL tier (docs/roadmap/fft2d_real_il_design.md): the
     * same il2d_* chain machinery over hp1 = N2/2+1 columns; il2d_row =
     * the TC K=N1 batched zr2c row door (one plan, one dispatch per
     * plane). il2d_rscr = the OOP c2r column-inverse plane (§2.6
     * input-preserving contract), 2*N1*hp1 doubles; NULL for r2c.
     * wl/tfuse/rowoop/staged stay 0 for real — the Hermitian fold does
     * not commute with the column stages (§2.5), rows sit OUTSIDE any
     * banded walk. */
    double *il2d_rscr;
    /* the ROWSPLIT row route (owner 2026-08-26: "il at the boundary,
     * split inside" — the cascade pattern, NOT the banned wrapper): rows
     * in bands of il2d_rw; per band = SIMD transpose rows->lanes ->
     * il2d_rows (the raced SPLIT r2c/c2r engine at (N2, K=rw)) ->
     * transpose + zip into the IL plane. Kills the per-row dispatch toll
     * at tiny N2 (priced 7.7 vs 29 ns/row at 4096x16). Lane planes are
     * padded to hp1p = (hp1+3)&~3 rows (the 4x4 transpose reads whole
     * quads; the garbage tail lands in rows nothing reads). NULL/0 =
     * the per-row TC door serves. Env VFFT_IL2D_ROWSPLIT=W (raced arm;
     * serving verdicts are M3's route race). */
    struct vfft_plan_s *il2d_rows;
    int il2d_rw;
    int il2d_colmt;  /* INC-3: the RACED column-MT verdict for this cell
                      * (1 = thread the column pass, 0 = serial). Never a
                      * structural default — at 512x32 (hp1=17, so the
                      * strip arm over 17 columns) threading the columns
                      * MEASURED SLOWER, and the race banks that "no"
                      * exactly as it banks a "yes". */
    int il2d_oddn2;  /* ODD N2 (2026-08-27): rows ride a K=1 c2c child
                      * (il2d_row) — promote/extend + transform + take,
                      * fft2d_real_il.h primitives. hp1 = (N2+1)/2; the
                      * column machinery is count-agnostic so everything
                      * past the rows is the even path unchanged. */
    double *il2d_orbuf; /* 2 x 2*N2 doubles: the odd row's in/out pair */
    /* ── ODD/PRIME N1 c2c (2026-08-27): the COLUMN-AXIS BLUESTEIN.
     * When N1 has no native chain, the column transform runs as a
     * chirp convolution at M = next pow2 >= 2*N1-1, riding the SHIPPED
     * pow2 chain machinery over an M x N2 scratch plane: modulate rows
     * by the chirp, fwd chain, pointwise multiply by the kernel (built
     * at create through the SAME chain => comb order matches), bwd
     * chain consumes the comb, demodulate. n1 comes out NATURAL. Zero
     * new codelets. The odd t2c/n1c EMISSION (the corpus has the odd
     * DFT bodies, radix 3..27 — the column kind was never emitted) is
     * the future raced arm for smooth-odd N1. il2d_blu = M (0 = off);
     * il2d_R/L/f/b/tf/tb hold the M-chain. */
    /* ── NATURAL n1 (2026-08-27, "M4-lite"): the leaf-only pitch
     * theorem (n1c loads legs at Ls, stores at OLs, independent —
     * verified in the emitted body) makes natural output a DRIVER
     * redirection: the leaf call for block b writes its R rows at
     * out-base perm[b*R] with OLs = (N1/R)*pitch — natural n1, zero
     * new codelets, any chain (pow2 AND odd). bwd mirrors on the
     * SOURCE side (the leaf runs first in the reversed chain and
     * gathers its legs from natural positions via Ls). il2d_natperm =
     * the scr->nat table, block-affine by construction (asserted at
     * create: perm[bR+r] == perm[bR] + r*(N1/R) — a wrong digit
     * convention fails the create, never serves silently). Natural
     * cells pin wl=0 (the leaf scatters across bands) and skip the MT
     * arms (v1). Single-stage cells stay natural-native as before;
     * blu cells are natural BY CONSTRUCTION and now accept the order. */
    int il2d_nat;
    int il2d_natarm; /* natural x MT partition, RACED at create: 0 = the
                      * matched arm (digit-split prefix + block-range
                      * leaf + row slabs), 1 = column STRIPS (the whole
                      * natural pass over a column range). Plan-local
                      * (banking rides the wisdom wave). */
    int *il2d_natperm; /* N1 entries, scr row -> natural row */
    double *il2d_natscr; /* 2*N1*rn: the pre-leaf plane — the natural
                          * leaf SCATTERS, so it must never write the
                          * plane it still reads (block b's natural
                          * targets can be block b' > b's unread comb
                          * rows — the clobber the first cut shipped) */
    int il2d_blu;
    double *il2d_bluchf, *il2d_bluchb; /* chirp, 2*N1 each, fwd/bwd  */
    double *il2d_blukf, *il2d_blukb;   /* comb-order kernels, 2*M    */
    double *il2d_bluscr;               /* the M x N2 plane, 2*M*N2   */
    int il2d_norowz; /* 1 = skip the fused row-mode doors (the staged
                      * 3-pass route serves) — the A/B race knob, read
                      * from VFFT_IL2D_NO_ROWZ at CREATE (env cost never
                      * reaches execute; two plans in one process can
                      * differ = same-run arms). */
    double *il2d_lx, *il2d_lre, *il2d_lim; /* lane-major: N2*rw, hp1p*rw x2 */
    double *il2d_tre, *il2d_tim;           /* row-major halves: rw*hp1p x2 */
    /* 1 = the c2c in-place plan's codelet IGNORES the partial-lane count `me` (processes the full baked K),
     * so a _c2c_mt K-split slab would overrun adjacent lanes -> wrong output. Detected once at create by a
     * whole-vs-split self-check; when set, the FFT runs WHOLE-BATCH under MT (the reorder pass still threads).
     * Root cause: radix-8 LOG3 last-stage codelet. See memory mt_c2c_16x8_wrong_output. */
    int mt_unsafe;
    /* VFFT_ORDER_NATURAL (in-place 1D c2c only): the per-cell verdict + its execute tape.
     * nat_mode==0 (UNSET) means order=DEFAULT — the scrambled path, byte-identical to
     * pre-natural builds (kill switch). P1a wires FREE + PURE_CYCLE; SCR/PSWAP/LEAF-IP in
     * P1b. nat_list = flattened cycle tape (natorder_perm.h), nat_tmp = 2*K doubles.
     * natural_order_inplace_design.md §2e. */
    int nat_mode;
    int *nat_list;           /* PURE/SCR: flattened cycle list; PSWAP: flat pair list        */
    double *nat_tmp;         /* nthreads*2*K: per-worker cycle scratch (slot nd = tmp+nd*2K); sized AND indexed by h->nthreads, never the live pool */
    int nat_ncyc;            /* PURE/SCR: cycle count (backward MT split); PSWAP: pair count */
    int *nat_cyc_off;        /* PURE/SCR: cycle start offsets (ncyc+1); PSWAP: NULL          */
    natorder_scr_t *nat_scr; /* SCR: scatter terminator (forward); backward reuses cycle tape */
    /* VFFT_ORDER_NATURAL for 2D c2c: per-axis digit-reversal reorder tapes. dim1 = whole matrix
     * rows (plan_col chain, N1 pts, K=N2 contiguous doubles/row); dim2 = within-row (plan_row chain,
     * N2 pts, K=1). Orthogonal axes => commute. nat2d==0 = scrambled (kill switch). First cut is
     * single-threaded PURE cycles; a NULL axis list = FREE (already natural). */
    int nat2d;
    int *nat2d_row_list;    /* dim1 (N1) reorder tape; NULL = FREE axis (see nat2d_row_is_pairs) */
    int nat2d_row_is_pairs; /* 1 = row tape is an involution PAIR list (pair_pass, no dep chain, fast);
                             * 0 = cycle list (cycle_pass). PSWAP when the column chain is palindromic. */
    int *nat2d_col_list;    /* dim2 (N2) cycle tape (fft2d scratch pass); NULL = FREE axis */
    double *nat2d_tmp;      /* nthreads slots of 2*N2 doubles: per-worker dim1 cycle scratch (MT); sized AND indexed by h->nthreads, never the live pool */
    int nat2d_ncyc;         /* dim1 unit count: cycles (cycle tape) or pairs (pair tape) — MT split */
    int *nat2d_cyc_off;     /* dim1 cycle start offsets (ncyc+1); NULL for a pair tape */
};

/* Opaque padded-batch handle (see vfft.h). Carries its own Kp stride so a padded
 * buffer can't be passed through the tight execute path by mistake, plus the feature
 * it was allocated for (a c2c handle must not be handed to an r2c create, etc.).
 *
 *   c2c (in-place):  real == NULL; re/im are the in-place split data, each N*Kp.
 *   c2c (OUT-OF-PLACE): re/im are the split INPUT, ore/oim the split OUTPUT, each N*Kp
 *                    (oop==1; Kp=roundup(K,8) so all 3 OOP kinds + wisdom caching work).
 *   r2c (fwd):       real = the real INPUT plane (N*Kp); re/im = the split spectrum
 *                    OUTPUT, each (N/2+1)*Kp.
 *   c2r (bwd):       re/im = the split spectrum INPUT, each (N/2+1)*Kp; real = the
 *                    real OUTPUT plane (N*Kp).
 *   trig:            real = real INPUT (N*Kp), re = real OUTPUT (N*Kp).
 * All planes are Kp-strided so the Kp-built plan addresses them correctly (element e
 * of transform t is at plane[e*Kp + t]); the pad columns t in [K,Kp) stay zeroed. */
struct vfft_batch_s
{
    double *real, *re, *im, *ore, *oim;
    size_t K, Kp;
    int N;
    int xform;
    int oop;
};
/* INTERNAL handle type. Since 2026-07-28 the batch is no longer part of the
 * public API: vfft_create owns it (config.owned_buffers) and vfft_destroy frees
 * it, so a plan and its buffers cannot disagree. Callers reach the planes/stride
 * through vfft_plan_planes() / vfft_plan_stride(). */
typedef struct vfft_batch_s *vfft_batch;

#endif /* VFFT_INTERNAL_H */
