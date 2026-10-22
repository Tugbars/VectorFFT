/* estimate_recipe.h -- the 1D estimate's interface: machine, request, recipe, contracts.
 *
 * Estimate is a planning method the user selects (never a fallback). From N, the request and the
 * machine's cache sizes it computes the recipe a race would have banked -- the route and every
 * parameter -- WITHOUT measuring, and builds the plan through the library's own replay door.
 * It never reads the user's wisdom store: the raced rows are only the evidence the formulas were
 * fitted to. Specification: docs/design/estimate_1d_sketch.md (its TEXT wins over the synth/ Python model).
 * Build reference: docs/research/estimate_1d/impl/REPLAY_API.md (§3 = this recipe).
 *
 * Scope: 1D, interleaved, FP64, c2c / r2c / c2r.
 * The request is the row's KEY, the recipe is the row's PAYLOAD (wisdom2's "@cell <key> | <payload>").
 *
 * Plain C, no library type, so the formulas compile on their own. Paths below are relative to
 * src/core/ unless they start with include/ or docs/. Zero means "absent" unless a comment says
 * otherwise.
 *
 * Where the contracts live (REPLAY_API §1.1 named them estimate_formulas.h / estimate_glue.h):
 *   est_plan_1d, est_recipe_print -> estimate_1d.h (no library header);
 *   est_build_1d, est_build_1d_ex -> estimate_build.h (after #include "vfft.c", REPLAY_API §1.2).
 */
#ifndef EST_RECIPE_H
#define EST_RECIPE_H

#include <stddef.h>

struct vfft_plan_s; /* = vfft_plan (include/vfft.h:174); kept opaque so the formulas need no library */

/* ---------------------------------------------------------------- the machine (sketch §2) */
/* Read ONCE at start-up by the tool and passed in: est_plan_1d never calls the library, so it can
 * also be fed another machine's sizes (e.g. to compare with that machine's store). The first
 * vfft_cpu_cache() call moves the thread to a P-core and back (cpu_cache.h:495-505): never make it
 * inside a create. Do not set VFFT_IDENTITY_PROBE (cpu_identity.h:39-49). */
typedef struct est_machine_s
{
    int  lanes; /* VW, 4 or 8: VFFT_IL_VW of the build (il/isa/il_isa.h:37-46; 4 = avx2, 8 = avx512).
                 * Sets the ZTURN-T floor 16 / 64 (policy_il.h:63-67), the 8-lane chain ends
                 * (ztt.h:356-364), and no zrp / ZTT-r at 8 lanes (zttr.h:164-172).
                 * [open decision 7, default] the 8-lane rules are written as the sketch specifies:
                 * legal, fitted to no row. */
    long l1d;   /* bytes: vfft_cpu_l1d_bytes() = vfft_cpu_cache()->l1d_used (cpu_cache.h:116,560).
                 * Read on a P-core on a hybrid CPU (cpu_cache.h:89-97, _vfft_cpu_pcore_enter :342);
                 * 32 KiB when that read fails (VFFT_L1D_FALLBACK_BYTES, :88). Gives
                 * s1 = floor(log2(l1d/16)), W_lo = 2^(s1-1), W_hi = 2^s1 (sketch §3), the serial
                 * bound 16*N_c <= 2*l1d (§7), the flat and zrf tiles (§4.4, §6), the Rader bound
                 * l1d/16 (§4.6).
                 * [open decision 3, default] a ZTURN-T tile the formula puts off the planner's raced
                 * ladder {1024, 2048} (dp_planner_il.h:1455) is clamped onto it. */
    long l2;    /* bytes: vfft_cpu_l2_bytes() = l2_used (cpu_cache.h:120,564), same P-core read;
                 * 1 MiB on a failed read (VFFT_L2_FALLBACK_BYTES, :113). Used only by the four-step
                 * band width fs_wl (sketch §4.5; il2d_tier.h:2000-2010). */
    long l3;    /* bytes: vfft_cpu_l3_bytes() = l3_seen (cpu_cache.h:122,569), shared, 0 = none
                 * reported. Used only by the threaded super-band il_mtsb (16N > L3, sketch §7). */
    int  cores; /* physical cores: vfft_topology()->ncores (cpu_topology.h:47,189). No rule reads
                 * it (sketch §2: no row relates a verdict to it); kept for the report. */
} est_machine_t;

/* ---------------------------------------------------------------- the request (the row's key) */
/* The public vfft_config_t fields (include/vfft.h:156-170) estimate takes; the values are the
 * public enums'. Layout is always interleaved.
 * [open decision 1, default] one transform only: the request has no howmany, est_build_1d creates
 * howmany = 1; batches (lane-major or contiguous) are not planned and are refused. */
typedef struct est_request_s
{
    int N;         /* the transform length = vfft_config_t.n[0]; real: the real length */
    int transform; /* VFFT_C2C 0 | VFFT_R2C 1 | VFFT_C2R 2 (vfft_transform_t, include/vfft.h:61-73);
                    * any other transform is refused */
    int order;     /* VFFT_ORDER_NATURAL 0 | VFFT_ORDER_SCRAMBLED 1 (include/vfft.h:91-96); real is
                    * natural only (vfft.c:1940-1957 refuses a scrambled real plan) -> row ord=nat|scr */
    int placement; /* VFFT_PLACEMENT_OUT 0 | VFFT_PLACEMENT_IN 1 (include/vfft.h:84-88) -> place=oop|ip */
    int nthreads;  /* T, 0 or 1 = one thread -> the row's nthreads=T. The route never depends on T
                    * (sketch §3, §7; wisdom2_oop_il.h:233-240); T only adds the threaded arm.
                    * [open decision 2, default] every T > 1 takes the T = 8 fitted rules; at
                    * T != 8 they are UNFITTED. No AUTO thread count exists here (sketch §7). */
} est_request_t;

/* ---------------------------------------------------------------- the recipe (the row's payload) */
/* Every field is named as the code names it: the c2c K=1 row (vfft_oop_il_entry_t,
 * wisdom2_oop_il.h:24-49), the prime method writer (vw2_prime_method_bank, :570), the real writers'
 * arguments (vw2_real_il_bank_*, wisdom2_real_il.h:100-682) and the zr2c child
 * (vw2_zr2c_child_t, :388-406, which maps 1:1 onto the c2c section). Row tokens in [brackets]. */
typedef struct est_recipe_s
{
    char eng[8];          /* [eng] the row's engine: "k1" for c2c; zrm | zrp | zttr | zr2c | zfsr | zrf | zrb */

    /* ---- c2c K=1 route; ALSO the zr2c child at N/2 (ch.* in brackets) ----------------------- */
    int  k1_il_route;     /* [il_route] VFFT_K1_IL_* (common/abi/route_ids.h:40-83): MONO 3 mono,
                           * 2P_PURE 5 2p, CHAIN3 6 chain3, PRIME 7 prime, FLAT 8 flat, ZTT 9 ztt,
                           * FS 10 fs                                                    (ch.route) */
    int  il_R1, il_R2;    /* [il_pair] pair: mid R1 x leaf R2 (sketch §4.3); FS: the ONE-THREAD split
                           * N1 x N2 (§4.5; at T > 1 too, §7).
                           * [open decision 6, default] in place reuses the out-of-place split.
                           *                                                         (ch.R1, ch.R2) */
    int  il_c3[3];        /* [il_chain] chain3 {R2 leaf, A, B} (§4.3)                       (ch.c3) */
    int  il_fl[10];       /* [il_flat] the flat DIT chain, leaf first, 2..10 stages
                           * (VFFT_ILFD_MAX_K, il_flatdit.h:29; §4.4)                       (ch.fl) */
    int  il_fl_n;         /*                                                              (ch.fl_n) */
    char il_flf[24];      /* [il_forms] per-stage letters t|m|n|o for stages 1..K-1, '.'-joined;
                           * "" = the create's defaults (§4.4)                             (ch.flf) */
    int  il_tw;           /* [il_tw] flat: the stage span R_s*D_s; ZTURN-T: the tile; 0 = untiled
                           * (§4.1, §4.4; open decision 3 applies to ZTURN-T)               (ch.tw) */
    int  il_rd[8];        /* [il_rd] the flat Rader stage's inner. ALWAYS EMPTY: estimate emits no
                           * Rader stage (§4.4), and the out-of-place replay ignores it
                           * (c2c_oop_create_il.h:224-231). The zr2c row cannot carry it.       */
    int  il_rd_n, il_rdw; /* [il_rd count, il_rdw] ALWAYS 0, as above                           */
    int  il_zt[7];        /* [il_ztt] ZTURN-T chain, 2..7 stages (§4.1-4.2). FS: the super-band
                           * chain (dp_planner_il.h:2250), written il_sb at T = 1 (never emitted,
                           * §4.5) and il_mtsb at T > 1 (§7; written as il_mtsb=0 when empty) (ch.zt) */
    int  il_zt_n;         /*                                                              (ch.zt_n) */
    int  il_kv;           /* [il_kv] forward forms, ALWAYS written (the glue sets il_kv_raced = 1):
                           * mono form (0); pair mid | leaf<<4; chain3 A | B<<4 | leaf<<8 (table
                           * 5.1); FS form 0 transpose sweep | 1 super-band; else 0          (ch.kv) */
    int  il_bkv;          /* [il_kv on the dir=bwd row] backward forms from table 5.1 (§4.3): pair
                           * mid | leaf<<4, chain3 leaf<<8 only. c2c: the glue banks it as the
                           * dir=bwd sibling row (vw2_oop_rec_k1_bwd, wisdom2_oop_il.h:359) when
                           * nonzero; zr2c: the only forms a c2r child runs [il_bkv]       (ch.bkv) */
    int  il_mt;           /* [il_mt] T > 1 only. Flat / ZTT / prime: the arm 0 serial, 1 BLOCKS,
                           * 2 TILES, written on every T > 1 row of those routes (k1_commit.h:1317,
                           * 1397,1448). FS: N1_T, the threaded split's column side (§7). zr2c:
                           * the prime child's arm, 0 = absent = serial                    (ch.pmt) */
    int  il_mt_tw;        /* [il_mt_tw] T > 1 flat arm 2: its width (= il_tw, §7)                   */
    char fs[512];         /* the ONE four-step child a recipe can hold -- FS's own, the prime or
                           * zrb kind-4 inner's, zfsr's, or a zr2c FS child's: the 2D row's tokens
                           * "fs_chain=.. fs_wl=.. fs_tf=.. fs_ro=0 fs_sw=0 fs_rbk=8 fs_turn=0
                           * fs_csk=0 fs_row_*=.. [fs_row_bwd_*=..]", space-joined (§4.5, table 5.7);
                           * at T > 1 also fs_cmt fs_mtarm fs_msw.
                           * [open decision 4, default] at T > 1 it is the THREADED split's own
                           * child (N1_T x N2_T at T, §7; k1_commit.h:1538-1552,1595-1620)   (ch.fs) */
    struct                /* route PRIME: the method row, vw2_prime_method_bank(st, N, method,
                           * in_kind, in_shape, in_tw) (wisdom2_oop_il.h:570), key ord=scr place=ip
                           * at T = 1 (:545); the route row then says il_route=prime il_kv=0.
                           * [open decision 5, default] never emits Bluestein's smooth lengths, the
                           * Rader-stage flat inner (kind 5), nor Rader on a four-step inner below
                           * 2^23 (sketch §12.5); the primes §4.6 refuses stay refused.        */
    {
        int  method;      /* [eng=rader|bluestein] 1 rader, 2 bluestein (§4.6)              (ch.pm)  */
        char in_kind[8];  /* [in] 2p | 3p | ztt | fs                                         (ch.pin) */
        char in_shape[64];/* [in_sh] R1.R2 | R2.A.B | the ZTT chain | N1.N2                  (ch.psh) */
        int  in_tw;       /* [in_tw] the ZTT inner's tile, 0 = untiled                       (ch.ptw) */
    } prime;

    /* ---- real: one row per (r2c|c2r, N, placement, T), ord=nat (wisdom2_real_il.h:1-44) ------- */
    struct { int R1, R2, form; } zrp;    /* [pair=R1.R2 leaf=n1t|r2z] form 0 A (n1t), 1 B (r2z);
                                          * 4 lanes only (§6)                                   */
    struct { int chain[8], nf; size_t tile; int stk, mt; } zttr;
                                         /* [chain= tile= stk= mt=] the chain at M = N/2, 2..8
                                          * stages; stk = 3; mt 0 = absent; 4 lanes only (§6)    */
    struct { int route, fold_mt; } zr2c; /* [route=child_oop_il|child_nat_ip fold=st|mt] route =
                                          * the CHILD's placement, 0 child_oop_il, 1 child_nat_ip
                                          * (not k1_il_route); fold_mt only on a T > 1 row (§6, §7).
                                          * The child = the c2c section above, natural, T = 1    */
    struct { int n1, n2; } zfsr;         /* [split=n1xn2] the split of N/2 (§6); child = fs     */
    struct { int chain[10], K, nomsz, tile, mt; } zrf;
                                         /* [chain= msz= tile= mt=] odd radices ascending, 2..10
                                          * (zrf.h:228); nomsz = !msz; tile = the budget in
                                          * complex, 0 = untiled; mt 2 LEVELS on a T > 1 row     */
    struct { int M; char kind[8], shape[64]; int tw, mt; } zrb;
                                         /* [m= in= in_sh= in_tw= mt=] the convolution length and
                                          * its inner, the prime row's spelling; a kind-4 inner's
                                          * child goes in fs; mt 0 = serial (§7)                 */
} est_recipe_t;

/* ---------------------------------------------------------------- the contracts */

/* est_plan_1d: the formulas. Machine + request -> the recipe, by the sketch's ordered tests
 * (§3 complex, §6 real, §7 threads) and parameter formulas (§4, table §5).
 * - Pure: reads only *m and *rq. No clock, no store, no library call, no allocation.
 * - Returns 0 with *out filled (zeroed first).
 * - Returns -1 for a refusal, *out unspecified. The reason, NUL-terminated and cut to whylen,
 *   goes in why (why may be NULL when whylen == 0). Refused:
 *   - a transform other than c2c / r2c / c2r; a scrambled real request; nthreads < 0;
 *   - N outside the sketch's bounds: c2c N < 2 or > 2^24 (§3.1); real N < 2, even N with
 *     N/2 > 2^24, an odd PRIME CELL (zrb) with (3N-1)/2 > 2^24 (§6). Odd smooth N (zrf) has no
 *     ceiling but its 10 stages (§6, after the §4.4 merges);
 *   - real even N with N/2 a power of two in 2^19..2^24 (zfsr) at T > 1: the sketch has no
 *     threaded rule for zfsr (§7) -- a gap reported, not filled;
 *   - a composite with 2N-1 > 2^24 (§3.3), the three lengths of §3.6, the primes of §4.6 with
 *     no Rader case and M > 2^24 (open decision 5);
 *   - the IN-PLACE SCRAMBLED PRIME cell at T = 1. The library would race it, because its route
 *     row and its prime method row share one key: vw2_oop_bank_k1_il refuses to overwrite a
 *     method row (wisdom2_oop_il.h:515-527, key :545-551), and the door's scan accepts only
 *     eng=k1 (wisdom2_oop_rows.h:103), so it finds no route row and races (k1_commit.h:993-1004).
 */
static inline int est_plan_1d(const est_machine_t *m, const est_request_t *rq,
                              est_recipe_t *out, char *why, size_t whylen);

/* est_build_1d: the glue. Recipe + its request -> a public plan built with NO measurement, or
 * NULL with the reason in why (as above). rq must be the request rc was planned for, on a machine
 * whose lanes equal this build's VFFT_IL_VW. Steps (REPLAY_API §1.4, §2):
 * 1. Refuse the in-place scrambled prime cell at T = 1 (see est_plan_1d), for a recipe made
 *    elsewhere.
 * 2. Clock-free pre-flight at T = 1. It calls the door's own constructor sequence and destroys
 *    what it built, and refuses a recipe that does not build as written: the door would mark such
 *    a row stale and race the cell. FS, zfsr and kind-4 inners cannot be built here (a nested
 *    create); their band, split, inner-length law (Rader M = N-1, Bluestein M >= 2N-1 and even,
 *    zrb M >= (3N-1)/2) and the child's two plans are checked without a create.
 * 3. Write the rows a race would bank (the c2c route row; the T row's il_mt / il_mt_tw / il_mtsb /
 *    fs_*; the prime method row; the dir=bwd row; or the real row) into a private IN-MEMORY store,
 *    vfft_child_store_new (wisdom2_child.h:40), then vw2_disown (wisdom2.h:1486). The disk store
 *    is never opened, read or written: cfg.wisdom is set (vfft.c:2276), wisdom_write = 0, and
 *    the create depth is raised (vfft.c:2523-2524). The glue detects a FIRST load of the default
 *    store (_def_loaded, vfft.c:496-505) or a new directory store (_vfft_api_nstores); a read of a
 *    default store the process had already loaded is not detectable. No IL nested create reaches
 *    it today (each sets .wisdom), so the guarantee rests on that, not on the detector.
 * 4. Call the library's own create, _vfft_create_cfg (vfft.c:2509), howmany = 1: its banked-verdict
 *    replay builds the handle, binds the executor, holds the pool, allocates the alias buffer.
 * 5. No-measurement proof: est_clock_reads (VFFT_CLOCK_PROBE, race_scope.h:636-642),
 *    _vfft_create_race_count (vfft.c:255) and vfft_measure_scopes() (vfft_measure.h:112) are read
 *    before the pre-flight and after the create, and the private store's row count around the
 *    create. Any change means the create measured: the plan is destroyed, NULL returned
 *    (EST_MEASURED). A row the door re-banked at an existing key ("own", the test
 *    vfft_child_store_raced makes, wisdom2_child.h:164) is compared with the row the glue wrote:
 *    the same payload is reported only; a changed payload means the plan is not the recipe, so
 *    it is destroyed and NULL returned (EST_REWRITTEN).
 * 6. Every threaded arm the recipe asked for must have taken (the mt fields, REPLAY_API §2.16):
 *    a silently serial plan is not the recipe, so it is destroyed and NULL returned. A T > 1 plan
 *    gets the caller pin the public create gives it (vfft_api.h:207-213).
 * Holds _vfft_create_lock (vfft.c:2485) from the pre-flight to the end. The private store is freed
 * before return; the caller owns the plan and frees it with vfft_destroy. The glue and the tool
 * refuse to run when a create-steering VFFT_* variable is set (REPLAY_API §1.3), matched without
 * case on Windows, where getenv ignores it. */
static inline struct vfft_plan_s *est_build_1d(const est_recipe_t *rc, const est_request_t *rq,
                                               char *why, size_t whylen);

/* est_recipe_print: the recipe in the store's own text, so it compares with a stored row line by
 * line. One line per row the glue banks: "<key> | <payload>", with no level-2 tail (ran, ns, src,
 * date: not recipe).
 * - c2c key:  t=c2c n=N q=1 ord=nat|scr place=oop|ip role=comp lay=il [nthreads=T]
 *   payload: eng=k1 il_route=.. and the route's tokens (il_pair il_chain il_flat il_forms il_tw
 *   il_ztt il_kv; T > 1 il_mt il_mt_tw il_mtsb; FS fs_*).
 * - prime: the method row first (key ord=scr place=ip, eng=rader|bluestein in= in_sh= in_tw=
 *   [fs_*]), then the route row.
 * - when il_bkv != 0, the dir=bwd row (key ord=nat place=oop dir=bwd; eng=k1 il_route il_pair
 *   il_kv [il_chain]).
 * - real key:  t=r2c|c2r n=N q=1 ord=nat place=oop|ip [nthreads=T] (vw2_real_il_key,
 *   wisdom2_real_il.h:60-68); payload eng=.. and the bracketed tokens above (zr2c: also the
 *   child's il_* tokens, il_bkv, and il_prime il_prime_in il_prime_sh il_prime_tw for a prime child).
 * Returns the length it wrote, or would have written (snprintf's contract); buf is always
 * NUL-terminated when cap > 0. */
static inline int est_recipe_print(const est_recipe_t *rc, const est_request_t *rq,
                                   char *buf, size_t cap);

#endif /* EST_RECIPE_H */
