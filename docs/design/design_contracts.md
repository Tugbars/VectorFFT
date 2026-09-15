# Design contracts — the owner's laws for the planner (2026-09-09)

These are rulings, not designs. They bind every create path, every
calibrator and every wisdom row. When code contradicts them, the code is the
defect; when a pool would be empty under them, the cell refuses — it is never
filled from another contract.

## 1. A request names a contract; only that contract is raced

A request is (transform, layout, order, placement, N, K, T). The planner
races engines that satisfy that contract and nothing else. An interleaved
natural-order request races natural-order interleaved plans only.

## 2. SPLIT and INTERLEAVED are two libraries

They never touch: not at create, not in the calibrator, not in the store's
rows. An interleaved request never builds, races, banks or reads a split
plan; a split request never touches an interleaved one. There is no
"both axes" design, and there will be none until the whole library is
finished — which is far away.

Known violations at the time of writing — REMOVED 2026-09-09 (evening):

- `c2c_oop_create.h` built a split plan (`psp`) for interleaved K=1
  requests whenever a split route resolved and committed it (`hk->k1sp`).
  Now the K=1 door resolves, builds and commits ONE layout's axis
  (`want_il`): an interleaved request has no split route, pair or plan, a
  split request has no interleaved one.
- `calibrate_k1` -> `vfft_sp_dp_plan_and_bank` drove the split race and the
  interleaved race in one call per cell, and `vfft_il_dp_emit_wisdom` wrote
  the lay=split row beside the interleaved rows. Now: `calibrate_k1_il`
  (dp_planner_il.h `vfft_il_dp_plan_and_bank(ctx, st, N, verbose)`, lay=il
  rows only) and `calibrate_k1_split` (dp_planner_split_oop.h
  `vfft_sp_dp_plan_and_bank(reg, wisdir, N, rigor, verbose)` +
  `vfft_sp_dp_emit_wisdom`, the lay=split row only). Smoke-verified on a
  scratch store: each rewrote exactly its own rows. `calibrate_k1.c` is
  deleted; the research scripts under `docs/research/.../probes/ZT/` that
  named it are historical.

## 3. Output order is a contract

- DEFAULT order is natural order.
- A NATURAL request races natural writers only.
- A SCRAMBLED request races scrambled writers only, and banks only those.
- Order classes never share candidates. A natural-writing engine in a
  scrambled pool is wrong (stage 2 of the sunset plan, 2026-09-09, did this
  and is reverted); a scrambled writer in a natural pool is unusable output.

## 4. Bands for 1D C2C interleaved K=1

| N | engines raced |
| --- | --- |
| <= 64 | the solo kernels (and the pairs where both exist) |
| 128..1024 | the Bailey pairs and ZTURN-T |
| 2048..262144 | ZTURN-T ALONE — no Bailey (the plans are too big to be efficient; the small L1-resident execution is unbeatable), no cascade |
| odd factors | the flat DIT (pure odd and every odd-factor N < 2048); the 2^a * odd cells at N >= 2048 with the odd part over {3, 5, 7, 9, 15} are ZTURN-T's ODD BAND (built 2026-09-15, docs/design/ztt_odd_design.md: the odd radix as a mid, staged; the natural pool keeps chain3 and the pairs until the band is banked; the scrambled pool is the plain class alone) |
| 262144 | the NATURAL cell races ZTURN-T against the four-step's splits and ZTURN-T keeps it (raced 2026-09-15 on a quiet machine: 682 us serial, 1.6x over MKL at T=8); the SCRAMBLED cell is the plain ZTURN-T's alone (8b) — the four-step's scrambled class begins at 524288 |
| 524288..4194304 (pow2) | the FOUR-STEP ALONE (owner 2026-09-15: "our Bailey engine is the best solution for this and MKL's code also shows that they are using Bailey for 256k and above"; docs/design/k1_fourstep_design.md): N = N1 x N2 on the 2D interleaved tier with the inter-pass twiddle fused into the row pass, every split with both sides in {256..4096} an arm, raced per cell at one thread and again per thread count; scrambled = the plane as it stands, natural = one blocked AVX2 streaming transpose; route 10, `il_route=fs il_pair=N1.N2`, per-T `il_mt=N1 il_mt_t=T`. Measured 2026-09-15: 1.0-1.44x over MKL at T=1, 1.15-1.58x at T=8, parity at 4194304 T=8 (v1_0_results.md). Above 2^22 and the 2^a*odd cells above 262144: not served |

Owner, 2026-09-09 evening, on the upper band: the ZTURN-T-alone ruling is
provisional above ~65536 — "we need to race cells above 65k, Bailey vs
ZTURN-T, and only then can we derive an opinion." That race is in the
pipeline (section 8); until it runs, nothing is banked above 65536.

The ZTURN-S cascade (zturn.h, zsplit.h, zturn_mt.h, its codelets, doors,
route axis, calibrators and wisdom tokens) is deleted — done 2026-09-15.

## 5. ZTURN-T is under development; refusal is allowed — for a MISSING
## ENGINE only. An uncalibrated cell is never refused and never falls back.

Owner, 2026-09-09 evening: "when user uses the front API, vfft.h, this
library should look for a wisdom entry, and if there is none it should
calibrate. NO FALLBACKS. NEVER. Always either race & bank the winner unless
it exists in wisdom." — "the user asks for a certain N FFT with a certain
layout and order and oop/in-place structure, the library checks for a
wisdom entry for that, if it doesn't exist, it starts a race and banks the
winner and then executes it."

So: a cell WITH an engine of its contract but no row races at create (the
K=1 interleaved race runs at every interleaved miss, the pow2 band
included, via `_k1_il_plan_race`); the prime engine is a route for its own
N, never a fallback for a power of two. Refusal is only for a cell whose
contract has NO engine yet:

- SCRAMBLED at power-of-two N >= 2048 has no engine (the scrambled ZTURN-T
  class is future work: the sequential-store ingest with strided-run mids
  and its own tile law).
- T > 1 at N >= 2048 has no engine (ZTURN-T's MT arm is future work).
- 2^a * odd cells at N >= 2048 (today the odd cascade) get a ZTURN-T engine
  first; the cascade is deleted only after that engine serves them (owner,
  2026-09-09 evening: "we will make sure odd mixed with pow2s have a zt-t
  engine and only then we will delete cascade"). DONE 2026-09-15: the odd
  band is ZTURN-T's at every door; the cascade's deletion waits on the
  band's calibration (the owner's run).

## 6. Within a contract, race everything the engine admits

Chains, tile widths, kernel forms, backward forms: each is a candidate, the
race decides, the row banks the winner. A ladder defines a pool; it is never
a rule that picks a value.

## 7. Recorded corrections, so they are not re-derived

- "Natural output is a legal scrambled answer" — no. The header's
  "order-agnostic" wording was my reading; the owner authors the contract.
- "The store needs both layouts' rows from one calibrator" — no.
- "The prime path's inner may take the scrambled row" — no; a prime
  transform's inner FFT at the padded length takes the NATURAL row at M.
- "Keep the cascade as the scrambled engine" — pressure-tested and dropped:
  the cascade wins no cell in 2048..16384.
- "A cold pow2 cell in the band refuses until its row is banked" — no
  (owner, 2026-09-09 evening): it RACES and banks; refusal is for a missing
  engine, not a missing row. The 32768 natural cell had been served by the
  prime engine (Bluestein) because the create-time race was fenced out of
  the band — a fallback, fixed the same evening.
- "Retire the radix-4-first ZTURN-T chains, they lost every cell 2048..32768"
  — no (owner, 2026-09-09 evening): "this can be platform dependent. we
  can't rule out radix 4 in the beginning as loser." The ingest-radix
  verdict is this host's measurement, not a law; the registry keeps every
  {4,8} chain.
- "Forms flip per host (Zen 4 banked M-128 at 1024)" — the Zen 4 store
  predates ZTURN-T; its 1024 pair row exists only because ZTURN-T was not
  raced there. The cross-host forms evidence reduces to one slot at 512.
- Scope (owner, 2026-09-09 evening): pure pow2 only for now; the odd-N
  machinery (chain3, flat DIT, 2^a * odd cells) is not touched until its
  turn.

## 8. Pipeline (owner-ordered, 2026-09-09 evening)

1. ZTURN-T for the 2^a * odd cells at N >= 2048 (odd factors in the chain),
   gated and raced like the pow2 cells; then the cascade deletion.
   Design: `docs/design/ztt_odd_design.md` (2026-09-14) — odd radix as a
   mid, staged executor per section 10, the cascade's exact cell set.
   BUILT, WIRED AND GATED 2026-09-15 (the doc's build order 1-5). Owner,
   2026-09-15: the cascade is NOT kept for its one remaining edge (scrambled
   OOP forward at L2 sizes); ZTURN-T gets its THREADED arm next, and then
   the cascade is deleted whole — codelets and every wiring — "we need a
   clean library, not a piled-up history of everything we developed before".
   DELETED 2026-09-15 (ztt_odd_design.md build order 8; the MT method kept
   in cascade_mt_method.md). T > 1 at N >= 2048 has no engine until
   ZTURN-T's threaded arm (section 5: a missing engine refuses).
   The threaded arm's design: `docs/design/ztt_mt_design.md` (2026-09-15) —
   the staged walk sectioned, raced and banked per T on the ztt row.
2. Above ~65536: race Bailey (four-step) against ZTURN-T at each cell, paced,
   and only then rule the upper band and the ceiling.
3. DONE 2026-09-09: the tile ladder is 16 KB and 32 KB (untiled stays the
   datum), ruled from the measured race at 2048..32768 — see
   `zturn_t_2048plus_plan.md` step 2. The overflow gate's census is
   restamped (37/90/127/175/247/354/480); it will move again when the
   scrambled pool loses the natural writers (S2 revert).
3a. DONE 2026-09-09 (evening, item 1 of the agreed order): the three
   ruled-but-open law violations — the public header's order text
   (`include/vfft.h`: DEFAULT = natural, SCRAMBLED = scrambled writers
   only, matched-roundtrip decode, refusal where no scrambled writer
   exists), the split plan built for interleaved requests (section 2), the
   coupled calibrator (section 2). Gates: k1_pow2, ztt, api_matrix (61
   cells, both layouts), vfft_natural_front — the last RESTAMPED to the
   band law (it asserted a MEASURE create must race the cascade and that
   the cascade must win from 8192 up); a stale in-place cascade row at
   16384 left the store on the way.
3b. DONE 2026-09-09: no Bailey pair in the pow2 pools at 2048 and above
   (`_il_dp_enumerate_natural_engines` returns ZTURN-T alone there); census
   74 / 123 at 2048 / 4096, overflow gate restamped and green.
3c. DONE 2026-09-09: the pow2 PAIR-POOL SUNSET (the 2026-08-11 pool-sunset
   policy applied, owner's slot rulings): no radix-64 slot at a power of two;
   the radix-8 and radix-16 slots race the tangent kernel alone (classic,
   blocked 4.4 and M-128 gone from the pow2 pools); radix 32 keeps its four
   forms ("all can stay"); radix 4 has one form. Enumerator only
   (`_il_dp_enumerate_natural_engines`, pow2 cells): the superseded kernels
   stay in the resolvers for the backward side (no tangent twins, item 4)
   and for the odd cells' banked rows (chain3 at 1536 banks the 4.4 leaf,
   3072 the M-128 mid) until the odd machinery's turn; file deletion follows
   then. Pools: 128/256/512 = 13 each (were 25/36/41), 1024 = 23 (was 37);
   overflow gate restamped; k1_pow2_gate ALL PASS (its store reader made
   lay=il-first — it had labelled 2048/4096 with the stale lay-less row).
3d. DONE 2026-09-09: S2 REVERTED at pow2 — the natural engines are out of
   the scrambled pools at 2048 and above, the explicit-SCRAMBLED door takes
   the cascade (the only scrambled writer there), the S2 rows (ord=scr K=1
   ztt rows and scrambled mode rows, 2048..32768) are out of the store; the
   LEGACY zsplit engine is out of the pow2 scrambled pools ("should not be
   part of the runs, it doesn't win anything" — never banked on any host).
   Scrambled pools: 48/77/113/166/246/340 at 2048..65536 = the ZTURN-S
   cascade alone (chains x stf/stf2 x its tile ladder). Gates: census,
   overflow, k1_pow2 (scrambled 2048/4096 = "comb served") ALL PASS.
   Still to do at pow2 >= 2048: the NATURAL and DEFAULT doors' cascade arms
   (the natord cascade race behind `VFFT_NO_NAT_ZCASC`, the DEFAULT door's
   scrambled-cascade race) — both violate section 4 on a cold cell.
3e. DONE 2026-09-09: NO CASCADE RACE ARM at any pow2 cell ("no cascade
   race arm please, eliminate"): `vfft_ztt_band(N)` in ztt.h (pow2,
   16..VFFT_ZTT_MAX_N) gates the natural OOP door, the DEFAULT OOP door
   and the in-place door in c2c_oop_create.h / c2c_ip_create.h — no cascade
   candidate is built or raced there; a stale in-place ZCASC row in the band
   is not a verdict (the K=1 engine builds and banks as ILP). In place the gate is for
   NATURAL and DEFAULT (DEFAULT = natural at pow2, `_ip_order_is_nat`); the
   explicit SCRAMBLED in-place cell keeps the cascade, its only scrambled
   writer. Outside the
   band (odd factors, above the ceiling) the doors are unchanged. FOUND on
   the way: the natural door had banked the cascade at 512 natural
   (mode=zcasc, 400 ns by the door's clock, over a 300 ns pair) and was
   serving it; that row and the cascade's four sub-2048 comp recipes are out
   of the store. Gates:
   k1_pow2_gate ALL PASS, ztt_gate ALL PASS (16..16384 fwd/bwd/in place
   bitwise the direct engine).
4. DONE 2026-09-11 (item 2 of the agreed order): the tangent BACKWARD twins
   at radix 8 and 16 — `radix{8,16}_z_t2ttan_bwd_avx2` (the turned-store
   mid, the pair's backward stage-1 kind) and `radix{8,16}_z_n1tan_bwd_avx2`
   (the leaf) — emitted with the forward recipes plus --cil-bwd, bit-identical
   (radix 8) / 5e-17 (radix 16) to the classic backward kernels
   (`benches/tangent_bwd_gate.c`), backward variant 3 in il2p.h's resolvers,
   raced by `benches/bwd_forms_race.c` for the shipped pairs (3 stable
   repeats) and banked: every slot offered a twin took it (32: -2.5%, 64:
   -3.0%, 256: -5.0% both slots, 512: -1.6% mid), then the radix-32
   backward LEAF twin (`radix32_z_n1bw32_bwd_avx2`: the tangent interior on
   the blocked 2.16 split without the wing combine, which the emitter
   allows forward only — VFFT_CX_W32TG): 128 -1.9%, 512 both slots -4.4%.
   Every backward slot of every shipped pow2 pair now runs a tangent kernel.
   The backward WING combine was then built in the emitter (owner: "let's
   do it"): `butterfly_pair_w32 ~sign` + the mirrored CRotPI ROTFMA fold,
   bit-exact, the four shipped forward wing kernels re-emit byte-identical
   in body. Its radix-32 backward leaf gated to 1e-16 and LOST the race to
   the plain tangent 2.16 leaf at both cells (128: 71.4 vs 70.7; 512: 347
   vs 344, 5/5) — retired by measurement; the emitter capability stays.
   LESSON: a twin of the wrong kind (plain-store t2 where the pair runs the
   turned-store t2t) builds, fails the planner's correctness gate and is
   silently "not an arm" — the race printed two arms and nothing said why.
   Was: COVERAGE GAP, not a search-space problem (owner, 2026-09-09 evening):
   the tangent kernels have no BACKWARD twins (radix 16 and 32; the backward
   variant resolvers in il2p.h offer only the 2.16 / 4.8 turned-store
   splits at 32 and 64, nothing at 16), so every backward row from 32 to
   1024 banked the default forms by default, not by verdict. Build the
   backward tangent twins; do not widen the pools to compensate.

The pow2 state after all of the above (what was cut and why, the remaining
pools per cell, the kernel kinds still bound) is `pow2_race_pools.md`,
gitignored beside this file.

5. DONE 2026-09-11 (owner: "we should add a gate later then?"). (a) The
   planner NAMES every refusal: `_il_dp_bench_dir(..., const char **why)`
   distinguishes "no such kernel (build refused)" from "BUILT but WRONG:
   backward roundtrip err ... > 1e-11" and from the executor refusals, the
   backward race prints it under verbose, and `_il_dp_run_once` returns -1
   (no kernel) vs -2 (built, executor refused) so the forward candidate
   loop's silent `continue` says which. (b) the RESOLVER INVARIANT gate: every
   kernel a form resolver hands back for a slot must be correct in that
   slot. It is NOT a bench (owner, 2026-09-12): the walk is
   `core/support/slot_check.h`, the slot list and probe are
   `core/planning/il_slot_probe.h`, and `benches/form_slot_gate.c` only
   parses arguments and calls in — the calibrators' rule. 22 arrangements, 238 slot-kernels
   built+ran+gated, 690 absent, 0 wrong.
   Two findings, both recorded because they change how gates get written:
   - The gate's FIRST cut read each cell's banked pair and gated only that.
     It PASSED with the original defect deliberately re-injected: no shipped
     pair carries a radix-8 MID (32/64/128 are 4xR, 256/512 are 16xR), so
     that resolver entry was never exercised. A resolver is indexed by
     (radix, variant, role), so a gate over the SHIPPED CONFIGURATION proves
     nothing about it — the gate now enumerates every legal arrangement at
     each N, which puts every pair radix in both roles.
   - A wrong-kind kernel is not merely wrong, it is MEMORY-UNSAFE: the
     plain-store t2 indexes zout[o*OLs+k] while the turned-store slot passes
     OLs=R, so it writes past the plan's mid buffer and the process dies
     (the injected run died at N=32 8x4). The gate announces each
     arrangement on stderr before running it, so a death still names the
     culprit. OPEN for the owner: the same kernels are raced at CREATE time
     on a cold cell, so a bad resolver entry would corrupt a caller's heap —
     guard slack in the planner arena would contain it. Not done unasked.

## 8b. The SCRAMBLED ZTURN-T class — the owner's four rulings (2026-09-12)

The scrambled writer for pow2 interleaved K=1, replacing the cascade.

1. CONSTRUCTION: Sande-Tukey with NEW kinds. Scrambled means the
   permutation is never performed; ZTURN-T carries it at the ingest, so the
   sibling drops it there. Contiguous loads in, butterflies with the twiddle
   applied AFTER (post-twiddle), contiguous stores out, and the output lands
   in whatever digit order that yields. Hosted in ZTURN-T's machinery: one
   fused driver per (cell, chain, direction, buffer mode), baked
   quarter-wave twiddle streams, the raced 16/32 KB tile. New kinds: a
   sequential ingest and a sequential terminator, both directions; the mid's
   post-twiddle math already exists (the cascade's msg, which ZTURN-T's tmg
   was derived from).
2. COVERAGE: every pow2 cell from 16 to the ceiling, the same registry the
   natural class walks. This also closes the sub-2048 scrambled cells, which
   today serve natural output against section 3.
3. PERMUTATION: its own, whatever the chain yields. No API reports the order
   and the only supported decode is the matched roundtrip, so the terminator
   is unconstrained and the chain races freely. It need not reproduce the
   cascade's comb.
4. PLACEMENT: out of place first, in place immediately after. With no
   permutation the butterflies can run in the caller's buffer, so this class
   is IN-PLACE NATIVE — unlike the natural class, which needed a shadow
   plane because a permuting ingest cannot read and write one buffer.

KNOWN RISK, stated before building: ZTURN-T's plane is run-contiguous
BECAUSE the permutation was absorbed at the ingest. Without it the mids'
access pattern has to be re-derived, and the tile law with it.

CORRECTION TO THE CASE I PUT WHEN ASKING (2026-09-12, same day, before any
code): I told the owner the saving was the ingest's gather, "56% of the
transform at 128". That conflated two things. ZTURN-T's ingest reads legs at
stride N/R0 because decimation in time TAKES every M-th element; the
permutation only decides where the resulting run is STORED. Dropping the
permutation alone was already measured on 2026-09-09 (probes/ZT/
zt_plane_skew.c, ZT_IDENT_RB=1 makes the ingest's stores sequential) and
bounded at 0-4%. The ruling stands for a DIFFERENT reason: Sande-Tukey is
another network, not ZTURN-T with a flag flipped — its first stage streams R0
CONTIGUOUS blocks rather than gathering R0 strided legs, so every pass loads
and stores contiguously. That upside is real (a vector built from four
strided doubles costs ~4 loads + 3 inserts against one contiguous load) but
UNMEASURED. Therefore: build a SPIKE first — the two new kinds and one
cell's driver, gated, then measured against the served plan at a few cells —
and only build the class if the spike pays.

## 8c. MEASURED BEFORE BUILDING (2026-09-12): the scrambled class does not
## pay, and the premise I asked the owner to rule on was wrong twice

Three facts, all from the tree and the clock, none from reasoning:

1. ZTURN-T IS ALREADY CONTIGUOUS AT BOTH ENDS. The ingest's loads are four
   `_mm256_loadu_pd` from `zin[2*(j*Ls + k)]` — R0 contiguous streams, no
   gather, no element assembly (radix4_z_t0tp_avx2.c). The terminator stores
   R contiguous streams at `leg*OLs + k`. Natural order comes from the
   ADDRESSING, not from any reordering pass. So "Sande-Tukey streams
   contiguously and DIT gathers" — the case I put when asking for the ruling
   — is false for this engine.
2. THE PERMUTATION IS LOAD-BEARING FOR THE MATH, not just for the output
   order. Dropping it (rb[c] = c) does not yield a scrambled DFT: the
   roundtrip goes from 3e-16 to 1.1-1.5, and the output is not even a
   permutation of the natural spectrum (the sorted magnitude multisets
   differ by 9e-2..3.9e-1). probes/ZT/zt_scr_ident.c, N=128/1024/4096/16384.
   So the cheap "scrambled = a plan flag" path does not exist.
3. WHAT THE PERMUTATION COSTS is bounded by running the same network with
   sequential run placement (probes/ZT/zt_plane_skew.c, ZT_IDENT_RB=1):
   128 1.4%, 1024 0.1%, 4096 0.9%, 16384 6.7% (single runs, unpaced — a
   BOUND, not a verdict).

Consequence: a scrambled writer can recover at most a few percent, and the
only scrambled network that exists (the cascade) measured 2.8-20% SLOWER
than ZTURN-T's natural output at 2048..16384. So no scrambled engine we
could build is likely to beat simply serving the natural plan.

RULED (owner, 2026-09-13), superseding the paragraph that stood here: a
scrambled request is served by scrambled writers ONLY, at every cell —
"natorder and scrambled are contracts not optimization angles". Serving the
natural plan to a scrambled caller is not a rule of this library, whatever
the speeds. The scrambled ZTURN-T class was then BUILT (the plain
Sande-Tukey schedule, in place; docs/design/ztt_scrambled_design.md) and
MEASURED (probes/ZT/zt_scr_spike_results.md): in place it wins 1-32% over
natural at every cell from 2048 up, out of place it wins above L2 and its
forward loses 5-22% at L2-resident sizes, and below 2048 it loses ~11% — and
it is served regardless, because the order is a contract. The 8c bound
above was a bound on the scatter only; the class's real structure (one
sweep fewer, one transpose more) is in the design doc.

## 9. Racing law: coverage gaps are built, never raced around

A pool is for deciding between engines that all satisfy the contract. It is
not a substitute for a missing kernel: when one side of a cell has no
candidate of a kind the other side has (a backward twin, an order class, a
placement), the fix is the kernel, and the pool stays exactly as large as
the set of things that can win. Candidates that have lost every cell on
every calibrated host are retired by that measurement (the pool sunset
policy), never kept "in case". Owner, 2026-09-09: "we are trying to fix a
coverage gap by making our search space very large. that's a big mistake";
"we are racing nonsense mostly."

## 10. Generated code: stage kernels are the product, fused codelets are the pow2 solution's form

Owner, 2026-09-14: "as a library provider, I can't give 'useless' files to
the people"; "the other non-fused codelets can be used as different stages
for different mixed etc FFT solutions. Fused is only for pow2 strictly for
their own solution"; "so we need both, but this should be documented, put
the fused codelets to a new folder ... and wherever they are used in the
core's logic, the header should mention what they are."

The law: two kinds of generated code, both kept, each with its role stated
where it is used.
- The STAGE KERNELS (`codelets/zil/avx2/boundary_split/radix{4,8}_z_*`) are
  the product. One stage each, composable: 2^a·odd chains, the 2D/3D pow2
  passes and both order classes' backwards are built from them.
- The FUSED CODELETS (`generator/generated/fused_codelets/`, README there)
  are one whole-transform function per pow2 cell with the stage bodies
  inlined and literal trip counts — the pow2 ZTURN-T solution's executable
  form and ONLY that. Nothing else may bind one; none can be recombined.
  Measured worth: 4-12% below 2048, 0-3% above; cost: ~7 MB of generated
  text, one file per (family, N) so a kernel change rebuilds in a minute.
- Every generated family says in its own header and at every core use site
  what it is and why it exists; a build-time convenience must never look
  like the product.
