# Interleaved coverage plan — sub-2048 routing, in-place tiers, small-K batching

Successor to `cascade_natural_inplace_plan.md` (B1–B6 + Phase C all closed
2026-08-03: natural + in-place SHIPPED for the K=1 IL cascade tier,
parity-or-ahead vs MKL in its own best discipline; DIT refuted by the race).
This plan closes the remaining interleaved holes, cheapest-first.

**The map today (verified in-tree 2026-08-03):**

| surface | K=1 <2048 | K=1 ≥2048 | K=2..4 (any N) |
|---|---|---|---|
| OOP, natural/default | ✅ native (mono/il2p/il3p/prime) | ✅ cascade (stfn) / il-tiers | 🔴 convert fallback |
| OOP, explicit SCRAMBLED | 🔴 convert (routing anomaly — vfft.c:3370 excludes the K=1 engine) | ✅ cascade | 🔴 convert fallback |
| IN-PLACE, natural | 🔴 tape+convert | ✅ ZCASC cascade | 🔴 convert fallback |
| IN-PLACE, scrambled | 🔴 convert (mono refuses; il2p not attached) | ✅ cascade | 🔴 convert fallback |

---

## Phase A — sub-2048 explicit-SCRAMBLED routing fix (the one-liner)

Asking for the CHEAPER contract currently gets the SLOWER route: `DEFAULT`
and `NATURAL` reach the native il2p/il3p/mono engines, explicit `SCRAMBLED`
falls to the convert fallback because the K=1 engine gate excludes it.
The scrambled contract is "any self-consistent permutation; a route's own
bwd consumes its own fwd comb" — **the identity permutation qualifies**, so
the natural-native engines may legally serve an explicit-SCRAMBLED request.
No cheaper genuinely-scrambled pipeline exists to build below 2048 (il2p
pays no reorder that scrambling could skip) — routing IS the whole fix.

- [x] **A1 ✅ 2026-08-03** — gate widened with a no-cascade guard
  (`order != SCRAMBLED || (!zs_pending && !zt_pending)`): sub-2048 gets the
  native engine; ≥2048 keeps the cascade WITHOUT building a dead-weight k1
  engine beside it (and gains a better-than-convert fallback if a cascade
  create ever fails). Identity-serving documented at the order fence.
- [x] **A2 ✅** — `vfft_k1scr_gate.c` ALL PASS: scr==nat memcmp-EXACT at
  128–1024 (the identity IS the route proof — the old convert route emits
  a permuted MODEB comb and cannot pass), natural anchored to naive DFT
  ~1e-15, matched roundtrips, speed 0.95–1.05×.
- [x] **A3 ✅** — same gate: 4096 SCRAMBLED emits a REAL permutation
  (cascade comb) with a clean matched roundtrip. Structural note recorded:
  execute prefers an attached cascade over `k1_on`, so the change's only
  possible ≥2048 failure mode was dead weight, never wrong output.

## Phase B — sub-2048 IN-PLACE interleaved tiers (the ZCASC template, one tier down)

il2p/il3p are already alias-safety-gated (A3 of the previous plan); mono is
structurally refused (`__restrict__`). What's missing is ONLY routing: the
in-place branch never attaches the IL engines, so both orders pay
tape/convert below 2048.

- [x] **B1 ✅ 2026-08-03** — recon (in code comments rather than a page,
  scope was small): the in-place sub-2048 incumbent is
  `_exec_c2c_interleaved` — deinterleave into split work planes, proto
  engine (+tape when natural), reinterleave; at K=1 the il_me A/B can even
  pick a PADDED Kp=8 plan (7 zero lanes computed for SIMD width). il2p and
  il3p are two-stage through internal scratch — zout written only by the
  last stage — and were already alias-gated (A3 record); il_prime aliasing
  is ungated, so prime cells stay with the incumbent.
- [x] **B2 ✅** — `VFFT_NAT_ILP = 7` in the verdict enum; shared
  candidate helper `_k1_il_candidate` (kind-3 pair else the balanced-pair
  heuristic — MIRRORS the OOP K=1 block's IL search, cross-referenced both
  sites; il3p chain fallback; mono excluded). Natural block: candidate at
  N<2048, consume short-circuit with `replay ILP` log, end-to-end MEASURE
  race under the ZCASC protocol, banked in the same `@nat` slot. Kill
  switch `VFFT_NO_NAT_ILP`. **Race result: ILP won 9.1×/7.2×/5.7×/4.0×
  at 128/256/512/1024** — the convert incumbent never came close.
- [x] **B3 ✅** — explicit-SCRAMBLED in-place attaches HIT-ONLY on the
  banked ILP verdict (single `@nat` writer; a miss serves the classic
  path). Execute dispatches aliased il2p/il3p before the convert path,
  both directions, both orders (attach implies verdict).
- [x] **B4 ✅** — `vfft_ilp_front_gate.c` ALL PASS: measure (raced) +
  consume (no re-race, bitwise-identical output, replay-line coherence) at
  all four cells, fwd vs naive IN ORDER + bwd vs N·x at ~1e-15, aliased,
  scrambled arm IDENT with matched roundtrip, and the 2048 boundary still
  goes ZCASC (no ILP shadowing).
- [x] **B5 ✅** — `--k1nat` gained the sub-2048 direct cell (no kind-4
  line exists for the IL band and no kind-3 K=1 lines ship, so file-driven
  enumeration cannot reach them; an explicit `[N] < 2048` runs the cell
  straight through the front door, label `z:ilp`). Recorded as tracked
  §6.4: **vs MKL in-place natural 0.83–0.91 / 0.76 / 0.68–0.71 /
  0.59–0.74 at 128/256/512/1024** (warm, ×3, cross-engine ~e-16). Read
  with the internal 4–9× win: users got 4–9× faster; the residue stands
  on MKL's deepest-investment ground (small-N batched IL natural) and its
  closure is the Phase C3 K-across-SIMD question, not routing.

## Phase C — K=2..4 interleaved batching (measure, then build)

Today K≥2 interleaved has NO native z route anywhere: every handle converts
(deinterleave → K-strided split engine → reinterleave, plus the il_me
fused-vs-padded verdict). The split engine is strong at K>1; the question
is purely whether two conversion passes lose to MKL's native interleaved
batching, and where.

- [ ] **C0** — contract recon: pin OUR interleaved K>1 layout (lane-
  contiguous `dist = N` complex vs element-interleaved) from
  `_il_pad_dein`/`_exec_c2c_interleaved` addressing, and write it into the
  layout contract docs. The MKL arm must mirror it exactly
  (`NUMBER_OF_TRANSFORMS=K`, `INPUT_DISTANCE` to match) or the bench
  measures a strawman.
- [ ] **C1** — the gap map: ONE new canonical-bench mode (`--kzb`),
  K ∈ {2,3,4} × N ∈ {256..32768}, ours-as-is (convert) vs MKL native
  batched interleaved, both placements' story starts OOP, cell-per-process
  under the noise protocol. Honest priors, stated pre-run: conversion
  dominates at small N (MKL wins big); at large N the split engine's
  interior may hold (gap shrinks). No routing change in this step.
- [ ] **C2** — the cheap native candidate: LANE-LOOP — run the K=1 z
  machinery per lane (`dist = 2N` doubles): cascade per lane at ≥2048
  (plane + tables shared, cache-warm across lanes), il2p per lane below.
  Zero new kernels. Wire as a create-time candidate raced vs the convert
  incumbent per (N,K) cell, banked — plans from machinery, never a
  hand cutoff.
- [ ] **C3** — decision point, from C1+C2 numbers: if lane-loop + convert
  covers the map competitively, STOP (record why). True K-across-SIMD z
  kernels (the emitter campaign — what MKL runs at small N) open ONLY if
  a high-volume region measurably loses, as their own spec'd campaign.

## Phase D — OOP natural ≥2048: il2p/il3p vs cascade race (carried from the previous plan)

`order=NATURAL` OOP ≥2048 still routes to il2p/il3p; the stfn cascade never
races there. Needs its own verdict home (the `@nat` table is documented
in-place-only — either a placement axis on the entry or a separate slot;
settle THAT design question first, two-writers history applies).

- [ ] **D1** — verdict-home design decision (placement axis vs new slot).
- [ ] **D2** — candidate + race + bank + consume, ZCASC pattern.
- [ ] **D3** — gate + the OOP natural-vs-natural table vs MKL
  (`--oop`-family mode or extend `--k1nat` with a placement flag —
  whichever keeps ONE canonical harness).

## Phase E — spill-control extension to the other kernel families
*(added 2026-08-04 after the t2b promotion; parent record =
`docs/performance/il_register_pressure.md` + the twmem campaign. The rule
that governed Phase-E scoping so far governs its execution: MEASURE the
family first, treat only where the numbers scream, race every treatment
per cell.)*

- [x] **E1 — CLOSED NULL BY MEASUREMENT 2026-08-04.** Three-layer
  resolution, each part load-bearing: (1) the planned fix was REFUTED
  mechanically — GCC canonicalizes named temps and inline load
  expressions to identical SSA (the hand-inlined variant compiled
  BYTE-IDENTICAL); cil's immunity is its per-iteration cursor
  ADDRESSES, not its render. (2) The bounce itself is r8-ONLY (r4
  register-hoists its 6 halves cleanly) and the "~7 %/iter" figure was
  a census REGEX BUG (byte-continuation lines counted as instructions):
  the true delta of the working fix (LICM-defeating opaque cursor,
  bit-exact at 13 cells) is +4 insns/iter traded for −34/group prologue
  and 18→0 stack touches. (3) Raced under the full protocol: HONEST
  NULL — no time win at any production geometry, 3 runs, clean
  controls. Do not re-open without new evidence; the emitter seam recon
  (e1msg_emitter_seam.md) is banked should the calculus ever change.
  Records: `e1msg_probe.md` (with correction banner), `e1msg_timing`
  logs, verifier adjudication in the workflow journal.
- [ ] **E2 — OOP `t1` r32 campaign (the worst body in the tree).**
  41.6 % total stack traffic (25.6 % ymm spill + 13.5 % scalar pointer
  reloads) — worse than pure-IL r32 ever was. TWO root causes, ordered:
  (a) the **pointer zoo** — ~100 stack-parked leg pointers vs cil's one
  cursor: cursor-ize the addressing first (its own win, prerequisite for
  anything else); (b) then evaluate a blocked-analog for that emitter
  family (does not exist there yet — new machinery, spec before build).
- [x] **E3 — CLOSED BY ANALYSIS 2026-08-04: already optimal, shuffle-
  conserved.** The premise was stale: the mid is ALREADY lane-cross-free
  (all 32 lane-cross ops/iter live in the leaf's turned stores — the
  reference-optimal placement, delivered long ago by the n1t design) and
  carries ZERO reint/deint (z interleaved end-to-end, plain contiguous
  loads AND stores). The residual 12–14 % shuffle share is exactly one
  in-lane swap per non-trivial complex rotation (31 tw + 29 ±i + 20
  fixed-root = 80/iter, IDENTICAL across t2/t2b/t2b48 — conserved under
  restructured math; per-leg identical to the reference design). On AVX2
  this is the floor for interleaved-in/out; only split planes remove it
  (the banned hybrid family). Remaining shuffle levers are MATH
  (fewer-rotation factorizations = the planner's stage-count axis, E6/E7
  territory), not layout. Verified cell-by-cell. Record:
  `docs/research/twmem_campaign/results/e3_shuffle_roles.md` +
  `e3_layout_design.md`.
- [ ] **E4 — pure-IL leaf question (`n1t(32)`, LOW priority).** 26.7 %
  spill but perfect ns/pt scaling at both N — its spills appear cheap.
  The blocked emitter REFUSES the leaf class today. Open question, only
  worth touching if E3's restructuring gets into the leaf anyway.
- [ ] **E5 — t2b16@512 quiet magnitude**: de-bias the timing probe's
  r16 section (systematic control bias, same sign 3/3 runs — rotate arm
  order / separate arena) so the banked t2b16 win gets a quotable
  number.
- [x] **E6 — CLOSED SUBSUMED-BY-T2B 2026-08-04.** The post-promotion
  re-race (pairs probe rebuilt so its il2p arms inherit the live blocked-
  mid race) erased il3p's margin: at 512 an il2p pair (32×16, t2b48 mid)
  took the crown at 383 ns — exactly the old il3p winning time; at 1024
  the 32×32 heuristic (1047 ns post-t2b48) beats every chain by 38 %+.
  The 3-stage advantage WAS avoiding the fat spilling mid; t2b fixed the
  mid in place — the levers do not compose. The E6 routing design (incl.
  the `VFFT_NAT_ILP3=8` fix for the confirmed engine-agnostic-replay bug
  risk) is BANKED in `twmem_campaign/results/il3p__routing_design.md`
  for any future engine-heterogeneous verdict need. SALVAGE SHIPPED
  instead: **the pair-ORDERING race** in `_k1_il_candidate` — with
  blocked mids, (32,16) vs (16,32) differ by mid class and the balanced
  heuristic can't see it (+4.5 % at 512, above spread); raced at create,
  3 % hysteresis, `VFFT_NO_T2B` kill.
  🔴 **COHERENCE RULE learned shipping it (two instances, one
  pre-existing since the t2b promotion): any create-time race with a
  non-bit-identical candidate MUST memoize its pick per process** — else
  two handles in one process (natural+scrambled, measure+consume) pick
  differently and the bitwise-identity contracts break. Both races now
  carry per-process memos (`_k1_il_candidate` keyed by N;
  `_vfft_il2p_race_mid_f` keyed by (R1,R2)); `vfft_k1scr_gate` is THE
  detector (the only gate comparing two independently-created handles
  byte-for-byte) and belongs in every future promotion's gate set. Both
  front gates ALL PASS with both memos.
- [ ] **E7 — TILED il3p (the Bailey-band tcut question).** The span-rule
  verdict, recorded so nobody re-derives it: the 2-stage Bailey (il2p)
  is STRUCTURALLY untileable — a balanced factorization gives every pass
  a whole-plane span (`R·s ≤ T` degenerates to `N ≤ T`), and the
  transpose dependency forbids fusing the pass-1→pass-2 seam. The
  3-stage chains are different: small-span stages exist, and the prize
  is concrete — il3p loses at 1024 TODAY purely on working set (3
  buffers ≈ 48 KB = all of L1d) while winning at 512; fusing stages 2+3
  per tile of mid1 shrinks the second intermediate to tile size
  (~33 KB total → fits). Steps: (a) recon — derive per-stage spans from
  the actual il3p strides for the 512-winning chain shapes at 1024 and
  check the fusion seam against the span rule (a derivation, not a
  probe); (b) if legal, probe-level tiled execute (driver-loop change,
  existing kernels — the tcut lesson says no new kernels needed until
  measured); (c) race vs il2p-with-t2b48 at 1024/2048 per cell. Honest
  prior: t2b48 already took 1024 to −25 %; tiled-il3p must beat THAT
  incumbent, not the old one.

## Later (explicitly deferred, not dropped)

- Natural-aware CHAIN race (natural replays scrambled-banked chains; the
  stfn r4/r8 terminator forms are not equal — dp planner under the natural
  objective; known headroom at cells whose banked chain ends in 8).
- kind-3 wisdom violation retirement (mono/Bailey OOP routing is still the
  vfft.c heuristic — a standing plans-from-machinery violation).
- MT K=256 natural perf question (MT ≈ 0.4–1.0× ST, memory-BW-bound —
  correctness re-validated 2026-08-03 by `mt_c2c_gate.c`, ALL PASS).
- Cool-machine repeats of every quotable table (tcut, k1zip/k1nat pair).
- ZTURN-T geometry campaign (the only route to a winning DIT) — PARKED.
- Tugbars' commit checkpoint (everything since `8e8a6625`).

## Rules that bind every step

- vs-MKL numbers ONLY via `bench_1d_vs_mkl.c --mode`; never a new harness;
  never run with no args. Calibration through `vfft.c` (the front door).
- 🔴 **Check the kind-4/wisdom VINTAGE (chains AND width fields) before
  quoting any table** — the burned-table rule, 2026-08-03.
- Plans/candidates from planner machinery raced per cell — never a hand
  cutoff; verdicts banked in wisdom, consume replays without racing.
- 🔴 Roundtrip cannot gate ordering — gate each direction vs an
  independent reference; cross-engine elementwise vs MKL where both are
  natural.
- Thermal protocol: pinned core 2, HIGH, paced, alternated order, ≥15–17
  rounds for races / cell-per-process ×3 for tables, control arms, spread
  reported; a delta under the control is NOT a result.
- Rebuild every wisdom-writing binary after any wisdom change; emitter
  builds via absolute `--root` + `DUNE_CACHE=disabled` (WSL opam 5.2.0).
- Mixed-radix chains mandatory in every ordering gate (rho involution
  masks table mix-ups on uniform chains).
