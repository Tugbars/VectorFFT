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

- [ ] **A1** — widen the K=1 engine gate to admit `order == SCRAMBLED`
  (serve via the natural engines, identity permutation). Document the
  identity-serving in the order-contract comment block. Kill-switch not
  needed (strictly-better route; convert fallback remains the miss path).
- [ ] **A2** — gate: explicit-SCRAMBLED sub-2048 IL OOP through the front
  door — matched roundtrip `bwd(fwd(x)) == N·x`, output ELEMENTWISE equal
  to the NATURAL handle's output (identity = same bytes), and create must
  route to the K=1 engine (no `il_wr2` convert scratch allocated). Speed
  arm: SCRAMBLED == NATURAL within noise (same engine, same plan).
- [ ] **A3** — regression: ≥2048 SCRAMBLED still routes to the cascade
  (the gate change must not shadow the kind-4 attach), and split-layout
  SCRAMBLED (MODEB) untouched.

## Phase B — sub-2048 IN-PLACE interleaved tiers (the ZCASC template, one tier down)

il2p/il3p are already alias-safety-gated (A3 of the previous plan); mono is
structurally refused (`__restrict__`). What's missing is ONLY routing: the
in-place branch never attaches the IL engines, so both orders pay
tape/convert below 2048.

- [ ] **B1** — recon + mini-spec: what exactly does in-place sub-2048 IL
  execute today per order (which convert path, where the tape runs); what
  the il2p in-place call form is (aliased z, the `(z,NULL,z,NULL)`
  contract); whether il3p/prime compose aliased. One page, measured
  incumbent times per cell included (the race's floor).
- [ ] **B2** — attach + race, NATURAL first: in-place natural block gains
  an IL-engine candidate below 2048 (same pattern as ZCASC: candidate
  built at create, raced END-TO-END vs the incumbent's real execute,
  verdict banked in the SAME `@nat` slot — new mode value, e.g.
  `VFFT_NAT_ILP`; consume replays, `replay` log line, kill switch
  `VFFT_NO_NAT_ILP`). 🔴 Never a parallel selection path.
- [ ] **B3** — SCRAMBLED in-place sub-2048 rides Phase A's identity rule:
  explicit-SCRAMBLED in-place routes to the same IL candidate (order axis
  serves both). Gate both orders per direction vs naive DFT — 🔴 roundtrip
  cannot gate ordering.
- [ ] **B4** — front gate (cold-start, scratch wisdom): measure+consume
  per cell {128, 256, 512, 1024} × {natural, scrambled} × in-place, fwd
  IN ORDER vs naive per direction, aliased arm, tape-tier fallback still
  correct where the IL engine refuses a cell.
- [ ] **B5** — the number: `--k1nat`/`--k1zip` runs gain the sub-2048
  cells (same canonical modes — the cell list follows the banked verdicts,
  no new mode needed). Record the sub-2048 in-place table vs MKL in the
  tracked doc next to §6.3.

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
