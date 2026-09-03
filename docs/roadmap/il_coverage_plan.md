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

- [x] **C0 — DONE 2026-08-04 (4-reader recon). The contract is
  ELEMENT-INTERLEAVED across lanes: element `e` of lane `t` at
  `z[2*(e*K+t)]`.** Unanimous in code and already public: vfft.h:47/:198/
  :256-258 state it verbatim; `_il_pad_dein` (vfft.c:4804) reads the K
  lanes of point p as the K adjacent pairs at `z+2*p*K`; the flat
  `_vfft_z_dein/_z_inter` converts are only correct under this layout (no
  transpose exists); the il2il MT lane-slabs offset `zi+2*k0`. So the
  interleaved z layout IS the split lane-major layout with (re,im) fused —
  the convert feeds the split engine with ZERO reshaping; the two flat
  passes are the entire tax. **Design-history contradiction adjudicated:**
  interleaved_design.md pitfall #10 committed v1 to transform-major
  (`dist=N`, the MKL/FFTW default) but the same-day as-built record
  (il_architecture.md §1/§7) shows the shipped wrapper REFUSED
  transform-major and served lane-major; the §6 "batch geometry
  commitment" decision was never recorded. Code wins; vfft.h is the
  contract doc and it is already correct. Findings that reshape C1/C2:
  (1) the in-place K≥2 native il2il fold/slab plumbing EXISTS
  (dispatch vfft.c:4933-4998, slice_K-parameterized) but its codelet
  population was RETIRED 2026-07-24 (il_execute.h stubs return 0) — today
  it always falls through to convert; the seam is the natural home for a
  native candidate. (2) il_me = fused-vs-padded verdict, in-place
  interleaved, K%8≠0, DEFAULT order only; at K∈{2,3,4} the padded arm
  pads to Kp=8 (2–4× zero-lane waste) yet may win since the fused tail
  runs narrow. (3) The bench's existing K=4 rows are SPLIT-plane
  in-place REAL_REAL — LAYOUT_INTERLEAVED at K≥2 is completely
  unexercised by the canonical bench. (4) K∈{2,3} have ZERO wisdom cells
  in either file — the C1 mode must use direct-cell targeting (the
  --k1nat fallback pattern, bench:2918-2923), not the wisdom walk.
  (5) 🔴 K's lane-major meaning inside stride plans and the flat converts
  is shared load-bearing code with 2D/3D column passes and R2C CCE —
  Phase C routing lives at the vfft.c dispatch level ONLY; never change
  what K means to the engines.
- [x] **C1 — MAP MEASURED 2026-08-04** (`--kzb` shipped in the canonical
  bench: k1z fairness shape, direct-cell dispatch, two MKL arms —
  mirror = COMPLEX_COMPLEX `NUMBER_OF_TRANSFORMS=K, DISTANCE=1,
  STRIDES={0,K}`; home = `dist=N` diagnostic; MKL mirror/home
  self-check via transposed input; cross-engine gate vs mirror. Cells
  ×3, cell-per-process, pinned, alternated; champions calibrated+banked
  round 1). **Results (ratio_mirror ranges over 3 runs, ~4e-16 xerr
  everywhere covered):**
  - K=2: 256=0.43–0.44 · 512=0.51–0.62 · 1024=0.52–0.57 ·
    2048=0.49–0.68 · 4096=0.55–0.59 · 8192=**0.39–0.44**
  - K=3: 256=0.40–0.50 · 512=0.60–0.70 · 1024=0.58–0.59 ·
    2048=0.44–0.64 · 4096=0.49–0.57 · 8192=0.42–0.50
  - K=4: 256=**0.31–0.38** · 512=0.46–0.52 · 1024=0.76–1.01 (the one
    near-parity cell) · 2048=0.65–0.75 · 4096=0.55–0.59 · 8192=0.51–0.66
  - ratio_home (positioning diagnostic): 0.13–0.46 across the map; MKL
    itself pays 2–3× on OUR lane-major layout vs its home layout.
  - 🔴 **COVERAGE HOLE: N ≥ 16384 at K∈{2,3,4} REFUSES natural OOP
    loudly** ("no natural-order out-of-place C2C champion — the natural
    kinds are gated on this cell") — every 16384/32768 cell, every K,
    all 3 rounds. Above 8192 these K have NO natural OOP route at all.
  **C1 verdict:** the convert route loses ~2× to like-for-like MKL
  essentially everywhere covered (prior refuted in one direction: it is
  flat-bad, not small-N-concentrated), and the map ends at a hard
  coverage wall. The two flat passes + the split interior at tiny K
  never approach parity.
- [x] **C2/C3 — RESOLVED 2026-08-04 BY A DIFFERENT ROUTE THAN CHARTERED:
  transform-contiguous batch geometry, served by looping the K=1
  engines. SHIPPED same day.** Tugbars: *"do we even need lane major
  codelets for K=1? why don't we have contiguous for all? How does MKL
  handle this"* — the question that dissolved the problem. MKL and FFTW
  both EXPOSE the geometry as an axis (DISTANCE/STRIDES, idist/istride)
  and both default to transform-contiguous; we had inherited lane-major
  from the split engines and applied it to IL at every K, which Tugbars
  called *"a mistake"* to have carried over.
  - **API**: `config.batch_geom` ∈ {`VFFT_BATCH_DEFAULT` (0),
    `VFFT_BATCH_TRANSFORM_CONTIGUOUS` (1), `VFFT_BATCH_LANE_MAJOR` (2)}.
    **DEFAULT resolves PER LAYOUT** (Tugbars, 2026-08-04): INTERLEAVED →
    transform-contiguous, SPLIT → lane-major. That is what makes a zeroed
    config always correct for the layout it asked for: interleaved callers
    get the canonical fast geometry, split callers get the only geometry
    their engines have. An EXPLICIT transform-contiguous request on SPLIT
    is **refused loudly**, never silently served as lane-major (the
    padding design's no-silent-corruption rule). At K==1 the geometries
    are the same addressing, so create never wraps (gated bitwise).
    🔴 BEHAVIOR CHANGE: INTERLEAVED K>1 used to default to lane-major.
    Callers who zero their config and pass lane-major data must now say
    `VFFT_BATCH_LANE_MAJOR`. Exactly one in-tree site was affected — the
    `--kzb` bridge arm, now explicit (left implicit it would have turned
    the baseline column into a second loop arm and read 1.0×).
  - **Implementation**: ~40 lines. A TC create builds ONE K=1 handle
    through the same front door and stores it as `h->tcb`; execute runs
    it K times at 2N-double strides. **Zero new kernels, zero layout
    conversion, no batched plan.** Inherits every K=1 route, wisdom
    verdict, and create-time race automatically — and every future K=1
    gain lands here for free (the sub-2048 directive compounding).
  - **MEASURED** (`--kzb` loop arm, K∈{2,3,4} × N∈{256..8192}, ×3,
    cell-per-process, pinned, alternated): **2.2–5.7× faster than the
    lane-major bridge at every cell.** vs MKL batched on its OWN home
    layout: 0.56–0.87 sub-2048 · **1.01–1.18 at 2048** · 0.90–1.13 at
    4096 · 0.79–0.99 at 8192. That profile IS our K=1 competitive
    position, inherited exactly as predicted — so the residual gap is
    now a pure K=1 sub-2048 question, not a batching question.
    ⚠ K=4 N=512 spread 0.56–0.80 is wide; re-measure before quoting.
  - **NO TAIL EXISTS** in this geometry: K=3, K=7, K=11 are just that
    many loops. No padding, no remainder kernel, no `exec_me`
    verdict, no even-K constraint — the property the campaign docs
    already noted about MKL ("odd K is just one more transform").
  - **Gate**: `build_tuned/benches/vfft_tcbatch_gate.c` ALL PASS —
    fwd/bwd vs independent scalar DFT per transform with DIFFERENT data
    in each block (catches a route that transformed only one block),
    BITWISE identity vs K separate K=1 executes, in-place both
    directions, lane-major still correct at the same cells, K=1
    non-wrapping bitwise. Includes an odd-N cell (96×3).
  - 🔴 Contract note found while gating: INTERLEAVED C2C in-place is
    spelled `(z,NULL,z,NULL)` — `dre` is REQUIRED and aliases `sre`;
    `(z,NULL,NULL,NULL)` is rejected by the signature check.
  - **Policy (Tugbars)**: transform-contiguous is the **canonical
    supported** IL batch geometry; lane-major "exists for the sake of
    completion" — served by the existing bridge, no further
    optimization investment. Mirrors MKL's own architecture (tuned path
    for the sensible layout, generic path for the strided one).
  - [x] RESOLVED (2026-08-04, his call): the DEFAULT resolves PER LAYOUT
    — INTERLEAVED → transform-contiguous, SPLIT → lane-major (SPLIT has
    no contiguous support and an explicit contiguous request on SPLIT is
    refused loudly). Gated: zeroed-config == explicit
    TRANSFORM_CONTIGUOUS bitwise (`run_default_geometry`).
- [x] **STRUCTURAL BLOCKED DEFAULT (2026-08-06)** — the emitted blocked
  kernels are now THE R≥32 forward kernels: `vfft_il2p_apply_blocked_default`
  (il2p.h, beside the registry) runs inside `vfft_il2p_create`, with a
  leaf-only analog inside `vfft_il3p_create` — one choke point, so every
  creator (vfft.c routes, il_prime inners, dp_planner candidates, gates)
  serves and measures the same kernels. Rationale: a monolithic R≥32 body
  holds ~40–64 live values against AVX2's 16 registers (~27% ymm spill,
  il_register_pressure.md) — the same tier rule the split emitters apply at
  generation time (codelet_oop.ml Tier A/B on `isa.vec_regs`); not a
  per-cell race. Scope: forward only (no blocked bwd twins), even counts
  only (monolithic keeps the odd-count tail duty), 4·8 preferred / 2·16
  fallback, R=16 deliberately excluded (fits the file — census control
  class). `VFFT_NO_ILBLK` create-time kill switch = the bench A/B hook;
  wisdom `il_kv` overrides the default, with new nibble 0xF
  (`VFFT_IL_KV_MONO`) forcing monolithic so a platform where blocked loses
  banks it as a VERDICT. Gates: ilp-front / k1scr / tcbatch PASS (⚠
  ilp-front + natural-front need an EMPTY scratch wisdir — they assert the
  race RAN; a populated copy reads "NO RACE" on every measure pass).
  - ⏸ OPEN: **R=64 blocked does not exist** — no files, despite (16,64)/
    (32,64)-class pairs placing monolithic R=64 bodies (the worst spillers)
    at 1024/2048. Emitter work; do the `Dft.select_expansion` extraction
    first (codelet_oop.ml's own drift warning), then the cil tier can move
    to emit time and this create-time rule collapses into it.
  - ⏸ OPEN: blocked **bwd** twins (t2t/n1 classes) — until they exist,
    backward keeps the monolithic spill profile.
- [x] **TC-batch MT (2026-08-06)** — the transform-contiguous path now
  slabs its K transforms over the stride pool: worker t runs its slab
  through `vfft_execute` on its OWN identically-created K=1 clone
  handle (`h->tcbw[]`), caller takes slab 0, no barriers. Clones exist
  because the K=1 IL engines are not reentrant (il2p/il3p `mid`
  scratch, zturn `plane`). Three guards, all create-time in vfft.c:
  `_tc_inner_mt_safe` (clones only for pool-free routes — a
  convert-fallback route would mutate the pool from a worker),
  `_tc_clone_equiv` (a clone must match the primary's attach + chain +
  natord + exact kernel pointers, else it is destroyed — one batch must
  never mix two scrambled combs), and the `VFFT_NO_TCMT` kill switch.
  Engage = the batch's RACED threading verdict (serial vs slabs at create,
  banked T-free as eng=tcb tcmt= on its q=K row, measurement_arms B5); the
  2048-point scalar floor and VFFT_TCMT_FLOOR are retired (2026-09-04).
- [x] **The K-across-SIMD campaign (chartered earlier the same day) is
  CLOSED UNBUILT.** Its premise — that K≥2 interleaved needs its own
  kernel family — was false. Once each transform saturates the vector
  unit, batching is a convenience, not a performance technique; MKL's
  own batched home time is ≈K× its K=1 time (2048: 2×2151=4302 vs
  4043–4087 measured; 4×2151=8604 vs 8114–9003). The RE finding that
  MKL serves lane-major with a generic spill-heavy body stands, but is
  now a curiosity rather than an opening: nobody should feed batches in
  that layout when transform-contiguous exists. MT reinforces it —
  lane-major K-split needs K ≥ 4T to own whole cache lines (K=32,T=4 →
  128B/thread ✓; K=4,T=4 → 16B/thread, false sharing on every line),
  while transform-contiguous gives each thread a private contiguous
  block. Historical record of the superseded charter:
- [x] **C2/C3 — the superseded charter (kept for the reasoning trail):
  the convert bridge is NOT an acceptable end state.** His ruling: the
  route-level dein → split-engines → inter bridge "is hybrid-ing IL and
  split with each other just because we do not have a z-consuming
  engine" — the same disease as the banned il_in/il_out hybrid codelets
  (🔴🔴🔴 never-build rule), one level up. The C1 escalation condition
  was met everywhere anyway (~0.4–0.7 flat). VERDICT: charter the
  native z-consuming K≥2 tier as ITS OWN SPEC'D CAMPAIGN (C3's
  language). Constraints fixed now:
  - **PURE IL end-to-end**: z in, z out, no split borrowing anywhere in
    the engine — codelets from the cil/pure-IL emitter families, NEVER
    derived from split bodies (the retired 2026-07-24 population was
    the derived kind; repopulating the seam with derived codelets would
    recreate the ban).
  - **The layout is K-across-SIMD-natural**: lane-major means the K
    lanes of one element are 2K contiguous doubles (K=4 ⇒ exactly 2
    ymm) — the vector unit spans lanes with NO gather, which is
    precisely "what MKL runs at small N". Per-lane-strided execution of
    K=1 kernels is the weaker shape; the campaign spec should start
    from K-across-SIMD.
  - The bridge REMAINS the serving mechanism until the native tier
    races past it per (N,K) cell (coverage never regresses during the
    campaign; create-time race + bank, plans from machinery).
  - Small independent item, do first: **close the ≥16384 K∈{2,3,4}
    natural-OOP refusal** (coverage defect; serve at bridge-class speed
    now, native later).
  Campaign spec = next step (kernel shapes, emitter seam, dispatch
  home, gap-map targets from C1's worst cells); it does NOT start
  until spec'd and approved.

## Phase D — OOP natural ≥2048: il2p/il3p vs cascade race (carried from the previous plan)

`order=NATURAL` OOP ≥2048 still routes to il2p/il3p; the stfn cascade never
races there. Needs its own verdict home (the `@nat` table is documented
in-place-only — either a placement axis on the entry or a separate slot;
settle THAT design question first, two-writers history applies).

- [x] **D1 — MEASURED 2026-08-04 (`--k1noop`, canonical bench, cells ×3).**
  The size question answered itself: **worst corner in the product.** vs
  MKL OOP-natural: 2048 = 0.60–0.74 (native il2p 32×64), 4096 = **0.39**
  (il2p 64×64), 8192 = **0.31**, 16384 = **0.24**, 32768 = **0.17**
  (convert fallback; 210 µs vs MKL's 37 µs). Monotone degradation = the
  convert-fallback signature. Same cells in-place natural: 0.94–1.16 —
  the placement axis alone cost ~5× at 32768 while the engine that fixes
  it sat unrouted: `vfft_zturn2_execute_fwd/bwd(p, zin, zout)` take
  distinct buffers as the BASE contract (in-place is the allowed special
  case) and natord is just a different terminator table. Pure routing gap.
  `--k1noop` = natural OOP both engines (DFTI NOT_INPLACE), own CSV,
  honest `nat-oop` path label.
- [x] **D2 — verdict home DECIDED + BUILT: `@natoop` sibling table.** Same
  entry shape and (N,K) key as `@nat`, separate table in the SAME proto
  wisdom file. Why not a placement axis on `@nat`: the two regimes have
  different incumbents (in-place: tape/ILP/ZCASC; OOP: the K=1 engine vs
  the natord cascade) and a shared (N,K) slot would let each regime's
  bank clobber the other's — the @nat single-writer rule, extended
  per-placement. Why not a kind-4 trailing field: kind-4 is rewritten by
  the scrambled t2q race and calibrate_zchain, writers that know nothing
  of the natural verdict (the calibrate_zchain two-writers incident,
  again). Verified before building: the loader skips every unknown `@`
  tag, so shipped binaries ignore `@natoop` and re-measure — never wrong
  (they DO strip the lines on their next save; accepted, same as `@nat`'s
  own introduction). Mode semantics: `ZCASC` = replay kind-4 +
  `set_natord` + attach; `FREE` = the engine handle as built. BOTH
  outcomes bank → the pick is process-coherent (create-race coherence
  rule: the candidates are not bit-identical).
- [x] **D3 — wired + gated 2026-08-04.** Race block at the end of the K=1
  OOP engine create (vfft.c): candidate = `_k1z_wisdom_replay` (recal
  cleared on the copy) + `set_natord`, raced END-TO-END against the
  finished handle's REAL execute path (public `vfft_execute`, src→dst
  distinct, src read-only ⇒ no reseed hazard), B5 protocol (5 rounds,
  alternated, medians), attach rides the existing zsplit||zturn-first
  dispatch — zero execute changes, destroy already frees the attach.
  Kill switch shared: `VFFT_NO_NAT_ZCASC`. Gate =
  `vfft_natural_front_gate.c` Phase D arm: per cell oop-meas (race must
  run; ZCASC must WIN ≥4096 — D1 says an engine win there is a wiring
  bug; 2048 is the one competitive cell, either winner legal) + oop-cons
  (no race, replay) + measure-vs-consume fwd BITWISE memcmp (coherence) +
  src-untouched check + @natoop save/load round-trip + N=256 OOP
  no-cascade regression. First cold run: ZCASC won 4096–32768, engine
  (il2p) legitimately won 2048 on the scratch default chain.
- [x] **D4 — MEASURED 2026-08-04, hole CLOSED.** `--k1noop` post-wiring,
  same protocol as D1 (cells ×3, cell-per-process, pinned core 2,
  alternated order, kind-4 vintage verified w1024@48KB). vs MKL
  OOP-natural, same-run ratios: 2048 = **0.99–1.11** (parity-or-ahead,
  the competitive cell), 4096 = **0.91–0.94**, 8192 = **0.95–0.98**,
  16384 = **0.94–0.98**, 32768 = **0.88–0.91**. Against D1 that is
  1.5×/2.4×/3.0×/3.9×/**5.1×** route speedup; 32768 went 210 µs → 40.5 µs.
  Cross-engine err ~7e-16 (same spectrum elementwise, natural both).
  @natoop verdicts banked in shipped wisdom (all ZCASC). Now sits 2–7 %
  under the in-place natural tier's ratios — consistent with OOP being
  MKL's strongest ground, no further routing residue.
- **Bonus fix (2026-08-04, found by this phase's gate work):** the B5
  in-place ZCASC/ILP race banks read the deployed chain from local `p`
  AFTER the tape race may have destroyed it (PSWAP/SCR install paths
  destroy `h->cplan` and swap without updating `p`) — freed-heap garbage
  was banked into @nat lines (one poisoned 512 ILP line WAS in shipped
  wisdom; scrubbed) and positive garbage `nf` made the saver's factor
  loop segfault nondeterministically (ILP gate 3/3 crash repro). Fixed:
  both banks read `h->cplan`. All three gates re-run green on the fixed
  tree.

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
- [~] **E2 — OOP `t1` r32 campaign: lever (a) REFUTED 2026-08-04, lever
  (b) is what remains.** The hand-probe
  (docs/research/twmem_campaign/results/e2t1_cursor_probe.md) killed the
  charter's ordering claim: plain cursors are a GCC-canonicalization
  no-op (E1's trap verbatim), and opaque cursors — bit-identical 7/7,
  scalar frame reloads −68%, body −11 — race CONSISTENTLY SLOWER
  (+16..34%, clean LOSS at me=128 outside a 4% control spread). The
  stack-parked pointers are independent loads the OOO absorbs under an
  FMA-bound body; a walking cursor trades them for a serial address
  chain. Static counts are not speed (r64-blocked, E1, now this).
  MKL's generic strided body independently agrees (per-leg imul, no
  walk). ⚠ placement luck ±9% measured on this kernel (ship-vs-ship2
  control) — any future promotion must race with a control.
  🔴 Do not re-attempt cursor forms on this family. REMAINING: lever
  (b) only — blocked-analog for the 25.6% ymm spill plane (Tier-B
  spill_re/im[32] PASS1/PASS2), the t2b-shaped MATH restructuring;
  new codelet_oop.ml machinery, spec before build.
  **SHAPE SPEC'd 2026-08-04** by the large-radix investigation
  (docs/research/mkl_smalln_campaign/results/large_radix_verdict.md —
  benchmark-derived summary only in tracked docs). Target census: 73
  frame slots, 9 write-once/read-once, **59 multi-stored, 27.7% ymm
  stack** = the most churn-dominated body measured anywhere, worse than
  the monolithic mid the same lever beat by −9..−21%. Constraints, in
  order: (1) **split p=8 (the 4·8 form), NOT 2·16** — this body is
  split-layout at 4 columns/iter (32 legs × 2 planes = 64 ymm live);
  peak-live max(p,m) puts 2·16 at 32 ymm (still 2× the file) and 4·8 at
  exactly 16; (2) **race the column unroll (4 vs 2) as a second axis —
  a 2×2, not one arm**; the 4-column unroll is an independent live-width
  multiplier blocking does not touch, so a blocked-at-4 null would be
  uninterpretable; (3) addressing stays untouched. Confounders: the
  existing 64-store spill plane means the goal is "make the deliberate
  plane the ONLY plane", and the family's unfolded twiddles are NOT
  additive with blocking (block first, re-census folding after).
  m=4 is not bit-identical ⇒ numeric gate, and promotion needs a raced
  control (placement luck ±9% measured here).
- [x] **E8 — ANSWERED 2026-08-04: the leaf's pressure is the RADIX, not
  the corner-turn.** (`docs/research/twmem_campaign/results/
  e8n1t_cornerturn.md`; static census only, both stripped arms
  numerically wrong by construction and never built.) Removing the
  ENTIRE corner-turn from `n1t(32)` — all 32 `vperm2f128`, verified by
  shuffles 81→49 — left frame slots **IDENTICAL (32→32)** and churn
  essentially unchanged (**27→25**), share 27.8→24.4%. Blocking on the
  same axes moved the mid 32 slots/26 churn/21.6% → **24/0/8.1%**. Not
  the same category of effect. ⇒ **The decision resolves to the CHEAPER
  option: port `emit_blocked` to the N1T kind** — an implementation gap
  (emit_blocked writes its own stores and never inspects `kind`), not a
  store-lattice redesign. The corner-turn rides along inside the blocked
  stores; this result says that is affordable because it was never the
  pressure source. Third independent confirmation that addressing form
  is not the lever: the leg-major arm pushed scalar frame reloads
  22→51 while vector pressure stayed put.
- [x] **E9 — REFUTED BEFORE BUILD, 2026-08-05.** E8's positive half was an
  inference carried from a twiddled MID to an untwiddled LEAF; measured on
  the closest proxy, it is wrong. A blocked untwiddled r32 leaf already
  existed in-tree (`radix32_z_n1b_avx2.c`, an untracked orphan referenced
  by nothing) and is **BITWISE IDENTICAL** to the shipped monolithic leaf —
  the ideal controlled comparison. It shows t2b's full static signature
  (churn 43→17 = −60%, spill st/ld 110/66→72/45, slots 53→37,
  write-once/read-once 5→19) and **no time**: +13.5% at the cleanest cell
  (count=32, control spread 0.5%), wash at the other clean cell (count=64),
  no win anywhere. The ~1-day emitter port is saved. Record +
  probe: `docs/research/twmem_campaign/results/e8n1t_cornerturn.md`
  (§E9 PREDICTOR), `probes/e8b__gate.c`.
  🔴 **META-LESSON, now three-for-three: static stack-traffic reduction
  does NOT predict time in these bodies** — E2(a) cursors (reloads −68%
  → +16..34% slower), reord (peak live 35→33 → marginal, unpromoted),
  E9 predictor (churn −60% → +13.5%). The ONE lever that ever translated
  was `t2b`, whose target was the **twiddled mid**, where restructuring
  changed what sat on the critical path rather than merely where values
  lived. **Any future "the census looks bad here" proposal must be raced
  on a proxy BEFORE emitter work.**
  🔴 CORRECTION (same day): an earlier note here called `radix32_z_n1b_avx2.c`
  an untracked orphan and said to delete it — WRONG. It is tracked
  (42f24533) and one of a whole emitted family (n1b at R = 9,15,16,21,25,
  27,32,45,49,64, most with bwd twins); the greps behind that claim ran
  from a drifted cwd and returned false negatives. Do not delete. The
  measurement stands (bitwise-identical, slower); the disposal advice did
  not. Separate open question, its own session: whether the n1b family is
  wired to any route at all.
  *(original charter, for the record:)*
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
