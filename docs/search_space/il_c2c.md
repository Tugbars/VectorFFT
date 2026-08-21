# IL C2C — measurement arms

The complete inventory of what `dp_planner_il.h` races for interleaved K=1
C2C: which axes exist, what each one's legal values are, and what the winner
records. Companion to [docs/wisdom/05_calibrator_pipeline.md](../wisdom/05_calibrator_pipeline.md),
which covers the SPLIT/stride calibrator (`dp_planner.h`) — a different
machine with a different search.

Code of record: `src/core/planning/dp_planner_il.h`
(`_il_dp_enumerate`, `_il_dp_race_bwd`, `vfft_il_dp_emit_wisdom`).
Kernel registry of record: `src/core/oop/il2p.h`.

> **Counts in this document are MEASURED**, by
> `build_tuned/benches/il_dp_cand_census.exe`, never inferred from the shape
> of the loops. That probe exists because an envelope estimate over the loop
> nesting was wrong by ~2.4x. Re-run it after adding any axis.

---

## 1 — Two order classes, two disjoint enumerators

`ord` is a key, not a ranking axis, and the two classes do not share a
candidate shape. `_il_dp_enumerate` branches on it and returns.

| | NATURAL | SCRAMBLED |
|---|---|---|
| routes | MONO, 2P_PURE | CASCADE only |
| shape | radix pair + kernel forms | factor chain + engine + terminator + tile |
| metric | `fwd1` | `joint2` |

The metric differs because the verdicts differ in kind. A 2P plan is measured
forward-only. A cascade route verdict cuts over atomically for both
directions, so it is measured as one fwd+bwd iteration and gated on the
roundtrip — the kind-4 metric law in `wisdom2_oop_reader.h`.

## 2 — NATURAL arms

### 2.1 Route MONO

One arm, when `vfft_k1_mono_il_fn(N, 0)` resolves. No sub-axes.

### 2.2 Route 2P_PURE — the radix pair

`R2` from `{4, 8, 16, 32, 64}`, `R1 = N / R2`, admitted when `R1` is a power
of two in `[4, 64]` and BOTH `vfft_il2p_leaf_fn(R2)` and
`vfft_il2p_mid_fn(R1)` resolve.

Pairs are ORDERED — `(R1,R2)` and `(R2,R1)` are different plans, because `R2`
is the column radix run at `count=R1` and `R1` the row radix run at
`count=R2`. The loop covers both by construction; there is no permutation
pass.

**Availability is asked of the il2p registry, never the split registry.**
Inheriting split's reach is a recorded measured bug: at N=16384 the balanced
split pick is 128x128 and both IL halves come back NULL, because IL kernels
stop at R=64 while split reaches 128.

### 2.3 The kernel-form axis (`il_kv`)

One byte, two nibbles: low = MID slot, high = LEAF slot
(`VFFT_IL_KV_PACK`, `VFFT_IL_KV_MID`, `VFFT_IL_KV_LEAF`).

Variant numbering, shared by both slots:

| value | meaning |
|---|---|
| `0` | the structural default `vfft_il2p_create` installed |
| `1` | 2·16 split (`t2b` / `n1tb`; at R=16 leaf this is the 4·4 form) |
| `2` | 4·8 split (`t2b48` / `n1tb48`) |
| `3` | tangent interior |
| `4` | tangent interior + the OTHER store edge (mid: M-128 half stores; leaf: T256 paired-permute wide) |
| `0xF` | MONO — force the monolithic registry kernel back |

Enumerated pools per slot, and the default each pool is measured against:

| slot | condition | default | pool |
|---|---|---|---|
| mid | `R1==32` and `R2` even | `2` | `{2,1,3,4}` |
| mid | `R1==16` | `0` | `{0,3,4}` plus `{1}` if `R2` even |
| mid | `R1==8` | `0` | `{0,3}` |
| mid | otherwise | `0` | `{0}` |
| leaf | `R2==32` and `R1` even | `2` | `{2,1,3,4}` |
| leaf | `R2==16` and `R1` even | `0` | `{0,1,3}` |
| leaf | `R2==8` | `0` | `{0,3}` |
| leaf | otherwise | `0` | `{0}` |

The full mid × leaf cross-product is pushed, minus the `(default, default)`
combination, which IS the base candidate already pushed at 2.2.

Two standing rules behind the pool shapes:

- **Parity guards are a count contract, not a preference.** Blocked kernels
  have no odd-count tail; the mid runs at `count=R2` and the leaf at
  `count=R1`, which is why a mid form is admitted only for even `R2` and a
  leaf form only for even `R1`.
- **Losing forms stay OUT on purpose.** Leaf forms multiply against mid forms
  on EVERY pair, and `_il_dp_push` refuses a cell outright past
  `VFFT_IL_DP_MAX_CAND` rather than truncating — a truncated pool is a BIASED
  pool, because the enumerator walks `nf` ascending and so drops
  systematically rather than randomly.

## 3 — SCRAMBLED arms

Four axes, multiplied. This section is the AXIS INVENTORY;
[docs/wisdom/10_zturn_calibration_flow.md](../wisdom/10_zturn_calibration_flow.md) is the same
calibrator as a SEQUENCE — enumerate → build → gate → bench → bank → replay,
colour-coded by what a model decides versus what the clock decides. Read 10
for the flow, this for the legal values.

### 3.1 Chain

Ordered chains over `{4, 8}` with `nf` in `[3, VFFT_ZSPLIT_MAX_NF]` (= 7) and
product exactly `N`. Most fail validation, which is why an envelope estimate
over the loop nesting overcounts so badly.

### 3.2 Engine (`zroute`)

`0` = legacy zsplit, `1` = ZTURN-S. **Legality is delegated to each route's
own create** — `vfft_zsplit_create` and `vfft_zturn2_create_chain` — never
re-implemented here, because a second copy of a validator drifts. A
fence-invalid chain simply yields no candidates for that engine.

### 3.3 Terminator (`t2q`)

Two values on both engines (`sterm`/`sterm2` on legacy, `stf`/`stf2` on
ZTURN). These are placement-order-sensitive twins and must be measured on the
installed binary, never hand-set.

One exception: ZTURN with `chain[nf-1] == 4` has no `stf2` twin, so the count
drops to 1 — otherwise the second arm would bench the same binary twice.

### 3.4 Tile width (`zt_tw`)

ZTURN only; zsplit has no tiled path. Widths come from
`vfft_zturn2_tile_candidates`, kept up to `VFFT_IL_DP_TILE_KEEP` (= 16).

- **Index `-1` is the UNTILED arm and must stay in the search.** Tiling is a
  per-cell verdict, not a default — 2048 measured a real +3.3% LOSS. Dropping
  the untiled arm would make "tiled" unfalsifiable.
- **No occupancy filter.** Every legal width is benched. Occupancy is
  reported, never used to narrow the set: a width that is never timed leaves
  no trace, so a wrong filter would be undetectable from its own output.
- Over-cap is a SIZING BUG, reported loudly, never silently dropped.

## 4 — The BACKWARD pass (`dir=bwd`)

A SECOND PASS over the forward winner, not a third dimension of section 2.

Enumerated per slot: `0` (whatever create installed) plus every variant that
is NOT the one create installed — `_il_bwd_default_variant` mirrors
`vfft_il2p_apply_blocked_default_bwd` exactly. Canonicalizing this way makes
every (mid, leaf) pair a DISTINCT plan by construction. Capped at
`VFFT_IL_DP_BKV_MAX_ARMS` (= 24), with any drop reported.

Two exclusions, both deliberate:

- **The default's twin.** At R >= 32 with an even partner, create installs
  variant 2, so `bkv=0` and `bkv=PACK(2,2)` are the SAME plan. The first cut
  of this pass walked a blind 6x6 grid and timed that kernel twice under two
  labels — measured at 32x32, `0x00` -> 914.2 ns against `0x22` -> 876.4 ns,
  a 4% "win" by one kernel over itself, with 7 of 16 arms duplicates. The
  forward has always eliminated this.
- **MONO.** It is the odd-count COVERAGE fallback, not a performance arm, and
  that coverage is automatic: an odd partner makes the blocked lookups return
  NULL and create leaves the monolithic kernel in place. Where monolithic
  genuinely competes is R <= 16 — it fits the 16 ymm registers — and there it
  IS variant 0 already, because create only overrides at R >= 32. The forward
  pools never enumerate it either.

Variants 1-4 are still walked BLIND rather than per-radix, so a newly emitted
backward codelet becomes raceable with no edit to the enumerator.

Backward kernels are SPARSER than forward ones — variants `3` and `4` have no
backward twin at any radix:

| slot | R=32 | R=64 |
|---|---|---|
| mid (`t2t_b`) | `1` = `t2bt216`, `2` = `t2bt48` | `1` = `t2bt416`, `2` = `t2bt88` |
| leaf (`n1_b_r2`) | `1` = `n1b216`, `2` = `n1b48` | `1` = `n1b416`, `2` = `n1b88` |

A combination with no emitted twin is REFUSED at build and never reaches the
timer, so it costs one create/destroy and does not appear as an arm. At 32×32
that leaves **4 arms** — `{default, 2·16}` per slot — because variant 2 is the
default and variants 3/4 have no backward twin. Measured: 939.8 / 960.4 /
984.1 / 1067.4 ns, the default winning and 2·16 costing ~2% per slot, 14% in
both. Once tangent backward lands the pool becomes `{0, 1, 3}` per slot = 9
arms.

Why a separate pass rather than a cross-product:

1. **Cap.** The forward pool already reaches 4 mid × 4 leaf per pair;
   cross-producing a backward axis onto that is how a cell stops being
   searchable at all.
2. **Independence.** Forward and backward slots are different function
   pointers, so the objective is separable. A joint search would only
   re-measure the forward winner once per backward form.
3. **Directional, not joint.** The zr2c child that motivated the axis runs
   exactly ONE direction per handle, so a summed metric would optimize a cost
   no caller pays. Measured at N=1024, the forward and backward winners for
   the same 32.32 plan are DIFFERENT variant codes.

## 5 — Gating

| arms | check |
|---|---|
| all forward candidates | `_il_dp_gate_err` before timing — gate-before-time |
| cascade (joint) | roundtrip `bwd(fwd(z)) ≈ N·z`, refuse above 1e-11 |
| backward pass | roundtrip verified per arm, before timing |

The backward gate is not redundant with the forward one: gate-before-time
gates the FORWARD, so without it a backward variant that is fast and WRONG
would win its race unopposed. It also subsumes the no-op case — a backward
that does nothing cannot reproduce `N·z`.

## 6 — Measured candidate counts

`il_dp_cand_census.exe`, `VFFT_IL_DP_MAX_CAND = 1024`:

| N | scrambled | natural |
|---|---|---|
| 1024 | 35 | 23 |
| 2048 | 50 | 8 |
| 4096 | 80 | 1 |
| 8192 | 117 | 0 |
| 16384 | 171 | 0 |
| 32768 | 253 | 0 |
| 65536 | 349 | 0 |

Plus, on the natural winner, the backward pass: 4 distinct arms at 32×32
today, 9 once tangent backward exists.

**Natural IL candidates collapse with N** — 23, 8, 1, 0. At 4096 the only
legal pair is 64×64, and above that no pair of powers of two both `<= 64` can
multiply to `N`, so the pair axis cannot cover the cell and the cascade owns
it. This is the three-tier rule falling out of the enumerator rather than
being asserted. A corollary worth knowing: at 4096 natural the FORWARD search
has no choice to make, so the backward axis is the only thing left to search
at that cell.

The cap is not binding anywhere. At the previous cap of 64 it would have
truncated from 4096 up — `4.4.4.4.4.4` sat at index 72.

## 7 — What is NOT an arm

Recorded so the gaps stay visible rather than being rediscovered:

- **The split engine has no kernel-form axis at all.** `sp_kv` is declared
  reserved in `wisdom2.h` and is written and read NOWHERE. The interleaved
  engine has two form verdicts (forward `il_kv` and the `dir=bwd` sibling);
  the split engine has none.
- **The r2c/c2r route is not raced here.** The interleaved real path stores
  one bit (`child_oop_il` vs `child_nat_ip`) and inherits its entire plan
  from the c2c cell at N/2. Section 4 is why: c2r is the consumer of the
  `dir=bwd` verdict.
- **The Hermitian fold has no axis.** One fixed implementation, applied
  unconditionally.
- **Arm ORDER is not rotated in the backward pass.** It walks the nibble grid
  once. The `eng=route` race alternates arm order across 9 rounds and takes
  the median, on the principle that order bias is tolerable while a verdict
  dies with the process but must not be frozen into a record. The backward
  pass does not yet do this.

## 8 — Rules for adding an axis

1. Re-run `il_dp_cand_census.exe` and record the new counts here. Never infer
   them from the loops.
2. Delegate legality to whatever builds the plan; never re-implement a
   validator inside the enumerator.
3. Never filter a legal arm on a model. If it is legal, time it.
4. If the axis is direction-dependent it is a `dir=` sibling CELL, not more
   payload bits — wisdom2 keys direction and does not key kernel forms.
5. If it multiplies an existing axis, check it against
   `VFFT_IL_DP_MAX_CAND` and prefer a separate pass.
