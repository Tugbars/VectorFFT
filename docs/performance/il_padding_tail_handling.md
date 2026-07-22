# IL padding-based tail handling — state, mechanism, verdict (2026-07-18)

> **The one-paragraph truth.** The interleaved 1D c2c path now carries the
> full padding machinery the tail-handling doctrine prescribes — Kp work
> buffers, zero-filled slack, full-width interior, true-K boundary — wired
> end to end and gated. It is **available, not default**: the same-process
> race showed the fused entry/exit folds (with the codelets' anyk hybrid
> tails) beating the padded-unfused arm at most cells, so today's default
> is hybrid-tailed-fused and padding engages by `VFFT_IL_PAD=1` only. The
> flip to per-cell defaults waits on one contained piece: an IL-specific
> verdict A/B at create. This mirrors, deliberately, the lifecycle the
> split path's `exec_me` went through before padding earned *its* defaults
> there.

Companion doctrine: `docs/roadmap/tail_handling/` (padding_design_decision,
arbitrary_k_pad_vs_tail, arbitrary_k_tail_strategy) and
`docs/performance/arbitrary_k_tail_handling.md`. Ledger sections:
§6a54–§6a58 in `mkl_geometry_contracts.md`.

## 1. Why IL is the padding-friendly layout

The split in-place path runs on **user-owned** buffers — padding is
unavailable there by ownership, which is why the anyk hybrid tail exists
and is doctrinally correct for it. The IL pipeline is architecturally
different: `z_in → il_wr/il_wi (engine-owned split work) → z_out`. That
ownership unlocks the padding branch the split path could never take:

- **buffers**: `il_wr/il_wi` allocated at `N·Kp`, `Kp = roundup8(K)`;
- **slack**: zeroed **once** at allocation — the interior stages are
  linear and lane-independent, so zero lanes stay zero through *both*
  directions; no per-execute re-zeroing exists or is needed;
- **interior**: `cplan_il` created at Kp (aligned wisdom chain when
  present, auto otherwise), jit tier resolved
  (`vfft_proto_plan_jit_{fwd,bwd}`) — every stage codelet runs full-width,
  the hybrid never fires;
- **boundary**: deinterleave in / interleave out at **true K** through the
  §6a57 explicit-intrinsic converts (`_vfft_z_dein/_vfft_z_inter`:
  avx512 8-complex `permutex2var`, avx2 4-complex unpack+perm, plain-C
  floor, scalar epilogue, **no masks** — one flat primitive shared with
  the fallback's convert-around).

Decision is **first-execute-lazy** (matching the `il_wr` lazy pattern);
`VFFT_IL_PAD=0/1` forces the arm for gates and same-process benches.

## 2. What the padded arm costs and buys — the structural trade

The tight (default) path is **fused**: the entry fold runs stage 0
directly on user z, the exit fold writes z from the last stage — one
whole memory pass saved each way, paid for with the codelets' hybrid
tails at `K % VW != 0` (masked full-width on avx512; SSE2-pair + scalar
straggler on avx2 — see `arbitrary_k_tail_handling.md`).

The padded arm is **unfused by necessity**: the fold codelets' z-side
addressing is bound to `z_lanes == plan->K`, so a Kp plan cannot drive
them against a K-lane z array. Padding therefore trades the fold fusion
for full-width purity: two extra boundary passes, zero tail code paths.

That is a pad-vs-tail trade in the doctrine's exact sense — and only a
per-cell measurement can arbitrate it.

## 3. The measured verdict (same-process, arm-locked, jit-fair, med9)

| cell | fused-K (tight) | padded-Kp | pad delta |
|---|---|---|---|
| (100, 7)  | 1.92 µs  | 1.87 µs  | **−2.4%** |
| (512, 7)  | 14.78 µs | 15.73 µs | +6.4% |
| (1000,12) | 35.86 µs | 66.76 µs | +86% |

Three lessons inside those numbers, all recorded in §6a55:

1. **Jit fairness** — the first cut ran the pad arm on the generic core
   executors and read +102% at (1000,12); wiring the jit tier is what
   made the race legitimate. Any future arm comparison inherits this
   rule.
2. **The +86% is chain quality, not padding physics** — (1000,12) rides
   the calibrated wisdom chain `{25,5,8}`; (1000,16) was a wisdom miss →
   cold auto chain `{25,20,2}` (radix-2 last stage). The padded arm's
   quality is coupled to the aligned cell's calibration state.
3. **`exec_me` does not transfer** — the split path's pad-vs-tail verdict
   was measured for arms without fold fusion; reusing it for IL is the
   §6a41 cross-context sin, now with numbers. The wisdom auto-engage was
   removed for exactly this reason.

## 4. Correctness record

Gate: `benches/gate_il_pad.c` (extended §6a57), ALL PASS both builds.

- Per-arm roundtrips 4.4e-16 … 8.9e-16.
- **Cross-arm BIT-IDENTICAL where the two arms' planners picked the same
  chain** ((100,5), (200,12)) — lane independence makes equal chains
  bit-equal; a stronger check than designed for.
- Sorted-magnitude spectrum equality at 2.4e-15 / 4.9e-15 where chains
  diverge.
- **Contract observation, recorded**: default-order z bin order is
  CHAIN-DEFINED — wisdom recalibration already changes it across plan
  creations today, so a padded arm choosing a different chain is
  contract-equivalent. Gates must compare order-free unless chains match.
- BIT residue sweep of the boundary converts vs the scalar reference
  across 13 lane counts (every epilogue class), 0 diffs; natural-order
  fallback cells through the same converts at e-16 (§6a57).
- §6a58: MT-vs-ST BIT = 0 fwd+bwd across six cells — the padded arm
  itself stays ST (env-only experimental); slabbing it at Kp is
  mechanical if it ever earns a default.

## 5. The flip condition — SHIPPED (§6a59)

The IL-specific verdict A/B is live: `_il_ab_race` at the first-execute
decision, both production arms on private scratch, alternating rounds,
med9, 3% hysteresis toward the fused incumbent, winner roundtrip-gated,
verdict stamped into the v7 wisdom field `il_me` (persists with the
bundle save; `VFFT_IL_PAD` still forces). The cold-chain hazard of §3.2
resolved itself by construction: (1000,12) raced and verdicted K — a
measured outcome instead of a shipped regression. Padding is now the IL
default per cell where it wins, exactly as `exec_me` does for split.
(The deluxe variant — calibrating the Kp cell as part of the race —
remains available as a refinement; the race is honest without it, since
an uncalibrated Kp arm simply loses.)

## 6. File map

| what | where |
|---|---|
| decision + padded arm + converts + MT | `src/core/vfft.c` (`_exec_c2c_interleaved`, `_vfft_z_dein/_vfft_z_inter`, `_il_pad_*`, `_il_mt_*`) |
| arm gate + perf race | `benches/gate_il_pad.c` |
| convert baseline (compiler-vs-hand, §6a56) | `benches/bench_il_convert_vec.c` |
| MT BIT gate | `benches/gate_il_mt.c` |
| ledger | `mkl_geometry_contracts.md` §6a55–§6a58 |
