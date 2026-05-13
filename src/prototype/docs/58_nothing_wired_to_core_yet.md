# State of the Wire: Nothing Wired to Core Yet

As of 2026-05-13 end-of-day, **none of the prototype codelets reach
production**. The OCaml DAG compiler in `src/prototype/` has accumulated
a substantial library of validated codelets — trig transforms, strided
2D codelets — but `src/core/` and `src/vectorfft_tune/generated/` are
unchanged. Production callers of `vfft.h` will get the same code paths
they got before any of this work.

This is by design, not oversight. The prototype is research scratch;
production integration goes through the tune harness, not by direct
`#include`. But that means the prototype wins don't materialize for
end users until someone does the wire-up step.

## What's accumulated in prototype waiting to ship

### Trig transform family (validated 2026-05-12 → 2026-05-13)

Math primitives in `lib/dft_r2c.ml`, CLI flags in `bin/gen_radix.ml`,
codelets emitted at N=8 and at N=16/32/64 scaling tier:

- **DCT-II** — Makhoul reduction; TIE at N=8 vs hand-tuned `dct2_n8_avx2`,
  fills gap at N=16/32/64 (no production codelet exists at these sizes).
- **DCT-III** — inverse Makhoul; fills general-N production gap.
- **DCT-IV** — Lee 1984 algorithm; fused DAG wins 24–79% vs the
  production runtime 3-pass (pre-twiddle + c2c-N/2 IFFT + post-twiddle)
  at all 24 measured cells.
- **DST-II / DST-III** — DCT wrappers with sign-flip + reversal; fused
  DAG wins 16–55% over production runtime 3-pass at N=8, fills gap at
  N=16/32/64.
- **DHT** — N-rdft + butterfly; fused DAG wins 11/12 cells + 1 tie
  vs production runtime 3-pass.

Doc: [55_trig_transforms_vs_production.md](55_trig_transforms_vs_production.md).

### Strided-batch 2D codelets (validated 2026-05-13)

OCaml emitter retrofit (`--strided` flag in `gen_radix.ml`, ~150 LOC
added to `lib/emit_c.ml`), composes with `--bwd`:

- **R=16, 32, 64, 128, 256** × **fwd, bwd** = 10 codelets in
  [src/prototype/codelets/avx2/strided/](../codelets/avx2/strided/).
- **40/40 directional microbench cells** WIN vs (gather + standard OOP
  codelet + scatter) reference. Speedups 1.15× to 3.67×, growing with
  batch size B because larger B = more gather/scatter overhead Design C
  eliminates.
- **20/20 roundtrip identity** `bwd_strided(fwd_strided(x)) / N == x`
  PASS at FP noise (3.6e-15 to 1.2e-14 absolute).
- **Phase-instrumented breakdown bench** (`bench_2d_breakdown.c`)
  showed gather/scatter at 21–37% of total 2D time pre-strided —
  much higher than the 33-day-old strategy memo claimed (~10%).
  Design C eliminates that traffic entirely.

Doc: [56_strided_batch_2d_design_c.md](56_strided_batch_2d_design_c.md).

## What "wired to core" actually means

The path from prototype codelet to production speedup is multiple steps:

1. **Codelet ingestion.** Generated codelet goes into
   `src/vectorfft_tune/generated/r{N}_*/` with the expected file
   naming, function symbols, and dispatch-table entries the production
   build expects.
2. **Calibrator runs.** Per-host calibration measures the new codelet
   against existing options in plan-level context (not just isolated
   bench). Wisdom file `vfft_wisdom_tuned.txt` gets updated with the
   winning choices.
3. **Planner / dispatcher consults wisdom.** When a user calls
   `vfft_plan_2d` (or 1D, R2C, etc.), the planner looks at wisdom and
   picks the codelet variant that wins for that cell on that host.
4. **Public API stays stable.** `vfft.h` doesn't change; the user sees
   the speedup transparently.

For the strided-batch 2D codelets specifically, there's an additional
wrinkle: the strided codelet does a SINGLE-STAGE radix-N FFT in one
straight-line emission, but production's multi-stage row FFT plan
uses a CT decomposition with twiddles between stages. The output
ORDERING convention must match between the strided codelet and the
plan-level contract — otherwise the column phase reads garbage.
A mid-session experiment confirmed this: directly plugging R=128/256
strided into `_fft2d_tiled_range` caused 128² / 256² / 64×256 to
FAIL `test_fft2d` with err ≈ 1.3 absolute, despite the strided
codelet being bit-identical to a single-stage reference. The fix is
either (a) wire strided only at cells where the production plan also
chooses single-stage, or (b) reconcile the output ordering as part
of integration.

## What's NOT wired (explicitly)

- `src/core/fft2d.h` — no awareness of strided codelets, still does
  gather → execute_slice → scatter for every tile.
- `src/core/dct.h`, `dst.h`, `dht.h`, `dct4.h` — production runtime
  still uses the 3-pass approach where the prototype's fused DAG
  would win.
- `src/vectorfft_tune/generated/` — no entries for any of the trig
  variants or strided codelets.
- Wisdom files — calibrator hasn't seen any of these codelets, so no
  per-host selection has been made.

## Why this matters

Today's measurement results — strided beating MKL on 2D, DCT-IV
beating production 3-pass 24/24 cells, DST/DHT winning 15-55% — are
**not yet user-visible**. They are validated as a *capability* of the
codelet emitter, not as a *behavior* of `libvfft`. A user calling
`vfft_plan_2d` from `vfft.h` today gets exactly the same 2D FFT they
got yesterday.

Wiring is the unlock. The codelets exist, they're correct, they're
fast. They just don't reach callers yet.

## Wire-up sequence (when we do it)

Suggested order based on risk and value:

1. **Trig transforms** — lowest-risk integration. Each trig transform
   has a single-stage codelet entry point with a defined input/output
   contract (real input, real output, well-known indexing). No
   plan-level ordering reconciliation needed because there's no
   multi-stage plan to disagree with — the production runtime 3-pass
   IS the alternative, and we're replacing it. Start with DCT-IV
   (biggest measured win, 24/24 wins) and work outward.
2. **Strided 2D at cells where single-stage = production plan choice**
   — for cells where the production row plan picks a direct radix-N
   codelet (no CT decomposition), our strided drops in cleanly. Likely
   N2 ∈ {16, 32, 64}.
3. **Strided 2D multi-stage reconciliation** — for cells where the
   production row plan is multi-stage (N2 ∈ {128, 256, …}), figure
   out the output ordering contract. Either bake a permutation into
   the strided codelet to match the plan, or change the plan to use
   our single-stage codelet at these cells (with the strided codelet's
   output ordering becoming the new convention for those cells).
4. **OOP API** — separate workstream from any of the above. Adds
   `vfft_execute_oop` shape; useful for MKL parity but doesn't depend
   on (or block) the perf wire-up.

## Files

- This doc: `src/prototype/docs/58_nothing_wired_to_core_yet.md`
- Related: [55_trig_transforms_vs_production.md](55_trig_transforms_vs_production.md),
  [56_strided_batch_2d_design_c.md](56_strided_batch_2d_design_c.md)
- Memory: [feedback_prototype_vs_production.md](../../../../.claude/projects/c--Users-Tugbars-Desktop-highSpeedFFT/memory/feedback_prototype_vs_production.md)
- Memory: [strided_batch_2d_v1_landed.md](../../../../.claude/projects/c--Users-Tugbars-Desktop-highSpeedFFT/memory/strided_batch_2d_v1_landed.md)
- Memory: [trig_transforms_dag_validated.md](../../../../.claude/projects/c--Users-Tugbars-Desktop-highSpeedFFT/memory/trig_transforms_dag_validated.md)
